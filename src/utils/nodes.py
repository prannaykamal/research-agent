import logging
import re
import statistics

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.types import interrupt

from src.utils.guardrails import (
    MAX_ANALYST_REVISIONS,
    MAX_REQUIREMENTS,
    MAX_SECTION_CHARS,
    MAX_SYNTHESIS_INPUT_CHARS,
    MAX_TOOL_ERROR_SHARE,
    assign_requirements,
    clip_text,
    uncovered_requirements,
    validate_max_analysts,
)
from src.utils.objects import Perspectives, QuestionRequirements
from src.utils.profiles import DEFAULT_PROFILE, get_models, get_profile
from src.utils.prompts import (
    analyst_instructions,
    current_date,
    dated,
    intro_conclusion_instructions,
    report_writer_instructions,
    requirements_instructions,
)
from src.utils.states import GenerateAnalystsState, ResearchGraphState

logger = logging.getLogger(__name__)

NO_RESEARCH_MESSAGE = "No analyst research completed, so no report could be written."
# A persona shorter than this gets one corrective retry.
MIN_PERSONA_WORDS = 40
# A report body citing fewer distinct URLs than this (when its sections cite) is retried.
MIN_REPORT_CITATIONS = 3
_INLINE_URL = re.compile(r"https?://[^\s)\]<>]+")
_SOURCES_HEADING = re.compile(r"^#{2,3} Sources\s*$", re.MULTILINE)
_RULE_LINES = {"---", "***", "___"}
_NO_LIMITS = re.compile(r"\bno (specific |stated |material )?(evidence )?limit", re.IGNORECASE)
MIN_REPEATED_LIMIT_CHARS = 40


def _format_sections(sections: object) -> str:
    """Normalize reducer wrappers and Gemini content parts into report prose."""

    flattened: list[str] = []

    def collect(value: object) -> None:
        if isinstance(value, str):
            if value.strip():
                flattened.append(value)
            return
        if isinstance(value, (list, tuple)):
            for item in value:
                collect(item)
            return
        if isinstance(value, dict):
            for key in ("draft", "sections", "content", "text", "output_text"):
                if key in value:
                    collect(value[key])
                    return
            raise TypeError("Report section dictionaries must contain text or content")
        if value is not None:
            raise TypeError(f"Report sections must be text, not {type(value).__name__}")

    collect(sections)
    return "\n\n".join(flattened)


def _bounded_sections(state: ResearchGraphState) -> str:
    """Join analyst sections, each clipped so ten always fit one synthesis request."""
    sections = [
        clip_text(_format_sections(section), MAX_SECTION_CHARS)
        for section in state.get("sections", [])
    ]
    return clip_text(_format_sections(sections), MAX_SYNTHESIS_INPUT_CHARS)


def _profile_name(state: GenerateAnalystsState | ResearchGraphState) -> str:
    return get_profile(state.get("model_profile") or DEFAULT_PROFILE).name


def _bullets(items: list[str]) -> str:
    return "\n".join(f"- {item}" for item in items)


def _requirements(state: GenerateAnalystsState | ResearchGraphState) -> list[str]:
    return list(state.get("requirements") or [state["topic"]])


def _question_requirements(state: GenerateAnalystsState, profile_name: str) -> list[str]:
    """Break the question into the requirements a complete answer must cover.

    A panel revision reuses the original list, and an unusable reply falls back
    to the question itself, so the panel always has something to own.
    """
    if state.get("requirements"):
        return list(state["requirements"])
    structured_llm = get_models(profile_name).panel.with_structured_output(
        QuestionRequirements, include_raw=True
    )
    result = structured_llm.invoke(
        [
            SystemMessage(
                content=dated(
                    requirements_instructions.format(
                        topic=state["topic"],
                        max_requirements=MAX_REQUIREMENTS,
                        today=current_date(),
                    )
                )
            ),
            HumanMessage(content="List the requirements."),
        ]
    )
    parsed = result.get("parsed") if isinstance(result, dict) else None
    requirements: list[str] = []
    if isinstance(parsed, QuestionRequirements):
        for requirement in parsed.requirements:
            cleaned = " ".join(requirement.split())
            if cleaned and cleaned.casefold() not in {item.casefold() for item in requirements}:
                requirements.append(cleaned)
    if not requirements:
        logger.warning("Question decomposition returned no requirements; using the question itself.")
        requirements = [state["topic"]]
    return requirements[:MAX_REQUIREMENTS]


def _shortest_persona(analysts: list) -> int:
    return min((len(analyst.description.split()) for analyst in analysts), default=0)


def create_analysts(state: GenerateAnalystsState):
    max_analysts = validate_max_analysts(state["max_analysts"])
    profile_name = _profile_name(state)
    requirements = _question_requirements(state, profile_name)
    structured_llm = get_models(profile_name).panel.with_structured_output(Perspectives)
    system_message = dated(
        analyst_instructions.format(
            topic=state["topic"],
            requirements=_bullets(requirements),
            human_analyst_feedback=state.get("human_analyst_feedback") or "",
            max_analysts=max_analysts,
        )
    )
    analysts = structured_llm.invoke(
        [
            SystemMessage(content=system_message),
            HumanMessage(content="Generate the analyst panel."),
        ]
    ).analysts
    if len(analysts) < max_analysts:
        retry = structured_llm.invoke(
            [
                SystemMessage(content=system_message),
                HumanMessage(
                    content=(
                        f"You returned {len(analysts)} analysts. Return exactly "
                        f"{max_analysts} distinct analysts."
                    )
                ),
            ]
        ).analysts
        if len(retry) > len(analysts):
            analysts = retry
    if not analysts:
        raise ValueError("The model returned no analysts; retry the run.")
    if len(analysts) < max_analysts:
        logger.warning("Requested %d analysts but the model returned %d.", max_analysts, len(analysts))
    analysts = analysts[:max_analysts]
    if _shortest_persona(analysts) < MIN_PERSONA_WORDS:
        retry = structured_llm.invoke(
            [
                SystemMessage(content=system_message),
                HumanMessage(
                    content=(
                        f"Some analyst descriptions are under {MIN_PERSONA_WORDS} words. Return "
                        f"the full panel of {len(analysts)} analysts again, each with a three- "
                        "or four-sentence description as instructed."
                    )
                ),
            ]
        ).analysts[: len(analysts)]
        if len(retry) == len(analysts) and _shortest_persona(retry) > _shortest_persona(analysts):
            analysts = retry
        else:
            logger.warning("Analyst descriptions remain short after a corrective retry.")
    owned = assign_requirements([analyst.requirements for analyst in analysts], requirements)
    analysts = [
        analyst.model_copy(update={"requirements": requirements_owned})
        for analyst, requirements_owned in zip(analysts, owned)
    ]
    return {"analysts": analysts, "requirements": requirements, "model_profile": profile_name}


def human_feedback(state: GenerateAnalystsState):
    revisions = state.get("revision_count", 0)
    feedback = interrupt(
        {
            "question": "Are these analysts okay?",
            "analysts": [
                analyst.model_dump() if hasattr(analyst, "model_dump") else analyst
                for analyst in state.get("analysts", [])
            ],
            "revisions_remaining": max(0, MAX_ANALYST_REVISIONS - revisions),
            "instructions": "Return revision feedback, or empty/perfect/continue to approve.",
        }
    )
    if not isinstance(feedback, str):
        return {"human_analyst_feedback": None}
    cleaned = feedback.strip()
    if not cleaned or cleaned.lower() in {"perfect", "continue", "approved", "yes"}:
        return {"human_analyst_feedback": None}
    if revisions >= MAX_ANALYST_REVISIONS:
        logger.warning(
            "Analyst revision limit (%d) reached; proceeding with the current panel.",
            MAX_ANALYST_REVISIONS,
        )
        return {"human_analyst_feedback": None}
    return {"human_analyst_feedback": cleaned, "revision_count": revisions + 1}


def dummy(_: GenerateAnalystsState):
    return {}


def _flagged_requirements(state: ResearchGraphState) -> list[str]:
    """Requirements no analyst reported covering, for the report writer to check."""
    covered = [
        requirement
        for entry in state.get("analyst_stats", [])
        for requirement in entry.get("requirements_covered", [])
    ]
    return uncovered_requirements(_requirements(state), covered)


def write_report(state: ResearchGraphState):
    sections = _bounded_sections(state)
    if not sections:
        return {"content": NO_RESEARCH_MESSAGE}
    flagged = _flagged_requirements(state)
    instructions = dated(
        report_writer_instructions.format(
            topic=state["topic"],
            requirements=_bullets(_requirements(state)),
            flagged="; ".join(flagged) if flagged else "none",
            context=sections,
        )
    )
    models = get_models(_profile_name(state))
    report = _format_sections(
        models.medium.invoke(
            [SystemMessage(content=instructions), HumanMessage(content="Write the report.")]
        ).content
    )
    available = inline_urls(sections)
    cited = len(inline_urls(report))
    if available and cited < min(MIN_REPORT_CITATIONS, len(available)):
        # Cheaper writers have dropped every inline citation under a long rule list;
        # retry once on the stronger panel model rather than ship an unattributed report.
        retry = _format_sections(
            models.panel.invoke(
                [
                    SystemMessage(content=instructions),
                    HumanMessage(
                        content=(
                            f"Your report has {cited} inline citations. Write the report again, "
                            "keeping the sections' inline [title](url) citations on every "
                            "paragraph, table, and Insights bullet."
                        )
                    ),
                ]
            ).content
        )
        if len(inline_urls(retry)) > cited:
            report = retry
        else:
            logger.warning("Report still has %d inline citations after a corrective retry.", cited)
    return {"content": report}


def _report_body(state: ResearchGraphState) -> str:
    """The synthesized report body, which the introduction and conclusion summarize.

    They run after the report so they inherit its reconciled figures and hedges
    instead of re-summarizing the raw analyst sections.
    """
    body = _format_sections(state.get("content", ""))
    if not body.strip() or body == NO_RESEARCH_MESSAGE:
        return ""
    return clip_text(body, MAX_SYNTHESIS_INPUT_CHARS)


def write_introduction(state: ResearchGraphState):
    body = _report_body(state)
    if not body:
        return {"introduction": f"# {state['topic']}"}
    instructions = intro_conclusion_instructions.format(topic=state["topic"], report_body=body)
    intro = get_models(_profile_name(state)).light.invoke(
        [SystemMessage(content=dated(instructions)), HumanMessage(content="Write the introduction.")]
    )
    return {"introduction": _format_sections(intro.content)}


def write_conclusion(state: ResearchGraphState):
    body = _report_body(state)
    if not body:
        return {"conclusion": ""}
    instructions = intro_conclusion_instructions.format(topic=state["topic"], report_body=body)
    conclusion = get_models(_profile_name(state)).light.invoke(
        [SystemMessage(content=dated(instructions)), HumanMessage(content="Write the conclusion.")]
    )
    return {"conclusion": _format_sections(conclusion.content)}


def _run_stats(state: ResearchGraphState) -> dict:
    analyst_stats = state.get("analyst_stats", [])
    durations = [entry["duration_seconds"] for entry in analyst_stats]
    tool_errors: dict[str, int] = {}
    for entry in analyst_stats:
        for tool, count in (entry.get("tool_errors") or {}).items():
            tool_errors[tool] = tool_errors.get(tool, 0) + count
    covered = [
        requirement
        for entry in analyst_stats
        for requirement in entry.get("requirements_covered", [])
    ]
    return {
        "model_profile": _profile_name(state),
        "analysts": len(analyst_stats),
        "completed": sum(entry["status"] == "completed" for entry in analyst_stats),
        "failed": [entry["analyst"] for entry in analyst_stats if entry["status"] != "completed"],
        "mean_analyst_seconds": round(statistics.fmean(durations), 3) if durations else 0.0,
        "max_analyst_seconds": round(max(durations), 3) if durations else 0.0,
        "llm_calls": sum(entry["llm_calls"] for entry in analyst_stats),
        "input_tokens": sum(entry["input_tokens"] for entry in analyst_stats),
        "tool_calls": sum(entry.get("tool_calls", 0) for entry in analyst_stats),
        "tool_errors": tool_errors,
        "uncovered_requirements": uncovered_requirements(state.get("requirements") or [], covered),
        "per_analyst": analyst_stats,
    }


def _coverage_notes(run_stats: dict) -> list[str]:
    """Plain statements about the run's health, appended after the report."""
    notes = []
    if run_stats["failed"]:
        notes.append(
            f"research failed for {', '.join(run_stats['failed'])}; "
            "their perspectives are not included."
        )
    rejected = sorted(
        {
            entry["stop_reason"].removeprefix("auth: ")
            for entry in run_stats["per_analyst"]
            if str(entry.get("stop_reason") or "").startswith("auth: ")
        }
    )
    if rejected:
        notes.append(
            f"{', '.join(rejected)} rejected its API key; check the key configuration "
            "and rerun."
        )
    # Requirement gaps are not noted here: the report writer, which reads the
    # sections, states them, so a note cannot contradict the report body.
    failed_calls = sum(run_stats["tool_errors"].values())
    if run_stats["tool_calls"] and failed_calls / run_stats["tool_calls"] > MAX_TOOL_ERROR_SHARE:
        tools = ", ".join(sorted(run_stats["tool_errors"]))
        notes.append(
            f"{failed_calls} of {run_stats['tool_calls']} research tool calls failed "
            f"({tools}), so the evidence base is thinner than intended."
        )
    return notes


def _strip_trailing_rules(text: str) -> str:
    """Drop horizontal rules at the end of the body; finalize adds its own."""
    lines = text.rstrip().splitlines()
    while lines and (not lines[-1].strip() or lines[-1].strip() in _RULE_LINES):
        lines.pop()
    return "\n".join(lines)


def _clean_url(url: str) -> str:
    return url.rstrip(".,;:'\"")


def inline_urls(text: str) -> list[str]:
    """Distinct URLs cited in the text before any Sources heading, in order."""
    body = _SOURCES_HEADING.split(text, maxsplit=1)[0]
    urls: list[str] = []
    for url in _INLINE_URL.findall(body):
        cleaned = _clean_url(url)
        if cleaned not in urls:
            urls.append(cleaned)
    return urls


def _normalise_line(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.casefold()).strip()


def _tidy_evidence_limits(body: str) -> str:
    """Drop limits that say there are none or repeat the body, and the heading if none remain.

    Writers filled an empty section with "no specific evidence limits" and
    restated the body's own "found no evidence on ..." sentences as limits.
    """
    lines = body.splitlines()
    start = next(
        (index for index, line in enumerate(lines) if line.strip().casefold() == "## evidence limits"),
        None,
    )
    if start is None:
        return body
    end = next(
        (index for index in range(start + 1, len(lines)) if lines[index].startswith("## ")),
        len(lines),
    )
    outside = lines[:start] + lines[end:]
    stated = _normalise_line("\n".join(outside))
    kept = []
    for line in lines[start + 1 : end]:
        text = _normalise_line(line)
        if not text or _NO_LIMITS.search(line):
            continue
        # Only whole sentences count as repeats; a short label could match by accident.
        if len(text) >= MIN_REPEATED_LIMIT_CHARS and text in stated:
            continue
        kept.append(line)
    if not kept:
        return "\n".join(lines[:start] + lines[end:])
    return "\n".join([*lines[: start + 1], *kept, *([""] if end < len(lines) else []), *lines[end:]])


def missing_sources(body: str, sources: str) -> list[str]:
    """Inline URLs from the body that the Sources list does not contain, in order."""
    listed = {_clean_url(url) for url in _INLINE_URL.findall(sources)}
    missing: list[str] = []
    for url in _INLINE_URL.findall(body):
        cleaned = _clean_url(url)
        if cleaned not in listed and cleaned not in missing:
            missing.append(cleaned)
    return missing


def finalize_report(state: ResearchGraphState):
    content = state["content"].strip()
    if content.startswith("## Insights"):
        content = content.removeprefix("## Insights").lstrip()
    # A bare wrapper heading adds nothing between the introduction and the body.
    content = re.sub(r"^## Report\s*$\n?", "", content, flags=re.MULTILINE)
    body, separator, sources = content.partition("\n## Sources\n")
    body = _strip_trailing_rules(_tidy_evidence_limits(body))
    sources = sources.strip() if separator else ""
    # Every inline citation must resolve to an entry in Sources.
    extra = missing_sources(body, sources)
    if extra:
        sources = "\n".join(part for part in (sources, *(f"- {url}" for url in extra)) if part)
    final_report = (
        f"{state['introduction']}\n\n---\n\n{body}\n\n---\n\n{state['conclusion']}"
    )
    if sources:
        final_report += f"\n\n## Sources\n{sources}"
    run_stats = _run_stats(state)
    for note in _coverage_notes(run_stats):
        final_report += f"\n\n> **Coverage note:** {note}"
    return {"final_report": final_report, "run_stats": run_stats}
