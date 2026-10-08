import logging
import statistics

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.types import interrupt

from src.utils.guardrails import (
    MAX_ANALYST_REVISIONS,
    MAX_SECTION_CHARS,
    MAX_SYNTHESIS_INPUT_CHARS,
    clip_text,
    validate_max_analysts,
)
from src.utils.objects import Perspectives
from src.utils.profiles import DEFAULT_PROFILE, get_models, get_profile
from src.utils.prompts import (
    analyst_instructions,
    intro_conclusion_instructions,
    report_writer_instructions,
)
from src.utils.states import GenerateAnalystsState, ResearchGraphState

logger = logging.getLogger(__name__)

NO_RESEARCH_MESSAGE = "No analyst research completed, so no report could be written."


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


def create_analysts(state: GenerateAnalystsState):
    max_analysts = validate_max_analysts(state["max_analysts"])
    profile_name = _profile_name(state)
    structured_llm = get_models(profile_name).medium.with_structured_output(Perspectives)
    system_message = analyst_instructions.format(
        topic=state["topic"],
        human_analyst_feedback=state.get("human_analyst_feedback") or "",
        max_analysts=max_analysts,
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
    return {"analysts": analysts[:max_analysts], "model_profile": profile_name}


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


def write_report(state: ResearchGraphState):
    sections = _bounded_sections(state)
    if not sections:
        return {"content": NO_RESEARCH_MESSAGE}
    instructions = report_writer_instructions.format(topic=state["topic"], context=sections)
    report = get_models(_profile_name(state)).medium.invoke(
        [SystemMessage(content=instructions), HumanMessage(content="Write the report.")]
    )
    return {"content": _format_sections(report.content)}


def write_introduction(state: ResearchGraphState):
    sections = _bounded_sections(state)
    if not sections:
        return {"introduction": f"# {state['topic']}"}
    instructions = intro_conclusion_instructions.format(
        topic=state["topic"], formatted_str_sections=sections
    )
    intro = get_models(_profile_name(state)).light.invoke(
        [SystemMessage(content=instructions), HumanMessage(content="Write the introduction.")]
    )
    return {"introduction": _format_sections(intro.content)}


def write_conclusion(state: ResearchGraphState):
    sections = _bounded_sections(state)
    if not sections:
        return {"conclusion": ""}
    instructions = intro_conclusion_instructions.format(
        topic=state["topic"], formatted_str_sections=sections
    )
    conclusion = get_models(_profile_name(state)).light.invoke(
        [SystemMessage(content=instructions), HumanMessage(content="Write the conclusion.")]
    )
    return {"conclusion": _format_sections(conclusion.content)}


def _run_stats(state: ResearchGraphState) -> dict:
    analyst_stats = state.get("analyst_stats", [])
    durations = [entry["duration_seconds"] for entry in analyst_stats]
    return {
        "model_profile": _profile_name(state),
        "analysts": len(analyst_stats),
        "completed": sum(entry["status"] == "completed" for entry in analyst_stats),
        "failed": [entry["analyst"] for entry in analyst_stats if entry["status"] != "completed"],
        "mean_analyst_seconds": round(statistics.fmean(durations), 3) if durations else 0.0,
        "max_analyst_seconds": round(max(durations), 3) if durations else 0.0,
        "llm_calls": sum(entry["llm_calls"] for entry in analyst_stats),
        "input_tokens": sum(entry["input_tokens"] for entry in analyst_stats),
        "per_analyst": analyst_stats,
    }


def finalize_report(state: ResearchGraphState):
    content = state["content"].strip()
    if content.startswith("## Insights"):
        content = content.removeprefix("## Insights").lstrip()
    body, separator, sources = content.partition("\n## Sources\n")
    final_report = (
        f"{state['introduction']}\n\n---\n\n{body}\n\n---\n\n{state['conclusion']}"
    )
    if separator and sources.strip():
        final_report += f"\n\n## Sources\n{sources.strip()}"
    run_stats = _run_stats(state)
    if run_stats["failed"]:
        final_report += (
            "\n\n> **Coverage note:** research failed for "
            f"{', '.join(run_stats['failed'])}; their perspectives are not included."
        )
    return {"final_report": final_report, "run_stats": run_stats}
