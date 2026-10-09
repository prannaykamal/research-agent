import asyncio
import contextvars
import hashlib
import json
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from typing import Any, Literal
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import httpx

from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage, SystemMessage, ToolMessage
from langgraph.config import get_stream_writer
from langgraph.errors import GraphBubbleUp
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langgraph.prebuilt import ToolNode, tools_condition

from src.utils.guardrails import (
    MAX_CLAIM_CHARS,
    MAX_CONCURRENT_ANALYSTS,
    MAX_EXCERPT_CHARS,
    MAX_FINDINGS_PER_ANALYST,
    MAX_FINDINGS_PER_PASS,
    MAX_SECTION_CHARS,
    analyst_stop_reason,
    clip_text,
    covered_requirements,
    is_recent,
    match_requirement,
)
from src.utils.models import ANALYST_SLOTS, UsageMeter, analyst_context, current_usage
from src.utils.objects import (
    Analyst,
    ResearchEvaluation,
    ResearchFinding,
    ResearchFindingBatch,
    ResearchPlan,
)
from src.utils import prompts
from src.utils.profiles import DEFAULT_PROFILE, ResearchProfile, get_models, get_profile
from src.utils.prompts import (
    dated,
    evaluator_instructions,
    finding_extraction_instructions,
    planner_instructions,
    researcher_instructions,
    writer_instructions,
)
from src.utils.nodes import _format_sections
from src.utils.states import AnalystResearchState
from src.utils.tools import (
    TIER_RANK,
    ToolAuthError,
    auth_failure,
    build_research_tools,
    source_tier,
    tool_failure,
)


logger = logging.getLogger(__name__)

MAX_EXTRACTION_TOOL_OUTPUT_CHARS = 4_000
MAX_EXTRACTION_TRANSCRIPT_CHARS = 24_000
EXTRACTION_TIMEOUT_ATTEMPTS = 2
TRACKING_QUERY_KEYS = {"fbclid", "gclid", "mc_cid", "mc_eid"}

# Dedicated threads so parallel analysts never queue behind the server's
# shared default executor.
ANALYST_EXECUTOR = ThreadPoolExecutor(MAX_CONCURRENT_ANALYSTS, thread_name_prefix="analyst")


def _analyst(value: Analyst | dict[str, Any]) -> Analyst:
    return value if isinstance(value, Analyst) else Analyst.model_validate(value)


def _profile(state: AnalystResearchState) -> ResearchProfile:
    return get_profile(state.get("model_profile") or DEFAULT_PROFILE)


def _models(state: AnalystResearchState):
    return get_models(state.get("model_profile") or DEFAULT_PROFILE)


def _usage_update() -> dict[str, int]:
    """Copy this analyst's metered LLM usage into state for guardrail routing."""
    meter = current_usage()
    if meter is None:
        return {}
    return {"llm_calls": meter.calls, "input_tokens": meter.input_tokens}


def _emit(event: dict[str, Any]) -> None:
    """Publish a small UI progress event.

    LangGraph makes the writer a no-op unless a client streams "custom" events,
    so this never adds work to a run nobody is watching.
    """
    try:
        writer = get_stream_writer()
    except RuntimeError:  # called outside a graph run, e.g. in unit tests
        return
    writer(event)


def _emit_step(state: AnalystResearchState, node: str, **details: Any) -> None:
    _emit(
        {
            "type": "analyst_step",
            "analyst_index": state.get("analyst_index"),
            "node": node,
            **details,
        }
    )


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, default=str)

def _decode_raw_structured_output(raw: Any) -> dict[str, Any] | None:
    """Decode the native Gemini JSON payload returned alongside a parse failure."""
    content = getattr(raw, "content", raw)
    if isinstance(content, dict):
        return content
    if isinstance(content, list):
        text_parts = [
            item.get("text", "") if isinstance(item, dict) else str(item)
            for item in content
        ]
        content = "".join(text_parts)
    if not isinstance(content, str):
        return None

    decoder = json.JSONDecoder()
    for index, character in enumerate(content):
        if character not in "{[":
            continue
        try:
            candidate, _ = decoder.raw_decode(content[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(candidate, dict):
            return candidate
    return None


def _parse_structured(result: Any, schema: type) -> Any | None:
    """Return the parsed object from an include_raw result, recovering from raw JSON."""
    if not isinstance(result, dict):
        return result
    if result.get("parsed") is not None:
        return result["parsed"]
    payload = _decode_raw_structured_output(result.get("raw"))
    if payload is None:
        return None
    try:
        return schema.model_validate(payload)
    except (TypeError, ValueError):
        return None


def _bounded_tool_outputs(
    messages: list[Any],
    per_tool_chars: int = MAX_EXTRACTION_TOOL_OUTPUT_CHARS,
    transcript_chars: int = MAX_EXTRACTION_TRANSCRIPT_CHARS,
) -> list[dict[str, str]]:
    """Keep structured extraction fast while retaining evidence from this pass only."""

    outputs: list[dict[str, str]] = []
    remaining = transcript_chars
    for message in messages:
        if not isinstance(message, ToolMessage) or remaining <= 0:
            continue
        content = str(message.content)
        limit = min(per_tool_chars, remaining)
        marker = "\n[tool output truncated for evidence extraction]"
        if len(content) > limit:
            bounded = content[: max(0, limit - len(marker))] + marker[:limit]
        else:
            bounded = content
        outputs.append({"tool_call_id": message.tool_call_id, "content": bounded})
        remaining -= len(bounded)
    return outputs
def _recover_valid_findings(raw: Any) -> tuple[list[ResearchFinding], int]:
    """Retain individually valid findings when one batch item violates the schema."""
    payload = _decode_raw_structured_output(raw)
    if payload is None:
        return [], 0
    candidates = payload.get("findings", [])
    if not isinstance(candidates, list):
        return [], 1

    valid: list[ResearchFinding] = []
    discarded = 0
    for candidate in candidates:
        try:
            finding = ResearchFinding.model_validate(candidate)
        except (TypeError, ValueError):
            discarded += 1
            continue
        if len(valid) == MAX_FINDINGS_PER_PASS:
            discarded += 1
            continue
        valid.append(finding)
    return valid, discarded


def _normalise_text(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip().casefold())


def canonicalize_url(url: str) -> str:
    """Normalize a source URL deterministically for evidence de-duplication."""
    parsed = urlsplit(url.strip())
    scheme = parsed.scheme.lower()
    netloc = parsed.netloc.lower()
    path = parsed.path.rstrip("/") or "/"
    query = urlencode(
        sorted(
            (key, value)
            for key, value in parse_qsl(parsed.query, keep_blank_values=True)
            if not key.lower().startswith("utm_") and key.lower() not in TRACKING_QUERY_KEYS
        )
    )
    return urlunsplit((scheme, netloc, path, query, ""))


def evidence_key(finding: ResearchFinding) -> str:
    payload = "␟".join(
        [
            canonicalize_url(finding.source_url),
            _normalise_text(finding.claim),
            _normalise_text(finding.excerpt),
        ]
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def deduplicate_findings(
    existing: list[ResearchFinding], candidates: list[ResearchFinding]
) -> list[ResearchFinding]:
    """Retain unique, source-addressable, size-bounded findings without an LLM decision.

    When more candidates qualify than the analyst has room for, higher-tier
    sources win; ties keep the extraction order.
    """
    seen = {evidence_key(finding) for finding in existing if finding.source_url.strip()}
    capacity = max(0, MAX_FINDINGS_PER_ANALYST - len(existing))
    unique: list[ResearchFinding] = []
    for finding in candidates:
        if not finding.source_url.strip() or not finding.claim.strip() or not finding.excerpt.strip():
            continue
        canonical = canonicalize_url(finding.source_url)
        if not canonical.startswith(("http://", "https://")):
            continue
        normalized = finding.model_copy(
            update={
                "source_url": canonical,
                "claim": clip_text(finding.claim.strip(), MAX_CLAIM_CHARS),
                "excerpt": clip_text(finding.excerpt.strip(), MAX_EXCERPT_CHARS),
            }
        )
        key = evidence_key(normalized)
        if key not in seen:
            seen.add(key)
            unique.append(normalized)
    if len(unique) <= capacity:
        return unique
    ranked = sorted(
        range(len(unique)),
        key=lambda index: (TIER_RANK[source_tier(unique[index].source_url)], index),
    )
    return [unique[index] for index in sorted(ranked[:capacity])]


def _current_research_brief(state: AnalystResearchState, questions: list[str]) -> str:
    analyst = _analyst(state["analyst"])
    finding_urls = sorted(
        {
            canonicalize_url(finding.source_url)
            for finding in state["research_findings"]
            if finding.source_url.strip()
        }
    )
    return _json(
        {
            "topic": state["topic"],
            "analyst": analyst.model_dump(),
            "current_sub_questions": questions,
            "feedback": state["feedback"],
            "existing_source_urls": finding_urls,
        }
    )


def _new_questions(candidates: list[str], history: set[str], accepted: list[str]) -> list[str]:
    """Append distinct, previously unseen questions to ``accepted`` (at most three)."""
    for question in candidates:
        if len(accepted) == 3:
            break
        cleaned = re.sub(r"\s+", " ", question).strip()
        normalized = _normalise_text(cleaned)
        if cleaned and normalized not in history and normalized not in {
            _normalise_text(item) for item in accepted
        }:
            accepted.append(cleaned)
    return accepted


def _fallback_questions(state: AnalystResearchState) -> list[str]:
    """Deterministic questions used when the Planner returns no usable plan."""
    analyst = _analyst(state["analyst"])
    topic = state["topic"]
    return [
        f"What is the strongest current evidence about {topic} relevant to a {analyst.role}?",
        f"What risks, limitations, or open questions about {topic} matter most to a {analyst.role}?",
        f"What recent developments in {topic} affect the concerns of {analyst.affiliation}?",
        f"What quantitative data is available on {topic} from the perspective of a {analyst.role}?",
    ]


def planner_node(state: AnalystResearchState) -> dict[str, Any]:
    """Create a fresh, non-overlapping research pass and clear tool transcripts."""
    analyst = _analyst(state["analyst"])
    if state["loop_count"] == 0:
        profile = _profile(state)
        _emit(
            {
                "type": "analyst_started",
                "analyst_index": state.get("analyst_index"),
                "analyst": analyst.model_dump(),
                "profile": profile.name,
                "limits": {
                    "max_passes": profile.max_research_loops,
                    "max_tool_calls_per_pass": profile.max_tool_calls_per_pass,
                    "max_researcher_turns": profile.max_researcher_turns,
                    "deadline_seconds": profile.analyst_deadline_seconds,
                    "allowed_tools": list(profile.allowed_tools),
                },
            }
        )
    _emit_step(state, "planner_node", research_pass=state["loop_count"] + 1)
    planner_input = {
        "topic": state["topic"],
        "analyst": analyst.model_dump(),
        "feedback": state["feedback"],
        "question_history": state["question_history"],
        "research_findings": [finding.model_dump() for finding in state["research_findings"]],
    }
    structured_llm = _models(state).heavy.with_structured_output(ResearchPlan, include_raw=True)
    plan = _parse_structured(
        structured_llm.invoke(
            [
                SystemMessage(content=dated(planner_instructions)),
                HumanMessage(content=_json(planner_input)),
            ]
        ),
        ResearchPlan,
    )
    proposed = plan.sub_questions if plan is not None else []
    history = {_normalise_text(question) for question in state["question_history"]}
    questions = _new_questions(proposed, history, [])

    if len(questions) < 2:
        correction = _parse_structured(
            structured_llm.invoke(
                [
                    SystemMessage(content=dated(planner_instructions)),
                    HumanMessage(
                        content=(
                            "Return two or three distinct additional questions. Do not repeat "
                            f"these rejected or duplicate questions: {_json(proposed)}.\n"
                            f"Context: {_json(planner_input)}"
                        )
                    ),
                ]
            ),
            ResearchPlan,
        )
        if correction is not None:
            questions = _new_questions(correction.sub_questions, history, questions)

    if len(questions) < 2:
        logger.warning("Planner returned no usable plan; using fallback questions.")
        questions = _new_questions(_fallback_questions(state), history, questions)

    return {
        "sub_questions": questions,
        "question_history": questions,
        "tool_call_count": 0,
        "researcher_turns": 0,
        "budget_exhausted": False,
        "messages": [
            RemoveMessage(id=REMOVE_ALL_MESSAGES),
            HumanMessage(content=_current_research_brief(state, questions)),
        ],
        **_usage_update(),
    }


@lru_cache(maxsize=None)
def _profile_tools(profile_name: str) -> tuple[Any, ...]:
    return tuple(build_research_tools(get_profile(profile_name).allowed_tools))


def _researcher_view(messages: list[Any], view_chars: int) -> list[Any]:
    """The Researcher only judges sufficiency, so it sees clipped tool outputs."""
    view = []
    for message in messages:
        if isinstance(message, ToolMessage) and len(str(message.content)) > view_chars:
            message = message.model_copy(
                update={"content": clip_text(str(message.content), view_chars)}
            )
        view.append(message)
    return view


def researcher_node(state: AnalystResearchState) -> dict[str, Any]:
    """Use tools only to answer the Planner's current research questions."""
    profile = _profile(state)
    _emit_step(
        state,
        "researcher_node",
        research_pass=state["loop_count"] + 1,
        turn=state.get("researcher_turns", 0) + 1,
    )
    bound_llm = _models(state).medium.bind_tools(list(_profile_tools(profile.name)))
    nudge = _search_floor_nudge(state, profile)
    transcript = [*state["messages"], *([nudge] if nudge else [])]
    response = bound_llm.invoke(
        [
            SystemMessage(content=dated(researcher_instructions)),
            *_researcher_view(transcript, profile.researcher_tool_view_chars),
        ]
    )
    return {
        "messages": [*([nudge] if nudge else []), response],
        "researcher_turns": state.get("researcher_turns", 0) + 1,
        **_usage_update(),
    }


def _search_floor_nudge(state: AnalystResearchState, profile: ResearchProfile) -> HumanMessage | None:
    """Ask a Researcher that stopped below the per-pass search floor to keep going.

    Cheaper research models tend to stop after one or two searches, which left
    whole requirements answered by a single finding.
    """
    last = state["messages"][-1] if state["messages"] else None
    if not isinstance(last, AIMessage) or last.tool_calls:
        return None
    focus = _requirements_needing_evidence(state) or list(state["sub_questions"])
    return HumanMessage(
        content=(
            f"You have made {state['tool_call_count']} of at least "
            f"{profile.min_tool_calls_per_pass} searches this pass. Search again with different "
            f"queries, prioritising: {'; '.join(focus)}. You may call several tools in one turn."
        )
    )


def _turn_limit(state: AnalystResearchState, profile: ResearchProfile) -> int:
    """Researcher turns allowed this pass.

    Below the search floor an analyst gets enough turns to reach it one search
    at a time; otherwise the profile's turn cap ended passes at three searches.
    """
    if state["tool_call_count"] < profile.min_tool_calls_per_pass:
        return max_researcher_turns(profile)
    return profile.max_researcher_turns


def max_researcher_turns(profile: ResearchProfile) -> int:
    """The most Researcher turns any pass can take, including the search-floor extension."""
    return max(profile.max_researcher_turns, profile.min_tool_calls_per_pass)


def _last_ai_message(state: AnalystResearchState) -> AIMessage | None:
    for message in reversed(state["messages"]):
        if isinstance(message, AIMessage):
            return message
    return None


def route_researcher(
    state: AnalystResearchState,
) -> Literal["limit_tool_calls", "researcher_node", "extract_findings"]:
    profile = _profile(state)
    message = _last_ai_message(state)
    requested = len(message.tool_calls) if message else 0
    remaining = profile.max_tool_calls_per_pass - state["tool_call_count"]
    if remaining <= 0 or analyst_stop_reason(state, profile):
        return "extract_findings"
    if requested == 0:
        # Below the search floor with turns left: send the Researcher back.
        if (
            state["tool_call_count"] < profile.min_tool_calls_per_pass
            and state.get("researcher_turns", 0) < _turn_limit(state, profile)
        ):
            return "researcher_node"
        return "extract_findings"
    return "limit_tool_calls"


def limit_tool_calls(state: AnalystResearchState) -> dict[str, Any]:
    """Drop disallowed tools and trim to the exact remaining hard tool-call allowance."""
    profile = _profile(state)
    message = _last_ai_message(state)
    if message is None:
        return {"budget_exhausted": True}
    remaining = max(0, profile.max_tool_calls_per_pass - state["tool_call_count"])
    permitted = [call for call in message.tool_calls if call["name"] in profile.allowed_tools]
    allowed_calls = permitted[:remaining]
    bounded_message = message.model_copy(update={"tool_calls": allowed_calls})
    new_count = state["tool_call_count"] + len(allowed_calls)
    return {
        "messages": [bounded_message],
        "tool_call_count": new_count,
        "budget_exhausted": new_count >= profile.max_tool_calls_per_pass,
    }


def route_limited_tools(state: AnalystResearchState) -> Literal["tools", "extract_findings"]:
    decision = tools_condition(state)
    return "tools" if decision == "tools" else "extract_findings"


def _pass_tool_results(messages: list[Any]) -> list[dict[str, Any]]:
    """One entry per tool result in this pass, recording whether it failed.

    Every tool reports failure as text rather than raising, so a run that is
    silently degraded (for example by a revoked key) would otherwise look healthy.
    """
    return [
        {
            "tool": message.name or "unknown",
            "failed": tool_failure(message.content, message.status) is not None,
        }
        for message in messages
        if isinstance(message, ToolMessage)
    ]


def _raise_on_auth_failure(messages: list[Any]) -> None:
    """Stop the analyst at once when a keyed service rejects its credentials."""
    for message in messages:
        if isinstance(message, ToolMessage):
            failure = auth_failure(message.name or "", message.content, message.status)
            if failure:
                raise ToolAuthError(message.name or "unknown", failure)


def route_after_tools(state: AnalystResearchState) -> Literal["researcher_node", "extract_findings"]:
    _raise_on_auth_failure(state["messages"])
    profile = _profile(state)
    if (
        state["budget_exhausted"]
        or state["tool_call_count"] >= profile.max_tool_calls_per_pass
        or state.get("researcher_turns", 0) >= _turn_limit(state, profile)
        or analyst_stop_reason(state, profile)
    ):
        return "extract_findings"
    return "researcher_node"


def extract_findings(state: AnalystResearchState) -> dict[str, Any]:
    """Convert one completed pass's temporary tool output into durable evidence."""
    profile = _profile(state)
    # Recorded before the Planner clears this pass's transcript.
    tool_results = _pass_tool_results(state["messages"])
    tool_outputs = _bounded_tool_outputs(
        state["messages"],
        profile.extraction_tool_output_chars,
        profile.extraction_transcript_chars,
    )
    if not tool_outputs:
        return {"loop_count": state["loop_count"] + 1, "tool_results": tool_results}
    _emit_step(
        state,
        "extract_findings",
        research_pass=state["loop_count"] + 1,
        tool_outputs=len(tool_outputs),
    )
    extraction_input = {
        "topic": state["topic"],
        "analyst": _analyst(state["analyst"]).model_dump(),
        "sub_questions": state["sub_questions"],
        "tool_outputs": tool_outputs,
    }
    messages = [
        SystemMessage(content=dated(finding_extraction_instructions)),
        HumanMessage(content=_json(extraction_input)),
    ]
    structured_llm = _models(state).medium.with_structured_output(
        ResearchFindingBatch,
        include_raw=True,
    )

    extraction: dict[str, Any] | None = None
    for attempt in range(1, EXTRACTION_TIMEOUT_ATTEMPTS + 1):
        try:
            extraction = structured_llm.invoke(messages)
            break
        except httpx.TimeoutException as exc:
            if attempt < EXTRACTION_TIMEOUT_ATTEMPTS:
                logger.warning(
                    "Research finding extraction timed out on attempt %d/%d; retrying: %r",
                    attempt,
                    EXTRACTION_TIMEOUT_ATTEMPTS,
                    exc,
                )
                time.sleep(1)
                continue
            logger.warning(
                "Research finding extraction timed out after %d attempts; "
                "continuing with accumulated evidence: %r",
                attempt,
                exc,
            )
            return {
                "loop_count": state["loop_count"] + 1,
                "tool_results": tool_results,
                **_usage_update(),
            }

    if extraction is None:
        raise RuntimeError("Finding extraction completed without a result or timeout.")

    batch = extraction.get("parsed")
    if batch is None:
        recovered, discarded = _recover_valid_findings(extraction.get("raw"))
        logger.warning(
            "Research finding extraction returned malformed structured output; "
            "recovered=%d discarded=%d parsing_error=%r raw_output=%r",
            len(recovered),
            discarded,
            extraction.get("parsing_error"),
            extraction.get("raw"),
        )
        additions = deduplicate_findings(state["research_findings"], recovered)
    else:
        additions = deduplicate_findings(state["research_findings"], batch.findings)
    return {
        "research_findings": additions,
        "loop_count": state["loop_count"] + 1,
        "tool_results": tool_results,
        **_usage_update(),
    }
def evaluate_research(state: AnalystResearchState) -> dict[str, Any]:
    """Assess evidence coverage without tools or external retrieval."""
    profile = _profile(state)
    if state["loop_count"] >= profile.max_research_loops or analyst_stop_reason(state, profile):
        # The Writer runs next regardless of the verdict, so skip the heavy call.
        return {}
    _emit_step(state, "evaluate_research", research_pass=state["loop_count"])
    analyst = _analyst(state["analyst"])
    structured_llm = _models(state).heavy.with_structured_output(
        ResearchEvaluation, include_raw=True
    )
    evaluation = _parse_structured(
        structured_llm.invoke(
            [
                SystemMessage(content=dated(evaluator_instructions)),
                HumanMessage(
                    content=_json(
                        {
                            "topic": state["topic"],
                            "analyst": analyst.model_dump(),
                            "current_sub_questions": state["sub_questions"],
                            "question_history": state["question_history"],
                            "research_findings": _annotated_findings(state["research_findings"]),
                        }
                    )
                ),
            ]
        ),
        ResearchEvaluation,
    )
    if evaluation is None:
        logger.warning("Evaluator returned no usable verdict; continuing research.")
        evaluation = ResearchEvaluation(
            is_complete=False,
            feedback="Broaden coverage of the analyst's core concerns with new sources.",
        )
    needing = _requirements_needing_evidence(state, evaluation)
    if evaluation.is_complete and needing:
        # Evidence that satisfies the persona is not enough: every owned
        # requirement must be supported, not merely touched, while budget remains.
        evaluation = evaluation.model_copy(
            update={
                "is_complete": False,
                "coverage_gaps": [*evaluation.coverage_gaps, *needing],
                "feedback": (
                    f"Still lacking enough evidence: {'; '.join(needing)}. {evaluation.feedback}"
                ).strip(),
            }
        )
    return {"evaluation": evaluation, "feedback": evaluation.feedback, **_usage_update()}


def _annotated_findings(findings: list[ResearchFinding]) -> list[dict[str, Any]]:
    """Findings with their source tier and whether the source is recent.

    ``recent`` is computed here rather than by the model, which otherwise
    presents years-old status evidence as current.
    """
    today = prompts.current_date()
    return [
        {
            **finding.model_dump(),
            "source_tier": source_tier(finding.source_url),
            "recent": is_recent(finding.source_date, today),
        }
        for finding in findings
    ]


def _unsupported_requirements(state: AnalystResearchState) -> list[str]:
    """The analyst's owned requirements that no finding supports yet."""
    owned = _analyst(state["analyst"]).requirements
    covered = covered_requirements(
        [finding.requirement for finding in state["research_findings"]], owned
    )
    return [requirement for requirement in owned if requirement not in covered]


def requirement_verdicts(evaluation: Any, owned: list[str]) -> dict[str, str]:
    """The evaluator's per-requirement statuses, keyed by canonical requirement text."""
    verdicts: dict[str, str] = {}
    for item in getattr(evaluation, "requirement_status", None) or []:
        match = match_requirement(item.requirement, owned)
        if match is not None:
            verdicts[match] = item.status
    return verdicts


def _requirements_needing_evidence(
    state: AnalystResearchState, evaluation: Any = None
) -> list[str]:
    """Owned requirements with no finding, or that the evaluator marked thin or missing.

    One finding is not enough: a "changed since X" requirement answered only
    with today's figures is thin, and research continues while budget allows.
    """
    owned = _analyst(state["analyst"]).requirements
    verdicts = requirement_verdicts(
        evaluation if evaluation is not None else state.get("evaluation"), owned
    )
    unfound = set(_unsupported_requirements(state))
    return [
        requirement
        for requirement in owned
        if requirement in unfound or verdicts.get(requirement) in ("thin", "missing")
    ]


def route_evaluation(state: AnalystResearchState) -> Literal["planner_node", "writer_node"]:
    profile = _profile(state)
    evaluation = state.get("evaluation")
    if evaluation is not None and evaluation.is_complete:
        return "writer_node"
    if state["loop_count"] >= profile.max_research_loops or analyst_stop_reason(state, profile):
        return "writer_node"
    return "planner_node"


def _final_stop_reason(state: AnalystResearchState, profile: ResearchProfile) -> str:
    reason = analyst_stop_reason(state, profile)
    if reason:
        return reason
    evaluation = state.get("evaluation")
    if evaluation is not None and evaluation.is_complete:
        return "evidence_sufficient"
    return "max_passes"


def writer_node(state: AnalystResearchState) -> dict[str, Any]:
    """Write from durable findings only; this model is deliberately not tool-bound."""
    profile = _profile(state)
    analyst = _analyst(state["analyst"])
    stop_reason = _final_stop_reason(state, profile)
    _emit_step(state, "writer_node", stop_reason=stop_reason)
    response = _models(state).writer.invoke(
        [
            SystemMessage(
                content=dated(writer_instructions.format(word_target=profile.writer_word_target))
            ),
            HumanMessage(
                content=_json(
                    {
                        "topic": state["topic"],
                        "analyst": analyst.model_dump(),
                        "research_findings": _annotated_findings(state["research_findings"]),
                    }
                )
            ),
        ]
    )
    return {
        "draft": clip_text(_format_sections(response.content), MAX_SECTION_CHARS),
        "stop_reason": stop_reason,
        **_usage_update(),
    }


TOOL_NODE = ToolNode(build_research_tools(), handle_tool_errors=True)


def build_analyst_research_graph():
    builder = StateGraph(AnalystResearchState)
    builder.add_node("planner_node", planner_node)
    builder.add_node("researcher_node", researcher_node)
    builder.add_node("limit_tool_calls", limit_tool_calls)
    builder.add_node("tool_node", TOOL_NODE)
    builder.add_node("extract_findings", extract_findings)
    builder.add_node("evaluate_research", evaluate_research)
    builder.add_node("writer_node", writer_node)

    builder.add_edge(START, "planner_node")
    builder.add_edge("planner_node", "researcher_node")
    builder.add_conditional_edges(
        "researcher_node",
        route_researcher,
        {
            "limit_tool_calls": "limit_tool_calls",
            "researcher_node": "researcher_node",
            "extract_findings": "extract_findings",
        },
    )
    builder.add_conditional_edges(
        "limit_tool_calls",
        route_limited_tools,
        {"tools": "tool_node", "extract_findings": "extract_findings"},
    )
    builder.add_conditional_edges(
        "tool_node",
        route_after_tools,
        {"researcher_node": "researcher_node", "extract_findings": "extract_findings"},
    )
    builder.add_edge("extract_findings", "evaluate_research")
    builder.add_conditional_edges(
        "evaluate_research",
        route_evaluation,
        {"planner_node": "planner_node", "writer_node": "writer_node"},
    )
    builder.add_edge("writer_node", END)
    return builder.compile()


analyst_research_graph = build_analyst_research_graph()


def analyst_recursion_limit(profile: ResearchProfile) -> int:
    """Supersteps a worst-case analyst can take under its profile's guardrails.

    Each pass is the planner, up to ``max_researcher_turns(profile)`` rounds of
    researcher -> limit_tool_calls -> tool_node (more than the profile's cap when
    the search floor extends a pass), extraction and evaluation; the writer runs
    once at the end. A small margin covers routing at the limits.
    The API server's default of 25 is too low for Deep Research.
    """
    per_pass = 3 + 3 * max_researcher_turns(profile)
    return profile.max_research_loops * per_pass + 1 + 5


def _covered_for_stats(
    analyst: Analyst, findings: list[ResearchFinding], evaluation: Any
) -> list[str]:
    """Owned requirements with evidence, by finding label or by the evaluator's verdict.

    Finding labels alone miss evidence the extractor filed under a neighbouring
    requirement, which produced a coverage note contradicting its own report.
    """
    owned = analyst.requirements
    labelled = set(covered_requirements([finding.requirement for finding in findings], owned))
    verdicts = requirement_verdicts(evaluation, owned)
    return [
        requirement
        for requirement in owned
        if requirement in labelled or verdicts.get(requirement) in ("supported", "thin")
    ]


def conduct_research(state: dict[str, Any]) -> dict[str, Any]:
    """Run one isolated analyst subgraph in its own quota slot.

    Only the completed draft and run statistics reach the parent graph. A
    failure is contained to this analyst and reported in its statistics.
    """
    analyst = _analyst(state["analyst"])
    profile_name = state.get("model_profile") or DEFAULT_PROFILE
    queued_at = time.time()
    meter = UsageMeter()
    with ANALYST_SLOTS.slot() as slot, analyst_context(slot, meter):
        started_at = time.time()
        sections: list[str] = []
        try:
            result = analyst_research_graph.invoke(
                {**state, "model_profile": profile_name, "started_at": started_at},
                {"recursion_limit": analyst_recursion_limit(get_profile(profile_name))},
            )
            sections = [result["draft"]]
            status, stop_reason = "completed", result.get("stop_reason")
            findings = result.get("research_findings", [])
            tool_results = result.get("tool_results", [])
            evaluation = result.get("evaluation")
        except GraphBubbleUp:
            raise
        except ToolAuthError as exc:
            # Fail loudly: a run on a rejected key must not look like a thin success.
            logger.error("Analyst %r stopped: %s", analyst.name, exc)
            status, stop_reason, findings, tool_results = "failed", f"auth: {exc.tool}", [], []
            evaluation = None
        except Exception as exc:
            logger.exception("Analyst %r failed; continuing without its section.", analyst.name)
            status, stop_reason = "failed", f"error: {type(exc).__name__}: {exc}"
            findings, tool_results, evaluation = [], [], None
        finished_at = time.time()
    tool_errors: dict[str, int] = {}
    for entry in tool_results:
        if entry["failed"]:
            tool_errors[entry["tool"]] = tool_errors.get(entry["tool"], 0) + 1
    stats = {
        "analyst_index": state.get("analyst_index"),
        "analyst": analyst.name,
        "role": analyst.role,
        "profile": profile_name,
        "status": status,
        "stop_reason": stop_reason,
        "findings": len(findings),
        "requirements": analyst.requirements,
        "requirements_covered": _covered_for_stats(analyst, findings, evaluation),
        "tool_calls": len(tool_results),
        "tool_errors": tool_errors,
        "llm_calls": meter.calls,
        "input_tokens": meter.input_tokens,
        "queued_seconds": round(started_at - queued_at, 3),
        "duration_seconds": round(finished_at - started_at, 3),
        "slot": slot.index,
    }
    return {"sections": sections, "analyst_stats": [stats]}


async def aconduct_research(state: dict[str, Any]) -> dict[str, Any]:
    """Async entry point: run the sync subgraph on the dedicated analyst executor."""
    context = contextvars.copy_context()
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(ANALYST_EXECUTOR, context.run, conduct_research, state)
