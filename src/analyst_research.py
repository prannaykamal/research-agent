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
)
from src.utils.models import ANALYST_SLOTS, UsageMeter, analyst_context, current_usage
from src.utils.objects import (
    Analyst,
    ResearchEvaluation,
    ResearchFinding,
    ResearchFindingBatch,
    ResearchPlan,
)
from src.utils.profiles import DEFAULT_PROFILE, ResearchProfile, get_models, get_profile
from src.utils.prompts import (
    evaluator_instructions,
    finding_extraction_instructions,
    planner_instructions,
    researcher_instructions,
    writer_instructions,
)
from src.utils.nodes import _format_sections
from src.utils.states import AnalystResearchState
from src.utils.tools import build_research_tools


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
    """Retain unique, source-addressable, size-bounded findings without an LLM decision."""
    seen = {evidence_key(finding) for finding in existing if finding.source_url.strip()}
    capacity = max(0, MAX_FINDINGS_PER_ANALYST - len(existing))
    unique: list[ResearchFinding] = []
    for finding in candidates:
        if len(unique) >= capacity:
            break
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
    return unique


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
                SystemMessage(content=planner_instructions),
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
                    SystemMessage(content=planner_instructions),
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
    bound_llm = _models(state).medium.bind_tools(list(_profile_tools(profile.name)))
    response = bound_llm.invoke(
        [
            SystemMessage(content=researcher_instructions),
            *_researcher_view(state["messages"], profile.researcher_tool_view_chars),
        ]
    )
    return {
        "messages": [response],
        "researcher_turns": state.get("researcher_turns", 0) + 1,
        **_usage_update(),
    }


def _last_ai_message(state: AnalystResearchState) -> AIMessage | None:
    for message in reversed(state["messages"]):
        if isinstance(message, AIMessage):
            return message
    return None


def route_researcher(state: AnalystResearchState) -> Literal["limit_tool_calls", "extract_findings"]:
    profile = _profile(state)
    message = _last_ai_message(state)
    requested = len(message.tool_calls) if message else 0
    remaining = profile.max_tool_calls_per_pass - state["tool_call_count"]
    if requested == 0 or remaining <= 0 or analyst_stop_reason(state, profile):
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


def route_after_tools(state: AnalystResearchState) -> Literal["researcher_node", "extract_findings"]:
    profile = _profile(state)
    if (
        state["budget_exhausted"]
        or state["tool_call_count"] >= profile.max_tool_calls_per_pass
        or state.get("researcher_turns", 0) >= profile.max_researcher_turns
        or analyst_stop_reason(state, profile)
    ):
        return "extract_findings"
    return "researcher_node"


def extract_findings(state: AnalystResearchState) -> dict[str, Any]:
    """Convert one completed pass's temporary tool output into durable evidence."""
    profile = _profile(state)
    tool_outputs = _bounded_tool_outputs(
        state["messages"],
        profile.extraction_tool_output_chars,
        profile.extraction_transcript_chars,
    )
    if not tool_outputs:
        return {"loop_count": state["loop_count"] + 1}
    extraction_input = {
        "topic": state["topic"],
        "analyst": _analyst(state["analyst"]).model_dump(),
        "sub_questions": state["sub_questions"],
        "tool_outputs": tool_outputs,
    }
    messages = [
        SystemMessage(content=finding_extraction_instructions),
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
            return {"loop_count": state["loop_count"] + 1, **_usage_update()}

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
        **_usage_update(),
    }
def evaluate_research(state: AnalystResearchState) -> dict[str, Any]:
    """Assess evidence coverage without tools or external retrieval."""
    profile = _profile(state)
    if state["loop_count"] >= profile.max_research_loops or analyst_stop_reason(state, profile):
        # The Writer runs next regardless of the verdict, so skip the heavy call.
        return {}
    analyst = _analyst(state["analyst"])
    structured_llm = _models(state).heavy.with_structured_output(
        ResearchEvaluation, include_raw=True
    )
    evaluation = _parse_structured(
        structured_llm.invoke(
            [
                SystemMessage(content=evaluator_instructions),
                HumanMessage(
                    content=_json(
                        {
                            "topic": state["topic"],
                            "analyst": analyst.model_dump(),
                            "current_sub_questions": state["sub_questions"],
                            "question_history": state["question_history"],
                            "research_findings": [
                                finding.model_dump()
                                for finding in state["research_findings"]
                            ],
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
    return {"evaluation": evaluation, "feedback": evaluation.feedback, **_usage_update()}


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
    response = _models(state).writer.invoke(
        [
            SystemMessage(
                content=writer_instructions.format(word_target=profile.writer_word_target)
            ),
            HumanMessage(
                content=_json(
                    {
                        "topic": state["topic"],
                        "analyst": analyst.model_dump(),
                        "research_findings": [
                            finding.model_dump()
                            for finding in state["research_findings"]
                        ],
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
        {"limit_tool_calls": "limit_tool_calls", "extract_findings": "extract_findings"},
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
                {**state, "model_profile": profile_name, "started_at": started_at}
            )
            sections = [result["draft"]]
            status, stop_reason, findings = "completed", result.get("stop_reason"), len(
                result.get("research_findings", [])
            )
        except GraphBubbleUp:
            raise
        except Exception as exc:
            logger.exception("Analyst %r failed; continuing without its section.", analyst.name)
            status, stop_reason, findings = "failed", f"error: {type(exc).__name__}: {exc}", 0
        finished_at = time.time()
    stats = {
        "analyst": analyst.name,
        "role": analyst.role,
        "profile": profile_name,
        "status": status,
        "stop_reason": stop_reason,
        "findings": findings,
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
