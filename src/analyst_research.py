import hashlib
import json
import logging
import re
import time
from typing import Any, Literal
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import httpx

from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage, SystemMessage, ToolMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langgraph.prebuilt import ToolNode, tools_condition

from src.utils.models import heavy_llm, medium_llm
from src.utils.objects import (
    Analyst,
    ResearchEvaluation,
    ResearchFinding,
    ResearchFindingBatch,
    ResearchPlan,
)
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

MAX_RESEARCH_LOOPS = 3
MAX_TOOL_CALLS_PER_PASS = 6
MAX_EXTRACTION_TOOL_OUTPUT_CHARS = 4_000
MAX_EXTRACTION_TRANSCRIPT_CHARS = 24_000
EXTRACTION_TIMEOUT_ATTEMPTS = 2
TRACKING_QUERY_KEYS = {"fbclid", "gclid", "mc_cid", "mc_eid"}


def _analyst(value: Analyst | dict[str, Any]) -> Analyst:
    return value if isinstance(value, Analyst) else Analyst.model_validate(value)


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



def _bounded_tool_outputs(messages: list[Any]) -> list[dict[str, str]]:
    """Keep structured extraction fast while retaining evidence from this pass only."""

    outputs: list[dict[str, str]] = []
    remaining = MAX_EXTRACTION_TRANSCRIPT_CHARS
    for message in messages:
        if not isinstance(message, ToolMessage) or remaining <= 0:
            continue
        content = str(message.content)
        limit = min(MAX_EXTRACTION_TOOL_OUTPUT_CHARS, remaining)
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
        if len(valid) == 4:
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
    payload = "\u241f".join(
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
    """Retain unique, source-addressable findings without an LLM decision."""
    seen = {evidence_key(finding) for finding in existing if finding.source_url.strip()}
    unique: list[ResearchFinding] = []
    for finding in candidates:
        if not finding.source_url.strip() or not finding.claim.strip() or not finding.excerpt.strip():
            continue
        canonical = canonicalize_url(finding.source_url)
        if not canonical.startswith(("http://", "https://")):
            continue
        normalized = finding.model_copy(update={"source_url": canonical})
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
    structured_llm = heavy_llm.with_structured_output(ResearchPlan)
    plan = structured_llm.invoke(
        [
            SystemMessage(content=planner_instructions),
            HumanMessage(content=_json(planner_input)),
        ]
    )
    history = {_normalise_text(question) for question in state["question_history"]}
    questions: list[str] = []
    for question in plan.sub_questions:
        cleaned = re.sub(r"\s+", " ", question).strip()
        normalized = _normalise_text(cleaned)
        if cleaned and normalized not in history and normalized not in {
            _normalise_text(item) for item in questions
        }:
            questions.append(cleaned)
        if len(questions) == 3:
            break

    if len(questions) < 2:
        correction = structured_llm.invoke(
            [
                SystemMessage(content=planner_instructions),
                HumanMessage(
                    content=(
                        "Return two or three distinct additional questions. Do not repeat "
                        f"these rejected or duplicate questions: {_json(plan.sub_questions)}.\n"
                        f"Context: {_json(planner_input)}"
                    )
                ),
            ], thinking_level="medium")
        for question in correction.sub_questions:
            cleaned = re.sub(r"\s+", " ", question).strip()
            normalized = _normalise_text(cleaned)
            if cleaned and normalized not in history and normalized not in {
                _normalise_text(item) for item in questions
            }:
                questions.append(cleaned)
            if len(questions) == 3:
                break

    return {
        "sub_questions": questions,
        "question_history": questions,
        "tool_call_count": 0,
        "budget_exhausted": False,
        "messages": [
            RemoveMessage(id=REMOVE_ALL_MESSAGES),
            HumanMessage(content=_current_research_brief(state, questions)),
        ],
    }


def researcher_node(state: AnalystResearchState) -> dict[str, Any]:
    """Use tools only to answer the Planner's current research questions."""
    bound_llm = medium_llm.bind_tools(RESEARCH_TOOLS)
    response = bound_llm.invoke(
        [SystemMessage(content=researcher_instructions), *state["messages"]]
    )
    return {"messages": [response]}


def _last_ai_message(state: AnalystResearchState) -> AIMessage | None:
    for message in reversed(state["messages"]):
        if isinstance(message, AIMessage):
            return message
    return None


def route_researcher(state: AnalystResearchState) -> Literal["limit_tool_calls", "extract_findings"]:
    message = _last_ai_message(state)
    requested = len(message.tool_calls) if message else 0
    remaining = MAX_TOOL_CALLS_PER_PASS - state["tool_call_count"]
    if requested == 0 or remaining <= 0:
        return "extract_findings"
    return "limit_tool_calls"


def limit_tool_calls(state: AnalystResearchState) -> dict[str, Any]:
    """Trim a model request to the exact remaining hard tool-call allowance."""
    message = _last_ai_message(state)
    if message is None:
        return {"budget_exhausted": True}
    remaining = max(0, MAX_TOOL_CALLS_PER_PASS - state["tool_call_count"])
    allowed_calls = list(message.tool_calls[:remaining])
    bounded_message = message.model_copy(update={"tool_calls": allowed_calls})
    new_count = state["tool_call_count"] + len(allowed_calls)
    return {
        "messages": [bounded_message],
        "tool_call_count": new_count,
        "budget_exhausted": new_count >= MAX_TOOL_CALLS_PER_PASS,
    }


def route_limited_tools(state: AnalystResearchState) -> Literal["tools", "extract_findings"]:
    decision = tools_condition(state)
    return "tools" if decision == "tools" else "extract_findings"


def route_after_tools(state: AnalystResearchState) -> Literal["researcher_node", "extract_findings"]:
    if state["budget_exhausted"] or state["tool_call_count"] >= MAX_TOOL_CALLS_PER_PASS:
        return "extract_findings"
    return "researcher_node"


def extract_findings(state: AnalystResearchState) -> dict[str, Any]:
    """Convert one completed pass's temporary tool output into durable evidence."""
    tool_outputs = _bounded_tool_outputs(state["messages"])
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
    structured_llm = medium_llm.with_structured_output(
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
            return {"loop_count": state["loop_count"] + 1}

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
    }
def evaluate_research(state: AnalystResearchState) -> dict[str, Any]:
    """Assess evidence coverage without tools or external retrieval."""
    analyst = _analyst(state["analyst"])
    evaluation = heavy_llm.with_structured_output(ResearchEvaluation).invoke(
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
    )
    return {"evaluation": evaluation, "feedback": evaluation.feedback}


def route_evaluation(state: AnalystResearchState) -> Literal["planner_node", "writer_node"]:
    evaluation = state.get("evaluation")
    if evaluation is not None and evaluation.is_complete:
        return "writer_node"
    if state["loop_count"] >= MAX_RESEARCH_LOOPS:
        return "writer_node"
    return "planner_node"


def writer_node(state: AnalystResearchState) -> dict[str, str]:
    """Write from durable findings only; this model is deliberately not tool-bound."""
    analyst = _analyst(state["analyst"])
    response = medium_llm.invoke(
        [
            SystemMessage(content=writer_instructions),
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
    return {"draft": _format_sections(response.content)}


RESEARCH_TOOLS = build_research_tools()
TOOL_NODE = ToolNode(RESEARCH_TOOLS, handle_tool_errors=True)


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


def conduct_research(state: dict[str, Any]) -> dict[str, list[str]]:
    """Run one isolated analyst subgraph and expose only its completed draft upstream."""
    result = analyst_research_graph.invoke(state)
    return {"sections": [result["draft"]]}
