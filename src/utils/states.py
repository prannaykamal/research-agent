import operator
from typing import Any, List, Optional

from langchain_core.messages import AnyMessage
from langgraph.graph import add_messages
from pydantic import Field
from typing_extensions import Annotated, NotRequired, TypedDict

from src.utils.guardrails import MAX_ANALYSTS, MIN_ANALYSTS
from src.utils.objects import Analyst, ResearchEvaluation, ResearchFinding
from src.utils.profiles import ProfileName


class ResearchInput(TypedDict):
    """Run input: max_analysts must be 1–10; model_profile defaults to "quality"."""

    topic: str
    max_analysts: Annotated[int, Field(ge=MIN_ANALYSTS, le=MAX_ANALYSTS)]
    model_profile: NotRequired[ProfileName]


class GenerateAnalystsState(TypedDict):
    topic: str
    max_analysts: int
    model_profile: NotRequired[ProfileName]
    human_analyst_feedback: NotRequired[Optional[str]]
    revision_count: NotRequired[int]
    analysts: NotRequired[List[Analyst]]


class AnalystResearchState(TypedDict):
    """Durable analyst evidence plus the current pass's temporary tool transcript."""

    topic: str
    analyst: Analyst
    model_profile: NotRequired[ProfileName]
    sub_questions: List[str]
    question_history: Annotated[List[str], operator.add]
    research_findings: Annotated[List[ResearchFinding], operator.add]
    feedback: str
    loop_count: int
    draft: str

    messages: Annotated[List[AnyMessage], add_messages]
    tool_call_count: int
    budget_exhausted: bool
    evaluation: Optional[ResearchEvaluation]

    # Per-analyst guardrail accounting.
    started_at: NotRequired[float]
    researcher_turns: NotRequired[int]
    llm_calls: NotRequired[int]
    input_tokens: NotRequired[int]
    stop_reason: NotRequired[Optional[str]]


class ResearchGraphState(TypedDict):
    topic: str
    max_analysts: int
    model_profile: NotRequired[ProfileName]
    human_analyst_feedback: NotRequired[Optional[str]]
    revision_count: NotRequired[int]
    analysts: List[Analyst]
    sections: Annotated[list[str], operator.add]
    analyst_stats: Annotated[list[dict[str, Any]], operator.add]
    introduction: str
    content: str
    conclusion: str
    final_report: str
    run_stats: dict[str, Any]
