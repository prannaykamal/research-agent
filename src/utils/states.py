import operator
from typing import List, Optional

from langchain_core.messages import AnyMessage
from langgraph.graph import add_messages
from typing_extensions import Annotated, NotRequired, TypedDict

from src.utils.objects import Analyst, ResearchEvaluation, ResearchFinding


class GenerateAnalystsState(TypedDict):
    topic: str
    max_analysts: int
    human_analyst_feedback: NotRequired[Optional[str]]
    analysts: NotRequired[List[Analyst]]


class AnalystResearchState(TypedDict):
    """Durable analyst evidence plus the current pass's temporary tool transcript."""

    topic: str
    analyst: Analyst
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


class ResearchGraphState(TypedDict):
    topic: str
    max_analysts: int
    human_analyst_feedback: NotRequired[Optional[str]]
    analysts: List[Analyst]
    sections: Annotated[list[str], operator.add]
    introduction: str
    content: str
    conclusion: str
    final_report: str
