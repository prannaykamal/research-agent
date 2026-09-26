from langchain_core.messages import HumanMessage
from langgraph.types import Send
from typing import Literal

from src.utils.states import GenerateAnalystsState, ResearchGraphState


def should_continue(state: GenerateAnalystsState) -> Literal["create_analysts", "dummy"]:
    """Regenerate analysts only when the reviewer supplied revision feedback."""
    return "create_analysts" if state.get("human_analyst_feedback") else "dummy"


def initiate_all_research(state: ResearchGraphState):
    """Fan out one isolated bounded research subgraph per approved analyst."""
    if state.get("human_analyst_feedback"):
        return "create_analysts"

    return [
        Send(
            "conduct_research",
            {
                "topic": state["topic"],
                "analyst": analyst,
                "sub_questions": [],
                "question_history": [],
                "research_findings": [],
                "feedback": "",
                "loop_count": 0,
                "draft": "",
                "messages": [],
                "tool_call_count": 0,
                "budget_exhausted": False,
                "evaluation": None,
            },
        )
        for analyst in state["analysts"]
    ]
