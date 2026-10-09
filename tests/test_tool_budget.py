from langchain_core.messages import AIMessage, HumanMessage

from src.analyst_research import (
    limit_tool_calls,
    max_researcher_turns,
    route_after_tools,
    route_researcher,
)
from src.utils.objects import Analyst
from src.utils.profiles import get_profile

MAX_TOOL_CALLS_PER_PASS = get_profile("quality").max_tool_calls_per_pass


def state_with_calls(call_count: int, used: int = 0):
    calls = [
        {"name": "wikipedia", "args": {"query": str(index)}, "id": str(index), "type": "tool_call"}
        for index in range(call_count)
    ]
    return {
        "topic": "topic",
        "analyst": Analyst(affiliation="A", name="N", role="R", description="D"),
        "sub_questions": ["q"],
        "question_history": ["q"],
        "research_findings": [],
        "feedback": "",
        "loop_count": 0,
        "draft": "",
        "messages": [HumanMessage(content="brief"), AIMessage(content="", tool_calls=calls)],
        "tool_call_count": used,
        "budget_exhausted": False,
        "evaluation": None,
    }


MIN_TOOL_CALLS_PER_PASS = get_profile("quality").min_tool_calls_per_pass
MAX_RESEARCHER_TURNS = get_profile("quality").max_researcher_turns


def test_zero_requested_calls_goes_to_extraction_once_the_floor_is_met() -> None:
    assert route_researcher(state_with_calls(0, MIN_TOOL_CALLS_PER_PASS)) == "extract_findings"


def test_stopping_below_the_search_floor_sends_the_researcher_back() -> None:
    state = {**state_with_calls(0, 1), "researcher_turns": 1}
    assert route_researcher(state) == "researcher_node"


def test_search_floor_outlasts_the_turn_cap_until_the_extended_limit() -> None:
    profile = get_profile("quality")
    at_cap = {**state_with_calls(0, 1), "researcher_turns": MAX_RESEARCHER_TURNS}
    assert route_researcher(at_cap) == "researcher_node"
    extended = {**state_with_calls(0, 1), "researcher_turns": max_researcher_turns(profile)}
    assert route_researcher(extended) == "extract_findings"


def test_turn_cap_applies_once_the_floor_is_met() -> None:
    state = {**state_with_calls(0, MIN_TOOL_CALLS_PER_PASS), "researcher_turns": MAX_RESEARCHER_TURNS}
    assert route_researcher(state) == "extract_findings"


def test_zero_remaining_budget_goes_to_extraction() -> None:
    assert route_researcher(state_with_calls(1, MAX_TOOL_CALLS_PER_PASS)) == "extract_findings"


def test_requested_calls_are_trimmed_to_remaining_budget() -> None:
    state = state_with_calls(4, MAX_TOOL_CALLS_PER_PASS - 1)
    update = limit_tool_calls(state)

    assert update["tool_call_count"] == MAX_TOOL_CALLS_PER_PASS
    assert len(update["messages"][0].tool_calls) == 1
    assert update["budget_exhausted"] is True


def test_within_budget_calls_return_to_researcher_after_tools() -> None:
    state = state_with_calls(1, 2)
    update = limit_tool_calls(state)
    state.update(update)

    assert state["tool_call_count"] == 3
    assert route_after_tools(state) == "researcher_node"
