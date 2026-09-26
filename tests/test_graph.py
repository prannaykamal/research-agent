from langchain_core.messages import AIMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.types import Send
from typing_extensions import TypedDict

from src.analyst_research import analyst_research_graph
from src.utils.edges import initiate_all_research
from src.utils.objects import Analyst


@tool
def echo_evidence(query: str) -> str:
    """Return evidence for a test query."""
    return f"evidence:{query}"


class ToolState(TypedDict):
    messages: list


def test_toolnode_executes_a_standard_tool_call() -> None:
    builder = StateGraph(ToolState)
    builder.add_node("tools", ToolNode([echo_evidence]))
    builder.add_edge(START, "tools")
    builder.add_edge("tools", END)
    graph = builder.compile()
    result = graph.invoke(
        {
            "messages": [
                AIMessage(
                    content="",
                    tool_calls=[{"name": "echo_evidence", "args": {"query": "q"}, "id": "1", "type": "tool_call"}],
                )
            ]
        }
    )
    assert result["messages"][0].content == "evidence:q"


def test_parent_send_initializes_isolated_research_state() -> None:
    analysts = [
        Analyst(affiliation="A", name="One", role="R", description="D"),
        Analyst(affiliation="B", name="Two", role="R", description="D"),
    ]
    sends = initiate_all_research({"topic": "Topic", "analysts": analysts, "sections": []})

    assert all(isinstance(send, Send) and send.node == "conduct_research" for send in sends)
    assert sends[0].arg["research_findings"] == []
    assert sends[1].arg["research_findings"] == []
    assert sends[0].arg is not sends[1].arg


def test_compiled_research_graph_is_available() -> None:
    assert analyst_research_graph is not None
