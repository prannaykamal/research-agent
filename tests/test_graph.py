from langchain_core.messages import AIMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.types import Send
from typing_extensions import TypedDict

from src.analyst_research import analyst_research_graph
from src.utils.edges import initiate_all_research
from src.utils.nodes import _format_sections
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


def test_section_formatter_flattens_parallel_reducer_output() -> None:
    assert _format_sections(
        [[{"text": "Draft A"}], {"draft": [{"content": "Draft B"}, "Draft C"]}]
    ) == "Draft A\n\nDraft B\n\nDraft C"

def test_introduction_and_conclusion_run_after_the_report() -> None:
    from src.deep_agent import graph

    edges = {(edge.source, edge.target) for edge in graph.get_graph().edges}
    assert ("write_report", "write_introduction") in edges
    assert ("write_report", "write_conclusion") in edges
    assert ("conduct_research", "write_conclusion") not in edges
    assert ("conduct_research", "write_introduction") not in edges


def test_conclusion_summarizes_the_report_body_not_raw_sections(monkeypatch) -> None:
    from types import SimpleNamespace

    import src.utils.nodes as nodes

    prompts = []

    class Recorder:
        def invoke(self, messages):
            prompts.append(messages[0].content)
            return AIMessage(content="## Conclusion")

    namespace = SimpleNamespace(heavy=None, medium=None, writer=None, light=Recorder())
    monkeypatch.setattr(nodes, "get_models", lambda _name=None: namespace)
    state = {
        "topic": "Climate",
        "sections": ["RAW SECTION: emissions rose 350 times"],
        "content": "## Insights" + chr(10) + "RECONCILED BODY: estimates range from 180 to 3,000 times.",
    }
    nodes.write_conclusion(state)
    nodes.write_introduction(state)
    assert all("RECONCILED BODY" in prompt for prompt in prompts)
    assert not any("RAW SECTION" in prompt for prompt in prompts)


def test_no_report_body_skips_the_introduction_and_conclusion_models() -> None:
    import src.utils.nodes as nodes

    state = {"topic": "Climate", "sections": [], "content": nodes.NO_RESEARCH_MESSAGE}
    assert nodes.write_introduction(state) == {"introduction": "# Climate"}
    assert nodes.write_conclusion(state) == {"conclusion": ""}
