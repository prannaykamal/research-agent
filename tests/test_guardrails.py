import socket
import time
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_tavily import TavilySearch

import src.analyst_research as research
import src.utils.nodes as nodes
from src.utils.guardrails import (
    MAX_ANALYST_REVISIONS,
    MAX_EXCERPT_CHARS,
    MAX_FINDINGS_PER_ANALYST,
    MAX_SECTION_CHARS,
    analyst_stop_reason,
    validate_max_analysts,
)
from src.utils.objects import Analyst, Perspectives, ResearchFinding
from src.utils.profiles import get_profile
from src.utils.tools import MAX_TOOL_OUTPUT_CHARS, BoundedTavilySearch, _validate_public_http_url


def analyst(index: int = 0) -> Analyst:
    return Analyst(affiliation=f"Org {index}", name=f"Analyst {index}", role="Role", description="D")


def finding(index: int, excerpt: str = "Evidence") -> ResearchFinding:
    return ResearchFinding(
        sub_question="Q",
        claim=f"Claim {index}",
        source_title="Source",
        source_url=f"https://example.com/{index}",
        excerpt=excerpt,
        source_type="web",
    )


def research_state(**overrides):
    state = {
        "topic": "Topic",
        "analyst": analyst(),
        "model_profile": "quality",
        "sub_questions": ["Q"],
        "question_history": ["Q"],
        "research_findings": [],
        "feedback": "",
        "loop_count": 0,
        "draft": "",
        "messages": [HumanMessage(content="brief")],
        "tool_call_count": 0,
        "budget_exhausted": False,
        "evaluation": None,
        "started_at": time.time(),
        "researcher_turns": 0,
        "llm_calls": 0,
        "input_tokens": 0,
    }
    state.update(overrides)
    return state


# System-wide guardrails


@pytest.mark.parametrize("value", [0, 11, -1, True, "5", 2.0, None])
def test_max_analysts_outside_one_to_ten_is_rejected(value) -> None:
    with pytest.raises(ValueError, match="max_analysts"):
        validate_max_analysts(value)


@pytest.mark.parametrize("value", [1, 5, 10])
def test_max_analysts_inside_range_is_accepted(value) -> None:
    assert validate_max_analysts(value) == value


class PanelModel:
    def __init__(self, count: int) -> None:
        self.count = count

    def with_structured_output(self, schema, **_kwargs):
        assert schema is Perspectives
        return self

    def invoke(self, _messages):
        return Perspectives(analysts=[analyst(index) for index in range(self.count)])


def use_node_models(monkeypatch, medium) -> None:
    namespace = SimpleNamespace(heavy=None, medium=medium, writer=None, light=None)
    monkeypatch.setattr(nodes, "get_models", lambda _name=None: namespace)


def test_create_analysts_returns_exactly_the_requested_count(monkeypatch) -> None:
    use_node_models(monkeypatch, PanelModel(count=12))
    update = nodes.create_analysts({"topic": "T", "max_analysts": 3, "model_profile": "fast"})
    assert len(update["analysts"]) == 3
    assert update["model_profile"] == "fast"


def test_create_analysts_rejects_out_of_range_before_calling_a_model(monkeypatch) -> None:
    use_node_models(monkeypatch, None)
    with pytest.raises(ValueError, match="between 1 and 10"):
        nodes.create_analysts({"topic": "T", "max_analysts": 11})


def test_create_analysts_rejects_unknown_profile(monkeypatch) -> None:
    use_node_models(monkeypatch, PanelModel(count=1))
    with pytest.raises(ValueError, match="model_profile"):
        nodes.create_analysts({"topic": "T", "max_analysts": 1, "model_profile": "turbo"})


def test_revision_feedback_is_capped(monkeypatch) -> None:
    monkeypatch.setattr(nodes, "interrupt", lambda _payload: "Add an economist")
    state = {"topic": "T", "max_analysts": 2, "analysts": []}

    first = nodes.human_feedback({**state, "revision_count": 0})
    assert first == {"human_analyst_feedback": "Add an economist", "revision_count": 1}

    capped = nodes.human_feedback({**state, "revision_count": MAX_ANALYST_REVISIONS})
    assert capped == {"human_analyst_feedback": None}


def test_synthesis_input_is_bounded() -> None:
    state = {"sections": ["x" * (MAX_SECTION_CHARS * 2)] * 10}
    assert len(nodes._bounded_sections(state)) <= 10 * MAX_SECTION_CHARS + 20


def test_finalize_report_records_stats_and_failures() -> None:
    stats = [
        {"analyst": "A", "status": "completed", "duration_seconds": 2.0, "llm_calls": 5, "input_tokens": 100},
        {"analyst": "B", "status": "failed", "duration_seconds": 4.0, "llm_calls": 1, "input_tokens": 10},
    ]
    update = nodes.finalize_report(
        {
            "topic": "T",
            "model_profile": "fast",
            "content": "## Insights\nBody",
            "introduction": "# T",
            "conclusion": "## Conclusion",
            "analyst_stats": stats,
        }
    )
    assert update["run_stats"]["mean_analyst_seconds"] == 3.0
    assert update["run_stats"]["failed"] == ["B"]
    assert "research failed for B" in update["final_report"]


# Per-analyst guardrails


def test_stop_reasons_for_deadline_tokens_and_calls() -> None:
    profile = get_profile("fast")
    now = time.time()
    assert analyst_stop_reason(research_state(started_at=now), profile, now) is None
    assert (
        analyst_stop_reason(
            research_state(started_at=now - profile.analyst_deadline_seconds), profile, now
        )
        == "deadline"
    )
    assert (
        analyst_stop_reason(
            research_state(input_tokens=profile.analyst_input_token_budget), profile, now
        )
        == "token_budget"
    )
    assert (
        analyst_stop_reason(research_state(llm_calls=profile.max_llm_calls_per_analyst), profile, now)
        == "llm_call_budget"
    )


def test_exhausted_deadline_skips_research_and_the_evaluator() -> None:
    expired = research_state(started_at=time.time() - 10_000)
    expired["messages"].append(
        AIMessage(content="", tool_calls=[{"name": "wikipedia", "args": {}, "id": "1", "type": "tool_call"}])
    )
    assert research.route_researcher(expired) == "extract_findings"
    assert research.route_after_tools(expired) == "extract_findings"
    assert research.evaluate_research(expired) == {}
    assert research.route_evaluation(expired) == "writer_node"


def test_researcher_turn_cap_ends_the_pass() -> None:
    profile = get_profile("quality")
    state = research_state(researcher_turns=profile.max_researcher_turns, tool_call_count=1)
    assert research.route_after_tools(state) == "extract_findings"
    assert research.route_after_tools(research_state(researcher_turns=1, tool_call_count=1)) == "researcher_node"


def test_disallowed_tools_are_dropped_for_the_fast_profile() -> None:
    calls = [
        {"name": "scrape_webpage", "args": {"url": "https://example.com"}, "id": "1", "type": "tool_call"},
        {"name": "wikipedia", "args": {"query": "q"}, "id": "2", "type": "tool_call"},
    ]
    state = research_state(model_profile="fast")
    state["messages"].append(AIMessage(content="", tool_calls=calls))
    update = research.limit_tool_calls(state)
    assert [call["name"] for call in update["messages"][0].tool_calls] == ["wikipedia"]
    assert update["tool_call_count"] == 1


def test_researcher_sees_clipped_tool_outputs() -> None:
    long_output = ToolMessage(content="x" * 10_000, tool_call_id="1")
    view = research._researcher_view([long_output], 1_500)
    assert len(view[0].content) <= 1_500
    assert len(long_output.content) == 10_000


def test_findings_are_truncated_and_capped_per_analyst() -> None:
    long = finding(0, excerpt="y" * 5_000)
    assert len(research.deduplicate_findings([], [long])[0].excerpt) <= MAX_EXCERPT_CHARS

    existing = [finding(index) for index in range(MAX_FINDINGS_PER_ANALYST - 1)]
    additions = research.deduplicate_findings(existing, [finding(100), finding(101)])
    assert len(additions) == 1


def test_evaluator_parse_failure_falls_back_to_continue(monkeypatch) -> None:
    class Unparseable:
        def with_structured_output(self, _schema, **_kwargs):
            return self

        def invoke(self, _messages):
            return {"parsed": None, "raw": AIMessage(content="not json"), "parsing_error": None}

    namespace = SimpleNamespace(heavy=Unparseable(), medium=None, writer=None, light=None)
    monkeypatch.setattr(research, "get_models", lambda _name=None: namespace)
    update = research.evaluate_research(research_state(loop_count=1))
    assert update["evaluation"].is_complete is False


def test_planner_falls_back_to_deterministic_questions(monkeypatch) -> None:
    class EmptyPlanner:
        def with_structured_output(self, _schema, **_kwargs):
            return self

        def invoke(self, _messages):
            return {"parsed": None, "raw": AIMessage(content=""), "parsing_error": None}

    namespace = SimpleNamespace(heavy=EmptyPlanner(), medium=None, writer=None, light=None)
    monkeypatch.setattr(research, "get_models", lambda _name=None: namespace)
    update = research.planner_node(research_state(question_history=[]))
    assert 2 <= len(update["sub_questions"]) <= 3


def test_analyst_failure_is_isolated(monkeypatch) -> None:
    class Exploding:
        def invoke(self, _state):
            raise RuntimeError("provider down")

    monkeypatch.setattr(research, "analyst_research_graph", Exploding())
    update = research.conduct_research(research_state())
    assert update["sections"] == []
    assert update["analyst_stats"][0]["status"] == "failed"
    assert "provider down" in update["analyst_stats"][0]["stop_reason"]


def test_ssrf_rejects_hostnames_resolving_to_private_addresses(monkeypatch) -> None:
    def resolve(address):
        return lambda *_args, **_kwargs: [(socket.AF_INET, 0, 0, "", (address, 0))]

    monkeypatch.setattr(socket, "getaddrinfo", resolve("127.0.0.1"))
    with pytest.raises(ValueError, match="Private network"):
        _validate_public_http_url("https://127.0.0.1.nip.io/admin")

    monkeypatch.setattr(socket, "getaddrinfo", resolve("169.254.169.254"))
    with pytest.raises(ValueError, match="Private network"):
        _validate_public_http_url("http://metadata.example/")

    monkeypatch.setattr(socket, "getaddrinfo", resolve("93.184.216.34"))
    assert _validate_public_http_url("https://example.com/page") == "https://example.com/page"

    with pytest.raises(ValueError, match="HTTP"):
        _validate_public_http_url("file:///etc/passwd")


def test_tavily_output_is_bounded(monkeypatch) -> None:
    huge = {"results": [{"url": "https://example.com", "content": "z" * 50_000}]}
    monkeypatch.setattr(TavilySearch, "_run", lambda self, *args, **kwargs: huge)
    output = BoundedTavilySearch(max_results=3)._run(query="q")
    assert isinstance(output, str)
    assert len(output) <= MAX_TOOL_OUTPUT_CHARS
