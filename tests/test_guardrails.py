import json
import socket
import uuid
import time
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.runnables import RunnableLambda
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
from src.utils.objects import Analyst, Perspectives, QuestionRequirements, ResearchFinding
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
    """Answers the question decomposition, then the panel request."""

    def __init__(self, count: int, requirements: list[str] | None = None) -> None:
        self.count = count
        self.requirements = requirements or ["Requirement"]
        self.schema = None

    def with_structured_output(self, schema, **_kwargs):
        assert schema in (Perspectives, QuestionRequirements)
        self.schema = schema
        return self

    def invoke(self, _messages):
        if self.schema is QuestionRequirements:
            parsed = QuestionRequirements(requirements=self.requirements)
            return {"parsed": parsed, "raw": AIMessage(content=""), "parsing_error": None}
        return Perspectives(analysts=[analyst(index) for index in range(self.count)])


def use_node_models(monkeypatch, panel) -> None:
    """Persona and requirement generation run on the profile's panel tier."""
    namespace = SimpleNamespace(heavy=None, medium=None, writer=None, light=None, panel=panel)
    monkeypatch.setattr(nodes, "get_models", lambda _name=None: namespace)


def test_create_analysts_returns_exactly_the_requested_count(monkeypatch) -> None:
    use_node_models(monkeypatch, PanelModel(count=12))
    update = nodes.create_analysts({"topic": "T", "max_analysts": 3, "model_profile": "fast"})
    assert len(update["analysts"]) == 3
    assert update["model_profile"] == "fast"


def test_create_analysts_rejects_out_of_range_before_calling_a_model(monkeypatch) -> None:
    use_node_models(monkeypatch, None)
    with pytest.raises(ValueError, match="between 1 and 10"):
        nodes.create_analysts({"topic": "T", "max_analysts": 11, "model_profile": "fast"})
    # The default profile is Deep Research, which caps the panel at 5.
    with pytest.raises(ValueError, match="between 1 and 5"):
        nodes.create_analysts({"topic": "T", "max_analysts": 6})


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
    floor = profile.min_tool_calls_per_pass
    at_cap = research_state(researcher_turns=profile.max_researcher_turns, tool_call_count=floor)
    assert research.route_after_tools(at_cap) == "extract_findings"
    assert research.route_after_tools(research_state(researcher_turns=1, tool_call_count=1)) == "researcher_node"


def test_search_floor_extends_the_turn_cap_by_turns_not_without_limit() -> None:
    profile = get_profile("fast")
    # One search per turn reaches the cap below the floor: one more turn is allowed.
    below_floor = research_state(
        model_profile="fast", researcher_turns=profile.max_researcher_turns, tool_call_count=3
    )
    assert research.route_after_tools(below_floor) == "researcher_node"
    extended = research.max_researcher_turns(profile)
    exhausted = research_state(model_profile="fast", researcher_turns=extended, tool_call_count=3)
    assert research.route_after_tools(exhausted) == "extract_findings"


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
        def invoke(self, _state, _config=None):
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


class WorstCaseModel:
    """Researcher always wants another tool; the evaluator is never satisfied."""

    def __init__(self, schema=None) -> None:
        self.schema = schema
        self.tools_bound = False
        self.calls = 0

    def with_structured_output(self, schema, **_kwargs):
        return WorstCaseModel(schema)

    def bind_tools(self, _tools, **_kwargs):
        model = WorstCaseModel(self.schema)
        model.tools_bound = True
        return model

    def invoke(self, _messages, *args, **kwargs):
        from src.utils.objects import ResearchEvaluation, ResearchFindingBatch, ResearchPlan

        self.calls += 1
        if self.schema is ResearchPlan:
            plan = ResearchPlan(sub_questions=[f"Question {uuid.uuid4().hex}", f"Question {uuid.uuid4().hex}"])
            return {"parsed": plan, "raw": AIMessage(content=""), "parsing_error": None}
        if self.schema is ResearchEvaluation:
            verdict = ResearchEvaluation(is_complete=False, coverage_gaps=["gap"], feedback="Dig deeper.")
            return {"parsed": verdict, "raw": AIMessage(content=""), "parsing_error": None}
        if self.schema is ResearchFindingBatch:
            batch = ResearchFindingBatch(findings=[finding(int(uuid.uuid4().int % 10_000))])
            return {"parsed": batch, "raw": AIMessage(content=""), "parsing_error": None}
        if self.tools_bound:
            call = {"name": "wikipedia", "args": {"query": "q"}, "id": uuid.uuid4().hex, "type": "tool_call"}
            return AIMessage(content="", tool_calls=[call])
        return AIMessage(content="Draft")


@pytest.mark.parametrize("profile_name", ["quality", "fast"])
def test_worst_case_analyst_finishes_within_the_recursion_limit(monkeypatch, profile_name) -> None:
    from src.utils import tools as tool_module

    model = WorstCaseModel()
    namespace = SimpleNamespace(heavy=model, medium=model, writer=model, light=model)
    monkeypatch.setattr(research, "get_models", lambda _name=None: namespace)
    monkeypatch.setattr(
        tool_module.WikipediaEvidenceTool,
        "_run",
        lambda self, query, run_manager=None: json.dumps([{"source_url": "https://example.org/a", "excerpt": "x"}]),
    )
    # The API server runs the parent graph with LangChain's default recursion_limit of 25,
    # which a node's nested subgraph inherits; reproduce that here.
    node = RunnableLambda(research.conduct_research)
    update = node.invoke(research_state(model_profile=profile_name, started_at=None), {"recursion_limit": 25})
    stats = update["analyst_stats"][0]
    assert stats["status"] == "completed", stats["stop_reason"]
    assert stats["stop_reason"] == "max_passes"


def test_arxiv_tool_uses_the_current_arxiv_client(monkeypatch) -> None:
    import datetime

    import arxiv

    from src.utils.tools import ArxivEvidenceTool

    result = arxiv.Result(
        entry_id="http://arxiv.org/abs/2401.00001v1",
        updated=datetime.datetime(2024, 1, 1),
        published=datetime.datetime(2023, 12, 30),
        title="A paper",
        authors=[arxiv.Result.Author("Ada Lovelace")],
        summary="We measure things.",
    )
    seen = {}

    def fake_results(self, search, offset=0):
        seen["query"] = search.query
        return iter([result])

    monkeypatch.setattr(arxiv.Client, "results", fake_results)
    output = json.loads(ArxivEvidenceTool()._run("grid storage"))
    assert seen["query"] == "grid storage"
    assert output == [
        {
            "source_title": "A paper",
            "source_url": "http://arxiv.org/abs/2401.00001v1",
            "excerpt": "We measure things.",
            "source_type": "arxiv",
            "published": "2023-12-30",
        }
    ]


def test_wikipedia_requests_identify_this_project() -> None:
    import wikipedia.wikipedia as wikipedia_client

    from src.utils.tools import USER_AGENT

    assert wikipedia_client.USER_AGENT == USER_AGENT
    assert "research-agent" in USER_AGENT


# Question requirements


def test_unowned_requirements_go_to_the_least_loaded_analyst() -> None:
    from src.utils.guardrails import assign_requirements

    requirements = ["Evolution of fusion research", "Commercial viability", "Remaining challenges"]
    # One analyst owns everything, however narrowly the model scoped it.
    assert assign_requirements([["Commercial viability"]], requirements) == [requirements]
    # Unclaimed requirements spread one per analyst; paraphrases map to the canonical text.
    assert assign_requirements(
        [["How fusion research has evolved"], [], []], requirements
    ) == [["Evolution of fusion research"], ["Commercial viability"], ["Remaining challenges"]]


def test_create_analysts_gives_a_single_analyst_every_requirement(monkeypatch) -> None:
    requirements = ["Evolution of fusion research", "Commercial viability"]
    use_node_models(monkeypatch, PanelModel(count=1, requirements=requirements))
    update = nodes.create_analysts({"topic": "Fusion", "max_analysts": 1})
    assert update["requirements"] == requirements
    assert update["analysts"][0].requirements == requirements


def test_panel_revision_reuses_the_original_requirements(monkeypatch) -> None:
    use_node_models(monkeypatch, PanelModel(count=2, requirements=["Should not be asked"]))
    update = nodes.create_analysts(
        {"topic": "T", "max_analysts": 2, "requirements": ["Kept A", "Kept B"]}
    )
    assert update["requirements"] == ["Kept A", "Kept B"]
    assert [a.requirements for a in update["analysts"]] == [["Kept A"], ["Kept B"]]


class SatisfiedEvaluator:
    def with_structured_output(self, _schema, **_kwargs):
        return self

    def invoke(self, _messages):
        from src.utils.objects import ResearchEvaluation

        verdict = ResearchEvaluation(is_complete=True, feedback="Looks sufficient.")
        return {"parsed": verdict, "raw": AIMessage(content=""), "parsing_error": None}


def test_evaluator_cannot_finish_while_an_owned_requirement_lacks_evidence(monkeypatch) -> None:
    namespace = SimpleNamespace(heavy=SatisfiedEvaluator(), medium=None, writer=None, light=None)
    monkeypatch.setattr(research, "get_models", lambda _name=None: namespace)
    owner = analyst().model_copy(update={"requirements": ["Costs", "Safety"]})
    costs = finding(1).model_copy(update={"requirement": "Costs"})

    partial = research.evaluate_research(
        research_state(analyst=owner, research_findings=[costs], loop_count=1)
    )
    assert partial["evaluation"].is_complete is False
    assert "Safety" in partial["evaluation"].coverage_gaps
    assert "Safety" in partial["feedback"]

    safety = finding(2).model_copy(update={"requirement": "Safety"})
    complete = research.evaluate_research(
        research_state(analyst=owner, research_findings=[costs, safety], loop_count=1)
    )
    assert complete["evaluation"].is_complete is True


def test_requirement_floor_never_outlasts_the_budget() -> None:
    owner = analyst().model_copy(update={"requirements": ["Costs"]})
    expired = research_state(analyst=owner, started_at=time.time() - 10_000, loop_count=1)
    assert research.evaluate_research(expired) == {}
    assert research.route_evaluation(expired) == "writer_node"


# Tool failures


class ToolCallingModel(WorstCaseModel):
    """Like WorstCaseModel, but the Researcher calls the named tool."""

    tool = "tavily_search"

    def with_structured_output(self, schema, **_kwargs):
        model = type(self)(schema)
        return model

    def bind_tools(self, _tools, **_kwargs):
        model = type(self)(self.schema)
        model.tools_bound = True
        return model

    def invoke(self, messages, *args, **kwargs):
        if self.tools_bound:
            call = {"name": self.tool, "args": {"query": "q"}, "id": uuid.uuid4().hex, "type": "tool_call"}
            return AIMessage(content="", tool_calls=[call])
        return super().invoke(messages, *args, **kwargs)


def run_analyst_with_tool(monkeypatch, model) -> dict:
    namespace = SimpleNamespace(heavy=model, medium=model, writer=model, light=model)
    monkeypatch.setattr(research, "get_models", lambda _name=None: namespace)
    update = research.conduct_research(research_state(model_profile="quality", started_at=None))
    return update["analyst_stats"][0]


def test_rejected_api_key_fails_the_analyst_loudly(monkeypatch) -> None:
    rejected = {"error": ValueError("Error 401: Unauthorized: missing or invalid API key.")}
    monkeypatch.setattr(TavilySearch, "_run", lambda self, *args, **kwargs: rejected)
    stats = run_analyst_with_tool(monkeypatch, ToolCallingModel())
    assert stats["status"] == "failed"
    assert stats["stop_reason"] == "auth: tavily_search"


def test_tool_failures_are_counted_not_hidden(monkeypatch) -> None:
    from src.utils import tools as tool_module

    class WikipediaModel(ToolCallingModel):
        tool = "wikipedia"

    monkeypatch.setattr(
        tool_module.WikipediaEvidenceTool,
        "_run",
        lambda self, query, run_manager=None: "Wikipedia search failed: HTTP 429",
    )
    stats = run_analyst_with_tool(monkeypatch, WikipediaModel())
    assert stats["status"] == "completed"
    assert stats["tool_calls"] > 0
    assert stats["tool_errors"] == {"wikipedia": stats["tool_calls"]}


def test_tavily_api_errors_read_as_failures_not_evidence(monkeypatch) -> None:
    from src.utils.tools import auth_failure, tool_failure

    rejected = {"error": ValueError("Error 401: Unauthorized: missing or invalid API key.")}
    monkeypatch.setattr(TavilySearch, "_run", lambda self, *args, **kwargs: rejected)
    output = BoundedTavilySearch(max_results=3)._run(query="q")
    assert output.startswith("Tavily search failed:")
    assert tool_failure(output) and auth_failure("tavily_search", output)
    # A site refusing a scraper is not a credentials problem.
    blocked = "Webpage scrape failed: Client error '403 Forbidden'"
    assert tool_failure(blocked) and auth_failure("scrape_webpage", blocked) is None
    # An empty search is a normal outcome.
    assert tool_failure("No search results found for 'q'", status="error") is None


def test_finalize_report_records_uncovered_requirements_and_notes_tool_failures() -> None:
    stats = [
        {
            "analyst": "A",
            "status": "completed",
            "stop_reason": "max_passes",
            "duration_seconds": 1.0,
            "llm_calls": 3,
            "input_tokens": 10,
            "requirements_covered": ["Costs"],
            "tool_calls": 4,
            "tool_errors": {"wikipedia": 3},
        },
        {
            "analyst": "B",
            "status": "failed",
            "stop_reason": "auth: tavily_search",
            "duration_seconds": 1.0,
            "llm_calls": 1,
            "input_tokens": 5,
        },
    ]
    update = nodes.finalize_report(
        {
            "topic": "T",
            "requirements": ["Costs", "Safety"],
            "content": "## Insights" + chr(10) + "Body",
            "introduction": "# T",
            "conclusion": "## Conclusion",
            "analyst_stats": stats,
        }
    )
    report, run_stats = update["final_report"], update["run_stats"]
    assert run_stats["uncovered_requirements"] == ["Safety"]
    assert run_stats["tool_errors"] == {"wikipedia": 3}
    # The report writer states requirement gaps; an appended note could contradict the body.
    assert "Safety" not in report
    assert "3 of 4 research tool calls failed" in report
    assert "tavily_search rejected its API key" in report
    assert "research failed for B" in report
