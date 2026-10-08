from types import SimpleNamespace

import httpx
from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage, ToolMessage

import src.analyst_research as research
from src.utils.objects import Analyst, ResearchFinding, ResearchFindingBatch, ResearchPlan


class StructuredResult:
    def __init__(self, result):
        self.result = result

    def invoke(self, _messages):
        return self.result


class PlannerModel:
    def with_structured_output(self, schema, **_kwargs):
        assert schema is ResearchPlan
        return StructuredResult(ResearchPlan(sub_questions=["Question one", "Question two", "Question three"]))


def use_models(monkeypatch, **models) -> None:
    """Replace the profile's models with stubs for the analyst nodes."""
    namespace = SimpleNamespace(**{"heavy": None, "medium": None, "writer": None, "light": None, **models})
    monkeypatch.setattr(research, "get_models", lambda _name=None: namespace)


def research_state():
    return {
        "topic": "Topic",
        "analyst": Analyst(affiliation="A", name="N", role="R", description="D"),
        "sub_questions": [],
        "question_history": [],
        "research_findings": [],
        "feedback": "",
        "loop_count": 0,
        "draft": "",
        "messages": [HumanMessage(content="old raw transcript")],
        "tool_call_count": 5,
        "budget_exhausted": True,
        "evaluation": None,
    }


def test_planner_resets_only_temporary_execution_state(monkeypatch) -> None:
    use_models(monkeypatch, heavy=PlannerModel())
    update = research.planner_node(research_state())

    assert update["question_history"] == ["Question one", "Question two", "Question three"]
    assert update["tool_call_count"] == 0
    assert update["researcher_turns"] == 0
    assert update["budget_exhausted"] is False
    assert isinstance(update["messages"][0], RemoveMessage)
    assert isinstance(update["messages"][1], HumanMessage)


def test_extract_findings_runs_once_and_increments_one_pass(monkeypatch) -> None:
    finding = ResearchFinding(
        sub_question="Question one",
        claim="Supported claim",
        source_title="Source",
        source_url="https://example.com/evidence",
        excerpt="Supporting excerpt",
        source_type="web",
    )

    class ExtractionModel:
        def with_structured_output(self, schema, *, include_raw=False):
            assert schema is ResearchFindingBatch
            assert include_raw is True
            return StructuredResult(
                {"parsed": ResearchFindingBatch(findings=[finding]), "parsing_error": None}
            )

    state = research_state()
    state["messages"] = [ToolMessage(content="tool result", tool_call_id="call-1")]
    use_models(monkeypatch, medium=ExtractionModel())
    update = research.extract_findings(state)

    assert update["loop_count"] == 1
    assert update["research_findings"] == [finding]


def test_malformed_extraction_recovers_valid_findings(monkeypatch, caplog) -> None:
    valid = {
        "sub_question": "Question one",
        "claim": "Supported claim",
        "source_title": "Source",
        "source_url": "https://example.com/evidence",
        "excerpt": "Supporting excerpt",
        "source_type": "web",
    }
    malformed = {"sub_question": "Question one", "claim": "Missing source fields"}

    class MalformedExtractionModel:
        def with_structured_output(self, schema, *, include_raw=False):
            assert schema is ResearchFindingBatch
            assert include_raw is True
            raw = AIMessage(content=research._json({"findings": [valid, malformed]}))
            return StructuredResult(
                {"parsed": None, "parsing_error": ValueError("partial finding"), "raw": raw}
            )

    state = research_state()
    state["messages"] = [ToolMessage(content="truncated tool result", tool_call_id="call-1")]
    use_models(monkeypatch, medium=MalformedExtractionModel())

    update = research.extract_findings(state)

    assert update["loop_count"] == 1
    assert update["research_findings"] == [ResearchFinding.model_validate(valid)]
    assert "recovered=1 discarded=1" in caplog.text
    assert "raw_output=" in caplog.text

def test_writer_never_binds_tools(monkeypatch) -> None:
    from langchain_core.messages import AIMessage

    class WriterModel:
        def bind_tools(self, _tools):
            raise AssertionError("writer must not bind tools")

        def invoke(self, _messages):
            return AIMessage(content=[{"type": "text", "text": "Evidence-grounded draft"}])

    state = research_state()
    state["research_findings"] = [
        ResearchFinding(
            sub_question="Question one",
            claim="Supported claim",
            source_title="Source",
            source_url="https://example.com/evidence",
            excerpt="Supporting excerpt",
            source_type="web",
        )
    ]
    use_models(monkeypatch, writer=WriterModel())

    update = research.writer_node(state)

    assert update["draft"] == "Evidence-grounded draft"
    assert update["stop_reason"] == "max_passes"


def test_extract_findings_retries_timeout_then_continues(monkeypatch, caplog) -> None:
    class TimeoutModel:
        def __init__(self) -> None:
            self.calls = 0

        def with_structured_output(self, schema, *, include_raw=False):
            assert schema is ResearchFindingBatch
            assert include_raw is True
            return self

        def invoke(self, _messages):
            self.calls += 1
            raise httpx.ReadTimeout("transient provider timeout")

    model = TimeoutModel()
    state = research_state()
    state["messages"] = [ToolMessage(content="tool result", tool_call_id="call-1")]
    use_models(monkeypatch, medium=model)
    monkeypatch.setattr(research.time, "sleep", lambda _seconds: None)

    update = research.extract_findings(state)

    assert model.calls == research.EXTRACTION_TIMEOUT_ATTEMPTS
    assert update == {"loop_count": 1}
    assert "timed out on attempt 1/2; retrying" in caplog.text
    assert "continuing with accumulated evidence" in caplog.text