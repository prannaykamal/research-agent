from langchain_core.messages import HumanMessage, RemoveMessage, ToolMessage

import src.analyst_research as research
from src.utils.objects import Analyst, ResearchFinding, ResearchFindingBatch, ResearchPlan


class StructuredResult:
    def __init__(self, result):
        self.result = result

    def invoke(self, _messages):
        return self.result


class PlannerModel:
    def with_structured_output(self, schema):
        assert schema is ResearchPlan
        return StructuredResult(ResearchPlan(sub_questions=["Question one", "Question two", "Question three"]))


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
    monkeypatch.setattr(research, "llm", PlannerModel())
    update = research.planner_node(research_state())

    assert update["question_history"] == ["Question one", "Question two", "Question three"]
    assert update["tool_call_count"] == 0
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
        def with_structured_output(self, schema):
            assert schema is ResearchFindingBatch
            return StructuredResult(ResearchFindingBatch(findings=[finding]))

    state = research_state()
    state["messages"] = [ToolMessage(content="tool result", tool_call_id="call-1")]
    monkeypatch.setattr(research, "llm", ExtractionModel())
    update = research.extract_findings(state)

    assert update["loop_count"] == 1
    assert update["research_findings"] == [finding]


def test_writer_never_binds_tools(monkeypatch) -> None:
    from langchain_core.messages import AIMessage

    class WriterModel:
        def bind_tools(self, _tools):
            raise AssertionError("writer must not bind tools")

        def invoke(self, _messages):
            return AIMessage(content="Evidence-grounded draft")

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
    monkeypatch.setattr(research, "llm", WriterModel())

    assert research.writer_node(state) == {"draft": "Evidence-grounded draft"}
