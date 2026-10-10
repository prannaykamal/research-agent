"""Search floor, per-requirement verdicts, date stamping, personas and report cleanup."""

import time
from datetime import date
from types import SimpleNamespace

from langchain_core.messages import AIMessage, HumanMessage

import src.analyst_research as research
import src.utils.nodes as nodes
import src.utils.prompts as prompts
from src.utils.objects import (
    Analyst,
    Perspectives,
    QuestionRequirements,
    ResearchEvaluation,
    ResearchFindingBatch,
    ResearchPlan,
)
from test_guardrails import analyst, finding, research_state

NL = chr(10)
PINNED_DATE = "2031-02-03"
LONG_DESCRIPTION = " ".join(["word"] * 45)


class Recorder:
    """Answers every node's model call and records each system prompt it receives."""

    def __init__(self, prompts_seen: list, schema=None, evaluation=None) -> None:
        self.prompts_seen = prompts_seen
        self.schema = schema
        self.evaluation = evaluation

    def with_structured_output(self, schema, **_kwargs):
        return Recorder(self.prompts_seen, schema, self.evaluation)

    def bind_tools(self, _tools, **_kwargs):
        return self

    def invoke(self, messages, *args, **kwargs):
        self.prompts_seen.append(messages[0].content)
        parsed = {
            QuestionRequirements: lambda: QuestionRequirements(requirements=["Costs", "Safety"]),
            ResearchPlan: lambda: ResearchPlan(sub_questions=["Question one", "Question two"]),
            ResearchEvaluation: lambda: self.evaluation
            or ResearchEvaluation(is_complete=False, feedback="More."),
            ResearchFindingBatch: lambda: ResearchFindingBatch(findings=[]),
        }.get(self.schema)
        if self.schema is Perspectives:
            return Perspectives(
                analysts=[
                    Analyst(affiliation="A", name="N", role="R", description=LONG_DESCRIPTION)
                ]
            )
        if parsed is not None:
            return {"parsed": parsed(), "raw": AIMessage(content=""), "parsing_error": None}
        return AIMessage(content="Text")


def use_recorder(monkeypatch, evaluation=None) -> list:
    seen: list = []
    model = Recorder(seen, evaluation=evaluation)
    namespace = SimpleNamespace(heavy=model, medium=model, writer=model, light=model, panel=model)
    monkeypatch.setattr(research, "get_models", lambda _name=None: namespace)
    monkeypatch.setattr(nodes, "get_models", lambda _name=None: namespace)
    monkeypatch.setattr(prompts, "current_date", lambda: PINNED_DATE)
    monkeypatch.setattr(nodes, "current_date", lambda: PINNED_DATE)
    return seen


def owner(*requirements: str) -> Analyst:
    return analyst().model_copy(update={"requirements": list(requirements)})


# Today's date


def test_every_system_prompt_carries_todays_date(monkeypatch) -> None:
    seen = use_recorder(monkeypatch)
    nodes.create_analysts({"topic": "T", "max_analysts": 1})
    report_state = {"topic": "T", "sections": ["Section"], "content": "## Insights" + NL + "Body"}
    nodes.write_report(report_state)
    nodes.write_introduction(report_state)
    nodes.write_conclusion(report_state)
    state = research_state(analyst=owner("Costs"), loop_count=1, question_history=[])
    research.planner_node(state)
    research.researcher_node(state)
    research.evaluate_research(state)
    research.writer_node(state)

    assert len(seen) >= 9
    assert all(prompt.startswith(f"Today's date is {PINNED_DATE}.") for prompt in seen)
    # Forward-looking questions ask for status as of today.
    assert any(f"current status as of {PINNED_DATE}" in prompt for prompt in seen)


# Search floor


def test_researcher_nudge_names_the_requirements_still_lacking_evidence(monkeypatch) -> None:
    use_recorder(monkeypatch)
    costs = finding(1).model_copy(update={"requirement": "Costs"})
    state = research_state(
        analyst=owner("Costs", "Safety"),
        research_findings=[costs],
        tool_call_count=2,
        researcher_turns=1,
        messages=[HumanMessage(content="brief"), AIMessage(content="Done.")],
    )
    update = research.researcher_node(state)
    nudge, response = update["messages"]
    assert isinstance(nudge, HumanMessage) and isinstance(response, AIMessage)
    assert "2 of at least 4 searches" in nudge.content
    assert "Safety" in nudge.content and "Costs" not in nudge.content
    assert update["researcher_turns"] == 2


def test_no_nudge_on_a_normal_researcher_turn(monkeypatch) -> None:
    use_recorder(monkeypatch)
    update = research.researcher_node(research_state())
    assert len(update["messages"]) == 1


# Per-requirement verdicts


def verdict(status: str) -> ResearchEvaluation:
    return ResearchEvaluation(
        is_complete=True,
        feedback="Looks done.",
        requirement_status=[{"requirement": "costs", "status": status}],
    )


def test_a_thin_requirement_keeps_research_going(monkeypatch) -> None:
    use_recorder(monkeypatch, evaluation=verdict("thin"))
    costs = finding(1).model_copy(update={"requirement": "Costs"})
    update = research.evaluate_research(
        research_state(analyst=owner("Costs"), research_findings=[costs], loop_count=1)
    )
    assert update["evaluation"].is_complete is False
    assert "Costs" in update["feedback"]


def test_a_supported_requirement_lets_research_finish(monkeypatch) -> None:
    use_recorder(monkeypatch, evaluation=verdict("supported"))
    costs = finding(1).model_copy(update={"requirement": "Costs"})
    update = research.evaluate_research(
        research_state(analyst=owner("Costs"), research_findings=[costs], loop_count=1)
    )
    assert update["evaluation"].is_complete is True


def test_thin_verdicts_never_outlast_the_budget() -> None:
    expired = research_state(
        analyst=owner("Costs"), evaluation=verdict("thin"), started_at=time.time() - 10_000, loop_count=1
    )
    assert research.evaluate_research(expired) == {}
    assert research.route_evaluation(expired) == "writer_node"


# Coverage notes


def test_coverage_counts_evaluator_verdicts_as_well_as_finding_labels() -> None:
    science = "Scientific objectives and robotic surface exploration"
    timeline = "Timeline of major missions"
    lunar = owner(timeline, science)
    # Evidence filed under the timeline requirement, but the evaluator saw it covers science too.
    mislabelled = [finding(1).model_copy(update={"requirement": timeline})]
    evaluation = ResearchEvaluation(
        is_complete=False,
        feedback="",
        requirement_status=[{"requirement": science, "status": "thin"}],
    )
    assert research._covered_for_stats(lunar, mislabelled, evaluation) == [timeline, science]
    assert research._covered_for_stats(lunar, mislabelled, None) == [timeline]


# Personas


class PersonaModel:
    """Short personas first; detailed ones on the corrective retry."""

    def __init__(self, retry_description: str) -> None:
        self.retry_description = retry_description
        self.persona_calls = 0
        self.schema = None

    def with_structured_output(self, schema, **_kwargs):
        self.schema = schema
        return self

    def invoke(self, _messages):
        if self.schema is QuestionRequirements:
            parsed = QuestionRequirements(requirements=["Costs"])
            return {"parsed": parsed, "raw": AIMessage(content=""), "parsing_error": None}
        self.persona_calls += 1
        description = "Short." if self.persona_calls == 1 else self.retry_description
        return Perspectives(
            analysts=[Analyst(affiliation="A", name="N", role="R", description=description)]
        )


def use_panel(monkeypatch, model) -> None:
    namespace = SimpleNamespace(heavy=None, medium=None, writer=None, light=None, panel=model)
    monkeypatch.setattr(nodes, "get_models", lambda _name=None: namespace)


def test_short_personas_get_one_corrective_retry(monkeypatch) -> None:
    model = PersonaModel(retry_description=LONG_DESCRIPTION)
    use_panel(monkeypatch, model)
    update = nodes.create_analysts({"topic": "T", "max_analysts": 1})
    assert model.persona_calls == 2
    assert update["analysts"][0].description == LONG_DESCRIPTION


def test_a_retry_that_is_no_better_keeps_the_original_panel(monkeypatch) -> None:
    model = PersonaModel(retry_description="Brief.")  # same length as the original
    use_panel(monkeypatch, model)
    update = nodes.create_analysts({"topic": "T", "max_analysts": 1})
    assert model.persona_calls == 2
    assert update["analysts"][0].description == "Short."


# Report cleanup


def test_finalize_lists_every_inline_url_and_leaves_no_doubled_rule() -> None:
    content = NL.join(
        [
            "## Insights",
            "Body cites [a](https://a.org/1) and https://b.org/2.",
            "",
            "---",
            "## Sources",
            "- [A](https://a.org/1)",
        ]
    )
    update = nodes.finalize_report(
        {
            "topic": "T",
            "content": content,
            "introduction": "# T",
            "conclusion": "## Conclusion",
            "analyst_stats": [],
        }
    )
    report = update["final_report"]
    sources = report.split("## Sources", 1)[1]
    assert "https://a.org/1" in sources and "https://b.org/2" in sources
    assert sources.count("https://a.org/1") == 1
    assert report.count("---") == 2  # one rule after the introduction, one before the conclusion


# Inline citations survive synthesis

CITED_SECTION = "Costs fell [https://a.org/1]. Safety improved [https://b.org/2]. More [https://c.org/3]."


class ReportModel:
    """A report writer that returns fixed text and records each prompt it receives."""

    def __init__(self, text: str) -> None:
        self.text = text
        self.calls: list = []

    def invoke(self, messages, *args, **kwargs):
        self.calls.append(messages)
        return AIMessage(content=self.text)


def use_report_models(monkeypatch, draft: str, retry: str) -> tuple[ReportModel, ReportModel]:
    medium, panel = ReportModel(draft), ReportModel(retry)
    namespace = SimpleNamespace(heavy=None, medium=medium, writer=None, light=None, panel=panel)
    monkeypatch.setattr(nodes, "get_models", lambda _name=None: namespace)
    return medium, panel


def report_state(**overrides) -> dict:
    state = {"topic": "T", "requirements": ["Costs", "Safety"], "sections": [CITED_SECTION]}
    state.update(overrides)
    return state


def test_inline_urls_ignore_the_sources_list() -> None:
    text = NL.join(["Body [a](https://a.org/1).", "", "## Sources", "- https://z.org/9"])
    assert nodes.inline_urls(text) == ["https://a.org/1"]


def test_a_report_without_citations_is_rewritten_by_the_panel_model(monkeypatch) -> None:
    cited = "## Insights" + NL + "- Costs fell [a](https://a.org/1) and [b](https://b.org/2), [c](https://c.org/3)."
    medium, panel = use_report_models(monkeypatch, draft="## Insights" + NL + "- Costs fell.", retry=cited)
    update = nodes.write_report(report_state())
    assert len(medium.calls) == 1 and len(panel.calls) == 1
    assert "0 inline citations" in panel.calls[0][-1].content
    assert update["content"] == cited


def test_a_retry_that_cites_no_more_keeps_the_original(monkeypatch) -> None:
    _, panel = use_report_models(monkeypatch, draft="Uncited.", retry="Still uncited.")
    assert nodes.write_report(report_state())["content"] == "Uncited."
    assert len(panel.calls) == 1


def test_a_cited_report_is_not_retried(monkeypatch) -> None:
    cited = "Costs fell [a](https://a.org/1), [b](https://b.org/2) and [c](https://c.org/3)."
    _, panel = use_report_models(monkeypatch, draft=cited, retry="unused")
    assert nodes.write_report(report_state())["content"] == cited
    assert panel.calls == []


def test_sections_without_citations_never_trigger_a_retry(monkeypatch) -> None:
    _, panel = use_report_models(monkeypatch, draft="Uncited.", retry="unused")
    nodes.write_report(report_state(sections=["No links here."]))
    assert panel.calls == []


# Requirement gaps are stated by the report writer


def test_the_report_writer_is_told_which_requirements_lack_evidence(monkeypatch) -> None:
    cited = "[a](https://a.org/1) [b](https://b.org/2) [c](https://c.org/3)"
    medium, _ = use_report_models(monkeypatch, draft=cited, retry="unused")
    nodes.write_report(report_state(analyst_stats=[{"requirements_covered": ["Costs"]}]))
    prompt = medium.calls[0][0].content
    assert "possibly lacking evidence: Safety." in prompt


def test_no_flagged_requirements_reads_none(monkeypatch) -> None:
    cited = "[a](https://a.org/1) [b](https://b.org/2) [c](https://c.org/3)"
    medium, _ = use_report_models(monkeypatch, draft=cited, retry="unused")
    nodes.write_report(report_state(analyst_stats=[{"requirements_covered": ["Costs", "Safety"]}]))
    assert "possibly lacking evidence: none." in medium.calls[0][0].content


# Dated evidence


def test_evidence_age_handles_partial_dates_and_garbage() -> None:
    from src.utils.guardrails import evidence_age_days, is_recent

    assert evidence_age_days("2026-10-09", "2026-10-10") == 1
    assert evidence_age_days("2026-02", "2026-10-10") == (date(2026, 10, 10) - date(2026, 2, 1)).days
    assert evidence_age_days("2024", "2026-10-10") == (date(2026, 10, 10) - date(2024, 1, 1)).days
    assert evidence_age_days("", "2026-10-10") is None
    assert evidence_age_days("June 2024", "2026-10-10") is None
    assert evidence_age_days("2024-13-40", "2026-10-10") is None
    assert is_recent("2026-01-15", "2026-10-10") is True
    assert is_recent("2024", "2026-10-10") is False
    assert is_recent("", "2026-10-10") is None


def test_findings_reach_the_evaluator_and_writer_marked_recent_or_not(monkeypatch) -> None:
    monkeypatch.setattr(prompts, "current_date", lambda: "2026-10-10")
    fresh = finding(1).model_copy(update={"source_date": "2026-08"})
    stale = finding(2).model_copy(update={"source_date": "2023-05-01"})
    undated = finding(3)
    annotated = research._annotated_findings([fresh, stale, undated])
    assert [item["recent"] for item in annotated] == [True, False, None]
    assert all("source_tier" in item for item in annotated)


def test_pubmed_results_carry_their_publication_date(monkeypatch) -> None:
    import json

    from src.utils.tools import PubMedEvidenceTool

    tool = PubMedEvidenceTool()
    article = {"uid": "123", "Title": "A trial", "Published": "2025-03-25", "Summary": "Results."}
    monkeypatch.setattr(type(tool.api_wrapper), "load", lambda self, query: [article])
    [payload] = json.loads(tool._run("gepotidacin"))
    assert payload["published"] == "2025-03-25"


# Report cleanup


def test_empty_evidence_limits_are_removed() -> None:
    body = NL.join(["Body text.", "", "## Evidence limits", "- The provided text mentions no specific evidence limits."])
    assert "Evidence limits" not in nodes._tidy_evidence_limits(body)


def test_limits_that_repeat_the_body_are_dropped_and_real_limits_kept() -> None:
    repeated = "This report found no evidence regarding municipal zoning approvals."
    real = "- Cost figures come from a single 2026 industry estimate."
    body = NL.join(["Body. " + repeated, "", "## Evidence limits", "- " + repeated, real, "", "## Next", "Tail."])
    tidied = nodes._tidy_evidence_limits(body)
    limits = tidied.split("## Evidence limits", 1)[1].split("## Next", 1)[0]
    assert real in limits and repeated not in limits
    assert tidied.endswith("## Next" + NL + "Tail.")


def test_finalize_drops_a_bare_report_heading() -> None:
    update = nodes.finalize_report(
        {
            "topic": "T",
            "content": NL.join(["- Insight [a](https://a.org/1)", "", "## Report", "### Part", "Body."]),
            "introduction": "# T",
            "conclusion": "## Conclusion",
            "analyst_stats": [],
        }
    )
    assert "## Report" not in update["final_report"] and "### Part" in update["final_report"]


# Per-profile analyst caps


def test_deep_research_allows_at_most_five_analysts(monkeypatch) -> None:
    import pytest

    use_recorder(monkeypatch)
    with pytest.raises(ValueError, match="between 1 and 5"):
        nodes.create_analysts({"topic": "T", "max_analysts": 6, "model_profile": "quality"})
    assert nodes.create_analysts({"topic": "T", "max_analysts": 5, "model_profile": "quality"})["analysts"]


def test_quick_research_still_allows_ten_analysts() -> None:
    from src.utils.guardrails import validate_max_analysts
    from src.utils.profiles import get_profile

    assert validate_max_analysts(10, get_profile("fast").max_analysts) == 10
    assert get_profile("quality").max_analysts == 5


# Topic check


class TopicModel:
    """Answers the question split with a fixed verdict and counts persona calls."""

    def __init__(self, verdict) -> None:
        self.verdict = verdict
        self.schema = None
        self.persona_calls = 0

    def with_structured_output(self, schema, **_kwargs):
        self.schema = schema
        return self

    def invoke(self, _messages):
        if self.schema is QuestionRequirements:
            return {"parsed": self.verdict, "raw": AIMessage(content=""), "parsing_error": None}
        self.persona_calls += 1
        return Perspectives(
            analysts=[Analyst(affiliation="A", name="N", role="R", description=LONG_DESCRIPTION)]
        )


def test_a_topic_with_nothing_to_research_is_rejected_before_any_persona(monkeypatch) -> None:
    import pytest

    from src.utils.guardrails import UnresearchableTopicError

    model = TopicModel(QuestionRequirements(researchable=False, reason="greeting"))
    use_panel(monkeypatch, model)
    with pytest.raises(UnresearchableTopicError, match='"hi" does not name a subject'):
        nodes.create_analysts({"topic": "hi", "max_analysts": 1})
    assert model.persona_calls == 0


def test_researchable_topics_and_unparsable_verdicts_proceed(monkeypatch) -> None:
    for verdict in (QuestionRequirements(requirements=["Costs"]), None):
        model = TopicModel(verdict)
        use_panel(monkeypatch, model)
        update = nodes.create_analysts({"topic": "Blockchain", "max_analysts": 1})
        assert model.persona_calls == 1 and update["analysts"]


def test_a_panel_revision_never_rechecks_the_topic(monkeypatch) -> None:
    model = TopicModel(QuestionRequirements(researchable=False))
    use_panel(monkeypatch, model)
    update = nodes.create_analysts({"topic": "hi", "max_analysts": 1, "requirements": ["Costs"]})
    assert update["requirements"] == ["Costs"] and model.persona_calls == 1


def test_a_long_rejected_topic_is_shortened_in_the_message() -> None:
    from src.utils.guardrails import unresearchable_topic_message

    message = unresearchable_topic_message("x" * 200)
    assert message.startswith('"' + "x" * 59 + '…"')
