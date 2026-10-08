"""End-to-end check that 1 and 10 analysts take the same time per analyst.

Stub models add fixed latency and pass through the real quota limiters and
analyst slots, so the test exercises the production concurrency path without
network calls.
"""

import asyncio
import time
import uuid
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

import src.analyst_research as research
import src.deep_agent as deep_agent
import src.utils.nodes as nodes
from src.utils.models import RateLimitedRunnable
from src.utils.objects import (
    Analyst,
    Perspectives,
    ResearchEvaluation,
    ResearchFindingBatch,
    ResearchPlan,
)

LATENCY_SECONDS = 0.2


class SlowModel:
    """Thread-safe stub that answers every node's call after a fixed delay."""

    def __init__(self, schema=None, include_raw: bool = False) -> None:
        self.schema = schema
        self.include_raw = include_raw

    def with_structured_output(self, schema, include_raw: bool = False, **_kwargs):
        return SlowModel(schema, include_raw)

    def bind_tools(self, _tools, **_kwargs):
        return self

    def invoke(self, _messages, *args, **kwargs):
        time.sleep(LATENCY_SECONDS)
        if self.schema is Perspectives:
            return Perspectives(
                analysts=[
                    Analyst(affiliation=f"Org {index}", name=f"Analyst {index}", role="Role", description="D")
                    for index in range(10)
                ]
            )
        parsed = {
            ResearchPlan: lambda: ResearchPlan(sub_questions=["Question one", "Question two"]),
            ResearchEvaluation: lambda: ResearchEvaluation(is_complete=True, feedback="Done"),
            ResearchFindingBatch: lambda: ResearchFindingBatch(findings=[]),
        }.get(self.schema)
        if parsed is not None:
            return {"parsed": parsed(), "raw": AIMessage(content=""), "parsing_error": None}
        return AIMessage(content="Section text")


@pytest.fixture
def stub_models(monkeypatch):
    run_id = uuid.uuid4().hex  # fresh quota windows for each test
    namespace = SimpleNamespace(
        heavy=RateLimitedRunnable(SlowModel(), f"stub-heavy-{run_id}"),
        medium=RateLimitedRunnable(SlowModel(), f"stub-medium-{run_id}"),
        writer=RateLimitedRunnable(SlowModel(), f"stub-medium-{run_id}"),
        light=RateLimitedRunnable(SlowModel(), f"stub-light-{run_id}"),
    )
    monkeypatch.setattr(research, "get_models", lambda _name=None: namespace)
    monkeypatch.setattr(nodes, "get_models", lambda _name=None: namespace)


def run_report(analysts: int, *, use_async: bool = False) -> dict:
    graph = deep_agent.builder.compile(checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": uuid.uuid4().hex}}
    request = {"topic": "Grid batteries", "max_analysts": analysts, "model_profile": "fast"}
    if use_async:
        async def run() -> dict:
            await graph.ainvoke(request, config)
            return await graph.ainvoke(Command(resume="approved"), config)

        return asyncio.run(run())
    graph.invoke(request, config)
    return graph.invoke(Command(resume="approved"), config)


@pytest.mark.parametrize("use_async", [False, True], ids=["sync", "async-server-path"])
def test_ten_analysts_take_as_long_as_one(stub_models, use_async) -> None:
    started = time.perf_counter()
    single = run_report(1, use_async=use_async)
    single_wall = time.perf_counter() - started

    started = time.perf_counter()
    panel = run_report(10, use_async=use_async)
    panel_wall = time.perf_counter() - started

    one, ten = single["run_stats"], panel["run_stats"]
    assert one["analysts"] == 1 and ten["analysts"] == 10
    assert ten["completed"] == 10 and ten["failed"] == []
    assert len({entry["slot"] for entry in ten["per_analyst"]}) == 10
    assert ten["mean_analyst_seconds"] == pytest.approx(one["mean_analyst_seconds"], rel=0.25)
    # Sequential execution would take ~10x longer; parallel stays close to 1x.
    assert panel_wall < single_wall * 1.5
