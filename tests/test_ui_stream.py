"""Contract and latency checks for the event stream the UI consumes."""

import time
import uuid

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command
from starlette.testclient import TestClient

import src.deep_agent as deep_agent
from src.webapp import app as web_app
from test_parallelism import stub_models  # noqa: F401  (pytest fixture)

UI_STREAM = {"stream_mode": ["updates", "custom"], "subgraphs": True}


def approved_run(graph, analysts: int, stream: bool) -> list:
    config = {"configurable": {"thread_id": uuid.uuid4().hex}}
    graph.invoke({"topic": "Topic", "max_analysts": analysts, "model_profile": "fast"}, config)
    if stream:
        return list(graph.stream(Command(resume="approved"), config, **UI_STREAM))
    graph.invoke(Command(resume="approved"), config)
    return []


def test_each_analyst_streams_under_its_own_namespace(stub_models) -> None:
    graph = deep_agent.builder.compile(checkpointer=InMemorySaver())
    events = approved_run(graph, analysts=3, stream=True)

    by_namespace: dict[str, list] = {}
    for namespace, mode, chunk in events:
        if namespace:
            by_namespace.setdefault(namespace[0], []).append((mode, chunk))
    assert len(by_namespace) == 3

    indexes = set()
    for namespace, items in by_namespace.items():
        assert namespace.startswith("conduct_research:")
        first_mode, first = items[0]
        assert first_mode == "custom" and first["type"] == "analyst_started"
        assert set(first["limits"]) >= {"max_passes", "max_tool_calls_per_pass", "deadline_seconds", "allowed_tools"}
        indexes.add(first["analyst_index"])
        updated = {node for mode, chunk in items if mode == "updates" for node in chunk}
        assert {"planner_node", "researcher_node", "extract_findings", "writer_node"} <= updated
        steps = [chunk["node"] for mode, chunk in items if mode == "custom" and chunk["type"] == "analyst_step"]
        assert steps[0] == "planner_node" and steps[-1] == "writer_node"
        assert all(chunk.get("analyst_index") == first["analyst_index"] for mode, chunk in items if mode == "custom")
    assert indexes == {0, 1, 2}

    top_level = [chunk for namespace, mode, chunk in events if not namespace and mode == "updates"]
    stats = [entry for chunk in top_level for entry in chunk.get("conduct_research", {}).get("analyst_stats", [])]
    assert sorted(entry["analyst_index"] for entry in stats) == [0, 1, 2]
    assert any("finalize_report" in chunk for chunk in top_level)


def test_streaming_to_the_ui_does_not_slow_the_run(stub_models) -> None:
    graph = deep_agent.builder.compile(checkpointer=InMemorySaver())
    approved_run(graph, analysts=5, stream=False)  # warm-up

    started = time.perf_counter()
    approved_run(graph, analysts=5, stream=False)
    unobserved = time.perf_counter() - started

    started = time.perf_counter()
    approved_run(graph, analysts=5, stream=True)
    observed = time.perf_counter() - started

    assert observed <= unobserved * 1.15 + 0.15, (observed, unobserved)


@pytest.fixture
def web() -> TestClient:
    return TestClient(web_app)


def test_ui_is_served_at_app(web: TestClient) -> None:
    redirect = web.get("/app", follow_redirects=False)
    assert redirect.status_code in (301, 302, 307, 308)
    assert redirect.headers["location"] == "/app/"
    page = web.get("/app/")
    assert page.status_code == 200
    assert "js/app.js" in page.text
    assert web.get("/app/js/app.js").status_code == 200


def test_profiles_endpoint_describes_both_profiles(web: TestClient) -> None:
    payload = web.get("/app/profiles.json").json()
    assert payload["default"] == "quality"
    assert (payload["min_analysts"], payload["max_analysts"]) == (1, 10)
    assert set(payload["profiles"]) == {"quality", "fast"}
    fast = payload["profiles"]["fast"]
    assert "pubmed" in fast["allowed_tools"] and "scrape_webpage" not in fast["allowed_tools"]
    assert fast["deadline_seconds"] == 150
