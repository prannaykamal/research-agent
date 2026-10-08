import threading
import time

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool
from pydantic import BaseModel

import src.utils.models as models
from src.utils.models import (
    AnalystSlotPool,
    RateLimitedRunnable,
    RequestPerMinuteLimiter,
    TokenPerMinuteLimiter,
    UsageMeter,
    analyst_context,
    global_quota,
    is_transient_error,
)
from src.utils.profiles import PROFILES, capacity_warnings, get_models, get_profile
from src.utils.prompts import evaluator_instructions, planner_instructions


def test_quality_profile_keeps_current_model_assignments() -> None:
    quality = get_profile("quality")
    assert quality.heavy.model == "gemini-3.1-pro-preview"
    assert quality.heavy.thinking_level == "high"
    assert quality.heavy.max_output_tokens >= 8_192
    assert quality.medium.model == "gemini-3.8-flash"
    assert quality.light.model == "gemini-3.8-flash"


def test_fast_profile_uses_only_flash_and_flash_lite() -> None:
    fast = get_profile("fast")
    assert fast.heavy.model == "gemini-3.8-flash"
    assert {fast.medium.model, fast.writer.model, fast.light.model} == {"gemini-3.5-flash-lite"}
    assert "scrape_webpage" not in fast.allowed_tools
    quality = get_profile("quality")
    assert fast.analyst_deadline_seconds < quality.analyst_deadline_seconds
    assert fast.max_research_loops < quality.max_research_loops


def test_default_and_unknown_profiles() -> None:
    assert get_profile(None).name == "quality"
    with pytest.raises(ValueError, match="model_profile must be one of"):
        get_profile("turbo")


def test_profile_models_are_built_once() -> None:
    assert get_models("fast") is get_models("fast")
    assert get_models("fast").medium.model_id == "gemini-3.5-flash-lite"


def test_default_tier1_quotas_fit_ten_parallel_analysts() -> None:
    for profile in PROFILES.values():
        assert capacity_warnings(profile) == []


def test_capacity_warning_names_the_needed_quota(monkeypatch) -> None:
    monkeypatch.setenv("GEMINI_QUOTA_GEMINI_3_8_FLASH_TPM", "10000")
    warnings = capacity_warnings(get_profile("quality"))
    assert warnings and "gemini-3.8-flash" in warnings[0] and "TPM" in warnings[0]


def test_limiter_rejects_a_request_that_would_exceed_its_tpm_budget() -> None:
    limiter = TokenPerMinuteLimiter(tokens_per_minute=100, safety_margin=0.8)
    with pytest.raises(ValueError, match="Estimated request token use"):
        limiter.acquire(81)


def test_non_strict_limiter_admits_an_oversized_request_into_an_empty_window() -> None:
    limiter = TokenPerMinuteLimiter(tokens_per_minute=100, strict=False)
    assert limiter.acquire(500)[1] == 500


def test_rpm_limiter_blocks_the_request_over_budget(monkeypatch) -> None:
    sleeps: list[float] = []

    def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)
        raise TimeoutError

    monkeypatch.setattr(models.time, "sleep", fake_sleep)
    limiter = RequestPerMinuteLimiter(requests_per_minute=2, safety_margin=1.0)
    limiter.acquire(1)
    limiter.acquire(1)
    with pytest.raises(TimeoutError):
        limiter.acquire(1)
    assert sleeps and sleeps[0] > 0


def test_tiers_sharing_a_model_share_one_quota() -> None:
    quality = get_models("quality")
    assert quality.medium.model_id == quality.light.model_id
    assert global_quota(quality.medium.model_id) is global_quota(quality.light.model_id)


class EchoModel:
    def __init__(self, usage: int | None = None) -> None:
        self.usage = usage
        self.calls = 0

    def invoke(self, _messages, *args, **kwargs):
        self.calls += 1
        message = AIMessage(content="ok")
        if self.usage is not None:
            message.usage_metadata = {
                "input_tokens": self.usage,
                "output_tokens": 1,
                "total_tokens": self.usage + 1,
            }
        return message

    def bind_tools(self, _tools, *args, **kwargs):
        return self

    def with_structured_output(self, _schema, *args, **kwargs):
        return self


@tool
def lookup(query: str) -> str:
    """Look up a fact about the query in a large reference corpus."""
    return query


class Answer(BaseModel):
    answer: str


def test_reservation_is_input_only_and_counts_bound_schemas() -> None:
    runnable = RateLimitedRunnable(EchoModel(), "test-model-estimate")
    tools_bound = runnable.bind_tools([lookup])
    structured = runnable.with_structured_output(Answer)

    assert runnable._extra_tokens == 0
    assert tools_bound._extra_tokens > 0
    assert structured._extra_tokens > 0


def test_actual_usage_replaces_the_estimate_and_is_metered() -> None:
    model_id = "test-model-usage"
    runnable = RateLimitedRunnable(EchoModel(usage=42), model_id)
    meter = UsageMeter()
    with analyst_context(None, meter):
        runnable.invoke([HumanMessage(content="x" * 4_000)])

    assert meter.calls == 1
    assert meter.input_tokens == 42
    assert global_quota(model_id).tokens._events[-1][1] == 42


def test_transient_errors_are_retried_once(monkeypatch) -> None:
    class Flaky(EchoModel):
        def invoke(self, *args, **kwargs):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("429 RESOURCE_EXHAUSTED")
            return AIMessage(content="ok")

    monkeypatch.setattr(models.time, "sleep", lambda _seconds: None)
    model = Flaky()
    RateLimitedRunnable(model, "test-model-retry").invoke([HumanMessage(content="hi")])
    assert model.calls == 2


def test_non_transient_errors_are_not_retried() -> None:
    class Broken(EchoModel):
        def invoke(self, *args, **kwargs):
            self.calls += 1
            raise ValueError("bad request")

    model = Broken()
    with pytest.raises(ValueError):
        RateLimitedRunnable(model, "test-model-broken").invoke([HumanMessage(content="hi")])
    assert model.calls == 1
    assert not is_transient_error(TimeoutError("read timeout"))


def test_analyst_slots_cap_concurrency_and_isolate_budgets() -> None:
    pool = AnalystSlotPool(2)
    with pool.slot() as first, pool.slot() as second:
        assert first is not second
        assert first.quota("gemini-3.8-flash") is not second.quota("gemini-3.8-flash")
        acquired = threading.Event()

        def take_third() -> None:
            with pool.slot():
                acquired.set()

        waiter = threading.Thread(target=take_third)
        waiter.start()
        time.sleep(0.05)
        assert not acquired.is_set()
    waiter.join(timeout=1)
    assert acquired.is_set()


def test_slot_share_is_a_tenth_of_the_global_budget() -> None:
    pool = AnalystSlotPool(10)
    with pool.slot() as slot:
        share = slot.quota("gemini-3.8-flash").tokens.budget
    assert share * 10 <= global_quota("gemini-3.8-flash").tokens.budget


def test_prompts_use_requested_planning_and_evaluation_language() -> None:
    assert "two or three" in planner_instructions
    assert "Is the evidence sufficient for the analyst's assigned objectives?" in evaluator_instructions
