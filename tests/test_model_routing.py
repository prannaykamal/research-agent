import pytest

from src.utils.models import HEAVY_TIER, LIGHT_TIER, MEDIUM_TIER, TokenPerMinuteLimiter
from src.utils.prompts import evaluator_instructions, planner_instructions


def test_screenshot_model_assignments_are_configured() -> None:
    assert HEAVY_TIER.model == "gemini-3.1-pro-preview"
    assert HEAVY_TIER.thinking_level == "high"
    assert MEDIUM_TIER.model == "gemini-3.8-flash"
    assert LIGHT_TIER.model == "gemini-3.8-flash"


def test_limiter_rejects_a_request_that_would_exceed_its_tpm_budget() -> None:
    limiter = TokenPerMinuteLimiter(tokens_per_minute=100, safety_margin=0.8)
    with pytest.raises(ValueError, match="Estimated request token use"):
        limiter.acquire(81)


def test_prompts_use_requested_planning_and_evaluation_language() -> None:
    assert "two or three" in planner_instructions
    assert "Is the evidence sufficient for the analyst's assigned objectives?" in evaluator_instructions
