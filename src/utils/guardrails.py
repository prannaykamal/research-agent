"""Deterministic system-wide and per-analyst guardrails.

Profile-specific limits (passes, tool calls, deadlines, budgets) live on
``ResearchProfile``; the constants here apply to every profile.
"""

import time
from typing import Any, Mapping

# System-wide limits.
MIN_ANALYSTS = 1
MAX_ANALYSTS = 10
MAX_CONCURRENT_ANALYSTS = 10
MAX_ANALYST_REVISIONS = 3

# Evidence limits per analyst.
MAX_FINDINGS_PER_PASS = 4
MAX_FINDINGS_PER_ANALYST = 12
MAX_CLAIM_CHARS = 300
MAX_EXCERPT_CHARS = 500

# Synthesis input bound: every analyst section is clipped, so ten sections
# always fit a single report request.
MAX_SECTION_CHARS = 8_000
MAX_SYNTHESIS_INPUT_CHARS = MAX_ANALYSTS * MAX_SECTION_CHARS

TRUNCATION_MARKER = " […]"


def validate_max_analysts(value: Any) -> int:
    """Return a valid analyst count or raise a clear error for the caller."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(
            f"max_analysts must be an integer between {MIN_ANALYSTS} and "
            f"{MAX_ANALYSTS}; got {value!r}."
        )
    if not MIN_ANALYSTS <= value <= MAX_ANALYSTS:
        raise ValueError(
            f"max_analysts must be between {MIN_ANALYSTS} and {MAX_ANALYSTS}; got {value}."
        )
    return value


def clip_text(text: str, limit: int) -> str:
    """Truncate text to at most ``limit`` characters, marking the cut."""
    if len(text) <= limit:
        return text
    keep = max(0, limit - len(TRUNCATION_MARKER))
    return text[:keep].rstrip() + TRUNCATION_MARKER[: limit - keep]


def analyst_stop_reason(
    state: Mapping[str, Any], profile: Any, now: float | None = None
) -> str | None:
    """Name the exhausted per-analyst budget, if any; ``None`` means keep researching.

    Once a budget is exhausted no new research starts: evidence already
    gathered is extracted and the section is written.
    """
    now = time.time() if now is None else now
    started_at = state.get("started_at") or now
    if now - started_at >= profile.analyst_deadline_seconds:
        return "deadline"
    if state.get("input_tokens", 0) >= profile.analyst_input_token_budget:
        return "token_budget"
    if state.get("llm_calls", 0) >= profile.max_llm_calls_per_analyst:
        return "llm_call_budget"
    return None
