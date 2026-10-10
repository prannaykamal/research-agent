"""Deterministic system-wide and per-analyst guardrails.

Profile-specific limits (passes, tool calls, deadlines, budgets) live on
``ResearchProfile``; the constants here apply to every profile.
"""

import re
import time
from datetime import date
from typing import Any, Mapping, Sequence

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

# A run whose research tool calls mostly failed gets a coverage note.
MAX_TOOL_ERROR_SHARE = 0.5

# A rejected topic is echoed back in the error, shortened to this many characters.
MAX_TOPIC_ECHO_CHARS = 60

# Evidence newer than this counts as current for status claims.
RECENT_EVIDENCE_DAYS = 365
_SOURCE_DATE = re.compile(r"^(\d{4})(?:-(\d{2})(?:-(\d{2}))?)?$")

# Question decomposition: the requirements a complete answer must cover.
MAX_REQUIREMENTS = 8
REQUIREMENT_MATCH_THRESHOLD = 0.6
_REQUIREMENT_STOPWORDS = frozenset(
    "a an and are as at be been by did do does for from has have how in is it its of on "
    "or over that the this time to was were what when which why with".split()
)


class UnresearchableTopicError(ValueError):
    """The topic names nothing to research (a greeting, a test string, gibberish).

    A ValueError subclass so the API server streams its message and class name, which the
    UI uses to show the message in the configuration form instead of as a failed run.
    """


def unresearchable_topic_message(topic: str) -> str:
    shown = " ".join(topic.split())
    if len(shown) > MAX_TOPIC_ECHO_CHARS:
        shown = shown[: MAX_TOPIC_ECHO_CHARS - 1].rstrip() + "…"
    return f'"{shown}" does not name a subject or question to research.'


def validate_max_analysts(value: Any, maximum: int = MAX_ANALYSTS) -> int:
    """Return a valid analyst count or raise a clear error for the caller.

    ``maximum`` is the selected profile's own cap, never above ``MAX_ANALYSTS``.
    """
    maximum = min(maximum, MAX_ANALYSTS)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(
            f"max_analysts must be an integer between {MIN_ANALYSTS} and "
            f"{maximum}; got {value!r}."
        )
    if not MIN_ANALYSTS <= value <= maximum:
        raise ValueError(
            f"max_analysts must be between {MIN_ANALYSTS} and {maximum}; got {value}."
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


def _requirement_tokens(text: str) -> set[str]:
    # A four-character prefix is a crude stem: "evolution" and "evolved" agree.
    return {
        token[:4]
        for token in re.findall(r"[a-z0-9]+", text.casefold())
        if len(token) > 1 and token not in _REQUIREMENT_STOPWORDS
    }


def match_requirement(label: str, requirements: Sequence[str]) -> str | None:
    """Map a model-written requirement label onto the canonical list, or ``None``.

    Models are told to copy requirements verbatim but sometimes paraphrase, so
    the best token overlap above a threshold also counts.
    """
    tokens = _requirement_tokens(label)
    if not tokens:
        return None
    best, best_score = None, 0.0
    for requirement in requirements:
        candidate = _requirement_tokens(requirement)
        if not candidate:
            continue
        score = len(tokens & candidate) / min(len(tokens), len(candidate))
        if score > best_score:
            best, best_score = requirement, score
    return best if best_score >= REQUIREMENT_MATCH_THRESHOLD else None


def assign_requirements(
    claimed: Sequence[Sequence[str]], requirements: Sequence[str]
) -> list[list[str]]:
    """Canonicalize each analyst's claimed requirements and give every unowned one an owner.

    An unowned requirement goes to the analyst owning the fewest, so a single
    analyst owns them all and no part of the question goes unresearched.
    """
    owned: list[list[str]] = []
    for labels in claimed:
        canonical: list[str] = []
        for label in labels:
            match = match_requirement(label, requirements)
            if match is not None and match not in canonical:
                canonical.append(match)
        owned.append(canonical)
    if not owned:
        return owned
    taken = {requirement for analyst in owned for requirement in analyst}
    for requirement in requirements:
        if requirement not in taken:
            min(owned, key=len).append(requirement)
    order = {requirement: index for index, requirement in enumerate(requirements)}
    return [sorted(analyst, key=order.__getitem__) for analyst in owned]


def covered_requirements(labels: Sequence[str], owned: Sequence[str]) -> list[str]:
    """Owned requirements that at least one finding's requirement label supports.

    An unlabelled finding counts toward the requirement when the analyst owns
    exactly one, since there is nothing else it could support.
    """
    covered: set[str] = set()
    for label in labels:
        match = match_requirement(label, owned)
        if match is None and not label.strip() and len(owned) == 1:
            match = owned[0]
        if match is not None:
            covered.add(match)
    return [requirement for requirement in owned if requirement in covered]


def uncovered_requirements(requirements: Sequence[str], covered: Sequence[str]) -> list[str]:
    """Requirements, in question order, that no evidence supports."""
    supported = set(covered)
    return [requirement for requirement in requirements if requirement not in supported]


def evidence_age_days(source_date: str, today: str) -> int | None:
    """Days from a source's date to ``today``, or ``None`` when the date is unusable.

    Partial dates (``YYYY`` or ``YYYY-MM``) count from their first day, so a
    source dated only by year is never treated as newer than it might be.
    """
    match = _SOURCE_DATE.match(source_date.strip())
    if not match:
        return None
    year, month, day = (int(part) if part else 1 for part in match.groups())
    try:
        return (date.fromisoformat(today) - date(year, month, day)).days
    except ValueError:
        return None


def is_recent(source_date: str, today: str) -> bool | None:
    """Whether a source is under ``RECENT_EVIDENCE_DAYS`` old; ``None`` when undated."""
    age = evidence_age_days(source_date, today)
    return None if age is None else age <= RECENT_EVIDENCE_DAYS
