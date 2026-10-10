"""Per-run model profiles: model routing plus the guardrail limits that go with it."""

import logging
import math
import os
import threading
from dataclasses import dataclass
from typing import Literal, get_args

from src.utils.guardrails import MAX_ANALYSTS, MAX_CONCURRENT_ANALYSTS, MAX_FINDINGS_PER_ANALYST
from src.utils.models import (
    SAFETY_MARGIN,
    ModelTier,
    RateLimitedRunnable,
    _environment_float,
    _environment_thinking,
    build_model,
    model_quota_limits,
)

logger = logging.getLogger(__name__)

ProfileName = Literal["quality", "fast"]
PROFILE_NAMES: tuple[str, ...] = get_args(ProfileName)
DEFAULT_PROFILE: ProfileName = "quality"

ALL_TOOLS = ("tavily_search", "wikipedia", "arxiv", "pubmed", "scrape_webpage")


@dataclass(frozen=True)
class ResearchProfile:
    name: str
    heavy: ModelTier  # Planner, Evaluator
    medium: ModelTier  # Researcher, extraction, report synthesis
    writer: ModelTier  # Analyst section writer (length-capped)
    light: ModelTier  # Introduction, conclusion
    panel: ModelTier  # Question decomposition and analyst personas
    max_research_loops: int
    max_tool_calls_per_pass: int
    # The Researcher is nudged to search again until it reaches this many calls.
    min_tool_calls_per_pass: int
    max_researcher_turns: int
    analyst_deadline_seconds: float
    analyst_input_token_budget: int
    max_llm_calls_per_analyst: int
    researcher_tool_view_chars: int
    extraction_tool_output_chars: int
    extraction_transcript_chars: int
    writer_word_target: int
    allowed_tools: tuple[str, ...]
    # Shortest realistic research pass; only used for the capacity estimate.
    min_pass_seconds: float
    # Most analysts a run may request; Deep Research caps lower because each of its
    # analysts costs far more and a question has at most MAX_REQUIREMENTS to divide.
    max_analysts: int


def _tier(
    profile: str,
    name: str,
    default_model: str,
    max_output_tokens: int,
    thinking_level: str | None,
    timeout_seconds: float,
    *,
    legacy_override: bool = False,
) -> ModelTier:
    prefix = f"GEMINI_{profile.upper()}_{name.upper()}"
    legacy = os.getenv(f"GEMINI_{name.upper()}_MODEL") if legacy_override else None
    return ModelTier(
        name=name,
        model=os.getenv(f"{prefix}_MODEL") or legacy or default_model,
        max_output_tokens=max_output_tokens,
        request_timeout_seconds=_environment_float(
            f"{prefix}_REQUEST_TIMEOUT_SECONDS", timeout_seconds
        ),
        thinking_level=_environment_thinking(f"{prefix}_THINKING", thinking_level),
    )


def _build_profiles() -> dict[str, ResearchProfile]:
    # Thinking tokens count against max_output_tokens on Gemini 3, so thinking
    # tiers keep a large output cap.
    quality = ResearchProfile(
        name="quality",
        heavy=_tier("quality", "heavy", "gemini-3.1-pro-preview", 8_192, "high", 120.0, legacy_override=True),
        medium=_tier("quality", "medium", "gemini-3.8-flash", 16_384, None, 120.0, legacy_override=True),
        writer=_tier("quality", "writer", "gemini-3.8-flash", 4_096, "low", 120.0),
        light=_tier("quality", "light", "gemini-3.8-flash", 8_192, None, 45.0, legacy_override=True),
        panel=_tier("quality", "panel", "gemini-3.8-flash", 16_384, None, 120.0),
        max_research_loops=3,
        max_tool_calls_per_pass=6,
        min_tool_calls_per_pass=4,
        max_researcher_turns=3,
        analyst_deadline_seconds=400.0,
        analyst_input_token_budget=150_000,
        max_llm_calls_per_analyst=25,
        researcher_tool_view_chars=2_000,
        extraction_tool_output_chars=4_000,
        extraction_transcript_chars=24_000,
        writer_word_target=600,
        allowed_tools=ALL_TOOLS,
        min_pass_seconds=30.0,
        max_analysts=5,
    )
    fast = ResearchProfile(
        name="fast",
        heavy=_tier("fast", "heavy", "gemini-3.8-flash", 4_096, "low", 45.0),
        medium=_tier("fast", "medium", "gemini-3.5-flash-lite", 8_192, "minimal", 45.0),
        writer=_tier("fast", "writer", "gemini-3.5-flash-lite", 2_048, "minimal", 45.0),
        light=_tier("fast", "light", "gemini-3.5-flash-lite", 2_048, "minimal", 30.0),
        # Personas and requirements set the shape of the whole run, so fast
        # writes them with Flash rather than Flash-Lite.
        panel=_tier("fast", "panel", "gemini-3.8-flash", 8_192, "low", 45.0),
        # Same research depth as quality; fast differs in models, tools and length.
        max_research_loops=3,
        max_tool_calls_per_pass=6,
        min_tool_calls_per_pass=4,
        max_researcher_turns=3,
        analyst_deadline_seconds=300.0,
        analyst_input_token_budget=150_000,
        max_llm_calls_per_analyst=25,
        researcher_tool_view_chars=1_500,
        extraction_tool_output_chars=3_000,
        extraction_transcript_chars=18_000,
        writer_word_target=500,
        allowed_tools=("tavily_search", "wikipedia", "arxiv", "pubmed"),
        min_pass_seconds=15.0,
        max_analysts=MAX_ANALYSTS,
    )
    return {"quality": quality, "fast": fast}


PROFILES = _build_profiles()


def get_profile(name: str | None = None) -> ResearchProfile:
    key = name or DEFAULT_PROFILE
    if key not in PROFILES:
        raise ValueError(f"model_profile must be one of {list(PROFILE_NAMES)}; got {name!r}.")
    return PROFILES[key]


@dataclass(frozen=True)
class ProfileModels:
    heavy: RateLimitedRunnable
    medium: RateLimitedRunnable
    writer: RateLimitedRunnable
    light: RateLimitedRunnable
    panel: RateLimitedRunnable


_models: dict[str, ProfileModels] = {}
_models_lock = threading.Lock()


def get_models(name: str | None = None) -> ProfileModels:
    """Build a profile's models once, on first use."""
    profile = get_profile(name)
    with _models_lock:
        if profile.name not in _models:
            _models[profile.name] = ProfileModels(
                heavy=build_model(profile.heavy),
                medium=build_model(profile.medium),
                writer=build_model(profile.writer),
                light=build_model(profile.light),
                panel=build_model(profile.panel),
            )
        return _models[profile.name]


# Measured sizes used by the capacity estimate.
_RESEARCH_BRIEF_TOKENS = 800
_TOOL_SCHEMA_TOKENS = 2_100
_FINDING_TOKENS = 190
_PROMPT_OVERHEAD_TOKENS = 600


def estimated_analyst_peak(profile: ResearchProfile) -> dict[str, tuple[int, int]]:
    """Upper-bound (TPM, RPM) one analyst can draw from each model during research."""
    turns = profile.max_researcher_turns
    researcher_tokens = turns * (
        _RESEARCH_BRIEF_TOKENS
        + _TOOL_SCHEMA_TOKENS
        + profile.max_tool_calls_per_pass * profile.researcher_tool_view_chars // 4
    )
    extraction_tokens = profile.extraction_transcript_chars // 4 + _PROMPT_OVERHEAD_TOKENS
    heavy_tokens = 2 * (_PROMPT_OVERHEAD_TOKENS + MAX_FINDINGS_PER_ANALYST * _FINDING_TOKENS)
    passes_per_minute = 60.0 / profile.min_pass_seconds

    per_pass: dict[str, list[int]] = {}
    for model, tokens, calls in (
        (profile.medium.model, researcher_tokens + extraction_tokens, turns + 1),
        (profile.heavy.model, heavy_tokens, 2),
    ):
        totals = per_pass.setdefault(model, [0, 0])
        totals[0] += tokens
        totals[1] += calls
    return {
        model: (math.ceil(tokens * passes_per_minute), math.ceil(calls * passes_per_minute))
        for model, (tokens, calls) in per_pass.items()
    }


def capacity_warnings(profile: ResearchProfile) -> list[str]:
    """Explain which quotas are too small for ten unthrottled parallel analysts."""
    warnings = []
    for model, (peak_tpm, peak_rpm) in estimated_analyst_peak(profile).items():
        tpm, rpm = model_quota_limits(model)
        slot_tpm = tpm * SAFETY_MARGIN / MAX_CONCURRENT_ANALYSTS
        slot_rpm = rpm * SAFETY_MARGIN / MAX_CONCURRENT_ANALYSTS
        if slot_tpm < peak_tpm or slot_rpm < peak_rpm:
            needed_tpm = math.ceil(peak_tpm * MAX_CONCURRENT_ANALYSTS / SAFETY_MARGIN)
            needed_rpm = math.ceil(peak_rpm * MAX_CONCURRENT_ANALYSTS / SAFETY_MARGIN)
            warnings.append(
                f"[{profile.name}] {model}: each analyst slot gets {slot_tpm:,.0f} TPM / "
                f"{slot_rpm:,.0f} RPM but may need {peak_tpm:,} TPM / {peak_rpm:,} RPM; "
                f"analysts may throttle. Equal-speed parallelism for "
                f"{MAX_CONCURRENT_ANALYSTS} analysts needs a quota of ≥{needed_tpm:,} TPM "
                f"and ≥{needed_rpm:,} RPM (GEMINI_QUOTA_* settings)."
            )
    return warnings


for _profile in PROFILES.values():
    for _warning in capacity_warnings(_profile):
        logger.warning(_warning)
