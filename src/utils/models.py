import contextvars
import json
import logging
import os
import re
import threading
import time
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Generic, Iterator, TypeVar

import httpx
from dotenv import load_dotenv
from langchain_core.messages import AIMessage
from langchain_core.utils.function_calling import convert_to_openai_tool
from langchain_google_genai import ChatGoogleGenerativeAI

from src.utils.guardrails import MAX_CONCURRENT_ANALYSTS

load_dotenv()

logger = logging.getLogger(__name__)

T = TypeVar("T")

SAFETY_MARGIN = 0.8
WINDOW_SECONDS = 60.0
LLM_ATTEMPTS = 2
RETRY_BACKOFF_SECONDS = 3.0
THINKING_LEVELS = {"minimal", "low", "medium", "high"}

# Gemini quotas are per project and per model; TPM counts input tokens only.
# Defaults mirror published Tier 1 limits. Override each with
# GEMINI_QUOTA_<MODEL_ID>_TPM / _RPM using the values shown in AI Studio.
DEFAULT_MODEL_QUOTAS: dict[str, tuple[int, int]] = {
    "gemini-3.1-pro-preview": (2_000_000, 150),
    "gemini-3.8-flash": (1_000_000, 1_000),
    "gemini-3.5-flash-lite": (4_000_000, 4_000),
}
FALLBACK_MODEL_QUOTA = (1_000_000, 150)


class RollingWindowLimiter:
    """A process-wide rolling one-minute budget of an abstract cost unit."""

    def __init__(self, capacity: int, *, strict: bool = True, unit: str = "token") -> None:
        if capacity < 1:
            raise ValueError("capacity must be positive")
        self._budget = capacity
        self._strict = strict
        self._unit = unit
        self._events: deque[list[float]] = deque()
        self._lock = threading.Lock()

    @property
    def budget(self) -> int:
        return self._budget

    def acquire(self, cost: int) -> list[float]:
        """Block until reserving ``cost`` is safe and return the reservation.

        A strict limiter rejects a single request larger than its budget. A
        non-strict one (an analyst slot) admits it once its window is empty, so
        an oversized request is slowed rather than failed.
        """
        reservation = max(1, cost)
        if reservation > self._budget and self._strict:
            raise ValueError(
                f"Estimated request {self._unit} use exceeds the configured per-minute "
                "budget; reduce the request or raise the matching GEMINI_QUOTA_* limit."
            )
        while True:
            with self._lock:
                now = time.monotonic()
                while self._events and now - self._events[0][0] >= WINDOW_SECONDS:
                    self._events.popleft()
                used = sum(event[1] for event in self._events)
                if used + reservation <= self._budget or not self._events:
                    event = [now, float(reservation)]
                    self._events.append(event)
                    return event
                wait_seconds = max(0.01, WINDOW_SECONDS - (now - self._events[0][0]))
            time.sleep(wait_seconds)

    def adjust(self, event: list[float], actual: int) -> None:
        """Replace a reservation's estimate with the provider-reported cost."""
        with self._lock:
            event[1] = float(max(1, actual))


class TokenPerMinuteLimiter(RollingWindowLimiter):
    def __init__(
        self, tokens_per_minute: int, safety_margin: float = SAFETY_MARGIN, *, strict: bool = True
    ) -> None:
        if tokens_per_minute < 1:
            raise ValueError("tokens_per_minute must be positive")
        super().__init__(max(1, int(tokens_per_minute * safety_margin)), strict=strict)


class RequestPerMinuteLimiter(RollingWindowLimiter):
    def __init__(
        self, requests_per_minute: int, safety_margin: float = SAFETY_MARGIN, *, strict: bool = True
    ) -> None:
        if requests_per_minute < 1:
            raise ValueError("requests_per_minute must be positive")
        super().__init__(
            max(1, int(requests_per_minute * safety_margin)), strict=strict, unit="request"
        )


class ModelQuota:
    """Paired TPM and RPM budgets for one model, or for one analyst's share of it."""

    def __init__(self, tpm: int, rpm: int, *, share: int = 1, strict: bool = True) -> None:
        self.tokens = TokenPerMinuteLimiter(max(1, tpm // share), strict=strict)
        self.requests = RequestPerMinuteLimiter(max(1, rpm // share), strict=strict)

    def acquire(self, tokens: int) -> list[float]:
        self.requests.acquire(1)
        return self.tokens.acquire(tokens)


def _sanitize_model_id(model_id: str) -> str:
    return re.sub(r"[^A-Z0-9]+", "_", model_id.upper()).strip("_")


def model_quota_limits(model_id: str) -> tuple[int, int]:
    """Return the configured (TPM, RPM) project quota for a model."""
    default_tpm, default_rpm = DEFAULT_MODEL_QUOTAS.get(model_id, FALLBACK_MODEL_QUOTA)
    prefix = f"GEMINI_QUOTA_{_sanitize_model_id(model_id)}"
    return (
        _environment_int(f"{prefix}_TPM", default_tpm),
        _environment_int(f"{prefix}_RPM", default_rpm),
    )


_global_quotas: dict[str, ModelQuota] = {}
_global_quotas_lock = threading.Lock()


def global_quota(model_id: str) -> ModelQuota:
    """The single process-wide quota shared by every tier that uses ``model_id``."""
    with _global_quotas_lock:
        if model_id not in _global_quotas:
            _global_quotas[model_id] = ModelQuota(*model_quota_limits(model_id))
        return _global_quotas[model_id]


class AnalystSlot:
    """A fixed 1/N share of every model quota, held by one running analyst."""

    def __init__(self, index: int, share: int) -> None:
        self.index = index
        self._share = share
        self._quotas: dict[str, ModelQuota] = {}
        self._lock = threading.Lock()

    def quota(self, model_id: str) -> ModelQuota:
        with self._lock:
            if model_id not in self._quotas:
                tpm, rpm = model_quota_limits(model_id)
                self._quotas[model_id] = ModelQuota(tpm, rpm, share=self._share, strict=False)
            return self._quotas[model_id]


class AnalystSlotPool:
    """Caps concurrent analysts process-wide and gives each an equal quota share."""

    def __init__(self, size: int) -> None:
        self.size = size
        self._semaphore = threading.BoundedSemaphore(size)
        self._free = deque(AnalystSlot(index, size) for index in range(size))
        self._lock = threading.Lock()

    @contextmanager
    def slot(self) -> Iterator[AnalystSlot]:
        self._semaphore.acquire()
        with self._lock:
            slot = self._free.popleft()
        try:
            yield slot
        finally:
            with self._lock:
                self._free.append(slot)
            self._semaphore.release()


ANALYST_SLOTS = AnalystSlotPool(MAX_CONCURRENT_ANALYSTS)


@dataclass
class UsageMeter:
    """Per-analyst LLM call and input-token totals."""

    calls: int = 0
    input_tokens: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def record(self, input_tokens: int) -> None:
        with self._lock:
            self.calls += 1
            self.input_tokens += input_tokens


_current_slot: contextvars.ContextVar[AnalystSlot | None] = contextvars.ContextVar(
    "current_analyst_slot", default=None
)
_current_meter: contextvars.ContextVar[UsageMeter | None] = contextvars.ContextVar(
    "current_usage_meter", default=None
)


@contextmanager
def analyst_context(slot: AnalystSlot | None, meter: UsageMeter | None) -> Iterator[None]:
    """Route LLM calls in this context through ``slot`` and record them in ``meter``."""
    slot_token = _current_slot.set(slot)
    meter_token = _current_meter.set(meter)
    try:
        yield
    finally:
        _current_meter.reset(meter_token)
        _current_slot.reset(slot_token)


def current_usage() -> UsageMeter | None:
    return _current_meter.get()


def is_transient_error(exc: BaseException) -> bool:
    """Return whether a provider error is worth one more attempt (429 or 5xx).

    Timeouts are excluded: retrying a full request timeout would consume the
    analyst deadline.
    """
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        code = getattr(current, "code", None) or getattr(current, "status_code", None)
        if isinstance(code, int) and (code == 429 or 500 <= code < 600):
            return True
        if isinstance(current, (httpx.ConnectError, httpx.RemoteProtocolError)):
            return True
        if "RESOURCE_EXHAUSTED" in str(current) or "UNAVAILABLE" in str(current):
            return True
        current = current.__cause__ or current.__context__
    return False


@dataclass(frozen=True)
class ModelTier:
    name: str
    model: str
    max_output_tokens: int
    request_timeout_seconds: float
    thinking_level: str | None = None


class RateLimitedRunnable(Generic[T]):
    """Preserve LangChain's fluent APIs while enforcing per-model quotas."""

    def __init__(self, runnable: T, model_id: str, extra_tokens: int = 0) -> None:
        self._runnable = runnable
        self.model_id = model_id
        self._extra_tokens = extra_tokens

    def invoke(self, input: Any, *args: Any, **kwargs: Any) -> Any:
        estimate = _estimate_tokens(input) + self._extra_tokens
        for attempt in range(1, LLM_ATTEMPTS + 1):
            reservations = self._acquire(estimate)
            try:
                result = self._runnable.invoke(input, *args, **kwargs)
            except Exception as exc:
                if attempt < LLM_ATTEMPTS and is_transient_error(exc):
                    logger.warning(
                        "Transient %s error on attempt %d/%d; retrying: %r",
                        self.model_id,
                        attempt,
                        LLM_ATTEMPTS,
                        exc,
                    )
                    time.sleep(RETRY_BACKOFF_SECONDS * attempt)
                    continue
                raise
            actual = _actual_input_tokens(result)
            if actual is not None:
                for limiter, event in reservations:
                    limiter.adjust(event, actual)
            meter = _current_meter.get()
            if meter is not None:
                meter.record(actual if actual is not None else estimate)
            return result
        raise RuntimeError("unreachable")  # pragma: no cover

    def _acquire(self, tokens: int) -> list[tuple[RollingWindowLimiter, list[float]]]:
        reservations = []
        slot = _current_slot.get()
        if slot is not None:
            quota = slot.quota(self.model_id)
            reservations.append((quota.tokens, quota.acquire(tokens)))
        quota = global_quota(self.model_id)
        reservations.append((quota.tokens, quota.acquire(tokens)))
        return reservations

    def with_structured_output(self, schema: Any, *args: Any, **kwargs: Any) -> "RateLimitedRunnable[Any]":
        return RateLimitedRunnable(
            self._runnable.with_structured_output(schema, *args, **kwargs),
            self.model_id,
            self._extra_tokens + _schema_tokens(schema),
        )

    def bind_tools(self, tools: Any, *args: Any, **kwargs: Any) -> "RateLimitedRunnable[Any]":
        return RateLimitedRunnable(
            self._runnable.bind_tools(tools, *args, **kwargs),
            self.model_id,
            self._extra_tokens + _tool_schema_tokens(tools),
        )

    def __getattr__(self, name: str) -> Any:
        return getattr(self._runnable, name)


def _actual_input_tokens(result: Any) -> int | None:
    message = result.get("raw") if isinstance(result, dict) else result
    if isinstance(message, AIMessage) and message.usage_metadata:
        tokens = message.usage_metadata.get("input_tokens")
        return int(tokens) if tokens else None
    return None


def _schema_tokens(schema: Any) -> int:
    try:
        payload = schema.model_json_schema() if hasattr(schema, "model_json_schema") else schema
        return _characters_to_tokens(len(json.dumps(payload, default=str)))
    except (TypeError, ValueError):
        return 0


def _tool_schema_tokens(tools: Any) -> int:
    characters = 0
    for tool in tools:
        try:
            characters += len(json.dumps(convert_to_openai_tool(tool), default=str))
        except (TypeError, ValueError):
            continue
    return _characters_to_tokens(characters)


def _characters_to_tokens(characters: int) -> int:
    return max(1, (characters + 3) // 4)


def _estimate_tokens(value: Any) -> int:
    """Conservative input-token estimate for request reservation without provider calls."""
    if isinstance(value, str):
        characters = len(value)
    elif isinstance(value, (list, tuple)):
        characters = sum(_estimate_characters(item) for item in value)
    else:
        characters = _estimate_characters(value)
    return _characters_to_tokens(characters)


def _estimate_characters(value: Any) -> int:
    content = getattr(value, "content", None)
    if content is None:
        return len(str(value))
    characters = len(str(content))
    tool_calls = getattr(value, "tool_calls", None)
    if tool_calls:
        characters += len(json.dumps(tool_calls, default=str))
    return characters


def _environment_float(name: str, default: float) -> float:
    value = os.getenv(name, str(default))
    try:
        parsed = float(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a positive number") from exc
    if parsed <= 0:
        raise ValueError(f"{name} must be a positive number")
    return parsed


def _environment_int(name: str, default: int) -> int:
    value = os.getenv(name, str(default))
    try:
        parsed = int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a positive integer") from exc
    if parsed < 1:
        raise ValueError(f"{name} must be a positive integer")
    return parsed


def _environment_thinking(name: str, default: str | None) -> str | None:
    value = os.getenv(name)
    if value is None:
        return default
    value = value.strip().lower()
    if value in {"", "none", "default"}:
        return None
    if value not in THINKING_LEVELS:
        raise ValueError(f"{name} must be one of {sorted(THINKING_LEVELS)} or 'none'")
    return value


def build_model(tier: ModelTier) -> RateLimitedRunnable[ChatGoogleGenerativeAI]:
    model_kwargs: dict[str, Any] = {
        "model": tier.model,
        "temperature": 0,
        "max_output_tokens": tier.max_output_tokens,
        "timeout": tier.request_timeout_seconds,
        # langchain-google-genai treats 1 as "no retries"; RateLimitedRunnable
        # retries instead so every attempt passes through the quota limiters.
        "max_retries": 1,
    }
    if tier.thinking_level is not None:
        model_kwargs["thinking_level"] = tier.thinking_level
    return RateLimitedRunnable(ChatGoogleGenerativeAI(**model_kwargs), tier.model)
