import os
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

T = TypeVar("T")


class TokenPerMinuteLimiter:
    """A process-wide rolling token budget shared by all calls to one model tier."""

    def __init__(self, tokens_per_minute: int, safety_margin: float = 0.8) -> None:
        if tokens_per_minute < 1:
            raise ValueError("tokens_per_minute must be positive")
        self._budget = max(1, int(tokens_per_minute * safety_margin))
        self._events: deque[tuple[float, int]] = deque()
        self._lock = threading.Lock()

    def acquire(self, tokens: int) -> None:
        """Block until reserving the request's worst-case token cost is safe."""
        if tokens > self._budget:
            raise ValueError(
                "Estimated request token use exceeds the configured per-minute "
                "budget; reduce the request or raise the matching GEMINI_*_TPM_LIMIT."
            )
        reservation = max(1, tokens)
        while True:
            with self._lock:
                now = time.monotonic()
                while self._events and now - self._events[0][0] >= 60:
                    self._events.popleft()
                used = sum(event_tokens for _, event_tokens in self._events)
                if used + reservation <= self._budget:
                    self._events.append((now, reservation))
                    return
                wait_seconds = max(0.01, 60 - (now - self._events[0][0]))
            time.sleep(wait_seconds)


@dataclass(frozen=True)
class ModelTier:
    name: str
    model: str
    tokens_per_minute: int
    max_output_tokens: int
    request_timeout_seconds: float
    thinking_level: str | None = None


class RateLimitedRunnable(Generic[T]):
    """Preserve LangChain's fluent APIs while enforcing a shared TPM reservation."""

    def __init__(
        self,
        runnable: T,
        limiter: TokenPerMinuteLimiter,
        max_output_tokens: int,
    ) -> None:
        self._runnable = runnable
        self._limiter = limiter
        self._max_output_tokens = max_output_tokens

    def invoke(self, input: Any, *args: Any, **kwargs: Any) -> Any:
        self._limiter.acquire(_estimate_tokens(input) + self._max_output_tokens)
        return self._runnable.invoke(input, *args, **kwargs)

    def with_structured_output(self, *args: Any, **kwargs: Any) -> "RateLimitedRunnable[Any]":
        return RateLimitedRunnable(
            self._runnable.with_structured_output(*args, **kwargs),
            self._limiter,
            self._max_output_tokens,
        )

    def bind_tools(self, *args: Any, **kwargs: Any) -> "RateLimitedRunnable[Any]":
        return RateLimitedRunnable(
            self._runnable.bind_tools(*args, **kwargs),
            self._limiter,
            self._max_output_tokens,
        )

    def __getattr__(self, name: str) -> Any:
        return getattr(self._runnable, name)


def _estimate_tokens(value: Any) -> int:
    """Conservative text-token estimate for request reservation without provider calls."""
    if isinstance(value, str):
        characters = len(value)
    elif isinstance(value, (list, tuple)):
        characters = sum(_estimate_characters(item) for item in value)
    else:
        characters = _estimate_characters(value)
    return max(1, (characters + 3) // 4)


def _estimate_characters(value: Any) -> int:
    content = getattr(value, "content", None)
    if content is not None:
        return len(str(content))
    return len(str(value))


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


def _tier(
    name: str,
    default_model: str,
    default_tpm: int,
    max_output_tokens: int,
    thinking_level: str | None = None,
    default_timeout_seconds: float = 45.0,
) -> ModelTier:
    prefix = f"GEMINI_{name.upper()}"
    return ModelTier(
        name=name,
        model=os.getenv(f"{prefix}_MODEL", default_model),
        tokens_per_minute=_environment_int(f"{prefix}_TPM_LIMIT", default_tpm),
        max_output_tokens=max_output_tokens,
        request_timeout_seconds=_environment_float(
            f"{prefix}_REQUEST_TIMEOUT_SECONDS", default_timeout_seconds
        ),
        thinking_level=thinking_level,
    )


def _build_model(tier: ModelTier) -> RateLimitedRunnable[ChatGoogleGenerativeAI]:
    model_kwargs: dict[str, Any] = {
        "model": tier.model,
        "temperature": 0,
        "max_output_tokens": tier.max_output_tokens,
        "timeout": tier.request_timeout_seconds,
        "max_retries": 1,
    }
    if tier.thinking_level is not None:
        model_kwargs["thinking_level"] = tier.thinking_level
    model = ChatGoogleGenerativeAI(**model_kwargs)
    return RateLimitedRunnable(
        model,
        TokenPerMinuteLimiter(tier.tokens_per_minute),
        tier.max_output_tokens,
    )


# Defaults reserve only 80% of each configured limit. Set *_TPM_LIMIT to the
# project quota for a lower or higher provider allocation without source edits.
HEAVY_TIER = _tier("heavy", "gemini-3.1-pro-preview", 10_000, 2_048, thinking_level="high", default_timeout_seconds=120.0)
MEDIUM_TIER = _tier("medium", "gemini-3.8-flash", 100_000, 16_384, default_timeout_seconds=120.0)
LIGHT_TIER = _tier("light", "gemini-3.8-flash", 100_000, 8_192)

heavy_llm = _build_model(HEAVY_TIER)
medium_llm = _build_model(MEDIUM_TIER)
light_llm = _build_model(LIGHT_TIER)
