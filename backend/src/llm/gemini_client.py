"""
Single point of contact with Gemini. (No session memory, no chats)

Robust single-turn wrapper around the Gemini API (google-genai SDK).

Designed for :
  - Pick a model chain based on task complexity ("simple" vs "complex").
  - If a model is overloaded/rate-limited, fall back to the next model in the chain.
  - Multiple API keys, rotate keys when a whole chain is quota-exhausted.
  - Retry transient errors (429 / 500 / 503 / 504) with exponential backoff + jitter.
  - Fail fast on non-retryable errors (400 / 403 / 404) — those are bugs, not congestion.
  - Every call is logged (prompt + outcome) for debugging later.

Reference for error handling: https://ai.google.dev/gemini-api/docs/troubleshooting

Usage:
    Parameters
    ----------
        system_prompt : str
            System-level instruction passed to Gemini (context/behavior guidance).
        user_prompt : str
            The user's prompt contents. This becomes the model input; responses are
            expected to be JSON matching `schema`.
        schema : Type[BaseModel]
            A Pydantic model class describing the expected JSON response schema.
            The function returns an instance of this class (parsed/validated).
        complexity : str, optional
            Which model tier to use from `MODEL_TIERS` (e.g. "default").
            Must be one of the keys in `MODEL_TIERS`. Default: "default".
        temperature : float, optional
            Sampling temperature for the generation call. Typical range [0.0, 1.0].

        Returns
        -------
        BaseModel
            An instance of `schema` populated with the parsed response from Gemini.

        from src.llm.client import call_ gemini

        plan = call_gemini(
            system_prompt=SYSTEM_PROMPT,
            user_prompt=user_prompt,
            schema=DayPlan,
            complexity="complex",
            temprature=0.7
        )

        Raises
        ------
        ValueError
            If `complexity` is not a key in `MODEL_TIERS`.
        google.genai.errors.APIError
            If the API returns a fatal error (e.g. 400/403/404) for the request.
        AllModelsFailedError
            If every model across every API key was tried and no successful response
            could be obtained.
"""

from __future__ import annotations

import json
import threading
import time
from typing import Type

from google import genai
from google.genai import types, errors
from pydantic import BaseModel


# Logger - src/core/log.py
from src.core.log import get_logger

logger = get_logger(__name__)


# Configuration
from src.config import (
    API_KEYS,
    API_RPM_LIMIT,
    LLM_CALL_DEADLINE_SECONDS,
    LLM_KEY_ACQUIRE_WAIT_SECONDS,
    LLM_REQUEST_TIMEOUT_MS,
    LLM_TRANSIENT_KEY_COOLDOWN_SECONDS,
    MEMORY_EMBEDDING_MODEL,
    MODEL_TIERS,
    TEMPERATURE,
)


REQUEST_TIMEOUT_MS = LLM_REQUEST_TIMEOUT_MS
CALL_DEADLINE_SECONDS = LLM_CALL_DEADLINE_SECONDS
TRANSIENT_KEY_COOLDOWN_SECONDS = LLM_TRANSIENT_KEY_COOLDOWN_SECONDS
KEY_ACQUIRE_WAIT_SECONDS = LLM_KEY_ACQUIRE_WAIT_SECONDS
MIN_PROVIDER_TIMEOUT_MS = 10_000

# Per-key rate limiting (configurable via .env SIM_API_RPM_LIMIT).
RPM_LIMIT = API_RPM_LIMIT
MIN_INTERVAL_S = 60.0 / RPM_LIMIT   # seconds between calls on the same key


# Errors
class AllModelsFailedError(RuntimeError):
    """Raised when every model in the chain, across every API key, failed."""


class ProviderFailureError(AllModelsFailedError):
    """Terminal, classified failure after one complete API-key rotation."""
    code = "provider_unavailable"
    title = "Provider unavailable"
    guidance = "Check the provider status and try again from the newest checkpoint."

    def payload(self) -> dict[str, str]:
        return {"code": self.code, "title": self.title, "guidance": self.guidance, "message": str(self)}


class APIQuotaExhaustedError(ProviderFailureError):
    code = "api_quota_exhausted"
    title = "API quota exhausted"
    guidance = "All configured keys reported quota/rate exhaustion. Wait for quota reset or add funded keys."


class APIRateLimitedError(ProviderFailureError):
    code = "api_rate_limited"
    title = "API rate limit reached"
    guidance = "The provider asked this project to wait before retrying. The daily quota is not exhausted; wait for the displayed retry period and start from the latest checkpoint."


class APICredentialsError(ProviderFailureError):
    code = "api_credentials_rejected"
    title = "API credentials rejected"
    guidance = "Check that each configured key is active and belongs to the expected project."


class ProviderNetworkError(ProviderFailureError):
    code = "provider_network_unavailable"
    title = "Network or DNS unavailable"
    guidance = "Check internet/DNS connectivity, then resume from the newest checkpoint."


class ProviderTimeoutError(ProviderFailureError):
    code = "provider_timeout"
    title = "Provider timed out"
    guidance = "The Gemini service did not respond in time. Check connectivity/provider status and retry."


class ModelUnavailableError(ProviderFailureError):
    code = "model_unavailable"
    title = "Configured model unavailable"
    guidance = "Verify the configured model name is available to every project."


class InvalidProviderRequestError(ProviderFailureError):
    code = "invalid_provider_request"
    title = "Provider rejected the request"
    guidance = "Inspect the request/schema configuration before restarting the simulation."


# ---------------------------------------------------------------------------
# Per-key rate limiter
# ---------------------------------------------------------------------------
# Google Gemini free tier: 5 requests per minute per key.
# We track last-used timestamps and cooldown expirations per key.

from dataclasses import dataclass, field


@dataclass
class _KeyState:
    client: genai.Client
    last_used: float = 0.0        # monotonic time this key was last reserved
    cooldown_until: float = 0.0   # monotonic time — skip this key until then


_clients: dict[str, genai.Client] = {}
_key_states: dict[str, _KeyState] = {}
_key_state_lock = threading.RLock()


class _CircularKeyRing:
    """Thread-safe circular key queue shared by generation and embeddings.

    Each request starts at the next node so concurrent planner workers distribute
    their first attempts.  A failed request traverses every remaining node once;
    the model itself is never changed as a fallback strategy.
    """
    def __init__(self) -> None:
        self._keys: tuple[str, ...] = ()
        self._cursor = 0
        self._lock = threading.Lock()

    def full_circle(self, keys: list[str]) -> list[str]:
        current = tuple(keys)
        if not current:
            return []
        with self._lock:
            if current != self._keys:
                self._keys, self._cursor = current, 0
            start = self._cursor
            self._cursor = (self._cursor + 1) % len(self._keys)
            return [self._keys[(start + offset) % len(self._keys)] for offset in range(len(self._keys))]


_key_ring = _CircularKeyRing()
_provider_failure = threading.Event()
_provider_failure_error: ProviderFailureError | None = None


def _mark_provider_failure(error: ProviderFailureError) -> None:
    global _provider_failure_error
    _provider_failure_error = error
    _provider_failure.set()


def api_quota_exhausted_reason() -> str | None:
    """Backward-compatible accessor for the terminal provider failure message."""
    return str(_provider_failure_error) if _provider_failure.is_set() and _provider_failure_error else None


def provider_failure() -> ProviderFailureError | None:
    return _provider_failure_error if _provider_failure.is_set() else None


def _failure_category(exc: Exception) -> str:
    if isinstance(exc, errors.APIError):
        if exc.code == 429:
            # Gemini includes the specific violated quota metric. An RPM/TPM
            # metric is a temporary throttle; only an explicit per-day metric
            # should be shown as quota exhaustion.
            message = str(exc.message).lower()
            if "requestsperday" in message or "requests_per_day" in message or "rpd" in message:
                return "quota"
            return "rate_limited"
        return {400: "invalid", 401: "credentials", 403: "credentials", 404: "model"}.get(exc.code, "provider")
    if isinstance(exc, TimeoutError):
        return "timeout"
    text = str(exc).lower()
    if "getaddrinfo" in text or "dns" in text or "connecterror" in text or "connection refused" in text:
        return "network"
    if "timeout" in text or "deadline" in text:
        return "timeout"
    return "provider"


def _terminal_failure(categories: list[str], model: str, last_error: Exception | None) -> ProviderFailureError:
    distinct = set(categories)
    failure_type = {
        frozenset({"rate_limited"}): APIRateLimitedError,
        frozenset({"credentials"}): APICredentialsError,
        frozenset({"network"}): ProviderNetworkError,
        frozenset({"timeout"}): ProviderTimeoutError,
        frozenset({"model"}): ModelUnavailableError,
        frozenset({"invalid"}): InvalidProviderRequestError,
    }.get(frozenset(distinct), ProviderFailureError)
    return failure_type(
        f"{failure_type.title}: all {len(API_KEYS)} keys failed one full circle for {model}. "
        f"Failure categories: {', '.join(sorted(distinct)) or 'unknown'}. Last error: {last_error!r}"
    )


def _get_client(api_key: str) -> genai.Client:
    if api_key not in _clients:
        _clients[api_key] = genai.Client(api_key=api_key)
    return _clients[api_key]


def _get_key_state(api_key: str) -> _KeyState:
    if api_key not in _key_states:
        _key_states[api_key] = _KeyState(client=_get_client(api_key))
    return _key_states[api_key]


def _pick_best_key(excluded_keys: set[str] | None = None) -> str | None:
    """
    Pick the API key that is:
      1. Not in cooldown (``cooldown_until`` in the future).
      2. Respects the per-key RPM limit (``last_used + MIN_INTERVAL_S``).
      3. Among eligible keys, the one with the oldest ``last_used``.
    If all keys are in cooldown or RPM-limited, return the one whose
    cooldown expires soonest (caller should wait).
    """
    excluded_keys = excluded_keys or set()
    now = time.monotonic()
    candidates: list[tuple[float, _KeyState, str]] = []

    for key in API_KEYS:
        if key in excluded_keys:
            continue
        st = _get_key_state(key)
        if now < st.cooldown_until:
            # In cooldown — not usable until cooldown expires
            candidates.append((st.cooldown_until, st, key))
        elif st.last_used == 0.0:
            # Never used — ideal
            candidates.append((0.0, st, key))
        else:
            # Compute when this key will be available (RPM-respecting)
            eligible_at = st.last_used + MIN_INTERVAL_S
            if now >= eligible_at:
                candidates.append((eligible_at, st, key))   # available now
            else:
                candidates.append((eligible_at, st, key))   # will be available later

    if not candidates:
        return None

    # Sort by the tuple's first element (lowest = most ready)
    candidates.sort(key=lambda x: x[0])
    eligible_at, st, key = candidates[0]

    # If the best candidate isn't usable yet, caller must wait
    now = time.monotonic()
    if now < st.cooldown_until or now < st.last_used + MIN_INTERVAL_S:
        return None

    return key


def _reserve_best_key(excluded_keys: set[str] | None = None) -> str | None:
    """Atomically choose and reserve an available key for one HTTP request."""
    with _key_state_lock:
        key = _pick_best_key(excluded_keys)
        if key is not None:
            # Reserve before the request starts so concurrent planner workers
            # cannot burst the same previously-unused credential.
            _get_key_state(key).last_used = time.monotonic()
    return key


def _reserve_specific_key(api_key: str) -> bool:
    """Atomically reserve one ring-selected key if it is currently ready."""
    with _key_state_lock:
        state = _get_key_state(api_key)
        now = time.monotonic()
        if now < state.cooldown_until or now < state.last_used + MIN_INTERVAL_S:
            return False
        state.last_used = now
        return True


def _reserve_key_with_brief_wait(
    excluded_keys: set[str] | None = None,
    max_wait_seconds: float = KEY_ACQUIRE_WAIT_SECONDS,
) -> str | None:
    """Reserve a ready key, waiting only briefly for rate-limit spacing.

    This allows parallel planners to share a finite key pool without stampeding
    it, while refusing to turn a provider cooldown into a long startup stall.
    """
    excluded_keys = excluded_keys or set()
    deadline = time.monotonic() + max(0.0, max_wait_seconds)
    while True:
        key = _reserve_best_key(excluded_keys)
        if key is not None:
            return key
        now = time.monotonic()
        if now >= deadline:
            return None
        with _key_state_lock:
            ready_at = [
                max(st.cooldown_until, st.last_used + MIN_INTERVAL_S)
                for candidate, st in _key_states.items()
                if candidate in API_KEYS and candidate not in excluded_keys
            ]
        if not ready_at:
            return None
        time.sleep(min(0.2, max(0.05, min(ready_at) - now), deadline - now))


def _mark_used(key: str) -> None:
    """Record that *key* was just used successfully."""
    with _key_state_lock:
        st = _get_key_state(key)
        st.last_used = time.monotonic()


def _mark_cooldown(key: str, duration: float = 60.0) -> None:
    """Put *key* into cooldown (won't be picked again for *duration* seconds)."""
    with _key_state_lock:
        st = _get_key_state(key)
        st.cooldown_until = max(st.cooldown_until, time.monotonic() + duration)


def _parse_retry_after(msg: str) -> float:
    """Extract server-suggested retry delay from a 429 error message."""
    import re as _re
    m = _re.search(r"retry in\s+([\d.]+)\s*s", msg, _re.IGNORECASE)
    if m:
        return float(m.group(1))
    # RESOURCE_EXHAUSTED without a retry hint is commonly the free-tier
    # per-minute quota.  Retrying after the normal two-request spacing (24 s
    # at 5 RPM) repeatedly hammers an already-exhausted key pool when several
    # agents decide at once.  Hold that credential for a complete RPM window;
    # callers immediately continue their deterministic plan in the meantime.
    if "resource_exhausted" in msg.lower() or "resource has been exhausted" in msg.lower():
        return max(60.0, MIN_INTERVAL_S * RPM_LIMIT)
    return MIN_INTERVAL_S * 2


def _log_attempt(model: str, ok: bool, detail: str) -> None:
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} | model={model} | ok={ok} | {detail}\n"
    logger.info(line.strip())


# Core caller - simple wrapper
def _single_attempt(
    api_key: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    schema: Type[BaseModel],
    temperature: float,
    timeout_ms: int = REQUEST_TIMEOUT_MS,
) -> BaseModel:
    client = _get_client(api_key)
    response = client.models.generate_content(
        model=model,
        contents=user_prompt,
        config=types.GenerateContentConfig(
            system_instruction=system_prompt,
            response_mime_type="application/json",
            response_schema=schema,
            temperature=temperature,
            http_options=types.HttpOptions(timeout=timeout_ms),
            thinking_config=types.ThinkingConfig(thinking_level="medium"),
        ),
    )

    if getattr(response, "parsed", None) is not None:
        return response.parsed
    # Defensive fallback if .parsed wasn't populated for some reason.
    return schema.model_validate(json.loads(response.text))


def call_gemini(
    system_prompt: str,
    user_prompt: str,
    schema: Type[BaseModel],
    complexity: str = "default",
    temperature: float = TEMPERATURE,
) -> BaseModel:
    if complexity not in MODEL_TIERS:
        raise ValueError(f"Unknown task_complexity '{complexity}', expected one of {list(MODEL_TIERS)}")

    if not API_KEYS:
        error = ProviderFailureError("No Gemini API keys are configured.")
        _mark_provider_failure(error)
        raise error

    # Key rotation is the only fallback policy.  Distinct configured accounts
    # have independent quotas; changing models would hide a quota outage and
    # produce inconsistent simulation behaviour.
    model = MODEL_TIERS[complexity][0]
    last_error: Exception | None = None
    categories: list[str] = []
    for attempt, api_key in enumerate(_key_ring.full_circle(API_KEYS), start=1):
        if not _reserve_specific_key(api_key):
            categories.append("rate_limited")
            continue
        try:
            result = _single_attempt(api_key, model, system_prompt, user_prompt, schema, temperature)
            _mark_used(api_key)
            _log_attempt(model, True, f"key_attempt={attempt}")
            try:
                from src.core.budget import GOVERNOR
                GOVERNOR.record(kind=complexity, model=model)
            except Exception:
                pass
            return result
        except errors.APIError as exc:
            last_error = exc
            categories.append(_failure_category(exc))
            _log_attempt(model, False, f"key_attempt={attempt} code={exc.code} msg={exc.message}")
            if exc.code == 400:
                error = InvalidProviderRequestError(f"{InvalidProviderRequestError.title}: {exc.message}")
                _mark_provider_failure(error)
                raise error
            _mark_cooldown(api_key, duration=3600.0 if exc.code in {401, 403} else _parse_retry_after(exc.message) if exc.code == 429 else TRANSIENT_KEY_COOLDOWN_SECONDS)
        except Exception as exc:
            last_error = exc
            categories.append(_failure_category(exc))
            _mark_cooldown(api_key, duration=TRANSIENT_KEY_COOLDOWN_SECONDS)
            _log_attempt(model, False, f"key_attempt={attempt} transport={exc!r}; rotating key")

    error = _terminal_failure(categories, model, last_error)
    _mark_provider_failure(error)
    raise error


def embed_content(text: str, task_type: str, output_dimensionality: int) -> list[float]:
    """Embed with the same circular key rotation policy as generation."""
    if not API_KEYS:
        error = ProviderFailureError("No Gemini API keys are configured.")
        _mark_provider_failure(error)
        raise error
    last_error: Exception | None = None
    categories: list[str] = []
    for attempt, api_key in enumerate(_key_ring.full_circle(API_KEYS), start=1):
        if not _reserve_specific_key(api_key):
            categories.append("rate_limited")
            continue
        try:
            result = _get_client(api_key).models.embed_content(
                model=MEMORY_EMBEDDING_MODEL,
                contents=text,
                config=types.EmbedContentConfig(task_type=task_type, output_dimensionality=output_dimensionality),
            )
            _mark_used(api_key)
            return list(result.embeddings[0].values)
        except Exception as exc:
            last_error = exc
            categories.append(_failure_category(exc))
            _mark_cooldown(api_key, duration=_parse_retry_after(str(exc)) if "429" in str(exc) else TRANSIENT_KEY_COOLDOWN_SECONDS)
            logger.warning("[gemini] embedding key_attempt=%d failed; rotating key", attempt)
    error = _terminal_failure(categories, "embeddings", last_error)
    _mark_provider_failure(error)
    raise error
