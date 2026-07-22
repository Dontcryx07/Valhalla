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
        raise AllModelsFailedError("No Gemini API keys are configured.")

    # Primary model first. For a transient failure we exhaust other keys before
    # paying for an alternate model, preserving output consistency.
    primary_models = list(MODEL_TIERS[complexity])
    fallback_models = [
        m for tier in MODEL_TIERS.values()
        for m in tier
        if m not in primary_models
    ]
    models = primary_models + fallback_models
    last_error: Exception | None = None
    deadline = time.monotonic() + CALL_DEADLINE_SECONDS

    for model in models:
        attempted_keys: set[str] = set()
        while len(attempted_keys) < len(API_KEYS):
            remaining_seconds = deadline - time.monotonic()
            if remaining_seconds <= 0:
                raise AllModelsFailedError(
                    f"Gemini fallback deadline ({CALL_DEADLINE_SECONDS:.1f}s) expired. "
                    f"Last error: {last_error!r}"
                )

            api_key = _reserve_key_with_brief_wait(
                excluded_keys=attempted_keys,
                max_wait_seconds=min(KEY_ACQUIRE_WAIT_SECONDS, remaining_seconds),
            )
            if api_key is None:
                # Do not sleep through key cooldowns. A later model cannot
                # help without an eligible credential either.
                raise AllModelsFailedError(
                    f"No Gemini API key is immediately available. Last error: {last_error!r}"
                )

            attempted_keys.add(api_key)
            remaining_ms = int(remaining_seconds * 1000)
            if remaining_ms < MIN_PROVIDER_TIMEOUT_MS:
                raise AllModelsFailedError(
                    "Gemini fallback deadline has less than the provider's 10s minimum remaining. "
                    f"Last error: {last_error!r}"
                )
            timeout_ms = min(REQUEST_TIMEOUT_MS, remaining_ms)

            try:
                result = _single_attempt(
                    api_key, model, system_prompt, user_prompt, schema, temperature, timeout_ms
                )
                _mark_used(api_key)
                _log_attempt(model, True, "attempt=0")
                try:
                    from src.core.budget import GOVERNOR
                    GOVERNOR.record(kind=complexity, model=model)
                except Exception:
                    pass
                return result

            except errors.APIError as exc:
                last_error = exc
                _log_attempt(model, False, f"attempt=0 code={exc.code} msg={exc.message}")

                if exc.code == 400:
                    # A malformed prompt/schema is not recoverable by retries.
                    raise
                if exc.code == 404:
                    # A retired model fails for every key: skip it immediately.
                    break
                if exc.code in {401, 403}:
                    _mark_cooldown(api_key, duration=3600.0)
                elif exc.code == 429:
                    _mark_cooldown(api_key, duration=_parse_retry_after(exc.message))
                else:
                    _mark_cooldown(api_key, duration=TRANSIENT_KEY_COOLDOWN_SECONDS)
                # Capacity, quota, and credential faults rotate to another key
                # while retaining this preferred model.
                continue

            except Exception as exc:
                # SDK transport timeouts do not consistently subclass
                # TimeoutError, so unknown transport faults rotate immediately.
                last_error = exc
                _mark_cooldown(api_key, duration=TRANSIENT_KEY_COOLDOWN_SECONDS)
                _log_attempt(model, False, f"attempt=0 transport={exc!r}; rotating key")
                continue

    raise AllModelsFailedError(
        f"All models {models} failed within {CALL_DEADLINE_SECONDS:.1f}s across "
        f"{len(API_KEYS)} key(s). Last error: {last_error!r}"
    )
