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
            Which model tier to use from `MODEL_TIERS` (e.g. "simple" or "complex").
            Must be one of the keys in `MODEL_TIERS`. Default: "simple".
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
import random
import time
from typing import Type

from google import genai
from google.genai import types, errors
from pydantic import BaseModel


# Logger - src/core/log.py
from src.core.log import get_logger

logger = get_logger(__name__)


# Configuration
from src.config import MODEL_TIERS, API_KEYS, TEMPERATURE


MAX_RETRIES_PER_MODEL = 1       # 1 attempt only — immediately skip overloaded models
BASE_BACKOFF_SECONDS = 0.5
REQUEST_TIMEOUT_MS = 15_000     # per-request deadline (lower = faster fallback)

# Per-key rate limiting (Google free tier: 5 RPM).
RPM_LIMIT = 5
MIN_INTERVAL_S = 60.0 / RPM_LIMIT   # 12 seconds between calls on the same key


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
    last_used: float = 0.0        # monotonic time of last successful call
    cooldown_until: float = 0.0   # monotonic time — skip this key until then


_clients: dict[str, genai.Client] = {}
_key_states: dict[str, _KeyState] = {}


def _get_client(api_key: str) -> genai.Client:
    if api_key not in _clients:
        _clients[api_key] = genai.Client(api_key=api_key)
    return _clients[api_key]


def _get_key_state(api_key: str) -> _KeyState:
    if api_key not in _key_states:
        _key_states[api_key] = _KeyState(client=_get_client(api_key))
    return _key_states[api_key]


def _pick_best_key() -> str | None:
    """
    Pick the API key that is:
      1. Not in cooldown (``cooldown_until`` in the future).
      2. Respects the per-key RPM limit (``last_used + MIN_INTERVAL_S``).
      3. Among eligible keys, the one with the oldest ``last_used``.
    If all keys are in cooldown or RPM-limited, return the one whose
    cooldown expires soonest (caller should wait).
    """
    now = time.monotonic()
    candidates: list[tuple[float, _KeyState, str]] = []

    for key in API_KEYS:
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


def _wait_until_key_available() -> str:
    """
    Block until at least one key is available, then return it.
    Sleeps in small increments so we don't busy-loop.
    """
    while True:
        key = _pick_best_key()
        if key is not None:
            return key
        # All keys are in cooldown — sleep until the shortest cooldown expires
        now = time.monotonic()
        min_cooldown = min(
            st.cooldown_until
            for st in _key_states.values()
        )
        sleep_for = max(0.1, min_cooldown - now)
        time.sleep(sleep_for)


def _mark_used(key: str) -> None:
    """Record that *key* was just used successfully."""
    st = _get_key_state(key)
    st.last_used = time.monotonic()


def _mark_cooldown(key: str, duration: float = 60.0) -> None:
    """Put *key* into cooldown (won't be picked again for *duration* seconds)."""
    st = _get_key_state(key)
    st.cooldown_until = time.monotonic() + duration


# Retry classification

# Codes worth retrying (transient / capacity related).
RETRYABLE_CODES = {429, 500, 503, 504}
# Codes that mean "your request or credentials are wrong" — retrying won't help.
FATAL_CODES = {400, 403, 404}


def _is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, errors.APIError):
        return exc.code in RETRYABLE_CODES
    # Network/timeout errors from the underlying httpx client also deserve a retry.
    return isinstance(exc, (TimeoutError, ConnectionError))


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
            http_options=types.HttpOptions(timeout=REQUEST_TIMEOUT_MS),
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
    complexity: str = "simple",
    temperature: float = TEMPERATURE,
) -> BaseModel:
    if complexity not in MODEL_TIERS:
        raise ValueError(f"Unknown task_complexity '{complexity}', expected one of {list(MODEL_TIERS)}")

    # Build the model list: primary tier first, then cascade to all other tiers
    # as fallbacks when overloaded.
    primary_models = list(MODEL_TIERS[complexity])
    fallback_models = [
        m for tier in MODEL_TIERS.values()
        for m in tier
        if m not in primary_models
    ]
    models = primary_models + fallback_models
    last_error: Exception | None = None
    quota_exhausted: set[str] = set()  # models that got 429 — skip for this call

    # Track which (key, model) pairs we've already tried so we don't
    # infinite-loop across the key pool.
    tried: set[tuple[str, str]] = set()

    while len(tried) < len(API_KEYS) * len(models):
        api_key = _wait_until_key_available()

        for model in models:
            if model in quota_exhausted:
                continue
            if (api_key, model) in tried:
                continue

            for attempt in range(0, MAX_RETRIES_PER_MODEL):
                tried.add((api_key, model))

                try:
                    result = _single_attempt(
                        api_key, model, system_prompt, user_prompt, schema, temperature
                    )
                    _mark_used(api_key)
                    _log_attempt(model, True, f"attempt={attempt}")
                    return result

                except errors.APIError as e:
                    last_error = e
                    detail = f"attempt={attempt} code={e.code} msg={e.message}"
                    _log_attempt(model, False, detail)

                    if e.code in FATAL_CODES:
                        _mark_cooldown(api_key, duration=3600.0)
                        break

                    if e.code not in RETRYABLE_CODES:
                        break

                    # Quota/rate limit — skip this model entirely for this call
                    # (all keys share the same free-tier quota for a given model)
                    if e.code == 429:
                        cooldown = _parse_retry_after(e.message)
                        _mark_cooldown(api_key, duration=cooldown)
                        quota_exhausted.add(model)
                        break

                    if attempt < MAX_RETRIES_PER_MODEL:
                        sleep_s = (
                            BASE_BACKOFF_SECONDS * (2 ** (attempt - 1))
                            + random.uniform(0, 0.5)
                        )
                        time.sleep(sleep_s)
                        continue
                    break

                except Exception as e:
                    last_error = e
                    _log_attempt(model, False, f"attempt={attempt} unexpected={e!r}")
                    if attempt < MAX_RETRIES_PER_MODEL:
                        time.sleep(BASE_BACKOFF_SECONDS)
                        continue
                    break

            # If we got a retryable error and broke out early, try a different key
            # before retrying the same model again.
            if isinstance(last_error, errors.APIError) and last_error.code in RETRYABLE_CODES:
                break

    raise AllModelsFailedError(
        f"All models {models} failed across {len(API_KEYS)} key(s). Last error: {last_error!r}"
    )