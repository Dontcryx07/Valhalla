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

        from src.llm.client import gemini

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

logger = get_logger("gemini_client")


# Configuration
from src.config import MODEL_TIERS, API_KEYS, TEMPERATURE


MAX_RETRIES_PER_MODEL = 2       # backoff retries before moving to next model
BASE_BACKOFF_SECONDS = 1.5
REQUEST_TIMEOUT_MS = 60_000     # per-request deadline


# Errors
class AllModelsFailedError(RuntimeError):
    """Raised when every model in the chain, across every API key, failed."""


# Client pool (one genai.Client per API key, created lazily)
_clients: dict[str, genai.Client] = {}


def _get_client(api_key: str) -> genai.Client:
    if api_key not in _clients: # Check weather client has current API_KEY or not
        _clients[api_key] = genai.Client(api_key=api_key)
    return _clients[api_key]


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

    models = MODEL_TIERS[complexity]
    last_error: Exception | None = None

    for api_key in API_KEYS:
        for model in models:
            for attempt in range(0, MAX_RETRIES_PER_MODEL):
                try:
                    result = _single_attempt(api_key, model, system_prompt, user_prompt, schema, temperature)
                    _log_attempt(model, True, f"attempt={attempt}")
                    return result

                except errors.APIError as e:
                    last_error = e
                    detail = f"attempt={attempt} code={e.code} msg={e.message}"
                    _log_attempt(model, False, detail)

                    if e.code in FATAL_CODES:
                        # Not worth retrying or switching models — it's a request/auth bug.
                        raise

                    if e.code not in RETRYABLE_CODES:
                        # Unknown code: don't loop forever, just move to next model.
                        break

                    if attempt < MAX_RETRIES_PER_MODEL:
                        sleep_s = BASE_BACKOFF_SECONDS * (2 ** (attempt - 1)) + random.uniform(0, 0.5)
                        time.sleep(sleep_s)
                        continue
                    # Retries exhausted for this model -> fall through to next model.
                    break

                except Exception as e:  # noqa: BLE001 - network hiccups, JSON parse issues, etc.
                    last_error = e
                    _log_attempt(model, False, f"attempt={attempt} unexpected={e!r}")
                    if attempt < MAX_RETRIES_PER_MODEL:
                        time.sleep(BASE_BACKOFF_SECONDS)
                        continue
                    break
            # move to next model in the chain
        # move to next API key (only relevant if all models on this key failed)
    # Tried all the possible combinations, nothing worked
    raise AllModelsFailedError(
        f"All models {models} failed across {len(API_KEYS)} key(s). Last error: {last_error!r}"
    )