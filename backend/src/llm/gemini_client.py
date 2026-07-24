"""Gemini access through a deterministic, head-first key ring.

Every API call starts with the first configured key.  On *any* exception it
tries the next node exactly once.  There are deliberately no delays, retries,
timeouts, cooldowns, key reservations, or model changes in this module.
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from typing import Iterator, Type

from google import genai
from google.genai import types
from pydantic import BaseModel

from src.config import API_KEYS, GEMINI_MODEL, MEMORY_EMBEDDING_MODEL, TEMPERATURE
from src.core.log import get_logger

logger = get_logger(__name__)


class ProviderFailureError(RuntimeError):
    code = "api_quota_exhausted"
    title = "API quota exhausted"
    guidance = "Every configured API key was unavailable for this request. Check API-key availability and restart from the latest checkpoint."

    def payload(self) -> dict[str, str]:
        return {
            "code": self.code,
            "title": self.title,
            "guidance": self.guidance,
            "message": str(self),
        }


class APIQuotaExhaustedError(ProviderFailureError):
    """Raised only after one call has tried every configured key once."""


@dataclass(frozen=True)
class _KeyNode:
    index: int
    key: str
    next: "_KeyNode | None" = None


class CircularKeyRing:
    """Immutable-order circular linked list whose traversal always starts at head."""

    def __init__(self, keys: list[str]) -> None:
        if not keys:
            raise ValueError("At least one Gemini API key is required")
        nodes = [_KeyNode(index=index + 1, key=key) for index, key in enumerate(keys)]
        for index, node in enumerate(nodes):
            object.__setattr__(node, "next", nodes[(index + 1) % len(nodes)])
        self.head = nodes[0]
        self.size = len(nodes)

    def traverse_from_head(self) -> Iterator[_KeyNode]:
        node = self.head
        for _ in range(self.size):
            yield node
            assert node.next is not None
            node = node.next


_clients: dict[str, genai.Client] = {}
_clients_lock = threading.Lock()
_provider_failure: ProviderFailureError | None = None


def _get_client(api_key: str) -> genai.Client:
    with _clients_lock:
        client = _clients.get(api_key)
        if client is None:
            client = genai.Client(api_key=api_key)
            _clients[api_key] = client
        return client


def _new_ring() -> CircularKeyRing:
    if not API_KEYS:
        raise APIQuotaExhaustedError("API quota exhausted: no Gemini API keys are configured.")
    return CircularKeyRing(list(API_KEYS))


def _quota_exhausted(model: str, errors: list[Exception]) -> APIQuotaExhaustedError:
    global _provider_failure
    last = errors[-1] if errors else None
    error = APIQuotaExhaustedError(
        f"API quota exhausted: all {len(API_KEYS)} configured keys failed one complete circle for {model}. "
        f"Last provider error: {last!r}"
    )
    _provider_failure = error
    return error


def provider_failure() -> ProviderFailureError | None:
    return _provider_failure


def api_quota_exhausted_reason() -> str | None:
    return str(_provider_failure) if _provider_failure else None


def _record_success(complexity: str) -> None:
    global _provider_failure
    _provider_failure = None
    try:
        from src.core.budget import GOVERNOR
        GOVERNOR.record(kind=complexity, model=GEMINI_MODEL)
    except Exception:
        pass


def call_gemini(
    system_prompt: str,
    user_prompt: str,
    schema: Type[BaseModel],
    complexity: str = "default",
    temperature: float = TEMPERATURE,
) -> BaseModel:
    """Call exactly one model, moving through the ring on any failure."""
    errors: list[Exception] = []
    for node in _new_ring().traverse_from_head():
        try:
            response = _get_client(node.key).models.generate_content(
                model=GEMINI_MODEL,
                contents=user_prompt,
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    response_mime_type="application/json",
                    response_schema=schema,
                    temperature=temperature,
                    thinking_config=types.ThinkingConfig(thinking_level="medium"),
                ),
            )
            result = response.parsed if getattr(response, "parsed", None) is not None else schema.model_validate(json.loads(response.text))
            logger.info("[gemini] model=%s key_index=%d ok", GEMINI_MODEL, node.index)
            _record_success(complexity)
            return result
        except Exception as exc:
            errors.append(exc)
            logger.warning("[gemini] model=%s key_index=%d failed (%s); advancing", GEMINI_MODEL, node.index, type(exc).__name__)
    raise _quota_exhausted(GEMINI_MODEL, errors)


def embed_content(text: str, task_type: str, output_dimensionality: int) -> list[float]:
    """Embed once per ring node, always beginning at the first configured key."""
    errors: list[Exception] = []
    for node in _new_ring().traverse_from_head():
        try:
            response = _get_client(node.key).models.embed_content(
                model=MEMORY_EMBEDDING_MODEL,
                contents=text,
                config=types.EmbedContentConfig(task_type=task_type, output_dimensionality=output_dimensionality),
            )
            _record_success("embedding")
            return list(response.embeddings[0].values)
        except Exception as exc:
            errors.append(exc)
            logger.warning("[gemini] embedding key_index=%d failed (%s); advancing", node.index, type(exc).__name__)
    raise _quota_exhausted(MEMORY_EMBEDDING_MODEL, errors)
