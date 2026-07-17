"""
Vector memory backend -- semantic long-term recall via Qdrant + Gemini
embeddings. OFF by default; selected with `SIM_MEMORY_BACKEND=vector`.

Design goals (budget-first)
---------------------------
- **Lazy everything.** `qdrant-client` and the embedding client are imported
  only when this backend is actually constructed, so the project runs fine
  without the package installed. Any failure falls back to the keyword
  backend, so the simulation never breaks because a vector dependency is
  missing or a network call fails.
- **Embed sparingly.** Only durable, high-value memories (daily summaries,
  conversation summaries, key events) should be embedded -- never raw
  per-tick observations. Each embedding call is gated by the LLM budget
  governor, since embeddings draw on the same scarce free-tier quota.
- **Same interface.** Implements the `MemoryRetriever` contract, so the day
  planner and conversation code never know which backend is active.

To enable (you install these yourself):
    pip install qdrant-client
    # run a local Qdrant (docker run -p 6333:6333 qdrant/qdrant) or use
    # an in-memory instance (default here, no server needed).
    set SIM_MEMORY_BACKEND=vector
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from typing import List, Optional

from src.config import API_KEYS
from src.core.log import get_logger

logger = get_logger(__name__)

# Gemini embedding model + vector size (text-embedding-004 -> 768 dims).
_EMBED_MODEL = "text-embedding-004"
_EMBED_DIM = 768
_COLLECTION = "valhalla_memories"

# Only these memory kinds are worth embedding (durable + high value).
_EMBEDDABLE_KINDS = {"summary", "conversation", "memory", "key_event"}


def _stable_id(agent_id: str, text: str) -> int:
    h = hashlib.sha1(f"{agent_id}:{text}".encode("utf-8")).hexdigest()
    return int(h[:15], 16)  # fits comfortably in a 64-bit int


class VectorMemoryRetriever:
    """Qdrant-backed semantic retriever with a keyword fallback."""

    def __init__(self, fallback=None, url: Optional[str] = None) -> None:
        self._fallback = fallback
        self._ok = False
        self._client = None
        self._genai = None

        try:
            from qdrant_client import QdrantClient
            from qdrant_client.http import models as qmodels
            from google import genai

            self._qmodels = qmodels
            # In-memory Qdrant needs no server; pass a url to use a running one.
            self._client = QdrantClient(url=url) if url else QdrantClient(":memory:")
            if not self._client.collection_exists(_COLLECTION):
                self._client.create_collection(
                    collection_name=_COLLECTION,
                    vectors_config=qmodels.VectorParams(
                        size=_EMBED_DIM, distance=qmodels.Distance.COSINE
                    ),
                )
            if not API_KEYS:
                raise RuntimeError("no API keys for embeddings")
            self._genai = genai.Client(api_key=API_KEYS[0])
            self._ok = True
            logger.info("[vector_memory] Qdrant + Gemini embeddings ready")
        except Exception as e:
            logger.warning("[vector_memory] unavailable (%s); using keyword fallback", e)
            self._ok = False

    # -- embeddings --------------------------------------------------------
    def _embed(self, text: str) -> Optional[List[float]]:
        if not self._ok:
            return None
        # Budget-gate embedding calls (same scarce quota as chat).
        try:
            from src.core.budget import GOVERNOR
            if not GOVERNOR.can_afford("embedding", cost=1):
                return None
        except Exception:
            GOVERNOR = None  # noqa: N806
        try:
            resp = self._genai.models.embed_content(model=_EMBED_MODEL, contents=text)
            # google-genai returns .embeddings[0].values
            vec = resp.embeddings[0].values
            try:
                from src.core.budget import GOVERNOR as _G
                _G.record("embedding", _EMBED_MODEL)
            except Exception:
                pass
            return list(vec)
        except Exception as e:
            logger.warning("[vector_memory] embed failed (%s); falling back", e)
            return None

    # -- interface ---------------------------------------------------------
    def store(self, agent_id: str, text: str, *, kind: str = "observation",
              importance: Optional[float] = None, date_str: Optional[str] = None) -> None:
        # Always keep the keyword/short-term copy so nothing is lost.
        if self._fallback is not None:
            try:
                self._fallback.store(agent_id, text, kind=kind,
                                     importance=importance, date_str=date_str)
            except Exception:
                pass
        if not self._ok or kind not in _EMBEDDABLE_KINDS:
            return
        vec = self._embed(text)
        if vec is None:
            return
        try:
            self._client.upsert(
                collection_name=_COLLECTION,
                points=[self._qmodels.PointStruct(
                    id=_stable_id(agent_id, text),
                    vector=vec,
                    payload={
                        "agent_id": agent_id,
                        "text": text,
                        "kind": kind,
                        "date": date_str or datetime.now(timezone.utc).date().isoformat(),
                    },
                )],
            )
        except Exception as e:
            logger.warning("[vector_memory] upsert failed (%s)", e)

    def retrieve(self, agent_id: str, query: str, k: int = 5) -> List[str]:
        if not self._ok:
            return self._fallback.retrieve(agent_id, query, k) if self._fallback else []
        vec = self._embed(query)
        if vec is None:
            return self._fallback.retrieve(agent_id, query, k) if self._fallback else []
        try:
            flt = self._qmodels.Filter(must=[self._qmodels.FieldCondition(
                key="agent_id", match=self._qmodels.MatchValue(value=agent_id))])
            hits = self._client.search(
                collection_name=_COLLECTION, query_vector=vec, query_filter=flt, limit=k)
            out = [f"[{h.payload.get('date','')}] ({h.payload.get('kind','')}) {h.payload.get('text','')}"
                   for h in hits]
            # Blend in keyword hits if vector store is still sparse.
            if len(out) < k and self._fallback is not None:
                out += [m for m in self._fallback.retrieve(agent_id, query, k - len(out))
                        if m not in out]
            return out[:k]
        except Exception as e:
            logger.warning("[vector_memory] search failed (%s); falling back", e)
            return self._fallback.retrieve(agent_id, query, k) if self._fallback else []

    def rolling_summary(self, agent_id: str, days: int = 3,
                        before_date: Optional[str] = None) -> Optional[str]:
        # Rolling summaries come straight from the archives; delegate.
        if self._fallback is not None:
            return self._fallback.rolling_summary(agent_id, days, before_date)
        return None
