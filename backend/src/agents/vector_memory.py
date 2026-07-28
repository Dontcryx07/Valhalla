"""Persistent, per-persona Cloud Qdrant long-term memory and RAG retrieval.

Qdrant is the sole long-term store.  The active day's short-term JSON is
converted into durable memory records during handoff; once indexing succeeds,
that operational file can be removed.  Retrieval returns query-relevant,
ranked context for model prompts.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Iterable, List, Optional

from src import config as _cfg
from src.core.log import get_logger
from src.llm.gemini_client import ProviderFailureError, embed_content

logger = get_logger(__name__)

_IMPORTANCE = {
    "summary": 3.0, "conversation": 2.5, "action": 1.5,
    "key_event": 2.0, "observation": 0.5, "plan": 1.0,
}
_GIB = 1024 ** 3


def safe_agent_id(agent_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", agent_id.strip()).strip("._-").lower() or "unknown"


def collection_name(agent_id: str) -> str:
    """One active collection per agent; later schema versions are isolated."""
    base = safe_agent_id(agent_id)
    return base if _cfg.MEMORY_COLLECTION_VERSION == "v1" else f"{base}__{_cfg.MEMORY_COLLECTION_VERSION}"


def source_hash(agent_id: str, date: str, kind: str, text: str) -> str:
    return hashlib.sha256(f"{agent_id}\x1f{date}\x1f{kind}\x1f{text}".encode("utf-8")).hexdigest()


def point_id(agent_id: str, date: str, kind: str, text: str) -> int:
    # Qdrant accepts unsigned 64-bit point IDs.  A deterministic ID makes
    # archive writes naturally idempotent.
    return int(source_hash(agent_id, date, kind, text)[:15], 16)


@dataclass(frozen=True)
class MemoryRecord:
    agent_id: str
    date: str
    kind: str
    text: str
    importance: float

    @property
    def hash(self) -> str:
        return source_hash(self.agent_id, self.date, self.kind, self.text)

    @property
    def id(self) -> int:
        return point_id(self.agent_id, self.date, self.kind, self.text)

    def payload(self) -> dict:
        return {"date": self.date, "kind": self.kind, "importance": self.importance,
                "source_hash": self.hash, "text": self.text,
                "recall_count": 0, "last_recalled_at": self.date,
                "retention_score": min(1.0, max(0.0, self.importance / 3.0))}


def archive_records(agent_id: str, day: dict) -> list[MemoryRecord]:
    """Convert a completed short-term day into durable Qdrant records.

    Full transcripts and periodic snapshots are intentionally excluded.  Plans,
    completed actions, conversation summaries, explicit durable events and the
    daily summary preserve the information future model calls can use.
    """
    date = str(day.get("date") or "")
    records: list[MemoryRecord] = []
    summary = str(day.get("summary") or day.get("daily_summary") or "").strip()
    if summary:
        records.append(MemoryRecord(agent_id, date, "summary", summary, _IMPORTANCE["summary"]))
    for action in day.get("day_plan", []) or []:
        text = str(action.get("action") or action.get("description") or "").strip()
        if text:
            location = str(action.get("location_id") or action.get("location") or "").strip()
            when = "-".join(filter(None, (str(action.get("start") or ""), str(action.get("end") or ""))))
            records.append(MemoryRecord(agent_id, date, "plan",
                f"Planned {text}" + (f" at {location}" if location else "") + (f" ({when})" if when else ""),
                _IMPORTANCE["plan"]))
    for event in day.get("events", []) or []:
        event_type = str(event.get("type") or "")
        if event_type == "action_completed":
            action = str(event.get("action") or "").strip()
            if action:
                outcome = str(event.get("outcome") or "").strip()
                location = str(event.get("location") or "").strip()
                text = f"Completed {action}" + (f" at {location}" if location else "") + (f": {outcome}" if outcome else "")
                records.append(MemoryRecord(agent_id, date, "action", text, _IMPORTANCE["action"]))
        elif event_type in {"key_event", "durable_memory", "memory"}:
            details = event.get("details") if isinstance(event.get("details"), dict) else {}
            text = str(event.get("summary") or details.get("content") or "").strip()
            if text:
                records.append(MemoryRecord(agent_id, date, "key_event", text,
                    float(details.get("importance", _IMPORTANCE["key_event"]))))
    for conversation in day.get("conversations", []) or []:
        text = str(conversation.get("summary") or "").strip()
        if text:
            records.append(MemoryRecord(agent_id, date, "conversation", text, _IMPORTANCE["conversation"]))
    for event in day.get("key_events", []) or []:
        text = str(event.get("summary") or event.get("text") or "").strip()
        if text:
            records.append(MemoryRecord(agent_id, date, "key_event", text,
                                        float(event.get("importance", _IMPORTANCE["key_event"]))))
    return records


class VectorMemoryRetriever:
    """Cloud Qdrant writer/retriever implementing the application's RAG layer."""

    def __init__(self, *, client=None, embedder=None, qmodels=None) -> None:
        self._client = client
        self._qmodels = qmodels
        self._embedder = embedder
        self._ok = False
        if not _cfg.SEMANTIC_MEMORY_ENABLED:
            return
        try:
            if self._client is None:
                if not _cfg.QDRANT_URL or not _cfg.QDRANT_API_KEY:
                    raise RuntimeError("QDRANT_URL and QDRANT_API_KEY are required")
                from qdrant_client import QdrantClient
                self._client = QdrantClient(url=_cfg.QDRANT_URL, api_key=_cfg.QDRANT_API_KEY)
            if self._qmodels is None:
                from qdrant_client.http import models as qmodels
                self._qmodels = qmodels
            self._ok = True
            logger.info("[vector_memory] Cloud Qdrant semantic memory enabled")
        except Exception as exc:
            logger.warning("[vector_memory] unavailable (%s); long-term recall is empty", exc)

    @property
    def available(self) -> bool:
        return self._ok

    def _ensure_collection(self, agent_id: str) -> str:
        name = collection_name(agent_id)
        if not self._client.collection_exists(name):
            self._client.create_collection(
                collection_name=name,
                vectors_config=self._qmodels.VectorParams(
                    size=_cfg.MEMORY_VECTOR_DIMENSIONS,
                    distance=self._qmodels.Distance.COSINE,
                ),
            )
        return name

    def _embed(self, text: str, task_type: str) -> Optional[list[float]]:
        if not self._ok:
            return None
        try:
            from src.core.budget import GOVERNOR
            if not GOVERNOR.can_afford("embedding", cost=1):
                return None
            vector = embed_content(text, task_type, _cfg.MEMORY_VECTOR_DIMENSIONS)
            if len(vector) != _cfg.MEMORY_VECTOR_DIMENSIONS:
                raise ValueError(f"embedding dimension {len(vector)} does not match configured dimension")
            GOVERNOR.record("embedding", _cfg.MEMORY_EMBEDDING_MODEL)
            return vector
        except ProviderFailureError:
            # This is terminal for the live simulation, not a recoverable
            # single-record indexing failure.  WorldEngine/Odin surface it.
            raise
        except Exception as exc:
            logger.warning("[vector_memory] embedding failed (%s)", exc)
            return None

    def index_records(self, agent_id: str, records: Iterable[MemoryRecord]) -> int:
        if not self._ok:
            return 0
        name = self._ensure_collection(agent_id)
        points = []
        for record in records:
            vector = self._embed(record.text, "RETRIEVAL_DOCUMENT")
            if vector is None:
                continue
            points.append(self._qmodels.PointStruct(id=record.id, vector=vector, payload=record.payload()))
        if not points:
            return 0
        try:
            self._client.upsert(collection_name=name, points=points, wait=True)
            self.prune_if_needed()
            return len(points)
        except Exception as exc:
            logger.warning("[vector_memory] upsert failed (%s)", exc)
            return 0

    @staticmethod
    def _payload_retention_score(payload: dict) -> float:
        """Retention is global, unlike semantic relevance which is query-specific."""
        importance = min(1.0, max(0.0, float(payload.get("importance", 0.0)) / 3.0))
        last_used = str(payload.get("last_recalled_at") or payload.get("date") or "")
        recency = math.exp(-VectorMemoryRetriever._days_ago(last_used) / 90.0)
        recall_count = max(0, int(payload.get("recall_count", 0)))
        popularity = min(1.0, math.log1p(recall_count) / math.log(11))
        return .50 * importance + .30 * recency + .20 * popularity

    @staticmethod
    def _estimated_point_bytes(payload: dict) -> int:
        """Conservative vector + serialized-payload estimate; JSON remains intact."""
        payload_bytes = len(json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
        return (_cfg.MEMORY_VECTOR_DIMENSIONS * 4) + payload_bytes + 128

    def _all_index_points(self) -> list[tuple[str, Any]]:
        points: list[tuple[str, Any]] = []
        collections = getattr(self._client.get_collections(), "collections", [])
        for collection in collections:
            name = getattr(collection, "name", str(collection))
            offset = None
            while True:
                batch, offset = self._client.scroll(
                    collection_name=name, limit=256, offset=offset,
                    with_payload=True, with_vectors=False,
                )
                # A Cloud cluster can contain other applications. Only points
                # carrying Valhalla's deterministic archive marker participate
                # in quota accounting or retention deletion.
                points.extend(
                    (name, point) for point in batch
                    if isinstance(getattr(point, "payload", None), dict)
                    and point.payload.get("source_hash")
                )
                if offset is None:
                    break
        return points

    def prune_if_needed(self) -> int:
        """Keep the rebuildable index below its configured storage budget.

        Cloud Qdrant's dashboard is authoritative for billed capacity. This
        conservative client-side estimate triggers ahead of that limit. Only
        the lowest-retention Qdrant points are removed; there is no parallel
        JSON long-term archive to consult or rebuild from.
        """
        if not self._ok:
            return 0
        try:
            points = self._all_index_points()
            limit = int(_cfg.MEMORY_MAX_STORAGE_GB * _GIB)
            threshold = int(limit * _cfg.MEMORY_STORAGE_PRUNE_THRESHOLD)
            target = int(limit * _cfg.MEMORY_STORAGE_PRUNE_TARGET)
            usage = sum(self._estimated_point_bytes(point.payload or {}) for _, point in points)
            if usage < threshold:
                return 0
            ranked = sorted(
                points,
                key=lambda item: self._payload_retention_score(item[1].payload or {}),
            )
            grouped: dict[str, list[int]] = {}
            removed = 0
            for name, point in ranked:
                if usage <= target:
                    break
                usage -= self._estimated_point_bytes(point.payload or {})
                grouped.setdefault(name, []).append(int(point.id))
                removed += 1
            for name, point_ids in grouped.items():
                self._client.delete(
                    collection_name=name,
                    points_selector=self._qmodels.PointIdsList(points=point_ids), wait=True,
                )
            if removed:
                logger.warning("[vector_memory] pruned %d low-retention vector memories to enforce quota", removed)
            return removed
        except Exception as exc:
            logger.warning("[vector_memory] quota check failed (%s); retaining index", exc)
            return 0

    def index_archive(self, agent_id: str, day: dict) -> int:
        """Persist a completed-day snapshot and discard superseded same-day points.

        The engine may archive once when the final activity starts and once at
        handoff. Upserting first, then deleting only no-longer-present records
        makes the later snapshot authoritative without a vulnerable delete-first
        window.
        """
        try:
            records = archive_records(agent_id, day)
            indexed = self.index_records(agent_id, records)
            if indexed != len(records):
                return indexed
            if not self._ok:
                return indexed
            date = str(day.get("date") or "")
            wanted = {record.hash for record in records}
            stale: list[int] = []
            name = collection_name(agent_id)
            offset = None
            while True:
                batch, offset = self._client.scroll(collection_name=name, limit=256, offset=offset,
                                                    with_payload=True, with_vectors=False)
                stale.extend(int(point.id) for point in batch
                             if str((point.payload or {}).get("date") or "") == date
                             and (point.payload or {}).get("source_hash") not in wanted)
                if offset is None:
                    break
            if stale:
                self._client.delete(collection_name=name,
                                    points_selector=self._qmodels.PointIdsList(points=stale), wait=True)
            return indexed
        except Exception as exc:
            logger.warning("[vector_memory] archive indexing failed (%s)", exc)
            return 0

    def collection_status(self, agent_id: str) -> dict:
        """Report one persona's durable-vector collection without a JSON comparison."""
        name = collection_name(agent_id)
        if not self._ok or not self._client.collection_exists(name):
            return {"available": self._ok, "collection": name, "indexed": 0}
        points, offset = [], None
        while True:
            batch, offset = self._client.scroll(collection_name=name, limit=256, offset=offset,
                                                with_payload=True, with_vectors=False)
            points.extend(batch)
            if offset is None:
                break
        return {"available": True, "collection": name, "indexed": len(points)}

    def clear_all_memory(self) -> dict:
        """Delete every non-empty Valhalla long-term-memory collection.

        A Qdrant cluster can be shared with unrelated applications, so the
        operation identifies a Valhalla collection by its durable
        ``source_hash`` payload marker before deleting it.  It does not read or
        mutate short-term JSON, checkpoints, plans, or live engine state.
        Empty collections contain no long-term memory and are deliberately
        retained rather than guessing ownership from their name.
        """
        if not self._ok:
            return {"available": False, "deleted_collections": [], "failed_collections": []}

        deleted: list[str] = []
        failed: list[str] = []
        try:
            collections = getattr(self._client.get_collections(), "collections", [])
        except Exception as exc:
            logger.warning("[vector_memory] could not list collections for memory clear (%s)", exc)
            return {"available": True, "deleted_collections": [], "failed_collections": ["<list failed>"]}

        for collection in collections:
            name = getattr(collection, "name", str(collection))
            try:
                points, _ = self._client.scroll(
                    collection_name=name, limit=1, with_payload=True, with_vectors=False,
                )
                is_valhalla_memory = any(
                    isinstance(getattr(point, "payload", None), dict)
                    and point.payload.get("source_hash")
                    for point in points
                )
                if not is_valhalla_memory:
                    continue
                self._client.delete_collection(collection_name=name)
                deleted.append(name)
            except Exception as exc:
                logger.warning("[vector_memory] could not clear collection '%s' (%s)", name, exc)
                failed.append(name)
        return {
            "available": True,
            "deleted_collections": deleted,
            "failed_collections": failed,
        }

    def delete_agent_memory(self, agent_id: str) -> bool:
        """Remove one agent's Qdrant collection without touching other agents."""
        name = collection_name(agent_id)
        if not self._ok:
            return False
        try:
            if self._client.collection_exists(name):
                self._client.delete_collection(name)
            return True
        except Exception as exc:
            logger.warning("[vector_memory] could not remove %s (%s)", name, exc)
            return False

    @staticmethod
    def _days_ago(date: str) -> int:
        try:
            return max(0, (datetime.now(timezone.utc).date() - datetime.fromisoformat(date).date()).days)
        except (TypeError, ValueError):
            return 0

    @classmethod
    def rank_scored_hits(cls, hits: Iterable[Any], k: int) -> list[tuple[float, Any]]:
        """Score query relevance, then select diverse top memories.

        Relevance is deliberately query-specific and is not persisted as a
        payload field. The returned score is 65% semantic similarity, 20%
        durable importance, and 15% recency.
        """
        hits = list(hits)
        if not hits:
            return []
        raw_scores = [float(getattr(hit, "score", 0.0)) for hit in hits]
        lo, hi = min(raw_scores), max(raw_scores)
        def semantic(score: float) -> float:
            return 1.0 if hi == lo else (score - lo) / (hi - lo)
        scored = []
        for hit in hits:
            payload = getattr(hit, "payload", {}) or {}
            importance = min(1.0, max(0.0, float(payload.get("importance", 0.0)) / 3.0))
            recency = math.exp(-cls._days_ago(str(payload.get("date", ""))) / _cfg.MEMORY_RECENCY_DECAY_DAYS)
            score = .65 * semantic(float(getattr(hit, "score", 0.0))) + .20 * importance + .15 * recency
            scored.append((score, hit))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        # Diversity: select at most one item from a day/kind pair before a
        # second from the same pair. This avoids one conversation dominating.
        selected, seen = [], set()
        for score, hit in scored:
            payload = getattr(hit, "payload", {}) or {}
            key = (payload.get("date"), payload.get("kind"))
            if key in seen and len(selected) < k:
                continue
            selected.append((score, hit)); seen.add(key)
            if len(selected) == k:
                return selected
        for score, hit in scored:
            if hit not in [selected_hit for _, selected_hit in selected]:
                selected.append((score, hit))
            if len(selected) == k:
                break
        return selected

    @classmethod
    def rank_hits(cls, hits: Iterable[Any], k: int) -> list[Any]:
        """Compatibility helper returning only the ranked memory hits."""
        return [hit for _, hit in cls.rank_scored_hits(hits, k)]

    def retrieve(self, agent_id: str, query: str, k: int = 5) -> List[str]:
        if not self._ok:
            return []
        vector = self._embed(query, "RETRIEVAL_QUERY")
        if vector is None:
            return []
        try:
            name = self._ensure_collection(agent_id)
            limit = max(k, k * _cfg.MEMORY_RETRIEVAL_CANDIDATE_MULTIPLIER)
            if hasattr(self._client, "query_points"):
                result = self._client.query_points(collection_name=name, query=vector, limit=limit)
                hits = result.points
            else:  # supports older qdrant-client releases
                hits = self._client.search(collection_name=name, query_vector=vector, limit=limit)
            ranked = self.rank_scored_hits(hits, k)
            chosen = [hit for _, hit in ranked]
            self._record_recall(name, chosen)
            return [
                f"[{hit.payload.get('date', '')}] ({hit.payload.get('kind', '')}; relevance={score:.2f}) "
                f"{hit.payload.get('text', '')}"
                for score, hit in ranked
            ]
        except Exception as exc:
            logger.warning("[vector_memory] search failed (%s); returning no RAG context", exc)
            return []

    def _record_recall(self, collection: str, hits: Iterable[Any]) -> None:
        """Promote memories that were actually useful in a semantic recall."""
        for hit in hits:
            payload = getattr(hit, "payload", {}) or {}
            updated = dict(payload)
            updated["recall_count"] = int(payload.get("recall_count", 0)) + 1
            updated["last_recalled_at"] = datetime.now(timezone.utc).date().isoformat()
            updated["retention_score"] = self._payload_retention_score(updated)
            try:
                self._client.set_payload(collection_name=collection, payload=updated, points=[int(hit.id)], wait=False)
            except Exception as exc:
                logger.debug("[vector_memory] recall-retention update failed (%s)", exc)

    def store(self, agent_id: str, text: str, *, kind: str = "observation", importance=None, date_str=None) -> None:
        """Store an explicit durable memory immediately in Qdrant."""
        date_str = date_str or datetime.now(timezone.utc).date().isoformat()
        memory_kind = kind if kind in _IMPORTANCE else "observation"
        self.index_records(agent_id, [MemoryRecord(
            agent_id, date_str, memory_kind, text,
            float(importance if importance is not None else _IMPORTANCE[memory_kind]),
        )])

    def rolling_summary(self, agent_id: str, days: int = 3, before_date=None) -> Optional[str]:
        if not self._ok:
            return None
        try:
            name = collection_name(agent_id)
            if not self._client.collection_exists(name):
                return None
            points, offset = [], None
            while True:
                batch, offset = self._client.scroll(collection_name=name, limit=256, offset=offset,
                                                    with_payload=True, with_vectors=False)
                points.extend(batch)
                if offset is None:
                    break
            cutoff = str(before_date or "9999-12-31")
            summaries = [p.payload for p in points if (p.payload or {}).get("kind") == "summary"
                         and str((p.payload or {}).get("date", "")) < cutoff]
            summaries.sort(key=lambda p: str(p.get("date", "")), reverse=True)
            if not summaries:
                return None
            return "\n".join(f"- {p.get('date', '')}: {p.get('text', '')}" for p in reversed(summaries[:days]))
        except Exception as exc:
            logger.warning("[vector_memory] rolling-summary retrieval failed (%s)", exc)
            return None
