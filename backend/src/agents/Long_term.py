"""
Long-term memory -- cross-day recall for agents.

This module gives agents a memory that persists across simulation days and is
actually consulted when they plan their day and when they talk to each other.

Two things live here:

  1. A small, stable *retrieval interface* (`MemoryRetriever`) that every
     backend must satisfy: store a memory, retrieve the most relevant ones for
     an agent, and produce a short rolling summary of recent days. Callers
     (day planner, conversation) only ever touch this interface, so the
     backend can be swapped without touching them.

  2. A default, dependency-free backend (`KeywordMemoryRetriever`) that reads
     the per-day JSON archives already written by Short_term.py and ranks past
     memories by keyword overlap + recency + importance. This is offline, fast,
     and always available.

A second backend (Qdrant + Gemini embeddings) plugs in behind the same
interface and is selected via config (`MEMORY_BACKEND = "vector"`); it falls
back to the keyword backend automatically if the vector stack is unavailable.

Use `get_retriever()` to obtain the configured backend.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List, Optional, Protocol, runtime_checkable

from src.config import DATA_DIR, MEMORY_BACKEND
from src.core.log import get_logger

logger = get_logger(__name__)

SHORT_TERM_DIR = DATA_DIR / "Short_term_db"
LONG_TERM_DIR = DATA_DIR / "Long_term_db"

# Default importance weights per memory kind (higher = more salient).
_KIND_IMPORTANCE = {
    "summary": 3.0,
    "conversation": 2.0,
    "action": 1.0,
    "observation": 0.5,
    "memory": 1.0,
}

# Tiny stop-word set so keyword overlap focuses on meaningful tokens.
_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "to", "of", "in", "on", "at", "for",
    "with", "is", "are", "was", "were", "be", "been", "it", "this", "that",
    "i", "you", "he", "she", "they", "we", "my", "his", "her", "their", "do",
    "did", "will", "would", "should", "could", "what", "next", "day", "time",
}


def _safe_name(name: str) -> str:
    """Mirror Short_term.py's persona-dir sanitization."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name.strip()).strip("._-").lower() or "unknown"


def _tokenize(text: str) -> set:
    tokens = re.findall(r"[a-zA-Z0-9]+", text.lower())
    return {t for t in tokens if len(t) > 2 and t not in _STOPWORDS}


# ---------------------------------------------------------------------------
# Memory item + retrieval interface
# ---------------------------------------------------------------------------

class MemoryItem:
    """One recallable memory, assembled from an archived/short-term entry."""

    __slots__ = ("agent_id", "date", "kind", "text", "importance", "days_ago")

    def __init__(self, agent_id: str, date: str, kind: str, text: str,
                 importance: float, days_ago: int) -> None:
        self.agent_id = agent_id
        self.date = date
        self.kind = kind
        self.text = text
        self.importance = importance
        self.days_ago = days_ago

    def format(self) -> str:
        return f"[{self.date}] ({self.kind}) {self.text}"


@runtime_checkable
class MemoryRetriever(Protocol):
    """The stable contract every memory backend implements."""

    def store(self, agent_id: str, text: str, *, kind: str = "observation",
              importance: Optional[float] = None, date_str: Optional[str] = None) -> None:
        ...

    def retrieve(self, agent_id: str, query: str, k: int = 5) -> List[str]:
        ...

    def rolling_summary(self, agent_id: str, days: int = 3,
                        before_date: Optional[str] = None) -> Optional[str]:
        ...


# ---------------------------------------------------------------------------
# Keyword + recency backend (default, offline)
# ---------------------------------------------------------------------------

class KeywordMemoryRetriever:
    """Reads the per-day JSON archives and ranks memories by
    keyword-overlap + recency + importance. No external dependencies."""

    def __init__(self, short_term_dir: Path = SHORT_TERM_DIR,
                 long_term_dir: Path = LONG_TERM_DIR) -> None:
        self.short_term_dir = short_term_dir
        self.long_term_dir = long_term_dir

    # -- writing -----------------------------------------------------------
    def store(self, agent_id: str, text: str, *, kind: str = "observation",
              importance: Optional[float] = None, date_str: Optional[str] = None) -> None:
        """Append a memory into today's short-term file (archived at day end).

        Delegates to Short_term so there is a single writer of that file.
        """
        from src.agents import Short_term
        if date_str is None:
            date_str = datetime.now(timezone.utc).date().isoformat()
        Short_term.append_event(agent_id, date_str, {
            "type": kind if kind in ("observation", "memory") else "memory",
            "summary": text,
            "details": {"content": text, "importance": importance, "kind": kind},
        })

    # -- reading -----------------------------------------------------------
    def _gather_items(self, agent_id: str, ref_date: Optional[str]) -> List[MemoryItem]:
        try:
            ref = datetime.fromisoformat(ref_date).date() if ref_date else datetime.now(timezone.utc).date()
        except Exception:
            ref = datetime.now(timezone.utc).date()

        def _days_ago(date_str: str) -> int:
            try:
                return max(0, (ref - datetime.fromisoformat(date_str).date()).days)
            except Exception:
                return 0

        items: List[MemoryItem] = []
        safe = _safe_name(agent_id)

        # 1) Long-term: a single compressed memory.json with a list of days.
        mem_file = self.long_term_dir / safe / "memory.json"
        if mem_file.exists():
            try:
                mem = json.loads(mem_file.read_text(encoding="utf-8"))
                for day in mem.get("days", []):
                    date_str = day.get("date", "")
                    da = _days_ago(date_str)
                    if day.get("summary"):
                        items.append(MemoryItem(agent_id, date_str, "summary", str(day["summary"]),
                                                _KIND_IMPORTANCE["summary"], da))
                    for conv in day.get("conversations", []):
                        parts = ", ".join(conv.get("participants", []))
                        text = f"Talked with {parts}: {conv.get('summary', '')}"
                        items.append(MemoryItem(agent_id, date_str, "conversation", text,
                                                _KIND_IMPORTANCE["conversation"], da))
            except Exception:
                pass

        # 2) Short-term: today's (and any not-yet-archived) day files.
        st_dir = self.short_term_dir / safe
        if st_dir.is_dir():
            for path in sorted(st_dir.glob("*.json")):
                try:
                    data = json.loads(path.read_text(encoding="utf-8"))
                except Exception:
                    continue
                date_str = data.get("date", path.stem)
                da = _days_ago(date_str)
                if data.get("daily_summary"):
                    items.append(MemoryItem(agent_id, date_str, "summary", str(data["daily_summary"]),
                                            _KIND_IMPORTANCE["summary"], da))
                for conv in data.get("conversations", []):
                    parts = ", ".join(conv.get("participants", []))
                    text = f"Talked with {parts}: {conv.get('summary', '')}"
                    items.append(MemoryItem(agent_id, date_str, "conversation", text,
                                            _KIND_IMPORTANCE["conversation"], da))
                for ev in data.get("events", []):
                    etype = ev.get("type", "")
                    if etype == "action_completed":
                        text = f"{ev.get('action', '')} at {ev.get('location', '')}"
                    elif etype in ("observation", "memory"):
                        text = ev.get("summary") or str(ev.get("details", ""))
                    else:
                        continue
                    if text.strip():
                        items.append(MemoryItem(agent_id, date_str, etype or "event", text,
                                                _KIND_IMPORTANCE.get(etype, 0.5), da))
        return items

    def retrieve(self, agent_id: str, query: str, k: int = 5) -> List[str]:
        items = self._gather_items(agent_id, ref_date=None)
        if not items:
            return []
        q_tokens = _tokenize(query or "")

        def score(item: MemoryItem) -> float:
            recency = 1.0 / (1.0 + item.days_ago)
            if not q_tokens:
                return recency + 0.1 * item.importance
            overlap = len(q_tokens & _tokenize(item.text))
            return overlap * 2.0 + item.importance * 0.5 + recency

        # If there is a query but nothing overlaps, prefer recent high-value items.
        scored = sorted(items, key=score, reverse=True)
        if q_tokens:
            with_overlap = [it for it in scored if q_tokens & _tokenize(it.text)]
            chosen = with_overlap[:k] if with_overlap else scored[:k]
        else:
            chosen = scored[:k]
        return [it.format() for it in chosen]

    def rolling_summary(self, agent_id: str, days: int = 3,
                        before_date: Optional[str] = None) -> Optional[str]:
        items = [it for it in self._gather_items(agent_id, ref_date=before_date)
                 if it.kind == "summary"]
        if not items:
            return None
        items.sort(key=lambda it: it.days_ago)
        recent = items[:days]
        if not recent:
            return None
        return "\n".join(f"- {it.date}: {it.text}" for it in reversed(recent))


# ---------------------------------------------------------------------------
# Backend factory
# ---------------------------------------------------------------------------

_retriever_singleton: Optional[MemoryRetriever] = None


def get_retriever(backend: Optional[str] = None) -> MemoryRetriever:
    """Return the configured memory backend (cached).

    backend: "keyword" (default) or "vector". Falls back to keyword if the
    vector stack is unavailable.
    """
    global _retriever_singleton
    if _retriever_singleton is not None and backend is None:
        return _retriever_singleton

    choice = (backend or MEMORY_BACKEND or "keyword").lower()
    retriever: MemoryRetriever

    if choice == "vector":
        try:
            from src.agents.vector_memory import VectorMemoryRetriever  # noqa: WPS433
            retriever = VectorMemoryRetriever(fallback=KeywordMemoryRetriever())
            logger.info("[Long_term] using vector memory backend (Qdrant + embeddings)")
        except Exception as e:
            logger.warning(
                "[Long_term] vector backend unavailable (%s) -- falling back to keyword", e
            )
            retriever = KeywordMemoryRetriever()
    else:
        retriever = KeywordMemoryRetriever()
        logger.info("[Long_term] using keyword memory backend")

    if backend is None:
        _retriever_singleton = retriever
    return retriever


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    from src.core.log import setup_logging
    setup_logging(run_id="long_term_test", console=True)

    parser = argparse.ArgumentParser(description="Long-term memory retrieval test")
    parser.add_argument("agent", help="agent_id, e.g. parv_singla")
    parser.add_argument("query", nargs="?", default="", help="search query")
    parser.add_argument("-k", type=int, default=5)
    args = parser.parse_args()

    r = get_retriever("keyword")
    print(f"\nTop {args.k} memories for '{args.agent}' matching '{args.query}':\n")
    for line in r.retrieve(args.agent, args.query, k=args.k):
        print("  " + line)
    print("\nRolling summary (last 3 days):")
    print(r.rolling_summary(args.agent, days=3) or "  (none)")
