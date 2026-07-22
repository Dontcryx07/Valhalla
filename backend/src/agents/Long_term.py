"""Qdrant-backed long-term memory interface.

Long-term agent memory is stored only in Qdrant.  Short-term JSON files remain
the operational record for the active simulation day; they are summarized and
indexed at handoff, then removed.  This module deliberately has no JSON
archive reader or keyword-search fallback.
"""
from __future__ import annotations

from typing import List, Optional, Protocol, runtime_checkable

from src.core.log import get_logger

logger = get_logger(__name__)


@runtime_checkable
class MemoryRetriever(Protocol):
    """Contract consumed by planning, conversation, and tick cognition."""

    def store(self, agent_id: str, text: str, *, kind: str = "observation",
              importance: Optional[float] = None, date_str: Optional[str] = None) -> None: ...

    def retrieve(self, agent_id: str, query: str, k: int = 5) -> List[str]: ...

    def rolling_summary(self, agent_id: str, days: int = 3,
                        before_date: Optional[str] = None) -> Optional[str]: ...


_retriever_singleton: Optional[MemoryRetriever] = None


def get_retriever() -> MemoryRetriever:
    """Return the process-wide Qdrant retriever.

    The retriever is intentionally fail-soft for reads: an unavailable cloud
    service yields no recalled context rather than allowing a memory outage to
    stop the simulation.  It never reads a local long-term JSON archive.
    """
    global _retriever_singleton
    if _retriever_singleton is None:
        from src.agents.vector_memory import VectorMemoryRetriever
        _retriever_singleton = VectorMemoryRetriever()
        logger.info("[Long_term] using Qdrant-only long-term memory")
    return _retriever_singleton
