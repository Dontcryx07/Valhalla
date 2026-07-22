"""
Short-term Memory -- per-agent, per-day detailed memory store.

Stores the full day's data (plan, events, conversations, world snapshots)
as a single JSON file per persona per simulation date.

File layout:
  data/Short_term_db/<persona_name>/<YYYY-MM-DD>.json

Implements MemoryStreamProtocol (from tick_graph.py) for tick-graph integration.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import tempfile
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel

from src.config import DATA_DIR
from src.core.log import get_logger

logger = get_logger(__name__)

# Directory for short-term memory files
SHORT_TERM_DIR = DATA_DIR / "Short_term_db"
_DAY_LOCKS_GUARD = threading.Lock()
_DAY_LOCKS: dict[Path, threading.RLock] = {}


# ---------------------------------------------------------------------------
# Data Models (internal)
# ---------------------------------------------------------------------------

class DayEvent(BaseModel):
    """One event that happened during the day."""
    timestamp: str              # ISO format with timezone
    type: str                   # action_started, action_completed, conversation, world_snapshot, observation
    action: Optional[str] = None
    outcome: Optional[str] = None
    success: Optional[bool] = None
    location: Optional[str] = None
    with_agent: Optional[str] = None
    participants: Optional[list[str]] = None
    topic: Optional[str] = None
    summary: Optional[str] = None
    messages: Optional[list[dict]] = None
    details: Optional[dict] = None


class ConversationEntry(BaseModel):
    """A conversation with another agent."""
    timestamp: str
    participants: list[str]
    messages: list[dict]
    summary: str


class WorldSnapshotEntry(BaseModel):
    """Periodic world state snapshot."""
    timestamp: str
    location: str
    nearby_agents: list[str]
    weather: Optional[str] = None
    details: Optional[dict] = None


class ShortTermDayData(BaseModel):
    """Complete day memory for one persona on one simulation date."""
    persona_name: str
    date: str
    day_plan: list[dict] = []
    events: list[DayEvent] = []
    conversations: list[ConversationEntry] = []
    world_snapshots: list[WorldSnapshotEntry] = []
    daily_summary: Optional[str] = None
    archived_to_long_term: bool = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _persona_dir(persona_name: str) -> Path:
    """Get the directory for a persona's short-term files."""
    safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", persona_name.strip()).strip("._-").lower() or "unknown"
    dir_path = SHORT_TERM_DIR / safe_name
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path


def _day_file_path(persona_name: str, date_str: str) -> Path:
    """Get the file path for a specific persona and date."""
    return _persona_dir(persona_name) / f"{date_str}.json"


def _day_lock(persona_name: str, date_str: str) -> threading.RLock:
    path = _day_file_path(persona_name, date_str).resolve()
    with _DAY_LOCKS_GUARD:
        return _DAY_LOCKS.setdefault(path, threading.RLock())


def _mutate_day_data(persona_name: str, date_str: str, mutate) -> None:
    """Serialize a complete same-day read/modify/write transaction."""
    with _day_lock(persona_name, date_str):
        data = _load_day_data(persona_name, date_str)
        if data is None:
            data = ShortTermDayData(persona_name=persona_name, date=date_str)
        mutate(data)
        _save_day_data(persona_name, date_str, data)


def date_from_simulation_time(time_str: str) -> str:
    """
    Extract simulation date from a time string like '2026-07-03 06:00'.
    Returns 'YYYY-MM-DD'.
    """
    # Expected format: "YYYY-MM-DD HH:MM" or just "YYYY-MM-DD HH:MM:SS"
    match = re.match(r"(\d{4}-\d{2}-\d{2})", time_str)
    if match:
        return match.group(1)
    # Fallback: try parsing
    try:
        dt = datetime.fromisoformat(time_str.replace("Z", "+00:00"))
        return dt.date().isoformat()
    except Exception:
        # Last resort: today's date (real)
        return datetime.now(timezone.utc).date().isoformat()


def _load_day_data(persona_name: str, date_str: str) -> Optional[ShortTermDayData]:
    """Load day data from JSON file."""
    path = _day_file_path(persona_name, date_str)
    if not path.exists():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        return ShortTermDayData.model_validate(raw)
    except Exception as e:
        logger.error("Failed to load short-term data for %s on %s: %s", persona_name, date_str, e)
        return None


def _save_day_data(persona_name: str, date_str: str, data: ShortTermDayData) -> None:
    """Atomically replace a day file so a process crash cannot leave partial JSON."""
    path = _day_file_path(persona_name, date_str)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = data.model_dump_json(indent=2)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.stem}.", suffix=".tmp", dir=path.parent, text=True,
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as temporary:
            temporary.write(payload)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)
    logger.debug("Saved short-term data for %s on %s to %s", persona_name, date_str, path)


# ---------------------------------------------------------------------------
# Public API -- MemoryStreamProtocol (for tick_graph.py)
# ---------------------------------------------------------------------------

def add_memory(agent_id: str, content: str, importance: Optional[int] = None, date_str: Optional[str] = None) -> None:
    """
    Add a memory entry for an agent.
    Stores as an event in the short-term file for the given (or current) date.
    """
    if date_str is None:
        date_str = datetime.now(timezone.utc).date().isoformat()
    data = _load_day_data(agent_id, date_str)
    if data is None:
        data = ShortTermDayData(persona_name=agent_id, date=date_str)
    data.events.append(DayEvent(
        timestamp=datetime.now(timezone.utc).isoformat(),
        type="memory",
        details={"content": content, "importance": importance}
    ))
    _save_day_data(agent_id, date_str, data)


def retrieve_memories(agent_id: str, query: str, k: int = 5, date_str: Optional[str] = None) -> list[str]:
    """Retrieve durable context through the Qdrant semantic-memory RAG layer."""
    from src.agents.Long_term import get_retriever
    return get_retriever().retrieve(agent_id, query, k)


# ---------------------------------------------------------------------------
# Public API -- Day Planner Integration
# ---------------------------------------------------------------------------

def save_day_plan(persona_name: str, date_str: str, plan: list[dict]) -> None:
    """Save the generated day plan at the start of the day."""
    data = _load_day_data(persona_name, date_str)
    if data is None:
        data = ShortTermDayData(persona_name=persona_name, date=date_str)
    data.day_plan = plan
    _save_day_data(persona_name, date_str, data)


def load_day_plan(persona_name: str, date_str: str) -> list[dict]:
    """Load the day plan for a given date."""
    data = _load_day_data(persona_name, date_str)
    return data.day_plan if data else []


# ---------------------------------------------------------------------------
# Single-file hygiene (one short-term file per agent; clean start each run)
# ---------------------------------------------------------------------------

def is_pristine_plan(persona_name: str, date_str: str) -> bool:
    """True if the day file holds a plan but none of the run-time record yet
    (no events, conversations, or world snapshots)."""
    data = _load_day_data(persona_name, date_str)
    return bool(
        data and data.day_plan
        and not data.events and not data.conversations and not data.world_snapshots
    )


def reset_day_runtime(persona_name: str, date_str: str) -> None:
    """Clear the run-time record (events / conversations / snapshots) but keep
    the plan, so a day starts fresh — the file holds only the plan again."""
    data = _load_day_data(persona_name, date_str)
    if data is None:
        return
    data.events = []
    data.conversations = []
    data.world_snapshots = []
    data.archived_to_long_term = False
    _save_day_data(persona_name, date_str, data)


def list_day_files(persona_name: str) -> list[str]:
    """All simulation dates that currently have a short-term file for this agent."""
    d = _persona_dir(persona_name)
    return sorted(p.stem for p in d.glob("*.json"))


def archive_copy_no_llm(persona_name: str, date_str: str) -> bool:
    """Index a completed operational day without making a summary LLM call.

    This is used only to clean up stray short-term files.  It never writes a
    local long-term JSON archive, and it retains the source file on failure.
    """
    data = _load_day_data(persona_name, date_str)
    if data is None:
        return False
    try:
        from src.agents.Long_term import get_retriever
        return bool(get_retriever().index_archive(persona_name, data.model_dump()))
    except Exception as exc:
        logger.warning("[Short_term] unable to index stray day %s for %s: %s", date_str, persona_name, exc)
        return False


def consolidate_to_single_day(persona_name: str, keep_date: str) -> list[str]:
    """Ensure the agent keeps only ONE short-term file (for `keep_date`).
    Any other day files are indexed into Qdrant (no LLM) and removed only
    after Qdrant confirms the write. Returns the dates safely removed.
    """
    moved: list[str] = []
    for date_str in list_day_files(persona_name):
        if date_str == keep_date:
            continue
        if archive_copy_no_llm(persona_name, date_str):
            clear_short_term_data(persona_name, date_str)
            moved.append(date_str)
    if moved:
        logger.info(
            "[Short_term] consolidated %s: indexed %d stray day file(s) into Qdrant (%s)",
            persona_name, len(moved), ", ".join(moved),
        )
    return moved


def append_event(persona_name: str, date_str: str, event: dict) -> None:
    """
    Append an event during the day.
    event dict should have: type, action, outcome, success, location, details, etc.
    """
    entry = dict(event)
    entry.setdefault("timestamp", datetime.now(timezone.utc).isoformat())
    _mutate_day_data(
        persona_name, date_str,
        lambda data: data.events.append(DayEvent(**entry)),
    )


def append_conversation(persona_name: str, date_str: str, conversation: dict) -> None:
    """
    Append a conversation during the day.
    conversation dict: participants, messages, summary
    """
    entry = dict(conversation)
    entry.setdefault("timestamp", datetime.now(timezone.utc).isoformat())
    _mutate_day_data(
        persona_name, date_str,
        lambda data: data.conversations.append(ConversationEntry(**entry)),
    )


def append_world_snapshot(persona_name: str, date_str: str, snapshot: dict) -> None:
    """
    Append a world state snapshot during the day.
    snapshot dict: location, nearby_agents, weather, details
    """
    entry = dict(snapshot)
    entry.setdefault("timestamp", datetime.now(timezone.utc).isoformat())
    _mutate_day_data(
        persona_name, date_str,
        lambda data: data.world_snapshots.append(WorldSnapshotEntry(**entry)),
    )


def get_yesterday_data(persona_name: str, today_date: str) -> Optional[ShortTermDayData]:
    """
    Get the full yesterday's data for a persona.
    today_date: 'YYYY-MM-DD' (simulation date)
    Returns yesterday's ShortTermDayData or None.
    """
    try:
        today_dt = datetime.fromisoformat(today_date).date()
        yesterday = (today_dt - timedelta(days=1)).isoformat()
    except Exception:
        # Fallback to real date
        yesterday = (datetime.now(timezone.utc).date() - timedelta(days=1)).isoformat()
    
    return _load_day_data(persona_name, yesterday)


def get_yesterday_summary(persona_name: str, today_date: str) -> Optional[str]:
    """
    Get a text summary of yesterday for the day planner prompt.
    Returns None if no yesterday data exists.
    """
    from src.agents.Long_term import get_retriever
    return get_retriever().rolling_summary(persona_name, days=1, before_date=today_date)


def get_relevant_memories(persona_name: str, today_date: str, query: str, k: int = 10) -> list[str]:
    """
    Get relevant long-term memories for model prompts through semantic search.
    """
    return retrieve_memories(persona_name, query, k, date_str=today_date)


# ---------------------------------------------------------------------------
# Public API -- End of Day Lifecycle
# ---------------------------------------------------------------------------

async def generate_daily_summary(persona_name: str, date_str: str) -> str:
    """
    Generate an LLM summary of the day's events for long-term storage.
    Uses the same gemini_client as day_planner.
    """
    from src.llm.gemini_client import call_gemini
    from pydantic import BaseModel
    from typing import List
    
    class SummaryOutput(BaseModel):
        summary: str
        key_events: List[str]
    
    data = _load_day_data(persona_name, date_str)
    if data is None or not data.events:
        return f"No events recorded for {persona_name} on {date_str}."
    
    # Build event text
    event_lines = []
    for e in data.events:
        if e.type == "action_completed":
            event_lines.append(f"- {e.action} at {e.location or 'unknown'} (outcome: {e.outcome}, success: {e.success})")
        elif e.type == "conversation":
            event_lines.append(f"- Conversation with {e.with_agent or ', '.join(e.participants) if e.participants else 'unknown'}: {e.summary or e.topic}")
        elif e.type == "observation":
            event_lines.append(f"- Observed: {e.details}")
    
    event_text = "\n".join(event_lines)
    
    system_prompt = (
        "You are summarizing one day in the life of a simulated college student agent. "
        "You will be provided the Initial day plan, the conversations the agent had throughout and some extra stuff. "
        "Write a concise summary of about 100 words prioritizing the most important:\n"
        "- High-priority tasks and whether they were completed\n"
        "- Important events and social interactions\n"
        "- Significant conversations and decisions\n"
        "- Anything likely to matter in future days\n"
        "Only focus on routine low-value chores unless no important event happened that day.\n"
        "Also list 3-5 key events as bullet points."
    )
    user_prompt = (
        f"Persona: {persona_name}\n"
        f"Date: {date_str}\n"
        f"Day plan: {len(data.day_plan)} scheduled actions\n\n"
        f"Initial day plan:\n" +
        "\n".join(f"  - {a.get('start','?')}-{a.get('end','?')} {a.get('action','?')} at {a.get('location_id','?')}" for a in data.day_plan[:15]) +
        f"\n\nEvents:\n{event_text}\n\n"
        f"Conversations ({len(data.conversations)}):\n" +
        "\n".join(f"- {c.summary} (with {', '.join(c.participants)})" for c in data.conversations[:5]) +
        "\n\nGenerate the daily summary (~100 words) and key events list."
    )
    
    try:
        result = call_gemini(system_prompt, user_prompt, SummaryOutput, "default", temperature=0.5)
        # Format for long-term storage
        key_events_text = "\n".join(f"- {evt}" for evt in result.key_events)
        return f"Summary: {result.summary}\nKey events:\n{key_events_text}"
    except Exception as e:
        logger.error("Failed to generate LLM daily summary for %s on %s: %s", persona_name, date_str, e)
        # Fallback: programmatic summary
        parts = [f"{persona_name} on {date_str}:"]
        for e in data.events:
            if e.type == "action_completed":
                parts.append(f"  - {e.action} ({e.outcome}, success={e.success})")
        return "\n".join(parts)


async def finalize_day(persona_name: str, date_str: str) -> dict:
    """
    End-of-day processing:
    1. Generate LLM summary of the day
    2. Mark as archived
    3. Return data ready for long-term storage
    """
    data = _load_day_data(persona_name, date_str)
    if data is None:
        return {"summary": f"No data for {persona_name} on {date_str}", "full_data": None}
    
    # Generate summary
    daily_summary = await generate_daily_summary(persona_name, date_str)
    data.daily_summary = daily_summary
    data.archived_to_long_term = True
    
    # Save final version with summary
    _save_day_data(persona_name, date_str, data)
    
    # Return data for long-term storage
    return {
        "summary": daily_summary,
        "full_data": data.model_dump()
    }


async def archive_to_long_term(persona_name: str, date_str: str) -> dict:
    """
    End-of-day archival: generate a daily summary and persist its durable
    records directly to Qdrant.  No Long_term_db JSON file is created.

    Returns the archive payload dict with keys: summary, persona_name, date,
    event_count, conversation_count.
    """
    result = await finalize_day(persona_name, date_str)
    data = result.get("full_data")
    if data is None:
        return {"summary": f"No data for {persona_name} on {date_str}"}

    # Offline/local runs intentionally disable the Qdrant-backed long-term
    # store.  Treat this as an explicit, successful skip rather than trying a
    # known-unavailable retriever and turning every midnight handoff into an
    # archive failure.
    from src import config as _cfg
    if not _cfg.SEMANTIC_MEMORY_ENABLED:
        logger.info("[Short_term] semantic archival disabled; skipped %s on %s", persona_name, date_str)
        return {
            "summary": result.get("summary", ""),
            "persona_name": persona_name,
            "date": date_str,
            "event_count": len(data.get("events", [])),
            "conversation_count": len(data.get("conversations", [])),
            "skipped": True,
        }

    try:
        from src.agents.Long_term import get_retriever
        retriever = get_retriever()
        indexed = await asyncio.to_thread(retriever.index_archive, persona_name, data)
        if indexed <= 0 and data.get("daily_summary"):
            raise RuntimeError("Qdrant did not acknowledge any durable memory records")
    except Exception as exc:
        # The short-term source remains in place so an operator can retry.  Do
        # not silently claim archival success when Qdrant is the only durable
        # long-term store.
        logger.error("[Short_term] Qdrant archive failed for %s: %s", persona_name, exc)
        raise

    logger.info(
        "[Short_term] archived %s on %s into Qdrant (%d events, %d conversations)",
        persona_name, date_str,
        len(data.get("events", [])), len(data.get("conversations", [])),
    )

    return {
        "summary": result.get("summary", ""),
        "persona_name": persona_name,
        "date": date_str,
        "event_count": len(data.get("events", [])),
        "conversation_count": len(data.get("conversations", [])),
    }


def clear_short_term_data(persona_name: str, date_str: str) -> bool:
    """Delete the short-term file after successful long-term archival.

    Not called automatically — archival keeps the short-term copy in place so
    the running day is not disturbed. Exposed for maintenance/cleanup use.
    """
    path = _day_file_path(persona_name, date_str)
    if path.exists():
        path.unlink()
        logger.info("Cleared short-term data for %s on %s", persona_name, date_str)
        return True
    return False


# ---------------------------------------------------------------------------
# MemoryStreamProtocol adapter
# ---------------------------------------------------------------------------

class ShortTermMemoryStream:
    """
    Adapter that wraps Short_term.py's module-level functions as
    MemoryStreamProtocol (defined in tick_graph.py) with sim-date awareness.

    Usage:
        stream = ShortTermMemoryStream("2026-07-03")
        # Tick graph nodes call:
        stream.add_memory(agent_id, content, importance)
        stream.retrieve_memories(agent_id, query, k)
    """

    def __init__(self, initial_date: str = "") -> None:
        self._date_str: str = initial_date or datetime.now(timezone.utc).date().isoformat()

    @property
    def date_str(self) -> str:
        return self._date_str

    def set_date(self, date_str: str) -> None:
        """Update the simulation date (called before each tick)."""
        self._date_str = date_str

    def add_memory(self, agent_id: str, content: str, importance: Optional[int] = None) -> None:
        add_memory(agent_id, content, importance, date_str=self._date_str)

    def retrieve_memories(self, agent_id: str, query: str, k: int = 5) -> list[str]:
        return retrieve_memories(agent_id, query, k, date_str=self._date_str)
