"""
Short-term Memory -- per-agent, per-day detailed memory store.

Stores the full day's data (plan, events, conversations, world snapshots)
as a single JSON file per persona per simulation date.

File layout:
  data/Short_term_db/<persona_name>/<YYYY-MM-DD>.json

Implements MemoryStreamProtocol (from tick_graph.py) for tick-graph integration.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel

from src.config import DATA_DIR
from src.core.log import get_logger

logger = get_logger(__name__)

# Directory for short-term memory files
SHORT_TERM_DIR = DATA_DIR / "Short_term_db"


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
    """Save day data to JSON file."""
    path = _day_file_path(persona_name, date_str)
    path.write_text(data.model_dump_json(indent=2), encoding="utf-8")
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
    """
    Retrieve relevant memories for an agent using keyword matching.
    Searches the given date (or real today) and yesterday by default.
    """
    results: list[str] = []
    query_lower = query.lower()
    
    if date_str is None:
        today = datetime.now(timezone.utc).date()
    else:
        try:
            today = datetime.fromisoformat(date_str).date()
        except Exception:
            today = datetime.now(timezone.utc).date()
    
    for days_back in range(2):
        check_date = (today - timedelta(days=days_back)).isoformat()
        data = _load_day_data(agent_id, check_date)
        if data is None:
            continue
        # Search events
        for event in data.events:
            event_text = json.dumps(event.model_dump(), default=str).lower()
            if query_lower in event_text:
                results.append(f"[{event.timestamp}] {event.type}: {event.details or event.action or event.outcome or event.summary}")
        # Search conversations
        for conv in data.conversations:
            conv_text = json.dumps(conv.model_dump(), default=str).lower()
            if query_lower in conv_text:
                results.append(f"[{conv.timestamp}] Conversation with {', '.join(conv.participants)}: {conv.summary}")
    
    return results[:k]


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


def append_event(persona_name: str, date_str: str, event: dict) -> None:
    """
    Append an event during the day.
    event dict should have: type, action, outcome, success, location, details, etc.
    """
    data = _load_day_data(persona_name, date_str)
    if data is None:
        data = ShortTermDayData(persona_name=persona_name, date=date_str)
    
    # Add timestamp if not provided
    if "timestamp" not in event:
        event["timestamp"] = datetime.now(timezone.utc).isoformat()
    
    data.events.append(DayEvent(**event))
    _save_day_data(persona_name, date_str, data)


def append_conversation(persona_name: str, date_str: str, conversation: dict) -> None:
    """
    Append a conversation during the day.
    conversation dict: participants, messages, summary
    """
    data = _load_day_data(persona_name, date_str)
    if data is None:
        data = ShortTermDayData(persona_name=persona_name, date=date_str)
    
    if "timestamp" not in conversation:
        conversation["timestamp"] = datetime.now(timezone.utc).isoformat()
    
    data.conversations.append(ConversationEntry(**conversation))
    _save_day_data(persona_name, date_str, data)


def append_world_snapshot(persona_name: str, date_str: str, snapshot: dict) -> None:
    """
    Append a world state snapshot during the day.
    snapshot dict: location, nearby_agents, weather, details
    """
    data = _load_day_data(persona_name, date_str)
    if data is None:
        data = ShortTermDayData(persona_name=persona_name, date=date_str)
    
    if "timestamp" not in snapshot:
        snapshot["timestamp"] = datetime.now(timezone.utc).isoformat()
    
    data.world_snapshots.append(WorldSnapshotEntry(**snapshot))
    _save_day_data(persona_name, date_str, data)


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
    yesterday_data = get_yesterday_data(persona_name, today_date)
    if yesterday_data is None:
        return None
    
    if yesterday_data.daily_summary:
        return yesterday_data.daily_summary
    
    # Fallback: generate a quick summary from events
    if not yesterday_data.events:
        return "No recorded events from yesterday."
    
    parts = [f"Yesterday ({yesterday_data.date}) summary for {yesterday_data.persona_name}:"]
    
    # Key events
    action_events = [e for e in yesterday_data.events if e.type in ("action_started", "action_completed")]
    for e in action_events[:5]:  # Limit
        if e.type == "action_completed":
            parts.append(f"  - Completed: {e.action} at {e.location} (outcome: {e.outcome}, success: {e.success})")
        else:
            parts.append(f"  - Started: {e.action} at {e.location}")
    
    # Conversations
    for c in yesterday_data.conversations[:3]:
        parts.append(f"  - Talked with {', '.join(c.participants)}: {c.summary}")
    
    return "\n".join(parts)


def get_relevant_memories(persona_name: str, today_date: str, query: str, k: int = 10) -> list[str]:
    """
    Get relevant memories for the day planner prompt.
    Loads yesterday + today's data and keyword-searches.
    """
    results: list[str] = []
    query_lower = query.lower()
    
    # Check today and yesterday
    for date_str in [today_date]:
        yesterday_data = get_yesterday_data(persona_name, today_date)
        if yesterday_data:
            # Yesterday's data
            for event in yesterday_data.events:
                event_text = json.dumps(event.model_dump(), default=str).lower()
                if query_lower in event_text:
                    results.append(f"[{event.timestamp}] {event.type}: {event.action or event.outcome or event.summary}")
            for conv in yesterday_data.conversations:
                conv_text = json.dumps(conv.model_dump(), default=str).lower()
                if query_lower in conv_text:
                    results.append(f"[{conv.timestamp}] Conversation: {conv.summary}")
    
    return results[:k]


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
        "Write a concise paragraph (3-5 sentences) capturing the key activities, "
        "social interactions, and outcomes. Be specific about locations, people, and results. "
        "Also list 3-5 key events as bullet points."
    )
    user_prompt = (
        f"Persona: {persona_name}\n"
        f"Date: {date_str}\n"
        f"Day plan: {len(data.day_plan)} scheduled actions\n\n"
        f"Events:\n{event_text}\n\n"
        f"Conversations ({len(data.conversations)}):\n" +
        "\n".join(f"- {c.summary} (with {', '.join(c.participants)})" for c in data.conversations[:5]) +
        "\n\nGenerate the daily summary and key events list."
    )
    
    try:
        result = call_gemini(system_prompt, user_prompt, SummaryOutput, "simple", temperature=0.5)
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


# Long-term archive directory
LONG_TERM_DIR = DATA_DIR / "Long_term_db"


async def archive_to_long_term(persona_name: str, date_str: str) -> dict:
    """
    End-of-day archival: generate daily summary, save a copy to Long_term_db,
    and mark the short-term entry as archived.

    Returns the archive payload dict with keys: summary, persona_name, date,
    event_count, conversation_count.
    """
    result = await finalize_day(persona_name, date_str)
    data = result.get("full_data")
    if data is None:
        return {"summary": f"No data for {persona_name} on {date_str}"}

    # Write to Long_term_db
    safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", persona_name.strip()).strip("._-").lower() or "unknown"
    archive_dir = LONG_TERM_DIR / safe_name
    archive_dir.mkdir(parents=True, exist_ok=True)
    archive_path = archive_dir / f"{date_str}.json"
    archive_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    logger.info(
        "[Short_term] archived %s on %s to Long_term_db (%d events, %d conversations)",
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
    """Delete the short-term file after successful long-term archival."""
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