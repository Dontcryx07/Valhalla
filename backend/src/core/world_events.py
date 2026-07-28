"""Deterministic, data-driven campus events and safe plan opportunities.

The event calendar is deliberately independent of the LLM.  It makes an
attendance decision from a persona, its social context, schedule conflicts,
and a seeded tie-breaker, then edits only an entirely-flexible time window.
This keeps festivals and interruptions lively without allowing them to erase
classes, meals, sleep, exams, or an in-progress route.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

from pydantic import BaseModel, Field, field_validator, model_validator

from src.config import ENVIRONMENT_DIR
from src.core.log import get_logger

logger = get_logger(__name__)

HARD_COMMITMENT_WORDS = (
    "class", "lecture", "lab", "exam", "quiz", "sleep", "breakfast",
    "lunch", "dinner", "medical", "appointment", "deadline",
)


def _minutes(value: str) -> int:
    hour, minute = value.split(":", 1)
    result = int(hour) * 60 + int(minute)
    if not 0 <= result <= 24 * 60:
        raise ValueError("time must be between 00:00 and 24:00")
    return result


class WorldEvent(BaseModel):
    """One scheduled campus event, loaded from ``events_calendar.json``."""

    id: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]*$")
    name: str = Field(min_length=3, max_length=120)
    description: str = Field(min_length=8, max_length=600)
    category: str = Field(min_length=3, max_length=40)
    date: str = Field(pattern=r"^\d{4}-\d{2}-\d{2}$")
    start_time: str = Field(pattern=r"^\d{2}:\d{2}$")
    end_time: str = Field(pattern=r"^\d{2}:\d{2}$")
    location_id: str = Field(min_length=1)
    capacity: int = Field(ge=1, le=500)
    interest_tags: List[str] = Field(default_factory=list, max_length=12)
    eligibility_tags: List[str] = Field(default_factory=list, max_length=12)
    energy_effect: float = Field(default=0.0, ge=-0.4, le=0.4)
    emotion_effect: float = Field(default=0.0, ge=-0.4, le=0.4)
    relationship_effect: float = Field(default=0.0, ge=-0.1, le=0.1)
    announcement: str = Field(min_length=3, max_length=280)

    @field_validator("start_time", "end_time")
    @classmethod
    def validate_time(cls, value: str) -> str:
        _minutes(value)
        return value

    @model_validator(mode="after")
    def validate_duration(self) -> "WorldEvent":
        if _minutes(self.end_time) <= _minutes(self.start_time):
            raise ValueError("event end_time must be after start_time")
        return self

    @property
    def duration_minutes(self) -> int:
        return _minutes(self.end_time) - _minutes(self.start_time)

    def is_active(self, date: str, hhmm: str) -> bool:
        now = _minutes(hhmm)
        return self.date == date and _minutes(self.start_time) <= now < _minutes(self.end_time)

    def is_upcoming(self, date: str, hhmm: str, within_minutes: int = 180) -> bool:
        now = _minutes(hhmm)
        return self.date == date and now < _minutes(self.start_time) <= now + within_minutes


class EventCalendar(BaseModel):
    schema_version: int = 1
    events: List[WorldEvent] = Field(default_factory=list)


class EventAttendance(BaseModel):
    event_id: str
    agent_id: str
    score: float
    reason: str


class WorldEventManager:
    """Loads events, chooses bounded attendance, and creates UI snapshots."""

    def __init__(self, path: Optional[Path] = None) -> None:
        self.path = path or ENVIRONMENT_DIR / "events_calendar.json"
        self.calendar = self._load()
        self._attendance: Dict[str, List[str]] = {}

    def _load(self) -> EventCalendar:
        if not self.path.exists():
            logger.warning("[Events] calendar missing at %s; events disabled", self.path)
            return EventCalendar()
        try:
            loaded = EventCalendar.model_validate_json(self.path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.error("[Events] invalid calendar %s: %s; events disabled", self.path, exc)
            return EventCalendar()
        ids = [event.id for event in loaded.events]
        if len(ids) != len(set(ids)):
            logger.error("[Events] duplicate event IDs in %s; events disabled", self.path)
            return EventCalendar()
        logger.info("[Events] loaded %d scheduled events", len(loaded.events))
        return loaded

    def for_date(self, date: str) -> List[WorldEvent]:
        return [event for event in self.calendar.events if event.date == date]

    def snapshot(self, date: str, hhmm: str) -> Dict[str, List[Dict[str, Any]]]:
        def render(event: WorldEvent) -> Dict[str, Any]:
            attendees = self._attendance.get(event.id, [])
            return {
                "id": event.id, "name": event.name, "description": event.description,
                "category": event.category, "start_time": event.start_time,
                "end_time": event.end_time, "location_id": event.location_id,
                "capacity": event.capacity, "attendee_ids": attendees,
                "attendance_count": len(attendees), "announcement": event.announcement,
            }
        events = self.for_date(date)
        return {
            "active": [render(event) for event in events if event.is_active(date, hhmm)],
            "upcoming": [render(event) for event in events if event.is_upcoming(date, hhmm)],
        }

    def restore_attendance(self, attendance: Dict[str, List[str]]) -> None:
        self._attendance = {
            str(event_id): [str(agent_id) for agent_id in agent_ids]
            for event_id, agent_ids in (attendance or {}).items()
        }

    def attendance_snapshot(self) -> Dict[str, List[str]]:
        return {event_id: list(agent_ids) for event_id, agent_ids in self._attendance.items()}

    @staticmethod
    def _is_flexible(action: Dict[str, Any]) -> bool:
        text = str(action.get("action", "")).lower()
        return not any(word in text for word in HARD_COMMITMENT_WORDS)

    @staticmethod
    def _persona_text(persona: Dict[str, Any]) -> str:
        return " ".join(str(persona.get(key, "")) for key in (
            "innate", "lifestyle", "hobbies", "goals", "Branch", "interests",
        )).lower()

    @staticmethod
    def _deterministic_noise(date: str, event_id: str, agent_id: str) -> float:
        digest = hashlib.sha256(f"{date}|{event_id}|{agent_id}".encode()).digest()
        return (digest[0] / 255.0 - 0.5) * 0.16

    def _candidate_score(
        self,
        event: WorldEvent,
        date: str,
        agent_id: str,
        persona: Dict[str, Any],
        social_score: float,
    ) -> Tuple[float, str]:
        text = self._persona_text(persona)
        tags = [tag.lower() for tag in event.interest_tags]
        matched = [tag for tag in tags if tag in text]
        eligible = [tag.lower() for tag in event.eligibility_tags]
        if eligible and not any(tag in text for tag in eligible):
            return 0.0, "does not meet the event's interest eligibility"
        score = 0.35 + min(0.42, 0.14 * len(matched))
        score += max(-0.12, min(0.12, (social_score - 0.5) * 0.24))
        score += self._deterministic_noise(date, event.id, agent_id)
        reason = ", ".join(matched[:3]) if matched else "a low-stakes campus opportunity"
        return round(score, 3), reason

    @staticmethod
    def _can_replace_window(plan: List[Dict[str, Any]], event: WorldEvent) -> bool:
        start, end = _minutes(event.start_time), _minutes(event.end_time)
        overlaps = [a for a in plan if _minutes(a.get("start", "00:00")) < end and _minutes(a.get("end", "24:00")) > start]
        return all(WorldEventManager._is_flexible(action) for action in overlaps)

    @staticmethod
    def _insert_event(plan: List[Dict[str, Any]], event: WorldEvent) -> List[Dict[str, Any]]:
        """Replace an entirely-flexible slice with a single event action."""
        start, end = _minutes(event.start_time), _minutes(event.end_time)
        result: List[Dict[str, Any]] = []
        for action in plan:
            a_start, a_end = _minutes(action.get("start", "00:00")), _minutes(action.get("end", "24:00"))
            if a_end <= start or a_start >= end:
                result.append(dict(action))
                continue
            if a_start < start:
                before = dict(action)
                before["end"] = event.start_time
                result.append(before)
            if a_end > end:
                after = dict(action)
                after["start"] = event.end_time
                result.append(after)
        result.append({
            "start": event.start_time, "end": event.end_time,
            "action": f"Attend {event.name}", "location_id": event.location_id,
            "sub_area": None, "energy_change": event.energy_effect,
            "emotion_change": event.emotion_effect, "world_event_id": event.id,
        })
        return sorted(result, key=lambda action: action.get("start", "00:00"))

    def apply_opportunities(
        self,
        date: str,
        plans: Dict[str, List[Dict[str, Any]]],
        personas: Dict[str, Dict[str, Any]],
        relationship_score: Callable[[str, str], float],
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Assign at most one compatible event per student, respecting capacity.

        A plan is changed only if every overlapping block is flexible.  Since
        selected candidates are sorted deterministically, restart/resume produces
        the same attendance without requiring an LLM or mutable random state.
        """
        output = {agent_id: [dict(action) for action in plan] for agent_id, plan in plans.items()}
        selected: set[str] = set()
        for event in self.for_date(date):
            candidates: List[EventAttendance] = []
            for agent_id, plan in output.items():
                if agent_id in selected or not self._can_replace_window(plan, event):
                    continue
                others = [other for other in personas if other != agent_id]
                social = sum(relationship_score(agent_id, other) for other in others) / max(1, len(others))
                score, reason = self._candidate_score(event, date, agent_id, personas[agent_id], social)
                if score >= 0.45:
                    candidates.append(EventAttendance(event_id=event.id, agent_id=agent_id, score=score, reason=reason))
            candidates.sort(key=lambda candidate: (-candidate.score, candidate.agent_id))
            attendees = candidates[:event.capacity]
            self._attendance[event.id] = [item.agent_id for item in attendees]
            for item in attendees:
                output[item.agent_id] = self._insert_event(output[item.agent_id], event)
                selected.add(item.agent_id)
                logger.info("[Events] %s will attend %s (score=%.2f; %s)", item.agent_id, event.id, item.score, item.reason)
        return output
