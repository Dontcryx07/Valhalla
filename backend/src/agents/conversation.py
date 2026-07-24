"""
Conversation -- generates dialogue between two agents via a single LLM call.

The WorldEngine calls `generate_conversation()` when two agents share a
location_id, are both in compatible actions (not sleeping), and neither is
already mid-conversation.

The single LLM call produces the full conversation (messages, summary,
duration, sentiment, relationship delta). Both agents get their current
action overwritten to "Chatting with X" for the duration, then naturally
replan via the tick graph when it expires.

Usage:
    from src.agents.conversation import generate_conversation, RelationshipMatrix

    matrix = RelationshipMatrix()
    result = generate_conversation(
        agent_a_id="parv_singla",
        agent_b_id="tanishq",
        persona_a=gray_wilder_persona,
        persona_b=jules_persona,
        plan_a=gray_wilder_plan,
        plan_b=jules_plan,
        action_a=gray_wilder_current_action,
        action_b=jules_current_action,
        rel_a_to_b=matrix.get("parv_singla", "tanishq"),
        rel_b_to_a=matrix.get("tanishq", "parv_singla"),
        location_id="mess",
        current_hhmm="08:05",
    )
"""

from __future__ import annotations

import sys
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[2]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.append(str(BACKEND_ROOT))


import json
import os
import threading
import time
from typing import Any, Dict, List, Literal, Optional, Tuple

from pydantic import BaseModel, Field

from src.core.log import get_logger
from src.core.world_state import CurrentAction
from src.llm.gemini_client import call_gemini, ProviderFailureError
from src.config import CONVERSATION_TEMPERATURE

logger = get_logger(__name__)


# The engine normally owns a single relationship matrix, but a day handoff or
# a standalone conversation job can briefly create another instance pointing
# at the same file.  Coordinate those instances in-process and make each save
# an atomic replacement so unrelated pairs cannot overwrite one another.
_MATRIX_LOCK_GUARD = threading.Lock()
_MATRIX_LOCKS: Dict[Path, threading.RLock] = {}

# Persona IDs are derived from display names. These aliases preserve saved
# relationship scores when the two seed personas were renamed, while ensuring
# a checkpoint restore cannot recreate obsolete matrix keys on disk.
_LEGACY_AGENT_IDS = {
    "aditi_menon": "riya_murarka",
    "meher_bansal": "lavanya_sharma",
}


def _matrix_lock(path: Path) -> threading.RLock:
    resolved = path.resolve()
    with _MATRIX_LOCK_GUARD:
        return _MATRIX_LOCKS.setdefault(resolved, threading.RLock())


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BLOCKED_KEYWORDS = ["sleep"]
COOLDOWN_TICKS = 30


# ---------------------------------------------------------------------------
# Output schema
# ---------------------------------------------------------------------------


class Message(BaseModel):
    """One line of dialogue in a conversation."""
    speaker: str = Field(min_length=1)
    text: str = Field(min_length=1, max_length=240)


class ConversationResult(BaseModel):
    """Full output of a single conversation generation call."""
    messages: List[Message] = Field(min_length=4, max_length=12)
    summary: str = Field(min_length=1, max_length=360)
    duration_minutes: int = Field(ge=6, le=20)
    sentiment: Literal["positive", "neutral", "negative"]
    relationship_delta: float = Field(ge=-0.15, le=0.15)
    # Folded-in replan decision: avoids a separate 4-call day-plan regeneration
    # per agent after every conversation. True only when the conversation
    # genuinely changes an agent's immediate intentions.
    should_replan: bool = False
    plan_change: Optional[str] = None


# ---------------------------------------------------------------------------
# Relationship matrix -- directional, expressive social context
# ---------------------------------------------------------------------------


class RelationshipRecord(BaseModel):
    """One student's grounded point of view toward another student."""

    score: float = Field(default=0.5, ge=-1.0, le=1.0)
    tags: List[str] = Field(default_factory=list, max_length=12)
    context: str = Field(default="", max_length=500)


class RelationshipMatrix:
    """
    Asymmetric relationship scores between agent pairs.

    Keys are f"{agent_a_id}->{agent_b_id}". A->B may differ from B->A.
    Scores are floats in [-1.0, 1.0] where:
      -1.0 = hostile / strongly negative
       0.0 = strangers / no relationship
       0.5 = neutral acquaintance (default)
       1.0 = very close / best friends
    """

    def __init__(self, path: Optional[Path] = None):
        from src.config import ENVIRONMENT_DIR

        self.path: Path = path or ENVIRONMENT_DIR / "relationship_matrix.json"
        self._matrix: Dict[str, RelationshipRecord] = {}
        self._pending_deltas: List[tuple[str, float]] = []
        self.load()

    def _pair_key(self, a: str, b: str) -> str:
        return f"{a}->{b}"

    def load(self) -> None:
        with _matrix_lock(self.path):
            raw = {}
            try:
                raw = json.loads(self.path.read_text(encoding="utf-8")) if self.path.exists() else {}
            except (OSError, ValueError, json.JSONDecodeError):
                raw = {}
            self._schema_version = int(raw.get("schema_version", 1)) if isinstance(raw, dict) else 1
            self._matrix = self._normalise(raw) if raw else {}
            self._pending_deltas.clear()
        if self._matrix:
            logger.info("[RelationshipMatrix] loaded %d pairs from %s", len(self._matrix), self.path.name)
        else:
            logger.warning("[RelationshipMatrix] %s not found -- starting empty", self.path)

    @staticmethod
    def _normalise(raw: Dict[str, Any]) -> Dict[str, RelationshipRecord]:
        """Accept the legacy flat score map and the structured v1 document."""
        pairs = raw.get("relationships", raw)
        if not isinstance(pairs, dict):
            raise ValueError("relationship entries must be a JSON object")
        result: Dict[str, RelationshipRecord] = {}
        for key, value in pairs.items():
            if "->" not in key:
                continue
            source, target = key.split("->", 1)
            canonical_key = f"{_LEGACY_AGENT_IDS.get(source, source)}->{_LEGACY_AGENT_IDS.get(target, target)}"
            # Prefer a current seed record over its legacy alias when both are
            # present in a migrated file. Legacy-only checkpoint data still
            # maps to the current key and therefore preserves its score.
            if canonical_key != key and canonical_key in result:
                continue
            result[canonical_key] = RelationshipRecord.model_validate(
                {"score": value} if isinstance(value, (int, float)) else value
            )
        return result

    def _read_disk(self) -> Dict[str, RelationshipRecord]:
        if not self.path.exists():
            return {}
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            if not isinstance(raw, dict):
                raise ValueError("relationship matrix must be a JSON object")
            return self._normalise(raw)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            logger.warning(
                "[RelationshipMatrix] unreadable %s; starting empty: %s",
                self.path.name, exc,
            )
            return {}

    def _write_atomic(self, matrix: Dict[str, RelationshipRecord]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        document = {
            "schema_version": 2,
            "relationships": {
                key: value.model_dump() for key, value in sorted(matrix.items())
            },
        }
        temporary.write_text(json.dumps(document, indent=2), encoding="utf-8")
        for attempt in range(3):
            try:
                os.replace(temporary, self.path)
                return
            except PermissionError:
                if attempt == 2:
                    raise
                time.sleep(0.05 * (attempt + 1))

    def save(self) -> None:
        # Reload while holding the file's lock, then replay only this
        # instance's changes.  Two independent conversations updating
        # different pairs therefore compose instead of last-writer-wins.
        with _matrix_lock(self.path):
            merged = self._read_disk()
            for key, delta in self._pending_deltas:
                record = merged.get(key, RelationshipRecord())
                record.score = round(max(-1.0, min(1.0, record.score + delta)), 4)
                merged[key] = record
            self._write_atomic(merged)
            self._matrix = merged
            self._pending_deltas.clear()
        logger.info("[RelationshipMatrix] saved %d pairs to %s", len(self._matrix), self.path.name)

    def get(self, a: str, b: str) -> float:
        """Relationship of agent `a` toward agent `b`. Defaults to 0.5 (neutral)."""
        return self._matrix.get(self._pair_key(a, b), RelationshipRecord()).score

    def remove_agent(self, agent_id: str) -> None:
        """Drop every directional relationship involving one retired agent."""
        self._matrix = {
            key: record for key, record in self._matrix.items()
            if agent_id not in key.split("->", 1)
        }
        self._pending_deltas.clear()
        with _matrix_lock(self.path):
            self._write_atomic(self._matrix)

    def seed(self, source: str, target: str, record: RelationshipRecord) -> None:
        """Set a relationship baseline during a stopped roster operation."""
        self._matrix[self._pair_key(source, target)] = record
        with _matrix_lock(self.path):
            self._write_atomic(self._matrix)

    def replace_display_name(self, old_name: str, new_name: str) -> None:
        """Update display-name references without changing stable agent IDs."""
        for record in self._matrix.values():
            record.context = record.context.replace(old_name, new_name)
        with _matrix_lock(self.path):
            self._write_atomic(self._matrix)

    def ensure_grounded_seed(self, agents: List[Dict[str, Any]]) -> None:
        """Upgrade the original uniform v1 matrix to a directional social baseline.

        Kept here rather than in prompts so the relationships remain data the
        conversation engine can inspect and checkpoint deterministically.
        """
        names = {str(item["agent_id"]): str(item.get("persona_name") or item["agent_id"]) for item in agents}
        expected_keys = {
            self._pair_key(source, target)
            for source in names
            for target in names
            if source != target
        }
        # A roster can be replaced outside a running world.  Version 2 means
        # the record format is current, not that its agent IDs are necessarily
        # current.  Preserve valid directional context, prune retired IDs and
        # seed only missing pairs.
        if getattr(self, "_schema_version", 1) >= 2 and set(self._matrix) == expected_keys:
            return
        profiles = {
            "ankit": ("table tennis and hardware tinkering", "patient practical advice"),
            "ansh_batra": ("football and visual storytelling", "enthusiastic plans"),
            "anubhav_prasad": ("co-op games and late-night chai", "calm listening"),
            "ghanisht_kaushal": ("badminton and blunt movie opinions", "reliable follow-through"),
            "gurnoor_singh": ("photography and road-trip playlists", "big social energy"),
            "lavanya_sharma": ("basketball and debate", "direct feedback"),
            "parv_singla": ("running and music", "impulsive invitations"),
            "riya_murarka": ("reading circles and long runs", "clear boundaries"),
            "saksham": ("badminton and strategy games", "dry humour"),
            "tanishq": ("strategy games and playlists", "quiet, improving confidence"),
        }
        special = {
            ("gurnoor_singh", "parv_singla"): (0.86, ["close-friends", "project-chaos"], "They are close friends, but Gurnoor sometimes has to make Parv slow down before a fun idea becomes three commitments."),
            ("parv_singla", "gurnoor_singh"): (0.84, ["close-friends", "adventure"], "Parv trusts Gurnoor to show up; he also enjoys trying to pull him into a spontaneous outing after a productive day."),
            ("lavanya_sharma", "ghanisht_kaushal"): (0.28, ["friendly-rivalry", "different-work-styles"], "Lavanya respects Ghanisht but thinks he can treat a prototype like a spreadsheet. Their friction is usually productive after a cooling-off break."),
            ("ghanisht_kaushal", "lavanya_sharma"): (0.36, ["friendly-rivalry", "respect"], "Ghanisht finds Lavanya's directness intimidating on tired days, yet trusts her to identify the practical flaw everyone else missed."),
            ("riya_murarka", "tanishq"): (0.18, ["careful-acquaintance", "boundaries"], "Riya is friendly but keeps the connection measured; Tanishq has been making an effort to be considerate and let trust develop at her pace."),
            ("tanishq", "riya_murarka"): (0.30, ["admiration", "respectful-distance"], "Tanishq values Riya's book recommendations and is practising being a considerate, unhurried conversational partner without reading extra meaning into it."),
            ("ansh_batra", "saksham"): (0.47, ["project-partners", "missed-deadline"], "Ansh still feels mildly guilty about an overambitious project timeline; Saksham is warming back up after Ansh started communicating earlier."),
            ("saksham", "ansh_batra"): (0.34, ["project-partners", "rebuilding-trust"], "Saksham likes Ansh's imagination but remembers the missed handoff. He appreciates concrete plans more than enthusiastic promises."),
        }
        rebuilt: Dict[str, RelationshipRecord] = {}
        ids = sorted(names)
        for source in ids:
            for target in ids:
                if source == target:
                    continue
                existing = self._matrix.get(self._pair_key(source, target))
                override = special.get((source, target))
                if existing is not None:
                    rebuilt[self._pair_key(source, target)] = existing
                    continue
                if override:
                    score, tags, context = override
                else:
                    source_interest = profiles.get(source, ("campus routines", "steady conversation"))[0]
                    target_style = profiles.get(target, ("campus routines", "steady conversation"))[1]
                    score = 0.30 + ((sum(map(ord, source + target)) % 33) / 100)
                    tags = ["campus-acquaintance", "low-pressure"]
                    context = f"{names[source]} and {names[target]} know each other through {source_interest}. {names[source]} appreciates {target}'s {target_style}, but the friendship is still finding its rhythm."
                rebuilt[self._pair_key(source, target)] = RelationshipRecord(score=score, tags=tags, context=context)
        with _matrix_lock(self.path):
            self._matrix = rebuilt
            self._pending_deltas.clear()
            self._schema_version = 2
            self._write_atomic(rebuilt)

    def context(self, a: str, b: str) -> RelationshipRecord:
        """Return a copy of the expressive directed relationship record."""
        return self._matrix.get(self._pair_key(a, b), RelationshipRecord()).model_copy(deep=True)

    def update(self, a: str, b: str, delta: float) -> None:
        """Adjust relationship of `a` toward `b` by `delta`, clamped to [-1, 1]."""
        key = self._pair_key(a, b)
        current = self.get(a, b)
        new_val = max(-1.0, min(1.0, current + delta))
        record = self._matrix.get(key, RelationshipRecord())
        record.score = round(new_val, 4)
        self._matrix[key] = record
        self._pending_deltas.append((key, delta))
        logger.info("[RelationshipMatrix] %s: %.2f -> %.2f (delta=%.2f)", key, current, new_val, delta)

    def snapshot(self) -> Dict[str, Any]:
        """Return a JSON-safe copy for the simulation checkpoint."""
        return {"schema_version": 2, "relationships": {
            key: value.model_dump() for key, value in self._matrix.items()
        }}

    def restore(self, matrix: Dict[str, Any]) -> None:
        """Restore scores from a checkpoint while retaining base social context.

        Older checkpoints only contain a numeric matrix. Their scores are
        authoritative runtime state, but they must not erase the structured
        tags/context seeded in the current relationship data file.
        """
        with _matrix_lock(self.path):
            base = self._read_disk()
            checkpoint = self._normalise(matrix)
            for key, record in checkpoint.items():
                if key in base:
                    base[key].score = record.score
                else:
                    base[key] = record
            self._matrix = base
            self._pending_deltas.clear()
            self._write_atomic(self._matrix)


# ---------------------------------------------------------------------------
# Helper: check if an action is sleep-blocked
# ---------------------------------------------------------------------------


def _is_action_blocked(action: CurrentAction) -> bool:
    """Return True if the agent's current action prevents conversation."""
    desc = action.description.lower()
    return any(kw in desc for kw in BLOCKED_KEYWORDS)


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------


SYSTEM_PROMPT = """You are a dialogue writer for a campus life simulation at IIT Ropar.
Two student personas with distinct personalities and daily schedules have encountered each other.

Generate a short, natural conversation between them based on:
- Their personalities, hobbies, and goals
- What each is currently doing at this moment
- Their day plans (what they've done today and what's coming up next)
- Their existing relationship with each other
- The current time of day and location on campus
- What each might have recently experienced or remembers about the other

Rules:
- Start naturally from the current situation and surroundings
- Write 4-12 alternating dialogue lines, each 4-25 words
- Choose a duration of 6-20 simulated minutes that fits the dialogue
- Match tone to personality (friendly, awkward, casual, competitive, etc.)
- Each message must have a "speaker" (the agent's full name) and "text" (what they say)
- The conversation must have a natural ending
- Reference the time of day and campus context naturally in the dialogue
- Return ONLY valid JSON matching the schema -- no extra text or commentary"""


def _personality_summary(persona: Dict[str, Any]) -> str:
    """Condense a persona dict into a 2-3 line readable summary."""
    parts = [
        f"Name: {persona.get('Name', '?')}",
        f"Branch: {persona.get('Branch', '?')}",
        f"Personality: {persona.get('innate', '?')}",
        f"Hobbies: {persona.get('hobbies', '?')}",
        f"Goals: {persona.get('goals', '?')}",
    ]
    return "\n".join(parts)


def _deprecated_plan_summary(plan: List[Dict[str, Any]], current_hhmm: str) -> str:
    """Deprecated compatibility helper; retained only for external callers."""
    if not plan:
        return "  (no plan)"

    lines = []
    for entry in sorted(plan, key=lambda e: e.get("start", "00:00")):
        marker = " ⇐ NOW" if entry.get("start") == current_hhmm else ""
        lines.append(
            f"  {entry.get('start', '??')}-{entry.get('end', '??')}  "
            f"{entry.get('action', '?')} at {entry.get('location_id', '?')}{marker}"
        )
    return "\n".join(lines)


def _plan_summary(plan: List[Dict[str, Any]], current_hhmm: str) -> str:
    """Render a plan and identify the entry that covers the current time."""
    if not plan:
        return "  (no plan)"

    def minutes(value: str) -> int:
        hour, minute = value.split(":")
        return int(hour) * 60 + int(minute)

    now = minutes(current_hhmm)
    lines = []
    for entry in sorted(plan, key=lambda item: item.get("start", "00:00")):
        try:
            active = minutes(entry.get("start", "00:00")) <= now < minutes(entry.get("end", "24:00"))
        except (TypeError, ValueError):
            active = False
        marker = " [NOW]" if active else ""
        lines.append(
            f"  {entry.get('start', '??')}-{entry.get('end', '??')}  "
            f"{entry.get('action', '?')} at {entry.get('location_id', '?')}{marker}"
        )
    return "\n".join(lines)


def _build_prompts(
    agent_a_id: str,
    agent_b_id: str,
    persona_a: Dict[str, Any],
    persona_b: Dict[str, Any],
    plan_a: List[Dict[str, Any]],
    plan_b: List[Dict[str, Any]],
    action_a: CurrentAction,
    action_b: CurrentAction,
    rel_a_to_b: float,
    rel_b_to_a: float,
    location_id: str,
    current_hhmm: str,
    memories_a: Optional[List[str]] = None,
    memories_b: Optional[List[str]] = None,
    energy_a: float = 0.5, emotion_a: float = 0.5,
    energy_b: float = 0.5, emotion_b: float = 0.5,
    relationship_context_a: str = "",
    relationship_context_b: str = "",
) -> Tuple[str, str]:
    """Build system and user prompts for the conversation LLM call."""

    def _mem_block(mems: Optional[List[str]]) -> str:
        if not mems:
            return "(nothing notable recalled)"
        return "\n".join(f"  - {m}" for m in mems[:4])
    def _rel_label(score: float) -> str:
        if score >= 0.8:
            return "close friends"
        elif score >= 0.6:
            return "friendly"
        elif score >= 0.4:
            return "neutral acquaintances"
        elif score >= 0.2:
            return "distant / barely know each other"
        else:
            return "negative / strained"

    hour = int(current_hhmm.split(":")[0]) if ":" in current_hhmm else 12
    if hour < 6:
        time_desc = "early morning — campus is quiet, most students are asleep"
    elif hour < 10:
        time_desc = "morning — students heading to breakfast or first classes"
    elif hour < 12:
        time_desc = "late morning — classes are in session around campus"
    elif hour < 14:
        time_desc = "lunchtime — the mess and cafeterias are busy"
    elif hour < 17:
        time_desc = "afternoon — afternoon classes and study sessions"
    elif hour < 19:
        time_desc = "late afternoon — some students heading to sports or hobbies"
    elif hour < 21:
        time_desc = "evening — dinner time, socializing on campus"
    elif hour < 23:
        time_desc = "late evening — students heading back to hostels"
    else:
        time_desc = "night — campus is winding down"

    def _emotion_label(v: float) -> str:
        if v >= 0.9: return "Extremely Joyful"
        if v >= 0.7: return "Very Happy"
        if v > 0.5: return "Happy"
        if v >= 0.3: return "Neutral"
        if v >= 0.1: return "Sad"
        return "Extremely Sad"

    user = f"""Generate a conversation between two students at IIT Ropar.

── AGENT: {agent_a_id} ──────────────────────
{_personality_summary(persona_a)}
Current action: "{action_a.description}"
Today's plan:
{_plan_summary(plan_a, current_hhmm)}
Energy: {energy_a:.2f}/1.0 | Emotion: {emotion_a:.2f}/1.0 ({_emotion_label(emotion_a)})

── AGENT: {agent_b_id} ──────────────────────
{_personality_summary(persona_b)}
Current action: "{action_b.description}"
Today's plan:
{_plan_summary(plan_b, current_hhmm)}
Energy: {energy_b:.2f}/1.0 | Emotion: {emotion_b:.2f}/1.0 ({_emotion_label(emotion_b)})

── CONTEXT ──────────────────────
Location: {location_id}
Current time: {current_hhmm} ({time_desc})
Relationship notes: A sees B as {relationship_context_a or 'an acquaintance'}; B sees A as {relationship_context_b or 'an acquaintance'}.
{persona_a.get('Name', agent_a_id)}'s relationship toward {persona_b.get('Name', agent_b_id)}: {rel_a_to_b} ({_rel_label(rel_a_to_b)})
{persona_b.get('Name', agent_b_id)}'s relationship toward {persona_a.get('Name', agent_a_id)}: {rel_b_to_a} ({_rel_label(rel_b_to_a)})

What {persona_a.get('Name', agent_a_id)} remembers about {persona_b.get('Name', agent_b_id)}:
{_mem_block(memories_a)}
What {persona_b.get('Name', agent_b_id)} remembers about {persona_a.get('Name', agent_a_id)}:
{_mem_block(memories_b)}

── OUTPUT ───────────────────────
Return a JSON object with:
- "messages": 4-12 alternating entries using exactly "{persona_a.get('Name', agent_a_id)}" and "{persona_b.get('Name', agent_b_id)}" as speakers; each line is 4-25 words
- "summary": one-sentence summary of what they talked about
- "duration_minutes": integer from 6 to 20 that matches the amount of dialogue
- "sentiment": "positive" | "neutral" | "negative"
- "relationship_delta": float between -0.15 and 0.15 (how this conversation changes their relationship)
- "should_replan": boolean — true ONLY if this conversation genuinely changes what one of them intends to do next (e.g. they agree to meet, go somewhere together, or drop a task). Default false; most casual chats do NOT require replanning.
- "plan_change": short string describing the change if should_replan is true, else null"""

    return SYSTEM_PROMPT, user


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def generate_conversation(
    agent_a_id: str,
    agent_b_id: str,
    persona_a: Dict[str, Any],
    persona_b: Dict[str, Any],
    plan_a: List[Dict[str, Any]],
    plan_b: List[Dict[str, Any]],
    action_a: CurrentAction,
    action_b: CurrentAction,
    rel_a_to_b: float,
    rel_b_to_a: float,
    location_id: str,
    current_hhmm: str,
    memories_a: Optional[List[str]] = None,
    memories_b: Optional[List[str]] = None,
    energy_a: float = 0.5, emotion_a: float = 0.5,
    energy_b: float = 0.5, emotion_b: float = 0.5,
    relationship_context_a: str = "",
    relationship_context_b: str = "",
) -> Optional[ConversationResult]:
    """
    Generate a conversation between two agents via a single Gemini call.

    Returns None if the LLM call fails (logged, not raised) so the
    WorldEngine can continue the tick without crashing.
    """
    system_prompt, user_prompt = _build_prompts(
        agent_a_id, agent_b_id,
        persona_a, persona_b,
        plan_a, plan_b,
        action_a, action_b,
        rel_a_to_b, rel_b_to_a,
        location_id, current_hhmm,
        memories_a, memories_b,
        energy_a=energy_a, emotion_a=emotion_a,
        energy_b=energy_b, emotion_b=emotion_b,
        relationship_context_a=relationship_context_a,
        relationship_context_b=relationship_context_b,
    )

    logger.info(
        "[Conversation] generating conversation between '%s' and '%s' at %s (%s)",
        agent_a_id, agent_b_id, location_id, current_hhmm,
    )

    try:
        result: ConversationResult = call_gemini(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            schema=ConversationResult,
            complexity="default",
            temperature=CONVERSATION_TEMPERATURE,
        )
        logger.info(
            "[Conversation] generated %d messages, %d min, sentiment=%s, delta=%.2f",
            len(result.messages), result.duration_minutes,
            result.sentiment, result.relationship_delta,
        )
        return result

    except ProviderFailureError:
        raise
    except Exception:
        logger.exception(
            "[Conversation] LLM call failed for '%s' <-> '%s' -- skipping conversation",
            agent_a_id, agent_b_id,
        )
        return None


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    from src.core.log import setup_logging
    setup_logging(run_id="conversation_test", console=True)

    from src.config import PERSONALITIES_DIR
    from src.agents.Short_term import load_day_plan, date_from_simulation_time, append_conversation

    parser = argparse.ArgumentParser(
        description="Conversation module self-test -- generate a conversation between two personas"
    )
    parser.add_argument("persona_a", nargs="?", default="parv_singla", help="First persona name")
    parser.add_argument("persona_b", nargs="?", default="tanishq", help="Second persona name")
    parser.add_argument(
        "--current-time", default="2026-07-03 08:00",
        help="Simulation time (default: 2026-07-03 08:00)",
    )
    args = parser.parse_args()

    def _resolve_persona(name: str) -> tuple[Path, Dict[str, Any]]:
        """Find a persona JSON by name, return (path, data)."""
        matches = sorted(PERSONALITIES_DIR.glob(f"**/{name}/{name}.json"))
        if not matches:
            matches = sorted(PERSONALITIES_DIR.glob(f"**/{name}.json"))
        if not matches:
            available = sorted({p.parent.name for p in PERSONALITIES_DIR.glob("**/*.json")})
            print(f"Persona '{name}' not found. Available: {', '.join(available)}")
            sys.exit(1)
        return matches[0], json.loads(matches[0].read_text())

    def _find_action_for_time(
        plan: List[Dict[str, Any]], hhmm: str
    ) -> Optional[Dict[str, Any]]:
        """Find the day plan entry covering HH:MM."""
        def _to_mins(t: str) -> int:
            h, m = t.split(":")
            return int(h) * 60 + int(m)

        target = _to_mins(hhmm)
        for entry in sorted(plan, key=lambda e: e.get("start", "00:00")):
            start = _to_mins(entry.get("start", "00:00"))
            end = _to_mins(entry.get("end", "24:00"))
            if start <= target < end:
                return entry
        return None

    # Resolve personas
    path_a, persona_a = _resolve_persona(args.persona_a)
    path_b, persona_b = _resolve_persona(args.persona_b)

    name_a = persona_a.get("Name", path_a.stem)
    name_b = persona_b.get("Name", path_b.stem)

    from src.agents.Short_term import _persona_dir
    id_a = _persona_dir(name_a).name
    id_b = _persona_dir(name_b).name

    # Load day plans
    sim_date = date_from_simulation_time(args.current_time)
    _, current_hhmm = args.current_time.split(" ") if " " in args.current_time else ("", args.current_time)

    plan_a = load_day_plan(name_a, sim_date) or []
    plan_b = load_day_plan(name_b, sim_date) or []

    if not plan_a or not plan_b:
        print(f"Need day plans for both personas. Run: python backend/src/core/Agent.py {args.persona_a}")
        print(f"  and: python backend/src/core/Agent.py {args.persona_b}")
        sys.exit(1)

    print(f"Personas: {name_a} <-> {name_b}")
    print(f"  Agent IDs: {id_a}, {id_b}")
    print(f"  Plans: {len(plan_a)} actions, {len(plan_b)} actions")
    print(f"  Time: {args.current_time} (HH:MM={current_hhmm})")

    # Find current action for each
    entry_a = _find_action_for_time(plan_a, current_hhmm)
    entry_b = _find_action_for_time(plan_b, current_hhmm)

    if not entry_a or not entry_b:
        print(f"No plan entry covers {current_hhmm} for one or both personas.")
        sys.exit(1)

    action_a = CurrentAction(
        description=entry_a["action"],
        start_tick=0,
        end_tick=1000,
        target_location_id=entry_a.get("location_id"),
    )
    action_b = CurrentAction(
        description=entry_b["action"],
        start_tick=0,
        end_tick=1000,
        target_location_id=entry_b.get("location_id"),
    )

    # Load relationship matrix
    matrix = RelationshipMatrix()

    print(f"\n{name_a} is: {action_a.description} at {entry_a.get('location_id', '?')}")
    print(f"{name_b} is: {action_b.description} at {entry_b.get('location_id', '?')}")
    print(f"Relationship A->B: {matrix.get(id_a, id_b)}")
    print(f"Relationship B->A: {matrix.get(id_b, id_a)}")
    print()

    # Location guess: use the first entry's location that has one
    location_a = entry_a.get("location_id", "")
    location_b = entry_b.get("location_id", "")
    location_id = location_a if location_a == location_b else (location_a or location_b or "campus")

    result = generate_conversation(
        agent_a_id=id_a,
        agent_b_id=id_b,
        persona_a=persona_a,
        persona_b=persona_b,
        plan_a=plan_a,
        plan_b=plan_b,
        action_a=action_a,
        action_b=action_b,
        rel_a_to_b=matrix.get(id_a, id_b),
        rel_b_to_a=matrix.get(id_b, id_a),
        location_id=location_id,
        current_hhmm=current_hhmm,
    )

    if result is None:
        print("Conversation generation failed.")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"SUMMARY: {result.summary}")
    print(f"Duration: {result.duration_minutes} min")
    print(f"Sentiment: {result.sentiment}")
    print(f"Relationship delta: {result.relationship_delta:+.2f}")
    print(f"{'='*60}")
    print()

    for msg in result.messages:
        print(f"  [{msg.speaker}] {msg.text}")
        print()

    # Save to both personas' Short_term memory
    date_str = sim_date
    conv_entry = {
        "participants": [name_a, name_b],
        "messages": [{"speaker": m.speaker, "text": m.text} for m in result.messages],
        "summary": result.summary,
    }

    append_conversation(name_a, date_str, conv_entry)
    append_conversation(name_b, date_str, conv_entry)
    print(f"Saved conversation to {id_a} and {id_b} for {date_str}")

    # Update relationship matrix
    matrix.update(id_a, id_b, result.relationship_delta)
    matrix.update(id_b, id_a, result.relationship_delta)
    matrix.save()
    print(f"Updated relationship matrix: {id_a}->{id_b} = {matrix.get(id_a, id_b):.2f}")

    print(f"\nConversation module self-test passed.")
