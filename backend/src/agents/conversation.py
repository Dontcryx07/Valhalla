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
        persona_a=parv_persona,
        persona_b=tanishq_persona,
        plan_a=parv_plan,
        plan_b=tanishq_plan,
        action_a=parv_current_action,
        action_b=tanishq_current_action,
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
from typing import Any, Dict, List, Literal, Optional, Tuple

from pydantic import BaseModel

from src.core.log import get_logger
from src.core.world_state import CurrentAction
from src.llm.gemini_client import call_gemini, AllModelsFailedError

logger = get_logger(__name__)


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
    speaker: str
    text: str


class ConversationResult(BaseModel):
    """Full output of a single conversation generation call."""
    messages: List[Message]
    summary: str
    duration_minutes: int
    sentiment: Literal["positive", "neutral", "negative"]
    relationship_delta: float


# ---------------------------------------------------------------------------
# Relationship matrix -- maps agent_a->agent_b relationship scores
# ---------------------------------------------------------------------------


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
        self._matrix: Dict[str, float] = {}
        self.load()

    def _pair_key(self, a: str, b: str) -> str:
        return f"{a}->{b}"

    def load(self) -> None:
        if self.path.exists():
            raw = json.loads(self.path.read_text())
            self._matrix = {k: float(v) for k, v in raw.items()}
            logger.info("[RelationshipMatrix] loaded %d pairs from %s", len(self._matrix), self.path.name)
        else:
            logger.warning("[RelationshipMatrix] %s not found -- starting empty", self.path)

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self._matrix, indent=2))
        logger.info("[RelationshipMatrix] saved %d pairs to %s", len(self._matrix), self.path.name)

    def get(self, a: str, b: str) -> float:
        """Relationship of agent `a` toward agent `b`. Defaults to 0.5 (neutral)."""
        return self._matrix.get(self._pair_key(a, b), 0.5)

    def update(self, a: str, b: str, delta: float) -> None:
        """Adjust relationship of `a` toward `b` by `delta`, clamped to [-1, 1]."""
        key = self._pair_key(a, b)
        current = self.get(a, b)
        new_val = max(-1.0, min(1.0, current + delta))
        self._matrix[key] = round(new_val, 4)
        logger.info("[RelationshipMatrix] %s: %.2f -> %.2f (delta=%.2f)", key, current, new_val, delta)


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
- What each is currently doing
- Their day plans (what they've done and what's coming up)
- Their existing relationship

Rules:
- Start naturally from the current situation
- Keep duration realistic (5-30 minutes typical)
- Match tone to personality (friendly, awkward, casual, competitive, etc.)
- Each message must have a "speaker" (the agent's full name) and "text" (what they say)
- The conversation must have a natural ending
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


def _plan_summary(plan: List[Dict[str, Any]], current_hhmm: str) -> str:
    """Format the day plan as a readable timeline, highlighting what's happening now."""
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
) -> Tuple[str, str]:
    """Build system and user prompts for the conversation LLM call."""

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

    user = f"""Generate a conversation between two students at IIT Ropar.

── AGENT: {agent_a_id} ──────────────────────
{_personality_summary(persona_a)}
Current action: "{action_a.description}"
Today's plan:
{_plan_summary(plan_a, current_hhmm)}

── AGENT: {agent_b_id} ──────────────────────
{_personality_summary(persona_b)}
Current action: "{action_b.description}"
Today's plan:
{_plan_summary(plan_b, current_hhmm)}

── CONTEXT ──────────────────────
Location: {location_id}
Current time: {current_hhmm}
{persona_a.get('Name', agent_a_id)}'s relationship toward {persona_b.get('Name', agent_b_id)}: {rel_a_to_b} ({_rel_label(rel_a_to_b)})
{persona_b.get('Name', agent_b_id)}'s relationship toward {persona_a.get('Name', agent_a_id)}: {rel_b_to_a} ({_rel_label(rel_b_to_a)})

── OUTPUT ───────────────────────
Return a JSON object with:
- "messages": list of {{"speaker": "<full name>", "text": "<dialogue line>"}} (each message has exactly these two string fields)
- "summary": one-sentence summary of what they talked about
- "duration_minutes": int (how many minutes the conversation lasts)
- "sentiment": "positive" | "neutral" | "negative"
- "relationship_delta": float between -1.0 and 1.0 (how this conversation changes their relationship)"""

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
            complexity="simple",
            temperature=0.7,
        )
        logger.info(
            "[Conversation] generated %d messages, %d min, sentiment=%s, delta=%.2f",
            len(result.messages), result.duration_minutes,
            result.sentiment, result.relationship_delta,
        )
        return result

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
    parser.add_argument("persona_a", nargs="?", default="parv", help="First persona name")
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
