"""
A script that plans the day of an agentic personality when handed over required data
Per plan takes 4 LLM calls (atlest) Coarse, Hourly, Fine, Validation, for Planning a
day in one agent's life.

Tier-1 LangGraph subgraph: agent day-planning.

Pipeline (mirrors Generative Agents' Planning module, coarse -> hourly -> fine,
with a validation/retry loop):

    generate_coarse_plan -> decompose_hourly -> decompose_fine -> validate_plan
                                                                        |
                                                        conflict? --yes-+ (loop back to generate_coarse_plan)
                                                            |
                                                            no -> END

LLM backend: Google Gemini via the `google-genai` SDK.

For now `relevant_memories` and `yesterday_summary` are expected to arrive
empty ([] / None) -- the prompts already handle that gracefully so you can
wire in real retrieval/memory later without touching this file's structure.

FILE NOTES:
Prompt structure can be improved
Places are being feed in Name : , Desc : format, this can be improved
disabled location check in validate plan : can add more places

Prompt templates have to improve

Have to figure out how to run this in the backend server, currently it is running standalone
"""

from __future__ import annotations

# Add root to sys.path
import sys
from pathlib import Path

# backend/ is two levels up from this file (agents -> src -> backend)
BACKEND_ROOT = Path(__file__).resolve().parents[2]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.append(str(BACKEND_ROOT))


import json
import re
import threading
from collections import defaultdict, deque
from src.core.log import get_logger
from src.llm.gemini_client import call_gemini, AllModelsFailedError, ProviderFailureError
from google.genai import errors as genai_errors
from typing import Any, Dict, List, Optional, TypedDict, Literal

from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field

# Setting up logger
logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Config -> in config.py
# ---------------------------------------------------------------------------
from src.config import ENVIRONMENT_DIR, DATA_DIR
from src.config import MAX_PLAN_RETRIES, PERSONA_FIELD_GLOSSARY

# The simulator represents an adult college community, not sexual or hateful
# roleplay. This deterministic guard runs before semantic LLM QA.
_UNSAFE_PLAN_TERMS = (
    "non-consensual", "porn", "explicit content", "gooning", "jerking off",
    "stalking profiles", "racial prejudice", "racist", "manipulating conversations",
    "soft porn", "sexual content",
)

# Academic locations are a behavioural contract, not a preference hidden in a
# prompt.  Models may still choose LHC for genuinely common/shared teaching or
# SAB for explicitly shared work, but branch-specific teaching belongs here.
_BRANCH_VENUES = (
    ("mechanical", "mechanical_department"),
    ("chemical", "chemical_department"),
    ("electrical", "electrical_department"),
    ("integrated circuit", "electrical_department"),
    ("computer science", "computer_science_department"),
    ("aide", "computer_science_department"),
    ("artificial intelligence", "computer_science_department"),
    ("data engineering", "computer_science_department"),
)
_BRANCH_ACTIVITY_PATTERN = re.compile(
    r"\b(?:class(?:es)?|lecture(?:s)?|lab(?:s)?|laborator(?:y|ies)|tutorial(?:s)?|practical(?:s)?)\b",
    re.IGNORECASE,
)
_SHARED_SESSION_WORDS = ("common", "core", "elective", "guest", "large", "shared", "interdisciplinary", "seminar")


def _academic_venue_policy(persona: Dict[str, Any]) -> str:
    branch = str(persona.get("Branch", ""))
    destination = next((venue for marker, venue in _BRANCH_VENUES if marker in branch.lower()), None)
    if not destination:
        return "No branch-specific policy is known; choose the listed location that explicitly fits."
    return (
        f"This student is in {branch}. Branch-specific classes, tutorials, and labs MUST use "
        f"`{destination}`. LHC is only for common/core/elective/guest/large shared sessions. "
        "SAB is only for shared academic services, interdisciplinary work, seminars, overflow labs, "
        "faculty meetings, or an activity explicitly described as happening there."
    )


def _local_academic_venue_check(actions: List[Dict[str, Any]], persona: Dict[str, Any]) -> Optional[str]:
    branch = str(persona.get("Branch", "")).lower()
    required = next((venue for marker, venue in _BRANCH_VENUES if marker in branch), None)
    if not required:
        return None
    for action in actions:
        description = str(action.get("action", "")).lower()
        location = str(action.get("location_id", ""))
        # Use whole academic terms.  A substring check classified ordinary
        # activities such as "chai break with classmates" as a class because
        # "class" is part of "classmates", causing needless replan retries.
        if not _BRANCH_ACTIVITY_PATTERN.search(description):
            continue
        if any(word in description for word in _SHARED_SESSION_WORDS):
            continue
        if location != required:
            return (
                f"branch-specific academic action '{action.get('action')}' for {persona.get('Branch')} "
                f"must use {required}, not {location}"
            )
    return None


def load_places(places_file: Optional[Path] = None) -> List[Place]:
    """
    Loads the college places JSON and returns a list of validated Place
    objects. Expects a top-level {"locations": [...]} structure where each
    entry matches the Place schema.

    Gets the places from ENVIRONMENT_DIR/places.json by default, or from a
    custom path if provided.
    """
    path = places_file or ENVIRONMENT_DIR / "places.json"

    if not Path(path).exists():
        logger.warning(
            "[day_planner] places file not found at %s -- proceeding with no known locations",
            path,
        )
        return []

    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    entries = raw.get("locations", [])

    places: List[Place] = []
    for i, entry in enumerate(entries):
        places.append(Place.model_validate(entry))

    logger.info("[day_planner] loaded %d places from %s", len(places), path)
    return places

def _places_block(places: List[Place]) -> str:
    if not places:
        return "(no places data available -- pick a reasonable generic label)"
    blocks = []
    for p in places:
        lines = [f"[{p.id}] {p.name} ({p.type})"]
        if p.sub_areas:
            lines.append(f"  sub-areas: {', '.join(p.sub_areas)}")
        if p.open_hours:
            hours = "; ".join(f"{k}: {v}" for k, v in p.open_hours.items())
            lines.append(f"  hours: {hours}")
        if p.typical_activities:
            lines.append(f"  good for: {', '.join(p.typical_activities)}")
        if p.notes:
            lines.append(f"  notes: {p.notes}")
        blocks.append("\n".join(lines))
    return "\n".join(blocks)



# ---------------------------------------------------------------------------
# Structured-output schemas (pydantic -> Gemini response_schema)
# ---------------------------------------------------------------------------

class CoarseBlock(BaseModel):
    activity: str = Field(description="Short label, e.g. 'work on farm', 'breakfast'")
    start: str = Field(description="24h HH:MM")
    end: str = Field(description="24h HH:MM")
    granularity: Literal["atomic", "flexible"] = Field(
        description=(
            "'atomic' = this block should NEVER be split into finer sub-steps, it "
            "stays as a single continuous action (e.g. sleep, attending a lecture, "
            "commuting, an exam, watching a movie). "
            "'flexible' = this block genuinely contains distinct sub-activities worth "
            "breaking down (e.g. 'morning routine', 'gym session', 'work on project')."
        )
    )
    energy_change: float = 0.0
    emotion_change: float = 0.0


class CoarsePlanOutput(BaseModel):
    blocks: List[CoarseBlock]


class HourlyBlock(BaseModel):
    activity: str
    start: str
    end: str
    parent_activity: str = Field(description="The coarse block this refines")
    energy_change: float = 0.0
    emotion_change: float = 0.0


class HourlyPlanOutput(BaseModel):
    blocks: List[HourlyBlock]


class FineAction(BaseModel):
    action: str = Field(description="Concrete, executable action, 5-30 min granularity")
    start: str
    end: str
    parent_activity: str = Field(description="The hourly block this refines")
    location_id: str = Field(description="Must exactly match an 'id' from the known places list")
    sub_area: Optional[str] = Field(default=None, description="One of that place's sub_areas, if applicable")
    energy_change: float = Field(description="Energy change [-1.0, 1.0] over this action; positive=restorative, negative=tiring")
    emotion_change: float = Field(description="Emotion change [-1.0, 1.0] over this action; positive=uplifting, negative=draining")


class FinePlanOutput(BaseModel):
    actions: List[FineAction]

# For Coars Atomic actions
class AtomicLocationAssignment(BaseModel):
    activity: str
    location_id: str
    sub_area: Optional[str] = None
    energy_change: float = 0.0
    emotion_change: float = 0.0

class AtomicLocationOutput(BaseModel):
    assignments: List[AtomicLocationAssignment]


class ValidationResult(BaseModel):
    valid: bool
    reason: Optional[str] = Field(
        default=None, description="If invalid: which items conflict/overlap/gap and why"
    )


class Place(BaseModel):
    id: str
    name: str
    type: str
    sub_areas: List[str] = []
    open_hours: Dict[str, str] = {}
    capacity: Optional[str] = None
    typical_activities: List[str] = []
    connected_locations: List[str] = []
    notes: Optional[str] = None


def _normalize_places(places: List[Any]) -> List[Place]:
    """Accept validated Place models or raw JSON-compatible place records."""
    normalized: List[Place] = []
    for place in places:
        try:
            normalized.append(place if isinstance(place, Place) else Place.model_validate(place))
        except Exception as exc:
            logger.warning("[day_planner] ignoring invalid place entry: %s", exc)
    return normalized


# ---------------------------------------------------------------------------
# Graph state
# ---------------------------------------------------------------------------

class DayPlannerState(TypedDict, total=False):
    # ---- inputs ----
    persona: dict
    relevant_memories: List[str]
    yesterday_summary: Optional[str]
    current_time: str  # e.g. "2026-07-03 06:00"
    places: List[Place]  # known campus locations, from places.json
    mode: str  # "full_day", "remaining", or "next_day"
    current_location_id: Optional[str] = None
    handoff_context: Optional[str] = None
    daily_theme: str  # random theme (e.g. "Sports", "Academics")
    daily_emotion: str  # random mood (e.g. "Happy", "Melancholic")
    upcoming_events: List[Dict[str, Any]]

    # ---- working state ----
    coarse_plan: List[Dict[str, Any]]
    hourly_plan: List[Dict[str, Any]]
    fine_plan: List[Dict[str, Any]]

    conflict_detected: bool
    conflict_reason: Optional[str]
    retry_count: int

    # ---- output ----
    day_plan: List[Dict[str, Any]]
    error: Optional[str]
    replan_rejected: bool


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _persona_block(persona: dict) -> str:
    """Renders whatever fields exist in the persona JSON -- don't assume a
    fixed schema beyond name, since personas will vary as you add more."""
    lines = [f"Name: {persona.get('Name') or persona.get('name', 'Unknown')}"]
    for field, meaning in PERSONA_FIELD_GLOSSARY.items():
        value = persona.get(field)
        if value:
            lines.append(f"- {field} ({meaning}): {value}")

    identity_fields = ["Age", "Gender", "Branch", "Home City", "Hostel"]
    lines.extend(f"{f}: {persona[f]}" for f in identity_fields if persona.get(f))

    return "\n".join(lines)


def _memories_block(memories: List[str]) -> str:
    if not memories:
        return "(none available yet)"
    return "\n".join(f"- {m}" for m in memories)


def _events_block(events: List[Dict[str, Any]]) -> str:
    if not events:
        return "(no announced events)"
    return "\n".join(
        f"- {event.get('start_time', '?')}-{event.get('end_time', '?')}: "
        f"{event.get('name', 'Campus event')} at {event.get('location_id', '?')} — "
        f"{event.get('announcement', '')}"
        for event in events[:5]
    )


def _flavor_block(state: DayPlannerState) -> str:
    theme = state.get("daily_theme", "Neutral")
    emotion = state.get("daily_emotion", "Neutral")
    return f"Today's vibe: {emotion}, leaning into {theme}."


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

def generate_coarse_plan(state: DayPlannerState) -> DayPlannerState:
    persona = state["persona"]
    mode = state.get("mode", "full_day")
    current_time = state.get("current_time", "unknown")

    if mode == "full_day":
        coverage = "covering the full 24 hours (00:00 to 24:00). The very first action MUST start at 00:00 — do not skip morning/midnight hours."
    elif mode == "next_day":
        coverage = "covering the full next calendar day from 00:00 to 24:00. The very first action MUST start at 00:00 — this is a brand-new day, do not skip ahead."
    elif mode == "remaining":
        coverage = f"covering ONLY the REMAINDER of the day from the current time onward -- do NOT schedule anything before the current time"
    else:
        coverage = "covering the full 24 hours (00:00 to 24:00)"

    loc_hint = ""
    current_loc = state.get("current_location_id")
    system_prompt = (
        "You are simulating one day in the life of a character in a generative-agents "
        "simulation. Produce a COARSE day plan: 5 to 8 broad blocks of activity "
        f"{coverage}, with no gaps and no overlaps. "
        "Stay true to the persona's traits, background, goals, and daily habits.\n\n"
        "The movement executor owns routes between locations. Plan only time spent "
        "at places; do not create separate walk, commute, travel, or transit blocks. "
        "Leave enough dwell time for meals, classes, study, rest, and social activities. "
        "All activities must be age-appropriate and respectful: never include explicit, coercive, harassing, discriminatory, or stalking behaviour.\n\n"
        "For EACH block, tag it as 'atomic' or 'flexible':\n"
        "- atomic: a single continuous activity with no meaningful internal sub-steps "
        "worth planning separately. Examples: sleeping, attending a class/lecture, "
        "sitting an exam, watching a movie, a long uninterrupted study/deep-work session.\n"
        "- flexible: an activity that naturally contains distinct on-site sub-activities.\n\n"
        "For EACH block, assign realistic energy_change and emotion_change values. Routine classes, labs, study, meals, and chores should be near neutral (usually -0.03 to +0.03); reserve larger positive changes for rare, meaningful events."
    )
    if current_loc:
        loc_hint = f"\nThe agent is currently at: {current_loc}. Start the plan from this location."
    user_prompt = (
        f"PERSONA:\n{_persona_block(persona)}\n\n"
        f"RELEVANT MEMORIES:\n{_memories_block(state.get('relevant_memories', []))}\n\n"
        f"YESTERDAY'S SUMMARY:\n{state.get('yesterday_summary') or '(no history yet, this is day 1)'}\n\n"
        f"ANNOUNCED CAMPUS EVENTS (optional; never displace classes, meals, sleep, exams, or deadlines):\n{_events_block(state.get('upcoming_events', []))}\n\n"
        f"{_flavor_block(state)}\n\n"
        f"Current in-simulation time: {current_time}\n"
        f"Plan mode: {mode}\n"
        f"Agent location: {current_loc or 'unknown'}{loc_hint}\n\n"
        f"DAY-HANDOFF CONTINUITY:\n{state.get('handoff_context') or '(none)'}\n\n"
        "Generate the coarse plan now."
    )

    if state.get("conflict_reason"):
        user_prompt += (
            f"\n\nNOTE: a previous attempt was rejected for this reason, avoid repeating it:\n"
            f"{state['conflict_reason']}"
        )

    result = call_gemini(system_prompt, user_prompt, CoarsePlanOutput, "default")
    logger.info("[day_planner] coarse plan generated: %d blocks", len(result.blocks))

    return {
        **state,
        "coarse_plan": [b.model_dump() for b in result.blocks],
    }


def decompose_hourly(state: DayPlannerState) -> DayPlannerState:
    persona = state["persona"]

    atomic_blocks = [b for b in state['coarse_plan'] if b.get("granularity") == "atomic"]
    flexible_blocks = [b for b in state['coarse_plan'] if b.get("granularity") == "flexible"]
    unknown_blocks = [
        b for b in state["coarse_plan"]
        if b.get("granularity") not in {"atomic", "flexible"}
    ]
    if unknown_blocks:
        logger.warning(
            "[day_planner] treating %d unknown-granularity block(s) as flexible",
            len(unknown_blocks),
        )
        flexible_blocks.extend({**block, "granularity": "flexible"} for block in unknown_blocks)

    passthrough_hourly = [
        {
            "activity": b["activity"],
            "start": b["start"],
            "end": b["end"],
            "parent_activity": b["activity"],
            "granularity": "atomic",
            "energy_change": b.get("energy_change", 0.0),
            "emotion_change": b.get("emotion_change", 0.0),
        }
        for b in atomic_blocks
    ]

    hourly_blocks = list(passthrough_hourly)

    if flexible_blocks:
        system_prompt = (
            "You refine flexible coarse blocks into hourly-resolution blocks. Each coarse "
            "block should be broken into one or more hourly blocks that together span "
            "exactly its start/end range with no gaps or overlaps. "
            "Make each sub-block feel natural and persona-aligned — a real person would "
            "think of these as distinct steps. Consider reasonable time boundaries "
            "(e.g. a meal block of 2 hours can contain 'walk to mess', 'eat', 'socialize'). "
            "Only the blocks provided here need refining. Do not create walk, commute, or "
            "transit sub-blocks: refine only activities performed at the destination.\n\n"
            "For EACH block, assign realistic energy_change and emotion_change values:\n"
            "- energy_change: positive = restorative, negative = tiring\n"
            "- emotion_change: positive = uplifting, negative = draining\n"
            "- Routine work, classes, and meals should usually stay within -0.03 to +0.03; do not make ordinary productivity euphoric\n"
            "- Be realistic for the persona"
        )
        user_prompt = (
            f"PERSONA:\n{_persona_block(persona)}\n\n"
            f"{_flavor_block(state)}\n\n"
            f"FULL COARSE PLAN (context only):\n{json.dumps(state['coarse_plan'], indent=2)}\n\n" # Hand the whole day
            f"BLOCKS TO REFINE:\n{json.dumps(flexible_blocks, indent=2)}\n\n" # The ONLY data to modify 
            "Produce the hourly-resolution plan for the blocks listed under "
            "'BLOCKS TO REFINE' only."
        )
        result = call_gemini(system_prompt, user_prompt, HourlyPlanOutput, "default")
        refined = [b.model_dump() for b in result.blocks]
        for b in refined:
            b["granularity"] = "flexible"
        hourly_blocks.extend(refined)

    hourly_blocks.sort(key=lambda b: b["start"])
    logger.info(
        "[day_planner] hourly plan: %d atomic passthrough + %d refined",
        len(passthrough_hourly), len(hourly_blocks) - len(passthrough_hourly),
    )

    return {**state, "hourly_plan": hourly_blocks}

def decompose_fine(state: DayPlannerState) -> DayPlannerState:
    persona = state["persona"]
    places = _normalize_places(state.get("places", []))
    current_loc = state.get("current_location_id", "unknown")

    atomic_blocks = [b for b in state["hourly_plan"] if b["granularity"] == "atomic"]
    flexible_blocks = [b for b in state["hourly_plan"] if b["granularity"] == "flexible"]

    fine_actions: List[Dict[str, Any]] = []

    # Now have to handle atomic vs flexible blocks differently

    # Atomic blocks: keep as single continuous actions, just resolve location.
    if atomic_blocks:
        system_prompt = (
            "For each atomic activity below, assign exactly one location_id (and optionally "
            "a sub_area) from the known places list. Pick whichever place fits the "
            "activity best. Consider commute distance: if the agent is moving between "
            "locations, ensure the distance is reasonable for the time available. "
            "Default to locations that make sense for this specific persona. "
            "Do not suggest splitting the activity.\n\n"
            "ACADEMIC VENUE POLICY: obey the branch-specific policy provided with the persona. "
            "Do not use SAB as a generic lecture/lab default.\n\n"
            "For EACH block, assign realistic energy_change and emotion_change values:\n"
            "- energy_change: positive = restorative, negative = tiring\n"
            "- emotion_change: positive = uplifting, negative = draining\n"
            "- Routine work, classes, and meals should usually stay within -0.03 to +0.03; do not make ordinary productivity euphoric\n"
            "- Be realistic for the persona"
        )
        user_prompt = (
            f"PERSONA:\n{_persona_block(persona)}\n\n"
            f"ACADEMIC VENUE POLICY:\n{_academic_venue_policy(persona)}\n\n"
            f"{_flavor_block(state)}\n\n"
            f"ACTIVITIES:\n{json.dumps(atomic_blocks, indent=2)}\n\n"
            f"KNOWN CAMPUS LOCATIONS:\n{_places_block(places)}\n\n"
            f"The agent's current position is: {current_loc}. This is only their "
            "starting point for the next activity; it is not necessarily their hostel. "
            f"Their home/hostel is: {persona.get('Hostel', 'unknown')}. Ensure sleep, "
            "rest, and personal activities use the hostel unless the activity explicitly "
            "requires another place.\n\n"
            "Assign a location to each activity now."
        )
        result = call_gemini(system_prompt, user_prompt, AtomicLocationOutput, "default")
        # Activity labels are not unique (for example, two separate study
        # blocks). Preserve the provider's ordered assignments for duplicates
        # instead of letting a dict silently overwrite earlier entries.
        loc_by_activity: Dict[str, deque] = defaultdict(deque)
        for assignment in result.assignments:
            loc_by_activity[assignment.activity].append(assignment)

        for b in atomic_blocks:
            candidates = loc_by_activity.get(b["activity"])
            assignment = candidates.popleft() if candidates else None
            fine_actions.append({
                "action": b["activity"],
                "start": b["start"],
                "end": b["end"],
                "parent_activity": b["parent_activity"],
                "location_id": assignment.location_id if assignment else None,
                "sub_area": assignment.sub_area if assignment else None,
                "energy_change": (
                    assignment.energy_change if assignment else b.get("energy_change", 0.0)
                ),
                "emotion_change": (
                    assignment.emotion_change if assignment else b.get("emotion_change", 0.0)
                ),
            })

    # Flexible blocks: full fine-grained breakdown, as before.
    if flexible_blocks:
        system_prompt = (
            "You refine hourly blocks into fine-grained, directly executable actions "
            "at roughly 5-15 minute granularity. Each hourly block should be broken "
            "into one or more fine actions spanning exactly its start/end range, no "
            "gaps or overlaps. Every action MUST be assigned a location_id, chosen "
            "EXACTLY from the provided list -- never invent one.\n\n"
            "Do NOT output walking, commuting, travel, transit, leaving, or arriving "
            "as an action. The runtime owns visible routes between places; every action "
            "you output must be an on-site activity at its assigned location.\n\n"
            "ACADEMIC VENUE POLICY: obey the branch-specific policy provided with the persona. "
            "Branch-specific classes/labs must not silently fall back to SAB.\n\n"
            "Make action boundaries feel natural — group related sub-actions together. "
            "Consider typical on-site durations: eating ~20-40min and studying "
            "~30-120min. Keep adjacent location changes realistic by leaving enough "
            "time for the executor to animate transit before the next activity.\n\n"
            "For EACH action, assign realistic energy_change and emotion_change values:\n"
            "- energy_change: positive = restorative, negative = tiring\n"
            "- emotion_change: positive = uplifting, negative = draining\n"
            "- Routine work, classes, and meals should usually stay within -0.03 to +0.03; do not make ordinary productivity euphoric\n"
            "- Be realistic for the persona"
        )
        user_prompt = (
            f"PERSONA:\n{_persona_block(persona)}\n\n"
            f"ACADEMIC VENUE POLICY:\n{_academic_venue_policy(persona)}\n\n"
            f"{_flavor_block(state)}\n\n"
            f"HOURLY BLOCKS TO REFINE:\n{json.dumps(flexible_blocks, indent=2)}\n\n"
            f"KNOWN CAMPUS LOCATIONS:\n{_places_block(places)}\n\n"
            f"The agent's current position is: {current_loc}. This is only their "
            "starting point for the next activity; it is not necessarily their hostel. "
            f"Their home/hostel is: {persona.get('Hostel', 'unknown')}. Ensure sleep, "
            "rest, and personal activities use the hostel unless the activity explicitly "
            "requires another place.\n\n"
            "Produce the fine-grained action plan for these blocks now."
        )
        result = call_gemini(system_prompt, user_prompt, FinePlanOutput, "default")
        fine_actions.extend(a.model_dump() for a in result.actions)

    fine_actions.sort(key=lambda a: a["start"])
    logger.info("[day_planner] fine plan: %d total actions", len(fine_actions))

    return {**state, "fine_plan": fine_actions}

def _local_overlap_check(
    actions: List[Dict[str, Any]],
    mode: str = "full_day",
    remaining_start: Optional[str] = None,
) -> Optional[str]:
    """Cheap deterministic pre-check before spending an LLM call on validation --
    catches the most common failure mode (bad overlaps/gaps) for free."""
    if not actions:
        return "fine plan is empty"

    def to_minutes(hhmm: str) -> int:
        h, m = hhmm.split(":")
        return int(h) * 60 + int(m)

    try:
        sorted_actions = sorted(actions, key=lambda a: to_minutes(a["start"]))
    except Exception as e:  # malformed time strings
        return f"unparsable time value: {e}"

    # Remaining-day replans replace the *whole* active plan.  They therefore
    # must begin now and continue through midnight; otherwise a valid old-plan
    # tail would be silently discarded and the agent would become idle.
    if mode == "remaining" and remaining_start:
        if to_minutes(sorted_actions[0]["start"]) != to_minutes(remaining_start):
            return (
                f"remaining-day plan does not start at {remaining_start} "
                f"(starts at {sorted_actions[0]['start']})"
            )
    elif mode != "remaining" and to_minutes(sorted_actions[0]["start"]) != 0:
        return f"plan does not start at 00:00 (starts at {sorted_actions[0]['start']})"

    for prev, curr in zip(sorted_actions, sorted_actions[1:]):
        if to_minutes(prev["end"]) <= to_minutes(prev["start"]):
            return f"action '{prev['action']}' has a non-positive duration"
        if to_minutes(prev["end"]) != to_minutes(curr["start"]):
            return (
                f"gap or overlap between '{prev['action']}' (ends {prev['end']}) "
                f"and '{curr['action']}' (starts {curr['start']})"
            )

    if to_minutes(sorted_actions[-1]["end"]) <= to_minutes(sorted_actions[-1]["start"]):
        return f"action '{sorted_actions[-1]['action']}' has a non-positive duration"

    # Even a remaining-day plan replaces the entire current plan, so it must
    # explicitly own the final minute of the day.
    last_end = sorted_actions[-1]["end"]
    if to_minutes(last_end) not in (24 * 60, 0):
        return f"plan does not end at 24:00 (ends at {last_end})"

    return None


def _local_location_check(actions: List[Dict[str, Any]], places: List[Place]) -> Optional[str]:
    """Deterministic check that every action's location is one of the known
    places -- catches hallucinated locations without spending an LLM call."""
    places = _normalize_places(places)
    if not places:
        # No places data loaded -- nothing to validate against, skip silently.
        return None

    valid_ids = {p.id for p in places}
    by_id = {p.id: p for p in places}
    loc_err = ""
    transit_words = ("walk", "commute", "travel", "transit", "arrive", "leave")
    for a in actions:
        loc_id = a.get("location_id")
        if loc_id not in valid_ids:
            loc_err += f"action '{a.get('action')}' has invalid location_id '{loc_id}'\n"
            continue
        if any(word in a.get("action", "").lower() for word in transit_words):
            loc_err += f"action '{a.get('action')}' is transit; routes are owned by the executor\n"
        sub = a.get("sub_area")
        if sub and sub not in by_id[loc_id].sub_areas:
            loc_err += f"action '{a.get('action')}' references unknown sub_area '{sub}' for {loc_id}\n"

    return loc_err.strip() if loc_err else None


def _local_content_safety_check(actions: List[Dict[str, Any]]) -> Optional[str]:
    for action in actions:
        text = str(action.get("action", "")).lower()
        matched = next((term for term in _UNSAFE_PLAN_TERMS if term in text), None)
        if matched:
            return f"action '{action.get('action')}' contains prohibited unsafe content ({matched})"
    return None


def validate_plan(state: DayPlannerState) -> DayPlannerState:
    current_time = state.get("current_time", "")
    remaining_start = current_time.rsplit(" ", 1)[-1] if " " in current_time else None
    local_issue = _local_overlap_check(
        state["fine_plan"],
        mode=state.get("mode", "full_day"),
        remaining_start=remaining_start,
    ) or _local_location_check(
        state["fine_plan"], state.get("places", [])
    ) or _local_academic_venue_check(state["fine_plan"], state["persona"]) or _local_content_safety_check(state["fine_plan"])
    if local_issue:
        logger.info("[day_planner] local validation failed: %s", local_issue)
        return {
            **state,
            "conflict_detected": True,
            "conflict_reason": local_issue,
            "retry_count": state.get("retry_count", 0) + 1,
        }

    # Local checks passed -- do one semantic sanity pass via the LLM
    # (catches persona-incoherence, not just arithmetic).
    system_prompt = (
        "You are QA-checking a simulated character's day plan for internal consistency "
        "with their persona (not just time-math, which has already been verified). "
        "Flag it invalid only for clear persona contradictions or nonsensical sequencing."
    )
    user_prompt = (
        f"PERSONA:\n{_persona_block(state['persona'])}\n\n"
        f"DETAILED PERSONA:\n{json.dumps(state['persona'], indent=2)}\n\n"
        f"COARSE PLAN (intent and priorities):\n{json.dumps(state.get('coarse_plan', []), indent=2)}\n\n"
        f"FINE PLAN:\n{json.dumps(state['fine_plan'], indent=2)}\n\n"
        "Is this plan valid for this persona? Flag only clear persona contradictions "
        "or nonsensical sequencing."
    )
    result = call_gemini(system_prompt, user_prompt, ValidationResult, "default")

    if not result.valid:
        logger.info("[day_planner] semantic validation failed: %s", result.reason)
        return {
            **state,
            "conflict_detected": True,
            "conflict_reason": result.reason,
            "retry_count": state.get("retry_count", 0) + 1,
        }

    logger.info("[day_planner] plan validated successfully")
    return {
        **state,
        "conflict_detected": False,
        "conflict_reason": None,
        "day_plan": state["fine_plan"],
    }


# ---------------------------------------------------------------------------
# Conditional routing
# ---------------------------------------------------------------------------

def route_after_validation(state: DayPlannerState) -> str:
    if not state.get("conflict_detected"):
        return "accept"
    if state.get("retry_count", 0) >= MAX_PLAN_RETRIES:
        logger.warning(
            "[day_planner] max retries (%d) reached, force-accepting last plan with error flag",
            MAX_PLAN_RETRIES,
        )
        return "give_up"
    return "retry"


def _force_accept(state: DayPlannerState) -> DayPlannerState:
    """Terminal fallback so a stubborn persona can never infinite-loop the graph.

    This path is intentionally deterministic, but it must still honour the
    same content constraints as a normally validated plan.  Retrying a model
    output must never turn a rejected unsafe action into an accepted schedule.
    """
    # A remaining-day replan is optional.  Unlike startup/day-handoff plans,
    # it has a known-good predecessor.  Never force-accept malformed output
    # here: the caller will retain the prior plan instead.
    if state.get("mode") == "remaining":
        return {
            **state,
            "day_plan": [],
            "replan_rejected": True,
            "error": (
                f"replan rejected after {MAX_PLAN_RETRIES} retries, last issue: "
                f"{state.get('conflict_reason')}"
            ),
        }

    plan = []
    for original in sorted(state.get("fine_plan", []), key=lambda action: action.get("start", "00:00")):
        action = dict(original)
        if _local_content_safety_check([action]):
            action["action"] = "Private downtime and preparation for rest"
            logger.warning(
                "[day_planner] replaced rejected unsafe fallback activity at %s",
                action.get("start", "unknown time"),
            )
        plan.append(action)
    mode = state.get("mode", "full_day")

    # Safety normalization for full-day output: fill every uncovered period so
    # the executor never encounters a silent gap after retry exhaustion.
    if mode in ("full_day", "next_day") and plan:
        def _hhmm(m):
            h, mn = m.split(":")
            return int(h) * 60 + int(mn)
        normalized = []
        cursor = 0
        location_id = plan[0].get("location_id", "")
        for action in plan:
            start, end = _hhmm(action.get("start", "00:00")), _hhmm(action.get("end", "00:00"))
            if end <= start or start < cursor:
                continue
            if start > cursor:
                normalized.append({"action": "Free time", "start": f"{cursor // 60:02}:{cursor % 60:02}", "end": action["start"], "location_id": location_id, "sub_area": None, "energy_change": 0.0, "emotion_change": 0.0})
            normalized.append(action)
            cursor, location_id = end, action.get("location_id", location_id)
        if cursor < 24 * 60:
            normalized.append({"action": "Free time", "start": f"{cursor // 60:02}:{cursor % 60:02}", "end": "24:00", "location_id": location_id, "sub_area": None, "energy_change": 0.0, "emotion_change": 0.0})
        plan = normalized

    return {
        **state,
        "day_plan": plan,
        "error": f"accepted after {MAX_PLAN_RETRIES} retries, last issue: {state.get('conflict_reason')}",
    }


def _provider_outage_fallback(state: DayPlannerState, error: Exception) -> DayPlannerState:
    """Create a safe local schedule when every configured model is unavailable.

    This keeps the agent embodied and schedulable during a provider incident.
    It is deliberately modest and location-stable; a later replan can restore
    a personalised schedule once the provider is healthy again.
    """
    current_time = str(state.get("current_time", "00:00")).split()[-1]
    try:
        hour, minute = (int(part) for part in current_time.split(":", 1))
        start_minute = min(24 * 60, max(0, hour * 60 + minute))
    except (TypeError, ValueError):
        start_minute = 0

    location_id = state.get("current_location_id") or "campus"
    segments = [
        (0, 7 * 60, "Sleep and recover", 0.25, 0.0),
        (7 * 60, 22 * 60, "Self-directed routine and coursework", -0.05, 0.0),
        (22 * 60, 24 * 60, "Wind down and rest", 0.15, 0.0),
    ]
    plan = []
    for segment_start, segment_end, action, energy_change, emotion_change in segments:
        actual_start = max(start_minute, segment_start)
        if actual_start >= segment_end:
            continue
        plan.append({
            "action": action,
            "start": f"{actual_start // 60:02}:{actual_start % 60:02}",
            "end": f"{segment_end // 60:02}:{segment_end % 60:02}",
            "location_id": location_id,
            "sub_area": None,
            "energy_change": energy_change,
            "emotion_change": emotion_change,
        })

    logger.error(
        "[day_planner] provider fallback for %s at %s: %s",
        state.get("persona", {}).get("Name", "unknown"), current_time, error,
    )
    return {
        **state,
        "day_plan": plan,
        "error": f"LLM provider unavailable; using local continuity plan: {error}",
    }


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def build_day_planner_graph():
    graph = StateGraph(DayPlannerState)

    graph.add_node("generate_coarse_plan", generate_coarse_plan)
    graph.add_node("decompose_hourly", decompose_hourly)
    graph.add_node("decompose_fine", decompose_fine)
    graph.add_node("validate_plan", validate_plan)
    graph.add_node("force_accept", _force_accept)

    graph.set_entry_point("generate_coarse_plan")
    graph.add_edge("generate_coarse_plan", "decompose_hourly")
    graph.add_edge("decompose_hourly", "decompose_fine")
    graph.add_edge("decompose_fine", "validate_plan")

    graph.add_conditional_edges(
        "validate_plan",
        route_after_validation,
        {
            "accept": END,
            "retry": "generate_coarse_plan",
            "give_up": "force_accept",
        },
    )
    graph.add_edge("force_accept", END)

    return graph.compile()


_compiled_graph = None
_compiled_graph_lock = threading.Lock()


def get_compiled_graph():
    global _compiled_graph
    if _compiled_graph is None:
        with _compiled_graph_lock:
            if _compiled_graph is None:
                _compiled_graph = build_day_planner_graph()
    return _compiled_graph


# ---------------------------------------------------------------------------
# AgentModule-compatible entrypoint
#
# Matches the run(agent, world_state) -> dict contract from core/module_base.py
# discussed earlier, so this drops straight into the tick orchestrator/registry
# once that's wired up. Falls back to a plain function if module_base isn't
# importable yet (e.g. running this file standalone).
# ---------------------------------------------------------------------------

def run(agent: Any, world_state: dict) -> dict:
    """
    agent is expected to expose:
        agent.persona            -> dict
        agent.relevant_memories  -> List[str]   (empty for now)
        agent.yesterday_summary  -> Optional[str] (empty for now)
    world_state is expected to expose:
        'current_time' key (str)
        optionally a 'places' key (List[Dict[str, str]]) -- if omitted,
            places are loaded from places.json via load_places().
        'persona_name' is used when saving the final plan to data/Short_term_db.
        'mode' can be 'full_day' (default), 'remaining', or 'next_day'.
        'current_location_id' optionally tells the planner where the agent is
            (useful for remaining-day and next-day planning).
    """
    places = world_state.get("places") or load_places()
    mode = world_state.get("mode", "full_day")

    from src.agents.daily_flavor import pick_theme, pick_emotion
    theme = world_state.get("daily_theme") or pick_theme()
    emotion = world_state.get("daily_emotion") or pick_emotion()

    initial_state: DayPlannerState = {
        "persona": getattr(agent, "persona", {}),
        "relevant_memories": getattr(agent, "relevant_memories", []) or [],
        "yesterday_summary": getattr(agent, "yesterday_summary", None),
        "current_time": world_state.get("current_time", "00:00"),
        "places": places,
        "mode": mode,
        "current_location_id": world_state.get("current_location_id"),
        "handoff_context": world_state.get("handoff_context"),
        "upcoming_events": world_state.get("upcoming_events", []),
        "daily_theme": theme,
        "daily_emotion": emotion,
        "retry_count": 0,
    }

    # The Gemini caller already rotates keys/models under a strict deadline.
    # Retrying the complete multi-step graph (formerly 10s then 15s sleeps)
    # only multiplied a provider outage into a startup stall.
    try:
        final_state = get_compiled_graph().invoke(initial_state)
    except ProviderFailureError:
        raise
    except (AllModelsFailedError, genai_errors.APIError) as exc:
        final_state = _provider_outage_fallback(initial_state, exc)

    # Save day plan to short-term memory
    from src.agents.Short_term import save_day_plan, date_from_simulation_time
    
    sim_date = date_from_simulation_time(world_state.get("current_time", "00:00"))
    save_day_plan(initial_state["persona"].get("Name") or initial_state["persona"].get("name", "unknown"), sim_date, final_state.get("day_plan", []))

    return {
        "day_plan": final_state.get("day_plan", []),
        "error": final_state.get("error"),
        "memory_entries": [
            f"Planned today: {len(final_state.get('day_plan', []))} scheduled actions."
        ],
    }


try:
    from src.core.module_base import AgentModule
    from src.core.registry import register_module


    @register_module
    class DayPlanner(AgentModule):
        name = "day_planner"

        def run(self, agent, world_state):
            return run(agent, world_state)

except ImportError:
    # core/module_base.py and core/registry.py not wired up yet -- fine,
    # `run()` above works standalone in the meantime.
    pass

# ---------------------------------------------------------------------------
# Standalone smoke test
#
# Uses only a persona -- relevant_memories and yesterday_summary are left
# empty, exactly as requested for this stage.
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    from src.config import PERSONALITIES_DIR

    parser = argparse.ArgumentParser(description="Run the day planner smoke test")
    parser.add_argument(
        "persona",
        nargs="?",
        default="gurnoor_singh",
        help="Persona name or persona JSON path (for example: tanishq or tanishq/tanishq.json)",
    )
    args = parser.parse_args()

    def resolve_persona_path(persona_arg: str) -> Path:
        candidate_path = Path(persona_arg)
        if candidate_path.exists():
            return candidate_path

        if candidate_path.suffix == ".json":
            matches = list(PERSONALITIES_DIR.glob(f"**/{candidate_path.name}"))
            if matches:
                return matches[0]

        matches = sorted(PERSONALITIES_DIR.glob(f"**/{persona_arg}/{persona_arg}.json"))
        if matches:
            return matches[0]

        matches = sorted(PERSONALITIES_DIR.glob(f"**/{persona_arg}.json"))
        if matches:
            return matches[0]

        candidates = sorted(PERSONALITIES_DIR.glob("**/*.json"))
        if not candidates:
            raise FileNotFoundError(f"No persona JSON files found under {PERSONALITIES_DIR}")
        raise FileNotFoundError(
            f"Could not find persona '{persona_arg}'. Available personas: "
            f"{', '.join(sorted({p.parent.name for p in candidates}))}"
        )

    sample_persona_path = resolve_persona_path(args.persona)

    sample_persona = json.loads(sample_persona_path.read_text())

    class _FakeAgent:
        persona = sample_persona
        relevant_memories: List[str] = []
        yesterday_summary: Optional[str] = None


    # Swap for load_places() once your real data/environment/places.json exists.
    # WorrdState is expected to have the Places

    result = run(
        _FakeAgent(),
        {"current_time": "2026-07-03 06:00", "places": None, "persona_name": sample_persona_path.stem},
    )

    print(result)

    col_widths = (10, 10, 60, 20)
    header = (
        f"{'START':<{col_widths[0]}}{'END':<{col_widths[1]}}"
        f"{'ACTION':<{col_widths[2]}}{'LOCATION':<{col_widths[3]}}"
    )
    print(header)
    print("-" * sum(col_widths))
    for item in result["day_plan"]:
        action = item.get("action", "")
        if len(action) > col_widths[2] - 2:
            action = action[: col_widths[2] - 5] + "..."
        print(
            f"{item['start']:<{col_widths[0]}}{item['end']:<{col_widths[1]}}"
            f"{action:<{col_widths[2]}}{item.get('location_id', ''):<{col_widths[3]}}"
        )

    if result.get("memory_entries"):
        print("\nMemory entries logged:")
        for m in result["memory_entries"]:
            print(f"  - {m}")

