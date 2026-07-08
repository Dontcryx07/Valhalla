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
from src.core.log import get_logger
from src.llm.gemini_client import call_gemini, AllModelsFailedError
from typing import Any, Dict, List, Optional, TypedDict, Literal

from google import genai
from google.genai import types
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field

# Setting up logger
logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Config -> in config.py
# ---------------------------------------------------------------------------
from src.config import GEMINI_MODEL, GEMINI_API_KEY, ENVIRONMENT_DIR, DATA_DIR
from src.config import MAX_PLAN_RETRIES, PERSONA_FIELD_GLOSSARY


# genai.Client() picks up GEMINI_API_KEY (or GOOGLE_API_KEY) from the
_client: genai.Client = genai.Client(api_key=GEMINI_API_KEY)


def get_client() -> genai.Client | None:
    """Returns google-genai client """
    global _client
    if _client is None:
        _client = genai.Client(api_key=GEMINI_API_KEY)
    return _client


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

    raw = json.loads(Path(path).read_text())
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


class CoarsePlanOutput(BaseModel):
    blocks: List[CoarseBlock]


class HourlyBlock(BaseModel):
    activity: str
    start: str
    end: str
    parent_activity: str = Field(description="The coarse block this refines")


class HourlyPlanOutput(BaseModel):
    blocks: List[HourlyBlock]


class FineAction(BaseModel):
    action: str = Field(description="Concrete, executable action, 5-30 min granularity")
    start: str
    end: str
    parent_activity: str = Field(description="The hourly block this refines")
    location_id: str = Field(description="Must exactly match an 'id' from the known places list")
    sub_area: Optional[str] = Field(default=None, description="One of that place's sub_areas, if applicable")


class FinePlanOutput(BaseModel):
    actions: List[FineAction]

# For Coars Atomic actions
class AtomicLocationAssignment(BaseModel):
    activity: str
    location_id: str
    sub_area: Optional[str] = None

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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _persona_block(persona: dict) -> str:
    """Renders whatever fields exist in the persona JSON -- don't assume a
    fixed schema beyond name, since personas will vary as you add more."""
    lines = [f"Name: {persona.get('name', 'Unknown')}"]
    for field, meaning in PERSONA_FIELD_GLOSSARY.items():
        value = persona.get(field)
        if value:
            lines.append(f"- {field} ({meaning}): {value}")

    identity_fields = ["Age", "Gender", "Branch", "Home City"]
    lines.extend(f"{f}: {persona[f]}" for f in identity_fields if persona.get(f))

    return "\n".join(lines)


def _memories_block(memories: List[str]) -> str:
    if not memories:
        return "(none available yet)"
    return "\n".join(f"- {m}" for m in memories)


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

def generate_coarse_plan(state: DayPlannerState) -> DayPlannerState:
    persona = state["persona"]
    mode = state.get("mode", "full_day")
    current_time = state.get("current_time", "unknown")

    if mode == "full_day":
        coverage = "covering the full 24 hours (00:00 to 24:00)"
    elif mode == "next_day":
        coverage = "covering the full next day (00:00 to 24:00)"
    elif mode == "remaining":
        coverage = f"covering ONLY the REMAINDER of the day from the current time onward -- do NOT schedule anything before the current time"
    else:
        coverage = "covering the full 24 hours (00:00 to 24:00)"

    loc_hint = ""
    current_loc = state.get("current_location_id")
    if current_loc:
        loc_hint = f"\nThe agent is currently at: {current_loc}. Start the plan from this location."

    system_prompt = (
        "You are simulating one day in the life of a character in a generative-agents "
        "simulation. Produce a COARSE day plan: 5 to 8 broad blocks of activity "
        f"{coverage}, with no gaps and no overlaps. "
        "Stay true to the persona's traits and background.\n\n"
        "For EACH block, tag it as 'atomic' or 'flexible':\n"
        "- atomic: a single continuous activity with no meaningful internal sub-steps "
        "worth planning separately. Examples: sleeping, attending a class/lecture, "
        "commuting/travel, sitting an exam, watching a movie, a long uninterrupted "
        "study/deep-work session.\n"
        "- flexible: an activity that naturally contains distinct sub-activities a person "
        "would actually think of as separate steps. Examples: 'morning routine' (wake up, "
        "shower, get dressed), 'gym session' (warm-up, lifting, cooldown), 'dinner with "
        "friends' (walk over, eat, chat).\n"
        "When in doubt, prefer 'atomic' -- do not manufacture sub-steps for something a "
        "person would just describe as one thing."
    )
    user_prompt = (
        f"PERSONA:\n{_persona_block(persona)}\n\n"
        f"RELEVANT MEMORIES:\n{_memories_block(state.get('relevant_memories', []))}\n\n"
        f"YESTERDAY'S SUMMARY:\n{state.get('yesterday_summary') or '(no history yet, this is day 1)'}\n\n"
        f"Current in-simulation time: {current_time}\n"
        f"Plan mode: {mode}\n"
        f"Agent location: {current_loc or 'unknown'}{loc_hint}\n\n"
        "Generate the coarse plan now."
    )

    if state.get("conflict_reason"):
        user_prompt += (
            f"\n\nNOTE: a previous attempt was rejected for this reason, avoid repeating it:\n"
            f"{state['conflict_reason']}"
        )

    result = call_gemini(system_prompt, user_prompt, CoarsePlanOutput, "complex")
    logger.info("[day_planner] coarse plan generated: %d blocks", len(result.blocks))

    return {
        **state,
        "coarse_plan": [b.model_dump() for b in result.blocks],
    }


def decompose_hourly(state: DayPlannerState) -> DayPlannerState:
    persona = state["persona"]

    atomic_blocks = [b for b in state['coarse_plan'] if b["granularity"] == "atomic"]
    flexible_blocks = [b for b in state['coarse_plan'] if b["granularity"] == "flexible"]

    passthrough_hourly = [
        {
            "activity": b["activity"],
            "start": b["start"],
            "end": b["end"],
            "parent_activity": b["activity"],
            "granularity": "atomic",
        }
        for b in atomic_blocks
    ]

    hourly_blocks = list(passthrough_hourly)

    if flexible_blocks:
        system_prompt = (
            "You refine a coarse day plan into hourly-resolution blocks. Each coarse "
            "block should be broken into one or more hourly blocks that together span "
            "exactly its start/end range. Do not introduce gaps or overlaps. "
            "Only the blocks provided here need refining -- they have already been "
            "identified as containing genuine sub-activities."
        )
        user_prompt = (
            f"PERSONA:\n{_persona_block(persona)}\n\n"
            f"FULL COARSE PLAN (context only):\n{json.dumps(state['coarse_plan'], indent=2)}\n\n" # Hand the whole day
            f"BLOCKS TO REFINE:\n{json.dumps(flexible_blocks, indent=2)}\n\n" # The ONLY data to modify 
            "Produce the hourly-resolution plan for the blocks listed under "
            "'BLOCKS TO REFINE' only."
        )
        result = call_gemini(system_prompt, user_prompt, HourlyPlanOutput)
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
    places = state.get("places", [])

    atomic_blocks = [b for b in state["hourly_plan"] if b["granularity"] == "atomic"]
    flexible_blocks = [b for b in state["hourly_plan"] if b["granularity"] == "flexible"]

    fine_actions: List[Dict[str, Any]] = []

    # Now have to handle atomic vs flexible blocks differently

    # Atomic blocks: keep as single continuous actions, just resolve location.
    if atomic_blocks:
        system_prompt = (
            "For each activity below, assign exactly one location_id (and optionally "
            "a sub_area) from the known places list -- pick whichever place fits the "
            "activity best. Do not suggest splitting the activity."
        )
        user_prompt = (
            f"ACTIVITIES:\n{json.dumps(atomic_blocks, indent=2)}\n\n"
            f"KNOWN CAMPUS LOCATIONS:\n{_places_block(places)}\n\n"
            "Assign a location to each activity now."
        )
        result = call_gemini(system_prompt, user_prompt, AtomicLocationOutput)
        loc_by_activity = {a.activity: a for a in result.assignments}

        for b in atomic_blocks:
            assignment = loc_by_activity.get(b["activity"])
            fine_actions.append({
                "action": b["activity"],
                "start": b["start"],
                "end": b["end"],
                "parent_activity": b["parent_activity"],
                "location_id": assignment.location_id if assignment else None,
                "sub_area": assignment.sub_area if assignment else None,
            })

    # Flexible blocks: full fine-grained breakdown, as before.
    if flexible_blocks:
        system_prompt = (
            "You refine hourly blocks into fine-grained, directly executable actions "
            "at roughly 5-15 minute granularity. Each hourly block should be broken "
            "into one or more fine actions spanning exactly its start/end range, no "
            "gaps or overlaps. Every action MUST be assigned a location_id, chosen "
            "EXACTLY from the provided list -- never invent one."
        )
        user_prompt = (
            f"PERSONA:\n{_persona_block(persona)}\n\n"
            f"HOURLY BLOCKS TO REFINE:\n{json.dumps(flexible_blocks, indent=2)}\n\n"
            f"KNOWN CAMPUS LOCATIONS:\n{_places_block(places)}\n\n"
            "Produce the fine-grained action plan for these blocks now."
        )
        result = call_gemini(system_prompt, user_prompt, FinePlanOutput)
        fine_actions.extend(a.model_dump() for a in result.actions)

    fine_actions.sort(key=lambda a: a["start"])
    logger.info("[day_planner] fine plan: %d total actions", len(fine_actions))

    return {**state, "fine_plan": fine_actions}

def _local_overlap_check(actions: List[Dict[str, Any]], mode: str = "full_day") -> Optional[str]:
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

    # Remaining-day plans start at the current time, not necessarily 00:00
    if mode != "remaining" and to_minutes(sorted_actions[0]["start"]) != 0:
        return f"plan does not start at 00:00 (starts at {sorted_actions[0]['start']})"

    for prev, curr in zip(sorted_actions, sorted_actions[1:]):
        if to_minutes(prev["end"]) != to_minutes(curr["start"]):
            return (
                f"gap or overlap between '{prev['action']}' (ends {prev['end']}) "
                f"and '{curr['action']}' (starts {curr['start']})"
            )

    # Remaining-day plans don't need to perfectly hit 24:00
    if mode != "remaining":
        last_end = sorted_actions[-1]["end"]
        if to_minutes(last_end) not in (24 * 60, 0):
            return f"plan does not end at 24:00 (ends at {last_end})"

    return None


def _local_location_check(actions: List[Dict[str, Any]], places: List[Place]) -> Optional[str]:
    """Deterministic check that every action's location is one of the known
    places -- catches hallucinated locations without spending an LLM call."""
    if not places:
        # No places data loaded -- nothing to validate against, skip silently.
        return None

    valid_ids = {p.id for p in places}
    by_id = {p.id: p for p in places}
    loc_err = ""
    for a in actions:
        loc_id = a.get("location_id")
        if loc_id not in valid_ids:
            loc_err += f"action '{a.get('action')}' has invalid location_id '{loc_id}'\n"
        sub = a.get("sub_area")
        if sub and sub not in by_id[loc_id].sub_areas:
            loc_err += f"action '{a.get('action')}' references unknown sub_area '{sub}' for {loc_id}\n"

    return loc_err.strip() if loc_err else None


def validate_plan(state: DayPlannerState) -> DayPlannerState:
    local_issue = _local_overlap_check(state["fine_plan"], mode=state.get("mode", "full_day")) or _local_location_check(
        state["fine_plan"], state.get("places", [])
    )
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
        f"FINE PLAN:\n{json.dumps(state['fine_plan'], indent=2)}\n\n"
        "Is this plan valid?"
    )
    result = call_gemini(system_prompt, user_prompt, ValidationResult, "complex")

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
    """Terminal fallback so a stubborn persona can never infinite-loop the graph."""
    return {
        **state,
        "day_plan": state.get("fine_plan", []),
        "error": f"accepted after {MAX_PLAN_RETRIES} retries, last issue: {state.get('conflict_reason')}",
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


def get_compiled_graph():
    global _compiled_graph
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

    initial_state: DayPlannerState = {
        "persona": getattr(agent, "persona", {}),
        "relevant_memories": getattr(agent, "relevant_memories", []) or [],
        "yesterday_summary": getattr(agent, "yesterday_summary", None),
        "current_time": world_state.get("current_time", "00:00"),
        "places": places,
        "mode": mode,
        "current_location_id": world_state.get("current_location_id"),
        "retry_count": 0,
    }

    # LangGraph context-preservation: if ALL model calls fail across every
    # API key (AllModelsFailedError), we catch it here and retry the *entire
    # graph* from scratch with the same input state.  The per-key rate limiter
    # in gemini_client.py will use a different key on retry, so a transient
    # quota exhaustion across all keys resolves itself on the next attempt.
    import time as _time
    _max_graph_retries = 3
    _last_exc: Exception | None = None
    for _attempt in range(_max_graph_retries):
        try:
            final_state = get_compiled_graph().invoke(initial_state)
            _last_exc = None
            break
        except AllModelsFailedError as _exc:
            _last_exc = _exc
            logger.warning(
                "[day_planner] all models failed (attempt %d/%d) — "
                "retaining state and retrying after backoff.",
                _attempt + 1, _max_graph_retries,
            )
            if _attempt < _max_graph_retries - 1:
                _time.sleep(10.0 + _attempt * 5.0)
            continue

    if _last_exc is not None:
        raise _last_exc  # Give up after exhausting graph retries

    # Save day plan to short-term memory
    from src.agents.Short_term import save_day_plan, date_from_simulation_time
    
    sim_date = date_from_simulation_time(world_state.get("current_time", "00:00"))
    save_day_plan(initial_state["persona"].get("Name") or initial_state["persona"].get("name", "unknown"), sim_date, final_state.get("day_plan", []))

    return {
        "day_plan": final_state.get("day_plan", []),
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
        default="gurnoor",
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

