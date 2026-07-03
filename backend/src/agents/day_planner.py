"""
<<<<<<< HEAD
Placeholder for the day-planner agent.
Will receive high-level goals and break them into a sequence of
sub-goals / waypoints for the navigation system to execute.
"""
=======
A script that plans the day of an agentic personality when handed over required data

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
import logging
from typing import Any, Dict, List, Optional, TypedDict

from google import genai
from google.genai import types
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


# ---------------------------------------------------------------------------
# Config -> in config.py
# ---------------------------------------------------------------------------
from src.config import GEMINI_MODEL, GEMINI_API_KEY, MAX_PLAN_RETRIES, ENVIRONMENT_DIR

# genai.Client() picks up GEMINI_API_KEY (or GOOGLE_API_KEY) from the
_client: Optional[genai.Client] = None


def get_client() -> genai.Client | None:
    """Lazy singleton so importing this module doesn't require an API key
    to be set (useful for tests that mock the client)."""
    global _client
    if _client is None:
        _client = genai.Client(api_key=GEMINI_API_KEY)
    return _client


def load_places(places_file: Optional[Path] = None) -> List[Dict[str, str]]:
    """
    Loads the college places JSON and normalizes it to a flat list of
    {"name": ..., "description": ...} dicts, regardless of which of these

    Gets the places from ENVIRONMENT_DIR/places.json by default, or from a custom path if provided.
    """
    path = places_file or ENVIRONMENT_DIR / "places.json"

    if not Path(path).exists():
        logger.warning("[day_planner] places file not found at %s -- proceeding with no known locations", path)
        return []

    # Getting the raw JSON data
    raw = json.loads(Path(path).read_text())

    return [{"name" : loc["name"], "description" : " ".join(f"{key}={value}" for key, value in loc)} for loc in raw["locations"]]


def _places_block(places: List[Dict[str, str]]) -> str:
    if not places:
        return "(no places data available -- pick a reasonable generic label)"
    lines = []
    for p in places:
        if p.get("description"):
            lines.append(f"- {p['name']}: {p['description']}")
        else:
            lines.append(f"- {p['name']}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Structured-output schemas (pydantic -> Gemini response_schema)
# ---------------------------------------------------------------------------

class CoarseBlock(BaseModel):
    activity: str = Field(description="Short label, e.g. 'work on farm', 'breakfast'")
    start: str = Field(description="24h HH:MM")
    end: str = Field(description="24h HH:MM")


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
    action: str = Field(description="Concrete, executable action, 5-15 min granularity")
    start: str
    end: str
    parent_activity: str = Field(description="The hourly block this refines")
    location: str = Field(description="Must exactly match one of the provided known place names")


class FinePlanOutput(BaseModel):
    actions: List[FineAction]


class ValidationResult(BaseModel):
    valid: bool
    reason: Optional[str] = Field(
        default=None, description="If invalid: which items conflict/overlap/gap and why"
    )


# ---------------------------------------------------------------------------
# Graph state
# ---------------------------------------------------------------------------

class DayPlannerState(TypedDict, total=False):
    # ---- inputs ----
    persona: dict
    relevant_memories: List[str]
    yesterday_summary: Optional[str]
    current_time: str  # e.g. "2026-07-03 06:00"
    places: List[Dict[str, str]]  # known campus locations, from places.json

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
    """Renders whatever fields exist in the persona json -- don't assume a
    fixed schema beyond name, since personas will vary as you add more."""
    name = persona.get("name", "The agent")
    lines = [f"Name: {name}"]
    for key in  ("Name", "Age", "Gender", "Branch", "daily_plan_req", "innate", "learned", "lifestyle", "hobbies", "goals",):
        if key in persona and persona[key]:
            lines.append(f"{key.replace('_', ' ').title()}: {persona[key]}")
    return "\n".join(lines)


def _memories_block(memories: List[str]) -> str:
    if not memories:
        return "(none available yet)"
    return "\n".join(f"- {m}" for m in memories)


def _call_gemini(system_prompt: str, user_prompt: str, schema: type[BaseModel]) -> BaseModel | None:
    """Single point of contact with Gemini. Structured JSON output via
    response_schema means no fragile manual JSON parsing/regex."""
    response = get_client().models.generate_content(
        model=GEMINI_MODEL,
        contents=user_prompt,
        config=types.GenerateContentConfig(
            system_instruction=system_prompt,
            response_mime_type="application/json",
            response_schema=schema,
            temperature=0.7,
        ),
    )
    # google-genai gives you .parsed already instantiated as `schema`,
    # but we fall back to manual parsing defensively.
    if getattr(response, "parsed", None) is not None:
        return response.parsed
    return schema.model_validate(json.loads(response.text))


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

def generate_coarse_plan(state: DayPlannerState) -> DayPlannerState:
    persona = state["persona"]
    system_prompt = (
        "You are simulating one day in the life of a character in a generative-agents "
        "simulation. Produce a COARSE day plan: 5 to 8 broad blocks of activity covering "
        "the full 24 hours (00:00 to 24:00), with no gaps and no overlaps. "
        "Stay true to the persona's traits and background."
    )
    user_prompt = (
        f"PERSONA:\n{_persona_block(persona)}\n\n"
        f"RELEVANT MEMORIES:\n{_memories_block(state.get('relevant_memories', []))}\n\n"
        f"YESTERDAY'S SUMMARY:\n{state.get('yesterday_summary') or '(no history yet, this is day 1)'}\n\n"
        f"Current in-simulation time: {state.get('current_time', 'unknown')}\n\n"
        "Generate the coarse plan now."
    )

    if state.get("conflict_reason"):
        user_prompt += (
            f"\n\nNOTE: a previous attempt was rejected for this reason, avoid repeating it:\n"
            f"{state['conflict_reason']}"
        )

    result = _call_gemini(system_prompt, user_prompt, CoarsePlanOutput)
    logger.info("[day_planner] coarse plan generated: %d blocks", len(result.blocks))

    return {
        **state,
        "coarse_plan": [b.model_dump() for b in result.blocks],
    }


def decompose_hourly(state: DayPlannerState) -> DayPlannerState:
    persona = state["persona"]
    system_prompt = (
        "You refine a coarse day plan into hourly-resolution blocks. Each coarse block "
        "should be broken into one or more hourly blocks that together span exactly its "
        "start/end range. Do not introduce gaps or overlaps."
    )
    user_prompt = (
        f"PERSONA:\n{_persona_block(persona)}\n\n"
        f"COARSE PLAN:\n{json.dumps(state['coarse_plan'], indent=2)}\n\n"
        "Produce the hourly-resolution plan now."
    )

    result = _call_gemini(system_prompt, user_prompt, HourlyPlanOutput)
    logger.info("[day_planner] hourly plan generated: %d blocks", len(result.blocks))

    return {
        **state,
        "hourly_plan": [b.model_dump() for b in result.blocks],
    }


def decompose_fine(state: DayPlannerState) -> DayPlannerState:
    persona = state["persona"]
    system_prompt = (
        "You refine an hourly day plan into fine-grained, directly executable actions "
        "at roughly 5-15 minute granularity. Each hourly block should be broken into one "
        "or more fine actions spanning exactly its start/end range, no gaps or overlaps. "
        "Every action MUST be assigned a location, chosen EXACTLY (character-for-character) "
        "from the provided list of known places -- never invent a location that isn't listed."
    )
    user_prompt = (
        f"PERSONA:\n{_persona_block(persona)}\n\n"
        f"HOURLY PLAN:\n{json.dumps(state['hourly_plan'], indent=2)}\n\n"
        f"KNOWN CAMPUS LOCATIONS (pick 'location' only from these):\n{_places_block(state.get('places', []))}\n\n"
        "Produce the fine-grained action plan now, with a location for every action."
    )

    result = _call_gemini(system_prompt, user_prompt, FinePlanOutput)
    logger.info("[day_planner] fine plan generated: %d actions", len(result.actions))

    return {
        **state,
        "fine_plan": [a.model_dump() for a in result.actions],
    }


def _local_overlap_check(actions: List[Dict[str, Any]]) -> Optional[str]:
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

    if to_minutes(sorted_actions[0]["start"]) != 0:
        return f"plan does not start at 00:00 (starts at {sorted_actions[0]['start']})"

    for prev, curr in zip(sorted_actions, sorted_actions[1:]):
        if to_minutes(prev["end"]) != to_minutes(curr["start"]):
            return (
                f"gap or overlap between '{prev['action']}' (ends {prev['end']}) "
                f"and '{curr['action']}' (starts {curr['start']})"
            )

    last_end = sorted_actions[-1]["end"]
    if to_minutes(last_end) not in (24 * 60, 0):
        return f"plan does not end at 24:00 (ends at {last_end})"

    return None


def _local_location_check(actions: List[Dict[str, Any]], places: List[Dict[str, str]]) -> Optional[str]:
    """Deterministic check that every action's location is one of the known
    places -- catches hallucinated locations without spending an LLM call."""
    if not places:
        # No places data loaded -- nothing to validate against, skip silently.
        return None

    valid_names = {p["name"] for p in places}
    for a in actions:
        loc = a.get("location")
        if not loc:
            return f"action '{a.get('action')}' has no location assigned"

        # Skipping this for now, have to add more places
        # if loc not in valid_names:
        #     return f"action '{a.get('action')}' has invalid location '{loc}' (not in known places)"

    return None


def validate_plan(state: DayPlannerState) -> DayPlannerState:
    local_issue = _local_overlap_check(state["fine_plan"]) or _local_location_check(
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
    result = _call_gemini(system_prompt, user_prompt, ValidationResult)

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
    world_state is expected to expose a 'current_time' key (str), and
    optionally a 'places' key (List[Dict[str, str]]) -- if omitted, places
    are loaded from places.json via load_places().
    """
    places = world_state.get("places") or load_places()

    initial_state: DayPlannerState = {
        "persona": getattr(agent, "persona", {}),
        "relevant_memories": getattr(agent, "relevant_memories", []) or [],
        "yesterday_summary": getattr(agent, "yesterday_summary", None),
        "current_time": world_state.get("current_time", "00:00"),
        "places": places,
        "retry_count": 0,
    }

    final_state = get_compiled_graph().invoke(initial_state)

    return {
        "day_plan": final_state.get("day_plan", []),
        "memory_entries": [
            f"Planned today: {len(final_state.get('day_plan', []))} scheduled actions."
        ],
    }


try:
    from backend.src.core.module_base import AgentModule
    from backend.src.core.registry import register_module


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
    sample_persona = {
  "Name": "Tanishq",
  "Age": "20",
  "Gender": "Male",
  "Branch": "B.Tech Artificial Intelligence and Data Engineering",
  "Home City": "Faridabad, India",
  "daily_plan_req": "Wake up late, attend CS lectures while secretly browsing forums, spend afternoons in library reading dark romance novels and coding side projects, evenings jerking off to fantasies, and late nights gooning to porn.",
  "innate": "Naturally introverted, socially awkward, with strong racial prejudices and an intense, obsessive attraction to women that borders on dangerous fixation.",
  "learned": "Mastered advanced programming, deepened knowledge of dark romance tropes, improved skills in manipulating conversations to simp for girls, and honed abilities in hiding inappropriate behaviors.",
  "lifestyle": "Irregular sleep with late nights gooning, minimal exercise but tall stature, mostly solitary with occasional awkward social interactions where he simps heavily, avoids diverse groups due to racist views.",
  "hobbies": "Reading dark romance novels, gooning to explicit content, fantasizing about non-consensual scenarios, coding nerdy projects, and lurking on controversial online forums.",
  "goals": "Short-term: Ace exams while avoiding detection for creepy behaviors; Long-term: Become a successful software engineer and live out dark romance fantasies in real life."
}


    class _FakeAgent:
        persona = sample_persona
        relevant_memories: List[str] = []
        yesterday_summary: Optional[str] = None


    # Swap for load_places() once your real data/environment/places.json exists.
    sample_places = [
        {"name": "Hostel Room", "description": "Gurnoor's dorm room"},
        {"name": "Mess Hall", "description": "canteen for meals"},
        {"name": "CS Building", "description": "classes and labs"},
        {"name": "Library", "description": "quiet study space"},
        {"name": "Sports Ground", "description": "outdoor exercise area"},
    ]

    result = run(
        _FakeAgent(),
        {"current_time": "2026-07-03 06:00", "places": sample_places},
    )

    print(result)

    col_widths = (10, 10, 45, 20)
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
            f"{action:<{col_widths[2]}}{item.get('location', ''):<{col_widths[3]}}"
        )

    if result.get("memory_entries"):
        print("\nMemory entries logged:")
        for m in result["memory_entries"]:
            print(f"  - {m}")


>>>>>>> 2c8eb1aca80139d958822218fffc1ba00c5a10f5
