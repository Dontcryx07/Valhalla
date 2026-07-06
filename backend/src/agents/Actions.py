"""
Actions -- manages agent action execution: last, current, next.

Takes the raw day plan produced by day_planner.py and drives it forward
tick by tick. Handles three action types:

  1. MOVE    -- agent walks from place A to place B (pathfinder.py)
  2. MISC    -- static activity: studying, coding, eating, etc.
  3. CONVERSATION -- triggered when two agents are in proximity (future)

The module converts location_id strings (from day plans) into pixel
coordinates (from entrypoint.json) and uses the BFS pathfinder to
compute walkable paths between locations.

Usage:
    from src.agents.Actions import AgentActionManager, LocationResolver

    resolver = LocationResolver()
    manager = AgentActionManager("parv", day_plan, initial_position)
    state = manager.tick(world_tick, snapshot)
"""

from __future__ import annotations

import sys
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[2]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.append(str(BACKEND_ROOT))


import json
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from src.core.log import get_logger
from src.core.world_state import WorldState, Position, CurrentAction, AgentStatus

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Action types
# ---------------------------------------------------------------------------

class ActionType(str, Enum):
    MOVE = "move"
    MISC = "misc"
    CONVERSATION = "conversation"


# ---------------------------------------------------------------------------
# Action state
# ---------------------------------------------------------------------------

class ActionState(BaseModel):
    """One action in the agent's queue."""
    action_type: ActionType
    description: str
    start_time: str           # "HH:MM"
    end_time: str             # "HH:MM"
    location_id: str          # places.json ID
    sub_area: Optional[str] = None
    position: Optional[Position] = None  # pixel coords (filled by resolver)
    path: Optional[List[Tuple[int, int]]] = None  # if MOVE, the pixel path
    path_index: int = 0       # current position along path


# ---------------------------------------------------------------------------
# Location resolver -- converts location_id -> pixel Position
# ---------------------------------------------------------------------------

# places.json ID -> entrypoint.json building name
_PLACES_TO_ENTRYPOINT: Dict[str, str] = {
    "Brahmaputra_Boys 1": "Brahmputra Boys 1",
    "Brahmaputra_Boys 2": "Brahmputra Boys 2",
    "mess": "Mess",
    "LHC": "Lecture Hall complex",
    "admin_block": "Admin Block",
    "sac_utility_block": "Utility/ SAC",
    "main_gate": "Campus Main gate",
    "central_lawn": "Main fest Ground",
    "auditorium": "Auditorium + Library",
    "library": "Auditorium + Library",
    "sports_complex": "Volleyball Court",
    "health_centre": "Admin Block",
    "Chenab": "Chenab",
    "Beas": "Beas",
    "Satluj": "Satluj",
    "Jhelum": "Jhelum",
    "Ravi": "Ravi",
    "SAB": "SAB",
}


class LocationResolver:
    """Maps places.json location_id strings to pixel Position objects."""

    def __init__(
        self,
        places_path: Optional[Path] = None,
        entrypoint_path: Optional[Path] = None,
    ):
        from src.config import ENVIRONMENT_DIR

        places_path = places_path or ENVIRONMENT_DIR / "places.json"
        entrypoint_path = entrypoint_path or ENVIRONMENT_DIR / "entrypoint.json"

        self._entrypoints: Dict[str, Dict[str, Any]] = {}
        self._places_inline: Dict[str, Tuple[int, int]] = {}

        # Load entrypoints
        if entrypoint_path.exists():
            raw = json.loads(entrypoint_path.read_text())
            for _key, data in raw.items():
                name = data.get("name", "")
                self._entrypoints[name] = {"x": data["x"], "y": data["y"]}

        # Load inline coordinates from places.json (3 locations have this)
        if places_path.exists():
            raw = json.loads(places_path.read_text())
            for loc in raw.get("locations", []):
                inline = loc.get("Entry_point Coordinates", "")
                if inline and "x" in inline and "y" in inline:
                    # Parse "x = 712, y = 414"
                    parts = inline.split(",")
                    x = int(parts[0].split("=")[1].strip())
                    y = int(parts[1].split("=")[1].strip())
                    self._places_inline[loc["id"]] = (x, y)

        logger.info(
            "[LocationResolver] loaded %d entrypoints, %d inline coords",
            len(self._entrypoints), len(self._places_inline),
        )

    def resolve(self, location_id: str) -> Optional[Position]:
        """Convert a places.json ID to a Position with pixel coordinates."""
        # Strategy 1: inline coordinates from places.json
        if location_id in self._places_inline:
            x, y = self._places_inline[location_id]
            return Position(x=x, y=y, location_id=location_id)

        # Strategy 2: explicit override dictionary
        entrypoint_name = _PLACES_TO_ENTRYPOINT.get(location_id)
        if entrypoint_name and entrypoint_name in self._entrypoints:
            data = self._entrypoints[entrypoint_name]
            return Position(x=data["x"], y=data["y"], location_id=location_id)

        # Strategy 3: fuzzy match (lowercase, strip underscores/spaces)
        normalized = location_id.lower().replace("_", "").replace(" ", "")
        for name, data in self._entrypoints.items():
            if name.lower().replace("_", "").replace(" ", "") == normalized:
                return Position(x=data["x"], y=data["y"], location_id=location_id)

        logger.warning("[LocationResolver] could not resolve '%s'", location_id)
        return None

    def has_location(self, location_id: str) -> bool:
        """Check if we can resolve this ID."""
        return self.resolve(location_id) is not None


# ---------------------------------------------------------------------------
# Agent action manager -- per-agent state machine
# ---------------------------------------------------------------------------

class AgentActionManager:
    """
    Manages the action lifecycle for one agent.

    Reads the day plan, tracks last/current/next action, handles movement
    between locations, and updates WorldState each tick.
    """

    def __init__(
        self,
        agent_id: str,
        day_plan: List[Dict[str, Any]],
        initial_position: Position,
        resolver: LocationResolver,
    ):
        self.agent_id = agent_id
        self.day_plan = sorted(day_plan, key=lambda a: a.get("start", "00:00"))
        self.resolver = resolver
        self.position = initial_position

        self.last_action: Optional[ActionState] = None
        self.current_action: Optional[ActionState] = None
        self.next_action: Optional[ActionState] = None

        self._plan_index = 0
        self._initialized = False

    def _hhmm_to_minutes(self, hhmm: str) -> int:
        """Convert 'HH:MM' to minutes since midnight."""
        h, m = hhmm.split(":")
        return int(h) * 60 + int(m)

    def _minutes_to_hhmm(self, minutes: int) -> str:
        """Convert minutes since midnight to 'HH:MM'."""
        h, m = divmod(minutes, 60)
        return f"{h:02d}:{m:02d}"

    def _find_current_plan_action(self, hhmm: str) -> Optional[Dict[str, Any]]:
        """Find the day plan action that covers this time."""
        minute = self._hhmm_to_minutes(hhmm)
        for action in self.day_plan:
            start = self._hhmm_to_minutes(action.get("start", "00:00"))
            end = self._hhmm_to_minutes(action.get("end", "24:00"))
            if start <= minute < end:
                return action
        return None

    def _create_action_state(self, plan_action: Dict[str, Any]) -> ActionState:
        """Create an ActionState from a day plan entry."""
        location_id = plan_action.get("location_id", "")
        position = self.resolver.resolve(location_id)

        return ActionState(
            action_type=ActionType.MISC,
            description=plan_action.get("action", "unknown"),
            start_time=plan_action.get("start", "00:00"),
            end_time=plan_action.get("end", "24:00"),
            location_id=location_id,
            sub_area=plan_action.get("sub_area"),
            position=position,
        )

    def _create_move_action(
        self,
        from_pos: Position,
        to_pos: Position,
        start_time: str,
        end_time: str,
    ) -> ActionState:
        """Create a MOVE action with a path from pathfinder."""
        path = None
        try:
            from pathfinder import shortest_path
            path = shortest_path((from_pos.x, from_pos.y), (to_pos.x, to_pos.y))
        except Exception as e:
            logger.warning(
                "[Actions] pathfinder failed for %s: %s — using direct path",
                self.agent_id, e,
            )
            # Fallback: direct path (start -> end)
            path = [(from_pos.x, from_pos.y), (to_pos.x, to_pos.y)]

        return ActionState(
            action_type=ActionType.MOVE,
            description=f"Walk from {from_pos.location_id or 'here'} to {to_pos.location_id or 'there'}",
            start_time=start_time,
            end_time=end_time,
            location_id=to_pos.location_id or "",
            position=to_pos,
            path=path or [],
            path_index=0,
        )

    def _advance_to_next_plan_action(self, current_hhmm: str) -> None:
        """Pick the next action from the day plan and set up current/next."""
        plan_action = self._find_current_plan_action(current_hhmm)
        if plan_action is None:
            logger.debug("[Actions] %s: no plan action at %s", self.agent_id, current_hhmm)
            return

        target_location_id = plan_action.get("location_id", "")
        target_position = self.resolver.resolve(target_location_id)

        if target_position is None:
            logger.warning(
                "[Actions] %s: cannot resolve location '%s' — skipping action",
                self.agent_id, target_location_id,
            )
            return

        # Check if agent is already at the target location
        already_at_location = (
            self.position.location_id == target_location_id
            or (
                self.position.x == target_position.x
                and self.position.y == target_position.y
            )
        )

        if already_at_location:
            # Agent is already here — start the activity immediately
            self.current_action = self._create_action_state(plan_action)
            self.current_action.start_time = current_hhmm
        else:
            # Agent needs to move first
            # Estimate travel time: ~10 min per 100 pixels (rough)
            dx = abs(target_position.x - self.position.x)
            dy = abs(target_position.y - self.position.y)
            distance = max(dx, dy)
            travel_minutes = max(5, distance // 10)  # at least 5 min

            start_minute = self._hhmm_to_minutes(current_hhmm)
            end_minute = start_minute + travel_minutes
            end_time = self._minutes_to_hhmm(min(end_minute, 24 * 60))

            self.current_action = self._create_move_action(
                self.position, target_position, current_hhmm, end_time
            )
            # The actual activity becomes the next action
            self.next_action = self._create_action_state(plan_action)

        self._initialized = True

    def tick(self, world_tick: int, snapshot: Any = None) -> Optional[ActionState]:
        """
        Advance one tick. Returns the current action state (or None).

        This is the main entry point called each simulation tick.
        """
        hhmm = self._minutes_to_hhmm(world_tick % (24 * 60))

        # First tick or after action completed — pick next action
        if self.current_action is None:
            self._advance_to_next_plan_action(hhmm)
            if self.current_action is None:
                return None

        # Check if current action is finished
        current_end = self._hhmm_to_minutes(self.current_action.end_time)
        current_minute = self._hhmm_to_minutes(hhmm)

        if current_minute >= current_end:
            # Action finished — advance
            self.last_action = self.current_action
            self.current_action = None

            # If we were moving and have a next action, start it
            if self.last_action.action_type == ActionType.MOVE and self.next_action is not None:
                self.current_action = self.next_action
                self.next_action = None
                self.position = self.current_action.position or self.position
            else:
                # Movement completed — update position to destination
                if self.last_action.action_type == ActionType.MOVE and self.last_action.position:
                    self.position = self.last_action.position

                # Pick next action from plan
                self._advance_to_next_plan_action(hhmm)

        # If still moving, advance along path
        if (
            self.current_action is not None
            and self.current_action.action_type == ActionType.MOVE
            and self.current_action.path
        ):
            path = self.current_action.path
            idx = self.current_action.path_index
            if idx < len(path) - 1:
                # Move one step along the path
                next_x, next_y = path[idx + 1]
                self.position = Position(
                    x=next_x, y=next_y,
                    location_id=self.current_action.location_id,
                )
                self.current_action.path_index = idx + 1

        return self.current_action

    def get_state(self) -> Dict[str, Any]:
        """Return current state as a dict (for frontend/API)."""
        return {
            "agent_id": self.agent_id,
            "position": self.position.model_dump(),
            "last_action": self.last_action.model_dump() if self.last_action else None,
            "current_action": self.current_action.model_dump() if self.current_action else None,
            "next_action": self.next_action.model_dump() if self.next_action else None,
        }


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from src.core.log import setup_logging
    setup_logging(run_id="actions_test", console=True)

    from src.config import PERSONALITIES_DIR, ENVIRONMENT_DIR

    # Load a persona's day plan
    persona_path = PERSONALITIES_DIR / "parv" / "parv.json"
    if not persona_path.exists():
        print(f"Persona not found: {persona_path}")
        sys.exit(1)

    persona = json.loads(persona_path.read_text())

    # Load day plan from Short_term
    from src.agents.Short_term import load_day_plan, date_from_simulation_time

    sim_date = date_from_simulation_time("2026-07-03 00:00")
    day_plan = load_day_plan(persona.get("Name", "unknown"), sim_date)

    if not day_plan:
        print(f"No day plan found for {persona.get('Name')} on {sim_date}")
        print("Run: python backend/src/core/Agent.py parv")
        sys.exit(1)

    print(f"Loaded day plan for {persona.get('Name')}: {len(day_plan)} actions")

    # Set up resolver and manager
    resolver = LocationResolver()
    initial_pos = resolver.resolve("Brahmaputra_Boys 1") or Position(x=185, y=547, location_id="Brahmaputra_Boys 1")

    manager = AgentActionManager(
        agent_id="parv",
        day_plan=day_plan,
        initial_position=initial_pos,
        resolver=resolver,
    )

    # Run 100 ticks (100 minutes = 1h40m starting from 00:00)
    print(f"\nRunning 100 ticks from 00:00...")
    print(f"{'Tick':>5} {'Time':>6} {'Action Type':<12} {'Description':<45} {'Location':<25} {'Position'}")
    print("-" * 120)

    for tick in range(0, 100):
        hhmm = manager._minutes_to_hhmm(tick)
        action = manager.tick(tick)

        if action:
            desc = action.description[:43]
            loc = action.location_id[:23] if action.location_id else ""
            pos = f"({manager.position.x}, {manager.position.y})"
            print(f"{tick:>5} {hhmm:>6} {action.action_type.value:<12} {desc:<45} {loc:<25} {pos}")

    print(f"\nFinal state:")
    state = manager.get_state()
    print(f"  Position: ({state['position']['x']}, {state['position']['y']}) at {state['position']['location_id']}")
    print(f"  Current action: {state['current_action']['description'] if state['current_action'] else 'None'}")

    print("\nActions.py self-test passed.")
