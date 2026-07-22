"""
Actions -- manages agent action execution: last, current, next.

Takes the raw day plan produced by day_planner.py and drives it forward
tick by tick. Handles three action types:

  1. MOVE    -- agent walks from place A to place B (pathfinder.py)
  2. MISC    -- static activity: studying, coding, eating, etc.
  3. CONVERSATION -- triggered when two agents are in proximity.

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
import math
import random
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from src.core.log import get_logger
from src.core.world_state import Position

logger = get_logger(__name__)


def _interpolate_line(a: Tuple[int, int], b: Tuple[int, int], steps: int = 40) -> List[Tuple[int, int]]:
    """A straight line of `steps` points from a to b (inclusive). Used as a
    fallback route so agents still animate when no walkable path is found."""
    (x0, y0), (x1, y1) = a, b
    steps = max(2, steps)
    return [
        (round(x0 + (x1 - x0) * i / (steps - 1)), round(y0 + (y1 - y0) * i / (steps - 1)))
        for i in range(steps)
    ]


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
    energy_change: float = 0.0      # total change over entire action
    emotion_change: float = 0.0     # total change over entire action
    is_final_plan_action: bool = False
    event_id: Optional[str] = None  # data-driven world event, when applicable


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
        self._name_to_polykey: Dict[str, str] = {}   # entrypoint name -> "PolygonBuilding_N"
        self._bbox_by_name: Dict[str, Tuple[int, int, int, int]] = {}  # name -> (xmin,ymin,xmax,ymax)

        # Load entrypoints
        if entrypoint_path.exists():
            raw = json.loads(entrypoint_path.read_text())
            for key, data in raw.items():
                name = data.get("name", "")
                self._entrypoints[name] = {"x": data["x"], "y": data["y"]}
                self._name_to_polykey[name] = key

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

        # Load building bounding boxes: union all decomposed parts per polygon.
        bbox_path = ENVIRONMENT_DIR / "buildings_polygon_decomposed.json"
        polykey_bbox: Dict[str, Tuple[int, int, int, int]] = {}
        if bbox_path.exists():
            try:
                parts = json.loads(bbox_path.read_text())
                for p in parts:
                    base = p.get("building_name", "").rsplit("_part", 1)[0]
                    corners = [p.get("top_left"), p.get("top_right"),
                               p.get("bottom_left"), p.get("bottom_right")]
                    xs = [c[0] for c in corners if c]
                    ys = [c[1] for c in corners if c]
                    if not xs or not ys:
                        continue
                    x0, y0, x1, y1 = min(xs), min(ys), max(xs), max(ys)
                    if base in polykey_bbox:
                        px0, py0, px1, py1 = polykey_bbox[base]
                        polykey_bbox[base] = (min(px0, x0), min(py0, y0), max(px1, x1), max(py1, y1))
                    else:
                        polykey_bbox[base] = (x0, y0, x1, y1)
            except Exception as e:
                logger.warning("[LocationResolver] failed to load building bboxes: %s", e)
        # Link entrypoint name -> bbox via the shared polygon key.
        for name, key in self._name_to_polykey.items():
            if key in polykey_bbox:
                self._bbox_by_name[name] = polykey_bbox[key]

        logger.info(
            "[LocationResolver] loaded %d entrypoints, %d inline coords, %d building bboxes",
            len(self._entrypoints), len(self._places_inline), len(self._bbox_by_name),
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

    def _entrypoint_name_for(self, location_id: str) -> Optional[str]:
        """Resolve a location_id to its entrypoint building name (for bbox lookup)."""
        name = _PLACES_TO_ENTRYPOINT.get(location_id)
        if name and name in self._entrypoints:
            return name
        normalized = location_id.lower().replace("_", "").replace(" ", "")
        for nm in self._entrypoints:
            if nm.lower().replace("_", "").replace(" ", "") == normalized:
                return nm
        return None

    def bbox_for(self, location_id: str) -> Optional[Tuple[int, int, int, int]]:
        """Return (xmin, ymin, xmax, ymax) for a location's building, or None."""
        name = self._entrypoint_name_for(location_id)
        if name and name in self._bbox_by_name:
            return self._bbox_by_name[name]
        return None

    def random_interior_point(
        self,
        location_id: str,
        occupied: Optional[List[Tuple[int, int]]] = None,
        min_gap: int = 6,
        max_tries: int = 40,
    ) -> Position:
        """A random pixel INSIDE the building's bounding box, kept at least
        `min_gap` px away from any `occupied` point. Falls back to a small
        jitter around the entrypoint when no bbox is known."""
        occupied = occupied or []
        bbox = self.bbox_for(location_id)
        ep = self.resolve(location_id)

        if bbox is not None:
            x0, y0, x1, y1 = bbox
            # small inset so points aren't exactly on the wall
            if x1 - x0 > 4:
                x0, x1 = x0 + 2, x1 - 2
            if y1 - y0 > 4:
                y0, y1 = y0 + 2, y1 - 2
        elif ep is not None:
            x0, y0, x1, y1 = ep.x - 14, ep.y - 14, ep.x + 14, ep.y + 14
        else:
            return Position(x=0, y=0, location_id=location_id)

        best = None
        for _ in range(max_tries):
            x = random.randint(int(x0), int(x1))
            y = random.randint(int(y0), int(y1))
            if all((x - ox) ** 2 + (y - oy) ** 2 >= min_gap * min_gap for ox, oy in occupied):
                return Position(x=x, y=y, location_id=location_id)
            best = (x, y)
        x, y = best if best else (int((x0 + x1) / 2), int((y0 + y1) / 2))
        return Position(x=x, y=y, location_id=location_id)


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

        # Conversation support
        self._conversation_mode: bool = False
        self._pending_plan_action: Optional[ActionState] = None

        # Last-action detection (end-of-day transition)
        self._entered_last_action: bool = False

    @property
    def is_last_action(self) -> bool:
        """True if the current action is the final entry in the day plan.
        Uses description match only — start_time may differ when rescheduled
        after a conversation or a long move that shifted the timeline."""
        return bool(self.current_action and self.current_action.is_final_plan_action)

    @property
    def is_idle(self) -> bool:
        """
        True when the day plan is fully consumed — no action covers the
        current time and there are no more plan entries.
        """
        return self.current_action is None

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
            energy_change=plan_action.get("energy_change", 0.0),
            emotion_change=plan_action.get("emotion_change", 0.0),
            is_final_plan_action=bool(self.day_plan and plan_action is self.day_plan[-1]),
            event_id=plan_action.get("world_event_id"),
        )

    def _create_move_action(
        self,
        from_pos: Position,
        to_pos: Position,
        start_time: str,
        end_time: str,
    ) -> ActionState:
        """Create a MOVE action that walks the agent DOOR to DOOR.

        The route is: current spot -> current building's entry door ->
        (pathfinder route along the walkable grid) -> destination building's
        entry door. Arrival then settles the agent inside the destination
        (handled in tick()). If the pathfinder can't produce a route we fall
        back to a densely interpolated straight line so motion is still visible.
        """
        # Resolve the two front doors (walkable entrypoints).
        from_door = self.resolver.resolve(from_pos.location_id) if from_pos.location_id else None
        to_door = self.resolver.resolve(to_pos.location_id) if to_pos.location_id else None
        d_from = (from_door.x, from_door.y) if from_door else (from_pos.x, from_pos.y)
        d_to = (to_door.x, to_door.y) if to_door else (to_pos.x, to_pos.y)

        road: List[Tuple[int, int]] = []
        try:
            from pathfinder import shortest_path
            road = shortest_path(d_from, d_to) or []
        except Exception as e:
            logger.warning("[Actions] pathfinder failed for %s: %s", self.agent_id, e)
            road = []

        if len(road) < 2:
            road = _interpolate_line(d_from, d_to, steps=40)

        # Prepend the agent's current spot so it visibly steps out to its door.
        path: List[Tuple[int, int]] = [(from_pos.x, from_pos.y)] + [p for p in road]

        return ActionState(
            action_type=ActionType.MOVE,
            description=f"Walk from {from_pos.location_id or 'here'} to {to_pos.location_id or 'there'}",
            start_time=start_time,
            end_time=end_time,
            location_id=to_pos.location_id or "",
            position=to_pos,
            path=path,
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
            # Agent needs to move first — create move action with BFS path,
            # then compute travel time from the schedule's allocated duration
            # so the agent arrives exactly when the plan expects.
            self.current_action = self._create_move_action(
                self.position, target_position, current_hhmm, "23:59"
            )
            path = self.current_action.path
            path_len = len(path) if path else 100
            # Schedule-aware speed: arrive by the plan's end time.
            plan_end_min = self._hhmm_to_minutes(plan_action.get("end", "24:00"))
            start_minute = self._hhmm_to_minutes(current_hhmm)
            activity_minutes = max(1, plan_end_min - start_minute)
            min_on_site_minutes = min(10, max(1, activity_minutes // 2))
            max_travel_minutes = max(1, activity_minutes - min_on_site_minutes)
            natural_travel_minutes = max(1, math.ceil(path_len / 50))
            # Transit must leave time for the scheduled on-site activity.
            travel_minutes = min(natural_travel_minutes, max_travel_minutes)
            end_minute = start_minute + travel_minutes
            self.current_action.end_time = self._minutes_to_hhmm(min(end_minute, 24 * 60))
            # The actual activity becomes the next action
            self.next_action = self._create_action_state(plan_action)

    def set_conversation_action(self, other_agent_id: str, world_tick: Optional[int] = None) -> ActionState:
        """
        Override the current action with a conversation (no pre-set duration).
        Agent stays in conversation mode until resume_from_conversation() is
        called by the background LLM task.
        """
        hhmm = self._minutes_to_hhmm((world_tick or 0) % (24 * 60))
        if self.current_action:
            self._pending_plan_action = self.current_action

        conv_action = ActionState(
            action_type=ActionType.CONVERSATION,
            description=f"Chatting with {other_agent_id}",
            start_time=hhmm,
            end_time="23:59",  # placeholder — overridden by resume_from_conversation
            location_id=self.position.location_id or "",
            position=self.position,
        )
        self.current_action = conv_action
        self._conversation_mode = True
        return conv_action

    def resume_from_conversation(
        self,
        new_day_plan: Optional[List[Dict[str, Any]]] = None,
        world_tick: Optional[int] = None,
    ) -> None:
        """
        Leave conversation mode and resume the interrupted action when the
        plan has not changed. A paused route resumes from its actual point.
        """
        if new_day_plan is not None:
            self.day_plan = sorted(new_day_plan, key=lambda a: a.get("start", "00:00"))
        pending = self._pending_plan_action
        self._conversation_mode = False
        self._pending_plan_action = None
        self._entered_last_action = False
        if pending is None or new_day_plan is not None:
            self.current_action = None
            self.next_action = None
            return

        if pending.action_type == ActionType.MOVE and pending.path:
            old_path = pending.path
            old_index = min(pending.path_index, len(old_path) - 1)
            fraction_done = old_index / max(1, len(old_path) - 1)
            pending.path = [(self.position.x, self.position.y)] + old_path[old_index + 1:]
            pending.path_index = 0
            if world_tick is not None:
                duration = max(1, self._hhmm_to_minutes(pending.end_time) - self._hhmm_to_minutes(pending.start_time))
                remaining = max(1, math.ceil(duration * (1.0 - fraction_done)))
                now = world_tick % (24 * 60)
                pending.start_time = self._minutes_to_hhmm(now)
                pending.end_time = self._minutes_to_hhmm(min(24 * 60, now + remaining))
            if len(pending.path) < 2:
                # The conversation happened at the final waypoint.  Do not
                # leave a one-point MOVE that cannot interpolate; commit the
                # arrival and continue its queued on-site activity instead.
                if pending.location_id:
                    self.position = self.resolver.random_interior_point(pending.location_id)
                if self.next_action is not None:
                    self.current_action = self.next_action
                    self.next_action = None
                else:
                    self.current_action = None
                return
        self.current_action = pending

    def replace_day_plan(self, new_day_plan: List[Dict[str, Any]]) -> None:
        """
        Replace the day plan mid-stream (e.g. at end-of-day transition).
        Keeps the current action if it's still valid, otherwise advances.
        """
        self.day_plan = sorted(new_day_plan, key=lambda a: a.get("start", "00:00"))
        self._pending_plan_action = None
        self._entered_last_action = False

    def begin_new_day(self, new_day_plan: List[Dict[str, Any]]) -> None:
        """Install a new calendar day's plan without moving the agent.

        Position is intentionally retained: the first action of the new plan
        must account for where the previous day actually ended.  Schedule
        bookkeeping is reset so a previous day's ``24:00`` action cannot wrap
        around and continue into the next day.
        """
        self.day_plan = sorted(new_day_plan, key=lambda a: a.get("start", "00:00"))
        self.last_action = self.current_action
        self.current_action = None
        self.next_action = None
        self._pending_plan_action = None
        self._conversation_mode = False
        self._entered_last_action = False

    def tick(self, world_tick: int, snapshot: Any = None) -> Optional[ActionState]:
        """
        Advance one tick. Returns the current action state (or None).

        This is the main entry point called each simulation tick.
        """
        hhmm = self._minutes_to_hhmm(world_tick % (24 * 60))

        # In conversation mode: don't advance, just return current action.
        # The agent stays frozen until resume_from_conversation() is called.
        if self._conversation_mode:
            return self.current_action

        # First tick or after action completed — pick next action
        if self.current_action is None:
            self._advance_to_next_plan_action(hhmm)
            if self.current_action is None:
                return None

        # Check if current action is finished
        current_end = self._hhmm_to_minutes(self.current_action.end_time)
        current_minute = self._hhmm_to_minutes(hhmm)

        # Let a MOVE render its terminal path coordinate for one tick before
        # committing the arrival and changing its action label.
        is_finished = (
            current_minute > current_end
            if self.current_action.action_type == ActionType.MOVE
            else current_minute >= current_end
        )
        if is_finished:
            # Action finished — advance
            was_last = self.is_last_action
            self.last_action = self.current_action
            self.current_action = None

            # If we were moving and have a next action, start it
            if self.last_action.action_type == ActionType.MOVE and self.next_action is not None:
                self.current_action = self.next_action
                self.next_action = None
                # Arrived at the destination building — settle at a random spot
                # INSIDE its footprint (not stacked on the entrypoint).
                if self.current_action.location_id:
                    self.position = self.resolver.random_interior_point(
                        self.current_action.location_id
                    )
                else:
                    self.position = self.current_action.position or self.position
            else:
                # Movement completed — settle inside the destination building
                if self.last_action.action_type == ActionType.MOVE and self.last_action.location_id:
                    self.position = self.resolver.random_interior_point(
                        self.last_action.location_id
                    )
                elif self.last_action.action_type == ActionType.MOVE and self.last_action.position:
                    self.position = self.last_action.position

                # Pick next action from plan
                self._advance_to_next_plan_action(hhmm)

            # If this was the last action and there's a next action (from the plan),
            # it's not really the last — clear the flag
            if was_last and self.current_action is not None:
                self._entered_last_action = False

        # Detect first tick of the last action
        if self.current_action and self.is_last_action and not self._entered_last_action:
            self._entered_last_action = True

        # If still moving, advance along the path proportionally to elapsed
        # time, so the agent visibly walks the whole route (start -> destination)
        # over the move's duration instead of jumping.
        if (
            self.current_action is not None
            and self.current_action.action_type == ActionType.MOVE
            and self.current_action.path
        ):
            path = self.current_action.path
            total = len(path)
            if total >= 2:
                start_min = self._hhmm_to_minutes(self.current_action.start_time)
                end_min = self._hhmm_to_minutes(self.current_action.end_time)
                dur = max(1, end_min - start_min)
                progress = (current_minute - start_min) / dur
                progress = min(1.0, max(0.0, progress))
                # Fractional index so we move continuously *between* path points,
                # not just land on them — smooth even for short/sparse routes.
                fidx = progress * (total - 1)
                i = int(fidx)
                if i >= total - 1:
                    nx, ny = path[-1]
                else:
                    frac = fidx - i
                    (ax, ay), (bx, by) = path[i], path[i + 1]
                    nx = round(ax + (bx - ax) * frac)
                    ny = round(ay + (by - ay) * frac)
                # A route has a destination, but the agent is not semantically
                # *at* that destination until arrival.  Retaining the current
                # location prevents UI, replanning, and conversations from
                # treating an in-transit agent as already there.
                self.position = Position(x=nx, y=ny, location_id=self.position.location_id)
                self.current_action.path_index = i

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
    import argparse

    from src.core.log import setup_logging
    setup_logging(run_id="actions_test", console=True)

    from src.config import PERSONALITIES_DIR

    parser = argparse.ArgumentParser(description="Actions module self-test — simulate an agent's day")
    parser.add_argument(
        "persona", nargs="?", default="parv",
        help="Persona name (e.g. parv, tanishq, gurnoor) or path to a persona JSON",
    )
    parser.add_argument(
        "-t", "--ticks", type=int, default=600,
        help="Number of ticks to simulate (default: 600 = 10 hours)",
    )
    parser.add_argument(
        "--step", type=int, default=5,
        help="Print status every N ticks (default: 5)",
    )
    args = parser.parse_args()

    # Resolve persona path
    candidate = Path(args.persona)
    if candidate.exists():
        persona_path = candidate
    elif candidate.suffix == ".json":
        matches = sorted(PERSONALITIES_DIR.glob(f"**/{candidate.name}"))
        persona_path = matches[0] if matches else candidate
    else:
        matches = sorted(PERSONALITIES_DIR.glob(f"**/{args.persona}/{args.persona}.json"))
        if not matches:
            matches = sorted(PERSONALITIES_DIR.glob(f"**/{args.persona}.json"))
        if not matches:
            available = sorted({p.parent.name for p in PERSONALITIES_DIR.glob("**/*.json")})
            print(f"Persona '{args.persona}' not found. Available: {', '.join(available)}")
            sys.exit(1)
        persona_path = matches[0]

    persona = json.loads(persona_path.read_text())
    persona_name = persona.get("Name", persona_path.stem)

    # Load day plan from Short_term
    from src.agents.Short_term import load_day_plan, date_from_simulation_time

    sim_date = date_from_simulation_time("2026-07-03 00:00")
    day_plan = load_day_plan(persona_name, sim_date)

    if not day_plan:
        print(f"No day plan found for {persona_name} on {sim_date}")
        print("Generate one first: python backend/src/core/Agent.py " + args.persona)
        sys.exit(1)

    print(f"Loaded day plan for {persona_name}: {len(day_plan)} actions")

    # Set up resolver and manager
    resolver = LocationResolver()
    initial_pos = resolver.resolve("Brahmaputra_Boys 1") or Position(x=185, y=547, location_id="Brahmaputra_Boys 1")

    manager = AgentActionManager(
        agent_id=persona_path.stem,
        day_plan=day_plan,
        initial_position=initial_pos,
        resolver=resolver,
    )

    # Run simulation
    print(f"\nSimulating {args.ticks} ticks (step={args.step}) from 00:00...")
    print(f"{'Tick':>5} {'Time':>6} {'Type':<6} {'Description':<45} {'Location':<25} {'Position'}")
    print("-" * 110)

    for tick in range(0, args.ticks):
        hhmm = manager._minutes_to_hhmm(tick)
        action = manager.tick(tick)

        if tick % args.step == 0 and action:
            desc = action.description[:43]
            loc = action.location_id[:23] if action.location_id else ""
            pos = f"({manager.position.x},{manager.position.y})"
            print(f"{tick:>5} {hhmm:>6} {action.action_type.value:<6} {desc:<45} {loc:<25} {pos}")

    print(f"\nFinal state:")
    state = manager.get_state()
    print(f"  Position: ({state['position']['x']}, {state['position']['y']}) at {state['position']['location_id']}")
    print(f"  Current action: {state['current_action']['description'] if state['current_action'] else 'None'}")

    print("\nActions.py self-test passed.")
