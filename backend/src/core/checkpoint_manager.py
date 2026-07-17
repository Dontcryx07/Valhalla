"""
Checkpoint Manager — per-tick state save/load for crash recovery.

Saves WorldState + AgentRegistry after every tick to
``backend/data/checkpoints/tick_{00001}.json``.

Supports:
  - Save: full simulation state as JSON
  - Load: reconstruct from any saved tick
  - List: available checkpoint ticks
  - Prune: auto-delete old checkpoints, keep last N
"""

from __future__ import annotations

import json
import time as _time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.core.log import get_logger
from src.core.world_state import WorldState, Position
from src.core.agent_registry import AgentRegistry, AgentRuntimeState
from src.agents.Actions import AgentActionManager, ActionState, ActionType, LocationResolver
from src.config import DATA_DIR

logger = get_logger(__name__)

CHECKPOINT_DIR = DATA_DIR / "checkpoints"
HISTORY_DIR = DATA_DIR / "history"
KEEP_LAST = 10


def _tick_filename(tick: int) -> str:
    return f"tick_{tick:05d}.json"


def _parse_tick_from_name(name: str) -> int:
    return int(name.replace("tick_", "").replace(".json", ""))


# ------------------------------------------------------------------ #
# Save
# ------------------------------------------------------------------ #

def _manager_to_dict(manager: AgentActionManager) -> Dict[str, Any]:
    """Serialize an AgentActionManager's internal state."""
    def _action_dict(a: Optional[ActionState]) -> Optional[Dict[str, Any]]:
        if a is None:
            return None
        return {
            "action_type": a.action_type.value,
            "description": a.description,
            "start_time": a.start_time,
            "end_time": a.end_time,
            "location_id": a.location_id,
            "sub_area": a.sub_area,
            "path": a.path,
            "path_index": a.path_index,
        }

    return {
        "day_plan": manager.day_plan,
        "last_action": _action_dict(manager.last_action),
        "current_action": _action_dict(manager.current_action),
        "next_action": _action_dict(manager.next_action),
        "_plan_index": manager._plan_index,
        "_initialized": manager._initialized,
        "_conversation_mode": manager._conversation_mode,
        "_pending_plan_action": _action_dict(manager._pending_plan_action),
        "_entered_last_action": manager._entered_last_action,
    }


def _agent_to_dict(state: AgentRuntimeState) -> Dict[str, Any]:
    """Serialize one agent's runtime state."""
    return {
        "agent_id": state.agent_id,
        "persona": state.persona,
        "persona_name": state.persona_name,
        "position": state.position.model_dump(),
        "paused": state.paused,
        "day_plan": state.day_plan,
        "day_archived": state.day_archived,
        "conversation_count": state.conversation_count,
        "color": state.color,
        "manager": _manager_to_dict(state.manager) if state.manager else None,
    }


def save_checkpoint(
    world: WorldState,
    registry: AgentRegistry,
    tick: int,
) -> Optional[Path]:
    """Save current simulation state to a checkpoint JSON file."""
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    path = CHECKPOINT_DIR / _tick_filename(tick)

    agents_data = [_agent_to_dict(s) for s in registry.all_states()]

    world_dict = json.loads(world.model_dump_json())
    world_dict.pop("history", None)

    payload = {
        "tick": tick,
        "world": world_dict,
        "agents": agents_data,
        "saved_at": _time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str))
    tmp.replace(path)

    return path


# ------------------------------------------------------------------ #
# History
# ------------------------------------------------------------------ #

def save_history(world: WorldState, date: str) -> Optional[Path]:
    """Save per-day action history to ``data/history/<date>_history.json``."""
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    path = HISTORY_DIR / f"{date}_history.json"
    history_data = [entry.model_dump() for entry in world.history]
    path.write_text(json.dumps(history_data, indent=2, default=str))
    logger.info("[Checkpoint] saved history for %s (%d entries)", date, len(history_data))
    return path


# ------------------------------------------------------------------ #
# Load
# ------------------------------------------------------------------ #

def _action_from_dict(d: Optional[Dict[str, Any]]) -> Optional[ActionState]:
    """Reconstruct an ActionState from a saved dict."""
    if d is None:
        return None
    return ActionState(
        action_type=ActionType(d["action_type"]),
        description=d["description"],
        start_time=d["start_time"],
        end_time=d["end_time"],
        location_id=d.get("location_id", ""),
        sub_area=d.get("sub_area"),
        path=d.get("path"),
        path_index=d.get("path_index", 0),
    )


def _restore_manager(
    data: Dict[str, Any],
    resolver: LocationResolver,
) -> AgentActionManager:
    """Reconstruct an AgentActionManager from saved state dict."""
    position = Position(**data["position"])
    manager = AgentActionManager(
        agent_id=data["agent_id"],
        day_plan=data.get("day_plan", data["manager"]["day_plan"]) if data.get("manager") else [],
        initial_position=position,
        resolver=resolver,
    )

    if data.get("manager"):
        m = data["manager"]
        manager.last_action = _action_from_dict(m.get("last_action"))
        manager.current_action = _action_from_dict(m.get("current_action"))
        manager.next_action = _action_from_dict(m.get("next_action"))
        manager._plan_index = m.get("_plan_index", 0)
        manager._initialized = m.get("_initialized", False)
        manager._conversation_mode = m.get("_conversation_mode", False)
        manager._pending_plan_action = _action_from_dict(m.get("_pending_plan_action"))
        manager._entered_last_action = m.get("_entered_last_action", False)

    return manager


def load_checkpoint(
    tick: int,
    resolver: LocationResolver,
) -> Tuple[WorldState, AgentRegistry]:
    """Load simulation state from a checkpoint file."""
    path = CHECKPOINT_DIR / _tick_filename(tick)
    if not path.exists():
        raise FileNotFoundError(f"No checkpoint for tick {tick} at {path}")

    raw = json.loads(path.read_text())

    world = WorldState.model_validate(raw["world"])
    registry = AgentRegistry()

    for agent_data in raw["agents"]:
        position = Position(**agent_data["position"])
        manager = _restore_manager(agent_data, resolver)

        state = AgentRuntimeState(
            agent_id=agent_data["agent_id"],
            persona=agent_data["persona"],
            persona_name=agent_data["persona_name"],
            manager=manager,
            position=position,
            paused=agent_data["paused"],
            day_plan=agent_data["day_plan"],
            day_archived=agent_data["day_archived"],
            conversation_count=agent_data.get("conversation_count", 0),
            color=agent_data.get("color", "#888888"),
        )
        state.current_action = (
            manager.current_action.model_dump()
            if manager.current_action else None
        )
        registry.register(state)

    logger.info(
        "[Checkpoint] loaded tick %d: %d agents, world.tick=%d",
        tick, len(registry), world.tick,
    )
    return world, registry


# ------------------------------------------------------------------ #
# List & Prune
# ------------------------------------------------------------------ #

def list_checkpoints() -> List[int]:
    """Return sorted list of available checkpoint tick numbers."""
    if not CHECKPOINT_DIR.exists():
        return []
    ticks = []
    for f in sorted(CHECKPOINT_DIR.iterdir()):
        if f.is_file() and f.name.startswith("tick_") and f.suffix == ".json":
            try:
                ticks.append(_parse_tick_from_name(f.stem))
            except (ValueError, IndexError):
                continue
    return sorted(ticks)


def prune_checkpoints(keep_last: int = KEEP_LAST) -> int:
    """Remove all checkpoints except the ``keep_last`` most recent ones.
    Returns the number of deleted files."""
    ticks = list_checkpoints()
    if len(ticks) <= keep_last:
        return 0

    to_delete = ticks[:-keep_last]
    count = 0
    for t in to_delete:
        path = CHECKPOINT_DIR / _tick_filename(t)
        try:
            path.unlink()
            count += 1
        except OSError as e:
            logger.warning("[Checkpoint] failed to delete %s: %s", path, e)

    if count:
        logger.info("[Checkpoint] pruned %d old checkpoints, kept last %d", count, keep_last)
    return count


def latest_tick() -> Optional[int]:
    """Return the highest available tick number, or None."""
    ticks = list_checkpoints()
    return ticks[-1] if ticks else None
