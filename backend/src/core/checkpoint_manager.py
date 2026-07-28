"""
Checkpoint Manager — per-tick state save/load for crash recovery.

Saves WorldState + AgentRegistry after every tick to compressed
``backend/data/checkpoints/tick_{00001}.json.gz`` files.

Supports:
  - Save: full simulation state as JSON
  - Load: reconstruct from any saved tick
  - List: available checkpoint ticks
  - Prune: auto-delete old checkpoints, keep last N
"""

from __future__ import annotations

import json
import gzip
import random
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
# One full simulated day of per-tick snapshots gives observers a complete
# rewind window without retaining prior days indefinitely.
KEEP_LAST = 1440
SCHEMA_VERSION = 2
CHECKPOINT_SUFFIX = ".json.gz"
LEGACY_CHECKPOINT_SUFFIX = ".json"


def _tick_filename(tick: int) -> str:
    return f"tick_{tick:05d}{CHECKPOINT_SUFFIX}"


def _legacy_tick_filename(tick: int) -> str:
    return f"tick_{tick:05d}{LEGACY_CHECKPOINT_SUFFIX}"


def _parse_tick_from_name(name: str) -> int:
    if name.endswith(CHECKPOINT_SUFFIX):
        stem = name[: -len(CHECKPOINT_SUFFIX)]
    elif name.endswith(LEGACY_CHECKPOINT_SUFFIX):
        stem = name[: -len(LEGACY_CHECKPOINT_SUFFIX)]
    else:
        raise ValueError(f"not a checkpoint filename: {name}")
    return int(stem.removeprefix("tick_"))


def _checkpoint_path(tick: int) -> Path:
    """Prefer compressed checkpoints while retaining read compatibility."""
    compressed = CHECKPOINT_DIR / _tick_filename(tick)
    if compressed.exists():
        return compressed
    return CHECKPOINT_DIR / _legacy_tick_filename(tick)


def _checkpoint_paths(tick: int) -> tuple[Path, Path]:
    return (
        CHECKPOINT_DIR / _tick_filename(tick),
        CHECKPOINT_DIR / _legacy_tick_filename(tick),
    )


# ------------------------------------------------------------------ #
# Save
# ------------------------------------------------------------------ #

def _manager_to_dict(manager: AgentActionManager) -> Dict[str, Any]:
    """Serialize an AgentActionManager's internal state."""
    def _action_dict(a: Optional[ActionState]) -> Optional[Dict[str, Any]]:
        if a is None:
            return None
        # ActionState contains more than the display fields.  In particular,
        # its resolved destination and energy/emotion deltas affect subsequent
        # movement and behaviour, so preserve the entire serializable model.
        return a.model_dump(mode="json")

    return {
        "day_plan": manager.day_plan,
        "last_action": _action_dict(manager.last_action),
        "current_action": _action_dict(manager.current_action),
        "next_action": _action_dict(manager.next_action),
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
        "conversation_start_tick": state.conversation_start_tick,
        "conversation_count": state.conversation_count,
        "replan_count": state.replan_count,
        "last_conversation_partner": state.last_conversation_partner,
        "active_conversation": state.active_conversation,
        "emotion_state": state.emotion_state,
        "emotion_baseline": state.emotion_baseline,
        "energy_level": state.energy_level,
        "color": state.color,
        "manager": _manager_to_dict(state.manager) if state.manager else None,
    }


def save_checkpoint(
    world: WorldState,
    registry: AgentRegistry,
    tick: int,
    engine_state: Optional[Dict[str, Any]] = None,
) -> Optional[Path]:
    """Save current simulation state as an atomically replaced gzip JSON file."""
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    path = CHECKPOINT_DIR / _tick_filename(tick)

    agents_data = [_agent_to_dict(s) for s in registry.all_states()]

    # History is not needed to advance an action, but it is part of the
    # observable world and is required for a faithful replay after restore.
    world_dict = json.loads(world.model_dump_json())

    payload = {
        "schema_version": SCHEMA_VERSION,
        "tick": tick,
        "world": world_dict,
        "agents": agents_data,
        "engine": engine_state or {},
        # The simulation uses Python's process-wide RNG for collision nudges
        # and interior placement.  JSON converts tuples to lists, which are
        # converted back on load before passing the state to random.setstate.
        "random_state": random.getstate(),
        "saved_at": _time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    tmp = path.with_suffix(path.suffix + ".tmp")
    with gzip.open(tmp, "wt", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, separators=(",", ":"), default=str))
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
    # model_validate also remains compatible with version-1 checkpoints,
    # whose omitted ActionState fields have model defaults.
    return ActionState.model_validate(d)


def _lists_to_tuples(value: Any) -> Any:
    """Restore the tuple structure required by random.setstate()."""
    if isinstance(value, list):
        return tuple(_lists_to_tuples(item) for item in value)
    return value


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
        manager._conversation_mode = m.get("_conversation_mode", False)
        manager._pending_plan_action = _action_from_dict(m.get("_pending_plan_action"))
        manager._entered_last_action = m.get("_entered_last_action", False)

    return manager


def load_checkpoint(
    tick: int,
    resolver: LocationResolver,
    return_metadata: bool = False,
) -> Tuple[WorldState, AgentRegistry] | Tuple[WorldState, AgentRegistry, Dict[str, Any]]:
    """Load simulation state from a checkpoint file."""
    path = _checkpoint_path(tick)
    if not path.exists():
        raise FileNotFoundError(f"No checkpoint for tick {tick} at {path}")

    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            raw = json.load(handle)
    else:
        raw = json.loads(path.read_text(encoding="utf-8"))

    if raw.get("random_state") is not None:
        random.setstate(_lists_to_tuples(raw["random_state"]))

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
            conversation_start_tick=agent_data.get("conversation_start_tick", 0),
            conversation_count=agent_data.get("conversation_count", 0),
            replan_count=agent_data.get("replan_count", 0),
            last_conversation_partner=agent_data.get("last_conversation_partner"),
            active_conversation=agent_data.get("active_conversation"),
            emotion_state=agent_data.get("emotion_state", 0.5),
            emotion_baseline=agent_data.get("emotion_baseline", 0.5),
            energy_level=agent_data.get("energy_level", 1.0),
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
    if return_metadata:
        return world, registry, raw.get("engine", {})
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
        if f.is_file() and f.name.startswith("tick_") and (
            f.name.endswith(CHECKPOINT_SUFFIX) or f.name.endswith(LEGACY_CHECKPOINT_SUFFIX)
        ):
            try:
                ticks.append(_parse_tick_from_name(f.name))
            except (ValueError, IndexError):
                continue
    return sorted(set(ticks))


def prune_checkpoints(keep_last: int = KEEP_LAST) -> int:
    """Remove all checkpoints except the ``keep_last`` most recent ones.
    Returns the number of deleted files."""
    ticks = list_checkpoints()
    if len(ticks) <= keep_last:
        return 0

    to_delete = ticks[:-keep_last]
    count = 0
    for t in to_delete:
        for path in _checkpoint_paths(t):
            if not path.exists():
                continue
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
