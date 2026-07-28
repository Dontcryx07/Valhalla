"""Bounded runtime health checks for a live Valhalla simulation."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

from src.agents.Actions import ActionType
from src.core.log import get_logger

logger = get_logger(__name__)


@dataclass
class _TravelProgress:
    position: Tuple[int, int]
    stagnant_ticks: int = 0


@dataclass
class RuntimeHealthMonitor:
    """Checks the invariants most likely to make a campus sim look frozen.

    State is O(number of agents), reports are emitted at a bounded cadence, and
    no complete tick history is retained.  The latest report can be safely sent
    in the frontend snapshot for a toggled developer panel.
    """

    report_every_ticks: int = 15
    stagnant_move_ticks: int = 5
    _travel: Dict[str, _TravelProgress] = field(default_factory=dict)
    _last_report: Dict[str, Any] = field(default_factory=dict)

    def observe(self, engine: Any, tick: int, hhmm: str) -> Dict[str, Any]:
        anomalies: List[Dict[str, str]] = []
        moving = paused = conversations = 0
        agent_ids = set(engine.registry.all_ids())
        for state in engine.registry.all_states():
            manager = state.manager
            action = manager.current_action if manager else None
            if state.paused:
                paused += 1
            if state.active_conversation:
                conversations += 1
                conversation = state.active_conversation
                if conversation.get("status") == "generating" and tick - state.conversation_start_tick > 30:
                    anomalies.append({"agent_id": state.agent_id, "kind": "conversation_generation_timeout"})
            if not action:
                self._travel.pop(state.agent_id, None)
                continue
            if manager and manager.position != state.position:
                anomalies.append({"agent_id": state.agent_id, "kind": "manager_registry_position_desync"})
            # ``run_tick`` publishes its frame after advancing the world clock.
            # A normal non-move action therefore must have been replaced by
            # then.  This catches stale BodyController/manager references after
            # a rewind before the UI can silently show an agent sleeping past
            # their scheduled wake-up time.
            try:
                start_minute = manager._hhmm_to_minutes(action.start_time)
                end_minute = manager._hhmm_to_minutes(action.end_time)
                day_start_tick = tick - (tick % (24 * 60))
                end_tick = day_start_tick + end_minute
                if end_minute <= start_minute:
                    end_tick += 24 * 60
                terminal_frame_allowance = 1 if action.action_type == ActionType.MOVE else 0
                if tick > end_tick + terminal_frame_allowance:
                    anomalies.append({"agent_id": state.agent_id, "kind": "action_overdue"})
            except (AttributeError, TypeError, ValueError):
                anomalies.append({"agent_id": state.agent_id, "kind": "invalid_action_time"})
            if action.action_type == ActionType.MOVE:
                moving += 1
                current = (state.position.x, state.position.y)
                previous = self._travel.get(state.agent_id)
                stagnant = previous.stagnant_ticks + 1 if previous and previous.position == current else 0
                self._travel[state.agent_id] = _TravelProgress(current, stagnant)
                if stagnant >= self.stagnant_move_ticks:
                    anomalies.append({"agent_id": state.agent_id, "kind": "travel_stalled"})
            else:
                self._travel.pop(state.agent_id, None)

            world_agent = engine.world.agents.get(state.agent_id)
            if world_agent and (world_agent.position.x, world_agent.position.y, world_agent.position.location_id) != (
                state.position.x, state.position.y, state.position.location_id,
            ):
                anomalies.append({"agent_id": state.agent_id, "kind": "registry_world_desync"})

        for stale_id in list(self._travel):
            if stale_id not in agent_ids:
                self._travel.pop(stale_id, None)
        try:
            event_loop_tasks = len(asyncio.all_tasks())
        except RuntimeError:
            # Unit tests and offline diagnostics can invoke health checks
            # without an active asyncio loop.
            event_loop_tasks = 0
        report = {
            "tick": tick, "time": hhmm, "agents": len(agent_ids), "moving": moving,
            "paused": paused, "conversations": conversations,
            "background_tasks": len(engine._conversation_tasks) + len(engine._decision_tasks),
            "background_decisions": len(engine._decision_tasks),
            "event_loop_tasks": event_loop_tasks,
            "anomalies": anomalies,
            "healthy": not anomalies,
        }
        self._last_report = report
        if anomalies or tick % self.report_every_ticks == 0:
            level = logger.warning if anomalies else logger.info
            level("[Health] tick=%d agents=%d moving=%d paused=%d conv=%d tasks=%d anomalies=%s",
                  tick, report["agents"], moving, paused, conversations,
                  report["event_loop_tasks"], anomalies or "none")
        return report

    @property
    def latest(self) -> Dict[str, Any]:
        return dict(self._last_report)
