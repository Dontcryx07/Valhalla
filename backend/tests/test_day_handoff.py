"""Regression coverage for the in-place calendar-day handoff."""

from __future__ import annotations

import asyncio
import unittest
from unittest.mock import AsyncMock, patch

from src.agents.Actions import ActionState, ActionType, AgentActionManager, LocationResolver
from src.core.agent_registry import AgentRegistry, AgentRuntimeState
from src.core.world_engine import WorldEngine
from src.core.world_state import Position


def _plan(location_id: str) -> list[dict]:
    return [{
        "action": "Sleep",
        "start": "00:00",
        "end": "07:00",
        "location_id": location_id,
        "energy_change": 0.4,
        "emotion_change": 0.1,
    }]


class DayHandoffTests(unittest.IsolatedAsyncioTestCase):
    async def test_tick_snapshot_matches_advanced_world_clock(self) -> None:
        """A rendered frame and the checkpoint it represents share one tick."""
        engine = WorldEngine(sim_start_date="2026-07-03")
        with patch("src.core.world_engine.save_checkpoint"), \
             patch("src.core.world_engine.prune_checkpoints"):
            snapshot = await engine.run_tick()

        self.assertEqual(engine.world.tick, 1)
        self.assertEqual(snapshot["tick"], engine.world.tick)
        self.assertEqual(snapshot["time"], "00:01")

    async def test_handoff_preserves_agent_runtime_and_installs_next_day_plan(self) -> None:
        engine = WorldEngine(sim_start_date="2026-07-03")
        engine.registry = AgentRegistry()
        engine.world.tick = 1440
        position = Position(x=12, y=34, location_id="library")
        engine.world.register_agent("alice", position)
        manager = AgentActionManager("alice", _plan("library"), position, LocationResolver())
        manager.current_action = ActionState(
            action_type=ActionType.MISC,
            description="Late study session",
            start_time="22:00",
            end_time="24:00",
            location_id="library",
            position=position,
        )
        state = AgentRuntimeState(
            agent_id="alice",
            persona={"Name": "Alice"},
            persona_name="Alice",
            manager=manager,
            position=position,
            day_plan=_plan("library"),
            energy_level=0.32,
            emotion_state=0.71,
            conversation_count=4,
            replan_count=2,
        )
        engine.registry.register(state)

        archive = AsyncMock(return_value={"summary": "done"})
        with patch("src.core.world_engine.archive_to_long_term", archive), \
             patch("src.core.world_engine.clear_short_term_data", return_value=True) as clear, \
             patch("src.core.world_engine.save_history"), \
             patch("src.core.world_engine.save_checkpoint"), \
             patch("src.core.world_engine.prune_checkpoints"), \
             patch("src.agents.day_planner.run", return_value={"day_plan": _plan("library")}):
            snapshot = await engine.handoff_to_next_day("2026-07-04", 0.01)

        self.assertEqual(engine.sim_start_date, "2026-07-04")
        self.assertEqual(engine.world.tick, 1440)
        self.assertIs(engine.registry.get("alice").manager, manager)
        self.assertEqual(state.position, position)
        self.assertEqual(state.energy_level, 0.32)
        self.assertEqual(state.emotion_state, 0.71)
        self.assertEqual(state.conversation_count, 0)
        self.assertEqual(state.replan_count, 0)
        self.assertEqual(manager.current_action.description, "Sleep")
        self.assertEqual(snapshot["type"], "day_initialized")
        self.assertEqual(snapshot["tick"], engine.world.tick)
        self.assertEqual(snapshot["time"], "00:00")
        self.assertEqual(snapshot["day"], 2)
        self.assertIn("speed", snapshot)
        self.assertIn("health", snapshot)
        archive.assert_awaited_once_with("Alice", "2026-07-03")
        clear.assert_called_once_with("Alice", "2026-07-03")

    async def test_handoff_cancels_stalled_conversation_after_deadline(self) -> None:
        engine = WorldEngine()
        position = Position(x=1, y=1, location_id="library")
        for agent_id in ("alice", "bob"):
            manager = AgentActionManager(agent_id, [], position, LocationResolver())
            state = AgentRuntimeState(
                agent_id=agent_id,
                persona={"Name": agent_id.title()},
                persona_name=agent_id.title(),
                manager=manager,
                position=position,
                paused=True,
                conversation_start_tick=1439,
            )
            engine.registry.register(state)

        async def never_finishes():
            await asyncio.sleep(60)

        request = {"agent_a_id": "alice", "agent_b_id": "bob"}
        task = asyncio.create_task(never_finishes())
        engine._conversation_tasks["alice|bob"] = task
        engine._pending_conversations["alice|bob"] = request

        cancelled = await engine._drain_conversations_for_handoff(0.01)

        self.assertEqual(cancelled, 1)
        self.assertFalse(engine.registry.get("alice").paused)
        self.assertFalse(engine.registry.get("bob").paused)
        self.assertFalse(engine._pending_conversations)

    async def test_handoff_discards_prior_day_decision_before_next_tick(self) -> None:
        """A late old-day decision must not request a replan at 00:00."""
        engine = WorldEngine(sim_start_date="2026-07-03")
        engine.world.tick = 1440
        position = Position(x=12, y=34, location_id="library")
        engine.world.register_agent("alice", position)
        manager = AgentActionManager("alice", _plan("library"), position, LocationResolver())
        manager.current_action = ActionState(
            action_type=ActionType.MISC,
            description="Late study session",
            start_time="22:00",
            end_time="24:00",
            location_id="library",
            position=position,
        )
        state = AgentRuntimeState(
            agent_id="alice",
            persona={"Name": "Alice"},
            persona_name="Alice",
            manager=manager,
            position=position,
            day_plan=_plan("library"),
        )
        engine.registry.register(state)

        async def stale_decision():
            await asyncio.sleep(60)
            return "replan"

        engine._decision_tasks["alice"] = asyncio.create_task(stale_decision())
        replan = AsyncMock()
        with patch("src.core.world_engine.archive_to_long_term", AsyncMock(return_value={"summary": "done"})), \
             patch("src.core.world_engine.clear_short_term_data", return_value=True), \
             patch("src.core.world_engine.save_history"), \
             patch("src.core.world_engine.save_checkpoint"), \
             patch("src.core.world_engine.prune_checkpoints"), \
             patch("src.agents.day_planner.run", return_value={"day_plan": _plan("library")}), \
             patch.object(engine, "_phase_replan", replan):
            await engine.handoff_to_next_day("2026-07-04", 0.01)
            await engine.run_tick()

        self.assertFalse(engine._decision_tasks)
        replan.assert_not_awaited()

    async def test_handoff_continues_when_optional_persistence_fails(self) -> None:
        """A replay/event/checkpoint failure must not freeze the next day."""
        engine = WorldEngine(sim_start_date="2026-07-03")
        engine.registry = AgentRegistry()
        engine.world.tick = 1440
        position = Position(x=12, y=34, location_id="library")
        engine.world.register_agent("alice", position)
        manager = AgentActionManager("alice", _plan("library"), position, LocationResolver())
        engine.registry.register(AgentRuntimeState(
            agent_id="alice", persona={"Name": "Alice"}, persona_name="Alice",
            manager=manager, position=position, day_plan=_plan("library"),
        ))

        with patch("src.core.world_engine.archive_to_long_term", AsyncMock(return_value={"summary": "done"})), \
             patch("src.core.world_engine.clear_short_term_data", return_value=True), \
             patch("src.core.world_engine.save_history", side_effect=OSError("history unavailable")), \
             patch.object(engine, "_apply_event_opportunities", side_effect=OSError("events unavailable")), \
             patch("src.core.world_engine.save_checkpoint", side_effect=OSError("checkpoint unavailable")), \
             patch("src.agents.day_planner.run", return_value={"day_plan": _plan("library")}):
            snapshot = await engine.handoff_to_next_day("2026-07-04", 0.01)

        self.assertEqual(engine.sim_start_date, "2026-07-04")
        self.assertEqual(engine.registry.get("alice").manager.current_action.description, "Sleep")
        self.assertEqual(len(snapshot["warnings"]), 3)


if __name__ == "__main__":
    unittest.main()
