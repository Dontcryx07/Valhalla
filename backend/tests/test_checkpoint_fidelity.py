"""Regression tests for behavioural checkpoint fidelity and relationship writes."""

from __future__ import annotations

import gzip
import asyncio
import json
import random
import socket
import tempfile
import unittest
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import patch

from src.agents.Actions import ActionState, ActionType, AgentActionManager, LocationResolver
from src.agents.conversation import RelationshipMatrix
from src.core.agent_registry import AgentRegistry, AgentRuntimeState
from src.core.checkpoint_manager import KEEP_LAST, list_checkpoints, load_checkpoint, prune_checkpoints, save_checkpoint
from src.core.world_engine import WorldEngine
from src.core.world_state import CurrentAction, Position, WorldState


class CheckpointFidelityTests(unittest.TestCase):
    def test_default_checkpoint_retention_keeps_one_full_day(self) -> None:
        with tempfile.TemporaryDirectory() as directory, patch(
            "src.core.checkpoint_manager.CHECKPOINT_DIR", Path(directory),
        ):
            checkpoint_dir = Path(directory)
            for tick in range(1, KEEP_LAST + 2):
                (checkpoint_dir / f"tick_{tick:05d}.json").write_text("{}", encoding="utf-8")

            self.assertEqual(prune_checkpoints(), 1)
            self.assertEqual(len(list_checkpoints()), KEEP_LAST)
            self.assertEqual(list_checkpoints()[0], 2)

    def test_round_trip_preserves_runtime_and_rng_state(self) -> None:
        resolver = LocationResolver()
        position = Position(x=42, y=24, location_id="library")
        manager = AgentActionManager("alice", [], position, resolver)
        manager.current_action = ActionState(
            action_type=ActionType.MOVE,
            description="Walk to library",
            start_time="10:30",
            end_time="10:45",
            location_id="library",
            position=position,
            path=[(40, 24), (41, 24), (42, 24)],
            path_index=1,
            energy_change=-0.2,
            emotion_change=0.1,
        )
        manager._conversation_mode = True
        manager._pending_plan_action = manager.current_action.model_copy()
        manager._entered_last_action = True

        state = AgentRuntimeState(
            agent_id="alice",
            persona={"Name": "Alice"},
            persona_name="Alice",
            manager=manager,
            position=position,
            paused=True,
            day_plan=[{"action": "Walk to library", "start": "10:30", "end": "10:45"}],
            day_archived=True,
            conversation_start_tick=617,
            conversation_count=2,
            replan_count=2,
            last_conversation_partner="bob",
            active_conversation={"partner_id": "bob", "status": "generating"},
            energy_level=0.32,
            emotion_state=0.81,
            color="#123456",
        )
        registry = AgentRegistry()
        registry.register(state)
        world = WorldState(tick=617, agent_last_conversation={"alice": 601})
        world.register_agent("alice", position)
        world.set_agent_action(
            "alice",
            CurrentAction(description="Walk to library", start_tick=600, end_tick=630),
        )

        original_rng_state = random.getstate()
        try:
            random.seed(8675309)
            with tempfile.TemporaryDirectory() as directory, patch(
                "src.core.checkpoint_manager.CHECKPOINT_DIR", Path(directory),
            ):
                checkpoint_path = save_checkpoint(
                    world,
                    registry,
                    617,
                    engine_state={
                        "day_index": 4,
                        "in_range": {"alice": ["bob"]},
                        "last_observations": {"alice": [["bob", "reading"]]},
                        "pending_conversations": [{"agent_a_id": "alice", "agent_b_id": "bob"}],
                    },
                )
                self.assertEqual(checkpoint_path.suffixes[-2:], [".json", ".gz"])
                self.assertEqual(checkpoint_path.read_bytes()[:2], b"\x1f\x8b")
                with gzip.open(checkpoint_path, "rt", encoding="utf-8") as handle:
                    legacy_payload = json.load(handle)
                plain_json = json.dumps(legacy_payload, separators=(",", ":")).encode("utf-8")
                self.assertLess(checkpoint_path.stat().st_size, len(plain_json))
                legacy_path = Path(directory) / "tick_00618.json"
                legacy_path.write_text(json.dumps(legacy_payload), encoding="utf-8")
                legacy_world, legacy_registry = load_checkpoint(618, resolver)
                self.assertEqual(legacy_world.tick, world.tick)
                self.assertEqual(legacy_registry.get("alice").persona_name, "Alice")
                expected_next_random = random.random()
                random.random()
                restored_world, restored_registry, metadata = load_checkpoint(
                    617, resolver, return_metadata=True,
                )

            restored = restored_registry.get("alice")
            self.assertEqual(restored_world.model_dump(), world.model_dump())
            self.assertEqual(restored.energy_level, 0.32)
            self.assertEqual(restored.emotion_state, 0.81)
            self.assertEqual(restored.replan_count, 2)
            self.assertEqual(restored.last_conversation_partner, "bob")
            self.assertEqual(restored.conversation_start_tick, 617)
            self.assertEqual(restored.active_conversation, state.active_conversation)
            self.assertEqual(restored.manager.current_action.model_dump(), manager.current_action.model_dump())
            self.assertEqual(restored.manager._pending_plan_action.model_dump(), manager._pending_plan_action.model_dump())
            self.assertEqual(restored_world.agent_last_conversation, {"alice": 601})
            self.assertEqual(metadata["day_index"], 4)
            self.assertEqual(random.random(), expected_next_random)
        finally:
            random.setstate(original_rng_state)


class RelationshipMatrixWriteTests(unittest.TestCase):
    def test_independent_instances_merge_their_deltas(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "relationships.json"
            first = RelationshipMatrix(path)
            second = RelationshipMatrix(path)

            first.update("alice", "bob", 0.2)
            second.update("charlie", "dana", -0.1)
            first.save()
            second.save()

            restored = RelationshipMatrix(path)
            self.assertEqual(restored.get("alice", "bob"), 0.7)
            self.assertEqual(restored.get("charlie", "dana"), 0.4)


class PendingConversationRecoveryTests(unittest.IsolatedAsyncioTestCase):
    async def test_restore_restarts_saved_conversation_request(self) -> None:
        engine = WorldEngine()
        position = Position(x=1, y=1, location_id="library")
        for agent_id in ("alice", "bob"):
            engine.registry.register(AgentRuntimeState(
                agent_id=agent_id,
                persona={"Name": agent_id.title()},
                persona_name=agent_id.title(),
                position=position,
                paused=True,
            ))

        request = {"agent_a_id": "alice", "agent_b_id": "bob", "location_id": "library"}
        engine.restore_checkpoint_state({"pending_conversations": [request]})
        with patch.object(engine, "_start_conversation_task") as start:
            await engine.resume_pending_conversations()

        start.assert_called_once_with(
            engine.registry.get("alice"), engine.registry.get("bob"), request,
        )


class ServerTimelineControlTests(unittest.IsolatedAsyncioTestCase):
    async def test_shutdown_cancels_the_owned_simulation_task(self) -> None:
        import Odin as server

        async def waits_forever():
            await asyncio.sleep(60)

        previous = server._sim_task
        task = asyncio.create_task(waits_forever())
        server._sim_task = task
        try:
            await server._stop_sim_task()
            self.assertTrue(task.cancelled())
            self.assertIsNone(server._sim_task)
        finally:
            server._sim_task = previous

    async def test_ui_stop_preserves_snapshot_and_reports_stopped(self) -> None:
        import Odin as server

        async def waits_forever():
            await asyncio.sleep(60)

        previous = (server._sim_task, server._latest_snapshot)
        task = asyncio.create_task(waits_forever())
        server._sim_task = task
        server._latest_snapshot = {"tick": 17, "agents": {"alice": {}}}
        try:
            result = await server.stop_sim()
            self.assertEqual(result, {"status": "stopped", "running": False})
            self.assertTrue(task.cancelled())
            self.assertEqual(server._latest_snapshot["tick"], 17)
            self.assertFalse(server._latest_snapshot["simulation"]["running"])
        finally:
            server._sim_task, server._latest_snapshot = previous

    async def test_ui_start_always_uses_the_latest_checkpoint(self) -> None:
        import Odin as server

        calls = []

        async def starts_from_checkpoint(*, resume_checkpoint=False):
            calls.append(resume_checkpoint)
            await asyncio.sleep(60)

        previous = (server._sim_task, server._sim_engine)
        server._sim_task = None
        server._sim_engine = object()
        try:
            with patch("Odin._run_sim", side_effect=starts_from_checkpoint), \
                 patch("src.core.checkpoint_manager.list_checkpoints", return_value=[40, 55]):
                result = await server.start_sim()
                await asyncio.sleep(0)
            self.assertEqual(result, {"status": "starting_from_checkpoint", "running": True, "checkpoint_tick": 55})
            self.assertIsNotNone(server._sim_task)
            self.assertEqual(calls, [True])
            await server._stop_sim_task()
        finally:
            server._sim_task, server._sim_engine = previous

    async def test_port_reservation_fails_before_lifespan_can_start(self) -> None:
        import Odin as server

        blocker = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        blocker.bind(("127.0.0.1", 0))
        blocker.listen(socket.SOMAXCONN)
        try:
            port = blocker.getsockname()[1]
            with self.assertRaises(OSError):
                server._reserve_listen_socket("127.0.0.1", port)
        finally:
            blocker.close()

    async def test_engine_applies_fast_forward_without_skipping_ticks(self) -> None:
        from src import config

        engine = WorldEngine()
        engine.world.tick = 0
        original_speed = config.TICK_SPEED
        sleeps = []

        async def fake_tick():
            engine.world.tick += 1
            return {"tick": engine.world.tick}

        async def on_tick(snapshot):
            if snapshot.get("tick") == 1:
                config.TICK_SPEED = 4.0

        async def fake_sleep(delay):
            sleeps.append(delay)

        engine.run_tick = fake_tick
        try:
            with patch("src.core.world_engine.asyncio.sleep", new=fake_sleep):
                await engine.run(max_tick=2, on_tick=on_tick)
            self.assertEqual(engine.world.tick, 2)
            self.assertEqual(sleeps, [config.REAL_SECONDS_PER_TICK / 4.0] * 2)
        finally:
            config.TICK_SPEED = original_speed

    async def test_fast_forward_doubles_speed_and_stops_at_safe_cap(self) -> None:
        import Odin as server
        from src import config

        original_speed = config.TICK_SPEED
        try:
            config.TICK_SPEED = 1.0
            self.assertEqual((await server.fast_forward_sim())["speed_multiplier"], 2.0)
            config.TICK_SPEED = 32.0
            self.assertEqual((await server.fast_forward_sim())["speed_multiplier"], 32.0)
        finally:
            config.TICK_SPEED = original_speed

    async def test_slow_down_halves_speed_and_stops_at_safe_floor(self) -> None:
        import Odin as server
        from src import config

        original_speed = config.TICK_SPEED
        try:
            config.TICK_SPEED = 8.0
            self.assertEqual((await server.slow_down_sim())["speed_multiplier"], 4.0)
            config.TICK_SPEED = 0.25
            self.assertEqual((await server.slow_down_sim())["speed_multiplier"], 0.25)
        finally:
            config.TICK_SPEED = original_speed

    async def test_rewind_input_converts_hours_to_simulation_ticks(self) -> None:
        import Odin as server

        self.assertEqual(server.RewindInput(hours=2).requested_ticks(1), 120)
        self.assertEqual(server.RewindInput(hours=2).requested_ticks(5), 24)

    async def test_rewind_restores_a_retained_checkpoint_and_reports_distance(self) -> None:
        import Odin as server

        class _Broadcaster:
            async def broadcast(self, data):
                self.data = data

        class _Engine:
            def __init__(self):
                self.world = SimpleNamespace(tick=50)
                self.resolver = object()
                self.registry = object()
                self._conversation_tasks = {}
                self._decision_tasks = {}

            def assert_checkpoint_roster_matches_seed(self, registry):
                self.registry = registry

            def restore_checkpoint_state(self, state):
                self.state = state

            async def resume_pending_conversations(self):
                self.resumed = True

            def current_frontend_snapshot(self):
                return {"tick": self.world.tick, "time": "00:40", "agents": {}}

        engine = _Engine()
        previous = (server._sim_engine, server._sim_tick_lock, server._sim_broadcaster, server._latest_snapshot)
        server._sim_engine = engine
        server._sim_tick_lock = __import__("asyncio").Lock()
        server._sim_broadcaster = _Broadcaster()
        try:
            restored_world = SimpleNamespace(tick=40)
            with patch("src.core.checkpoint_manager.list_checkpoints", return_value=[35, 40, 45]), patch(
                "src.core.checkpoint_manager.load_checkpoint", return_value=(restored_world, "registry", {"day_index": 1}),
            ):
                result = await server.rewind_sim(server.RewindInput(ticks=10))
        finally:
            server._sim_engine, server._sim_tick_lock, server._sim_broadcaster, server._latest_snapshot = previous

        self.assertEqual(result["status"], "rewound")
        self.assertEqual(result["tick"], 40)
        self.assertEqual(result["rewound_ticks"], 10)
        self.assertTrue(engine.resumed)


if __name__ == "__main__":
    unittest.main()
