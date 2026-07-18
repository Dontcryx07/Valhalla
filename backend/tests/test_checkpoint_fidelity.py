"""Regression tests for behavioural checkpoint fidelity and relationship writes."""

from __future__ import annotations

import random
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.agents.Actions import ActionState, ActionType, AgentActionManager, LocationResolver
from src.agents.conversation import RelationshipMatrix
from src.core.agent_registry import AgentRegistry, AgentRuntimeState
from src.core.checkpoint_manager import load_checkpoint, save_checkpoint
from src.core.world_engine import WorldEngine
from src.core.world_state import CurrentAction, Position, WorldState


class CheckpointFidelityTests(unittest.TestCase):
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
                save_checkpoint(
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


if __name__ == "__main__":
    unittest.main()
