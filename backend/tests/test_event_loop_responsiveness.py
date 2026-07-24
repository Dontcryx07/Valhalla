"""Regression tests for synchronous Gemini work in the simulation loop."""

from __future__ import annotations

import asyncio
import sys
import time
import unittest
from pathlib import Path
from unittest.mock import patch

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.append(str(BACKEND_ROOT))

from src.agents.brain import TickDecision
from src.core.agent_registry import AgentRuntimeState
from src.core.world_engine import WorldEngine
from src.core.world_state import Position
from src.llm import gemini_client


class _SlowBrain:
    """A stand-in for a slow blocking Gemini response."""

    def decide_tick(self, **_kwargs):
        time.sleep(0.15)
        return TickDecision(decision="continue", reason="test")


class EventLoopResponsivenessTests(unittest.IsolatedAsyncioTestCase):
    async def test_decision_does_not_block_other_coroutines(self):
        engine = WorldEngine(sim_start_date="2099-01-01")
        state = AgentRuntimeState(
            agent_id="test",
            persona={"Name": "Test"},
            persona_name="Test",
            position=Position(x=0, y=0, location_id="test"),
            day_plan=[],
        )
        engine._tick_observations[state.agent_id] = []

        # The governor is imported inside the method, so configure the actual
        # process singleton instead of altering production code.
        from src.core.budget import GOVERNOR
        with patch.object(GOVERNOR, "can_afford", return_value=True), \
                patch.object(engine, "_brain_for", return_value=_SlowBrain()):
            task = asyncio.create_task(engine._phase_llm_decide(state, 0, "00:00"))
            await asyncio.sleep(0.02)
            self.assertFalse(task.done(), "blocking decision ran on the event loop")
            self.assertEqual(await task, "continue")


class GeminiAttemptTests(unittest.TestCase):
    def test_ring_always_starts_at_first_key_and_advances_after_any_error(self):
        ring = gemini_client.CircularKeyRing(["key-one", "key-two", "key-three"])
        self.assertEqual([node.index for node in ring.traverse_from_head()], [1, 2, 3])
        self.assertEqual([node.index for node in ring.traverse_from_head()], [1, 2, 3])

    def test_any_failure_moves_to_next_key_without_timeout_or_model_change(self):
        expected = TickDecision(decision="continue", reason="recovered")
        attempted = []

        class Models:
            def generate_content(self, **_kwargs):
                attempted.append(len(attempted) + 1)
                if len(attempted) == 1:
                    raise RuntimeError("any provider error")
                return type("Response", (), {"parsed": expected})()

        client = type("Client", (), {"models": Models()})()
        with patch.object(gemini_client, "API_KEYS", ["key-one", "key-two"]), \
                patch.object(gemini_client, "_get_client", return_value=client):
            result = gemini_client.call_gemini("system", "user", TickDecision)

        self.assertIs(result, expected)
        self.assertEqual(attempted, [1, 2])

    def test_full_circle_shows_quota_exhausted(self):
        class Models:
            def generate_content(self, **_kwargs):
                raise RuntimeError("unavailable")

        client = type("Client", (), {"models": Models()})()
        with patch.object(gemini_client, "API_KEYS", ["key-one", "key-two"]), \
                patch.object(gemini_client, "_get_client", return_value=client):
            with self.assertRaises(gemini_client.APIQuotaExhaustedError) as raised:
                gemini_client.call_gemini("system", "user", TickDecision)

        self.assertEqual(raised.exception.code, "api_quota_exhausted")


if __name__ == "__main__":
    unittest.main()
