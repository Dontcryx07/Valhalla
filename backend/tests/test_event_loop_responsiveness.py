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
    def test_resource_exhaustion_without_retry_hint_cools_for_a_full_rpm_window(self):
        duration = gemini_client._parse_retry_after(
            "429 RESOURCE_EXHAUSTED: Resource has been exhausted (e.g. check quota)."
        )

        self.assertGreaterEqual(duration, 60.0)

    def test_fast_fail_mode_still_attempts_the_selected_model(self):
        expected = TickDecision(decision="continue", reason="test")
        with patch.object(gemini_client, "API_KEYS", ["test-key"]), \
                patch.object(gemini_client, "_reserve_key_with_brief_wait", return_value="test-key"), \
                patch.object(gemini_client, "_mark_used"), \
                patch.object(gemini_client, "_single_attempt", return_value=expected) as attempt:
            result = gemini_client.call_gemini(
                "system", "user", TickDecision, complexity="default"
            )

        self.assertIs(result, expected)
        self.assertEqual(attempt.call_count, 1)

    def test_transport_timeout_rotates_to_another_key_before_fallback_model(self):
        expected = TickDecision(decision="continue", reason="recovered")
        model_tiers = {"default": ["primary", "fallback"]}
        with patch.object(gemini_client, "API_KEYS", ["key-one", "key-two"]), \
                patch.object(gemini_client, "MODEL_TIERS", model_tiers), \
                patch.object(gemini_client, "_reserve_key_with_brief_wait", side_effect=["key-one", "key-two"]), \
                patch.object(gemini_client, "_mark_cooldown") as cooldown, \
                patch.object(gemini_client, "_mark_used"), \
                patch.object(gemini_client, "_single_attempt", side_effect=[TimeoutError("stalled"), expected]) as attempt:
            result = gemini_client.call_gemini(
                "system", "user", TickDecision, complexity="default"
            )

        self.assertIs(result, expected)
        self.assertEqual(
            [call.args[:2] for call in attempt.call_args_list],
            [("key-one", "primary"), ("key-two", "primary")],
        )
        cooldown.assert_called_once_with(
            "key-one", duration=gemini_client.TRANSIENT_KEY_COOLDOWN_SECONDS
        )


if __name__ == "__main__":
    unittest.main()
