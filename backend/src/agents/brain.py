"""
Brain -- the agent's cognition / command centre.

Each tick the brain may be called to decide (via LLM) whether the agent
should continue their current plan or replan, based on novel observations.
The LLM call is gated: it only fires when the perceive phase detects a change
in the set of (agent_id, action_description) within 50px.

When no novel observations exist, the brain returns "continue" without an LLM
call — the agent follows its existing plan.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from pydantic import BaseModel

from src.core.log import get_logger
from src.config import TEMPERATURE
from src.llm.gemini_client import call_gemini

logger = get_logger(__name__)


class TickDecision(BaseModel):
    decision: str  # "continue" or "replan"
    reason: str


DECISION_SYSTEM_PROMPT = """You are the decision-making layer for a simulated college student at IIT Ropar.
You are given the persona's current situation and a list of nearby agents they just noticed.
Decide whether they should continue their current activity or replan their day.

Rules:
- "continue" — the agent stays on their current plan. This is the default for routine observations.
- "replan" — the agent should regenerate its remaining-day plan. Use this only for genuinely
  significant events: a close friend appears, an urgent opportunity arises, or the current
  situation conflicts with their goals.

Be conservative. Most observations (someone walking past, someone studying nearby) do NOT
warrant a replan. Only replan when this specific persona would clearly change their plans.

Respond ONLY with the requested JSON schema — no extra commentary."""


class Brain:
    """Per-agent decision-maker that commands a BodyController."""

    def __init__(self, agent_id: str, persona_name: str, body: Any,
                 persona: Optional[dict] = None) -> None:
        self.agent_id = agent_id
        self.persona_name = persona_name
        self.persona = persona or {}
        self.body = body

    def decide_tick(self, tick: int, hhmm: str, observations: List,
                    day_plan: List, replan_count: int = 0,
                    max_replans: int = 3,
                    energy_level: float = 0.5, emotion_state: float = 0.5,
                    relevant_memories: Optional[List[str]] = None) -> TickDecision:
        """LLM-based decision: continue or replan?

        Called only when the perceive phase detected novel observations.
        If replan_count >= max_replans, returns "continue" without an LLM call.
        """
        if replan_count >= max_replans:
            logger.info(
                "[Brain] '%s' replan count exhausted (%d/%d) — continuing plan",
                self.persona_name, replan_count, max_replans,
            )
            return TickDecision(decision="continue", reason="replan budget exhausted")

        current_action_desc = "nothing"
        if self.body.current_action:
            current_action_desc = (
                f"{self.body.current_action.description} at {self.body.current_action.location_id or '?'}"
            )

        plan_summary = self._plan_summary(day_plan, hhmm)
        obs_text = self._observations_text(observations)
        memory_text = "\n".join(f"- {memory}" for memory in (relevant_memories or [])[:4]) or "(nothing relevant recalled)"

        user_prompt = (
            f"You are {self.persona_name} at IIT Ropar.\n"
            f"Personality: {self.persona.get('innate', '?')}\n"
            f"Hobbies: {self.persona.get('hobbies', '?')}\n"
            f"Goals: {self.persona.get('goals', '?')}\n"
            f"Current activity: {current_action_desc}\n"
            f"Time: {hhmm}\n\n"
            f"Remaining plan:\n{plan_summary}\n\n"
            f"Nearby people:\n{obs_text}\n\n"
            f"Relevant long-term memories:\n{memory_text}\n\n"
            f"Energy level: {energy_level:.2f}/1.0\n"
            f"Emotion state: {emotion_state:.2f}/1.0\n"
            'Return {"decision": "continue"} to stay on the current plan, or '
            '{"decision": "replan", "reason": "..."} to regenerate the remaining day plan. '
            "Only replan if this genuinely warrants a change of plans."
        )

        try:
            result: TickDecision = call_gemini(
                system_prompt=DECISION_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                schema=TickDecision,
                complexity="default",
                temperature=TEMPERATURE,
            )
            logger.info(
                "[Brain] '%s' decide_tick: decision=%s (%s)",
                self.persona_name, result.decision, result.reason,
            )
            return result
        except Exception as e:
            logger.warning(
                "[Brain] '%s' decide_tick LLM failed: %s — continuing plan",
                self.persona_name, e,
            )
            return TickDecision(decision="continue", reason="LLM call failed, defaulting to continue")

    def act(self, tick: int):
        """Phase 2 (Act): advance the body's state machine."""
        return self.body.advance(tick)

    # -- helpers -----------------------------------------------------------

    @staticmethod
    def _plan_summary(plan: List[Dict], current_hhmm: str) -> str:
        if not plan:
            return "  (no plan remaining)"
        lines = []
        for entry in sorted(plan, key=lambda e: e.get("start", "00:00")):
            if entry.get("start", "00:00") >= current_hhmm:
                marker = " ⇐ NOW" if entry.get("start") == current_hhmm else ""
                lines.append(
                    f"  {entry.get('start', '??')}-{entry.get('end', '??')}  "
                    f"{entry.get('action', '?')} at {entry.get('location_id', '?')}{marker}"
                )
        return "\n".join(lines) if lines else "  (no plan remaining)"

    @staticmethod
    def _observations_text(observations: List) -> str:
        if not observations:
            return "  (no one nearby)"
        lines = []
        for obs in observations:
            if hasattr(obs, "model_dump"):
                obs = obs.model_dump()
            if not isinstance(obs, dict):
                lines.append(f"  - {obs}")
                continue
            if "description" in obs and "current_action" not in obs:
                location = obs.get("location_id", "?")
                lines.append(f"  - {obs['description']} at {location}")
                continue
            name = obs.get("name", obs.get("agent_id", "?"))
            action = obs.get("current_action", "?")
            pos = obs.get("position", {})
            lines.append(f"  - {name} at ({pos.get('x', '?')}, {pos.get('y', '?')}) — {action}")
        return "\n".join(lines)


if __name__ == "__main__":
    from src.core.log import setup_logging
    setup_logging(run_id="brain_test", console=True)

    class _FakeManager:
        def __init__(self):
            self.position = "P"
            self.current_action = None
            self.is_last_action = False
            self.ticks = []
        def tick(self, t):
            self.ticks.append(t)
            return f"action@{t}"

    from src.agents.body import BodyController
    fm = _FakeManager()
    brain = Brain("a", "Agent A", BodyController(fm))
    out = brain.act(5)
    assert out == "action@5", out
    assert fm.ticks == [5]
    print("brain.py sanity check passed (0 LLM, pure body.advance).")
