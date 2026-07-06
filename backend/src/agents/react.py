"""
React -- decides, given an agent's current action (if any) and what it just
perceived, whether to keep executing that action or interrupt and replan.

Design notes
------------
Most calls into this module are cheap and need no LLM round-trip:
  - no current action yet                       -> always replan
  - current action's end_tick has passed         -> always replan
  - mid-action, nothing new perceived this tick  -> always continue

The only case needing real judgement is: agent is mid-action AND perceived
something new (event_bus.py flagged them INTERRUPTED). That's the single
branch that calls Gemini, schema-constrained, to decide whether the
observation is worth interrupting the current plan for.

Fail-safe policy: if the Gemini call itself fails (quota exhausted, all
models down), this module does NOT propagate the exception up into the tick
graph -- an LLM outage forcing every mid-action agent to spuriously replan
(or crashing the whole tick) is worse than just letting them finish what
they were doing and logging the failure loudly. Contrast this with
gemini_client.py's own fail-fast policy on non-retryable errors -- that's
correct at the client layer (it's a bug signal); this is a different layer
making a product decision about degraded behavior.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel

from src.core.log import get_logger
from src.core.world_state import CurrentAction
from src.core.perceive import Observation
from src.llm.gemini_client import call_gemini

logger = get_logger(__name__)

REACT_SYSTEM_PROMPT = """You are the reactive-judgement layer for a simulated 
persona in a life-simulation sandbox. You are given the persona's currently 
committed action and a short list of things they just observed. Decide 
whether any observation is significant enough that this persona would 
plausibly interrupt what they're doing to respond, versus finishing their 
current action first. Be conservative: most day-to-day observations (someone 
walking past, someone eating lunch nearby) do NOT warrant an interruption. 
Interrupt only for things this specific persona would clearly care about 
given their personality, goals, and relationships. Respond ONLY with the 
requested JSON schema -- no extra commentary."""


class ReactionDecision(BaseModel):
    should_replan: bool
    reason: str


def decide_reaction(
    agent_id: str,
    persona: Dict[str, Any],
    current_action: Optional[CurrentAction],
    observations: List[Observation],
    memories: List[str],
    tick: int,
) -> ReactionDecision:
    """Entry point used by tick_graph.py's react node."""
    if current_action is None:
        return ReactionDecision(should_replan=True, reason="no current action to continue")

    if current_action.is_finished(tick):
        return ReactionDecision(should_replan=True, reason="current action has finished")

    if not observations:
        return ReactionDecision(should_replan=False, reason="mid-action, nothing new perceived")

    return _llm_reaction(agent_id, persona, current_action, observations, memories)


def _llm_reaction(
    agent_id: str,
    persona: Dict[str, Any],
    current_action: CurrentAction,
    observations: List[Observation],
    memories: List[str],
) -> ReactionDecision:
    observation_lines = "\n".join(f"- {o.description}" for o in observations)
    memory_lines = "\n".join(f"- {m}" for m in memories) if memories else "(none retrieved)"

    user_prompt = (
        f"Persona: {persona}\n\n"
        f"Currently doing: \"{current_action.description}\" "
        f"(committed until tick {current_action.end_tick})\n\n"
        f"Just observed:\n{observation_lines}\n\n"
        f"Relevant memories:\n{memory_lines}\n\n"
        "Should this persona interrupt their current action and replan?"
    )

    try:
        decision = call_gemini(
            system_prompt=REACT_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            schema=ReactionDecision,
            complexity="simple",
            temperature=0.3,
        )
        logger.info(
            "Agent '%s' reaction decision: should_replan=%s (%s)",
            agent_id, decision.should_replan, decision.reason,
        )
        return decision
    except Exception:
        logger.exception(
            "React LLM call failed for agent '%s'; defaulting to should_replan=False "
            "(agent will finish its current action this tick).",
            agent_id,
        )
        return ReactionDecision(
            should_replan=False,
            reason="LLM call failed, defaulting to continue current action",
        )


# Standalone sanity check (doesn't hit the network)
if __name__ == "__main__":
    action = CurrentAction(description="sleeping", start_tick=0, end_tick=60)

    r1 = decide_reaction("a", {}, None, [], [], tick=0)
    assert r1.should_replan is True

    r2 = decide_reaction("a", {}, action, [], [], tick=61)
    assert r2.should_replan is True

    r3 = decide_reaction("a", {}, action, [], [], tick=30)
    assert r3.should_replan is False

    print("react.py cheap-path sanity checks passed (LLM branch not exercised here).")