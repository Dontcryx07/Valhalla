"""
React -- decides, given an agent's current action (if any) and what it just
perceived, whether to keep executing that action or interrupt and replan.

Design notes
------------
All calls into this module are cheap heuristic checks with no LLM round-trip:
  - no current action yet                       -> always replan
  - current action's end_tick has passed         -> always replan
  - mid-action, nothing new perceived this tick  -> always continue

The LLM-based decision layer has moved to brain.decide_tick(), which runs
only when the perceive phase detects novel observations.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel

from src.core.log import get_logger
from src.core.world_state import CurrentAction
from src.core.perceive import Observation

logger = get_logger(__name__)


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
    """Cheap heuristic decision: no LLM involved."""
    if current_action is None:
        return ReactionDecision(should_replan=True, reason="no current action to continue")

    if current_action.is_finished(tick):
        return ReactionDecision(should_replan=True, reason="current action has finished")

    if not observations:
        return ReactionDecision(should_replan=False, reason="mid-action, nothing new perceived")

    # Mid-action with observations: cheap heuristic — always continue.
    # The brain's decide_tick (LLM) will handle novel observation decisions.
    return ReactionDecision(should_replan=False, reason="mid-action, continuing current action")


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