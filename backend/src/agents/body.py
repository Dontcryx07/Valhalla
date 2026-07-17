"""
Body -- the agent's "limbs". The motor layer the brain commands.

In the human-like architecture the *brain* (brain.py) does the thinking:
it perceives, recalls, and decides. It never moves the agent directly.
Instead it issues motor commands to this Body, which is the only thing that
actually changes the agent's position and current activity.

The Body is a thin, behaviour-preserving adapter around the existing action
state machine (`AgentActionManager` in Actions.py) -- the proven executor
that walks paths and steps through the day plan. Wrapping it (rather than
replacing it) means the body/brain split is a clean architectural layer with
zero change to how movement and actions actually run.

Motor command surface (all 0-LLM):
  - advance(tick)              : take the next step of the current plan
  - enter_conversation(name)   : freeze into a conversation with someone
  - resume(day_plan)           : leave conversation / reload the plan
Read-only senses of the body's own state:
  - position, current_action, is_last_action
"""

from __future__ import annotations

from typing import Any, List

from src.core.log import get_logger

logger = get_logger(__name__)


class BodyController:
    """Wraps one agent's `AgentActionManager` as a commandable motor tool."""

    def __init__(self, manager: Any) -> None:
        self._manager = manager

    # -- read-only body state ---------------------------------------------
    @property
    def manager(self) -> Any:
        return self._manager

    @property
    def position(self):
        return self._manager.position if self._manager else None

    @property
    def current_action(self):
        return self._manager.current_action if self._manager else None

    @property
    def is_last_action(self) -> bool:
        return bool(self._manager and self._manager.is_last_action)

    # -- motor commands ----------------------------------------------------
    def advance(self, tick: int):
        """Advance the manager state machine by one tick."""
        if self._manager is None:
            return None
        return self._manager.tick(tick)

    def enter_conversation(self, partner_name: str):
        """Freeze the body into a conversation with `partner_name`."""
        if self._manager is None:
            return None
        return self._manager.set_conversation_action(partner_name)

    def resume(self, day_plan: List[dict]) -> None:
        """Leave conversation mode and (re)load a day plan to execute."""
        if self._manager is not None:
            self._manager.resume_from_conversation(day_plan)
