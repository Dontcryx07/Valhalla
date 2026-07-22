"""
Agent Registry — single source of truth for every agent's runtime state.

The WorldEngine owns one `AgentRegistry` instance. All modules (Actions,
conversation, day_planner, Short_term) read from and write to it through
the engine — never directly.

This replaces the dual-source problem where Actions.py had its own
position/action and WorldState had a separate copy that drifted.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel

from src.core.world_state import Position


class AgentRuntimeState(BaseModel):
    """Everything the engine needs to know about one agent at runtime."""

    agent_id: str
    persona: Dict[str, Any]
    persona_name: str          # display name, e.g. "Parv Singla"
    manager: Optional[Any] = None  # AgentActionManager instance (set after init)
    position: Position
    current_action: Optional[Dict[str, Any]] = None  # serialised ActionState or None
    paused: bool = False       # True while in conversation or other blocking state
    day_plan: list = []        # current day plan actions
    day_archived: bool = False  # True when the current sim day has been archived
    conversation_start_tick: int = 0  # tick when the active conversation started (0 = not in one)
    conversation_count: int = 0  # how many conversations today
    replan_count: int = 0        # mid-day full replans today (budget cap)
    last_conversation_partner: Optional[str] = None  # agent_id of most recent chat partner
    active_conversation: Optional[dict] = None  # {"partner_name": str, "partner_id": str, "messages": list[dict]}
    emotion_state: float = 0.5   # [0, 1], starts neutral
    emotion_baseline: float = 0.5  # personality-derived normal mood; used for gentle recovery
    energy_level: float = 1.0    # [0, 1], starts full
    color: str = "#888888"  # frontend color (set by server on init)


class AgentRegistry:
    """Global runtime registry for all agents."""

    def __init__(self) -> None:
        self._agents: Dict[str, AgentRuntimeState] = {}

    def register(self, state: AgentRuntimeState) -> None:
        self._agents[state.agent_id] = state

    def get(self, agent_id: str) -> AgentRuntimeState:
        state = self._agents.get(agent_id)
        if state is None:
            raise KeyError(f"Unknown agent '{agent_id}'")
        return state

    def all_ids(self) -> list[str]:
        return list(self._agents.keys())

    def all_states(self) -> list[AgentRuntimeState]:
        return list(self._agents.values())

    def __contains__(self, agent_id: str) -> bool:
        return agent_id in self._agents

    def __len__(self) -> int:
        return len(self._agents)
