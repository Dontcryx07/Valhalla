"""
World State -- the single source of truth for the simulation.

`WorldState` holds everything that is true about the world at a given tick:
where every agent is, what they're currently doing, and who/what currently
holds any contested resource (a chair, an NPC's attention, a location slot).

Ownership rule:
    Only `WorldEngine`'s resolve phase should ever call the mutating methods
    on this class directly (`set_agent_action`, `move_agent`, `occupy`, ...).

    Every agent tick graph (perceive -> retrieve -> react -> day_planner ->
    act) must only ever see a frozen copy produced by `core/snapshot.py`.

    That separation is what keeps the decide phase safely parallelizable
    with asyncio.gather() -- nobody is reading a WorldState that something
    else is mutating mid-tick.

Usage
-----
    from src.core.world_state import WorldState, Position

    world = WorldState()
    world.register_agent("gurnoor", Position(x=4, y=2, location_id="dorm_room_1"))
    world.register_resource("cafeteria_table_3")

    # inside the resolve phase, after an agent's tick graph proposed an action:
    world.occupy("cafeteria_table_3", "gurnoor")
    world.set_agent_action("gurnoor", CurrentAction(
        description="eating breakfast",
        start_tick=world.tick,
        end_tick=world.tick + 20,
        target_object_id="cafeteria_table_3",
    ))

    world.advance_tick(minutes=10)
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel, Field

from src.core.log import get_logger

# Logging
logger = get_logger(__name__)


# Enums / small value types

class AgentStatus(str, Enum):
    IDLE = "idle"           # no current action, needs a decision
    ACTING = "acting"       # mid-action, action_end_tick in the future
    INTERRUPTED = "interrupted"  # was acting, got interrupted by a perceived event


class Position(BaseModel):
    """Tile coordinates plus an optional semantic zone name."""
    x: int
    y: int
    location_id: Optional[str] = None  # e.g. "cafeteria", "dorm_room_1"


class CurrentAction(BaseModel):
    """An action an agent has committed to, spanning [start_tick, end_tick)."""
    description: str
    start_tick: int
    end_tick: int
    target_location_id: Optional[str] = None
    target_object_id: Optional[str] = None
    # set by the resolver if a conflict forced a different outcome than requested
    was_reprioritized: bool = False

    def is_finished(self, tick: int) -> bool:
        return tick >= self.end_tick


class AgentState(BaseModel):
    agent_id: str
    position: Position
    status: AgentStatus = AgentStatus.IDLE
    current_action: Optional[CurrentAction] = None
    last_updated_tick: int = 0


class ActionLogEntry(BaseModel):
    """One row of the append-only history, used later for replay/debugging."""
    tick: int
    agent_id: str
    action: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


# WorldState
class WorldState(BaseModel):
    """
    Mutable, canonical world state.

    `agents`: dict of : agent_id -> AgentState
    `occupancy`: resource_id (a location slot or object) -> agent_id holding
        it, or None if free. Register a resource once with `register_resource`
        before anyone can occupy it -- this catches typos in resource ids
        early instead of silently creating them on first use.
    `history`: append-only action log, source of truth for replay/debugging.
    """
    tick: int = 0
    agents: Dict[str, AgentState] = Field(default_factory=dict)
    occupancy: Dict[str, Optional[str]] = Field(default_factory=dict)
    history: List[ActionLogEntry] = Field(default_factory=list)

    # Agent registry

    def register_agent(self, agent_id: str, position: Position) -> None:
        if agent_id in self.agents:
            logger.warning("Agent '%s' already registered; overwriting.", agent_id)
        self.agents[agent_id] = AgentState(
            agent_id=agent_id, position=position, last_updated_tick=self.tick
        )
        logger.info(
            "Registered agent '%s' at (%d, %d) [tick=%d]",
            agent_id, position.x, position.y, self.tick,
        )

    def get_agent(self, agent_id: str) -> AgentState:
        if agent_id not in self.agents:
            raise KeyError(f"Unknown agent_id '{agent_id}'")
        return self.agents[agent_id]

    def all_agent_ids(self) -> List[str]:
        return list(self.agents.keys())

    # Occupancy / resource contention with agents

    def register_resource(self, resource_id: str) -> None:
        """Idempotent -- safe to call every time a location/object is defined."""
        self.occupancy.setdefault(resource_id, None)

    def is_free(self, resource_id: str) -> bool:
        if resource_id not in self.occupancy:
            logger.warning("Resource '%s' was never registered.", resource_id)
        return self.occupancy.get(resource_id) is None

    def occupy(self, resource_id: str, agent_id: str) -> bool:
        """
        Attempt to claim a resource for an agent.

        Returns False (and does NOT mutate state) if it's already held by a
        *different* agent -- resolver.py is expected to check this return
        value and arbitrate, not assume success.
        """
        holder = self.occupancy.get(resource_id)
        if holder is not None and holder != agent_id:
            return False
        self.occupancy[resource_id] = agent_id
        return True

    def release(self, resource_id: str) -> None:
        if resource_id in self.occupancy:
            self.occupancy[resource_id] = None

    def release_all_for_agent(self, agent_id: str) -> None:
        """Call this when clearing/interrupting an agent's action."""
        for resource_id, holder in self.occupancy.items():
            if holder == agent_id:
                self.occupancy[resource_id] = None

    # Mutations -- resolve phase only

    def set_agent_action(self, agent_id: str, action: CurrentAction) -> None:
        agent = self.get_agent(agent_id)
        agent.current_action = action
        agent.status = AgentStatus.ACTING
        agent.last_updated_tick = self.tick
        self.history.append(
            ActionLogEntry(tick=self.tick, agent_id=agent_id, action=action.description)
        )

    def move_agent(self, agent_id: str, position: Position) -> None:
        agent = self.get_agent(agent_id)
        agent.position = position
        agent.last_updated_tick = self.tick

    def interrupt_agent(self, agent_id: str) -> None:
        """Called by event_bus.py when a perceived event forces a replan."""
        agent = self.get_agent(agent_id)
        agent.status = AgentStatus.INTERRUPTED
        self.release_all_for_agent(agent_id)

    def clear_agent_action(self, agent_id: str) -> None:
        agent = self.get_agent(agent_id)
        agent.current_action = None
        agent.status = AgentStatus.IDLE

    # Scheduling helper (used by engine/scheduler.py)

    def agents_ready_for_decision(self) -> List[str]:
        """
        Agents who need a decision this tick: no current action, current
        action just finished, or were flagged INTERRUPTED.
        """
        ready = []
        for agent_id, agent in self.agents.items():
            if (
                agent.current_action is None
                or agent.current_action.is_finished(self.tick)
                or agent.status == AgentStatus.INTERRUPTED
            ):
                ready.append(agent_id)
        return ready

    # Tick control, moving the ticks

    def advance_tick(self, minutes: int = 10) -> None:
        self.tick += minutes
        logger.debug("World tick advanced to %d", self.tick)