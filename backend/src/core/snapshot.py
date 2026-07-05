"""
Snapshot -- point-in-time read-only view of WorldState (Immutable)

Every tick takes exactly one `WorldSnapshot` and hands the
*same* frozen object to every agent's tick graph via asyncio.gather(). That's
what makes parallel agent decisions safe -- nobody is reading a WorldState
that's being mutated mid-tick by someone else's action.

`WorldSnapshot` is a deliberately separate class from `WorldState`, not just
a deep copy of it. It exposes zero mutating methods, so there is no method
an agent's perceive/react/plan code could accidentally call that would
corrupt the resolve phase's assumptions. If you need a new read-only query
(e.g. "what's the nearest free table"), add it here as a method on
`WorldSnapshot` -- don't reach into `.agents`/`.occupancy` directly from
perceive.py and reimplement the same query logic in multiple places.

Usage
-----
    from src.core.world_state import WorldState, Position
    from src.core.snapshot import take_snapshot

    world = WorldState()
    world.register_agent("gurnoor", Position(x=4, y=2, location_id="dorm_room_1"))

    snap = take_snapshot(world)          # take ONCE per tick, before decide phase
    snap.get_agent("gurnoor")            # read-only query
    snap.agents_near("gurnoor", radius=3)
    snap.is_free("cafeteria_table_3")

    # snap.tick = 999          <- raises, frozen model
    # snap.agents["x"] = ...   <- raises, frozen model
"""

from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel, ConfigDict

from src.core.log import get_logger
from src.core.world_state import AgentState, WorldState, Position

logger = get_logger(__name__)


class WorldSnapshot(BaseModel):
    """
    Frozen copy of WorldState taken at the start of a tick's decide phase.

    Note on depth of immutability: the top-level model is frozen (you can't
    reassign `snap.tick`, `snap.agents`, etc.), and every `AgentState` inside
    `agents` is a deep copy made at snapshot time -- so even if something
    reaches in and mutates `snap.agents["x"].position`, that mutation is
    inert: it's a throwaway copy, isolated from the real WorldState and from
    every other agent's snapshot reference. It cannot leak into the resolve
    phase. Don't rely on that as a substitute for writing correct code, but
    it does mean a stray mutation is a bug in one agent's own reasoning, not
    a simulation-wide correctness issue.
    """
    model_config = ConfigDict(frozen=True)

    tick: int
    agents: Dict[str, AgentState]
    occupancy: Dict[str, Optional[str]]

    # Read-only queries

    def get_agent(self, agent_id: str) -> AgentState:
        if agent_id not in self.agents:
            raise KeyError(f"Unknown agent_id '{agent_id}' in snapshot")
        return self.agents[agent_id]

    def all_agent_ids(self) -> List[str]:
        return list(self.agents.keys())

    def other_agents(self, agent_id: str) -> List[AgentState]:
        """Every agent except `agent_id`. Convenience for perceive.py."""
        return [a for a in self.agents.values() if a.agent_id != agent_id]

    def agents_near(self, agent_id: str, radius: int) -> List[AgentState]:
        """
        Chebyshev-distance ("square") radius query -- i.e. everyone within
        a (radius*2+1) x (radius*2+1) tile box centered on `agent_id`.
        perceive.py should not reimplement this math, just call this method.
        """
        ### NOTE : Can swap the distance metric to squared euclidian distance.
        me = self.get_agent(agent_id)
        nearby = []
        for other in self.other_agents(agent_id):
            dx = abs(other.position.x - me.position.x)
            dy = abs(other.position.y - me.position.y)
            if max(dx, dy) <= radius:
                nearby.append(other)
        return nearby

    def agents_at_location(self, location_id: str) -> List[AgentState]:
        """Everyone currently standing in a named semantic zone."""
        return [a for a in self.agents.values() if a.position.location_id == location_id]

    def is_free(self, resource_id: str) -> bool:
        if resource_id not in self.occupancy:
            logger.warning("Resource '%s' was never registered.", resource_id)
        return self.occupancy.get(resource_id) is None

    def holder_of(self, resource_id: str) -> Optional[str]:
        return self.occupancy.get(resource_id)

    def free_resources(self) -> List[str]:
        return [rid for rid, holder in self.occupancy.items() if holder is None]


def take_snapshot(world: WorldState) -> WorldSnapshot:
    """
    Build an immutable snapshot from the live WorldState.

    Call this exactly once per tick, at the very start of the decide phase,
    before any agent tick graph runs -- then pass the *same* `WorldSnapshot`
    instance to every agent invoked via asyncio.gather() this tick. Do not
    call this once per agent; that would let a slow tick graph observe a
    world one tick "younger" than a fast one that started later in the same
    gather() batch, which reintroduces the exact ordering bug lockstep is
    meant to eliminate.
    """
    agents_copy = {
        agent_id: agent_state.model_copy(deep=True)
        for agent_id, agent_state in world.agents.items()
    }
    occupancy_copy = dict(world.occupancy)  # values are str | None, shallow copy is sufficient

    snapshot = WorldSnapshot(
        tick=world.tick,
        agents=agents_copy,
        occupancy=occupancy_copy,
    )
    logger.debug(
        "Snapshot taken at tick=%d (%d agents, %d resources)",
        world.tick, len(agents_copy), len(occupancy_copy),
    )
    return snapshot


# Standalone sanity check -- matches the "Done when" check from the build plan:
# instantiate WorldState with one dummy agent, take a snapshot, mutate the
# original, confirm the snapshot didn't change.

if __name__ == "__main__":
    from src.core.world_state import CurrentAction

    world = WorldState()
    world.register_agent("test_agent", Position(x=0, y=0, location_id="start"))
    world.register_resource("test_chair")

    snap = take_snapshot(world)
    assert snap.tick == 0
    assert snap.get_agent("test_agent").position.x == 0
    assert snap.is_free("test_chair")

    # mutate the REAL world after the snapshot was taken
    world.move_agent("test_agent", Position(x=5, y=5, location_id="cafeteria"))
    world.occupy("test_chair", "test_agent")
    world.advance_tick()

    # snapshot must be unaffected
    assert snap.tick == 0, "snapshot tick leaked a live update"
    assert snap.get_agent("test_agent").position.x == 0, "snapshot agent position leaked a live update"
    assert snap.is_free("test_chair"), "snapshot occupancy leaked a live update"

    # frozen model: reassignment must raise
    try:
        snap.tick = 999
        raise AssertionError("expected snapshot mutation to raise, it didn't")
    except Exception as e:
        assert type(e).__name__ in ("ValidationError", "TypeError"), f"unexpected error type: {type(e)}"

    print("snapshot.py sanity check passed.")