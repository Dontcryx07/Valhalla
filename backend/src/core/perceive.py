"""
Perceive -- turns a WorldSnapshot into what one agent can currently observe.

This is a pure function of snapshot data: no LLM calls, no side effects.
With only one agent registered (your current single-agent phase), this
naturally returns an empty list every tick -- no special-casing needed to
"turn on" perception later, it already does the real spatial query.

Radius/distance logic itself lives on `WorldSnapshot` (core/snapshot.py) so
there's exactly one implementation of "who's nearby" in the codebase. (in snapshot.py)
"""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel

from src.core.log import get_logger
from src.core.snapshot import WorldSnapshot
from src.core.world_state import AgentState

from src.config import DEFAULT_PERCEPTION_RADIUS

logger = get_logger(__name__)


class Observation(BaseModel):
    """One thing an agent noticed this tick."""
    tick: int
    observer_id: str
    subject_agent_id: str
    description: str            # human-readable, e.g. "Gurnoor is eating breakfast"
    location_id: Optional[str] = None
    distance: int = 0           # chebyshev tiles from observer; lower = more salient


def perceive(
    snapshot: WorldSnapshot,
    agent_id: str,
    radius: int = DEFAULT_PERCEPTION_RADIUS,
) -> List[Observation]:
    """
    Return everything `agent_id` can currently observe in `snapshot`,
    nearest first.

    Called from tick_graph.py's perceive node with the tick's single shared
    snapshot -- never call this once the underlying WorldState might have
    moved on; always pass the snapshot the whole tick is using.
    """
    if agent_id not in snapshot.agents:
        raise KeyError(f"'{agent_id}' not present in this snapshot")

    me = snapshot.get_agent(agent_id)
    nearby = snapshot.agents_near(agent_id, radius)

    observations: List[Observation] = []
    for other in nearby:
        distance = max(
            abs(other.position.x - me.position.x),
            abs(other.position.y - me.position.y),
        )
        observations.append(
            Observation(
                tick=snapshot.tick,
                observer_id=agent_id,
                subject_agent_id=other.agent_id,
                description=_describe(other),
                location_id=other.position.location_id,
                distance=distance,
            )
        )

    observations.sort(key=lambda o: o.distance)
    if observations:
        logger.debug(
            "Agent '%s' perceived %d nearby agent(s) at tick %d",
            agent_id, len(observations), snapshot.tick,
        )
    return observations


def perceive_px(
    snapshot: WorldSnapshot,
    agent_id: str,
    radius_px: float,
) -> List[Observation]:
    """
    Pixel-space (Euclidean circle) perception -- everything `agent_id` can see
    within `radius_px` pixels, nearest first. This is the 0-LLM sensory input
    behind the 50px proximity circle. Distance is rounded pixel distance.
    """
    if agent_id not in snapshot.agents:
        raise KeyError(f"'{agent_id}' not present in this snapshot")

    me = snapshot.get_agent(agent_id)
    nearby = snapshot.agents_within_px(agent_id, radius_px)

    observations: List[Observation] = []
    for other in nearby:
        dx = other.position.x - me.position.x
        dy = other.position.y - me.position.y
        dist = int(round((dx * dx + dy * dy) ** 0.5))
        observations.append(
            Observation(
                tick=snapshot.tick,
                observer_id=agent_id,
                subject_agent_id=other.agent_id,
                description=_describe(other),
                location_id=other.position.location_id,
                distance=dist,
            )
        )
    # agents_within_px already returns nearest-first; keep that order.
    if observations:
        logger.debug(
            "Agent '%s' perceived %d agent(s) within %spx at tick %d",
            agent_id, len(observations), radius_px, snapshot.tick,
        )
    return observations


def _describe(agent: AgentState) -> str:
    if agent.current_action is not None:
        return f"{agent.agent_id} is {agent.current_action.description}"
    return f"{agent.agent_id} is idle"


# Standalone sanity check
if __name__ == "__main__":
    from src.core.world_state import CurrentAction, Position, WorldState
    from src.core.snapshot import take_snapshot

    world = WorldState()
    world.register_agent("a", Position(x=0, y=0, location_id="quad"))
    world.register_agent("b", Position(x=2, y=1, location_id="quad"))
    world.set_agent_action(
        "b", CurrentAction(description="reading a book", start_tick=0, end_tick=30)
    )

    snap = take_snapshot(world)

    obs_a = perceive(snap, "a", radius=5)

    print(obs_a)

    assert len(obs_a) == 1
    assert obs_a[0].subject_agent_id == "b"
    assert "reading a book" in obs_a[0].description

    obs_a_tight = perceive(snap, "a", radius=1)
    assert len(obs_a_tight) == 0, "radius 1 should not see an agent at chebyshev distance 2"

    print("perceive.py sanity check passed.")