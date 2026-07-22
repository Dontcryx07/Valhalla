"""
Tests for conversation detection logic in WorldEngine.
No API keys needed — all LLM calls are mocked.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from unittest.mock import patch, AsyncMock

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.append(str(BACKEND_ROOT))

from src.core.world_engine import WorldEngine
from src.core.agent_registry import AgentRuntimeState
from src.core.world_state import Position
from src.agents.Actions import ActionState, ActionType, AgentActionManager, LocationResolver
from src import config as _cfg

THROWAWAY_DATE = "2099-06-01"
SAME_SPOT = Position(x=100, y=100, location_id="assembly_point")
FAR_SPOT = Position(x=1000, y=1000, location_id="far_away")
HOSTEL_SPOT = Position(x=185, y=547, location_id="Brahmaputra_Boys 1")


def _plan(actions=None):
    if actions is None:
        actions = [
            {"action": "Stand", "start": "00:00", "end": "00:10",
             "location_id": "assembly_point"},
            {"action": "Walk", "start": "00:10", "end": "23:00",
             "location_id": "assembly_point"},
        ]
    return actions


def _register(engine, agent_id, name, position, plan=None):
    plan = plan or _plan()
    resolver = LocationResolver()
    mgr = AgentActionManager(agent_id, plan, position, resolver)
    state = AgentRuntimeState(
        agent_id=agent_id,
        persona={"Name": name, "Hostel": position.location_id or "assembly_point"},
        persona_name=name,
        manager=mgr,
        position=position,
        day_plan=plan,
    )
    engine.registry.register(state)
    engine.world.register_agent(agent_id, position)
    return state


async def _trigger_and_detect(engine, tick=0):
    """Call _check_conversations with mocked LLM, return after bg task completes."""
    with patch("src.core.world_engine.generate_conversation", return_value=None):
        await engine._check_conversations(tick, engine._minutes_to_hhmm(tick))
        await asyncio.sleep(0)


# ── Tests ──

async def test_pair_detected():
    engine = WorldEngine(sim_start_date=THROWAWAY_DATE)
    _register(engine, "a1", "Agent A", SAME_SPOT)
    _register(engine, "a2", "Agent B", SAME_SPOT)
    await _trigger_and_detect(engine)
    a1, a2 = engine.registry.get("a1"), engine.registry.get("a2")
    assert a1.paused
    assert a2.paused
    assert a1.conversation_count == 1
    assert a2.conversation_count == 1
    assert a1.last_conversation_partner == "a2"
    assert a2.last_conversation_partner == "a1"


async def test_crowd_skipped():
    engine = WorldEngine(sim_start_date=THROWAWAY_DATE)
    _register(engine, "a1", "Agent A", SAME_SPOT)
    _register(engine, "a2", "Agent B", SAME_SPOT)
    _register(engine, "a3", "Agent C", SAME_SPOT)
    await _trigger_and_detect(engine)
    for aid in ("a1", "a2", "a3"):
        assert not engine.registry.get(aid).paused, f"{aid} should NOT be paused (crowd rule)"


async def test_far_agents_no_conversation():
    engine = WorldEngine(sim_start_date=THROWAWAY_DATE)
    _register(engine, "a1", "Agent A", SAME_SPOT)
    _register(engine, "a2", "Agent B", FAR_SPOT)
    await _trigger_and_detect(engine)
    assert not engine.registry.get("a1").paused
    assert not engine.registry.get("a2").paused


async def test_transit_or_different_venues_cannot_start_a_conversation():
    """Proximity is meaningful only after both agents have arrived together."""
    engine = WorldEngine(sim_start_date=THROWAWAY_DATE)
    a = _register(engine, "a1", "Agent A", Position(x=100, y=100, location_id="mess"))
    b = _register(engine, "a2", "Agent B", Position(x=106, y=100, location_id="library"))
    b.manager.current_action = ActionState(
        action_type=ActionType.MOVE,
        description="Walk to mess",
        start_time="08:00",
        end_time="08:10",
        location_id="mess",
        position=Position(x=100, y=100, location_id="mess"),
        path=[(106, 100), (100, 100)],
    )
    await _trigger_and_detect(engine)
    assert not a.paused
    assert not b.paused

    # Even stationary agents do not chat across different semantic venues.
    b.manager.current_action = ActionState(
        action_type=ActionType.MISC,
        description="Studying",
        start_time="08:00",
        end_time="09:00",
        location_id="library",
        position=b.position,
    )
    await _trigger_and_detect(engine)
    assert not a.paused
    assert not b.paused


async def test_cooldown_respected():
    engine = WorldEngine(sim_start_date=THROWAWAY_DATE)
    _register(engine, "a1", "Agent A", SAME_SPOT)
    _register(engine, "a2", "Agent B", SAME_SPOT)
    # Simulate a recent conversation: cooldown is 30 ticks
    engine.world.record_conversation("a1")
    engine.world.record_conversation("a2")
    engine.world.tick = 5  # well within 30-tick cooldown
    await _trigger_and_detect(engine, tick=5)
    assert not engine.registry.get("a1").paused, "Cooldown should prevent conversation"


async def test_blocked_action_skipped():
    engine = WorldEngine(sim_start_date=THROWAWAY_DATE)
    blocked_plan = [{"action": "Sleep", "start": "00:00", "end": "23:00",
                     "location_id": "Brahmaputra_Boys 1"}]
    _register(engine, "a1", "Agent A", HOSTEL_SPOT, plan=blocked_plan)
    _register(engine, "a2", "Agent B", HOSTEL_SPOT)
    # Initialize managers so they have current_action set (Sleep)
    for s in engine.registry.all_states():
        if s.manager:
            s.manager.tick(0)
    await _trigger_and_detect(engine)
    assert not engine.registry.get("a1").paused, "Sleeping agent should not converse"


async def test_max_conversations_cap():
    engine = WorldEngine(sim_start_date=THROWAWAY_DATE)
    _register(engine, "a1", "Agent A", SAME_SPOT)
    _register(engine, "a2", "Agent B", SAME_SPOT)
    # Set one agent at the max cap
    engine.registry.get("a1").conversation_count = _cfg.MAX_CONVERSATIONS_PER_AGENT
    await _trigger_and_detect(engine)
    assert not engine.registry.get("a1").paused, "Capped agent should not converse"


async def test_back_to_back_repeat_blocked():
    engine = WorldEngine(sim_start_date=THROWAWAY_DATE)
    _register(engine, "a1", "Agent A", SAME_SPOT)
    _register(engine, "a2", "Agent B", SAME_SPOT)
    # Same partners as last time → blocked
    engine.registry.get("a1").last_conversation_partner = "a2"
    engine.registry.get("a2").last_conversation_partner = "a1"
    await _trigger_and_detect(engine)
    assert not engine.registry.get("a1").paused, "Repeat partners should be blocked"


# ── Runner ──

async def main() -> int:
    tests = [
        ("pair_detected", test_pair_detected),
        ("crowd_skipped", test_crowd_skipped),
        ("far_agents", test_far_agents_no_conversation),
        ("transit_or_different_venues", test_transit_or_different_venues_cannot_start_a_conversation),
        ("cooldown", test_cooldown_respected),
        ("blocked_action", test_blocked_action_skipped),
        ("max_cap", test_max_conversations_cap),
        ("back_to_back", test_back_to_back_repeat_blocked),
    ]
    failures = 0
    for name, fn in tests:
        try:
            await fn()
            print(f"  PASS  {name}")
        except Exception as e:
            print(f"  FAIL  {name}: {e}")
            failures += 1
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    return failures


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
