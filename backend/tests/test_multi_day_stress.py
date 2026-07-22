"""
Multi-day stress test: run a full simulation across the end-of-day transition
with mocked LLM calls. Verifies archive fires, next-day plan loads, and the
engine continues without corruption.
"""

from __future__ import annotations

import asyncio
import sys
import os
from pathlib import Path
from unittest.mock import patch, AsyncMock

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.append(str(BACKEND_ROOT))

from src.core.world_engine import WorldEngine
from src.core.agent_registry import AgentRuntimeState
from src.core.world_state import Position
from src.agents.Actions import AgentActionManager, LocationResolver
from src.agents.Short_term import (
    date_from_simulation_time, load_day_plan, save_day_plan,
)
from src import config as _cfg

THROWAWAY_DATE = "2099-06-15"
TOTAL_TICKS = 60

# Day-1 plan: last action starts early so transition fires during test window
DAY1_PLAN = [
    {"action": "Stand", "start": "00:00", "end": "00:15",
     "location_id": "Brahmaputra_Boys 1"},
    {"action": "Rest", "start": "00:15", "end": "03:00",
     "location_id": "Brahmaputra_Boys 1"},
]

DAY2_PLAN = [
    {"action": "Breakfast", "start": "00:00", "end": "00:10",
     "location_id": "Brahmaputra_Boys 1"},
    {"action": "Study", "start": "00:10", "end": "03:00",
     "location_id": "Brahmaputra_Boys 1"},
]

def _register(engine, agent_id, name, plan, hostel="Brahmaputra_Boys 1"):
    resolver = LocationResolver()
    pos = resolver.resolve(hostel) or Position(x=185, y=547, location_id=hostel)
    mgr = AgentActionManager(agent_id, plan, pos, resolver)
    state = AgentRuntimeState(
        agent_id=agent_id,
        persona={"Name": name, "Hostel": hostel},
        persona_name=name,
        manager=mgr,
        position=pos,
        day_plan=plan,
    )
    engine.registry.register(state)
    engine.world.register_agent(agent_id, pos)
    return state


# ── Tests ──

async def test_transition_runs_cleanly():
    """Full end-of-day transition with mocked LLM: archive + next plan.
    Agents placed far apart to avoid conversation detection pausing them."""
    engine = WorldEngine(sim_start_date=THROWAWAY_DATE)
    _register(engine, "a1", "Agent One", DAY1_PLAN, hostel="Brahmaputra_Boys 1")
    _register(engine, "a2", "Agent Two", DAY1_PLAN, hostel="Chenab")  # far from Brahmaputra

    with patch("src.core.world_engine.archive_to_long_term",
               new_callable=AsyncMock) as mock_archive, \
         patch("src.agents.day_planner.run",
               return_value={"day_plan": DAY2_PLAN}), \
         patch("src.core.world_engine.generate_conversation",
               return_value=None):
        mock_archive.return_value = {"summary": "mock archive"}
        for _ in range(TOTAL_TICKS):
            result = await engine.run_tick()
            if not result:
                break

    assert mock_archive.call_count == 2, \
        f"Expected 2 archive calls (one per agent), got {mock_archive.call_count}"

    a1 = engine.registry.get("a1")
    a2 = engine.registry.get("a2")
    assert a1.day_archived, "Agent One should be archived"
    assert a2.day_archived, "Agent Two should be archived"
    assert len(a1.manager.day_plan) == len(DAY2_PLAN), \
        f"Agent One should have day-2 plan, got {len(a1.manager.day_plan)} actions"
    assert len(a2.manager.day_plan) == len(DAY2_PLAN), \
        f"Agent Two should have day-2 plan, got {len(a2.manager.day_plan)} actions"


async def test_no_crash_during_transition():
    """Engine should not crash when archive + planner run mid-sim."""
    engine = WorldEngine(sim_start_date=THROWAWAY_DATE)
    _register(engine, "a1", "Agent One", DAY1_PLAN)

    with patch("src.core.world_engine.archive_to_long_term",
               new_callable=AsyncMock) as mock_archive, \
         patch("src.agents.day_planner.run", return_value={"day_plan": DAY2_PLAN}):
        mock_archive.return_value = {"summary": "mock"}
        for _ in range(TOTAL_TICKS):
            result = await engine.run_tick()
            if not result:
                break

    assert engine.registry.get("a1").day_archived
    assert engine.world.tick <= TOTAL_TICKS


async def test_next_day_plan_has_actions():
    """After transition, the loaded next-day plan should have valid actions."""
    engine = WorldEngine(sim_start_date=THROWAWAY_DATE)
    _register(engine, "a1", "Agent One", DAY1_PLAN)

    with patch("src.core.world_engine.archive_to_long_term",
               new_callable=AsyncMock) as mock_archive, \
         patch("src.agents.day_planner.run", return_value={"day_plan": DAY2_PLAN}):
        mock_archive.return_value = {"summary": "mock"}
        for _ in range(TOTAL_TICKS):
            result = await engine.run_tick()
            if not result:
                break

    mgr = engine.registry.get("a1").manager
    assert len(mgr.day_plan) > 0, "Next-day plan should have actions"
    first = mgr.day_plan[0]
    assert "start" in first, "Each action should have a start time"
    assert "action" in first, "Each action should have a description"
    assert "location_id" in first, "Each action should have a location_id"


async def test_multi_agent_transition_ordering():
    """Multiple agents should all transition, regardless of order in registry."""
    engine = WorldEngine(sim_start_date=THROWAWAY_DATE)
    for i in range(3):
        _register(engine, f"a{i}", f"Agent {i}", DAY1_PLAN)

    transitioned = set()

    async def _track_archive(name, date_str):
        transitioned.add(name)
        return {"summary": f"archived {name}"}

    with patch("src.core.world_engine.archive_to_long_term",
               new_callable=AsyncMock) as mock_archive, \
         patch("src.agents.day_planner.run", return_value={"day_plan": DAY2_PLAN}):
        mock_archive.side_effect = _track_archive
        for _ in range(TOTAL_TICKS):
            result = await engine.run_tick()
            if not result:
                break

    for i in range(3):
        assert engine.registry.get(f"a{i}").day_archived, \
            f"Agent {i} should be archived"
        assert len(engine.registry.get(f"a{i}").manager.day_plan) == len(DAY2_PLAN), \
            f"Agent {i} should have day-2 plan"
    assert len(transitioned) == 3, f"All 3 agents should transition, got {transitioned}"


async def test_plan_not_corrupted_after_transition():
    """After transition, the agent's manager should have a clean plan state."""
    engine = WorldEngine(sim_start_date=THROWAWAY_DATE)
    _register(engine, "a1", "Agent One", DAY1_PLAN)

    with patch("src.core.world_engine.archive_to_long_term",
               new_callable=AsyncMock) as mock_archive, \
         patch("src.agents.day_planner.run", return_value={"day_plan": DAY2_PLAN}):
        mock_archive.return_value = {"summary": "mock"}
        for _ in range(TOTAL_TICKS):
            result = await engine.run_tick()
            if not result:
                break

    mgr = engine.registry.get("a1").manager
    # Plan should be sorted (the manager sorts by start time)
    for i in range(len(mgr.day_plan) - 1):
        t1 = mgr._hhmm_to_minutes(mgr.day_plan[i].get("start", "99:99"))
        t2 = mgr._hhmm_to_minutes(mgr.day_plan[i + 1].get("start", "99:99"))
        assert t1 <= t2, f"Plan actions out of order at index {i}"
    # The marker is expected while the agent is still executing the final
    # activity. The engine must instead use the archival guard to prevent
    # duplicate archival until the calendar-day handoff occurs.
    assert engine.registry.get("a1").day_archived


# ── Runner ──

async def main() -> int:
    tests = [
        ("transition_runs_cleanly", test_transition_runs_cleanly),
        ("no_crash_during_transition", test_no_crash_during_transition),
        ("next_day_plan_has_actions", test_next_day_plan_has_actions),
        ("multi_agent_ordering", test_multi_agent_transition_ordering),
        ("plan_not_corrupted", test_plan_not_corrupted_after_transition),
    ]
    failures = 0
    for name, fn in tests:
        try:
            await fn()
            print(f"  PASS  {name}")
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"  FAIL  {name}: {e}")
            failures += 1
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    return failures


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
