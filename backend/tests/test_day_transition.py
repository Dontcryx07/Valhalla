"""
Tests for end-of-day archival and next-day planning in WorldEngine.
No API keys needed — all LLM calls are mocked.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from unittest.mock import patch, AsyncMock
from types import SimpleNamespace
from datetime import datetime, timedelta

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.append(str(BACKEND_ROOT))

from src.core.world_engine import WorldEngine
from src.core.agent_registry import AgentRuntimeState
from src.core.world_state import Position
from src.agents.Actions import AgentActionManager, LocationResolver
from src.agents.Short_term import (
    load_day_plan, save_day_plan, date_from_simulation_time,
    consolidate_to_single_day, reset_day_runtime,
)
from src import config as _cfg

THROWAWAY_DATE = "2099-06-01"

# A plan where last action starts at tick 15 so transition fires early
END_OF_DAY_PLAN = [
    {"action": "Stand", "start": "00:00", "end": "00:10",
     "location_id": "Brahmaputra_Boys 1"},
    {"action": "Sleep", "start": "00:10", "end": "03:00",
     "location_id": "Brahmaputra_Boys 1"},
]

NEXT_DAY_PLAN = [
    {"action": "Breakfast", "start": "00:00", "end": "00:10",
     "location_id": "Brahmaputra_Boys 1"},
    {"action": "Study", "start": "00:10", "end": "03:00",
     "location_id": "Brahmaputra_Boys 1"},
]


def _register(engine, agent_id, name, plan, hostel="Brahmaputra_Boys 1"):
    pos = Position(x=185, y=547, location_id=hostel)
    resolver = LocationResolver()
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


def _tick_until_last_action(engine, target_tick=15):
    """Tick agents so they enter their last action at or before target_tick."""
    for s in engine.registry.all_states():
        if s.manager:
            for t in range(target_tick + 1):
                s.manager.tick(t)


# ── Tests ──

async def test_archive_on_last_action():
    engine = WorldEngine(sim_start_date=THROWAWAY_DATE)
    _register(engine, "a1", "Agent A", END_OF_DAY_PLAN)
    _tick_until_last_action(engine, 15)
    assert engine.registry.get("a1").manager._entered_last_action, \
        "Agent should have entered last action by tick 15"

    with patch("src.core.world_engine.archive_to_long_term",
               new_callable=AsyncMock) as mock_archive, \
         patch("src.agents.day_planner.run",
               return_value={"day_plan": NEXT_DAY_PLAN}):
        mock_archive.return_value = {"summary": "mock archive success"}
        await engine._check_last_action_triggers(15, "00:15")

    assert engine.registry.get("a1").day_archived, "day_archived flag should be set"
    mock_archive.assert_called_once()
    # Verify the next-day plan was loaded into the manager
    mgr = engine.registry.get("a1").manager
    assert len(mgr.day_plan) == len(NEXT_DAY_PLAN), \
        f"Expected {len(NEXT_DAY_PLAN)} actions, got {len(mgr.day_plan)}"


async def test_no_double_archive():
    engine = WorldEngine(sim_start_date=THROWAWAY_DATE)
    _register(engine, "a1", "Agent A", END_OF_DAY_PLAN)
    _tick_until_last_action(engine, 15)
    engine.registry.get("a1").day_archived = True  # Already archived

    call_count = 0

    async def _fake_archive(*a, **kw):
        nonlocal call_count
        call_count += 1
        return {"summary": "mock"}

    with patch("src.core.world_engine.archive_to_long_term",
               new_callable=AsyncMock) as mock_archive, \
         patch("src.agents.day_planner.run",
               return_value={"day_plan": NEXT_DAY_PLAN}):
        mock_archive.side_effect = _fake_archive
        await engine._check_last_action_triggers(15, "00:15")

    assert call_count == 0, "archive should NOT be called when day_archived is True"


async def test_archive_error_isolation():
    """If one agent's archive fails, other agents should still transition."""
    engine = WorldEngine(sim_start_date=THROWAWAY_DATE)
    _register(engine, "a_ok", "Agent OK", END_OF_DAY_PLAN)
    _register(engine, "a_fail", "Agent Fail", END_OF_DAY_PLAN)
    _tick_until_last_action(engine, 15)

    call_log = []

    async def _mock_archive(name, date_str):
        call_log.append((name, date_str))
        if name == "Agent Fail":
            raise RuntimeError("Simulated archive failure")
        return {"summary": f"archived {name}"}

    def _mock_planner(proxy, params):
        return {"day_plan": NEXT_DAY_PLAN}

    with patch("src.core.world_engine.archive_to_long_term",
               new_callable=AsyncMock) as mock_archive, \
         patch("src.agents.day_planner.run", side_effect=_mock_planner):
        mock_archive.side_effect = _mock_archive
        await engine._check_last_action_triggers(15, "00:15")

    ok_state = engine.registry.get("a_ok")
    fail_state = engine.registry.get("a_fail")
    assert ok_state.day_archived, "OK agent should be archived"
    assert fail_state.day_archived, "Fail agent should still have day_archived set"
    assert len(call_log) == 2, "archive should be called for both agents"
    assert ok_state.manager.day_plan == NEXT_DAY_PLAN, \
        "OK agent should have next-day plan loaded"


async def test_replace_day_plan_mid_stream():
    """Verify replace_day_plan correctly swaps the plan."""
    engine = WorldEngine(sim_start_date=THROWAWAY_DATE)
    _register(engine, "a1", "Agent A", END_OF_DAY_PLAN)
    _tick_until_last_action(engine, 15)
    agent = engine.registry.get("a1")
    agent.manager.replace_day_plan(NEXT_DAY_PLAN)
    assert len(agent.manager.day_plan) == len(NEXT_DAY_PLAN)
    assert not agent.manager._entered_last_action, \
        "replace_day_plan should reset _entered_last_action"


async def test_next_day_plan_starts_at_00_00():
    """Verify the generated next-day plan covers 00:00 onward."""
    engine = WorldEngine(sim_start_date=THROWAWAY_DATE)
    _register(engine, "a1", "Agent A", END_OF_DAY_PLAN)
    _tick_until_last_action(engine, 15)

    with patch("src.core.world_engine.archive_to_long_term",
               new_callable=AsyncMock) as mock_archive, \
         patch("src.agents.day_planner.run",
               return_value={"day_plan": NEXT_DAY_PLAN}):
        mock_archive.return_value = {"summary": "mock"}
        await engine._check_last_action_triggers(15, "00:15")

    mgr = engine.registry.get("a1").manager
    first = mgr.day_plan[0]
    assert first.get("start") == "00:00", \
        f"Next-day plan should start at 00:00, got {first.get('start')}"


# ── Runner ──

async def main() -> int:
    tests = [
        ("archive_on_last_action", test_archive_on_last_action),
        ("no_double_archive", test_no_double_archive),
        ("archive_error_isolation", test_archive_error_isolation),
        ("replace_day_plan", test_replace_day_plan_mid_stream),
        ("next_day_starts_00_00", test_next_day_plan_starts_at_00_00),
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
