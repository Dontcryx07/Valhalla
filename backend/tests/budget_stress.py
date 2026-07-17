"""
Budget stress test (headless, no API keys needed).

Proves the core budget guarantee: on quiet ticks -- agents executing their
plans, perceiving, and being checked for proximity -- the simulation makes
ZERO LLM calls. This exercises the real tick loop (perception + proximity +
brain/body + checkpoint), just without triggering any cognition (agents kept
far apart; reflex-LLM off by default; no end-of-day archival in-window).

Run:
    cd backend
    $env:PYTHONPATH="."; ..\\venv\\Scripts\\python.exe tests\\budget_stress.py
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.append(str(BACKEND_ROOT))

from src.core.log import setup_logging
setup_logging(run_id="budget_stress", console=False)

from src.core.budget import GOVERNOR
from src.core.world_engine import WorldEngine
from src.core.agent_registry import AgentRuntimeState
from src.core.world_state import Position
from src.agents.Actions import AgentActionManager, LocationResolver
from src.agents.Short_term import load_day_plan

THROWAWAY_DATE = "2099-01-01"
N_TICKS = 40


def _cleanup():
    """Remove any throwaway short-term files this test may have written."""
    from src.config import DATA_DIR
    for base in ("Short_term_db",):
        for f in (DATA_DIR / base).glob(f"*/{THROWAWAY_DATE}.json"):
            try:
                f.unlink()
            except Exception:
                pass


async def main() -> int:
    GOVERNOR.reset()
    resolver = LocationResolver()

    engine = WorldEngine(sim_start_date=THROWAWAY_DATE)
    engine.resolver = resolver

    # Build two agents from real day plans, placed FAR apart so no proximity
    # conversation ever triggers (which would be the only LLM call in-window).
    specs = [
        ("gurnoor_singh", "Gurnoor Singh", Position(x=100, y=100, location_id="far_a")),
        ("parv_singla", "Parv Singla", Position(x=3000, y=3000, location_id="far_b")),
    ]
    for agent_id, name, pos in specs:
        plan = load_day_plan(name, "2026-07-03") or []
        mgr = AgentActionManager(agent_id, plan, pos, resolver)
        mgr.position = pos  # keep them apart regardless of plan
        engine.world.register_agent(agent_id, pos)
        engine.registry.register(AgentRuntimeState(
            agent_id=agent_id, persona={"Name": name}, persona_name=name,
            manager=mgr, position=pos, day_plan=plan,
        ))

    for _ in range(N_TICKS):
        await engine.run_tick()
        # Force them apart every tick so plan-movement can't make them meet.
        for aid, _n, pos in specs:
            engine.registry.get(aid).position = pos

    stats = GOVERNOR.stats()
    _cleanup()

    print("Budget stress test result:")
    print(f"  ticks run          : {N_TICKS}")
    print(f"  LLM calls (total)  : {stats['total_calls']}")
    print(f"  calls last hour    : {stats['calls_last_hour']}")
    print(f"  by kind            : {stats['by_kind']}")

    ok = stats["total_calls"] == 0
    print("\nPASS: quiet ticks made zero LLM calls." if ok
          else "\nFAIL: expected 0 LLM calls on quiet ticks.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
