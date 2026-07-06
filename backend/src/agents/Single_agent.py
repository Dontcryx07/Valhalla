"""
Main brain / command centre of a single agent.

Makes decisions, calls and delegates tasks to sub-modules (day_planner,
memory, reflection, etc.), and runs the agent's action loop.

Exports:
  create_agent_graph() -> CompiledGraph[AgentState]
    A single-agent LangGraph. Currently one node: generate_day_plan.
    Future: execute_tick, reflect, update_memory, conversation.

Usage as a library (for the multi-agent orchestrator):
  graph = create_agent_graph()
  result = graph.invoke({
      "persona_name": "parv",
      "persona": {...},
      "current_time": "2026-07-03 06:00",
  })

Usage from CLI:
  python Single_agent.py parv
"""

from __future__ import annotations

import sys
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[2]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.append(str(BACKEND_ROOT))


import json
import time
from types import SimpleNamespace
from typing import Optional, TypedDict

from langgraph.graph import StateGraph, START, END

from src.core.log import get_logger, setup_logging
setup_logging(run_id="single_agent", console=False)
from src.config import PERSONALITIES_DIR
from src.agents import day_planner


logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# State schemas
# ---------------------------------------------------------------------------

class AgentState(TypedDict, total=False):
    """State for a single agent's brain graph."""
    persona_name: str
    persona: dict
    current_time: str
    day_plan: list[dict]
    relevant_memories: list[str]
    yesterday_summary: Optional[str]
    error: Optional[str]


# ---------------------------------------------------------------------------
# Single-agent graph
# ---------------------------------------------------------------------------

def retrieve_memories(state: AgentState) -> dict:
    """Node: fetch yesterday's summary and relevant memories from Short_term."""
    from src.core.Short_term import (
        date_from_simulation_time,
        get_yesterday_summary,
        get_relevant_memories,
    )

    persona_name = state["persona_name"]
    sim_date = date_from_simulation_time(state.get("current_time", "00:00"))
    persona = state.get("persona", {})

    yesterday_summary = get_yesterday_summary(persona_name, sim_date)

    traits = persona.get("Traits", persona.get("traits", []))
    query = " ".join(traits) if traits else "daily life"
    memories = get_relevant_memories(persona_name, sim_date, query, k=5)

    if yesterday_summary:
        logger.info("[Single_agent] %s: loaded yesterday's summary", persona_name)
    if memories:
        logger.info("[Single_agent] %s: loaded %d relevant memories", persona_name, len(memories))

    return {
        "relevant_memories": memories,
        "yesterday_summary": yesterday_summary,
    }


def generate_day_plan(state: AgentState) -> dict:
    """Node: generate a day plan for this agent by calling day_planner.run()."""
    persona = state["persona"]
    persona_name = state["persona_name"]
    print(f'  Name: "{persona_name}", State: "Processing"')

    try:
        agent = SimpleNamespace(
            persona=persona,
            relevant_memories=state.get("relevant_memories", []),
            yesterday_summary=state.get("yesterday_summary"),
        )

        result = day_planner.run(agent, {
            "current_time": state.get("current_time", "2026-07-03 06:00"),
            "places": None,
            "persona_name": persona_name,
        })

        plan_count = len(result.get("day_plan", []))
        print(f'  Name: "{persona_name}", State: "Completed" — {plan_count} actions')

        return {
            "day_plan": result.get("day_plan", []),
            "error": result.get("error"),
        }
    except Exception as exc:
        print(f'  Name: "{persona_name}", State: "Failed"')
        logger.error("[Single_agent] %s: failed — %s", persona_name, exc)
        return {"day_plan": [], "error": str(exc)}


_agent_graph = None


def create_agent_graph():
    """Build and return the compiled single-agent brain graph (cached)."""
    global _agent_graph
    if _agent_graph is not None:
        return _agent_graph

    builder = StateGraph(AgentState)
    builder.add_node("retrieve_memories", retrieve_memories)
    builder.add_node("generate_day_plan", generate_day_plan)
    builder.add_edge(START, "retrieve_memories")
    builder.add_edge("retrieve_memories", "generate_day_plan")
    builder.add_edge("generate_day_plan", END)

    _agent_graph = builder.compile()
    return _agent_graph


# ---------------------------------------------------------------------------
# Plan table printer
# ---------------------------------------------------------------------------

def _print_day_plan(persona_name: str, day_plan: list[dict]) -> None:
    if not day_plan:
        return
    print(f'\n  Generated plan for "{persona_name}":')
    print(f'  {"Action":25s} {"Start":7s} {"End":7s} {"Location":25s} {"Area":20s}')
    print(f'  {"-"*25} {"-"*7} {"-"*7} {"-"*25} {"-"*20}')
    for a in day_plan:
        loc = (a.get("location_id") or "")[:25]
        area = (a.get("sub_area") or "")[:20]
        print(f'  {a.get("action", ""):25s} {a.get("start", ""):7s} {a.get("end", ""):7s} {loc:25s} {area:20s}')


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run day planner for a single persona")
    parser.add_argument(
        "persona",
        help="Persona name (e.g. parv) or path to a persona JSON file",
    )
    args = parser.parse_args()

    # Resolve path
    candidate = Path(args.persona)
    if candidate.exists():
        persona_path = candidate
    elif candidate.suffix == ".json":
        matches = sorted(PERSONALITIES_DIR.glob(f"**/{candidate.name}"))
        persona_path = matches[0] if matches else candidate
    else:
        matches = sorted(PERSONALITIES_DIR.glob(f"**/{args.persona}/{args.persona}.json"))
        if not matches:
            matches = sorted(PERSONALITIES_DIR.glob(f"**/{args.persona}.json"))
        if not matches:
            available = sorted({p.parent.name for p in PERSONALITIES_DIR.glob("**/*.json")})
            raise FileNotFoundError(
                f"Could not find persona '{args.persona}'. "
                f"Available: {', '.join(available)}"
            )
        persona_path = matches[0]

    persona_data = json.loads(persona_path.read_text())
    persona_name = persona_data.get("Name", persona_path.stem)

    t0 = time.perf_counter()
    print(f'Day planner — running "{persona_name}"\n')

    graph = create_agent_graph()
    result = graph.invoke({
        "persona_name": persona_name,
        "persona": persona_data,
        "current_time": "2026-07-03 06:00",
    })

    elapsed = time.perf_counter() - t0

    print(f"\n{'='*60}")
    if result.get("error"):
        print(f"  Failed: {result['error']}  |  Elapsed: {elapsed:.1f}s")
    else:
        print(f"  Completed  |  Elapsed: {elapsed:.1f}s")
    print(f"{'='*60}")

    day_plan = result.get("day_plan", [])
    if not result.get("error") and day_plan:
        _print_day_plan(persona_name, day_plan)
