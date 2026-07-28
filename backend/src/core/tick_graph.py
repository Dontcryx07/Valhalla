"""
Agent -- per-agent LangGraph subgraph + standalone CLI debug tool.

Production pipeline (used by WorldEngine via build_tick_graph):

    perceive -> retrieve_memories -> react --[replan]--> day_planner -> write_back_memory
                                              \\_[continue]__> keep_current /

Only agents where scheduler.py's `agents_ready_for_decision()` returns
True are invoked each tick -- mid-action agents are skipped entirely.

Standalone CLI debug mode (python Agent.py <persona>):

    retrieve_memories from Short_term -> call day_planner.run() -> print plan table

Useful for testing a single persona's day plan without standing up the
full tick loop.
"""

from __future__ import annotations

import sys
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[2]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.append(str(BACKEND_ROOT))


from datetime import date, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, TypedDict

from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel

from src.core.log import get_logger
from src.core.snapshot import WorldSnapshot
from src.core.world_state import CurrentAction
from src.core.perceive import perceive, Observation
from src.agents.react import decide_reaction, ReactionDecision

logger = get_logger(__name__)


# --------------------------------------------------------------------------- #
# Sim-tick <-> wallclock helpers
# --------------------------------------------------------------------------- #

MINUTES_PER_DAY = 24 * 60


def tick_to_wallclock(tick: int) -> Tuple[int, str]:
    """(sim day index, 'HH:MM') for a given absolute tick."""
    day_index, minute_of_day = divmod(tick, MINUTES_PER_DAY)
    hh, mm = divmod(minute_of_day, 60)
    return day_index, f"{hh:02d}:{mm:02d}"


def hhmm_to_minute(hhmm: str) -> int:
    hh, mm = hhmm.split(":")
    return int(hh) * 60 + int(mm)


# --------------------------------------------------------------------------- #
# Day-planner integration contract
# --------------------------------------------------------------------------- #


class DayPlannerRequest(BaseModel):
    agent_id: str
    persona: Dict[str, Any]
    relevant_memories: List[str]
    yesterday_summary: Optional[str]
    day_index: int
    current_hhmm: str
    interrupt_context: Optional[str] = None


class DayPlanEntry(BaseModel):
    action: str
    start: str
    end: str
    parent_activity: Optional[str] = None
    location_id: Optional[str] = None
    sub_area: Optional[str] = None


DayPlannerFn = Callable[[DayPlannerRequest], List[DayPlanEntry]]


def build_default_day_planner_fn(sim_start_date: Optional[date] = None) -> DayPlannerFn:
    epoch = sim_start_date or date.today()

    def _adapter(request: DayPlannerRequest) -> List[DayPlanEntry]:
        from src.agents.day_planner import run as day_planner_run

        calendar_date = epoch + timedelta(days=request.day_index)
        current_time_str = f"{calendar_date.isoformat()} {request.current_hhmm}"

        memories = list(request.relevant_memories)
        if request.interrupt_context:
            memories.append(f"IMPORTANT -- just happened, account for this: {request.interrupt_context}")

        class _AgentProxy:
            persona = request.persona
            relevant_memories = memories
            yesterday_summary = request.yesterday_summary

        result = day_planner_run(_AgentProxy(), {"current_time": current_time_str, "places": None})
        raw_plan = result.get("day_plan", [])

        entries = [DayPlanEntry.model_validate(item) for item in raw_plan]
        logger.info(
            "day_planner.run() produced %d entries for agent '%s', day %d%s",
            len(entries), request.agent_id, request.day_index,
            " (interrupt-triggered replan)" if request.interrupt_context else "",
        )
        return entries

    return _adapter


def _find_slot(plan: List[DayPlanEntry], hhmm: str) -> Optional[DayPlanEntry]:
    minute = hhmm_to_minute(hhmm)
    for entry in plan:
        if hhmm_to_minute(entry.start) <= minute < hhmm_to_minute(entry.end):
            return entry
    return None


# --------------------------------------------------------------------------- #
# Per-agent, per-sim-day plan cache
# --------------------------------------------------------------------------- #


class DayPlanStoreProtocol:
    def get(self, agent_id: str, day_index: int) -> Optional[List[DayPlanEntry]]: ...
    def set(self, agent_id: str, day_index: int, plan: List[DayPlanEntry]) -> None: ...
    def invalidate(self, agent_id: str, day_index: int) -> None: ...


class InMemoryDayPlanStore(DayPlanStoreProtocol):
    def __init__(self) -> None:
        self._plans: Dict[Tuple[str, int], List[DayPlanEntry]] = {}

    def get(self, agent_id: str, day_index: int) -> Optional[List[DayPlanEntry]]:
        return self._plans.get((agent_id, day_index))

    def set(self, agent_id: str, day_index: int, plan: List[DayPlanEntry]) -> None:
        self._plans[(agent_id, day_index)] = plan

    def invalidate(self, agent_id: str, day_index: int) -> None:
        self._plans.pop((agent_id, day_index), None)


# --------------------------------------------------------------------------- #
# Memory stream integration contract
# --------------------------------------------------------------------------- #


class MemoryStreamProtocol:
    def add_memory(self, agent_id: str, content: str, importance: Optional[int] = None) -> None: ...
    def retrieve_memories(self, agent_id: str, query: str, k: int = 5) -> List[str]: ...


class _NullMemoryStream:
    def add_memory(self, agent_id: str, content: str, importance: Optional[int] = None) -> None:
        logger.debug("[NullMemoryStream] would store for '%s': %s", agent_id, content)

    def retrieve_memories(self, agent_id: str, query: str, k: int = 5) -> List[str]:
        return []


# --------------------------------------------------------------------------- #
# Tick outcome / result
# --------------------------------------------------------------------------- #


class TickOutcome(str, Enum):
    KEEP_CURRENT = "keep_current"
    NEW_ACTION = "new_action"


class TickResult(BaseModel):
    agent_id: str
    tick: int
    outcome: TickOutcome
    action: Optional[CurrentAction] = None
    reaction: Optional[ReactionDecision] = None


# --------------------------------------------------------------------------- #
# Internal graph state
# --------------------------------------------------------------------------- #


class TickState(TypedDict, total=False):
    agent_id: str
    tick: int
    persona: Dict[str, Any]
    yesterday_summary: Optional[str]
    world_snapshot: WorldSnapshot
    current_action: Optional[CurrentAction]
    observations: List[Observation]
    memories: List[str]
    reaction: ReactionDecision
    result: TickResult


def build_initial_state(
    agent_id: str,
    persona: Dict[str, Any],
    yesterday_summary: Optional[str],
    world_snapshot: WorldSnapshot,
) -> TickState:
    current_action = world_snapshot.get_agent(agent_id).current_action
    return TickState(
        agent_id=agent_id,
        tick=world_snapshot.tick,
        persona=persona,
        yesterday_summary=yesterday_summary,
        world_snapshot=world_snapshot,
        current_action=current_action,
    )


# --------------------------------------------------------------------------- #
# Graph builder
# --------------------------------------------------------------------------- #


def build_tick_graph(
    memory_stream: Optional[MemoryStreamProtocol] = None,
    day_planner_fn: Optional[DayPlannerFn] = None,
    day_plan_store: Optional[DayPlanStoreProtocol] = None,
    sim_start_date: Optional[date] = None,
):
    memory = memory_stream or _NullMemoryStream()
    if memory_stream is None:
        logger.warning(
            "build_tick_graph() called with no memory_stream -- using a "
            "no-op stand-in. retrieve_memories() will always return []. "
            "Pass the real MemoryStream in once it's ready."
        )

    day_plans = day_plan_store or InMemoryDayPlanStore()
    plan_fn = day_planner_fn or build_default_day_planner_fn(sim_start_date)

    def perceive_node(state: TickState) -> Dict[str, Any]:
        observations = perceive(state["world_snapshot"], state["agent_id"])
        return {"observations": observations}

    def retrieve_memories_node(state: TickState) -> Dict[str, Any]:
        agent_id = state["agent_id"]
        observations = state.get("observations", [])
        if observations:
            query = "; ".join(o.description for o in observations)
        else:
            action = state.get("current_action")
            query = action.description if action else "what should I do next"
        memories = memory.retrieve_memories(agent_id, query=query, k=5)
        return {"memories": memories}

    def react_node(state: TickState) -> Dict[str, Any]:
        reaction = decide_reaction(
            agent_id=state["agent_id"],
            persona=state["persona"],
            current_action=state.get("current_action"),
            observations=state.get("observations", []),
            memories=state.get("memories", []),
            tick=state["tick"],
        )
        return {"reaction": reaction}

    def route_after_react(state: TickState) -> str:
        return "day_planner" if state["reaction"].should_replan else "keep_current"

    def day_planner_node(state: TickState) -> Dict[str, Any]:
        agent_id = state["agent_id"]
        tick = state["tick"]
        day_index, hhmm = tick_to_wallclock(tick)

        current_action = state.get("current_action")
        interrupted = current_action is not None and not current_action.is_finished(tick)

        cached_plan = day_plans.get(agent_id, day_index)

        if cached_plan is None or interrupted:
            interrupt_context = None
            if interrupted:
                obs_text = "; ".join(o.description for o in state.get("observations", []))
                interrupt_context = (
                    f"Was in the middle of '{current_action.description}' when this "
                    f"happened: {obs_text or '(unspecified event)'}."
                )
            request = DayPlannerRequest(
                agent_id=agent_id,
                persona=state["persona"],
                relevant_memories=state.get("memories", []),
                yesterday_summary=state.get("yesterday_summary"),
                day_index=day_index,
                current_hhmm=hhmm,
                interrupt_context=interrupt_context,
            )
            cached_plan = plan_fn(request)
            day_plans.set(agent_id, day_index, cached_plan)

        slot = _find_slot(cached_plan, hhmm)
        if slot is None:
            logger.warning(
                "No day-plan slot covers %s on day %d for agent '%s'; "
                "using a placeholder action and invalidating the cached plan.",
                hhmm, day_index, agent_id,
            )
            day_plans.invalidate(agent_id, day_index)
            slot = DayPlanEntry(action="idle (no plan slot found)", start=hhmm, end=hhmm)
            duration = 10
        else:
            duration = hhmm_to_minute(slot.end) - hhmm_to_minute(hhmm)
            duration = max(duration, 1)

        action = CurrentAction(
            description=slot.action,
            start_tick=tick,
            end_tick=tick + duration,
            target_location_id=slot.location_id,
        )
        result = TickResult(
            agent_id=agent_id,
            tick=tick,
            outcome=TickOutcome.NEW_ACTION,
            action=action,
            reaction=state.get("reaction"),
        )
        return {"result": result}

    def keep_current_node(state: TickState) -> Dict[str, Any]:
        result = TickResult(
            agent_id=state["agent_id"],
            tick=state["tick"],
            outcome=TickOutcome.KEEP_CURRENT,
            action=state.get("current_action"),
            reaction=state.get("reaction"),
        )
        return {"result": result}

    def write_back_memory_node(state: TickState) -> Dict[str, Any]:
        result: TickResult = state["result"]
        if result.outcome == TickOutcome.NEW_ACTION and result.action is not None:
            memory.add_memory(
                agent_id=state["agent_id"],
                content=f"Decided to: {result.action.description}",
                importance=None,
            )
        return {}

    graph = StateGraph(TickState)
    graph.add_node("perceive", perceive_node)
    graph.add_node("retrieve_memories", retrieve_memories_node)
    graph.add_node("react", react_node)
    graph.add_node("day_planner", day_planner_node)
    graph.add_node("keep_current", keep_current_node)
    graph.add_node("write_back_memory", write_back_memory_node)

    graph.add_edge(START, "perceive")
    graph.add_edge("perceive", "retrieve_memories")
    graph.add_edge("retrieve_memories", "react")
    graph.add_conditional_edges(
        "react",
        route_after_react,
        {"day_planner": "day_planner", "keep_current": "keep_current"},
    )
    graph.add_edge("day_planner", "write_back_memory")
    graph.add_edge("keep_current", "write_back_memory")
    graph.add_edge("write_back_memory", END)

    compiled = graph.compile()
    logger.info("Tick graph compiled.")
    return compiled


async def run_tick(
    tick_graph,
    agent_id: str,
    persona: Dict[str, Any],
    yesterday_summary: Optional[str],
    world_snapshot: WorldSnapshot,
) -> TickResult:
    initial_state = build_initial_state(agent_id, persona, yesterday_summary, world_snapshot)
    final_state = await tick_graph.ainvoke(initial_state)
    return final_state["result"]


# --------------------------------------------------------------------------- #
# Standalone day-plan CLI (debug tool)
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    import argparse
    import asyncio
    import json
    import time
    from types import SimpleNamespace

    from src.core.log import setup_logging
    setup_logging(run_id="agent_cli", console=False)
    from src.config import PERSONALITIES_DIR

    parser = argparse.ArgumentParser(description="Agent debug tool — day-plan or self-test")
    parser.add_argument(
        "persona", nargs="?",
        help="Persona name (e.g. parv) or path to a persona JSON file",
    )
    parser.add_argument(
        "--self-test", action="store_true",
        help="Run tick graph sanity check with a fake day planner (no Gemini)",
    )
    parser.add_argument(
        "--current-time", default="2026-07-03 06:00",
        help="Simulation time for the day planner (default: 2026-07-03 06:00)",
    )
    args = parser.parse_args()

    # --self-test mode: tick graph sanity check
    if args.self_test:
        from src.core.world_state import WorldState, Position
        from src.core.snapshot import take_snapshot

        def fake_day_planner(request: DayPlannerRequest) -> List[DayPlanEntry]:
            return [
                DayPlanEntry(action="sleeping", start="00:00", end="06:00"),
                DayPlanEntry(action="testing the tick graph", start="06:00", end="24:00"),
            ]

        async def _run_self_test() -> None:
            world = WorldState()
            world.register_agent("test_agent", Position(x=0, y=0, location_id="start"))
            world.advance_tick(minutes=6 * 60)
            snap = take_snapshot(world)
            graph = build_tick_graph(day_planner_fn=fake_day_planner)
            result = await run_tick(
                graph, "test_agent", persona={"name": "Test"}, yesterday_summary=None, world_snapshot=snap
            )
            assert result.outcome == TickOutcome.NEW_ACTION
            assert result.action is not None
            assert result.action.description == "testing the tick graph"
            print("Self-test passed:", result)

        asyncio.run(_run_self_test())
        sys.exit(0)

    # Day-plan mode: generate a full day plan for one persona
    if args.persona is None:
        parser.print_help()
        sys.exit(1)

    # Resolve persona path (same logic as Single_agent.py)
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
    current_time = args.current_time

    t0 = time.perf_counter()
    print(f'Day planner — running "{persona_name}" at {current_time}\n')

    # Retrieve memories from Short_term
    from src.agents.Short_term import date_from_simulation_time, get_yesterday_summary, get_relevant_memories

    sim_date = date_from_simulation_time(current_time)
    yesterday_summary = get_yesterday_summary(persona_name, sim_date)
    traits = persona_data.get("Traits", persona_data.get("traits", []))
    query = " ".join(traits) if traits else "daily life"
    memories = get_relevant_memories(persona_name, sim_date, query, k=5)

    if yesterday_summary:
        print(f"  Loaded yesterday's summary")
    if memories:
        print(f"  Loaded {len(memories)} relevant memories")

    # Generate day plan via day_planner.run()
    from src.agents.day_planner import run as day_planner_run

    agent = SimpleNamespace(
        persona=persona_data,
        relevant_memories=memories,
        yesterday_summary=yesterday_summary,
    )

    try:
        result = day_planner_run(agent, {
            "current_time": current_time,
            "places": None,
            "persona_name": persona_name,
        })
        plan = result.get("day_plan", [])
        error = result.get("error")

        elapsed = time.perf_counter() - t0

        print(f"\n{'='*60}")
        if error:
            print(f"  Failed: {error}  |  Elapsed: {elapsed:.1f}s")
        else:
            print(f"  Completed — {len(plan)} actions  |  Elapsed: {elapsed:.1f}s")
        print(f"{'='*60}")

        if not error and plan:
            print(f'\n  Generated plan for "{persona_name}":')
            print(f'  {"Action":25s} {"Start":7s} {"End":7s} {"Location":25s} {"Area":20s}')
            print(f'  {"-"*25} {"-"*7} {"-"*7} {"-"*25} {"-"*20}')
            for a in plan:
                loc = (a.get("location_id") or "")[:25]
                area = (a.get("sub_area") or "")[:20]
                print(f'  {a.get("action", ""):25s} {a.get("start", ""):7s} {a.get("end", ""):7s} {loc:25s} {area:20s}')

    except Exception as exc:
        elapsed = time.perf_counter() - t0
        print(f"\n  Failed after {elapsed:.1f}s: {exc}")
