"""
Tick graph -- the per-agent LangGraph subgraph invoked once per tick for
every agent scheduler.py's `agents_ready_for_decision()` returns.

Pipeline:

    perceive -> retrieve_memories -> react --[replan]--> day_planner -> write_back_memory -> END
                                        \\_____[continue]___> keep_current -> write_back_memory -> END

- perceive:          core/snapshot.py + agents/perceive.py, no LLM call
- retrieve_memories: delegates to a `MemoryStreamProtocol` implementation
- react:             agents/react.py -- cheap path most of the time, one LLM
                     call only when mid-action + something new perceived
- day_planner:       wired directly to your real `agents/day_planner.py`
                     (see "Day-planner integration" below)
- write_back_memory: records the decision via the memory stream (no-op
                     until the real module lands)

The compiled graph's output is a `TickResult`, which is what
engine/world_engine.py's resolve phase should read and apply to the real
`WorldState`. This module never mutates `WorldState` directly.

Day-planner integration
-------------------------
`day_planner.py` is a FULL-DAY planner: `run(agent, world_state)`
returns a `day_plan` covering 00:00 -> 24:00 of one calendar day as a list
of fine actions, not a single "next action." That changes what "replan"
means at the tick-graph level:

  - The graph does NOT call day_planner.run() every time an action ends.
    Once a day's plan exists, finishing an action just means "look up the
    next slot in the plan already on file" -- a dict lookup, zero LLM calls.
  - day_planner.run() is only actually invoked when:
      1. no plan exists yet for the current sim day (first tick of a new
         day), or
      2. react.py's LLM decided an in-progress action should be interrupted
         (a genuinely new circumstance the existing plan didn't account
         for) -- in which case the whole day is regenerated, with a note
         about what triggered the interruption folded into the memories
         fed to day_planner.run(), since it always plans from 00:00 rather
         than "from now."
    That second case is a real limitation worth knowing: because
    day_planner.py always re-derives the *entire* day from scratch, a
    replan triggered at, say, 15:00 isn't guaranteed to keep everything
    before 15:00 identical to what already happened -- it's a fresh plan,
    just one nudged by an extra note about the interruption. Revisit this
    once day_planner.py supports "plan the remainder of the day" if that
    mismatch turns out to matter in practice.
  - Per-agent, per-sim-day plans are cached via `DayPlanStoreProtocol`
    (in-memory default provided) so agents don't recompute a plan they
    already have.
  - `current_time` for day_planner.run() needs an actual calendar date (its
    prompts reference "yesterday," day-of-week, etc.) even though the sim
    only tracks an integer tick. `build_tick_graph(sim_start_date=...)`
    anchors day_index=0 to a real date so that string can be built; pass
    your simulation's actual start date, otherwise it defaults to today.

Build the graph ONCE at startup (main.py or WorldEngine.__init__), not once
per tick -- it's stateless between `.ainvoke()` calls.
"""

from __future__ import annotations

from datetime import date, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, TypedDict

from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel

from src.core.log import get_logger
from src.core.snapshot import WorldSnapshot
from src.core.world_state import CurrentAction
from src.agents.perceive import perceive, Observation
from src.agents.react import decide_reaction, ReactionDecision

logger = get_logger(__name__)


# Sim-tick <-> wallclock helpers
#
# The world engine's tick is a running integer count of simulated minutes
# (see WorldState.advance_tick). day_planner.py thinks in HH:MM within a
# single calendar day. These two conversions are the only place that
# mapping happens. (1 tick -> 1 minute)

MINUTES_PER_DAY = 24 * 60


def tick_to_wallclock(tick: int) -> Tuple[int, str]:
    """(sim day index, 'HH:MM') for a given absolute tick."""
    day_index, minute_of_day = divmod(tick, MINUTES_PER_DAY)
    hh, mm = divmod(minute_of_day, 60)
    return day_index, f"{hh:02d}:{mm:02d}"


def hhmm_to_minute(hhmm: str) -> int:
    hh, mm = hhmm.split(":")
    return int(hh) * 60 + int(mm)


# Day-planner integration contract

class DayPlannerRequest(BaseModel):
    agent_id: str
    persona: Dict[str, Any]
    relevant_memories: List[str]
    yesterday_summary: Optional[str]
    day_index: int
    current_hhmm: str
    # Set only when this is a forced midday replan triggered by react.py's
    # LLM deciding an observation was worth interrupting for.
    interrupt_context: Optional[str] = None


class DayPlanEntry(BaseModel):
    """Mirrors the dict shape day_planner.py's `run()` puts in `day_plan`
    (FineAction / atomic-location-assignment fields, whichever produced it)."""
    action: str
    start: str
    end: str
    parent_activity: Optional[str] = None
    location_id: Optional[str] = None
    sub_area: Optional[str] = None


DayPlannerFn = Callable[[DayPlannerRequest], List[DayPlanEntry]]


def build_default_day_planner_fn(sim_start_date: Optional[date] = None) -> DayPlannerFn:
    """
    The real bridge to `src.agents.day_planner.run()`. Imported lazily
    inside the returned closure so importing tick_graph.py doesn't require
    google-genai/config/API keys to be set up unless a day plan is
    actually being generated.

    `sim_start_date` anchors day_index=0 to a real calendar date, since
    day_planner.py's prompts want an actual date/day-of-week even though
    the sim only tracks an integer tick count. Defaults to today if you
    don't have a fixed sim epoch -- pass your real one if you have it.
    """
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
    """The plan entry covering `hhmm`, i.e. start <= hhmm < end."""
    minute = hhmm_to_minute(hhmm)
    for entry in plan:
        if hhmm_to_minute(entry.start) <= minute < hhmm_to_minute(entry.end):
            return entry
    return None


# Per-agent, per-sim-day plan cache

class DayPlanStoreProtocol:
    def get(self, agent_id: str, day_index: int) -> Optional[List[DayPlanEntry]]: ...

    def set(self, agent_id: str, day_index: int, plan: List[DayPlanEntry]) -> None: ...

    def invalidate(self, agent_id: str, day_index: int) -> None: ...


class InMemoryDayPlanStore(DayPlanStoreProtocol):
    """Fine for a single-process sim. Swap for a persistence-backed store
    once Phase 4 (replay log / SQLite) is wired up, if you want plans to
    survive a process restart."""

    def __init__(self) -> None:
        self._plans: Dict[Tuple[str, int], List[DayPlanEntry]] = {}

    def get(self, agent_id: str, day_index: int) -> Optional[List[DayPlanEntry]]:
        return self._plans.get((agent_id, day_index))

    def set(self, agent_id: str, day_index: int, plan: List[DayPlanEntry]) -> None:
        self._plans[(agent_id, day_index)] = plan

    def invalidate(self, agent_id: str, day_index: int) -> None:
        self._plans.pop((agent_id, day_index), None)


# --------------------------------------------------------------------------- #
# Memory stream integration contract -- matches the stub interface you and
# your teammate agreed on.
# --------------------------------------------------------------------------- #

class MemoryStreamProtocol:
    """Structural contract only -- your teammate's real class satisfies this
    without needing to inherit from it; this exists purely for readability
    at the call sites in this file."""

    def add_memory(self, agent_id: str, content: str, importance: Optional[int] = None) -> None: ...

    def retrieve_memories(self, agent_id: str, query: str, k: int = 5) -> List[str]: ...


class _NullMemoryStream:
    """Used automatically if no memory_stream is supplied, so this graph is
    runnable standalone before the memory module lands. Logs loudly rather
    than silently pretending to remember things."""

    def add_memory(self, agent_id: str, content: str, importance: Optional[int] = None) -> None:
        logger.debug("[NullMemoryStream] would store for '%s': %s", agent_id, content)

    def retrieve_memories(self, agent_id: str, query: str, k: int = 5) -> List[str]:
        return []


# Tick outcome / result -- what this graph hands back to world_engine.py

class TickOutcome(str, Enum):
    KEEP_CURRENT = "keep_current"   # no change; agent's existing action/status stands
    NEW_ACTION = "new_action"       # resolver should call world.set_agent_action(...)


class TickResult(BaseModel):
    agent_id: str
    tick: int
    outcome: TickOutcome
    action: Optional[CurrentAction] = None
    reaction: Optional[ReactionDecision] = None


# --------------------------------------------------------------------------- #
# Internal graph state -- scratch state, never leaves this module. TypedDict
# per your state-design split (cross-tick-boundary state is Pydantic;
# subgraph-local scratch state is TypedDict).
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
    """Convenience builder for the state fed into `tick_graph.ainvoke(...)`."""
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
    """
    Compile the per-agent tick subgraph. Call once at startup; reuse the
    compiled graph for every agent, every tick.

    `day_planner_fn` defaults to the real bridge to `agents/day_planner.py`
    (`build_default_day_planner_fn`). Override it (e.g. with a fake that
    returns canned `DayPlanEntry` lists) for tests that shouldn't hit
    Gemini -- see the `__main__` block below for an example.
    """
    memory = memory_stream or _NullMemoryStream()
    if memory_stream is None:
        logger.warning(
            "build_tick_graph() called with no memory_stream -- using a "
            "no-op stand-in. retrieve_memories() will always return []. "
            "Pass the real MemoryStream in once it's ready."
        )

    day_plans = day_plan_store or InMemoryDayPlanStore()
    plan_fn = day_planner_fn or build_default_day_planner_fn(sim_start_date)

    # ------------------------------------------------------------------ #
    # Nodes
    # ------------------------------------------------------------------ #

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
        # If we got routed here with a current_action that hasn't actually
        # finished yet, react.py's LLM decided to interrupt it -- that's the
        # one case that forces a full-day regeneration rather than a cache
        # lookup for the next scheduled slot.
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
            # Plan doesn't cover this minute (malformed plan, or day boundary
            # edge case). Don't crash the tick -- log it, drop the bad cache
            # entry so the next tick forces a clean regeneration, and fall
            # back to a short safe placeholder for this one tick only.
            logger.warning(
                "No day-plan slot covers %s on day %d for agent '%s'; "
                "using a placeholder action and invalidating the cached plan.",
                hhmm, day_index, agent_id,
            )
            day_plans.invalidate(agent_id, day_index)
            slot = DayPlanEntry(action="idle (no plan slot found)", start=hhmm, end=hhmm)
            duration = 10  # minutes; matches the default tick granularity
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

    # ------------------------------------------------------------------ #
    # Wiring
    # ------------------------------------------------------------------ #

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
    """
    Convenience wrapper: builds the initial state, invokes the graph, and
    returns a `TickResult` directly. This is what world_engine.py's decide
    phase should call inside `asyncio.gather(...)` for each ready agent:

        results = await asyncio.gather(*[
            run_tick(tick_graph, aid, personas[aid], summaries[aid], snap)
            for aid in ready_agent_ids
        ])
    """
    initial_state = build_initial_state(agent_id, persona, yesterday_summary, world_snapshot)
    final_state = await tick_graph.ainvoke(initial_state)
    return final_state["result"]


# --------------------------------------------------------------------------- #
# Standalone sanity check -- uses a fake day_planner_fn so it doesn't hit
# Gemini. Requires `langgraph` installed; not executed in this environment.
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    import asyncio

    from src.core.world_state import WorldState, Position
    from src.core.snapshot import take_snapshot

    def fake_day_planner(request: DayPlannerRequest) -> List[DayPlanEntry]:
        # A trivial two-block "day" so _find_slot has something to match.
        return [
            DayPlanEntry(action="sleeping", start="00:00", end="06:00"),
            DayPlanEntry(action="testing the tick graph", start="06:00", end="24:00"),
        ]

    async def _demo() -> None:
        world = WorldState()
        world.register_agent("test_agent", Position(x=0, y=0, location_id="start"))
        world.advance_tick(minutes=6 * 60)  # jump to 06:00 so we land in the second block
        snap = take_snapshot(world)

        graph = build_tick_graph(day_planner_fn=fake_day_planner)
        result = await run_tick(
            graph, "test_agent", persona={"name": "Test"}, yesterday_summary=None, world_snapshot=snap
        )
        assert result.outcome == TickOutcome.NEW_ACTION
        assert result.action is not None
        assert result.action.description == "testing the tick graph"
        print("tick_graph.py sanity check passed:", result)

    asyncio.run(_demo())