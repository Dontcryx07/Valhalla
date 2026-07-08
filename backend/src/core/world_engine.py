"""
WorldEngine — the main simulation orchestrator.

Controls the tick loop: advances time, runs agent actions in parallel,
detects proximity for conversations, handles end-of-day transitions,
and keeps WorldState in sync with the agent registry.

Usage:
    from src.core.world_engine import WorldEngine

    engine = WorldEngine()
    await engine.initialize()
    await engine.run(max_ticks=1440)   # one full day at 1 tick/sec
"""

from __future__ import annotations

import asyncio
import json
import sys
import time as _time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from types import SimpleNamespace

from src.core.log import get_logger

BACKEND_ROOT = Path(__file__).resolve().parents[2]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.append(str(BACKEND_ROOT))


from src.agents.Actions import AgentActionManager, LocationResolver, ActionState
from src.agents.conversation import (
    generate_conversation,
    RelationshipMatrix,
    _is_action_blocked,
)
from src.agents.Short_term import (
    save_day_plan,
    load_day_plan,
    date_from_simulation_time,
    append_conversation,
    archive_to_long_term,
    get_yesterday_summary,
    get_relevant_memories,
)
from src.config import (
    PERSONALITIES_DIR,
    SIM_MINUTES_PER_TICK,
    REAL_SECONDS_PER_TICK,
    MAX_CONVERSATIONS_PER_AGENT,
)
from src.core.agent_registry import AgentRegistry, AgentRuntimeState
from src.core.checkpoint_manager import (
    save_checkpoint,
    save_history,
    load_checkpoint,
    list_checkpoints,
    prune_checkpoints,
    latest_tick,
)
from src.core.snapshot import take_snapshot
from src.core.world_state import WorldState, Position, CurrentAction, AgentStatus

logger = get_logger(__name__)


class WorldEngine:
    """Main simulation orchestrator."""

    def __init__(
        self,
        sim_start_date: str = "2026-07-03",
        sim_start_hhmm: str = "00:00",
        sim_end_hhmm: str = "24:00",
    ):
        self.sim_start_date = sim_start_date
        self.sim_start_hhmm = sim_start_hhmm
        self.sim_end_hhmm = sim_end_hhmm

        self.world = WorldState()
        self.registry = AgentRegistry()
        self.resolver = LocationResolver()
        self.relationship_matrix = RelationshipMatrix()

        self._day_index = 0

    # ------------------------------------------------------------------ #
    # Persona discovery
    # ------------------------------------------------------------------ #

    def _discover_personas(self) -> List[Dict[str, Any]]:
        """Find all persona JSON files and return their parsed data."""
        personas = []
        for json_path in sorted(PERSONALITIES_DIR.glob("**/*.json")):
            try:
                data = json.loads(json_path.read_text(encoding="utf-8"))
                if "Name" in data and "Hostel" in data:
                    personas.append(data)
            except Exception as e:
                logger.warning("[WorldEngine] failed to load %s: %s", json_path, e)
        logger.info(
            "[WorldEngine] discovered %d personas from %s", len(personas), PERSONALITIES_DIR
        )
        return personas

    def _resolve_hostel_position(self, hostel_id: str) -> Position:
        """Convert a Hostel field value to a pixel Position."""
        pos = self.resolver.resolve(hostel_id)
        if pos is None:
            logger.warning(
                "[WorldEngine] could not resolve hostel '%s', using default (0,0)", hostel_id
            )
            pos = Position(x=0, y=0, location_id=hostel_id)
        return pos

    def _persona_name_to_id(self, name: str) -> str:
        """Convert a display name like 'Parv Singla' to a safe agent_id."""
        import re
        safe = re.sub(r"[^A-Za-z0-9._-]+", "_", name.strip()).strip("._-").lower()
        return safe or "unknown"

    # ------------------------------------------------------------------ #
    # Initialization
    # ------------------------------------------------------------------ #

    async def initialize(self) -> None:
        """
        Phase 1: Discover personas, generate day plans, register agents.
        Does NOT advance the clock — all agents are ready at tick 0.
        """
        logger.info("[WorldEngine] ========== INITIALIZATION ==========")
        t0 = _time.perf_counter()

        personas = self._discover_personas()
        if not personas:
            logger.error("[WorldEngine] no personas found — aborting")
            return

        # Phase 1a: Generate day plans for every persona
        logger.info("[WorldEngine] generating day plans for %d agents ...", len(personas))
        plan_t0 = _time.perf_counter()

        for persona in personas:
            name = persona.get("Name", "unknown")
            hostel = persona.get("Hostel", "")
            agent_id = self._persona_name_to_id(name)
            position = self._resolve_hostel_position(hostel)

            # Register agent in WorldState
            self.world.register_agent(agent_id, position)

            # Generate full-day plan
            from src.agents.day_planner import run as day_planner_run

            agent_proxy = SimpleNamespace(
                persona=persona,
                relevant_memories=[],
                yesterday_summary=None,
            )
            plan_result = day_planner_run(
                agent_proxy,
                {
                    "current_time": f"{self.sim_start_date} {self.sim_start_hhmm}",
                    "places": None,
                    "persona_name": name,
                    "mode": "full_day",
                    "current_location_id": hostel,
                },
            )
            day_plan = plan_result.get("day_plan", [])
            error = plan_result.get("error")
            if error:
                logger.warning(
                    "[WorldEngine] day plan for '%s' has issues: %s", name, error
                )

            # Create agent action manager
            manager = AgentActionManager(agent_id, day_plan, position, self.resolver)

            # Store in registry
            self.registry.register(
                AgentRuntimeState(
                    agent_id=agent_id,
                    persona=persona,
                    persona_name=name,
                    manager=manager,
                    position=position,
                    day_plan=day_plan,
                )
            )

            logger.info(
                "[WorldEngine]   registered '%s' (hostel=%s) — %d plan actions",
                name, hostel, len(day_plan),
            )

        plan_elapsed = _time.perf_counter() - plan_t0
        logger.info(
            "[WorldEngine] all day plans generated in %.1fs", plan_elapsed
        )

        # Register hostel rooms as world resources
        for state in self.registry.all_states():
            hostel = state.persona.get("Hostel", "")
            if hostel:
                self.world.register_resource(hostel)

        elapsed = _time.perf_counter() - t0
        logger.info(
            "[WorldEngine] initialized %d agents in %.1fs — starting simulation",
            len(self.registry), elapsed,
        )

    # ------------------------------------------------------------------ #
    # Tick loop
    # ------------------------------------------------------------------ #

    async def run_tick(self) -> Dict[str, Any]:
        """
        Execute one simulation tick.

        Returns a dict of agent_id -> ActionState (serialized) for frontend use.
        """
        current_tick = self.world.tick
        hhmm = self._minutes_to_hhmm(current_tick % (24 * 60))

        # Detect end-of-day (tick >= 24*60)
        if current_tick >= 24 * 60:
            logger.info("[WorldEngine] day %d complete — ending simulation", self._day_index)
            return {}

        # ----- Step 1: Run agent actions in parallel (including paused/conversation) -----
        agent_states = self.registry.all_states()

        if agent_states:
            tick_results = await asyncio.gather(
                *[self._run_agent_tick(s, current_tick, hhmm) for s in agent_states],
                return_exceptions=True,
            )
            for state, result in zip(agent_states, tick_results):
                if isinstance(result, Exception):
                    logger.error(
                        "[WorldEngine] agent '%s' tick failed: %s", state.agent_id, result,
                    )

        # ----- Step 2: Check for conversation proximity -----
        await self._check_conversations(current_tick, hhmm)

        # ----- Step 3: Check for end-of-day triggers -----
        await self._check_last_action_triggers(current_tick, hhmm)

        # ----- Step 4: Sync registry to WorldState -----
        self._sync_to_world_state()

        # ----- Step 5: Advance tick -----
        self.world.advance_tick(minutes=SIM_MINUTES_PER_TICK)

        # ----- Step 6: Save checkpoint -----
        save_checkpoint(self.world, self.registry, self.world.tick)
        prune_checkpoints(keep_last=10)

        # Build frontend snapshot
        snapshot = {
            s.agent_id: {
                "position": {
                    "x": s.position.x,
                    "y": s.position.y,
                    "location_id": s.position.location_id,
                },
                "current_action": (
                    s.manager.current_action.model_dump()
                    if s.manager and s.manager.current_action
                    else None
                ),
                "paused": s.paused,
                "in_last_action": s.manager.is_last_action if s.manager else False,
            }
            for s in agent_states
        }
        return snapshot

    async def _run_agent_tick(
        self, state: AgentRuntimeState, tick: int, hhmm: str
    ) -> None:
        """Run one tick of action execution for a single agent.
        Paused agents (in conversation) are skipped — they resume via
        the background conversation pipeline."""
        if state.paused:
            return
        manager = state.manager
        if manager is None:
            return

        action = manager.tick(tick)

        # Update registry from manager
        state.position = manager.position
        state.current_action = action.model_dump() if action else None

    async def _check_conversations(self, tick: int, hhmm: str) -> None:
        """
        Detect pairs of agents at the same location and fire a background
        LLM task for the conversation. The agents are immediately set to
        conversation mode (paused) and stay frozen until the background
        task completes, at which point they get a remaining-day plan and
        are unpaused.
        """
        location_groups: Dict[str, List[AgentRuntimeState]] = {}
        for s in self.registry.all_states():
            if s.paused:
                continue
            if s.manager and s.manager.is_last_action:
                continue
            loc = s.position.location_id
            if loc:
                location_groups.setdefault(loc, []).append(s)

        for loc_id, agents in location_groups.items():
            if len(agents) < 2:
                continue
            if len(agents) > 2:
                logger.debug(
                    "[WorldEngine] %d agents at '%s' — skipping conversations (>2 rule)",
                    len(agents), loc_id,
                )
                continue

            a, b = agents[0], agents[1]

            # Cooldown check
            if not self.world.can_converse(a.agent_id, b.agent_id):
                continue

            # Max conversations cap
            if a.conversation_count >= MAX_CONVERSATIONS_PER_AGENT:
                continue
            if b.conversation_count >= MAX_CONVERSATIONS_PER_AGENT:
                continue

            # Check both agents are in compatible actions (not sleeping)
            if a.manager and a.manager.current_action:
                ca_a = CurrentAction(
                    description=a.manager.current_action.description,
                    start_tick=tick,
                    end_tick=tick + 10,
                )
                if _is_action_blocked(ca_a):
                    continue
            if b.manager and b.manager.current_action:
                ca_b = CurrentAction(
                    description=b.manager.current_action.description,
                    start_tick=tick,
                    end_tick=tick + 10,
                )
                if _is_action_blocked(ca_b):
                    continue

            # Fire background conversation pipeline
            logger.info(
                "[WorldEngine] triggering conversation: '%s' <-> '%s' at %s",
                a.persona_name, b.persona_name, loc_id,
            )

            asyncio.create_task(
                self._run_conversation_pipeline(a, b, loc_id, tick, hhmm)
            )

            # Immediately set agents to conversation mode (frozen until bg task completes)
            if a.manager:
                a.manager.set_conversation_action(b.persona_name)
                a.paused = True
                a.conversation_count += 1
            if b.manager:
                b.manager.set_conversation_action(a.persona_name)
                b.paused = True
                b.conversation_count += 1

            self.world.record_conversation(a.agent_id)
            self.world.record_conversation(b.agent_id)

            logger.info(
                "[WorldEngine] conversation '%s' <-> '%s' started (async, waiting for LLM)",
                a.persona_name, b.persona_name,
            )

    async def _run_conversation_pipeline(
        self,
        a: AgentRuntimeState,
        b: AgentRuntimeState,
        loc_id: str,
        tick: int,
        hhmm: str,
    ) -> None:
        """
        Background task: generates the conversation via LLM, saves results,
        and resumes both agents with new remaining-day plans.
        Runs in a thread pool so it doesn't block the event loop.
        """
        loop = asyncio.get_event_loop()

        a_action = CurrentAction(
            description=a.manager.current_action.description if a.manager and a.manager.current_action else "unknown",
            start_tick=tick,
            end_tick=tick + 10,
            target_location_id=loc_id,
        )
        b_action = CurrentAction(
            description=b.manager.current_action.description if b.manager and b.manager.current_action else "unknown",
            start_tick=tick,
            end_tick=tick + 10,
            target_location_id=loc_id,
        )
        rel_a_to_b = self.relationship_matrix.get(a.agent_id, b.agent_id)
        rel_b_to_a = self.relationship_matrix.get(b.agent_id, a.agent_id)

        # Step 1: Generate conversation (blocking LLM call → thread pool)
        conv_result = await loop.run_in_executor(
            None,
            generate_conversation,
            a.agent_id, b.agent_id,
            a.persona, b.persona,
            a.day_plan, b.day_plan,
            a_action, b_action,
            rel_a_to_b, rel_b_to_a,
            loc_id, hhmm,
        )

        if conv_result is None:
            logger.warning(
                "[WorldEngine] conversation LLM failed for '%s' <-> '%s' — unpausing",
                a.persona_name, b.persona_name,
            )
            a.paused = False
            b.paused = False
            if a.manager:
                a.manager._conversation_mode = False
            if b.manager:
                b.manager._conversation_mode = False
            return

        # Step 2: Save conversation to Short_term
        date_str = date_from_simulation_time(f"{self.sim_start_date} {hhmm}")
        conv_entry = {
            "participants": [a.persona_name, b.persona_name],
            "messages": [
                {"speaker": m.speaker, "text": m.text}
                for m in conv_result.messages
            ],
            "summary": conv_result.summary,
        }
        append_conversation(a.persona_name, date_str, conv_entry)
        append_conversation(b.persona_name, date_str, conv_entry)

        # Step 3: Update relationship matrix
        self.relationship_matrix.update(a.agent_id, b.agent_id, conv_result.relationship_delta)
        self.relationship_matrix.update(b.agent_id, a.agent_id, conv_result.relationship_delta)
        self.relationship_matrix.save()

        # Step 4: Generate remaining-day plans (blocking → thread pool)
        from src.agents.day_planner import run as day_planner_run

        def _run_planner(state: AgentRuntimeState) -> list:
            proxy = SimpleNamespace(
                persona=state.persona,
                relevant_memories=[],
                yesterday_summary=None,
            )
            plan_result = day_planner_run(
                proxy,
                {
                    "current_time": f"{self.sim_start_date} {hhmm}",
                    "places": None,
                    "persona_name": state.persona_name,
                    "mode": "remaining",
                    "current_location_id": state.position.location_id,
                },
            )
            return plan_result.get("day_plan", [])

        for state in (a, b):
            try:
                new_plan = await loop.run_in_executor(None, _run_planner, state)
                if new_plan and state.manager:
                    state.manager.resume_from_conversation(new_plan)
                    state.day_plan = new_plan
                    logger.debug(
                        "[WorldEngine] remaining-day plan for '%s': %d actions",
                        state.persona_name, len(new_plan),
                    )
            except Exception as e:
                logger.error(
                    "[WorldEngine] remaining-day plan failed for '%s': %s",
                    state.persona_name, e,
                )

        # Step 5: Unpause agents
        a.paused = False
        b.paused = False

        logger.info(
            "[WorldEngine] conversation '%s' <-> '%s' complete — agents resumed",
            a.persona_name, b.persona_name,
        )

    async def _check_last_action_triggers(self, tick: int, hhmm: str) -> None:
        """
        When an agent enters their last action of the day plan:
        1. Archive the day to Long_term_db
        2. Generate the next day's plan
        3. Load the new plan into the agent's manager
        """
        for state in self.registry.all_states():
            manager = state.manager
            if manager is None:
                continue
            if not manager._entered_last_action:
                continue
            # Already triggered? Use a flag on the registry state
            if state.day_archived:
                continue

            logger.info(
                "[WorldEngine] '%s' entered last action — archiving day, generating next plan",
                state.persona_name,
            )

            # Archive current day
            date_str = date_from_simulation_time(f"{self.sim_start_date} {hhmm}")
            try:
                archive_result = await archive_to_long_term(state.persona_name, date_str)
                logger.info(
                    "[WorldEngine]   archive: %s", archive_result.get("summary", "")[:80],
                )
            except Exception as e:
                logger.error("[WorldEngine]   archive failed for '%s': %s", state.persona_name, e)

            # Generate next day's plan
            next_date = (datetime.fromisoformat(self.sim_start_date) + timedelta(days=1)).isoformat()[:10]
            hostel = state.persona.get("Hostel", "")

            from src.agents.day_planner import run as day_planner_run

            next_day_proxy = SimpleNamespace(
                persona=state.persona,
                relevant_memories=[],
                yesterday_summary=None,
            )

            try:
                plan_result = day_planner_run(
                    next_day_proxy,
                    {
                        "current_time": f"{next_date} 00:00",
                        "places": None,
                        "persona_name": state.persona_name,
                        "mode": "next_day",
                        "current_location_id": hostel,
                    },
                )
                new_plan = plan_result.get("day_plan", [])
                if new_plan:
                    state.day_plan = new_plan
                    manager.replace_day_plan(new_plan)
                    logger.info(
                        "[WorldEngine]   next day plan: %d actions for %s",
                        len(new_plan), next_date,
                    )
                else:
                    logger.warning(
                        "[WorldEngine]   no next-day plan generated for '%s'",
                        state.persona_name,
                    )
            except Exception as e:
                logger.error(
                    "[WorldEngine]   next-day planning failed for '%s': %s",
                    state.persona_name, e,
                )

            state.day_archived = True

    def _sync_to_world_state(self) -> None:
        """Mirror the agent registry into WorldState for frontend queries."""
        for state in self.registry.all_states():
            agent_id = state.agent_id
            try:
                ws_agent = self.world.get_agent(agent_id)
            except KeyError:
                continue

            # Update position
            if state.position != ws_agent.position:
                self.world.move_agent(agent_id, state.position)

            # Update current action
            if state.manager and state.manager.current_action:
                action = state.manager.current_action
                ca = CurrentAction(
                    description=action.description,
                    start_tick=self.world.tick,
                    end_tick=self.world.tick + SIM_MINUTES_PER_TICK,
                    target_location_id=action.location_id or state.position.location_id,
                )
                self.world.set_agent_action(agent_id, ca)
            else:
                self.world.clear_agent_action(agent_id)

    # ------------------------------------------------------------------ #
    # Main run loop
    # ------------------------------------------------------------------ #

    async def run(self, max_tick: int = 1440) -> None:
        """Run simulation until world.tick reaches max_tick (absolute)."""
        logger.info("[WorldEngine] ========== STARTING SIMULATION ==========")
        logger.info(
            "[WorldEngine] %d agents, max tick %d (current=%d), %.1f real sec/tick",
            len(self.registry), max_tick, self.world.tick, REAL_SECONDS_PER_TICK,
        )

        while self.world.tick < max_tick:
            result = await self.run_tick()
            if not result:
                logger.info("[WorldEngine] simulation complete at tick %d", self.world.tick)
                break

            # Log periodic status
            elapsed = self.world.tick
            if elapsed % 60 == 0:
                active = sum(
                    1 for s in self.registry.all_states()
                    if s.current_action is not None
                )
                paused = sum(1 for s in self.registry.all_states() if s.paused)
                hhmm = self._minutes_to_hhmm(self.world.tick % (24 * 60))
                logger.info(
                    "[WorldEngine] tick=%d time=%s agents=%d active=%d paused=%d",
                    elapsed, hhmm, len(self.registry), active, paused,
                )

            # Real-time pacing (1 sim-minute = 1 real second)
            await asyncio.sleep(REAL_SECONDS_PER_TICK)

        logger.info("[WorldEngine] ========== SIMULATION ENDED ==========")

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _minutes_to_hhmm(minutes: int) -> str:
        h, m = divmod(minutes, 60)
        return f"{h:02d}:{m:02d}"

    @staticmethod
    def _hhmm_to_minutes(hhmm: str) -> int:
        h, m = hhmm.split(":")
        return int(h) * 60 + int(m)


# ------------------------------------------------------------------ #
# Standalone CLI entry point
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    import argparse

    from src.core.log import setup_logging
    setup_logging(run_id="world_engine", console=True)

    parser = argparse.ArgumentParser(description="WorldEngine — multi-agent simulation")
    parser.add_argument(
        "--days", type=int, default=1,
        help="Number of simulation days to run (default: 1)",
    )
    parser.add_argument(
        "--start-date", default="2026-07-03",
        help="Simulation start date (default: 2026-07-03)",
    )
    parser.add_argument(
        "--start-time", default="00:00",
        help="Simulation start time HH:MM (default: 00:00)",
    )
    parser.add_argument(
        "--tick-speed", type=float, default=None,
        help="Override real seconds per tick (default: from config)",
    )
    parser.add_argument(
        "--resume", nargs="?", const=None, default=False,
        help="Resume from checkpoint. Pass a tick number, or no arg to prompt.",
    )
    args = parser.parse_args()

    if args.tick_speed is not None:
        global REAL_SECONDS_PER_TICK
        REAL_SECONDS_PER_TICK = args.tick_speed

    async def _main() -> None:
        if args.resume is not False:
            ticks = list_checkpoints()
            if not ticks:
                logger.error("[WorldEngine] no checkpoints found — cannot resume")
                return

            if args.resume is None:
                # --resume with no value → prompt
                print(f"\nAvailable checkpoints ({len(ticks)} total, showing last 20):")
                for t in ticks[-20:]:
                    print(f"  {t:>5}")
                print()

                while True:
                    tick_str = input("Resume from which tick? (or 'new' to start fresh): ").strip()
                    if tick_str.lower() == "new":
                        break
                    try:
                        resume_tick = int(tick_str)
                        if resume_tick in ticks:
                            break
                        print(f"Tick {resume_tick} not found. Try again.")
                    except ValueError:
                        print("Invalid input. Enter a tick number or 'new'.")

                if tick_str.lower() == "new":
                    pass  # fall through to fresh start
                else:
                    engine = WorldEngine(
                        sim_start_date=args.start_date,
                        sim_start_hhmm=args.start_time,
                    )
                    world, registry = load_checkpoint(resume_tick, engine.resolver)
                    engine.world = world
                    engine.registry = registry
                    logger.info(
                        "[WorldEngine] resumed from tick %d — %d agents",
                        resume_tick, len(registry),
                    )
                    await engine.run(max_tick=1440)
                    save_history(engine.world, args.start_date)
                    return
            else:
                # --resume <tick>
                resume_tick = int(args.resume)
                if resume_tick not in ticks:
                    logger.error("[WorldEngine] tick %d not found in checkpoints", resume_tick)
                    return
                engine = WorldEngine(
                    sim_start_date=args.start_date,
                    sim_start_hhmm=args.start_time,
                )
                world, registry = load_checkpoint(resume_tick, engine.resolver)
                engine.world = world
                engine.registry = registry
                logger.info(
                    "[WorldEngine] resumed from tick %d — %d agents",
                    resume_tick, len(registry),
                )
                await engine.run(max_tick=1440)
                save_history(engine.world, args.start_date)
                return

        # Fresh start
        engine = WorldEngine(
            sim_start_date=args.start_date,
            sim_start_hhmm=args.start_time,
        )
        await engine.initialize()
        for day in range(args.days):
            logger.info("[WorldEngine] --- Day %d ---", day + 1)
            await engine.run(max_tick=1440)

            # Save per-day action history for replay/analytics
            date_str = (
                datetime.fromisoformat(args.start_date) + timedelta(days=day)
            ).strftime("%Y-%m-%d")
            save_history(engine.world, date_str)

            if day < args.days - 1:
                engine.world = WorldState()
                engine.registry = AgentRegistry()
                await engine.initialize()

    asyncio.run(_main())
