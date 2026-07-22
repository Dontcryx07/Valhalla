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
import random
import sys
import time as _time
from collections import deque
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
from types import SimpleNamespace

from src.core.log import get_logger

BACKEND_ROOT = Path(__file__).resolve().parents[2]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.append(str(BACKEND_ROOT))


from src.agents.Actions import AgentActionManager, LocationResolver, ActionState, ActionType
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
    append_event,
    archive_to_long_term,
    consolidate_to_single_day,
    clear_short_term_data,
    reset_day_runtime,
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
    KEEP_LAST,
)
from src.core.snapshot import take_snapshot, WorldSnapshot
from src.core.world_state import WorldState, Position, CurrentAction, AgentStatus
from src.core.world_events import WorldEventManager
from src.core.runtime_health import RuntimeHealthMonitor
from src import config as _cfg

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
        self.event_manager = WorldEventManager()
        self._applied_event_effects: set[str] = set()
        self.health_monitor = RuntimeHealthMonitor()

        self._day_index = 0
        # Per-agent set of subject ids currently within the perception circle
        # (edge-triggered so we only record a memory when someone *enters* range).
        self._in_range: Dict[str, set] = {}
        # Per-agent Brain (cognition) instances, built lazily. Each commands a
        # Body (the action state machine) — the human-like brain/body split.
        self._brains: Dict[str, Any] = {}
        # Rolling feed of recently-completed conversations (for the UI feed).
        self._recent_convs: deque = deque(maxlen=12)
        # Pixels already used at spawn, so agents don't stack on each other.
        self._spawn_points: List[tuple] = []
        # Per-agent observation fingerprint from the previous tick (novelty detection).
        # Key: agent_id, Value: frozenset of (agent_id, action_description) tuples.
        self._last_obs: Dict[str, frozenset] = {}
        # Per-agent cached observations from the current tick's perceive phase,
        # so Phase 4 (LLM decide) can reuse them without recomputing.
        self._tick_observations: Dict[str, list] = {}
        # Background LLM work is not serializable, but its deterministic input
        # envelope is.  Keep that envelope so a restore can restart an
        # in-flight conversation instead of silently losing it.
        self._pending_conversations: Dict[str, Dict[str, Any]] = {}
        self._conversation_tasks: Dict[str, asyncio.Task] = {}
        # Decisions are advisory; a slow provider response must not freeze the
        # simulation clock or WebSocket snapshots at an action boundary.
        self._decision_tasks: Dict[str, asyncio.Task] = {}

    @staticmethod
    def _conversation_key(first_id: str, second_id: str) -> str:
        return "|".join(sorted((first_id, second_id)))

    def checkpoint_state(self) -> Dict[str, Any]:
        """Serialize engine-owned state that is not held by WorldState."""
        return {
            "sim_start_date": self.sim_start_date,
            "sim_start_hhmm": self.sim_start_hhmm,
            "sim_end_hhmm": self.sim_end_hhmm,
            "day_index": self._day_index,
            "in_range": {
                agent_id: sorted(subject_ids)
                for agent_id, subject_ids in self._in_range.items()
            },
            # Novelty fingerprints affect whether an agent makes an LLM
            # decision next tick, so they are behavioural state rather than a
            # disposable UI cache.
            "last_observations": {
                agent_id: [list(item) for item in observations]
                for agent_id, observations in self._last_obs.items()
            },
            "recent_conversations": list(self._recent_convs),
            "relationship_matrix": self.relationship_matrix.snapshot(),
            "event_attendance": self.event_manager.attendance_snapshot(),
            "applied_event_effects": sorted(self._applied_event_effects),
            "pending_conversations": list(self._pending_conversations.values()),
        }

    def restore_checkpoint_state(self, state: Dict[str, Any]) -> None:
        """Restore engine-owned state after the world and registry are loaded."""
        if not state:
            return
        self.sim_start_date = state.get("sim_start_date", self.sim_start_date)
        self.sim_start_hhmm = state.get("sim_start_hhmm", self.sim_start_hhmm)
        self.sim_end_hhmm = state.get("sim_end_hhmm", self.sim_end_hhmm)
        self._day_index = state.get("day_index", self._day_index)
        self._in_range = {
            agent_id: set(subject_ids)
            for agent_id, subject_ids in state.get("in_range", {}).items()
        }
        self._last_obs = {
            agent_id: frozenset(tuple(item) for item in observations)
            for agent_id, observations in state.get("last_observations", {}).items()
        }
        self._recent_convs = deque(state.get("recent_conversations", []), maxlen=12)
        relationship_state = state.get("relationship_matrix")
        if relationship_state is not None:
            self.relationship_matrix.restore(relationship_state)
        self.event_manager.restore_attendance(state.get("event_attendance", {}))
        self._applied_event_effects = set(state.get("applied_event_effects", []))

        # Older checkpoints did not store a personality baseline. Upgrade
        # them on restore so a resumed day gets the same mood normalization as
        # a fresh day without invalidating its current action state.
        for agent in self.registry.all_states():
            if agent.emotion_baseline == 0.5:
                agent.emotion_baseline = self._emotion_baseline(agent.persona)

        self._pending_conversations = {}
        for pending in state.get("pending_conversations", []):
            first_id = pending.get("agent_a_id")
            second_id = pending.get("agent_b_id")
            if first_id and second_id:
                self._pending_conversations[self._conversation_key(first_id, second_id)] = pending

    async def resume_pending_conversations(self) -> None:
        """Restart durable conversation requests after a checkpoint restore.

        An API call already executing at save time cannot be serialized.  Its
        request envelope is retained, so this resumes the same logical work
        from the original participants, prompt inputs, and simulation time.
        """
        for key, pending in list(self._pending_conversations.items()):
            try:
                first = self.registry.get(pending["agent_a_id"])
                second = self.registry.get(pending["agent_b_id"])
            except KeyError:
                self._pending_conversations.pop(key, None)
                continue
            if not first.paused or not second.paused:
                self._pending_conversations.pop(key, None)
                continue
            self._start_conversation_task(first, second, pending)

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

    def _apply_event_opportunities(self, date: str) -> None:
        """Overlay deterministic, conflict-safe optional event attendance.

        The overlay happens after cached or generated plans are available, so
        it works equally for fresh, resumed, and handoff days.  Only entirely
        flexible windows can be replaced; the event manager rejects hard
        commitments before this method sees a changed plan.
        """
        states = self.registry.all_states()
        if not states:
            return
        plans = {state.agent_id: state.day_plan for state in states}
        personas = {state.agent_id: state.persona for state in states}
        updated = self.event_manager.apply_opportunities(
            date, plans, personas, self.relationship_matrix.get,
        )
        for state in states:
            plan = updated.get(state.agent_id, state.day_plan)
            if plan == state.day_plan:
                continue
            state.day_plan = plan
            if state.manager:
                state.manager.replace_day_plan(plan)
            save_day_plan(state.persona_name, date, plan)

    def _apply_finished_event_effects(self, date: str, hhmm: str) -> None:
        """Persist bounded social/mood effects once an attended event ends."""
        now = self._hhmm_to_minutes(hhmm)
        changed_relationships = False
        for event in self.event_manager.for_date(date):
            if event.id in self._applied_event_effects or now < self._hhmm_to_minutes(event.end_time):
                continue
            attendees = self.event_manager.attendance_snapshot().get(event.id, [])
            for agent_id in attendees:
                try:
                    state = self.registry.get(agent_id)
                except KeyError:
                    continue
                state.energy_level = max(0.0, min(1.0, state.energy_level + event.energy_effect))
                state.emotion_state = max(0.0, min(1.0, state.emotion_state + event.emotion_effect))
                append_event(state.persona_name, date, {
                    "type": "world_event", "action": event.name,
                    "location": event.location_id,
                    "summary": f"Attended {event.name}: {event.description}",
                    "details": {"event_id": event.id, "ended_at": event.end_time},
                })
            if event.relationship_effect and len(attendees) > 1:
                for first in attendees:
                    for second in attendees:
                        if first != second:
                            self.relationship_matrix.update(first, second, event.relationship_effect)
                            changed_relationships = True
            self._applied_event_effects.add(event.id)
            logger.info("[Events] completed %s with %d attendee(s)", event.id, len(attendees))
        if changed_relationships:
            self.relationship_matrix.save()

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

    def assert_checkpoint_roster_matches_seed(self, registry: AgentRegistry) -> None:
        """Reject a resume that would silently discard newly configured agents."""
        seed_ids = {self._persona_name_to_id(persona["Name"]) for persona in self._discover_personas()}
        checkpoint_ids = set(registry.all_ids())
        if seed_ids != checkpoint_ids:
            raise ValueError(
                "Checkpoint roster differs from the current persona roster; start a fresh simulation "
                "instead of resuming this checkpoint."
            )

    @staticmethod
    def _emotion_baseline(persona: Dict[str, Any]) -> float:
        """Derive a stable, modest mood baseline from stated personality traits.

        This is deliberately narrow: students can have a good or difficult day,
        but their normal state should cluster around neutral rather than begin
        at an arbitrary emotional extreme.
        """
        traits = " ".join(str(persona.get(key, "")) for key in ("innate", "lifestyle", "learned")).lower()
        baseline = 0.50
        for marker in ("friendly", "warm", "approachable", "outgoing", "energetic", "playful", "resilient", "optimistic"):
            if marker in traits:
                baseline += 0.015
        for marker in ("introverted", "quiet", "reserved", "awkward", "irregular sleep", "bad sleep", "procrastinate"):
            if marker in traits:
                baseline -= 0.015
        return max(0.44, min(0.57, baseline))

    @staticmethod
    def _bounded_action_emotion_change(action: Any) -> tuple[float, float]:
        """Return a realistic action delta and its allowed mood ceiling.

        Planner output can be optimistic. Classes and lab work are usually
        emotionally neutral; social/sport blocks can lift mood modestly, but
        only rare standout moments should approach the top of the scale.
        """
        description = (getattr(action, "description", "") or "").lower()
        delta = max(-0.08, min(0.06, float(getattr(action, "emotion_change", 0.0))))
        routine_words = ("class", "lecture", "lab", "study", "assignment", "coding", "project", "tutorial", "review")
        restorative_words = ("sleep", "rest", "shower", "meal", "breakfast", "lunch", "dinner")
        uplifting_words = ("sport", "gym", "run", "basketball", "music", "open mic", "friends", "club", "game")
        if any(word in description for word in routine_words):
            return min(delta, 0.012), 0.14
        if any(word in description for word in restorative_words):
            return min(delta, 0.025), 0.14
        if any(word in description for word in uplifting_words):
            return min(delta, 0.050), 0.22
        return delta, 0.18

    def _memory_context(self, persona_name, persona, before_date=None, query_hint=""):
        """Build (relevant_memories, rolling_summary) for a day-planner call.

        Uses the configured long-term memory backend (keyword by default). This
        is 0-LLM for the keyword backend. Failures degrade to empty memory so
        planning never breaks.
        """
        try:
            from src.agents.Long_term import get_retriever
            retr = get_retriever()
            query = query_hint or " ".join(
                str(persona.get(k, ""))
                for k in ("goals", "hobbies", "daily_plan_req", "Branch")
            )
            memories = retr.retrieve(persona_name, query, k=6)
            summary = retr.rolling_summary(persona_name, days=2, before_date=before_date)
            if memories or summary:
                logger.info(
                    "[WorldEngine]   memory for '%s': %d recalls, summary=%s",
                    persona_name, len(memories), "yes" if summary else "no",
                )
            return memories, summary
        except Exception as e:
            logger.warning("[WorldEngine] memory context failed for %s: %s", persona_name, e)
            return [], None

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

        # Initialise the Qdrant-only long-term memory backend once at startup.
        try:
            from src.agents.Long_term import get_retriever
            get_retriever()
        except Exception as exc:
            logger.warning("[WorldEngine] memory backend initialization failed: %s", exc)

        personas = self._discover_personas()
        if not personas:
            logger.error("[WorldEngine] no personas found — aborting")
            return

        from src.agents.day_planner import run as day_planner_run

        # ── Phase 1a (sequential, fast): resolve positions, check saved plans ──
        logger.info("[WorldEngine] preparing %d agents ...", len(personas))
        agent_setups = []
        for persona in personas:
            try:
                name = persona.get("Name", "unknown")
                hostel = persona.get("Hostel", "")
                agent_id = self._persona_name_to_id(name)
                position = self.resolver.random_interior_point(hostel, occupied=self._spawn_points)
                self._spawn_points.append((position.x, position.y))
                consolidate_to_single_day(name, self.sim_start_date)
                existing_plan = load_day_plan(name, self.sim_start_date)
                if existing_plan:
                    reset_day_runtime(name, self.sim_start_date)
                    logger.info(
                        "[WorldEngine]   reusing saved day plan for '%s' (%d actions) — no LLM",
                        name, len(existing_plan),
                    )
                    day_plan = existing_plan
                    error = None
                    manager = AgentActionManager(agent_id, day_plan, position, self.resolver)
                    self.registry.register(
                        AgentRuntimeState(
                            agent_id=agent_id, persona=persona, persona_name=name,
                            manager=manager, position=position, day_plan=day_plan,
                            emotion_state=self._emotion_baseline(persona),
                            emotion_baseline=self._emotion_baseline(persona), energy_level=1.0,
                        )
                    )
                    self.world.register_agent(agent_id, position)
                    logger.info("[WorldEngine]   registered '%s' (hostel=%s) — %d plan actions (cached)",
                                name, hostel, len(day_plan))
                else:
                    agent_setups.append((name, hostel, agent_id, position, persona))
            except Exception as e:
                logger.error("[WorldEngine]   failed to prepare agent '%s': %s — skipping",
                             persona.get("Name", "unknown"), e)
                import traceback; traceback.print_exc()
                continue

        # ── Phase 1b (parallel, slow): generate day plans for agents that need one ──
        logger.info("[WorldEngine] generating day plans for %d agents (parallel) ...", len(agent_setups))
        plan_t0 = _time.perf_counter()

        async def _plan_one(name: str, hostel: str, agent_id: str,
                            position, persona: dict) -> None:
            logger.info("[WorldEngine]   starting day plan for '%s' ...", name)
            mem, yday = self._memory_context(name, persona, before_date=self.sim_start_date)
            proxy = SimpleNamespace(persona=persona, relevant_memories=mem, yesterday_summary=yday)
            try:
                _loop = asyncio.get_event_loop()
                plan_result = await _loop.run_in_executor(
                    None,
                    lambda: day_planner_run(
                        proxy,
                        {
                            "current_time": f"{self.sim_start_date} {self.sim_start_hhmm}",
                            "places": None,
                            "persona_name": name,
                            "mode": "full_day",
                            "current_location_id": hostel,
                            "upcoming_events": self.event_manager.snapshot(self.sim_start_date, self.sim_start_hhmm).get("upcoming", []),
                        },
                    ),
                )
                day_plan = plan_result.get("day_plan", [])
                error = plan_result.get("error")
                if error:
                    logger.warning("[WorldEngine]   day plan for '%s' has issues: %s", name, error)
                manager = AgentActionManager(agent_id, day_plan, position, self.resolver)
                self.registry.register(
                    AgentRuntimeState(
                        agent_id=agent_id, persona=persona, persona_name=name,
                        manager=manager, position=position, day_plan=day_plan,
                        emotion_state=self._emotion_baseline(persona),
                        emotion_baseline=self._emotion_baseline(persona), energy_level=1.0,
                    )
                )
                self.world.register_agent(agent_id, position)
                logger.info("[WorldEngine]   registered '%s' (hostel=%s) — %d plan actions",
                            name, hostel, len(day_plan))
            except Exception as e:
                logger.error("[WorldEngine]   day plan failed for '%s': %s — skipping", name, e)
                import traceback; traceback.print_exc()
                # Register anyway with empty plan so the sim doesn't crash
                manager = AgentActionManager(agent_id, [], position, self.resolver)
                self.registry.register(
                    AgentRuntimeState(
                        agent_id=agent_id, persona=persona, persona_name=name,
                        manager=manager, position=position, day_plan=[],
                        emotion_state=self._emotion_baseline(persona),
                        emotion_baseline=self._emotion_baseline(persona), energy_level=1.0,
                    )
                )
                self.world.register_agent(agent_id, position)
                logger.info("[WorldEngine]   registered '%s' with empty plan (fallback)", name)

        tasks = [_plan_one(n, h, a, p, pe) for n, h, a, p, pe in agent_setups]
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

        self._apply_event_opportunities(self.sim_start_date)

        plan_elapsed = _time.perf_counter() - plan_t0
        logger.info("[WorldEngine] all day plans generated in %.1fs", plan_elapsed)

        # Assign frontend colors
        _COLORS = [
            "#ffcc33", "#ff6b6b", "#51cf66", "#339af0", "#cc5de8",
            "#f76707", "#20c997", "#f06595", "#748ffc", "#ffd43b",
        ]
        for i, state in enumerate(self.registry.all_states()):
            state.color = _COLORS[i % len(_COLORS)]

        # Register hostel rooms as world resources
        for state in self.registry.all_states():
            hostel = state.persona.get("Hostel", "")
            if hostel:
                self.world.register_resource(hostel)

        elapsed = _time.perf_counter() - t0
        # Start the clock at the configured time of day (e.g. 08:00) so the day
        # begins with real activity instead of everyone asleep at midnight.
        self.world.tick = self._hhmm_to_minutes(self.sim_start_hhmm)
        logger.info(
            "[WorldEngine] initialized %d agents in %.1fs — starting simulation at %s",
            len(self.registry), elapsed, self.sim_start_hhmm,
        )

    # ------------------------------------------------------------------ #
    # Tick loop
    # ------------------------------------------------------------------ #

    async def run_tick(self) -> Dict[str, Any]:
        """
        Execute one simulation tick in six phases:

          1. Snapshot  — freeze world state (one copy for all agents).
          2. Act       — all agents advance their body state machines in parallel.
          3. Perceive  — build observations from snapshot, detect novelty, set
                         flags for LLM decision; conversation detection happens here.
          4. LLM Decide— parallel LLM decision for agents with novel observations.
          5. Replan    — regenerate remaining-day plan for agents that chose replan.
          6. Resolve   — sequential: timeouts, day transitions, collisions, sync.

        Returns a dict snapshot for frontend use.
        """
        current_tick = self.world.tick
        hhmm = self._minutes_to_hhmm(current_tick % (24 * 60))

        # A generated conversation remains a real, visible simulation state
        # until its declared end, rather than disappearing with the LLM task.
        self._complete_finished_conversations(current_tick)

        agent_states = self.registry.all_states()

        # ══════ PHASE 2: Act (parallel) ══════
        if agent_states:
            await self._phase_act(agent_states, current_tick, hhmm)

        # Perception must observe the positions and actions produced by this
        # tick's act phase, rather than a one-tick-old world mirror.
        self._sync_to_world_state()
        snapshot = take_snapshot(self.world)

        # ══════ PHASE 3: Perceive (parallel) + conversation detection (sequential) ══════
        novelty_flags: Dict[str, bool] = {}  # agent_id -> has novel observations
        self._tick_observations.clear()
        if agent_states:
            perceive_results = await asyncio.gather(
                *[self._phase_perceive(s, snapshot) for s in agent_states],
                return_exceptions=True,
            )
            for state, result in zip(agent_states, perceive_results):
                if isinstance(result, Exception):
                    logger.error(
                        "[WorldEngine] agent '%s' perceive failed: %s", state.agent_id, result,
                    )
                else:
                    novelty_flags[state.agent_id], obs = result
                    self._tick_observations[state.agent_id] = obs

        # Phase 3b: Conversation detection (sequential)
        await self._check_conversations(current_tick, hhmm)
        self._timeout_conversations(current_tick)

        # ══════ PHASE 4: LLM Decide (parallel, only non-paused agents with novelty) ══════
        decide_results: Dict[str, Any] = self._collect_finished_decisions()
        decide_agents = [
            s for s in agent_states
            if not s.paused and novelty_flags.get(s.agent_id, False)
            and s.energy_level >= _cfg.DECIDE_MIN_ENERGY
            and s.emotion_state >= _cfg.DECIDE_MIN_EMOTION
        ]
        if decide_agents:
            for state in decide_agents:
                if state.agent_id in self._decision_tasks:
                    continue
                self._decision_tasks[state.agent_id] = asyncio.create_task(
                    self._phase_llm_decide(state, current_tick, hhmm),
                    name=f"decide:{state.agent_id}:{current_tick}",
                )

        # ══════ PHASE 5: Replan (parallel, agents whose LLM decision was "replan") ══════
        replan_agents = [
            s for s in agent_states
            if decide_results.get(s.agent_id) == "replan"
            and s.replan_count < _cfg.MAX_REPLANS_PER_AGENT_PER_DAY
        ]
        if replan_agents:
            await asyncio.gather(
                *[self._phase_replan(s, current_tick, hhmm) for s in replan_agents],
                return_exceptions=True,
            )

        # ══════ PHASE 6: Resolve (sequential) ══════
        await self._check_last_action_triggers(current_tick, hhmm)
        self._apply_finished_event_effects(self.sim_start_date, hhmm)
        self._resolve_collisions()
        self._sync_to_world_state()
        # ----- Advance tick -----
        self.world.advance_tick(minutes=SIM_MINUTES_PER_TICK)
        snapshot_tick = self.world.tick
        snapshot_hhmm = self._minutes_to_hhmm(snapshot_tick % (24 * 60))
        # A frontend frame must describe the same clock state as the engine
        # and checkpoint it was built from.  Returning the pre-advance clock
        # made the UI one tick behind rewind/fast-forward controls.
        health = self.health_monitor.observe(self, snapshot_tick, snapshot_hhmm)

        # ----- Save checkpoint -----
        save_checkpoint(
            self.world,
            self.registry,
            self.world.tick,
            engine_state=self.checkpoint_state(),
        )
        prune_checkpoints(keep_last=KEEP_LAST)

        # Build frontend snapshot
        return self._frontend_snapshot(snapshot_tick, snapshot_hhmm, health)

    def _frontend_snapshot(self, tick: int, hhmm: str, health: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Return the current UI-safe projection without advancing the world."""
        real_min_per_day = (1440 * _cfg.REAL_SECONDS_PER_SIM_MINUTE
                            / max(_cfg.TICK_SPEED, 1e-9)) / 60.0
        return {
            "tick": tick,
            "time": hhmm,
            "day": self._day_index + 1,
            "speed": {
                "multiplier": _cfg.TICK_SPEED,
                "real_min_per_day": round(real_min_per_day),
                "real_ms_per_sim_minute": round((_cfg.REAL_SECONDS_PER_SIM_MINUTE / max(_cfg.TICK_SPEED, 1e-9)) * 1000),
            },
            "recent_conversations": list(self._recent_convs),
            "events": self.event_manager.snapshot(self.sim_start_date, hhmm),
            "health": health if health is not None else self.health_monitor.latest,
            "agents": {
                s.agent_id: {
                    "name": s.persona_name,
                    "color": s.color,
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
                    "activity": (
                        s.manager.current_action.description
                        if s.manager and s.manager.current_action
                        else "Idle"
                    ),
                    "paused": s.paused,
                    "in_conversation": bool(s.paused and s.active_conversation),
                    "in_last_action": s.manager.is_last_action if s.manager else False,
                    "energy_level": s.energy_level,
                    "emotion_state": s.emotion_state,
                    "conversation": s.active_conversation,
                }
                for s in self.registry.all_states()
            },
        }

    def current_frontend_snapshot(self) -> Dict[str, Any]:
        """Expose restored state immediately after a checkpoint rewind."""
        tick = self.world.tick
        hhmm = self._minutes_to_hhmm(tick % (24 * 60))
        return self._frontend_snapshot(tick, hhmm)

    async def _phase_act(self, agent_states, current_tick, hhmm) -> None:
        """Phase 2: all agents advance their body state machines in parallel."""
        act_results = await asyncio.gather(
            *[self._run_agent_act(s, current_tick, hhmm) for s in agent_states],
            return_exceptions=True,
        )
        for state, result in zip(agent_states, act_results):
            if isinstance(result, Exception):
                logger.error(
                    "[WorldEngine] agent '%s' act failed: %s", state.agent_id, result,
                )

    def _collect_finished_decisions(self) -> Dict[str, str]:
        """Consume completed advisory decisions without delaying a tick."""
        completed: Dict[str, str] = {}
        for agent_id, task in list(self._decision_tasks.items()):
            if not task.done():
                continue
            self._decision_tasks.pop(agent_id, None)
            try:
                completed[agent_id] = task.result()
            except asyncio.CancelledError:
                continue
            except Exception as exc:
                logger.error("[WorldEngine] agent '%s' LLM decide failed: %s", agent_id, exc)
                completed[agent_id] = "continue"
        return completed

    async def _phase_perceive(self, state: AgentRuntimeState, snapshot: WorldSnapshot) -> tuple:
        """Phase 3: build observations for one agent, detect novelty.
        Returns (novel: bool, observations: list).
        """
        if state.paused:
            return False, []
        observations = self._build_observations(state, snapshot)
        # Compute fingerprint: frozenset of (agent_id, action_description)
        fingerprint = frozenset(
            (o["agent_id"], o.get("current_action", "")) for o in observations
        )
        prev = self._last_obs.get(state.agent_id, frozenset())
        novel = fingerprint != prev
        self._last_obs[state.agent_id] = fingerprint
        return novel, observations

    async def _phase_llm_decide(self, state: AgentRuntimeState, tick: int, hhmm: str) -> str:
        """Phase 4: call the brain's LLM decision for one agent.
        Returns "continue" or "replan".
        """
        from src.core.budget import GOVERNOR
        if not GOVERNOR.can_afford("decide", cost=1):
            logger.info("[WorldEngine] budget denied decide for '%s' — continuing", state.persona_name)
            return "continue"
        brain = self._brain_for(state)
        observations = self._tick_observations.get(state.agent_id, [])
        query = " ".join(str(item.get("current_action", "")) for item in observations)
        memories, _ = self._memory_context(state.persona_name, state.persona, query_hint=query)
        # Gemini is a synchronous SDK call.  Run it in the default worker
        # pool so a slow response, rate-limit wait, or retry never prevents
        # WebSocket broadcasts and the rest of the simulation from advancing.
        decision = await asyncio.to_thread(
            brain.decide_tick,
            tick=tick,
            hhmm=hhmm,
            observations=observations,
            day_plan=state.day_plan,
            replan_count=state.replan_count,
            max_replans=_cfg.MAX_REPLANS_PER_AGENT_PER_DAY,
            energy_level=state.energy_level,
            emotion_state=state.emotion_state,
            relevant_memories=memories,
        )
        return decision.decision

    async def _phase_replan(self, state: AgentRuntimeState, tick: int, hhmm: str) -> None:
        """Phase 5: regenerate remaining-day plan for one agent.
        Called only when the LLM returned "replan" and budget allows.
        """
        from src.core.budget import GOVERNOR
        if not GOVERNOR.can_afford("replan", cost=4):
            logger.info("[WorldEngine] budget denied replan for '%s' — skipping", state.persona_name)
            return
        from src.agents.day_planner import run as day_planner_run
        from types import SimpleNamespace

        loop = asyncio.get_event_loop()
        mem, yday = self._memory_context(
            state.persona_name, state.persona,
            before_date=date_from_simulation_time(f"{self.sim_start_date} {hhmm}"),
        )
        proxy = SimpleNamespace(
            persona=state.persona,
            relevant_memories=mem,
            yesterday_summary=yday,
        )
        try:
            plan_result = await loop.run_in_executor(
                None,
                lambda: day_planner_run(
                    proxy,
                    {
                        "current_time": f"{self.sim_start_date} {hhmm}",
                        "places": None,
                        "persona_name": state.persona_name,
                        "mode": "remaining",
                        "current_location_id": state.position.location_id,
                        "upcoming_events": self.event_manager.snapshot(self.sim_start_date, hhmm).get("upcoming", []),
                    },
                ),
            )
            new_plan = plan_result.get("day_plan", [])
            if new_plan and state.manager:
                state.day_plan = new_plan
                state.manager.replace_day_plan(new_plan)
                state.replan_count += 1
                logger.info(
                    "[WorldEngine] replan for '%s': %d actions (count=%d)",
                    state.persona_name, len(new_plan), state.replan_count,
                )
        except Exception as e:
            logger.error(
                "[WorldEngine] replan failed for '%s': %s", state.persona_name, e,
            )

    async def _run_agent_act(
        self, state: AgentRuntimeState, tick: int, hhmm: str
    ) -> None:
        """Phase 2: execute the decision — advance the body's state machine.

        Updates registry position and action from the manager after the body
        advances. Paused agents are skipped.
        """
        if state.paused:
            return
        manager = state.manager
        if manager is None:
            return
        brain = self._brain_for(state)
        action = brain.act(tick)
        state.position = manager.position
        state.current_action = action.model_dump() if action else None

        # Per-tick energy/emotion update
        self._update_energy_emotion(state)

    def _timeout_conversations(self, current_tick: int) -> None:
        """Fail safe only for LLM requests that are still generating."""
        for state in self.registry.all_states():
            if (
                not state.paused
                or not state.active_conversation
                or state.active_conversation.get("status") != "generating"
            ):
                continue
            if current_tick - state.conversation_start_tick < 30:
                continue
            partner_id = state.last_conversation_partner
            logger.info(
                "[WorldEngine] conversation timeout for '%s' (started tick %d, now tick %d)",
                state.persona_name, state.conversation_start_tick, current_tick,
            )
            self._end_conversation(state, current_tick)
            if partner_id:
                try:
                    partner = self.registry.get(partner_id)
                    self._end_conversation(partner, current_tick)
                    self._discard_pending_conversation(state.agent_id, partner.agent_id)
                except KeyError:
                    pass

    @staticmethod
    def _end_conversation(state: AgentRuntimeState, current_tick: Optional[int] = None) -> None:
        """Restore agent state after a conversation ends (timeout or normal)."""
        state.paused = False
        state.conversation_start_tick = 0
        state.active_conversation = None
        mgr = state.manager
        if mgr:
            mgr.resume_from_conversation(world_tick=current_tick)

    def _complete_finished_conversations(self, current_tick: int) -> None:
        """Resume each completed conversation once its simulated duration ends."""
        completed: set[str] = set()
        for state in self.registry.all_states():
            conversation = state.active_conversation or {}
            partner_id = conversation.get("partner_id")
            if (
                not state.paused
                or conversation.get("status") != "active"
                or current_tick < conversation.get("ends_at_tick", current_tick + 1)
                or not partner_id
                or state.agent_id in completed
            ):
                continue
            try:
                partner = self.registry.get(partner_id)
            except KeyError:
                self._end_conversation(state, current_tick)
                continue
            self._end_conversation(state, current_tick)
            self._end_conversation(partner, current_tick)
            self.world.record_conversation(state.agent_id, current_tick)
            self.world.record_conversation(partner.agent_id, current_tick)
            completed.update((state.agent_id, partner.agent_id))
            logger.info(
                "[WorldEngine] conversation '%s' <-> '%s' ended after its simulated duration",
                state.persona_name, partner.persona_name,
            )

    def _resolve_collisions(self, min_gap: int = 4) -> None:
        """Ensure no two agents share (nearly) the same pixel — nudge later ones
        a few pixels away. Keeps the map readable and honours 'never same pixel'."""
        seen: List[tuple] = []
        g2 = min_gap * min_gap
        for s in self.registry.all_states():
            x, y = s.position.x, s.position.y
            tries = 0
            while tries < 15 and any((x - ox) ** 2 + (y - oy) ** 2 < g2 for ox, oy in seen):
                x = s.position.x + random.randint(-7, 7)
                y = s.position.y + random.randint(-7, 7)
                tries += 1
            if (x, y) != (s.position.x, s.position.y):
                # Registry and action manager are two views of the live body.
                # Keep them atomic here; otherwise _run_agent_act restores the
                # old manager position on the next tick and the collision
                # reappears.
                corrected = Position(x=x, y=y, location_id=s.position.location_id)
                s.position = corrected
                if s.manager is not None:
                    s.manager.position = corrected
            seen.append((x, y))

    def _update_energy_emotion(self, state: AgentRuntimeState) -> None:
        """Update wellbeing without allowing routine plans to create euphoria."""
        manager = state.manager
        if manager is None or manager.current_action is None:
            return
        action = manager.current_action
        try:
            start_min = self._hhmm_to_minutes(action.start_time)
            end_min = self._hhmm_to_minutes(action.end_time)
            duration = max(1, end_min - start_min)
            tick_step = _cfg.SIM_MINUTES_PER_TICK
            energy_tick = (action.energy_change / duration) * tick_step
            action_emotion_change, mood_headroom = self._bounded_action_emotion_change(action)
            emotion_tick = (action_emotion_change / duration) * tick_step
            state.energy_level = max(0.0, min(1.0, state.energy_level + energy_tick))
            baseline = state.emotion_baseline
            # A small regression to the agent's normal mood prevents a chain
            # of mildly positive schedule entries from permanently saturating
            # the display at "Very Happy".
            recovery = (baseline - state.emotion_state) * min(0.05, 0.002 * tick_step)
            ceiling = min(0.82, baseline + mood_headroom)
            state.emotion_state = max(0.18, min(ceiling, state.emotion_state + emotion_tick + recovery))
        except Exception:
            pass

    def _brain_for(self, state: AgentRuntimeState):
        """Lazily build and cache one Brain (+ Body) per agent."""
        brain = self._brains.get(state.agent_id)
        if brain is None:
            from src.agents.body import BodyController
            from src.agents.brain import Brain
            brain = Brain(
                agent_id=state.agent_id,
                persona_name=state.persona_name,
                body=BodyController(state.manager),
                persona=state.persona,
            )
            self._brains[state.agent_id] = brain
        return brain

    def _build_observations(self, state: AgentRuntimeState, w_snapshot: WorldSnapshot) -> list:
        """Build structured observations for one agent from the frozen snapshot.

        Reports every agent within the configured pixel radius, including their
        position and current action. Placeholder for future world-events.
        """
        radius = float(_cfg.PERCEPTION_RADIUS_PX)
        nearby_snaps = w_snapshot.agents_within_px(state.agent_id, radius)
        observed = []
        for ns in nearby_snaps:
            try:
                rs = self.registry.get(ns.agent_id)
                name = rs.persona_name if rs else ns.agent_id
            except KeyError:
                name = ns.agent_id
            observed.append({
                "agent_id": ns.agent_id,
                "name": name,
                "position": {"x": ns.position.x, "y": ns.position.y},
                "current_action": ns.current_action.description if ns.current_action else "idle",
            })
        return observed

    # ------------------------------------------------------------------ #
    # Perception (0-LLM) + spatial helpers
    # ------------------------------------------------------------------ #

    def _neighbors_within_px(self, state, radius_px: float) -> List:
        """Registry states whose pixel position is within radius_px of `state`
        (excludes self). Nearest first."""
        out = []
        for other in self.registry.all_states():
            if other.agent_id == state.agent_id:
                continue
            dx = other.position.x - state.position.x
            dy = other.position.y - state.position.y
            d2 = dx * dx + dy * dy
            if d2 <= radius_px * radius_px:
                out.append((d2, other))
        out.sort(key=lambda t: t[0])
        return [o for _d, o in out]

    def _perceive_and_record(self, tick: int, hhmm: str) -> None:
        """For each agent, find who is within the perception circle and record a
        memory the moment someone *enters* range (edge-triggered, de-duplicated).
        Pure spatial math — no LLM. Cheap keyword-backend writes only.
        """
        radius = float(_cfg.PERCEPTION_RADIUS_PX)
        try:
            from src.agents.Long_term import get_retriever
            retriever = get_retriever()
        except Exception:
            retriever = None

        date_str = date_from_simulation_time(f"{self.sim_start_date} {hhmm}")

        for state in self.registry.all_states():
            neighbors = self._neighbors_within_px(state, radius)
            current_ids = {n.agent_id for n in neighbors}
            previous_ids = self._in_range.get(state.agent_id, set())
            newly_entered = current_ids - previous_ids
            self._in_range[state.agent_id] = current_ids

            if not newly_entered or retriever is None:
                continue

            loc = state.position.location_id or "campus"
            crowd = len(neighbors)
            for n in neighbors:
                if n.agent_id not in newly_entered:
                    continue
                doing = ""
                if n.manager and n.manager.current_action:
                    doing = f" ({n.manager.current_action.description})"
                text = (
                    f"At {hhmm} near {loc}, saw {n.persona_name}{doing}"
                    + (f"; {crowd} people around." if crowd > 1 else ".")
                )
                try:
                    retriever.store(state.persona_name, text, kind="observation",
                                    importance=0.5, date_str=date_str)
                except Exception as e:
                    logger.debug("[WorldEngine] observation store failed: %s", e)

    def _conversation_request(
        self,
        a: AgentRuntimeState,
        b: AgentRuntimeState,
        loc_id: str,
        tick: int,
        hhmm: str,
    ) -> Dict[str, Any]:
        """Capture the inputs needed to restart an in-flight LLM request."""
        date_ctx = date_from_simulation_time(f"{self.sim_start_date} {hhmm}")
        memories_a, _ = self._memory_context(
            a.persona_name, a.persona, before_date=date_ctx,
            query_hint=f"{b.persona_name} {loc_id}",
        )
        memories_b, _ = self._memory_context(
            b.persona_name, b.persona, before_date=date_ctx,
            query_hint=f"{a.persona_name} {loc_id}",
        )
        return {
            "agent_a_id": a.agent_id,
            "agent_b_id": b.agent_id,
            "location_id": loc_id,
            "start_tick": tick,
            "hhmm": hhmm,
            "action_a": a.manager.current_action.description if a.manager and a.manager.current_action else "unknown",
            "action_b": b.manager.current_action.description if b.manager and b.manager.current_action else "unknown",
            "relationship_a_to_b": self.relationship_matrix.get(a.agent_id, b.agent_id),
            "relationship_b_to_a": self.relationship_matrix.get(b.agent_id, a.agent_id),
            "relationship_context_a": self.relationship_matrix.context(a.agent_id, b.agent_id).model_dump(),
            "relationship_context_b": self.relationship_matrix.context(b.agent_id, a.agent_id).model_dump(),
            "memories_a": memories_a,
            "memories_b": memories_b,
            "energy_a": a.energy_level,
            "emotion_a": a.emotion_state,
            "energy_b": b.energy_level,
            "emotion_b": b.emotion_state,
        }

    @staticmethod
    def _relationship_context_text(record: Dict[str, Any]) -> str:
        """Compact structured relationship context for a conversation prompt."""
        if not isinstance(record, dict):
            return ""
        tags = ", ".join(str(tag) for tag in record.get("tags", [])[:4])
        context = str(record.get("context", "")).strip()
        return "; ".join(part for part in (tags, context) if part)

    def _start_conversation_task(
        self,
        a: AgentRuntimeState,
        b: AgentRuntimeState,
        request: Dict[str, Any],
    ) -> None:
        """Run one durable conversation request and clear its bookkeeping."""
        key = self._conversation_key(a.agent_id, b.agent_id)
        if key in self._conversation_tasks:
            return

        async def _runner() -> None:
            try:
                await self._run_conversation_pipeline(a, b, request)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception(
                    "[WorldEngine] conversation pipeline crashed for '%s' <-> '%s'",
                    a.persona_name, b.persona_name,
                )
                self._end_conversation(a)
                self._end_conversation(b)
            finally:
                self._pending_conversations.pop(key, None)
                self._conversation_tasks.pop(key, None)

        self._conversation_tasks[key] = asyncio.create_task(_runner())

    def _discard_pending_conversation(self, first_id: str, second_id: str) -> None:
        """Prevent a timed-out conversation from being restarted on restore."""
        self._pending_conversations.pop(self._conversation_key(first_id, second_id), None)

    async def _check_conversations(self, tick: int, hhmm: str) -> None:
        """
        Detect settled pairs at the same venue whose proximity circles overlap
        and fire a background LLM conversation. A close pass on two unrelated
        routes is not a conversation: each agent must have completed transit
        and share one resolved location id. The pair-only rule, cooldown,
        per-day cap, and blocked-action (e.g. sleeping) guard still apply.
        Triggered pairs are paused immediately and resume when the background
        task completes.
        """
        radius = float(_cfg.CONVERSATION_RADIUS_PX)

        def is_settled_candidate(state: AgentRuntimeState) -> bool:
            action = state.manager.current_action if state.manager else None
            return (
                not state.paused
                and not (state.manager and state.manager.is_last_action)
                and bool(state.position.location_id)
                and not (action and action.action_type == ActionType.MOVE)
                and state.energy_level >= _cfg.CONVERSATION_MIN_ENERGY
                and state.emotion_state >= _cfg.CONVERSATION_MIN_EMOTION
            )

        candidates = [s for s in self.registry.all_states() if is_settled_candidate(s)]
        candidate_ids = {state.agent_id for state in candidates}
        busy: set = set()  # agents already committed to a conversation this tick

        for a in candidates:
            if a.agent_id in busy:
                continue
            neighbors = [
                n for n in self._neighbors_within_px(a, radius)
                if n.agent_id in candidate_ids
                and n.agent_id not in busy
                and n.position.location_id == a.position.location_id
            ]
            if not neighbors:
                continue
            # Pair-only rule: a clean 1:1 encounter. More than one other agent
            # inside the circle is a crowd — skip (mirrors the old >2 rule).
            if len(neighbors) > 1:
                logger.debug(
                    "[WorldEngine] %d agents within %.0fpx of '%s' — skipping (pair-only rule)",
                    len(neighbors) + 1, radius, a.persona_name,
                )
                continue
            b = neighbors[0]
            # Require a symmetric clean pair: b's circle must also contain only a.
            b_neighbors = [
                n for n in self._neighbors_within_px(b, radius)
                if n.agent_id in candidate_ids
                and n.position.location_id == b.position.location_id
            ]
            if len(b_neighbors) > 1:
                continue

            # Cooldown check
            if not self.world.can_converse(a.agent_id, b.agent_id):
                continue
            # No back-to-back repeats: they can't talk again until at least one
            # of them has spoken with someone else since (avoids the two of them
            # looping conversations while standing together).
            if a.last_conversation_partner == b.agent_id or b.last_conversation_partner == a.agent_id:
                continue
            # Max conversations cap
            if a.conversation_count >= MAX_CONVERSATIONS_PER_AGENT:
                continue
            if b.conversation_count >= MAX_CONVERSATIONS_PER_AGENT:
                continue
            # Both must be in compatible (non-blocked, e.g. not sleeping) actions
            if a.manager and a.manager.current_action:
                if _is_action_blocked(CurrentAction(
                    description=a.manager.current_action.description,
                    start_tick=tick, end_tick=tick + 10,
                )):
                    continue
            if b.manager and b.manager.current_action:
                if _is_action_blocked(CurrentAction(
                    description=b.manager.current_action.description,
                    start_tick=tick, end_tick=tick + 10,
                )):
                    continue

            loc_id = a.position.location_id or b.position.location_id or "campus"

            logger.info(
                "[WorldEngine] triggering conversation: '%s' <-> '%s' near %s (within %.0fpx)",
                a.persona_name, b.persona_name, loc_id, radius,
            )

            # Capture the original request before replacing the actions with a
            # visible conversation state.  This payload is checkpointed while
            # the provider call is in flight.
            request = self._conversation_request(a, b, loc_id, tick, hhmm)

            # Immediately set agents to conversation mode (frozen until bg task completes)
            if a.manager:
                a.manager.set_conversation_action(b.persona_name, tick)
                a.current_action = a.manager.current_action.model_dump()
                a.paused = True
                a.conversation_count += 1
                a.last_conversation_partner = b.agent_id
                a.conversation_start_tick = tick
                a.active_conversation = {
                    "partner_name": b.persona_name,
                    "partner_id": b.agent_id,
                    "location_id": loc_id,
                    "started_tick": tick,
                    "status": "generating",
                }
            if b.manager:
                b.manager.set_conversation_action(a.persona_name, tick)
                b.current_action = b.manager.current_action.model_dump()
                b.paused = True
                b.conversation_count += 1
                b.last_conversation_partner = a.agent_id
                b.conversation_start_tick = tick
                b.active_conversation = {
                    "partner_name": a.persona_name,
                    "partner_id": a.agent_id,
                    "location_id": loc_id,
                    "started_tick": tick,
                    "status": "generating",
                }

            self._pending_conversations[self._conversation_key(a.agent_id, b.agent_id)] = request
            self._start_conversation_task(a, b, request)
            busy.add(a.agent_id)
            busy.add(b.agent_id)

            logger.info(
                "[WorldEngine] conversation '%s' <-> '%s' started (async, waiting for LLM)",
                a.persona_name, b.persona_name,
            )

    async def _run_conversation_pipeline(
        self,
        a: AgentRuntimeState,
        b: AgentRuntimeState,
        request: Dict[str, Any],
    ) -> None:
        """
        Background task: generates the conversation via LLM, saves results,
        and resumes both agents with new remaining-day plans.
        Runs in a thread pool so it doesn't block the event loop.
        """
        loop = asyncio.get_event_loop()

        loc_id = request["location_id"]
        tick = request["start_tick"]
        hhmm = request["hhmm"]
        a_action = CurrentAction(
            description=request["action_a"],
            start_tick=tick,
            end_tick=tick + 10,
            target_location_id=loc_id,
        )
        b_action = CurrentAction(
            description=request["action_b"],
            start_tick=tick,
            end_tick=tick + 10,
            target_location_id=loc_id,
        )
        # Step 1: Generate conversation (blocking LLM call → thread pool)
        conv_result = await loop.run_in_executor(
            None,
            lambda: generate_conversation(
                a.agent_id, b.agent_id,
                a.persona, b.persona,
                a.day_plan, b.day_plan,
                a_action, b_action,
                request["relationship_a_to_b"], request["relationship_b_to_a"],
                loc_id, hhmm,
                request["memories_a"], request["memories_b"],
                energy_a=request["energy_a"], emotion_a=request["emotion_a"],
                energy_b=request["energy_b"], emotion_b=request["emotion_b"],
                relationship_context_a=self._relationship_context_text(request.get("relationship_context_a", {})),
                relationship_context_b=self._relationship_context_text(request.get("relationship_context_b", {})),
            ),
        )

        if conv_result is None:
            logger.warning(
                "[WorldEngine] conversation LLM failed for '%s' <-> '%s' — unpausing",
                a.persona_name, b.persona_name,
            )
            self._end_conversation(a)
            self._end_conversation(b)
            return

        # If the conversation was already ended by timeout while the LLM was
        # running, discard the result and return silently.
        if not a.paused or not b.paused:
            logger.info(
                "[WorldEngine] conversation '%s' <-> '%s' ended by timeout while LLM ran — discarding result",
                a.persona_name, b.persona_name,
            )
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

        # Add to the rolling UI feed (last N conversations across the campus).
        self._recent_convs.appendleft({
            "time": hhmm,
            "participants": [a.persona_name, b.persona_name],
            "summary": conv_result.summary,
            "sentiment": conv_result.sentiment,
            "location": loc_id,
        })

        # Step 3: Store conversation messages on agent states (for frontend snapshot)
        messages = conv_entry["messages"]
        a.active_conversation = {
            "partner_name": b.persona_name,
            "partner_id": b.agent_id,
            "messages": messages,
            "duration_minutes": conv_result.duration_minutes,
            "started_tick": self.world.tick,
            "ends_at_tick": self.world.tick + conv_result.duration_minutes,
            "status": "active",
        }
        b.active_conversation = {
            "partner_name": a.persona_name,
            "partner_id": a.agent_id,
            "messages": messages,
            "duration_minutes": conv_result.duration_minutes,
            "started_tick": self.world.tick,
            "ends_at_tick": self.world.tick + conv_result.duration_minutes,
            "status": "active",
        }
        # The action placeholder is intentionally open-ended while the LLM is
        # generating. Once a duration exists, make the UI/action snapshot tell
        # the same truth as ``ends_at_tick`` instead of displaying 23:59.
        conversation_end = self._minutes_to_hhmm((self.world.tick + conv_result.duration_minutes) % (24 * 60))
        for state in (a, b):
            if state.manager and state.manager.current_action:
                state.manager.current_action.end_time = conversation_end
                state.current_action = state.manager.current_action.model_dump()

        # Step 4: Update relationship matrix
        self.relationship_matrix.update(a.agent_id, b.agent_id, conv_result.relationship_delta)
        self.relationship_matrix.update(b.agent_id, a.agent_id, conv_result.relationship_delta)
        self.relationship_matrix.save()
        logger.info(
            "[WorldEngine] conversation '%s' <-> '%s' active until tick %d",
            a.persona_name, b.persona_name, self.world.tick + conv_result.duration_minutes,
        )
        return

        # Step 5: Generate remaining-day plans (blocking → thread pool)
        from src.agents.day_planner import run as day_planner_run

        def _run_planner(state: AgentRuntimeState) -> list:
            mem, yday = self._memory_context(
                state.persona_name, state.persona,
                before_date=date_from_simulation_time(f"{self.sim_start_date} {hhmm}"),
            )
            proxy = SimpleNamespace(
                persona=state.persona,
                relevant_memories=mem,
                yesterday_summary=yday,
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

        # Step 5: Resume both agents. Replanning is disabled by default — kept
        # behind a flag for future re-enablement.
        _ENABLE_POST_CONVERSATION_REPLAN = False

        for state in (a, b):
            try:
                if _ENABLE_POST_CONVERSATION_REPLAN:
                    should_replan = bool(getattr(conv_result, "should_replan", False))
                    try:
                        from src.core.budget import GOVERNOR
                    except Exception:
                        GOVERNOR = None
                    allow_replan = (
                        should_replan
                        and state.replan_count < _cfg.MAX_REPLANS_PER_AGENT_PER_DAY
                        and (GOVERNOR is None or GOVERNOR.can_afford("replan", cost=4))
                    )
                    if allow_replan:
                        new_plan = await loop.run_in_executor(None, _run_planner, state)
                        if new_plan and state.manager:
                            state.manager.resume_from_conversation(new_plan)
                            state.day_plan = new_plan
                            state.replan_count += 1
                            logger.info(
                                "[WorldEngine] post-conversation replan for '%s' (%d actions): %s",
                                state.persona_name, len(new_plan),
                                getattr(conv_result, "plan_change", "") or "",
                            )
                        elif state.manager:
                            state.manager.resume_from_conversation(state.day_plan)
                    else:
                        if state.manager:
                            state.manager.resume_from_conversation(state.day_plan)
                else:
                    # No replan — resume the day already planned (0 LLM).
                    if state.manager:
                        state.manager.resume_from_conversation(state.day_plan)
                state.active_conversation = None
            except Exception as e:
                logger.error(
                    "[WorldEngine] resume after conversation failed for '%s': %s",
                    state.persona_name, e,
                )
                if state.manager:
                    try:
                        state.manager.resume_from_conversation(state.day_plan)
                    except Exception:
                        pass

        # Step 6: Unpause agents
        a.paused = False
        b.paused = False

        logger.info(
            "[WorldEngine] conversation '%s' <-> '%s' complete — agents resumed",
            a.persona_name, b.persona_name,
        )

    async def _check_last_action_triggers(self, tick: int, hhmm: str) -> None:
        """
        When an agent enters their last action of the day plan:
        Archive durable day memories to Qdrant.
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
                "[WorldEngine] '%s' entered last action — archiving day",
                state.persona_name,
            )

            # Archive current day
            date_str = date_from_simulation_time(f"{self.sim_start_date} {hhmm}")
            try:
                archive_result = await archive_to_long_term(state.persona_name, date_str)
                logger.info(
                    "[WorldEngine]   archive: %s", archive_result.get("summary", "")[:80],
                )
                state.day_archived = True
            except Exception as e:
                logger.error("[WorldEngine]   archive failed for '%s': %s", state.persona_name, e)

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
                if state.active_conversation:
                    start_tick = state.active_conversation.get(
                        "started_tick", state.conversation_start_tick,
                    )
                    end_tick = state.active_conversation.get("ends_at_tick", start_tick + 30)
                else:
                    day_start_tick = self.world.tick - (self.world.tick % (24 * 60))
                    start_minute = self._hhmm_to_minutes(action.start_time)
                    end_minute = self._hhmm_to_minutes(action.end_time)
                    start_tick = day_start_tick + start_minute
                    end_tick = day_start_tick + end_minute
                    if end_minute <= start_minute:
                        end_tick += 24 * 60
                ca = CurrentAction(
                    description=action.description,
                    start_tick=start_tick,
                    end_tick=end_tick,
                    target_location_id=action.location_id or state.position.location_id,
                )
                self.world.set_agent_action(agent_id, ca)
            else:
                self.world.clear_agent_action(agent_id)

    async def _drain_conversations_for_handoff(self, timeout_seconds: float) -> int:
        """Give active conversations a bounded chance to persist their results."""
        tasks = list(self._conversation_tasks.values())
        if not tasks:
            return 0

        logger.info(
            "[WorldEngine] day handoff: waiting up to %.0fs for %d conversation(s)",
            timeout_seconds, len(tasks),
        )
        _done, pending = await asyncio.wait(tasks, timeout=max(0.0, timeout_seconds))
        if not pending:
            return 0

        logger.warning(
            "[WorldEngine] day handoff timed out with %d conversation(s); cancelling them",
            len(pending),
        )
        timed_out_requests = list(self._pending_conversations.values())
        for task in pending:
            task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)

        for request in timed_out_requests:
            try:
                self._end_conversation(self.registry.get(request["agent_a_id"]))
                self._end_conversation(self.registry.get(request["agent_b_id"]))
            except KeyError:
                pass
            self._pending_conversations.pop(
                self._conversation_key(request["agent_a_id"], request["agent_b_id"]), None,
            )
        return len(pending)

    async def _discard_pending_decisions_for_handoff(self) -> int:
        """Prevent a prior day's advisory decision from affecting a new day.

        Decision tasks deliberately run in the background so provider latency
        cannot stall a tick.  Their observation and plan inputs belong to one
        simulated day, however, so a result that arrives after midnight must
        never be consumed as a next-day replan.
        """
        tasks = list(self._decision_tasks.values())
        if not tasks:
            return 0

        pending = [task for task in tasks if not task.done()]
        for task in pending:
            task.cancel()
        # Consume every outcome before releasing the references.  Cancelling a
        # to_thread wrapper cannot stop its SDK call, but it does ensure its
        # eventual result cannot mutate this engine on the following day.
        await asyncio.gather(*tasks, return_exceptions=True)
        self._decision_tasks.clear()
        logger.info(
            "[WorldEngine] discarded %d prior-day decision task(s) at handoff",
            len(tasks),
        )
        return len(tasks)

    async def handoff_to_next_day(
        self,
        next_date: str,
        conversation_timeout_seconds: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Finish one calendar day without replacing the engine or agents.

        Conversation output is allowed to settle first, then every agent's
        completed day is compressed into long-term memory.  New plans are
        generated with the ending action, location, and wellbeing as explicit
        continuity context; positions, brains, relationships, and action
        managers remain owned by this same engine instance.
        """
        old_date = self.sim_start_date
        timeout = (
            _cfg.DAY_HANDOFF_CONVERSATION_TIMEOUT_SECONDS
            if conversation_timeout_seconds is None else conversation_timeout_seconds
        )
        cancelled = await self._drain_conversations_for_handoff(timeout)
        await self._discard_pending_decisions_for_handoff()

        # Preserve the full replay before resetting the per-day action log.
        # History/checkpoint storage is observability, not a reason to stop a
        # living simulation if a file is temporarily unavailable.
        handoff_warnings: list[str] = []
        try:
            save_history(self.world, old_date)
        except Exception as exc:
            logger.exception("[WorldEngine] unable to save replay for %s", old_date)
            handoff_warnings.append(f"history save failed: {exc}")

        async def _archive_and_clear(state: AgentRuntimeState) -> bool:
            try:
                # Re-index even when we made the early final-activity archive:
                # conversations and actions completed afterward must be part of
                # the authoritative end-of-day Qdrant snapshot.
                await archive_to_long_term(state.persona_name, old_date)
                state.day_archived = True
                clear_short_term_data(state.persona_name, old_date)
                return True
            except Exception as exc:
                logger.error(
                    "[WorldEngine] final archive failed for '%s': %s",
                    state.persona_name, exc,
                )
                return False

        archive_results = await asyncio.gather(
            *[_archive_and_clear(state) for state in self.registry.all_states()],
            return_exceptions=False,
        )

        from src.agents.day_planner import run as day_planner_run

        def _handoff_context(state: AgentRuntimeState) -> str:
            action = state.manager.current_action if state.manager else None
            action_text = action.description if action else "no active action"
            location = state.position.location_id or "their current campus position"
            return (
                f"The previous day ended while the agent was {action_text} at {location}. "
                f"Energy is {state.energy_level:.2f}/1.0 and emotion is "
                f"{state.emotion_state:.2f}/1.0. Continue naturally from this "
                "physical and emotional state; do not abruptly relocate them."
            )

        async def _plan_next_day(state: AgentRuntimeState) -> tuple[AgentRuntimeState, list]:
            memories, yesterday = self._memory_context(
                state.persona_name, state.persona, before_date=next_date,
            )
            proxy = SimpleNamespace(
                persona=state.persona,
                relevant_memories=memories,
                yesterday_summary=yesterday,
            )
            try:
                result = await asyncio.to_thread(
                    day_planner_run,
                    proxy,
                    {
                        "current_time": f"{next_date} 00:00",
                        "places": None,
                        "persona_name": state.persona_name,
                        "mode": "next_day",
                        "current_location_id": state.position.location_id,
                        "handoff_context": _handoff_context(state),
                        "upcoming_events": self.event_manager.snapshot(next_date, "00:00").get("upcoming", []),
                    },
                )
                plan = result.get("day_plan", [])
                if plan:
                    return state, plan
            except Exception as exc:
                logger.error(
                    "[WorldEngine] next-day plan failed for '%s': %s",
                    state.persona_name, exc,
                )
            # A failed planner must not strand the agent. Reusing the previous
            # schedule is the safest fallback because it preserves continuity.
            return state, state.day_plan

        planned = await asyncio.gather(
            *[_plan_next_day(state) for state in self.registry.all_states()],
            return_exceptions=False,
        )

        self.sim_start_date = next_date
        self.sim_start_hhmm = "00:00"
        self._day_index += 1
        self._recent_convs.clear()
        self._in_range.clear()
        self._last_obs.clear()
        self._tick_observations.clear()
        self._applied_event_effects.clear()
        self.world.history.clear()

        for state, plan in planned:
            state.day_plan = plan
            state.day_archived = False
            state.conversation_count = 0
            state.replan_count = 0
            state.last_conversation_partner = None
            state.conversation_start_tick = 0
            state.active_conversation = None
            state.paused = False
            if state.manager:
                state.manager.begin_new_day(plan)
                action = state.manager.tick(self.world.tick)
                state.position = state.manager.position
                state.current_action = action.model_dump() if action else None

        try:
            self._apply_event_opportunities(next_date)
        except Exception as exc:
            logger.exception("[WorldEngine] unable to apply events for %s", next_date)
            handoff_warnings.append(f"event overlay failed: {exc}")

        self._sync_to_world_state()
        try:
            save_checkpoint(
                self.world,
                self.registry,
                self.world.tick,
                engine_state=self.checkpoint_state(),
            )
            prune_checkpoints(keep_last=KEEP_LAST)
        except Exception as exc:
            logger.exception("[WorldEngine] unable to checkpoint handoff at tick %d", self.world.tick)
            handoff_warnings.append(f"handoff checkpoint failed: {exc}")

        return {
            "type": "day_initialized",
            "phase": "initialized",
            "date": next_date,
            "day": self._day_index + 1,
            "cancelled_conversations": cancelled,
            "archived_agents": sum(bool(result) for result in archive_results),
            "warnings": handoff_warnings,
            "agents": {
                state.agent_id: {
                    "name": state.persona_name,
                    "color": state.color,
                    "position": state.position.model_dump(),
                    "current_action": state.manager.current_action.model_dump()
                    if state.manager and state.manager.current_action else None,
                    "activity": state.manager.current_action.description
                    if state.manager and state.manager.current_action else "Idle",
                    "paused": state.paused,
                    "in_conversation": False,
                    "in_last_action": state.manager.is_last_action if state.manager else False,
                    "energy_level": state.energy_level,
                    "emotion_state": state.emotion_state,
                    "conversation": None,
                }
                for state in self.registry.all_states()
            },
            "recent_conversations": [],
        }

    # ------------------------------------------------------------------ #
    # Main run loop
    # ------------------------------------------------------------------ #

    async def run(
        self,
        max_tick: int = 1440,
        on_tick: Optional[callable] = None,
        tick_speed: Optional[float] = None,
        tick_lock: Optional[asyncio.Lock] = None,
    ) -> None:
        """Run simulation until world.tick reaches max_tick (absolute).

        If ``on_tick`` is provided, it is called with each tick's snapshot dict
        (after the tick is processed). This is used by Odin.py to broadcast
        state to WebSocket clients.

        ``tick_speed`` is a speed multiplier (1.0 = configured real-time, 2.0 =
        double speed). When None, the value from config (env/CLI) is used.
        """
        dynamic_speed = tick_speed is None
        initial_speed = _cfg.TICK_SPEED if dynamic_speed else tick_speed
        effective_sleep = _cfg.REAL_SECONDS_PER_TICK / max(initial_speed, 1e-9)
        logger.info("[WorldEngine] ========== STARTING SIMULATION ==========")
        logger.info(
            "[WorldEngine] %d agents, max tick %d (current=%d), %.1f real sec/tick (speed=%.1fx)",
            len(self.registry), max_tick, self.world.tick, effective_sleep, initial_speed,
        )

        while self.world.tick < max_tick:
            if tick_lock is None:
                result = await self.run_tick()
            else:
                async with tick_lock:
                    result = await self.run_tick()
            if not result:
                logger.info("[WorldEngine] simulation complete at tick %d", self.world.tick)
                break

            # Invoke callback if provided
            if on_tick is not None:
                try:
                    await on_tick(result)
                except Exception:
                    logger.exception("[WorldEngine] on_tick callback failed; simulation will continue")

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

            # Read the mutable runtime speed on every iteration so UI controls
            # take effect without restarting a day-long run.
            active_speed = _cfg.TICK_SPEED if dynamic_speed else tick_speed
            await asyncio.sleep(_cfg.REAL_SECONDS_PER_TICK / max(active_speed, 1e-9))

        logger.info("[WorldEngine] ========== SIMULATION ENDED ==========")
        if on_tick is not None:
            try:
                await on_tick({"type": "simulation_ended", "tick": self.world.tick})
            except Exception:
                logger.exception("[WorldEngine] final on_tick callback failed")

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
        "--start-date", default=_cfg.SIM_START_DATE,
        help="Simulation start date (default: from SIM_START_DATE / 2026-07-03)",
    )
    parser.add_argument(
        "--start-time", default="00:00",
        help="Simulation start time HH:MM (default: 00:00)",
    )
    parser.add_argument(
        "--resume", nargs="?", const=None, default=False,
        help="Resume from checkpoint. Pass a tick number, or no arg to prompt.",
    )
    # Register all simulation toggles (--tick-speed, --reflex-llm,
    # --perception-radius-px, --real-seconds-per-sim-minute, etc.)
    _cfg.add_cli_arguments(parser)
    args = parser.parse_args()

    # Apply CLI overrides onto config (CLI > .env > built-in defaults).
    _cfg.apply_overrides(**_cfg.overrides_from_args(args))
    logger.info("[WorldEngine] active settings:\n%s", _cfg.describe_settings())

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
                    world, registry, checkpoint_state = load_checkpoint(
                        resume_tick, engine.resolver, return_metadata=True,
                    )
                    engine.world = world
                    engine.registry = registry
                    engine.restore_checkpoint_state(checkpoint_state)
                    await engine.resume_pending_conversations()
                    logger.info(
                        "[WorldEngine] resumed from tick %d — %d agents",
                        resume_tick, len(registry),
                    )
                    day_end_tick = ((engine.world.tick // (24 * 60)) + 1) * (24 * 60)
                    await engine.run(max_tick=day_end_tick)
                    save_history(engine.world, engine.sim_start_date)
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
                world, registry, checkpoint_state = load_checkpoint(
                    resume_tick, engine.resolver, return_metadata=True,
                )
                engine.world = world
                engine.registry = registry
                engine.restore_checkpoint_state(checkpoint_state)
                await engine.resume_pending_conversations()
                logger.info(
                    "[WorldEngine] resumed from tick %d — %d agents",
                    resume_tick, len(registry),
                )
                day_end_tick = ((engine.world.tick // (24 * 60)) + 1) * (24 * 60)
                await engine.run(max_tick=day_end_tick)
                save_history(engine.world, engine.sim_start_date)
                return

        # Fresh start
        engine = WorldEngine(
            sim_start_date=args.start_date,
            sim_start_hhmm=args.start_time,
        )
        await engine.initialize()
        for day in range(args.days):
            logger.info("[WorldEngine] --- Day %d ---", day + 1)
            day_end_tick = ((engine.world.tick // (24 * 60)) + 1) * (24 * 60)
            await engine.run(max_tick=day_end_tick)
            if day < args.days - 1:
                next_date = (
                    datetime.fromisoformat(engine.sim_start_date) + timedelta(days=1)
                ).strftime("%Y-%m-%d")
                await engine.handoff_to_next_day(next_date)
            else:
                save_history(engine.world, engine.sim_start_date)

    asyncio.run(_main())
