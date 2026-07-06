# Valhalla — Generative Agents Simulation — Handoff Document

## 1. Project Overview

**Goal:** A generative-agents simulation (Park et al. "Generative Agents" style) — multiple LLM-driven personas living in a shared simulated world, each with memory, perception, planning, and reactive behavior. Personas perceive their surroundings, retrieve relevant memories, plan their day (coarse → hourly → fine), react to observed events, and act — all coordinated through a central simulation clock.

**Tech stack:**
- Python
- **LangGraph** as the orchestrator (both for the per-agent cognitive pipeline and the day-planning subgraph)
- **Pydantic** for all schemas that cross a tick/graph boundary or get logged/serialized
- **Google Gemini** (via `google-genai` SDK) as the LLM backend, wrapped in a custom client with model-tier fallback, key rotation, and retry logic
- SQLite (planned, not yet built) for memory persistence and action/tick logging
- Custom tile-based map system (`generate_map_tiles.py`, `nav_graph.py`, `pathfinder.py`, `pixel_pathfinder.py`) — pre-existing, not part of this build sequence
- `server.py`, `render_buildings.py`, `odin.py` — pre-existing, frontend/rendering related, not touched in this build sequence

**Core architecture — two-tier graph design:**
- **Tier 1** (`day_planner.py`, already built by user): a LangGraph subgraph doing coarse→hourly→fine plan decomposition with a validation/retry loop (conflict detected → loop back to coarse).
- **Tier 2** (`tick_graph.py`, built this session): a per-agent LangGraph subgraph — perceive → retrieve_memories → react → (conditionally) day_planner → write_back_memory — invoked once per agent per tick.
- **Orchestration model:** ONE centralized `WorldEngine` (not yet built) owns the global sim clock and drives a **three-phase lockstep tick**:
  1. **Decide phase (parallel):** freeze an immutable `WorldSnapshot`; run every ready agent's `tick_graph` concurrently via `asyncio.gather()` against that same snapshot (safe because these are I/O-bound LLM calls, not CPU-bound, and nobody mutates the snapshot).
  2. **Resolve phase (sequential):** apply each agent's proposed `TickResult` to the real `WorldState` one at a time; arbitrate resource conflicts.
  3. **Broadcast phase:** compute observability (who saw what) and push observations into other agents' memory streams / interrupt flags for next tick.
- This deliberately rejects both (a) naive sequential per-agent processing (causes agents to perceive an artificially "younger" or "older" world depending on iteration order) and (b) fully free-running async listeners per agent (non-deterministic, unreplayable, real race conditions on shared state). Lockstep with a frozen snapshot was chosen specifically to keep the sim deterministic and replayable while still parallelizing the expensive (LLM) part.

---

## 2. Current State — Completed Modules

### `backend/src/core/log.py` — *pre-existing, built by user, not touched this session*
Centralized logging setup. Exposes `setup_logging(run_id: str = None)` (call once, early, in the real entrypoint) and `get_logger(__name__)` (call everywhere else). Lazily calls `setup_logging()` with defaults if a module is imported before it's been called, so early-import order accidents don't produce silence.

### `backend/src/llm/gemini_client.py` — *pre-existing, built by user, not touched this session*
Single-turn wrapper around Gemini (`google-genai` SDK). No session/chat memory.

```python
def call_gemini(
    system_prompt: str,
    user_prompt: str,
    schema: Type[BaseModel],
    complexity: str = "simple",   # key into MODEL_TIERS, e.g. "simple" | "complex"
    temperature: float = ...,
) -> BaseModel:                    # instance of `schema`, parsed & validated
    ...
```
Behavior: picks a model chain by `complexity`; falls back through the chain on overload/rate-limit; rotates across multiple API keys when a whole chain is quota-exhausted; retries transient errors (429/500/503/504) with exponential backoff + jitter; **fails fast** (raises `google.genai.errors.APIError`) on non-retryable errors (400/403/404) — these are treated as bugs, not congestion. Raises `ValueError` if `complexity` isn't a valid `MODEL_TIERS` key. Raises `AllModelsFailedError` if every model/key combination failed.

> **Known naming inconsistency (flagged, unresolved):** the module's own docstring shows both `call_gemini(...)` usage AND a conflicting import line `from src.llm.client import gemini`. All code written this session assumes the former (`from src.llm.gemini_client import call_gemini`) since it matches the actual filename in the tree. **Verify this before running anything that imports it.**

### `backend/src/agents/day_planner.py` — *pre-existing, built by user, not touched this session, exact interface unknown to this handoff's author*
Tier-1 LangGraph subgraph. Pipeline: `generate_coarse_plan → decompose_hourly → decompose_fine → validate_plan`, with a conditional loop back to `generate_coarse_plan` on conflict detection, else `END`. State fields (from user's own notes, not verified against source): `persona, relevant_memories, yesterday_summary, current_time, coarse_plan, hourly_plan, fine_plan, conflict_detected`. **The exact compiled-graph export name and callable signature were never confirmed in this session** — `tick_graph.py` (below) does NOT hardcode a call into this file; it uses a dependency-injected adapter instead specifically because of this gap. See "Known Issues" (§5) and "Next Steps" (§6).

### `backend/src/core/world_state.py` — **built this session, Phase 1**
Canonical, mutable world state. Only `WorldEngine`'s resolve phase should mutate it; everything else reads a frozen `WorldSnapshot` instead.

```python
class AgentStatus(str, Enum):
    IDLE = "idle"
    ACTING = "acting"
    INTERRUPTED = "interrupted"

class Position(BaseModel):
    x: int
    y: int
    location_id: Optional[str] = None   # semantic zone, e.g. "cafeteria"

class CurrentAction(BaseModel):
    description: str
    start_tick: int
    end_tick: int
    target_location_id: Optional[str] = None
    target_object_id: Optional[str] = None
    was_reprioritized: bool = False
    def is_finished(self, tick: int) -> bool: ...

class AgentState(BaseModel):
    agent_id: str
    position: Position
    status: AgentStatus = AgentStatus.IDLE
    current_action: Optional[CurrentAction] = None
    last_updated_tick: int = 0

class ActionLogEntry(BaseModel):
    tick: int
    agent_id: str
    action: str
    timestamp: datetime

class WorldState(BaseModel):
    tick: int = 0
    agents: Dict[str, AgentState] = {}
    occupancy: Dict[str, Optional[str]] = {}   # resource_id -> holder agent_id | None
    history: List[ActionLogEntry] = []

    def register_agent(self, agent_id: str, position: Position) -> None: ...
    def get_agent(self, agent_id: str) -> AgentState: ...            # raises KeyError if unknown
    def all_agent_ids(self) -> List[str]: ...
    def register_resource(self, resource_id: str) -> None: ...        # idempotent
    def is_free(self, resource_id: str) -> bool: ...
    def occupy(self, resource_id: str, agent_id: str) -> bool: ...    # False if held by someone else; does NOT raise
    def release(self, resource_id: str) -> None: ...
    def release_all_for_agent(self, agent_id: str) -> None: ...
    def set_agent_action(self, agent_id: str, action: CurrentAction) -> None: ...  # also appends to history
    def move_agent(self, agent_id: str, position: Position) -> None: ...
    def interrupt_agent(self, agent_id: str) -> None: ...             # sets INTERRUPTED, releases resources
    def clear_agent_action(self, agent_id: str) -> None: ...
    def agents_ready_for_decision(self) -> List[str]: ...             # no action, OR finished, OR INTERRUPTED
    def advance_tick(self, minutes: int = 10) -> None: ...
```

### `backend/src/core/snapshot.py` — **built this session, Phase 1**
Frozen, read-only, point-in-time view of `WorldState`. Deliberately a **different class**, not just a deep copy — exposes zero mutating methods, so a tick graph can't accidentally corrupt resolve-phase assumptions.

```python
class WorldSnapshot(BaseModel):   # model_config = ConfigDict(frozen=True)
    tick: int
    agents: Dict[str, AgentState]      # deep-copied at snapshot time
    occupancy: Dict[str, Optional[str]]

    def get_agent(self, agent_id: str) -> AgentState: ...           # raises KeyError if unknown
    def all_agent_ids(self) -> List[str]: ...
    def other_agents(self, agent_id: str) -> List[AgentState]: ...
    def agents_near(self, agent_id: str, radius: int) -> List[AgentState]: ...   # Chebyshev distance
    def agents_at_location(self, location_id: str) -> List[AgentState]: ...
    def is_free(self, resource_id: str) -> bool: ...
    def holder_of(self, resource_id: str) -> Optional[str]: ...
    def free_resources(self) -> List[str]: ...

def take_snapshot(world: WorldState) -> WorldSnapshot: ...
```
**Contract:** call `take_snapshot()` exactly ONCE per tick, before the decide phase; pass the same instance into every agent's `asyncio.gather()`'d tick graph call that tick. Never take one snapshot per agent (reintroduces the exact ordering bug lockstep exists to avoid).

### `backend/src/agents/perceive.py` — **built this session, Phase 2**
Pure function, no LLM call, no side effects — converts a `WorldSnapshot` into what one agent currently observes. Already fully functional (not a stub) since it only depends on Phase 1 primitives; with a single agent registered it naturally returns `[]`.

```python
DEFAULT_PERCEPTION_RADIUS = 5   # tiles; tune to actual map scale

class Observation(BaseModel):
    tick: int
    observer_id: str
    subject_agent_id: str
    description: str            # e.g. "Gurnoor is eating breakfast"
    location_id: Optional[str] = None
    distance: int = 0            # Chebyshev tiles; lower = more salient

def perceive(
    snapshot: WorldSnapshot,
    agent_id: str,
    radius: int = DEFAULT_PERCEPTION_RADIUS,
) -> List[Observation]: ...     # sorted nearest-first; raises KeyError if agent_id not in snapshot
```

### `backend/src/agents/react.py` — **built this session, Phase 2**
Decides whether an agent should continue its current action or replan.

```python
class ReactionDecision(BaseModel):
    should_replan: bool
    reason: str

def decide_reaction(
    agent_id: str,
    persona: Dict[str, Any],
    current_action: Optional[CurrentAction],
    observations: List[Observation],
    memories: List[str],
    tick: int,
) -> ReactionDecision: ...
```
Logic:
- `current_action is None` → `should_replan=True` (no LLM call)
- `current_action.is_finished(tick)` → `should_replan=True` (no LLM call)
- `observations` empty → `should_replan=False` (no LLM call)
- mid-action AND observations present → **only** case that calls Gemini (`complexity="simple"`, `temperature=0.3`, schema=`ReactionDecision`)

**Fail-safe policy (deliberate, differs from `gemini_client.py`'s fail-fast stance):** if the Gemini call raises, `react.py` catches it, logs via `logger.exception(...)`, and returns `ReactionDecision(should_replan=False, reason="LLM call failed, defaulting to continue")` rather than propagating — an LLM outage should degrade to "keep going," not crash the tick or force spurious replans everywhere.

### `backend/src/agents/tick_graph.py` — **built this session, Phase 2**
The per-agent LangGraph subgraph. Compiled once at startup, reused every tick/agent.

```python
class DayPlannerRequest(BaseModel):
    agent_id: str
    persona: Dict[str, Any]
    relevant_memories: List[str]
    yesterday_summary: str
    current_time: datetime

class DayPlannerResult(BaseModel):
    action_description: str
    duration_minutes: int
    target_location_id: Optional[str] = None
    target_object_id: Optional[str] = None

DayPlannerFn = Callable[[DayPlannerRequest], DayPlannerResult]   # dependency-injected adapter, see §5

class MemoryStreamProtocol:      # structural contract only, duck-typed
    def add_memory(self, agent_id: str, content: str, importance: Optional[int] = None) -> None: ...
    def retrieve_memories(self, agent_id: str, query: str, k: int = 5) -> List[str]: ...

class TickOutcome(str, Enum):
    KEEP_CURRENT = "keep_current"
    NEW_ACTION = "new_action"

class TickResult(BaseModel):
    agent_id: str
    tick: int
    outcome: TickOutcome
    action: Optional[CurrentAction] = None
    reaction: Optional[ReactionDecision] = None

class TickState(TypedDict, total=False):   # internal scratch state, never leaves this module
    agent_id: str
    tick: int
    persona: Dict[str, Any]
    yesterday_summary: str
    world_snapshot: WorldSnapshot
    current_action: Optional[CurrentAction]
    observations: List[Observation]
    memories: List[str]
    reaction: ReactionDecision
    result: TickResult

def build_initial_state(agent_id, persona, yesterday_summary, world_snapshot) -> TickState: ...

def build_tick_graph(
    memory_stream: Optional[MemoryStreamProtocol] = None,   # defaults to logging no-op if omitted
    day_planner_fn: Optional[DayPlannerFn] = None,           # raises NotImplementedError if omitted and reached
):
    ...
    return compiled_graph   # has .ainvoke(TickState) -> dict with "result": TickResult

async def run_tick(
    tick_graph, agent_id, persona, yesterday_summary, world_snapshot,
) -> TickResult: ...   # convenience wrapper: build_initial_state + ainvoke + unwrap "result"
```
Graph wiring: `START → perceive → retrieve_memories → react →`(conditional: `should_replan` → `day_planner`, else → `keep_current`)` → write_back_memory → END`.

Intended call pattern from `world_engine.py`'s decide phase:
```python
results = await asyncio.gather(*[
    run_tick(tick_graph, aid, personas[aid], summaries[aid], snap)
    for aid in ready_agent_ids
])
```

---

## 3. File / Folder Structure (current)

```
Valhalla/
├── .venv/
├── backend/
│   ├── data/
│   │   ├── environment/
│   │   ├── personalities/
│   │   └── output/
│   │       ├── logs/
│   │       └── runs/
│   ├── prompt_templates/
│   ├── src/
│   │   ├── agents/
│   │   │   ├── day_planner.py        # Tier-1 subgraph: coarse→hourly→fine plan + conflict retry loop (pre-existing)
│   │   │   ├── perceive.py           # [DONE] snapshot -> List[Observation], pure fn, real spatial query
│   │   │   ├── react.py              # [DONE] decide continue-vs-replan, cheap paths + 1 LLM branch
│   │   │   ├── tick_graph.py         # [DONE] per-agent LangGraph subgraph (Tier 2), day_planner/memory injected
│   │   │   ├── memory_stream.py      # [NOT BUILT] owned by teammate; interface agreed, stub not yet written
│   │   ├── core/
│   │   │   ├── config.py             # project config (pre-existing)
│   │   │   ├── log.py                # centralized logging setup (pre-existing)
│   │   │   ├── world_state.py        # [DONE] canonical mutable WorldState/AgentState/CurrentAction schema
│   │   │   ├── snapshot.py           # [DONE] frozen WorldSnapshot + take_snapshot()
│   │   ├── llm/
│   │   │   ├── gemini_client.py      # call_gemini() wrapper: model tiers, key rotation, retry/backoff (pre-existing)
│   │   ├── tiles/
│   │   │   ├── generate_map_tiles.py # pre-existing, map generation, untouched
│   │   │   ├── nav_graph.py          # pre-existing, navigation graph, untouched
│   │   ├── pathfinder.py             # pre-existing, untouched
│   │   ├── pixel_pathfinder.py       # pre-existing, untouched
│   │   ├── render_buildings.py       # pre-existing, untouched
│   │   ├── server.py                 # pre-existing, untouched
│   │   ├── engine/                   # [NOT BUILT YET — Phase 3]
│   │   │   ├── world_engine.py       # top-level orchestrator: sim clock + 3-phase tick loop
│   │   │   ├── scheduler.py          # picks active agents this tick (wraps agents_ready_for_decision())
│   │   │   ├── resolver.py           # applies proposed actions to WorldState, arbitrates conflicts
│   │   │   ├── event_bus.py          # computes observability, dispatches into memory/interrupt flags
│   │   ├── persistence/              # [NOT BUILT YET — Phase 4]
│   │   │   ├── db.py                 # SQLite connection/session management
│   │   │   ├── models.py             # memory rows, action log, tick log table schemas
│   ├── odin.py                       # pre-existing, untouched
│   ├── requirements.txt
│   └── temp.py                       # scratch file, pre-existing
├── frontend/
└── .gitignore
```

---

## 4. Key Design Decisions

1. **One centralized `WorldEngine`, not one engine per agent.** Per-agent engines would drift independently (no shared clock), can't broadcast observations (no shared bus), can't arbitrate shared-resource contention, and multiply debugging complexity by N independent loops. Matches Park et al.'s design: a shared world state all agents read/write through in defined order.

2. **Lockstep two-phase (really three-phase) tick, not free-running async listeners.** Naive sequential (`for agent in agents: agent.act()`) creates an ordering bug — later agents perceive a world "younger" than it should be within the same tick. Full event-driven listeners create real race conditions on shared state and non-determinism (unreplayable interleaving). The frozen-snapshot + parallel-decide / sequential-resolve / sequential-broadcast split gets the actual speed win (LLM calls are I/O-bound, parallelizable) while keeping the sim deterministic and replayable — replay/debugging was treated as a first-class requirement, not an afterthought, given how hard multi-agent LLM bugs are to reproduce otherwise.

3. **`WorldSnapshot` is a distinct class from `WorldState`, not a deep copy wearing a trenchcoat.** Exposes zero mutating methods so nothing downstream can physically call `.occupy()`/`.set_agent_action()` on it — a stronger guarantee than a docstring warning. Nested `AgentState` objects are deep-copied at snapshot time so even a stray in-place mutation on the copy can't leak into the real `WorldState` or another agent's view.

4. **Pydantic vs TypedDict split:** Pydantic `BaseModel` used for anything that crosses a tick/graph boundary, gets validated, logged, or persisted (`WorldState`, `WorldSnapshot`, `CurrentAction`, `Observation`, `ReactionDecision`, `TickResult`, `DayPlannerRequest/Result`). `TypedDict` used only for `TickState` — scratch state internal to `tick_graph.py`'s LangGraph execution that never leaves the module.

5. **`occupy()` returns `bool` instead of raising.** Resource contention (two agents wanting the same chair) is routine, expected behavior, not an exceptional case — `resolver.py` (Phase 3) is expected to check the return value and arbitrate, not wrap calls in try/except.

6. **`register_resource()` required before a resource can be occupied/checked.** Turns a typo'd resource id into a loud warning instead of silently treating an unregistered resource as perpetually free.

7. **Dependency injection for `day_planner_fn` and `memory_stream` in `tick_graph.py`**, rather than hardcoded imports. Chosen because (a) the memory module is owned by a teammate and doesn't exist yet, and (b) `day_planner.py`'s exact compiled-graph export name/signature was never confirmed in this session (its own docstring has an internal inconsistency — see §5). This keeps `tick_graph.py` fully buildable/testable today; the real integration is a ~10-line adapter function written once either module is confirmed.

8. **`react.py` fails safe, `gemini_client.py` fails fast — deliberately different policies at different layers.** `gemini_client.py` raising on 400/403/404 is correct (client-level bug signal). `react.py` swallowing a Gemini failure and defaulting to `should_replan=False` is a product decision: an LLM outage degrading to "an agent finishes what it was doing" is preferable to it crashing the tick or forcing mass spurious replanning.

9. **Chebyshev (square-radius) distance for `agents_near()` / perception**, not Euclidean, chosen as the default because it's a single `max(dx, dy)` comparison and matches typical tile-grid "box around me" visibility. Isolated to one method on `WorldSnapshot`, so swapping metrics later is a one-line change in one place — `perceive.py` never reimplements this math itself.

10. **Gemini import path assumption:** all new code imports `from src.llm.gemini_client import call_gemini`, matching the actual filename. Not yet verified against the real file (flagged in §5).

---

## 5. Known Issues / Open Questions

- **`gemini_client.py` naming inconsistency, unresolved:** its own docstring shows both `call_gemini(...)` usage and a conflicting `from src.llm.client import gemini` import example. All code this session assumes `from src.llm.gemini_client import call_gemini` is correct. **Must be verified against the actual file before running `react.py`.**
- **`day_planner.py`'s exact compiled-graph interface was never confirmed** — only the state-field list from the user's own prior notes (`persona, relevant_memories, yesterday_summary, current_time, coarse_plan, hourly_plan, fine_plan, conflict_detected`) is known; the compiled graph's export name, and whether it's invoked via `.invoke()` and what dict shape it expects/returns, were never shown to the assistant. `tick_graph.py` sidesteps this via the `DayPlannerFn` adapter contract, but the actual adapter function still needs to be written once this is confirmed.
- **`memory_stream.py` does not exist yet.** Owned by a teammate. Interface was agreed conversationally (`add_memory(agent_id, content, importance=None)`, `retrieve_memories(agent_id, query, k=5) -> List[str]`) but no stub file has actually been committed. `tick_graph.py` currently falls back to an internal `_NullMemoryStream` (logs, returns `[]`) if none is injected — functional for single-agent testing, but no real memories accumulate until this lands.
- **Coordinate system assumption unconfirmed.** `Position(x: int, y: int, location_id: Optional[str])` was chosen as a reasonable default for tile-grid coordinates, but this was never checked against `nav_graph.py`/`pathfinder.py`'s actual coordinate/node-id scheme. If those use something else (pixel coords, graph node ids, etc.), `Position` and `perceive.py`'s distance math need to match exactly before Phase 5 (real multi-agent perception) — flagged twice in conversation, never answered.
- **`DEFAULT_PERCEPTION_RADIUS = 5` (tiles) is a placeholder**, not derived from actual map scale — tune once real map dimensions are known.
- **Simulated-time ↔ wall-clock mapping is unresolved.** `tick_graph.py`'s `day_planner_node` currently calls `datetime.now()` for `DayPlannerRequest.current_time` with an inline comment flagging this should be replaced by whatever mapping from `WorldState.tick` (an int, minutes-since-sim-start) to an actual simulated `datetime` the project uses (if any).
- **None of the Phase 1/2 code has been executed/tested in a real environment** — the sandbox used to write this code has no network access and no `pydantic`/`langgraph` installed, so `pip install` and actual test runs weren't possible. Each file has an `if __name__ == "__main__":` sanity check block, but **none have been run**. Run all three (`world_state.py`'s implicit checks via `snapshot.py`, `snapshot.py`, `perceive.py`, `react.py` cheap-path checks, `tick_graph.py`) as the very first step in the new chat/environment, before building further.
- **Persona schema is undefined.** `persona: Dict[str, Any]` is used loosely throughout (`tick_graph.py`, `react.py`) as a placeholder — no `Persona` Pydantic model has been designed yet, despite `data/personalities/` existing in the folder structure. This should probably be formalized before or during Phase 3.
- **`ReactionDecision`'s LLM branch has never been exercised or reviewed for prompt quality** — only the three cheap paths were sanity-checked. The Gemini-calling branch in `react.py` (`_llm_reaction`) has not been tested against a real API key/response.

---

## 6. Next Steps (sequential, exact)

### Step 1 — Verify environment + run existing sanity checks
1. Confirm `gemini_client.py`'s real exported callable name and fix the import in `react.py` if it isn't `call_gemini` from `src.llm.gemini_client`.
2. Add empty `__init__.py` to `src/`, `src/core/`, `src/agents/`, `src/llm/`, `src/engine/` (new), `src/persistence/` (new) if not already present.
3. Run `python -m src.core.snapshot`, `python -m src.agents.perceive`, `python -m src.agents.react`, `python -m src.agents.tick_graph` (this last one requires `langgraph` installed and will make a real Gemini call only if you push it into the LLM branch — the provided demo doesn't). Confirm all pass.

### Step 2 — Resolve the two open integration gaps from §5
4. Get `day_planner.py`'s actual compiled graph export name + exact input/output shape (either paste the file or state it precisely). Write the adapter function matching `DayPlannerFn = Callable[[DayPlannerRequest], DayPlannerResult]` and pass it into `build_tick_graph(day_planner_fn=...)`.
5. Confirm coordinate system used by `nav_graph.py`/`pathfinder.py`; adjust `Position` in `world_state.py` if it doesn't match plain `(x: int, y: int)`.
6. Either get the real `memory_stream.py` from the teammate, or write the agreed-upon stub (`add_memory`, `retrieve_memories` as no-ops) yourself as a placeholder so `tick_graph.py` can be exercised beyond the null default.

### Step 3 — Phase 3: World Engine (build these three/four files, in order)
7. **`backend/src/engine/scheduler.py`** — thin wrapper: `def get_ready_agents(world: WorldState) -> List[str]: return world.agents_ready_for_decision()`. Keep it this simple for now (single/dual agent); this is the natural place to later add priority ordering or throttling once scaling to 5 personas.
8. **`backend/src/engine/resolver.py`** — Phase 3 version is a pass-through: given a `List[TickResult]` from the decide phase, apply each to the real `WorldState` sequentially: if `outcome == NEW_ACTION`, call `world.set_agent_action(agent_id, result.action)` (and `world.occupy(...)` if the action has a `target_object_id`/`target_location_id`); if `outcome == KEEP_CURRENT`, just clear any `INTERRUPTED` status back to `ACTING` (no-op otherwise). Real conflict arbitration (two agents wanting the same resource) is explicitly deferred to Phase 5 — do not build it yet, there's nothing to conflict over with one agent.
9. **`backend/src/engine/event_bus.py`** — Phase 3 version is a no-op stub: `def broadcast(world: WorldState, results: List[TickResult]) -> None: pass`. Real observability/broadcast logic deferred to Phase 5.
10. **`backend/src/engine/world_engine.py`** — the async loop:
    - `__init__(self, world: WorldState, tick_graph, personas: Dict[str, dict], yesterday_summaries: Dict[str, str])` — stores references, does NOT build the tick graph itself (build it once outside and pass in).
    - `async def run_tick(self) -> None:` implementing the three phases: `snapshot = take_snapshot(self.world)` → `ready = scheduler.get_ready_agents(self.world)` → `results = await asyncio.gather(*[run_tick(self.tick_graph, aid, self.personas[aid], self.yesterday_summaries[aid], snapshot) for aid in ready])` → `resolver.apply(self.world, results)` → `event_bus.broadcast(self.world, results)` → `self.world.advance_tick()`.
    - `async def run(self, num_ticks: int) -> None:` loops `run_tick()` `num_ticks` times, logging tick number + each agent's resulting action.
11. **Wire `backend/main.py`**: load one persona from `data/personalities/`, build a `WorldState` with `register_agent`, build `tick_graph` via `build_tick_graph(...)` with real/stub dependencies from Step 2, construct `WorldEngine`, call `await engine.run(num_ticks=...)`.

**Done-when for Phase 3:** running `main.py` ticks one agent through a simulated day, logs show sensible sequential behavior (wake → plan → act → repeat), no crashes, no unresolved `NotImplementedError` from the day_planner adapter.

### Step 4 — Phase 4: Persistence / Replay Log (do this BEFORE adding a second agent)
12. **`backend/src/persistence/models.py`** — table schemas: `memory` (if not owned entirely by teammate's module), `action_log` (tick, agent_id, action, resulting_state_diff), `tick_log`.
13. **`backend/src/persistence/db.py`** — SQLite connection/session management, shared by `memory_stream.py` and `world_engine.py`'s tick logging.
14. Wire `world_engine.py` to write each tick's `TickResult`s to `action_log` before/as it applies them via the resolver.
15. **Done-when:** re-running a logged sequence of ticks against the same starting `WorldState` produces identical output (determinism check) — this is the point of doing persistence before scaling to multiple agents, since replay is your main debugging tool once agent count > 1.

### Step 5 — Phase 5: Second agent, real resolver + event bus
16. Add a second persona to `main.py`'s setup.
17. Flesh out `resolver.py`'s real conflict logic: if two `TickResult`s in the same tick both target the same `resource_id` (via `target_location_id`/`target_object_id`), decide precedence (first-come by gather-completion order, or a priority score) and reject/modify the loser's action (set `was_reprioritized=True` on the losing `CurrentAction`).
18. Flesh out `event_bus.py`'s real broadcast logic: after resolve phase, for each acted agent, use `WorldSnapshot`-equivalent-but-post-resolve logic (or re-snapshot) to find nearby agents via `agents_near()`, and for each observer within range, either call `memory.add_memory(observer_id, f"observed {actor_id} {action.description}")` and/or `world.interrupt_agent(observer_id)` depending on significance (this is where `react.py`'s LLM branch actually starts getting exercised in practice).
19. Verify `asyncio.gather` in the decide phase is genuinely running agents' LLM calls concurrently (temporary logging/timing check).
20. **Done-when:** agent A does something observable; agent B (nearby) picks it up via `retrieve_memories` next tick and reacts to it (even trivially, e.g. logs "noticed X, decided not to interrupt").

### Step 6 — Phase 6: Scale to all 5 personas
21. Only after Step 5's two-agent loop is stable and replayable. Architecture doesn't change — just increases `scheduler.get_ready_agents()`'s candidate pool and stresses `resolver.py`'s contention handling further. Load `data/personalities/` fully instead of one/two hardcoded personas.

---


