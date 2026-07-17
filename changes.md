# Valhalla — Change Log

This file tracks every change made while adding long-term memory, spatial
awareness, the brain/body architecture, the frontend UX pass, and the
configuration/toggles layer. Newest entries at the top.

---

## At a Glance — What This Project Added

- **Long-term memory (hybrid).** A real cross-day memory: an offline
  keyword+recency retriever over the daily archives (default) plus an optional
  Qdrant + embeddings backend behind the same interface. Memory is now actually
  fed into day-planning and conversations (it used to be empty).
- **Spatial awareness (50px).** Agents perceive who/what is within a 50-pixel
  circle each tick (0 AI), record what they see, and start conversations by true
  proximity instead of exact-location matching — with crowd/pair rules.
- **Brain/body architecture.** A cognition layer (perceive → recall → decide)
  that commands a separate motor layer (the action executor). Cheap by default:
  reflex AI is OFF and gated behind a whitelist + rate-limit + budget check.
- **Budget safety.** A governor counts every AI call; conversations fold their
  own replan decision in (killing the old 8-calls-per-meeting); quiet ticks cost
  zero AI (proven by a headless stress test).
- **Frontend storytelling.** Click-to-focus with camera centring, live activity
  labels, a colour legend, a campus conversation feed, and day/night map blending.
- **Config + slower clock.** One env+CLI toggle layer; default day slowed to
  ~96 real minutes to ease free-tier rate limits.

### Toggle reference (`.env` / CLI flag / default)

| Controls | `.env` | CLI flag | Default |
|---|---|---|---|
| Reflex AI thinking | `SIM_REFLEX_LLM` | `--reflex-llm` | off |
| Speed multiplier | `SIM_TICK_SPEED` | `--tick-speed` | 1.0 |
| Real sec / sim-minute | `SIM_REAL_SECONDS_PER_SIM_MINUTE` | `--real-seconds-per-sim-minute` | 4.0 (~96 min/day) |
| Proximity radius (px) | `SIM_PERCEPTION_RADIUS_PX` | `--perception-radius-px` | 50 |
| Perception on/off | `SIM_PERCEPTION_ENABLED` | `--perception` | on |
| Memory backend | `SIM_MEMORY_BACKEND` | `--memory-backend` | keyword |
| Conversations/agent/day | `SIM_MAX_CONVERSATIONS_PER_AGENT` | `--max-conversations` | 5 |
| Replans/agent/day | `SIM_MAX_REPLANS_PER_AGENT_PER_DAY` | `--max-replans` | 3 |
| AI calls/real-hour ceiling | `SIM_LLM_HOURLY_CEILING` | `--llm-hourly-ceiling` | 0 (none) |

---

## Post-integration fixes

### Real pathfinding + no repeated conversations  ✅
- **Pathfinder was always failing** (`frontend/path.png` not found — the file is
  under `frontend/public/`), so every move used a straight-line fallback. Fixed:
  the pathfinder now locates `path.png` under `frontend/public` (or `frontend`,
  or `frontend/dist`), and snaps a route's start/end to the nearest walkable
  pixel so building doors/interiors that sit just off the path grid still route.
  Verified: 138k walkable pixels load; a hostel→LHC move now yields a real
  662-point road route instead of a 2-point line.
- **Full door-to-door walk:** a move now goes interior spot → building door →
  (pathfinder road) → destination door → random interior spot at the destination.
- **No back-to-back conversations with the same person:** two agents can't chat
  again until at least one of them has spoken with someone else in between
  (tracked via `last_conversation_partner`). This stops the loop where Parv and
  Gurnoor re-chatted every cooldown window (5 times in a row) and appeared stuck.


### Door-to-door movement that actually animates  ✅
- **Root cause of the "teleport":** agents departed from a non-walkable interior
  pixel, so the pathfinder returned nothing and fell back to a 2-point line — the
  position then sat at the start and jumped to the end.
- **Fix:** movement is now routed **door → door** — from the current building's
  entry point, along the pathfinder's walkable route, to the destination's entry
  point, then settling inside the destination on arrival. The position is
  interpolated *between* path points by elapsed-time fraction, so the dot walks
  the whole route smoothly. Verified: a hostel→LHC move produced 40 distinct
  positions across the map (animates) instead of a single jump.
- **Start-of-day time (`SIM_START_TIME`, default now 08:00 in .env):** the sim
  clock now begins at the configured hour, so the day opens with students walking
  to class instead of everyone asleep at midnight — which is also why movement
  wasn't visible before (it was still night).


### Movement, building interiors, tighter chats, single-file long-term  ✅
- **Visible walking (was teleporting):** agents now traverse their whole path
  from start to destination proportionally over the move's duration, so the dot
  actually walks the route instead of snapping to the end.
- **Inside buildings, not on the doorstep:** loaded each building's bounding box
  (union of its decomposed polygon parts) and, on arrival, agents settle at a
  random pixel *inside* that footprint. Spawns are scattered inside the hostel
  too — no more everyone stacked on one entry pixel.
- **Never the same pixel:** a per-tick collision pass nudges any overlapping
  agents a few pixels apart, guaranteeing no two share a pixel.
- **Conversation radius = 20px:** split from perception. Agents now *see* others
  within 50px but only *start a chat* within 20px (`SIM_CONVERSATION_RADIUS_PX`).
- **Perception throttled:** runs every `SIM_PERCEPTION_EVERY_N_TICKS` (default 3).
- **Long-term = one file per agent:** replaced the per-day long-term dumps with a
  single compressed `Long_term_db/<agent>/memory.json` (a list of compact day
  records: summary + conversation summaries + counts). Legacy per-date files are
  auto-migrated into it (and removed) on startup. Short-term stays one file per
  agent and is compressed into long-term at day's end. Retrieval reads the single
  long-term file plus the current short-term day. Verified: gurnoor's long-term
  folder now holds only `memory.json`, and recall still returns results.


### Auto-reloader dropping the port + short-term hygiene + perception throttle  ✅
- **Server would not stay open:** the dev auto-reloader watched the whole
  project, and the sim writes a checkpoint + memory files every tick — so the
  reloader kept restarting the server and dropping port 8000 mid-run. Disabled
  reload for the sim server (it doesn't benefit from it) and it now prints the
  exact URL. Server-side serving verified correct (`/` → index.html, `.js` →
  application/javascript).
- **Item 3 — clean start:** a day's short-term file now holds *only the plan*
  when a run begins. On startup the run-time record (events / conversations /
  snapshots) is reset while the plan is kept and reused — so no stale data
  carries over and no LLM is spent (avoids the earlier freeze). Verified all
  agents report a pristine start.
- **Item 4 — one file per agent:** startup consolidates each agent to a single
  short-term file for the active date; any stray day files are pushed to
  long-term memory (no LLM) and removed. Verified: agents went from two files
  to one.
- **Item 5 — perception throttle:** perception now runs every N ticks
  (`SIM_PERCEPTION_EVERY_N_TICKS`, default 3) instead of every tick. Still 0-LLM.


### Blank page on the server port (JS MIME type)  ✅
- **Symptom:** `localhost:8000` served a blank page (only the map, or nothing),
  even though the server was healthy and ticking.
- **Cause:** on Windows the server handed `.js` files to the browser with the
  wrong content type, so the browser refused to run the app's module script.
- **Fix:** the server now registers correct MIME types for `.js`/`.mjs`/`.css`/
  `.svg`. Verified `.js => application/javascript`, `.css => text/css`.

### Simulation date as input; no more end-of-day replay  ✅
- The run date is now a real setting (`SIM_START_DATE`, default 2026-07-03),
  usable from `.env` or the CLI. Startup looks up that date's plans in the
  short-term store and **reuses them if present, generates them only if absent**.
- The web server no longer resumes a stale *end-of-day* checkpoint (which just
  replayed the finished day). It only resumes genuine *mid-day* checkpoints;
  otherwise it starts the configured date fresh (reusing plans, 0 LLM).

### `.env` now lists every toggle  ✅
- Appended all simulation settings (date, reflex AI, clock/speed, perception
  radius, memory backend, budget caps) to `.env` with explanatory comments, so
  they're visible and editable in one place. Verified config reads them (and the
  API-key count now correctly shows 8, placeholders excluded).

---
- **Symptom:** the web server showed only the map; `localhost:8000` and the dev
  proxy's WebSocket both failed to connect for the first ~2 minutes after launch.
- **Cause:** startup regenerated every agent's day plan from scratch via the LLM
  every time (it never checked for an existing plan). Those were blocking calls,
  so they froze the async event loop and the server couldn't answer HTTP or the
  WebSocket handshake until planning finished (~120s in the logs).
- **Fix 1 (the big one):** startup now reuses a saved day plan for the
  (agent, date) when one exists, and only calls the planner when there is none.
  Verified: startup dropped from ~120s to ~1s with **0 LLM calls** when plans exist.
- **Fix 2:** the planner calls at startup (and the end-of-day next-day planner)
  run in a worker thread, so even a genuine first-time planning pass no longer
  freezes the web server.
- **Fix 3:** the web server now uses the configured tick speed instead of a
  hardcoded value, and the Vite dev proxy was hardened (broader `/ws` match,
  `127.0.0.1`, `changeOrigin`). For the demo, serving the built UI directly from
  `localhost:8000` avoids the dev proxy entirely.

---

## Phase A — Foundations & safety fixes

### Task 1 — Regression baseline + data/code hygiene  ✅
- Baseline: ran existing offline self-tests (snapshot, perceive, react) — all pass.
- Fixed corrupted keys in `data/environment/relationship_matrix.json`: removed the
  mojibake-arrow duplicate keys (the code only ever reads the clean `->` keys),
  keeping all 20 clean ordered pairs with their real scores.
- Turned the unreachable dead code after archival in `Short_term.py` into a real,
  documented maintenance function (`clear_short_term_data`) instead of leaving it
  stranded after a `return`.
- Replaced the empty `Long_term.py` docstring stub with a real module (see Task 3).
- Added this `changes.md`.

### Task 2 — Central toggles layer (env + CLI) + memory interface  ✅
- Extended `src/config.py` with an env-backed, CLI-overridable settings surface.
  Precedence: CLI flag > `.env` > built-in default. New toggles:
  `SIM_REFLEX_LLM` (default OFF), `SIM_TICK_SPEED`, `SIM_REAL_SECONDS_PER_SIM_MINUTE`
  (default 4.0 => ~96 real-min/sim-day), `SIM_MINUTES_PER_TICK`,
  `SIM_PERCEPTION_RADIUS_PX` (50), `SIM_PERCEPTION_ENABLED`, `SIM_MEMORY_BACKEND`
  (keyword|vector), `SIM_MAX_CONVERSATIONS_PER_AGENT`, `SIM_MAX_REPLANS_PER_AGENT_PER_DAY`,
  `SIM_LLM_HOURLY_CEILING`.
- Added `add_cli_arguments()`, `apply_overrides()`, `overrides_from_args()`, and
  `describe_settings()` to config; wired them into `world_engine.py`'s CLI. Removed
  the previously broken `--tick-speed` handling; the engine now reads the clock and
  speed from config dynamically so overrides actually take effect.
- Filtered obvious placeholder API keys (e.g. `your_google_api_key_here`) so the
  client doesn't waste attempts / trigger cooldowns on non-functional keys.
- **Default clock is now ~96 real minutes per sim-day** (was ~24), spreading LLM
  calls over 4x more wall-clock time to ease the per-key 5-RPM free-tier limit.
- Defined the swappable memory-retrieval interface (`MemoryRetriever`) in
  `Long_term.py`, with a `get_retriever()` factory selected by the config toggle.

### Task 2b — LLM budget governor + instrumentation  ✅
- Added `src/core/budget.py`: a thread-safe rolling-one-hour LLM call accountant
  (`GOVERNOR`) with `can_afford()`, `record()`, and `stats()`. A `SIM_LLM_HOURLY_CEILING`
  of 0 means no limit; above 0, cognition degrades gracefully when exceeded.
- Instrumented `gemini_client.py` to record every successful call in the governor
  (by tier and model), giving real visibility into spend.

## Phase B — Long-term memory (hybrid)

### Task 3 — Keyword + recency retriever over archives  ✅
- Implemented `KeywordMemoryRetriever` in `Long_term.py`: reads the per-day JSON
  archives (long-term + short-term), assembles daily summaries, conversations, and
  notable events into memory items, and ranks them by keyword overlap + recency +
  importance. Also produces a rolling multi-day summary. Fully offline.
- Verified against real archives (query "project club coding" for gurnoor_singh
  returned the relevant past conversations, ranked).

### Task 4 — Wire real memory into planning + conversation  ✅
- Added `WorldEngine._memory_context()` (0-LLM for keyword backend): pulls top-k
  relevant recalls + a rolling multi-day summary from the configured backend.
- Replaced the previously hardcoded-empty memory at **all three** day-planner call
  sites: start-of-day (`initialize`), post-conversation remaining-day plan, and the
  end-of-day next-day plan. Agents now actually plan with memory of prior days.
- Threaded optional recalled-memory context into the conversation prompt
  (`generate_conversation` / `_build_prompts`); the engine passes each agent's
  relevant memories into the single conversation call. Defaults preserve old
  behavior when no memory is available.
- Verified offline: engine returns 6 recalls + a summary for a seeded agent, and
  the recalled lines appear in the assembled conversation prompt.

### Task 5 — Swappable Qdrant + Gemini-embeddings backend (off by default)  ✅
- Added `src/agents/vector_memory.py` implementing the same `MemoryRetriever`
  interface via Qdrant + Gemini `text-embedding-004`. Everything is lazy-imported,
  so the project runs fine without `qdrant-client`; any failure (missing package,
  no server, embedding error) transparently falls back to the keyword backend.
- Only durable, high-value memories (summaries, conversation summaries, key events)
  are embedded — never raw per-tick observations — and every embedding call is
  gated by the budget governor.
- Added `qdrant-client` to `requirements.txt` as an explicitly optional dependency
  (you install it yourself). Verified `SIM_MEMORY_BACKEND=vector` selects the backend
  and cleanly falls back to keyword when the package is absent.

## Phase C — Spatial awareness (50px)

### Task 6 — 50px Euclidean spatial queries  ✅
- Added to `WorldSnapshot`: `agents_within_px()` (true circle proximity, nearest
  first), `distance_px()`, and `crowd_at()`. The legacy tile-based query is untouched.
- Added `perceive_px()` to `perceive.py` (Euclidean sibling of the tile perceive).
- Verified boundary behavior: an agent ~42px away is inside the 50px circle, one
  ~141px away is outside; crowd count correct.

### Task 7 — Activate perception each tick (0-LLM) + observations to memory  ✅
- `WorldEngine` now runs a perception step every tick (gated by `SIM_PERCEPTION_ENABLED`)
  that records, edge-triggered and de-duplicated, who each agent sees enter their
  50px circle (with location + crowd size). Pure spatial math; writes go through the
  memory backend (cheap keyword writes). No LLM.

### Task 8 — 50px proximity conversations + folded replan  ✅
- Replaced exact location-id matching with true 50px Euclidean proximity pairing.
  Preserves the pair-only rule (a 3+ cluster is skipped), cooldown, per-day cap, and
  the sleeping/blocked-action guard; a per-tick "busy" set prevents double-triggering.
- **Budget win:** folded the replan decision into the single conversation call
  (`ConversationResult.should_replan` + `plan_change`). The former automatic 8-calls-
  per-meeting remaining-day replan now only happens when the conversation flags it,
  the agent is under the per-day replan cap, and the governor allows it. Otherwise
  both agents resume their existing plan with zero extra LLM.
- Verified offline: engine imports; 50px neighbor pairing correct.

## Phase D — Human-like brain/body (layered, cheap-by-default)

### Task 9 — Body-as-tool adapter  ✅
- Added `src/agents/body.py` (`BodyController`): a thin, behaviour-preserving
  adapter around the action state machine — the agent's "limbs". Exposes a small
  motor command surface (advance / enter_conversation / resume) and read-only body
  state (position, current_action, is_last_action). No change to how movement runs.

### Task 10 — Brain wired as decision-maker  ✅
- Added `src/agents/brain.py` (`Brain`): the agent's cognition/command centre. Each
  tick it senses + recalls + decides, then commands the Body. Reflex-first and
  **0-LLM by default**: it only escalates to an LLM when `SIM_REFLEX_LLM` is on
  (OFF by default), the situation matches a conservative whitelist, the per-agent
  rate limit allows it, AND the budget governor has headroom. With reflex off,
  `step()` is behaviourally identical to the old direct executor call.
- Wired `WorldEngine._run_agent_tick` to drive each agent through a lazily-built,
  cached per-agent Brain → Body. The proven executor still does the actual work.

### Task 10b — Budget stress test  ✅
- Added `backend/tests/budget_stress.py`: runs 40 real ticks (perception + proximity
  + brain/body + checkpointing) with two agents and asserts **zero LLM calls**.
  Result: 40 ticks, 0 LLM calls. Confirms quiet ticks are free.

## Phase E — Frontend readability & storytelling

### Task 11 — Expanded tick snapshot  ✅
- The per-tick WebSocket snapshot now carries UI-facing fields: `speed`
  (multiplier + real-minutes-per-sim-day), per-agent `activity` label and
  `in_conversation` flag, and a campus-wide `recent_conversations` feed (rolling
  last 12, appended when a conversation completes). Verified the engine still
  makes 0 LLM calls on quiet ticks after the change.

### Task 12 — Focus, live labels, legend  ✅
- Rewrote `SimCanvas.jsx`: click a dot (or a legend row / agent window) to focus
  an agent — the camera smoothly centres on them, their dot gets a highlight ring,
  and their current activity is labelled on the map. Names are always labelled;
  drag still pans; a click on empty space clears focus.
- Added `Legend.jsx`: a colour→name legend (click to focus, shows a "chatting" tag).
- Confirmed `SimCanvas.jsx` and `AgentWindow.jsx` were NOT duplicates — just
  coincidentally the same byte size. Both are distinct, correct components.

### Task 13 — Conversation feed + day/night  ✅
- Added `ConversationFeed.jsx`: a live, auto-updating panel of recent campus
  conversations (participants, one-line summary, sentiment dot, time, location).
- `SimCanvas` now blends the day and night map assets by sim hour (dark overnight,
  clear midday, smooth dawn/dusk ramps) for a day/night feel.
- Made the web server honour the configured tick speed (was hardcoded 2x) so the
  on-screen pace matches the ~96 real-min/day clock.
- Frontend builds clean (`vite build`, 33 modules).

### Task 14 — Integration pass + docs  ✅
- Full offline regression pass — all green: snapshot, perceive, react, budget,
  brain self-tests; the headless budget stress test (40 ticks, 0 LLM calls); and a
  clean import of the server + engine + brain/body + memory backends. Frontend
  builds. The existing tick loop, checkpoint format, and day-transition flow are
  unchanged — every addition is additive and toggle-gated.
- Rewrote `README.md` in plain English: the full perceive → move → talk → react →
  remember → day/night pipeline, the brain/body idea, simplified run commands, the
  complete toggle table, and a step-by-step demo run-book for faculty & new students.
- Finalized this changelog with an at-a-glance summary and toggle reference.

---

## Phase A — Foundations & safety fixes (baseline notes)

---

## Conventions

- **No breaking changes** to the tick loop, checkpoint format, or day-transition
  flow. All new behavior is additive and gated behind toggles.
- **Budget-first:** quiet ticks make zero LLM calls; all cognition is gated by
  the LLM budget governor.
- Toggles default to the safest, cheapest behavior (reflex-LLM OFF, keyword
  memory backend, perception ON but 0-LLM).
