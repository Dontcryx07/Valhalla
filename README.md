# Valhalla — IIT Ropar Agent Simulation

AI-powered student personas (Parv, Tanishq, Gurnoor, etc.) with unique personalities live on a pixel-map of IIT Ropar campus. They plan their days, walk around, and talk to each other.

---


## Quick Start

```bash
# 1. Create and activate virtual environment
python -m venv venv
.\venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set up API keys
#    Edit .env — add Google Gemini API keys to GEMINI_API_KEY_1 .. _10
#    At least 1 key is required; more keys = better rate-limit resilience.

# 4. Run the full simulation (requires API keys)
$env:PYTHONPATH="backend"; python backend/src/core/world_engine.py --days 1

# 5. Run the web server
python backend/server.py
#    → http://127.0.0.1:8000
```

---

## How It Works (Big Picture)

The simulation runs in a tick loop orchestrated by the **WorldEngine**:

```
Initialize → Discover personas → Generate day plans (4 LLM calls each) → Register agents

Loop every tick (1 sim-minute = 1 real second):
  Run all agent actions in parallel (0 LLM cost)
  Check proximity for conversations
  Handle end-of-day transitions (archive, generate next plan)
  Save checkpoint
  Sync state → Advance clock
```

**Initialization:** The engine finds every persona JSON under `data/personalities/`, generates a full day plan for each agent via the day planner (4 Gemini calls per agent), creates an action state machine for each, and registers them at their hostel positions. The simulation clock doesn't start until all plans are ready.

**Each tick:** Every agent's action state machine advances one step — moving along a path, finishing the current action, or picking the next action from their day plan. This runs in parallel for all agents with zero LLM cost (it's a pure state machine). The engine then checks if any two agents are at the same location and may trigger a conversation. After all agents tick, a lightweight checkpoint (~55 KB) is saved to `data/checkpoints/` — the cumulative action history is excluded from checkpoints to minimize I/O.

**Conversations:** When two agents share a `location_id`, both are in non-blocked actions (not sleeping), neither is mid-conversation, and fewer than 3 agents are present — a single Gemini call generates the full dialogue. A remaining-day plan is generated immediately for both agents, and they resume normal action execution when the conversation ends. A 30-tick cooldown prevents back-to-back conversations with the same partner. Max 5 conversations per agent per day.

**End-of-day:** When an agent enters their last planned action of the day, the engine archives the day's data to `Long_term_db`, generates the next day's plan via the day planner, and pre-loads it into the agent's state machine. Per-day action history is saved to `data/history/<date>_history.json` for replay and analytics.

**Crash recovery:** Use `--resume <tick>` to restore from any saved checkpoint. Checkpoints store agent positions, action state machines, and conversation state — everything needed to pick up mid-sim. The last 10 checkpoints are kept automatically.

---

## How to Run

### Full simulation (needs Gemini keys)

```powershell
$env:PYTHONPATH="backend"; python backend/src/core/world_engine.py --days 1
```

Generates day plans for all personas, runs the simulation at 1 tick/second, detects conversations, and archives each day on completion.

### Skip plan generation for speed

```powershell
$env:PYTHONPATH="backend"; python backend/src/core/world_engine.py --days 1 --tick-speed 0.1
```

Runs 10 ticks/second for faster testing.

### Resume from a checkpoint

```powershell
$env:PYTHONPATH="backend"; python backend/src/core/world_engine.py --resume 1320
```

Loads state from the saved checkpoint at tick 1320 and continues running. Omit the tick number to see and choose from available checkpoints interactively.

### Action module self-test (no API keys needed)

```powershell
$env:PYTHONPATH="backend"; python backend/src/agents/Actions.py
```

Loads Parv's day plan, runs 600 ticks, and shows how the agent moves between locations and switches actions.

### Generate a conversation between two personas (needs Gemini keys)

```powershell
$env:PYTHONPATH="backend"; python backend/src/agents/conversation.py parv tanishq --current-time "2026-07-03 12:00"
```

Loads both personas' JSON files and day plans, then calls Gemini once to generate a natural conversation between them.

### Generate one person's day plan (needs Gemini keys)

```powershell
$env:PYTHONPATH="backend"; python backend/src/core/Agent.py parv
$env:PYTHONPATH="backend"; python backend/src/core/Agent.py tanishq
```

Loads the persona, calls the day planner (4 Gemini calls), and prints a table of everything they'll do from 00:00 to 24:00.

### Web server

```bash
python backend/server.py
```

Opens `http://127.0.0.1:8000` with a campus map UI, pathfinding API, and WebSocket for live updates.

---

## File Map

| File | What it does |
|------|-------------|
| `src/core/world_engine.py` | Main loop: init agents, run ticks, detect conversations, handle day transitions, save checkpoints. CLI: `python world_engine.py --days 1` |
| `src/core/world_state.py` | Mutable simulation database — agent positions, actions, occupancy, conversation cooldowns, action history |
| `src/core/agent_registry.py` | Single source of truth for all agent runtime state (position, manager, paused flag, conversation count) |
| `src/core/checkpoint_manager.py` | Per-tick crash recovery saves (~55 KB each, history excluded), per-day action history export, resume-from-checkpoint |
| `src/core/snapshot.py` | Freezes world state into an immutable snapshot for safe parallel reads |
| `src/core/perceive.py` | Filters what an agent can see based on position (pure math, no LLM) |
| `src/agents/Actions.py` | Action execution state machine: manages last/current/next action, movement, conversation overriding |
| `src/agents/day_planner.py` | Plans a day for one agent: 4 Gemini calls, supports full_day/remaining/next_day modes |
| `src/agents/react.py` | Decides if an agent should replan or continue their current action |
| `src/agents/conversation.py` | Generates dialogue between two agents via a single Gemini call, manages relationship matrix |
| `src/agents/Short_term.py` | Per-agent per-day JSON storage — plans, events, conversations, archival to Long_term_db |
| `src/agents/Long_term.py` | Long-term memory stub |
| `src/llm/gemini_client.py` | Multi-key Gemini wrapper with rotation, model fallback cascade, and 429/503 handling |
| `src/config.py` | Configuration — paths, API keys, time constants, max conversations per agent |
| `server.py` | FastAPI web server |
| `pathfinder.py` | Pixel-level BFS pathfinding |
| `Pathfinder_test.py` | Pathfinding visualizer (matplotlib) |

---

## Data Storage

| Path | Contents |
|------|----------|
| `data/personalities/<name>/<name>.json` | Persona definitions (Name, Branch, Hostel, hobbies, goals, etc.) |
| `data/Short_term_db/<name>/<YYYY-MM-DD>.json` | Per-agent per-day runtime data: plans, events, conversations, world snapshots |
| `data/Long_term_db/<name>/<YYYY-MM-DD>.json` | End-of-day archive (copy from Short_term with daily summary) |
| `data/checkpoints/tick_XXXXX.json` | Per-tick simulation snapshots for crash recovery (last 10 kept, ~55 KB each) |
| `data/history/<YYYY-MM-DD>_history.json` | Append-only action log per day for replay and analytics |

---

## LLM Calls Per Tick

| Component | Calls | When |
|-----------|-------|------|
| perceive | 0 | Pure math |
| react | 0 | Not used in current WorldEngine flow |
| day_planner | 4 | Once per agent at init (full_day), after conversations (remaining), at end-of-day (next_day) |
| actions | 0 | Pure state machine — reads plan, moves agent, no LLM |
| conversation | 1 per pair | Tick, when two agents share a location and conditions are met |
| Short_term.archive | 1 | Once per agent when entering last action (generates daily summary via LLM) |

---

## Environment Variables (`.env`)

```
GEMINI_API_KEY_1=...   # Primary (at least one required)
GEMINI_API_KEY_2=...   # Additional keys for rate-limit rotation
...
GEMINI_API_KEY_10=...
```

Up to 10 keys supported. The client rotates between them to avoid rate limits. On 429 (quota) or 503 (overloaded), the client immediately falls through to the next available model tier: gemini-3.5-flash → gemini-3.1-flash-lite → gemini-2.5-flash-lite.
