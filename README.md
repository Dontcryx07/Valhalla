# Valhalla — IIT Ropar Agent Simulation

AI-powered student personas (Parv, Tanishq, Gurnoor, etc.) with unique personalities live on a pixel-map of IIT Ropar campus. They plan their days, walk around, and will eventually talk to each other.

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

# 4. Run the tick graph self-test (no API keys needed)
$env:PYTHONPATH="backend"; python backend/src/core/Agent.py --self-test

# 5. Generate a day plan for one persona (requires API keys)
$env:PYTHONPATH="backend"; python backend/src/core/Agent.py parv

# 6. Run the web server
python backend/server.py
#    → http://127.0.0.1:8000
```

---

## How It Works (Big Picture)

The simulation runs in a **tick loop** (not yet built — but this is the plan):

```
Advance time → Snapshot the world → Run agents in parallel → Apply results → Repeat
```

Each tick moves the simulation forward by a few minutes. Before any agent thinks, we take a **snapshot** — a frozen copy of the whole world (where everyone is, what they're doing, what's occupied). Then we hand that snapshot to every agent that needs to make a decision. Agents who are mid-action (e.g., halfway through "eating lunch") get skipped — zero cost.

Each agent runs a **decision pipeline** called a tick graph:

```
See what's around → Remember relevant stuff → Decide (replan or stay) → Write to memory
```

- **Seeing** is pure math — just checking how far things are. No AI cost.
- **Remembering** pulls recent memories from a JSON file for that agent.
- **Deciding** is either "keep doing what I'm doing" (free) or "call the AI to replan my day."
- **Replanning** calls Gemini 4 times to build a full day plan from scratch.
- **Writing to memory** saves what the agent decided to a JSON file.

**Conversations** work separately from the tick graph. The WorldEngine detects when two agents share the same `location_id`, both are in non-blocked actions (not sleeping), and neither is already mid-conversation. A single Gemini call generates the full dialogue — messages, summary, duration, and relationship impact. Both agents get their current action overwritten to "Chatting with X" for the duration, then naturally replan via the tick graph when it expires.

---

## How to Run

### Action module self-test (no API keys needed)

```powershell
$env:PYTHONPATH="backend"; python backend/src/agents/Actions.py
```

Loads Parv's day plan, runs 100 ticks, and shows how the agent moves between locations and switches actions. Demonstrates the location resolver, pathfinder integration, and action state machine.

### Tick graph self-test (no API keys needed)

```powershell
$env:PYTHONPATH="backend"; python backend/src/core/Agent.py --self-test
```

Creates a fake agent, runs the full decision pipeline without calling Gemini, and checks the output looks right.

### Generate a conversation between two personas (needs Gemini keys)

```powershell
$env:PYTHONPATH="backend"; python backend/src/agents/conversation.py parv tanishq --current-time "2026-07-03 12:00"
```

Loads both personas' JSON files and day plans, then calls Gemini once to generate a natural conversation between them. The LLM decides the topic, duration, and relationship impact based on personalities, schedules, and current activity.

### Generate one person's day plan (needs Gemini keys)

```powershell
$env:PYTHONPATH="backend"; python backend/src/core/Agent.py parv
$env:PYTHONPATH="backend"; python backend/src/core/Agent.py tanishq
$env:PYTHONPATH="backend"; python backend/src/core/Agent.py gurnoor --current-time "2026-08-15 08:00"
```

Loads the persona's JSON file, grabs any memories from yesterday, calls the day planner (which hits Gemini 4 times), and prints a table of everything they'll do from 00:00 to 24:00.

### Alternative debug CLI

```powershell
$env:PYTHONPATH="backend"; python backend/src/agents/Single_agent.py parv
```

Older simpler version of the same thing — just `get memories → plan day`. Useful for comparing outputs.

### Run the day planner directly

```powershell
$env:PYTHONPATH="backend"; python backend/src/agents/day_planner.py parv
```

Skips the tick graph wrapper and runs only the planning part. Shows full verbose output (Gemini calls, etc.).

### Web server

```bash
python backend/server.py
```

Opens `http://127.0.0.1:8000` with a campus map UI, pathfinding API, and WebSocket for live updates.

### Pathfinding visualizer

```bash
python backend/Pathfinder_test.py [x1 y1 x2 y2]
```

---

## File Map

| File | What it does |
|------|-------------|
| `src/core/Agent.py` | The tick graph pipeline + CLI debug tool. Run it standalone with `python Agent.py parv` |
| `src/core/world_state.py` | The mutable simulation database — agent positions, actions, occupancy |
| `src/core/snapshot.py` | Freezes world state into an immutable snapshot for safe parallel reads |
| `src/core/perceive.py` | Filters what an agent can see based on position (pure math, no LLM) |
| `src/agents/Actions.py` | Action execution: manages last/current/next action, movement between locations, pathfinder integration |
| `src/agents/day_planner.py` | Plans a full day for one agent: 4 Gemini calls to go from rough blocks → hourly plan → fine actions → validation |
| `src/agents/react.py` | Decides if an agent should replan or continue their current action |
| `src/agents/conversation.py` | Generates dialogue between two agents via a single Gemini call — feeds personas, day plans, current actions, and relationship scores into one prompt |
| `src/agents/Short_term.py` | Per-agent per-day JSON storage — plans, events, conversations |
| `src/agents/Long_term.py` | Qdrant long-term memory — currently a stub, not implemented |
| `src/agents/Single_agent.py` | Older standalone debug CLI for testing one persona |
| `src/llm/gemini_client.py` | Multi-key Gemini wrapper with rotation and model fallback |
| `src/config.py` | Configuration — paths, API keys, constants |
| `server.py` | FastAPI web server |
| `pathfinder.py` | Pixel-level BFS pathfinding |
| `Pathfinder_test.py` | Pathfinding visualizer (matplotlib) |

## Persona Data

Persona JSON files live in `backend/data/personalities/<name>/<name>.json`. Each has fields like `Name`, `Age`, `Branch`, `Hostel`, `innate`, `lifestyle`, `hobbies`, `goals`.

## Memory System

- **Short-term:** JSON files at `backend/data/Short_term_db/<persona_name>/<YYYY-MM-DD>.json` — one file per persona per simulation day
- **Long-term:** Not built yet — will use Qdrant vector DB for cross-day semantic search

## What's Not Built Yet

- `WorldEngine` — the main simulation orchestrator loop
- Frontend integration with action states
- Long-term memory (Qdrant vector DB)

## LLM Calls Per Tick

| Component | Calls | When |
|-----------|-------|------|
| perceive | 0 | Pure math |
| react | 0-1 | Only if mid-action + sees something new |
| day_planner | 4 | Once per sim day (coarse, hourly, fine, validate) |
| actions | 0 | Pure state machine — reads plan, moves agent, no LLM |
| conversation | 1 per pair | When two agents share a location, neither is sleeping, and neither is already mid-conversation |

## Environment Variables (`.env`)

```
GEMINI_API_KEY_1=...   # Primary (at least one required)
GEMINI_API_KEY_2=...   # Additional keys for rate-limit rotation
...
GEMINI_API_KEY_10=...
```

Up to 10 keys supported. The client rotates between them to avoid rate limits.
