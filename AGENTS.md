# Valhalla — Agent Instructions

## Project Overview

Valhalla is a multi-agent simulation of IIT Ropar campus. AI-powered student personas (Parv, Tanishq, Gurnoor, etc.) with distinct personalities navigate a pixel-map, plan their days using LLMs, and will eventually interact with each other.

## How to Explain Things to Me

**CRITICAL — READ THIS:**

When I ask you to explain a file, a concept, or write documentation:

- **DO NOT** list out individual functions, their signatures, line numbers, or what each line of code does.
- **DO NOT** describe function internals or go through the code line by line.
- **DO** tell me in simple terms what the file's purpose is, what it produces, who calls it, what it depends on, and how to run or test it.
- **DO** use high-level analogies and plain English. Think "explain like I'm a teammate who knows the project exists but hasn't read every file."

**Good example:**
> "`day_planner.py` is the file that plans a full day for one agent. It calls Gemini 4 times to go from rough blocks → hourly plan → fine-grained actions → validation. The tick graph (`Agent.py`) calls it when the agent needs a new plan. You run it standalone with `python day_planner.py parv`."

**Bad example (do not do this):**
> "`day_planner.py` line 42 defines `class DayPlannerState(TypedDict)` with fields... The `run()` function at line 553 calls `_generate_coarse_plan()` at line 213 which makes an LLM call via `gemini_client.py` line 87..."

If I wanted to read the code, I'd open the file myself. I'm asking you so I don't have to.

## Project Stack

- **Agent brains:** LangGraph (Python)
- **LLM:** Google Gemini (via `google-genai` SDK, multi-key wrapper)
- **Memory:** JSON files per agent per day (short-term), Qdrant vector DB (long-term, not yet built)
- **Frontend:** SPA with campus map overlay
- **Server:** FastAPI (WebSocket + REST)
- **Pathfinding:** BFS on pixel grid

## Key Architecture

```
WorldEngine (not built yet) loops:
  advance_tick → snapshot → asyncio.gather(run_tick for each ready agent) → resolve

Each agent's tick graph (Agent.py):
  perceive → retrieve_memories → react → [day_planner OR keep_current] → write_back_memory

Action execution (Actions.py):
  last_action → current_action → next_action
  (state machine that reads day plan, handles movement, updates WorldState)
```

- One compiled LangGraph is reused for every agent every tick
- Mid-action agents are skipped entirely (zero cost)
- Day plans cached per-agent per-sim-day; regenerated on interrupt only
- Action module is a pure state machine — no LLM calls, just reads plan and moves agent

## File Map (Simplified)

| File | What it does |
|------|-------------|
| `src/core/Agent.py` | The tick graph pipeline + CLI debug tool. Production graph or standalone `python Agent.py parv` |
| `src/core/world_state.py` | The mutable simulation database — agent positions, actions, occupancy |
| `src/core/snapshot.py` | Freezes world state into an immutable snapshot for safe parallel reads |
| `src/core/perceive.py` | Filters what an agent can see based on position (pure math, no LLM) |
| `src/agents/Actions.py` | Action execution: manages last/current/next action, movement between locations, pathfinder integration |
| `src/agents/day_planner.py` | Plans a full day for one agent: 4 Gemini calls in a LangGraph |
| `src/agents/react.py` | Decides if an agent should replan or continue current action |
| `src/agents/Short_term.py` | Per-agent per-day JSON storage — plans, events, conversations |
| `src/agents/Long_term.py` | Qdrant long-term memory — currently a stub, not implemented |
| `src/agents/Single_agent.py` | Older standalone debug CLI for testing one persona |
| `src/llm/gemini_client.py` | Multi-key Gemini wrapper with rotation and model fallback |

## How to Run Things

```powershell
# Action module self-test (no API keys needed)
$env:PYTHONPATH="backend"; python backend/src/agents/Actions.py

# Tick graph self-test (no API keys needed)
$env:PYTHONPATH="backend"; python backend/src/core/Agent.py --self-test

# Day plan for one persona (needs Gemini keys in .env)
$env:PYTHONPATH="backend"; python backend/src/core/Agent.py parv
$env:PYTHONPATH="backend"; python backend/src/core/Agent.py tanishq

# Web server
python backend/server.py
# → http://127.0.0.1:8000

# Day planner standalone (verbose output)
$env:PYTHONPATH="backend"; python backend/src/agents/day_planner.py parv
```

## Persona Data

Persona JSON files live in `backend/data/personalities/<name>/<name>.json`. Each has fields like `Name`, `Age`, `Branch`, `Hostel`, `innate`, `lifestyle`, `hobbies`, `goals`.

## Memory System

- Short-term: `backend/data/Short_term_db/<persona_name>/<YYYY-MM-DD>.json`
- One JSON file per persona per simulation day
- Stores: day plan, events, conversations, world snapshots, daily summary

## What's Not Built Yet

- `WorldEngine` — the main simulation orchestrator loop
- Agent-to-agent conversations (triggered by proximity)
- Frontend integration with action states

## LLM Calls Per Tick

| Component | Calls | When |
|-----------|-------|------|
| perceive | 0 | Pure math |
| react | 0-1 | Only if mid-action + sees something new |
| day_planner | 4 | Once per sim day (coarse, hourly, fine, validate) |
| actions | 0 | Pure state machine — reads plan, moves agent, no LLM |
