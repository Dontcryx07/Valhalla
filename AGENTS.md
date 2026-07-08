# Valhalla — Agent Instructions

## Project Overview

Valhalla is a multi-agent simulation of IIT Ropar campus. AI-powered student personas (Parv, Tanishq, Gurnoor, etc.) with distinct personalities navigate a pixel-map, plan their days using LLMs, and interact with each other through conversations.

## How to Explain Things to Me

**CRITICAL — READ THIS:**

When I ask you to explain a file, a concept, or write documentation:

- **DO NOT** mention specific function names, class names, variable names, line numbers, or code-level details.
- **DO NOT** describe function internals, method signatures, or go through the code line by line.
- **DO NOT** use backtick-quoted function names in explanations.
- **DO** tell me in simple terms what the file's purpose is, what it produces, who calls it, what it depends on, and how to run or test it.
- **DO** use high-level analogies and plain English. Think "explain like I'm a teammate who knows the project exists but hasn't read every file."

## Project Stack

- **Agent brains:** LangGraph (Python)
- **LLM:** Google Gemini (via `google-genai` SDK, multi-key wrapper)
- **Memory:** JSON files per agent per day (short-term), JSON archive (long-term, Qdrant not implemented)
- **Frontend:** SPA with campus map overlay
- **Server:** FastAPI (WebSocket + REST)
- **Pathfinding:** BFS on pixel grid

## Key Architecture

```
WorldEngine (orchestrator) loops:

  initialize() — ONE-TIME:
    discover personas from data/personalities/
    generate full-day plans for all (day_planner, mode=full_day)
    register agents in WorldState at their hostel positions
    create AgentActionManager per agent
    → simulation starts with all agents ready, clock at 00:00

  run_tick() — EVERY TICK:
    1. Run all agent actions in parallel (asyncio.gather)
       Each: AgentActionManager.tick() advances the state machine
             (move along path, finish current → pick next from plan)
    2. Check proximity for conversations:
       - Two agents at same location_id → trigger
       - 3+ agents at same location → skip (>2 rule)
       - Skip if cooldown active or last conversation was with same partner
       - Generate conversation (1 Gemini call)
       - Generate remaining-day plans for both agents immediately
       - Set both agents to "Chatting with X" for the duration
       - When conversation ends → load remaining-day plan → resume
    3. Check last-action triggers (end-of-day):
       - Archive day to Long_term_db (Short_term.archive_to_long_term)
       - Generate next day's plan (day_planner, mode=next_day)
       - Load new plan into agent's state machine
    4. Sync registry → WorldState
    5. Advance tick by 1 sim-minute (1 real second)
```

- The AgentRegistry is the single source of truth for position/action; WorldState mirrors it
- `day_planner.py` supports three modes: `full_day` (00:00-24:00), `remaining` (current_time to 24:00), `next_day` (next full day)
- `Short_term.finalize_day` runs once when the agent enters their last plan action
- `archive_to_long_term` copies the day data to `data/Long_term_db/`

## File Map (Simplified)

| File | What it does |
|------|-------------|
| `src/core/world_engine.py` | The main loop: init agents → run ticks → detect conversations → handle day transitions. CLI: `python world_engine.py --days 1` |
| `src/core/world_state.py` | The mutable simulation database — agent positions, actions, occupancy, conversation cooldowns |
| `src/core/agent_registry.py` | Single source of truth for all agent runtime state (position, manager, paused flag, pending plans) |
| `src/core/snapshot.py` | Freezes world state into an immutable snapshot for safe parallel reads |
| `src/core/perceive.py` | Filters what an agent can see based on position (pure math, no LLM) |
| `src/agents/Actions.py` | Action execution state machine: manages last/current/next action, movement, conversation overriding, last-action detection |
| `src/agents/day_planner.py` | Plans a day (or remainder of day) for one agent: 4 Gemini calls in a LangGraph, supports full_day/remaining/next_day modes |
| `src/agents/react.py` | Decides if an agent should replan or continue current action |
| `src/agents/conversation.py` | Generates dialogue between two agents via a single Gemini call |
| `src/agents/Short_term.py` | Per-agent per-day JSON storage — plans, events, conversations, archival to Long_term_db |
| `src/agents/Long_term.py` | Long-term memory stub |
| `src/llm/gemini_client.py` | Multi-key Gemini wrapper with rotation and model fallback |

## How to Run Things

```powershell
# Full simulation (needs Gemini keys in .env)
$env:PYTHONPATH="backend"; python backend/src/core/world_engine.py --days 1

# Skip plan generation and archive for speed
$env:PYTHONPATH="backend"; python backend/src/core/world_engine.py --days 1 --tick-speed 0.1

# Action module self-test (no API keys needed)
$env:PYTHONPATH="backend"; python backend/src/agents/Actions.py

# Day plan for one persona (needs Gemini keys)
$env:PYTHONPATH="backend"; python backend/src/core/Agent.py parv
$env:PYTHONPATH="backend"; python backend/src/core/Agent.py tanishq

# Conversation between two personas (needs Gemini keys)
$env:PYTHONPATH="backend"; python backend/src/agents/conversation.py parv tanishq --current-time "2026-07-03 12:00"

# Day planner standalone (verbose output)
$env:PYTHONPATH="backend"; python backend/src/agents/day_planner.py parv

# Web server
python backend/server.py
# → http://127.0.0.1:8000
```

## Persona Data

Persona JSON files live in `backend/data/personalities/<name>/<name>.json`. Each has fields like `Name`, `Age`, `Branch`, `Hostel`, `innate`, `lifestyle`, `hobbies`, `goals`. The `Hostel` field is used as the agent's spawn position.

## Memory System

- Short-term: `backend/data/Short_term_db/<persona_name>/<YYYY-MM-DD>.json`
- One JSON file per persona per simulation day
- Stores: day plan, events, conversations, world snapshots, daily summary
- Long-term: `backend/data/Long_term_db/<persona_name>/<YYYY-MM-DD>.json` (created by `archive_to_long_term` at end-of-day)

## What's Not Built Yet

- Frontend integration with action states
- Qdrant vector DB for long-term memory

## LLM Calls Per Tick

| Component | Calls | When |
|-----------|-------|------|
| perceive | 0 | Pure math |
| react | 0 | Not used in current WorldEngine flow |
| day_planner | 4 | Once per agent at init (full_day), after conversations (remaining), at end-of-day (next_day) |
| actions | 0 | Pure state machine — reads plan, moves agent, no LLM |
| conversation | 1 per pair | WorldEngine tick, when two agents share a location |
| Short_term.archive | 1 | Once per agent when entering last action (generates daily summary via LLM) |
