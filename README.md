# Valhalla

Valhalla is a multi-agent campus-life simulation set at IIT Ropar.
AI student personas plan their day, move across a shared map, talk when they meet,
and build memory over time.
 
## What is Valhalla?

Valhalla is built for developers, AI hobbyists, and simulation researchers who want a practical sandbox for testing autonomous-agent behavior in a socially rich world. Its value is a full end-to-end loop (planning, movement, conversation, and memory) that you can run locally and inspect live.

## Quick Start

```bash
# 1) Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # Windows PowerShell: .\\venv\\Scripts\\Activate.ps1

# 2) Install Python dependencies
pip install -r requirements.txt

# 3) Build frontend
cd frontend && npm install && npm run build && cd ..

# 4) Configure environment variables
cp .env.local .env  # Windows PowerShell: Copy-Item .env.local .env
# Add at least one Gemini key in .env (for example GEMINI_API_KEY_1)

# 5) Run
python backend/Odin.py
```

Open <http://127.0.0.1:8000>.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture / Pipeline](#architecture--pipeline)
- [Setup](#setup)
- [Run](#run)
- [Configuration](#configuration)
- [Project Structure](#project-structure)
- [Security & Secrets](#security--secrets)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)

## Overview

Valhalla simulates a 24-hour social day in ticks. Each agent has personality traits,
energy and emotion state, location context, relationships, and memory.
The engine updates movement and interactions continuously while LLM calls are used for
high-level planning and dialogue.

https://github.com/user-attachments/assets/1e48772a-e8a9-45d3-a462-ddfe83b1c640



## Features

- AI-generated day plans for each persona with mid-day replanning.
- Tick-based movement on a campus map with location-aware behavior.
- 1:1 conversations that affect energy, emotion, and relationships.
- Persistent long-term memory via Qdrant (with runtime fallback behavior).
- Checkpoint/resume workflow for long-running simulations.
- Roster tools to add, rename, or retire agents.

  <img src="Screenshot 2026-07-25 002349.png" alt="Valhalla simulation UI" width="100%">

## Architecture / Pipeline

```text
Load Personas -> Generate Day Plans -> Start Clock
                                      |
                          +-----------+
                          v
              +---- Perceive (each tick) <----+
              |                               |
              v                               |
        Detect Conversations                  |
              |                               |
              v                               |
      LLM Decide: continue/replan            |
              |                               |
              v                               |
        Replan if needed                     |
              |                               |
              v                               |
          Advance agents ---------------------+
                (repeat for 1440 ticks / day)
                          |
                          v
                Archive Day -> Next Day Plans
```

## Setup

### Requirements

- Python 3.11+
- Node.js + npm (frontend build)
- Gemini API key(s)
- Optional: Qdrant Cloud account for semantic long-term memory

### Install

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cd frontend && npm install && npm run build && cd ..
```

Windows PowerShell equivalent:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
cd frontend; npm install; npm run build; cd ..
Copy-Item .env.local .env
```

## Run

```bash
# Fresh start (opens browser UI)
python backend/Odin.py

# Resume from last checkpoint
python backend/Odin.py --resume-checkpoint

# Headless single-day run
PYTHONPATH=backend python backend/src/core/world_engine.py --days 1
```

Windows PowerShell (headless):

```powershell
$env:PYTHONPATH="backend"; python backend/src/core/world_engine.py --days 1
```

## Configuration

Set values in `.env` (copy from `.env.local` first).

### Configuration Reference

| Variable | Default | Purpose |
|---|---|---|
| `SIM_TICK_SPEED` | `1.0` | Global simulation speed multiplier |
| `SIM_REAL_SECONDS_PER_SIM_MINUTE` | `7.5` | Wall-clock seconds per simulated minute |
| `SIM_MINUTES_PER_TICK` | `1` | Simulated minutes advanced each tick |
| `SIM_PERCEPTION_ENABLED` | `true` | Enables/disables perception step |
| `SIM_PERCEPTION_RADIUS_PX` | `50` | Nearby-agent detection radius |
| `SIM_MAX_CONVERSATIONS_PER_AGENT` | `5` | Daily conversation cap per agent |
| `SIM_MAX_REPLANS_PER_AGENT_PER_DAY` | `3` | Mid-day replan cap per agent |
| `SIM_LLM_HOURLY_CEILING` | `0` | Soft LLM call cap per real hour (`0` = unlimited) |
| `SIM_DECIDE_MIN_ENERGY` | `0.1` | Minimum energy to trigger LLM decisions |
| `SIM_DECIDE_MIN_EMOTION` | `0.1` | Minimum emotion to trigger LLM decisions |
| `SIM_CONVERSATION_MIN_ENERGY` | `0.05` | Minimum energy to start conversation |
| `SIM_CONVERSATION_MIN_EMOTION` | `0.05` | Minimum emotion to start conversation |
| `SIM_SEMANTIC_MEMORY_ENABLED` | `true` | Enables semantic long-term memory path |
| `QDRANT_URL` | `""` | Qdrant endpoint |
| `QDRANT_API_KEY` | `""` | Qdrant API key |
| `SIM_MEMORY_EMBEDDING_MODEL` | `gemini-embedding-001` | Embedding model for memory |
| `SIM_MEMORY_VECTOR_DIMENSIONS` | `768` | Embedding dimensions |
| `SIM_MEMORY_COLLECTION_VERSION` | `v1` | Memory schema/version tag |
| `SIM_GEMINI_MODEL` | `gemini-3.1-flash-lite` | Gemini model used by the simulation |
| `SIM_CREATIVITY` | `1.0` | Creativity dial for plans/dialogue |
| `SIM_WELLBEING_VARIABILITY` | `0.75` | Non-LLM variability in wellbeing updates |

## Project Structure

```text
backend/
  src/               # simulation engine, agent logic, planners, memory integrations
  data/              # personas, environment, checkpoints, relationships, archives
frontend/            # React + Vite UI
README.md            # project documentation
requirements.txt     # Python dependencies
```

## Security & Secrets

- Never commit real secrets (`.env`, API keys, private tokens, cloud credentials).
- Use `.env` for local secrets; keep only non-sensitive templates in version control.
- If a key is exposed, rotate it immediately in the provider dashboard.
- Remove leaked secrets from Git history before sharing or releasing the repository.
- Prefer least-privilege keys and separate keys for dev/staging/prod use.

## Troubleshooting

- **No LLM output / planning fails**: verify at least one valid `GEMINI_API_KEY*` is set in `.env`.
- **Frontend not updating**: rebuild with `cd frontend && npm run build`.
- **Import/path errors in headless mode**: ensure `PYTHONPATH=backend` is set.
- **Qdrant errors**: verify `QDRANT_URL` and `QDRANT_API_KEY`, or disable with `SIM_SEMANTIC_MEMORY_ENABLED=false`.

## Contributing

1. Fork and create a feature branch.
2. Keep changes focused and documented.
3. Validate local run paths before opening a PR.
4. Include a clear summary of behavior changes and test notes.
