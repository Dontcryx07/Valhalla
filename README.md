# Valhalla

Valhalla is a multi-agent simulation of IIT Ropar. AI student personas plan
their day, move around a campus map, talk when they meet, and retain memories
across simulation days. The backend is a FastAPI service; the browser client
shows the live campus state.

## What the simulation does

- Each persona has a profile, hostel, goals, interests, energy, and emotion.
- The simulation clock advances in configurable simulated-minute steps.
- Students follow AI-generated day plans and use grid pathfinding to move.
- When two eligible students are close enough, a Gemini-generated conversation
  is stored for both of them and can affect later memories.
- At a day boundary, the running simulation keeps the same students and action
  state. It allows active conversation work a bounded time to finish, archives
  the completed day, then installs continuity-aware plans for the next day.
- Checkpoints preserve the world, agent action state, pending conversations,
  relationships, and random state for recovery.

## Requirements

- Python 3.11 or later
- A Google Gemini API key
- Node.js and npm only when building or developing the frontend
- Optional: a Cloud Qdrant cluster for semantic long-term memory

## Start from scratch

Run all commands below from the repository root in PowerShell.

### 1. Create and activate a Python environment

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

`requirements.txt` includes the Qdrant client. It is harmless when semantic
memory is disabled.

### 2. Create `.env`

Create a file named `.env` in the repository root. At minimum it needs a
working Gemini key:

```text
GEMINI_API_KEY=your-google-gemini-api-key
```

Multiple keys can be supplied for rate-limit resilience:

```text
GEMINI_API_KEYS=key-one,key-two,key-three
```

Alternatively, use `GEMINI_API_KEY_1`, `GEMINI_API_KEY_2`, and so on. Do not
commit `.env` or any API key.

### 3. Build the frontend

The server can serve an existing frontend build. For a clean setup, create it:

```powershell
cd frontend
npm install
npm run build
cd ..
```

For frontend development, use `npm run dev` from `frontend` instead.

### 4. Start a fresh simulation

```powershell
python backend/Odin.py
```

Open <http://127.0.0.1:8000>.

This is intentionally a **fresh** start: it clears short-term runtime memory
and checkpoints before it initializes a new simulation. Long-term archives are
not cleared.

### 5. Resume a saved simulation

```powershell
python backend/Odin.py --resume-checkpoint
```

This command preserves runtime data and restores the newest checkpoint. It
fails instead of silently starting a new simulation when no checkpoint exists
or when the checkpoint roster differs from the configured personas. Start a
fresh simulation after adding, removing, or renaming personas.

Use a different port when needed:

```powershell
python backend/Odin.py --port 8080
```

## Running without the browser

```powershell
# Run one simulated day.
$env:PYTHONPATH="backend"; python backend/src/core/world_engine.py --days 1

# Run several consecutive days while retaining the same agents and positions.
$env:PYTHONPATH="backend"; python backend/src/core/world_engine.py --days 3

# Resume from a checkpoint.
$env:PYTHONPATH="backend"; python backend/src/core/world_engine.py --resume

# Print the active configuration.
$env:PYTHONPATH="backend"; python backend/src/config.py
```

## Runtime model

Each tick advances all agents, checks proximity, updates the live state, and
writes checkpoints. The wall-clock duration of a full simulated day is:

```text
1440 × SIM_REAL_SECONDS_PER_SIM_MINUTE ÷ SIM_TICK_SPEED
```

The code defaults are one simulated minute per tick, four real seconds per
simulated minute, and a speed multiplier of one. `.env` values can be
overridden by the corresponding engine command-line flags.

At midnight, the same simulation instance continues into the next day. It waits
up to `SIM_DAY_HANDOFF_CONVERSATION_TIMEOUT_SECONDS` for active conversations,
cancels any that exceed the limit, archives the finished day, and generates
next-day plans with each student's ending location and wellbeing as context.

## Runtime safeguards

- Agents only start conversations after both have finished travelling and are
  settled at the same location; close passes on unrelated routes do not pause
  them for a chat.
- Gemini decisions run in the background, so provider latency does not freeze
  simulation ticks or WebSocket updates. Any unfinished decision is discarded
  at day handoff and cannot replan the next day from stale context.
- The planner validates time ranges, known locations, and unsafe activity text.
  If retries are exhausted, its deterministic fallback replaces rejected
  activities with neutral downtime while preserving a complete schedule.
- The frontend shows a clear error state when a protected resume is rejected,
  instead of rendering an empty campus.

## Live UI and observability

- Every agent card and the conversation feed have minimize/maximize controls.
  A completed conversation is automatically minimized when either participant
  starts their next task.
- The feed retains recent completed conversations with their simulated time,
  venue, participants, and summary. Active conversations stay visible for their
  generated simulated duration rather than disappearing when the model call
  completes.
- The `DEBUG` control exposes bounded health telemetry: tick/time, moving and
  paused agents, active conversations, background task counts, and detected
  state anomalies.
- Scheduled campus events are included in the live snapshot. Eligible agents
  can add them to their plans, and the events panel shows upcoming and active
  entries.

## Memory

Short-term runtime records live in `backend/data/Short_term_db/` only while a
simulation day is active. Qdrant is the sole long-term database: each persona
has an isolated collection. At day handoff, the system stores the daily
summary, planned actions, completed actions, conversation summaries, and
explicit durable events. It does not embed raw conversation transcripts or
periodic world snapshots. After Qdrant acknowledges the archive, the completed
short-term JSON is removed; no `Long_term_db/<persona>/memory.json` is written.

### Qdrant semantic memory and RAG

RAG is enabled by default. Before each model-facing planning, replan, decision,
or conversation call, the application embeds a context-specific query,
semantically retrieves candidate memories from that persona's Qdrant
collection, ranks them by 65% semantic similarity, 20% importance, and 15%
recency, then adds the highest-relevance, diverse results to the prompt.

Qdrant must therefore be configured before running a persistent simulation:

Add the following to `.env`:

```text
SIM_SEMANTIC_MEMORY_ENABLED=true
QDRANT_URL=https://your-cluster.cloud.qdrant.io
QDRANT_API_KEY=your-qdrant-api-key
SIM_MEMORY_EMBEDDING_MODEL=gemini-embedding-001
SIM_MEMORY_VECTOR_DIMENSIONS=768
SIM_MEMORY_COLLECTION_VERSION=v1
```

If Qdrant or Gemini embeddings are unavailable, a model call receives no
long-term context. A failed archival write retains the short-term source file
and is reported as an archive failure; it is never replaced by keyword recall.

### Cloud storage retention

The application estimates its vector and payload footprint across all persona
collections against a 4 GiB budget by default. When that estimate reaches 90%
of the budget, it prunes Cloud Qdrant points until it reaches 85%.

Pruning uses a persistent retention score: memory importance, how recently it
was created or recalled, and how often semantic recall selected it. Low-value,
old, and rarely recalled memories are removed first. Because Qdrant is the only
long-term store, pruning is permanent; choose a larger budget/threshold if
retaining every memory matters.

Qdrant Cloud's dashboard remains the authoritative source for billed storage
usage. Adjust the application guard if your Cloud plan differs from 4 GiB:

```text
SIM_MEMORY_MAX_STORAGE_GB=4
SIM_MEMORY_STORAGE_PRUNE_THRESHOLD=0.90
SIM_MEMORY_STORAGE_PRUNE_TARGET=0.85
```

To migrate old JSON archives once, verify the reported counts and then delete
the legacy files explicitly:

```powershell
$env:PYTHONPATH="backend"; python -m src.agents.memory_index migrate-json --delete-source
```

Inspect a Qdrant collection:

```powershell
$env:PYTHONPATH="backend"; python -m src.agents.memory_index status --agent "Parv Singla"
```

### Clear all long-term memory

This permanently deletes only non-empty Qdrant collections identified as
Valhalla memory. It does **not** clear the current simulation, short-term
runtime data, checkpoints, plans, personas, or environment files.

```powershell
$env:PYTHONPATH="backend"; python -m src.agents.memory_index clear --yes
```

The command does not stop a running simulator. If another process archives a
completed day while or after the command runs, it can create new long-term
memory again. The operation intentionally does not delete unrelated collections
in a shared Qdrant cluster.

If the embedding model or vector dimensions change, set a new collection
version before storing new data. Do not mix embeddings from different models or
dimensions in one collection.

## Configuration reference

| Purpose | Environment variable | Default |
|---|---|---|
| Initial simulation date | `SIM_START_DATE` | `2026-07-03` |
| Initial simulation time | `SIM_START_TIME` | `00:00` |
| Simulated minutes per tick | `SIM_MINUTES_PER_TICK` | `1` |
| Real seconds per simulated minute | `SIM_REAL_SECONDS_PER_SIM_MINUTE` | `4.0` |
| Simulation speed multiplier | `SIM_TICK_SPEED` | `1.0` |
| Per-Gemini request deadline | `SIM_LLM_REQUEST_TIMEOUT_MS` | `10000` |
| Entire Gemini fallback deadline | `SIM_LLM_CALL_DEADLINE_SECONDS` | `30` |
| Retryable-key cooldown | `SIM_LLM_TRANSIENT_KEY_COOLDOWN_SECONDS` | `15` |
| Maximum wait for a rate-spaced key | `SIM_LLM_KEY_ACQUIRE_WAIT_SECONDS` | `2` |
| Perception radius | `SIM_PERCEPTION_RADIUS_PX` | `50` |
| Conversation radius | `SIM_CONVERSATION_RADIUS_PX` | `20` |
| Memory backend | `SIM_MEMORY_BACKEND` | Qdrant-only (fixed) |
| Enable Cloud Qdrant memory | `SIM_SEMANTIC_MEMORY_ENABLED` | `true` |
| Semantic-memory storage budget | `SIM_MEMORY_MAX_STORAGE_GB` | `4.0` |
| Start pruning at this fraction of budget | `SIM_MEMORY_STORAGE_PRUNE_THRESHOLD` | `0.90` |
| Prune down to this fraction of budget | `SIM_MEMORY_STORAGE_PRUNE_TARGET` | `0.85` |
| Maximum conversations per agent/day | `SIM_MAX_CONVERSATIONS_PER_AGENT` | `5` |
| Maximum replans per agent/day | `SIM_MAX_REPLANS_PER_AGENT_PER_DAY` | `3` |
| LLM calls per real hour (`0` = unlimited) | `SIM_LLM_HOURLY_CEILING` | `0` |
| Conversation handoff wait | `SIM_DAY_HANDOFF_CONVERSATION_TIMEOUT_SECONDS` | `45` |

Useful engine overrides:

```powershell
$env:PYTHONPATH="backend"; python backend/src/core/world_engine.py --days 1 --tick-speed 4
```

## Data locations

| Path | Contents |
|---|---|
| `backend/data/personalities/` | Persona profiles and initial hostels |
| `backend/data/environment/` | Campus locations, entry points, and relationships |
| `backend/data/Short_term_db/` | Active day plans, events, and conversations |
| Qdrant Cloud persona collections | Durable semantic long-term memory |
| `backend/data/checkpoints/` | Recovery checkpoints |
| `backend/output/logs/` | Application logs |

Short-term records, checkpoints, logs, frontend build output, and local test
artifacts are generated runtime files and are gitignored. Persona profiles,
environment definitions, relationships, and scheduled-event definitions are
source data and should remain under version control.

## Tests

```powershell
$env:PYTHONPATH="backend"; python -m unittest backend.tests.test_vector_memory -v
$env:PYTHONPATH="backend"; python -m unittest discover -s backend/tests -p "test_*.py" -v
```

The Cloud Qdrant integration test is intentionally guarded. Run it only with a
dedicated disposable Cloud Qdrant test cluster:

```powershell
$env:RUN_QDRANT_INTEGRATION="1"
$env:PYTHONPATH="backend"
python -m unittest backend.tests.test_vector_memory -v
```

The test creates and removes its dedicated test collection.
