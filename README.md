# Valhalla — IIT Ropar Agent Simulation

Generative-agents simulation set on the IIT Ropar campus. AI-powered student personas with distinct personalities navigate a pixel-map, plan their days, and (future) interact with each other.

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

# 4. Run the web server
python backend/server.py
#    → http://127.0.0.1:8000

# 5. Run the day planner (standalone, no server needed)
python backend/src/agents/Single_agent.py <agent-name> - example: py Single_agents.py tanishq
```

---

## Architecture

```
                     ┌─────────────────────┐
                     │   frontend/          │
                     │   index.html         │
                     │   map.png            │
                     │   path.png           │
                     └──────────┬───────────┘
                                │ HTTP / WS
                     ┌──────────┴───────────┐
                     │   server.py           │  ← FastAPI + uvicorn :8000
                     │                      │
                     │  GET  /api/path      │  ← BFS shortest path
                     │  POST /api/paths     │  ← batch pathfinding
                     │  WS   /ws            │  ← real-time agent streams
                     │  GET  /api/buildings │  ← building rectangles
                     └──────────┬───────────┘
                                │
              ┌─────────────────┴──────────────────┐
              │            pathfinder.py            │
              │  BFS on frontend/path.png pixels    │
              └────────────────────────────────────┘

                     src/agents/
              ┌────────────────────────────────────┐
              │  Single_agent.py                   │
              │    └── create_agent_graph()        │  ← LangGraph brain
              │         └── generate_day_plan      │
              │              └── day_planner.run() │
              └────────────────────────────────────┘
                                │
              ┌─────────────────┴──────────────────┐
              │         day_planner.py              │
              │  LangGraph: coarse → hourly → fine  │
              │  → validate (retry up to 3×)        │
              └────────────────────────────────────┘
                                │
              ┌─────────────────┴──────────────────┐
              │         gemini_client.py            │
              │  Multi-key rate limiter             │
              │  Model fallback per tier            │
              │  Retry with backoff                 │
              └────────────────────────────────────┘
```

### Layers

| Layer | What |
|-------|------|
| **Agent brain** | `Single_agent.py` — LangGraph orchestrator per agent (day planning, future: memory, tick loop) |
| **Cognitive module** | `day_planner.py` — LangGraph subgraph: coarse → hourly → fine → validate |
| **LLM client** | `gemini_client.py` — API wrapper with key rotation, rate limiting, retries |
| **Web server** | `server.py` — FastAPI: pathfinding API + WebSocket + frontend static files |
| **Pathfinding** | `pathfinder.py` — BFS on pixel-level walkable paths (`path.png`) |
| **Frontend** | `index.html` — Canvas map viewer, agent controls, path animation |
| **Memory (future)** | `Short_term.py`, `Long_term.py` — stubs |

---

## How to Run

### Web server (full simulation UI)

```bash
python backend/server.py
```
- Opens `http://127.0.0.1:8000`
- Frontend shows campus map, agent slots, buildings overlay
- Pathfinding REST API available at `/api/path`, `/api/paths`, etc.
- WebSocket at `/ws`

### Day planner (standalone)

```bash
python backend/src/agents/Single_agent.py parv
python backend/src/agents/Single_agent.py tanishq
python backend/src/agents/Single_agent.py gurnoor
```

Runs the full LLM pipeline for one persona: coarse plan → hourly decompose → fine decompose → validate. Prints a plan table to terminal.

Backend equivalent (full output, no log suppression):

```bash
python backend/src/agents/day_planner.py parv
```

### Pathfinding visualizer

```bash
python backend/Pathfinder_test.py [x1 y1 x2 y2]
```

Shows the BFS path on `map.png` using matplotlib.

### Tile-based pathfinder (legacy)

```bash
python backend/tiles/nav_graph.py
```

BFS on 4×3 pixel tiles. Predecessor to pixel-level `pathfinder.py`.

### Map tile generator

```bash
python backend/tiles/generate_map_tiles.py
```

Produces `frontend/tiles/tilles.jsonl` (131,109 tile records).

---

## Data Files

| File | Purpose |
|------|---------|
| `data/personalities/<name>/<name>.json` | 5 student personas with traits, lifestyle, goals |
| `data/environment/places.json` | 20 campus locations with hours, activities, sub-areas |
| `data/environment/entrypoint.json` | Building entry points (pixel coords) |
| `data/environment/buildings_polygon_decomposed.json` | Building rectangles decomposed into parts |
| `data/environment/num.txt` | Building number → name mapping |
| `data/temp/<name>.json` | Generated day-plan output (created by `day_planner.py`) |
| `output/logs/` | Run logs (created by `log.py`) |

## Frontend Files

| File | Purpose |
|------|---------|
| `frontend/index.html` | SPA: Canvas map, agent UI, WebSocket |
| `frontend/map.png` | Campus map (1276×1233) |
| `frontend/map_night.png` | Night variant |
| `frontend/path.png` | Walkable paths (white = walkable) |
| `frontend/tiles/tilles.jsonl` | 131,109 tile records |
| `frontend/tiles/agent_path_tiles.json` | Traversable tile IDs |

## Environment Variables (`.env`)

```
GEMINI_API_KEY_1=...   # Primary (at least one required)
GEMINI_API_KEY_2=...   # Additional keys for rate-limit rotation
...
GEMINI_API_KEY_10=...
```

Up to 10 keys supported. The client rotates through all keys with a 12s interval per key. More keys = higher effective throughput.

---

## File Tree

```
valhalla/
├── backend/
│   ├── data/            # environment, personalities, temp output
│   ├── output/          # logs
│   ├── src/
│   │   ├── agents/      # Single_agent.py, day_planner.py
│   │   ├── core/        # log.py, Short_term.py, Long_term.py
│   │   └── llm/         # gemini_client.py
│   ├── tiles/           # generate_map_tiles.py, nav_graph.py
│   ├── server.py        # FastAPI server
│   ├── pathfinder.py    # pixel BFS
│   └── Pathfinder_test.py
├── frontend/
│   ├── index.html       # SPA
│   ├── map.png          # campus map
│   ├── path.png         # walkability mask
│   └── tiles/           # tile data
├── .env                 # API keys
├── requirements.txt
└── README.md
```
