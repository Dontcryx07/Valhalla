"""
FastAPI web server for the Valhalla agent map.
Serves the frontend (React SPA), exposes REST + WebSocket
endpoints for pathfinding (/api/path, /api/path/stream, /ws), and
streams simulation state via /ws/sim for live agent visualization.
"""

import os
import sys
import json
import asyncio
import logging
from datetime import datetime, timedelta
from pathlib import Path
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from typing import List, Optional
import re
import uvicorn

logger = logging.getLogger(__name__)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pathfinder import shortest_path, stats, ROOT

app = FastAPI(title="Valhalla Agent Map")

FRONTEND = os.path.join(ROOT, "frontend")
DATA_DIR = os.path.join(ROOT, "backend", "data")


# --------------------------------------------------------------------------- #
# Sim Manager — connection broadcast + background WorldEngine
# --------------------------------------------------------------------------- #

class SimBroadcaster:
    """Manages WebSocket clients subscribed to live simulation state."""

    def __init__(self):
        self.connections: set[WebSocket] = set()

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.connections.add(ws)

    def disconnect(self, ws: WebSocket):
        self.connections.discard(ws)

    async def broadcast(self, data: dict):
        dead = set()
        for ws in self.connections:
            try:
                await ws.send_json(data)
            except Exception:
                dead.add(ws)
        self.connections -= dead


_latest_snapshot: Optional[dict] = None
_sim_broadcaster = SimBroadcaster()
_sim_task: Optional[asyncio.Task] = None; bool


def _clear_runtime_state():
    """Remove short-term memory and checkpoints for an intentional fresh run."""
    import shutil
    for d in [
        os.path.join(DATA_DIR, "Short_term_db"),
        os.path.join(ROOT, "backend", "data", "checkpoints"),
    ]:
        if os.path.isdir(d):
            shutil.rmtree(d)


def _resume_checkpoint_requested() -> bool:
    """Read only Valhalla's startup flag without consuming Uvicorn arguments."""
    import argparse

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--resume-checkpoint", action="store_true")
    args, _ = parser.parse_known_args()
    return args.resume_checkpoint


def _print_agent_plans(engine):
    """Print each agent's full action plan to the CLI."""
    from src.core.agent_registry import AgentRuntimeState
    for state in engine.registry.all_states():
        actions = state.day_plan or []
        print(f"\n  {state.persona_name} — {len(actions)} actions:")
        if not actions:
            print("    (no plan)")
            continue
        for a in actions:
            start = a.get("start", "??")
            end = a.get("end", "??")
            loc = a.get("location_id", "?")
            desc = a.get("action", "?")
            print(f"    {start}-{end}  {loc:<30s} {desc}")
    print()


async def _run_sim(resume_checkpoint: bool = False):
    """Background task: initialize WorldEngine and drive the tick loop.
    
    Args:
        resume_checkpoint: If True, resume from the latest checkpoint without
            deleting saved runtime data.
    """
    global _latest_snapshot
    from src.core.world_engine import WorldEngine
    from src.core.checkpoint_manager import list_checkpoints, load_checkpoint
    from src.core.log import setup_logging

    setup_logging(run_id="server_sim", console=False)

    from src import config as _cfg
    TICK_SPEED = _cfg.TICK_SPEED
    start_date = _cfg.SIM_START_DATE

    engine = WorldEngine(sim_start_date=start_date, sim_start_hhmm=_cfg.SIM_START_TIME)

    if resume_checkpoint:
        ticks = list_checkpoints()
        if ticks:
            resume_tick = ticks[-1]
            print(f"[SimManager] found checkpoint at tick {resume_tick} — resuming")
            world, registry, checkpoint_state = load_checkpoint(
                resume_tick, engine.resolver, return_metadata=True,
            )
            engine.world = world
            engine.registry = registry
            engine.restore_checkpoint_state(checkpoint_state)
            await engine.resume_pending_conversations()
            _latest_snapshot = {"status": "initialized", "agents": {}}
        else:
            message = "No checkpoint exists. Start a fresh simulation with: python backend/server.py"
            logger.error("[SimManager] %s", message)
            _latest_snapshot = {"status": "error", "message": message}
            return
    else:
        # The normal command deliberately starts from no short-term memory or
        # checkpoint data. Resuming is available only via the explicit flag.
        _clear_runtime_state()
        try:
            logger.info("[SimManager] starting simulation for %s", start_date)
            await engine.initialize()
        except Exception as e:
            _latest_snapshot = {"status": "error", "message": str(e)}
            import traceback
            traceback.print_exc()
            return
        _latest_snapshot = {"status": "initialized", "agents": {}}
        _print_agent_plans(engine)

    while True:
        try:
            await engine.run(max_tick=1440, on_tick=_on_tick, tick_speed=TICK_SPEED)

            # Day transition — advance the simulation date by one day
            start_date = (datetime.strptime(start_date, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")
            logger.info("[SimManager] day transition — advancing to %s", start_date)

            await _sim_broadcaster.broadcast({"type": "day_reset", "date": start_date})
            await asyncio.sleep(1)

            engine = WorldEngine(sim_start_date=start_date, sim_start_hhmm="00:00")
            await engine.initialize()
            _print_agent_plans(engine)
        except asyncio.CancelledError:
            print("[SimManager] sim task cancelled")
            break
        except Exception as e:
            print(f"[SimManager] error: {e}")
            import traceback
            traceback.print_exc()
            break


async def _on_tick(snapshot: dict):
    """Called after each simulation tick — store + broadcast."""
    global _latest_snapshot
    _latest_snapshot = snapshot
    await _sim_broadcaster.broadcast(snapshot)


# --------------------------------------------------------------------------- #
# Startup
# --------------------------------------------------------------------------- #

@app.on_event("startup")
async def _start_sim():
    global _sim_task
    _sim_task = asyncio.create_task(
        _run_sim(resume_checkpoint=_resume_checkpoint_requested())
    )

# Suppress FastAPI on_event deprecation warning — the lifespan equivalent is
# fine, but on_event is simpler and works identically here. The warning is
# harmless noise. If it bothers you, migrate to:
#   @app.lifespan("startup")
#   async def _lifespan(app): async with SimLifespan(): yield


# --------------------------------------------------------------------------- #
# Sim state endpoint
# --------------------------------------------------------------------------- #

@app.post("/api/sim/reset")
async def reset_sim():
    """Wipe cache + checkpoints and restart the sim from 00:00."""
    global _sim_task, _latest_snapshot

    # Cancel the running sim
    if _sim_task is not None and not _sim_task.done():
        _sim_task.cancel()
        try:
            await _sim_task
        except (asyncio.CancelledError, Exception):
            pass

    _latest_snapshot = {"status": "resetting"}
    await _sim_broadcaster.broadcast({"type": "reset"})

    _clear_runtime_state()

    _sim_task = asyncio.create_task(_run_sim())
    return {"status": "reset", "message": "Simulation reset — starting fresh"}


@app.get("/api/sim/state")
async def get_sim_state():
    """Return the latest tick snapshot (or status if not yet running)."""
    if _latest_snapshot is None:
        return {"status": "initializing"}
    return _latest_snapshot


@app.websocket("/ws/sim")
async def sim_websocket(ws: WebSocket):
    """Subscribe to live tick-by-tick simulation state."""
    await _sim_broadcaster.connect(ws)
    try:
        # Send latest state immediately on connect
        if _latest_snapshot is not None:
            await ws.send_json(_latest_snapshot)
        while True:
            await ws.receive_text()  # keep connection alive
    except WebSocketDisconnect:
        _sim_broadcaster.disconnect(ws)


_BLD_NAME_MAP = None

def _load_bld_names():
    global _BLD_NAME_MAP
    if _BLD_NAME_MAP is not None:
        return
    _BLD_NAME_MAP = {}
    p = os.path.join(DATA_DIR, "environment", "num.txt")
    with open(p) as f:
        for line in f:
            m = re.match(r'building (\d+) - (.+)', line.strip())
            if m:
                _BLD_NAME_MAP[int(m.group(1))] = m.group(2)


@app.get("/api/buildings")
def get_buildings():
    _load_bld_names()
    with open(os.path.join(DATA_DIR, "environment", "buildings_polygon_decomposed.json")) as f:
        buildings = json.load(f)

    seen = []
    for b in buildings:
        base = b["building_name"].rsplit("_part", 1)[0]
        if base not in seen:
            seen.append(base)

    base_to_label = {}
    for i, base in enumerate(seen):
        num = i + 1
        base_to_label[base] = _BLD_NAME_MAP.get(num, base)

    simplified = []
    for b in buildings:
        tl = b["top_left"]
        br = b["bottom_right"]
        base = b["building_name"].rsplit("_part", 1)[0]
        simplified.append({
            "label": base_to_label[base],
            "id": base,
            "x": tl[0],
            "y": tl[1],
            "w": br[0] - tl[0],
            "h": br[1] - tl[1],
        })
    return simplified


@app.get("/api/stats")
def get_stats():
    return stats()


@app.get("/api/path")
def get_path(
    x1: int = Query(32),
    y1: int = Query(297),
    x2: int = Query(959),
    y2: int = Query(1205),
):
    start = (x1, y1)
    end = (x2, y2)
    path = shortest_path(start, end)
    if path is None:
        return JSONResponse({"error": "No path found"}, status_code=404)
    return {"path": path, "length": len(path), "start": list(start), "end": list(end)}


@app.get("/api/path/stream")
async def stream_path(
    x1: int = Query(32),
    y1: int = Query(297),
    x2: int = Query(959),
    y2: int = Query(1205),
):
    start = (x1, y1)
    end = (x2, y2)
    path = shortest_path(start, end)
    if path is None:
        from fastapi.responses import JSONResponse
        return JSONResponse({"error": "No path found"}, status_code=404)

    from fastapi.responses import StreamingResponse
    import time

    async def point_stream():
        yield json.dumps({"meta": {"length": len(path), "start": list(start), "end": list(end)}}) + "\n"
        for pt in path:
            yield json.dumps({"x": pt[0], "y": pt[1]}) + "\n"
            await asyncio.sleep(0)
        yield json.dumps({"done": True}) + "\n"

    return StreamingResponse(point_stream(), media_type="application/x-ndjson")


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            try:
                msg = json.loads(data)
            except json.JSONDecodeError:
                await websocket.send_json({"error": "invalid JSON"})
                continue

            if msg.get("type") == "path":
                start = (msg["x1"], msg["y1"])
                end = (msg["x2"], msg["y2"])
                path = shortest_path(start, end)
                if path is None:
                    await websocket.send_json({"type": "path_result", "error": "No path found"})
                else:
                    meta = {"length": len(path), "start": list(start), "end": list(end)}
                    await websocket.send_json({"type": "path_meta", "meta": meta})
                    for pt in path:
                        await websocket.send_json({"type": "path_point", "x": pt[0], "y": pt[1]})
                        await asyncio.sleep(0)
                    await websocket.send_json({"type": "path_done"})
            elif msg.get("type") == "ping":
                await websocket.send_json({"type": "pong"})
    except WebSocketDisconnect:
        pass


class AgentInput(BaseModel):
    x1: int
    y1: int
    x2: int
    y2: int

class PathsInput(BaseModel):
    agents: List[AgentInput]


@app.post("/api/paths")
async def get_paths(input_data: PathsInput):
    results = []
    for a in input_data.agents:
        start = (a.x1, a.y1)
        end = (a.x2, a.y2)
        path = shortest_path(start, end)
        results.append({
            "path": path,
            "length": len(path) if path else 0,
            "start": [a.x1, a.y1],
            "end": [a.x2, a.y2],
        })
    return {"paths": results}


@app.get("/api/entrypoints")
def get_entrypoints():
    with open(os.path.join(DATA_DIR, "environment", "entrypoint.json")) as f:
        return json.load(f)


# Serve React production build (dist/) if it exists, else fallback to frontend/
import mimetypes
# On Windows, the registry often maps .js to text/plain, which makes browsers
# refuse the ES module (strict MIME checking) and the whole SPA fails to load —
# leaving only a blank page. Force the correct types explicitly.
mimetypes.add_type("application/javascript", ".js")
mimetypes.add_type("application/javascript", ".mjs")
mimetypes.add_type("text/css", ".css")
mimetypes.add_type("image/svg+xml", ".svg")
mimetypes.add_type("image/png", ".png")

@app.get("/{full_path:path}")
async def serve_spa(full_path: str):
    dist = os.path.join(FRONTEND, "dist")
    if os.path.isdir(dist):
        file_path = os.path.join(dist, full_path) if full_path else os.path.join(dist, "index.html")
        if os.path.isfile(file_path):
            return FileResponse(file_path)
        return FileResponse(os.path.join(dist, "index.html"))
    file_path = os.path.join(FRONTEND, full_path) if full_path else os.path.join(FRONTEND, "index.html")
    if os.path.isfile(file_path):
        return FileResponse(file_path)
    return FileResponse(os.path.join(FRONTEND, "index.html"))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Valhalla simulation server")
    parser.add_argument(
        "--resume-checkpoint", action="store_true", default=False,
        help="Resume the latest checkpoint without clearing runtime data",
    )
    parser.add_argument(
        "--port", type=int, default=8000,
        help="Port to listen on (default: 8000)",
    )
    args, _ = parser.parse_known_args()

    if args.resume_checkpoint:
        print(f"\n  Valhalla is live => http://127.0.0.1:{args.port}   (resuming from checkpoint)\n")
    else:
        print(f"\n  Valhalla is live => http://127.0.0.1:{args.port}   (fresh simulation; runtime data will be cleared)\n")

    uvicorn.run("server:app", host="127.0.0.1", port=args.port, reload=False)
