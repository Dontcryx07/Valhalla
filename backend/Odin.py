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
import socket
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from pathlib import Path
from urllib.parse import urlparse
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query, Depends, HTTPException, Header
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional
import re
import shutil
import uvicorn

logger = logging.getLogger(__name__)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pathfinder import shortest_path, stats, ROOT

FRONTEND = os.path.join(ROOT, "frontend")
DATA_DIR = os.path.join(ROOT, "backend", "data")


def _public_asset_url(value: str) -> str:
    """Return a browser-loadable URL, correcting GitHub's HTML blob links."""
    value = value.strip()
    parsed = urlparse(value)
    parts = parsed.path.strip("/").split("/")
    if parsed.scheme == "https" and parsed.netloc in {"github.com", "www.github.com"} \
            and len(parts) >= 5 and parts[2] == "blob":
        owner, repository, _, revision, *path = parts
        return f"https://raw.githubusercontent.com/{owner}/{repository}/{revision}/{'/'.join(path)}"
    return value


# --------------------------------------------------------------------------- #
# Sim Manager — connection broadcast + background WorldEngine
# --------------------------------------------------------------------------- #

class SimBroadcaster:
    """Manages WebSocket clients via per-client bounded queues.

    Each connected client gets an asyncio.Queue(maxsize=5).  The simulation
    pushes snapshots into every queue with put_nowait (dropping oldest on
    overflow).  A lightweight drain task per client sends frames at the pace
    that individual connection can handle — a slow client only affects itself.
    """

    def __init__(self):
        self._queues: dict[WebSocket, asyncio.Queue] = {}
        self._tasks: dict[WebSocket, asyncio.Task] = {}

    async def connect(self, ws: WebSocket, initial_snapshot: Optional[dict] = None):
        await ws.accept()
        q: asyncio.Queue = asyncio.Queue(maxsize=5)
        self._queues[ws] = q
        if initial_snapshot is not None:
            q.put_nowait(initial_snapshot)
        self._tasks[ws] = asyncio.create_task(self._drain(ws, q))

    def disconnect(self, ws: WebSocket):
        task = self._tasks.pop(ws, None)
        if task:
            task.cancel()
        self._queues.pop(ws, None)

    async def broadcast(self, data: dict):
        for q in list(self._queues.values()):
            try:
                q.put_nowait(data)
            except asyncio.QueueFull:
                try:
                    q.get_nowait()
                except asyncio.QueueEmpty:
                    pass
                q.put_nowait(data)

    async def _drain(self, ws: WebSocket, q: asyncio.Queue):
        try:
            while True:
                data = await q.get()
                try:
                    await ws.send_json(data)
                except Exception:
                    break
        except asyncio.CancelledError:
            pass
        finally:
            self._queues.pop(ws, None)
            self._tasks.pop(ws, None)


_latest_snapshot: Optional[dict] = None
_sim_broadcaster = SimBroadcaster()
_sim_task: Optional[asyncio.Task] = None
_sim_engine = None
_sim_tick_lock = asyncio.Lock()


def _clear_runtime_state():
    """Remove all local runtime records for an intentional fresh run."""
    for d in [
        os.path.join(DATA_DIR, "Short_term_db"),
        os.path.join(ROOT, "backend", "data", "checkpoints"),
        os.path.join(DATA_DIR, "history"),
    ]:
        if os.path.isdir(d):
            shutil.rmtree(d)


def _clear_long_term_memory() -> dict:
    """Clear only Valhalla-owned semantic collections for an explicit reset."""
    try:
        from src.agents.Long_term import get_retriever
        clear = getattr(get_retriever(), "clear_all_memory", None)
        return clear() if callable(clear) else {"available": False}
    except Exception as exc:
        logger.warning("[SimManager] unable to clear semantic memory: %s", exc)
        return {"available": False, "error": str(exc)}


def _resume_checkpoint_requested() -> bool:
    """Read only Valhalla's startup flag without consuming Uvicorn arguments.

    A near-miss such as ``--resume--checkpoint`` must never silently become a
    fresh start, because the fresh-start path intentionally clears local
    runtime state.
    """
    import argparse

    invalid_resume_flags = [arg for arg in sys.argv[1:] if arg.startswith("--resume") and arg != "--resume-checkpoint"]
    if invalid_resume_flags:
        raise SystemExit(
            "Invalid resume flag: " + ", ".join(invalid_resume_flags)
            + ". Did you mean --resume-checkpoint?"
        )
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--resume-checkpoint", action="store_true")
    args, _ = parser.parse_known_args()
    return args.resume_checkpoint


def _reserve_listen_socket(host: str, port: int) -> socket.socket:
    """Bind before app startup, so an occupied port cannot start the sim."""
    listen_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        if os.name == "nt" and hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
            listen_socket.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        else:
            listen_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listen_socket.bind((host, port))
        listen_socket.listen(socket.SOMAXCONN)
        return listen_socket
    except Exception:
        listen_socket.close()
        raise


async def _stop_sim_task() -> None:
    """Stop the background engine before the ASGI app releases resources."""
    global _sim_task
    task = _sim_task
    _sim_task = None
    if task is None or task.done():
        return
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)


@asynccontextmanager
async def _sim_lifespan(_app: FastAPI):
    """Own the simulation task for exactly the ASGI app lifetime."""
    global _sim_task
    _sim_task = asyncio.create_task(
        _run_sim(resume_checkpoint=_resume_checkpoint_requested()),
        name="valhalla-simulation",
    )
    try:
        yield
    finally:
        await _stop_sim_task()


app = FastAPI(title="Valhalla Agent Map", lifespan=_sim_lifespan)

# ---------------------------------------------------------------------------
# CORS — allow cross-origin requests from the Vercel frontend
# ---------------------------------------------------------------------------

from fastapi.middleware.cors import CORSMiddleware

_cors_raw = os.environ.get("CORS_ORIGINS", "*").strip()
CORS_ORIGINS = [o.strip() for o in _cors_raw.split(",") if o.strip()] if _cors_raw else ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Auth — admin login protects simulation control routes
# ---------------------------------------------------------------------------

from src.auth.manager import auth_manager
from src.auth.routes import router as auth_router

app.include_router(auth_router)


async def require_admin(authorization: str = Header(None)):
    """FastAPI dependency: accept a Bearer token, return user or 401."""
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Not authenticated")
    user = auth_manager.validate_session(authorization[7:])
    if not user:
        raise HTTPException(status_code=401, detail="Invalid or expired session")
    return user


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


def _print_sim_status(snapshot: dict) -> None:
    """Emit compact simulation-time telemetry rather than checkpoint noise."""
    if not snapshot.get("agents"):
        return
    print(
        f"[SIM] day={snapshot.get('day', '?')} tick={snapshot.get('tick', '?')} "
        f"time={snapshot.get('time', '??:??')} speed=x{snapshot.get('speed', {}).get('multiplier', '?')}"
    )
    for agent in snapshot["agents"].values():
        action = agent.get("current_action") or {}
        state = "paused" if agent.get("paused") else "active"
        print(
            f"  {agent['name']:<18} {state:<6} "
            f"at={agent['position'].get('location_id', '?'):<22} "
            f"action={action.get('description', 'Idle'):<42} "
            f"energy={agent.get('energy_level', 0):.2f} emotion={agent.get('emotion_state', 0):.2f}"
        )
    print()


async def _run_sim(resume_checkpoint: bool = False):
    """Background task: initialize WorldEngine and drive the tick loop.
    
    Args:
        resume_checkpoint: If True, resume from the latest checkpoint without
            deleting saved runtime data.
    """
    global _latest_snapshot, _sim_engine
    from src.core.world_engine import WorldEngine
    from src.core.checkpoint_manager import list_checkpoints, load_checkpoint
    from src.core.log import setup_logging
    from src.llm.gemini_client import ProviderFailureError, provider_failure

    setup_logging(run_id="server_sim", console=False)

    from src import config as _cfg
    start_date = _cfg.SIM_START_DATE

    engine = WorldEngine(sim_start_date=start_date, sim_start_hhmm=_cfg.SIM_START_TIME)
    _sim_engine = engine

    if resume_checkpoint:
        ticks = list_checkpoints()
        if ticks:
            resume_tick = ticks[-1]
            print(f"[SimManager] found checkpoint at tick {resume_tick} — resuming")
            world, registry, checkpoint_state = load_checkpoint(
                resume_tick, engine.resolver, return_metadata=True,
            )
            # Checkpoints own historical membership. Comparing against the
            # mutable persona folder would make an old checkpoint impossible
            # to rewind after a roster add/remove operation.
            engine.world = world
            engine.registry = registry
            engine.restore_checkpoint_state(checkpoint_state)
            await engine.resume_pending_conversations()
            _latest_snapshot = {"status": "initialized", "agents": {}}
        else:
            message = "No checkpoint exists. Start a fresh simulation with: python backend/Odin.py"
            logger.error("[SimManager] %s", message)
            _latest_snapshot = {"status": "error", "message": message}
            return
    else:
        # The normal command deliberately starts from no short-term memory or
        # checkpoint data. Resuming is available only via the explicit flag.
        _clear_runtime_state()
        _clear_long_term_memory()
        try:
            logger.info("[SimManager] starting simulation for %s", start_date)
            await engine.initialize()
        except ProviderFailureError as exc:
            failure = provider_failure() or exc
            _latest_snapshot = {"status": "provider_failure", "message": str(failure), "failure": failure.payload(),
                                "simulation": {"running": False, "stop_reason": failure.code}}
            print(f"[SimManager] {failure.title.upper()} — simulation stopped: {failure}")
            await _sim_broadcaster.broadcast(_latest_snapshot)
            return
        except Exception as e:
            _latest_snapshot = {"status": "error", "message": str(e)}
            import traceback
            traceback.print_exc()
            return
        _latest_snapshot = {"status": "initialized", "agents": {}}
        _print_agent_plans(engine)

    while True:
        try:
            day_end_tick = ((engine.world.tick // (24 * 60)) + 1) * (24 * 60)
            await engine.run(max_tick=day_end_tick, on_tick=_on_tick, tick_lock=_sim_tick_lock)
            start_date = (
                datetime.strptime(engine.sim_start_date, "%Y-%m-%d") + timedelta(days=1)
            ).strftime("%Y-%m-%d")
            logger.info("[SimManager] day transition — advancing to %s", start_date)
            async with _sim_tick_lock:
                _latest_snapshot = engine.current_frontend_snapshot()
                _latest_snapshot.update({"type": "day_handoff", "phase": "draining", "date": start_date})
                await _sim_broadcaster.broadcast(_latest_snapshot)
                _latest_snapshot = await engine.handoff_to_next_day(start_date)
                await _sim_broadcaster.broadcast(_latest_snapshot)
            _print_agent_plans(engine)
        except asyncio.CancelledError:
            print("[SimManager] sim task cancelled")
            raise
        except ProviderFailureError as exc:
            failure = provider_failure() or exc
            _latest_snapshot = {"status": "provider_failure", "message": str(failure), "failure": failure.payload(),
                                "simulation": {"running": False, "stop_reason": failure.code}}
            print(f"[SimManager] {failure.title.upper()} — simulation stopped: {failure}")
            await _sim_broadcaster.broadcast(_latest_snapshot)
            break
        except Exception as exc:
            print(f"[SimManager] error: {exc}")
            import traceback
            traceback.print_exc()
            break


async def _on_tick(snapshot: dict):
    """Called after each simulation tick — store + broadcast."""
    global _latest_snapshot
    _latest_snapshot = snapshot
    await _sim_broadcaster.broadcast(snapshot)
    if snapshot.get("tick", 0) % 15 == 0:
        _print_sim_status(snapshot)


# --------------------------------------------------------------------------- #
# Startup
# --------------------------------------------------------------------------- #

# Suppress FastAPI on_event deprecation warning — the lifespan equivalent is
# fine, but on_event is simpler and works identically here. The warning is
# harmless noise. If it bothers you, migrate to:
#   @app.lifespan("startup")
#   async def _lifespan(app): async with SimLifespan(): yield


# --------------------------------------------------------------------------- #
# Sim state endpoint
# --------------------------------------------------------------------------- #

@app.post("/api/sim/reset")
async def reset_sim(_user: dict = Depends(require_admin)):
    """Wipe cache + checkpoints and restart the sim from 00:00."""
    global _sim_task, _latest_snapshot

    # Reset and rewind/handoff all replace mutable engine state.  Serialising
    # them avoids a cancelled task writing a stale checkpoint after reset.
    async with _sim_tick_lock:
        if _sim_task is not None and not _sim_task.done():
            _sim_task.cancel()
            try:
                await _sim_task
            except (asyncio.CancelledError, Exception):
                pass

        _latest_snapshot = {"status": "resetting"}
        _clear_runtime_state()
        _clear_long_term_memory()
        _sim_task = asyncio.create_task(_run_sim())

    await _sim_broadcaster.broadcast({"type": "reset"})
    return {"status": "reset", "message": "Simulation reset — starting fresh"}


def _simulation_is_running() -> bool:
    return _sim_task is not None and not _sim_task.done()


def _snapshot_with_simulation_status(snapshot: Optional[dict]) -> dict:
    """Expose task state without mutating the engine-owned snapshot object."""
    payload = dict(snapshot or {"status": "initializing"})
    existing = payload.get("simulation") if isinstance(payload.get("simulation"), dict) else {}
    payload["simulation"] = {**existing, "running": _simulation_is_running()}
    return payload


@app.post("/api/sim/stop")
async def stop_sim(_user: dict = Depends(require_admin)):
    """Stop the tick driver without deleting persisted checkpoints."""
    global _latest_snapshot, _sim_engine
    if not _simulation_is_running():
        return {"status": "already_stopped", "running": False}

    await _stop_sim_task()
    _sim_engine = None
    _latest_snapshot = _snapshot_with_simulation_status(_latest_snapshot)
    await _sim_broadcaster.broadcast(_latest_snapshot)
    return {"status": "stopped", "running": False}


@app.post("/api/sim/start")
async def start_sim(_user: dict = Depends(require_admin)):
    """Start from the newest persisted checkpoint, never stale RAM state."""
    global _sim_task
    if _simulation_is_running():
        return {"status": "already_running", "running": True}

    from src.core.checkpoint_manager import list_checkpoints
    checkpoints = list_checkpoints()
    if not checkpoints:
        return JSONResponse(
            {"error": "No checkpoint is available. Use Reset or restart Valhalla to create a fresh simulation."},
            status_code=409,
        )

    _sim_task = asyncio.create_task(
        _run_sim(resume_checkpoint=True),
        name="valhalla-simulation",
    )
    return {
        "status": "starting_from_checkpoint",
        "running": True,
        "checkpoint_tick": checkpoints[-1],
    }


class RewindInput(BaseModel):
    ticks: Optional[int] = Field(default=None, ge=1, le=1440)
    hours: Optional[float] = Field(default=None, gt=0, le=24)

    def requested_ticks(self, minutes_per_tick: int) -> int:
        if self.ticks is not None:
            return self.ticks
        if self.hours is not None:
            return max(1, round((self.hours * 60) / max(1, minutes_per_tick)))
        raise ValueError("Provide a positive rewind amount in ticks or hours.")


# --------------------------------------------------------------------------- #
# Stopped-only roster editing.  Each edit writes a successor checkpoint rather
# than overwriting history, so rewinding to a pre-edit checkpoint restores the
# roster that actually existed at that point in the timeline.
# --------------------------------------------------------------------------- #

class AddAgentInput(BaseModel):
    description: str = Field(min_length=30, max_length=4000)


class RemoveAgentInput(BaseModel):
    agent_id: str = Field(min_length=1, max_length=128)


class ReplaceAgentInput(BaseModel):
    agent_id: str = Field(min_length=1, max_length=128)
    name: str = Field(min_length=2, max_length=80)


class GeneratedPersona(BaseModel):
    name: str = Field(min_length=2, max_length=80)
    age: int = Field(ge=18, le=30)
    gender: str = Field(min_length=1, max_length=32)
    branch: str = Field(min_length=2, max_length=160)
    home_city: str = Field(min_length=2, max_length=120)
    hostel: str
    daily_plan_req: str = Field(min_length=40, max_length=900)
    innate: str = Field(min_length=40, max_length=900)
    learned: str = Field(min_length=20, max_length=900)
    lifestyle: str = Field(min_length=40, max_length=900)
    hobbies: str = Field(min_length=20, max_length=900)
    goals: str = Field(min_length=30, max_length=900)
    interests: List[str] = Field(min_length=3, max_length=12)


def _require_stopped() -> Optional[JSONResponse]:
    if _simulation_is_running():
        return JSONResponse({"error": "Stop the simulation before editing the roster."}, status_code=409)
    return None


def _safe_agent_id(name: str) -> str:
    result = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
    return result or "agent"


def _short_term_dir_for(name: str) -> Path:
    return Path(DATA_DIR) / "Short_term_db" / _safe_agent_id(name)


def _persona_file_for(name: str) -> Optional[Path]:
    root = Path(DATA_DIR) / "personalities"
    for path in root.glob("**/*.json"):
        try:
            if json.loads(path.read_text(encoding="utf-8")).get("Name") == name:
                return path
        except (OSError, json.JSONDecodeError):
            continue
    return None


def _save_roster_successor(world, registry, engine_state: dict) -> int:
    """Persist a roster edit without mutating its source checkpoint."""
    from src.core.checkpoint_manager import latest_tick, save_checkpoint
    # Same simulation minute, next storage tick: previous checkpoint remains
    # a faithful historical roster while /start picks this successor.
    successor = max(int(world.tick), int(latest_tick() or 0)) + 1
    world.tick = successor
    save_checkpoint(world, registry, successor, engine_state)
    return successor


def _load_stopped_roster():
    from src.core.checkpoint_manager import list_checkpoints, load_checkpoint
    from src.core.world_engine import WorldEngine
    ticks = list_checkpoints()
    if not ticks:
        raise ValueError("No checkpoint exists yet. Start and stop a fresh simulation before editing its roster.")
    engine = WorldEngine()
    world, registry, state = load_checkpoint(ticks[-1], engine.resolver, return_metadata=True)
    engine.world, engine.registry = world, registry
    engine.restore_checkpoint_state(state)
    return engine, state


@app.get("/api/roster")
async def get_roster():
    from src.core.checkpoint_manager import list_checkpoints, load_checkpoint
    from src.core.world_engine import WorldEngine
    ticks = list_checkpoints()
    if not ticks:
        return {"agents": [], "editable": not _simulation_is_running()}
    engine = WorldEngine()
    _, registry, _ = load_checkpoint(ticks[-1], engine.resolver, return_metadata=True)
    return {"editable": not _simulation_is_running(), "agents": [
        {"id": state.agent_id, "name": state.persona_name, "branch": state.persona.get("Branch", ""), "hostel": state.persona.get("Hostel", "")}
        for state in registry.all_states()
    ]}


@app.post("/api/roster/remove")
async def remove_agent(request: RemoveAgentInput, _user: dict = Depends(require_admin)):
    blocked = _require_stopped()
    if blocked:
        return blocked
    try:
        engine, state = _load_stopped_roster()
        removed = engine.registry.remove(request.agent_id)
    except (ValueError, KeyError) as exc:
        return JSONResponse({"error": str(exc)}, status_code=404)

    engine.world.remove_agent(request.agent_id)
    engine.relationship_matrix.remove_agent(request.agent_id)
    archive = Path(DATA_DIR) / "retired_agents" / datetime.now().strftime("%Y%m%d_%H%M%S") / request.agent_id
    archive.mkdir(parents=True, exist_ok=True)
    persona_path = _persona_file_for(removed.persona_name)
    if persona_path:
        shutil.move(str(persona_path.parent), str(archive / "persona"))
    short_term = _short_term_dir_for(removed.persona_name)
    if short_term.exists():
        shutil.move(str(short_term), str(archive / "short_term"))
    try:
        from src.agents.Long_term import get_retriever
        delete = getattr(get_retriever(), "delete_agent_memory", None)
        if callable(delete):
            delete(request.agent_id)
    except Exception as exc:
        logger.warning("[Roster] could not delete semantic memory for %s: %s", request.agent_id, exc)
    tick = _save_roster_successor(engine.world, engine.registry, engine.checkpoint_state())
    return {"status": "removed", "agent_id": request.agent_id, "checkpoint_tick": tick, "archive": str(archive)}


@app.post("/api/roster/replace")
async def replace_agent(request: ReplaceAgentInput, _user: dict = Depends(require_admin)):
    blocked = _require_stopped()
    if blocked:
        return blocked
    try:
        engine, state = _load_stopped_roster()
        agent = engine.registry.get(request.agent_id)
    except (ValueError, KeyError) as exc:
        return JSONResponse({"error": str(exc)}, status_code=404)
    old_name = agent.persona_name
    if _persona_file_for(request.name):
        return JSONResponse({"error": f"An agent named '{request.name}' already exists."}, status_code=409)
    agent.persona_name = request.name
    agent.persona["Name"] = request.name
    persona_path = _persona_file_for(old_name)
    if persona_path:
        new_dir = persona_path.parent.parent / _safe_agent_id(request.name)
        new_dir.mkdir(parents=True, exist_ok=True)
        (new_dir / f"{_safe_agent_id(request.name)}.json").write_text(json.dumps(agent.persona, indent=2), encoding="utf-8")
        shutil.rmtree(persona_path.parent)
    old_short = _short_term_dir_for(old_name)
    if old_short.exists():
        old_short.rename(_short_term_dir_for(request.name))
    engine.relationship_matrix.replace_display_name(old_name, request.name)
    tick = _save_roster_successor(engine.world, engine.registry, engine.checkpoint_state())
    return {"status": "replaced", "agent_id": request.agent_id, "name": request.name, "checkpoint_tick": tick}


@app.post("/api/roster/add")
async def add_agent(request: AddAgentInput, _user: dict = Depends(require_admin)):
    """Generate an adult student persona from observer-provided notes, while stopped."""
    blocked = _require_stopped()
    if blocked:
        return blocked
    try:
        engine, state = _load_stopped_roster()
        from src.agents.day_planner import load_places
        hostels = [place.id for place in load_places() if place.type == "residential"]
        from src.llm.gemini_client import call_gemini
        generated = call_gemini(
            "Create one grounded adult (18-30) Indian college-student simulation persona. "
            "Keep it respectful and non-explicit: no harassment, coercion, discrimination, stalking, or pornography. "
            "Balance academics with routines, friendship, hobbies, flaws, and downtime. Use exactly one supplied hostel id.",
            f"OBSERVER NOTES:\n{request.description}\n\nAVAILABLE HOSTEL IDS: {', '.join(hostels)}\n"
            "Return a complete persona matching the requested schema.",
            GeneratedPersona,
            "default",
        )
    except Exception as exc:
        logger.exception("[Roster] agent generation failed")
        return JSONResponse({"error": f"Could not generate agent: {exc}"}, status_code=502)

    agent_id = _safe_agent_id(generated.name)
    if agent_id in engine.registry:
        return JSONResponse({"error": f"An agent with id '{agent_id}' already exists."}, status_code=409)
    if generated.hostel not in hostels:
        return JSONResponse({"error": "Generated persona selected an invalid hostel."}, status_code=502)
    persona = {
        "Name": generated.name, "Age": str(generated.age), "Gender": generated.gender,
        "Branch": generated.branch, "Home City": generated.home_city, "Hostel": generated.hostel,
        "daily_plan_req": generated.daily_plan_req, "innate": generated.innate,
        "learned": generated.learned, "lifestyle": generated.lifestyle,
        "hobbies": generated.hobbies, "goals": generated.goals, "interests": generated.interests,
    }
    # A roster addition is not merely a display record: generate the same
    # executable remaining-day plan used by the regular runtime before the
    # successor checkpoint is committed. If the provider cannot do so, leave
    # every persisted surface untouched and report the failure to the observer.
    try:
        from types import SimpleNamespace
        from src.agents.day_planner import run as plan_day
        current_hhmm = engine._minutes_to_hhmm(engine.world.tick % (24 * 60))
        current_date = (datetime.strptime(engine.sim_start_date, "%Y-%m-%d") + timedelta(days=engine.world.tick // (24 * 60))).strftime("%Y-%m-%d")
        proxy = SimpleNamespace(persona=persona, relevant_memories=[], yesterday_summary=None)
        plan_result = await asyncio.get_running_loop().run_in_executor(None, lambda: plan_day(proxy, {
            "current_time": f"{current_date} {current_hhmm}", "places": None,
            "persona_name": generated.name, "mode": "remaining" if engine.world.tick else "full_day",
            "current_location_id": generated.hostel, "upcoming_events": [],
            "energy_level": engine._energy_baseline(persona),
            "emotion_state": engine._emotion_baseline(persona),
        }))
        day_plan = plan_result.get("day_plan", [])
        if not day_plan:
            raise ValueError(plan_result.get("error") or "planner returned no executable actions")
    except Exception as exc:
        return JSONResponse({"error": f"Persona was generated but no executable plan could be created: {exc}"}, status_code=502)
    from src.agents.Actions import AgentActionManager
    from src.core.agent_registry import AgentRuntimeState
    position = engine.resolver.random_interior_point(generated.hostel)
    manager = AgentActionManager(agent_id, day_plan, position, engine.resolver)
    engine.registry.register(AgentRuntimeState(
        agent_id=agent_id, persona=persona, persona_name=generated.name,
        manager=manager, position=position, day_plan=day_plan,
        energy_level=engine._energy_baseline(persona),
        emotion_state=engine._emotion_baseline(persona), emotion_baseline=engine._emotion_baseline(persona),
    ))
    engine.world.register_agent(agent_id, position)
    persona_dir = Path(DATA_DIR) / "personalities" / agent_id
    persona_dir.mkdir(parents=True, exist_ok=False)
    (persona_dir / f"{agent_id}.json").write_text(json.dumps(persona, indent=2), encoding="utf-8")

    from src.agents.conversation import RelationshipRecord
    for other in engine.registry.all_states():
        if other.agent_id == agent_id:
            continue
        engine.relationship_matrix.seed(agent_id, other.agent_id, RelationshipRecord(
            score=0.32, tags=["new-acquaintance"],
            context=f"{generated.name} is new to this circle and is still learning {other.persona_name}'s rhythm."
        ))
        engine.relationship_matrix.seed(other.agent_id, agent_id, RelationshipRecord(
            score=0.32, tags=["new-acquaintance"],
            context=f"{other.persona_name} has only recently met {generated.name}; the connection is open but untested."
        ))
    tick = _save_roster_successor(engine.world, engine.registry, engine.checkpoint_state())
    return {"status": "added", "agent_id": agent_id, "name": generated.name, "checkpoint_tick": tick}


@app.post("/api/sim/fast-forward")
async def fast_forward_sim(_user: dict = Depends(require_admin)):
    """Increase the live tick-speed multiplier without restarting the day."""
    from src import config as _cfg

    _cfg.TICK_SPEED = min(32.0, max(1.0, _cfg.TICK_SPEED * 2.0))
    return {"status": "ok", "speed_multiplier": _cfg.TICK_SPEED}


@app.post("/api/sim/slow-down")
async def slow_down_sim(_user: dict = Depends(require_admin)):
    """Halve the live tick-speed multiplier without restarting the day."""
    from src import config as _cfg

    # The floor avoids accidentally making a single tick take minutes while
    # still allowing an observer to inspect the simulation at quarter speed.
    _cfg.TICK_SPEED = max(0.25, _cfg.TICK_SPEED / 2.0)
    return {"status": "ok", "speed_multiplier": _cfg.TICK_SPEED}


@app.post("/api/sim/rewind")
async def rewind_sim(request: RewindInput, _user: dict = Depends(require_admin)):
    """Restore the nearest retained checkpoint for a requested tick or hour offset."""
    global _latest_snapshot
    from src.core.checkpoint_manager import list_checkpoints, load_checkpoint
    from src import config as _cfg

    try:
        requested_ticks = request.requested_ticks(_cfg.SIM_MINUTES_PER_TICK)
    except ValueError as exc:
        return JSONResponse({"error": str(exc)}, status_code=422)

    async with _sim_tick_lock:
        engine = _sim_engine
        if engine is None:
            return JSONResponse({"error": "Simulation is still initializing."}, status_code=409)
        current_tick = engine.world.tick
        target_tick = current_tick - requested_ticks
        available = [tick for tick in list_checkpoints() if tick <= target_tick]
        if not available:
            return JSONResponse({
                "error": f"Cannot rewind {requested_ticks} ticks; no retained checkpoint before tick {target_tick}.",
                "current_tick": current_tick,
            }, status_code=409)

        restore_tick = available[-1]
        # Decisions and conversations created after the target checkpoint
        # belong to the discarded future. Cancel both before replacing state.
        background_tasks = [
            *engine._conversation_tasks.values(),
            *engine._decision_tasks.values(),
        ]
        for task in background_tasks:
            task.cancel()
        if background_tasks:
            await asyncio.gather(*background_tasks, return_exceptions=True)
        engine._conversation_tasks.clear()
        engine._decision_tasks.clear()

        world, registry, state = load_checkpoint(restore_tick, engine.resolver, return_metadata=True)
        # A checkpoint is the authority for the roster at its timeline point.
        engine.world = world
        engine.registry = registry
        engine.restore_checkpoint_state(state)
        await engine.resume_pending_conversations()
        _latest_snapshot = engine.current_frontend_snapshot()
        _latest_snapshot["rewound_from_tick"] = current_tick
        _latest_snapshot["rewound_ticks"] = current_tick - restore_tick

    await _sim_broadcaster.broadcast(_latest_snapshot)
    _print_sim_status(_latest_snapshot)
    return {"status": "rewound", "tick": restore_tick, "rewound_ticks": current_tick - restore_tick}


@app.get("/api/sim/state")
async def get_sim_state():
    """Return the latest tick snapshot (or status if not yet running)."""
    return _snapshot_with_simulation_status(_latest_snapshot)


@app.websocket("/ws/sim")
async def sim_websocket(ws: WebSocket):
    """Subscribe to live tick-by-tick simulation state."""
    await _sim_broadcaster.connect(ws, _latest_snapshot)
    try:
        while True:
            msg = await ws.receive_text()
            if msg == "ping":
                await ws.send_text("pong")
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


@app.get("/api/assets")
def get_asset_urls():
    """Return public map URLs configured on the running Space.

    Hugging Face Space variables are runtime environment variables, while the
    Vite bundle is built earlier. The browser reads this endpoint so changing
    a Space variable does not require committing image files to the Space.
    """
    return {
        "map_image_url": _public_asset_url(os.environ.get("MAP_IMAGE_URL", "")),
        "night_image_url": _public_asset_url(os.environ.get("MAP_NIGHT_IMAGE_URL", "")),
    }


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

@app.api_route("/{full_path:path}", methods=["GET", "HEAD"])
async def serve_spa(full_path: str):
    dist = os.path.join(FRONTEND, "dist")
    public = os.path.join(FRONTEND, "public")
    
    # 1. Try dist directory first
    if os.path.isdir(dist) and full_path:
        file_path = os.path.join(dist, full_path)
        if os.path.isfile(file_path):
            return FileResponse(file_path)
            
    # 2. Try frontend/public directory (source static assets)
    if os.path.isdir(public) and full_path:
        pub_path = os.path.join(public, full_path)
        if os.path.isfile(pub_path):
            return FileResponse(pub_path)
            
    # 3. Try frontend root directory
    if full_path:
        fe_path = os.path.join(FRONTEND, full_path)
        if os.path.isfile(fe_path):
            return FileResponse(fe_path)

    # 4. Fallback to index.html for SPA client-side routing
    if os.path.isdir(dist):
        return FileResponse(os.path.join(dist, "index.html"))
    return FileResponse(os.path.join(FRONTEND, "index.html"))


if __name__ == "__main__":
    import argparse
    invalid_resume_flags = [arg for arg in sys.argv[1:] if arg.startswith("--resume") and arg != "--resume-checkpoint"]
    if invalid_resume_flags:
        raise SystemExit(
            "Invalid resume flag: " + ", ".join(invalid_resume_flags)
            + ". Did you mean --resume-checkpoint?"
        )
    parser = argparse.ArgumentParser(description="Valhalla simulation server")
    parser.add_argument(
        "--resume-checkpoint", action="store_true", default=False,
        help="Resume the latest checkpoint without clearing runtime data",
    )
    parser.add_argument(
        "--port", type=int, default=int(os.environ.get("PORT", "8000")),
        help="Port to listen on (default: 8000)",
    )
    parser.add_argument(
        "--host", type=str, default=os.environ.get("HOST", "127.0.0.1"),
        help="Host to bind to (default: 127.0.0.1, use 0.0.0.0 for public)",
    )
    args, _ = parser.parse_known_args()

    # Reserve the address before ASGI startup. Uvicorn normally starts the
    # lifespan before binding, which used to let a doomed duplicate launch
    # clear state and spend model calls without ever serving the browser.
    try:
        listen_socket = _reserve_listen_socket(args.host, args.port)
    except OSError as exc:
        print(
            f"\n  Cannot start Valhalla on http://{args.host}:{args.port}: {exc}.\n"
            f"  Stop the existing server or choose a free port, e.g. --port {args.port + 1}.\n",
            file=sys.stderr,
        )
        raise SystemExit(1)

    if args.resume_checkpoint:
        print(f"\n  Valhalla is live => http://{args.host}:{args.port}   (resuming from checkpoint)\n")
    else:
        print(f"\n  Valhalla is live => http://{args.host}:{args.port}   (fresh simulation; runtime data will be cleared)\n")

    config = uvicorn.Config(app, host=args.host, port=args.port, reload=False)
    try:
        uvicorn.Server(config).run(sockets=[listen_socket])
    except KeyboardInterrupt:
        print("\n[Server] Valhalla stopped cleanly.")
    finally:
        listen_socket.close()
