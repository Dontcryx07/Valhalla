"""
FastAPI web server for the Valhalla agent map.
Serves the frontend (index.html, map images), exposes REST + WebSocket
endpoints for pathfinding (/api/path, /api/path/stream, /ws), and
streams path points for real-time agent animation.
"""

import os
import sys
import json
import asyncio
from pathlib import Path
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pathfinder import shortest_path, stats, ROOT

app = FastAPI(title="Valhalla Agent Map")

FRONTEND = os.path.join(ROOT, "frontend")
DATA_DIR = os.path.join(ROOT, "backend", "data")


@app.get("/api/buildings")
def get_buildings():
    with open(os.path.join(DATA_DIR, "environment", "buildings_polygon_decomposed.json")) as f:
        buildings = json.load(f)
    label_seen = {}
    simplified = []
    for b in buildings:
        tl = b["top_left"]
        br = b["bottom_right"]
        name = b["building_name"]
        base = name.rsplit("_part", 1)[0]
        if base not in label_seen:
            label_seen[base] = len(label_seen) + 1
        simplified.append({
            "name": name,
            "label": str(label_seen[base]),
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


app.mount("/", StaticFiles(directory=FRONTEND, html=True), name="frontend")

if __name__ == "__main__":
    uvicorn.run("server:app", host="127.0.0.1", port=8000, reload=True)
