"""
Core pathfinding module — imported by Odin.py and pixel_pathfinder.py.
Loads path.png into a set of walkable (white) pixels and provides
BFS shortest_path(), stats(), and is_walkable() helpers.
"""

import os
import threading
from collections import deque
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_path_img = None
_white_pixels = None
_W = _H = 0
_load_lock = threading.Lock()


def _load():
    global _path_img, _white_pixels, _W, _H
    if _white_pixels is not None:
        return
    with _load_lock:
        if _white_pixels is not None:
            return
        # path.png may live under frontend/, frontend/public/ (source) or
        # frontend/dist/ (after a build). Use whichever exists.
        candidates = [
            os.path.join(ROOT, "frontend", "public", "path.png"),
            os.path.join(ROOT, "frontend", "path.png"),
            os.path.join(ROOT, "frontend", "dist", "path.png"),
        ]
        path_file = next((p for p in candidates if os.path.exists(p)), None)
        if path_file is None:
            raise FileNotFoundError(
                "path.png not found in frontend/public, frontend, or frontend/dist"
            )
        # Copy decoded pixels while the image handle is open, then release the
        # handle immediately so the source image is not locked on Windows.
        with Image.open(path_file) as image:
            rgb = image.convert("RGB")
            _W, _H = rgb.size
            pix = rgb.load()
            _white_pixels = {
                (x, y)
                for y in range(_H)
                for x in range(_W)
                if pix[x, y] == (255, 255, 255)
            }


def _nearest_walkable(pt, max_r=60):
    """Return the closest walkable pixel to `pt` (or None). Lets us route from
    a building door / interior point that isn't exactly on a path pixel."""
    if pt in _white_pixels:
        return pt
    x0, y0 = pt
    for r in range(1, max_r + 1):
        for x in range(x0 - r, x0 + r + 1):
            if (x, y0 - r) in _white_pixels:
                return (x, y0 - r)
            if (x, y0 + r) in _white_pixels:
                return (x, y0 + r)
        for y in range(y0 - r + 1, y0 + r):
            if (x0 - r, y) in _white_pixels:
                return (x0 - r, y)
            if (x0 + r, y) in _white_pixels:
                return (x0 + r, y)
    return None


def stats():
    _load()
    return {"white_pixels": len(_white_pixels), "width": _W, "height": _H}


def is_walkable(x, y):
    _load()
    return (x, y) in _white_pixels


def neighbors(px, py):
    n = []
    if py > 0: n.append((px, py - 1))
    if py < _H - 1: n.append((px, py + 1))
    if px > 0: n.append((px - 1, py))
    if px < _W - 1: n.append((px + 1, py))
    return n


def shortest_path(start, end):
    _load()
    # Snap endpoints onto the walkable network (doors/interiors may sit just off it).
    start = _nearest_walkable(tuple(start))
    end = _nearest_walkable(tuple(end))
    if start is None or end is None:
        return None
    q = deque([start])
    parents = {start: None}
    while q:
        cur = q.popleft()
        if cur == end:
            path = []
            while cur is not None:
                path.append(cur)
                cur = parents[cur]
            path.reverse()
            return path
        for nb in neighbors(*cur):
            if nb not in parents and nb in _white_pixels:
                parents[nb] = cur
                q.append(nb)
    return None
