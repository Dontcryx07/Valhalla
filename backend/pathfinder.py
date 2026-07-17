"""
Core pathfinding module — imported by server.py and pixel_pathfinder.py.
Loads path.png into a set of walkable (white) pixels and provides
BFS shortest_path(), stats(), and is_walkable() helpers.
"""

import os
from collections import deque
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_path_img = None
_white_pixels = None
_W = _H = 0


def _load():
    global _path_img, _white_pixels, _W, _H
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
    _path_img = Image.open(path_file).convert("RGB")
    _W, _H = _path_img.size
    pix = _path_img.load()
    _white_pixels = set()
    for y in range(_H):
        for x in range(_W):
            r, g, b = pix[x, y]
            if r == 255:
                _white_pixels.add((x, y))


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
    q = deque([(start, [start])])
    seen = {start}
    while q:
        cur, path = q.popleft()
        if cur == end:
            return path
        for nb in neighbors(*cur):
            if nb not in seen and nb in _white_pixels:
                seen.add(nb)
                q.append((nb, path + [nb]))
    return None
