"""
Core pathfinding module — imported by Odin.py and pixel_pathfinder.py.
Loads path.png into a set of walkable (white) pixels and provides
BFS shortest_path(), stats(), and is_walkable() helpers.
"""

import os
import threading
from io import BytesIO
from collections import deque
from urllib.parse import urlparse
from urllib.request import Request, urlopen
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_path_img = None
_white_pixels = None
_W = _H = 0
_load_lock = threading.Lock()


def _public_asset_url(value):
    """Convert a GitHub blob page URL to its downloadable raw-file URL."""
    value = value.strip()
    parsed = urlparse(value)
    parts = parsed.path.strip("/").split("/")
    if parsed.scheme == "https" and parsed.netloc in {"github.com", "www.github.com"} \
            and len(parts) >= 5 and parts[2] == "blob":
        owner, repository, _, revision, *path = parts
        return f"https://raw.githubusercontent.com/{owner}/{repository}/{revision}/{'/'.join(path)}"
    return value


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
        path_url = _public_asset_url(os.environ.get("PATH_IMAGE_URL", ""))
        try:
            if path_file:
                image_source = path_file
            else:
                if not path_url:
                    raise FileNotFoundError("path.png not found and PATH_IMAGE_URL is not configured")
                if urlparse(path_url).scheme != "https":
                    raise ValueError("PATH_IMAGE_URL must be an https URL")
                request = Request(path_url, headers={"User-Agent": "Valhalla/1.0"})
                with urlopen(request, timeout=20) as response:
                    image_bytes = response.read(20 * 1024 * 1024 + 1)
                if len(image_bytes) > 20 * 1024 * 1024:
                    raise ValueError("PATH_IMAGE_URL exceeds the 20 MB download limit")
                image_source = BytesIO(image_bytes)

            with Image.open(image_source) as image:
                rgb = image.convert("RGB")
                _W, _H = rgb.size
                pix = rgb.load()
                _white_pixels = {
                    (x, y)
                    for y in range(_H)
                    for x in range(_W)
                    if pix[x, y] == (255, 255, 255)
                }
        except Exception as exc:
            import logging
            logging.getLogger(__name__).warning("Failed to load path.png (%s). Falling back to open map grid.", exc)
            _W, _H = 1276, 1233
            _white_pixels = {(x, y) for y in range(_H) for x in range(_W)}


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
