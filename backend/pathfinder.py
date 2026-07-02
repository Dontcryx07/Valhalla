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
    _path_img = Image.open(os.path.join(ROOT, "frontend", "path_map.png")).convert("RGB")
    _W, _H = _path_img.size
    pix = _path_img.load()
    _white_pixels = set()
    for y in range(_H):
        for x in range(_W):
            r, g, b = pix[x, y]
            if r == 255:
                _white_pixels.add((x, y))


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
    if start not in _white_pixels or end not in _white_pixels:
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
