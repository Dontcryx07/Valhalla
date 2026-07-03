"""
Tile-based navigation graph for the Valhalla map.
Loads agent_path_tiles.json, runs BFS on a 319×411 tile grid (4×3 px each),
and visualizes paths on map.png via matplotlib. The predecessor to the
pixel-level pathfinder.
"""

import json
import sys
from collections import deque
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt

COLS = 319
ROWS = 411
TILE_W = 4
TILE_H = 3

BASE_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = BASE_DIR.parent / "frontend"

with open(FRONTEND_DIR / "agent_path_tiles.json") as f:
    data = json.load(f)
TRAVERSABLE = set(data["traversable_tiles"])


def neighbors(tile_id):
    col = tile_id % COLS
    row = tile_id // COLS
    nbrs = []
    if row > 0:
        n = (row - 1) * COLS + col
        if n in TRAVERSABLE: nbrs.append(n)
    if row < ROWS - 1:
        n = (row + 1) * COLS + col
        if n in TRAVERSABLE: nbrs.append(n)
    if col > 0:
        n = row * COLS + (col - 1)
        if n in TRAVERSABLE: nbrs.append(n)
    if col < COLS - 1:
        n = row * COLS + (col + 1)
        if n in TRAVERSABLE: nbrs.append(n)
    return nbrs


def shortest_path(start_id, end_id):
    if start_id not in TRAVERSABLE or end_id not in TRAVERSABLE:
        return None
    q = deque([(start_id, [start_id])])
    seen = {start_id}
    while q:
        cur, path = q.popleft()
        if cur == end_id:
            return path
        for n in neighbors(cur):
            if n not in seen:
                seen.add(n)
                q.append((n, path + [n]))
    return None


def tile_center(tile_id):
    return ((tile_id % COLS) * TILE_W + TILE_W // 2,
            (tile_id // COLS) * TILE_H + TILE_H // 2)


def find_components():
    unvisited = set(TRAVERSABLE)
    comps = []
    while unvisited:
        seed = next(iter(unvisited))
        q = [seed]
        comp = set()
        while q:
            cur = q.pop()
            if cur not in unvisited:
                continue
            unvisited.remove(cur)
            comp.add(cur)
            for nb in neighbors(cur):
                if nb in unvisited:
                    q.append(nb)
        comps.append(comp)
    comps.sort(key=len, reverse=True)
    return comps


def show_path_on_map(path):
    img = Image.open(FRONTEND_DIR / "map.png").convert("RGBA")
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    for tid in TRAVERSABLE:
        col = tid % COLS
        row = tid // COLS
        x0, y0 = col * TILE_W, row * TILE_H
        draw.rectangle([x0, y0, x0 + TILE_W - 1, y0 + TILE_H - 1], fill=(0, 255, 0, 20))

    if path:
        pts = [tile_center(t) for t in path]
        draw.line(pts, fill=(255, 30, 30, 255), width=3)

        sx, sy = tile_center(path[0])
        ex, ey = tile_center(path[-1])
        r = 6
        draw.ellipse([sx - r, sy - r, sx + r, sy + r], fill=(0, 220, 0, 255))
        draw.ellipse([ex - r, ey - r, ex + r, ey + r], fill=(30, 100, 255, 255))

        try:
            font = ImageFont.truetype("arial.ttf", 14)
        except (IOError, OSError):
            font = ImageFont.load_default()
        draw.text((sx + 8, sy - 8), "S", fill=(0, 220, 0, 255), font=font)
        draw.text((ex + 8, ey - 8), "E", fill=(30, 100, 255, 255), font=font)

    composited = Image.alpha_composite(img, overlay).convert("RGB")

    plt.figure(figsize=(14, 14))
    plt.imshow(composited)
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.show()


if __name__ == "__main__":
    cmds = [a for a in sys.argv[1:] if not a.startswith("--")]
    flags = set(a for a in sys.argv[1:] if a.startswith("--"))

    if "--help" in flags or "-h" in flags:
        print("Usage: python backend/nav_graph.py [start_tile_id end_tile_id]")
        print()
        print("  start_tile_id end_tile_id    pair of tile IDs to pathfind between")
        print("  --components                 list connected components and exit")
        sys.exit(0)

    if "--components" in flags:
        comps = find_components()
        print(f"Connected components: {len(comps)}")
        for i, c in enumerate(comps):
            print(f"  {i}: {len(c)} tiles")
        sys.exit(0)

    if len(cmds) >= 2:
        start, end = int(cmds[0]), int(cmds[1])
    else:
        start, end = 30953, 128158

    print(f"Graph: {len(TRAVERSABLE)} nodes")
    print(f"Path: {start} -> {end}  "
          f"(col={start%COLS}, row={start//COLS}) -> "
          f"(col={end%COLS}, row={end//COLS})")

    path = shortest_path(start, end)
    if path:
        print(f"Found: {len(path)} tiles")
        print("Displaying map...")
        show_path_on_map(path)
    else:
        print("No path found (tiles are in different connected components)")
        comps = find_components()
        for i, c in enumerate(comps):
            if start in c:
                print(f"  start is in component {i} ({len(c)} tiles)")
            if end in c:
                print(f"  end is in component {i} ({len(c)} tiles)")
