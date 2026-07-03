"""
Generates the tile layout for the Valhalla map.
Splits the 1276×1233 pixel map into 4×3 pixel tiles (319×411 grid)
and writes each tile's ID, col, row, and pixel offset to map_tiles.jsonl.
"""

import json

WIDTH, HEIGHT = 1276, 1233
TILE_W, TILE_H = 4, 3
COLS = WIDTH // TILE_W
ROWS = HEIGHT // TILE_H

assert WIDTH % TILE_W == 0, f"Width {WIDTH} not divisible by {TILE_W}"
assert HEIGHT % TILE_H == 0, f"Height {HEIGHT} not divisible by {TILE_H}"

OUTPUT = "frontend/map_tiles.jsonl"

tile_id = 0
with open(OUTPUT, "w", encoding="utf-8") as f:
    for row in range(ROWS):
        for col in range(COLS):
            x = col * TILE_W
            y = row * TILE_H
            line = json.dumps({
                "tile_id": tile_id,
                "col": col,
                "row": row,
                "x": x,
                "y": y,
            }, separators=(",", ":"))
            f.write(line + "\n")
            tile_id += 1

print(f"Generated {tile_id} tiles -> {OUTPUT}")
