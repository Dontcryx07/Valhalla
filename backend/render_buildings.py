"""
Renders building bounding boxes onto map.png and saves the result.
Useful for a static reference — the web frontend renders buildings
dynamically via /api/buildings + canvas.
"""

import json, os
from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "backend", "data", "environment", "buildings_polygon_decomposed.json")
MAP = os.path.join(ROOT, "frontend", "map.png")
OUT = os.path.join(ROOT, "frontend", "map_with_buildings.png")

with open(DATA) as f:
    buildings = json.load(f)

img = Image.open(MAP).convert("RGBA")
overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
draw = ImageDraw.Draw(overlay)

try:
    font = ImageFont.truetype("arial.ttf", 11)
except:
    font = ImageFont.load_default()

label_seen = {}
for b in buildings:
    x1, y1 = b["top_left"]
    x2, y2 = b["bottom_right"]
    draw.rectangle([x1, y1, x2, y2], fill=(255, 0, 0, 77), outline=(200, 0, 0, 180), width=1)

    base = b["building_name"].rsplit("_part", 1)[0]
    if base not in label_seen:
        label_seen[base] = len(label_seen) + 1
        label = str(label_seen[base])
        bbox = draw.textbbox((0, 0), label, font=font)
        tw = bbox[2] - bbox[0]
        tx = x1 + 2
        ty = y1 - 4
        draw.rectangle([tx - 1, ty - bbox[3] + bbox[1] - 1, tx + tw + 1, ty + 3],
                       fill=(0, 0, 0, 160))
        draw.text((tx, ty), label, fill=(255, 255, 255, 230), font=font)

result = Image.alpha_composite(img, overlay)
result.convert("RGB").save(OUT)
print(f"Saved: {OUT}")
