"""
CLI tool for pixel-level pathfinding on the Valhalla map.
Loads map.png, computes the shortest path between two pixel coordinates
via BFS, and displays the result with start/end markers in a matplotlib window.
"""

import sys
import os
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
from pathfinder import shortest_path, stats, ROOT


def show_path_on_map(path, start, end, save_path=None):
    img = Image.open(os.path.join(ROOT, "frontend", "map.png")).convert("RGBA")
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    if path:
        draw.line(path, fill=(255, 30, 30, 255), width=2)

        r = 5
        sx, sy = start
        ex, ey = end
        draw.ellipse([sx - r, sy - r, sx + r, sy + r], fill=(0, 220, 0, 255))
        draw.ellipse([ex - r, ey - r, ex + r, ey + r], fill=(30, 100, 255, 255))

        try:
            font = ImageFont.truetype("arial.ttf", 14)
        except (IOError, OSError):
            font = ImageFont.load_default()
        draw.text((sx + 7, sy - 9), "S", fill=(0, 220, 0, 255), font=font)
        draw.text((ex + 7, ey - 9), "E", fill=(30, 100, 255, 255), font=font)

    composited = Image.alpha_composite(img, overlay).convert("RGB")

    if save_path:
        composited.save(os.path.join(ROOT, save_path))
        print(f"Saved: {save_path}")
    else:
        plt.figure(figsize=(14, 14))
        plt.imshow(composited)
        plt.axis("off")
        plt.tight_layout(pad=0)
        plt.show()


if __name__ == "__main__":
    cmds = [a for a in sys.argv[1:] if not a.startswith("--")]
    flags = set(a for a in sys.argv[1:] if a.startswith("--"))

    if "--help" in flags or "-h" in flags:
        print("Usage: python backend/pixel_pathfinder.py [x1 y1 x2 y2]")
        print()
        print("  x1 y1 x2 y2    pixel coordinates for start and end")
        print("  --count        count white pixels only")
        sys.exit(0)

    if "--count" in flags:
        s = stats()
        print(f"White pixels: {s['white_pixels']} / {s['width']*s['height']}")
        sys.exit(0)

    if len(cmds) >= 4:
        sx, sy, ex, ey = int(cmds[0]), int(cmds[1]), int(cmds[2]), int(cmds[3])
    else:
        sx, sy = 32, 297
        ex, ey = 959, 1205

    start = (sx, sy)
    end = (ex, ey)

    print(f"Path: ({sx},{sy}) -> ({ex},{ey})")

    path = shortest_path(start, end)
    if path:
        print(f"Found: {len(path)} pixels")
        show_path_on_map(path, start, end)
    else:
        print("No path found")
