#!/usr/bin/env python3
"""Assemble curvature comparison grid: 4 objects (rows) x 6 radii (cols).

Reads renders from each object's curvature_tests/ directory.
"""

from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import sys


EVAL_ROOT = Path("/home/babrument/dev/alicevision/dataset_temp/eval")
OBJECTS = ["11_ecorce", "01_rock", "06_can", "09_jam_pot"]
RADII_TAGS = ["current", "r0.5mm", "r1.0mm", "r2.0mm", "r4.0mm", "r8.0mm"]
RADII_LABELS = ["current\n(5×edge)", "r=0.5mm", "r=1mm", "r=2mm", "r=4mm", "r=8mm"]
VIEW = "view_000"
OUTPUT = EVAL_ROOT / "curvature_comparison.png"


def load_image(obj, tag):
    render_dir = EVAL_ROOT / obj / "Groundtruth" / "curvature_tests" / tag / "renders"
    candidates = sorted(render_dir.glob("view_*.png"))
    if candidates:
        return Image.open(candidates[0])
    return None


def main():
    images = {}
    max_w, max_h = 0, 0

    for obj in OBJECTS:
        for tag in RADII_TAGS:
            img = load_image(obj, tag)
            if img is not None:
                images[(obj, tag)] = img
                max_w = max(max_w, img.width)
                max_h = max(max_h, img.height)

    if not images:
        print("No images found. Jobs may still be running.")
        sys.exit(1)

    n_found = len(images)
    n_expected = len(OBJECTS) * len(RADII_TAGS)
    print(f"Found {n_found}/{n_expected} images")

    MAX_GRID_W = 3000
    label_h = 40
    row_label_w = 120
    padding = 4

    n_cols = len(RADII_TAGS)
    avail_w = MAX_GRID_W - row_label_w - (n_cols + 1) * padding
    cell_w = avail_w // n_cols
    aspect = max_h / max_w if max_w > 0 else 1.0
    cell_h = int(cell_w * aspect)

    grid_w = row_label_w + n_cols * (cell_w + padding) + padding
    grid_h = label_h + len(OBJECTS) * (cell_h + padding) + padding

    grid = Image.new("RGB", (grid_w, grid_h), (255, 255, 255))
    draw = ImageDraw.Draw(grid)

    try:
        font = ImageFont.truetype("/usr/share/fonts/dejavu/DejaVuSans.ttf", 14)
        font_small = ImageFont.truetype("/usr/share/fonts/dejavu/DejaVuSans.ttf", 12)
    except Exception:
        font = ImageFont.load_default()
        font_small = font

    for j, (tag, label) in enumerate(zip(RADII_TAGS, RADII_LABELS)):
        x = row_label_w + padding + j * (cell_w + padding) + cell_w // 2
        for k, line in enumerate(label.split("\n")):
            bbox = draw.textbbox((0, 0), line, font=font_small)
            tw = bbox[2] - bbox[0]
            draw.text((x - tw // 2, 4 + k * 16), line, fill=(0, 0, 0), font=font_small)

    for i, obj in enumerate(OBJECTS):
        y = label_h + padding + i * (cell_h + padding)
        draw.text((4, y + cell_h // 2 - 8), obj.replace("_", "\n"), fill=(0, 0, 0), font=font_small)

        for j, tag in enumerate(RADII_TAGS):
            x = row_label_w + padding + j * (cell_w + padding)
            img = images.get((obj, tag))
            if img is not None:
                resized = img.resize((cell_w, cell_h), Image.LANCZOS)
                grid.paste(resized, (x, y))
            else:
                draw.rectangle([x, y, x + cell_w, y + cell_h], fill=(200, 200, 200))
                draw.text((x + 10, y + cell_h // 2), "pending", fill=(128, 128, 128), font=font_small)

    grid.save(OUTPUT)
    print(f"Saved: {OUTPUT}")
    print(f"Size: {grid_w}x{grid_h}")


if __name__ == "__main__":
    main()
