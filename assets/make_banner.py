"""Build assets/banner.png: a row of six Martine objects shown as accuracy error maps.

Shared template of the Martine repositories: 1600x400, white background, no text, six 220 px square
tiles, vertically centred, equal spacing, 60 px side margins. The tiles are `visualize` renders
(`<eval_root>/<object>/<method>/visualizations/<metric>/view_XXX.png`), read only.

    python assets/make_banner.py --eval-root /path/to/eval --method realityscan_d1 \
        --views 01_rock:0 07_glass:0 17_knife:0 15_lego:0 25_plant:0 23_whisk:0
    python assets/make_banner.py ... --sheet sheet.png   # every available view, to pick them

Not run by the tests.
"""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

W, H, TILE, MARGIN = 1600, 400, 220, 60
OBJECTS = ["01_rock", "07_glass", "17_knife", "15_lego", "25_plant", "23_whisk"]


def tile(path: Path, size: int = TILE) -> Image.Image:
    """Crop the render to its non-white content, pad it to a white square, resize."""
    img = Image.open(path).convert("RGB")
    a = np.asarray(img).astype(int)
    fg = (np.abs(a - 255).sum(axis=2) > 30)
    ys, xs = np.nonzero(fg)
    if len(xs):
        img = img.crop((xs.min(), ys.min(), xs.max() + 1, ys.max() + 1))
    side = max(img.size)
    sq = Image.new("RGB", (side, side), "white")
    sq.paste(img, ((side - img.width) // 2, (side - img.height) // 2))
    return sq.resize((size, size), Image.LANCZOS)


def render_path(root: Path, obj: str, method: str, metric: str, view: int) -> Path:
    return root / obj / method / "visualizations" / metric / f"view_{view:03d}.png"


def banner(root: Path, method: str, metric: str, views: dict[str, int]) -> Image.Image:
    out = Image.new("RGB", (W, H), "white")
    gap = (W - 2 * MARGIN - len(OBJECTS) * TILE) / (len(OBJECTS) - 1)
    for i, obj in enumerate(OBJECTS):
        x = round(MARGIN + i * (TILE + gap))
        out.paste(tile(render_path(root, obj, method, metric, views.get(obj, 0))), (x, (H - TILE) // 2))
    return out


def sheet(root: Path, method: str, metric: str) -> Image.Image:
    """Contact sheet: one row per object, one column per available view, labelled."""
    rows = []
    for obj in OBJECTS:
        files = sorted((root / obj / method / "visualizations" / metric).glob("view_*.png"))
        rows.append([(f.stem, tile(f, 160)) for f in files])
    ncol = max(len(r) for r in rows)
    out = Image.new("RGB", (ncol * 170 + 10, len(rows) * 185 + 10), "white")
    d = ImageDraw.Draw(out)
    for r, row in enumerate(rows):
        for c, (name, im) in enumerate(row):
            out.paste(im, (10 + c * 170, 10 + r * 185))
            d.text((10 + c * 170, 172 + r * 185), f"{OBJECTS[r]} {name}", fill="black")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--eval-root", type=Path, required=True)
    ap.add_argument("--method", required=True)
    ap.add_argument("--metric", default="accuracy")
    ap.add_argument("--views", nargs="*", default=[], help="object:view pairs (default view 0)")
    ap.add_argument("--out", type=Path, default=Path(__file__).with_name("banner.png"))
    ap.add_argument("--sheet", type=Path, help="write a contact sheet of every view instead")
    a = ap.parse_args()
    if a.sheet:
        sheet(a.eval_root, a.method, a.metric).save(a.sheet)
        return
    views = {k: int(v) for k, v in (p.split(":") for p in a.views)}
    banner(a.eval_root, a.method, a.metric, views).save(a.out, optimize=True)


if __name__ == "__main__":
    main()
