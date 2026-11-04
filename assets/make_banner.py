"""Build the README banner of eval-3d: one object, ground truth plus five methods, accuracy error in viridis.

Usage:
    python make_banner.py --renders <dir> --out assets/banner.png

<dir> holds one sub-folder per column, each with view_000.png (RGBA on a transparent background, or RGB on white):
gt_uniform, neuralangelo_d4, neus2_d4, pgsr_d4, realityscan_d2, meshroom_d4. The error maps are unlit viridis
renders, vmin 0 and vmax 4 mm (eval-pipeline visualization.max_dist). The banner is a 1600x400 RGBA PNG: rounded
#111111 card with a #30363d border (readable on GitHub light and dark themes) and the Martine palette gradient rule,
shared by the banners of the three Martine repositories. Requires numpy, opencv-python, Pillow and matplotlib.
"""

import argparse
import os

import cv2
import matplotlib
import numpy as np
from PIL import Image, ImageDraw, ImageFont

W, H, RADIUS = 1600, 400, 24
CARD, BORDER, TEXT, MUTED = "#111111", "#30363d", "#e6edf3", "#9198a1"
PALETTE = ["#003d5c", "#5749a5", "#b35280", "#f64a4b", "#f8a700"]
COLUMNS = [("gt_uniform", "Ground truth"), ("neuralangelo_d4", "Neuralangelo"), ("neus2_d4", "NeuS2"),
           ("pgsr_d4", "PGSR"), ("realityscan_d2", "RealityScan"), ("meshroom_d4", "Meshroom")]
VMAX_MM = 4
FONTS = ["DejaVuSans.ttf", "/usr/share/fonts/dejavu/DejaVuSans.ttf",
         "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]


def rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def font(size, path=None):
    for f in ([path] if path else []) + FONTS:
        try:
            return ImageFont.truetype(f, size)
        except OSError:
            pass
    return ImageFont.load_default(size)


def load(path):
    """RGBA render cropped to its content; a white background is keyed out when there is no alpha."""
    im = Image.open(path)
    if im.mode != "RGBA" or np.array(im.getchannel("A")).min() == 255:
        px = np.array(im.convert("RGB"))
        reg = (px.min(axis=2) >= 247).astype(np.uint8)
        ff = np.zeros((reg.shape[0] + 2, reg.shape[1] + 2), np.uint8)
        for y, x in [(0, 0), (0, reg.shape[1] - 1), (reg.shape[0] - 1, 0), (reg.shape[0] - 1, reg.shape[1] - 1)]:
            if reg[y, x] == 1:
                cv2.floodFill(reg, ff, (x, y), 2)
        a = cv2.GaussianBlur(np.where(reg == 2, 0, 255).astype(np.uint8), (3, 3), 0)
        im = Image.fromarray(np.dstack([px, a]), "RGBA")
    im = im.convert("RGBA")
    return im.crop(im.getchannel("A").point(lambda v: 255 if v > 8 else 0).getbbox())


def card():
    base = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    mask = Image.new("L", (W, H), 0)
    ImageDraw.Draw(mask).rounded_rectangle([0, 0, W - 1, H - 1], RADIUS, fill=255)
    base.paste(Image.new("RGBA", (W, H), rgb(CARD) + (255,)), (0, 0), mask)
    return base, mask


def finish(base, mask, rule_h=6):
    xs = np.linspace(0, len(PALETTE) - 1, W)
    cols = np.array([rgb(c) for c in PALETTE], float)
    i = np.clip(xs.astype(int), 0, len(PALETTE) - 2)
    t = (xs - i)[:, None]
    row = cols[i] * (1 - t) + cols[i + 1] * t
    rule = np.dstack([np.repeat(row[None], rule_h, 0), np.full((rule_h, W), 255)]).astype(np.uint8)
    base.alpha_composite(Image.fromarray(rule, "RGBA"), (0, H - rule_h))
    ImageDraw.Draw(base).rounded_rectangle([0, 0, W - 1, H - 1], RADIUS, outline=rgb(BORDER) + (255,), width=2)
    base.putalpha(Image.fromarray(np.minimum(np.array(base.getchannel("A")), np.array(mask))))
    return base


def colorbar(base, x1, y, w, h, f):
    """Horizontal viridis bar ending at x1, with its 0 and VMAX labels and a caption on the left."""
    row = (matplotlib.colormaps["viridis"](np.linspace(0, 1, w))[:, :3] * 255).astype(np.uint8)
    bar = Image.fromarray(np.repeat(row[None], h, 0)).convert("RGBA")
    m = Image.new("L", (w, h), 0)
    ImageDraw.Draw(m).rounded_rectangle([0, 0, w - 1, h - 1], h // 2, fill=255)
    bar.putalpha(m)
    d = ImageDraw.Draw(base)
    tail = f"{VMAX_MM} mm"
    x0 = x1 - w - 12 - d.textlength(tail, font=f)
    base.alpha_composite(bar, (round(x0), y))
    d.text((x0 - 12, y + h / 2), "0", fill=rgb(MUTED), font=f, anchor="rm")
    d.text((x0 + w + 12, y + h / 2), tail, fill=rgb(MUTED), font=f, anchor="lm")
    zero_w = d.textlength("0", font=f)
    d.text((x0 - 12 - zero_w - 18, y + h / 2), "accuracy error", fill=rgb(MUTED), font=f, anchor="rm")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--renders", required=True, help="directory with <column>/view_000.png")
    ap.add_argument("--out", default="banner.png")
    ap.add_argument("--font", default=None, help="TrueType font (default: DejaVu Sans)")
    a = ap.parse_args()

    base, mask = card()
    pad, gap, top, tile_h = 34, 14, 26, 262
    tw = (W - 2 * pad - gap * (len(COLUMNS) - 1)) / len(COLUMNS)
    ims = [load(os.path.join(a.renders, key, "view_000.png")) for key, _ in COLUMNS]
    s = min(min(tw / im.width, tile_h / im.height) for im in ims)  # one scale for all columns: sizes compare
    d = ImageDraw.Draw(base)
    f_lab, f_small = font(24, a.font), font(19, a.font)
    for i, (im, (_, label)) in enumerate(zip(ims, COLUMNS)):
        im = im.resize((round(im.width * s), round(im.height * s)), Image.LANCZOS)
        x = pad + i * (tw + gap)
        base.alpha_composite(im, (round(x + (tw - im.width) / 2), round(top + (tile_h - im.height) / 2)))
        d.text((x + tw / 2, top + tile_h + 26), label, fill=rgb(TEXT if i else MUTED), font=f_lab, anchor="mm")
    colorbar(base, W - pad, H - 52, 220, 12, f_small)
    out = finish(base, mask)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    out.save(a.out, optimize=True)
    print(f"wrote {a.out} ({W}x{H})")


if __name__ == "__main__":
    main()
