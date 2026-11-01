"""
Draw the cold-plate designs of bench/cold_plate.py as a section through the
middle of the cavity, seen from above: metal dark, water light, dies
outlined, inlet and outlet marked. Geometry only (no flow solve, seconds).

    python bench/cold_plate_render.py --out docs/images/case-studies/cold-plate-designs.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from cold_plate import BASE, DIES, ENTRY, GAP, IN_Y, OUT_Y, L, W, Plate, build  # noqa: E402
from lattice_heatsink import keep_connected  # noqa: E402

DESIGNS = [
    ("pins-3", "Staggered pins, 3 mm\n42.5 K (0.1875 mm grid)"),
    ("channels-10", "Streamline channels, 10\n32.6 K (0.15 mm grid)"),
    ("graded-3-s4", "Gyroid graded over the dies\n32.3 K (0.15 mm grid)"),
    ("hybrid-8-2", "Channels + graded gyroid\n30.4 K (0.15 mm grid)"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pitch", type=float, default=0.15)
    ap.add_argument("--out", default="docs/images/case-studies/cold-plate-designs.png")
    a = ap.parse_args()
    p = Plate(a.pitch)
    k = int((BASE + 0.5 * GAP) / p.h)
    fig, axes = plt.subplots(1, len(DESIGNS), figsize=(4.2 * len(DESIGNS), 4.6), dpi=150)
    for ax, (name, title) in zip(axes, DESIGNS):
        fins = keep_connected(p, build(p, name))
        solid, fluid = p.masks(fins)
        img = np.full(solid.shape[:2], np.nan)
        img[fluid[:, :, k]] = 0.0
        img[solid[:, :, k]] = 1.0
        ax.imshow(img.T, origin="lower", cmap=matplotlib.colors.ListedColormap(["#cfe3f5", "#3b3f46"]),
                  extent=(-ENTRY, L, 0, W), interpolation="nearest", vmin=0, vmax=1)
        for (cx, cy), s, w in DIES:
            ax.add_patch(Rectangle((cx - s / 2, cy - s / 2), s, s, fill=False, ec="#e4572e", lw=1.6))
            ax.text(cx, cy + s / 2 + 0.6, f"{w:g} W", color="#e4572e", ha="center", fontsize=8)
        ax.annotate("in", xy=(0, np.mean(IN_Y)), xytext=(-4.6, np.mean(IN_Y)), color="#1f6fb2",
                    fontsize=9, va="center", arrowprops=dict(arrowstyle="->", color="#1f6fb2"))
        ax.annotate("out", xy=(-4.8, np.mean(OUT_Y)), xytext=(-0.2, np.mean(OUT_Y)), color="#1f6fb2",
                    fontsize=9, va="center", ha="right", arrowprops=dict(arrowstyle="->", color="#1f6fb2"))
        ax.set_title(title, fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
    fig.suptitle("Cold plate designs at 30% metal, section through the middle of the 2.5 mm cavity "
                 "(hottest die's peak rise at 1 kPa; lower is better)", fontsize=10)
    fig.tight_layout()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, bbox_inches="tight")
    print(a.out)


if __name__ == "__main__":
    main()
