"""
Render tendon_grip.py's result: each design deformed at full wind (blue)
over its undeformed cells (grey), the object, and grip force against tendon
tension.

    python examples/mechanism/tendon_grip_render.py [--out tendon-arm-out/] [--png tendon-grip.png]
"""
import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.collections import PolyCollection  # noqa: E402
from matplotlib.patches import Circle  # noqa: E402

QUAD = [0, 1, 5, 4]          # the hex8 corners on the y = 0 face, in order around it (x, z)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="tendon-arm-out")
    ap.add_argument("--png", default=None)
    a = ap.parse_args()
    s = json.load(open(os.path.join(a.out, "summary.json")))
    d = np.load(os.path.join(a.out, "designs.npz"))
    r, gap = s["objectRadius"], s["gap"]
    fig, axes = plt.subplots(1, 3, figsize=(18, 4.2), gridspec_kw={"width_ratios": [1, 1, 0.8]})
    for ax, name in zip(axes, ["optimized", "palm-side bar"]):
        key = name.replace(" ", "_")
        g = d[key]
        ii, kk = np.nonzero(g > 0.5)
        ax.add_collection(PolyCollection([[(i, k), (i + 1, k), (i + 1, k + 1), (i, k + 1)] for i, k in zip(ii, kk)],
                                         facecolor="#d9d9d9", edgecolor="none"))
        ax.add_collection(PolyCollection(d["deformed_" + key][:, QUAD], facecolor="#1f77b4", edgecolor="#1f77b4",
                                         linewidth=0.5))
        ax.add_patch(Circle((30.0, -r - gap), r, color="#fdd0a2"))
        ax.plot([-1, -1], [0, g.shape[1]], color="k", linewidth=4)
        row = s["rows"][name]
        ax.set_title(f"{name}: grip {row['grip']:.2f} N at {row['tension']:.2f} N tendon pull")
        ax.set_xlim(-3, 57)
        ax.set_ylim(-14, 15)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
    ax = axes[2]
    for name, color in (("palm-side bar", "grey"), ("optimized", "#1f77b4")):
        c = np.array(s["rows"][name]["curve"])
        ax.plot(c[:, 0], c[:, 1], ".-", color=color, label=name)
    ax.set_xlabel("tendon tension (N)")
    ax.set_ylabel("grip force (N)")
    ax.set_title(f"per mm of finger width, servo winds in {s['stroke']:g} mm")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(a.png or os.path.join(a.out, "tendon-grip.png"), dpi=100)


if __name__ == "__main__":
    main()
