"""
The images in this tutorial, from a run's outputs (not part of the
tutorial; needs Node with Playwright's Chromium, see docs/render3d.py):

    python render.py            # after the README's commands, in this folder
"""
import json
import os
import sys

import numpy as np
import trimesh

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(HERE, "..", "..", ".."))

from render3d import KEEP_OUT, LOAD, Scene  # noqa: E402

img = os.path.join(HERE, "images")
WARN = (0.95, 0.55, 0.10)
PART = {"door": (0.55, 0.55, 0.60), "board": (0.15, 0.55, 0.30), "switch": (0.20, 0.20, 0.22),
        "coin-cell": (0.75, 0.75, 0.80), "radio": (0.10, 0.25, 0.55)}
VIEW, UNDER = (28, -55), (-35, -60)


def received():
    """The generator's mesh as it came (unit scale, Y-up) and the skin the
    intake made of it, its holes and dropped debris marked."""
    rep = json.load(open(os.path.join(HERE, "exterior", "exterior.json")))
    raw = trimesh.load(os.path.join(HERE, "generated.glb"), force="mesh", process=False)
    raw.merge_vertices()
    T = np.asarray(rep["transform"])
    raw.apply_transform(T)                    # into the result's frame, to sit beside it
    ext = trimesh.load(os.path.join(HERE, "exterior", "exterior.stl"))
    # the holes: the source's open edges, drawn as small tubes
    e = np.sort(raw.edges, axis=1)
    u, c = np.unique(e, axis=0, return_counts=True)
    rims = [trimesh.creation.cylinder(0.25, segment=raw.vertices[a]) for a in u[c == 1]
            if np.linalg.norm(np.subtract(*raw.vertices[a])) > 1e-6][:4000]
    parts = raw.split(only_watertight=False)
    big = max(parts, key=lambda b: len(b.faces))
    debris = [b for b in parts if len(b.faces) < 0.01 * len(big.faces)]
    s = Scene()
    p = s.panel(f"as generated: {rep['defects']['bodies']} bodies, {rep['defects']['boundaryEdges']} open edges, "
                f"unit scale, Y up", view=(28, -55))
    p.mesh(raw, color=(0.80, 0.78, 0.74), opacity=0.55)
    if rims:
        p.mesh(trimesh.util.concatenate(rims), color=KEEP_OUT)
    for d in debris:
        p.mesh(d, color=WARN)
    cap = rep["capacity"]
    thin = cap.get("thinRegions", [])[:2]
    q = s.panel(f"wrapped: one closed skin, p95 {rep['wrap']['onSource']['p95']:.2f} mm from the source", view=VIEW)
    q.mesh(ext)
    for t in thin:
        lo, hi = np.asarray(t["box"][0]), np.asarray(t["box"][1])
        q.mesh(trimesh.creation.box(hi - lo, trimesh.transformations.translation_matrix((lo + hi) / 2)),
               color=WARN, opacity=0.25)
    s.render(os.path.join(img, "exterior.png"))


def layout():
    """The parts the description put inside, under a see-through skin."""
    lay = json.load(open(os.path.join(HERE, "layout.json")))
    ext = trimesh.load(os.path.join(HERE, "exterior", "exterior.stl"))
    s = Scene()
    for label, view in (("the parts inside, from above", VIEW), ("from below: the door they go in with", UNDER)):
        p = s.panel(label, view=view)
        p.mesh(ext, color=(0.85, 0.85, 0.88), opacity=0.22)
        for k, col in PART.items():
            p.mesh(trimesh.load(os.path.join(HERE, "parts", k + ".stl")), color=col)
        bx, by = lay["button"]["at"]
        top = ext.ray.intersects_location([[bx, by, 50]], [[0, 0, -1]], multiple_hits=False)[0][0]
        p.arrow(top + [0, 0, 0.3], [0, 0, -12])
    s.render(os.path.join(img, "layout.png"))


def part():
    """The delivered part: outside (the exterior as generated), and cut
    through the button to show what loadpath put inside."""
    rep = json.load(open(os.path.join(HERE, "handoff", "switch", "report.json")))
    m = trimesh.load(os.path.join(HERE, "handoff", "switch", "shutter-remote.stl"))
    lay = json.load(open(os.path.join(HERE, "layout.json")))
    bx, by = lay["button"]["at"]
    s = Scene()
    s.panel(f"delivered: {rep['mass']:.1f} g of SLS TPU 88A", view=VIEW).mesh(m)
    cut = m.slice_plane([0, by, 0], [0, 1, 0], cap=True)
    p = s.panel("section through the button", view=(4, -90))
    p.mesh(cut)
    for k in ("switch", "board", "door", "coin-cell"):
        p.mesh(trimesh.load(os.path.join(HERE, "parts", k + ".stl")).slice_plane([0, by, 0], [0, 1, 0], cap=True),
               color=PART[k], opacity=0.6)
    top = m.ray.intersects_location([[bx, by, 50]], [[0, 0, -1]], multiple_hits=False)[0][0]
    p.arrow(top + [0, 0, 0.3], [0, 0, -10])
    s.render(os.path.join(img, "part.png"))


def forces():
    """Force to click at each press, against its window."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rep = json.load(open(os.path.join(HERE, "handoff", "switch", "report.json")))
    pr = rep["presses"]
    # the finest reading there is: the delivered part on 0.5 mm cut cells
    # when the skin review ran that check, else on the design grid's
    chk = rep.get("skinReview", {}).get("grid", {}).get("check", {}).get("part", {})
    fine = chk.get("fineGf")
    tol = 1.02      # the read-back's 2% band
    fig, ax = plt.subplots(figsize=(8.5, 3.4), dpi=150)
    for i, p in enumerate(pr):
        lo = p["min"] / 0.00980665 if p.get("min") else 0
        hi = p["max"] / 0.00980665 if p.get("max") else 3000
        ax.add_patch(plt.Rectangle((i - 0.35, lo), 0.7, hi - lo, color=(0.85, 0.9, 1.0)))
        if fine:
            d = fine[i]
            ok = (p.get("min") is None or d >= lo / tol) and (p.get("max") is None or d <= hi * tol)
        else:
            d = p.get("deliveredForceGf", p["forceGf"])
            ok = p.get("deliveredOk", p["ok"])
        col = (0.15, 0.6, 0.3) if ok else KEEP_OUT
        if d > 2150:        # off the top: an arrow and the number
            ax.annotate(f"{d:.0f}", (i, 2150), ha="center", va="top", fontsize=7, color=col)
            ax.plot([i], [2180], "^", color=col, ms=7)
        else:
            ax.plot([i], [d], "o", color=col, ms=7)
    ax.set_xticks(range(len(pr)))
    ax.set_xticklabels(["button" if (p.get("max") is not None) else "elsewhere" for p in pr], fontsize=8)
    ax.set_ylim(0, 2200)
    ax.set_ylabel("force to click, gf")
    ax.set_title("delivered part, read back on its own cut cells (dots) against each press window (bands)"
                 + (", 0.5 mm cells" if fine else ""),
                 fontsize=9)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(img, "forces.png"))


if __name__ == "__main__":
    os.makedirs(img, exist_ok=True)
    received()
    layout()
    part()
    forces()
