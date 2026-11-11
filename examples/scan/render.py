"""
Renders for docs/SCAN.md: the scan as points, and the part designed round
the scanned object (bar and light translucent, the part in clay).

    python examples/scan/render.py SCAN.ply PART.stl OUT.png [--units mm --device artec-eva]
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "docs"))


def point_glyphs(P, r):
    """Each point as a small octahedron (a mesh three.js can light)."""
    import trimesh
    o = np.array([[r, 0, 0], [-r, 0, 0], [0, r, 0], [0, -r, 0], [0, 0, r], [0, 0, -r]], float)
    f = np.array([[0, 2, 4], [2, 1, 4], [1, 3, 4], [3, 0, 4], [2, 0, 5], [1, 2, 5], [3, 1, 5], [0, 3, 5]])
    V = (P[:, None, :] + o[None]).reshape(-1, 3)
    F = (f[None] + 6 * np.arange(len(P))[:, None, None]).reshape(-1, 3)
    return trimesh.Trimesh(V, F, process=False)


def main():
    import trimesh
    from render3d import Scene

    from loadpath.scan import load_scan, reconstruct
    ap = argparse.ArgumentParser()
    ap.add_argument("scan")
    ap.add_argument("part")
    ap.add_argument("out")
    ap.add_argument("--units", default="mm")
    ap.add_argument("--device", default="artec-eva")
    ap.add_argument("--title", default=None)
    a = ap.parse_args()
    s = load_scan(a.scan, device=a.device, units=a.units)
    rng = np.random.default_rng(0)
    sel = rng.choice(len(s.points), min(len(s.points), 25000), replace=False)
    pts = point_glyphs(s.points[sel], 0.45)
    obj, info = reconstruct(s)
    part = trimesh.load(a.part, force="mesh")
    lo, hi = part.bounds
    c = 0.5 * (lo + hi)
    span = float(np.max(hi - lo)) * 1.45
    sc = Scene(a.title, panel_size=(640, 520))
    p1 = sc.panel("the scan: points", view=(22, -60), center=c, span=span)
    p1.mesh(pts, color=(0.25, 0.45, 0.85), crease=10)
    p2 = sc.panel("the part, round the scanned bar and light", view=(22, -60), center=c, span=span)
    p2.mesh(part)
    p2.mesh(obj, color=(0.35, 0.55, 0.9), opacity=0.28)
    p3 = sc.panel("from below", view=(-28, -120), center=c, span=span)
    p3.mesh(part)
    p3.mesh(obj, color=(0.35, 0.55, 0.9), opacity=0.28)
    sc.render(a.out)
    print(a.out)


if __name__ == "__main__":
    main()
