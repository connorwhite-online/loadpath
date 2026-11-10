"""
Renders for docs/SHROUD.md: a shrouded part whole, cut open along its
length (the interior, the skin and the clear channel), and the envelope's
shape steps (seed hull against the load-shaped envelope).

    python examples/shroud/render.py RUN_DIR STUDY.json OUT.png [--title T]

RUN_DIR is a `loadpath run` output folder (its best STL). Needs Node and
Playwright's Chromium (docs/render3d.py).
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import trimesh

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "docs"))
sys.path.insert(0, ROOT)

from render3d import KEEP_OUT, Scene  # noqa: E402


def best_stl(run_dir):
    rep = os.path.join(run_dir, "report.json")
    if os.path.exists(rep):
        r = json.load(open(rep))
        best = (r.get("best") or {}).get("mesh") or r.get("mesh")
        if best and os.path.exists(os.path.join(run_dir, os.path.basename(best))):
            return os.path.join(run_dir, os.path.basename(best))
    stls = sorted(glob.glob(os.path.join(run_dir, "*.stl")), key=os.path.getsize, reverse=True)
    return stls[0]


def keepout_meshes(study_path, ids):
    from loadpath.spec import study_from_dict
    d = json.load(open(study_path))
    d.pop("shroud", None)
    st = study_from_dict(d, os.path.dirname(os.path.abspath(study_path)))
    out = []
    for r in st.keep_out:
        if r.id in ids:
            m = r.shape.to_manifold()
            from loadpath.shapes import trimesh_from_manifold
            out.append(trimesh_from_manifold(m))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("study")
    ap.add_argument("out")
    ap.add_argument("--title", default=None)
    ap.add_argument("--ghost", nargs="*", default=["cable-channel", "shoulder-port", "wrist-port"])
    ap.add_argument("--cut", default="z", help="axis of the section plane through the centre")
    a = ap.parse_args()
    part = trimesh.load(best_stl(a.run_dir), force="mesh")
    ghosts = keepout_meshes(a.study, set(a.ghost))
    s = Scene(a.title, panel_size=(760, 520))
    p = s.panel("whole", view=(28, -62))
    p.mesh(part)
    ax = "xyz".index(a.cut)
    n = np.zeros(3)
    n[ax] = -1.0
    c = part.bounds.mean(axis=0)
    half = part.slice_plane(c, n, cap=True)
    q = s.panel("cut open: skin, interior and the clear cable channel", view=(28, -62))
    q.mesh(half)
    for g in ghosts:
        gh = g.slice_plane(c, n, cap=True)
        q.mesh(gh if len(gh.faces) else g, color=KEEP_OUT, opacity=0.35)
    s.render(a.out)
    print(a.out)


if __name__ == "__main__":
    main()
