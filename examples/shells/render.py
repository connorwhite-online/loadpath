"""
Render a split (after verify.py): the base and the cover side by side, open
faces up, the components ghosted, the joints coloured.

    python examples/shells/render.py /tmp/handheld/split out.png
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(HERE, "..", "..", "docs"))

HOOK = (0.95, 0.55, 0.15)
LEDGE = (0.20, 0.55, 0.90)
LIP = (0.30, 0.70, 0.40)
COMP = (0.55, 0.35, 0.75)


def main(argv=None):
    import trimesh
    from render3d import Scene

    from loadpath.shapes import shape_from_spec, trimesh_from_manifold
    argv = argv or sys.argv[1:]
    d, out = argv[0], argv[1]
    rep = json.load(open(os.path.join(d, "split.json")))
    nrm = rep["cut"]["normal"]
    if isinstance(nrm, str):
        a, sg = "xyz".index(nrm[1]), (1 if nrm[0] == "+" else -1)
    else:
        a = int(np.argmax(np.abs(nrm)))
        sg = 1 if nrm[a] > 0 else -1
    hook_on = (rep.get("snaps") or {}).get("hookOn")
    halves = {s: trimesh.load(os.path.join(d, f"run-{s}", "best.stl"), force="mesh") for s in ("base", "cover")}
    # the cover turned over (about the axis across), its open face up
    flip = np.eye(4)
    b = [i for i in range(3) if i != a][0]
    flip[a, a] = -1.0
    flip[b, b] = -1.0

    def tm(spec):
        return trimesh_from_manifold(shape_from_spec(spec).to_manifold())

    title = (f"split at {nrm[1]} = {rep['cut']['at']:g} mm" if isinstance(nrm, str) and rep["cut"].get("at") is not None
             else f"split along {rep['cut'].get('label', 'the parting line')}")
    s = Scene(f"{title}: base and cover, open faces up")
    up_view = (52, -60) if sg > 0 else (-52, -60)
    for side in ("base", "cover"):
        p = s.panel(side, view=up_view)
        m = halves[side].copy()
        extra = []
        for g in rep.get("snapsGeometry") or []:
            if side == hook_on:
                extra.append((tm(g["hook"]), HOOK))
            else:
                extra.append((tm(g["ledge"]), LEDGE))
        if side == "base" and os.path.isfile(os.path.join(d, "cad", "lip.stl")):
            extra.append((trimesh.load(os.path.join(d, "cad", "lip.stl")), LIP))
        comps = []
        st = json.load(open(os.path.join(d, [f for f in os.listdir(d) if f.endswith(f"-{side}.study.json")][0])))
        for k in st.get("keepOut") or []:
            if k["shape"].get("type") == "mesh" and k["id"] in rep["assembly"]["componentsIn"] \
                    and rep["assembly"]["componentsIn"][k["id"]] == side:
                comps.append(trimesh.load(os.path.join(d, k["shape"]["path"]), force="mesh"))
        if side == "cover":
            m.apply_transform(flip)
            for x, _c in extra:
                x.apply_transform(flip)
            for c in comps:
                c.apply_transform(flip)
        p.mesh(m)
        for x, c in extra:
            p.mesh(x, color=c)
        for c in comps:
            p.mesh(c, color=COMP, opacity=0.35)
    s.render(out)
    print(out)


if __name__ == "__main__":
    main()
