"""
The images in this tutorial, from a full run's output (not part of the
tutorial; needs Node with Playwright's Chromium, see docs/render3d.py):

    python render.py out/            # after `loadpath run bracket.json --out out/`
"""
import json
import os
import sys

import numpy as np
import trimesh
from scipy.spatial import cKDTree

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(HERE, "..", "..", ".."))

from gallery import _study_panel  # noqa: E402
from render3d import Scene, ramp  # noqa: E402

from loadpath.spec import study_from_dict  # noqa: E402
from loadpath.verify import verify_part  # noqa: E402

out = sys.argv[1] if len(sys.argv) > 1 else "out"
with open(os.path.join(out, "report.json")) as fh:
    rep = json.load(fh)
with open(os.path.join(HERE, "bracket.json")) as fh:
    study = study_from_dict(json.load(fh), base_dir=HERE)
best = rep["outcomes"][rep["best"]]
mesh = trimesh.load(best["stl"] if os.path.exists(best["stl"]) else os.path.join(out, "best.stl"), force="mesh")
img = os.path.join(HERE, "images")

s = Scene(f"bracket: {best['mass']:.1f} g PETG, SF {best['safetyFactor']:.2f} (goal 3)", panel_size=(520, 460))
_study_panel(s, study, mesh, (24, -58), "loads (blue), screw keep-outs (red)")
_study_panel(s, study, mesh, (24, 32), "from behind")
s.render(os.path.join(img, "bracket.png"))

# stress on the shape, the shelf case, coloured to its 98th percentile
fields = verify_part(study, mesh, refine=0, fields=True)["fields"]
case = next(c for c in fields["cases"] if c["name"] == "shelf")
vm = np.asarray(case["vonMises"])
dist, idx = cKDTree(fields["nodes"]).query(mesh.vertices, k=8)
w = 1.0 / np.maximum(dist, 1e-6)
w /= w.sum(axis=1, keepdims=True)
colors = ramp((w * vm[idx]).sum(axis=1) / float(np.percentile(vm, 98)))
s = Scene("von Mises stress under the shelf load", panel_size=(520, 460),
          colorbar=("von Mises stress", "0", "98th percentile"))
s.panel(view=(24, -58)).mesh(mesh, vertex_colors=colors)
s.panel(view=(24, 32)).mesh(mesh, vertex_colors=colors)
s.render(os.path.join(img, "stress.png"))

# the outcomes the search tried, lightest passing one picked
rows = sorted(rep["outcomes"], key=lambda o: o["volumeFraction"])
s = Scene("outcomes: mass and safety factor (passing ones meet SF 3)", panel_size=(300, 280))
for o in rows[::2] if len(rows) > 6 else rows:
    m = trimesh.load(o["stl"], force="mesh")
    s.panel(f"{o['mass']:.0f} g, SF {o['safetyFactor']:.1f}" + ("" if o["passes"] else " (fails)"),
            view=(24, -58)).mesh(m)
s.render(os.path.join(img, "outcomes.png"))
