"""
The images in this tutorial, from a full run's outputs (not part of the
tutorial; needs Node with Playwright's Chromium, see docs/render3d.py, and
OpenCascade for the STEP):

    python render.py            # after the README's commands, in this folder
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

from render3d import KEEP_OUT, LOAD, Scene, ramp  # noqa: E402

from loadpath.cadimport import load_part  # noqa: E402
from loadpath.spec import study_from_dict  # noqa: E402
from loadpath.verify import verify_part  # noqa: E402

PRESERVE = (0.25, 0.50, 0.90)
VIEW, BACK = (24, -58), (24, 32)
img = os.path.join(HERE, "images")

with open(os.path.join(HERE, "out", "report.json")) as fh:
    rep = json.load(fh)
best = rep["outcomes"][rep["best"]]
opt = trimesh.load(os.path.join(HERE, "out", os.path.basename(best["stl"])), force="mesh")
drawn = load_part(os.path.join(HERE, "elbow.step")).mesh
with open(os.path.join(HERE, "study.json")) as fh:
    raw = json.load(fh)
study = study_from_dict(raw, base_dir=HERE)
drawn_g = drawn.volume * 1.01e-3                       # PA12, g/cm3 x mm3
center, span = 0.5 * np.add(*drawn.bounds), 62.0

# 1. drawn and optimized
s = Scene(f"elbow bracket: as drawn ({drawn_g:.1f} g) and optimized ({best['mass']:.1f} g, "
          f"SF {best['safetyFactor']:.1f}), PA12", panel_size=(480, 440))
s.panel("as drawn (elbow.step)", view=VIEW, center=center, span=span).mesh(drawn)
s.panel("optimized", view=VIEW, center=center, span=span).mesh(opt)
s.panel("optimized, along the bearing", view=BACK, center=center, span=span).mesh(opt)
s.render(os.path.join(img, "elbow.png"))


# 2. the draft: design space, preserved seats, keep-outs (clipped to the picture), loads
def mesh_of(path):
    return trimesh.load(os.path.join(HERE, path), force="mesh")


s = Scene("the draft: hull design space, preserved seats (blue), keep-outs (red), the three loads",
          panel_size=(560, 480))
for view in (VIEW, BACK):
    p = s.panel(view=view, center=center, span=span)
    p.mesh(mesh_of(raw["designSpace"][0]["path"]), opacity=0.25)
    for pr in raw["preserve"]:
        p.mesh(mesh_of(pr["shape"]["path"]), color=PRESERVE)
    for ko in study.keep_out:
        if ko.id.endswith("-clear") or ko.id.endswith("-path"):
            continue                                   # long slabs and lines: they would fill the picture
        m = ko.shape.to_manifold().to_mesh()
        km = trimesh.Trimesh(np.asarray(m.vert_properties)[:, :3], np.asarray(m.tri_verts), process=False)
        km = km.slice_plane([0, 0, 52.0], [0, 0, -1.0], cap=True)      # the drivers run on up
        p.mesh(km, color=KEEP_OUT, opacity=0.3, crease=30)
    seat = np.array([28.0, 18.0, 36.0])
    for case in raw["loadCases"]:
        f = np.asarray(case["loads"][0]["force"], float)
        p.arrow(seat + f / np.linalg.norm(f) * 8.0, f / np.linalg.norm(f) * 18.0, color=LOAD)
s.render(os.path.join(img, "draft.png"))

# 3. stress under hang on the optimized part
fields = verify_part(study, opt, refine=0, fields=True)["fields"]
case = next(c for c in fields["cases"] if c["name"] == "hang")
vm = np.asarray(case["vonMises"])
dist, idx = cKDTree(fields["nodes"]).query(opt.vertices, k=8)
w = 1.0 / np.maximum(dist, 1e-6)
w /= w.sum(axis=1, keepdims=True)
colors = ramp((w * vm[idx]).sum(axis=1) / float(np.percentile(vm, 98)))
s = Scene("von Mises stress under hang (400 N down on the bearing)", panel_size=(520, 460),
          colorbar=("von Mises stress", "0", "98th percentile"))
s.panel(view=VIEW, center=center, span=span).mesh(opt, vertex_colors=colors)
s.panel(view=BACK, center=center, span=span).mesh(opt, vertex_colors=colors)
s.render(os.path.join(img, "stress.png"))
