"""
The images in this tutorial, from a full run's output (not part of the
tutorial; needs Node with Playwright's Chromium, see docs/render3d.py):

    python render.py out/            # after `loadpath field link.json --out out/`
"""
import json
import os
import sys

import numpy as np
import trimesh

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(HERE, "..", "..", ".."))

from render3d import Scene, ramp  # noqa: E402

from loadpath.fields import FieldGraph  # noqa: E402

out = sys.argv[1] if len(sys.argv) > 1 else "out"
with open(os.path.join(out, "report.json")) as fh:
    rep = json.load(fh)
mesh = trimesh.load(os.path.join(out, "link.stl"), force="mesh")
img = os.path.join(HERE, "images")
holes = rep["drains"]["holes"]
HOLE = (0.95, 0.45, 0.1)


# the underside is shown turned over (half a turn about x), lit like the top
FLIP = trimesh.transformations.rotation_matrix(np.pi, [1, 0, 0])
flipped = mesh.copy().apply_transform(FLIP)


def arrows(p, side):
    """An arrow into each drain hole on this side (z sign) of the part."""
    for h in holes:
        a, at = np.asarray(h["axis"], float), np.asarray(h["at"], float)
        if np.sign(a[2]) == side:
            if side < 0:
                a, at = FLIP[:3, :3] @ a, FLIP[:3, :3] @ at
            p.arrow(at + 0.5 * a, a * 14, color=HOLE)


s = Scene(f"graded gyroid link: {rep['mass']:.1f} g PA12, {len(holes)} drain holes (orange)", panel_size=(620, 420))
arrows(s.panel("top", view=(38, -62)).mesh(mesh), 1)
arrows(s.panel("underside (turned over)", view=(38, -62)).mesh(flipped), -1)
s.render(os.path.join(img, "link.png"), out_scale=1.2)

# a cut at mid-height, coloured by the solid link's stress (what set the walls)
with open(os.path.join(HERE, "link.json")) as fh:
    doc = json.load(fh)
g = FieldGraph(doc, base_dir=HERE)
try:
    cut = trimesh.intersections.slice_mesh_plane(mesh, [0, 0, -1], [0, 0, 0], cap=True)
except Exception as exc:  # noqa: BLE001 - no triangulator: show it open
    print("no cap:", exc)
    cut = trimesh.intersections.slice_mesh_plane(mesh, [0, 0, -1], [0, 0, 0])
# the cap's long fan triangles would smear the colours: split them to 0.6 mm
v, f = trimesh.remesh.subdivide_to_size(cut.vertices, cut.faces, 0.6)
cut = trimesh.Trimesh(v, f, process=False)
stress = g.evaluate(cut.vertices, "stress")
cols = ramp(np.clip(stress, 0, 1))
s = Scene("section at mid-height: skin, collars and graded gyroid", panel_size=(1100, 420),
          colorbar=("solid link's stress", "0", "its peak"))
s.panel(view=(62, -90)).mesh(cut, vertex_colors=cols)
s.render(os.path.join(img, "section.png"), out_scale=1.2)

# the drain holes close up: one on top near the shoulder, the wrist end from below
s = Scene("drain holes: 6 mm, bossed, on the calm flat faces", panel_size=(520, 420))
for h in (holes[1], holes[0]):
    at, a, m = np.asarray(h["at"]), np.asarray(h["axis"]), mesh
    if a[2] < 0:
        at, a, m = FLIP[:3, :3] @ at, FLIP[:3, :3] @ a, flipped
    s.panel(f"x {h['at'][0]:.0f} mm, {'top' if h['axis'][2] > 0 else 'underside'}, "
            f"stress {h['stressRatio']:.2f} of peak", view=(55, -70), center=at - 2 * a, span=26).mesh(m)
s.render(os.path.join(img, "drains.png"), out_scale=1.2)
