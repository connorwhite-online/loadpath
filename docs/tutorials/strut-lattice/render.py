"""
The images in this tutorial, from a full run's outputs (run the README's
commands without --quick first, and the conformal variant for the last
image). Not run by the tests; needs Node with Playwright's Chromium
(docs/render3d.py).

    python render.py [run folder]       # default: this folder
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

IMAGES = os.path.join(HERE, "images")
RUN = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else HERE
VIEW = (35, -60)

doc = json.load(open(os.path.join(RUN, "tune", "tuned.json")))
graph = FieldGraph(doc, base_dir=HERE)
part = trimesh.load(os.path.join(RUN, "out", "forearm.stl"), force="mesh")
rep = json.load(open(os.path.join(RUN, "out", "report.json")))



def by_stress(mesh):
    """Vertex colours: the solid link's stress there, the field that grades the rim and struts."""
    d = np.asarray(graph.evaluate(np.asarray(mesh.vertices, float), "stress"), float)
    return ramp(d)


def cut(mesh, origin, normal):
    return mesh.slice_plane(origin, normal, cap=True)


# the part, coloured by the solid link's stress, whole and cut through its mid-plane
s = Scene(f"forearm: octet lattice, {rep['mass']:.1f} g PA12", panel_size=(560, 400),
          colorbar=("solid link's stress", "0", "peak"))
s.panel("from above", view=VIEW).mesh(part, vertex_colors=by_stress(part))
half = cut(part, [0, 0, 0], [0, 0, -1])
s.panel("cut at mid-thickness", view=(55, -70)).mesh(half, vertex_colors=by_stress(half))
s.render(os.path.join(IMAGES, "forearm.png"), out_scale=1.0)

# the cartesian core against the conformal one, cut across the beam
conf = os.path.join(RUN, "conformal", "forearm-conformal.stl")
if os.path.exists(conf):
    cm = trimesh.load(conf, force="mesh")
    crep = json.load(open(os.path.join(RUN, "conformal", "report.json")))
    s = Scene("cut across the beam at x = 60 mm", panel_size=(420, 380))
    for label, m, r in (("cartesian octet (delivered)", part, rep), ("conformal octet", cm, crep)):
        c = cut(m, [60, 0, 0], [-1, 0, 0])
        s.panel(f"{label}: {r['bodies']} bod{'y' if r['bodies'] == 1 else 'ies'}", view=(12, -8),
                center=(60, 0, 0), span=34).mesh(c)
    s.render(os.path.join(IMAGES, "conformal.png"), out_scale=1.0)
