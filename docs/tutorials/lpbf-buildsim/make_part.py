"""
How bracket.stl was made (once; the STL is committed, this is not part of
the tutorial).

A Ti64 sensor bracket printed standing on the plate: a 6 x 10 mm post
20 mm tall, and an arm 3 mm thick reaching 26 mm out along x from its top,
with an M3 hole at the tip for the sensor. A 45-degree gusset carries the
first 8 mm of the arm; the rest overhangs.
"""
import os

import numpy as np
import trimesh

from loadpath.shapes import Box, Cylinder, manifold_from_trimesh, trimesh_from_manifold

W = 10.0                   # mm, y
TOP = 20.0                 # mm, top of the arm
T = 3.0                    # mm, arm thickness
POST = 6.0                 # mm, x

body = Box([0, 0, 0], [POST, W, TOP]).to_manifold() + Box([0, 0, TOP - T], [26, W, TOP]).to_manifold()
# 45-degree gusset under the arm: a triangle in x-z, extruded along y
g = TOP - T
tri = np.array([[POST, g - 8], [POST, g], [POST + 8, g]], float)
pts = np.array([[x, y, z] for y in (0.0, W) for x, z in tri])
body = body + manifold_from_trimesh(trimesh.convex.convex_hull(pts))
body = body - Cylinder([22, W / 2, TOP - T - 1], [22, W / 2, TOP + 1], 1.7).to_manifold()
m = trimesh_from_manifold(body)
m.export(os.path.join(os.path.dirname(os.path.abspath(__file__)), "bracket.stl"))
print(f"{m.volume:.0f} mm3, {len(m.faces)} triangles, watertight {m.is_watertight}")
