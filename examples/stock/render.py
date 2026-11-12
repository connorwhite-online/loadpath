"""
Renders of a study with bought stock members: the printed part with each
member in place, and a close-up of every joint, whole and cut along the
member so the seat, the sleeve and the blend into the node can be judged.

    python examples/stock/render.py examples/stock/tube-arm.json out/tube-arm.stl docs/images/stock/tube-arm

writes <prefix>.png (the assembly) and <prefix>-joint-<member>-<k>.png.
"""
import json
import os
import sys

import numpy as np
import trimesh

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "docs"))

from render3d import CLAY, Scene  # noqa: E402

from loadpath.shapes import manifold_from_trimesh, trimesh_from_manifold  # noqa: E402
from loadpath.sharp import exact_manifold  # noqa: E402
from loadpath.spec import study_from_dict  # noqa: E402

CARBON = (0.13, 0.13, 0.15)


def cut(mesh_m, normal, at):
    """The half of a manifold on the far side of a plane through `at`."""
    n = np.asarray(normal, float)
    return mesh_m.trim_by_plane(tuple(-n), float(-n @ np.asarray(at, float)))


def main(study_path, stl, prefix):
    study = study_from_dict(json.load(open(study_path)), base_dir=os.path.dirname(os.path.abspath(study_path)))
    part = trimesh.load(stl)
    part_m = manifold_from_trimesh(part)
    members = [(m, exact_manifold(m.body())) for m in study.stock]

    s = Scene(panel_size=(760, 470))
    for label, view in (("", (22, -62)), ("", (28, 118))):
        p = s.panel(label, view=view)
        p.mesh(part)
        for _, mm in members:
            p.mesh(trimesh_from_manifold(mm), color=CARBON, roughness=0.35, metalness=0.2)
    s.render(prefix + ".png")
    for m, mm in members:
        for j in m.joints:
            c = m.point(0.5 * (j.span[0] + j.span[1]))
            span = max(j.span[1] - j.span[0], 4.0 * m.ro) * 1.45
            # a view across the member, and the same cut through its axis
            side = m.e2 if abs(m.e2[2]) < 0.5 else m.e3
            az = float(np.degrees(np.arctan2(side[1], side[0])))
            s = Scene(panel_size=(620, 520))
            p = s.panel("", view=(24, az - 28), center=c, span=span)
            p.mesh(part)
            p.mesh(trimesh_from_manifold(mm), color=CARBON, roughness=0.35, metalness=0.2)
            p = s.panel("", view=(14, az - 12), center=c, span=span)
            p.mesh(trimesh_from_manifold(cut(part_m, side, c)))
            p.mesh(trimesh_from_manifold(cut(mm, side, c)), color=CARBON, roughness=0.35, metalness=0.2)
            p = s.panel("", view=(30, az + 150), center=c, span=span)
            p.mesh(part)
            p.mesh(trimesh_from_manifold(mm), color=CARBON, roughness=0.35, metalness=0.2)
            s.render(f"{prefix}-joint-{m.id}-{j.index}.png")


if __name__ == "__main__":
    main(*sys.argv[1:4])
