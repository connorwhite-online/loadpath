"""A study's `mesh` block: larger closing and fillet radii only add material, and they add it where members meet,
so joints come out grown rather than assembled.

Run: python -m pytest -q tests/test_mesh_block.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from loadpath.domain import build_domain
from loadpath.reconstruct import reconstruct
from loadpath.spec import study_from_dict


def _t_joint(mesh=None):
    # a bar along x meeting a post along z: one concave joint
    def box(lo, hi):
        return {"type": "box", "min": lo, "max": hi}
    spec = {"name": "t-joint", "designSpace": [box([-30, -8, 0], [30, 8, 40])],
            "supports": [{"shape": box([-31, -9, -1], [31, 9, 0.01])}],
            "preserve": [{"id": "pad", "shape": box([-6, -6, 34], [6, 6, 40])}],
            "loadCases": [{"name": "a", "loads": [{"region": "pad", "force": [0, 0, -100]}]}],
            "material": "pa12", "goal": {"volumeFraction": 0.3}, "resolution": {"pitch": 2.0}, "minMember": 4.0}
    if mesh:
        spec["mesh"] = mesh
    s = study_from_dict(spec)
    s.spec = spec
    d = build_domain(s)
    c = d.origin + (d.mesh.elem_ijk + 0.5) * d.pitch
    x = np.zeros(len(c))
    x[(np.abs(c[:, 1]) < 4) & (c[:, 2] < 8)] = 1                           # the bar on the support
    x[(np.abs(c[:, 0]) < 4) & (np.abs(c[:, 1]) < 4)] = 1                   # the post up to the pad
    return s, d, x


def test_mesh_block_rounds_the_joint():
    s0, d0, x0 = _t_joint()
    s1, d1, x1 = _t_joint({"close": 8.0, "fillet": 8.0})
    m0, _ = reconstruct(s0, d0, x0)
    m1, _ = reconstruct(s1, d1, x1)
    assert m0.is_watertight and m1.is_watertight
    assert m1.volume > m0.volume * 1.02                                     # it adds material ...

    def at(m, lo, hi):
        v = m.voxelized(0.5).fill().points
        sel = np.all((v >= lo) & (v <= hi), axis=1)
        return sel.sum()
    corner = ([4.5, -3, 8.5], [10, 3, 14])                                 # the inside of the joint
    far = ([20, -3, 8.5], [26, 3, 14])                                     # over the bar, far from the post
    assert at(m1, *corner) > at(m0, *corner) + 50                          # ... into the joint ...
    assert at(m1, *far) <= at(m0, *far) + 20                               # ... and not along the plain bar


if __name__ == "__main__":
    test_mesh_block_rounds_the_joint()
    print("ok")
