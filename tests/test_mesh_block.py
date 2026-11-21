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


def _corner_profile(union, k, step=0.002):
    """The blended right-angle corner of two half-planes (x < 0, y < 0) as
    a curve: (points, curvature)."""
    from skimage import measure
    x = np.arange(-0.5, 3.0 * k, step)
    X, Y = np.meshgrid(x, x, indexing="ij")
    c = max(measure.find_contours(union(X, Y, k), 0.0), key=len) * step + x[0]
    d1 = np.gradient(c, axis=0)
    d2 = np.gradient(d1, axis=0)
    kap = np.abs(d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0]) / np.linalg.norm(d1, axis=1) ** 3
    return c, kap


def test_junction_blend_is_g2_and_no_tighter():
    """The blend where members meet preserved geometry leaves each surface
    with that surface's own curvature (zero on a flat), where the old one
    jumped to a third of its peak in one step; scaled by G2_SCALE its
    tightest radius is the old one's, so it raises stress no more."""
    from loadpath.reconstruct import G2_SCALE, _g2_union, _smooth_union
    _, kq = _corner_profile(_smooth_union, 1.0)
    c, kg = _corner_profile(_g2_union, G2_SCALE)
    assert abs(kg.max() - kq.max()) / kq.max() < 0.03, (kg.max(), kq.max())
    on = kq > 1e-3
    assert kq[on][1] > 0.1 * kq.max()                      # the old blend: a step in curvature at its edge
    edge = c[:, 0] > 0.97 * G2_SCALE                       # the new one, along the last 3% of its reach
    assert kg[edge].max() < 0.02 * kg.max()                 # rises from zero


if __name__ == "__main__":
    test_mesh_block_rounds_the_joint()
    print("ok")


def _crotch_field(step=0.25):
    # two square bars (4 mm thick) meeting in an L: one concave corner at the origin
    g = np.arange(-12.0, 12.0, step) + 0.5 * step
    X, Y, Z = np.meshgrid(g, g, g, indexing="ij")
    bar_x = (X > -2) & (Y > -2) & (Y < 2) & (np.abs(Z) < 2)
    bar_y = (X > -2) & (X < 2) & (Y > -2) & (np.abs(Z) < 2)
    from scipy import ndimage
    f = ndimage.gaussian_filter((bar_x | bar_y).astype(float), 0.7)
    return f, np.array([g[0]] * 3), step


def test_member_thickness_is_the_bar_it_sits_on():
    from loadpath.reconstruct import member_thickness
    f, origin, step = _crotch_field()
    # a point on the bar's flank far from the corner: the bar's 4 mm
    assert abs(member_thickness(f, origin, step, np.array([8.0, 2.0, 0.0])) - 4.0) < 0.6


def test_stress_round_fills_only_the_crotch():
    from loadpath.reconstruct import _round_near, member_thickness
    f, origin, step = _crotch_field()
    at = np.array([2.0, 2.0, 0.0])             # the inside corner
    r = member_thickness(f, origin, step, at)
    assert 3.5 < r < 7.0
    g = _round_near(f, origin, step, at, r)
    assert np.all(g >= f - 1e-9)               # only ever adds
    added = (g >= 0.5) & (f < 0.5)
    idx = np.argwhere(added) * step + origin
    assert len(idx) > 0
    # all of it inside the corner's quadrant, none on the outer faces
    assert np.all(idx[:, 0] > 1.5) and np.all(idx[:, 1] > 1.5)
    # and the fill reaches about r along each leg (a round, not a chamfer of the corner cell)
    assert idx[:, 0].max() > 2.0 + 0.5 * r
