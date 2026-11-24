"""
Where generated members meet drawn (preserved) shapes (2026-11-21, after
the held-out parts came out with "weird transitions galore" at their bosses,
pads and keep-outs):

- the mesh never approximates a drawn shape: its cells' density is carried
  from the design round it, so the exact union shows the drawn face itself,
  not a patchwork of it and a cell-sized copy (reconstruct._drawn_carried);
- a member meets a drawn shape in a fillet sized to the member, crossing the
  drawn face cleanly (reconstruct._stress_sized_k, DRAWN_INSET);
- a drawn face no member stands on (a boss's end face: a seat) stays clear
  of fillets (reconstruct._clear_free_faces);
- a pin bears on the drawn boss it passes through, not on the clearance
  past it, so no collar grows over the boss's end face (domain).
"""
import numpy as np

from loadpath.domain import boundary_conditions, build_domain
from loadpath.reconstruct import reconstruct
from loadpath.spec import study_from_dict

LUG = {
    "name": "lug",
    "material": "petg",
    "process": {"type": "fdm"},
    "minMember": 2.4,
    "designSpace": [{"type": "box", "min": [-10, -10, 0], "max": [50, 10, 14]}],
    "preserve": [
        {"id": "boss-a", "shape": {"type": "cylinder", "center": [0, 0, 7], "axis": "z", "length": 14, "r": 7}},
        {"id": "boss-b", "shape": {"type": "cylinder", "center": [40, 0, 5], "axis": "z", "length": 10, "r": 6}},
    ],
    "keepOut": [
        {"id": "bore-a", "shape": {"type": "cylinder", "center": [0, 0, 7], "axis": "z", "length": 20, "r": 3.5}},
        # runs 3 mm past boss-b's top: clearance for the bolt
        {"id": "bore-b", "shape": {"type": "cylinder", "center": [40, 0, 5], "axis": "z", "length": 16, "r": 3}},
    ],
    "supports": [{"region": "bore-a"}],
    "loadCases": [{"name": "pull", "loads": [{"region": "bore-b", "force": [200, 0, 0]}]}],
    "goal": {"volumeFraction": 0.3},
    "resolution": {"pitch": 1.2},
}


def _capsule(p, a, b, r):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ab = b - a
    t = np.clip(((p - a) @ ab) / (ab @ ab), 0.0, 1.0)
    return np.linalg.norm(p - (a + t[:, None] * ab), axis=1) - r


def _lug():
    """The lug's domain and a design: one rod, 6 mm across, boss to boss."""
    st = study_from_dict(LUG)
    d = build_domain(st)
    c = d.mesh.elem_centers()
    x = np.clip(0.5 - _capsule(c, (0, 0, 5), (40, 0, 5), 3.0) / d.pitch, 0.0, 1.0)
    return st, d, np.where(d.preserve, 1.0, x)


def test_a_pin_bears_on_its_boss_not_the_clearance_past_it():
    st, d, _ = _lug()
    bc = boundary_conditions(st, d.mesh)
    z = d.mesh.node_xyz[bc.regions[1]][:, 2]
    assert z.max() < 10.0 + 0.5 * d.pitch + 1e-9
    # and nothing is pinned over the boss's end face
    c = d.mesh.elem_centers()
    over = (np.hypot(c[:, 0] - 40, c[:, 1]) < 7.0) & (c[:, 2] > 10.0)
    assert not np.asarray(d.preserve, bool)[over].any()


def test_drawn_faces_come_out_exact_and_clear():
    st, d, x = _lug()
    mesh, info = reconstruct(st, d, x)
    assert info["watertight"] and info["bodies"] == 1
    v = np.asarray(mesh.vertices)
    # boss-b's end face (z = 10) is a seat: nothing stands over it
    r = np.hypot(v[:, 0] - 40, v[:, 1])
    assert not ((r < 5.9) & (v[:, 2] > 10.0 + 1e-6)).any()
    # away from the rod, each boss's side is the drawn cylinder itself: no
    # cell-sized copy of it standing proud
    for cx, rad, top in ((0.0, 7.0, 14.0), (40.0, 6.0, 10.0)):
        side = (np.abs(v[:, 1]) > 4.5) & (np.abs(v[:, 0] - cx) < rad) & (v[:, 2] > 0.5) & (v[:, 2] < top - 0.5)
        rr = np.hypot(v[side, 0] - cx, v[side, 1])
        far = np.abs(v[side, 0] - cx) < 0.5 * rad    # the side facing away from the rod's axis
        assert np.all(rr[far] < rad + 0.05), (cx, rr[far].max())   # a cell is ~1 mm


def test_the_fillet_grows_with_the_member():
    """A fillet into a boss sized to the rod: a thicker rod gets a bigger
    one (tightest radius about a quarter of its thickness)."""
    from loadpath.reconstruct import _stress_sized_k

    step = 0.4
    n = 80
    ax = (np.arange(n) - n / 2) * step
    X, Y, Z = np.meshgrid(ax, ax, ax, indexing="ij")
    for t in (3.0, 8.0):
        dm = np.hypot(X, Y) - 0.5 * t
        k = _stress_sized_k(dm, step, 1.0)
        assert abs(0.2526 * k[n // 2, n // 2, n // 2] - 0.25 * t) < 0.1 * t



PLATE = {"name": "plate", "material": "petg", "process": {"type": "fdm"}, "minMember": 2.4,
         "designSpace": [{"type": "box", "min": [-30, -15, 0], "max": [30, 15, 6]}],
         "keepOut": [
             {"id": "bore", "shape": {"type": "cylinder", "center": [-18, 0, 3], "axis": "z", "length": 12, "r": 2.2}},
             {"id": "slot", "shape": {"type": "box", "min": [5, -6, -2], "max": [15, 6, 8]}}],
         "supports": [{"region": "bore"}],
         "loadCases": [{"name": "pull", "loads": [{"shape": {"type": "box", "min": [26, -15, 0], "max": [30, 15, 6]},
                                                  "force": [0, 0, -100]}]}],
         "goal": {"volumeFraction": 0.5}, "resolution": {"pitch": 1.0}}


def _sharp_rim(mesh, at, r, angle=50.0):
    """mm of mesh edge within r of `at` that turns more than `angle` degrees."""
    ang = mesh.face_adjacency_angles
    e = mesh.vertices[mesh.face_adjacency_edges]
    mid = e.mean(axis=1)
    near = np.linalg.norm(mid - np.asarray(at, float), axis=1) < r
    sharp = near & (np.degrees(ang) > angle)
    return float(np.linalg.norm(e[sharp, 0] - e[sharp, 1], axis=1).sum())


def test_keep_out_rims_are_rounded_to_the_member_and_bearing_rims_only_broken():
    """A keep-out cut exactly after meshing leaves a knife edge round its
    rim (2026-11-24, after the bike mount's phone side). Now it is cut in the
    field first (reconstruct._soft_cut): a free opening's rim is a round of
    about a quarter of the plate (Peterson's shoulder rule), a rim something
    bears on (the supported bore) only an edge break, so the bore keeps its
    bearing length."""
    import loadpath.reconstruct as rc

    st = study_from_dict(PLATE)
    d = build_domain(st)
    x = np.ones(d.mesh.n_elem)
    soft = rc.SOFT_CUT
    try:
        rc.SOFT_CUT = False
        hard, _ = reconstruct(st, d, x)
    finally:
        rc.SOFT_CUT = soft
    mesh, _ = reconstruct(st, d, x)
    def top(m, x):
        loc, _, _ = m.ray.intersects_location([[x, 0.0, 20.0]], [[0.0, 0.0, -1.0]])
        return float(loc[:, 2].max()) if len(loc) else 0.0

    def bore(m, z):
        loc, _, _ = m.ray.intersects_location([[-18.0, 0.0, z]], [[1.0, 0.0, 0.0]])
        return float(loc[:, 0].min() + 18.0)

    # The slot's rim (its face at x = 15, the plate's top at 5.8): exact, the
    # top ran on flat to within 0.6 mm of the face and dropped (5.25 mm at
    # 0.5 mm off it); now a round of about 1.5 mm, a quarter of the plate
    assert top(hard, 15.7) > 5.1, top(hard, 15.7)
    assert top(mesh, 15.7) < 5.0, top(mesh, 15.7)
    assert top(mesh, 16.5) < top(hard, 16.5) - 0.1
    assert abs(top(mesh, 20.0) - top(hard, 20.0)) < 0.05      # and nothing past it
    # the supported bore keeps its bearing: the same wall over its depth
    for z in (1.0, 2.0, 3.0, 4.0, 5.0):
        assert abs(bore(mesh, z) - bore(hard, z)) < 0.1, (z, bore(mesh, z), bore(hard, z))
    assert 0.0 < hard.volume - mesh.volume < 0.02 * hard.volume
