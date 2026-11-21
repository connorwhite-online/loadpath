"""reconstruct() and the material it drops: a subtract-shaped preserve
(a duct rim round an opening) must not sever what the optimizer ran up
through its opening, a fragment meshing cuts loose from the field's own
body is reported loudly, and each outcome's field is saved to re-mesh.

Run: python -m pytest -q tests/test_reconstruct_fragments.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from loadpath.domain import build_domain
from loadpath.reconstruct import reconstruct
from loadpath.spec import study_from_dict


def _fan_plate(rim_shape, keep_out=None):
    # A fan plate's deck with a duct rim round an opening (preserved, and
    # loaded), and ribs the optimizer runs up from the deck through the
    # opening. The design space is two boxes: the plate, and the space
    # over the opening standing on it.
    def box(lo, hi):
        return {"type": "box", "min": lo, "max": hi}
    spec = {
        "name": "fan-plate",
        "designSpace": [box([-18, -18, 10], [18, 18, 30]), box([-24, -24, 0], [24, 24, 10])],
        "preserve": [{"id": "rim", "shape": rim_shape}],
        "supports": [{"shape": box([-25, -25, -1], [25, 25, 0.01])}],
        "loadCases": [{"name": "a", "loads": [{"region": "rim", "force": [0, 0, -100]}]}],
        "material": "pa12", "goal": {"volumeFraction": 0.3}, "resolution": {"pitch": 2.0}, "minMember": 2.5}
    if keep_out:
        spec["keepOut"] = [{"shape": keep_out}]
    s = study_from_dict(spec)
    s.spec = spec
    d = build_domain(s)
    c = d.origin + (d.mesh.elem_ijk + 0.5) * d.pitch
    x = np.zeros(len(c))
    x[c[:, 2] < 5] = 1                                                       # deck
    x[(np.abs(c[:, 0]) < 2.5) & (np.abs(c[:, 1]) < 14) & (c[:, 2] < 26)] = 1  # spine
    x[(np.abs(c[:, 1]) < 2.5) & (np.abs(c[:, 0]) < 14) & (c[:, 2] < 26)] = 1  # cross rib
    x[(np.maximum(np.abs(c[:, 0]), np.abs(c[:, 1])) > 18) & (c[:, 2] < 10)] = 1   # rim and its wall
    return s, d, x


def test_reconstruct_keeps_ribs_through_a_subtract_rims_opening():
    # The fan plate (MJF PA12): its duct rim, a box minus its opening
    # (cutter 1 mm taller than the rim), was a flush face, and its sdf
    # max(a, -b) read every point of the opening at the rim's top as on
    # the rim; seating on the flush faces then shifted the field there by
    # the height of an interior design-space face, and cut the ribs the
    # optimizer ran up through the opening (12 bodies, 113-137 cm^3
    # dropped). The same rim as four bars meshed whole: so must this.
    def box(lo, hi):
        return {"type": "box", "min": lo, "max": hi}
    rim = {"type": "subtract", "shapes": [box([-24, -24, 5], [24, 24, 10]), box([-18, -18, 0], [18, 18, 11])]}
    bars = {"type": "union", "shapes": [box([-24, -24, 5], [24, -18, 10]), box([-24, 18, 5], [24, 24, 10]),
                                        box([-24, -18, 5], [-18, 18, 10]), box([18, -18, 5], [24, 18, 10])]}
    s, d, x = _fan_plate(rim)
    mesh, info = reconstruct(s, d, x)
    ref, _ = reconstruct(*_fan_plate(bars))
    assert info["bodies"] == 1 and info["droppedFragmentVolume"] == 0.0, info
    zs = np.arange(0.5, 25, 0.5)
    for xy in [(0, 8), (8, 0), (0, -8), (0, 0)]:
        inside = mesh.contains(np.column_stack([np.full_like(zs, xy[0]), np.full_like(zs, xy[1]), zs]))
        assert inside.all(), (xy, "".join("#" if m else "." for m in inside))
    assert abs(mesh.volume - ref.volume) < 0.01 * ref.volume, (mesh.volume, ref.volume)


def test_reconstruct_reports_fragments_the_field_had_joined(monkeypatch):
    # Material meshing cuts loose from a body the optimized field had
    # joined to the part is a reconstruction artifact: it is dropped (the
    # part is one body) but reported, with where it was, and the outcome
    # warns. Here a stand-in meshing bug saws the ribs off above the rim.
    import manifold3d as m3d
    import loadpath.reconstruct as R
    from loadpath.study import fragment_warnings
    def box(lo, hi):
        return {"type": "box", "min": lo, "max": hi}
    rim = {"type": "union", "shapes": [box([-24, -24, 5], [24, -18, 10]), box([-24, 18, 5], [24, 24, 10]),
                                       box([-24, -18, 5], [-18, 18, 10]), box([18, -18, 5], [24, 18, 10])]}
    real = R._isosurface

    def sawn(*a, **k):
        m = real(*a, **k)
        cut = R.manifold_from_trimesh(m) - m3d.Manifold.cube([40.0, 40.0, 2.0]).translate([-20.0, -20.0, 14.0])
        return R.trimesh_from_manifold(cut)
    monkeypatch.setattr(R, "_isosurface", sawn)
    s, d, x = _fan_plate(rim)
    mesh, info = reconstruct(s, d, x)
    assert info["bodies"] == 1 and info["droppedFragmentVolume"] > 500, info
    top = info["droppedFragments"][0]
    assert top["connectedInField"] and top["bboxMin"][2] > 15.9 and top["bboxMax"][2] > 24, top
    assert info["severedFragmentVolume"] > 500, info
    w = fragment_warnings(info, mesh)
    assert w and "optimized field had joined" in w[0] and "z 16" in w[0], w
    assert fragment_warnings({"droppedFragmentVolume": 0.0}, mesh) == []


def test_saved_field_round_trips_and_remeshes():
    import json
    import os
    import subprocess
    import tempfile
    from loadpath.study import load_field, save_field
    s, d, x = _fan_plate({"type": "box", "min": [-24, -24, 5], "max": [24, -18, 10]})
    with tempfile.TemporaryDirectory() as t:
        path = save_field(os.path.join(t, "o.field.npz"), x, d.pitch, d.n_elem)
        assert path and save_field(os.path.join(t, "p.field.npz"), x[:-1], d.pitch, d.n_elem) is None
        x2, pitch = load_field(path)
        assert pitch == d.pitch and np.allclose(x2, x)
        with open(os.path.join(t, "study.json"), "w") as fh:
            json.dump(s.spec, fh)
        r = subprocess.run([sys.executable, "-m", "loadpath", "remesh", os.path.join(t, "study.json"), path,
                            "--out", os.path.join(t, "re"), "--no-verify"], capture_output=True, text=True,
                           cwd=str(Path(__file__).resolve().parents[1]))
        assert r.returncode == 0, r.stderr[-2000:]
        rep = json.loads(open(os.path.join(t, "re", "report.json")).read())
        assert rep["ok"] and os.path.exists(rep["stl"]) and rep["reconstruction"]["bodies"] == 1, rep


def test_feather_edges_at_a_bore_are_cut_to_what_prints():
    """A bore leaving a block's top at a slant (11 degrees) leaves the
    material beside it tapering to a knife edge: MJF forms what a 0.8 mm
    ball sweeps, and the edge reached 2.7 mm past that (processcheck reads
    it as walls that will not form). reconstruct cuts it along the ball's
    sweep, so the part ends in a round, as it will print, losing a few
    mm^3."""
    from loadpath.processcheck import check_process

    spec = {"name": "groove", "designSpace": {"type": "box", "min": [-20, -15, 0], "max": [20, 15, 12]},
            "keepOut": [{"id": "bore", "shape": {"type": "cylinder", "p0": [-25, 0, 10], "p1": [25, 0, 14],
                                                 "r": 3}}],
            "supports": [{"shape": {"type": "box", "min": [-21, -16, -1], "max": [21, 16, 0.01]}}],
            "loadCases": [{"name": "a", "loads": [{"shape": {"type": "box", "min": [-21, -16, 11],
                                                             "max": [21, 16, 13]}, "force": [0, 0, -100]}]}],
            "material": "pa12", "process": {"type": "mjf", "buildDirection": "+z"},
            "goal": {"volumeFraction": 0.9}, "resolution": {"pitch": 1.0}, "minMember": 2.0}
    import loadpath.reconstruct as rc
    s = study_from_dict(spec)
    d = build_domain(s)
    # the cut itself, with the room for a member (below) not yet taken out
    room = rc._member_room
    rc._member_room = lambda *a, **k: None
    try:
        mesh, info = reconstruct(s, d, np.ones(d.mesh.n_elem))
    finally:
        rc._member_room = room
    assert info.get("unformedDropped"), info
    w = check_process(mesh, "mjf", build="+z", material="pa12")["walls"]
    assert w["ok"] is True and not w["unformed"], w
    assert mesh.volume > 13500, mesh.volume   # 13543 mm^3 before: a round, not a groove widened
    from loadpath.verify import _creases
    assert _creases(s, mesh)["creaseLength"] < 10.0
    # As delivered: beside the bore there is no room for a 2 mm member where
    # the edge was, so the material ends in that member's round before any
    # feather forms, and nothing is left to cut (13459 mm^3, crease 0.2 mm).
    mesh, info = reconstruct(s, d, np.ones(d.mesh.n_elem))
    assert not info.get("unformedDropped"), info
    w = check_process(mesh, "mjf", build="+z", material="pa12")["walls"]
    assert w["ok"] is True and not w["unformed"], w
    assert mesh.volume > 13400, mesh.volume
    # and the cut runs out into the surface: ending at a wall it left 41.6 mm
    # of crease along the bore (38 of the wrist yoke's 50 mm, for 40 allowed)
    from loadpath.verify import _creases
    assert _creases(s, mesh)["creaseLength"] < 10.0


def test_pinch_balls_never_stand_in_a_bore():
    """Two blocks kissing along an edge that runs through a bore: the
    fallback balls (free of the envelope) still close the contact outside
    the bore and put nothing inside it, where a pin or a driver goes."""
    import manifold3d as m3d

    from loadpath.reconstruct import _fill_pinches, _outside

    a = m3d.Manifold.cube((10, 10, 10))
    b = m3d.Manifold.cube((10, 10, 10)).translate((10, 10, 0))   # shares the edge x=y=10
    bore = m3d.Manifold.cylinder(4, 2, 2, 32).translate((10, 10, 3))   # round the edge, z 3..7
    part = (a + b) - bore
    out, n = _fill_pinches(part, rounds=4, clip=_outside(part, bore))
    assert n > 0
    assert (out ^ bore).volume() < 1e-9, (out ^ bore).volume()
    unclipped, _ = _fill_pinches(part, rounds=4)
    assert (unclipped ^ bore).volume() > 1e-4   # what the bound keeps out


def test_fill_pinches_keeps_only_what_helps():
    """Two blocks that touch along an edge read as one non-manifold edge
    once an STL reader welds them: a neck joins them. Where no neck may go
    (the clip leaves no room at the contact), the contact is opened by a
    cut instead, and a part with no such contact comes back as it was."""
    import manifold3d as m3d

    from loadpath.reconstruct import _fill_pinches, _stl_pinches

    a = m3d.Manifold.cube((4.0, 4.0, 0.1))
    b = m3d.Manifold.cube((4.0, 4.0, 0.1)).translate((4.0, 4.0, 0.0))      # shares the edge x = y = 4, 0.1 mm long
    part = a + b
    assert len(_stl_pinches(part)[1]) > 0 and abs(_stl_pinches(part)[2] - 0.1) < 1e-6
    joined, n = _fill_pinches(part)
    assert n >= 1 and not len(_stl_pinches(joined)[1])
    assert len(joined.decompose()) == 1 and abs(joined.volume() - part.volume()) < 0.5
    # no room for a neck: the clip stops short of the contact
    clip = m3d.Manifold.cube((3.0, 3.0, 0.1))
    opened, n = _fill_pinches(part, clip=clip)
    assert n >= 1 and not len(_stl_pinches(opened)[1])
    assert opened.volume() < part.volume() and part.volume() - opened.volume() < 0.5
    same, n = _fill_pinches(a)
    assert n == 0 and abs(same.volume() - a.volume()) < 1e-9


def test_drawn_bodies_the_field_joins_across_a_hairline_are_welded():
    # A drawn cradle standing 0.05 mm off a wall: less than the grid can
    # see. Their cells share faces, so the optimizer carried the load
    # through that joint with no material of its own (a speaker's driver
    # cradle at its grille); the delivered part must have it too, not a
    # loose cradle with the load on it. A body the field holds apart stays.
    import manifold3d as m3d

    from loadpath.reconstruct import _weld_field_joints

    class Field:
        def __init__(self, labels):
            self.labels = labels

        def of(self, man):
            return self.labels[round(man.volume())]

    wall = m3d.Manifold.cube((24, 24, 4), True)
    cradle = m3d.Manifold.cube((10, 10, 6), True).translate((0, 0, 5.05))
    apart = m3d.Manifold.cube((4, 4, 4), True).translate((0, 0, 5.05 + 3.0 + 0.05 + 2.0))
    part = wall + cradle + apart
    kept = sorted(part.decompose(), key=lambda b: -b.volume())
    assert len(kept) == 3
    clip = m3d.Manifold.cube((24, 24, 40), True)
    field = Field({round(wall.volume()): {1}, round(cradle.volume()): {1}, round(apart.volume()): {2}})
    out, left, welded = _weld_field_joints(part, kept, field, 2.0, clip)
    assert welded > 0
    assert len(left) == 2, [b.volume() for b in left]
    bodies = sorted(out.decompose(), key=lambda b: -b.volume())
    assert len(bodies) == 2
    # the weld spans the gap, with at most a small fillet round the joint
    # (the facing surfaces within half a cell of each other)
    w = bodies[0] - wall - cradle
    assert 10 * 10 * 0.05 * 0.95 <= w.volume() <= 2 * 10 * 10 * 0.05
    from loadpath.reconstruct import trimesh_from_manifold

    lo, hi = trimesh_from_manifold(w).bounds
    assert lo[2] >= 2.0 - 1e-6 and hi[2] <= 2.05 + 1.0 + 1e-6
    assert max(abs(lo[0]), abs(hi[0]), abs(lo[1]), abs(hi[1])) <= 5.0 + 1.0 + 1e-6


def test_gap_weld_leaves_bodies_a_cell_apart_alone():
    import manifold3d as m3d

    from loadpath.reconstruct import _gap_weld, trimesh_from_manifold

    a = m3d.Manifold.cube((10, 10, 4), True)
    b = m3d.Manifold.cube((10, 10, 4), True).translate((0, 0, 7.0))
    assert _gap_weld(trimesh_from_manifold(a), trimesh_from_manifold(b), 1.0, 2.0) is None
    b = m3d.Manifold.cube((10, 10, 4), True).translate((0, 0, 4.1))
    w = _gap_weld(trimesh_from_manifold(a), trimesh_from_manifold(b), 1.0, 2.0)
    assert w is not None and len((a + b + w).decompose()) == 1
