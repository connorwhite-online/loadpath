"""
What makes a delivered part look made on purpose, held as library behaviour
(2026-11-21, after the held-out parts came out lumpy and lopsided):

- a problem that is its own mirror image is designed mirror-symmetric
  without being asked (domain._found_symmetry);
- a load carried on a face that lines a keep-out rests on pinned cells on a
  small part as on a large one (domain: the strap under loaded nodes);
- the mesh is drawn from a field that says where in a cell the surface is
  (simp.boundary_field), not from the solid-or-void one.
"""
import copy

import numpy as np

from loadpath.domain import build_domain
from loadpath.simp import boundary_field
from loadpath.spec import study_from_dict

BAR = {
    "name": "bar",
    "designSpace": [{"type": "box", "min": [0, 0, 0], "max": [40, 16, 16]}],
    "preserve": [{"id": "foot", "shape": {"type": "box", "min": [0, 0, 0], "max": [3, 16, 16]}},
                 {"id": "tip", "shape": {"type": "box", "min": [37, 5, 5], "max": [40, 11, 11]}}],
    "supports": [{"region": "foot", "fix": "xyz"}],
    "loadCases": [{"name": "down", "loads": [{"region": "tip", "force": [0, 0, -50]}]}],
    "material": "pla",
    "goal": {"volumeFraction": 0.3},
    "resolution": {"pitch": 2.0},
}


def _domain(raw):
    study = study_from_dict(copy.deepcopy(raw))
    return study, build_domain(study, pitch=2.0)


def test_a_mirror_symmetric_problem_is_held_symmetric():
    study, d = _domain(BAR)
    assert sorted(d.mirror) == ["y"], "the bar and its load are their own image across y, and only y"
    assert study.symmetry_missed.get("z", "").startswith("load case"), study.symmetry_missed
    assert "x" in study.symmetry_missed


def test_a_face_on_a_sampling_plane_does_not_cost_the_symmetry():
    # 54 x 42 at 0.8: the grid overhangs each face by a quarter cell, so the
    # sub-cell samples of the edge cells fall on the faces themselves. Decided
    # by rounding, one corner column came out and both mirror planes with it.
    raw = {"designSpace": [{"type": "box", "min": [-27, -21, 0], "max": [27, 21, 6]}],
           "supports": [{"shape": {"type": "box", "min": [-27, -21, -0.5], "max": [27, 21, 0.5]}, "fix": "xyz"}],
           "loadCases": [{"name": "p", "loads": [{"shape": {"type": "cylinder", "center": [0, 0, 5.8], "axis": "z",
                                                            "length": 0.8, "r": 3}, "force": [0, 0, -50]}]}],
           "material": "petg", "goal": {"safetyFactor": 2}}
    d = build_domain(study_from_dict(raw, base_dir="."), 0.8)
    assert {"x", "y"} <= set(d.mirror)


def test_a_one_way_side_load_is_not_taken_as_symmetric():
    raw = copy.deepcopy(BAR)
    raw["loadCases"].append({"name": "side", "loads": [{"region": "tip", "force": [0, 20, 0]}]})
    study, d = _domain(raw)
    assert "y" not in d.mirror and study.symmetry_missed["y"].startswith("load case")
    # ... and is, with the opposite push beside it
    raw["loadCases"].append({"name": "other side", "loads": [{"region": "tip", "force": [0, -20, 0]}]})
    assert "y" in _domain(raw)[1].mirror


def test_symmetry_can_be_turned_off_and_declared():
    raw = dict(copy.deepcopy(BAR), symmetry=False)
    assert not _domain(raw)[1].mirror
    raw["symmetry"] = ["z"]
    assert sorted(_domain(raw)[1].mirror) == ["y", "z"], "a declared plane is held with the found one"


def test_overhang_control_is_not_mirrored_across_the_bed():
    raw = dict(copy.deepcopy(BAR), process={"type": "fdm", "buildDirection": "+y"})
    study, d = _domain(raw)
    if study.process.overhangs:
        assert "y" not in d.mirror and study.symmetry_missed["y"] == "build direction"


ROD = {
    "name": "rod",
    "designSpace": [{"type": "box", "min": [-24, -7, 0], "max": [24, 7, 44]}],
    "keepOut": [{"id": "rod", "bore": False,
                 "shape": {"type": "cylinder", "center": [0, 0, 28], "axis": "y", "length": 18, "r": 10}}],
    "preserve": [{"id": "feet", "shape": {"type": "box", "min": [-24, -7, 0], "max": [24, 7, 2]}}],
    "supports": [{"region": "feet", "fix": "xyz"}],
    "loadCases": [{"name": "up", "loads": [{"region": "rod", "force": [0, 0, 180], "distribution": "bearing"}]}],
    "material": "petg",
    "goal": {"volumeFraction": 0.3},
}


def test_a_load_on_a_keep_out_face_rests_on_pinned_cells():
    """The rod's saddle touches far more than 3% of this small part's cells,
    which used to leave it unpinned: the bearing load then sat on cells the
    optimizer could (and did) empty, the compliance rose 80x as the
    projection sharpened, and the design came out ragged."""
    study = study_from_dict(copy.deepcopy(ROD))
    d = build_domain(study, pitch=1.3)
    f = np.abs(np.asarray(d.loads[0], float)).reshape(-1, 3).sum(axis=1) > 0
    loaded = f[d.mesh.enode].any(axis=1)                 # cells with a loaded node
    assert loaded.sum() > 20
    assert d.preserve[loaded].mean() > 0.9, "the cells the load is applied to are pinned solid"
    assert d.level_hint is not None and np.isfinite(d.level_hint).any()
    hint = d.level_hint[np.isfinite(d.level_hint)]
    assert ((hint > 0.05) & (hint < 0.95)).any(), "the strap's surface is placed inside its cells"


def test_boundary_field_places_the_surface_inside_cells():
    # a plane tilted off the grid, filtered field ramping over 3 cells
    n = 24
    i, j = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    dist = (0.37 * i + j - 14.2) / np.hypot(0.37, 1.0)           # cells from the surface
    xt = np.clip(0.5 - dist / 6.0, 0.0, 1.0).ravel()             # crosses 0.5 at the surface, slope 1/(2R), R = 3
    xp = (xt >= 0.5).astype(float)
    lev = boundary_field(xp, [(xt, 0.5)], 3.0)
    assert np.array_equal(lev >= 0.5, xp >= 0.5), "the same design"
    grey = (lev > 0.02) & (lev < 0.98)
    assert grey.sum() >= n, "cells the surface crosses are part full"
    # the level's own 0.5 crossing along each row sits on the plane to a small fraction of a cell
    L = lev.reshape(n, n)
    for r in range(n):
        k = int(np.argmax(L[r] < 0.5))
        cross = (k - 1) + (L[r, k - 1] - 0.5) / (L[r, k - 1] - L[r, k])
        assert abs(cross - (14.2 - 0.37 * r)) < 0.08, (r, cross)
    # a cell changed after the projection keeps its side, a quarter off the cut
    xq = xp.copy()
    far = int(np.argmin(xt))
    xq[far] = 1.0
    assert boundary_field(xq, [(xt, 0.5)], 3.0)[far] == 0.75


def test_boundary_field_keeps_a_pinned_strap_whole():
    xt = np.array([0.9, 0.6, 0.2, 0.1])
    xp = np.array([1.0, 1.0, 1.0, 0.0])                # third cell pinned solid over a void design
    hint = np.array([np.nan, np.nan, 0.8, 0.3])
    lev = boundary_field(xp, [(xt, 0.5)], 2.0, hint)
    assert lev[2] == 0.8 and lev[3] == 0.3 and lev[0] == 1.0


def test_boundary_field_leaves_pinned_cells_full():
    xt = np.array([0.9, 0.2, 0.1])
    xp = np.array([1.0, 1.0, 0.0])                     # second cell: a boss, pinned over a void design
    lev = boundary_field(xp, [(xt, 0.5)], 2.0, None, np.array([False, True, False]))
    assert lev[1] == 1.0, "a drawn solid's cells are full: its surface is the drawn shape's"


def test_beside_a_screw_head_clearance_the_part_ends_clean():
    """A boss 0.25 mm inside a small part's wall, the screw head's clearance
    standing on its seat. Between the clearance and the wall there is no
    room for a member: nothing is left there (it was a wall tapering to a
    feather edge, then chewed back as unformable). Where there is room the
    material runs on through the clearance's wall and is cut on it, and the
    surface beside the cut is smooth (smoothing was undone all along it,
    which left the marching cubes' facets as a torn rim round the seat)."""
    from loadpath.reconstruct import reconstruct

    spec = {"name": "peg", "designSpace": [{"type": "box", "min": [-10, -5, 0], "max": [10, 5, 12]}],
            "fasteners": [{"id": "a", "size": "M3", "seat": [6, 0, 3.5], "direction": "-z", "grip": 3.5,
                           "end": "face", "driver": 12}],
            "supports": [{"region": "a-bore", "fix": "xy"},
                         {"shape": {"type": "cylinder", "center": [6, 0, -0.1], "axis": "z", "length": 0.8,
                                    "r": 3.2}, "fix": "z"}],
            "loadCases": [{"name": "pull", "loads": [{"shape": {"type": "box", "min": [-10, -5, 11],
                                                                "max": [-6, 5, 12.5]}, "force": [0, 0, 50]}]}],
            "material": "pa12", "process": "sls", "goal": {"volumeFraction": 0.5},
            "resolution": {"pitch": 1.04}}
    s = study_from_dict(spec)
    d = build_domain(s)
    m, _ = reconstruct(s, d, np.ones(d.mesh.n_elem))
    # the crescent between the clearance (r 3.75 about x = 6) and the wall at x = 10
    crescent = np.array([[9.87, 0.0, z] for z in (4.5, 6, 8)] + [[9.8, 1.0, z] for z in (4.5, 6, 8)])
    assert not m.contains(crescent).any()
    # and where a member fits, beside the clearance and away from it, the block is whole
    assert m.contains(np.array([[8.8, 4.2, 6.0], [2.0, 4.0, 6.0]])).all()
    # the organic surface within 0.6 mm of the clearance's wall has no torn edges
    c = m.triangles_center
    rad = np.hypot(c[:, 0] - 6, c[:, 1])
    sel = (rad > 3.76) & (rad < 4.4) & (c[:, 2] > 2.5) & (c[:, 2] < 5.0) & (c[:, 0] < 9.9)
    both = sel[m.face_adjacency].all(axis=1)
    assert sel.sum() > 100 and not (np.degrees(m.face_adjacency_angles[both]) > 50).any()


def test_a_drawn_wall_thinner_than_a_member_keeps_its_place():
    """Two air passages meeting at an angle, walled 1.2 mm: between the end
    of one and the floor of the other the wall runs out to a knife edge.
    The room for a member is the optimizer's material's affair, not the
    drawn wall's: taken out of the field there, the wall came back as the
    blueprint alone beside what was left, and the surfaces crossed (a
    duct, 60 faces, not delivered)."""
    from loadpath.buildcheck import sound_check
    from loadpath.reconstruct import reconstruct

    spec = {"name": "elbow", "designSpace": [{"type": "box", "min": [-12, -12, 0], "max": [12, 12, 24]}],
            "keepOut": [{"id": "air", "wall": 1.2, "shape": {"type": "union", "shapes": [
                {"type": "cylinder", "p0": [0, 0, 25], "p1": [0, 0, 14], "r": 10},
                {"type": "cylinder", "p0": [0, 0, 14], "p1": [0, 8, 3], "r": 5},
                {"type": "box", "min": [-8, 6, -1], "max": [8, 10, 5]}]}}],
            "supports": [{"shape": {"type": "box", "min": [-12, -12.5, 10], "max": [12, -11.7, 22]}, "fix": "xyz"}],
            "loadCases": [{"name": "pull", "loads": [{"shape": {"type": "box", "min": [-12, 11.5, 14],
                                                                "max": [12, 12.5, 24]}, "force": [0, 0, -5]}]}],
            "material": "pla", "process": {"type": "fdm", "buildDirection": "+z"}, "goal": {"volumeFraction": 0.5},
            "resolution": {"pitch": 0.63}}
    s = study_from_dict(spec)
    d = build_domain(s)
    m, _ = reconstruct(s, d, np.ones(d.mesh.n_elem))
    r = sound_check(m)
    assert r["ok"], r.get("detail")


def test_a_loaded_bore_is_lined_a_member_thick():
    """A pin through a fork's two arms, pushing up. The cells lining the
    bore are pinned so the load rests on material; one cell of them is a
    wall thinner than the mesh keeps, so the analysis had a ring round the
    pin and the part had half of one (a fork's axle bores, open under the
    axle). The lining is a member: pinned to the minimum member's
    thickness, all the way round."""
    spec = {"name": "fork", "designSpace": [{"type": "box", "min": [-12, -14, -40], "max": [12, 14, 0]}],
            "keepOut": [{"id": "pin", "shape": {"type": "cylinder", "center": [0, 0, -30], "axis": "y",
                                                "length": 30, "r": 3.0}},
                        {"id": "roller", "shape": {"type": "cylinder", "center": [0, 0, -30], "axis": "y",
                                                   "length": 14, "r": 11.5}, "bore": False}],
            "supports": [{"shape": {"type": "box", "min": [-12, -14, -1], "max": [12, 14, 0.5]}, "fix": "xyz"}],
            "loadCases": [{"name": "up", "loads": [{"region": "pin", "force": [0, 0, 200],
                                                    "distribution": "bearing"}]}],
            "material": "petg", "process": {"type": "fdm", "buildDirection": "+z"},
            "goal": {"volumeFraction": 0.3}, "resolution": {"pitch": 0.9}}
    s = study_from_dict(spec)
    d = build_domain(s)
    c = d.mesh.elem_centers()
    r = np.hypot(c[:, 0], c[:, 2] + 30)
    wall = (np.abs(c[:, 1]) > 8.0) & (r > 3.0) & (r < 3.0 + s.min_member - 0.5 * d.pitch)
    assert wall.sum() > 300 and np.asarray(d.preserve, bool)[wall].all()
    # and no further: the rest is the optimizer's
    beyond = (np.abs(c[:, 1]) > 8.0) & (r > 3.0 + s.min_member + d.pitch) & (r < 8.0)
    assert not np.asarray(d.preserve, bool)[beyond].any()
