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
