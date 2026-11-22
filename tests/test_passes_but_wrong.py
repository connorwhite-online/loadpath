"""
"Passes but wrong" holes found by the intercooler agent (Nov 2026): a part
reported passes: true that was not what it said.

- goal.maxDeflection is the loads' mean travel; goal.maxDisplacement caps
  the peak anywhere on the part, and a mean that hides a larger peak over
  the number asked is said.
- Every number that gates a pass says how it was read (deliver.confirmation),
  and one read by a single method with little margin is flagged unconfirmed.
- A tet-confirmed part holds its travel limits on tets too, not only on the
  voxel grid that reads thin decks stiff.
- The delivered STL is checked sound (edges on two faces, no face crossing
  another, sections simple loops); a part that is not is not delivered.

Run: `python -m pytest -q tests/test_passes_but_wrong.py`
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest
import trimesh

from loadpath.delivery import confirmation, delivery
from loadpath.spec import study_from_dict
from loadpath.study import _margin, _sound, _travel_on_both, _whole
from loadpath.verify import verify_part

# a cantilever held at x=0, pushed down over its last 10 mm: the load's
# mean travel is under the tip's
BAR = {
    "designSpace": {"type": "box", "min": [0, 0, 0], "max": [100, 10, 10]},
    "supports": [{"shape": {"type": "box", "min": [-1, -1, -1], "max": [0.01, 11, 11]}}],
    "loadCases": [{"name": "tip", "loads": [
        {"shape": {"type": "box", "min": [60.0, -1, 9.9], "max": [70, 11, 11]}, "force": [0, 0, -10]}]}],
    "material": {"E": 2000, "nu": 0.3, "strength": 50, "density": 1.2},
    "goal": {"safetyFactor": 2},
    "resolution": {"pitch": 2.0},
}


def _bar():
    return trimesh.creation.box(extents=[100, 10, 10]).apply_translation([50, 5, 5])


def test_max_displacement_caps_the_peak_not_the_loads_mean():
    s = study_from_dict({**BAR, "goal": {"safetyFactor": 2, "maxDeflection": 10.0, "maxDisplacement": 0.5}})
    assert s.goal.max_displacement == 0.5
    rep = verify_part(s, _bar(), refine=0)
    c = rep["cases"][0]
    row = {r["name"]: r for r in c["clearances"]}["maxDisplacement"]
    # the free end beyond the load travels most: the peak, over the loads' mean
    assert row["travel"] == pytest.approx(c["maxDisplacement"], rel=1e-9)
    assert row["travel"] > 1.2 * c["deflection"], (row, c["deflection"])
    assert row["at"][0] > 95
    # it gates the margin like any clearance
    assert _margin(s, rep) <= 0.5 / row["travel"] + 1e-12
    with pytest.raises(ValueError, match="maxDisplacement"):
        study_from_dict({**BAR, "goal": {"safetyFactor": 2, "maxDisplacement": -1}})
    with pytest.raises(ValueError, match="safetyFactor or maxDeflection"):
        study_from_dict({**BAR, "goal": {"volumeFraction": 0.3, "maxDisplacement": 1}})


def test_a_mean_that_hides_the_peak_is_said():
    goal = {"safetyFactor": 2, "maxDeflection": 1.0}
    o = {"passes": True, "safetyFactor": 5.0, "volumeFraction": 0.3, "mass": 10.0,
         "confirm": {"method": "tet", "safetyFactor": 5.0, "margin": 2.5},
         "cases": [{"name": "lean", "safetyFactor": 5.0, "deflection": 0.8, "maxDisplacement": 3.9}]}
    d = delivery({"outcomes": [o], "best": 0}, {"goal": goal})
    assert any("loads' mean travel (0.80 mm)" in w and "3.90 mm" in w for w in d["warnings"]), d["warnings"]
    assert d["cases"][0]["maxDisplacement"] == 3.9
    # with the peak capped, nothing to say
    d = delivery({"outcomes": [o], "best": 0}, {"goal": {**goal, "maxDisplacement": 4.0}})
    assert not any("loads' mean travel" in w for w in d["warnings"])


def test_gating_numbers_say_how_they_were_read():
    goal = {"safetyFactor": 2, "maxDeflection": 2.0}
    cases = [{"name": "a", "deflection": 1.6, "clearances": [{"name": "deck", "travel": 0.9, "maxTravel": 1.0}]}]
    # the design grid alone: every limit near its margin is unconfirmed
    o = {"safetyFactor": 2.4, "cases": cases, "bucklingFactor": 30.0}
    c = confirmation(o, goal)
    assert c["limits"]["safetyFactor"]["method"] == "voxel"
    assert set(c["unconfirmed"]) == {"safetyFactor", "maxDeflection", "deck"}, c
    # a finer grid that agrees with the design grid confirms them
    o2 = {**o, "confirm": {"safetyFactor": 2.4, "margin": 1.1, "designGridMargin": 1.15}}
    assert confirmation(o2, goal)["unconfirmed"] == []
    # one that moved the reading by over a quarter has not converged
    o3 = {**o, "confirm": {"safetyFactor": 2.4, "margin": 1.1, "designGridMargin": 1.8}}
    c3 = confirmation(o3, goal)
    assert set(c3["unconfirmed"]) == {"safetyFactor", "maxDeflection", "deck"} and "voxelGridsDisagree" in c3
    # tets: the safety factor on tets, travel on both
    o4 = {**o, "confirm": {"method": "tet", "safetyFactor": 2.4, "margin": 1.1}}
    c4 = confirmation(o4, goal)
    assert c4["limits"]["maxDeflection"]["method"] == "voxel+tet" and c4["unconfirmed"] == []
    # buckling is read on one coarse grid: flagged when it is close
    o5 = {**o4, "bucklingFactor": 3.0}
    assert confirmation(o5, goal)["unconfirmed"] == ["buckling"]
    d = delivery({"outcomes": [{**o, "passes": True}], "best": 0}, {"goal": goal})
    assert d["confirmation"]["unconfirmed"] and any(w.startswith("unconfirmed: ") for w in d["warnings"])


def test_travel_gates_on_the_worse_of_voxel_and_tet():
    vox = [{"name": "a", "deflection": 1.0, "maxDisplacement": 1.5,
            "clearances": [{"name": "deck", "travel": 0.5, "maxTravel": 1.0}]},
           {"name": "b", "deflection": 2.0}]
    tet = [{"name": "a", "deflection": 1.8, "maxDisplacement": 3.0,
            "clearances": [{"name": "deck", "travel": 0.9, "maxTravel": 1.0, "at": [0, 0, 0]}]},
           {"name": "b", "deflection": 1.5}]
    out = _travel_on_both(vox, tet)
    a, b = out
    assert a["deflection"] == 1.8 and a["deflectionMethod"] == "tet" and a["maxDisplacement"] == 3.0
    assert a["clearances"][0]["travel"] == 0.9 and a["clearances"][0]["method"] == "tet"
    assert b["deflection"] == 2.0 and "deflectionMethod" not in b   # the voxel read more: kept
    assert vox[0]["deflection"] == 1.0   # not changed in place


def test_tet_crosscheck_reads_clearances():
    # the tet cross-check reads each clearance on its own mesh, as verify does on voxels
    from loadpath.crosscheck import _clearance_nodes, _clearance_rows

    s = study_from_dict({**BAR, "goal": {"safetyFactor": 2, "maxDisplacement": 1.0, "clearances": [
        {"shape": {"type": "box", "min": [45, -1, -1], "max": [55, 11, 11]}, "maxTravel": 0.2, "name": "mid",
         "direction": "-z"}]}})
    pts = np.array([[50.0, 5, 5], [99, 5, 5], [10, 5, 5]])
    u = np.array([[0, 0, -0.1], [0, 0, -2.0], [0, 0.01, 0]])
    nodes = _clearance_nodes(s, pts, 0.5)
    rows = {r["name"]: r for r in _clearance_rows(s, "tip", pts, u, nodes)}
    assert rows["mid"]["travel"] == pytest.approx(0.1) and rows["mid"]["at"][0] == 50
    assert rows["maxDisplacement"]["travel"] == pytest.approx(2.0) and rows["maxDisplacement"]["at"][0] == 99


def test_an_unsound_stl_is_not_delivered():
    # watertight, one closed surface per solid, and two solids pushed
    # through each other: the voxel analysis reads it as solid
    a = trimesh.creation.box((10, 10, 10))
    b = trimesh.creation.box((10, 10, 10)).apply_translation((4.3, 3.7, 2.9))
    crossed = trimesh.util.concatenate([a, b])
    rep = {"bodies": 1}
    _sound(rep, crossed)
    assert rep["sound"]["ok"] is False and rep["sound"]["crossingFaces"] > 0, rep["sound"]
    assert not _whole(rep)
    assert any("not sound" in w for w in rep["warnings"])
    ok = {"bodies": 1}
    _sound(ok, a)
    assert ok["sound"]["ok"] and _whole(ok)
    # deliver: not passing, and why
    o = {"passes": True, "safetyFactor": 5.0, "mass": 1.0, "volumeFraction": 0.3, "cases": [],
         "sound": rep["sound"]}
    d = delivery({"outcomes": [o], "best": 0}, {"goal": {"safetyFactor": 2}})
    assert d["passes"] is False and any("not sound" in w for w in d["warnings"])


def test_a_four_way_edge_is_not_sound():
    # two cubes sharing one edge: manifold in memory as two shells, but an
    # STL welds them into an edge on four faces
    a = trimesh.creation.box((10, 10, 10)).apply_translation((5, 5, 5))
    b = trimesh.creation.box((10, 10, 10)).apply_translation((15, 15, 5))
    rep = {}
    _sound(rep, trimesh.util.concatenate([a, b]))
    assert rep["sound"]["ok"] is False and rep["sound"]["nonManifoldEdges"] > 0, rep["sound"]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))


def test_clearance_relative_to_its_surround_ignores_rigid_motion():
    # a socket read against the opening around it: the whole shell swinging
    # on far supports moves both together, so only the socket's own shift counts
    from loadpath.crosscheck import _clearance_nodes, _clearance_rows

    s = study_from_dict({**BAR, "goal": {"safetyFactor": 2, "clearances": [
        {"shape": {"type": "box", "min": [45, -1, -1], "max": [55, 11, 11]}, "maxTravel": 0.2, "name": "port",
         "relativeTo": {"shape": {"type": "box", "min": [30, -1, -1], "max": [70, 11, 11]}}},
        {"shape": {"type": "box", "min": [45, -1, -1], "max": [55, 11, 11]}, "maxTravel": 0.2, "name": "abs"}]}})
    rng = np.random.default_rng(0)
    pts = np.c_[rng.uniform(30, 70, 200), rng.uniform(0, 10, 200), rng.uniform(0, 10, 200)]
    w, t = np.array([0.0, 0.002, 0.01]), np.array([0.3, -0.5, 0.8])
    u = t + np.cross(w, pts - [50, 5, 5])
    port = np.flatnonzero((pts[:, 0] >= 45) & (pts[:, 0] <= 55))
    u[port] += [0, 0, 0.05]   # the socket's own give
    rows = {r["name"]: r for r in _clearance_rows(s, "c", pts, u, _clearance_nodes(s, pts, 0.5))}
    assert rows["abs"]["travel"] > 0.8
    assert 0.02 < rows["port"]["travel"] < 0.06


def test_unfolding_leaves_sound_and_truly_crossed_parts_alone():
    # the float32 unfold pass only takes back sub-micron folds: a sound part
    # comes back as it was, and two solids pushed through each other are not
    # "repaired" into a part (the soundness check still refuses them)
    from loadpath.reconstruct import _unfold_float32

    box = trimesh.creation.box((10, 10, 10))
    assert _unfold_float32(box) is box
    a = trimesh.creation.box((10, 10, 10))
    b = trimesh.creation.box((10, 10, 10)).apply_translation((4.3, 3.7, 2.9))
    crossed = trimesh.util.concatenate([a, b])
    info = {}
    out = _unfold_float32(crossed, info)
    rep = {"bodies": 1}
    _sound(rep, out)
    assert rep["sound"]["ok"] is False
