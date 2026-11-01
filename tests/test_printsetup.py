import trimesh

from loadpath.printsetup import search_print_setup
from loadpath.spec import study_from_dict


def _study(limit):
    return study_from_dict({
        "name": "bar", "material": "petg", "process": {"type": "fdm", "buildDirection": "+z"},
        "designSpace": [{"type": "box", "min": [0, 0, 0], "max": [40, 8, 4]}],
        "preserve": [{"id": "a", "shape": {"type": "box", "min": [0, 0, 0], "max": [4, 8, 4]}},
                     {"id": "b", "shape": {"type": "box", "min": [36, 0, 0], "max": [40, 8, 4]}}],
        "supports": [{"region": "a"}], "loads": [{"region": "b", "force": [0, 0, -5]}],
        "goal": {"safetyFactor": 2.0}, "resolution": {"maxElements": 800},
        "buildSim": {"maxDistortion": limit, "cells": 1500}})


def test_print_setup_search_ranks_by_goal_then_warp():
    """A flat PETG bar: a brim and a warmer bed cut the lift, and the best
    row is the lightest setup that meets the goal and the warp limit (here
    one material, so the least lift among the passing ones)."""
    mesh = trimesh.creation.box(extents=(40, 8, 4))
    mesh.apply_translation((20, 4, 2))
    st = _study(10.0)
    r = search_print_setup(st, mesh, builds=["+z", "+y"], beds=[60.0, 80.0], brims=(0.0, 5.0), cells=1500,
                           verify_cells=3000, top=1)
    rows = r["rows"]
    assert r["best"] is rows[0] and r["best"]["passes"]
    flat = {(q["bed"], q["brim"]): q["lift"] for q in rows if q["build"] == "+z"}
    if (60.0, 0.0) in flat and (80.0, 0.0) in flat:
        assert flat[(80.0, 0.0)] < flat[(60.0, 0.0)]
    brims = [q for q in rows if q["brim"] == 5.0]
    assert brims and all(q["lift"] < flat.get((q["bed"], 0.0), 1e9) + 1e-9 for q in brims if q["build"] == "+z")
    passing = [q for q in rows if q["passes"]]
    assert all(q["meetsGoal"] for q in passing)
    tight = search_print_setup(_study(1e-6), mesh, builds=["+z"], beds=[80.0], brims=(0.0,), cells=1500,
                               verify_cells=3000)
    assert not tight["best"]["passes"]
