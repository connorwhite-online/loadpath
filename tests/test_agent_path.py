"""
Tests for the path from a request to a delivered part, as an agent walks it:
the carried-mass load shorthand, check's lints for mistakes agents made in
the brief evals (a keep-out eating a preserve, a moment in N·m), the report's
`deliver` block, `verify` on any STL, the coarse warm start falling back
instead of crashing, and MCP tools that never raise.

Run: `python3 tests/test_agent_path.py` (well under a minute).
"""
import copy
import json
import os
import sys
import tempfile
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from loadpath import mcp_server  # noqa: E402
from loadpath.delivery import delivery  # noqa: E402
from loadpath.domain import build_domain  # noqa: E402
from loadpath.spec import study_from_dict  # noqa: E402
from loadpath.tasks import check_study, verify_mesh  # noqa: E402

LINK = {
    "name": "link",
    "designSpace": [{"type": "box", "min": [-10, -10, -5], "max": [110, 10, 5]}],
    "preserve": [{"id": "boss-a", "shape": {"type": "cylinder", "center": [0, 0, 0], "axis": "z", "length": 10, "r": 7}},
                 {"id": "boss-b", "shape": {"type": "cylinder", "center": [100, 0, 0], "axis": "z", "length": 10, "r": 7}}],
    "keepOut": [{"id": "a", "shape": {"type": "cylinder", "center": [0, 0, 0], "axis": "z", "length": 12, "r": 3}},
                {"id": "b", "shape": {"type": "cylinder", "center": [100, 0, 0], "axis": "z", "length": 12, "r": 3}}],
    "supports": [{"region": "a"}],
    "loadCases": [{"name": "p", "loads": [{"region": "b", "force": [0, -20, 0]}]}],
    "material": "pla",
    "goal": {"safetyFactor": 2},
    "resolution": {"pitch": 2.0},
}


def test_mass_load_is_weight_and_lever():
    s = copy.deepcopy(LINK)
    s["loadCases"][0]["loads"] = [{"region": "b", "mass": 1.0, "dynamic": 2, "gravity": "-y", "at": [160, 0, 0]}]
    assert check_study(s)["ok"]
    ld = study_from_dict(s).load_cases[0].loads[0]
    assert np.allclose(ld.force, [0, -19.62, 0]) and np.allclose(ld.moment, [0, 0, -60 * 19.62]), (ld.force, ld.moment)
    s["loadCases"][0]["loads"][0]["direction"] = "any"
    r = check_study(s)
    assert not r["ok"] and "turn with an" in r["error"], r


def test_check_flags_keepout_eating_a_preserve_but_not_a_bore():
    assert not [w for w in check_study(LINK)["warnings"] if "removes" in w]   # bores through their bosses
    s = copy.deepcopy(LINK)
    s["keepOut"].append({"id": "driver", "shape": {"type": "cylinder", "p0": [8.5, 0, -6], "p1": [8.5, 0, 6], "r": 4}})
    w = [w for w in check_study(s)["warnings"] if "removes" in w]
    assert w and "'driver'" in w[0] and "'boss-a'" in w[0], w


def test_check_flags_a_moment_in_newton_metres():
    s = copy.deepcopy(LINK)
    s["loadCases"][0]["loads"][0]["moment"] = [0, 0, 2.5]
    assert any("N·m" in w for w in check_study(s)["warnings"])


def test_cancelling_load_names_its_cause():
    s = copy.deepcopy(LINK)
    s["loadCases"] = [{"name": "self", "loads": [{"region": "boss-b", "force": [0, 5, 0]},
                                                 {"region": "boss-b", "force": [0, -5, 0]}]}]
    try:
        build_domain(study_from_dict(s))
    except ValueError as exc:
        assert "adds up to nothing" in str(exc), exc
    else:
        raise AssertionError("a load case that cancels itself should not build")


def test_deliver_gathers_the_best_outcome_and_its_warnings():
    summary = {"best": 1, "warnings": ["run-level"], "outcomes": [
        {"volumeFraction": 0.2, "passes": False, "safetyFactor": 1.0, "cases": []},
        {"volumeFraction": 0.3, "passes": True, "mass": 12.0, "safetyFactor": 3.0, "stl": "out/x-30.stl",
         "confirm": {"method": "tet", "safetyFactor": 3.0, "p5Member": 2.2},
         "cases": [{"name": "p", "safetyFactor": 3.0, "deflection": 0.45}],
         "print": {"warnings": ["needs a brim"]}, "durability": {"safetyFactor": 1.2, "target": 1.5},
         "warnings": []}]}
    d = delivery(summary, {"goal": {"safetyFactor": 2, "maxDeflection": 0.5}})
    assert d["passes"] and d["stl"] == "out/x-30.stl" and d["safetyFactorMethod"] == "tet", d
    assert d["governedBy"] == "maxDeflection" and d["p5Member"] == 2.2, d
    assert "run-level" in d["warnings"] and "needs a brim" in d["warnings"], d
    assert any("durability" in w for w in d["warnings"]), d
    none = delivery({**summary, "best": None}, {"goal": {"safetyFactor": 2}})
    assert none["passes"] is False and "not a passing part" in none["warnings"][0], none
    capped = {"best": None, "warnings": [], "outcomes": [{"volumeFraction": 0.7, "safetyFactor": 1.5, "cases": []}]}
    assert any("ceiling" in w for w in delivery(capped, {"goal": {"safetyFactor": 2}})["warnings"])


def test_verify_any_stl():
    import trimesh

    with tempfile.TemporaryDirectory() as d:
        bar = trimesh.creation.box(extents=(120, 20, 10))
        bar.apply_translation((50, 0, 0))
        for c in ((0, 0, 0), (100, 0, 0)):
            hole = trimesh.creation.cylinder(radius=3.0 / np.cos(np.pi / 128), height=12, sections=128)
            hole.apply_translation(c)
            bar = bar.difference(hole)
        path = os.path.join(d, "bar.stl")
        bar.export(path)
        r = verify_mesh(LINK, path, max_cells=8000)
        assert r["ok"] and r["bodies"] == 1 and r["passes"] and r["safetyFactor"] > 2, r
        assert not verify_mesh(LINK, os.path.join(d, "none.stl"))["ok"]


def test_mcp_tools_return_errors_instead_of_raising():
    assert mcp_server.guarded(mcp_server.cancel)("nope")["ok"] is False
    r = mcp_server.guarded(mcp_server.status)("nope")      # annotated tuple: the error dict comes back
    assert (r if isinstance(r, dict) else json.loads(r[0]))["ok"] is False
    s = copy.deepcopy(LINK)
    s["keepOut"][0]["shape"] = {"type": "mesh", "path": "missing.stl"}
    r = mcp_server.check(s, tempfile.gettempdir())
    assert not r["ok"] and "not found in" in r["error"], r


if __name__ == "__main__":
    failed = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print("ok  ", name)
            except Exception:  # noqa: BLE001
                failed += 1
                print("FAIL", name)
                traceback.print_exc()
    sys.exit(failed)
