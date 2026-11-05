"""
explore.py and costing.py: the routes a study fans out to, the cost model's
shape, and one tiny end-to-end explore (two methods, quick mode).

Run: `python -m pytest tests/test_explore.py`.
"""
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import trimesh

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from loadpath import costing  # noqa: E402
from loadpath.explore import DEFAULT_METHODS, plan  # noqa: E402
from loadpath.schema import lint  # noqa: E402

ARM = json.loads((ROOT / "loadpath/evals/cases/arm-link.json").read_text())
TINY = {
    "name": "tiny",
    "designSpace": {"type": "box", "min": [0, -15, -4], "max": [60, 15, 4]},
    "preserve": [{"id": "eye", "shape": {"type": "cylinder", "center": [48, 0, 0], "axis": "z", "length": 8, "r": 9}}],
    "keepOut": [{"id": "pin", "shape": {"type": "cylinder", "center": [48, 0, 0], "axis": "z", "length": 12, "r": 4}}],
    "supports": [{"shape": {"type": "box", "min": [-1, -16, -5], "max": [0.01, 16, 5]}}],
    "loadCases": [{"name": "pull", "loads": [{"region": "pin", "force": [0, -80, 0]}]}],
    "material": "pa12",
    "goal": {"volumeFraction": 0.4},
    "minMember": 3.0,
    "maxIterations": 25,
    "resolution": {"pitch": 2.0},
}


def test_default_plan_one_route_per_method():
    p = plan(ARM, base_dir=str(ROOT))
    assert [r["method"] for r in p["routes"]] == list(DEFAULT_METHODS)
    by = {r["method"]: r["study"] for r in p["routes"]}
    assert by["fdm"]["process"] == {"type": "fdm"} and by["fdm"]["material"] == "petg"
    # cast and milled: no layered process, the access constraint along the thin axis (z here)
    assert p["thinAxis"] == "z"
    assert "process" not in by["cast"] and by["cast"]["casting"] == {"draw": "+z", "sides": 2}
    assert by["milled-3axis"]["milling"]["directions"] == ["+z", "-z"]
    for r in p["routes"]:
        assert not [x for x in lint(r["study"], "study") if x["level"] == "error"], r["id"]
        assert "explore" not in r["study"] and r["study"]["name"].endswith(r["id"])


def test_materials_go_to_the_methods_that_make_them():
    p = plan(ARM, {"methods": ["fdm", "mjf", "lpbf", "milled-5axis"],
                   "materials": ["petg", "pa12", "ti64", "7075-t6"]}, str(ROOT))
    ids = [r["id"] for r in p["routes"]]
    assert ids == ["fdm-petg", "fdm-pa12", "mjf-pa12", "lpbf-ti64", "milled-5axis-7075-t6"], ids
    # 5-axis cuts from all six sides: no directions given
    assert "directions" not in p["routes"][-1]["study"]["milling"]
    p = plan(ARM, {"routes": [{"method": "cast", "material": "pla"}, {"method": "cnc", "material": "6061"}]},
             str(ROOT))
    assert [r["id"] for r in p["routes"]] == ["milled-3axis-6061-t6"]
    assert p["skipped"] and "pla" in p["skipped"][0]["why"]


def test_route_keeps_the_studys_own_process_settings():
    st = {**ARM, "process": {"type": "fdm", "buildDirection": "+y", "interlayer": 0.5}, "material": "petg"}
    by = {r["method"]: r["study"] for r in plan(st, {"methods": ["fdm", "lpbf"]}, str(ROOT))["routes"]}
    assert by["fdm"]["process"] == {"type": "fdm", "buildDirection": "+y", "interlayer": 0.5}
    assert by["lpbf"]["process"] == {"type": "lpbf", "buildDirection": "+y"}


def test_cost_model_shape():
    box = trimesh.creation.box((100, 30, 20))      # 60 cm^3
    rows = {m: costing.estimate(m, mat, box, mass=60 * d)
            for m, mat, d in (("fdm", "petg", 1.27), ("mjf", "pa12", 1.01), ("sla", "resin-tough", 1.1),
                              ("lpbf", "alsi10mg", 2.67), ("lpbf", "ti64", 4.41), ("cast", "a356-t6", 2.685),
                              ("milled-3axis", "6061-t6", 2.7), ("milled-5axis", "6061-t6", 2.7))}
    for m, r in rows.items():
        assert r["unitCost"] > 0 and r["leadDays"] > 0, m
    # titanium costs more than aluminium on the same machine; 5 axes more than 3
    assert rows["lpbf"]["material"] == "ti64" and rows["lpbf"]["unitCost"] > \
        costing.estimate("lpbf", "alsi10mg", box, mass=160)["unitCost"]
    assert rows["milled-5axis"]["unitCost"] > rows["milled-3axis"]["unitCost"]
    # tooling spreads over the order: a cast part gets much cheaper per piece
    one = costing.estimate("cast", "a356-t6", box, mass=161)
    many = costing.estimate("cast", "a356-t6", box, mass=161, quantity=500)
    assert many["unitCost"] < 0.2 * one["unitCost"] and many["leadDays"] > one["leadDays"]
    # a slicer's print time replaces the estimate
    sliced = costing.estimate("fdm", "petg", box, mass=76, print_seconds=3600)
    assert sliced["machineHours"] == 1.0 and "slicer" in sliced["note"]
    # overrides: a cheaper machine rate, a material price
    cheap = costing.estimate("fdm", "petg", box, mass=76, rates={"fdm": {"machinePerHour": 1.0},
                                                                 "materialPrice": {"petg": 10}})
    assert cheap["unitCost"] < rows["fdm"]["unitCost"] and cheap["materialPricePerKg"] == 10
    lp = costing.estimate("lpbf", "alsi10mg", box, mass=160)
    xo = costing.crossover([{**lp, "route": "lpbf"}, {**one, "route": "cast"}], (1, 1000))
    assert xo[0]["route"] == "lpbf" and xo[1]["route"] == "cast"


def test_explore_end_to_end_tiny():
    with tempfile.TemporaryDirectory() as tmp:
        sp = os.path.join(tmp, "tiny.json")
        Path(sp).write_text(json.dumps(TINY))
        out = os.path.join(tmp, "out")
        env = dict(os.environ, PYTHONPATH=str(ROOT), LOADPATH_NO_RENDER="1")
        p = subprocess.run([sys.executable, "-m", "loadpath", "explore", sp, "--out", out, "--quick",
                            "--methods", "mjf,milled-3axis", "--quantity", "5", "--rank", "cost"],
                           capture_output=True, text=True, env=env, timeout=900)
        assert p.returncode in (0, 1), p.stderr[-2000:]
        rep = json.loads(Path(out, "report.json").read_text())
        assert [r["method"] for r in rep["routes"]].count("mjf") == 1 and len(rep["routes"]) == 2
        for r in rep["routes"]:
            assert r["mass"] > 0 and r["unitCost"] > 0 and os.path.exists(r["stl"]), r
        # the milled part is aluminium: heavier than the nylon one at the same volume fraction
        by = {r["method"]: r for r in rep["routes"]}
        assert by["milled-3axis"]["mass"] > by["mjf"]["mass"]
        assert rep["rank"] == "cost" and rep["quantity"] == 5
        assert Path(out, "explore.md").exists()
        # a second explore in the same folder reads the routes back
        p2 = subprocess.run([sys.executable, "-m", "loadpath", "explore", sp, "--out", out, "--quick",
                             "--methods", "mjf,milled-3axis", "--quantity", "5", "--rank", "cost"],
                            capture_output=True, text=True, env=env, timeout=300)
        rep2 = json.loads(Path(out, "report.json").read_text())
        assert all(r.get("reused") for r in rep2["routes"]), p2.stderr[-2000:]
