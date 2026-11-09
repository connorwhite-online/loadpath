"""Heat-exchanger path fixes: `check` on an exchanger spec, sizing that takes
its options and returns only designs that mesh, placed in the envelope, the
design report's mass, warnings and status file, and the duty's sign. Small
grids only."""
import json
import os

import numpy as np
import pytest

from loadpath.__main__ import main
from loadpath.heatx import design_heat_exchanger, grid_plan, heat_exchanger_from_spec
from loadpath.heatx_thermal import rate, size_core
from loadpath.sizing import PROCESS_LIMITS, size_exchanger
from loadpath.tasks import check_document, heat_exchanger, size_heat_exchanger

DUTY = {"A": {"fluid": "water", "lPerMin": 1.0, "tIn": 80}, "B": {"fluid": "water", "lPerMin": 1.0, "tIn": 20},
        "wall": "alsi10mg"}
ENV = [80, 70, 40]
CUBE = {"name": "cube", "core": {"type": "box", "min": [0, 0, 0], "max": [30, 30, 30]}, "layout": "cross",
        "axes": "xy", "cell": 8, "wall": 1.0, "header": 4, "portDiameter": 6, "portLength": 4, "step": 0.45}


def _write(tmp_path, doc, name="spec.json"):
    p = tmp_path / name
    p.write_text(json.dumps(doc))
    return str(p)


def test_check_reads_an_exchanger_spec_with_its_grid_and_memory(tmp_path, capsys):
    res = check_document(CUBE)
    assert res["ok"] and res["kind"] == "heatExchanger"
    assert res["gridStep"] == 0.45 and res["gridPoints"] > 1e6 and 0 < res["peakMemoryGB"] < 3
    assert main(["check", _write(tmp_path, CUBE)]) == 0
    out = json.loads(capsys.readouterr().out)
    assert "designSpace" not in json.dumps(out) and out["gridPoints"] == res["gridPoints"]
    # a core too big for the point cap at this wall: refused, saying how to raise the cap
    big = {**CUBE, "core": {"type": "box", "min": [0, 0, 0], "max": [300, 300, 300]}, "wall": 0.5, "cell": 4}
    big.pop("step")
    bad = check_document(big)
    assert not bad["ok"] and "--max-points" in bad["error"] and bad["maxPointsNeeded"] > 16e6
    # an envelope spec checks for sizing; lattice options lint as lattice
    env = check_document({"envelope": ENV, "duty": DUTY})
    assert env["ok"] and env["mode"] == "envelope"
    assert check_document({"cell": 6, "minWall": 0.8, "maxWall": 2})["kind"] == "lattice"
    assert not check_document({"envelope": ENV})["ok"]


def test_envelope_sizing_takes_min_wall_splits_and_axes_and_sits_in_the_envelope():
    kw = dict(process="lpbf", cells=[6.0, 8.0], fractions=(0.6, 1.0), max_pressure_drop={"A": 5000, "B": 5000})
    res = size_exchanger(DUTY, ENV, min_wall=0.8, splits=(0.5,), axes="yx", **kw)
    assert res["candidates"]
    for c in res["candidates"]:
        assert c["wall"] >= 0.8 and c["split"] == 0.5 and c["axes"] == "yx"
        assert c["fluidAxes"] == {"A": "y", "B": "x"} and "coreMass" in c and "mass" not in c
        s = c["spec"]
        lo, hi = np.array(s["core"]["min"]), np.array(s["core"]["max"])
        allow = s["header"] + s["shell"] + s["portLength"]
        # centred in the envelope's frame: ports and plenums inside [0, ENV]
        assert np.all(lo[:2] >= allow - 0.1) and np.all(hi[:2] <= np.array(ENV[:2]) - allow + 0.1)
        assert lo[2] >= s["shell"] - 0.1 and hi[2] <= ENV[2] - s["shell"] + 0.1
    assert res["envelopeFrame"]["max"] == ENV
    with pytest.raises(ValueError):
        size_exchanger(DUTY, ENV, axes="xz", **kw)
    # the CLI names: mass (metal the old name) and pressure, in both sizers
    for obj in ("mass", "metal", "pressure", "effectiveness"):
        assert size_exchanger(DUTY, ENV, objective=obj, walls=[0.8], splits=(0.5,), **kw)["candidates"]
        assert size_core("gyroid", 30e3, [30, 30], DUTY, objective=obj, cells=[6.0, 8.0])["best"]


def test_the_sizer_returns_only_designs_that_mesh_within_the_cap():
    kw = dict(process="lpbf", cells=[5.0, 6.0], walls=[0.5, 0.8], splits=(0.5,), fractions=(0.6, 1.0),
              max_pressure_drop={"A": 5000, "B": 5000}, objective="mass")
    res = size_exchanger(DUTY, ENV, max_points=2_000_000, **kw)
    assert res["unmeshable"] > 0 and "--max-points" in res["note"]
    for c in res["candidates"]:
        hx = heat_exchanger_from_spec({k: v for k, v in c["spec"].items() if k not in ("name", "duty")})
        hx.max_points = 2_000_000
        assert grid_plan(hx)["fits"] and c["grid"]["points"] <= 2.1e6
    # and the design step's refusal says how to raise the cap
    hx = heat_exchanger_from_spec({k: v for k, v in CUBE.items() if k not in ("name", "step")})
    hx.max_points = 200_000
    with pytest.raises(ValueError, match="--max-points"):
        design_heat_exchanger(hx)


def test_size_out_writes_the_ranked_json(tmp_path, capsys):
    spec = {**CUBE, "duty": DUTY, "process": "lpbf"}
    out = tmp_path / "sized"
    assert main(["heatx", _write(tmp_path, spec), "--size", "--out", str(out), "--objective", "mass",
                 "--min-wall", "0.6"]) == 0
    printed = json.loads(capsys.readouterr().out)
    saved = json.loads((out / "sizing.json").read_text())
    assert saved["candidates"] == printed["candidates"] and saved["objective"] == "mass"
    assert all(c["wall"] >= 0.6 for c in saved["candidates"])
    best = json.loads((out / "best.json").read_text())
    assert best["wall"] == saved["best"]["wall"] and best["cell"] == saved["best"]["cell"]
    # an envelope spec writes there too
    res = size_heat_exchanger({"name": "env", "envelope": [60, 50, 30], "duty": DUTY, "layout": "cross",
                               "sizing": {"process": "lpbf"}}, splits=[0.5], axes="xy", out_dir=str(tmp_path / "e"))
    assert os.path.exists(res["files"]["sizing"]) and json.load(open(res["files"]["best"]))["axes"] == "xy"


def test_rated_duty_is_a_magnitude_whichever_fluid_comes_first():
    hot_first = rate("gyroid", 8, 0.8, 40e3, [50, 50], DUTY, "counter")
    cold_first = rate("gyroid", 8, 0.8, 40e3, [50, 50], {"B": DUTY["B"], "A": DUTY["A"], "wall": "alsi10mg"},
                      "counter")
    assert hot_first["duty"] > 0 and np.isclose(cold_first["duty"], hot_first["duty"])
    assert hot_first["dutySign"] == 1 and cold_first["dutySign"] == -1 and cold_first["hotFluid"] == "A"


def test_design_reports_mass_warnings_and_status(tmp_path):
    # cold fluid listed first, a shell under lpbf's minimum (Materialize: 1 mm), a big sideways port (its teardrop)
    spec = {"name": "thin", "core": {"type": "box", "min": [0, 0, 0], "max": [24, 24, 20]}, "layout": "cross",
            "fluids": ["cold", "hot"], "cell": 8, "wall": 1.0, "header": 6, "shell": 0.35, "portDiameter": 10,
            "portLength": 3, "process": "lpbf", "step": 0.3,
            "duty": {"cold": {"fluid": "water", "lPerMin": 1, "tIn": 20},
                     "hot": {"fluid": "water", "lPerMin": 1, "tIn": 80}, "wall": "alsi10mg"}}
    out = tmp_path / "out"
    assert main(["heatx", _write(tmp_path, spec), "--out", str(out), "--no-render", "--no-viewer"]) == 0
    status = json.loads((out / "status.json").read_text())
    assert status["state"] == "done" and status["pid"] == os.getpid()
    rep = json.loads((out / "report.json").read_text())
    r = rep["report"]
    assert np.isclose(r["mass"], r["solidVolume"] * 2.67 / 1000.0) and r["massMaterial"]["name"] == "alsi10mg"
    assert rep["rating"]["duty"] > 0 and rep["rating"]["hotFluid"] == "hot"
    assert r["minOuterWall"] < 0.5
    assert any("outer wall" in w and "1 mm" in w for w in r["warnings"]), r["warnings"]


def test_sideways_teardrop_port_has_no_thin_outer_wall():
    """A 14 mm port run level: its teardrop bore peaks r sqrt(2) over the axis.
    The mouth's check leaves that far out, and the tube closes the bore's
    root where the peak rises over the plenum's leaning top edge."""
    hx = heat_exchanger_from_spec({"core": {"type": "box", "min": [0, 0, 0], "max": [36, 36, 30]},
                                   "layout": "cross", "cell": 8, "wall": 1.0, "header": 8, "portDiameter": 14,
                                   "portLength": 4, "process": "lpbf"})
    hx.step = 0.45
    _, rep = design_heat_exchanger(hx)
    assert rep["isolated"] and all(rep["connected"].values())
    assert rep["minOuterWall"] > 0.9 * hx.shell, (rep["minOuterWall"], rep["minOuterWallAt"])
    assert not any("outer wall" in w for w in rep["warnings"])


def test_design_refused_up_front_when_memory_is_short(tmp_path, monkeypatch, capsys):
    import loadpath.study as study

    monkeypatch.setattr(study, "_available_bytes", lambda: 0.2e9)
    with pytest.raises(ValueError, match="GB is free"):
        heat_exchanger(CUBE, str(tmp_path / "x"), render=False, viewer=False)
    out = tmp_path / "y"
    assert main(["heatx", _write(tmp_path, CUBE), "--out", str(out), "--no-render", "--no-viewer"]) == 2
    status = json.loads((out / "status.json").read_text())
    assert status["state"] == "failed" and "GB is free" in status["error"]
    assert "GB is free" in json.loads(capsys.readouterr().out)["error"]
    assert any("GB" in w for w in check_document(CUBE)["warnings"])
    assert PROCESS_LIMITS["lpbf"]["minWall"] == 1.0


def test_exterior_blend_fillets_the_outside_and_leaves_the_fluids_alone():
    """exteriorBlend smooth-unions the core's, headers' and ports' outer walls:
    material is only added outside the sharp union (in the valleys where they
    meet), so each fluid space and the part's isolation stay as designed."""
    out = {}
    for xb in (0.0, 6.0):
        hx = heat_exchanger_from_spec({**{k: v for k, v in CUBE.items() if k not in ("name", "step")},
                                       "build": "none", "exteriorBlend": xb})
        hx.step = 0.45
        mesh, rep, spaces = design_heat_exchanger(hx, fluid_meshes=True)
        assert rep["isolated"] and all(rep["connected"].values()) and rep["watertight"]
        out[xb] = (rep["solidVolume"], {k: m.volume for k, m in spaces.items()})
    (v0, f0), (v1, f1) = out[0.0], out[6.0]
    assert v1 > v0 * 1.02, (v0, v1)
    for k in f0:
        assert abs(f1[k] - f0[k]) < 0.01 * f0[k], (k, f0[k], f1[k])
