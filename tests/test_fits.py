"""
Fits (fits.py): the hole drawn for each process, the windows, the delivered
hole measured off the mesh, and the report rows. Fast: one tiny run.

Run: `python3 tests/test_fits.py`.
"""
import copy
import json
import os
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from loadpath.fasteners import expand_fasteners
from loadpath.fits import expand_fits, fit_window, measure_bore
from loadpath.schema import lint
from loadpath.spec import study_from_dict

LUG = {
    "name": "lug",
    "designSpace": {"type": "box", "min": [0, -15, -4], "max": [60, 15, 4]},
    "preserve": [{"id": "eye", "shape": {"type": "cylinder", "center": [48, 0, 0], "axis": "z", "length": 8,
                                         "r": 11, "rInner": 4}}],
    "keepOut": [{"id": "pin", "shape": {"type": "cylinder", "center": [48, 0, 0], "axis": "z", "length": 12, "r": 4},
                 "fit": "slip"}],
    "supports": [{"shape": {"type": "box", "min": [-1, -16, -5], "max": [0.01, 16, 5]}}],
    "loadCases": [{"name": "pull", "loads": [{"region": "pin", "force": [0, -80, 0]}]}],
    "material": "petg",
    "process": "fdm",
    "goal": {"volumeFraction": 0.4},
    "minMember": 3.0,
    "maxIterations": 25,
    "resolution": {"pitch": 2.0},
}


def _row(d, rid="pin"):
    return next(r for r in expand_fits(d)["_fits"] if r["id"] == rid)


def test_windows():
    # ISO 286 at 22 mm (the 18-30 step): H7/p6 is -35..-1 um, H7/g6 +7..+41, H8/f7 +20..+74
    w = fit_window("press", 22.0, "metal")
    assert np.allclose(w["clearance"], [-0.035, -0.001], atol=0.0015), w
    w = fit_window("slip", 22.0, "metal")
    assert np.allclose(w["clearance"], [0.007, 0.041], atol=0.0015), w
    w = fit_window("clearance", 22.0, "metal")
    assert np.allclose(w["clearance"], [0.020, 0.074], atol=0.002), w
    # plastic windows are wider and in order: press < transition < slip < clearance
    ws = [fit_window(f, 22.0, "plastic") for f in ("press", "transition", "slip", "clearance")]
    assert all(a["hi"] <= b["lo"] + 1e-9 for a, b in zip(ws, ws[1:])), ws
    assert fit_window("normal", 5.0, size="M5")["target"] == 5.5


def test_compensation_per_process():
    d = copy.deepcopy(LUG)
    r = _row(d)
    # FDM along the build: drawn the slip target plus 0.2 mm
    assert r["orientation"] == "along" and abs(r["designed"] - (r["target"] + 0.2)) < 1e-3, r
    # across the layers the top of the hole sags: drawn bigger still
    d["keepOut"][0]["shape"]["axis"] = "x"
    assert _row(d)["designed"] > r["designed"] + 0.1
    # resin closes holes less than FDM
    d = copy.deepcopy(LUG)
    d["process"] = "sla"
    assert _row(d)["designed"] < r["designed"]
    # a measured hole bias replaces the default
    d["process"] = {"type": "fdm", "holeBias": -0.1}
    assert abs(_row(d)["designed"] - (r["target"] + 0.1)) < 1e-3
    # milled: drawn at the window, bored to it
    d = copy.deepcopy(LUG)
    d.pop("process")
    d["material"] = "al6061"
    d["milling"] = {"directions": ["+z", "-z"]}
    rm = _row(d)
    assert rm["finish"] == "machine" and rm["standard"] == "H7/g6", rm


def test_press_fit_in_lpbf_is_reamed():
    d = copy.deepcopy(LUG)
    d["process"] = "lpbf"
    d["material"] = "alsi10mg"
    d["keepOut"][0]["fit"] = "press"
    r = _row(d)
    # the printer's scatter (+-0.1) dwarfs H7: drawn undersize, reamed to the window
    assert r["finish"] == "ream" and r["designed"] < r["window"][0] - 0.15, r


def test_resizes_the_ring_round_the_bore_and_parses_once():
    st = study_from_dict(copy.deepcopy(LUG))
    r_pin = st.keep_out[0].shape.r
    assert abs(st.preserve[0].shape.r_inner - r_pin) < 1e-9
    # a study re-parsed from its raw (orientation, explore) is not compensated twice
    st2 = study_from_dict(st.raw)
    assert st2.keep_out[0].shape.r == r_pin and len(st2.fits) == 1


def test_no_process_leaves_plain_bores_alone():
    d = copy.deepcopy(LUG)
    d.pop("process")
    d["keepOut"][0].pop("fit")
    st = study_from_dict(d)
    assert st.keep_out[0].shape.r == 4 and not st.fits
    # with a process every bore is held at its drawn size
    d["process"] = "mjf"
    st = study_from_dict(d)
    assert st.fits[0]["fit"] == "nominal" and st.keep_out[0].shape.r > 4


def test_fastener_holes():
    d = {k: v for k, v in LUG.items() if k not in ("keepOut",)}
    d["preserve"] = []
    d["fasteners"] = [{"id": "m4", "size": "M4", "seat": [48, 0, 4], "direction": "-z", "grip": 8},
                      {"id": "m3", "size": "M3", "seat": [30, 0, 4], "direction": "-z", "grip": 8, "fit": "close"}]
    out = expand_fits(expand_fasteners(d))
    rows = {r["id"]: r for r in out["_fits"]}
    assert set(rows) == {"m4-bore", "m3-bore"}, rows          # the head and driver path is no fit
    assert rows["m4-bore"]["target"] == 4.5 and rows["m3-bore"]["target"] == 3.2
    assert rows["m4-bore"]["designed"] > 4.5                  # FDM prints it small
    seat = next(p for p in out["preserve"] if p["id"] == "m4-seat")
    assert abs(seat["shape"]["rInner"] - 0.5 * rows["m4-bore"]["designed"]) < 1e-4


def test_measure_bore():
    import trimesh

    block = trimesh.creation.box((20, 20, 10))
    for sections, want in ((128, 8.0), (12, 8.0 * np.cos(np.pi / 12))):
        hole = trimesh.creation.cylinder(radius=4, height=14, sections=sections)
        part = block.difference(hole)
        m = measure_bore(part, [0, 0, -7], [0, 0, 7], 4.0)
        # a 12-sided hole passes only the pin its flats allow
        assert m["walled"] and abs(m["measured"] - want) < 0.01, (sections, m)
    m = measure_bore(block, [30, 0, -5], [30, 0, 5], 4.0)
    assert not m["walled"]


def test_lint():
    assert not lint(LUG, "study")
    bad = copy.deepcopy(LUG)
    bad["keepOut"][0]["fit"] = "snug"
    assert any(i["level"] == "error" for i in lint(bad, "study"))


def test_run_reports_fits():
    with tempfile.TemporaryDirectory() as tmp:
        p = os.path.join(tmp, "lug.json")
        with open(p, "w") as fh:
            json.dump(LUG, fh)
        subprocess.run([sys.executable, "-m", "loadpath", "run", p, "--out", os.path.join(tmp, "out")],
                       check=True, capture_output=True, cwd=ROOT)
        rep = json.load(open(os.path.join(tmp, "out", "report.json")))
        f = rep["deliver"]["fits"][0]
        assert f["id"] == "pin" and f["verdict"] in ("holds", "tune"), f
        # the delivered hole is the drawn one, within the tessellation's chord
        assert abs(f["measured"] - f["designed"]) < 0.02, f
        lo, hi = f["window"]
        mid = 0.5 * sum(f["asBuilt"])
        assert lo <= mid <= hi, f


if __name__ == "__main__":
    failed = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print("ok  ", name)
            except Exception:
                failed += 1
                print("FAIL", name)
                traceback.print_exc()
    sys.exit(1 if failed else 0)
