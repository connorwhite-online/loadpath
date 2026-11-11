"""
Tests for scans in (scanio.py, scan.py, scansim.py, scanstudy.py) and the
fits on a measured mate (fits.py mateUncertainty). Small synthetic scans of
known geometry; about a minute.

Run: `python3 tests/test_scan.py`
"""
import json
import os
import sys
import tempfile
import time
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from loadpath.scan import apply_reference, fit_primitives, load_scan, reconstruct, scale_features  # noqa: E402
from loadpath.scanio import _lzf_decompress, _lzf_decompress_py, read_scan  # noqa: E402
from loadpath.scansim import bar_and_light, ring_views, simulate, write_ply, write_splat_ply  # noqa: E402

TMP = tempfile.mkdtemp(prefix="lp-scan-")
_CACHE = {}


def _bar_scan(seed=0, sigma_sys=0.1, scale=0.0003, noise=0.05, spacing=1.0):
    key = (seed, sigma_sys, scale, noise, spacing)
    if key not in _CACHE:
        mesh, truth = bar_and_light()
        P, N, meta = simulate(mesh, ring_views([0, 30, -30], 350, 5), spacing=spacing, sigma_noise=noise,
                              sigma_sys=sigma_sys, scale=scale, seed=seed)
        path = os.path.join(TMP, f"bar{seed}-{spacing}.ply")
        write_ply(path, P)
        _CACHE[key] = (path, P, N, truth)
    return _CACHE[key]


def test_lzf_decoders_agree():
    # "abc" as a literal run, then a back-reference copying it (length 3, offset 3)
    src = bytes([2]) + b"abc" + bytes([0x20, 2])
    assert _lzf_decompress_py(src, 6) == b"abcabc"
    assert _lzf_decompress(src, 6) == b"abcabc"
    # an overlapping copy: "a" then 5 more from one back (run-length)
    src = bytes([0]) + b"a" + bytes([0x60, 0])
    assert _lzf_decompress_py(src, 6) == b"aaaaaa"
    assert _lzf_decompress(src, 6) == b"aaaaaa"


def test_readers_round_trip():
    rng = np.random.default_rng(0)
    P = rng.normal(size=(200, 3))
    N = rng.normal(size=(200, 3))
    N /= np.linalg.norm(N, axis=1, keepdims=True)
    write_ply(os.path.join(TMP, "c.ply"), P, N)
    d = read_scan(os.path.join(TMP, "c.ply"))
    assert np.allclose(d["points"], P, atol=1e-5) and np.allclose(d["normals"], N, atol=1e-5)
    np.savetxt(os.path.join(TMP, "c.xyz"), np.hstack([P, N]))
    d = read_scan(os.path.join(TMP, "c.xyz"))
    assert np.allclose(d["points"], P) and "normals" in d
    # PCD ascii with a viewpoint
    with open(os.path.join(TMP, "c.pcd"), "w") as fh:
        fh.write("VERSION 0.7\nFIELDS x y z\nSIZE 4 4 4\nTYPE F F F\nCOUNT 1 1 1\nWIDTH 200\nHEIGHT 1\n"
                 "VIEWPOINT 1 2 3 1 0 0 0\nPOINTS 200\nDATA ascii\n")
        np.savetxt(fh, P)
    d = read_scan(os.path.join(TMP, "c.pcd"))
    assert np.allclose(d["points"], P, atol=1e-6) and np.allclose(d["viewpoint"], [1, 2, 3]) and d["unitsHint"] == "m"
    # 3DGS splat PLY: means, smallest axis = the normal
    write_splat_ply(os.path.join(TMP, "s.ply"), P, N, spacing=0.5, thickness=0.01, floaters=0.0)
    d = read_scan(os.path.join(TMP, "s.ply"))
    assert "splat" in d and np.allclose(d["points"], P, atol=1e-5)
    from loadpath.scan import _splat_points
    keep, nrm, thick = _splat_points(d["splat"], d["points"])
    assert np.all(np.abs(np.einsum("ij,ij->i", nrm, N)) > 0.999), "the splat's thin axis is its normal"
    assert np.all(thick < 0.02)
    # the .splat web format
    rec = np.zeros(200, dtype=[("p", "<f4", 3), ("s", "<f4", 3), ("c", "u1", 4), ("q", "u1", 4)])
    rec["p"] = P
    rec["s"] = 0.1
    rec["c"] = 255
    rec["q"] = [255, 128, 128, 128]
    rec.tofile(os.path.join(TMP, "w.splat"))
    d = read_scan(os.path.join(TMP, "w.splat"))
    assert np.allclose(d["points"], P, atol=1e-5) and d["splat"]["opacity"].min() == 1.0


def test_units_are_required():
    path, *_ = _bar_scan()
    try:
        load_scan(path)
    except ValueError as exc:
        assert "units" in str(exc)
    else:
        raise AssertionError("a PLY with no units must not be read as mm silently")


def test_fits_recover_known_geometry_within_their_bands():
    path, P, N, truth = _bar_scan()
    s = load_scan(path, device="artec-eva", units="mm")
    assert s.report["noise"]["median"] < 0.1, s.report["noise"]
    f = fit_primitives(s)
    bars = [c for c in f["cylinders"] if c["kind"] == "bar" and abs(c["axis"][0]) > 0.99]
    assert bars, f["cylinders"]
    b = bars[0]
    assert abs(b["diameter"] - 31.8) <= b["diameterBand"], (b["diameter"], b["diameterBand"])
    # the band is the device's sigma (0.1 mm) at 95%, two-sided, on the radius: about 0.39 on the diameter
    assert 0.3 < b["diameterBand"] < 0.5, b["diameterBand"]
    tops = [p for p in f["planes"] if p["normal"][2] > 0.99 and abs(p["center"][2] - truth["lightTop"]["z"]) < 3]
    assert tops, f["planes"]
    assert abs(tops[0]["center"][2] - truth["lightTop"]["z"]) <= tops[0]["offsetBand"]


def test_reconstruction_matches_and_covers_the_truth():
    import trimesh
    path, P, N, truth = _bar_scan()
    s = load_scan(path, device="artec-eva", units="mm")
    m, info = reconstruct(s)
    mesh, _ = bar_and_light()
    assert m.is_watertight
    assert abs(m.volume - mesh.volume) / mesh.volume < 0.02, (m.volume, mesh.volume)
    T, _ = trimesh.sample.sample_surface(mesh, 3000, seed=1)
    out = -trimesh.proximity.ProximityQuery(m).signed_distance(T)
    U = s.uncertainty(P=T)
    assert np.mean(out <= U) > 0.95, np.mean(out <= U)


def test_splat_needs_and_takes_a_reference():
    path, P, N, truth = _bar_scan()
    sp = os.path.join(TMP, "bar-splat.ply")
    write_splat_ply(sp, P, N, spacing=1.0, thickness=0.05, scale_factor=0.0123, seed=2)
    s = load_scan(sp)
    assert not s.scaled and any("scale" in t for t in s.report["todo"])
    f = fit_primitives(s)
    bar = max((c for c in f["cylinders"] if c["kind"] == "bar"), key=lambda c: c["points"])
    s2 = apply_reference(s, [{"feature": bar["id"], "diameter": 31.8, "tolerance": 0.05}], f)
    assert s2.scaled
    f2 = scale_features(f, s2)
    b2 = next(c for c in f2["cylinders"] if c["id"] == bar["id"])
    assert abs(b2["diameter"] - 31.8) < 0.05
    # the light's top, now in mm, against the truth (the scale fixed by the bar alone)
    tops = [p for p in f2["planes"] if p["normal"][2] > 0.99]
    top = min(tops, key=lambda p: abs(p["center"][2] - truth["lightTop"]["z"]))
    assert abs(top["center"][2] - truth["lightTop"]["z"]) < 0.3, top["center"]


def test_fit_with_a_measured_mate():
    from loadpath.fits import expand_fits
    base = {"material": "pa12", "process": "sls", "keepOut": [{"id": "bar", "shape": {
        "type": "cylinder", "p0": [0, 0, 0], "p1": [10, 0, 0], "r": 15.9}, "fit": "clearance", "mate": 31.8}]}
    plain = expand_fits(json.loads(json.dumps(base)))["_fits"][0]
    k = json.loads(json.dumps(base))
    k["keepOut"][0]["mateUncertainty"] = 0.02
    tight = expand_fits(k)["_fits"][0]
    # the window shrinks by the band from each side
    assert abs(tight["window"][0] - plain["window"][0] - 0.02) < 1e-6
    assert abs(plain["window"][1] - tight["window"][1] - 0.02) < 1e-6
    k["keepOut"][0]["mateUncertainty"] = 5.0
    wide = expand_fits(k)["_fits"][0]
    # wider than the window: drawn for the largest mate, said so
    assert wide["window"][0] >= 31.8 + 5.0 and "largest mate" in wide["mateNote"]
    k["keepOut"][0]["fit"] = "press"
    press = expand_fits(k)["_fits"][0]
    assert press["verdictAsBuilt"] == "measure" and "no hole holds" in press["mateNote"]


def test_draft_round_the_object_checks():
    from loadpath.scanstudy import draft_around
    from loadpath.schema import lint
    from loadpath.tasks import check_study
    path, P, N, truth = _bar_scan()
    s = load_scan(path, device="artec-eva", units="mm")
    f = fit_primitives(s)
    bar = next(c for c in f["cylinders"] if c["kind"] == "bar")
    top = min((p for p in f["planes"] if p["normal"][2] > 0.99),
              key=lambda p: abs(p["center"][2] - truth["lightTop"]["z"]))
    out = os.path.join(TMP, "draft")
    r = draft_around(s, f, out, support=bar["id"], loads={top["id"]: [0, 0, -60]}, process="sls")
    st = r["study"]
    assert not lint(st), lint(st)
    ko = {k["id"]: k for k in st["keepOut"]}
    assert ko[bar["id"]]["fit"] == "slip" and ko[bar["id"]]["mateUncertainty"] > 0
    assert "object" in ko
    # the clearance is the scan's and the process's, not a fixed margin: Artec + SLS about 0.3-0.6 mm
    c = r["clearance"]["clearance"]["median"]
    assert 0.25 < c < 0.7, r["clearance"]
    chk = check_study(st, out)
    assert chk["ok"], chk


def test_iphone_scan_says_it_cannot_size_a_bar():
    """Phone lidar's +-1 cm accuracy (Luetzenburg et al. 2021) makes a bar's
    band wider than any fit's window: the study says measure it, and a
    caliper-measured diameter (known) replaces the scan's."""
    from loadpath.scanstudy import known_dimensions
    path, P, N, truth = _bar_scan(seed=3, sigma_sys=5.0, scale=0.01, noise=1.5, spacing=2.0)
    s = load_scan(path, device="iphone-lidar", units="mm")
    f = fit_primitives(s)
    bar = max((c for c in f["cylinders"] if c["kind"] == "bar"), key=lambda c: c["points"])
    assert bar["diameterBand"] > 10, bar["diameterBand"]
    f2 = known_dimensions(f, {bar["id"]: {"diameter": 31.8, "tolerance": 0.05}})
    b2 = next(c for c in f2["cylinders"] if c["id"] == bar["id"])
    assert b2["diameter"] == 31.8 and b2["diameterBand"] < 0.06 and b2["source"].startswith("known")


def test_calibrate_device_bounds_the_bias():
    from loadpath.scan import calibrate_device
    path, P, N, truth = _bar_scan(seed=5, sigma_sys=1.0, scale=0.0, noise=0.05)
    s = load_scan(path, device="measured", units="mm")
    f = fit_primitives(s)
    bar = max((c for c in f["cylinders"] if c["kind"] == "bar"), key=lambda c: c["points"])
    top = min((p for p in f["planes"] if p["normal"][2] > 0.99), key=lambda p: abs(p["center"][2] - truth["lightTop"]["z"]))
    bot = min((p for p in f["planes"] if p["normal"][2] < -0.99), key=lambda p: abs(p["center"][2] - truth["lightTop"]["z"] + 30))
    cal = calibrate_device(s, f, [{"feature": bar["id"], "diameter": 31.8}, {"feature": top["id"], "to": bot["id"], "distance": 30.0}])
    # a 1 mm common range bias: the measured bound is a real, positive number and an upper bound
    assert cal["n"] == 2 and cal["sigma"] >= cal["sigmaEstimate"] > 0.05, cal


TESTS = [test_calibrate_device_bounds_the_bias, test_lzf_decoders_agree, test_readers_round_trip, test_units_are_required,
         test_fits_recover_known_geometry_within_their_bands, test_reconstruction_matches_and_covers_the_truth,
         test_splat_needs_and_takes_a_reference, test_fit_with_a_measured_mate, test_draft_round_the_object_checks,
         test_iphone_scan_says_it_cannot_size_a_bar]

if __name__ == "__main__":
    failed = 0
    for t in TESTS:
        t0 = time.time()
        try:
            t()
            print(f"ok   {t.__name__} ({time.time() - t0:.1f}s)")
        except Exception:
            failed += 1
            print(f"FAIL {t.__name__}")
            traceback.print_exc()
    sys.exit(1 if failed else 0)
