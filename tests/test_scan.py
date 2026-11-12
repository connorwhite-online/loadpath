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


def test_snap_to_standard_sizes():
    from loadpath.scanstd import snap
    # an Artec-grade bar band covers 31.75 (1.25 in tube) and 31.8 (bar clamp): a pick, not a size
    r = snap(31.803, 0.392, "bar")
    assert r["snapped"] is None and {31.75, 31.8} <= {c["diameter"] for c in r["candidates"]}
    # told it is a handlebar, one size is left: the bar's standard and tolerance
    h = snap(31.803, 0.392, "bar", ["handlebar"])
    assert h["snapped"]["diameter"] == 31.8 and h["snapped"]["tolerance"] == 0.1
    # a hole reading 3.42 +- 0.05 is an M3 clearance hole (ISO 273 medium 3.4)
    b = snap(3.42, 0.05, "bore", ["bolt-hole"])
    assert b["snapped"]["name"] == "M3"
    # a 33 mm bar is no handlebar size: nothing snaps, a caliper decides
    assert not snap(33.0, 0.2, "bar", ["handlebar"])["candidates"]


def test_a_card_in_the_frame_sets_the_scale():
    """A splat (no scale) with an ID-1 card propped in the scene: the card is
    found by its aspect and fill, its two sides set the scale, and the bar
    then reads its true size within its band."""
    from loadpath.scanstudy import scan_to_study
    mesh, truth = bar_and_light(card=True)
    P, N, _ = simulate(mesh, ring_views([20, 30, -30], 350, 6), spacing=1.0, sigma_noise=0.05, seed=1)
    sp = os.path.join(TMP, "card.splat.ply")
    k = 0.0123
    write_splat_ply(sp, P, N, spacing=1.0, thickness=0.05, scale_factor=k, seed=1)
    r = scan_to_study(sp)
    assert r["scaledBy"] and r["scaledBy"][0]["object"] == "card", r.get("scaledBy")
    sc = r["report"]["scale"]
    assert abs(sc["factor"] * k - 1) < 3 * sc["relSigma"] and abs(sc["factor"] * k - 1) < 0.005, sc
    bar = max((c for c in r["features"]["cylinders"] if c["kind"] == "bar"), key=lambda c: c["points"])
    assert abs(bar["diameter"] - 31.8) <= bar["diameterBand"], bar


def test_a_brief_picks_the_features_and_asks_for_the_rest():
    from loadpath.scanstudy import scan_to_study
    from loadpath.schema import lint
    from loadpath.tasks import check_study
    path, P, N, truth = _bar_scan()
    out = os.path.join(TMP, "brief")
    r = scan_to_study(path, out, device="artec-eva", units="mm", brief={"mountOn": "bar", "family": "handlebar"})
    # no mass, no shock: asked, nothing drafted
    assert {a["ask"] for a in r["asks"]} >= {"mass", "shock"} and "study" not in r
    r = scan_to_study(path, out, device="artec-eva", units="mm",
                      brief={"mountOn": "bar", "family": "handlebar", "mass": 0.12, "shock": 5})
    assert not [a for a in r["asks"] if a["ask"] in ("mass", "shock", "mount", "hold")], r["asks"]
    f = {x["id"]: x for x in r["features"]["planes"] + r["features"]["cylinders"]}
    bar = f[r["plan"]["support"][0]]
    assert bar["kind"] == "bar" and bar["diameter"] == 31.8 and bar["source"].startswith("standard")
    held = f[r["plan"]["loads"][0]["loads"][0]["feature"]]
    assert held["normal"][2] > 0.99 and abs(held["center"][2] - truth["lightTop"]["z"]) < 0.5, held
    # the mass centre: the light's own centre (its surface centroid)
    assert np.linalg.norm(np.asarray(r["plan"]["centre"]) - truth["light"]["center"]) < 3.0, r["plan"]["centre"]
    assert len(r["study"]["loadCases"]) == 6 and r["study"]["fasteners"]
    c = r["clamp"]
    assert c["ok"] and c["bolts"] == 2 and c["slipMargin"] >= 2.0 and c["gap"] > 0, c
    # a standard bar: no caliper asked
    assert not [a for a in r["asks"] if a["ask"] == "caliper"], r["asks"]
    assert not lint(r["study"]), lint(r["study"])
    chk = check_study(r["study"], out)
    assert chk["ok"], chk


def test_clamp_sizing_is_the_friction_arithmetic():
    from loadpath.scanplan import clamp_plan, WASHERS
    from loadpath.fits import BOLT_HOLES
    bar = {"axis": [1, 0, 0], "p0": [-100, 0, 0], "r": 15.9, "diameter": 31.8, "diameterBand": 0.1}
    # 1 kg, 4 g of shock, its centre 50 mm below the bar: the worst shock is sideways, across the bar
    c = clamp_plan(bar, (90, 110), {"mass": 1.0, "shock": 4.0}, [0, 0, -50], [0, 0, -1], sf=2.0, mu=0.2)
    W = 9.81
    # worst torque: weight + shock both across the lever, |T| <= (W + 4W) x 50 at most
    assert c["torqueDemandNmm"] <= 2 * 5 * W * 50 + 1e-6
    need = c["slipDemandN"] / (0.2 * 8 / np.pi)
    assert abs(c["preloadNeededN"] - max(need, c["openingDemandN"])) < 0.2
    # the preload a bolt gets is the bearing a quarter of PA12's 50 MPa allows under its washer
    A = np.pi / 4 * (WASHERS[c["size"]] ** 2 - BOLT_HOLES[c["size"]][1] ** 2)
    assert abs(c["preloadN"] - 12.5 * A) < 0.2 and c["bolts"] * c["preloadN"] >= c["preloadNeededN"]
    assert abs(c["tighteningTorqueNm"] - 0.2 * c["preloadN"] * int(c["size"][1:]) / 1000) < 1e-3
    # a heavier load needs more bolt
    big = clamp_plan(bar, (90, 110), {"mass": 5.0, "shock": 4.0}, [0, 0, -50], [0, 0, -1], sf=2.0, mu=0.2)
    assert (big.get("ok") is False) or big["bolts"] * big["preloadN"] > c["bolts"] * c["preloadN"]


def test_split_clamp_cuts_body_and_cap():
    import trimesh
    from loadpath.scanplan import split_clamp
    from loadpath.shapes import manifold_from_trimesh, trimesh_from_manifold
    block = trimesh.creation.box(extents=[20, 50, 50])
    hole = trimesh.creation.cylinder(radius=10, height=30, sections=96)
    hole.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [0, 1, 0]))
    part = trimesh_from_manifold(manifold_from_trimesh(block) - manifold_from_trimesh(hole))
    plan = {"split": {"point": [0, 0, 0], "normal": [0, 0, 1]}, "gap": 0.6}
    r = split_clamp(part, plan, TMP, name="blk")
    slab = 0.6 * 20 * (50 - 20)
    assert r["body"]["bodies"] == 1 and r["cap"]["bodies"] == 1
    assert abs(r["body"]["volume"] + r["cap"]["volume"] + slab - part.volume) < 0.01 * part.volume, r


def test_a_part_bolts_to_tapped_holes_in_a_scanned_bay():
    """The bay stand-in (scansim.bay_with_pipe): two bosses with M8 tapped
    holes. Their tops as supports, the holes snapped to tapped, the draft
    bolts through them: an M8 fastener per hole on its fitted axis, its
    seat above the boss top by the grip, and no bare 'kept open' bore."""
    from loadpath.scansim import bay_with_pipe
    from loadpath.scanstd import snap_features
    from loadpath.scanstudy import draft_around
    from loadpath.schema import lint
    from loadpath.tasks import check_study
    mesh, truth = bay_with_pipe()
    views = ring_views([0, -10, 20], 350, 8, elevations=(20.0, 45.0, 70.0))
    views = views[views[:, 1] < 30]
    P, N, meta = simulate(mesh, views, spacing=0.8, sigma_noise=0.05, sigma_sys=0.1, scale=0.0003, seed=1)
    path = write_ply(os.path.join(TMP, "bay.ply"), P, N)
    s = load_scan(path, device="artec-eva", units="mm")
    f = fit_primitives(s)
    taps = [c["id"] for c in f["cylinders"] if c["kind"] == "bore"]
    snap_features(f, {t: ["tapped"] for t in taps})
    assert sorted(c.get("standard", {}).get("name") for c in f["cylinders"] if c["id"] in taps) == ["M8", "M8"]
    tops = [min((p for p in f["planes"] if p["normal"][2] > 0.99),
                key=lambda p: np.linalg.norm(np.asarray(p["center"]) - b["center"]))["id"] for b in truth["bosses"]]
    pipe = min((c for c in f["cylinders"] if c["kind"] == "bar"), key=lambda c: abs(c["p0"][2] - 45))
    out = os.path.join(TMP, "bay-draft")
    r = draft_around(s, f, out, support=tops, loads={pipe["id"]: [0, 0, -75]}, space=truth["region"],
                     process="sls", keep_object=False)
    st = r["study"]
    assert not lint(st), lint(st)
    fs = st["fasteners"]
    assert len(fs) == 2 and {x["size"] for x in fs} == {"M8"}
    for b in truth["bosses"]:
        c = np.asarray(b["center"], float)
        x = min(fs, key=lambda x: np.linalg.norm(np.asarray(x["seat"])[:2] - c[:2]))
        seat = np.asarray(x["seat"])
        # on the true hole's centre to far inside the M8 clearance hole's 0.5 mm slack
        assert np.linalg.norm(seat[:2] - c[:2]) < 0.1, (seat, c)
        # the grip down from the seat lands on the boss top, within the face's band
        assert 0 < seat[2] - x["grip"] - c[2] < 0.5 and np.asarray(x["direction"])[2] < -0.99
    assert not [k for k in st["keepOut"] if k["id"].endswith("-open")]
    assert not [t for t in r["todo"] if "slack" in t], r["todo"]
    chk = check_study(st, out)
    assert chk["ok"], chk


def test_free_space_from_rays_stays_off_the_true_scene():
    """The bay scanned from the front and top, each point with its sensor
    position: every free node is outside the true scene by at least a
    spacing, the space behind the pipe (no ray reached it) is not free, the
    open bay above the bosses is, and the keep-out mesh is closed and holds
    every node that is not free."""
    import trimesh
    from loadpath.freespace import carve, keepout_mesh
    from loadpath.scansim import bay_with_pipe
    mesh, truth = bay_with_pipe()
    views = ring_views([0, -10, 20], 350, 8, elevations=(20.0, 45.0, 70.0))
    views = views[views[:, 1] < 30]
    P, N, meta = simulate(mesh, views, spacing=1.0, sigma_noise=0.05, sigma_sys=0.1, scale=0.0003, seed=2)
    path = write_ply(os.path.join(TMP, "bay-rays.ply"), P, N, meta["viewpoints"])
    s = load_scan(path, device="artec-eva", units="mm")
    assert s.sensors is not None and s.sensors.shape == s.points.shape
    lo, hi = (np.asarray(x, float) for x in truth["region"])
    fs = carve(s, lo, hi, pitch=1.5)
    p = fs["pitch"]
    idx = np.argwhere(fs["free"])
    X = lo + idx * p
    # outside the true solid, clear of its surface
    d = trimesh.proximity.signed_distance(mesh, X[::7])     # > 0 inside
    assert d.max() < -0.5, d.max()
    def node(x):
        return tuple(np.round((np.asarray(x, float) - lo) / p).astype(int))
    pipe = np.asarray(truth["pipe"]["point"], float)
    r = 0.5 * truth["pipe"]["diameter"]
    assert not fs["free"][node(pipe + [0, r + 4, -r - 4])]  # behind and under the pipe: never seen
    assert fs["free"][node([35, 0, 30])]                    # above a boss, open to the sensor
    k = keepout_mesh(fs)
    assert k.is_watertight and k.volume > 0
    nf = np.argwhere(~fs["free"])
    sample = lo + nf[np.random.default_rng(0).choice(len(nf), 2000)] * p
    assert k.contains(sample).all()
    assert k.contains(X[np.random.default_rng(1).choice(len(X), 2000)]).mean() < 0.01


TESTS = [test_calibrate_device_bounds_the_bias, test_free_space_from_rays_stays_off_the_true_scene, test_lzf_decoders_agree, test_readers_round_trip, test_units_are_required,
         test_fits_recover_known_geometry_within_their_bands, test_reconstruction_matches_and_covers_the_truth,
         test_splat_needs_and_takes_a_reference, test_fit_with_a_measured_mate, test_draft_round_the_object_checks,
         test_iphone_scan_says_it_cannot_size_a_bar, test_snap_to_standard_sizes, test_a_card_in_the_frame_sets_the_scale,
         test_a_brief_picks_the_features_and_asks_for_the_rest, test_clamp_sizing_is_the_friction_arithmetic,
         test_split_clamp_cuts_body_and_cap, test_a_part_bolts_to_tapped_holes_in_a_scanned_bay]

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
