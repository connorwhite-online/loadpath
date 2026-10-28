"""
Tests for loadpath (topology optimization, MTR-352). Small grids only: the
whole file runs in well under a minute. The slow, realistic parts live in
the eval suite (`python -m loadpath.evals.run`), which scores delivered
meshes against engineering requirements rather than unit contracts.

Run: `python3 cad-runner/tests/test_loadpath.py` (numpy, scipy, pyamg,
scikit-image, trimesh, manifold3d — the sidecar image's stack).
"""
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import trimesh

from loadpath.domain import build_domain
from loadpath.fem import Solver, Stiffness
from loadpath.reconstruct import reconstruct
from loadpath.shapes import Box, Cylinder, shape_from_spec
from loadpath.simp import SimpOptions, filter_radius, optimize
from loadpath.spec import study_from_dict
from loadpath.study import run_study
from loadpath.verify import verify_part, voxelize

BEAM = {
    "designSpace": {"type": "box", "min": [0, 0, 0], "max": [100, 10, 10]},
    "supports": [{"shape": {"type": "box", "min": [-1, -1, -1], "max": [0.01, 11, 11]}}],
    "loadCases": [{"name": "tip", "loads": [
        {"shape": {"type": "box", "min": [99.99, -1, -1], "max": [101, 11, 11]}, "force": [0, 0, -10]}]}],
    "material": {"E": 2000, "nu": 0.3, "strength": 50, "density": 1.0},
    "goal": {"volumeFraction": 0.5},
    "resolution": {"pitch": 2.0},
}

# A small plate with a bore near the free end: exercises preserve, keep-out
# and the exact booleans without a long solve.
LUG = {
    "name": "lug",
    "designSpace": {"type": "box", "min": [0, -15, -4], "max": [60, 15, 4]},
    "preserve": [{"id": "eye", "shape": {"type": "cylinder", "center": [48, 0, 0], "axis": "z",
                                         "length": 8, "r": 9}}],
    "keepOut": [{"id": "pin", "shape": {"type": "cylinder", "center": [48, 0, 0], "axis": "z",
                                        "length": 12, "r": 4}}],
    "supports": [{"shape": {"type": "box", "min": [-1, -16, -5], "max": [0.01, 16, 5]}}],
    "loadCases": [{"name": "pull", "loads": [{"region": "pin", "force": [0, -80, 0]}]}],
    "material": "pa12",
    "goal": {"volumeFraction": 0.35},
    "symmetry": ["z"],
    "resolution": {"pitch": 2.0},
    "maxIterations": 60,
}


def test_shapes_sdf_and_exact_volume():
    b = Box([0, 0, 0], [10, 20, 30])
    assert abs(b.sdf(np.array([[5, 10, 15]]))[0] + 5) < 1e-9
    assert abs(b.sdf(np.array([[15, 10, 15]]))[0] - 5) < 1e-9
    c = Cylinder([0, 0, 0], [0, 0, 10], 5)
    assert abs(c.sdf(np.array([[0, 0, 5]]))[0] + 5) < 1e-9
    assert abs(c.sdf(np.array([[8, 0, 5]]))[0] - 3) < 1e-9
    tube = shape_from_spec({"type": "cylinder", "center": [0, 0, 0], "axis": "x", "length": 10,
                            "r": 6, "rInner": 4})
    assert tube.contains(np.array([[0, 5, 0]]))[0]
    assert not tube.contains(np.array([[0, 0, 0]]))[0]
    vol = tube.to_manifold().volume()
    exact = np.pi * (36 - 16) * 10
    assert abs(vol - exact) / exact < 0.002, vol
    lo, hi = tube.bounds()
    assert np.allclose(lo, [-5, -6, -6]) and np.allclose(hi, [5, 6, 6])


def test_spec_errors_name_the_field():
    cases = [
        ({**BEAM, "supports": []}, "support"),
        ({**BEAM, "loadCases": [{"loads": [{"region": "nope", "force": [0, 0, 1]}]}]}, None),
        ({**BEAM, "loadCases": [{"loads": [{"shape": BEAM["supports"][0]["shape"], "force": [0, 0, 0]}]}]},
         "force"),
        ({**BEAM, "goal": {"volumeFraction": 1.5}}, "volumeFraction"),
        ({**BEAM, "material": "unobtainium"}, "material"),
    ]
    for raw, word in cases:
        try:
            s = study_from_dict(raw)
            build_domain(s)
        except ValueError as exc:
            if word:
                assert word in str(exc), str(exc)
            continue
        raise AssertionError(f"expected a ValueError for {raw}")


def test_fem_matches_beam_theory():
    s = study_from_dict({**BEAM, "resolution": {"pitch": 1.25}})
    d = build_domain(s)
    st = Stiffness(d.mesh, 0.3, d.fixed)
    u = Solver(d.mesh, rtol=1e-8).solve(st.assemble(np.full(d.n_elem, 2000.0)), d.loads[0])
    tip = u[2::3][d.load_nodes].mean()
    analytic = -10 * 100 ** 3 / (3 * 2000 * 10 * 10 ** 3 / 12)   # -2.0 mm
    assert abs(tip - analytic) / abs(analytic) < 0.03, tip


def test_simp_keeps_volume_and_symmetry():
    s = study_from_dict(LUG)
    d = build_domain(s)
    res = optimize(d, 0.39, SimpOptions(volume_fraction=0.35, filter_radius=filter_radius(d, None),
                                        max_iterations=60))
    design = ~d.preserve
    assert abs(res.xphys[design].mean() - 0.35) < 0.01, res.xphys[design].mean()
    assert res.history[-1]["compliance"] < res.history[0]["compliance"] * 0.5
    m = d.mirror["z"]
    has = m >= 0
    assert np.abs(res.xphys[has] - res.xphys[m[has]]).max() < 1e-6
    grey = ((res.xphys > 0.1) & (res.xphys < 0.9))[design].mean()
    assert grey < 0.1, grey


def test_reconstruct_keeps_bores_exact_and_preserve_whole():
    s = study_from_dict(LUG)
    d = build_domain(s)
    res = optimize(d, 0.39, SimpOptions(volume_fraction=0.35, filter_radius=filter_radius(d, None),
                                        max_iterations=60))
    mesh, info = reconstruct(s, d, res.xphys)
    assert mesh.is_watertight and info["bodies"] == 1
    rep = verify_part(s, mesh, contact_band=d.pitch)
    assert rep["keepOutIntrusion"] < 1e-3, rep["keepOutIntrusion"]
    assert rep["preserveMissing"] < 1e-3, rep["preserveMissing"]
    # The bore is exactly r=4: a 3.98 pin fits with no material inside it,
    # and the ring 4.02..4.5 is solid all the way round.
    from loadpath.shapes import manifold_from_trimesh
    part = manifold_from_trimesh(mesh)
    pin = Cylinder([48, 0, -3.9], [48, 0, 3.9], 3.98).to_manifold()
    assert (part ^ pin).volume() < 1e-6
    ring = shape_from_spec({"type": "cylinder", "center": [48, 0, 0], "axis": "z", "length": 7.8,
                            "r": 4.5, "rInner": 4.02}).to_manifold()
    assert abs((part ^ ring).volume() - ring.volume()) < 1e-3 * ring.volume()
    assert rep["safetyFactor"] > 1 and rep["mass"] > 0


def test_verify_bar_stress_matches_beam_theory():
    s = study_from_dict(BEAM)
    bar = trimesh.creation.box(extents=[100, 10, 10])
    bar.apply_translation([50, 5, 5])
    rep = verify_part(s, bar, max_cells=40_000)
    # Root bending stress M c / I = (10 N * 100 mm) * 5 / (10^4 / 12) = 6 MPa;
    # the cells next to the clamp are excluded, so expect slightly under.
    peak = rep["cases"][0]["peakStress"]
    assert 4.5 < peak < 6.6, peak
    assert abs(rep["cases"][0]["deflection"] - 2.0) < 0.15, rep["cases"][0]["deflection"]
    assert abs(rep["mass"] - 10.0) < 1e-6
    assert rep["bodies"] == 1 and rep["keepOutIntrusion"] == 0.0


def test_verify_hole_stress_concentration():
    # Plate with a hole in tension: Peterson's net-section Kt = 2.43 for
    # d/W = 0.25, so the peak is ~81 MPa against a 33.3 MPa net stress. Voxel
    # stress at the hole is the hard case for this method; the submodel
    # refinement plus nodal averaging has to land near it, and the reported
    # safety factor has to carry the allowance.
    from loadpath.shapes import trimesh_from_manifold
    from loadpath.verify import STRESS_ALLOWANCE
    plate = trimesh_from_manifold(Box([0, -20, -2], [120, 20, 2]).to_manifold()
                                  - Cylinder([60, 0, -5], [60, 0, 5], 5).to_manifold())
    s = study_from_dict({
        "designSpace": {"type": "box", "min": [0, -20, -2], "max": [120, 20, 2]},
        "supports": [{"shape": {"type": "box", "min": [-1, -21, -3], "max": [0.01, 21, 3]}}],
        "loadCases": [{"name": "pull", "loads": [
            {"shape": {"type": "box", "min": [119.99, -21, -3], "max": [121, 21, 3]}, "force": [4000, 0, 0]}]}],
        "material": {"E": 70000, "nu": 0.33, "strength": 250, "density": 2.7},
        "goal": {"volumeFraction": 0.5},
    })
    rep = verify_part(s, plate, max_cells=10 ** 7, min_pitch=0.7, contact_band=3.0)
    c = rep["cases"][0]
    assert 70 < c["peakStress"] < 100, c["peakStress"]
    assert abs(c["peakAt"][0] - 60) < 2.5 and 3.5 < abs(c["peakAt"][1]) < 6.5, c["peakAt"]
    assert abs(rep["safetyFactor"] - 250 / (c["peakStress"] * STRESS_ALLOWANCE)) < 1e-9


def test_am_filter_gradient_and_behavior():
    from loadpath.am import AMFilter
    rng = np.random.default_rng(0)
    x = rng.random((5, 4, 6))
    pin = np.zeros(x.shape, bool)
    pin[2, 1, 3] = True
    f = AMFilter(x.shape, (1, -1), pin)
    w = rng.random(x.shape)
    y = f.forward(x)
    g = f.backward(w)
    num = np.zeros_like(x)
    for idx in [(0, 0, 0), (2, 3, 4), (4, 2, 1), (1, 1, 5)]:
        xp = x.copy()
        xp[idx] += 1e-6
        num[idx] = ((f.forward(xp) * w).sum() - (y * w).sum()) / 1e-6
        assert abs(num[idx] - g[idx]) < 1e-4 * max(1.0, abs(g[idx])), (idx, num[idx], g[idx])
    # A floating slab is not printable; a column from the plate is.
    field = np.zeros((20, 3, 20))
    field[:, 1, 15] = 1.0
    field[2:4, 1, 0:16] = 1.0
    out = AMFilter(field.shape, (2, 1)).forward(field)
    assert out[15, 1, 15] < 0.05 and out[3, 1, 10] > 0.95


def test_voxelize_volume():
    m = trimesh.creation.cylinder(radius=10, height=20, sections=96)
    solid, _ = voxelize(m, 0.5)
    vol = solid.sum() * 0.125
    assert abs(vol - m.volume) / m.volume < 0.03, (vol, m.volume)


def test_unreachable_load_is_refused_helpfully():
    raw = {**BEAM, "loadCases": [{"loads": [
        {"shape": {"type": "sphere", "center": [500, 0, 0], "r": 1}, "force": [0, 0, -1]}]}]}
    try:
        build_domain(study_from_dict(raw))
    except ValueError as exc:
        assert "touches no material" in str(exc), str(exc)
        return
    raise AssertionError("expected refusal")


def test_run_study_end_to_end():
    s = study_from_dict({**LUG, "goal": {"volumeFraction": 0.35, "outcomes": [0.5]}, "maxIterations": 40})
    result = run_study(s)
    assert len(result.outcomes) == 2
    a, b = result.outcomes
    assert b.report["mass"] > a.report["mass"]
    assert b.report["safetyFactor"] > a.report["safetyFactor"] * 0.9
    out = result.to_dict()
    assert out["outcomes"][0]["bodies"] == 1


def test_topopt_endpoints_round_trip():
    import base64
    import os
    import tempfile

    os.environ["CAD_RUNNER_ALLOW_NO_AUTH"] = "true"
    os.environ["TOPOPT_DIR"] = tempfile.mkdtemp(prefix="topopt-test-")
    import importlib

    import topopt_jobs
    importlib.reload(topopt_jobs)
    import app as app_mod
    from fastapi.testclient import TestClient

    client = TestClient(app_mod.app)
    # A path-based mesh would read the sidecar's disk: refused.
    bad = {**LUG, "keepOut": [{"id": "x", "shape": {"type": "mesh", "path": "/etc/passwd"}}]}
    r = client.post("/topopt", json={"study": bad})
    assert r.status_code == 400 and "stlBase64" in r.json()["detail"], r.text
    r = client.post("/topopt", json={"study": {**LUG, "supports": []}})
    assert r.status_code == 400 and "support" in r.json()["detail"], r.text

    # Inline mesh keep-out (a pin as an STL) + a quick solve.
    pin = trimesh.creation.cylinder(radius=4, height=12, sections=48)
    pin.apply_translation([48, 0, 0])
    stl = base64.b64encode(pin.export(file_type="stl")).decode()
    study = {**LUG, "keepOut": [{"id": "pin", "shape": {"type": "mesh", "stlBase64": stl}}],
             "maxIterations": 25}
    r = client.post("/topopt", json={"study": study})
    assert r.status_code == 200, r.text
    jid = r.json()["id"]
    r2 = client.post("/topopt", json={"study": study})
    assert r2.status_code == 429, r2.text          # one job at a time by default
    deadline = time.time() + 300
    while True:
        st = client.get(f"/topopt/{jid}").json()
        if st["status"] != "running" or time.time() > deadline:
            break
        time.sleep(1.0)
    assert st["status"] == "done", st
    out = st["result"]["outcomes"][0]
    assert out["bodies"] == 1 and out["keepOutIntrusion"] < 0.5, out
    f = client.get(f"/topopt/{jid}/files/{out['stl']}")
    assert f.status_code == 200 and len(f.content) > 1000
    assert client.get(f"/topopt/{jid}/files/..%2Fstudy.json").status_code == 404
    assert client.get("/topopt/" + "0" * 32).status_code == 404


TESTS = [
    test_shapes_sdf_and_exact_volume,
    test_spec_errors_name_the_field,
    test_fem_matches_beam_theory,
    test_simp_keeps_volume_and_symmetry,
    test_reconstruct_keeps_bores_exact_and_preserve_whole,
    test_verify_bar_stress_matches_beam_theory,
    test_verify_hole_stress_concentration,
    test_am_filter_gradient_and_behavior,
    test_voxelize_volume,
    test_unreachable_load_is_refused_helpfully,
    test_run_study_end_to_end,
    test_topopt_endpoints_round_trip,
]


def main():
    failures = 0
    for t in TESTS:
        t0 = time.time()
        try:
            t()
            print(f"PASS {t.__name__} ({time.time() - t0:.1f}s)")
        except Exception:  # noqa: BLE001
            failures += 1
            print(f"FAIL {t.__name__}")
            traceback.print_exc()
    if failures:
        print(f"\n{failures}/{len(TESTS)} FAILED")
        sys.exit(1)
    print(f"\nall {len(TESTS)} passed")


if __name__ == "__main__":
    main()
