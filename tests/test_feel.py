"""
Feel targets inside a fixed exterior (loadpath/feel.py): grip squeeze,
drop cushioning, vibration isolation. Small problems; about a minute.

Run: `python3 tests/test_feel.py`
"""
import sys
import tempfile
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from loadpath import feel  # noqa: E402
from loadpath.domain import build_domain  # noqa: E402
from loadpath.schema import lint  # noqa: E402
from loadpath.spec import study_from_dict  # noqa: E402
from loadpath.switch import _thin_skin  # noqa: E402


def body(**over):
    """A 16 x 16 x 30 mm TPU box body: a 1 mm skin open underneath, a 6 mm
    'capsule' on a mount near the top, its path out through the bottom."""
    grip = {"type": "box", "min": [-9, -9, 4], "max": [9, 9, 18]}
    doc = {
        "name": "body",
        "designSpace": {"type": "box", "min": [-8, -8, 0], "max": [8, 8, 30]},
        "skin": {"thickness": 1.0, "open": ["-z"]},
        "preserve": [{"id": "mount", "shape": {"type": "box", "min": [-4.5, -4.5, 21], "max": [4.5, 4.5, 26]}}],
        "keepOut": [{"id": "capsule", "bore": False, "insertFrom": "-z",
                     "shape": {"type": "box", "min": [-3, -3, 22], "max": [3, 3, 26.01]}}],
        "material": "tpu88a",
        "minMember": 2.0,
        "goal": {"volumeFraction": 0.3},
        "resolution": {"pitch": 2.0},
        "feel": {"targets": [
            {"kind": "squeeze", "name": "grip", "hold": {"face": "-x", "shape": grip},
             "press": {"face": "+x", "shape": grip}, "direction": "-x", "force": 10, "travel": [0.2, 1.0]},
            {"kind": "drop", "name": "drop", "lands": {"face": "+z"}, "mass": {"region": "mount"},
             "kg": 0.004, "height": 1.0, "maxG": 1500, "maxTravel": 3.0},
            {"kind": "isolate", "name": "noise", "hold": {"region": "skin", "shape": grip},
             "mount": {"region": "mount"}, "kg": 0.004, "band": 500, "maxTransmissibility": 0.5,
             "maxTravel": 0.5, "shockG": 20}],
            "iterations": 30},
    }
    doc["feel"].update(over)
    return doc


def test_lints_clean_without_supports_or_loads():
    assert not lint(body(), "study"), lint(body(), "study")
    bad = body(targets=[{"kind": "squish"}])
    assert lint(bad, "study")
    d = body()
    del d["feel"]
    assert any(i["path"] == "$.supports" for i in lint(d, "study"))


def test_windows():
    # drop: 4 g, 1 m: E = 39.2 N mm; 1500 g peak -> kmax = (58.8)^2 / 78.5
    kmin, kmax = feel.drop_window(0.004, 1.0, 1500, 3.0)
    e = 0.004 * feel.G0 * 1000
    assert abs(kmax - (1500 * 0.004 * feel.G0) ** 2 / (2 * e)) < 1e-9
    assert abs(kmin - 2 * e / 9) < 1e-9
    # the response at either end lands on the limit it came from
    r = feel.drop_response(kmax, 0.004, 1.0)
    assert abs(r["peakG"] - 1500) < 1e-6
    assert abs(feel.drop_response(kmin, 0.004, 1.0)["travel"] - 3.0) < 1e-9
    # isolation: at f_n = band / r the transmissibility is exactly the target
    from loadpath.modal import transmissibility
    eta = 0.12
    _, kmax = feel.isolation_window(0.004, 500, 0.5, eta)
    fn = np.sqrt(kmax * 1000 / 0.004) / (2 * np.pi)
    assert abs(transmissibility(500, fn, eta) - 0.5) < 1e-9
    # conflicting drop numbers say how much room the fall needs
    st = study_from_dict(body(targets=[{"kind": "drop", "lands": {"face": "+z"}, "mass": {"region": "mount"},
                                        "kg": 0.004, "height": 1.0, "maxG": 500, "maxTravel": 2.0}]))
    dom = build_domain(st, 2.0)
    try:
        feel.feel_from_study(st, dom)
        raise AssertionError("expected the conflict to be reported")
    except ValueError as exc:
        assert "4.00 mm of travel" in str(exc), exc


def test_locations():
    st = study_from_dict(body())
    dom = build_domain(st, 2.0)
    ts = feel.feel_from_study(st, dom)
    xyz = dom.mesh.node_xyz
    grip, drop, noise = ts
    # a face is the exterior on that side, limited to the shape
    assert np.allclose(xyz[grip.ground, 0], -8) and np.allclose(xyz[grip.load, 0], 8)
    assert xyz[grip.ground, 2].min() >= 4 - 1e-9 and xyz[grip.ground, 2].max() <= 18 + 1e-9
    assert np.allclose(xyz[drop.ground, 2], 30)
    assert drop.directions == [(2, 1)]            # falls toward the face it lands on
    # the skin where the hand is: on the walls, none of the inside
    g = xyz[noise.ground]
    assert (np.abs(g[:, :2]).max(axis=1) >= 6.99).all(), g
    assert len(noise.directions) == 3 and noise.key != drop.key


def test_sensitivities_match_finite_differences():
    st = study_from_dict(body())
    dom, floor = _thin_skin(st, build_domain(st, 2.0))
    prob = feel.FeelProblem(st, dom, st.elastic(design=True), float(st.material.E))
    rng = np.random.default_rng(3)
    x = np.clip(0.3 + 0.5 * rng.random(dom.n_elem), 0, 1)
    x[dom.preserve] = 1.0
    x = np.maximum(x, floor)
    base = prob.evaluate(x)
    assert len(base) == prob.m
    scale = [np.abs(g).max() for _, g, _ in base]
    for e in rng.choice(np.flatnonzero(~dom.preserve), 3, replace=False):
        h = 1e-6
        xp = x.copy()
        xp[e] += h
        for (v, g, n), (v2, _, _), sc in zip(base, prob.evaluate(xp), scale):
            fd = (v2 - v) / h
            assert abs(fd - g[e]) <= 1e-3 * max(abs(fd), 1e-3 * sc), (n, e, fd, g[e])


def test_robust_sensitivities_match_finite_differences():
    """Floors on the eroded design, ceilings on the dilated one: gradients
    with respect to the filtered design."""
    st = study_from_dict(body())
    dom, floor = _thin_skin(st, build_domain(st, 2.0))
    prob = feel.FeelProblem(st, dom, st.elastic(design=True), float(st.material.E))
    assert prob.robust
    rng = np.random.default_rng(5)
    xt = np.clip(0.3 + 0.5 * rng.random(dom.n_elem), 0, 1)

    def ctx(v):
        return {"xt": v, "beta": 4.0, "floor": floor, "preserve": dom.preserve}

    base = prob.evaluate(None, ctx(xt))
    assert len(base) == prob.m and all(len(b) == 4 and b[3] == "xt" for b in base)
    scale = [np.abs(b[1]).max() for b in base]
    for e in rng.choice(np.flatnonzero(~dom.preserve), 3, replace=False):
        h = 1e-6
        x2 = xt.copy()
        x2[e] += h
        for b, b2, sc in zip(base, prob.evaluate(None, ctx(x2)), scale):
            fd = (b2[0] - b[0]) / h
            assert abs(fd - b[1][e]) <= 1e-3 * max(abs(fd), 1e-3 * sc), (b[2], e, fd, b[1][e])


def test_big_model_eigensolve_matches_the_direct_one():
    """modal_analysis above DIRECT_EIG_DOFS: shift-invert with
    bucklingopt._factor on the whole system (held dofs as identity rows)."""
    from loadpath import modal
    from loadpath.fem import HexMesh
    ijk = np.array([[i, j, k] for i in range(30) for j in range(4) for k in range(4)])
    mesh = HexMesh(ijk, 1.0, np.zeros(3))
    fixed = np.zeros(mesh.n_dof, bool)
    n0 = np.flatnonzero(mesh.node_xyz[:, 0] < 1e-9)
    fixed[(3 * n0[:, None] + np.arange(3)).ravel()] = True
    r1 = modal.modal_analysis(mesh, 2000.0, 1.2, 0.35, fixed, n_modes=6)
    keep = modal.DIRECT_EIG_DOFS
    try:
        modal.DIRECT_EIG_DOFS = 10
        r2 = modal.modal_analysis(mesh, 2000.0, 1.2, 0.35, fixed, n_modes=6)
    finally:
        modal.DIRECT_EIG_DOFS = keep
    assert np.allclose(r1.frequencies, r2.frequencies, rtol=1e-6), (r1.frequencies, r2.frequencies)
    assert np.allclose(r1.participation.sum(axis=0), r2.participation.sum(axis=0), atol=1e-3)


def test_a_part_cut_off_from_the_hold_reads_soft():
    """With nothing between the skin and the mount, the mount floats: it must
    read as having no stiffness, not (as a reduced system that drops it
    would say) as not moving at all."""
    st = study_from_dict(body())
    dom, floor = _thin_skin(st, build_domain(st, 2.0))
    prob = feel.FeelProblem(st, dom, st.elastic(design=True), float(st.material.E))
    bare = np.maximum(np.maximum(dom.preserve.astype(float), floor), 1e-9)
    rows = {r["name"]: r for r in feel.feel_readings(prob, bare)}
    full = {r["name"]: r for r in feel.feel_readings(prob, np.ones(dom.n_elem))}
    for name in ("noise", "drop"):
        for d, f in zip(rows[name]["directions"], full[name]["directions"]):
            assert d["stiffness"] < 1e-2 * f["stiffness"], (name, d, f)
    assert full["drop"]["directions"][0]["stiffness"] > 100


def test_optimizes_into_the_windows():
    # robust reads off: 2 mm cells are too coarse for a suspension that
    # survives erosion in a 16 mm body; this checks the loop and the report
    # (grey cells meet the windows here, and threshold away: see docs/FEEL.md)
    st = study_from_dict(body(robust=False))
    with tempfile.TemporaryDirectory() as out:
        rep = feel.run_feel(st, out, iterations=[40])
    rows = {r["name"]: r for r in rep["feel"]}
    assert rep["feelOk"], [(r["name"], r["directions"]) for r in rep["feel"]]
    assert 0.0 < rep["volume"] < 0.6, rep["volume"]
    assert rep["watertight"]
    # the plumbing of the checks after it (whether a 2 mm, non-robust design
    # survives them is examples/feel/mic_body.py's question, at 1 mm)
    assert rep["verifyPitch"] == 1.0 and all("delivered" in r and "thresholded" in r for r in rep["feel"])
    assert "modal" in rows["noise"]


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"ok   {t.__name__}")
        except Exception:  # noqa: BLE001
            failed += 1
            print(f"FAIL {t.__name__}")
            traceback.print_exc()
    print(f"{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
