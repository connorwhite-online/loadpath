"""
Tests for loads of unknown direction ("direction": "any", "cone"):
the worst-direction algebra against brute force, the optimizer's worst case
against direct solves, and verification picking the most stressful
direction. Small grids; runs in well under a minute.

Run: `python3 tests/test_directions.py`.
"""
import copy
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import trimesh

from loadpath.directions import collapse, expand, fibonacci_sphere, worst_direction
from loadpath.domain import build_domain
from loadpath.simp import E_MIN, PENALTY, SimpOptions, _Analysis, filter_radius, optimize
from loadpath.spec import study_from_dict
from loadpath.verify import verify_part

# A short square post on the floor, pushed at its top: a load from the side
# is the post's weak direction, from above its strong one.
POST = {
    "designSpace": {"type": "box", "min": [0, 0, 0], "max": [16, 16, 40]},
    "supports": [{"shape": {"type": "box", "min": [-1, -1, -1], "max": [17, 17, 0.01]}}],
    "loadCases": [{"name": "push", "loads": [
        {"shape": {"type": "box", "min": [-1, -1, 39.99], "max": [17, 17, 41]}, "force": [0, 0, -10],
         "direction": "any"}]}],
    "material": {"E": 2000, "nu": 0.3, "strength": 50, "density": 1.0},
    "goal": {"volumeFraction": 0.3},
    "resolution": {"pitch": 2.0},
    "maxIterations": 40,
}

# A cantilever loaded down at its tip: tuned for that load alone it becomes
# a deep, narrow web, weak sideways.
BEAM = {
    "designSpace": {"type": "box", "min": [0, 0, 0], "max": [60, 20, 20]},
    "supports": [{"shape": {"type": "box", "min": [-1, -1, -1], "max": [0.01, 21, 21]}}],
    "loadCases": [{"name": "tip", "loads": [
        {"shape": {"type": "box", "min": [59.99, 8, 8], "max": [61, 12, 12]}, "force": [0, 0, -10],
         "direction": "any"}]}],
    "material": {"E": 2000, "nu": 0.3, "strength": 50, "density": 1.0},
    "goal": {"volumeFraction": 0.25},
    "resolution": {"pitch": 2.0},
    "maxIterations": 60,
}


def _with(load_extra, base=POST):
    d = copy.deepcopy(base)
    ld = d["loadCases"][0]["loads"][0]
    ld.pop("direction", None)
    ld.update(load_extra)
    return d


def test_worst_direction_matches_brute_force():
    rng = np.random.default_rng(0)
    pts = fibonacci_sphere(100_000)
    for t in range(16):
        m = rng.normal(size=(3, 3))
        a = m @ m.T
        g = rng.normal(size=3) * (t % 4) * 0.5
        axis = rng.normal(size=3)
        axis /= np.linalg.norm(axis)
        cos_half = np.cos(np.radians(rng.uniform(5, 120))) if t % 2 else -1.0
        d, v = worst_direction(3.0, g, a, axis, cos_half)
        inside = pts @ axis >= cos_half if cos_half > -1 else np.ones(len(pts), bool)
        p = pts[inside]
        brute = (3.0 + 2 * p @ g + np.einsum("ij,jk,ik->i", p, a, p)).max()
        assert v >= brute - 1e-3 * abs(brute), (t, v, brute)
        assert abs(np.linalg.norm(d) - 1) < 1e-9 and (cos_half <= -1 or d @ axis >= cos_half - 1e-9)
    # no fixed load: exactly the top eigenvalue
    a = np.diag([1.0, 5.0, 2.0])
    d, v = worst_direction(0.0, np.zeros(3), a)
    assert abs(v - 5.0) < 1e-12 and abs(abs(d[1]) - 1) < 1e-12


def test_spec_and_schema():
    for bad, msg in [({"direction": "up"}, "direction"), ({"cone": 0}, "cone"),
                     ({"cone": 30, "direction": "any"}, "cone")]:
        try:
            study_from_dict(_with(bad))
            raise AssertionError(f"accepted {bad}")
        except ValueError as e:
            assert msg in str(e), e
    two = copy.deepcopy(POST)
    two["loadCases"][0]["loads"].append(dict(two["loadCases"][0]["loads"][0]))
    try:
        study_from_dict(two)
        raise AssertionError("accepted two uncertain loads in one case")
    except ValueError as e:
        assert "at most one" in str(e)
    st = study_from_dict(_with({"cone": 180}))
    assert st.load_cases[0].loads[0].direction == "any"
    from loadpath.schema import lint
    assert not lint(POST, "study")
    assert not lint(_with({"cone": 25}), "study")


def _solve_one(domain, xp, f):
    fe = _Analysis(domain, 0.3, 1e-8)
    return fe.solve(E_MIN + xp ** PENALTY * (1 - E_MIN), xp, [None], rtol=1e-9, loads=[f])[0]


def test_worst_case_compliance_is_exact():
    # three solves give the compliance in every direction; check the worst
    # one against direct solves along it and along a spread of others
    st = study_from_dict(_with({"direction": "any", "moment": [0, 0, 30]}))
    dom = build_domain(st)
    sp = dom.spreads[0]
    assert sp is not None and sp.kind == "any"
    xp = np.random.default_rng(1).uniform(0.2, 1.0, dom.n_elem)
    fe = _Analysis(dom, 0.3, 1e-8)
    rhs, lay = expand(dom.loads, dom.spreads)
    assert len(rhs) == 4   # the moment is a fixed part, plus three axes
    us = fe.solve(E_MIN + xp ** PENALTY * (1 - E_MIN), xp, [None] * 4, rtol=1e-9, loads=rhs)
    w = collapse(rhs, lay, us, dom.spreads)[0]
    c, d, f = w.compliance, w.direction, w.force
    cd = float(f @ _solve_one(dom, xp, f))
    assert abs(cd - c) < 1e-5 * c, (cd, c)
    for e in fibonacci_sphere(12):
        fe_ = sp.load(e)
        assert float(fe_ @ _solve_one(dom, xp, fe_)) <= c * (1 + 1e-6)
    # sideways is the post's weak direction
    assert abs(d[2]) < 0.5, d


def test_optimizer_designs_for_the_worst_direction():
    # tuned for the written load (straight down) the cantilever is a web
    # that folds under a push from the side; tuned for any direction not
    nominal = study_from_dict(_with({}, BEAM))
    anyd = study_from_dict(BEAM)
    probe = build_domain(anyd)
    out = {}
    for name, st in [("nominal", nominal), ("any", anyd)]:
        dom = build_domain(st)
        r = optimize(dom, 0.3, SimpOptions(volume_fraction=0.25, filter_radius=filter_radius(dom, None),
                                           max_iterations=80))
        fe = _Analysis(probe, 0.3, 1e-8)
        rhs, lay = expand(probe.loads, probe.spreads)
        us = fe.solve(E_MIN + r.xphys ** PENALTY * (1 - E_MIN), r.xphys, [None] * len(rhs), rtol=1e-8, loads=rhs)
        out[name] = collapse(rhs, lay, us, probe.spreads)[0].compliance
        if name == "any":
            assert r.case_direction[0] is not None and abs(np.linalg.norm(r.case_direction[0]) - 1) < 1e-9
            assert abs(r.case_compliance[0] - out[name]) < 0.05 * out[name]
        else:
            assert r.case_direction == [None]
    # measured 2687 against 13532
    assert out["any"] < 0.4 * out["nominal"], out


def test_cone_is_between_nominal_and_any():
    dom_n = build_domain(study_from_dict(_with({})))
    dom_c = build_domain(study_from_dict(_with({"cone": 20})))
    dom_a = build_domain(study_from_dict(POST))
    xp = np.ones(dom_a.n_elem)
    worst = []
    for dom in (dom_n, dom_c, dom_a):
        fe = _Analysis(dom, 0.3, 1e-8)
        rhs, lay = expand(dom.loads, dom.spreads or [None])
        us = fe.solve(np.ones(dom.n_elem), xp, [None] * len(rhs), rtol=1e-9, loads=rhs)
        worst.append(collapse(rhs, lay, us, dom.spreads or [None])[0])
    cn, cc, ca = (w.compliance for w in worst)
    assert cn < cc < ca, (cn, cc, ca)
    d = worst[1].direction
    assert d @ np.array([0, 0, -1.0]) >= np.cos(np.radians(20)) - 1e-9


def test_verify_reports_the_most_stressful_direction():
    part = trimesh.creation.box(extents=[16, 16, 40])
    part.apply_translation([8, 8, 20])
    rep_n = verify_part(study_from_dict(_with({})), part, max_cells=6000, refine=0)
    rep_a = verify_part(study_from_dict(POST), part, max_cells=6000, refine=0)
    ca = rep_a["cases"][0]
    d = np.array(ca["worstDirection"])
    assert abs(np.linalg.norm(d) - 1) < 1e-9 and abs(d[2]) < 0.3, d
    # a push from the side bends the post: far more stress than the axial one
    assert ca["peakStress"] > 3 * rep_n["cases"][0]["peakStress"]
    assert "worstDirection" not in rep_n["cases"][0]


TESTS = [test_worst_direction_matches_brute_force, test_spec_and_schema, test_worst_case_compliance_is_exact,
         test_optimizer_designs_for_the_worst_direction, test_cone_is_between_nominal_and_any,
         test_verify_reports_the_most_stressful_direction]


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
