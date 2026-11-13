"""
The search's load cases weighted by what binds (simp.binding_weights,
study._binding): the weights make the weighted compliance the p-norm of
the cases' binding ratios with its exact gradient; the ratios come from a
verified part's deflections and safety factors; and on a cantilever whose
side push is far under its limit, weighting by what binds lowers the worst
case's deflection at the same volume.

Run: `python3 tests/test_binding.py` (about half a minute).
"""
import sys
import traceback
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from loadpath.domain import build_domain
from loadpath.simp import SimpOptions, binding_weights, filter_radius, optimize
from loadpath.spec import study_from_dict
from loadpath.study import _binding

BEAM = {
    "designSpace": {"type": "box", "min": [0, 0, 0], "max": [60, 20, 20]},
    "supports": [{"shape": {"type": "box", "min": [-1, -1, -1], "max": [0.01, 21, 21]}}],
    "loadCases": [
        {"name": "down", "loads": [{"shape": {"type": "box", "min": [59.99, 8, 8], "max": [61, 12, 12]},
                                    "force": [0, 0, -10]}]},
        {"name": "side", "loads": [{"shape": {"type": "box", "min": [59.99, 8, 8], "max": [61, 12, 12]},
                                    "force": [0, 20, 0]}]}],
    "material": {"E": 2000, "nu": 0.3, "strength": 50, "density": 1.0},
    "goal": {"safetyFactor": 2.0, "maxDeflection": 1.0},
    "resolution": {"pitch": 2.0},
    "maxIterations": 60,
}


def test_weights_are_the_p_norm_and_its_gradient():
    beta = np.array([0.3, 1.2, 0.05])
    c = np.array([2.0, 1.5, 40.0])
    for p in (2.0, 8.0):
        w, j = binding_weights(beta, c, p)
        assert abs(j - float(np.sum((beta * c) ** p)) ** (1 / p)) < 1e-12 * j
        assert abs(float(w @ c) - j) < 1e-12 * j
        for i in range(3):
            h = 1e-6 * c[i]
            cp = c.copy()
            cp[i] += h
            _, jp = binding_weights(beta, cp, p)
            assert abs((jp - j) / h - w[i]) < 1e-5 * max(w[i], 1e-9), (p, i)
    # far under the worst: next to no weight
    w, _ = binding_weights([1.0, 1.0], [1.0, 0.5], 8.0)
    assert w[1] / w[0] < 0.01


def _outcome(cases, sf, comp, margin):
    return SimpleNamespace(optimizer={"caseCompliance": comp},
                           report={"cases": cases, "safetyFactor": sf, "margin": margin})


def test_ratios_read_off_the_verified_part():
    st = study_from_dict(BEAM)
    dom = build_domain(st)
    # down binds its deflection (0.9 of 1 mm), side has room (0.3 mm, SF 6
    # against 2): side's ratio is max(0.3, 2/6) = 1/3
    src = _outcome([{"name": "down", "deflection": 0.9, "safetyFactor": 5.0},
                    {"name": "side", "deflection": 0.3, "safetyFactor": 6.0}], 5.0, [9.0, 6.0], 1 / 0.9)
    beta = np.asarray(_binding(st, dom, src))
    b = beta * np.array([9.0, 6.0])
    assert abs(b[1] / b[0] - (1 / 3) / 0.9) < 1e-9
    # scaled so that on the part it was read from it is that part's compliance
    j = float(np.sum(b ** 8)) ** (1 / 8)
    assert abs(j - 15.0) < 1e-9
    # a tetrahedral confirm that reads the part weaker scales the cases' SFs:
    # side's 6 counts as 3, its ratio 2/3
    src = _outcome([{"name": "down", "deflection": 0.5, "safetyFactor": 5.0},
                    {"name": "side", "deflection": 0.3, "safetyFactor": 6.0}], 2.5, [9.0, 6.0], 0.5 / 0.5 * 2.5 / 2)
    src.report["margin"] = min(1 / 0.5, 2.5 / 2.0)
    beta = np.asarray(_binding(st, dom, src))
    b = beta * np.array([9.0, 6.0])
    assert abs(b[1] / b[0] - (2 / 3) / (2 / 2.5)) < 1e-9
    # something else sets the margin (buckling): the plain sum
    src.report["margin"] = 0.5
    assert _binding(st, dom, src) is None
    # a case with no reading, or no source: the plain sum
    assert _binding(st, dom, _outcome([{"name": "down", "deflection": 0.5, "safetyFactor": 5.0}], 5.0,
                                      [9.0, 6.0], 2.0)) is None
    assert _binding(st, dom, None) is None


def test_weighting_by_what_binds_lowers_the_worst_deflection():
    st = study_from_dict(BEAM)
    dom = build_domain(st)
    r = filter_radius(dom, st.length_scale, st.min_hole)
    f = np.array([10.0, 20.0])
    worst = []
    for b in (None, tuple(1.0 / f)):
        res = optimize(dom, 0.3, SimpOptions(volume_fraction=0.2, filter_radius=r, max_iterations=60, binding=b))
        # tip deflection under a point load: compliance over the force; at
        # equal volume (pruning moves it a little; stiffness ~ volume)
        worst.append(float(np.max(np.asarray(res.case_compliance) / f)) * res.volume_fraction)
    assert worst[1] < 0.95 * worst[0], worst


TESTS = [test_weights_are_the_p_norm_and_its_gradient, test_ratios_read_off_the_verified_part,
         test_weighting_by_what_binds_lowers_the_worst_deflection]


if __name__ == "__main__":
    failed = 0
    for t in TESTS:
        try:
            t()
            print(f"PASS {t.__name__}")
        except Exception:
            failed += 1
            print(f"FAIL {t.__name__}")
            traceback.print_exc()
    sys.exit(1 if failed else 0)
