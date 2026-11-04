"""The buckling constraint of the constrained method (bucklingopt.py): its
load factor on a solid column against Euler, and its design sensitivities
against finite differences on a grey design."""
import numpy as np

from loadpath.bucklingopt import BucklingModel
from loadpath.domain import build_domain
from loadpath.spec import study_from_dict

A, L, E = 4.0, 40.0, 2000.0


def _domain(side=0.0):
    s = study_from_dict({
        "designSpace": {"type": "box", "min": [0, 0, 0], "max": [A, A, L]},
        "supports": [{"shape": {"type": "box", "min": [-1, -1, -1], "max": [A + 1, A + 1, 0.01]}}],
        "loadCases": [{"name": "press", "loads": [
            {"shape": {"type": "box", "min": [-1, -1, L - 0.01], "max": [A + 1, A + 1, L + 1]},
             "force": [side, 0, -10.0]}]}],
        "material": {"E": E, "nu": 0.3, "strength": 50, "density": 1.2},
        "goal": {"safetyFactor": 2, "volumeFraction": 0.5}, "resolution": {"pitch": 1.0}})
    return build_domain(s)


def test_solid_column_reads_euler():
    d = _domain()
    (mu, _, _), = BucklingModel(d, 0.3).evaluate(np.ones(d.n_elem), 3.0)
    p_euler = np.pi ** 2 * E * (A ** 4 / 12) / (4 * L ** 2)     # fixed base, free top
    # at unit modulus the factor is lambda / E; KS reads the two equal
    # sway modes a little high (conservative)
    lam = E / mu
    assert 0.9 < lam * 10.0 / p_euler < 1.05, lam * 10.0 / p_euler


def test_sensitivities_match_finite_differences():
    d = _domain(side=0.3)
    rng = np.random.default_rng(0)
    x = rng.uniform(0.35, 1.0, d.n_elem)
    bm = BucklingModel(d, 0.3)
    (_, g, _), = bm.evaluate(x, 3.0)
    for e in rng.choice(d.n_elem, 3, replace=False):
        h = 1e-5
        xp, xm = x.copy(), x.copy()
        xp[e] += h
        xm[e] -= h
        fd = (bm.evaluate(xp, 3.0)[0][0] - bm.evaluate(xm, 3.0)[0][0]) / (2 * h)
        assert abs(g[e] - fd) < 0.01 * abs(fd) + 1e-12, (e, g[e], fd)
