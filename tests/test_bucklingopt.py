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


def test_layered_material():
    """FDM studies pass a fem.Layered for nu: the constraint took the Lame
    constants of it and failed every iteration ('int' + 'Layered'), so the
    buckling constraint never ran for layered parts. Isotropic layers read
    the same factor as the plain Poisson's ratio; weak layers read lower."""
    from loadpath.fem import Layered

    d = _domain()
    x = np.ones(d.n_elem)
    (iso, _, _), = BucklingModel(d, 0.3).evaluate(x, 3.0)
    (lay, _, _), = BucklingModel(d, Layered(0.3)).evaluate(x, 3.0)
    assert abs(lay - iso) < 1e-6 * abs(iso), (lay, iso)
    (soft, _, _), = BucklingModel(d, Layered(0.3, ez=0.6, gz=0.6, normal=(1, 0, 0))).evaluate(x, 3.0)
    assert np.isfinite(soft)


def test_block_modes_reads_the_largest_eigenpairs():
    """block_modes against a dense solve of a phi = mu K phi: from a random
    start (one power step first), from nearly converged vectors with K
    given (no power step: one block solve confirms them), and stopping
    early, within a percent, while mu reads under `loose_below`."""
    from scipy.linalg import cho_factor, cho_solve, eigh

    from loadpath.bucklingopt import BLOCK_ITERS_COLD, block_modes

    rng = np.random.default_rng(0)
    n, nm = 120, 3
    q = np.linalg.qr(rng.standard_normal((n, n)))[0]
    k = (q * np.geomspace(1.0, 1e3, n)) @ q.T
    k = 0.5 * (k + k.T)
    b = rng.standard_normal((n, n))
    a = b + b.T
    ref, vec = eigh(a, k)
    ref, vec = ref[::-1], vec[:, ::-1]
    cf = cho_factor(k)
    calls = []

    def solve(r):
        calls.append(1)
        return cho_solve(cf, r)

    mu, x, its, ok = block_modes(a, solve, rng.standard_normal((n, nm + 2)), nm, maxiter=BLOCK_ITERS_COLD)
    assert ok and np.allclose(mu[:nm], ref[:nm], rtol=1e-4), (mu[:nm], ref[:nm])
    assert np.allclose(x[:, :nm].T @ k @ x[:, :nm], np.eye(nm), atol=1e-8)
    cold = len(calls)
    calls.clear()
    mu, x, its, ok = block_modes(a, solve, vec[:, :nm + 2] @ rng.standard_normal((nm + 2, nm + 2)), nm, k=k)
    assert ok and its == 0 and len(calls) == 1 and np.allclose(mu[:nm], ref[:nm], rtol=1e-6)
    calls.clear()
    mu, x, its, ok = block_modes(a, solve, rng.standard_normal((n, nm + 2)), nm, maxiter=BLOCK_ITERS_COLD,
                                 loose_below=np.inf)
    assert ok and len(calls) < cold and abs(mu[0] - ref[0]) < 0.02 * ref[0], (len(calls), cold, mu[0], ref[0])


def test_relaxed_reading_and_carried_modes():
    """evaluate's `relax`: a case far from its limit is read to a percent
    and gets no gradient; one near it is read as before. The modes a model
    leaves on its domain (keep) start the next model's solve."""
    d = _domain(side=0.3)
    x = np.random.default_rng(0).uniform(0.35, 1.0, d.n_elem)
    bm = BucklingModel(d, 0.3)
    (mu, g, _), = bm.evaluate(x, 3.0)
    (far, g_far, _), = BucklingModel(d, 0.3).evaluate(x, 3.0, relax=[(np.inf, np.inf)])
    assert abs(far - mu) < 0.02 * mu and not g_far.any(), (far, mu)
    (near, g_near, _), = BucklingModel(d, 0.3).evaluate(x, 3.0, relax=[(0.5 * mu, 0.2 * mu)])
    assert abs(near - mu) < 1e-4 * mu and np.allclose(g_near, g, rtol=1e-2, atol=1e-3 * np.abs(g).max())
    cold = bm.last["eigIterations"]
    bm.keep()
    nxt = BucklingModel(d, 0.3)
    (again, _, _), = nxt.evaluate(x, 3.0)
    assert abs(again - mu) < 1e-4 * mu and nxt.last["eigIterations"] < max(cold, 1), (nxt.last, cold)
    crisp = BucklingModel(_domain(side=0.3), 0.3)
    crisp.seed(bm)
    assert set(crisp._v0) == set(bm._v0)
