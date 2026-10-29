"""
Large deflection and buckling (loadpath.nonlinear) against closed forms:
the tangent against finite differences, small loads against the linear
solver, a tip-loaded cantilever against the elastica, a column against
Euler and its post-buckled shape against the elastica of the buckled
column. Run: `python tests/test_nonlinear.py`.
"""
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from scipy.integrate import solve_bvp, trapezoid
from scipy.special import ellipk

from loadpath.fem import HexMesh, Stiffness
from loadpath.nonlinear import _element, linear_buckling, solve_nonlinear, verify_mechanism

E, NU = 1.0, 0.3
L, T, B, H = 40.0, 2.0, 2.0, 0.5       # 80 x 4 x 4 cells
I = B * T ** 3 / 12


def bar():
    ijk = np.stack(np.meshgrid(np.arange(round(L / H)), np.arange(round(T / H)), np.arange(round(B / H)),
                               indexing="ij"), -1).reshape(-1, 3)
    m = HexMesh(ijk, H, np.zeros(3))
    x = m.node_xyz
    fixed = np.repeat(x[:, 0] < 1e-9, 3)
    tip = np.nonzero(np.abs(x[:, 0] - L) < 1e-9)[0]
    return m, fixed, tip


def tip_vector(m, tip, axis, total):
    v = np.zeros(m.n_dof)
    v[3 * tip + axis] = total / len(tip)
    return v


def elastica_tip(alpha):
    """Tip (along, across) deflection over L of a cantilever with a dead end
    load across it, alpha = P L^2 / EI: theta'' = -alpha cos(theta)."""
    s = np.linspace(0, 1, 201)

    def f(s, y):
        return np.vstack([y[1], -alpha * np.cos(y[0])])

    def bc(a, b):
        return np.array([a[0], b[1]])

    sol = solve_bvp(f, bc, s, np.vstack([0.5 * s, np.zeros_like(s)]), tol=1e-10, max_nodes=100000)
    th = sol.sol(s)[0]
    return trapezoid(1 - np.cos(th), s), trapezoid(np.sin(th), s)


def test_tangent_matches_finite_differences():
    rng = np.random.default_rng(1)
    for kind in ("neo-hookean", "svk"):
        ue = rng.normal(size=(1, 8, 3)) * 0.15
        f, k, _ = _element(ue, 0.7, NU, kind, True)
        kn = np.zeros((24, 24))
        for d in range(24):
            up = ue.reshape(1, 24).copy()
            up[0, d] += 1e-6
            kn[:, d] = (_element(up.reshape(1, 8, 3), 0.7, NU, kind, False)[0][0] - f[0]) / 1e-6
        assert np.abs(kn - k[0]).max() < 1e-5 * np.abs(k).max(), kind


def test_small_load_matches_linear():
    m, fixed, tip = bar()
    f = tip_vector(m, tip, 1, 1e-3 * E * I / L ** 2)
    st = Stiffness(m, NU, fixed)
    from scipy.sparse.linalg import spsolve
    ul = spsolve(st.assemble(np.full(m.n_elem, E)).tocsc(), f)
    r = solve_nonlinear(m, NU, E, fixed, f, steps=1)
    assert r.converged
    vl, vn = ul[3 * tip + 1].mean(), r.u[3 * tip + 1].mean()
    assert abs(vn / vl - 1) < 1e-3, (vn, vl)


def test_cantilever_large_deflection_matches_elastica():
    m, fixed, tip = bar()
    w_v = tip_vector(m, tip, 1, 1.0)
    w_u = tip_vector(m, tip, 0, 1.0)
    # linear FE at unit load: its ratio to beam theory is the element's
    # bending error (shear locking, ~4% at four cells through the thickness)
    st = Stiffness(m, NU, fixed)
    from scipy.sparse.linalg import spsolve
    lin = spsolve(st.assemble(np.full(m.n_elem, E)).tocsc(), tip_vector(m, tip, 1, 1.0))[3 * tip + 1].mean()
    lock = lin / (L ** 3 / (3 * E * I))
    assert 0.94 < lock <= 1.0, lock
    for alpha in (1.0, 3.0):
        P = alpha * E * I / L ** 2
        r = solve_nonlinear(m, NU, E, fixed, tip_vector(m, tip, 1, P), steps=4, probes={"v": w_v, "u": w_u})
        assert r.converged, r.message
        v, u = r.history[-1]["v"] / L, -r.history[-1]["u"] / L
        ua, va = elastica_tip(alpha)
        # within the element's own bending error of the exact elastica ...
        assert abs(v / va - 1) < 0.05, (alpha, v, va)
        assert abs(u / ua - 1) < 0.08, (alpha, u, ua)
        # ... and the nonlinear stiffening itself to about 3%
        assert abs((v / (lin * P / L)) / (va / (alpha / 3)) - 1) < 0.03, (alpha, v, va)


def test_stroke_control_reaction_matches_force():
    # push the tip to where a force put it: the reaction is that force
    m, fixed, tip = bar()
    P = 2.0 * E * I / L ** 2
    w_v = tip_vector(m, tip, 1, 1.0)
    r = solve_nonlinear(m, NU, E, fixed, tip_vector(m, tip, 1, P), steps=4, probes={"v": w_v})
    v = r.u[3 * tip + 1]
    d = solve_nonlinear(m, NU, E, fixed, prescribed=(3 * tip + 1, v), steps=4)
    assert d.converged, d.message
    ry = d.history[-1]["reaction"][1]
    assert abs(ry / P - 1) < 0.02, (ry, P)


def test_column_buckles_at_euler_and_bends_like_the_elastica():
    m, fixed, tip = bar()
    pcr = np.pi ** 2 * E * I / (4 * L ** 2)
    b = linear_buckling(m, NU, E, fixed, tip_vector(m, tip, 0, -1.0), n_modes=2)
    assert abs(b.factors[0] / pcr - 1) < 0.05, (b.factors, pcr)
    # the mode is a sideways sway, largest at the tip
    mode = b.modes[0].reshape(-1, 3)
    assert np.abs(mode[tip, 1:]).max() > 0.9
    # Past the bifurcation: at P = 1.152 Pcr the elastica's tip turns 60
    # degrees and sways k / K(k) * 2 L sideways, k = sin 30. (Pcr from the
    # eigenvalue above, so the element's bending error cancels.)
    k = 0.5
    ratio = (2 * ellipk(k ** 2) / np.pi) ** 2
    sway = 2 * k / ellipk(k ** 2)
    # With a slight side load the path bends away smoothly; with none (a
    # perfect column) the solver has to find the bifurcation itself, leave
    # the unstable straight branch, and sway either way.
    for side in (1e-3, 0.0):
        f = tip_vector(m, tip, 0, -ratio * b.factors[0]) + tip_vector(m, tip, 1, side * b.factors[0])
        r = solve_nonlinear(m, NU, E, fixed, f, steps=10,
                            probes={"v": tip_vector(m, tip, 1, 1.0), "w": tip_vector(m, tip, 2, 1.0)})
        assert r.converged, r.message
        got = np.hypot(r.history[-1]["v"], r.history[-1]["w"]) / L
        assert abs(got / sway - 1) < 0.03, (side, got, sway)
        assert r.history[-1]["lowestEigenvalue"] > 0
        assert side or any(h.get("branchSwitch") for h in r.history)


def test_large_mesh_path_follows_a_perfect_column():
    # Above DIRECT_MAX_DOF the Newton steps run on CG + AMG, which can't
    # step onto an unstable branch: the solver seeds an imperfection along
    # the buckling mode instead. Forced here on the small bar.
    import loadpath.nonlinear as nl
    m, fixed, tip = bar()
    pcr = linear_buckling(m, NU, E, fixed, tip_vector(m, tip, 0, -1.0), n_modes=1).factors[0]
    k = 0.5
    ratio = (2 * ellipk(k ** 2) / np.pi) ** 2
    sway = 2 * k / ellipk(k ** 2)
    keep = nl.DIRECT_MAX_DOF
    nl.DIRECT_MAX_DOF = 0
    try:
        r = solve_nonlinear(m, NU, E, fixed, tip_vector(m, tip, 0, -ratio * pcr), steps=10)
    finally:
        nl.DIRECT_MAX_DOF = keep
    assert r.converged and "imperfection" in r.message, r.message
    got = np.hypot(r.u[3 * tip + 1].mean(), r.u[3 * tip + 2].mean()) / L
    assert abs(got / sway - 1) < 0.03, (got, sway)


def test_verify_mechanism_reports_a_finger():
    # A plain TPU strip (all solid) as a "mechanism": push the tip sideways,
    # output is the same tip. Nonlinear travel is less than linear (the strip
    # stiffens as it curls), and an axial push buckles it.
    from types import SimpleNamespace
    m, fixed, tip = bar()
    P = 3.0 * 26 * I / L ** 2
    dom = SimpleNamespace(mesh=m, fixed=fixed,
                          loads=[tip_vector(m, tip, 1, P), tip_vector(m, tip, 1, 1.0)])
    rep = verify_mechanism(dom, np.ones(m.n_elem), 26.0, NU, steps=4)
    assert rep["converged"] and rep["elements"] == m.n_elem
    assert 0.6 < rep["nonlinearOverLinear"] < 0.85, rep["nonlinearOverLinear"]
    assert rep["bucklingFactor"] == np.inf or rep["bucklingFactor"] > 1


def test_mechanism_optimizer_reports_nonlinear_check():
    # Sigmund's inverter (as in test_loadpath), with the check switched on:
    # against the optimization's own springs the travel is small, so linear
    # and nonlinear agree, and the input force is well short of buckling.
    from loadpath.domain import build_domain
    from loadpath.mechanism import MechanismOptions, optimize_mechanism
    from loadpath.spec import study_from_dict
    spec = {
        "designSpace": {"type": "box", "min": [0, 0, 0], "max": [40, 20, 2]},
        "supports": [{"shape": {"type": "box", "min": [-1, -1, -1], "max": [2.1, 2.1, 3]}},
                     {"shape": {"type": "box", "min": [-1, 17.9, -1], "max": [2.1, 21, 3]}}],
        "loadCases": [
            {"name": "in", "loads": [{"shape": {"type": "box", "min": [-1, 8, -1], "max": [1.1, 12, 3]}, "force": [10, 0, 0]}]},
            {"name": "out", "loads": [{"shape": {"type": "box", "min": [38.9, 8, -1], "max": [41, 12, 3]}, "force": [-1, 0, 0]}]}],
        "material": {"E": 26, "nu": 0.45, "strength": 30, "density": 1.2},
        "goal": {"volumeFraction": 0.3}, "resolution": {"pitch": 2.0}}
    d = build_domain(study_from_dict(spec))
    r = optimize_mechanism(d, 0.45, MechanismOptions(volume_fraction=0.3, filter_radius=3.0, modulus=26, k_in=26.0,
                                                      k_out=1.3, extrusion=2, max_iterations=60, verify_nonlinear=True))
    nl = r.nonlinear
    assert nl is not None and nl["converged"], nl
    assert nl["nonlinearOutput"] > 0, nl["nonlinearOutput"]
    assert abs(nl["nonlinearOverLinear"] - 1) < 0.1, nl["nonlinearOverLinear"]
    assert nl["bucklingFactor"] > 1, nl["bucklingFactor"]


TESTS = [
    test_tangent_matches_finite_differences,
    test_small_load_matches_linear,
    test_cantilever_large_deflection_matches_elastica,
    test_stroke_control_reaction_matches_force,
    test_column_buckles_at_euler_and_bends_like_the_elastica,
    test_large_mesh_path_follows_a_perfect_column,
    test_verify_mechanism_reports_a_finger,
    test_mechanism_optimizer_reports_nonlinear_check,
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
