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
from loadpath.nonlinear import Tendon, _element, linear_buckling, solve_nonlinear, verify_mechanism

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


def test_incompatible_mode_tangent_matches_finite_differences():
    # the condensed tangent is the derivative of the condensed internal
    # force (bubble amplitudes in local equilibrium), and at u = 0 it is
    # fem.py's linear incompatible-mode element
    from loadpath.fem import hex8_ke
    h = 0.7
    z = _element(np.zeros((1, 8, 3)), h, NU, "neo-hookean", True, np.zeros((1, 3, 3)))
    assert np.allclose(z[1][0], hex8_ke(NU, "im") * h, atol=1e-14)
    rng = np.random.default_rng(1)
    for kind in ("neo-hookean", "svk"):
        ue = rng.normal(size=(1, 8, 3)) * 0.1

        def condensed(u, a, tangent):
            for _ in range(30):
                out = _element(u, h, NU, kind, tangent, a)
                a = out[3]
            return out, a

        (f, k, _, _, ra, _), a = condensed(ue, np.zeros((1, 3, 3)), True)
        assert np.abs(ra).max() < 1e-12
        kn = np.zeros((24, 24))
        for d in range(24):
            du = np.zeros(24)
            du[d] = 1e-6
            fp = condensed(ue + du.reshape(1, 8, 3), a, False)[0][0][0]
            fm = condensed(ue - du.reshape(1, 8, 3), a, False)[0][0][0]
            kn[:, d] = (fp - fm) / 2e-6
        assert np.abs(kn - k[0]).max() < 1e-6 * np.abs(k).max(), kind


def test_small_load_matches_linear():
    m, fixed, tip = bar()
    f = tip_vector(m, tip, 1, 1e-3 * E * I / L ** 2)
    from scipy.sparse.linalg import spsolve
    for element in ("im", "plain"):
        st = Stiffness(m, NU, fixed, element=element)   # the nonlinear element's own linear limit
        ul = spsolve(st.assemble(np.full(m.n_elem, E)).tocsc(), f)
        r = solve_nonlinear(m, NU, E, fixed, f, steps=1, element=element)
        assert r.converged
        vl, vn = ul[3 * tip + 1].mean(), r.u[3 * tip + 1].mean()
        assert abs(vn / vl - 1) < 1e-3, (element, vn, vl)


def test_cantilever_large_deflection_matches_elastica():
    m, fixed, tip = bar()
    w_v = tip_vector(m, tip, 1, 1.0)
    w_u = tip_vector(m, tip, 0, 1.0)
    # linear FE at unit load: its ratio to beam theory is the element's
    # bending error (the plain hex8 shear-locked ~4% at four cells through
    # the thickness; with the incompatible modes it is beam theory plus a
    # little shear deflection)
    st = Stiffness(m, NU, fixed)   # the nonlinear element's own linear limit
    from scipy.sparse.linalg import spsolve
    lin = spsolve(st.assemble(np.full(m.n_elem, E)).tocsc(), tip_vector(m, tip, 1, 1.0))[3 * tip + 1].mean()
    lock = lin / (L ** 3 / (3 * E * I))
    assert 0.995 < lock <= 1.02, lock
    for alpha in (1.0, 3.0):
        P = alpha * E * I / L ** 2
        r = solve_nonlinear(m, NU, E, fixed, tip_vector(m, tip, 1, P), steps=4, probes={"v": w_v, "u": w_u})
        assert r.converged, r.message
        v, u = r.history[-1]["v"] / L, -r.history[-1]["u"] / L
        ua, va = elastica_tip(alpha)
        # within ~1% of the exact elastica (the plain element was 5-8% off) ...
        assert abs(v / va - 1) < 0.015, (alpha, v, va)
        assert abs(u / ua - 1) < 0.03, (alpha, u, ua)
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
    assert abs(b.factors[0] / pcr - 1) < 0.02, (b.factors, pcr)
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


def test_one_cell_strip_buckles_and_bends_like_beam_theory():
    # one cell through the thickness: the plain hex8 shear-locked to 1.54x
    # Euler's load and a tip deflection ~20% short of the elastica
    Ls, t, b = 40, 1, 2
    ijk = np.stack(np.meshgrid(np.arange(Ls), np.arange(t), np.arange(b), indexing="ij"), -1).reshape(-1, 3)
    m = HexMesh(ijk, 1.0, np.zeros(3))
    x = m.node_xyz
    fixed = np.repeat(x[:, 0] < 1e-9, 3)
    tip = np.nonzero(np.abs(x[:, 0] - Ls) < 1e-9)[0]
    iy = b * t ** 3 / 12
    pcr = np.pi ** 2 * E * iy / (4 * Ls ** 2)
    got = linear_buckling(m, NU, E, fixed, tip_vector(m, tip, 0, -1.0), n_modes=1).factors[0]
    assert abs(got / pcr - 1) < 0.01, got / pcr
    assert linear_buckling(m, NU, E, fixed, tip_vector(m, tip, 0, -1.0), n_modes=1, element="plain").factors[0] > 1.4 * pcr
    alpha = 3.0
    r = solve_nonlinear(m, NU, E, fixed, tip_vector(m, tip, 1, alpha * E * iy / Ls ** 2), steps=4,
                        probes={"v": tip_vector(m, tip, 1, 1.0)}, stability=False)
    assert r.converged, r.message
    va = elastica_tip(alpha)[1]
    assert abs(r.history[-1]["v"] / Ls / va - 1) < 0.01, (r.history[-1]["v"] / Ls, va)


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


def test_tendon_curls_a_cantilever_into_an_arc():
    # A cable at tension T through closely spaced guides along the top face
    # (offset e = T/2 from the axis), tied at the tip, pulled from the wall:
    # every section carries the same moment T e however far it curls, so the
    # beam bends into a circular arc with curvature T e / EI. A dead force
    # at the tip would bend it ever harder near the root as it curls.
    m, fixed, tip = bar()
    x = m.node_xyz
    top = np.abs(x[:, 1] - T) < 1e-9
    path = [np.array([0.0, T, B / 2])] + [np.nonzero(top & (np.abs(x[:, 0] - xs) < 1e-9))[0]
                                          for xs in np.linspace(L / 40, L, 40)]
    e, theta = T / 2, np.pi / 2
    tension = theta * E * I / (L * e)
    tendon = Tendon(tension, path)
    # the load stiffness is the derivative of the load
    tendon._setup(m)
    u0 = np.random.default_rng(2).normal(size=m.n_dof) * 0.05
    f0, k0, _ = tendon._load(u0, 1.0)
    du = np.random.default_rng(3).normal(size=m.n_dof) * 1e-6
    f1, _, _ = tendon._load(u0 + du, 1.0)
    assert np.linalg.norm((f0 - f1) - k0 @ du) < 1e-4 * np.linalg.norm(f0 - f1)

    r = solve_nonlinear(m, NU, E, fixed, tendons=[tendon], steps=8)
    assert r.converged, r.message
    du3 = r.u.reshape(-1, 3)

    def rotation(xs):
        sec = np.abs(x[:, 0] - xs) < 1e-9
        pt = (x + du3)[sec & top].mean(axis=0)
        pb = (x + du3)[sec & (np.abs(x[:, 1]) < 1e-9)].mean(axis=0)
        d = pt - pb
        return np.arctan2(-d[0], d[1])

    # rotation grows linearly along the beam (uniform curvature) ...
    rot = np.array([rotation(xs) for xs in (10, 20, 30, 40)])
    assert np.abs(rot / (rot[-1] * np.array([0.25, 0.5, 0.75, 1.0])) - 1).max() < 0.02, np.degrees(rot)
    # ... to 90 degrees at the tip (the plain hex8 fell ~4% short)
    assert abs(rot[-1] / theta - 1.0) < 0.01, np.degrees(rot[-1])
    # cable pulled in: the top fibre's shortening, bending plus compression
    travel = r.history[-1]["tendon0Travel"]
    expect = e * rot[-1] + tension * L / (E * T * B)
    assert abs(travel / expect - 1) < 0.03, (travel, expect)


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


def test_stroke_mechanism_pushes_a_bar():
    # A straight bar pushed axially at its tip by a small stroke: the output
    # (the same tip, same direction) moves by the stroke, and the force is
    # E A / L x stroke. A workpiece spring there carries k x stroke (on the
    # driven dofs themselves, so the actuator does not feel it here).
    from types import SimpleNamespace
    from loadpath.nonlinear import stroke_mechanism
    m, fixed, tip = bar()
    dom = SimpleNamespace(mesh=m, fixed=fixed, loads=[tip_vector(m, tip, 0, -1.0), tip_vector(m, tip, 0, -1.0)])
    s = 0.02
    r = stroke_mechanism(dom, np.ones(m.n_elem), 26.0, NU, s, steps=2)
    assert r["converged"] and abs(r["geometricAdvantage"] - 1) < 1e-6
    ea_l = 26.0 * T * B / L
    assert abs(r["actuatorForce"] / (ea_l * s) - 1) < 0.05, r["actuatorForce"] / (ea_l * s)
    assert abs(r["linearActuatorForce"] / r["actuatorForce"] - 1) < 0.02
    r = stroke_mechanism(dom, np.ones(m.n_elem), 26.0, NU, s, k_out=5.0, steps=2)
    assert abs(r["gripForce"] - 5.0 * s) < 1e-6


def test_robust_mechanism_has_no_corner_hinges():
    # the benchmark inverter, small: the robust (eroded / dilated) formulation
    # keeps a positive output and leaves no one-node hinges
    from loadpath.benchmarks.mechanisms import MechEvaluator, hinges, inverter, mech_domain
    from loadpath.mechanism import MechanismOptions, optimize_mechanism
    p = inverter(30, 15)
    dom = mech_domain(p)
    r = optimize_mechanism(dom, 0.3, MechanismOptions(volume_fraction=0.3, filter_radius=2.0, k_in=p.k_in,
                                                      k_out=p.k_out, extrusion=1, robust=0.15, max_iterations=80))
    g = np.zeros(dom.dims)
    g[tuple(dom.mesh.elem_ijk.T)] = r.xphys
    rob = r.history[-1]["robustOutputs"]
    assert rob[0] <= rob[1] + 1e-9 and r.output_displacement > 0
    assert hinges((g > 0.5).astype(float)) == 0
    assert MechEvaluator(p).output((g > 0.5).astype(float)) > 0.5 * r.output_displacement


def test_nonlinear_mechanism_gradient_matches_finite_differences():
    # mechanism_nl's adjoint sensitivity of the output travel at a large
    # stroke, on a random density field of a small inverter, including the
    # solid/linear blend of the void cells
    from loadpath.benchmarks.mechanisms import inverter, mech_domain
    from loadpath.mechanism import _spring_diag
    from loadpath.mechanism_nl import _Model, nonlinear_response, stroke_dofs
    p = inverter(12, 6)
    dom = mech_domain(p)
    m = dom.mesh
    pd, pv = stroke_dofs(m, dom.loads[0], dom.fixed, 2.0)
    lo = dom.loads[1]
    wts = np.abs(lo.reshape(-1, 3)).sum(axis=1) > 0
    d = lo.reshape(-1, 3).sum(axis=0)
    lv = (wts[:, None] * (d / np.linalg.norm(d))[None, :]).ravel() / wts.sum()
    springs = _spring_diag(lo, 0.05, m.n_dof)
    springs[dom.fixed] = 0.0
    model = _Model(m, 0.3, dom.fixed, pd, pv, springs)
    rng = np.random.default_rng(1)
    xp = np.clip(0.5 + 0.35 * rng.standard_normal(m.n_elem), 0.05, 1.0)
    j, dj, u = nonlinear_response(model, xp, 1.0, lv)
    for e in (3, 20, 41, 60):
        h = 1e-5  # 1e-6 sits inside the Newton tolerance on the tiny ones
        xa, xb = xp.copy(), xp.copy()
        xa[e] += h
        xb[e] -= h
        fd = (nonlinear_response(model, xa, 1.0, lv, u0=u)[0] - nonlinear_response(model, xb, 1.0, lv, u0=u)[0]) / (2 * h)
        assert abs(dj[e] - fd) <= 1e-5 * max(abs(fd), 1e-3), (e, dj[e], fd)


def test_followed_load_path_does_not_depend_on_the_last_design():
    # mechanism_nl follows each design's load path from rest; seeding the
    # levels with another design's path must not change the equilibrium it
    # lands on (a warm start straight at the full stroke could)
    from loadpath.benchmarks.mechanisms import inverter, mech_domain
    from loadpath.mechanism import _spring_diag
    from loadpath.mechanism_nl import _Model, nonlinear_response, stroke_dofs
    p = inverter(12, 6)
    dom = mech_domain(p)
    m = dom.mesh
    pd, pv = stroke_dofs(m, dom.loads[0], dom.fixed, 2.0)
    lo = dom.loads[1]
    wts = np.abs(lo.reshape(-1, 3)).sum(axis=1) > 0
    d = lo.reshape(-1, 3).sum(axis=0)
    lv = (wts[:, None] * (d / np.linalg.norm(d))[None, :]).ravel() / wts.sum()
    springs = _spring_diag(lo, 0.05, m.n_dof)
    springs[dom.fixed] = 0.0
    model = _Model(m, 0.3, dom.fixed, pd, pv, springs)
    rng = np.random.default_rng(1)
    xp = np.clip(0.5 + 0.35 * rng.standard_normal(m.n_elem), 0.05, 1.0)
    xq = np.clip(xp + 0.2 * rng.standard_normal(m.n_elem), 0.05, 1.0)
    j, dj, path = nonlinear_response(model, xp, 1.0, lv, follow=True)
    _, _, other = nonlinear_response(model, xq, 1.0, lv, follow=True)
    j2, dj2, path2 = nonlinear_response(model, xp, 1.0, lv, u0=other, follow=True)
    assert len(path) == 4 and abs(j2 - j) <= 1e-7 * abs(j)
    assert np.abs(dj2 - dj).max() <= 1e-6 * np.abs(dj).max()


def test_keep_best_returns_the_best_printed_iterate():
    # keep_best: every iteration of the sharpest phase is checked as it would
    # print (thresholded at 0.5); the result is the best of them, not
    # wherever the run happened to stop
    from loadpath.benchmarks.mechanisms import inverter, mech_domain
    from loadpath.mechanism import MechanismOptions
    from loadpath.mechanism_nl import optimize_mechanism_nonlinear
    dom = mech_domain(inverter(16, 8))
    o = MechanismOptions(volume_fraction=0.3, filter_radius=1.5, modulus=1.0, k_out=0.05, robust=0.15, extrusion=1,
                         beta_every=5, beta_max=8, max_iterations=40)
    r = optimize_mechanism_nonlinear(dom, 0.3, o, stroke=2.0, keep_best=True)
    seen = [h["thresholdedOutput"] for h in r.history[:-1] if h.get("thresholdedOutput") is not None]
    assert len(seen) >= 5, seen
    assert r.history[-1]["thresholdedOutput"] >= max(seen) - 1e-12
    assert r.history[-1]["thresholdedOutput"] > seen[-1] and "keptIteration" in r.history[-1]


def test_tendon_grip_gradient_matches_finite_differences():
    # a servo-wound tendon curls a small finger onto a rigid cylinder; the
    # adjoint sensitivity of the grip force (contact penalty weighted by the
    # density, tendon load stiffness in the tangent) against central
    # differences on a random density field
    from loadpath.benchmarks.mechanisms import MechProblem, mech_domain
    from loadpath.benchmarks.problems import _nodes
    from loadpath.mechanism_nl import RigidCylinder, ServoTendon, _Model, nonlinear_response
    from loadpath.nonlinear import nodes_near
    p = MechProblem(name="f", title="f", source="", dims=(16, 1, 4), volfrac=0.4, rmin=1.5,
                    fixed=[(_nodes(0, 0, range(0, 5)), "xz")], sym=[], inp=(_nodes(16, 0, range(0, 2)), "x", -1.0),
                    out=(_nodes(16, 0, range(0, 2)), "z", -1.0))
    dom = mech_domain(p)
    m = dom.mesh
    ten = ServoTendon(path=[np.array([-2.0, 0.5, 0.5]), nodes_near(m, (8, 0.5, 0), 0.8),
                            nodes_near(m, (16, 0.5, 0), 0.8)], stroke=2.0, stiffness=0.5)
    obj = RigidCylinder(center=(13, 0, -4.6), radius=4.0)
    model = _Model(m, 0.45, dom.fixed, np.zeros(0, int), np.zeros(0), np.zeros(m.n_dof), tendons=[ten], contact=obj)
    rng = np.random.default_rng(2)
    xp = np.clip(0.6 + 0.3 * rng.standard_normal(m.n_elem), 0.05, 1.0)
    j, dj, u = nonlinear_response(model, xp, 26.0, None, steps=6)
    assert j > 0.01                     # it touches
    for e in (3, 20, 40, 60):
        h = 1e-5
        xa, xb = xp.copy(), xp.copy()
        xa[e] += h
        xb[e] -= h
        fd = (nonlinear_response(model, xa, 26.0, None, u0=u)[0] - nonlinear_response(model, xb, 26.0, None, u0=u)[0]) / (2 * h)
        assert abs(dj[e] - fd) <= 1e-4 * max(abs(fd), 1e-2 * np.abs(dj).max()), (e, dj[e], fd)


def test_grip_on_any_obstacle_shape_gradient_matches_finite_differences():
    # the same finger closing on a sphere (RigidObject over contact.RigidObstacle):
    # 3D normals and the sphere's curvature in the contact tangent
    from loadpath.benchmarks.mechanisms import MechProblem, mech_domain
    from loadpath.benchmarks.problems import _nodes
    from loadpath.contact import RigidObstacle
    from loadpath.mechanism_nl import RigidObject, ServoTendon, _Model, nonlinear_response
    from loadpath.nonlinear import nodes_near
    p = MechProblem(name="f", title="f", source="", dims=(16, 1, 4), volfrac=0.4, rmin=1.5,
                    fixed=[(_nodes(0, 0, range(0, 5)), "xz")], sym=[], inp=(_nodes(16, 0, range(0, 2)), "x", -1.0),
                    out=(_nodes(16, 0, range(0, 2)), "z", -1.0))
    dom = mech_domain(p)
    m = dom.mesh
    ten = ServoTendon(path=[np.array([-2.0, 0.5, 0.5]), nodes_near(m, (8, 0.5, 0), 0.8),
                            nodes_near(m, (16, 0.5, 0), 0.8)], stroke=2.0, stiffness=0.5)
    obj = RigidObject(RigidObstacle("sphere", center=(13, 0.5, -4.6), radius=4.0))
    model = _Model(m, 0.45, dom.fixed, np.zeros(0, int), np.zeros(0), np.zeros(m.n_dof), tendons=[ten], contact=obj)
    rng = np.random.default_rng(2)
    xp = np.clip(0.6 + 0.3 * rng.standard_normal(m.n_elem), 0.05, 1.0)
    j, dj, u = nonlinear_response(model, xp, 26.0, None, steps=6)
    assert j > 0.01
    for e in (3, 20, 40, 60):
        h = 1e-5
        xa, xb = xp.copy(), xp.copy()
        xa[e] += h
        xb[e] -= h
        fd = (nonlinear_response(model, xa, 26.0, None, u0=u)[0] - nonlinear_response(model, xb, 26.0, None, u0=u)[0]) / (2 * h)
        assert abs(dj[e] - fd) <= 1e-4 * max(abs(fd), 1e-2 * np.abs(dj).max()), (e, dj[e], fd)


def test_grip_with_a_stop_gradient_matches_finite_differences():
    # the finger gripping a sphere while it presses on a second object that
    # only resists (mechanism_nl.Contacts: the stop counts for nothing),
    # and a stop alone under the tip, where the tendon's knot nodes have a
    # held direction (their stiffness used to leak into the held rows and
    # threw the adjoint off at the tip cell)
    from loadpath.benchmarks.mechanisms import MechProblem, mech_domain
    from loadpath.benchmarks.problems import _nodes
    from loadpath.contact import RigidObstacle
    from loadpath.mechanism_nl import Contacts, RigidObject, ServoTendon, _Model, nonlinear_response
    from loadpath.nonlinear import nodes_near
    p = MechProblem(name="f", title="f", source="", dims=(16, 1, 4), volfrac=0.4, rmin=1.5,
                    fixed=[(_nodes(0, 0, range(0, 5)), "xz")], sym=[], inp=(_nodes(16, 0, range(0, 2)), "x", -1.0),
                    out=(_nodes(16, 0, range(0, 2)), "z", -1.0))
    dom = mech_domain(p)
    m = dom.mesh
    ten = ServoTendon(path=[np.array([-2.0, 0.5, 0.5]), nodes_near(m, (8, 0.5, 0), 0.8),
                            nodes_near(m, (16, 0.5, 0), 0.8)], stroke=2.0, stiffness=0.5)
    grip = RigidObstacle("sphere", center=(13, 0.5, -4.6), radius=4.0)
    for obj in (Contacts([grip, RigidObstacle("sphere", center=(8, 0.5, 7.4), radius=3.0)]),
                RigidObject(RigidObstacle("sphere", center=(15, 0.5, -2.75), radius=2.0))):
        model = _Model(m, 0.45, dom.fixed, np.zeros(0, int), np.zeros(0), np.zeros(m.n_dof), tendons=[ten],
                       contact=obj)
        rng = np.random.default_rng(2)
        xp = np.clip(0.6 + 0.3 * rng.standard_normal(m.n_elem), 0.05, 1.0)
        j, dj, u = nonlinear_response(model, xp, 26.0, None, steps=6)
        assert j > 0.01
        if isinstance(obj, Contacts):
            stop = obj.forces(u, model.w)[1]
            assert stop[2] > 0.05, stop          # the finger's back presses up on the stop
        for e in (3, 20, 40, 60):
            h = 1e-5
            xa, xb = xp.copy(), xp.copy()
            xa[e] += h
            xb[e] -= h
            fd = (nonlinear_response(model, xa, 26.0, None, u0=u)[0]
                  - nonlinear_response(model, xb, 26.0, None, u0=u)[0]) / (2 * h)
            assert abs(dj[e] - fd) <= 1e-4 * max(abs(fd), 1e-2 * np.abs(dj).max()), (type(obj).__name__, e, dj[e], fd)


def test_compiled_kernel_matches_numpy_element():
    # nlkernel (compiled, all cores) and the numpy element: same force,
    # tangent, bubble update and coupling, plain and incompatible modes
    import os

    from loadpath import nlkernel
    from loadpath.nonlinear import _assemble
    if not nlkernel.available():
        return
    ijk = np.array([(i, j, k) for i in range(6) for j in range(2) for k in range(3)])
    m = HexMesh(ijk, 0.7, np.zeros(3))
    rng = np.random.default_rng(3)
    u = 0.05 * rng.standard_normal(m.n_dof)
    mod = rng.uniform(1, 30, m.n_elem)
    al = 0.01 * rng.standard_normal((m.n_elem, 3, 3))
    st = Stiffness(m, NU, np.zeros(m.n_dof, bool), element="im")
    old = os.environ.get("LOADPATH_NLKERNEL")
    try:
        for kind in ("neo-hookean", "svk"):
            for a in (None, al):
                outs = []
                for env in ("1", "0"):
                    os.environ["LOADPATH_NLKERNEL"] = env
                    c = None if a is None else np.zeros((m.n_elem, 9, 24))
                    outs.append((_assemble(st, u, mod, NU, kind, alpha=a, coupling=c), c))
                (o1, c1), (o0, c0) = outs
                assert np.abs(o1[0] - o0[0]).max() < 1e-11 * np.abs(o0[0]).max()
                assert abs(o1[1] - o0[1]).max() < 1e-11 * abs(o0[1]).max()
                assert abs(o1[2] - o0[2]) < 1e-12
                if a is not None:
                    assert np.abs(o1[3] - o0[3]).max() < 1e-11 * np.abs(o0[3]).max()
                    assert abs(o1[4] - o0[4]) < 1e-9 * o0[4]
    finally:
        if old is None:
            os.environ.pop("LOADPATH_NLKERNEL", None)
        else:
            os.environ["LOADPATH_NLKERNEL"] = old


TESTS = [
    test_keep_best_returns_the_best_printed_iterate,
    test_compiled_kernel_matches_numpy_element,
    test_tangent_matches_finite_differences,
    test_small_load_matches_linear,
    test_cantilever_large_deflection_matches_elastica,
    test_stroke_control_reaction_matches_force,
    test_column_buckles_at_euler_and_bends_like_the_elastica,
    test_large_mesh_path_follows_a_perfect_column,
    test_tendon_curls_a_cantilever_into_an_arc,
    test_verify_mechanism_reports_a_finger,
    test_mechanism_optimizer_reports_nonlinear_check,
    test_stroke_mechanism_pushes_a_bar,
    test_robust_mechanism_has_no_corner_hinges,
    test_nonlinear_mechanism_gradient_matches_finite_differences,
    test_tendon_grip_gradient_matches_finite_differences,
    test_grip_with_a_stop_gradient_matches_finite_differences,
    test_grip_on_any_obstacle_shape_gradient_matches_finite_differences,
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
