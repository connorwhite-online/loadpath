"""
Tests for loadpath.thermal: the conduction solver against closed-form
answers, the adjoint gradients against finite differences, and small
optimizations that must land on the known answer. Runs in about half a
minute.

Run: `python3 tests/test_thermal.py`.
"""
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from loadpath import thermal as T
from loadpath.domain import build_domain
from loadpath.shapes import Box
from loadpath.spec import study_from_dict


def test_slab_with_generation_is_exact():
    """Heat generated uniformly in a slab held at 25 C on both ends: the peak
    rise is q L^2 / 8k, and linear elements get the nodes exactly."""
    L, k, q = 40.0, 0.2, 2.0
    g = T.thermal_grid(Box([0, 0, 0], [L, 4, 4]), 1.0)
    bc = T.thermal_bc(g.mesh, ambient=25, temperatures=[(Box([-1, -1, -1], [0, 5, 5]), 25),
                                                        (Box([L, -1, -1], [L + 1, 5, 5]), 25)],
                      heat=[(np.ones(g.mesh.n_elem, bool), q, "volume")])
    r = T.solve_temperature(g.mesh, k, bc)
    analytic = q / (L * 16) * L ** 2 / (8 * k * 1e-3)
    assert abs(r.max_t - 25 - analytic) < 1e-6 * analytic, (r.max_t, analytic)
    assert abs(r.heat_out_fixed - q) < 1e-6


def test_convective_fin_matches_closed_form():
    """A square aluminium pin, base at 75 C, convecting from its sides with an
    insulated tip: theta(L) = theta_b / cosh(mL), q = sqrt(hPkA) theta_b tanh(mL)."""
    L, a, h, k, tb = 60.0, 4.0, 50.0, 125.0, 75.0
    g = T.thermal_grid(Box([0, 0, 0], [L, a, a]), 1.0)
    bc = T.thermal_bc(g.mesh, ambient=25, temperatures=[(Box([-1, -1, -1], [0, a + 1, a + 1]), tb)],
                      convection=[T.Convection(h, where=lambda c, n: np.abs(n[:, 0]) < 0.5)])
    r = T.solve_temperature(g.mesh, k, bc)
    m = np.sqrt(4 * h * 1e-6 / (k * 1e-3 * a))
    tip = r.temperature[np.isclose(g.mesh.node_xyz[:, 0], L)].mean()
    assert abs(tip - (25 + (tb - 25) / np.cosh(m * L))) < 0.02, tip
    q = np.sqrt(h * 1e-6 * 4 * a * k * 1e-3 * a * a) * (tb - 25) * np.tanh(m * L)
    assert abs(-r.heat_out_fixed - q) < 1e-3 * q
    assert abs(r.heat_out_convection - q) < 1e-3 * q        # what goes in comes out


def test_composite_wall_and_face_flux():
    """1 W through 20 mm of PETG then 20 mm of aluminium: the rise is the sum
    of the two resistances, with the flux spread over the face by area."""
    g = T.thermal_grid(Box([0, 0, 0], [40, 5, 5]), 1.0)
    k = np.where(g.mesh.elem_centers()[:, 0] < 20, 0.2, 125.0)
    bc = T.thermal_bc(g.mesh, ambient=25, temperatures=[(Box([40, -1, -1], [41, 6, 6]), 25)],
                      heat=[(Box([-1, -1, -1], [0, 6, 6]), 1.0)])
    r = T.solve_temperature(g.mesh, k, bc)
    hot = r.temperature[np.isclose(g.mesh.node_xyz[:, 0], 0)]
    analytic = 20 / (0.2e-3 * 25) + 20 / (125e-3 * 25)
    assert np.abs(hot - 25 - analytic).max() < 1e-6 * analytic


def _fd_check(fun, grad, rho, cells, rel=1e-4):
    for i in cells:
        e = 1e-6
        r2 = rho.copy()
        r2[i] += e
        jp = fun(r2)
        r2[i] -= 2 * e
        jm = fun(r2)
        fd = (jp - jm) / (2 * e)
        assert abs(fd - grad[i]) <= rel * max(abs(fd), 1e-3 * np.abs(grad).max()), (i, fd, grad[i])


def test_thermal_gradients_match_finite_differences():
    """Design-dependent convection (interfaces and porous grey cells), a hot
    fixed temperature, fluids off ambient and a target region: the adjoint
    gradient is the derivative of the objective."""
    g = T.thermal_grid(Box([0, 0, 0], [8, 6, 4]), 1.0)
    mesh = g.mesh
    bc = T.thermal_bc(mesh, ambient=25, temperatures=[(Box([-1, -1, -1], [0, 7, 5]), 60)],
                      heat=[(Box([7.9, 2, 1], [9, 4, 3]), 0.3)],
                      convection=[T.Convection(20.0, fluid_t=40, where=lambda c, n: c[:, 1] > 3),
                                  T.Convection(30.0, where=lambda c, n: c[:, 1] <= 3)])
    th = T._Thermal(g, bc, 0.2, mesh.node_xyz[:, 2] > 3.5, porous=1.5)
    th.solver.rtol = 1e-12
    rho = np.random.default_rng(0).uniform(0.1, 1, mesh.n_elem)

    def fun(r):
        th.solve(r)
        return th.objective()

    fun(rho)
    grad = th.objective_gradient(rho)
    _fd_check(fun, grad, rho, [107, 30, 133, 157])


def test_thermomechanical_gradient_matches_finite_differences():
    """Compliance of a beam softened by a hot end, plus a temperature term:
    the thermal adjoint carries the softening into the gradient."""
    beam = {
        "designSpace": {"type": "box", "min": [0, 0, 0], "max": [16, 6, 4]},
        "supports": [{"shape": {"type": "box", "min": [-1, -1, -1], "max": [0.01, 11, 11]}}],
        "loadCases": [{"name": "tip", "loads": [
            {"shape": {"type": "box", "min": [15.99, -1, -1], "max": [17, 11, 11]}, "force": [0, 0, -10]}]}],
        "material": "petg", "goal": {"volumeFraction": 0.5}, "resolution": {"pitch": 1.0},
    }
    d = build_domain(study_from_dict(beam))
    mesh = d.mesh
    bc = T.thermal_bc(mesh, ambient=25, temperatures=[(Box([15.5, -1, -1], [17, 7, 5]), 75)],
                      convection=[T.Convection(10.0)])
    opts = T.ThermalOptions(volume_fraction=0.5, filter_radius=1.5, k=0.2, target=mesh.node_xyz[:, 0] > 8)
    ev, fe, th = T._thermomech_evaluator(d, 0.38, bc, opts, "petg", 0.5, rtol=1e-11)
    th.solver.rtol = 1e-12
    rho = np.random.default_rng(1).uniform(0.3, 1, mesh.n_elem)
    _, grad, rec = ev(rho)
    assert rec["minRetention"] < 0.5                          # the hot end really is soft
    _fd_check(lambda r: ev(r)[0], grad, rho, [183, 164, 75, 248])


def test_volume_to_point_grows_a_tree():
    """The classic volume-to-point problem: heat generated everywhere in a
    plate, one small sink. 30% conductor must cut the mean rise of the solid
    plate's 30%-conductivity equivalent by far, and reach every corner."""
    g = T.thermal_grid(Box([0, 0, 0], [40, 40, 1]), 1.0, symmetry="x")
    m = g.mesh
    bc = T.thermal_bc(m, ambient=25, temperatures=[(Box([18, -1, -1], [22, 0, 2]), 25)],
                      heat=[(np.ones(m.n_elem, bool), 1.0, "volume")])
    o = T.ThermalOptions(volume_fraction=0.3, filter_radius=1.5, k=0.2, max_iterations=80, extrusion=2)
    r = T.optimize_thermal(g, bc, o)
    assert abs(r.volume_fraction - 0.3) < 0.02
    # the same material smeared uniformly (k * 0.3^3 under SIMP) is the
    # fair "no layout" reference; a tree should beat it many times over
    grey = T.solve_temperature(m, 0.2, bc, density=np.full(m.n_elem, 0.3))
    assert r.objective < 0.2 * (grey.resistance * bc.power), (r.objective, grey.resistance)
    assert r.converged


def test_heat_sink_beats_bare_plate_and_conserves_heat():
    """A small convective sink on a die: the optimized fins must take most of
    the rise out of the bare plate's, and all 2 W must leave by convection."""
    g = T.thermal_grid(Box([-10, -10, 0], [10, 10, 10]), 1.0, preserve=[Box([-10, -10, 0], [10, 10, 1])],
                       symmetry="xy")
    m = g.mesh
    skin = lambda c, n: ~((n[:, 2] < -0.5) & (c[:, 2] < 0.1))
    bc = T.thermal_bc(m, ambient=25, heat=[(Box([-3, -3, -1], [3, 3, 0]), 2.0)],
                      convection=[T.Convection(25.0, where=skin)])
    o = T.ThermalOptions(volume_fraction=0.15, filter_radius=1.6, k=125.0, max_iterations=60)
    r = T.optimize_thermal(g, bc, o)
    bare = T.solve_temperature(m, 125.0, bc, density=g.preserve.astype(float))
    bare_rise = bare.mean_t(bc.heat_nodes) - 25
    assert r.objective < 0.6 * bare_rise, (r.objective, bare_rise)
    assert abs(r.temperature.heat_out_convection - 2.0) < 1e-3


def test_thermal_expansion_matches_closed_forms():
    """The expansion load (fem.thermal_load) against two closed forms: a
    bar held between rollers carries -E alpha dT exactly and grows sideways
    by (1 + nu) alpha dT; an aluminium-on-PETG bimetal cantilever bends to
    Timoshenko's curvature (within 2.5% at 4 cells per layer: hex8 is a
    little stiff in bending)."""
    import scipy.sparse.linalg as spl

    from loadpath.fem import HexMesh, Stiffness, thermal_load

    def box(n, h):
        return HexMesh(np.stack(np.meshgrid(*[np.arange(k) for k in n], indexing="ij"), -1).reshape(-1, 3), h,
                       np.zeros(3))

    E, nu, a, dT = 2000.0, 0.38, 68e-6, 35.0
    m = box((20, 3, 3), 1.0)
    X = m.node_xyz
    fixed = np.zeros(m.n_dof, bool)
    fixed[3 * np.flatnonzero((X[:, 0] < 1e-9) | (X[:, 0] > 20 - 1e-9))] = True
    fixed[3 * np.flatnonzero(X[:, 1] < 1e-9) + 1] = True
    fixed[3 * np.flatnonzero(X[:, 2] < 1e-9) + 2] = True
    st = Stiffness(m, nu, fixed)
    u = spl.spsolve(st.assemble(np.full(m.n_elem, E)).tocsc(), thermal_load(m, nu, E, a * dT, fixed))
    s = st.stress(u, E, a * dT)
    assert np.allclose(s[:, 0], -E * a * dT, rtol=1e-9) and np.abs(s[:, 1:]).max() < 1e-9
    assert abs(u[3 * np.flatnonzero(X[:, 1] > 3 - 1e-9) + 1].mean() / ((1 + nu) * a * dT * 3) - 1) < 1e-9

    t, L, E1, E2, a1, a2, dT = 1.0, 40.0, 70e3, 2000.0, 23e-6, 68e-6, 50.0
    h = t / 4
    m = box((int(L / h), 4, 8), h)
    X = m.node_xyz
    top = m.elem_centers()[:, 2] > t
    Ee, eps = np.where(top, E2, E1), np.where(top, a2, a1) * dT
    fixed = np.zeros(m.n_dof, bool)
    fixed[(3 * np.flatnonzero(X[:, 0] < 1e-9)[:, None] + np.arange(3)).ravel()] = True
    st = Stiffness(m, 0.3, fixed)
    u = spl.spsolve(st.assemble(Ee).tocsc(), thermal_load(m, 0.3, Ee, eps, fixed))
    tip = u[3 * np.flatnonzero(X[:, 0] > L - 1e-9) + 2].mean()
    n = E1 / E2
    kappa = 6 * (a2 - a1) * dT * 4 / (2 * t * (3 * 4 + (1 + n) * (1 + 1 / n)))
    assert abs(tip / (-kappa * L * L / 2) - 1) < 0.025, (tip, -kappa * L * L / 2)


def test_expansion_gradient_matches_finite_differences():
    """The thermo-mechanical gradient with thermal expansion on: the
    objective (twice the elastic strain energy) has a load that depends on
    density and on the design's own temperature (stiffness times
    alpha (T - t_ref)), so it takes a mechanical adjoint as well as the
    thermal one, and both paths are in the sensitivity."""
    beam = {
        "designSpace": {"type": "box", "min": [0, 0, 0], "max": [16, 6, 4]},
        "supports": [{"shape": {"type": "box", "min": [-1, -1, -1], "max": [0.01, 11, 11]}},
                     {"shape": {"type": "box", "min": [15.99, -1, -1], "max": [17, 11, 11]}}],
        "loadCases": [{"name": "mid", "loads": [
            {"shape": {"type": "box", "min": [7.5, -1, 3.5], "max": [8.5, 11, 5]}, "force": [0, 0, -10]}]}],
        "material": "petg", "goal": {"volumeFraction": 0.5}, "resolution": {"pitch": 1.0},
    }
    d = build_domain(study_from_dict(beam))
    mesh = d.mesh
    bc = T.thermal_bc(mesh, ambient=25, temperatures=[(Box([15.5, -1, -1], [17, 7, 5]), 60)],
                      convection=[T.Convection(10.0)])
    opts = T.ThermalOptions(volume_fraction=0.5, filter_radius=1.5, k=0.2, target=mesh.node_xyz[:, 0] > 8)
    ev, fe, th = T._thermomech_evaluator(d, 0.38, bc, opts, "petg", 0.0, rtol=1e-12, alpha=True)
    th.solver.rtol = 1e-13
    rho = np.random.default_rng(2).uniform(0.3, 1, mesh.n_elem)
    _, grad, rec = ev(rho)
    # (rel 1e-3: PETG's retention curve is piecewise linear, and a cell
    # near a knee sees the kink in a central difference)
    _fd_check(lambda r: ev(r)[0], grad, rho, [183, 164, 75, 248, 300], rel=1e-3)


def _hot_study(thermal, **extra):
    d = {"designSpace": {"type": "box", "min": [0, 0, 0], "max": [24, 8, 8]},
         "keepOut": [{"id": "motor", "shape": {"type": "box", "min": [20, 2, -1], "max": [25, 6, 9]},
                      "bore": False}],
         "supports": [{"shape": {"type": "box", "min": [-1, -1, -1], "max": [0.5, 9, 9]}}],
         "loads": [{"shape": {"type": "box", "min": [23, -1, -1], "max": [25, 9, 9]}, "force": [0, 0, -20]}],
         "material": "petg", "resolution": {"pitch": 2.0}, "thermal": thermal, **extra}
    return study_from_dict(d)


def test_study_thermal_block_parses_and_lints():
    """The study JSON's thermal block: schema and parser agree, defaults
    fill in (still-air convection, stress-free at ambient, card values), and
    the mistakes an agent makes get a message naming the field."""
    from loadpath.schema import lint

    th = {"ambient": 30, "temperatures": [{"region": "motor", "t": 70}],
          "heat": [{"region": "motor", "watts": 2, "spread": "volume"}]}
    st = _hot_study(th)
    assert st.thermal.t_ref == 30 and st.thermal.convection == [(None, None, 8.0, None)]
    assert st.thermal.heat[0][2:] == (2.0, "volume") and st.thermal.expansion is True
    doc = {"designSpace": {"type": "box", "min": [0, 0, 0], "max": [1, 1, 1]}, "thermal": th}
    assert not [i for i in lint(doc) if "thermal" in i["path"]]
    bad = {**doc, "thermal": {**th, "convecton": 5}}
    assert any("thermal" in i["path"] for i in lint(bad))
    for wrong, msg in [({"temperatures": [{"region": "motor"}]}, "t (C)"),
                       ({"heat": [{"region": "motor", "watts": 1}], "convection": 0}, "way for heat out"),
                       ({"heat": [{"region": "nope", "watts": 1}]}, None),
                       ({"expansion": "yes", "temperatures": [{"region": "motor", "t": 50}]}, "expansion"),
                       ({"temperatures": [{"region": "motor", "t": 50}], "cycle": {"period": 60, "on": 30}},
                        "give thermal.heat"),
                       ({"heat": [{"region": "motor", "watts": 1}], "cycle": {"period": 60, "on": 60}},
                        "shorter than the period")]:
        try:
            st = _hot_study(wrong)
            build_domain(st)
        except ValueError as e:
            assert msg is None or msg in str(e), (wrong, e)
        else:
            raise AssertionError(f"accepted {wrong}")


def test_study_thermal_gradient_matches_finite_differences():
    """simp's objective with a thermal block (thermal.StudyHeat): softening,
    expansion and self-weight together, each through density and through
    the design's own temperature."""
    from loadpath.domain import gravity_force
    from loadpath.simp import E_MIN, PENALTY, _Analysis, _heat_state

    st = _hot_study({"temperatures": [{"region": "motor", "t": 70}], "convection": 10}, gravity=[0, 0, -9.81e3])
    d = build_domain(st)
    heat = d.thermal
    heat.temperature()
    heat.th.solver.rtol = 1e-13
    fe = _Analysis(d, st.elastic(design=True), 1e-12)
    w = np.asarray(d.case_weights, float)

    def run(x):
        fg = gravity_force(d.mesh, x, d.gravity, d.fixed)
        fg, fth = _heat_state(heat, x, fg)
        loads = [f + fg for f in d.loads]
        us = fe.solve((E_MIN + x ** PENALTY * (1 - E_MIN)) * heat.g, x, [None], rtol=1e-12, loads=loads)
        return heat.sensitivities(fe, x, loads, us, fth, w, d.gravity, {})

    rho = np.random.default_rng(1).uniform(0.3, 1, d.n_elem)
    grad = run(rho)[2]
    _fd_check(lambda r: run(r)[0], grad, rho, [3, 171, 85, 33, 104], rel=1e-4)


def test_study_thermal_with_unknown_direction():
    """Heat with a force of unknown direction (a cone): the worst direction
    of twice the elastic strain energy is still a 3x3 problem, and with the
    soft maximum sharpened to the plain maximum the gradient is the
    derivative of the worst case."""
    from loadpath.directions import expand
    from loadpath.simp import E_MIN, PENALTY, _Analysis, _heat_state

    st = _hot_study({"temperatures": [{"region": "motor", "t": 70}], "convection": 10},
                    loads=[{"shape": {"type": "box", "min": [23, -1, -1], "max": [25, 9, 9]},
                            "force": [0, 0, -20], "cone": 60}],
                    supports=[{"shape": {"type": "box", "min": [-1, -1, -1], "max": [0.5, 9, 9]}},
                              {"shape": {"type": "box", "min": [-1, -1, -1], "max": [25, 9, 0.5]}, "fix": "z"}])
    d = build_domain(st)
    heat = d.thermal
    heat.temperature()
    heat.th.solver.rtol = 1e-13
    fe = _Analysis(d, st.elastic(design=True), 1e-12)
    w = np.asarray(d.case_weights, float)

    def run(x, smooth=1e6):
        fg, fth = _heat_state(heat, x, None)
        rhs, layout = expand(d.loads, d.spreads, fg)
        us = fe.solve((E_MIN + x ** PENALTY * (1 - E_MIN)) * heat.g, x, [None] * len(rhs), rtol=1e-12, loads=rhs)
        return heat.sensitivities(fe, x, rhs, us, fth, w, None, {}, layout, d.spreads, smooth=smooth)

    rho = np.random.default_rng(3).uniform(0.3, 1, d.n_elem)
    _, _, grad, dirs = run(rho)
    assert dirs[0] is not None and dirs[0] @ np.array([0, 0, -1.0]) >= 0.5 - 1e-9
    _fd_check(lambda r: run(r)[0], grad, rho, [3, 171, 85, 33, 104], rel=1e-4)


def test_verify_thermal_stress_matches_a_held_bar():
    """Verification with a thermal block: a PETG bar at 45 C between two
    walls that stop it growing (rollers, free sideways) carries
    sigma = -E g(45 C) alpha (45 - 25) exactly, and its safety factor is on
    strength derated as the modulus is."""
    import trimesh

    from loadpath.verify import verify_part

    L = 60
    big = {"type": "box", "min": [-1, -1, -1], "max": [L + 1, 11, 11]}
    st = study_from_dict({
        "designSpace": {"type": "box", "min": [0, 0, 0], "max": [L, 10, 10]},
        "supports": [{"shape": {"type": "box", "min": [-1, -1, -1], "max": [0.3, 11, 11]}, "fix": "x"},
                     {"shape": {"type": "box", "min": [L - 0.3, -1, -1], "max": [L + 1, 11, 11]}, "fix": "x"},
                     {"shape": {"type": "box", "min": [-1, 4.5, 4.5], "max": [0.3, 5.5, 5.5]}, "fix": "yz"}],
        "loads": [{"shape": {"type": "box", "min": [L / 2 - 1, -1, 9.5], "max": [L / 2 + 1, 11, 11]},
                   "force": [0, 0, -1e-3]}],
        "material": "petg", "thermal": {"temperatures": [{"shape": big, "t": 45}], "convection": 0}})
    m = trimesh.creation.box(extents=[L, 10, 10])
    m.apply_translation([L / 2, 5, 5])
    r = verify_part(st, m, max_cells=20000, refine=0, fields=True)
    g = T.retention("petg", 45.0)[0]
    exact = st.material.E * g * T.expansion("petg") * 20
    x, vm = r["fields"]["nodes"], r["fields"]["cases"][0]["vonMises"]
    mid = np.abs(x[:, 0] - 20) < 1
    assert np.allclose(vm[mid], exact, rtol=1e-3), (vm[mid].min(), vm[mid].max(), exact)
    c = r["cases"][0]
    assert r["thermal"]["maxTemperature"] == 45 and abs(c["strengthRetention"] - g) < 1e-9
    assert abs(c["safetyFactor"] * c["peakStress"] * r["stressAllowance"] - g * st.material.strength) < 1e-6


def test_verify_thermal_with_unknown_direction():
    """Verification of a hot part under a force of any direction: the worst
    direction found from three unit solves (stress over strength retention,
    the expansion in the fixed part) is at least as bad as every sampled
    direction solved on its own."""
    import trimesh

    from loadpath.directions import fibonacci_sphere
    from loadpath.verify import verify_part

    L = 40

    def study(load):
        return study_from_dict({
            "designSpace": {"type": "box", "min": [0, 0, 0], "max": [L, 10, 10]},
            "supports": [{"shape": {"type": "box", "min": [-1, -1, -1], "max": [0.3, 11, 11]}}],
            "loads": [{"shape": {"type": "box", "min": [L - 0.3, -1, -1], "max": [L + 1, 11, 11]}, **load}],
            "material": "petg",
            "thermal": {"temperatures": [{"shape": {"type": "box", "min": [L - 3, -1, -1], "max": [L + 1, 11, 11]},
                                          "t": 75}], "convection": 10}})

    m = trimesh.creation.box(extents=[L, 10, 10])
    m.apply_translation([L / 2, 5, 5])
    c = verify_part(study({"force": [0, 0, -30], "direction": "any"}), m, max_cells=8000, refine=0)["cases"][0]
    for d in list(fibonacci_sphere(12)) + [np.array(c["worstDirection"])]:
        sf = verify_part(study({"force": list(30 * d)}), m, max_cells=8000, refine=0)["safetyFactor"]
        assert sf >= c["safetyFactor"] * (1 - 1e-6), (d, sf, c["safetyFactor"])


def test_transient_bar_matches_series_solution():
    """A PETG bar at 0 C whose end is held at 1 C from t = 0, insulated
    elsewhere: the backward-Euler, lumped-capacity solve against the Fourier
    series, first order in the step (0.3% at 50 steps, 0.07% at 200)."""
    from loadpath.fem import HexMesh

    n, L = 40, 40.0
    m = HexMesh(np.stack([np.arange(n), np.zeros(n, int), np.zeros(n, int)], 1), 1.0, np.zeros(3))
    x = m.node_xyz[:, 0]
    bc = T.thermal_bc(m, ambient=0.0, temperatures=[(x < 1e-9, 1.0)])
    k, rc = 0.2, 1.27e-6 * T.specific_heat("petg")
    a = k * 1e-3 / rc
    t = 0.3 * L * L / a
    exact = 1 - sum(4 / ((2 * j + 1) * np.pi) * np.sin((2 * j + 1) * np.pi * x / (2 * L))
                    * np.exp(-((2 * j + 1) * np.pi / (2 * L)) ** 2 * a * t) for j in range(200))
    errs = [np.abs(T.transient(m, k, bc, T.node_capacity(m, rc), [t / s] * s)[-1] - exact).max() for s in (50, 200)]
    assert errs[0] < 4e-3 and errs[1] < 1e-3 and 3 < errs[0] / errs[1] < 5, errs


def test_duty_cycle_of_a_lumped_block_matches_closed_form():
    """A block that conducts far better than it convects, heated by a square
    wave: its repeating cycle swings between
    T_max = P/(hA) (1 - e^(-t_on/tau)) / (1 - e^(-period/tau)) and
    T_max e^(-t_off/tau), tau = C/(hA)."""
    from loadpath.fem import HexMesh

    n = 4
    ii, jj, kk = np.meshgrid(*[np.arange(n)] * 3, indexing="ij")
    m = HexMesh(np.stack([ii.ravel(), jj.ravel(), kk.ravel()], 1), 5.0, np.zeros(3))
    bc = T.thermal_bc(m, ambient=25, heat=[(np.ones(m.n_elem, bool), 2.0, "volume")],
                      convection=[T.Convection(10.0)])
    cap = T.node_capacity(m, 1.27e-6 * 1200)
    ha = 10e-6 * 6 * 20.0 ** 2
    tau = cap.sum() / ha
    cy = T.Cycle(period=0.5 * tau, on=0.15 * tau, steps=400)
    r = T.solve_cycle(m, 1e4, bc, cap, cy)
    tmax = 2.0 / ha * (1 - np.exp(-cy.on / tau)) / (1 - np.exp(-cy.period / tau))
    tmin = tmax * np.exp(-(cy.period - cy.on) / tau)
    assert r.converged and abs((r.node_max.max() - 25) / tmax - 1) < 5e-3, (r.node_max.max() - 25, tmax)
    assert abs((r.node_min.min() - 25) / tmin - 1) < 5e-3, (r.node_min.min() - 25, tmin)
    assert abs(r.t_hot - cy.on) < 1e-9 and abs(r.t_cold - cy.period) < 1e-9


def test_duty_cycle_gradient_and_verification():
    """A study with a duty cycle: its optimizer temperature is the steady
    rise scaled node by node to the cycle's hot moment, and with that scale
    held the gradient is exact; verification reports hot and cold safety
    factors and thermal-cycling fatigue."""
    import trimesh

    from loadpath.simp import E_MIN, PENALTY, _Analysis, _heat_state
    from loadpath.verify import verify_part

    th = {"heat": [{"region": "motor", "watts": 0.5}], "convection": 10,
          "temperatures": [{"shape": {"type": "box", "min": [-1, -1, -1], "max": [0.5, 9, 9]}, "t": 25}],
          "cycle": {"period": 600, "on": 200, "cycles": 1e4, "steps": 16}}
    st = _hot_study(th)
    d = build_domain(st)
    heat = d.thermal
    heat.REFRESH = 10 ** 9
    heat.temperature()
    assert heat.r.min() >= 0 and heat.r.max() <= 1
    assert np.allclose(heat.t_node, heat.cycle_result.hot)
    heat.th.solver.rtol = 1e-13
    fe = _Analysis(d, st.elastic(design=True), 1e-12)
    w = np.asarray(d.case_weights, float)

    def run(x):
        fg, fth = _heat_state(heat, x, None, reuse=True)
        loads = [f + fg for f in d.loads]
        us = fe.solve((E_MIN + x ** PENALTY * (1 - E_MIN)) * heat.g, x, [None], rtol=1e-12, loads=loads)
        return heat.sensitivities(fe, x, loads, us, fth, w, None, {})

    rho = np.random.default_rng(4).uniform(0.3, 1, d.n_elem)
    _fd_check(lambda r: run(r)[0], run(rho)[2], rho, [3, 171, 85, 33, 104], rel=1e-4)

    m = trimesh.creation.box(extents=[24, 8, 8])
    m.apply_translation([12, 4, 4])
    rep = verify_part(st, m, max_cells=6000, refine=0)
    t = rep["thermal"]
    assert t["periods"] >= 1 and t["converged"] and t["maxTemperature"] > t["meanHot"] > t["meanCold"] >= 25 - 1e-9
    c = rep["cases"][0]
    assert c["safetyFactor"] <= c["coldSafetyFactor"] + 1e-12 or c["safetyFactor"] == c["coldSafetyFactor"]
    assert c["cycleStress"] > 0
    tc = rep["durability"]["thermalCycling"]
    assert tc["cycles"] == 1e4 and abs(tc["damage"] - 1e4 / tc["life"]) < 1e-12


TESTS = [
    test_slab_with_generation_is_exact,
    test_convective_fin_matches_closed_form,
    test_composite_wall_and_face_flux,
    test_thermal_gradients_match_finite_differences,
    test_thermomechanical_gradient_matches_finite_differences,
    test_volume_to_point_grows_a_tree,
    test_heat_sink_beats_bare_plate_and_conserves_heat,
    test_thermal_expansion_matches_closed_forms,
    test_expansion_gradient_matches_finite_differences,
    test_study_thermal_block_parses_and_lints,
    test_study_thermal_gradient_matches_finite_differences,
    test_study_thermal_with_unknown_direction,
    test_verify_thermal_stress_matches_a_held_bar,
    test_verify_thermal_with_unknown_direction,
    test_transient_bar_matches_series_solution,
    test_duty_cycle_of_a_lumped_block_matches_closed_form,
    test_duty_cycle_gradient_and_verification,
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
