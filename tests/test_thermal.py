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


TESTS = [
    test_slab_with_generation_is_exact,
    test_convective_fin_matches_closed_form,
    test_composite_wall_and_face_flux,
    test_thermal_gradients_match_finite_differences,
    test_thermomechanical_gradient_matches_finite_differences,
    test_volume_to_point_grows_a_tree,
    test_heat_sink_beats_bare_plate_and_conserves_heat,
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
