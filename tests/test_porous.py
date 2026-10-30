"""Whole-exchanger porous simulation (porous.py) against exact answers:
the Darcy-Forchheimer law through a core, and counterflow and crossflow
effectiveness against epsilon-NTU (Mason's exact series for crossflow)."""
from math import exp, factorial

import numpy as np

from loadpath.heatx_thermal import FLUIDS
from loadpath.porous import PorousDomain, savoldi_medium, solve_flow, solve_heat, summarize

WATER = FLUIDS["water"]


def _block(nx, ny, nz, layout="counter", h=1e-3, cell=8e-3, phi=0.4):
    """A core n cells long, fluid A along x; B back along x (counter) or along y (cross)."""
    core = np.zeros((nx + 2, ny + 2, nz + 2), bool)
    core[1:-1, 1:-1, 1:-1] = True
    a = 3.09 / cell * 0.95
    med = savoldi_medium(phi, a, 4 * phi / a, cell)
    ends = []
    for ax, n in ((0, nx), (1, ny)):
        for k in (1, n):
            e = np.zeros_like(core)
            sl = [slice(1, -1)] * 3
            sl[ax] = k
            e[tuple(sl)] = True
            ends.append(e)
    a_in, a_out, b_in, b_out = ends if layout == "cross" else (ends[0], ends[1], ends[1], ends[0])
    # phi_sheet 1: no conduction along the wall, as epsilon-NTU assumes
    return PorousDomain(h=h, lo=np.zeros(3), core=core, cells=[core.copy(), core.copy()], inlet=[a_in, b_in],
                        outlet=[a_out, b_out], media=[med, med], wall=0.8e-3, phi_sheet=1.0)


def _run(dom, m, k_wall=125.0):
    flows = [solve_flow(dom, f, WATER, m) for f in (0, 1)]
    heat = solve_heat(dom, [WATER, WATER], flows, [80.0, 20.0], k_wall)
    s = summarize(dom, [WATER, WATER], flows, heat, [80.0, 20.0])
    w = heat["wallCells"]
    g = [dom.media[f].area_density * dom.h ** 3 / (1 / heat["h"][f][w] + 0.5 * dom.wall / k_wall) for f in (0, 1)]
    ntu = float((1 / (1 / g[0] + 1 / g[1])).sum()) / (m * WATER.cp)
    return s, ntu


def test_core_pressure_gradient_is_darcy_forchheimer():
    dom = _block(30, 5, 5)
    m = 2.0 / 60000 * WATER.rho
    fl = solve_flow(dom, 0, WATER, m)
    assert fl["converged"] and abs(fl["outflow"] / m - 1) < 1e-6
    p = fl["p"][1:-1, 3, 3]
    med = dom.media[0]
    us = m / WATER.rho / (5e-3 * 5e-3)
    ref = WATER.mu * us / med.K + WATER.rho * med.CF * us ** 2 / np.sqrt(med.K)
    assert abs(-(p[22] - p[7]) / 15e-3 / ref - 1) < 1e-6


def test_counterflow_matches_epsilon_ntu():
    dom = _block(60, 1, 1)
    s, ntu = _run(dom, 1.5e-6)
    exact = ntu / (1 + ntu)
    assert 1.0 < ntu < 3.0
    assert abs(s["effectiveness"] / exact - 1) < 0.005
    assert abs(s["energyBalance"]) < 1e-9


def test_crossflow_converges_to_the_exact_series():
    errs = []
    for n in (10, 20):
        s, ntu = _run(_block(n, n, 1, "cross"), 1e-5 * (n / 40) ** 2)

        def P(k, y):
            return 1 - exp(-y) * sum(y ** j / factorial(j) for j in range(k + 1))

        exact = sum(P(k, ntu) * P(k, ntu) for k in range(80)) / ntu
        errs.append(1 - s["effectiveness"] / exact)
    # second order: the error drops about 3-4x per halving of the voxel
    assert 0 < errs[1] < 0.05 and errs[0] / errs[1] > 2.5


def test_whole_exchanger_agrees_with_the_rating():
    from loadpath.heatx import HeatExchangerSpec
    from loadpath.porous import simulate_exchanger
    from loadpath.shapes import Box

    hx = HeatExchangerSpec(core=Box((0, 0, 0), (24, 24, 24)), cell=8, wall=1.0, layout="cross", step=0.45,
                           header=5, port_diameter=6)
    duty = {"A": {"fluid": "water", "lPerMin": 0.5, "tIn": 80}, "B": {"fluid": "water", "lPerMin": 0.5, "tIn": 20},
            "wall": "alsi10mg"}
    _, _, s = simulate_exchanger(hx, duty, voxel=2.0)
    assert abs(s["energyBalance"]) < 1e-6
    assert abs(s["effectiveness"] / s["rating"]["effectiveness"] - 1) < 0.15
    for n, f in s["fluids"].items():
        assert f["converged"] and abs(f["massBalance"] - 1) < 1e-4
        assert 0.5 < f["entry"]["uniformity"] <= 1.0
        assert 0 < f["pressureDropCore"] < f["pressureDrop"]
    assert 20 < s["wall"]["min"] < s["wall"]["max"] < 80
