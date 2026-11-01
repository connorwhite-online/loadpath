"""
Tests for flowtopo.py: the adjoint gradient through the flow and heat
solves against finite differences, on a small duct. Run:
`python3 tests/test_flowtopo.py`.
"""
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from loadpath.flowtopo import FlowSink, FlowTopoOptions, optimize_flow_thermal
from loadpath.heatx_thermal import FLUIDS


def _duct(n=(10, 5, 5)):
    z = np.arange(n[2])[None, None, :] * np.ones(n)
    x = np.arange(n[0])[:, None, None] * np.ones(n)
    solid = (z == 0) & (x >= 2) & (x < n[0] - 2)
    wall = (z == 0) & ~solid
    design = (z > 0) & (x >= 2) & (x < n[0] - 2)
    fluid = ~(solid | design | wall)
    heat = solid & (x >= 4) & (x < 6)
    return design, solid, fluid, heat


def test_adjoint_matches_finite_differences():
    design, solid, fluid, heat = _duct()
    fs = FlowSink(design, solid, fluid, heat, 1.0, 125.0, FLUIDS["air"], 20.0, 0.5)
    rng = np.random.default_rng(1)
    xd = rng.uniform(0.2, 0.8, fs.design.sum())
    r = fs.rho(xd)
    s = fs.solve(r)
    g = fs.gradient(s)
    J0 = fs.objective(s)
    cells = np.nonzero(fs.design)[0]
    for c in rng.choice(cells, 8, replace=False):
        e = 1e-6
        rp = r.copy()
        rp[c] += e
        rm = r.copy()
        rm[c] -= e
        fd = (fs.objective(fs.solve(rp)) - fs.objective(fs.solve(rm))) / (2 * e)
        assert abs(fd - g[c]) <= 1e-4 * abs(fd) + 1e-6 * abs(J0), (c, fd, g[c])


def test_optimizer_cools_and_holds_volume():
    design, solid, fluid, heat = _duct((14, 6, 6))
    fs = FlowSink(design, solid, fluid, heat, 1.0, 125.0, FLUIDS["air"], 20.0, 0.5)
    res = optimize_flow_thermal(fs, design, 1.0, FlowTopoOptions(0.25, filter_radius=1.5, max_iterations=60,
                                                                ratio_iterations=20))
    h = res["history"]
    grey = fs.objective(fs.solve(fs.rho(np.full(fs.design.sum(), 0.25))))
    assert h[-1]["objective"] < 0.5 * grey, (grey, h[-1])
    assert h[-1]["volume"] <= 0.26, h[-1]["volume"]


TESTS = [test_adjoint_matches_finite_differences, test_optimizer_cools_and_holds_volume]


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
