"""
Tests for conjugate.py's heat solve (the flow side is flow.py's, tested
there). Run: `python3 tests/test_conjugate.py`.
"""
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from loadpath.conjugate import solve_heat
from loadpath.heatx_thermal import FLUIDS

AIR = FLUIDS["air"]


def test_plug_flow_over_a_heated_plate_balances():
    """A plate under a channel of plug flow: every watt leaves with the air,
    and the air's mean outlet rise is P / (m cp)."""
    n = (30, 6, 8)
    h = 1.0
    solid = np.zeros(n, bool)
    solid[:, :, :2] = True
    fluid = ~solid
    U = np.zeros((3,) + n)
    U[0][fluid] = 0.5
    heat = np.zeros(n, bool)
    heat[10:20, :, 0] = True
    r = solve_heat(solid, fluid, U, h, 125.0, AIR, heat, 5.0)
    assert abs(r["balance"] - 1.0) < 1e-6, r["balance"]
    m_cp = AIR.rho * AIR.cp * 0.5 * fluid[-1].sum() * (h * 1e-3) ** 2
    t_out = r["T"][-1][fluid[-1]].mean()
    assert abs(t_out - 5.0 / m_cp) < 1e-6 * (5.0 / m_cp), (t_out, 5.0 / m_cp)
    assert r["maxRise"] > t_out


def test_conduction_bar_matches_fourier():
    """A metal bar heated at one end, cooled by a fast stream at the other:
    the bar's drop is P L / (k A) plus what the stream takes."""
    n = (40, 4, 4)
    h = 1.0
    solid = np.zeros(n, bool)
    solid[:30] = True
    fluid = ~solid
    U = np.zeros((3,) + n)
    U[1][fluid] = 0.0
    U[0][fluid] = 50.0
    heat = np.zeros(n, bool)
    heat[0] = True
    k = 10.0
    r = solve_heat(solid, fluid, U, h, k, AIR, heat, 1.0)
    T = r["T"]
    drop = T[0].mean() - T[29].mean()
    want = 1.0 * 29e-3 / (k * 16e-6)
    assert abs(drop - want) < 1e-3 * want, (drop, want)


TESTS = [test_plug_flow_over_a_heated_plate_balances, test_conduction_bar_matches_fourier]


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
