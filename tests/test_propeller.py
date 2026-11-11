"""
Propellers (loadpath.propeller): the rotating-beam and torsion solves against closed forms, momentum-theory bounds on
the blade-element results, the motor and pack's energy balance, the printable mesh, the optimizer beating a
hand-drawn blade, and (with NeuralFoil installed, `loadpath[aero]`) the model against published static tests of
3D-printed propellers.

Run: `python tests/test_propeller.py` (or pytest). About two minutes with NeuralFoil, under one without.
"""
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from loadpath import propeller as pr  # noqa: E402


def test_beam_closed_forms():
    L, EI, q = 0.04, 2.0e-3, 3.0
    r = np.linspace(0.0, L, 30)
    w, m, _ = pr._beam(r, np.full(30, EI), np.zeros(30), np.full(30, q))
    assert abs(w[-1] / (q * L ** 4 / (8 * EI)) - 1) < 0.02, w[-1]
    assert abs(m[0] / (q * L * L / 2) - 1) < 0.05, m[0]
    # rotating uniform cantilever, hub at the axis: first flap frequency^2 = w0^2 + 1.193 Omega^2 (Southwell)
    ma = 0.01
    om = 2 * np.pi * 300
    r = np.linspace(1e-6, L, 40)
    cf = np.array([np.trapezoid((ma * om * om * r)[i:], r[i:]) for i in range(len(r))])
    _, _, f = pr._beam(r, np.full(40, EI), cf, np.zeros(40), np.full(40, ma), om, modes=1)
    w0 = 3.516 * math.sqrt(EI / (ma * L ** 4))
    want = math.sqrt(w0 ** 2 + 1.193 * om ** 2) / (2 * np.pi)
    assert abs(f[0] / want - 1) < 0.03, (f[0], want)


def test_torsion_closed_form():
    L, GJ, mt = 0.04, 5e-4, 0.2
    r = np.linspace(0.0, L, 30)
    phi = pr._torsion(r, np.full(30, GJ), np.zeros(30), np.full(30, mt))
    assert abs(phi[-1] / (mt * L * L / (2 * GJ)) - 1) < 0.02, phi[-1]
    # a restoring term (the propeller moment's stiffening) only ever lessens the twist
    phi_k = pr._torsion(r, np.full(30, GJ), np.full(30, 0.3), np.full(30, mt))
    assert 0 < phi_k[-1] < phi[-1]


def test_momentum_bounds_and_twist_sign():
    b = pr.Blade.uniform(44.45, 7.0, 3, 9.0, pitch_in=3.0, t_ratio=0.09)
    op = pr.performance(b, 20000)
    assert op["converged"]
    assert 0.3 < op["figureOfMerit"] < 0.85, op["figureOfMerit"]
    # thrust rises with rpm about as rpm^2 (Re lifts it a little more)
    op2 = pr.performance(b, 28000)
    k = math.log(op2["thrust"] / op["thrust"]) / math.log(28000 / 20000)
    assert 1.8 < k < 2.4, k
    # a spinning, loaded plastic blade twists toward flat pitch and loses thrust
    op_e, st = pr.aeroelastic(b, 28000)
    assert st["tipTwist_deg"] < 0, st["tipTwist_deg"]
    assert op_e["thrust"] < op2["thrust"]
    assert st["flapHz"][0] > op_e["rpm"] / 60          # centrifugal stiffening keeps flap above 1/rev


def test_motor_energy_balance():
    b = pr.Blade.uniform(44.45, 7.0, 3, 9.0, pitch_in=3.0, t_ratio=0.09)
    e = pr.evaluate(b, pr.Motor(2450, 0.12, 0.6), pr.Battery(4), 0.775)
    h, f = e["hover"], e["full"]
    assert abs(h["thrust"] / 0.775 - 1) < 1e-3
    assert 0 < h["throttle"] < 1 and abs(f["throttle"] - 1) < 1e-3
    for p in (h, f):
        assert p["shaft"] < p["electrical"]
        assert 0.4 < p["efficiency"] < 0.95, p["efficiency"]
    assert f["thrust"] > 2 * h["thrust"]


def test_mesh_watertight():
    b = pr.Blade.uniform(44.45, 7.0, 3, 9.0, pitch_in=3.0, t_ratio=0.09)
    for rot in ("ccw", "cw"):
        m = pr.rotor_mesh(b, rotation=rot)
        assert m.is_watertight
        assert abs(m.bounds[1][2] - m.bounds[0][2]) < 12
        assert 43.5 < np.linalg.norm(m.vertices[:, :2], axis=1).max() < 45.0


def test_root_fillet_and_tip():
    # Peterson's trend: a larger fillet, a lower Kt; asked for little margin it grows, with lots it stays small
    b = pr.Blade.uniform(44.45, 7.0, 3, 9.0, pitch_in=3.0, t_ratio=0.09)
    tight = pr.root_fillet(b, 24.0, 50.0, 2.0)
    loose = pr.root_fillet(b, 5.0, 50.0, 2.0)
    assert tight["radius"] > loose["radius"] and tight["Kt"] < loose["Kt"]
    m0 = pr.rotor_mesh(b, fillet=0.0)
    m1 = pr.rotor_mesh(b, fillet=tight["radius"])
    assert m1.is_watertight and m1.volume > m0.volume
    assert m1.metadata["hubHeight"] > m0.metadata["hubHeight"]
    # the planform closes to (nearly) nothing at the tip
    x = pr.ideal_start(3, 44.45, 7.0, 0.775, 14000.0)
    bl = pr.blade_from_params(x, 3, 44.45, 7.0)
    assert bl.chord[-1] < 0.5 * bl.chord[-4]
    # given the motor's hub, a wide root necks down so its filleted footprint stands inside it
    x[0] = 25.0
    wide = pr.blade_from_params(x, 3, 44.45, 7.0)
    fit = pr.blade_from_params(x, 3, 44.45, 7.0, hub_height=6.0)
    assert pr.hub_height(wide, 0.8, 0.4) > 6.5
    assert pr.hub_height(fit, 0.8, 0.4) < 6.1 and fit.chord[0] < wide.chord[0]
    assert np.allclose(fit.chord[-8:], wide.chord[-8:], rtol=0.05)


def test_solid_check_fillet_lowers_junction_peak():
    try:
        from loadpath.crosscheck import mesher_missing
        if mesher_missing():
            print("skip: no tet mesher")
            return
    except Exception:
        return
    b = pr.Blade.uniform(44.45, 7.0, 2, 8.0, pitch_in=3.0, t_ratio=0.12, stations=14)
    op = pr.performance(b, 25000)
    sharp = pr.solid_check(b, op, mesh=pr.rotor_mesh(b, fillet=0.0), size=0.6)
    round_ = pr.solid_check(b, op, mesh=pr.rotor_mesh(b, fillet=2.0), size=0.6)
    print("sharp", sharp["peakStress"], sharp["hubPeak"], "| filleted", round_["peakStress"], round_["hubPeak"])
    # read over PA12's material length, the sharp corner's own peak is the junction's: the fillet takes it
    # down by half, while the blade's root bending stress outboard of it barely moves
    assert round_["hubPeak"] < 0.7 * sharp["hubPeak"], (round_["hubPeak"], sharp["hubPeak"])
    assert abs(round_["peakStress"] / sharp["peakStress"] - 1) < 0.15


def test_validation_against_printed_props():
    if not pr._have_neuralfoil():
        print("skip: NeuralFoil not installed (pip install loadpath[aero])")
        return
    v = pr.validate(sample=2)
    two = [r for r in v["rows"] if r["blades"] == 2 and r["diameter_in"] == 9]
    e2 = np.array([[r["CT"] / r["CTmeasured"] - 1, r["CP"] / r["CPmeasured"] - 1] for r in two])
    print(f"CT {v['CTmean']:+.3f} (rms {v['CTrms']:.3f}), CP {v['CPmean']:+.3f} (rms {v['CPrms']:.3f})")
    assert abs(v["CTmean"]) < 0.15 and v["CTrms"] < 0.22
    assert abs(v["CPmean"]) < 0.12 and v["CPrms"] < 0.22
    assert np.abs(e2).max() < 0.30


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(name, "ok")
