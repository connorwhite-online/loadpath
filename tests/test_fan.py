"""
Axial fan rotors (loadpath.fan): the blade-element design reproduces what it was asked for, the curve behaves like a
fan's, the physics bounds hold (Euler work, efficiency under one), the optimizer beats a hand design on a system,
and the printed rotor is one watertight body.

Run: `python tests/test_fan.py` (or pytest). About a minute.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from loadpath import fan

AIR = fan.air(35.0)
KW = dict(r_tip=57.3, r_hub=23.5, air=AIR)


def _rotor(**k):
    a = dict(blades=9, work_exp=1.5, exit="uniform")
    a.update(k)
    return fan.design(0.030, 83.0, 6400, **KW, **a)


def test_air():
    a = fan.air(20.0)
    assert abs(a["rho"] - 1.204) < 0.005
    assert abs(a["nu"] - 1.51e-5) < 0.03e-5


def test_design_hits_its_point():
    rot = _rotor()
    assert rot.design_point["converged"]
    p = fan.performance(rot, 0.030, air=AIR)
    assert abs(p["dp_ts"] / 83.0 - 1) < 0.08, p["dp_ts"]
    assert not p["stalled"]
    assert 0.44 < rot.design_point["D_max"] < 0.47        # asked for 0.45
    assert np.all(np.diff(rot.kappa1) > 0)                  # flatter blades outboard, where the blade speed is


def test_curve_is_a_fans():
    rot = _rotor()
    qs = np.linspace(0.026, 0.034, 6)
    dps = [fan.performance(rot, q, air=AIR)["dp_ts"] for q in qs]
    assert np.all(np.diff(dps) < 0)                         # pressure falls as flow rises past the design point
    lo = fan.performance(rot, 0.012, air=AIR)
    assert lo["stalled"]                                    # far below design the outer span passes D 0.6


def test_physics_bounds():
    rot = _rotor()
    om = 6400 * 2 * np.pi / 60
    for q in (0.027, 0.030, 0.033):
        p = fan.performance(rot, q, air=AIR)
        cu2 = np.array(p["stations"]["cu2"])
        assert p["dp_t"] <= AIR["rho"] * om * 57.3e-3 * cu2.max() + 1e-9    # no more than the Euler work at the tip
        assert 0 < p["eff_ts"] < p["eff_t"] < 1


def test_free_vortex_overloads_a_small_hub():
    # a free vortex puts the same work at the hub, where the blade is slow: the hub can't turn the flow that much
    rot = fan.design(0.030, 83.0, 6400, **KW, blades=9, work_exp=0.0, exit="uniform")
    assert rot.design_point["sigma_clipped"] or not rot.design_point["converged"]


def test_open_tip_and_shroud_gap_cost_pressure():
    # the same blades with a band, in a shroud with a 1.5 mm gap, and with nothing round them
    import dataclasses
    rot = _rotor(tip="band")
    band = fan.performance(rot, 0.030, air=AIR)["dp_ts"]
    gap = fan.performance(dataclasses.replace(rot, tip="shroud", gap=1.5), 0.030, air=AIR)["dp_ts"]
    open_ = fan.performance(dataclasses.replace(rot, tip="open"), 0.030, air=AIR)["dp_ts"]
    assert open_ < gap < band


def test_optimizer_beats_a_hand_design():
    system = lambda q: 92600 * q * q                         # a heat exchanger and duct: dp ~ q^2
    hand = _rotor()
    op_hand = fan.operating_point(hand, system, air=AIR)
    best, rep = fan.optimize(system, **KW, power=op_hand["power"], rpm=(3500, 6400), blades=[9, 11], axial=14.0,
                             starts=2, exit="uniform")
    op = rep["operating"]
    assert op["power"] <= op_hand["power"] * 1.01
    assert op["q"] > op_hand["q"]
    assert rep["axial_depth_max"] <= 14.0 * 1.02
    assert rep["stall_flow"] < op["q"]


def test_rotor_mesh_is_one_watertight_body():
    rot = _rotor()
    m = fan.rotor_mesh(rot, z0=4.0, z1=22.0)
    assert m.is_watertight
    assert len(m.split(only_watertight=False)) == 1        # no sealed pockets where a blade meets the band
    r = np.hypot(m.vertices[:, 0], m.vertices[:, 1])
    assert r.max() <= 57.3 + rot.band[0] + 1.2 + 1e-6       # band and its lip
    assert abs(m.bounds[0, 2] - 4.0) < 1e-6 and abs(m.bounds[1, 2] - 22.0) < 1e-6


def test_strength_at_speed():
    rot = _rotor()
    op = fan.performance(rot, 0.030, air=AIR)
    s = fan.strength(rot, op=op, retention=0.40)            # PA12 at 90 C
    om = 6400 * 2 * np.pi / 60
    rb = (57.3 + 0.8) * 1e-3
    assert abs(s["band_hoop"] - 1010 * (om * rb) ** 2 / 1e6) < 1e-9
    assert s["safety_factor"] > 1.0 and 5 < s["rotor_mass_g"] < 80


def test_skew_spreads_the_pass_and_still_meshes():
    import dataclasses
    rot = _rotor()
    straight = fan.tonal_coherence(rot)[1]
    pitch_deg = 360.0 / rot.blades
    skewed = dataclasses.replace(rot, skew=-pitch_deg, lean=2.0)       # the tip a whole blade pitch behind the hub
    assert fan.tonal_coherence(skewed)[1] < 0.5 * straight
    m = fan.rotor_mesh(skewed, z0=4.0, z1=22.0)
    assert m.is_watertight and len(m.split(only_watertight=False)) == 1
    op = fan.performance(rot, 0.030, air=AIR)
    assert fan.strength(skewed, op=op)["root_bending_Nm"] > fan.strength(rot, op=op)["root_bending_Nm"]


def test_optimizer_uses_diameter_when_coverage_pays():
    # a core whose resistance falls as the fan covers more of it: dp ~ q^2 / covered area^2, plus a fixed exit loss
    def system(q, rotor=None):
        a = (rotor.r_tip if rotor is not None else 70.0) ** 2 / 100.0 ** 2
        return 6000 * q * q / a ** 2 + 8000 * q * q
    rot, rep = fan.optimize(system, r_tip=(57.0, 100.0), r_hub=25.0, power=10.0, rpm=(1500, 6400), blades=[9],
                            axial=14.0, air=AIR, starts=2, exit="uniform", iters=160)
    small, rs = fan.optimize(system, r_tip=57.0, r_hub=25.0, power=10.0, rpm=(1500, 6400), blades=[9],
                             axial=14.0, air=AIR, starts=2, exit="uniform", iters=160)
    assert rot.r_tip > 70.0 and rep["operating"]["q"] > rs["operating"]["q"]


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
