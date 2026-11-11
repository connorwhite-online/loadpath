"""
Metal casting (loadpath/casting.py) against results that are exact or
published: Schwarz's solution for a metal freezing against a mold, the
end-zone length Ou, Carlson, Hardin and Beckermann measured on steel plates
(2002), Carlson and Beckermann's dimensionless Niyama criterion (2009), and
closed forms for the wall metal will enter and the speed an ingate may run.
Then a part with a hot spot: the riser is sized by solving until it is sound.

    python tests/test_casting.py
"""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from loadpath import casting as C                      # noqa: E402

SLAB = ("x-", "y-", "y+", "z-", "z+")                  # a 1-D slab: the mold is on x+ only


def test_schwarz_exact_solution():
    # a pure metal poured at its melting point, perfect contact: the front is at 2 phi sqrt(alpha t)
    pure = C.alloy({"base": "a356", "name": "pure-al", "fraction": [(660.0, 0.0), (660.0, 1.0)],
                    "liquidus": 660.0, "solidus": 659.98})
    s = C.setup("sand", pure, pour_temperature=660.0, htc=1e9)
    phi = C.schwarz_front(pure, s, 660.0)
    alpha = pure.conductivity / (pure.density * pure.specific_heat)
    n, half = 20, 10.0
    r = C.solidify(np.ones((n, 1, 1), bool), half / n, s, insulated=SLAB, pad=[0, 8 * n, 0, 0, 0, 0])
    assert r.complete and r.energy_error < 1e-9, r.energy_error
    # a cell is solid when the front has crossed it: its far face is (n - i) cells from the mold
    d = (n - np.arange(n)) * half / n * 1e-3
    exact = d ** 2 / (4 * phi ** 2 * alpha)
    got = r.time[:, 0, 0]
    assert abs(got[0] / exact[0] - 1) < 0.01, (got[0], exact[0])           # the centre: 189.8 s against 189.7
    assert np.all(np.abs(got[:n // 2] / exact[:n // 2] - 1) < 0.03), got[:n // 2] / exact[:n // 2]


def test_plate_against_heat_balance():
    # an alloy with superheat and a freezing range in sand: the solve against Chvorinov's balance with
    # the contact resistance kept (the balance leaves out the metal's own gradient, so it reads short)
    for proc, name, th in (("sand", "wcb", 50.0), ("permanent", "a356", 10.0)):
        s = C.setup(proc, name)
        n = 20
        r = C.solidify(np.ones((n // 2, 1, 1), bool), th / n, s, insulated=SLAB, pad=[0, 6 * n, 0, 0, 0, 0])
        est = C.plate_freeze_time(th, s)
        assert 1.0 < r.time[0, 0, 0] / est < 1.15, (proc, name, r.time[0, 0, 0], est)
    # Chvorinov: time goes with the square of the modulus
    s = C.setup("sand", "a356")
    assert abs(C.plate_freeze_time(20.0, s) / C.plate_freeze_time(10.0, s) - 4.0) < 0.5


def test_end_zone_length_of_a_steel_plate():
    # Ou et al. 2002: a wide WCB plate 3 in thick fed from one end is sound (radiographically) over
    # 4.2 thicknesses from its free end; the unsound stretch is where Niyama < 0.1 (K s)^0.5 / mm
    T, n = 76.2, 13
    pitch = T / n
    L, rw, rh = 22 * n, int(1.5 * n), 3 * n
    metal = np.zeros((L, 1, n + rh), bool)
    metal[:, :, :n] = True
    metal[:rw, :, n:] = True
    feed = np.zeros_like(metal)
    feed[:rw, :, n:] = True
    s = C.setup("sand", "wcb", pour_temperature=1509 + 60, htc=1000.0)
    r = C.solidify(metal, pitch, s, feeders=feed, insulated=("x-", "y-", "y+"), records=600)
    assert r.complete
    x = (np.arange(L) + 0.5) * pitch
    bad = r.niyama[:, 0, n // 2] < 0.1
    assert bad.any()
    ezl = (L * pitch - x[bad].max()) / T
    assert abs(ezl - 4.2) < 0.4, ezl
    # the riser pipes and the plate holds no open cavity
    assert np.nansum(r.void[:, :, n:]) > 0 and np.nansum(r.void[:, :, :n]) < 0.02 * np.nansum(r.void)


def test_dimensionless_niyama():
    w, a = C.alloy("wcb"), C.alloy("a356")
    # Ny* = C_lambda G Tdot^(-5/6) sqrt(dP / (mu beta dT_f)), written out
    ny = C.niyama_star(0.1, 1.0, w, pressure=C.P_ATM)
    by_hand = w.sdas * 100.0 * math.sqrt(C.P_ATM / (w.viscosity * w.shrinkage * (w.liquidus - w.solidus)))
    assert abs(ny / by_hand - 1) < 1e-9
    assert abs(float(C.niyama_star(0.2, 1.0, w, pressure=C.P_ATM)) / ny - 2.0) < 1e-9
    # the pore fraction from the integral over the steel's own fraction curve against the paper's fit
    for x in (1.0, 3.0, 10.0, 30.0, 100.0):
        fit, own = float(C.pore_fraction(x, w)), float(C.pore_fraction(x, w, fit=False))
        assert abs(own / fit - 1) < 0.12, (x, fit, own)
    # steeper gradients leave less; nothing can exceed the total shrinkage
    for al in (w, a):
        p = C.pore_fraction(np.array([0.3, 1.0, 10.0, 100.0, 1000.0]), al)
        assert np.all(np.diff(p) < 0) and p[0] <= al.beta_prime() + 1e-9 and p[-1] < 2e-4, p


def test_what_the_metal_will_enter():
    a = C.alloy("a356")
    s = C.setup("sand", "a356")
    # surface tension against the head: 2 gamma / (rho g h)
    assert abs(C.capillary_wall(a, 100.0) - 2 * a.surface_tension / (a.density * C.G0 * 0.1) * 1e3) < 2e-3
    assert C.capillary_wall(a, 400.0) < C.capillary_wall(a, 100.0) / 3.9
    assert C.fluidity_length(0.5, s) == 0.0                                # under the capillary wall: no entry
    run = [C.fluidity_length(t, s) for t in (1.0, 2.0, 4.0, 8.0)]
    assert all(b > a_ for a_, b in zip(run, run[1:])), run                 # thicker runs further
    assert run[2] > 500.0                                                  # sand-cast A356 strips of 4 mm and up
    w = C.min_wall(200.0, s)
    assert 0.8 < w < 2.5 and abs(C.fluidity_length(w, s) - 200.0) < 5.0, w
    # hotter metal runs further; a die's steel stops it sooner than sand at the same head
    assert C.fluidity_length(2.0, C.setup("sand", "a356", pour_temperature=780.0)) > run[1]
    assert C.fluidity_length(2.0, C.setup("permanent", "a356")) < run[1]


def test_gating_and_contraction():
    a, w = C.alloy("a356"), C.alloy("wcb")
    assert 0.45 < C.critical_velocity(a) < 0.55                            # Campbell: about 0.5 m/s
    assert 0.011 < C.contraction(a) < 0.015 and 0.018 < C.contraction(w) < 0.023   # patternmaker's 1.3%, 2%
    g = C.gating(2.0e5, C.setup("sand", "a356"), 4.0)
    assert g["pourTime"] > 0 and g["ingateArea"] > 0 and g["sprueExitVelocity"] > g["criticalVelocity"], g
    assert g["sprueTopDiameter"] > g["sprueExitDiameter"] > 0              # tapered: the stream stays full
    # the ingates pass the sprue's flow at the critical velocity
    assert abs(g["ingateArea"] * 1e-6 * g["criticalVelocity"] / (g["flowRate"] * 1e-6) - 1) < 0.01, g


def _bossed_plate():
    import trimesh

    plate = trimesh.creation.box(extents=[120, 60, 8])
    plate.apply_translation([60, 30, 4])
    boss = trimesh.creation.cylinder(radius=18, height=30, sections=48)
    boss.apply_translation([30, 30, 22.5])
    return trimesh.boolean.union([plate, boss], engine="manifold")


def test_riser_is_sized_until_the_part_is_sound():
    part = _bossed_plate()
    alone, _ = C.cast_part(part, "sand", "a356", pitch=3.0, risers=None)
    assert not alone["ok"] and not alone["shrinkage"]["sound"]
    spot = alone["shrinkage"]["hotSpots"][0]
    assert abs(spot["position"][0] - 30) < 6 and abs(spot["position"][1] - 30) < 6     # in the boss
    rep, fields = C.cast_part(part, "sand", "a356", pitch=3.0)
    assert rep["ok"] and rep["shrinkage"]["sound"], rep["warnings"]
    assert rep["shrinkage"]["inPart"] < 0.05 * rep["shrinkage"]["inPartWithoutRisers"]
    assert len(rep["risers"]) == 1
    r = rep["risers"][0]
    assert abs(r["position"][0] - 30) < 6 and abs(r["position"][1] - 30) < 6           # on the boss
    assert 18 < r["diameter"] < 36 and r["pipeDepth"] > 3
    assert r["freezesAt"] > spot["freezesAt"]                                           # the riser freezes last
    assert 0.7 < rep["yield"] < 0.95
    # what the alloy loses in freezing went somewhere: the pipe, not the part
    assert abs(rep["shrinkage"]["total"] / (rep["volume"] / rep["yield"] * C.alloy("a356").beta_prime()) - 1) < 0.25
    assert 1.012 < rep["contraction"]["patternScale"] < 1.015


def test_pattern_and_mold_meshes():
    from loadpath import castmold

    part = _bossed_plate()
    s = C.setup("sand", "a356")
    rep, _ = C.cast_part(part, "sand", "a356", pitch=3.0)
    made = castmold.metal_pattern(part, rep, s)
    for name in ("casting-tree", "pattern-cope", "pattern-drag", "mold-cope", "mold-drag"):
        m = made["meshes"][name]
        assert m.is_watertight and m.volume > 0, name
    tree = made["meshes"]["casting-tree"]
    assert tree.volume > part.volume * rep["contraction"]["patternScale"] ** 3      # part, riser, sprue, gate
    assert made["info"]["undercutShare"] < 0.002, made["info"]


def test_study_field():
    from loadpath.manufacture import parse_casting, parse_foundry
    from loadpath.schema import lint

    fd = parse_foundry({"draw": "+z", "process": "sand", "alloy": "a356", "head": 80, "risers": False})
    assert fd == {"process": "sand", "alloy_name": "a356", "head": 80.0, "risers": None}
    assert parse_foundry({"draw": "+z"}) is None and parse_foundry("+z") is None
    assert parse_casting({"process": "investment"}) is None        # a printed pattern needs no draw
    for bad in ({"process": "sand"}, {"draft": 2}):
        try:
            parse_casting(bad)
        except ValueError:
            continue
        raise AssertionError(bad)
    base = {"designSpace": [{"type": "box", "min": [0, 0, 0], "max": [10, 10, 10]}],
            "supports": [{"shape": {"type": "box", "min": [0, 0, 0], "max": [1, 10, 10]}}],
            "loads": [{"shape": {"type": "box", "min": [9, 0, 0], "max": [10, 10, 10]}, "force": [0, 0, -10]}]}
    errs = lambda c: [x for x in lint(dict(base, casting=c)) if x["level"] == "error" and "casting" in x["path"]]
    assert not errs({"draw": "+z", "process": "sand", "alloy": "wcb", "up": "+z", "pourTemperature": 1600})
    assert not errs({"process": "investment", "alloy": "silicon-bronze"})
    assert errs({"draft": 2}) and errs({"draw": "+z", "process": "lost-foam"})


def test_study_pours_the_delivered_part():
    import types

    from loadpath.study import _castability

    o = types.SimpleNamespace(mesh=_bossed_plate(), report={"warnings": []})
    study = types.SimpleNamespace(foundry={"process": "sand", "alloy_name": "a356", "risers": None, "pitch": 3.0})
    res = types.SimpleNamespace(study=study, best=0, outcomes=[o])
    assert _castability(res) is res
    c = o.report["castability"]
    assert not c["ok"] and c["shrinkage"]["hotSpots"] and "error" not in c
    assert any(w.startswith("casting: ") for w in o.report["warnings"]), o.report["warnings"]
    # a study without a process is left alone
    o2 = types.SimpleNamespace(mesh=o.mesh, report={})
    _castability(types.SimpleNamespace(study=types.SimpleNamespace(foundry=None), best=0, outcomes=[o2]))
    assert o2.report == {}


TESTS = [test_schwarz_exact_solution, test_plate_against_heat_balance, test_end_zone_length_of_a_steel_plate,
         test_dimensionless_niyama, test_what_the_metal_will_enter, test_gating_and_contraction,
         test_riser_is_sized_until_the_part_is_sound, test_pattern_and_mold_meshes, test_study_field,
         test_study_pours_the_delivered_part]

if __name__ == "__main__":
    import time

    for t in TESTS:
        t0 = time.time()
        t()
        print(f"PASS {t.__name__} ({time.time() - t0:.1f}s)")
    print(f"\nall {len(TESTS)} passed")
