"""
The study's vibration block: a natural-frequency floor held by the search,
carried masses, and their inertial load cases in the optimizer.

Run: `python tests/test_vibration.py` (or pytest).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest
import trimesh

from loadpath.domain import build_domain
from loadpath.spec import study_from_dict
from loadpath.study import _margin, run_study
from loadpath.verify import modal_check

TIP = {"type": "box", "min": [99.0, -1, -1], "max": [101, 11, 11]}
BAR = {
    "designSpace": {"type": "box", "min": [0, 0, 0], "max": [100, 10, 10]},
    "supports": [{"shape": {"type": "box", "min": [-1, -1, -1], "max": [0.01, 11, 11]}}],
    "loadCases": [{"name": "tip", "loads": [{"shape": TIP, "force": [0, 0, -10]}]}],
    "material": {"E": 2000, "nu": 0.3, "strength": 50, "density": 1.2},
    "goal": {"safetyFactor": 2},
    "resolution": {"pitch": 2.0},
}


def _bar():
    bar = trimesh.creation.box(extents=[100, 10, 10])
    bar.apply_translation([50, 5, 5])
    return bar


def test_modal_check_matches_beam_theory():
    # clamped bar with a tip mass: f = sqrt(3EI / (L^3 (M + 0.2357 m))) / 2 pi
    s = study_from_dict({**BAR, "vibration": {"minFrequency": 10, "masses": [{"shape": TIP, "mass": 0.05}]}})
    rep = modal_check(s, _bar())
    k = 3 * 2000 * (10 * 10 ** 3 / 12) / 100 ** 3              # N/mm
    m = 0.05e-3 + 0.2357 * 1.2e-9 * 100 * 10 * 10               # tonnes
    f = np.sqrt(k / m) / (2 * np.pi)
    assert abs(rep["firstFrequency"] - f) / f < 0.05, (rep["firstFrequency"], f)
    assert rep["ok"] and rep["carried"] == 0.05
    assert rep["firstModeAxis"] in ("y", "z")       # square section: the two bending modes tie
    # a floor above it fails, and the search's margin goes with f squared
    s2 = study_from_dict({**BAR, "vibration": {"minFrequency": 2 * f, "masses": [{"shape": TIP, "mass": 0.05}]}})
    rep2 = modal_check(s2, _bar())
    assert not rep2["ok"]
    m2 = _margin(s2, {"safetyFactor": 4.0, "cases": [], "vibration": rep2})
    assert abs(m2 - 0.25) < 0.03, m2


def test_inertial_cases_are_stiffness_only():
    s = study_from_dict({**BAR, "vibration": {"minFrequency": 10, "masses": [{"shape": TIP, "mass": 0.05}]}})
    d = build_domain(s)
    assert d.case_names == ["tip", "inertia-x", "inertia-y", "inertia-z"]
    assert d.inertial_cases == [1, 2, 3]
    # scaled to the largest real case's resultant
    for c in d.inertial_cases:
        assert abs(np.linalg.norm(d.loads[c].reshape(-1, 3).sum(axis=0)) - 10.0) < 0.5
    # off without masses, or with inOptimizer false
    for vib in ({"minFrequency": 10}, {"minFrequency": 10, "inOptimizer": False,
                                        "masses": [{"shape": TIP, "mass": 0.05}]}):
        assert build_domain(study_from_dict({**BAR, "vibration": vib})).case_names == ["tip"]


def test_vibration_needs_a_search_goal_and_mass():
    with pytest.raises(ValueError, match="safetyFactor or maxDeflection"):
        study_from_dict({**BAR, "goal": {"volumeFraction": 0.3}, "vibration": {"minFrequency": 10}})
    with pytest.raises(ValueError, match="mass"):
        study_from_dict({**BAR, "vibration": {"minFrequency": 10, "masses": [{"shape": TIP}]}})


LUG = {
    "name": "lug",
    "designSpace": {"type": "box", "min": [0, -15, -4], "max": [60, 15, 4]},
    "preserve": [{"id": "eye", "shape": {"type": "cylinder", "center": [48, 0, 0], "axis": "z",
                                         "length": 8, "r": 9}}],
    "keepOut": [{"id": "pin", "shape": {"type": "cylinder", "center": [48, 0, 0], "axis": "z",
                                        "length": 12, "r": 4}}],
    "supports": [{"shape": {"type": "box", "min": [-1, -16, -5], "max": [0.01, 16, 5]}}],
    "loadCases": [{"name": "pull", "loads": [{"region": "pin", "force": [0, -80, 0]}]}],
    "material": "pa12",
    "symmetry": ["z"],
    "resolution": {"pitch": 2.0},
    "maxIterations": 40,
}


def test_search_holds_the_frequency_floor():
    # with a 200 g mass on the pin the lightest lug for the pull alone rings
    # at 21.5 Hz (3.41 g); held to 28 Hz the search delivers a stiffer one
    # (measured 4.11 g at 38 Hz with the inertial cases in the optimizer,
    # 5.31 g at 29.7 Hz checking the frequency only)
    floor = 28.0
    res = run_study(study_from_dict({**LUG, "goal": {"safetyFactor": 2, "confirm": "voxel"},
                                     "vibration": {"minFrequency": floor,
                                                   "masses": [{"region": "pin", "mass": 0.2}]}}))
    r = res.outcomes[res.best].report
    assert res.outcomes[res.best].passes
    assert r["vibration"]["firstFrequency"] >= floor, (r["vibration"]["firstFrequency"], floor)
    assert r["vibration"]["carried"] == 0.2


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))


def test_excitation_bands_keep_modes_out():
    # the clamped bar's modes: two bending near 200 Hz, two near 1200, torsion ~1800
    from loadpath.modal import band_keepout, magnification

    def check(excitations, **kw):
        s = study_from_dict({**BAR, "vibration": {"excitations": excitations, **kw}})
        return s, modal_check(s, _bar())

    s, rep = check([{"name": "once per rev", "band": [100, 400]}, {"name": "clear", "band": [400, 1000]}])
    f = rep["frequencies"]
    assert 180 < f[0] < 230 and 1100 < f[2] < 1300, f
    hit, clear = rep["excitations"]
    assert not hit["ok"] and [m["mode"] for m in hit["modesInBand"]] == [1, 2] and not rep["ok"]
    assert abs(hit["amplification"] - 50.0) < 0.5          # 1 / loss factor (0.02) at resonance
    assert clear["ok"] and not clear["modesInBand"] and clear["covered"]
    # outside the band a mode is still magnified at the band's nearer edge
    worst = max(magnification(400, f[1], 0.02), magnification(1000, f[2], 0.02))
    assert abs(clear["amplification"] - worst) < 0.02 and clear["worstMode"]["mode"] == 3
    # the search's margin: the lowest mode in the band has to pass its top
    m = _margin(s, {"safetyFactor": 4.0, "cases": [], "vibration": rep})
    assert abs(m - (f[0] / 400) ** 2) < 1e-4, m
    # a limit on magnification widens the band: 3x keeps modes out of 0.87 lo .. 1.22 hi
    lo, hi = band_keepout(400, 1000, 0.02, 3.0)
    assert abs(lo / 400 - 0.866) < 0.002 and abs(hi / 1000 - 1.224) < 0.002
    assert abs(magnification(1000, hi, 0.02) - 3.0) < 1e-6 and abs(magnification(400, lo, 0.02) - 3.0) < 1e-6
    _, rep3 = check([{"band": [400, 1000], "maxAmplification": 3}])
    assert not rep3["excitations"][0]["ok"] and rep3["excitations"][0]["keepClear"][1] > f[2]
    # a limit past resonance itself (1 / eta) is always met
    _, rep_r = check([{"band": [100, 400], "maxAmplification": 60}])
    assert rep_r["ok"] and rep_r["excitations"][0]["ok"]
    # modes are computed up to the band's top, however few were asked for
    _, rep_w = check([{"band": [2000, 3000]}], modes=1)
    assert rep_w["frequencies"][-1] >= 3000 and rep_w["excitations"][0]["covered"]
    assert rep_w["excitations"][0]["ok"]                    # torsion at ~1830, next modes past 3200
    for bad in ({"excitations": [{"band": [400, 100]}]}, {"excitations": [{"band": [100, 400], "maxAmplification": 1}]},
                {"modes": 3}):
        with pytest.raises(ValueError):
            study_from_dict({**BAR, "vibration": bad})


def test_warm_part_rings_lower():
    # the thermal block's temperature softens the modal check too: PA12 keeps
    # 45% of its modulus at 80 C, and frequency goes with its square root
    cold = modal_check(study_from_dict({**BAR, "material": "pa12", "vibration": {"minFrequency": 10}}), _bar())
    warm = modal_check(study_from_dict({**BAR, "material": "pa12", "vibration": {"minFrequency": 10},
                                        "thermal": {"ambient": 80}}), _bar())
    assert abs((warm["firstFrequency"] / cold["firstFrequency"]) ** 2 - 0.45) < 0.01
    assert warm["thermal"]["maxTemperature"] == 80.0 and "thermal" not in cold


def _box(a, b):
    return {"type": "box", "min": a, "max": b}


_MOTORS = [_box([x - 1, y - 1, -1], [x + 7, y + 7, 4]) for x in (0, 54) for y in (0, 54)]
_TOP = _box([20, 20, 26], [40, 40, 31])
# a free craft: thrust at four corners, a heavy canopy carried on top of the middle
CRAFT = {"designSpace": _box([0, 0, 0], [60, 60, 30]),
         "preserve": [{"shape": m} for m in _MOTORS] + [{"shape": _TOP}],
         "loadCases": [{"name": "thrust", "loads": [{"shape": m, "force": [0, 0, 3]} for m in _MOTORS]}],
         "material": "pa12", "goal": {"safetyFactor": 2}, "resolution": {"pitch": 3.0},
         "inertiaRelief": {"masses": [{"shape": _TOP, "mass": 0.14}] + [{"shape": m, "mass": 0.02} for m in _MOTORS]}}


def test_optimizer_modes_have_exact_gradients():
    # the design's own modes on the optimizer's grid (modalopt.py), free-free with what it
    # carries and clamped: each limit's gradient against finite differences
    from loadpath.modalopt import modal_model

    clamped = {k: v for k, v in CRAFT.items() if k != "inertiaRelief"}
    clamped["supports"] = [{"shape": _MOTORS[0]}]
    vib = {"minFrequency": 2000, "masses": [{"shape": _TOP, "mass": 0.14}],
           "excitations": [{"name": "rev", "band": [2500, 4000]}]}
    for doc in ({**CRAFT, "vibration": vib}, {**clamped, "vibration": vib}):
        s = study_from_dict(doc)
        d = build_domain(s)
        mm = modal_model(s, d)
        rng = np.random.default_rng(0)
        x = 0.3 + 0.6 * rng.random(d.mesh.n_elem)
        f, _, _ = mm.modes(x, 3.0)
        mm.floor, mm.bands = 0.5 * f[0], [("rev", 0.5 * (f[2] + f[3]), 3.0 * f[3])]     # modes 4.. in the band
        cons = mm.evaluate(x, 3.0)
        assert [n for n, _, _ in cons] == ["band.rev", "minFrequency"] and set(mm.slots()) == {"band.rev", "minFrequency"}
        for e in rng.choice(d.mesh.n_elem, 3, replace=False):
            xa, xb = x.copy(), x.copy()
            xa[e] += 1e-4
            xb[e] -= 1e-4
            for (n, _, g), (_, va, _), (_, vb, _) in zip(cons, mm.evaluate(xa, 3.0), mm.evaluate(xb, 3.0)):
                fd = (va - vb) / 2e-4
                # (small entries carry the eigensolver's own tolerance through the difference)
                assert abs(fd - g[e]) < 0.03 * abs(fd) + 0.01 * np.abs(g).max(), (n, e, fd, g[e])
    # a band reaching down to the floor raises the floor to its top; a mode under a band is left alone
    s = study_from_dict({**CRAFT, "vibration": {"minFrequency": 500, "excitations": [
        {"name": "blade pass", "band": [400, 900]}, {"name": "far", "band": [1e5, 2e5]}]}})
    mm = modal_model(s, build_domain(s))
    cons = mm.evaluate(np.full(mm.domain.mesh.n_elem, 0.5), 3.0)
    assert [n for n, _, _ in cons] == ["minFrequency"] and mm.last["floor"] == 900.0
    # nothing to hold, or switched off: no model
    assert build_domain(study_from_dict({**CRAFT, "vibration": {"minFrequency": 500, "inOptimizer": False}})).modal is None
    assert build_domain(study_from_dict(CRAFT)).modal is None


def test_free_part_is_designed_for_its_first_mode():
    # the canopy rocks on whatever joins it to the corners: with the floor in the optimizer the
    # same mass rings higher (the free part's modes were only read after meshing before)
    from loadpath.modalopt import ModalModel
    from loadpath.simp import SimpOptions, optimize

    def first(vib):
        s = study_from_dict({**CRAFT, **({"vibration": vib} if vib else {})})
        res = optimize(build_domain(s), 0.3, SimpOptions(volume_fraction=0.1, filter_radius=4.5, max_iterations=60))
        ref = study_from_dict({**CRAFT, "vibration": {"minFrequency": 1}})
        f, _, _ = ModalModel(ref, build_domain(ref), 1.0).modes(np.where(res.xphys > 0.5, 1.0, 1e-3), 3.0)
        return float(f[0]), res

    f_plain, _ = first(None)
    f_held, res = first({"minFrequency": 2.0 * f_plain})
    assert f_held > 1.1 * f_plain, (f_plain, f_held)
    assert "vibration" in res.history[-1] and res.history[1].get("frequencies")
