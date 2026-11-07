"""
Cooling inside a fixed exterior (loadpath/cooling.py): component heat,
skin touch limits, vents, designed together with feel targets. Small
problems; under a minute.

Run: `python3 tests/test_cooling.py`
"""
import sys
import tempfile
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from loadpath import cooling  # noqa: E402
from loadpath.domain import build_domain  # noqa: E402
from loadpath.schema import lint  # noqa: E402
from loadpath.spec import study_from_dict  # noqa: E402
from loadpath.switch import _thin_skin  # noqa: E402


def box(**over):
    """A 40 x 24 x 12 mm PA12 box, 1.2 mm skin, a board with a 0.5 W chip."""
    doc = {
        "name": "box",
        "designSpace": {"type": "box", "min": [-20, -12, 0], "max": [20, 12, 12]},
        "skin": {"thickness": 1.2},
        "keepOut": [
            {"id": "board", "bore": False, "shape": {"type": "box", "min": [-16, -9, 2], "max": [16, 9, 3.6]}},
            {"id": "chip", "bore": False, "shape": {"type": "box", "min": [-10, -4, 3.6], "max": [-2, 4, 4.8]}}],
        "material": "pa12", "minMember": 2.0, "resolution": {"pitch": 2.0},
        "cooling": {"ambient": 25, "components": [
            {"region": "chip", "watts": 0.5, "maxTemp": 60},
            {"region": "board", "conductivity": 20}],
            "skin": {"contact": "held"}, "iterations": 30},
    }
    doc["cooling"].update(over)
    return doc


def _dom(doc, pitch=None):
    st = study_from_dict(doc)
    dom, fl = _thin_skin(st, build_domain(st, pitch or doc["resolution"]["pitch"]))
    return st, dom, fl


def test_lints_clean_without_supports_or_loads():
    assert not lint(box(), "study"), lint(box(), "study")
    assert lint(box(components=[{"region": "chip", "watt": 1}]), "study")
    bad = box(skin={"contact": "warm"})
    assert lint(bad, "study")
    try:
        study_from_dict(box(components=[{"region": "chip", "watts": 1, "maxTemp": 20}]))
        cooling.cooling_from_study(study_from_dict(box(components=[{"region": "chip", "watts": 1,
                                                                   "maxTemp": 20}])))
        raise AssertionError("a limit under the ambient should be refused")
    except ValueError as exc:
        assert "ambient" in str(exc)


def test_heat_balance_and_uniform_skin():
    # all heat leaves through the outside: sum h A theta over the faces is
    # the power, whatever is inside
    st, dom, fl = _dom(box(), 1.0)
    for x in (np.maximum(dom.preserve.astype(float), fl), np.ones(dom.n_elem)):
        r = cooling.cooling_readings(st, dom, np.maximum(x, 1e-9), floor=fl)
        assert abs(r["heatOut"]["skin"] - 0.5) < 1e-4, r["heatOut"]
        # every temperature over ambient, the chip the hottest point
        t = r["_theta"]
        assert t.min() >= 25.0 - 1e-6
        assert abs(r["components"][0]["temperature"] - t.max()) < 0.01
    # an exterior of area A at uniform temperature rise W / (h A): a part
    # filled with a much better conductor is nearly isothermal
    doc = box()
    doc["cooling"]["conductivity"] = 5000.0
    st, dom, fl = _dom(doc, 1.0)
    r = cooling.cooling_readings(st, dom, np.ones(dom.n_elem), floor=fl)
    area = 2 * (40 * 24 + 40 * 12 + 24 * 12) * 1e-6
    expect = 25 + 0.5 / (cooling.H_OUT * area)
    assert abs(r["skin"]["mean"] - expect) < 0.03 * (expect - 25), (r["skin"], expect)


def test_sensitivities_match_finite_differences():
    st, dom, fl = _dom(box(), 1.0)
    prob = cooling.CoolingProblem(st, dom, floor=fl)
    mdl = prob.model
    rng = np.random.default_rng(1)
    x = np.clip(0.3 + 0.4 * rng.random(dom.n_elem), 0, 1)
    x[dom.preserve] = 1.0
    x = np.maximum(x, fl)
    cand = np.flatnonzero(~dom.preserve & (x > fl + 0.05))
    for lim in prob.limits:
        th = mdl.solve(x)
        v, d, _ = lim.value(th)
        g = mdl.gradient(x, th, mdl.adjoint(d))
        for e in cand[[5, len(cand) // 2, -5]]:
            h = 1e-4
            xp, xm = x.copy(), x.copy()
            xp[e] += h
            xm[e] -= h
            fd = (lim.value(mdl.solve(xp))[0] - lim.value(mdl.solve(xm))[0]) / (2 * h)
            assert abs(fd - g[e]) <= 1e-3 * abs(g[e]) + 1e-12, (lim.name, e, g[e], fd)
    mdl.release()


def test_stack_flow():
    # 1 cm^2 in and out, 100 mm apart, 10 K: Cd A_eff sqrt(2 g H dT / T)
    g, q = cooling.stack_conductance([(100.0, 0.0), (100.0, 100.0)], 10.0, 25.0)
    a = 1e-4 / np.sqrt(2)
    q0 = cooling.CD * a * np.sqrt(2 * 9.81 * 0.1 * 10 / 298.15)
    assert abs(q - q0) < 1e-12 and abs(g - cooling.RHO_CP_AIR * q0) < 1e-9
    assert cooling.stack_conductance([(100.0, 0.0), (100.0, 0.0)], 10.0, 25.0) == (0.0, 0.0)
    # vents in the box's ends, held upright: cooler, and the heat still balances
    doc = box(gravity="-y", vents=[{"shape": {"type": "box", "min": [-8, -13, 2], "max": [8, -11, 10]}},
                                   {"shape": {"type": "box", "min": [-8, 11, 2], "max": [8, 13, 10]}}])
    st, dom, fl = _dom(doc, 1.0)
    x = np.maximum(np.maximum(dom.preserve.astype(float), fl), 1e-9)
    vented = cooling.cooling_readings(st, dom, x, floor=fl)
    shut = cooling.cooling_readings(*_dom(box(), 1.0)[:2], x, floor=fl)
    out = vented["heatOut"]
    assert abs(out["skin"] + out["vents"] - 0.5) < 2e-3 * 0.5, out
    assert out["vents"] > 0 and vented["components"][0]["temperature"] < shut["components"][0]["temperature"]


def test_run_meets_the_limits_with_a_grip():
    # cooling and a squeeze target designed together
    doc = box(components=[{"region": "chip", "watts": 0.5, "maxTemp": 50},
                          {"region": "board", "conductivity": 20}])
    doc["feel"] = {"targets": [{"kind": "squeeze", "name": "grip", "hold": {"face": "-y"},
                                "press": {"face": "+y"}, "direction": "-y", "force": 10,
                                "travel": {"max": 0.05}}]}
    st = study_from_dict(doc)
    assert not lint(doc, "study"), lint(doc, "study")
    with tempfile.TemporaryDirectory() as d:
        rep = cooling.run_cooling(st, d, pitches=[2.0], iterations=[30], verify_pitches=[1.0])
    c = rep["cooling"]
    assert c["ok"], c["design"]
    assert rep["feelOk"], rep["feel"]
    assert 0.0 < rep["volume"] < 0.9, rep["volume"]
    assert rep["watertight"] and len(c["delivered"]) == 1
    # the bare skin misses the chip's limit: the interior is doing the work
    st2, dom, fl = _dom(doc, 2.0)
    bare = cooling.cooling_readings(st2, dom, np.maximum(np.maximum(dom.preserve.astype(float), fl), 1e-9),
                                    floor=fl)
    assert not bare["components"][0]["ok"], bare["components"]


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"ok   {t.__name__}")
        except Exception:  # noqa: BLE001
            failed += 1
            print(f"FAIL {t.__name__}")
            traceback.print_exc()
    print(f"{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
