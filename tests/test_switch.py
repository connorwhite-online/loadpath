"""
Target-force push pads (loadpath/switch.py) and assembly paths
(keepOut.insertFrom). Small problems; well under a minute.

Run: `python3 tests/test_switch.py`
"""
import sys
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from loadpath import switch as sw  # noqa: E402
from loadpath.domain import build_domain  # noqa: E402
from loadpath.schema import lint  # noqa: E402
from loadpath.spec import study_from_dict  # noqa: E402


def pad(**switch):
    """A 24 x 24 x 6 mm box pad: a 1 mm skin open underneath, a plunger
    under the top centre 0.3 mm over a stem, held round the bottom rim."""
    return {
        "name": "pad",
        "designSpace": {"type": "box", "min": [-12, -12, 0], "max": [12, 12, 6]},
        "skin": {"thickness": 1.0, "open": ["-z"]},
        "preserve": [{"id": "plunger", "shape": {"type": "box", "min": [-1.5, -1.5, 2.5], "max": [1.5, 1.5, 5]}}],
        "keepOut": [{"id": "board", "bore": False, "insertFrom": "-z",
                     "shape": {"type": "box", "min": [-9, -9, -1], "max": [9, 9, 1.5]}},
                    {"id": "stem", "bore": False, "insertFrom": "-z",
                     "shape": {"type": "box", "min": [-2, -2, 1.5], "max": [2, 2, 2.5]}}],
        "supports": [{"shape": {"type": "box", "min": [-13, -13, -0.5], "max": [13, 13, 0.2]}}],
        "loadCases": [{"name": "squeeze", "loads": [
            {"shape": {"type": "box", "min": [11, -4, 2], "max": [13, 4, 5]}, "force": [-1, 0, 0]},
            {"shape": {"type": "box", "min": [-13, -4, 2], "max": [-11, 4, 5]}, "force": [1, 0, 0]}]}],
        "material": "tpu88a",
        "minMember": 1.0,
        "goal": {"volumeFraction": 0.3},
        "resolution": {"pitch": 1.0},
        "switch": {"unit": "gf",
                   "output": {"region": "plunger", "direction": "-z", "gap": 0.3, "trigger": 65, "travel": 0.25},
                   "field": {"center": [0, 0], "radius": 6, "rings": 1, "perRing": 4, "max": [500, 900],
                             "min": 400},
                   "clearance": {"region": "skin", "gap": 2.0},
                   "hold": [{"case": "squeeze", "maxDeflection": 0.5}],
                   **switch},
    }


def test_lints_clean():
    assert not lint(pad(), "study"), lint(pad(), "study")
    bad = pad(output={"region": "plunger", "direction": "down"})
    assert lint(bad, "study")


def test_insert_from_sweeps_keep_out_out_of_the_part():
    st = study_from_dict(pad())
    board = {r.id: r for r in st.keep_out}["board"]
    lo, hi = board.shape.bounds()
    # swept out through the bottom: past the design space's -z face
    assert lo[2] < -1.0 and hi[2] == 1.5, (lo, hi)
    assert not board.bore
    # a non-box shape moving sideways: generic sweep, union of copies
    from loadpath.shapes import Sphere, sweep
    s = sweep(Sphere([0, 0, 0], 1.0), [1, 0, 0], 5.0)
    assert s.contains(np.array([[2.5, 0, 0], [5.0, 0.0, 0.9]])).all()
    assert not s.contains(np.array([[2.5, 1.2, 0], [-1.1, 0, 0]])).any()
    man = s.to_manifold()
    assert abs(man.volume() - (4 / 3 * np.pi + np.pi * 5.0)) < 0.15 * (4 / 3 * np.pi + np.pi * 5.0)


def test_sensitivities_match_finite_differences():
    st = study_from_dict(pad())
    d = build_domain(st, 1.5)
    spec = sw.switch_from_study(st, d)
    prob = sw._problem(d, spec)
    nu, E = st.elastic(design=True), float(st.material.E)
    rng = np.random.default_rng(1)
    xp = rng.uniform(0.3, 0.9, d.n_elem)
    xp[d.preserve] = 1.0
    sv = sw._Solves(d, nu, rtol=1e-12)
    out, G = sw._evaluate(d, nu, prob, xp, E, sv)
    for e in rng.choice(np.flatnonzero(~d.preserve), 3, replace=False):
        h = 1e-5
        x2 = xp.copy()
        x2[e] += h
        o2, _ = sw._evaluate(d, nu, prob, x2, E, sv, grads=True)
        for name, fd, an in (("logF", (np.log(o2["F"][2]) - np.log(out["F"][2])) / h, G["logF"][2][e]),
                             ("hold", (o2["hold"][0] - out["hold"][0]) / h, G["hold"][0][e]),
                             ("clearance", (o2["clearanceKS"] - out["clearanceKS"]) / h, G["clearance"][e])):
            assert abs(fd - an) <= 1e-3 * abs(an) + 1e-9, (name, e, fd, an)


def test_fire_force_formula_by_superposition():
    """F = (g + s + T c) / a, checked by solving press + stem reaction."""
    st = study_from_dict(pad())
    d = build_domain(st, 1.5)
    spec = sw.switch_from_study(st, d)
    prob = sw._problem(d, spec)
    nu, E = st.elastic(design=True), float(st.material.E)
    xp = np.ones(d.n_elem)
    r = sw.fire_forces(d, nu, spec, xp, E)
    F = r["forces"][0]
    sv = sw._Solves(d, nu, rtol=1e-12)
    sv.factor(xp)
    u = np.asarray(sv.solve([F * prob.loads[0] - spec.trigger * prob.l])[0]) / E
    assert abs(prob.l @ u - (spec.gap + spec.travel)) < 1e-6 * (spec.gap + spec.travel) + 1e-9


def test_optimizer_moves_the_force_map():
    """From grey, the forces end inside their windows. The skin alone fires
    the centre under its floor (324 gf against 400) and a filled pad is far
    over every ceiling, so the optimum is structure in between."""
    st = study_from_dict(pad())
    r = sw.optimize_switch(st, pitches=[1.5], iterations=[40])
    first = np.asarray(r.history[0]["forces"]) / sw.GF
    last = r.forces / sw.GF
    assert r.history[0]["worstViolation"] > 0.1, first
    assert r.history[-1]["worstViolation"] < 0.05, (r.history[-1]["worstViolation"], last)
    assert r.volume < 0.5
    for p, f in zip(r.report["presses"], last):
        assert p["ok"], (p, f)


def test_thin_skin_is_a_density_floor():
    """A 1 mm skin on 1.5 mm cells: its cells become design cells floored
    at their share of skin volume, so the skin bends as thin as it is."""
    st = study_from_dict(pad())
    d0 = build_domain(st, 1.5)
    d, floor = sw._thin_skin(st, d0)
    skin = floor > 0
    assert skin.sum() > 100
    assert not d.preserve[skin].any()
    assert (floor[skin] < 1).any() and floor.max() <= 1
    # the plunger stays a whole preserved block
    assert d.preserve.sum() > 0
    # on a grid fine enough for the skin, nothing changes
    d2, f2 = sw._thin_skin(st, build_domain(st, 0.5))
    assert not f2.any()


if __name__ == "__main__":
    fails = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print("ok  ", name, flush=True)
            except Exception:
                fails += 1
                print("FAIL", name, flush=True)
                traceback.print_exc()
    sys.exit(1 if fails else 0)
