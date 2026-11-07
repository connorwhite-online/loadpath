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


def test_skin_review_names_where_the_skin_alone_misses():
    """Ceilings of 600 gf at the outer presses, where the bare 1 mm skin
    alone fires at about 665 gf: the review says the skin, not the
    interior, decides those presses, by how much, and what the designer
    could change. It changes nothing itself."""
    import copy
    import json

    from loadpath.skinreview import _rescale, skin_preflight

    doc = pad(field={"center": [0, 0], "radius": 6, "rings": 1, "perRing": 4, "max": [500, 600], "min": 400},
              skinReview={"minWall": 0.6})
    st = study_from_dict(doc)
    raw0 = copy.deepcopy(st.raw)
    rv = skin_preflight(st, 1.5)
    json.dumps(rv)                         # agent-readable as it is
    assert st.raw == raw0
    assert rv["reading"] == "bare skin" and rv["minWall"] == 0.6
    assert rv["skinLimited"] == 4, rv["summary"]
    assert rv["skinOnlyOver"] == [1, 2, 3, 4]
    rim = [m for m in rv["misses"] if m["limitedBy"] == "skin"]
    for m in rim:
        assert m["side"] == "over" and 5 < m["byPercent"] < 20, m
        assert m["where"] and m["where"]["cells"] > 0
        kinds = [s["kind"] for s in m["suggestions"]]
        assert kinds[0] == "target" and m["suggestions"][0]["toGf"] >= m["forceGf"]
        local = [s for s in m["suggestions"] if s["kind"] == "localSkin" and s["reached"]]
        assert local, m["suggestions"]
        for s in local:
            assert s["toThickness"] < 1.0 and s["printable"]
            assert s["forcesGf"][m["press"]] <= m["windowGf"][1] * 1.02
    # the centre is under its floor with no interior: the interior's to fix
    centre = next(m for m in rv["misses"] if m["press"] == 0)
    assert centre["side"] == "under" and centre["limitedBy"] == "interior"
    whole = rv["skinThickness"]
    assert whole["reached"] and whole["toThickness"] < 1.0
    assert all(f <= 600 * 1.02 for f in whole["forcesGf"][1:])
    assert "designer" in rv["summary"]
    # 1.5 mm cells against a 1 mm skin: coarse, so the bare skin is read
    # again on finer cells and the review says whether it held
    g = rv["grid"]
    assert g["coarse"] and g["check"]["finePitch"] < 1.5
    assert len(g["check"]["bareSkin"]["fineGf"]) == 5
    assert rv["trusted"] == g["check"]["converged"]
    if not rv["trusted"]:
        assert rv["summary"].startswith("Grid too coarse")
    # a 1 mm skin floored on 1.5 mm cells is mostly split across cells: the
    # review says the skin reads soft
    assert g["splitSkinShare"] > 0.5 and rv["skinModel"]["readsSoft"]
    # the default minimum wall for TPU powder is 1.0 mm: no thinner skin is
    # printable, and the summary falls back on the ceiling
    rv2 = skin_preflight(study_from_dict(pad(field=doc["switch"]["field"])), 1.5)
    assert rv2["minWall"] == 1.0 and rv2["skinAtMinWall"]
    assert "raise the ceiling" in rv2["summary"], rv2["summary"]
    # scaling the skin by 1 is the design itself
    d, fl = sw._thin_skin(st, build_domain(st, 1.5))
    x = np.maximum(np.random.default_rng(0).uniform(0, 1, d.n_elem), fl)
    from loadpath.skinreview import skin_cells
    _, base = skin_cells(st, d, fl)
    assert np.allclose(_rescale(x, base, base > 0, 1.0), x)


def test_run_switch_report_carries_the_skin_review():
    import json
    import tempfile

    st = study_from_dict(pad(pitches=[1.5], iterations=20))
    with tempfile.TemporaryDirectory() as out:
        rep = sw.run_switch(st, out)
    sr = rep["skinReview"]
    json.dumps(sr)
    assert sr["reading"] == "design" and len(sr["skinOnly"]) == 5
    assert sr["skinLimited"] == 0, sr["summary"]
    assert "part" in sr["grid"]["check"] and len(sr["grid"]["check"]["part"]["fineGf"]) == 5
    st2 = study_from_dict(pad(pitches=[1.5], iterations=2, skinReview=False))
    with tempfile.TemporaryDirectory() as out:
        assert "skinReview" not in sw.run_switch(st2, out)


def test_large_deflection_fire_force_converges_on_a_thin_skin():
    """The bare 1 mm skin and plunger on 1.5 mm cells: the skin is a density
    floor (cells at E x^3), the case the check crawled on and stopped short
    of the click. Every press solves to the plunger's travel at the click,
    near the linear reading at this small a bow."""
    st = study_from_dict(pad())
    dom, floor = sw._thin_skin(st, build_domain(st, 1.5))
    spec = sw.switch_from_study(st, dom)
    xphys = np.where(dom.preserve, 1.0, floor)
    rows = sw.check_fire_nonlinear(dom, st.elastic(design=True), spec, xphys, float(st.material.E),
                                   presses=[0, 1], floor=floor)
    # its linear baseline is the linear reading of the same cells: skin
    # cells more than half skin are that share of a cell, not solid ones
    ff = sw.fire_forces(dom, st.elastic(design=True), spec, np.maximum(xphys, 1e-9), float(st.material.E))
    for row in rows:
        assert "error" not in row, row
        assert abs(row["linear"] / ff["forces"][row["press"]] - 1) < 0.03, (row["linear"], ff["forces"])
        assert row["converged"], row
        assert abs(row["plungerTravel"] / (spec.gap + spec.travel) - 1) <= 0.01, row
        assert 0.8 < row["nonlinearOverLinear"] < 2.0, row


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
