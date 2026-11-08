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


def test_skin_zone_thins_the_skin_under_a_touch_area():
    """skin.zones: a 0.6 mm skin inside a 5 mm disc of the top, 1.0 mm
    elsewhere, the exterior unchanged; the bare skin presses lighter there."""
    zone = {"shape": {"type": "cylinder", "p0": [0, 0, 0], "p1": [0, 0, 7], "r": 5}, "thickness": 0.6,
            "blend": 1.0}
    doc = pad()
    doc["skin"] = dict(doc["skin"], zones=[zone])
    assert not lint(doc, "study"), lint(doc, "study")
    st = study_from_dict(doc)
    assert st.shells == [("skin", 0.6)]
    skin = next(r for r in st.preserve if r.id == "skin").shape
    p = np.array([[0, 0, 5.7], [0, 0, 5.2], [9, 9, 5.2], [9, 9, 4.9], [12.5, 0, 3]])
    assert list(skin.contains(p)) == [True, False, True, False, False]
    # the outer surface is the design space's own
    assert abs(skin.sdf(np.array([[0, 0, 6.0]]))[0]) < 1e-6

    def bare_centre(study):
        d, floor = sw._thin_skin(study, build_domain(study, 0.75))
        spec = sw.switch_from_study(study, d)
        bare = np.maximum(np.maximum(d.preserve.astype(float), floor), 1e-9)
        return sw.fire_forces(d, study.elastic(design=True), spec, bare, float(study.material.E))["forces"][0]

    f1, f0 = bare_centre(st), bare_centre(study_from_dict(pad()))
    assert f1 < 0.8 * f0, (f1, f0)


def test_skin_review_names_where_the_skin_alone_misses():
    """Ceilings of 550 gf at the outer presses, where the bare 1 mm skin
    alone fires at about 615 gf (1.5 mm cut cells): the review says the skin, not the
    interior, decides those presses, by how much, and what the designer
    could change. It changes nothing itself."""
    import copy
    import json

    from loadpath.skinreview import _rescale, skin_preflight

    doc = pad(field={"center": [0, 0], "radius": 6, "rings": 1, "perRing": 4, "max": [500, 550], "min": 400},
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
    whole = rv["wholeSkin"]
    assert whole["toThickness"] < 1.0
    assert all(f <= 600 * 1.02 for f in whole["forcesGf"][1:])
    # the thickness is read again on cut cells, and only that reading counts
    assert whole["cutCell"]["thickness"] == whole["toThickness"]
    assert whole["reached"] == whole["cutCellReached"]
    assert "designer" in rv["estimateSummary"]
    # on cut cells (the converged reading) the 1 mm skin's rim fires at about
    # 430 gf: under 600, so the exterior rules nothing out after all
    over = rv["grid"]["check"]["bareSkin"]["overFine"]
    assert not over and "rules nothing out" in rv["summary"], rv["summary"]
    # 1.5 mm cells against a 1 mm skin: coarse, so the bare skin is read
    # again on finer cells and the review says whether it held
    g = rv["grid"]
    assert g["coarse"] and g["check"]["finePitch"] < 1.5
    assert len(g["check"]["bareSkin"]["fineGf"]) == 5
    assert rv["trusted"] == g["check"]["converged"]
    if not rv["trusted"]:
        assert rv["summary"].startswith("On ") and rv["estimateSummary"]
    # a 1 mm skin on 1.5 mm cells is mostly split across cells: read as cut
    # cells, not soft (LOADPATH_SKIN_MODEL=floor: the review says it is)
    assert g["splitSkinShare"] > 0.5 and rv["skinModel"]["model"] == "cut"
    assert not rv["skinModel"]["readsSoft"]
    # the default minimum wall for TPU powder is 1.0 mm: no thinner skin is
    # printable, and the summary falls back on the ceiling
    rv2 = skin_preflight(study_from_dict(pad(field=doc["switch"]["field"])), 1.5)
    assert rv2["minWall"] == 1.0 and rv2["skinAtMinWall"]
    assert "raise the ceiling" in rv2["estimateSummary"], rv2["estimateSummary"]
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


def seated(**switch):
    """The pad with the switch standing on a soft column (a flap's seat):
    a preserved 4 x 4 mm TPU post from the bottom support up to the stem,
    a component (a 3 mm tall connector) on the board, and the board itself
    as a keep-out."""
    d = pad(**switch)
    d["keepOut"] = [{"id": "board", "bore": False, "shape": {"type": "box", "min": [-9, -9, -1], "max": [9, 9, 0.6]}},
                    {"id": "connector", "bore": False, "insertFrom": "-z",
                     "shape": {"type": "box", "min": [4, -3, -1], "max": [9, 3, 3.0]}},
                    {"id": "stem", "bore": False, "insertFrom": "-z",
                     "shape": {"type": "box", "min": [-2, -2, 1.5], "max": [2, 2, 2.5]}}]
    d["preserve"].append({"id": "seat", "shape": {"type": "box", "min": [-2, -2, 0], "max": [2, 2, 1.5]}})
    return d


def test_bears_on_a_seat_in_the_mesh():
    """With bearsOn naming the switch's seat, the closure is the plunger's
    travel less the seat's: F = (g + s + T r.K^-1 r) / r.K^-1 f with
    r = l - m, checked by solving the press and the reaction on both. A
    seat that gives makes every press fire harder than on rigid ground."""
    st0 = study_from_dict(seated())
    out = dict(seated()["switch"]["output"], bearsOn={"region": "seat"})
    st = study_from_dict(seated(output=out))
    d = build_domain(st, 1.0)
    spec = sw.switch_from_study(st, d)
    assert spec.base_nodes is not None and spec.base_nodes.sum() > 4
    prob = sw._problem(d, spec)
    nu, E = st.elastic(design=True), float(st.material.E)
    xp = np.full(d.n_elem, 0.4)
    xp[d.preserve] = 1.0
    r = sw.fire_forces(d, nu, spec, xp, E)
    sv = sw._Solves(d, nu, rtol=1e-12)
    sv.factor(xp)
    F = r["forces"][0]
    u = np.asarray(sv.solve([F * prob.loads[0] - spec.trigger * prob.l])[0]) / E
    assert abs(prob.l @ u - spec.stroke) < 1e-6 * spec.stroke, (prob.l @ u, spec.stroke)
    r0 = sw.fire_forces(d, nu, sw.switch_from_study(st0, d), xp, E)
    assert (r["forces"] > r0["forces"]).all(), (r["forces"] / sw.GF, r0["forces"] / sw.GF)


def test_bears_on_a_separate_part():
    """A separate part under the switch (a flap on the desk): its seat's
    compliance comes from its own cut-cell solve, close to h / (E A) for a
    block in compression, and the plunger has to take up T c_b more."""
    block = {"type": "box", "min": [-2, -2, -4], "max": [2, 2, -2]}
    part = {"shape": block, "seat": {"type": "box", "min": [-2.5, -2.5, -2.1], "max": [2.5, 2.5, -1.9]},
            "supports": [{"type": "box", "min": [-3, -3, -4.2], "max": [3, 3, -3.95]}], "pitch": 0.5}
    st = study_from_dict(pad())
    E = float(st.material.E)
    pc = sw.part_compliance(st, part, 2, -1)
    ref = 2.0 / (E * 16.0)
    assert 0.6 * ref < pc["compliance"] < 1.4 * ref, (pc["compliance"], ref)
    out = dict(pad()["switch"]["output"], bearsOn={"part": part})
    st2 = study_from_dict(pad(output=out))
    d = build_domain(st2, 1.5)
    spec = sw.switch_from_study(st2, d)
    assert abs(spec.stroke - (spec.gap + spec.travel + spec.trigger * pc["compliance"])) < 1e-12
    nu = st.elastic(design=True)
    xp = np.ones(d.n_elem)
    f1 = sw.fire_forces(d, nu, spec, xp, E)
    f0 = sw.fire_forces(d, nu, sw.switch_from_study(st, d), xp, E)
    a = (spec.gap + spec.travel + spec.trigger * f0["plungerCompliance"]) / f0["forces"]
    assert np.allclose(f1["forces"] - f0["forces"], spec.trigger * pc["compliance"] / a, rtol=1e-6)


def test_clearance_per_component():
    """A clearance entry per component: each node over the connector gets
    the room to it (the skin's underside at z 5 over a top at z 3, less the
    margin), not the whole board's smallest gap; nodes over the bare
    board get theirs."""
    clr = [{"component": "connector", "margin": 0.2}, {"component": "board", "margin": 0.2, "region": "skin"}]
    st = study_from_dict(seated(clearance=clr))
    d = build_domain(st, 1.0)
    spec = sw.switch_from_study(st, d)
    rows = {e["id"]: e for e in spec.clearance_entries}
    assert set(rows) == {"connector", "board"}, rows
    over = spec.clearance_nodes & (spec.clearance_entry == 0)
    z = d.mesh.node_xyz[over, 2]
    assert np.allclose(spec.clearance_gap[over], z - 3.0 - 0.2, atol=0.02), (z, spec.clearance_gap[over])
    assert rows["board"]["minGap"] > rows["connector"]["minGap"] + 1.0
    # gradients of the clearance KS with a room per node
    prob = sw._problem(d, spec)
    nu, E = st.elastic(design=True), float(st.material.E)
    rng = np.random.default_rng(2)
    xp = rng.uniform(0.3, 0.9, d.n_elem)
    xp[d.preserve] = 1.0
    sv = sw._Solves(d, nu, rtol=1e-12)
    out, G = sw._evaluate(d, nu, prob, xp, E, sv)
    assert {r["id"] for r in out["clearance"]} == {"connector", "board"}
    for e in rng.choice(np.flatnonzero(~d.preserve), 3, replace=False):
        h = 1e-5
        x2 = xp.copy()
        x2[e] += h
        o2, _ = sw._evaluate(d, nu, prob, x2, E, sv, grads=True)
        fd = (o2["clearanceKS"] - out["clearanceKS"]) / h
        assert abs(fd - G["clearance"][e]) <= 1e-3 * abs(G["clearance"][e]) + 1e-9, (e, fd, G["clearance"][e])
    assert not lint(seated(clearance=clr, output=dict(pad()["switch"]["output"], bearsOn={"region": "seat"})),
                    "study")


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
