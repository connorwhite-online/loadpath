"""
Seats with friction and preload (pushseat.py, supports[].friction and
.preload), and seats in the CalculiX check and under heat with a force of
unknown direction. Run: `python3 tests/test_seat_friction.py`.
"""
import sys
import time
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from loadpath.spec import study_from_dict  # noqa: E402


def _box(a, b):
    return {"type": "box", "min": a, "max": b}


def _seated(d, extents, center, cells=6000):
    """The study `d` on a box part's analysis grid, its cases and unit
    pushes solved on the seats' springs: (study, Patches, Seat, answers on
    the springs alone per case)."""
    from dataclasses import replace
    from types import SimpleNamespace

    import trimesh

    from loadpath import pushseat
    from loadpath.verify import _analysis_grid, _solve_grid

    st = study_from_dict(d)
    m = trimesh.creation.box(extents=extents)
    m.apply_translation(center)
    g = _analysis_grid(st, m, float(m.volume), cells)
    hm, bc = g.hm, g.bc
    ps = pushseat.patches(bc.seats, hm.node_xyz, hm.n_dof)
    ks, held = pushseat.hold(ps, hm.node_xyz, g.pitch)
    gs = SimpleNamespace(**{**vars(g), "bc": replace(bc, springs=held)})
    _, us = _solve_grid(st, gs, st.elastic(st.material), st.material.E, list(bc.loads) + list(ps))
    n = len(bc.loads)
    return st, ps, pushseat.Seat(ps, us[n:], st.material.E * ks), us[:n]


def _resultants(ps, lam):
    """Each seat's force on the part (N, x y z) from the rows' forces."""
    p = np.array(ps)
    r = ps.rows
    return [(lam[r.seat == s] @ p[r.seat == s]).reshape(-1, 3).sum(axis=0) for s in range(len(r.mu))]


def test_a_part_resting_on_a_seat_holds_by_friction_until_it_slides():
    """A block resting on a table, nothing else holding it, pressed down
    with 100 N and pushed sideways: with friction 0.3 the table holds it up
    to 30 N across (Coulomb), its force equal and opposite to the load; past
    that it slides and the seat says so. Without friction `check` refuses
    the push across (nothing holds it), with friction it lets it through."""
    from loadpath.tasks import check_study

    def study(mu, fxs):
        sup = {"shape": _box([0, 0, -0.5], [40, 20, 0.1]), "push": "+z"}
        if mu:
            sup["friction"] = mu
        return {"designSpace": [_box([0, 0, 0], [40, 20, 10])], "supports": [sup],
                "loadCases": [{"name": f"x{fx:g}", "loads": [{"shape": _box([0, 0, 9.9], [40, 20, 10.1]),
                                                              "force": [fx, 0, -100]}]} for fx in fxs],
                "material": "petg", "process": "fdm", "goal": {"safetyFactor": 2}, "resolution": {"pitch": 2.0}}

    _, ps, seat, u0s = _seated(study(0.3, (20.0, 29.0, 31.0)), [40, 20, 10], [20, 10, 5], 4000)
    for fx, u0 in zip((20.0, 29.0), u0s):
        _, _, lam, lifts = seat.push(seat.gaps(u0))
        r = _resultants(ps, lam)[0]
        assert not lifts and np.abs(r - [-fx, 0, 100]).max() < 1e-3 * 100, (fx, r)
        # Coulomb on every patch: no more across than mu times its push
        nr = seat.normal
        across = np.linalg.norm(lam[seat.ft], axis=1)
        assert (across <= 0.3 * lam[seat.fi] * (1 + 1e-6) + 1e-9).all()
        assert lam[nr].min() >= 0
    assert seat.push(seat.gaps(u0s[2]))[3]                          # 31 N: it slides
    try:
        seat.push(seat.gaps(u0s[2]), strict=True)
        raise AssertionError("a block pushed past its friction was solved")
    except ValueError as exc:
        assert "slides the part along its seat" in str(exc), exc
    c = check_study(study(None, (20.0,)))
    assert not c["ok"] and "give it friction" in c["error"], c
    assert check_study(study(0.3, (20.0,)))["ok"]


def test_a_clamp_holds_by_its_preload_on_both_jaws():
    """A plate squeezed between two jaws by a screw's 200 N (the lower jaw's
    preload) and pulled across: it holds by friction on both faces, about
    2 mu times the clamp force, and slides past it; the clamp's force is the
    preload with no other load (the interference that gives it is solved),
    relaxed by a year of PETG creep. Without the preload it holds nothing
    across."""
    from loadpath import pushseat
    from loadpath.durability import clamp_retention
    from loadpath.materials import material_from_spec

    keep = clamp_retention(material_from_spec("petg"))
    assert 0.5 < keep < 0.75, keep

    def study(pre):
        low = {"shape": _box([10, 5, -0.5], [20, 15, 0.1]), "push": "+z", "friction": 0.3}
        if pre:
            low["preload"] = pre
        return {"designSpace": [_box([0, 0, 0], [30, 20, 6])],
                "supports": [{"shape": _box([10, 5, 5.9], [20, 15, 6.5]), "push": "-z", "friction": 0.3}, low],
                "loadCases": [{"name": "clamp", "loads": [], "_clamp": True}] +
                [{"name": f"x{fx:g}", "loads": [{"shape": _box([29.9, 0, 0], [30.1, 20, 6]), "force": [fx, 0, 0]}]}
                 for fx in (50.0, 100.0)],
                "material": "petg", "process": "fdm", "goal": {"safetyFactor": 2}, "resolution": {"pitch": 1.0}}

    _, ps, seat, u0s = _seated(study(200.0), [30, 20, 6], [15, 10, 3])
    assert seat.e.min() < 0 and not seat.unpressed                # set into the part
    top, low = _resultants(ps, seat.push(seat.gaps(u0s[0]))[2])
    assert abs(low[2] - 200.0 * keep) < 1.0 and abs(top[2] + 200.0 * keep) < 1.0, (top, low)
    _, _, lam, lifts = seat.push(seat.gaps(u0s[1]))
    top, low = _resultants(ps, lam)
    assert not lifts and abs(top[0] + low[0] + 50.0) < 0.1, (top, low)
    assert abs(top[0] + low[0]) <= 0.3 * (low[2] - top[2]) * (1 + 1e-6)
    assert seat.push(seat.gaps(u0s[2]))[3]                          # past 2 mu N: it slides
    assert 0.5 < seat.held < 1.0                                    # ...part of the way there
    _, ps0, seat0, u0s0 = _seated(study(None), [30, 20, 6], [15, 10, 3])
    assert isinstance(seat0, pushseat.Seat) and seat0.push(seat0.gaps(u0s0[1]))[3]


def test_a_screws_head_seat_takes_its_preload_when_the_seats_have_friction():
    """Fasteners whose bores hold the part across: the head seat each screw
    adds takes the screw's preload once the study's seats have friction
    (the clamp is what the joint's friction holds by), as much of it as a
    year of creep leaves; the preload case is then the fresh clamp: the
    seat's relaxed share and a pair of forces for the rest. Without friction
    the joint is as before."""
    from loadpath.durability import clamp_retention
    from loadpath.fasteners import expand_fasteners
    from loadpath.materials import material_from_spec

    keep = clamp_retention(material_from_spec("petg"))

    def plate(mu):
        sup = {"shape": _box([-30, -10, -0.5], [30, 10, 0.3]), "push": "+z"}
        if mu:
            sup["friction"] = mu
        return {"designSpace": [_box([-30, -10, 0], [30, 10, 6])],
                "supports": [sup, {"region": "a-bore", "fix": "xy"}, {"region": "b-bore", "fix": "xy"}],
                "loads": [{"shape": _box([-5, -5, 5.5], [5, 5, 6]), "force": [30, 0, 40]}],
                "fasteners": [{"id": i, "size": "M4", "seat": [x, 0, 6], "direction": "-z", "grip": 6,
                               "end": "face", "preload": 300} for i, x in (("a", -22), ("b", 22))],
                "material": "petg", "process": "fdm", "goal": {"safetyFactor": 2}, "resolution": {"pitch": 1.0}}

    d = expand_fasteners(plate(0.3))
    heads = [s for s in d["supports"] if s.get("region", "").endswith("-seat")]
    assert all(abs(s.get("preload") - 300.0 * keep) < 1e-9 for s in heads), heads
    pre = next(c for c in d["loadCases"] if c["name"] == "preload")
    assert pre["_clamp"] and len(pre["loads"]) == 4, pre
    assert abs(abs(pre["loads"][0]["force"][2]) - 300.0 * (1 - keep)) < 1e-9, pre
    st = study_from_dict(d)
    pl = [s.preload for s in st.supports if s.push is not None]
    assert pl[0] is None and all(abs(v - 300.0 * keep) < 1e-9 for v in pl[1:]), pl      # relaxed once
    d0 = expand_fasteners(plate(None))
    assert not [s for s in d0["supports"] if s.get("preload")]
    pre0 = next(c for c in d0["loadCases"] if c["name"] == "preload")
    assert any(any(ld["force"]) for ld in pre0["loads"])


def test_friction_and_preload_are_a_seats_only():
    base = {"designSpace": [_box([0, 0, 0], [60, 10, 4])],
            "supports": [{"shape": _box([19, 0, -0.1], [21, 10, 4.1])},
                         {"shape": _box([0, 0, -0.2], [60, 10, 0.05]), "push": "+z", "friction": 0.4,
                          "preload": 50}],
            "loads": [{"shape": _box([58, 0, 3.9], [60, 10, 4.1]), "force": [0, 0, 50]}],
            "material": "petg", "process": "fdm", "goal": {"safetyFactor": 2}, "resolution": {"pitch": 1.0}}
    s = study_from_dict(base).supports[1]
    assert s.friction == 0.4 and 25.0 < s.preload < 40.0, s.preload     # PETG keeps ~62% after a year
    metal = study_from_dict({**base, "material": "316l"}).supports[1]
    assert metal.preload > 49.0
    for key, v in (("friction", 0.3), ("preload", 10), ("friction", -1)):
        bad = {**base, "supports": [dict(base["supports"][0], **{key: v}), base["supports"][1]]}
        if key == "friction" and v < 0:
            bad = {**base, "supports": [base["supports"][0], dict(base["supports"][1], friction=v)]}
        try:
            study_from_dict(bad)
            raise AssertionError(f"{key}={v} accepted")
        except ValueError as exc:
            assert key in str(exc)


def _pried(cases, thermal=None, pitch=1.0):
    import trimesh

    d = {"designSpace": [_box([0, 0, 0], [60, 10, 4])],
         "supports": [{"shape": _box([19.9, 0, -0.1], [20.1, 10, 4.1]), "fix": "z"},
                      {"shape": _box([19.9, 0, -0.1], [20.1, 10, 4.1]), "fix": "y"},
                      {"shape": _box([19.9, 0, -0.5], [20.1, 10, 0.1]), "fix": "x"},
                      {"shape": _box([0, 0, -0.5], [60, 10, 0.1]), "push": "+z"}],
         "loadCases": [{"name": n, "loads": [{"shape": _box([58, 0, 3.9], [60, 10, 4.1]), **ld}]} for n, ld in cases],
         "material": "petg", "process": "fdm", "goal": {"safetyFactor": 2}, "resolution": {"pitch": pitch}}
    if thermal:
        d["thermal"] = thermal
    m = trimesh.creation.box(extents=[60, 10, 4])
    m.apply_translation([30, 5, 2])
    return d, m


def test_calculix_solves_a_seat_as_the_native_tet_check_does():
    """The CalculiX backend takes push supports: the same contact on its
    solves, the patches' springs as springs on their means (*EQUATION and
    SPRING1). Pried up and pressed down, it reads what the native check
    reads."""
    import shutil

    if shutil.which("ccx") is None:
        print("  (skipped: CalculiX ccx not on PATH)")
        return
    from loadpath.crosscheck import crosscheck

    d, m = _pried([("up", {"force": [0, 0, 50]}), ("down", {"force": [0, 0, -50]})])
    r = crosscheck(study_from_dict(d), m, size=2.0, backend="both")
    for a in r["agreement"]:
        assert abs(a["deflection"] - 1) < 0.01 and abs(a["peakStress"] - 1) < 0.02, r["agreement"]
    up, down = r["cases"]
    assert down["deflection"] < 0.01 * up["deflection"]


THERMAL = {"ambient": 25, "convection": 30, "expansion": True,
           "temperatures": [{"shape": _box([50, -1, -1], [61, 11, 5]), "t": 70}]}


def test_a_seated_part_under_heat_is_judged_in_the_worst_direction():
    """A force of unknown direction on a seated part with a thermal block
    (refused before): the check solves the contact per direction with the
    expansion in the fixed part and reads the worst no lower than the axis
    it was written along, and the optimizer runs it."""
    from loadpath.domain import build_domain
    from loadpath.simp import SimpOptions, optimize
    from loadpath.verify import verify_part

    d, m = _pried([("up", {"force": [0, 0, 50]}), ("any", {"force": [0, 0, 50], "direction": "any"})], THERMAL)
    rep = {c["name"]: c for c in verify_part(study_from_dict(d), m)["cases"]}
    assert rep["any"]["peakStress"] >= rep["up"]["peakStress"] * 0.999, rep
    assert len(rep["any"]["worstDirection"]) == 3
    d2 = dict(d, goal={"volumeFraction": 0.5}, resolution={"pitch": 2.0})
    r = optimize(build_domain(study_from_dict(d2)), 0.35,
                 SimpOptions(volume_fraction=0.5, filter_radius=4.0, max_iterations=4, escape=None))
    assert np.isfinite(r.compliance) and r.case_direction[1] is not None


def test_the_optimizer_designs_a_part_that_stands_on_its_friction():
    """A stand on a desk (a seat with friction, nothing else) pressed down
    and pushed across: the optimizer solves it with the seat holding the
    push by friction, and the design it delivers holds it the same way."""
    from loadpath.directions import collapse, expand
    from loadpath.domain import build_domain
    from loadpath.simp import E_MIN, PENALTY, SimpOptions, _Analysis, optimize

    d = build_domain(study_from_dict({
        "designSpace": [_box([0, 0, 0], [40, 10, 30])],
        "supports": [{"shape": _box([0, 0, -0.5], [40, 10, 0.6]), "push": "+z", "friction": 0.4}],
        "loads": [{"shape": _box([15, 0, 28.5], [25, 10, 30.5]), "force": [15, 0, -60]}],
        "material": "petg", "process": "fdm", "goal": {"volumeFraction": 0.35}, "resolution": {"pitch": 2.0}}))
    r = optimize(d, 0.35, SimpOptions(volume_fraction=0.35, filter_radius=4.0, max_iterations=12, escape=None))
    fe = _Analysis(d, 0.35, 1e-8)
    rhs, layout = expand(d.loads, d.spreads)
    us = fe.solve(E_MIN + r.xphys ** PENALTY * (1 - E_MIN), r.xphys, [None] * len(rhs), loads=rhs, layout=layout)
    lam = fe._seat["lam"][0]
    ps = fe._seat["ps"]
    f = _resultants(ps, lam)[0]
    assert abs(f[0] + 15.0) < 0.2 and abs(f[2] - 60.0) < 0.2, f
    c = collapse(rhs, layout, us, d.spreads, seated=fe.seated_worst())[0].compliance
    assert np.isfinite(c) and c > 0


def test_seats_hold_a_part_in_buckling_and_vibration_where_they_bear():
    """A column standing on a seat with friction buckles at the fixed base's
    load (the seat bears all across its foot and sticks); a bar held by a
    preloaded clamp with friction rings within a few % of the bar fixed at
    that end, while a clamp without friction holds nothing along the bar (a
    zero mode); a seat that nothing presses (no weight, no preload) holds
    nothing, and a part on it alone has no modal check."""
    import trimesh

    from loadpath.verify import _buckling, modal_check

    def column(sup):
        return study_from_dict({
            "designSpace": [_box([0, 0, 0], [10, 10, 60])], "supports": [sup],
            "loadCases": [{"name": "down", "loads": [{"shape": _box([0, 0, 59.9], [10, 10, 60.1]),
                                                      "force": [0, 0, -100]}]}],
            "material": "petg", "process": "fdm", "goal": {"safetyFactor": 2}, "resolution": {"pitch": 2.0}})

    m = trimesh.creation.box(extents=[10, 10, 60])
    m.apply_translation([5, 5, 30])
    foot = _box([0, 0, -0.5], [10, 10, 0.1])
    fixed = _buckling(column({"shape": foot}), m, float(m.volume), 2.0, None, [None])["factor"]
    seat = _buckling(column({"shape": foot, "push": "+z", "friction": 0.5}), m, float(m.volume), 2.0, None,
                     [None])["factor"]
    assert abs(seat / fixed - 1) < 0.03, (seat, fixed)

    def bar(sups):
        return study_from_dict({
            "designSpace": [_box([0, 0, 0], [80, 10, 4])], "supports": sups,
            "loadCases": [{"name": "tip", "loads": [{"shape": _box([79.9, 0, 0], [80.1, 10, 4]),
                                                     "force": [0, 0, -5]}]}],
            "vibration": {"minFrequency": 50},
            "material": "petg", "process": "fdm", "goal": {"safetyFactor": 2}, "resolution": {"pitch": 1.0}})

    b = trimesh.creation.box(extents=[80, 10, 4])
    b.apply_translation([40, 5, 2])
    lo, hi = _box([0, 0, -0.5], [10, 10, 0.1]), _box([0, 0, 3.9], [10, 10, 4.5])

    def jaws(mu):
        out = [{"shape": lo, "push": "+z", "preload": 200}, {"shape": hi, "push": "-z", "preload": 200}]
        if mu:
            for j in out:
                j["friction"] = mu
        return out

    f_fixed = modal_check(bar([{"shape": _box([0, 0, -0.5], [10, 10, 4.5])}]), b)["frequencies"][0]
    f_clamp = modal_check(bar(jaws(0.3)), b)["frequencies"][0]
    assert 0.9 < f_clamp / f_fixed <= 1.0 + 1e-6, (f_clamp, f_fixed)
    loose = modal_check(bar(jaws(None)), b)
    assert loose["frequencies"][0] < 1.0 and not loose["ok"], loose
    assert modal_check(bar([{"shape": _box([0, 0, -0.5], [80, 10, 0.1]), "push": "+z", "friction": 0.5}]),
                       b) is None


TESTS = [v for k, v in dict(globals()).items() if k.startswith("test_")]


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
