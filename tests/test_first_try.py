"""
What a first, cold run of an ordinary part needs from the tool (the
held-out set, loadpath/evals/heldout): a setup that cannot work is named
before an optimization is spent on it, a warning means the study is wrong
and not that the grid is coarse, and a named print direction is not a ban
on supports. Check-level fixtures, a few seconds. Run: `python3 tests/test_first_try.py`.
"""
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from loadpath.spec import study_from_dict  # noqa: E402
from loadpath.study import _envelope_first, _pieces_again, bound_note  # noqa: E402
from loadpath.tasks import check_study  # noqa: E402


def _box(a, b):
    return {"type": "box", "min": a, "max": b}


def _cyl(c, ax, length, r):
    return {"type": "cylinder", "center": c, "axis": ax, "length": length, "r": r}


def _clamp(top):
    """A saddle clamp over a 25 mm pipe lying on the wall z = 0."""
    return {
        "designSpace": [_box([-32, -8, 0], [32, 8, top])],
        "keepOut": [{"id": "pipe", "shape": _cyl([0, 0, 12.5], "y", 20, 12.5), "bore": False}],
        "fasteners": [{"id": "l", "size": "M4", "seat": [-24, 0, 5], "direction": "-z", "grip": 5, "end": "face"},
                      {"id": "r", "size": "M4", "seat": [24, 0, 5], "direction": "-z", "grip": 5, "end": "face"}],
        "supports": [{"region": "l-bore", "fix": "xy"}, {"region": "r-bore", "fix": "xy"},
                     {"shape": _cyl([-24, 0, -0.1], "z", 0.8, 4.5), "fix": "z"},
                     {"shape": _cyl([24, 0, -0.1], "z", 0.8, 4.5), "fix": "z"}],
        "loadCases": [{"name": "pull-off", "loads": [{"region": "pipe", "force": [0, 0, 200],
                                                       "distribution": "bearing"}]}],
        "material": "petg", "process": {"type": "fdm", "buildDirection": "+y"},
        "goal": {"safetyFactor": 2.5}, "resolution": {"maxElements": 20000}}


def _warnings(raw, about):
    c = check_study(raw)
    assert c["ok"], c
    return [w for w in c["warnings"] if about in w]


def test_design_space_in_pieces_is_said_at_check():
    """An envelope that stops under the top of the pipe leaves two feet and
    nothing over the pipe: said by check, not found as a two-piece part
    after six optimizations."""
    assert _warnings(_clamp(18), "separate pieces"), "a design space cut in two went unsaid"
    assert not _warnings(_clamp(29), "separate pieces"), "a whole design space was called split"


def test_a_fastener_does_not_warn_about_its_own_rings():
    """A fastener's seat ring ends on the face its head path starts from.
    On a grid that face falls inside a cell, the cell goes to the keep-out,
    and the ring read as 43% "inside" the path: a warning against
    geometry the tool itself drew."""
    for top in (18, 29):
        assert not _warnings(_clamp(top), "inside keep-out")
    plate = {
        "designSpace": [_box([-60, -60, 0], [60, 60, 6])],
        "fasteners": [{"id": f"m{i}", "size": "M4", "seat": [sx * 37.5, sy * 37.5, 0], "direction": "+z",
                       "grip": 6, "end": "face"} for i, (sx, sy) in enumerate([(-1, -1), (1, 1)])]
        + [{"id": f"a{i}", "size": "M4", "seat": [sx * 50, sy * 50, 3], "direction": "-z", "grip": 3,
            "end": "face"} for i, (sx, sy) in enumerate([(-1, 1), (1, -1)])],
        "supports": [{"region": "a0-bore"}, {"region": "a1-bore"}],
        "loadCases": [{"name": "w", "loads": [{"region": "m0-bore", "force": [0, -50, 0]},
                                               {"region": "m1-bore", "force": [0, -50, 0]}]}],
        "material": "petg", "process": "fdm", "goal": {"safetyFactor": 3}, "resolution": {"maxElements": 40000}}
    assert not _warnings(plate, "inside keep-out")


def _boss(bore_r, clear=None):
    raw = {
        "designSpace": [_box([-10, -10, 0], [60, 10, 12])],
        "preserve": [{"id": "boss", "shape": _cyl([0, 0, 6], "z", 12, 8)},
                     {"id": "pad", "shape": _box([50, -10, 0], [60, 10, 12])}],
        "keepOut": [{"id": "bore", "shape": _cyl([0, 0, 6], "z", 16, bore_r)}],
        "supports": [{"region": "bore"}],
        "loadCases": [{"name": "w", "loads": [{"region": "pad", "force": [0, 0, -50]}]}],
        "material": "petg", "process": "fdm", "goal": {"safetyFactor": 2}, "resolution": {"maxElements": 20000}}
    if clear is not None:
        raw["keepOut"].append(clear)
    return raw


def test_a_hole_through_a_boss_is_not_a_warning_but_a_swallowed_part_is():
    """The bore through its boss is the bore doing its job (a bearing takes
    half its boss's volume). What the cut leaves is what counts: a wall the
    process cannot make, or almost nothing, is a part taken away."""
    assert not _warnings(_boss(5.5), "inside keep-out"), "a 2.5 mm wall round a bore warned"
    assert _warnings(_boss(7.7), "inside keep-out"), "a 0.3 mm wall left round a bore went unsaid"
    zone = {"id": "zone", "shape": _box([44, -12, -1], [62, 12, 13]), "cutsPreserve": True}
    c = check_study(_boss(5.5, zone))
    # declared: the cut is meant, so it is not the keep-out warning; the load left on nothing is still refused
    assert not c["ok"] and "touches no material" in c["error"], c
    zone = {"id": "zone", "shape": _box([44, -12, -1], [62, 12, 13])}
    try:
        w = _warnings(_boss(5.5, zone), "inside keep-out")
    except AssertionError:
        w = ["refused"]     # a load on a pad that is gone is an error, which says as much
    assert w, "a keep-out that swallows a preserved pad went unsaid"


def test_a_named_build_direction_leaves_supports_open():
    """`buildDirection: "+z"` says which way the layers lie. Whether the
    part is grown self-supporting is decided by cost unless the study says
    (`overhangs`, or the old printDirection)."""
    base = _clamp(29)
    s = study_from_dict(base)
    assert s.process.overhangs and not s.process.overhangs_declared
    for v in (True, False):
        s = study_from_dict({**base, "process": {"type": "fdm", "buildDirection": "+y", "overhangs": v}})
        assert s.process.overhangs is v and s.process.overhangs_declared
    s = study_from_dict({**base, "printDirection": "+y"})
    assert s.process.overhangs_declared
    s = study_from_dict({**base, "process": "sls"})
    assert not s.process.overhangs


def test_bound_note_says_what_the_envelope_can_do():
    s = study_from_dict({**_clamp(29), "goal": {"safetyFactor": 2.5, "maxDeflection": 1.0}})
    case = {"name": "pull-off", "safetyFactor": 3.2, "deflection": 2.2, "peakAt": [-24.0, 0.0, 4.0]}
    note, out = bound_note(s, {"mass": 20.0, "cases": [case], "margin": 0.45})
    assert out and "2.2 mm" in note and "2.2x the stiffness" in note and "1.30x the depth" in note, note
    case = {"name": "pull-off", "safetyFactor": 0.8, "deflection": 0.2, "peakAt": [-24.0, 0.0, 4.0]}
    note, out = bound_note(s, {"mass": 20.0, "cases": [case], "margin": 0.32})
    assert out and "32%" in note and "'l-" in note, note        # the peak is named by the ring it sits at
    # how much room: 2.5 / 0.8 = 3.1x the area pulled, its root in depth where it bends
    assert "1.8x the depth" in note and "3.1x the area" in note, note
    case = {"name": "pull-off", "safetyFactor": 2.4, "deflection": 0.2, "peakAt": [-24.0, 0.0, 4.0]}
    note, out = bound_note(s, {"mass": 20.0, "cases": [case], "margin": 0.96})
    assert not out, "4% under the goal on the solid block is inside the readings' spread, not out of reach"
    case = {"name": "pull-off", "safetyFactor": 4.0, "deflection": 0.2, "peakAt": [-24.0, 0.0, 4.0]}
    note, out = bound_note(s, {"mass": 20.0, "cases": [case], "margin": 1.6})
    assert not out and "meets the goal" in note
    note, out = bound_note(s, {"mass": 20.0, "cases": [case], "margin": 1.6, "bodies": 2})
    assert out and "2 separate pieces" in note and "meets the goal" not in note, note


def test_a_part_in_pieces_gets_more_material_once():
    """More material joins a loose pad. It does not join two straps that
    each carry their own load: the second fuller part in pieces ends it."""
    assert not _pieces_again([], 0.30, False), "the first part in pieces stopped the search"
    assert _pieces_again([0.30], 0.39, False), "a fuller part in pieces again went on adding material"
    assert not _pieces_again([0.30], 0.25, False), "a lighter try in pieces says nothing about adding material"
    assert not _pieces_again([0.30], 0.39, True), "a search that has a passing part was stopped"


def test_the_self_supporting_search_runs_only_where_the_screen_says_it_can_win():
    """Deciding supports by cost must not cost a second search that the
    screen already shows cannot win."""
    import copy
    from types import SimpleNamespace as NS

    from loadpath import orient

    st = study_from_dict(_clamp(29))
    res = NS(study=st, best=0, outcomes=[NS(volume_fraction=0.2, mesh=None)])
    real = orient.choose_build_direction
    try:
        for ss_cost, runs in ((60.0, 0), (9.0, 1)):
            rows = [{"direction": "+y", "overhangs": True, "designed": True, "makeCost": ss_cost, "safetyFactor": 1.25},
                    {"direction": "+y", "overhangs": False, "designed": True, "makeCost": 12.0}]
            orient.choose_build_direction = lambda *a, **k: {"candidates": rows, "study": st}
            calls = []
            out = orient.settle_supports(st, copy.copy(res), lambda s2: calls.append(s2) or NS(best=None, study=s2))
            assert len(calls) == runs, (ss_cost, len(calls))
            if runs:   # ...and it starts from the screen's reading: SF 1.25 for 2.5 at 20% doubles to 40%
                assert abs(calls[0].goal.volume_fraction - 0.4) < 1e-9, calls[0].goal
            assert out.study.raw["supportDecision"]["rerun"] is bool(runs)
    finally:
        orient.choose_build_direction = real


def test_an_impossible_limit_is_found_on_the_block_before_any_search():
    """A deflection limit the solid envelope cannot meet is answered by the
    block alone; a limit it can meet starts the searches as before."""
    arm = {"designSpace": [_box([0, -6, 0], [60, 6, 10])],
           "supports": [{"shape": _box([-1, -7, -1], [1, 7, 11])}],
           "loadCases": [{"name": "tip", "loads": [{"shape": _box([58, -6, 8], [60, 6, 10]), "force": [0, 0, -20]}]}],
           "material": "petg", "process": "none", "resolution": {"maxElements": 3000}}
    r = _envelope_first(study_from_dict({**arm, "goal": {"safetyFactor": 2, "maxDeflection": 0.001}}), None, None)
    assert r is not None and len(r.outcomes) == 1 and r.best is None
    assert any("no shape in it meets the limit" in w for w in r.warnings), r.warnings
    assert _envelope_first(study_from_dict({**arm, "goal": {"safetyFactor": 2, "maxDeflection": 5.0}}), None, None) is None


def test_a_peak_on_a_keep_outs_square_edge_is_named_as_the_notch_to_round():
    """A hook: a bar with a square bite out of one side for what hangs in
    it. The block's peak at the bite's inside corner is the study's own
    notch; the note names the edge. A peak elsewhere, or a pocket's edge
    with no material round it, is not blamed on one."""
    from loadpath.domain import _sharp_inside_edges

    hook = {"designSpace": [_box([0, -6, 0], [40, 6, 80])],
            "keepOut": [{"id": "bite", "shape": _box([15, -7, 20], [41, 7, 50]), "bore": False},
                        # through the whole bar, its edges on the design space's faces: nothing is formed round them
                        {"id": "slot", "shape": _box([-1, -7, 70], [41, 7, 74]), "bore": False}],
            "supports": [{"shape": _box([0, -6, 79], [40, 6, 81])}],
            "loadCases": [{"name": "hang", "loads": [{"shape": _box([15, -6, 17], [40, 6, 20.5]), "force": [0, 0, -300]}]}],
            "material": "petg", "goal": {"safetyFactor": 2}}
    s = study_from_dict(hook)
    edges = _sharp_inside_edges(s, 1.0)
    assert {e[0] for e in edges} == {"bite"} and {e[1] for e in edges} == {"y"}, edges
    assert sorted(round(float(e[2][2])) for e in edges) == [20, 50], edges      # the bite's two inside corners
    case = {"name": "hang", "safetyFactor": 1.2, "deflection": 0.1, "peakAt": [15.2, 1.0, 20.3]}
    note, out = bound_note(s, {"mass": 30.0, "cases": [case], "margin": 0.6}, 1.0)
    assert out and "square edge of keep-out 'bite'" in note and "x 15, z 20" in note and "round that edge" in note, note
    case["peakAt"] = [5.0, 0.0, 60.0]
    note, out = bound_note(s, {"mass": 30.0, "cases": [case], "margin": 0.6}, 1.0)
    assert out and "square edge" not in note, note
    note, out = bound_note(s, {"mass": 30.0, "cases": [dict(case, peakAt=[15.2, 1.0, 20.3])], "margin": 0.6})
    assert "square edge" not in note, "no grid given: nothing to hold the peak against"


def test_a_pinned_wall_sets_the_cell_when_the_study_did_not():
    """A lid with a 1.5 mm skin at a 20k budget: the grid comes out two
    cells across the skin and check has nothing to say about it; an
    explicit pitch is left alone, and a wall too thin for four times the
    budget still warns."""
    from loadpath.domain import build_domain

    def lid(skin, **res):
        return {"designSpace": [_box([-30, -20, 0], [30, 20, 9])], "skin": {"thickness": skin, "open": ["-z"]},
                "supports": [{"shape": _box([-30, -20, -0.5], [-24, 20, 0.3])},
                             {"shape": _box([24, -20, -0.5], [30, 20, 0.3])}],
                "loads": [{"shape": _cyl([0, 0, 8.8], "z", 0.8, 6), "force": [0, 0, -60]}],
                "material": "petg", "process": "fdm", "goal": {"safetyFactor": 2},
                "resolution": res or {"maxElements": 20000}}

    thin = lambda r: [w for w in r["warnings"] if "cells thick" in w or "two cells" in w]  # noqa: E731
    d = build_domain(study_from_dict(lid(1.5)))
    assert d.pitch <= 0.75 + 1e-9 and d.n_elem <= 80000, (d.pitch, d.n_elem)
    assert not thin(check_study(lid(1.5)))
    assert abs(build_domain(study_from_dict(lid(1.5, pitch=1.2))).pitch - 1.2) < 1e-9
    assert thin(check_study(lid(1.5, pitch=1.2)))
    assert thin(check_study(lid(0.6)))


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
