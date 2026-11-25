"""
What a first, cold run of an ordinary part needs from the tool (the
held-out set, loadpath/evals/heldout): a setup that cannot work is named
before an optimization is spent on it, a warning means the study is wrong
and not that the grid is coarse, and a named print direction is not a ban
on supports. Check-level fixtures, a few seconds. Run: `python3 tests/test_first_try.py`.
"""
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from loadpath.spec import study_from_dict  # noqa: E402
from loadpath.study import (  # noqa: E402
    _BOUND_KEY,
    StudyResult,
    _envelope_first,
    _pieces_again,
    bound_note,
)
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
    ok = _envelope_first(study_from_dict({**arm, "goal": {"safetyFactor": 2, "maxDeflection": 5.0}}), None, None)
    # in reach: the study goes on to the searches, carrying the block's margin
    assert ok is not None and not isinstance(ok, StudyResult)
    assert ok.raw[_BOUND_KEY]["margin"] >= 1.0 and ok.raw[_BOUND_KEY]["passes"]


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


def _lid(skin, **res):
    return {"designSpace": [_box([-30, -20, 0], [30, 20, 9])], "skin": {"thickness": skin, "open": ["-z"]},
            "supports": [{"shape": _box([-30, -20, -0.5], [-24, 20, 0.3])},
                         {"shape": _box([24, -20, -0.5], [30, 20, 0.3])}],
            "loads": [{"shape": _cyl([0, 0, 8.8], "z", 0.8, 6), "force": [0, 0, -60]}],
            "material": "petg", "process": "fdm", "goal": {"safetyFactor": 2},
            "resolution": res or {"maxElements": 20000}}


def _thin_warnings(r):
    return [w for w in r["warnings"] if "cells thick" in w or "two cells" in w or "a cell thick" in w]


def test_a_pinned_wall_sets_the_cell_when_the_study_did_not():
    """On whole cells (LOADPATH_WALL_MODEL=cells) a lid's 1.5 mm skin at
    a 20k budget sets the grid two cells across the skin and check has nothing to say about it; an explicit
    pitch is left alone, and a wall too thin for four times the budget
    still warns."""
    import os

    from loadpath.domain import build_domain, wall_pitch

    lid = _lid
    os.environ["LOADPATH_WALL_MODEL"] = "cells"
    try:
        _whole_cell_walls(lid, build_domain, wall_pitch)
    finally:
        del os.environ["LOADPATH_WALL_MODEL"]


def _whole_cell_walls(lid, build_domain, wall_pitch):
    st = study_from_dict(lid(1.5))
    d = build_domain(st, wall_pitch(st))
    assert d.pitch <= 0.75 + 1e-9 and d.n_elem <= 84000, (d.pitch, d.n_elem)
    assert not _thin_warnings(check_study(lid(1.5)))
    assert wall_pitch(study_from_dict(lid(1.5, pitch=1.2))) is None
    assert wall_pitch(study_from_dict({k: v for k, v in lid(1.5).items() if k != "skin"})) is None
    assert _thin_warnings(check_study(lid(1.5, pitch=1.2)))
    assert _thin_warnings(check_study(lid(0.6)))


def test_a_thin_wall_is_cut_cells_on_the_studys_own_grid():
    """The same lid in a plain run: the skin is cut cells on the study's
    own grid (thinwall.py), a cell through it is enough, the grid is
    refined only for a skin under a cell, and check says so then. Its
    cells cost the optimizer the rest of the cell the skin leaves."""
    from loadpath.domain import build_domain, wall_pitch

    st = study_from_dict(_lid(1.5))
    assert wall_pitch(st) is None
    d = build_domain(st, cut_walls=True)
    # (only the cells under the supports and the load are pinned solid over it)
    assert d.walls is not None and d.walls.sum() > 1000 and (d.preserve & d.walls).sum() < 0.1 * d.walls.sum()
    assert d.pitch > 0.95, d.pitch
    share = d.mesh.wall_share[d.walls]
    assert 0.0 < share.min() and share.max() <= 1.0
    assert np.allclose(d.volume_weights[d.walls], 1.0 - share)
    assert not _thin_warnings(check_study(_lid(1.5)))
    assert not _thin_warnings(check_study(_lid(1.5, pitch=1.2)))
    assert build_domain(st).walls is None          # the other solves keep whole cells
    constrained = study_from_dict(dict(_lid(1.5), goal={"safetyFactor": 2, "method": "constrained"}))
    assert wall_pitch(constrained) is None           # cut walls too (thinwall.CONSTRAINED)
    st6 = study_from_dict(_lid(0.6))
    assert wall_pitch(st6) < 0.7                    # a cell through it, within 4x the budget
    assert _thin_warnings(check_study(_lid(0.6)))   # ... which still leaves it under a cell


def test_the_constrained_method_reads_a_cut_wall_at_its_own_sub_cells():
    """constrained._Averaged reads a thin wall's stress where the wall is,
    from the cut cell's own kinematics (its bending lives in the condensed
    bubbles), as the verification reads a cut cell: a cantilever plate's
    bending at mid-span comes out near beam theory's, and the adjoint of
    the reading is its transpose."""
    from loadpath.constrained import _VM, _Averaged
    from loadpath.domain import build_domain
    from loadpath.simp import E_MIN, _Analysis

    L, b, t = 60.0, 20.0, 1.5
    d = {"name": "plate", "designSpace": {"type": "box", "min": [0, -b / 2, -7.63], "max": [L, b / 2, 2.37]},
         "skin": {"thickness": t, "open": ["-z", "+x", "-x", "+y", "-y"]},
         "supports": [{"shape": {"type": "box", "min": [0, -b / 2, -8], "max": [0.5, b / 2, 3]}}],
         "loadCases": [{"name": "tip", "loads": [{"shape": {"type": "box", "min": [L - 0.5, -b / 2, -8],
                                                            "max": [L, b / 2, 3]}, "force": [0, 0, -1]}]}],
         "material": "pla", "goal": {"volumeFraction": 0.2}, "minMember": 2.0, "resolution": {"pitch": 1.4}}
    st = study_from_dict(d)
    nu = st.elastic(design=True)
    dom = build_domain(st, cut_walls=True)
    mesh = dom.mesh
    x = dom.preserve.astype(float)
    u = _Analysis(dom, nu, 1e-10).solve(E_MIN + (1 - E_MIN) * x ** 3, x, [None])[0]
    av = _Averaged(mesh, nu, 0.86, np.ones(mesh.n_nodes, bool))
    assert av.wall is not None
    rng = np.random.default_rng(0)
    h, uu = rng.normal(size=(len(av.wall["w"]), 6)), rng.normal(size=mesh.n_dof)
    a, b_ = float(np.sum(h * av.wall_stress(uu))), float(av.wall_rhs(h, mesh.n_dof) @ uu)
    assert abs(a - b_) <= 1e-9 * abs(a)
    ss = np.concatenate([np.zeros((av.n_base, 6)), av.wall_stress(u)])
    v = np.concatenate([np.zeros(av.n_base), av.wall["w"]])
    dn = av.ball @ v
    sig = (av.ball @ (v[:, None] * ss)) / np.maximum(dn, 1e-30)[:, None]
    vm = np.sqrt(np.einsum("ij,ij->i", sig @ _VM, sig))
    xyz = mesh.node_ijk * mesh.pitch
    mid = (np.abs(xyz[:, 0] - 30) < mesh.pitch) & (np.abs(xyz[:, 1]) < 5) & (dn > 0)
    beam = 6 * 30.0 / (b * t ** 2)
    assert 0.8 * beam < vm[mid].max() < 1.25 * beam, (vm[mid].max(), beam)


def _plate_tip(pitch, cut, t=1.5):
    """Tip deflection of a cantilever plate (a skin left open on five faces)
    at unit modulus over the beam's F L^3 / 3 E I, nothing designed."""
    from loadpath.domain import build_domain
    from loadpath.simp import E_MIN, _Analysis

    L, b, z0 = 60.0, 20.0, 0.37
    st = study_from_dict({"designSpace": [_box([0, -b / 2, -8 + z0], [L, b / 2, 2 + z0])],
                          "skin": {"thickness": t, "open": ["-z", "+x", "-x", "+y", "-y"]},
                          "supports": [{"shape": _box([0, -b / 2, -8], [0.5, b / 2, 3])}],
                          "loads": [{"shape": _box([L - 0.5, -b / 2, -8], [L, b / 2, 3]), "force": [0, 0, -1]}],
                          "material": "pla", "goal": {"volumeFraction": 0.2}, "minMember": 2.0,
                          "resolution": {"pitch": pitch}})
    d = build_domain(st, cut_walls=cut)
    fe = _Analysis(d, st.elastic(design=True), 1e-9)
    x = d.preserve.astype(float)
    u = fe.solve(E_MIN + (1 - E_MIN) * x ** 3, x, [None])[0]
    return float(d.loads[0] @ u) / (L ** 3 / (3 * b * t ** 3 / 12))


def test_a_cut_cell_wall_bends_like_the_wall_where_whole_cells_round_it():
    """A 1.5 mm plate off the grid's planes: as cut cells it reads within
    15% of its own reading at three cells through, from 1.4 cells; on whole
    cells at the same grids it rounds to 2.2 mm and 1.7 mm and reads 3.5x
    and 1.7x too stiff (bending goes as thickness cubed)."""
    ref = _plate_tip(0.5, True)
    assert 0.85 < ref < 0.95, ref        # beam to plate (1 - nu^2) theory
    for p in (1.1, 0.9):
        cut, whole = _plate_tip(p, True), _plate_tip(p, False)
        assert abs(cut / ref - 1.0) < 0.15, (p, cut, ref)
        assert whole / ref < 0.6, (p, whole, ref)


def test_a_searchs_candidates_read_a_pinned_wall_at_two_cut_cells():
    """A cut cell does not round a wall's thickness, so a search's own
    reading of a candidate spends two cells through it; the delivered
    reading keeps four, and so does any reading on whole cells."""
    import trimesh

    from loadpath.verify import SEARCH_SHELL_CELLS, _analysis_grid

    st = study_from_dict({"designSpace": [_box([-30, -20, 0], [30, 20, 2])], "skin": {"thickness": 1.6, "open": ["-z"]},
                          "supports": [{"shape": _box([-30, -20, -0.5], [-24, 20, 0.3])},
                                       {"shape": _box([24, -20, -0.5], [30, 20, 0.3])}],
                          "loads": [{"shape": _cyl([0, 0, 1.8], "z", 0.8, 6), "force": [0, 0, -60]}],
                          "material": "petg", "process": "fdm", "goal": {"safetyFactor": 2},
                          "resolution": {"maxElements": 20000}})
    lid = trimesh.creation.box(extents=[60, 40, 2])
    lid.apply_translation([0, 0, 1])
    vol = float(lid.volume)
    grid = lambda **kw: _analysis_grid(st, lid, vol, 2000, **kw).pitch  # noqa: E731
    assert abs(grid() - 0.4) < 1e-9
    assert abs(grid(shell_cells=SEARCH_SHELL_CELLS) - 0.8) < 1e-9
    assert abs(grid(shell_cells=SEARCH_SHELL_CELLS, cut_cells=False) - 0.4) < 1e-9


def test_a_push_support_carries_pressure_and_lets_go_where_it_is_lifted():
    """A plate lying on a seat, held down along one line and pried up at
    its far end: the seat pushes only on the heel, with the moment the
    lever needs about the line, and its gaps and forces are never both
    there. Pressed down instead, the seat carries the lever on the load's
    side. Held "fix: all" the face would pull."""
    from dataclasses import replace
    from types import SimpleNamespace

    import trimesh

    from loadpath import pushseat
    from loadpath.verify import _analysis_grid, _solve_grid

    def plate(fz):
        st = study_from_dict({
            "designSpace": [_box([0, 0, 0], [60, 10, 4])],
            "supports": [{"shape": _box([19.9, 0, -0.1], [20.1, 10, 4.1]), "fix": "z"},
                         {"shape": _box([19.9, 0, -0.1], [20.1, 10, 4.1]), "fix": "y"},
                         {"shape": _box([19.9, 0, -0.5], [20.1, 10, 0.1]), "fix": "x"},
                         {"shape": _box([0, 0, -0.5], [60, 10, 0.1]), "push": "+z"}],
            "loads": [{"shape": _box([58, 0, 3.9], [60, 10, 4.1]), "force": [0, 0, fz]}],
            "material": "petg", "process": "fdm", "goal": {"safetyFactor": 2}, "resolution": {"pitch": 1.0}})
        m = trimesh.creation.box(extents=[60, 10, 4])
        m.apply_translation([30, 5, 2])
        g = _analysis_grid(st, m, float(m.volume), 20000)
        hm, bc = g.hm, g.bc
        line = np.unique(hm.node_xyz[bc.fixed[2::3], 0])
        assert len(line) == 1 and np.ptp(hm.node_xyz[bc.fixed[0::3], 2]) < 0.6, line   # a hinge line, no clamp
        ps = pushseat.patches(bc.seats, hm.node_xyz, hm.n_dof)
        ks, held = pushseat.hold(ps, hm.node_xyz, g.pitch)
        gs = SimpleNamespace(**{**vars(g), "bc": replace(bc, springs=held)})
        _, us, lam = pushseat.solve(lambda r: _solve_grid(st, gs, st.elastic(st.material), st.material.E, r),
                                    bc.loads, ps, st.material.E * ks)
        x = np.array([np.abs(p[2::3]) @ hm.node_xyz[:, 0] / np.abs(p[2::3]).sum() for p in ps])
        gap = np.array([p @ us[0] for p in ps])
        arm = float(np.mean(hm.node_xyz[bc.loads[0][2::3] != 0, 0])) - line[0]
        return lam[0], x - line[0], gap, arm

    lam, x, gap, arm = plate(50.0)
    assert lam.min() >= 0 and gap.min() > -1e-8 and np.abs(lam * gap).max() < 1e-6
    assert lam[x > 0].sum() < 1e-6 < lam[x < 0].sum()                      # the heel alone
    assert abs(-(lam * x).sum() / (50.0 * arm) - 1) < 0.03, (lam * x).sum()
    lam, x, gap, arm = plate(-50.0)
    assert lam[x < -10].sum() < 1e-6 and abs((lam * x).sum() / (50.0 * arm) - 1) < 0.03, (lam, x)


def test_a_part_that_only_its_seat_stops_is_solved_and_one_that_lifts_off_is_refused():
    """A plate on a seat with one hinge line: nothing but the seat stops it
    rocking, so the stiffness without the seat is singular. The check
    solves it all the same (the seat is held on springs for the solves and
    they are taken out again exactly, whatever their stiffness), the seat's
    force is in the report and balances the hinge's, and a load that lifts
    the plate with nothing to hold it down is said to."""
    import trimesh

    from loadpath import pushseat
    from loadpath.verify import verify_part

    def study(fix_z, fz):
        hinge = [{"shape": _box([19.9, 0, -0.1], [20.1, 10, 4.1]), "fix": "y"},
                 {"shape": _box([19.9, 0, -0.5], [20.1, 10, 0.1]), "fix": "x"}]
        if fix_z:
            hinge.append({"shape": _box([19.9, 0, -0.1], [20.1, 10, 4.1]), "fix": "z"})
        return study_from_dict({
            "designSpace": [_box([0, 0, 0], [60, 10, 4])],
            "supports": hinge + [{"shape": _box([0, 0, -0.5], [60, 10, 0.1]), "push": "+z"}],
            "loads": [{"shape": _box([58, 0, 3.9], [60, 10, 4.1]), "force": [0, 0, fz]}],
            "material": "petg", "process": "fdm", "goal": {"safetyFactor": 2}, "resolution": {"pitch": 1.0}})

    m = trimesh.creation.box(extents=[60, 10, 4])
    m.apply_translation([30, 5, 2])
    seat, was = [], pushseat.HOLD
    try:
        for hold in (0.1, 1.0):
            pushseat.HOLD = hold
            rows = verify_part(study(True, 50.0), m)["reactions"]
            fz = [r["cases"][0]["force"][2] for r in rows]
            assert fz[3] < -50 and abs(sum(fz) - 50.0) < 0.05, fz       # pried: the heel bears, the hinge holds both
            seat.append(fz[3])
    finally:
        pushseat.HOLD = was
    assert abs(seat[0] / seat[1] - 1) < 5e-3, seat                      # the solves' own accuracy
    rep = verify_part(study(False, -50.0), m)                          # lying on the seat, pressed down at one end
    rows = rep["reactions"]
    assert abs(rows[-1]["cases"][0]["force"][2] + 50.0) < 0.05, rows[-1]
    # it bears on a band under the load (the patches there are cut finer), so the far end stays down
    assert rep["cases"][0]["deflection"] < 0.25, rep["cases"][0]["deflection"]
    try:
        verify_part(study(False, 50.0), m)
        raise AssertionError("a plate lifted off its seat was solved")
    except ValueError as exc:
        assert "lifts the part off its seat" in str(exc)


def test_a_push_support_is_refused_with_fix():
    base = {"designSpace": [_box([0, 0, 0], [60, 10, 4])],
            "supports": [{"shape": _box([19, 0, -0.1], [21, 10, 4.1])},
                         {"shape": _box([0, 0, -0.2], [60, 10, 0.05]), "push": "+z"}],
            "loads": [{"shape": _box([58, 0, 3.9], [60, 10, 4.1]), "force": [0, 0, 50]}],
            "material": "petg", "process": "fdm", "goal": {"safetyFactor": 2}, "resolution": {"pitch": 1.0}}
    assert study_from_dict(base).supports[1].push == (0.0, 0.0, 1.0)
    bad = json.loads(json.dumps(base))
    bad["supports"][1]["fix"] = "z"
    try:
        study_from_dict(bad)
        raise AssertionError("push with fix was accepted")
    except ValueError as exc:
        assert "push" in str(exc)


def test_the_optimizer_solves_a_push_support_as_contact():
    """The optimizer's own solve of a domain with a seat: the answer is in
    equilibrium with seat forces that push only where the part bears, on a
    fresh reading of the seat's compliance and on the solves between."""
    from loadpath import pushseat
    from loadpath.domain import build_domain
    from loadpath.simp import _Analysis

    st = study_from_dict({
        "designSpace": [_box([0, -10, 0], [30, 10, 20])],
        "supports": [{"shape": _box([-0.5, -8, 13], [0.6, -4, 17])}, {"shape": _box([-0.5, 4, 13], [0.6, 8, 17])},
                     {"shape": _box([-0.5, -10, 0], [0.2, 10, 20]), "push": "+x"}],
        "loads": [{"shape": _box([26, -8, 19], [30, 8, 20.5]), "force": [0, 0, -60]}],
        "material": "petg", "process": "fdm", "goal": {"safetyFactor": 2}, "resolution": {"pitch": 2.0}})
    d = build_domain(st)
    assert d.seats and not d.fixed[3 * d.seats[0][0]].all()
    fe = _Analysis(d, st.elastic(st.material), 1e-8)
    ps = pushseat.patches(d.seats, d.mesh.node_xyz, d.mesh.n_dof)
    x = np.ones(d.mesh.n_elem)
    us = [None] * len(d.loads)
    for k in range(3):                       # a fresh reading, then two solves on it as the design moves
        e = x * (1.0 - 0.1 * k * (d.mesh.node_xyz[d.mesh.enode[:, 0], 0] > 15))
        us = fe.solve(e, x, us)
        gap = np.array([p @ us[0] for p in ps])
        lam = fe._seat["lam"][0]
        z = np.array([np.abs(p[0::3]) @ d.mesh.node_xyz[:, 2] / np.abs(p[0::3]).sum() for p in ps])
        assert lam.min() >= 0 and lam.sum() > 0
        assert gap.min() > -0.03 * np.abs(us[0]).max(), (k, gap.min())
        assert lam[z > 15].sum() < 0.02 * lam.sum()       # the load tips the top off the wall: the heel bears


def _pried_plate(cases, pitch=1.0):
    """A plate on a seat under its whole face, hinged on a line across it at
    x = 20 (held there in z and y, and in x along the bottom), loaded at
    its far end: pried up, the heel bears on the seat; pressed down, the
    seat carries it. `cases`: [(name, load fields)]."""
    import trimesh

    st = study_from_dict({
        "designSpace": [_box([0, 0, 0], [60, 10, 4])],
        "supports": [{"shape": _box([19.9, 0, -0.1], [20.1, 10, 4.1]), "fix": "z"},
                     {"shape": _box([19.9, 0, -0.1], [20.1, 10, 4.1]), "fix": "y"},
                     {"shape": _box([19.9, 0, -0.5], [20.1, 10, 0.1]), "fix": "x"},
                     {"shape": _box([0, 0, -0.5], [60, 10, 0.1]), "push": "+z"}],
        "loadCases": [{"name": n, "loads": [{"shape": _box([58, 0, 3.9], [60, 10, 4.1]), **ld}]} for n, ld in cases],
        "material": "petg", "process": "fdm", "goal": {"safetyFactor": 2}, "resolution": {"pitch": pitch}})
    m = trimesh.creation.box(extents=[60, 10, 4])
    m.apply_translation([30, 5, 2])
    return st, m


def test_a_force_of_unknown_direction_on_a_seat_is_solved_in_each_direction():
    """A seat lets go where the force lifts it, so a force of unknown
    direction is not three axis solves added up: the contact is solved per
    direction (pushseat.SeatedSpread), in the gaps' space from the axis
    solves and the patches' unit pushes. Every direction's answer is the
    one a seated solve of that force alone gives, and the worst direction's
    compliance is the most over the sphere."""
    from dataclasses import replace
    from types import SimpleNamespace

    from loadpath import pushseat
    from loadpath.directions import expand
    from loadpath.verify import _analysis_grid, _solve_grid

    st, m = _pried_plate([("any", {"force": [0, 0, 50], "direction": "any"})])
    g = _analysis_grid(st, m, float(m.volume), 20000)
    hm, bc = g.hm, g.bc
    ps = pushseat.patches(bc.seats, hm.node_xyz, hm.n_dof)
    ks, held = pushseat.hold(ps, hm.node_xyz, g.pitch)
    gs = SimpleNamespace(**{**vars(g), "bc": replace(bc, springs=held)})

    def inner(r):
        return _solve_grid(st, gs, st.elastic(st.material), st.material.E, r)

    rhs, layout = expand(bc.loads, bc.spreads)
    _, _, _, cases = pushseat.solve_cases(inner, rhs, ps, st.material.E * ks, layout=layout, spreads=bc.spreads)
    case = cases[0]
    rows = case.samples(200)
    worst = rows[-1]
    assert worst[1] >= max(r[1] for r in rows[:-1]) * (1 - 1e-12)
    for d in (worst[0], np.array([0, 0, 1.0]), np.array([0, 0, -1.0]), np.array([0, 0.6, 0.8])):
        c, _, mu, lam, _ = case.at(d)
        _, us, lam1 = pushseat.solve(inner, [case.f(d)], ps, st.material.E * ks)
        assert abs(case.f(d) @ us[0] / c - 1) < 1e-9, d
        assert np.abs(case.u(d, mu) - us[0]).max() < 1e-9 * np.abs(us[0]).max(), d
        assert np.abs(lam - lam1[0]).max() < 1e-6 * max(lam.sum(), 1e-9)
    # pried up the heel bears; pressed down the plate lies on the seat, far stiffer
    up, down = case.at(np.array([0, 0, 1.0])), case.at(np.array([0, 0, -1.0]))
    assert up[0] > 100 * down[0] and up[3].sum() > 0 and down[3].sum() > 0


def test_the_checks_judge_a_seated_part_in_the_worst_direction_and_the_tet_check_solves_the_seat():
    """The grid's check and the tetrahedral one solve a force of unknown
    direction on a seated part per direction and judge it in the one that
    stresses the part most: no less than the worst axis. The tet check
    solves the seat itself now (it declined push studies): pried up it
    reads the grid's lift, pressed down the seat holds the plate."""
    from loadpath.crosscheck import crosscheck, unsupported
    from loadpath.verify import verify_part

    st, m = _pried_plate([("up", {"force": [0, 0, 50]}), ("down", {"force": [0, 0, -50]}),
                          ("any", {"force": [0, 0, 50], "direction": "any"})])
    assert unsupported(st) is None
    grid = {c["name"]: c for c in verify_part(st, m)["cases"]}
    assert grid["any"]["peakStress"] >= grid["up"]["peakStress"] * 0.999
    assert len(grid["any"]["worstDirection"]) == 3
    tet = {c["name"]: c for c in crosscheck(st, m, size=1.0)["cases"]}
    assert abs(tet["up"]["deflection"] / grid["up"]["deflection"] - 1) < 0.1, (tet["up"], grid["up"])
    assert tet["down"]["deflection"] < 0.01 * tet["up"]["deflection"]
    assert tet["any"]["peakStress"] >= tet["up"]["peakStress"] * 0.999
    assert tet["any"]["deflection"] > 0.5 * tet["up"]["deflection"]


def test_the_optimizer_designs_a_seated_part_for_a_force_of_unknown_direction():
    """The optimizer judges a force of unknown direction on a seated part in
    its worst direction with the contact solved in each: the design made for
    it is several times stiffer in its worst direction than the one made for
    the force as written, read the same way; and the run's own number is the
    delivered design's."""
    from loadpath.directions import collapse, expand
    from loadpath.domain import build_domain
    from loadpath.simp import E_MIN, PENALTY, SimpOptions, _Analysis, optimize

    def study(direction):
        ld = {"shape": _box([26, -8, 19], [30, 8, 20.5]), "force": [0, 0, -60]}
        if direction:
            ld["direction"] = direction
        return study_from_dict({
            "designSpace": [_box([0, -10, 0], [30, 10, 20])],
            "supports": [{"shape": _box([-0.5, -8, 13], [0.6, -4, 17])},
                         {"shape": _box([-0.5, 4, 13], [0.6, 8, 17])},
                         {"shape": _box([-0.5, -10, 0], [0.2, 10, 20]), "push": "+x"}],
            "loads": [ld], "material": "petg", "process": "fdm", "goal": {"safetyFactor": 2},
            "resolution": {"pitch": 2.0}})

    opts = SimpOptions(volume_fraction=0.25, filter_radius=3.0, max_iterations=30, escape=None)
    d = build_domain(study("any"))
    robust = optimize(d, 0.35, opts)
    nominal = optimize(build_domain(study(None)), 0.35, opts)
    rhs, layout = expand(d.loads, d.spreads)

    def worst(xp):
        fe = _Analysis(d, 0.35, 1e-8)
        us = fe.solve(E_MIN + xp ** PENALTY * (1 - E_MIN), xp, [None] * len(rhs), loads=rhs, layout=layout)
        return collapse(rhs, layout, us, d.spreads, seated=fe.seated_worst())[0].compliance

    assert abs(worst(robust.xphys) / robust.compliance - 1) < 1e-3, (worst(robust.xphys), robust.compliance)
    assert 2 * worst(robust.xphys) < worst(nominal.xphys)


def test_the_constrained_limit_steps_by_its_measured_response():
    """While every constrained run passes, the limit is relaxed by the
    square root of the first reading, then by the step the measured slope
    of reading against limit says lands near the target. A reading that
    did not move asks for a larger step, never an endless creep."""
    from loadpath.study import _relax_step

    assert abs(_relax_step([(1.0, 1.44)]) - 1.2) < 1e-9
    # a proportional response (slope -1) steps straight to the aim
    s = _relax_step([(1.0, 1.44), (1.2, 1.2)])
    assert abs(1.2 / s - 1.06) < 1e-6, s
    # a snap clip's 1.30 -> 1.32: no response, so the step is the largest
    # a quarter-power response allows, not another square root of 1.32
    s = _relax_step([(1.0, 1.30), (1.14, 1.32)])
    assert s > 2.0 and s <= 4.0, s
    assert _relax_step([(1.0, 1.3), (1.1, 1.3), (1.2, 1.29)]) <= 4.0


def test_a_clock_reaches_worker_searches_and_skips_the_settle():
    """--time-limit is a deadline every stage can read: pitch-start workers
    get it (they never saw it before), and a run past it delivers the part
    on supports rather than starting a second search."""
    import copy
    import time
    from types import SimpleNamespace as NS

    from loadpath import orient
    from loadpath.study import _clock, _deadline, _time_left

    assert _clock(None) is None
    stop = _clock(time.time() - 1.0)
    assert stop() and _deadline(stop) is not None and _time_left(stop) < 0
    inner = _clock(time.time() + 5.0)
    both = _clock(time.time() + 100.0, inner)
    assert abs(_deadline(both) - _deadline(inner)) < 1e-9 and not both()

    st = study_from_dict(_clamp(29))
    res = NS(study=st, best=0, seconds=60.0, outcomes=[NS(volume_fraction=0.2, mesh=None)])
    real = orient.choose_build_direction
    try:
        orient.choose_build_direction = lambda *a, **k: (_ for _ in ()).throw(AssertionError("screened"))
        calls = []
        out = orient.settle_supports(st, copy.copy(res), lambda s2: calls.append(s2), should_stop=stop)
        assert not calls and out.study.raw["supportDecision"]["skipped"]
        # 90 s left, a part takes 60: no room for the screen and a search part
        out = orient.settle_supports(st, copy.copy(res), lambda s2: calls.append(s2),
                                     should_stop=_clock(time.time() + 90.0))
        assert not calls and out.study.raw["supportDecision"]["chosen"] == "on supports"
    finally:
        orient.choose_build_direction = real


def test_a_block_that_barely_passes_draws_the_search_up_to_it():
    """The solid block's margin brackets the search: with a block at 1.11
    and a part at 30% reading 0.90, the next step reaches toward the top,
    not 35%; a strong block leaves the proportional step alone."""
    from loadpath.study import _toward_block

    nxt = 0.3 * 1.05 / 0.9
    assert _toward_block(nxt, 0.9, (0.3, 0.9), 1.11) > 0.75
    assert _toward_block(nxt, 0.9, (0.3, 0.9), 12.0) == nxt
    assert _toward_block(nxt, 0.9, (0.3, 0.9), None) == nxt
    assert _toward_block(nxt, 0.9, (0.3, 0.9), 0.8) == nxt     # a failing block says nothing here


def _wall_plate(fasteners=True, fix="xy"):
    """A plate screwed to a wall (the plane z = 0) by two M4 screws, sitting
    on the wall (a seat) and pulled straight off it: the held-out pillow
    block and Pi mount, written the way SKILL.md says (bores held across
    the screw, the face a push support)."""
    d = {"designSpace": [_box([-30, -10, 0], [30, 10, 6])],
         "supports": [{"shape": _box([-30, -10, -0.5], [30, 10, 0.3]), "push": "+z"}],
         "loads": [{"shape": _box([-5, -5, 5.5], [5, 5, 6]), "force": [0, 0, 40]}],
         "material": "petg", "process": "fdm", "goal": {"safetyFactor": 2}, "resolution": {"pitch": 1.0}}
    if fasteners:
        d["fasteners"] = [{"id": i, "size": "M4", "seat": [x, 0, 6], "direction": "-z", "grip": 6, "end": "face"}
                          for i, x in (("a", -22), ("b", 22))]
        d["supports"] += [{"region": "a-bore", "fix": fix}, {"region": "b-bore", "fix": fix}]
    else:
        d["supports"] += [{"shape": _cyl([x, 0, 3], "z", 8, 2.6), "fix": fix} for x in (-22, 22)]
    return d


def test_a_screw_holds_its_part_by_the_head_and_an_unheld_case_is_refused_at_check():
    """A screw whose bore is a support holds the part along the screw, one
    way, by its head: the plate pulled off its wall bears on its heads and
    check passes. Held only across the screws (inline shapes, no screw),
    nothing holds it off the wall and check says so in a second, naming the
    case and the way it goes, instead of a run that flies off and then calls
    it a limit other than strength."""
    from loadpath.fasteners import expand_fasteners

    d = expand_fasteners(_wall_plate())
    holds = [s for s in d["supports"] if s.get("region", "").endswith("-seat")]
    assert len(holds) == 2 and all(s["push"] == [0, 0, -1] for s in holds), d["supports"]
    c = check_study(_wall_plate())
    assert c["ok"], c
    # fixed along the screw too: nothing to add
    assert not [s for s in expand_fasteners(_wall_plate(fix="all"))["supports"] if s.get("region", "").endswith("-seat")]
    t0 = time.time()
    c = check_study(_wall_plate(fasteners=False))
    assert not c["ok"] and "lifts the part off its seat" in c["error"] and "along +z" in c["error"], c
    assert time.time() - t0 < 30


def test_a_holes_roof_across_the_layers_is_a_bridge_not_unreachable_drawn_geometry():
    """A screw hole lying across the layers (a bracket printed on its side)
    has a roof the 45-degree reach misses: a printer bridges it, so check
    does not call the ring round it drawn geometry that needs support. A
    peg drawn in mid-air still is."""
    from loadpath.domain import _unprintable_preserve

    dims = (20, 20, 20)
    g = np.indices(dims).transpose(1, 2, 3, 0) + 0.5
    active = np.ones(dims, bool)
    # a 6-cell hole along x through the block, its axis at y = 10, z = 10
    hole = (g[..., 1] - 10) ** 2 + (g[..., 2] - 10) ** 2 < 9
    active &= ~hole
    ring = ~hole & ((g[..., 1] - 10) ** 2 + (g[..., 2] - 10) ** 2 < 25)
    args = ((2, 1), dims, active.reshape(-1), (ring & active).reshape(-1), 1.0, (0, 0, 0))
    assert _unprintable_preserve(*args) != []                    # the 45-degree reach alone: the roof
    assert _unprintable_preserve(*args, process="fdm") == []     # bridged
    # a peg standing out over nothing (the bed is z = 0; the peg at z 12-14 from x 0-6, the block x 6-20)
    active2 = np.zeros(dims, bool)
    active2[6:, :, :] = True
    peg = np.zeros(dims, bool)
    peg[:6, 8:12, 12:14] = True
    active2 |= peg
    w = _unprintable_preserve((2, 1), dims, active2.reshape(-1), peg.reshape(-1), 1.0, (0, 0, 0), process="fdm")
    assert w and "cannot be reached" in w[0]


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


def _socket(support):
    """A block with a 20 mm tube through it along x (a tee's rail), the
    block pushed down 100 N and pulled 30 N along the tube."""
    return {"designSpace": [_box([-20, -16, -16], [20, 16, 16])],
            "keepOut": [{"id": "rail", "shape": _cyl([0, 0, 0], "x", 60, 10), "fit": "slip"}],
            "supports": [dict(support, region="rail")],
            "loadCases": [{"name": "down", "loads": [{"shape": _box([-20, -16, 15.5], [20, 16, 16]),
                                                      "force": [0, 0, -100]}]},
                          {"name": "along", "loads": [{"shape": _box([-20, -16, 15.5], [20, 16, 16]),
                                                       "force": [30, 0, -100]}]}],
            "material": "petg", "process": "fdm", "goal": {"safetyFactor": 2}, "resolution": {"pitch": 1.5}}


def test_a_tube_in_a_bore_bears_radially_and_a_glued_slip_bore_is_named():
    """push 'radial' on a bore: the wall is twelve seats round the axis,
    each pushing out along its middle's normal; with friction the part is
    held along the tube too, without it check refuses the case that pulls
    it along. Held fixed all round where the fit says the tube slips,
    check names the glue."""
    from loadpath.domain import build_domain, radial_seats

    pts = np.array([[0.0, 10 * np.cos(t), 10 * np.sin(t)] for t in np.linspace(0, 2 * np.pi, 72, endpoint=False)])
    secs = radial_seats(pts, [0, 0, 0], [1, 0, 0])
    assert len(secs) == 12 and sum(len(s) for s, _ in secs) == 72
    for sub, n in secs:
        assert abs(n[0]) < 1e-12 and abs(np.linalg.norm(n) - 1) < 1e-12
        assert (pts[sub] @ n).min() > 10 * np.cos(np.pi / 12) - 1e-9   # each sector faces its normal
    dom = build_domain(study_from_dict(_socket({"push": "radial", "friction": 0.3})))
    assert len(dom.seats) == 12
    c = check_study(_socket({"push": "radial"}))
    assert not c["ok"] and "'along'" in c["error"], c
    c = check_study(_socket({"fix": "all"}))
    assert c["ok"] and any("as if glued to what is in it" in w for w in c["warnings"]), c


def _hex_socket(support):
    """A 13 mm hex through a 40 x 40 x 8 block (a wrench's ring), turned by
    20 N across its +x end."""
    def side(a):
        a = np.radians(a)
        return {"type": "box", "center": [0, 0, 4], "size": [13.2, 40, 12],
                "axes": [[np.cos(a), np.sin(a), 0], [-np.sin(a), np.cos(a), 0]]}
    return {"designSpace": [_box([-20, -20, 0], [20, 20, 8])],
            "keepOut": [{"id": "nut", "shape": {"type": "intersect", "shapes": [side(0), side(60), side(120)]}}],
            "supports": [dict(support, region="nut")],
            "loadCases": [{"name": "turn", "loads": [{"shape": _box([19, -20, 0], [20, 20, 8]),
                                                      "force": [0, 20, 0]}]}],
            "material": "petg", "process": "fdm", "goal": {"safetyFactor": 2}, "resolution": {"pitch": 1.0}}


def test_a_nut_in_a_hex_bears_on_its_flats_and_a_glued_socket_is_named():
    """push 'normal' on a keep-out: its wall's nodes grouped by the way the
    wall faces, each group pushing along its mean normal; a hex's six
    flats are six seats, and the torque on them is held by bearing. Held
    fixed on every wall, check names the glue."""
    from loadpath.domain import build_domain, wall_seats
    from loadpath.shapes import shape_from_spec

    hexs = shape_from_spec(_hex_socket({"fix": "all"})["keepOut"][0]["shape"])
    t = np.linspace(0, 2 * np.pi, 360, endpoint=False)
    r = 6.6 / np.cos((t + np.pi / 6) % (np.pi / 3) - np.pi / 6)        # on the flats, apothem 6.6
    pts = np.stack([r * np.cos(t), r * np.sin(t), np.full_like(t, 4.0)], axis=1)
    groups = wall_seats(pts, hexs)
    big = [(s, n) for s, n in groups if len(s) > 10]
    assert len(big) == 6, [len(s) for s, _ in groups]
    for sub, n in big:
        assert abs(n[2]) < 1e-6 and abs(np.linalg.norm(n) - 1) < 1e-9
        assert np.allclose(pts[sub] @ n, 6.6, atol=1e-3)               # each flat at the apothem along its normal
    dom = build_domain(study_from_dict(_hex_socket({"push": "normal", "friction": 0.2})))
    assert 6 <= len(dom.seats) <= 12
    c = check_study(_hex_socket({"push": "normal", "friction": 0.2}))
    assert c["ok"] and not any("as if glued" in w for w in c["warnings"]), c
    c = check_study(_hex_socket({"fix": "all"}))
    assert c["ok"] and any("fixed on every wall, as if glued" in w for w in c["warnings"]), c


def test_a_fit_on_a_bore_with_a_flat_moves_every_wall_by_the_fits_clearance():
    """A D-shaft's hole (a cylinder less a box) with fit slip: the fit is
    the round member's, and the flat moves out by the same clearance."""
    from loadpath.fits import expand_fits

    d = {"designSpace": [_box([-20, -20, 0], [20, 20, 12])],
         "keepOut": [{"id": "shaft", "fit": "slip", "shape": {"type": "subtract", "shapes": [
             _cyl([0, 0, 4.5], "z", 11, 4), _box([3.5, -5, -2], [5, 5, 11])]}}],
         "supports": [{"region": "shaft"}], "material": "pa12", "process": "sls",
         "loadCases": [{"name": "turn", "loads": [{"shape": _box([19, -20, 0], [20, 20, 12]), "force": [0, 5, 0]}]}]}
    e = expand_fits(d)
    row = e["_fits"][0]
    cyl, cut = e["keepOut"][0]["shape"]["shapes"]
    g = 0.5 * row["designed"] - 4.0                           # the clearance on the radius
    assert abs(cyl["r"] - (4.0 + g)) < 1e-4 and abs(cut["min"][0] - (3.5 + g)) < 1e-4
    assert abs(row["inscribed"] - 2 * (3.5 + g)) < 1e-3       # the largest pin through it: the flat's
    bad = dict(d, keepOut=[{"id": "nut", "fit": "slip", "shape": {"type": "intersect", "shapes": [
        _box([-6.6, -20, -1], [6.6, 20, 13]), _box([-20, -6.6, -1], [20, 6.6, 13])]}}], supports=[{"region": "nut"}])
    try:
        expand_fits(bad)
        raise AssertionError("a fit on a socket with no round member was taken")
    except ValueError as exc:
        assert "plain cylinders" in str(exc)


def test_a_part_held_only_on_its_pressed_bearing_is_told_it_turns():
    """A pulley on a 608 pressed into it: a rope's pull written at one
    angle is named, a pull at every angle round the axis is not."""
    st = {"designSpace": [_cyl([0, 0, 0], "z", 10, 30)],
          "keepOut": [{"id": "bearing", "fit": "press", "shape": _cyl([0, 0, 0], "z", 12, 11)}],
          "supports": [{"region": "bearing"}],
          "loadCases": [{"name": "lift", "loads": [{"shape": _box([-5, -30, -5], [5, -29, 5]),
                                                     "force": [0, -500, 0]}]}],
          "material": "petg", "process": "fdm", "goal": {"safetyFactor": 2}, "resolution": {"pitch": 1.5}}
    c = check_study(st)
    assert c["ok"] and any("turns on that bearing" in w and '"direction": "xy"' in w for w in c["warnings"]), c
    st["loadCases"][0]["loads"][0]["direction"] = "xy"
    c = check_study(st)
    assert c["ok"] and not any("turns on that bearing" in w for w in c["warnings"]), c
