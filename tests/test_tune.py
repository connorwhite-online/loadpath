"""Tuning a field graph's numbers against an FE check of its part (tune.py),
from Python and through the editor's server."""
import json
import os
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from loadpath.schema import lint  # noqa: E402
from loadpath.tune import get_at, tune_field  # noqa: E402

# a 60 mm cantilever, 10 mm wide, its height the knob; 25 N at the tip
STUDY = {"name": "bar", "designSpace": [{"type": "box", "min": [0, -6, -6], "max": [60, 6, 6]}],
         "preserve": [{"id": "root", "shape": {"type": "box", "min": [0, -5, -5], "max": [4, 5, 5]}},
                      {"id": "tip", "shape": {"type": "box", "min": [56, -5, -5], "max": [60, 5, 5]}}],
         "supports": [{"region": "root"}],
         "loadCases": [{"name": "bend", "loads": [{"region": "tip", "force": [0, 0, -25]}]}],
         "material": "petg", "goal": {"safetyFactor": 4}}
GRAPH = {"name": "bar", "nodes": {"bar": {"op": "shape", "shape": {"type": "box", "min": [0, -5, -5],
                                                                   "max": [60, 5, 5]}}},
         "output": "bar",
         "tune": {"study": "study.json",
                  "params": [{"path": "nodes/bar/shape/max/2", "also": ["-nodes/bar/shape/min/2"],
                              "min": 1.5, "max": 5}],
                  "pitch": 0.8, "cells": 6000, "evaluations": 8}}


def _dir():
    d = tempfile.mkdtemp()
    with open(os.path.join(d, "study.json"), "w") as fh:
        json.dump(STUDY, fh)
    with open(os.path.join(d, "graph.json"), "w") as fh:
        json.dump(GRAPH, fh)
    return d


def test_tune_finds_a_lighter_bar_that_still_holds():
    """The bar starts over-strong (SF ~5.5 against 4): the tuner thins it,
    stays inside the knob's range, keeps the box symmetric (`also`) and
    returns the lightest candidate that met the limit."""
    d = _dir()
    assert not lint(GRAPH, "field")
    out = tune_field(json.loads(json.dumps(GRAPH)), d, os.path.join(d, "out"))
    best, start = out["best"], out["start"]
    assert start["ok"] and best["ok"] and best["safetyFactor"] >= 4
    assert best["mass"] < 0.92 * start["mass"], (best, start)
    assert all(1.5 <= r["values"][0] <= 5 for r in out["history"])
    assert len(out["history"]) <= 8
    assert best["mass"] == min(r["mass"] for r in out["history"] if r["ok"])
    h = get_at(out["doc"], "nodes/bar/shape/max/2")
    assert abs(h - best["values"][0]) < 1e-9 and abs(get_at(out["doc"], "nodes/bar/shape/min/2") + h) < 1e-9
    saved = json.load(open(os.path.join(d, "out", "tuned.json")))
    assert saved["nodes"] == out["doc"]["nodes"] and saved["tune"] == GRAPH["tune"]


def test_tune_names_its_mistakes():
    d = _dir()
    for change, msg in (({"path": "nodes/bar/shape/max/7"}, "nothing at"), ({"path": "nodes/bar/op"}, "not a number"),
                        ({"min": 5, "max": 1}, "min < max")):
        g = json.loads(json.dumps(GRAPH))
        g["tune"]["params"][0].update(change)
        with pytest.raises(ValueError, match=msg):
            tune_field(g, d)
    g = json.loads(json.dumps(GRAPH))
    del g["tune"]
    with pytest.raises(ValueError, match="no 'tune' block"):
        tune_field(g, d)


def _request(url, body=None):
    req = urllib.request.Request(url, data=None if body is None else json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


def test_editor_server_tunes_and_hands_back_the_values():
    from loadpath.editor import serve_editor

    d = _dir()
    holder = {}
    threading.Thread(target=serve_editor, args=(os.path.join(d, "graph.json"),),
                     kwargs=dict(port=0, open_browser=False, ready=lambda s: holder.update(srv=s)), daemon=True).start()
    for _ in range(100):
        if "srv" in holder:
            break
        time.sleep(0.05)
    base = f"http://127.0.0.1:{holder['srv'].server_port}/"
    try:
        g = json.loads(json.dumps(GRAPH))
        g["tune"]["evaluations"] = 4
        code, body = _request(base + "api/tune", {"doc": {k: v for k, v in g.items() if k != "tune"}})
        assert code == 400
        code, body = _request(base + "api/tune", {"doc": g})
        jid = json.loads(body)["id"]
        for _ in range(1200):
            job = json.loads(_request(base + f"api/job?id={jid}")[1])
            if job["state"] != "running":
                break
            time.sleep(0.1)
        assert job["state"] == "done", job.get("error")
        assert job["kind"] == "tune" and any(e.get("event") == "tune" for e in job["events"])
        assign = dict(job["assign"])
        assert set(assign) == {"nodes/bar/shape/max/2", "nodes/bar/shape/min/2"}
        assert abs(assign["nodes/bar/shape/max/2"] + assign["nodes/bar/shape/min/2"]) < 1e-9
        assert assign["nodes/bar/shape/max/2"] == job["best"]["values"][0]
        assert _request(base + job["files"]["tuned"].lstrip("/"))[0] == 200
    finally:
        holder["srv"].shutdown()


# a skin-and-gyroid bar under the same kind of cantilever load, for the
# homogenized check: fully resolved (meshed at 0.12 mm, verified on 150k,
# 400k and 0.9M voxels) it deflects 0.95-1.00 mm and weighs 6.86 g at the
# old PETG card's E = 2000 MPa (linear: deflection scales with 1/E). That
# reference is the designed geometry, an ideal solid, so the study names
# "process": "none"; with the material's own process (fdm) the lattice also
# takes the printed-lattice knockdown (homogenized._print_factor) and reads
# softer, which test_homogenized_lattice_matches_the_resolved_one checks.
LATTICE_STUDY = {**STUDY, "process": "none", "preserve": [{"id": "root", "shape": {"type": "box", "min": [0, -6, -6], "max": [4, 6, 6]}},
                                       {"id": "tip", "shape": {"type": "box", "min": [56, -6, -6], "max": [60, 6, 6]}}],
                 "designSpace": [{"type": "box", "min": [0, -6, -6], "max": [60, 6, 6]}],
                 "loadCases": [{"name": "bend", "loads": [{"region": "tip", "force": [0, 0, -40]}]}],
                 "goal": {"safetyFactor": 2}}
LATTICE_BAR = {"name": "latbar", "nodes": {
    "bar": {"op": "shape", "shape": {"type": "box", "min": [0, -6, -6], "max": [60, 6, 6]}},
    "skin": {"op": "shell", "input": "bar", "thickness": 1.0},
    "core": {"op": "lattice", "type": "gyroid", "cell": 6, "thickness": 0.8},
    "fill": {"op": "intersect", "inputs": ["core", "bar"]},
    "ends": {"op": "shape", "shape": {"type": "union", "shapes": [
        {"type": "box", "min": [0, -6, -6], "max": [4, 6, 6]}, {"type": "box", "min": [56, -6, -6], "max": [60, 6, 6]}]}},
    "part": {"op": "union", "inputs": ["skin", "fill", "ends"]}}, "output": "part"}


def test_homogenized_lattice_matches_the_resolved_one():
    """With the lattice homogenized, the bar weighs what the resolved one
    does (within 3%) and deflects up to a fifth more on this coarse grid
    (11% from 40k cells), never much less (the fits are the conservative
    reading); a thicker wall is stiffer and
    heavier; a solid core is stiffer still."""
    from loadpath.homogenized import homogenized_check
    from loadpath.spec import study_from_dict

    st = study_from_dict(LATTICE_STUDY)
    r = homogenized_check(LATTICE_BAR, st, cells=8000)
    d = r["cases"][0]["deflection"]
    ref = 0.975 * 2000 / st.material.E
    assert -0.05 < d / ref - 1 < 0.2 and abs(r["mass"] / 6.86 - 1) < 0.03, r
    assert r["lattice"] == "core" and r["bodies"] == 1 and r["cases"][0]["inLattice"] in (True, False)
    # printed in PETG filament (the material's process): the published
    # printed-lattice tests make it softer than designed, never stiffer
    fdm = study_from_dict({k: v for k, v in LATTICE_STUDY.items() if k != "process"})
    assert fdm.process is not None and fdm.process.kind == "fdm"
    r_fdm = homogenized_check(LATTICE_BAR, fdm, cells=8000)
    assert r_fdm["cases"][0]["deflection"] > 1.02 * d and r_fdm["latticePrint"]["factor"] is not None, r_fdm
    thick = json.loads(json.dumps(LATTICE_BAR))
    thick["nodes"]["core"]["thickness"] = 1.6
    r2 = homogenized_check(thick, st, cells=8000)
    assert r2["cases"][0]["deflection"] < 0.95 * d and r2["mass"] > r["mass"]
    solid = json.loads(json.dumps(LATTICE_BAR))
    solid["nodes"]["fill"] = {"op": "shape", "shape": {"type": "box", "min": [0, -6, -6], "max": [60, 6, 6]}}
    with pytest.raises(ValueError, match="no lattice"):
        homogenized_check(solid, st, cells=8000)
    net = json.loads(json.dumps(LATTICE_BAR))
    net["nodes"]["core"]["form"] = "network"
    with pytest.raises(ValueError, match="sheet"):
        homogenized_check(net, st, cells=8000)


def test_tune_grades_a_lattice_homogenized():
    """A lattice part is tuned on the homogenized analysis: the wall comes
    down from an over-strong start while the bar still meets its limit."""
    d = tempfile.mkdtemp()
    with open(os.path.join(d, "study.json"), "w") as fh:
        json.dump({**LATTICE_STUDY, "goal": {"maxDeflection": 1.3}}, fh)
    g = json.loads(json.dumps(LATTICE_BAR))
    g["nodes"]["core"]["thickness"] = 1.6
    g["tune"] = {"study": "study.json", "params": [{"path": "nodes/core/thickness", "min": 0.5, "max": 2.0}],
                 "cells": 5000, "evaluations": 6}
    out = tune_field(g, d)
    assert out["analysis"] == "homogenized"
    best = out["best"]
    assert best and best["ok"] and best["values"][0] < 1.6 and best["mass"] < out["start"]["mass"], out["history"]
    assert best["deflection"] <= 1.3


def test_sweep_traces_the_trade_off():
    """Held to a higher safety factor the bar comes out heavier; every
    candidate on the front is beaten by none on mass, deflection and SF."""
    d = _dir()
    g = json.loads(json.dumps(GRAPH))
    # the thickest bar the knob allows reads SF ~5.4, so 5 is the top
    g["tune"].update(evaluations=6, sweep={"safetyFactor": [3, 5]})
    assert not lint(g, "field")
    out = tune_field(g, d)
    (lo_sf, hi_sf) = out["sweep"]
    assert lo_sf["safetyFactor"] == 3 and hi_sf["safetyFactor"] == 5
    assert lo_sf["best"]["mass"] < out["best"]["mass"] < hi_sf["best"]["mass"], out["sweep"]
    assert hi_sf["best"]["safetyFactor"] >= 5 and lo_sf["best"]["safetyFactor"] >= 3
    front = out["front"]
    assert len(front) >= 3 and [r["mass"] for r in front] == sorted(r["mass"] for r in front)
    for r in front:
        assert not any(o["mass"] < r["mass"] and o["deflection"] < r["deflection"]
                       and o["safetyFactor"] > r["safetyFactor"] for o in front)
    bad = json.loads(json.dumps(g))
    bad["tune"]["sweep"] = {"mass": [1]}
    assert lint(bad, "field")


def test_fractional_cells_read_smoothly():
    """Whole voxels read a bar's height in whole-cell steps, so the safety
    factor jumped about as the tuner thinned it (and the sweep found a
    thinner bar 'stronger'); filled shares move it with the height."""
    from loadpath.fields import FieldGraph
    from loadpath.spec import study_from_dict
    from loadpath.verify import verify_part

    study = study_from_dict(STUDY)
    sf = []
    for h in np.linspace(3.2, 4.4, 7):
        g = json.loads(json.dumps(GRAPH))
        g["nodes"]["bar"]["shape"].update(min=[0, -5, -h], max=[60, 5, h])
        mesh = FieldGraph(g).mesh(pitch=0.8, project=1)
        sf.append(verify_part(study, mesh, max_cells=6000, refine=0, fractional=True)["safetyFactor"])
    assert all(b > a for a, b in zip(sf, sf[1:])), sf


def test_the_confirming_grid_resolves_the_thinnest_wall():
    """The check runs at two cells or more through the thinnest shell wall
    (a field-driven wall reads its least value), capped."""
    from loadpath.fields import FieldGraph
    from loadpath.tune import CELLS_THROUGH_WALL, check_cells, thinnest_wall

    g = FieldGraph(LATTICE_BAR)
    assert thinnest_wall(g) == 1.0
    n, wall = check_cells(g, None, 5000, homogenized=True)
    assert wall == 1.0 and n >= int(60 * 12 * 12 * 0.6 / (1.0 / CELLS_THROUGH_WALL) ** 3) - 1
    assert check_cells(g, None, 5000, homogenized=True, cap=20_000)[0] == 20_000
    graded = json.loads(json.dumps(LATTICE_BAR))
    graded["nodes"]["t"] = {"op": "remap", "input": {"op": "coordinate", "axis": "x"}, "from": [0, 60], "to": [0.6, 2]}
    graded["nodes"]["skin"]["thickness"] = "t"
    assert abs(thinnest_wall(FieldGraph(graded)) - 0.6) < 0.02
    solid = {"nodes": {"b": {"op": "shape", "shape": {"type": "box", "min": [0, 0, 0], "max": [9, 9, 9]}}}}
    assert check_cells(FieldGraph(solid), None, 5000, homogenized=False, volume=729.0) == (10_000, None)


def test_a_thin_skin_that_only_passes_on_the_coarse_grid_is_not_ok(monkeypatch):
    """A thin skin read on a coarse grid can look stiffer than it is. If no
    candidate holds its limit on the confirming grids, the tuner says so
    (ok false, unconfirmed) instead of handing back a part that fails."""
    import loadpath.homogenized as hom

    seen = []

    def fake(doc, study, base_dir=None, cells=30_000, graph=None, **kw):
        t = doc["nodes"]["core"]["thickness"]
        seen.append(cells)
        coarse = cells <= 5000
        return {"mass": 5.0 + t, "safetyFactor": 3.0, "bodies": 1,
                "cases": [{"deflection": 1.0 / t + (0.0 if coarse else 0.6)}]}

    monkeypatch.setattr(hom, "homogenized_check", fake)
    d = tempfile.mkdtemp()
    with open(os.path.join(d, "study.json"), "w") as fh:
        json.dump({**LATTICE_STUDY, "goal": {"maxDeflection": 1.0}}, fh)
    g = json.loads(json.dumps(LATTICE_BAR))
    g["nodes"]["core"]["thickness"] = 1.6
    g["tune"] = {"study": "study.json", "params": [{"path": "nodes/core/thickness", "min": 0.5, "max": 2.0}],
                 "cells": 5000, "evaluations": 5}
    out = tune_field(g, d)
    best = out["best"]
    # nothing holds 1.0 mm on the fine grids: no result, or one marked
    # unconfirmed, never ok
    assert best is None or (best["ok"] is False and best["unconfirmed"]), best
    fine = sorted({c for c in seen if c > 5000})
    assert len(fine) == 2 and fine[1] >= 40_000 and abs(fine[0] / fine[1] - 0.7) < 0.01, fine
    checked = [r for r in out["history"] if "check" in r]
    assert checked and all(r["check"]["thinnestWall"] == 1.0 and not r["check"]["ok"] for r in checked)


def test_a_failed_check_teaches_the_search(monkeypatch):
    """The fine grid reads thin candidates worse than the start's ratio
    says: the first check fails, its ratio is learned, and a thicker
    candidate that really holds the limit comes back ok."""
    import loadpath.homogenized as hom

    def fake(doc, study, base_dir=None, cells=30_000, graph=None, **kw):
        t = doc["nodes"]["core"]["thickness"]
        d = 1.0 / t * (1.0 if cells <= 5000 else 1.0 + 0.3 * (2.0 - t))
        return {"mass": 5.0 + t, "safetyFactor": 3.0, "bodies": 1, "cases": [{"deflection": d}]}

    monkeypatch.setattr(hom, "homogenized_check", fake)
    d = tempfile.mkdtemp()
    with open(os.path.join(d, "study.json"), "w") as fh:
        json.dump({**LATTICE_STUDY, "goal": {"maxDeflection": 1.0}}, fh)
    g = json.loads(json.dumps(LATTICE_BAR))
    g["nodes"]["core"]["thickness"] = 1.6
    g["tune"] = {"study": "study.json", "params": [{"path": "nodes/core/thickness", "min": 0.5, "max": 2.0}],
                 "cells": 5000, "evaluations": 10}
    best = tune_field(g, d)["best"]
    t = best["values"][0]
    assert best["ok"] and best["check"]["ok"] and 1.25 < t < 1.6, best
    assert best["check"]["deflection"] <= 1.0


def test_a_part_with_no_lattice_reads_on_the_same_grid():
    """solid_ok: a hollow bar (the lattice bar's skin and ends, no core)
    reads on the fill-weighted grid, weighs its shell and bends more than
    the same bar with its gyroid; without solid_ok it is refused."""
    from loadpath.homogenized import homogenized_check
    from loadpath.spec import study_from_dict

    st = study_from_dict(LATTICE_STUDY)
    hollow = json.loads(json.dumps(LATTICE_BAR))
    hollow["nodes"]["part"]["inputs"] = ["skin", "ends"]
    with pytest.raises(ValueError, match="no lattice"):
        homogenized_check(hollow, st, cells=8000)
    r = homogenized_check(hollow, st, cells=8000, solid_ok=True)
    shell = 60 * 12 * 12 - 52 * 10 * 10      # 1 mm skin round a box, solid 4 mm ends
    assert abs(r["volume"] / shell - 1) < 0.03 and r["lattice"] is None and r["latticeDensity"] is None, r
    cored = homogenized_check(LATTICE_BAR, st, cells=8000)
    assert r["cases"][0]["deflection"] > cored["cases"][0]["deflection"] and r["mass"] < cored["mass"]
