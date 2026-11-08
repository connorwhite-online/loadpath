"""
FE model export: the delivered part as a Nastran (.bdf) or Abaqus /
CalculiX (.inp) deck, read back and solved to beam theory.

Run: `python tests/test_fe_export.py` (or pytest).
"""
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest
import trimesh

from loadpath.crosscheck import _p2_mesh, _solve, mesher_missing
from loadpath.fe_export import export_fe, read_nastran
from loadpath.spec import study_from_dict

pytestmark = pytest.mark.skipif(mesher_missing() is not None, reason="no tet mesher")

BAR = {
    "name": "cantilever",
    "designSpace": {"type": "box", "min": [0, 0, 0], "max": [100, 10, 10]},
    "supports": [{"shape": {"type": "box", "min": [-1, -1, -1], "max": [0.01, 11, 11]}}],
    "loadCases": [{"name": "down", "loads": [{"shape": {"type": "box", "min": [99.99, -1, -1], "max": [101, 11, 11]},
                                              "force": [0, 0, -10]}]},
                  {"name": "side", "loads": [{"shape": {"type": "box", "min": [99.99, -1, -1], "max": [101, 11, 11]},
                                              "force": [0, 5, 0]}]}],
    "material": {"E": 2000, "nu": 0.3, "strength": 50, "density": 1.2},
    "goal": {"safetyFactor": 2},
}


def _bar():
    bar = trimesh.creation.box(extents=[100, 10, 10])
    bar.apply_translation([50, 5, 5])
    return bar


def test_nastran_deck_reads_back_and_bends_as_beam_theory(tmp_path):
    path = tmp_path / "bar.bdf"
    info = export_fe(study_from_dict(BAR), _bar(), str(path), size=3.0)
    assert info["format"] == "nastran" and info["cases"] == ["down", "side"]
    text = path.read_text()
    assert "SOL 101" in text and text.count("SUBCASE") == 2 and text.rstrip().endswith("ENDDATA")
    # every card in large field fits its columns
    assert all(len(ln) <= 72 for ln in text.splitlines() if not ln.startswith("$"))
    d = read_nastran(str(path))
    assert len(d["nodes"]) == info["nodes"] and d["conn"].shape == (info["elements"], 10)
    assert d["mat"] == {"type": "MAT1", "E": 2000.0, "nu": 0.3, "rho": pytest.approx(1.2e-9)}
    assert d["subcases"] == ["down", "side"]
    # the nodal forces sum to each case's load
    assert np.allclose(d["forces"][101][1].sum(axis=0), [0, 0, -10], atol=1e-9)
    assert np.allclose(d["forces"][102][1].sum(axis=0), [0, 5, 0], atol=1e-9)
    # solved as read: the tip deflection is the cantilever's, F L^3 / 3 E I
    nodes, conn = d["nodes"], d["conn"]
    fixed = np.zeros(3 * len(nodes), bool)
    for comps, ids in d["spc"]:
        for c in str(comps):
            fixed[3 * np.asarray(ids) + int(c) - 1] = True
    nv = int(conn[:, :4].max()) + 1
    edges = np.sort(conn[:, [0, 1]], axis=1)       # (only its count is used for vertices)
    _n, _c, edges = _p2_mesh(nodes[:nv], conn[:, :4])
    loads = []
    for sid in (101, 102):
        f = np.zeros((nv, 3))
        ids, vec = d["forces"][sid]
        np.add.at(f, ids, vec)
        loads.append(("x", f, ids))
    us = _solve(nodes, conn, 2000.0, 0.3, fixed, None, loads, nv, edges)
    tip = nodes[:nv, 0] > 99.9
    want = 10 * 100 ** 3 / (3 * 2000 * (10 * 10 ** 3 / 12))
    got = -us[0][:nv][tip, 2].mean()
    assert abs(got - want) / want < 0.05, (got, want)


def test_abaqus_deck_and_pins(tmp_path):
    lug = {"designSpace": {"type": "box", "min": [0, -15, -4], "max": [60, 15, 4]},
           "keepOut": [{"id": "pin", "shape": {"type": "cylinder", "center": [48, 0, 0], "axis": "z",
                                               "length": 12, "r": 4}}],
           "supports": [{"shape": {"type": "box", "min": [-1, -16, -5], "max": [0.01, 16, 5]}},
                        {"region": "pin", "fix": "pin"}],
           "loadCases": [{"name": "push", "loads": [{"shape": {"type": "box", "min": [20, 14.99, -5],
                                                               "max": [30, 16, 5]}, "force": [0, -50, 0]}]}],
           "material": "pa12", "process": {"type": "fdm", "build": "+z"}, "goal": {"safetyFactor": 2}}
    body = trimesh.creation.box(extents=[60, 30, 8])
    body.apply_translation([30, 0, 0])
    hole = trimesh.creation.cylinder(radius=4, height=12, sections=48)
    hole.apply_translation([48, 0, 0])
    part = body.difference(hole)
    s = study_from_dict(lug)
    inp = tmp_path / "lug.inp"
    info = export_fe(s, part, str(inp), size=2.5)
    t = inp.read_text()
    assert info["format"] == "abaqus" and "*ELEMENT,TYPE=C3D10" in t and "*TRANSFORM,NSET=S1,TYPE=C" in t
    assert "SOLVER=" not in t and t.count("*STEP") == 1
    # the layered FDM material as engineering constants in the layers' frame
    assert "TYPE=ENGINEERING CONSTANTS" in t
    bdf = tmp_path / "lug.bdf"
    export_fe(s, part, str(bdf), size=2.5)
    d = read_nastran(str(bdf))
    assert d["mat"]["type"] == "MAT9" and len(d["mat"]["G"]) == 21
    assert 1 in d["cords"] and (d["cd"] == 1).sum() > 10
    assert any(c == "13" for c, _ in d["spc"])


def test_cli(tmp_path):
    study = tmp_path / "bar.json"
    study.write_text(json.dumps(BAR))
    stl = tmp_path / "bar.stl"
    _bar().export(stl)
    out = tmp_path / "bar.nas"
    r = subprocess.run([sys.executable, "-m", "loadpath", "fe-export", str(study), str(stl), "--out", str(out),
                        "--size", "4"], capture_output=True, text=True, cwd=Path(__file__).resolve().parents[1])
    assert r.returncode == 0, r.stdout + r.stderr
    assert json.loads(r.stdout)["format"] == "nastran" and out.exists()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
