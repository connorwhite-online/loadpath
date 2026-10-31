"""
Tests for cadimport.py: a part brought in from CAD as a draft study. Run:
`python3 tests/test_cadimport.py` (about half a minute; the STEP tests skip
without OpenCascade, loadpath[step]).

The test part is a link: a 60 x 30 x 8 mm plate held by four M3 bolts
(counterbored at one end) with a 10 mm bearing bore at the other end.
"""
import json
import os
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import trimesh  # noqa: E402

from loadpath.cadimport import detect_features, heal_mesh, import_part, load_part, suggest_setup  # noqa: E402
from loadpath.shapes import Box, Cylinder, trimesh_from_manifold  # noqa: E402
from loadpath.step import available  # noqa: E402
from loadpath.tasks import check_study  # noqa: E402

BOLTS = [(6.0, 7.0), (6.0, 23.0), (18.0, 7.0), (18.0, 23.0)]   # x, y of the M3 holes
BORE = (48.0, 15.0, 5.0)                                         # x, y, r of the bearing bore


def _link_mesh():
    body = Box([0, 0, 0], [60, 30, 8]).to_manifold()
    for x, y in BOLTS:
        body = body - Cylinder([x, y, -1], [x, y, 9], 1.7).to_manifold()
        body = body - Cylinder([x, y, 5], [x, y, 9], 3.0).to_manifold()    # counterbore 3 mm deep
    body = body - Cylinder([BORE[0], BORE[1], -1], [BORE[0], BORE[1], 9], BORE[2]).to_manifold()
    # trimesh_from_manifold tessellates cylinders coarsely; refine for the fit
    return trimesh_from_manifold(body)


def _link_step(path):
    from OCP.BRepAlgoAPI import BRepAlgoAPI_Cut
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox, BRepPrimAPI_MakeCylinder
    from OCP.gp import gp_Ax2, gp_Dir, gp_Pnt
    from OCP.IFSelect import IFSelect_RetDone
    from OCP.STEPControl import STEPControl_AsIs, STEPControl_Writer

    s = BRepPrimAPI_MakeBox(gp_Pnt(0, 0, 0), gp_Pnt(60, 30, 8)).Shape()

    def cut(s, x, y, z0, h, r):
        c = BRepPrimAPI_MakeCylinder(gp_Ax2(gp_Pnt(x, y, z0), gp_Dir(0, 0, 1)), r, h).Shape()
        return BRepAlgoAPI_Cut(s, c).Shape()

    for x, y in BOLTS:
        s = cut(s, x, y, -1, 10, 1.7)
        s = cut(s, x, y, 5, 4, 3.0)
    s = cut(s, BORE[0], BORE[1], -1, 10, BORE[2])
    w = STEPControl_Writer()
    w.Transfer(s, STEPControl_AsIs)
    assert w.Write(str(path)) == IFSelect_RetDone


def _check_link(feats, exact):
    bolts = [b for b in feats["bores"] if b["kind"] == "bolt"]
    bores = [b for b in feats["bores"] if b["kind"] == "bore"]
    assert len(bolts) == 4, [b["r"] for b in feats["bores"]]
    assert len(bores) == 1, bores
    tol = 1e-6 if exact else 0.05
    for b in bolts:
        assert abs(b["r"] - 1.7) < tol and b["through"], b
        rs = sorted(s["r"] for s in b["segments"])
        assert len(rs) == 2 and abs(rs[1] - 3.0) < tol, rs    # the counterbore is a step of the same bore
        assert abs(abs(b["axis"][2]) - 1) < 1e-6
    assert abs(bores[0]["r"] - BORE[2]) < tol and bores[0]["through"]
    assert abs(bores[0]["center"][0] - BORE[0]) < 0.05
    # the bolts open onto both big faces; the mount is a bolted face holding all four
    mounts = [i for i in feats["interfaces"] if i["kind"] == "bolted-face"]
    assert mounts and len(mounts[0]["bores"]) == 4, feats["interfaces"]
    s = suggest_setup(feats, 50.0)
    assert s["support"] == mounts[0]["id"]
    assert s["load"] == bores[0]["id"]
    f = np.asarray(s["force"])
    assert abs(np.linalg.norm(f) - 50.0) < 1e-2
    assert f[2] < -49.0              # square to the lever (along x), so straight down


def test_heal_closes_a_cracked_mesh():
    m = _link_mesh()
    assert m.is_watertight
    cracked = m.copy()
    cracked.update_faces(np.arange(len(m.faces)) != 7)        # drop a triangle: an open shell
    cracked.remove_unreferenced_vertices()
    assert not cracked.is_watertight
    healed, rep = heal_mesh(cracked)
    assert healed.is_watertight and rep["loops"] == 1
    assert abs(healed.volume - m.volume) / m.volume < 0.01


def test_mesh_features_and_draft_study():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "link.stl")
        _link_mesh().export(path)
        part = load_part(path)
        feats = detect_features(part)
        assert feats["source"] == "mesh"
        _check_link(feats, exact=False)
        out = import_part(path, os.path.join(d, "study"), force=50.0)
        study = out["study"]
        sdir = os.path.join(d, "study")
        r = check_study(study, sdir)
        assert r["ok"], r
        assert study["supports"][0]["region"].endswith("-hole")
        assert study["loadCases"][0]["loads"][0]["distribution"] == "bearing"
        # every preserved region is a closed mesh inside the part
        for p in study["preserve"]:
            m = trimesh.load(os.path.join(sdir, p["shape"]["path"]), force="mesh")
            assert m.is_watertight and m.volume < part.mesh.volume
        assert os.path.exists(os.path.join(sdir, "link.features.json"))


def test_grown_space_clears_mating_faces():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "link.stl")
        _link_mesh().export(path)
        out = import_part(path, os.path.join(d, "s"), space="box", grow=5.0)
        study = out["study"]
        clears = [k for k in study["keepOut"] if k["id"].endswith("-clear")]
        assert clears and all(k["bore"] is False for k in clears)
        r = check_study(study, os.path.join(d, "s"))
        assert r["ok"], r


def test_step_features_are_exact():
    if not available():
        print("  (skipped: no OpenCascade)")
        return
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "link.step")
        _link_step(path)
        part = load_part(path)
        assert part.mesh.is_watertight
        feats = detect_features(part)
        assert feats["source"] == "brep"
        _check_link(feats, exact=True)


def test_step_as_mesh_shape_and_cli():
    if not available():
        print("  (skipped: no OpenCascade)")
        return
    with tempfile.TemporaryDirectory() as d:
        _link_step(os.path.join(d, "link.step"))
        study = {"name": "s", "designSpace": [{"type": "mesh", "path": "link.step"}],
                 "keepOut": [{"id": "bore", "shape": {"type": "cylinder", "p0": [48, 15, -1], "p1": [48, 15, 9],
                                                      "r": 5}}],
                 "supports": [{"shape": {"type": "box", "min": [-1, -1, -1], "max": [1, 31, 9]}}],
                 "loadCases": [{"name": "a", "loads": [{"region": "bore", "force": [0, 0, -10]}]}],
                 "material": "petg", "goal": {"volumeFraction": 0.4}}
        r = check_study(study, d)
        assert r["ok"], r
        out = subprocess.run([sys.executable, "-m", "loadpath", "import", os.path.join(d, "link.step"),
                              "--out", os.path.join(d, "o")], capture_output=True, text=True, cwd=ROOT)
        assert out.returncode == 0, out.stderr
        rep = json.loads(out.stdout)
        assert rep["ok"] and rep["bores"] == 5
        drafted = json.load(open(rep["study"]))
        assert check_study(drafted, os.path.join(d, "o"))["ok"]


TESTS = [test_heal_closes_a_cracked_mesh, test_mesh_features_and_draft_study, test_grown_space_clears_mating_faces,
         test_step_features_are_exact, test_step_as_mesh_shape_and_cli]

if __name__ == "__main__":
    failed = 0
    for t in TESTS:
        t0 = time.time()
        try:
            t()
            print(f"PASS {t.__name__} ({time.time() - t0:.1f}s)")
        except Exception:  # noqa: BLE001
            failed += 1
            print(f"FAIL {t.__name__}")
            traceback.print_exc()
    print(f"\n{'all %d passed' % len(TESTS) if not failed else '%d failed' % failed}")
    sys.exit(1 if failed else 0)
