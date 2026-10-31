"""
Tests for sharp.py (exact booleans in field graphs, the edge-accuracy
measure) and step.py (STEP export). Run: `python3 tests/test_sharp.py`
(about a minute; the STEP tests skip without OpenCascade, loadpath[step]).
"""
import os
import sys
import tempfile
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from loadpath.fields import FieldGraph, run_field  # noqa: E402
from loadpath.sharp import Exact, Op, Organic, edge_accuracy, mesh_distance, plan, plan_refs, study_refs  # noqa: E402
from loadpath.step import available, export_step, plan_of, read_step, shape_to_trimesh  # noqa: E402

# A plate with a collar round a through bore. The collar is a shell of a
# boss coaxial with the bore, so before the exact cut its inner wall sits
# on the bore's own surface: the case the extended field is for.
PLATE = {"nodes": {
    "plate": {"op": "shape", "shape": {"type": "box", "min": [-20, -15, -5], "max": [20, 15, 5]}},
    "soft": {"op": "offset", "input": "plate", "distance": 0.0},
    "bore": {"op": "shape", "shape": {"type": "cylinder", "center": [0, 0, 0], "axis": "z", "length": 30, "r": 6}},
    "boss": {"op": "shape", "shape": {"type": "cylinder", "center": [0, 0, 0], "axis": "z", "length": 14, "r": 6}},
    "collar": {"op": "shell", "input": "boss", "side": "outside", "thickness": 3},
    "body": {"op": "union", "inputs": ["soft", "collar"], "blend": 2},
    "part": {"op": "subtract", "inputs": ["body", "bore"]}}, "output": "part", "name": "plate"}


def _acc(g, m):
    F = g.field()
    return edge_accuracy(m, plan_refs(plan(g)), boundary=lambda P: np.abs(F(P)) < 1e-6)


def test_plan_splits_at_hard_booleans_with_shapes():
    g = FieldGraph(PLATE)
    p = plan(g)
    assert isinstance(p, Op) and p.op == "subtract"
    assert isinstance(p.kids[0], Organic) and p.kids[0].names == ["body"]
    assert isinstance(p.kids[1], Exact) and p.kids[1].name == "bore"
    assert plan_refs(p)[0][1] == "cut"
    # a blended boolean, and one between fields only, stay in the field
    doc = {"nodes": {**PLATE["nodes"], "part": {"op": "subtract", "inputs": ["body", "bore"], "blend": 1}}}
    assert isinstance(plan(FieldGraph(doc)), Organic)
    doc = {"nodes": {**PLATE["nodes"], "part": {"op": "intersect", "inputs": ["body", "soft"]}}}
    assert isinstance(plan(FieldGraph(doc)), Organic)


def test_sharp_bore_is_exact_where_the_field_rounds_it():
    """Meshed from the field the bore's rims are rounded (a few tenths of a
    millimetre off at a 0.5 mm pitch); with the exact cut every sample of
    the bore wall is within the chord tolerance, and the collar's coincident
    inner wall leaves nothing behind."""
    g = FieldGraph(PLATE)
    before = g.mesh(pitch=0.5, sharp=False)
    a0 = _acc(g, before)
    sharp = g.mesh(pitch=0.5)
    a1 = _acc(g, sharp)
    assert a0["edges"]["max"] > 0.1, a0
    assert a1["max"] < 0.005 and a1["edges"]["exactFraction"] == 1.0, a1
    assert sharp.is_watertight and len(sharp.split(only_watertight=False)) == 1
    assert abs(sharp.volume / before.volume - 1) < 0.003, (sharp.volume, before.volume)
    # nothing inside the bore: a 5.99 mm pin passes
    r = np.linalg.norm(sharp.vertices[:, :2], axis=1)
    assert r.min() > 6 - 0.003, r.min()
    assert g.plan is not None and plan_of(sharp) is g.plan


def test_mesh_distance_matches_trimesh():
    import trimesh

    m = trimesh.creation.cylinder(radius=5, height=40, sections=64)
    P = np.random.default_rng(0).uniform([-8, -8, -23], [8, 8, 23], (2000, 3))
    want = trimesh.proximity.closest_point(m, P)[1]
    assert np.abs(mesh_distance(m, P) - want).max() < 1e-9


def test_step_keeps_the_bore_a_cylinder():
    if not available():
        print("  (OpenCascade not installed: skipped)")
        return
    g = FieldGraph(PLATE)
    m = g.mesh(pitch=0.5)
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "plate.step")
        rep = export_step(m, path, faces=4000, name="plate")
        assert rep["valid"] and rep["solids"] == 1, rep
        assert rep["faces"].get("cylinder", 0) >= 1, rep["faces"]
        assert abs(rep["volume"] / m.volume - 1) < 0.003, (rep["volume"], m.volume)
        back = shape_to_trimesh(read_step(path))
        assert back.is_watertight
        a = _acc(g, back)
        assert a["max"] < 0.003, a


def test_study_part_exports_exact_interfaces():
    """A study's delivered part carries its plan: the STEP file has the
    keep-out bore as a cylinder and the design space's faces as planes."""
    from loadpath.domain import build_domain
    from loadpath.reconstruct import reconstruct
    from loadpath.spec import study_from_dict
    sys.path.insert(0, str(ROOT / "tests"))
    from test_loadpath import LUG

    s = study_from_dict(LUG)
    d = build_domain(s)
    x = np.zeros(d.n_elem)
    c = d.mesh.centers if hasattr(d.mesh, "centers") else None
    if c is None:
        x[:] = 1.0
    else:
        x[np.abs(c[:, 1]) < 6] = 1.0           # a bar from the support to the eye
    x = np.maximum(x, d.preserve.astype(float))
    mesh, info = reconstruct(s, d, x)
    assert plan_of(mesh) is not None
    a = edge_accuracy(mesh, study_refs(s))
    assert a["byRole"]["cut"]["max"] < 0.003, a
    if not available():
        return
    with tempfile.TemporaryDirectory() as td:
        rep = export_step(mesh, os.path.join(td, "lug.step"), faces=4000)
        assert rep["valid"] and rep["solids"] == 1, rep
        assert rep["faces"].get("cylinder", 0) >= 1, rep["faces"]       # the pin bore
        assert abs(rep["volume"] / mesh.volume - 1) < 0.01, (rep["volume"], mesh.volume)


def test_run_field_reports_edges_and_writes_step():
    with tempfile.TemporaryDirectory() as td:
        doc = {**PLATE, "mesh": {"pitch": 0.6}, "check": False, "step": available() and {"faces": 3000}}
        _, rep = run_field(doc, td, render=False)
        assert rep["sharpEdges"]["edges"]["exactFraction"] == 1.0, rep["sharpEdges"]
        if available():
            assert rep["step"]["valid"] and os.path.exists(rep["step"]["path"]), rep["step"]


def test_collapse_keeps_holes():
    import trimesh

    from loadpath.step import _collapse

    t = trimesh.creation.torus(20, 3, major_sections=96, minor_sections=48)
    V, F, _ = _collapse(t.vertices, t.faces, 200, turn=0.5)
    c = trimesh.Trimesh(V, F, process=False)
    assert len(c.faces) <= 210 and c.is_watertight and c.euler_number == 0, (len(c.faces), c.euler_number)


def test_step_smooth_patches():
    if not available():
        print("  (OpenCascade not installed: skipped)")
        return
    import trimesh

    b = trimesh.creation.icosphere(4, radius=15)
    v = b.vertices
    b.vertices = v * (1 + 0.15 * np.sin(v[:, 0] / 3) * np.cos(v[:, 1] / 4))[:, None]
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "blob.step")
        rep = export_step(b, path, surface="smooth", patches=400)
        assert rep["valid"] and rep["solids"] == 1, rep
        assert rep["faces"].get("bspline", 0) >= 380 and not rep["faces"].get("facet"), rep["faces"]
        assert abs(rep["volume"] / b.volume - 1) < 0.002, (rep["volume"], b.volume)
        dev = rep["decimated"][0]["deviation"]
        assert dev["p99"] < 0.05, dev
        back = shape_to_trimesh(read_step(path), deflection=0.005, angle=0.05)
        assert np.percentile(mesh_distance(b, back.sample(5000)), 99) < 0.05


TESTS = [test_plan_splits_at_hard_booleans_with_shapes, test_sharp_bore_is_exact_where_the_field_rounds_it,
         test_mesh_distance_matches_trimesh, test_step_keeps_the_bore_a_cylinder,
         test_study_part_exports_exact_interfaces, test_run_field_reports_edges_and_writes_step,
         test_collapse_keeps_holes, test_step_smooth_patches]


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
