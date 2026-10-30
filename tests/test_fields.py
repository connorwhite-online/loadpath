"""
Tests for fields.py, the nTop-style field graph: exact walls, field-driven
shells, blends and lattices, smooth meshing, exact mesh distance, result
fields, the expression whitelist, the schema and the CLI. Run:
`python3 tests/test_fields.py` (about a minute).
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

from loadpath.fields import FieldGraph, MeshField, _Samples, _compile_expr  # noqa: E402
from loadpath.schema import lint  # noqa: E402

SPHERE = {"op": "shape", "shape": {"type": "sphere", "center": [0, 0, 0], "r": 10}}


def test_shell_volume_and_projection():
    """A 2 mm inner shell of a 10 mm sphere has the analytic volume, and
    every vertex lies on the field's zero level after projection (raw
    marching cubes is off it by a good fraction of the pitch)."""
    g = FieldGraph({"nodes": {"s": SPHERE, "shell": {"op": "shell", "input": "s", "thickness": 2}}})
    want = 4 / 3 * np.pi * (10 ** 3 - 8 ** 3)
    m = g.mesh(pitch=0.5)
    assert m.is_watertight
    assert abs(m.volume / want - 1) < 0.005, (m.volume, want)
    err = np.abs(g.evaluate(m.vertices)).max()
    assert err < 1e-4, err
    raw = g.mesh(pitch=0.5, project=0)
    assert np.abs(g.evaluate(raw.vertices)).max() > 20 * err
    # an outer sphere's radius from the vertices: exact once projected
    r = np.linalg.norm(m.vertices, axis=1)
    assert abs(r.max() - 10) < 1e-3 and abs(r[r > 9].min() - 10) < 1e-3


def test_variable_shell_follows_its_field():
    """Wall = 1 mm at x = -10 rising to 3 mm at x = +10, on a box: probe
    inside the wall and just past it at both ends."""
    doc = {"nodes": {
        "b": {"op": "shape", "shape": {"type": "box", "min": [-10, -10, -10], "max": [10, 10, 10]}},
        "t": {"op": "remap", "input": {"op": "coordinate", "axis": "x"}, "from": [-10, 10], "to": [1, 3]},
        "w": {"op": "shell", "input": "b", "thickness": "t"}}}
    g = FieldGraph(doc)
    # top face (z = 10): wall goes down to z = 10 - t(x)
    for x, t in ((-6, 1.4), (6, 2.6)):
        assert g.evaluate([[x, 0, 10 - t + 0.1]])[0] < 0
        assert g.evaluate([[x, 0, 10 - t - 0.1]])[0] > 0


def test_variable_blend_is_a_bigger_fillet_where_asked():
    """Two boxes meeting at a right angle, blend 0.5 mm at y = -10 and
    5 mm at y = +10: the concave corner fills in only at the big end."""
    doc = {"nodes": {
        "a": {"op": "shape", "shape": {"type": "box", "min": [0, -10, 0], "max": [20, 10, 2]}},
        "b": {"op": "shape", "shape": {"type": "box", "min": [0, -10, 0], "max": [2, 10, 20]}},
        "k": {"op": "expr", "expr": "lerp(0.5, 5, (y + 10) / 20)"},
        "u": {"op": "union", "inputs": ["a", "b"], "blend": "k"}}}
    g = FieldGraph(doc)
    corner = lambda y: [[2.6, y, 2.6]]  # noqa: E731
    assert g.evaluate(corner(9.5))[0] < 0       # filleted
    assert g.evaluate(corner(-9.5))[0] > 0      # still a sharp corner
    lo, hi = g.bounds()
    assert hi[2] >= 20 and lo[0] < 0             # the bulge allowance


def test_graded_lattice_stays_one_body():
    """A gyroid whose cell grows 5 -> 12 mm across a block, wall from a
    density field: one watertight body, and coarser cells mean fewer
    faces per mm^3 at the coarse end."""
    doc = {"nodes": {
        "box": {"op": "shape", "shape": {"type": "box", "min": [0, 0, 0], "max": [60, 20, 20]}},
        "cell": {"op": "remap", "input": {"op": "coordinate", "axis": "x"}, "from": [5, 55], "to": [5, 12]},
        "lat": {"op": "lattice", "type": "gyroid", "cell": "cell", "density": 0.3},
        "part": {"op": "intersect", "inputs": ["lat", "box"]}}, "output": "part", "mesh": {"pitch": 0.35}}
    g = FieldGraph(doc)
    m = g.mesh()
    assert m.is_watertight
    bodies = m.split(only_watertight=False)
    big = max(b.volume for b in bodies)
    assert sum(b.volume for b in bodies) - big < 0.01 * big, [b.volume for b in bodies]
    x = m.triangles_center[:, 0]
    assert (x < 15).sum() > 1.5 * (x > 45).sum()
    # density 0.3 -> volume fraction near 0.3 overall
    assert abs(m.volume / (60 * 20 * 20) - 0.3) < 0.06, m.volume / 24000


def test_cell_warp_period_follows_the_field():
    """With the cell a function of x alone, the warp is u = integral of
    dx / cell: the local period along x is the cell, and y and z keep
    theirs."""
    from loadpath.fields import _CellWarp

    cell = lambda P: 4.0 + 0.1 * P[:, 0]  # noqa: E731
    w = _CellWarp(cell, [0, 0, 0], [80, 20, 20], max_cells=200_000)
    x = np.linspace(5, 75, 15)
    P = np.stack([x, np.full_like(x, 10), np.full_like(x, 10)], 1)
    dx = 0.5
    du = (w(P + [dx, 0, 0], cell(P + [dx, 0, 0]))[:, 0] - w(P, cell(P))[:, 0]) / dx
    assert np.allclose(du * cell(P), 1.0, rtol=0.02), du * cell(P)
    dv = (w(P + [0, dx, 0], cell(P))[:, 1] - w(P, cell(P))[:, 1]) / dx
    # u_y cannot be exact (1/cell has curl along y): least squares keeps it close
    assert np.allclose(dv * cell(P), 1.0, rtol=0.08), dv * cell(P)


def test_mesh_field_is_exact_near_the_surface():
    """A coarse icosphere as a mesh field: distances near the surface are
    the mesh's own (to the triangles, not a voxel staircase), and the sign
    is right inside and out."""
    s = trimesh.creation.icosphere(subdivisions=3, radius=10)
    mf = MeshField(s, pitch=0.5)
    rng = np.random.default_rng(0)
    d = rng.normal(size=(2000, 3))
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    P = d * rng.uniform(8.8, 11.2, size=(2000, 1))
    got = mf(P)
    sp = s.nearest.signed_distance(P)            # trimesh: positive inside
    assert np.abs(got + sp).max() < 1e-6, np.abs(got + sp).max()
    far = mf(np.array([[0, 0, 0], [30, 0, 0]]))
    assert far[0] < -9 and far[1] > 15


def test_samples_on_a_grid_interpolate_exactly():
    """FE-style nodes (a subset of a regular grid) are put back on the grid:
    a linear field comes back exactly, and holes take the nearest value."""
    xs = np.arange(0, 10.01, 0.5)
    G = np.stack(np.meshgrid(xs, xs, xs, indexing="ij"), -1).reshape(-1, 3)
    keep = np.linalg.norm(G - 5, axis=1) < 5.5
    pts = G[keep]
    f = 2 * pts[:, 0] - pts[:, 1] + 0.5 * pts[:, 2]
    s = _Samples(pts, f)
    assert s.tree is None
    q = np.random.default_rng(1).uniform(3, 7, (500, 3))
    assert np.allclose(s(q), 2 * q[:, 0] - q[:, 1] + 0.5 * q[:, 2], atol=1e-4)
    # scattered points fall back to the tree
    sc = _Samples(np.random.default_rng(2).uniform(0, 10, (300, 3)), np.ones(300))
    assert sc.tree is not None and np.allclose(sc(q), 1.0)


def test_result_node_from_verify_fields():
    """A `result` node reads verify_part's fields from memory: the max over
    load cases, or one case."""
    nodes = np.array([[x, y, z] for x in range(3) for y in range(3) for z in range(3)], float)
    fields = {"nodes": nodes, "cases": [
        {"name": "a", "vonMises": nodes[:, 0], "displacement": np.zeros((27, 3))},
        {"name": "b", "vonMises": 2 - nodes[:, 0], "displacement": np.zeros((27, 3))}]}
    g = FieldGraph({"nodes": {"s": {"op": "result", "source": "r"},
                              "a": {"op": "result", "source": "r", "case": "a"}}}, sources={"r": fields})
    assert np.isclose(g.evaluate([[0, 1, 1]], "s")[0], 2.0)
    assert np.isclose(g.evaluate([[1, 1, 1]], "s")[0], 1.0)
    assert np.isclose(g.evaluate([[0.5, 1, 1]], "a")[0], 0.5)


def test_expressions_are_whitelisted():
    names, fn = _compile_expr("clamp(a * 2 + x, 0, 5) if b > 1 else -pi")
    assert names == {"a", "b", "x"}
    v = fn({"a": np.array([1.0, 1.0]), "b": np.array([2.0, 0.0]), "x": np.array([0.5, 0.5])})
    assert np.allclose(v, [2.5, -np.pi])
    for bad in ("__import__('os').system('true')", "a.real", "a[0]", "open('x')", "lambda: 1"):
        try:
            _compile_expr(bad)
        except ValueError:
            continue
        raise AssertionError(f"{bad!r} compiled")


def test_graph_errors_and_lint():
    def err(doc):
        try:
            FieldGraph(doc)
        except ValueError as exc:
            return str(exc)
        raise AssertionError("no error")

    assert "not a node" in err({"nodes": {"a": {"op": "shell", "input": "b", "thickness": 1}}})
    assert "cycle" in err({"nodes": {"a": {"op": "offset", "input": "b", "distance": 1},
                                     "b": {"op": "offset", "input": "a", "distance": 1}}})
    assert "did you mean 'shell'" in err({"nodes": {"a": {"op": "shel"}}})
    assert "reads 'q'" in err({"nodes": {"a": {"op": "expr", "expr": "q + 1"}}})
    issues = lint({"nodes": {"s": SPHERE, "w": {"op": "shell", "input": "s", "thicknes": 1}}}, "field")
    assert any("did you mean 'thickness'" in i["message"] for i in issues), issues
    issues = lint({"nodes": {"w": {"op": "shell", "input": "nope", "thickness": 1}}}, "field")
    assert any("not a node" in i["message"] for i in issues), issues
    assert lint(json.loads((ROOT / "examples" / "fields" / "arm-link-graded.json").read_text()), "field") == []


def test_unbounded_lattice_asks_for_bounds():
    g = FieldGraph({"nodes": {"l": {"op": "lattice", "cell": 6}}})
    try:
        g.mesh(pitch=1.0)
    except ValueError as exc:
        assert "unbounded" in str(exc)
    else:
        raise AssertionError("meshed an unbounded lattice")
    m = g.mesh(pitch=0.5, bounds=([0, 0, 0], [12, 12, 12]))
    assert m.is_watertight


def test_cli_writes_part_and_report():
    doc = {"name": "puck", "nodes": {
        "c": {"op": "shape", "shape": {"type": "cylinder", "center": [0, 0, 0], "axis": "z", "length": 6, "r": 12}},
        "core": {"op": "lattice", "type": "diamond", "cell": 6, "thickness": 0.8},
        "fill": {"op": "intersect", "inputs": ["core", "c"]},
        "skin": {"op": "shell", "input": "c", "thickness": 1.0},
        "part": {"op": "union", "inputs": ["skin", "fill"], "blend": 0.6}},
        "material": "petg", "mesh": {"pitch": 0.4}, "check": False}
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "puck.json")
        with open(p, "w") as fh:
            json.dump(doc, fh)
        r = subprocess.run([sys.executable, "-m", "loadpath", "field", p, "--out", d, "--no-render"],
                           capture_output=True, text=True, cwd=ROOT)
        assert r.returncode == 0, r.stdout + r.stderr
        rep = json.loads(r.stdout)
        assert rep["ok"] and rep["watertight"] and rep["bodies"] == 1, rep
        assert rep["enclosedVoids"] == 2, rep          # a sealed sheet lattice: both labyrinths
        assert 0 < rep["mass"] < 12 * 12 * np.pi * 6 * 1.27 / 1000
        assert os.path.exists(os.path.join(d, "puck.stl"))
        bad = dict(doc, nodes={**doc["nodes"], "part": {"op": "union", "inputs": ["skin", "flll"]}})
        with open(p, "w") as fh:
            json.dump(bad, fh)
        r = subprocess.run([sys.executable, "-m", "loadpath", "field", p, "--out", d],
                           capture_output=True, text=True, cwd=ROOT)
        assert r.returncode == 2 and "flll" in r.stdout


def test_mcp_field_job():
    """loadpath_field: a bad graph comes back with every issue; a good one
    runs as a job whose status carries the report and the sealed-void note."""
    from loadpath import mcp_server

    with tempfile.TemporaryDirectory() as d:
        os.environ["LOADPATH_HOME"] = d
        try:
            bad = mcp_server.field({"nodes": {"a": {"op": "shell", "input": "zz", "thickness": 1}}})
            assert not bad["ok"] and "zz" in bad["error"], bad
            doc = {"name": "box", "check": False, "mesh": {"pitch": 0.5}, "nodes": {
                "b": {"op": "shape", "shape": {"type": "box", "min": [0, 0, 0], "max": [12, 12, 12]}},
                "core": {"op": "intersect", "inputs": [{"op": "lattice", "cell": 6, "thickness": 0.8}, "b"]},
                "part": {"op": "union", "inputs": [{"op": "shell", "input": "b", "thickness": 1}, "core"]}}}
            job = mcp_server.field(doc)["jobId"]
            for _ in range(240):
                info, _ = mcp_server.status(job)
                if info["status"] != "running":
                    break
                time.sleep(0.5)
            assert info["status"] == "done" and info["kind"] == "field", info
            assert info["watertight"] and info["enclosedVoids"] >= 1 and info["attention"], info
        finally:
            os.environ.pop("LOADPATH_HOME", None)


TESTS = [test_shell_volume_and_projection, test_variable_shell_follows_its_field,
         test_variable_blend_is_a_bigger_fillet_where_asked, test_graded_lattice_stays_one_body, test_cell_warp_period_follows_the_field,
         test_mesh_field_is_exact_near_the_surface, test_samples_on_a_grid_interpolate_exactly,
         test_result_node_from_verify_fields, test_expressions_are_whitelisted, test_graph_errors_and_lint,
         test_unbounded_lattice_asks_for_bounds, test_cli_writes_part_and_report, test_mcp_field_job]


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
