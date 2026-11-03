"""
Tests for fieldjit.py, field graphs compiled to one numba kernel: every node
of the editor's all-ops graph gives numpy's values (bit for bit, but for
the few that numpy hands a 3-vector dot product to BLAS), the nodes that do
not compile run on numpy inside a compiled graph, a tune's parameter change
reuses the kernel, and a graded lattice meshes to the very same triangles.
Run: `python3 -m pytest tests/test_fieldjit.py` (about 1 min, mostly compiles).
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from loadpath import fieldjit  # noqa: E402
from loadpath.fields import FieldGraph  # noqa: E402
from loadpath.lattice import single_precision  # noqa: E402
from tests.test_editor import ALL_OPS, _workdir  # noqa: E402

pytestmark = pytest.mark.skipif(not fieldjit.available(), reason="needs numba")

# numpy takes a dot product with a 3-vector through BLAS (matmul / einsum) for these,
# its summation order the library's, and its own SIMD pow/exp/log: they agree to
# rounding, not bit for bit
DOTS = {"dir", "dseg", "dpl", "ex"}
ROUNDED = {"slant", "turned", "moved", "sq", "fx", "hollow", "all", "soft"}

GRADED = {"nodes": {
    "box": {"op": "shape", "shape": {"type": "box", "min": [0, 0, 0], "max": [60, 20, 20]}},
    "cell": {"op": "remap", "input": {"op": "coordinate", "axis": "x"}, "from": [5, 55], "to": [5, 12]},
    "wall": {"op": "expr", "expr": "0.6 + 0.3 * sin(y / 4)"},
    "lat": {"op": "lattice", "type": "gyroid", "cell": "cell", "thickness": "wall"},
    "core": {"op": "intersect", "inputs": ["lat", "box"]},
    "skin": {"op": "shell", "input": "box", "thickness": 1.2},
    "part": {"op": "union", "inputs": ["core", "skin"], "blend": 0.8}}, "output": "part"}

EXTRA = {"nodes": {
    "slant": {"op": "shape", "shape": {"type": "cylinder", "p0": [-10, -3, 2], "p1": [12, 5, -4], "r": 3, "rInner": 1}},
    "boxes": {"op": "shape", "shape": {"type": "subtract", "shapes": [
        {"type": "box", "min": [-10, -10, -10], "max": [10, 10, 10]},
        {"type": "intersect", "shapes": [{"type": "sphere", "center": [0, 0, 0], "r": 8},
                                          {"type": "box", "min": [-20, -2, -20], "max": [20, 2, 20]}]}]}},
    "turned": {"op": "lattice", "type": "diamond", "cell": 5, "thickness": 0.7, "orient": [0, 30, 15],
               "origin": [1, 2, 3]},
    "moved": {"op": "transform", "input": {"op": "union", "inputs": ["boxes", "turned"], "blend": "soft"},
              "translate": [2, 0, 1], "rotate": {"axis": [1, 2, 3], "degrees": 25, "center": [1, 1, 1]}, "scale": 1.3},
    "soft": {"op": "remap", "input": {"op": "coordinate", "axis": "y"}, "from": [-10, 10], "to": [0, 2],
             "curve": "smootherstep"},
    "sq": {"op": "remap", "input": "slant", "from": [-2, 6], "to": [1, 0], "curve": "square", "clamp": False},
    "fx": {"op": "expr", "expr": "log(abs(a) + 1) * exp(-abs(z) / 20) + tan(x / 50) - floor(y / 3) + ceil(z / 4)"
                                 " + sign(a) + clamp(x, -3, 3) + lerp(1, 2, step(0, y)) + min(a, 2) / (x - 3)",
           "vars": {"a": "slant"}},
    "hollow": {"op": "shell", "input": "boxes", "thickness": "sq", "side": "outside"},
    "net": {"op": "lattice", "type": "gyroid", "cell": 6, "thickness": 1.0, "form": "network"},
    "all": {"op": "subtract", "inputs": ["moved", "slant", "hollow"], "blend": "soft"}}, "output": "all"}


def _both(doc, monkeypatch, run):
    """run(graph) on the numpy graph, then on the compiled one."""
    out = []
    for flag in ("0", "1"):
        monkeypatch.setenv("LOADPATH_FIELDJIT", flag)
        g = FieldGraph(doc)
        out.append((g, run(g)))
    return out


def _numpy(g, P, name):
    return np.broadcast_to(np.asarray(g._eval(name, P, {}), float), (len(P),))


@pytest.mark.parametrize("doc", ["all-ops", "extra"])
def test_every_node_matches_numpy(doc, monkeypatch):
    kw = {"base_dir": _workdir()} if doc == "all-ops" else {}
    g = FieldGraph(ALL_OPS if doc == "all-ops" else EXTRA, **kw)
    P = np.random.default_rng(0).uniform(-40, 40, (40_000, 3))
    P[0] = [3.0, 0.0, 0.0]          # fx divides by zero there: inf, as numpy
    compiled = 0
    for name in g.nodes:
        cf = fieldjit.compile_graph(g, name)
        if cf is None:
            continue
        compiled += 1
        a, b = _numpy(g, P, name), cf(P)
        both = np.isnan(a) & np.isnan(b)
        if name in DOTS or (doc == "extra" and name.split(".")[0] in ROUNDED):
            ok = np.isfinite(a)
            assert np.array_equal(ok, np.isfinite(b)), name
            assert np.abs(a[ok] - b[ok]).max() <= 1e-12 * max(1.0, np.abs(a[ok]).max()), name
        else:
            assert np.array_equal(a[~both], b[~both]), (name, np.abs(a - b).max())
        with single_precision():
            a, b = _numpy(g, P, name), cf(P)
        ok = np.isfinite(a)
        assert np.abs(a[ok] - b[ok]).max() < 2e-5, name
    assert compiled >= (len(g.nodes) - 8 if doc == "all-ops" else len(g.nodes))
    assert not getattr(g, "_jit_errors", None), g._jit_errors


def test_opaque_nodes_run_on_numpy(monkeypatch):
    """A mesh distance, distance to points, a cylindrical or density lattice
    stay numpy, read by the kernel; a transform over a mesh stays numpy whole."""
    g = FieldGraph(ALL_OPS, base_dir=_workdir())
    cf = fieldjit.compile_graph(g, "all")
    assert {"tor2", "coreD"} <= set(cf.opaque)
    assert "torus" not in cf.opaque              # under tor2, numpy's
    cf = fieldjit.compile_graph(g, "cellF")
    assert cf.opaque == ["dp"]
    assert fieldjit.compile_graph(g, "torus") is None
    assert fieldjit.compile_graph(g, "tor2") is None


def test_evaluate_uses_the_kernel_and_matches(monkeypatch):
    P = np.random.default_rng(1).uniform(-5, 65, (300_000, 3))
    (a, va), (b, vb) = _both(GRADED, monkeypatch, lambda g: g.evaluate(P))
    assert b._jit["part"] is not None and a.__dict__.get("_jit") is None
    assert np.array_equal(va, vb)


def test_parameters_reuse_the_kernel():
    """Numbers are kernel arguments, not code: a tune's next candidate
    compiles nothing."""
    import copy

    d2 = copy.deepcopy(GRADED)
    d2["nodes"]["skin"]["thickness"] = 2.0
    d2["nodes"]["cell"]["to"] = [6, 9]
    d2["nodes"]["box"]["shape"]["max"] = [50, 25, 20]
    k1 = fieldjit.compile_graph(FieldGraph(GRADED), "part")
    k2 = fieldjit.compile_graph(FieldGraph(d2), "part")
    assert k1.kernel is k2.kernel and not np.array_equal(k1.C, k2.C)


def test_mesh_is_the_same(monkeypatch):
    (_, a), (_, b) = _both(GRADED, monkeypatch, lambda g: g.mesh(pitch=0.6))
    assert b.is_watertight
    # the same triangles; vertices apart by the single-precision lattice trig's
    # noise (numpy's SIMD float32 sin vs libm's), carried along by the relax
    assert np.array_equal(a.faces, b.faces)
    assert np.abs(a.vertices - b.vertices).max() < 2e-4
    assert abs(a.volume - b.volume) < 1e-7 * a.volume


def test_knn_is_the_trees():
    """knn.KDTree gives cKDTree's k nearest, in its order, on ordered and
    scattered points alike."""
    from scipy.spatial import cKDTree

    from loadpath.knn import KDTree

    rng = np.random.default_rng(3)
    pts = rng.normal(size=(20_000, 3)) * [20, 10, 5]
    tree = cKDTree(pts)
    P = np.concatenate([rng.uniform(-30, 30, (20_000, 3)),                       # scattered
                        np.sort(rng.uniform(0, 4, (30_000, 3)), axis=0) + [5, -3, 1],  # in order
                        rng.uniform(-60, 60, (500, 3))])                         # far away
    want = tree.query(P, k=12)[1]
    got = KDTree(pts).query(P, 12)
    assert np.array_equal(want, got)


def test_mesh_distance_unchanged(monkeypatch):
    """MeshField's exact band reads the same values through knn.KDTree."""
    import trimesh

    from loadpath.fields import MeshField

    m = trimesh.creation.torus(major_radius=14, minor_radius=5, major_sections=48, minor_sections=24)
    P = np.random.default_rng(4).uniform(-22, 22, (60_000, 3)) * [1, 1, 0.4]
    vals = []
    for mode in ("kdtree", ""):
        monkeypatch.setenv("LOADPATH_KNN", mode)
        vals.append(MeshField(m, band=6.0)(P))
    assert np.array_equal(vals[0], vals[1])


def test_compiled_shapes_match():
    """A sharp plan's don't-care region (sharp._Extended.g): primitives, a
    union and a mesh shape's voxel distance, compiled, bit for bit."""
    import trimesh

    from loadpath.shapes import Box, Cylinder, MeshShape, Union

    m = trimesh.creation.torus(major_radius=14, minor_radius=5, major_sections=48, minor_sections=24)
    regions = [(Union([Cylinder([0, 0, -20], [0, 0, 20], 4), Cylinder([-30, 2, 0], [30, 2, 0], 2, 1),
                       Box([-5, -5, 8], [5, 5, 12])]), 1), (MeshShape(m), -1)]
    P = np.random.default_rng(5).uniform(-40, 40, (100_000, 3))
    want = np.min([s * sh.sdf(P) for sh, s in regions], axis=0)
    assert np.array_equal(fieldjit.compile_shapes(regions)(P), want)
