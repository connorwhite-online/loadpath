"""
Tests for narrowband.py (field meshing that samples only near the
surface) and the exact-TPMS lattice surface it enables. Run:
`python3 tests/test_narrowband.py` (about a minute).
"""
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import trimesh  # noqa: E402

from loadpath.fields import FieldGraph, field_mesh  # noqa: E402
from loadpath.narrowband import narrowband_mc  # noqa: E402

LINK = {"nodes": {
    "a": {"op": "shape", "shape": {"type": "cylinder", "center": [0, 0, 0], "axis": "z", "length": 20, "r": 12}},
    "b": {"op": "shape", "shape": {"type": "cylinder", "center": [60, 0, 0], "axis": "z", "length": 20, "r": 10}},
    "web": {"op": "shape", "shape": {"type": "box", "min": [0, -7, -6], "max": [60, 7, 6]}},
    "blob": {"op": "union", "inputs": ["a", "b", "web"], "blend": 8},
    "bores": {"op": "shape", "shape": {"type": "union", "shapes": [
        {"type": "cylinder", "center": [0, 0, 0], "axis": "z", "length": 30, "r": 7},
        {"type": "cylinder", "center": [60, 0, 0], "axis": "z", "length": 30, "r": 6}]}},
    "part": {"op": "subtract", "inputs": ["blob", "bores"]}}, "output": "part"}


def _dense(f, lo, dims, pitch):
    from skimage.measure import marching_cubes

    ax = [lo[k] + np.arange(dims[k]) * pitch for k in range(3)]
    G = np.stack(np.meshgrid(*ax, indexing="ij"), -1).reshape(-1, 3)
    F = np.asarray(f(G), np.float32).reshape(tuple(dims))
    F[0], F[-1], F[:, 0], F[:, -1], F[:, :, 0], F[:, :, -1] = (pitch,) * 6
    F[F == 0.0] = -1e-4 * pitch
    v, fa, _, _ = marching_cubes(F, 0.0, spacing=(pitch,) * 3)
    return v + lo, fa


def test_same_mesh_as_dense_with_fewer_samples():
    """On a solid part the narrow band marches the same triangles as the
    dense grid, vertex for vertex, from about half the samples (less the finer the pitch)."""
    from scipy.spatial import cKDTree

    g = FieldGraph(LINK)
    f = g.field()
    lo, hi = (np.asarray(b, float) for b in g.bounds())
    pitch = 0.3
    lo = lo - 2 * pitch - pitch * 0.013717
    dims = np.ceil((hi + 2 * pitch - lo) / pitch).astype(int) + 1
    V, F, st = narrowband_mc(f, lo, dims, pitch)
    v, fa = _dense(f, lo, dims, pitch)
    assert len(F) == len(fa) and len(V) == len(v), (len(F), len(fa))
    assert cKDTree(v).query(V)[0].max() < 1e-5
    assert st["samples"] < 0.6 * st["denseSamples"], st
    m = trimesh.Trimesh(V, F[:, ::-1])
    assert m.is_watertight
    assert abs(m.volume / trimesh.Trimesh(v, fa[:, ::-1]).volume - 1) < 1e-6


def test_finer_than_the_dense_budget():
    """The budget counts sampled cells: a pitch whose box is over it still
    meshes when the surface fits."""
    from loadpath import fields

    g = FieldGraph(LINK)
    budget = fields._CELL_BUDGET
    try:
        fields._CELL_BUDGET = 4_000_000
        m = field_mesh(g.field(), *g.bounds(), pitch=0.2, project=1)
        assert m.metadata["sampling"]["denseSamples"] > 4_000_000
        assert m.is_watertight
        try:
            field_mesh(g.field(), *g.bounds(), pitch=0.2, project=1, sparse=False)
            raise AssertionError("the dense grid should be over budget")
        except ValueError:
            pass
    finally:
        fields._CELL_BUDGET = budget


def test_steep_lattice_field_matches_dense():
    """A warp-graded gyroid's field is far steeper than a distance: the
    Lipschitz estimate sees it and nothing is skipped that the dense grid
    would mesh."""
    doc = {"nodes": {
        "box": {"op": "shape", "shape": {"type": "box", "min": [0, 0, 0], "max": [40, 16, 16]}},
        "cell": {"op": "remap", "input": {"op": "coordinate", "axis": "x"}, "from": [5, 35], "to": [5, 10]},
        "lat": {"op": "lattice", "type": "gyroid", "cell": "cell", "density": 0.3},
        "part": {"op": "intersect", "inputs": ["lat", "box"]}}, "output": "part"}
    g = FieldGraph(doc)
    a = field_mesh(g.field(), *g.bounds(), pitch=0.4, project=0)
    b = field_mesh(g.field(), *g.bounds(), pitch=0.4, project=0, sparse=False)
    assert a.is_watertight and a.metadata["sampling"]["lipschitz"] > 5
    assert abs(len(a.faces) - len(b.faces)) <= 4 and abs(a.volume / b.volume - 1) < 1e-5


def test_lattice_walls_are_exact():
    """lattice_infill meshes the gyroid sheet itself: the open lattice's
    vertices sit on the wall_min sheet (the grid's marching cubes left
    them ~0.05 mm proud, walls ~12% thick), and the part is lighter."""
    from loadpath.lattice import LatticeOptions, lattice_infill, tpms
    from loadpath.spec import study_from_dict

    st = study_from_dict({"material": {"E": 2000, "nu": 0.3, "strength": 50, "density": 1.0},
                          "goal": {"volumeFraction": 0.5}, "resolution": {"pitch": 2.0},
                          "designSpace": {"type": "box", "min": [0, 0, 0], "max": [32, 16, 16]},
                          "supports": [{"shape": {"type": "box", "min": [-1, -1, -1], "max": [0.01, 17, 17]}}],
                          "loadCases": [{"name": "tip", "loads": [
                              {"shape": {"type": "box", "min": [31.99, -1, -1], "max": [33, 17, 17]},
                               "force": [0, 0, -100]}]}]})
    block = trimesh.creation.box(extents=(32, 16, 16))
    block.apply_translation((16, 8, 8))
    out = {}
    for exact in (False, True):
        o = LatticeOptions(cell=8.0, step=0.4, skin=0.0, collar=0.0, exact_surface=exact)
        m, info = lattice_infill(st, block, None, o)
        V = m.vertices[np.all((m.vertices > 3) & (m.vertices < [29, 13, 13]), axis=1)]
        h = 1e-4
        gr = np.stack([(tpms("gyroid", V + e * h, 8.0) - tpms("gyroid", V - e * h, 8.0)) / (2 * h)
                       for e in np.eye(3)], 1)
        err = (np.abs(tpms("gyroid", V, 8.0)) - 0.5 * o.wall_min) / np.linalg.norm(gr, axis=1)
        out[exact] = (float(np.abs(err).max()), info["volume"], m.is_watertight)
    assert out[True][0] < 1e-3 and out[False][0] > 0.02, out
    assert out[True][1] < 0.95 * out[False][1] and out[True][2], out


TESTS = [test_same_mesh_as_dense_with_fewer_samples, test_finer_than_the_dense_budget,
         test_steep_lattice_field_matches_dense, test_lattice_walls_are_exact]


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
