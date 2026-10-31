"""
Tests for crease.py (sharp creases between field surfaces, mesh.creases).
Run: `python3 tests/test_crease.py` (a few seconds).
"""
import math
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from loadpath.crease import CreaseTree, _edges, meet  # noqa: E402
from loadpath.fields import FieldGraph  # noqa: E402
from loadpath.sharp import mesh_distance  # noqa: E402


def _ball(cx, r):
    return {"op": "expr", "expr": f"sqrt((x-({cx}))**2 + y**2 + z**2) - {r}"}


def _lens(op="intersect", phi=90.0, R=10.0, **mesh):
    """Two expr spheres (organic: no plain shape for sharp.py to cut exactly)
    meeting with `phi` degrees between their normals."""
    d = math.sqrt(2 * R * R * (1 - math.cos(math.radians(phi))))
    return {"nodes": {"a": _ball(-d / 2, R), "b": _ball(d / 2, R), "part": {"op": op, "inputs": ["a", "b"]}},
            "output": "part", "bounds": {"min": [-d / 2 - R - 1, -R - 1, -R - 1], "max": [d / 2 + R + 1, R + 1, R + 1]},
            "mesh": mesh}, d


def _circle(d, R, n=720):
    """The spheres' intersection: a circle in x = 0."""
    rc = math.sqrt(R * R - d * d / 4)
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.stack([np.zeros(n), rc * np.cos(t), rc * np.sin(t)], 1)


def _closed(m):
    F = np.asarray(m.faces)
    e = np.sort(np.stack([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]], 1).reshape(-1, 2), 1)
    _, c = np.unique(e, axis=0, return_counts=True)
    return bool((c == 2).all()) and m.is_winding_consistent


def test_tree_from_graph():
    doc, _ = _lens()
    t = CreaseTree.from_graph(FieldGraph(doc))
    assert [lf.name for lf in t.leaves] == ["a", "b"]
    P = np.random.default_rng(0).uniform(-12, 12, (500, 3))
    f, lab = t.evaluate(P)
    assert np.allclose(f, FieldGraph(doc).evaluate(P))
    # subtract negates, a shell is max(f, -(f + t)), a body met twice is one leaf
    nodes = {"a": _ball(-3, 6), "b": _ball(3, 5), "s": {"op": "shell", "input": "a", "thickness": 1},
             "c": {"op": "intersect", "inputs": ["b", "a"]},
             "part": {"op": "union", "inputs": [{"op": "subtract", "inputs": ["s", "b"]}, "c"]}}
    g = FieldGraph({"nodes": nodes, "output": "part", "bounds": {"min": [-10] * 3, "max": [10] * 3}})
    t = CreaseTree.from_graph(g)
    assert sorted((lf.name, lf.sign) for lf in t.leaves) == [("a", 1.0), ("a+1", -1.0), ("b", -1.0), ("b", 1.0)]
    assert np.allclose(t.evaluate(P)[0], g.evaluate(P))
    # nothing to sharpen: a blended boolean, a plain node
    doc["nodes"]["part"]["blend"] = 1.0
    assert CreaseTree.from_graph(FieldGraph(doc)) is None


def test_meet_finds_the_intersection_curve():
    doc, d = _lens()
    t = CreaseTree.from_graph(FieldGraph(doc))
    X0 = np.random.default_rng(1).normal(0, 0.3, (200, 3)) + _circle(d, 10.0, 200)
    X, ok = meet(t, X0, np.tile([0, 1], (200, 1)), 0.5)
    assert ok.all()
    rc = math.sqrt(100 - d * d / 4)
    assert np.abs(X[:, 0]).max() < 1e-6 and np.abs(np.linalg.norm(X[:, 1:], axis=1) - rc).max() < 1e-6


def test_creases_are_sharp_where_the_field_rounds_them():
    """Meshed plainly, the 90 degree rim of two organic spheres is rounded
    (about 0.65 pitch off at the rim); with mesh.creases the true circle is
    within the chord sag of the mesh, and the mesh stays closed and clean."""
    for op in ("intersect", "union"):
        doc, d = _lens(op)
        g = FieldGraph(doc)
        C = _circle(d, 10.0)
        before = g.mesh(pitch=0.4)
        after = g.mesh(pitch=0.4, creases=True)
        e0 = mesh_distance(before, C).max()
        e1 = mesh_distance(after, C).max()
        assert e0 > 0.2, (op, e0)
        assert e1 < 0.01, (op, e1)
        assert _closed(after) and len(after.split(only_watertight=False)) == 1
        assert after.metadata["creases"]["split"] > 50, after.metadata["creases"]
        assert abs(after.volume / before.volume - 1) < 0.01, (after.volume, before.volume)
        # the rim is a crease: an edge on the circle with ~90 degrees between its faces
        V, F = np.asarray(after.vertices), np.asarray(after.faces)
        E, fe = _edges(F)
        N = after.face_normals
        ef = [[] for _ in range(len(E))]
        for fi, row in enumerate(fe):
            for e in row:
                ef[e].append(fi)
        mid = 0.5 * (V[E[:, 0]] + V[E[:, 1]])
        on = np.abs(mid[:, 0]) < 0.02
        ang = np.degrees(np.arccos(np.clip([N[ef[i][0]] @ N[ef[i][1]] for i in np.flatnonzero(on)], -1, 1)))
        assert np.median(ang) > 80, np.median(ang)


def test_a_graph_option_and_the_default():
    doc, _ = _lens(creases=True)
    m = FieldGraph(doc).mesh(pitch=0.6)
    assert "creases" in m.metadata
    doc, _ = _lens()
    assert "creases" not in FieldGraph(doc).mesh(pitch=0.6).metadata


TESTS = [test_tree_from_graph, test_meet_finds_the_intersection_curve,
         test_creases_are_sharp_where_the_field_rounds_them, test_a_graph_option_and_the_default]


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
