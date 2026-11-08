"""
The lattice library past the built-in cells: custom strut cells from a
node-and-beam graph, stochastic Delaunay (tet) lattices, and a tet lattice
that follows a body with its nodes on the skin.

Run: `python tests/test_lattice_library.py` (or pytest).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest

from loadpath import struts
from loadpath.fields import FieldGraph

CORNERS = [[i, j, k] for i in (0, 1) for j in (0, 1) for k in (0, 1)]
BCC = {"nodes": [[0.5, 0.5, 0.5]] + CORNERS, "beams": [[0, i] for i in range(1, 9)]}


def test_custom_cell_matches_the_builtin_bcc():
    g = FieldGraph({"nodes": {
        "a": {"op": "lattice", "type": "bcc", "cell": 6, "thickness": 0.9},
        "b": {"op": "lattice", "type": "custom", "cellGraph": BCC, "cell": 6, "thickness": 0.9}}})
    P = np.random.default_rng(0).uniform(-20, 20, (5000, 3))
    assert np.allclose(g.evaluate(P, "a"), g.evaluate(P, "b"), atol=1e-9)
    # the same graph is the same kind, and its density table is the built-in one's
    k = struts.register_cell(BCC["nodes"], BCC["beams"])
    assert k == struts.register_cell(BCC["nodes"], BCC["beams"]) and struts.is_strut(k)
    assert abs(struts.strut_relative_density(k, 1.2, 6.0) - struts.strut_relative_density("bcc", 1.2, 6.0)) < 1e-3


def test_custom_cell_by_density_and_errors():
    # a cell nobody ships: cube edges plus one face diagonal per face pair
    cell = {"nodes": CORNERS, "beams": [[0, 4], [0, 2], [0, 1], [0, 6], [0, 3], [0, 5]]}
    g = FieldGraph({"nodes": {"l": {"op": "lattice", "type": "custom", "cellGraph": cell, "cell": 5,
                                    "density": 0.2}}})
    P = np.random.default_rng(1).uniform(0, 20, (60_000, 3))
    frac = float((g.evaluate(P, "l") <= 0).mean())
    assert abs(frac - 0.2) < 0.03, frac
    for bad, msg in (({"nodes": [[0, 0, 0]], "beams": [[0, 0]]}, "two nodes"),
                     ({"nodes": CORNERS, "beams": [[0, 9]]}, "index"),
                     ({"nodes": CORNERS}, "cellGraph")):
        with pytest.raises(ValueError, match=msg):
            FieldGraph({"nodes": {"l": {"op": "lattice", "type": "custom", "cellGraph": bad}}})


BOX = {"op": "shape", "shape": {"type": "box", "min": [0, 0, 0], "max": [30, 30, 30]}}


def test_delaunay_lattice_in_a_box():
    g = FieldGraph({"bounds": {"min": [0, 0, 0], "max": [30, 30, 30]}, "nodes": {
        "box": BOX,
        "tet": {"op": "lattice", "type": "delaunay", "cell": 7, "density": 0.15},
        "part": {"op": "intersect", "inputs": ["box", "tet"]}}})
    P = np.random.default_rng(2).uniform(2, 28, (80_000, 3))
    frac = float((g.evaluate(P, "tet") <= 0).mean())
    assert 0.08 < frac < 0.25, frac
    mesh = g.mesh(pitch=0.6, name="part")
    assert mesh.is_watertight and mesh.volume > 0
    # its periodic stand-in cell is a tet cell too: struts join seeds
    seg = struts.cell_segments("delaunay")
    assert len(seg) > 20


def test_tet_lattice_follows_a_body():
    R = 15.0
    doc = {"bounds": {"min": [-16, -16, -16], "max": [16, 16, 16]}, "nodes": {
        "ball": {"op": "shape", "shape": {"type": "sphere", "center": [0, 0, 0], "r": R}},
        "tet": {"op": "lattice", "type": "delaunay", "cell": 6, "thickness": 1.0,
                "frame": {"type": "conformal", "surface": "ball"}}}}
    g = FieldGraph(doc)
    g.evaluate(np.zeros((1, 3)), "tet")
    vor = next(v for k, v in g._built.items() if hasattr(v, "seeds")) if g._built else None
    if vor is None:
        vor = g._voronoi(g.nodes["tet"])
    r = np.linalg.norm(vor.seeds, axis=1)
    on_skin = np.abs(r - R) < 0.3
    assert on_skin.sum() > 4 * np.pi * R ** 2 / 6.0 ** 2 * 0.5, on_skin.sum()
    # every strut stays inside the body (its mid and quarter points)
    for t in (0.25, 0.5, 0.75):
        q = vor.edges[:, 0] + t * (vor.edges[:, 1] - vor.edges[:, 0])
        assert np.linalg.norm(q, axis=1).max() <= R + 0.1 * 6.0 + 1e-6
    # struts reach the skin: the lattice is solid just inside it in many places
    d = np.random.default_rng(3).normal(size=(4000, 3))
    skin = 0.98 * R * d / np.linalg.norm(d, axis=1, keepdims=True)
    assert (g.evaluate(skin, "tet") <= 0).mean() > 0.05


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
