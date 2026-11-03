"""
Tests for lattice3mf.py (lattices as lattices in 3MF: implicit TPMS level
sets and beam lattices). Run: `python3 tests/test_lattice3mf.py` (about a
minute; skips without lib3mf, loadpath[3mf]).
"""
import os
import sys
import tempfile
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from loadpath import lattice3mf as L
from loadpath.fields import FieldGraph, run_field


def _box(kind, thickness):
    return {"name": f"box-{kind}", "nodes": {
        "env": {"op": "shape", "shape": {"type": "box", "min": [0, 0, 0], "max": [32, 16, 10]}},
        "skin": {"op": "shell", "input": "env", "thickness": 1.2},
        "grad": {"op": "remap", "input": {"op": "coordinate", "axis": "x"}, "from": [0, 32], "to": [0.8, 1.6]},
        "core": {"op": "lattice", "type": kind, "cell": 8, "thickness": thickness},
        "fill": {"op": "intersect", "inputs": ["core", "env"]},
        "bore": {"op": "shape", "shape": {"type": "cylinder", "center": [16, 8, 5], "axis": "z", "length": 30,
                                          "r": 3}},
        "part": {"op": "subtract", "inputs": [{"op": "union", "inputs": ["skin", "fill"]}, "bore"]}},
        "output": "part", "mesh": {"pitch": 0.4}, "check": False, "lattice3mf": True}


def test_conformal_phases_go_in_as_sine_and_cosine():
    """A conformal frame's in-plane coordinates are phases (mod 1): as raw
    images they jumped a cycle at every wrap and a fifth of the part came
    out wrong. Shipped as sine and cosine (and the defect fill) the
    implicit matches the field to about a percent of the volume."""
    tube = {"op": "shape", "shape": {"type": "cylinder", "center": [20, 0, 0], "axis": "x", "length": 40, "r": 10}}
    g = FieldGraph({"nodes": {"S": tube, "core": {"op": "lattice", "type": "gyroid", "cell": 6, "thickness": 0.8,
                                                  "frame": {"type": "conformal", "surface": "S"}},
                              "part": {"op": "intersect", "inputs": ["core", "S"]}}, "output": "part"})
    lo, hi = g.bounds()
    lat = L.lattice_from_graph(g, "core", lo, hi)
    assert {"core-s0", "core-c0", "core-s1", "core-c1", "core-u2"} <= set(lat["images"]), sorted(lat["images"])
    assert "core-u0" not in lat["images"]
    P = np.random.default_rng(0).uniform(lo, hi, (20000, 3))
    P = P[g.evaluate(P, "S") < 0]
    a, b = L.evaluate(lat["expr"], P, lat["images"]), g.evaluate(P, "core")
    assert ((a < 0) != (b < 0)).mean() < 0.02, ((a < 0) != (b < 0)).mean()


def test_tpms_expression_is_the_field():
    """The implicit function is loadpath's own TPMS field: same numbers,
    constant or graded wall (the graded one through the 3D image), and a
    graded cell through images of the lattice coordinates."""
    P = np.random.default_rng(0).uniform((0, 0, 0), (32, 16, 10), (5000, 3))
    for kind in ("gyroid", "diamond", "schwarz-p"):
        for th, tol in ((1.0, 1e-9), ("grad", 0.01)):
            g = FieldGraph(_box(kind, th))
            lat = L.lattice_from_graph(g, "core", (0, 0, 0), (32, 16, 10))
            d = np.abs(L.evaluate(lat["expr"], P, lat.get("images")) - g.evaluate(P, "core"))
            assert d.max() < tol, (kind, th, d.max())
    nodes = {**_box("gyroid", 1.0)["nodes"], "grad": {"op": "remap", "input": {"op": "coordinate", "axis": "x"},
                                                      "from": [0, 32], "to": [5, 9]}}
    nodes["core"] = {"op": "lattice", "type": "gyroid", "cell": "grad"}
    g = FieldGraph({**_box("gyroid", 1.0), "nodes": nodes})
    lat = L.lattice_from_graph(g, "core", (0, 0, 0), (32, 16, 10))
    assert {"core-u0", "core-u1", "core-u2", "core-cell"} <= set(lat["images"]), sorted(lat["images"])
    d = np.abs(L.evaluate(lat["expr"], P, lat["images"]) - g.evaluate(P, "core"))
    near = np.abs(g.evaluate(P, "core")) < 1.0          # where the surface is
    assert np.percentile(d[near], 99) < 0.02 and d[near].max() < 0.05, (np.percentile(d[near], 99), d.max())
    nodes["core"] = {"op": "lattice", "type": "octet", "cell": "grad"}
    try:
        L.lattice_from_graph(FieldGraph({**_box("gyroid", 1.0), "nodes": nodes}), "core", (0, 0, 0), (32, 16, 10))
        raise AssertionError("a graded strut cell should be refused")
    except ValueError:
        pass


def test_beams_tile_once():
    V, B = L.lattice_beams("cubic", 5, (0, 0, 0), (10, 10, 10))
    k = np.sort(B, axis=1)
    assert len(np.unique(k, axis=0)) == len(B)
    lengths = np.linalg.norm(V[B[:, 0]] - V[B[:, 1]], axis=1)
    assert np.allclose(lengths, 5.0), lengths


def _write(kind, th, doc=None):
    with tempfile.TemporaryDirectory() as td:
        _, rep = run_field(doc or _box(kind, th), td, render=False)
        lm = rep["lattice3mf"]
        assert "error" not in lm, lm
        stl = os.path.getsize(rep["stl"])
        return lm, stl


def test_gyroid_writes_a_levelset():
    if not L.available():
        print("  (lib3mf not installed: skipped)")
        return
    lm, stl = _write("gyroid", "grad")
    rb = lm["readBack"]
    assert rb["levelsets"] == 1 and rb["functions"] == 2 and rb["build"] == 2, rb   # implicit + image function
    assert rb["meshes"] == 2, rb                                                     # body + the lattice's region
    assert lm["lattices"][0]["images"] == ["core-wall"] and lm["bytes"] < 0.1 * stl, (lm, stl)


def test_octet_writes_a_beam_lattice():
    if not L.available():
        print("  (lib3mf not installed: skipped)")
        return
    lm, stl = _write("octet", 1.0)
    rb = lm["readBack"]
    assert rb["beams"] > 300 and rb["levelsets"] == 0 and rb["build"] == 2, rb
    assert lm["bytes"] < 0.1 * stl, (lm, stl)


def test_two_lattices_and_a_blend():
    """A gyroid and an octet side by side, each clipped by its own region,
    and the gyroid blended into the skin: the blend is in the level set
    (the body and region as mesh distances)."""
    if not L.available():
        print("  (lib3mf not installed: skipped)")
        return
    d = _box("gyroid", 1.0)
    n = d["nodes"]
    n["left"] = {"op": "shape", "shape": {"type": "box", "min": [-1, -1, -1], "max": [16, 17, 11]}}
    n["right"] = {"op": "shape", "shape": {"type": "box", "min": [16, -1, -1], "max": [33, 17, 11]}}
    n["oct"] = {"op": "lattice", "type": "octet", "cell": 5, "thickness": 0.9}
    n["fill"] = {"op": "intersect", "inputs": ["core", "env", "left"]}
    n["fill2"] = {"op": "intersect", "inputs": ["oct", "env", "right"]}
    n["part"] = {"op": "subtract", "inputs": [{"op": "union", "inputs": ["skin", "fill", "fill2"], "blend": 0.8},
                                              "bore"]}
    d["lattice3mf"] = {"meshReaders": True}
    lm, _ = _write(None, None, d)
    rb = lm["readBack"]
    assert rb["levelsets"] == 1 and rb["beams"] > 100 and rb["build"] == 3 and rb["meshes"] == 4, rb
    by = {x["name"]: x for x in lm["lattices"]}
    assert by["core"]["as"] == "levelset" and by["core"]["blended"] and by["core"]["blend"] == 0.8, by
    assert by["oct"]["as"] == "beamlattice" and by["oct"]["prismTriangles"] > 0, by


TESTS = [test_tpms_expression_is_the_field, test_beams_tile_once, test_gyroid_writes_a_levelset,
         test_octet_writes_a_beam_lattice, test_two_lattices_and_a_blend]


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
