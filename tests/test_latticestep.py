"""
Tests for latticestep.py and ibrep.py: lattice parts as exact CAD (smooth
patches on the true surface): strut and TPMS lattices, skinned or open.
Run: `python3 tests/test_latticestep.py` (a few minutes; skips without
OpenCascade, loadpath[step]).
"""
import sys
import tempfile
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from loadpath.fields import run_field
from loadpath.ibrep import Family, Sheet, manifold_nets, nonmanifold
from loadpath.step import available

# a slab, 1.2 mm skin over a gyroid core (clipped: a blend would bulge the outside)
SANDWICH = {"name": "slab", "nodes": {
    "box": {"op": "shape", "shape": {"type": "box", "min": [-16, -10, -8], "max": [16, 10, 8]}},
    "body": {"op": "offset", "input": "box", "distance": 0.0},
    "skin": {"op": "shell", "input": "body", "thickness": 1.2},
    "core": {"op": "lattice", "type": "gyroid", "cell": 8.0, "thickness": 0.8},
    "fill": {"op": "intersect", "inputs": ["core", "body"]},
    "joined": {"op": "union", "inputs": ["skin", "fill"], "blend": 0.8},
    "part": {"op": "intersect", "inputs": ["joined", "body"]}},
    "output": "part", "mesh": {"pitch": 0.3}, "check": False}


def test_surface_nets_closed_and_patches_on_the_surface():
    """A ball: one closed manifold quad mesh, patches within microns."""
    fam = Family("ball", lambda P: np.linalg.norm(P, axis=1) - 5.0, 1.0)
    V, Q, bad = manifold_nets(fam, np.full(3, -7.0), np.full(3, 7.0))
    assert bad == 0 and nonmanifold(Q) == 0
    sh = Sheet(fam, V, Q)
    dev = sh.deviation()
    assert dev.max() < 0.01, dev.max()
    assert not sh.folds().any()


def test_exact_skinned_gyroid():
    """A TPMS sheet sealed under a skin takes the cavity route (each side of
    the walls one smooth level set)."""
    if not available():
        print("  (skipped: no OpenCascade)")
        return
    with tempfile.TemporaryDirectory() as td:
        doc = {**SANDWICH, "step": {"lattice": "exact"}}
        mesh, rep = run_field(doc, out_dir=td, render=False)
        st = rep["step"]
        assert "error" not in st, st
        assert st["route"] == "cavities" and st["ok"], st
        assert st["cavityShells"] >= 2, st
        assert all(c["openCavities"] == 0 for c in st["cavities"]), st["cavities"]
        # as printed: the STL's volume, to a fraction of a percent
        assert abs(st["volume"] - mesh.volume) < 0.01 * mesh.volume, (st["volume"], mesh.volume)
        assert st["readBack"]["ok"], st["readBack"]


def test_exact_open_gyroid():
    """"exact" on an 8 mm gyroid block with no skin: the sheet opens onto
    every face, so the void route (open pores cut away)."""
    if not available():
        print("  (skipped: no OpenCascade)")
        return
    with tempfile.TemporaryDirectory() as td:
        doc = {**_block({"op": "lattice", "type": "gyroid", "cell": 6.0, "thickness": 0.8}),
               "step": {"lattice": "exact"}}
        mesh, rep = run_field(doc, out_dir=td, render=False)
        st = rep["step"]
        assert "error" not in st, st
        assert st["route"] == "void" and st["ok"], st
        assert st["openPores"] >= 1, st
        assert abs(st["volume"] - mesh.volume) < 0.02 * mesh.volume, (st["volume"], mesh.volume)
        assert st["stlDistance"]["stlToStep"]["p99"] < 0.1, st["stlDistance"]


def _block(core, skin=None, size=4.0):
    nodes = {"box": {"op": "shape", "shape": {"type": "box", "min": [-size] * 3, "max": [size] * 3}},
             "core": core, "fill": {"op": "intersect", "inputs": ["core", "box"]}}
    if skin:
        nodes["skin"] = {"op": "shell", "input": "box", "thickness": skin}
        nodes["joined"] = {"op": "union", "inputs": ["skin", "fill"], "blend": 0.5}
        nodes["part"] = {"op": "intersect", "inputs": ["joined", "box"]}
    else:
        nodes["part"] = {"op": "intersect", "inputs": ["fill", "box"]}
    return {"name": "block", "nodes": nodes, "output": "part", "mesh": {"pitch": 0.15}, "check": False}


def test_exact_open_strut_lattice():
    """An octet block open on every face, node blends: the struts cut by the
    box faces (the envelope split by the open void), valid, on the STL."""
    if not available():
        print("  (skipped: no OpenCascade)")
        return
    doc = {**_block({"op": "lattice", "type": "octet", "cell": 4.0, "thickness": 0.9, "blend": 0.3}),
           "step": {"lattice": "exact"}}
    with tempfile.TemporaryDirectory() as td:
        mesh, rep = run_field(doc, out_dir=td, render=False)
        st = rep["step"]
        assert "error" not in st, st
        assert st["valid"] and st["readBack"]["ok"], st
        assert st["openPores"] >= 1 and st["split"]["freeEdges"] == 0 and st["ok"], st
        # the STL's facets sag inside the struts' curve: a percent or two
        assert abs(st["volume"] - mesh.volume) < 0.03 * mesh.volume, (st["volume"], mesh.volume)
        d = st["stlDistance"]
        assert d["stlToStep"]["max"] < 0.15 and d["stepToStl"]["max"] < 0.15, d


def test_exact_skinned_strut_lattice():
    """A BCC core under a skin: the void is sealed (inner shells, no cut)."""
    if not available():
        print("  (skipped: no OpenCascade)")
        return
    doc = {**_block({"op": "lattice", "type": "bcc", "cell": 4.0, "thickness": 1.0, "blend": 0.3}, skin=0.8),
           "step": {"lattice": "exact"}}
    with tempfile.TemporaryDirectory() as td:
        mesh, rep = run_field(doc, out_dir=td, render=False)
        st = rep["step"]
        assert "error" not in st, st
        assert st["valid"] and st["readBack"]["ok"], st
        assert st["sealedCavities"] >= 1 and st["openPores"] == 0, st
        # wedge pockets where the diagonal struts meet the skin are thinner
        # than the patches; some fall between grid nodes (up to ~2.5% of the
        # part here), and the report must say so
        assert abs(st["volume"] - mesh.volume) < 0.03 * mesh.volume, (st["volume"], mesh.volume)
        heavy = st["volume"] > 1.015 * mesh.volume
        assert st["complete"] == (not heavy), st


def test_cavities_needs_a_sheet_lattice():
    if not available():
        print("  (skipped: no OpenCascade)")
        return
    doc = {**SANDWICH, "nodes": {**SANDWICH["nodes"], "core": {"op": "lattice", "type": "bcc", "cell": 8.0,
                                                               "thickness": 1.0}},
           "step": {"lattice": "cavities"}}
    with tempfile.TemporaryDirectory() as td:
        _, rep = run_field(doc, out_dir=td, render=False)
        assert "error" in rep["step"] and "sheet" in rep["step"]["error"], rep["step"]


if __name__ == "__main__":
    failed = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            t0 = time.time()
            try:
                fn()
                print(f"ok   {name} ({time.time() - t0:.1f}s)")
            except Exception:  # noqa: BLE001
                failed += 1
                print(f"FAIL {name}")
                traceback.print_exc()
    sys.exit(1 if failed else 0)
