"""
Tests for latticestep.py and ibrep.py: a skinned TPMS lattice part as exact
CAD (the envelope less its cavities, smooth patches on the true walls).
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


def test_exact_lattice_step():
    if not available():
        print("  (skipped: no OpenCascade)")
        return
    with tempfile.TemporaryDirectory() as td:
        doc = {**SANDWICH, "step": {"lattice": "exact"}}
        mesh, rep = run_field(doc, out_dir=td, render=False)
        st = rep["step"]
        assert "error" not in st, st
        assert st["valid"], st
        assert st["cavityShells"] >= 2, st
        assert all(c["openCavities"] == 0 for c in st["cavities"]), st["cavities"]
        # as printed: the STL's volume, to a fraction of a percent
        assert abs(st["volume"] - mesh.volume) < 0.01 * mesh.volume, (st["volume"], mesh.volume)
        assert st["readBack"]["ok"], st["readBack"]


def test_exact_needs_a_sheet_lattice():
    if not available():
        print("  (skipped: no OpenCascade)")
        return
    doc = {**SANDWICH, "nodes": {**SANDWICH["nodes"], "core": {"op": "lattice", "type": "bcc", "cell": 8.0,
                                                               "thickness": 1.0}},
           "step": {"lattice": "exact"}}
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
