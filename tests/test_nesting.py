"""
Build-volume nesting: parts and copies packed on a plate or stacked in a
powder cake, the gap between them measured.

Run: `python tests/test_nesting.py` (or pytest).
"""
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest
import trimesh

from loadpath.nesting import nest, orientations


def test_orientations():
    assert len(orientations("z")) == 4 and len(orientations("any")) == 24
    for r in orientations("any"):
        assert np.allclose(r @ r.T, np.eye(3)) and abs(np.linalg.det(r) - 1) < 1e-9


def test_plate_keeps_parts_upright_apart_and_inside():
    box = trimesh.creation.box(extents=[40, 20, 10])
    cyl = trimesh.creation.cylinder(radius=8, height=30)
    r = nest([(box, 4), (cyl, 6)], (120, 120, 80), gap=3, mode="bed")
    rep = r.report()
    assert rep["placed"] == 10 and not rep["unplaced"]
    assert rep["minGap"] >= 3.0 - 1e-6
    for p in r.placements:
        # on the plate, turned only about z
        assert abs(p.min[2]) < 1e-9 and np.allclose(p.transform[2, :3], [0, 0, 1])
    b = r.combined().bounds
    assert (b[0][:2] >= 3 - 1e-6).all() and (b[1][:2] <= 117 + 1e-6).all() and abs(b[0][2]) < 1e-9
    assert rep["buildHeight"] == pytest.approx(30.0)


def test_powder_cake_stacks_low_and_reports_what_does_not_fit():
    rod = trimesh.creation.box(extents=[60, 10, 10])
    r = nest([(rod, 12)], (80, 80, 80), gap=2, mode="volume")
    rep = r.report()
    assert rep["placed"] == 12 and rep["minGap"] >= 2.0 - 1e-6
    # laid flat and side by side, not stood on end
    assert rep["buildHeight"] < 30
    # too many for the cake: the rest are named
    big = nest([(trimesh.creation.box(extents=[50, 50, 50]), 3)], (80, 80, 80), gap=2, mode="volume")
    assert len(big.placements) == 1 and len(big.unplaced) == 2


def test_cli(tmp_path):
    stl = tmp_path / "bracket.stl"
    trimesh.creation.box(extents=[30, 20, 8]).export(stl)
    out = tmp_path / "plate.stl"
    r = subprocess.run([sys.executable, "-m", "loadpath", "nest", f"{stl}:5", "--volume", "100", "100", "50",
                        "--process", "fdm", "--out", str(out)], capture_output=True, text=True,
                       cwd=Path(__file__).resolve().parents[1])
    assert r.returncode == 0, r.stdout + r.stderr
    rep = json.loads(r.stdout)
    assert rep["mode"] == "bed" and rep["placed"] == 5 and out.exists()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
