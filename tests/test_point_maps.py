"""
Point maps: another tool's results (a CSV of x, y, z and values) as a field
graph's `result` node, read by inverse distance, a thin-plate spline or the
nearest sample, driving a variable wall.

Run: `python tests/test_point_maps.py` (or pytest).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest

from loadpath.fields import FieldGraph, read_point_map


def _csv(tmp_path, header=True, delim=","):
    rng = np.random.default_rng(0)
    pts = rng.uniform(-10, 10, (400, 3))
    temp = 20 + 3 * pts[:, 0] + 0.5 * pts[:, 2]             # linear: a spline reproduces it exactly
    p = tmp_path / "export.csv"
    rows = [delim.join(f"{v:.6f}" for v in (*q, t, -t)) for q, t in zip(pts, temp)]
    p.write_text(("X,Y,Z,Temperature,Neg\n" if header else "") + "\n".join(rows) + "\n")
    return p, pts, temp


def test_read_point_map_columns(tmp_path):
    p, pts, temp = _csv(tmp_path)
    d = read_point_map(str(p))
    assert np.allclose(d["points"], pts, atol=1e-5) and np.allclose(d["values"], temp, atol=1e-5)
    d = read_point_map(str(p), "neg")
    assert np.allclose(d["neg"], -temp, atol=1e-5)
    with pytest.raises(ValueError, match="no column"):
        read_point_map(str(p), "pressure")
    q, _, _ = _csv(tmp_path, header=False)
    assert np.allclose(read_point_map(str(q))["values"], temp, atol=1e-5)


def test_point_map_interpolation(tmp_path):
    p, _, _ = _csv(tmp_path)
    probe = np.array([[0.0, 0.0, 0.0], [5.0, -3.0, 2.0], [-7.0, 6.0, -4.0]])
    exact = 20 + 3 * probe[:, 0] + 0.5 * probe[:, 2]
    errs = {}
    for how in ("rbf", "idw", "nearest"):
        g = FieldGraph({"nodes": {"t": {"op": "result", "path": str(p), "quantity": "temperature",
                                        "interpolate": how}}})
        errs[how] = float(np.abs(g.evaluate(probe, "t") - exact).max())
    assert errs["rbf"] < 1e-3, errs
    assert errs["idw"] < 6.0 and errs["nearest"] < 12.0, errs
    # a spline held inside the samples' range far outside them
    g = FieldGraph({"nodes": {"t": {"op": "result", "path": str(p), "interpolate": "rbf"}}})
    far = g.evaluate(np.array([[100.0, 0.0, 0.0]]), "t")
    assert far[0] <= 20 + 3 * 10 + 0.5 * 10 + 1e-6


def test_point_map_drives_a_wall(tmp_path):
    p, _, _ = _csv(tmp_path)
    doc = {"nodes": {
        "box": {"op": "shape", "shape": {"type": "box", "min": [-8, -8, -8], "max": [8, 8, 8]}},
        "t": {"op": "result", "path": str(p), "interpolate": "rbf"},
        "wall": {"op": "remap", "input": "t", "from": [-10, 50], "to": [0.8, 3.0]},
        "shell": {"op": "shell", "input": "box", "thickness": "wall"}}}
    g = FieldGraph(doc)
    # the wall is thicker on the hot (+x) face than the cold one
    hot = g.evaluate(np.array([[8.0 - 2.0, 0.0, 0.0]]), "shell")[0]
    cold = g.evaluate(np.array([[-8.0 + 2.0, 0.0, 0.0]]), "shell")[0]
    assert hot < 0 < cold, (hot, cold)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
