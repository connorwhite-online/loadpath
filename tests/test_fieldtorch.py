"""
Tests for fieldtorch.py, field graphs evaluated with torch: every node of
the editor's all-ops graph matches numpy (float64 to 1e-6, float32 to
1e-3), and a graded lattice meshes to the same part. Skipped without
torch. Run: `python3 -m pytest tests/test_fieldtorch.py` (about 20 s).
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pytest  # noqa: E402

torch = pytest.importorskip("torch")

from loadpath.fields import FieldGraph  # noqa: E402
from loadpath.fieldtorch import TorchField, field_device  # noqa: E402
from tests.test_editor import ALL_OPS, _workdir  # noqa: E402

GRADED = {"nodes": {
    "box": {"op": "shape", "shape": {"type": "box", "min": [0, 0, 0], "max": [60, 20, 20]}},
    "cell": {"op": "remap", "input": {"op": "coordinate", "axis": "x"}, "from": [5, 55], "to": [5, 12]},
    "wall": {"op": "expr", "expr": "0.25 + 0.1 * sin(y / 4)"},
    "lat": {"op": "lattice", "type": "gyroid", "cell": "cell", "density": "wall"},
    "core": {"op": "intersect", "inputs": ["lat", "box"]},
    "skin": {"op": "shell", "input": "box", "thickness": 1.2},
    "part": {"op": "union", "inputs": ["core", "skin"], "blend": 0.8}}, "output": "part"}


@pytest.fixture
def on_numpy(monkeypatch):
    monkeypatch.setenv("LOADPATH_FIELD_DEVICE", "numpy")


def test_device_choice(monkeypatch):
    assert field_device("numpy") is None
    assert field_device("torch").type == "cpu"
    monkeypatch.setenv("LOADPATH_FIELD_DEVICE", "torch")
    g = FieldGraph({"nodes": {"s": {"op": "shape", "shape": {"type": "sphere", "center": [0, 0, 0], "r": 5}}}})
    assert np.allclose(g.evaluate([[0, 0, 0], [8, 0, 0]]), [-5, 3])
    assert g._tf is not None


def test_every_node_matches_numpy(on_numpy):
    g = FieldGraph(ALL_OPS, base_dir=_workdir())
    P = np.random.default_rng(0).uniform(-40, 40, (50_000, 3))
    for dtype, tol in ((torch.float64, 1e-6), (torch.float32, 1e-3)):
        tf = TorchField(g, "cpu", dtype)
        for name in g.nodes:
            a, t = g.evaluate(P, name), tf(P, name)
            ok = np.isfinite(a)
            assert np.array_equal(ok, np.isfinite(t)), name
            err = np.max(np.abs(a[ok] - t[ok])) if ok.any() else 0.0
            assert err < tol, (name, dtype, err)


def test_mesh_matches_numpy(monkeypatch):
    meshes = []
    for dev in ("numpy", "torch"):
        monkeypatch.setenv("LOADPATH_FIELD_DEVICE", dev)
        g = FieldGraph(GRADED)
        meshes.append(g.mesh(pitch=0.5))
        if dev == "torch":
            assert not g._tf.fallbacks, g._tf.fallbacks
    a, b = meshes
    assert b.is_watertight
    assert abs(a.volume - b.volume) < 1e-6 * a.volume
    assert abs(len(a.faces) - len(b.faces)) <= 0.001 * len(a.faces)
