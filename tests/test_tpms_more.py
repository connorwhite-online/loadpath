"""
The lidinoid, split P and Neovius sheets beside gyroid, diamond and
Schwarz P: their level sets and gradients, densities, field graphs, and the
same numbers on every evaluation path.

Run: `python tests/test_tpms_more.py` (or pytest).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest

from loadpath.fields import FieldGraph
from loadpath.lattice import SHEETS, TRIG_KINDS, relative_density, tpms, tpms_trig, wall_for_density

KINDS = TRIG_KINDS


def _trig(P):
    return [f(P[:, i]) for i in range(3) for f in (np.sin, np.cos)]


@pytest.mark.parametrize("kind", KINDS)
def test_level_set_gradient_and_period(kind):
    P = np.random.default_rng(1).uniform(0, 2 * np.pi, (400, 3))
    f, g = tpms_trig(kind, *_trig(P))
    h = 1e-6
    num = np.stack([(tpms_trig(kind, *_trig(P + h * e))[0] - tpms_trig(kind, *_trig(P - h * e))[0]) / (2 * h)
                    for e in np.eye(3)], 1)
    assert np.abs(np.stack(g, 1) - num).max() < 1e-6
    # one period a cell, and the distance reads zero on the surface
    pts = np.random.default_rng(2).uniform(0, 7, (300, 3))
    assert np.allclose(tpms(kind, pts, 7.0), tpms(kind, pts + 7.0 * np.array([1, -2, 3]), 7.0), atol=1e-9)
    # the published forms
    x, y, z = P.T
    if kind == "neovius":
        ref = 3 * (np.cos(x) + np.cos(y) + np.cos(z)) + 4 * np.cos(x) * np.cos(y) * np.cos(z)
    elif kind == "lidinoid":
        ref = (0.5 * (np.sin(2 * x) * np.cos(y) * np.sin(z) + np.sin(2 * y) * np.cos(z) * np.sin(x)
                      + np.sin(2 * z) * np.cos(x) * np.sin(y))
               - 0.5 * (np.cos(2 * x) * np.cos(2 * y) + np.cos(2 * y) * np.cos(2 * z)
                        + np.cos(2 * z) * np.cos(2 * x)) + 0.15)
    else:
        ref = (1.1 * (np.sin(2 * x) * np.sin(z) * np.cos(y) + np.sin(2 * y) * np.sin(x) * np.cos(z)
                      + np.sin(2 * z) * np.sin(y) * np.cos(x))
               - 0.2 * (np.cos(2 * x) * np.cos(2 * y) + np.cos(2 * y) * np.cos(2 * z) + np.cos(2 * z) * np.cos(2 * x))
               - 0.4 * (np.cos(2 * x) + np.cos(2 * y) + np.cos(2 * z)))
    assert np.allclose(f, ref, atol=1e-12)


@pytest.mark.parametrize("kind", KINDS)
def test_density_and_fits(kind):
    w = np.linspace(0.02, 0.3, 8)
    rho = relative_density(kind, w, 1.0)
    assert np.all(np.diff(rho) > 0) and rho[0] > 0
    assert abs(relative_density(kind, wall_for_density(kind, 0.3, 8.0), 8.0) - 0.3) < 0.01
    assert kind in SHEETS and SHEETS[kind]["C"] > 0


@pytest.mark.parametrize("kind", KINDS)
def test_field_graph_sheet_meshes_at_its_density(kind):
    doc = {"name": kind, "nodes": {
        "env": {"op": "shape", "shape": {"type": "box", "min": [0, 0, 0], "max": [16, 16, 16]}},
        "core": {"op": "lattice", "type": kind, "cell": 8, "density": 0.25},
        "part": {"op": "intersect", "inputs": ["core", "env"]}}, "output": "part"}
    g = FieldGraph(doc)
    n = 48
    c = (np.arange(n) + 0.5) * 16 / n
    P = np.stack(np.meshgrid(c, c, c, indexing="ij"), -1).reshape(-1, 3)
    frac = (g.evaluate(P, "part") < 0).mean()
    assert abs(frac - 0.25) < 0.03, frac


def test_torch_path_matches():
    torch = pytest.importorskip("torch")
    from loadpath.fieldtorch import TorchField  # noqa: F401  (exists when torch does)

    P = np.random.default_rng(3).uniform(0, 1, (500, 3))
    for kind in KINDS:
        t = torch.tensor(P, dtype=torch.float64)
        k = 2 * np.pi
        trig = [f(k * t[:, i]) for i in range(3) for f in (torch.sin, torch.cos)]
        f, g = tpms_trig(kind, *trig)
        gn = torch.sqrt(g[0] ** 2 + g[1] ** 2 + g[2] ** 2) * k
        d = (f / torch.clamp(gn, min=0.3 * k)).numpy()
        assert np.allclose(d, tpms(kind, P, 1.0), atol=1e-12)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
