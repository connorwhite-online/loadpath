"""
Noise fields and surface textures: `noise` (gradient) and `cellular`
(Worley) in expressions, and a texture as an offset of a solid by a field,
meshed watertight with the texture's relief where the field puts it.

Run: `python tests/test_textures.py` (or pytest).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest

from loadpath.fields import FieldGraph, _cellular, _noise


def test_noise_is_smooth_bounded_and_repeatable():
    P = np.random.default_rng(0).uniform(-40, 40, (100_000, 3))
    n = _noise(*P.T)
    assert np.abs(n).max() <= 1.0 and 0.15 < n.std() < 0.4
    assert np.allclose(_noise(*np.round(P).T), 0.0)                 # zero on the lattice
    # Lipschitz: about 2.5 per unit, so a texture's field slope is known
    d = 1e-3
    slope = np.abs(_noise(*(P + [d, 0, 0]).T) - n) / d
    assert slope.max() < 3.0
    c = _cellular(*P.T)
    assert c.min() >= 0 and c.max() < 1.8 and 0.4 < c.mean() < 0.65
    assert np.array_equal(n, _noise(*P.T)) and np.array_equal(c, _cellular(*P.T))


@pytest.mark.parametrize("pattern", ["0.6 * noise(x / 3, y / 3, z / 3)",
                                     "0.8 * (0.5 - cellular(x / 4, y / 4, z / 4))"])
def test_texture_meshes_watertight_with_its_relief(pattern):
    doc = {"bounds": {"min": [-14, -14, -14], "max": [14, 14, 14]}, "nodes": {
        "box": {"op": "shape", "shape": {"type": "box", "min": [-10, -10, -10], "max": [10, 10, 10]}},
        "relief": {"op": "expr", "expr": pattern},
        "part": {"op": "offset", "input": "box", "distance": "relief"}}}
    g = FieldGraph(doc)
    mesh = g.mesh(pitch=0.4, name="part")
    assert mesh.is_watertight
    # the mesh's volume is the field's (Monte Carlo inside the bounds)
    P = np.random.default_rng(1).uniform(-14, 14, (300_000, 3))
    frac = float((g.evaluate(P, "part") <= 0).mean())
    assert abs(mesh.volume - frac * 28 ** 3) / (frac * 28 ** 3) < 0.02, (mesh.volume, frac * 28 ** 3)
    # the surface really is textured: faces lie off the plain box's faces
    off = np.abs(np.abs(mesh.vertices).max(axis=1) - 10.0)
    assert np.percentile(off, 90) > 0.15


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
