import numpy as np
import trimesh

from loadpath.envelope import Keepout


def test_mesh_keepout_is_exact_near_the_surface():
    """A sphere's keep-out field, just off its surface, within a few hundredths of a mm (the voxel
    transform alone was out by up to half a cell, in a voxel pattern: lumpy cuts)."""
    m = trimesh.creation.icosphere(subdivisions=5, radius=10.0)
    k = Keepout("ball", m, 0.0)
    rng = np.random.default_rng(0)
    d = rng.normal(size=(2000, 3))
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    for off in (-0.4, 0.0, 0.3, 0.7):
        P = d * (10.0 + off)
        assert np.abs(k.sdf(P) - off).max() < 0.06
