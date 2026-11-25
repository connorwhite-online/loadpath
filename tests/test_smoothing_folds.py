"""Surface smoothing never folds the surface through itself
(reconstruct._unfold_smoothing)."""
import os

import numpy as np
import trimesh

from loadpath import reconstruct as R
from loadpath.crease import crossing_faces

DATA = os.path.join(os.path.dirname(__file__), "data", "caster-grey-band.npz")


def _smoothed(mesh, lap):
    v = np.asarray(mesh.vertices, float).copy()
    for k in range(25):                      # _smooth_surface's steps
        v = v + (0.5 if k % 2 == 0 else 0.53) * (lap @ v - v)
    return v


def test_smoothing_unfolds_a_sub_cell_tunnel():
    # 457 marching-cubes faces from the held-out caster fork, where its
    # field hovered on the level over a grey band at the wheel's clearance:
    # a tunnel 0.2-0.5 mm across that 25 smoothing steps fold through itself
    d = np.load(DATA)
    mesh = trimesh.Trimesh(d["vertices"].astype(float), d["faces"], process=False)
    f = np.asarray(mesh.faces)
    every = np.arange(len(f))
    v0 = np.asarray(mesh.vertices, float).copy()
    assert not len(crossing_faces(v0, f, every))
    lap = R._umbrella(mesh)
    v = _smoothed(mesh, lap)
    x = crossing_faces(v, f, every)
    assert len(x) > 0                                      # the defect as it was
    at = v[f[x].ravel()].mean(axis=0)
    mesh.vertices = v
    R._unfold_smoothing(mesh, v0, lap)
    out = np.asarray(mesh.vertices, float)
    assert not len(crossing_faces(out, f, every))
    # undone only round the fold (two rings and the ease, about 2 mm):
    # further out the smoothing stands
    far = np.linalg.norm(v - at, axis=1) > 2.5
    assert far.sum() > 50
    assert np.abs(out[far] - v[far]).max() < 1e-12


def test_unfold_leaves_a_sound_smoothing_alone():
    mesh = trimesh.creation.icosphere(subdivisions=3, radius=5.0)
    v0 = np.asarray(mesh.vertices, float).copy()
    lap = R._umbrella(mesh)
    v = _smoothed(mesh, lap)
    mesh.vertices = v
    R._unfold_smoothing(mesh, v0, lap)
    assert np.array_equal(np.asarray(mesh.vertices, float), v)
