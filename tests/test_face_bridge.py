"""Drawn geometry thinner than a cell stays joined through faces."""
import numpy as np
from scipy import ndimage

from loadpath.domain import _face_bridge


def _faces(g):
    return ndimage.label(g, structure=ndimage.generate_binary_structure(3, 1))[1]


def test_staircase_wall_is_joined_through_faces():
    # a one-cell diagonal wall: every cell meets the next along an edge only
    dims = (12, 12, 3)
    g = np.zeros(dims, bool)
    for i in range(12):
        g[i, i, :] = True
    f = g.astype(float) * 0.6
    assert _faces(g) == 12
    out = _face_bridge(g.reshape(-1), f.reshape(-1), dims).reshape(dims)
    assert _faces(out) == 1
    assert out[g].all()
    # one bridge per step and layer, nothing more
    assert out.sum() == g.sum() + 11 * 3


def test_corner_contact_is_joined():
    dims = (4, 4, 4)
    g = np.zeros(dims, bool)
    g[1, 1, 1] = g[2, 2, 2] = True
    out = _face_bridge(g.reshape(-1), np.zeros(g.size), dims).reshape(dims)
    assert _faces(out) == 1
    assert out.sum() == 4


def test_joined_geometry_is_left_alone():
    dims = (5, 5, 5)
    g = np.zeros(dims, bool)
    g[1:4, 1:4, 2] = True
    out = _face_bridge(g.reshape(-1), np.zeros(g.size), dims)
    assert (out == g.reshape(-1)).all()


def test_a_cell_the_design_may_fill_is_left_to_the_design():
    dims = (4, 4, 1)
    g = np.zeros(dims, bool)
    g[1, 1, 0] = g[2, 2, 0] = True
    open_cells = np.zeros(dims, bool)
    open_cells[2, 1, 0] = True
    out = _face_bridge(g.reshape(-1), np.zeros(g.size), dims, open_cells.reshape(-1))
    assert (out == g.reshape(-1)).all()
