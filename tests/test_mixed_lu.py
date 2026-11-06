"""The buckling solves' factorization (nonlinear._splu with node
coordinates): nested-dissection order, single precision LU refined by CG.
Its answers must be the double precision ones."""
import numpy as np
from scipy.sparse.linalg import spsolve

from loadpath.fem import HexMesh, Stiffness
from loadpath.nonlinear import _MixedLU, _splu, nested_dissection


def _bracket():
    g = np.zeros((10, 4, 8), bool)
    g[:2] = True          # wall plate
    g[:, :, 6:] = True    # arm
    m = HexMesh(np.argwhere(g), 1.5, np.zeros(3))
    fixed = np.zeros(m.n_dof, bool)
    fixed[(3 * np.flatnonzero(m.node_ijk[:, 0] == 0)[:, None] + np.arange(3)).ravel()] = True
    # grey, soft cells as the optimizer's designs have
    e = np.random.default_rng(1).uniform(1e-6, 1.0, m.n_elem) ** 3
    return m, fixed, Stiffness(m, 0.3, fixed).assemble(e)


def test_nested_dissection_is_a_permutation():
    m, _, _ = _bracket()
    order = nested_dissection(m.node_ijk, leaf=8)
    assert np.array_equal(np.sort(order), np.arange(len(m.node_ijk)))


def test_mixed_lu_matches_double_precision():
    m, _, k = _bracket()
    lu = _splu(k, m.node_ijk)
    assert isinstance(lu, _MixedLU)
    b = np.random.default_rng(0).standard_normal((m.n_dof, 3))
    x = lu.solve(b)
    ref = spsolve(k.tocsc(), b)
    assert np.linalg.norm(x - ref) <= 1e-9 * np.linalg.norm(ref)
    assert np.allclose(lu.solve(b[:, 0]), x[:, 0], rtol=1e-9, atol=0)
