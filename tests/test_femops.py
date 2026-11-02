"""femops: compiled SIMP kernels give numpy's numbers bit for bit."""
import numpy as np
import pytest

from loadpath import femops
from loadpath.fem import HexMesh, Stiffness

pytestmark = pytest.mark.skipif(not femops.available(), reason="numba not installed")


def _stiff():
    rng = np.random.default_rng(0)
    g = rng.random((14, 9, 7)) < 0.6
    g[:, :, 0] = True
    mesh = HexMesh(np.argwhere(g), 1.3, np.zeros(3))
    fixed = np.zeros(mesh.n_dof, bool)
    fixed[: 3 * 20] = True
    return mesh, Stiffness(mesh, 0.35, fixed), rng


def test_assemble_and_energy_match_numpy(monkeypatch):
    mesh, st, rng = _stiff()
    scale = 1e-9 + rng.random(mesh.n_elem) ** 3
    u = rng.standard_normal(mesh.n_dof)
    v = rng.standard_normal(mesh.n_dof)
    k1 = st.assemble(scale)
    e1 = st.strain_energy(u)
    f1 = femops.element_energy(u, v, mesh.edof, st.ke)
    monkeypatch.setenv("LOADPATH_FEMOPS", "numpy")
    k0 = st.assemble(scale)
    e0 = st.strain_energy(u)
    f0 = np.einsum("ij,jk,ik->i", u[mesh.edof], st.ke, v[mesh.edof])
    for a, b in ((k1, k0),):
        np.testing.assert_array_equal(a.data, b.data)
        np.testing.assert_array_equal(a.indices, b.indices)
    np.testing.assert_array_equal(e1, e0)
    np.testing.assert_array_equal(f1, f0)
