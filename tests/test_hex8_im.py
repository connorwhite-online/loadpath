"""The voxel hex8 with incompatible modes (fem.hex8_element): constant-strain
patch test, pure bending one cell deep against beam theory, and the plain
element behind LOADPATH_HEX8=plain."""
import numpy as np
import pytest
from scipy.sparse.linalg import spsolve

from loadpath.fem import HexMesh, Stiffness, elastic_c, hex8_ke
from loadpath.verify import nodal_stress

E, NU = 1000.0, 0.3


def _solve(mesh, fixed, f, element, prescribed=None):
    st = Stiffness(mesh, NU, fixed, element=element)
    k = st.assemble(np.full(mesh.n_elem, E))
    u = np.zeros(mesh.n_dof)
    if prescribed is not None:
        u[fixed] = prescribed[fixed]
        kraw = Stiffness(mesh, NU, np.zeros(mesh.n_dof, bool), element=element).assemble(np.full(mesh.n_elem, E))
        f = f - kraw @ u
    free = ~fixed
    u[free] = spsolve(k[free][:, free].tocsc(), f[free])
    return st, u


@pytest.mark.parametrize("element", ["im", "plain"])
def test_constant_strain_patch(element):
    # a 3x3x3 block, a linear field held on its outer nodes: the inner nodes
    # and every nodal stress must come back exact
    ijk = np.argwhere(np.ones((3, 3, 3), bool))
    mesh = HexMesh(ijk, 0.7, np.array([0.1, -0.2, 0.3]))
    a = np.array([[1e-3, 2e-4, -3e-4], [5e-4, -2e-3, 1e-4], [-1e-4, 3e-4, 7e-4]])
    x = mesh.node_xyz
    exact = (x @ a.T).ravel()
    edge = np.any((mesh.node_ijk == 0) | (mesh.node_ijk == 3), axis=1)
    fixed = np.repeat(edge, 3)
    st, u = _solve(mesh, fixed, np.zeros(mesh.n_dof), element, prescribed=exact)
    assert np.allclose(u, exact, atol=1e-12)
    eps = np.array([a[0, 0], a[1, 1], a[2, 2], a[0, 1] + a[1, 0], a[1, 2] + a[2, 1], a[2, 0] + a[0, 2]])
    sig = elastic_c(E, NU) @ eps
    s = nodal_stress(st, u, E)
    assert np.allclose(s, sig[None], rtol=0, atol=1e-9 * np.abs(sig).max())


def _pure_bending(t, element, w=2, L=12):
    """A strip t cells deep under a couple on its end face (sigma_x = y - t/2),
    held only on its root symmetry plane. Returns (top-fibre stress / M c / I,
    end rotation / (M L / E I))."""
    ijk = np.array([(i, j, k) for i in range(L) for j in range(t) for k in range(w)])
    mesh = HexMesh(ijk, 1.0, np.zeros(3))
    x = mesh.node_xyz
    yc = t / 2.0
    root, end = np.isclose(x[:, 0], 0), np.isclose(x[:, 0], L)
    fixed = np.zeros(mesh.n_dof, bool)
    fixed[3 * np.nonzero(root)[0]] = True
    n0 = np.nonzero(root & np.isclose(x[:, 1], 0) & np.isclose(x[:, 2], 0))[0][0]
    n1 = np.nonzero(root & np.isclose(x[:, 1], 0) & np.isclose(x[:, 2], w))[0][0]
    fixed[[3 * n0 + 1, 3 * n0 + 2, 3 * n1 + 1]] = True
    # consistent nodal forces of the linear traction: for a bilinear face,
    # weight h/3 (y_j) + h/6 (y_neighbour) per cell edge, times z share
    f = np.zeros(mesh.n_dof)
    for n in np.nonzero(end)[0]:
        y, z = x[n, 1], x[n, 2]
        fy = sum((y - yc) / 3.0 + (y + d - yc) / 6.0 for d in (-1, 1) if 0 <= y + d <= t)
        fz = sum(0.5 for d in (-1, 1) if 0 <= z + d <= w)
        f[3 * n] = fy * fz
    st, u = _solve(mesh, fixed, f, element)
    s = nodal_stress(st, u, E)
    top = np.isclose(x[:, 1], t) & (x[:, 0] > 2) & (x[:, 0] < L - 2)
    ix = w * t ** 3 / 12.0
    slope = np.polyfit(x[end, 1] - yc, u.reshape(-1, 3)[end, 0], 1)[0]
    return s[top, 0].mean() / yc, slope / (ix * L / (E * ix))


def test_pure_bending_one_cell_deep():
    for t in (1, 2):
        stress, rot = _pure_bending(t, "im")
        assert stress == pytest.approx(1.0, abs=1e-6)
        assert rot == pytest.approx(1.0, abs=1e-6)
    # the plain trilinear hex8 shear-locks: a third too stiff one cell deep
    stress, rot = _pure_bending(1, "plain")
    assert rot < 0.7 and stress < 0.8


def test_switch(monkeypatch):
    im = hex8_ke(NU)
    monkeypatch.setenv("LOADPATH_HEX8", "plain")
    plain = hex8_ke(NU)
    assert not np.allclose(im, plain)
    assert np.allclose(plain, hex8_ke(NU, "plain"))
    for k in (im, plain):
        ev = np.linalg.eigvalsh(k)
        assert np.sum(np.abs(ev) < 1e-10) == 6 and ev.min() > -1e-10
    monkeypatch.setenv("LOADPATH_HEX8", "bogus")
    with pytest.raises(ValueError):
        hex8_ke(NU)


def test_multigrid_matches_assembled():
    # the matrix-free multigrid takes the same element
    from loadpath.mg import GridElasticity

    ijk = np.array([(i, j, k) for i in range(16) for j in range(2) for k in range(2)])
    mesh = HexMesh(ijk, 1.0, np.zeros(3))
    x = mesh.node_xyz
    fixed = np.repeat(np.isclose(x[:, 0], 0), 3)
    f = np.zeros(mesh.n_dof)
    tip = np.nonzero(np.isclose(x[:, 0], 16))[0]
    f[3 * tip + 1] = -1.0 / len(tip)
    _, u = _solve(mesh, fixed, f, None)
    g = GridElasticity(mesh, NU, fixed, rtol=1e-10, maxiter=2000, device="cpu")
    g.set_moduli(np.full(mesh.n_elem, E))
    um = g.solve_many([f])[0]
    assert np.allclose(um, u, rtol=0, atol=1e-6 * np.abs(u).max())
