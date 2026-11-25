"""A bearing load through a rigid ring that only pushes (insert.py), and the
fixed pressure's shape (domain.bearing_forces).

- contact(): the patch forces minimize the complementary energy under
  equilibrium with every force a push, against a general solver;
- a bore's end face (a bearing's shoulder) takes none of a radial load,
  and the wall takes pressure square to it;
- a block with a 2 mm wall over its bore, pulled up through the bore: the
  ring's forces add up to the load, every one pushes, and the thin wall
  bends half what the fixed pressure bends it (the ring keeps the
  bore round, so the pillars beside it carry the load).
"""
import numpy as np
import scipy.sparse.linalg as spla

from loadpath.domain import bearing_forces, boundary_conditions, build_domain
from loadpath.fem import Stiffness
from loadpath.insert import _rigid_basis, contact, redistribute
from loadpath.shapes import Cylinder
from loadpath.spec import study_from_dict


def test_contact_is_the_constrained_minimum():
    from scipy.optimize import minimize

    rng = np.random.default_rng(3)
    n = 10
    q = rng.normal(size=(n, n))
    c = q @ q.T + n * np.eye(n)
    ang = np.linspace(0, 2 * np.pi, n, endpoint=False)
    wrench = np.zeros((n, 6))
    wrench[:, 1], wrench[:, 2] = np.cos(ang), np.sin(ang)          # patches round an x axis
    wrench[:, 4] = rng.normal(size=n) * 0.1                        # a little tilt from each
    load = np.array([0, 40.0, -25.0, 0, 0, 0])
    basis = _rigid_basis(np.array([1.0, 0, 0]))
    lam = contact(c, wrench, load, basis)
    assert (lam >= -1e-12).all()
    a, b = basis.T @ wrench.T, basis.T @ load
    keep = np.abs(a).sum(axis=1) > 0
    a, b = a[keep], b[keep]
    np.testing.assert_allclose(a @ lam, b, atol=1e-6 * np.linalg.norm(b))
    ref = minimize(lambda x: 0.5 * x @ c @ x, np.full(n, 10.0), jac=lambda x: c @ x, bounds=[(0, None)] * n,
                   constraints=[{"type": "eq", "fun": lambda x: a @ x - b, "jac": lambda x: a}], method="SLSQP",
                   options={"ftol": 1e-14, "maxiter": 500})
    assert 0.5 * lam @ c @ lam <= 0.5 * ref.x @ c @ ref.x * (1 + 1e-6)


def test_bearing_pressure_is_square_to_the_wall():
    cyl = Cylinder([-5, 0, 0], [5, 0, 0], 4.0)
    th = np.linspace(0, 2 * np.pi, 24, endpoint=False)
    side = np.stack([np.repeat([-2.5, 2.5], 24), 4.2 * np.tile(np.cos(th), 2), 4.2 * np.tile(np.sin(th), 2)], 1)
    cap = np.stack([np.full(24, 5.2), 2.5 * np.cos(th), 2.5 * np.sin(th)], 1)    # a shoulder beyond the end
    pts = np.vstack([side, cap])
    share = np.full(len(pts), 1.0 / len(pts))
    f = bearing_forces(cyl, pts, share, np.array([0.0, 100.0, 0.0]))
    np.testing.assert_allclose(f.sum(axis=0), [0, 100, 0], atol=1e-9)
    assert np.abs(f[48:]).max() < 1e-12                 # the shoulder takes none of it
    on = np.abs(f[:48]).sum(axis=1) > 0
    radial = side[on, 1:] / np.linalg.norm(side[on, 1:], axis=1, keepdims=True)
    t = f[:48][on, 1:]
    # square to the wall: no part along it, beyond the even share that balances the grid's sampling
    tangential = np.abs(t[:, 0] * radial[:, 1] - t[:, 1] * radial[:, 0])
    assert tangential.max() < 0.05 * np.linalg.norm(t, axis=1).max()
    # the thrust bears on the shoulder alone
    g = bearing_forces(cyl, pts, share, np.array([30.0, 0.0, 0.0]))
    np.testing.assert_allclose(g[48:].sum(axis=0), [30, 0, 0], atol=1e-9)
    assert np.abs(g[:48]).max() < 1e-12


def _block():
    return study_from_dict({
        "designSpace": {"type": "box", "min": [-4, -15, 0], "max": [4, 15, 24]},
        "keepOut": [{"id": "bore", "shape": {"type": "cylinder", "center": [0, 0, 16], "axis": "x", "length": 12,
                                              "r": 6}}],
        "supports": [{"shape": {"type": "box", "min": [-4, -15, -0.5], "max": [4, 15, 0.3]}, "fix": "all"}],
        "loadCases": [{"name": "up", "loads": [{"region": "bore", "force": [0, 0, 100],
                                                  "distribution": "bearing"}]}],
        "material": "petg", "pitch": 1.0, "goal": {"safetyFactor": 2}})


def test_ring_keeps_the_bore_round():
    st = _block()
    d = build_domain(st)
    m = d.mesh
    bc = boundary_conditions(st, m)
    assert len(bc.inserts) == 1
    k = Stiffness(m, st.material.nu, bc.fixed).assemble(np.full(m.n_elem, st.material.E))
    free = ~bc.fixed
    lu = spla.splu(k[free][:, free].tocsc())

    def solve(rhs):
        out = []
        for r in rhs:
            u = np.zeros(m.n_dof)
            u[free] = lu.solve(np.asarray(r)[free])
            out.append(u)
        return out

    fixed_load = bc.loads[0]
    ringed = redistribute(solve, [fixed_load], bc.inserts, lambda ci: ci, m.n_dof, m.node_xyz)[0]
    np.testing.assert_allclose(ringed.reshape(-1, 3).sum(axis=0), [0, 0, 100], atol=1e-6)
    ins = bc.inserts[0]
    on = ringed.reshape(-1, 3)[ins.nodes]
    assert (np.einsum("ij,ij->i", on, ins.normals) >= -1e-9).all()      # every patch pushes
    xyz = m.node_xyz
    top = np.abs(xyz[:, 2] - xyz[:, 2].max()) < 1e-9
    h = m.pitch

    def bend(f):
        u = solve([f])[0].reshape(-1, 3)
        mid = top & (np.abs(xyz[:, 1]) < h)
        side = top & (np.abs(np.abs(xyz[:, 1]) - 10) < h)
        return float(u[mid, 2].mean() - u[side, 2].mean())

    b_fixed, b_ring = bend(fixed_load), bend(ringed)
    assert b_fixed > 0 and b_ring < 0.6 * b_fixed, (b_fixed, b_ring)


if __name__ == "__main__":
    test_contact_is_the_constrained_minimum()
    test_bearing_pressure_is_square_to_the_wall()
    test_ring_keeps_the_bore_round()
    print("ok")
