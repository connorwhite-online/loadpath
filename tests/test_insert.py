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



def test_multigrid_takes_seat_springs_as_their_product():
    """A seat's patch springs (pushseat.hold: k p p') given to the multigrid
    as rows and k (mg.GridElasticity `lowrank`) solve as the assembled ones:
    the 608 pillow block's fine check held 32M assembled entries."""
    import scipy.sparse as sp

    from loadpath.fem import HexMesh
    from loadpath.mg import GridElasticity

    ijk = np.stack(np.meshgrid(np.arange(16), np.arange(6), np.arange(6), indexing="ij"), -1).reshape(-1, 3)
    m = HexMesh(ijk, 1.0, np.zeros(3))
    fixed = np.zeros(m.n_dof, bool)
    end = np.flatnonzero(m.node_xyz[:, 0] < 1e-9)
    fixed[3 * end] = fixed[3 * end + 1] = fixed[3 * end + 2] = True
    rows = []
    for lo, hi in ((10, 13), (13, 16.1)):                      # two patches on the top face, pushing along z
        on = np.flatnonzero((m.node_xyz[:, 2] > 5.9) & (m.node_xyz[:, 0] >= lo) & (m.node_xyz[:, 0] < hi))
        p = np.zeros(m.n_dof)
        p[3 * on + 2] = 1.0 / len(on)
        rows.append(p)
    r = sp.csr_matrix(np.array(rows))
    k = np.array([300.0, 500.0])
    held = (r.T @ sp.diags(k) @ r).tocsr()
    f = np.zeros(m.n_dof)
    f[3 * np.flatnonzero(m.node_xyz[:, 0] > 15.9) + 1] = 1.0
    e = np.full(m.n_elem, 1000.0)
    us = []
    for lr in (None, (None, r, k)):
        g = GridElasticity(m, 0.3, fixed, springs=held, rtol=1e-10, maxiter=3000, coarse_cells=50, lowrank=lr)
        g.set_moduli(e)
        us.append(g.solve_many([f])[0])
    np.testing.assert_allclose(us[1], us[0], rtol=1e-6, atol=1e-9 * np.abs(us[0]).max())


if __name__ == "__main__":
    test_contact_is_the_constrained_minimum()
    test_bearing_pressure_is_square_to_the_wall()
    test_ring_keeps_the_bore_round()
    test_multigrid_takes_seat_springs_as_their_product()
    print("ok")


def test_a_pinned_bore_is_read_at_its_net_section():
    """A plate pulled through a pin in a hole breaks at the hole's sides
    (net tension; Hertel et al. 2024, docs/VALIDATION.md). The ring's
    contact is solved, so the check reads the bore like any wall: the peak
    is at the hole's side, not on the edge of a band left out round it."""
    import trimesh

    from loadpath.verify import verify_part

    L, W, t, D, xh = 40.0, 12.0, 3.0, 4.0, 30.0
    box = trimesh.creation.box(extents=[L, W, t])
    box.apply_translation([L / 2, 0, 0])
    cyl = trimesh.creation.cylinder(radius=D / 2, height=3 * t, sections=64)
    cyl.apply_translation([xh, 0, 0])
    part = box.difference(cyl)
    st = study_from_dict({
        "designSpace": {"type": "box", "min": [0, -W / 2, -t / 2], "max": [L, W / 2, t / 2]},
        "keepOut": [{"id": "hole", "shape": {"type": "cylinder", "center": [xh, 0, 0], "axis": "z", "length": t,
                                              "r": D / 2}}],
        "supports": [{"shape": {"type": "box", "min": [-0.5, -W, -t], "max": [6, W, t]}, "fix": "all"}],
        "loadCases": [{"name": "pin", "loads": [{"region": "hole", "force": [300, 0, 0],
                                                  "distribution": "bearing"}]}],
        "material": "petg", "goal": {"safetyFactor": 1.0}})
    rep = verify_part(st, part, max_cells=40_000, min_pitch=None, contact_band=0.5, max_pitch=None)
    c = rep["cases"][0]
    x, y = c["peakAt"][0] - xh, c["peakAt"][1]
    assert np.hypot(x, y) < D / 2 + 1.0, c["peakAt"]                  # at the bore
    assert np.degrees(np.arctan2(abs(y), x)) > 60, c["peakAt"]         # at its side, not its bearing face
    net = 300 / ((W - D) * t)
    assert 1.5 * net < c["peakStress"] < 4 * net, (c["peakStress"], net)
