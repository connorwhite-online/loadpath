"""The lattice-Boltzmann flow module against exact solutions: plane Poiseuille
flow (f Re = 96), its heat transfer under both heating conventions (Nu 10 and
140/17), and Hagen-Poiseuille in a voxel pipe run port to port."""
import numpy as np
import pytest

from loadpath import flow

pytestmark = pytest.mark.skipif(not flow.available(), reason="loadpath.flow needs numba")

H = 10


def _channel():
    m = np.zeros((4, 4, H + 2), bool)
    m[:, :, 1:-1] = True
    return m


@pytest.mark.parametrize("nu", [0.1, 0.01])
def test_plane_poiseuille_friction_is_exact_at_any_viscosity(nu):
    g = 1e-4 * nu                     # u ~ 1e-3: well above float32 noise
    r = flow.cell_flow(_channel(), 1.0, 0, forces=[g], nu=nu, heat=False, cs=0.0, tol=1e-5, max_steps=200000, develop=0)
    p = r["points"][0]
    assert r["dh"] == pytest.approx(2 * H, rel=1e-6)
    assert p["f"] * p["re"] == pytest.approx(96.0, rel=0.01)


@pytest.mark.parametrize("heating,exact", [("uniform", 10.0), ("axial", 140.0 / 17.0)])
def test_plane_channel_nusselt(heating, exact):
    r = flow.cell_flow(_channel(), 1.0, 0, forces=[1e-6], nu=0.1, pr=1.0, heating=heating, cs=0.0, tol=1e-4, develop=0,
                       max_steps=200000)
    assert r["points"][0]["nu"] == pytest.approx(exact, rel=0.02)     # 10 voxels across


def test_pipe_network_matches_hagen_poiseuille():
    h, d, length = 0.25, 4.0, 40.0
    n, nz = int(d / h) + 2, int(length / h)
    ax = (np.arange(n) - (n - 1) / 2) * h
    X, Y = np.meshgrid(ax, ax, indexing="ij")
    disk = X ** 2 + Y ** 2 < (d / 2) ** 2
    fluid = np.repeat(disk[:, :, None], nz, 2)
    reg = np.zeros(fluid.shape, int)
    reg[:, :, nz // 4:] = 1
    reg[:, :, 3 * nz // 4:] = 2
    nu = 0.89e-3 / 997.0
    u = 100 * nu / (d * 1e-3)                                   # Re 100
    q = u * disk.sum() * (h * 1e-3) ** 2 * 60000.0
    r = flow.network_flow(fluid, h, {"point": [0, 0, 0], "axis": [0, 0, -1], "radius": d / 2},
                          {"point": [0, 0, length - h], "axis": [0, 0, 1], "radius": d / 2},
                          medium="water", flow_lpm=q, regions=reg, region_names=["entry", "middle", "exit"],
                          origin=np.array([ax[0], ax[0], 0.0]))
    assert r["converged"]
    dpdx = r["pressureDropByRegion"]["middle"] / (length / 2 * 1e-3)
    assert dpdx == pytest.approx(32 * 0.89e-3 * u / (d * 1e-3) ** 2, rel=0.03)
    # the developing entry costs more than fully developed flow over the same length
    assert r["pressureDropByRegion"]["entry"] > 0.5 * r["pressureDropByRegion"]["middle"]


@pytest.mark.parametrize("periodic", [(1, 1, 1), (0, 0, 0), (1, 0, 1)])
def test_generated_kernels_match_the_cell_by_cell_ones(periodic):
    """The generated row kernels (packed blocks, vectorized collision) against
    the cell-by-cell kernels they were written from, step for step, on a grid
    with every kind of node: walls, inlets and outlets on three faces,
    adiabatic, isothermal and conducting solid, closed and periodic faces,
    and a row longer than one block."""
    K = flow._kernels()
    n = 20
    shape = (n, n - 3, flow._BLOCK + 9)
    rng = np.random.default_rng(0)
    ax = [np.arange(s) + 0.5 for s in shape]
    X, Y, Z = np.meshgrid(*ax, indexing="ij")
    w = 2 * np.pi / n
    fluid = np.sin(w * X) * np.cos(w * Y) + np.sin(w * Y) * np.cos(w * Z) + np.sin(w * Z) * np.cos(w * X) > -0.3
    ntype = np.where(fluid, flow.FLUID, flow.SOLID).astype(np.int8)
    uin = np.zeros((3,) + shape, np.float32)
    if not all(periodic):
        m = ntype[0] == 0
        ntype[0][m] = flow.INLET
        uin[0, 0][m] = 0.03
        m = ntype[-1] == 0
        ntype[-1][m] = flow.OUTLET
        uin[0, -1][m] = -1
        m = ntype[:, :, 3] == 0
        ntype[:, :, 3][m] = flow.OUTLET
        uin[:, :, :, 3][:, m] = np.array([0, 0, 1], np.float32)[:, None]
        m = ntype[:, 5, :] == 0
        ntype[:, 5, :][m] = flow.INLET
        uin[:, :, 5, :][:, m] = np.array([0, -0.02, 0], np.float32)[:, None]
    tnode = np.where(fluid, 0, 2).astype(np.int8)
    tnode[(~fluid) & (rng.random(shape) < 0.3)] = 1
    tnode[(~fluid) & (rng.random(shape) < 0.3)] = 0
    tnode[ntype == flow.INLET] = 3
    tnode[ntype == flow.OUTLET] = 4
    tfix = rng.random(shape).astype(np.float32)
    alpha = (0.004 + 0.01 * rng.random(shape)).astype(np.float32)
    qsrc = np.where(fluid, 1e-3, 0).astype(np.float32)
    qvel = np.array([1e-3, -2e-3, 5e-4])
    per = np.array(periodic, np.int64)
    u0 = [(0.02 * rng.standard_normal(shape)).astype(np.float32) for _ in range(3)]
    t0 = rng.random(shape).astype(np.float32)

    def run(step, step_t):
        rho = np.ones(shape, np.float32)
        u = [a.copy() for a in u0]
        f = flow._equilibrium(rho, *u)
        fn = f.copy()
        nut = np.zeros(shape, np.float32)
        T = t0.copy()
        g = flow._equilibrium_t(T, *u)
        gn = g.copy()
        for _ in range(12):
            step(f, fn, ntype, 0.53, 0.01, 1e-6, 2e-6, -1e-6, uin, 1.0, per, rho, u[0], u[1], u[2], flow.LAMBDA, nut)
            f, fn = fn, f
            step_t(g, gn, tnode, u[0], u[1], u[2], alpha, tfix, qsrc, qvel, per, T, nut, 1 / 0.85)
            g, gn = gn, g
        return f, rho, np.array(u), nut, g, T

    a = run(K["step_cells"], K["step_t_cells"])
    b = run(flow._kernel_rows("step"), flow._kernel_rows("step_t"))
    assert np.isfinite(a[0]).all() and np.isfinite(a[4]).all()
    assert np.abs(a[2]).max() > 0.01 and np.ptp(a[5]) > 0.5          # something is moving
    for p, q in zip(a, b):
        assert np.abs(p.astype(float) - q).max() < 2e-6
