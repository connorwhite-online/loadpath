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
