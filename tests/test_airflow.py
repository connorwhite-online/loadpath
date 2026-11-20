"""airflow.py against textbook cases: de Vahl Davis' heated cavity, a heated
vertical plate (Churchill & Chu), isothermal parallel-plate channels (Bar-Cohen
& Rohsenow), fully developed forced flow between plates (f Re 96, Nu 7.54),
a fan zone on its fan line, and the heat balance of a conjugate block."""
import numpy as np
import pytest

from loadpath import airflow
from loadpath.convection import air_properties


def _air(tc):
    rho, cp, k, mu = (float(v) for v in air_properties(tc))
    return rho, cp, k, mu


def _cavity_nu(ra, n):
    rho, cp, k, mu = _air(25.0)
    nu, al = mu / rho, k / (rho * cp)
    L = (ra * nu * al / (9.80665 / 298.15 * 10.0)) ** (1 / 3)
    pitch = L / n * 1e3
    r = airflow.solve(np.zeros((n, 1, n), bool), pitch, ambient=25.0, film=25.0, verbose=False, max_iter=2000,
                      sides={"x-": {"kind": "wall", "t": 30.0}, "x+": {"kind": "wall", "t": 20.0}, "z": "wall",
                             "y": "periodic"})
    h = pitch * 1e-3
    q = np.sum(k * h * h * (30.0 - r.t[0, 0, :]) / (0.5 * h))
    assert r.converged
    assert abs(r.balance["closure"]) < 5e-3
    return q / (k * 10.0 * h)


@pytest.mark.parametrize("ra,n,ref", [(1e4, 32, 2.243), (1e5, 48, 4.519)])
def test_heated_cavity(ra, n, ref):
    """de Vahl Davis (1983), the benchmark square cavity, mean Nu."""
    nu = _cavity_nu(ra, n)
    assert abs(nu / ref - 1) < 0.025, nu


def _isothermal(pitch, solid, sides, amb=25.0, th=45.0):
    return airflow.solve(solid, pitch, k=200.0, ambient=amb, film=0.5 * (amb + th), fixed=(solid, th), sides=sides,
                         verbose=False, max_iter=3000)


def test_vertical_plate():
    """A 40 mm plate 20 K over still air: Churchill & Chu's laminar mean Nu."""
    pitch, L, amb, th = 1.0, 40.0, 25.0, 45.0
    nzb, nzp, nza = 20, 40, 60
    solid = np.zeros((30, 1, nzb + nzp + nza), bool)
    solid[0, :, nzb:nzb + nzp] = True
    r = _isothermal(pitch, solid, {"x-": "symmetry", "y": "periodic", "x+": "open", "z": "open"})
    assert r.converged
    w = r.wall_h()
    m = w["axis"] == 0
    hbar = w["q"][m].sum() / (m.sum() * w["area"] * (th - amb))
    rho, cp, k, mu = _air(35.0)
    nu_, al = mu / rho, k / (rho * cp)
    ra = 9.80665 / 308.15 * 20.0 * (L * 1e-3) ** 3 / (nu_ * al)
    nu_cc = 0.68 + 0.670 * ra ** 0.25 / (1 + (0.492 * al / nu_) ** (9 / 16)) ** (4 / 9)
    assert abs(hbar * L * 1e-3 / k / nu_cc - 1) < 0.06


@pytest.mark.parametrize("s,lo,hi", [(8.0, 0.9, 1.1), (4.0, 0.78, 1.0)])
def test_parallel_plate_channel(s, lo, hi):
    """A periodic array of isothermal vertical plates, 40 mm tall, s apart,
    20 K over ambient: Bar-Cohen & Rohsenow's composite of the developed
    (El/24) and isolated-plate limits. Toward the developed limit the solver
    reads lower than the composite: the composite leaves out the entry and
    exit losses, which are a third of the buoyant head there."""
    pitch, L, amb, th = 0.5, 40.0, 25.0, 45.0
    nxg = int(round(s / pitch))
    nzb, nzp = int(15 / pitch), int(L / pitch)
    solid = np.zeros((nxg + 1, 1, nzb + nzp + 2), bool)
    solid[0, :, nzb:nzb + nzp] = True
    r = _isothermal(pitch, solid, {"x": "periodic", "y": "periodic", "z": "open"})
    assert r.converged
    w = r.wall_h()
    m = w["axis"] == 0
    hbar = w["q"][m].sum() / (m.sum() * w["area"] * (th - amb))
    rho, cp, k, mu = _air(35.0)
    nu_, al = mu / rho, k / (rho * cp)
    el = 9.80665 / 308.15 * 20.0 * (s * 1e-3) ** 4 / (nu_ * al * L * 1e-3)
    nu_br = (576 / el ** 2 + 2.873 / el ** 0.5) ** -0.5
    ratio = hbar * s * 1e-3 / k / nu_br
    assert lo < ratio < hi, ratio


def test_forced_channel():
    """Fully developed laminar flow between isothermal plates: f Re 96 and
    Nu 7.54 on the hydraulic diameter."""
    pitch, s, U, L = 0.25, 4.0, 0.2, 80.0
    nx, nz = int(round(s / pitch)), int(L / pitch)
    r = airflow.solve(np.zeros((nx, 1, nz), bool), pitch, ambient=25.0, film=25.0, gravity=(0, 0, 0), verbose=False,
                      sides={"x-": {"kind": "wall", "t": 35.0}, "x+": {"kind": "wall", "t": 35.0}, "y": "periodic",
                             "z-": {"kind": "inlet", "velocity": U}, "z+": "open"})
    assert r.converged
    rho, mu, k = r.air["rho"], r.air["mu"], r.air["k"]
    h = pitch * 1e-3
    z = (np.arange(nz) + 0.5) * h
    p = r.p[:, 0, :].mean(axis=0)
    m = (z > 0.6 * L * 1e-3) & (z < 0.9 * L * 1e-3)
    dpdz = -np.polyfit(z[m], p[m], 1)[0]
    dh = 2 * s * 1e-3
    fre = dpdz * dh / (0.5 * rho * U * U) * rho * U * dh / mu
    assert abs(fre / 96 - 1) < 0.03, fre
    w, T = r.velocity[2, :, 0, :], r.t[:, 0, :]
    zi = int(0.45 * nz)
    tb = (w[:, zi] * T[:, zi]).sum() / w[:, zi].sum()
    nu = k * (35.0 - T[0, zi]) / (0.5 * h) / (35.0 - tb) * dh / k
    assert abs(nu / 7.54 - 1) < 0.02, nu
    assert abs(r.balance["closure"]) < 1e-3


def test_fan_on_its_line():
    """A fan zone in a plate channel settles where its straight fan line
    meets the channel's friction and the opening losses."""
    pitch, s, L = 0.5, 4.0, 60.0
    nx, nz = int(s / pitch), int(L / pitch)
    fan_cells = np.zeros((nx, 1, nz), bool)
    fan_cells[:, :, 10:16] = True
    dp0, q0 = 2.0, 4e-4 * pitch * 1e-3        # Pa, m^3/s through this one-cell-deep slice
    r = airflow.solve(np.zeros((nx, 1, nz), bool), pitch, ambient=25.0, film=25.0, gravity=(0, 0, 0), verbose=False,
                      sides={"x": "wall", "y": "periodic", "z": "open"}, max_iter=3000,
                      fan=airflow.Fan(fan_cells, axis=2, sign=1, curve=(dp0, q0)))
    assert r.converged
    q = r.air["fan flow m3/s"]
    dp = r.air["fan dp Pa"]
    assert abs(dp - dp0 * (1 - q / q0)) < 1e-6 + 0.01 * dp0
    U = q / (s * 1e-3 * pitch * 1e-3)
    rho, mu = r.air["rho"], r.air["mu"]
    friction = 12 * mu * U * L * 1e-3 / (s * 1e-3) ** 2
    # inlet: room air to the channel's mean speed; outlet: the parabolic
    # profile's kinetic energy (1.54 x the mean's) is lost
    loss = friction + 0.5 * rho * U * U * (1.0 + 1.54)
    assert abs(dp / loss - 1) < 0.12, (dp, loss)
    assert U > 0.05


def test_conjugate_balance():
    """A heated aluminium block in a box of air with open sides: all its heat
    leaves through the openings, and the block is hotter than the air."""
    pitch = 1.0
    solid = np.zeros((24, 1, 40), bool)
    solid[10:14, :, 8:20] = True
    heat = np.zeros(solid.shape)
    heat[11:13, :, 12:16] = 0.05 / 8               # 50 mW per mm of depth
    r = airflow.solve(solid, pitch, k=200.0, heat=heat, ambient=25.0, verbose=False, max_iter=3000,
                      sides={"y": "periodic"})
    assert r.converged
    assert abs(r.balance["closure"]) < 5e-3
    assert abs(r.balance["carried out W"] - 0.05) < 1e-3
    assert r.t[solid].min() > r.t[~solid].mean()
    assert r.max_speed() > 0.01
    w = r.wall_h()
    assert np.all(w["h"][w["q"] > 1e-6] > 0)


def test_energy_multigrid_matches_direct(monkeypatch):
    """Past 30,000 cells the energy equation goes through classical algebraic
    multigrid, kept from pass to pass, not a direct solve. On the conjugate
    case (aluminium at 200 W/(m K) in air at 0.026, with advection) the two
    give the same field."""
    pitch = 1.0
    solid = np.zeros((24, 1, 40), bool)
    solid[10:14, :, 8:20] = True
    heat = np.zeros(solid.shape)
    heat[11:13, :, 12:16] = 0.05 / 8
    kw = dict(k=200.0, heat=heat, ambient=25.0, verbose=False, max_iter=3000, sides={"y": "periodic"})
    direct = airflow.solve(solid, pitch, **kw)
    seen = []
    setup = airflow._TSolve._setup
    monkeypatch.setattr(airflow, "_DIRECT_BELOW", 0)
    monkeypatch.setattr(airflow._TSolve, "_setup", lambda self, M: (seen.append(self), setup(self, M))[1])
    amg = airflow.solve(solid, pitch, **kw)
    assert amg.converged and abs(amg.balance["closure"]) < 5e-3
    assert abs(amg.iterations - direct.iterations) <= 3
    assert np.abs(amg.t - direct.t).max() < 5e-3, np.abs(amg.t - direct.t).max()
    assert np.abs(amg.velocity - direct.velocity).max() < 1e-3 * direct.max_speed()
    stats = seen[0].stats
    assert stats["factorizations"] == 0
    # the hierarchy is reused: far fewer setups than passes
    assert stats["amg setups"] < 0.25 * amg.iterations, stats


def test_side_checks():
    with pytest.raises(ValueError):
        airflow.solve(np.zeros((4, 4, 4), bool), 1.0, sides={"x-": "periodic"}, verbose=False)
    with pytest.raises(ValueError):
        airflow.solve(np.ones((4, 4, 4), bool) & (np.arange(4) == 1)[:, None, None], 1.0, verbose=False)
