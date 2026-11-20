"""
External flow round a part, held to published drag: a sphere at Re 100
(Dennis & Walker 1971, Johnson & Patel 1999: Cd 1.087), a cylinder
spanning the tunnel at Re 100 (Henderson 1995, Williamson 1996: Cd 1.33 to
1.35, Strouhal 0.164 to 0.166), the stream turned to any direction, the
surface forces as a study load case, the nested grids' bookkeeping, and a
rotor disk in still air against momentum theory.

Run: `python tests/test_aero.py` (or pytest). About six minutes.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest
import trimesh

from loadpath.aero import boundary_layer_cells, external_flow
from loadpath.spec import study_from_dict

D = 20.0
NU_AIR = 1.85e-5 / 1.18


@pytest.fixture(scope="module")
def sphere_y():
    # Re 100 on the diameter, the stream along +y, the part off the origin
    s = trimesh.creation.icosphere(subdivisions=4, radius=D / 2)
    s.apply_translation([5.0, 7.0, -3.0])
    v = 100 * NU_AIR / (D * 1e-3)
    return external_flow(s, v, (0, 1, 0), cells=16), v


def test_sphere_drag_at_re_100(sphere_y):
    r, _v = sphere_y
    # 16 cells across read 3.5% high (24: 1.7%; the error falls as the cell squared)
    assert abs(r["dragCoefficient"] / 1.087 - 1) < 0.05, r["dragCoefficient"]
    assert r["dragCoefficient"] > 1.087
    # the force is along the stream, in the part's frame
    F = np.array(r["force"])
    assert F[1] > 0 and abs(F[0]) < 0.02 * F[1] and abs(F[2]) < 0.02 * F[1]
    assert abs(r["reynolds"] - 100) < 1 and r["reScale"] == 1.0 and r["layers"] == 0
    assert r["boundaryLayerCells"] > 4 and r["blockage"] < 0.003
    # the frontal area is the sphere's shadow, not a voxel count
    assert abs(r["frontalArea"] / (np.pi * D * D / 4) - 1) < 0.01
    # the moment about the centre of a sphere is near zero
    assert np.linalg.norm(r["moment"]) < 0.02 * F[1] * D


def test_surface_forces_go_into_a_study(sphere_y, tmp_path):
    r, _v = sphere_y
    lc = r["load_case"]
    tot = sum(np.array(ld["force"]) for ld in lc["loads"])
    assert np.allclose(tot, r["wallForceTotal"], rtol=1e-6) and abs(tot[1] / r["dragMeasured"] - 1) < 0.01
    study = study_from_dict({
        "designSpace": {"type": "box", "min": [-6, -4, -14], "max": [16, 18, 8]},
        "supports": [{"shape": {"type": "box", "min": [-7, -5, -15], "max": [-5.9, 19, 9]}}],
        "loadCases": [lc], "material": "pa12", "goal": {"safetyFactor": 2}})
    assert study.load_cases[0].name.startswith("flow-")
    (tmp_path / "case.json").write_text(json.dumps(lc))


def test_cylinder_sheds_at_re_100():
    # a slice of a long cylinder, periodic along its axis: the wake sheds, and the drag
    # and the shedding frequency are the published ones
    c = trimesh.creation.cylinder(radius=D / 2, height=D / 2, sections=128)
    r = external_flow(c, 100 * NU_AIR / (D * 1e-3), (1, 0, 0), span=(0, 0, 1), cells=16)
    assert abs(r["dragCoefficient"] / 1.34 - 1) < 0.05, r["dragCoefficient"]
    assert r["strouhal"] is not None and abs(r["strouhal"] / 0.165 - 1) < 0.08, r["strouhal"]
    q = r["dynamicPressure"] * r["frontalArea"] * 1e-6
    assert 0.15 < r["sideFluctuation"] / q < 0.3, r["sideFluctuation"] / q       # lift, rms: 0.22-0.24 published
    assert r["blockage"] < 0.015


def test_sharp_edged_cube_runs_at_the_reynolds_number_the_grid_resolves():
    # a cube face-on at 15 m/s (Re 2e4): the grid holds a boundary layer for a lower
    # Reynolds number, the run is made there, and says so. At Re 81 friction still adds to
    # a cube's drag (Hoelzer & Sommerfeld 2008: about 1.5; 1.05 past Re 1e3), and the note warns
    m = trimesh.creation.box((20, 20, 20))
    r = external_flow(m, 15.0, (-1, 0, 0), cells=12, max_layers=0, max_time=40)
    assert r["reScale"] < 1 and r["reynoldsRun"] == pytest.approx((0.75 * 12) ** 2)
    assert "reading is high" in r["note"]
    assert 1.35 < r["dragCoefficient"] < 1.8, r["dragCoefficient"]
    assert r["force"][0] < 0 and abs(r["force"][1]) < 0.1 * abs(r["force"][0])
    assert boundary_layer_cells(r["reynoldsRun"], r["cellsAcross"]) == pytest.approx(4.0)


def test_nested_levels_only_read_cells_that_hold_data():
    from loadpath.cutcell import scan_voxelize
    from loadpath.flow import C19
    from loadpath.windtunnel import FLUID, SKIP, Tunnel, layout

    m = trimesh.creation.icosphere(subdivisions=3, radius=D / 2)
    h = D / 16

    def vox(hh, corner, dims):
        return scan_voxelize(m, hh, np.asarray(corner, float), np.asarray(dims))

    lev = layout(*m.bounds, h * 2, D, shells=1, voxelize=vox)
    assert lev[-1]["mask"] is not None and abs(lev[-1]["h"] - h) < 1e-12
    solid = vox(h, lev[-1]["origin"] - 0.5 * h, lev[-1]["shape"])
    t = Tunnel(lev, solid, 0.01, 0.1)
    for L in t.levels:
        valid = L.ntype != SKIP
        for name in ("gi", "ci"):
            if hasattr(L, name):
                valid[tuple(getattr(L, name).T)] = True
        fl = np.argwhere(L.ntype == FLUID)
        for i in range(1, 19):
            nb = fl - C19[i]
            assert np.all((nb >= 0) & (nb < np.array(L.shape)))
            assert valid[tuple(nb.T)].all()
    # the parent's own cells keep two cells clear of the body under a surface layer
    from scipy.ndimage import distance_transform_edt

    par, ch = t.levels[-2], t.levels[-1]
    body_p = vox(par.h, lev[-2]["origin"] - 0.5 * par.h, par.shape)
    assert distance_transform_edt(~body_p)[par.ntype == FLUID].min() >= 2.0


def test_row_step_kernel_matches_the_cell_by_cell_one():
    """The generated step kernel (straight-line source, a row at a time) and
    the cell-by-cell kernel it replaced are the same arithmetic: the same
    populations, moments and wall forces out, on every kind of level (far
    boundary and sponge, plain, body with cut links) under each closure."""
    from loadpath.cutcell import scan_voxelize
    from loadpath.windtunnel import Tunnel, _kernels, layout, wall_links

    m = trimesh.creation.icosphere(subdivisions=3, radius=D / 2)
    h = D / 12

    def vox(hh, corner, dims):
        return scan_voxelize(m, hh, np.asarray(corner, float), np.asarray(dims))

    lev = layout(*m.bounds, h * 2, D, shells=1, voxelize=vox)
    origin = lev[-1]["origin"] - 0.5 * h
    solid = vox(h, origin, lev[-1]["shape"])
    bid, q, _, _ = wall_links(solid, np.ascontiguousarray(m.triangles), lev[-1]["origin"], h)
    t = Tunnel(lev, solid, 0.004, 0.1, q=q, bid=bid)
    for _ in range(6):
        t.step()
    K = _kernels()
    if "step_cells" not in K:
        pytest.skip("LOADPATH_WINDTUNNEL_STEP=cells: only the one kernel")
    n = len(t.levels)
    rng = np.random.default_rng(0)
    cases = 0
    for k, L in enumerate(t.levels):
        sc = 2 ** (n - 1 - k)
        for sgs, force in ((0, False), (1, True), (2, False), (2, True)):
            frc = (1e-5 * rng.standard_normal((3,) + L.shape)).astype(np.float32) if force else L.frc
            # the stream is still starting: a disturbance on top, so every term is at work
            f0 = L.f + (1e-3 * rng.standard_normal(L.f.shape)).astype(np.float32)
            u0 = L.u + (1e-2 * rng.standard_normal(L.u.shape)).astype(np.float32)
            outs = []
            for name in ("step", "step_cells"):
                a = [f0.copy(), L.fn.copy(), L.ntype, t.per, L.tau0, sgs, L.csq, u0.copy(), L.un.copy(), L.rho.copy(),
                     L.P.copy(), L.tau.copy(), L.bid, L.q, np.zeros_like(L.facc), np.zeros_like(L.fb), True,
                     L.sig, L.has_sig, t.ufar, 1e-6 * sc, 0.0, -2e-6 * sc, frc, force or L.has_frc]
                K[name](*a)
                outs.append([a[i] for i in (1, 8, 9, 10, 11, 14, 15)])
            for new, old in zip(*outs):
                assert np.isfinite(old).all()
                # float32 fields of order 1e-2 to 1: equal to the last bit or two
                assert np.allclose(new, old, rtol=1e-5, atol=1e-8), (k, sgs, force, float(np.abs(new - old).max()))
            assert np.abs(outs[1][0]).max() > 1e-3
            cases += 1
    assert cases == 4 * n and any(L.has_sig for L in t.levels) and (q[q >= 0] != 0.5).any()


@pytest.mark.parametrize("periodic", [(False, False, False), (False, True, True)])
def test_gather_matches_the_cell_by_cell_sum(periodic):
    """The rim interpolation summed a line at a time is the sum one cell at
    a time (the same 64 cells and weights), to rounding."""
    from loadpath.windtunnel import _kernels

    K = _kernels()
    rng = np.random.default_rng(0)
    shape = (14, 9, 11)
    rho = rng.standard_normal(shape).astype(np.float32)
    u = rng.standard_normal((3,) + shape).astype(np.float32)
    P = rng.standard_normal((6,) + shape).astype(np.float32)
    tau = (0.5 + rng.random(shape)).astype(np.float32)
    off = np.array([3.0, 0.0, 0.0]) if periodic[1] else np.array([3.0, 2.0, 3.0])
    # child cells whose 4-cell stencils stay inside the parent along closed axes and wrap along periodic ones
    n = [np.arange(16), np.arange(18) if periodic[1] else np.arange(8), np.arange(22) if periodic[2] else np.arange(10)]
    gi = np.stack(np.meshgrid(*n, indexing="ij"), -1).reshape(-1, 3).astype(np.int64)
    per = np.array(periodic)
    a = np.zeros((len(gi), 11))
    b = np.zeros((len(gi), 11))
    K["gather"](rho, u, P, tau, gi, off, per, a)
    K["gather_cells"](rho, u, P, tau, gi, off, per, b)
    assert np.abs(b).max() > 1.0
    assert np.abs(a - b).max() < 1e-12


def test_rotor_disk_in_still_air_matches_momentum_theory():
    # an actuator disk in hover passes v_i = sqrt(T / (2 rho A))
    hub = trimesh.creation.icosphere(subdivisions=3, radius=6.0)
    r = external_flow(hub, 0.0, rotors=[{"center": [0, 0, 0], "axis": [0, 0, -1], "radius": 20.0, "thrust": 0.16,
                                         "hub": 7.0}], cells=6, max_layers=0, develop=6.0, window=3.0, max_time=15)
    rot = r["rotors"][0]
    assert abs(rot["diskVelocity"] / rot["inducedVelocity"] - 1) < 0.1, rot
    assert r["dragCoefficient"] is None


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
