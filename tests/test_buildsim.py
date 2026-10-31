"""
Tests for loadpath.buildsim (build-process simulation): the layer-by-layer
solve against the progressive-deposition beam, a uniform shrink of a free
part leaving no stress, the FDM bed and LPBF plate physics pointing the
right way, compensation shrinking the printed error, the warp optimizer's
adjoint against finite differences, and the study key. About a minute.

Run: `python3 tests/test_buildsim.py`.
"""
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from loadpath import buildsim as B
from loadpath.fem import HexMesh
from loadpath.thermal import ThermalGrid


def test_layers_match_progressive_beam():
    """A strip laid layer by layer on a free substrate, each layer stress-
    free when laid then shrinking along x: the curvature matches the
    analytic progressive-deposition beam (Tsui & Clyne 1997) per lumping."""
    pitch, length, hs, h, eps = 0.5, 40.0, 1.5, 2.0, -0.002
    solid = np.ones((int(length / pitch), 2, int(h / pitch)), bool)
    for lump in (1, 2):
        st = B.BuildSetup(process="lpbf", material="ti64", plate=hs, plate_margin=0.0, plate_fixed=False,
                          strain=(eps, 0, 0), release="none", layer=lump * pitch)
        r = B.simulate_build(solid, pitch, (0, 0, 0), st)
        m = r.mesh
        sel = (m.node_ijk[:, 1] == 0) & (m.node_ijk[:, 2] == 0)
        x, w = m.node_xyz[sel, 0], r.u_built[sel, 2]
        mid = (x > 0.2 * length) & (x < 0.8 * length)
        fe = 2 * np.polyfit(x[mid], w[mid], 2)[0]
        an = -B.progressive_beam(1.0, hs, [lump * pitch] * int(h / (lump * pitch)), eps)
        assert abs(fe / an - 1) < 0.04, (lump, fe, an)


def test_uniform_shrink_of_free_part_is_stress_free():
    """A compatible (uniform) strain on a free part: it shrinks and stays
    stress-free, and with rigid motion removed it moves eps * distance."""
    solid = np.ones((10, 6, 4), bool)
    r = B.simulate_build(solid, 1.0, (0, 0, 0), B.BuildSetup(process="sls", material="pa12",
                                                              strain=(-0.01, -0.01, -0.01)))
    assert np.abs(r.stress).max() < 1e-6 * 1900 * 0.01
    assert abs(r.report["distortion"]["max"] - 0.01 * np.linalg.norm([5, 3, 2])) < 1e-6


def test_cut_parts_spring_up():
    """Built on a plate the bar carries residual stress; cut off along most
    of its length, its free end rises, as the AM-Bench bridge does."""
    solid = np.ones((30, 3, 6), bool)
    st = B.BuildSetup(process="lpbf", material="in625", plate=2.0, plate_margin=0.0, release="cut",
                      cut=((-1, -1, -1e-6), (24.0, 5, 1.0)), layer=1.0)
    r = B.simulate_build(solid, 1.0, (0, 0, 0), st)
    tip = (r.mesh.node_ijk[:, 0] == 0) & (r.mesh.node_ijk[:, 2] == r.mesh.node_ijk[:, 2].max())
    rise = (r.u[tip, 2] - r.u_built[tip, 2]).mean()
    assert rise > 0.01, rise
    assert r.report["residualStress"]["vmMaxBuilt"] > 100


def test_hot_bed_and_chamber_cut_fdm_warp():
    """FDM: the bed and chamber held near Tg leave less of the shrink in
    the part, so a hot enclosure warps less than an open printer."""
    solid = np.ones((60, 4, 8), bool)
    cold = B.simulate_build(solid, 0.5, (0, 0, 0), B.BuildSetup(process="fdm", material="abs", bed=60.0,
                                                                 chamber=25.0, bed_stiffness=3.0))
    hot = B.simulate_build(solid, 0.5, (0, 0, 0), B.BuildSetup(process="fdm", material="abs", bed=100.0,
                                                                chamber=70.0, bed_stiffness=3.0))
    assert hot.report["distortion"]["max"] < 0.7 * cold.report["distortion"]["max"]
    assert hot.report["peel"]["meanPeelStress"] < cold.report["peel"]["meanPeelStress"]


def test_weak_bed_bond_lets_the_ends_lift():
    """FDM: with a bond weaker than the peel tension the bar's ends let go
    of the bed during the build (the report says how much area) and lift
    higher, as built, than on a bond that holds."""
    solid = np.ones((80, 2, 6), bool)
    kw = dict(process="fdm", material="abs", bed=60.0, chamber=25.0, bed_stiffness=10.0, release="none")
    held = B.simulate_build(solid, 0.5, (0, 0, 0), B.BuildSetup(**kw))
    weak = B.simulate_build(solid, 0.5, (0, 0, 0), B.BuildSetup(bed_adhesion=0.3, **kw))
    assert held.report["detachedArea"] == 0 and 0 < weak.report["detachedArea"] < 40 * 1.0
    end = held.mesh.node_ijk[:, 0] == 0
    assert weak.u_built[end, 2].max() > 1.5 * held.u_built[end, 2].max()


def test_compensation_cuts_the_printed_error():
    """An LPBF overhang bracket: printing the pre-distorted mesh lands at
    least 50% closer to the design than printing the design, after one
    round on 1 mm voxels (bench: 93% after two rounds at 0.5 mm)."""
    import trimesh
    a = trimesh.creation.box((20, 4, 2))
    a.apply_translation((10, 2, 7))
    b = trimesh.creation.box((4, 4, 8))
    b.apply_translation((2, 2, 4))
    m = trimesh.boolean.union([a, b], engine="manifold")
    _, rep = B.compensate_mesh(m, B.BuildSetup(process="lpbf", material="ti64", plate=2.0), pitch=1.0,
                               iterations=1)
    r0, r1 = rep["rounds"][0]["maxDeviation"], rep["rounds"][-1]["maxDeviation"]
    assert r0 > 0.1 and r1 < 0.5 * r0, rep


def test_warp_gradient_matches_finite_differences():
    nx, ny, nz = 6, 2, 6
    mesh = HexMesh(np.argwhere(np.ones((nx, ny, nz), bool)), 1.0, np.zeros(3))
    grid = ThermalGrid(1.0, np.zeros(3), (nx, ny, nz), mesh, np.zeros(mesh.n_elem, bool))
    sup = np.repeat(mesh.node_ijk[:, 2] == 0, 3)
    f = np.zeros(mesh.n_dof)
    f[3 * np.flatnonzero((mesh.node_ijk[:, 2] == nz) & (mesh.node_ijk[:, 0] == nx))] = 1.0
    ev = B._warp_problem(grid, sup, f, B.WarpOptions(0.5, 1.5, process=B.BuildSetup(material="ti64"), layer=2.0))
    rng = np.random.default_rng(0)
    x = rng.uniform(0.3, 0.9, mesh.n_elem)
    _, g, _ = ev(x)
    for i in rng.choice(mesh.n_elem, 4, replace=False):
        h = 1e-6
        xp, xm = x.copy(), x.copy()
        xp[i] += h
        xm[i] -= h
        fd = (ev(xp)[0] - ev(xm)[0]) / (2 * h)
        assert abs(g[i] - fd) < 1e-5 * abs(fd) + 1e-12, (i, g[i], fd)


def test_study_key_parses():
    from loadpath.schema import lint
    from loadpath.spec import study_from_dict
    d = {"name": "t", "material": "ti64", "process": {"type": "lpbf"},
         "designSpace": [{"type": "box", "min": [0, 0, 0], "max": [20, 10, 10]}],
         "preserve": [{"id": "a", "shape": {"type": "box", "min": [0, 0, 0], "max": [2, 10, 10]}},
                      {"id": "b", "shape": {"type": "box", "min": [18, 0, 0], "max": [20, 10, 10]}}],
         "loads": [{"region": "b", "force": [0, 0, -100]}], "supports": [{"region": "a"}],
         "buildSim": {"compensate": True, "scale": 0.8, "layer": 1.0}}
    assert not lint(d), lint(d)
    s = study_from_dict(d)
    assert s.build_sim["compensate"] and s.build_sim["setup"].scale == 0.8
    d["process"] = {"type": "sla"}
    try:
        study_from_dict(d)
        raise AssertionError("buildSim on sla should be refused")
    except ValueError:
        pass


def test_warp_term_gradient_along_any_grid_axis():
    """The study optimizer's warp term (layers stacking along -y here)
    against finite differences."""
    mesh = HexMesh(np.argwhere(np.ones((5, 6, 3), bool)), 1.0, np.zeros(3))
    w = B.WarpTerm(mesh, B.BuildSetup(material="ti64"), build=(1, -1), layer=2.0, nu=0.34)
    rng = np.random.default_rng(1)
    x = rng.uniform(0.3, 0.9, mesh.n_elem)

    def e(v):
        return 1e-9 + v ** 3 * (1 - 1e-9)

    _, g = w.evaluate(e(x), 3 * x ** 2 * (1 - 1e-9))
    for i in rng.choice(mesh.n_elem, 4, replace=False):
        h = 1e-6
        xp, xm = x.copy(), x.copy()
        xp[i] += h
        xm[i] -= h
        fd = (w.evaluate(e(xp))[0] - w.evaluate(e(xm))[0]) / (2 * h)
        assert abs(g[i] - fd) < 1e-5 * abs(fd) + 1e-15, (i, g[i], fd)


def _shelf(w):
    return {"name": f"shelf-w{w:g}", "material": "ti64", "process": {"type": "lpbf", "buildDirection": "+z"},
            "designSpace": [{"type": "box", "min": [0, 0, 0], "max": [24, 4, 14]}],
            "preserve": [{"id": "flange", "shape": {"type": "box", "min": [0, 0, 0], "max": [2, 4, 14]}},
                         {"id": "pad", "shape": {"type": "box", "min": [21, 0, 11], "max": [24, 4, 14]}}],
            "supports": [{"region": "flange"}], "loads": [{"region": "pad", "force": [0, 0, -400]}],
            "goal": {"volumeFraction": 0.3}, "resolution": {"pitch": 1.0}, "extrusion": "y",
            "maxIterations": 40, "buildSim": {"warpWeight": w}}


def test_study_warp_weight_cuts_as_built_distortion():
    """buildSim.warpWeight through the study optimizer: the same shelf at
    the same volume comes out with less as-built distortion than the
    compliance-only design."""
    from loadpath.domain import build_domain
    from loadpath.simp import SimpOptions, optimize
    from loadpath.spec import study_from_dict

    dist = {}
    for w in (0.0, 1.0):
        st = study_from_dict(_shelf(w))
        dom = build_domain(st)
        assert (dom.warp is not None) == (w > 0)
        r = optimize(dom, 0.34, SimpOptions(volume_fraction=0.3, filter_radius=1.5, max_iterations=40))
        term = B.WarpTerm(dom.mesh, st.build_sim["setup"], build=(2, 1), nu=0.34)
        dist[w] = term.evaluate(1e-9 + (r.xphys > 0.5) * (1 - 1e-9))[0]
    assert dist[1.0] < 0.7 * dist[0.0], dist
    try:
        d = _shelf(1.0)
        d["process"]["buildDirection"] = "auto"
        study_from_dict(d)
        raise AssertionError("warpWeight with an auto build direction should be refused")
    except ValueError:
        pass


TESTS = [
    test_layers_match_progressive_beam,
    test_uniform_shrink_of_free_part_is_stress_free,
    test_cut_parts_spring_up,
    test_hot_bed_and_chamber_cut_fdm_warp,
    test_weak_bed_bond_lets_the_ends_lift,
    test_compensation_cuts_the_printed_error,
    test_warp_gradient_matches_finite_differences,
    test_study_key_parses,
    test_warp_term_gradient_along_any_grid_axis,
    test_study_warp_weight_cuts_as_built_distortion,
]


def main():
    failures = 0
    for t in TESTS:
        t0 = time.time()
        try:
            t()
            print(f"PASS {t.__name__} ({time.time() - t0:.1f}s)")
        except Exception:  # noqa: BLE001
            failures += 1
            print(f"FAIL {t.__name__}")
            traceback.print_exc()
    if failures:
        print(f"\n{failures}/{len(TESTS)} FAILED")
        sys.exit(1)
    print(f"\nall {len(TESTS)} passed")


if __name__ == "__main__":
    main()
