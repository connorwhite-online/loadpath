"""
Tests for loadpath (topology optimization). Small grids only: the
whole file runs in well under a minute. The slow, realistic parts live in
the eval suite (`python -m loadpath.evals.run`), which scores delivered
meshes against engineering requirements rather than unit contracts.

Run: `python3 tests/test_loadpath.py` (numpy, scipy, pyamg,
scikit-image, trimesh, manifold3d — the default dependency set).
"""
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import trimesh

from loadpath.domain import build_domain
from loadpath.fem import Solver, Stiffness
from loadpath.reconstruct import reconstruct
from loadpath.shapes import Box, Cylinder, shape_from_spec
from loadpath.simp import SimpOptions, filter_radius, optimize
from loadpath.spec import study_from_dict
from loadpath.study import run_study
from loadpath.verify import verify_part, voxelize

BEAM = {
    "designSpace": {"type": "box", "min": [0, 0, 0], "max": [100, 10, 10]},
    "supports": [{"shape": {"type": "box", "min": [-1, -1, -1], "max": [0.01, 11, 11]}}],
    "loadCases": [{"name": "tip", "loads": [
        {"shape": {"type": "box", "min": [99.99, -1, -1], "max": [101, 11, 11]}, "force": [0, 0, -10]}]}],
    "material": {"E": 2000, "nu": 0.3, "strength": 50, "density": 1.0},
    "goal": {"volumeFraction": 0.5},
    "resolution": {"pitch": 2.0},
}

# A small plate with a bore near the free end: exercises preserve, keep-out
# and the exact booleans without a long solve.
LUG = {
    "name": "lug",
    "designSpace": {"type": "box", "min": [0, -15, -4], "max": [60, 15, 4]},
    "preserve": [{"id": "eye", "shape": {"type": "cylinder", "center": [48, 0, 0], "axis": "z",
                                         "length": 8, "r": 9}}],
    "keepOut": [{"id": "pin", "shape": {"type": "cylinder", "center": [48, 0, 0], "axis": "z",
                                        "length": 12, "r": 4}}],
    "supports": [{"shape": {"type": "box", "min": [-1, -16, -5], "max": [0.01, 16, 5]}}],
    "loadCases": [{"name": "pull", "loads": [{"region": "pin", "force": [0, -80, 0]}]}],
    "material": "pa12",
    "goal": {"volumeFraction": 0.35},
    "symmetry": ["z"],
    "resolution": {"pitch": 2.0},
    "maxIterations": 60,
}


def test_shapes_sdf_and_exact_volume():
    b = Box([0, 0, 0], [10, 20, 30])
    assert abs(b.sdf(np.array([[5, 10, 15]]))[0] + 5) < 1e-9
    assert abs(b.sdf(np.array([[15, 10, 15]]))[0] - 5) < 1e-9
    c = Cylinder([0, 0, 0], [0, 0, 10], 5)
    assert abs(c.sdf(np.array([[0, 0, 5]]))[0] + 5) < 1e-9
    assert abs(c.sdf(np.array([[8, 0, 5]]))[0] - 3) < 1e-9
    tube = shape_from_spec({"type": "cylinder", "center": [0, 0, 0], "axis": "x", "length": 10,
                            "r": 6, "rInner": 4})
    assert tube.contains(np.array([[0, 5, 0]]))[0]
    assert not tube.contains(np.array([[0, 0, 0]]))[0]
    vol = tube.to_manifold().volume()
    exact = np.pi * (36 - 16) * 10
    assert abs(vol - exact) / exact < 0.002, vol
    lo, hi = tube.bounds()
    assert np.allclose(lo, [-5, -6, -6]) and np.allclose(hi, [5, 6, 6])


def test_spec_errors_name_the_field():
    cases = [
        ({**BEAM, "supports": []}, "support"),
        ({**BEAM, "loadCases": [{"loads": [{"region": "nope", "force": [0, 0, 1]}]}]}, None),
        ({**BEAM, "loadCases": [{"loads": [{"shape": BEAM["supports"][0]["shape"], "force": [0, 0, 0]}]}]},
         "force"),
        ({**BEAM, "goal": {"volumeFraction": 1.5}}, "volumeFraction"),
        ({**BEAM, "material": "unobtainium"}, "material"),
    ]
    for raw, word in cases:
        try:
            s = study_from_dict(raw)
            build_domain(s)
        except ValueError as exc:
            if word:
                assert word in str(exc), str(exc)
            continue
        raise AssertionError(f"expected a ValueError for {raw}")


def test_fem_matches_beam_theory():
    s = study_from_dict({**BEAM, "resolution": {"pitch": 1.25}})
    d = build_domain(s)
    st = Stiffness(d.mesh, 0.3, d.fixed)
    u = Solver(d.mesh, rtol=1e-8).solve(st.assemble(np.full(d.n_elem, 2000.0)), d.loads[0])
    tip = u[2::3][d.load_nodes].mean()
    analytic = -10 * 100 ** 3 / (3 * 2000 * 10 * 10 ** 3 / 12)   # -2.0 mm
    assert abs(tip - analytic) / abs(analytic) < 0.03, tip


def test_simp_keeps_volume_and_symmetry():
    s = study_from_dict(LUG)
    d = build_domain(s)
    res = optimize(d, 0.39, SimpOptions(volume_fraction=0.35, filter_radius=filter_radius(d, None),
                                        max_iterations=60))
    design = ~d.preserve
    assert abs(res.xphys[design].mean() - 0.35) < 0.01, res.xphys[design].mean()
    assert res.history[-1]["compliance"] < res.history[0]["compliance"] * 0.5
    m = d.mirror["z"]
    has = m >= 0
    assert np.abs(res.xphys[has] - res.xphys[m[has]]).max() < 1e-6
    grey = ((res.xphys > 0.1) & (res.xphys < 0.9))[design].mean()
    assert grey < 0.1, grey


def test_reconstruct_keeps_bores_exact_and_preserve_whole():
    s = study_from_dict(LUG)
    d = build_domain(s)
    res = optimize(d, 0.39, SimpOptions(volume_fraction=0.35, filter_radius=filter_radius(d, None),
                                        max_iterations=60))
    mesh, info = reconstruct(s, d, res.xphys)
    assert mesh.is_watertight and info["bodies"] == 1
    rep = verify_part(s, mesh, contact_band=d.pitch)
    assert rep["keepOutIntrusion"] < 1e-3, rep["keepOutIntrusion"]
    assert rep["preserveMissing"] < 1e-3, rep["preserveMissing"]
    # The bore is exactly r=4: a 3.98 pin fits with no material inside it,
    # and the ring 4.02..4.5 is solid all the way round.
    from loadpath.shapes import manifold_from_trimesh
    part = manifold_from_trimesh(mesh)
    pin = Cylinder([48, 0, -3.9], [48, 0, 3.9], 3.98).to_manifold()
    assert (part ^ pin).volume() < 1e-6
    ring = shape_from_spec({"type": "cylinder", "center": [48, 0, 0], "axis": "z", "length": 7.8,
                            "r": 4.5, "rInner": 4.02}).to_manifold()
    assert abs((part ^ ring).volume() - ring.volume()) < 1e-3 * ring.volume()
    assert rep["safetyFactor"] > 1 and rep["mass"] > 0


def test_verify_bar_stress_matches_beam_theory():
    s = study_from_dict(BEAM)
    bar = trimesh.creation.box(extents=[100, 10, 10])
    bar.apply_translation([50, 5, 5])
    rep = verify_part(s, bar, max_cells=40_000)
    # Root bending stress M c / I = (10 N * 100 mm) * 5 / (10^4 / 12) = 6 MPa;
    # the cells next to the clamp are excluded, so expect slightly under.
    peak = rep["cases"][0]["peakStress"]
    assert 4.5 < peak < 6.6, peak
    assert abs(rep["cases"][0]["deflection"] - 2.0) < 0.15, rep["cases"][0]["deflection"]
    assert abs(rep["mass"] - 10.0) < 1e-6
    assert rep["bodies"] == 1 and rep["keepOutIntrusion"] == 0.0


def test_verify_hole_stress_concentration():
    # Plate with a hole in tension: Peterson's net-section Kt = 2.43 for
    # d/W = 0.25, so the peak is ~81 MPa against a 33.3 MPa net stress. Voxel
    # stress at the hole is the hard case for this method; the submodel
    # refinement plus nodal averaging has to land near it, and the reported
    # safety factor has to carry the allowance.
    from loadpath.shapes import trimesh_from_manifold
    from loadpath.verify import STRESS_ALLOWANCE
    plate = trimesh_from_manifold(Box([0, -20, -2], [120, 20, 2]).to_manifold()
                                  - Cylinder([60, 0, -5], [60, 0, 5], 5).to_manifold())
    s = study_from_dict({
        "designSpace": {"type": "box", "min": [0, -20, -2], "max": [120, 20, 2]},
        "supports": [{"shape": {"type": "box", "min": [-1, -21, -3], "max": [0.01, 21, 3]}}],
        "loadCases": [{"name": "pull", "loads": [
            {"shape": {"type": "box", "min": [119.99, -21, -3], "max": [121, 21, 3]}, "force": [4000, 0, 0]}]}],
        "material": {"E": 70000, "nu": 0.33, "strength": 250, "density": 2.7},
        "goal": {"volumeFraction": 0.5},
    })
    rep = verify_part(s, plate, max_cells=10 ** 7, min_pitch=0.7, contact_band=3.0)
    c = rep["cases"][0]
    assert 70 < c["peakStress"] < 100, c["peakStress"]
    assert abs(c["peakAt"][0] - 60) < 2.5 and 3.5 < abs(c["peakAt"][1]) < 6.5, c["peakAt"]
    assert abs(rep["safetyFactor"] - 250 / (c["peakStress"] * STRESS_ALLOWANCE)) < 1e-9


def test_am_filter_gradient_and_behavior():
    from loadpath.am import AMFilter
    rng = np.random.default_rng(0)
    x = rng.random((5, 4, 6))
    pin = np.zeros(x.shape, bool)
    pin[2, 1, 3] = True
    f = AMFilter(x.shape, (1, -1), pin)
    w = rng.random(x.shape)
    y = f.forward(x)
    g = f.backward(w)
    num = np.zeros_like(x)
    for idx in [(0, 0, 0), (2, 3, 4), (4, 2, 1), (1, 1, 5)]:
        xp = x.copy()
        xp[idx] += 1e-6
        num[idx] = ((f.forward(xp) * w).sum() - (y * w).sum()) / 1e-6
        assert abs(num[idx] - g[idx]) < 1e-4 * max(1.0, abs(g[idx])), (idx, num[idx], g[idx])
    # A floating slab is not printable; a column from the plate is.
    field = np.zeros((20, 3, 20))
    field[:, 1, 15] = 1.0
    field[2:4, 1, 0:16] = 1.0
    out = AMFilter(field.shape, (2, 1)).forward(field)
    assert out[15, 1, 15] < 0.05 and out[3, 1, 10] > 0.95


def test_voxelize_volume():
    m = trimesh.creation.cylinder(radius=10, height=20, sections=96)
    solid, _ = voxelize(m, 0.5)
    vol = solid.sum() * 0.125
    assert abs(vol - m.volume) / m.volume < 0.03, (vol, m.volume)


def test_unreachable_load_is_refused_helpfully():
    raw = {**BEAM, "loadCases": [{"loads": [
        {"shape": {"type": "sphere", "center": [500, 0, 0], "r": 1}, "force": [0, 0, -1]}]}]}
    try:
        build_domain(study_from_dict(raw))
    except ValueError as exc:
        assert "touches no material" in str(exc), str(exc)
        return
    raise AssertionError("expected refusal")


def test_run_study_end_to_end():
    s = study_from_dict({**LUG, "goal": {"volumeFraction": 0.35, "outcomes": [0.5]}, "maxIterations": 40})
    result = run_study(s)
    assert len(result.outcomes) == 2
    a, b = result.outcomes
    assert b.report["mass"] > a.report["mass"]
    assert b.report["safetyFactor"] > a.report["safetyFactor"] * 0.9
    out = result.to_dict()
    assert out["outcomes"][0]["bodies"] == 1



def test_local_mcp_job_functions():
    # The tool bodies of `loadpath mcp`, without the MCP SDK: validate, run
    # a job in a child process, read back outcomes with STL paths.
    import os
    import tempfile
    os.environ["LOADPATH_HOME"] = tempfile.mkdtemp(prefix="loadpath-home-")
    from loadpath import mcp_server
    bad = mcp_server.check({**LUG, "supports": []})
    assert not bad["ok"] and "support" in bad["error"]
    job = mcp_server.start({**LUG, "maxIterations": 20})
    assert job["ok"], job
    deadline = time.time() + 300
    while True:
        info, png = mcp_server.status(job["jobId"])
        if info["status"] != "running" or time.time() > deadline:
            break
        time.sleep(1.0)
    assert info["status"] == "done", info
    assert os.path.exists(info["outcomes"][0]["stl"]) and png and png[:4] == b"\x89PNG"
    try:
        mcp_server.status("../etc")
    except ValueError:
        pass
    else:
        raise AssertionError("job ids must not be paths")


def test_boundary_condition_types():
    from loadpath.domain import boundary_conditions
    from loadpath.fem import HexMesh
    s = study_from_dict({
        **LUG,
        "supports": [{"region": "pin", "fix": "pin"},
                     {"shape": {"type": "box", "min": [-1, -16, -5], "max": [0.01, 16, 5]}}],
        "loadCases": [
            {"name": "torque", "loads": [{"region": "eye", "moment": [0, 0, 5000]}]},
            {"name": "bearing", "loads": [{"region": "pin", "force": [0, -300, 0], "distribution": "bearing"}]},
        ],
    })
    d = build_domain(s)
    bc = boundary_conditions(s, d.mesh, check_loads=False)
    xyz = d.mesh.node_xyz
    # Pure moment: no net force, exactly the requested moment about the centroid.
    f = bc.loads[0].reshape(-1, 3)
    on = np.abs(f).sum(axis=1) > 0
    c = xyz[on].mean(axis=0)
    assert np.allclose(f.sum(axis=0), 0, atol=1e-6)
    assert np.allclose(np.cross(xyz[on] - c, f[on]).sum(axis=0), [0, 0, 5000], rtol=1e-6, atol=1e-6)
    # Bearing: sums to the force and only pushes the half of the bore it faces.
    fb = bc.loads[1].reshape(-1, 3)
    onb = np.abs(fb).sum(axis=1) > 0
    assert np.allclose(fb.sum(axis=0), [0, -300, 0], atol=1e-6)
    assert np.all(xyz[onb][:, 1] <= 1e-6)
    # Pin: springs resist radial and axial motion, not rotation about the axis.
    rows = bc.springs.tocoo()
    n0 = rows.row[0] // 3
    blk = bc.springs[3 * n0:3 * n0 + 3, 3 * n0:3 * n0 + 3].toarray()
    rad = xyz[n0] - np.array([48, 0, xyz[n0][2]])
    tang = np.cross([0, 0, 1], rad / np.linalg.norm(rad))
    assert abs(tang @ blk @ tang) < 1e-9 * np.abs(blk).max()
    assert blk[2, 2] > 0


def test_gravity_matches_beam_theory():
    # Self-weight only on a clamped bar: w L^4 / (8 E I), with w = rho g A.
    s = study_from_dict({**BEAM, "gravity": [0, 0, -9.81], "loadCases": [{"name": "tip", "loads": [
        {"shape": {"type": "box", "min": [99.99, -1, -1], "max": [101, 11, 11]}, "force": [0, 0, -1e-9]}]}],
        "material": {"E": 2000, "nu": 0.3, "strength": 50, "density": 7.85}})
    bar = trimesh.creation.box(extents=[100, 10, 10])
    bar.apply_translation([50, 5, 5])
    rep = verify_part(s, bar, max_cells=40_000, refine=0)
    w = 7.85e-6 * 9.81 * 100          # N/mm
    analytic = w * 100 ** 4 / (8 * 2000 * 10 * 10 ** 3 / 12)
    assert abs(rep["cases"][0]["deflection"] - analytic) / analytic < 0.05, (rep["cases"][0]["deflection"], analytic)


def test_extrusion_and_materials():
    # Extrusion: every column along z carries one density (a 2.5D profile).
    s = study_from_dict({**LUG, "extrusion": "z", "symmetry": []})
    d = build_domain(s)
    res = optimize(d, 0.39, SimpOptions(volume_fraction=0.35, filter_radius=filter_radius(d, None),
                                        max_iterations=40, extrusion=s.extrusion))
    g = d.full_grid(res.x)
    act = d.full_grid(np.ones(d.n_elem)) > 0
    des = act & ~(d.full_grid(d.preserve.astype(float)) > 0.5)
    for i in range(g.shape[0]):
        for j in range(g.shape[1]):
            col = g[i, j][des[i, j]]
            if col.size:
                assert np.ptp(col) < 1e-9
    # Several materials: one outcome each, tagged, best is the lightest.
    r = run_study(study_from_dict({**LUG, "material": ["pa12", "petg"], "maxIterations": 25}))
    names = [o.summary()["material"] for o in r.outcomes]
    assert names == ["pa12", "petg"], names
    assert r.best == int(np.argmin([o.report["mass"] for o in r.outcomes]))


def test_mma_matches_svanberg_toy_problem():
    # Svanberg's reference problem: minimize |x|^2 inside two spheres.
    from loadpath.mma import MMA
    opt = MMA(3, 2, xmin=0, xmax=5, move=1.0)
    x = np.array([4.0, 3.0, 2.0])
    for _ in range(40):
        f = np.array([((x - [5, 2, 1]) ** 2).sum() - 9, ((x - [3, 4, 3]) ** 2).sum() - 9])
        df = np.array([2 * (x - [5, 2, 1]), 2 * (x - [3, 4, 3])])
        xn = opt.update(x, (x ** 2).sum(), 2 * x, f, df)
        if np.abs(xn - x).max() < 1e-8:
            break
        x = xn
    assert np.allclose(x, [2.0175, 1.7800, 1.2375], atol=2e-4), x


def test_constrained_deflection_limit_is_met():
    # Minimum mass with a deflection limit: the optimizer's own measure must
    # land on the limit, from a solid start.
    from loadpath.constrained import Limits, optimize_constrained
    s = study_from_dict(LUG)
    d = build_domain(s)
    res = optimize_constrained(d, s.material.nu,
                               SimpOptions(volume_fraction=0.5, filter_radius=filter_radius(d, None),
                                           max_iterations=60),
                               Limits(None, 0.3, s.material.E))
    last = res.history[-1]
    assert abs(last["deflection.pull"] - 0.3) < 0.03, last
    assert res.volume_fraction < 0.5

def test_union_region_and_dead_material_pruning():
    u = shape_from_spec({"type": "union", "shapes": [
        {"type": "box", "min": [0, 0, 0], "max": [1, 1, 1]},
        {"type": "sphere", "center": [5, 0, 0], "r": 1}]})
    assert u.contains(np.array([[0.5, 0.5, 0.5], [5, 0, 0]])).all()
    assert not u.contains(np.array([[3, 0, 0]]))[0]
    assert abs(u.to_manifold().volume() - (1 + 4 / 3 * np.pi)) < 0.05

    # A cantilever with a strut hanging off its middle into free space: the
    # strut carries nothing and goes, all but a stub at its root (strained
    # by the beam's surface it is welded to); the cantilever stays whole.
    from loadpath.simp import _Analysis, _prune_dead, E_MIN, PENALTY
    spec = dict(BEAM, designSpace={"type": "box", "min": [0, 0, 0], "max": [100, 10, 30]})
    d = build_domain(study_from_dict(spec))
    c = d.mesh.node_xyz[d.mesh.enode].mean(axis=1)
    beam = c[:, 2] < 10
    strut = (np.abs(c[:, 0] - 50) < 3) & (c[:, 2] >= 10)
    xp = np.where(beam | strut, 1.0, 0.0)
    fe = _Analysis(d, 0.3, 1e-8)
    e = E_MIN + xp ** PENALTY * (1 - E_MIN)
    us = fe.solve(e, xp, [None] * len(d.loads), rtol=1e-8)
    out, _ = _prune_dead(fe, d, xp, us, d.loads, ~d.preserve, d.preserve)
    assert (out[beam] == 1.0).all()
    # (stress dies away over about one strut width from the root)
    assert (out[strut & (c[:, 2] > 18)] == 0.0).all()


def test_compliant_inverter():
    # Sigmund's force inverter, small: push right at the left middle, and the
    # right middle should move LEFT. A stiffness design moves it right.
    from loadpath.mechanism import MechanismOptions, optimize_mechanism
    spec = {
        "designSpace": {"type": "box", "min": [0, 0, 0], "max": [40, 20, 2]},
        "supports": [{"shape": {"type": "box", "min": [-1, -1, -1], "max": [2.1, 2.1, 3]}},
                     {"shape": {"type": "box", "min": [-1, 17.9, -1], "max": [2.1, 21, 3]}}],
        "loadCases": [
            {"name": "in", "loads": [{"shape": {"type": "box", "min": [-1, 8, -1], "max": [1.1, 12, 3]}, "force": [10, 0, 0]}]},
            {"name": "out", "loads": [{"shape": {"type": "box", "min": [38.9, 8, -1], "max": [41, 12, 3]}, "force": [-1, 0, 0]}]}],
        "material": {"E": 26, "nu": 0.45, "strength": 30, "density": 1.2},
        "goal": {"volumeFraction": 0.3}, "resolution": {"pitch": 2.0}}
    d = build_domain(study_from_dict(spec))
    r = optimize_mechanism(d, 0.45, MechanismOptions(volume_fraction=0.3, filter_radius=3.0, modulus=26,
                                                      k_in=26.0, k_out=1.3, extrusion=2, max_iterations=60))
    assert r.output_displacement > 0, r.output_displacement
    assert r.history[-1]["outputDisplacement"] > 2 * r.history[0]["outputDisplacement"]
    assert abs(r.xphys[~d.preserve].mean() - 0.3) < 0.02


def test_crosscheck_cantilever():
    # the independent tet check against beam theory; optional dependencies
    try:
        import gmsh  # noqa: F401
    except (ImportError, OSError):  # not installed, or its system libs (libGLU) are missing
        print("  (skipped: pip install loadpath[crosscheck])")
        return
    import manifold3d as m3d
    from loadpath.crosscheck import crosscheck
    from loadpath.shapes import trimesh_from_manifold
    spec = dict(BEAM, minMember=4)
    mesh = trimesh_from_manifold(m3d.Manifold.cube((100, 10, 10)).refine(20))
    r = crosscheck(study_from_dict(spec), mesh, size=2.5, contact_band=12)
    c = r["cases"][0]
    # tip deflection F L^3 / 3 E I = 2.0 mm (a clamped face reads a little stiff)
    assert 1.8 < c["deflection"] < 2.1, c["deflection"]
    # bending stress just outside the band: M c / I at x = 15 is 5.1 MPa
    assert abs(c["peakStress"] - 5.1) < 0.4, c["peakStress"]


def test_process_layers_weaken_across_build_direction():
    # A cantilever bent along x: printed with layers stacking along x (FDM,
    # buildDirection +x) its bending tension pulls the layers apart, so the
    # safety factor drops to the interlayer fraction; stacked along z it
    # does not. SLS sets no layer penalty and no overhang control.
    mesh = trimesh.creation.box(extents=[100, 10, 10])
    mesh.apply_translation([50, 5, 5])
    plain = verify_part(study_from_dict(BEAM), mesh, refine=0)["safetyFactor"]
    across = verify_part(study_from_dict(dict(BEAM, process={"type": "fdm", "buildDirection": "+x"})), mesh,
                         refine=0)["safetyFactor"]
    along = verify_part(study_from_dict(dict(BEAM, process={"type": "fdm", "buildDirection": "+z"})), mesh,
                        refine=0)["safetyFactor"]
    assert abs(across / plain - 0.6) < 0.05, (across, plain)
    assert along > 0.9 * plain, (along, plain)
    s = study_from_dict(dict(BEAM, process="fdm"))
    assert s.print_direction == (2, 1) and s.min_member >= 1.6
    assert study_from_dict(dict(BEAM, process="sls")).print_direction is None




def test_crease_metric_finds_unfilleted_junctions():
    """A post stuck on a plate has a sharp concave ring where they meet; the
    same post on the plate as preserved geometry is the user's, and a plate
    alone has none."""
    from loadpath.verify import _creases

    plate = trimesh.creation.box(extents=(40, 40, 4))
    post = trimesh.creation.cylinder(radius=5, height=20, sections=64)
    post.apply_translation((0, 0, 10))
    part = trimesh.boolean.union([plate, post], engine="manifold")
    st = study_from_dict({**BEAM, "designSpace": {"type": "box", "min": [-20, -20, -2], "max": [20, 20, 20]}})
    ring = _creases(st, part)
    assert abs(ring["creaseLength"] - 2 * np.pi * 5) < 1.5, ring["creaseLength"]
    assert np.allclose(ring["creaseAt"][0]["at"][2], 2.0, atol=0.2), ring["creaseAt"]
    assert _creases(st, plate)["creaseLength"] == 0.0
    st_p = study_from_dict({**BEAM, "designSpace": {"type": "box", "min": [-20, -20, -2], "max": [20, 20, 20]},
                            "preserve": [{"id": "plate", "shape": {"type": "box", "min": [-20, -20, -2], "max": [20, 20, 2]}},
                                         {"id": "post", "shape": {"type": "cylinder", "p0": [0, 0, 0], "p1": [0, 0, 20], "r": 5}}]})
    assert _creases(st_p, part)["creaseLength"] < 1.0


def test_closed_bore_is_flagged():
    """A bore the part closes over at one end is a blind hole no screw can be
    driven through; a through bore is fine."""
    from loadpath.verify import _closed_bores

    block = trimesh.creation.box(extents=(20, 20, 20))
    base = {**BEAM, "designSpace": {"type": "box", "min": [-10, -10, -10], "max": [10, 10, 10]}}
    for z1, closed in ((4.0, True), (12.0, False)):
        st = study_from_dict({**base, "keepOut": [
            {"id": "bore", "shape": {"type": "cylinder", "p0": [0, 0, -12], "p1": [0, 0, z1], "r": 2.5}}]})
        bore = trimesh.creation.cylinder(radius=2.5, segment=[[0, 0, -12], [0, 0, z1]])
        part = trimesh.boolean.difference([block, bore], engine="manifold")
        found = _closed_bores(st, part)
        assert bool(found) == closed, (z1, found)
        if closed:
            assert found[0]["end"] == "p1"


def test_lattice_infill_is_lighter_and_walled():
    """A gyroid infill under a skin: much lighter than solid, one body, its
    voids enclosed; open (no skin) leaves nothing enclosed; graded by the
    stress, it keeps more material than the uniform one."""
    from loadpath.lattice import LatticeOptions, lattice_infill, tpms

    # the distance approximation: a step along the gradient by d lands on
    # the surface, to within a few percent of d
    p = np.random.default_rng(0).uniform(0, 10, (2000, 3))
    d = tpms("gyroid", p, 10.0)
    near = np.abs(d) < 0.5
    eps = 1e-4
    g = np.stack([(tpms("gyroid", p + eps * e, 10.0) - d) / eps for e in np.eye(3)], axis=1)
    q = p - d[:, None] * g / np.linalg.norm(g, axis=1, keepdims=True)
    assert np.abs(tpms("gyroid", q[near], 10.0)).max() < 0.05
    st = study_from_dict({**BEAM, "designSpace": {"type": "box", "min": [0, 0, 0], "max": [40, 20, 20]},
                          "supports": [{"shape": {"type": "box", "min": [-1, -1, -1], "max": [0.01, 21, 21]}}],
                          "loadCases": [{"name": "tip", "loads": [
                              {"shape": {"type": "box", "min": [39.99, -1, -1], "max": [41, 21, 21]},
                               "force": [0, 0, -400]}]}]})
    block = trimesh.creation.box(extents=(40, 20, 20))
    block.apply_translation((20, 10, 10))
    o = LatticeOptions(cell=8.0, step=0.4)
    m, info = lattice_infill(st, block, None, o)
    assert 0.3 < info["volumeRatio"] < 0.7, info
    rep = verify_part(st, m, refine=0)
    assert rep["bodies"] == 1 and rep["enclosedCavityVolume"] > 1000, rep["enclosedCavityVolume"]
    o_open = LatticeOptions(cell=8.0, step=0.4, skin=0.0)
    m2, info2 = lattice_infill(st, block, None, o_open)
    assert info2["volumeRatio"] < info["volumeRatio"]
    assert verify_part(st, m2, refine=0)["enclosedCavityVolume"] < 1.0
    solid = verify_part(st, block, refine=0, fields=True)
    _, graded = lattice_infill(st, block, solid["fields"], o)
    assert graded["volumeRatio"] > info["volumeRatio"]

TESTS = [
    test_shapes_sdf_and_exact_volume,
    test_spec_errors_name_the_field,
    test_fem_matches_beam_theory,
    test_simp_keeps_volume_and_symmetry,
    test_reconstruct_keeps_bores_exact_and_preserve_whole,
    test_verify_bar_stress_matches_beam_theory,
    test_verify_hole_stress_concentration,
    test_am_filter_gradient_and_behavior,
    test_voxelize_volume,
    test_unreachable_load_is_refused_helpfully,
    test_run_study_end_to_end,
    test_local_mcp_job_functions,
    test_boundary_condition_types,
    test_gravity_matches_beam_theory,
    test_extrusion_and_materials,
    test_mma_matches_svanberg_toy_problem,
    test_constrained_deflection_limit_is_met,
    test_union_region_and_dead_material_pruning,
    test_compliant_inverter,
    test_crosscheck_cantilever,
    test_process_layers_weaken_across_build_direction,
    test_crease_metric_finds_unfilleted_junctions,
    test_closed_bore_is_flagged,
    test_lattice_infill_is_lighter_and_walled,
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
