"""
Tests for crush and impact (impact.py) and modal analysis (modal.py): the
plateau law, the layer model's energy balance, grading a tapered core, the
solid-part spring, the 3D limit load of a bar, a cantilever's frequency
against beam theory, point masses and transmissibility. Under a minute.
Run: `python3 tests/test_impact.py`.
"""
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from loadpath.fem import HexMesh  # noqa: E402
from loadpath.impact import (CRUSH, CRUSH_DATA, crush_curve, density_for_plateau, face_load, grade_for_impact,  # noqa: E402
                             impact_response, limit_load, plateau_stress, solid_impact)
from loadpath.isolator import (density_for_stiffness, foot_stiffness, mount_modes, rigid_body,  # noqa: E402
                               tune_mount)
from loadpath.modal import base_transmissibility, modal_analysis, transmissibility  # noqa: E402


def approx(a, b, rel=1e-6, abs_=0.0):
    return abs(a - b) <= max(rel * abs(b), abs_)


def test_crush_fits_match_their_data():
    for kind, pts in CRUSH_DATA.items():
        for rho, s in pts:
            assert approx(float(plateau_stress(kind, rho, 1.0)), s, rel=0.06), (kind, rho)


def test_collapse_strength_of_a_cell():
    from loadpath.impact import collapse_strength
    from loadpath.lattice import wall_for_density

    r = collapse_strength("gyroid", float(wall_for_density("gyroid", 0.3, 1.0)), n=16, iters=6)
    # a coarse cell: collapse above first yield, near the n=32 fit
    assert r["collapse"] > 1.2 * r["firstYield"]
    assert approx(r["collapse"], float(plateau_stress("gyroid", r["rho"], 1.0)), rel=0.25)


def test_foam_stress_counts_pressure():
    from loadpath.impact import foam_stress

    uni = np.array([[0, 0, -5.0, 0, 0, 0]])
    hyd = np.array([[-5.0, -5.0, -5.0, 0, 0, 0]])
    assert approx(float(foam_stress(uni)[0]), 5.0, rel=1e-9)
    assert float(foam_stress(hyd)[0]) > 4.0          # von Mises would say 0
    assert approx(float(foam_stress(hyd, alpha=0.0)[0]), 0.0, abs_=1e-9)


def test_plateau_inverse_and_monotone():
    for kind in CRUSH:
        rho = np.array([0.12, 0.2, 0.3, 0.45])
        s = plateau_stress(kind, rho, 48.0)
        assert np.all(np.diff(s) > 0)
        assert np.allclose(density_for_plateau(kind, s, 48.0), rho)
        assert plateau_stress(kind, 1.0, 48.0) == 48.0


def test_prism_crushes_flat_and_energy_balances():
    # a uniform prism: one plateau, energy = F x crush (+ the elastic ramp)
    area = np.full(40, 400.0)
    cv = crush_curve("gyroid", area, 0.25, 0.5, 48.0, 1800.0)
    f = plateau_stress("gyroid", 0.25, 48.0) * 400.0
    assert np.allclose(cv.layer_force, f)
    # densification plus the elastic squeeze at the plateau force
    assert approx(cv.stroke, 20 * (1 - 1.4 * 0.25) + cv.d[1], rel=1e-6)
    r = impact_response(cv, 0.5 * cv.capacity, mass=1.0)
    assert not r["bottomedOut"]
    assert approx(r["peakForce"], f, rel=1e-4)
    assert r["efficiency"] > 0.9
    assert impact_response(cv, 2 * cv.capacity)["bottomedOut"]


def test_grading_flattens_a_tapered_core():
    h = 0.25
    z = (np.arange(80) + 0.5) * h
    area = (40 - 18 * z / 20) ** 2
    energy = 44_000.0
    g = grade_for_impact("gyroid", area, h, 48.0, 1800.0, energy, rho_min=0.2, rho_max=0.5)
    assert g["ok"]
    # the narrow top is denser than the wide foot
    assert g["rho"][-1] > g["rho"][0]
    uni = impact_response(crush_curve("gyroid", area, g["meanRho"], h, 48.0, 1800.0), energy)
    assert g["response"]["peakForce"] < 0.9 * uni["peakForce"]
    assert g["response"]["efficiency"] > uni["efficiency"]


def test_solid_impact_is_a_spring():
    r = solid_impact(10_000.0, 0.01, 48.0, 5000.0, force_limit=2000.0)
    assert approx(r["peakForce"], np.sqrt(2 * 5000 * 10_000))
    assert approx(r["energyAtForceLimit"], 2000 ** 2 / 2e4)
    assert r["yields"] is True     # 0.01 MPa/N x 10 kN = 100 MPa > 48


def test_field_graph_impact_and_vibration_blocks():
    from loadpath.fields import FieldGraph
    from loadpath.impact import impact_report
    from loadpath.materials import material_from_spec
    from loadpath.modal import vibration_report

    doc = {"nodes": {"pad": {"op": "shape", "shape": {"type": "box", "min": [0, 0, 0], "max": [30, 30, 15]}},
                     "rho": {"op": "constant", "value": 0.25},
                     "foot": {"op": "shape", "shape": {"type": "box", "min": [-1, -1, -1], "max": [31, 31, 0.5]}},
                     "top": {"op": "shape", "shape": {"type": "box", "min": [-1, -1, 14.5], "max": [31, 31, 16]}}},
           "material": "pa12"}
    g = FieldGraph(doc)
    mat = material_from_spec("pa12")
    r = impact_report(g, {"envelope": "pad", "density": "rho", "energy": 10.0, "forceLimit": 20_000,
                          "pitch": 0.5, "grade": True}, mat)
    f = float(plateau_stress("gyroid", 0.25, mat.strength)) * 900.0
    assert approx(r["response"]["peakForce"], f, rel=0.01)
    assert r["ok"]
    # a prism grades to (nearly) uniform
    assert np.ptp(r["graded"]["density"]["to"]) < 0.02
    v = vibration_report(g, {"envelope": "pad", "density": "rho", "fixed": "foot", "modes": 2, "pitch": 1.5,
                             "pointMasses": [{"region": "top", "mass": 1.0}], "excitation": 50}, mat)
    assert 0 < v["frequencies"][0] <= v["frequencies"][1] * (1 + 1e-6)   # square pad: x and y sway pair
    assert v["dampingRatio"] == 0.02


def test_run_field_writes_the_graded_graph():
    import json
    import tempfile

    from loadpath.fields import run_field

    doc = {"name": "pad", "material": "pa12", "check": False,
           "nodes": {"pad": {"op": "expr", "expr": "max(max(abs(x), abs(y)) - (15 - 0.4 * z), max(-z, z - 15))",
                             "bounds": {"min": [-15, -15, 0], "max": [15, 15, 15]}},
                     "rho": {"op": "constant", "value": 0.25},
                     "core": {"op": "lattice", "type": "gyroid", "cell": 10, "density": "rho"},
                     "part": {"op": "intersect", "inputs": ["core", "pad"]}},
           "output": "part", "mesh": {"pitch": 0.8},
           "impact": {"envelope": "pad", "density": "rho", "energy": 15.0, "pitch": 0.5, "grade": True}}
    with tempfile.TemporaryDirectory() as d:
        _, rep = run_field(doc, d, render=False)
        imp = rep["impact"]
        assert len(imp["uniformSameMass"]["curve"]["force"]) > 10
        with open(imp["gradedGraph"]) as fh:
            gd = json.load(fh)
    assert gd["name"] == "pad-graded" and gd["impact"]["grade"] is False
    assert gd["nodes"]["rho"] == imp["graded"]["density"]
    # a taper grades denser where it is narrow
    assert gd["nodes"]["rho"]["to"][-1] > gd["nodes"]["rho"]["to"][0]


def test_elastomers_use_the_buckling_law():
    from loadpath.impact import crush_law
    from loadpath.materials import material_from_spec

    law, ref = crush_law("gyroid", material_from_spec("tpu"))
    assert law == "gyroid:elastic" and ref == material_from_spec("tpu").E
    # Gibson & Ashby elastic collapse: 0.05 E rho^2
    assert approx(float(plateau_stress(law, 0.3, ref)), 0.05 * ref * 0.09, rel=1e-9)
    assert crush_law("gyroid", material_from_spec("pa12")) == ("gyroid", material_from_spec("pa12").strength)


def test_grade_3d_moves_load_off_the_idle_rim():
    from loadpath.impact import grade_3d, limit_load
    from loadpath.lattice import sheet_modulus
    from loadpath.materials import material_from_spec

    mat = material_from_spec("pa12")
    h, n, nz = 2.0, 14, 8
    c = (np.arange(n) + 0.5) * h - n * h / 2
    zc = (np.arange(nz) + 0.5) * h
    X, Y, Z = np.meshgrid(c, c, zc, indexing="ij")
    w = 14.0 - 6.0 * Z / (nz * h)                 # 28 mm foot to 16 mm top
    inside = (np.abs(X) < w) & (np.abs(Y) < w)
    lo = np.array([-n * h / 2, -n * h / 2, 0.0])
    area = inside.sum(axis=(0, 1)) * h * h
    g = grade_for_impact("gyroid", area, h, 48.0, 1800.0, 20_000.0, rho_min=0.15, rho_max=0.5)
    rho_z = np.broadcast_to(g["rho"], inside.shape)
    r = grade_3d(inside, lo, h, "gyroid", mat, g["force"], rho0=rho_z, rho_min=0.15, iters=10)

    def onset(rho, keep):
        m = HexMesh(np.argwhere(keep), h, lo)
        z = m.node_xyz[:, 2]
        fixed = np.zeros(m.n_dof, bool)
        bot = np.flatnonzero(z < 1e-9)
        fixed[3 * bot + 2] = True
        fixed[3 * bot[0]] = fixed[3 * bot[0] + 1] = fixed[3 * bot[-1] + 1] = True
        rr = rho[keep]
        return limit_load(m, mat.E * sheet_modulus("gyroid", rr), plateau_stress("gyroid", rr, 48.0), mat.nu,
                          fixed, face_load(m, 2, -1.0), iters=6)["loadFactor"]

    keep3 = inside & (r["rho"] > 0)
    assert keep3.sum() < inside.sum()             # some idle rim emptied
    assert onset(r["rho"], keep3) > 1.05 * onset(np.asarray(rho_z, float), inside)


def _bar(n=(4, 4, 12), p=1.0):
    ijk = np.argwhere(np.ones(n, bool))
    return HexMesh(ijk, p, np.zeros(3))


def test_limit_load_of_a_bar_is_area_times_strength():
    m = _bar()
    z = m.node_xyz[:, 2]
    fixed = np.zeros(m.n_dof, bool)
    bot = np.flatnonzero(z < 1e-9)
    fixed[3 * bot + 2] = True
    fixed[3 * bot[0]] = fixed[3 * bot[0] + 1] = fixed[3 * bot[-1] + 1] = True
    f = face_load(m, 2, -1.0)
    r = limit_load(m, 1000.0, 10.0, 0.3, fixed, f, alpha=0.0)
    # 16 mm^2 at 10 MPa: 160 N, from below (a lower bound)
    assert 0.95 * 160.0 <= r["loadFactor"] <= 160.0 * 1.001
    assert r["loadFactor"] >= r["firstYieldFactor"]
    assert 160.0 * 0.999 <= r["upperBound"] <= 160.0 * 1.1


def test_cantilever_frequency_matches_beam_theory():
    L, b, p = 100.0, 10.0, 1.0
    m = HexMesh(np.argwhere(np.ones((100, 10, 10), bool)), p, np.zeros(3))
    fixed = np.zeros(m.n_dof, bool)
    root = np.flatnonzero(m.node_xyz[:, 0] < 1e-9)
    for a in range(3):
        fixed[3 * root + a] = True
    r = modal_analysis(m, 1800.0, 1.01, 0.39, fixed, n_modes=3)
    f1 = 1.8751 ** 2 / (2 * np.pi) * np.sqrt(1800.0 * b ** 4 / 12 / (1.01e-9 * b * b * L ** 4))
    assert approx(r.frequencies[0], f1, rel=0.02)
    assert approx(r.mass, 0.0101, rel=1e-6)
    # first bending pair moves ~61% of the mass (beam theory 0.613)
    assert approx(r.participation[0].sum(), 0.613, abs_=0.03)


def test_point_mass_lowers_frequency():
    m = _bar((4, 4, 20))
    fixed = np.zeros(m.n_dof, bool)
    z = m.node_xyz[:, 2]
    bot = np.flatnonzero(z < 1e-9)
    for a in range(3):
        fixed[3 * bot + a] = True
    top = np.flatnonzero(z > z.max() - 1e-9)
    f0 = modal_analysis(m, 1800.0, 1.0, 0.35, fixed, 2).frequencies[0]
    f1 = modal_analysis(m, 1800.0, 1.0, 0.35, fixed, 2, point_masses=[(top, 0.05)]).frequencies[0]
    assert f1 < 0.5 * f0


def test_transmissibility():
    assert approx(transmissibility(100.0, 100.0, 0.04), 1 / 0.04, rel=1e-3)
    assert transmissibility(300.0, 100.0, 0.04) < 0.2


def test_foot_stiffness_inverts():
    kz, ks = foot_stiffness("gyroid", 0.3, 26.0, 24.0, 30.0)
    assert 0 < ks < kz
    assert approx(float(density_for_stiffness("gyroid", kz, 26.0, 24.0, 30.0)), 0.3, rel=1e-9)


def test_mount_bounce_and_elastic_centre():
    feet = np.array([[50, 30, 0], [50, -30, 0], [-50, 30, 0], [-50, -30, 0]], float)
    m, cg, inertia = rigid_body([[-40, -40, 60], [40, 40, 60], [40, -40, 60], [-40, 40, 60]], [0.75] * 4)
    f, _, mf = mount_modes(m, cg, inertia, feet, 25.0, 8.0)
    i = int(np.argmax(mf[:, 2]))
    assert approx(float(f[i]), np.sqrt(4 * 25e3 / 3.0) / (2 * np.pi), rel=1e-6) and mf[i, 2] > 0.999
    # off-centre CG: equal feet couple bounce with rocking, tuned feet do not
    m, cg, inertia = rigid_body([[-28, -36, 60], [52, 44, 60], [52, -36, 60], [-28, 44, 60]], [0.75] * 4)
    _, _, mf_eq = mount_modes(m, cg, inertia, feet, 25.0, 8.0)
    k, resid = tune_mount(cg[:2], feet, 100.0)
    assert resid < 1e-3 and approx(float(k.sum()), 100.0, rel=1e-4)
    assert approx(float((k * feet[:, 0]).sum() / k.sum()), cg[0], abs_=1e-3)
    f, _, mf = mount_modes(m, cg, inertia, feet, k, 8.0)
    assert mf[:, 2].max() > 0.999 and mf_eq[:, 2].max() < 0.99
    assert approx(float(f[int(np.argmax(mf[:, 2]))]), np.sqrt(100e3 / 3.0) / (2 * np.pi), rel=1e-6)
    # bounds bind: the leftover offset is reported
    k, resid = tune_mount(cg[:2], feet, 100.0, 20.0, 30.0)
    assert k.min() >= 20.0 - 1e-9 and k.max() <= 30.0 + 1e-9 and resid > 1.0


def test_base_transmissibility_of_a_mass_on_a_soft_post():
    # a soft 4x4x10 post, clamped at the bottom, with 1 kg on top: one axial mode
    ijk = np.argwhere(np.ones((4, 4, 10), bool))
    mesh = HexMesh(ijk, 2.0, np.zeros(3))
    z = mesh.node_xyz[:, 2]
    fixed = np.zeros(mesh.n_dof, bool)
    bot = np.flatnonzero(z < 1e-6)
    for a in range(3):
        fixed[3 * bot + a] = True
    top = np.flatnonzero(z > 20 - 1e-6)
    res = modal_analysis(mesh, 10.0, 0.01, 0.3, fixed, 12, [(top, 1.0)])
    fz = float(res.frequencies[int(np.argmax(res.participation[:, 2]))])
    eta = 0.1
    t = base_transmissibility(res, 3 * top + 2, 2, [0.1 * fz, fz, 3 * fz], eta, fixed)
    assert approx(float(t[0]), 1.0, rel=0.02)
    assert approx(float(t[1]), float(transmissibility(fz, fz, eta)), rel=0.1)     # the post itself is 7% of the mode
    assert approx(float(t[2]), float(transmissibility(3 * fz, fz, eta)), rel=0.1)


def test_viscoelastic_factors():
    from loadpath.isolator import shock_margin
    from loadpath.materials import material_from_spec
    from loadpath.viscoelastic import YEAR, creep_factor, dynamic_factor, event_factor

    tpu, pa12 = material_from_spec("tpu"), material_from_spec("pa12")
    assert approx(float(creep_factor(tpu, 10.0)), 1.0)
    # inside the 1.3-1.6 dynamic/static band quoted for PU isolators
    assert 1.3 < float(dynamic_factor(tpu, 35.0)) < 1.6
    assert 1.5 < float(creep_factor(tpu, 10 * YEAR)) < 2.0
    assert float(event_factor(tpu, 0.01)) > 1.3 and float(dynamic_factor(pa12, 35.0)) == 1.0
    # creep eats the strain budget for shock; a fast pulse adds some back
    base = shock_margin("gyroid", [0.25, 0.3], 26.0, 24.0, [10.0, 10.0])
    assert shock_margin("gyroid", [0.25, 0.3], 26.0, 24.0, [10.0, 10.0], creep=1.7) < base - 0.69
    assert approx(shock_margin("gyroid", [0.25, 0.3], 26.0, 24.0, [10.0, 10.0], rate=1.4), 1.4 * base)


def _cap_grid(R=20.0, L=30.0, h=0.5):
    """A half cylinder (axis along y, round side up) on a grid: inside, its field, the grid corner."""
    lo = np.array([-R - 1.0, 0.0, 0.0])
    n = (int((2 * R + 2) / h), int(L / h), int((R + 1) / h))
    x, _, z = np.meshgrid(*[lo[i] + (np.arange(n[i]) + 0.5) * h for i in range(3)], indexing="ij")
    phi = np.sqrt(x * x + z * z) - R
    return phi < 0, phi, lo


def test_round_cap_against_a_wall_grows_with_the_contact_patch():
    # a half-cylinder tyre meets a flat wall on a line: F = sigma_pl x chord x length,
    # chord = 2 sqrt(2 R d - d^2), while nothing has locked
    from loadpath.impact import contact_columns, contact_curve

    R, L, h, rho, sy, E = 20.0, 30.0, 0.5, 0.2, 40.0, 1700.0
    inside, phi, lo = _cap_grid(R, L, h)
    cols = contact_columns(inside, rho, h, 2, "+", None, lo, field=phi)
    cv = contact_curve("gyroid", cols, sy, 1e3 * E)            # stiff walls: every touched column on its plateau
    soft = contact_curve("gyroid", cols, sy, E)
    sp = float(plateau_stress("gyroid", rho, sy))
    for d in (1.0, 2.0, 4.0):
        want = sp * 2.0 * np.sqrt(2 * R * d - d * d) * L
        assert approx(float(np.interp(d, cv.d, cv.f)), want, rel=0.05), d
        assert approx(float(np.interp(d, cv.d, cv.layer_area)), want / sp, rel=0.05), d
        # with the real modulus the columns just touched are still on their elastic ramp
        assert 0.75 * want < float(np.interp(d, soft.d, soft.f)) < float(np.interp(d, cv.d, cv.f))


def test_ball_and_pole_on_a_flat_pad():
    from loadpath.impact import contact_columns, contact_curve

    h, rho, sy, E, R = 0.5, 0.2, 40.0, 1700.0, 25.0
    inside = np.ones((80, 80, 40), bool)                       # 40 x 40 x 20 mm pad
    sp = float(plateau_stress("gyroid", rho, sy))
    E = 1e3 * E                                                # stiff walls: plateau on all that is touched
    ball = contact_curve("gyroid", contact_columns(inside, rho, h, 2, "+", {"shape": "sphere", "radius": R}), sy, E)
    pole = contact_curve("gyroid", contact_columns(inside, rho, h, 2, "+",
                                                   {"shape": "cylinder", "radius": R, "along": "y"}), sy, E)
    for d in (1.0, 2.0, 3.0):
        assert approx(float(np.interp(d, ball.d, ball.f)), sp * np.pi * (2 * R * d - d * d), rel=0.06), d
        assert approx(float(np.interp(d, pole.d, pole.f)), sp * 2 * np.sqrt(2 * R * d - d * d) * 40.0, rel=0.06), d
    # a flat striker on the same pad is the layer model
    flat = contact_curve("gyroid", contact_columns(inside, rho, h, 2, "+", None), sy, E)
    lay = crush_curve("gyroid", np.full(40, 1600.0), rho, h, sy, E)
    e = 0.5 * lay.energy(lay.stroke)
    a, b = impact_response(flat, e), impact_response(lay, e)
    assert approx(a["peakForce"], b["peakForce"], rel=0.01) and approx(a["crush"], b["crush"], rel=0.02), (a, b)


def test_field_graph_striker_block():
    from loadpath.fields import FieldGraph
    from loadpath.impact import impact_report
    from loadpath.materials import material_from_spec

    # a domed pad (a ball's cap on a block) dropped dome first onto a floor
    doc = {"nodes": {"pad": {"op": "expr", "expr": "max(sqrt(x*x + y*y + (z - 0)*(z - 0)) - 20, -z)",
                             "bounds": {"min": [-21, -21, 0], "max": [21, 21, 21]}},
                     "rho": {"op": "constant", "value": 0.25}},
           "material": "pa12"}
    g = FieldGraph(doc)
    mat = material_from_spec("pa12")
    spec = {"envelope": "pad", "density": "rho", "energy": 3.0, "pitch": 0.5, "grade": True}
    lay = impact_report(g, spec, mat)
    r = impact_report(g, {**spec, "striker": {"side": "+"}}, mat)
    c = r["contact"]
    assert c["model"] == "columns" and c["striker"] == "flat"
    # the force is the plateau on the patch the floor has reached: pi (2 R d - d^2)
    d = r["response"]["crush"]
    sp = float(plateau_stress("gyroid", 0.25, mat.strength))
    assert approx(c["area"], np.pi * (2 * 20 * d - d * d), rel=0.08), c
    # ... less the rim of the patch, still on its elastic ramp
    assert 0.75 * sp * c["area"] < r["response"]["peakForce"] <= sp * c["area"] * 1.001
    # the layer model crushed the dome's tip ahead of the floor and stopped the hit in less travel
    assert lay["response"]["crush"] < d
    # the grade found is no worse than the same mass spread evenly, and comes back as a table
    gr = r["graded"]
    assert gr["response"]["peakForce"] <= r["uniformSameMass"]["response"]["peakForce"] * 1.001
    assert gr["density"]["op"] == "remap" and len(gr["density"]["from"]) == len(gr["density"]["to"])
    try:
        impact_report(g, {**spec, "striker": {"shape": "sphere"}}, mat)
        raise AssertionError("a ball needs a radius")
    except ValueError:
        pass


TESTS = [v for k, v in dict(globals()).items() if k.startswith("test_")]


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


def test_impact_response_on_a_linear_ramp_is_exact():
    # a spring of k = 1000 N/mm taking 500 N·mm stops at 1 mm and 1000 N;
    # straight-line interpolation of energy between the curve's corners
    # read a stiff core like this about 4x soft
    import numpy as np
    from loadpath.impact import CrushCurve, impact_response
    d = np.array([0.0, 10.0])
    c = CrushCurve(d=d, f=1000.0 * d, stroke=10.0, layer_force=np.zeros(1), layer_area=np.ones(1),
                   layer_rho=np.ones(1), layer_stroke=np.ones(1))
    r = impact_response(c, 500.0)
    assert abs(r["crush"] - 1.0) < 1e-3 and abs(r["peakForce"] - 1000.0) < 1.0, r
