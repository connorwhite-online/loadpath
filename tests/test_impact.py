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
from loadpath.modal import modal_analysis, transmissibility  # noqa: E402


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


def test_elastomers_use_the_buckling_law():
    from loadpath.impact import crush_law
    from loadpath.materials import material_from_spec

    law, ref = crush_law("gyroid", material_from_spec("tpu"))
    assert law == "gyroid:elastic" and ref == 26.0
    # Gibson & Ashby elastic collapse: 0.05 E rho^2
    assert approx(float(plateau_stress(law, 0.3, ref)), 0.05 * 26.0 * 0.09, rel=1e-9)
    assert crush_law("gyroid", material_from_spec("pa12")) == ("gyroid", 48.0)


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
