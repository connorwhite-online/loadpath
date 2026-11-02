"""Anisotropic lattice homogenization (lattice_tensors.py) and the
anisotropic homogenized part check (homogenized.py)."""
import copy
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from loadpath import lattice_tensors as lt  # noqa: E402
from loadpath.fem import Layered, _rotate_c, elastic_c, hex8_ke, hex8_ke_batch  # noqa: E402


def _rot(seed):
    from scipy.spatial.transform import Rotation

    return Rotation.random(random_state=seed).as_matrix()


def test_rotations_agree_with_the_solver_and_keep_energy():
    """rotate_stiffness is fem's own rotation; von Mises is frame free; a
    stress surface rotated with its stiffness reads the same peak for the
    same strain, whichever frame it is written in."""
    r = _rot(3)
    c = Layered(0.3, 0.5, 0.4).c()
    assert np.allclose(lt.rotate_stiffness(c, r), _rotate_c(c, r), atol=1e-12)
    assert np.allclose(lt.rotate_stress_form(lt.VM_FORM, r), lt.VM_FORM, atol=1e-12)
    rng = np.random.default_rng(0)
    cl = rng.random((6, 6))
    cl = cl @ cl.T
    pl = rng.random((6, 6))
    pl = pl @ pl.T
    eps = rng.random(6)
    sig_g = lt.rotate_stiffness(cl, r) @ eps
    sig_l = cl @ (lt.strain_rotation(r) @ eps)
    assert np.allclose(lt.stress_rotation(r) @ sig_g, sig_l)
    assert np.isclose(sig_g @ lt.rotate_stress_form(pl, r) @ sig_g, sig_l @ pl @ sig_l)


def test_batched_elements_match_the_single_material_element():
    for nu in (0.3, Layered(0.35, 0.6, 0.5, (1, 1, 0))):
        c = elastic_c(1.0, nu)
        for el in ("im", "plain"):
            ke, _ = hex8_ke_batch(np.stack([c, 3 * c]), el)
            assert np.allclose(ke[0], hex8_ke(nu, el), atol=1e-13)
            assert np.allclose(ke[1], 3 * hex8_ke(nu, el), atol=1e-13)


def test_simple_cubic_cell_is_stiff_on_its_axes_and_soft_on_the_diagonal():
    """Analytic: thin axis rods carry a pull along their axis as the rod
    area does (E100 = rho/3 less the nodes), and bend off-axis, so E111 is
    a small fraction of it; C is symmetric, positive definite and cubic;
    the stress surface is on or above every sampled direction, and a
    hydrostatic load (zero von Mises) still stresses the walls."""
    from loadpath.struts import strut_diameter_for_density

    d = float(strut_diameter_for_density("cubic", 0.12, 1.0))
    r = lt.unit_cell_tensors("cubic", d, n=32)
    rho = r["rho"]
    C = r["C"]
    assert np.allclose(C, C.T) and np.linalg.eigvalsh(C).min() > 0
    assert np.isclose(C[0, 0], C[2, 2]) and np.isclose(C[3, 3], C[5, 5])
    assert 0.8 * rho / 3 < r["E100"] < 1.5 * rho / 3, r["E100"]
    assert r["E111"] < 0.25 * r["E100"]
    fit = np.sqrt(np.einsum("ka,ab,kb->k", r["directions"], r["P"], r["directions"]))
    assert (fit >= r["peaks"] * (1 - 1e-9)).all()
    hyd = np.array([1, 1, 1, 0, 0, 0]) / np.sqrt(3)
    assert np.sqrt(hyd @ r["P"] @ hyd) > 1.0


def test_tables_run_to_the_solid_and_match_the_scalar_tables():
    """The shipped tables: every sheet and strut cell; at rho 1 the solid
    (stiffness and von Mises); stiffness rises with density; the lesser of
    E100 and E111 is the scalar tables' conservative modulus."""
    from loadpath.lattice import SHEETS, sheet_modulus
    from loadpath.struts import STRUT_KINDS

    for kind in list(SHEETS) + list(STRUT_KINDS):
        assert lt.has_tensors(kind), kind
        C, P, Q, F = lt.tensors_at(kind, [1.0])
        assert np.allclose(C[0], elastic_c(1.0, 0.35), atol=1e-9)
        assert np.allclose(P[0], lt.VM_FORM, atol=1e-9) and F[0] == 1.0
        s = np.random.default_rng(0).normal(size=(50, 6))
        vm = np.sqrt(np.einsum("ka,ab,kb->k", s, lt.VM_FORM, s))
        assert np.allclose(lt.stress_surface(s, P[0], Q[0], F[0]), vm, rtol=1e-6)
        # inside the range the scalar fits were made over (sheets 0.14-0.6)
        rho = np.array([0.15, 0.2, 0.3, 0.4, 0.55])
        C, P, Q, F = lt.tensors_at(kind, rho)
        e100 = 1.0 / np.linalg.inv(C)[:, 0, 0]
        assert (np.diff(e100) > 0).all(), kind
        e111 = np.array([lt.directional_modulus(c, (1, 1, 1)) for c in C])
        emin = np.minimum(e100, e111)
        ref = sheet_modulus(kind, rho)
        # the scalar tables took BCCz/FCCz (tetragonal) as cubic: looser there
        tol = 0.3 if lt.symmetry_of(kind) == "tetragonal" else 0.2
        assert np.all(np.abs(emin / ref - 1) < tol), (kind, emin, ref)


# a 48 mm lattice-only cantilever between solid end blocks, 1 mm octet struts in 6 mm cells
def _bar(kind="octet", thickness=1.1, rotate=None):
    lat = {"op": "lattice", "type": kind, "cell": 6.0, "thickness": thickness, "origin": [0, -6, -6]}
    nodes = {"lat": lat}
    core = "lat"
    if rotate:
        nodes["turned"] = {"op": "transform", "input": "lat",
                           "rotate": {"axis": rotate[0], "degrees": rotate[1], "center": [24, 0, 0]}}
        core = "turned"
    nodes.update({
        "box": {"op": "shape", "shape": {"type": "box", "min": [0, -6, -6], "max": [48, 6, 6]}},
        "fill": {"op": "intersect", "inputs": ["box", core]},
        "ends": {"op": "shape", "shape": {"type": "union", "shapes": [
            {"type": "box", "min": [0, -6, -6], "max": [4, 6, 6]}, {"type": "box", "min": [44, -6, -6], "max": [48, 6, 6]}]}},
        "part": {"op": "union", "inputs": ["fill", "ends"]}})
    return {"name": "latbar", "nodes": nodes, "output": "part"}


_STUDY = {"name": "bar", "designSpace": [{"type": "box", "min": [0, -6, -6], "max": [48, 6, 6]}],
          "preserve": [{"id": "root", "shape": {"type": "box", "min": [0, -6, -6], "max": [4, 6, 6]}},
                       {"id": "tip", "shape": {"type": "box", "min": [44, -6, -6], "max": [48, 6, 6]}}],
          "supports": [{"region": "root"}],
          "loadCases": [{"name": "bend", "loads": [{"region": "tip", "force": [0, 0, -20]}]},
                        {"name": "pull", "loads": [{"region": "tip", "force": [100, 0, 0]}]}],
          "material": {"E": 1700.0, "nu": 0.35, "strength": 48.0, "density": 1.01}, "goal": {"safetyFactor": 2}}


def test_anisotropic_check_reads_the_cell_the_way_it_lies():
    """A BCCz cell with its z struts along the bar is several times stiffer
    than the scalar reading (its weakest direction everywhere) says;
    turning an octet 90 degrees about the bar changes nothing (cubic
    symmetry, the frame comes from the transform node), turning a simple
    cubic cell 45 degrees makes the bar far softer in a pull."""
    from loadpath.homogenized import homogenized_check
    from loadpath.spec import study_from_dict

    st = study_from_dict(_STUDY)
    zx = _bar("bccz", 1.1, ("y", 90))
    ani = homogenized_check(zx, st, cells=6000)
    iso = homogenized_check(zx, st, cells=6000, anisotropic=False)
    assert ani["latticeModel"] == "anisotropic" and iso["latticeModel"] == "isotropic"
    assert ani["mass"] == pytest.approx(iso["mass"])
    for a, i in zip(ani["cases"], iso["cases"]):
        assert a["compliance"] < 0.5 * i["compliance"] and a["peakStress"] < 0.5 * i["peakStress"]
    ani = homogenized_check(_bar(), st, cells=6000)
    turned = homogenized_check(_bar(rotate=("x", 90)), st, cells=6000)
    for a, t in zip(ani["cases"], turned["cases"]):
        assert t["compliance"] == pytest.approx(a["compliance"], rel=0.03)
    cub = homogenized_check(_bar("cubic", 1.3), st, cells=6000)
    cub45 = homogenized_check(_bar("cubic", 1.3, ("z", 45)), st, cells=6000)
    assert cub45["cases"][1]["compliance"] > 2.0 * cub["cases"][1]["compliance"]


def test_matrix_free_anisotropic_solve_matches_the_assembled_one(monkeypatch):
    """Large grids solve matrix-free under the isotropic multigrid; the
    answer is the assembled one's."""
    from loadpath.homogenized import homogenized_check
    from loadpath.spec import study_from_dict

    st = study_from_dict(_STUDY)
    doc = _bar(rotate=("z", 30))
    a = homogenized_check(copy.deepcopy(doc), st, cells=6000)
    monkeypatch.setenv("LOADPATH_SOLVER", "mg")
    m = homogenized_check(copy.deepcopy(doc), st, cells=6000)
    for x, y in zip(a["cases"], m["cases"]):
        assert y["compliance"] == pytest.approx(x["compliance"], rel=1e-4)
        assert y["peakStress"] == pytest.approx(x["peakStress"], rel=1e-3)
