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

    # (Delaunay and custom cells have no shipped tables: scalar fits only)
    shipped = [k for k in STRUT_KINDS if k != "delaunay" and not k.startswith("custom-")]
    for kind in list(SHEETS) + shipped:
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


def test_orient_turns_the_cells_and_not_their_fields():
    """`orient` on a lattice node: the field at P is the unturned lattice's
    at R^T P, for strut and TPMS cells, the axis/degrees form included; the
    homogenized check reads the turned frame (a cubic cell turned 45 degrees
    by orient reads as turned by a transform node)."""
    from scipy.spatial.transform import Rotation

    from loadpath.fields import FieldGraph
    from loadpath.homogenized import homogenized_check
    from loadpath.spec import study_from_dict

    P = np.random.default_rng(0).uniform(0, 12, (40, 3))
    R = Rotation.from_rotvec(np.radians([10.0, -20.0, 30.0])).as_matrix()
    for kind in ("octet", "gyroid"):
        plain = FieldGraph({"nodes": {"lat": {"op": "lattice", "type": kind, "cell": 6.0, "thickness": 1.0}},
                            "output": "lat"})
        turned = FieldGraph({"nodes": {"lat": {"op": "lattice", "type": kind, "cell": 6.0, "thickness": 1.0,
                                               "orient": [10.0, -20.0, 30.0]}}, "output": "lat"})
        assert np.allclose(turned.evaluate(P), plain.evaluate(P @ R), atol=1e-9)
    ax = FieldGraph({"nodes": {"lat": {"op": "lattice", "type": "octet", "cell": 6.0, "thickness": 1.0,
                                       "orient": {"axis": "z", "degrees": 30}}}, "output": "lat"})
    vec = FieldGraph({"nodes": {"lat": {"op": "lattice", "type": "octet", "cell": 6.0, "thickness": 1.0,
                                        "orient": [0, 0, 30]}}, "output": "lat"})
    assert np.allclose(ax.evaluate(P), vec.evaluate(P))
    st = study_from_dict(_STUDY)
    by_node = homogenized_check(_bar("cubic", 1.3, ("z", 45)), st, cells=6000)
    doc = _bar("cubic", 1.3)
    doc["nodes"]["lat"]["orient"] = [0, 0, 45]
    by_orient = homogenized_check(doc, st, cells=6000)
    for a, b in zip(by_node["cases"], by_orient["cases"]):
        assert b["compliance"] == pytest.approx(a["compliance"], rel=0.05)
    # a skinless bar: the turned cells are cut open at its faces, which the
    # check flags (homogenization reads those too stiff); laid square, none
    assert by_orient["latticeTurnedExposed"] > 0.1 and by_orient["warnings"]
    plain = homogenized_check(_bar("cubic", 1.3), st, cells=6000)
    assert plain["latticeTurnedExposed"] == 0.0 and not plain.get("warnings")


def test_orientation_screen_finds_the_stiff_axis():
    """A BCCz bar with its stiff z struts across the bar: the screen (macro
    stresses held) ranks the turn that lays z along the bar far above the
    cell as it lies; the full optimizer, a few analyses only, turns it and
    the bar gets several times stiffer (and stronger) at the same wall."""
    from loadpath.lattice_orient import optimize_orientation
    from loadpath.spec import study_from_dict

    st = study_from_dict(_STUDY)
    r = optimize_orientation(_bar("bccz", 1.1), st, cells=5000, full_checks=2, refine=0, confirm="none")
    assert r["guard"] is None
    assert abs(r["dominantStress"][0]) > 0.95                  # the bar's axis
    # (the bias-cut faces of a skinless bar cost the turned cells strength:
    # FACE_CUT; the stiffness gain stands)
    assert r["best"]["safetyFactor"] > 1.05 * r["unrotated"]["safetyFactor"]
    assert r["best"]["compliance"] < 0.4 * r["unrotated"]["compliance"]
    R = np.asarray(_rotation(r["orient"]))
    assert abs(R[0, 2]) > 0.85                                  # the cell's z on the bar's x
    assert r["doc"]["nodes"]["lat"]["orient"] == [round(x, 3) for x in r["orient"]]


def test_orientation_keeps_bare_faced_cells_as_laid(monkeypatch):
    """A skinless bar is cut open at its faces, where the homogenized check
    misreads turned cells against square-cut ones: by default the optimizer
    keeps the cells as laid and says why, the turn it found kept as the
    candidate. With confirm="resolved" a full-resolution check decides: a
    turn that comes out weaker (here faked) is refused, a stronger one kept."""
    import loadpath.resolved as resolved
    from loadpath.lattice_orient import optimize_orientation
    from loadpath.spec import study_from_dict

    st = study_from_dict(_STUDY)
    r = optimize_orientation(_bar("bccz", 1.1), st, cells=5000, full_checks=2, refine=0)
    assert r["guard"] and r["guard"]["confirmed"] is None and "bare faces" in r["guard"]["reason"]
    assert r["orient"] == [0.0, 0.0, 0.0] and not any(r["doc"]["nodes"]["lat"].get("orient") or [])
    assert r["best"] == r["unrotated"]
    assert r["candidate"]["safetyFactor"] > r["unrotated"]["safetyFactor"]

    for turned_peak, kept in ((300.0, False), (50.0, True)):
        def fake(doc, study, **kw):
            o = doc["nodes"]["lat"].get("orient")
            p = turned_peak if o and any(o) else 100.0
            return {"safetyFactor": 48.0 / p, "cases": [{"peakStress": p, "compliance": 1.0}], "mass": 1.0,
                    "cells": 1, "pitch": 0.2, "seconds": 0.0}
        monkeypatch.setattr(resolved, "resolved_check", fake)
        r2 = optimize_orientation(_bar("bccz", 1.1), st, cells=5000, full_checks=2, refine=0, confirm="resolved")
        assert r2["guard"]["confirmed"] is kept
        assert r2["guard"]["ratio"] == pytest.approx(turned_peak / 100.0)
        assert (r2["candidate"] is None) is kept
        assert any(r2["orient"]) is kept


def test_resolved_check_of_a_small_lattice_bar():
    """resolved.resolved_check voxelizes the struts and solves them as they
    are: mass matches the homogenized check's to a few percent and, under a
    skin (the regime the tensors are validated in), so does the stiffness."""
    from loadpath.homogenized import homogenized_check
    from loadpath.resolved import default_pitch, resolved_check
    from loadpath.spec import study_from_dict

    doc = _bar("cubic", 1.6)
    doc["nodes"]["skin"] = {"op": "shell", "input": "box", "thickness": 1.0}
    doc["nodes"]["part"]["inputs"].append("skin")
    sd = copy.deepcopy(_STUDY)
    sd["loadCases"] = [c for c in sd["loadCases"] if c["name"] == "bend"] or sd["loadCases"][:1]
    st = study_from_dict(sd)
    assert default_pitch(doc) == pytest.approx(1.6 / 6)
    r = resolved_check(doc, st, pitch=0.4)
    hom = homogenized_check(doc, st, cells=6000)
    assert r["mass"] == pytest.approx(hom["mass"], rel=0.08)
    ratio = hom["cases"][0]["compliance"] / r["cases"][0]["compliance"]
    assert 0.7 < ratio < 1.4
    assert r["safetyFactor"] > 0 and r["cells"] > 1000


def _rotation(deg):
    from scipy.spatial.transform import Rotation

    return Rotation.from_rotvec(np.radians(deg)).as_matrix()


def test_zoned_orient_follows_each_zone_and_its_frames():
    """orient.zones: away from where zones meet, the lattice is that zone's
    turned lattice; the weights sum to one and cross-fade over the blend;
    the homogenized frames are each zone's rotation."""
    from loadpath.fields import FieldGraph, orient_weights
    from loadpath.homogenized import lattice_frames

    lat = {"op": "lattice", "type": "octet", "cell": 6.0, "thickness": 1.0,
           "orient": {"zones": [{"at": [-20, 0, 0], "rotate": [0, 0, 0]}, {"at": [20, 0, 0], "rotate": [0, 0, 45]}],
                      "blend": 6.0}}
    g = FieldGraph({"nodes": {"lat": lat}, "output": "lat"})
    P = np.random.default_rng(1).uniform(-12, 12, (3000, 3))
    a = FieldGraph({"nodes": {"lat": {**lat, "orient": [0, 0, 0]}}, "output": "lat"}).evaluate(P)
    b = FieldGraph({"nodes": {"lat": {**lat, "orient": [0, 0, 45]}}, "output": "lat"}).evaluate(P)
    f = g.evaluate(P)
    left, right = P[:, 0] < -6, P[:, 0] > 6
    assert np.abs(f[left] - a[left]).max() < 1e-3 and np.abs(f[right] - b[right]).max() < 1e-3
    w = orient_weights(g.nodes["lat"], P)
    assert np.allclose(w.sum(axis=1), 1.0)
    mid = np.abs(P[:, 0]) < 0.5
    assert np.all(np.abs(w[mid, 0] - 0.5) < 0.1)
    fr = lattice_frames(g, "lat", np.array([[-10.0, 0, 0], [10.0, 0, 0]]), 1.0)
    assert np.allclose(fr[0], np.eye(3), atol=1e-6)
    assert np.allclose(fr[1], _rotation([0, 0, 45]), atol=1e-6)


def test_zone_seam_plates_are_solid_and_read_as_skin():
    """orient.seam: a solid plate on the face between zones (the bisector
    plane of their points), a hard split either side of it; the homogenized
    check counts the plate as solid (skin), not lattice."""
    from loadpath.fields import FieldGraph, orient_weights
    from loadpath.homogenized import _EMPTY, _variant

    lat = {"op": "lattice", "type": "cubic", "cell": 6.0, "thickness": 0.8,
           "orient": {"zones": [{"at": [-20, 0, 0], "rotate": [0, 0, 0]}, {"at": [20, 0, 0], "rotate": [45, 0, 0]}],
                      "seam": 1.0}}
    doc = {"nodes": {"lat": lat}, "output": "lat"}
    g = FieldGraph(doc)
    P = np.random.default_rng(2).uniform(-12, 12, (4000, 3))
    f = g.evaluate(P)
    on = np.abs(P[:, 0]) < 0.45
    assert np.all(f[on] < 0) and np.all(f[np.abs(P[:, 0]) < 0.5] <= np.abs(P[np.abs(P[:, 0]) < 0.5, 0]) - 0.5 + 1e-9)
    w = orient_weights(g.nodes["lat"], P)
    assert set(np.unique(w)) <= {0.0, 1.0}
    b = FieldGraph({"nodes": {"lat": {**lat, "orient": [45, 0, 0]}}, "output": "lat"}).evaluate(P)
    right = P[:, 0] > 0
    assert np.abs(f[right] - np.minimum(b, np.abs(P[:, 0]) - 0.5)[right]).max() < 1e-9
    plates = _variant(g, doc, None, "lat", _EMPTY).evaluate(P)
    assert np.allclose(plates, np.abs(P[:, 0]) - 0.5)


def test_zones_turn_each_arm_of_an_l_bracket_its_own_way():
    """BCCz is stiff along z only: an L-bracket's arm along x wants z on x,
    its arm up z wants it left alone; zones get the bracket stronger than
    the cells as laid."""
    sys.path.insert(0, str(ROOT / "bench"))
    from lattice_zones_mass import case

    from loadpath.lattice_orient import optimize_zones
    from loadpath.spec import study_from_dict

    doc, sd, _ = case("l-bracket")
    r = optimize_zones(doc, study_from_dict(sd), zones=2, cells=5000, rounds=1, alternatives=1, confirm="none")
    assert r["best"]["safetyFactor"] > 1.5 * r["unrotated"]["safetyFactor"]
    zones = r["doc"]["nodes"]["lat"]["orient"]["zones"]
    arm_x = min(zones, key=lambda z: z["at"][2])           # the lower zone: the arm along x
    assert abs(_rotation(arm_x["rotate"])[0, 2]) > 0.7      # its cells' z laid along x


def test_graded_lattice_meets_the_target_lighter():
    """Fully stressed grading of a BCCz L-bracket's wall: at the safety
    factor the uniform wall gives, the graded lattice is much lighter, its
    density varies, and the density lands in a grid node of the graph."""
    sys.path.insert(0, str(ROOT / "bench"))
    from lattice_zones_mass import case

    from loadpath.homogenized import homogenized_check
    from loadpath.lattice_grade import SOURCE, grade_lattice
    from loadpath.spec import study_from_dict

    doc, sd, _ = case("l-bracket")
    st = study_from_dict(sd)
    ref = homogenized_check(doc, st, cells=5000)
    r = grade_lattice(doc, st, target_sf=ref["safetyFactor"], cells=5000, iterations=6)
    b = r["best"]
    assert b is not None and b["safetyFactor"] >= 0.99 * ref["safetyFactor"]
    assert b["mass"] < 0.7 * ref["mass"]
    lo, hi = b["densityRange"]
    assert hi > 2 * lo
    node = r["doc"]["nodes"][r["densityNode"]]
    assert node == {"op": "grid", "source": SOURCE}
    assert r["doc"]["nodes"][r["lattice"]]["density"] == r["densityNode"]
    assert "thickness" not in r["doc"]["nodes"][r["lattice"]]


def test_face_cut_knockdown_follows_the_cut():
    """The bare-face knockdown follows how the face cuts the cells: a box's
    faces cut cells laid square read alignment 1 (the square knockdown,
    corners and edges included), cells turned 45 degrees about z read
    1/sqrt(2) on the x and y faces; bending cells lose far more cut on the
    bias than an octet does. An octet's turn is trusted (FACE_CUT_VALIDATED),
    a cubic's is guarded."""
    from loadpath.homogenized import FACE_CUT, face_alignment, face_cut_knockdown
    from loadpath.lattice_orient import _guard

    box = np.array([[-6.0, -6, -6], [6, 6, 6]])

    def field(P):
        q = np.abs(P) - box[1]
        return np.maximum(q.max(axis=1), -1e9)

    g = np.stack(np.meshgrid(*[np.arange(12)] * 3, indexing="ij"), -1).reshape(-1, 3)
    centers = -6 + (g + 0.5)
    exposed = np.ones(len(g))
    a = face_alignment(g, (12, 12, 12), exposed, None, field=field, centers=centers, h=0.25)
    assert a.min() > 0.93
    R = _rotation([0, 0, 45])
    a = face_alignment(g, (12, 12, 12), exposed, R, field=field, centers=centers, h=0.25)
    side = (np.abs(centers[:, 0]) > 5) & (np.abs(centers[:, 1]) < 3) & (np.abs(centers[:, 2]) < 3)
    assert np.allclose(a[side], 1 / np.sqrt(2), atol=0.02)
    sq, bias, _ = FACE_CUT["bending"]
    assert face_cut_knockdown("cubic", [1.0])[0] == pytest.approx(sq)
    assert face_cut_knockdown("cubic", [1 / np.sqrt(3)])[0] == pytest.approx(bias)
    assert face_cut_knockdown("octet", [1 / np.sqrt(3)])[0] < 0.5 * bias
    assert _guard("auto", 0.7, True, None, None, None, "safetyFactor", None, None, validated=True) is None
    assert _guard("auto", 0.7, True, None, None, None, "safetyFactor", None, None)["confirmed"] is None
