"""Cut cells (loadpath.cutcell): the constant-strain patch test on cut cells,
strips at an angle to the grid against beam theory, and the multigrid path
against the assembled one."""
import numpy as np
import pytest
from scipy.sparse.linalg import spsolve

from loadpath.cutcell import CutGrid
from loadpath.fem import Stiffness, elastic_c

E, NU = 1000.0, 0.3


def _block(n=4, cells=(5, 4, 3)):
    # a box cut by an oblique plane: many cut cells of every shape
    fd = np.array(cells) * n
    g = [(np.arange(d) + 0.5) / n for d in fd]
    x, y, z = np.meshgrid(*g, indexing="ij")
    fine = (0.6 * x + 0.5 * y + 0.3 * z) < 3.1
    return CutGrid(fine, n, np.zeros(3), 1.0, NU)


def test_linear_field_is_exact_on_cut_cells():
    cg = _block()
    assert cg.cut.sum() > 10
    a = np.array([[1e-3, 2e-4, -3e-4], [5e-4, -2e-3, 1e-4], [-1e-4, 3e-4, 7e-4]])
    u = (cg.mesh.node_xyz @ a.T).ravel()
    eps = np.array([a[0, 0], a[1, 1], a[2, 2], a[0, 1] + a[1, 0], a[1, 2] + a[2, 1], a[2, 0] + a[0, 2]])
    sig = elastic_c(E, NU) @ eps
    s = cg.nodal_stress(u, E)
    assert np.allclose(s, sig[None], rtol=0, atol=1e-6 * np.abs(sig).max())
    # rigid motion: no force anywhere
    st = Stiffness(cg.mesh, NU, np.zeros(cg.mesh.n_dof, bool))
    k = cg.assemble(st, E)
    w = np.array([0.1, -0.2, 0.3])
    ur = (np.cross(w, cg.mesh.node_xyz) + [1.0, 2.0, 3.0]).ravel()
    assert np.abs(k @ ur).max() < 1e-9 * E


@pytest.mark.parametrize("theta", [0, 30, 45])
def test_angled_strip_reads_beam_theory(theta):
    # two and three cells thick: the staircase read 0.75-1.28x here
    from loadpath.benchmarks.beams import Strip, cut

    for t in (2, 3):
        r = cut(Strip(t, theta), "bending")
        assert 0.84 < r["peak"] < 1.10, (t, theta, r["peak"])


def test_multigrid_matches_assembled():
    from loadpath.mg import GridElasticity

    cg = _block(cells=(12, 6, 4))
    x = cg.mesh.node_xyz
    fixed = np.repeat(x[:, 0] < 1e-9, 3)
    f = np.zeros(cg.mesh.n_dof)
    tip = np.nonzero(x[:, 0] > x[:, 0].max() - 1.5)[0]
    f[3 * tip + 1] = -1.0 / len(tip)
    st = Stiffness(cg.mesh, NU, fixed)
    k = cg.assemble(st, E)
    free = ~fixed
    u = np.zeros(cg.mesh.n_dof)
    u[free] = spsolve(k[free][:, free].tocsc(), f[free])
    g = GridElasticity(cg.mesh, NU, fixed, springs=cg.cut_operator(E), rtol=1e-10, maxiter=2000, device="cpu")
    g.set_moduli(cg.full_moduli(E))
    um = g.solve_many([f])[0]
    assert np.allclose(um, u, rtol=0, atol=1e-6 * np.abs(u).max())


def test_verify_hole_with_cut_cells():
    # verify.py's plate-with-hole case (Peterson Kt 2.43: 81 MPa) on cut
    # cells: 78.8 at 1 mm cells and 86.0 at 0.7 (whole cells: 92.7, 91.3)
    from loadpath.shapes import Box, Cylinder, trimesh_from_manifold
    from loadpath.spec import study_from_dict
    from loadpath.verify import verify_part

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
    rep = verify_part(s, plate, max_cells=10 ** 7, min_pitch=1.0, contact_band=3.0, cut_cells=True)
    c = rep["cases"][0]
    assert rep["cutCells"] > 0
    assert 74 < c["peakStress"] < 88, c["peakStress"]
    assert abs(c["peakAt"][0] - 60) < 2.5 and 3.5 < abs(c["peakAt"][1]) < 6.5, c["peakAt"]


def test_cut_cells_are_the_default_except_under_pressure():
    # whole cells read the robot arm's wrist link 0.35-3.5x off a 0.9 mm tet
    # reference on its sub-cell struts, cut cells 0.89-1.13x: verification
    # uses cut cells unless the study says otherwise, or carries a
    # pressure (its wetted nodes come from the whole-cell staircase)
    from loadpath.spec import study_from_dict
    from loadpath.verify import use_cut_cells

    base = {
        "designSpace": {"type": "box", "min": [0, 0, 0], "max": [40, 10, 10]},
        "supports": [{"shape": {"type": "box", "min": [-1, -1, -1], "max": [0.5, 11, 11]}}],
        "material": "petg", "goal": {"volumeFraction": 0.5},
    }
    force = {"loads": [{"shape": {"type": "box", "min": [39, -1, -1], "max": [41, 11, 11]}, "force": [0, 0, -10]}]}
    assert use_cut_cells(study_from_dict(base | force))
    assert not use_cut_cells(study_from_dict(base | force | {"resolution": {"cutCells": False}}))
    import os
    import sys

    sys.path.insert(0, os.path.dirname(__file__))
    from test_pressure import _sphere_vessel

    assert not use_cut_cells(study_from_dict(_sphere_vessel()))
    # nor with a thermal block (the expansion load is assembled on whole cells)
    assert not use_cut_cells(study_from_dict(base | force | {"thermal": {"ambient": 25, "heat": [
        {"shape": {"type": "box", "min": [39, -1, -1], "max": [41, 11, 11]}, "watts": 1.0}]}}))


def test_scan_voxelize_back_to_back_faces():
    """Two faces meeting back to back (a sheet a boolean leaves where a surface
    grazes a cut plane) are two crossings: merged as one shared edge they
    turned the rest of each column over."""
    import trimesh
    from loadpath.cutcell import scan_voxelize
    a = trimesh.creation.box([2, 2, 1]); a.apply_translation([0, 0, 0.5])
    b = trimesh.creation.box([2, 2, 1]); b.apply_translation([0, 0, 1.5])
    m = trimesh.util.concatenate([a, b])
    v = scan_voxelize(m, 0.25, np.array([-1.0, -1.0, 0.0]), np.array([8, 8, 8]))
    assert v.all()
