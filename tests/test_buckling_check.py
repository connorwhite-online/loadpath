"""verify's buckling check (verify._buckling) and the pass rule built on it
(study._margin): a slender column's load factor against Euler, and a part
strong enough but not stable enough fails its safety factor."""
import numpy as np

from loadpath.shapes import Box, trimesh_from_manifold
from loadpath.spec import study_from_dict
from loadpath.study import _margin
from loadpath.verify import verify_part

E, A, L = 2000.0, 4.0, 80.0


def _column(force, sf=2.0, buckling=True):
    goal = {"safetyFactor": sf, "volumeFraction": 0.5}
    if not buckling:
        goal["buckling"] = False
    s = study_from_dict({
        "designSpace": {"type": "box", "min": [0, 0, 0], "max": [A, A, L]},
        "supports": [{"shape": {"type": "box", "min": [-1, -1, -1], "max": [A + 1, A + 1, 0.01]}}],
        "loadCases": [{"name": "press", "loads": [
            {"shape": {"type": "box", "min": [-1, -1, L - 0.01], "max": [A + 1, A + 1, L + 1]},
             "force": [0, 0, -force]}]}],
        "material": {"E": E, "nu": 0.3, "strength": 50, "density": 1.2},
        "goal": goal,
    })
    mesh = trimesh_from_manifold(Box([0, 0, 0], [A, A, L]).to_manifold())
    return s, verify_part(s, mesh, max_cells=int(A * A * L / 0.5 ** 3), refine=0)  # 0.5 mm cells: 8 across


def test_column_buckles_at_euler_load():
    # cantilever (fixed base, free top): P = pi^2 E I / (4 L^2) = 16.4 N
    p_euler = np.pi ** 2 * E * (A ** 4 / 12) / (4 * L ** 2)
    s, rep = _column(1.0)
    assert rep["safetyFactor"] > 10            # nowhere near yield: 0.06 MPa
    assert 0.9 < rep["bucklingFactor"] / p_euler < 1.1, rep["buckling"]
    # loaded to 0.8 of its buckling load: strong enough, not stable enough
    s, rep = _column(0.8 * p_euler)
    assert rep["safetyFactor"] > 2.0
    assert _margin(s, rep) < 0.7
    assert any("buckles" in w for w in rep.get("warnings", []))


def test_buckling_false_reports_without_failing():
    p_euler = np.pi ** 2 * E * (A ** 4 / 12) / (4 * L ** 2)
    s, rep = _column(0.8 * p_euler, buckling=False)
    assert rep["bucklingFactor"] < 2.0
    assert _margin(s, rep) >= 1.0


def test_tension_does_not_buckle():
    s, rep = _column(-50.0)
    assert rep.get("bucklingFactor") is None or rep["bucklingFactor"] > 100
