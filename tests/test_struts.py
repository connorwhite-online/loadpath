"""
Tests for strut and conformal lattices (struts.py, the lattice node's strut
types and frames): exact strut distance against brute force, density
tables, meshing, the conformal frame's cells round a tube, overhangs, the
homogenized fits and the schema. Run: `python3 -m pytest tests/test_struts.py`.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from loadpath import struts  # noqa: E402
from loadpath.fields import FieldGraph  # noqa: E402
from loadpath.schema import lint  # noqa: E402

PERIODIC = [k for k in struts.STRUT_KINDS if k != "voronoi"]
TUBE = {"op": "shape", "shape": {"type": "cylinder", "center": [20, 0, 0], "axis": "x", "length": 40, "r": 10}}


@pytest.mark.parametrize("kind", PERIODIC)
def test_folded_distance_is_the_brute_force_distance(kind):
    """The symmetry fold and candidate pruning give exactly the distance to
    the nearest strut of the whole periodic lattice, stretched cells too."""
    rng = np.random.default_rng(0)
    U = rng.uniform(-3, 3, (4000, 3))
    T = struts._all_translates(struts.cell_segments(kind), 2)
    f = U - np.floor(U)
    for aspect in (None, (1.0, 1.0, 1.5)):
        a = np.ones(3) if aspect is None else np.asarray(aspect)
        brute = struts._seg_dist(f * a, T[:, 0] * a, T[:, 1] * a).min(axis=1)
        fast = struts.unit_cell_distance(kind, U, aspect=aspect)
        near = brute < 0.3                      # as far as a strut surface reaches
        assert np.abs(brute - fast)[near].max() < 1e-9, kind


def test_density_tables():
    """Thin cubic struts: three rods a cell, rho ~ 3 pi r^2; tables invert."""
    d = 0.08
    rho = float(struts.strut_relative_density("cubic", d, 1.0))
    assert abs(rho / (3 * np.pi * d * d / 4) - 1) < 0.08, rho
    for kind in ("bcc", "octet", "kelvin", "voronoi"):
        r = struts.strut_relative_density(kind, 1.2, 6.0)
        assert abs(float(struts.strut_diameter_for_density(kind, r, 6.0)) - 1.2) < 0.05, kind


def test_voronoi_distance_is_exact():
    v = struts.VoronoiStruts([0, 0, 0], [30, 20, 20], 6.0, seed=1)
    P = np.random.default_rng(2).uniform([0, 0, 0], [30, 20, 20], (2000, 3))
    brute = struts._seg_dist(P, v.edges[:, 0], v.edges[:, 1]).min(axis=1)
    near = brute < 2.0
    assert np.abs(v.distance(P) - brute)[near].max() < 1e-9


def test_strut_lattice_meshes_at_its_density():
    """An octet bar with a 1 mm strut in 5 mm cells meshes watertight and
    holds the volume its density table says (within the clipped edges)."""
    box = {"op": "shape", "shape": {"type": "box", "min": [0, 0, 0], "max": [20, 20, 20]}}
    g = FieldGraph({"nodes": {"box": box, "lat": {"op": "lattice", "type": "octet", "cell": 5, "thickness": 1.0},
                              "part": {"op": "intersect", "inputs": ["lat", "box"]}}})
    m = g.mesh(pitch=0.2)
    assert m.is_watertight
    want = 8000 * float(struts.strut_relative_density("octet", 1.0, 5.0))
    assert abs(m.volume / want - 1) < 0.06, (m.volume, want)


def test_strut_thickness_and_density_follow_fields():
    x = {"op": "coordinate", "axis": "x"}
    g = FieldGraph({"nodes": {"lat": {"op": "lattice", "type": "bcc", "cell": 5,
                                      "density": {"op": "remap", "input": x, "from": [0, 40], "to": [0.05, 0.3]}}}})
    perp = np.array([1.0, 1.0, -2.0]) / np.sqrt(6)       # normal to the [111] strut
    for xx, rho in ((0.0, 0.05), (40.0, 0.3)):
        r = 0.5 * float(struts.strut_diameter_for_density("bcc", rho, 5.0))
        mid = np.array([xx, 0, 0]) + 1.25 * np.ones(3)      # halfway along a strut
        inside, outside = mid + perp * (r - 0.03), mid + perp * (r + 0.03)
        assert g.evaluate(np.array([inside]), "lat")[0] < 0 < g.evaluate(np.array([outside]), "lat")[0], (xx, r)


def test_conformal_frame_fits_whole_cells_round_a_tube():
    """Whole cells of about 6 mm round a 20 mm tube at its skin (2 pi 9.5 /
    6 = 9.95; the frame field settles on 11, 5.4 mm, the two-stripe frame
    on 10), and the layer coordinate is the depth over the cell (a phase in
    the frame field, so up to a whole number of cells and its sign)."""
    g = FieldGraph({"nodes": {"tube": TUBE,
                              "lat": {"op": "lattice", "type": "octet", "cell": 6, "thickness": 1,
                                      "frame": {"type": "conformal", "surface": "tube"}},
                              "part": {"op": "intersect", "inputs": ["lat", "tube"]}}})
    nd = g.nodes["lat"]
    th = np.linspace(0, 2 * np.pi, 400, endpoint=False)
    P = np.stack([np.full(400, 20.0), 9.5 * np.cos(th), 9.5 * np.sin(th)], 1)
    U, cell = g.lattice_coords(nd, P)
    turns = [abs(np.unwrap(np.r_[U[:, k], U[0, k]] * 2 * np.pi)[-1] - np.unwrap(np.r_[U[:, k], U[0, k]] * 2 * np.pi)[0])
             / (2 * np.pi) for k in (0, 1)]
    assert round(max(turns)) in (10, 11), turns
    assert 5.4 < 2 * np.pi * 9.5 / round(max(turns)) < 6.6, turns
    layer = np.abs(U[:, 2] - np.round(U[:, 2]))
    assert np.allclose(layer, 0.5 / 6.0, atol=0.01), (layer.min(), layer.max())
    # the two-stripe frame: exactly 10, the layer coordinate the exact depth
    g2 = FieldGraph({"nodes": {"tube": TUBE,
                               "lat": {"op": "lattice", "type": "octet", "cell": 6, "thickness": 1,
                                       "frame": {"type": "conformal", "surface": "tube", "frameField": False}},
                               "part": {"op": "intersect", "inputs": ["lat", "tube"]}}})
    U, cell = g2.lattice_coords(g2.nodes["lat"], P)
    turns = [abs(np.unwrap(np.r_[U[:, k], U[0, k]] * 2 * np.pi)[-1] - np.unwrap(np.r_[U[:, k], U[0, k]] * 2 * np.pi)[0])
             / (2 * np.pi) for k in (0, 1)]
    assert round(max(turns)) == 10, turns
    assert np.allclose(U[:, 2], 0.5 / 6.0, atol=1e-6)


def test_conformal_frame_field_turns_box_edges():
    """The frame field turns the cells round a box's edges and corners: on
    a box (here turned 30 degrees, so no world axis helps) the cells are
    the box's own straight cells, a node layer on every face; the
    two-stripe frame sheared over 40% of them along the edges and miters
    (two of its three coordinates turn parallel there)."""
    box = {"op": "transform", "input": {"op": "shape", "shape": {"type": "box", "min": [0, 0, 0], "max": [24, 18, 12]}},
           "rotate": {"axis": "z", "degrees": 30, "center": [12, 9, 6]}}
    out = {}
    for ff in (True, False):
        g = FieldGraph({"nodes": {"S": box,
                                  "lat": {"op": "lattice", "type": "octet", "cell": 6, "thickness": 1,
                                          "frame": {"type": "conformal", "surface": "S", "frameField": ff}},
                                  "part": {"op": "intersect", "inputs": ["lat", "S"]}}, "output": "part"})
        nd = g.nodes["lat"]
        lo, hi = g.bounds()
        P = np.random.default_rng(0).uniform(lo, hi, (30000, 3))
        P = P[g.evaluate(P, "S") < 0]
        e = 1e-3
        J = np.stack([(g.lattice_coords(nd, P + e * d)[0] - g.lattice_coords(nd, P - e * d)[0]) / (2 * e)
                      for d in np.eye(3)], 2) * 6.0
        sv = np.linalg.svd(J, compute_uv=False)
        out[ff] = float((sv[:, 2] / sv[:, 0] < 0.3).mean())
        if ff:
            # straight, unit cells along the box's own axes
            R = np.array([[np.cos(np.pi / 6), -np.sin(np.pi / 6), 0], [np.sin(np.pi / 6), np.cos(np.pi / 6), 0], [0, 0, 1]])
            JR = np.abs(J @ R)
            assert np.percentile(np.abs(np.sort(JR.reshape(len(J), -1), 1)[:, -3:] - 1), 95) < 0.05
            # a layer of nodes on the faces (the layer phase is whole there)
            F = np.array([[0.0, 9.0, 6.0], [24.0, 9.0, 6.0], [12.0, 0.0, 6.0], [12.0, 9.0, 0.0], [12.0, 9.0, 12.0]])
            F = (F - [12, 9, 6]) @ R.T + [12, 9, 6]
            U, _ = g.lattice_coords(nd, F)
            assert (np.abs(U - np.round(U)).min(1) < 0.02).all(), U
    assert out[True] < 0.01 and out[False] > 0.2, out


def test_conformal_field_is_continuous_across_phase_grid_faces():
    """The conformal frame's phases are read without seams: a step of
    1 um never jumps the lattice field (before, two voxels of the phase grid
    unwrapped against different corners and the struts jumped ~1 mm at
    their shared face)."""
    box = {"op": "shape", "shape": {"type": "box", "min": [0, 0, 0], "max": [12, 12, 12]}}
    g = FieldGraph({"nodes": {"box": box,
                              "lat": {"op": "lattice", "type": "octet", "cell": 4, "thickness": 0.8,
                                      "frame": {"type": "conformal", "surface": "box"}},
                              "part": {"op": "intersect", "inputs": ["lat", "box"]}}, "output": "part"})
    rng = np.random.default_rng(1)
    P = rng.uniform(0, 12, (60000, 3))
    D = rng.normal(size=P.shape)
    D /= np.linalg.norm(D, axis=1)[:, None]
    f0 = g.evaluate(P, "lat")
    j1 = np.abs(g.evaluate(P + 0.01 * D, "lat") - f0)
    j2 = np.abs(g.evaluate(P + 0.001 * D, "lat") - f0)
    # steep is fine (round a phase defect); a jump does not shrink with the step
    jumps = (j1 > 0.05) & (j2 > 0.5 * j1)
    assert not jumps.any(), (int(jumps.sum()), float(j2[jumps].max()))


def test_overhangs_and_tall_cells():
    """Axis-aligned BCC struts rise 35.3 degrees: all bridge on FDM; a cell
    1.5x tall lifts them past 45 and the lattice prints unsupported."""
    r = struts.overhang_report("bcc", 5.0)
    assert r["unsupportedFraction"] == 1.0 and abs(r["minRise"] - 35.26) < 0.1
    assert struts.overhang_report("octet", 5.0)["horizontalFraction"] == pytest.approx(1 / 3, abs=1e-6)
    box = {"op": "shape", "shape": {"type": "box", "min": [0, 0, 0], "max": [20, 20, 20]}}
    g = FieldGraph({"nodes": {"box": box,
                              "lat": {"op": "lattice", "type": "bcc", "cell": 5, "thickness": 1.0,
                                      "aspect": [1, 1, 1.5]},
                              "part": {"op": "intersect", "inputs": ["lat", "box"]}}})
    rep = struts.lattice_printability(g, "lat", samples=50)
    assert rep["selfSupporting"] and rep["minRise"] > 46, rep


def test_schema_accepts_strut_and_conformal_lattices():
    doc = {"nodes": {"tube": TUBE,
                     "lat": {"op": "lattice", "type": "kelvin", "cell": 6, "thickness": 1.0, "blend": 0.5,
                             "frame": {"type": "conformal", "surface": "tube", "normalCell": 5}},
                     "foam": {"op": "lattice", "type": "voronoi", "cell": 6, "thickness": 1.0, "seed": 3},
                     "part": {"op": "intersect", "inputs": [{"op": "union", "inputs": ["lat", "foam"]}, "tube"]}},
           "output": "part"}
    assert not [m for m in lint(doc, "field") if m[0] == "error"], lint(doc, "field")
    bad = {**doc, "nodes": {**doc["nodes"], "lat": {**doc["nodes"]["lat"], "form": "sheet"}}}
    with pytest.raises(ValueError):
        FieldGraph(bad).evaluate(np.zeros((1, 3)), "lat")


def test_homogenized_strut_fits_are_conservative():
    """The octet fit sits at or under the [100] and [111] moduli of a coarse
    unit-cell solve, and its stress factor at or over the solve's."""
    from loadpath.lattice import STRUTS, homogenize, sheet_modulus, sheet_stress_factor

    if "octet" not in STRUTS:
        pytest.skip("no octet fit")
    d = float(struts.strut_diameter_for_density("octet", 0.16, 1.0))
    h = homogenize("octet", d, n=32)
    assert float(sheet_modulus("octet", h["rho"])) <= 1.15 * min(h["E"], h["E111"])
    assert float(sheet_stress_factor("octet", h["rho"])) >= 0.85 * max(h["stressFactor"], h["stressFactor111"])
