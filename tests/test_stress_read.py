"""The tet cross-check's stress reading is the part's, not the mesh's:
stress averaged over the material's length (failure.MATERIAL_LENGTH,
crosscheck._averaged_stress), regions captured to the surface's fidelity,
grids sized from the machine's memory (study._plan_bytes)."""
import dataclasses

import numpy as np
import pytest

from loadpath import crosscheck as cc
from loadpath import study as st
from loadpath.failure import material_length
from loadpath.materials import MATERIALS
from loadpath.spec import study_from_dict


def _block(n=(12, 6, 6), h=1.0):
    """A box of n cubes of edge h, six tetrahedra each: (pts, tets)."""
    ax = [np.arange(k + 1) * h for k in n]
    g = np.stack(np.meshgrid(*ax, indexing="ij"), axis=-1).reshape(-1, 3)
    idx = np.arange(len(g)).reshape([k + 1 for k in n])
    c = [idx[i:i + n[0], j:j + n[1], k:k + n[2]].ravel() for i in (0, 1) for j in (0, 1) for k in (0, 1)]
    # corners c[4i + 2j + k]; six tets round the 0-7 diagonal
    paths = [(0, 4, 6, 7), (0, 6, 2, 7), (0, 2, 3, 7), (0, 3, 1, 7), (0, 1, 5, 7), (0, 5, 4, 7)]
    tets = np.concatenate([np.stack([c[a] for a in p], axis=1) for p in paths])
    x = g[tets]
    neg = np.linalg.det(x[:, 1:] - x[:, :1]) < 0
    tets[neg] = tets[neg][:, [0, 2, 1, 3]]
    return g, tets


_BEAM = {"designSpace": [{"type": "box", "min": [0, 0, 0], "max": [120, 40, 40]}],
         "material": "pa12", "maxElements": 40_000,
         "supports": [{"shape": {"type": "box", "min": [0, 0, 0], "max": [2, 40, 40]}}],
         "loadCases": [{"name": "tip", "loads": [{"shape": {"type": "box", "min": [118, 0, 0], "max": [120, 40, 40]},
                                                 "force": [0, 0, -100]}]}],
         "goal": {"safetyFactor": 2, "method": "constrained"}}


def test_material_lengths_are_sourced():
    r, source, quality = material_length(MATERIALS["pa12"])
    assert r == 0.86 and "Avanzini" in source and quality.startswith("measured")
    # the glass-filled grade is its family's; metals and brittle resins read at the mesh's resolution
    assert material_length(MATERIALS["pa12-gf"])[0] == 0.86
    assert material_length(MATERIALS["ti64"])[0] == 0.0
    assert material_length(MATERIALS["resin"])[0] == 0.0
    study = study_from_dict(_BEAM)
    assert cc.stress_radius(study, 1.5)[0] == 0.86
    # a coarse mesh cannot resolve less than half an element
    r, why = cc.stress_radius(study, 3.0)
    assert r == 1.5 and why["of"] == "mesh" and why["materialLength"] == 0.86


def test_average_of_uniform_and_bending_fields():
    E, nu, k = 2000.0, 0.3, 2.0
    pts, tets = _block()
    nodes, conn, _edges = cc._p2_mesh(pts, tets)
    x, y, z = (nodes - nodes.mean(axis=0)).T
    # uniform pull along x, and pure bending (sigma_xx = k z): both exact on quadratic elements
    pull = np.stack([x / E, -nu * y / E, -nu * z / E], axis=1) * 10.0
    bend = np.stack([k * x * z / E, -nu * k * y * z / E, -k * (x * x + nu * (z * z - y * y)) / (2 * E)], axis=1)
    r = 1.2
    s_pull, s_bend = cc._averaged_stress(nodes, conn, [pull, bend], E, nu, len(pts), r)
    assert np.allclose(s_pull[:, 0], 10.0, rtol=1e-9) and np.abs(s_pull[:, 1:]).max() < 1e-8
    zc = pts[:, 2] - nodes[:, 2].mean()
    mid = (np.abs(pts[:, 0] - 6) < 3) & (np.abs(pts[:, 1] - 3) < 1.5)
    deep = mid & (np.abs(zc) < 1.5)
    # inside, the ball is whole and a linear field averages to its centre's value
    assert np.allclose(s_bend[deep, 0], k * zc[deep], atol=0.05 * k)
    # at the surface the ball is a hemisphere, whose centroid is 3/8 r deep:
    # the support a bending member has over a pulled one
    top = mid & (zc > 2.99)
    want = k * (3.0 - 3.0 / 8.0 * r)
    assert np.allclose(s_bend[top, 0], want, rtol=0.04), (s_bend[top, 0], want)


def test_region_capture_follows_the_surface_not_the_mesh():
    from loadpath.shapes import Box

    pts = np.array([[0.0, 0, 0], [0.25, 0, 0], [0.8, 0, 0], [3.0, 0, 0]])
    among = np.arange(4)
    shape = Box((-1, -1, -1), (0, 1, 1))
    assert list(cc._region_nodes(None, shape, pts, 0.3, among)) == [0, 1]
    # a region drawn a little off the surface still finds it (as it did at 0.6 elements)
    assert list(cc._region_nodes(None, shape, pts[2:], 0.3, np.arange(2))) == [0]
    assert len(cc._region_nodes(None, shape, pts[3:], 0.3, np.arange(1))) == 0


def test_grids_are_sized_from_the_machine_not_from_free_memory(monkeypatch):
    monkeypatch.setattr(st, "_machine_bytes", lambda: 16e9)
    monkeypatch.setattr(st, "_own_bytes", lambda: 0.0)
    monkeypatch.delenv("LOADPATH_TET_MEMORY", raising=False)
    seen = []
    for free in (15e9, 13e9, 11e9):
        monkeypatch.setattr(st, "_available_bytes", lambda free=free: free)
        # a solve that needs 5 GB fits in all three: the plan does not move
        seen.append((st._plan_bytes(lambda b: 5e9), cc.fit_size(180_000.0, 1.5), cc.fit_size(2_000_000.0, 1.5),
                     st._confirm_cells()))
    assert seen[0] == seen[1] == seen[2], seen
    assert seen[0][0] == st._PLAN_SHARE * 16e9
    assert seen[0][3] == st._CONFIRM_CELLS
    # only when the memory really there cannot hold the plan's grid does it shrink
    monkeypatch.setattr(st, "_available_bytes", lambda: 3e9)
    assert st._plan_bytes(lambda b: 5e9) == st._PLAN_SHARE * 3e9


def test_constrained_grid_is_the_same_whatever_is_free(monkeypatch):
    study = study_from_dict(_BEAM)
    monkeypatch.setattr(st, "_machine_bytes", lambda: 20e9)
    monkeypatch.setattr(st, "_own_bytes", lambda: 0.0)
    fits = []
    for free in (19.5e9, 18e9, 17e9):
        monkeypatch.setattr(st, "_available_bytes", lambda free=free: free)
        fits.append(st._constrained_fit(study))
    assert fits[0] is not None and "pitch" in fits[0], fits
    assert fits[0] == fits[1] == fits[2], fits
    # and a study that names its pitch is left alone
    assert st._constrained_fit(dataclasses.replace(study, pitch=2.0)) is None


def test_peak_at_a_sharp_corner_does_not_move_with_the_mesh():
    """An L bracket with a sharp inside corner, meshed at two sizes that
    both resolve the averaging radius: the nodal peak there grows as the
    mesh is refined (the stress is unbounded at the corner); the stress
    averaged over the material's length is the same on both. (Measured:
    16.4 to 17.2 MPa at 1.2 mm on two meshes, 16.4 at 0.9 mm. Elements
    twice the radius under-read a corner this sharp: 14.7 to 15.0 MPa at
    1.7 mm.)"""
    m3d = pytest.importorskip("manifold3d")
    from loadpath.shapes import trimesh_from_manifold

    if cc.mesher_missing():
        pytest.skip("no tetrahedral mesher")
    body = m3d.Manifold.cube((40, 10, 10)) + m3d.Manifold.cube((10, 10, 40))
    mesh = trimesh_from_manifold(body)
    study = study_from_dict({
        "designSpace": [{"type": "box", "min": [0, 0, 0], "max": [40, 10, 40]}], "material": "pa12", "minMember": 4,
        "supports": [{"shape": {"type": "box", "min": [0, 0, 39], "max": [10, 10, 40]}}],
        "loadCases": [{"name": "tip", "loads": [{"shape": {"type": "box", "min": [39, 0, 0], "max": [40, 10, 10]},
                                                "force": [0, 0, -60]}]}]})
    peaks, sizes = [], (1.2, 0.9)
    for size in sizes:
        r = cc.crosscheck(study, mesh, size=size, contact_band=6.0)
        assert r["stressRadius"] == 0.86
        c = r["cases"][0]
        # the corner governs
        assert np.linalg.norm(np.asarray(c["peakAt"])[[0, 2]] - [10, 10]) < 2.0, c["peakAt"]
        peaks.append(c["peakStress"])
    assert abs(peaks[0] - peaks[1]) < 0.10 * peaks[1], peaks
