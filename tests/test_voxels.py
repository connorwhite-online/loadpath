"""
Tests for voxels.py (the scanline voxelizer and tiled thickness probe) and
the checks built on it: check_dfm and check_networks from a mesh and from an
implicit field, and the MCP tools that run them on a mesh file. Run:
`python3 tests/test_voxels.py`.
"""
import os
import sys
import tempfile
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import trimesh  # noqa: E402

import loadpath.voxels as vox  # noqa: E402
from loadpath import mcp_server  # noqa: E402
from loadpath.checks.dfm import check_dfm  # noqa: E402
from loadpath.exchanger import exchanger_core  # noqa: E402
from loadpath.checks.networks import check_networks  # noqa: E402
from loadpath.sdf_kit import to_mesh  # noqa: E402


def _hollow_box(outer, inner):
    a = trimesh.creation.box(extents=(outer,) * 3)
    b = trimesh.creation.box(extents=(inner,) * 3)
    b.invert()
    return trimesh.util.concatenate([a, b])


def test_scanline_volume_matches_mesh():
    """Sphere and a box with a sealed cavity: the grid's volume is the
    mesh's, within the staircase, and the cavity stays void."""
    s = trimesh.creation.icosphere(subdivisions=4, radius=10)
    g, _, info = vox.mesh_solid(s, 0.4)
    assert abs(g.sum() * 0.4 ** 3 / s.volume - 1) < 0.02, (g.sum() * 0.064, s.volume)
    assert info["unclosedColumns"] == {"z": 0}, info
    hb = _hollow_box(30, 16)
    g, o, _ = vox.mesh_solid(hb, 0.5)
    assert abs(g.sum() * 0.125 / (30 ** 3 - 16 ** 3) - 1) < 0.03
    c = np.rint(-o / 0.5).astype(int)  # the voxel at the origin is in the cavity
    assert not g[tuple(c)]


def test_scanline_votes_over_a_holed_mesh():
    """A box missing one face is not watertight: three-axis voting still
    reads it solid instead of leaking columns to the top of the grid."""
    m = trimesh.creation.box(extents=(20, 20, 20))
    m.update_faces(m.face_normals[:, 2] > -0.5)  # drop the bottom face (two triangles)
    m.remove_unreferenced_vertices()
    assert not m.is_watertight
    g, _, info = vox.mesh_solid(m, 0.5)
    assert info["axes"] == ["x", "y", "z"], info
    assert abs(g.sum() * 0.125 / 8000 - 1) < 0.05, g.sum() * 0.125


def test_overlapping_shells_are_flagged():
    """Two boxes concatenated without a union: parity cancels their overlap
    to void, and the report must say so rather than pass it quietly. A
    proper cavity (inverted inner shell) is not an overlap."""
    a = trimesh.creation.box(extents=(20, 20, 20))
    b = trimesh.creation.box(extents=(20, 20, 20))
    b.apply_translation((10, 0, 0))
    r = check_dfm(trimesh.util.concatenate([a, b]), {"minWall": 1.5})
    assert abs(r["overlapMm3"] - 4000) < 400 and "overlapWarning" in r, r
    assert "overlapMm3" not in check_dfm(_hollow_box(30, 16), {"minWall": 1.5})


def test_field_and_mesh_grids_agree():
    """The same sphere from its field and from its mesh."""
    def f(P):
        return np.linalg.norm(P, axis=1) - 10.0
    gf, _ = vox.field_solid(f, (-10, -10, -10), (10, 10, 10), 0.5)
    gm, _, _ = vox.mesh_solid(trimesh.creation.icosphere(subdivisions=5, radius=10), 0.5,
                              bounds=((-10, -10, -10), (10, 10, 10)))
    assert gf.shape == gm.shape
    assert (gf ^ gm).sum() < 0.03 * gf.sum(), ((gf ^ gm).sum(), gf.sum())


def test_tiled_thickness_matches_untiled():
    """Tiles must not change the answer: a 1.5 mm plate and a 6 mm bar, read
    whole and in tiles a few cells long."""
    plate = trimesh.creation.box(extents=(40, 30, 1.5))
    bar = trimesh.creation.box(extents=(6, 6, 30))
    bar.apply_translation((30, 0, 0))  # apart: overlapping shells would cancel by parity
    part = trimesh.util.concatenate([plate, bar])
    g, o, _ = vox.mesh_solid(part, 0.25, axes=(2,))
    whole = vox.wall_thickness(g, 0.25, cap_mm=12, limit_mm=2.0, origin=o)
    saved = vox._TILE_VOXELS
    try:
        vox._TILE_VOXELS = g.shape[1] * g.shape[2] * 40
        tiled = vox.wall_thickness(g, 0.25, cap_mm=12, limit_mm=2.0, origin=o)
    finally:
        vox._TILE_VOXELS = saved
    for k in ("minMm", "p1Mm", "p5Mm", "ridgeCells", "thinAreaMm2"):
        assert abs(whole[k] - tiled[k]) < 1e-6, (k, whole[k], tiled[k])
    assert abs(whole["p1Mm"] - 1.5) <= 0.25, whole


def test_dfm_on_a_field():
    """check_dfm straight from a field: a 1.0 mm sheet gyroid block reads
    near 1.0 mm and fails a 1.5 mm minimum; a thick one passes."""
    from loadpath.sdf_kit import box, gyroid, intersect

    def thin(P):
        return intersect(gyroid(P, 10.0, 1.0), box(P, (0, 0, 0), (12, 12, 12)))
    r = check_dfm(thin, {"minWall": 1.5, "bounds": ((-13, -13, -13), (13, 13, 13))})
    assert r["source"] == "field" and r["minWallOk"] is False, r
    assert 0.7 <= r["wallP1Mm"] <= 1.3, r
    assert r["thinSpots"], r

    def thick(P):
        return box(P, (0, 0, 0), (10, 10, 10))
    r = check_dfm(thick, {"minWall": 1.5, "bounds": ((-11, -11, -11), (11, 11, 11))})
    assert r["minWallOk"] is True and r["drainsOk"] is True, r


def test_dfm_says_inconclusive_when_too_coarse():
    """A voxel budget too small to resolve the wall must not be read as a
    pass or a fail."""
    import loadpath.checks.dfm as dfm

    saved = dfm._VOXEL_BUDGET
    try:
        dfm._VOXEL_BUDGET = 20_000
        r = dfm.check_dfm(trimesh.creation.box(extents=(100, 100, 2)), {"minWall": 1.5})
    finally:
        dfm._VOXEL_BUDGET = saved
    assert r["minWallOk"] is None and "minWallInconclusive" in r, r
    assert r["probesRan"] is False and r["ok"] is False, r


def test_networks_field_matches_mesh():
    """A crossflow exchanger with a 1.2 mm wall: isolated from the field and
    from its mesh, with the wall between the fluids in mm within a pitch."""
    field, ports, plugs, bounds = exchanger_core((24, 24, 24), cell=8.0, wall=1.2)
    t = time.time()
    rf = check_networks(field, ports, plugs=plugs, min_feature=1.2, bounds=bounds)
    tf = time.time() - t
    mesh = to_mesh(field, *bounds, pitch=0.4)
    rm = check_networks(mesh, ports, plugs=plugs, min_feature=1.2)
    for r in (rf, rm):
        assert r["isolated"] and r["components"] == 2, r
        assert abs(r["minWallMm"] - 1.2) <= r["minWallToleranceMm"], r
    assert rf["source"] == "field" and rm["source"] == "mesh"
    print(f"    field {tf:.1f}s: wall {rf['minWallMm']} +/- {rf['minWallToleranceMm']}; "
          f"mesh: {rm['minWallMm']}")


def test_networks_field_counterflow_and_hoods():
    """Counterflow (each X face split between the fluids) and hooded
    crossflow, both isolated from the field at two pitches. A distance-level
    thickening shut the channel throats where the field's gradient is
    shallow and read each fluid split in two."""
    for size, kw in (((60, 48, 32), {"flow": "counter"}),
                     ((60, 48, 40), {"flow": "cross", "hose": {"a": 20, "b": 10}})):
        field, ports, plugs, bounds = exchanger_core(size, cell=8.0, wall=1.0, **kw)
        for pitch in (0.3, 0.5):
            r = check_networks(field, ports, plugs=plugs, bounds=bounds, pitch=pitch)
            assert r["isolated"] and r["components"] == 2, (kw, pitch, r)


def test_networks_field_finds_a_leak():
    """Drill a hole through the separator: the field path must call it."""
    from loadpath.sdf_kit import capsule, subtract

    field, ports, plugs, bounds = exchanger_core((24, 24, 24), cell=8.0, wall=1.2)

    def leaky(P):
        return subtract(field(P), capsule(P, (-14, -14, -14), (14, 14, 14), 1.5))
    r = check_networks(leaky, ports, plugs=plugs, min_feature=1.2, bounds=bounds)
    assert not r["isolated"], r


def test_mcp_checks_on_a_mesh_file():
    """The MCP tools read a mesh file and return the checks' reports."""
    field, ports, plugs, bounds = exchanger_core((24, 24, 24), cell=8.0, wall=1.2)
    with tempfile.TemporaryDirectory() as d:
        to_mesh(field, *bounds, pitch=0.4).export(os.path.join(d, "hx.stl"))
        trimesh.creation.box(extents=(40, 40, 0.8)).export(os.path.join(d, "plate.stl"))
        r = mcp_server.check_printability("plate.stl", min_wall=1.5, build="+z", base_dir=d)
        assert r["minWallOk"] is False and r["minWallMm"] < 1.5, r
        r = mcp_server.check_printability("plate.stl", build="sideways", base_dir=d)
        assert r["ok"] is False and "build" in r["error"], r
        r = mcp_server.check_isolation(os.path.join(d, "hx.stl"), ports, plugs, min_feature=1.2)
        assert r["isolated"], r
        r = mcp_server.check_isolation("missing.stl", ports, base_dir=d)
        assert r["ok"] is False and "no mesh" in r["error"], r
        r = mcp_server.check_component_fit("plate.stl", [], base_dir=d)
        assert r["results"] == [], r


TESTS = [v for k, v in sorted(globals().items()) if k.startswith("test_")]


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
