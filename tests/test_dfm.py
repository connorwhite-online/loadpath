"""
Plain-python tests for the printability checks (dfm.py). No CAD kernel
needed — fixtures are trimesh primitives. Run:
`python3 cad-runner/tests/test_dfm.py`.

The sealed-cavity fixture is assembled by hand (an outer shell plus an
inverted inner shell) rather than by subtracting one box from another. That
is deliberate twice over: it is exactly the topology the trapped-void probe
has to recognise, and it does not depend on trimesh having a boolean backend
— a dependency this file originally took by accident, passing locally and
failing CI with "No boolean backend". Only test_open_cavity_drains still
needs one (manifold3d, pinned in requirements.txt).

Each test pins a contract the repair loop depends on: a thin part must be
CALLED thin, a sealed cavity must be found, and a plain printable box must
come back clean on all three probes (no false alarms — a check that cries
wolf sends the loop drilling drain holes through parts that never needed
them).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import trimesh  # noqa: E402

from dfm import check_dfm  # noqa: E402


def test_solid_box_is_clean():
    """A 20mm cube: thick walls, no enclosed voids, and no overhang beyond
    its own footprint (which rests on the plate and must be excluded)."""
    mesh = trimesh.creation.box(extents=(20, 20, 20))
    r = check_dfm(mesh, {"minWall": 1.5})
    assert r["watertight"], r
    assert r["minWallOk"], r
    assert r["minWallMm"] >= 15.0, f"20mm cube should measure ~20mm thick, got {r['minWallMm']}"
    assert r["drainsOk"], r
    assert r["trappedVoidCount"] == 0, r
    assert r["overhangFraction"] == 0.0, f"a cube has no overhangs, got {r['overhangFraction']}"
    assert r["ok"], r


def test_thin_plate_fails_min_wall():
    """A 0.8mm plate is watertight and unprintable — exactly the case the
    validity flags miss."""
    mesh = trimesh.creation.box(extents=(40, 40, 0.8))
    r = check_dfm(mesh, {"minWall": 1.5})
    assert r["watertight"], r
    assert not r["minWallOk"], f"0.8mm plate must fail a 1.5mm target: {r}"
    assert r["minWallMm"] < 1.5, r
    assert r["thinAreaFraction"] > 0.5, f"most of a plate's area is thin: {r}"
    assert not r["ok"], r


def _hollow_box(outer_mm, inner_mm):
    """Outer shell + inner shell with INVERTED winding = a solid with one
    sealed cavity. No boolean backend involved: flipping the inner faces is
    what makes its enclosed volume read as void rather than material."""
    outer = trimesh.creation.box(extents=(outer_mm,) * 3)
    inner = trimesh.creation.box(extents=(inner_mm,) * 3)
    inner.invert()
    return trimesh.util.concatenate([outer, inner])


def test_sealed_cavity_is_trapped():
    """A hollow shell with no drain: one enclosed void, correctly located."""
    mesh = _hollow_box(30, 16)
    r = check_dfm(mesh, {"minWall": 1.5, "pitch": 0.8})
    assert r["watertight"], r
    assert not r["drainsOk"], f"a sealed cavity must be reported: {r}"
    assert r["trappedVoidCount"] == 1, r
    void = r["trappedVoids"][0]
    # 16^3 = 4096mm^3, minus voxelization slack at this pitch.
    assert 2500 < void["volumeMm3"] < 5000, f"cavity volume off: {void}"
    assert np.allclose(void["center"], [0, 0, 0], atol=2.0), f"cavity centre off: {void}"
    assert not r["ok"], r


def test_open_cavity_drains():
    """The same shell with a hole through it drains — the void reaches the
    outside, so it must NOT be flagged."""
    outer = trimesh.creation.box(extents=(30, 30, 30))
    inner = trimesh.creation.box(extents=(16, 16, 16))
    drain = trimesh.creation.cylinder(radius=3.0, height=60.0)
    mesh = trimesh.boolean.difference([outer, inner, drain])
    r = check_dfm(mesh, {"minWall": 1.5, "pitch": 0.8})
    assert r["drainsOk"], f"a drained cavity must not be flagged: {r}"
    assert r["trappedVoidCount"] == 0, r


def test_overhang_is_measured():
    """A sphere's underside is the textbook overhang; a cube's is not.
    Relative, not absolute — the point is that the probe discriminates."""
    sphere = trimesh.creation.icosphere(subdivisions=3, radius=12.0)
    cube = trimesh.creation.box(extents=(20, 20, 20))
    rs = check_dfm(sphere, {"overhangDeg": 45})
    rc = check_dfm(cube, {"overhangDeg": 45})
    assert rs["overhangFraction"] > 0.1, f"a sphere overhangs: {rs}"
    assert rs["overhangFraction"] > rc["overhangFraction"], (rs, rc)


def test_build_direction_is_honoured():
    """Print the same part on its side and the overhang picture changes —
    proof the check reads `build`, not a hardcoded Z."""
    mesh = trimesh.creation.cylinder(radius=10.0, height=30.0)
    up = check_dfm(mesh, {"build": (0, 0, 1)})
    side = check_dfm(mesh, {"build": (1, 0, 0)})
    assert up["overhangFraction"] != side["overhangFraction"], (up, side)


def test_never_raises_on_degenerate_mesh():
    """An empty mesh must come back as a report, not an exception — one bad
    probe can never fail a whole generation."""
    mesh = trimesh.Trimesh(vertices=np.zeros((0, 3)), faces=np.zeros((0, 3), int))
    r = check_dfm(mesh)
    assert isinstance(r, dict) and "ok" in r, r


def test_unrunnable_probe_reports_unknown_not_pass():
    """The honesty rail: if the thickness probe cannot run (trimesh's ray
    engine needs rtree, an optional dep), the report must say UNKNOWN, never
    "no thin walls found". A missing dep silently greenlighting an
    unprintable part is exactly the silent-degradation failure
    requirements.txt warns about for shapely/rtree/mapbox-earcut."""
    import dfm

    def boom(*args, **kwargs):
        raise ModuleNotFoundError("No module named 'rtree'")

    mesh = trimesh.creation.box(extents=(40, 40, 0.8))
    original = dfm._wall_thickness
    try:
        dfm._wall_thickness = boom
        r = dfm.check_dfm(mesh, {"minWall": 1.5})
    finally:
        dfm._wall_thickness = original

    assert "minWallError" in r, f"the probe failure must be recorded: {r}"
    assert r["minWallOk"] is None, f"unrun must be None, not a pass: {r}"
    assert r["probesRan"] is False, r
    assert r["ok"] is False, f"an unverified part is not ok: {r}"


def _run():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in tests:
        try:
            fn()
            print(f"  ok   {fn.__name__}")
        except Exception as err:  # noqa: BLE001
            failed += 1
            print(f"  FAIL {fn.__name__}: {err}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(_run())
