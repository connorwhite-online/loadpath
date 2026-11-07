"""
The exterior intake (loadpath/exterior.py): a broken, generated-style mesh,
an SDF function and a build123d-free B-rep all come out one clean closed
skin; SDF grids go out and back in as a study shape; handoff takes a
generated GLB as its exterior. About a minute.

Run: `python3 tests/test_exterior.py`
"""
import json
import os
import sys
import tempfile
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "docs" / "tutorials" / "generated-exterior"))

from loadpath.exterior import _Dist, load_exterior, save_sdf  # noqa: E402
from loadpath.schema import lint  # noqa: E402
from loadpath.shapes import shape_from_spec  # noqa: E402

_DIR = tempfile.mkdtemp()


def _standin():
    """The broken stand-in (docs/tutorials/generated-exterior/generated_standin.py) at a
    coarse pitch, and the true shape in the frame the intake puts it in."""
    import generated_standin as gs
    clean = gs.clean_mesh(pitch=0.6)
    broken = gs.break_it(clean)
    path = os.path.join(_DIR, "generated.glb")
    broken.export(path)
    lo, hi = clean.bounds
    clean.vertices -= np.array([0.5 * (lo[0] + hi[0]), 0.5 * (lo[1] + hi[1]), lo[2]])
    return path, clean


def _check_wrap(method, p95_max):
    path, truth = _standin()
    size = float(truth.extents.max())
    e = load_exterior(path, size=size, skin=1.2, pitch=0.5, method=method)
    r = e.report
    assert r["defects"]["watertight"] is False and r["defects"]["bodies"] > 3, r["defects"]
    assert e.mesh.is_watertight and len(e.mesh.split(only_watertight=False)) == 1
    assert r["wrap"]["method"] == method
    # unit scale and Y-up in, mm and Z-up out, on z = 0
    assert np.allclose(e.mesh.extents, truth.extents, atol=1.5 * 0.5), (e.mesh.extents, truth.extents)
    assert abs(e.mesh.bounds[0][2]) < 1e-6
    d = _Dist(truth, 0.1)(e.mesh.vertices)
    assert np.percentile(d, 95) < p95_max, np.percentile(d, [50, 95, 99])
    # the holes were found, the debris dropped
    assert r["defects"]["boundaryEdges"] > 0
    if method == "winding":     # (the flood erodes debris a couple of cells across away)
        assert len(r["droppedBodies"]) >= 4
    # the thin button rim and lanyard loop are found
    cap = r["capacity"]
    assert cap["cavityShare"] > 0.5 and cap["thinRegions"], cap
    centres = np.array([t["center"] for t in cap["thinRegions"][:3]])
    assert (centres[:, 0] < -30).any(), centres          # the lanyard loop, off the back
    return e


def test_wrap_flood():
    _check_wrap("flood", 1.25 * 0.5)    # the flood: within about a pitch; the winding number: a fraction of one


def test_wrap_winding():
    try:
        import igl  # noqa: F401
    except ImportError:
        print("  (libigl not installed: skipped)")
        return
    _check_wrap("winding", 0.25)


def test_sdf_function_and_roundtrip():
    def ellipsoid(P):
        return (np.linalg.norm(np.asarray(P) / np.array([30.0, 20.0, 10.0]), axis=1) - 1.0) * 10.0
    e = load_exterior(ellipsoid, pitch=0.5)
    assert e.report["kind"] == "sdf-function" and e.mesh.is_watertight
    v = 4 / 3 * np.pi * 30 * 20 * 10
    assert abs(e.mesh.volume - v) / v < 0.01, e.mesh.volume
    assert np.allclose(e.mesh.extents, [60, 40, 20], atol=0.3)
    # out as a grid, back in as a study shape
    npz = os.path.join(_DIR, "ellipsoid.sdf.npz")
    save_sdf(e.mesh, npz, pitch=0.5)
    spec = {"type": "sdf", "path": "ellipsoid.sdf.npz"}
    g = shape_from_spec(spec, _DIR)
    d = g.sdf(np.array([[0, 0, 0], [29.0, 0, 0], [31.0, 0, 0], [0, 0, 11.0]]))
    assert d[0] < -9 and -1.5 < d[1] < -0.5 and 0.5 < d[2] < 1.5 and 0.5 < d[3] < 1.5, d
    assert abs(g.mesh().volume - v) / v < 0.02
    st = {"designSpace": spec, "supports": [{"shape": {"type": "box", "min": [-31, -21, -11], "max": [-29, 21, 11]}}],
          "loadCases": [{"loads": [{"shape": {"type": "sphere", "center": [29, 0, 0], "r": 2}, "force": [0, 0, -10]}]}],
          "material": "pla"}
    assert not [x for x in lint(st, "study") if x["level"] == "error"], lint(st, "study")


def test_clean_mesh_is_untouched():
    import trimesh
    m = trimesh.creation.box((40, 30, 12))
    path = os.path.join(_DIR, "box.stl")
    m.export(path)
    e = load_exterior(path)
    assert "wrap" not in e.report and e.report["result"]["place"] == "keep"
    assert np.allclose(e.mesh.bounds, m.bounds)
    assert not e.report["warnings"]
    # metres flagged
    m2 = trimesh.creation.box((0.04, 0.03, 0.012))
    p2 = os.path.join(_DIR, "box_m.stl")
    m2.export(p2)
    assert any("unit scale or metres" in w for w in load_exterior(p2).report["warnings"])
    e3 = load_exterior(p2, scale=1000)
    assert np.allclose(e3.mesh.extents, [40, 30, 12])


def test_handoff_takes_a_generated_exterior():
    """A generated pad exterior (a GLB at unit scale, Y-up, with a hole
    torn in it) through handoff: the wrap is in the report, the study's
    exterior is the clean skin."""
    import manifold3d as m3d
    import trimesh
    from loadpath.handoff import handoff

    def stl(name, man):
        mm = man.to_mesh()
        tm = trimesh.Trimesh(np.asarray(mm.vert_properties)[:, :3], np.asarray(mm.tri_verts))
        p = os.path.join(_DIR, name)
        tm.export(p)
        return p, tm
    box = m3d.Manifold.cube((30, 30, 10)).translate((-15, -15, 0))
    hole = m3d.Manifold.cube((16, 16, 3)).translate((-8, -8, -1))
    _, ext = stl("pad.stl", box - hole)
    ext = ext.subdivide().subdivide().subdivide()
    keep = np.linalg.norm(ext.triangles_center - np.array([10.0, 3.0, 10.0]), axis=1) > 2.5   # a hole in the top
    g = trimesh.Trimesh(ext.vertices, ext.faces[keep])
    g.vertices = g.vertices / 30.0
    g.vertices = g.vertices @ np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], float).T     # z up -> y up
    glb = os.path.join(_DIR, "pad.glb")
    g.export(glb)
    mouth, _ = stl("mouth.stl", hole)
    flap, _ = stl("flap.stl", m3d.Manifold.cube((15.6, 15.6, 2.0)).translate((-7.8, -7.8, 0)))
    sw, _ = stl("switch.stl", m3d.Manifold.cube((4, 4, 2)).translate((-2, -2, 2)) +
                m3d.Manifold.cylinder(1.0, 1.0, 1.0, 32).translate((0, 0, 4)))
    b = {"name": "generated-pad", "exterior": {"path": glb, "size": 30, "place": "center", "pitch": 0.25},
         "skin": 1.0, "openings": [mouth], "material": "tpu88a",
         "components": [{"id": "flap", "shape": flap, "role": "seat"},
                        {"id": "switch", "shape": sw, "role": "switch", "installedOn": "flap"}],
         "switch": {"unit": "gf", "trigger": 65, "travel": 0.25, "gap": 0.4,
                    "presses": [{"at": [0, 0], "max": 300, "min": 100}]}}
    assert not [x for x in lint(b, "handoff") if x["level"] == "error"], lint(b, "handoff")
    r = handoff(b)
    rep = r["report"]
    assert not rep["lint"], rep["lint"]
    ix = rep["intake"]["exterior"]
    assert ix["wrap"]["patched"]["area"] > 0 and ix["result"]["watertight"], ix
    assert "capacity" in ix
    assert r["study"]["designSpace"]["path"].endswith(".handoff.stl")
    out = trimesh.load(r["study"]["designSpace"]["path"])
    assert out.is_watertight and np.allclose(out.bounds, [[-15, -15, 0], [15, 15, 10]], atol=0.3), out.bounds
    assert rep["plunger"]["tip"][2] > 5.0


def main():
    tests = [v for k, v in globals().items() if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"ok   {t.__name__}")
        except Exception:  # noqa: BLE001
            failed += 1
            print(f"FAIL {t.__name__}")
            traceback.print_exc()
    print(json.dumps({"failed": failed, "of": len(tests)}))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
