"""
The joints' exact solids (jointcad.py) and a build with joints in it.

- a screw and the nut cut with the tool for it: valid solids, nothing of
  the nut inside the screw (the gap is all round), and the tessellations
  sound (no folds where a helical flank meets the core), over forms,
  pitches and sizes;
- a hinge's knuckles, rooms, bores and pin; a print-in-place bore's
  teardrop roof up the build;
- a snap's beam, the slot it bends in and the catch it drops into;
- `loadpath build` on a block with a nut and a print-in-place hinge's
  inner knuckles: the thread and the bore drawn exactly, the wall round
  the nut and round the bore measured on the part and the joints' checks
  run at those, the screw written next to the part and fitting it.

Needs build123d (the step group). Run: `python3 tests/test_jointcad.py`.
"""
import json
import os
import sys
import tempfile
import time
import traceback

import numpy as np

from loadpath import joints as J


def _sound(solid, tol=0.01):
    from loadpath.brepio import load_brep
    from loadpath.crease import crossing_faces

    m = load_brep(solid, tolerance=tol).mesh
    x = crossing_faces(np.asarray(m.vertices, float), np.asarray(m.faces), np.arange(len(m.faces)))
    return m.is_watertight and len(x) == 0


def test_screw_and_nut_are_exact_and_clear():
    import build123d as bd

    from loadpath.brepio import load_brep
    from loadpath.jointcad import _c, _i, nut_tool, screw_solid

    for d, P, form, sb in ((12, 2.5, "print", (0, 0, 1)), (8, 1.5, "iso", None), (20, 3.0, "print", (1, 0, 0)),
                           (10, 2.0, "print", (1, 0, 0)), (16, 2.5, "buttress", (0, 0, 1)), (24, 3.5, "iso", None)):
        t = J.thread(clamp=50, material="petg", process="fdm", d=d, pitch=P, engage=1.2 * d, form=form, torque=500,
                     screw_build=sb)
        L = 1.2 * d
        screw, tool = screw_solid(t, L), nut_tool(t, L)
        assert screw.is_valid and tool.is_valid, (d, P, form)
        block = bd.Solid.make_box(3 * d, 3 * d, L).moved(bd.Location((-1.5 * d, -1.5 * d, 0)))
        nut = _c(block, tool)
        # read on the meshes: an independent judge of the kernel's booleans
        clash = (load_brep(screw, tolerance=0.01).manifold() ^ load_brep(nut, tolerance=0.01).manifold()).volume()
        assert clash < 1e-3, (d, P, form, clash)
        # the screw fills most of the hole: the gap is a thin shell, not a loose fit
        hole = block.volume - nut.volume
        inside = _i(screw, block).volume
        assert 0.6 < inside / hole < 1.0, (d, P, form, inside / hole)
        assert _sound(screw) and _sound(tool), (d, P, form)
        if sb == (1, 0, 0):                                  # printed lying: a flat at the minor radius
            bb = screw.bounding_box()
            from loadpath.jointcad import FLAT_DEPTH

            assert abs(bb.min.Y + t["profile"]["screwMinor"] - FLAT_DEPTH) < 0.02


def test_hinge_solids():
    from loadpath.jointcad import _i, hinge_solids

    h = J.hinge(load=200, material="petg", process="fdm", d=6, width=14, ligament=3)
    s = hinge_solids(h)
    for k in ("outer", "inner", "outer_room", "inner_room", "outer_bore", "inner_bore", "pin"):
        assert s[k] is not None and s[k].is_valid, k
    # the two members' knuckles never touch; each turns in the room cut for it
    assert _i(s["outer"], s["inner"]) is None
    assert abs(_i(s["inner"], s["inner_room"]).volume - s["inner"].volume) < 1e-6
    p = J.hinge(load=100, material="petg", process="fdm", kind="print-in-place", axis=(1, 0, 0),
                load_dir=(0, 0, 1), build=(0, 0, 1))
    q = hinge_solids(p, up=(1, 0, 0))
    bb = q["inner_bore"].bounding_box()
    rb = 0.5 * p["bore"]
    assert abs(bb.max.X - rb * np.sqrt(2)) < 0.02 and abs(bb.min.X + rb) < 0.02     # the roof points up +x


def test_snap_solids():
    from loadpath.jointcad import _i, snap_solids

    s = J.snap(hold=15, material="petg", process="fdm")
    ss = snap_solids(s)
    assert all(v.is_valid for v in ss.values())
    bb = ss["beam"].bounding_box()
    assert abs(bb.max.Z - s["undercut"]) < 1e-6 and abs(bb.max.X - s["L"]) < 1e-6
    # the barb lies in the catch's window
    assert _i(ss["beam"], ss["catch"]).volume > 0.5 * s["b"] * s["undercut"] * 0.5


def test_build_with_joints():
    from loadpath.build import run_build

    doc = {"name": "joint-block", "material": "petg", "process": {"type": "fdm", "buildDirection": [0, 0, 1]},
           "designSpace": {"type": "box", "min": [-20, -12, 0], "max": [20, 12, 16]},
           "skin": "solid",
           "joints": [
               {"id": "screw", "type": "thread", "member": "nut", "at": [-8, 0, 0], "axis": [0, 0, 1],
                "length": 12, "clamp": 40, "knob": "auto", "slip": 1.3, "screwBuild": [1, 0, 0], "d": 10,
                "pitch": 2.0, "engage": 12},
               {"id": "hinge", "type": "hinge", "member": "inner", "at": [20, 0, 8], "axis": [0, 1, 0],
                "load": 150, "loadDir": [0, 0, 1], "kind": "print-in-place", "proud": True}]}
    with tempfile.TemporaryDirectory() as out:
        rep = run_build(doc, out=out, render=False)
        rows = {r["id"]: r for r in rep["checks"]}
        assert rows["open:screw-thread"]["ok"] and rows["drawn:screw-thread"]["ok"]
        assert rows["joint-fit:screw"]["ok"], rows["joint-fit:screw"]
        assert rows["joint:screw"]["measured"]["boss"] > 2 * 10.0     # the wall round the nut, read on the part
        assert rows["joint:hinge"]["ok"], rows["joint:hinge"]
        assert "ligament" in rows["joint:hinge"]["measured"] and "roof" in rows["joint:hinge"]["measured"]
        assert rows["sound"]["ok"] and rows["one-piece"]["ok"], (rows["sound"], rows["one-piece"])
        files = rep["joints"][0]["files"]
        assert any(f.endswith(".step") for f in files) and all(os.path.exists(os.path.join(out, f)) for f in files)
        json.dumps(rep)


TESTS = [test_screw_and_nut_are_exact_and_clear, test_hinge_solids, test_snap_solids, test_build_with_joints]


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
