"""
A part in one run from a spec (loadpath/build.py, docs/BUILD.md): the pod
example builds and passes every check; the checks fail, by name and with
the number, when the spec asks for what the part is not (a square port
under a fillet assertion, a clearance the pocket does not have, a mass
out of reach, a form that drifted from its reference).

Needs the step extra and build123d. Run: `python3 tests/test_build.py`
(about three minutes).
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from loadpath.build import build_spec, check_part, has_build, run_build  # noqa: E402
from loadpath.schema import lint  # noqa: E402

POD = ROOT / "examples" / "one-run" / "pod"
_BUILT = {}


def _pod():
    """The pod built once (full settings) and kept for the tests after it."""
    if not _BUILT:
        out = tempfile.mkdtemp(prefix="lp-build-")
        doc = json.loads((POD / "spec.json").read_text())
        _BUILT.update(doc=doc, out=out, report=run_build(doc, base_dir=str(POD), out=out))
    return _BUILT


def _row(rep, cid):
    rows = [c for c in rep["checks"] if c["id"] == cid]
    assert rows, (cid, [c["id"] for c in rep["checks"]])
    return rows[0]


def test_spec_lints_and_parses():
    doc = json.loads((POD / "spec.json").read_text())
    assert has_build(doc) and not lint(doc, "study"), lint(doc, "study")
    spec = build_spec(doc, str(POD))
    assert [i.id for i in spec.interfaces] == ["port", "plunger", "posts", "lid-opening"]
    # the exterior came in as a CAD solid: its faces carry their kernel radii
    faces = spec.interface("port").brep.faces()
    assert any(f["minRadius"] is not None and abs(f["minRadius"] - 1.2) < 1e-3 for f in faces), faces
    # what an agent gets wrong is said in its words
    for bad, want in (({**doc, "interfaces": [{"id": "port", "op": "hole", "shape": doc["interfaces"][0]["shape"]}]},
                       "op is 'add'"),
                      ({**doc, "assert": [{"type": "fillet", "interface": "nope", "min": 1}]}, "does not have"),
                      ({**doc, "targetMass": {"grams": 9, "adjust": [{"zone": "roof"}]}}, "zone id")):
        try:
            build_spec(bad, str(POD))
        except ValueError as exc:
            assert want in str(exc), (want, str(exc))
        else:
            raise AssertionError(want)


def test_pod_builds_and_passes_every_check():
    b = _pod()
    rep = b["report"]
    assert rep["ok"] and not rep["failed"], [c for c in rep["checks"] if c["ok"] is False]
    # the exterior is the designer's own surface
    assert _row(rep, "exterior")["value"] <= 0.02, _row(rep, "exterior")
    # the mass landed in its window by moving the floor only
    m = rep["targetMass"]
    assert abs(rep["mass"]["assembly"] - 9.0) <= 0.15 and set(m["walls"]) == {"floor"}, (rep["mass"], m)
    # the port's round survived: the kernel's radius and the mesh's agree
    f = _row(rep, "port-mouth-round")
    assert abs(f["value"] - 1.2) < 0.01 and abs(f["measuredRadius"] - 1.2) < 0.1 and f["sharpestEdgeDeg"] < 50, f
    # every pocket is its component grown by its clearance, exactly
    assert abs(_row(rep, "clear:board")["value"] - 0.2) < 0.02, _row(rep, "clear:board")
    # by region: the port's drawn surface is the part's surface there, and nothing else is
    s = _row(rep, "port-mouth")
    assert s["ok"] and s["value"] > 0.99 and s["strayShare"] < 0.02, s
    assert _row(rep, "sound")["ok"] and _row(rep, "drawn:lid-opening")["ok"]
    for name in ("part.stl", "checks.json", "report.json", "views.png", "closeups/port.png"):
        assert os.path.exists(os.path.join(b["out"], name)), name
    import trimesh
    mesh = trimesh.load(os.path.join(b["out"], "part.stl"))
    assert mesh.is_watertight and mesh.is_winding_consistent


def test_a_square_port_fails_its_fillet_assertion():
    with tempfile.TemporaryDirectory() as tmp:
        shutil.copy(POD / "spec.json", tmp)
        cad = (POD / "cad.py").read_text()
        assert "MOUTH_R = 1.2" in cad
        Path(tmp, "cad.py").write_text(cad.replace("MOUTH_R = 1.2", "MOUTH_R = 0.3"))
        doc = json.loads(Path(tmp, "spec.json").read_text())
        doc.pop("targetMass")
        rep = run_build(doc, base_dir=tmp, out=os.path.join(tmp, "out"), quick=True, render=False)
        f = _row(rep, "port-mouth-round")
        assert rep["ok"] is False and f["ok"] is False and abs(f["value"] - 0.3) < 0.01, f
        assert "port-mouth-round" in rep["failed"]


def test_checks_run_on_a_part_made_elsewhere():
    b = _pod()
    part = os.path.join(b["out"], "part.stl")
    rep = check_part(b["doc"], part, base_dir=str(POD), render=False)
    assert rep["ok"], rep["failed"]
    # the same part against a spec that wants more room round the board
    doc = json.loads(json.dumps(b["doc"]))
    doc["components"][0]["clearance"] = 0.6
    rep = check_part(doc, part, base_dir=str(POD), render=False)
    c = _row(rep, "clear:board")
    assert c["ok"] is False and c["value"] < 0.3, c
    # and a part with the port filled in is not open there
    import trimesh
    from loadpath.shapes import manifold_from_trimesh, trimesh_from_manifold
    mesh = trimesh.load(part)
    plug = trimesh.creation.box((14, 6, 8))
    plug.apply_translation((0, 24.0, 6.8))
    ext = build_spec(b["doc"], str(POD)).exterior_mesh()
    filled = trimesh_from_manifold(manifold_from_trimesh(mesh) + (manifold_from_trimesh(plug)
                                                                  ^ manifold_from_trimesh(ext)))
    rep = check_part(b["doc"], filled, base_dir=str(POD), render=False)
    assert _row(rep, "open:port")["ok"] is False, _row(rep, "open:port")
    # by region too: the port's drawn surface is no longer the part's surface
    d = _row(rep, "drawn:port")
    assert d["ok"] is False and d["value"] < 0.6, d


def test_a_patched_seam_fails_the_region_check():
    # the defect that started this: a row of balls laid along a seam near the port mouth. No fillet
    # reading moves (the round is still there) and nothing is in the port; the window does not match the drawing
    import trimesh
    from loadpath.buildcheck import drawn_surface_check, sound_check
    from loadpath.shapes import manifold_from_trimesh, shape_from_spec, trimesh_from_manifold

    b = _pod()
    spec = build_spec(b["doc"], str(POD))
    mesh = trimesh.load(os.path.join(b["out"], "part.stl"))
    near = shape_from_spec({"type": "box", "min": [-8, 24.0, 3], "max": [8, 27, 11]})
    good = drawn_surface_check(spec, mesh, spec.interface("port"), near=near, both=True)
    assert good["ok"] and good["value"] > 0.99 and good["strayShare"] < 0.02, good
    man = manifold_from_trimesh(mesh)
    for x in np.linspace(-5.5, 5.5, 8):
        bead = trimesh.creation.icosphere(subdivisions=2, radius=0.7)
        bead.apply_translation((x, 25.9, 9.3))
        man = man + manifold_from_trimesh(bead)
    patched = trimesh_from_manifold(man)
    bad = drawn_surface_check(spec, patched, spec.interface("port"), near=near, both=True)
    assert bad["ok"] is False and ("on nothing drawn" in bad["detail"] or "not on the part" in bad["detail"]), bad
    # sound: the built part is; two solids pushed through each other close and are not
    assert sound_check(mesh)["ok"], sound_check(mesh)
    a = trimesh.creation.box((10, 10, 10))
    c = trimesh.creation.box((10, 10, 10))
    c.apply_translation((4.3, 3.7, 2.9))
    crossed = trimesh.util.concatenate([a, c])
    s = sound_check(crossed)
    assert crossed.is_watertight and s["ok"] is False and s["selfCrossings"] > 0, s


def test_a_round_against_a_mesh_exterior():
    # an exterior that came as a mesh has no face to fillet against: outside_of lays one through it
    import trimesh
    from build123d import Cylinder, Rot, fillet

    from loadpath.brepio import load_brep, outside_of

    ball = trimesh.creation.icosphere(subdivisions=5, radius=30.0)
    outside, off = outside_of(ball, "+y", center=(0, 0, 0), size=(14.0, 10.0), spacing=0.5, depth=5.0)
    assert off < 0.02, off
    tool = Rot(90, 0, 0) * Cylinder(3.0, 80.0) + outside
    rim = [e for e in tool.edges() if 28.0 < e.center().Y < 31.0 and abs(e.length - 18.85) < 0.3]
    assert rim, [e.center() for e in tool.edges()]
    b = load_brep(fillet(rim, 1.0))
    radii = [f["minRadius"] for f in b.faces() if f["minRadius"] is not None and abs(f["minRadius"] - 1.0) < 0.05]
    assert radii, [f["minRadius"] for f in b.faces()]


def test_a_mass_out_of_reach_says_so():
    doc = json.loads((POD / "spec.json").read_text())
    doc["targetMass"] = {"grams": 30.0, "adjust": [{"zone": "floor", "min": 1.2, "max": 3.0}]}
    doc.pop("assert")
    rep = run_build(doc, base_dir=str(POD), quick=True, render=False)
    m = _row(rep, "mass")
    lo, hi = rep["targetMass"]["reachable"]
    assert m["ok"] is False and hi < 30.0 and "widen their limits" in m["detail"], (m, rep["targetMass"])


def test_references_score_a_form_and_catch_a_drifted_one():
    from PIL import Image

    from loadpath.raster import Camera, render, shade_image
    from loadpath.reference import score_image, score_mesh

    b = _pod()
    ext = build_spec(b["doc"], str(POD)).exterior_mesh()
    cam = Camera(position=(90, -120, 80), look_at=(0, 0, 6), lens=85, width=640, height=400)
    with tempfile.TemporaryDirectory() as tmp:
        ref = os.path.join(tmp, "concept.png")
        Image.fromarray(shade_image(render([ext], cam))).save(ref)
        good = score_image(ext, ref, cam, out=os.path.join(tmp, "good.png"))
        assert good["ok"] and good["match"] > 0.95 and os.path.exists(good["image"]), good
        drifted = ext.copy()
        drifted.apply_scale((1.0, 1.08, 0.9))
        bad = score_image(drifted, ref, cam)
        assert not bad["ok"] and bad["match"] < 0.8, bad
        # a spec's references are scored on every build, from the camera given
        doc = json.loads(json.dumps(b["doc"]))
        doc["references"] = [{"id": "concept", "image": ref, "camera": cam.to_spec()}]
        assert not lint(doc, "study"), lint(doc, "study")
        rep = check_part(doc, os.path.join(b["out"], "part.stl"), base_dir=str(POD), out=os.path.join(tmp, "o"))
        r = _row(rep, "reference:concept")
        assert r["ok"] and os.path.exists(os.path.join(tmp, "o", "references", "concept.png")), r
    same = score_mesh(ext, ext)
    assert same["ok"] and same["iou"] > 0.999, same
    off = score_mesh(drifted, ext)
    assert not off["ok"], off


def test_cli_exit_code_follows_the_checks():
    b = _pod()
    part = os.path.join(b["out"], "part.stl")
    p = subprocess.run([sys.executable, "-m", "loadpath", "check-build", str(POD / "spec.json"), part, "--no-render"],
                       capture_output=True, text=True, cwd=ROOT)
    assert p.returncode == 0 and json.loads(p.stdout.splitlines()[-1])["ok"] is True, p.stdout + p.stderr
    p = subprocess.run([sys.executable, "-m", "loadpath", "check", str(POD / "spec.json")],
                       capture_output=True, text=True, cwd=ROOT)
    assert p.returncode == 0 and json.loads(p.stdout)["kind"] == "build", p.stdout + p.stderr
    with tempfile.TemporaryDirectory() as tmp:
        bad = os.path.join(tmp, "bad.json")
        with open(bad, "w") as fh:
            json.dump({"designSpace": {"type": "box", "min": [0, 0, 0], "max": [9, 9, 9]}, "material": "pa12",
                       "interfaces": [{"id": "a", "op": "cut", "shape": {"type": "sphere", "center": [4, 4, 9],
                                                                          "r": 2}}]}, fh)
        p = subprocess.run([sys.executable, "-m", "loadpath", "build", bad, "--out", os.path.join(tmp, "o")],
                           capture_output=True, text=True, cwd=ROOT)
        assert p.returncode == 2 and "skin" in json.loads(p.stdout)["error"], p.stdout + p.stderr


def test_mcp_job_reads_back():
    from loadpath import mcp_server

    r = mcp_server.build({"designSpace": {"type": "box", "min": [0, 0, 0], "max": [9, 9, 9]}})
    assert r["ok"] is False and "interfaces" in r["error"], r
    b = _pod()
    os.environ.setdefault("LOADPATH_HOME", tempfile.mkdtemp(prefix="lp-mcp-"))
    doc = json.loads(json.dumps(b["doc"]))
    doc.pop("targetMass")
    r = mcp_server.build(doc, base_dir=str(POD), quick=True)
    assert r["ok"] and r["kind"] == "build", r
    t0 = time.time()
    while True:
        info, pngs = mcp_server.status(r["jobId"])
        if info["status"] != "running":
            break
        assert time.time() - t0 < 600, info
        time.sleep(2)
    assert info["status"] == "done" and info["ok"] is True and not info["failed"], info
    assert os.path.exists(info["part"]) and pngs and pngs[0][:4] == b"\x89PNG", info


def test_a_part_drawn_whole_is_checked_without_a_skin():
    """A bracket or a mount drawn whole in CAD has no exterior to hollow:
    `skin: "solid"` checks it as drawn (it was refused for want of a skin)."""
    d = tempfile.mkdtemp(prefix="lp-solid-")
    Path(d, "cad.py").write_text(
        "from build123d import Box, Cylinder, Pos\n"
        "def body():\n    return Box(40, 20, 6) - Pos(10, 0, 0) * Cylinder(2.25, 10)\n"
        "def bolt():\n    return Pos(10, 0, 0) * Cylinder(2.0, 20)\n")
    doc = {"name": "plate", "designSpace": {"type": "brep", "path": "cad.py:body"}, "skin": "solid",
           "material": "petg", "process": "fdm",
           "components": [{"id": "bolt", "shape": {"type": "brep", "path": "cad.py:bolt"}, "clearance": 0.0}],
           "assert": [{"type": "wall", "min": 1.0}]}
    assert not lint(doc, "study"), lint(doc, "study")
    spec = build_spec(doc, d)
    assert spec.solid
    import trimesh
    part = trimesh.creation.box((40, 20, 6))
    hole = trimesh.creation.cylinder(2.25, 10, sections=96)
    hole.apply_translation((10, 0, 0))
    part = part.difference(hole)
    rep = check_part(doc, part, base_dir=d, render=False)
    assert _row(rep, "closed")["ok"] and _row(rep, "exterior")["ok"], rep["failed"]
    c = _row(rep, "clear:bolt")
    assert c["ok"] and c["value"] >= 0.2, c
    try:
        build_spec({**doc, "skin": "hollow"}, d)
    except ValueError as exc:
        assert "solid" in str(exc)
    else:
        raise AssertionError("an unknown skin word was taken")


def test_a_cut_is_backed_by_the_process_wall_and_an_opening_stays_open():
    """A cut that slices the body leaves at least the process's minimum
    wall beside it (with its margin), however near the cavity comes; a cut
    face that lies in the cavity (an opening's inner end) grows nothing. A
    piece too thin between drawn surfaces only is the spec's: the checks say
    which inputs bound it, and a declared `thin` feature passes."""
    d = tempfile.mkdtemp(prefix="lp-backing-")
    Path(d, "cad.py").write_text(
        "from build123d import Box, Pos\n"
        "def shell():\n    return Pos(0, 0, 6) * Box(30, 30, 12)\n"
        # takes the floor's lower half from outside: 0.5 mm of floor would be left
        "def recess():\n    return Pos(0, 0, -0.25) * Box(16, 16, 1.5)\n"
        # runs through the floor into the cavity: its top face is in the cavity
        "def opening():\n    return Pos(0, 0, 1.0) * Box(10, 10, 4)\n")
    doc = {"name": "tray", "designSpace": {"type": "brep", "path": "cad.py:shell"}, "skin": 1.0,
           "material": "pa12", "process": {"type": "sls", "minFeature": 0.8},
           "interfaces": [{"id": "recess", "op": "cut", "shape": {"type": "brep", "path": "cad.py:recess"}},
                          {"id": "opening", "op": "cut", "shape": {"type": "brep", "path": "cad.py:opening"}}]}
    assert not lint(doc, "study"), lint(doc, "study")
    from loadpath.build import BACKING_MARGIN, BodyField

    spec = build_spec(doc, d)
    assert spec.min_wall == 0.8
    body = BodyField(spec)
    w = 0.8 * BACKING_MARGIN
    # over the recess (top face z 0.5), between the floor's old top (z 1) and 0.5 + w
    P = np.array([[6.5, 0.0, 0.5 + 0.9 * w], [6.5, 0.0, 0.5 + 1.1 * w]])
    with_b = body._solid(P, 0.0)[0]
    without = body._solid(P, 0.0, backing=False)[0]
    assert without[0] > 0 and with_b[0] < 0 and with_b[1] > 0, (with_b, without)
    # over the opening's top face (z 3, in the cavity): nothing grows
    Q = np.array([[0.0, 0.0, 3.3], [3.0, 3.0, 3.5]])
    assert np.allclose(body._solid(Q, 0.0)[0], body._solid(Q, 0.0, backing=False)[0])
    # backing: false is the old body
    off = BodyField(build_spec({**doc, "interfaces": [{**doc["interfaces"][0], "backing": False},
                                                      doc["interfaces"][1]]}, d))
    assert off._solid(P, 0.0)[0][0] > 0

    # a sliver between drawn surfaces only (a pocket 0.5 mm inside the exterior's face) is the spec's
    from loadpath.buildcheck import _judge_pieces

    Path(d, "cad2.py").write_text(
        "from build123d import Box, Pos\n"
        "def block():\n    return Pos(0, 0, 5) * Box(30, 30, 10)\n"
        "def pocket():\n    return Pos(0, 0, 6) * Box(20, 29, 10)\n")
    doc2 = {"name": "block", "designSpace": {"type": "brep", "path": "cad2.py:block"}, "skin": "solid",
            "material": "pa12", "process": {"type": "sls", "minFeature": 0.8},
            "interfaces": [{"id": "pocket", "op": "cut", "shape": {"type": "brep", "path": "cad2.py:pocket"}}]}
    spec2 = build_spec(doc2, d)
    piece = lambda: {"volumeMm3": 5.0, "reachMm": 3.0, "wallMm": 0.5, "at": [0.0, 14.75, 6.0],
                     "_points": np.array([[x, 14.75, z] for x in (-3.0, 0.0, 3.0) for z in (4.0, 6.0, 8.0)]),
                     "_reach": np.full(9, 2.0)}
    bad, declared = _judge_pieces(spec2, [piece()], 0.32, 0.8)
    assert len(bad) == 1 and bad[0]["owner"] == "drawn" and set(bad[0]["boundedBy"]) == {"exterior", "pocket"}, bad
    spec2.doc = {**doc2, "thin": [{"id": "lip", "near": {"type": "box", "min": [-15, 14, 0], "max": [15, 15, 10]},
                                   "min": 0.4, "why": "a snap lip"}]}
    assert not lint(spec2.doc, "study"), lint(spec2.doc, "study")
    spec2.__dict__.pop("_thin", None)
    bad, declared = _judge_pieces(spec2, [piece()], 0.32, 0.8)
    assert not bad and declared[0]["declared"] == "lip", (bad, declared)
    # never under what the process forms as a detail (Materialize: 0.3 mm for SLS)
    spec2.doc["thin"][0]["min"] = 0.2
    spec2.__dict__.pop("_thin", None)
    bad, _ = _judge_pieces(spec2, [{**piece(), "wallMm": 0.25}], 0.32, 0.8)
    assert bad and "under the 0.3 mm" in bad[0]["declaredButFails"], bad


def test_a_pocket_face_inside_another_cut_is_not_backed():
    """Backing grows only beside a face that slices material. A mating
    part's pocket whose face lies inside the opening it closes (the
    Touch-Mic flap in its underside opening, 2026-11-24) slices nothing
    there: the opening removes that body anyway, so nothing grows on it."""
    from loadpath.build import _backed, _backing_frames

    class Box:
        def __init__(self, lo, hi):
            self.lo, self.hi = np.asarray(lo, float), np.asarray(hi, float)

        def sdf(self, P):
            c, h = (self.lo + self.hi) / 2, (self.hi - self.lo) / 2
            q = np.abs(np.asarray(P, float) - c) - h
            return np.linalg.norm(np.maximum(q, 0), axis=-1) + np.minimum(q.max(axis=-1), 0)

    opening = Box((-10, -10, -5), (10, 10, 5.45))      # the opening, up into the cavity
    pocket = Box((-8, -8, 0), (8, 8, 5.39))            # the closing part, inside it
    solid = lambda Q: -np.ones(len(Q))                 # the body before its cuts: all material
    P = np.array([[0.0, 0.0, 5.9], [0.0, 0.0, 6.2]])   # above both: in the cavity
    sol = np.full(len(P), 1.0)
    for shapes in ([(pocket, 1.0)], [(opening, 1.0), (pocket, 1.0)]):
        out = _backed(sol, _backing_frames(P, shapes), solid)
        if len(shapes) == 1:
            assert (out < 0).all(), out                # alone, its top face slices the body: backed
        else:
            # the pocket's face is inside the opening: it backs nothing; the
            # opening's top is backed only because the stand-in solid is all material
            fr = _backing_frames(P, [(pocket, 1.0)])
            assert len(fr) == 1
            only = _backed(sol, _backing_frames(P, [(pocket, 1.0), (opening, 0.0)]), solid)
            assert (only > 0).all(), only


TESTS = [
    test_spec_lints_and_parses,
    test_pod_builds_and_passes_every_check,
    test_a_square_port_fails_its_fillet_assertion,
    test_checks_run_on_a_part_made_elsewhere,
    test_a_part_drawn_whole_is_checked_without_a_skin,
    test_a_cut_is_backed_by_the_process_wall_and_an_opening_stays_open,
    test_a_pocket_face_inside_another_cut_is_not_backed,
    test_a_patched_seam_fails_the_region_check,
    test_a_round_against_a_mesh_exterior,
    test_a_mass_out_of_reach_says_so,
    test_references_score_a_form_and_catch_a_drifted_one,
    test_cli_exit_code_follows_the_checks,
    test_mcp_job_reads_back,
]


def main():
    try:
        import build123d  # noqa: F401
    except ImportError:
        print("build123d not installed: skipped (pip install build123d)")
        return
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
