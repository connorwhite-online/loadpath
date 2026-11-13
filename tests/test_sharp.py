"""
Tests for sharp.py (exact booleans in field graphs, the edge-accuracy
measure) and step.py (STEP export). Run: `python3 tests/test_sharp.py`
(about a minute; the STEP tests skip without OpenCascade, loadpath[step]).
"""
import os
import sys
import tempfile
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from loadpath.fields import FieldGraph, run_field  # noqa: E402
from loadpath.sharp import Exact, Op, Organic, edge_accuracy, mesh_distance, plan, plan_refs, study_refs  # noqa: E402
from loadpath.step import available, export_step, plan_of, read_step, shape_to_trimesh  # noqa: E402

# A plate with a collar round a through bore. The collar is a shell of a
# boss coaxial with the bore, so before the exact cut its inner wall sits
# on the bore's own surface: the case the extended field is for.
PLATE = {"nodes": {
    "plate": {"op": "shape", "shape": {"type": "box", "min": [-20, -15, -5], "max": [20, 15, 5]}},
    "soft": {"op": "offset", "input": "plate", "distance": 0.0},
    "bore": {"op": "shape", "shape": {"type": "cylinder", "center": [0, 0, 0], "axis": "z", "length": 30, "r": 6}},
    "boss": {"op": "shape", "shape": {"type": "cylinder", "center": [0, 0, 0], "axis": "z", "length": 14, "r": 6}},
    "collar": {"op": "shell", "input": "boss", "side": "outside", "thickness": 3},
    "body": {"op": "union", "inputs": ["soft", "collar"], "blend": 2},
    "part": {"op": "subtract", "inputs": ["body", "bore"]}}, "output": "part", "name": "plate"}


def _acc(g, m):
    F = g.field()
    return edge_accuracy(m, plan_refs(plan(g)), boundary=lambda P: np.abs(F(P)) < 1e-6)


def test_plan_splits_at_hard_booleans_with_shapes():
    g = FieldGraph(PLATE)
    p = plan(g)
    assert isinstance(p, Op) and p.op == "subtract"
    assert isinstance(p.kids[0], Organic) and p.kids[0].names == ["body"]
    assert isinstance(p.kids[1], Exact) and p.kids[1].name == "bore"
    assert plan_refs(p)[0][1] == "cut"
    # a blended boolean, and one between fields only, stay in the field
    doc = {"nodes": {**PLATE["nodes"], "part": {"op": "subtract", "inputs": ["body", "bore"], "blend": 1}}}
    assert isinstance(plan(FieldGraph(doc)), Organic)
    doc = {"nodes": {**PLATE["nodes"], "part": {"op": "intersect", "inputs": ["body", "soft"]}}}
    assert isinstance(plan(FieldGraph(doc)), Organic)


def test_sharp_bore_is_exact_where_the_field_rounds_it():
    """Meshed from the field the bore's rims are rounded (a few tenths of a
    millimetre off at a 0.5 mm pitch); with the exact cut every sample of
    the bore wall is within the chord tolerance, and the collar's coincident
    inner wall leaves nothing behind."""
    g = FieldGraph(PLATE)
    before = g.mesh(pitch=0.5, sharp=False, creases=False)
    a0 = _acc(g, before)
    sharp = g.mesh(pitch=0.5)
    a1 = _acc(g, sharp)
    assert a0["edges"]["max"] > 0.1, a0
    assert a1["max"] < 0.005 and a1["edges"]["exactFraction"] == 1.0, a1
    assert sharp.is_watertight and len(sharp.split(only_watertight=False)) == 1
    assert abs(sharp.volume / before.volume - 1) < 0.003, (sharp.volume, before.volume)
    # nothing inside the bore: a 5.99 mm pin passes
    r = np.linalg.norm(sharp.vertices[:, :2], axis=1)
    assert r.min() > 6 - 0.003, r.min()
    assert g.plan is not None and plan_of(sharp) is g.plan


def test_mesh_distance_matches_trimesh():
    import trimesh

    m = trimesh.creation.cylinder(radius=5, height=40, sections=64)
    P = np.random.default_rng(0).uniform([-8, -8, -23], [8, 8, 23], (2000, 3))
    want = trimesh.proximity.closest_point(m, P)[1]
    assert np.abs(mesh_distance(m, P) - want).max() < 1e-9


def test_step_keeps_the_bore_a_cylinder():
    if not available():
        print("  (OpenCascade not installed: skipped)")
        return
    g = FieldGraph(PLATE)
    m = g.mesh(pitch=0.5)
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "plate.step")
        rep = export_step(m, path, faces=4000, name="plate")
        assert rep["valid"] and rep["solids"] == 1, rep
        assert rep["faces"].get("cylinder", 0) >= 1, rep["faces"]
        assert abs(rep["volume"] / m.volume - 1) < 0.003, (rep["volume"], m.volume)
        back = shape_to_trimesh(read_step(path))
        assert back.is_watertight
        a = _acc(g, back)
        assert a["max"] < 0.003, a


def test_study_part_exports_exact_interfaces():
    """A study's delivered part carries its plan: the STEP file has the
    keep-out bore as a cylinder and the design space's faces as planes."""
    from loadpath.domain import build_domain
    from loadpath.reconstruct import reconstruct
    from loadpath.spec import study_from_dict
    sys.path.insert(0, str(ROOT / "tests"))
    from test_loadpath import LUG

    s = study_from_dict(LUG)
    d = build_domain(s)
    x = np.zeros(d.n_elem)
    c = d.mesh.centers if hasattr(d.mesh, "centers") else None
    if c is None:
        x[:] = 1.0
    else:
        x[np.abs(c[:, 1]) < 6] = 1.0           # a bar from the support to the eye
    x = np.maximum(x, d.preserve.astype(float))
    mesh, info = reconstruct(s, d, x)
    assert plan_of(mesh) is not None
    a = edge_accuracy(mesh, study_refs(s))
    assert a["byRole"]["cut"]["max"] < 0.003, a
    if not available():
        return
    with tempfile.TemporaryDirectory() as td:
        rep = export_step(mesh, os.path.join(td, "lug.step"), faces=4000)
        assert rep["valid"] and rep["solids"] == 1, rep
        assert rep["faces"].get("cylinder", 0) >= 1, rep["faces"]       # the pin bore
        assert abs(rep["volume"] / mesh.volume - 1) < 0.01, (rep["volume"], mesh.volume)


def test_run_field_reports_edges_and_writes_step():
    with tempfile.TemporaryDirectory() as td:
        doc = {**PLATE, "mesh": {"pitch": 0.6}, "check": False, "step": available() and {"faces": 3000}}
        _, rep = run_field(doc, td, render=False)
        assert rep["sharpEdges"]["edges"]["exactFraction"] == 1.0, rep["sharpEdges"]
        if available():
            assert rep["step"]["valid"] and os.path.exists(rep["step"]["path"]), rep["step"]


def test_collapse_keeps_holes():
    import trimesh

    from loadpath.step import _collapse

    t = trimesh.creation.torus(20, 3, major_sections=96, minor_sections=48)
    V, F, _ = _collapse(t.vertices, t.faces, 200, turn=0.5)
    c = trimesh.Trimesh(V, F, process=False)
    assert len(c.faces) <= 210 and c.is_watertight and c.euler_number == 0, (len(c.faces), c.euler_number)


def test_step_smooth_patches():
    if not available():
        print("  (OpenCascade not installed: skipped)")
        return
    import trimesh

    b = trimesh.creation.icosphere(4, radius=15)
    v = b.vertices
    b.vertices = v * (1 + 0.15 * np.sin(v[:, 0] / 3) * np.cos(v[:, 1] / 4))[:, None]
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "blob.step")
        rep = export_step(b, path, surface="smooth", patches=400)
        assert rep["valid"] and rep["solids"] == 1, rep
        assert rep["faces"].get("bspline", 0) >= 150 and not rep["faces"].get("facet"), rep["faces"]
        assert abs(rep["volume"] / b.volume - 1) < 0.002, (rep["volume"], b.volume)
        dev = rep["decimated"][0]["deviation"]
        assert rep["decimated"][0]["patches"] < 450, rep["decimated"]        # pairs, plus refined bends
        assert dev["max"] < 0.1, dev
        assert dev["p99"] < 0.05, dev
        back = shape_to_trimesh(read_step(path), deflection=0.005, angle=0.05)
        assert np.percentile(mesh_distance(b, back.sample(5000)), 99) < 0.05


def test_smooth_step_keeps_the_bore_exact():
    """surface="smooth" on a field part with a bore: the organic body in
    smooth patches, the bore one true cylinder, a valid solid on re-read."""
    if not available():
        print("  (OpenCascade not installed: skipped)")
        return
    import copy

    g = copy.deepcopy(PLATE)
    g["nodes"]["plate"]["shape"] = {"type": "box", "min": [-17, -12, -2], "max": [17, 12, 2]}
    g["nodes"]["soft"]["distance"] = 3.0
    g["nodes"]["body"]["blend"] = 4
    m = FieldGraph(g).mesh(pitch=0.5)
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "soft.step")
        rep = export_step(m, path, surface="smooth", patches=400)
        assert rep["valid"] and rep["solids"] == 1, rep
        assert rep["faces"].get("cylinder") == 1 and rep["faces"].get("bspline", 0) > 300, rep["faces"]
        sm = rep["decimated"][0]
        assert sm["leaf"] == "smooth" and sm["deviation"]["p99"] < 0.08, sm
        assert abs(rep["volume"] / m.volume - 1) < 0.002
        from OCP.BRepCheck import BRepCheck_Analyzer

        from loadpath.step import _volume

        back = read_step(path)
        assert BRepCheck_Analyzer(back).IsValid() and abs(_volume(back) / m.volume - 1) < 0.002, _volume(back)


def test_smooth_step_keeps_sealed_voids():
    """A body with sealed voids inside (a sandwich's skin round its lattice
    core): one solid, an outer shell and one inner shell per void."""
    if not available():
        print("  (OpenCascade not installed: skipped)")
        return
    import trimesh
    from OCP.BRepCheck import BRepCheck_Analyzer

    from loadpath.step import _volume

    b = trimesh.creation.icosphere(4, radius=15)
    v = b.vertices
    b.vertices = v * (1 + 0.1 * np.sin(v[:, 0] / 3))[:, None]
    voids = [trimesh.creation.icosphere(3, radius=6), trimesh.creation.icosphere(3, radius=3)]
    voids[1].apply_translation([0, 9, 0])
    for h in voids:
        h.invert()
    m = trimesh.util.concatenate([b, *voids])
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "voids.step")
        rep = export_step(m, path, surface="smooth", patches=600)
        assert rep["valid"] and rep["solids"] == 1 and not rep["faces"].get("facet"), rep
        assert abs(rep["volume"] / m.volume - 1) < 0.002, (rep["volume"], m.volume)
        back = read_step(path)
        assert BRepCheck_Analyzer(back).IsValid() and abs(_volume(back) / m.volume - 1) < 0.002, _volume(back)


def test_write_step_keeps_points_unrounded_when_rounding_breaks_the_part():
    """Points are rounded to 10 nm in the file; a part with features finer
    than that (a 4 nm thick box here; a few-nm edge where facets meet a
    bolt bore in the step-import tutorial) reads back invalid rounded, so
    write_step(verify=True) writes it again unrounded."""
    if not available():
        print("  (OpenCascade not installed: skipped)")
        return
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox
    from OCP.gp import gp_Pnt

    from loadpath.step import write_step

    box = BRepPrimAPI_MakeBox(gp_Pnt(0.3, 0.3, 0.3), gp_Pnt(1.3, 1.3, 0.300004)).Shape()
    plain = BRepPrimAPI_MakeBox(gp_Pnt(0.3, 0.3, 0.3), gp_Pnt(1.3, 1.3, 1.3)).Shape()
    with tempfile.TemporaryDirectory() as td:
        w = write_step(box, os.path.join(td, "thin.step"), verify=True)
        assert w["readBackValid"] and not w["roundedPoints"], w
        w = write_step(plain, os.path.join(td, "cube.step"), verify=True)
        assert w["readBackValid"] and w["roundedPoints"] and not w["pcurves"], w


# A skinned gyroid cube with a bore through it: the core is open to the
# bore, so the part is one closed surface with tens of tunnels (the
# robot-arm SLS links, drained, are this at full size).
GYROID_BLOCK = {"nodes": {
    "box": {"op": "shape", "shape": {"type": "box", "min": [-7, -7, -7], "max": [7, 7, 7]}},
    "skin": {"op": "shell", "input": "box", "thickness": 1.0},
    "core": {"op": "lattice", "type": "gyroid", "cell": 6.0, "thickness": 0.8},
    "fill": {"op": "intersect", "inputs": ["core", "box"]},
    "joined": {"op": "union", "inputs": ["skin", "fill"]},
    "bore": {"op": "shape", "shape": {"type": "cylinder", "center": [0, 0, 0], "axis": "z", "length": 20, "r": 2.5}},
    "part": {"op": "subtract", "inputs": ["joined", "bore"]}}, "output": "part", "name": "gyroid-block"}


def test_step_keeps_a_gyroid_cores_tunnels():
    """A TPMS core open to the outside does not decimate to a small face
    budget without closing tunnels or tearing; the STEP goes over the
    budget instead, within TOPOLOGY_TOL, one valid solid of the STL's
    volume and genus (was: 'does not decimate' after hours on the SLS links).
    Also through a field graph's `"step": {"lattice": "mesh"}`."""
    from loadpath.step import _decimate

    import trimesh

    m = FieldGraph(GYROID_BLOCK).mesh(pitch=0.6)
    m = trimesh.Trimesh(m.vertices, m.faces, process=False)     # the STL as handed over (no CSG plan)
    assert m.is_watertight and m.euler_number < -10, (m.is_watertight, m.euler_number)
    d = _decimate(m, 4000)
    assert d.is_watertight and d.euler_number == m.euler_number and len(d.faces) > 4000, \
        (len(d.faces), d.euler_number, m.euler_number)
    assert len(d.split(only_watertight=False)) == 1
    assert abs(d.volume / m.volume - 1) < 0.01, d.volume / m.volume
    if not available():
        print("  (OpenCascade not installed: skipped the STEP)")
        return
    with tempfile.TemporaryDirectory() as td:
        doc = {**GYROID_BLOCK, "mesh": {"pitch": 0.6}, "check": False, "step": {"lattice": "mesh", "faces": 4000}}
        _, rep = run_field(doc, td, render=False)
        st = rep["step"]
        assert st["readBackValid"] and abs(st["volume"] / st["meshVolume"] - 1) < 0.01, st
        assert st["decimated"][0]["deviation"]["p99"] < 0.1, st["decimated"]
        assert st.get("valid") and st["solids"] == 1 and st["lattice"]["export"] == "mesh", st


def test_flip_needles_removes_zero_area_faces():
    """A square pyramid whose front face is split at the middle of its base
    edge, closed by a zero-area triangle (M on A-B): the flip leaves every
    face with area and the same volume."""
    import trimesh

    from loadpath.step import _flip_needles

    A, B, C, D, M, P = range(6)
    V = np.array([[0, 0, 0], [2, 0, 0], [2, 2, 0], [0, 2, 0], [1, 0, 0], [1, 1, 1]], float)
    F = np.array([[A, M, P], [M, B, P], [M, A, B], [A, C, B], [A, D, C], [B, C, P], [C, D, P], [D, A, P]])
    m = trimesh.Trimesh(V, F, process=False)
    if m.volume < 0:
        m = trimesh.Trimesh(V, F[:, ::-1], process=False)
    assert m.is_watertight and m.is_winding_consistent and m.area_faces.min() == 0
    out = _flip_needles(m)
    assert out.is_watertight and out.is_winding_consistent and out.area_faces.min() > 0.1, out.area_faces
    assert abs(out.volume - m.volume) < 1e-9

def _padded_plan():
    """A meshed body clipped to a meshed envelope, two kept pads (one mesh,
    two bodies, their bases on the envelope's base) unioned back, and a
    screw hole drilled through each pad: the step-import tutorial's plan."""
    import trimesh

    from loadpath.shapes import Cylinder, MeshShape, Union
    from loadpath.sharp import Exact, Op, Organic

    def box(lo, hi):
        m = trimesh.creation.box(extents=np.subtract(hi, lo))
        m.apply_translation(np.add(lo, hi) / 2.0)
        return m

    body = trimesh.creation.icosphere(subdivisions=4, radius=1.0)
    body.apply_scale([22.0, 12.0, 9.0])
    env = MeshShape(box([-20, -10, 0], [20, 10, 12]))
    pads = MeshShape(trimesh.util.concatenate([box([-18, -4, 0], [-10, 4, 3]), box([10, -4, 0], [18, 4, 3])]))
    holes = Union([Cylinder([-14, 0, -1], [-14, 0, 14], 1.7), Cylinder([14, 0, -1], [14, 0, 14], 1.7)])
    p = Op("intersect", [Organic(None, ["organic"], mesh=body), Exact(Union([env, pads]), "envelope")])
    p = Op("union", [p, Exact(pads, "preserve")])
    return Op("subtract", [p, Exact(holes, "keep-out")])


def test_step_of_a_part_with_meshed_pads_is_one_valid_solid_with_exact_holes():
    """Facets against facets (the body, a meshed envelope, kept pads) are
    manifold3d's; OpenCascade cuts the exact holes. Fused by OpenCascade,
    pad faces lying on the body's came back empty, invalid, or never (the
    step-import tutorial ran past 30 minutes in one boolean)."""
    if not available():
        print("  (OpenCascade not installed: skipped)")
        return
    from loadpath import step

    two = step._faceted(_padded_plan().kids[0].kids[1].shape.mesh)       # one mesh, two bodies
    from OCP.BRepCheck import BRepCheck_Analyzer

    assert BRepCheck_Analyzer(two).IsValid()
    with tempfile.TemporaryDirectory() as td:
        t0 = time.time()
        rep = export_step(_padded_plan(), os.path.join(td, "padded.step"), faces=4000)
        assert rep["valid"] and rep["solids"] == 1, rep
        assert rep["faces"].get("cylinder", 0) >= 2, rep["faces"]
        assert not [n for n in rep.get("decimated", []) if "warning" in n], rep["decimated"]
        body, tools = step._normal(_padded_plan(), {}, 4000, [])
        want = body.volume() - sum((t[1] ^ body).volume() for t in tools)
        assert abs(rep["volume"] / want - 1) < 2e-3, (rep["volume"], want)
        assert time.time() - t0 < 120


def test_step_boolean_that_does_not_finish_is_written_as_facets():
    """The OpenCascade cut (used when the exact faces cannot be built on the
    finished mesh) runs under a clock; a boolean that does not come back in
    time leaves its cutters to manifold3d (facets), noted, and the file is
    still one valid solid of the right volume."""
    if not available():
        print("  (OpenCascade not installed: skipped)")
        return
    from loadpath import step

    def refuse(*a, **k):
        raise RuntimeError("refused for the test")

    old = step.BOOL_INLINE, os.environ.get("LOADPATH_STEP_BOOL_SECONDS"), step._cut_on_mesh
    step.BOOL_INLINE, os.environ["LOADPATH_STEP_BOOL_SECONDS"], step._cut_on_mesh = 0, "0.01", refuse
    try:
        with tempfile.TemporaryDirectory() as td:
            rep = export_step(_padded_plan(), os.path.join(td, "padded.step"), faces=4000)
    finally:
        step.BOOL_INLINE, step._cut_on_mesh = old[0], old[2]
        if old[1] is None:
            del os.environ["LOADPATH_STEP_BOOL_SECONDS"]
        else:
            os.environ["LOADPATH_STEP_BOOL_SECONDS"] = old[1]
    assert rep["valid"] and rep["solids"] == 1, rep
    assert rep["faces"].get("cylinder", 0) == 0, rep["faces"]
    assert any("facets" in n.get("warning", "") for n in rep["decimated"]), rep["decimated"]
    body, tools = step._normal(_padded_plan(), {}, 4000, [])
    want = body.volume() - sum((t[1] ^ body).volume() for t in tools)
    assert abs(rep["volume"] / want - 1) < 2e-3, (rep["volume"], want)


def _drilled_pad_plan():
    """The step-import elbow's bolt seats: pads that come from the drawn
    part with their screw holes already in them (a coarser polygon than the
    cutter's), each drilled again by its screw's keep-out, a counterbore
    whose floor is the pad's top, and a drive clearance coaxial with the
    counterbore running out of the part. Every one of those cutter
    surfaces lies on a surface the body already has."""
    import manifold3d as m3d
    import trimesh

    from loadpath.shapes import Cylinder, MeshShape, Union, trimesh_from_manifold
    from loadpath.sharp import Exact, Op, Organic

    pads = []
    for x in (-14.0, 14.0):
        pad = m3d.Manifold.cube((8.0, 8.0, 3.0)).translate((x - 4.0, -4.0, 0.0))
        pad = pad - m3d.Manifold.cylinder(5.0, 1.7, 1.7, 40).translate((x, 0.0, -1.0))
        pads.append(pad)
    pm = trimesh_from_manifold(m3d.Manifold.batch_boolean(pads, m3d.OpType.Add))
    pads = MeshShape(trimesh.Trimesh(np.array(pm.vertices, float), np.array(pm.faces), process=False))
    body = trimesh.creation.icosphere(subdivisions=4, radius=1.0)
    body.apply_scale([22.0, 12.0, 9.0])
    env = trimesh.creation.box(extents=[40, 20, 12])
    env.apply_translation([0, 0, 6])
    cut = []
    for x in (-14.0, 14.0):
        cut += [Cylinder([x, 0, -0.5], [x, 0, 3.0], 1.7), Cylinder([x, 0, 3.0], [x, 0, 6.5], 3.0),
                Cylinder([x, 0, 40.0], [x, 0, 3.0], 3.0)]
    p = Op("intersect", [Organic(None, ["organic"], mesh=body), Exact(MeshShape(env), "envelope")])
    p = Op("union", [p, Exact(pads, "preserve")])
    return Op("subtract", [p, Exact(Union(cut), "keep-out")])


def test_step_bores_exact_where_the_body_already_lies_on_them():
    """Bores drilled through drawn holes, counterbores on a pad's top and
    a coaxial clearance all come out as exact cylinders, in one valid solid
    of the cut's volume. OpenCascade's cut of the faceted body by these
    cylinders failed or came back uncut on some bodies and not others (the
    step-import tutorial passed on one machine and lost every bore on
    another); the faces are now made on manifold3d's cut."""
    if not available():
        print("  (OpenCascade not installed: skipped)")
        return
    from loadpath import step

    with tempfile.TemporaryDirectory() as td:
        rep = export_step(_drilled_pad_plan(), os.path.join(td, "drilled.step"), faces=4000)
    assert rep["valid"] and rep["solids"] == 1 and rep.get("readBackValid", True), rep
    assert rep["faces"].get("cylinder", 0) >= 4, rep["faces"]          # two screw bores, two counterbores
    assert not [n for n in rep.get("decimated", []) if "warning" in n], rep["decimated"]
    body, tools = step._normal(_drilled_pad_plan(), {}, 4000, [])
    import manifold3d as m3d

    want = (body - m3d.Manifold.batch_boolean([t[1] for t in tools], m3d.OpType.Add)).volume()
    assert abs(rep["volume"] / want - 1) < 1e-3, (rep["volume"], want)


def test_step_bores_exact_on_the_elbow_bolt_seat():
    """A bolt seat cut out of the step-import elbow's optimized body (one
    that lost every exact bore on main: OpenCascade's cut of these facets by
    the screw bore came back invalid, the counterbore and its drive
    clearance uncut, at every fuzz): the screw bore and the counterbore wall
    are exact cylinders, and the solid is the cut's volume."""
    if not available():
        print("  (OpenCascade not installed: skipped)")
        return
    import manifold3d as m3d
    import trimesh

    from loadpath import step
    from loadpath.shapes import Cylinder, Union
    from loadpath.sharp import Exact, Op, Organic

    z = np.load(os.path.join(os.path.dirname(__file__), "data", "elbow-bolt-seat.npz"))
    seat = trimesh.Trimesh(z["V"], z["F"], process=False)
    cut = [Cylinder([8, 8, -0.5], [8, 8, 3.0], 1.7), Cylinder([8, 8, 3.0], [8, 8, 6.5], 3.0),
           Cylinder([8, 8, 83.92], [8, 8, 3.0], 3.0), Cylinder([8, 8, -0.25], [8, 8, -80.9], 1.69)]
    plan = Op("subtract", [Organic(None, ["organic"], mesh=seat), Exact(Union(cut), "keep-out")])
    with tempfile.TemporaryDirectory() as td:
        rep = export_step(plan, os.path.join(td, "seat.step"), faces=100000)
    assert rep["valid"] and rep["solids"] == 1, rep
    assert rep["faces"].get("cylinder", 0) >= 2, rep["faces"]
    assert not [n for n in rep.get("decimated", []) if "warning" in n], rep["decimated"]
    body, tools = step._normal(plan, {}, 100000, [])
    want = (body - m3d.Manifold.batch_boolean([t[1] for t in tools], m3d.OpType.Add)).volume()
    assert abs(rep["volume"] / want - 1) < 1e-3, (rep["volume"], want)


TESTS = [test_plan_splits_at_hard_booleans_with_shapes, test_sharp_bore_is_exact_where_the_field_rounds_it,
         test_mesh_distance_matches_trimesh, test_step_keeps_the_bore_a_cylinder,
         test_study_part_exports_exact_interfaces, test_run_field_reports_edges_and_writes_step,
         test_collapse_keeps_holes, test_step_smooth_patches,
         test_smooth_step_keeps_the_bore_exact, test_smooth_step_keeps_sealed_voids,
         test_write_step_keeps_points_unrounded_when_rounding_breaks_the_part,
         test_step_keeps_a_gyroid_cores_tunnels, test_flip_needles_removes_zero_area_faces,
         test_step_of_a_part_with_meshed_pads_is_one_valid_solid_with_exact_holes,
         test_step_boolean_that_does_not_finish_is_written_as_facets,
         test_step_bores_exact_where_the_body_already_lies_on_them,
         test_step_bores_exact_on_the_elbow_bolt_seat]


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
