"""
Organic surfacing (loadpath/subd.py): a quad cage's Catmull-Clark surface
fitted to a mesh and written as smooth patches in one closed STEP solid.
Needs the `surface` extra. About a minute.

Run: `python3 tests/test_subd.py`.
"""
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import trimesh  # noqa: E402

from loadpath import subd  # noqa: E402


def _torus_cage(R=30.0, r=10.0, nu=24, nv=12):
    U, V = np.meshgrid(np.arange(nu), np.arange(nv), indexing="ij")
    u, v = 2 * np.pi * U / nu, 2 * np.pi * V / nv
    P = np.stack([(R + r * np.cos(v)) * np.cos(u), (R + r * np.cos(v)) * np.sin(u), r * np.sin(v)], -1).reshape(-1, 3)
    idx = lambda i, j: (i % nu) * nv + (j % nv)  # noqa: E731
    Q = np.array([[idx(i, j), idx(i + 1, j), idx(i + 1, j + 1), idx(i, j + 1)] for i in range(nu) for j in range(nv)])
    return subd.Cage(P, Q)


def test_regular_cage_gives_exact_c2_patches():
    """All-regular cage: every patch is the limit surface itself (bicubic),
    and neighbours meet with the same normal; the exact limit normals
    agree with the patches'."""
    cage = _torus_cage()
    H = subd.Hierarchy(cage, 3)
    bs, lay = subd.patches(H, cage.V)
    q = subd.patch_quality(H, cage.V, bs, lay)
    assert q["patchToLimit"]["max"] < 1e-9, q
    assert q["edgeAngle"]["max"] < 1e-4, q
    N = subd.grid_normals(H, cage.V)
    t = np.arange(9) / 8
    n0 = subd._patch_normals(bs[0], np.repeat(t, 9), np.tile(t, 9)).reshape(9, 9, 3)
    assert np.abs(np.einsum("ijk,ijk->ij", n0, N[0]) - 1).max() < 1e-9


def _blob():
    m = trimesh.creation.icosphere(5, radius=20.0)
    V = m.vertices.copy()
    V[:, 2] *= 0.6
    V[:, 0] += 3 * np.sin(V[:, 1] / 5)
    return trimesh.Trimesh(V, m.faces)


def test_blob_to_step():
    """A curved body with extraordinary vertices: fitted within the
    tolerance, the patches G1 to a fraction of a degree, one valid closed
    solid holding the mesh's volume, and the cage written beside it."""
    from OCP.BRep import BRep_Tool
    from OCP.BRepMesh import BRepMesh_IncrementalMesh
    from OCP.TopAbs import TopAbs_FACE
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopLoc import TopLoc_Location
    from OCP.TopoDS import TopoDS

    from loadpath.step import read_step

    m = _blob()
    with tempfile.TemporaryDirectory() as td:
        rep = subd.export_surface(m, os.path.join(td, "blob.step"), cage_path=os.path.join(td, "blob.cage.obj"))
        assert os.path.getsize(rep["cage"]) > 0
        # every face of the file as read back meshes (rounded poles did not)
        back = read_step(rep["path"])
        BRepMesh_IncrementalMesh(back, 0.005, False, 0.1, True)
        ex = TopExp_Explorer(back, TopAbs_FACE)
        while ex.More():
            assert BRep_Tool.Triangulation_s(TopoDS.Face(ex.Current()), TopLoc_Location()) is not None
            ex.Next()
    rep.pop("shape")
    assert rep["valid"] and rep["freeEdges"] == 0, rep
    assert rep["within"], rep["deviation"]
    assert rep["folds"] == 0
    assert rep["irregularVertices"] > 0
    assert rep["patches"]["edgeAngle"]["max"] < 1.0, rep["patches"]
    # rectangles along the cage's lines: the faces away from star points
    # share poles with the corner faces fitted round them, so they are
    # within a micron or so of the limit, not to rounding
    assert rep["patches"]["patchToLimit"]["max"] < rep["tolerance"] / 3, rep["patches"]
    assert rep["patches"]["patchToLimit"]["regularMax"] < rep["tolerance"] / 30, rep["patches"]
    assert abs(rep["volume"] / rep["meshVolume"] - 1) < 2e-3, rep
    # regular quad regions packed into single B-spline faces
    assert rep["patches"]["packedSurfaces"] > 0
    assert rep["stepFaces"] < 0.5 * rep["cageFaces"], rep
    assert rep["patches"]["layout"] == "lines", rep["patches"]
    assert rep["stepFaces"] < 0.25 * rep["cageFaces"], rep      # 24 of 142 (52 face by face)
    # the worst point, not just the 99th percentile: every place within the
    # process accuracy, patches included, and no hole closed over
    w = rep["worst"]
    assert w["within"] and w["patchesMax"] <= w["bound"] and not w["holesClosed"], w
    assert w["eulerNumber"]["mesh"] == w["eulerNumber"]["cage"] == 2, w


def test_closed_holes_told_from_surface_misses():
    """A surface that spans a through-hole is a closed hole, sized from
    its miss, printable or not by the process's smallest hole; elsewhere
    it is within."""
    import manifold3d as mf
    import trimesh

    def body(hole):
        b = mf.Manifold.cube([20.0, 20.0, 4.0], True)
        if hole:
            b = b - mf.Manifold.cylinder(10.0, hole / 2, hole / 2, 64, True)
        o = b.to_mesh()
        return trimesh.Trimesh(o.vert_properties[:, :3], o.tri_verts, process=False)
    # a plain plate's surface against the same plate drilled through
    plate = body(0)
    cage = subd.quad_cage(plate, 600)
    cage, H = subd.fit_cage(cage, plate)
    for d, printable in ((1.2, False), (3.0, True)):
        drilled = body(d)
        dev, (d1, d2, X) = subd.deviation(H, cage.V, drilled)
        places, kind = subd.misses(H, cage.V, drilled, d1, d2, X, 0.3, subd.MIN_HOLE)
        holes = [p for p in places if p["kind"] == "hole"]
        assert holes and all(p["printable"] == printable for p in holes), places
        assert abs(max(p["diameter"] for p in holes) - d) < 0.35 * d, places
        assert all(np.hypot(*p["at"][:2]) < d for p in holes), places


def test_regular_rectangles_pack_into_one_face():
    """A torus cage with no star points: four faces (a rectangle may not
    close round the tube), the same limit surface, one valid solid."""
    from OCP.BRepCheck import BRepCheck_Analyzer

    cage = _torus_cage()
    H = subd.Hierarchy(cage, 3)
    bs, lay = subd.patches(H, cage.V)
    pk = subd.pack(H, cage.V, bs, lay)
    assert len(pk) <= 4 and sum(x[0] == "bspline" for x in pk) >= 3, [x[0] for x in pk]
    shape, free = subd.to_brep(pk)
    assert free == 0 and BRepCheck_Analyzer(shape).IsValid()


def test_tolerance_from_process_and_noise():
    tol, why = subd.pick_tolerance()
    assert abs(tol - 0.1) < 1e-12 and "accuracy" in why
    tol, why = subd.pick_tolerance(accuracy=0.2, noise=0.2)
    assert abs(tol - 0.4) < 1e-12 and "noise" in why
    # a clean mesh reads as nearly noiseless; jitter reads as about itself
    m = _blob()
    assert subd.noise(m) < 0.01
    j = m.copy()
    j.vertices += np.random.default_rng(0).normal(0, 0.05, j.vertices.shape)
    assert 0.03 < subd.noise(j) < 0.08


def test_cage_repairs():
    """The remesher's leftovers: a 4-sided hole closes with one quad, a
    6-sided one round a centre; a doublet's two quads become one."""
    cage = _torus_cage(nu=8, nv=6)
    V, Q = cage.V, cage.Q
    v, f = subd._close_holes(V, Q[1:])
    assert len(f) == len(Q) and len(v) == len(V)
    subd.Cage(v, f)
    # a doublet: split one quad (a, b, c, d) with a vertex on its diagonal
    a, b, c, d = Q[0]
    w = len(V)
    v2 = np.vstack([V, V[[a, c]].mean(0)])
    f2 = np.vstack([Q[1:], [[w, a, b, c], [w, c, d, a]]])
    v3, f3 = subd._dissolve_doublets(v2, f2)
    assert len(f3) == len(Q) and len(v3) == len(V)
    subd.Cage(v3, f3)


def test_cage_over_slivers():
    """A plate with an exact bore and boss (manifold booleans: long thin
    triangles across the flats, as an optimized part's bores and mount
    faces are) gives Instant Meshes nothing to extract; the cage is laid
    over an even resampling of the same shape instead and keeps its hole."""
    import manifold3d as m3d

    from loadpath.shapes import trimesh_from_manifold
    b = (m3d.Manifold.cube([40, 20, 8], True) - m3d.Manifold.cylinder(20, 5, 5, 256, True)
         + m3d.Manifold.cylinder(8, 10, 10, 256, True).translate([20, 0, 0]))
    m = trimesh_from_manifold(b)
    v, f = subd._instant_quads(m, dict(target_face_count=300, pure_quad=True, deterministic=True,
                                       smooth_iterations=2, crease_angle=-1.0))
    assert not len(f)            # the raw mesh: nothing (so this test still tests the fallback)
    cage = subd.quad_cage(m, 1200)
    tri = np.concatenate([cage.Q[:, [0, 1, 2]], cage.Q[:, [0, 2, 3]]])
    assert trimesh.Trimesh(cage.V, tri, process=False).euler_number == m.euler_number == 0
    assert len(cage.Q) > 400


def test_cage_finer_only_where_asked():
    """A spot asked k times finer gets edges about k times shorter there,
    the rest at the density asked, and the cage still lies on the mesh
    (Instant Meshes laid over space blown up round the spot, mapped back)."""
    import trimesh

    m = trimesh.creation.icosphere(5, radius=10.0)
    c0 = np.array([0.0, 0.0, 10.0])
    c = subd.quad_cage(m, 800, finer=[(c0, 3.0, 6.0)])
    L = np.linalg.norm(c.V[c.E[:, 0]] - c.V[c.E[:, 1]], axis=1)
    mid = c.V[c.E].mean(axis=1)
    near = np.median(L[np.linalg.norm(mid - c0, axis=1) < 2.0])
    far = np.median(L[np.linalg.norm(mid + c0, axis=1) < 8.0])
    assert far / near > 2.0, (near, far)
    assert len(c.Q) < 1.5 * 800, len(c.Q)
    assert np.abs(np.linalg.norm(c.V, axis=1) - 10.0).max() < 0.05
    P = np.random.default_rng(0).normal(size=(500, 3)) * 5
    spots = [(np.zeros(3), 3.0, 2.0), (np.array([1.0, 1.0, 0.0]), 2.5, 4.0)]
    assert np.abs(subd._warp(subd._warp(P, spots), spots, inverse=True) - P).max() < 1e-9


def test_star_points_cancelled_in_pairs():
    """Edge rotations and diagonal collapses take paired star points away
    and keep the cage a closed manifold of the same genus."""
    import trimesh

    from loadpath import subdcoarse

    m = trimesh.creation.icosphere(5, radius=10.0)
    c = subd.quad_cage(m, 1500)
    V, Q, touched, taken = subdcoarse.cancel(c.V, c.Q)
    c2 = subd.Cage(V, Q)                      # raises on a cage that is not closed and manifold
    lines = lambda v: int(v[v != 4].sum())
    assert taken > 0 and lines(c2.valence) < lines(c.valence), (taken, lines(c.valence), lines(c2.valence))
    assert c2.n - len(c2.E) + len(c2.Q) == 2
    assert c2.valence.min() >= 3 and len(touched) > 0


def test_rectangles_side_by_side_merge():
    """Two rectangles of the lines layout that together make one rectangle
    become one B-spline: fewer faces, still on the limit surface, every
    star point still a corner."""
    import trimesh

    from loadpath import subdlines

    m = trimesh.creation.icosphere(5, radius=10.0)
    c = subd.quad_cage(m, 1500)
    c, _ = subd.fit_cage(c, m, iterations=4)
    H = subd.Hierarchy(subd.Cage(c.V, c.Q), subd.LEVEL + 1)      # as export_surface lays it out
    plain, r0 = subdlines.lay_out(H, c.V, 0.1, merge=False)
    out, r1 = subdlines.lay_out(H, c.V, 0.1)
    assert plain is not None and out is not None, (r0, r1)
    assert r1["merged"] > 0 and r1["rectangles"] == r0["rectangles"] - r1["merged"], (r0, r1)
    assert r1["patchToLimit"]["max"] < 0.1 / 3 and r1["edgeAngle"]["max"] < 1.0, r1


def test_star_point_pairs_slide_together():
    """3-5 pairs no single move takes away are slid whole (edge rotations)
    until they meet star points they cancel with: fewer star points than
    cancelling alone, the cage still closed, manifold, same genus."""
    import trimesh

    from loadpath import subdcoarse

    m = trimesh.creation.icosphere(5, radius=10.0)
    c = subd.quad_cage(m, 3000)
    V, Q, _, _ = subdcoarse.cancel(c.V, c.Q)
    V2, Q2, touched, taken = subdcoarse.simplify(c.V, c.Q)
    a, b = subd.Cage(V, Q), subd.Cage(V2, Q2)
    stars = lambda k: int((k.valence != 4).sum())  # noqa: E731
    assert stars(b) < stars(a), (stars(a), stars(b))
    assert b.n - len(b.E) + len(b.Q) == 2 and b.valence.min() >= 3 and len(touched) > 0


def test_two_quads_closed_on_each_other_are_dropped():
    # Instant Meshes can leave two quads sharing all four corners (a shell
    # of no volume where a thin member pinches off): not a doublet to merge
    v = np.random.default_rng(0).random((8, 3))
    f = np.array([[0, 1, 2, 3], [0, 3, 2, 1]])
    v2, f2 = subd._dissolve_doublets(v, f)
    assert len(f2) == 0 and len(v2) == 0


def test_long_triangles_laid_out_on_even_resampling():
    # a flat face exported as a few long triangles: Instant Meshes cannot
    # resolve its field there, so the layout runs on an even resampling
    seen = []
    orig = subd._instant_quads

    def spy(mesh, kw):
        seen.append(float(mesh.edges_unique_length.max()))
        return orig(mesh, kw)
    subd._instant_quads = spy
    try:
        m = trimesh.creation.box([40.0, 30.0, 4.0])
        c = subd.quad_cage(m, 600)
    finally:
        subd._instant_quads = orig
    step = float(np.sqrt(m.area / 600))
    assert seen[0] <= step and len(c.Q) > 300


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
