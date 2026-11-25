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


def test_creased_cube_cage_is_the_cube():
    # every edge of a cube cage a crease: the limit surface is the cube
    V = np.array([[x, y, z] for x in (0, 1) for y in (0, 1) for z in (0, 1)], float)
    Q = np.array([[0, 1, 3, 2], [4, 6, 7, 5], [0, 4, 5, 1], [2, 3, 7, 6], [0, 2, 6, 4], [1, 5, 7, 3]])
    c = subd.Cage(V, Q)
    P = subd.Hierarchy(subd.Cage(V, Q, sharp=c.E), 3).limit()
    assert np.minimum(np.abs(P), np.abs(P - 1)).min(axis=1).max() < 1e-12
    P0 = subd.Hierarchy(c, 3).limit()
    assert np.minimum(np.abs(P0), np.abs(P0 - 1)).min(axis=1).max() > 0.1      # smooth: rounded


def test_box_edges_laid_as_creases():
    from loadpath import subdcrease

    m = trimesh.creation.box([40.0, 30.0, 6.0])
    tol, n = 0.1, 736
    h = float(np.sqrt(m.area / n))
    lines, _ = subdcrease.feature_lines(m, tol, h)
    assert len(lines) == 12                     # the 6 mm sides too: corner to corner
    c = subd.quad_cage(m, n, crease=np.degrees(8 * tol / h))
    c, rep = subdcrease.lay_creases(c, m, tol, h=h)
    assert rep["taken"] == 12
    c, H = subd.fit_cage(c, m, settle=tol / 50)
    dev, _ = subd.deviation(H, c.V, m)
    assert dev["max"] < tol                    # sharp edges, not rounded (smooth: 0.3 mm and more)


def test_ragged_rims_are_not_feature_lines():
    # a hole cut on a voxel grid: its rim turns sharply but zigzags at the
    # grid, branching at every step; no crease is laid on it
    from loadpath import subdcrease

    box = trimesh.creation.box([30.0, 30.0, 4.0])
    hole = trimesh.creation.cylinder(radius=5.0, height=10.0, sections=48)
    vox = box.difference(hole).voxelized(0.5).fill()
    m = vox.marching_cubes
    m.apply_transform(vox.transform)
    lines, _ = subdcrease.feature_lines(m, 0.1, 2.0)
    for P, closed, _ in lines:
        assert subdcrease._jag(P, closed, 2.0) <= 0.1
    rim = [P for P, c, _ in lines if np.linalg.norm(P[:, :2], axis=1).max() < 7.0]
    assert not rim


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)


def _pocket_plate():
    # a 40 x 30 x 6 slab, a 4 mm hole through it and a 2 mm deep pocket
    # in its top: CAD-like (a few long triangles per face)
    slab = trimesh.creation.box([40.0, 30.0, 6.0])
    hole = trimesh.creation.cylinder(radius=2.0, height=10.0, sections=64)
    hole.apply_translation([-13.0, -8.0, 0.0])
    pocket = trimesh.creation.box([16.0, 10.0, 4.0])
    pocket.apply_translation([4.0, 2.0, 3.0])
    return slab.difference(hole).difference(pocket)


def test_feature_remesh_keeps_the_lines_and_evens_the_triangles():
    from loadpath import subdcrease

    m = _pocket_plate()
    assert float(m.edges_unique_length.max()) > 30          # long CAD triangles
    lines, _ = subdcrease.feature_lines(m, 0.1, 1.5)
    r = subdcrease.feature_remesh(m, 0.5, 8 * 0.1 / 1.5, lines)
    assert r.is_watertight and r.euler_number == m.euler_number
    assert float(r.edges_unique_length.max()) < 1.5         # even, about the step
    # every feature line is still a chain of the remesh's own sharp edges
    # (a hole's 64-gon rim to within the sagitta of a remesh edge on it,
    # about 0.8^2 / (8 * 2) mm)
    E = r.face_adjacency_edges[r.face_adjacency_angles > 8 * 0.1 / 1.5]
    t = np.linspace(0, 1, 9)[:, None, None]
    S = (r.vertices[E[:, 0]][None] * (1 - t) + r.vertices[E[:, 1]][None] * t).reshape(-1, 3)
    from scipy.spatial import cKDTree
    D = np.vstack([subdcrease._dense(P, c, 0.05) for P, c, _ in lines])
    assert cKDTree(S).query(D)[0].max() < 0.05


def test_feature_remesh_presplit_grows_with_area_not_sliver_length_squared(monkeypatch):
    # the split ahead of the remesh is the run's memory: a CAD flat's long
    # slivers split to the remesh step go as (length / step)^2 each; split
    # to four steps the remesh still comes out even, from 1/16 the triangles
    import trimesh as tm

    from loadpath import subdcrease

    m = _pocket_plate()
    seen = []
    real = tm.remesh.subdivide_to_size

    def spy(v, f, max_edge, **kw):
        out = real(v, f, max_edge=max_edge, **kw)
        seen.append((max_edge, len(out[1])))
        return out
    monkeypatch.setattr(tm.remesh, "subdivide_to_size", spy)
    lines, _ = subdcrease.feature_lines(m, 0.1, 1.5)
    r = subdcrease.feature_remesh(m, 0.25, 8 * 0.1 / 1.5, lines)
    assert seen and abs(seen[0][0] - subdcrease.PRESPLIT * 0.25) < 1e-9
    assert seen[0][1] < 2.0 * len(r.faces)                  # the split is not the peak
    assert r.is_watertight and r.euler_number == m.euler_number
    assert float(r.edges_unique_length.max()) < 0.75


def test_lines_layout_opens_a_crease_ring_with_no_star_point():
    # a hole's rim creased right round with every vertex regular (a square
    # frame: a torus of quads, its four section corners creases) used to
    # fail the whole layout ("a line closes on itself") and send a
    # many-hole part to thousands of face-by-face patches; each ring is now
    # opened at a seam with a line drawn out from it
    from OCP.BRepCheck import BRepCheck_Analyzer

    from loadpath.subdlines import lay_out

    corners = np.array([[-2, -1], [2, -1], [2, 1], [-2, 1]], float)
    sec = np.array([corners[i] + (corners[(i + 1) % 4] - corners[i]) * j / 3 for i in range(4) for j in range(3)])
    ns, nv = len(sec), 24
    t = 2 * np.pi * np.arange(nv) / nv
    V = np.array([[(10 + x) * np.cos(a), (10 + x) * np.sin(a), z] for a in t for x, z in sec])
    Q = np.array([[j * ns + i, ((j + 1) % nv) * ns + i, ((j + 1) % nv) * ns + (i + 1) % ns, j * ns + (i + 1) % ns]
                  for j in range(nv) for i in range(ns)])
    S = np.array([[j * ns + i, ((j + 1) % nv) * ns + i] for i in range(0, ns, 3) for j in range(nv)])
    cage = subd.Cage(V, Q, sharp=S)
    H = subd.Hierarchy(cage, subd.LEVEL + 1)
    b, rep = lay_out(H, cage.V, 0.1)
    assert b is not None, rep
    assert len(b) <= 16                                     # a few rectangles, not one a face
    shape, free = subd.to_brep(b)
    assert free == 0 and BRepCheck_Analyzer(shape).IsValid()
    assert rep["patchToLimit"]["max"] < 1e-3 and rep["edgeAngle"]["max"] < 0.1


def test_feature_gap_is_the_narrowest_flat_between_creases():
    from loadpath import subdcrease

    box = trimesh.creation.box([40.0, 30.0, 6.0])
    lines, _ = subdcrease.feature_lines(box, 0.1, 2.0)
    assert abs(subdcrease.feature_gap(box, lines, 0.2, 9.0) - 6.0) < 1e-6    # the sides
    m = _pocket_plate()
    lines, _ = subdcrease.feature_lines(m, 0.1, 1.5)
    assert abs(subdcrease.feature_gap(m, lines, 0.15, 9.0) - 2.0) < 1e-6      # the pocket wall
    # a skin's two faces through its thickness are not a flat between them
    shell = trimesh.creation.box([40.0, 30.0, 10.0]).difference(
        trimesh.creation.box([36.0, 26.0, 10.0]).apply_translation([0, 0, -2.0]))
    lines, _ = subdcrease.feature_lines(shell, 0.1, 1.5)
    assert subdcrease.feature_gap(shell, lines, 0.15, 3.0) >= 2.0 - 1e-6       # the rim band, 2 mm


def test_creases_laid_on_a_cad_part_with_a_hole_and_a_pocket():
    # the long-triangle CAD mesh remeshed evenly along its lines, the hole's
    # rims traced as rings: every network laid whole
    from loadpath import subdcrease

    m = _pocket_plate()
    tol = 0.1
    lines, _ = subdcrease.feature_lines(m, tol, 2.0)
    h = (2.0 / 3.0) * subdcrease.feature_gap(m, lines, 0.2, 3.0)
    n = int(m.area / h ** 2)
    c = subd.quad_cage(m, n, crease=np.degrees(8 * tol / h))
    c, rep = subdcrease.lay_creases(c, m, tol, h=h)
    assert rep["takenLength"] >= 0.95 * rep["length"], rep["refusedAt"]
    closed = [P for P, cl, _ in subdcrease.feature_lines(m, tol, h)[0] if cl]
    assert len(closed) == 2 and not any(r["closed"] for r in rep["refusedAt"])   # both rims of the hole


def test_a_crease_corner_is_not_a_fold_and_a_flipped_face_is():
    # quads beside a crease are compared with their neighbours on their own
    # side: a creased square frame has no folds, the same frame with a
    # top control point dragged past its neighbours (its faces turned over) has
    corners = np.array([[-2, -1], [2, -1], [2, 1], [-2, 1]], float)
    sec = np.array([corners[i] + (corners[(i + 1) % 4] - corners[i]) * j / 3 for i in range(4) for j in range(3)])
    ns, nv = len(sec), 24
    t = 2 * np.pi * np.arange(nv) / nv
    V = np.array([[(10 + x) * np.cos(a), (10 + x) * np.sin(a), z] for a in t for x, z in sec])
    Q = np.array([[j * ns + i, ((j + 1) % nv) * ns + i, ((j + 1) % nv) * ns + (i + 1) % ns, j * ns + (i + 1) % ns]
                  for j in range(nv) for i in range(ns)])
    S = np.array([[j * ns + i, ((j + 1) % nv) * ns + i] for i in range(0, ns, 3) for j in range(nv)])
    cage = subd.Cage(V, Q, sharp=S)
    H = subd.Hierarchy(cage, subd.LEVEL)
    assert len(subd._folds(H, cage.V)) == 0
    C = cage.V.copy()
    C[ns * 5 + 7] += [0.0, 6.0, 0.0]
    assert len(subd._folds(H, C)) > 0


def test_a_crevice_narrower_than_the_process_prints_is_a_gap_not_a_miss():
    # a 0.1 mm slot 3 mm deep: no cage follows it and no printer opens it,
    # so the surface's miss there is reported as a closed gap
    box = trimesh.creation.box([20.0, 20.0, 10.0])
    slot = trimesh.creation.box([0.1, 30.0, 3.0]).apply_translation([0.0, 0.0, 3.6])
    m = box.difference(slot)
    cage = subd.quad_cage(m, 600)
    data = subd._samples(m, 60_000)
    check = subd._samples(m, 40_000, seed=1)
    cage, H = subd.fit_cage(cage, m, data=data)
    dev, (d1, d2, X) = subd.deviation(H, cage.V, m, data=check)
    places, kind = subd.misses(H, cage.V, m, d1, d2, X, 0.3, 2.0)
    deep = [p for p in places if p["miss"] > 1.0]
    assert deep and all(p["kind"] == "gap" and p["width"] < 0.2 for p in deep), places[:4]


def test_lines_layout_fits_round_a_crease_that_turns_at_a_regular_vertex():
    # a crease that turns a corner at a valence-4 vertex (a boss seat's rim
    # on an optimized lid): the limit round it is no bicubic, and its faces
    # were fitted as if it were (0.03-0.07 mm off, the whole lid went to
    # face-by-face patches). It is a stop: the rectangles beside it each
    # get a corner there (lines drawn out from it), knots halved toward it
    from OCP.BRepCheck import BRepCheck_Analyzer

    from loadpath.subdlines import lay_out

    corners = np.array([[-2, -1], [2, -1], [2, 1], [-2, 1]], float)
    sec = np.array([corners[i] + (corners[(i + 1) % 4] - corners[i]) * j / 3 for i in range(4) for j in range(3)])
    ns, nv = len(sec), 24
    t = 2 * np.pi * np.arange(nv) / nv
    V = np.array([[(10 + x) * np.cos(a), (10 + x) * np.sin(a), z] for a in t for x, z in sec])
    Q = np.array([[j * ns + i, ((j + 1) % nv) * ns + i, ((j + 1) % nv) * ns + (i + 1) % ns, j * ns + (i + 1) % ns]
                  for j in range(nv) for i in range(ns)])
    # a closed crease loop, a rectangle on one side of the frame turning
    # at four regular vertices (no dart: a free crease end is irregular too)
    S = [[j * ns + i, (j + 1) * ns + i] for j in range(6) for i in (0, 3)] + \
        [[j * ns + i, j * ns + i + 1] for j in (0, 6) for i in range(3)]
    cage = subd.Cage(V, Q, sharp=np.array(S))
    H = subd.Hierarchy(cage, subd.LEVEL + 1)
    b, rep = lay_out(H, cage.V, 0.1)
    assert b is not None, rep
    assert rep["patchToLimit"]["max"] < 0.1 / 3 and rep["edgeAngle"]["max"] < 0.5, rep
    shape, free = subd.to_brep(b)
    assert free == 0 and BRepCheck_Analyzer(shape).IsValid()


def test_lines_layout_fits_round_a_crease_that_ends():
    # a crease's free end (a dart, one crease edge at a regular vertex):
    # the limit round it is no bicubic, and the line used to run on
    # through it (0.076 mm a face off here; a wrist link's 0.11 sent the whole
    # part to face-by-face patches). It is a stop and a rectangle corner now
    from OCP.BRepCheck import BRepCheck_Analyzer

    from loadpath.subdlines import lay_out

    corners = np.array([[-2, -1], [2, -1], [2, 1], [-2, 1]], float)
    sec = np.array([corners[i] + (corners[(i + 1) % 4] - corners[i]) * j / 3 for i in range(4) for j in range(3)])
    ns, nv = len(sec), 24
    t = 2 * np.pi * np.arange(nv) / nv
    V = np.array([[(10 + x) * np.cos(a), (10 + x) * np.sin(a), z] for a in t for x, z in sec])
    Q = np.array([[j * ns + i, ((j + 1) % nv) * ns + i, ((j + 1) % nv) * ns + (i + 1) % ns, j * ns + (i + 1) % ns]
                  for j in range(nv) for i in range(ns)])
    S = np.array([[j * ns, (j + 1) * ns] for j in range(6)])       # along a section corner, ends both ways
    cage = subd.Cage(V, Q, sharp=S)
    H = subd.Hierarchy(cage, subd.LEVEL + 1)
    b, rep = lay_out(H, cage.V, 0.1)
    assert b is not None, rep
    assert rep["patchToLimit"]["max"] < 0.1 / 3 and rep["edgeAngle"]["max"] < 0.5, rep
    shape, free = subd.to_brep(b)
    assert free == 0 and BRepCheck_Analyzer(shape).IsValid()
