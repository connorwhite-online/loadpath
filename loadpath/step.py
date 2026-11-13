"""
STEP export: the part as a B-rep a CAD package can work with.

The part's exact pieces stay exact: every box face is a plane, every bore,
boss and bearing seat a true cylinder, every sphere a sphere, so a CAD user
can dimension, mate and drill against them. Those pieces come from the CSG
plan a part was built with (sharp.py for field graphs, reconstruct.py for
studies): the same exact shapes manifold3d used for the STL, rebuilt in
OpenCascade, and the organic leaves as faceted solids (one planar face per
triangle, decimated to a face budget first). OpenCascade's booleans then cut
the exact shapes from the facets, so the seam where a bore meets the organic
body is an exact intersection curve on the cylinder, and coplanar facets
merge into single faces.

Facets against facets (the organic body against a meshed design space, or a
kept pad that came in as a mesh) are manifold3d's work, done first: those
booleans always end and are always closed, and there is no exact face in
them to keep. OpenCascade gets the one job it is needed for, the faceted
body less the exact cutters. Each cutter is tried alone against the piece of
the body around it; the ones OpenCascade cannot cut cleanly (a counterbore
grazing a sliver) are cut by manifold3d as facets and noted, the rest are
cut exactly. Every boolean on a big faceted solid runs in a process of its
own under a clock, because some never return.

A mesh with no plan (any STL) exports fully faceted.

Needs OpenCascade's Python bindings (`pip install "loadpath[step]"`, which is
cadquery-ocp).
"""
from __future__ import annotations

import os

import numpy as np

__all__ = ["attach_plan", "available", "export_step", "plan_of", "read_step", "shape_to_trimesh"]

DEFAULT_FACES = 20_000      # organic facets in the B-rep: ~14 MB of STEP, ~20 s
FUZZY = 1e-4                # boolean fuzz (mm): facets snapped onto exact faces
POINT_DECIMALS = 5          # mm places for written points (10 nm)
DEFAULT_PATCHES = 2000      # surface="smooth": organic triangles to start the patch layout from (refined where it misses)
BOOL_SECONDS = 60.0         # one OpenCascade boolean on a big faceted solid: 5-15 s when it works; some never return
BOOL_SECONDS_PER_FACE = 2e-3   # ... plus this for each face past BOOL_INLINE (a lattice core is 500k faces)
BOOL_INLINE = 2000          # faces: below this a boolean runs in this process (it cannot be stopped there)
TOPOLOGY_TOL = 0.05         # mm: how far facets may move when a TPMS-cored body must keep every tunnel (_keep_topology)


def available():
    try:
        import OCP  # noqa: F401
        return True
    except ImportError:
        return False


def _need():
    if not available():
        raise RuntimeError('STEP export needs OpenCascade: pip install "loadpath[step]" (cadquery-ocp)')


def attach_plan(mesh, plan):
    """Remember the CSG plan a mesh was built from (an attribute, not
    metadata: metadata is copied and serialized)."""
    mesh._loadpath_csg = plan
    return mesh


def plan_of(mesh):
    return getattr(mesh, "_loadpath_csg", None)


# ---- OpenCascade shapes ------------------------------------------------------------

def _pnt(v):
    from OCP.gp import gp_Pnt
    return gp_Pnt(*(float(x) for x in v))


def _bodies(mesh):
    """`mesh` as its separate closed bodies, when it is several solid
    ones (four screw pads kept as one region); a body with a void in it
    (an inward-facing piece) stays whole."""
    if len(mesh.faces) < 8:
        return [mesh]
    parts = mesh.split(only_watertight=False)
    if len(parts) < 2 or any((not p.is_watertight) or p.volume <= 0 for p in parts):
        return [mesh]
    return list(parts)


def _solids(mesh):
    """A closed trimesh as OpenCascade solids, one for each separate body:
    one solid whose single shell holds several closed surfaces is not a
    valid solid, and every boolean it enters comes out invalid."""
    return [_faceted1(b) for b in _bodies(mesh)]


def _faceted(mesh):
    """A closed trimesh as one OpenCascade shape: a solid, or a compound of
    solids when the mesh is several bodies."""
    from OCP.BRep import BRep_Builder
    from OCP.TopoDS import TopoDS_Compound

    solids = _solids(mesh)
    if len(solids) == 1:
        return solids[0]
    b, comp = BRep_Builder(), TopoDS_Compound()
    b.MakeCompound(comp)
    for s in solids:
        b.Add(comp, s)
    return comp


def _faceted1(mesh):
    """A closed one-body trimesh as an OpenCascade solid, one planar face
    per triangle, sharing edges (BRepBuilderAPI_MakeShapeOnMesh)."""
    from OCP.BRep import BRep_Builder
    from OCP.BRepBuilderAPI import (
        BRepBuilderAPI_MakeShapeOnMesh,
        BRepBuilderAPI_MakeSolid,
    )
    from OCP.Poly import Poly_Triangle, Poly_Triangulation
    from OCP.TopAbs import TopAbs_FACE
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopoDS import TopoDS_Shell

    V = np.asarray(mesh.vertices, float)
    F = np.asarray(mesh.faces, np.int64)
    tri = Poly_Triangulation(len(V), len(F), False)
    for i, v in enumerate(V):
        tri.SetNode(i + 1, _pnt(v))
    for i, f in enumerate(F):
        tri.SetTriangle(i + 1, Poly_Triangle(int(f[0]) + 1, int(f[1]) + 1, int(f[2]) + 1))
    mk = BRepBuilderAPI_MakeShapeOnMesh(tri)
    mk.Build()
    b = BRep_Builder()
    shell = TopoDS_Shell()
    b.MakeShell(shell)
    ex = TopExp_Explorer(mk.Shape(), TopAbs_FACE)
    while ex.More():
        b.Add(shell, ex.Current())
        ex.Next()
    shell.Closed(True)
    return BRepBuilderAPI_MakeSolid(shell).Solid()


def _exact(shape, faces):
    """An exact shapes.Shape as an OpenCascade solid."""
    from OCP.BRepPrimAPI import (
        BRepPrimAPI_MakeBox,
        BRepPrimAPI_MakeCylinder,
        BRepPrimAPI_MakeSphere,
    )
    from OCP.gp import gp_Ax2, gp_Dir

    from .shapes import (
        Box,
        Cylinder,
        MeshShape,
        Shell,
        Sphere,
        Union,
        trimesh_from_manifold,
    )

    if isinstance(shape, Box):
        return BRepPrimAPI_MakeBox(_pnt(shape.lo), _pnt(shape.hi)).Shape()
    if isinstance(shape, Cylinder):
        ax = gp_Ax2(_pnt(shape.p0), gp_Dir(*(float(x) for x in shape.axis)))
        outer = BRepPrimAPI_MakeCylinder(ax, shape.r, shape.length).Shape()
        if shape.r_inner > 0:
            a = shape.axis
            ax_i = gp_Ax2(_pnt(shape.p0 - a), gp_Dir(*(float(x) for x in a)))
            inner = BRepPrimAPI_MakeCylinder(ax_i, shape.r_inner, shape.length + 2.0).Shape()
            outer = _bool("subtract", [outer, inner])
        return outer
    if isinstance(shape, Sphere):
        return BRepPrimAPI_MakeSphere(_pnt(shape.center), shape.r).Shape()
    if isinstance(shape, Union):
        return _bool("union", [_exact(s, faces) for s in shape.shapes])
    if isinstance(shape, Shell):
        man = _bool("subtract", [_exact(shape.outer, faces), _exact(shape.inner, faces)])
        return man if shape.clip is None else _bool("intersect", [man, _exact(shape.clip, faces)])
    if isinstance(shape, MeshShape):
        return _faceted(_decimate(shape.mesh, faces))
    return _faceted(_decimate(trimesh_from_manifold(shape.to_manifold()), faces))


def _bool(op, shapes):
    from OCP.BRepAlgoAPI import BRepAlgoAPI_Common, BRepAlgoAPI_Cut, BRepAlgoAPI_Fuse

    if len(shapes) == 1:
        return shapes[0]
    cls = {"union": BRepAlgoAPI_Fuse, "intersect": BRepAlgoAPI_Common, "subtract": BRepAlgoAPI_Cut}[op]
    if op == "intersect":
        out = shapes[0]
        for s in shapes[1:]:
            out = _bool1(cls, [out], [s])
        return out
    return _bool1(cls, shapes[:1], shapes[1:])


def _bool1(cls, args, tools, fuzzy=FUZZY):
    from OCP.OCP.collections import List_TopoDS_Shape

    a, t = List_TopoDS_Shape(), List_TopoDS_Shape()
    for s in args:
        a.Append(s)
    for s in tools:
        t.Append(s)
    op = cls()
    op.SetArguments(a)
    op.SetTools(t)
    op.SetFuzzyValue(fuzzy)
    op.SetRunParallel(True)
    op.SetNonDestructive(True)
    op.Build()
    if not op.IsDone():
        raise RuntimeError(f"OpenCascade {cls.__name__} failed")
    return op.Shape()


def _closest(mesh, P, chunk=20_000):
    import trimesh

    out = np.empty_like(P)
    for i in range(0, len(P), chunk):
        out[i:i + chunk] = trimesh.proximity.closest_point(mesh, P[i:i + chunk])[0]
    return out


def _sharpness(V, F):
    """Per vertex: 1 - the least cosine between its faces' normals and its
    own normal (0 on a flat patch, ~1 on a right-angle crease)."""
    n = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-300)
    vn = np.zeros_like(V)
    for c in range(3):
        np.add.at(vn, F[:, c], n)
    vn /= np.maximum(np.linalg.norm(vn, axis=1, keepdims=True), 1e-300)
    k = np.zeros(len(V))
    for c in range(3):
        np.maximum.at(k, F[:, c], 1 - np.einsum("ij,ij->i", n, vn[F[:, c]]))
    return k


def _collapse(V, F, target, locked=None, turn=0.9):
    """A closed mesh (V, F) coarsened to about `target` faces, shortest
    edges first, keeping its topology: decimate.py's passes (the link
    condition, no face turning by more than acos(`turn`)) with no field to
    hold to. Each collapse keeps the sharper end where it is, so creases
    and seams stay put and every vertex is one of the original ones.
    Vertices flagged in `locked` never move. Returns the vertices, faces and
    lock flags of the result."""
    from .decimate import decimate

    V = np.asarray(V, float)
    F = np.asarray(F, np.int64)
    k = _sharpness(V, F)

    def place(W, a, b):
        pick = k[a] >= k[b]
        k[a] = np.maximum(k[a], k[b])
        return np.where(pick[:, None], W[a], W[b])

    lock = np.zeros(len(V), bool) if locked is None else np.asarray(locked, bool)
    V2, F2, _, kept = decimate(V, F, None, 0.0, 1.0, locked=lock, max_passes=300, min_gain=0.002,
                               target=int(target), place=place, turn=turn)
    return V2, F2, lock[kept]


def _fast(mesh, faces):
    """fast_simplification's quadric decimation, without the zero-area
    faces (a repeated vertex) it leaves behind, which make an otherwise
    closed result read as torn."""
    import trimesh

    c = mesh.simplify_quadric_decimation(face_count=int(faces))
    F = np.asarray(c.faces)
    ok = (F[:, 0] != F[:, 1]) & (F[:, 1] != F[:, 2]) & (F[:, 0] != F[:, 2])
    if ok.all():
        return c
    out = trimesh.Trimesh(c.vertices, F[ok], process=False)
    out.remove_unreferenced_vertices()
    return out


def _decimate(mesh, faces):
    """`mesh` cut down to about `faces` triangles, still closed. A lattice
    or TPMS core does not decimate to a budget (its walls collapse first):
    it is simplified within TOPOLOGY_TOL instead, every tunnel kept, over the
    budget up to 25x; past that (or not a manifold) it is an error, not a
    STEP file of millions of faces."""
    if len(mesh.faces) <= faces:
        return mesh
    try:
        out = _flip_needles(_drop_specks(_fast(mesh, faces)))
        if (out.is_watertight and abs(out.volume) > 0.9 * abs(mesh.volume)
                and out.euler_number == mesh.euler_number       # no tunnel closed, no sliver body left
                and _deviation(mesh, out, 2000)["mean"] <= 4 * TOPOLOGY_TOL):   # nor walls folded flat
            return out
    except Exception:  # noqa: BLE001 - decimation is best-effort
        pass
    import trimesh

    kept = _keep_topology(mesh, faces)         # seconds; within TOPOLOGY_TOL of the surface
    if kept is not None and len(kept.faces) <= 1.5 * faces:
        return kept
    if kept is None and len(mesh.faces) <= 20 * faces and mesh.is_watertight and mesh.is_winding_consistent:
        V, F, _ = _collapse(mesh.vertices, mesh.faces, int(faces))      # tore: coarsen carefully
        out = trimesh.Trimesh(V, F, process=False)
        if len(out.faces) <= 1.5 * faces and out.is_watertight:
            return out
    if kept is not None:
        # a TPMS core's tunnels: over the budget rather than closed up or
        # torn. Quadric decimation from the simplified surface, at twice the
        # budget and up, often keeps them all in fewer faces than the
        # tolerance alone leaves (the upper arm link: 368k -> 160k).
        n = 2 * faces
        while n < 0.6 * len(kept.faces):
            try:
                out = _flip_needles(_drop_specks(_fast(kept, n)))
                if (out.is_watertight and out.euler_number == kept.euler_number
                        and abs(out.volume) > 0.995 * abs(kept.volume)):
                    return out
            except Exception:  # noqa: BLE001 - best-effort
                pass
            n *= 2
        return kept
    if len(mesh.faces) > 4 * faces:
        raise ValueError(f"an organic region of {len(mesh.faces)} triangles does not decimate to {faces} and stay "
                         f"closed within {TOPOLOGY_TOL} mm (a lattice?): raise the STEP face budget, or hand over "
                         "the STL for it")
    return mesh


def _flip_needles(mesh, rel=1e-10):
    """`mesh` with its zero-area triangles (three vertices on a line, which
    quadric decimation can leave) flipped away: the edge opposite the
    middle vertex is swapped with its neighbour's, so no face is flat to
    nothing. OpenCascade reads a solid with one such face as invalid, and
    ShapeFix then leaves a shell, not a solid."""
    import trimesh

    F = np.array(mesh.faces, np.int64)
    V = np.asarray(mesh.vertices, float)

    def areas(idx):
        t = V[F[idx]]
        return np.linalg.norm(np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0]), axis=1) / 2

    a = areas(np.arange(len(F)))
    eps = rel * max(float(np.median(a)), 1e-12) * 1e4
    t = V[F]
    L = np.linalg.norm(t[:, [1, 2, 0]] - t, axis=2).max(1)
    bad = np.flatnonzero((a <= eps) | (a <= 1e-4 * L * L))       # flat to nothing, or a cap (angle ~180)
    if not len(bad):
        return mesh
    edge_face = {}
    for i, f in enumerate(F):
        for k in range(3):
            edge_face[(int(f[k]), int(f[(k + 1) % 3]))] = i
    for i in bad:
        f = F[i]
        L = [np.linalg.norm(V[f[(k + 1) % 3]] - V[f[k]]) for k in range(3)]
        k = int(np.argmax(L))
        a_, b_, c_ = int(f[k]), int(f[(k + 1) % 3]), int(f[(k + 2) % 3])
        j = edge_face.get((b_, a_))
        if j is None:
            continue
        g = [int(x) for x in F[j]]
        d_ = [x for x in g if x not in (a_, b_)]
        if len(d_) != 1 or d_[0] == c_:
            continue
        d_ = d_[0]
        if (c_, d_) in edge_face or (d_, c_) in edge_face:       # the flip would make an edge twice
            continue
        new = np.array([[c_, a_, d_], [c_, d_, b_]])
        old = F[[i, j]].copy()
        F[i], F[j] = new
        tt = V[F[[i, j]]]
        if (areas(np.array([i, j])) <= np.maximum(eps, 1e-4 * np.linalg.norm(tt[:, [1, 2, 0]] - tt, axis=2).max(1)
                                                   ** 2)).any():
            F[i], F[j] = old
            continue
        for e in ((a_, b_), (b_, a_)):
            edge_face.pop(e, None)
        for idx in (i, j):
            for k in range(3):
                edge_face[(int(F[idx][k]), int(F[idx][(k + 1) % 3]))] = idx
    return trimesh.Trimesh(V, F, process=False)


def _collapse_short(mesh, length):
    """`mesh` with its edges shorter than `length` collapsed to their
    midpoints (the two triangles on each go), where that keeps the surface
    a closed manifold (the ends share exactly the two corners opposite the
    edge). A boolean of near-coincident surfaces leaves edges of 1e-7 mm,
    whose ends OpenCascade takes for one point."""
    import trimesh

    V = np.array(mesh.vertices, float)
    F = np.array(mesh.faces, np.int64)
    T = V[F]
    L = np.linalg.norm(T[:, [1, 2, 0]] - T, axis=2)
    if L.min() >= length:
        return mesh
    alive = np.ones(len(F), bool)
    vf = {}
    for i, f in enumerate(F):
        for v in f:
            vf.setdefault(int(v), set()).add(i)
    short = np.argwhere(L < length)
    for i, k in short[np.argsort(L[short[:, 0], short[:, 1]])]:
        if not alive[i]:
            continue
        a, b = int(F[i][k]), int(F[i][(k + 1) % 3])
        if a == b or np.linalg.norm(V[a] - V[b]) >= length:
            continue
        both = [t for t in vf[a] & vf[b] if alive[t]]
        na = {int(x) for t in vf[a] if alive[t] for x in F[t]} - {a, b}
        nb = {int(x) for t in vf[b] if alive[t] for x in F[t]} - {a, b}
        opp = {int(x) for t in both for x in F[t]} - {a, b}
        if len(both) != 2 or na & nb != opp:
            continue
        V[a] = 0.5 * (V[a] + V[b])
        for t in both:
            alive[t] = False
        for t in vf[b]:
            if alive[t]:
                F[t][F[t] == b] = a
                vf[a].add(t)
        vf[b] = set()
    out = trimesh.Trimesh(V, F[alive], process=False)
    out.remove_unreferenced_vertices()
    return out if out.is_watertight and out.is_winding_consistent else mesh


def _split_needles(mesh, rel=1e-9):
    """`mesh` without triangles whose corners lie on a line (a needle the
    flips of _flip_needles could not remove): the neighbour across the long
    edge is split at the middle corner and the needle dropped, so the
    surface stays closed. OpenCascade's mesh-to-shape leaves such a face
    out, and the shell then has a slit."""
    import trimesh

    F = [list(map(int, f)) for f in np.asarray(mesh.faces)]
    V = np.asarray(mesh.vertices, float)
    for _ in range(8):
        T = V[np.array(F)]
        L = np.linalg.norm(T[:, [1, 2, 0]] - T, axis=2)
        a = np.linalg.norm(np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0]), axis=1) / 2
        bad = set(np.flatnonzero(a <= rel * L.max(1) ** 2).tolist())
        if not bad:
            break
        owner = {}
        for i, f in enumerate(F):
            for k in range(3):
                owner[(f[k], f[(k + 1) % 3])] = i
        gone, out = set(), []
        for i in sorted(bad):
            if i in gone:
                continue
            f = F[i]
            k = int(np.argmax(L[i]))                 # long edge f[k] -> f[k+1], middle corner f[k+2]
            p, q, m = f[k], f[(k + 1) % 3], f[(k + 2) % 3]
            j = owner.get((q, p))
            if j is None or j in gone or j in bad:
                continue
            g = F[j]
            d = [x for x in g if x not in (p, q)]
            if len(d) != 1:
                continue
            d = d[0]
            gone.update((i, j))
            out += [[q, m, d], [m, p, d]]            # the neighbour (q, p, d) split at m
        if not gone:
            break
        F = [f for i, f in enumerate(F) if i not in gone] + out
    return trimesh.Trimesh(V, np.array(F, np.int64), process=False)


def _drop_specks(mesh, share=1e-5):
    """`mesh` without the bodies under `share` of its volume: the folded
    zero-volume slivers decimation leaves, each a solid of its own in a
    STEP file."""
    import trimesh

    lab = trimesh.graph.connected_component_labels(mesh.face_adjacency, node_count=len(mesh.faces))
    if lab.max() == 0:
        return mesh
    F = np.asarray(mesh.faces)
    tri = mesh.vertices[F]
    v = np.einsum("ij,ij->i", tri[:, 0], np.cross(tri[:, 1], tri[:, 2])) / 6.0
    vol = np.bincount(lab, weights=v)
    keep = np.abs(vol) >= share * abs(vol.sum())
    if keep.all():
        return mesh
    out = trimesh.Trimesh(mesh.vertices, F[keep[lab]], process=False)
    out.remove_unreferenced_vertices()
    return out


def _keep_topology(mesh, faces, tol=None, limit=25):
    """`mesh` simplified by manifold3d (edge collapses that keep it
    manifold, every vertex within `tol` mm of the surface) at the finest
    tolerance that meets the `faces` budget, else at `tol` whatever it
    costs: a part with a gyroid or other TPMS core open to its outside (a
    skinned, drained link) is one surface with tens of tunnels, which
    quadric decimation tears and edge collapse to the budget closes. Specks
    the collapses leave (under 1e-5 of the volume) are dropped. None when
    even `tol` leaves more than `limit` x `faces` triangles."""
    import manifold3d as m3d
    import trimesh

    from .shapes import manifold_from_trimesh, trimesh_from_manifold

    tol = TOPOLOGY_TOL if tol is None else float(tol)
    try:
        M = manifold_from_trimesh(mesh)
    except Exception:  # noqa: BLE001 - not a manifold: nothing to keep
        return None
    if M.status() != m3d.Error.NoError or M.is_empty():
        return None
    best = None
    for t in (tol / 16, tol / 8, tol / 4, tol / 2, tol):
        best = M.simplify(t)
        if best.num_tri() <= faces:
            break
    if best.num_tri() > limit * faces:
        return None
    pieces = best.decompose()
    if len(pieces) > 1:
        total = sum(p.volume() for p in pieces)
        keep = [p for p in pieces if p.volume() >= 1e-5 * total]
        best = m3d.Manifold.batch_boolean(keep, m3d.OpType.Add) if len(keep) > 1 else keep[0]
    out = trimesh_from_manifold(best)
    out = _flip_needles(trimesh.Trimesh(np.array(out.vertices, np.float64), np.array(out.faces, np.int64),
                                        process=False))
    return out if out.is_watertight else None


def _organic_budget(plan, faces):
    from .sharp import Organic, _leaves

    org = [lf for lf in _leaves(plan) if isinstance(lf, Organic) and lf.mesh is not None]
    for lf in org:
        if not hasattr(lf.mesh, "faces"):          # a manifold (reconstruct.py)
            from .shapes import trimesh_from_manifold

            lf.mesh = trimesh_from_manifold(lf.mesh)
    total = sum(len(lf.mesh.faces) for lf in org) or 1
    return {id(lf): max(500, int(faces * len(lf.mesh.faces) / total)) for lf in org}


def _pieces(shape, faces):
    """An exact shape as separate OpenCascade solids (a union's members):
    booleans take them as separate tools, which survives members that share
    faces (a motor box against its driver's box) where a pre-fused union
    can come out invalid."""
    from .shapes import MeshShape, Union

    if isinstance(shape, Union):
        return [p for s in shape.shapes for p in _pieces(s, faces)]
    if isinstance(shape, MeshShape):
        return _solids(_decimate(shape.mesh, faces))
    return [_exact(shape, faces)]


def _on_float32(mesh):
    """The mesh with its vertices on float32's grid. manifold3d's results
    come back in double precision now (shapes.trimesh_from_manifold), and a
    study part's organic leaf, decimated from those, leaves OpenCascade's
    booleans against the exact planes with no clean result (the lug's
    STEP came out with no solid); rounded as manifold3d's float32 output
    used to round them, they fuse. Moves a vertex ~1e-5 mm at most."""
    import trimesh

    V = np.asarray(mesh.vertices, np.float32).astype(float)
    if np.array_equal(V, mesh.vertices):
        return mesh
    return trimesh.Trimesh(V, np.asarray(mesh.faces), process=False)


def _build(plan, budget, faces, notes):
    """(OpenCascade solids, the same thing as a manifold3d Manifold). The
    manifold is the referee: each OpenCascade boolean is tried a few ways
    (fuzz, tools separate or fused) until it is valid and its volume agrees
    with manifold3d's."""
    import manifold3d as m3d

    from .shapes import manifold_from_trimesh
    from .sharp import Exact, Organic, exact_manifold

    if isinstance(plan, Exact):
        return _pieces(plan.shape, faces), exact_manifold(plan.shape)
    if isinstance(plan, Organic):
        m = _on_float32(plan.mesh)
        d = _decimate(m, budget.get(id(plan), faces))
        if len(d.faces) < len(m.faces):
            notes.append({"leaf": ",".join(plan.names) or "organic", "faces": len(m.faces),
                          "stepFaces": len(d.faces), "deviation": _deviation(m, d)})
        return [_faceted(d)], manifold_from_trimesh(d)
    kids = [_build(k, budget, faces, notes) for k in plan.kids]
    kind = {"union": m3d.OpType.Add, "intersect": m3d.OpType.Intersect, "subtract": m3d.OpType.Subtract}[plan.op]
    want = m3d.Manifold.batch_boolean([k[1] for k in kids], kind)
    if plan.op == "union":
        first, tools = kids[0][0][:1], kids[0][0][1:] + [s for k in kids[1:] for s in k[0]]
    else:
        first = [_fuse_all(kids[0][0], kids[0][1])]
        tools = [s for k in kids[1:] for s in k[0]]
        if plan.op == "intersect":
            tools = [_fuse_all(tools, m3d.Manifold.batch_boolean([k[1] for k in kids[1:]], m3d.OpType.Add))]
    shape = _referee(plan.op, first, tools, want, notes)
    return [shape], want


def _primitive(shape):
    from .shapes import Box, Cylinder, Shell, Sphere

    return isinstance(shape, (Box, Cylinder, Sphere, Shell))


def _flat(shape):
    from .shapes import Union

    if isinstance(shape, Union):
        return [p for s in shape.shapes for p in _flat(s)]
    return [shape]


OVERLAP = 2e-3      # of the smaller volume: a cutter and a kept mesh that share less than this only touch
                    # (a drawn bore's facets against its exact cylinder)


def _normal(plan, budget, faces, notes):
    """The plan as (a faceted body, exact cutters): body - cutters, each
    cutter (its OpenCascade solids, its manifold). Facets against facets
    (the organic body, a meshed design space, kept regions that came in as
    meshes) are manifold3d's work, which always ends and is always closed;
    OpenCascade is left the one boolean it is needed for and does well, the
    faceted body less the exact cylinders, boxes and spheres. Its booleans
    between two faceted solids whose faces lie on each other (the body was
    clipped to a kept pad, then the pad unioned back) come out empty, or
    invalid, or never come out. None when the plan has an exact shape as a
    positive body (a box envelope): that goes the general way (_build)."""
    import manifold3d as m3d

    from .shapes import manifold_from_trimesh
    from .sharp import Exact, Organic, exact_manifold

    add = m3d.OpType.Add
    if isinstance(plan, Organic):
        m = _on_float32(plan.mesh)
        d = _decimate(m, budget.get(id(plan), faces))
        if len(d.faces) < len(m.faces):
            notes.append({"leaf": ",".join(plan.names) or "organic", "faces": len(m.faces),
                          "stepFaces": len(d.faces), "deviation": _deviation(m, d)})
        return manifold_from_trimesh(d), []
    if isinstance(plan, Exact):
        pieces = _flat(plan.shape)
        if any(_primitive(q) for q in pieces):
            return None
        return m3d.Manifold.batch_boolean([exact_manifold(q) for q in pieces], add), []
    first = _normal(plan.kids[0], budget, faces, notes)
    if first is None:
        return None
    body, tools = first[0], list(first[1])
    for kid in plan.kids[1:]:
        if plan.op == "subtract":
            if not isinstance(kid, Exact):
                return None
            for q in _flat(kid.shape):
                if _primitive(q):
                    tools.append(([_exact(q, faces)], exact_manifold(q), q))
                else:
                    body = body - exact_manifold(q)
            continue
        other = _normal(kid, budget, faces, notes)
        if other is None:
            return None
        if plan.op == "intersect":            # (A - T) & (B - U) = (A & B) - T - U
            body, tools = body ^ other[0], tools + list(other[1])
            continue
        if other[1]:                          # union: (A - T) | B = (A | B) - (T - B)
            return None
        kept = []
        for occ, man, q in tools:
            shared = (man ^ other[0]).volume()
            if shared > OVERLAP * min(man.volume(), other[0].volume()):
                less = man - other[0]
                if less.is_empty():
                    continue
                pads = [x for b in other[0].decompose() for x in _solids(_on_float32(_mesh_of(b)))]
                occ = [_referee("subtract", [_fuse_all(occ)], pads, less, notes)]
                man = less
            kept.append((occ, man, q))
        body, tools = body + other[0], kept
    return body, tools


def _mesh_of(man):
    import trimesh

    from .shapes import trimesh_from_manifold

    out = trimesh_from_manifold(man)
    return trimesh.Trimesh(np.array(out.vertices, np.float64), np.array(out.faces, np.int64), process=False)


def _build_normal(plan, budget, faces, notes):
    """The OpenCascade shape of a plan in normal form (_normal), or None."""
    got = _normal(plan, budget, faces, notes)
    if got is None or got[0].is_empty():
        return None
    body, tools = got
    clean = body.simplify(1e-4)               # slivers the booleans left: each would be a degenerate face
    if not clean.is_empty():
        body = clean
    tools = [t for t in tools if (t[1] ^ body).volume() > 1e-9]
    if not tools:
        return _body_shape(body)
    try:
        return _cut_on_mesh(body, tools, notes)
    except RuntimeError as e:
        notes.append({"warning": f"exact faces on the finished mesh failed ({e}); cut by OpenCascade instead"})
    return _cut_exact(body, [t[:2] for t in tools], notes)


LABEL_TOL = 0.005     # mm: a triangle with its corners this close to a cutter's cylinder lies on it (brep.TOL)
WELD = 2e-3           # mm: edges the cut leaves shorter than this collapse (slivers where a drawn hole meets its cutter)
EDGE_TOL = 1e-3       # mm: a seam edge's chord against the cylinder (CHORD_TOL sag, welded)


def _cut_on_mesh(body, tools, notes):
    """The manifold `body` less the exact cutters `tools`, as an OpenCascade
    solid built from the finished mesh: no OpenCascade boolean at all.

    manifold3d makes the cut (body less each cutter at CHORD_TOL, 2 um, the
    same mesh the STL is). Each connected stretch of the result lying on a
    cutter's cylinder (a bore wall, a counterbore) becomes one face on that
    exact cylinder, bounded by its seam edges with the facets next to it;
    every other triangle is a planar face sharing those edges, and coplanar
    ones merge into planes afterwards (UnifySameDomain). The geometry is the
    mesh's, so the solid is the STL's to within the chord, whatever the
    body looks like near the cutter. OpenCascade's own cut of a faceted body
    by an exact cylinder fails where the body's facets lie on the cutter (a
    drawn pad's screw hole drilled by its own screw's keep-out, a counterbore
    floor on the pad's top, a drive clearance coaxial with the counterbore):
    which of them failed changed with the optimized body, so how many bores
    came out exact changed from machine to machine. Raises RuntimeError when
    the result is not valid solids of the cut's volume."""
    import manifold3d as m3d

    from .shapes import Box
    from .sharp import CHORD_TOL, exact_manifold

    # the cutters a quarter of the STL's chord: a seam edge then sags under
    # the 2 um a fine triangulation of the file is read at (shape_to_trimesh),
    # so it meets the facets beside it node for node. A cutter trimmed by a
    # kept pad (_normal's union) stays as it is.
    cut = []
    for t in tools:
        man = t[1]
        if not isinstance(t[2], Box):
            fine = exact_manifold(t[2], CHORD_TOL / 4)
            if abs(exact_manifold(t[2]).volume() - man.volume()) <= 1e-9 * max(man.volume(), 1.0):
                man = fine
        cut.append(man)
    part = body - m3d.Manifold.batch_boolean(cut, m3d.OpType.Add)
    clean = part.simplify(WELD)
    if not clean.is_empty():
        part = clean
    pieces = part.decompose()
    if len(pieces) > 1:                   # zero-volume specks the cut leaves, each a solid of its own
        total = sum(abs(p.volume()) for p in pieces)
        pieces = [p for p in pieces if abs(p.volume()) >= 1e-5 * total]
        part = m3d.Manifold.batch_boolean(pieces, m3d.OpType.Add) if len(pieces) > 1 else pieces[0]
    mesh = _split_needles(_flip_needles(_collapse_short(_mesh_of(part), WELD)))
    return _mesh_solid(mesh, [t[2] for t in tools], part.volume(), notes)


def _mesh_solid(mesh, cutters, want, notes):
    """The closed `mesh` (`want` mm^3) as an OpenCascade solid, a compound
    of solids if it is several bodies: its triangles lying on the
    `cutters`' cylinders as faces on those cylinders, the rest planar
    facets (_cut_on_mesh)."""
    import scipy.sparse as sp
    import trimesh
    from OCP.BRep import BRep_Builder, BRep_Tool
    from OCP.BRepBuilderAPI import (
        BRepBuilderAPI_MakeShapeOnMesh,
        BRepBuilderAPI_MakeSolid,
    )
    from OCP.BRepCheck import BRepCheck_Analyzer
    from OCP.Poly import Poly_Triangle, Poly_Triangulation
    from OCP.ShapeFix import ShapeFix_Solid
    from OCP.TopAbs import TopAbs_FACE, TopAbs_VERTEX
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopoDS import TopoDS, TopoDS_Shell
    from scipy.sparse.csgraph import connected_components

    from .brep import ISLAND, _shape_surfaces, _uv, islands, label

    if not len(mesh.faces) or not mesh.is_watertight:
        raise RuntimeError("the cut mesh is not closed")
    surfs = []
    for q in cutters:
        _shape_surfaces(q, surfs)
    surfs = [s for s in surfs if s.kind == "cylinder"]
    keys, geo = {}, {}
    for k, s in enumerate(surfs):
        lk = keys.setdefault(s.key(), k + 1)
        geo.setdefault(lk, s)
    lab = label(mesh, surfs, LABEL_TOL) if surfs else np.zeros(len(mesh.faces), np.int64)
    lab = islands(mesh, lab, ISLAND, geo, tol=LABEL_TOL)
    V = np.asarray(mesh.vertices, float)
    F = np.asarray(mesh.faces, np.int64)

    # every triangle a planar face, all sharing their edges and vertices
    tri = Poly_Triangulation(len(V), len(F), False)
    for i, v in enumerate(V):
        tri.SetNode(i + 1, _pnt(v))
    for i, f in enumerate(F):
        tri.SetTriangle(i + 1, Poly_Triangle(int(f[0]) + 1, int(f[1]) + 1, int(f[2]) + 1))
    mk = BRepBuilderAPI_MakeShapeOnMesh(tri)
    mk.Build()
    # each face's triangle, by its corners (MakeShapeOnMesh leaves out a
    # triangle with no area: three corners on a line)
    index = {tuple(v): i for i, v in enumerate(V.tolist())}
    tri_index = {tuple(sorted(f)): i for i, f in enumerate(F.tolist())}
    faces = [None] * len(F)
    ex = TopExp_Explorer(mk.Shape(), TopAbs_FACE)
    while ex.More():
        fc = TopoDS.Face(ex.Current())
        got = set()
        ee = TopExp_Explorer(fc, TopAbs_VERTEX)
        while ee.More():
            got.add(index.get(BRep_Tool.Pnt_s(TopoDS.Vertex(ee.Current())).Coord()))
            ee.Next()
        i = tri_index.get(tuple(sorted(got))) if None not in got else None
        if i is None:
            raise RuntimeError("a face is not one of the mesh's triangles")
        faces[i] = fc
        ex.Next()

    def ends(e):
        a = BRep_Tool.Pnt_s(TopoDS.Vertex(_first_vertex(e)))
        b = BRep_Tool.Pnt_s(TopoDS.Vertex(_last_vertex(e)))
        return np.array([a.X(), a.Y(), a.Z()]), np.array([b.X(), b.Y(), b.Z()])

    A = np.asarray(mesh.face_adjacency)
    same = (lab[A[:, 0]] == lab[A[:, 1]]) & (lab[A[:, 0]] > 0)
    keep = np.ones(len(F), bool)
    region_faces, region_tri, fell = [], [], []
    # each stretch in sectors of a third of a turn, so no face wraps the
    # cylinder (a ring round a bore closes in 3-D but not in its (u, v): it
    # would need a seam edge); UnifySameDomain joins the sectors again
    C = V[F].mean(axis=1)
    sector = np.full(len(F), -1)
    for lk, s in geo.items():
        on = lab == lk
        u = _uv(s, C[on])[:, 0]
        sector[on] = np.floor((u + np.pi) / (2 * np.pi / 3)).clip(0, 2).astype(int)
    same = same & (sector[A[:, 0]] == sector[A[:, 1]])
    G = sp.coo_matrix((np.ones(same.sum()), (A[same, 0], A[same, 1])), shape=(len(F), len(F)))
    _, comp = connected_components(G, directed=False)
    comp = np.where(lab > 0, comp, -1)
    work = [np.flatnonzero(comp == c) for c in np.unique(comp[comp >= 0])]
    tries = 0
    while work:
        tris = work.pop()
        s = geo[int(lab[tris[0]])]
        try:
            region_faces.append(_cylinder_region(s, tris, V, F, faces, ends))
            region_tri.append(int(tris[0]))
            keep[tris] = False
        except _Crossing as e:
            # the triangles at the crossing stay facets, the rest is tried again
            tries += 1
            rest = tris[~np.isin(F[tris], e.verts).any(axis=1)]
            if not e.verts or tries > 200 or len(rest) == len(tris):
                fell.append(f"{len(tris)} triangles: {e}")
                continue
            if len(rest):
                sub = np.isin(A, rest).all(axis=1) & same
                G = sp.coo_matrix((np.ones(sub.sum()), (A[sub, 0], A[sub, 1])), shape=(len(F), len(F)))
                _, cc = connected_components(G, directed=False)
                work += [rest[cc[rest] == k] for k in np.unique(cc[rest])]
        except RuntimeError as e:
            fell.append(f"{len(tris)} triangles: {e}")
    # one solid per separate body: one shell holding several closed
    # surfaces is not a valid solid (fragments are dropped by the caller)
    body_of = trimesh.graph.connected_component_labels(A, node_count=len(F))
    groups = {}
    for i in np.flatnonzero(keep):
        if faces[i] is not None:
            groups.setdefault(int(body_of[i]), []).append(faces[i])
    for f, t in zip(region_faces, region_tri):
        groups.setdefault(int(body_of[t]), []).append(f)
    bb = BRep_Builder()
    solids = []
    for k in sorted(groups):
        shell = TopoDS_Shell()
        bb.MakeShell(shell)
        for f in groups[k]:
            bb.Add(shell, f)
        shell.Closed(True)
        sol = BRepBuilderAPI_MakeSolid(shell).Solid()
        if not BRepCheck_Analyzer(sol).IsValid():
            fx = ShapeFix_Solid(sol)
            fx.Perform()
            sol = fx.Solid()
        if not BRepCheck_Analyzer(sol).IsValid():
            raise RuntimeError(f"the solid from the cut mesh is not valid (body {k}, {len(groups[k])} faces)")
        solids.append(sol)
    if len(solids) == 1:
        solid = solids[0]
    else:
        from OCP.TopoDS import TopoDS_Compound

        solid = TopoDS_Compound()
        bb.MakeCompound(solid)
        for sol in solids:
            bb.Add(solid, sol)
    if abs(_volume(solid) - want) > 1e-3 * want + 1e-6:
        raise RuntimeError(f"the solid's volume {_volume(solid):.3f} is not the cut mesh's {want:.3f}")
    if fell:
        notes.append({"warning": f"{len(fell)} stretch(es) of cutter surface would not make a face on the "
                                 "exact cylinder and are facets in the file", "why": fell[:5]})
    return solid


def _first_vertex(e):
    from OCP.TopExp import TopExp
    from OCP.TopoDS import TopoDS

    return TopExp.FirstVertex_s(TopoDS.Edge(e))


def _last_vertex(e):
    from OCP.TopExp import TopExp
    from OCP.TopoDS import TopoDS

    return TopExp.LastVertex_s(TopoDS.Edge(e))


def _cylinder_region(s, tris, Vx, F, faces, ends):
    """One face on the exact cylinder `s` over the triangles `tris`, bounded
    by the edges those triangles share with the rest (the edges of `faces`,
    the triangles' own planar faces, so the solid stays one piece). Each
    edge gets a line in the cylinder's (u, v) between its ends, the angle
    unwrapped along its loop (the caller hands over pieces under a third of
    a turn, so no loop wraps the cylinder and no seam edge is needed).
    RuntimeError unless the face is valid; _Crossing when its boundary
    crosses itself, naming the corners where."""
    from OCP.BRep import BRep_Builder, BRep_Tool
    from OCP.BRepCheck import BRepCheck_Analyzer
    from OCP.BRepLib import BRepLib
    from OCP.Geom2d import Geom2d_BSplineCurve
    from OCP.gp import gp_Pnt2d
    from OCP.OCP.collections import Array1_double, Array1_gp_Pnt2d, Array1_int
    from OCP.ShapeFix import ShapeFix_Face
    from OCP.TopAbs import TopAbs_EDGE, TopAbs_FORWARD, TopAbs_REVERSED
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopoDS import TopoDS, TopoDS_Face, TopoDS_Wire

    from .brep import _uv

    tri_of = {}
    for t in tris:
        f = [int(x) for x in F[t]]
        for k in range(3):
            tri_of[(f[k], f[(k + 1) % 3])] = int(t)
    # boundary half-edges a -> b (counterclockwise about the part's outside)
    nxt = {}
    for (a, b_), t in tri_of.items():
        if (b_, a) not in tri_of:
            nxt.setdefault(a, []).append(b_)
    if not nxt:
        raise RuntimeError("no boundary")
    # the triangles' own edges, by their ends
    edge = {}
    for t in tris:
        f = [int(x) for x in F[int(t)]]
        if faces[int(t)] is None:
            continue
        ee = TopExp_Explorer(faces[int(t)], TopAbs_EDGE)
        while ee.More():
            e = TopoDS.Edge(ee.Current().Oriented(TopAbs_FORWARD))
            i, j = (f[int(np.argmin(np.linalg.norm(Vx[f] - p, axis=1)))] for p in ends(e))
            edge[(i, j)] = (e, False)
            edge[(j, i)] = (e, True)
            ee.Next()
    loops = []
    while nxt:
        a0 = next(iter(nxt))
        loop, a = [], a0
        while True:
            outs = nxt.get(a)
            if not outs:
                raise RuntimeError("boundary does not close")
            b_ = outs.pop()
            if not outs:
                del nxt[a]
            loop.append((a, b_))
            a = b_
            if a == a0:
                break
        loops.append(loop)
    T = Vx[F[tris]]
    tn = np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0])
    # the part's outside against the surface's own normal (a bore): the loops
    # run the other way round the surface, and the face is reversed
    flip = float(np.einsum("ij,ij->", tn, s.normal(T.mean(axis=1)))) < 0
    if flip:
        loops = [[(b_, a) for a, b_ in reversed(lp)] for lp in loops]
    geom = s.geom()
    bb = BRep_Builder()
    face = TopoDS_Face()
    bb.MakeFace(face, geom, 1e-7)
    ref = None
    for loop in sorted(loops, key=len, reverse=True):
        P = []
        for a, b_ in loop:
            if (a, b_) not in edge:
                raise RuntimeError("a seam edge has no face")
            e, rev = edge[(a, b_)]
            p = BRep_Tool.Pnt_s(TopoDS.Vertex(_first_vertex(e) if not rev else _last_vertex(e)))
            P.append((p.X(), p.Y(), p.Z()))
        P = np.array(P + [P[0]])
        UV = _uv(s, P)
        UV[:, 0] = np.unwrap(UV[:, 0])
        if abs(UV[-1, 0] - UV[0, 0]) > np.pi:
            raise RuntimeError("a loop wraps the cylinder")
        if ref is None:
            ref = float(UV[:, 0].mean())
        UV[:, 0] += 2 * np.pi * np.round((ref - UV[:, 0].mean()) / (2 * np.pi))   # holes inside the outer loop
        wire = TopoDS_Wire()
        bb.MakeWire(wire)
        for k, (a, b_) in enumerate(loop):
            e, rev = edge[(a, b_)]
            p, q = UV[k], UV[k + 1]
            if rev:
                p, q = q, p
            d = q - p
            L = float(np.hypot(*d))
            if L < 1e-12:
                raise RuntimeError("a degenerate seam edge")
            c0, c1 = BRep_Tool.Range_s(e)
            poles = Array1_gp_Pnt2d(1, 2)
            poles.SetValue(1, gp_Pnt2d(float(p[0]), float(p[1])))
            poles.SetValue(2, gp_Pnt2d(float(q[0]), float(q[1])))
            knots, mults = Array1_double(1, 2), Array1_int(1, 2)
            knots.SetValue(1, float(c0))
            knots.SetValue(2, float(c1))
            mults.SetValue(1, 2)
            mults.SetValue(2, 2)
            pc = Geom2d_BSplineCurve(poles, knots, mults, 1)
            bb.UpdateEdge(e, pc, face, EDGE_TOL)
            bb.Add(wire, e.Oriented(TopAbs_REVERSED) if rev else e)
        bb.Add(face, wire)
    BRepLib.SameParameter_s(face, EDGE_TOL)
    fx = ShapeFix_Face(face)
    fx.Perform()
    out = fx.Face()
    if not BRepCheck_Analyzer(out).IsValid():
        raise _crossing(out, Vx, np.unique(F[tris]), ends)
    return out.Reversed() if flip else out


class _Crossing(RuntimeError):
    """A cylinder face whose boundary crosses itself in (u, v) within the
    edges' tolerance: two seam edges a micron apart, where the cut's slivers
    meet (a drawn hole's rim on the counterbore floor). `verts` are their
    ends."""

    def __init__(self, msg, verts=()):
        super().__init__(msg)
        self.verts = list(verts)


def _crossing(face, Vx, near, ends):
    from OCP.BRepCheck import BRepCheck_Wire
    from OCP.TopAbs import TopAbs_WIRE
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopoDS import TopoDS, TopoDS_Edge

    verts = []
    ex = TopExp_Explorer(face, TopAbs_WIRE)
    while ex.More():
        e1, e2 = TopoDS_Edge(), TopoDS_Edge()
        if "Self" in str(BRepCheck_Wire(TopoDS.Wire(ex.Current())).SelfIntersect(face, e1, e2, True)):
            for e in (e1, e2):
                for p in ends(e):
                    verts.append(int(near[np.argmin(np.linalg.norm(Vx[near] - p, axis=1))]))
        ex.Next()
    return _Crossing("cylinder face is not valid", verts)


SCREEN_PAD = 2.0            # mm of the body kept around a cutter when it is tried alone (_screen)
CUT_FUZZ = (FUZZY, 1e-6, 1e-5)


def _body_shape(man):
    mesh = _on_float32(_mesh_of(man))
    return _faceted(mesh if mesh.is_watertight else _mesh_of(man))


def _try_cut(shape, n_faces, occ, fuzzy, want, removed):
    """`shape` less the OpenCascade solids `occ`, or None unless it is valid
    and took out what manifold3d says it should (`removed` mm^3, leaving
    `want`): a cutter that grazes the body often comes back valid and uncut."""
    from OCP.BRepCheck import BRepCheck_Analyzer

    try:
        if n_faces <= BOOL_INLINE:
            r = _bool1(_cls("subtract"), [shape], occ, fuzzy)
        else:
            r = _bool_timed("subtract", [shape], occ, fuzzy, _bool_seconds(n_faces))
    except RuntimeError:
        return None
    tol = min(2e-3 * want, 0.25 * removed) + 1e-6
    if abs(_volume(r) - want) > tol:
        return None
    if BRepCheck_Analyzer(r).IsValid():
        return r
    # A cut that took out the right volume can still fail the check on one
    # face: a counterbore's flat bottom landing on a facet plane of a pad
    # (z = 3 on the step-import elbow) came back with that one planar face
    # flagged, at every fuzz. ShapeFix mends the face's wires and
    # tolerances without moving geometry; it is kept only if the result is
    # then valid and still the volume manifold3d says.
    from OCP.ShapeFix import ShapeFix_Shape

    fix = ShapeFix_Shape(r)
    fix.Perform()
    r = fix.Shape()
    if abs(_volume(r) - want) > tol or not BRepCheck_Analyzer(r).IsValid():
        return None
    return r


def _screen(body, tools):
    """{fuzz: [tool indices]} and the indices OpenCascade cannot cut at any
    fuzz, each cutter tried alone against the piece of the body around it
    (a second or less; the whole body is ten). Which cutters fail is not
    predictable from their size or what they remove: a counterbore grazing
    1.3 mm^3 of a pad came back invalid at every fuzz, its neighbour taking
    1.4 mm^3 was clean, and a bore through solid needed the smaller fuzz."""
    import manifold3d as m3d

    groups, bad = {}, []
    for i, (occ, man) in enumerate(tools):
        box = man.bounding_box()
        lo, hi = np.array(box[:3]) - SCREEN_PAD, np.array(box[3:]) + SCREEN_PAD
        local = body ^ m3d.Manifold.cube(tuple(hi - lo)).translate(tuple(lo))
        left = local - man
        removed = local.volume() - left.volume()
        if local.is_empty() or removed <= 1e-9:
            continue
        shape = _body_shape(local)
        for fuzzy in CUT_FUZZ:
            if _try_cut(shape, local.num_tri(), occ, fuzzy, left.volume(), removed) is not None:
                groups.setdefault(fuzzy, []).append(i)
                break
        else:
            bad.append(i)
    return groups, bad


def _cut_exact(body, tools, notes):
    """The manifold `body` less the exact cutters `tools`, as an OpenCascade
    shape: the cutters OpenCascade can do are cut exactly (their faces are
    cylinders and planes in the file), together where they share a fuzz;
    the others are cut by manifold3d first, as facets, and noted."""
    import manifold3d as m3d

    def union(idx):
        return m3d.Manifold.batch_boolean([tools[i][1] for i in idx], m3d.OpType.Add)

    if not tools:
        return _body_shape(body)
    groups, facets = _screen(body, tools)
    while True:
        base = body - union(facets) if facets else body
        shape, now, n = _body_shape(base), base, base.num_tri()
        for fuzzy, idx in list(groups.items()):
            left = now - union(idx)
            removed = now.volume() - left.volume()
            if removed <= 1e-9:
                continue
            r = _try_cut(shape, n, [x for i in idx for x in tools[i][0]], fuzzy, left.volume(), removed)
            if r is None:                    # clean alone, not together: these go as facets, and start again
                facets += groups.pop(fuzzy)
                break
            shape, now = r, left
        else:
            break
    if facets:
        notes.append({"warning": f"{len(facets)} of {len(tools)} exact cutters had no clean OpenCascade cut "
                                 "and are facets in the file; the rest are exact"})
    return shape


def _fuse_all(shapes, man=None):
    """`shapes` as one. Big faceted ones are fused under the clock
    (_bool_timed); with `man` (the same union as a manifold) a fuse that
    fails or runs out of time is that manifold's facets instead."""
    if len(shapes) == 1:
        return shapes[0]
    n = _n_faces(shapes)
    if n <= BOOL_INLINE:
        return _bool1(_cls("union"), shapes[:1], shapes[1:])
    try:
        return _bool_timed("union", shapes[:1], shapes[1:], FUZZY, _bool_seconds(n))
    except RuntimeError:
        if man is None:
            raise
        from .shapes import trimesh_from_manifold

        return _faceted(_on_float32(trimesh_from_manifold(man)))


def _bool_seconds(n_faces):
    return float(os.environ.get("LOADPATH_STEP_BOOL_SECONDS", BOOL_SECONDS + BOOL_SECONDS_PER_FACE * n_faces))


def _cls(op):
    from OCP.BRepAlgoAPI import BRepAlgoAPI_Common, BRepAlgoAPI_Cut, BRepAlgoAPI_Fuse

    return {"union": BRepAlgoAPI_Fuse, "intersect": BRepAlgoAPI_Common, "subtract": BRepAlgoAPI_Cut}[op]


def _n_faces(shapes):
    from OCP.TopAbs import TopAbs_FACE
    from OCP.TopExp import TopExp_Explorer

    n = 0
    for s in shapes:
        ex = TopExp_Explorer(s, TopAbs_FACE)
        while ex.More():
            n += 1
            ex.Next()
    return n


class _TimedOut(RuntimeError):
    pass


def _bool_timed(op, first, tools, fuzzy, seconds):
    """`_bool1` in a process of its own, stopped after `seconds`.
    OpenCascade's boolean on two faceted solids with faces lying on each
    other (a kept pad's base on the design space's) sometimes never
    returns (hours in BOPAlgo_ShellSplitter), and it cannot be interrupted
    from the thread that called it."""
    import subprocess
    import sys
    import tempfile

    from OCP.BinTools import BinTools
    from OCP.TopoDS import TopoDS_Shape

    with tempfile.TemporaryDirectory(prefix="loadpath-step-") as d:
        for i, sh in enumerate(first):
            BinTools.Write_s(sh, os.path.join(d, f"a{i}.bin"))
        for i, sh in enumerate(tools):
            BinTools.Write_s(sh, os.path.join(d, f"t{i}.bin"))
        env = dict(os.environ)
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        env["PYTHONPATH"] = root + os.pathsep + env.get("PYTHONPATH", "")
        code = ("import sys; from loadpath.step import _bool_child; "
                "_bool_child(sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), float(sys.argv[5]))")
        try:
            done = subprocess.run([sys.executable, "-c", code, d, op, str(len(first)), str(len(tools)), repr(fuzzy)],
                                  env=env, timeout=seconds, check=False,
                                  stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except subprocess.TimeoutExpired:
            raise _TimedOut(f"OpenCascade {op} did not finish in {seconds:.0f} s") from None
        out = os.path.join(d, "out.bin")
        if done.returncode != 0 or not os.path.exists(out):
            raise RuntimeError(f"OpenCascade {op} failed")
        r = TopoDS_Shape()
        BinTools.Read_s(r, out)
        return r


def _bool_child(d, op, n_first, n_tools, fuzzy):
    from OCP.BinTools import BinTools
    from OCP.TopoDS import TopoDS_Shape

    def read(name):
        sh = TopoDS_Shape()
        BinTools.Read_s(sh, os.path.join(d, name))
        return sh

    r = _bool1(_cls(op), [read(f"a{i}.bin") for i in range(n_first)],
               [read(f"t{i}.bin") for i in range(n_tools)], fuzzy)
    BinTools.Write_s(r, os.path.join(d, "out.bin"))


def _referee(op, first, tools, want, notes):
    """The OpenCascade boolean that is valid and whose volume agrees with
    manifold3d's `want`; failing that, `want` itself as facets. An invalid
    or empty B-rep is never passed on: every later boolean fails on it, and
    the file that results has no solid in it."""
    from OCP.BRepCheck import BRepCheck_Analyzer

    from .shapes import trimesh_from_manifold

    volume = want.volume()
    n = _n_faces(first) + _n_faces(tools)
    seconds = _bool_seconds(n)
    tries = [(FUZZY, False), (1e-6, False), (FUZZY, True), (0.0, False), (1e-3, False)]
    closest, why = None, "no clean OpenCascade result"
    for fuzzy, fused in tries:
        try:
            t = [_fuse_all(tools)] if fused and len(tools) > 1 else tools
            if n <= BOOL_INLINE:
                r = _bool1(_cls(op), first, t, fuzzy)
            else:
                r = _bool_timed(op, first, t, fuzzy, seconds)
        except _TimedOut as e:
            why = str(e)          # these operands are the kind that hang: the next try is as likely to
            break
        except RuntimeError:
            continue
        ok = bool(BRepCheck_Analyzer(r).IsValid())
        err = abs(_volume(r) - volume) / max(abs(volume), 1e-9)
        if ok and err < 2e-3:
            return r
        if ok and (closest is None or err < closest[0]):
            closest = (err, r)
    if closest is not None and closest[0] < 2e-2:
        notes.append({"warning": f"{op}: {why}; kept the closest (volume off {closest[0]:.2%})"})
        return closest[1]
    mesh = _on_float32(trimesh_from_manifold(want))
    if not len(mesh.faces):
        raise RuntimeError(f"OpenCascade could not {op} the part")
    notes.append({"warning": f"{op}: {why}; this step is written as facets ({len(mesh.faces)}), "
                             "so exact faces cut before it are facets in the file"})
    return _faceted(mesh)


def _deviation(full, coarse, n=4000):
    """Distance (mm) from a sample of the full mesh's surface to the
    decimated one: what the faceting costs."""
    import trimesh

    P, _ = trimesh.sample.sample_surface(full, n, seed=2)
    _, d, _ = trimesh.proximity.closest_point(coarse, P)
    return {"mean": round(float(d.mean()), 4), "p99": round(float(np.percentile(d, 99)), 4)}


def _keep_bodies(shape, keep_with=None):
    """The largest solid and any solid touching `keep_with` (preserved
    geometry): what reconstruct keeps, fragments dropped."""
    from OCP.BRep import BRep_Builder
    from OCP.TopAbs import TopAbs_SOLID
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopoDS import TopoDS_Compound

    solids = []
    ex = TopExp_Explorer(shape, TopAbs_SOLID)
    while ex.More():
        solids.append(ex.Current())
        ex.Next()
    if len(solids) <= 1:
        return shape, 0
    vols = [_volume(s) for s in solids]
    order = np.argsort(vols)[::-1]
    kept = [solids[order[0]]]
    for i in order[1:]:
        if keep_with is not None and _volume(_bool("intersect", [solids[i], keep_with])) > 1e-6:
            kept.append(solids[i])
    b = BRep_Builder()
    comp = TopoDS_Compound()
    b.MakeCompound(comp)
    for s in kept:
        b.Add(comp, s)
    return comp, len(solids) - len(kept)


def _volume(shape):
    from OCP.BRepGProp import BRepGProp
    from OCP.GProp import GProp_GProps

    p = GProp_GProps()
    BRepGProp.VolumeProperties_s(shape, p)
    return float(p.Mass())


def _faces_report(shape, exact_planes):
    """Faces of the B-rep by surface type, and the share of its area on
    exact geometry (cylinders, spheres, and planes of the exact shapes)."""
    from OCP.BRepAdaptor import BRepAdaptor_Surface
    from OCP.BRepGProp import BRepGProp
    from OCP.GeomAbs import (
        GeomAbs_BezierSurface,
        GeomAbs_BSplineSurface,
        GeomAbs_Cylinder,
        GeomAbs_Plane,
        GeomAbs_Sphere,
    )
    from OCP.GProp import GProp_GProps
    from OCP.TopAbs import TopAbs_FACE
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopoDS import TopoDS

    kinds = {}
    area = {"exact": 0.0, "faceted": 0.0, "smooth": 0.0}
    ex = TopExp_Explorer(shape, TopAbs_FACE)
    while ex.More():
        f = TopoDS.Face(ex.Current())
        s = BRepAdaptor_Surface(f)
        t = s.GetType()
        p = GProp_GProps()
        BRepGProp.SurfaceProperties_s(f, p)
        a = float(p.Mass())
        if t == GeomAbs_Plane:
            pl = s.Plane()
            n = pl.Axis().Direction()
            o = pl.Location()
            nv = np.array([n.X(), n.Y(), n.Z()])
            ov = np.array([o.X(), o.Y(), o.Z()])
            exact = any(abs(abs(nv @ en) - 1) < 1e-7 and abs((ov - ep) @ en) < 1e-4 for ep, en in exact_planes)
            key = "plane" if exact else "facet"
        elif t == GeomAbs_Cylinder:
            key, exact = "cylinder", True
        elif t == GeomAbs_Sphere:
            key, exact = "sphere", True
        elif t in (GeomAbs_BSplineSurface, GeomAbs_BezierSurface):
            key, exact = "bspline", None
        else:
            key, exact = "other", True
        kinds[key] = kinds.get(key, 0) + 1
        area["smooth" if exact is None else "exact" if exact else "faceted"] += a
        ex.Next()
    total = sum(area.values()) or 1.0
    return {"faces": kinds, "exactArea": round(area["exact"], 1), "facetedArea": round(area["faceted"], 1),
            "smoothArea": round(area["smooth"], 1), "exactAreaFraction": round(area["exact"] / total, 4)}


def _planes(shape, out):
    from .shapes import Box, Cylinder, Shell, Union

    if isinstance(shape, Box):
        for a in range(3):
            n = np.zeros(3)
            n[a] = 1.0
            out += [(shape.lo.copy(), n), (shape.hi.copy(), n.copy())]
    elif isinstance(shape, Cylinder):
        out += [(shape.p0.copy(), shape.axis.copy()), (shape.p1.copy(), shape.axis.copy())]
    elif isinstance(shape, Union):
        for s in shape.shapes:
            _planes(s, out)
    elif isinstance(shape, Shell):
        for s in (shape.outer, shape.inner, shape.clip):
            if s is not None:
                _planes(s, out)
    return out


def _bezier_faces(shape):
    """A compound of the shape's Bezier and B-spline faces."""
    from OCP.BRep import BRep_Builder
    from OCP.BRepAdaptor import BRepAdaptor_Surface
    from OCP.GeomAbs import GeomAbs_BezierSurface, GeomAbs_BSplineSurface
    from OCP.TopAbs import TopAbs_FACE
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopoDS import TopoDS, TopoDS_Compound

    bb = BRep_Builder()
    out = TopoDS_Compound()
    bb.MakeCompound(out)
    ex = TopExp_Explorer(shape, TopAbs_FACE)
    while ex.More():
        if BRepAdaptor_Surface(TopoDS.Face(ex.Current())).GetType() in (GeomAbs_BezierSurface, GeomAbs_BSplineSurface):
            bb.Add(out, ex.Current())
        ex.Next()
    return out


def _smooth_shape(part, plan, patches, notes, degree=None):
    """surface="smooth": the finished mesh as a B-rep of smooth patches with
    its exact faces exact (brep.py), or None (and a note) when that does not
    come out a valid solid within 0.2% of the mesh's volume."""
    if not hasattr(part, "faces"):
        notes.append({"warning": "smooth surfaces need the part's mesh, not only its plan; faceted instead"})
        return None
    from OCP.BRepCheck import BRepCheck_Analyzer

    from .brep import smooth_brep
    from .sharp import mesh_distance

    try:
        solid, rep = smooth_brep(part, plan, patches, **({"order": int(degree)} if degree else {}))
        vol = _volume(solid)
        if not BRepCheck_Analyzer(solid).IsValid() or abs(vol - abs(part.volume)) > 2e-3 * abs(part.volume):
            raise RuntimeError(f"invalid or {abs(vol / abs(part.volume) - 1):.2%} off in volume")
    except Exception as e:  # noqa: BLE001 - facets always work
        notes.append({"warning": f"smooth surfaces failed ({e}); faceted instead"})
        return None
    # the patches against the mesh; the exact faces are the truth there (a
    # mesh's facets sag off a big bore by more than the patches miss)
    back = shape_to_trimesh(_bezier_faces(solid), deflection=0.005, angle=0.05)
    if len(back.faces):                       # a part of exact faces only has no patches to measure
        d = mesh_distance(part, back.sample(20000), cap=1.0)
        rep["deviation"] = {"mean": round(float(d.mean()), 4), "p99": round(float(np.percentile(d, 99)), 4),
                            "max": round(float(d.max()), 4)}
    notes.append({"leaf": "smooth", **rep})
    return solid


def _unify(shape):
    """Coplanar facets merged into planes, pieces of one cylinder into one
    face (ShapeUpgrade_UnifySameDomain), kept only if the result is valid.
    A bore's faces built on the mesh (_cut_on_mesh) sit on their seam edges
    within EDGE_TOL; merging them, or the edges along them, can leave a
    face OpenCascade rejects, and then they stay as built (a bore in three
    sectors) while the planes still merge."""
    from OCP.BRepAdaptor import BRepAdaptor_Surface
    from OCP.BRepCheck import BRepCheck_Analyzer
    from OCP.GeomAbs import GeomAbs_Plane
    from OCP.ShapeUpgrade import ShapeUpgrade_UnifySameDomain
    from OCP.TopAbs import TopAbs_EDGE, TopAbs_FACE
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopoDS import TopoDS

    valid = BRepCheck_Analyzer(shape).IsValid()
    for planes_only in (False, True):
        u = ShapeUpgrade_UnifySameDomain(shape, not planes_only, True, False)
        u.SetLinearTolerance(1e-5)
        if planes_only:
            fe = TopExp_Explorer(shape, TopAbs_FACE)
            while fe.More():
                if BRepAdaptor_Surface(TopoDS.Face(fe.Current())).GetType() != GeomAbs_Plane:
                    ee = TopExp_Explorer(fe.Current(), TopAbs_EDGE)
                    while ee.More():
                        u.KeepShape(ee.Current())
                        ee.Next()
                fe.Next()
        u.Build()
        out = u.Shape()
        if not valid or BRepCheck_Analyzer(out).IsValid():
            return out
    return shape


def export_step(part, path, faces=DEFAULT_FACES, name=None, unify=True, check=True, surface="facets",
                patches=DEFAULT_PATCHES, degree=None, tolerance=None):
    """Write `part` (a trimesh from a field graph or a study, or a CSG plan)
    to a STEP file at `path`. `faces` is the budget of organic facets; the
    exact shapes cost a handful of faces each. Returns a report: faces by
    type, the share of the area that is exact geometry, solids, volume (and
    the STL's, to compare), what decimating the organic surface cost, and
    the file's size. `tolerance` (mm, default TOPOLOGY_TOL) bounds how far
    facets move where a TPMS core's tunnels keep an organic region over its
    budget."""
    global TOPOLOGY_TOL
    if tolerance is not None:
        saved, TOPOLOGY_TOL = TOPOLOGY_TOL, float(tolerance)
        try:
            return export_step(part, path, faces, name, unify, check, surface, patches, degree)
        finally:
            TOPOLOGY_TOL = saved
    _need()
    import time

    from OCP.BRepCheck import BRepCheck_Analyzer

    from .sharp import Exact, Op, Organic, _leaves

    t0 = time.time()
    plan = part if isinstance(part, (Op, Exact, Organic)) else plan_of(part)
    mesh_volume = None if isinstance(part, (Op, Exact, Organic)) else float(abs(part.volume))
    notes = []
    keep_with = None
    mesh_plan = plan
    if plan is None:
        plan = Organic(None, ["mesh"], mesh=part)
    keep = getattr(plan, "keep_with", None)
    budget = _organic_budget(plan, faces)
    shape = None
    if surface == "smooth":
        shape = _smooth_shape(part, mesh_plan if mesh_volume is not None else None, patches, notes, degree)
    if shape is None:
        shape = _build_normal(plan, budget, faces, notes)
        if shape is None:
            shapes, _ = _build(plan, budget, faces, notes)
            shape = _fuse_all(shapes) if len(shapes) > 1 else shapes[0]
        dropped = 0
        if keep is not None:
            keep_with = _exact(keep, faces)
            shape, dropped = _keep_bodies(shape, keep_with)
        if unify:
            shape = _unify(shape)
    else:
        dropped = 0
    valid = bool(BRepCheck_Analyzer(shape).IsValid()) if check else None
    if check and not valid:
        from OCP.ShapeFix import ShapeFix_Shape

        fx = ShapeFix_Shape(shape)
        fx.Perform()
        shape = fx.Shape()
        valid = bool(BRepCheck_Analyzer(shape).IsValid())
    planes = []
    for lf in _leaves(plan):
        if isinstance(lf, Exact):
            _planes(lf.shape, planes)
    rep = _faces_report(shape, planes)
    wrote = write_step(shape, path, name=name, verify=bool(check and valid))
    from OCP.TopAbs import TopAbs_SOLID
    from OCP.TopExp import TopExp_Explorer

    n = 0
    ex = TopExp_Explorer(shape, TopAbs_SOLID)
    while ex.More():
        n += 1
        ex.Next()
    rep.update({"path": str(path), "bytes": os.path.getsize(path), "solids": n, "valid": valid,
                "volume": round(_volume(shape), 2), "seconds": round(time.time() - t0, 1)})
    if "pcurves" in wrote:
        rep["pcurves"] = wrote["pcurves"]
        rep["readBackValid"] = wrote["readBackValid"]
        if not wrote["roundedPoints"]:
            rep["roundedPoints"] = False
        rep["valid"] = bool(valid and wrote["readBackValid"])
    if mesh_volume is not None:
        rep["meshVolume"] = round(mesh_volume, 2)
    if dropped:
        rep["droppedFragments"] = dropped
    if notes:
        rep["decimated"] = notes
    rep["shape"] = shape
    return rep


def write_step(shape, path, name=None, pcurves=False, verify=False, decimals=POINT_DECIMALS):
    """Write an OpenCascade shape to STEP (mm, AP214), without the faces'
    2-D curves unless `pcurves` (readers rebuild them, and they are half the
    file), points rounded to `decimals` places (None: as OpenCascade writes
    them) and each written once. `verify` reads the file back and, if the
    shape no longer passes OpenCascade's check or comes back as a different
    number of solids, writes it again: with the 2-D curves (a reader
    projecting an edge onto a sliver patch can fail where the writer's own
    2-D curve was fine), then with the points unrounded (the step-import
    tutorial's bracket: booleans of facets against the bolt bores leave
    edges a few nm long, and rounding to 10 nm folded the bore faces), then
    both. The first that reads back valid is kept; if none does, the first.
    Returns {path, bytes, writeSeconds} (and with verify: pcurves,
    roundedPoints, readBackValid)."""
    import time

    t0 = time.time()
    tries = [(pcurves, decimals)]
    if verify:
        tries += [t for t in [(True, decimals), (pcurves, None), (True, None)] if t not in tries]
    want = _count(shape, "solid") if verify else None
    for pc, dec in tries:
        out = _write_step(shape, path, name, pc, dec)
        if not verify:
            return out
        back = read_step(path)
        ok = bool(shape_valid(back) and _count(back, "solid") == want)
        out.update({"pcurves": bool(pc), "roundedPoints": dec is not None, "readBackValid": ok})
        if ok:
            break
    else:
        if len(tries) > 1:
            out = _write_step(shape, path, name, *tries[0])
            out.update({"pcurves": bool(tries[0][0]), "roundedPoints": tries[0][1] is not None,
                        "readBackValid": False})
    out["writeSeconds"] = round(time.time() - t0, 1)
    return out


def _count(shape, kind):
    from OCP.TopAbs import TopAbs_SOLID
    from OCP.TopExp import TopExp_Explorer

    ex, n = TopExp_Explorer(shape, {"solid": TopAbs_SOLID}[kind]), 0
    while ex.More():
        n += 1
        ex.Next()
    return n


def _write_step(shape, path, name, pcurves, decimals):
    import time

    from OCP.Interface import Interface_Static
    from OCP.STEPControl import (
        STEPControl_AsIs,
        STEPControl_Controller,
        STEPControl_Writer,
    )

    t0 = time.time()
    STEPControl_Controller.Init_s()      # registers the write.* parameters before they are set
    Interface_Static.SetCVal_s("write.step.unit", "MM")
    Interface_Static.SetCVal_s("write.step.schema", "AP214IS")
    Interface_Static.SetIVal_s("write.surfacecurve.mode", 1 if pcurves else 0)
    if name:
        Interface_Static.SetCVal_s("write.step.product.name", str(name))
    w = STEPControl_Writer()
    w.Transfer(shape, STEPControl_AsIs)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    if int(w.Write(str(path))) != 1:
        raise RuntimeError(f"OpenCascade could not write {path}")
    _compact(path, decimals)
    return {"path": str(path), "bytes": os.path.getsize(path), "writeSeconds": round(time.time() - t0, 1)}


BIG_SHELL = 8000      # faces: past this a whole-solid check is checked shell by shell


def shape_valid(shape, geom=True):
    """OpenCascade's check (BRepCheck). A solid with an inner shell of more
    than BIG_SHELL faces is checked shell by shell (each as a solid of its
    own): the whole-solid check classifies every shell against every other
    and went past 12 GB on an arm link's 38k-patch void."""
    from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeSolid
    from OCP.BRepCheck import BRepCheck_Analyzer
    from OCP.TopAbs import TopAbs_FACE, TopAbs_SHELL
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopoDS import TopoDS

    def nfaces(x):
        ex, n = TopExp_Explorer(x, TopAbs_FACE), 0
        while ex.More():
            n += 1
            ex.Next()
        return n

    shells = []
    ex = TopExp_Explorer(shape, TopAbs_SHELL)
    while ex.More():
        shells.append(TopoDS.Shell(ex.Current()))
        ex.Next()
    if len(shells) < 2 or max(nfaces(x) for x in shells) <= BIG_SHELL:
        return bool(BRepCheck_Analyzer(shape, geom).IsValid())
    return all(BRepCheck_Analyzer(BRepBuilderAPI_MakeSolid(x).Solid(), geom).IsValid() for x in shells)


def _compact(path, decimals=POINT_DECIMALS):
    """Rewrite a STEP file smaller: points to `decimals` places (mm; 10 nm
    is a thousandth of the sewing tolerance) and each distinct point once.
    Three quarters of a smooth STEP file is points, and a patch's edge poles
    are the same points as its edge curve's and its neighbour's. `decimals`
    None leaves the numbers as written and merges identical points only."""
    import re

    with open(path) as fh:
        text = fh.read()
    head, sep, rest = text.partition("DATA;")
    data, sep2, tail = rest.partition("ENDSEC;")
    if not sep or not sep2:
        return
    ents = re.split(r"\n(?=#\d+\s*=)", data)
    num = re.compile(r"-?\d+\.\d*(?:E[-+]?\d+)?")
    point = re.compile(r"#(\d+)\s*=\s*CARTESIAN_POINT\('([^']*)',\((.*)\)\)\s*;\s*$", re.DOTALL)

    def short(x):
        if decimals is None:
            return x
        t = f"{float(x):.{decimals}f}".rstrip("0")
        return "0." if t in ("-0.", "0.") else t

    first, alias, out = {}, {}, []
    for e in ents:
        mo = point.match(e.strip())
        if mo:
            coords = ",".join(short(x) for x in num.findall(mo.group(3)))
            key = (mo.group(2), coords)
            if key in first:
                alias[mo.group(1)] = first[key]
                continue
            first[key] = mo.group(1)
            out.append(f"#{mo.group(1)} = CARTESIAN_POINT('{mo.group(2)}',({coords}));")
        else:
            out.append(e)
    if alias:
        ref = re.compile(r"#(\d+)")
        body = []
        for e in out:
            lhs, eq, rhs = e.partition("=")
            body.append(lhs + eq + ref.sub(lambda m: "#" + alias.get(m.group(1), m.group(1)), rhs) if eq else e)
        out = body
    with open(path, "w") as fh:
        fh.write(head + sep + "\n".join(x.strip("\n") for x in out) + "\n" + sep2 + tail)


def read_step(path):
    """The shape in a STEP file (OpenCascade TopoDS_Shape)."""
    _need()
    from OCP.IFSelect import IFSelect_RetDone
    from OCP.STEPControl import STEPControl_Reader

    r = STEPControl_Reader()
    if r.ReadFile(str(path)) != IFSelect_RetDone:
        raise ValueError(f"could not read STEP file {path}")
    r.TransferRoots()
    return r.OneShape()


def shape_to_trimesh(shape, deflection=0.002, angle=0.05, chunk=2000):
    """Triangulate an OpenCascade shape finely (chord `deflection` mm) into a
    trimesh: how to measure a B-rep with the mesh tools."""
    import trimesh
    from OCP.BRep import BRep_Builder, BRep_Tool
    from OCP.BRepMesh import BRepMesh_IncrementalMesh
    from OCP.BRepTools import BRepTools
    from OCP.TopAbs import TopAbs_FACE, TopAbs_REVERSED
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopLoc import TopLoc_Location
    from OCP.TopoDS import TopoDS, TopoDS_Compound

    # in chunks of faces, each meshed (in parallel), read into numpy and its
    # triangulation dropped: one mesh of a whole lattice part (46k patches,
    # tens of millions of nodes) held at once went past 13 GB
    faces = []
    ex = TopExp_Explorer(shape, TopAbs_FACE)
    while ex.More():
        faces.append(TopoDS.Face(ex.Current()))
        ex.Next()
    V, F = [], []
    off = 0
    bb = BRep_Builder()
    for c0 in range(0, len(faces), chunk):
        comp = TopoDS_Compound()
        bb.MakeCompound(comp)
        for f in faces[c0:c0 + chunk]:
            bb.Add(comp, f)
        BRepMesh_IncrementalMesh(comp, deflection, False, angle, True)
        for f in faces[c0:c0 + chunk]:
            loc = TopLoc_Location()
            tri = BRep_Tool.Triangulation_s(f, loc)
            if tri is None:
                continue
            tr = loc.Transformation()
            nn, nt = tri.NbNodes(), tri.NbTriangles()
            v = np.empty((nn, 3))
            for i in range(nn):
                p = tri.Node(i + 1).Transformed(tr)
                v[i] = p.X(), p.Y(), p.Z()
            t = np.empty((nt, 3), np.int64)
            for i in range(nt):
                t[i] = tri.Triangle(i + 1).Get()
            if f.Orientation() == TopAbs_REVERSED:
                t = t[:, [0, 2, 1]]
            V.append(v)
            F.append(t - 1 + off)
            off += nn
        BRepTools.Clean_s(comp)
    m = trimesh.Trimesh(np.concatenate(V) if V else np.zeros((0, 3)),
                        np.concatenate(F) if F else np.zeros((0, 3), np.int64), process=True)
    m.merge_vertices(digits_vertex=6)
    return m
