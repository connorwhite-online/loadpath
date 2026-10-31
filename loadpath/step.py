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

A mesh with no plan (any STL) exports fully faceted.

Needs OpenCascade's Python bindings (`pip install "loadpath[step]"`, which is
cadquery-ocp).
"""
from __future__ import annotations

import math
import os

import numpy as np

__all__ = ["available", "export_step", "read_step", "shape_to_trimesh", "attach_plan", "plan_of"]

DEFAULT_FACES = 20_000      # organic facets in the B-rep: ~45 MB of STEP, ~20 s
FUZZY = 1e-4                # boolean fuzz (mm): facets snapped onto exact faces
DEFAULT_PATCHES = 1500      # smooth B-spline patches: ~8 kB of STEP each
SMOOTH_SAMPLES = 8          # points per patch edge pulled onto the surface
NEAR_EXACT = 0.5            # mm: organic surface this close to exact geometry stays faceted


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


def _faceted(mesh):
    """A closed trimesh as an OpenCascade solid, one planar face per
    triangle, sharing edges (BRepBuilderAPI_MakeShapeOnMesh)."""
    from OCP.BRep import BRep_Builder
    from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeShapeOnMesh, BRepBuilderAPI_MakeSolid
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
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox, BRepPrimAPI_MakeCylinder, BRepPrimAPI_MakeSphere
    from OCP.gp import gp_Ax2, gp_Dir

    from .shapes import Box, Cylinder, MeshShape, Shell, Sphere, Union, trimesh_from_manifold

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


def _collapse(V, F, target, locked=None, max_turn=0.5):
    """Quadric edge collapse of a closed mesh (V, F) to `target` faces that
    keeps its topology: an edge collapses only when its ends share exactly
    two neighbours (the link condition), so thin members and holes survive,
    and only when no face around it turns by more than ~60 degrees. Slower
    than fast_simplification, which at a few thousand faces tears a part's
    thin members. Vertices flagged in `locked` never move. Returns the
    vertices, faces and lock flags of the result."""
    import heapq

    V = np.array(V, float); F = np.array(F, np.int64)
    locked = np.zeros(len(V), bool) if locked is None else np.asarray(locked, bool)
    nf = len(F)
    alive = np.ones(nf, bool)
    vf = [set() for _ in range(len(V))]
    for i, f in enumerate(F):
        for v in f: vf[v].add(i)
    n = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    area = np.linalg.norm(n, axis=1); nn = n / np.maximum(area, 1e-30)[:, None]
    d = -np.einsum("ij,ij->i", nn, V[F[:, 0]])
    p = np.hstack([nn, d[:, None]])
    K = np.einsum("i,ij,ik->ijk", area, p, p)
    Q = np.zeros((len(V), 4, 4))
    for c in range(3): np.add.at(Q, F[:, c], K)
    ver = np.zeros(len(V), np.int64)
    def nbrs(v):
        s = set()
        for f in vf[v]: s.update(F[f])
        s.discard(v); return s
    def cost(a, b):
        q = Q[a] + Q[b]; best = None
        for x in (V[a], V[b], 0.5 * (V[a] + V[b])):
            h = np.append(x, 1.0); c = h @ q @ h
            if best is None or c < best[0]: best = (c, x)
        return best
    heap = []
    E = set()
    for f in F:
        for k in range(3):
            a, b = int(f[k]), int(f[(k + 1) % 3]); E.add((min(a, b), max(a, b)))
    for a, b in E:
        c, _ = cost(a, b); heap.append((c, a, b, 0, 0))
    heapq.heapify(heap)
    live = nf
    while live > target and heap:
        c, a, b, va, vb = heapq.heappop(heap)
        if ver[a] != va or ver[b] != vb or not vf[a] or not vf[b] or locked[a] or locked[b]: continue
        shared = vf[a] & vf[b]
        if len(shared) != 2: continue
        if len(nbrs(a) & nbrs(b)) != 2: continue           # link condition
        _, x = cost(a, b)
        ok = True
        for f in (vf[a] | vf[b]) - shared:
            tri = F[f].copy(); P = V[tri].copy()
            P[(tri == a) | (tri == b)] = x
            m = np.cross(P[1] - P[0], P[2] - P[0]); L = np.linalg.norm(m)
            o = np.cross(V[tri[1]] - V[tri[0]], V[tri[2]] - V[tri[0]]); Lo = np.linalg.norm(o)
            if L < 1e-12 or (m @ o) / (L * Lo + 1e-30) < max_turn: ok = False; break
        if not ok:
            continue
        for f in shared:
            alive[f] = False
            for v in F[f]: vf[v].discard(f)
        live -= 2
        for f in vf[b]:
            F[f][F[f] == b] = a; vf[a].add(f)
        vf[b] = set()
        V[a] = x; Q[a] += Q[b]
        ver[a] += 1; ver[b] += 1
        for w in nbrs(a):
            cc, _ = cost(a, w)
            i, j = (a, w) if a < w else (w, a)
            heapq.heappush(heap, (cc, i, j, ver[i], ver[j]))
    keep = F[alive]
    used, inv = np.unique(keep, return_inverse=True)
    return V[used], inv.reshape(-1, 3), locked[used]


def _layout(mesh, patches, near=None, facets=DEFAULT_FACES):
    """`mesh` decimated to `patches` triangles with its topology intact: the
    fast decimator down to where it still keeps it, the careful one after.
    `near(P)` flags points close to exact geometry: those vertices stay put
    (and their faces stay flat facets, as fine as `facets` allows).
    Returns (mesh, lock flags)."""
    import trimesh

    start = mesh
    for n in ((facets,) if near is not None else (8 * patches, 4 * patches)):
        if n >= len(start.faces):
            continue
        try:
            c = mesh.simplify_quadric_decimation(face_count=int(n))
        except Exception:  # noqa: BLE001
            break
        if not (c.is_watertight and c.is_winding_consistent and c.euler_number == mesh.euler_number):
            break
        start = c
    lock = near(np.asarray(start.vertices)) if near is not None else None
    if len(start.faces) <= patches:
        return start, (np.zeros(len(start.vertices), bool) if lock is None else lock)
    if lock is not None:        # the flat faces stay as they are: `patches` counts the rest
        patches += int(lock[np.asarray(start.faces)].any(axis=1).sum())
    V, F, lock = _collapse(start.vertices, start.faces, int(patches), lock)
    return trimesh.Trimesh(V, F, process=False), lock


def _smooth(mesh, patches, near=None, facets=DEFAULT_FACES, samples=SMOOTH_SAMPLES):
    """A closed trimesh as an OpenCascade solid of smooth B-spline patches.

    The mesh is decimated to `patches` triangles (keeping its holes), which
    become the patch layout: each coarse edge is a B-spline curve through
    points pulled onto the full surface, shared by the two patches beside
    it, and each coarse triangle a B-spline surface (OpenCascade's plate
    filling) bounded by its three curves and pulled through points of the
    full surface inside it. Patches meet with matching edges (C0); a coarse
    triangle is a few millimetres, so the kinks between them are fractions
    of a degree on a smooth body.

    Where the body runs along exact geometry (`near(P)` true: it hugs a
    bearing seat, or the envelope trims it), the mesh stays fine and flat
    facets: a B-spline lying almost on a cylinder defeats OpenCascade's
    booleans, facets with fuzz do not. Raises if any step fails, so the
    caller can fall back to facets. Returns (solid, layout, flat faces)."""
    from OCP.BRepBuilderAPI import (BRepBuilderAPI_MakeEdge, BRepBuilderAPI_MakeFace,
                                    BRepBuilderAPI_MakeSolid, BRepBuilderAPI_MakeVertex,
                                    BRepBuilderAPI_MakeWire, BRepBuilderAPI_Sewing)
    from OCP.BRepOffsetAPI import BRepOffsetAPI_MakeFilling
    from OCP.GeomAbs import GeomAbs_C0, GeomAbs_C2
    from OCP.GeomAPI import GeomAPI_PointsToBSpline
    from OCP.OCP.collections import Array1_gp_Pnt
    from OCP.ShapeFix import ShapeFix_Solid
    from OCP.TopAbs import TopAbs_SHELL
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopoDS import TopoDS

    coarse, lock = _layout(mesh, patches, near, facets)
    if not coarse.is_watertight or not coarse.is_winding_consistent:
        raise ValueError("the patch layout (decimated mesh) is not closed")
    n = samples
    V = np.asarray(coarse.vertices, float).copy()
    V[~lock] = _closest(mesh, V[~lock])
    F = np.asarray(coarse.faces)
    flat = lock[F].any(axis=1)
    straight = {tuple(sorted((int(f[k]), int(f[(k + 1) % 3])))) for f in F[flat] for k in range(3)}
    E = np.sort(np.asarray(coarse.edges_unique), axis=1)
    curved = np.array([tuple(e) not in straight for e in E.tolist()], bool)
    t = np.linspace(0, 1, n + 1)[1:-1, None, None]
    Ec = E[curved]
    mid = _closest(mesh, (V[Ec[:, 0]] * (1 - t) + V[Ec[:, 1]] * t).reshape(-1, 3)).reshape(n - 1, len(Ec), 3)
    verts = [BRepBuilderAPI_MakeVertex(_pnt(p)).Vertex() for p in V]
    edges = {}
    for i, (a, b) in enumerate(Ec):
        P = np.vstack([V[a], mid[:, i], V[b]])
        arr = Array1_gp_Pnt(1, len(P))
        for j, p in enumerate(P):
            arr.SetValue(j + 1, _pnt(p))
        crv = GeomAPI_PointsToBSpline(arr, 3, 3, GeomAbs_C2, 1e-3).Curve()
        edges[(int(a), int(b))] = BRepBuilderAPI_MakeEdge(crv, verts[a], verts[b]).Edge()
    for a, b in E[~curved]:
        edges[(int(a), int(b))] = BRepBuilderAPI_MakeEdge(verts[a], verts[b]).Edge()
    h = max(2, n // 2)
    bc = np.array([(i, j, h - i - j) for i in range(1, h) for j in range(1, h) if h - i - j > 0], float) / h
    S = F[~flat]
    inner = _closest(mesh, np.einsum("kc,fcd->fkd", bc, V[S]).reshape(-1, 3)).reshape(len(S), len(bc), 3)
    sew = BRepBuilderAPI_Sewing(1e-3)
    for fi, f in enumerate(S):
        # degree 3, 10 points per curve, 2 iterations, max degree 4 and 2 segments:
        # ~40 poles a patch, 0.01-0.04 mm off a smooth body
        mk = BRepOffsetAPI_MakeFilling(3, 10, 2, False, 1e-5, 1e-4, 0.01, 0.1, 4, 2)
        for k in range(3):
            a, b = int(f[k]), int(f[(k + 1) % 3])
            mk.Add(edges[(min(a, b), max(a, b))], GeomAbs_C0)
        for p in inner[fi]:
            mk.Add(_pnt(p))
        mk.Build()
        if not mk.IsDone():
            raise RuntimeError("a smooth patch failed")
        sew.Add(mk.Shape())
    for f in F[flat]:
        w = BRepBuilderAPI_MakeWire()
        for k in range(3):
            a, b = int(f[k]), int(f[(k + 1) % 3])
            w.Add(edges[(min(a, b), max(a, b))])
        mf = BRepBuilderAPI_MakeFace(w.Wire(), True)
        if not mf.IsDone():
            raise RuntimeError("a flat facet failed")
        sew.Add(mf.Face())
    sew.Perform()
    ex = TopExp_Explorer(sew.SewedShape(), TopAbs_SHELL)
    shells = []
    while ex.More():
        shells.append(TopoDS.Shell(ex.Current()))
        ex.Next()
    if len(shells) != 1 or sew.NbFreeEdges():
        raise RuntimeError(f"smooth patches did not close ({len(shells)} shells, {sew.NbFreeEdges()} free edges)")
    fx = ShapeFix_Solid(BRepBuilderAPI_MakeSolid(shells[0]).Solid())
    fx.Perform()
    solid = fx.Solid()
    if _volume(solid) < 0:
        solid.Reverse()
    return solid, coarse, int(flat.sum())


def _near_exact(plan, r=NEAR_EXACT, spacing=0.2):
    """P -> flags of points within about `r` of the plan's exact shapes'
    surfaces (a k-d tree over points spread on them)."""
    from scipy.spatial import cKDTree

    from .sharp import Exact, _leaves, exact_manifold, surface_samples
    from .shapes import trimesh_from_manifold

    pts = []
    for lf in _leaves(plan):
        if not isinstance(lf, Exact):
            continue
        try:
            P, _ = surface_samples(lf.shape, spacing)
        except Exception:  # noqa: BLE001 - a shape surface_samples does not know
            P = np.zeros((0, 3))
        if not len(P):
            m = trimesh_from_manifold(exact_manifold(lf.shape))
            P = m.sample(max(1000, int(m.area / spacing ** 2)))
        pts.append(np.asarray(P, float))
    if not pts:
        return None
    tree = cKDTree(np.vstack(pts))

    def near(P):
        d, _ = tree.query(np.asarray(P, float), distance_upper_bound=r)
        return np.isfinite(d)
    return near


def _decimate(mesh, faces):
    """`mesh` cut down to about `faces` triangles, still closed. A lattice
    does not decimate (its walls collapse first): past 4x the budget that is
    an error, not a STEP file of millions of faces."""
    if len(mesh.faces) <= faces:
        return mesh
    try:
        out = mesh.simplify_quadric_decimation(face_count=int(faces))
        if out.is_watertight and abs(out.volume) > 0.9 * abs(mesh.volume):
            return out
    except Exception:  # noqa: BLE001 - decimation is best-effort
        pass
    if len(mesh.faces) > 4 * faces:
        raise ValueError(f"an organic region of {len(mesh.faces)} triangles does not decimate to {faces} and stay "
                         "closed (a lattice?): raise the STEP face budget, or hand over the STL for it")
    return mesh


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
    from .shapes import Union

    if isinstance(shape, Union):
        return [p for s in shape.shapes for p in _pieces(s, faces)]
    return [_exact(shape, faces)]


def _build(plan, budget, faces, notes, patches=0, near=None):
    """(OpenCascade solids, the same thing as a manifold3d Manifold). The
    manifold is the referee: each OpenCascade boolean is tried a few ways
    (fuzz, tools separate or fused) until it is valid and its volume agrees
    with manifold3d's."""
    import manifold3d as m3d

    from .sharp import Exact, Organic, exact_manifold
    from .shapes import manifold_from_trimesh

    if isinstance(plan, Exact):
        return _pieces(plan.shape, faces), exact_manifold(plan.shape)
    if isinstance(plan, Organic):
        m = plan.mesh
        if patches:
            share = max(50, int(patches * budget.get(id(plan), faces) / max(faces, 1)))
            try:
                solid, coarse, nflat = _smooth(m, share, near, budget.get(id(plan), faces))
                notes.append({"leaf": ",".join(plan.names) or "organic", "faces": int(len(m.faces)),
                              "patches": int(len(coarse.faces) - nflat), "flatFacets": nflat,
                              "deviation": _deviation(m, shape_to_trimesh(solid, deflection=0.005, angle=0.05))})
                return [solid], manifold_from_trimesh(m)
            except Exception as e:  # noqa: BLE001 - facets always work
                notes.append({"leaf": ",".join(plan.names) or "organic",
                              "warning": f"smooth patches failed ({e}); faceted instead"})
        d = _decimate(m, budget.get(id(plan), faces))
        if len(d.faces) < len(m.faces):
            notes.append({"leaf": ",".join(plan.names) or "organic", "faces": int(len(m.faces)),
                          "stepFaces": int(len(d.faces)), "deviation": _deviation(m, d)})
        return [_faceted(d)], manifold_from_trimesh(d)
    kids = [_build(k, budget, faces, notes, patches, near) for k in plan.kids]
    kind = {"union": m3d.OpType.Add, "intersect": m3d.OpType.Intersect, "subtract": m3d.OpType.Subtract}[plan.op]
    want = m3d.Manifold.batch_boolean([k[1] for k in kids], kind)
    if plan.op == "union":
        first, tools = kids[0][0][:1], kids[0][0][1:] + [s for k in kids[1:] for s in k[0]]
    else:
        first = [_fuse_all(kids[0][0])]
        tools = [s for k in kids[1:] for s in k[0]]
        if plan.op == "intersect":
            tools = [_fuse_all(tools)]
    shape = _referee(plan.op, first, tools, want.volume(), notes)
    return [shape], want


def _fuse_all(shapes):
    return shapes[0] if len(shapes) == 1 else _bool1(_cls("union"), shapes[:1], shapes[1:])


def _cls(op):
    from OCP.BRepAlgoAPI import BRepAlgoAPI_Common, BRepAlgoAPI_Cut, BRepAlgoAPI_Fuse

    return {"union": BRepAlgoAPI_Fuse, "intersect": BRepAlgoAPI_Common, "subtract": BRepAlgoAPI_Cut}[op]


def _referee(op, first, tools, want, notes):
    from OCP.BRepCheck import BRepCheck_Analyzer

    tries = [(FUZZY, False), (1e-6, False), (FUZZY, True), (0.0, False), (1e-3, False)]
    best = None
    for fuzzy, fused in tries:
        t = [_fuse_all(tools)] if fused and len(tools) > 1 else tools
        try:
            r = _bool1(_cls(op), first, t, fuzzy)
        except RuntimeError:
            continue
        ok = bool(BRepCheck_Analyzer(r).IsValid())
        err = abs(_volume(r) - want) / max(abs(want), 1e-9)
        if ok and err < 2e-3:
            return r
        score = (not ok, err)
        if best is None or score < best[0]:
            best = (score, r)
    if best is None:
        raise RuntimeError(f"OpenCascade could not {op} the part")
    notes.append({"warning": f"{op}: no clean OpenCascade result; kept the closest "
                             f"(valid {not best[0][0]}, volume off {best[0][1]:.2%})"})
    return best[1]


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
    from OCP.BRep import BRep_Tool
    from OCP.BRepAdaptor import BRepAdaptor_Surface
    from OCP.BRepGProp import BRepGProp
    from OCP.GeomAbs import GeomAbs_BSplineSurface, GeomAbs_Cylinder, GeomAbs_Plane, GeomAbs_Sphere
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
        elif t == GeomAbs_BSplineSurface:
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


def export_step(part, path, faces=DEFAULT_FACES, name=None, unify=True, check=True, surface="facets",
                patches=DEFAULT_PATCHES):
    """Write `part` (a trimesh from a field graph or a study, or a CSG plan)
    to a STEP file at `path`. `faces` is the budget of organic facets; the
    exact shapes cost a handful of faces each. Returns a report: faces by
    type, the share of the area that is exact geometry, solids, volume (and
    the STL's, to compare), what decimating the organic surface cost, and
    the file's size."""
    _need()
    import time

    from OCP.BRepCheck import BRepCheck_Analyzer
    from OCP.Interface import Interface_Static
    from OCP.STEPControl import STEPControl_AsIs, STEPControl_Controller, STEPControl_Writer

    from .sharp import Exact, Op, Organic, _leaves

    t0 = time.time()
    plan = part if isinstance(part, (Op, Exact, Organic)) else plan_of(part)
    mesh_volume = None if isinstance(part, (Op, Exact, Organic)) else float(abs(part.volume))
    notes = []
    keep_with = None
    if plan is None:
        plan = Organic(None, ["mesh"], mesh=part)
    keep = getattr(plan, "keep_with", None)
    budget = _organic_budget(plan, faces)
    smooth = patches if surface == "smooth" else 0
    shapes = None
    if smooth:
        from OCP.BRepCheck import BRepCheck_Analyzer as _Check

        try:
            shapes, want = _build(plan, budget, faces, notes, smooth, _near_exact(plan))
            got = sum(_volume(x) for x in shapes)
            if not all(_Check(x).IsValid() for x in shapes) or abs(got - want.volume()) > 2e-3 * abs(want.volume()):
                raise RuntimeError(f"invalid or {abs(got / want.volume() - 1):.2%} off in volume")
        except RuntimeError as e:
            # smooth patches meeting exact faces at a grazing angle can defeat
            # OpenCascade's booleans; facets with fuzz do not
            notes[:] = [{"warning": f"smooth surfaces did not survive the booleans ({e}); faceted instead"}]
            shapes = None
    if shapes is None:
        shapes, _ = _build(plan, budget, faces, notes, 0)
    shape = _fuse_all(shapes) if len(shapes) > 1 else shapes[0]
    dropped = 0
    if keep is not None:
        keep_with = _exact(keep, faces)
        shape, dropped = _keep_bodies(shape, keep_with)
    if unify:
        from OCP.ShapeUpgrade import ShapeUpgrade_UnifySameDomain

        u = ShapeUpgrade_UnifySameDomain(shape, True, True, False)
        u.SetLinearTolerance(1e-5)
        u.Build()
        shape = u.Shape()
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
    STEPControl_Controller.Init_s()      # registers the write.* parameters before they are set
    Interface_Static.SetCVal_s("write.step.unit", "MM")
    Interface_Static.SetCVal_s("write.step.schema", "AP214IS")
    # no 2-D curves on faces: readers rebuild them, and they were a quarter of the file
    Interface_Static.SetIVal_s("write.surfacecurve.mode", 0)
    if name:
        Interface_Static.SetCVal_s("write.step.product.name", str(name))
    w = STEPControl_Writer()
    w.Transfer(shape, STEPControl_AsIs)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    if int(w.Write(str(path))) != 1:
        raise RuntimeError(f"OpenCascade could not write {path}")
    from OCP.TopAbs import TopAbs_SOLID
    from OCP.TopExp import TopExp_Explorer

    n = 0
    ex = TopExp_Explorer(shape, TopAbs_SOLID)
    while ex.More():
        n += 1
        ex.Next()
    rep.update({"path": str(path), "bytes": os.path.getsize(path), "solids": n, "valid": valid,
                "volume": round(_volume(shape), 2), "seconds": round(time.time() - t0, 1)})
    if mesh_volume is not None:
        rep["meshVolume"] = round(mesh_volume, 2)
    if dropped:
        rep["droppedFragments"] = dropped
    if notes:
        rep["decimated"] = notes
    rep["shape"] = shape
    return rep


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


def shape_to_trimesh(shape, deflection=0.002, angle=0.05):
    """Triangulate an OpenCascade shape finely (chord `deflection` mm) into a
    trimesh: how to measure a B-rep with the mesh tools."""
    import trimesh
    from OCP.BRep import BRep_Tool
    from OCP.BRepMesh import BRepMesh_IncrementalMesh
    from OCP.TopAbs import TopAbs_FACE, TopAbs_REVERSED
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopLoc import TopLoc_Location
    from OCP.TopoDS import TopoDS

    BRepMesh_IncrementalMesh(shape, deflection, False, angle, True)
    V, F = [], []
    off = 0
    ex = TopExp_Explorer(shape, TopAbs_FACE)
    while ex.More():
        f = TopoDS.Face(ex.Current())
        loc = TopLoc_Location()
        tri = BRep_Tool.Triangulation_s(f, loc)
        if tri is not None:
            tr = loc.Transformation()
            for i in range(1, tri.NbNodes() + 1):
                p = tri.Node(i).Transformed(tr)
                V.append((p.X(), p.Y(), p.Z()))
            rev = f.Orientation() == TopAbs_REVERSED
            for i in range(1, tri.NbTriangles() + 1):
                a, b, c = tri.Triangle(i).Get()
                F.append((off + a - 1, off + c - 1, off + b - 1) if rev else (off + a - 1, off + b - 1, off + c - 1))
            off += tri.NbNodes()
        ex.Next()
    m = trimesh.Trimesh(np.asarray(V, float), np.asarray(F, np.int64), process=True)
    m.merge_vertices(digits_vertex=6)
    return m
