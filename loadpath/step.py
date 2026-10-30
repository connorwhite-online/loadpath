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


def _bool1(cls, args, tools):
    from OCP.OCP.collections import List_TopoDS_Shape

    a, t = List_TopoDS_Shape(), List_TopoDS_Shape()
    for s in args:
        a.Append(s)
    for s in tools:
        t.Append(s)
    op = cls()
    op.SetArguments(a)
    op.SetTools(t)
    op.SetFuzzyValue(FUZZY)
    op.SetRunParallel(True)
    op.SetNonDestructive(True)
    op.Build()
    if not op.IsDone():
        raise RuntimeError(f"OpenCascade {cls.__name__} failed")
    return op.Shape()


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


def _build(plan, budget, faces, notes):
    from .sharp import Exact, Organic

    if isinstance(plan, Exact):
        return _exact(plan.shape, faces)
    if isinstance(plan, Organic):
        m = plan.mesh
        d = _decimate(m, budget.get(id(plan), faces))
        if len(d.faces) < len(m.faces):
            notes.append({"leaf": ",".join(plan.names) or "organic", "faces": int(len(m.faces)),
                          "stepFaces": int(len(d.faces)), "deviation": _deviation(m, d)})
        return _faceted(d)
    return _bool(plan.op, [_build(k, budget, faces, notes) for k in plan.kids])


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
    from OCP.GeomAbs import GeomAbs_Cylinder, GeomAbs_Plane, GeomAbs_Sphere
    from OCP.GProp import GProp_GProps
    from OCP.TopAbs import TopAbs_FACE
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopoDS import TopoDS

    kinds = {}
    area = {"exact": 0.0, "faceted": 0.0}
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
        else:
            key, exact = "other", True
        kinds[key] = kinds.get(key, 0) + 1
        area["exact" if exact else "faceted"] += a
        ex.Next()
    total = area["exact"] + area["faceted"] or 1.0
    return {"faces": kinds, "exactArea": round(area["exact"], 1), "facetedArea": round(area["faceted"], 1),
            "exactAreaFraction": round(area["exact"] / total, 4)}


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


def export_step(part, path, faces=DEFAULT_FACES, name=None, unify=True, check=True):
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
    from OCP.STEPControl import STEPControl_AsIs, STEPControl_Writer

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
    shape = _build(plan, budget, faces, notes)
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
    Interface_Static.SetCVal_s("write.step.unit", "MM")
    Interface_Static.SetCVal_s("write.step.schema", "AP214IS")
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
