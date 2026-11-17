"""
B-rep interfaces: the exact solids a part is built round.

Everything a part mates with is drawn, not grown: a port mouth, a pocket, a
snap, a boss a plunger presses on. An agent draws those in a B-rep kernel
(build123d or CadQuery, both OpenCascade; a STEP file from any CAD) and
hands them to loadpath, which keeps them as they were drawn: each becomes
a hard mask for whatever fills the space between them, and is added or cut
back exactly, last, with manifold3d. loadpath does not write a kernel and
never edits these solids.

`load_brep(src)` reads one:

- `"file.py:name"`: the module's `name`, called if it is a function of no
  arguments. A build123d / CadQuery object, an OpenCascade shape, a trimesh
  or a loadpath shape;
- `"file.step"`: sewn and read as it is;
- a mesh file (STL, 3MF, OBJ): an exact solid with no B-rep faces behind it
  (it can still be cut, added and checked; its radii are read off the mesh).

A `Brep` carries the OpenCascade shape (when there is one), a tessellation
true to `tolerance` mm with each triangle's B-rep face, and what the checks
read from it:

- `faces()`: per face, the surface type and the smallest radius of
  curvature on it (a plane has none): what a "fillet >= r" assertion reads,
  from the kernel's own second derivatives, not from a mesh;
- `sharp_edges()`: the edges where two faces meet at an angle: where a
  rounded mouth that "went square" shows;
- `samples(spacing)`: points and normals on the faces, with their face, for
  measuring how much of the drawn surface is on a delivered part.

Needs OpenCascade's Python bindings for B-rep sources: `pip install
build123d` (or loadpath[step]). Mesh sources need neither.
"""
from __future__ import annotations

import os

import numpy as np

__all__ = ["Brep", "load_brep", "outside_of", "DEFAULT_TOLERANCE"]

DEFAULT_TOLERANCE = 0.005     # mm: chord error of an interface's tessellation
SHARP_DEG = 20.0              # two faces meeting at more than this are a sharp edge
_MESH_EXT = (".stl", ".3mf", ".obj", ".ply", ".glb", ".gltf", ".off")
_STEP_EXT = (".step", ".stp")


def _occ(obj):
    from .exterior import _occ_shape

    return _occ_shape(obj)


class Brep:
    """An exact solid: `mesh` (closed trimesh), and, when it came from a
    kernel, `shape` (TopoDS_Shape), `owner` (B-rep face index per triangle)
    and `occ_faces`."""

    def __init__(self, mesh, shape=None, owner=None, occ_faces=None, source=None, tolerance=DEFAULT_TOLERANCE):
        self.mesh = mesh
        self.shape = shape
        self.owner = owner
        self.occ_faces = occ_faces or []
        self.source = source
        self.tolerance = float(tolerance)
        self._faces = None
        self._shape_sdf = None

    @property
    def exact(self) -> bool:
        """True when a kernel solid stands behind the mesh."""
        return self.shape is not None

    def manifold(self):
        from .shapes import manifold_from_trimesh

        return manifold_from_trimesh(self.mesh)

    def as_shape(self):
        """The solid as a loadpath shape (signed distance, bounds)."""
        if self._shape_sdf is None:
            from .shapes import MeshShape

            self._shape_sdf = MeshShape(self.mesh)
        return self._shape_sdf

    def bounds(self):
        b = np.asarray(self.mesh.bounds, float)
        return b[0].copy(), b[1].copy()

    # ---- what the kernel knows --------------------------------------------------

    def faces(self) -> list:
        """[{index, type, minRadius (mm, None for a plane), area, center}]
        per B-rep face; one row per smooth patch of a mesh-only solid."""
        if self._faces is not None:
            return self._faces
        rows = []
        if not self.exact:
            self._faces = _mesh_patches(self)
            return self._faces
        from OCP.BRepAdaptor import BRepAdaptor_Surface
        from OCP.BRepGProp import BRepGProp
        from OCP.BRepLProp import BRepLProp_SLProps
        from OCP.BRep import BRep_Tool
        from OCP.GProp import GProp_GProps
        from OCP.TopLoc import TopLoc_Location

        names = _surface_names()
        for i, f in enumerate(self.occ_faces):
            ad = BRepAdaptor_Surface(f)
            kind = names.get(ad.GetType(), "other")
            gp = GProp_GProps()
            BRepGProp.SurfaceProperties_s(f, gp)
            c = gp.CentreOfMass()
            row = {"index": i, "type": kind, "minRadius": None, "area": float(gp.Mass()),
                   "center": [c.X(), c.Y(), c.Z()]}
            if kind != "plane":
                # the smallest radius of curvature over the face, read where
                # its own triangulation has nodes (inside the trimmed face)
                tri = BRep_Tool.Triangulation_s(f, TopLoc_Location())
                k = 0.0
                if tri is not None and tri.HasUVNodes():
                    n = tri.NbNodes()
                    step = max(1, n // 400)
                    props = BRepLProp_SLProps(ad, 2, 1e-7)
                    for j in range(1, n + 1, step):
                        uv = tri.UVNode(j)
                        props.SetParameters(uv.X(), uv.Y())
                        if props.IsCurvatureDefined():
                            k = max(k, abs(props.MaxCurvature()), abs(props.MinCurvature()))
                row["minRadius"] = (1.0 / k) if k > 1e-9 else None
            rows.append(row)
        self._faces = rows
        return rows

    def face_triangles(self, index: int):
        """The triangles of one B-rep face (indices into mesh.faces)."""
        if self.owner is None:
            return np.arange(len(self.mesh.faces))
        return np.flatnonzero(self.owner == index)

    def sharp_edges(self, angle: float = SHARP_DEG):
        """(midpoints (n, 3), angles in degrees (n,)) of the mesh edges where
        the solid's surface turns by more than `angle`: between two B-rep
        faces that are not tangent, or anywhere on a mesh-only solid."""
        m = self.mesh
        adj = np.asarray(m.face_adjacency)
        if not len(adj):
            return np.zeros((0, 3)), np.zeros(0)
        ang = np.degrees(np.asarray(m.face_adjacency_angles))
        keep = ang > angle
        if self.owner is not None:
            keep &= self.owner[adj[:, 0]] != self.owner[adj[:, 1]]
        e = np.asarray(m.face_adjacency_edges)[keep]
        mid = np.asarray(m.vertices)[e].mean(axis=1)
        return mid, ang[keep]

    def samples(self, spacing: float = 0.2, faces=None):
        """(points, outward normals, B-rep face index) about `spacing` apart
        on the solid's surface (on the faces `faces` only, when given)."""
        m = self.mesh
        tri = np.arange(len(m.faces))
        if faces is not None and self.owner is not None:
            tri = np.flatnonzero(np.isin(self.owner, np.asarray(list(faces), int)))
        if not len(tri):
            return np.zeros((0, 3)), np.zeros((0, 3)), np.zeros(0, int)
        area = np.asarray(m.area_faces)[tri]
        n = int(np.clip(area.sum() / max(spacing, 1e-3) ** 2, 50, 400_000))
        rng = np.random.default_rng(0)
        pick = rng.choice(len(tri), size=n, p=area / area.sum())
        t = tri[pick]
        r1, r2 = np.sqrt(rng.random(n)), rng.random(n)
        V = np.asarray(m.vertices)[np.asarray(m.faces)[t]]
        P = (1 - r1)[:, None] * V[:, 0] + (r1 * (1 - r2))[:, None] * V[:, 1] + (r1 * r2)[:, None] * V[:, 2]
        own = self.owner[t] if self.owner is not None else np.zeros(n, int)
        return P, np.asarray(m.face_normals)[t], own


def _surface_names():
    from OCP.GeomAbs import (GeomAbs_BezierSurface, GeomAbs_BSplineSurface, GeomAbs_Cone, GeomAbs_Cylinder,
                             GeomAbs_Plane, GeomAbs_Sphere, GeomAbs_SurfaceOfExtrusion,
                             GeomAbs_SurfaceOfRevolution, GeomAbs_Torus)

    return {GeomAbs_Plane: "plane", GeomAbs_Cylinder: "cylinder", GeomAbs_Cone: "cone", GeomAbs_Sphere: "sphere",
            GeomAbs_Torus: "torus", GeomAbs_BezierSurface: "bezier", GeomAbs_BSplineSurface: "bspline",
            GeomAbs_SurfaceOfRevolution: "revolution", GeomAbs_SurfaceOfExtrusion: "extrusion"}


def mesh_edge_radii(mesh, min_angle: float = 2.0):
    """A mesh's own reading of how tightly it bends, edge by edge: for an
    edge its two faces turn by an angle a across (radians), with those faces
    reaching h1 and h2 from the edge, the radius of the arc they are chords
    of is (h1 + h2) / 2 / a. Returns (midpoints, radii mm, angles deg,
    edge lengths) for the edges that turn by more than `min_angle` degrees.
    A flat region has no such edges; a sharp edge reads a radius near zero."""
    adj = np.asarray(mesh.face_adjacency)
    if not len(adj):
        z = np.zeros(0)
        return np.zeros((0, 3)), z, z, z
    ang = np.asarray(mesh.face_adjacency_angles)
    keep = ang > np.radians(min_angle)
    adj, ang = adj[keep], ang[keep]
    e = np.asarray(mesh.face_adjacency_edges)[keep]
    V = np.asarray(mesh.vertices)
    a, b = V[e[:, 0]], V[e[:, 1]]
    length = np.linalg.norm(b - a, axis=1)
    area = np.asarray(mesh.area_faces)
    # a triangle's height over the shared edge, halved: the facet's centre
    # line sits that far from the edge
    h = (area[adj[:, 0]] + area[adj[:, 1]]) / np.maximum(length, 1e-12)
    radius = h / np.maximum(ang, 1e-9)
    return 0.5 * (a + b), radius, np.degrees(ang), length


def _mesh_patches(brep):
    """One row for a mesh-only solid: its tightest bend read off the mesh."""
    _, r, ang, length = mesh_edge_radii(brep.mesh)
    row = {"index": 0, "type": "mesh", "minRadius": None, "area": float(brep.mesh.area),
           "center": [float(x) for x in brep.mesh.centroid]}
    if len(r):
        row["minRadius"] = float(np.min(r))
    return [row]


def _resolve(path, base_dir):
    if base_dir is not None and not os.path.isabs(path):
        return os.path.join(base_dir, path)
    return path


def _load_py(path, name):
    import importlib.util
    import inspect
    import sys

    spec = importlib.util.spec_from_file_location(f"_loadpath_brep_{abs(hash(os.path.abspath(path)))}", path)
    mod = importlib.util.module_from_spec(spec)
    here = os.path.dirname(os.path.abspath(path))
    sys.path.insert(0, here)       # the file's own helpers beside it
    try:
        spec.loader.exec_module(mod)
    finally:
        if sys.path and sys.path[0] == here:
            sys.path.pop(0)
    if not name:
        raise ValueError(f"{path}: name the object to take, as 'file.py:name'")
    if not hasattr(mod, name):
        have = sorted(k for k in vars(mod) if not k.startswith("_"))
        raise ValueError(f"{path} defines no {name!r} (it has: {', '.join(have[:20])})")
    obj = getattr(mod, name)
    if callable(obj) and _occ(obj) is None and not hasattr(obj, "vertices"):
        try:
            need = [p for p in inspect.signature(obj).parameters.values()
                    if p.default is p.empty and p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]
        except (TypeError, ValueError):
            need = [None]
        if not need:
            obj = obj()
    return obj


_PY_CACHE: dict = {}


def load_brep(src, base_dir=None, tolerance: float = DEFAULT_TOLERANCE) -> Brep:
    """An exact solid from `file.py:name`, a STEP or mesh file, a build123d /
    CadQuery / OpenCascade object, a trimesh, or a loadpath shape (dict or
    object)."""
    import trimesh

    from .shapes import Shape, shape_from_spec, trimesh_from_manifold

    source = None
    if isinstance(src, (str, os.PathLike)):
        text = os.fspath(src)
        source = text
        if ".py:" in text or text.endswith(".py"):
            path, _, name = text.partition(".py")
            path = _resolve(path + ".py", base_dir)
            if not os.path.isfile(path):
                raise ValueError(f"interface source {text!r}: {path} not found")
            key = (os.path.abspath(path), os.path.getmtime(path), name.lstrip(":"))
            if key not in _PY_CACHE:
                _PY_CACHE[key] = _load_py(path, name.lstrip(":"))
            src = _PY_CACHE[key]
        else:
            path = _resolve(text, base_dir)
            if not os.path.isfile(path):
                raise ValueError(f"interface source {text!r}: {path} not found")
            ext = os.path.splitext(path)[1].lower()
            if ext in _STEP_EXT:
                from .step import read_step

                src = read_step(path)
            elif ext in _MESH_EXT:
                src = trimesh.load(path, force="mesh")
            else:
                raise ValueError(f"interface source {text!r}: use file.py:name, a STEP file or a mesh file")
    if isinstance(src, dict) and "type" in src:
        src = shape_from_spec(src, base_dir)
    if isinstance(src, Shape):
        from .sharp import exact_manifold

        m = trimesh_from_manifold(exact_manifold(src, tolerance))
        return Brep(_closed(m, source or "shape"), source=source, tolerance=tolerance)
    if isinstance(src, trimesh.Trimesh):
        m = src.copy()
        m.merge_vertices()
        return Brep(_closed(m, source or "mesh"), source=source, tolerance=tolerance)
    occ = _occ(src)
    if occ is None:
        raise ValueError(f"interface source {source or type(src).__name__!r} is not a solid loadpath can read "
                         "(a build123d or CadQuery object, a STEP file, a mesh, or a loadpath shape)")
    from .cadimport import _tessellate

    # angular deflection 0.1 rad keeps a small fillet's facets a few degrees
    # apart, so its mesh reads as a round, not as a row of edges
    mesh, owner, faces = _tessellate_fine(occ, tolerance)
    if len(mesh.faces) == 0:
        raise ValueError(f"interface source {source!r} has no surfaces")
    if not mesh.is_watertight:
        raise ValueError(f"interface source {source!r} is not a closed solid (its tessellation has open edges): "
                         "an interface is a solid, not a face or a sheet")
    if mesh.volume < 0:
        mesh.invert()
    return Brep(mesh, shape=occ, owner=owner, occ_faces=faces, source=source, tolerance=tolerance)


def _tessellate_fine(shape, tolerance):
    """cadimport._tessellate with a finer angular deflection, kept in step
    with it (the owner of each triangle, the faces in explorer order)."""
    import trimesh
    from OCP.BRep import BRep_Tool
    from OCP.BRepMesh import BRepMesh_IncrementalMesh
    from OCP.TopAbs import TopAbs_FACE, TopAbs_REVERSED
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopLoc import TopLoc_Location
    from OCP.TopoDS import TopoDS

    BRepMesh_IncrementalMesh(shape, tolerance, False, 0.1, True)
    V, F, owner, faces = [], [], [], []
    off = 0
    ex = TopExp_Explorer(shape, TopAbs_FACE)
    while ex.More():
        f = TopoDS.Face_s(ex.Current()) if hasattr(TopoDS, "Face_s") else TopoDS.Face(ex.Current())
        fi = len(faces)
        faces.append(f)
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
                owner.append(fi)
            off += tri.NbNodes()
        ex.Next()
    m = trimesh.Trimesh(np.asarray(V, float).reshape(-1, 3), np.asarray(F, np.int64).reshape(-1, 3), process=False)
    m.merge_vertices(digits_vertex=6)
    keep = m.nondegenerate_faces()
    owner = np.asarray(owner, np.int64)[keep]
    m.update_faces(keep)
    m.remove_unreferenced_vertices()
    return m, owner, faces


def _closed(mesh, what):
    if not mesh.is_watertight:
        raise ValueError(f"interface source {what!r} is not a closed mesh")
    if mesh.volume < 0:
        mesh.invert()
    return mesh


def outside_of(mesh, toward, center, size, spacing: float = 0.5, depth: float = 6.0, tolerance: float = 0.01):
    """The space just outside a mesh exterior, as a CAD solid: for drawing
    a round where a cut meets an exterior that came as a mesh (a scan, a
    sculpt, a designer's STL), which has no face to fillet against.

    Looking along -`toward` ('+y': the exterior's +y side) at a window
    `size` (two numbers, mm) round `center`, the exterior's surface is read
    on a grid of rays, a B-spline surface is laid through the points (within
    `tolerance` mm) and swept `depth` mm outward. Unite the result with the
    cut's own solid and fillet the edge where they meet: the round is then
    tangent to the cut and to the real exterior, and a kernel face with a
    radius to read. Returns (build123d solid, the largest distance in mm
    from the spline surface back to the mesh, which says how well the
    window can be a single smooth face). Needs build123d."""
    import trimesh
    from build123d import Face, Vector, extrude

    if isinstance(mesh, str):
        mesh = trimesh.load(mesh, force="mesh")
    sign = -1.0 if toward.startswith("-") else 1.0
    ax = "xyz".index(toward.lstrip("+-"))
    u, v = [i for i in range(3) if i != ax]
    out = np.zeros(3)
    out[ax] = sign
    c = np.asarray(center, float)
    nu = max(int(round(size[0] / spacing)), 4) + 1
    nv = max(int(round(size[1] / spacing)), 4) + 1
    us = c[u] + np.linspace(-size[0] / 2, size[0] / 2, nu)
    vs = c[v] + np.linspace(-size[1] / 2, size[1] / 2, nv)
    far = float(np.abs(mesh.bounds).max()) * 2 + 10.0
    rows = []
    for a in us:
        o = np.zeros((nv, 3))
        o[:, u], o[:, v], o[:, ax] = a, vs, sign * far
        loc, ray, _ = mesh.ray.intersects_location(o, np.tile(-out, (nv, 1)), multiple_hits=False)
        if len(ray) != nv:
            raise ValueError(f"the exterior does not cover the window at {'xyz'[u]} = {a:.2f}: move or shrink it")
        h = np.empty(nv)
        h[ray] = loc[:, ax]
        o[:, ax] = h
        rows.append([Vector(*map(float, q)) for q in o])
    face = Face.make_surface_from_array_of_points(rows, tol=tolerance)
    solid = extrude(face, amount=depth, dir=tuple(out))
    # how far the smooth face sits off the mesh it stands for
    fm, _, _ = _tessellate_fine(face.wrapped, 0.005)
    from trimesh.proximity import closest_point
    off = float(closest_point(mesh, np.asarray(fm.vertices))[1].max()) if len(fm.vertices) else 0.0
    return solid, off
