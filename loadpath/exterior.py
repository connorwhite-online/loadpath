"""
Any exterior in, one clean watertight skin out.

You or your agent draw the shape; loadpath makes everything inside it work.
The shape can come from anywhere, and each source has its own way of being
wrong for a solver:

- a designer's **STEP** (open shells, unsewn faces): tessellated, sewn,
  closed (cadimport);
- a **mesh** (STL, OBJ, PLY, 3MF, GLB/glTF): used as is when it is closed;
- an **AI-generated or scanned mesh** (Tripo, Meshy, a phone scan): open,
  self-intersecting, with duplicate inner shells, flipped patches, floating
  debris, at unit-cube scale and Y-up. It is *wrapped*: inside and outside
  are decided on a grid by the generalized winding number (libigl, the
  `exterior` extra) or, without it, by a flood fill from outside that
  seals holes up to `seal` mm, and the new surface is put back on the
  original triangles to a fraction of the grid pitch, so the surface
  detail stays where the generator put it;
- an **SDF**: a Python function `f(P) -> distance` over (N, 3) points in mm
  (negative inside; numpy, the `sdf` package, libfive-style expressions),
  a loadpath shape, or a sampled grid file (`.npz`, see `save_sdf`);
- a **build123d or CadQuery** object, tessellated through OpenCascade.

The result is an `Exterior`: the mesh, the transform from the source's
frame (scale to the real `size`, `up` axis to +z), and a report of what was
wrong with the source, what the wrap changed (how far the new surface sits
from the source's, where holes were patched, what was dropped), and where
the shape is too thin for anything to go inside it once the skin is in.

The exterior is never redesigned: the report says where it limits the
part, and that is the designer's to fix.

    from loadpath.exterior import load_exterior
    ext = load_exterior("generated.glb", size=72, up="+y")
    ext.save("out/")            # out/exterior.stl, out/exterior.json

    loadpath exterior generated.glb --size 72 --up +y --skin 1.2 --out out/
"""
from __future__ import annotations

import json
import os
import time

import numpy as np

__all__ = ["Exterior", "load_exterior", "wrap_mesh", "mesh_from_sdf", "save_sdf", "load_sdf", "sdf_of"]

MESH_EXT = (".stl", ".obj", ".ply", ".off", ".3mf", ".glb", ".gltf")
STEP_EXT = (".step", ".stp")
_CELL_BUDGET = 40_000_000


# ---------------------------------------------------------------- sources

def _as_points_fn(f):
    """A callable SDF, taking (N, 3) points, returning (N,) distances."""
    def g(P):
        P = np.asarray(P, float)
        try:
            d = f(P)
        except TypeError:
            d = f(P[:, 0], P[:, 1], P[:, 2])
        d = np.asarray(d, float).reshape(-1)
        if d.shape[0] != P.shape[0]:
            d = np.asarray(f(P[:, 0], P[:, 1], P[:, 2]), float).reshape(-1)
        return d
    return g


def _occ_shape(obj):
    """The OpenCascade TopoDS_Shape inside a build123d / CadQuery object."""
    if hasattr(obj, "val") and callable(obj.val) and not hasattr(obj, "wrapped"):   # cq.Workplane
        try:
            vals = obj.vals()
            if len(vals) > 1:
                import cadquery as cq
                return cq.Compound.makeCompound(vals).wrapped
            obj = obj.val()
        except Exception:
            obj = obj.val()
    w = getattr(obj, "wrapped", None)
    if w is not None and type(w).__module__.startswith("OCP"):
        return w
    if type(obj).__module__.startswith("OCP"):
        return obj
    return None


def _from_source(src, *, bounds=None, pitch=None, name=None):
    """(trimesh, kind, info, sdf or None) for any source."""
    import trimesh
    from .shapes import Shape
    info = {}
    if isinstance(src, (str, os.PathLike)):
        path = os.fspath(src)
        fn = None
        if ".py" in path and (path.endswith(".py") or ".py:" in path):
            path, _, fn = path.partition(".py")
            path += ".py"
            fn = fn.lstrip(":") or None
        if not os.path.isfile(path):
            raise ValueError(f"exterior: {path!r} not found")
        ext = os.path.splitext(path)[1].lower()
        info["file"] = os.path.abspath(path)
        if ext == ".py":
            obj = _load_py(path, fn)
            m, kind, sub, sdf = _from_source(obj, bounds=bounds, pitch=pitch)
            return m, f"python:{kind}", {**info, **sub, "function": fn or "exterior"}, sdf
        if ext in STEP_EXT:
            from .cadimport import load_part
            part = load_part(path)
            return part.mesh, "step", {**info, "healedLoops": part.heal.get("loops", 0)
                                       if isinstance(getattr(part, "heal", None), dict) else None}, None
        if ext == ".npz":
            g = load_sdf(path)
            lo, hi = g.bounds()
            return None, "sdf-grid", {**info, "pitch": g.pitch}, (g.sdf, (lo, hi), g.pitch)
        if ext in MESH_EXT:
            m = trimesh.load(path, force="mesh", process=False)
            if not isinstance(m, trimesh.Trimesh) or len(m.faces) == 0:
                raise ValueError(f"exterior: {path} holds no triangles")
            m.merge_vertices()        # STL stores each triangle's corners apart
            return m, "mesh", info, None
        raise ValueError(f"exterior: {ext} is not a known format (STEP, STL/OBJ/PLY/3MF/GLB, .npz SDF, .py)")
    if isinstance(src, trimesh.Scene):
        m = src.dump(concatenate=True)
        m.merge_vertices()
        return m, "mesh", info, None
    if isinstance(src, trimesh.Trimesh):
        m = src.copy()
        m.merge_vertices()
        return m, "mesh", info, None
    if isinstance(src, Shape):
        lo, hi = src.bounds()
        return None, "shape", info, (src.sdf, (lo, hi), None)
    if isinstance(src, dict) and "type" in src:
        from .shapes import shape_from_spec
        return _from_source(shape_from_spec(src), bounds=bounds, pitch=pitch)
    occ = _occ_shape(src)
    if occ is not None:
        from .cadimport import _tessellate, heal_mesh
        m, _, _ = _tessellate(occ, 0.02)
        m, rep = heal_mesh(m)
        return m, "brep", {**info, "healedLoops": rep.get("loops", 0)}, None
    if callable(src):
        f = _as_points_fn(src)
        b = bounds if bounds is not None else _estimate_bounds(f)
        return None, "sdf-function", {**info, "boundsEstimated": bounds is None}, (f, b, None)
    raise ValueError(f"exterior: cannot read a {type(src).__name__} (a path, trimesh, loadpath shape, "
                     "build123d/CadQuery object, or an SDF function of (N, 3) points)")


def _load_py(path, fn):
    """`file.py[:name]`: the module's `name` (default `exterior`); called if
    it takes no arguments (a builder), used as is otherwise (an SDF)."""
    import importlib.util
    import inspect
    spec = importlib.util.spec_from_file_location(f"_loadpath_exterior_{abs(hash(path))}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    name = fn or "exterior"
    if not hasattr(mod, name):
        raise ValueError(f"exterior: {path} defines no {name!r}")
    obj = getattr(mod, name)
    if callable(obj) and not hasattr(obj, "wrapped"):
        try:
            sig = inspect.signature(obj)
            if not [p for p in sig.parameters.values() if p.default is p.empty and
                    p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]:
                obj = obj()
        except (TypeError, ValueError):
            pass
    return obj


def _estimate_bounds(f, search: float = 2000.0, n: int = 24):
    """The box an SDF's solid sits in, without being told: sample a big box,
    keep the cells whose distance says the surface could be in them, box
    those, repeat until the box stops shrinking."""
    lo, hi = np.full(3, -search / 2), np.full(3, search / 2)
    for _ in range(40):
        axes = [np.linspace(lo[i], hi[i], n) for i in range(3)]
        P = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)
        step = (hi - lo) / (n - 1)
        r = 0.5 * float(np.linalg.norm(step))
        near = f(P) <= r
        if not near.any():
            raise ValueError("exterior: the SDF is positive everywhere in a 2 m box; give bounds=(lo, hi)")
        q = P[near]
        nlo, nhi = q.min(0) - step, q.max(0) + step
        if np.all(np.abs(nlo - lo) < 1e-3 * (hi - lo).max()) and np.all(np.abs(nhi - hi) < 1e-3 * (hi - lo).max()):
            break
        lo, hi = nlo, nhi
    return lo, hi


# ---------------------------------------------------------------- meshing

def _pitch_for(lo, hi, pitch=None, cells: int = 320):
    ext = np.asarray(hi, float) - np.asarray(lo, float)
    p = float(pitch) if pitch else max(float(ext.max()) / cells, 0.05)
    while np.prod(np.ceil(ext / p) + 8) > _CELL_BUDGET:
        p *= 1.15
    return p


def _grid(lo, hi, p, pad=4):
    lo = np.asarray(lo, float) - pad * p
    n = np.ceil((np.asarray(hi, float) + pad * p - lo) / p).astype(int) + 1
    # a sub-cell shift keeps flat faces off grid planes (marching cubes
    # slivers there)
    lo = lo - 0.013717 * p
    return lo, n


def _marching(F, lo, p):
    import trimesh
    from skimage import measure
    F = np.asarray(F, np.float32)
    F[F == 0] = -1e-4 * p
    v, f, _, _ = measure.marching_cubes(F, 0.0, spacing=(p, p, p))
    m = trimesh.Trimesh(v + lo, f[:, ::-1], process=False)
    m.merge_vertices()
    if m.volume < 0:
        m.invert()
    return m


def mesh_from_sdf(f, bounds, pitch=None):
    """Marching cubes of an SDF over `bounds`, closed by padding."""
    lo, hi = (np.asarray(b, float) for b in bounds)
    p = _pitch_for(lo, hi, pitch)
    org, n = _grid(lo, hi, p)
    axes = [org[i] + np.arange(n[i]) * p for i in range(3)]
    F = np.empty(tuple(n), np.float32)
    # slab by slab: a 300^3 grid's points would be 650 MB at once
    yz = np.stack(np.meshgrid(axes[1], axes[2], indexing="ij"), -1).reshape(-1, 2)
    for i, x in enumerate(axes[0]):
        P = np.column_stack([np.full(len(yz), x), yz])
        F[i] = np.asarray(f(P), float).reshape(n[1], n[2])
    F[0], F[-1], F[:, 0], F[:, -1], F[:, :, 0], F[:, :, -1] = (abs(p) * 3,) * 6
    return _marching(F, org, p), p


def _surface_samples(mesh, spacing):
    """Points on every triangle, no farther than `spacing` apart."""
    import trimesh
    area = float(mesh.area)
    n = int(min(max(area / (0.5 * spacing ** 2), 2000), 6_000_000))
    pts, _ = trimesh.sample.sample_surface(mesh, n, seed=0)
    return np.vstack([pts, mesh.vertices])


class _Dist:
    """Unsigned distance to a triangle soup: exact with libigl, else to a
    dense sample of its surface (error under a quarter pitch)."""

    def __init__(self, mesh, spacing):
        self.mesh = mesh
        try:
            import igl  # noqa: F401
            self.igl = True
        except ImportError:
            from scipy.spatial import cKDTree
            self.igl = False
            self.tree = cKDTree(_surface_samples(mesh, spacing))

    def __call__(self, P):
        P = np.asarray(P, float)
        if self.igl:
            import igl
            out = np.empty(len(P))
            V = np.asarray(self.mesh.vertices, float)
            F = np.asarray(self.mesh.faces, np.int64)
            for s in range(0, len(P), 2_000_000):
                d2 = igl.point_mesh_squared_distance(P[s:s + 2_000_000], V, F)[0]
                out[s:s + 2_000_000] = np.sqrt(np.maximum(d2, 0))
            return out
        return self.tree.query(P, workers=-1)[0]


def _inside_winding(mesh, org, n, p):
    import igl
    V = np.asarray(mesh.vertices, float)
    F = np.asarray(mesh.faces, np.int64)
    axes = [org[i] + np.arange(n[i]) * p for i in range(3)]
    inside = np.zeros(tuple(n), bool)
    yz = np.stack(np.meshgrid(axes[1], axes[2], indexing="ij"), -1).reshape(-1, 2)
    step = max(1, int(4_000_000 // len(yz)))
    for i in range(0, n[0], step):
        xs = axes[0][i:i + step]
        P = np.column_stack([np.repeat(xs, len(yz)), np.tile(yz, (len(xs), 1))])
        w = igl.fast_winding_number(V, F, P)
        inside[i:i + len(xs)] = (np.abs(w) > 0.5).reshape(len(xs), n[1], n[2])
    return inside


def _inside_flood(mesh, org, n, p, seal, regrow=1.0, enclosed=False):
    """Inside = not reachable from outside once every hole narrower than
    `seal` is plugged: the surface's cells grown by seal/2, outside flooded
    from the grid's border, then the outside grown back by the same."""
    from scipy import ndimage
    pts = _surface_samples(mesh, 0.5 * p)
    idx = np.clip(np.round((pts - org) / p).astype(int), 0, np.asarray(n) - 1)
    surf = np.zeros(tuple(n), bool)
    surf[idx[:, 0], idx[:, 1], idx[:, 2]] = True
    r = max(int(np.ceil(0.5 * seal / p)), 1)
    wall = ndimage.distance_transform_edt(~surf) <= r
    lbl, _ = ndimage.label(~wall)
    border = np.unique(np.concatenate([lbl[0].ravel(), lbl[-1].ravel(), lbl[:, 0].ravel(), lbl[:, -1].ravel(),
                                       lbl[:, :, 0].ravel(), lbl[:, :, -1].ravel()]))
    out = np.isin(lbl, border[border > 0])
    if enclosed:
        # only what the surface walls off, less the surface's own cells
        # (their side is the winding number's to say)
        return ~out & ~wall
    out = ndimage.distance_transform_edt(~out) <= r + regrow
    return ~out


def _orient(mesh):
    """Consistent winding within each connected sheet, each sheet facing
    out: a flipped patch would otherwise cancel the winding number under
    it and dent the wrap. libigl's breadth-first orientation; a sheet's
    outward sign from its signed volume."""
    import igl
    import trimesh
    m = mesh.copy()
    m.merge_vertices()
    V = np.asarray(m.vertices, float)
    F, C = igl.bfs_orient(np.asarray(m.faces, np.int64))
    F = np.asarray(F)
    C = np.asarray(C).reshape(-1)
    t = V[F]
    vol6 = np.einsum("ij,ij->i", t[:, 0], np.cross(t[:, 1], t[:, 2]))
    sv = np.bincount(C, weights=vol6)
    flip = sv[C] < 0
    F[flip] = F[flip][:, ::-1]
    return trimesh.Trimesh(V, F, process=False)


def wrap_mesh(mesh, pitch=None, method: str = "auto", seal: float | None = None, band: float = 3.0):
    """A watertight solid from any triangle soup, its surface back on the
    source's triangles. Returns (mesh, inside grid, (origin, pitch), info).

    method: "winding" (libigl's generalized winding number: open holes,
    self-intersections, duplicate and inverted shells all read right),
    "flood" (no libigl: holes up to `seal` mm, default 4 pitches, are
    sealed; wider ones leak and the wrap fails loudly), or "auto"."""
    from scipy import ndimage
    lo, hi = np.asarray(mesh.bounds, float)
    p = _pitch_for(lo, hi, pitch)
    org, n = _grid(lo, hi, p)
    if method == "auto":
        try:
            import igl  # noqa: F401
            method = "winding"
        except ImportError:
            method = "flood"
    seal = float(seal) if seal else 4 * p
    if method == "winding":
        mesh = _orient(mesh)
    if method == "winding":
        # winding number: right across open holes; and everything sealed
        # off from outside by the surface itself (a 1-cell flood): right
        # where folded or inverted inner sheets cancel the winding near the
        # skin (a generator's double wall under a tight rim)
        inside = _inside_winding(mesh, org, n, p)
    else:
        inside = _inside_flood(mesh, org, n, p, seal)
    # an exterior is solid: pockets sealed inside it are a generator's
    # folded inner walls (winding 0 between them), never a designed cavity;
    # the interior is loadpath's to design
    inside = ndimage.binary_fill_holes(inside)
    if inside.sum() < 8:
        raise ValueError("exterior: the wrap found no inside. The mesh may be a single open sheet, or (flood) its "
                         f"holes are wider than seal={seal:g} mm; install loadpath[exterior] (libigl) or raise seal")
    # sign from the inside grid, magnitude from the source's triangles near
    # the surface: the zero level lands on the source's own surface
    d_out = ndimage.distance_transform_edt(~inside)
    d_in = ndimage.distance_transform_edt(inside)
    F = ((d_out - d_in) * p).astype(np.float32)
    F -= np.float32(0.5 * p) * np.sign(F)
    near = np.abs(F) <= band * p
    idx = np.nonzero(near)
    P = org + np.column_stack(idx) * p
    dist = _Dist(mesh, 0.5 * p)(P)
    # holes: where the source has no triangle nearby the grid's distance
    # stands (the patch is where inside meets outside)
    sgn = np.where(inside[idx], -1.0, 1.0)
    exact = np.minimum(dist, np.abs(F[idx]))
    F[idx] = (sgn * np.maximum(exact, 1e-3 * p)).astype(np.float32)
    F[0], F[-1], F[:, 0], F[:, -1], F[:, :, 0], F[:, :, -1] = (3 * p,) * 6
    out = _marching(F, org, p)
    return out, inside, (org, p), {"method": method, "pitch": round(p, 4),
                                   **({"seal": round(seal, 3)} if method == "flood" else {})}


# ---------------------------------------------------------------- the exterior

def _defects(m):
    """What is wrong with a triangle soup, in numbers."""
    import trimesh
    e = np.sort(m.edges, axis=1)
    _, cnt = np.unique(e, axis=0, return_counts=True)
    bodies = m.split(only_watertight=False) if len(m.faces) < 2_000_000 else []
    inv = 0
    for b in bodies:
        if b.is_watertight and b.volume < 0:
            inv += 1
    deg = int((~m.nondegenerate_faces()).sum()) if hasattr(m, "nondegenerate_faces") else 0
    return {"faces": int(len(m.faces)), "watertight": bool(m.is_watertight),
            "boundaryEdges": int((cnt == 1).sum()), "nonManifoldEdges": int((cnt > 2).sum()),
            "bodies": len(bodies), "invertedBodies": inv, "degenerateFaces": deg,
            "windingConsistent": bool(m.is_winding_consistent),
            "extents": np.round(m.extents, 4).tolist()}


class Exterior:
    """A clean exterior: `mesh` (watertight, mm, +z up), `transform` (4 x 4,
    source frame to this one), `report`. `save(dir)` writes exterior.stl
    and exterior.json; `spec` is the study shape that names the STL."""

    def __init__(self, mesh, transform, report, grid=None):
        self.mesh, self.transform, self.report, self._grid = mesh, transform, report, grid
        self.path = None

    @property
    def spec(self):
        if self.path is None:
            raise ValueError("save() the exterior first: a study names it by its STL path")
        return {"type": "mesh", "path": self.path}

    def shape(self):
        from .shapes import MeshShape
        return MeshShape(self.mesh, self.path)

    def apply(self, mesh):
        """The same move (scale, up axis, placement) on a mesh in the
        source's frame: a component the designer drew next to it."""
        m = mesh.copy()
        m.apply_transform(self.transform)
        return m

    def save(self, out_dir, stem: str = "exterior"):
        os.makedirs(out_dir, exist_ok=True)
        stl = os.path.join(out_dir, stem + ".stl")
        self.mesh.export(stl)
        self.path = os.path.abspath(stl)
        rep = {**self.report, "stl": stl, "transform": np.round(self.transform, 9).tolist()}
        json.dump(rep, open(os.path.join(out_dir, stem + ".json"), "w"), indent=1, default=float)
        return stl


def _up_matrix(up):
    """The rotation that takes the source's `up` axis to +z."""
    s = str(up).strip().lower()
    s = s if s[:1] in "+-" else "+" + s
    a = {"x": 0, "y": 1, "z": 2}[s[1]]
    u = np.zeros(3)
    u[a] = 1.0 if s[0] == "+" else -1.0
    z = np.array([0, 0, 1.0])
    if np.allclose(u, z):
        return np.eye(4)
    if np.allclose(u, -z):
        R = np.diag([1.0, -1.0, -1.0])
    else:
        k = np.cross(u, z)
        k /= np.linalg.norm(k)
        K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
        R = np.eye(3) + K + K @ K      # 90 degrees: sin 1, 1 - cos 1
    T = np.eye(4)
    T[:3, :3] = R
    return T


def load_exterior(src, *, size: float | None = None, scale: float | None = None, up: str | None = None,
                  place: str | None = None, pitch: float | None = None, bounds=None, wrap: str = "auto",
                  method: str = "auto", seal: float | None = None, keep: str = "largest",
                  skin: float | None = None, max_faces: int | None = 400_000) -> Exterior:
    """Any exterior in, an `Exterior` out (see the module docstring).

    size: the real longest extent in mm (a generated mesh comes at unit
      scale); scale: a factor instead (0.001 m -> mm is scale=1000).
    up: the source's up axis ("+y" for glTF/GLB, Tripo's and most
      generators' convention; default +y for .glb/.gltf, else +z).
    place: "keep" (the source's frame, the default for a designer's CAD so
      the component CAD still lines up), "center" (x, y centred, resting on
      z = 0; the default when size, scale or up moved it), or "origin".
    wrap: "auto" (wrap only when the mesh is not one clean closed body),
      "always", or "never" (refuse a broken mesh).
    keep: "largest" body (drop floating debris) or "all".
    skin: the wall the study will keep (mm): the report says where the
      shape is too thin for anything but skin.
    """
    t0 = time.time()
    m, kind, info, sdf = _from_source(src, bounds=bounds, pitch=pitch)
    report = {"kind": kind, **info}
    ext_name = os.path.splitext(os.fspath(src))[1].lower() if isinstance(src, (str, os.PathLike)) else ""
    if up is None:
        up = "+y" if ext_name in (".glb", ".gltf") else "+z"
    T = _up_matrix(up)
    grid = None
    if sdf is not None:
        f, b, p0 = sdf
        m, p = mesh_from_sdf(f, b, pitch or p0)
        report["meshed"] = {"pitch": round(p, 4)}
        # how well the mesh sits on the SDF's zero level
        dv = np.abs(f(m.vertices[:: max(1, len(m.vertices) // 50_000)]))
        report["meshed"]["sdfAtVertices"] = {"p95": round(float(np.percentile(dv, 95)), 4),
                                             "max": round(float(dv.max()), 4)}
        src_defects = None
    else:
        src_defects = _defects(m)
        report["defects"] = src_defects
    # scale and orientation first: a wrap's pitch is chosen in mm
    s = 1.0
    if scale:
        s = float(scale)
    elif size:
        ext0 = (m.bounds[1] - m.bounds[0])
        s = float(size) / float(ext0.max())
    S = np.eye(4)
    S[:3, :3] *= s
    T = T @ S
    moved = s != 1.0 or not np.allclose(T, np.eye(4))
    m.apply_transform(T)
    ext_mm = float(m.extents.max())
    warn = []
    if not size and not scale and ext_mm < 5.0:
        warn.append(f"the shape is {ext_mm:.3g} across: unit scale or metres, not mm. Give size= (the real "
                    "longest extent in mm) or scale=")
    if place is None:
        place = "center" if moved else "keep"
    clean = src_defects is None or (src_defects["watertight"] and src_defects["bodies"] == 1
                                    and src_defects["nonManifoldEdges"] == 0 and src_defects["windingConsistent"])
    if src_defects is not None and wrap == "never" and not clean:
        raise ValueError(f"exterior: the mesh is not one clean closed body ({src_defects}); wrap it (wrap='auto')")
    if src_defects is not None and (wrap == "always" or (wrap == "auto" and not clean)):
        src = m
        w, inside, (org, p), winfo = wrap_mesh(src, pitch, method, seal)
        grid = (inside, org, p)
        report["wrap"] = {**winfo, **_wrap_deviation(src, w, p)}
        m = w
    elif src_defects is not None and m.volume < 0:
        m.invert()
    # bodies: keep the largest (debris off a generator)
    bodies = m.split(only_watertight=False)
    if len(bodies) > 1:
        # marching-cubes dust (a few cells, no volume) is not a body
        dust = (2.0 * report.get("meshed", {}).get("pitch", 0.0)) ** 3
        bodies = [b for b in bodies if abs(float(b.volume)) > dust] or bodies
        if len(bodies) == 1:
            m = bodies[0]
    if len(bodies) > 1:
        vols = np.array([abs(float(b.volume)) for b in bodies])
        order = np.argsort(-vols)
        if keep == "largest":
            m = bodies[order[0]]
            report["droppedBodies"] = [{"volume": round(float(vols[i]), 3),
                                        "center": np.round(bodies[i].bounds.mean(0), 2).tolist()}
                                       for i in order[1:21]]
            if len(order) > 21:
                report["droppedBodies"].append({"more": int(len(order) - 21)})
    if size and not scale and abs(float(m.extents.max()) - float(size)) > 1e-6 * float(size):
        # debris widened the source's box: the size is the kept body's
        k = float(size) / float(m.extents.max())
        c = m.bounds[0].copy()
        M = np.eye(4)
        M[:3, :3] *= k
        M[:3, 3] = c - k * c
        m.apply_transform(M)
        T = M @ T
        s *= k
    if place in ("center", "origin"):
        lo, hi = m.bounds
        off = -np.array([0.5 * (lo[0] + hi[0]), 0.5 * (lo[1] + hi[1]), lo[2]]) if place == "center" else -lo
        M = np.eye(4)
        M[:3, 3] = off
        m.apply_transform(M)
        if grid is not None:
            grid = (grid[0], grid[1] + off, grid[2])
        for d in report.get("droppedBodies", []):
            if "center" in d:
                d["center"] = np.round(np.asarray(d["center"]) + off, 2).tolist()
        if report.get("wrap", {}).get("patched", {}).get("largestAt"):
            report["wrap"]["patched"]["largestAt"] = np.round(
                np.asarray(report["wrap"]["patched"]["largestAt"]) + off, 2).tolist()
        T = M @ T
    if max_faces and len(m.faces) > max_faces:
        m = _decimate(m, max_faces, report)
    if not m.is_watertight:
        from .cadimport import heal_mesh
        m, _ = heal_mesh(m)
    report["result"] = {"faces": int(len(m.faces)), "watertight": bool(m.is_watertight),
                        "volume": round(float(m.volume), 3), "area": round(float(m.area), 2),
                        "bounds": np.round(m.bounds, 3).tolist(), "extents": np.round(m.extents, 3).tolist(),
                        "scale": s, "up": up, "place": place}
    if skin:
        report["capacity"] = capacity(m, float(skin), grid, T)
        warn += report["capacity"].pop("warnings", [])
    if report.get("wrap", {}).get("patched", {}).get("area", 0) > 0:
        pa = report["wrap"]["patched"]
        warn.append(f"{pa['area']:.0f} mm^2 of the surface was made up where the source had none (holes; "
                    f"largest around {pa['largestAt']}): check it is where the shape is closed")
    if report.get("droppedBodies"):
        warn.append(f"{len(report['droppedBodies'])} loose bodies dropped (largest "
                    f"{report['droppedBodies'][0].get('volume', 0):.3g} mm^3): keep='all' keeps them")
    report["warnings"] = warn
    report["seconds"] = round(time.time() - t0, 1)
    return Exterior(m, T, report, grid)


def _decimate(m, target, report):
    try:
        import fast_simplification
        import trimesh
        v, f = fast_simplification.simplify(np.asarray(m.vertices, np.float32), np.asarray(m.faces, np.int32),
                                            target_reduction=1.0 - target / len(m.faces))
        d = trimesh.Trimesh(v, f, process=True)
        if d.is_watertight:
            report["decimated"] = {"from": int(len(m.faces)), "to": int(len(d.faces))}
            return d
    except ImportError:
        pass
    return m


def _buried(out, pts, p):
    """Points more than 1.5 pitches inside a closed mesh (a distance grid
    of it, at the pitch)."""
    from scipy import ndimage
    from .verify import voxelize
    sol, org = voxelize(out, p)
    sol = np.pad(sol, 1)          # the grid's border is outside, not more solid
    org = org - p
    d = ndimage.distance_transform_edt(sol) * p
    k = np.floor((pts - org) / p).astype(int)
    ok = np.all((k >= 0) & (k < np.asarray(sol.shape)), axis=1)
    res = np.zeros(len(pts), bool)
    res[ok] = d[k[ok, 0], k[ok, 1], k[ok, 2]] > 1.5 * p
    return res


def _wrap_deviation(src, out, p):
    """How far the wrap moved the surface. out -> source: where the new
    surface has no source triangle near (patched holes). source -> out:
    source surface the wrap left behind (inner shells, debris, fins
    thinner than the grid)."""
    import trimesh
    from scipy.spatial import cKDTree
    # the skin of the source: its triangles not buried in the result (an
    # inner wall under a hole would read the hole as covered)
    deep = np.zeros(len(src.faces), bool)
    if len(out.faces) < 3_000_000:
        import trimesh.proximity
        c = src.triangles_center
        try:
            deep = trimesh.proximity.signed_distance(out, c) > 1.0 * p if len(c) < 60_000 else \
                _buried(out, c, p)
        except Exception:  # noqa: BLE001
            deep = np.zeros(len(src.faces), bool)
    skin = trimesh.Trimesh(src.vertices, src.faces[~deep], process=False) if (~deep).sum() > 10 else src
    d_out = _Dist(skin, 0.5 * p)
    ov = out.vertices[:: max(1, len(out.vertices) // 200_000)]
    a = d_out(ov)
    patch_tol = max(1.5 * p, 0.3)
    # patched area: faces whose corners are all off the source
    far_v = np.zeros(len(out.vertices), bool)
    if len(ov) == len(out.vertices):
        far_v = a > patch_tol
    else:
        far_v = d_out(out.vertices) > patch_tol
    far_f = far_v[out.faces].all(axis=1)
    area = float(out.area_faces[far_f].sum())
    largest = None
    if far_f.any():
        import trimesh
        sub = out.submesh([np.nonzero(far_f)[0]], append=True)
        parts = sub.split(only_watertight=False)
        big = max(parts, key=lambda q: q.area) if parts else sub
        largest = np.round(big.centroid, 2).tolist()
    pts = _surface_samples(src, 2 * p)[:300_000]
    tree = cKDTree(_surface_samples(out, 0.5 * p))
    b = tree.query(pts, workers=-1)[0]
    on = a[a <= patch_tol]
    left = b > patch_tol
    from .verify import voxelize  # noqa: F401
    ins = out.contains(pts[left]) if left.any() and len(out.faces) < 3_000_000 else np.zeros(0, bool)
    return {"onSource": {"p50": round(float(np.median(on)), 4) if len(on) else None,
                         "p95": round(float(np.percentile(on, 95)), 4) if len(on) else None},
            "patched": {"area": round(area, 1), "largestAt": largest, "beyond": round(patch_tol, 3),
                        "note": "new surface farther than `beyond` mm from any source triangle: holes"},
            "sourceLeftBehind": {"inside": round(float(ins.sum()) / len(pts), 4),
                                 "outside": round(float((~ins).sum()) / len(pts), 4) if len(ins) else
                                 round(float(left.mean()), 4),
                                 "note": "share of the source's surface: inside = inner shells, outside = "
                                         "debris and features thinner than the pitch"}}


def capacity(mesh, skin: float, grid=None, T=None, clearance: float = 0.5) -> dict:
    """Room inside: the cavity left once a `skin` wall lines the exterior,
    the largest ball that fits, and where the shape is too thin to hold
    anything (all skin there)."""
    from scipy import ndimage
    from .verify import voxelize
    p = min(max(float(mesh.extents.max()) / 240, skin / 3), 1.0)
    solid, org = voxelize(mesh, p)
    solid = np.pad(solid, 2)
    org = org - 2 * p
    edt = ndimage.distance_transform_edt(solid) * p
    cavity = edt > skin + clearance
    surf = solid & ~ndimage.binary_erosion(solid)
    d_cav = ndimage.distance_transform_edt(~cavity) * p if cavity.any() else np.full(solid.shape, np.inf)
    thin = surf & (d_cav > skin + clearance + 1.5 * p)
    vol = float(solid.sum()) * p ** 3
    k = np.unravel_index(int(np.argmax(edt)), edt.shape)
    out = {"skin": skin, "pitch": round(p, 3), "cavityVolume": round(float(cavity.sum()) * p ** 3, 1),
           "cavityShare": round(float(cavity.sum()) * p ** 3 / max(vol, 1e-9), 3),
           "largestBall": {"r": round(float(edt[k]), 2),
                           "center": np.round(org + (np.asarray(k) + 0.5) * p, 2).tolist()},
           "thinSurfaceShare": round(float(thin.sum()) / max(float(surf.sum()), 1), 3), "warnings": []}
    if thin.any():
        lbl, nl = ndimage.label(ndimage.binary_dilation(thin, iterations=1) & surf, structure=np.ones((3, 3, 3)))
        cnt = np.bincount(lbl.ravel())[1:]
        regions = []
        for k in np.argsort(-cnt)[:6]:
            ii = np.argwhere(lbl == k + 1)
            lo, hi = org + ii.min(0) * p, org + (ii.max(0) + 1) * p
            regions.append({"area": round(float(cnt[k]) * p * p, 1), "center": np.round(0.5 * (lo + hi), 1).tolist(),
                            "box": [np.round(lo, 1).tolist(), np.round(hi, 1).tolist()]})
        out["thinRegions"] = regions
        if out["thinSurfaceShare"] > 0.02:
            out["warnings"].append(f"{100 * out['thinSurfaceShare']:.0f}% of the surface is over sections thinner "
                                   f"than 2 x {skin:g} mm skin + {clearance:g} mm: solid there, nothing fits inside")
    if not cavity.any():
        out["warnings"].append(f"no room inside at all once a {skin:g} mm skin is in")
    return out


# ---------------------------------------------------------------- SDF files

def sdf_of(src, pitch=None, bounds=None):
    """(distance grid, origin, pitch) of any solid: a mesh, a loadpath
    shape, an SDF function. Node-sampled: grid[i, j, k] is the signed
    distance (mm, negative inside) at origin + (i, j, k) * pitch."""
    import trimesh
    from scipy import ndimage
    from .shapes import Shape
    if isinstance(src, (str, os.PathLike)):
        ext = os.path.splitext(os.fspath(src))[1].lower()
        if ext in MESH_EXT or ext in STEP_EXT:
            src = load_exterior(src, max_faces=None).mesh
        elif ext == ".npz":
            g = load_sdf(src)
            return g.grid, g.origin, g.pitch
    if isinstance(src, trimesh.Trimesh):
        lo, hi = src.bounds
        p = _pitch_for(lo, hi, pitch, cells=256)
        org, n = _grid(lo, hi, p)
        inside = _inside_winding(src, org, n, p) if _has_igl() else None
        if inside is None:
            from .verify import voxelize
            # ray parity on the closed mesh, at the grid's nodes
            sol, o2 = voxelize(src, p, box=(org - 0.5 * p, org + (n - 0.5) * p))
            inside = np.zeros(tuple(n), bool)
            s = tuple(min(a, b) for a, b in zip(sol.shape, n))
            inside[:s[0], :s[1], :s[2]] = sol[:s[0], :s[1], :s[2]]
        F = (ndimage.distance_transform_edt(~inside) - ndimage.distance_transform_edt(inside)) * p
        F -= 0.5 * p * np.sign(F)
        near = np.abs(F) <= 3 * p
        idx = np.nonzero(near)
        P = org + np.column_stack(idx) * p
        F[idx] = np.where(inside[idx], -1, 1) * _Dist(src, 0.5 * p)(P)
        return F.astype(np.float32), org, p
    if isinstance(src, Shape):
        f, b = src.sdf, src.bounds()
    elif callable(src):
        f = _as_points_fn(src)
        b = bounds if bounds is not None else _estimate_bounds(f)
    else:
        raise ValueError(f"sdf_of: cannot read a {type(src).__name__}")
    lo, hi = b
    p = _pitch_for(lo, hi, pitch, cells=256)
    org, n = _grid(lo, hi, p)
    axes = [org[i] + np.arange(n[i]) * p for i in range(3)]
    F = np.empty(tuple(n), np.float32)
    yz = np.stack(np.meshgrid(axes[1], axes[2], indexing="ij"), -1).reshape(-1, 2)
    for i, x in enumerate(axes[0]):
        F[i] = np.asarray(f(np.column_stack([np.full(len(yz), x), yz])), float).reshape(n[1], n[2])
    return F, org, p


def _has_igl():
    try:
        import igl  # noqa: F401
        return True
    except ImportError:
        return False


def save_sdf(src, path, pitch=None, bounds=None) -> dict:
    """Write a solid's signed distance as an .npz any SDF tool can read:
    `distance` (float32, nx x ny x nz, mm, negative inside), `origin` (mm,
    the node [0, 0, 0]), `pitch` (mm), `units` ("mm"). Trilinear between
    nodes. Loads back as a study shape {"type": "sdf", "path": ...}."""
    F, org, p = sdf_of(src, pitch, bounds)
    np.savez_compressed(path, distance=F.astype(np.float32), origin=np.asarray(org, float), pitch=float(p),
                        units="mm")
    return {"path": path, "shape": list(F.shape), "pitch": float(p), "origin": np.round(org, 4).tolist(),
            "insideNodes": int((F < 0).sum())}


def load_sdf(path):
    from .shapes import GridSDF
    return GridSDF.load(path)
