"""
Primitive shapes for study regions: the design space, preserved geometry,
keep-outs, and where supports and loads attach.

Every shape answers two questions, and they are deliberately separate:

- `sdf(points)`: signed distance in mm (negative inside). The voxel solver
  classifies elements and picks load/support nodes with it, so it must be
  vectorized over (N, 3) arrays.
- `to_manifold()`: the exact solid, for the final booleans. Marching cubes
  cannot hold a crease, so a bore cut from the density field comes out
  rounded and undersized; subtracting the exact cylinder afterwards is what
  keeps a bearing seat a bearing seat.

Shapes are plain JSON dicts on the wire (`shape_from_spec`), so an agent can
write a study without importing anything.
"""
from __future__ import annotations

import numpy as np

__all__ = [
    "Box", "Cylinder", "Sphere", "MeshShape", "Union", "Offset", "Shell", "Sweep", "grow", "shrink", "sweep",
    "shape_from_spec",
    "manifold_from_trimesh", "trimesh_from_manifold",
]

# Segments per full circle for exact cylinders. 96 keeps a 22mm bore's
# polygonal error at ~0.012mm, well inside any printer's tolerance.
_CIRCLE_SEGMENTS = 96


def _vec(v, n=3, name="value"):
    a = np.asarray(v, float).reshape(-1)
    if a.shape != (n,) or not np.all(np.isfinite(a)):
        raise ValueError(f"{name} must be {n} finite numbers, got {v!r}")
    return a


class Shape:
    kind = "shape"

    def sdf(self, p: np.ndarray) -> np.ndarray:  # pragma: no cover - abstract
        raise NotImplementedError

    def bounds(self) -> tuple[np.ndarray, np.ndarray]:  # pragma: no cover
        raise NotImplementedError

    def to_manifold(self):  # pragma: no cover - abstract
        raise NotImplementedError

    def contains(self, p: np.ndarray) -> np.ndarray:
        return self.sdf(p) <= 0.0

    def to_spec(self) -> dict:  # pragma: no cover - abstract
        raise NotImplementedError


class Box(Shape):
    """Axis-aligned box: {"type": "box", "min": [..], "max": [..]} or
    {"type": "box", "center": [..], "size": [..]}."""

    kind = "box"

    def __init__(self, lo, hi):
        self.lo = _vec(lo, name="box min")
        self.hi = _vec(hi, name="box max")
        if np.any(self.hi <= self.lo):
            raise ValueError(f"box max {self.hi.tolist()} must exceed min {self.lo.tolist()}")

    def sdf(self, p):
        # column by column: ~10x faster than the (n, 3) array form, and the
        # field graphs evaluate primitives tens of millions of times
        p = np.asarray(p, float)
        c = 0.5 * (self.lo + self.hi)
        h = 0.5 * (self.hi - self.lo)
        q0 = np.abs(p[..., 0] - c[0]) - h[0]
        q1 = np.abs(p[..., 1] - c[1]) - h[1]
        q2 = np.abs(p[..., 2] - c[2]) - h[2]
        a, b, d = np.maximum(q0, 0.0), np.maximum(q1, 0.0), np.maximum(q2, 0.0)
        return np.sqrt(a * a + b * b + d * d) + np.minimum(np.maximum(np.maximum(q0, q1), q2), 0.0)

    def bounds(self):
        return self.lo.copy(), self.hi.copy()

    def to_manifold(self):
        import manifold3d as m3d
        return m3d.Manifold.cube(tuple(self.hi - self.lo)).translate(tuple(self.lo))

    def to_spec(self):
        return {"type": "box", "min": self.lo.tolist(), "max": self.hi.tolist()}


class OrientedBox(Shape):
    """A box turned to any frame: {"type": "box", "center": [..], "size":
    [..], "axes": [[x axis], [y axis]]}, size measured along the two given
    axes and their cross product. A snap-fit hook on a curved wall, a rib
    across a seam: features that follow a surface, not the grid."""

    kind = "box"

    def __init__(self, center, size, axes):
        self.center = _vec(center, name="box center")
        self.size = _vec(size, name="box size")
        if np.any(self.size <= 0):
            raise ValueError(f"box size {self.size.tolist()} must be positive")
        ax = np.asarray(axes, float)
        if ax.shape != (2, 3):
            raise ValueError("box axes are two direction vectors [[x], [y]]; the third is their cross product")
        e0 = ax[0] / np.linalg.norm(ax[0])
        e1 = ax[1] - (ax[1] @ e0) * e0
        if np.linalg.norm(e1) < 1e-9:
            raise ValueError("box axes must not be parallel")
        e1 = e1 / np.linalg.norm(e1)
        self.R = np.stack([e0, e1, np.cross(e0, e1)])      # rows: the box's axes in world coordinates

    def sdf(self, p):
        p = np.asarray(p, float)
        q = np.abs((p - self.center) @ self.R.T) - 0.5 * self.size
        outside = np.linalg.norm(np.maximum(q, 0.0), axis=-1)
        return outside + np.minimum(q.max(axis=-1), 0.0)

    def bounds(self):
        ext = np.abs(self.R.T) @ (0.5 * self.size)
        return self.center - ext, self.center + ext

    def to_manifold(self):
        import manifold3d as m3d
        m = np.zeros((3, 4))
        m[:, :3] = self.R.T
        m[:, 3] = self.center
        return m3d.Manifold.cube(tuple(self.size), center=True).transform(m)

    def to_spec(self):
        return {"type": "box", "center": self.center.tolist(), "size": self.size.tolist(),
                "axes": self.R[:2].tolist()}


class HeightField(Shape):
    """One side of a surface that is a height over a plane: the points
    with n.x below (or above) h(u, v), h sampled on a square grid in the
    plane's (u, v) = (e1.x, e2.x) and bilinear between samples, held at the
    edge value past the grid. A stepped or design-line-following parting
    surface (shells.py): any such surface parts along n.

    {"type": "heightfield", "normal": [..], "axes": [[e1], [e2]],
     "origin": [u0, v0], "step": d, "heights": [[h(u0, v0), h(u0, v0 + d),
     ..], ..], "side": "below" | "above", "offset"?: mm, "depth": mm}

    `offset` moves the surface along n. `depth` bounds the region: how
    far it reaches past the surface. The distance is the height gap over
    sqrt(1 + |grad h|^2): exact on a plane, close where the surface tilts
    smoothly, an overestimate beside a steep step."""

    kind = "heightfield"

    def __init__(self, normal, axes, origin, step, heights, side="below", offset=0.0, depth=1000.0):
        n = np.asarray(normal, float)
        self.n = n / np.linalg.norm(n)
        ax = np.asarray(axes, float)
        e1 = ax[0] - (ax[0] @ self.n) * self.n
        e1 /= np.linalg.norm(e1)
        e2 = np.cross(self.n, e1)
        if ax[1] @ e2 < 0:
            e2 = -e2
        self.e1, self.e2 = e1, e2
        self.origin = np.asarray(origin, float).reshape(2)
        self.step = float(step)
        self.H = np.atleast_2d(np.asarray(heights, float))
        if self.H.shape[0] < 2 or self.H.shape[1] < 2:
            raise ValueError("heightfield heights need at least a 2 x 2 grid")
        if side not in ("below", "above"):
            raise ValueError("heightfield side is 'below' or 'above'")
        self.side = side
        self.offset = float(offset)
        self.depth = float(depth)
        gy, gx = np.gradient(self.H, self.step)
        self._G = (gy, gx)

    def _interp(self, P, F):
        uv = np.stack([P @ self.e1, P @ self.e2], -1)
        f = (uv - self.origin) / self.step
        nu, nv = self.H.shape
        fi = np.clip(f[..., 0], 0.0, nu - 1.000001)
        fj = np.clip(f[..., 1], 0.0, nv - 1.000001)
        i, j = np.floor(fi).astype(int), np.floor(fj).astype(int)
        a, b = fi - i, fj - j
        return ((1 - a) * (1 - b) * F[i, j] + a * (1 - b) * F[i + 1, j] + (1 - a) * b * F[i, j + 1]
                + a * b * F[i + 1, j + 1])

    def height(self, P):
        return self._interp(np.asarray(P, float), self.H) + self.offset

    def sdf(self, p):
        P = np.asarray(p, float)
        gap = P @ self.n - self.height(P)
        g2 = self._interp(P, self._G[0]) ** 2 + self._interp(P, self._G[1]) ** 2
        d = gap / np.sqrt(1.0 + g2)
        d = d if self.side == "below" else -d
        # bounded: `depth` past the surface
        far = (-gap - self.depth) if self.side == "below" else (gap - self.depth)
        return np.maximum(d, far)

    def _corners(self):
        nu, nv = self.H.shape
        u = self.origin[0] + np.array([0.0, (nu - 1) * self.step])
        v = self.origin[1] + np.array([0.0, (nv - 1) * self.step])
        lo_h, hi_h = float(self.H.min()) + self.offset, float(self.H.max()) + self.offset
        hs = (lo_h - self.depth, hi_h) if self.side == "below" else (lo_h, hi_h + self.depth)
        return np.array([uu * self.e1 + vv * self.e2 + hh * self.n for uu in u for vv in v for hh in hs])

    def bounds(self):
        c = self._corners()
        return c.min(axis=0), c.max(axis=0)

    def to_manifold(self):
        import manifold3d as m3d
        nu, nv = self.H.shape
        U = self.origin[0] + np.arange(nu) * self.step
        V = self.origin[1] + np.arange(nv) * self.step
        top = self.H + self.offset
        far = (top.min() - self.depth) if self.side == "below" else (top.max() + self.depth)
        uu, vv = np.meshgrid(U, V, indexing="ij")
        surf = uu[..., None] * self.e1 + vv[..., None] * self.e2 + top[..., None] * self.n
        flat = uu[..., None] * self.e1 + vv[..., None] * self.e2 + far * self.n
        verts = np.concatenate([surf.reshape(-1, 3), flat.reshape(-1, 3)])
        off = nu * nv
        idx = np.arange(off).reshape(nu, nv)
        tris = []
        a, b, c, d = idx[:-1, :-1].ravel(), idx[1:, :-1].ravel(), idx[1:, 1:].ravel(), idx[:-1, 1:].ravel()
        tris += [np.stack([a, b, c], 1), np.stack([a, c, d], 1)]
        tris += [np.stack([a, c, b], 1) + off, np.stack([a, d, c], 1) + off]
        ring = np.concatenate([idx[:, 0], idx[-1, 1:], idx[-2::-1, -1], idx[0, -2:0:-1]])
        r1 = np.roll(ring, -1)
        tris += [np.stack([ring, ring + off, r1], 1), np.stack([r1, ring + off, r1 + off], 1)]
        F = np.concatenate(tris).astype(np.uint32)
        mesh = m3d.Mesh(vert_properties=np.asarray(verts, np.float32), tri_verts=F)
        man = m3d.Manifold(mesh)
        if man.is_empty() or man.volume() <= 0:
            mesh = m3d.Mesh(vert_properties=np.asarray(verts, np.float32), tri_verts=F[:, ::-1].copy())
            man = m3d.Manifold(mesh)
        return man

    def to_spec(self):
        return {"type": "heightfield", "normal": self.n.tolist(), "axes": [self.e1.tolist(), self.e2.tolist()],
                "origin": self.origin.tolist(), "step": self.step,
                "heights": np.round(self.H, 4).tolist(), "side": self.side, "offset": self.offset,
                "depth": self.depth}


class Cylinder(Shape):
    """Flat-capped cylinder from p0 to p1 with radius r. With `rInner` it is a
    tube (a bearing boss, a bolt collar): material between rInner and r.
    {"type": "cylinder", "p0": [..], "p1": [..], "r": 11, "rInner"?: 6}.
    {"center", "axis": "x"|"y"|"z"|[..], "length", "r"} is accepted too."""

    kind = "cylinder"

    def __init__(self, p0, p1, r, r_inner=0.0):
        self.p0 = _vec(p0, name="cylinder p0")
        self.p1 = _vec(p1, name="cylinder p1")
        self.r = float(r)
        self.r_inner = float(r_inner or 0.0)
        self.length = float(np.linalg.norm(self.p1 - self.p0))
        if self.length <= 0:
            raise ValueError("cylinder p0 and p1 must differ")
        if self.r <= 0 or not (0 <= self.r_inner < self.r):
            raise ValueError("cylinder needs r > 0 and 0 <= rInner < r")
        self.axis = (self.p1 - self.p0) / self.length

    def sdf(self, p):
        p = np.asarray(p, float)
        ax = self.axis
        k = int(np.argmax(np.abs(ax)))
        if abs(ax[k]) == 1.0:
            # along x, y or z (the usual bore, boss or channel): two columns
            i, j = [x for x in range(3) if x != k]
            t = (p[..., k] - self.p0[k]) * ax[k]
            u, v = p[..., i] - self.p0[i], p[..., j] - self.p0[j]
            radial = np.sqrt(u * u + v * v)
        else:
            d = p - self.p0
            t = d @ ax
            radial = np.sqrt(np.maximum(np.einsum("...i,...i->...", d, d) - t * t, 0.0))
        half = 0.5 * self.length
        # 2D (radial, axial) box SDF; a tube is the band rInner..r.
        if self.r_inner > 0:
            mid = 0.5 * (self.r + self.r_inner)
            rad_q = np.abs(radial - mid) - 0.5 * (self.r - self.r_inner)
        else:
            rad_q = radial - self.r
        ax_q = np.abs(t - half) - half
        a, b = np.maximum(rad_q, 0.0), np.maximum(ax_q, 0.0)
        return np.sqrt(a * a + b * b) + np.minimum(np.maximum(rad_q, ax_q), 0.0)

    def bounds(self):
        # Exact AABB of a capped cylinder: each cap disc extends r*sqrt(1-a_i^2)
        # along axis i.
        ext = self.r * np.sqrt(np.clip(1.0 - self.axis ** 2, 0.0, 1.0))
        lo = np.minimum(self.p0, self.p1) - ext
        hi = np.maximum(self.p0, self.p1) + ext
        return lo, hi

    def _align(self, man):
        """Map a +Z cylinder standing on the origin onto p0 -> p1."""
        z = np.array([0.0, 0.0, 1.0])
        a = self.axis
        v = np.cross(z, a)
        s = np.linalg.norm(v)
        c = float(z @ a)
        if s < 1e-12:
            rot = np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
        else:
            vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
            rot = np.eye(3) + vx + vx @ vx * ((1 - c) / s ** 2)
        mat = np.hstack([rot, self.p0[:, None]])
        return man.transform(mat)

    def to_manifold(self):
        import manifold3d as m3d
        outer = m3d.Manifold.cylinder(self.length, self.r, self.r, _CIRCLE_SEGMENTS)
        if self.r_inner > 0:
            # Overshoot the inner cut so its caps never coincide with the outer.
            inner = m3d.Manifold.cylinder(self.length + 2.0, self.r_inner, self.r_inner,
                                          _CIRCLE_SEGMENTS).translate((0, 0, -1.0))
            outer = outer - inner
        return self._align(outer)

    def to_spec(self):
        out = {"type": "cylinder", "p0": self.p0.tolist(), "p1": self.p1.tolist(), "r": self.r}
        if self.r_inner > 0:
            out["rInner"] = self.r_inner
        return out


class Sphere(Shape):
    kind = "sphere"

    def __init__(self, center, r):
        self.center = _vec(center, name="sphere center")
        self.r = float(r)
        if self.r <= 0:
            raise ValueError("sphere r must be > 0")

    def sdf(self, p):
        p = np.asarray(p, float)
        x, y, z = (p[..., i] - self.center[i] for i in range(3))
        return np.sqrt(x * x + y * y + z * z) - self.r

    def bounds(self):
        return self.center - self.r, self.center + self.r

    def to_manifold(self):
        import manifold3d as m3d
        return m3d.Manifold.sphere(self.r, _CIRCLE_SEGMENTS).translate(tuple(self.center))

    def to_spec(self):
        return {"type": "sphere", "center": self.center.tolist(), "r": self.r}


class MeshShape(Shape):
    """An arbitrary closed mesh (STL/3MF/OBJ path, or a trimesh in code): the
    way an agent hands over geometry it already modeled — a motor body as a
    keep-out, a flange it drew in build123d as preserved material.

    Signed distance comes from a voxel grid built once (ray-parity occupancy
    plus distance transforms, trilinearly interpolated), so it costs the same
    per point as the analytic shapes. Accuracy is about half a grid cell,
    1/160 of the mesh's largest extent, which is far below the optimizer's
    resolution; the exact mesh is still what the final booleans use."""

    kind = "mesh"
    _GRID_CELLS = 160

    def __init__(self, mesh, path=None):
        import trimesh
        if not isinstance(mesh, trimesh.Trimesh):
            raise ValueError("MeshShape needs a trimesh.Trimesh")
        if not mesh.is_watertight:
            raise ValueError("mesh shape must be watertight (closed) to have an inside")
        self.mesh = mesh
        self.path = path
        self._grid = None
        self._band = 0.0

    def _build_grid(self):
        from scipy import ndimage
        from .verify import voxelize

        pitch = float(np.max(self.mesh.extents)) / self._GRID_CELLS
        solid, origin = voxelize(self.mesh, pitch)
        pad = 4
        solid = np.pad(solid, pad)
        origin = origin - pad * pitch
        dist = (ndimage.distance_transform_edt(~solid) - ndimage.distance_transform_edt(solid)) * pitch
        dist -= 0.5 * pitch * np.sign(dist)   # EDT is centre to centre; the surface is between
        if self._band:
            dist = self._exact_band(dist, origin, pitch, self._band)
        self._grid = (dist, origin, pitch)

    def _exact_band(self, dist, origin, pitch, width):
        """The grid's distances within `width` mm of the surface made exact
        (closest point on the faces), cached on disk by mesh and grid. The
        voxel distances are good to half a cell, 0.37 mm on the touch
        microphone: a 1.0 mm skin offset from them printed 0.65-1.18 mm
        thick, in ridges a cell apart. Interpolated between exact nodes the
        error is about h^2 / 8R, a hundredth of a millimetre."""
        import hashlib
        import os
        key = hashlib.sha1(np.ascontiguousarray(self.mesh.vertices, np.float64).tobytes()
                           + np.ascontiguousarray(self.mesh.faces, np.int64).tobytes()
                           + np.array([pitch, width, *origin, *dist.shape], float).tobytes()).hexdigest()[:20]
        root = os.environ.get("LOADPATH_SDF_CACHE", os.path.join(os.path.expanduser("~"), ".cache", "loadpath", "sdf"))
        path = os.path.join(root, key + ".npy")
        try:
            return np.load(path)
        except (OSError, ValueError):
            pass
        band = np.abs(dist) <= width + pitch
        idx = np.argwhere(band)
        pts = origin + (idx + 0.5) * pitch
        coarse = dist[band]
        self._grid = (dist, origin, pitch)
        exact = self.exact_sdf(pts, grid_sign=coarse)
        out = dist.copy()
        out[band] = exact
        try:
            os.makedirs(root, exist_ok=True)
            np.save(path, out)
        except OSError:
            pass
        return out

    def exact_band(self, width: float):
        """Ask for exact distances within `width` mm of the surface (an
        offset surface that deep is traced on them). Rebuilds the grid."""
        if width > self._band:
            self._band = float(width)
            self._grid = None

    def sdf(self, p):
        from scipy import ndimage
        if self._grid is None:
            self._build_grid()
        dist, origin, pitch = self._grid
        p = np.asarray(p, float).reshape(-1, 3)
        coords = ((p - origin) / pitch - 0.5).T
        inside_grid = np.all((coords >= 0) & (coords <= (np.asarray(dist.shape)[:, None] - 1)), axis=0)
        out = ndimage.map_coordinates(dist, coords, order=1, mode="nearest")
        # Beyond the padded grid, distance to the box is a fine lower bound.
        lo, hi = self.bounds()
        q = np.maximum(np.maximum(lo - p, p - hi), 0.0)
        return np.where(inside_grid, out, np.maximum(np.linalg.norm(q, axis=1), 4 * pitch))

    def contains(self, p):
        return self.sdf(p) <= 0.0

    def exact_sdf(self, p, chunk=20_000, grid_sign=None):
        """Signed distance to the mesh itself (closest point on its faces),
        for where half a grid cell matters: an offset surface 1 mm inside
        this mesh, which the grid sdf put up to a sixth of a millimetre off,
        in ridges a grid cell apart. Sign from the grid sdf, which is right
        away from the surface, and from ray parity within a cell of it."""
        from trimesh.proximity import closest_point
        p = np.asarray(p, float).reshape(-1, 3)
        out = np.empty(len(p))
        if grid_sign is None:
            g = self.sdf(p)
        else:
            g = np.asarray(grid_sign, float)
        cell = self._grid[2] if self._grid is not None else float(np.max(self.mesh.extents)) / self._GRID_CELLS
        try:
            import igl
        except ImportError:
            igl = None
        if igl is not None:
            # libigl (loadpath[exterior]): an AABB tree in C++, two orders
            # faster than trimesh's closest_point on a dense wrapped skin
            V = np.asarray(self.mesh.vertices, float)
            F = np.asarray(self.mesh.faces, np.int64)
            for s in range(0, len(p), 1_000_000):
                q = p[s:s + 1_000_000]
                d = np.sqrt(np.maximum(igl.point_mesh_squared_distance(q, V, F)[0], 0.0))
                sign = np.sign(g[s:s + 1_000_000])
                near = np.abs(g[s:s + 1_000_000]) < 1.5 * cell
                if near.any():
                    sign[near] = np.where(igl.fast_winding_number(V, F, q[near]) > 0.5, -1.0, 1.0)
                out[s:s + 1_000_000] = sign * d
            return out
        for s in range(0, len(p), chunk):
            q = p[s:s + chunk]
            _, d, _ = closest_point(self.mesh, q)
            sign = np.sign(g[s:s + chunk])
            near = np.abs(g[s:s + chunk]) < 1.5 * cell
            if near.any():
                sign[near] = np.where(self.mesh.contains(q[near]), -1.0, 1.0)
            out[s:s + chunk] = sign * d
        return out

    def bounds(self):
        b = np.asarray(self.mesh.bounds, float)
        return b[0].copy(), b[1].copy()

    def to_manifold(self):
        return manifold_from_trimesh(self.mesh)

    def to_spec(self):
        if self.path is None:
            raise ValueError("an in-memory mesh shape has no JSON form")
        return {"type": "mesh", "path": self.path}


_AXES = {"x": [1.0, 0.0, 0.0], "y": [0.0, 1.0, 0.0], "z": [0.0, 0.0, 1.0]}


class GridSDF(Shape):
    """A sampled signed distance field: {"type": "sdf", "path": "x.npz"}, the
    file `exterior.save_sdf` writes (distance in mm on a node grid, negative
    inside; origin; pitch). How an SDF tool's shape (libfive, the `sdf`
    package, a field graph) comes in without code in the study, and how an
    interior goes back out to one."""

    kind = "sdf"

    def __init__(self, grid, origin, pitch, path=None):
        self.grid = np.asarray(grid, np.float32)
        self.origin = np.asarray(origin, float)
        self.pitch = float(pitch)
        self.path = path
        self._mesh = None

    @classmethod
    def load(cls, path, name=None, grid=None):
        if str(path).lower().endswith(".vdb"):
            from .vdb import read_vdb
            gs = read_vdb(path)
            if grid is not None:
                gs = [g for g in gs if g["name"] == grid] or gs[:0]
            g = next((g for g in gs if g["class"] == "level set"), gs[0] if gs else None)
            if g is None:
                raise ValueError(f"sdf file {path!r} has no grid {grid!r}")
            return cls(g["values"], g["origin"], g["pitch"], name or path)
        z = np.load(path)
        for k in ("distance", "origin", "pitch"):
            if k not in z:
                raise ValueError(f"sdf file {path!r} has no {k!r} (see loadpath.exterior.save_sdf)")
        if "units" in z and str(z["units"]) != "mm":
            raise ValueError(f"sdf file {path!r}: units must be mm")
        return cls(z["distance"], z["origin"], float(z["pitch"]), name or path)

    def sdf(self, p):
        from scipy import ndimage
        p = np.asarray(p, float).reshape(-1, 3)
        coords = ((p - self.origin) / self.pitch).T
        n = np.asarray(self.grid.shape)[:, None]
        inside = np.all((coords >= 0) & (coords <= n - 1), axis=0)
        out = ndimage.map_coordinates(self.grid, coords, order=1, mode="nearest")
        lo, hi = self.bounds()
        q = np.maximum(np.maximum(lo - p, p - hi), 0.0)
        return np.where(inside, out, np.maximum(np.linalg.norm(q, axis=1), self.pitch))

    def contains(self, p):
        return self.sdf(p) <= 0.0

    def bounds(self):
        return self.origin.copy(), self.origin + (np.asarray(self.grid.shape) - 1) * self.pitch

    def mesh(self):
        if self._mesh is None:
            from .exterior import _marching
            F = self.grid.copy()
            F[0], F[-1], F[:, 0], F[:, -1], F[:, :, 0], F[:, :, -1] = (3 * self.pitch,) * 6
            self._mesh = _marching(F, self.origin, self.pitch)
        return self._mesh

    def to_manifold(self):
        return manifold_from_trimesh(self.mesh())

    def to_spec(self):
        if self.path is None:
            raise ValueError("an in-memory SDF grid has no JSON form; save it with exterior.save_sdf")
        return {"type": "sdf", "path": self.path}


class Union(Shape):
    """Several shapes as one region: {"type": "union", "shapes": [..]}. For
    an interface that is several separate pieces, such as the four screw
    bosses a servo mounts on, so one load or support can act on all of them."""

    kind = "union"

    def __init__(self, shapes):
        if not shapes:
            raise ValueError("union needs at least one shape in 'shapes'")
        self.shapes = list(shapes)

    def sdf(self, p):
        out = self.shapes[0].sdf(p)
        for s in self.shapes[1:]:
            out = np.minimum(out, s.sdf(p))
        return out

    def bounds(self):
        b = [s.bounds() for s in self.shapes]
        return np.min([x[0] for x in b], axis=0), np.max([x[1] for x in b], axis=0)

    def to_manifold(self):
        import manifold3d as m3d
        mans = [s.to_manifold() for s in self.shapes]
        return m3d.Manifold.batch_boolean(mans, m3d.OpType.Add) if len(mans) > 1 else mans[0]

    def to_spec(self):
        return {"type": "union", "shapes": [s.to_spec() for s in self.shapes]}


class Boolean(Shape):
    """The overlap of shapes ({"type": "intersect", "shapes": [..]}) or the
    first with the rest cut away ({"type": "subtract", "shapes": [..]}): a
    half ring is a tube intersected with a box, a ring with a gap a tube
    minus a box. Brief-eval agents built these as STLs by hand."""

    def __init__(self, op, shapes):
        if op not in ("intersect", "subtract"):
            raise ValueError(f"unknown boolean {op!r}")
        if len(shapes) < 2:
            raise ValueError(f"{op} needs at least two shapes in 'shapes'")
        self.kind = op
        self.parts = list(shapes)

    def sdf(self, p):
        out = self.parts[0].sdf(p)
        for s in self.parts[1:]:
            out = np.maximum(out, s.sdf(p)) if self.kind == "intersect" else np.maximum(out, -s.sdf(p))
        return out

    def bounds(self):
        if self.kind == "subtract":
            return self.parts[0].bounds()
        b = [s.bounds() for s in self.parts]
        lo, hi = np.max([x[0] for x in b], axis=0), np.min([x[1] for x in b], axis=0)
        if np.any(hi <= lo):
            raise ValueError("intersect: the shapes do not overlap")
        return lo, hi

    def to_manifold(self):
        import manifold3d as m3d
        mans = [s.to_manifold() for s in self.parts]
        if self.kind == "intersect":
            return m3d.Manifold.batch_boolean(mans, m3d.OpType.Intersect)
        return mans[0] - m3d.Manifold.batch_boolean(mans[1:], m3d.OpType.Add)

    def to_spec(self):
        return {"type": self.kind, "shapes": [s.to_spec() for s in self.parts]}


class Offset(Shape):
    """Another shape grown (offset > 0) or shrunk (offset < 0) by a distance:
    the level set sdf = offset. `open` lists bounding-box faces, as
    (axis, sign), that a shrunk shape runs out through instead of stopping
    short of: the shape is extruded straight out of each one from a cross
    section taken 2|offset| inside it. That is how a skin is left off the
    face a cover's ribs stand on.

    Used where no exact primitive does it (`grow`, `shrink`); the exact solid
    is marching cubes of the level set at ~1/200 of the shape's size."""

    kind = "offset"
    _CELLS = 200

    def __init__(self, base: Shape, offset: float, open=()):
        self.base = base
        self.offset = float(offset)
        self.open = [(int(a), int(np.sign(s))) for a, s in open]
        if isinstance(base, MeshShape):
            # its level set is traced on the grid sdf this deep
            base.exact_band(abs(self.offset) + 0.5)

    def _clamp(self, p):
        if not self.open:
            return p
        p = np.array(p, float, copy=True)
        lo, hi = self.base.bounds()
        inset = 2.0 * abs(self.offset)
        for a, s in self.open:
            if s < 0:
                p[..., a] = np.maximum(p[..., a], lo[a] + inset)
            else:
                p[..., a] = np.minimum(p[..., a], hi[a] - inset)
        return p

    def sdf(self, p):
        return self.base.sdf(self._clamp(np.asarray(p, float))) - self.offset

    def bounds(self):
        lo, hi = self.base.bounds()
        lo, hi = lo - self.offset, hi + self.offset
        reach = 2.0 * abs(self.offset) + 1.0
        for a, s in self.open:
            if s < 0:
                lo[a] -= reach
            else:
                hi[a] += reach
        return lo, hi

    def to_manifold(self):
        import trimesh
        from skimage import measure

        lo, hi = self.bounds()
        step = float(np.max(hi - lo)) / self._CELLS
        pad = 2
        dims = np.ceil((hi - lo) / step).astype(int) + 1 + 2 * pad
        origin = lo - pad * step
        axes = [origin[i] + np.arange(dims[i]) * step for i in range(3)]
        pts = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)
        # the grid's own border counts as outside, so the surface closes
        vals = self.sdf(pts)
        vals = vals.reshape(tuple(dims))
        vals[[0, -1], :, :] = vals[:, [0, -1], :] = vals[:, :, [0, -1]] = step
        verts, faces, _, _ = measure.marching_cubes(vals, level=0.0, spacing=(step, step, step))
        mesh = trimesh.Trimesh(verts + origin, faces[:, ::-1], process=True)
        if mesh.volume < 0:
            mesh.invert()
        return manifold_from_trimesh(mesh)

    def to_spec(self):
        raise ValueError("an offset shape is derived from the study and has no JSON form of its own")


class Thickened(Offset):
    """An Offset shrunk further by a thickness field: sdf + extra(p), with
    extra(p) >= 0 in mm (a callable on points). A skin's inner surface moved
    inward where a design thickens the skin, as one smooth surface instead
    of the cells the design grid gave it (switch._skin_lining)."""

    kind = "thickened"

    def __init__(self, base: Offset, extra):
        super().__init__(base.base, base.offset, [(a, s) for a, s in base.open])
        if isinstance(base, Thickened):
            # a skin of zoned thickness thickened again by a lining: both hold
            first = base._more
            self._more = lambda p: first(p) + extra(p)
        else:
            self._more = extra

    def thickness_at(self, p):
        """The skin thickness at points (mm): the shrink plus the field."""
        return -self.offset + self._more(np.asarray(p, float))

    def sdf(self, p):
        p = np.asarray(p, float)
        return super().sdf(p) + self._more(p)


class Shell(Shape):
    """The material of `outer` that is not in `inner`, kept inside `clip`:
    a keep-out's wall (its grown copy minus itself, inside the design space)
    or a design space's skin (itself minus its shrunk copy)."""

    kind = "shell"

    def __init__(self, outer: Shape, inner: Shape, clip: Shape | None = None):
        self.outer = outer
        self.inner = inner
        self.clip = clip

    def sdf(self, p):
        d = np.maximum(self.outer.sdf(p), -self.inner.sdf(p))
        return d if self.clip is None else np.maximum(d, self.clip.sdf(p))

    def bounds(self):
        lo, hi = self.outer.bounds()
        if self.clip is not None:
            clo, chi = self.clip.bounds()
            lo, hi = np.maximum(lo, clo), np.minimum(hi, chi)
        return lo, hi

    def to_manifold(self):
        man = self.outer.to_manifold() - self.inner.to_manifold()
        return man if self.clip is None else man ^ self.clip.to_manifold()

    def to_spec(self):
        raise ValueError("a shell is derived from the study and has no JSON form of its own")


def grow(shape: Shape, t: float) -> Shape:
    """`shape` grown outward by t mm. Boxes and cylinders stay sharp-edged
    (a little more than a true offset at their edges, never less)."""
    if isinstance(shape, Box):
        return Box(shape.lo - t, shape.hi + t)
    if isinstance(shape, OrientedBox):
        return OrientedBox(shape.center, shape.size + 2 * t, shape.R[:2])
    if isinstance(shape, Cylinder):
        a = shape.axis
        return Cylinder(shape.p0 - t * a, shape.p1 + t * a, shape.r + t, max(shape.r_inner - t, 0.0))
    if isinstance(shape, Sphere):
        return Sphere(shape.center, shape.r + t)
    if isinstance(shape, Union):
        return Union([grow(s, t) for s in shape.shapes])
    return Offset(shape, t)


def shrink(shape: Shape, t: float, open=()) -> Shape:
    """`shape` shrunk inward by t mm, running out through the `open`
    bounding-box faces ((axis, sign) pairs)."""
    open = list(open)
    if isinstance(shape, Box):
        lo, hi = shape.lo + t, shape.hi - t
        for a, s in open:
            if s < 0:
                lo[a] = shape.lo[a] - t
            else:
                hi[a] = shape.hi[a] + t
        if np.any(hi <= lo):
            raise ValueError(f"a {t:g} mm skin fills the whole box; make it thinner")
        return Box(lo, hi)
    if isinstance(shape, Sphere) and not open:
        if shape.r <= t:
            raise ValueError(f"a {t:g} mm skin fills the whole sphere; make it thinner")
        return Sphere(shape.center, shape.r - t)
    return Offset(shape, -t, open)


class Sweep(Shape):
    """The space a shape passes through moving `length` mm along `direction`
    from where it is drawn: a component's assembly path (a board slid in from
    below, a switch dropped onto it, a driver reaching a screw). As a
    keep-out it keeps generated structure out of the way of assembly, not
    only of the component's final place.

    Boxes and cylinders moving along an axis of their own are swept exactly
    (`sweep` returns a longer box or cylinder); anything else is the union of
    copies a fraction of a millimetre apart, so a curved shape moving
    sideways reads as a slightly scalloped slot."""

    kind = "sweep"

    def __init__(self, base: Shape, direction, length: float, step: float | None = None):
        d = np.asarray(direction, float)
        self.base = base
        self.direction = d / np.linalg.norm(d)
        self.length = float(length)
        n = max(int(np.ceil(self.length / (step or max(self.length / 200.0, 0.1)))), 1)
        self.offsets = np.linspace(0.0, self.length, n + 1)

        self._grid = None

    def _mesh_base(self):
        """(MeshShape, offset) when the base is a mesh, maybe moved."""
        b, off = self.base, np.zeros(3)
        if isinstance(b, Moved):
            b, off = b.base, b.offset
        return (b, off) if isinstance(b, MeshShape) else (None, None)

    def _build_grid(self):
        """A mesh swept along an axis: its distance grid's occupancy smeared
        along the axis, then a distance transform, once. Evaluating ~200
        shifted copies point by point instead took 14 GB and minutes in the
        delivered mesh's isosurface (a board's two-leg path on the touch
        microphone)."""
        from scipy import ndimage
        m, off = self._mesh_base()
        if m._grid is None:
            m._build_grid()
        dist, origin, pitch = m._grid
        k = int(np.argmax(np.abs(self.direction)))
        n = int(round(self.length / pitch))
        solid = dist <= 0
        pad = [(0, 0)] * 3
        sg = 1 if self.direction[k] > 0 else -1
        pad[k] = (0, n) if sg > 0 else (n, 0)
        solid = np.pad(solid, pad)
        org = origin.astype(float).copy() + off
        if sg < 0:
            org[k] -= n * pitch
        # a cell is in the sweep if the base covered it from up to n cells
        # back along the motion: a sliding window along the axis
        c = np.cumsum(solid.astype(np.int32), axis=k)
        c = np.concatenate([np.zeros_like(np.take(c, [0], axis=k)), c], axis=k)
        N = solid.shape[k]
        idx = np.arange(N)
        if sg > 0:
            hi_i, lo_i = idx + 1, np.maximum(idx - n, 0)
        else:
            hi_i, lo_i = np.minimum(idx + n + 1, N), idx
        swept = (np.take(c, hi_i, axis=k) - np.take(c, lo_i, axis=k)) > 0
        d = (ndimage.distance_transform_edt(~swept) - ndimage.distance_transform_edt(swept)) * pitch
        d -= 0.5 * pitch * np.sign(d)
        self._grid = (d, org, pitch)

    def sdf(self, p):
        p = np.asarray(p, float)
        m, _ = self._mesh_base()
        if m is not None and np.max(np.abs(self.direction)) > 1 - 1e-9:
            from scipy import ndimage
            if self._grid is None:
                self._build_grid()
            dist, origin, pitch = self._grid
            p = p.reshape(-1, 3)
            coords = ((p - origin) / pitch - 0.5).T
            inside = np.all((coords >= 0) & (coords <= (np.asarray(dist.shape)[:, None] - 1)), axis=0)
            out = ndimage.map_coordinates(dist, coords, order=1, mode="nearest")
            lo, hi = self.bounds()
            q = np.maximum(np.maximum(lo - p, p - hi), 0.0)
            return np.where(inside, out, np.maximum(np.linalg.norm(q, axis=1), 4 * pitch))
        out = None
        for t in self.offsets:
            v = self.base.sdf(p - t * self.direction)
            out = v if out is None else np.minimum(out, v)
        return out

    def bounds(self):
        lo, hi = self.base.bounds()
        end = self.length * self.direction
        return np.minimum(lo, lo + end), np.maximum(hi, hi + end)

    def to_manifold(self):
        import manifold3d as m3d
        base = self.base.to_manifold()
        # consecutive copies' hulls: exact for a convex shape
        mans = [m3d.Manifold.batch_hull([base.translate(tuple(a * self.direction)),
                                         base.translate(tuple(b * self.direction))])
                for a, b in zip(self.offsets[:-1], self.offsets[1:])] if _convex(self.base) else \
            [base.translate(tuple(t * self.direction)) for t in self.offsets]
        return m3d.Manifold.batch_boolean(mans, m3d.OpType.Add) if len(mans) > 1 else mans[0]

    def to_spec(self):
        raise ValueError("a sweep is derived from a keep-out's insertFrom and has no JSON form of its own")


class Moved(Shape):
    """A shape moved by an offset (a leg of a multi-leg assembly path)."""

    kind = "moved"

    def __init__(self, base: Shape, offset):
        self.base = base
        self.offset = np.asarray(offset, float)

    def sdf(self, p):
        return self.base.sdf(np.asarray(p, float) - self.offset)

    def contains(self, p):
        return self.sdf(p) <= 0.0

    def bounds(self):
        lo, hi = self.base.bounds()
        return lo + self.offset, hi + self.offset

    def to_manifold(self):
        return self.base.to_manifold().translate(tuple(self.offset))

    def to_spec(self):
        raise ValueError("a moved shape is derived from a keep-out's insertFrom path and has no JSON form of its own")


def _convex(shape) -> bool:
    return isinstance(shape, (Box, Sphere)) or (isinstance(shape, Cylinder) and shape.r_inner == 0)


def sweep(shape: Shape, direction, length: float) -> Shape:
    """`shape` swept `length` mm along `direction` (see Sweep). Exact boxes
    and cylinders where the motion is along one of their own axes."""
    d = np.asarray(direction, float)
    d = d / np.linalg.norm(d)
    end = length * d
    k = int(np.argmax(np.abs(d)))
    if isinstance(shape, Box) and abs(d[k]) > 1 - 1e-12:
        return Box(np.minimum(shape.lo, shape.lo + end), np.maximum(shape.hi, shape.hi + end))
    if isinstance(shape, Cylinder) and abs(abs(float(shape.axis @ d)) - 1.0) < 1e-12:
        if float(shape.axis @ d) > 0:
            return Cylinder(shape.p0, shape.p1 + end, shape.r, shape.r_inner)
        return Cylinder(shape.p0 + end, shape.p1, shape.r, shape.r_inner)
    if isinstance(shape, Union):
        return Union([sweep(s, d, length) for s in shape.shapes])
    return Sweep(shape, d, length)


def shape_from_spec(spec, base_dir=None) -> Shape:
    """Build a Shape from its JSON dict. Raises ValueError with a message an
    agent can act on."""
    if isinstance(spec, Shape):
        return spec
    if not isinstance(spec, dict) or "type" not in spec:
        raise ValueError(f"shape must be an object with a 'type', got {spec!r}")
    try:
        return _shape(spec, base_dir)
    except KeyError as exc:
        raise ValueError(f"{spec['type']} shape is missing {exc.args[0]!r}; "
                         "see `loadpath schema study` for each shape's fields") from None


def _shape(spec, base_dir):
    kind = spec["type"]
    if kind == "box":
        if "min" in spec:
            return Box(spec["min"], spec["max"])
        if spec.get("axes") is not None:
            return OrientedBox(spec["center"], spec["size"], spec["axes"])
        c, s = _vec(spec["center"], name="box center"), _vec(spec["size"], name="box size")
        return Box(c - s / 2, c + s / 2)
    if kind in ("cylinder", "tube"):
        r_inner = spec.get("rInner", spec.get("r_inner", 0.0))
        if "p0" in spec:
            return Cylinder(spec["p0"], spec["p1"], spec["r"], r_inner)
        axis = spec.get("axis", "z")
        a = np.asarray(_AXES[axis] if isinstance(axis, str) else axis, float)
        a = a / np.linalg.norm(a)
        c = _vec(spec["center"], name="cylinder center")
        half = 0.5 * float(spec["length"]) * a
        return Cylinder(c - half, c + half, spec["r"], r_inner)
    if kind == "sphere":
        return Sphere(spec["center"], spec["r"])
    if kind == "heightfield":
        return HeightField(spec["normal"], spec["axes"], spec["origin"], spec["step"], spec["heights"],
                           spec.get("side", "below"), spec.get("offset", 0.0), spec.get("depth", 1000.0))
    if kind == "union":
        parts = spec.get("shapes")
        if not isinstance(parts, list) or not parts:
            raise ValueError("union needs a non-empty 'shapes' list")
        return Union([shape_from_spec(x, base_dir) for x in parts])
    if kind in ("intersect", "subtract"):
        parts = spec.get("shapes")
        if not isinstance(parts, list) or len(parts) < 2:
            raise ValueError(f"{kind} needs a 'shapes' list of at least two shapes")
        return Boolean(kind, [shape_from_spec(x, base_dir) for x in parts])
    if kind == "mesh":
        import base64
        import io
        import os
        import trimesh
        if "stlBase64" in spec:
            # Inline geometry: how a remote caller (a remote endpoint, an
            # agent over HTTP) hands over a mesh without touching our disk.
            data = base64.b64decode(spec["stlBase64"], validate=True)
            mesh = trimesh.load(io.BytesIO(data), file_type="stl", force="mesh")
            return MeshShape(mesh)
        path = spec.get("path")
        if not path:
            raise ValueError("mesh shape needs 'path' (STL/3MF/OBJ/STEP) or 'stlBase64'")
        if base_dir is not None:
            # Paths resolve inside the study's own folder and never escape it,
            # so a study file cannot read arbitrary files off the host.
            root = os.path.realpath(base_dir)
            full = os.path.realpath(os.path.join(root, path))
            if os.path.commonpath([root, full]) != root:
                raise ValueError(f"mesh path {path!r} must stay inside the study's folder")
            path = full
        if not os.path.isfile(path):
            where = os.path.realpath(base_dir) if base_dir is not None else os.getcwd()
            raise ValueError(f"mesh file {spec['path']!r} not found in {where}: relative paths resolve against "
                             "the study file's folder (CLI) or base_dir (MCP tools)")
        if str(path).lower().endswith((".step", ".stp")):
            # CAD straight in: tessellated, healed if it was an open shell
            from .cadimport import load_part
            return MeshShape(load_part(path).mesh, spec["path"])
        mesh = trimesh.load(path, force="mesh")
        return MeshShape(mesh, spec["path"])
    if kind == "sdf":
        import os
        path = spec.get("path")
        if not path:
            raise ValueError("sdf shape needs 'path' (an .npz from loadpath.exterior.save_sdf)")
        full = path
        if base_dir is not None:
            root = os.path.realpath(base_dir)
            full = os.path.realpath(os.path.join(root, path))
            if os.path.commonpath([root, full]) != root:
                raise ValueError(f"sdf path {path!r} must stay inside the study's folder")
        if not os.path.isfile(full):
            raise ValueError(f"sdf file {path!r} not found")
        return GridSDF.load(full, path)
    raise ValueError(f"unknown shape type {kind!r} (box, cylinder, tube, sphere, mesh, sdf, union, intersect, "
                     "subtract)")


def manifold_from_trimesh(mesh):
    """A manifold3d Manifold from a trimesh, in double precision where the
    library has it (Mesh64): rounding a lattice to float32 moves a vertex
    up to ~5e-6 mm at 100 mm, enough to push one sheet of a wall a few
    microns thick through the other."""
    import manifold3d as m3d
    if hasattr(m3d, "Mesh64"):
        # copies: the binding refuses read-only arrays (trimesh's caches)
        verts = np.array(mesh.vertices, dtype=np.float64, order="C", copy=True)
        faces = np.array(mesh.faces, dtype=np.uint64, order="C", copy=True).reshape(-1, 3)
        man = m3d.Manifold(m3d.Mesh64(vert_properties=verts, tri_verts=faces))
    else:
        verts = np.ascontiguousarray(mesh.vertices, dtype=np.float32)
        faces = np.ascontiguousarray(mesh.faces, dtype=np.uint32)
        man = m3d.Manifold(m3d.Mesh(vert_properties=verts, tri_verts=faces))
    if man.status() != m3d.Error.NoError:
        raise ValueError(f"mesh is not a valid manifold ({man.status()})")
    return man


def trimesh_from_manifold(man):
    import trimesh
    mesh = man.to_mesh64() if hasattr(man, "to_mesh64") else man.to_mesh()
    verts = np.asarray(mesh.vert_properties, float)[:, :3]
    faces = np.asarray(mesh.tri_verts, np.int64)
    return trimesh.Trimesh(verts, faces, process=False)
