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
    "Box", "Cylinder", "Sphere", "MeshShape", "Union", "shape_from_spec",
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
        c = 0.5 * (self.lo + self.hi)
        h = 0.5 * (self.hi - self.lo)
        q = np.abs(np.asarray(p, float) - c) - h
        outside = np.linalg.norm(np.maximum(q, 0.0), axis=-1)
        inside = np.minimum(q.max(axis=-1), 0.0)
        return outside + inside

    def bounds(self):
        return self.lo.copy(), self.hi.copy()

    def to_manifold(self):
        import manifold3d as m3d
        return m3d.Manifold.cube(tuple(self.hi - self.lo)).translate(tuple(self.lo))

    def to_spec(self):
        return {"type": "box", "min": self.lo.tolist(), "max": self.hi.tolist()}


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
        d = np.asarray(p, float) - self.p0
        t = d @ self.axis
        radial = np.linalg.norm(d - t[..., None] * self.axis, axis=-1)
        half = 0.5 * self.length
        # 2D (radial, axial) box SDF; a tube is the band rInner..r.
        if self.r_inner > 0:
            mid = 0.5 * (self.r + self.r_inner)
            rad_q = np.abs(radial - mid) - 0.5 * (self.r - self.r_inner)
        else:
            rad_q = radial - self.r
        ax_q = np.abs(t - half) - half
        q = np.stack([rad_q, ax_q], axis=-1)
        return np.linalg.norm(np.maximum(q, 0.0), axis=-1) + np.minimum(q.max(axis=-1), 0.0)

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
        return np.linalg.norm(np.asarray(p, float) - self.center, axis=-1) - self.r

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
        self._grid = (dist, origin, pitch)

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
        return np.min([s.sdf(p) for s in self.shapes], axis=0)

    def bounds(self):
        b = [s.bounds() for s in self.shapes]
        return np.min([x[0] for x in b], axis=0), np.max([x[1] for x in b], axis=0)

    def to_manifold(self):
        import manifold3d as m3d
        mans = [s.to_manifold() for s in self.shapes]
        return m3d.Manifold.batch_boolean(mans, m3d.OpType.Add) if len(mans) > 1 else mans[0]

    def to_spec(self):
        return {"type": "union", "shapes": [s.to_spec() for s in self.shapes]}


def shape_from_spec(spec, base_dir=None) -> Shape:
    """Build a Shape from its JSON dict. Raises ValueError with a message an
    agent can act on."""
    if isinstance(spec, Shape):
        return spec
    if not isinstance(spec, dict) or "type" not in spec:
        raise ValueError(f"shape must be an object with a 'type', got {spec!r}")
    kind = spec["type"]
    if kind == "box":
        if "min" in spec:
            return Box(spec["min"], spec["max"])
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
    if kind == "union":
        parts = spec.get("shapes")
        if not isinstance(parts, list) or not parts:
            raise ValueError("union needs a non-empty 'shapes' list")
        return Union([shape_from_spec(x, base_dir) for x in parts])
    if kind == "mesh":
        import base64
        import io
        import os
        import trimesh
        if "stlBase64" in spec:
            # Inline geometry: how a remote caller (the sidecar endpoint, an
            # agent over HTTP) hands over a mesh without touching our disk.
            data = base64.b64decode(spec["stlBase64"], validate=True)
            mesh = trimesh.load(io.BytesIO(data), file_type="stl", force="mesh")
            return MeshShape(mesh)
        path = spec.get("path")
        if not path:
            raise ValueError("mesh shape needs 'path' (STL/3MF/OBJ) or 'stlBase64'")
        if base_dir is not None:
            # Paths resolve inside the study's own folder and never escape it,
            # so a study file cannot read arbitrary files off the host.
            root = os.path.realpath(base_dir)
            full = os.path.realpath(os.path.join(root, path))
            if os.path.commonpath([root, full]) != root:
                raise ValueError(f"mesh path {path!r} must stay inside the study's folder")
            path = full
        mesh = trimesh.load(path, force="mesh")
        return MeshShape(mesh, spec["path"])
    raise ValueError(f"unknown shape type {kind!r} (box, cylinder, sphere, mesh)")


def manifold_from_trimesh(mesh):
    import manifold3d as m3d
    verts = np.ascontiguousarray(mesh.vertices, dtype=np.float32)
    faces = np.ascontiguousarray(mesh.faces, dtype=np.uint32)
    man = m3d.Manifold(m3d.Mesh(vert_properties=verts, tri_verts=faces))
    if man.status() != m3d.Error.NoError:
        raise ValueError(f"mesh is not a valid manifold ({man.status()})")
    return man


def trimesh_from_manifold(man):
    import trimesh
    mesh = man.to_mesh()
    verts = np.asarray(mesh.vert_properties, float)[:, :3]
    faces = np.asarray(mesh.tri_verts, np.int64)
    return trimesh.Trimesh(verts, faces, process=False)
