"""
Field-driven design, nTop style: a part is a graph of implicit fields that an
agent writes as JSON (or builds in Python), and every operation can be driven
by another field.

    {"nodes": {
       "body":   {"op": "mesh", "path": "arm-link.stl"},
       "stress": {"op": "result", "path": "arm-link.fields.npz", "quantity": "vonMises"},
       "wall":   {"op": "remap", "input": "stress", "from": [2, 30], "to": [0.8, 2.0]},
       "core":   {"op": "lattice", "type": "gyroid", "cell": 8, "thickness": "wall"},
       "skin":   {"op": "shell", "input": "body", "thickness": 1.2},
       "part":   {"op": "union", "inputs": ["skin", {"op": "intersect", "inputs": ["core", "body"]}],
                  "blend": 1.5}},
     "output": "part",
     "mesh": {"pitch": 0.3}}

    python -m loadpath field part.json --out out/

Two kinds of field, one representation. A SOLID is a signed distance in mm,
negative inside (shapes, meshes, lattices, booleans, shells). A SCALAR is
any number per point (a coordinate, a distance, a stress, a temperature, a
remapped ramp). Every parameter that takes a number (a wall thickness, a
fillet radius, a lattice cell, an offset) also takes a scalar field, which
is the whole idea: the wall follows the stress, the cell grows away from
the bolts, the fillet is larger where the load comes in.

A parameter is a number, the name of another node, or an inline node
object. Nodes are evaluated lazily at whatever points are asked for, each
once per batch of points, so a graph can share a subtree (the body used by
the skin, the clip and the distance ramp) without paying for it three
times.

Meshes come out smooth: the field is marching-cubed on a grid and each
vertex is then moved onto the field's own zero level along its gradient (a
Newton step), so a curved wall is curved, not faceted at the grid, and no
step of a voxel mask survives into the part. A mesh imported as a field
keeps its own surface to within the projection's tolerance: near the
surface its distance is exact (closest point on the closest triangle), and
only far from it does the voxel distance transform stand in.

The pieces this builds on: shapes.py's exact primitives, sdf_kit's smooth
booleans, lattice.py's TPMS level sets and homogenized density tables, and
voxels.py's scanline voxelizer. See `schema.FIELD_SCHEMA` for every node.
"""
from __future__ import annotations

import ast
import json
import math
import os
from dataclasses import dataclass

import numpy as np
from scipy import ndimage

__all__ = ["FieldGraph", "MeshField", "field_mesh", "save_result_field", "result_samples", "OPS"]

# the operations and what they take (the schema says the same, with docs)
OPS = ("constant", "coordinate", "distance", "shape", "mesh", "result", "grid", "image", "remap", "expr",
       "union", "intersect", "subtract", "shell", "offset", "lattice", "transform")

_CELL_BUDGET = 60_000_000
_CHUNK = 1_000_000


# ---- expressions -----------------------------------------------------------

def _smoothstep(e0, e1, x):
    u = np.clip((x - e0) / np.where(e1 == e0, 1e-12, e1 - e0), 0.0, 1.0)
    return u * u * (3.0 - 2.0 * u)


_FUNCS = {
    "abs": np.abs, "sqrt": lambda a: np.sqrt(np.maximum(a, 0.0)), "exp": np.exp,
    "log": lambda a: np.log(np.maximum(a, 1e-300)), "sin": np.sin, "cos": np.cos, "tan": np.tan,
    "floor": np.floor, "ceil": np.ceil, "sign": np.sign,
    "min": lambda *a: _reduce(np.minimum, a), "max": lambda *a: _reduce(np.maximum, a),
    "clamp": lambda x, lo, hi: np.clip(x, lo, hi),
    "lerp": lambda a, b, t: a + (b - a) * t,
    "smoothstep": _smoothstep,
    "step": lambda edge, x: (x >= edge).astype(float),
    "where": lambda c, a, b: np.where(c > 0, a, b),
}
_CONSTS = {"pi": math.pi, "e": math.e}


def _reduce(fn, args):
    out = args[0]
    for a in args[1:]:
        out = fn(out, a)
    return out


_BINOPS = {ast.Add: np.add, ast.Sub: np.subtract, ast.Mult: np.multiply, ast.Div: np.divide,
           ast.Pow: np.power, ast.Mod: np.mod}
_CMPOPS = {ast.Lt: np.less, ast.LtE: np.less_equal, ast.Gt: np.greater, ast.GtE: np.greater_equal}


def _compile_expr(text: str, lib=None):
    """(names it reads, fn(env) -> value) for an arithmetic expression over
    node names, x, y, z, pi, e and the functions in _FUNCS. A whitelist over
    Python's AST: no attributes, no subscripts, no calls but those. `lib`
    swaps the array functions (funcs, binops, cmpops, where, float) for
    another array library's (fieldtorch.py)."""
    lib = lib or {}
    funcs, binops, cmpops = lib.get("funcs", _FUNCS), lib.get("binops", _BINOPS), lib.get("cmpops", _CMPOPS)
    where = lib.get("where", lambda c, a, b: np.where(np.asarray(c) > 0, a, b))
    as_float = lib.get("float", lambda a: a.astype(float))
    try:
        tree = ast.parse(text, mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"expr {text!r} is not a valid expression: {exc.msg}") from None
    names = set()

    def build(n):
        if isinstance(n, ast.Expression):
            return build(n.body)
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)) and not isinstance(n.value, bool):
            v = float(n.value)
            return lambda env: v
        if isinstance(n, ast.Name):
            if n.id in _CONSTS:
                v = _CONSTS[n.id]
                return lambda env: v
            names.add(n.id)
            k = n.id
            return lambda env: env[k]
        if isinstance(n, ast.BinOp) and type(n.op) in binops:
            f, a, b = binops[type(n.op)], build(n.left), build(n.right)
            return lambda env: f(a(env), b(env))
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, (ast.USub, ast.UAdd)):
            a = build(n.operand)
            return (lambda env: -a(env)) if isinstance(n.op, ast.USub) else a
        if isinstance(n, ast.Compare) and len(n.ops) == 1 and type(n.ops[0]) in cmpops:
            f, a, b = cmpops[type(n.ops[0])], build(n.left), build(n.comparators[0])
            return lambda env: as_float(f(a(env), b(env)))
        if isinstance(n, ast.IfExp):
            c, a, b = build(n.test), build(n.body), build(n.orelse)
            return lambda env: where(c(env), a(env), b(env))
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in funcs and not n.keywords:
            f, args = funcs[n.func.id], [build(a) for a in n.args]
            return lambda env: f(*(a(env) for a in args))
        bad = ast.unparse(n) if hasattr(ast, "unparse") else type(n).__name__
        raise ValueError(f"expr {text!r}: {bad!r} is not allowed (arithmetic, comparisons, "
                         f"node names, x/y/z and {', '.join(sorted(_FUNCS))})")

    fn = build(tree)
    return names, fn


# ---- exact distance to a triangle mesh ---------------------------------------

def _closest_on_triangles(p, a, b, c):
    """Closest points on triangles (a, b, c) to points p, all (N, 3)
    (Ericson, Real-Time Collision Detection 5.1.5, vectorized)."""
    ab, ac, ap = b - a, c - a, p - a
    d1 = np.einsum("ij,ij->i", ab, ap)
    d2 = np.einsum("ij,ij->i", ac, ap)
    bp = p - b
    d3 = np.einsum("ij,ij->i", ab, bp)
    d4 = np.einsum("ij,ij->i", ac, bp)
    cp = p - c
    d5 = np.einsum("ij,ij->i", ab, cp)
    d6 = np.einsum("ij,ij->i", ac, cp)
    va = d3 * d6 - d5 * d4
    vb = d5 * d2 - d1 * d6
    vc = d1 * d4 - d3 * d2
    with np.errstate(divide="ignore", invalid="ignore"):
        denom = 1.0 / (va + vb + vc)
        v = vb * denom
        w = vc * denom
        out = a + ab * v[:, None] + ac * w[:, None]
        # edges
        t_ab = d1 / (d1 - d3)
        e_ab = (vc <= 0) & (d1 >= 0) & (d3 <= 0)
        out = np.where(e_ab[:, None], a + ab * t_ab[:, None], out)
        t_ac = d2 / (d2 - d6)
        e_ac = (vb <= 0) & (d2 >= 0) & (d6 <= 0)
        out = np.where(e_ac[:, None], a + ac * t_ac[:, None], out)
        t_bc = (d4 - d3) / ((d4 - d3) + (d5 - d6))
        e_bc = (va <= 0) & ((d4 - d3) >= 0) & ((d5 - d6) >= 0)
        out = np.where(e_bc[:, None], b + (c - b) * t_bc[:, None], out)
    # vertices
    out = np.where(((d1 <= 0) & (d2 <= 0))[:, None], a, out)
    out = np.where(((d3 >= 0) & (d4 <= d3))[:, None], b, out)
    out = np.where(((d6 >= 0) & (d5 <= d6))[:, None], c, out)
    return out


class MeshField:
    """Signed distance to a closed triangle mesh, exact near the surface.

    Far from the surface (beyond `band` mm) it is a voxel distance transform
    at `pitch`, interpolated: good to half a voxel, which is all a grading
    ramp needs. Within the band it is the distance to the closest point on
    the closest triangle (candidates from a k-d tree over the centroids of
    the mesh subdivided to `pitch`), signed by the angle-weighted normal of
    the triangles that share that point. That is what keeps an imported part
    smooth when it is re-meshed: the zero level is the mesh's own surface,
    not a voxel staircase."""

    def __init__(self, mesh, pitch=None, band=None, k=12):
        import trimesh
        from scipy.spatial import cKDTree

        from .voxels import mesh_solid

        if not isinstance(mesh, trimesh.Trimesh):
            raise ValueError("MeshField needs a trimesh.Trimesh")
        if not mesh.is_watertight:
            raise ValueError("a mesh field needs a closed (watertight) mesh to have an inside")
        ext = float(np.max(mesh.extents))
        self.pitch = float(pitch or max(ext / 200.0, 0.25))
        self.band = float(band if band is not None else 3.0 * self.pitch)
        self.k = int(k)
        self.mesh = mesh
        self.lo, self.hi = (np.asarray(b, float) for b in mesh.bounds)
        solid, origin, _ = mesh_solid(mesh, self.pitch, pad=4)
        dist = (ndimage.distance_transform_edt(~solid) - ndimage.distance_transform_edt(solid)) * self.pitch
        dist -= 0.5 * self.pitch * np.sign(dist)
        self._grid = dist.astype(np.float32)
        self._origin = origin
        v, f = np.asarray(mesh.vertices, float), np.asarray(mesh.faces, np.int64)
        edge = max(self.pitch, ext / 400.0)
        if np.linalg.norm(v[f[:, 0]] - v[f[:, 1]], axis=1).max() > edge:
            v, f = trimesh.remesh.subdivide_to_size(v, f, max_edge=edge, max_iter=12)
        self._tri = v[f]
        n = np.cross(self._tri[:, 1] - self._tri[:, 0], self._tri[:, 2] - self._tri[:, 0])
        self._normal = n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-300)
        self._tree = cKDTree(self._tri.mean(axis=1))

    def _coarse(self, P):
        idx = ((P - self._origin) / self.pitch).T
        n = np.asarray(self._grid.shape)[:, None] - 1
        cl = np.clip(idx, 0, n)
        out = ndimage.map_coordinates(self._grid, cl, order=1, mode="nearest").astype(float)
        # past the grid: the distance at its nearest point plus the way there (continuous at the grid's
        # face, never short of the true distance by more than the grid's error), so an offset of the mesh
        # by more than the grid's padding stays the mesh's shape, not its box
        return out + np.linalg.norm(idx - cl, axis=0) * self.pitch

    def _exact(self, P):
        k = min(self.k, len(self._tri))
        _, cand = self._tree.query(P, k=k)
        cand = cand.reshape(len(P), k)
        T = self._tri[cand]                              # (N, k, 3, 3)
        Pk = np.repeat(P[:, None, :], k, axis=1).reshape(-1, 3)
        C = _closest_on_triangles(Pk, T[:, :, 0].reshape(-1, 3), T[:, :, 1].reshape(-1, 3),
                                  T[:, :, 2].reshape(-1, 3)).reshape(len(P), k, 3)
        d2 = np.sum((C - P[:, None, :]) ** 2, axis=2)
        best = d2.min(axis=1)
        # sign: normals of every candidate whose closest point is (nearly) the
        # closest point, weighted by how squarely they face it
        tie = d2 <= best[:, None] * (1 + 1e-6) + (1e-9 * self.pitch) ** 2
        ib = np.argmin(d2, axis=1)
        cb = C[np.arange(len(P)), ib]
        nrm = (self._normal[cand] * tie[:, :, None]).sum(axis=1)
        s = np.einsum("ij,ij->i", P - cb, nrm)
        dist = np.sqrt(best)
        return np.where(s < 0, -dist, dist)

    def __call__(self, P):
        P = np.asarray(P, float).reshape(-1, 3)
        d = self._coarse(P)
        near = np.abs(d) < self.band
        if near.any():
            d[near] = self._exact(P[near])
        return d


# ---- result fields -------------------------------------------------------------

def result_samples(fields: dict, quantity: str = "vonMises", case=None):
    """(points, values) from verify_part(..., fields=True)["fields"]: von
    Mises (MPa, the worst case at each node unless `case` names one) or
    displacement magnitude (mm)."""
    cases = fields["cases"]
    if case is not None:
        cases = [c for c in cases if c["name"] == case]
        if not cases:
            raise ValueError(f"no load case {case!r} in the result (have {[c['name'] for c in fields['cases']]})")
    if quantity == "vonMises":
        vals = np.max([np.asarray(c["vonMises"], float) for c in cases], axis=0)
    elif quantity == "displacement":
        vals = np.max([np.linalg.norm(np.asarray(c["displacement"], float), axis=1) for c in cases], axis=0)
    else:
        raise ValueError("quantity must be 'vonMises' or 'displacement' for a verify result")
    return np.asarray(fields["nodes"], float), vals


def save_result_field(path, points, values, **extra):
    """Write scattered samples (N, 3) / (N,) as the .npz a `result` node
    reads. Extra arrays (other quantities) go in under their own names."""
    np.savez_compressed(path, points=np.asarray(points, np.float32), values=np.asarray(values, np.float32),
                        **{k: np.asarray(v) for k, v in extra.items()})


class _Samples:
    """A scalar field from samples: on a regular grid (origin, pitch,
    values[nx, ny, nz]) trilinear; from scattered points (verify's FE nodes,
    a CFD export) put onto the grid they lie on when they do (FE nodes
    always do), else inverse-distance weighted over the 8 nearest. Outside
    the samples the nearest value carries on, so a remap never sees NaN."""

    def __init__(self, points=None, values=None, grid=None, origin=None, pitch=None):
        if grid is not None:
            self.grid = np.asarray(grid, np.float32)
            self.origin = np.asarray(origin, float)
            self.pitch = np.broadcast_to(np.asarray(pitch, float), (3,)).copy()
            self.tree = None
            return
        pts = np.asarray(points, float).reshape(-1, 3)
        vals = np.asarray(values, float).ravel()
        if len(pts) != len(vals):
            raise ValueError(f"result field has {len(pts)} points but {len(vals)} values")
        g = self._regular(pts, vals)
        if g is not None:
            self.grid, self.origin, self.pitch = g
            self.tree = None
        else:
            from scipy.spatial import cKDTree

            self.tree = cKDTree(pts)
            self.vals = vals

    @staticmethod
    def _regular(pts, vals):
        lo = pts.min(axis=0)
        steps = []
        for i in range(3):
            u = np.unique(np.round(pts[:, i] - lo[i], 6))
            if len(u) < 2:
                return None
            du = np.diff(u)
            h = du.min()
            if h <= 0 or not np.allclose(du / h, np.round(du / h), atol=1e-3):
                return None
            steps.append(h)
        pitch = np.asarray(steps)
        ijk = np.round((pts - lo) / pitch).astype(np.int64)
        dims = ijk.max(axis=0) + 1
        if np.prod(dims) > 4 * _CELL_BUDGET:
            return None
        grid = np.full(tuple(dims), np.nan, np.float32)
        grid[ijk[:, 0], ijk[:, 1], ijk[:, 2]] = vals
        hole = np.isnan(grid)
        if hole.any():
            idx = ndimage.distance_transform_edt(hole, return_distances=False, return_indices=True)
            grid = grid[tuple(idx)]
        return grid, lo, pitch

    def __call__(self, P):
        P = np.asarray(P, float).reshape(-1, 3)
        if self.tree is None:
            idx = ((P - self.origin) / self.pitch).T
            return ndimage.map_coordinates(self.grid, idx, order=1, mode="nearest").astype(float)
        k = min(8, len(self.vals))
        d, j = self.tree.query(P, k=k)
        d, j = d.reshape(len(P), k), j.reshape(len(P), k)
        w = 1.0 / np.maximum(d, 1e-9) ** 2
        return (w * self.vals[j]).sum(axis=1) / w.sum(axis=1)


# ---- TPMS with field-driven cell and wall ---------------------------------------

_TPMS_KIND = {"gyroid": "gyroid", "diamond": "diamond", "schwarz-p": "schwarz-p", "schwarz_p": "schwarz-p",
              "schwarzP": "schwarz-p"}


def _frame_points(P, frame):
    """Lattice coordinates: cartesian (P itself), or cylindrical, which wraps
    the lattice round an axis so cells follow a tube or a ring (r, arc length
    at `radius`, height). The cell count round the circumference is rounded
    to a whole number so the lattice closes on itself."""
    if not frame or frame.get("type", "cartesian") == "cartesian":
        return P, None
    if frame["type"] != "cylindrical":
        raise ValueError("lattice frame type must be 'cartesian', 'cylindrical', 'spherical' or 'conformal'")
    c = np.asarray(frame.get("center", (0, 0, 0)), float)
    ax = frame.get("axis", "z")
    a = np.asarray({"x": (1, 0, 0), "y": (0, 1, 0), "z": (0, 0, 1)}[ax] if isinstance(ax, str) else ax, float)
    a = a / np.linalg.norm(a)
    e1 = np.cross(a, (1.0, 0, 0) if abs(a[0]) < 0.9 else (0, 1.0, 0))
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(a, e1)
    Q = P - c
    h = Q @ a
    u, v = Q @ e1, Q @ e2
    r = np.hypot(u, v)
    th = np.arctan2(v, u)
    return np.stack([r, th, h], axis=1), float(frame.get("radius", 0) or 0)


def _tpms_distance(kind, P, cell, frame=None):
    """Signed mm distance to the TPMS midsurface with period `cell` (a scalar
    or a per-point array). This is the `grading: "blend"` path for a varying
    cell: the distance fields of a ladder of fixed periods (at most 1.25x
    apart) cross-faded with a smoothstep. Exact inside a level; across a step
    the two lattices disagree and leave crumbs, which is why the default is
    the warp (_CellWarp)."""
    from .lattice import tpms

    kind = _TPMS_KIND.get(kind)
    if kind is None:
        raise ValueError(f"lattice type must be one of gyroid, diamond, schwarz-p")
    Q, radius = _frame_points(P, frame)
    cell = np.broadcast_to(np.asarray(cell, float), (len(P),))
    if np.any(cell <= 0):
        raise ValueError("lattice cell must be positive everywhere")

    def at(c, idx):
        q = Q[idx]
        if radius is None:
            return tpms(kind, q, c)
        # cylindrical: arc length at the frame radius, whole cells round
        R = radius or float(np.median(q[:, 0]))
        n = max(1, round(2 * np.pi * R / c))
        q = np.stack([q[:, 0], q[:, 1] * n * c / (2 * np.pi), q[:, 2]], axis=1)
        return tpms(kind, q, c)

    cmin, cmax = float(cell.min()), float(cell.max())
    if cmax - cmin <= 1e-9 * cmax:
        return at(cmin, slice(None))
    ratio = 1.25
    n = max(1, int(np.ceil(np.log(cmax / cmin) / np.log(ratio))))
    ratio = (cmax / cmin) ** (1.0 / n)
    s = np.clip(np.log(cell / cmin) / np.log(ratio), 0.0, n - 1e-9)
    lev = np.floor(s).astype(int)
    t = s - lev
    t = t * t * (3 - 2 * t)
    out = np.empty(len(P))
    for k in np.unique(lev):
        idx = np.nonzero(lev == k)[0]
        lo_c, hi_c = cmin * ratio ** k, cmin * ratio ** (k + 1)
        tk = t[idx]
        d = at(lo_c, idx)
        mix = tk > 1e-6
        if mix.any():
            d_hi = at(hi_c, idx[mix])
            d[mix] = (1 - tk[mix]) * d[mix] + tk[mix] * d_hi
        out[idx] = d
    return out


def _lsq_potential(T, h):
    """The grid scalar u whose differences along every grid edge best match
    the target gradient T (dims + (3,), per unit length) times the step h:
    least squares, D^T D u = D^T t. D^T D is the grid graph's Laplacian with
    free (Neumann) ends, which the DCT-II diagonalizes exactly, so the solve
    is direct, O(n log n). Defined up to a constant (its mean is zero)."""
    from scipy import fft

    dims = T.shape[:3]
    lam = sum((2 - 2 * np.cos(np.pi * np.arange(dims[a]) / dims[a])).reshape([-1 if i == a else 1
                                                                              for i in range(3)])
              for a in range(3))
    lam[0, 0, 0] = 1.0
    b = np.zeros(dims)
    for a in range(3):
        sl0 = [slice(None)] * 3
        sl1 = [slice(None)] * 3
        sl0[a] = slice(0, -1)
        sl1[a] = slice(1, None)
        te = 0.5 * (T[tuple(sl0) + (a,)] + T[tuple(sl1) + (a,)]) * h      # target step along each edge
        b[tuple(sl0)] -= te
        b[tuple(sl1)] += te                                                 # D^T t
    B = fft.dctn(b, type=2, norm="ortho") / lam
    B[0, 0, 0] = 0.0
    return fft.idctn(B, type=2, norm="ortho")


class _CellWarp:
    """Lattice coordinates u(x) for a cell size that varies in space: the
    least-squares map whose gradient is I / cell(x) (a Poisson solve on a
    grid over the lattice's domain, direct by DCT), so one period
    of u is one local cell. In 1D that is exactly u = integral of dx / cell;
    in 3D, where a cell field has curl the map cannot follow everywhere, the
    misfit spreads out smoothly instead of tearing. Every wall stays
    continuous across a grading, unlike cross-fading lattices of different
    periods, which leaves crumbs where they disagree."""

    def __init__(self, cell_fn, lo, hi, max_cells=400_000):
        lo, h, dims = self.grid(cell_fn, lo, hi, max_cells)
        axes = [lo[i] + np.arange(dims[i]) * h for i in range(3)]
        G = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)
        c = np.asarray(cell_fn(G), float)
        if np.any(c <= 0):
            raise ValueError("lattice cell must be positive everywhere")
        g = (1.0 / c).reshape(tuple(dims))
        n = int(np.prod(dims))
        self.u = []
        for comp in range(3):
            T = np.zeros(tuple(dims) + (3,))
            T[..., comp] = g
            u = _lsq_potential(T, h).ravel()
            # anchor: u = x / cell at the domain's centre point, so a constant
            # cell gives back the plain lattice
            self.u.append((u - u[n // 2]).reshape(tuple(dims)) + G[n // 2, comp] * g.ravel()[n // 2])
        self.lo, self.h, self.dims = lo, h, dims

    @staticmethod
    def grid(cell_fn, lo, hi, max_cells):
        """(lo, h, dims) of the grid the map is solved on."""
        lo = np.asarray(lo, float)
        hi = np.asarray(hi, float)
        ext = np.maximum(hi - lo, 1e-6)
        h = float((np.prod(ext) / max_cells) ** (1 / 3))
        # a few grid steps per cell at least, where the grid budget allows
        probe = np.stack(np.meshgrid(*[np.linspace(lo[i], hi[i], 12) for i in range(3)], indexing="ij"),
                         -1).reshape(-1, 3)
        h = max(h, 0.0) if h > 0 else 1.0
        h = min(max(h, float(np.min(cell_fn(probe))) / 6.0), max(h, 1e-3))
        lo = lo - 2 * h
        dims = np.ceil((hi + 2 * h - lo) / h).astype(int) + 1
        return lo, h, dims

    def anchor(self):
        """The point where u = x / cell: the grid's centre node."""
        n = int(np.prod(self.dims))
        ijk = np.asarray(np.unravel_index(n // 2, tuple(self.dims)))
        return self.lo + ijk * self.h

    def __call__(self, P, cell):
        idx = ((P - self.lo) / self.h).T
        n = (self.dims - 1)[:, None]
        cl = np.clip(idx, 0, n)
        out = np.stack([ndimage.map_coordinates(u, cl, order=1, mode="nearest") for u in self.u], axis=1)
        # past the grid: carry on at the local rate
        return out + (idx - cl).T * self.h / np.broadcast_to(np.asarray(cell, float), (len(P),))[:, None]


class _ConformalMap:
    """Lattice coordinates that follow a surface (nTop's conformal lattice):
    u3 counts cell layers in from the surface (its signed distance over the
    through-thickness cell, exact, so every layer is parallel to the skin
    and the first one sits on it), and (u1, u2) run along the surface. The
    tangent directions are a reference axis projected onto the tangent
    plane and its cross with the normal (where the normal turns towards the
    reference, a second axis takes over smoothly). Round a closed surface (a
    tube, a boss, the section of a link) the tangent field circulates and no
    coordinate follows it, so (u1, u2) are phases, defined modulo one cell,
    which is all a periodic cell reads: stripe patterns (_stripe_phase). Where
    a whole number of cells cannot fit round a loop the pattern puts a phase
    defect deep inside, not a shear in every cell at the skin.

    `surface_fn(P)` is the surface's signed distance, `cell_fn(P)` the
    in-plane cell (mm, a field grades it), `normal_cell` the cell through
    the thickness (mm; default the cell)."""

    def __init__(self, surface_fn, cell_fn, lo, hi, axis=None, normal_cell=None, max_cells=40_000):
        # a phase needs only a few grid steps a cell (_phase_at unwraps
        # between nodes); the solve is a sparse LU, so the grid stays small
        lo = np.asarray(lo, float)
        hi = np.asarray(hi, float)
        probe = np.stack(np.meshgrid(*[np.linspace(lo[i], hi[i], 12) for i in range(3)], indexing="ij"),
                         -1).reshape(-1, 3)
        h = max(float(np.min(cell_fn(probe))) / 3.5, float((np.prod(hi - lo + 4.0) / max_cells) ** (1 / 3)))
        ext = hi - lo
        lo = lo - 2 * h
        dims = np.ceil((hi + 2 * h - lo) / h).astype(int) + 1
        axes = [lo[i] + np.arange(dims[i]) * h for i in range(3)]
        G = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)
        c = np.asarray(cell_fn(G), float).reshape(tuple(dims))
        if np.any(c <= 0):
            raise ValueError("lattice cell must be positive everywhere")
        d = np.asarray(surface_fn(G), float).reshape(tuple(dims))
        n = np.stack(np.gradient(d, h), -1)
        n /= np.maximum(np.linalg.norm(n, axis=-1, keepdims=True), 1e-9)
        if axis is None:
            a = np.eye(3)[int(np.argmax(ext))]
            b = np.eye(3)[int(np.argsort(ext)[1])]
        else:
            a = np.asarray(_axis_vector(axis), float)
            b = np.eye(3)[int(np.argmin(np.abs(a)))]
        b = b - (b @ a) * a
        b /= np.linalg.norm(b)
        w = np.abs(n @ a)[..., None]
        sw = np.clip((w - 0.8) / 0.15, 0, 1)
        sw = sw * sw * (3 - 2 * sw)
        ref = (1 - sw) * a + sw * b
        t1 = ref - (ref * n).sum(-1, keepdims=True) * n
        t1 /= np.maximum(np.linalg.norm(t1, axis=-1, keepdims=True), 1e-9)
        t2 = np.cross(n, t1)
        # what matters is the cells at and just under the skin: weight
        # there, so a cell count that has to change with depth (fewer cells
        # round a smaller circle) changes deep inside, and the box outside
        # the solid barely counts
        depth = np.maximum(-d, 0.0) / c
        wgt = np.where(d > 0, np.exp(-(d / h) ** 2) + 1e-3, np.exp(-depth / 1.5))
        self.phase = [_stripe_phase(t / c[..., None], h, weight=wgt) for t in (t1, t2)]
        self.lo, self.h, self.dims = lo, h, dims
        self.normal_cell = normal_cell
        self.surface_fn = surface_fn

    def __call__(self, P, cell, surface=None):
        """(N, 3) coordinates, period 1 a cell (the two tangent ones are
        phases, modulo 1). `surface` is the surface's distance at P when the
        caller has it already."""
        cell = np.broadcast_to(np.asarray(cell, float), (len(P),))
        out = [_phase_at(phi, self.lo, self.h, P) for phi in self.phase]
        dist = np.asarray(self.surface_fn(P) if surface is None else surface, float)
        cn = cell if self.normal_cell is None else self.normal_cell
        return np.stack(out + [-dist / cn], axis=1)


def _stripe_phase(T, h, weight=None):
    """The phase (cycles, 0-1, on the grid) that advances by T . e h along
    every grid edge as nearly as it can: Knoeppel et al.'s stripe patterns,
    psi = exp(2 pi i phase) the lowest eigenvector of the connection
    Laplacian sum |psi_j - exp(2 pi i w_ij) psi_i|^2. Where the target
    cannot be followed (a whole number of cells has to fit round a loop)
    the pattern puts a phase defect, a line where |psi| -> 0, instead of
    shearing every cell. `weight` (dims) says where following the target
    matters (each edge counts the mean of its ends'): a cell count that has
    to change with depth is settled where the weight is low."""
    import scipy.sparse as sp
    from scipy.sparse.linalg import eigsh

    dims = T.shape[:3]
    n = int(np.prod(dims))
    ids = np.arange(n).reshape(dims)
    I, J, W, K = [], [], [], []
    wgt = np.ones(dims) if weight is None else weight
    for a in range(3):
        sl0 = [slice(None)] * 3
        sl1 = [slice(None)] * 3
        sl0[a] = slice(0, -1)
        sl1[a] = slice(1, None)
        I.append(ids[tuple(sl0)].ravel())
        J.append(ids[tuple(sl1)].ravel())
        W.append((0.5 * (T[tuple(sl0) + (a,)] + T[tuple(sl1) + (a,)]) * h).ravel())
        K.append((0.5 * (wgt[tuple(sl0)] + wgt[tuple(sl1)])).ravel())
    I, J, W, K = np.concatenate(I), np.concatenate(J), np.concatenate(W), np.concatenate(K)
    deg = np.bincount(I, K, minlength=n) + np.bincount(J, K, minlength=n)
    rot = K * np.exp(2j * np.pi * W)
    # psi_j ~ rot psi_i: L_jj += 1, L_ii += 1, L_ji -= rot, L_ij -= conj(rot)
    L = sp.csr_matrix((np.concatenate([-rot, -np.conj(rot)]), (np.concatenate([J, I]), np.concatenate([I, J]))),
                      shape=(n, n)) + sp.diags(deg.astype(complex))
    # shift-invert Lanczos on one sparse LU: the grid is a few steps a cell,
    # so the factorization takes seconds
    psi0 = np.exp(2j * np.pi * _lsq_potential(T, h).ravel())
    _, vec = eigsh(L.tocsc(), k=1, sigma=-1e-6 * float(deg.mean()), which="LM", v0=psi0, tol=1e-8)
    return (np.angle(vec[:, 0]) / (2 * np.pi) % 1.0).reshape(dims)


def _phase_at(phi, lo, h, P):
    """Trilinear interpolation of a phase grid (cycles, mod 1): the eight
    corners are unwrapped against the first, so a phase that advances by
    less than half a cycle a step interpolates exactly where it is linear."""
    dims = np.asarray(phi.shape)
    x = np.clip((P - lo) / h, 0, dims - 1 - 1e-9)
    i0 = np.floor(x).astype(int)
    f = x - i0
    i1 = np.minimum(i0 + 1, dims - 1)
    base = phi[i0[:, 0], i0[:, 1], i0[:, 2]]
    out = np.zeros(len(P))
    for cx in (0, 1):
        for cy in (0, 1):
            for cz in (0, 1):
                v = phi[(i1 if cx else i0)[:, 0], (i1 if cy else i0)[:, 1], (i1 if cz else i0)[:, 2]] - base
                v -= np.round(v)
                out += ((f[:, 0] if cx else 1 - f[:, 0]) * (f[:, 1] if cy else 1 - f[:, 1])
                        * (f[:, 2] if cz else 1 - f[:, 2])) * v
    return base + out


def _axis_vector(ax):
    if isinstance(ax, str):
        return {"x": (1, 0, 0), "y": (0, 1, 0), "z": (0, 0, 1)}[ax]
    a = np.asarray(ax, float)
    return a / np.linalg.norm(a)


def _frame_coords(P, frame, cell):
    """Lattice coordinates, period 1 a cell, of a cylindrical or spherical
    frame with a constant cell: whole cells round each closed direction."""
    c = float(np.asarray(cell, float).ravel()[0])
    if frame["type"] == "cylindrical":
        Q, radius = _frame_points(P, frame)
        R = radius or float(np.median(Q[:, 0]))
        n = max(1, round(2 * np.pi * R / c))
        return np.stack([Q[:, 0] / c, Q[:, 1] * n / (2 * np.pi), Q[:, 2] / c], axis=1)
    if frame["type"] == "spherical":
        Q = P - np.asarray(frame.get("center", (0, 0, 0)), float)
        r = np.linalg.norm(Q, axis=1)
        az = np.arctan2(Q[:, 1], Q[:, 0])
        pol = np.arccos(np.clip(Q[:, 2] / np.maximum(r, 1e-12), -1, 1))
        R = float(frame.get("radius", 0) or np.median(r))
        n_az = max(1, round(2 * np.pi * R / c))
        n_pol = max(1, round(np.pi * R / c))
        return np.stack([r / c, az * n_az / (2 * np.pi), pol * n_pol / np.pi], axis=1)
    raise ValueError(f"lattice frame type {frame['type']!r} has no constant-cell coordinates")


# ---- smooth booleans with a field radius -----------------------------------------

def _smin(a, b, k):
    """Polynomial smooth minimum with a per-point radius k (0 = hard)."""
    k = np.broadcast_to(np.asarray(k, float), np.shape(a))
    hard = k <= 1e-9
    kk = np.where(hard, 1.0, k)
    h = np.clip(0.5 + 0.5 * (b - a) / kk, 0.0, 1.0)
    s = b * (1 - h) + a * h - kk * h * (1 - h)
    return np.where(hard, np.minimum(a, b), s)


# ---- groups: reusable sub-graphs ---------------------------------------------------

# a material body's field keys (FieldGraph._material_bodies)
_BODY_KEYS = ("solid", "where", "above", "below", "mix")

_REF_KEYS = ("input", "value", "thickness", "distance", "blend", "cell", "density", "offset")


def _rename_spec(spec, names):
    """A copy of a node spec with every reference in `names` (old -> new)
    renamed: field parameters, inputs, vars, an analyzed part, expr text,
    and inline nodes inside them."""
    import re

    s = json.loads(json.dumps(spec))

    def ref(v):
        if isinstance(v, str):
            return names.get(v, v)
        if isinstance(v, dict) and "op" in v:
            return _rename_spec(v, names)
        return v

    for k in _REF_KEYS:
        if k in s and not (s.get("op") == "constant" and k == "value"):
            s[k] = ref(s[k])
    if isinstance(s.get("inputs"), list):
        s["inputs"] = [ref(v) for v in s["inputs"]]
    elif s.get("op") == "group" and isinstance(s.get("inputs"), dict):
        s["inputs"] = {k: ref(v) for k, v in s["inputs"].items()}
    if isinstance(s.get("vars"), dict):
        s["vars"] = {k: ref(v) for k, v in s["vars"].items()}
    if s.get("op") == "result" and isinstance(s.get("part"), str):
        s["part"] = names.get(s["part"], s["part"])
    if s.get("op") == "lattice" and isinstance(s.get("frame"), dict) and "surface" in s["frame"]:
        s["frame"]["surface"] = ref(s["frame"]["surface"])
    if s.get("op") == "expr" and isinstance(s.get("expr"), str):
        # names the expr reads directly (not through vars) follow too, as vars so any name works
        used = {m for m in re.findall(r"[A-Za-z_]\w*", s["expr"]) if m in names}
        for m in used:
            s.setdefault("vars", {})
            if m not in s["vars"]:
                s["vars"][m] = names[m]
    return s


def expand_groups(doc, _depth=0):
    """The graph with every `group` node replaced by a copy of its
    definition from `groups`: the group's nodes come in as
    <instance>__<node>, its params as constant nodes <instance>__<param>
    (the instance's own values over the definition's defaults), its inputs
    are the instance's nodes, and the group's output takes the instance's
    name, so the rest of the graph refers to it as before. Groups may use
    groups (eight deep)."""
    nodes = doc.get("nodes")
    if not isinstance(nodes, dict) or not any(isinstance(v, dict) and v.get("op") == "group" for v in nodes.values()):
        return doc
    if _depth > 8:
        raise ValueError("groups nest more than eight deep (does a group use itself?)")
    groups = doc.get("groups") or {}
    out = {}
    for name, spec in nodes.items():
        if not (isinstance(spec, dict) and spec.get("op") == "group"):
            out[name] = spec
            continue
        gname = spec.get("group")
        if gname not in groups:
            raise ValueError(f"node {name!r}: no group {gname!r} (have {', '.join(groups) or 'none'})")
        g = groups[gname]
        gnodes = g.get("nodes") or {}
        if not gnodes:
            raise ValueError(f"group {gname!r} has no nodes")
        gout = g.get("output") or list(gnodes)[-1]
        if gout not in gnodes:
            raise ValueError(f"group {gname!r}: output {gout!r} is not one of its nodes")
        params = {**(g.get("params") or {}), **(spec.get("params") or {})}
        unknown = set(spec.get("params") or {}) - set(g.get("params") or {})
        if unknown:
            raise ValueError(f"node {name!r}: group {gname!r} has no param {sorted(unknown)[0]!r} "
                             f"(has {', '.join(g.get('params') or {}) or 'none'})")
        given = spec.get("inputs") or {}
        missing = [i for i in g.get("inputs") or [] if i not in given]
        if missing:
            raise ValueError(f"node {name!r}: group {gname!r} needs input {missing[0]!r}")
        extra = set(given) - set(g.get("inputs") or [])
        if extra:
            raise ValueError(f"node {name!r}: group {gname!r} has no input {sorted(extra)[0]!r}")
        names = {n: (name if n == gout else f"{name}__{n}") for n in gnodes}
        names.update({p: f"{name}__{p}" for p in params})
        names.update({i: given[i] for i in g.get("inputs") or []})
        for p, v in params.items():
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                raise ValueError(f"node {name!r}: param {p!r} must be a number")
            out[f"{name}__{p}"] = {"op": "constant", "value": float(v)}
        for n, ns in gnodes.items():
            out[names[n]] = _rename_spec(ns, names)
    return expand_groups({**doc, "nodes": out}, _depth + 1)


# ---- the graph -----------------------------------------------------------------

@dataclass
class _Node:
    name: str
    op: str
    spec: dict


class FieldGraph:
    """A graph of named field nodes. `FieldGraph(doc)` from the JSON form
    (see the module docstring and schema.FIELD_SCHEMA); `sources` names
    in-memory data a `result` or `grid` node can refer to by `source`
    (verify_part fields, a temperature array), and `meshes` in-memory
    trimeshes a `mesh` node can name by `source`.

    graph.field()      -> callable(P) of the output node
    graph.evaluate(P, "wall")
    graph.bounds()     -> (lo, hi) of the output's solid
    graph.mesh()       -> smooth watertight trimesh (field_mesh)"""

    def __init__(self, doc, base_dir=None, sources=None, meshes=None):
        if isinstance(doc, str):
            doc = json.loads(doc)
        if not isinstance(doc, dict) or not isinstance(doc.get("nodes"), dict) or not doc["nodes"]:
            raise ValueError("a field graph needs a non-empty 'nodes' object")
        doc = expand_groups(doc)
        self.doc = doc
        self.base_dir = base_dir
        self.sources = dict(sources or {})
        self.meshes = dict(meshes or {})
        self.nodes: dict[str, _Node] = {}
        self._anon = 0
        for name, spec in doc["nodes"].items():
            if name in ("x", "y", "z") or name in _CONSTS or name in _FUNCS:
                raise ValueError(f"node name {name!r} is reserved (x, y, z, pi, e and the expr functions)")
            self._add(name, spec)
        self.output = doc.get("output") or list(doc["nodes"])[-1]
        if self.output not in self.nodes:
            raise ValueError(f"output {self.output!r} is not a node (have {', '.join(doc['nodes'])})")
        self.part_node = self.output     # the output as written (drains move `output` on)
        self._built = {}
        self.plan = None             # the sharp CSG plan of the last mesh(), when it had one
        self.analyses = {}          # verify reports behind `result` nodes that ran a study
        self.bodies = self._material_bodies(doc.get("materials"))
        self._check_refs()

    # -- construction --

    def _add(self, name, spec):
        if not isinstance(spec, dict) or "op" not in spec:
            raise ValueError(f"node {name!r} must be an object with an 'op' ({', '.join(OPS)})")
        if spec["op"] not in OPS:
            import difflib

            near = difflib.get_close_matches(str(spec["op"]), OPS, 1)
            raise ValueError(f"node {name!r}: unknown op {spec['op']!r}"
                             + (f"; did you mean {near[0]!r}?" if near else f" ({', '.join(OPS)})"))
        self.nodes[name] = _Node(name, spec["op"], spec)
        return name

    def _ref(self, v, owner, key):
        """A parameter: number, node name, or inline node -> node name or float."""
        if isinstance(v, bool):
            raise ValueError(f"node {owner!r}: {key} must be a number or a field, got {v!r}")
        if isinstance(v, (int, float)):
            return float(v)
        if isinstance(v, str):
            if v not in self.nodes:
                raise ValueError(f"node {owner!r}: {key} refers to {v!r}, which is not a node")
            return v
        if isinstance(v, dict):
            self._anon += 1
            return self._add(f"{owner}.{key}#{self._anon}", v)
        raise ValueError(f"node {owner!r}: {key} must be a number, a node name or a node object, got {v!r}")

    def _check_refs(self):
        # resolve every reference once (inline nodes get names) and refuse cycles
        self.deps = {}
        names = list(self.nodes)
        i = 0
        while i < len(names):
            nd = self.nodes[names[i]]
            self.deps[nd.name] = self._deps(nd)
            names += [n for n in self.deps[nd.name] if n not in names]
            i += 1
        state = {}

        def visit(n, path):
            if state.get(n) == 1:
                raise ValueError("field graph has a cycle: " + " -> ".join(path + [n]))
            if state.get(n) == 2:
                return
            state[n] = 1
            for d in self.deps[n]:
                visit(d, path + [n])
            state[n] = 2

        for n in list(self.nodes):
            visit(n, [])

    _FIELD_KEYS = ("input", "value", "thickness", "distance", "blend", "cell", "density", "offset")

    def _material_bodies(self, spec):
        """The graph's `materials`: [{name, material, solid, where, above,
        below, mix, extruder, color}], each field key resolved to a node name
        or a number (inline nodes get names materials.<body>.<key>#n), in
        precedence order (see material_bodies)."""
        if spec is None:
            return []
        if not isinstance(spec, list) or not spec:
            raise ValueError("materials must be a non-empty list of bodies")
        out, seen = [], set()
        for i, b in enumerate(spec):
            if not isinstance(b, dict):
                raise ValueError(f"materials[{i}] must be an object")
            name = b.get("name")
            if not isinstance(name, str) or not name.strip():
                raise ValueError(f"materials[{i}] needs a 'name'")
            if name in seen:
                raise ValueError(f"materials: two bodies are called {name!r}")
            seen.add(name)
            mat = b.get("material")
            if "mix" in b:
                if not (isinstance(mat, list) and len(mat) == 2):
                    raise ValueError(f"material body {name!r}: a mix body's material is a pair [A, B] "
                                     "(mix 0 is all A, 1 all B)")
            elif mat is None or isinstance(mat, list):
                raise ValueError(f"material body {name!r}: needs one 'material' (a pair only with 'mix')")
            if "where" in b and "above" not in b and "below" not in b:
                raise ValueError(f"material body {name!r}: 'where' needs 'above' and/or 'below'")
            if ("above" in b or "below" in b) and "where" not in b:
                raise ValueError(f"material body {name!r}: 'above'/'below' threshold the 'where' field; "
                                 "give one")
            owner = f"materials.{name}"
            body = {"name": name, "material": mat, "index": i}
            for key in _BODY_KEYS:
                if key in b:
                    body[key] = self._ref(b[key], owner, key)
            for key in ("extruder", "color", "voxel"):
                if key in b:
                    body[key] = b[key]
            out.append(body)
        return out

    def _deps(self, nd):
        s = nd.spec
        refs = {}
        for key in self._FIELD_KEYS:
            if key in s and not (nd.op == "constant" and key == "value"):
                refs[key] = self._ref(s[key], nd.name, key)
        if "inputs" in s:
            if not isinstance(s["inputs"], list) or not s["inputs"]:
                raise ValueError(f"node {nd.name!r}: inputs must be a non-empty list")
            refs["inputs"] = [self._ref(v, nd.name, f"inputs[{i}]") for i, v in enumerate(s["inputs"])]
        if nd.op == "lattice" and isinstance(s.get("frame"), dict) and s["frame"].get("type") == "conformal":
            if "surface" not in s["frame"]:
                raise ValueError(f"node {nd.name!r}: a conformal frame needs 'surface' (the solid it follows)")
            refs["surface"] = self._ref(s["frame"]["surface"], nd.name, "frame.surface")
            if isinstance(refs["surface"], float):
                raise ValueError(f"node {nd.name!r}: frame.surface must be a solid node")
        if nd.op == "expr":
            if not isinstance(s.get("expr"), str):
                raise ValueError(f"node {nd.name!r}: expr needs an 'expr' string")
            used, fn = _compile_expr(s["expr"])
            shadow = [k for k in (s.get("vars") or {}) if k in ("x", "y", "z") or k in _CONSTS or k in _FUNCS]
            if shadow:
                # the expr would read the constant or coordinate, never the var
                raise ValueError(f"node {nd.name!r}: var name {shadow[0]!r} is reserved (x, y, z, pi, e and the "
                                 f"expr functions); rename it")
            local = {k: self._ref(v, nd.name, f"vars.{k}") for k, v in (s.get("vars") or {}).items()}
            for u in used - set(local) - {"x", "y", "z"}:
                if u not in self.nodes:
                    raise ValueError(f"node {nd.name!r}: expr reads {u!r}, which is not a node, a var or x/y/z")
                local[u] = u
            refs["vars"] = local
            refs["fn"] = fn
        nd.refs = refs
        out = []
        if nd.op == "result" and s.get("part") in self.nodes:
            out.append(s["part"])        # analyzed from this graph: a dependency too
        for v in refs.values():
            for x in (v if isinstance(v, list) else v.values() if isinstance(v, dict) else [v]):
                if isinstance(x, str):
                    out.append(x)
        return out

    # -- evaluation --

    def field(self, name=None):
        name = name or self.output

        def f(P):
            return self.evaluate(P, name)
        if self._torch_field() is not None and self._tf.batch:
            f.batch = self._tf.batch
        return f

    def evaluate(self, P, name=None):
        tf = self._torch_field()
        if tf is not None:
            return tf(P, name)
        P = np.asarray(P, float).reshape(-1, 3)
        memo = {}
        return np.broadcast_to(self._eval(name or self.output, P, memo), (len(P),)).astype(float)

    def _torch_field(self):
        """fieldtorch.TorchField on the device LOADPATH_FIELD_DEVICE picks
        (a GPU by default when there is one), or None for numpy."""
        if not hasattr(self, "_tf"):
            from .fieldtorch import TorchField, field_device

            dev = field_device()
            self._tf = TorchField(self, dev) if dev is not None else None
        return self._tf

    def _val(self, r, P, memo):
        if isinstance(r, float):
            return r
        return self._eval(r, P, memo)

    def _eval(self, name, P, memo):
        key = (name, id(P))
        if key in memo:
            return memo[key][1]
        out = self._compute(self.nodes[name], P, memo)
        memo[key] = (P, out)          # keep P alive so its id stays unique
        return out

    def seed(self, prebuilt):
        """Use resources built by another graph of the same nodes (an
        analysis, a meshed field, a loaded grid): {name: (resource,
        analysis or None)}. The caller vouches that each node and what it
        depends on are unchanged; the editor keys them by the node's closure."""
        for name, (res, analysis) in (prebuilt or {}).items():
            if name in self.nodes:
                self._built[name] = res
                if analysis is not None:
                    self.analyses[name] = analysis
        return self

    def _resource(self, nd, make):
        if nd.name not in self._built:
            self._built[nd.name] = make()
        return self._built[nd.name]

    def _path(self, path):
        if self.base_dir is None:
            return path
        root = os.path.realpath(self.base_dir)
        full = os.path.realpath(os.path.join(root, path))
        if os.path.commonpath([root, full]) != root:
            raise ValueError(f"path {path!r} must stay inside the graph's folder")
        if not os.path.exists(full):
            raise ValueError(f"no file {path!r} in {root} (paths resolve from the graph's folder; "
                             "--base-dir picks another)")
        return full

    def _compute(self, nd, P, memo):
        s, r, op = nd.spec, nd.refs, nd.op
        v = lambda key, default=None: self._val(r[key], P, memo) if key in r else default  # noqa: E731
        if op == "constant":
            return float(s["value"])
        if op == "coordinate":
            ax = s.get("axis", "z")
            if isinstance(ax, str):
                return P[:, "xyz".index(ax)].copy()
            a = np.asarray(ax, float)
            a = a / np.linalg.norm(a)
            return (P - np.asarray(s.get("origin", (0, 0, 0)), float)) @ a
        if op == "distance":
            return self._distance(nd, P, memo)
        if op == "shape":
            from .shapes import shape_from_spec

            if s["shape"].get("type") == "mesh":
                raise ValueError(f"node {nd.name!r}: use op 'mesh' for meshes (exact surface distance)")
            shp = self._resource(nd, lambda: shape_from_spec(s["shape"], self.base_dir))
            return shp.sdf(P)
        if op == "mesh":
            mf = self._resource(nd, lambda: self._mesh_field(nd))
            return mf(P)
        if op in ("result", "grid", "image"):
            smp = self._resource(nd, lambda: self._samples(nd))
            return smp(P)
        if op == "remap":
            return _remap(v("input"), s, nd.name)
        if op == "expr":
            env = {"x": P[:, 0], "y": P[:, 1], "z": P[:, 2]}
            for k, ref in r["vars"].items():
                env[k] = self._val(ref, P, memo)
            return np.asarray(r["fn"](env), float)
        if op in ("union", "intersect", "subtract"):
            vals = [self._val(x, P, memo) for x in r["inputs"]]
            k = v("blend", 0.0)
            if op == "union":
                out = vals[0]
                for b in vals[1:]:
                    out = _smin(out, b, k)
                return out
            if op == "intersect":
                out = vals[0]
                for b in vals[1:]:
                    out = -_smin(-out, -b, k)
                return out
            out = vals[0]
            for b in vals[1:]:
                out = -_smin(-out, b, k)
            return out
        if op == "shell":
            f = v("input")
            t = np.asarray(v("thickness"), float)
            side = s.get("side", "inside")
            if side == "inside":
                return np.maximum(f, -(f + t))
            if side == "outside":
                return np.maximum(-f, f - t)
            if side == "center":
                return np.abs(f) - 0.5 * t
            raise ValueError(f"node {nd.name!r}: shell side must be inside, outside or center")
        if op == "offset":
            return v("input") - v("distance")
        if op == "lattice":
            return self._lattice(nd, P, memo)
        if op == "transform":
            return self._transform(nd, P, memo)
        raise AssertionError(op)

    def _distance(self, nd, P, memo):
        s = nd.spec
        if "input" in nd.refs:
            d = np.abs(self._val(nd.refs["input"], P, memo))
        elif "points" in s:
            pts = np.asarray(s["points"], float).reshape(-1, 3)
            from scipy.spatial import cKDTree

            d = cKDTree(pts).query(P)[0]
        elif "segment" in s:
            a, b = (np.asarray(x, float) for x in s["segment"])
            ab = b - a
            t = np.clip(((P - a) @ ab) / max(ab @ ab, 1e-300), 0.0, 1.0)
            d = np.linalg.norm(P - (a + t[:, None] * ab), axis=1)
        elif "plane" in s:
            n = np.asarray(s["plane"]["normal"], float)
            n = n / np.linalg.norm(n)
            d = np.abs((P - np.asarray(s["plane"]["point"], float)) @ n)
        elif "shape" in s:
            from .shapes import shape_from_spec

            shp = self._resource(nd, lambda: shape_from_spec(s["shape"], self.base_dir))
            d = np.abs(shp.sdf(P))
        else:
            raise ValueError(f"node {nd.name!r}: distance needs input, points, segment, plane or shape")
        if s.get("signed") and "input" in nd.refs:
            d = self._val(nd.refs["input"], P, memo)
        return d

    def _mesh_field(self, nd):
        import trimesh

        s = nd.spec
        if "source" in s:
            if s["source"] not in self.meshes:
                raise ValueError(f"node {nd.name!r}: no in-memory mesh named {s['source']!r}")
            mesh = self.meshes[s["source"]]
        else:
            if "path" not in s:
                raise ValueError(f"node {nd.name!r}: mesh needs a 'path' (STL/3MF/OBJ) or a 'source'")
            mesh = trimesh.load(self._path(s["path"]), force="mesh")
        return MeshField(mesh, pitch=s.get("pitch"), band=s.get("band"))

    _PLANES = {"xy": (0, 1, 2), "xz": (0, 2, 1), "yz": (1, 2, 0)}

    def _image(self, nd):
        """A picture as a scalar field: `path` (PNG, JPEG, ...) laid on
        `plane` (xy, xz or yz) with its lower-left corner at `origin` and
        `size` [w, h] mm, and carried straight through along the third axis.
        Values are the `channel` (luma, r, g, b or a) from 0 to 1, `invert`
        flips them; bilinear between pixels, the edge pixel carries on past
        the picture. Pictures over 1024 pixels a side are resized to it."""
        from PIL import Image

        s = nd.spec
        if "path" not in s or "size" not in s:
            raise ValueError(f"node {nd.name!r}: an image needs a 'path' and a 'size' [w, h] in mm")
        plane = s.get("plane", "xy")
        if plane not in self._PLANES:
            raise ValueError(f"node {nd.name!r}: plane must be xy, xz or yz")
        im = Image.open(self._path(s["path"]))
        im.load()
        im = im.convert("RGBA")
        if max(im.size) > 1024:
            k = 1024 / max(im.size)
            im = im.resize((max(2, round(im.size[0] * k)), max(2, round(im.size[1] * k))), Image.BOX)
        a = np.asarray(im, float) / 255.0                            # rows top to bottom
        ch = s.get("channel", "luma")
        if ch == "luma":
            v = a[..., 0] * 0.2126 + a[..., 1] * 0.7152 + a[..., 2] * 0.0722
        elif ch in ("r", "g", "b", "a"):
            v = a[..., "rgba".index(ch)]
        else:
            raise ValueError(f"node {nd.name!r}: channel must be luma, r, g, b or a")
        if s.get("invert"):
            v = 1.0 - v
        v = v[::-1].T                                                 # [u, v]: u right, v up
        H, W = v.shape[1], v.shape[0]
        w, h = (float(x) for x in s["size"])
        if w <= 0 or h <= 0:
            raise ValueError(f"node {nd.name!r}: size must be positive")
        u0, v0 = (float(x) for x in s.get("origin", (0.0, 0.0)))
        iu, iv, ia = self._PLANES[plane]
        # a regular grid in world axes, two samples deep along the projection axis (the same everywhere)
        grid = np.repeat(v[:, :, None], 2, axis=2)
        order = np.argsort([iu, iv, ia])
        grid = np.transpose(grid, order)
        origin, pitch = np.zeros(3), np.zeros(3)
        origin[iu], pitch[iu] = u0 + 0.5 * w / W, w / W
        origin[iv], pitch[iv] = v0 + 0.5 * h / H, h / H
        origin[ia], pitch[ia] = -5e5, 1e6
        return _Samples(grid=grid, origin=origin, pitch=pitch)

    def _samples(self, nd):
        s = nd.spec
        if nd.op == "image":
            return self._image(nd)
        if "source" in s:
            if s["source"] not in self.sources:
                raise ValueError(f"node {nd.name!r}: no in-memory source named {s['source']!r}")
            data = self.sources[s["source"]]
        elif "study" in s:
            data = self._analyze(nd)
        elif "path" in s:
            p = self._path(s["path"])
            data = dict(np.load(p)) if p.endswith(".npz") else json.load(open(p))
        else:
            raise ValueError(f"node {nd.name!r}: {nd.op} needs a 'path' (.npz or .json), a 'study' and "
                             "'part' to analyze, or a 'source'")
        if isinstance(data, dict) and "cases" in data and "nodes" in data:   # verify_part fields
            pts, vals = result_samples(data, s.get("quantity", "vonMises"), s.get("case"))
            return _Samples(pts, self._scaled(nd, vals))
        if isinstance(data, dict) and "grid" in data or nd.op == "grid":
            key = s.get("quantity", "values")
            if "origin" not in data or "pitch" not in data:
                raise ValueError(f"node {nd.name!r}: a grid needs origin, pitch and a 3D '{key}' array")
            return _Samples(grid=data.get("grid", data.get(key)), origin=data["origin"], pitch=data["pitch"])
        key = s.get("quantity", "values")
        if key not in data:
            raise ValueError(f"node {nd.name!r}: no {key!r} in the result (have {', '.join(data)})")
        return _Samples(data["points"], self._scaled(nd, np.asarray(data[key], float)))

    @staticmethod
    def _scaled(nd, vals):
        """`scale`: a number to divide by, or "peak" for the 99.5th
        percentile of the samples, so a remap can be written in fractions of
        the part's own worst stress without knowing it in advance (1.0 = the
        peak, ignoring the few hottest nodes at a load or support)."""
        sc = nd.spec.get("scale")
        if sc is None:
            return vals
        if sc == "peak":
            sc = float(np.percentile(vals, 99.5))
        if not float(sc) > 0:
            raise ValueError(f"node {nd.name!r}: scale must be positive or 'peak'")
        return vals / float(sc)

    def _analyze(self, nd):
        """verify_part fields of `part` under `study`: the stress a part
        actually carries, every load case. `study` is a JSON path or
        "example:<name>" (a packaged example); `part` a mesh path or the name
        of a node of this graph, meshed for the analysis (at `pitch`, default
        1 mm), so a graph can grade itself from its own solid's stress."""
        import trimesh

        from .verify import verify_part

        s = nd.spec
        if "part" not in s:
            raise ValueError(f"node {nd.name!r}: a result from a study needs the 'part' it analyzes")
        ref = s["study"] if s["study"].startswith("example:") else self._path(s["study"])
        study = load_study(ref, self.base_dir, self.doc.get("material"))
        if s["part"] in self.nodes:
            mesh = self.mesh(pitch=float(s.get("pitch", 1.0)), name=s["part"], project=1)
        else:
            mesh = trimesh.load(self._path(s["part"]), force="mesh")
        from . import quick

        rep = verify_part(study, mesh, max_cells=quick.result_cells(int(s.get("cells", 120_000))), fields=True)
        self.analyses[nd.name] = {k: v for k, v in rep.items() if k != "fields"}
        return rep["fields"]

    def _lattice(self, nd, P, memo):
        s, r = nd.spec, nd.refs
        kind = s.get("type", "gyroid")
        from .struts import canonical, is_strut

        frame = s.get("frame") or {}
        ftype = frame.get("type", "cartesian")
        if is_strut(kind) or ftype in ("conformal", "spherical") or "aspect" in s:
            return self._lattice_general(nd, P, memo, canonical(kind) if is_strut(kind) else kind)
        cell = self._val(r["cell"], P, memo) if "cell" in r else 8.0
        Q = P
        if "origin" in s:
            Q = P - np.asarray(s["origin"], float)
        if isinstance(r.get("cell"), str) and s.get("grading", "warp") == "warp":
            from .lattice import tpms

            if s.get("frame"):
                raise ValueError(f"node {nd.name!r}: a graded cell (warp) works in the cartesian frame only; "
                                 "use grading: 'blend' with a cylindrical frame")
            warp = self._resource(nd, lambda: self._warp(nd))
            d = tpms(_TPMS_KIND[kind], warp(Q, cell), 1.0) * np.asarray(cell, float)
        else:
            d = _tpms_distance(kind, Q, cell, s.get("frame"))
        if "offset" in r:
            d = d - self._val(r["offset"], P, memo)
        if "thickness" in r:
            t = np.asarray(self._val(r["thickness"], P, memo), float)
        elif "density" in r:
            from .lattice import wall_for_density

            rho = np.clip(np.asarray(self._val(r["density"], P, memo), float), 0.05, 0.95)
            rho, cl = np.broadcast_arrays(rho, np.asarray(cell, float))
            t = wall_for_density(_TPMS_KIND[kind], rho, cl)
        else:
            t = 1.0
        form = s.get("form", "sheet")
        if form == "sheet":
            return np.abs(d) - 0.5 * t
        if form == "network":
            return d - 0.5 * t
        raise ValueError(f"node {nd.name!r}: lattice form must be 'sheet' or 'network'")

    def lattice_coords(self, nd, P, memo=None):
        """(U, cell): a lattice node's coordinates at P (period 1 a cell)
        and its local cell (mm, per point), whatever its frame and grading.
        Voronoi foams have no coordinates (None)."""
        memo = {} if memo is None else memo
        s, r = nd.spec, nd.refs
        cell = self._val(r["cell"], P, memo) if "cell" in r else 8.0
        cell = np.broadcast_to(np.asarray(cell, float), (len(P),))
        Q = P - np.asarray(s["origin"], float) if "origin" in s else P
        frame = s.get("frame") or {}
        ftype = frame.get("type", "cartesian")
        aspect = np.asarray(s.get("aspect", (1, 1, 1)), float)
        if ftype == "conformal":
            cmap = self._resource(nd, lambda: self._conformal(nd))
            U = cmap(Q, cell, surface=self._val(r["surface"], P, memo))
        elif ftype in ("cylindrical", "spherical"):
            if isinstance(r.get("cell"), str):
                raise ValueError(f"node {nd.name!r}: a {ftype} frame takes a constant cell; grade the struts' "
                                 "thickness instead, or use a conformal frame")
            U = _frame_coords(Q, frame, cell[:1])
        elif isinstance(r.get("cell"), str):
            warp = self._resource(nd, lambda: self._warp(nd))
            U = warp(Q, cell)
        else:
            U = Q / cell[:, None]
        if "aspect" in s:
            U = U / aspect
        return U, cell

    def _lattice_general(self, nd, P, memo, kind):
        """Strut lattices (any frame), and TPMS in the conformal and
        spherical frames or with a stretched cell: distance in cell units
        through lattice_coords, scaled back to mm by the local cell."""
        from . import struts

        s, r = nd.spec, nd.refs
        strut = struts.is_strut(kind)
        if not strut and _TPMS_KIND.get(kind) is None:
            raise ValueError(f"node {nd.name!r}: lattice type must be a TPMS ({', '.join(sorted(set(_TPMS_KIND.values())))})"
                             f" or a strut cell ({', '.join(struts.STRUT_KINDS)})")
        form = s.get("form", "beam" if strut else "sheet")
        if strut and form != "beam":
            raise ValueError(f"node {nd.name!r}: a strut lattice's form is 'beam' (got {form!r})")
        blend = np.asarray(self._val(r["blend"], P, memo), float) if "blend" in r else 0.0
        if kind == "voronoi":
            if (s.get("frame") or {}).get("type", "cartesian") != "cartesian" or "aspect" in s:
                raise ValueError(f"node {nd.name!r}: a voronoi foam is cartesian (no frame or aspect)")
            vor = self._resource(nd, lambda: self._voronoi(nd))
            Q = P - np.asarray(s["origin"], float) if "origin" in s else P
            cell = np.broadcast_to(np.asarray(self._val(r["cell"], P, memo) if "cell" in r else 8.0, float),
                                   (len(P),))
            d_c = vor.distance(Q, blend)                   # mm
            scale = 1.0
        else:
            U, cell = self.lattice_coords(nd, P, memo)
            if strut:
                aspect = np.asarray(s.get("aspect", (1, 1, 1)), float) if "aspect" in s else None
                d_c = struts.unit_cell_distance(kind, U, np.asarray(blend, float) / cell, aspect=aspect)
            else:
                from .lattice import tpms

                d_c = tpms(_TPMS_KIND[kind], U, 1.0)
                if "aspect" in s:
                    d_c = d_c * float(np.min(s["aspect"]))
            scale = cell
        d = d_c * scale
        if "offset" in r:
            d = d - self._val(r["offset"], P, memo)
        bl = float(np.mean(blend)) if np.ndim(blend) else float(blend)
        if "thickness" in r:
            t = np.asarray(self._val(r["thickness"], P, memo), float)
        elif "density" in r:
            rho = np.clip(np.asarray(self._val(r["density"], P, memo), float), 0.02 if strut else 0.05, 0.95)
            rho, cl = np.broadcast_arrays(rho, np.asarray(cell, float))
            if strut:
                t = struts.strut_diameter_for_density(kind, rho, cl, blend=bl / float(np.mean(cl)))
            else:
                from .lattice import wall_for_density

                t = wall_for_density(_TPMS_KIND[kind], rho, cl)
        else:
            t = 1.0
        if strut:
            return d - 0.5 * t
        if form == "sheet":
            return np.abs(d) - 0.5 * t
        if form == "network":
            return d - 0.5 * t
        raise ValueError(f"node {nd.name!r}: lattice form must be 'sheet' or 'network'")

    def _conformal(self, nd, max_cells=40_000):
        """The surface-following map over the lattice's domain."""
        frame = nd.spec["frame"]
        cell_fn, lo, hi = self._warp_domain(nd)
        off = np.asarray(nd.spec.get("origin", (0, 0, 0)), float)
        surf = nd.refs["surface"]
        return _ConformalMap(lambda G: self.evaluate(G + off, surf), cell_fn, lo, hi, axis=frame.get("axis"),
                             normal_cell=frame.get("normalCell"), max_cells=max_cells)

    def _voronoi(self, nd):
        from .struts import VoronoiStruts

        cell_fn, lo, hi = self._warp_domain(nd)
        return VoronoiStruts(lo, hi, cell_fn, seed=int(nd.spec.get("seed", 0)))

    def _warp(self, nd, max_cells=400_000):
        """The cell-grading map over the lattice's domain: its own bounds,
        else the graph's, else the output solid's box."""
        cell_fn, lo, hi = self._warp_domain(nd)
        return _CellWarp(cell_fn, lo, hi, max_cells=max_cells)

    def _warp_domain(self, nd):
        """(cell field, lo, hi) the warp is solved over, in the lattice's own
        coordinates (less its origin)."""
        if "bounds" in nd.spec:
            lo, hi = nd.spec["bounds"]["min"], nd.spec["bounds"]["max"]
        elif "bounds" in self.doc:
            lo, hi = self.doc["bounds"]["min"], self.doc["bounds"]["max"]
        else:
            b = self.bounds()
            if b is None:
                raise ValueError(f"node {nd.name!r}: a graded, conformal or voronoi lattice needs a domain: give the lattice or the "
                                 "graph 'bounds'")
            lo, hi = b
        off = np.asarray(nd.spec.get("origin", (0, 0, 0)), float)
        cell_ref = nd.refs.get("cell", 8.0)
        if isinstance(cell_ref, float):
            return (lambda G, c=cell_ref: np.full(len(G), c)), np.asarray(lo, float) - off, np.asarray(hi, float) - off
        return lambda G: self.evaluate(G + off, cell_ref), np.asarray(lo, float) - off, np.asarray(hi, float) - off

    def _transform(self, nd, P, memo):
        s = nd.spec
        Q = P
        if "translate" in s:
            Q = Q - np.asarray(s["translate"], float)
        if "rotate" in s:
            rot = s["rotate"]
            ax = rot.get("axis", "z")
            a = np.asarray({"x": (1, 0, 0), "y": (0, 1, 0), "z": (0, 0, 1)}[ax] if isinstance(ax, str) else ax,
                           float)
            a = a / np.linalg.norm(a)
            c = np.asarray(rot.get("center", (0, 0, 0)), float)
            th = -math.radians(float(rot["degrees"]))
            K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
            R = np.eye(3) + math.sin(th) * K + (1 - math.cos(th)) * K @ K
            Q = (Q - c) @ R.T + c
        scale = float(s.get("scale", 1.0))
        if scale != 1.0:
            Q = Q / scale
        out = self._val(nd.refs["input"], Q, memo)
        return out * scale

    # -- bounds --

    def bounds(self, name=None):
        """(lo, hi) of a solid node, or None when it is unbounded (a bare
        lattice). Booleans combine their inputs' boxes; shells and offsets
        grow them by the largest thickness they could have (a field one is
        estimated from its values over the input's box)."""
        b = self._bounds(name or self.output, set())
        return None if b is None else (np.asarray(b[0], float), np.asarray(b[1], float))

    def _bounds(self, name, seen):
        if name in seen:
            return None
        seen = seen | {name}
        nd = self.nodes[name]
        s, r = nd.spec, nd.refs
        if "bounds" in s:
            return np.asarray(s["bounds"]["min"], float), np.asarray(s["bounds"]["max"], float)
        if nd.op == "shape":
            from .shapes import shape_from_spec

            return self._resource(nd, lambda: shape_from_spec(s["shape"], self.base_dir)).bounds()
        if nd.op == "mesh":
            mf = self._resource(nd, lambda: self._mesh_field(nd))
            return mf.lo.copy(), mf.hi.copy()
        sub = [self._bounds(x, seen) for x in r.get("inputs", [])]
        if nd.op == "union":
            if any(b is None for b in sub):
                return None
            lo, hi = np.min([b[0] for b in sub], axis=0), np.max([b[1] for b in sub], axis=0)
            # a smooth union bulges past both inputs by up to blend / 4
            g = 0.25 * max(self._max_over(r.get("blend"), (lo, hi)), 0.0)
            return lo - g, hi + g
        if nd.op == "intersect":
            sub = [b for b in sub if b is not None]
            if not sub:
                return None
            lo, hi = np.max([b[0] for b in sub], axis=0), np.min([b[1] for b in sub], axis=0)
            return lo, np.maximum(hi, lo)
        if nd.op == "subtract":
            return sub[0]
        if nd.op in ("shell", "offset") and isinstance(r.get("input"), str):
            b = self._bounds(r["input"], seen)
            if b is None:
                return None
            grow = self._max_over(r.get("thickness" if nd.op == "shell" else "distance"), b)
            if nd.op == "shell" and s.get("side", "inside") == "inside":
                grow = 0.0
            g = max(grow, 0.0) + 1e-6
            return b[0] - g, b[1] + g
        if nd.op == "transform" and isinstance(r.get("input"), str):
            b = self._bounds(r["input"], seen)
            if b is None:
                return None
            # a rotated box's box: transform its corners
            corners = np.array([[b[i][0], b[j][1], b[k][2]] for i in (0, 1) for j in (0, 1) for k in (0, 1)])
            sc = float(s.get("scale", 1.0))
            Q = corners * sc
            if "rotate" in s:
                rot = s["rotate"]
                ax = rot.get("axis", "z")
                a = np.asarray({"x": (1, 0, 0), "y": (0, 1, 0), "z": (0, 0, 1)}[ax] if isinstance(ax, str)
                               else ax, float)
                a = a / np.linalg.norm(a)
                c = np.asarray(rot.get("center", (0, 0, 0)), float)
                th = math.radians(float(rot["degrees"]))
                K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
                R = np.eye(3) + math.sin(th) * K + (1 - math.cos(th)) * K @ K
                Q = (Q - c) @ R.T + c
            Q = Q + np.asarray(s.get("translate", (0, 0, 0)), float)
            return Q.min(axis=0), Q.max(axis=0)
        return None

    def _max_over(self, ref, b, n=24):
        if ref is None:
            return 0.0
        if isinstance(ref, float):
            return ref
        axes = [np.linspace(b[0][i], b[1][i], n) for i in range(3)]
        G = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)
        return float(np.max(self.evaluate(G, ref))) * 1.1

    # -- meshing --

    def mesh(self, pitch=None, bounds=None, project=None, name=None, progress=None, sharp=None, creases=None):
        """`sharp` (default the graph's mesh.sharp, else true): hard booleans
        with plain shapes and meshes are done exactly (sharp.py), so bores,
        flat faces and keep-outs keep their edges; false meshes the whole
        field, every crease rounded over about one pitch.

        `creases` (default the graph's mesh.creases, else false): hard
        booleans between field surfaces (a lattice clipped by a body, two
        bodies unioned without a blend, a shell's rim) get their creases back
        on the meshed field (crease.py)."""
        m = self.doc.get("mesh") or {}
        pitch = pitch or m.get("pitch")
        project = m.get("project", 3) if project is None else project
        sharp = m.get("sharp", True) if sharp is None else sharp
        creases = m.get("creases", False) if creases is None else creases
        if bounds is None and "bounds" in self.doc:
            bounds = (self.doc["bounds"]["min"], self.doc["bounds"]["max"])
        if bounds is None:
            bounds = self.bounds(name)
            if bounds is None:
                raise ValueError(f"{name or self.output!r} is unbounded (a lattice with nothing to clip it): "
                                 "intersect it with a solid or give the graph 'bounds'")
        from . import quick
        pitch = quick.mesh_pitch(pitch, bounds)
        if sharp:
            from .sharp import has_exact, mesh_plan, plan

            p = plan(self, name)
            if has_exact(p):
                # set after meshing: a result node meshes another node on the way
                mesh = mesh_plan(p, bounds[0], bounds[1], pitch=pitch, project=project, progress=progress,
                                 decimate=m.get("decimate"), creases=self if creases else None)
                self.plan = p
                return mesh
        tree = None
        if creases:
            from .crease import CreaseTree

            tree = CreaseTree.from_graph(self, name)
        mesh = field_mesh(self.field(name), bounds[0], bounds[1], pitch=pitch, project=project, progress=progress,
                          decimate=m.get("decimate"), creases=tree)
        self.plan = None
        return mesh


def _run_streamed(doc, g, out_dir, drains, process, progress, t0):
    """run_field with mesh.stream: the STL goes to disk slab by slab
    (field_stl) and is never held, so the checks that need the whole mesh
    (printability, drains' mesh escape, bodies, the render) are skipped."""
    import time

    if not out_dir:
        raise ValueError("mesh.stream writes the STL as it goes: give an output folder")
    m = doc.get("mesh") or {}
    bounds = (doc["bounds"]["min"], doc["bounds"]["max"]) if "bounds" in doc else g.bounds()
    if bounds is None:
        raise ValueError(f"{g.output!r} is unbounded (a lattice with nothing to clip it): "
                         "intersect it with a solid or give the graph 'bounds'")
    os.makedirs(out_dir, exist_ok=True)
    name = doc.get("name", "field")
    path = os.path.join(out_dir, f"{name}.stl")
    st = field_stl(g.field(), bounds[0], bounds[1], path, pitch=m.get("pitch"), project=m.get("project", 3),
                   progress=progress, decimate=m.get("decimate"))
    rep = {"name": name, "output": g.output, "nodes": len(g.nodes), "streamed": True,
           **{k: st[k] for k in ("pitch", "faces", "watertight", "volume", "area", "slabs", "maxSlabFaces",
                                 "droppedShells", "sampling")}, "stl": path}
    if m.get("decimate"):
        rep["decimation"] = {"tolerance": float(m["decimate"]), "facesBefore": st["facesBeforeDecimation"]}
    if process is not None:
        rep["process"] = process.kind
    if drains is not None:
        rep["drains"] = drains
    if "material" in doc:
        from .materials import material_from_spec

        mat = material_from_spec(doc["material"])
        rep["mass"] = rep["volume"] * mat.density / 1000.0
        rep["material"] = mat.name
    rep["skipped"] = ["printability", "bodies", "render"]
    rep["seconds"] = round(time.time() - t0, 1)
    return rep


def _remap(x, s, name):
    """`from` [a, b] -> `to` [c, d] through `curve` (a name, or points
    [[u, v], ...] over the unit square drawn in the editor); clamped unless
    clamp: false. With more than two points, `from` (ascending) and `to`
    are a piecewise-linear table (always clamped at its ends), e.g. a
    density per layer from impact.grade_for_impact."""
    if len(s.get("from", ())) > 2 or len(s.get("to", ())) > 2:
        xs, ys = np.asarray(s["from"], float), np.asarray(s["to"], float)
        if len(xs) != len(ys) or np.any(np.diff(xs) <= 0):
            raise ValueError(f"node {name!r}: a remap table needs equal-length 'from' (ascending) and 'to'")
        return np.interp(np.asarray(x, float), xs, ys)
    a, b = (float(u) for u in s.get("from", (0.0, 1.0)))
    c, d = (float(u) for u in s.get("to", (0.0, 1.0)))
    if a == b:
        raise ValueError(f"node {name!r}: remap 'from' needs two different values")
    u = (np.asarray(x, float) - a) / (b - a)
    if s.get("clamp", True):
        u = np.clip(u, 0.0, 1.0)
    curve = s.get("curve", "linear")
    if isinstance(curve, (list, tuple)):
        # a drawn curve: points (u, v) over the unit square, u ascending from 0 to 1, straight between them
        pts = np.asarray(curve, float)
        if pts.ndim != 2 or pts.shape[1] != 2 or len(pts) < 2 or np.any(np.diff(pts[:, 0]) <= 0):
            raise ValueError(f"node {name!r}: a remap curve as points needs [[u, v], ...] with u ascending")
        return c + (d - c) * np.interp(np.clip(u, pts[0, 0], pts[-1, 0]), pts[:, 0], pts[:, 1])
    if curve == "smoothstep":
        uc = np.clip(u, 0.0, 1.0)
        u = uc * uc * (3 - 2 * uc)
    elif curve == "smootherstep":
        uc = np.clip(u, 0.0, 1.0)
        u = uc ** 3 * (uc * (6 * uc - 15) + 10)
    elif curve == "sqrt":
        u = np.sqrt(np.clip(u, 0.0, None))
    elif curve == "square":
        u = u * u
    elif curve != "linear":
        raise ValueError(f"node {name!r}: remap curve must be linear, smoothstep, smootherstep, sqrt or square")
    return c + (d - c) * u


# ---- meshing ---------------------------------------------------------------------

def field_mesh(field, lo, hi, pitch=None, project=3, pad=2, progress=None, sparse=True, decimate=None, creases=None):
    """Smooth, watertight trimesh of field(P) < 0 over [lo, hi] (mm).

    Marching cubes on a grid at `pitch` (default: the box over 4M cells,
    and at most the 60M budget), then `project` rounds of relaxing the
    vertices along the surface and Newton-projecting them onto the field's
    zero level (see _relax). Marching cubes alone puts vertices on the
    grid's edges, up to ~pitch/3 off a curved surface; after the
    projection every vertex is on the surface to 1e-5 mm or so, which is
    what removes the grid from the part.

    `sparse` (default) samples the field only in blocks near its zero
    level (narrowband.py): the same marching-cubes mesh as the dense grid,
    with far fewer field evaluations and memory for one batch of blocks,
    so the cell budget applies to the sampled cells, not the box.

    `decimate` (mm) thins the triangles wherever the mesh stays that close
    to the surface (decimate.py): flat and gently curved regions lose most
    of theirs, creases and small features keep them.

    `creases` (a crease.CreaseTree of the field: its hard min/max over
    smooth leaves) puts the creases between field surfaces back after the
    projection: the edges that cross from one leaf's surface to another's
    are split on the true intersection curve (crease.py)."""
    import trimesh
    from skimage.measure import marching_cubes

    lo = np.asarray(lo, float)
    hi = np.asarray(hi, float)
    ext = hi - lo
    if pitch is None:
        pitch = float((np.prod(ext) / 4e6) ** (1 / 3))
    pitch = float(pitch)
    lo = lo - pad * pitch - pitch * 0.013717        # off-grid, as sdf_kit.to_mesh
    dims = np.ceil((hi + pad * pitch - lo) / pitch).astype(int) + 1
    cells = int(np.prod(dims))
    if sparse:
        from .narrowband import narrowband_mc, watertight

        verts, faces, stats = narrowband_mc(field, lo, dims, pitch, budget=_CELL_BUDGET, progress=progress)
        if not len(faces):
            raise ValueError("the field has no inside over these bounds (nothing is below zero)")
        if not watertight(faces, len(verts)):
            # the Lipschitz bound let a crossing through: sample every block
            verts, faces, stats = narrowband_mc(field, lo, dims, pitch, lipschitz=np.inf, budget=_CELL_BUDGET,
                                                progress=progress)
            stats["fallback"] = True
        return _finish(verts, faces[:, ::-1], field, pitch, project, stats, decimate, creases)
    if cells > _CELL_BUDGET:
        p = float((np.prod(ext + 2 * pad * pitch) / _CELL_BUDGET) ** (1 / 3))
        raise ValueError(f"grid {dims[0]}x{dims[1]}x{dims[2]} = {cells / 1e6:.1f}M cells exceeds the "
                         f"{_CELL_BUDGET // 1_000_000}M budget: use pitch >= {math.ceil(p * 100) / 100:g}")
    F = np.empty(tuple(dims), np.float32)
    ys = lo[1] + np.arange(dims[1]) * pitch
    zs = lo[2] + np.arange(dims[2]) * pitch
    Y, Z = np.meshgrid(ys, zs, indexing="ij")
    per = max(1, _CHUNK // (dims[1] * dims[2]))
    for i0 in range(0, dims[0], per):
        i1 = min(dims[0], i0 + per)
        xs = lo[0] + np.arange(i0, i1) * pitch
        P = np.stack([np.repeat(xs, Y.size), np.tile(Y.ravel(), len(xs)), np.tile(Z.ravel(), len(xs))], axis=1)
        F[i0:i1] = np.asarray(field(P), np.float32).reshape(i1 - i0, dims[1], dims[2])
        if progress:
            progress({"stage": "field", "done": i1, "of": int(dims[0])})
    F[0], F[-1], F[:, 0], F[:, -1], F[:, :, 0], F[:, :, -1] = (pitch,) * 6
    F[F == 0.0] = -1e-4 * pitch
    if F.min() >= 0:
        raise ValueError("the field has no inside over these bounds (nothing is below zero)")
    verts, faces, _, _ = marching_cubes(F, 0.0, spacing=(pitch,) * 3)
    del F
    m = trimesh.Trimesh(verts + lo, faces[:, ::-1], process=True)
    return _finish(np.asarray(m.vertices), np.asarray(m.faces), field, pitch, project, None, decimate, creases)


def field_stl(field, lo, hi, path, pitch=None, project=3, pad=2, progress=None, block=32, lipschitz=None,
              decimate=None):
    """field(P) < 0 straight to a binary STL at `path`, one x slab of
    narrow-band blocks at a time, for parts whose mesh does not fit in
    memory (lattices at fine pitch: tens of millions of triangles). Memory
    is one slab (`block` cells thick), whatever the part; there is no cell
    budget. Returns the report: faces, volume, area, watertight, sampling.

    The same marching cubes, relaxation and projection as field_mesh, slab
    by slab. A vertex on the plane two slabs share is relaxed along the
    seam only and projected by the first slab, and reused, bit for bit, by the second, so the STL
    closes across the seam; the open edges each slab leaves on its two
    planes are matched against its neighbours', which checks the whole
    file is watertight without ever holding it. Small closed shells inside
    one slab, under (2 pitch)^3, are dropped as field_mesh drops debris. A
    crossing the Lipschitz skip missed shows as an unmatched edge, and the
    export starts over with every block sampled. `decimate` (mm) thins each
    slab as field_mesh does, its seam vertices left where they are."""
    lo, dims, pitch = _grid(lo, hi, pitch, pad)
    try:
        return _stream_stl(field, lo, dims, pitch, path, project, progress, block, lipschitz, decimate)
    except _Unmatched:
        rep = _stream_stl(field, lo, dims, pitch, path, project, progress, block, np.inf, decimate)
        rep["sampling"]["fallback"] = True
        return rep


class _Unmatched(Exception):
    pass


_STL_DTYPE = np.dtype([("n", "<f4", (3,)), ("v", "<f4", (3, 3)), ("a", "<u2")])


def _stream_stl(field, lo, dims, pitch, path, project, progress, block, lipschitz, decimate=None):
    import time

    from .narrowband import march_blocks, merge_blocks, shell_volumes

    t0 = time.time()
    stats = {}
    rep = {"faces": 0, "volume": 0.0, "area": 0.0, "slabs": 0, "maxSlabFaces": 0, "droppedShells": 0,
           "facesBeforeDecimation": 0}
    carry, carry_edges, carry_x = {}, set(), None
    tol = (2.0 * pitch) ** 3

    def keys(V):
        return [r.tobytes() for r in np.ascontiguousarray(V)]

    def slab(blocks, fh):
        nonlocal carry, carry_edges, carry_x
        x0 = int(blocks[0][0][0])
        x1 = x0 + int(blocks[0][1][0]) - 1
        if carry_x != x0 and carry_edges:
            raise _Unmatched()              # the last slab left edges open and nothing meets them
        nv, Vs, Fs = 0, [], []
        for _, _, v, fa in blocks:
            Vs.append(v)
            Fs.append(fa + nv)
            nv += len(v)
        V, F = merge_blocks(np.concatenate(Vs), np.concatenate(Fs), block)
        del Vs, Fs
        F = F[:, ::-1]
        start = V[:, 0] == x0
        end = V[:, 0] == x1
        # open edges: allowed only on the two planes, and matched across them
        n = len(V)
        a, b = F.ravel(), np.roll(F, -1, axis=1).ravel()
        e = np.minimum(a, b) * n + np.maximum(a, b)
        u, cnt = np.unique(e, return_counts=True)
        if (cnt > 2).any():
            raise _Unmatched()
        op = u[cnt == 1]
        ea, eb = op // n, op % n
        at_start = start[ea] & start[eb]
        at_end = end[ea] & end[eb]
        if not (at_start | at_end).all():
            raise _Unmatched()
        kv = {}

        def key(i):
            if i not in kv:
                kv[i] = V[i].tobytes()
            return kv[i]

        mine = {frozenset((key(i), key(j))) for i, j in zip(ea[at_start], eb[at_start])}
        if mine != (carry_edges if carry_x == x0 else set()):
            raise _Unmatched()
        carry_edges = {frozenset((key(i), key(j))) for i, j in zip(ea[at_end], eb[at_end])}
        # debris: a small shell that touches neither plane is whole and here
        lab, vol = shell_volumes(V, F)
        seam = np.zeros(len(vol), bool)
        seam[np.unique(lab[np.flatnonzero((start | end)[F].any(axis=1))])] = True
        drop = ~seam & (np.abs(vol) * pitch ** 3 <= tol)
        if drop.any():
            rep["droppedShells"] += int(drop.sum())
            F = F[~drop[lab]]
        Vmm = lo + V * pitch
        si = np.flatnonzero(start)
        if len(si):
            got = [carry.get(k) for k in keys(V[si])]
            if any(g is None for g in got):
                raise _Unmatched()
            Vmm[si] = np.array(got)
        end_keys = keys(V[end])
        if project:
            Vmm = _relax(Vmm, F, field, pitch, int(project), pinned=start, seam=end)
        carry = dict(zip(end_keys, Vmm[end]))
        if decimate:
            from .decimate import decimate as thin

            Vmm, F, dst, _ = thin(Vmm, F, field, float(decimate), pitch, locked=start | end)
            rep["facesBeforeDecimation"] += dst["facesBefore"]
        carry_x = x1
        # write
        tri = Vmm[F]
        nrm = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        ln = np.linalg.norm(nrm, axis=1)
        rep["area"] += float(ln.sum() / 2)
        rep["volume"] += float(np.einsum("ij,ij->i", tri[:, 0], np.cross(tri[:, 1], tri[:, 2])).sum() / 6)
        out = np.zeros(len(F), _STL_DTYPE)
        out["n"] = nrm / np.maximum(ln, 1e-300)[:, None]
        out["v"] = tri
        fh.write(out.tobytes())
        rep["faces"] += len(F)
        rep["slabs"] += 1
        rep["maxSlabFaces"] = max(rep["maxSlabFaces"], len(F))

    with open(path, "wb") as fh:
        fh.write(b"loadpath field_stl".ljust(80, b" ") + b"\0\0\0\0")
        cur = []
        for blk in march_blocks(field, lo, dims, pitch, block=block, lipschitz=lipschitz, progress=progress,
                                stats=stats):
            if cur and blk[0][0] != cur[0][0][0]:
                slab(cur, fh)
                cur = []
            cur.append(blk)
        if cur:
            slab(cur, fh)
        if carry_edges:
            raise _Unmatched()
        if not rep["faces"]:
            raise ValueError("the field has no inside over these bounds (nothing is below zero)")
        fh.seek(80)
        fh.write(np.uint32(rep["faces"]).tobytes())
    if rep["volume"] < 0:
        # wound inside out: flip every triangle in place
        mm = np.memmap(path, dtype=_STL_DTYPE, mode="r+", offset=84, shape=(rep["faces"],))
        for i in range(0, len(mm), 1_000_000):
            c = mm[i:i + 1_000_000]
            c["v"] = c["v"][:, ::-1]
            c["n"] = -c["n"]
        mm.flush()
        del mm
        rep["volume"] = -rep["volume"]
    rep.update({"watertight": True, "pitch": pitch, "sampling": stats, "path": path,
                "seconds": round(time.time() - t0, 1)})
    return rep


def _grid(lo, hi, pitch, pad):
    """field_mesh's sampling grid: (lo, dims, pitch), off-grid by a hair."""
    lo = np.asarray(lo, float)
    hi = np.asarray(hi, float)
    if pitch is None:
        pitch = float((np.prod(hi - lo) / 4e6) ** (1 / 3))
    pitch = float(pitch)
    lo = lo - pad * pitch - pitch * 0.013717        # off-grid, as sdf_kit.to_mesh
    dims = np.ceil((hi + pad * pitch - lo) / pitch).astype(int) + 1
    return lo, dims, pitch


def _finish(V, F, field, pitch, project, stats=None, decimate=None, creases=None):
    """Relax and project the marching-cubes mesh, orient it, drop debris
    (shells under (2 pitch)^3), as a trimesh. The bookkeeping is numpy on
    the arrays: trimesh's own (process, split, volume) took longer than
    the meshing on a lattice of a few million faces."""
    import trimesh

    from .narrowband import shell_volumes, signed_volume

    V = np.asarray(V, float)
    F = np.asarray(F, np.int64)
    if project:
        V = _relax(V, F, field, pitch, int(project))
    locked, sharp_st = None, None
    if creases is not None and project:
        from .crease import sharpen

        sharp_st = {}
        if signed_volume(V, F) < 0:          # sharpen checks faces against the surface normals
            F = F[:, ::-1].copy()
        V, F, locked = sharpen(V, F, creases, pitch, field=field, stats=sharp_st)
    dec = None
    if decimate:
        from .decimate import decimate as thin

        V, F, dec, kept = thin(V, F, field, float(decimate), pitch, locked=locked)
    try:
        lab, vol = shell_volumes(V, F)
        tol = (2.0 * pitch) ** 3
        if vol.sum() < 0:
            F, vol = F[:, ::-1].copy(), -vol
        small = np.abs(vol) <= tol
        if len(vol) > 1 and small.any() and not small.all():
            F = F[~small[lab]]
            used = np.zeros(len(V), bool)
            used[F] = True
            new = np.cumsum(used) - 1
            V, F = V[used], new[F]
    except Exception:  # noqa: BLE001 - debris filtering is best-effort
        pass
    if signed_volume(V, F) < 0:
        F = F[:, ::-1].copy()
    mesh = trimesh.Trimesh(V, F, process=False)
    mesh.metadata["pitch"] = pitch
    if stats:
        mesh.metadata["sampling"] = stats
    if dec:
        mesh.metadata["decimation"] = dec
    if sharp_st is not None:
        mesh.metadata["creases"] = sharp_st
    return mesh


def _gradient(field, V, h):
    """(f, grad f) at V from a tetrahedral stencil (five evaluations)."""
    from .narrowband import peval

    tet = np.array([[1, -1, -1], [-1, -1, 1], [-1, 1, -1], [1, 1, 1]], float) * h
    f0 = np.empty(len(V))
    g = np.empty((len(V), 3))
    chunk = 400_000
    for i in range(0, len(V), chunk):
        v = V[i:i + chunk]
        n = len(v)
        f = peval(field, np.concatenate([v] + [v + t for t in tet]))
        ft = f[n:].reshape(4, n)
        f0[i:i + chunk] = f[:n]
        g[i:i + chunk] = (tet[:, :, None] * ft[:, None, :]).sum(axis=0).T / (4 * h * h)
    return f0, g


def _project(field, V, pitch, f0=None, g=None):
    """One Newton step of every vertex onto field = 0, step capped at
    pitch / 2 and kept only where |f| drops."""
    from .narrowband import peval

    if f0 is None:
        f0, g = _gradient(field, V, 0.05 * pitch)
    gg = np.maximum((g * g).sum(axis=1), 1e-12)
    step = -(f0 / gg)[:, None] * g
    ln = np.linalg.norm(step, axis=1)
    step *= np.minimum(1.0, 0.5 * pitch / np.maximum(ln, 1e-300))[:, None]
    cand = V + step
    f1 = peval(field, cand)
    ok = np.abs(f1) < np.abs(f0)
    return np.where(ok[:, None], cand, V)


def _relax(V, F, field, pitch, rounds, pinned=None, seam=None):
    """Even out the marching-cubes vertices on the surface: each round moves
    every vertex halfway to its neighbours' centroid, within the tangent
    plane of the field (so the surface does not shrink), then projects it
    back onto the zero level. Marching cubes leaves a sawtooth along every
    crease (a vertex lands on one face or the other at random); relaxed and
    projected, a crease comes out as a clean edge rounded over about one
    pitch. Exact sharp features are mesh booleans' job (sdf_kit.mesh_subtract).

    Two closing Newton passes take every vertex the rest of the way; a
    vertex already within 1e-6 pitch of the zero level is left alone
    there, so those passes cost about one evaluation per vertex instead of
    six. `pinned` vertices do not move at all, and `seam` ones slide only
    along the seam they form (toward their seam neighbours' centroid): the
    streamed export's planes between slabs, which the next slab reuses
    as they are, so they must not lean on one side's vertices."""
    import scipy.sparse as sp

    from .narrowband import peval

    V = np.array(V, float)
    n = len(V)
    a = np.r_[F[:, 0], F[:, 1], F[:, 2]]
    b = np.r_[F[:, 1], F[:, 2], F[:, 0]]
    r, c = np.r_[a, b], np.r_[b, a]
    if seam is not None:
        seam = np.asarray(seam, bool)
        k = ~seam[r] | seam[c]
        r, c = r[k], c[k]
    A = sp.coo_matrix((np.ones(len(r), np.float32), (r, c)), shape=(n, n)).tocsr()
    A.data[:] = 1.0                       # an edge counts once, from either face
    deg = np.maximum(np.diff(A.indptr), 1)
    move = np.ones(n, bool) if pinned is None else ~np.asarray(pinned, bool)
    h = 0.05 * pitch
    for _ in range(rounds):
        idx = np.flatnonzero(move)
        f0, g = _gradient(field, V[idx], h)
        V[idx] = _project(field, V[idx], pitch, f0, g)
        nrm = np.zeros((n, 3))
        nrm[idx] = g / np.maximum(np.linalg.norm(g, axis=1, keepdims=True), 1e-12)
        d = (A @ V) / deg[:, None] - V
        d -= (d * nrm).sum(axis=1, keepdims=True) * nrm
        V[move] += 0.5 * d[move]
    tol = 1e-6 * pitch
    for _ in range(2):
        idx = np.flatnonzero(move)
        f0 = peval(field, V[idx])
        far = np.abs(f0) > tol
        idx = idx[far]
        if len(idx):
            V[idx] = _project(field, V[idx], pitch)
    return V


# ---- material bodies -------------------------------------------------------------

class _Cutter:
    """A material body's partition region (its solid, if it is not the
    part, and its where/above/below thresholds, intersected), carried on
    outside the part so its surface never runs along the part's own: inside
    the part within `eps` of its surface, and outside it, the region reads
    its value at the point `eps` deep along the part's normal (as sharp.py's
    extended field), and from 3 * eps outside it is solid. Only where the
    region's own surface crosses the part's interior does it draw one, and
    that is the interface. Clipped to `box` so it is closed."""

    def __init__(self, g, terms, part, eps, box):
        self.g, self.terms, self.part = g, terms, part
        self.eps = float(eps)
        self.lo, self.hi = (np.asarray(b, float) for b in box)

    def raw(self, P):
        vals = []
        for kind, a, b in self.terms:
            if kind == "solid":
                vals.append(self.g.evaluate(P, a))
            else:
                w = self.g.evaluate(P, a)
                t = b if isinstance(b, float) else self.g.evaluate(P, b)
                vals.append(t - w if kind == "above" else w - t)
        return np.max(vals, axis=0)

    def __call__(self, P):
        P = np.asarray(P, float).reshape(-1, 3)
        u = self.part(P)
        e, far = self.eps, 3 * self.eps
        # far outside: solid from `far` out, so the crease where the region's
        # surface meets that (rounded over a pitch by the mesher) stays clear of the part
        out = far - u
        near = u > -e
        mid = near & (u < far + e)
        inner = ~near
        if inner.any():
            out[inner] = self.raw(P[inner])
        if mid.any():
            Pn = P[mid]
            h = 1e-3
            grad = np.stack([(self.part(Pn + h * e) - self.part(Pn - h * e)) / (2 * h) for e in np.eye(3)], axis=1)
            nrm = grad / np.maximum(np.linalg.norm(grad, axis=1, keepdims=True), 1e-12)
            Q = Pn - (u[mid] + self.eps)[:, None] * nrm
            v = np.maximum(self.raw(Q), -e)
            out[mid] = np.minimum(v, far - u[mid])
        if inner.any():
            out[inner] = np.maximum(out[inner], -self.eps)
        box = np.max(np.concatenate([self.lo - P, P - self.hi], axis=1), axis=1)
        return np.maximum(out, box)


def _manifold(mesh):
    from .shapes import manifold_from_trimesh

    try:
        return manifold_from_trimesh(mesh)
    except ValueError:
        import trimesh

        m = mesh.copy()
        trimesh.repair.fix_winding(m)
        trimesh.repair.fill_holes(m)
        return manifold_from_trimesh(m)


def _body_terms(g, b):
    """[(kind, a, b)] that bound body `b` inside the part, or [] when it
    takes all that the bodies before it left."""
    terms = []
    sol = b.get("solid")
    if isinstance(sol, float):
        raise ValueError(f"material body {b['name']!r}: solid must be a solid node, not a number")
    if sol is not None and sol not in (g.output, g.part_node):
        terms.append(("solid", sol, None))
    for key in ("above", "below"):
        if key in b:
            terms.append((key, b["where"], b[key]))
    if terms and isinstance(terms[0][1], float):
        raise ValueError(f"material body {b['name']!r}: where must be a field, not a number")
    return terms


def material_bodies(g, part, progress=None):
    """Split the meshed part into the graph's material bodies (g.bodies),
    earlier bodies first: body i is what of the part is inside its region
    and not in an earlier body's, so bodies never overlap and together they
    are the part (less `rest`, what no body claimed).

    The part is meshed once (it keeps its sharp edges); each region is
    meshed from its field (_Cutter: carried on outside the part, so its
    surface is only the interface) and manifold3d splits what is left of
    the part by it: one exact boolean gives both sides, so the two bodies
    on an interface share that surface exactly, and each is watertight.
    Returns ([(body, trimesh or None)], rest manifold, stats)."""
    import manifold3d as m3d

    from .shapes import trimesh_from_manifold

    pitch = float(part.metadata.get("pitch") or (np.prod(part.extents) / 4e6) ** (1 / 3))
    lo, hi = np.asarray(part.bounds[0], float), np.asarray(part.bounds[1], float)
    box = (lo - 3 * pitch, hi + 3 * pitch)
    rest = _manifold(part)
    pf = g.field()
    out, stats = [], {"pitch": pitch, "regions": 0}
    for b in g.bodies:
        terms = _body_terms(g, b)
        if rest.is_empty():
            out.append((b, None))
            continue
        if not terms:
            piece, rest = rest, m3d.Manifold()
        else:
            cut = _Cutter(g, terms, pf, pitch, box)
            dims = np.ceil((box[1] - box[0]) / pitch + 5)
            # thresholds of an arbitrary field (a stress) have no Lipschitz
            # bound for the narrow band to lean on: sample densely while it fits
            sparse = float(np.prod(dims)) > 16e6
            cm = field_mesh(cut, box[0] - pitch, box[1] + pitch, pitch=pitch, project=3, sparse=sparse,
                            progress=progress)
            stats["regions"] += 1
            piece, rest = rest.split(_manifold(cm))
        out.append((b, None if piece.is_empty() else trimesh_from_manifold(piece)))
    return out, rest, stats


def _mix_job(g, b, mesh, mats, out_dir, stem, opts):
    """A `mix` body as a voxel jetting job: the fraction of material B on a
    voxel grid of the body (voxelprint.export_voxel_slices) and the mass it
    gives. Returns (mix report, mean density g/cm^3)."""
    from .voxels import mesh_solid

    pitch = float(opts.get("pitch") or mesh.metadata.get("pitch") or 0.25)
    occ, origin, _ = mesh_solid(mesh, pitch, pad=0)
    dims = occ.shape
    idx = np.argwhere(occ)
    frac = np.zeros(dims, np.float32)
    if len(idx):
        P = origin + idx * pitch
        mix = b["mix"]
        f = np.full(len(P), mix) if isinstance(mix, float) else g.evaluate(P, mix)
        frac[tuple(idx.T)] = np.clip(f, 0.0, 1.0)
    rho = np.array([m.density for m in mats])
    mean_f = float(frac[occ].mean()) if occ.any() else 0.0
    rep = {"materials": [m.name for m in mats], "voxelPitch": pitch, "voxels": int(occ.sum()),
           "meanFraction": mean_f, "fractionRange": [float(frac[occ].min()), float(frac[occ].max())]
           if occ.any() else [0.0, 0.0]}
    if out_dir:
        try:
            from .voxelprint import export_voxel_slices
        except ImportError as exc:
            rep["voxelJob"] = {"skipped": f"loadpath.voxelprint.export_voxel_slices is not available ({exc})"}
        else:
            try:
                job = export_voxel_slices(os.path.join(out_dir, f"{stem}-voxels"), frac, origin - pitch / 2, pitch,
                                          _jetting_names(mats, opts), printer=opts.get("printer", "j850"),
                                          solid=occ, max_layers=opts.get("maxLayers"))
                if isinstance(job, dict):
                    files = job.get("files") or []
                    job = {**{k: v for k, v in job.items() if k != "files"}, "fileCount": len(files)}
                rep["voxelJob"] = {"dir": os.path.join(out_dir, f"{stem}-voxels"),
                                   **(job if isinstance(job, dict) else {"result": str(job)})}
            except Exception as exc:  # noqa: BLE001 - the meshes are the deliverable
                rep["voxelJob"] = {"error": f"{type(exc).__name__}: {exc}"}
    return rep, float(rho[0] * (1 - mean_f) + rho[1] * mean_f)


# loadpath's jetting cards as GrabCAD Voxel Print resins; other materials get a colour each
_JETTING = {"vero": "VeroCyan", "agilus30": "Agilus30Blk", "resin": "VeroPureWht"}
_PALETTE = ("#005a9e", "#f0f0f0", "#a62162", "#c8bd03", "#284d6c", "#1a1a1d")


def _jetting_names(mats, opts):
    if opts.get("materials"):
        return list(opts["materials"])
    from .voxelprint import STRATASYS_COLORS

    out = []
    for i, m in enumerate(mats):
        name = _JETTING.get(m.name, m.name)
        out.append(name if name.lower() in STRATASYS_COLORS else (name, _PALETTE[i % len(_PALETTE)]))
    return out


def _safe(name):
    import re

    return re.sub(r"[^A-Za-z0-9_.-]+", "-", name).strip("-") or "body"


def _material_report(g, doc, part, out_dir, stem, progress):
    """run_field's `materials`: each body's mesh, volume and mass, the
    interface area between each pair, the checks that they tile the part,
    one STL per body and a multi-material 3MF."""
    from .materials import material_from_spec
    from .narrowband import signed_volume, watertight

    pieces, rest, stats = material_bodies(g, part, progress=progress)
    V0 = signed_volume(np.asarray(part.vertices), np.asarray(part.faces))
    bodies, mans, extruders = [], [], {}
    for b, mesh in pieces:
        mix = "mix" in b
        mats = [material_from_spec(m) for m in (b["material"] if mix else [b["material"]])]
        key = "+".join(m.name for m in mats)
        ext = b.get("extruder") or extruders.setdefault(key, len(extruders) + 1)
        r = {"name": b["name"], "material": key if not mix else [m.name for m in mats], "extruder": ext}
        if mesh is None:
            r.update(volume=0.0, mass=0.0, empty=True)
            bodies.append((b, r, None, mats))
            mans.append(None)
            continue
        Vb, Fb = np.asarray(mesh.vertices), np.asarray(mesh.faces)
        r.update(volume=signed_volume(Vb, Fb), area=float(mesh.area), faces=int(len(Fb)),
                 watertight=watertight(Fb, len(Vb)))
        r["share"] = r["volume"] / V0 if V0 > 0 else 0.0
        rho = mats[0].density
        if mix:
            r["mix"], rho = _mix_job(g, b, mesh, mats, out_dir, f"{stem}-{_safe(b['name'])}", b.get("voxel") or {})
        r["density"] = rho
        r["mass"] = r["volume"] * rho / 1000.0
        bodies.append((b, r, mesh, mats))
        mans.append(_manifold(mesh))
    inter, overlap = [], 0.0
    for i in range(len(mans)):
        for j in range(i + 1, len(mans)):
            if mans[i] is None or mans[j] is None:
                continue
            both = mans[i] + mans[j]
            area = 0.5 * (mans[i].surface_area() + mans[j].surface_area() - both.surface_area())
            ov = (mans[i] ^ mans[j]).volume()
            overlap = max(overlap, ov)
            if area > 1e-6:
                inter.append({"bodies": [bodies[i][1]["name"], bodies[j][1]["name"]], "area": float(area)})
    total = sum(r["volume"] for _, r, _, _ in bodies)
    left = 0.0 if rest.is_empty() else float(rest.volume())
    rep = {"bodies": [r for _, r, _, _ in bodies], "interfaces": inter,
           "check": {"partVolume": V0, "bodyVolume": total, "unassignedVolume": left,
                     "volumeError": (total + left - V0) / V0 if V0 else 0.0, "maxOverlap": float(overlap),
                     "watertight": all(r.get("watertight", True) for _, r, _, _ in bodies)},
           "regionsMeshed": stats["regions"], "mass": sum(r["mass"] for _, r, _, _ in bodies)}
    warn = []
    if left > 1e-4 * max(V0, 1e-9):
        warn.append(f"{left:.1f} mm^3 of the part is in no material body: end with a body that has no "
                    "solid or where, to take the rest")
    warn += [f"material body {r['name']!r} is empty (earlier bodies took all of its region)"
             for _, r, _, _ in bodies if r.get("empty")]
    if warn:
        rep["warnings"] = warn
    if out_dir:
        for _, r, mesh, _ in bodies:
            if mesh is not None:
                path = os.path.join(out_dir, f"{stem}-{_safe(r['name'])}.stl")
                mesh.export(path)
                r["stl"] = path
        try:
            from .threemf import write_multimaterial_3mf
        except ImportError as exc:
            rep["3mf"] = {"skipped": f"loadpath.threemf.write_multimaterial_3mf is not available ({exc})"}
        else:
            spec = []
            for b, r, mesh, mats in bodies:
                if mesh is None:
                    continue
                d = {"name": r["name"], "mesh": mesh, "material": mats[0].name, "extruder": r["extruder"]}
                if "color" in b:
                    d["color"] = b["color"]
                spec.append(d)
            path = os.path.join(out_dir, f"{stem}.3mf")
            try:
                res = write_multimaterial_3mf(path, spec, direction=doc.get("build"), name=stem,
                                              process=doc.get("process"))
                rep["3mf"] = path
                if isinstance(res, dict):
                    rep["3mfReport"] = res
            except Exception as exc:  # noqa: BLE001 - the STLs are written; say why the 3MF is not
                rep["3mf"] = {"error": f"{type(exc).__name__}: {exc}"}
    return rep


# ---- the job ---------------------------------------------------------------------

def load_study(ref, base_dir=None, material=None):
    """The study a graph analyzes its part under: a JSON path (relative to
    base_dir) or "example:<name>". The graph's own `material`, when it has
    one, replaces the study's, so the part is checked in what it will be
    printed in (and its process takes that card's across-layer values)."""
    from .spec import study_from_dict

    if ref.startswith("example:"):
        from .tasks import example

        d, sdir = example(ref.split(":", 1)[1]), base_dir
    else:
        sp = ref if os.path.isabs(ref) else os.path.join(base_dir or ".", ref)
        with open(sp) as fh:
            d = json.load(fh)
        sdir = os.path.dirname(sp)
    if material is not None:
        d = {**d, "material": material}
    return study_from_dict(d, base_dir=sdir)


def node_doc(doc, node):
    """The graph made to mesh one of its nodes instead of its output: groups
    expanded (so a node inside a copy is named <copy>__<node>), `output`
    set to `node` and the name suffixed with it, so its files don't
    overwrite the whole part's."""
    full = expand_groups(doc)
    if node not in (full.get("nodes") or {}):
        raise ValueError(f"no node {node!r} to mesh (have {', '.join(full.get('nodes') or {})})")
    return {**full, "output": node, "name": f"{doc.get('name', 'field')}-{node}"}


def _write_graded(doc, rep, out_dir):
    """With the impact block's `grade`: the graph again with its density node
    replaced by the graded table (<name>-graded.json beside the STL), ready
    to mesh and re-check with `loadpath field`. File paths in it resolve
    against the graph's folder: pass --base-dir if it reads any."""
    imp, spec = rep.get("impact") or {}, doc.get("impact") or {}
    dens = spec.get("density")
    if not imp.get("graded") or not isinstance(dens, str) or dens not in (doc.get("nodes") or {}):
        return
    gd = json.loads(json.dumps(doc))
    gd["name"] = f"{rep['name']}-graded"
    gd["nodes"][dens] = imp["graded"]["density"]
    gd["impact"] = {**spec, "grade": False}
    path = os.path.join(out_dir, f"{gd['name']}.json")
    with open(path, "w") as fh:
        json.dump(gd, fh, indent=2)
    imp["gradedGraph"] = path


def run_field(doc, out_dir=None, base_dir=None, sources=None, meshes=None, progress=None, render=True,
              prebuilt=None, node=None):
    """Build a field graph, mesh its output and report on it: volume,
    area, mass (with `material`), watertightness, bodies, and the printability
    check on the field itself (walls, overhangs, trapped voids) unless
    `check` is false. Writes <name>.stl (and <name>.step with `step`) to
    `out_dir`; with sharp booleans, `sharpEdges` says how exactly the mesh
    keeps the plain shapes' surfaces (sharp.edge_accuracy).
    `prebuilt` reuses resources another graph built (FieldGraph.seed);
    `node` meshes that node instead of the output (node_doc)."""
    import time

    t0 = time.time()
    if node is not None:
        doc = node_doc(doc, node)
    g = FieldGraph(doc, base_dir=base_dir, sources=sources, meshes=meshes).seed(prebuilt)
    process = None
    if doc.get("process") is not None:
        from .spec import _process

        process = _process(doc["process"])
    drains = None
    from .drains import DrainOptions, drain_graph, drains_wanted

    if drains_wanted(doc.get("drains", "auto"), process):
        opts = DrainOptions.from_spec(doc.get("drains"))
        from . import quick

        pitch = (doc.get("mesh") or {}).get("pitch")
        drains = drain_graph(g, process, opts, build=doc.get("build"), min_wall=doc.get("minWall"),
                             pitch=pitch and quick.mesh_pitch(pitch, None), progress=progress)
    if (doc.get("mesh") or {}).get("stream"):
        return None, _run_streamed(doc, g, out_dir, drains, process, progress, t0)
    mesh = g.mesh(progress=progress)
    from .narrowband import shell_volumes, signed_volume, watertight

    V, F = np.asarray(mesh.vertices), np.asarray(mesh.faces)
    rep = {"name": doc.get("name", "field"), "output": g.output, "nodes": len(g.nodes),
           "pitch": mesh.metadata.get("pitch"), "faces": int(len(mesh.faces)), "watertight": watertight(F, len(V)),
           "volume": signed_volume(V, F), "area": float(mesh.area),
           "bounds": [mesh.bounds[0].tolist(), mesh.bounds[1].tolist()]}
    if "decimation" in mesh.metadata:
        rep["decimation"] = mesh.metadata["decimation"]
    # shells: a body's outside, or the wall of a void sealed inside it (normals in)
    shells = shell_volumes(V, F)[1].tolist()
    rep["bodies"] = sum(v > 0 for v in shells)
    rep["enclosedVoids"] = sum(v < 0 for v in shells)
    rep["enclosedVoidVolume"] = -sum(v for v in shells if v < 0)
    if process is not None:
        rep["process"] = process.kind
    if drains is not None:
        # the mesh is the last word on whether anything is still sealed, and
        # for powder and resin processcheck's escape ball says whether it
        # gets out (LPBF has no card there: sealed shells only)
        ok = rep["enclosedVoids"] == 0
        if drains["holes"] or drains["filledVolume"]:
            from .processcheck import RULES, check_process

            if process is not None and process.kind in RULES:
                try:
                    esc = check_process(mesh, process, slicer=False)["escape"]
                    drains["meshEscape"] = {k: esc[k] for k in ("enclosedCount", "enclosedVolumeMm3",
                                                                "narrowExitCount", "narrowExitVolumeMm3")}
                    ok = ok and esc["enclosedCount"] == 0 and esc["narrowExitCount"] == 0
                except Exception as exc:  # noqa: BLE001 - a check never fails the job
                    drains["meshEscape"] = {"error": str(exc)}
        drains["ok"] = bool(drains["ok"] and ok)
        rep["drains"] = drains
    if "material" in doc:
        from .materials import material_from_spec

        mat = material_from_spec(doc["material"])
        rep["mass"] = rep["volume"] * mat.density / 1000.0
        rep["material"] = mat.name
    for key, mod, fn in (("impact", "impact", "impact_report"), ("vibration", "modal", "vibration_report")):
        if doc.get(key) is not None:
            if "material" not in doc:
                raise ValueError(f"field graph: '{key}' needs a material")
            import importlib

            run = getattr(importlib.import_module(f".{mod}", __package__), fn)
            rep[key] = run(g, doc[key], mat)
    if g.analyses:
        # what each `result` node's study said of the part it analyzed
        rep["analyses"] = {k: {"part": g.nodes[k].spec.get("part"), "safetyFactor": a.get("safetyFactor"),
                               "deflection": max((c["deflection"] for c in a.get("cases") or []), default=None),
                               "cells": a.get("analysisCells")}
                           for k, a in g.analyses.items() if k in g.nodes}
    if g.plan is not None:
        from .sharp import edge_accuracy, plan_refs

        Fg = g.field()
        try:
            rep["sharpEdges"] = edge_accuracy(mesh, plan_refs(g.plan), boundary=lambda P: np.abs(Fg(P)) < 1e-6)
        except Exception as exc:  # noqa: BLE001 - a measurement never fails the job
            rep["sharpEdges"] = {"error": str(exc)}
    if doc.get("check", True):
        from .checks.dfm import check_dfm

        spec = {"minWall": float(doc.get("minWall", process.min_feature if process is not None else 0.8)),
                "build": doc.get("build", (0, 0, 1))}
        try:
            d = check_dfm(mesh, spec)
            rep["printability"] = d
        except Exception as exc:  # noqa: BLE001 - a check never fails the job
            rep["printability"] = {"error": str(exc)}
        # strut lattices: how much strut length bridges at the build
        # direction (FDM prints a strut flatter than 45 degrees as a bridge;
        # powder and resin processes support them, so it is information there)
        from .homogenized import lattice_nodes
        from .struts import is_strut, lattice_printability

        for lat in lattice_nodes(g, g.output):
            if not is_strut(g.nodes[lat].spec.get("type", "gyroid")):
                continue
            try:
                lp = lattice_printability(g, lat, build=doc.get("build", (0, 0, 1)))
                lp["matters"] = process is None or process.kind == "fdm"
            except Exception as exc:  # noqa: BLE001
                lp = {"error": str(exc)}
            rep.setdefault("lattices", {})[lat] = lp
    if g.bodies and node is None:
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        rep["materials"] = _material_report(g, doc, mesh, out_dir, rep["name"], progress)
        rep["mass"] = rep["materials"]["mass"]     # the bodies' own materials, not `material`
    rep["seconds"] = round(time.time() - t0, 1)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, f"{rep['name']}.stl")
        mesh.export(path)
        rep["stl"] = path
        _write_graded(doc, rep, out_dir)
        if doc.get("step"):
            from .step import DEFAULT_FACES, DEFAULT_PATCHES, export_step

            opts = doc["step"] if isinstance(doc["step"], dict) else {}
            try:
                st = export_step(mesh, os.path.join(out_dir, f"{rep['name']}.step"),
                                 faces=int(opts.get("faces", DEFAULT_FACES)), name=rep["name"],
                                 surface=opts.get("surface", "facets"),
                                 patches=int(opts.get("patches", DEFAULT_PATCHES)))
                st.pop("shape", None)
                rep["step"] = st
            except Exception as exc:  # noqa: BLE001 - the STL is the deliverable; say why STEP is missing
                rep["step"] = {"error": f"{type(exc).__name__}: {exc}"}
        if render:
            try:
                from .render import render_views

                img = os.path.join(out_dir, f"{rep['name']}.png")
                render_views(mesh, img, title=rep["name"])
                rep["render"] = img
            except Exception:  # noqa: BLE001 - renders are optional
                pass
    return mesh, rep
