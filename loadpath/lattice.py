"""
Field-driven lattices: fill a part's interior with a triply periodic minimal
surface (TPMS) sheet whose wall thickness follows the stress, keeping a solid
skin and solid material wherever the part works hard.

Everything is an implicit field, nTop style:

- the part as a signed distance (negative inside), from a voxelization of
  its mesh and distance transforms;
- the TPMS as an approximate distance, the level-set function divided by
  its gradient's magnitude, so a sheet's wall comes out in millimetres;
- the stress as a utilization field (von Mises over the allowable),
  interpolated from a verification of the solid part.

The lattice wall runs from `wall_min` where the part idles to `wall_max`,
and material above `solid_above` utilization stays solid, joined to the
lattice by a smooth union. The skin closes the lattice in: right for FDM
(it prints like infill), but powder (SLS, MJF) and resin (SLA) would be
trapped inside, so `open_skin` leaves the lattice exposed instead.

Units are millimetres throughout.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from .spec import Study

__all__ = ["LatticeOptions", "tpms", "lattice_infill", "TPMS"]


def _gyroid(x, y, z):
    return np.sin(x) * np.cos(y) + np.sin(y) * np.cos(z) + np.sin(z) * np.cos(x)


def _gyroid_grad(x, y, z):
    return (np.cos(x) * np.cos(y) - np.sin(z) * np.sin(x),
            np.cos(y) * np.cos(z) - np.sin(x) * np.sin(y),
            np.cos(z) * np.cos(x) - np.sin(y) * np.sin(z))


def _schwarz_p(x, y, z):
    return np.cos(x) + np.cos(y) + np.cos(z)


def _schwarz_p_grad(x, y, z):
    return -np.sin(x), -np.sin(y), -np.sin(z)


def _diamond(x, y, z):
    return (np.sin(x) * np.sin(y) * np.sin(z) + np.sin(x) * np.cos(y) * np.cos(z)
            + np.cos(x) * np.sin(y) * np.cos(z) + np.cos(x) * np.cos(y) * np.sin(z))


def _diamond_grad(x, y, z):
    sx, cx, sy, cy, sz, cz = np.sin(x), np.cos(x), np.sin(y), np.cos(y), np.sin(z), np.cos(z)
    return (cx * sy * sz + cx * cy * cz - sx * sy * cz - sx * cy * sz,
            sx * cy * sz - sx * sy * cz + cx * cy * cz - cx * sy * sz,
            sx * sy * cz - sx * cy * sz - cx * sy * sz + cx * cy * cz)


TPMS = {"gyroid": (_gyroid, _gyroid_grad),
        "schwarz-p": (_schwarz_p, _schwarz_p_grad),
        "diamond": (_diamond, _diamond_grad)}


def tpms(kind: str, pts: np.ndarray, cell: float) -> np.ndarray:
    """Approximate signed distance (mm) to the TPMS mid-surface, `cell` mm per
    period: the level-set value over its gradient's magnitude. Exact on the
    surface, and good to a few percent out to a quarter cell, which is as far
    as a sheet wall reaches."""
    if kind not in TPMS:
        raise ValueError(f"lattice type must be one of {sorted(TPMS)}, got {kind!r}")
    f, grad = TPMS[kind]
    k = 2.0 * np.pi / cell
    x, y, z = (k * pts[:, i] for i in range(3))
    g = np.stack(grad(x, y, z), axis=-1) * k
    return f(x, y, z) / np.maximum(np.linalg.norm(g, axis=-1), 0.3 * k)


@dataclass
class LatticeOptions:
    kind: str = "gyroid"
    cell: float = 8.0               # mm per period
    skin: float = 1.2               # mm of solid shell; 0 leaves the lattice open
    wall_min: float = 0.8           # mm, where the part idles (>= the process's min feature)
    wall_max: float = 2.0           # mm, where it nears solid_above
    solid_above: float = 0.5        # utilization above which material stays solid
    safety_factor: float = 2.0      # the allowable is strength / this
    step: float | None = None       # field sampling, mm (default: wall_min / 2.5)
    max_points: int = 12_000_000

    @classmethod
    def from_spec(cls, spec: dict, process=None) -> "LatticeOptions":
        spec = dict(spec or {})
        o = cls()
        for key, attr in (("type", "kind"), ("cell", "cell"), ("skin", "skin"), ("minWall", "wall_min"),
                          ("maxWall", "wall_max"), ("solidAbove", "solid_above")):
            if key in spec:
                setattr(o, attr, spec[key] if attr == "kind" else float(spec[key]))
        if process is not None and o.wall_min < process.min_feature:
            o.wall_min = process.min_feature
        if o.wall_max < o.wall_min:
            raise ValueError("lattice maxWall must be at least minWall")
        if o.cell < 4.0 * o.wall_max:
            raise ValueError(f"lattice cell {o.cell:g}mm is too small for {o.wall_max:g}mm walls "
                             "(at least 4 x maxWall, or the channels close)")
        return o


def _part_distance(mesh, step, lo, dims):
    """Signed distance to the mesh surface on the grid (negative inside),
    from an inside/outside voxelization and two distance transforms."""
    from .verify import voxelize

    hi = lo + np.asarray(dims) * step
    solid, origin = voxelize(mesh, step, box=(lo, hi))
    if solid.shape != tuple(dims):     # rounding at the last cell
        out = np.zeros(dims, bool)
        s = tuple(slice(0, min(a, b)) for a, b in zip(solid.shape, dims))
        out[s] = solid[s]
        solid = out
    d = (ndimage.distance_transform_edt(~solid) - ndimage.distance_transform_edt(solid)) * step
    return d + np.where(solid, 0.5 * step, -0.5 * step)


def lattice_infill(study: Study, mesh, fields: dict | None = None, opts: LatticeOptions | None = None):
    """Returns (trimesh.Trimesh, info). `fields` is verify_part(...,
    fields=True)["fields"] for the solid part: without it the lattice is
    uniform at wall_min."""
    import manifold3d as m3d
    import trimesh
    from skimage.measure import marching_cubes

    from .reconstruct import _smooth_union
    from .shapes import manifold_from_trimesh, trimesh_from_manifold

    o = opts or LatticeOptions()
    lo, hi = mesh.bounds
    lo = lo - 2.0 * o.wall_max
    hi = hi + 2.0 * o.wall_max
    step = o.step or o.wall_min / 2.5
    step = max(step, (float(np.prod(hi - lo)) / o.max_points) ** (1.0 / 3.0))
    dims = np.maximum(np.ceil((hi - lo) / step).astype(int), 2)
    d_part = _part_distance(mesh, step, lo, dims)

    idx = np.stack(np.meshgrid(*[np.arange(n) for n in dims], indexing="ij"), -1).reshape(-1, 3)
    pts = lo + (idx + 0.5) * step
    inside = d_part.ravel() < 0

    util = np.zeros(len(pts))
    if fields is not None:
        from scipy.spatial import cKDTree

        vm = np.max([c["vonMises"] for c in fields["cases"]], axis=0)
        allow = study.material.strength / o.safety_factor
        tree = cKDTree(fields["nodes"])
        dist, near = tree.query(pts[inside], k=4)
        w = 1.0 / np.maximum(dist, 1e-6) ** 2
        util[inside] = (w * vm[near]).sum(axis=1) / w.sum(axis=1) / allow
    # wall from utilization, square-root graded so moderate stress already
    # thickens it; solid above solid_above
    u = np.clip(util / o.solid_above, 0.0, 1.0)
    wall = o.wall_min + (o.wall_max - o.wall_min) * np.sqrt(u)
    d_lat = np.abs(tpms(o.kind, pts, o.cell)) - 0.5 * wall
    d_solid = (o.solid_above - util) * o.cell          # negative where it must stay solid
    d_core = _smooth_union(d_lat, d_solid, o.wall_max)
    dp = d_part.ravel()
    if o.skin > 0:
        d = np.maximum(dp, np.minimum(d_core, -dp - o.skin))
    else:
        d = np.maximum(dp, d_core)
    d = d.reshape(dims)

    verts, faces, _, _ = marching_cubes(d, 0.0, spacing=(step, step, step))
    verts = verts + lo + 0.5 * step
    lat = trimesh.Trimesh(verts, faces[:, ::-1], process=True)
    if lat.volume < 0:
        lat.invert()
    part = manifold_from_trimesh(lat)
    solid_vol = float(manifold_from_trimesh(mesh).volume())
    # the user's geometry stays exact: preserved shapes solid, keep-outs clear
    if study.preserve:
        part = part + m3d.Manifold.batch_boolean([r.shape.to_manifold() for r in study.preserve],
                                                 m3d.OpType.Add)
        part = part ^ manifold_from_trimesh(mesh)
    if study.keep_out:
        part = part - m3d.Manifold.batch_boolean([r.shape.to_manifold() for r in study.keep_out],
                                                 m3d.OpType.Add)
    # Drop lattice fragments the trim cut loose. (decompose() also returns
    # each enclosed cavity, as a negative volume: those stay.)
    pieces = part.decompose()
    big = max(b.volume() for b in pieces)
    loose = [b for b in pieces if 0 < b.volume() < 0.01 * big]
    for b in loose:
        part = part - b
    out = trimesh_from_manifold(part)
    vol = float(part.volume())
    # with a skin, everything the lattice leaves open is walled in: where
    # powder or resin would stay trapped
    voids = solid_vol - vol if o.skin > 0 else 0.0
    info = {"kind": o.kind, "cell": o.cell, "skin": o.skin, "wallMin": o.wall_min, "wallMax": o.wall_max,
            "step": step, "volume": vol, "solidVolume": solid_vol, "volumeRatio": vol / solid_vol,
            "mass": vol * study.material.density / 1000.0, "graded": fields is not None,
            "solidFraction": float((util[inside] >= o.solid_above).mean()) if inside.any() else 0.0,
            "enclosedVoid": voids, "droppedBodies": len(loose)}
    return out, info
