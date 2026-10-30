"""
Internal ceilings: the overhangs inside a part that no support can reach.

In powder-bed metal (LPBF: AlSi10Mg, Ti64, 316L) an overhang outside the
part gets a support that is cut off afterwards. An overhang inside a closed
channel or plenum cannot: a support there stays in the part, and without
one the ceiling curls, drosses or collapses. So a heat exchanger, a manifold
or anything with internal flow paths must be self-supporting inside, along
the direction it is built, and the same part built another way up can go
from hundreds of square centimetres of flat ceiling to almost none (a box
plenum built along +z has a flat roof; built along the cube diagonal every
face of it leans 54.7 degrees).

`internal_ceilings` measures that: every face of the part whose material is
above it and whose cavity below it is enclosed (not open to the outside once
the ports are plugged), leaning flatter than `limit` from level. Faces deep
inside a TPMS core can be left out with `exclude`: a sheet gyroid or diamond
prints unsupported at the usual cell sizes, though its surface has
down-facing points.

`rank_build_directions` measures it for many directions at once (the mesh
and the cavity labelling are done once; each direction is a dot product), so
an agent can be told which way up to print before it designs for one.

The part can be a field (`field(P) -> sdf`, negative = solid, mm) over
bounds, or a mesh.

    rep = internal_ceilings(field, bounds, up="+z", plugs=plugs, pitch=0.5)
    best = rank_build_directions(field, bounds, plugs=plugs, pitch=0.5)[0]
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage

__all__ = ["CavitySurface", "cavity_surface", "internal_ceilings", "rank_build_directions",
           "candidate_directions", "LIMIT"]

# self-supporting limit, degrees from level, for powder-bed metal and FDM alike
LIMIT = 45.0
# "flat": within this many degrees of level, the ceilings that fail first
FLAT = 20.0


class CavitySurface:
    """The part's surface split once into faces: centres, unit normals
    (out of the material), areas, and whether the space just off each face
    is an enclosed cavity. Every ceiling question below is a dot product on
    these."""

    def __init__(self, centers, normals, areas, internal, pitch, cavity_volume):
        self.centers = centers
        self.normals = normals
        self.areas = areas
        self.internal = internal
        self.pitch = pitch
        self.cavity_volume = cavity_volume

    def ceilings(self, up, limit=LIMIT, flat=FLAT) -> dict:
        from .am import build_vector, parse_build_direction

        u = build_vector(parse_build_direction(up) if isinstance(up, str) else up)
        down = -(self.normals @ u)                # 1 for a face looking straight down
        over = self.internal & (down > np.cos(np.radians(90.0 - limit)))
        flat_ = self.internal & (down > np.cos(np.radians(flat)))
        out = {"up": np.round(u, 4).tolist(),
               "internalOverhangArea": float(self.areas[over].sum()),
               "internalFlatArea": float(self.areas[flat_].sum()),
               "limit": limit, "pitch": self.pitch}
        if over.any():
            # where the worst of it is: the 10 mm cube holding the most flat ceiling, else overhang
            from .heatx import _densest

            sel = flat_ if flat_.any() else over
            out["worstAt"] = _densest(self.centers[sel], self.areas[sel], 10.0)
        return out


def _plugged(solid, lo, pitch, plugs):
    if not plugs:
        return solid
    from .checks.networks import _refill_plugs

    s = solid.copy()
    _refill_plugs(s, np.asarray(lo, float), pitch, plugs)
    return s


def _enclosed(solid):
    """Void not connected to the grid's border (6-connected: a cavity behind
    a one-voxel wall stays a cavity)."""
    lbl, _ = ndimage.label(~solid)
    border = np.unique(np.concatenate([lbl[0].ravel(), lbl[-1].ravel(), lbl[:, 0].ravel(), lbl[:, -1].ravel(),
                                       lbl[:, :, 0].ravel(), lbl[:, :, -1].ravel()]))
    return (lbl > 0) & ~np.isin(lbl, border)


def cavity_surface(part, bounds=None, pitch=None, plugs=None, exclude=None, max_cells=40_000_000) -> CavitySurface:
    """Split `part` (a field with `bounds` = (lo, hi), or a trimesh) into
    faces and label the ones that face an enclosed cavity. `plugs` (as for
    check_networks) close the ports first, so the fluid spaces count as
    enclosed; `exclude(points) -> bool` drops faces (e.g. the TPMS core's
    own sheet)."""
    if callable(part):
        from skimage.measure import marching_cubes

        lo, hi = (np.asarray(b, float) for b in bounds)
        if pitch is None:
            pitch = max(float(np.prod(hi - lo) / max_cells) ** (1 / 3), 0.2)
        lo = lo - 0.0137 * pitch
        dims = np.maximum(np.ceil((hi - lo) / pitch).astype(int) + 1, 2)
        if np.prod(dims) > max_cells:
            raise ValueError(f"a {pitch:g} mm grid over this part is {np.prod(dims) / 1e6:.0f}M cells; "
                             f"use pitch >= {float(np.prod(hi - lo) / max_cells) ** (1 / 3):.2f}")
        axes = [lo[i] + np.arange(dims[i]) * pitch for i in range(3)]
        F = np.empty(tuple(dims), np.float32)
        for i, x in enumerate(axes[0]):           # a slab at a time: the field's temporaries stay small
            Y, Z = np.meshgrid(axes[1], axes[2], indexing="ij")
            F[i] = np.asarray(part(np.stack([np.full(Y.size, x), Y.ravel(), Z.ravel()], 1)), float).reshape(Y.shape)
        F = np.pad(F, 1, constant_values=float(3 * pitch))
        lo = lo - pitch
        F[F == 0] = 1e-6
        verts, faces, _, _ = marching_cubes(F, 0.0, spacing=(pitch,) * 3, allow_degenerate=False)
        verts += lo
        tri = verts[faces]
        solid = F < 0
    else:
        from .sdf_kit import _solid_voxels

        mesh = part
        if pitch is None:
            pitch = max(float(np.prod(mesh.extents) / max_cells) ** (1 / 3), 0.2)
        solid, origin = _solid_voxels(mesh, pitch, pad=2)
        lo = origin
        tri = np.asarray(mesh.triangles)
    cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    areas = 0.5 * np.linalg.norm(cross, axis=1)
    keep = areas > 0
    normals = np.zeros_like(cross)
    normals[keep] = cross[keep] / (2.0 * areas[keep, None])
    centers = tri.mean(axis=1)
    if callable(part):
        # marching cubes on (negative = solid) winds its faces into the material: flip
        normals = -normals
        # vertex-averaged normals: the facets of marching cubes scatter a few degrees
        g = np.stack(np.gradient(F, pitch), -1)
        idx = np.clip(np.round((centers - lo) / pitch).astype(int), 0, np.asarray(F.shape) - 1)
        gn = g[idx[:, 0], idx[:, 1], idx[:, 2]]
        gl = np.linalg.norm(gn, axis=1)
        ok = gl > 1e-9
        normals[ok] = gn[ok] / gl[ok, None]
        del g
    cav = _enclosed(_plugged(solid, lo, pitch, plugs))
    probe = centers + normals * (1.0 * pitch)
    ijk = np.round((probe - lo) / pitch).astype(int)
    inb = np.all((ijk >= 0) & (ijk < np.asarray(cav.shape)), axis=1)
    internal = np.zeros(len(centers), bool)
    internal[inb] = cav[ijk[inb, 0], ijk[inb, 1], ijk[inb, 2]]
    internal &= keep
    if exclude is not None:
        internal &= ~np.asarray(exclude(centers), bool)
    return CavitySurface(centers, normals, areas, internal, float(pitch), float(cav.sum()) * pitch ** 3)


def internal_ceilings(part, bounds=None, up="+z", pitch=None, plugs=None, exclude=None,
                      limit=LIMIT, flat=FLAT) -> dict:
    """Area (mm^2) of the part's internal faces that lean flatter than
    `limit` degrees from level when built along `up` (internalOverhangArea),
    and of those within `flat` degrees of level (internalFlatArea)."""
    return cavity_surface(part, bounds, pitch, plugs, exclude).ceilings(up, limit, flat)


def candidate_directions(extra: int = 0):
    """The 6 axes, 12 edge diagonals and 8 corner diagonals of a cube, both
    ways up, plus `extra` more spread over the sphere (Fibonacci)."""
    d = []
    for v in np.array(np.meshgrid([-1, 0, 1], [-1, 0, 1], [-1, 0, 1])).reshape(3, -1).T:
        if v.any():
            d.append(v / np.linalg.norm(v))
    if extra:
        k = np.arange(extra) + 0.5
        phi = np.arccos(1 - 2 * k / extra)
        th = np.pi * (1 + 5 ** 0.5) * k
        d.extend(np.stack([np.cos(th) * np.sin(phi), np.sin(th) * np.sin(phi), np.cos(phi)], 1))
    return np.asarray(d)


def rank_build_directions(part, bounds=None, pitch=None, plugs=None, exclude=None, candidates=None,
                          limit=LIMIT, flat=FLAT, surface: CavitySurface | None = None) -> list:
    """Every candidate build direction, least internal overhang first (ties:
    least flat ceiling). Each entry is internal_ceilings' report plus
    "label"."""
    from .am import direction_label, parse_build_direction

    s = surface or cavity_surface(part, bounds, pitch, plugs, exclude)
    cands = candidate_directions() if candidates is None else [
        parse_build_direction(c) if isinstance(c, str) else np.asarray(c, float) for c in candidates]
    out = []
    for c in cands:
        if isinstance(c, tuple):
            from .am import build_vector
            v, lab = build_vector(c), direction_label(c)
        else:
            v = np.asarray(c, float) / np.linalg.norm(c)
            pd = parse_build_direction(v.tolist())
            lab = direction_label(pd)
        r = s.ceilings(v, limit, flat)
        r["label"] = lab
        out.append(r)
    out.sort(key=lambda r: (round(r["internalOverhangArea"], 1), r["internalFlatArea"]))
    return out
