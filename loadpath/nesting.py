"""
Build-volume nesting: many parts (and copies) packed into one printer's
build volume, as a print bureau or a powder-bed job does.

Two modes:

- "bed" (FDM, resin, LPBF metal): every part stands on the plate in the
  orientation it was designed to print in, turned only about the build
  axis; footprints are packed bottom-left on the plate.
- "volume" (SLS, MJF): parts float in the powder cake and stack in 3D;
  each takes the lowest free position, so the build (and its powder and
  time) stays short. `rotations="any"` also tries the 24 axis-aligned
  orientations, which the anisotropy of a powder part rarely forbids.

Parts are voxelized conservatively (every cell the solid or its surface
touches), the placed ones grown by the gap, and a part's free positions are
found for every translation at once by an FFT correlation of the two grids.
The gap between the delivered surfaces is then measured, not assumed.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import numpy as np
import trimesh

__all__ = ["nest", "NestResult", "orientations", "BED_PROCESSES", "VOLUME_PROCESSES"]

BED_PROCESSES = ("fdm", "sla", "dlp", "lpbf", "jetting", "polyjet")
VOLUME_PROCESSES = ("sls", "mjf")


@dataclass
class Placement:
    part: int                       # index into the parts given
    copy: int
    transform: np.ndarray           # 4x4, part as given -> build volume
    min: list
    max: list


@dataclass
class NestResult:
    placements: list
    unplaced: list                  # (part, copy) pairs that did not fit
    volume: tuple
    mode: str
    pitch: float
    gap: float
    meshes: list = field(repr=False, default_factory=list)

    def combined(self) -> trimesh.Trimesh:
        out = []
        for p in self.placements:
            m = self.meshes[p.part].copy()
            m.apply_transform(p.transform)
            out.append(m)
        return trimesh.util.concatenate(out) if out else trimesh.Trimesh()

    def report(self) -> dict:
        placed_vol = sum(abs(self.meshes[p.part].volume) for p in self.placements)
        height = max((p.max[2] for p in self.placements), default=0.0)
        bx, by, bz = self.volume
        rep = {
            "mode": self.mode, "buildVolume": list(self.volume), "gap": self.gap, "pitch": self.pitch,
            "placed": len(self.placements), "unplaced": [{"part": a, "copy": b} for a, b in self.unplaced],
            "buildHeight": float(height),
            # of the build volume up to the build height (powder beds pay
            # for the whole cake that high)
            "packingDensity": float(placed_vol / (bx * by * height)) if height > 0 else 0.0,
            "placements": [{"part": p.part, "copy": p.copy, "transform": p.transform.round(6).tolist(),
                            "min": [round(v, 3) for v in p.min], "max": [round(v, 3) for v in p.max]}
                           for p in self.placements],
        }
        if self.mode == "bed":
            foot = sum(_footprint_area(self.meshes[p.part], p.transform) for p in self.placements)
            rep["plateCoverage"] = float(foot / (bx * by))
        rep["minGap"] = _min_gap(self)
        return rep


def orientations(kind: str = "z") -> list[np.ndarray]:
    """3x3 rotations to try: "z" the four quarter turns about the build
    axis (a part keeps its print orientation), "any" the 24 axis-aligned
    orientations, "none" only the part as given."""
    if kind == "none":
        return [np.eye(3)]
    rz = [np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], float)
          for c, s in ((1, 0), (0, 1), (-1, 0), (0, -1))]
    if kind == "z":
        return rz
    if kind != "any":
        raise ValueError("rotations must be 'z', 'any' or 'none'")
    out = []
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product((1, -1), repeat=3):
            r = np.zeros((3, 3))
            for i, (p, s) in enumerate(zip(perm, signs)):
                r[i, p] = s
            if np.linalg.det(r) > 0:
                out.append(r)
    return out


def _occupancy(mesh: trimesh.Trimesh, pitch: float):
    """Every cell (at `pitch`, from the mesh's min corner) the solid or its
    surface touches: (bool grid, min corner)."""
    from .cutcell import scan_voxelize

    lo, hi = mesh.bounds
    dims = np.maximum(np.ceil((hi - lo) / pitch).astype(int), 1)
    grid = scan_voxelize(mesh, pitch, lo, dims) if mesh.is_watertight else np.zeros(dims, bool)
    # the surface, sampled finer than a cell, marks every cell it crosses
    n = int(min(2_000_000, max(2000, 8 * mesh.area / (pitch * pitch))))
    pts = np.concatenate([mesh.vertices, trimesh.sample.sample_surface_even(mesh, n)[0]
                          if mesh.area > 0 else np.zeros((0, 3))])
    ijk = np.clip(np.floor((pts - lo) / pitch).astype(int), 0, dims - 1)
    grid[tuple(ijk.T)] = True
    return grid, lo


def _grow(grid: np.ndarray, k: int) -> np.ndarray:
    """Grid grown by k cells (a ball), padded so nothing is cut off."""
    from scipy import ndimage

    if k <= 0:
        return grid
    g = np.pad(grid, k)
    r = np.arange(-k, k + 1)
    ball = (r[:, None, None] ** 2 + r[None, :, None] ** 2 + r[None, None, :] ** 2) <= k * k + 1e-9
    if g.shape[2] == 1 + 2 * k and grid.shape[2] == 1:
        ball = ball[:, :, k:k + 1]
        g = g[:, :, k:k + 1]
    return ndimage.binary_dilation(g, structure=ball)


def _free(occ: np.ndarray, part: np.ndarray) -> np.ndarray:
    """Overlap count for every translation of `part` that keeps it inside
    `occ` (FFT correlation): 0 where it fits."""
    from scipy.signal import fftconvolve

    if any(p > o for p, o in zip(part.shape, occ.shape)):
        return np.zeros((0,) * occ.ndim)
    return fftconvolve(occ.astype(np.float32), part[::-1, ::-1, ::-1].astype(np.float32)
                       if part.ndim == 3 else part[::-1, ::-1].astype(np.float32), mode="valid")


def _footprint_area(mesh, transform) -> float:
    m = mesh.copy()
    m.apply_transform(transform)
    try:
        poly = trimesh.path.polygons.projected(m, normal=[0, 0, 1])
        return float(poly.area)
    except Exception:
        ext = m.bounds[1] - m.bounds[0]
        return float(ext[0] * ext[1])


def _min_gap(res: NestResult) -> float | None:
    """The least distance between any two placed parts' surfaces (mm),
    from dense surface samples (an upper bound within the sample spacing)."""
    from scipy.spatial import cKDTree

    if len(res.placements) < 2:
        return None
    clouds = []
    for p in res.placements:
        m = res.meshes[p.part]
        n = int(min(200_000, max(2000, 4 * m.area / (0.25 * res.pitch) ** 2)))
        pts = np.concatenate([m.vertices, trimesh.sample.sample_surface_even(m, n)[0]])
        clouds.append(trimesh.transform_points(pts, p.transform))
    best = np.inf
    boxes = [(c.min(axis=0), c.max(axis=0)) for c in clouds]
    for i in range(len(clouds)):
        tree = None
        for j in range(i + 1, len(clouds)):
            # boxes further apart than the best so far cannot beat it
            sep = np.maximum(0, np.maximum(boxes[i][0] - boxes[j][1], boxes[j][0] - boxes[i][1]))
            if np.linalg.norm(sep) >= best:
                continue
            tree = tree or cKDTree(clouds[i])
            d = tree.query(clouds[j], k=1, distance_upper_bound=best)[0]
            best = min(best, float(d.min()))
    return None if not np.isfinite(best) else best


def nest(parts, volume, gap: float = 2.0, mode: str = "volume", rotations: str | None = None,
         pitch: float | None = None, margin: float | None = None, max_cells: int = 1_000_000) -> NestResult:
    """Pack `parts` into a build volume.

    parts: meshes, or (mesh, copies) pairs. volume: (x, y, z) mm. gap: the
    least distance between parts (mm). mode: "bed" (on the plate, turned
    about z) or "volume" (stacked in the powder). rotations: "z", "any" or
    "none" (default "z" on a bed, "any" in a volume). margin: kept clear of
    the volume's walls (default the gap; on a bed the plate itself is not a
    wall). pitch: the packing grid (default from the volume, about
    max_cells cells). Largest parts go first; each copy takes the lowest,
    then frontmost, then leftmost free position over the orientations
    tried (on a bed the one with the least footprint row, so the plate
    fills from one edge)."""
    if mode not in ("bed", "volume"):
        raise ValueError("mode must be 'bed' or 'volume'")
    rotations = rotations or ("z" if mode == "bed" else "any")
    vol = np.asarray(volume, float)
    if vol.shape != (3,) or (vol <= 0).any():
        raise ValueError("volume must be three positive lengths (mm)")
    if gap < 0:
        raise ValueError("gap must be >= 0")
    margin = gap if margin is None else float(margin)
    items = []
    for i, p in enumerate(parts):
        mesh, copies = (p if isinstance(p, tuple) else (p, 1))
        if copies < 1:
            raise ValueError(f"parts[{i}]: copies must be >= 1")
        items.append((i, mesh, int(copies)))
    meshes = [m for _, m, _ in items]
    if pitch is None:
        # about max_cells cells over the volume (a plate's area on a bed)
        cells = float(np.prod(vol) if mode == "volume" else vol[0] * vol[1] * 50)
        pitch = max(float(cells / max_cells) ** (1 / 3), 0.5)
    k = int(np.ceil(gap / pitch - 1e-9))
    inner = vol - 2 * margin
    if mode == "bed":
        inner[2] = vol[2] - margin
    dims = np.floor(inner / pitch).astype(int)
    if (dims < 1).any():
        raise ValueError("the margin leaves no room in the build volume")
    origin = np.array([margin, margin, 0.0 if mode == "bed" else margin])
    occ = np.zeros(dims if mode == "volume" else dims[:2], bool)

    # each part's grid in each orientation, once
    shapes = {}
    for i, mesh, _ in items:
        rows = []
        for r in orientations(rotations):
            m = mesh.copy()
            t = np.eye(4)
            t[:3, :3] = r
            m.apply_transform(t)
            g, lo = _occupancy(m, pitch)
            if mode == "bed":
                ext = m.bounds[1] - m.bounds[0]
                if ext[2] > inner[2] + 1e-9:
                    continue
                g = g.any(axis=2)
            # identical grids (a symmetric part) need trying once
            if any(g.shape == q[0].shape and np.array_equal(g, q[0]) for q in rows):
                continue
            rows.append((g, t, lo, m.bounds[1] - m.bounds[0]))
        shapes[i] = rows

    order = sorted(((i, c) for i, mesh, n in items for c in range(n)),
                   key=lambda ic: -float(np.prod(meshes[ic[0]].extents)))
    placements, unplaced = [], []
    for i, c in order:
        best = None
        for g, t, lo, ext in shapes[i]:
            if mode == "volume" and (ext > inner + 1e-9).any():
                continue
            if mode == "bed" and (ext[:2] > inner[:2] + 1e-9).any():
                continue
            ov = _free(occ, g)
            if ov.size == 0:
                continue
            ok = np.argwhere(ov < 0.5)
            if not len(ok):
                continue
            if mode == "volume":
                # lowest top, then front, then left
                key = (ok[:, 2] + g.shape[2]) * 1e12 + ok[:, 1] * 1e6 + ok[:, 0]
                j = ok[int(np.argmin(key))]
                score = (j[2] + g.shape[2], j[1], j[0])
            else:
                key = (ok[:, 1] + g.shape[1]) * 1e6 + ok[:, 0] + g.shape[0] * 1e-3
                j = ok[int(np.argmin(key))]
                score = (j[1] + g.shape[1], j[0] + g.shape[0], 0)
            if best is None or score < best[0]:
                best = (score, j, g, t, lo, ext)
        if best is None:
            unplaced.append((i, c))
            continue
        _, j, g, t, lo, ext = best
        jj = np.append(j, 0) if mode == "bed" else j
        pos = origin + jj * pitch
        move = np.eye(4)
        move[:3, 3] = pos - lo
        tf = move @ t
        # occupy it, grown by the gap
        sl = tuple(slice(a, a + s) for a, s in zip(j, g.shape))
        placed = np.zeros_like(occ)
        placed[sl] = g
        if k:
            gk = _grow(placed[..., None] if mode == "bed" else placed, k)
            gk = gk[..., 0] if mode == "bed" else gk
            crop = tuple(slice(k, k + s) for s in placed.shape)
            placed = gk[crop]
        occ |= placed
        placements.append(Placement(i, c, tf, (pos).tolist(), (pos + ext).tolist()))
    return NestResult(placements, unplaced, tuple(vol.tolist()), mode, float(pitch), float(gap), meshes)
