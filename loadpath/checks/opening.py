"""
Does the part OPEN from above? The "can you put things in it" probe.

A container read as a container only when its cavities break through the top.
Concept blockouts kept coming back as closed pills with the compartments
inside them, or with cavities that only nicked the rounded ends, and neither
the fill ratio (a sealed hollow is mostly empty) nor the trapped-void probe (a
nicked cavity drains) can tell those apart from a real organizer.

Looking straight down can. Sample the surface densely, keep the HIGHEST point
in each grid cell (the first thing you'd hit from above), and measure how much
of the footprint's interior sits deep below the top. An open pocket shows its
floor; a lidded hollow or a solid shows its top. The footprint is eroded first
so a soft rounded rim, which is low near the edge, doesn't count as open.
"""
import numpy as np
from scipy import ndimage
import trimesh

# How far below the top a cell must sit to count as open, as a share of the
# part's height.
_DEPTH_FRACTION = 0.3
# Footprint erosion, as a share of the smaller plan dimension.
_ERODE_FRACTION = 0.12


def check_opening(mesh, spec=None):
    spec = spec or {}
    lo, hi = mesh.bounds
    size = hi - lo
    height = float(size[2])
    plan = float(min(size[0], size[1]))
    if height <= 0 or plan <= 0:
        return {"error": "degenerate bounds"}

    cell = float(spec.get("cellMm") or max(plan / 60.0, 0.5))
    nx = int(np.ceil(size[0] / cell)) + 1
    ny = int(np.ceil(size[1] / cell)) + 1

    count = int(min(max(mesh.area / (cell * cell) * 6, 20_000), 400_000))
    pts, _ = trimesh.sample.sample_surface(mesh, count, seed=7)
    ix = np.clip(((pts[:, 0] - lo[0]) / cell).astype(int), 0, nx - 1)
    iy = np.clip(((pts[:, 1] - lo[1]) / cell).astype(int), 0, ny - 1)

    top = np.full((nx, ny), -np.inf)
    np.maximum.at(top, (ix, iy), pts[:, 2])
    covered = np.isfinite(top)
    # Cells a pocket's vertical walls cover only sparsely can miss samples;
    # close small holes so the footprint is the part's silhouette.
    footprint = ndimage.binary_closing(covered, iterations=2)
    footprint = ndimage.binary_fill_holes(footprint)
    erode = max(1, int(round(plan * _ERODE_FRACTION / cell)))
    interior = ndimage.binary_erosion(footprint, iterations=erode)
    n_interior = int(interior.sum())
    if n_interior == 0:
        return {"error": "footprint too small to probe", "cellMm": round(cell, 3)}

    deep = float(hi[2]) - _DEPTH_FRACTION * height
    # An empty interior cell (no surface at all beneath it) is a through-hole:
    # open by definition.
    open_cells = interior & (~covered | (top < deep))
    frac = float(open_cells.sum()) / n_interior
    return {
        "openFraction": round(frac, 4),
        "cellMm": round(cell, 3),
        "interiorCells": n_interior,
    }
