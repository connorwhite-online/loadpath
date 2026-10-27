"""
Turn a study's shapes into the voxel problem the solver works on: which
cells exist, which are pinned solid (preserve), and which nodes carry the
supports and loads.

Classification is by supersampling each cell at 8 interior points, so a
cell's membership is its covered fraction rather than whether its center
happens to fall inside:

- preserve wins over design space, and is pinned to full density;
- a keep-out removes any design cell it touches at all (the final part must
  clear it), but only removes a preserve cell when it covers most of it, so
  a thin bearing boss around a bore keeps its wall in the analysis;
- islands of material that cannot reach a support are dropped: they carry
  no load and would make the stiffness matrix singular.

The exact keep-out and preserve geometry is restored after optimization by
mesh booleans, so the voxel classification only has to be right about load
paths, not about surfaces.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage

from .fem import HexMesh
from .shapes import Shape
from .spec import Study

__all__ = ["Domain", "build_domain", "choose_pitch"]

_SUB = np.array([[sx, sy, sz] for sx in (-0.25, 0.25) for sy in (-0.25, 0.25) for sz in (-0.25, 0.25)])
_CONN26 = np.ones((3, 3, 3), bool)


@dataclass
class Domain:
    pitch: float
    origin: np.ndarray            # min corner of cell (0, 0, 0), mm
    dims: tuple                   # cells per axis
    mesh: HexMesh
    preserve: np.ndarray          # bool per active element: pinned solid
    fixed: np.ndarray             # bool per dof
    loads: list                   # per load case: force vector per dof (N)
    case_names: list
    case_weights: list
    support_nodes: np.ndarray     # bool per node
    load_nodes: np.ndarray        # bool per node (any case)
    mirror: dict = field(default_factory=dict)   # axis -> element index of the mirror cell (-1 none)
    warnings: list = field(default_factory=list)

    @property
    def n_elem(self):
        return self.mesh.n_elem

    def full_grid(self, values: np.ndarray, fill=0.0) -> np.ndarray:
        g = np.full(self.dims, fill, float)
        ijk = self.mesh.elem_ijk
        g[ijk[:, 0], ijk[:, 1], ijk[:, 2]] = values
        return g


def _fraction(shapes: list[Shape], centers: np.ndarray, pitch: float) -> np.ndarray:
    if not shapes:
        return np.zeros(len(centers))
    hit = np.zeros((len(centers), len(_SUB)), bool)
    for k, off in enumerate(_SUB):
        pts = centers + off * pitch
        for s in shapes:
            hit[:, k] |= s.contains(pts)
    return hit.mean(axis=1)


def choose_pitch(study: Study) -> float:
    """Cell size: explicit, else the size that puts roughly maxElements cells
    in the design space (estimated from a coarse occupancy sample)."""
    if study.pitch:
        return float(study.pitch)
    lo, hi = study.bounds()
    vol = float(np.prod(hi - lo))
    # Coarse 24^3 sample of what fraction of the box is actually design space.
    n = 24
    axes = [np.linspace(lo[i], hi[i], n) for i in range(3)]
    pts = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)
    inside = np.zeros(len(pts), bool)
    for s in study.design_space + [r.shape for r in study.preserve]:
        inside |= s.contains(pts)
    frac = max(float(inside.mean()), 0.05)
    return (vol * frac / study.max_elements) ** (1.0 / 3.0)


def _region_nodes(study: Study, mesh: HexMesh, region: str | None, shape: Shape | None, label: str):
    if shape is None:
        shape, kind = study.region_shape(region)
    else:
        kind = "inline"
    # A keep-out is a hole: the nodes that load or hold it are the ones
    # lining it, up to one cell away. Solid regions take nodes inside or on
    # their surface.
    tol = mesh.pitch * (1.0 if kind == "keepOut" else 0.5)
    d = shape.sdf(mesh.node_xyz)
    nodes = d <= tol
    if not nodes.any():
        name = region or "inline shape"
        raise ValueError(f"{label} on {name!r} touches no material: it lies outside the design space, "
                         "or every cell around it was removed. Move it onto the part.")
    return nodes


def build_domain(study: Study, pitch: float | None = None) -> Domain:
    pitch = float(pitch) if pitch else choose_pitch(study)
    lo, hi = study.bounds()
    center = 0.5 * (lo + hi)
    dims = np.maximum(np.ceil((hi - lo) / pitch - 1e-9).astype(int), 1)
    # Centered grid, so a symmetry plane through the design space center is
    # also a plane between cells and every cell has an exact mirror.
    origin = center - 0.5 * dims * pitch
    ii, jj, kk = np.meshgrid(*[np.arange(d) for d in dims], indexing="ij")
    ijk = np.stack([ii.ravel(), jj.ravel(), kk.ravel()], axis=1)
    centers = origin + (ijk + 0.5) * pitch

    design_f = _fraction(study.design_space, centers, pitch)
    pres_f = _fraction([r.shape for r in study.preserve], centers, pitch)
    keep_f = _fraction([r.shape for r in study.keep_out], centers, pitch)
    preserve = (pres_f >= 0.5) & (keep_f < 0.5)
    design = (design_f >= 0.5) & ~preserve & (keep_f == 0)
    active = preserve | design
    if not active.any():
        raise ValueError("the design space is empty after removing keep-outs")

    warnings = []
    if study.min_member is not None and study.min_member < 1.5 * pitch:
        warnings.append(
            f"minMember {study.min_member:g}mm is under 1.5 cells ({1.5 * pitch:.2f}mm); members that thin "
            "cannot be resolved. Raise resolution.maxElements or set resolution.pitch.")

    # Support nodes on a provisional mesh, then drop islands that do not
    # reach one, then rebuild the mesh over what remains.
    mesh = HexMesh(ijk[active], pitch, origin)
    support_nodes = np.zeros(mesh.n_nodes, bool)
    for i, s in enumerate(study.supports):
        support_nodes |= _region_nodes(study, mesh, s.region, s.shape, f"supports[{i}]")
    grid = np.zeros(dims, bool)
    grid[tuple(ijk[active].T)] = True
    lbl, _ = ndimage.label(grid, structure=_CONN26)
    el_lbl = lbl[tuple(mesh.elem_ijk.T)]
    touches = np.zeros(mesh.n_elem, bool)
    touches |= support_nodes[mesh.enode].any(axis=1)
    keep_lbls = np.unique(el_lbl[touches])
    keep = np.isin(el_lbl, keep_lbls)
    if (~keep).any():
        dropped = int((~keep).sum())
        if preserve[active][~keep].any():
            warnings.append("some preserved geometry is not connected to any support and was left out of the "
                            "analysis; it will still be in the final part but carries no load")
        warnings.append(f"{dropped} cells not connected to a support were removed")
        act_idx = np.nonzero(active)[0]
        active[act_idx[~keep]] = False
        mesh = HexMesh(ijk[active], pitch, origin)

    pres_el = preserve[active]
    support_nodes = np.zeros(mesh.n_nodes, bool)
    fixed = np.zeros(mesh.n_dof, bool)
    for i, s in enumerate(study.supports):
        nodes = _region_nodes(study, mesh, s.region, s.shape, f"supports[{i}]")
        support_nodes |= nodes
        for a in range(3):
            if s.fix[a]:
                fixed[3 * np.nonzero(nodes)[0] + a] = True

    load_nodes = np.zeros(mesh.n_nodes, bool)
    loads, names, weights = [], [], []
    for ci, case in enumerate(study.load_cases):
        f = np.zeros(mesh.n_dof)
        for li, ld in enumerate(case.loads):
            nodes = _region_nodes(study, mesh, ld.region, ld.shape, f"loadCases[{ci}].loads[{li}]")
            idx = np.nonzero(nodes)[0]
            load_nodes[idx] = True
            for a in range(3):
                f[3 * idx + a] += ld.force[a] / len(idx)
        f[fixed] = 0.0
        if not np.any(f):
            raise ValueError(f"load case {case.name!r} acts only on fixed nodes; "
                             "a load applied where the part is held does nothing")
        loads.append(f)
        names.append(case.name)
        weights.append(case.weight)

    # A load or support needs material to act through: cells touching their
    # nodes are pinned solid, as if preserved. Without this, a support on the
    # edge of the design space (the textbook MBB beam's corners) can be
    # optimized away, since the boundary-padded filter (simp.density_filter)
    # makes edge cells expensive to fill.
    anchor = (support_nodes | load_nodes)[mesh.enode].any(axis=1)
    pres_el = pres_el | anchor

    mirror = {}
    for axis in study.symmetry:
        a = "xyz".index(axis)
        m_ijk = mesh.elem_ijk.copy()
        m_ijk[:, a] = dims[a] - 1 - m_ijk[:, a]
        lut = np.full(dims, -1, np.int64)
        lut[tuple(mesh.elem_ijk.T)] = np.arange(mesh.n_elem)
        mirror[axis] = lut[tuple(m_ijk.T)]
        unmatched = float((mirror[axis] < 0).mean())
        if unmatched > 0.02:
            warnings.append(f"symmetry '{axis}': {unmatched:.0%} of cells have no mirror; the design space, "
                            "preserve or keep-outs are not symmetric about that plane")

    return Domain(pitch=pitch, origin=origin, dims=tuple(int(d) for d in dims), mesh=mesh,
                  preserve=pres_el, fixed=fixed, loads=loads, case_names=names, case_weights=weights,
                  support_nodes=support_nodes, load_nodes=load_nodes, mirror=mirror, warnings=warnings)
