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
    # Pin supports as penalty springs at unit modulus (scale by E), or None.
    springs: object = None
    # Self-weight: N per mm^3 of solid, or None (see BoundaryConditions).
    gravity: np.ndarray | None = None

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
    pitch = (vol * frac / study.max_elements) ** (1.0 / 3.0)
    if study.min_member:
        # No finer than two cells per filter radius (1.3 x minMember, see
        # simp.filter_radius): the filter sets the feature scale, so finer
        # cells only cost time. On the arm link this cut the cells from 35k to
        # 18k and the study from ~255s to ~155s, and the design came out
        # stiffer (0.58mm vs 1.06mm deflection at the same 30% volume): the
        # finer grid had settled in a worse local optimum.
        pitch = max(pitch, 1.3 * float(study.min_member) / 2.0)
    return pitch


def _region_nodes(study: Study, mesh: HexMesh, region: str | None, shape: Shape | None, label: str,
                  tol_floor: float = 0.0):
    if shape is None:
        shape, kind = study.region_shape(region)
    else:
        kind = "inline"
    # A keep-out is a hole: the nodes that load or hold it are the ones
    # lining it, up to one cell away. Solid regions take nodes inside or on
    # their surface.
    # `tol_floor`: a verification grid finer than the optimization's still
    # reaches as far as the optimization did. On the printed hook the design
    # grid's support took the node plane 0.75 mm off the wall, so the part
    # was (correctly) designed to start there, and a 0.6 mm verification
    # grid then found nothing at the wall to hold.
    # Only as a fallback: widening it everywhere clamped more of every bolt
    # ring in verification and read the GRIP upper arm at SF 10 instead of 5.
    tol = mesh.pitch * (1.0 if kind == "keepOut" else 0.5)
    d = shape.sdf(mesh.node_xyz)
    nodes = d <= tol
    if kind != "keepOut" and (d <= 1e-9).sum() >= 4:
        # A solid region with nodes inside it holds or loads those, not a
        # plane half a cell outside it too. The extra plane let the printed
        # hook's optimizer hang the part off nodes 0.75 mm short of the wall:
        # it was designed not to touch the wall it is screwed to.
        nodes = d <= 1e-9
    if not nodes.any() and tol_floor > tol:
        nodes = d <= tol_floor
    if not nodes.any():
        name = region or "inline shape"
        raise ValueError(f"{label} on {name!r} touches no material: it lies outside the design space, "
                         "or every cell around it was removed. Move it onto the part.")
    return nodes


# Pin springs, relative to a cell's own stiffness: stiff enough to hold the
# pin to well under a percent of the part's deflection, soft enough not to
# wreck the solver's conditioning.
_PIN_STIFFNESS = 100.0
# Supports and loads touching at most this share of the cells are pinned
# solid (see build_domain).
_ANCHOR_MAX_FRACTION = 0.03


@dataclass
class BoundaryConditions:
    fixed: np.ndarray
    springs: object                # scipy CSR at unit modulus, or None
    support_nodes: np.ndarray
    loads: list                    # per case, force per dof (N)
    case_load_nodes: list          # per case, bool per node
    load_nodes: np.ndarray
    gravity: np.ndarray | None     # N per mm^3 of solid material
    regions: list = field(default_factory=list)   # node mask of every support and load region


def _cylinder_of(study: Study, region, shape, label):
    sh = shape if shape is not None else study.region_shape(region)[0]
    if not hasattr(sh, "axis"):
        raise ValueError(f"{label} needs a cylinder region (a bore keep-out or a cylinder) for its axis")
    return sh.p0, sh.axis


def boundary_conditions(study: Study, mesh: HexMesh, check_loads=True, tol_floor: float = 0.0) -> BoundaryConditions:
    """Supports, pins and loads on a mesh's nodes. Shared by the optimizer
    and the verification, so both analyze exactly the same problem."""
    xyz = mesh.node_xyz
    support_nodes = np.zeros(mesh.n_nodes, bool)
    fixed = np.zeros(mesh.n_dof, bool)
    spring_blocks = []
    regions = []
    for i, s in enumerate(study.supports):
        label = f"supports[{i}]"
        nodes = _region_nodes(study, mesh, s.region, s.shape, label, tol_floor)
        regions.append(nodes)
        support_nodes |= nodes
        idx = np.nonzero(nodes)[0]
        for a in range(3):
            if s.fix[a]:
                fixed[3 * idx + a] = True
        if s.pin:
            p0, ax = _cylinder_of(study, s.region, s.shape, label)
            d = xyz[idx] - p0
            rad = d - (d @ ax)[:, None] * ax
            rad /= np.maximum(np.linalg.norm(rad, axis=1, keepdims=True), 1e-12)
            blocks = rad[:, :, None] * rad[:, None, :] + np.outer(ax, ax)[None]
            spring_blocks.append((idx, blocks * _PIN_STIFFNESS * mesh.pitch))
    springs = None
    if spring_blocks:
        import scipy.sparse as sp
        rows, cols, vals = [], [], []
        for idx, blocks in spring_blocks:
            for i in range(3):
                for j in range(3):
                    rows.append(3 * idx + i)
                    cols.append(3 * idx + j)
                    vals.append(blocks[:, i, j])
        rows, cols, vals = (np.concatenate(v) for v in (rows, cols, vals))
        keep = ~(fixed[rows] | fixed[cols])
        springs = sp.csr_matrix((vals[keep], (rows[keep], cols[keep])), shape=(mesh.n_dof, mesh.n_dof))

    load_nodes = np.zeros(mesh.n_nodes, bool)
    loads, case_nodes = [], []
    for ci, case in enumerate(study.load_cases):
        f = np.zeros(mesh.n_dof)
        cn = np.zeros(mesh.n_nodes, bool)
        for li, ld in enumerate(case.loads):
            label = f"loadCases[{ci}].loads[{li}]"
            nodes = _region_nodes(study, mesh, ld.region, ld.shape, label, tol_floor)
            regions.append(nodes)
            idx = np.nonzero(nodes)[0]
            cn[idx] = True
            pts = xyz[idx]
            force = ld.force
            if ld.distribution == "bearing":
                # A shaft pushes on the half of the bore that faces the force,
                # most at the centre of that half: a cosine over it.
                p0, ax = _cylinder_of(study, ld.region, ld.shape, label)
                axial = (force @ ax) * ax
                radial = force - axial
                if np.any(radial):
                    d = pts - p0
                    rad = d - (d @ ax)[:, None] * ax
                    rad /= np.maximum(np.linalg.norm(rad, axis=1, keepdims=True), 1e-12)
                    w = np.maximum(rad @ (radial / np.linalg.norm(radial)), 0.0)
                    if w.sum() <= 0:
                        raise ValueError(f"{label}: no material on the loaded side of the bore")
                    f.reshape(-1, 3)[idx] += (w / w.sum())[:, None] * radial[None]
                force = axial
            if np.any(force):
                f.reshape(-1, 3)[idx] += force / len(idx)
            if np.any(ld.moment):
                # Forces that carry the moment and nothing else: the rigid
                # rotation field w x r, with w from the nodes' polar inertia.
                r = pts - pts.mean(axis=0)
                inertia = (r * r).sum() * np.eye(3) - r.T @ r
                omega = np.linalg.lstsq(inertia, ld.moment, rcond=None)[0]
                f.reshape(-1, 3)[idx] += np.cross(omega, r)
        f[fixed] = 0.0
        if check_loads and not np.any(f) and study.gravity is None:
            raise ValueError(f"load case {case.name!r} acts only on fixed nodes; "
                             "a load applied where the part is held does nothing")
        loads.append(f)
        case_nodes.append(cn)
        load_nodes |= cn

    gravity = None
    if study.gravity is not None:
        # g/cm^3 -> kg/mm^3 is 1e-6; times m/s^2 gives N per mm^3.
        gravity = study.material.density * 1e-6 * study.gravity
    return BoundaryConditions(fixed, springs, support_nodes, loads, case_nodes, load_nodes, gravity, regions)


def gravity_force(mesh: HexMesh, density: np.ndarray, gravity: np.ndarray, fixed: np.ndarray) -> np.ndarray:
    """Nodal self-weight for per-element solid fraction `density`."""
    per_node = (np.asarray(density, float) * mesh.pitch ** 3 / 8.0)[:, None] * gravity[None, :]
    f = np.zeros(mesh.n_dof)
    for a in range(3):
        f += np.bincount(mesh.edof[:, a::3].ravel(), weights=np.repeat(per_node[:, a], 8),
                         minlength=mesh.n_dof)
    f[fixed] = 0.0
    return f


def build_domain(study: Study, pitch: float | None = None) -> Domain:
    pitch = float(pitch) if pitch else choose_pitch(study)
    lo, hi = study.bounds()
    center = 0.5 * (lo + hi)
    dims = np.maximum(np.ceil((hi - lo) / pitch - 1e-9).astype(int), 1)
    # Centered grid, so a symmetry plane through the design space center is
    # also a plane between cells and every cell has an exact mirror. (Starting
    # the grid on the design space's minimum face along non-symmetric axes
    # was tried for the printed hook's half-cell overhang at its wall; it
    # moved every case's search path and lost 3 of 8 evals. Not worth it.)
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
    bc = boundary_conditions(study, mesh)
    support_nodes, fixed, load_nodes = bc.support_nodes, bc.fixed, bc.load_nodes
    loads = bc.loads
    names = [c.name for c in study.load_cases]
    weights = [c.weight for c in study.load_cases]

    # A small load or support (a bolt hole, a pin, a corner support) needs
    # material to act through: the cells touching it are pinned solid, as if
    # preserved, or the boundary-padded filter (simp.density_filter) makes
    # them too expensive to fill and the support is optimized away (the
    # textbook MBB beam's corners). A large one (a whole wall face, a shelf
    # surface) is left to the optimizer: pinning it forced a full skin over
    # the face, which read as part of the design not being optimized at all.
    for nodes in bc.regions:
        touch = nodes[mesh.enode].any(axis=1)
        if touch.sum() <= _ANCHOR_MAX_FRACTION * mesh.n_elem:
            pres_el = pres_el | touch

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
                  support_nodes=support_nodes, load_nodes=load_nodes, mirror=mirror, warnings=warnings,
                  springs=bc.springs, gravity=bc.gravity)
