"""
Field-driven lattices: fill a part's interior with a triply periodic minimal
surface (TPMS) sheet whose wall thickness follows the stress, keeping a solid
skin and solid material wherever the part works hard.

Everything is an implicit field, nTop style:

- the part as a signed distance (negative inside), from a voxelization of
  its mesh and distance transforms;
- the TPMS as an approximate distance, the level-set function divided by
  its gradient's magnitude, so a sheet's wall comes out in millimetres;
- solid zones (the skin, collars round preserved geometry, keep-outs,
  supports and loads, and cells the sizing turned solid) as distances too,
  joined to the lattice by a smooth union, so every sheet meets the skin
  and every boss with a fillet instead of a crease.

Two ways to grade the wall:

- `target_sf` set (the sizing that means something): the core is a
  homogenized material, a sheet of relative density rho having modulus
  E * C rho^n and a peak wall stress A rho^-m times the stress it carries
  on average (unit-cell FEA, `homogenize`; the fits are in SHEETS). A
  voxel FEA of the part with the skin solid and every core cell at its
  sheet's modulus gives each cell's stress; each cell's density is set,
  fully-stressed-design style, to the least that keeps its walls at the
  target safety factor, cells that would need more than `wall_max` turn
  solid, and the analysis repeats as the load moves to wherever it is
  stiffest. That is the lattice an engineer can sign off on: the voids are
  where the stress is not.
- otherwise, from a verification of the solid part (`fields`): the wall runs
  from `wall_min` where the part idles to `wall_max`, and material above
  `solid_above` utilization stays solid. Quick, but the solid part's stress
  is not the lattice part's.

Printability is part of the design, not an afterthought:

- every wall is at least the process's minimum feature;
- powder (SLS, MJF) and resin (SLA, volumetric) would stay trapped in a
  lattice under a closed skin, so each enclosed void gets drain holes
  (lowest and highest along the build, away from the stress), pockets too
  small to drain are filled solid, and the report says what is still
  trapped (it should be nothing);
- FDM prints the closed skin like infill; the report gives the lattice's
  own down-facing area (gyroid and diamond sheets self-support at the
  usual cell sizes; Schwarz P has flat ceilings).

Units are millimetres throughout.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from scipy import ndimage

from .spec import Study

__all__ = ["LatticeOptions", "tpms", "lattice_infill", "TPMS", "SHEETS", "homogenize",
           "relative_density", "wall_for_density", "envelope_mesh"]


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

# Homogenized sheets, from `homogenize` (periodic unit cell, 40 voxels per
# cell, nu = 0.35, relative density 0.14-0.6; the fits are within 5% of
# every point): modulus E*/E = C rho^n (uniaxial; the cells are cubic, so
# that is every axis), and the peak wall von Mises (99.5th percentile of
# nodal values) over the macro von Mises, = A rho^-m, the larger of a
# uniaxial and a shear macro stress (uniaxial for gyroid and Schwarz P,
# shear for thin diamond). Elastic first yield: the walls start yielding
# there, well before the cell collapses, which is the right reading for a
# linear check. Per gram, diamond is the stiffest and strongest of the
# three and Schwarz P the weakest.
SHEETS = {
    "gyroid": {"C": 0.648, "n": 1.43, "A": 3.01, "m": 1.10},
    "diamond": {"C": 0.671, "n": 1.35, "A": 2.62, "m": 1.22},
    "schwarz-p": {"C": 0.669, "n": 1.68, "A": 3.82, "m": 1.04},
}

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


@lru_cache(maxsize=None)
def _density_table(kind: str):
    """Relative density of a sheet |tpms| < wall/2 against wall/cell, sampled
    on one cell (the distance approximation makes it a little sub-linear)."""
    n = 96
    c = (np.arange(n) + 0.5) / n
    g = np.stack(np.meshgrid(c, c, c, indexing="ij"), -1).reshape(-1, 3)
    d = np.sort(np.abs(tpms(kind, g, 1.0)))
    t = np.linspace(0.0, 0.8, 161)
    rho = np.searchsorted(d, 0.5 * t) / d.size
    return t, rho


def relative_density(kind: str, wall, cell: float):
    """Volume fraction of a `wall` mm sheet in `cell` mm cells."""
    t, rho = _density_table(kind)
    return np.interp(np.asarray(wall, float) / cell, t, rho)


def wall_for_density(kind: str, rho, cell: float):
    """The sheet wall (mm) that gives relative density `rho`."""
    t, r = _density_table(kind)
    return np.interp(np.asarray(rho, float), r, t) * cell


def sheet_modulus(kind: str, rho):
    s = SHEETS[kind]
    rho = np.asarray(rho, float)
    return np.where(rho >= 1.0, 1.0, s["C"] * np.clip(rho, 1e-3, 1.0) ** s["n"])


def sheet_stress_factor(kind: str, rho):
    """Peak wall stress over the macro stress the lattice carries."""
    s = SHEETS[kind]
    rho = np.asarray(rho, float)
    return np.where(rho >= 1.0, 1.0, s["A"] * np.clip(rho, 1e-3, 1.0) ** -s["m"])


def homogenize(kind: str, wall_over_cell: float, n: int = 40, nu: float = 0.35) -> dict:
    """Effective properties of a TPMS sheet by periodic homogenization of one
    unit cell on an n^3 voxel grid (only the solid voxels; their nodes wrap
    across the cell faces). Returns relative density, uniaxial modulus and
    Poisson ratio, shear modulus (all over the solid's E), the Zener
    anisotropy ratio, and the peak wall stress factor under a uniaxial and a
    shear macro stress (99.5th percentile and max of nodal von Mises over
    the macro von Mises). Slow (tens of seconds at n = 40): the fits in
    SHEETS come from it."""
    import pyamg
    import scipy.sparse as sp
    from scipy.sparse.linalg import cg

    from .fem import CORNERS, _b_matrix, elastic_c, hex8_ke, von_mises

    h = 1.0 / n
    c = (np.arange(n) + 0.5) * h
    g = np.stack(np.meshgrid(c, c, c, indexing="ij"), -1).reshape(-1, 3)
    solid = np.abs(tpms(kind, g, 1.0)) < 0.5 * wall_over_cell
    eijk = np.argwhere(solid.reshape(n, n, n))
    ne = len(eijk)
    corner = (eijk[:, None, :] + CORNERS[None]) % n
    used, inv = np.unique((corner[..., 0] * n + corner[..., 1]) * n + corner[..., 2], return_inverse=True)
    enode = inv.reshape(ne, 8)
    nn = len(used)
    edof = (3 * enode[:, :, None] + np.arange(3)).reshape(ne, 24)
    ke = hex8_ke(nu) * h
    k = sp.csr_matrix((np.tile(ke.ravel(), ne), (np.repeat(edof, 24, axis=1).ravel(),
                                                 np.tile(edof, (1, 24)).ravel())), shape=(3 * nn, 3 * nn))
    free = np.ones(3 * nn, bool)
    free[:3] = False                     # one node holds the cell still
    kf = k[free][:, free]
    ml = pyamg.smoothed_aggregation_solver(kf, symmetry="hermitian", max_coarse=400, coarse_solver="splu")
    xc = (eijk[:, None, :] + CORNERS[None]) * h
    c0 = elastic_c(1.0, nu)
    bc = _b_matrix(0, 0, 0) / h

    def load(e):
        # u = macro strain . x + periodic fluctuation
        eps = np.array([[e[0], e[3] / 2, e[5] / 2], [e[3] / 2, e[1], e[4] / 2], [e[5] / 2, e[4] / 2, e[2]]])
        u0 = (xc @ eps.T).reshape(ne, 24)
        f = np.zeros(3 * nn)
        np.add.at(f, edof, -(u0 @ ke.T))
        uf, info = cg(kf, f[free], rtol=1e-8, maxiter=2000, M=ml.aspreconditioner())
        if info != 0:
            raise RuntimeError("unit cell solve did not converge")
        u = np.zeros(3 * nn)
        u[free] = uf
        ue = u[edof] + u0
        mean = ((ue @ bc.T) @ c0.T).sum(axis=0) / n ** 3
        acc = np.zeros((nn, 6))
        for kk, (a, b, cc) in enumerate(2.0 * CORNERS - 1.0):
            np.add.at(acc, enode[:, kk], (ue @ (_b_matrix(a, b, cc) / h).T) @ c0.T)
        vm = von_mises(acc / np.bincount(enode.ravel(), minlength=nn)[:, None])
        return mean, vm

    s1, _ = load([1, 0, 0, 0, 0, 0])
    c11, c12 = s1[0], s1[1]
    c44 = load([0, 0, 0, 1, 0, 0])[0][3]
    e_uni = (c11 - c12) * (c11 + 2 * c12) / (c11 + c12)
    nu_h = c12 / (c11 + c12)
    _, vm_z = load([-nu_h / e_uni, -nu_h / e_uni, 1 / e_uni, 0, 0, 0])
    _, vm_s = load([0, 0, 0, 1 / c44, 0, 0])
    return {"rho": float(solid.mean()), "E": float(e_uni), "nu": float(nu_h), "G": float(c44),
            "zener": float(2 * c44 / (c11 - c12)),
            "stressFactor": float(np.percentile(vm_z, 99.5)), "stressFactorMax": float(vm_z.max()),
            "shearStressFactor": float(np.percentile(vm_s, 99.5) / np.sqrt(3.0))}


@dataclass
class LatticeOptions:
    kind: str = "gyroid"
    cell: float = 8.0               # mm per period
    skin: float = 1.2               # mm of solid shell; 0 leaves the lattice open
    wall_min: float = 0.8           # mm, where the part idles (>= the process's min feature)
    wall_max: float = 2.0           # mm; a cell that needs more turns solid
    solid_above: float = 0.5        # (solid-part grading) utilization above which material stays solid
    safety_factor: float = 2.0      # (solid-part grading) the allowable is strength / this
    target_sf: float | None = None  # homogenized sizing to this safety factor (every load case)
    collar: float = 2.0             # mm of solid round preserved shapes, keep-outs, supports and loads
    blend: float | None = None      # mm, smooth-union radius between solid and lattice (default wall_max)
    drains: str = "auto"            # "auto" (powder and resin processes), "yes", "no"
    drain_diameter: float | None = None   # mm (default 4 for powder, 3 for resin)
    fill_pockets: float | None = None     # mm^3: enclosed voids smaller than this are filled (default cell^3/8)
    iterations: int = 12            # sizing passes
    analysis_cells: int = 120_000   # homogenized FEA size
    step: float | None = None       # field sampling, mm (default: wall_min / 2.5)
    max_points: int = 12_000_000

    @classmethod
    def from_spec(cls, spec: dict, process=None, safety_factor: float | None = None) -> "LatticeOptions":
        """Options from the JSON spec (schema.LATTICE_SCHEMA). `safety_factor`
        is the study goal's, used when the spec names none."""
        spec = dict(spec or {})
        o = cls()
        for key, attr in (("type", "kind"), ("cell", "cell"), ("skin", "skin"), ("minWall", "wall_min"),
                          ("maxWall", "wall_max"), ("solidAbove", "solid_above"), ("collar", "collar"),
                          ("blend", "blend"), ("drains", "drains"), ("drainDiameter", "drain_diameter"),
                          ("fillPockets", "fill_pockets"), ("step", "step")):
            if key in spec:
                setattr(o, attr, spec[key] if attr in ("kind", "drains") else float(spec[key]))
        sizing = spec.get("sizing", "homogenized")
        if sizing not in ("homogenized", "solid-stress"):
            raise ValueError("lattice sizing must be 'homogenized' or 'solid-stress'")
        sf = float(spec.get("safetyFactor") or safety_factor or 2.0)
        if sizing == "homogenized":
            o.target_sf = sf
        else:
            o.safety_factor = sf
        if process is not None and o.wall_min < process.min_feature:
            o.wall_min = process.min_feature
        if o.wall_max < o.wall_min:
            raise ValueError("lattice maxWall must be at least minWall")
        if o.cell < 4.0 * o.wall_max:
            raise ValueError(f"lattice cell {o.cell:g}mm is too small for {o.wall_max:g}mm walls "
                             "(at least 4 x maxWall, or the channels close)")
        if o.drains not in ("auto", "yes", "no"):
            raise ValueError("lattice drains must be 'auto', 'yes' or 'no'")
        return o


def envelope_mesh(study: Study):
    """The whole envelope as a solid: design space and preserved shapes, less
    the keep-outs. A lattice part that keeps its outer shape starts here."""
    import manifold3d as m3d

    from .shapes import trimesh_from_manifold

    add = m3d.Manifold.batch_boolean([s.to_manifold() for s in study.design_space]
                                     + [r.shape.to_manifold() for r in study.preserve], m3d.OpType.Add)
    if study.keep_out:
        add = add - m3d.Manifold.batch_boolean([r.shape.to_manifold() for r in study.keep_out], m3d.OpType.Add)
    return trimesh_from_manifold(add)


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
    return _signed_distance(solid, step)


def _signed_distance(solid, step):
    d = (ndimage.distance_transform_edt(~solid) - ndimage.distance_transform_edt(solid)) * step
    return d + np.where(solid, 0.5 * step, -0.5 * step)


def _collar_distance(study: Study, pts, collar):
    """Distance (negative inside) to the solid zones round everything the
    user drew: preserved shapes, keep-outs (a bore's wall), and where loads
    and supports attach. Nothing there should be lattice."""
    shapes = [r.shape for r in study.preserve] + [r.shape for r in study.keep_out]
    for s in study.supports:
        shapes.append(s.shape if s.shape is not None else study.region_shape(s.region)[0])
    for case in study.load_cases:
        for ld in case.loads:
            shapes.append(ld.shape if ld.shape is not None else study.region_shape(ld.region)[0])
    d = np.full(len(pts), np.inf)
    seen = set()
    for s in shapes:
        if id(s) in seen:
            continue
        seen.add(id(s))
        d = np.minimum(d, s.sdf(pts) - collar)
    return d


def _block_mean(a, r):
    nx, ny, nz = (n // r for n in a.shape)
    return a[:nx * r, :ny * r, :nz * r].reshape(nx, r, ny, r, nz, r).mean(axis=(1, 3, 5))


def _size(study: Study, o: LatticeOptions, lo, step, inside, solid_zone):
    """Homogenized fully-stressed sizing on a grid r fine cells to the side.
    Returns (rho per analysis cell, 1 = solid; the strain-equivalent stress
    at full modulus, MPa; r; info)."""
    from scipy.spatial import cKDTree

    from .domain import boundary_conditions, gravity_force
    from .fem import HexMesh, Solver, Stiffness, von_mises

    mat = study.material
    kind = o.kind
    vol = inside.sum() * step ** 3
    pitch = max(o.cell / 5.0, (vol / o.analysis_cells) ** (1.0 / 3.0))
    r = max(1, int(round(pitch / step)))
    pitch = r * step
    f_in = _block_mean(inside.astype(np.float32), r)
    f_solid = _block_mean((inside & solid_zone).astype(np.float32), r)
    active = f_in >= 0.5
    lbl, _ = ndimage.label(active)
    ijk = np.argwhere(active)
    hm = HexMesh(ijk, pitch, lo)
    bc = boundary_conditions(study, hm, check_loads=False, tol_floor=0.5 * pitch)
    sup = np.unique(lbl[tuple(ijk[bc.support_nodes[hm.enode].any(axis=1)].T)])
    keep = np.isin(lbl[tuple(ijk.T)], sup)
    if not keep.all():
        ijk = ijk[keep]
        hm = HexMesh(ijk, pitch, lo)
        bc = boundary_conditions(study, hm, check_loads=False, tol_floor=0.5 * pitch)
    fs = (f_solid[tuple(ijk.T)] / f_in[tuple(ijk.T)]).astype(float)
    fc = 1.0 - fs
    lattice_cell = fc > 0.05
    rho_min = float(relative_density(kind, o.wall_min, o.cell))
    rho_max = float(relative_density(kind, o.wall_max, o.cell))
    rho = np.full(hm.n_elem, rho_min)
    rho[~lattice_cell] = 1.0
    stiff = Stiffness(hm, mat.nu, bc.fixed)
    solver = Solver(hm, rtol=1e-6, maxiter=3000)
    target = o.target_sf * 1.15             # verify's discretization allowance
    allow = mat.strength / target
    contact = bc.support_nodes.copy()
    for ln in bc.case_load_nodes:
        contact |= ln
    band = 1.5 * pitch + o.collar
    near = cKDTree(hm.node_xyz[contact]).query(hm.elem_centers(), k=1, distance_upper_bound=band)[0] < np.inf
    # a stress concentration spreads over a cell before the lattice can
    # follow it: grow each requirement over a quarter cell
    grow = max(1, int(round(0.25 * o.cell / pitch)))
    grid = np.zeros(active.shape)
    history = []
    analyzed = rho
    for it in range(o.iterations):
        e_cell = mat.E * (fs + fc * sheet_modulus(kind, rho))
        k = stiff.assemble(e_cell)
        if bc.springs is not None:
            k = (k + mat.E * bc.springs).tocsr()
        fg = gravity_force(hm, np.ones(hm.n_elem), bc.gravity, bc.fixed) if bc.gravity is not None else 0.0
        us = solver.solve_many(k, [bc.loads[ci] + fg for ci in range(len(study.load_cases))])
        # stress at the solid's modulus for each cell's strain: what the
        # skin carries, and E*/E of it is the lattice's macro stress
        se = np.max([von_mises(stiff.stress(u, mat.E)) for u in us], axis=0)
        sm = se * sheet_modulus(kind, rho)
        # a solid cell is judged as the densest lattice under the same
        # strain: whether it could go back to lattice (with some hysteresis,
        # or cells flicker), not what it needs as solid, which would spread
        # solid outwards a ring per pass
        s = SHEETS[kind]
        sm_eval = se * sheet_modulus(kind, np.minimum(rho, rho_max))
        need = np.where(lattice_cell, (s["A"] * sm_eval / allow) ** (1.0 / s["m"]), 0.0)
        grid[:] = 0.0
        grid[tuple(ijk.T)] = need
        need = ndimage.maximum_filter(grid, size=2 * grow + 1)[tuple(ijk.T)]
        new = np.clip(np.maximum(need, 0.5 * np.minimum(rho, rho_max)), rho_min, None)    # move limit
        stay = (rho >= 1.0) & (need > 0.8 * rho_max)
        new = np.where((new > rho_max) | stay, 1.0, np.minimum(new, rho_max))
        new[~lattice_cell] = 1.0
        wall_u = np.where(lattice_cell & (rho < 1.0), sheet_stress_factor(kind, rho) * sm, 0.0)
        solid_u = np.where(~lattice_cell | (rho >= 1.0), se, 0.0)
        peak = float(np.max(np.where(near, 0.0, np.maximum(wall_u, solid_u))))
        frac = float((fs + fc * rho).mean())
        history.append({"iteration": it, "predictedSF": mat.strength / (1.15 * peak) if peak > 0 else np.inf,
                        "solidCells": float((rho >= 1).mean()), "fill": frac,
                        "compliance": float(sum((bc.loads[ci] + fg) @ u for ci, u in enumerate(us)))})
        # the sizing assumes each cell's macro stress stays put as it
        # changes; when the analysis keeps reading under the target (the
        # load moved onto what got stiffer), tighten the allowable to match
        short = history[-1]["predictedSF"] / o.target_sf
        if it >= 2:
            allow *= float(np.clip(short, 0.85, 1.15)) ** 0.5
        if np.allclose(new, rho, atol=0.01) and abs(short - 1.0) < 0.05:
            break
        analyzed, rho = rho, new
    else:
        rho = analyzed           # report the field that was analyzed
    rho_grid = np.zeros(active.shape)
    rho_grid[tuple(ijk.T)] = rho
    se_grid = np.zeros(active.shape)
    se_grid[tuple(ijk.T)] = se
    mask = np.zeros(active.shape, bool)
    mask[tuple(ijk.T)] = True
    info = {"analysisPitch": pitch, "analysisCells": int(hm.n_elem), "history": history,
            "predictedSF": history[-1]["predictedSF"],
            "solidCellFraction": float((rho[lattice_cell] >= 1).mean()) if lattice_cell.any() else 0.0}
    return rho_grid, se_grid, mask, r, info


def _to_fine(coarse, mask, r, dims):
    """Trilinear upsampling of a per-analysis-cell field to the fine grid,
    the cells outside the analysis filled from their nearest neighbour."""
    idx = ndimage.distance_transform_edt(~mask, return_distances=False, return_indices=True)
    filled = coarse[tuple(idx)]
    coords = [(np.arange(n) + 0.5) / r - 0.5 for n in dims]
    out = np.empty(dims, np.float32)
    gy, gz = np.meshgrid(coords[1], coords[2], indexing="ij")
    for i, x in enumerate(coords[0]):    # a slab at a time keeps the coordinates small
        out[i] = ndimage.map_coordinates(filled, [np.full_like(gy, x), gy, gz], order=1, mode="nearest")
    return out


def _capsule(pts, a, b, radius):
    ab = b - a
    t = np.clip(((pts - a) @ ab) / (ab @ ab), 0.0, 1.0)
    return np.linalg.norm(pts - (a + t[:, None] * ab), axis=1) - radius


def _local(lo, step, shape, a, b, reach):
    """The grid block round segment a-b out to `reach`, and its points."""
    box_lo = np.floor((np.minimum(a, b) - reach - lo) / step - 1).astype(int).clip(0)
    box_hi = np.ceil((np.maximum(a, b) + reach - lo) / step + 1).astype(int).clip(None, np.array(shape))
    bs = tuple(slice(x0, x1) for x0, x1 in zip(box_lo, box_hi))
    ii = np.stack(np.meshgrid(*[np.arange(x0, x1) for x0, x1 in zip(box_lo, box_hi)], indexing="ij"), -1)
    return bs, lo + (ii.reshape(-1, 3) + 0.5) * step, ii.shape[:3]


def _drains(o, d, dp, d_collar, stress, lo, step, process):
    """Pockets too small to drain (a mask, to fill solid), and drain holes
    from every other enclosed void to the outside: (a, b, radius, report)
    per hole, a running from inside the void out through the skin.

    Holes go where the part is calmest, just under the skin on a flat face
    clear of everything the user drew, two per void well apart so powder
    can be blown and resin rinsed through. Each gets a solid boss."""
    material = d < 0.0
    lbl, n = ndimage.label(~material)
    open_ids = set(np.unique(lbl[dp > 0.0]).tolist())
    vox = step ** 3
    sizes = ndimage.sum_labels(np.ones_like(lbl), lbl, index=np.arange(1, n + 1))
    fill = o.fill_pockets if o.fill_pockets is not None else o.cell ** 3 / 8.0
    trapped = [i for i in range(1, n + 1) if i not in open_ids]
    small = [i for i in trapped if sizes[i - 1] * vox < fill]
    pockets = np.isin(lbl, small) if small else None
    big = [i for i in trapped if i not in small]
    holes = []
    if not big or o.drains == "no" or (o.drains == "auto" and (process is None or process.kind == "fdm")):
        return pockets, holes
    if o.drain_diameter is not None:
        dia = o.drain_diameter
    else:
        dia = 3.0 if process is not None and process.kind in ("sla", "volumetric") else 4.0
    rad = 0.5 * dia
    axis, sign = process.build if process is not None and process.build is not None else (2, 1)
    objs = ndimage.find_objects(lbl)
    for i in big:
        sl = objs[i - 1]
        comp = lbl[sl] == i
        depth = -dp[sl]
        cand = comp & (depth < o.skin + 3.0 * step) & (d_collar[sl] > rad + o.collar + 1.0)
        if not cand.any():
            cand = comp & (depth <= depth[comp].min() + step)
        idx = np.argwhere(cand)
        sv = stress[sl][cand]
        off = np.asarray([s_.start for s_ in sl])
        # a hole is a stress raiser (about 3x in a plate): the calmest flat
        # site first, then the calmest one well away from it, so air or
        # rinse can flow through; on a tie, lower along the build drains
        order = np.lexsort((idx[:, axis] * sign, np.round(sv, 1)))
        reach = rad + max(o.collar, o.wall_max)
        extent = float(np.linalg.norm((idx.max(axis=0) - idx.min(axis=0)) * step))
        picks = []
        for j in order[:2000]:
            pt0 = lo + (off + idx[j] + 0.5) * step
            if picks and np.linalg.norm(pt0 - picks[0][0]) < max(4.0 * dia, 0.5 * extent):
                continue
            site = _drain_site(dp, tuple(off + idx[j]), lo, step, reach)
            if site is not None:
                picks.append(site)
                if len(picks) == 2:
                    break
        for pt, nrm, g in picks:
            b = pt + nrm * (-dp[g] + 1.5)
            holes.append((pt, b, rad, {"at": (pt + nrm * -dp[g]).tolist(), "normal": nrm.tolist(),
                                      "diameter": dia, "stress": float(stress[g])}))
    return pockets, holes


def _drain_site(dp, g, lo, step, reach):
    """The site's point and outward normal, if the surface over it is flat
    out to `reach`: a hole on an edge or a corner loses half its boss to
    the edge and is where the stress concentrates."""
    nb = tuple(slice(max(c - 2, 0), c + 3) for c in g)
    gr = np.array([np.gradient(dp[nb], step, axis=k).mean() for k in range(3)])
    nrm = gr / max(np.linalg.norm(gr), 1e-9)
    pt = lo + (np.asarray(g) + 0.5) * step
    surf = pt + nrm * -dp[g]
    u = np.cross(nrm, [1.0, 0.0, 0.0] if abs(nrm[0]) < 0.9 else [0.0, 1.0, 0.0])
    u /= np.linalg.norm(u)
    v = np.cross(nrm, u)
    ang = np.linspace(0.0, 2.0 * np.pi, 12, endpoint=False)
    ring = surf + reach * (np.cos(ang)[:, None] * u + np.sin(ang)[:, None] * v)
    ijk = np.floor((ring - lo) / step).astype(int)
    if (ijk < 0).any() or (ijk >= np.array(dp.shape)).any():
        return None
    if np.abs(dp[tuple(ijk.T)]).max() > 1.5 * step:
        return None
    return pt, nrm, g


def _trapped(d, dp, step):
    lbl, n = ndimage.label(d >= 0.0)
    open_ids = np.unique(lbl[dp > 0.0])
    tr = (lbl > 0) & ~np.isin(lbl, open_ids)
    return float(tr.sum() * step ** 3), int(len(np.setdiff1d(np.arange(1, n + 1), open_ids)))


def lattice_infill(study: Study, mesh, fields: dict | None = None, opts: LatticeOptions | None = None):
    """Returns (trimesh.Trimesh, info). With `opts.target_sf` the wall is
    sized by the homogenized analysis (see the module docstring); otherwise
    `fields` is verify_part(..., fields=True)["fields"] for the solid part,
    and without it the lattice is uniform at wall_min. `mesh` is the solid
    to fill: a topology-optimized part, or `envelope_mesh(study)` to keep
    the envelope's outer shape."""
    import manifold3d as m3d
    import trimesh
    from skimage.measure import marching_cubes

    from .reconstruct import _smooth_union
    from .shapes import manifold_from_trimesh, trimesh_from_manifold

    o = opts or LatticeOptions()
    process = study.process
    lo, hi = mesh.bounds
    lo = lo - 2.0 * o.wall_max
    hi = hi + 2.0 * o.wall_max
    step = o.step or o.wall_min / 2.5
    step = max(step, (float(np.prod(hi - lo)) / o.max_points) ** (1.0 / 3.0))
    dims = tuple(np.maximum(np.ceil((hi - lo) / step).astype(int), 2))
    d_part = _part_distance(mesh, step, lo, dims)
    inside = d_part < 0

    idx = np.stack(np.meshgrid(*[np.arange(n) for n in dims], indexing="ij"), -1).reshape(-1, 3)
    pts = lo + (idx + 0.5) * step
    del idx
    flat_in = inside.ravel()
    d_collar = np.full(len(pts), np.inf)
    if o.collar > 0:
        d_collar[flat_in] = _collar_distance(study, pts[flat_in], o.collar)
    d_collar = d_collar.reshape(dims)
    sizing = None
    stress = np.zeros(dims, np.float32)
    if o.target_sf:
        zone = (d_part > -o.skin) | (d_collar < 0)
        rho_c, se_c, mask, r, sizing = _size(study, o, lo, step, inside, zone)
        rho = _to_fine(rho_c, mask, r, dims)
        stress = _to_fine(se_c, mask, r, dims)
        rho_max = float(relative_density(o.kind, o.wall_max, o.cell))
        wall = np.clip(wall_for_density(o.kind, np.minimum(rho, rho_max), o.cell), o.wall_min, o.wall_max)
        d_cells = _signed_distance(rho > 0.5 * (1.0 + rho_max), step)
        util = None
    else:
        util = np.zeros(len(pts))
        if fields is not None:
            from scipy.spatial import cKDTree

            vm = np.max([c["vonMises"] for c in fields["cases"]], axis=0)
            allow = study.material.strength / o.safety_factor
            tree = cKDTree(fields["nodes"])
            dist, near = tree.query(pts[flat_in], k=4)
            w = 1.0 / np.maximum(dist, 1e-6) ** 2
            util[flat_in] = (w * vm[near]).sum(axis=1) / w.sum(axis=1) / allow
        # wall from utilization, square-root graded so moderate stress already
        # thickens it; solid above solid_above
        u = np.clip(util / o.solid_above, 0.0, 1.0)
        wall = (o.wall_min + (o.wall_max - o.wall_min) * np.sqrt(u)).reshape(dims)
        d_cells = ((o.solid_above - util) * o.cell).reshape(dims)
        stress = (util * study.material.strength / o.safety_factor).reshape(dims)
    # marching cubes reads a 2.5-sample wall about 5% thin: a hair more
    # keeps the thinnest wall at the process minimum
    d_lat = (np.abs(tpms(o.kind, pts, o.cell)) - 0.5 * wall.ravel() - 0.15 * step).reshape(dims)
    del pts
    d_solid = np.minimum(d_cells, d_collar)
    if o.skin > 0:
        d_solid = np.minimum(d_solid, -d_part - o.skin)
    blend = o.blend or o.wall_max
    d = np.maximum(d_part, _smooth_union(d_lat, d_solid, blend))
    del d_cells

    # a boss can wall off a scrap of channel next to it: a second round
    # fills or drains whatever the first left
    filled = 0.0
    all_holes = []
    boss = max(o.collar, o.wall_max)
    for _ in range(3):
        pockets, holes = _drains(o, d, d_part, d_collar, stress, lo, step, process)
        if pockets is not None:
            d = np.where(pockets, -0.5 * step, d)
            filled += float(pockets.sum() * step ** 3)
        if not holes:
            break
        for a, b, rad, _ in holes:
            bs, hp, shp = _local(lo, step, d.shape, a, b, rad + boss + blend)
            cap = _capsule(hp, a, b, rad).reshape(shp)
            ds = np.minimum(d_solid[bs], cap - boss)
            d[bs] = np.minimum(d[bs], np.maximum(d_part[bs], _smooth_union(d_lat[bs], ds, blend)))
            d_solid[bs] = ds
        for a, b, rad, _ in all_holes + holes:
            # through the boss's rounded end and on into the void
            a = a - (b - a) / np.linalg.norm(b - a) * (boss + rad + blend)
            bs, hp, shp = _local(lo, step, d.shape, a, b, rad)
            d[bs] = np.maximum(d[bs], -_capsule(hp, a, b, rad).reshape(shp))
        all_holes += holes
    holes = all_holes
    holes = [h[3] for h in holes]
    del d_lat, d_solid
    trapped_vol, trapped_n = _trapped(d, d_part, step)

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
    info = {"kind": o.kind, "cell": o.cell, "skin": o.skin, "wallMin": o.wall_min, "wallMax": o.wall_max,
            "step": step, "volume": vol, "solidVolume": solid_vol, "volumeRatio": vol / solid_vol,
            "mass": vol * study.material.density / 1000.0, "graded": fields is not None or bool(o.target_sf),
            "sizing": sizing, "droppedBodies": len(loose),
            "solidFraction": float((util[flat_in] >= o.solid_above).mean()) if util is not None and flat_in.any()
            else (sizing or {}).get("solidCellFraction", 0.0)}
    info["printability"] = _printability(out, d_part, lo, step, o, process, holes, filled, trapped_vol, trapped_n)
    # what stays walled in, where powder or resin would be trapped
    info["enclosedVoid"] = trapped_vol
    return out, info


def _printability(mesh, dp, lo, step, o, process, holes, filled, trapped_vol, trapped_n):
    from .verify import surface_thickness

    p1, p5 = surface_thickness(mesh, samples=4000)
    min_feature = process.min_feature if process is not None else 0.8
    rep = {"process": process.kind if process is not None else None, "minFeature": min_feature,
           "wallP1": p1, "wallP5": p5, "wallOk": p1 is not None and p1 >= 0.9 * min_feature,
           "drainHoles": holes, "filledPocketVolume": filled,
           "trappedVolume": trapped_vol, "trappedVoids": trapped_n}
    if process is None or process.kind != "fdm":
        rep["drainOk"] = trapped_vol < 1.0
    # the lattice's own ceilings and the skin's over it (inside the part): FDM
    # bridges these, so report how much faces down past the 45 degree limit
    if process is not None and process.build is not None:
        axis, sign = process.build
        up = np.zeros(3)
        up[axis] = sign
        c = mesh.triangles_center
        ijk = np.clip(((c - lo) / step).astype(int), 0, np.array(dp.shape) - 1)
        interior = dp[tuple(ijk.T)] < -(0.5 * o.skin + step)
        down = -(mesh.face_normals @ up) > np.cos(np.radians(40.0))
        a = mesh.area_faces
        rep["internalOverhangFraction"] = float(a[interior & down].sum() / max(a[interior].sum(), 1e-9))
    return rep
