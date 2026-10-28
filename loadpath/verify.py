"""
Check the part we actually deliver, not the density field we optimized.

The final mesh differs from the optimizer's field (smoothing, the exact
booleans, dropped fragments), so verification re-voxelizes the mesh,
runs a fresh linear static analysis at the material's real modulus for every
load case, and measures geometry with exact booleans:

- mass, and how much lighter than the solid design space it is;
- peak von Mises stress per load case and the safety factor against the
  material's strength;
- deflection at the loads;
- keep-out intrusion and preserve loss (mm^3, from booleans: both should be
  zero to rounding);
- body count and thinnest member.

Stress next to where loads and supports are applied is excluded from the
peak. A load spread over a few nodes is a modeling idealization, and the
stress it produces there is a property of the idealization, not of the
part; every FEA package has the same caveat. Everything else counts.
"""
from __future__ import annotations

import time

import numpy as np
from scipy import ndimage

from .domain import boundary_conditions, gravity_force
from .fem import HexMesh, Solver, Stiffness, von_mises
from .spec import Study

__all__ = ["voxelize", "verify_part"]

STRESS_ALLOWANCE = 1.15


def voxelize(mesh, pitch: float, box=None):
    """Solid cells of a closed mesh by ray parity along +z through each
    column of cell centers. Returns (bool grid, origin = min corner of cell
    (0,0,0)). `box` = (lo, hi) voxelizes only that region (the rays still
    start below the whole mesh, so parity is right)."""
    mlo = np.asarray(mesh.bounds, float)[0]
    lo, hi = (np.asarray(b, float) for b in box) if box is not None else np.asarray(mesh.bounds, float)
    dims = np.maximum(np.ceil((hi - lo) / pitch).astype(int), 1)
    origin = lo - 0.5 * (dims * pitch - (hi - lo))
    xs = origin[0] + (np.arange(dims[0]) + 0.5) * pitch
    ys = origin[1] + (np.arange(dims[1]) + 0.5) * pitch
    gx, gy = np.meshgrid(xs, ys, indexing="ij")
    # A hair of jitter keeps rays off exact edges and vertices, where a hit
    # could be reported twice and flip the parity.
    jit = pitch * 1e-4
    ray_o = np.stack([gx.ravel() + 0.37 * jit, gy.ravel() + 0.61 * jit,
                      np.full(gx.size, mlo[2] - pitch)], axis=1)
    ray_d = np.tile([0.0, 0.0, 1.0], (len(ray_o), 1))
    locs, ray_idx, _ = mesh.ray.intersects_location(ray_o, ray_d, multiple_hits=True)
    zc = origin[2] + (np.arange(dims[2]) + 0.5) * pitch
    solid = np.zeros(dims, bool)
    order = np.lexsort((locs[:, 2], ray_idx))
    ray_idx, zs = ray_idx[order], locs[order, 2]
    starts = np.searchsorted(ray_idx, np.arange(len(ray_o)))
    ends = np.searchsorted(ray_idx, np.arange(len(ray_o)), side="right")
    for r in np.nonzero(ends > starts)[0]:
        z = zs[starts[r]:ends[r]]
        # Merge near-duplicate hits (shared edges) before pairing.
        if len(z) > 1:
            z = z[np.concatenate([[True], np.diff(z) > 1e-7])]
        if len(z) % 2:
            z = z[:-1]
        inside = np.zeros(dims[2], bool)
        for a, b in z.reshape(-1, 2):
            inside |= (zc > a) & (zc < b)
        i, j = np.unravel_index(r, (dims[0], dims[1]))
        solid[i, j] = inside
    return solid, origin


def _thickness(solid, pitch):
    """Member thickness on the medial ridge: twice the distance to void at
    cells that are local maxima of that distance."""
    edt = ndimage.distance_transform_edt(np.pad(solid, 1))[1:-1, 1:-1, 1:-1]
    ridge = solid & (edt >= ndimage.maximum_filter(edt, size=3)) & (edt > 0)
    # A ridge cell's distance to void is ceil(n/2) for a member n cells
    # thick, so 2*edt over-reads odd and 2*edt-1 under-reads even
    # thicknesses by a whole cell; 2*edt-0.5 is within half a cell of both.
    vals = (2.0 * edt[ridge] - 0.5) * pitch
    if vals.size == 0:
        return None, None
    return float(vals.min()), float(np.percentile(vals, 5))


def nodal_von_mises(stiff, u, modulus):
    """Von Mises at the nodes: each element's stress evaluated at its own
    corners, averaged over the elements sharing a node. Element-centre stress
    sits half a cell inside the surface, which is exactly where stress
    concentrates; on a plate with a hole (Peterson Kt_net = 2.43, peak 81 MPa)
    centre values read 64-74 MPa and neighbor-smoothed centres 59-68, while
    nodal averages read 73 at 7 cells per hole radius and 81.7 from 9 cells
    up, converged. Averaging across elements is also what keeps a single
    voxel staircase corner from reading as a spike."""
    from .fem import CORNERS, _b_matrix, elastic_c

    mesh = stiff.mesh
    c = elastic_c(1.0, stiff.nu) * modulus
    ue = u[mesh.edof]
    acc = np.zeros((mesh.n_nodes, 6))
    for k, (a, b, cc) in enumerate(2.0 * CORNERS - 1.0):
        sk = (ue @ (_b_matrix(a, b, cc) / mesh.pitch).T) @ c.T
        np.add.at(acc, mesh.enode[:, k], sk)
    cnt = np.bincount(mesh.enode.ravel(), minlength=mesh.n_nodes)
    return von_mises(acc / cnt[:, None])


def _overhang(mesh, direction, study=None, max_angle_deg=45.0, tolerance_deg=5.0):
    """Down-facing surface that needs support: faces whose normal is within
    (max_angle - tolerance) of straight down, i.e. ceilings and slopes
    shallower than the self-supporting limit, excluding the face on the build
    plate. The tolerance keeps marching-cubes facets on a true 45-degree
    slope from counting. `overhangArea` is everything; `designOverhangArea`
    leaves out faces on preserved or keep-out geometry, which the user drew
    (a horizontal bore's ceiling is theirs, not the optimizer's)."""
    axis, sign = direction
    up = np.zeros(3)
    up[axis] = sign
    down = -(mesh.face_normals @ up)
    centers = mesh.triangles_center
    height = centers @ up
    bad = (down > np.cos(np.radians(max_angle_deg - tolerance_deg))) & (height > height.min() + 0.5)
    area = float(mesh.area_faces[bad].sum())
    out = {"overhangArea": area, "overhangFraction": area / float(mesh.area)}
    if study is not None:
        drawn = np.zeros(len(centers), bool)
        for r in study.preserve + study.keep_out:
            drawn |= np.abs(r.shape.sdf(centers)) < 0.6
        d_area = float(mesh.area_faces[bad & ~drawn].sum())
        out["designOverhangArea"] = d_area
        out["designOverhangFraction"] = d_area / float(mesh.area)
    return out


def _union(shapes):
    import manifold3d as m3d
    mans = [s.to_manifold() for s in shapes]
    if not mans:
        return None
    return m3d.Manifold.batch_boolean(mans, m3d.OpType.Add) if len(mans) > 1 else mans[0]


def _hotspots(xyz, vm, body, peak, radius, count=3):
    """Up to `count` local stress maxima at least 2*radius apart, among nodes
    within half of the peak."""
    order = np.argsort(-np.where(body, vm, -1.0))
    picked = []
    for i in order:
        if not body[i] or vm[i] < 0.5 * peak or len(picked) >= count:
            break
        if all(np.linalg.norm(xyz[i] - xyz[j]) >= 2 * radius for j in picked):
            picked.append(i)
    return [xyz[i] for i in picked]


def _submodel_peak(study, mesh, hm, u, center, half, pitch, mat, contact_xyz, band):
    """Re-solve a small box around a global hotspot at a finer pitch, driven by
    the global displacement field on the box's cut faces (classic FE
    submodeling). Voxel stress converges only with about 8 cells across a
    feature's radius; the global grid rarely affords that, the box does.
    Returns (peak MPa, position) or None."""
    from scipy.spatial import cKDTree

    lo, hi = center - half, center + half
    solid, origin = voxelize(mesh, pitch, box=(lo, hi))
    if solid.sum() < 64:
        return None
    sm = HexMesh(np.argwhere(solid), pitch, origin)
    blo = origin
    bhi = origin + np.asarray(solid.shape) * pitch
    eps = 0.25 * pitch
    cut = np.any((sm.node_xyz < blo + eps) | (sm.node_xyz > bhi - eps), axis=1)
    near_contact = np.zeros(sm.n_nodes, bool)
    if len(contact_xyz):
        near_contact = cKDTree(contact_xyz).query(sm.node_xyz, k=1, distance_upper_bound=band)[0] < np.inf
    driven = cut | near_contact
    if driven.all() or not driven.any():
        return None
    # Global displacement at the driven nodes: inverse-distance weights over
    # the nearest global nodes (the field is smooth at this scale).
    gd, gi = cKDTree(hm.node_xyz).query(sm.node_xyz[driven], k=8)
    w = 1.0 / np.maximum(gd, 1e-9) ** 2
    w /= w.sum(axis=1, keepdims=True)
    ug = u.reshape(-1, 3)[gi]
    up = (w[..., None] * ug).sum(axis=1).ravel()
    pd = np.repeat(driven, 3)
    st = Stiffness(sm, mat.nu, np.zeros(sm.n_dof, bool))
    k = st.assemble(np.full(sm.n_elem, mat.E)).tocsr()
    free = ~pd
    kff = k[free][:, free]
    rhs = -(k[free][:, pd] @ up)
    solver = Solver(sm, rtol=1e-7, maxiter=4000)
    # Rigid-body modes are pinned by the driven nodes; AMG still wants a
    # near-null space sized to the free unknowns, so use plain CG+AMG on kff.
    import pyamg
    from scipy.sparse.linalg import cg
    ml = pyamg.smoothed_aggregation_solver(kff, symmetry="hermitian", strength="symmetric",
                                           max_coarse=400, coarse_solver="splu")
    uf, info = cg(kff, rhs, rtol=1e-7, maxiter=4000, M=ml.aspreconditioner(cycle="V"))
    del solver
    if info != 0:
        return None
    us = np.zeros(sm.n_dof)
    us[pd] = up
    us[free] = uf
    vm = nodal_von_mises(st, us, mat.E)
    # Trust the inside of the box only: two global cells from the cut faces.
    margin = 2.0 * hm.pitch
    inner = np.all((sm.node_xyz > blo + margin) & (sm.node_xyz < bhi - margin), axis=1) & ~near_contact
    if not inner.any():
        return None
    i = int(np.argmax(np.where(inner, vm, -1.0)))
    return float(vm[i]), sm.node_xyz[i].tolist()


def verify_part(study: Study, mesh, max_cells: int = 60_000, min_pitch: float | None = None,
                contact_band: float | None = None, refine: int = 3) -> dict:
    """Analyze a finished part (trimesh) against the study. Returns a JSON-able
    report. `contact_band` (mm) is how far from applied loads and supports
    stress is left out of the peak; it defaults to 1.5 analysis cells, and the
    study runner sets it to the optimization cell size so the excluded band
    does not shrink toward zero as the analysis grid gets finer (a clamped
    edge's stress grows without bound under refinement). `refine` is how many
    hotspots per load case get a 3x finer submodel (see _submodel_peak)."""
    from .shapes import manifold_from_trimesh

    t0 = time.time()
    mat = study.material
    part = manifold_from_trimesh(mesh)
    vol = float(part.volume())
    report = {"volume": vol, "mass": vol * mat.density / 1000.0, "material": mat.to_dict()}

    design = _union(study.design_space + [r.shape for r in study.preserve])
    keep = _union([r.shape for r in study.keep_out])
    pres = _union([r.shape for r in study.preserve])
    envelope = design - keep if keep is not None else design
    env_vol = float(envelope.volume())
    report["designSpaceMass"] = env_vol * mat.density / 1000.0
    report["massReduction"] = 1.0 - vol / env_vol if env_vol > 0 else None
    report["keepOutIntrusion"] = float((part ^ keep).volume()) if keep is not None else 0.0
    if pres is not None:
        # Preserve minus keep-out is what must survive: a boss keeps its bore.
        pres_eff = pres - keep if keep is not None else pres
        report["preserveMissing"] = float((pres_eff - part).volume())
    else:
        report["preserveMissing"] = 0.0
    report["outsideDesignSpace"] = float((part - design).volume())
    report["bodies"] = len(part.decompose())

    if study.print_direction is not None:
        report.update(_overhang(mesh, study.print_direction, study))

    pitch = (vol / max_cells) ** (1.0 / 3.0)
    if min_pitch:
        pitch = max(pitch, min_pitch)
    solid, origin = voxelize(mesh, pitch)
    report["analysisPitch"] = pitch
    report["minMember"], report["p5Member"] = _thickness(solid, pitch)

    lbl, nlab = ndimage.label(solid, structure=np.ones((3, 3, 3), bool))
    elem_ijk = np.argwhere(solid)
    hm = HexMesh(elem_ijk, pitch, origin)
    bc = boundary_conditions(study, hm, check_loads=False)
    # Material not connected to a support would float; it is reported as a
    # separate body above and left out of the stress analysis.
    supported_lbls = np.unique(lbl[tuple(elem_ijk[bc.support_nodes[hm.enode].any(axis=1)].T)])
    keep_el = np.isin(lbl[tuple(elem_ijk.T)], supported_lbls)
    if not keep_el.all():
        hm = HexMesh(elem_ijk[keep_el], pitch, origin)
        bc = boundary_conditions(study, hm, check_loads=False)
    support_nodes, fixed = bc.support_nodes, bc.fixed

    band = max(1.5 * pitch, contact_band or 0.0)
    report["contactBand"] = band
    stiff = Stiffness(hm, mat.nu, fixed)
    k = stiff.assemble(np.full(hm.n_elem, mat.E))
    if bc.springs is not None:
        k = (k + mat.E * bc.springs).tocsr()
    # 1e-6 is three-plus significant figures of stress; 1e-7 cost a third more
    # iterations. All cases share one multigrid hierarchy and run on threads.
    solver = Solver(hm, rtol=1e-6, maxiter=3000, rebuild_after_iters=10_000)
    fg = gravity_force(hm, np.ones(hm.n_elem), bc.gravity, fixed) if bc.gravity is not None else 0.0
    displacements = solver.solve_many(k, [bc.loads[ci] + fg for ci in range(len(study.load_cases))])
    from scipy.spatial import cKDTree
    cases = []
    worst_sf = np.inf
    for ci, case in enumerate(study.load_cases):
        f = bc.loads[ci] + fg
        load_nodes = bc.case_load_nodes[ci]
        u = displacements[ci]
        vm = nodal_von_mises(stiff, u, mat.E)
        # Exclude nodes within the contact band of an applied load or support.
        contact = support_nodes | load_nodes
        near = cKDTree(hm.node_xyz[contact]).query(hm.node_xyz, k=1, distance_upper_bound=band)[0] < np.inf
        body = ~near
        peak = float(vm[body].max()) if body.any() else float(vm.max())
        at = hm.node_xyz[np.argmax(np.where(body, vm, -1))].tolist()
        global_peak = peak
        refined = []
        if refine and body.any():
            half = max(8.0 * pitch, 6.0)
            for c in _hotspots(hm.node_xyz, vm, body, peak, half, count=refine):
                r = _submodel_peak(study, mesh, hm, u, c, half, pitch / 2.5, mat,
                                   hm.node_xyz[contact], band)
                if r is None:
                    continue
                refined.append({"at": r[1], "peakStress": r[0]})
                if r[0] > peak:
                    peak, at = r
        disp = u.reshape(-1, 3)
        defl = float(np.linalg.norm(disp[load_nodes].mean(axis=0)))
        # The safety factor carries a discretization allowance: on the
        # plate-with-hole benchmark (tests) refined voxel peaks land within
        # -12%..+13% of Peterson's Kt, depending on how the grid falls on the
        # hole. Under-reading stress is the unsafe direction, so the factor
        # is computed against peak * STRESS_ALLOWANCE.
        sf = mat.strength / (peak * STRESS_ALLOWANCE) if peak > 0 else np.inf
        worst_sf = min(worst_sf, sf)
        cases.append({
            "name": case.name,
            "peakStress": peak,
            "peakAt": at,
            "globalPeakStress": global_peak,
            "refined": refined,
            "p99Stress": float(np.percentile(vm[body] if body.any() else vm, 99)),
            "contactPeakStress": float(vm.max()),
            "safetyFactor": sf,
            "deflection": defl,
            "maxDisplacement": float(np.linalg.norm(disp, axis=1).max()),
            "compliance": float(f @ u),
        })
    report["cases"] = cases
    report["safetyFactor"] = worst_sf
    report["stressAllowance"] = STRESS_ALLOWANCE
    report["analysisCells"] = int(hm.n_elem)
    report["seconds"] = time.time() - t0
    return report
