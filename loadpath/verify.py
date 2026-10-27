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

from .domain import _region_nodes
from .fem import HexMesh, Solver, Stiffness, von_mises
from .spec import Study

__all__ = ["voxelize", "verify_part"]

_FACE6 = ndimage.generate_binary_structure(3, 1)


def voxelize(mesh, pitch: float):
    """Solid cells of a closed mesh by ray parity along +z through each
    column of cell centers. Returns (bool grid, origin = min corner of cell
    (0,0,0))."""
    lo, hi = np.asarray(mesh.bounds, float)
    dims = np.maximum(np.ceil((hi - lo) / pitch).astype(int), 1)
    origin = lo - 0.5 * (dims * pitch - (hi - lo))
    xs = origin[0] + (np.arange(dims[0]) + 0.5) * pitch
    ys = origin[1] + (np.arange(dims[1]) + 0.5) * pitch
    gx, gy = np.meshgrid(xs, ys, indexing="ij")
    # A hair of jitter keeps rays off exact edges and vertices, where a hit
    # could be reported twice and flip the parity.
    jit = pitch * 1e-4
    ray_o = np.stack([gx.ravel() + 0.37 * jit, gy.ravel() + 0.61 * jit,
                      np.full(gx.size, lo[2] - pitch)], axis=1)
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
    vals = (2.0 * edt[ridge] - 1.0) * pitch
    if vals.size == 0:
        return None, None
    return float(vals.min()), float(np.percentile(vals, 5))


def _smoothed_vm(vm, mesh):
    """Average each element's von Mises with its face neighbors: single-cell
    spikes at voxel staircase corners are discretization, not stress."""
    n = mesh.n_elem
    dims = mesh.node_dims - 1
    lut = np.full(tuple(dims), -1, np.int64)
    lut[tuple(mesh.elem_ijk.T)] = np.arange(n)
    acc = vm.copy()
    cnt = np.ones(n)
    for d in np.eye(3, dtype=np.int64):
        for sgn in (1, -1):
            nb = mesh.elem_ijk + sgn * d
            ok = np.all((nb >= 0) & (nb < dims), axis=1)
            j = np.full(n, -1, np.int64)
            j[ok] = lut[tuple(nb[ok].T)]
            hit = j >= 0
            acc[hit] += vm[j[hit]]
            cnt[hit] += 1
    return acc / cnt


def _union(shapes):
    import manifold3d as m3d
    mans = [s.to_manifold() for s in shapes]
    if not mans:
        return None
    return m3d.Manifold.batch_boolean(mans, m3d.OpType.Add) if len(mans) > 1 else mans[0]


def verify_part(study: Study, mesh, max_cells: int = 90_000, min_pitch: float | None = None,
                contact_band: float | None = None) -> dict:
    """Analyze a finished part (trimesh) against the study. Returns a JSON-able
    report. `contact_band` (mm) is how far from applied loads and supports
    stress is left out of the peak; it defaults to 1.5 analysis cells, and the
    study runner sets it to the optimization cell size so the excluded band
    does not shrink toward zero as the analysis grid gets finer (a clamped
    edge's stress grows without bound under refinement)."""
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

    pitch = (vol / max_cells) ** (1.0 / 3.0)
    if min_pitch:
        pitch = max(pitch, min_pitch)
    solid, origin = voxelize(mesh, pitch)
    report["analysisPitch"] = pitch
    report["minMember"], report["p5Member"] = _thickness(solid, pitch)

    lbl, nlab = ndimage.label(solid, structure=np.ones((3, 3, 3), bool))
    elem_ijk = np.argwhere(solid)
    hm = HexMesh(elem_ijk, pitch, origin)
    support_nodes = np.zeros(hm.n_nodes, bool)
    fixed = np.zeros(hm.n_dof, bool)
    for i, s in enumerate(study.supports):
        nodes = _region_nodes(study, hm, s.region, s.shape, f"supports[{i}]")
        support_nodes |= nodes
        for a in range(3):
            if s.fix[a]:
                fixed[3 * np.nonzero(nodes)[0] + a] = True
    # Material not connected to a support would float; it is reported as a
    # separate body above and left out of the stress analysis.
    supported_lbls = np.unique(lbl[tuple(elem_ijk[support_nodes[hm.enode].any(axis=1)].T)])
    keep_el = np.isin(lbl[tuple(elem_ijk.T)], supported_lbls)
    if not keep_el.all():
        hm = HexMesh(elem_ijk[keep_el], pitch, origin)
        support_nodes = np.zeros(hm.n_nodes, bool)
        fixed = np.zeros(hm.n_dof, bool)
        for i, s in enumerate(study.supports):
            nodes = _region_nodes(study, hm, s.region, s.shape, f"supports[{i}]")
            support_nodes |= nodes
            for a in range(3):
                if s.fix[a]:
                    fixed[3 * np.nonzero(nodes)[0] + a] = True

    band = max(1.5 * pitch, contact_band or 0.0)
    report["contactBand"] = band
    stiff = Stiffness(hm, mat.nu, fixed)
    k = stiff.assemble(np.full(hm.n_elem, mat.E))
    solver = Solver(hm, rtol=1e-7, maxiter=3000)
    centers = hm.elem_centers()
    cases = []
    worst_sf = np.inf
    for case in study.load_cases:
        f = np.zeros(hm.n_dof)
        load_nodes = np.zeros(hm.n_nodes, bool)
        for li, ld in enumerate(case.loads):
            nodes = _region_nodes(study, hm, ld.region, ld.shape, f"{case.name}.loads[{li}]")
            idx = np.nonzero(nodes)[0]
            load_nodes[idx] = True
            for a in range(3):
                f[3 * idx + a] += ld.force[a] / len(idx)
        f[fixed] = 0.0
        u = solver.solve(k, f)
        vm = _smoothed_vm(von_mises(stiff.stress(u, mat.E)), hm)
        # Exclude cells within the contact band of an applied load or support.
        contact = support_nodes | load_nodes
        cxyz = hm.node_xyz[contact]
        from scipy.spatial import cKDTree
        near = cKDTree(cxyz).query(centers, k=1, distance_upper_bound=band)[0] < np.inf
        body = ~near
        peak = float(vm[body].max()) if body.any() else float(vm.max())
        at = centers[np.argmax(np.where(body, vm, -1))].tolist()
        disp = u.reshape(-1, 3)
        defl = float(np.linalg.norm(disp[load_nodes].mean(axis=0)))
        sf = mat.strength / peak if peak > 0 else np.inf
        worst_sf = min(worst_sf, sf)
        cases.append({
            "name": case.name,
            "peakStress": peak,
            "peakAt": at,
            "p99Stress": float(np.percentile(vm[body] if body.any() else vm, 99)),
            "contactPeakStress": float(vm.max()),
            "safetyFactor": sf,
            "deflection": defl,
            "maxDisplacement": float(np.linalg.norm(disp, axis=1).max()),
            "compliance": float(f @ u),
        })
    report["cases"] = cases
    report["safetyFactor"] = worst_sf
    report["analysisCells"] = int(hm.n_elem)
    report["seconds"] = time.time() - t0
    return report
