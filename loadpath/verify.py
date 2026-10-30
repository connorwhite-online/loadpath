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
- fatigue life, creep and bolt-preload relaxation for load cases that
  repeat or are held (durability.py);
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
    if len(ray_idx) == 0 and len(mesh.faces):
        # A mesh handed back from a worker process (GRIP's turret) answered
        # no ray at all, where the same mesh saved and reloaded answers
        # 16,740: rebuild it fresh, with a fresh intersector, and ask again.
        import trimesh

        fresh = trimesh.Trimesh(np.array(mesh.vertices, float), np.array(mesh.faces), process=False)
        locs, ray_idx, _ = fresh.ray.intersects_location(ray_o, ray_d, multiple_hits=True)
    locs = np.asarray(locs, float).reshape(-1, 3)
    ray_idx = np.asarray(ray_idx, int).reshape(-1)
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


def surface_thickness(mesh, samples: int = 6000, seed: int = 0, study: Study | None = None):
    """Member thickness without a grid: from evenly spread surface points, a
    ray straight in along the inward normal to where it leaves the part (the
    shape diameter). Returns (p1, p5) in mm. The voxel ridge reading
    (_thickness) is quantized to the analysis cell, under-reading a 2.05 mm
    wall at a 0.98 mm pitch as 1.46; this one is not, but it costs a ray
    cast per sample, so it is for scoring a finished part, not every outcome.
    With the study, a ray that both enters and leaves on the user's own
    surfaces (preserved or keep-out) is left out: that thickness is the
    study's, not the optimizer's (GRIP's wrist yoke: its J5 servo sits 1.9 mm
    above the J4 horn, so the preserved horn plate between them is 1.9 mm)."""
    import trimesh

    # a fresh, merged copy: a part straight from the booleans can carry
    # degenerate slivers whose bounds break the ray tree (the motor mount)
    mesh = trimesh.Trimesh(np.array(mesh.vertices, float), np.array(mesh.faces), process=True)
    mesh.update_faces(mesh.nondegenerate_faces())
    pts, fi = trimesh.sample.sample_surface_even(mesh, samples, seed=seed)
    n = mesh.face_normals[fi]
    ok = np.isfinite(pts).all(axis=1) & (np.linalg.norm(n, axis=1) > 0.5)
    pts, n = pts[ok], n[ok]
    origin = pts - n * 1e-3
    loc, ri, _ = mesh.ray.intersects_location(origin, -n, multiple_hits=False)
    if len(ri) == 0:
        return None, None
    d = np.linalg.norm(loc - origin[ri], axis=1)
    if study is not None:
        shapes = [r.shape for r in list(study.preserve) + list(study.keep_out)]
        if shapes:
            def on_user(q):
                return np.min([np.abs(sh.sdf(q)) for sh in shapes], axis=0) < 0.05
            mine = ~(on_user(pts[ri]) & on_user(loc))
            if mine.any():
                d = d[mine]
    return float(np.percentile(d, 1)), float(np.percentile(d, 5))


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


def nodal_stress(stiff, u, modulus):
    """The nodal-averaged stress tensor (sx, sy, sz, txy, tyz, tzx), as
    nodal_von_mises averages it."""
    from .fem import CORNERS, _b_matrix, elastic_c

    mesh = stiff.mesh
    c = elastic_c(1.0, stiff.nu) * modulus
    ue = u[mesh.edof]
    acc = np.zeros((mesh.n_nodes, 6))
    for k, (a, b, cc) in enumerate(2.0 * CORNERS - 1.0):
        sk = (ue @ (_b_matrix(a, b, cc) / mesh.pitch).T) @ c.T
        np.add.at(acc, mesh.enode[:, k], sk)
    cnt = np.bincount(mesh.enode.ravel(), minlength=mesh.n_nodes)
    return acc / cnt[:, None]


def _overhang(mesh, direction, study=None, max_angle_deg=45.0, tolerance_deg=5.0):
    """Down-facing surface that needs support: faces whose normal is within
    (max_angle - tolerance) of straight down, i.e. ceilings and slopes
    shallower than the self-supporting limit, excluding the face on the build
    plate. The tolerance keeps marching-cubes facets on a true 45-degree
    slope from counting. `overhangArea` is everything; `designOverhangArea`
    leaves out faces on preserved or keep-out geometry, which the user drew
    (a horizontal bore's ceiling is theirs, not the optimizer's)."""
    from .am import build_vector

    up = build_vector(direction)
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


def _boundary_cut_area(study: Study, mesh) -> float:
    """mm^2 of the part's surface lying flat on a face of a box design space,
    away from every support and preserved shape: organic material sawn off
    where the design box happened to end, not where anything mates. Zero on
    a clean result (GRIP's wrist yoke QA, 10/28)."""
    from .shapes import Box

    boxes = [d for d in study.design_space if isinstance(d, Box)]
    if not boxes:
        return 0.0
    c = mesh.triangles_center
    nrm = mesh.face_normals
    area = mesh.area_faces
    keep = [r.shape for r in study.preserve]
    for sup in study.supports:
        try:
            keep.append(sup.shape if sup.shape is not None else study.region_shape(sup.region)[0])
        except ValueError:
            pass
    near = np.min([k.sdf(c) for k in keep], axis=0) < 1.0 if keep else np.zeros(len(c), bool)
    cut = np.zeros(len(c), bool)
    for b in boxes:
        for ax in range(3):
            for side, plane in ((-1, b.lo[ax]), (1, b.hi[ax])):
                cut |= (np.abs(c[:, ax] - plane) < 0.05) & (nrm[:, ax] * side > 0.99)
    return float(area[cut & ~near].sum())


# A concave edge sharper than this (the faces turn by more than 60 degrees)
# is a crease: a notch that concentrates stress and that a fillet should have
# rounded. Marching-cubes facets on an organic surface turn by a few degrees.
CREASE_ANGLE_DEG = 60.0


def _creases(study: Study, mesh, angle_deg: float = CREASE_ANGLE_DEG) -> dict:
    """Sharp concave edges the optimizer made: where an organic member meets
    preserved geometry without a fillet, or two members meet in a notch.
    GRIP's wrist yoke passed every number with hard creases at its bosses,
    found only by zooming in (QA, 10/28); this measures them. Edges lying on
    keep-out surfaces (a blind bore's floor) or between two faces of the
    user's own preserved geometry are the user's, and are left out.
    Returns crease length (mm) and up to five crease locations."""
    ang = mesh.face_adjacency_angles
    sharp = (~mesh.face_adjacency_convex) & (ang > np.radians(angle_deg))
    out = {"creaseLength": 0.0, "creaseAt": []}
    if not sharp.any():
        return out
    e = mesh.face_adjacency_edges[sharp]
    a, b = mesh.vertices[e[:, 0]], mesh.vertices[e[:, 1]]
    mid = 0.5 * (a + b)
    length = np.linalg.norm(b - a, axis=1)
    users = np.zeros(len(mid), bool)
    if study.keep_out:
        users |= np.min([np.abs(r.shape.sdf(mid)) for r in study.keep_out], axis=0) < 0.3
    if study.preserve:
        fc = mesh.triangles_center[mesh.face_adjacency[sharp]]          # (n, 2, 3)
        on = np.min([np.abs(r.shape.sdf(fc.reshape(-1, 3))) for r in study.preserve], axis=0).reshape(-1, 2)
        users |= (on < 0.05).all(axis=1)
    mine = ~users
    out["creaseLength"] = float(length[mine].sum())
    pts = mid[mine]
    w = length[mine]
    while len(pts) and len(out["creaseAt"]) < 5:
        # heaviest cluster first, 5 mm across
        i = int(np.argmax([w[np.linalg.norm(pts - p, axis=1) < 5.0].sum() for p in pts]))
        near = np.linalg.norm(pts - pts[i], axis=1) < 5.0
        out["creaseAt"].append({"at": pts[near].mean(axis=0).round(2).tolist(), "length": float(w[near].sum())})
        pts, w = pts[~near], w[~near]
    return out


def _closed_bores(study: Study, mesh) -> list:
    """Cylindrical keep-outs (bores) with an end the part closes over.

    A fastener needs its whole path clear: the shank through the bore, and
    the head, washer and driver out past its end. A bore that stops inside
    material is a blind hole no screw can be driven through, the way the
    wall bracket's screw holes came out capped (and GRIP's yoke had material
    over its bolt seats). The fix is in the study: extend the keep-out over
    the head and the driver's reach. Probes a short way past each end cap,
    on the axis and on a ring at 0.3 of the radius; an end is closed when
    most probes land inside the part. A keep-out marked "bore": false (a
    clearance volume that may end against material) is not checked."""
    from .shapes import Cylinder

    out = []
    for r in study.keep_out:
        if not r.bore:
            continue
        sh = r.shape
        if not isinstance(sh, Cylinder) or sh.r_inner > 0:
            continue
        a = sh.axis
        u = np.cross(a, [1.0, 0.0, 0.0] if abs(a[0]) < 0.9 else [0.0, 1.0, 0.0])
        u /= np.linalg.norm(u)
        v = np.cross(a, u)
        # on and near the axis only: a head-clearance bore ending on its
        # seat is closed round the rim by design, with the screw's own
        # narrower bore running on through the middle (the motor mount's
        # access holes); a capped hole is closed across the middle too
        ring = [np.zeros(3)] + [0.3 * sh.r * (np.cos(t) * u + np.sin(t) * v)
                                for t in np.linspace(0, 2 * np.pi, 6, endpoint=False)]
        for end, cap, out_dir in (("p0", sh.p0, -a), ("p1", sh.p1, a)):
            probes = np.array([cap + d * out_dir + o for d in (1.5, 3.0) for o in ring])
            inside = mesh.contains(probes)
            if inside.mean() > 0.5:
                out.append({"id": r.id, "end": end, "at": cap.round(2).tolist()})
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


# A wetted wall is a voxel staircase: the stress at its nodes alternates
# between the step's teeth (unloaded corners, low) and its re-entrant corners
# (high), and refining the grid does not settle it, the staircase being the
# same shape at every pitch. On a closed cylinder under pressure (r 20, wall
# 3, Lamé 7.10 MPa at the bore) the wetted nodes read 5.6 at the median and
# 8.6-9.2 at the peak from 2.5 to 5 cells through the wall; averaged over the
# wetted nodes within 1.5 cells of each, the peak reads 7.6-7.9, the real
# wall's to within the discretization allowance.
_WET_SMOOTH = 1.5
# cells through the thinnest pinned wall or skin (study.shells), and the most
# cells verification will spend getting there
_SHELL_CELLS = 4.0
_SHELL_MAX_CELLS = 400_000


def _smooth_wetted(xyz, vm, wet, pitch):
    """von Mises at wetted nodes averaged over the wetted nodes near each."""
    from scipy.spatial import cKDTree

    idx = np.nonzero(wet)[0]
    if len(idx) == 0:
        return vm
    import scipy.sparse as sp

    tree = cKDTree(xyz[idx])
    pairs = tree.query_pairs(_WET_SMOOTH * pitch, output_type="ndarray")
    n = len(idx)
    rows = np.concatenate([pairs[:, 0], pairs[:, 1], np.arange(n)])
    cols = np.concatenate([pairs[:, 1], pairs[:, 0], np.arange(n)])
    m = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
    out = vm.copy()
    out[idx] = (m @ vm[idx]) / np.asarray(m.sum(axis=1)).ravel()
    return out


def _rigid_free(xyz, u):
    """u less its best-fit rigid motion (least squares over the nodes): the
    deformation of a part held only by a 3-2-1 support."""
    from .fem import rigid_body_modes

    b = rigid_body_modes(xyz)
    coef = np.linalg.lstsq(b, u, rcond=None)[0]
    return u - b @ coef


def _submodel_peak(study, mesh, hm, u, center, half, pitch, mat, contact_xyz, band, case=None):
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
    st = Stiffness(sm, study.elastic(mat), np.zeros(sm.n_dof, bool))
    k = st.assemble(np.full(sm.n_elem, mat.E)).tocsr()
    free = ~pd
    kff = k[free][:, free]
    rhs = -(k[free][:, pd] @ up)
    if case is not None:
        # a pressure inside the box pushes on its walls here too
        from .pressure import pressure_force

        fp = np.zeros(sm.n_dof)
        for ld in case.loads:
            if ld.pressure:
                fp += pressure_force(sm, study.region_shape(ld.region)[0], ld.pressure)[0]
        rhs = rhs + fp[free]
    solver = Solver(sm, rtol=1e-7, maxiter=4000)
    # Rigid-body modes are pinned by the driven nodes; AMG still wants a
    # near-null space sized to the free unknowns, so use plain CG+AMG on kff.
    from scipy.sparse.linalg import cg

    from .fem import amg_solver
    ml = amg_solver(kff, symmetry="hermitian", strength="symmetric", max_coarse=400, coarse_solver="splu")
    uf, info = cg(kff, rhs, rtol=1e-7, maxiter=4000, M=ml.aspreconditioner(cycle="V"))
    del solver
    if info != 0:
        return None
    us = np.zeros(sm.n_dof)
    us[pd] = up
    us[free] = uf
    vm = nodal_von_mises(st, us, mat.E)
    if case is not None and fp.any():
        # over the global grid's reach: a finer staircase is no smoother
        vm = _smooth_wetted(sm.node_xyz, vm, np.abs(fp.reshape(-1, 3)).sum(axis=1) > 0, hm.pitch)
    # Trust the inside of the box only: two global cells from the cut faces.
    margin = 2.0 * hm.pitch
    inner = np.all((sm.node_xyz > blo + margin) & (sm.node_xyz < bhi - margin), axis=1) & ~near_contact
    if not inner.any():
        return None
    i = int(np.argmax(np.where(inner, vm, -1.0)))
    return float(vm[i]), sm.node_xyz[i].tolist()



def _preserve_missing(pres_eff, part, mesh, samples: int = 400) -> float:
    """Volume of preserved geometry the part lacks (mm^3). The boolean finds
    candidate pieces; each is then confirmed by sampling points inside it and
    asking the mesh itself (by ray parity) whether they are solid. Where a
    keep-out shares a preserved surface (a bore the size of a boss's inner
    wall), manifold3d can report slivers of the preserved shape missing from
    a part that, point by point, has them: the grip upper arm's elbow socket
    read 300 mm^3 short with every sampled point inside."""
    import trimesh

    from .shapes import trimesh_from_manifold

    total = 0.0
    for b in (pres_eff - part).decompose():
        v = b.volume()
        if v <= 1e-3:
            continue
        tb = trimesh_from_manifold(b)
        pts = trimesh.sample.volume_mesh(tb, samples)
        if len(pts) < 10:
            pts = trimesh.sample.volume_mesh(tb, 20 * samples)
        if len(pts) < 10:
            # too thin a sliver for sampling to land in: missing only if it
            # is thicker than a print resolves (thickness ~ 2 V / A)
            total += v if 2.0 * v / max(b.surface_area(), 1e-9) > 0.1 else 0.0
            continue
        total += v * float((~mesh.contains(pts)).mean())
    return total


def verify_part(study: Study, mesh, max_cells: int = 60_000, min_pitch: float | None = None,
                contact_band: float | None = None, refine: int = 3, fields: bool = False) -> dict:
    """Analyze a finished part (trimesh) against the study. Returns a JSON-able
    report. `contact_band` (mm) is how far from applied loads and supports
    stress is left out of the peak; it defaults to 1.5 analysis cells, and the
    study runner sets it to the optimization cell size so the excluded band
    does not shrink toward zero as the analysis grid gets finer (a clamped
    edge's stress grows without bound under refinement). `refine` is how many
    hotspots per load case get a 3x finer submodel (see _submodel_peak).
    `fields` adds report["fields"]: the analysis nodes and, per load case,
    nodal von Mises stress and displacement (numpy arrays, not JSON) for
    stress maps (render.render_stress)."""
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
        report["preserveMissing"] = _preserve_missing(pres_eff, part, mesh)
    else:
        report["preserveMissing"] = 0.0
    report["outsideDesignSpace"] = float((part - design).volume())
    report["boundaryCutArea"] = _boundary_cut_area(study, mesh)
    report.update(_creases(study, mesh))
    report["closedBores"] = _closed_bores(study, mesh)
    # decompose() lists each enclosed cavity too, as a negative volume
    pieces = [b.volume() for b in part.decompose()]
    report["bodies"] = sum(1 for v in pieces if v > 0)
    report["enclosedCavityVolume"] = float(-sum(v for v in pieces if v < 0))

    if study.print_direction is not None:
        report.update(_overhang(mesh, study.print_direction, study))
    proc = study.process
    build = study.print_direction if study.print_direction is not None else \
        (proc.build if proc is not None and proc.build != "auto" else None)
    if build is not None:
        # sliced as a printer would: bridges, islands, support volume, bed
        from .printcheck import check_print

        report["print"] = check_print(mesh, build, process=proc.kind if proc is not None else "fdm",
                                      material=mat)

    pitch = (vol / max_cells) ** (1.0 / 3.0)
    if min_pitch:
        pitch = max(pitch, min_pitch)
    if study.shells:
        # A wall voxelizes to a whole number of cells, so at two or three
        # cells through it reads up to a cell thick or thin: bending stress
        # goes as 1/t^2, and a plenum's 2 mm wall read 2.4 mm and 30% under
        # its stress at 0.8 mm cells. Four cells through the thinnest wall or
        # skin, within a cell budget.
        thin = min(t for _, t in study.shells)
        pitch = min(pitch, max(thin / _SHELL_CELLS, (vol / _SHELL_MAX_CELLS) ** (1.0 / 3.0)))
    solid, origin = voxelize(mesh, pitch)
    report["analysisPitch"] = pitch
    report["minMember"], report["p5Member"] = _thickness(solid, pitch)

    # Face connectivity: cells that meet only at an edge or a corner share
    # two nodes or one, a hinge the stiffness matrix cannot hold, and CG
    # never converges (GRIP's upper arm: 44 such islands of 4-6 cells at a
    # 1.04 mm pitch stalled verification for over an hour).
    lbl, nlab = ndimage.label(solid)
    elem_ijk = np.argwhere(solid)
    hm = HexMesh(elem_ijk, pitch, origin)
    bc = boundary_conditions(study, hm, check_loads=False, tol_floor=0.5 * (contact_band or 0.0) + 0.5 * pitch)
    # Material not connected to a support would float; it is reported as a
    # separate body above and left out of the stress analysis.
    supported_lbls = np.unique(lbl[tuple(elem_ijk[bc.support_nodes[hm.enode].any(axis=1)].T)])
    keep_el = np.isin(lbl[tuple(elem_ijk.T)], supported_lbls)
    if not keep_el.all():
        hm = HexMesh(elem_ijk[keep_el], pitch, origin)
        bc = boundary_conditions(study, hm, check_loads=False, tol_floor=0.5 * (contact_band or 0.0) + 0.5 * pitch)
    support_nodes, fixed = bc.support_nodes, bc.fixed

    band = max(1.5 * pitch, contact_band or 0.0)
    report["contactBand"] = band
    stiff = Stiffness(hm, study.elastic(mat), fixed)
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
        if bc.case_pressure_nodes[ci].any():
            vm = _smooth_wetted(hm.node_xyz, vm, bc.case_pressure_nodes[ci], pitch)
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
                                   hm.node_xyz[contact], band, case)
                if r is None:
                    continue
                refined.append({"at": r[1], "peakStress": r[0]})
                if r[0] > peak:
                    peak, at = r
        disp = u.reshape(-1, 3)
        if fields:
            report.setdefault("fields", {"nodes": hm.node_xyz, "cases": []})["cases"].append(
                {"name": case.name, "vonMises": vm, "displacement": disp, "wetted": bc.case_pressure_nodes[ci]})
        wet = bc.case_pressure_nodes[ci]
        if load_nodes.any():
            defl = float(np.linalg.norm(disp[load_nodes].mean(axis=0)))
        elif wet.any():
            # a pressure's deflection is how far its walls bulge: the most any
            # wetted node moves (relative to the part, when nothing holds it)
            dw = _rigid_free(hm.node_xyz, u).reshape(-1, 3) if study.self_equilibrated else disp
            defl = float(np.linalg.norm(dw[wet], axis=1).max())
        else:
            defl = 0.0
        if bc.pressure_imbalance[ci] > 0.02 and study.self_equilibrated:
            report.setdefault("warnings", []).append(
                f"{case.name}: the pressure no longer balances on the delivered part "
                f"({bc.pressure_imbalance[ci]:.1%} net): a wall the cavity relies on is open (a leak)")
        # The safety factor carries a discretization allowance: on the
        # plate-with-hole benchmark (tests) refined voxel peaks land within
        # -12%..+13% of Peterson's Kt, depending on how the grid falls on the
        # hole. Under-reading stress is the unsafe direction, so the factor
        # is computed against peak * STRESS_ALLOWANCE.
        sf = mat.strength / (peak * STRESS_ALLOWANCE) if peak > 0 else np.inf
        inter = None
        proc = study.process
        if proc is not None and proc.build not in (None, "auto") and proc.interlayer < 1.0:
            # Layered parts part along their layers: the tension pulling
            # layers apart (the stress normal to them) against the bond's
            # strength, a fraction of the in-plane one.
            from .am import build_vector

            nb = build_vector(proc.build)
            sig = nodal_stress(stiff, u, mat.E)
            sn = sig[:, :3] @ (nb * nb) + 2.0 * (sig[:, 3] * nb[0] * nb[1] + sig[:, 4] * nb[1] * nb[2]
                                               + sig[:, 5] * nb[2] * nb[0])
            tens = np.where(body, np.maximum(sn, 0.0), 0.0)
            inter = float(tens.max())
            sf_inter = proc.interlayer * mat.strength / (inter * STRESS_ALLOWANCE) if inter > 0 else np.inf
            sf = min(sf, sf_inter)
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
            "interlayerStress": inter,
            "deflection": defl,
            "maxDisplacement": float(np.linalg.norm(disp, axis=1).max()),
            "compliance": float(f @ u),
            **({"pressureImbalance": bc.pressure_imbalance[ci]} if wet.any() else {}),
        })
    report["cases"] = cases
    report["safetyFactor"] = worst_sf
    report["stressAllowance"] = STRESS_ALLOWANCE
    from .durability import active, assess
    if active(study):
        # fatigue (cases with cycles), creep and relaxation (hours, preload)
        report["durability"] = assess(study, cases, STRESS_ALLOWANCE)
    report["analysisCells"] = int(hm.n_elem)
    report["seconds"] = time.time() - t0
    return report
