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

import threading
import time

import numpy as np
from scipy import ndimage

from .domain import boundary_conditions, gravity_force
from .fem import HexMesh, Solver, Stiffness, von_mises
from .spec import Study

__all__ = ["voxelize", "verify_part"]

STRESS_ALLOWANCE = 1.15


def voxelize(mesh, pitch: float, box=None, fractional=False):
    """Solid cells of a closed mesh by ray parity along +z through each
    column of cell centers. Returns (bool grid, origin = min corner of cell
    (0,0,0)). `box` = (lo, hi) voxelizes only that region (the rays still
    start below the whole mesh, so parity is right). `fractional` returns
    each cell's filled share instead (float grid): the z overlap is exact
    from the ray's crossings, averaged over 3 x 3 sub-columns per cell."""
    if fractional:
        return _voxel_fractions(mesh, pitch, box)
    mlo = np.asarray(mesh.bounds, float)[0]
    lo, hi = (np.asarray(b, float) for b in box) if box is not None else np.asarray(mesh.bounds, float)
    # a box a whole number of cells across (a design grid's) must come out
    # that many: (hi - lo) / pitch rounds to just over n at some pitches
    # (the arm link's 57 cells read 58 at 2.887 mm, and its access report
    # failed on the shape mismatch)
    dims = np.maximum(np.ceil((hi - lo) / pitch - 1e-6).astype(int), 1)
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
        # A mesh handed back from a worker process (the robot arm's turret) answered
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


def _voxel_fractions(mesh, pitch, box=None, sub=3):
    lo, hi = (np.asarray(b, float) for b in box) if box is not None else np.asarray(mesh.bounds, float)
    dims = np.maximum(np.ceil((hi - lo) / pitch - 1e-6).astype(int), 1)
    origin = lo - 0.5 * (dims * pitch - (hi - lo))
    # the sub-columns are one bool grid at pitch/sub in x and y; each z
    # interval's overlap with each cell is added exactly
    mlo = np.asarray(mesh.bounds, float)[0]
    fine = np.array([dims[0] * sub, dims[1] * sub])
    xs = origin[0] + (np.arange(fine[0]) + 0.5) * pitch / sub
    ys = origin[1] + (np.arange(fine[1]) + 0.5) * pitch / sub
    gx, gy = np.meshgrid(xs, ys, indexing="ij")
    jit = pitch * 1e-4
    ray_o = np.stack([gx.ravel() + 0.37 * jit, gy.ravel() + 0.61 * jit,
                      np.full(gx.size, mlo[2] - pitch)], axis=1)
    locs, ray_idx, _ = mesh.ray.intersects_location(ray_o, np.tile([0.0, 0.0, 1.0], (len(ray_o), 1)),
                                                    multiple_hits=True)
    locs = np.asarray(locs, float).reshape(-1, 3)
    ray_idx = np.asarray(ray_idx, int).reshape(-1)
    order = np.lexsort((locs[:, 2], ray_idx))
    ray_idx, zs = ray_idx[order], locs[order, 2]
    starts = np.searchsorted(ray_idx, np.arange(len(ray_o)))
    ends = np.searchsorted(ray_idx, np.arange(len(ray_o)), side="right")
    z0 = origin[2] + np.arange(dims[2]) * pitch
    frac = np.zeros(dims, float)
    for r in np.nonzero(ends > starts)[0]:
        z = zs[starts[r]:ends[r]]
        if len(z) > 1:
            z = z[np.concatenate([[True], np.diff(z) > 1e-7])]
        if len(z) % 2:
            z = z[:-1]
        cover = np.zeros(dims[2])
        for a, b in z.reshape(-1, 2):
            cover += np.clip(np.minimum(b, z0 + pitch) - np.maximum(a, z0), 0.0, pitch)
        i, j = np.unravel_index(r, (fine[0], fine[1]))
        frac[i // sub, j // sub] += cover
    return np.clip(frac / (pitch * sub * sub), 0.0, 1.0), origin


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
    study's, not the optimizer's (the robot arm's wrist yoke: its J5 servo sits 1.9 mm
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
    # in batches: without embree the ray tree's candidate pairs for all 6000
    # rays at once took 7 GB on a 75k-face yoke (the cloud box OOMed)
    hits = [mesh.ray.intersects_location(origin[k:k + 500], -n[k:k + 500], multiple_hits=False)[:2]
            for k in range(0, len(origin), 500)]
    loc = np.concatenate([h[0].reshape(-1, 3) for h in hits]) if hits else np.zeros((0, 3))
    ri = np.concatenate([h[1] + k for h, k in zip(hits, range(0, len(origin), 500))]).astype(int) if hits else np.zeros(0, int)
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


def nodal_von_mises(stiff, u, modulus, thermal=None):
    """Von Mises at the nodes: each element's stress evaluated at its own
    corners, averaged over the elements sharing a node. Element-centre stress
    sits half a cell inside the surface, which is exactly where stress
    concentrates; on a plate with a hole (Peterson Kt_net = 2.43, peak 81 MPa)
    centre values read 64-74 MPa and neighbor-smoothed centres 59-68, while
    nodal averages read 73 at 7 cells per hole radius and 81.7 from 9 cells
    up, converged. Averaging across elements is also what keeps a single
    voxel staircase corner from reading as a spike."""
    return von_mises(nodal_stress(stiff, u, modulus, thermal))


def nodal_stress(stiff, u, modulus, thermal=None):
    """The nodal-averaged stress tensor (sx, sy, sz, txy, tyz, tzx), as
    nodal_von_mises averages it. `modulus` is one number or one per element;
    `thermal` (per element) is a free expansion strain, taken off the strain
    before the stress is formed."""
    from .fem import CORNERS, elastic_c, hex8_b

    cg = getattr(stiff, "cutgrid", None)
    if cg is not None:
        return cg.nodal_stress(u, modulus, thermal)
    mesh = stiff.mesh
    c = elastic_c(1.0, stiff.nu)
    element = getattr(stiff, "element", None)
    mod = np.broadcast_to(np.asarray(modulus, float), (mesh.n_elem,))[:, None]
    ue = u[mesh.edof]
    acc = np.zeros((mesh.n_nodes, 6))
    # the cells whose stress is the printed material's: a bought member's are not (stock.py), and a node
    # on the joint reads the printed cells round it alone
    mask = getattr(stiff, "stress_mask", None)
    if mask is not None:
        mod = mod * mask[:, None]
    for k, (a, b, cc) in enumerate(2.0 * CORNERS - 1.0):
        strain = ue @ (hex8_b(a, b, cc, stiff.nu, element) / mesh.pitch).T
        if thermal is not None:
            strain[:, :3] -= np.asarray(thermal, float)[:, None]
        np.add.at(acc, mesh.enode[:, k], (strain @ c.T) * mod)
    if mask is not None:
        cnt = np.bincount(mesh.enode[mask].ravel(), minlength=mesh.n_nodes)
        return acc / np.maximum(cnt, 1)[:, None]
    cnt = np.bincount(mesh.enode.ravel(), minlength=mesh.n_nodes)
    return acc / cnt[:, None]


# mm above the lowest point that still counts as standing on the build plate
_PLATE_BAND = 1.0


def _overhang(mesh, direction, study=None, max_angle_deg=45.0, tolerance_deg=5.0):
    """Down-facing surface that needs support: faces whose normal is within
    (max_angle - tolerance) of straight down, i.e. ceilings and slopes
    shallower than the self-supporting limit, excluding the face on the build
    plate and the rounded bottom edge within 1 mm of it (a few layers, which
    print without support; marching cubes rounds every edge on the plate, and
    on the printed hook the 0.5-1 mm band alone was a point of its 4.3%). The
    tolerance keeps marching-cubes facets on a true 45-degree
    slope from counting. `overhangArea` is everything; `designOverhangArea`
    leaves out faces on preserved or keep-out geometry, which the user drew
    (a horizontal bore's ceiling is theirs, not the optimizer's)."""
    from .am import build_vector

    up = build_vector(direction)
    down = -(mesh.face_normals @ up)
    centers = mesh.triangles_center
    height = centers @ up
    bad = (down > np.cos(np.radians(max_angle_deg - tolerance_deg))) & (height > height.min() + _PLATE_BAND)
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
    a clean result (the robot arm's wrist yoke QA, 10/28)."""
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
    The robot arm's wrist yoke passed every number with hard creases at its bosses,
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
    wall bracket's screw holes came out capped (and the robot arm's yoke had material
    over its bolt seats). The fix is in the study: extend the keep-out over
    the head and the driver's reach. Probes a short way past each end cap,
    on the axis and on a ring at 0.3 of the radius; an end is closed when
    most probes land inside the part. A keep-out marked "bore": false (a
    clearance volume that may end against material) is not checked."""
    from .shapes import Cylinder

    out = []
    rng = np.random.default_rng(0)
    members = {m.id for m in getattr(study, "stock", None) or []}
    for r in study.keep_out:
        # (a bought member's seat may end in the part: a socket's floor is its stop, stock.py)
        if not r.bore or r.id in members:
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
            # ray parity (_inside): trimesh's contains (no embree) took
            # 4-5 s a verify on the wrist yoke for these few dozen points
            inside = _inside(mesh, probes, rng)
            if inside.mean() > 0.5:
                hit = {"id": r.id, "end": end, "at": cap.round(2).tolist()}
                filled = _filled_clearances(study, probes[inside])
                if filled:
                    hit["filledClearances"] = filled
                out.append(hit)
    return out


def _filled_clearances(study: Study, pts) -> list:
    """Why a bore end is closed, when the cause is in the study: the material
    past it is drawn geometry standing in a keep-out that does not cut what
    is drawn ("bore": false without cutsPreserve: a clearance zone, spec.
    Region.cuts). Where a part sits in that space (a bearing in its pocket,
    an insert, a screw, a cable) the keep-out has to cut the drawn shape, or
    the drawn shape fills it: the wrist link's idler hub filled its bearing
    pocket, closing the insert bore that runs into it. Lists each such
    keep-out holding most of `pts` with the preserved shapes there."""
    out = []
    for k in study.keep_out:
        if getattr(k, "cuts", True):
            continue
        ink = k.shape.sdf(pts) < 0
        if ink.mean() <= 0.5:
            continue
        drawn = [p.id for p in study.preserve if (p.shape.sdf(pts[ink]) < 0).mean() > 0.5]
        if drawn:
            out.append({"keepOut": k.id, "drawn": drawn})
    return out


def drawn_kept(study, regions=None):
    """What of the preserved shapes must be in the part, as one manifold
    (None if nothing): each less the keep-outs that cut it (spec.cuts_of)."""
    from .sharp import exact_manifold
    from .spec import cuts_of
    out = []
    for r in (study.preserve if regions is None else regions):
        p = exact_manifold(r.shape)
        cut = _union([k.shape for k in cuts_of(study, r)])
        out.append(p - cut if cut is not None else p)
    if not out:
        return None
    import manifold3d as m3d
    return m3d.Manifold.batch_boolean(out, m3d.OpType.Add) if len(out) > 1 else out[0]


def _union(shapes):
    import manifold3d as m3d

    from .sharp import exact_manifold
    # the same exact solids reconstruct.py cut and added (circles 0.002 mm true)
    mans = [exact_manifold(s) for s in shapes]
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
# A grid coarser than the design's reads the part it was designed on
# coarser still: a 3 mm deck designed at 1.5 mm cells, analysed at 1.91,
# sagged 2.16 mm where 1.41 mm cells read 4.09 and tets 6.8 (the fan
# plate). With max_pitch, the cell budget stretches up to this many times
# to reach it, and the report says when it still cannot.
_MAX_PITCH_STRETCH = 2.5
# Under this many cells through, a plate reads stiff in bending (hex cells
# lock in shear): that deck at 2 cells, 1.7x stiffer than tets.
_THIN_CELLS = 3.0


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


def _user_edges(study: Study, band: float):
    """Where a keep-out's surface meets a preserved body's (a screw-head
    counterbore's floor corner on a boss face, a bore's rim): a sharp edge
    the user drew, which the optimizer cannot round. Its linear-elastic stress
    grows without bound as the grid gets finer, like a clamped edge's, so it
    is left out of the peak over the same band. (The wall bracket's 150k-cell
    confirm check peaked there and read SF 1.6, 2.6, 2.5, 1.9, 3.7 at 11.5-15%
    of the space, out of order, against 2.7-3.8 on the 60k check.)

    Being near both surfaces is not enough: a pressure vessel's wall, or a
    ring whose bore is the keep-out's, lines the keep-out, and the two
    surfaces coincide over the whole cavity (a thick sphere's bore, Lame's
    peak, was left out entirely). An edge is where the surfaces cross
    (their normals not opposed) or where the keep-out's own surface turns
    within the band (a tunnel's rim on the face it ends on). Returns a
    function of points -> bool, or None when the study has no such edges."""
    keep = [r.shape for r in study.keep_out]
    pres = [r.shape for r in study.preserve]
    if not keep or not pres:
        return None
    h = 0.05 * band
    axes = np.eye(3)

    def sdf(shapes, xyz):
        return np.min([sh.sdf(xyz) for sh in shapes], axis=0)

    def normal(shapes, xyz):
        g = np.stack([sdf(shapes, xyz + h * a) - sdf(shapes, xyz - h * a) for a in axes], axis=1)
        return g / np.maximum(np.linalg.norm(g, axis=1, keepdims=True), 1e-12)

    def near(xyz):
        xyz = np.asarray(xyz, float)
        out = (np.abs(sdf(keep, xyz)) < band) & (np.abs(sdf(pres, xyz)) < band)
        if not out.any():
            return out
        x = xyz[out]
        nk = normal(keep, x)
        edge = np.einsum("ij,ij->i", nk, normal(pres, x)) > -0.9
        for o in np.concatenate([axes, -axes]) * band:
            edge |= np.einsum("ij,ij->i", nk, normal(keep, x + o)) < 0.9
        out[out] = edge
        return out
    return near


# trimesh's ray/contains queries share the mesh's rtree, which is not
# thread-safe (concurrent queries corrupt its node reads: "RTreeError:
# readNode"); the submodels' voxelizing takes turns, their solves overlap
_MESH_QUERY = threading.Lock()


def _submodel_peak(study, mesh, hm, u, center, half, pitch, mat, contact_xyz, band, case=None, edges=None,
                   cut=False, ratio=1.0):
    """Re-solve a small box around a global hotspot at a finer pitch, driven by
    the global displacement field on the box's cut faces (classic FE
    submodeling). Voxel stress converges only with about 8 cells across a
    feature's radius; the global grid rarely affords that, the box does.
    Returns (peak MPa, position) or None."""
    from scipy.spatial import cKDTree

    lo, hi = center - half, center + half
    cgrid = None
    with _MESH_QUERY:
        if cut:
            from .cutcell import cut_from_mesh

            cgrid = cut_from_mesh(mesh, pitch, study.elastic(mat), box=(lo, hi))
            solid, origin = cgrid.solid_grid()
        else:
            solid, origin = voxelize(mesh, pitch, box=(lo, hi))
    if solid.sum() < 64:
        return None
    sm = cgrid.mesh if cgrid is not None else HexMesh(np.argwhere(solid), pitch, origin)
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
    k = (cgrid.assemble(st, mat.E) if cgrid is not None else st.assemble(np.full(sm.n_elem, mat.E))).tocsr()
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
    from .failure import equivalent_stress

    vm = equivalent_stress(cgrid.nodal_stress(us, mat.E) if cgrid is not None else nodal_stress(st, us, mat.E), ratio)
    if case is not None and fp.any():
        # over the global grid's reach: a finer staircase is no smoother
        vm = _smooth_wetted(sm.node_xyz, vm, np.abs(fp.reshape(-1, 3)).sum(axis=1) > 0, hm.pitch)
    # Trust the inside of the box only: two global cells from the cut faces.
    margin = 2.0 * hm.pitch
    inner = np.all((sm.node_xyz > blo + margin) & (sm.node_xyz < bhi - margin), axis=1) & ~near_contact
    if edges is not None:
        inner &= ~edges(sm.node_xyz)
    if not inner.any():
        return None
    i = int(np.argmax(np.where(inner, vm, -1.0)))
    return float(vm[i]), sm.node_xyz[i].tolist()



def _inside(mesh, pts, rng):
    """Point-in-solid by ray-crossing parity (odd = inside): a ray along +x,
    tilted a little at random so edge and vertex grazes are measure-zero,
    tested exactly (Moller-Trumbore) against the triangles whose y-z box it
    can pass through. trimesh's own test (no embree) took 80 s of a 90 s
    verify on the arm link."""
    tri = np.asarray(mesh.triangles, float)
    v0, e1, e2 = tri[:, 0], tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]
    lo, hi = tri.min(axis=1), tri.max(axis=1)
    span = float(np.ptp(tri[..., 0])) + 1.0
    out = np.zeros(len(pts), bool)
    for i, o in enumerate(np.asarray(pts, float)):
        tilt = rng.uniform(-1e-3, 1e-3, 2)
        d = np.array([1.0, *tilt]) / np.sqrt(1.0 + tilt @ tilt)
        g = np.abs(tilt) * span + 1e-9
        c = np.flatnonzero((hi[:, 0] >= o[0]) & (lo[:, 1] <= o[1] + g[0]) & (hi[:, 1] >= o[1] - g[0])
                           & (lo[:, 2] <= o[2] + g[1]) & (hi[:, 2] >= o[2] - g[1]))
        if not len(c):
            continue
        p = np.cross(d, e2[c])
        det = np.einsum("ij,ij->i", e1[c], p)
        ok = np.abs(det) > 1e-12
        inv = np.where(ok, 1.0 / np.where(ok, det, 1.0), 0.0)
        tv = o - v0[c]
        u = np.einsum("ij,ij->i", tv, p) * inv
        q = np.cross(tv, e1[c])
        v = (q @ d) * inv
        t = np.einsum("ij,ij->i", q, e2[c]) * inv
        out[i] = int((ok & (u >= 0) & (v >= 0) & (u + v <= 1) & (t > 1e-9)).sum()) % 2 == 1
    return out


def _preserve_missing(pres_eff, part, mesh, samples: int = 400) -> float:
    """Volume of preserved geometry the part lacks (mm^3). The boolean finds
    candidate pieces; each is then confirmed by sampling points inside it and
    asking the mesh itself (by ray parity) whether they are solid. Where a
    keep-out shares a preserved surface (a bore the size of a boss's inner
    wall), manifold3d can report slivers of the preserved shape missing from
    a part that, point by point, has them: the grip upper arm's elbow socket
    read 300 mm^3 short with every sampled point inside."""
    from .shapes import trimesh_from_manifold

    rng = np.random.default_rng(0)
    total = 0.0
    for b in (pres_eff - part).decompose():
        v = b.volume()
        if v <= 1e-3:
            continue
        tb = trimesh_from_manifold(b)
        lo, hi = tb.bounds
        pts = np.empty((0, 3))
        for n in (samples, 20 * samples):     # rejection sampling in the piece's box
            cand = lo + rng.random((n, 3)) * (hi - lo)
            pts = cand[_inside(tb, cand, rng)][:samples]
            if len(pts) >= 10:
                break
        if len(pts) < 10:
            # too thin a sliver for sampling to land in: missing only if it
            # is thicker than a print resolves (thickness ~ 2 V / A)
            total += v if 2.0 * v / max(b.surface_area(), 1e-9) > 0.1 else 0.0
            continue
        total += v * float((~_inside(mesh, pts, rng)).mean())
    return total


def use_cut_cells(study, cut_cells=None) -> bool:
    """Whether verification analyzes cut cells: the argument, else
    LOADPATH_CUT_CELLS (1/0), else the study's resolution.cutCells, else
    yes unless the study carries a pressure or a thermal block."""
    import os

    if cut_cells is not None:
        return bool(cut_cells)
    env = os.environ.get("LOADPATH_CUT_CELLS", "").strip().lower()
    if env in ("1", "true", "yes", "on"):
        return True
    if env in ("0", "false", "no", "off"):
        return False
    asked = (getattr(study, "raw", None) or {}).get("resolution", {}).get("cutCells")
    if asked is not None:
        return bool(asked)
    # Default on, but not under pressure: a pressure's wetted nodes come
    # from the whole-cell staircase, and on a cut grid they sit off the
    # true wall (the thick sphere read Lame's 21 MPa as 105). Nor with a
    # thermal block: the expansion load is assembled on whole cells, and on
    # a cut grid the part-filled cells take a whole cell's push (the duty
    # cycle motor mount read SF 0.34 where whole cells read 8.6).
    if getattr(study, "thermal", None) is not None:
        return False
    return not any(ld.pressure for case in getattr(study, "load_cases", ()) for ld in case.loads)


def _solve_cut(cg, nu, fixed, springs, e_mat, moduli, rhs):
    """Solve the cut-cell model (cutcell.CutGrid) for each right-hand side:
    the shared-matrix cells as usual, the cut cells' own matrices added
    (assembled), or as the multigrid's extra operator, which its Galerkin
    coarsening carries down exactly. Returns (stiffness object with
    .cutgrid, displacements)."""
    from .simp import _use_mg

    hm = cg.mesh
    if _use_mg(hm.n_elem):
        from .mg import GridElasticity

        extra = cg.cut_operator(moduli)
        if springs is not None:
            extra = (extra + e_mat * springs).tocsr()
        stiff = GridElasticity(hm, nu, fixed, springs=extra, rtol=1e-6, maxiter=3000)
        stiff.set_moduli(cg.full_moduli(moduli))
    else:
        stiff = Stiffness(hm, nu, fixed)
        k = cg.assemble(stiff, moduli)
        if springs is not None:
            k = (k + e_mat * springs).tocsr()
        solver = Solver(hm, rtol=1e-6, maxiter=3000, rebuild_after_iters=10_000)
        stiff.cutgrid = cg
        return stiff, solver.solve_many(k, rhs)
    stiff.cutgrid = cg
    return stiff, stiff.solve_many(rhs)


# cells through a member below which it gets a submodel (cut cells)
_THIN_CELLS = 3.0


def _thin_spots(hm, solid, origin, pitch, vm, body, peak, radius, taken, count=3):
    """Hotspots on members under _THIN_CELLS cells thick reading at least
    0.8 of the peak, at least 2 * radius from each other and from `taken`."""
    edt = ndimage.distance_transform_edt(np.pad(solid, 1))[1:-1, 1:-1, 1:-1]
    # a cell's member thickness: twice the largest distance to void nearby
    thick = 2.0 * ndimage.maximum_filter(edt, size=5) - 0.5
    thin_el = thick[tuple(hm.elem_ijk.T)] < _THIN_CELLS
    thin = np.zeros(hm.n_nodes, bool)
    thin[hm.enode[thin_el].ravel()] = True
    cand = body & thin & (vm >= 0.8 * peak)
    out = []
    for i in np.argsort(-np.where(cand, vm, -1.0)):
        if not cand[i] or len(out) >= count:
            break
        x = hm.node_xyz[i]
        if all(np.linalg.norm(x - y) >= 2 * radius for y in list(taken) + out):
            out.append(x)
    return out


# Buckling is read on a grid of at most this many cells. The eigensolver
# factors the stiffness once and solves with it every iteration, so the cost
# grows fast with size: the 23 g wall bracket took 11 s at 5k cells, 17 s at
# 21k and 144 s at 71k (5.7 GB). The coarse grid reads low, the safe side:
# 1.44 / 1.54 / 1.37 at 5k / 11k / 21k cells against 1.81-1.83 at 42-71k.
# The search verifies every outcome, so this is per outcome: at 20k cells
# the arm link and turret spent 37-52 s each.
BUCKLING_CELLS = 12_000


def _buckling(study: Study, mesh, vol: float, pitch: float, contact_band: float | None,
              directions: list) -> dict | None:
    """The part's linear buckling load factor per load case: the multiple of
    the case's loads at which a perfect version of it buckles (nonlinear.
    linear_buckling, incompatible-mode hex8: a column one cell thick reads
    Euler's load within 1%). Real parts buckle below it (imperfections), and
    the goal's safety factor is the margin asked of it. Gravity counts; heat
    does not; the material is read isotropic (a layered process's softer
    across-layer stiffness is not in it). `directions` are the cases' worst force directions from the
    stress check (None for a fixed force). None when nothing holds the part."""
    from .directions import expand
    from .nonlinear import linear_buckling, solid_submesh

    t0 = time.time()
    mat = study.material
    if getattr(study, "stock", None):
        # with its bought members (stock.py): their stiffness is in bc.springs; their own buckling
        # (column, wall) is read in closed form from what they carry (stock.assess)
        from .stock import assembly_mesh
        mesh = assembly_mesh(study, mesh)
        vol = float(mesh.volume)
    pb = max(pitch, (vol / BUCKLING_CELLS) ** (1.0 / 3.0))
    solid, origin = voxelize(mesh, pb)
    hm = HexMesh(np.argwhere(solid), pb, origin)
    hm.stock_elastic = mat.nu
    bc = boundary_conditions(study, hm, check_loads=False, tol_floor=0.5 * (contact_band or 0.0) + 0.5 * pb)
    if not bc.fixed.any() and not bc.pinned:
        return None
    # Face connectivity, as the stress check has it: a piece that meets the
    # rest only at a cell's edge or corner hangs on a hinge the stiffness
    # cannot hold, and its rotation about it is a buckling mode at almost no
    # load (the Ti64 arm link's factors of 0.003-0.01, the mode on a sliver
    # that was all but cut loose). Such a piece is not part of the structure.
    lbl, _ = ndimage.label(solid)
    ijk = hm.elem_ijk
    held_lbl = np.unique(lbl[tuple(ijk[bc.support_nodes[hm.enode].any(axis=1)].T)])
    keep_el = np.isin(lbl[tuple(ijk.T)], held_lbl)
    if not keep_el.all():
        hm = HexMesh(ijk[keep_el], pb, origin)
        hm.stock_elastic = mat.nu
        bc = boundary_conditions(study, hm, check_loads=False, tol_floor=0.5 * (contact_band or 0.0) + 0.5 * pb)
    fg = gravity_force(hm, np.ones(hm.n_elem), bc.gravity, bc.fixed) if bc.gravity is not None else None
    spreads = bc.spreads or [None] * len(study.load_cases)
    rhs, layout = expand(bc.loads, spreads, fg)
    forces = []
    for ci, case in enumerate(study.load_cases):
        idx, d = layout[ci], directions[ci]
        if len(idx) == 1:
            f = rhs[idx[0]]
        else:
            f = sum(d[j] * rhs[i] for j, i in enumerate(idx[1:]))
            if idx[0] is not None:
                f = f + rhs[idx[0]]
        forces.append(f)
    springs = mat.E * bc.springs if bc.springs is not None else None
    # pieces no support holds would float: left out (a pin's springs hold)
    hold = bc.fixed.copy()
    if springs is not None and bc.pinned and not getattr(study, "stock", None):
        hold |= np.asarray(abs(springs).sum(axis=1)).ravel() > 0
    sub, el, _, vecs = solid_submesh(hm, np.ones(hm.n_elem, bool), hold, [bc.fixed.astype(float)] + forces)
    fixed, forces = vecs[0] > 0.5, vecs[1:]
    loaded = [i for i, f in enumerate(forces) if np.any(f)]
    cases = [{"name": c.name, "factor": None} for c in study.load_cases]
    if sub.n_elem and loaded:
        ks = None
        if springs is not None:
            dofs = (3 * hm.node_id(sub.node_ijk)[:, None] + np.arange(3)).ravel()
            ks = springs[dofs][:, dofs].tocsr()
        try:
            # one factorization for every case; three to four digits of the
            # factor are plenty (tol)
            res = linear_buckling(sub, mat.nu, mat.E, fixed, [forces[i] for i in loaded], n_modes=1,
                                  springs=ks, direct_max_dof=10 ** 6, tol=1e-5)
        except (RuntimeError, ValueError, ArithmeticError) as exc:
            for i in loaded:
                cases[i]["error"] = str(exc)
            res = []
        for i, r in zip(loaded, res):
            lam = float(r.factors[0])
            if np.isfinite(lam):
                mode = r.modes[0].reshape(-1, 3)
                cases[i].update(factor=lam, modeAt=sub.node_xyz[int(np.argmax(np.linalg.norm(mode, axis=1)))].tolist())
    got = [c["factor"] for c in cases if c["factor"] is not None]
    return {"factor": min(got) if got else None, "cases": cases, "pitch": pb,
            "cells": int(hm.n_elem), "seconds": time.time() - t0}


def mass_properties(mesh, density: float, axis=None) -> dict:
    """Mass (g), centre of mass (mm) and inertia tensor about it (kg mm^2)
    of a closed mesh at `density` g/cm^3; with a spec.MassProperties whose
    axis is set, the moment of inertia about that axis too (parallel axis
    from the centre of mass)."""
    kg_per_mm3 = float(density) * 1e-6
    vol = float(abs(mesh.volume))
    m = vol * kg_per_mm3
    com = np.asarray(mesh.center_mass, float)
    ic = np.asarray(mesh.moment_inertia, float) * kg_per_mm3
    if mesh.volume < 0:
        ic = -ic
    def sig(v):
        return float(f"{float(v):.6g}")

    out = {"mass": m * 1e3, "centerOfMass": [round(float(v), 3) for v in com],
           "inertia": [[sig(v) for v in row] for row in ic],
           "principalInertia": [sig(v) for v in np.linalg.eigvalsh(ic)]}
    if axis is not None and getattr(axis, "through", None) is not None:
        d = axis.direction
        r = com - axis.through
        r_perp2 = float(r @ r - (r @ d) ** 2)
        about = float(d @ ic @ d + m * r_perp2)
        out["axisInertia"] = sig(about)
        if axis.max_inertia is not None:
            out["maxInertia"] = axis.max_inertia
            out["ok"] = bool(about <= axis.max_inertia)
    tgt = getattr(axis, "com_target", None)
    if tgt is not None:
        held = np.isfinite(tgt)
        off = np.where(held, com - np.nan_to_num(tgt), 0.0)
        out["centerOfMassTarget"] = [None if not h else float(v) for h, v in zip(held, tgt, strict=True)]
        out["centerOfMassOffset"] = [round(float(v), 3) for v in off]
        out["centerOfMassTolerance"] = axis.com_tolerance
        out["centerOfMassOk"] = bool(np.all(np.abs(off) <= axis.com_tolerance + 1e-9))
    return out


# The modal check's grid, cells at their filled share: a clamped bar with a
# tip mass reads within 2% of beam theory from 2,000 cells on.
MODAL_CELLS = 12_000


MODAL_MODES_MAX = 60      # modes computed at most while reaching for the top of an excitation band


def band_keepout_(ex, eta):
    from .modal import band_keepout

    return band_keepout(ex[1], ex[2], eta, ex[3])


def modal_check(study: Study, mesh, vol: float | None = None, pitch: float | None = None) -> dict | None:
    """The study's vibration block on a delivered part: its lowest natural
    frequencies clamped at the supports (pins as the stress check holds
    them), the carried masses spread over the nodes of their regions (a
    keep-out's mass on the material lining it). Pieces no support holds
    are left out, as the buckling check does. None without a vibration
    block or anything holding the part."""
    from .domain import _region_nodes
    from .modal import modal_analysis
    from .shapes import manifold_from_trimesh

    vib = getattr(study, "vibration", None)
    if vib is None:
        return None
    t0 = time.time()
    mat = study.material
    stock = bool(getattr(study, "stock", None))
    if stock:
        # with its bought members (stock.py): their stiffness through bc.springs, their mass in the cells
        from .stock import assembly_mesh
        mesh, vol = assembly_mesh(study, mesh), None
    if vol is None:
        vol = float(manifold_from_trimesh(mesh).volume())
    pm = max(pitch or 0.0, (vol / MODAL_CELLS) ** (1.0 / 3.0))

    def members_on(m):
        if stock:
            m.stock_fill = share[tuple(m.elem_ijk.T)]
            m.stock_elastic = float(mat.nu)
        return m

    # each cell at its filled share of stiffness and mass: whole cells
    # stair-step a 10 mm bar to 10.3-10.8 mm and read its bending
    # frequency 4-16% high
    share, origin = voxelize(mesh, pm, fractional=True)
    solid = share > 0.02
    hm = members_on(HexMesh(np.argwhere(solid), pm, origin))
    bc = boundary_conditions(study, hm, check_loads=False, tol_floor=0.5 * pm)
    free = bool(study.inertia_relief)
    if not free and not bc.fixed.any() and not bc.pinned:
        return None
    lbl, nb = ndimage.label(solid)
    ijk = hm.elem_ijk
    if free:
        # free-free: the largest body (loose crumbs would add zero modes)
        sizes = np.bincount(lbl[tuple(ijk.T)], minlength=nb + 1)
        sizes[0] = 0
        held = np.array([int(np.argmax(sizes))])
    else:
        held = np.unique(lbl[tuple(ijk[bc.support_nodes[hm.enode].any(axis=1)].T)])
    keep = np.isin(lbl[tuple(ijk.T)], held)
    if not keep.all():
        hm = members_on(HexMesh(ijk[keep], pm, origin))
        bc = boundary_conditions(study, hm, check_loads=False, tol_floor=0.5 * pm)
    fill = share[tuple(hm.elem_ijk.T)]
    rho = float(mat.density) * fill
    if stock and getattr(hm, "stock_density", None) is not None:
        rho = rho + float(mat.density) * hm.stock_density
    masses = []
    for i, (region, shape, kg) in enumerate(vib.masses):
        nodes = np.flatnonzero(_region_nodes(study, hm, region, shape, f"vibration.masses[{i}]", tol_floor=pm))
        masses.append((nodes, kg))
    springs = mat.E * bc.springs if bc.springs is not None else None
    if free:
        # a free part also carries inertiaRelief.masses, unless the vibration
        # block names them itself
        named = {(r, id(sh)) for r, sh, _ in vib.masses}
        for i, (region, shape, kg) in enumerate(getattr(study, "inertia_masses", []) or []):
            if (region, id(shape)) in named or (region is not None and region in {r for r, _, _ in vib.masses}):
                continue
            nodes = np.flatnonzero(_region_nodes(study, hm, region, shape, f"inertiaRelief.masses[{i}]",
                                                 tol_floor=pm))
            masses.append((nodes, kg))
    blocks = []
    if free and getattr(study, "inertia_bodies", None):
        from .domain import carried_bodies
        from .fem import rigid_body_modes

        to_t = float(study.material.density) * 1e-6 * 1e-3       # mm^3 of material -> tonnes
        for dofs, dmat, mc in carried_bodies(study, hm, rigid_body_modes(hm.node_xyz), tol_floor=pm):
            blocks.append((dofs, dmat @ (mc * to_t) @ dmat.T))
    # Heat (the study's thermal block): the part's own temperature on this
    # grid, each cell's modulus at it (a PA12 motor mount at 80 C keeps 45%
    # of its stiffness and rings a third lower than the cold part)
    moduli, hot = float(mat.E) * fill, None
    if stock and getattr(hm, "stock_scale", None) is not None:
        moduli = moduli * hm.stock_scale
    if study.thermal is not None:
        from .thermal import StudyHeat

        heat = StudyHeat(study, hm, study.elastic(mat), bc.fixed, modulus=mat.E)
        heat.temperature()
        if heat.curve is not None:
            moduli = moduli * heat.g
            hot = {"maxTemperature": round(float(heat.t_node.max()), 1), "minRetention": round(float(heat.g.min()), 3)}
    from .modal import band_response, loss_factor

    eta = loss_factor(mat)
    tops = [k[1] for k in (band_keepout_(ex, eta) for ex in vib.excitations) if k is not None]
    n_modes = vib.modes if not tops else max(vib.modes, 12)
    while True:
        res = modal_analysis(hm, moduli, rho, float(mat.nu), bc.fixed, n_modes,
                             masses,
                             springs=springs if (stock or not free) else None, free=free, mass_blocks=blocks)
        # modes up to the top of the highest band: a mode count that stops
        # short of it has not shown the band clear
        if not tops or res.frequencies[-1] >= max(tops) or n_modes >= MODAL_MODES_MAX \
                or len(res.frequencies) < n_modes:
            break
        n_modes = min(MODAL_MODES_MAX, 2 * n_modes)
    f1 = float(res.frequencies[0])
    out = {**res.to_dict(), "minFrequency": vib.min_frequency, "firstFrequency": round(f1, 2),
           "ok": bool(vib.min_frequency is None or f1 >= vib.min_frequency), "pitch": pm, "cells": int(hm.n_elem),
           "carried": round(sum(kg for _, kg in masses) + sum(b[2] for b in (getattr(study, "inertia_bodies", None)
                                                                              or []) if free), 5)}
    if hot is not None:
        out["thermal"] = hot
    if vib.excitations:
        rows = band_response(res.frequencies, vib.excitations, eta)
        out["excitations"] = rows
        out["lossFactor"] = eta
        out["ok"] = bool(out["ok"] and all(r["ok"] for r in rows))
    out["seconds"] = round(time.time() - t0, 2)
    if free:
        out["free"] = True
    # which way the first mode moves most: what to brace
    part = res.participation[0]
    out["firstModeAxis"] = "xyz"[int(np.argmax(part))] if part.sum() > 0.05 else "rotation"
    return out


def _thin_preserves(study: Study, pitch: float) -> dict:
    """Preserved walls and decks under _THIN_CELLS analysis cells through,
    when the study limits how far the part moves (maxDeflection, clearances):
    {"thinPreserves": [{id, thickness, cells}], "warnings": [...]}, or {}.
    The thickness is the median across the preserve's medial ridge, on a
    grid a quarter of the analysis cell (at most ~2M points): a deck with
    a few thick bosses still reads as the deck."""
    g = study.goal
    if g.max_deflection is None and not getattr(g, "clearances", None):
        return {}
    rows = []
    for r in study.preserve:
        try:
            lo, hi = r.shape.bounds()
        except (NotImplementedError, AttributeError):
            continue
        lo, hi = np.asarray(lo, float), np.asarray(hi, float)
        size = hi - lo
        if not np.all(size > 0):
            continue
        h = max(pitch / 4.0, (float(np.prod(size)) / 2e6) ** (1.0 / 3.0))
        n = np.maximum(np.ceil(size / h).astype(int), 1)
        axes = [lo[i] + h * (np.arange(n[i]) + 0.5) for i in range(3)]
        pts = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
        inside = (r.shape.sdf(pts) <= 0).reshape(n)
        if inside.sum() < 8:
            continue
        edt = ndimage.distance_transform_edt(np.pad(inside, 1))[1:-1, 1:-1, 1:-1]
        ridge = inside & (edt >= ndimage.maximum_filter(edt, size=3)) & (edt > 0)
        if not ridge.any():
            continue
        t = float(np.median(2.0 * edt[ridge] - 0.5)) * h
        if t < _THIN_CELLS * pitch:
            rows.append({"id": r.id, "thickness": round(t, 2), "cells": round(t / pitch, 2)})
    if not rows:
        return {}
    warn = [f"preserve {q['id']!r} is about {q['thickness']:g} mm thick, {q['cells']:g} analysis cells of "
            f"{pitch:.2f} mm: a wall or deck under {_THIN_CELLS:g} cells reads stiff in bending (a 3 mm deck at "
            "2 cells sagged 0.6x what tets read), so its deflection and travel read low. Check it on a finer grid "
            "(`loadpath verify --cells`) or on tets (`loadpath crosscheck`)" for q in rows]
    return {"thinPreserves": rows, "warnings": warn}


def _clearance_nodes(study: Study, hm, pitch: float, report: dict) -> list:
    """Per goal.clearances entry, the analysis nodes it covers (an empty
    array, with a warning, when the part does not reach it)."""
    from .domain import _region_nodes

    out = []
    for i, c in enumerate(getattr(study.goal, "clearances", None) or []):
        try:
            out.append(np.flatnonzero(_region_nodes(study, hm, c.region, c.shape, f"goal.clearances[{i}]",
                                                    tol_floor=pitch)))
        except ValueError as exc:
            out.append(np.zeros(0, int))
            report.setdefault("warnings", []).append(str(exc))
    return out


def _clearance_rows(study: Study, case_name: str, xyz, disp, nodes_per) -> list:
    """cases[].clearances: per goal.clearances entry that applies to the
    case, the most any of its nodes travels (along its direction, when it
    has one), against its limit."""
    rows = []
    for c, nodes in zip(getattr(study.goal, "clearances", None) or [], nodes_per):
        if c.cases and case_name not in c.cases:
            continue
        if not len(nodes):
            rows.append({"name": c.name, "travel": None, "maxTravel": c.max_travel})
            continue
        d = disp[nodes]
        t = d @ c.direction if c.direction is not None else np.linalg.norm(d, axis=1)
        k = int(np.argmax(t))
        rows.append({"name": c.name, "travel": max(float(t[k]), 0.0), "maxTravel": c.max_travel,
                     "at": [round(float(v), 3) for v in xyz[nodes[k]]]})
    return rows


def _joined_through_faces(mesh, pitch, solid, origin):
    """Whole cells (half full or more) of a wall about a cell thick and
    curved across the grid meet only along edges: hinges the solve cannot
    hold, so verification kept one sliver and dropped the rest (a drone
    frame's 1.2 mm ducts at 1.15 mm cells, its crash load then on nothing).
    Where that happens the cell the part fills most joins them
    (domain._face_bridge), as cut cells do by keeping every cell it enters."""
    from .domain import _face_bridge

    six = ndimage.label(solid)[1]
    if six <= ndimage.label(solid, structure=np.ones((3, 3, 3)))[1]:
        return solid
    share, o2 = voxelize(mesh, pitch, fractional=True)
    if share.shape != solid.shape or not np.allclose(o2, origin):
        return solid
    return _face_bridge(solid.reshape(-1), share.reshape(-1), solid.shape).reshape(solid.shape)


def _analysis_grid(study: Study, mesh, vol: float, max_cells: int, min_pitch=None, contact_band=None,
                   cut_cells=None, fractional: bool = False, nu=None, max_pitch=None):
    """The grid a finished part is analyzed on (verify_part's, and the
    across-layer readings behind orient.rank_orientations, so the two read
    the same part the same way): the pitch, the cells (cut cells unless the
    study turns them off), the boundary conditions, pieces no support holds
    left out, and the contact band stress is read beyond. `nu` is the
    elasticity the cut cells are built for (default study.elastic)."""
    from types import SimpleNamespace

    mat = study.material
    stock = bool(getattr(study, "stock", None))
    if stock:
        # bought members (stock.py): the grid is cut from the printed part and the members as one solid,
        # so their cells are there to carry their stiffness (boundary_conditions adds it)
        from .stock import assembly_mesh
        mesh = assembly_mesh(study, mesh)
        vol = float(mesh.volume)
    nu_grid = study.elastic(mat) if nu is None else nu

    def members_on(m, fill_el, keep=None):
        if stock:
            m.stock_fill = None if fill_el is None else np.asarray(fill_el, float)
            m.stock_elastic = nu_grid
        return m

    pitch = (vol / max_cells) ** (1.0 / 3.0)
    if min_pitch:
        pitch = max(pitch, min_pitch)
    if max_pitch and pitch > max_pitch:
        # no coarser than the grid the part was designed on, within a budget
        pitch = max(max_pitch, (vol / (_MAX_PITCH_STRETCH * max_cells)) ** (1.0 / 3.0))
    if study.shells:
        # A wall voxelizes to a whole number of cells, so at two or three
        # cells through it reads up to a cell thick or thin: bending stress
        # goes as 1/t^2, and a plenum's 2 mm wall read 2.4 mm and 30% under
        # its stress at 0.8 mm cells. Four cells through the thinnest wall or
        # skin, within a cell budget.
        thin = min(t for _, t in study.shells)
        pitch = min(pitch, max(thin / _SHELL_CELLS, (vol / _SHELL_MAX_CELLS) ** (1.0 / 3.0)))
    share = None
    cut = use_cut_cells(study, cut_cells) and not fractional
    cg = None
    if cut:
        from .cutcell import cut_from_mesh

        cg = cut_from_mesh(mesh, pitch, study.elastic(mat) if nu is None else nu)
        solid, origin = cg.solid_grid()
        share = np.zeros(solid.shape)
        share[tuple(cg.mesh.elem_ijk.T)] = cg.fill
    elif fractional:
        share, origin = voxelize(mesh, pitch, fractional=True)
        # a sliver under a twentieth of a cell adds nothing but conditioning
        solid = share >= 0.05
    else:
        solid, origin = voxelize(mesh, pitch)
        solid = _joined_through_faces(mesh, pitch, solid, origin)

    # Face connectivity: cells that meet only at an edge or a corner share
    # two nodes or one, a hinge the stiffness matrix cannot hold, and CG
    # never converges (the robot arm's upper arm: 44 such islands of 4-6 cells at a
    # 1.04 mm pitch stalled verification for over an hour).
    lbl, nlab = ndimage.label(solid)
    elem_ijk = np.argwhere(solid)
    hm = members_on(HexMesh(elem_ijk, pitch, origin), None if share is None else share[tuple(elem_ijk.T)])
    bc = boundary_conditions(study, hm, check_loads=False, tol_floor=0.5 * (contact_band or 0.0) + 0.5 * pitch)
    # Material not connected to a support would float; it is reported as a
    # separate body and left out of the stress analysis.
    supported_lbls = np.unique(lbl[tuple(elem_ijk[bc.support_nodes[hm.enode].any(axis=1)].T)])
    if study.inertia_relief and nlab > 1:
        # a free part is held by three nodes it is free to put anywhere: the
        # part is its largest body, as on the design grid (domain), not
        # whichever piece the hold happened to land on (a drone frame's
        # verify analyzed an 884-cell sliver of one duct)
        sizes = np.bincount(lbl[tuple(elem_ijk.T)], minlength=nlab + 1)
        sizes[0] = 0
        supported_lbls = np.array([int(np.argmax(sizes))])
    keep_el = np.isin(lbl[tuple(elem_ijk.T)], supported_lbls)
    if not keep_el.all():
        hm = members_on(HexMesh(elem_ijk[keep_el], pitch, origin),
                        None if share is None else share[tuple(elem_ijk[keep_el].T)], keep_el)
        bc = boundary_conditions(study, hm, check_loads=False, tol_floor=0.5 * (contact_band or 0.0) + 0.5 * pitch)
    cut_count = None
    if cg is not None:
        cut_count = int(cg.cut.sum())
        cg = cg.restrict(keep_el)
        hm = cg.mesh
    kept_ijk = elem_ijk if keep_el.all() else elem_ijk[keep_el]
    fill = None if share is None else share[tuple(kept_ijk.T)]
    band = max(1.5 * pitch, contact_band or 0.0)
    edges = _user_edges(study, band)
    on_edge = edges(hm.node_xyz) if edges is not None else None
    stress_mask, printed_solid = None, None
    if stock:
        from .stock import stock_cells
        sc = stock_cells(study, members_on(hm, fill))
        # the cells with printed material in them: the rest are a member's alone
        stress_mask = (sc.owner < 0) | (sc.printed > 0.05)
        if cg is not None:
            cg.stress_mask = stress_mask
        printed_solid = np.zeros(solid.shape, bool)
        printed_solid[tuple(hm.elem_ijk[stress_mask].T)] = True
    return SimpleNamespace(pitch=pitch, cg=cg, solid=solid, origin=origin, share=share, hm=hm, bc=bc, fill=fill,
                           band=band, edges=edges, on_edge=on_edge, cut_count=cut_count, keep_el=keep_el,
                           stress_mask=stress_mask, printed_solid=printed_solid)


def _solve_grid(study: Study, g, nu, moduli, rhs):
    """Solve the analysis grid `g` (_analysis_grid) with elasticity `nu` for
    each right-hand side: (stiffness object for nodal_stress, displacements)."""
    from .simp import _use_mg

    mat = study.material
    hm, bc, fixed = g.hm, g.bc, g.bc.fixed
    scale = getattr(hm, "stock_scale", None)
    if scale is not None:
        # a bought member's cells hold only their printed share of the modulus (stock.py)
        moduli = np.broadcast_to(moduli, (hm.n_elem,)) * scale
    if g.cg is not None:
        return _solve_cut(g.cg, nu, fixed, bc.springs, mat.E, moduli, rhs)
    if g.fill is not None:
        # a part-filled cell is that share as stiff; its stress stays
        # the material's (nodal_stress at the full modulus)
        moduli = np.broadcast_to(moduli, (hm.n_elem,)) * g.fill
    if _use_mg(hm.n_elem):
        # From simp.MG_MIN_CELLS up, matrix-free multigrid (as the optimizer
        # uses): no assembled matrix, a fraction of the memory and time. It
        # raises RuntimeError on non-convergence like Solver does. It also
        # carries .mesh and .nu, which is all the stress recovery reads.
        from .mg import GridElasticity

        stiff = GridElasticity(hm, nu, fixed, springs=mat.E * bc.springs if bc.springs is not None else None,
                               rtol=1e-6, maxiter=3000)
        stiff.set_moduli(np.broadcast_to(moduli, (hm.n_elem,)).astype(float))
        stiff.stress_mask = getattr(g, "stress_mask", None)
        return stiff, stiff.solve_many(rhs)
    stiff = Stiffness(hm, nu, fixed)
    stiff.stress_mask = getattr(g, "stress_mask", None)
    k = stiff.assemble(np.broadcast_to(moduli, (hm.n_elem,)).astype(float))
    if bc.springs is not None:
        k = (k + mat.E * bc.springs).tocsr()
    # 1e-6 is three-plus significant figures of stress; 1e-7 cost a third
    # more iterations. All cases share one multigrid hierarchy and run on
    # threads.
    solver = Solver(hm, rtol=1e-6, maxiter=3000, rebuild_after_iters=10_000)
    return stiff, solver.solve_many(k, rhs)


def _layer_bond(sig, n, body, g_node=None, shear_ratio=float(np.sqrt(3.0))):
    """What loads the bond between layers stacked along unit `n`, at its
    worst node of `body`: the tension pulling the layers apart and the shear
    sliding them, combined (failure.bond_stress; read against the bond's
    tensile strength). Returns (stress, strength retention there, node,
    the tension there, the shear there). With g_node (hot parts) worst means
    stress over retention."""
    from .failure import bond_stress

    eq, tens, tau = bond_stress(sig, n, shear_ratio)
    eq = np.where(body, eq, 0.0)
    ti = int(np.argmax(eq if g_node is None else eq / np.maximum(g_node, 1e-3)))
    return float(eq[ti]), (1.0 if g_node is None else float(g_node[ti])), ti, float(tens[ti]), float(tau[ti])


def interlayer_readings(study: Study, mesh, builds, max_cells: int = 60_000, min_pitch: float | None = None,
                        contact_band: float | None = None, cut_cells: bool | None = None) -> list[dict]:
    """The across-layer bond check verify_part makes for the study's own
    build direction, made for each of `builds` (build directions) on the
    same analysis: the same grid, cut cells, contact band and load
    directions, with the layered stiffness of each direction (one solve per
    distinct layer axis: +z and -z stack the same layers). Per direction:
    {"interlayerStress": worst over the cases (MPa), "interlayerSafetyFactor",
    "cases": [{name, interlayerStress, at}]}. The study's own direction
    reads exactly what its verification reads (thermal studies excepted:
    this is the cold, mechanical reading).

    orient.rank_orientations used to read this from one isotropic solve on
    a plain voxel grid with its own, narrower contact band: on a coarse
    wall bracket that read the stair-stepped corner where an arm meets a
    screw boss at 9.0 MPa across the layers (SF 1.7 printed flat) against
    verify's 4.1 MPa (SF 3.9) and the tet cross-check's 2.7 MPa, and
    recommended tilting a part that holds flat."""
    import dataclasses

    from .am import build_vector, parse_build_direction
    from .directions import expand, worst_stress_direction
    from .shapes import manifold_from_trimesh

    builds = [parse_build_direction(b) if isinstance(b, (str, list)) else b for b in builds]
    mat = study.material
    proc = study.process
    vol = float(manifold_from_trimesh(mesh).volume())
    vecs = [np.asarray(build_vector(b), float) for b in builds]

    def elastic_for(b):
        if proc is None:
            return mat.nu
        return dataclasses.replace(study, process=dataclasses.replace(
            proc, build=b)).elastic(mat)

    from .failure import bond_shear_ratio
    shear_ratio = bond_shear_ratio(proc) if proc is not None else float(np.sqrt(3.0))
    g = _analysis_grid(study, mesh, vol, max_cells, min_pitch, contact_band, cut_cells, nu=mat.nu)
    hm, bc = g.hm, g.bc
    fg = gravity_force(hm, np.ones(hm.n_elem) if g.fill is None else g.fill, bc.gravity, bc.fixed) \
        if bc.gravity is not None else None
    spreads = bc.spreads or [None] * len(study.load_cases)
    rhs, layout = expand(bc.loads, spreads, fg)
    from scipy.spatial import cKDTree

    bodies = []
    for ci in range(len(study.load_cases)):
        contact = bc.support_nodes | bc.case_load_nodes[ci]
        near = cKDTree(hm.node_xyz[contact]).query(hm.node_xyz, k=1, distance_upper_bound=g.band)[0] < np.inf
        if g.on_edge is not None:
            near |= g.on_edge
        bodies.append(~near)
    solved = {}
    out = []
    for b, n in zip(builds, vecs):
        nu = elastic_for(b)
        key = ("layered", tuple(np.round(n * np.sign(n[np.argmax(np.abs(n))]), 9)),
               nu.ez, nu.gz) if not isinstance(nu, float) else ("iso",)
        if key not in solved:
            gg = g
            if g.cg is not None and not isinstance(nu, float):
                # the cut cells' matrices carry the elasticity: rebuild them
                # for this direction's layers on the same sub-cell grid
                from types import SimpleNamespace

                from .cutcell import CutGrid
                c0 = g.cg
                cg = CutGrid(c0._fine, c0.n, c0.origin, c0.pitch, nu).restrict(g.keep_el)
                gg = SimpleNamespace(**{**vars(g), "cg": cg})
            solved[key] = _solve_grid(study, gg, nu, mat.E, rhs)
        stiff, disp = solved[key]
        cases, worst = [], 0.0
        for ci, case in enumerate(study.load_cases):
            idx = layout[ci]
            if len(idx) == 1:
                u = disp[idx[0]]
            else:
                s0 = nodal_stress(stiff, disp[idx[0]], mat.E) if idx[0] is not None else None
                sb = [nodal_stress(stiff, disp[i], mat.E) for i in idx[1:]]
                direction, _ = worst_stress_direction(spreads[ci], s0, sb, bodies[ci])
                u = sum(direction[j] * disp[i] for j, i in enumerate(idx[1:]))
                if idx[0] is not None:
                    u = u + disp[idx[0]]
            t, _, ti, tn, ts = _layer_bond(nodal_stress(stiff, u, mat.E), n, bodies[ci], None, shear_ratio)
            cases.append({"name": case.name, "interlayerStress": t, "interlayerTension": tn,
                          "interlayerShear": ts, "at": hm.node_xyz[ti].tolist()})
            worst = max(worst, t)
        ratio = proc.interlayer if proc is not None else 1.0
        out.append({"interlayerStress": worst, "cases": cases,
                    "interlayerSafetyFactor": (ratio * mat.strength / (worst * STRESS_ALLOWANCE)
                                               if worst > 0 else float("inf"))})
    return out


def _undeclared_layers(study):
    """(process kind, bond strength ratio) when the study names no build
    direction but its material is made by a layered process whose bonds are
    weaker than the layers; None when the direction is given (the bond is
    checked for it) or nothing says the part is layered."""
    from .spec import PROCESSES

    proc = study.process
    if proc is not None and proc.build not in (None, "auto"):
        return None
    mat = study.material
    kind = proc.kind if proc is not None else (mat.processes[0] if mat.processes else None)
    if kind not in PROCESSES:
        return None
    ratio = proc.interlayer if proc is not None else PROCESSES[kind]["interlayer"]
    if proc is None and mat.measured == kind and mat.across_layers()[0] is not None:
        ratio = mat.across_layers()[0]
    return (kind, float(ratio)) if ratio < 1.0 else None


def _internal_force(stiff, hm, u, moduli, fill=None, cg=None, el=None):
    """K u per dof from the grid's own cells (`el`: only these), without the
    springs: what the material's stress puts on each node."""
    n_el = hm.n_elem
    el = np.arange(n_el) if el is None else el
    mod = np.broadcast_to(np.asarray(moduli, float), (n_el,))[el].copy()
    if fill is not None and cg is None:
        mod *= fill[el]
    scale = getattr(hm, "stock_scale", None)
    if scale is not None:
        mod *= scale[el]
    edof = hm.edof[el]
    ue = u[edof]
    fe = (ue @ stiff.ke) * mod[:, None]
    if cg is not None and len(cg.cut_index):
        pos = np.full(n_el, -1, np.int64)
        pos[cg.cut_index] = np.arange(len(cg.cut_index))
        k = pos[el]
        cut = k >= 0
        if cut.any():
            fe[cut] = np.einsum("nij,nj->ni", cg.ke_cut[k[cut]], ue[cut]) * mod[cut, None]
    return np.bincount(edof.ravel(), weights=fe.ravel(), minlength=hm.n_dof)


def support_reactions(stiff, hm, u, f, moduli, nodes_of, fill=None, cg=None, extra=None):
    """What each support carries in one solved case: per support (node mask
    in `nodes_of`) the resultant force the part puts on it (N) and the
    moment of that force about the support's centroid (N·mm). Read as the
    load less the part's own internal force, f - K u, at the support's
    nodes, which is the clamp's reaction on the held dofs and a pin's
    spring force on the rest (so pins and partly fixed supports read right,
    and the supports' forces sum to the applied load). Only the cells
    touching a support are visited. `extra`: stiffness the cells do not
    hold that is the part's own all the same (a bought member's, stock.py:
    sparse, N/mm)."""
    held = np.any(nodes_of, axis=0) if len(nodes_of) else np.zeros(hm.n_nodes, bool)
    el = np.nonzero(held[hm.enode].any(axis=1))[0]
    ku = _internal_force(stiff, hm, u, moduli, fill, cg, el)
    if extra is not None:
        ku = ku + extra @ u
    carried = (f - ku).reshape(-1, 3)
    xyz = hm.node_xyz
    out, taken = [], np.zeros(hm.n_nodes, bool)
    for nodes in nodes_of:
        own = nodes & ~taken           # a node two supports share counts once
        taken |= nodes
        fn = carried[own]
        c = xyz[nodes].mean(axis=0) if nodes.any() else np.zeros(3)
        F = fn.sum(axis=0)
        M = np.cross(xyz[own] - c, fn).sum(axis=0)
        out.append((F, M, c))
    return out


def _reaction_report(study, rows):
    """report["reactions"] from per-support, per-case (force, moment,
    centroid) rows: the worst resultant of each support, its share of the
    total, and its cap when the study sets one."""
    out = []
    for i, s in enumerate(study.supports):
        cases = [{"name": name, "force": [float(v) for v in r[i][0]], "magnitude": float(np.linalg.norm(r[i][0])),
                  "moment": [float(v) for v in r[i][1]], "momentMagnitude": float(np.linalg.norm(r[i][1]))}
                 for name, r in rows]
        worst = max((c["magnitude"] for c in cases), default=0.0)
        worst_m = max((c["momentMagnitude"] for c in cases), default=0.0)
        row = {"support": s.region or f"supports[{i}]", "at": [float(v) for v in rows[0][1][i][2]] if rows else None,
               "maxForce": worst, "maxMoment": worst_m, "cases": cases}
        if s.max_force is not None:
            row["forceLimit"] = s.max_force
        if s.max_moment is not None:
            row["momentLimit"] = s.max_moment
        out.append(row)
    for name_i, (name, r) in enumerate(rows):
        tot = sum(np.linalg.norm(x[0]) for x in r)
        for row in out:
            c = row["cases"][name_i]
            c["share"] = c["magnitude"] / tot if tot > 0 else 0.0
    return out


def reaction_margin(reactions) -> float | None:
    """The least of limit / carried over the supports that have a cap."""
    ms = []
    for r in reactions or []:
        if r.get("forceLimit"):
            ms.append(r["forceLimit"] / r["maxForce"] if r["maxForce"] > 0 else float("inf"))
        if r.get("momentLimit"):
            ms.append(r["momentLimit"] / r["maxMoment"] if r["maxMoment"] > 0 else float("inf"))
    return min(ms) if ms else None


def verify_part(study: Study, mesh, *args, **kwargs) -> dict:
    """verify_part_ with the study's impact cases at their coupled forces
    for this part only (study.impact_scale is cleared before and after)."""
    study.impact_scale = {}
    try:
        return _verify_part(study, mesh, *args, **kwargs)
    finally:
        study.impact_scale = {}


def _verify_part(study: Study, mesh, max_cells: int = 60_000, min_pitch: float | None = None,
                 contact_band: float | None = None, refine: int = 3, fields: bool = False,
                 fractional: bool = False, cut_cells: bool | None = None, max_pitch: float | None = None) -> dict:
    """Analyze a finished part (trimesh) against the study. Returns a JSON-able
    report. `contact_band` (mm) is how far from applied loads and supports
    stress is left out of the peak; it defaults to 1.5 analysis cells, and the
    study runner sets it to the optimization cell size so the excluded band
    does not shrink toward zero as the analysis grid gets finer (a clamped
    edge's stress grows without bound under refinement). `refine` is how many
    hotspots per load case get a 3x finer submodel (see _submodel_peak).
    `fields` adds report["fields"]: the analysis nodes and, per load case,
    nodal von Mises stress and displacement (numpy arrays, not JSON) for
    stress maps (render.render_stress); with a thermal block, also the
    temperature per node at the hot moment. `fractional` analyzes the part's
    surface cells at their filled share of the modulus (the stress formed at
    the full modulus), so the readings move smoothly with the geometry
    rather than in whole-cell steps: for searches and live previews, whose
    neighbouring candidates must compare fairly; the delivered check is
    whole cells. `cut_cells` (default: use_cut_cells, which is on unless
    the study says resolution.cutCells false, LOADPATH_CUT_CELLS=0, or
    it has a pressure load) analyzes the true boundary inside each cell
    (cutcell.py): sub-cell stiffness, stress read on the surface, and
    finer submodels on members under 3 cells thick; member thickness no
    longer rounds to whole cells (the robot arm's wrist link struts: whole cells
    0.35-3.5x of fine tets, cut cells 0.74-1.13x; docs/BENCHMARKS.md)."""
    from .shapes import manifold_from_trimesh

    t0 = time.time()
    mat = study.material
    part = manifold_from_trimesh(mesh)
    vol = float(part.volume())
    report = {"volume": vol, "mass": vol * mat.density / 1000.0, "material": mat.to_dict()}
    try:
        report["massProperties"] = mass_properties(mesh, mat.density, getattr(study, "mass_properties", None))
        mpr = report["massProperties"]
        if mpr.get("centerOfMassOk") is False:
            report.setdefault("warnings", []).append(
                f"centre of mass {mpr['centerOfMass']} is {mpr['centerOfMassOffset']} mm off its target "
                f"(tolerance {mpr['centerOfMassTolerance']} mm)")
    except (ValueError, IndexError, FloatingPointError) as exc:
        report.setdefault("warnings", []).append(f"mass properties failed: {exc}")

    aero = getattr(study, "aero", None)
    if aero is not None:
        # the delivered part in each rotor's stream (aero.rotor_blockage), and
        # what that drag is worth in carried mass: the search weighs outcomes
        # by mass plus it
        from .aero import rotor_blockage

        rows = [{"name": r.name, **rotor_blockage(mesh, r.center, r.axis, r.radius, r.thrust, r.hub, r.cd, r.rho,
                                                  r.reach, float(np.clip(r.radius / 90.0, 0.25, 1.0)), r.ducted)}
                for r in aero.rotors]
        drag = float(sum(r["drag"] for r in rows))
        report["aero"] = {"rotors": rows, "drag": round(drag, 5),
                          "equivalentMass": round(drag * aero.grams_per_newton, 3),
                          "gramsPerNewton": round(aero.grams_per_newton, 2)}
        report["effectiveMass"] = report["mass"] + drag * aero.grams_per_newton
        # what each drawn duct gives (thrust over the same rotor in the open
        # at equal induced power: a credit in the same currency) and, at a
        # declared cruise speed, costs
        from .aero import duct_performance

        gained = cruise_drag = 0.0
        for r, row in zip(aero.rotors, rows):
            if r.duct is None:
                continue
            d = r.duct
            perf = duct_performance(r.radius, r.thrust, d.get("tipGap", 0.0), d.get("lipRadius", 0.0),
                                    d.get("lengthBelow", 0.0), d.get("diffuserAngle", 0.0), d.get("outerRadius"),
                                    d.get("lengthAbove", 0.0) + d.get("lengthBelow", 0.0), r.hub, r.rho,
                                    cruise=aero.cruise_speed)
            row["duct"] = {k: ({kk: round(vv, 4) for kk, vv in v.items()} if isinstance(v, dict) else round(v, 4))
                           for k, v in perf.items()}
            gained += perf["thrustGained"]
            cruise_drag += perf.get("drag", 0.0)
        if any(r.duct is not None for r in aero.rotors):
            report["aero"]["ductThrustGained"] = round(gained, 5)
            report["aero"]["ductEquivalentMass"] = round(-gained * aero.grams_per_newton, 3)
            report["effectiveMass"] -= gained * aero.grams_per_newton
            if aero.cruise_speed is not None:
                # level flight: thrust holds the weight and the drag at right angles
                hover = float(sum(r.thrust for r in aero.rotors))
                extra = float(np.hypot(hover, cruise_drag) - hover)
                report["aero"]["cruise"] = {"speed": aero.cruise_speed, "ductDrag": round(cruise_drag, 4),
                                            "extraThrust": round(extra, 4),
                                            "equivalentMass": round(extra * aero.grams_per_newton, 3)}

    design = _union(study.design_space + [r.shape for r in study.preserve])
    keep = _union([r.shape for r in study.keep_out])
    envelope = design - keep if keep is not None else design
    env_vol = float(envelope.volume())
    report["designSpaceMass"] = env_vol * mat.density / 1000.0
    report["massReduction"] = 1.0 - vol / env_vol if env_vol > 0 else None
    # What must survive is each preserve less the keep-outs that cut it (a
    # boss keeps its bore; rails stand in a board's clear zone, the skin
    # under a part's way in). Anything else of the part in a keep-out
    # intrudes.
    pres_eff = drawn_kept(study)
    designed = part - pres_eff if pres_eff is not None else part
    report["keepOutIntrusion"] = float((designed ^ keep).volume()) if keep is not None else 0.0
    if pres_eff is not None:
        report["preserveMissing"] = _preserve_missing(pres_eff, part, mesh)
    else:
        report["preserveMissing"] = 0.0
    report["outsideDesignSpace"] = float((part - design).volume())
    report["boundaryCutArea"] = _boundary_cut_area(study, mesh)
    report.update(_creases(study, mesh))
    report["closedBores"] = _closed_bores(study, mesh)
    if getattr(study, "fits", None):
        # bearing seats, pins and bolt holes: the hole in the file, the band
        # it prints to, and whether the fit holds (fits.py)
        from .fits import fits_report
        report["fits"] = fits_report(study, mesh)
    # decompose() lists each enclosed cavity too, as a negative volume
    pieces = [b.volume() for b in part.decompose()]
    report["bodies"] = sum(1 for v in pieces if v > 0)
    stock_on = bool(getattr(study, "stock", None))
    if stock_on:
        # printed pieces that only the bought members join (two nodes on one tube) are one assembly:
        # the count that has to be 1 is the assembly's
        from .stock import assembly_mesh, stock_mass

        report["printedBodies"] = report["bodies"]
        report["bodies"] = sum(1 for b in manifold_from_trimesh(assembly_mesh(study, mesh)).decompose()
                               if b.volume() > 0)
        report["stockMass"] = stock_mass(study)
        report["assemblyMass"] = report["mass"] + report["stockMass"]
        # the submodels re-solve a box of printed material alone: skipped, as with heat
        refine = 0
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
                                      material=mat, bridge_span=getattr(proc, "bridge_span", None))

    g = _analysis_grid(study, mesh, vol, max_cells, min_pitch, contact_band, cut_cells, fractional,
                       max_pitch=max_pitch)
    pitch, cg, solid, origin, share = g.pitch, g.cg, g.solid, g.origin, g.share
    hm, bc, fill, band, edges, on_edge = g.hm, g.bc, g.fill, g.band, g.edges, g.on_edge
    support_nodes, fixed = bc.support_nodes, bc.fixed
    if cg is not None:
        report["cutCells"] = g.cut_count
    report["analysisPitch"] = pitch
    if max_pitch and pitch > 1.01 * max_pitch:
        report.setdefault("warnings", []).append(
            f"analysed at {pitch:.2f} mm cells, coarser than the {max_pitch:.2f} mm grid it was designed on (the "
            f"cell budget, {max_cells} x {_MAX_PITCH_STRETCH:g}): thin walls read stiff and their stress low "
            "there; `loadpath verify --cells` a finer grid, or `loadpath crosscheck` on tets")
    thin = _thin_preserves(study, pitch)
    if thin:
        report["thinPreserves"] = thin["thinPreserves"]
        report.setdefault("warnings", []).extend(thin["warnings"])
    printed_cells = solid if share is None else share >= 0.5
    if g.printed_solid is not None:
        printed_cells = printed_cells & g.printed_solid
    report["minMember"], report["p5Member"] = _thickness(printed_cells, pitch)
    report["contactBand"] = band
    fg = gravity_force(hm, np.ones(hm.n_elem) if fill is None else fill, bc.gravity, fixed) if bc.gravity is not None else None
    # Heat (the study's thermal block): the part's own temperature, solved on
    # the part as delivered; each cell's modulus derated by it, the
    # expansion its neighbours and the supports stop as a load in every
    # case, and the stress formed from the strain less the free expansion.
    # Strength falls with the modulus (the same retention curve). The
    # submodels re-solve without heat, so they are skipped.
    moduli, eps, g_node = mat.E, None, None
    if study.thermal is not None:
        from .thermal import StudyHeat, retention

        heat = StudyHeat(study, hm, study.elastic(mat), fixed, modulus=mat.E)
        heat.temperature()
        moduli = mat.E * heat.g
        if heat.expands:
            eps = heat.eps
            fth = heat.load(np.ones(hm.n_elem), mat.E)
            fg = fth if fg is None else fg + fth
        g_node = retention(heat.curve, heat.t_node)[0] if heat.curve is not None else np.ones(hm.n_nodes)
        refine = 0
        report["thermal"] = {"maxTemperature": float(heat.t_node.max()),
                             "minRetention": float(heat.g.min()),
                             "expansion": heat.alpha if heat.expands else 0.0,
                             "stressFree": heat.t_ref,
                             "heatIn": float(heat.bc.power),
                             **({"convection": heat.convection} if heat.convection is not None else {})}
        cyc = heat.cycle_result
        if cyc is not None:
            report["thermal"].update(
                maxTemperature=float(cyc.node_max.max()), minTemperature=float(cyc.node_min.min()),
                swing=float((cyc.node_max - cyc.node_min).max()), hotAt=cyc.t_hot, coldAt=cyc.t_cold,
                meanHot=float(cyc.mean.max()), meanCold=float(cyc.mean.min()), periods=cyc.periods,
                converged=cyc.converged, period=heat.cycle.period, on=heat.cycle.on)
    # A force of unknown direction is solved along each axis and judged in
    # the direction that stresses the part most (directions.py).
    from .directions import expand, worst_stress_direction
    from .failure import bond_shear_ratio, compression_ratio, criterion, equivalent_stress
    # what breaks it (failure.py): the pressure-modified von Mises stress
    # against the tensile strength (a polymer is stronger squeezed than
    # pulled), and on layered parts the bond's tension and shear together
    m_c = compression_ratio(mat)[0]
    report["failure"] = criterion(mat, study.process)
    # A part with no build direction is still printed in layers: read what
    # the bond would carry for layers stacked along each axis, from the
    # same stresses (failure block, below)
    layers = _undeclared_layers(study)
    axis_bond = np.zeros(3)
    spreads = bc.spreads or [None] * len(study.load_cases)

    def solve(moduli, rhs):
        return _solve_grid(study, g, study.elastic(mat), moduli, rhs)

    # Impact cases (loadCases[].impact): the force is what stops the energy
    # through the crusher and this part together, springs in series. The
    # part's give at the patch is its own f.u / F^2 under the hit as written
    # (a free part's with inertia relief: its mass and what it carries take
    # part); the case's loads are then scaled to the coupled peak force, here
    # and wherever the study's loads are built again (study.impact_scale).
    impact_rows = {}
    for ci, case in enumerate(study.load_cases):
        if getattr(case, "impact", None) is None:
            continue
        from .domain import impact_energy, patch_acceleration
        from .impact import series_impact

        f0 = bc.loads[ci]
        f_raw = bc.raw_loads[ci] if bc.raw_loads else f0
        fn = float(np.linalg.norm(f_raw.reshape(-1, 3).sum(axis=0)))
        if fn <= 0:
            continue
        u0 = solve(moduli, [f0])[1][0]
        c_f = max(float(f0 @ u0), 0.0) / fn ** 2
        acc = patch_acceleration(study, hm, f_raw, 1.0 if fill is None else fill, 0.5 * pitch) \
            if study.inertia_relief else None
        energy, m_eff = impact_energy(study, case, f_raw, acc)
        ic = case.impact
        r = series_impact(energy, c_f, ic.distance, ic.force)
        rigid = series_impact(energy, 0.0, ic.distance, ic.force) if ic.distance is not None else None
        row = {"energy": energy / 1000.0, "compliance": c_f, "stiffness": 1.0 / c_f if c_f > 0 else None,
               "peakForce": r["peakForce"], "crush": r["crush"], "deflection": r["deflection"],
               "crusherEnergy": r["crusherEnergy"] / 1000.0, "structureEnergy": r["structureEnergy"] / 1000.0,
               "structureShare": r["structureShare"], "bottomedOut": r["bottomedOut"],
               "rigidBackingForce": rigid["peakForce"] if rigid is not None else None,
               "forceAsWritten": fn}
        if m_eff is not None:
            row["mass"] = m_eff
        if ic.velocity is not None:
            # (the time to stop at a steady deceleration: for judging the quasi-static reading
            # against the part's first period)
            row["stopSeconds"] = 2.0 * (r["crush"] + r["deflection"]) * 1e-3 / ic.velocity
        size = float(np.ptp(hm.node_xyz, axis=0).max())
        # past a few percent of the part's size the linear reading stiffens or buckles in ways it cannot show
        row["linear"] = bool(r["deflection"] <= 0.05 * size)
        if not row["linear"]:
            report.setdefault("warnings", []).append(
                f"impact case {case.name!r}: the part gives {r['deflection']:.1f} mm at the patch, over 5% of its "
                f"{size:.0f} mm size; the linear reading of its force and stress is rough")
        if r["peakForce"] is not None:
            sc = r["peakForce"] / fn
            bc.loads[ci] = f0 * sc
            if bc.raw_loads:
                bc.raw_loads[ci] = f_raw * sc
            study.impact_scale[case.name] = sc
        impact_rows[ci] = row
    rhs, layout = expand(bc.loads, spreads, fg)

    # The buckling check (below) needs the stress solve only for a force of
    # unknown direction (its worst direction): otherwise it runs on a
    # thread beside the stress solve (its own grid and factorization, a
    # fifth of a verify on the arm link).
    goal_sf = study.goal.safety_factor
    early_bk = None
    if goal_sf and not study.self_equilibrated and all(len(layout[ci]) == 1 for ci in range(len(study.load_cases))):
        from .pools import pool

        # (pool(2): a real thread only where numba's layer takes two at once;
        # otherwise the check runs here, before the stress solve)
        _bk_pool = pool(2)
        early_bk = _bk_pool.submit(_buckling, study, mesh.copy(), vol, pitch, contact_band, [None] * len(study.load_cases))
        _bk_pool.shutdown(wait=False)
    stiff, displacements = solve(moduli, rhs)
    cold = None
    if study.thermal is not None and heat.cold is not None:
        # The cycle's coldest moment, solved the same way: a part held at a
        # hot stress-free temperature is stressed cold, and its swing is what
        # thermal cycling fatigues (each case keeps its hot worst direction).
        t_hot = heat.t_node
        heat.use(heat.cold)
        mod_c = mat.E * heat.g
        eps_c = heat.eps if heat.expands else None
        g_c = retention(heat.curve, heat.cold)[0] if heat.curve is not None else np.ones(hm.n_nodes)
        fg_c = gravity_force(hm, np.ones(hm.n_elem) if fill is None else fill, bc.gravity, fixed) if bc.gravity is not None else None
        if heat.expands:
            fth_c = heat.load(np.ones(hm.n_elem), mat.E)
            fg_c = fth_c if fg_c is None else fg_c + fth_c
        rhs_c = expand(bc.loads, spreads, fg_c)[0]
        stiff_c, disp_c = solve(mod_c, rhs_c)
        cold = (stiff_c, disp_c, mod_c, eps_c, g_c)
    from scipy.spatial import cKDTree
    cases = []
    worst_sf = np.inf
    worst_dirs = []
    env = None   # von Mises per node, the worst over the cases
    reaction_rows = []
    sup_nodes = bc.regions[:len(study.supports)]
    clear_nodes = _clearance_nodes(study, hm, pitch, report)
    stock_k, stock_cases = None, []
    if stock_on:
        from .stock import stock_cells

        _sc = stock_cells(study, hm)
        stock_k = _sc.operator(None) if _sc is not None else None
    for ci, case in enumerate(study.load_cases):
        load_nodes = bc.case_load_nodes[ci]
        # Exclude nodes within the contact band of an applied load or support.
        contact = support_nodes | load_nodes
        near = cKDTree(hm.node_xyz[contact]).query(hm.node_xyz, k=1, distance_upper_bound=band)[0] < np.inf
        if on_edge is not None:
            near |= on_edge
        body = ~near
        idx = layout[ci]
        direction = None
        if len(idx) == 1:
            f, u = rhs[idx[0]], displacements[idx[0]]
        else:
            sp = spreads[ci]
            # the fixed part carries the expansion, the uncertain force none;
            # stress over retention, so the worst direction is the one
            # nearest failure where hot material is weak
            s0 = nodal_stress(stiff, displacements[idx[0]], moduli, eps) if idx[0] is not None else None
            sb = [nodal_stress(stiff, displacements[i], moduli) for i in idx[1:]]
            if g_node is not None:
                kn = 1.0 / np.maximum(g_node, 1e-3)[:, None]
                s0 = s0 * kn if s0 is not None else None
                sb = [b * kn for b in sb]
            direction, _ = worst_stress_direction(sp, s0, sb, body)
            f = sum(direction[j] * rhs[i] for j, i in enumerate(idx[1:]))
            u = sum(direction[j] * displacements[i] for j, i in enumerate(idx[1:]))
            if idx[0] is not None:
                f, u = f + rhs[idx[0]], u + displacements[idx[0]]
        worst_dirs.append(direction)
        if sup_nodes and not study.self_equilibrated:
            reaction_rows.append((case.name, support_reactions(stiff, hm, u, f, moduli, sup_nodes, fill, cg,
                                                               extra=stock_k)))
        if stock_on:
            stock_cases.append((case.name, u, f))
        sig_u = nodal_stress(stiff, u, moduli, eps)
        vm = equivalent_stress(sig_u, m_c)
        env = vm if env is None else np.maximum(env, vm)
        if bc.case_pressure_nodes[ci].any():
            vm = _smooth_wetted(hm.node_xyz, vm, bc.case_pressure_nodes[ci], pitch)
        # hot material is weaker: the peak is where stress / retention is
        util = vm if g_node is None else vm / np.maximum(g_node, 1e-3)
        pk = int(np.argmax(np.where(body, util, -1))) if body.any() else int(np.argmax(util))
        peak = float(vm[pk])
        at = hm.node_xyz[pk].tolist()
        keep = 1.0 if g_node is None else float(g_node[pk])
        global_peak = peak
        refined = []
        if refine and body.any():
            half = max(8.0 * pitch, 6.0)
            spots = list(_hotspots(hm.node_xyz, vm, body, peak, half, count=refine))
            if cg is not None:
                # members under 3 cells thick read up to ~15% low even cut
                # (two cells across a diagonal strip are too few to bend
                # in): the highest of them get a submodel too
                spots += _thin_spots(hm, solid, origin, pitch, vm, body, peak, half, spots, count=refine)

            def sub(c, u=u, case=case, half=half, contact=contact):
                return _submodel_peak(study, mesh, hm, u, c, half, pitch / 2.5, mat,
                                      hm.node_xyz[contact], band, case, edges=edges, cut=cg is not None,
                                      ratio=m_c)
            if len(spots) > 1:
                # independent boxes, on threads (the solves release the GIL;
                # amg_solver sets up one at a time, so the same numbers)
                from .pools import pool

                with pool(min(len(spots), 3)) as ex:
                    subs = list(ex.map(sub, spots))
            else:
                subs = [sub(c) for c in spots]
            for r in subs:
                if r is None:
                    continue
                refined.append({"at": r[1], "peakStress": r[0]})
                if r[0] > peak:
                    peak, at = r
        disp = u.reshape(-1, 3)
        if fields:
            if "fields" not in report and study.thermal is not None:
                # C per node at the hot moment (the cycle's hottest, or the steady state)
                report["fields"] = {"nodes": hm.node_xyz, "cases": [],
                                    "temperature": t_hot if cold is not None else heat.t_node}
            flds = report.setdefault("fields", {"nodes": hm.node_xyz, "cases": []})
            if "elements" not in flds:
                # the analysis cells themselves (8 node ids each, fem.CORNERS
                # order) and the share of each a cut cell fills
                flds["elements"] = hm.enode
                flds["pitch"] = float(hm.pitch)
                if fill is not None:
                    flds["solidShare"] = np.asarray(fill, np.float32)
            flds["cases"].append(
                {"name": case.name, "vonMises": vm, "displacement": disp, "wetted": bc.case_pressure_nodes[ci],
                 "stress": np.asarray(sig_u, np.float32)})
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
        clear_rows = _clearance_rows(study, case.name, hm.node_xyz,
                                     _rigid_free(hm.node_xyz, u).reshape(-1, 3) if study.self_equilibrated else disp,
                                     clear_nodes)
        if bc.pressure_imbalance[ci] > 0.02 and study.self_equilibrated:
            report.setdefault("warnings", []).append(
                f"{case.name}: the pressure no longer balances on the delivered part "
                f"({bc.pressure_imbalance[ci]:.1%} net): a wall the cavity relies on is open (a leak)")
        # The safety factor carries a discretization allowance: on the
        # plate-with-hole benchmark (tests) refined voxel peaks land within
        # -12%..+13% of Peterson's Kt, depending on how the grid falls on the
        # hole. Under-reading stress is the unsafe direction, so the factor
        # is computed against peak * STRESS_ALLOWANCE.
        sf = keep * mat.strength / (peak * STRESS_ALLOWANCE) if peak > 0 else np.inf
        if layers is not None:
            for ax in range(3):
                b_eq, b_keep = _layer_bond(sig_u, np.eye(3)[ax], body, g_node)[:2]
                axis_bond[ax] = max(axis_bond[ax], b_eq / max(b_keep, 1e-3))
        inter, bond_row = None, {}
        proc = study.process
        if proc is not None and proc.build not in (None, "auto") and proc.interlayer < 1.0:
            # Layered parts part along their layers: the tension pulling
            # layers apart (the stress normal to them) against the bond's
            # strength, a fraction of the in-plane one.
            from .am import build_vector

            inter, k_in, bi, b_tens, b_shear = _layer_bond(sig_u, build_vector(proc.build), body, g_node,
                                                           bond_shear_ratio(proc))
            sf_inter = k_in * proc.interlayer * mat.strength / (inter * STRESS_ALLOWANCE) if inter > 0 else np.inf
            bond_row = {"interlayerTension": b_tens, "interlayerShear": b_shear,
                        "interlayerAt": hm.node_xyz[bi].tolist(), "interlayerSafetyFactor": sf_inter,
                        "limitedBy": "layer bond" if sf_inter < sf else "in-plane"}
            sf = min(sf, sf_inter)
        cyc_row = {}
        if cold is not None:
            stiff_c, disp_c, mod_c, eps_c, g_c = cold
            if len(idx) == 1:
                u_c = disp_c[idx[0]]
            else:
                u_c = sum(direction[j] * disp_c[i] for j, i in enumerate(idx[1:]))
                if idx[0] is not None:
                    u_c = u_c + disp_c[idx[0]]
            s_h = nodal_stress(stiff, u, moduli, eps) / np.maximum(g_node, 1e-3)[:, None]
            s_c = nodal_stress(stiff_c, u_c, mod_c, eps_c) / np.maximum(g_c, 1e-3)[:, None]
            util_c = equivalent_stress(s_c, m_c)
            pk_c = int(np.argmax(np.where(body, util_c, -1))) if body.any() else int(np.argmax(util_c))
            sf_c = mat.strength / (util_c[pk_c] * STRESS_ALLOWANCE) if util_c[pk_c] > 0 else np.inf
            sf = min(sf, sf_c)
            # Thermal-cycling fatigue: durability's design curves read the
            # cycle's maximum stress (which carries the mean-stress effect for
            # R up to ~0.3); a stress that barely moves over the cycle is
            # creep, not fatigue, so above that the swing counts: the
            # equivalent is min(max, range / 0.7). Stresses over retention,
            # so they read against room-temperature strength.
            smax = np.maximum(equivalent_stress(s_h, m_c), util_c)
            rng = von_mises(s_h - s_c)
            seq = np.where(body, np.minimum(smax, rng / 0.7), 0.0)
            pc = int(np.argmax(seq))
            cyc_row = {"coldSafetyFactor": sf_c, "coldPeakStress": float(util_c[pk_c] * g_c[pk_c]),
                       "cycleStress": float(seq[pc]), "cycleStressRange": float(rng[pc]),
                       "cycleAt": hm.node_xyz[pc].tolist()}
        cf = study.case_factor(case.name)
        if fields:
            # the safety factor at every node, by the same criteria as the
            # case's reading (the equivalent stress over hot retention, the
            # layer bond, the cycle's cold end), against the same allowance.
            # The case's own number also reads refined submodels at its
            # hotspots, so this map's least can sit above it.
            with np.errstate(divide="ignore", invalid="ignore"):
                sf_n = np.where(util > 0, mat.strength / (util * STRESS_ALLOWANCE), np.inf)
                limited = np.zeros(len(sf_n), np.int8)        # 0 in-plane, 1 layer bond, 2 cold
                if inter is not None:
                    from .failure import bond_stress
                    eq_b = bond_stress(sig_u, build_vector(proc.build), bond_shear_ratio(proc))[0]
                    k_b = 1.0 if g_node is None else g_node
                    sf_b = np.where(eq_b > 0, k_b * proc.interlayer * mat.strength / (eq_b * STRESS_ALLOWANCE), np.inf)
                    limited[sf_b < sf_n] = 1
                    sf_n = np.minimum(sf_n, sf_b)
                if cold is not None:
                    sf_cn = np.where(util_c > 0, mat.strength / (util_c * STRESS_ALLOWANCE), np.inf)
                    limited[sf_cn < sf_n] = 2
                    sf_n = np.minimum(sf_n, sf_cn)
            fc = report["fields"]["cases"][-1]
            fc.update({"safetyFactor": sf_n, "limitedBy": limited, "body": body, "caseFactor": float(cf),
                       "safetyFactorReported": float(sf)})
        worst_sf = min(worst_sf, sf * cf)
        cases.append({
            "name": case.name,
            **({"impact": impact_rows[ci]} if ci in impact_rows else {}),
            **({"requiredSafetyFactor": float(case.safety_factor), "countsAs": float(sf * cf)}
               if getattr(case, "safety_factor", None) and cf != 1.0 else {}),
            "peakStress": peak,
            "peakAt": at,
            "globalPeakStress": global_peak,
            "refined": refined,
            "p99Stress": float(np.percentile(vm[body] if body.any() else vm, 99)),
            "contactPeakStress": float(vm.max()),
            "safetyFactor": sf,
            **({"peakTemperature": float((t_hot if cold is not None else heat.t_node)[pk]),
                "strengthRetention": keep}
               if g_node is not None else {}),
            **cyc_row,
            "interlayerStress": inter,
            **bond_row,
            "deflection": defl,
            "maxDisplacement": float(np.linalg.norm(disp, axis=1).max()),
            **({"clearances": clear_rows} if clear_rows else {}),
            "compliance": float(f @ u),
            **({"pressureImbalance": bc.pressure_imbalance[ci]} if wet.any() else {}),
            # the uncertain force's most stressful direction, which the
            # numbers above are for (unit vector)
            **({"worstDirection": [float(v) for v in direction]} if direction is not None else {}),
        })
    report["cases"] = cases
    report["safetyFactor"] = worst_sf
    if stock_on and stock_k is not None:
        # the members and their joints, from what the solved assembly puts on them (stock.assess):
        # section forces against the member's strength and buckling, interface forces against the
        # bond, the clamp's grip, the pin's bearing and the tube wall's crushing
        from .stock import assess as stock_assess

        def residual(u, f):
            return _internal_force(stiff, hm, u, moduli, fill, cg) + stock_k @ u - f

        report["stock"] = stock_assess(study, hm, stock_cases, residual, study.goal.safety_factor)
        report.setdefault("warnings", []).extend(report["stock"].pop("warnings", []))
    if layers is not None:
        kind, ratio = layers
        sfs = {a: (float(ratio * mat.strength / (axis_bond[i] * STRESS_ALLOWANCE)) if axis_bond[i] > 0 else None)
               for i, a in enumerate("xyz")}
        low = min((v for v in sfs.values() if v is not None), default=None)
        report["failure"]["layers"] = {
            "checked": False, "process": kind, "interlayer": ratio,
            "ifBuiltAlong": {a: {"interlayerStress": float(axis_bond[i]), "safetyFactor": sfs[a]}
                             for i, a in enumerate("xyz")}}
        if low is not None and low < worst_sf:
            worst_ax = min((a for a in "xyz" if sfs[a] is not None), key=lambda a: sfs[a])
            best_ax = max((a for a in "xyz" if sfs[a] is not None), key=lambda a: sfs[a])
            report["failure"]["layers"]["note"] = (
                f"no build direction given, so the layer bonds were not checked and the safety factor "
                f"{worst_sf:.2f} is the solid material's. {mat.name} is printed in layers ({kind}: bonds about "
                f"{ratio:.0%} of the in-plane strength): built along {worst_ax} the bonds would hold "
                f"{sfs[worst_ax]:.2f}x the loads, along {best_ax} {sfs[best_ax]:.2f}x. Give the study "
                "process.buildDirection (or 'auto') to design for it.")
    if reaction_rows:
        report["reactions"] = _reaction_report(study, reaction_rows)
        for r in report["reactions"]:
            for key, lim, unit in (("maxForce", "forceLimit", "N"), ("maxMoment", "momentLimit", "N·mm")):
                if r.get(lim) and r[key] > r[lim]:
                    report.setdefault("warnings", []).append(
                        f"{r['support']} carries {r[key]:.4g} {unit}, over its limit {r[lim]:g} {unit}: share the "
                        "load with another support, or move material so the path to the others is stiffer")
    if goal_sf and not study.self_equilibrated:
        # the safety factor holds against buckling too (study._margin). Read
        # for every part, however far under its stress safety factor: the
        # tet confirm can lift a voxel reading 2.5x (the MBB beam, 0.78 to
        # 1.97), and a part it lifts must not pass unchecked.
        try:
            bk = early_bk.result() if early_bk is not None else \
                _buckling(study, mesh, vol, pitch, contact_band, worst_dirs)
        except (RuntimeError, ValueError, MemoryError) as exc:
            bk = None
            report.setdefault("warnings", []).append(f"buckling check failed: {exc}")
        if bk is not None:
            report["buckling"] = bk
            if bk["factor"] is not None:
                report["bucklingFactor"] = bk["factor"]
                if bk["factor"] < goal_sf:
                    worst = min((c for c in bk["cases"] if c["factor"] is not None), key=lambda c: c["factor"])
                    report.setdefault("warnings", []).append(
                        f"{worst['name']}: buckles at {bk['factor']:.2f}x its loads (linear buckling), under "
                        f"the safety factor {goal_sf:g}; the mode is largest near "
                        f"({', '.join(f'{v:.1f}' for v in worst.get('modeAt', []))}). Thicken or brace the "
                        "slender member carrying compression there.")
    if env is not None and len(env):
        # how evenly the material is worked: mean von Mises over the near
        # peak (99.9th percentile, so one singular node does not set it).
        # Low (about 0.1, the wall brackets) means most of the part idles
        # at a few percent of the peak, which the stress-constrained method
        # trims (study.py, method auto)
        pk = float(np.quantile(env, 0.999))
        report["stressUtilization"] = float(env.mean() / pk) if pk > 0 else None
    report["stressAllowance"] = STRESS_ALLOWANCE
    if getattr(study, "vibration", None) is not None:
        try:
            vib = modal_check(study, mesh, vol, pitch)
        except (RuntimeError, ValueError, MemoryError) as exc:
            vib = None
            report.setdefault("warnings", []).append(f"modal check failed: {exc}")
        if vib is not None:
            report["vibration"] = vib
            report["firstFrequency"] = vib["firstFrequency"]
            if vib["minFrequency"] is not None and vib["firstFrequency"] < vib["minFrequency"]:
                report.setdefault("warnings", []).append(
                    f"first natural frequency {vib['firstFrequency']:.1f} Hz is under vibration.minFrequency "
                    f"{vib['minFrequency']:g} Hz (the mode moves mostly along {vib['firstModeAxis']}): stiffen "
                    "the path from the carried masses to the supports in that direction")
            for row in vib.get("excitations") or []:
                if row["ok"]:
                    continue
                lo, hi = row["keepClear"]
                bad = [f for f in vib["frequencies"] if lo < f < hi]
                if bad:
                    report.setdefault("warnings", []).append(
                        f"{len(bad)} mode{'s' if len(bad) > 1 else ''} ({', '.join(f'{f:.0f}' for f in bad[:6])}"
                        f"{', ...' if len(bad) > 6 else ''} Hz) where '{row['name']}' ({row['band'][0]:g}-"
                        f"{row['band'][1]:g} Hz) drives {'them' if len(bad) > 1 else 'it'}: the forcing is "
                        f"magnified up to {row['amplification']:g}x; every mode has to sit under {lo:.0f} Hz or "
                        f"over {hi:.0f} Hz")
                else:
                    report.setdefault("warnings", []).append(
                        f"the {len(vib['frequencies'])} modes computed stop at {vib['frequencies'][-1]:.0f} Hz, "
                        f"short of the top of '{row['name']}' ({hi:.0f} Hz): the band is not shown clear")
    from .durability import active, assess, envelope
    try:
        # held how long, repeated how often, how warm: the limits the
        # delivered stresses imply for loads nobody timed (durability.envelope)
        report["serviceEnvelope"] = envelope(study, cases, STRESS_ALLOWANCE)
    except (KeyError, ValueError, ZeroDivisionError) as exc:
        report.setdefault("warnings", []).append(f"service envelope failed: {exc}")
    if active(study):
        # fatigue (cases with cycles), creep and relaxation (hours, preload)
        report["durability"] = assess(study, cases, STRESS_ALLOWANCE)
    report["analysisCells"] = int(hm.n_elem)
    report["seconds"] = time.time() - t0
    return report
