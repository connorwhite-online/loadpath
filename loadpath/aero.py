"""
External flow: a part in a stream of air or water, as a wind tunnel would
put it, by the same lattice Boltzmann kernels as flow.py (D3Q19, TRT,
Smagorinsky subgrid viscosity, halfway bounce-back walls).

`external_flow` voxelizes the part in a tunnel (upstream, downstream and
side clearance in body lengths), drives a uniform stream in at the
upstream face and out at fixed pressure downstream, with the sides
periodic, and measures the force on the part by momentum exchange on every
wall link, time-averaged once the wake has developed. It returns the drag,
side force and lift (N) and their coefficients on the frontal area, the
moment about a point, the surface pressure as forces per wall voxel, and
`load_case`: that pressure as a study load case of surface patches, so a
wind or water load goes straight into a structural study (a drone arm, a
fairing, a bracket on a vehicle, a mount in a current).

What it is and is not: a coarse large-eddy simulation on a cubic grid,
and it reads high, the safe side for a structural load. The laminar
sphere at Re 100 (Cd 1.09, Schiller-Naumann) reads 1.21 at 12 cells
across in the default tunnel (11% high; 18 cells across moves it 1.5%, so
most of it is the tunnel: 1.26 with half the upstream and downstream run,
the periodic sides' 2% blockage the rest).
Sponge layers at both ends of the tunnel soak up the sound waves a bluff
body throws; without them a sharp-edged cube on 10 cells rang up and blew
(with them it reads Cd 1.00 at Re 250, and a 10 x 30 mm plate face-on 1.03
at Re 200).
Smooth bodies at high Reynolds number, whose separation point moves with
the boundary layer, need a resolution this grid does not reach: past
`re_max` (default 2e4 on the body length), or past what the grid holds
stably (about 25 times the body length in voxels), the run is made at that
lower Reynolds number with the physical dynamic pressure, and the report
says so (`reScale`, `note`). For those the coefficient is the subcritical one, a
conservative load for structure (more drag than the drag crisis gives).
"""
from __future__ import annotations

import time

import numpy as np

from .flow import C19, FLUID, INLET, LAMBDA, OUTLET, SOLID, _equilibrium, _kernels

__all__ = ["external_flow", "tunnel_grid", "rotor_blockage", "RotorStream", "BlockageCost", "duct_performance",
           "inlet_loss"]


def _rotation_to_x(direction):
    """Rotation taking unit `direction` to +x (rows: the new axes)."""
    d = np.asarray(direction, float)
    d = d / np.linalg.norm(d)
    a = np.array([0.0, 0.0, 1.0]) if abs(d[2]) < 0.9 else np.array([0.0, 1.0, 0.0])
    y = np.cross(a, d)
    y /= np.linalg.norm(y)
    z = np.cross(d, y)
    return np.stack([d, y, z])


def tunnel_grid(mesh, direction=(1, 0, 0), cells: int = 32, upstream: float = 1.5, downstream: float = 4.0,
                side: float = 1.5, max_nodes: int = 3_000_000):
    """The part in a tunnel along `direction`: (solid mask, voxel h mm,
    origin of voxel (0,0,0)'s centre in the tunnel frame, rotation R with
    tunnel = R @ part, body length mm). `cells` voxels across the body's
    largest frontal extent; clearances in body lengths (frontal size),
    shrunk to stay under `max_nodes`."""
    from .cutcell import scan_voxelize

    R = _rotation_to_x(direction)
    m = mesh.copy()
    T = np.eye(4)
    T[:3, :3] = R
    m.apply_transform(T)
    lo, hi = m.bounds
    ext = hi - lo
    front = float(max(ext[1], ext[2]))
    L = float(max(ext))
    h = front / cells
    while True:
        pad_lo = np.array([upstream * L, side * front, side * front])
        pad_hi = np.array([downstream * L, side * front, side * front])
        g_lo = lo - pad_lo
        dims = np.ceil((ext + pad_lo + pad_hi) / h).astype(int)
        if np.prod(dims) <= max_nodes:
            break
        h *= 1.1
    body = scan_voxelize(m, h, g_lo, dims)
    return body, h, g_lo + 0.5 * h, R, L, front


def external_flow(mesh, speed: float, direction=(1, 0, 0), *, medium="air", cells: int = 16, about=None,
                  upstream: float = 3.0, downstream: float = 6.0, side: float = 2.5, re_max: float = 2e4,
                  u_lattice: float = 0.1, cs: float = 0.1, flow_throughs: float = 5.0, average: float = 3.5,
                  max_nodes: int = 3_000_000, sponge: float = 0.1, sponge_strength: float = 0.3,
                  patches: int = 24, nu_floor: float = 0.004, verbose: bool = False) -> dict:
    """Steady (time-averaged) flow round `mesh` (mm) in a stream of `speed`
    m/s along `direction` (the way the fluid moves). medium: a heatx_thermal
    fluid name ("air", "water", ...) or Fluid. about: the point moments are
    taken about (mm, default the part's centroid).

    Runs `flow_throughs` passes of the stream through the tunnel, averaging
    the force over the last `average` of them. Returns drag (along the
    stream), the force (N, part frame), coefficients on the frontal area,
    the moment (N mm), the Reynolds number on the body length, the run's
    own (reScale < 1 when capped at re_max), and `load_case`."""
    from .heatx_thermal import FLUIDS, Fluid

    K = _kernels()
    fl = medium if isinstance(medium, Fluid) else FLUIDS[medium]
    t0 = time.time()
    body, h, origin, R, L, front = tunnel_grid(mesh, direction, cells, upstream, downstream, side, max_nodes)
    shape = body.shape
    nu_phys = fl.mu / fl.rho
    re_phys = speed * L * 1e-3 / nu_phys
    # the grid holds a Reynolds number of about u L / nu_floor (L in voxels):
    # under nu_floor (tau 0.512) a coarse grid's shear layers blow up even
    # with the subgrid viscosity (a 30 mm plate at 15 m/s on 8 cells did)
    re_grid = u_lattice * (L / h) / nu_floor
    re_run = min(re_phys, re_max, re_grid)
    re_scale = re_run / re_phys
    # lattice viscosity for the run's Re on the body length in voxels
    nu_lu = u_lattice * (L / h) / re_run
    tau0 = 0.5 + 3.0 * nu_lu
    ntype = np.where(body, SOLID, FLUID).astype(np.int8)
    ntype[0] = INLET
    ntype[-1] = OUTLET
    ntype[0][body[0]] = SOLID
    ntype[-1][body[-1]] = SOLID
    uin = np.zeros((3,) + shape, np.float32)
    uin[0][0] = u_lattice
    uin[0][-1] = -1.0                       # the outlet's inward normal
    per = np.array([0, 1, 1], np.int64)
    rho = np.ones(shape, np.float32)
    # still fluid, the inlet ramped up over a few sound crossings: an
    # impulsive start round a bluff body throws 20% density waves that blow
    # a coarse grid up
    u = [np.zeros(shape, np.float32) for _ in range(3)]
    f = _equilibrium(rho, *u)
    fn = f.copy()
    nut = np.zeros(shape, np.float32)

    # wall links: fluid node x, direction i, with x + c_i solid
    fluid = ~body
    links = []
    for i in range(1, 19):
        c = C19[i]
        nb = np.roll(body, shift=(-c[0], -c[1], -c[2]), axis=(0, 1, 2))
        if c[0] == 1:
            nb[-1] = False
        if c[0] == -1:
            nb[0] = False
        m = fluid & nb
        if m.any():
            idx = np.nonzero(m)
            links.append((i, idx))
    steps_per_pass = int(np.ceil(shape[0] / u_lattice))
    total = int(flow_throughs * steps_per_pass)
    avg_from = int(max(0.0, flow_throughs - average) * steps_per_pass)
    chunk = 50
    F_hist = []
    wall = np.zeros((3,) + shape)               # time-summed force on each wall voxel (lattice)
    n_avg = 0
    it = 0
    # links gathered as flat indices once: the force is summed every step
    # while averaging (the tunnel's sound waves ring at a few hundred steps,
    # which sparse samples alias), the wall map every few
    flat = [(i, np.ravel_multi_index(idx, shape),
             np.ravel_multi_index((idx[0] + C19[i, 0], idx[1] + C19[i, 1], idx[2] + C19[i, 2]), shape))
            for i, idx in links]
    F_sum = np.zeros(3)
    n_f = 0
    ramp = int(4 * shape[0] * np.sqrt(3.0))
    total += ramp
    avg_from += ramp
    # sponge slabs at both ends of the tunnel: the populations there are
    # pulled toward the free stream each step, which soaks up the sound
    # waves a bluff body throws (a velocity inlet reflects them, and a
    # sharp-edged cube on 10 cells rang up until it blew)
    ns = max(3, min(int(sponge * shape[0]), shape[0] // 6))
    sig = (np.arange(ns, dtype=np.float32)[::-1] + 1.0) / ns
    sig = (sponge_strength * sig * sig)[None, :, None, None]
    sig_out = sig[:, ::-1]
    ones = np.ones((ns,) + shape[1:], np.float32)
    zeros = np.zeros_like(ones)
    while it < total:
        if it <= ramp and it % 10 == 0:
            uin[0][0] = u_lattice * 0.5 * (1.0 - np.cos(np.pi * min(1.0, it / ramp)))
            feq_in = _equilibrium(ones, ones * uin[0][0][None], zeros, zeros)
        K["step"](f, fn, ntype, tau0, cs * cs, 0.0, 0.0, 0.0, uin, 1.0, per, rho, u[0], u[1], u[2], LAMBDA, nut)
        f, fn = fn, f
        f[:, :ns] += sig * (feq_in - f[:, :ns])
        f[:, -ns:] += sig_out * (feq_in - f[:, -ns:])
        it += 1
        if verbose and it % 200 == 0:
            print("  step", it, "max u", float(np.abs(u[0]).max()) / u_lattice, "rho", float(rho.min()), float(rho.max()),
                  "at", np.unravel_index(int(np.argmax(rho)), shape), np.unravel_index(int(np.argmax(np.abs(u[0]))), shape),
                  shape, flush=True)
        if it % 200 == 0 and (not np.isfinite(rho[shape[0] // 2].mean())
                              or float(np.abs(u[0]).max()) > 4.0 * u_lattice):
            if verbose:
                print("  max u", float(np.abs(u[0]).max()) / u_lattice, flush=True)
            raise RuntimeError("the flow run went unstable: give more cells across the body or a lower re_max")
        sample = it >= avg_from
        if not sample and it % chunk:
            continue
        F = np.zeros(3)
        wall_now = sample and it % 5 == 0
        for i, src, tgt in flat:
            fi = 2.0 * f[i].ravel()[src].astype(np.float64)
            F += C19[i] * fi.sum()
            if wall_now:
                for k in range(3):
                    if C19[i, k]:
                        np.add.at(wall[k].ravel(), tgt, C19[i, k] * fi)
        if wall_now:
            n_avg += 1
        if sample:
            F_sum += F
            n_f += 1
        if it % chunk == 0:
            F_hist.append(F)
        if verbose and it % (40 * chunk) == 0:
            print(f"  {it}/{total}  F {F}", flush=True)
    F_hist = np.array(F_hist)
    F_lu = F_sum / max(n_f, 1)
    # lattice -> physical: the run keeps the physical dynamic pressure
    vel = speed / u_lattice                       # m/s per lattice velocity
    f_unit = fl.rho * (h * 1e-3) ** 2 * vel ** 2   # N per lattice force
    F_t = F_lu * f_unit                           # tunnel frame
    F_part = R.T @ F_t
    q = 0.5 * fl.rho * speed ** 2
    area = float(body.any(axis=0).sum()) * (h * 1e-3) ** 2   # frontal area (projected on the stream)
    wall = wall / max(n_avg, 1) * f_unit          # N per wall voxel, tunnel frame
    sv = np.nonzero(np.abs(wall).sum(axis=0) > 0)
    xyz_t = origin + np.stack(sv, 1) * h
    xyz = xyz_t @ R                               # part frame (R orthonormal: part = R^T tunnel)
    fw = (wall[:, sv[0], sv[1], sv[2]].T) @ R
    c0 = np.asarray(about, float) if about is not None else np.asarray(mesh.center_mass if mesh.is_volume
                                                                         else mesh.centroid, float)
    M = np.cross(xyz - c0, fw).sum(axis=0)
    drag = float(F_t[0])
    rep = {
        "speed": speed, "direction": (np.asarray(direction, float) / np.linalg.norm(direction)).tolist(),
        "medium": fl.name, "drag": drag, "force": F_part.tolist(), "moment": M.tolist(), "about": c0.tolist(),
        "dragCoefficient": drag / (q * area), "frontalArea": area * 1e6,
        "sideCoefficient": float(F_t[1] / (q * area)), "liftCoefficient": float(F_t[2] / (q * area)),
        "dynamicPressure": q, "reynolds": re_phys, "reynoldsRun": re_run, "reScale": re_scale,
        "bodyLength": L, "voxel": h, "grid": list(shape), "cellsAcross": front / h,
        "steps": it, "seconds": round(time.time() - t0, 1),
        "forceHistory": [float(f"{v:.4g}") for v in (F_hist * f_unit)[:, 0][-40:]],
        "wallForceTotal": fw.sum(axis=0).tolist(),
    }
    if re_scale < 1:
        rep["note"] = (f"run at Re {re_run:.3g} (the physical {re_phys:.3g} is past what this grid resolves; "
                       "more --cells raises it); "
                       "forces are at the physical dynamic pressure with the lower-Re coefficient")
    rep["load_case"] = _load_case(xyz, fw, h, patches, speed)
    rep["_surface"] = (xyz, fw)
    return rep


def _load_case(xyz, fw, h, patches, speed):
    """The wall forces as a study load case: the wetted surface cut into
    `patches` patches (k-means on position and normal-ish force direction),
    each a box round its voxels carrying its share."""
    from scipy.cluster.vq import kmeans2

    if len(xyz) == 0:
        return None
    k = int(min(patches, len(xyz)))
    scale = np.ptp(xyz, axis=0).max() or 1.0
    dirs = fw / np.maximum(np.linalg.norm(fw, axis=1, keepdims=True), 1e-30)
    feats = np.hstack([xyz / scale, 0.3 * dirs])
    _, lab = kmeans2(feats, k, seed=0, minit="++")
    loads = []
    for j in range(k):
        s = lab == j
        if not s.any():
            continue
        F = fw[s].sum(axis=0)
        if np.linalg.norm(F) < 1e-9:
            continue
        lo = xyz[s].min(axis=0) - h
        hi = xyz[s].max(axis=0) + h
        loads.append({"shape": {"type": "box", "min": lo.round(4).tolist(), "max": hi.round(4).tolist()},
                      "force": [float(v) for v in F]})
    return {"name": f"flow-{speed:g}ms", "loads": loads}


# --- structure in a rotor's or fan's stream --------------------------------

G = 9.81                       # m/s^2
OPACITY_POWER = 3.0            # a grey cell's share of a column it blocks: density ** this (as SIMP's stiffness)


def _actuator_speed(v_disk: float, radius: float, s):
    """Axial speed on a rotor's axis at signed distance s (mm, + downstream)
    from an actuator disk passing v_disk: v_disk (1 + s / sqrt(R^2 + s^2)),
    twice v_disk far behind and falling off to nothing far ahead."""
    s = np.asarray(s, float)
    return v_disk * (1.0 + s / np.sqrt(radius * radius + s * s))


def _disk_frame(axis):
    a = np.asarray(axis, float)
    a = a / np.linalg.norm(a)
    t = np.array([1.0, 0.0, 0.0]) if abs(a[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = np.cross(a, t)
    e1 /= np.linalg.norm(e1)
    return a, e1, np.cross(a, e1)


class RotorStream:
    """A rotor's stream through a grid of cells, as momentum theory draws
    it, and the drag of the material it meets.

    The disk (centre, unit `axis` pointing the way the air goes, `radius`
    mm, `hub` mm left out) passes v_i = sqrt(T / (2 rho A)) at `thrust` N.
    The stream is followed along its own tubes: the speed on the axis,
    v(s) = v_i (1 + s / sqrt(R^2 + s^2)), sets every tube's radius by
    continuity, r(s) = r_disk sqrt(v_i / v(s)), so the wake narrows to
    R / sqrt(2) far behind the disk and the inflow widens ahead of it:
    material beside the narrowed wake is out of the stream, and material
    ahead of the disk beyond its radius is in it. `ducted`: a duct with no
    diffuser holds the stream at the disk's area, so it passes the disk and
    leaves at sqrt(T / (rho A)) (all the thrust on the rotor), straight.

    Each tube (a bin of the disk's area dA) is walked in the direction the
    air goes, ahead of the disk and behind it apart (the rotor fills in what
    is ahead of it). The first material it meets, at s, drags
    cd (rho v(s)^2 / 2) on the tube's section there, dA v_disk / v(s); what
    lies behind that sits in its wake and adds nothing. On a density field
    a cell blocks its share density^3 of the tube (OPACITY_POWER, as SIMP's
    stiffness: a grey cell is mostly air), so the drag is
    sum_k q_k o_k prod_{j<k} (1 - o_j) down each tube, exact for a part
    that is solid or void, and smooth for the optimizer.

    The grid: `origin` (min corner), `pitch`, `dims`, and `cell_index`
    (dims-shaped ints: each cell's place in the density vector, -1 for
    none), or None for a dense field in C order. `smooth`: densities read
    by trilinear interpolation at half-cell steps (False: the cell each
    point falls in, at whole-cell steps; a crisp voxel model, which
    rotor_blockage reads with tubes half a cell across).
    A slender member's cd is 1.0-1.2 (round to square), a flat plate face
    on 1.28; a streamlined one is about 0.1."""

    def __init__(self, origin, pitch: float, dims, center, axis, radius: float, thrust: float, hub: float = 0.0,
                 cd: float = 1.2, rho: float = 1.2, reach: float | None = None, ducted: bool = False,
                 cell_index=None, smooth: bool = True, bin: float | None = None):
        import scipy.sparse as sp

        origin = np.asarray(origin, float)
        dims = np.asarray(dims, int)
        c = np.asarray(center, float)
        a, e1, e2 = _disk_frame(axis)
        self.radius, self.hub, self.thrust, self.cd, self.ducted = float(radius), float(hub), float(thrust), cd, ducted
        reach = 3.0 * radius if reach is None else float(reach)
        area_m2 = np.pi * (radius * 1e-3) ** 2
        self.v_i = float(np.sqrt(max(thrust, 0.0) / (2.0 * rho * area_m2)))
        v_disk = self.v_i * (np.sqrt(2.0) if ducted else 1.0)
        self.disk_area = float(np.pi * (radius ** 2 - hub ** 2))
        n_cells = int(np.prod(dims)) if cell_index is None else int(cell_index.max()) + 1
        self.n = n_cells
        # the disk's area in bins of one cell (`bin`: of another size)
        b = float(pitch if bin is None else bin)
        nb = int(np.ceil(radius / b))
        uc = (np.arange(-nb, nb) + 0.5) * b
        u0, v0 = (g.ravel() for g in np.meshgrid(uc, uc, indexing="ij"))
        r0 = np.hypot(u0, v0)
        keep = (r0 <= radius) & (r0 >= hub)
        u0, v0 = u0[keep], v0[keep]
        self.d_a = b * b * self.disk_area / max(len(u0) * b * b, 1e-30)     # bins sum to the disk's area exactly
        self.n_bins = len(u0)
        # the stretch of the axis the grid spans
        corners = origin + np.array([[i, j, k] for i in (0, 1) for j in (0, 1) for k in (0, 1)]) * dims * pitch
        sc = (corners - c) @ a
        ds = 0.5 * pitch if smooth else pitch
        s_lo, s_hi = max(-reach, float(sc.min())), min(reach, float(sc.max()))
        n_s = max(int(np.floor((s_hi - s_lo) / ds)), 0)
        rows, cols, vals, samp_bin, samp_s = [], [], [], [], []
        n_samp = 0
        base = np.outer(u0, e1) + np.outer(v0, e2)
        for k in range(n_s):
            s_k = s_lo + (k + 0.5) * ds
            v = float(_actuator_speed(v_disk, radius, s_k))
            if ducted and s_k >= 0:
                v = v_disk
            if v <= 0.05 * v_disk:
                continue                        # far ahead: next to no air moves, and the tubes fan out without end
            pts = c + s_k * a + np.sqrt(v_disk / v) * base
            g = (pts - origin) / pitch - 0.5
            if smooth:
                i0 = np.floor(g).astype(int)
                fr = g - i0
                ids, wts = [], []
                for di in (0, 1):
                    for dj in (0, 1):
                        for dk in (0, 1):
                            ijk = i0 + (di, dj, dk)
                            w = (np.where(di, fr[:, 0], 1 - fr[:, 0]) * np.where(dj, fr[:, 1], 1 - fr[:, 1])
                                 * np.where(dk, fr[:, 2], 1 - fr[:, 2]))
                            ids.append(ijk)
                            wts.append(w)
            else:
                ids, wts = [np.round(g).astype(int)], [np.ones(len(g))]
            got = np.zeros(len(g), bool)
            layer = []
            for ijk, w in zip(ids, wts):
                ok = np.all((ijk >= 0) & (ijk < dims), axis=1) & (w > 1e-9)
                flat = np.zeros(len(g), np.int64)
                if cell_index is None:
                    flat[ok] = np.ravel_multi_index(ijk[ok].T, dims)
                else:
                    flat[ok] = cell_index[tuple(ijk[ok].T)]
                    ok &= flat >= 0
                got |= ok
                layer.append((ok, flat, w))
            if not got.any():
                continue
            local = np.cumsum(got) - 1
            for ok, flat, w in layer:
                rows.append(n_samp + local[ok])
                cols.append(flat[ok])
                vals.append(w[ok])
            m = int(got.sum())
            samp_bin.append(np.flatnonzero(got))
            samp_s.append(np.full(m, s_k))
            # N per unit opacity: cd q on the tube's section here, dA v_disk / v
            n_samp += m
        if n_samp == 0:
            self.w = None
            return
        samp_bin = np.concatenate(samp_bin)
        samp_s = np.concatenate(samp_s)
        w = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n_samp, n_cells))
        # each tube's samples in the order the air meets them, ahead of the disk and behind it apart
        side = (samp_s >= 0).astype(int)
        order = np.lexsort((samp_s, side, samp_bin))
        self.w = w[order].tocsr()
        self.wt = self.w.T.tocsr()
        samp_bin, samp_s, side = samp_bin[order], samp_s[order], side[order]
        seg = samp_bin * 2 + side
        self.start = np.flatnonzero(np.r_[True, seg[1:] != seg[:-1]])
        self.seg_of = np.cumsum(np.r_[True, seg[1:] != seg[:-1]]) - 1
        self.seg_side = side[self.start]
        v = _actuator_speed(v_disk, radius, samp_s)
        if ducted:
            v = np.where(samp_s >= 0, v_disk, v)
        self.q = cd * 0.5 * rho * v * v_disk * self.d_a * 1e-6           # N: cd (rho v^2 / 2) (dA v_disk / v)

    def drag(self, x, grad: bool = False):
        """Drag (N) of the field `x` (one density per cell), with what is
        blocked: {"drag", "downstream": {"drag", "blockedArea"}, "upstream":
        {...}} (areas in mm^2 of the disk whose tubes meet material), and
        with `grad` its derivative per cell."""
        out = {"drag": 0.0, "downstream": {"blockedArea": 0.0, "drag": 0.0}, "upstream": {"blockedArea": 0.0, "drag": 0.0}}
        if self.w is None:
            return (out, np.zeros(self.n)) if grad else out
        rho = np.clip(self.w @ np.asarray(x, float), 0.0, 1.0)
        o = np.minimum(rho ** OPACITY_POWER, 1.0 - 1e-9)
        lg = np.log1p(-o)
        cs = np.cumsum(lg)
        excl = cs - lg                                   # sum of logs before each sample, over all tubes
        t = np.exp(excl - excl[self.start][self.seg_of])  # transmission down to each sample in its own tube
        term = self.q * o * t
        seg_drag = np.add.reduceat(term, self.start)
        ends = np.r_[self.start[1:], len(o)] - 1
        seg_block = 1.0 - t[ends] * (1.0 - o[ends])
        for name, sd in (("upstream", 0), ("downstream", 1)):
            m = self.seg_side == sd
            out[name] = {"blockedArea": float(seg_block[m].sum() * self.d_a), "drag": float(seg_drag[m].sum())}
        out["drag"] = float(seg_drag.sum())
        if not grad:
            return out
        ct = np.cumsum(term)
        behind = (seg_drag + (ct - term)[self.start])[self.seg_of] - ct      # drag of what lies behind each sample
        d_o = self.q * t - behind / (1.0 - o)
        d_rho = d_o * OPACITY_POWER * rho ** (OPACITY_POWER - 1.0)
        return out, self.wt @ d_rho


def rotor_blockage(mesh, center, axis, radius: float, thrust: float, hub: float = 0.0, cd: float = 1.2,
                   rho: float = 1.2, reach: float | None = None, pitch: float = 0.5, ducted: bool = False) -> dict:
    """What a part costs a rotor by sitting in its stream (drone arms and
    motor mounts under a prop, a bracket in front of a fan): the drag of the
    material the disk's air meets, from momentum theory (RotorStream, on the
    part voxelized at `pitch`). The same model is a cost inside the
    optimizer (a study's `aero.rotors`, BlockageCost).

    Returns the blocked area (mm^2 of the disk whose stream meets material,
    and as a share of the disk), the drag (N) and the same as thrust lost
    (share) and as a carried mass that would cost a hovering craft as much
    (g, drag / 9.81), upstream and downstream apart."""
    import trimesh

    from .verify import voxelize

    m = mesh if isinstance(mesh, trimesh.Trimesh) else trimesh.Trimesh(*mesh)
    a, e1, e2 = _disk_frame(axis)
    c = np.asarray(center, float)
    reach = 3.0 * radius if reach is None else float(reach)
    # a box round everything the stream can reach: the inflow widens ahead of the disk
    wide = radius / np.sqrt(0.05)
    pts = np.array([c + s * a + u * e1 + v * e2 for s in (-reach, reach) for u in (-wide, wide) for v in (-wide, wide)])
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    mlo, mhi = np.asarray(m.bounds, float)
    out = {"inducedVelocity": None, "diskArea": round(float(np.pi * (radius ** 2 - hub ** 2)), 1)}
    if np.any(mhi < lo) or np.any(mlo > hi):
        stream = RotorStream(lo, pitch, (1, 1, 1), c, a, radius, thrust, hub, cd, rho, reach, ducted, smooth=False)
        got = stream.drag(np.zeros(1))
    else:
        solid, origin = voxelize(m, pitch, box=(np.maximum(lo, mlo - pitch), np.minimum(hi, mhi + pitch)))
        stream = RotorStream(origin, pitch, solid.shape, c, a, radius, thrust, hub, cd, rho, reach, ducted,
                             smooth=False, bin=0.5 * pitch)
        got = stream.drag(solid.ravel().astype(float))
    out["inducedVelocity"] = round(stream.v_i, 3)
    for side in ("downstream", "upstream"):
        out[side] = {"blockedArea": round(got[side]["blockedArea"], 1), "drag": round(got[side]["drag"], 5)}
    total = out["downstream"]["drag"] + out["upstream"]["drag"]
    out.update({"blockedShare": round((out["downstream"]["blockedArea"] + out["upstream"]["blockedArea"])
                                      / max(out["diskArea"], 1e-9), 4),
                "drag": round(total, 5), "thrustLost": round(total / thrust, 4) if thrust > 0 else 0.0,
                "equivalentMass": round(total / G * 1000.0, 2), "cd": cd})
    return out


class BlockageCost:
    """A study's `aero.rotors` as a cost in the optimizer's material budget:
    the drag of the design in each rotor's stream (RotorStream on the
    design grid), charged as the material that would weigh on a hovering
    craft as much, drag / g, in cells of the study's material. A gram of
    thrust lost under a prop then trades one for one against a gram of
    structure, so the optimizer moves an arm out of the stream, or thins it
    where the air is fastest, exactly when that is lighter overall.
    `rotors`: spec.Rotor list; `density` g/cm^3."""

    def __init__(self, domain, rotors, density: float, grams_per_newton: float = 1000.0 / G):
        mesh = domain.mesh
        idx = np.full(domain.dims, -1, np.int64)
        idx[tuple(mesh.elem_ijk.T)] = np.arange(mesh.n_elem)
        self.streams = [RotorStream(mesh.origin, mesh.pitch, domain.dims, r.center, r.axis, r.radius, r.thrust,
                                    r.hub, r.cd, r.rho, r.reach, r.ducted, cell_index=idx) for r in rotors]
        self.names = [r.name for r in rotors]
        cell_g = density * 1e-3 * mesh.pitch ** 3
        self.cells_per_newton = grams_per_newton / cell_g
        self.grams_per_newton = grams_per_newton
        self.base = 0.0
        # what the drawn (preserved) geometry alone costs: no design choice, so off the budget
        self.base = self.cost(domain.preserve.astype(float))[0]

    def drag(self, x) -> float:
        return float(sum(st.drag(x)["drag"] for st in self.streams))

    def cost(self, x, grad: bool = False):
        """(cells of material the field's drag is worth, less the drawn
        geometry's own; its derivative per cell or None; drag N)."""
        total, g = 0.0, (np.zeros(len(x)) if grad else None)
        for st in self.streams:
            if grad:
                got, gi = st.drag(x, grad=True)
                g += gi * self.cells_per_newton
            else:
                got = st.drag(x)
            total += got["drag"]
        return total * self.cells_per_newton - self.base, g, total


# Entrance loss of a pipe inlet by the radius of its lip over the bore
# (fraction of the dynamic head inside; Idelchik, Handbook of Hydraulic
# Resistance, and Crane TP-410: a sharp flush edge 0.5, a well-rounded
# bellmouth 0.03-0.04). What a duct's lip fails to recover.
_INLET_K = ((0.0, 0.50), (0.02, 0.28), (0.04, 0.24), (0.06, 0.15), (0.10, 0.09), (0.15, 0.04), (0.20, 0.03))
DIFFUSER_MAX_HALF_ANGLE = 7.0     # deg: a conical diffuser's wall holds the flow to about here
DIFFUSER_EFFICIENCY = 0.85        # pressure recovered of the ideal, attached
TIP_DISCHARGE = 0.8               # tip-gap leakage jet (turbomachinery tip-leakage models use 0.6-0.84)


def inlet_loss(lip_radius: float, bore: float) -> float:
    """Entrance loss coefficient of a duct lip of radius `lip_radius` on a
    bore of diameter `bore` (same unit)."""
    r = max(float(lip_radius), 0.0) / float(bore)
    xs, ks = zip(*_INLET_K)
    return float(np.interp(r, xs, ks))


def _duct_state(thrust, speed, area, sigma, k_loss, gap_area, rho):
    """A ducted disk making `thrust` in an axial stream `speed`: (disk speed,
    exit speed, rotor pressure rise, leak share, shaft power)."""
    # T = rho A v (w - V), w = v / sigma
    b, c = -speed * sigma, -thrust * sigma / (rho * area)
    v = 0.5 * (-b + np.sqrt(b * b - 4.0 * c))
    w = v / sigma
    dp = 0.5 * rho * (w * w - speed * speed) + k_loss * 0.5 * rho * v * v
    leak = TIP_DISCHARGE * gap_area / area * np.sqrt(max(2.0 * dp / rho, 0.0)) / v
    return v, w, dp, leak, dp * area * v * (1.0 + leak)


def _open_thrust(power, speed, area, rho, kappa):
    """Thrust an open rotor makes from `power` (induced, with factor kappa)
    in an axial stream: P = kappa T (V + v_i), T = 2 rho A v_i (V + v_i)."""
    lo, hi = 0.0, 1e6
    for _ in range(200):
        t = 0.5 * (lo + hi)
        vi = 0.5 * (-speed + np.sqrt(speed * speed + 2.0 * t / (rho * area)))
        if kappa * t * (speed + vi) > power:
            hi = t
        else:
            lo = t
    return 0.5 * (lo + hi)


def duct_performance(radius: float, thrust: float, tip_gap: float = 0.0, lip_radius: float = 0.0,
                     length: float = 0.0, diffuser_angle: float = 0.0, outer_radius: float | None = None,
                     height: float | None = None, hub: float = 0.0, rho: float = 1.2, speed: float = 0.0,
                     cruise: float | None = None, open_factor: float = 1.0) -> dict:
    """What a duct round a rotor gives and costs, against the same rotor in
    the open at equal induced power. mm, N, m/s; `radius` is the blade tip's.

    The ideal first (momentum theory): a duct whose flow leaves at sigma x
    the disk's area keeps the wake from contracting, so the same thrust
    needs less induced power, T^1.5 / sqrt(4 sigma rho A) against T^1.5 /
    sqrt(2 rho A) in the open; at equal power that is (2 sigma)^(1/3) the
    thrust (1.26 with no diffuser), and the rotor itself carries only 1 /
    (2 sigma) of it. The rest is suction on the inlet lip: the duct's
    thrust, which its wall carries into the frame.

    Then what a small duct loses, each as total pressure the rotor must
    make up or as flow it must pump twice:

    - the lip: a pipe entrance of that radius (inlet_loss), from a sharp
      edge's half a dynamic head to a bellmouth's few percent;
    - the wall: pipe friction over `length` below the disk (Blasius);
    - a diffuser (`diffuser_angle`, half-angle in degrees over `length`):
      it widens the exit only as far as its wall holds the flow
      (DIFFUSER_MAX_HALF_ANGLE) and recovers DIFFUSER_EFFICIENCY of the
      ideal;
    - the tip gap: the rotor's pressure rise drives a jet back through the
      gap's ring, C_d A_gap sqrt(2 dp / rho), air that is pumped and makes
      no thrust. (The same order as turbomachinery's rule of 1-2% of
      efficiency per 1% of blade height. A gap wide enough for the tip
      vortex to re-form, a few percent of the radius, loses more than
      this: read the number there as the most the duct gives.)

    Both rotors are ideal disks otherwise, so the ratio is the duct's own
    doing. A real open rotor also loses at its blade tips, which a duct's
    wall stops: `open_factor` is the open rotor's induced power over the
    ducted one's for that (a blade-element model's tip loss; about 1.05 to
    1.1 for a small two- or three-blade prop), and the ratio rises by its
    2/3 power. Profile power is left out of both: at equal shaft power the
    gain is smaller by the share of power that is not induced.

    `speed`: an axial stream (climb), where the duct's gain fades as the
    wake needs less holding. `cruise`: an edgewise speed (a multirotor
    flying forward). A duct that turns all the air it swallows to
    its axis pays the momentum drag m' V: `momentumDrag` is that bound (a
    duct short against its bore turns less of it), and `profileDrag` its own
    outside as a short cylinder in cross flow (`outer_radius`, `height`).
    Past a cross flow of about half the disk speed the windward lip stalls
    and the hover gain is not kept, which this does not model
    (`crossflowRatio` says where it is).

    Returns thrustRatio (ducted over open, same blade radius, equal induced
    power), thrustGained (N at that power), thrustRatioSameSize (the open
    rotor as big as the duct's outside: what the duct's width would do as
    blade), idealRatio, ductThrust and rotorThrust (N at `thrust`),
    ductShare, the losses, diskVelocity and exitVelocity (m/s), power (W,
    induced), and with `cruise`: momentumDrag, profileDrag, drag (N)."""
    r_d = (radius + tip_gap) * 1e-3
    # the blades' own annulus makes the thrust: the gap's ring carries the leak back
    area = np.pi * ((radius * 1e-3) ** 2 - (hub * 1e-3) ** 2)
    bore = 2.0 * r_d
    lm = max(length, 0.0) * 1e-3
    half = min(max(diffuser_angle, 0.0), DIFFUSER_MAX_HALF_ANGLE)
    r_e = r_d + lm * np.tan(np.radians(half))
    sigma = float(r_e ** 2 / r_d ** 2)
    k_in = inlet_loss(lip_radius, 2.0 * (radius + tip_gap))
    k_diff = (1.0 - DIFFUSER_EFFICIENCY) * (1.0 - 1.0 / sigma ** 2)
    gap_area = np.pi * (r_d ** 2 - (radius * 1e-3) ** 2)
    # wall friction at the disk speed's Reynolds number (two passes: K depends on v weakly)
    k_f = 0.0
    for _ in range(3):
        v, w, dp, leak, power = _duct_state(thrust, speed, area, sigma, k_in + k_diff + k_f, gap_area, rho)
        re = max(v * bore / 1.5e-5, 1.0)
        k_f = 0.316 * re ** -0.25 * lm / bore
    v, w, dp, leak, power = _duct_state(thrust, speed, area, sigma, k_in + k_diff + k_f, gap_area, rho)
    ideal_power = _duct_state(thrust, speed, area, sigma, 0.0, 0.0, rho)[4]
    a_open = np.pi * ((radius * 1e-3) ** 2 - (hub * 1e-3) ** 2)
    t_open = _open_thrust(power, speed, a_open, rho, open_factor)
    t_open_ideal = _open_thrust(ideal_power, speed, a_open, rho, 1.0)
    rotor = float(min(dp * area, thrust * 10.0))
    out = {"thrustRatio": float(thrust / t_open), "idealRatio": float(thrust / t_open_ideal),
           "rotorThrust": rotor, "ductThrust": float(thrust - rotor), "ductShare": float(1.0 - rotor / thrust),
           "exitAreaRatio": sigma, "diskVelocity": float(v), "exitVelocity": float(w), "power": float(power),
           "losses": {"lip": k_in, "wall": float(k_f), "diffuser": float(k_diff), "tipLeak": float(leak)},
           "thrustGained": float(thrust - t_open)}
    if outer_radius is not None:
        a_big = np.pi * ((outer_radius * 1e-3) ** 2 - (hub * 1e-3) ** 2)
        out["thrustRatioSameSize"] = float(thrust / _open_thrust(power, speed, a_big, rho, open_factor))
    if cruise is not None and cruise > 0:
        hv = _duct_state(thrust, 0.0, area, sigma, k_in + k_diff + k_f, gap_area, rho)[0]
        mom = rho * area * hv * cruise
        ro = (outer_radius if outer_radius is not None else radius + tip_gap) * 1e-3
        hh = (height if height is not None else length) * 1e-3
        prof = 1.0 * 0.5 * rho * cruise ** 2 * 2.0 * ro * hh
        out.update(cruiseSpeed=float(cruise), momentumDrag=float(mom), profileDrag=float(prof),
                   drag=float(mom + prof), crossflowRatio=float(cruise / hv))
    return out

