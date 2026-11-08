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

__all__ = ["external_flow", "tunnel_grid"]


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
