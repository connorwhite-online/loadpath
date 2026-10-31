"""
Flow and heat transfer on voxel grids, by the lattice Boltzmann method.

The rating in heatx_thermal.py takes its heat transfer coefficient and
pressure drop from fits to published or simulated cells, each with its own
range. This module computes them directly, for any channel geometry that
can be voxelized:

- `cell_flow` runs one periodic cell (a TPMS side, or any fluid mask that
  tiles) driven by a body force along an axis, to fully developed flow, and
  returns Re, the Darcy friction factor and Nu on the hydraulic diameter.
  Heat transfer is the fully developed problem with walls held at T = 0 and
  uniform heating in the fluid (the convention of the 1KZ charge-cooler cells
  the rating's LBM fit came from), or conjugate, with the wall conducting
  and the far side of the wall held at T = 0.
- `network_flow` runs a whole channel network, ports and plenums included,
  from a velocity inlet to a pressure outlet, and reports the pressure drop,
  how it splits along the path, and how evenly the core's channels share the
  flow.

Method: D3Q19 lattice Boltzmann with two-relaxation-time (TRT) collisions
and a Smagorinsky subgrid viscosity (Hou et al. 1996 effective relaxation
time, C_s 0.1), so it stays stable well into turbulent Re on cells of 48-96
voxels. TRT with Lambda = 3/16 puts halfway bounce-back walls exactly
halfway whatever the viscosity, so a cell's permeability does not move with
the lattice viscosity as it does under BGK (on a 64-voxel gyroid cell, BGK's
f Re in creeping flow runs 1% high at tau 0.8 and 4% at tau 0.6, and more
toward tau 0.5, where high-Re runs live);
Guo body forcing; halfway bounce-back walls; equilibrium velocity inlets and
fixed-density outlets. Temperature is a passive scalar on D3Q7 with
two-relaxation-time collisions (Lambda 1/4) and anti-bounce-back isothermal
walls. Both run as numba kernels in parallel over x (`pip install numba`,
the `flow` extra); there is no pure-numpy path, it would be ~100x slower.

Lattice units throughout the kernels; Re, f and Nu are dimensionless and the
physical results (Pa, W/m^2K) come from them in `cell_flow`'s callers.
"""
from __future__ import annotations

import time

import numpy as np

__all__ = ["cell_flow", "tpms_cell", "wetted_area", "network_flow", "exchanger_flow", "available"]

try:
    import numba as nb

    _HAVE_NUMBA = True
except ImportError:      # pragma: no cover
    nb = None
    _HAVE_NUMBA = False


def available() -> bool:
    return _HAVE_NUMBA


def _need_numba():
    if not _HAVE_NUMBA:
        raise ImportError("loadpath.flow needs numba: pip install numba (or loadpath[flow])")


# D3Q19 velocities, weights and opposites
C19 = np.array([[0, 0, 0],
                [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1],
                [1, 1, 0], [-1, -1, 0], [1, -1, 0], [-1, 1, 0],
                [1, 0, 1], [-1, 0, -1], [1, 0, -1], [-1, 0, 1],
                [0, 1, 1], [0, -1, -1], [0, 1, -1], [0, -1, 1]], np.int64)
W19 = np.array([1 / 3] + [1 / 18] * 6 + [1 / 36] * 12)
OPP19 = np.array([0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15, 18, 17], np.int64)
C7 = C19[:7].copy()
W7 = np.array([1 / 4] + [1 / 8] * 6)
OPP7 = OPP19[:7].copy()

# node types
FLUID, SOLID, INLET, OUTLET = 0, 1, 2, 3


def _kernels():
    """Compile the kernels on first use (numba import is optional)."""
    if "step" in _K:
        return _K
    _need_numba()
    C, W, OPP = C19, W19, OPP19
    Ct, Wt, OPPt = C7, W7, OPP7

    @nb.njit(parallel=True, fastmath=True, cache=True)
    def step(f, fn, ntype, tau0, cs2, gx, gy, gz, u_in, rho_out, periodic, rho, ux, uy, uz, lam, nut):
        """One collide-and-stream step (pull). f holds post-collision populations."""
        nx, ny, nz = ntype.shape
        for x in nb.prange(nx):
            fin = np.empty(19)
            feq = np.empty(19)
            for y in range(ny):
                for z in range(nz):
                    t = ntype[x, y, z]
                    if t == 1:
                        continue
                    for i in range(19):
                        sx = x - C[i, 0]
                        sy = y - C[i, 1]
                        sz = z - C[i, 2]
                        if periodic[0]:
                            sx %= nx
                        if periodic[1]:
                            sy %= ny
                        if periodic[2]:
                            sz %= nz
                        if sx < 0 or sx >= nx or sy < 0 or sy >= ny or sz < 0 or sz >= nz:
                            fin[i] = f[OPP[i], x, y, z]
                        elif ntype[sx, sy, sz] == 1:
                            fin[i] = f[OPP[i], x, y, z]
                        else:
                            fin[i] = f[i, sx, sy, sz]
                    r = 0.0
                    jx = 0.0
                    jy = 0.0
                    jz = 0.0
                    for i in range(19):
                        r += fin[i]
                        jx += fin[i] * C[i, 0]
                        jy += fin[i] * C[i, 1]
                        jz += fin[i] * C[i, 2]
                    if t == 2 or t == 3:
                        # inlet: prescribed velocity, density from the neighbour inward;
                        # outlet: fixed density, velocity from the neighbour inward.
                        # u_in holds the inlet velocity, or the outlet's inward normal.
                        if t == 2:
                            vx = u_in[0, x, y, z]
                            vy = u_in[1, x, y, z]
                            vz = u_in[2, x, y, z]
                            ax = abs(vx)
                            ay = abs(vy)
                            az = abs(vz)
                            ox = 0
                            oy = 0
                            oz = 0
                            if ax >= ay and ax >= az:
                                ox = 1 if vx > 0 else -1
                            elif ay >= az:
                                oy = 1 if vy > 0 else -1
                            else:
                                oz = 1 if vz > 0 else -1
                        else:
                            ox = int(u_in[0, x, y, z])
                            oy = int(u_in[1, x, y, z])
                            oz = int(u_in[2, x, y, z])
                            vx = jx / r
                            vy = jy / r
                            vz = jz / r
                        qx = x + ox
                        qy = y + oy
                        qz = z + oz
                        if 0 <= qx < nx and 0 <= qy < ny and 0 <= qz < nz and ntype[qx, qy, qz] == 0:
                            if t == 2:
                                r = rho[qx, qy, qz]
                            else:
                                vx = ux[qx, qy, qz]
                                vy = uy[qx, qy, qz]
                                vz = uz[qx, qy, qz]
                        if t == 3:
                            r = rho_out
                        if r <= 0.0:
                            r = 1.0
                        usq = vx * vx + vy * vy + vz * vz
                        for i in range(19):
                            cu = C[i, 0] * vx + C[i, 1] * vy + C[i, 2] * vz
                            fn[i, x, y, z] = W[i] * r * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * usq)
                        rho[x, y, z] = r
                        ux[x, y, z] = vx
                        uy[x, y, z] = vy
                        uz[x, y, z] = vz
                        continue
                    Fx = gx * r
                    Fy = gy * r
                    Fz = gz * r
                    vx = (jx + 0.5 * Fx) / r
                    vy = (jy + 0.5 * Fy) / r
                    vz = (jz + 0.5 * Fz) / r
                    usq = vx * vx + vy * vy + vz * vz
                    pxx = 0.0
                    pyy = 0.0
                    pzz = 0.0
                    pxy = 0.0
                    pxz = 0.0
                    pyz = 0.0
                    for i in range(19):
                        cu = C[i, 0] * vx + C[i, 1] * vy + C[i, 2] * vz
                        feq[i] = W[i] * r * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * usq)
                        d = fin[i] - feq[i]
                        pxx += d * C[i, 0] * C[i, 0]
                        pyy += d * C[i, 1] * C[i, 1]
                        pzz += d * C[i, 2] * C[i, 2]
                        pxy += d * C[i, 0] * C[i, 1]
                        pxz += d * C[i, 0] * C[i, 2]
                        pyz += d * C[i, 1] * C[i, 2]
                    # the forcing's share of the second moment (Guo)
                    pxx += vx * Fx
                    pyy += vy * Fy
                    pzz += vz * Fz
                    pxy += 0.5 * (vx * Fy + vy * Fx)
                    pxz += 0.5 * (vx * Fz + vz * Fx)
                    pyz += 0.5 * (vy * Fz + vz * Fy)
                    Q = np.sqrt(pxx * pxx + pyy * pyy + pzz * pzz + 2.0 * (pxy * pxy + pxz * pxz + pyz * pyz))
                    tau = 0.5 * (tau0 + np.sqrt(tau0 * tau0 + 18.0 * 1.41421356 * cs2 * Q / r))
                    # TRT: even moments relax at the (subgrid-augmented) viscous rate, odd
                    # ones at the rate that puts halfway bounce-back walls exactly halfway
                    # (Lambda 3/16), whatever the viscosity
                    nut[x, y, z] = (tau - tau0) / 3.0          # subgrid viscosity, for the heat
                    op = 1.0 / tau
                    om = 1.0 / (0.5 + lam / (tau - 0.5))
                    pp = 1.0 - 0.5 * op
                    pm = 1.0 - 0.5 * om
                    uF = vx * Fx + vy * Fy + vz * Fz
                    fn[0, x, y, z] = fin[0] - op * (fin[0] - feq[0]) - pp * W[0] * 3.0 * uF
                    for i in range(1, 19, 2):
                        j = i + 1
                        cu = C[i, 0] * vx + C[i, 1] * vy + C[i, 2] * vz
                        cF = C[i, 0] * Fx + C[i, 1] * Fy + C[i, 2] * Fz
                        dp = op * (0.5 * (fin[i] + fin[j]) - 0.5 * (feq[i] + feq[j]))
                        dm = om * (0.5 * (fin[i] - fin[j]) - 0.5 * (feq[i] - feq[j]))
                        sp = pp * W[i] * (9.0 * cu * cF - 3.0 * uF)
                        sm = pm * W[i] * 3.0 * cF
                        fn[i, x, y, z] = fin[i] - dp - dm + sp + sm
                        fn[j, x, y, z] = fin[j] - dp + dm + sp - sm
                    rho[x, y, z] = r
                    ux[x, y, z] = vx
                    uy[x, y, z] = vy
                    uz[x, y, z] = vz

    @nb.njit(parallel=True, fastmath=True, cache=True)
    def step_t(g, gn, tnode, ux, uy, uz, alpha, tfix, qsrc, qvel, periodic, T, nut, prt_inv):
        """D3Q7 TRT advection-diffusion. tnode: 0 active (fluid or conducting
        solid), 1 adiabatic wall, 2 isothermal wall at tfix (anti-bounce-back,
        halfway), 3 fixed-temperature node (inlet), 4 zero-gradient outlet.
        The source is qsrc + qvel . u per node (qvel: the flow-weighted
        heating of a fully developed mean axial gradient)."""
        nx, ny, nz = tnode.shape
        for x in nb.prange(nx):
            gin = np.empty(7)
            for y in range(ny):
                for z in range(nz):
                    t = tnode[x, y, z]
                    if t == 1 or t == 2:
                        continue
                    for i in range(7):
                        sx = x - Ct[i, 0]
                        sy = y - Ct[i, 1]
                        sz = z - Ct[i, 2]
                        if periodic[0]:
                            sx %= nx
                        if periodic[1]:
                            sy %= ny
                        if periodic[2]:
                            sz %= nz
                        if sx < 0 or sx >= nx or sy < 0 or sy >= ny or sz < 0 or sz >= nz:
                            gin[i] = g[OPPt[i], x, y, z]
                        else:
                            s = tnode[sx, sy, sz]
                            if s == 1:
                                gin[i] = g[OPPt[i], x, y, z]
                            elif s == 2:
                                gin[i] = -g[OPPt[i], x, y, z] + 2.0 * Wt[i] * tfix[sx, sy, sz]
                            else:
                                gin[i] = g[i, sx, sy, sz]
                    Tl = 0.0
                    for i in range(7):
                        Tl += gin[i]
                    if t == 3:
                        Tl = tfix[x, y, z]
                    vx = ux[x, y, z]
                    vy = uy[x, y, z]
                    vz = uz[x, y, z]
                    if t == 3 or t == 4:
                        for i in range(7):
                            cu = Ct[i, 0] * vx + Ct[i, 1] * vy + Ct[i, 2] * vz
                            gn[i, x, y, z] = Wt[i] * Tl * (1.0 + 4.0 * cu)
                        T[x, y, z] = Tl
                        continue
                    # odd moments set the diffusivity: molecular plus the flow's subgrid
                    # viscosity over a turbulent Prandtl number
                    tm = 0.5 + 4.0 * (alpha[x, y, z] + nut[x, y, z] * prt_inv)
                    tp = 0.5 + 0.25 / (tm - 0.5)             # Lambda = 1/4
                    om = 1.0 / tm
                    op = 1.0 / tp
                    S = qsrc[x, y, z] + qvel[0] * vx + qvel[1] * vy + qvel[2] * vz
                    e0 = Wt[0] * Tl
                    gn[0, x, y, z] = gin[0] - op * (gin[0] - e0) + Wt[0] * S
                    for i in range(1, 7, 2):
                        j = i + 1
                        cu = Ct[i, 0] * vx + Ct[i, 1] * vy + Ct[i, 2] * vz
                        ep = Wt[i] * Tl
                        em = Wt[i] * Tl * 4.0 * cu
                        gp = 0.5 * (gin[i] + gin[j])
                        gm = 0.5 * (gin[i] - gin[j])
                        dp = op * (gp - ep)
                        dm = om * (gm - em)
                        gn[i, x, y, z] = gin[i] - dp - dm + Wt[i] * S
                        gn[j, x, y, z] = gin[j] - dp + dm + Wt[j] * S
                    T[x, y, z] = Tl

    _K["step"] = step
    _K["step_t"] = step_t
    return _K


_K = {}
LAMBDA = 3.0 / 16.0


def _equilibrium(rho, ux, uy, uz):
    f = np.empty((19,) + rho.shape, np.float32)
    usq = ux * ux + uy * uy + uz * uz
    for i in range(19):
        cu = C19[i, 0] * ux + C19[i, 1] * uy + C19[i, 2] * uz
        f[i] = W19[i] * rho * (1 + 3 * cu + 4.5 * cu * cu - 1.5 * usq)
    return f


def _equilibrium_t(T, ux, uy, uz):
    g = np.empty((7,) + T.shape, np.float32)
    for i in range(7):
        cu = C7[i, 0] * ux + C7[i, 1] * uy + C7[i, 2] * uz
        g[i] = W7[i] * T * (1 + 4 * cu)
    return g


# ---------------------------------------------------------------- geometry

def wetted_area(field: np.ndarray, h: float, periodic: bool = True) -> float:
    """Area (h units squared) of the zero level of `field` (positive in the
    fluid) over the box, by marching cubes; periodic wraps the box so the
    faces close."""
    from skimage.measure import marching_cubes

    d = np.pad(field, 1, mode="wrap" if periodic else "edge").astype(np.float32)
    v, fc, _, _ = marching_cubes(d, 0.0, spacing=(h, h, h))
    a, b, c = v[fc[:, 0]], v[fc[:, 1]], v[fc[:, 2]]
    cen = (a + b + c) / 3.0
    n = np.array(field.shape) * h
    inside = np.all((cen >= h) & (cen < h + n), axis=1)
    return float(0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)[inside].sum())


def tpms_cell(kind: str = "gyroid", cell: float = 10.0, wall: float = 1.0, split: float = 0.5,
              side: str = "a", n: int = 64):
    """One fluid's side of a sheet TPMS over one periodic cell, on n^3 voxels:
    (fluid mask, voxel size mm, signed distance into the fluid in mm, wall
    signed distance (positive in the wall)). The sheet sits where heatx puts
    it: moved off the mid-surface so A has `split` of the void."""
    from .heatx_thermal import sheet_offset
    from .lattice import tpms

    c = sheet_offset(lambda p: tpms(kind, p, 1.0), wall / cell, split) * cell
    h = cell / n
    ax = (np.arange(n) + 0.5) * h
    pts = np.stack(np.meshgrid(ax, ax, ax, indexing="ij"), -1).reshape(-1, 3)
    d = tpms(kind, pts, cell).reshape(n, n, n) - c
    if side == "a":
        sdf = d - 0.5 * wall
    elif side == "b":
        sdf = -d - 0.5 * wall
    else:
        raise ValueError("side is 'a' or 'b'")
    wall_sd = 0.5 * wall - np.abs(d)
    return sdf > 0, h, sdf, wall_sd


# ---------------------------------------------------------------- periodic cell

def _window(u, length, chunk, passes):
    """Chunks in an averaging window: `passes` flow-throughs of the cell at
    interstitial velocity u, from 4 to 200 chunks (a creeping flow's
    flow-through is far longer than it takes to settle)."""
    return int(np.clip(np.ceil(passes * length / max(abs(u), 1e-5) / chunk), 4, 200))


def _settled(hist, w, tol):
    """The last window's mean within tol of the one before (a flow that
    settles to unsteady but statistically steady state passes too)."""
    if len(hist) < 2 * w:
        return False
    a, b = np.mean(hist[-w:]), np.mean(hist[-2 * w:-w])
    return abs(a - b) < tol * abs(a)


def _estimate_force(re, nu, dh):
    """Body force (lattice) that drives a gyroid side to about `re`: Darcy
    friction from the LBM cells' trend, f_D ~ 2.4 + 64/Re (laminar onset)."""
    u = re * nu / dh
    fd = 2.3 + 75.0 / max(re, 1.0)
    return fd * u * u / (2.0 * dh), u


def cell_flow(fluid: np.ndarray, h: float = 1.0, axis: int = 0, re=(100.0,), *, pr: float = 0.71,
              sdf: np.ndarray | None = None, wall: np.ndarray | None = None, k_ratio: float = 0.0,
              u_max: float = 0.06, nu_min: float = 0.0008, nu_max: float = 0.05, cs: float = 0.1,
              heat: bool = True, heating: str = "uniform", prt: float | None = None, tol: float = 5e-3,
              passes: float = 4.0, develop: float = 15.0, max_steps: int = 300000, chunk: int = 250,
              forces=None, nu=None, verbose: bool = False) -> dict:
    """Fully developed flow and heat transfer in one periodic cell.

    fluid: boolean mask (True = fluid) that tiles along all three axes;
    h: voxel size (any unit; results are dimensionless); axis: flow
    direction; re: target Reynolds numbers on Dh and the interstitial
    velocity (each run lands near its target: the body force comes from a
    friction estimate, and the achieved Re is what is reported). Pass
    `forces` (lattice) and `nu` instead to set the drive exactly.

    Heat: walls at T = 0 and the fluid heated, either uniformly
    (heating "uniform", the convention of the LBM cells the rating's fit
    came from) or in proportion to its axial velocity ("axial": the fully
    developed state under a mean temperature gradient along the flow, the
    textbook convention, with wall temperature uniform around each section);
    Nu on the velocity-weighted bulk temperature. For a conjugate wall, give `wall`
    (positive inside the wall, e.g. tpms_cell's 4th item) and k_ratio =
    k_wall / k_fluid: the wall conducts (its diffusivity is k_ratio times
    the fluid's: right at steady state, where heat capacity drops out), and
    the far side of the wall (where `wall` < 0 outside this fluid) stays at
    T = 0. Nu then is the overall coefficient to the far face, wall included.
    prt: turbulent Prandtl number for a subgrid heat diffusivity (the LES
    eddy viscosity over prt, e.g. 0.85, added to the fluid's; default off).
    Past Re ~2,000 the heat lattice is unstable without it, but at 64 voxels
    a cell it dominates: on a 10 mm air cell at Re ~1,300 it lifts Nu from
    38 to 64. Turbulent Nu needs finer cells (and a wall-adapting subgrid
    model) before it is a number to design on.

    Converged when, after `develop` flow-throughs of the cell, the mean over
    the last `passes` flow-throughs (and Tb's, with heat) is within `tol` of
    the window before. Flows past Re ~200 in a gyroid do not settle: on a
    10 mm air cell at Re ~250 the flow swings a few percent about one state
    for ~13 flow-throughs, then may break down into a more chaotic one with
    ~25% more friction (and back: one cell has more than one long-lived
    state, so f scatters ~20% between runs from Re ~200 to ~1,500), and a
    short stop window catches either. "unsteadiness" is the rms of the flow over the last
    window, over its mean. Successive drives start from the previous one's
    flow.

    The lattice viscosity is chosen so the interstitial velocity stays
    under u_max (Mach), never below nu_min (the LES carries the rest).
    Returns {"phi", "areaDensity" (1/h), "dh" (h), "points": [{re, f, nu,
    j (Colburn), ...}]}."""
    K = _kernels()
    fluid = np.ascontiguousarray(fluid, bool)
    shape = fluid.shape
    phi = float(fluid.mean())
    area = wetted_area(sdf, h) if sdf is not None else wetted_area(fluid.astype(np.float32) - 0.5, h)
    beta = area / (np.prod(shape) * h ** 3)
    dh = 4.0 * phi / beta
    dh_lu = dh / h
    ntype = np.where(fluid, FLUID, SOLID).astype(np.int8)
    per = np.array([1, 1, 1], np.int64)
    uin = np.zeros((3,) + shape, np.float32)
    nfl = int(fluid.sum())
    nut = np.zeros(shape, np.float32)
    prt_inv = 1.0 / prt if prt else 0.0

    if heat:
        tnode = np.where(fluid, 0, 2).astype(np.int8)
        if wall is not None and k_ratio > 0:
            tnode[(~fluid) & (wall > 0)] = 0
        tfix = np.zeros(shape, np.float32)
    runs = []
    if forces is not None:
        if nu is None:
            raise ValueError("forces need nu")
        runs = [(float(g), float(nu), None) for g in forces]
    else:
        for r in np.atleast_1d(re):
            # the fastest flow Mach allows, unless that needs a viscosity under nu_min (then the
            # flow runs faster) or over nu_max (then slower)
            nu_lu = float(np.clip(u_max * dh_lu / float(r), nu_min, nu_max))
            g, _ = _estimate_force(float(r), nu_lu, dh_lu)
            runs.append((g, nu_lu, float(r)))

    out = []
    prev = None
    for g, nu_lu, r_target in runs:
        t0 = time.time()
        tau0 = 0.5 + 3.0 * nu_lu
        alpha_f = nu_lu / pr
        g3 = [0.0, 0.0, 0.0]
        g3[axis] = g
        if prev is None:
            rho = np.ones(shape, np.float32)
            u = [np.zeros(shape, np.float32) for _ in range(3)]
            f = _equilibrium(rho, *u)
            fn = f.copy()
        else:                   # the next drive starts from the last one's flow
            f, fn, rho, u = prev

        def run(n_steps, f, fn):
            for _ in range(n_steps):
                K["step"](f, fn, ntype, tau0, cs * cs, g3[0], g3[1], g3[2], uin, 1.0, per, rho, u[0], u[1], u[2], LAMBDA, nut)
                f, fn = fn, f
            return f, fn

        hist = []
        it = 0
        flow_ok = False
        while it < max_steps:
            f, fn = run(chunk, f, fn)
            it += chunk
            m = float(u[axis].mean())
            if not np.isfinite(m):
                break
            hist.append(m)
            through = shape[axis] / max(abs(m / phi), 1e-5)          # steps per flow-through
            w = _window(m / phi, shape[axis], chunk, passes)
            # a flow that has gone steady needs no long development; an unsteady one may yet
            # change state, so it runs `develop` flow-throughs first
            steady = len(hist) >= w and np.std(hist[-w:]) < 1e-4 * abs(m)
            if (steady or it >= develop * through) and _settled(hist, w, tol):
                flow_ok = True
                break
        rec = {"force": g, "nuLattice": nu_lu, "reTarget": r_target}
        if not np.isfinite(hist[-1] if hist else np.nan):
            rec.update({"re": float("nan"), "f": float("nan"), "nu": float("nan"), "ok": False,
                        "error": "the lattice went unstable"})
            out.append(rec)
            prev = None
            continue
        nu_v = float("nan")
        if heat:
            alpha = np.where(fluid, alpha_f, alpha_f * k_ratio).astype(np.float32)
            q = 1e-5
            qvel = np.zeros(3)
            if heating == "uniform":
                qsrc = np.where(fluid, q, 0.0).astype(np.float32)
            elif heating == "axial":
                # a mean temperature gradient along the flow: each fluid node heats in
                # proportion to its axial velocity (net q per fluid node, as above)
                qsrc = np.zeros(shape, np.float32)
                qvel[axis] = q * phi / float(u[axis].mean())
            else:
                raise ValueError("heating is 'uniform' or 'axial'")
            T = np.zeros(shape, np.float32)
            gt = _equilibrium_t(T, *u)
            gtn = gt.copy()
            tb_hist, u_hist = [], []
            heat_ok = False
            n_t = 0
            while n_t < max_steps:
                for _ in range(chunk):
                    K["step"](f, fn, ntype, tau0, cs * cs, g3[0], g3[1], g3[2], uin, 1.0, per, rho, u[0], u[1], u[2], LAMBDA, nut)
                    f, fn = fn, f
                    K["step_t"](gt, gtn, tnode, u[0], u[1], u[2], alpha, tfix, qsrc, qvel, per, T, nut, prt_inv)
                    gt, gtn = gtn, gt
                n_t += chunk
                ua = u[axis]
                tb_hist.append(float((T * ua)[fluid].sum() / ua[fluid].sum()))
                u_hist.append(float(ua.mean()))
                if not (np.isfinite(tb_hist[-1]) and np.isfinite(u_hist[-1])):
                    break
                w = _window(u_hist[-1] / phi, shape[axis], chunk, passes)
                if _settled(tb_hist, w, tol) and _settled(u_hist, w, tol):
                    heat_ok = True
                    break
            it += n_t
            if not (np.isfinite(tb_hist[-1]) and np.isfinite(u_hist[-1])):
                rec.update({"re": float("nan"), "f": float("nan"), "nu": float("nan"), "ok": False,
                            "error": "the lattice went unstable with heat on"})
                out.append(rec)
                prev = None
                continue
            tb = float(np.mean(tb_hist[-w:]))
            hc = q * nfl / (beta * h * np.prod(shape) * tb)     # per wetted area, lattice units
            nu_v = hc * dh_lu / alpha_f
            hist = u_hist
        else:
            heat_ok = True
            w = _window(hist[-1] / phi, shape[axis], chunk, passes)
        last = np.asarray(hist[-w:])
        U = float(last.mean())
        rec["unsteadiness"] = float(last.std() / abs(U))
        prev = (f, fn, rho, u)
        ui = U / phi
        re_v = ui * dh_lu / nu_lu
        fd = 2.0 * g * dh_lu / ui ** 2
        rec.update({"re": float(re_v), "f": float(fd), "nu": float(nu_v),
                    "j": float(nu_v / (re_v * pr ** (1.0 / 3.0))) if heat else None,
                    "mach": float(ui * np.sqrt(3.0)), "uMax": float(np.abs(u[axis]).max() * np.sqrt(3.0)),
                    "steps": it, "seconds": round(time.time() - t0, 1), "ok": bool(flow_ok and heat_ok)})
        if verbose:
            print(f"  Re {re_v:8.1f}  f_D {fd:7.4f}  Nu {nu_v:7.2f}  ({it} steps, {rec['seconds']:.0f} s)", flush=True)
        out.append(rec)
    return {"phi": phi, "areaDensity": float(beta), "dh": float(dh), "n": list(shape), "pr": pr,
            "kRatio": k_ratio, "heating": heating if heat else None, "points": out}


# ---------------------------------------------------------------- whole networks

def _port_slice(fluid, h, origin, point, axis, radius):
    """The fluid voxels at a port's open end: within `radius` of the port's
    axis, in the last fluid layer along `axis` (pointing out of the part)
    within two voxels of `point`. Returns (mask, integer inward normal)."""
    axis = np.asarray(axis, float)
    axis /= np.linalg.norm(axis)
    k = int(np.argmax(np.abs(axis)))
    inward = np.zeros(3, int)
    inward[k] = -int(np.sign(axis[k]))
    idx = np.argwhere(fluid)
    P = origin + idx * h
    q = P - np.asarray(point, float)
    along = q @ axis
    radial = np.linalg.norm(q - along[:, None] * axis, axis=1)
    near = (radial <= radius + 0.5 * h) & (np.abs(along) <= 3.0 * h + radius)
    if not near.any():
        raise ValueError(f"no fluid within {radius:g} mm of the port at {np.round(point, 2).tolist()}")
    cand = idx[near]
    # the outermost voxel of each column along the port axis
    out = np.zeros(fluid.shape, bool)
    o = [a for a in range(3) if a != k]
    key = cand[:, o[0]] * fluid.shape[o[1]] + cand[:, o[1]]
    order = np.argsort(cand[:, k] * inward[k])          # outermost first
    seen = set()
    for i in order:
        if key[i] in seen:
            continue
        seen.add(key[i])
        out[tuple(cand[i])] = True
    return out, inward


def _boundary_flux_mean(mask_a, mask_b, value, weight):
    """Weighted mean of `value` over voxels of A that touch B (6-neighbour)."""
    from scipy import ndimage

    touch = mask_a & ndimage.binary_dilation(mask_b)
    w = weight[touch]
    if not touch.any() or w.sum() <= 0:
        return float("nan"), 0
    return float((value[touch] * w).sum() / w.sum()), int(touch.sum())


def network_flow(fluid: np.ndarray, h: float, inlet: dict, outlet: dict, *, medium="water", flow_lpm=None,
                 mass_flow=None, regions: np.ndarray | None = None, region_names=None, core: np.ndarray | None = None,
                 core_axis=None, cell: float | None = None, heat: bool = False, prt: float | None = None, u_port: float = 0.05,
                 nu_min: float = 0.0003, cs: float = 0.1, tol: float = 1e-2, max_steps: int = 300000,
                 chunk: int = 500, ramp_steps: int = 4000, develop: float = 3.0, origin=(0.0, 0.0, 0.0),
                 fields: bool = False, verbose: bool = False) -> dict:
    """Steady flow through one channel network, port to port.

    fluid: boolean voxel mask of the network (True = fluid), voxel size h
    (mm), the centre of voxel (0,0,0) at `origin` (mm). inlet / outlet:
    {"point": the port's open end (mm), "axis": pointing out of the part,
    "radius": mm}. The flow (`flow_lpm` L/min or `mass_flow` kg/s of
    `medium`, a heatx_thermal fluid name or Fluid) enters as a plug at the
    inlet and leaves at fixed pressure.

    The lattice runs at the physical Reynolds number when it can: the
    viscosity is set from the port velocity u_port (lattice, Mach ~0.09);
    where that would need a viscosity under nu_min the port runs faster
    (reported as "mach") and, past Mach 0.3, at a lower Re ("reScale").

    regions: optional int labels (one per voxel, e.g. port in, plenum in,
    core, plenum out, port out, in path order) with `region_names`; the
    report then splits the pressure drop along the path, from the
    flow-weighted total pressure where each region meets the next.
    core: optional mask of the core's voxels, and core_axis its mean flow
    direction (default: the mean velocity there), for the flow-sharing
    measure: the flow through the core's mid-section on cell-sized bins.

    Runs `develop` residence times (the network's volume over the inflow)
    and on until the inlet pressure's mean over one residence time is within
    `tol` of the one before; what it reports comes from fields averaged over
    the last residence times, since a port's jet into a plenum never goes
    steady. fields: also return those averaged fields (m/s, Pa).

    heat: also run temperature, with the core's walls held at 0 and the
    inlet at 1 (every other wall adiabatic): the effectiveness of this side
    alone against a wall at a fixed temperature, so NTU = -ln(1 - eff) and
    the core's mean h = NTU * m cp / A_wet."""
    from .heatx_thermal import FLUIDS, Fluid

    K = _kernels()
    fl = medium if isinstance(medium, Fluid) else FLUIDS[medium]
    fluid = np.pad(np.asarray(fluid, bool), 1)
    origin = np.asarray(origin, float) - h
    pad = (lambda a: None if a is None else np.pad(a, 1))
    regions, core = pad(regions), pad(core)
    shape = fluid.shape
    if mass_flow is None:
        if flow_lpm is None:
            raise ValueError("give flow_lpm or mass_flow")
        mass_flow = flow_lpm / 60000.0 * fl.rho
    qv = mass_flow / fl.rho                                  # m^3/s

    m_in, n_in = _port_slice(fluid, h, origin, inlet["point"], inlet["axis"], inlet["radius"])
    m_out, n_out = _port_slice(fluid, h, origin, outlet["point"], outlet["axis"], outlet["radius"])
    a_in = int(m_in.sum())
    u_phys = qv / (a_in * (h * 1e-3) ** 2)                   # plug velocity over the inlet voxels
    nu_phys = fl.mu / fl.rho
    re_h = u_phys * h * 1e-3 / nu_phys                       # Re on one voxel
    u_lu = u_port
    nu_lu = u_lu / re_h
    if nu_lu < nu_min:
        nu_lu = nu_min
        u_lu = nu_lu * re_h
    re_scale = 1.0
    if u_lu > 0.17:                                          # Mach 0.3
        re_scale = 0.17 / u_lu
        u_lu = 0.17
    dt = nu_lu * (h * 1e-3) ** 2 / (nu_phys / re_scale)      # s per step at the simulated Re
    vel = (h * 1e-3) / dt                                    # m/s per lattice velocity

    ntype = np.where(fluid, FLUID, SOLID).astype(np.int8)
    ntype[m_in] = INLET
    ntype[m_out] = OUTLET
    uin = np.zeros((3,) + shape, np.float32)
    for k in range(3):
        uin[k][m_in] = n_in[k] * u_lu
        uin[k][m_out] = n_out[k]
    per = np.zeros(3, np.int64)
    tau0 = 0.5 + 3.0 * nu_lu
    nut = np.zeros(shape, np.float32)
    prt_inv = 1.0 / prt if prt else 0.0
    rho = np.ones(shape, np.float32)
    u = [np.zeros(shape, np.float32) for _ in range(3)]
    f = _equilibrium(rho, *u)
    fn = f.copy()

    def run(n, f, fn):
        for _ in range(n):
            K["step"](f, fn, ntype, tau0, cs * cs, 0.0, 0.0, 0.0, uin, 1.0, per, rho, u[0], u[1], u[2], LAMBDA, nut)
            f, fn = fn, f
        return f, fn

    t0 = time.time()
    hist = []
    it = 0
    ok = False
    ramp = max(int(ramp_steps), 1)
    residence = float(fluid.sum()) / (u_lu * a_in)             # steps for the inflow to fill the network
    acc_r = np.zeros(shape)
    acc_u = np.zeros((3,) + shape)
    n_acc = 0
    while it < max_steps:
        if it < ramp:
            # start gently: an impulsive plug into still fluid near tau 0.5 throws a pressure
            # wave that blows the lattice up
            fac = min(1.0, (it + 100) / ramp)
            for k in range(3):
                uin[k][m_in] = n_in[k] * u_lu * fac
            f, fn = run(100, f, fn)
            it += 100
            continue
        f, fn = run(chunk, f, fn)
        it += chunk
        dp = float(rho[m_in].mean() - 1.0) / 3.0
        if not np.isfinite(dp):
            break
        hist.append(dp)
        if verbose and len(hist) % 20 == 0:
            print(f"  {it} steps  dp {dp:.3e}", flush=True)
        if it >= (develop - 1.0) * residence:          # time-average the fields for the report
            acc_r += rho
            for k in range(3):
                acc_u[k] += u[k]
            n_acc += 1
        # windows of one residence time (the network's volume over the inflow), after
        # `develop` of them: a plenum's jet takes several to settle into the core
        w = max(8, int(np.ceil(residence / chunk)))
        if it >= develop * residence and _settled(hist, w, tol):
            ok = True
            break
    p_unit = fl.rho * vel ** 2                               # Pa per lattice pressure
    if n_acc:
        R, U = acc_r / n_acc, acc_u / n_acc
    else:
        R, U = rho, np.stack(u)
    del acc_r, acc_u
    speed = np.sqrt((U ** 2).sum(0))
    p = (R - 1.0) / 3.0
    pt = p + 0.5 * R * speed ** 2
    dp = float(np.mean(hist[-w:])) if hist else float("nan")
    out = {"pressureDrop": dp * p_unit,
           "steps": it, "residenceSteps": residence, "seconds": round(time.time() - t0, 1), "converged": ok,
           "voxel": h, "voxels": int(fluid.sum()), "inletVoxels": a_in, "portVelocity": u_phys,
           "reVoxel": re_h, "reScale": re_scale, "mach": float(u_lu * np.sqrt(3.0)),
           "maxMach": float(speed[fluid].max() * np.sqrt(3.0))}
    # (a run at a lower Re than the physical one keeps the physical velocity, with a more
    # viscous fluid: dt carries re_scale, so pressures are already at the physical velocity)
    to_pa = p_unit

    if regions is not None:
        names = list(region_names or range(int(regions.max()) + 1))
        segs = {}
        # flow-weighted throughout, so a profile's kinetic energy counts the same at every cut
        p_in = float((pt[m_in] * speed[m_in]).sum() / speed[m_in].sum())
        bounds = []
        for i in range(len(names) - 1):
            v, n = _boundary_flux_mean(fluid & (regions == i), fluid & (regions == i + 1), pt, speed)
            bounds.append(v)
        p_out = float((pt[m_out] * speed[m_out]).sum() / speed[m_out].sum())
        pts = [p_in] + bounds + [p_out]
        for i, nme in enumerate(names):
            segs[str(nme)] = (pts[i] - pts[i + 1]) * to_pa
        out["pressureDropByRegion"] = segs
        out["totalPressureDrop"] = (p_in - p_out) * to_pa

    if core is not None:
        c = fluid & core
        idx = np.argwhere(c)
        if core_axis is None:
            mv = np.array([U[k][c].mean() for k in range(3)])
            core_axis = mv / np.linalg.norm(mv)
        ax = np.asarray(core_axis, float)
        ax /= np.linalg.norm(ax)
        P = origin + idx * h
        along = P @ ax
        mid = 0.5 * (along.min() + along.max())
        slab = np.abs(along - mid) <= 0.5 * h * 1.0001
        un = (U[:, idx[:, 0], idx[:, 1], idx[:, 2]].T @ ax)
        bin_ = cell or 4.0 * h
        e1 = np.cross(ax, [1.0, 0, 0] if abs(ax[0]) < 0.9 else [0, 1.0, 0])
        e1 /= np.linalg.norm(e1)
        e2 = np.cross(ax, e1)
        b = np.stack([np.floor(P[slab] @ e1 / bin_), np.floor(P[slab] @ e2 / bin_)], 1).astype(int)
        keys, inv = np.unique(b, axis=0, return_inverse=True)
        inv = inv.ravel()
        q = np.bincount(inv, weights=un[slab])
        nvox = np.bincount(inv)
        full = nvox >= 0.5 * np.median(nvox)
        qf = q[full]
        mean_u = float(un.mean())
        out["coreFlow"] = {
            "bins": int(full.sum()), "binSize": bin_,
            "cv": float(qf.std() / qf.mean()) if qf.size else float("nan"),
            "minOverMean": float(qf.min() / qf.mean()) if qf.size else float("nan"),
            "maxOverMean": float(qf.max() / qf.mean()) if qf.size else float("nan"),
            "stagnantShare": float(np.mean(np.abs(un) < 0.1 * abs(mean_u))),
            "reverseShare": float(np.mean(un * np.sign(mean_u) < 0)),
            "interstitialVelocity": mean_u * vel}
        out["_coreBins"] = {"keys": keys[full].tolist(), "flow": (qf / qf.mean()).tolist()}

    if heat:
        alpha = np.full(shape, nu_lu / fl.pr, np.float32)
        tnode = np.where(fluid, 0, 1).astype(np.int8)
        if core is not None:
            tnode[(~fluid) & core] = 2
        else:
            tnode[~fluid] = 2
        tnode[m_in] = 3
        tnode[m_out] = 4
        tfix = np.zeros(shape, np.float32)
        tfix[m_in] = 1.0
        T = np.zeros(shape, np.float32)
        T[m_in] = 1.0
        gt = _equilibrium_t(T, *u)
        gtn = gt.copy()
        zero = np.zeros(shape, np.float32)
        qv0 = np.zeros(3)
        th = []
        un_out = np.abs(U[:, m_out].T @ n_out.astype(float))
        hok = False
        n_t = 0
        while n_t < max_steps:
            for _ in range(chunk):
                K["step"](f, fn, ntype, tau0, cs * cs, 0.0, 0.0, 0.0, uin, 1.0, per, rho, u[0], u[1], u[2], LAMBDA, nut)
                f, fn = fn, f
                K["step_t"](gt, gtn, tnode, u[0], u[1], u[2], alpha, tfix, zero, qv0, per, T, nut, prt_inv)
                gt, gtn = gtn, gt
            n_t += chunk
            th.append(float((T[m_out] * un_out).sum() / un_out.sum()))
            if len(th) >= 16:
                a, b = np.mean(th[-8:]), np.mean(th[-16:-8])
                if abs(a - b) < tol * max(abs(1 - a), 1e-3):
                    hok = True
                    break
        t_out = float(np.mean(th[-8:]))
        eff = 1.0 - t_out
        ntu = -np.log(max(1.0 - eff, 1e-9))
        out["heat"] = {"effectiveness": eff, "ntu": float(ntu), "steps": n_t, "converged": hok}
        if core is not None:
            from skimage.measure import marching_cubes

            # the core's wetted area: the fluid surface inside the core
            fld = np.where(fluid, 1.0, -1.0).astype(np.float32)
            v, fc, _, _ = marching_cubes(fld, 0.0, spacing=(h, h, h))
            cen = (v[fc[:, 0]] + v[fc[:, 1]] + v[fc[:, 2]]) / 3.0
            ci = np.clip(np.round(cen / h).astype(int), 0, np.array(shape) - 1)
            inside = core[ci[:, 0], ci[:, 1], ci[:, 2]]
            tri = v[fc[inside]]
            area = float(0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1).sum())
            hc = ntu * mass_flow * fl.cp / (area * 1e-6)
            out["heat"].update({"coreWettedArea": area, "h": float(hc)})
    out["simulated"] = {"nuLattice": nu_lu, "uPortLattice": u_lu, "dt": dt}
    if fields:
        # time-averaged fields (padded grid: voxel (0,0,0) is at `origin`), for renders
        out["fields"] = {"velocity": (U * vel).astype(np.float32), "pressure": (p * p_unit).astype(np.float32),
                         "fluid": fluid, "origin": origin, "h": h}
    return out


def exchanger_flow(hx, fluid: str, duty: dict | None = None, *, pitch: float | None = None, heat: bool = False,
                   design=None, verbose: bool = False, **kw) -> dict:
    """Simulate one fluid's network of a heatx design, port to port, and set
    it beside the rating's prediction for the same side.

    hx: a HeatExchangerSpec (or a heatExchanger spec dict); fluid: its name
    (hx.names); duty: the rating's duty ({name: {"fluid", "lPerMin" |
    "kgPerS"}, "wall"}), by default the spec's own. pitch: voxel size (mm,
    default wall / 2.5, at least two voxels across each sheet's gaps).
    design: (mesh, report, spaces) from design_heat_exchanger(...,
    fluid_meshes=True), to skip rebuilding it. The pressure drop is split
    into inlet port, inlet plenum, core (mouths included), outlet plenum and
    outlet port, and the core's flow sharing is measured on cell-sized
    bins. Extra keywords go to network_flow."""
    from scipy import ndimage

    from .heatx import design_heat_exchanger, heat_exchanger_from_spec
    from .heatx_thermal import _fluid, _flow, rate_design
    from .voxels import mesh_solid

    spec_duty = None
    if isinstance(hx, dict):
        spec_duty = hx.get("duty")
        hx = heat_exchanger_from_spec(hx)
    duty = duty or spec_duty
    if duty is None:
        raise ValueError("give a duty: {name: {\"fluid\": \"water\", \"lPerMin\": 1}, ...}")
    if fluid not in hx.names:
        raise ValueError(f"fluid is one of {list(hx.names)}")
    if design is None:
        design = design_heat_exchanger(hx, fluid_meshes=True)
    mesh, report, spaces = design
    pitch = pitch or hx.wall / 2.5
    solid, origin, _ = mesh_solid(spaces[fluid], pitch, pad=1)
    med = _fluid(duty[fluid].get("fluid", "water"))
    m = _flow(duty[fluid], med)

    idx = np.indices(solid.shape).reshape(3, -1).T
    P = origin + idx * pitch
    in_core = (hx.core.sdf(P) < 0).reshape(solid.shape)
    zi, zo = report["zones"][f"{fluid}-in"], report["zones"][f"{fluid}-out"]
    r_port = 0.5 * hx.port_diameter

    def port_region(z):
        p1, ax = np.asarray(z["port"][0], float), np.asarray(z["port"][1], float)
        q = P - p1
        a = q @ ax
        rad = np.linalg.norm(q - a[:, None] * ax, axis=1)
        return ((rad <= r_port + pitch) & (a >= -hx.port_length) & (a <= 3.0)).reshape(solid.shape), p1 + 2.0 * ax, ax

    port_in, end_in, ax_in = port_region(zi)
    port_out, end_out, ax_out = port_region(zo)
    port_in &= solid
    port_out &= solid & ~port_in
    rest = solid & ~in_core & ~port_in & ~port_out
    lab, n = ndimage.label(rest)
    near_in = ndimage.binary_dilation(port_in) & rest
    ids_in = set(np.unique(lab[near_in]).tolist()) - {0}
    regions = np.full(solid.shape, -1, int)
    regions[port_in] = 0
    regions[rest & np.isin(lab, list(ids_in))] = 1
    regions[solid & in_core] = 2
    regions[rest & ~np.isin(lab, list(ids_in))] = 3
    regions[port_out] = 4
    core_axis = None
    if hx.layout == "cross":
        k = hx.axes[hx.names.index(fluid)]
        core_axis = np.eye(3)[k]
    elif fluid == hx.names[0]:
        core_axis = np.eye(3)[hx.axes[0]]
    res = network_flow(solid, pitch, {"point": end_in, "axis": ax_in, "radius": r_port},
                       {"point": end_out, "axis": ax_out, "radius": r_port}, medium=med, mass_flow=m,
                       regions=regions, region_names=["portIn", "plenumIn", "core", "plenumOut", "portOut"],
                       core=in_core, core_axis=core_axis, cell=hx.cell, heat=heat, origin=origin,
                       verbose=verbose, **kw)
    try:
        rd = dict(duty)
        if "wall" not in rd:
            rd["wall"] = "petg"
        rating = rate_design(hx, report, rd)["fluids"][fluid]
        res["rating"] = {"pressureDrop": rating["pressureDrop"], "core": rating["pressureDropCore"],
                         "mouths": rating["pressureDropMouths"], "ports": rating["pressureDropPorts"],
                         "re": rating["re"], "h": rating["h"], "hSpread": rating["hSpread"]}
    except (KeyError, ValueError) as e:      # the other fluid's duty missing: no rating to compare
        res["rating"] = {"error": str(e)}
    res["fluid"] = fluid
    return res
