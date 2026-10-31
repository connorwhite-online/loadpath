"""
Whole heat exchangers, simulated cheaply: the TPMS core as a porous
medium, the plenums and ports as open flow, both fluids and the wall
between them as three temperature fields on one coarse grid.

A resolved simulation (LBM, flow.py) needs voxels a fraction of the wall
thick, so a whole core is millions of cells and hours. Averaged over a cell
of the lattice, though, a sheet gyroid's channels behave like a porous
medium whose laws are known from cell-level runs:

    -grad p = (mu / K + rho C_F |u| / sqrt K) u          (Darcy-Forchheimer)
    K = 0.0157 phi^2.72 a^2,  C_F = 0.0784 phi^-2.04     (Savoldi et al. 2026)

with u the superficial velocity, phi the fluid's share of the core and a
the cell (the same law heatx_thermal rates on, checked there against the
1KZ cooler's LBM cells), and each fluid trades heat with the wall through

    q = h a_w (T_fluid - T_wall),  Nu = 0.49 Re^0.62 Pr^0.4   (Reynolds 2023)

on the local Re (closures "lbm": the LBM cells' 1.66 Re^0.464 Pr^0.217,
and K and C_F refitted to the same cells, lbm_medium), so a slow
corner of the core gets the weak h it deserves. That is cheap on voxels
a few mm across (a core is 10-100 thousand cells, not millions) and gives
what the one-number epsilon-NTU rating cannot:

- how the flow shares out over the core (a port jetting into a shallow
  plenum starves the far corners), as a uniformity index and a map;
- the pressure field port to port, core and headers apart;
- the temperature of both fluids and the wall everywhere (hot spots, the
  wall's own conduction along the flow, which caps a metal counterflow
  core), and the outlet temperatures that follow;
- so header and port shapes can be compared and tuned (`sweep`).

The core is not uniform: outside its own zones each fluid is sealed off
within `cap` of the surface, and with a build axis its channels close under
surfaces that look up. exchanger_domain measures each fluid's local share
of the core from the design's own grid (averaged over about a cell) and
scales K, C_F and the wetted area to it. On the 30 mm crossflow cube this
is what sends the flow away from the +z edge, as the full LBM shows, and
doubles the core's pressure drop (144 -> 252 Pa; LBM about 250 after its
LES correction, the rating 127).

Flow, one fluid at a time (they only meet through heat): pseudo-transient
pressure correction on a staggered (MAC) grid. Open cells (plenums,
ports) carry momentum (upwind advection and viscosity, incompressible),
core cells Darcy-Forchheimer drag with the fluid's inertia rho / phi; a
face between the two takes half the core's drag. Inlets are volume
sources in the port's end cells, outlets hold p = 0. Each step solves one
pressure correction with algebraic multigrid; the drag is implicit, so the
core relaxes in a few hundred steps and the steady state does not depend
on the step.

Heat, once the flow is steady: steady upwind advection-conduction for each
fluid (its own cells), conduction for the wall (the core's cells, at the
sheet's effective conductivity, (1 - phi_sheet) k / 1.5), and the exchange
between them, h' a V with h' = 1 / (1/h + t / 2k) so fluid-wall-fluid is the
rating's 1/U = 1/h_A + t/k + 1/h_B. Headers and the outer shell are
adiabatic. One sparse solve (ILU-preconditioned GMRES).

Not captured: turbulence in the headers beyond the coarse grid's own
mixing (a port jet spreads faster than in life, so maldistribution is if
anything understated), entry effects in the channels (the Nu law is fully
developed), heat through the plenum walls, and the minor losses of a
port's own mouth (`portLoss` adds the rating's 1.5 velocity heads per port
pair so totals compare).

Units: SI inside; mm for positions and the grid at the edges.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage, sparse

from .heatx_thermal import WALL_CONDUCTIVITY, _flow, _fluid

__all__ = ["PorousDomain", "Medium", "exchanger_domain", "solve_flow", "solve_heat", "simulate_exchanger",
           "simulate_design", "sweep", "savoldi_medium", "lbm_medium", "summarize", "plot_fields"]


@dataclass
class Medium:
    """One fluid's side of the core, homogenized: share of the core volume,
    wetted area per volume (1/m), hydraulic diameter (m), permeability K
    (m^2) and Forchheimer coefficient C_F."""
    phi: float
    area_density: float
    dh: float
    K: float
    CF: float
    nK: float = 2.72                # K ~ phi^nK and C_F ~ phi^nC, for a local porosity
    nC: float = -2.04


def savoldi_medium(phi, area_density, dh, cell):
    """The Savoldi 2026 network-gyroid law (as heatx_thermal): `cell` in m."""
    return Medium(float(phi), float(area_density), float(dh), 0.0157 * phi ** 2.72 * cell ** 2,
                  0.0784 * phi ** -2.04)


def lbm_medium(phi, area_density, dh, cell):
    """The same form refitted to periodic gyroid-sheet cells run with LBM
    (/mnt/project-files/flow/gyroid-cells.json: the 1KZ cooler's 39 BGK
    points, loadpath.flow's TRT re-runs and its creeping-flow limit, f Re
    183 on the a10 air cell): K = 0.00932 phi^2.257 a^2, C_F = 0.0569
    phi^-2.283. Median ratio 0.99, 90% of points within 13% (the rest is the
    ~20% scatter between flow states past Re ~200), against Savoldi's 1.08
    and 18%; the gain is at phi 0.17-0.26, below Savoldi's range, where
    Savoldi reads 10-29% high."""
    return Medium(float(phi), float(area_density), float(dh), 0.00932 * phi ** 2.257 * cell ** 2,
                  0.0569 * phi ** -2.283, 2.257, -2.283)


# local porosity: averaging window (cells) and a percolation-like threshold below
# which a band of partial channels stops carrying flow
_PERC = {"window": 0.5, "rc": 0.0}

# closures: (pressure-drop law, Nu law)
_CLOSURES = {"design": (savoldi_medium, "design"), "lbm": (lbm_medium, "lbm")}


@dataclass
class PorousDomain:
    """A coarse grid (cell `h` m, corner `lo` mm) with the core and, per
    fluid, its cells (core plus its open plenums and ports), its inlet and
    outlet cells, and its homogenized medium in the core. Arrays are padded
    so the outer layer is never a fluid cell."""
    h: float
    lo: np.ndarray
    core: np.ndarray
    cells: list
    inlet: list
    outlet: list
    media: list
    names: tuple = ("A", "B")
    wall: float = 0.8e-3            # m, the sheet
    phi_sheet: float = 0.8          # the core's open share (both fluids)
    extra: dict = field(default_factory=dict)

    @property
    def dims(self):
        return self.core.shape

    def centers(self, mask):
        """Cell centres (mm) of a mask."""
        return self.lo + (np.argwhere(mask) + 0.5) * self.h * 1e3


# ---------------------------------------------------------------- the grid

def _shift(a, d, ax):
    """b[c] = a[c + d e_ax], zero past the edge (the domain is padded)."""
    out = np.zeros_like(a)
    n = a.shape[ax]
    src = [slice(None)] * 3
    dst = [slice(None)] * 3
    if d > 0:
        src[ax], dst[ax] = slice(d, n), slice(0, n - d)
    else:
        src[ax], dst[ax] = slice(0, n + d), slice(-d, n)
    out[tuple(dst)] = a[tuple(src)]
    return out


def _faces(m):
    """Face masks: face ax at cell c joins c and c + e_ax."""
    return [m & _shift(m, 1, ax) for ax in range(3)]


def _cell_speed(U, active):
    """|u| at cell centres from the face velocities."""
    s = np.zeros(active.shape)
    for ax in range(3):
        s += (0.5 * (U[ax] + _shift(U[ax], -1, ax))) ** 2
    return np.sqrt(s) * active


# ---------------------------------------------------------------- flow

def _local(dom, f):
    """Per cell: this fluid's porosity (1 in open cells), permeability and
    Forchheimer coefficient, and its share relative to the medium's (which
    scales the wetted area). Where the design seals or closes channels near
    the core's surface (dom.extra["phi"]), the local share is lower."""
    med = dom.media[f]
    core = dom.core & dom.cells[f]
    loc = (dom.extra.get("phi") or [None, None])[f]
    r = np.where(core, 1.0 if loc is None else np.maximum(loc / med.phi, 0.05), 1.0)
    phi = np.where(core, med.phi * r, 1.0)
    rc = _PERC["rc"]
    rk = np.maximum((r - rc) / (1.0 - rc), 0.02) if rc > 0 else r
    K = med.K * rk ** med.nK
    CF = med.CF * rk ** med.nC
    return phi, K, CF, r


def _face_props(dom, f, fluid):
    """Per face: inertia (rho / phi, averaged over the two cells), whether
    it's a core face (drag), and the share of its length in the core."""
    m = dom.cells[f]
    core = dom.core & m
    phi = _local(dom, f)[0]
    inert = fluid.rho / phi
    out = []
    for ax, fa in enumerate(_faces(m)):
        share = 0.5 * (core.astype(float) + _shift(core, 1, ax).astype(float)) * fa  # 0 open, 0.5 interface, 1 core
        rho_eff = np.where(fa, 0.5 * (inert + _shift(inert, 1, ax)), fluid.rho)
        out.append((fa, share, rho_eff))
    return out


def solve_flow(dom: PorousDomain, f: int, fluid, mass_flow: float, max_steps: int = 4000, tol: float = 1e-4,
               cfl: float = 0.4, local_dt: float = 20.0, progress=None):
    """Steady flow of one fluid through its cells: inflow `mass_flow` kg/s
    spread over its inlet cells, out at p = 0 through its outlet cells.
    Returns face velocities (superficial, m/s), cell pressures (Pa) and
    convergence info."""
    import pyamg

    fluid = _fluid(fluid)
    m = dom.cells[f]
    h = dom.h
    med = dom.media[f]
    rho, mu = fluid.rho, fluid.mu
    nu = mu / rho
    q = mass_flow / rho                                   # m^3/s
    inlet, outlet = dom.inlet[f] & m, dom.outlet[f] & m
    if not inlet.any() or not outlet.any():
        raise ValueError(f"fluid {dom.names[f]} has no inlet or outlet cells on the grid")
    src = np.zeros(m.shape)
    src[inlet] = q / inlet.sum()
    props = _face_props(dom, f, fluid)
    fa = [p[0] for p in props]
    share = [p[1] for p in props]
    rho_eff = [p[2] for p in props]
    open_face = [a & (s == 0) for a, s in zip(fa, share)]
    # a reference speed for the step: the port's mean speed
    n_port = max(inlet.sum() ** (2.0 / 3.0), 1.0)
    u_ref = max(q / (n_port * h * h), 1e-6)
    dt_visc = 0.15 * h * h / nu
    dt = min(cfl * h / (2.0 * u_ref), dt_visc)

    # numbering of the unknowns (outlet cells hold p = 0)
    free = m & ~outlet
    idx = -np.ones(m.shape, dtype=np.int64)
    idx[free] = np.arange(int(free.sum()))
    n = int(free.sum())

    _, K_c, CF_c, _ = _local(dom, f)
    in_core = dom.core & m

    def drag(U):
        """Face drag R (Pa s / m^2): Darcy-Forchheimer, half of each cell's
        on either side of the face (open cells none)."""
        speed = _cell_speed(U, m)
        rc = np.where(in_core, mu / K_c + rho * CF_c * speed / np.sqrt(K_c), 0.0)
        return [0.5 * (rc + _shift(rc, 1, ax)) * fa[ax] for ax in range(3)]

    def assemble(a):
        """The pressure-correction matrix for face coefficients a (Pa s / m^2 per
        unit speed): flux = -h^2 (dp / h) / a."""
        rows, cols, vals = [], [], []
        diag = np.zeros(n)
        for ax in range(3):
            g = np.where(fa[ax], h / np.where(fa[ax], a[ax], 1.0), 0.0)
            i0 = idx
            i1 = _shift(idx + 1, 1, ax) - 1             # index of c + e_ax (-1 where none)
            sel = fa[ax]
            gi = g[sel]
            a0, a1 = i0[sel], i1[sel]
            for p_, q_ in ((a0, a1), (a1, a0)):
                ok = p_ >= 0
                np.add.at(diag, p_[ok], gi[ok])
                both = ok & (q_ >= 0)
                rows.append(p_[both])
                cols.append(q_[both])
                vals.append(-gi[both])
        rows.append(np.arange(n))
        cols.append(np.arange(n))
        vals.append(diag)
        L = sparse.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n, n))
        return L

    # initial guess: Darcy everywhere (open faces with an inertial resistance at the port speed)
    U = [np.zeros(m.shape) for _ in range(3)]
    R = drag(U)
    a_init = [np.where(share[ax] > 0, R[ax] + rho * u_ref / h * (1 - share[ax]), rho * u_ref / h) for ax in range(3)]
    L = assemble(a_init)
    ml = pyamg.smoothed_aggregation_solver(L, symmetry="symmetric", max_coarse=500)
    p = np.zeros(m.shape)
    p[free] = ml.solve(src[free], tol=1e-10, accel="cg", maxiter=200)
    for ax in range(3):
        U[ax] = np.where(fa[ax], -(_shift(p, 1, ax) - p) / h / np.where(fa[ax], a_init[ax], 1.0), 0.0)

    hist = []
    a = None
    ml = None
    converged = False
    step = 0
    for step in range(max_steps):
        R = drag(U)
        if step % 25 == 0 or a is None:
            # a_f = rho' / dt + R: implicit drag; refresh the correction's matrix now and then
            # local pseudo-time steps (only the steady state counts): open faces by their own
            # speed's CFL, up to 20x the port's, core faces long enough for the drag to settle
            speed = _cell_speed(U, m)
            dt_f = []
            for ax in range(3):
                sp = np.maximum(speed, _shift(speed, 1, ax))
                d_open = np.clip(cfl * h / (2.0 * np.maximum(sp, 1e-12)), dt, min(local_dt * dt, dt_visc))
                dt_f.append(np.where(share[ax] > 0, np.maximum(d_open, 2.0 * rho_eff[ax] / np.maximum(R[ax], 1e-30)),
                                     d_open))
            a = [rho_eff[ax] / dt_f[ax] + R[ax] for ax in range(3)]
            ml = pyamg.smoothed_aggregation_solver(assemble(a), symmetry="symmetric", max_coarse=500)
            inert = [rho_eff[ax] / dt_f[ax] for ax in range(3)]
        # explicit momentum in open faces: upwind advection, viscosity (no-slip walls)
        Ustar = []
        for ax in range(3):
            u = U[ax]
            rhs = inert[ax] * u - (_shift(p, 1, ax) - p) / h * fa[ax]
            of = open_face[ax]
            if of.any():
                adv = np.zeros(m.shape)
                lap = np.zeros(m.shape)
                for d in range(3):
                    if d == ax:
                        vd = u
                    else:
                        V = U[d]
                        vd = 0.25 * (V + _shift(V, 1, ax) + _shift(V, -1, d) + _shift(_shift(V, 1, ax), -1, d))
                    up = _shift(u, 1, d)
                    dn = _shift(u, -1, d)
                    adv += np.where(vd > 0, vd * (u - dn), vd * (up - u)) / h
                    fup = _shift(fa[ax], 1, d)
                    fdn = _shift(fa[ax], -1, d)
                    if d == ax:
                        lap += (up + dn - 2 * u) / (h * h)
                    else:
                        # a missing neighbour face across a wall: no-slip ghost -u
                        lap += (np.where(fup, up, -u) + np.where(fdn, dn, -u) - 2 * u) / (h * h)
                rhs = rhs + np.where(of, -rho * adv + mu * lap, 0.0)
            Ustar.append(np.where(fa[ax], rhs / np.where(fa[ax], a[ax], 1.0), 0.0))
        # pressure correction: div(u) = source
        div = src.copy()
        for ax in range(3):
            div -= h * h * (Ustar[ax] - _shift(Ustar[ax], -1, ax))
        dp = np.zeros(m.shape)
        dp[free] = ml.solve(div[free], tol=1e-8, accel="cg", maxiter=100)
        change = 0.0
        for ax in range(3):
            un = np.where(fa[ax], Ustar[ax] - (_shift(dp, 1, ax) - dp) / h / np.where(fa[ax], a[ax], 1.0), 0.0)
            change = max(change, float(np.abs(un - U[ax]).max()))
            U[ax] = un
        p = p + dp
        rel = change / u_ref
        if not np.isfinite(rel) or rel > 50.0:
            # the explicit header momentum outran its local step: start again, more carefully
            if local_dt <= 1.0 and cfl < 0.1:
                raise ValueError(f"fluid {dom.names[f]}'s flow solve diverged")
            return solve_flow(dom, f, fluid, mass_flow, max_steps, tol, cfl * (0.7 if local_dt <= 1.0 else 1.0),
                              max(1.0, local_dt / 4.0), progress)
        hist.append(rel)
        if progress is not None and step % 100 == 0:
            progress(step=step, change=rel)
        if step > 20 and rel < tol:
            converged = True
            break
    # mass balance check: what leaves through the outlet cells
    div = np.zeros(m.shape)
    for ax in range(3):
        div += h * h * (U[ax] - _shift(U[ax], -1, ax))
    out_flow = -float(div[outlet].sum())
    return {"U": U, "p": p, "steps": step + 1, "converged": converged, "change": hist[-1] if hist else 0.0,
            "outflow": out_flow * rho, "massFlow": mass_flow, "dt": dt, "uRef": u_ref}


# ---------------------------------------------------------------- heat

_NU = {"design": (0.49, 0.62, 0.4), "lbm": (1.66, 0.464, 0.217)}
_NU_MIN = 3.0          # a floor for near-stagnant channels (conduction across a channel, roughly)


def _pairs(m, ax, idx):
    """Index pairs (c, c + e_ax) of the faces inside mask m."""
    sel = m & _shift(m, 1, ax)
    return sel, idx[sel], (_shift(idx + 1, 1, ax) - 1)[sel]


def _laplacian(n, pairs_g):
    """Sparse conduction matrix from (i, j, conductance) lists."""
    rows, cols, vals = [], [], []
    diag = np.zeros(n)
    for i, j, g in pairs_g:
        np.add.at(diag, i, g)
        np.add.at(diag, j, g)
        rows += [i, j]
        cols += [j, i]
        vals += [-g, -g]
    rows.append(np.arange(n))
    cols.append(np.arange(n))
    vals.append(diag)
    return sparse.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n, n))


def solve_heat(dom: PorousDomain, fluids, flows, t_in, k_wall, nu="design"):
    """Steady temperatures of both fluids and the wall, given each fluid's
    steady flow (solve_flow's results) and inlet temperature (deg C).
    `k_wall` W/m K, the sheet's material.

    Advection is upwind (each cell's outflow leaves at the cell's
    temperature), but the heat a cell trades with the wall is driven by the
    mean temperature along its stream, between what flows in and what flows
    out, weighted as in a single stream exchanging at fixed wall temperature
    (w = (1 - (1 - e^-N) / N) / (1 - e^-N), N the cell's own NTU): plain
    upwind would use the outflow's and run 15% low on a crossflow core 10
    cells across."""
    from scipy.sparse.linalg import LinearOperator, gmres, spilu

    h = dom.h
    vol = h ** 3
    c0, e_re, e_pr = _NU[nu] if isinstance(nu, str) else nu
    t_half = 0.5 * dom.wall / k_wall
    core = dom.core
    wcells = core & dom.cells[0] & dom.cells[1]
    frac = dom.extra.get("coreFraction", 1.0)
    n_w = int(wcells.sum())
    iw = -np.ones(core.shape, dtype=np.int64)
    iw[wcells] = np.arange(n_w)
    blocks, rhs, couple, h_maps = [], [], [], []
    g_sum = np.zeros(n_w)
    wall_rhs = np.zeros(n_w)
    for f in (0, 1):
        fl = _fluid(fluids[f])
        m = dom.cells[f]
        n = int(m.sum())
        i = -np.ones(core.shape, dtype=np.int64)
        i[m] = np.arange(n)
        med = dom.media[f]
        phi_c, _, _, r_c = _local(dom, f)
        U = flows[f]["U"]
        rc = fl.rho * fl.cp
        # inflow matrix P[i, j] = capacity flowing from j into i (W/K); conduction
        pr, pc, pv, cond = [], [], [], []
        for ax in range(3):
            sel, a0, a1 = _pairs(m, ax, i)
            F = U[ax][sel] * h * h * rc
            pos = F > 0
            pr += [a1[pos], a0[~pos]]
            pc += [a0[pos], a1[~pos]]
            pv += [F[pos], -F[~pos]]
            cond.append((a0, a1, fl.k * 0.5 * (phi_c + _shift(phi_c, 1, ax))[sel] * h))
        P = sparse.csr_matrix((np.concatenate(pv), (np.concatenate(pr), np.concatenate(pc))), shape=(n, n))
        inl = (dom.inlet[f] & m)[m]
        s = np.where(inl, flows[f]["massFlow"] * fl.cp / inl.sum(), 0.0)
        fin = np.asarray(P.sum(axis=1)).ravel() + s
        A = sparse.diags(fin) - P + _laplacian(n, cond)
        b = s * t_in[f]
        # exchange with the wall, on the local Re
        speed = _cell_speed(U, m)
        re = fl.rho * (speed / phi_c) * med.dh / fl.mu
        hc = np.maximum(c0 * re ** e_re * fl.pr ** e_pr, _NU_MIN) * fl.k / med.dh
        h_maps.append(np.where(wcells, hc, 0.0))
        G = (med.area_density * r_c * vol * frac / (1.0 / hc + t_half))[wcells]
        fi = i[wcells]
        N = G / np.maximum(fin[fi], 1e-30)
        en = np.exp(-np.minimum(N, 50.0))
        w = np.where(N > 1e-6, (1.0 - (1.0 - en) / np.maximum(N, 1e-30)) / np.maximum(1.0 - en, 1e-30), 0.5)
        w = np.where(fin[fi] > 0, w, 1.0)
        # the driving temperature at wall cell k: w T_i + (1 - w) (P T + s t_in)_i / fin_i
        E = sparse.csr_matrix((np.ones(n_w), (np.arange(n_w), fi)), shape=(n_w, n))
        dinv = np.where(fin[fi] > 0, (1.0 - w) / np.maximum(fin[fi], 1e-30), 0.0)
        Mf = sparse.diags(w) @ E + sparse.diags(dinv) @ (E @ P)
        m0 = dinv * s[fi] * t_in[f]
        GM = sparse.diags(G) @ Mf
        A = A + E.T @ GM
        b = b - E.T @ (G * m0)
        blocks.append((A, -(E.T @ sparse.diags(G)), -GM))
        rhs.append(b)
        g_sum += G
        wall_rhs += G * m0
        couple.append((i, m))
    kw = (1.0 - dom.phi_sheet) * k_wall / 1.5
    W = _laplacian(n_w, [_pairs(wcells, ax, iw)[1:] + (np.full(int(_pairs(wcells, ax, iw)[0].sum()), kw * h),)
                         for ax in range(3)]) + sparse.diags(g_sum)
    (A0, C0, R0), (A1, C1, R1) = blocks
    K = sparse.bmat([[A0, None, C0], [None, A1, C1], [R0, R1, W]], format="csc")
    b = np.concatenate([rhs[0], rhs[1], wall_rhs])
    n_tot = K.shape[0]
    if n_tot < 200_000:
        x = sparse.linalg.spsolve(K, b)
    else:
        ilu = spilu(K, drop_tol=1e-5, fill_factor=20)
        x, _ = gmres(K, b, M=LinearOperator(K.shape, ilu.solve), rtol=1e-10, restart=50, maxiter=200,
                     x0=np.full(n_tot, float(np.mean(t_in))))
    T, off = [], 0
    for i, m in couple:
        t = np.full(core.shape, np.nan)
        t[m] = x[off:off + int(m.sum())]
        off += int(m.sum())
        T.append(t)
    tw = np.full(core.shape, np.nan)
    tw[wcells] = x[off:]
    return {"T": T, "Twall": tw, "h": h_maps, "wallCells": wcells}


# ---------------------------------------------------------------- results

def _outlet_temperature(dom, f, flow, T):
    """Mixed-cup temperature leaving through the outlet cells: what flows
    into them, at their temperature."""
    h = dom.h
    out = dom.outlet[f] & dom.cells[f]
    U = flow["U"]
    # net inflow into each outlet cell = -div
    div = np.zeros(out.shape)
    for ax in range(3):
        div += h * h * (U[ax] - _shift(U[ax], -1, ax))
    q = np.maximum(-div, 0.0) * out
    return float((q * np.nan_to_num(T)).sum() / max(q.sum(), 1e-30))


def _core_entry(dom, f, flow):
    """Volume flux (m^3/s) into the core through each face between this
    fluid's open cells and the core (positive in)."""
    m = dom.cells[f]
    core = dom.core & m
    opn = m & ~dom.core
    h = dom.h
    out = []
    for ax in range(3):
        U = flow["U"][ax]
        # open at c, core at c + e: flux in = +U; core at c, open at c + e: in = -U
        a = opn & _shift(core, 1, ax)
        b = core & _shift(opn, 1, ax)
        out.append(U[a] * h * h)
        out.append(-U[b] * h * h)
    q = np.concatenate(out)
    return q


def _midplane(dom, f, flow, cell_mm):
    """Flow sharing across the core's middle, as flow.py's coreFlow
    measures it: the slab at the core's mid-plane normal to this fluid's
    mean core flow, binned on cell-sized squares from the world origin;
    bins under half the median cell count dropped. Returns cv, min and max
    over the mean, and the bins (flow over mean, rows and columns along
    the other two axes ascending)."""
    m = dom.cells[f] & dom.core
    U = flow["U"]
    mean = [float((0.5 * (U[ax] + _shift(U[ax], -1, ax)))[m].sum()) for ax in range(3)]
    ax = int(np.argmax(np.abs(mean)))
    sgn = np.sign(mean[ax])
    idx = np.argwhere(m)
    k = int(round(0.5 * (idx[:, ax].min() + idx[:, ax].max())))
    sl = [slice(None)] * 3
    sl[ax] = k
    sl = tuple(sl)
    u = (sgn * 0.5 * (U[ax] + _shift(U[ax], -1, ax)))[sl]
    mm = m[sl]
    other = [a for a in range(3) if a != ax]
    ij = np.argwhere(mm)
    pos = dom.lo[other] + (ij + 0.5) * dom.h * 1e3
    b = np.floor(pos / cell_mm).astype(int)
    b -= b.min(axis=0)
    shape = tuple(b.max(axis=0) + 1)
    q = np.zeros(shape)
    n = np.zeros(shape)
    np.add.at(q, (b[:, 0], b[:, 1]), u[mm])
    np.add.at(n, (b[:, 0], b[:, 1]), 1)
    ok = n >= 0.5 * np.median(n[n > 0])
    qm = q[ok].mean()
    return {"axis": "xyz"[ax], "cv": float(q[ok].std() / qm), "minOverMean": float(q[ok].min() / qm),
            "maxOverMean": float(q[ok].max() / qm), "bins": np.round(q / qm, 3).tolist()}


def _uniformity(q):
    """Flow uniformity over the faces where the fluid enters the core
    (Weltens' index, 1 = perfectly even), its spread, and the share of
    faces taking less than half the mean."""
    q = q[q > 0]
    if len(q) == 0:
        return {}
    mean = float(q.mean())
    return {"uniformity": float(1.0 - np.abs(q - mean).sum() / (2.0 * len(q) * mean)),
            "cov": float(q.std() / mean), "minOverMean": float(q.min() / mean), "maxOverMean": float(q.max() / mean),
            "starvedShare": float((q < 0.5 * mean).mean()), "entryFaces": int(len(q))}


def _mean_p(dom, f, flow, mask):
    p = flow["p"]
    sel = mask & dom.cells[f]
    return float(p[sel].mean()) if sel.any() else float("nan")


def _plenum_faces(dom, f, which):
    """Core cells next to this fluid's open cells that connect (through the
    open space) to its inlet or its outlet."""
    m = dom.cells[f]
    opn = m & ~dom.core
    lbl, _ = ndimage.label(opn)
    seeds = dom.inlet[f] if which == "in" else dom.outlet[f]
    keep = np.isin(lbl, np.unique(lbl[seeds & opn])) & (lbl > 0)
    near = ndimage.binary_dilation(keep) & dom.core & m
    return near


def summarize(dom, fluids, flows, heat, t_in):
    out = {"grid": {"voxel": dom.h * 1e3, "dims": list(dom.dims),
                    "coreCells": int(dom.core.sum()),
                    "cells": {n: int(c.sum()) for n, c in zip(dom.names, dom.cells)}},
           "fluids": {}}
    caps = []
    for f, name in enumerate(dom.names):
        fl = _fluid(fluids[f])
        flow = flows[f]
        p = flow["p"]
        m = dom.cells[f]
        p_in = _mean_p(dom, f, flow, dom.inlet[f])
        core_in = _plenum_faces(dom, f, "in")
        core_out = _plenum_faces(dom, f, "out")
        dp_core = _mean_p(dom, f, flow, core_in) - _mean_p(dom, f, flow, core_out)
        speed = _cell_speed(flow["U"], m)
        cs = speed[dom.core & m]
        med = dom.media[f]
        cap = flow["massFlow"] * fl.cp
        caps.append(cap)
        row = {"massFlow": flow["massFlow"], "pressureDrop": p_in, "pressureDropCore": dp_core,
               "pressureDropHeaders": p_in - dp_core,
               "converged": flow["converged"], "steps": flow["steps"],
               "massBalance": flow["outflow"] / flow["massFlow"],
               "coreVelocity": {"mean": float(cs.mean()), "p05": float(np.percentile(cs, 5)),
                                "p95": float(np.percentile(cs, 95))},
               "coreRe": float(fl.rho * cs.mean() / med.phi * med.dh / fl.mu),
               "deadShare": float((cs < 0.1 * cs.mean()).mean()),
               "entry": _uniformity(_core_entry(dom, f, flow)),
               "midplane": _midplane(dom, f, flow, dom.extra.get("cell", 8.0))}
        if heat is not None:
            row["tIn"] = t_in[f]
            row["tOut"] = _outlet_temperature(dom, f, flow, heat["T"][f])
            hm = heat["h"][f][heat["wallCells"]]
            row["h"] = {"mean": float(hm.mean()), "p05": float(np.percentile(hm, 5)),
                        "p95": float(np.percentile(hm, 95))}
        out["fluids"][name] = row
    if heat is not None:
        qa = caps[0] * (t_in[0] - out["fluids"][dom.names[0]]["tOut"])
        qb = caps[1] * (out["fluids"][dom.names[1]]["tOut"] - t_in[1])
        qmax = min(caps) * (t_in[0] - t_in[1])
        tw = heat["Twall"][heat["wallCells"]]
        wc = dom.centers(heat["wallCells"])
        hot = int(np.argmax(tw))
        cold = int(np.argmin(tw))
        out.update(duty=0.5 * (qa + qb), effectiveness=0.5 * (qa + qb) / qmax if qmax else None,
                   energyBalance=(qa - qb) / max(abs(qa), abs(qb), 1e-30),
                   wall={"max": float(tw.max()), "maxAt": np.round(wc[hot], 1).tolist(),
                         "min": float(tw.min()), "minAt": np.round(wc[cold], 1).tolist()})
    return out


# ---------------------------------------------------------------- heat exchangers

def _block_mean(a, r):
    nx, ny, nz = (-(-s // r) for s in a.shape)
    a = np.pad(a.astype(np.float32), [(0, n * r - s) for n, s in zip((nx, ny, nz), a.shape)])
    return a.reshape(nx, r, ny, r, nz, r).mean(axis=(1, 3, 5))


def exchanger_domain(hx, report, grids, voxel=None, open_fraction=0.3, closures="design", local_porosity=True):
    """The coarse porous grid of a designed exchanger (design_heat_exchanger
    with grids=True): core cells from the core's shape, each fluid's
    plenums and ports from its space on the design grid, inlets and outlets
    at the ports' ends."""
    step = float(grids["step"])
    if voxel is None:
        from .heatx import port_diameter
        voxel = min(hx.cell / 2.0, min(port_diameter(hx, 0), port_diameter(hx, 1)) / 4.0, hx.header / 3.0)
    r = max(1, int(round(voxel / step)))
    h_mm = r * step
    fr = [_block_mean(m.reshape(grids["dims"]), r) for m in grids["masks"]]
    dims = fr[0].shape
    lo = np.asarray(grids["lo"], float)
    # the core's share of each cell (4^3 samples), so the core's volume (and so its
    # wetted area) is right however its faces fall on the grid
    sub = (np.indices((4, 4, 4)).reshape(3, -1).T + 0.5) / 4.0
    base = lo + np.indices(dims).reshape(3, -1).T * h_mm
    frac = np.zeros(int(np.prod(dims)))
    for o in sub:
        frac += hx.core.sdf(base + o * h_mm) < 0
    frac = (frac / len(sub)).reshape(dims)
    core = frac >= 0.5
    opn = [(fr[f] >= open_fraction) & ~core for f in (0, 1)]
    both = opn[0] & opn[1]
    opn[0] &= ~(both & (fr[1] > fr[0]))
    opn[1] &= ~(both & (fr[0] >= fr[1]))
    # each fluid's local share of the core, over about a cell: lower where the design
    # seals it off the surface (outside its zones) or closes its channels (closure);
    # measured on cells wholly inside the core, spread to the rest
    inner = (frac > 0.95).astype(np.float32)
    win = max(3, int(round(_PERC["window"] * hx.cell / h_mm)) | 1)
    wsum = ndimage.uniform_filter(inner, win, mode="constant")
    phi_loc = []
    for f in (0, 1):
        pl = ndimage.uniform_filter(fr[f] * inner, win, mode="constant") / np.maximum(wsum, 1e-6)
        deep = ndimage.binary_erosion(core, iterations=win) & (inner > 0)
        ref = float(np.median(pl[deep])) if deep.sum() > 8 else float(np.median(pl[core]))
        phi_loc.append(pl / max(ref, 1e-6))          # relative; scaled by the medium's phi below
    pad = lambda a: np.pad(a, 1)
    # cells in the core carry all of its volume between them
    frac = pad(np.where(core, frac * frac.sum() / max(frac[core].sum(), 1e-9), 0.0))
    core = pad(core)
    opn = [pad(o) for o in opn]
    lo = lo - h_mm
    dims = core.shape
    pts = lo + (np.indices(dims).reshape(3, -1).T + 0.5) * h_mm
    v = report["coreVolume"]
    cells, inlet, outlet, media = [], [], [], []
    for f, name in enumerate(hx.names):
        rep = report[name]
        phi = rep["coreVolume"] / v
        med = _CLOSURES[closures][0](phi, rep["coreWettedArea"] / v * 1e3, rep["hydraulicDiameter"] * 1e-3, hx.cell * 1e-3)
        m = core | opn[f]
        ends = []
        for end in ("in", "out"):
            z = report["zones"][f"{name}-{end}"]
            p1, ax = np.asarray(z["port"][0], float), np.asarray(z["port"][1], float)
            rel = pts - p1
            t = rel @ ax
            rad = np.linalg.norm(rel - np.outer(t, ax), axis=1)
            sel = ((t <= 0) & (t >= -max(0.5 * hx.port_length, 1.5 * h_mm)) &
                   (rad <= 0.5 * z.get("portDiameter", hx.port_diameter) + 0.5 * h_mm)).reshape(dims) & opn[f]
            if not sel.any():
                d = np.linalg.norm(rel, axis=1).reshape(dims)
                d[~opn[f]] = np.inf
                sel = d <= d.min() + 0.5 * h_mm
            ends.append(sel)
        # keep what the inlet reaches
        lbl, _ = ndimage.label(m)
        keep = np.isin(lbl, np.unique(lbl[ends[0]])) & (lbl > 0)
        if not (keep & ends[1]).any():
            raise ValueError(f"fluid {name}'s inlet does not reach its outlet on a {h_mm:.2f} mm grid; "
                             "use a finer voxel")
        phi_loc[f] = pad(phi_loc[f]) * phi
        cells.append(keep)
        inlet.append(ends[0] & keep)
        outlet.append(ends[1] & keep)
        media.append(med)
    phi_sheet = (report[hx.names[0]]["coreVolume"] + report[hx.names[1]]["coreVolume"]) / v
    return PorousDomain(h=h_mm * 1e-3, lo=lo, core=core, cells=cells, inlet=inlet, outlet=outlet, media=media,
                        names=tuple(hx.names), wall=hx.wall * 1e-3, phi_sheet=phi_sheet,
                        extra={"cell": hx.cell, "coreFraction": frac, "phi": phi_loc if local_porosity else None})


def simulate_design(hx, report, grids, duty, voxel=None, closures="design", progress=None, fields=False):
    """Simulate a designed exchanger (design_heat_exchanger(..., grids=True))
    under `duty` (the rating's: {"A": {"fluid", "lPerMin" | "kgPerS",
    "tIn"}, "B": {...}, "wall": "petg" | W/m K})."""
    import time

    t0 = time.time()
    if closures not in _CLOSURES:
        raise ValueError(f"closures is one of {sorted(_CLOSURES)}")
    dom = exchanger_domain(hx, report, grids, voxel, closures=closures)
    kw = duty.get("wall", "petg")
    k_wall = float(WALL_CONDUCTIVITY[kw] if isinstance(kw, str) else kw)
    fluids = [_fluid(duty[n].get("fluid", "water")) for n in hx.names]
    flows_kg = [_flow(duty[n], fl) for n, fl in zip(hx.names, fluids)]
    flows = []
    for f in (0, 1):
        flows.append(solve_flow(dom, f, fluids[f], flows_kg[f],
                                progress=(lambda **k: progress(fluid=hx.names[f], **k)) if progress else None))
    t_in = [duty[n].get("tIn") for n in hx.names]
    heat = None
    if None not in t_in:
        heat = solve_heat(dom, fluids, flows, [float(t) for t in t_in], k_wall, _CLOSURES[closures][1])
    out = summarize(dom, fluids, flows, heat, t_in)
    out["seconds"] = round(time.time() - t0, 1)
    out["closures"] = closures
    if fields:
        out["_fields"] = {"domain": dom, "flows": flows, "heat": heat}
    return out


def simulate_exchanger(hx, duty, voxel=None, closures="design", progress=None, fields=False):
    """Design `hx` and simulate it whole under `duty`; with the epsilon-NTU
    rating alongside for comparison."""
    from .heatx import design_heat_exchanger
    from .heatx_thermal import rate_design

    mesh, report, grids = design_heat_exchanger(hx, grids=True)
    sim = simulate_design(hx, report, grids, duty, voxel, closures, progress, fields)
    try:
        sim["rating"] = rate_design(hx, report, duty)
    except ValueError as exc:
        sim["ratingError"] = str(exc)
    return mesh, report, sim


def sweep(hx, duty, variants, voxel=None, closures="design", progress=None):
    """Compare header and port shapes: each variant is a dict of
    HeatExchangerSpec fields ({"header": 4}, {"port_diameter": 10}, ...)
    applied to a copy of `hx`. Rows: each variant's pressure drops, flow
    uniformity and effectiveness, best effectiveness first."""
    import copy

    rows = []
    for v in variants:
        h2 = copy.deepcopy(hx)
        for k, val in v.items():
            setattr(h2, k, val)
        _, rep, sim = simulate_exchanger(h2, duty, voxel, closures, progress)
        rows.append({"variant": v, "effectiveness": sim.get("effectiveness"),
                     "fluids": {n: {"pressureDrop": r["pressureDrop"], "pressureDropCore": r["pressureDropCore"],
                                    "uniformity": r["entry"].get("uniformity"), "deadShare": r["deadShare"]}
                                for n, r in sim["fluids"].items()},
                     "rating": (sim.get("rating") or {}).get("effectiveness"),
                     "solidVolume": rep["solidVolume"]})
    rows.sort(key=lambda r: -(r["effectiveness"] or 0))
    return rows


def plot_fields(sim_fields, path, axis=2, at=None, title=None):
    """A section through the exchanger (normal `axis`, at `at` mm, default
    the core's middle): each fluid's speed and temperature and the wall's
    temperature, as a PNG."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dom, flows, heat = sim_fields["domain"], sim_fields["flows"], sim_fields["heat"]
    hmm = dom.h * 1e3
    core_idx = np.argwhere(dom.core)
    k = int(round((at - dom.lo[axis]) / hmm - 0.5)) if at is not None else \
        int(round(0.5 * (core_idx[:, axis].min() + core_idx[:, axis].max())))
    sl = [slice(None)] * 3
    sl[axis] = k
    sl = tuple(sl)
    ij = [a for a in range(3) if a != axis]
    ext = [dom.lo[ij[0]], dom.lo[ij[0]] + dom.dims[ij[0]] * hmm, dom.lo[ij[1]], dom.lo[ij[1]] + dom.dims[ij[1]] * hmm]
    panels = []
    for f, n in enumerate(dom.names):
        sp = _cell_speed(flows[f]["U"], dom.cells[f])
        # scaled to the core: the ports run far faster
        top = 2.0 * float(np.percentile(sp[dom.core & dom.cells[f]], 95))
        panels.append((f"{n}: speed (m/s, superficial)", np.where(dom.cells[f], np.minimum(sp, top), np.nan)[sl],
                       "viridis"))
    if heat is not None:
        for f, n in enumerate(dom.names):
            panels.append((f"{n}: temperature (C)", heat["T"][f][sl], "inferno"))
        panels.append(("wall temperature (C)", heat["Twall"][sl], "inferno"))
    fig, axs = plt.subplots(1, len(panels), figsize=(3.6 * len(panels), 3.6), dpi=150)
    for ax_, (name, img, cmap) in zip(np.atleast_1d(axs), panels):
        im = ax_.imshow(img.T, origin="lower", extent=ext, cmap=cmap, interpolation="bilinear")
        ax_.set_title(name, fontsize=8)
        ax_.set_xlabel("xyz"[ij[0]] + " (mm)", fontsize=7)
        ax_.set_ylabel("xyz"[ij[1]] + " (mm)", fontsize=7)
        ax_.tick_params(labelsize=6)
        fig.colorbar(im, ax=ax_, shrink=0.75).ax.tick_params(labelsize=6)
    if title:
        fig.suptitle(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path
