"""
SIMP topology optimization: minimize weighted compliance (maximize
stiffness) over every load case, for a fixed volume of material.

The pieces, each standard and each here for a reason:

- Density per cell x in [0, 1], stiffness E(x) = Emin + x^p (1 - Emin), p = 3,
  so intermediate densities are inefficient and the optimizer avoids them.
- A cone density filter of radius R (mm). It is what gives the result a
  minimum member size and stops checkerboarding; `minMember` sets R.
- A Heaviside projection with continuation (beta 1 -> 32) that pushes the
  filtered field to crisp 0/1, so the part we mesh is the part we analyzed
  rather than a grey cloud thresholded after the fact.
- Optimality-criteria updates with a bisection on the volume multiplier,
  measured on the projected densities.
- Symmetry by averaging sensitivities and densities across mirror cells.

Preserved cells are pinned to 1 and still feed the filter, which is what
makes members grow into bosses instead of stopping a cell short of them.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp

from scipy import ndimage

from .domain import Domain
from .fem import HexMesh, Solver, Stiffness, hex8_ke

__all__ = ["SimpOptions", "SimpResult", "filter_radius", "density_filter", "optimize"]

E_MIN = 1e-6
PENALTY = 3.0
# Cells below this density are left out of the linear solve (see _Analysis).
ALIVE = 1e-3
# ...but only once the design has emptied out enough for it to pay.
REDUCE_BELOW = 0.8
_CONN26 = np.ones((3, 3, 3), bool)


class _Analysis:
    """The FE solves inside the loop, on the cells that still carry material.

    Once a design has settled, most of the domain is void at E_MIN: those
    cells add unknowns, make the system badly conditioned (stiffness ratios
    of a million to one), and contribute nothing. Each iteration we solve on
    the cells above ALIVE, plus anything touching a load or support, keeping
    only the pieces connected to a support (a floating island would make the
    matrix singular; its zero sensitivity lets it fade away). Void cells
    bordering the structure still get sensitivities from the nodes they share
    with it, which is what lets the boundary move. If a load would end up
    disconnected, we fall back to the full system for that iteration."""

    def __init__(self, domain: Domain, nu: float, rtol: float):
        self.domain = domain
        self.nu = nu
        self.rtol = rtol
        mesh = domain.mesh
        self.ke = hex8_ke(nu) * mesh.pitch
        self.support_touch = domain.support_nodes[mesh.enode].any(axis=1)
        self.anchor = self.support_touch | domain.load_nodes[mesh.enode].any(axis=1) | domain.preserve
        self._full = None
        self.last_iters = 0
        self.last_cells = mesh.n_elem

    def _full_system(self):
        if self._full is None:
            mesh = self.domain.mesh
            self._full = (Stiffness(mesh, self.nu, self.domain.fixed), Solver(mesh, rtol=self.rtol))
        return self._full

    def _connected(self, alive):
        mesh = self.domain.mesh
        g = np.zeros(self.domain.dims, bool)
        ijk = mesh.elem_ijk
        g[tuple(ijk[alive].T)] = True
        lbl, _ = ndimage.label(g, structure=_CONN26)
        el = lbl[tuple(ijk.T)]
        keep_l = np.unique(el[alive & self.support_touch])
        return alive & np.isin(el, keep_l[keep_l > 0])

    def solve(self, e, xp, us, rtol=None):
        """Solve every load case for moduli e. Returns per-case displacement on
        the full node set."""
        d = self.domain
        mesh = d.mesh
        rtol = rtol or self.rtol
        alive = (xp > ALIVE) | self.anchor
        if alive.mean() < REDUCE_BELOW:
            keep = self._connected(alive)
            sub = HexMesh(mesh.elem_ijk[keep], mesh.pitch, mesh.origin)
            full_node = mesh.node_id(sub.node_ijk)
            fd = (3 * full_node[:, None] + np.arange(3)).ravel()
            in_sub = np.zeros(mesh.n_dof, bool)
            in_sub[fd] = True
            if all(not np.any(f[~in_sub]) for f in d.loads):
                stiff = Stiffness(sub, self.nu, d.fixed[fd])
                solver = Solver(sub, rtol=rtol)
                k = stiff.assemble(e[keep])
                subs = solver.solve_many(k, [f[fd] for f in d.loads],
                                         [u[fd] if u is not None else None for u in us])
                out = []
                for us_sub in subs:
                    u = np.zeros(mesh.n_dof)
                    u[fd] = us_sub
                    out.append(u)
                self.last_iters, self.last_cells = solver.last_iters, int(keep.sum())
                return out
        stiff, solver = self._full_system()
        solver.rtol = rtol
        k = stiff.assemble(e)
        out = solver.solve_many(k, list(d.loads), list(us))
        self.last_iters, self.last_cells = solver.last_iters, mesh.n_elem
        return out

    def strain_energy(self, u):
        ue = u[self.domain.mesh.edof]
        return np.einsum("ij,jk,ik->i", ue, self.ke, ue)


@dataclass
class SimpOptions:
    volume_fraction: float
    filter_radius: float              # mm
    max_iterations: int = 120
    beta_max: float = 32.0
    beta_every: int = 15              # iterations per beta doubling at most
    move: float = 0.2
    tol: float = 0.01
    # Sensitivities only need a loose solve; the final analysis is tight.
    solve_rtol: float = 1e-3
    # Stop once beta is at max and compliance has moved less than this
    # (relative) over the last `stall_window` iterations. Projected designs
    # can chatter at the move limit forever on a handful of boundary cells
    # while the compliance is flat, so `change` alone never settles.
    stall_rtol: float = 2e-3
    stall_window: int = 6
    eta: float = 0.5
    # Stiffness is credited to the ERODED design (projected at eta + this),
    # while volume is counted on the blueprint (eta) we deliver. A member
    # that vanishes under erosion earns nothing, so the optimizer stops
    # building sub-minimum sheets: the eroded-design half of the robust
    # formulation (Wang, Lazarov & Sigmund 2011), at no extra solve per
    # iteration. Round 2 of the evals still grew 0.8mm skins without it.
    erode: float = 0.15
    x0: np.ndarray | None = None      # warm start (design densities)
    # (axis, sign) layers stack along, or None: see am.py. Every density the
    # optimizer sees is then the printable one.
    build_direction: tuple[int, int] | None = None
    # A warm start is already a design; restarting projection at beta 1
    # would blur it back to grey before sharpening it again.
    beta_start: float = 1.0


@dataclass
class SimpResult:
    x: np.ndarray                     # design variables
    xphys: np.ndarray                 # projected densities, what we mesh
    compliance: float
    case_compliance: list
    volume_fraction: float
    iterations: int
    converged: bool
    seconds: float
    history: list = field(default_factory=list)


def filter_radius(domain: Domain, min_member: float | None) -> float:
    """Filter radius in mm, never below 1.5 cells (or the checkerboard comes
    back).

    With minMember t, the radius is chosen so a sheet thinner than t does not
    survive erosion: for this cone filter a slab of thickness 0.72 R reaches
    a filtered density of 0.65 (the eroded threshold, eta + erode), so
    R = 1.3 t puts the cut at 0.94 t. The earlier R = t/2 + pitch/2 let
    2-cell sheets through and they meshed thinner still (eval round 4:
    a 0.9mm web in the bell crank against a 2.5mm minimum)."""
    r = 1.5 * domain.pitch
    if min_member:
        r = max(r, 1.3 * float(min_member))
    return r


def density_filter(domain: Domain, radius: float):
    """Row-normalized cone filter over active cells as a sparse matrix."""
    mesh = domain.mesh
    n = mesh.n_elem
    reach = int(np.floor(radius / domain.pitch))
    lut = np.full(domain.dims, -1, np.int64)
    lut[tuple(mesh.elem_ijk.T)] = np.arange(n)
    rows, cols, vals = [], [], []
    dims = np.asarray(domain.dims)
    for dx in range(-reach, reach + 1):
        for dy in range(-reach, reach + 1):
            for dz in range(-reach, reach + 1):
                dist = np.sqrt(dx * dx + dy * dy + dz * dz) * domain.pitch
                w = radius - dist
                if w <= 0:
                    continue
                nb = mesh.elem_ijk + np.array([dx, dy, dz])
                ok = np.all((nb >= 0) & (nb < dims), axis=1)
                j = np.full(n, -1, np.int64)
                j[ok] = lut[tuple(nb[ok].T)]
                hit = j >= 0
                rows.append(np.nonzero(hit)[0])
                cols.append(j[hit])
                vals.append(np.full(int(hit.sum()), w))
    h = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n, n))
    # Normalize by the FULL kernel, not just the active neighbors: cells
    # outside the design space (and inside keep-outs) count as void. With the
    # usual truncated normalization a member lying along the domain boundary
    # is filtered as if it were twice as thick, so the minimum member size
    # does not hold there: round 1 of the evals grew 0.8mm skins on the faces
    # of the arm link's design box with minMember = 3mm.
    full = sum(max(radius - np.sqrt(dx * dx + dy * dy + dz * dz) * domain.pitch, 0.0)
               for dx in range(-reach, reach + 1) for dy in range(-reach, reach + 1)
               for dz in range(-reach, reach + 1))
    hs = np.full(n, full)
    return h, hs


def _project(xt, beta, eta):
    den = np.tanh(beta * eta) + np.tanh(beta * (1 - eta))
    return (np.tanh(beta * eta) + np.tanh(beta * (xt - eta))) / den


def _dproject(xt, beta, eta):
    den = np.tanh(beta * eta) + np.tanh(beta * (1 - eta))
    return beta * (1 - np.tanh(beta * (xt - eta)) ** 2) / den


def _symmetrize(v, mirror):
    for m in mirror.values():
        has = m >= 0
        out = v.copy()
        out[has] = 0.5 * (v[has] + v[m[has]])
        v = out
    return v


def optimize(domain: Domain, nu: float, opts: SimpOptions, on_iter=None, should_stop=None) -> SimpResult:
    t0 = time.time()
    mesh = domain.mesh
    n = mesh.n_elem
    pres = domain.preserve
    design = ~pres
    n_design = int(design.sum())
    if n_design == 0:
        raise ValueError("nothing to optimize: every cell is preserved")
    fe = _Analysis(domain, nu, opts.solve_rtol)
    h, hs = density_filter(domain, opts.filter_radius)
    ht = h.T.tocsr()

    vf = float(opts.volume_fraction)
    x = np.full(n, vf)
    if opts.x0 is not None:
        x = np.clip(np.asarray(opts.x0, float), 0.0, 1.0)
    x[pres] = 1.0
    target = vf * n_design

    beta = float(opts.beta_start)
    since_beta = 0
    us = [None] * len(domain.loads)
    weights = np.asarray(domain.case_weights, float)
    history = []
    c0 = None
    converged = False
    it = 0
    change = 1.0

    ijk = tuple(mesh.elem_ijk.T)
    if opts.build_direction is not None:
        from .am import AMFilter

        pinned = domain.full_grid(pres.astype(float)) > 0.5
        am_b = AMFilter(domain.dims, opts.build_direction, pinned)   # blueprint
        am_e = AMFilter(domain.dims, opts.build_direction, pinned)   # eroded
        am_v = AMFilter(domain.dims, opts.build_direction, pinned)   # volume search

        def printed(am, v):
            return am.forward(domain.full_grid(v))[ijk]

        def printed_grad(am, g):
            return am.backward(domain.full_grid(g))[ijk]
    else:
        am_b = am_e = am_v = None

        def printed(am, v):
            return v

        def printed_grad(am, g):
            return g

    def physical(xv, b, am=None):
        xt = (h @ xv) / hs
        xp = _project(xt, b, opts.eta)
        xp[pres] = 1.0
        return xt, printed(am if am is not None else am_v, xp)

    eta_e = opts.eta + opts.erode

    def eroded(xt, b):
        xe = _project(xt, b, eta_e)
        xe[pres] = 1.0
        return printed(am_e, xe)

    for it in range(1, opts.max_iterations + 1):
        if should_stop is not None and should_stop():
            break
        xt, xp = physical(x, beta, am_b)
        xe = eroded(xt, beta)
        e = E_MIN + xe ** PENALTY * (1 - E_MIN)
        us = fe.solve(e, xe, us)
        dc_xp = np.zeros(n)
        comp = 0.0
        case_c = []
        for ci, f in enumerate(domain.loads):
            ce = fe.strain_energy(us[ci])
            cc = float(f @ us[ci])
            case_c.append(cc)
            comp += weights[ci] * cc
            dc_xp -= weights[ci] * PENALTY * xe ** (PENALTY - 1) * (1 - E_MIN) * ce
        if c0 is None:
            c0 = comp
        dproj_e = _dproject(xt, beta, eta_e)
        dproj_e[pres] = 0.0
        dproj = _dproject(xt, beta, opts.eta)
        dproj[pres] = 0.0
        dc = ht @ (printed_grad(am_e, dc_xp) * dproj_e / hs)
        dv = ht @ (printed_grad(am_b, design.astype(float)) * dproj / hs)
        dc = np.minimum(dc, 0.0)
        dv = np.maximum(dv, 1e-12)
        if domain.mirror:
            dc = _symmetrize(dc, domain.mirror)
            dv = _symmetrize(dv, domain.mirror)

        move = opts.move if beta <= 4 else opts.move / 2
        lo_x = np.maximum(0.0, x - move)
        hi_x = np.minimum(1.0, x + move)
        ratio = np.sqrt(-dc / dv)
        l1, l2 = 1e-12, 1e12
        x_new = x
        for _ in range(80):
            lm = np.sqrt(l1 * l2)
            x_new = np.clip(x * ratio / np.sqrt(lm), lo_x, hi_x)
            x_new[pres] = 1.0
            if domain.mirror:
                x_new = _symmetrize(x_new, domain.mirror)
            vol = float(physical(x_new, beta)[1][design].sum())
            if vol > target:
                l1 = lm
            else:
                l2 = lm
            if l2 / l1 < 1 + 1e-4:
                break
        change = float(np.abs(x_new - x)[design].max())
        x = x_new
        since_beta += 1

        rec = {"iteration": it, "compliance": comp, "relative": comp / c0 if c0 else 1.0,
               "volume": float(xp[design].mean()), "beta": beta, "change": change,
               "seconds": time.time() - t0, "cgIterations": fe.last_iters,
               "solvedCells": fe.last_cells}
        history.append(rec)
        if on_iter is not None:
            on_iter(rec, xp)

        if beta < opts.beta_max and (change < opts.tol * 2 or since_beta >= opts.beta_every):
            beta = min(beta * 2, opts.beta_max)
            since_beta = 0
        elif beta >= opts.beta_max and it >= 3:
            if change < opts.tol:
                converged = True
                break
            w = opts.stall_window
            if since_beta >= w and len(history) > w:
                old = history[-1 - w]["compliance"]
                if abs(comp - old) <= opts.stall_rtol * abs(old):
                    converged = True
                    break

    xt, xp = physical(x, beta)
    if opts.erode > 0:
        # Deliver the OPENING of the blueprint: erode, then dilate back by the
        # same amount with the same filter. Everything the optimizer was
        # credited for survives at full size; a sheet thinner than the
        # erosion (the membrane round 3 of the evals left inside the bell
        # crank's frame) does not come back.
        xe = eroded(xt, beta)
        xp = _project((h @ xe) / hs, beta, opts.eta - opts.erode)
        xp[pres] = 1.0
        xp = printed(am_v, xp)
    # Final analysis on the delivered field (it moved since the last solve).
    e = E_MIN + xp ** PENALTY * (1 - E_MIN)
    us = fe.solve(e, xp, us, rtol=1e-6)
    case_c, comp = [], 0.0
    for ci, f in enumerate(domain.loads):
        cc = float(f @ us[ci])
        case_c.append(cc)
        comp += weights[ci] * cc
    return SimpResult(x=x, xphys=xp, compliance=comp, case_compliance=case_c,
                      volume_fraction=float(xp[design].mean()), iterations=it,
                      converged=converged, seconds=time.time() - t0, history=history)
