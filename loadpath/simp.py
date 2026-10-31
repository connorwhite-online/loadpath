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

from .directions import collapse, expand
from .domain import Domain
from .fem import HexMesh, Solver, Stiffness, hex8_ke
from .manufacture import AccessFilter, MaxMember, machinable, tie_groups

import os

__all__ = ["SimpOptions", "SimpResult", "filter_radius", "density_filter", "optimize"]

E_MIN = 1e-6
PENALTY = 3.0
# Cells below this density are left out of the linear solve (see _Analysis).
ALIVE = 1e-3
# ...but only once the design has emptied out enough for it to pay.
REDUCE_BELOW = 0.8
_CONN26 = np.ones((3, 3, 3), bool)
# From this many cells up the analysis is matrix-free geometric multigrid
# (mg.GridElasticity) and the density filter a grid stencil (GridFilter):
# the assembled stiffness and sparse filter cost ~3 KB per cell and pyamg's
# setup grows faster than the grid. Below it the assembled path stays, so
# every eval case (all at 80k cells or fewer) runs exactly as before.
# LOADPATH_SOLVER=mg|assembled forces one or the other.
MG_MIN_CELLS = 100_000


def _use_mg(n_cells: int) -> bool:
    force = os.environ.get("LOADPATH_SOLVER", "").lower()
    if force in ("mg", "multigrid"):
        return True
    if force in ("assembled", "amg"):
        return False
    return n_cells >= MG_MIN_CELLS


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
        self._sub = None
        self.last_iters = 0
        self.last_cells = mesh.n_elem
        self._mg = None
        if _use_mg(mesh.n_elem):
            from .mg import GridElasticity

            self._mg = GridElasticity(mesh, nu, domain.fixed, springs=domain.springs, rtol=rtol, maxiter=2000)

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

    # Reuse one reduced system (its CSR layout and its multigrid hierarchy)
    # while the live cells stay inside it, for at most REUSE iterations.
    # Measured on the arm link: rebuilding is not the bottleneck (the CG
    # iterations are), and dilating the live set so it can be reused more
    # often made each iteration slower (bigger system), so DILATE is off.
    REUSE = 6
    DILATE = False

    def _subsystem(self, alive, rtol):
        d = self.domain
        mesh = d.mesh
        c = self._sub
        if c is not None and c["age"] < self.REUSE and not np.any(alive & ~c["keep"]):
            c["age"] += 1
            return c["keep"], c["fd"], c["stiff"], c["solver"]
        g = np.zeros(d.dims, bool)
        ijk = tuple(mesh.elem_ijk.T)
        g[ijk] = alive
        grown = ndimage.binary_dilation(g, structure=_CONN26)[ijk] if self.DILATE else alive
        keep = self._connected(grown)
        sub = HexMesh(mesh.elem_ijk[keep], mesh.pitch, mesh.origin)
        full_node = mesh.node_id(sub.node_ijk)
        fd = (3 * full_node[:, None] + np.arange(3)).ravel()
        in_sub = np.zeros(mesh.n_dof, bool)
        in_sub[fd] = True
        if any(np.any(f[~in_sub]) for f in d.loads):
            self._sub = None
            return None
        # (Self-weight on cells left out is negligible: they are below ALIVE
        # density, so it is dropped with them.)
        self._sub = {"keep": keep, "fd": fd, "stiff": Stiffness(sub, self.nu, d.fixed[fd]),
                     "solver": Solver(sub, rtol=rtol), "age": 0}
        return keep, fd, self._sub["stiff"], self._sub["solver"]

    def solve(self, e, xp, us, rtol=None, loads=None):
        """Solve every load case for moduli e. Returns per-case displacement on
        the full node set. `loads` overrides the domain's (self-weight makes
        them depend on the design)."""
        d = self.domain
        loads = loads if loads is not None else d.loads
        mesh = d.mesh
        rtol = rtol or self.rtol
        alive = (xp > ALIVE) | self.anchor
        if self._mg is not None:
            return self._solve_mg(e, alive, us, rtol, loads)
        self._last_e = e
        if alive.mean() < REDUCE_BELOW:
            sub = self._subsystem(alive, rtol)
            subs = None
            if sub is not None:
                keep, fd, stiff, solver = sub
                solver.rtol = rtol
                k = stiff.assemble(e[keep])
                if d.springs is not None:
                    k = (k + d.springs[fd][:, fd]).tocsr()
                try:
                    subs = solver.solve_many(k, [f[fd] for f in loads],
                                             [u[fd] if u is not None else None for u in us])
                except RuntimeError:
                    # The live cells are joined to the rest by an edge or a
                    # corner somewhere (26-connected but a hinge): singular.
                    # The full system, void at E_MIN, holds the hinge.
                    self._sub = None
            if subs is not None:
                out = []
                for us_sub in subs:
                    u = np.zeros(mesh.n_dof)
                    u[fd] = us_sub
                    out.append(u)
                self.last_iters, self.last_cells = solver.last_iters, int(keep.sum())
                self._last = (k, fd, solver)
                return out
        stiff, solver = self._full_system()
        solver.rtol = rtol
        k = stiff.assemble(e)
        if d.springs is not None:
            k = (k + d.springs).tocsr()
        out = solver.solve_many(k, list(loads), list(us))
        self.last_iters, self.last_cells = solver.last_iters, mesh.n_elem
        self._last = (k, None, solver)
        return out

    def _solve_mg(self, e, alive, us, rtol, loads):
        """The multigrid path: the same reduction (cells below ALIVE and
        islands left out), done by zeroing their moduli on the fixed grid
        rather than building a smaller mesh."""
        mg = self._mg
        mg.rtol = rtol
        keep = self._connected(alive) if alive.mean() < REDUCE_BELOW else np.ones(len(e), bool)
        mg.set_moduli(np.where(keep, e, 0.0))
        if not keep.all() and any(np.any(f[~mg.live_dofs()]) for f in loads):
            keep = np.ones(len(e), bool)
            mg.set_moduli(e)
        out = mg.solve_many(list(loads), list(us))
        self.last_iters, self.last_cells = mg.last_iters, int(keep.sum())
        self._last = None
        return out

    def resolve(self, rhss, x0s=None):
        """Solve more right-hand sides (adjoints) against the matrix of the
        last solve(); full-size in, full-size out."""
        if self._mg is not None:
            return self._mg.solve_many(list(rhss), x0s)
        k, fd, solver = self._last
        n_dof = self.domain.mesh.n_dof
        if fd is None:
            return solver.solve_many(k, list(rhss), x0s)
        try:
            sub = solver.solve_many(k, [r[fd] for r in rhss],
                                    [x[fd] for x in x0s] if x0s else None)
        except RuntimeError:
            # a hinge the loads did not excite but an adjoint does (see
            # solve): solve on the full system instead
            self._sub = None
            stiff, solver = self._full_system()
            k = stiff.assemble(self._last_e)
            if self.domain.springs is not None:
                k = (k + self.domain.springs).tocsr()
            self._last = (k, None, solver)
            return solver.solve_many(k, list(rhss), x0s)
        out = []
        for v in sub:
            u = np.zeros(n_dof)
            u[fd] = v
            out.append(u)
        return out

    def strain_energy(self, u):
        if self._mg is not None:
            return self._mg.strain_energy(u)
        ue = u[self.domain.mesh.edof]
        return np.einsum("ij,jk,ik->i", ue, self.ke, ue)


@dataclass
class SimpOptions:
    volume_fraction: float
    filter_radius: float              # mm
    max_iterations: int = 120
    beta_max: float = 32.0
    beta_every: int = 10              # iterations per beta doubling at most (15 cost 20% more iterations for the same design on the arm link)
    # Iterations per doubling by projection level (beta 1->2, 2->4, ...),
    # then beta_every past its end; None uses beta_every throughout. (20, 20)
    # is stiffer on every benchmark (docs/BENCHMARKS.md: MBB -2.5%, bridge
    # -11%, 1.4x the iterations) but not lighter at a safety factor: on the
    # evals it was 2-14% heavier at the target SF on the small parts, so it
    # is not the default.
    beta_schedule: tuple | None = None
    move: float = 0.2
    tol: float = 0.01
    # Sensitivities only need a loose solve; the final analysis is tight.
    solve_rtol: float = 1e-3
    adaptive_rtol: bool = True
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
    # The other half, for holes: the delivered part is CLOSED (dilated at
    # eta - this, eroded back) with the same filter, so a hole or gap
    # narrower than ~0.94 R comes out filled. 0 = off (holes of any size);
    # minHole turns it on (see filter_radius). Counting the volume on the
    # dilated design inside the loop instead (the textbook robust
    # formulation, target rescaled to the blueprint) left a sealed-tube test
    # with a quarter of its blueprint in members thinner than the erosion:
    # 0.185 delivered for 0.25 asked and 2.7x the compliance (2026-10-30).
    # Closing at delivery adds a little mass the verification then weighs.
    dilate: float = 0.0
    # bool per element pinned void (escape holes drilled into a sealed void),
    # or None; see escape.py
    void: np.ndarray | None = None
    # escape.EscapeOptions: drill escape holes into the voids the design
    # seals, then re-converge round them; or None
    escape: object = None
    x0: np.ndarray | None = None      # warm start (design densities)
    # (axis, sign) layers stack along, or None: see am.py. Every density the
    # optimizer sees is then the printable one.
    build_direction: tuple[int, int] | None = None
    # Axis index the part is extruded along (2.5D: cut from plate), or None.
    extrusion: int | None = None
    # Milling / casting access (manufacture.Access), a circular pattern
    # (manufacture.Pattern), and the thickest member (mm): see manufacture.py.
    access: object = None
    pattern: object = None
    max_member: float | None = None
    # A warm start is already a design; restarting projection at beta 1
    # would blur it back to grey before sharpening it again.
    beta_start: float = 1.0
    # Remove delivered material that carries no load in any case (see
    # _prune_dead).
    prune: bool = True
    # Normalize the filter by the full kernel (outside the design space reads
    # as void, see density_filter), or None for the default: padded unless
    # the part is extruded. The benchmarks turn it off to match top88/top3d.
    pad_filter: bool | None = None


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
    # per case: the worst direction of a force of unknown direction
    # (directions.py), or None
    case_direction: list = field(default_factory=list)
    # per element: strain energy density of the delivered design (weighted
    # over cases), for siting escape holes where the part is calm
    energy: np.ndarray | None = None
    # escape holes drilled (escape.drill's report), or None
    escape: dict | None = None


def filter_radius(domain: Domain, min_member: float | None, min_hole: float | None = None) -> float:
    """Filter radius in mm, never below 1.5 cells (or the checkerboard comes
    back).

    With minMember t, the radius is chosen so a sheet thinner than t does not
    survive erosion: for this cone filter a slab of thickness 0.72 R reaches
    a filtered density of 0.65 (the eroded threshold, eta + erode), so
    R = 1.3 t puts the cut at 0.94 t. The earlier R = t/2 + pitch/2 let
    2-cell sheets through and they meshed thinner still (eval round 4:
    a 0.9mm web in the bell crank against a 2.5mm minimum).

    The dilated design (eta - dilate, minHole) mirrors it: a gap narrower
    than 0.94 R closes, so R = 1.3 x the larger of the two."""
    r = 1.5 * domain.pitch
    if min_member:
        r = max(r, 1.3 * float(min_member))
    if min_hole:
        r = max(r, 1.3 * float(min_hole))
    return r


class GridFilter:
    """The cone filter's matrix H (unnormalized, symmetric) as a stencil on
    the domain grid: `h @ x` gives what the sparse matrix would, without
    storing ~100 weights per cell. Used from MG_MIN_CELLS up."""

    def __init__(self, domain: Domain, radius: float):
        self.ijk = tuple(domain.mesh.elem_ijk.T)
        self.dims = tuple(domain.dims)
        self.n = domain.mesh.n_elem
        self.shape = (self.n, self.n)
        reach = int(np.floor(radius / domain.pitch))
        self.reach = reach
        self.taps = []
        for dx in range(-reach, reach + 1):
            for dy in range(-reach, reach + 1):
                for dz in range(-reach, reach + 1):
                    w = radius - np.sqrt(dx * dx + dy * dy + dz * dz) * domain.pitch
                    if w > 0:
                        self.taps.append(((dx, dy, dz), float(w)))
        self.T = self

    def __matmul__(self, v):
        r = self.reach
        g = np.zeros(tuple(d + 2 * r for d in self.dims))
        nx, ny, nz = self.dims
        g[r:r + nx, r:r + ny, r:r + nz][self.ijk] = v
        out = np.zeros(self.dims)
        for (dx, dy, dz), w in self.taps:
            out += w * g[r + dx:r + dx + nx, r + dy:r + dy + ny, r + dz:r + dz + nz]
        return out[self.ijk]

    def tocsr(self):
        return self

    def sum(self, axis=1):
        return self @ np.ones(self.n)


def density_filter(domain: Domain, radius: float, pad: bool = True):
    """Row-normalized cone filter over active cells as a sparse matrix
    (a GridFilter stencil from MG_MIN_CELLS cells up)."""
    mesh = domain.mesh
    if _use_mg(mesh.n_elem):
        h = GridFilter(domain, radius)
        full = sum(w for _, w in h.taps)
        hs = np.full(mesh.n_elem, full) if pad else h.sum()
        return h, hs
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
    hs = np.full(n, full) if pad else np.asarray(h.sum(axis=1)).ravel()
    return h, hs


# A cell is dead when its strain energy density, in every load case, is
# below this fraction of that case's median over the solid. A member that
# ends in free space carries no stress at all (nothing pulls on its tip), so
# it reads many orders below this; a member that matters to even the
# lightest case reads far above it, because each case is judged on its own
# scale.
PRUNE_BELOW = 1e-3
PRUNE_PASSES = 3


def _prune_dead(fe, domain, xp, us, loads, design, pres):
    """Delete dead-end material from the delivered design.

    Projection continuation freezes the layout once beta is high: a member
    the optimizer stopped needing can no longer shrink away (its gradient
    lives only in the thin transition band), and compliance does not notice
    a strut that carries nothing. GRIP's wrist link came out with several
    such tendrils hanging in free space. They are pure deadweight and look
    unfinished, so after the final solve, cells that carry essentially no
    energy in any case are removed, the design re-solved, and repeated,
    because removing a tendril's tip leaves the next stretch of it dead too.
    Returns (xp, us)."""
    for _ in range(PRUNE_PASSES):
        solid = design & (xp > 0.5) & ~pres
        if not solid.any():
            break
        live = np.zeros(len(xp), bool)
        for u in us:
            w = fe.strain_energy(u) * (xp ** PENALTY)
            ref = np.median(w[xp > 0.5])
            if ref <= 0:
                live[:] = True
                break
            live |= w >= PRUNE_BELOW * ref
        dead = solid & ~live
        # skip specks: a lone low-energy cell inside a member is a numerical
        # neutral axis, not a dead end
        if dead.sum() < 0.002 * solid.sum():
            break
        g = np.zeros(domain.dims, bool)
        ijk = tuple(domain.mesh.elem_ijk.T)
        g[ijk] = dead
        lbl, n = ndimage.label(g, structure=_CONN26)
        if n == 0:
            break
        sizes = np.bincount(lbl[ijk][dead], minlength=n + 1)
        el = lbl[ijk]
        dead &= sizes[el] >= 4
        if not dead.any():
            break
        xp = xp.copy()
        xp[dead] = 0.0
        # and whatever that cut loose from the supports (a tip whose root
        # was a dead cluster under the size floor)
        loose = (xp > 0.5) & ~fe._connected((xp > 0.5) | fe.anchor) & ~pres
        xp[loose] = 0.0
        e = E_MIN + xp ** PENALTY * (1 - E_MIN)
        us = fe.solve(e, xp, us, rtol=1e-6, loads=loads)
    return xp, us


def _pad(opts) -> bool:
    return opts.extrusion is None if opts.pad_filter is None else bool(opts.pad_filter)


def _beta_hold(opts, beta):
    """Most iterations to spend at projection sharpness beta."""
    s = opts.beta_schedule
    k = int(round(np.log2(max(beta, 1.0))))
    return s[k] if s is not None and k < len(s) else opts.beta_every


def _project(xt, beta, eta):
    den = np.tanh(beta * eta) + np.tanh(beta * (1 - eta))
    return (np.tanh(beta * eta) + np.tanh(beta * (xt - eta))) / den


def _dproject(xt, beta, eta):
    den = np.tanh(beta * eta) + np.tanh(beta * (1 - eta))
    return beta * (1 - np.tanh(beta * (xt - eta)) ** 2) / den


def _extrude(v, columns, design):
    """Average over each extrusion column's design cells, so every column is
    one value along the extrusion axis (a 2.5D part, cut from plate)."""
    if columns is None:
        return v
    col = columns[design]
    n = col.max() + 1
    mean = np.bincount(col, weights=v[design], minlength=n) / np.maximum(np.bincount(col, minlength=n), 1)
    out = v.copy()
    out[design] = mean[col]
    return out


def _machined(domain, xp, access):
    """Fill every design cell a cutter of the access's radius (and 2.5-axis
    floors) cannot reach: the exact delivery step (manufacture.machinable)."""
    ijk = tuple(domain.mesh.elem_ijk.T)
    blocked = machinable(domain.full_grid(xp) > 0.5, access, domain.pitch)[ijk]
    xp = xp.copy()
    xp[blocked] = np.maximum(xp[blocked], 1.0)
    return xp


def _symmetrize(v, mirror):
    for m in mirror.values():
        has = m >= 0
        out = v.copy()
        out[has] = 0.5 * (v[has] + v[m[has]])
        v = out
    return v


def optimize(domain: Domain, nu: float, opts: SimpOptions, on_iter=None, should_stop=None) -> SimpResult:
    """Optimize, then (opts.escape) open every void the design sealed: drill
    escape holes into it (escape.drill), pin them void, and run on from the
    design for a few more iterations so the part carries its load round
    them. Up to three rounds, as the re-converged design can seal a new one."""
    if opts.escape is None or opts.extrusion is not None or opts.access is not None:
        # a profile cut from plate or a milled or cast part has no sealed
        # voids to drain (machining fills them)
        return _optimize(domain, nu, opts, on_iter, should_stop)
    from dataclasses import replace

    from .escape import drill

    t0 = time.time()
    res = _optimize(domain, nu, replace(opts, escape=None), on_iter, should_stop)
    void = None if opts.void is None else opts.void.copy()
    log = {"holes": [], "filled": 0.0, "rounds": 0, "left": None}
    for _ in range(3):
        rep = drill(domain, res.xphys, res.energy, opts.escape, void)
        log.setdefault("sealed", rep["sealed"])      # what the optimizer sealed
        log["left"] = rep["sealed"] if not rep["holes"] and not rep["fill"].any() else None
        log["unopened"] = rep.get("unopened", 0)
        if rep.get("maxPath") is not None:
            log["maxPath"] = rep["maxPath"]
        if not rep["holes"] and not rep["fill"].any():
            break
        log["rounds"] += 1
        log["holes"] += rep["holes"]
        log["filled"] += rep["filledVolume"]
        void = rep["void"] if void is None else (void | rep["void"])
        x0 = res.x.copy()
        x0[rep["fill"]] = 1.0
        x0[void] = 0.0
        if should_stop is not None and should_stop():
            break
        hist = res.history
        res = _optimize(domain, nu, replace(opts, escape=None, void=void, x0=x0,
                                            beta_start=float(opts.beta_max),
                                            max_iterations=opts.escape.iterations), on_iter, should_stop)
        res.history = hist + res.history
    if log["left"] is None:
        # the last round drilled: check what it delivered
        log["left"] = drill(domain, res.xphys, None, opts.escape, void)["sealed"]
    res.escape = log
    res.seconds = time.time() - t0
    return res


def _optimize(domain: Domain, nu: float, opts: SimpOptions, on_iter=None, should_stop=None) -> SimpResult:
    t0 = time.time()
    mesh = domain.mesh
    n = mesh.n_elem
    pres = domain.preserve
    design = ~pres
    n_design = int(design.sum())
    if n_design == 0:
        raise ValueError("nothing to optimize: every cell is preserved")
    fe = _Analysis(domain, nu, opts.solve_rtol)
    columns = tie_groups(domain, opts.extrusion, opts.pattern)
    # An extruded part's top and bottom are the stock's faces: padding the
    # filter there (outside = void) would thin them, so pad only free designs.
    h, hs = density_filter(domain, opts.filter_radius, pad=_pad(opts))
    ht = h.T.tocsr()

    vf = float(opts.volume_fraction)
    x = np.full(n, vf)
    if opts.x0 is not None:
        x = np.clip(np.asarray(opts.x0, float), 0.0, 1.0)
    x[pres] = 1.0
    void = opts.void if opts.void is not None and opts.void.any() else None
    if void is not None:
        x[void] = 0.0
    target = vf * n_design

    beta = float(opts.beta_start)
    since_beta = 0
    spreads = domain.spreads or [None] * len(domain.loads)
    us = [None] * len(expand(domain.loads, spreads)[0])
    robust = any(s is not None for s in spreads)
    dc_prev = None
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
    elif opts.access is not None:
        # milling / casting: material that would leave an undercut is
        # credited as filled (manufacture.AccessFilter)
        dirs = opts.access.directions
        am_b, am_e = AccessFilter(domain.dims, dirs), AccessFilter(domain.dims, dirs)
        am_v = AccessFilter(domain.dims, dirs, track=False)

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

    eta_d = opts.eta - opts.dilate

    def physical(xv, b, am=None, eta=opts.eta):
        xt = (h @ xv) / hs
        xp = _project(xt, b, eta)
        xp[pres] = 1.0
        if void is not None:
            xp[void] = 0.0
        return xt, printed(am if am is not None else am_v, xp)

    eta_e = opts.eta + opts.erode
    thick = MaxMember(domain, opts.max_member, design) if opts.max_member else None
    mu = 0.0

    def eroded(xt, b):
        xe = _project(xt, b, eta_e)
        xe[pres] = 1.0
        if void is not None:
            xe[void] = 0.0
        return printed(am_e, xe)

    for it in range(1, opts.max_iterations + 1):
        if should_stop is not None and should_stop():
            break
        xt, xp = physical(x, beta, am_b)
        xe = eroded(xt, beta)
        e = E_MIN + xe ** PENALTY * (1 - E_MIN)
        fg = None
        if domain.gravity is not None:
            from .domain import gravity_force
            fg = gravity_force(mesh, xe, domain.gravity, domain.fixed)
        # A force of unknown direction is three solves, one per axis, and
        # each case is judged in its worst direction (directions.py).
        rhs, layout = expand(domain.loads, spreads, fg)
        # Sensitivities only need to point the right way while the design is
        # still soft; tighten the solve as the projection sharpens.
        rtol = opts.solve_rtol * (10.0 if beta <= 2 else 3.0 if beta <= 8 else 1.0) if opts.adaptive_rtol \
            else opts.solve_rtol
        us = fe.solve(e, xe, us, rtol=rtol, loads=rhs)
        dc_xp = np.zeros(n)
        comp = 0.0
        case_c = []
        for ci, wc in enumerate(collapse(rhs, layout, us, spreads)):
            # the worst direction's sensitivity is its own (envelope theorem),
            # softened over near-ties (directions.SMOOTH)
            cc = wc.compliance
            ce = sum(m * fe.strain_energy(v) for m, v in wc.energy)
            case_c.append(cc)
            comp += weights[ci] * cc
            dc_xp -= weights[ci] * PENALTY * xe ** (PENALTY - 1) * (1 - E_MIN) * ce
            if domain.gravity is not None:
                # d(f.u)/dx with design-dependent weight: + 2 u . df/dx.
                ue = wc.u_mean.reshape(-1, 3)[mesh.enode].sum(axis=1)
                dc_xp += weights[ci] * 2.0 * (ue @ domain.gravity) * mesh.pitch ** 3 / 8.0
        if c0 is None:
            c0 = comp
        dproj_e = _dproject(xt, beta, eta_e)
        dproj_e[pres] = 0.0
        dproj = _dproject(xt, beta, opts.eta)
        dproj[pres] = 0.0
        if void is not None:
            dproj[void] = 0.0
            dproj_e[void] = 0.0
        dc = ht @ (printed_grad(am_e, dc_xp) * dproj_e / hs)
        dv = ht @ (printed_grad(am_b, design.astype(float)) * dproj / hs)
        dc = np.minimum(dc, 0.0)
        if robust:
            # A worst case over directions swings: stiffening the weakest
            # direction makes another the weakest, and the full step of the
            # optimality criteria overshoots into it every iteration (a
            # square cantilever's webs flipped diagonal to diagonal and its
            # worst case came out 4.6 times its tuned-for-down twin's). The
            # mean of this iteration's and the last one's sensitivities
            # damps the swing: 5 times stiffer in its worst direction than
            # the twin, where undamped it was barely better.
            if dc_prev is not None:
                dc = 0.5 * (dc + dc_prev)
            dc_prev = dc
        dv = np.maximum(dv, 1e-12)
        excess = None
        if thick is not None:
            # max member: a cost on top of volume where a ball is too full,
            # weighted up while the solid's 95th percentile is over and
            # eased off when it is not (manufacture.MaxMember). The worst
            # ball sits at a junction and stays over whatever the weight:
            # steering on it ran the weight to 1000 and broke the beam.
            # Measured on the delivered (projected) field, but carried back
            # through the filter only, as if the projection were the
            # identity: deep in a thick member the projection's slope is
            # nil, so the true gradient only ever shaved the member's skin,
            # and measured on the filtered field the penalty let go as soon
            # as the design variables sank to 0.7, still projecting solid.
            excess, over, dp = thick(xp)
            mu = min(max(mu * 1.15, 0.1), 100.0) if over > 0.05 else mu * 0.9
            if mu > 0:
                dv = dv + mu * (ht @ (dp / hs))
        if domain.mirror:
            dc = _symmetrize(dc, domain.mirror)
            dv = _symmetrize(dv, domain.mirror)
        dc = _extrude(dc, columns, design)
        dv = _extrude(dv, columns, design)

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
            if void is not None:
                x_new[void] = 0.0
            if domain.mirror:
                x_new = _symmetrize(x_new, domain.mirror)
            x_new = _extrude(x_new, columns, design)
            if void is not None:
                x_new[void] = 0.0
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
        if excess is not None:
            rec["maxMemberExcess"] = excess
            rec["maxMemberWeight"] = mu
        history.append(rec)
        if on_iter is not None:
            on_iter(rec, xp)

        if beta < opts.beta_max and (change < opts.tol * 2 or since_beta >= _beta_hold(opts, beta)):
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
    if opts.dilate > 0:
        # ...then CLOSE it: dilate, erode back. A hole or gap narrower than
        # the dilation comes out filled. Only ever adds: with this filter a
        # member near R wide erodes by more than it dilated (a sealed-tube
        # test lost 6% of its volume and tripled its compliance), so the
        # closing is the union with what it closed.
        xd = _project((h @ xp) / hs, beta, eta_d)
        xd[pres] = 1.0
        xp = np.maximum(xp, _project((h @ xd) / hs, beta, opts.eta + opts.dilate))
        xp[pres] = 1.0
        xp = printed(am_v, xp)
    if void is not None:
        xp = xp.copy()
        xp[void] = 0.0
    if opts.access is not None:
        xp = _machined(domain, xp, opts.access)
    # Final analysis on the delivered field (it moved since the last solve).
    e = E_MIN + xp ** PENALTY * (1 - E_MIN)
    fg = None
    if domain.gravity is not None:
        from .domain import gravity_force
        fg = gravity_force(mesh, xp, domain.gravity, domain.fixed)
    loads, layout = expand(domain.loads, spreads, fg)
    us = fe.solve(e, xp, us, rtol=1e-6, loads=loads)
    if opts.prune:
        xp, us = _prune_dead(fe, domain, xp, us, loads, design, pres)
        if opts.access is not None:
            # pruning a dead end can leave a void under material: refill
            xq = _machined(domain, xp, opts.access)
            if (xq != xp).any():
                xp = xq
                us = fe.solve(E_MIN + xp ** PENALTY * (1 - E_MIN), xp, us, rtol=1e-6, loads=loads)
    case_c, case_d, comp = [], [], 0.0
    energy = np.zeros(n)
    stiff = E_MIN + xp ** PENALTY * (1 - E_MIN)
    for ci, wc in enumerate(collapse(loads, layout, us, spreads)):
        case_c.append(wc.compliance)
        case_d.append(None if wc.direction is None else [float(v) for v in wc.direction])
        comp += weights[ci] * wc.compliance
        energy += weights[ci] * stiff * sum(m * fe.strain_energy(v) for m, v in wc.energy)
    return SimpResult(x=x, xphys=xp, compliance=comp, case_compliance=case_c,
                      volume_fraction=float(xp[design].mean()), iterations=it,
                      converged=converged, seconds=time.time() - t0, history=history,
                      case_direction=case_d, energy=energy / mesh.pitch ** 3)
