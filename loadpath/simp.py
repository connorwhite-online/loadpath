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

from . import checkpoint as _checkpoint
from .directions import collapse, expand
from .domain import Domain
from .fem import HexMesh, Solver, Stiffness, hex8_ke
from .am import overhang_eps
from .manufacture import AccessFilter, MaxMember, machinable, tie_groups

import os

__all__ = ["SimpOptions", "SimpResult", "filter_radius", "density_filter", "optimize"]

E_MIN = 1e-6
# casting.feed: the full pour (castfeed.FeedTerm.pour) judges the design every
# this many iterations, once the projection is this sharp (below it the design
# is too grey to say which cells are metal).
FEED_POUR_EVERY = 8
FEED_POUR_BETA = 2.0

# The search's buckling penalty acts on load cases whose buckling factor is
# under 1 / (1 - BUCKLE_MMA) times the limit, from projection sharpness
# BUCKLE_BETA on (a grey design reads far below its sharpened self: the hook
# 0.65 at its first iteration, 4.7 at its last).
BUCKLE_MMA = 0.15
BUCKLE_BETA = 8.0
PENALTY = 3.0
# iterations between refreshes of the in-loop cutter fill (SimpOptions.tool_fill)
TOOL_FILL_EVERY = 5
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
# LOADPATH_SOLVER=mg|assembled|direct forces one.
MG_MIN_CELLS = 100_000
# From this many cells the multigrid is also the solve of choice for a
# system too big to factor quickly (direct.small_dofs), on its own grid:
# it ran the printed hook's 17k- and 71k-cell grids 3x and 13x quicker
# than AMG-CG. Below, a factorization always is.
MG_AUTO_CELLS = 20_000
# ... until it takes this many CG iterations on a design: its coarse grids
# cannot carry members one or two cells thick (the wrist yoke at 4%: 470-520
# iterations for AMG's 28, and a delivered design a quarter as stiff). From
# then on that run factors (or, too big for that, runs AMG-CG). The slow
# answer itself stands where the multigrid still converged inside its own
# stall limit (mg.GridElasticity.STALL): the first solve of the yoke's later
# trial parts, the whole 119k-cell grid live at a soft projection, took 65
# iterations and came as close to a factorization's as AMG-CG's does (energy
# error 7e-5 against 5e-5), and solving that system again by AMG-CG with its
# adjoints was 75 s of the part's 340 against 20.
MG_SLOW_ITERS = 60
# A slow multigrid gives way to a factorization wherever one fits. Where the
# live system is too big for that, it gives way to AMG-CG only once less
# than this share of the grid is live. The multigrid's work is the whole
# grid's whatever is live; AMG-CG's is the live cells', four to five times
# as much per dof plus a hierarchy to build for each design. A free-flying
# frame (130k cells, seven load cases on a 3-2-1 hold, its inertia loads on
# every node so the whole grid is the system at any design, _live_nodes):
# the multigrid 13-20 s an iteration at 65-80 CG iterations, AMG-CG 20-44 s.
# The wrist yoke at 4% (1,650 live cells of 27,634): the multigrid's 470-520
# iterations against AMG-CG's 28 on the live cells alone. They cross near
# half the grid.
MG_SLOW_LIVE = 0.5


def _forced_solver() -> str:
    force = os.environ.get("LOADPATH_SOLVER", "").lower()
    return {"multigrid": "mg", "amg": "assembled"}.get(force, force)


def _use_mg(n_cells: int) -> bool:
    force = _forced_solver()
    if force == "mg":
        return True
    if force in ("assembled", "direct"):
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
        # multigrid still trusted on this run's designs (MG_SLOW_ITERS)
        self._mg_ok = True
        # ... and a factorization still on offer: not once a factor would not
        # fit in the memory that is free (direct.factor_budget)
        self._direct_ok = True
        self.last_solver = None
        # nodes that carry a load in any case (_live_nodes)
        self._loaded = np.zeros(mesh.n_nodes, bool)
        for f in domain.loads:
            self._loaded |= (np.asarray(f).reshape(-1, 3) != 0).any(axis=1)
        if _use_mg(mesh.n_elem):
            self._multigrid()

    def _multigrid(self):
        if self._mg is None:
            from .mg import GridElasticity

            d = self.domain
            self._mg = GridElasticity(d.mesh, self.nu, d.fixed, springs=d.springs, rtol=self.rtol, maxiter=2000)
        return self._mg

    def _live_nodes(self, alive):
        """The nodes of the system the live cells leave, or None where the
        whole grid is solved: most of it live (REDUCE_BELOW), or a load on a
        node none of them touches. A free-flying part's loads are balanced by
        its inertia on every node of the grid, so it is always the whole
        grid, and a route chosen by its live cells would be for a system
        that is never solved."""
        if alive.mean() >= REDUCE_BELOW:
            return None
        mesh = self.domain.mesh
        nodes = np.zeros(mesh.n_nodes, bool)
        nodes[mesh.enode[alive].ravel()] = True
        if self._loaded[~nodes].any():
            return None
        return nodes

    def _route(self, alive) -> str:
        """Which solver takes this design: "direct" (a sparse factorization,
        direct.py), "mg" (matrix-free multigrid, mg.py) or "amg" (assembled,
        AMG-preconditioned CG, fem.Solver). By the size of the system the
        live cells leave and the grid: see direct.py for the measurements."""
        from . import direct

        force = _forced_solver()
        mesh = self.domain.mesh
        if force == "mg":
            return "mg"
        if force == "assembled":
            return "amg"
        nodes = self._live_nodes(alive)
        if nodes is None:
            live, n_dof = 1.0, mesh.n_dof
        else:
            live, n_dof = float(alive.mean()), 3 * int(nodes.sum())
        factor = self._direct_ok and direct.backend() is not None
        has_mg = mesh.n_elem >= min(MG_AUTO_CELLS, MG_MIN_CELLS)
        if factor and (force == "direct" or n_dof <= direct.small_dofs()):
            return "direct"
        if force == "direct" and direct.backend() is None:
            return "amg"
        if self._mg_ok and has_mg:
            return "mg"
        if factor and n_dof <= direct.large_dofs():
            return "direct"
        # Nothing to factor with, or too big for it. The multigrid if it
        # still suits this run, and, slow, while most of the grid is live
        # (MG_SLOW_LIVE); AMG-CG on what is live otherwise.
        if (self._mg is not None or has_mg) and (self._mg_ok or live >= MG_SLOW_LIVE):
            return "mg"
        return "amg"

    def _solver(self, holder: dict, mesh, mode: str, rtol):
        """The direct or the AMG solver of a (sub)system, made on first use."""
        if mode == "direct":
            if holder.get("direct") is None:
                from .direct import DirectSolver

                holder["direct"] = DirectSolver(mesh.node_ijk)
            return holder["direct"]
        if holder.get("amg") is None:
            holder["amg"] = Solver(mesh, rtol=rtol)
        holder["amg"].rtol = rtol
        return holder["amg"]

    def _full_system(self):
        if self._full is None:
            mesh = self.domain.mesh
            self._full = (Stiffness(mesh, self.nu, self.domain.fixed), {})
        return self._full

    def _connected(self, alive):
        mesh = self.domain.mesh
        g = np.zeros(self.domain.dims, bool)
        ijk = mesh.elem_ijk
        g[tuple(ijk[alive].T)] = True
        lbl, _ = ndimage.label(g, structure=_CONN26)
        el = lbl[tuple(ijk.T)]
        keep_l = np.unique(el[alive & self.support_touch])
        keep = alive & np.isin(el, keep_l[keep_l > 0])
        stock = getattr(self.domain, "stock", None)
        if stock is not None:
            # a bought member's cells stay in every reduced system: its stiffness is a fixed operator
            # on them (stock.py) less the printed material's, which only holds with the cell there
            keep = keep | stock
        return keep

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
            return c["keep"], c["fd"], c["stiff"], (c["solver"], c["mesh"])
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
                     "solver": {}, "mesh": sub, "age": 0}
        return keep, fd, self._sub["stiff"], (self._sub["solver"], sub)

    def solve(self, e, xp, us, rtol=None, loads=None):
        """Solve every load case for moduli e. Returns per-case displacement on
        the full node set. `loads` overrides the domain's (self-weight makes
        them depend on the design)."""
        try:
            return self._solve(e, xp, us, rtol, loads)
        except MemoryError:
            # a factor that would not fit beside what else is running: the
            # iterative solvers take this design and the rest of the run
            self._direct_ok = False
            return self._solve(e, xp, us, rtol, loads)

    def _solve(self, e, xp, us, rtol=None, loads=None):
        d = self.domain
        loads = loads if loads is not None else d.loads
        mesh = d.mesh
        rtol = rtol or self.rtol
        scale = getattr(mesh, "stock_scale", None)
        if scale is not None:
            # a bought member's cells hold only their printed share of this modulus (stock.py)
            e = e * scale
        alive = (xp > ALIVE) | self.anchor
        mode = self._route(alive)
        if mode == "mg":
            try:
                out = self._solve_mg(e, alive, us, rtol, loads)
            except RuntimeError:
                if _forced_solver() == "mg":
                    raise
                out = None
            mg = self._mg
            if out is not None and not (mg.last_iters > MG_SLOW_ITERS and _forced_solver() != "mg"):
                self.last_solver = "mg"
                return out
            # Stalled or slow: the rest of the run leaves the multigrid for
            # a factorization where one fits, for AMG-CG where little of the
            # grid is live (_route). This design's answer stands if the
            # multigrid converged and is to carry on, or converged within
            # its own stall limit (MG_SLOW_ITERS).
            self._mg_ok = False
            mode = self._route(alive)
            if out is not None and (mode == "mg" or mg.last_iters <= mg.STALL):
                self.last_solver = "mg"
                return out
            if mode == "mg":
                mode = "amg"         # (it did not converge on this design)
        self._last_e = e
        self._mg_reduced = False
        if self._live_nodes(alive) is not None:
            sub = self._subsystem(alive, rtol)
            subs = None
            if sub is not None:
                keep, fd, stiff, (holder, smesh) = sub
                solver = self._solver(holder, smesh, mode, rtol)
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
                self.last_solver = mode
                return out
        stiff, holder = self._full_system()
        if mode == "direct":
            from . import direct

            if mesh.n_dof > direct.large_dofs():
                mode = "amg"
        solver = self._solver(holder, mesh, mode, rtol)
        k = stiff.assemble(e)
        if d.springs is not None:
            k = (k + d.springs).tocsr()
        out = solver.solve_many(k, list(loads), list(us))
        self.last_iters, self.last_cells = solver.last_iters, mesh.n_elem
        self._last = (k, None, solver)
        self.last_solver = mode
        return out

    def _solve_mg(self, e, alive, us, rtol, loads):
        """The multigrid path: the same reduction (cells below ALIVE and
        islands left out), done by zeroing their moduli on the fixed grid
        rather than building a smaller mesh."""
        mg = self._multigrid()
        mg.rtol = rtol
        keep = self._connected(alive) if alive.mean() < REDUCE_BELOW else np.ones(len(e), bool)
        try:
            mg.set_moduli(np.where(keep, e, 0.0))
            if not keep.all() and any(np.any(f[~mg.live_dofs()]) for f in loads):
                keep = np.ones(len(e), bool)
                mg.set_moduli(e)
            out = mg.solve_many(list(loads), list(us))
        except RuntimeError:
            if keep.all():
                # stalled on the whole grid (its coarse levels average a
                # sharp design's moduli and can lose a thin load path: a
                # drone frame at beta 8, its crash cases on thin duct
                # skins): solve() sends this design to a factorization or
                # to AMG-CG, which builds its hierarchy from the matrix
                raise
            # The live cells hang on the rest by an edge or a corner (a hinge),
            # or a coarse level lost a thin support: singular or stalled. The
            # full grid, void at E_MIN, holds it (as in the direct path).
            keep = np.ones(len(e), bool)
            mg.set_moduli(e)
            out = mg.solve_many(list(loads), [None] * len(loads))
        self.last_iters, self.last_cells = mg.last_iters, int(keep.sum())
        self._last = None
        self._mg_reduced = not keep.all()
        self._last_e = e
        return out

    def resolve(self, rhss, x0s=None):
        """Solve more right-hand sides (adjoints) against the matrix of the
        last solve(); full-size in, full-size out."""
        if self._last is None:
            try:
                return self._mg.solve_many(list(rhss), x0s)
            except RuntimeError:
                if not getattr(self, "_mg_reduced", False):
                    raise
                # a hinge the loads did not excite but an adjoint does (see
                # solve): solve on the full grid, void at E_MIN
                self._mg_reduced = False
                self._mg.set_moduli(self._last_e)
                return self._mg.solve_many(list(rhss), None)
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
            from . import direct

            stiff, holder = self._full_system()
            mesh = self.domain.mesh
            mode = "direct" if isinstance(solver, direct.DirectSolver) and self._direct_ok \
                and mesh.n_dof <= direct.large_dofs() else "amg"
            k = stiff.assemble(self._last_e)
            if self.domain.springs is not None:
                k = (k + self.domain.springs).tocsr()
            try:
                solver = self._solver(holder, mesh, mode, self.rtol)
                out = solver.solve_many(k, list(rhss), x0s)
            except MemoryError:       # (see solve)
                self._direct_ok = False
                solver = self._solver(holder, mesh, "amg", self.rtol)
                out = solver.solve_many(k, list(rhss), x0s)
            self._last = (k, None, solver)
            return out
        out = []
        for v in sub:
            u = np.zeros(n_dof)
            u[fd] = v
            out.append(u)
        return out

    def strain_energy(self, u):
        if self._mg is not None:
            return self._mg.strain_energy(u)
        from . import femops

        if femops.available():
            return femops.element_energy(u, None, self.domain.mesh.edof, self.ke)
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
    # Measure each step's length from how the gradient changed along the last
    # one (see the loop and docs/CONVERGENCE.md), and keep the printed-part
    # overhang rule only as sharp as the projection. False: the fixed limits
    # (move, then move / 2 past beta 4) the benchmarks' reference codes use.
    adaptive_move: bool = True
    tol: float = 0.01
    # How closely each iteration's displacements are solved. A Krylov solve
    # stopped at 1e-2 or 1e-3 gives a gradient good enough to descend on, but
    # which answer it stops at depends on the last bit of its input: two
    # runs one rounding error apart were 4e-7 apart in compliance after the
    # second solve and delivered hooks 0.5% apart in mass. At 1e-4 they stay
    # within 1e-8 until the sharpest stage and deliver within 0.13%, for a
    # tenth more time; tighter than that buys nothing the sharpest stage
    # does not undo (docs/CONVERGENCE.md). adaptive_rtol loosens the early
    # stages 10x and 3x, as before.
    solve_rtol: float = 1e-4
    adaptive_rtol: bool = False
    # Stop once beta is at max and compliance has moved less than this
    # (relative) over the last `stall_window` iterations. With the measured
    # step the compliance falls every iteration, so this reads "six more
    # iterations bought under 0.2%". (With fixed limits a projected design
    # chatters at the limit and the compliance flickers and drifts up, so
    # the same test fired on whichever iteration the flicker happened to
    # cross it: docs/CONVERGENCE.md.)
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
    # Count the cutter's fill (milling toolDiameter / stepDown) inside the
    # loop once the projection is sharp (see tool_fill in simp); False
    # leaves it to delivery alone. LOADPATH_TOOL_FILL=0 turns it off.
    tool_fill: bool = field(default_factory=lambda: os.environ.get("LOADPATH_TOOL_FILL", "1") != "0")
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
    # Deflation (after Papadopoulos, Farrell & Surowiec 2021): delivered
    # designs (xphys per element) to repel this run from. The objective is
    # compliance times prod_k (deflate_weight * d_k^-deflate_power + 1), d_k
    # the RMS density difference to design k over the design cells, so a run
    # that heads back into a basin already found pays for it and settles in
    # another. Empty: off.
    deflate: tuple = ()
    deflate_weight: float = 0.05
    deflate_power: float = 2.0
    # Linear buckling as a constraint (bucklingopt.py): the smallest load
    # factor the eroded design may buckle at, or None; buckling_modulus (MPa)
    # turns the unit-modulus model's factor into this one's. Off the limit
    # (see BUCKLE_MMA) nothing changes; near it the compliance takes an
    # adaptive buckling penalty the optimality criteria step on.
    buckling: float | None = None
    buckling_modulus: float = 1.0
    # What binds, per load case (study._binding): b_i = binding[i] * C_i is
    # how close case i is to its limit (its deflection over the goal's, or
    # the goal's safety factor over its own, whichever is nearer), read on a
    # verified design and carried through this run in proportion to the
    # case's compliance C_i. The objective is then the p-norm of the b_i
    # (binding_power p), a smooth maximum: the stiffness goes where the
    # limit binds instead of to a case far under it. Scaled so that on the
    # design it was read from it equals that design's weighted compliance.
    # None: the weighted sum of compliances.
    binding: tuple | None = None
    binding_power: float = 8.0


def binding_weights(binding, case_c, power: float) -> tuple[np.ndarray, float]:
    """The case weights that make sum_i w_i C_i the p-norm J = (sum_i
    (beta_i C_i)^p)^(1/p) of the cases' binding ratios, and its gradient
    sum_i w_i dC_i/dx: w_i = beta_i (beta_i C_i / J)^(p-1). Returns (w, J)."""
    beta = np.asarray(binding, float)
    b = beta * np.maximum(np.asarray(case_c, float), 0.0)
    top = float(b.max()) if len(b) else 0.0
    if not top > 0.0:
        return beta.copy(), 0.0
    t = b / top
    j = top * float(np.sum(t ** power)) ** (1.0 / power)
    return beta * (b / j) ** (power - 1.0), j


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
    # buildSim warp design: the delivered field's as-built measures, or None
    warp: dict | None = None
    # constrained: the history record of the delivered design (the lightest
    # one inside the limits at the sharpest projection), or None when the run
    # ended before there was one and the last iterate is delivered
    delivered: dict | None = None


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
# How closely each step holds the volume (relative): to rounding, so the step
# is a continuous function of the design it came from
VOLUME_RTOL = 1e-13
# The measured step (SimpOptions.adaptive_move): the limit never falls below
# half or grows past 1.2 times itself in one iteration. An overshoot is acted
# on at once; room to grow is taken slowly, because the reading is one number
# for the whole part and the part is not one valley.
STEP_SHRINK = 0.5
STEP_GROW = 1.2
PRUNE_PASSES = 3


def _prune_dead(fe, domain, xp, us, loads, design, pres, soften=None):
    """Delete dead-end material from the delivered design.

    Projection continuation freezes the layout once beta is high: a member
    the optimizer stopped needing can no longer shrink away (its gradient
    lives only in the thin transition band), and compliance does not notice
    a strut that carries nothing. The robot arm's wrist link came out with several
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
        if getattr(domain, "joints", None) is not None:
            dead &= ~domain.joints        # what holds a thin drawn wall together
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
        before = (xp, us)
        held = _held_pieces(fe, domain, xp)
        xp = xp.copy()
        xp[dead] = 0.0
        # and whatever that cut loose from the supports (a tip whose root
        # was a dead cluster under the size floor)
        loose = (xp > 0.5) & ~fe._connected((xp > 0.5) | fe.anchor) & ~pres
        xp[loose] = 0.0
        if _held_pieces(fe, domain, xp) > held:
            # the cut parted the design: a drawn piece, a load or a support
            # that was joined to the rest through members with next to no
            # energy in them (a free frame's ducts, held on against their
            # own inertia alone) is now on its own. Pruning is a clean-up:
            # the pass is undone. Asked of the shape, not of the solve: a
            # factorization returns the floating piece's displacement on
            # the void's stiffness where CG ran out of iterations.
            xp, us = before
            break
        e = E_MIN + xp ** PENALTY * (1 - E_MIN)
        if soften is not None:
            e = e * soften          # each cell's modulus at its temperature (thermal.StudyHeat.g)
        try:
            us = fe.solve(e, xp, us, rtol=1e-6, loads=loads)
        except RuntimeError:
            # the cut left a mechanism some other way (a hinge): undone too,
            # rather than the run lost (a drone frame, iteration 50)
            xp, us = before
            break
    return xp, us


def _keep_cores(domain, opened, analysed, blueprint, radius):
    """The opening, with every member the optimizer analysed still in it.

    The opening grows the analysed (eroded) design back by filtering it and
    cutting at eta - erode. That returns a member whose analysed core is at
    least about a third of the filter radius across, and nothing for a
    thinner core: a round column whose blueprint is the minimum member wide
    has a core one cell across, the filter spreads that cell below the cut,
    and the column is not delivered. The optimizer does build such columns.
    Under overhang control a column that only holds up what is printed above
    it carries no load, so it is made as thin as the analysed design allows;
    its stiffness was credited for nothing, but everything above it was
    credited as printable because of it. Delivered without it, the overhang
    rule takes that away too (the printed hook lost its top chord's root at
    the wall and read safety factor 0.2 for an analysed 2.5).

    So a cell analysed as solid and opened as void brings back the blueprint
    round it, out to half the minimum member the filter enforces (R / 1.3,
    see filter_radius): the member returns at the width the volume was
    counted at, and a sheet with no analysed core (the membranes the opening
    is there to drop) still has nothing to bring it back. A design with no
    such cell is delivered exactly as before."""
    lost = (analysed > 0.5) & (opened < 0.5)
    if not lost.any():
        return opened
    # a cell is in reach when any of it is: its centre within the reach plus half a cell
    reach = radius / 2.6 + 0.5 * domain.pitch
    n = int(np.ceil(reach / domain.pitch))
    g = np.mgrid[-n:n + 1, -n:n + 1, -n:n + 1]
    ball = (g ** 2).sum(axis=0) * domain.pitch ** 2 <= reach ** 2 * (1.0 + 1e-9)
    ijk = tuple(domain.mesh.elem_ijk.T)
    near = ndimage.binary_dilation(domain.full_grid(lost.astype(float)) > 0.5, structure=ball)[ijk]
    return np.where(near, np.maximum(opened, blueprint), opened)


def _held_pieces(fe, domain, xp) -> int:
    """How many separate pieces of the design hold something: a drawn
    region, a load or a support (fe.anchor). One for a sound part."""
    solid = (xp > 0.5) | fe.anchor
    g = np.zeros(domain.dims, bool)
    ijk = tuple(domain.mesh.elem_ijk.T)
    g[ijk] = solid
    lbl, _ = ndimage.label(g, structure=_CONN26)
    return len(np.unique(lbl[ijk][fe.anchor]))


def _heat_state(heat, x, fg, reuse=False):
    """Solve the temperature at densities x and return (self-weight plus
    expansion load, expansion load or None), in the optimizer's units: forces
    in N against unit-modulus stiffness, so the expansion's E0 is in the
    load."""
    heat.temperature(x, reuse=reuse)
    if not heat.expands:
        return fg, None
    fth = heat.load(np.where(x > ALIVE, x ** PENALTY * (1 - E_MIN), 0.0), heat.E0)
    return (fth if fg is None else fg + fth), fth


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


def _warp_report(domain: Domain, xp):
    """The delivered field's as-built measures (buildsim): max distortion
    and recoater lift, mm, on the optimizer's build model and, when that
    one was coarsened, on the full grid too (one forward build, which is
    what the limits are checked on)."""
    ws = domain.warp
    e = E_MIN + xp ** PENALTY * (1 - E_MIN)
    v, _ = ws.term.measures(e)
    out = {"maxDistortion": v["max"], "recoaterLift": v["lift"], "meanSquare": v["mean"],
           "limits": {"maxDistortion": ws.max_distortion, "maxRecoaterLift": ws.max_lift},
           "weight": ws.weight, "coarsen": ws.coarsen}
    if ws.coarsen > 1:
        vf, _ = ws.term.fine().measures(e)
        out["optimizerGrid"] = {"maxDistortion": v["max"], "recoaterLift": v["lift"],
                                "fullOverCoarse": dict(ws.term.ratio)}
        out.update(maxDistortion=vf["max"], recoaterLift=vf["lift"], meanSquare=vf["mean"])
    return out


def _relieved(domain, xe):
    """The cases' loads: as the domain holds them, or (inertiaRelief) the
    loads as written balanced by the design's own inertia, its mass from
    the densities `xe`. Returns (loads, per case Phi a or None)."""
    raw = getattr(domain, "raw_loads", None)
    if raw is None:
        return domain.loads, None
    from .domain import inertia_balance, node_mass

    phi = _rigid_modes(domain)
    m = node_mass(domain.mesh, np.maximum(xe, 1e-3)) + _carried(domain)
    loads, accs = [], []
    bodies = getattr(domain, "bodies", None) or []
    for f in raw:
        fb, acc = inertia_balance(f, m, phi, bodies)
        fb[domain.fixed] = 0.0
        loads.append(fb)
        accs.append(acc)
    return loads, accs


def _rigid_modes(domain):
    phi = getattr(domain, "_phi", None)
    if phi is None:
        from .fem import rigid_body_modes

        phi = domain._phi = rigid_body_modes(domain.mesh.node_xyz)
    return phi


def _carried(domain):
    """inertiaRelief.masses per node (domain.carried_volume), or 0."""
    c = getattr(domain, "carried", None)
    return 0.0 if c is None else c


def _rigid_free_m(domain, xe, u):
    """u less its rigid motion, taken orthogonal to every inertia force a
    rigid acceleration raises (domain.rigid_free: M from the densities and
    what the part carries)."""
    from .domain import node_mass, rigid_free

    phi = _rigid_modes(domain)
    m = node_mass(domain.mesh, np.maximum(xe, 1e-3)) + _carried(domain)
    return rigid_free(u, m, phi, getattr(domain, "bodies", None) or [])


def _relief_heat_grad(domain, heat, heat_state, xe, us, accs, layout, weights):
    """Inertia relief under heat (thermal.StudyHeat.sensitivities): the
    load f - M Phi a moves with the design's mass, and the objective's
    adjoint is lam (K lam = f - 2 f_th with expansion, u without), so
    dJ/dx_e = -(pitch^3 / 8) sum over the cell's nodes of (Phi a) . (u + lam),
    both taken M-orthogonal to the rigid modes (the change in a drops out,
    as in the cold case, where lam = u and this is the -2 (Phi a) . u there)."""
    mesh = domain.mesh
    lams = heat_state.get("lams") if heat.expands else None
    out = np.zeros(mesh.n_elem)
    for ci, acc in enumerate(accs):
        i = layout[ci][0]
        s = _rigid_free_m(domain, xe, us[i])
        s = s + (_rigid_free_m(domain, xe, lams[ci]) if lams is not None else s)
        au = (acc.reshape(-1, 3) * s.reshape(-1, 3)).sum(axis=1)
        out -= weights[ci] * au[mesh.enode].sum(axis=1) * mesh.pitch ** 3 / 8.0
    return out


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
    """One optimization, checkpointed when LOADPATH_CHECKPOINT names a folder
    (checkpoint.py): a call this run already finished returns its stored
    result, and one a reset interrupted continues from its last loop state."""
    ck = _checkpoint.Call.open("simp", domain, nu, opts)
    if ck is None:
        return _optimize_run(domain, nu, opts, on_iter, should_stop, None)
    res = ck.done()
    if res is not None:
        return res
    res = _optimize_run(domain, nu, opts, on_iter, should_stop, ck)
    ck.finish(res)
    return res


# The loop's state between two iterations, as checkpoint.Call saves and
# restores it: everything a later iteration reads that an earlier one wrote.
_LOOP_STATE = ("x", "beta", "since_beta", "us", "dc_prev", "weights", "history", "c0", "change", "heat_state",
               "mma_s", "warp_state", "buck_last", "buck_age", "buck_nu", "modal_last", "modal_age", "modal_nu",
               "mu", "tool_fill", "com_state", "feed_state", "lam_last", "move_k", "step_prev")


def _optimize_run(domain: Domain, nu: float, opts: SimpOptions, on_iter, should_stop, ck) -> SimpResult:
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
    jit = os.environ.get("LOADPATH_JITTER")
    if jit:
        # Repeatability probe: noise of one part in 10^9 on the starting
        # design, seeded by the variable's value. It stands in for what
        # another machine, thread count or library build does to the last
        # bits; runs with different seeds show how far a study's answer
        # moves for no reason (docs/CONVERGENCE.md).
        # "<seed>:<size>" sets another size, e.g. 3:1e-15 for last-bit noise.
        seed, _, size = jit.partition(":")
        x = np.clip(x * (1.0 + float(size or 1e-9) * np.random.default_rng(int(seed)).standard_normal(n)), 0.0, 1.0)
    x[pres] = 1.0
    void = opts.void if opts.void is not None and opts.void.any() else None
    if void is not None:
        x[void] = 0.0
    # massProperties: each cell's cost as volume plus its share of inertia
    # about the axis (domain.volume_weights, mean 1 over the design cells),
    # so the budget buys material near the axis cheaper than far out
    vw = getattr(domain, "volume_weights", None)
    wdesign = design.astype(float) if vw is None else np.where(design, vw, 0.0)
    blk = getattr(domain, "blockage", None)
    com_cfg = getattr(domain, "center_of_mass", None)
    com_state = {}
    feed_t = getattr(domain, "feeding", None)
    feed_state = {}
    com_cen = (mesh.origin + (mesh.elem_ijk + 0.5) * mesh.pitch) if com_cfg is not None else None
    target = vf * float(wdesign.sum())

    beta = float(opts.beta_start)
    since_beta = 0
    spreads = domain.spreads or [None] * len(domain.loads)
    us = [None] * len(expand(domain.loads, spreads)[0])
    robust = any(s is not None for s in spreads)
    dc_prev = None
    weights = np.asarray(domain.case_weights, float)
    weights0 = weights.copy()
    impacts = getattr(domain, "impacts", None) or []
    # what binds (SimpOptions.binding): not with heat (its own sensitivities)
    # or impacts (whose weights are the stopping force's)
    bind = opts.binding if opts.binding is not None and len(opts.binding) == len(weights) \
        and domain.thermal is None and not impacts else None
    history = []
    c0 = None
    converged = False
    it = 0
    change = 1.0

    heat = domain.thermal
    heat_state = {}
    mma_s = None
    warp_state = {}
    buck = None
    if opts.buckling is not None and not robust and domain.gravity is None and heat is None:
        from .bucklingopt import BucklingModel

        buck = BucklingModel(domain, nu)
    buck_last, buck_age, buck_nu = None, 0, 0.0
    modal = getattr(domain, "modal", None) if not robust else None
    modal_last, modal_age, modal_nu = None, 0, 0.0

    ijk = tuple(mesh.elem_ijk.T)
    # The cutter's own fill (manufacture.machinable: slots narrower than the
    # cutter, inside corners tighter than its radius, 2.5-axis floors),
    # held as passive solid once the projection is sharp and refreshed
    # every TOOL_FILL_EVERY iterations: the volume it adds is counted, so
    # the optimizer takes it from elsewhere instead of finding it added at
    # delivery. Passive (no gradient), so it cannot pull the design apart
    # the way a differentiable fill did (manufacture.py).
    tool_fill = {"mask": None, "on": bool(opts.access is not None and opts.tool_fill
                                         and (opts.access.tool_radius or opts.access.step))}
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
            out = am.forward(domain.full_grid(v))[ijk]
            m = tool_fill["mask"]
            return out if m is None else np.where(m, 1.0, out)

        def printed_grad(am, g):
            m = tool_fill["mask"]
            if m is not None:
                g = np.where(m, 0.0, g)
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
    # The step is measured (see the loop) except under a maximum member: that
    # limit is a weight ramped on a percentile, not a function whose slope
    # along a step can be read, and with the weight at its cap the measured
    # step is so short the limit is still unmet after 200 iterations.
    measured = opts.adaptive_move and thick is None
    # The sharpest stage ends on its schedule like every stage before it,
    # for the plain compliance and volume problem. Its iterations past the
    # schedule were measured on the printed hook: cut at 35 to 75 iterations
    # the part read safety factor 1.97 to 2.16 at 32.6 to 33.6 g, all on one
    # line of strength against mass, so they buy nothing a larger volume
    # fraction does not; and a stop that waits for the compliance to stall
    # ends two runs a last bit apart at different iterations (83 against 85,
    # 0.1% apart in strength; on the schedule, equal to 8 digits). A problem
    # with a weight still climbing (buckling, vibration, warp, centre of
    # mass, casting feed, a maximum member, a temperature swing) is not at
    # rest on a schedule and keeps the stall rule.
    # (LOADPATH_STOP=stall restores the stall rule, to measure against.)
    scheduled = (measured and os.environ.get("LOADPATH_STOP", "hold") != "stall"
                 and not (heat is not None and heat.swing_limit is not None)
                 and buck is None and modal is None and domain.warp is None and com_cfg is None and feed_t is None)
    mu = 0.0

    def eroded(xt, b):
        xe = _project(xt, b, eta_e)
        xe[pres] = 1.0
        if void is not None:
            xe[void] = 0.0
        return printed(am_e, xe)

    step_prev = None
    move_k = opts.move
    lam_last = None
    vol_evals = 0
    it0 = 0
    if ck is not None:
        ck.add_helpers(buck=buck, thick=thick, am_b=am_b, am_e=am_e, am_v=am_v)
        st = ck.restore()
        if st is not None:
            (x, beta, since_beta, us, dc_prev, weights, history, c0, change, heat_state, mma_s, warp_state,
             buck_last, buck_age, buck_nu, modal_last, modal_age, modal_nu, mu, tool_fill, com_state,
             feed_state, lam_last, move_k, step_prev) = (st[k] for k in _LOOP_STATE)
            it0 = it = st["it"]
            t0 -= st["seconds"]
            _checkpoint.set_flags(fe, st["fe"])
    for it in range(it0 + 1, opts.max_iterations + 1):
        if ck is not None and ck.boundary(it - 1):
            # A checkpoint iteration. The solver's caches (a reduced system
            # reused while the live cells stay inside it, a preconditioner
            # built on an earlier iteration's matrix) start over here, as a
            # resumed run's do, its flags (a fallback it settled on) kept: the
            # uninterrupted run and a resumed one go on from the same state,
            # to the bit.
            fe = _checkpoint.fresh(fe, lambda: _Analysis(domain, nu, opts.solve_rtol))
            if it - 1 != it0 and ck.due():
                ck.save({"it": it - 1, "seconds": time.time() - t0, "fe": _checkpoint.flags(fe), **dict(zip(_LOOP_STATE, (
                x, beta, since_beta, us, dc_prev, weights, history, c0, change, heat_state, mma_s, warp_state,
                buck_last, buck_age, buck_nu, modal_last, modal_age, modal_nu, mu, tool_fill, com_state,
                feed_state, lam_last, move_k, step_prev)))})
        if should_stop is not None and should_stop():
            break
        if opts.build_direction is not None and opts.adaptive_move:
            # the overhang rule only as sharp as the projection it reads (am.overhang_eps)
            am_b.eps = am_e.eps = am_v.eps = overhang_eps(beta)
        sec = None
        if tool_fill["on"] and beta >= opts.beta_max and since_beta % TOOL_FILL_EVERY == 0:
            tool_fill["mask"] = None
            _, xraw = physical(x, beta)
            blocked = machinable(domain.full_grid(xraw) > 0.5, opts.access, domain.pitch)[ijk]
            m = blocked & design & (xraw <= 0.5)
            if void is not None:
                m &= ~void
            tool_fill["mask"] = m if m.any() else None
        xt, xp = physical(x, beta, am_b)
        xe = eroded(xt, beta)
        e = E_MIN + xe ** PENALTY * (1 - E_MIN)
        fg = None
        if domain.gravity is not None:
            from .domain import gravity_force
            fg = gravity_force(mesh, xe, domain.gravity, domain.fixed)
        fth = None
        if heat is not None:
            # the design's own temperature, softening and expansion load
            # (thermal.StudyHeat)
            fg, fth = _heat_state(heat, xe, fg, reuse=True)
            e = e * heat.g
        # A force of unknown direction is three solves, one per axis, and
        # each case is judged in its worst direction (directions.py).
        loads_now, accs = _relieved(domain, xe)
        rhs, layout = expand(loads_now, spreads, fg)
        if len(us) != len(rhs):
            # an uncertain force alone in its case gains a fixed part (the
            # expansion load) once the design is warm
            us = [None] * len(rhs)
        # Sensitivities only need to point the right way while the design is
        # still soft; tighten the solve as the projection sharpens.
        rtol = opts.solve_rtol * (10.0 if beta <= 2 else 3.0 if beta <= 8 else 1.0) if opts.adaptive_rtol \
            else opts.solve_rtol
        us = fe.solve(e, xe, us, rtol=rtol, loads=rhs)
        dc_xp = np.zeros(n)
        comp = 0.0
        case_c = []
        if heat is not None:
            comp, case_c, dc_xp, _ = heat.sensitivities(fe, xe, rhs, us, fth, weights, domain.gravity, heat_state,
                                                        layout, spreads)
            if accs is not None:
                dc_xp += _relief_heat_grad(domain, heat, heat_state, xe, us, accs, layout, weights)
        wcs = collapse(rhs, layout, us, spreads) if heat is None else ()
        if bind is not None:
            weights, _ = binding_weights(bind, [wc.compliance for wc in wcs], opts.binding_power)
        for ci, wc in enumerate(wcs):
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
            if accs is not None and ci < len(accs):
                # inertia relief: f - M Phi a, M from the design. With u taken
                # M-orthogonal to the rigid modes the change in a drops out,
                # and d(f.u)/dx_e = -2 (pitch^3 / 8) sum over the cell's
                # nodes of (Phi a) . u
                ur = _rigid_free_m(domain, xe, wc.u_mean)
                au = (accs[ci].reshape(-1, 3) * ur.reshape(-1, 3)).sum(axis=1)
                dc_xp -= weights[ci] * 2.0 * au[mesh.enode].sum(axis=1) * mesh.pitch ** 3 / 8.0
        imp_rec = {}
        if impacts:
            # impact cases (loadCases[].impact): the force is what stops the
            # energy through the crusher and this design together
            # (impact.series_impact); the case is weighed at that force from
            # the next iteration on. The design is then the stiffest for the
            # force it will see, which spreads the energy it has to store as
            # evenly as its volume allows: the lowest peak stress.
            from .domain import impact_energy
            from .impact import series_impact

            e_mat, _ = domain.impact_material
            raw_l = getattr(domain, "raw_loads", None) or domain.loads
            scale = np.ones(len(weights0))
            for ci, case in impacts:
                fn2 = float(np.sum(raw_l[ci].reshape(-1, 3).sum(axis=0) ** 2))
                if fn2 <= 0 or ci >= len(case_c):
                    continue
                c_f = max(float(case_c[ci]), 0.0) / (e_mat * fn2)
                en, _ = impact_energy(domain.impact_study, case, raw_l[ci], accs[ci] if accs is not None else None)
                r = series_impact(en, c_f, case.impact.distance, case.impact.force)
                if r["peakForce"] is None:
                    continue
                scale[ci] = r["peakForce"] ** 2 / fn2
                imp_rec[case.name] = {"peakForce": round(r["peakForce"], 2), "structureShare": round(r["structureShare"], 3)}
            weights = weights0 * scale
        warp_v = None
        if domain.warp is not None:
            # buildSim warp design (buildsim.WarpTerm on the build axis):
            # compliance * (1 + w D / D0) for warpWeight, plus each limit
            # (maxDistortion, maxRecoaterLift) held by an adaptive weight,
            # raised while the design is over it and eased off under it (a
            # penalty the optimality-criteria step can take, as maxMember's)
            ws = domain.warp
            lam = warp_state.setdefault("lam", {"max": 0.0, "lift": 0.0})
            c_now = comp

            def warp_weights(v):
                if warp_state.get("d0") is None:
                    warp_state["d0"], warp_state["c0"] = max(v["mean"], 1e-30), c_now
                c0w = warp_state["c0"]
                w = {"mean": ws.weight * c0w / warp_state["d0"]}
                for key, lim in (("max", ws.max_distortion), ("lift", ws.max_lift)):
                    if lim:
                        over = v[key] / lim
                        lam[key] = min(max(lam[key] * 1.3, 0.05), 50.0) if over > 1.0 else lam[key] * 0.85
                        w[key] = lam[key] * c0w / lim
                return w

            n_warp = warp_state["n"] = warp_state.get("n", -1) + 1
            dxe = PENALTY * xe ** (PENALTY - 1) * (1 - E_MIN)
            if ws.coarsen > 1:
                # re-read the coarse grid's bias on the full grid every 10
                warp_v, dd = ws.term.measures(e, dxe, warp_weights, refresh=n_warp % 10 == 0)
            else:
                warp_v, dd = ws.term.measures(e, dxe, warp_weights)
            c0w = warp_state["c0"]
            extra = ws.weight * c0w * warp_v["mean"] / warp_state["d0"]
            if ws.max_distortion:
                extra += lam["max"] * c0w * warp_v["max"] / ws.max_distortion
            if ws.max_lift:
                extra += lam["lift"] * c0w * warp_v["lift"] / ws.max_lift
            dc_xp = dc_xp + dd
            comp = comp + extra
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
        if opts.deflate:
            # d log M / d xp, scaled by compliance (J = c M, divided by M so
            # the step size stays the plain compliance one), carried back
            # through the blueprint's projection and the filter
            g = np.zeros(n)
            for xk in opts.deflate:
                diff = np.where(design, xp - xk, 0.0)
                d2 = float(diff @ diff) / n_design
                dk = max(d2, 1e-12) ** 0.5
                a = opts.deflate_weight * dk ** -opts.deflate_power
                g += (-opts.deflate_power * a / (a + 1.0)) * diff / (d2 * n_design)
            dc = dc + ht @ (printed_grad(am_b, comp * g) * dproj / hs)
        com_rec = None
        if com_cfg is not None:
            # massProperties.centerOfMass: a penalty on the blueprint's centre
            # of mass past half the tolerance on each held axis, its weight
            # raised while the design is off target and eased once on it (as
            # the warp limits), carried back through the projection and filter
            tgt, tol = com_cfg
            M = float(xp.sum())
            C = (xp @ com_cen) / max(M, 1e-30)
            d = np.where(np.isfinite(tgt), C - np.nan_to_num(tgt), 0.0)
            over = np.maximum(np.abs(d) - 0.5 * tol, 0.0)
            w_c = com_state.setdefault("w", np.zeros(3))
            w_c[:] = np.where(np.abs(d) > tol, np.maximum(w_c * 1.2, 1.0), w_c * 0.9)
            w_c[:] = np.minimum(w_c, 50.0)
            cref = com_state.setdefault("c", comp)
            comp = comp + cref * float(np.sum(w_c * (over / tol) ** 2))
            gk = cref * w_c * 2.0 * np.sign(d) * over / tol ** 2          # d pen / d C_k
            gcom = ((com_cen - C) @ gk) / max(M, 1e-30)                    # d pen / d xp_e
            dc = dc + ht @ (printed_grad(am_b, gcom) * dproj / hs)
            com_rec = {"centerOfMass": [round(float(v), 3) for v in C],
                       "centerOfMassWeight": [round(float(v), 3) for v in w_c]}
        feed_rec = None
        if feed_t is not None:
            # casting.feed (castfeed.FeedTerm): the metal that freezes cut off from the feeder, as the
            # cavity it would leave, read on the eroded design as stiffness is (its necks are the
            # thinnest the delivered part's can be). The fast measure says which neck to thicken and
            # which pool to lighten, but it over-predicts where heavy lumps keep a short thin section
            # open, so the full pour is the judge: every FEED_POUR_EVERY iterations once the design
            # is sharp enough to pour, and the weight climbs only while that pour leaves more cavity
            # than a sound casting may hold, holds while the pour is just inside that, and eases once
            # it is sound with room to spare.
            xe_grid = domain.full_grid(xe)
            f_val, f_grad, feed_rec = feed_t.measure(xe_grid)
            if beta >= FEED_POUR_BETA and (feed_state.get("age", FEED_POUR_EVERY) >= FEED_POUR_EVERY):
                try:
                    feed_state["pour"] = feed_t.pour(xe_grid, order=feed_t.last["theta"].reshape(domain.dims)
                                                     * mesh.pitch)["cavityShare"]
                except Exception:  # noqa: BLE001 - a failed pour leaves the last reading standing
                    pass
                feed_state["age"] = 0
            feed_state["age"] = feed_state.get("age", FEED_POUR_EVERY) + 1
            poured = feed_state.get("pour")
            w_f = feed_state.get("w", 0.0)
            if poured is not None and poured > feed_t.limit and f_val > 0:
                # (per cell the term's gradient is about n / 3 times a compliance sensitivity at
                # weight 1, n the cell count: the climb starts under parity)
                w_f = min(max(w_f * 1.5, 30.0 / n_design), 3000.0 / n_design)
            elif poured is not None and poured < 0.5 * feed_t.limit:
                # the pour is sound with room to spare: what the measure still reads is its own
                # over-prediction, and holding the weight would spend stiffness on it
                w_f *= 0.8
            feed_state["w"] = w_f
            # while the pour leaves a cavity the design is still being reshaped: the projection is
            # not sharpened under it until the weight has reached its cap, and a flat compliance
            # is not a stall
            feed_state["over"] = bool(poured is not None and poured > feed_t.limit)
            feed_state["climbing"] = feed_state["over"] and w_f < 3000.0 / n_design
            if w_f > 0:
                cref = feed_state.setdefault("c", comp)
                comp = comp + cref * w_f * f_val / feed_t.limit
                dc = dc + ht @ (printed_grad(am_e, cref * w_f / feed_t.limit * f_grad[ijk]) * dproj_e / hs)
            feed_rec = {"cavityShare": None if poured is None else round(poured, 5),
                        "cavityMeasure": round(f_val, 5), "feedWeight": round(w_f * n_design, 2)}
        dv = ht @ (printed_grad(am_b, wdesign) * dproj / hs)
        blk_rec = None
        if blk is not None:
            # aero.rotors: the blueprint's drag in each rotor's stream, in
            # cells of material that weigh as much on a hovering craft, is
            # part of the budget (aero.BlockageCost). A cell that shadows
            # faster air behind it has a negative cost of its own; in the
            # step it is never cheaper than a tenth of its volume.
            b_cells, b_grad, b_drag = blk.cost(xp, grad=True)
            dv_b = ht @ (printed_grad(am_b, np.where(design, b_grad, 0.0)) * dproj / hs)
            dv = np.maximum(dv + dv_b, 0.1 * dv)
            blk_rec = {"blockageDrag": b_drag, "blockageCells": b_cells}
        if com_cfg is not None or feed_t is not None:
            # the penalty's pull makes some sensitivities positive; shift by
            # s dv as the thermal swing does (the volume multiplier absorbs it)
            live = design & (dv > 1e-12)
            if live.any():
                s_c = float(np.max(dc[live] / dv[live]))
                if s_c > 0:
                    dc = dc - 1.05 * s_c * dv - 1e-12
        if heat is not None and heat.swing_weight > 0:
            # The thermal-cycling swing's stress rises where material is
            # added near the heat (a stiffer part holds more of its own
            # expansion), so its sensitivities are positive there, and the
            # clip below threw that half away: the optimizer never saw why
            # to remove material, and at 45k cells its own swing measure
            # rose 4.3 -> 6.8 with fatigueWeight 0.3. Shifting dc by s dv
            # keeps every sensitivity negative without moving the optimum
            # (the volume multiplier absorbs s: -dc/dv = l becomes
            # s - dc/dv = l + s).
            live = design & (dv > 1e-12)
            if live.any():
                s_sh = float(np.max(dc[live] / dv[live]))
                if s_sh > 0:
                    dc = dc - 1.05 * s_sh * dv - 1e-12
        buck_rec = {}
        if buck is not None and beta >= BUCKLE_BETA:
            # (the eroded design, as stiffness reads it; an inactive reading
            # stands for up to BUCKLING_SKIP iterations, see constrained.py)
            from .constrained import BUCKLING_SKIP

            req = opts.buckling / opts.buckling_modulus
            if buck_last is None or buck_age >= BUCKLING_SKIP or max(v for v, _ in buck_last) > -0.5:
                try:
                    got = buck.evaluate(xe, PENALTY)
                except Exception as exc:  # noqa: BLE001 - a failed eigensolve must not end the run
                    got = None
                    buck_rec["bucklingError"] = f"{type(exc).__name__}: {exc}"
                if got is not None:
                    buck_last = []
                    for mu_ks, dmu, _mus in got:
                        if mu_ks <= 0:
                            continue
                        g_b = ht @ (printed_grad(am_e, req * dmu) * dproj_e / hs)
                        buck_last.append((req * mu_ks - 1.0, g_b))
                    buck_rec["bucklingSeconds"] = buck.last.get("seconds")
                buck_age = 0
            else:
                buck_age += 1
            if buck_last:
                worst_b = max(v for v, _ in buck_last)
                buck_rec["buckling"] = opts.buckling / (worst_b + 1.0) if worst_b > -1.0 else float("inf")
                # a penalty the optimality criteria can take (as the warp
                # limits): compliance + nu c sum(max(g, 0)), nu raised while
                # the design buckles under the limit and eased off over it.
                # (MMA with buckling as a constraint built weaker layouts on
                # the hook, SF 0.3-2.0 at 28-31 g against 3.4 at 32 g.)
                if buck_age == 0:
                    if worst_b > 0:
                        buck_nu = min(max(buck_nu * 1.3, 0.2), 50.0)
                    elif worst_b < -BUCKLE_MMA:
                        buck_nu *= 0.8
                if buck_nu > 0:
                    for v, g in buck_last:
                        if v > -BUCKLE_MMA:
                            dc = dc + buck_nu * comp * g
                buck_rec["bucklingWeight"] = buck_nu
        if modal is not None:
            # The vibration block's floor and bands on the design's own modes
            # (modalopt.py), as a penalty the optimality criteria can take,
            # like buckling's: compliance + nu c sum(max(g, 0)), nu raised
            # while a limit is missed and eased off once it is held.
            from .constrained import BUCKLING_SKIP

            # (a missed limit is read again every other iteration: the eigensolve
            # costs about a compliance solve, and a penalty steers on a gradient one step old)
            if modal_last is None or modal_age >= BUCKLING_SKIP or (
                    max((v for _, v, _ in modal_last), default=1.0) > -0.3 and (modal_age >= 1 or it <= 10)):
                try:
                    modal_last = [(nm_, float(v), ht @ (printed_grad(am_e, g) * dproj_e / hs))
                                  for nm_, v, g in modal.evaluate(xe, PENALTY)]
                    buck_rec["frequencies"] = modal.last.get("frequencies")
                    buck_rec["modalSeconds"] = modal.last.get("seconds")
                except Exception as exc:  # noqa: BLE001 - a failed eigensolve must not end the run
                    buck_rec["modalError"] = f"{type(exc).__name__}: {exc}"
                    modal_last = modal_last or []
                modal_age = 0
            else:
                modal_age += 1
            if modal_last:
                worst_m = max(v for _, v, _ in modal_last)
                if modal_age == 0:
                    if worst_m > 0:
                        modal_nu = min(max(modal_nu * 1.3, 0.2), 50.0)
                    elif worst_m < -0.05:
                        modal_nu *= 0.8
                if modal_nu > 0:
                    for _, v, g in modal_last:
                        if v > -0.05:
                            # (scaled by its own size: a frequency ratio's gradient, not a compliance's)
                            dc = dc + modal_nu * comp * g
                buck_rec["vibration"] = {nm_: round(v, 4) for nm_, v, _ in modal_last}
                buck_rec["vibrationWeight"] = modal_nu
        dc_raw = dc
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

        if measured and not (heat is not None and heat.swing_limit is not None):
            # The step's length is measured, not scheduled. Along the last
            # step dx the Lagrangian's slope was a = g_old . dx (downhill)
            # and is now b = g_new . dx: a parabola through the two puts its
            # minimum at t = a / (a - b) of that step. t < 1 says the step
            # overshot (b > 0: the far side of the valley), and the limit
            # shrinks to what was measured; t > 1 lets it grow, a fifth at a
            # time, back toward opts.move. See docs/CONVERGENCE.md.
            if step_prev is not None:
                dxp, gp, lam_p = step_prev
                a_s = float(gp @ dxp)
                b_s = float((dc + lam_p * dv) @ dxp)
                if a_s < 0:
                    t_s = a_s / (a_s - b_s) if b_s > a_s else np.inf
                    sec = t_s
                    move_k = min(move_k * min(max(t_s, STEP_SHRINK), STEP_GROW), opts.move)
            move = move_k
        else:
            move = opts.move if beta <= 4 else opts.move / 2
        lo_x = np.maximum(0.0, x - move)
        hi_x = np.minimum(1.0, x + move)
        ratio = np.sqrt(-dc / dv)
        l1, l2 = 1e-12, 1e12
        x_new = x
        swing_step = heat is not None and heat.swing_limit is not None and "swing_grad" in heat_state
        if swing_step:
            # Thermal cycling as a constraint (thermal.cycle.
            # fatigueSafetyFactor): MMA on the hot objective with two
            # constraints, volume and the swing's peak stress range. The
            # peak is the p-norm scaled to this iteration's max (Le et al.
            # 2010), its gradient the p-norm's times that scale.
            from .mma import MMA

            free = design & ~pres if void is None else design & ~pres & ~void
            dgs = ht @ (printed_grad(am_e, heat_state["swing_grad"]) * dproj_e / hs)
            dco = dc_raw
            if domain.mirror:
                dgs = _symmetrize(dgs, domain.mirror)
                dco = _symmetrize(dco, domain.mirror)
            dgs = _extrude(dgs, columns, design)
            dco = _extrude(dco, columns, design)
            scale = heat_state["swing_max"] / max(heat_state["swing"], 1e-300)
            g_s = scale * heat_state["swing"] / heat.swing_limit - 1.0
            vol = float(physical(x, beta)[1] @ wdesign)
            if mma_s is None:
                mma_s = MMA(int(free.sum()), 2, move=opts.move)
            mma_s.move = move
            x_new = x.copy()
            with np.errstate(all="ignore"):
                x_new[free] = mma_s.update(x[free], comp / c0, dco[free] / c0,
                                           np.array([vol / target - 1.0, g_s]),
                                           np.array([dv[free] / target, scale * dgs[free] / heat.swing_limit]))
            if not np.all(np.isfinite(x_new)):
                # the MMA subproblem broke down (seen once at 45k cells):
                # take this step with the optimality criteria, restart MMA
                swing_step, mma_s, x_new = False, None, x
            x_new[pres] = 1.0
            if void is not None:
                x_new[void] = 0.0
            if domain.mirror:
                x_new = _symmetrize(x_new, domain.mirror)
            x_new = _extrude(x_new, columns, design)
        if swing_step and "swing_max" in heat_state:
            swing_rec = {"swingPeak": float(heat_state["swing_max"]), "swingLimit": float(heat.swing_limit)}
        n_eval = [0]

        def stepped(lm_):
            n_eval[0] += 1
            xn = np.clip(x * ratio / np.sqrt(lm_), lo_x, hi_x)
            xn[pres] = 1.0
            if void is not None:
                xn[void] = 0.0
            if domain.mirror:
                xn = _symmetrize(xn, domain.mirror)
            xn = _extrude(xn, columns, design)
            if void is not None:
                xn[void] = 0.0
            xpn = physical(xn, beta)[1]
            v = float(xpn @ wdesign)
            if blk is not None:
                v += blk.cost(xpn)[0]
            return xn, v - target

        # The volume multiplier, solved to rounding (VOLUME_RTOL). Stopped at
        # 1 part in 10^4, as a bisection was, the step is a staircase in its
        # inputs: a difference in the last bit of a sensitivity could move
        # every free cell by 5e-5 in one iteration, and two runs were that far
        # apart from then on. The volume falls as the multiplier rises, so:
        # bracket the root outward from the last step's multiplier (it moves
        # little between steps), then false position (Illinois) lands on it.
        # About 8 passes through the filters, where bisecting 24 decades took
        # 20 before the landing.
        f1 = f2 = None
        if not swing_step:
            s_lo, s_hi = np.log(l1), np.log(l2)
            sm = min(max(np.log(lam_last), s_lo), s_hi) if lam_last else 0.0
            x_new, f = stepped(float(np.exp(sm)))
            lm = float(np.exp(sm))
            s1 = s2 = sm
            if f > 0:
                f1 = f
            else:
                f2 = f
            reach = 0.25
            while (f1 is None or f2 is None) and f != 0.0:
                # outward, doubling the reach, until the volume changes side
                # or the multiplier is at the end of its range: then the step
                # limits hold the volume off its target (after a sharpening,
                # or with no room left) and the nearest end is the step
                if f1 is None:
                    if s1 <= s_lo:
                        break
                    s1 = max(s1 - reach, s_lo)
                    sm = s1
                else:
                    if s2 >= s_hi:
                        break
                    s2 = min(s2 + reach, s_hi)
                    sm = s2
                reach *= 4.0
                lm = float(np.exp(sm))
                x_new, f = stepped(lm)
                if f > 0:
                    f1, s1 = f, sm
                else:
                    f2, s2 = f, sm
        if not swing_step and f1 is not None and f2 is not None:
            side = 0
            for _ in range(60):
                if abs(f) <= VOLUME_RTOL * target or f2 == f1 or s2 - s1 <= 4e-16 * max(abs(s1), abs(s2), 1.0):
                    break
                sm = (s1 * f2 - s2 * f1) / (f2 - f1)
                if not s1 < sm < s2:
                    sm = 0.5 * (s1 + s2)
                lm = float(np.exp(sm))
                x_new, f = stepped(lm)
                if f > 0:
                    s1, f1 = sm, f
                    if side == 1:
                        f2 *= 0.5
                    side = 1
                else:
                    s2, f2 = sm, f
                    if side == -1:
                        f1 *= 0.5
                    side = -1
        if not swing_step:
            lam_last = lm
            vol_evals = n_eval[0]
        vol_held = f1 is not None and f2 is not None
        change = float(np.abs(x_new - x)[design].max())
        if measured and not swing_step:
            step_prev = (x_new - x, dc + lm * dv, lm)
        x = x_new
        since_beta += 1

        rec = {"iteration": it, "compliance": comp, "relative": comp / c0 if c0 else 1.0,
               "volume": float(xp[design].mean()), "beta": beta, "change": change,
               "seconds": time.time() - t0, "cgIterations": fe.last_iters,
               "solvedCells": fe.last_cells}
        rec["volumeSolves"] = vol_evals
        if opts.adaptive_move:
            rec["move"] = move
            if sec is not None:
                # where along the last step the measured minimum sat (1 = on it)
                rec["stepFit"] = float(min(sec, 99.0))
        if warp_v is not None:
            rec["warpDistortion"] = warp_v["mean"]
            rec["warpMax"] = warp_v["max"]
            rec["warpLift"] = warp_v["lift"]
            if domain.warp.max_distortion or domain.warp.max_lift:
                rec["warpMultipliers"] = dict(warp_state["lam"])
        if swing_step and "swing_max" in heat_state:
            rec.update(swing_rec)
        rec.update(buck_rec)
        if imp_rec:
            rec["impact"] = imp_rec
        if blk_rec is not None:
            rec.update(blk_rec)
        if com_rec is not None:
            rec.update(com_rec)
        if feed_rec is not None:
            rec.update(feed_rec)
        if excess is not None:
            rec["maxMemberExcess"] = excess
            rec["maxMemberWeight"] = mu
        history.append(rec)
        if on_iter is not None:
            on_iter(rec, xp)

        # (a change held down by the step limit is not a design at rest)
        free_step = not measured or change < 0.999 * move
        if beta < opts.beta_max and feed_state.get("climbing") and beta >= FEED_POUR_BETA:
            pass
        elif beta < opts.beta_max and ((change < opts.tol * 2 and free_step) or since_beta >= _beta_hold(opts, beta)):
            beta = min(beta * 2, opts.beta_max)
            since_beta = 0
            # the projection is twice as steep: so is the problem's curvature
            # along any step, and the step it can take is half as long
            step_prev = None
            move_k = move_k / 2
        elif beta >= opts.beta_max and it >= 3:
            if change < opts.tol and free_step and not feed_state.get("over"):
                converged = True
                break
            w = opts.stall_window
            # a max member still over its limit is not a stall: its weight
            # is still climbing (with the incompatible-mode element the
            # test beam's compliance went flat at iteration 65 with the
            # flanges still 8 mm against a 5 mm limit)
            mm_over = excess is not None and (over > 0.05 or excess > 0.05)
            if scheduled and vol_held:
                if since_beta >= _beta_hold(opts, beta):
                    converged = True
                    break
            elif since_beta >= w and len(history) > w and not mm_over and not feed_state.get("over"):
                old = history[-1 - w]["compliance"]
                if abs(comp - old) <= opts.stall_rtol * abs(old):
                    converged = True
                    break

    if opts.build_direction is not None:
        # the delivered part prints by the rule its schedule ends on, wherever the loop stopped
        am_b.eps = am_e.eps = am_v.eps = overhang_eps(max(beta, opts.beta_max))
    xt, xp = physical(x, beta)
    if opts.erode > 0:
        # Deliver the OPENING of the blueprint: erode, then dilate back by the
        # same amount with the same filter. Everything the optimizer was
        # credited for survives at full size; a sheet thinner than the
        # erosion (the membrane round 3 of the evals left inside the bell
        # crank's frame) does not come back.
        xe = eroded(xt, beta)
        xp = _project((h @ xe) / hs, beta, opts.eta - opts.erode)
        xp = _keep_cores(domain, xp, xe, _project(xt, beta, opts.eta), opts.filter_radius)
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
    fth = None
    fg0 = fg
    if heat is not None:
        fg, fth = _heat_state(heat, xp, fg)
        e = e * heat.g
    loads, layout = expand(_relieved(domain, xp)[0], spreads, fg)
    us = fe.solve(e, xp, us, rtol=1e-6, loads=loads)
    if opts.prune:
        if fth is not None:
            # dead ends are judged by what the loads put through them. The
            # displacements under heat carry the free expansion too, whose
            # strain is no stress: a hot seat growing freely read thousands
            # of times the energy of the members that carry the load, and
            # the pruning took the whole design for dead (a free part with
            # a 80 C motor seat came back empty). So: the loads alone, at
            # the warm moduli.
            loads_p, _ = expand(_relieved(domain, xp)[0], spreads, fg0)
            us_p = fe.solve(e, xp, [None] * len(loads_p), rtol=1e-6, loads=loads_p)
            xp, _ = _prune_dead(fe, domain, xp, us_p, loads_p, design, pres, heat.g)
        else:
            xp, us = _prune_dead(fe, domain, xp, us, loads, design, pres, None if heat is None else heat.g)
        if opts.access is not None:
            # pruning a dead end can leave a void under material: refill
            xq = _machined(domain, xp, opts.access)
            if (xq != xp).any():
                xp = xq
                us = fe.solve(E_MIN + xp ** PENALTY * (1 - E_MIN), xp, us, rtol=1e-6, loads=loads)
    case_c, case_d, comp = [], [], 0.0
    energy = np.zeros(n)
    stiff = E_MIN + xp ** PENALTY * (1 - E_MIN)
    if heat is not None:
        if opts.prune:
            # pruning moved the design: its temperature and load with it
            fg = gravity_force(mesh, xp, domain.gravity, domain.fixed) if domain.gravity is not None else None
            fg, fth = _heat_state(heat, xp, fg)
            loads, layout = expand(_relieved(domain, xp)[0], spreads, fg)
            us = fe.solve(stiff * heat.g, xp, us, rtol=1e-6, loads=loads)
        comp, case_c, _, heat_d = heat.sensitivities(fe, xp, loads, us, fth, weights, domain.gravity, heat_state,
                                                     layout, spreads)
        stiff = stiff * heat.g
    for ci, wc in enumerate(collapse(loads, layout, us, spreads)):
        d = wc.direction if heat is None else heat_d[ci]
        case_d.append(None if d is None else [float(v) for v in d])
        if heat is None:
            case_c.append(wc.compliance)
            comp += (weights0 if bind is not None else weights)[ci] * wc.compliance
        energy += weights[ci] * stiff * sum(m * fe.strain_energy(v) for m, v in wc.energy)
    warp = _warp_report(domain, xp) if domain.warp is not None else None
    return SimpResult(x=x, xphys=xp, compliance=comp, case_compliance=case_c,
                      volume_fraction=float(xp[design].mean()), iterations=it,
                      converged=converged, seconds=time.time() - t0, history=history,
                      case_direction=case_d, energy=energy / mesh.pitch ** 3, warp=warp)
