"""
Minimum mass subject to stress and deflection limits, solved directly with
MMA: the formulation commercial generative design uses ("as light as
possible, safety factor at least 3, deflection at most 0.5 mm").

The volume-constrained compliance problem (simp.optimize) answers a
different question and needs an outer search over volume fractions to meet
a safety factor, several full optimizations deep. Here the limits are the
constraints, so one run lands on (or near) them, and a short correction
loop in study.py closes the remaining gap against the verified mesh.

- Stress: relaxed von Mises per cell, sigma_e = x_e^q * vm(C B u_e) with
  q = 1 (qp-relaxation), aggregated with a P-mean (P = 8) and the
  adaptive normalization of Le et al. 2010, which rescales the P-mean to
  the true maximum from the previous iteration. Cells touching loads and
  supports are left out, as in verification. One adjoint solve per case.
- Deflection: compliance over the load's magnitude is the displacement at
  the load along its direction; self-adjoint, no extra solve.
- Everything else (filter, robust erosion, projection continuation,
  symmetry, extrusion, the AM overhang filter for a build direction, the
  reduced solve) is shared with simp.optimize.
"""
from __future__ import annotations

import os
import time

import numpy as np

from .domain import Domain
from .fem import elastic_c, hex8_b, hex8_b_center
from .manufacture import tie_groups
from .mma import MMA
from . import checkpoint as _checkpoint
from .simp import (E_MIN, PENALTY, SimpOptions, SimpResult, _Analysis, _dproject, _extrude, _pad,
                   _project, _symmetrize, density_filter)

__all__ = ["Limits", "optimize_constrained"]

# q = 0.5 (Le et al.'s choice) let near-void cells next to a member, strained
# by the member's motion across them, read 100-1000 times the limit: the
# aggregate chased those cells, and MMA either emptied the part or
# oscillated on the L-bracket benchmark. At q = 1 they fade with their
# density and the run converges onto the limit.
_Q = 1.0
_P = 8.0
# iterations an inactive buckling constraint (load factor over twice the one
# asked) keeps its last reading
BUCKLING_SKIP = 3
# ... one with a third to spare (BUCKLING_NEAR or under): every other
# iteration, where the printed hook's sat for most of a run at 10 s a reading
BUCKLING_NEAR = -0.25
# ... and one at BUCKLING_FAR or under (five times the load factor asked),
# unless the projection sharpened or the volume moved by BUCKLING_VOLUME_STEP
# since: a strut's Euler load goes with its section squared, so a design
# cannot shed that margin in fewer steps than that
BUCKLING_FAR = -0.8
BUCKLING_SKIP_FAR = 7
BUCKLING_VOLUME_STEP = 0.1
# The run is over when the lightest design inside the limits, at the
# sharpest projection, has gained less than this (relative) in
# `stall_window` iterations. The rule before it called a volume that moved
# under 0.002 of the box a stall: 4% of a part that fills 5% of its box,
# measured over a window that reached back across the last sharpening,
# where the volume steps the other way. The wrist yoke's first part stopped
# at iteration 35 with 5-7% of its stress allowance unused, 19.0 g; run on
# at the same limits it is 15.2 g with members 2.6 mm across against 1 mm.
STALL_GAIN = 5e-3
# The buckling constraint reads the eroded, grey design, as stress and
# stiffness do, and that design buckles far sooner than the part delivered:
# its members are thinner by the erosion and softer by the penalization, and
# a strut's Euler load goes with its thickness to the fourth power. The
# wrist yoke's optimizer read 2.6 on a part that verified at 15.9, the
# printed hook's parts came out at 44 g holding a limit they passed four
# times over (the search's: 32 g), and a limit that looked active was
# re-read every iteration, 10 s of an iteration's 12. So the reading is
# scaled to the design that will be delivered: the same model on the crisp
# blueprint (xp > 0.5), read once the projection has reached BUCKLING_CRISP_BETA
# and again at each sharpening (and every BUCKLING_CRISP_EVERY readings),
# gives the ratio crisp / eroded (never under 1, at most BUCKLING_CRISP_MAX,
# averaged with the last so the constraint does not jump). What is left
# between that and the verified part is the study's own correction
# (study._BUCKLING_START, then each run's ratio). LOADPATH_BUCKLING_CRISP=0
# reads the eroded design alone, as before.
BUCKLING_CRISP = os.environ.get("LOADPATH_BUCKLING_CRISP", "1") != "0"
BUCKLING_CRISP_BETA = 4.0
BUCKLING_CRISP_EVERY = 12
BUCKLING_CRISP_MAX = 8.0
STRESS_POINTS = "corners"     # where cell stress is sampled: "corners" or "center"
# the node (fem.CORNERS order: x, then y, then z) each corner sample sits on, in the order they are taken (x outermost)
_CORNER_OF = (0, 4, 2, 6, 1, 5, 3, 7)
_VM = np.array([
    [1.0, -0.5, -0.5, 0, 0, 0],
    [-0.5, 1.0, -0.5, 0, 0, 0],
    [-0.5, -0.5, 1.0, 0, 0, 0],
    [0, 0, 0, 3.0, 0, 0],
    [0, 0, 0, 0, 3.0, 0],
    [0, 0, 0, 0, 0, 3.0],
])


class Limits:
    def __init__(self, stress: float | None, deflection: float | None, modulus: float,
                 buckling: float | None = None, bond: tuple | None = None,
                 contact_band: float | None = None, radius: float | None = None):
        self.stress = stress            # MPa, allowable peak von Mises
        self.deflection = deflection    # mm at the loads
        self.modulus = modulus          # MPa, to turn unit-E displacements into mm
        self.buckling = buckling        # smallest load factor at which the design may buckle
        # A part built in layers: (the bond's tensile strength over the
        # in-plane strength, the unit normal the layers stack along, the
        # bond's tensile over its shear strength). The stress limit then
        # holds the layer bond as verification reads it (failure.bond_stress:
        # the tension pulling the layers apart and the shear sliding them,
        # against the bond's strength) beside von Mises, each cell judged on
        # whichever is nearer its limit.
        self.bond = bond
        # Stress is left out this far (mm) from a load or a support, node by
        # node, as the checks leave it out (study.contact_band): nearer, the
        # peak is the edge of an idealized rigid patch, not the part, and a
        # limit held there spends material the checks never ask for. None:
        # only the cells the loads and supports touch are left out.
        self.contact_band = contact_band
        # mm: stress is read as the checks read it (verify._MaterialAverage,
        # crosscheck._averaged_stress), the tensor averaged over the material
        # within this distance of each node, then judged (_Averaged). None:
        # the peak of each cell's corners, as before.
        self.radius = radius


class _Averaged:
    """The optimizer's stress reading made the checks' own: at each node, the
    stress tensor averaged over the material within `radius` (the material's
    length, failure.MATERIAL_LENGTH), then von Mises and the layer bond read
    from that average.

    The corner-point reading it replaces is the mesh's and not the part's: a
    cell corner on a staircase or at a re-entrant step reads a spike the
    meshed part does not have, and as the design moves, the spikes move with
    it. Holding a limit on them, the printed hook's check fell from 2.56 to
    2.04 over 120 iterations while the optimizer's own reading held. On the
    averaged reading the optimizer holds what the voxel and tetrahedral
    checks judge (docs/CONVERGENCE.md).

    The material is sampled as verify._MaterialAverage samples it: each
    cell's own stress at its eight quarter points while the ball is under
    three cells across, at its centre on finer grids, each sample weighted by
    the cell's (eroded) density, so a grey or void cell carries what material
    it has. A node's reading is then relaxed by the density of the material
    round it (its density-weighted mean density, 1 at the surface of a solid
    member): a near-void node strained by a member's motion fades, as the
    corner reading's x^q did. Built once per run (the pairs of nodes and
    samples do not depend on the design)."""

    def __init__(self, mesh, nu: float, radius: float, readable):
        import scipy.sparse as sp
        from scipy.spatial import cKDTree

        from .fem import CORNERS

        pitch = mesh.pitch
        rho = radius / pitch
        nat = (CORNERS - 0.5) if rho <= 1.5 else np.zeros((1, 3))      # natural coordinates
        self.ns = len(nat)
        c0 = elastic_c(1.0, nu)
        self.cbs = [c0 @ hex8_b(a, b, cz, nu) / pitch for a, b, cz in nat]
        ne = mesh.n_elem
        xs = ((mesh.elem_ijk[:, None, :] + 0.5 + 0.5 * nat[None, :, :]) * pitch).reshape(-1, 3)
        nodes = np.flatnonzero(readable)
        pairs = cKDTree(mesh.node_ijk[nodes] * pitch).sparse_distance_matrix(
            cKDTree(xs), radius, output_type="coo_matrix")
        self.nodes = nodes
        # (read nodes, samples), sample s = cell s // ns, point s % ns
        self.ball = sp.csr_matrix((np.ones(len(pairs.row)), (pairs.row, pairs.col)), shape=(len(nodes), ne * self.ns))
        self.ballT = self.ball.T.tocsr()
        self.n_elem = ne
        self.w0 = pitch ** 3 / self.ns
        self.radius = float(radius)

    def cell_sum(self, per_sample):
        return per_sample.reshape(self.n_elem, self.ns).sum(axis=1)


def _normalize(prev: float, smax: float, pm: float, it: int) -> float:
    """Le et al.'s adaptive normalization: the factor that brings the P-mean
    up to the peak, from this reading, averaged with the last one."""
    k = smax / max(pm, 1e-30)
    return 0.5 * (prev + k) if it > 1 else k


def _bond_form(normal):
    """The layer bond's stress in Voigt terms (sx, sy, sz, txy, tyz, tzx), for
    layers stacked along unit `normal`: (m, q) with the stress normal to the
    layers m . sigma and the squared shear on them sigma . q . sigma."""
    n = np.asarray(normal, float)
    n = n / np.linalg.norm(n)
    t = np.array([[n[0], 0, 0, n[1], 0, n[2]],
                  [0, n[1], 0, n[0], n[2], 0],
                  [0, 0, n[2], 0, n[1], n[0]]])           # traction on the layer plane
    m = n @ t
    return m, t.T @ t - np.outer(m, m)


def optimize_constrained(domain: Domain, nu: float, opts: SimpOptions, limits: Limits,
                         on_iter=None, should_stop=None) -> SimpResult:
    """Checkpointed as simp._optimize is (checkpoint.py)."""
    ck = _checkpoint.Call.open("constrained", domain, nu, opts, limits)
    if ck is None:
        return _constrained_run(domain, nu, opts, limits, on_iter, should_stop, None)
    res = ck.done()
    if res is not None:
        return res
    res = _constrained_run(domain, nu, opts, limits, on_iter, should_stop, ck)
    ck.finish(res)
    return res


# the loop's state between two iterations (checkpoint.Call)
_LOOP_STATE = ("x", "mma", "move_cap", "worst_prev", "kappa", "us", "lams", "beta", "since_beta", "history",
               "buck_last", "buck_age", "modal_last", "modal_age", "buck_beta", "buck_vol", "crisp_ratio", "crisp_beta",
               "crisp_age", "best", "trail")


def _constrained_run(domain: Domain, nu: float, opts: SimpOptions, limits: Limits, on_iter, should_stop,
                     ck) -> SimpResult:
    t0 = time.time()
    mesh = domain.mesh
    n = mesh.n_elem
    pres = domain.preserve
    design = ~pres
    fe = _Analysis(domain, nu, opts.solve_rtol)
    if opts.access is not None or opts.max_member:
        raise ValueError("optimize_constrained does not take milling, casting or maxMember: use the search")
    columns = tie_groups(domain, opts.extrusion, opts.pattern)
    h, hs = density_filter(domain, opts.filter_radius, pad=_pad(opts))
    ht = h.T.tocsr()
    eta_e = opts.eta + opts.erode

    blk = getattr(domain, "blockage", None)
    cases = len(domain.loads)
    force_mag = [float(np.linalg.norm(f.reshape(-1, 3).sum(axis=0))) for f in domain.loads]
    # the vibration block's inertial cases only steer stiffness (domain._inertial_cases)
    real = [c for c in range(cases) if c not in set(getattr(domain, "inertial_cases", ()))]
    defl_cases = [c for c in real if limits.deflection is not None and force_mag[c] > 0]
    stress_cases = list(real) if limits.stress is not None else []
    buck_cases = [c for c in real if limits.buckling is not None and np.any(domain.loads[c])]
    m = len(defl_cases) + len(stress_cases) + len(buck_cases)
    if m == len(buck_cases):
        raise ValueError("optimize_constrained needs a stress or deflection limit")
    buck = None
    if buck_cases:
        from .bucklingopt import BucklingModel

        buck = BucklingModel(domain, nu)
    buck_last, buck_age = None, 0
    buck_beta, buck_vol = None, None
    # crisp / eroded reading per buckling case (BUCKLING_CRISP), its own
    # model (the crisp design's modes warm-start from the crisp design's)
    buck_crisp = BucklingModel(domain, nu) if buck is not None and BUCKLING_CRISP else None
    crisp_ratio = [1.0] * len(buck_cases)
    crisp_beta, crisp_age = None, 0

    # Stress is not counted in cells where loads and supports are applied.
    contact = (domain.support_nodes | domain.load_nodes)[mesh.enode].any(axis=1)
    counted = ~contact
    # ... or, with limits.contact_band, every sample point within that
    # distance of a loaded or supported node (pt_counted: per cell, per corner)
    pt_counted = None
    if limits.contact_band and STRESS_POINTS == "corners" and (domain.support_nodes | domain.load_nodes).any():
        from scipy.spatial import cKDTree

        held = domain.support_nodes | domain.load_nodes
        xyz = mesh.node_ijk * mesh.pitch
        near = cKDTree(xyz[held]).query(xyz, k=1, distance_upper_bound=float(limits.contact_band))[0] < np.inf
        pt_counted = ~near[mesh.enode]                       # (cells, 8), in fem.CORNERS order
        counted = pt_counted.any(axis=1)
    # Stress per unit-E displacement (6x24) at each sample point: the cell's
    # 8 corners (STRESS_POINTS), with the element's own strain there
    # (fem.hex8_b: the incompatible-mode bubbles carry part of a member's
    # bending). Verification reads the surface of the meshed part, where a
    # bent member peaks; the cell centre read it 1.5-3.4x low on the eval
    # cases (MBB, bell crank, hook, motor mount, wrist yoke at 15k cells),
    # the corners 1.2-2x.
    bond = limits.bond if limits.stress is not None else None
    avg = None
    if limits.stress is not None and limits.radius:
        # read at the nodes the checks read: outside the contact band (or, with
        # no band, off the cells the loads and supports touch)
        if pt_counted is not None:
            readable = np.zeros(mesh.n_nodes, bool)
            readable[mesh.enode[pt_counted]] = True
            readable[mesh.enode[~pt_counted]] = False
        else:
            readable = np.ones(mesh.n_nodes, bool)
            readable[mesh.enode[contact].ravel()] = False
        avg = _Averaged(mesh, nu, float(limits.radius), readable)
    if bond is not None:
        bond_ratio = float(bond[0])
        bond_m, bond_q = _bond_form(bond[1])
        bond_k2 = float(bond[2]) ** 2
    c0 = elastic_c(1.0, nu)
    if STRESS_POINTS == "corners":
        cbs = [c0 @ hex8_b(a, b, cz, nu) / mesh.pitch for a in (-1.0, 1.0) for b in (-1.0, 1.0) for cz in (-1.0, 1.0)]
    else:
        cbs = [c0 @ hex8_b_center() / mesh.pitch]

    # Printed without supports along build_direction: the AM filter (am.py)
    # after each projection, as simp.optimize applies it.
    if opts.build_direction is not None:
        from .am import AMFilter

        pinned = domain.full_grid(pres.astype(float)) > 0.5
        am_b = AMFilter(domain.dims, opts.build_direction, pinned)   # blueprint
        am_e = AMFilter(domain.dims, opts.build_direction, pinned)   # eroded
        ijk = tuple(mesh.elem_ijk.T)

        def printed(am, v):
            return am.forward(domain.full_grid(v))[ijk]

        def printed_grad(am, g):
            return am.backward(domain.full_grid(g))[ijk]
    else:
        am_b = am_e = None

        def printed(am, v):
            return v

        def printed_grad(am, g):
            return g

    free = np.flatnonzero(design)
    x = np.full(n, opts.volume_fraction)
    if opts.x0 is not None:
        x = np.clip(np.asarray(opts.x0, float), 0.0, 1.0)
    x[pres] = 1.0
    modal = getattr(domain, "modal", None)
    modal_last, modal_age = None, 0
    # (a fixed number of slots: which side of a band a mode is held on can change as the design does)
    modal_slots = modal.slots() if modal is not None else []
    m += len(modal_slots)
    mma = MMA(len(free), m, xmin=0.0, xmax=1.0, move=min(opts.move, 0.1))
    move_cap = mma.move
    worst_prev = np.inf
    kappa = [1.0] * cases
    us = [None] * cases
    lams = [None] * cases
    beta = float(opts.beta_start)
    since_beta = 0
    history = []
    converged = False
    it = 0
    # the lightest design inside the limits at the sharpest projection
    # (its variables, its record), and its volume after each iteration there
    best = None
    trail = []

    it0 = 0
    if ck is not None:
        ck.add_helpers(buck=buck, buck_crisp=buck_crisp, am_b=am_b, am_e=am_e)
        st = ck.restore()
        if st is not None:
            (x, mma, move_cap, worst_prev, kappa, us, lams, beta, since_beta, history, buck_last, buck_age,
             modal_last, modal_age, buck_beta, buck_vol, crisp_ratio, crisp_beta, crisp_age, best,
             trail) = (st[k] for k in _LOOP_STATE)
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
                x, mma, move_cap, worst_prev, kappa, us, lams, beta, since_beta, history, buck_last, buck_age,
                modal_last, modal_age, buck_beta, buck_vol, crisp_ratio, crisp_beta, crisp_age, best, trail)))})
        if should_stop is not None and should_stop():
            break
        x_now = x
        xt = (h @ x) / hs
        xp = _project(xt, beta, opts.eta)
        xp[pres] = 1.0
        xp = printed(am_b, xp)
        xe = _project(xt, beta, eta_e)
        xe[pres] = 1.0
        xe = printed(am_e, xe)
        dpb = _dproject(xt, beta, opts.eta)
        dpb[pres] = 0.0
        dpe = _dproject(xt, beta, eta_e)
        dpe[pres] = 0.0

        e = E_MIN + xe ** PENALTY * (1 - E_MIN)
        de = PENALTY * xe ** (PENALTY - 1) * (1 - E_MIN)
        us = fe.solve(e, xe, us, loads=domain.loads)

        def to_x(g_xe):
            return ht @ (printed_grad(am_e, g_xe) * dpe / hs)

        cons, grads, record = [], [], {}
        for c in defl_cases:
            comp = float(domain.loads[c] @ us[c])
            defl = comp / (force_mag[c] * limits.modulus)
            ce = fe.strain_energy(us[c])
            g_xe = -de * ce / (force_mag[c] * limits.modulus)
            cons.append(defl / limits.deflection - 1.0)
            grads.append(to_x(g_xe) / limits.deflection)
            record[f"deflection.{domain.case_names[c]}"] = defl

        if stress_cases and avg is not None:
            adj_rhs, partials = [], []
            bond_limit = limits.stress * bond_ratio if bond is not None else None
            nc = max(len(avg.nodes), 1) * (2 if bond is not None else 1)
            v = np.repeat(avg.w0 * np.maximum(xe, 0.0), avg.ns)              # sample weights: material
            dn = avg.ball @ v
            on = dn > 0
            dinv = np.where(on, 1.0 / np.where(on, dn, 1.0), 0.0)
            xs_ = np.repeat(xe, avg.ns)
            rho_n = (avg.ball @ (v * xs_)) * dinv                         # density round the node
            for c in stress_cases:
                moved = np.flatnonzero(us[c].reshape(-1, 3).any(axis=1)[mesh.enode].any(axis=1))
                edof = mesh.edof[moved]
                ue = us[c][edof]
                ss = np.zeros((n, avg.ns, 6))
                for k, cbk in enumerate(avg.cbs):
                    ss[moved, k] = ue @ cbk.T
                ss = ss.reshape(-1, 6)
                sig = (avg.ball @ (v[:, None] * ss)) * dinv[:, None]      # (read nodes, 6) averaged
                parts = []
                sv = sig @ _VM
                vm = np.sqrt(np.maximum(np.einsum("ij,ij->i", sv, sig), 1e-30))
                parts.append((sv, vm, limits.stress))
                if bond is not None:
                    sn = np.maximum(sig @ bond_m, 0.0)
                    sb = sn[:, None] * bond_m + bond_k2 * (sig @ bond_q)
                    eq = np.sqrt(np.maximum(np.einsum("ij,ij->i", sb, sig), 1e-30))
                    parts.append((sb, eq, bond_limit))
                srs = [rho_n ** _Q * f / lim for _, f, lim in parts]
                smax = max((float(sr.max()) for sr in srs if len(sr)), default=0.0)
                sc = max(smax, 1e-30)
                pm = sc * (sum(float(np.sum((sr / sc) ** _P)) for sr in srs) / nc) ** (1.0 / _P)
                kappa[c] = _normalize(kappa[c], smax, pm, it)
                g = kappa[c] * pm
                gsig = np.zeros_like(sig)          # dg / d(averaged tensor)
                grho = np.zeros(len(sig))          # dg / d(node density)
                for (sh, f, lim), sr in zip(parts, srs):
                    a = kappa[c] * (sr / max(pm, 1e-30)) ** (_P - 1) / nc
                    gsig += (a * rho_n ** _Q / (f * lim))[:, None] * sh
                    grho += a * _Q * np.maximum(rho_n, 1e-12) ** (_Q - 1) * f / lim
                # back to the samples: d sig_n / d s_s = ball_ns v_s / D_n
                hs_ = avg.ballT @ (gsig * dinv[:, None])                   # (samples, 6), per unit weight
                h_s = v[:, None] * hs_
                # explicit, through the weights: d sig_n / d x(s) = ball w0 (s_s - sig_n) / D_n,
                # and d rho_n / d x(s) = ball w0 (2 x_s - rho_n) / D_n
                t_sig = np.einsum("ij,ij->i", ss, hs_) \
                    - avg.ballT @ (np.einsum("ij,ij->i", gsig, sig) * dinv)
                t_rho = 2.0 * xs_ * (avg.ballT @ (grho * dinv)) - avg.ballT @ (grho * rho_n * dinv)
                g_xe = avg.w0 * avg.cell_sum(t_sig + t_rho)
                h_s = h_s.reshape(n, avg.ns, 6)
                rhs = np.zeros(mesh.n_dof)
                for k, cbk in enumerate(avg.cbs):
                    rhs += np.bincount(edof.ravel(), weights=(h_s[moved, k] @ cbk).ravel(), minlength=mesh.n_dof)
                rhs[domain.fixed] = 0.0
                adj_rhs.append(rhs)
                partials.append((c, g, g_xe, smax, moved))
            lams = fe.resolve(adj_rhs, [l for l in lams] if lams[0] is not None and len(lams) == len(adj_rhs) else None)
            for (c, g, g_xe, smax, moved), lam in zip(partials, lams):
                edof = mesh.edof[moved]
                g_xe[moved] -= de[moved] * np.einsum("ij,ij->i", lam[edof] @ fe.ke, us[c][edof])
                cons.append(g - 1.0)
                grads.append(to_x(g_xe))
                record[f"stress.{domain.case_names[c]}"] = smax * limits.stress
        elif stress_cases:
            adj_rhs, partials = [], []
            for c in stress_cases:
                # Only over the cells that moved. The solve leaves every dof
                # outside the live cells at exactly zero (simp._Analysis),
                # and a cell with no displacement has no stress, no share
                # of the p-norm and no term in either gradient: on a
                # density design that is most of the grid (the wrist yoke:
                # 8-30k live cells of 119k), and these passes, eight points
                # a cell for every case, were a quarter of an iteration.
                moved = np.flatnonzero(us[c].reshape(-1, 3).any(axis=1)[mesh.enode].any(axis=1))
                act = moved[counted[moved]]
                edof = mesh.edof[act]
                ue = us[c][edof]                             # (cells, 24)
                xa = xe[act]
                nc = (max(int(counted.sum()), 1) * len(cbs) if pt_counted is None else max(int(pt_counted.sum()), 1)) \
                    * (2 if bond is not None else 1)
                bond_limit = limits.stress * bond_ratio if bond is not None else None
                parts = []
                for kp, cbk in enumerate(cbs):
                    sig = ue @ cbk.T                         # (cells, 6) solid stress, MPa
                    if pt_counted is not None:
                        # a point inside the contact band reads no stress: no share of the
                        # aggregate and no term in either gradient
                        sig = sig * pt_counted[act, _CORNER_OF[kp]][:, None]
                    sv = sig @ _VM
                    vm = np.sqrt(np.maximum(np.einsum("ij,ij->i", sv, sig), 1e-30))
                    parts.append((cbk, sv, vm, limits.stress, xa ** _Q * vm / limits.stress))
                    if bond is not None:
                        # the layer bond there, as an equivalent stress whose
                        # square is quadratic in sigma on each side of zero
                        # tension (so the same adjoint terms carry it)
                        sn = np.maximum(sig @ bond_m, 0.0)
                        sb = sn[:, None] * bond_m + bond_k2 * (sig @ bond_q)
                        eq = np.sqrt(np.maximum(np.einsum("ij,ij->i", sb, sig), 1e-30))
                        parts.append((cbk, sb, eq, bond_limit, xa ** _Q * eq / bond_limit))
                smax = max((float(sr.max()) for *_, sr in parts if len(sr)), default=0.0)
                # the p-norm scaled by the peak, so a design that lets go
                # (stress 1e40) does not overflow sr ** P to inf and hand
                # MMA a NaN step (the robot-arm upper arm)
                sc = max(smax, 1e-30)
                pm = sc * (sum(float(np.sum((sr / sc) ** _P)) for *_, sr in parts) / nc) ** (1.0 / _P)
                kappa[c] = _normalize(kappa[c], smax, pm, it)
                g = kappa[c] * pm
                g_xe = np.zeros(n)
                rhs = np.zeros(mesh.n_dof)
                for cbk, sv, vm, lim, srel in parts:
                    dg_ds = kappa[c] * (srel / max(pm, 1e-30)) ** (_P - 1) / nc
                    # explicit: d s / d xe at fixed u
                    g_xe[act] += dg_ds * (_Q * np.maximum(xa, 1e-12) ** (_Q - 1) * vm / lim)
                    # adjoint right-hand side: dg/du
                    coef = dg_ds * (xa ** _Q / (vm * lim))                                  # (cells,)
                    dvm_du = sv @ cbk                                                       # d(vm^2)/2du
                    rhs += np.bincount(edof.ravel(), weights=(coef[:, None] * dvm_du).ravel(),
                                       minlength=mesh.n_dof)
                rhs[domain.fixed] = 0.0
                adj_rhs.append(rhs)
                partials.append((c, g, g_xe, smax, moved))
            lams = fe.resolve(adj_rhs, [l for l in lams] if lams[0] is not None and len(lams) == len(adj_rhs) else None)
            for (c, g, g_xe, smax, moved), lam in zip(partials, lams):
                edof = mesh.edof[moved]
                g_xe[moved] -= de[moved] * np.einsum("ij,ij->i", lam[edof] @ fe.ke, us[c][edof])
                cons.append(g - 1.0)
                grads.append(to_x(g_xe))
                record[f"stress.{domain.case_names[c]}"] = smax * limits.stress

        if buck is not None:
            # linear buckling of the eroded design, as stress and stiffness
            # read it (bucklingopt.py): lambda_min >= limits.buckling as
            # limits.buckling * mu_ks - 1 <= 0
            # (at unit modulus the load factor is lambda / E)
            # Away from the limit the last reading stands: one iteration
            # with a third to spare, BUCKLING_SKIP at twice the load factor
            # asked, BUCKLING_SKIP_FAR at five times. The eigensolve is most
            # of an iteration on a big grid, and an inactive constraint's
            # stale value and gradient do not steer. From twice on the
            # reading is also taken loosely (bucklingopt.evaluate's `relax`):
            # its value to a percent, and past BUCKLING_FAR no gradient.
            req = limits.buckling / limits.modulus
            worst_b = max(v for v, _ in buck_last) if buck_last is not None else 0.0
            vol_now = float(xp[design].mean())
            hold = 0 if worst_b > BUCKLING_NEAR else 1 if worst_b > -0.5 else BUCKLING_SKIP
            if worst_b <= BUCKLING_FAR and buck_beta == beta \
                    and abs(vol_now - buck_vol) <= BUCKLING_VOLUME_STEP * buck_vol:
                hold = BUCKLING_SKIP_FAR
            if buck_last is None or buck_age >= hold:
                # (v = req / cr * mu - 1: the mu at v = -0.5 and at BUCKLING_FAR)
                relax = [(0.5 * cr / req, (1.0 + BUCKLING_FAR) * cr / req) for cr in crisp_ratio]
                try:
                    got = buck.evaluate(xe, PENALTY, buck_cases, relax=relax)
                except Exception as exc:  # noqa: BLE001 - a failed eigensolve must not end the run
                    got = None
                    record["bucklingError"] = f"{type(exc).__name__}: {exc}"
                if got is not None:
                    buck_beta, buck_vol = beta, vol_now
                    crisp_age += 1
                    if buck_crisp is not None and beta >= BUCKLING_CRISP_BETA \
                            and (crisp_beta != beta or crisp_age >= BUCKLING_CRISP_EVERY):
                        xc = (xp > 0.5).astype(float)
                        xc[pres] = 1.0
                        buck_crisp.seed(buck)
                        try:
                            # (a reading alone: the loose tolerance whatever it reads)
                            ref = buck_crisp.evaluate(xc, PENALTY, buck_cases, gradient=False,
                                                      relax=[(np.inf, 0.0)] * len(buck_cases))
                        except Exception:  # noqa: BLE001 - no reading: the last ratio stands
                            ref = None
                        if ref is not None:
                            for j, ((mu_e, _, _), (mu_c, _, _)) in enumerate(zip(got, ref)):
                                # (a crisp design that let go of the load reads 0: no reading)
                                if mu_e > 0 and mu_c > 0:
                                    r = float(np.clip(mu_e / mu_c, 1.0, BUCKLING_CRISP_MAX))
                                    crisp_ratio[j] = r if crisp_beta is None else 0.5 * (crisp_ratio[j] + r)
                            crisp_beta, crisp_age = beta, 0
                            record["bucklingCrispSeconds"] = buck_crisp.last.get("seconds")
                    buck_last = [(req / cr * mu_ks - 1.0, to_x(req / cr * dmu))
                                 for (mu_ks, dmu, _mus), cr in zip(got, crisp_ratio)]
                    record["bucklingSeconds"] = buck.last.get("seconds")
                    if buck_crisp is not None:
                        record["bucklingCrispRatio"] = [float(v) for v in crisp_ratio]
                elif buck_last is None:
                    # nothing read yet: inactive this iteration (verify still checks the part)
                    buck_last = [(-1.0, np.zeros(n)) for _ in buck_cases]
                buck_age = 0
            else:
                buck_age += 1
            for c, (v, g) in zip(buck_cases, buck_last):
                cons.append(v)
                grads.append(g)
                record[f"buckling.{domain.case_names[c]}"] = limits.buckling / (v + 1.0) if v > -1.0 else float("inf")

        if modal is not None:
            # the vibration block's floor and bands on the design's own modes
            # (modalopt.py); a reading well inside its limits stands for up
            # to BUCKLING_SKIP iterations, as buckling's does
            if modal_last is None or modal_age >= BUCKLING_SKIP or max(v for _, v, _ in modal_last) > -0.3:
                try:
                    modal_last = [(nm_, float(v), to_x(g)) for nm_, v, g in modal.evaluate(xe, PENALTY)]
                    record["frequencies"] = modal.last.get("frequencies")
                    record["modalSeconds"] = modal.last.get("seconds")
                except Exception as exc:  # noqa: BLE001 - a failed eigensolve must not end the run
                    record["modalError"] = f"{type(exc).__name__}: {exc}"
                    modal_last = modal_last or []
                modal_age = 0
            else:
                modal_age += 1
            got_m = {nm_: (v, g) for nm_, v, g in modal_last}
            for nm_ in modal_slots:
                v, g = got_m.get(nm_, (-1.0, np.zeros(n)))
                cons.append(v)
                grads.append(g)
                if nm_ in got_m:
                    record[f"vibration.{nm_}"] = v

        f0 = 10.0 * float(xp[design].mean())
        g0 = 10.0 * design / design.sum()
        if blk is not None:
            # aero.rotors: the mass minimized is the part's plus what its drag
            # in the rotors' streams weighs on a hovering craft (aero.BlockageCost)
            b_cells, b_grad, b_drag = blk.cost(xp, grad=True)
            f0 += 10.0 * b_cells / design.sum()
            g0 = g0 + 10.0 * np.where(design, b_grad, 0.0) / design.sum()
            record["blockageDrag"] = b_drag
        df0 = ht @ (printed_grad(am_b, g0) * dpb / hs)
        if domain.mirror:
            df0 = _symmetrize(df0, domain.mirror)
            grads = [_symmetrize(g, domain.mirror) for g in grads]
        df0 = _extrude(df0, columns, design)
        grads = [_extrude(g, columns, design) for g in grads]

        worst = max(cons)
        if worst > 0.02 and worst > worst_prev:
            # Getting less feasible: the convex model promised a step that
            # cut volume and stress together, and the stress went up (from a
            # grey start MMA otherwise ran the L-bracket down to nothing).
            # Halve the step and pull the asymptotes in, a more conservative
            # model, until the design is back inside the limits.
            mma.move = max(0.02, 0.5 * mma.move)
            mma.tighten(0.5)
        elif worst <= 0.02:
            mma.move = min(move_cap, 1.2 * mma.move)
        worst_prev = worst
        xnew_free = mma.update(x[free], f0, df0[free], np.array(cons), np.array([g[free] for g in grads]))
        xnew = x.copy()
        xnew[free] = xnew_free
        if domain.mirror:
            xnew = _symmetrize(xnew, domain.mirror)
        xnew = _extrude(xnew, columns, design)
        change = float(np.abs(xnew - x)[design].max())
        x = xnew
        since_beta += 1

        rec = {"iteration": it, "volume": float(xp[design].mean()), "beta": beta, "change": change,
               "constraints": [float(v) for v in cons], "seconds": time.time() - t0,
               "cgIterations": fe.last_iters, "solvedCells": fe.last_cells, **record}
        history.append(rec)
        if on_iter is not None:
            on_iter(rec, xp)

        feasible = max(cons) <= 0.02
        # Sharpen only once the design is inside the limits (or after three
        # times the usual wait): each doubling raises the peak, and doubling
        # on the timer while infeasible stacked those jumps until the run
        # ended at twice the volume it needed (L-bracket, 0.59 against 0.31).
        if beta < opts.beta_max and ((feasible and (since_beta >= opts.beta_every or change < 2 * opts.tol))
                                     or since_beta >= 3 * opts.beta_every):
            beta = min(beta * 2, opts.beta_max)
            since_beta = 0
            # Stress is far more nonlinear in the design than compliance, and
            # more so as the projection sharpens: with the full move limit a
            # single slack step at beta 8 cut the lug's volume in half and
            # disconnected it. Shrink the steps instead of resetting them.
            move_cap = max(0.02, move_cap * 0.6)
            mma.move = min(mma.move, move_cap)
        elif beta >= opts.beta_max:
            if feasible and (best is None or rec["volume"] < best[1]["volume"]):
                best = (x_now, rec)
            trail.append(best[1]["volume"] if best is not None else np.inf)
            w = opts.stall_window
            if best is not None and ((feasible and change < opts.tol) or (
                    len(trail) > w and trail[-1] > (1.0 - STALL_GAIN) * trail[-1 - w])):
                converged = True
                break

    # Deliver the blueprint, the design whose volume was minimized. The
    # compliance optimizer delivers the opening of the eroded design instead
    # (simp.optimize), but here that reshaped thin members after their stress
    # was checked: on the L-bracket the opening read up to 4x the limit where
    # the blueprint reads within about 10% of it, at the same volume.
    # And of the designs checked at the sharpest projection, the lightest one
    # inside the limits: the last iterate is a step nobody has checked, and
    # MMA walks along the limit with every few steps a little over it.
    if best is not None:
        x = best[0]
    xp = _project((h @ x) / hs, beta, opts.eta)
    xp[pres] = 1.0
    xp = printed(am_b, xp)
    if buck is not None:
        buck.keep()     # the study's next run on this grid starts from these modes
    return SimpResult(x=x, xphys=xp, compliance=float("nan"), case_compliance=[],
                      volume_fraction=float(xp[design].mean()), iterations=it,
                      converged=converged, seconds=time.time() - t0, history=history,
                      delivered=best[1] if best is not None else None)

