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
  symmetry, extrusion, the reduced solve) is shared with simp.optimize.
"""
from __future__ import annotations

import time

import numpy as np

from .domain import Domain
from .fem import elastic_c, hex8_b_center
from .manufacture import tie_groups
from .mma import MMA
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
_VM = np.array([
    [1.0, -0.5, -0.5, 0, 0, 0],
    [-0.5, 1.0, -0.5, 0, 0, 0],
    [-0.5, -0.5, 1.0, 0, 0, 0],
    [0, 0, 0, 3.0, 0, 0],
    [0, 0, 0, 0, 3.0, 0],
    [0, 0, 0, 0, 0, 3.0],
])


class Limits:
    def __init__(self, stress: float | None, deflection: float | None, modulus: float):
        self.stress = stress            # MPa, allowable peak von Mises
        self.deflection = deflection    # mm at the loads
        self.modulus = modulus          # MPa, to turn unit-E displacements into mm


def optimize_constrained(domain: Domain, nu: float, opts: SimpOptions, limits: Limits,
                         on_iter=None, should_stop=None) -> SimpResult:
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

    weights = np.asarray(domain.case_weights, float)
    cases = len(domain.loads)
    force_mag = [float(np.linalg.norm(f.reshape(-1, 3).sum(axis=0))) for f in domain.loads]
    defl_cases = [c for c in range(cases) if limits.deflection is not None and force_mag[c] > 0]
    stress_cases = list(range(cases)) if limits.stress is not None else []
    m = len(defl_cases) + len(stress_cases)
    if m == 0:
        raise ValueError("optimize_constrained needs a stress or deflection limit")

    # Stress is not counted in cells where loads and supports are applied.
    contact = (domain.support_nodes | domain.load_nodes)[mesh.enode].any(axis=1)
    counted = ~contact
    b0 = hex8_b_center() / mesh.pitch
    cb = elastic_c(1.0, nu) @ b0                      # (6, 24): stress per unit-E displacement

    free = np.flatnonzero(design)
    x = np.full(n, opts.volume_fraction)
    if opts.x0 is not None:
        x = np.clip(np.asarray(opts.x0, float), 0.0, 1.0)
    x[pres] = 1.0
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

    for it in range(1, opts.max_iterations + 1):
        if should_stop is not None and should_stop():
            break
        xt = (h @ x) / hs
        xp = _project(xt, beta, opts.eta)
        xp[pres] = 1.0
        xe = _project(xt, beta, eta_e)
        xe[pres] = 1.0
        dpb = _dproject(xt, beta, opts.eta)
        dpb[pres] = 0.0
        dpe = _dproject(xt, beta, eta_e)
        dpe[pres] = 0.0

        e = E_MIN + xe ** PENALTY * (1 - E_MIN)
        de = PENALTY * xe ** (PENALTY - 1) * (1 - E_MIN)
        us = fe.solve(e, xe, us, loads=domain.loads)

        def to_x(g_xe):
            return ht @ (g_xe * dpe / hs)

        cons, grads, record = [], [], {}
        for c in defl_cases:
            comp = float(domain.loads[c] @ us[c])
            defl = comp / (force_mag[c] * limits.modulus)
            ce = fe.strain_energy(us[c])
            g_xe = -de * ce / (force_mag[c] * limits.modulus)
            cons.append(defl / limits.deflection - 1.0)
            grads.append(to_x(g_xe) / limits.deflection)
            record[f"deflection.{domain.case_names[c]}"] = defl

        if stress_cases:
            adj_rhs, partials = [], []
            for c in stress_cases:
                ue = us[c][mesh.edof]                        # (n, 24)
                sig = ue @ cb.T                              # (n, 6) solid stress, MPa
                vm = np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", sig, _VM, sig), 1e-30))
                srel = np.where(counted, xe ** _Q * vm / limits.stress, 0.0)
                nc = max(int(counted.sum()), 1)
                pm = (np.sum(srel ** _P) / nc) ** (1.0 / _P)
                smax = float(srel.max())
                kappa[c] = 0.5 * (kappa[c] + smax / max(pm, 1e-30)) if it > 1 else smax / max(pm, 1e-30)
                g = kappa[c] * pm
                dg_ds = kappa[c] * (srel ** (_P - 1)) * pm ** (1.0 - _P) / nc
                # explicit: d s / d xe at fixed u
                g_xe = dg_ds * np.where(counted, _Q * np.maximum(xe, 1e-12) ** (_Q - 1) * vm / limits.stress, 0.0)
                # adjoint right-hand side: dg/du
                coef = dg_ds * np.where(counted, xe ** _Q / (vm * limits.stress), 0.0)   # (n,)
                dvm_du = (sig @ _VM) @ cb                                               # (n, 24) = d(vm^2)/2du
                rhs = np.bincount(mesh.edof.ravel(), weights=(coef[:, None] * dvm_du).ravel(),
                                  minlength=mesh.n_dof)
                rhs[domain.fixed] = 0.0
                adj_rhs.append(rhs)
                partials.append((c, g, g_xe, smax))
            lams = fe.resolve(adj_rhs, [l for l in lams] if lams[0] is not None and len(lams) == len(adj_rhs) else None)
            for (c, g, g_xe, smax), lam in zip(partials, lams):
                le = lam[mesh.edof]
                ue = us[c][mesh.edof]
                g_xe = g_xe - de * np.einsum("ij,jk,ik->i", le, fe.ke, ue)
                cons.append(g - 1.0)
                grads.append(to_x(g_xe))
                record[f"stress.{domain.case_names[c]}"] = smax * limits.stress

        f0 = 10.0 * float(xp[design].mean())
        df0 = ht @ (10.0 * design / design.sum() * dpb / hs)
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
        elif beta >= opts.beta_max and since_beta >= 5 and feasible:
            w = opts.stall_window
            if change < opts.tol or (len(history) > w and
                                     abs(history[-1]["volume"] - history[-1 - w]["volume"]) < 2e-3):
                converged = True
                break

    # Deliver the blueprint, the design whose volume was minimized. The
    # compliance optimizer delivers the opening of the eroded design instead
    # (simp.optimize), but here that reshaped thin members after their stress
    # was checked: on the L-bracket the opening read up to 4x the limit where
    # the blueprint reads within about 10% of it, at the same volume.
    xp = _project((h @ x) / hs, beta, opts.eta)
    xp[pres] = 1.0
    return SimpResult(x=x, xphys=xp, compliance=float("nan"), case_compliance=[],
                      volume_fraction=float(xp[design].mean()), iterations=it,
                      converged=converged, seconds=time.time() - t0, history=history)

