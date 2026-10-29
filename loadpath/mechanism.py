"""
Compliant mechanisms: parts that are meant to bend.

Everything else in loadpath makes a part as stiff as possible for its mass.
A gripper finger, a flexure hinge, a snap fit or a clip needs the opposite:
a push at one place (the input) should move another place (the output) as
far as possible in a chosen direction, against something that pushes back
(the workpiece). That is the classic compliant-mechanism problem (Sigmund
1997; Bendsøe & Sigmund, "Topology Optimization", ch. 2.6):

    maximize   u_out = l . u
    subject to (K(x) + springs) u = f_in,  sum(x) <= V

with a spring k_in at the input (the actuator is not infinitely stiff) and
k_out at the output (the object being gripped). Without the springs the
optimum is a mechanism of one-node hinges with no strength at all; the
output spring is what makes the result carry a grip force.

The sensitivity needs one adjoint solve, K lambda = l, and is
lambda_e . k_e u_e scaled by the SIMP derivative: unlike compliance, it can
take either sign. Sigmund's 104-line code uses damped optimality criteria
with the sensitivities clipped positive; here it is MMA (mma.py), which
takes the gradient whole.

The density filter, Heaviside projection, extrusion (2.5D) and
reconstruction are the same as for stiffness designs, so the result comes
back as a smooth, printable part. The optimization is linear: a TPU finger
that curls 60 degrees is well outside small deflections, so treat the output
as the layout. `verify_nonlinear` re-solves the solid cells of the result
with large deflections and a buckling check (nonlinear.verify_mechanism):
how far the output really travels, and whether the input force buckles a
thin member first.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp

from .domain import Domain
from .fem import Solver, Stiffness
from .simp import E_MIN, PENALTY, _dproject, _extrude, _project, density_filter

__all__ = ["MechanismOptions", "MechanismResult", "optimize_mechanism"]


@dataclass
class MechanismOptions:
    volume_fraction: float
    filter_radius: float
    input_case: int = 0               # domain load case that is the actuator force
    output_case: int = 1              # domain load case whose force direction is the output (any magnitude)
    # Several (input, output) pairs at once, e.g. contact anywhere along a
    # finger: the objective is the sum of the pairs' output displacements.
    pairs: list | None = None
    k_in: float = 0.0                 # N/mm, spread over the input's loaded dofs
    k_out: float = 0.0                # N/mm, spread over the output's dofs
    modulus: float = 1.0              # MPa: the material's E (springs are in real units)
    max_iterations: int = 120
    beta_max: float = 16.0
    beta_every: int = 20
    move: float = 0.1
    extrusion: int | None = None
    tol: float = 0.005
    verify_nonlinear: bool = False    # large-deflection + buckling check of the result (first pair)


@dataclass
class MechanismResult:
    x: np.ndarray
    xphys: np.ndarray
    output_displacement: float        # mm along the output direction, under the input force
    iterations: int
    seconds: float
    history: list = field(default_factory=list)
    nonlinear: dict | None = None     # nonlinear.verify_mechanism's report, with verify_nonlinear


def _spring_diag(f, k, n_dof):
    dofs = np.nonzero(np.abs(f) > 0)[0]
    d = np.zeros(n_dof)
    if len(dofs) and k > 0:
        d[dofs] = k / len(dofs) * 3.0   # per node, three dofs share one spring
    return d


def optimize_mechanism(domain: Domain, nu: float, opts: MechanismOptions, on_iter=None) -> MechanismResult:
    t0 = time.time()
    mesh = domain.mesh
    n = mesh.n_elem
    pres = domain.preserve
    design = ~pres
    pairs = opts.pairs or [(opts.input_case, opts.output_case)]
    ins = sorted({i for i, _ in pairs})
    outs = sorted({o for _, o in pairs})
    # l . u = the output region's mean displacement along the output direction
    ls = {o: domain.loads[o] / max(np.linalg.norm(domain.loads[o].reshape(-1, 3).sum(axis=0)), 1e-30) for o in outs}
    sd = sum(_spring_diag(domain.loads[i], opts.k_in, mesh.n_dof) for i in ins) + \
        sum(_spring_diag(domain.loads[o], opts.k_out, mesh.n_dof) for o in outs)
    sd[domain.fixed] = 0.0
    springs = sp.diags(sd).tocsr()

    stiff = Stiffness(mesh, nu, domain.fixed)
    solver = Solver(mesh, rtol=1e-5)
    columns = None
    if opts.extrusion is not None:
        a = opts.extrusion
        others = [i for i in range(3) if i != a]
        ij = mesh.elem_ijk[:, others]
        _, columns = np.unique(ij[:, 0] * (domain.dims[others[1]] + 1) + ij[:, 1], return_inverse=True)
    h, hs = density_filter(domain, opts.filter_radius, pad=opts.extrusion is None)
    ht = h.T.tocsr()
    from .fem import hex8_ke
    ke = hex8_ke(nu) * mesh.pitch      # as Stiffness assembles it
    edof = mesh.edof

    x = np.full(n, opts.volume_fraction)
    x[pres] = 1.0
    target = opts.volume_fraction * design.sum()
    beta, since = 1.0, 0
    prev = None
    u0 = None
    from .mma import MMA
    mma = MMA(int(design.sum()), 1, xmin=0.0, xmax=1.0, move=opts.move)
    history = []
    it = 0
    for it in range(1, opts.max_iterations + 1):
        xt = _extrude((h @ x) / hs, columns, design)
        xp = _project(xt, beta, 0.5)
        xp[pres] = 1.0
        e = opts.modulus * (E_MIN + xp ** PENALTY * (1 - E_MIN))
        k = (stiff.assemble(e) + springs).tocsr()
        sol = solver.solve_many(k, [domain.loads[i] for i in ins] + [ls[o] for o in outs], prev)
        prev = sol
        U = dict(zip(ins, sol[:len(ins)]))
        LAM = dict(zip(outs, sol[len(ins):]))
        dE = opts.modulus * PENALTY * xp ** (PENALTY - 1) * (1 - E_MIN)
        raw = [float(ls[o] @ U[i]) for i, o in pairs]
        # Sum the outputs in mm, with one overall scale for MMA. (Scaling each
        # pair by its own starting value let a pair that started near zero
        # dominate; the fin-ray finger traded one contact's wrap for another's.)
        if u0 is None:
            u0 = max(abs(sum(raw)), max(abs(r) for r in raw), 1e-12)
        uout = sum(raw) / u0
        # d(u_out)/d(xp_e) = -lambda_e . k_e u_e * dE/dxp   (K lambda = l, K symmetric)
        g = np.zeros(n)
        for i, o in pairs:
            g -= np.einsum("ij,jk,ik->i", LAM[o][edof], ke, U[i][edof])
        g = g * dE / u0
        g = g * _dproject(xt, beta, 0.5)
        g = _extrude(g, columns, design)
        g = ht @ (g / hs)
        dv = ht @ (_extrude(_dproject(xt, beta, 0.5), columns, design) / hs)
        # MMA on (-u_out, volume): the output sensitivity takes both signs,
        # which optimality criteria can only handle by clipping (the first
        # version did, and stalled on the fin-ray problem with half its
        # gradient thrown away)
        vol = float(xp[design].sum())
        xn = x.copy()
        xn[design] = mma.update(x[design], -uout * 10.0, (-g * 10.0)[design],
                                np.array([vol / target - 1.0]), np.array([(dv / target)[design]]))
        change = float(np.abs(xn - x).max())
        x = xn
        rec = {"iteration": it, "outputDisplacement": raw[0], "objective": uout, "outputs": raw, "beta": beta, "change": change,
               "volume": float(xp[design].mean())}
        history.append(rec)
        if on_iter is not None:
            on_iter(rec, xp)
        since += 1
        if beta < opts.beta_max and (since >= opts.beta_every or change < opts.tol * 2):
            beta, since = min(beta * 2, opts.beta_max), 0
        elif beta >= opts.beta_max:
            if change < opts.tol:
                break
            # mechanisms chatter at the move limit on a few hinge cells while
            # the output has stopped moving: stop on the output instead
            if since >= 10 and len(history) > 10:
                old = history[-11]["objective"]
                if abs(uout - old) <= 2e-3 * abs(old):
                    break
    xt = _extrude((h @ x) / hs, columns, design)
    xp = _project(xt, beta, 0.5)
    xp[pres] = 1.0
    nonlinear = None
    if opts.verify_nonlinear:
        from .nonlinear import verify_mechanism
        nonlinear = verify_mechanism(domain, xp, opts.modulus, nu, input_case=pairs[0][0], output_case=pairs[0][1],
                                     k_in=opts.k_in, k_out=opts.k_out)
    return MechanismResult(x=x, xphys=xp, output_displacement=float(history[-1]["outputDisplacement"]),
                           iterations=it, seconds=time.time() - t0, history=history, nonlinear=nonlinear)
