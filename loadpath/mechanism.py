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
take either sign, so the update is the damped optimality criteria of
Sigmund's 104-line mechanism code (sensitivities clipped positive,
exponent 0.3).

The density filter, Heaviside projection, extrusion (2.5D) and
reconstruction are the same as for stiffness designs, so the result comes
back as a smooth, printable part. Linear elasticity only: a TPU finger that
curls 60 degrees is well outside small deflections, so treat the output as
the layout, and check the curl on a print.
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
    k_in: float = 0.0                 # N/mm, spread over the input's loaded dofs
    k_out: float = 0.0                # N/mm, spread over the output's dofs
    modulus: float = 1.0              # MPa: the material's E (springs are in real units)
    max_iterations: int = 120
    beta_max: float = 16.0
    beta_every: int = 20
    move: float = 0.1
    damping: float = 0.3
    extrusion: int | None = None
    tol: float = 0.005


@dataclass
class MechanismResult:
    x: np.ndarray
    xphys: np.ndarray
    output_displacement: float        # mm along the output direction, under the input force
    iterations: int
    seconds: float
    history: list = field(default_factory=list)


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
    f_in = domain.loads[opts.input_case]
    f_out = domain.loads[opts.output_case]
    # l . u = the output region's mean displacement along the output direction
    l = f_out / max(np.linalg.norm(f_out.reshape(-1, 3).sum(axis=0)), 1e-30)
    springs = sp.diags(_spring_diag(f_in, opts.k_in, mesh.n_dof) + _spring_diag(f_out, opts.k_out, mesh.n_dof))
    springs = springs.multiply(~domain.fixed[:, None]).tocsr() if domain.fixed.any() else springs.tocsr()

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
    u = lam = None
    history = []
    it = 0
    for it in range(1, opts.max_iterations + 1):
        xt = _extrude((h @ x) / hs, columns, design)
        xp = _project(xt, beta, 0.5)
        xp[pres] = 1.0
        e = opts.modulus * (E_MIN + xp ** PENALTY * (1 - E_MIN))
        k = (stiff.assemble(e) + springs).tocsr()
        u, lam = solver.solve_many(k, [f_in, l], [u, lam])
        uout = float(l @ u)
        ue, le = u[edof], lam[edof]
        # d(u_out)/d(xp_e) = -lambda_e . k_e u_e * dE/dxp   (K lambda = l, K symmetric)
        dE = opts.modulus * PENALTY * xp ** (PENALTY - 1) * (1 - E_MIN)
        g = -np.einsum("ij,jk,ik->i", le, ke, ue) * dE
        g = g * _dproject(xt, beta, 0.5)
        g = _extrude(g, columns, design)
        g = ht @ (g / hs)
        dv = ht @ (_dproject(xt, beta, 0.5) / hs)
        # damped OC on a maximization: grow where g > 0
        gp = np.maximum(g, 1e-30)
        lo_l, hi_l = 1e-30, 1e30
        while hi_l / lo_l > 1 + 1e-4:
            mid = np.sqrt(lo_l * hi_l)
            xn = np.clip(x * (gp / (mid * np.maximum(dv, 1e-30))) ** opts.damping,
                         np.maximum(0.0, x - opts.move), np.minimum(1.0, x + opts.move))
            xn[pres] = 1.0
            vol = _project(_extrude((h @ xn) / hs, columns, design), beta, 0.5)[design].sum()
            if vol > target:
                lo_l = mid
            else:
                hi_l = mid
        change = float(np.abs(xn - x).max())
        x = xn
        rec = {"iteration": it, "outputDisplacement": uout, "beta": beta, "change": change,
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
                old = history[-11]["outputDisplacement"]
                if abs(uout - old) <= 2e-3 * abs(old):
                    break
    xt = _extrude((h @ x) / hs, columns, design)
    xp = _project(xt, beta, 0.5)
    xp[pres] = 1.0
    return MechanismResult(x=x, xphys=xp, output_displacement=float(history[-1]["outputDisplacement"]),
                           iterations=it, seconds=time.time() - t0, history=history)
