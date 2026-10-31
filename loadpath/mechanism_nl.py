"""
Compliant mechanisms optimized on their large-deflection response.

mechanism.py designs on the linear model, which is right for small strokes
and wrong for a TPU finger that curls tens of degrees: members swing round,
the leverage changes as they do, and a linkage that looks best at the
start of the stroke can stall or reverse halfway (Pedersen, Buhl & Sigmund
2001, "Topology synthesis of large-displacement compliant mechanisms").
Here every design iteration solves the geometrically nonlinear problem
(total Lagrangian hex8, neo-Hookean, nonlinear.py's element) with the input
driven through its full stroke, and the sensitivity comes from one adjoint
solve on the converged tangent:

    maximize  J = l . u(stroke)          (output travel against k_out)
    R(u, x) = f_int(u, x) + k_out u = 0  on the free dofs, input prescribed
    K_T lambda = l,   dJ/dx_e = -lambda_e . dR_e/dx_e

Void cells are the classic trouble: at 1e-9 of the modulus they turn inside
out under the large motion of their solid neighbours and Newton fails. The
fix is Wang, Lazarov & Sigmund 2014 ("Interpolation scheme for fictitious
domain techniques and topology optimization of finite strain elastic
problems"): each cell's internal force is

    f_e = E_e [ gamma_e f_NL(gamma_e u_e) + (1 - gamma_e^2) K0 u_e ]

with gamma_e a sharp step of the density at `void_threshold` (1 above, 0
below), so solid cells are fully nonlinear and void cells linear, which
cannot invert. The default, 0.1, treats grey material as nonlinear too, so
the optimizer sees the large-deflection behaviour from the first
iteration; 0.5 keeps grey cells linear (a continuation from the linear
design, more forgiving of Newton failures, and on the inverter a worse
optimum: 2.4 against 3.5). A design step whose Newton solve fails is
backtracked halfway to the last design that converged.

The filter, projection, extrusion, robust (eroded / dilated) option and MMA
are mechanism.py's. Each iteration costs one Newton solve per realization
(warm-started from the last design's displacement, with load steps only
when that fails) plus one linear adjoint solve.
"""
from __future__ import annotations

import time

import numpy as np
import scipy.sparse as sp

from .fem import Stiffness, hex8_ke
from .mechanism import MechanismOptions, MechanismResult, _spring_diag
from .nonlinear import _CENTER, _CHUNK, _element, _Linear
from .simp import E_MIN, PENALTY, _dproject, _extrude, _project, density_filter

__all__ = ["optimize_mechanism_nonlinear", "nonlinear_response", "stroke_dofs"]

GAMMA_ETA, GAMMA_BETA = 0.1, 100.0    # the solid/void step on the physical density


def _gamma(xp, eta=GAMMA_ETA, beta=GAMMA_BETA):
    t0 = np.tanh(beta * eta)
    den = t0 + np.tanh(beta * (1 - eta))
    g = (t0 + np.tanh(beta * (xp - eta))) / den
    dg = beta * (1 - np.tanh(beta * (xp - eta)) ** 2) / den
    return g, dg


def stroke_dofs(mesh, f, fixed, stroke):
    """The input region's dofs moved by `stroke` mm along the input force's
    direction: (dofs, values)."""
    fv = np.asarray(f, float).reshape(-1, 3)
    loaded = np.flatnonzero(np.abs(fv).sum(axis=1) > 0)
    d = fv.sum(axis=0)
    d /= np.linalg.norm(d)
    pd, pv = [], []
    for c in range(3):
        if abs(d[c]) > 1e-9:
            dofs = 3 * loaded + c
            dofs = dofs[~np.asarray(fixed, bool)[dofs]]
            pd.append(dofs)
            pv.append(np.full(len(dofs), stroke * d[c]))
    return np.concatenate(pd), np.concatenate(pv)


class _Model:
    """Mixed linear/nonlinear assembly on one mesh, with the input prescribed."""

    def __init__(self, mesh, nu, fixed, pd, pv, springs, kind="neo-hookean"):
        self.mesh, self.kind = mesh, kind
        # Poisson's ratio per element (two materials) or one for all
        nu_e = np.broadcast_to(np.asarray(nu, float), (mesh.n_elem,))
        self.groups = [(float(v), np.flatnonzero(nu_e == v)) for v in np.unique(nu_e)]
        self.k0s = {v: hex8_ke(v) * mesh.pitch for v, _ in self.groups}
        nu = float(nu_e[0])
        cons = np.asarray(fixed, bool).copy()
        cons[pd] = True
        self.cons, self.pd, self.pv = cons, pd, pv
        self.free = ~cons
        self.stiff = Stiffness(mesh, nu, cons)
        self.springs = np.where(cons, 0.0, springs)
        self.lin = _Linear(mesh)

    def assemble(self, u, mod, gam, tangent=True, parts=False):
        mesh = self.mesh
        fint = np.zeros(mesh.n_dof)
        data = np.zeros(self.stiff.nnz) if tangent else None
        jmin = np.inf
        br = np.zeros((mesh.n_elem, 24)) if parts else None
        dbr = np.zeros((mesh.n_elem, 24)) if parts else None
        chunks = [(v, idx[s:s + _CHUNK]) for v, idx in self.groups for s in range(0, len(idx), _CHUNK)]
        for nu, el in chunks:
            k0 = self.k0s[nu]
            edof = mesh.edof[el]
            ue = u[edof]
            g = gam[el]
            fh, kh, _ = _element((g[:, None] * ue).reshape(-1, 8, 3), mesh.pitch, nu, self.kind, True)
            # J of the solid cells only: void ones are linear and may not invert
            solid = g > 0.5
            if solid.any():
                F = np.eye(3)[None] + np.einsum("mai,aj->mij", ue[solid].reshape(-1, 8, 3), _CENTER / mesh.pitch)
                jmin = min(jmin, float(np.linalg.det(F).min()))
            ku = ue @ k0.T
            b = g[:, None] * fh + (1 - g ** 2)[:, None] * ku
            me = mod[el]
            np.add.at(fint, edof.ravel(), (b * me[:, None]).ravel())
            if tangent:
                ke = (g ** 2)[:, None, None] * kh + (1 - g ** 2)[:, None, None] * k0[None]
                self.stiff.add_elements(data, ke * me[:, None, None], el)
            if parts:
                br[el] = b
                dbr[el] = fh + g[:, None] * np.einsum("mij,mj->mi", kh, ue) - 2 * g[:, None] * ku
        fint += self.springs * u
        k = None
        if tangent:
            k = (self.stiff.finish(data) + sp.diags(self.springs)).tocsr()
        return fint, k, jmin, br, dbr

    def newton(self, u, mod, gam, lam, rtol=1e-8, max_it=25):
        u = u.copy()
        u[self.pd] = lam * self.pv
        r0 = None
        for _ in range(max_it):
            fint, k, jmin, _, _ = self.assemble(u, mod, gam)
            if not np.isfinite(jmin) or jmin <= 0:
                return None
            r = np.where(self.free, fint, 0.0)
            rn = float(np.linalg.norm(r))
            if r0 is None:
                r0 = max(rn, float(np.linalg.norm(fint[self.pd])), 1e-30)
            if rn <= rtol * r0:
                return u
            du = self.lin.solve(k, -r)
            du[~self.free] = 0.0
            u += du
            if not np.all(np.isfinite(u)):
                return None
        return None

    def solve(self, mod, gam, u0=None, steps=4):
        """Equilibrium at the full stroke: straight from u0 (the last design's
        answer), else in load steps from rest, halving a step that fails."""
        if u0 is not None:
            u = self.newton(u0, mod, gam, 1.0)
            if u is not None:
                return u
        u = np.zeros(self.mesh.n_dof)
        lam, dl = 0.0, 1.0 / steps
        while lam < 1.0 - 1e-12:
            t = min(1.0, lam + dl)
            un = self.newton(u * (t / lam) if lam > 0 else u, mod, gam, t)
            if un is None:
                dl /= 2
                if dl < 1e-3:
                    raise RuntimeError(f"Newton failed at {lam:.3f} of the stroke")
                continue
            u, lam = un, t
            dl = min(dl * 1.5, 1.0 - lam if lam < 1 else dl)
        return u


def nonlinear_response(model: _Model, xp, modulus, lvec, u0=None, steps=4, void_threshold=GAMMA_ETA):
    """J = lvec . u at the full stroke and dJ/dxp per element. modulus:
    scalar or per element (a two-material part: the solid's E in each cell)."""
    modulus = np.asarray(modulus, float)
    mod = modulus * (E_MIN + xp ** PENALTY * (1 - E_MIN))
    dmod = modulus * PENALTY * xp ** (PENALTY - 1) * (1 - E_MIN)
    gam, dgam = _gamma(xp, void_threshold)
    u = model.solve(mod, gam, u0, steps)
    _, k, _, br, dbr = model.assemble(u, mod, gam, parts=True)
    lam = model.lin.solve(k, np.where(model.free, lvec, 0.0))
    lam[~model.free] = 0.0
    le = lam[model.mesh.edof]
    dj = -(np.einsum("mi,mi->m", le, br) * dmod + np.einsum("mi,mi->m", le, dbr) * mod * dgam)
    return float(lvec @ u), dj, u


def optimize_mechanism_nonlinear(domain, nu, opts: MechanismOptions, stroke: float, steps: int = 4,
                                 material: str = "neo-hookean", void_threshold: float = GAMMA_ETA,
                                 on_iter=None) -> MechanismResult:
    """mechanism.optimize_mechanism on the large-deflection response: the
    input case's region is driven `stroke` mm along its force, the output
    case gives the wanted direction, k_out is the workpiece. k_in is
    ignored (the stroke is prescribed). nu and opts.modulus may be per
    element (a two-material part). Returns the output travel at full stroke
    as output_displacement."""
    t0 = time.time()
    mesh = domain.mesh
    n = mesh.n_elem
    pres = domain.preserve
    design = ~pres
    fin = domain.loads[opts.input_case]
    lo = domain.loads[opts.output_case]
    wts = np.abs(lo.reshape(-1, 3)).sum(axis=1) > 0
    d = lo.reshape(-1, 3).sum(axis=0)
    lvec = (wts[:, None] * (d / np.linalg.norm(d))[None, :]).ravel() / max(wts.sum(), 1)
    pd, pv = stroke_dofs(mesh, fin, domain.fixed, stroke)
    springs = _spring_diag(lo, opts.k_out, mesh.n_dof)
    springs[domain.fixed] = 0.0
    model = _Model(mesh, nu, domain.fixed, pd, pv, springs, material)

    columns = None
    if opts.extrusion is not None:
        a = opts.extrusion
        others = [i for i in range(3) if i != a]
        ij = mesh.elem_ijk[:, others]
        _, columns = np.unique(ij[:, 0] * (domain.dims[others[1]] + 1) + ij[:, 1], return_inverse=True)
    h, hs = density_filter(domain, opts.filter_radius, pad=opts.extrusion is None)
    ht = h.T.tocsr()
    x = np.full(n, opts.volume_fraction)
    x[pres] = 1.0
    target = opts.volume_fraction * design.sum()
    beta, since = 1.0, 0
    etas = [0.5] if not opts.robust else [0.5 + opts.robust, 0.5, 0.5 - opts.robust]
    from .mma import MMA
    mma = MMA(int(design.sum()), 1, xmin=0.0, xmax=1.0, move=opts.move)
    history, u_prev, j0 = [], {}, None
    x_ok, beta_ok, backtracks, beta_cap, stopped = None, beta, 0, opts.beta_max, None
    it = 0
    for it in range(1, opts.max_iterations + 1):
        real = None
        for tries in range(8):
            xt = _extrude((h @ x) / hs, columns, design)
            try:
                real, up = [], {}
                for eta in etas:
                    xp = _project(xt, beta, eta)
                    xp[pres] = 1.0
                    j, dj, u = nonlinear_response(model, xp, opts.modulus, lvec, u_prev.get(eta), steps,
                                                  void_threshold)
                    up[eta] = u
                    real.append((j, dj * _dproject(xt, beta, eta)))
                u_prev.update(up)
                break
            except RuntimeError:
                real = None
                if x_ok is None:
                    raise
                backtracks += 1
                if beta > beta_ok:
                    # the sharper projection is what broke it: keep the last
                    # beta and stop sharpening
                    beta, beta_cap = beta_ok, beta_ok
                else:
                    x = 0.5 * (x + x_ok)      # backtrack toward the last design that converged
        if real is None:
            # no nearby design reaches the full stroke: keep the last one that did
            x, beta, stopped = x_ok, beta_ok, "the stroke could not be reached from the last design"
            break
        x_ok, beta_ok = x.copy(), beta
        jb = real[len(etas) // 2][0]
        if j0 is None:
            j0 = max(abs(jb), 1e-9)
        vals = np.array([r[0] / j0 for r in real])
        if len(etas) == 1:
            obj, g = vals[0], real[0][1] / j0
        else:
            q = 30.0
            w = np.exp(-q * (vals - vals.min()))
            obj = float(vals.min() - np.log(w.sum()) / q)
            w /= w.sum()
            g = sum(wk * gk for wk, (_, gk) in zip(w, real)) / j0
        g = ht @ (_extrude(g, columns, design) / hs)
        dv = ht @ (_extrude(_dproject(xt, beta, 0.5), columns, design) / hs)
        xp = _project(xt, beta, 0.5)
        xp[pres] = 1.0
        vol = float(xp[design].sum())
        xn = x.copy()
        xn[design] = mma.update(x[design], -obj * 10.0, (-g * 10.0)[design],
                                np.array([vol / target - 1.0]), np.array([(dv / target)[design]]))
        change = float(np.abs(xn - x).max())
        x = xn
        rec = {"iteration": it, "outputDisplacement": jb, "objective": obj, "beta": beta, "change": change,
               "volume": float(xp[design].mean()), "backtracks": backtracks}
        if len(etas) > 1:
            rec["robustOutputs"] = [float(r[0]) for r in real]
        history.append(rec)
        if on_iter is not None:
            on_iter(rec, xp)
        since += 1
        if beta < beta_cap and (since >= opts.beta_every or change < opts.tol * 2):
            beta, since = min(beta * 2, beta_cap), 0
        elif beta >= beta_cap:
            if change < opts.tol:
                break
            if since >= 10 and len(history) > 10:
                old = history[-11]["objective"]
                if abs(obj - old) <= 2e-3 * abs(old):
                    break
    if stopped:
        history[-1]["stopped"] = stopped
    if beta_cap < opts.beta_max:
        history[-1]["betaCap"] = beta_cap
    xt = _extrude((h @ x) / hs, columns, design)
    xp = _project(xt, beta, 0.5)
    xp[pres] = 1.0
    return MechanismResult(x=x, xphys=xp, output_displacement=float(history[-1]["outputDisplacement"]),
                           iterations=it, seconds=time.time() - t0, history=history)
