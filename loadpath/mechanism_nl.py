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

__all__ = ["optimize_mechanism_nonlinear", "nonlinear_response", "stroke_dofs", "RigidCylinder", "ServoTendon",
           "tendon_grip"]

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


class ServoTendon:
    """A tendon wound in by a servo: the actuator pulls `stroke` mm of cable
    in through `path` (as nonlinear.Tendon: fixed points and guides on the
    part), and the cable and servo give like a spring of `stiffness` N/mm,
    so the tension is

        T = stiffness * (stroke - how far the part has let the cable in)

    (never negative: a slack cable pushes nothing). It is displacement
    driven, as a position-controlled servo is, so the solve can follow a
    finger past the point where a constant tension would snap it over, and
    the tension can never exceed stiffness * stroke (the stall force, say).
    Load steps scale the stroke."""

    def __init__(self, path, stroke, stiffness, slack=0.05):
        from .nonlinear import Tendon
        self._t = Tendon(tension=1.0, path=path)
        self.stroke, self.k, self.w = float(stroke), float(stiffness), float(slack)
        self.tension = self.k * self.stroke          # the most it can pull

    def _setup(self, mesh):
        self._t._setup(mesh)
        self._len0 = self._t._len0

    def pull(self, u, lam):
        """Tension (N), force on the part, and the load stiffness."""
        f1, k1, length = self._t._load(u, 1.0)          # f1 = -dL/du
        z = (length - (self._len0 - lam * self.stroke)) / self.w
        t = self.k * self.w * np.logaddexp(0.0, z)
        dt = self.k * 0.5 * (1 + np.tanh(0.5 * z))
        dl = sp.csr_matrix(-f1)
        return t, t * f1, t * k1 + dt * (dl.T @ dl)

    def _load(self, u, scale):
        t, f, k = self.pull(u, scale)
        return f, k, self._t._load(u, 1.0)[2]


class RigidCylinder:
    """A rigid, fixed object for the part to close on: a cylinder along
    `axis` (default y, the thickness of a plate) through `center`, radius
    `radius` (mm). Contact is a smooth penalty on the part's nodes:

        p(g) = k s ln(1 + exp(-g / s)),   g = distance to the axis - radius

    so a node pressed `d` into the surface pushes back about k d (N), one
    `s` outside it a little, and the response has a gradient everywhere. A
    node's share is weighted by the density around it (cubed, like SIMP),
    so void cells cannot grip. The grip force is the sum of the normal
    forces."""

    def __init__(self, center, radius, axis="y", stiffness=50.0, smooth=0.05):
        self.center = np.asarray(center, float)
        self.radius, self.axis = float(radius), "xyz".index(axis)
        self.k, self.s = float(stiffness), float(smooth)

    def _setup(self, mesh):
        self.plane = [i for i in range(3) if i != self.axis]
        adj = sp.csr_matrix((np.ones(mesh.n_elem * 8), (mesh.enode.ravel(), np.repeat(np.arange(mesh.n_elem), 8))),
                            shape=(mesh.n_nodes, mesh.n_elem))
        cnt = np.asarray(adj.sum(axis=1)).ravel()
        self.avg = sp.diags(1.0 / np.maximum(cnt, 1)) @ adj      # node density = mean of its cells
        self.x0 = mesh.node_xyz[:, self.plane] - self.center[self.plane]
        self.dofs = 3 * np.arange(mesh.n_nodes)[:, None] + np.array(self.plane)[None, :]

    def weights(self, xp):
        rho = self.avg @ xp
        return 1e-3 + (1 - 1e-3) * rho ** 3, self.avg.multiply((3 * (1 - 1e-3) * rho ** 2)[:, None]).tocsr()

    def state(self, u):
        r = self.x0 + u[self.dofs]
        dist = np.linalg.norm(r, axis=1)
        nrm = r / np.maximum(dist, 1e-12)[:, None]
        z = -(dist - self.radius) / self.s
        sig = 0.5 * (1 + np.tanh(0.5 * z))
        p = self.k * self.s * np.logaddexp(0.0, z)
        dp = -self.k * sig                                   # dp/dg
        ddp = self.k / self.s * sig * (1 - sig)
        return dist, nrm, p, dp, ddp

    def load(self, u, w, tangent=True):
        """Contact as a resisting force (adds to f_int) and its stiffness."""
        dist, nrm, p, dp, ddp = self.state(u)
        act = np.flatnonzero(p * w > 1e-12 * self.k)
        f = np.zeros(u.shape[0])
        np.add.at(f, self.dofs[act].ravel(), (-(w * p)[act, None] * nrm[act]).ravel())
        k = None
        if tangent:
            nn = np.einsum("ni,nj->nij", nrm[act], nrm[act])
            ke = -(w[act] * dp[act])[:, None, None] * nn - (w[act] * p[act] / dist[act])[:, None, None] * \
                (np.eye(2)[None] - nn)
            rows = np.repeat(self.dofs[act], 2, axis=1).ravel()
            cols = np.tile(self.dofs[act], (1, 2)).ravel()
            k = sp.csr_matrix((ke.ravel(), (rows, cols)), shape=(u.shape[0], u.shape[0]))
        return f, k

    def grip(self, u, w):
        """Grip force (N), its gradient in u, and per-node pressures."""
        dist, nrm, p, dp, _ = self.state(u)
        dj = np.zeros(u.shape[0])
        np.add.at(dj, self.dofs.ravel(), ((w * dp)[:, None] * nrm).ravel())
        return float(w @ p), dj, p, nrm


class _Model:
    """Mixed linear/nonlinear assembly on one mesh, with the input prescribed."""

    def __init__(self, mesh, nu, fixed, pd, pv, springs, kind="neo-hookean", tendons=None, contact=None):
        self.mesh, self.kind = mesh, kind
        self.tendons = list(tendons or [])
        for t in self.tendons:
            t._setup(mesh)
        self.contact = contact
        if contact is not None:
            contact._setup(mesh)
        self.w = None                     # contact weights, set per design
        # Poisson's ratio per element (two materials) or one for all
        nu_e = np.broadcast_to(np.asarray(nu, float), (mesh.n_elem,))
        self.groups = [(float(v), np.flatnonzero(nu_e == v)) for v in np.unique(nu_e)]
        # plain hex8: the linear cells must match the nonlinear element's
        # small-strain limit where the two meet
        self.k0s = {v: hex8_ke(v, "plain") * mesh.pitch for v, _ in self.groups}
        nu = float(nu_e[0])
        cons = np.asarray(fixed, bool).copy()
        cons[pd] = True
        self.cons, self.pd, self.pv = cons, pd, pv
        self.free = ~cons
        self.stiff = Stiffness(mesh, nu, cons)
        self.springs = np.where(cons, 0.0, springs)
        self.lin = _Linear(mesh)

    def assemble(self, u, mod, gam, tangent=True, parts=False, lam=1.0):
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
            k = self.stiff.finish(data) + sp.diags(self.springs)
        for t in self.tendons:             # the tendon pulls: an outside load, with its own stiffness
            ft, kt, _ = t._load(u, lam)     # (a ServoTendon's lam scales its stroke)
            fint -= ft
            if tangent:
                k = k + kt
        if self.contact is not None:
            fc, kc = self.contact.load(u, self.w, tangent)
            fint += fc
            if tangent:
                k = k + kc
        if tangent:
            k = k.tocsr()
        return fint, k, jmin, br, dbr

    def newton(self, u, mod, gam, lam, rtol=1e-8, max_it=40):
        u = u.copy()
        u[self.pd] = lam * self.pv
        r0 = None
        for _ in range(max_it):
            fint, k, jmin, _, _ = self.assemble(u, mod, gam, lam=lam)
            if not np.isfinite(jmin) or jmin <= 0:
                return None
            r = np.where(self.free, fint, 0.0)
            rn = float(np.linalg.norm(r))
            if r0 is None:
                r0 = max(rn, float(np.linalg.norm(fint[self.pd])), lam * sum(t.tension for t in self.tendons), 1e-30)
            if rn <= rtol * r0:
                return u
            du = self.lin.solve(k, -r)
            du[~self.free] = 0.0
            # damped step: a full Newton step can invert a solid cell next
            # to soft void ones; halve it until no cell inverts and the
            # residual does not blow up
            for _ in range(6):
                un = u + du
                fn, _, jn, _, _ = self.assemble(un, mod, gam, tangent=False, lam=lam)
                if np.all(np.isfinite(un)) and np.isfinite(jn) and jn > 0 and \
                        np.linalg.norm(np.where(self.free, fn, 0.0)) < 2.0 * rn:
                    break
                du *= 0.5
            else:
                return None
            u = un
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
    """J at the full stroke (or tension) and dJ/dxp per element: J = lvec . u
    (output travel), or with lvec None the grip force on model.contact.
    modulus: scalar or per element (a two-material part: the solid's E in
    each cell)."""
    modulus = np.asarray(modulus, float)
    mod = modulus * (E_MIN + xp ** PENALTY * (1 - E_MIN))
    dmod = modulus * PENALTY * xp ** (PENALTY - 1) * (1 - E_MIN)
    gam, dgam = _gamma(xp, void_threshold)
    dw = None
    if model.contact is not None:
        model.w, dw = model.contact.weights(xp)
    u = model.solve(mod, gam, u0, steps)
    _, k, _, br, dbr = model.assemble(u, mod, gam, parts=True)
    if lvec is None:
        j, dju, p, nrm = model.contact.grip(u, model.w)
    else:
        j, dju = float(lvec @ u), lvec
    lam = model.lin.solve(k, np.where(model.free, dju, 0.0))
    lam[~model.free] = 0.0
    le = lam[model.mesh.edof]
    dj = -(np.einsum("mi,mi->m", le, br) * dmod + np.einsum("mi,mi->m", le, dbr) * mod * dgam)
    if dw is not None:
        # the contact weights depend on the density: the grip term itself
        # and the contact force in the residual
        dist, nrm, p, _, _ = model.contact.state(u)
        ln = np.einsum("ni,ni->n", lam[model.contact.dofs], nrm)
        dj += dw.T @ (p * ((1.0 if lvec is None else 0.0) + ln))
    return j, dj, u


def _tension(tendons, u, lam):
    return float(sum(t.pull(u, lam)[0] if isinstance(t, ServoTendon) else lam * t.tension for t in tendons))


def tendon_grip(domain, xphys, modulus, nu, tendons, contact, threshold=0.5, steps=8,
                material="neo-hookean"):
    """Check a design the way it will be printed: cells above `threshold`
    solid, the rest empty (linear, at E_MIN, so they carry nothing), the
    tendons pulled to full tension in load steps, the part closing on
    `contact`. Returns the grip force (N), how much cable the actuator
    pulled in (mm), the nodes touching the object, the grip at each step,
    and the deformed positions (for a picture). A design that cannot take
    the full tension reports how far it got and its grip there."""
    mesh = domain.mesh
    xp = (np.asarray(xphys) > threshold).astype(float)
    model = _Model(mesh, nu, domain.fixed, np.zeros(0, np.int64), np.zeros(0), np.zeros(mesh.n_dof), material,
                   tendons, contact)
    model.w, _ = contact.weights(xp)
    mod = modulus * (E_MIN + xp ** PENALTY * (1 - E_MIN))
    gam, _ = _gamma(xp, 0.5)
    u, lam, dl, history = np.zeros(mesh.n_dof), 0.0, 1.0 / steps, []
    while lam < 1.0 - 1e-12:
        t = min(1.0, lam + dl)
        un = model.newton(u, mod, gam, t)
        if un is None:
            dl /= 2
            if dl < 1e-3:
                break
            continue
        u, lam = un, t
        history.append({"load": lam, "tension": _tension(model.tendons, u, lam), "grip": contact.grip(u, model.w)[0]})
        dl = min(dl * 1.5, 1.0 - lam if lam < 1 else dl)
    j, _, p, _ = contact.grip(u, model.w)
    rep = {"converged": lam >= 1.0 - 1e-12, "loadReached": lam, "tension": _tension(model.tendons, u, lam),
           "grip": j, "contactNodes": int((model.w * p > 0.01 * max(j, 1e-9)).sum()),
           "tendonTravel": [float(t._len0 - t._load(u, lam)[2]) for t in model.tendons], "history": history,
           "deformed": mesh.node_xyz + u.reshape(-1, 3), "solid": xp > 0.5}
    if not rep["converged"]:
        rep["message"] = "the solve stopped short of the full pull (the part snaps or wraps past the object)"
    return rep


def optimize_mechanism_nonlinear(domain, nu, opts: MechanismOptions, stroke: float | None = None, steps: int = 4,
                                 material: str = "neo-hookean", void_threshold: float = GAMMA_ETA,
                                 on_iter=None, tendons: list | None = None,
                                 contact: RigidCylinder | None = None) -> MechanismResult:
    """mechanism.optimize_mechanism on the large-deflection response.

    Input: the input case's region driven `stroke` mm along its force, or
    `tendons` (nonlinear.Tendon: a cable at constant tension through guides
    on the part, which follows the part as it curls; the guides should sit
    on preserved cells).

    Objective: the output case's travel along its direction against k_out
    (the workpiece as a spring), or with `contact` (a RigidCylinder: the
    object the part closes on) the grip force on it.

    k_in is ignored. nu and opts.modulus may be per element (a two-material
    part). output_displacement is the travel (mm) or the grip force (N)."""
    t0 = time.time()
    mesh = domain.mesh
    n = mesh.n_elem
    pres = domain.preserve
    design = ~pres
    if (stroke is None) == (not tendons):
        raise ValueError("drive the mechanism with a stroke or with tendons (one of them)")
    if tendons:
        pd, pv = np.zeros(0, np.int64), np.zeros(0)
    else:
        pd, pv = stroke_dofs(mesh, domain.loads[opts.input_case], domain.fixed, stroke)
    if contact is None:
        lo = domain.loads[opts.output_case]
        wts = np.abs(lo.reshape(-1, 3)).sum(axis=1) > 0
        d = lo.reshape(-1, 3).sum(axis=0)
        lvec = (wts[:, None] * (d / np.linalg.norm(d))[None, :]).ravel() / max(wts.sum(), 1)
        springs = _spring_diag(lo, opts.k_out, mesh.n_dof)
        springs[domain.fixed] = 0.0
    else:
        lvec, springs = None, np.zeros(mesh.n_dof)
    model = _Model(mesh, nu, domain.fixed, pd, pv, springs, material, tendons, contact)

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
    x_ok, beta_ok, backtracks, beta_cap, stopped, refused = None, beta, 0, opts.beta_max, None, 0
    it = 0
    full = [t.tension for t in tendons or []]
    # (servo tendons are displacement driven and need no ramp)
    ts_ok = 0.08 if tendons and not any(isinstance(t, ServoTendon) for t in tendons) else 0.8
    for it in range(1, opts.max_iterations + 1):
        # tension continuation: the uniform grey start is far too soft for
        # the full pull, so the tension rises 25% an iteration from 10% for
        # as long as the design can take it
        ts = min(1.0, ts_ok * 1.25)
        real = None
        for tries in range(8):
            for t, t0 in zip(tendons or [], full):
                if not isinstance(t, ServoTendon):
                    t.tension = t0 * ts
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
                if ts > ts_ok:
                    ts = ts_ok                # too much pull for this design yet: hold the tension
                elif beta > beta_ok:
                    # the sharper projection is what broke it: keep the last
                    # beta for now and try again later; stop sharpening
                    # after three refusals
                    beta, since, refused = beta_ok, 0, refused + 1
                    if refused >= 3:
                        beta_cap = beta_ok
                else:
                    x = 0.5 * (x + x_ok)      # backtrack toward the last design that converged
        if real is None:
            # no nearby design reaches the full stroke: keep the last one that did
            x, beta, stopped = x_ok, beta_ok, "the stroke could not be reached from the last design"
            break
        x_ok, beta_ok, ts_ok = x.copy(), beta, ts
        jb = real[len(etas) // 2][0]
        if j0 is None:
            # a grip starts at zero when the part does not reach the object yet
            j0 = max(abs(jb), 0.05 * sum(full), 1e-9)
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
        if tendons:
            rec["tensionScale"] = ts
        if len(etas) > 1:
            rec["robustOutputs"] = [float(r[0]) for r in real]
        history.append(rec)
        if on_iter is not None:
            on_iter(rec, xp)
        since += 1
        if beta < beta_cap and (since >= opts.beta_every or change < opts.tol * 2):
            beta, since = min(beta * 2, beta_cap), 0
        elif beta >= beta_cap and ts >= 1.0:
            if change < opts.tol:
                break
            if since >= 10 and len(history) > 10:
                old = history[-11]["objective"]
                if abs(obj - old) <= 2e-3 * abs(old):
                    break
    for t, t0 in zip(tendons or [], full):
        if not isinstance(t, ServoTendon):
            t.tension = t0
    if stopped:
        history[-1]["stopped"] = stopped
    if beta_cap < opts.beta_max:
        history[-1]["betaCap"] = beta_cap
    xt = _extrude((h @ x) / hs, columns, design)
    xp = _project(xt, beta, 0.5)
    xp[pres] = 1.0
    return MechanismResult(x=x, xphys=xp, output_displacement=float(history[-1]["outputDisplacement"]),
                           iterations=it, seconds=time.time() - t0, history=history)
