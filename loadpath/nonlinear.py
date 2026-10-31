"""
Large deflection and buckling, for parts that are meant to bend.

Everything in fem.py is linear: stiffness fixed at the undeformed shape,
strains small. That is right for a bracket and wrong for a TPU finger that
curls 60 degrees, a flexure hinge, a snap fit or a slender strut near
buckling. This module adds the two analyses those parts need, on the same
voxel hex mesh:

- `solve_nonlinear`: geometrically nonlinear static analysis. Total
  Lagrangian 8-node hexahedra (2x2x2 Gauss), a hyperelastic material
  (compressible neo-Hookean by default, St. Venant-Kirchhoff on request:
  both reduce to fem.py's linear element at small strain), and Newton's
  method on the residual with the full tangent (material + geometric
  stiffness). Load is applied in increments that grow when Newton converges
  quickly and halve when it does not. Loads are dead (they keep their
  direction as the part deforms). Supports can be pushed instead of loaded
  (`prescribed`), which follows a part through a snap-through that load
  control cannot pass, and reports the force it takes (`reaction`).
- `linear_buckling`: the classic eigenvalue estimate. Solve the linear
  problem, build the geometric stiffness of its stresses, and find load
  factors lambda with (K + lambda K_sigma) phi = 0. lambda < 1 means the
  load buckles the part; the modes say how.

`verify_mechanism` puts both behind one call for a compliant mechanism
design from mechanism.py: the solid cells of the result, the real input
force, and the output's travel compared with what the linear model
promised.

Linear hexahedra are stiff in bending (shear locking), and more so with few
cells across a member: about 3-4 cells across a flexure keeps the error to
a few percent of beam theory; one or two cells overestimate its stiffness
badly. Validation against the elastica and Euler's column is in
tests/test_nonlinear.py.

Units as everywhere in loadpath: mm, N, MPa.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import LinearOperator, eigsh, splu

from .fem import CORNERS, HexMesh, Solver, Stiffness, von_mises

__all__ = ["NonlinearResult", "BucklingResult", "Tendon", "nodes_near", "solve_nonlinear", "linear_buckling", "stroke_mechanism",
           "solid_submesh", "verify_mechanism"]

# Above this many dofs the Newton steps use CG + AMG instead of a direct
# factorization (SuperLU's fill on 3D hex meshes grows fast: 1.5 s at 20k
# dofs, 9 s at 40k). CG needs the tangent positive definite, so there a load
# step that lands on an unstable state is cut back, and the stability check
# runs LOBPCG with the multigrid preconditioner instead of shift-invert.
DIRECT_MAX_DOF = 30_000
_CHUNK = 1024   # elements per batch: big enough to vectorize, small enough to stay in cache
_VOIGT = [(0, 0), (1, 1), (2, 2), (0, 1), (1, 2), (2, 0)]   # fem.py order: xx yy zz xy yz zx
_VI = np.array([p[0] for p in _VOIGT])
_VJ = np.array([p[1] for p in _VOIGT])


def _dn(xi, eta, zeta):
    """Shape function gradients (8, 3) of the unit cube at a natural point."""
    s = 2.0 * CORNERS - 1.0
    dn = np.empty((8, 3))
    dn[:, 0] = 0.25 * s[:, 0] * (1 + eta * s[:, 1]) * (1 + zeta * s[:, 2])
    dn[:, 1] = 0.25 * s[:, 1] * (1 + xi * s[:, 0]) * (1 + zeta * s[:, 2])
    dn[:, 2] = 0.25 * s[:, 2] * (1 + xi * s[:, 0]) * (1 + eta * s[:, 1])
    return dn


_G = 1.0 / np.sqrt(3.0)
_GAUSS = [_dn(a, b, c) for a in (-_G, _G) for b in (-_G, _G) for c in (-_G, _G)]
_CENTER = _dn(0.0, 0.0, 0.0)


def _lame(nu):
    return nu / ((1 + nu) * (1 - 2 * nu)), 1.0 / (2 * (1 + nu))   # at E = 1


_I3 = np.eye(3)
# Products of shape function gradients at each Gauss point, for the tangent
# as one matrix product: W[g][(l, n), (a, b)] = dN_a/dX_l dN_b/dX_n (unit cube).
_W = np.stack([np.einsum("al,bn->lnab", g, g).reshape(9, 64) for g in _GAUSS])


def _material(F, nu, kind):
    """2nd Piola-Kirchhoff stress S (m, 3, 3) and the material tangent
    dS/dE as a 4th-order tensor (m, 3, 3, 3, 3), at unit modulus."""
    lam, mu = _lame(nu)
    C = np.einsum("mki,mkj->mij", F, F)
    if kind == "svk":
        E = 0.5 * (C - _I3)
        tr = np.trace(E, axis1=1, axis2=2)
        S = lam * tr[:, None, None] * _I3 + 2 * mu * E
        A = (lam * np.einsum("ij,kl->ijkl", _I3, _I3)
             + mu * (np.einsum("ik,jl->ijkl", _I3, _I3) + np.einsum("il,jk->ijkl", _I3, _I3)))
        return S, np.broadcast_to(A, (len(F), 3, 3, 3, 3))
    J = np.linalg.det(F)
    lnJ = np.log(np.maximum(J, 1e-12))
    Ci = np.linalg.inv(C)
    S = mu * (_I3 - Ci) + (lam * lnJ)[:, None, None] * Ci
    a = Ci[:, :, None, :, None] * Ci[:, None, :, None, :]           # Ci_ik Ci_jl
    A = lam * Ci[:, :, :, None, None] * Ci[:, None, None, :, :] \
        + (mu - lam * lnJ)[:, None, None, None, None] * (a + a.transpose(0, 1, 2, 4, 3))
    return S, A


def _element(ue, h, nu, kind, tangent):
    """Internal force (m, 24) and tangent (m, 24, 24) at unit modulus for
    elements with nodal displacements ue (m, 8, 3), and the smallest volume
    ratio J seen at a Gauss point (<= 0 means an element turned inside out).

    With P = F S the first Piola-Kirchhoff stress, f_ai = sum_g w P_il dN_al,
    and the tangent is K_aibj = sum_g w dN_al Q_iljn dN_bn with
    Q_iljn = F_ik A_klpn F_jp + S_ln delta_ij (material + geometric): a
    (m*9, 9) x (9, 64) matrix product per Gauss point instead of 24x24 products
    per element (all eight stacked into one)."""
    m = len(ue)
    w = h ** 3 / 8.0
    fint = np.zeros((m, 8, 3))
    qall = np.empty((m, 9, 8, 9)) if tangent else None
    jmin = np.inf
    for g, dn0 in enumerate(_GAUSS):
        dn = dn0 / h
        F = _I3 + np.einsum("mai,aj->mij", ue, dn)
        jmin = min(jmin, float(np.linalg.det(F).min()))
        S, A = _material(F, nu, kind)
        P = F @ S
        fint += w * (P @ dn.T).transpose(0, 2, 1)
        if tangent:
            fa = (F @ A.reshape(m, 3, 27)).reshape(m, 3, 3, 3, 3)
            Q = np.einsum("milpn,mjp->mijln", fa, F, optimize=True)
            Q += _I3[None, :, :, None, None] * S[:, None, None, :, :]
            qall[:, :, g, :] = Q.reshape(m, 9, 9)
    kt = None
    if tangent:
        kq = qall.reshape(m * 9, 72) @ (_W.reshape(72, 64) * (w / h ** 2))
        kt = kq.reshape(m, 3, 3, 8, 8).transpose(0, 3, 1, 4, 2).reshape(m, 24, 24)
    return fint.reshape(m, 24), kt, jmin


def _assemble(stiff, u, mod, nu, kind, tangent=True):
    mesh = stiff.mesh
    fint = np.zeros(mesh.n_dof)
    data = np.zeros(stiff.nnz) if tangent else None
    jmin = np.inf
    for s in range(0, mesh.n_elem, _CHUNK):
        el = np.arange(s, min(s + _CHUNK, mesh.n_elem))
        edof = mesh.edof[el]
        fe, ke, j = _element(u[edof].reshape(-1, 8, 3), mesh.pitch, nu, kind, tangent)
        jmin = min(jmin, j)
        me = mod[el]
        np.add.at(fint, edof.ravel(), (fe * me[:, None]).ravel())
        if tangent:
            stiff.add_elements(data, ke * me[:, None, None], el)
    k = stiff.finish(data) if tangent else None
    return fint, k, jmin


def _splu(k):
    # symmetric pattern: minimum degree on A^T + A, no pivoting off the diagonal
    return splu(k.tocsc(), permc_spec="MMD_AT_PLUS_A", diag_pivot_thresh=0.0, options={"SymmetricMode": True})


class _Linear:
    """Direct factorization for small and medium problems, CG + AMG above
    (the hierarchy is reused across Newton iterations until it goes stale)."""

    def __init__(self, mesh):
        self.direct = mesh.n_dof <= DIRECT_MAX_DOF
        self.iter = None if self.direct else Solver(mesh, rtol=1e-9, maxiter=2000)

    def solve(self, k, r):
        if self.direct:
            return _splu(k).solve(r)
        return self.iter.solve(k, r)

    def lowest_mode(self, k):
        """Smallest eigenvalue of a (symmetric) tangent and its mode."""
        n = k.shape[0]
        if self.direct:
            lu = _splu(k)
            op = LinearOperator((n, n), matvec=lu.solve, dtype=float)
            vals, vecs = eigsh(k, k=1, sigma=0.0, which="LM", OPinv=op, tol=1e-6)
            return float(vals[0]), vecs[:, 0]
        from scipy.sparse.linalg import lobpcg
        m = self.iter._precond(k)
        x0 = np.random.default_rng(0).standard_normal((n, 1))
        import warnings
        with warnings.catch_warnings():
            # only the sign matters here; a loose eigenvalue is fine
            warnings.simplefilter("ignore", UserWarning)
            vals, vecs = lobpcg(k, x0, M=m, largest=False, tol=1e-6, maxiter=300)
        return float(vals[0]), vecs[:, 0]


@dataclass
class NonlinearResult:
    u: np.ndarray                      # displacement per dof at the last converged load
    load_factor: float                 # fraction of the load (or stroke) reached; 1.0 when complete
    converged: bool                    # reached the full load
    history: list = field(default_factory=list)   # per converged step
    stress: np.ndarray | None = None   # element-center Cauchy von Mises, MPa
    strain: np.ndarray | None = None   # element-center largest principal stretch - 1
    message: str = ""


def _post(mesh, u, mod, nu, kind):
    ue = u[mesh.edof].reshape(-1, 8, 3)
    F = np.eye(3) + np.einsum("mai,aj->mij", ue, _CENTER / mesh.pitch)
    S, _ = _material(F, nu, kind)
    J = np.linalg.det(F)
    sig = np.einsum("mik,mkl,mjl->mij", F, S, F) / J[:, None, None] * mod[:, None, None]
    vm = von_mises(sig[:, _VI, _VJ])
    C = np.einsum("mki,mkj->mij", F, F)
    stretch = np.sqrt(np.maximum(np.linalg.eigvalsh(C)[:, -1], 0.0))
    return vm, stretch - 1.0


@dataclass
class Tendon:
    """A cable at constant tension through guides on the part, like a
    tendon in its channel: each point of `path` is either a fixed point in
    space (a float array of 3, e.g. the servo's spool or a palm pulley) or
    a guide on the part (an int array of node ids; the guide sits at their
    mean and moves with them). The cable runs straight between consecutive
    points and slides freely through the guides, so each guide is pulled
    toward both its neighbours along the deformed cable and the last point
    (the knot) toward the one before it. The load follows the part as it
    curls, which a dead force does not: on a curling finger a fixed-
    direction force gains moment arm, a tendon's stays in its channel.

    Constant tension makes it conservative (potential T * cable length), so
    the load stiffness is symmetric and Newton keeps its tangent symmetric.
    Tension scales with the load factor like `force`. Each step records the
    tendon's travel (how much cable the actuator has pulled in, mm)."""
    tension: float
    path: list

    def _setup(self, mesh):
        pts = []
        for p in self.path:
            a = np.asarray(p)
            if a.dtype.kind in "iu":
                ids = np.unique(a.ravel()).astype(np.int64)
                if len(ids) == 0:
                    raise ValueError("Tendon: a guide with no nodes")
                w = np.full(len(ids), 1.0 / len(ids))
                rows = np.repeat(np.arange(3), len(ids))
                cols = (3 * ids[None, :] + np.arange(3)[:, None]).ravel()
                g = sp.csr_matrix((np.tile(w, 3), (rows, cols)), shape=(3, mesh.n_dof))
                pts.append((mesh.node_xyz[ids].mean(axis=0), g))
            else:
                if a.shape != (3,):
                    raise ValueError("Tendon: a fixed point is 3 floats; a guide is an int array of node ids")
                pts.append((a.astype(float), None))
        if len(pts) < 2:
            raise ValueError("Tendon: the path needs at least two points")
        self._segs = []
        for (xa, ga), (xb, gb) in zip(pts[:-1], pts[1:]):
            gd = (gb if gb is not None else 0) - (ga if ga is not None else 0)
            if isinstance(gd, int):
                continue                       # between two fixed points: no load on the part
            self._segs.append((xb - xa, sp.csr_matrix(gd)))
        self._len0 = sum(np.linalg.norm(x0) for x0, _ in self._segs)

    def _load(self, u, scale):
        """Force on the part (per dof), load stiffness, and cable length
        (over the segments that touch the part)."""
        t = self.tension * scale
        n = u.shape[0]
        f = np.zeros(n)
        k = sp.csr_matrix((n, n))
        length = 0.0
        for d0, gd in self._segs:
            d = d0 + gd @ u
            ell = float(np.linalg.norm(d))
            e = d / ell
            length += ell
            f -= t * (gd.T @ e)
            k = k + gd.T @ sp.csr_matrix(t * (np.eye(3) - np.outer(e, e)) / ell) @ gd
        return f, k.tocsr(), length


def nodes_near(mesh: HexMesh, point, radius: float) -> np.ndarray:
    """Node ids within `radius` of `point`: a tendon guide or knot."""
    d = np.linalg.norm(mesh.node_xyz - np.asarray(point, float), axis=1)
    ids = np.nonzero(d <= radius)[0]
    if len(ids) == 0:
        raise ValueError(f"no nodes within {radius} of {point}")
    return ids


def solve_nonlinear(mesh: HexMesh, nu: float, modulus, fixed: np.ndarray, force: np.ndarray | None = None,
                    prescribed: tuple | None = None, *, material: str = "neo-hookean", steps: int = 10,
                    max_step: float = 0.25, min_step: float = 1e-4, rtol: float = 1e-7, max_newton: int = 25,
                    springs: np.ndarray | None = None, tendons: list | None = None, stability: bool = True,
                    probes: dict | None = None,
                    on_step=None) -> NonlinearResult:
    """Geometrically nonlinear static solve.

    modulus: E, scalar or per element (MPa). fixed: bool per dof, held at 0.
    force: force per dof (N) at full load. prescribed: (dofs, values) moved to
    `values` (mm) at full load, e.g. an actuator's stroke. probes: name ->
    weight per dof; each step records weight . u (with unit-sum weights over
    a face along a direction, the face's mean travel along it). tendons:
    Tendon loads (cables through guides) that follow the deformed part.
    springs:
    grounded linear springs, N/mm per dof (an actuator's compliance, a
    workpiece pushing back).

    stability: after each step, check the tangent's lowest eigenvalue. Newton
    happily converges to an equilibrium that is unstable (a column kept
    straight past its buckling load, a snap-through that hasn't snapped);
    when the tangent has a negative eigenvalue the solve kicks the part
    along that mode and finds the stable state nearby, as a real part would
    (a print is never perfectly straight).

    Each history entry has the load factor, Newton iterations, the probes,
    with prescribed dofs the reaction there (the force the actuator pushes
    with, N, as a 3-vector summed over the prescribed nodes), and with
    stability the tangent's lowest eigenvalue and whether the step had to
    leave an unstable branch ("branchSwitch").
    """
    n = mesh.n_dof
    mod = np.broadcast_to(np.asarray(modulus, float), (mesh.n_elem,)).copy()
    f = np.zeros(n) if force is None else np.asarray(force, float)
    cons = np.asarray(fixed, bool).copy()
    pd, pv = (np.zeros(0, np.int64), np.zeros(0)) if prescribed is None else \
        (np.asarray(prescribed[0], np.int64), np.asarray(prescribed[1], float))
    cons[pd] = True
    stiff = Stiffness(mesh, nu, cons)
    lin = _Linear(mesh)
    probes = probes or {}
    pnodes = np.unique(pd // 3)
    size = float(np.linalg.norm(np.ptp(mesh.node_xyz, axis=0)))
    # Don't let one Newton update move any node more than a fraction of the
    # part's size (a wild first iterate after a big step).
    cap = 0.25 * size
    kick_step = min(0.02, 0.25 / steps)
    seeded = False

    spr = None
    if springs is not None:
        sd = np.where(cons, 0.0, np.asarray(springs, float))
        spr = sp.diags(sd).tocsr()

    tendons = tendons or []
    keep_free = sp.diags((~cons).astype(float))
    for t in tendons:
        t._setup(mesh)

    def newton(v, target):
        rnorm0 = None
        for its in range(1, max_newton + 1):
            fint, k, jmin = _assemble(stiff, v, mod, nu, material)
            if not np.isfinite(jmin) or jmin <= 0:
                return None
            if spr is not None:
                fint = fint + spr @ v
                k = (k + spr).tocsr()
            fext = target * f
            for t in tendons:
                ft, kt, _ = t._load(v, target)
                fext = fext + ft
                # constrained rows and columns stay identity, as Stiffness leaves them
                k = (k + keep_free @ kt @ keep_free).tocsr()
            r = fext - fint
            r[cons] = 0.0
            ref = max(np.linalg.norm(fext), np.linalg.norm(fint[pd]) if len(pd) else 0.0, 1e-30)
            rnorm = np.linalg.norm(r)
            if rnorm <= rtol * ref:
                return v, its, fint, k
            if rnorm0 is None:
                rnorm0 = rnorm
            elif its > 6 and rnorm > rnorm0:
                return None          # diverging
            try:
                d = lin.solve(k, r)
            except RuntimeError:
                return None
            if not np.all(np.isfinite(d)):
                return None
            d[cons] = 0.0
            big = np.abs(d).max()
            if big > cap:
                d *= cap / big
            v = v + d
        return None

    u = np.zeros(n)
    lam, dlam = 0.0, min(1.0 / steps, max_step)
    du_prev, dlam_prev = None, None
    history = []
    message = ""
    while lam < 1.0 - 1e-12:
        dlam = min(dlam, 1.0 - lam)
        target = lam + dlam
        # Predictor: extrapolate the last step's increment (the prescribed
        # dofs move exactly by it).
        v = u.copy()
        if du_prev is not None:
            v += du_prev * (dlam / dlam_prev)
        v[pd] = target * pv
        out = newton(v, target)
        if out is None:
            if dlam <= min_step:
                message = (f"stopped at {lam:.3f} of the load: no equilibrium past it under load control "
                           "(a limit point, i.e. buckling or snap-through, or the part folded). "
                           "Try a prescribed displacement instead of a force.")
                break
            dlam *= 0.5
            continue
        v, its, fint, k = out
        rec = {"loadFactor": target, "newton": its}
        if stability:
            lmin, phi = lin.lowest_mode(k)
            rec["lowestEigenvalue"] = lmin
            tol = 1e-9 * abs(k.diagonal()).max()
            if lmin < -tol:
                # An unstable equilibrium. First creep up on it: with any
                # imperfection the stable path bends away smoothly and small
                # steps follow it. At the bifurcation itself (a perfect
                # part), kick along the unstable mode until Newton lands on
                # a stable state. CG can't solve with an indefinite tangent,
                # so on large meshes seed a small imperfection instead (a
                # force along the unstable mode, 0.1% of the load) and
                # redo the step from the last stable state.
                if not lin.direct and not seeded:
                    scale = max(np.linalg.norm(f), np.linalg.norm(fint[pd]) / max(target, 1e-12) if len(pd) else 0.0)
                    f = f + 1e-3 * scale * np.where(cons, 0.0, phi) / np.linalg.norm(phi)
                    seeded = True
                    dlam *= 0.5
                    continue
                if dlam > kick_step:
                    dlam *= 0.5
                    continue
                phi = phi / np.abs(phi).max()
                for amp in ((0.01, 0.03, 0.1, 0.3) if lin.direct else ()):
                    found = None
                    for sgn in (1.0, -1.0):
                        o = newton(v + sgn * amp * size * phi, target)
                        if o is not None:
                            l2 = lin.lowest_mode(o[3])[0]
                            if l2 >= -tol:
                                found = o, l2
                                break
                    if found is not None:
                        (v, its2, fint, k), l2 = found
                        rec["newton"] += its2
                        rec["branchSwitch"] = True
                        rec["lowestEigenvalue"] = l2
                        break
                else:
                    rec["unstable"] = True
        du_prev, dlam_prev = (None if rec.get("branchSwitch") else v - u), dlam
        u, lam = v, target
        for name, wv in probes.items():
            rec[name] = float(wv @ u)
        if len(pd):
            rec["reaction"] = fint.reshape(-1, 3)[pnodes].sum(axis=0).tolist()
        for i, t in enumerate(tendons):
            rec[f"tendon{i}Travel"] = t._len0 - t._load(u, lam)[2]
        history.append(rec)
        if on_step is not None:
            on_step(rec, u)
        if rec["newton"] <= 5:
            dlam = min(dlam * 1.5, max_step)
    vm, strain = _post(mesh, u, mod, nu, material)
    done = lam >= 1.0 - 1e-12
    if done:
        message = ""
    if seeded:
        message = (message + " " if message else "") + \
            "The part went unstable (buckled); a 0.1% imperfection along the buckling mode was added to follow it."
    return NonlinearResult(u=u, load_factor=lam, converged=done, history=history, stress=vm, strain=strain,
                           message=message)


@dataclass
class BucklingResult:
    factors: np.ndarray        # load multipliers that buckle the part, smallest first
    modes: np.ndarray          # (n_modes, n_dof), each scaled to a largest nodal travel of 1
    u: np.ndarray              # the linear displacement under the reference load


def _geometric(stiff, u, mod, nu):
    """Geometric stiffness of the linear stresses in displacement u (fixed
    dofs cleared, zero on their diagonal)."""
    mesh = stiff.mesh
    h, w = mesh.pitch, mesh.pitch ** 3 / 8.0
    lam, mu = _lame(nu)
    data = np.zeros(stiff.nnz)
    for s in range(0, mesh.n_elem, _CHUNK):
        el = np.arange(s, min(s + _CHUNK, mesh.n_elem))
        m = len(el)
        ue = u[mesh.edof[el]].reshape(m, 8, 3)
        kq = np.zeros((m, 64))
        for g, dn0 in enumerate(_GAUSS):
            grad = np.einsum("mai,aj->mij", ue, dn0 / h)
            eps = 0.5 * (grad + grad.transpose(0, 2, 1))
            sig = (lam * np.trace(eps, axis1=1, axis2=2)[:, None, None] * _I3 + 2 * mu * eps) * mod[el, None, None]
            kq += (w / h ** 2) * (sig.reshape(m, 9) @ _W[g])
        kg = np.einsum("mab,ij->maibj", kq.reshape(m, 8, 8), _I3).reshape(m, 24, 24)
        stiff.add_elements(data, kg, el)
    return stiff.finish(data, diag=0.0)


def linear_buckling(mesh: HexMesh, nu: float, modulus, fixed: np.ndarray, force: np.ndarray,
                    n_modes: int = 3, springs: np.ndarray | None = None) -> BucklingResult:
    """Smallest positive load factors lambda at which lambda * force buckles
    the part (linear eigenvalue buckling: exact for a perfect column, an
    upper bound for real parts with imperfections; take a margin of 2-3 on
    printed slender members). Below 1, the load as given buckles it."""
    mod = np.broadcast_to(np.asarray(modulus, float), (mesh.n_elem,)).copy()
    stiff = Stiffness(mesh, nu, fixed)
    k = stiff.assemble(mod)
    if springs is not None:
        k = (k + sp.diags(np.where(fixed, 0.0, springs))).tocsr()
    if mesh.n_dof <= DIRECT_MAX_DOF:
        kinv = _splu(k).solve
    else:
        sol = Solver(mesh, rtol=1e-10, maxiter=3000)
        kinv = lambda r: sol.solve(k, r)  # noqa: E731
    f = np.asarray(force, float).copy()
    f[np.asarray(fixed, bool)] = 0.0
    u = kinv(f)
    kg = _geometric(stiff, u, mod, nu)
    # K phi = lambda (-Kg) phi  <=>  (-Kg) phi = (1 / lambda) K phi: the
    # largest positive 1/lambda are the smallest positive buckling loads.
    n = mesh.n_dof
    nm = min(n_modes, n - 2)
    minv = LinearOperator((n, n), matvec=kinv, dtype=float)
    mu, vec = eigsh(-kg, k=nm, M=k, Minv=minv, which="LA", tol=1e-8)
    order = np.argsort(-mu)
    mu, vec = mu[order], vec[:, order]
    pos = mu > 1e-12
    factors = np.full(len(mu), np.inf)
    factors[pos] = 1.0 / mu[pos]
    modes = vec.T.copy()
    for m in modes:
        big = np.abs(m.reshape(-1, 3)).max() or 1.0
        m /= big
    return BucklingResult(factors=factors, modes=modes, u=u)


def solid_submesh(mesh: HexMesh, keep: np.ndarray, fixed: np.ndarray, vectors: list):
    """The mesh of elements `keep`, minus any piece not connected to a
    support (it would float). Returns (submesh, element index into the old
    mesh, fixed, vectors) with the per-dof arrays carried over to the new
    nodes; load on nodes that fell away is dropped."""
    from scipy.sparse.csgraph import connected_components
    el = np.nonzero(keep)[0]
    sub = HexMesh(mesh.elem_ijk[el], mesh.pitch, mesh.origin)
    old = mesh.node_id(sub.node_ijk)
    fx = np.asarray(fixed, bool).reshape(-1, 3)[old]
    # elements joined through shared nodes
    ne = sub.n_elem
    inc = sp.csr_matrix((np.ones(ne * 8), (np.repeat(np.arange(ne), 8), sub.enode.ravel())), shape=(ne, sub.n_nodes))
    _, lab = connected_components(inc @ inc.T, directed=False)
    # a piece is held only if its supports restrain all three directions: a
    # plate's symmetry plane (one direction) does not stop it floating
    held = np.zeros((lab.max() + 1, 3), bool)
    np.logical_or.at(held, lab, fx[sub.enode].any(axis=1))
    hold = held[lab].all(axis=1)
    if not hold.all():
        el = el[hold]
        sub = HexMesh(mesh.elem_ijk[el], mesh.pitch, mesh.origin)
        old = mesh.node_id(sub.node_ijk)
        fx = np.asarray(fixed, bool).reshape(-1, 3)[old]
    out = [np.asarray(v, float).reshape(-1, 3)[old].ravel() for v in vectors]
    return sub, el, fx.ravel(), out


def verify_mechanism(domain, xphys: np.ndarray, modulus: float, nu: float, input_case: int = 0,
                     output_case: int = 1, threshold: float = 0.5, input_scale: float = 1.0,
                     steps: int = 10, buckling: bool = True, material: str = "neo-hookean",
                     k_in: float = 0.0, k_out: float = 0.0) -> dict:
    """Check a compliant mechanism design (mechanism.optimize_mechanism) with
    large deflections: the solid cells (xphys > threshold) under the input
    case's force, output travel along the output case's direction, compared
    with the linear answer on the same solid mesh. With buckling, also the
    load factor at which the input force buckles the design.

    k_in, k_out: the optimization's springs (N/mm) at the input and output.
    Leave them at 0 for the free stroke under the actuator's force (a finger
    before it touches anything); give the optimization's values to check
    the case it was designed for.
    """
    from .mechanism import _spring_diag
    mesh = domain.mesh
    fin = domain.loads[input_case] * input_scale
    lout = domain.loads[output_case]
    keep = np.asarray(xphys) > threshold
    sub, el, fx, (f, lo) = solid_submesh(mesh, keep, domain.fixed, [fin, lout])
    dirn = lo.reshape(-1, 3).sum(axis=0)
    nrm = np.linalg.norm(dirn)
    if nrm == 0:
        raise ValueError("the output region has no solid cells under it")
    wts = np.abs(lo.reshape(-1, 3)).sum(axis=1) > 0
    lw = (wts[:, None] * (dirn / nrm)[None, :]).ravel() / max(wts.sum(), 1)

    report = {"elements": int(sub.n_elem), "inputForceKept": float(np.linalg.norm(f.reshape(-1, 3).sum(axis=0))
                                                                    / max(np.linalg.norm(fin.reshape(-1, 3).sum(axis=0)), 1e-30))}
    sd = _spring_diag(f, k_in, sub.n_dof) + _spring_diag(lo, k_out, sub.n_dof)
    sd[fx] = 0.0
    stiff = Stiffness(sub, nu, fx)
    k = (stiff.assemble(np.full(sub.n_elem, modulus)) + sp.diags(sd)).tocsr()
    ul = _Linear(sub).solve(k, np.where(fx, 0.0, f))
    report["linearOutput"] = float(lw @ ul)
    r = solve_nonlinear(sub, nu, modulus, fx, f, material=material, steps=steps, springs=sd, probes={"output": lw})
    report["nonlinearOutput"] = float(lw @ r.u)
    report["loadFactorReached"] = r.load_factor
    report["converged"] = r.converged
    report["maxStress"] = float(r.stress.max())
    report["maxStrain"] = float(r.strain.max())
    report["history"] = r.history
    if r.message:
        report["message"] = r.message
    if report["linearOutput"]:
        report["nonlinearOverLinear"] = report["nonlinearOutput"] / report["linearOutput"]
    if buckling:
        b = linear_buckling(sub, nu, modulus, fx, f, n_modes=1, springs=sd)
        report["bucklingFactor"] = float(b.factors[0])
    report["u"] = r.u
    report["elementIndex"] = el
    return report


def stroke_mechanism(domain, xphys: np.ndarray, modulus: float, nu: float, stroke: float, input_case: int = 0,
                     output_case: int = 1, k_out: float = 0.0, threshold: float = 0.5, steps: int = 10,
                     material: str = "neo-hookean") -> dict:
    """Drive a compliant mechanism the way a servo or a lead screw does: move
    the input region `stroke` mm along the input force's direction
    (prescribed, so a snap-through cannot run away) and follow the output
    with large deflections. k_out (N/mm): the workpiece, a grounded spring
    on the output; 0 for the free stroke.

    Returns the output's travel along its direction (nonlinear and linear),
    the actuator force it took (N, along the stroke), the grip force on the
    workpiece (k_out x travel), the geometric advantage (output travel /
    stroke) and mechanical advantage (grip force / actuator force), the
    peak stress and strain, and the per-step history."""
    from .mechanism import _spring_diag
    mesh = domain.mesh
    keep = np.asarray(xphys) > threshold
    sub, el, fx, (f, lo) = solid_submesh(mesh, keep, domain.fixed, [domain.loads[input_case],
                                                                    domain.loads[output_case]])
    fv = f.reshape(-1, 3)
    loaded = np.flatnonzero(np.abs(fv).sum(axis=1) > 0)
    if len(loaded) == 0:
        raise ValueError("the input region has no solid cells under it")
    din = fv.sum(axis=0)
    din /= np.linalg.norm(din)
    pd, pv = [], []
    for c in range(3):
        if abs(din[c]) > 1e-9:
            dofs = 3 * loaded + c
            dofs = dofs[~fx[dofs]]
            pd.append(dofs)
            pv.append(np.full(len(dofs), stroke * din[c]))
    pd, pv = np.concatenate(pd), np.concatenate(pv)
    dirn = lo.reshape(-1, 3).sum(axis=0)
    nrm = np.linalg.norm(dirn)
    if nrm == 0:
        raise ValueError("the output region has no solid cells under it")
    wts = np.abs(lo.reshape(-1, 3)).sum(axis=1) > 0
    lw = (wts[:, None] * (dirn / nrm)[None, :]).ravel() / max(wts.sum(), 1)
    sd = _spring_diag(lo, k_out, sub.n_dof)
    sd[fx] = 0.0
    # linear answer on the same mesh: K u = 0 with the input moved
    cons = fx.copy()
    cons[pd] = True
    stiff = Stiffness(sub, nu, cons)
    k_free = Stiffness(sub, nu, np.zeros(sub.n_dof, bool)).assemble(np.full(sub.n_elem, modulus))
    kc = (stiff.assemble(np.full(sub.n_elem, modulus)) + sp.diags(np.where(cons, 0.0, sd))).tocsr()
    up = np.zeros(sub.n_dof)
    up[pd] = pv
    rhs = np.where(cons, up, -(k_free @ up))
    ul = _Linear(sub).solve(kc, rhs)
    lin_force = float(((k_free @ ul).reshape(-1, 3)[np.unique(pd // 3)].sum(axis=0)) @ din)
    r = solve_nonlinear(sub, nu, modulus, fx, None, prescribed=(pd, pv), material=material, steps=steps,
                        springs=sd, probes={"output": lw})
    out_nl = float(lw @ r.u)
    force = float(np.asarray(r.history[-1]["reaction"]) @ din) if r.history else float("nan")
    rep = {"elements": int(sub.n_elem), "stroke": stroke, "kOut": k_out, "converged": r.converged,
           "strokeReached": r.load_factor * stroke,
           "linearOutput": float(lw @ ul), "output": out_nl, "linearActuatorForce": lin_force,
           "actuatorForce": force, "geometricAdvantage": out_nl / (r.load_factor * stroke) if r.load_factor else 0.0,
           "maxStress": float(r.stress.max()), "maxStrain": float(r.strain.max()),
           "history": [{"stroke": h["loadFactor"] * stroke, "output": h.get("output"),
                        "actuatorForce": float(np.asarray(h["reaction"]) @ din) if "reaction" in h else None,
                        **({"branchSwitch": True} if h.get("branchSwitch") else {})} for h in r.history]}
    if k_out > 0:
        rep["gripForce"] = k_out * out_nl
        rep["mechanicalAdvantage"] = rep["gripForce"] / force if force else None
    if r.message:
        rep["message"] = r.message
    return rep
