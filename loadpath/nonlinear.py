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
  quickly and halve when it does not; a Newton update that would turn a
  cell inside out is halved instead of failing the step. Elements are
  built by a compiled kernel on all cores (nlkernel.py). Loads are dead (they keep their
  direction as the part deforms). Supports can be pushed instead of loaded
  (`prescribed`), which follows a part through a snap-through that load
  control cannot pass, and reports the force it takes (`reaction`).
- `linear_buckling`: the classic eigenvalue estimate. Solve the linear
  problem, build the geometric stiffness of its stresses, and find load
  factors lambda with (K + lambda K_sigma) phi = 0. lambda < 1 means the
  load buckles the part; the modes say how.

Contact (contact.py: rigid planes, spheres, cylinders and boxes, fixed or
moving with the load, and the part's surface on itself) and tendons
(cables through guides at a tension, or wound in by a servo) go in as
`contact=` and `tendons=`; both follow the deformed part.
`large_deflection_report` runs all of it from a field graph's
`largeDeflection` block.

`verify_mechanism` puts both behind one call for a compliant mechanism
design from mechanism.py: the solid cells of the result, the real input
force, and the output's travel compared with what the linear model
promised.

Plain linear hexahedra are stiff in bending (shear locking): one or two
cells across a flexure overestimate its stiffness badly. As in fem.py, the
element carries Wilson's nine incompatible bubble modes, here as an enhanced
displacement gradient (Simo-Armero) condensed element by element at each
Newton iteration, which removes the locking; at small strain it is fem.py's
linear element exactly. `element="plain"` (or LOADPATH_HEX8=plain) gives the
old trilinear element. mechanism_nl.py's optimizer still uses the plain one. Validation against the elastica and Euler's column is in
tests/test_nonlinear.py.

Units as everywhere in loadpath: mm, N, MPa.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import LinearOperator, eigsh, splu

from .fem import CORNERS, HexMesh, Solver, Stiffness, _hex8_parts, hex8_element, von_mises

__all__ = ["NonlinearResult", "BucklingResult", "Tendon", "nodes_near", "solve_nonlinear", "linear_buckling", "stroke_mechanism",
           "large_deflection_report",
           "solid_submesh", "verify_mechanism"]

# Above this many dofs the Newton steps use CG + AMG instead of a direct
# factorization (SuperLU's fill on 3D hex meshes grows fast: 1.5 s at 20k
# dofs, 9 s at 40k). CG needs the tangent positive definite, so there a load
# step that lands on an unstable state is cut back, and the stability check
# runs LOBPCG with the multigrid preconditioner instead of shift-invert.
DIRECT_MAX_DOF = 30_000
# Step control. A step that converged in at most GROW_NEWTON iterations
# (assemblies, the converged one included) lets the next one grow 1.5x.
# A retry with line search (after plain Newton failed a contact step) that
# has not halved its best residual in STALL_NEWTON iterations is chattering
# (contacts switching on and off) and gives up the step: a shorter step is
# cheaper than more iterations. (Plain Newton gets its full max_newton: a
# contact plateau of ten or more iterations often ends in convergence.)
GROW_NEWTON = 6
STALL_NEWTON = 8
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
# The incompatible-mode element: three more "nodes" whose rows are the
# gradients of the bubbles 1 - xi_m^2 on the unit cube (-4 xi_m along axis m;
# fem._b_bubble's modes), so their amplitudes alpha (3 modes x 3 components)
# enter F like nodal displacements.
_GAUSS_IM = [np.vstack([dn, np.diag(-4.0 * np.array(p))]) for dn, p in
             zip(_GAUSS, [(a, b, c) for a in (-_G, _G) for b in (-_G, _G) for c in (-_G, _G)])]
_W_IM = np.stack([np.einsum("al,bn->lnab", g, g).reshape(9, 121) for g in _GAUSS_IM])


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


def _gauss_terms(ux, grads, wmat, h, nu, kind, tangent):
    """Internal force (m, 3n) and tangent (m, 3n, 3n) at unit modulus for
    elements with generalized displacements ux (m, n, 3), whose deformation
    gradient at Gauss point g is I + sum_a ux_a (x) grads[g][a] / h, and the
    smallest volume ratio J seen at a Gauss point.

    With P = F S the first Piola-Kirchhoff stress, f_ai = sum_g w P_il G_al,
    and the tangent is K_aibj = sum_g w G_al Q_iljn G_bn with
    Q_iljn = F_ik A_klpn F_jp + S_ln delta_ij (material + geometric): a
    (m*9, 9) x (9, n^2) matrix product per Gauss point instead of 3n x 3n
    products per element (all eight stacked into one)."""
    m, n = ux.shape[:2]
    w = h ** 3 / 8.0
    fint = np.zeros((m, n, 3))
    qall = np.empty((m, 9, 8, 9)) if tangent else None
    jmin = np.inf
    for g, dn0 in enumerate(grads):
        dn = dn0 / h
        F = _I3 + np.einsum("mai,aj->mij", ux, dn)
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
        kq = qall.reshape(m * 9, 72) @ (wmat.reshape(72, n * n) * (w / h ** 2))
        kt = kq.reshape(m, 3, 3, n, n).transpose(0, 3, 1, 4, 2).reshape(m, 3 * n, 3 * n)
    return fint.reshape(m, 3 * n), kt, jmin


def _element(ue, h, nu, kind, tangent, alpha=None):
    """Internal force (m, 24) and tangent (m, 24, 24) at unit modulus for
    elements with nodal displacements ue (m, 8, 3), and the smallest volume
    ratio J seen at a Gauss point (<= 0 means an element turned inside out).

    alpha None: the plain trilinear hexahedron. Otherwise (m, 3, 3), the
    amplitudes of the nine incompatible modes (fem.hex8_element's bubbles,
    here as an enhanced displacement gradient, Simo-Armero): the element
    takes one local Newton step on them, condenses them out, and returns
    (fint, kt, jmin, alpha_new, R_alpha (m, 9)). fint is the condensed
    residual R_u - K_ua K_aa^-1 R_a, i.e. R_u at alpha_new to first order,
    so the global Newton on u stays quadratic. At u = 0 the condensed
    tangent is fem.py's linear incompatible-mode element exactly."""
    if alpha is None:
        return _gauss_terms(ue, _GAUSS, _W, h, nu, kind, tangent)
    m = len(ue)
    ux = np.concatenate([ue, alpha], axis=1)               # (m, 11, 3)
    fx, kx, jmin = _gauss_terms(ux, _GAUSS_IM, _W_IM, h, nu, kind, True)
    ru, ra = fx[:, :24], fx[:, 24:]
    kua, kaa = kx[:, :24, 24:], kx[:, 24:, 24:]
    sol = np.linalg.solve(kaa, np.concatenate([ra[:, :, None], kua.transpose(0, 2, 1)], axis=2))
    da, kaa_kau = sol[:, :, 0], sol[:, :, 1:]
    fint = ru - np.einsum("mij,mj->mi", kua, da)
    kt = kx[:, :24, :24] - kua @ kaa_kau if tangent else None
    alpha_new = alpha - da.reshape(m, 3, 3)
    return fint, kt, jmin, alpha_new, ra, kaa_kau


def _assemble(stiff, u, mod, nu, kind, tangent=True, alpha=None, coupling=None):
    """Global internal force and tangent; with alpha (n_elem, 3, 3), the
    incompatible-mode element, also (alpha_new, |R_alpha|) with R_alpha
    weighted by the element moduli as the forces are."""
    mesh = stiff.mesh
    from . import nlkernel
    if nlkernel.available():
        grads = np.stack(_GAUSS_IM if alpha is not None else _GAUSS)
        fe, ke, jm, a_new, ra, cpl = nlkernel.elements(u, mesh.edof, alpha, mod, mesh.pitch, nu, kind, tangent, grads)
        fint, data = nlkernel.scatter(mesh.n_dof, stiff.nnz, stiff._slots, mesh.edof, fe, ke, tangent)
        k = stiff.finish(data) if tangent else None
        jmin = float(jm.min()) if len(jm) else np.inf
        if alpha is None:
            return fint, k, jmin
        if coupling is not None:
            coupling[:] = cpl
        return fint, k, jmin, a_new, float(np.sqrt(((ra * mod[:, None]) ** 2).sum()))
    fint = np.zeros(mesh.n_dof)
    data = np.zeros(stiff.nnz) if tangent else None
    jmin = np.inf
    a_new = None if alpha is None else np.empty_like(alpha)
    ra2 = 0.0
    for s in range(0, mesh.n_elem, _CHUNK):
        el = np.arange(s, min(s + _CHUNK, mesh.n_elem))
        edof = mesh.edof[el]
        ue = u[edof].reshape(-1, 8, 3)
        me = mod[el]
        if alpha is None:
            fe, ke, j = _element(ue, mesh.pitch, nu, kind, tangent)
        else:
            fe, ke, j, a_new[el], ra, cpl = _element(ue, mesh.pitch, nu, kind, tangent, alpha[el])
            if coupling is not None:
                coupling[el] = cpl
            ra2 += float(((ra * me[:, None]) ** 2).sum())
        jmin = min(jmin, j)
        np.add.at(fint, edof.ravel(), (fe * me[:, None]).ravel())
        if tangent:
            stiff.add_elements(data, ke * me[:, None, None], el)
    k = stiff.finish(data) if tangent else None
    if alpha is None:
        return fint, k, jmin
    return fint, k, jmin, a_new, np.sqrt(ra2)


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
            # a fixed start vector: ARPACK's random one made the branch
            # a perfect column takes (and so whether it found it) vary run to run
            v0 = np.random.default_rng(0).standard_normal(n)
            vals, vecs = eigsh(k, k=1, sigma=0.0, which="LM", OPinv=op, tol=1e-6, v0=v0)
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
        segs = []
        for (xa, ga), (xb, gb) in zip(pts[:-1], pts[1:]):
            gd = (gb if gb is not None else 0) - (ga if ga is not None else 0)
            if isinstance(gd, int):
                continue                       # between two fixed points: no load on the part
            segs.append((xb - xa, sp.csr_matrix(gd)))
        self._segs = segs
        self._len0 = sum(np.linalg.norm(x0) for x0, _ in segs)
        # all segments at once: d = d0 + G u, (S, 3)
        self._d0 = np.array([x0 for x0, _ in segs]).reshape(-1, 3)
        self._G = sp.vstack([gd for _, gd in segs]).tocsr() if segs else sp.csr_matrix((0, mesh.n_dof))
        S = len(segs)
        self._bi = (3 * np.arange(S)[:, None, None] + np.arange(3)[None, :, None] + 0 * np.arange(3)[None, None, :]).ravel()
        self._bj = (3 * np.arange(S)[:, None, None] + 0 * np.arange(3)[None, :, None] + np.arange(3)[None, None, :]).ravel()

    def _load(self, u, scale):
        """Force on the part (per dof), load stiffness, and cable length
        (over the segments that touch the part)."""
        t = self.tension * scale
        n = u.shape[0]
        S = len(self._d0)
        if S == 0:
            return np.zeros(n), sp.csr_matrix((n, n)), 0.0
        d = self._d0 + (self._G @ u).reshape(S, 3)
        ell = np.linalg.norm(d, axis=1)
        e = d / ell[:, None]
        f = -t * (self._G.T @ e.ravel())
        blocks = t * (np.eye(3)[None] - np.einsum("si,sj->sij", e, e)) / ell[:, None, None]
        mb = sp.csr_matrix((blocks.ravel(), (self._bi, self._bj)), shape=(3 * S, 3 * S))
        k = (self._G.T @ mb @ self._G).tocsr()
        return f, k, float(ell.sum())


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
                    springs: np.ndarray | None = None, tendons: list | None = None, contact: list | None = None,
                    stability: bool = True, probes: dict | None = None,
                    on_step=None, element: str | None = None) -> NonlinearResult:
    """Geometrically nonlinear static solve.

    modulus: E, scalar or per element (MPa). fixed: bool per dof, held at 0.
    force: force per dof (N) at full load. prescribed: (dofs, values) moved to
    `values` (mm) at full load, e.g. an actuator's stroke. probes: name ->
    weight per dof; each step records weight . u (with unit-sum weights over
    a face along a direction, the face's mean travel along it). tendons:
    Tendon loads (cables through guides) that follow the deformed part, at
    constant tension, or mechanism_nl.ServoTendon (a servo winding in a
    stroke of cable through a compliant cable: displacement driven).
    contact: contact.RigidObstacle (a rigid plane, sphere, cylinder or box,
    fixed or moving with the load) and contact.SelfContact (the part's
    surface on itself), penalty, frictionless or with Coulomb friction
    (`friction=mu`: sticks until the tangential force reaches mu times the
    normal force, then slides; history dependent, updated at each converged
    step). Each step records each contact's force, nodes touching and
    deepest penetration, with friction the friction force and the nodes
    sliding.
    springs:
    grounded linear springs, N/mm per dof (an actuator's compliance, a
    workpiece pushing back).

    stability: after each step, check the tangent's lowest eigenvalue. Newton
    happily converges to an equilibrium that is unstable (a column kept
    straight past its buckling load, a snap-through that hasn't snapped);
    when the tangent has a negative eigenvalue the solve kicks the part
    along that mode and finds the stable state nearby, as a real part would
    (a print is never perfectly straight).

    element: "im" (fem.hex8_element's default) or "plain". The incompatible
    modes take bending without shear locking, so one or two cells across a
    flexure no longer read it too stiff.

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
    element = element or hex8_element()
    stiff = Stiffness(mesh, nu, cons, element=element)
    lin = _Linear(mesh)
    # incompatible-mode amplitudes per element at the last converged state
    alpha = np.zeros((mesh.n_elem, 3, 3)) if element == "im" else None
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
    from .contact import SelfContact
    contact = contact or []
    for c in contact:
        if isinstance(c, SelfContact):
            c._setup(mesh, mod, fixed=cons)
        else:
            c._setup(mesh, mod)
    loads = tendons + contact
    if contact:
        # active sets settle over a few iterations, and a node sliding from
        # one face to the next leaves a residual of about 1e-7 of the
        # contact force: converge to 1e-5 of the load (the penalty itself
        # is a far bigger approximation)
        max_newton = max(max_newton, 40)
        rtol = max(rtol, 1e-5)
    # Coulomb friction's sliding tangent is not symmetric: the direct LU
    # takes it as it is; CG and the stability check take its symmetric part
    # (and so is self contact's, with the faces' turning in it)
    def _unsym(c):
        return getattr(c, "friction", 0.0) > 0 or getattr(c, "rotation", False)
    asym = any(_unsym(c) for c in contact)

    def sym(k):
        return (0.5 * (k + k.T)).tocsr() if asym else k

    def stab(k, v, target):
        """The tangent for the stability check: with friction, without the
        sliding force's growth with the normal force, and self contact's
        without the faces' turning (both unsymmetric, and their symmetric
        parts are indefinite: not a buckle)."""
        if not asym:
            return k
        for c in contact:
            if _unsym(c):
                kf = c._load(v, target)[1]
                c._symmetric = True
                try:
                    ks = c._load(v, target)[1]
                finally:
                    c._symmetric = False
                k = k + keep_free @ (ks - kf) @ keep_free
        return sym(k)

    # the smallest force worth resolving: a strain of 1e-6 on one cell face
    # (a part only carried along rigidly, e.g. towards a contact it has not
    # reached yet, has no load to measure its residual against)
    fscale = 1e-6 * float(mod.max()) * mesh.pitch ** 2

    def newton(v, target, a=None, search=False):
        rnorm0 = None
        a = alpha if a is None else a
        # K_aa^-1 K_au per element (incompatible modes), two buffers so a
        # backtrack can still use the last good iterate's
        bufs = None if a is None else [np.empty((mesh.n_elem, 9, 24)) for _ in range(2)]
        cur, prev, cuts, best = 0, None, 0, None

        def follow(a_lin, c, d):
            # the bubbles follow the nodes: alpha += -K_aa^-1 K_au du
            if c is None:
                return a_lin
            return a_lin - np.einsum("mij,mj->mi", c, d[mesh.edof]).reshape(-1, 3, 3)

        for its in range(1, max_newton + 1):
            ra = 0.0
            cpl = None if bufs is None else bufs[cur]
            if a is None:
                fint, k, jmin = _assemble(stiff, v, mod, nu, material)
                a_lin = None
            else:
                fint, k, jmin, a_lin, ra = _assemble(stiff, v, mod, nu, material, alpha=a, coupling=cpl)
            if not np.isfinite(jmin) or jmin <= 0:
                # the update turned a cell inside out: halve it (a damped
                # Newton step), rather than give up the whole load step
                if prev is None or cuts >= 8:
                    return None
                v0, a0, d, c0 = prev
                d = 0.5 * d
                prev = v0, a0, d, c0
                v, a = v0 + d, follow(a0, c0, d)
                cuts += 1
                continue
            if spr is not None:
                fint = fint + spr @ v
                k = (k + spr).tocsr()
            fext = target * f
            for t in loads:
                ft, kt, _ = t._load(v, target)
                fext = fext + ft
                # constrained rows and columns stay identity, as Stiffness leaves them
                k = (k + keep_free @ kt @ keep_free).tocsr()
            r = fext - fint
            r[cons] = 0.0
            ref = max(np.linalg.norm(fext), np.linalg.norm(fint[pd]) if len(pd) else 0.0, fscale)
            rnorm = np.hypot(np.linalg.norm(r), ra)
            if rnorm <= rtol * ref:
                return v, its, fint, k, a_lin
            if rnorm0 is None:
                rnorm0 = rnorm
            elif its > 6 and rnorm > rnorm0:
                return None          # diverging
            if best is None or rnorm < 0.5 * best[0]:
                best = rnorm, its
            elif search and its - best[1] >= STALL_NEWTON:
                return None          # stalled (contact chatter): a shorter step is cheaper
            try:
                d = lin.solve(k if lin.direct else sym(k), r)
            except RuntimeError:
                return None
            if not np.all(np.isfinite(d)):
                return None
            d[cons] = 0.0
            big = np.abs(d).max()
            if big > cap:
                d *= cap / big
            if search:
                # contact switches on and off between iterations, and its
                # tangent leaves out the faces' rotation, so Newton can
                # cycle: on a retry, backtrack a step that makes the
                # residual worse
                for _ in range(5):
                    rt = trial(v + d, target, follow(a_lin, cpl, d) if a_lin is not None else None)
                    if rt is not None and rt < rnorm:
                        break
                    d *= 0.5
            prev = v, a_lin, d, cpl
            v, a = v + d, (follow(a_lin, cpl, d) if a_lin is not None else None)
            cur = 1 - cur
        return None

    def trial(v, target, a):
        """Residual norm at v without the tangent (None if a cell inverts)."""
        if a is None:
            fint, _, jmin = _assemble(stiff, v, mod, nu, material, tangent=False)
            ra = 0.0
        else:
            fint, _, jmin, _, ra = _assemble(stiff, v, mod, nu, material, tangent=False, alpha=a)
        if not np.isfinite(jmin) or jmin <= 0:
            return None
        if spr is not None:
            fint = fint + spr @ v
        fext = target * f
        for t in loads:
            fext = fext + t._load(v, target)[0]
        r = fext - fint
        r[cons] = 0.0
        return float(np.hypot(np.linalg.norm(r), ra))

    u = np.zeros(n)
    lam, dlam = 0.0, min(1.0 / steps, max_step)
    du_prev, dlam_prev = None, None
    history = []
    message = ""
    touch_step = max(4 * min_step, 1.0 / 128)
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
        if out is None and contact:
            out = newton(v, target, search=True)
        if out is None:
            if dlam <= min_step:
                message = (f"stopped at {lam:.3f} of the load: no equilibrium past it under load control "
                           "(a limit point, i.e. buckling or snap-through, or the part folded). "
                           "Try a prescribed displacement instead of a force.")
                break
            dlam *= 0.5
            continue
        v, its, fint, k, a_conv = out
        if asym and dlam > touch_step:
            # friction: a contact that touches during a step measures its
            # slip from the touch along a straight path through the step,
            # which the part did not take (it moved rigidly up to the touch
            # and deformed after; the stiff penalty turns the difference
            # into a spurious stick force). When contacts carrying a good
            # share of the load touched during the step, bisect towards the
            # touch so the step that makes it is short.
            for c in contact:
                if getattr(c, "friction", 0.0) > 0:
                    c._load(v, target)
            share = max((c._early[0] for c in contact if getattr(c, "friction", 0.0) > 0), default=0.0)
            if share > 0.25:
                dlam = max(0.5 * dlam, touch_step)
                continue
        rec = {"loadFactor": target, "newton": its}
        if stability:
            lmin, phi = lin.lowest_mode(stab(k, v, target))
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
                            l2 = lin.lowest_mode(stab(o[3], o[0], target))[0]
                            if l2 >= -tol:
                                found = o, l2
                                break
                    if found is not None:
                        (v, its2, fint, k, a_conv), l2 = found
                        rec["newton"] += its2
                        rec["branchSwitch"] = True
                        rec["lowestEigenvalue"] = l2
                        break
                else:
                    rec["unstable"] = True
        du_prev, dlam_prev = (None if rec.get("branchSwitch") else v - u), dlam
        u, lam = v, target
        alpha = a_conv
        for name, wv in probes.items():
            rec[name] = float(wv @ u)
        if len(pd):
            rec["reaction"] = fint.reshape(-1, 3)[pnodes].sum(axis=0).tolist()
        for i, t in enumerate(tendons):
            rec[f"tendon{i}Travel"] = t._len0 - t._load(u, lam)[2]
            if hasattr(t, "pull"):
                rec[f"tendon{i}Tension"] = float(t.pull(u, lam)[0])
        for i, c in enumerate(contact):
            info = c._load(u, lam)[2]
            rec[f"contact{i}"] = {k: v for k, v in info.items()}
            if hasattr(c, "_commit"):
                c._commit(u, lam)       # friction: where each contact sticks from here on
        history.append(rec)
        if on_step is not None:
            on_step(rec, u)
        if rec["newton"] <= GROW_NEWTON:
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
    dofs cleared, zero on their diagonal). With the incompatible-mode
    element the bubbles carry their linear amplitudes (R u) both in the
    stresses and in the mode shapes: Kg = T^T Kg_ext T, T = [I; R]."""
    mesh = stiff.mesh
    h, w = mesh.pitch, mesh.pitch ** 3 / 8.0
    lam, mu = _lame(nu)
    im = getattr(stiff, "element", "plain") == "im"
    grads, wmat, n = (_GAUSS_IM, _W_IM, 11) if im else (_GAUSS, _W, 8)
    if im:
        r = _hex8_parts(nu, "im")[1]                       # (9, 24): amplitudes 3*mode + component
        t = np.vstack([np.eye(24), r])                     # (33, 24)
    data = np.zeros(stiff.nnz)
    for s in range(0, mesh.n_elem, _CHUNK):
        el = np.arange(s, min(s + _CHUNK, mesh.n_elem))
        m = len(el)
        ue = u[mesh.edof[el]].reshape(m, 24)
        ux = (ue @ t.T if im else ue).reshape(m, n, 3)
        kq = np.zeros((m, n * n))
        for g, dn0 in enumerate(grads):
            grad = np.einsum("mai,aj->mij", ux, dn0 / h)
            eps = 0.5 * (grad + grad.transpose(0, 2, 1))
            sig = (lam * np.trace(eps, axis1=1, axis2=2)[:, None, None] * _I3 + 2 * mu * eps) * mod[el, None, None]
            kq += (w / h ** 2) * (sig.reshape(m, 9) @ wmat[g])
        kg = np.einsum("mab,ij->maibj", kq.reshape(m, n, n), _I3).reshape(m, 3 * n, 3 * n)
        if im:
            kg = t.T @ kg @ t
        stiff.add_elements(data, kg, el)
    return stiff.finish(data, diag=0.0)


def linear_buckling(mesh: HexMesh, nu: float, modulus, fixed: np.ndarray, force: np.ndarray,
                    n_modes: int = 3, springs: np.ndarray | None = None,
                    element: str | None = None) -> BucklingResult:
    """Smallest positive load factors lambda at which lambda * force buckles
    the part (linear eigenvalue buckling: exact for a perfect column, an
    upper bound for real parts with imperfections; take a margin of 2-3 on
    printed slender members). Below 1, the load as given buckles it."""
    mod = np.broadcast_to(np.asarray(modulus, float), (mesh.n_elem,)).copy()
    # solve_nonlinear's element (incompatible modes by default), so the load
    # factor and the post-buckling path it seeds are one model's; the plain
    # hex8 shear-locks, which put the load of a column one / two cells thick
    # 54% / 14% above Euler's (incompatible modes: within 0.3%)
    stiff = Stiffness(mesh, nu, fixed, element=element)
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
    # the linear answer on the nonlinear element's own small-strain limit
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


def large_deflection_report(graph, spec: dict, material) -> dict:
    """The field graph's `largeDeflection` block: the `part` (a solid node,
    default the graph's output) voxelized at `pitch`, clamped inside
    `fixed`, under dead `loads`, prescribed `displacements` (an actuator's
    stroke), `tendons` (cables through guides at a tension, or wound in by
    a servo: stroke and stiffness) and `contact` (rigid obstacles, fixed or
    moving, and the part on itself), solved with large deflections.

    Reports how far the load got, peak strain and stress, each probe's
    travel, each tendon's travel and tension, each contact's force, nodes
    touching and penetration, and the step history."""
    from .contact import SelfContact, obstacle_from_spec
    from .impact import sample_core
    from .mechanism_nl import ServoTendon

    part = spec.get("part") or graph.output
    pitch = float(spec.get("pitch", 1.0))
    max_cells = int(spec.get("maxCells", 20000))
    b = graph.bounds(part)
    if b is None:
        raise ValueError(f"largeDeflection: node {part!r} has no bounds")
    for _ in range(8):
        inside, _, lo, h = sample_core(graph, part, None, pitch, bounds=b)
        if inside.sum() <= max_cells:
            break
        pitch *= 1.05 * (inside.sum() / max_cells) ** (1 / 3)
    ijk = np.argwhere(inside)
    if len(ijk) == 0:
        raise ValueError(f"largeDeflection: no cells inside {part!r} at pitch {pitch:g}")
    mesh = HexMesh(ijk, h, lo)

    def nodes_in(node, what):
        ids = np.flatnonzero(graph.evaluate(mesh.node_xyz, node) <= 0.5 * h)
        if len(ids) == 0:
            raise ValueError(f"largeDeflection: no nodes inside {what} {node!r}")
        return ids

    def where(item, what):
        if "region" in item:
            return nodes_in(item["region"], what)
        if "at" in item:
            return nodes_near(mesh, item["at"], float(item.get("radius", 1.5 * h)))
        raise ValueError(f"largeDeflection: {what} needs a region (node) or at (point)")

    fixed = np.zeros(mesh.n_dof, bool)
    fn = nodes_in(spec["fixed"], "fixed")
    for a in range(3):
        fixed[3 * fn + a] = True
    force = np.zeros(mesh.n_dof)
    for ld in spec.get("loads", []):
        ids = where(ld, "load")
        force[(3 * ids[:, None] + np.arange(3)).ravel()] += np.tile(np.asarray(ld["force"], float) / len(ids), len(ids))
    pd, pv = [], []
    for dsp in spec.get("displacements", []):
        ids = where(dsp, "displacement")
        mv = np.asarray(dsp["move"], float)
        for c in range(3):
            if dsp.get("free") and abs(mv[c]) < 1e-12:
                continue
            dofs = 3 * ids + c
            dofs = dofs[~fixed[dofs]]
            pd.append(dofs)
            pv.append(np.full(len(dofs), mv[c]))
    prescribed = (np.concatenate(pd), np.concatenate(pv)) if pd else None
    tendons = []
    for t in spec.get("tendons", []):
        path = []
        for pt in t["path"]:
            path.append(where(pt, "tendon guide") if isinstance(pt, dict) else np.asarray(pt, float))
        if "stroke" in t:
            tendons.append(ServoTendon(path, float(t["stroke"]), float(t["stiffness"])))
        else:
            tendons.append(Tendon(float(t["tension"]), path))
    contact = []
    cs = spec.get("contact") or {}
    stiff = {k: cs[k] for k in ("stiffness", "friction") if k in cs}
    for ob in cs.get("obstacles", []):
        contact.append(obstacle_from_spec({**stiff, **ob}))
    if cs.get("self"):
        contact.append(SelfContact(**stiff))
    probes = {}
    for p in spec.get("probes", []):
        ids = where(p, "probe")
        d = np.asarray(p["direction"], float)
        d = d / np.linalg.norm(d)
        w = np.zeros(mesh.n_dof)
        for c in range(3):
            w[3 * ids + c] = d[c] / len(ids)
        probes[p.get("name") or f"probe{len(probes)}"] = w
    E, nu = float(material.E), float(spec.get("nu", material.nu))
    r = solve_nonlinear(mesh, nu, E, fixed, force if force.any() else None, prescribed,
                        material=spec.get("model", "neo-hookean"), steps=int(spec.get("steps", 10)),
                        tendons=tendons, contact=contact, probes=probes)
    last = r.history[-1] if r.history else {}
    out = {"cells": int(mesh.n_elem), "pitch": h, "converged": r.converged, "loadReached": r.load_factor,
           "maxStrain": float(r.strain.max()), "maxStress": float(r.stress.max()),
           "maxDisplacement": float(np.linalg.norm(r.u.reshape(-1, 3), axis=1).max())}
    if probes:
        out["probes"] = {k: last.get(k) for k in probes}
    if tendons:
        out["tendons"] = [{"travel": last.get(f"tendon{i}Travel"),
                           "tension": last.get(f"tendon{i}Tension", r.load_factor * getattr(t, "tension", 0.0))}
                          for i, t in enumerate(tendons)]
    if contact:
        names = [getattr(c, "name", None) or (c.shape if hasattr(c, "shape") else "self") for c in contact]
        out["contact"] = [{"name": nm, **(last.get(f"contact{i}") or {})} for i, nm in enumerate(names)]
    if prescribed is not None and "reaction" in last:
        out["actuatorForce"] = last["reaction"]
    if spec.get("maxStrain") is not None:
        out["ok"] = bool(r.converged and out["maxStrain"] <= float(spec["maxStrain"]))
    if r.message:
        out["message"] = r.message
    keep = ("loadFactor", "newton", "branchSwitch", *probes)
    out["history"] = [{**{k: h_[k] for k in keep if k in h_},
                       **{k: v for k, v in h_.items() if k.startswith(("tendon", "contact"))}} for h_ in r.history]
    return out
