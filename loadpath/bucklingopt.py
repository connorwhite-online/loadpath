"""
Linear buckling as an optimization constraint: the smallest buckling load
factors of a density design and their sensitivities, for the constrained
method (constrained.py).

The check (verify._buckling) reads the finished part; the optimizer needs
the same number on the grey design, every iteration, with its gradient.
For a design x with stiffness K(x) and geometric stiffness Kg(x, u(x)) of
the linear stresses under the load case,

    K phi = lambda (-Kg) phi   <=>   (-Kg) phi = mu K phi,  mu = 1 / lambda,

and the largest mu are the smallest positive buckling loads. With phi
normalized to phi^T K phi = 1, the sensitivity of mu to cell e is

    dmu/dx_e = -E_G'(x_e) phi_e^T kg_e phi_e - mu E'(x_e) phi_e^T ke phi_e
               + E'(x_e) w_e^T ke u_e,      K w = a,  a = d(phi^T Kg phi)/du,

the last term because Kg depends on the displacements, which depend on the
design (one more solve per mode, on the same factorization). The modes are
aggregated with a Kreisselmeier-Steinhauser max, which keeps the constraint
smooth where two modes cross.

Pseudo-modes. Grey cells are soft in K (E ~ x^3) but carry stress in Kg,
and a soft cell far from the load path can "buckle" at a tiny load (Neves
et al. 1995; Ferrari and Sigmund 2019). The model is built on the live
cells only (density above simp.ALIVE and joined to a support, as the
stiffness solve), and Kg's stresses take a harsher interpolation than K
below a density of `cutoff`, so a nearly empty cell neither stiffens nor
loads the mode.

Element: incompatible-mode hex8 (fem.hex8_element), the verify check's own,
so a member 2-3 cells thick reads its Euler load within a few percent.
The factorization is PARDISO (pypardiso, if installed) or SuperLU.
"""
from __future__ import annotations

import time

import numpy as np
import scipy.sparse as sp
from scipy import ndimage
from scipy.sparse.linalg import LinearOperator, eigsh

from .fem import HexMesh, Stiffness, _hex8_parts, hex8_element

__all__ = ["BucklingModel", "block_modes", "geometric_parts"]

def _factor(k: sp.csr_matrix, node_ijk=None):
    """A solve with K for one right-hand side or a block of them: PARDISO
    when available (symmetric positive definite, upper triangle; 10-30x
    SuperLU on 3D hex meshes, and a block of eight costs about two single
    solves), else SuperLU in nested-dissection order (node_ijk, the grid
    nodes the dofs belong to; nonlinear._splu). Returns (solve, release)."""
    from .nonlinear import _Pardiso, _have_pardiso, _splu

    if not _have_pardiso():
        lu = _splu(k, node_ijk)
        return lu.solve, None
    try:
        # positive definite, else symmetric indefinite (a zero pivot: a
        # hinge in the live cells), both on the upper triangle. The handle
        # comes from nonlinear's pool: a new one looks for MKL with a
        # recursive glob, 0.3-0.5 s of every evaluation
        fac = _Pardiso(k)
    except RuntimeError:
        fac = _Pardiso(k, symmetric=False)      # pivoting LU: twice the factor
    return fac.solve, fac.close


def _gauss_im():
    from .nonlinear import _GAUSS, _GAUSS_IM

    return _GAUSS_IM if hex8_element() == "im" else _GAUSS


def geometric_parts(ue, phis, h, nu):
    """Per element, for displacements ue (m, 24) and modes phis [(m, 24)]:
    g (m,) = phi_e^T kg_e(u_e) phi_e and a (m, 24) = d g / d u_e, at unit
    stress modulus. kg_e is nonlinear._geometric's element matrix (the
    bubbles' linear amplitudes in both the stresses and the modes)."""
    from .nonlinear import _linear_stress

    grads = _gauss_im()
    r = _hex8_parts(nu, hex8_element())[1]
    im = r is not None and len(grads[0]) == 11
    t = np.vstack([np.eye(24), r]) if im else np.eye(24)
    n, ng, m = t.shape[0] // 3, len(grads), len(ue)
    w = h ** 3 / 8.0
    # every Gauss point's gradients as one matrix, so each mode is two
    # GEMMs (an einsum per point ran 5x slower on the 130k-cell yoke):
    # (ue @ td)[m, (g, k, i)] = d u_k / d x_i at point g
    d = np.zeros((n, 3, ng, 3, 3))
    for g, dn0 in enumerate(grads):
        for k in range(3):
            d[:, k, g, k, :] = dn0 / h
    td = t.T @ d.reshape(3 * n, ng * 9)                     # (24, 9 ng)
    gu = (ue @ td).reshape(m, ng * 9)
    out = []
    for ph in phis:
        gp = (ph @ td).reshape(m * ng, 3, 3)                # [k, i] = d phi_k / d x_i
        hh = np.matmul(gp.transpose(0, 2, 1), gp)
        s = w * _linear_stress(hh, nu).reshape(m, ng * 9)
        out.append((np.einsum("mq,mq->m", gu, s), s @ td.T))
    return out


# block_modes' iteration caps from the last design's modes / from nothing
BLOCK_ITERS_WARM = 30
BLOCK_ITERS_COLD = 60
# ... and how far a wanted mode's residual may be from zero, in the K^-1
# norm over mu_max (the eigenvalue is then good to about its square)
BLOCK_TOL = 3e-3
# ... and for a reading alone, where the modes' shapes do not matter (a
# constraint far from its limit, the crisp reference): mu is good to a
# percent here, in a third of the iterations (the wrist yoke, warm: 0.9% off
# after one iteration at 0.10-0.12, six to reach BLOCK_TOL)
BLOCK_TOL_LOOSE = 0.1


def block_modes(a, solve, x0, nm: int, tol: float = BLOCK_TOL, maxiter: int = BLOCK_ITERS_WARM, k=None,
                loose_below: float = 0.0):
    """The largest eigenpairs of a phi = mu K phi for symmetric a and a K
    given by its inverse (`solve`: a block of right-hand sides in, K^-1 of
    them out), by locally optimal block iteration from the columns of x0:
    Rayleigh-Ritz on [X, W, P], W = K^-1 (a X - K X mu) and P the last
    step. Returns (mu (p,), X (n, p), iterations, converged), mu largest
    first, X^T K X = I; converged means the first nm. With `k` (the matrix
    itself) x0 is taken as it stands, which is right for the last design's
    modes: a design one optimizer step on is then confirmed with a single
    block solve. Without it x0 goes through one power step first (a random
    start is all stiff-dof noise until it has). While the largest mu reads
    under `loose_below`, BLOCK_TOL_LOOSE stands in for tol.

    scipy's lobpcg does the same with K's inverse as its preconditioner,
    but multiplies every block by K as well as a (here K W is the residual
    itself and K X, K P follow from the combinations), and it judges
    convergence on the residual's 2-norm, which the stiffest dofs dominate:
    on the wrist yoke it ran its 25 iterations whether or not the modes had
    settled, then fell back to ARPACK. The residual is measured here in the
    K^-1 norm, r^T K^-1 r = r^T w, which costs nothing and is the one the
    eigenvalue's error follows."""
    from scipy.linalg import LinAlgError, eigh

    def ritz(s, a_s, k_s, take):
        g = s.T @ a_s
        m = s.T @ k_s
        g, m = 0.5 * (g + g.T), 0.5 * (m + m.T)
        d = np.diag(m)
        if not np.all(d > 0):
            return None
        d = 1.0 / np.sqrt(d)
        try:
            th, cv = eigh(g * d[:, None] * d[None, :], m * d[:, None] * d[None, :])
        except (LinAlgError, ValueError):
            return None
        o = np.argsort(-th)[:take]
        return th[o], d[:, None] * cv[:, o]

    p = x0.shape[1]
    got = None
    if k is not None:
        x = np.asarray(x0, float)
        kx = np.asarray(k @ x)
        ax = np.asarray(a @ x)
        got = ritz(x, ax, kx, p)
    if got is None:
        # one power step: X = K^-1 a x0, so K X = a x0 without multiplying
        # by K (and X is zero on held dofs, as a's rows are)
        kx = np.asarray(a @ x0)
        x = np.asarray(solve(kx)).reshape(kx.shape)
        ax = np.asarray(a @ x)
        got = ritz(x, ax, kx, p)
    if got is None:
        return np.zeros(p), x, 0, False
    th, cv = got
    x, ax, kx = x @ cv, ax @ cv, kx @ cv
    pk = apk = kpk = None
    for it in range(1, maxiter + 1):
        r = ax - kx * th
        w = np.asarray(solve(r)).reshape(r.shape)
        rn = np.sqrt(np.maximum(np.einsum("ij,ij->j", r, w), 0.0))
        if th[0] > 0 and np.all(rn[:nm] <= (max(tol, BLOCK_TOL_LOOSE) if th[0] < loose_below else tol) * th[0]):
            return th, x, it - 1, True
        aw = np.asarray(a @ w)
        got = None
        if pk is not None:
            got = ritz(np.hstack([x, w, pk]), np.hstack([ax, aw, apk]), np.hstack([kx, r, kpk]), p)
            if got is not None:
                th_n, cv = got
                cx, cw, cp = cv[:p], cv[p:2 * p], cv[2 * p:]
                pn, apn, kpn = w @ cw + pk @ cp, aw @ cw + apk @ cp, r @ cw + kpk @ cp
        if got is None:
            got = ritz(np.hstack([x, w]), np.hstack([ax, aw]), np.hstack([kx, r]), p)
            if got is None:
                return th, x, it, False
            th_n, cv = got
            cx, cw = cv[:p], cv[p:]
            pn, apn, kpn = w @ cw, aw @ cw, r @ cw
        x, ax, kx = x @ cx + pn, ax @ cx + apn, kx @ cx + kpn
        pk, apk, kpk = pn, apn, kpn
        th = th_n
    return th, x, maxiter, False


# A piece is held when its least-stopped rigid motion keeps this share of
# its most-stopped one's energy in the supports (BucklingModel._held). Fixed
# nodes on one line, or one pin, read zero to rounding (1e-16); a support
# patch two cells across at the far end of a 90-cell part reads 5e-4.
_HELD_TOL = 1e-8

# Cells under this density are left out of the buckling model: x^3 there is
# under the void floor every cell is given anyway (simp.E_MIN, 1e-6), and
# the stress stiffness faded to nothing well above it (BucklingModel.interp).
# At simp.ALIVE (1e-3) a run started from the last passing design factored
# the whole grid on its first reading: resumed at a soft projection, the
# void of that design sits at 0.001-0.01 everywhere (the wrist yoke: 116,833
# of 119,398 cells over 1e-3, 21,906 over 1e-2), a 400k-dof factorization
# and its 3.4 GB for cells that carry nothing: 97 s of a 245 s run.
ALIVE = 1e-2


class BucklingModel:
    """Buckling load factors and their gradients for density designs on
    `domain`'s grid. `n_modes` modes per load case are computed and
    aggregated (KS, `rho`)."""

    def __init__(self, domain, nu: float, n_modes: int = 4, rho: float = 40.0, cutoff: float = 0.3,
                 alive: float = ALIVE):
        self.domain = domain
        self.nu = nu
        self.n_modes = n_modes
        self.rho = rho
        self.cutoff = cutoff
        self.alive = alive
        mesh = domain.mesh
        self.support_touch = domain.support_nodes[mesh.enode].any(axis=1)
        self.anchor = self.support_touch | domain.load_nodes[mesh.enode].any(axis=1) | domain.preserve
        # the last design's modes per load case, where the next solve
        # starts; a model made later on the same domain (the study's next
        # constrained run) starts from them too
        self._v0 = dict(getattr(domain, "_buckling_modes", None) or {})
        self._mu0 = {}
        self.last = {}

    def interp(self, xe, p):
        """(E, dE/dx, E_G, dE_G/dx) per cell at unit modulus: E is SIMP's
        (E_MIN floor), E_G the stress interpolation of Kg, x^p above the
        cutoff and faded to zero below it as (x / cutoff)^3 more."""
        from .simp import E_MIN

        e = E_MIN + xe ** p * (1 - E_MIN)
        de = p * xe ** (p - 1) * (1 - E_MIN)
        c = self.cutoff
        if c > 0:
            r = np.clip(xe / c, 0.0, 1.0)
            f = np.where(xe < c, r ** 3, 1.0)
            df = np.where(xe < c, 3 * r ** 2 / c, 0.0)
        else:
            f, df = np.ones_like(xe), np.zeros_like(xe)
        eg = xe ** p * f
        deg = p * xe ** (p - 1) * f + xe ** p * df
        return e, de, eg, deg

    def _modes(self, c, a, k, minv, nm, fd, n_full, loose_below=0.0):
        """The nm largest mu of a phi = mu k phi, largest first: block
        iteration on the factorization (block_modes), from the last
        iteration's modes when there are any, else ARPACK in shift-invert-like
        mode 2 (tens of solves) if the block iteration does not settle."""
        prev = self._v0.get(c)
        p = nm + 2
        warm = False
        if prev is not None:
            x0 = prev[fd]
            warm = x0.shape[1] >= p
            if not warm:
                x0 = np.hstack([x0, np.random.default_rng(c).standard_normal((len(fd), p - x0.shape[1]))])
        else:
            x0 = np.random.default_rng(c).standard_normal((len(fd), p))
        try:
            with np.errstate(all="ignore"):
                mu, vec, its, ok = block_modes(a, minv.matmat, x0[:, :p], nm,
                                               maxiter=BLOCK_ITERS_WARM if prev is not None else BLOCK_ITERS_COLD,
                                               k=k if warm else None, loose_below=loose_below)
            self.last_its = its
            if ok and mu[0] > 0 and np.all(np.isfinite(mu)):
                self._mu0[c] = float(mu[0])
                self._store(c, vec, fd, n_full)
                self.last_method = "block"
                return mu[:nm], vec[:, :nm]
        except Exception:  # noqa: BLE001 - fall back to ARPACK below
            pass
        try:
            mu, vec = eigsh(a, k=nm, M=k, Minv=minv, which="LA", tol=1e-4, maxiter=2000)
        except Exception:  # noqa: BLE001 - ARPACK did not converge: once more, looser
            mu, vec = eigsh(a, k=nm, M=k, Minv=minv, which="LA", tol=1e-3, maxiter=5000)
        order = np.argsort(-mu)
        mu, vec = mu[order], vec[:, order]
        self._store(c, vec, fd, n_full)
        self._mu0[c] = float(mu[0])
        self.last_method = "arpack"
        return mu, vec

    def _store(self, c, vec, fd, n_full):
        full = np.zeros((n_full, vec.shape[1]))
        full[fd] = vec
        self._v0[c] = full

    def seed(self, other: "BucklingModel") -> None:
        """Start the cases this model has no modes for from `other`'s (the
        crisp blueprint's from the eroded design's: the same members,
        thicker): 6 iterations where a random start took 22."""
        for c, v in other._v0.items():
            self._v0.setdefault(c, v)

    def keep(self) -> None:
        """Leave the modes on the domain for the next model made on it."""
        try:
            self.domain._buckling_modes = dict(self._v0)
        except AttributeError:
            pass

    def _live(self, xe):
        """The cells the model is made of: live ones (over `alive`, or drawn
        in) in pieces the supports hold still.

        A piece is cells joined face to face. A clump that meets the part
        only along an edge or at a corner is hinged there: hex8 cells share
        those nodes and nothing that carries a moment, the printed part has
        no material across such a contact at all, and one hinge is a
        singular K. The wrist link's first constrained part (130k cells) had
        23 such clumps of 4-14 cells at iteration 39, remnants of members on
        their way out: mu read 1e15 with modes of negative stiffness, the
        constraint 1e11 over its limit with overflowing gradients, and the
        optimizer never came back (a 650 s part and a 2130 s one, 4-11 g
        heavier than the search's). At `alive` 1e-3 a few more fringe cells
        tied the same clumps in.

        Held still means every rigid motion of the piece is stopped by its
        fixed dofs and pin springs (_held): a piece on a pin alone turns
        about it, and that is singular too."""
        mesh = self.domain.mesh
        alive = (xe > self.alive) | self.anchor
        g = np.zeros(self.domain.dims, bool)
        ijk = tuple(mesh.elem_ijk.T)
        g[ijk] = alive
        lbl, _ = ndimage.label(g)       # face neighbours
        el = lbl[ijk]
        keep = np.zeros(len(xe), bool)
        for piece in np.unique(el[alive & self.support_touch]):
            cells = el == piece
            if piece > 0 and self._held(cells):
                keep |= cells
        return keep

    def _held(self, cells) -> bool:
        """Whether the supports stop all six rigid motions of these cells:
        the motions' energy in the fixed dofs (as unit springs) and the pin
        springs has no zero direction."""
        from .fem import rigid_body_modes

        d = self.domain
        mesh = d.mesh
        nodes = np.unique(mesh.enode[cells])
        xyz = mesh.node_xyz[nodes]
        span = float(np.ptp(xyz, axis=0).max()) or 1.0
        r = rigid_body_modes(xyz / span)
        dofs = (3 * nodes[:, None] + np.arange(3)).ravel()
        rf = r[d.fixed[dofs]]
        g = rf.T @ rf
        if d.springs is not None:
            ks = d.springs[dofs][:, dofs]
            top = ks.diagonal().max() if ks.nnz else 0.0
            if top > 0:
                g = g + r.T @ (ks @ r) / top
        w = np.linalg.eigvalsh(g)
        return bool(w[0] > _HELD_TOL * max(w[-1], 1e-300))

    def evaluate(self, xe: np.ndarray, p: float, cases=None, gradient: bool = True, relax=None):
        """For each load case (default all): (mu_ks, dmu_ks/dxe (n,), mus),
        where 1 / mu_ks is the aggregated smallest buckling load factor
        (inf when the case does not compress anything). `gradient` False
        skips the sensitivities (zeros): the adjoint block solve and the
        modes' element products, for a reading alone. `relax` gives, per
        case, (mu_loose, mu_far): a case whose largest mu reads under
        mu_loose is solved to BLOCK_TOL_LOOSE (its value is good to a
        percent, its gradient rough), and one whose mu_ks is under mu_far
        gets no gradient at all. For a constraint that far from its limit
        neither steers the optimizer."""
        t0 = time.time()
        tm = {"setup": 0.0, "factor": 0.0, "stress": 0.0, "modes": 0.0, "gradient": 0.0}
        tl = t0

        def lap(key):
            nonlocal tl
            now = time.time()
            tm[key] += now - tl
            tl = now
        d = self.domain
        mesh = d.mesh
        if cases is None:
            # not the vibration block's stiffness-only inertial cases
            skip = set(getattr(d, "inertial_cases", ()))
            cases = [c for c in range(len(d.loads)) if c not in skip]
        cases = list(cases)
        keep = self._live(xe)
        el = np.flatnonzero(keep)
        sub = HexMesh(mesh.elem_ijk[el], mesh.pitch, mesh.origin)
        fn = mesh.node_id(sub.node_ijk)
        fd = (3 * fn[:, None] + np.arange(3)).ravel()
        fixed = d.fixed[fd]
        stiff = Stiffness(sub, self.nu, fixed)
        e, de, eg, deg = self.interp(xe[el], p)
        k = stiff.assemble(e)
        if d.springs is not None:
            free = sp.diags((~fixed).astype(float))
            k = (k + free @ d.springs[fd][:, fd] @ free).tocsr()
        lap("setup")
        solve, free_memory = _factor(k, sub.node_ijk)
        lap("factor")
        try:
            ndof = sub.n_dof
            minv = LinearOperator((ndof, ndof), matvec=solve, matmat=solve, dtype=float)
            ke = stiff.ke
            out = []
            for j, c in enumerate(cases):
                mu_loose, mu_far = relax[j] if relax is not None else (0.0, 0.0)
                f = d.loads[c][fd].copy()
                f[fixed] = 0.0
                if not np.any(f):
                    out.append((0.0, np.zeros(mesh.n_elem), np.zeros(0)))
                    continue
                u = solve(f)
                from .nonlinear import _geometric

                kg = _geometric(stiff, u, eg, self.nu)
                lap("stress")
                nm = min(self.n_modes, ndof - 2)
                mu, vec = self._modes(c, -kg, k, minv, nm, fd, mesh.n_dof, mu_loose)
                # phi^T K phi is positive for every mode of a K that holds
                # its cells. Where it is not, K was singular (a mechanism
                # _live did not see) and what came back is not a reading:
                # the wrist link's hinged clumps gave mu 1e15 at norms of
                # -0.97 and -7.7. Nothing is kept of it, the caller's last
                # reading stands.
                kn = np.einsum("ij,ij->j", vec, np.asarray(k @ vec))
                if not (np.all(np.isfinite(mu)) and np.all(np.isfinite(kn)) and np.all(kn > 0)):
                    self._v0.pop(c, None)
                    self._mu0.pop(c, None)
                    raise RuntimeError(f"no buckling reading for load case {c}: the live cells move freely "
                                       f"(a mode's stiffness reads {float(np.min(kn)):.3g})")
                lap("modes")
                if mu[0] <= 0:
                    out.append((0.0, np.zeros(mesh.n_elem), mu))
                    continue
                # KS over the modes, on mu / mu_max
                mu = np.asarray(mu[:nm], float)
                mmax = float(mu[0])
                wts = np.exp(self.rho * (mu / mmax - 1.0))
                ks = mmax * (1.0 + np.log(wts.sum()) / self.rho)
                if not gradient or ks < mu_far:
                    out.append((ks, np.zeros(mesh.n_elem), mu))
                    continue
                wts = wts / wts.sum()
                use = [i for i in range(nm) if wts[i] >= 1e-6]
                # modes normalized phi^T K phi = 1
                phis = [vec[:, i] / np.sqrt(kn[i]) for i in use]
                ue = u[sub.edof]
                parts = geometric_parts(ue, [ph[sub.edof] for ph in phis], mesh.pitch, self.nu)
                # the adjoints of every counted mode in one block solve
                arhs = np.zeros((ndof, len(use)))
                for q, (_, a_e) in enumerate(parts):
                    arhs[:, q] = np.bincount(sub.edof.ravel(), weights=(eg[:, None] * a_e).ravel(), minlength=ndof)
                arhs[fixed] = 0.0
                wsol = np.asarray(solve(arhs)).reshape(ndof, len(use))
                grad = np.zeros(mesh.n_elem)
                for q, i in enumerate(use):
                    pe = phis[q][sub.edof]
                    g_e = parts[q][0]
                    kphi = np.einsum("mi,mi->m", pe @ ke, pe)
                    term = np.einsum("mi,mi->m", wsol[:, q][sub.edof] @ ke, ue)
                    grad[el] += wts[i] * (-deg * g_e - mu[i] * de * kphi + de * term)
                out.append((ks, grad, mu))
                lap("gradient")
        finally:
            if free_memory is not None:
                free_memory()
        self.last = {"cells": int(len(el)), "seconds": time.time() - t0, "parts": tm,
                     "method": getattr(self, "last_method", None), "eigIterations": getattr(self, "last_its", None)}
        return out
