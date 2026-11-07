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
from scipy.sparse.linalg import LinearOperator, eigsh, lobpcg

from .fem import HexMesh, Stiffness, _hex8_parts, hex8_element

__all__ = ["BucklingModel", "geometric_parts"]

_CONN26 = np.ones((3, 3, 3), bool)


def _factor(k: sp.csr_matrix, node_ijk=None):
    """A solve with K for one right-hand side or a block of them: PARDISO
    when available (symmetric positive definite, upper triangle; 10-30x
    SuperLU on 3D hex meshes, and a block of eight costs about two single
    solves), else SuperLU in nested-dissection order (node_ijk, the grid
    nodes the dofs belong to; nonlinear._splu)."""
    try:
        import pypardiso
    except ImportError:
        from .nonlinear import _splu

        lu = _splu(k, node_ijk)
        return lu.solve, None
    a = sp.triu(k, format="csr")
    a.sort_indices()
    solver = pypardiso.PyPardisoSolver()
    solver.set_matrix_type(2)
    try:
        solver.factorize(a)
    except Exception:  # noqa: BLE001 - a zero pivot (a hinge in the live cells): pivoting LU
        solver.free_memory(everything=True)
        a = k.tocsr()
        a.sort_indices()
        solver = pypardiso.PyPardisoSolver()
        solver.set_matrix_type(11)
        solver.factorize(a)
    solver.set_phase(33)

    def solve(b):
        b = np.asarray(b, float)
        # straight to the solve phase: pypardiso.solve() hashes the matrix
        # every call to see whether it is factored already
        return solver._call_pardiso(a, np.asfortranarray(b) if b.ndim == 2 else np.ascontiguousarray(b))

    return solve, lambda: solver.free_memory(everything=True)


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
    n = t.shape[0] // 3
    m = len(ue)
    w = h ** 3 / 8.0
    ux = (ue @ t.T).reshape(m, n, 3)
    out = []
    for ph in phis:
        px = (ph @ t.T).reshape(m, n, 3)
        g = np.zeros(m)
        da = np.zeros((m, n, 3))
        for dn0 in grads:
            dn = dn0 / h
            gp = np.einsum("mak,ai->mki", px, dn)              # d phi_k / d x_i
            hh = np.einsum("mki,mkj->mij", gp, gp)
            s = _linear_stress(hh, nu)
            gu = np.einsum("mai,aj->mij", ux, dn)              # d u_i / d x_j
            g += w * np.einsum("mij,mij->m", gu, s)
            da += w * np.einsum("aj,mij->mai", dn, s)
        out.append((g, da.reshape(m, 3 * n) @ t))
    return out


class BucklingModel:
    """Buckling load factors and their gradients for density designs on
    `domain`'s grid. `n_modes` modes per load case are computed and
    aggregated (KS, `rho`)."""

    def __init__(self, domain, nu: float, n_modes: int = 4, rho: float = 40.0, cutoff: float = 0.3,
                 alive: float = 1e-3):
        self.domain = domain
        self.nu = nu
        self.n_modes = n_modes
        self.rho = rho
        self.cutoff = cutoff
        self.alive = alive
        mesh = domain.mesh
        self.support_touch = domain.support_nodes[mesh.enode].any(axis=1)
        self.anchor = self.support_touch | domain.load_nodes[mesh.enode].any(axis=1) | domain.preserve
        self._v0 = {}
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

    def _modes(self, c, a, k, minv, nm, fd, n_full):
        """The nm largest mu of a phi = mu k phi, largest first. Warm
        (LOBPCG from the last iteration's modes, preconditioned with the
        factorization: a few block solves) when there are last modes,
        else ARPACK in shift-invert-like mode 2 (tens of solves)."""
        prev = self._v0.get(c)
        if prev is not None:
            p = nm + 2
            x0 = prev[fd]
            if x0.shape[1] < p:
                x0 = np.hstack([x0, np.random.default_rng(c).standard_normal((len(fd), p - x0.shape[1]))])
            try:
                with np.errstate(all="ignore"):
                    import warnings

                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        # residuals to 0.3% of mu: the eigenvalues come out
                        # good to about the square of that
                        tol = 3e-3 * self._mu0.get(c, 1.0)
                        mu, vec, res = lobpcg(a, x0, B=k, M=minv, largest=True, tol=tol, maxiter=25,
                                              retResidualNormsHistory=True)
                order = np.argsort(-mu)
                mu, vec = mu[order], vec[:, order]
                # converged on the wanted modes (relative to mu_max)?
                r = np.asarray(res[-1])[order][:nm] if len(res) else np.full(nm, np.inf)
                if mu[0] > 0 and np.all(np.isfinite(mu)) and np.all(r < 1e-2 * abs(mu[0])):
                    self._mu0[c] = float(mu[0])
                    self._store(c, vec, fd, n_full)
                    self.last_method = "lobpcg"
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

    def _live(self, xe):
        mesh = self.domain.mesh
        alive = (xe > self.alive) | self.anchor
        g = np.zeros(self.domain.dims, bool)
        ijk = tuple(mesh.elem_ijk.T)
        g[ijk] = alive
        lbl, _ = ndimage.label(g, structure=_CONN26)
        el = lbl[ijk]
        keep_l = np.unique(el[alive & self.support_touch])
        return alive & np.isin(el, keep_l[keep_l > 0])

    def evaluate(self, xe: np.ndarray, p: float, cases=None):
        """For each load case (default all): (mu_ks, dmu_ks/dxe (n,), mus),
        where 1 / mu_ks is the aggregated smallest buckling load factor
        (inf when the case does not compress anything)."""
        t0 = time.time()
        d = self.domain
        mesh = d.mesh
        cases = list(range(len(d.loads))) if cases is None else list(cases)
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
        solve, free_memory = _factor(k, sub.node_ijk)
        try:
            ndof = sub.n_dof
            minv = LinearOperator((ndof, ndof), matvec=solve, matmat=solve, dtype=float)
            ke = stiff.ke
            out = []
            for c in cases:
                f = d.loads[c][fd].copy()
                f[fixed] = 0.0
                if not np.any(f):
                    out.append((0.0, np.zeros(mesh.n_elem), np.zeros(0)))
                    continue
                u = solve(f)
                from .nonlinear import _geometric

                kg = _geometric(stiff, u, eg, self.nu)
                nm = min(self.n_modes, ndof - 2)
                mu, vec = self._modes(c, -kg, k, minv, nm, fd, mesh.n_dof)
                if mu[0] <= 0:
                    out.append((0.0, np.zeros(mesh.n_elem), mu))
                    continue
                # KS over the modes, on mu / mu_max
                mu = np.asarray(mu[:nm], float)
                mmax = float(mu[0])
                wts = np.exp(self.rho * (mu / mmax - 1.0))
                ks = mmax * (1.0 + np.log(wts.sum()) / self.rho)
                wts = wts / wts.sum()
                use = [i for i in range(nm) if wts[i] >= 1e-6]
                # modes normalized phi^T K phi = 1
                phis = []
                for i in use:
                    ph = vec[:, i]
                    phis.append(ph / np.sqrt(max(float(ph @ (k @ ph)), 1e-300)))
                ue = u[sub.edof]
                parts = geometric_parts(ue, [ph[sub.edof] for ph in phis], mesh.pitch, self.nu)
                # the adjoints of every counted mode in one block solve
                arhs = np.zeros((ndof, len(use)))
                for j, (_, a_e) in enumerate(parts):
                    arhs[:, j] = np.bincount(sub.edof.ravel(), weights=(eg[:, None] * a_e).ravel(), minlength=ndof)
                arhs[fixed] = 0.0
                wsol = np.asarray(solve(arhs)).reshape(ndof, len(use))
                grad = np.zeros(mesh.n_elem)
                for j, i in enumerate(use):
                    pe = phis[j][sub.edof]
                    g_e = parts[j][0]
                    kphi = np.einsum("mi,ij,mj->m", pe, ke, pe)
                    term = np.einsum("mi,ij,mj->m", wsol[:, j][sub.edof], ke, ue)
                    grad[el] += wts[i] * (-deg * g_e - mu[i] * de * kphi + de * term)
                out.append((ks, grad, mu))
        finally:
            if free_memory is not None:
                free_memory()
        self.last = {"cells": int(len(el)), "seconds": time.time() - t0}
        return out
