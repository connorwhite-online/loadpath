"""
Compiled element kernel for the large-deflection solver (nonlinear.py).

nonlinear._element builds each hexahedron's internal force and tangent from
stacks of small numpy operations (3x3 inverses, determinants, 4th-order
einsums, transposed copies), whose overhead, not their arithmetic, was most
of a large-deflection solve: about 54 us per element per Newton iteration on
the 1056-cell tendon finger. Here the same element (total Lagrangian hex8,
2x2x2 Gauss, neo-Hookean or St. Venant-Kirchhoff, optionally with the nine
incompatible modes condensed out) is one compiled loop over elements, run on
all cores, and the scatter into the CSR tangent is a second compiled loop.

Same arithmetic as the numpy element, in a different summation order, so
forces and tangents agree to ~1e-14 relative, not bit for bit.
LOADPATH_NLKERNEL=0 runs the numpy element instead.
"""
from __future__ import annotations

import os

import numpy as np

try:
    import numba as _nb
except ImportError:  # pragma: no cover - numba is a dependency, kept optional
    _nb = None

__all__ = ["available", "elements", "scatter"]


def available() -> bool:
    return _nb is not None and os.environ.get("LOADPATH_NLKERNEL", "1") != "0"


if _nb is not None:
    @_nb.njit(cache=True, nogil=True)
    def _gauss(ux, G, h, lam, mu, svk, fx, kx):
        """Accumulate internal force fx (3n) and tangent kx (3n, 3n) of one
        element with generalized displacements ux (n, 3); returns min det F."""
        nn = ux.shape[0]
        w = h ** 3 / 8.0
        wk = w / (h * h)
        jmin = np.inf
        F = np.empty((3, 3))
        S = np.empty((3, 3))
        P = np.empty((3, 3))
        Ci = np.empty((3, 3))
        A = np.empty((3, 3, 3, 3))
        FA = np.empty((3, 3, 3, 3))
        Q = np.empty((3, 3, 3, 3))
        T = np.empty((nn, 3, 3, 3))
        for g in range(G.shape[0]):
            for i in range(3):
                for j in range(3):
                    s = 1.0 if i == j else 0.0
                    for a in range(nn):
                        s += ux[a, i] * G[g, a, j] / h
                    F[i, j] = s
            J = (F[0, 0] * (F[1, 1] * F[2, 2] - F[1, 2] * F[2, 1])
                 - F[0, 1] * (F[1, 0] * F[2, 2] - F[1, 2] * F[2, 0])
                 + F[0, 2] * (F[1, 0] * F[2, 1] - F[1, 1] * F[2, 0]))
            if J < jmin:
                jmin = J
            # C = F^T F
            c00 = F[0, 0] * F[0, 0] + F[1, 0] * F[1, 0] + F[2, 0] * F[2, 0]
            c11 = F[0, 1] * F[0, 1] + F[1, 1] * F[1, 1] + F[2, 1] * F[2, 1]
            c22 = F[0, 2] * F[0, 2] + F[1, 2] * F[1, 2] + F[2, 2] * F[2, 2]
            c01 = F[0, 0] * F[0, 1] + F[1, 0] * F[1, 1] + F[2, 0] * F[2, 1]
            c12 = F[0, 1] * F[0, 2] + F[1, 1] * F[1, 2] + F[2, 1] * F[2, 2]
            c02 = F[0, 0] * F[0, 2] + F[1, 0] * F[1, 2] + F[2, 0] * F[2, 2]
            if svk:
                e00, e11, e22 = 0.5 * (c00 - 1.0), 0.5 * (c11 - 1.0), 0.5 * (c22 - 1.0)
                tr = e00 + e11 + e22
                S[0, 0] = lam * tr + 2 * mu * e00
                S[1, 1] = lam * tr + 2 * mu * e11
                S[2, 2] = lam * tr + 2 * mu * e22
                S[0, 1] = S[1, 0] = mu * c01
                S[1, 2] = S[2, 1] = mu * c12
                S[0, 2] = S[2, 0] = mu * c02
                for i in range(3):
                    for j in range(3):
                        for k in range(3):
                            for l in range(3):
                                A[i, j, k, l] = (lam * (i == j) * (k == l)
                                                 + mu * ((i == k) * (j == l) + (i == l) * (j == k)))
            else:
                # C^-1 by cofactors (C symmetric); det C = J^2
                d = c00 * (c11 * c22 - c12 * c12) - c01 * (c01 * c22 - c12 * c02) + c02 * (c01 * c12 - c11 * c02)
                Ci[0, 0] = (c11 * c22 - c12 * c12) / d
                Ci[1, 1] = (c00 * c22 - c02 * c02) / d
                Ci[2, 2] = (c00 * c11 - c01 * c01) / d
                Ci[0, 1] = Ci[1, 0] = (c02 * c12 - c01 * c22) / d
                Ci[1, 2] = Ci[2, 1] = (c01 * c02 - c00 * c12) / d
                Ci[0, 2] = Ci[2, 0] = (c01 * c12 - c02 * c11) / d
                lnJ = np.log(max(J, 1e-12))
                for i in range(3):
                    for j in range(3):
                        S[i, j] = mu * ((1.0 if i == j else 0.0) - Ci[i, j]) + lam * lnJ * Ci[i, j]
                cc = mu - lam * lnJ
                for i in range(3):
                    for j in range(3):
                        for k in range(3):
                            for l in range(3):
                                A[i, j, k, l] = (lam * Ci[i, j] * Ci[k, l]
                                                 + cc * (Ci[i, k] * Ci[j, l] + Ci[i, l] * Ci[j, k]))
            for i in range(3):
                for l in range(3):
                    P[i, l] = F[i, 0] * S[0, l] + F[i, 1] * S[1, l] + F[i, 2] * S[2, l]
            for a in range(nn):
                for i in range(3):
                    fx[3 * a + i] += w * (P[i, 0] * G[g, a, 0] + P[i, 1] * G[g, a, 1] + P[i, 2] * G[g, a, 2]) / h
            # Q_iljn = F_ik A_klpn F_jp + S_ln delta_ij
            for i in range(3):
                for l in range(3):
                    for p in range(3):
                        for n in range(3):
                            FA[i, l, p, n] = F[i, 0] * A[0, l, p, n] + F[i, 1] * A[1, l, p, n] + F[i, 2] * A[2, l, p, n]
            for i in range(3):
                for l in range(3):
                    for j in range(3):
                        for n in range(3):
                            v = FA[i, l, 0, n] * F[j, 0] + FA[i, l, 1, n] * F[j, 1] + FA[i, l, 2, n] * F[j, 2]
                            if i == j:
                                v += S[l, n]
                            Q[i, l, j, n] = v
            # K_aibj += wk G_al Q_iljn G_bn, upper triangle in (a, b)
            for a in range(nn):
                for i in range(3):
                    for j in range(3):
                        for n in range(3):
                            T[a, i, j, n] = G[g, a, 0] * Q[i, 0, j, n] + G[g, a, 1] * Q[i, 1, j, n] + G[g, a, 2] * Q[i, 2, j, n]
            for a in range(nn):
                for b in range(a, nn):
                    gb0, gb1, gb2 = G[g, b, 0], G[g, b, 1], G[g, b, 2]
                    for i in range(3):
                        for j in range(3):
                            kx[3 * a + i, 3 * b + j] += wk * (T[a, i, j, 0] * gb0 + T[a, i, j, 1] * gb1 + T[a, i, j, 2] * gb2)
        n3 = 3 * nn
        for r in range(n3):
            for c in range(r):
                if c // 3 < r // 3:
                    kx[r, c] = kx[c, r]
        return jmin

    @_nb.njit(cache=True, parallel=True)
    def _elements(u, edof, alpha, mod, h, lam, mu, svk, tangent, G, im):
        m = edof.shape[0]
        nn = G.shape[1]
        fe = np.zeros((m, 24))
        ke = np.zeros((m, 24, 24)) if tangent else np.zeros((1, 24, 24))
        jm = np.empty(m)
        anew = np.zeros((m, 3, 3)) if im else np.zeros((1, 3, 3))
        ra = np.zeros((m, 9)) if im else np.zeros((1, 9))
        cpl = np.zeros((m, 9, 24)) if im else np.zeros((1, 9, 24))
        for e in _nb.prange(m):
            ux = np.zeros((nn, 3))
            for a in range(8):
                for i in range(3):
                    ux[a, i] = u[edof[e, 3 * a + i]]
            if im:
                for a in range(3):
                    for i in range(3):
                        ux[8 + a, i] = alpha[e, a, i]
            fx = np.zeros(3 * nn)
            kx = np.zeros((3 * nn, 3 * nn))
            jm[e] = _gauss(ux, G, h, lam, mu, svk, fx, kx)
            me = mod[e]
            if not im:
                for r in range(24):
                    fe[e, r] = fx[r] * me
                if tangent:
                    for r in range(24):
                        for c in range(24):
                            ke[e, r, c] = kx[r, c] * me
                continue
            # condense the nine incompatible modes: one local Newton step
            kaa = kx[24:, 24:].copy()
            rhs = np.empty((9, 25))
            for r in range(9):
                rhs[r, 0] = fx[24 + r]
                for c in range(24):
                    rhs[r, 1 + c] = kx[c, 24 + r]          # K_au = K_ua^T
            sol = np.linalg.solve(kaa, rhs)
            for r in range(24):
                s = fx[r]
                for q in range(9):
                    s -= kx[r, 24 + q] * sol[q, 0]
                fe[e, r] = s * me
            if tangent:
                for r in range(24):
                    for c in range(24):
                        s = kx[r, c]
                        for q in range(9):
                            s -= kx[r, 24 + q] * sol[q, 1 + c]
                        ke[e, r, c] = s * me
            for a in range(3):
                for i in range(3):
                    anew[e, a, i] = alpha[e, a, i] - sol[3 * a + i, 0]
            for q in range(9):
                ra[e, q] = fx[24 + q]
                for c in range(24):
                    cpl[e, q, c] = sol[q, 1 + c]
        return fe, ke, jm, anew, ra, cpl

    @_nb.njit(cache=True, nogil=True)
    def _scatter(data, slots, ke, fint, fe, edof, tangent):
        m = edof.shape[0]
        for e in range(m):
            for r in range(24):
                fint[edof[e, r]] += fe[e, r]
            if tangent:
                for a in range(8):
                    for b in range(8):
                        p = 8 * a + b
                        for i in range(3):
                            for j in range(3):
                                data[slots[p, e, 3 * i + j]] += ke[e, 3 * a + i, 3 * b + j]


def elements(u, edof, alpha, mod, h, nu, kind, tangent, grads):
    """All elements' (fint (m, 24), ke (m, 24, 24), det F per element,
    alpha_new, R_alpha (m, 9), K_aa^-1 K_au (m, 9, 24)), scaled by the
    moduli (fint, ke) as nonlinear._assemble scatters them. grads: (8, n, 3)
    unit-cube gradients, n = 8 (plain) or 11 (incompatible modes)."""
    lam, mu = nu / ((1 + nu) * (1 - 2 * nu)), 1.0 / (2 * (1 + nu))
    im = alpha is not None
    a = np.ascontiguousarray(alpha, float) if im else np.zeros((1, 3, 3))
    return _elements(np.ascontiguousarray(u, float), np.ascontiguousarray(edof), a,
                     np.ascontiguousarray(mod, float), float(h), lam, mu, kind == "svk", bool(tangent),
                     np.ascontiguousarray(grads, float), im)


def scatter(n_dof, nnz, slots, edof, fe, ke, tangent):
    """Global internal force and the CSR data array of the tangent."""
    fint = np.zeros(n_dof)
    data = np.zeros(nnz) if tangent else np.zeros(1)
    if ke is None or not tangent:
        ke = np.zeros((1, 24, 24))
    _scatter(data, np.ascontiguousarray(slots), np.ascontiguousarray(ke), fint, np.ascontiguousarray(fe),
             np.ascontiguousarray(edof), bool(tangent))
    return fint, (data if tangent else None)
