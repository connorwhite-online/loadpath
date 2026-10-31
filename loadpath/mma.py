"""
The Method of Moving Asymptotes (Svanberg 1987; the 2007 formulation in
"MMA and GCMMA — two methods for nonlinear optimization"), written for
loadpath from the published method so it stays MIT.

MMA is what lets an optimization carry real constraints (a stress limit, a
deflection limit, several at once) instead of the single volume constraint
optimality criteria can handle. Each iteration builds a convex separable
approximation of the problem around the current design, using moving
asymptotes that tighten when the design oscillates and relax when it moves
steadily, and solves it exactly with a primal-dual interior-point method.
The problem solved is

    minimize    f0(x) + a0 z + sum_i (c_i y_i + 0.5 d_i y_i^2)
    subject to  f_i(x) - a_i z - y_i <= 0,   xmin <= x <= xmax,
                y_i >= 0, z >= 0

with the usual choice a0 = 1, a_i = 0, d_i = 1 and large c_i, which makes
the artificial y_i zero whenever the real constraints can be met.

Sized for topology optimization: n (design cells) is large and m
(constraints) is small, so the Newton steps solve an m x m system.
"""
from __future__ import annotations

import numpy as np

__all__ = ["MMA"]


class MMA:
    def __init__(self, n, m, xmin=0.0, xmax=1.0, move=0.2, c=1000.0, asyinit=0.5,
                 asyincr=1.2, asydecr=0.7):
        self.n, self.m = n, m
        self.xmin = np.broadcast_to(np.asarray(xmin, float), (n,)).copy()
        self.xmax = np.broadcast_to(np.asarray(xmax, float), (n,)).copy()
        self.move = move
        self.a0 = 1.0
        self.a = np.zeros(m)
        self.c = np.full(m, float(c))
        self.d = np.ones(m)
        self.asyinit, self.asyincr, self.asydecr = asyinit, asyincr, asydecr
        self.xold1 = None
        self.xold2 = None
        self.low = None
        self.upp = None
        self.iter = 0

    def tighten(self, factor):
        """Pull the asymptotes toward the last design by `factor`: a more
        curved, more conservative model for the next step (what GCMMA's
        inner loop does, without its extra function evaluations)."""
        if self.xold1 is not None and self.low is not None:
            self.low = self.xold1 - factor * (self.xold1 - self.low)
            self.upp = self.xold1 + factor * (self.upp - self.xold1)

    def update(self, x, f0, df0, f, df):
        """One MMA step from design x with objective value/gradient (f0, df0)
        and constraint values/gradients (f: (m,), df: (m, n)). Returns the new
        design."""
        x = np.asarray(x, float)
        f = np.asarray(f, float).reshape(self.m)
        df = np.asarray(df, float).reshape(self.m, self.n)
        self.iter += 1
        xmin, xmax = self.xmin, self.xmax
        span = xmax - xmin
        if self.iter <= 2 or self.low is None:
            self.low = x - self.asyinit * span
            self.upp = x + self.asyinit * span
        else:
            zzz = (x - self.xold1) * (self.xold1 - self.xold2)
            factor = np.ones(self.n)
            factor[zzz > 0] = self.asyincr
            factor[zzz < 0] = self.asydecr
            self.low = x - factor * (self.xold1 - self.low)
            self.upp = x + factor * (self.upp - self.xold1)
            self.low = np.clip(self.low, x - 10 * span, x - 0.01 * span)
            self.upp = np.clip(self.upp, x + 0.01 * span, x + 10 * span)
        low, upp = self.low, self.upp
        albefa = 0.1
        alfa = np.maximum.reduce([low + albefa * (x - low), x - self.move * span, xmin])
        beta = np.minimum.reduce([upp - albefa * (upp - x), x + self.move * span, xmax])

        raa0 = 1e-5
        xmami = np.maximum(span, 1e-5)
        ux1 = upp - x
        xl1 = x - low
        ux2, xl2 = ux1 * ux1, xl1 * xl1
        p0 = np.maximum(df0, 0.0)
        q0 = np.maximum(-df0, 0.0)
        pq0 = 0.001 * (p0 + q0) + raa0 / xmami
        p0 = (p0 + pq0) * ux2
        q0 = (q0 + pq0) * xl2
        P = np.maximum(df, 0.0)
        Q = np.maximum(-df, 0.0)
        PQ = 0.001 * (P + Q) + raa0 / xmami
        P = (P + PQ) * ux2
        Q = (Q + PQ) * xl2
        b = P @ (1.0 / ux1) + Q @ (1.0 / xl1) - f

        xnew = self._subsolv(low, upp, alfa, beta, p0, q0, P, Q, b)
        self.xold2 = self.xold1 if self.xold1 is not None else x.copy()
        self.xold1 = x.copy()
        return xnew

    def _subsolv(self, low, upp, alfa, beta, p0, q0, P, Q, b, epsimin=1e-7):
        m = self.m
        a0, a, c, d = self.a0, self.a, self.c, self.d
        x = 0.5 * (alfa + beta)
        y = np.ones(m)
        z = 1.0
        lam = np.ones(m)
        xsi = np.maximum(1.0 / (x - alfa), 1.0)
        eta = np.maximum(1.0 / (beta - x), 1.0)
        mu = np.maximum(1.0, 0.5 * c)
        zet = 1.0
        s = np.ones(m)
        epsi = 1.0

        def residual(x, y, z, lam, xsi, eta, mu, zet, s, epsi):
            ux1 = upp - x
            xl1 = x - low
            plam = p0 + P.T @ lam
            qlam = q0 + Q.T @ lam
            gvec = P @ (1.0 / ux1) + Q @ (1.0 / xl1)
            dpsidx = plam / (ux1 * ux1) - qlam / (xl1 * xl1)
            r = np.concatenate([
                dpsidx - xsi + eta,
                c + d * y - mu - lam,
                [a0 - zet - a @ lam],
                gvec - a * z - y + s - b,
                xsi * (x - alfa) - epsi,
                eta * (beta - x) - epsi,
                mu * y - epsi,
                [zet * z - epsi],
                lam * s - epsi,
            ])
            return r

        while epsi > epsimin:
            r = residual(x, y, z, lam, xsi, eta, mu, zet, s, epsi)
            norm = np.linalg.norm(r)
            rmax = np.abs(r).max()
            it = 0
            while rmax > 0.9 * epsi and it < 200:
                it += 1
                ux1 = upp - x
                xl1 = x - low
                ux2, xl2 = ux1 * ux1, xl1 * xl1
                ux3, xl3 = ux1 * ux2, xl1 * xl2
                plam = p0 + P.T @ lam
                qlam = q0 + Q.T @ lam
                gvec = P @ (1.0 / ux1) + Q @ (1.0 / xl1)
                GG = P / ux2 - Q / xl2
                dpsidx = plam / ux2 - qlam / xl2
                delx = dpsidx - epsi / (x - alfa) + epsi / (beta - x)
                dely = c + d * y - lam - epsi / y
                delz = a0 - a @ lam - epsi / z
                dellam = gvec - a * z - y - b + epsi / lam
                diagx = 2.0 * (plam / ux3 + qlam / xl3) + xsi / (x - alfa) + eta / (beta - x)
                diagy = d + mu / y
                diaglamyi = s / lam + 1.0 / diagy
                blam = dellam + dely / diagy - GG @ (delx / diagx)
                alam = np.diag(diaglamyi) + (GG / diagx) @ GG.T
                aa = np.zeros((m + 1, m + 1))
                aa[:m, :m] = alam
                aa[:m, m] = a
                aa[m, :m] = a
                aa[m, m] = -zet / z
                sol = np.linalg.solve(aa, np.concatenate([blam, [delz]]))
                dlam, dz = sol[:m], sol[m]
                dx = -delx / diagx - (GG.T @ dlam) / diagx
                dy = -dely / diagy + dlam / diagy
                dxsi = -xsi + epsi / (x - alfa) - xsi * dx / (x - alfa)
                deta = -eta + epsi / (beta - x) + eta * dx / (beta - x)
                dmu = -mu + epsi / y - mu * dy / y
                dzet = -zet + epsi / z - zet * dz / z
                ds = -s + epsi / lam - s * dlam / lam

                xx = np.concatenate([y, [z], lam, xsi, eta, mu, [zet], s])
                dxx = np.concatenate([dy, [dz], dlam, dxsi, deta, dmu, [dzet], ds])
                stmxx = np.max(-1.01 * dxx / xx)
                stmalfa = np.max(-1.01 * dx / (x - alfa))
                stmbeta = np.max(1.01 * dx / (beta - x))
                steg = 1.0 / max(stmalfa, stmbeta, stmxx, 1.0)
                old = (x, y, z, lam, xsi, eta, mu, zet, s)
                for _ in range(50):
                    x = old[0] + steg * dx
                    y = old[1] + steg * dy
                    z = old[2] + steg * dz
                    lam = old[3] + steg * dlam
                    xsi = old[4] + steg * dxsi
                    eta = old[5] + steg * deta
                    mu = old[6] + steg * dmu
                    zet = old[7] + steg * dzet
                    s = old[8] + steg * ds
                    r = residual(x, y, z, lam, xsi, eta, mu, zet, s, epsi)
                    newnorm = np.linalg.norm(r)
                    if newnorm < 2.0 * norm:
                        break
                    steg /= 2.0
                norm = newnorm
                rmax = np.abs(r).max()
            epsi *= 0.1
        self.lam = lam
        return x
