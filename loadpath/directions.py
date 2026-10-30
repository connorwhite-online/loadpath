"""
Loads whose direction is not known: a force of known size that may point
anywhere ("direction": "any"), or anywhere within a cone round its nominal
direction ("cone": degrees).

Compliance is quadratic in the force, so for f(d) = f0 + sum_i d_i b_i (f0
the load case's fixed loads, b_i the uncertain force pointing along axis i,
d a unit vector) it is

    c(d) = c00 + 2 g.d + d' A d,   c00 = f0.u0,  g_i = b_i.u0,  A_ij = b_i.u_j

with u0, u_i the displacements under f0 and b_i. Four solves (three when
nothing else loads the case) give the compliance in every direction, and
the worst one is a 3x3 problem: with no fixed load it is the largest
eigenvalue of A, exactly. By the envelope theorem the worst case's
sensitivity is the ordinary compliance sensitivity of the displacement
under the worst force, u0 + sum_i d_i u_i, so the optimizer's update is
unchanged (robust compliance, Ben-Tal and Nemirovski's worst case over an
ellipsoid of loads, here the sphere or a cap of it).

A von Mises stress at a point is the same shape (a quadratic in d through
the six stress components), which is how verification finds the worst
direction for stress.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["Spread", "Worst", "worst_direction", "expand", "collapse", "fibonacci_sphere", "worst_stress_direction"]


@dataclass
class Spread:
    """One load case's uncertain force, on a mesh's dofs."""
    base: np.ndarray        # the case's fixed loads (the rest of the case)
    basis: list             # 3 force vectors: the uncertain force along x, y, z
    nominal: np.ndarray     # unit vector: the force as written
    cos_half: float = -1.0  # cone: cos of the half-angle round `nominal`; -1 is any direction

    @property
    def kind(self) -> str:
        return "any" if self.cos_half <= -1.0 else "cone"

    def load(self, d) -> np.ndarray:
        d = np.asarray(d, float)
        return self.base + d[0] * self.basis[0] + d[1] * self.basis[1] + d[2] * self.basis[2]

    def with_base(self, extra) -> "Spread":
        return Spread(self.base + extra, self.basis, self.nominal, self.cos_half)


def fibonacci_sphere(n: int) -> np.ndarray:
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    t = np.pi * (1.0 + 5 ** 0.5) * i
    return np.column_stack([r * np.cos(t), r * np.sin(t), z])


_SAMPLES = fibonacci_sphere(600)


def _to_cap(d, axis, cos_half):
    """Nearest point of the cap {|d| = 1, d.axis >= cos_half} to unit d."""
    if axis is None or cos_half <= -1.0:
        return d
    c = d @ axis
    if c >= cos_half:
        return d
    perp = d - c * axis
    n = np.linalg.norm(perp)
    if n < 1e-12:
        perp = np.cross(axis, [1.0, 0, 0] if abs(axis[0]) < 0.9 else [0, 1.0, 0])
        n = np.linalg.norm(perp)
    return cos_half * axis + np.sqrt(max(0.0, 1.0 - cos_half ** 2)) * perp / n


def worst_direction(c00: float, g, a, axis=None, cos_half: float = -1.0):
    """Maximize c00 + 2 g.d + d'Ad over unit d (within the cap round `axis`
    when cos_half > -1). Returns (d, value)."""
    g = np.asarray(g, float)
    a = 0.5 * (np.asarray(a, float) + np.asarray(a, float).T)

    def q(d):
        return float(c00 + 2.0 * g @ d + d @ a @ d)

    capped = axis is not None and cos_half > -1.0
    lam, vec = np.linalg.eigh(a)
    gn = float(np.linalg.norm(g))
    if not capped and gn <= 1e-12 * max(abs(lam[-1]), 1e-300):
        d = vec[:, -1]
        return d, q(d)
    # Candidates: the eigenvectors, a dense spread of directions, and the
    # nominal axis; then a monotone ascent from the best of them. The
    # quadratic on a sphere has at most a couple of local maxima, which 600
    # samples (7 degrees apart) separate.
    cands = np.vstack([vec.T, -vec.T, _SAMPLES] + ([axis[None]] if capped else []))
    if capped:
        cands = np.array([_to_cap(c, axis, cos_half) for c in cands])
    vals = c00 + 2.0 * cands @ g + np.einsum("ij,jk,ik->i", cands, a, cands)
    d = cands[int(np.argmax(vals))]
    # minorize-maximize: d <- normalize((A + sI) d + g) never lowers q on the
    # sphere for s >= -lambda_min (a shifted power step)
    s = max(0.0, -lam[0]) + 1e-12 * max(abs(lam[-1]), 1.0)
    best, bq = d, q(d)
    for _ in range(500):
        n = (a + s * np.eye(3)) @ d + g
        nn = np.linalg.norm(n)
        if nn < 1e-300:
            break
        dn = _to_cap(n / nn, axis, cos_half) if capped else n / nn
        if np.linalg.norm(dn - d) < 1e-12:
            d = dn
            break
        d = dn
        qd = q(d)
        if qd > bq:
            best, bq = d, qd
    return best, bq


def expand(loads, spreads, extra=None):
    """The right-hand sides to solve for every case, and per case the
    indices of its fixed part and basis among them (None for a zero fixed
    part, which needs no solve). `extra` (self-weight) adds to every case's
    fixed part."""
    rhs, layout = [], []
    for ci, f in enumerate(loads):
        s = spreads[ci] if spreads else None
        if s is None:
            layout.append((len(rhs),))
            rhs.append(f + extra if extra is not None else f)
            continue
        base = s.base + extra if extra is not None else s.base
        if np.any(base):
            i0 = len(rhs)
            rhs.append(base)
        else:
            i0 = None
        layout.append((i0, len(rhs), len(rhs) + 1, len(rhs) + 2))
        rhs.extend(s.basis)
    return rhs, layout


@dataclass
class Worst:
    """One load case judged in its worst direction."""
    u: np.ndarray            # displacement under the worst force
    compliance: float        # f.u there, the exact worst case
    direction: np.ndarray | None
    force: np.ndarray
    # the optimizer's sensitivity: sum of w * strain energy of v over
    # (w, v), and the mean displacement under the same weights (self-weight)
    energy: list
    u_mean: np.ndarray


# Smoothing of the worst case for the optimizer. The exact worst direction
# jumps when two directions tie (a square section is as weak sideways as
# down), and following it alone the design swings between them. The
# optimizer instead
# follows a soft maximum over directions, each weighted by
# exp(SMOOTH * (c - c_max) / c_max): directions within a few percent of the
# worst share the gradient, as for a repeated eigenvalue. (With the
# optimizer's sensitivity damping, simp.optimize, 10 and 40 land within 3%.)
SMOOTH = 40.0


def collapse(rhs, layout, us, spreads, smooth: float = SMOOTH):
    """Per case, a Worst: the exact worst direction and its compliance, and
    the smoothed sensitivity terms."""
    out = []
    for ci, idx in enumerate(layout):
        if len(idx) == 1:
            u, f = us[idx[0]], rhs[idx[0]]
            out.append(Worst(u, float(f @ u), None, f, [(1.0, u)], u))
            continue
        s = spreads[ci]
        i0, ib = idx[0], idx[1:]
        ub = [us[i] for i in ib]
        bb = [rhs[i] for i in ib]
        a = np.array([[bb[i] @ ub[j] for j in range(3)] for i in range(3)])
        a = 0.5 * (a + a.T)
        if i0 is None:
            c00, g, u0, f0 = 0.0, np.zeros(3), np.zeros_like(ub[0]), 0.0
        else:
            u0, f0 = us[i0], rhs[i0]
            c00 = float(f0 @ u0)
            g = np.array([f0 @ ub[j] for j in range(3)])
        d, c = worst_direction(c00, g, a, s.nominal, s.cos_half)
        u = u0 + d[0] * ub[0] + d[1] * ub[1] + d[2] * ub[2]
        f = f0 + d[0] * bb[0] + d[1] * bb[1] + d[2] * bb[2]
        # soft maximum over a spread of directions (and the worst itself):
        # sum_k w_k (1, d_k)(1, d_k)' = W, then strain energy of U q for W's
        # eigenpairs (mu, q), U = [u0, u1, u2, u3]
        dirs = np.vstack([_cap_samples(s), d[None]])
        vals = c00 + 2.0 * dirs @ g + np.einsum("ij,jk,ik->i", dirs, a, dirs)
        w = np.exp(smooth * (vals - c) / max(abs(c), 1e-300))
        w /= w.sum()
        dd = np.column_stack([np.ones(len(dirs)), dirs])
        wm = (dd * w[:, None]).T @ dd
        mu, q = np.linalg.eigh(wm)
        uu = [u0] + ub
        energy = [(float(m), sum(q[k, j] * uu[k] for k in range(4)))
                  for j, m in enumerate(mu) if m > 1e-9 * mu[-1]]
        mean = dd.T @ w
        u_mean = sum(mean[k] * uu[k] for k in range(4))
        out.append(Worst(u, c, d, f, energy, u_mean))
    return out


# von Mises squared as a quadratic form on (sx, sy, sz, txy, tyz, tzx)
_VM = np.array([[1, -.5, -.5, 0, 0, 0], [-.5, 1, -.5, 0, 0, 0], [-.5, -.5, 1, 0, 0, 0],
                [0, 0, 0, 3, 0, 0], [0, 0, 0, 0, 3, 0], [0, 0, 0, 0, 0, 3]], float)


def _cap_samples(spread: Spread, n: int = 600) -> np.ndarray:
    if spread.cos_half <= -1.0:
        return fibonacci_sphere(n)
    # a Fibonacci spread over the cap only: z uniform on [cos_half, 1]
    i = np.arange(n) + 0.5
    z = 1.0 - (1.0 - spread.cos_half) * i / n
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    t = np.pi * (1.0 + 5 ** 0.5) * i
    local = np.column_stack([r * np.cos(t), r * np.sin(t), z])
    a = spread.nominal
    e1 = np.cross(a, [1.0, 0, 0] if abs(a[0]) < 0.9 else [0, 1.0, 0])
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(a, e1)
    return np.vstack([local @ np.array([e1, e2, a]), a[None]])


def worst_stress_direction(spread: Spread, s0, sb, mask, top: int = 40):
    """The direction of the uncertain force that raises the peak von Mises
    over the nodes in `mask` most. s0: nodal stress (n, 6) under the case's
    fixed loads (or None); sb: the three (n, 6) stresses under the basis.
    Samples directions for the peak, then solves the exact 3x3 problem at
    the nodes that came out on top. Returns (d, peak von Mises)."""
    idx = np.nonzero(mask)[0]
    if idx.size == 0:
        idx = np.arange(sb[0].shape[0])
    s = np.stack([b[idx] for b in sb], axis=1)              # (m, 3, 6)
    z = s0[idx] if s0 is not None else np.zeros((idx.size, 6))
    dirs = _cap_samples(spread)
    best = np.zeros(idx.size)
    for k in range(0, len(dirs), 32):
        sig = z[:, None, :] + np.einsum("kd,mdj->mkj", dirs[k:k + 32], s)
        vm2 = np.einsum("mkj,jl,mkl->mk", sig, _VM, sig)
        best = np.maximum(best, vm2.max(axis=1))
    order = np.argsort(best)[::-1][:top]
    bd, bv = spread.nominal, -1.0
    for m in order:
        sq = s[m] @ _VM                                      # (3, 6)
        a = sq @ s[m].T
        g = sq @ z[m]
        c00 = float(z[m] @ _VM @ z[m])
        d, v = worst_direction(c00, g, a, spread.nominal, spread.cos_half)
        if v > bv:
            bd, bv = d, v
    return bd, float(np.sqrt(max(bv, 0.0)))
