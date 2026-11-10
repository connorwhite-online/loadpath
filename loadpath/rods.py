"""
Elastic rods: how a wire, a cable or a flat flex lies in the room it is given.

A cable inside a product is not a line drawn between two connectors. It has
a length, it resists bending (and a flat flex resists it very differently
across its width than through its thickness), it carries twist from one end
to the other, it leaves each connector the way the connector points, and it
leans on whatever is in its way. Whether a stack of components can be built
often comes down to that: the lead that would have to fold tighter than its
bend radius, the flex that has to twist a quarter turn in 12 mm, the bundle
that pulls sideways on a 1 mm-pitch header when the joint it crosses turns.

This module is the physics: a discrete Cosserat rod (Simo-Reissner kinematics;
the same model the cable tools used in automotive packaging solve). The rod
is nodes joined by edges; each edge carries a frame (a unit quaternion) whose
third axis is the edge's tangent and whose first two are the cross-section's
principal axes. Its strains are

- stretch and shear, per edge:  G = R^T (x1 - x0) / l - e3
- bending and twist, per inner node:  K = (2 / l) Im(conj(q0) q1) - K0

and its energy is quadratic in them, with separate stiffnesses for bending
about each section axis (EI1, EI2), twist (GJ) and stretch (EA; shear is
held stiff, so the rod is a Kirchhoff rod in practice). Weight, contact
with obstacles given as signed distance fields (a penalty on points across
the section, so a flat cable is as wide as it is) and springs to fixed
points add to it.

`Rod.solve` finds a stable equilibrium by Newton's method on the exact
Hessian (the geometric terms are in, so it converges quadratically on a
loaded, bent rod and sees buckling), in the rotations' own tangent space,
with a banded Cholesky factorization: a rod of 150 edges takes a few
milliseconds an iteration. A Hessian that is not positive definite is
shifted until it is; a converged state that is not a minimum (a strut that
should have buckled) is pushed off along its softest mode and solved again.
`Rod.drive` walks a boundary condition (an end moved along an assembly
path, a joint turned through its range) in steps, halving a step that does
not converge.

Checked in tests/test_rods.py against closed forms: the cantilever's
elastica (Bisshopp and Drucker), a circle bent by end moments, a twisted
bar, Euler's buckling load and the writhing of a twisted rod (Greenhill),
a ribbon's two bending stiffnesses, and a rod laid on a plane.

Units: mm, N, MPa (so stiffnesses in N mm^2, weight in N/mm).
"""
from __future__ import annotations

import numpy as np

__all__ = ["Rod", "GridField", "SdfField", "MovingField", "frame_quat", "quat_matrix", "parallel_frames",
           "qmul", "qconj", "qexp"]

_E3 = np.array([0.0, 0.0, 1.0])


# --- quaternions (w, x, y, z), rows -------------------------------------------

def qmul(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    w = a[..., 0] * b[..., 0] - np.sum(a[..., 1:] * b[..., 1:], axis=-1)
    v = a[..., :1] * b[..., 1:] + b[..., :1] * a[..., 1:] + np.cross(a[..., 1:], b[..., 1:])
    return np.concatenate([w[..., None], v], axis=-1)


def qconj(a):
    a = np.asarray(a, float)
    return np.concatenate([a[..., :1], -a[..., 1:]], axis=-1)


def qexp(d):
    """The rotation by the vector d (radians, about d) as a quaternion."""
    d = np.asarray(d, float)
    th = np.linalg.norm(d, axis=-1, keepdims=True)
    half = 0.5 * th
    # sin(x/2)/x, safe at 0
    s = np.where(th > 1e-8, np.sin(half) / np.where(th > 1e-8, th, 1.0), 0.5 - th * th / 48.0)
    return np.concatenate([np.cos(half), s * d], axis=-1)


def quat_matrix(q):
    """Rotation matrices (n, 3, 3) of unit quaternions (n, 4): columns are
    the frame's axes d1, d2, d3."""
    q = np.asarray(q, float)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    R = np.empty(q.shape[:-1] + (3, 3))
    R[..., 0, 0] = 1 - 2 * (y * y + z * z)
    R[..., 0, 1] = 2 * (x * y - w * z)
    R[..., 0, 2] = 2 * (x * z + w * y)
    R[..., 1, 0] = 2 * (x * y + w * z)
    R[..., 1, 1] = 1 - 2 * (x * x + z * z)
    R[..., 1, 2] = 2 * (y * z - w * x)
    R[..., 2, 0] = 2 * (x * z - w * y)
    R[..., 2, 1] = 2 * (y * z + w * x)
    R[..., 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def frame_quat(d3, d1=None):
    """The quaternion of a frame whose third axis is d3 and whose first is
    d1 made square to it (any square direction when d1 is None or along d3)."""
    d3 = np.asarray(d3, float)
    d3 = d3 / np.linalg.norm(d3)
    if d1 is not None:
        d1 = np.asarray(d1, float)
        d1 = d1 - (d1 @ d3) * d3
    if d1 is None or np.linalg.norm(d1) < 1e-9:
        d1 = np.eye(3)[int(np.argmin(np.abs(d3)))]
        d1 = d1 - (d1 @ d3) * d3
    d1 = d1 / np.linalg.norm(d1)
    R = np.column_stack([d1, np.cross(d3, d1), d3])
    return _mat_quat(R)


def _mat_quat(R):
    t = np.trace(R)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2
        q = [0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s]
    else:
        i = int(np.argmax(np.diag(R)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = np.sqrt(1.0 + R[i, i] - R[j, j] - R[k, k]) * 2
        q = [0.0] * 4
        q[0] = (R[k, j] - R[j, k]) / s
        q[1 + i] = 0.25 * s
        q[1 + j] = (R[j, i] + R[i, j]) / s
        q[1 + k] = (R[k, i] + R[i, k]) / s
    q = np.asarray(q, float)
    return q / np.linalg.norm(q)


def parallel_frames(x, d1_start=None, twist: float = 0.0):
    """Edge frames along a polyline: the first frame's d1 carried along
    without twisting (parallel transport), then `twist` radians spread
    evenly from the first edge to the last."""
    x = np.asarray(x, float)
    t = np.diff(x, axis=0)
    t /= np.linalg.norm(t, axis=1, keepdims=True)
    n = len(t)
    q = np.empty((n, 4))
    q[0] = frame_quat(t[0], d1_start)
    d1 = quat_matrix(q[0])[:, 0]
    for i in range(1, n):
        ax = np.cross(t[i - 1], t[i])
        s, c = np.linalg.norm(ax), float(t[i - 1] @ t[i])
        if s > 1e-12:
            ax /= s
            ang = np.arctan2(s, c)
            d1 = d1 * np.cos(ang) + np.cross(ax, d1) * np.sin(ang) + ax * (ax @ d1) * (1 - np.cos(ang))
        q[i] = frame_quat(t[i], d1)
        if q[i] @ q[i - 1] < 0:
            q[i] = -q[i]
    if twist:
        a = twist * np.arange(n) / max(n - 1, 1)
        spin = np.column_stack([np.cos(a / 2), np.zeros(n), np.zeros(n), np.sin(a / 2)])
        q = qmul(q, spin)
    return q


def _skew(v):
    v = np.asarray(v, float)
    S = np.zeros(v.shape[:-1] + (3, 3))
    S[..., 0, 1], S[..., 0, 2] = -v[..., 2], v[..., 1]
    S[..., 1, 0], S[..., 1, 2] = v[..., 2], -v[..., 0]
    S[..., 2, 0], S[..., 2, 1] = -v[..., 1], v[..., 0]
    return S



# --- compiled kernels (numba; the numpy paths below are the fallback and the reference) ----

try:
    from numba import njit
    _HAVE_NB = True
except Exception:                                  # pragma: no cover
    _HAVE_NB = False

    def njit(*a, **k):
        def wrap(f):
            return f
        return wrap


@njit(cache=True)
def _nb_spline(c, o, h, p, d, g):
    """Cubic B-spline value and gradient of the coefficient grid c at points p."""
    nx, ny, nz = c.shape
    W = np.empty((3, 4))
    D = np.empty((3, 4))
    i0 = np.empty(3, np.int64)
    dims = (nx, ny, nz)
    for n in range(p.shape[0]):
        for k in range(3):
            u = (p[n, k] - o[k]) / h
            if u < 1.0:
                u = 1.0
            top = dims[k] - 2.001
            if u > top:
                u = top
            fl = int(np.floor(u))
            t = u - fl
            i0[k] = fl
            W[k, 0] = (1 - t) ** 3 / 6
            W[k, 1] = (3 * t ** 3 - 6 * t ** 2 + 4) / 6
            W[k, 2] = (-3 * t ** 3 + 3 * t ** 2 + 3 * t + 1) / 6
            W[k, 3] = t ** 3 / 6
            D[k, 0] = -(1 - t) ** 2 / 2
            D[k, 1] = (3 * t ** 2 - 4 * t) / 2
            D[k, 2] = (-3 * t ** 2 + 2 * t + 1) / 2
            D[k, 3] = t ** 2 / 2
        v = 0.0
        gx = 0.0
        gy = 0.0
        gz = 0.0
        for a in range(4):
            ia = min(max(i0[0] + a - 1, 0), nx - 1)
            for b in range(4):
                ib = min(max(i0[1] + b - 1, 0), ny - 1)
                for e in range(4):
                    ie = min(max(i0[2] + e - 1, 0), nz - 1)
                    cv = c[ia, ib, ie]
                    v += cv * W[0, a] * W[1, b] * W[2, e]
                    gx += cv * D[0, a] * W[1, b] * W[2, e]
                    gy += cv * W[0, a] * D[1, b] * W[2, e]
                    gz += cv * W[0, a] * W[1, b] * D[2, e]
        d[n] = v
        g[n, 0] = gx / h
        g[n, 1] = gy / h
        g[n, 2] = gz / h


@njit(cache=True)
def _nb_rot(q, R):
    w, x, y, z = q[0], q[1], q[2], q[3]
    R[0, 0] = 1 - 2 * (y * y + z * z)
    R[0, 1] = 2 * (x * y - w * z)
    R[0, 2] = 2 * (x * z + w * y)
    R[1, 0] = 2 * (x * y + w * z)
    R[1, 1] = 1 - 2 * (x * x + z * z)
    R[1, 2] = 2 * (y * z - w * x)
    R[2, 0] = 2 * (x * z - w * y)
    R[2, 1] = 2 * (y * z + w * x)
    R[2, 2] = 1 - 2 * (x * x + y * y)


@njit(cache=True)
def _nb_skew(v, S):
    S[0, 0] = 0.0
    S[1, 1] = 0.0
    S[2, 2] = 0.0
    S[0, 1] = -v[2]
    S[0, 2] = v[1]
    S[1, 0] = v[2]
    S[1, 2] = -v[0]
    S[2, 0] = -v[1]
    S[2, 1] = v[0]


@njit(cache=True)
def _nb_points(x, q, offs, Y):
    """Contact points: each edge's middle plus R c for every offset c."""
    R = np.empty((3, 3))
    n = q.shape[0]
    k = offs.shape[0]
    for i in range(n):
        _nb_rot(q[i], R)
        for j in range(k):
            for a in range(3):
                Y[i * k + j, a] = 0.5 * (x[i, a] + x[i + 1, a]) + R[a, 0] * offs[j, 0] + R[a, 1] * offs[j, 1] \
                    + R[a, 2] * offs[j, 2]


@njit(cache=True)
def _nb_assemble(x, q, l, B, S, K0, offs, pen, nrm, kc, mode, g, ab, u):
    """Elastic and contact energy; mode 1 adds the gradient into g, mode 2
    the Hessian into ab (upper banded) as well. Returns (energy, largest
    force resultant)."""
    n = q.shape[0]
    nk = offs.shape[0]
    E = 0.0
    carried = 0.0
    R = np.empty((3, 3))
    Sk = np.empty((3, 3))
    v = np.empty(3)
    f = np.empty(3)
    G = np.empty(3)
    ge = np.empty(9)
    J = np.empty((3, 9))
    He = np.empty((9, 9))
    nj = np.empty(9)
    F = np.empty(3)
    Fb = np.empty(3)
    for i in range(n):
        _nb_rot(q[i], R)
        li = l[i]
        for a in range(3):
            v[a] = (R[0, a] * (x[i + 1, 0] - x[i, 0]) + R[1, a] * (x[i + 1, 1] - x[i, 1])
                    + R[2, a] * (x[i + 1, 2] - x[i, 2])) / li
        G[0], G[1], G[2] = v[0], v[1], v[2] - 1.0
        for a in range(3):
            f[a] = li * S[a] * G[a]
            E += 0.5 * G[a] * f[a]
            if abs(S[a] * G[a]) > carried:
                carried = abs(S[a] * G[a])
        if mode >= 1:
            for a in range(3):
                rf = (R[a, 0] * f[0] + R[a, 1] * f[1] + R[a, 2] * f[2]) / li
                ge[a] = -rf
                ge[6 + a] = rf
            ge[3] = f[1] * v[2] - f[2] * v[1]
            ge[4] = f[2] * v[0] - f[0] * v[2]
            ge[5] = f[0] * v[1] - f[1] * v[0]
        if mode >= 2:
            _nb_skew(v, Sk)
            for a in range(3):
                for b in range(3):
                    J[a, b] = -R[b, a] / li
                    J[a, 3 + b] = Sk[a, b]
                    J[a, 6 + b] = R[b, a] / li
            for a in range(9):
                for b in range(9):
                    He[a, b] = li * (S[0] * J[0, a] * J[0, b] + S[1] * J[1, a] * J[1, b] + S[2] * J[2, a] * J[2, b])
            vf = v[0] * f[0] + v[1] * f[1] + v[2] * f[2]
            for a in range(3):
                for b in range(3):
                    He[3 + a, 3 + b] += 0.5 * (v[a] * f[b] + f[a] * v[b])
                He[3 + a, 3 + a] -= vf
            _nb_skew(f, Sk)
            for a in range(3):
                for b in range(3):
                    rfk = (R[a, 0] * Sk[0, b] + R[a, 1] * Sk[1, b] + R[a, 2] * Sk[2, b]) / li
                    He[a, 3 + b] += rfk
                    He[3 + b, a] += rfk
                    He[6 + a, 3 + b] -= rfk
                    He[3 + b, 6 + a] -= rfk
        # contact at the edge's middle, across the section
        cw = li / nk
        for k in range(nk):
            pk = pen[i, k]
            if pk <= 0.0:
                continue
            E += 0.5 * kc * cw * pk * pk
            if mode < 1:
                continue
            for a in range(3):
                F[a] = -kc * cw * pk * nrm[i, k, a]
            for a in range(3):
                Fb[a] = R[0, a] * F[0] + R[1, a] * F[1] + R[2, a] * F[2]
            c = offs[k]
            for a in range(3):
                ge[a] += 0.5 * F[a]
                ge[6 + a] += 0.5 * F[a]
            ge[3] += c[1] * Fb[2] - c[2] * Fb[1]
            ge[4] += c[2] * Fb[0] - c[0] * Fb[2]
            ge[5] += c[0] * Fb[1] - c[1] * Fb[0]
            if mode >= 2:
                _nb_skew(c, Sk)
                for b in range(3):
                    nj[b] = 0.5 * nrm[i, k, b]
                    nj[6 + b] = 0.5 * nrm[i, k, b]
                    t = 0.0
                    for a in range(3):
                        t -= nrm[i, k, a] * (R[a, 0] * Sk[0, b] + R[a, 1] * Sk[1, b] + R[a, 2] * Sk[2, b])
                    nj[3 + b] = t
                for a in range(9):
                    for b in range(9):
                        He[a, b] += kc * cw * nj[a] * nj[b]
                cF = c[0] * Fb[0] + c[1] * Fb[1] + c[2] * Fb[2]
                for a in range(3):
                    for b in range(3):
                        He[3 + a, 3 + b] += 0.5 * (c[a] * Fb[b] + Fb[a] * c[b])
                    He[3 + a, 3 + a] -= cF
        if mode >= 1:
            base = 6 * i
            for a in range(9):
                g[base + a] += ge[a]
            if mode >= 2:
                for a in range(9):
                    for b in range(a, 9):
                        ab[u + a - b, base + b] += He[a, b]
    # bending and twist at the inner nodes
    Ja = np.empty((3, 3))
    Jb = np.empty((3, 3))
    m = np.empty(3)
    pv = np.empty(3)
    Jab = np.empty((3, 6))
    Hb = np.empty((6, 6))
    idx = np.empty(6, np.int64)
    for j in range(n - 1):
        a0, a1, a2, a3 = q[j, 0], -q[j, 1], -q[j, 2], -q[j, 3]
        b0, b1, b2, b3 = q[j + 1, 0], q[j + 1, 1], q[j + 1, 2], q[j + 1, 3]
        w = a0 * b0 - a1 * b1 - a2 * b2 - a3 * b3
        pv[0] = a0 * b1 + b0 * a1 + (a2 * b3 - a3 * b2)
        pv[1] = a0 * b2 + b0 * a2 + (a3 * b1 - a1 * b3)
        pv[2] = a0 * b3 + b0 * a3 + (a1 * b2 - a2 * b1)
        if w < 0:
            w = -w
            pv[0], pv[1], pv[2] = -pv[0], -pv[1], -pv[2]
        lv = 0.5 * (l[j] + l[j + 1])
        for a in range(3):
            dK = 2.0 * pv[a] / lv - K0[j, a]
            m[a] = lv * B[a] * dK
            E += 0.5 * dK * m[a]
        if mode < 1:
            continue
        _nb_skew(pv, Sk)
        for a in range(3):
            for b in range(3):
                Ja[a, b] = Sk[a, b] / lv
                Jb[a, b] = Sk[a, b] / lv
            Ja[a, a] -= w / lv
            Jb[a, a] += w / lv
        for a in range(3):
            idx[a] = 6 * j + 3 + a
            idx[3 + a] = 6 * j + 9 + a
        for a in range(3):
            g[idx[a]] += Ja[0, a] * m[0] + Ja[1, a] * m[1] + Ja[2, a] * m[2]
            g[idx[3 + a]] += Jb[0, a] * m[0] + Jb[1, a] * m[1] + Jb[2, a] * m[2]
        if mode >= 2:
            for a in range(3):
                for b in range(3):
                    Jab[a, b] = Ja[a, b]
                    Jab[a, 3 + b] = Jb[a, b]
            for a in range(6):
                for b in range(6):
                    Hb[a, b] = lv * (B[0] * Jab[0, a] * Jab[0, b] + B[1] * Jab[1, a] * Jab[1, b]
                                     + B[2] * Jab[2, a] * Jab[2, b])
            mv = m[0] * pv[0] + m[1] * pv[1] + m[2] * pv[2]
            _nb_skew(m, Sk)
            for a in range(3):
                Hb[a, a] -= mv / (2 * lv)
                Hb[3 + a, 3 + a] -= mv / (2 * lv)
                for b in range(3):
                    hab = (0.5 * (pv[a] * m[b] + m[a] * pv[b]) + 0.5 * w * Sk[a, b]) / lv
                    if a == b:
                        hab -= 0.5 * mv / lv
                    Hb[a, 3 + b] += hab
                    Hb[3 + b, a] += hab
            for a in range(6):
                for b in range(a, 6):
                    ab[u + idx[a] - idx[b], idx[b]] += Hb[a, b]
    return E, carried


# --- obstacles ------------------------------------------------------------------

class SdfField:
    """An obstacle from any signed distance function f((n, 3)) -> (n,),
    negative inside the obstacle. Gradients by central differences."""

    def __init__(self, sdf, eps: float = 1e-3):
        self.sdf, self.eps = sdf, float(eps)

    def eval(self, p):
        p = np.asarray(p, float)
        d = self.sdf(p)
        g = np.empty_like(p)
        for k in range(3):
            e = np.zeros(3)
            e[k] = self.eps
            g[:, k] = (self.sdf(p + e) - self.sdf(p - e)) / (2 * self.eps)
        return d, g


class GridField:
    """An obstacle sampled once on a grid and read back by a cubic B-spline
    (twice differentiable, so Newton sees a smooth wall): the room around a
    whole layout as one field. `values` are signed distances at
    origin + index * pitch; outside the grid the edge value carries on."""

    def __init__(self, values, origin, pitch: float):
        from scipy import ndimage
        self.shape = np.asarray(values.shape)
        self.c = np.ascontiguousarray(ndimage.spline_filter(np.asarray(values, float), order=3, mode="nearest"))
        self.fast = True
        self.o = np.asarray(origin, float)
        self.h = float(pitch)

    @classmethod
    def sample(cls, sdf, lo, hi, pitch: float, pad: float = 2.0, cells: float = 4e6):
        lo, hi = np.asarray(lo, float) - pad, np.asarray(hi, float) + pad
        pitch = max(float(pitch), float((np.prod(hi - lo) / cells) ** (1 / 3)))
        n = np.ceil((hi - lo) / pitch).astype(int) + 1
        ax = [lo[i] + np.arange(n[i]) * pitch for i in range(3)]
        P = np.stack(np.meshgrid(*ax, indexing="ij"), -1).reshape(-1, 3)
        v = np.concatenate([sdf(P[i:i + 400_000]) for i in range(0, len(P), 400_000)])
        return cls(v.reshape(tuple(n)), lo, pitch)

    def reach(self, p):
        """Distance for path finding: as eval, but outside the sampled box
        a free point's distance grows with how far out it is (eval carries
        the edge value on, which is right for contact and too small for
        asking how much room there is)."""
        p = np.asarray(p, float)
        d = self.eval(p)[0]
        lo, hi = self.o + self.h, self.o + (self.shape - 2) * self.h
        out = np.linalg.norm(p - np.clip(p, lo, hi), axis=1)
        return np.where(d > 0, d + out, d)

    def eval(self, p):
        p = np.ascontiguousarray(p, float)
        if _HAVE_NB and self.fast:
            d = np.empty(len(p))
            g = np.empty((len(p), 3))
            _nb_spline(self.c, self.o, self.h, p, d, g)
            return d, g
        u = (p - self.o) / self.h
        u = np.clip(u, 1.0, self.shape - 2.001)          # edge values carry on outside
        i0 = np.floor(u).astype(int)
        f = u - i0
        W, D = [], []
        for k in range(3):
            t = f[:, k]
            W.append(np.stack([(1 - t) ** 3 / 6, (3 * t ** 3 - 6 * t ** 2 + 4) / 6,
                               (-3 * t ** 3 + 3 * t ** 2 + 3 * t + 1) / 6, t ** 3 / 6], axis=1))
            D.append(np.stack([-(1 - t) ** 2 / 2, (3 * t ** 2 - 4 * t) / 2,
                               (-3 * t ** 2 + 2 * t + 1) / 2, t ** 2 / 2], axis=1))
        idx = [np.clip(i0[:, k:k + 1] + np.arange(-1, 3)[None, :], 0, self.shape[k] - 1) for k in range(3)]
        C = self.c[idx[0][:, :, None, None], idx[1][:, None, :, None], idx[2][:, None, None, :]]
        d = np.einsum("nijk,ni,nj,nk->n", C, W[0], W[1], W[2])
        g = np.stack([np.einsum("nijk,ni,nj,nk->n", C, D[0], W[1], W[2]),
                      np.einsum("nijk,ni,nj,nk->n", C, W[0], D[1], W[2]),
                      np.einsum("nijk,ni,nj,nk->n", C, W[0], W[1], D[2])], axis=1) / self.h
        return d, g


class MovingField:
    """An obstacle carried by a rigid motion: field(R^T (p - t)). Set `R`
    and `t` between solves (a part on its way in, a link that turns)."""

    def __init__(self, field, R=None, t=None):
        self.field = field
        self.R = np.eye(3) if R is None else np.asarray(R, float)
        self.t = np.zeros(3) if t is None else np.asarray(t, float)

    def eval(self, p):
        d, g = self.field.eval((np.asarray(p, float) - self.t) @ self.R)
        return d, g @ self.R.T

    def reach(self, p):
        q = (np.asarray(p, float) - self.t) @ self.R
        return self.field.reach(q) if hasattr(self.field, "reach") else self.field.eval(q)[0]


# --- the rod --------------------------------------------------------------------

class Rod:
    """A discrete Cosserat rod. See the module docstring.

    x: (n+1, 3) node positions. q: (n, 4) edge frames (unit quaternions,
    d3 along the edge). bend: (EI1, EI2) about the section's d1 and d2;
    twist: GJ; stretch: EA (None: stiff enough to be inextensible).
    lengths: rest length per edge (default: as drawn). rest: (n-1, 3) rest
    curvature and twist per inner node in the section's frame (default
    straight). weight: force per length, a 3-vector (gravity on the cable's
    mass). radius: the section's half thickness for contact; offsets: (k, 3)
    points across the section in its own frame at which contact is felt
    (default the centreline: a round wire; a flat cable takes points across
    its width)."""

    def __init__(self, x, q=None, *, bend, twist, stretch=None, lengths=None, rest=None, weight=None,
                 radius: float = 0.0, offsets=None):
        self.x = np.array(x, float)
        self.n = len(self.x) - 1
        if self.n < 2:
            raise ValueError("a rod needs at least two edges")
        self.q = parallel_frames(self.x) if q is None else np.array(q, float)
        self.q /= np.linalg.norm(self.q, axis=1, keepdims=True)
        self.l = np.linalg.norm(np.diff(self.x, axis=0), axis=1) if lengths is None else \
            np.broadcast_to(np.asarray(lengths, float), (self.n,)).copy()
        b = np.broadcast_to(np.asarray(bend, float), (2,))
        self.B = np.array([b[0], b[1], float(twist)])
        self.EA_true = None if stretch is None else float(stretch)
        self.K0 = np.zeros((self.n - 1, 3)) if rest is None else np.array(rest, float)
        self.weight = None if weight is None else np.asarray(weight, float)
        self.radius = float(radius)
        self.offsets = np.zeros((1, 3)) if offsets is None else np.asarray(offsets, float).reshape(-1, 3)
        self.obstacles = []
        self.clearance = 0.0
        self.contact_k = None          # N/mm^2 per mm of penetration; None: set from the bending stiffness
        self.exempt = None             # (n,) bool: edges that touch nothing (inside a connector's own body)
        self.springs = []              # (node, point, k N/mm)
        self.loads = []                # (node, force 3-vector N): dead loads
        self.stiff = 300.0             # stretch and shear stiffness cap, in EI / l^2
        self.fast = _HAVE_NB           # compiled kernels (False: the numpy reference)
        self.N = 6 * self.n + 3
        self.fixed = np.zeros(self.N, bool)
        self._index()

    # dofs: node i -> 6i..6i+2 (translation), edge i -> 6i+3..6i+5 (rotation
    # increments in the edge's own frame)
    def node_dofs(self, i):
        i = i % (self.n + 1)
        return np.arange(6 * i, 6 * i + 3)

    def edge_dofs(self, i):
        i = i % self.n
        return np.arange(6 * i + 3, 6 * i + 6)

    def clamp(self, end: int, roll: bool = True, direction: bool = True):
        """Hold an end (0 or -1): its node, and its edge's direction (and
        roll about it). roll=False leaves the end free to spin about its own
        axis (a round wire in a crimp); direction=False is a pin."""
        self.fixed[self.node_dofs(0 if end == 0 else self.n)] = True
        e = self.edge_dofs(0 if end == 0 else self.n - 1)
        if direction:
            self.fixed[e[:2]] = True
            self.fixed[e[2]] = bool(roll)
        return self

    def hold(self, node: int, direction: bool = False):
        """Hold an inner node where it is (a clip); with direction, the
        edge after it too (a clip that grips)."""
        self.fixed[self.node_dofs(node)] = True
        if direction:
            self.fixed[self.edge_dofs(min(node, self.n - 1))] = True
        return self

    def _index(self):
        n = self.n
        self.u = 11
        e = np.arange(n)
        ds = 6 * e[:, None] + np.arange(9)[None, :]                    # stretch (and contact): x_i, th_i, x_i+1
        v = np.arange(1, n)
        db = np.concatenate([6 * (v[:, None] - 1) + 3 + np.arange(3)[None, :],
                             6 * v[:, None] + 3 + np.arange(3)[None, :]], axis=1)   # bend: th_i-1, th_i
        self._ds, self._db = ds, db

        def band(d):
            I = np.repeat(d[:, :, None], d.shape[1], axis=2)
            J = np.repeat(d[:, None, :], d.shape[1], axis=1)
            up = I <= J
            return up, ((self.u + I - J) * self.N + J)[up]
        self._ups, self._flats = band(ds)
        self._upb, self._flatb = band(db)

    # --- stiffness in use -------------------------------------------------------
    def _S(self):
        """Stretch and shear stiffness: the real EA when given, capped so a
        rod stays numerically inextensible without swamping the bending
        terms (strain under a cable's own loads stays under 1e-4 either way)."""
        cap = self.stiff * float(self.B[:2].max()) / float(np.mean(self.l)) ** 2
        ea = cap if self.EA_true is None else min(self.EA_true, cap)
        return np.array([ea, ea, ea])

    def _contact_k(self):
        if self.contact_k is not None:
            return float(self.contact_k)
        # a line load bending the rod to a 5 mm radius sinks 0.02 mm
        return float(np.sqrt(self.B[0] * self.B[1])) / (0.02 * 5.0 ** 3)

    # --- strains ----------------------------------------------------------------
    def strains(self, x=None, q=None):
        x = self.x if x is None else x
        q = self.q if q is None else q
        R = quat_matrix(q)
        d = (x[1:] - x[:-1]) / self.l[:, None]
        v = np.einsum("nji,nj->ni", R, d)                                # R^T d
        p = qmul(qconj(q[:-1]), q[1:])
        p = np.where(p[:, :1] < 0, -p, p)
        lv = 0.5 * (self.l[:-1] + self.l[1:])
        K = 2.0 * p[:, 1:] / lv[:, None]
        return R, v, p, lv, K

    def _extras(self, x, g=None, ab=None):
        """Weight, springs and dead loads: energy, and into g and ab."""
        E = 0.0
        if self.weight is not None:
            nl = self._node_len()
            E -= float(np.sum(nl[:, None] * x * self.weight))
            if g is not None:
                g[(6 * np.arange(self.n + 1))[:, None] + np.arange(3)] -= nl[:, None] * self.weight
        for node, pt, k in self.springs:
            dd = x[node] - pt
            E += 0.5 * k * float(dd @ dd)
            if g is not None:
                g[6 * node:6 * node + 3] += k * dd
            if ab is not None:
                ab[self.u, 6 * node:6 * node + 3] += k
        for node, F in self.loads:
            F = np.asarray(F, float)
            E -= float(F @ x[node])
            if g is not None:
                g[6 * node:6 * node + 3] -= F
        return E

    def _fast(self, x, q, mode):
        n, k = self.n, len(self.offsets)
        x = np.ascontiguousarray(x)
        q = np.ascontiguousarray(q)
        if self.obstacles:
            Y = np.empty((n * k, 3))
            _nb_points(x, q, self.offsets, Y)
            d = np.full(len(Y), np.inf)
            gr = np.zeros_like(Y)
            for ob in self.obstacles:
                dd, gg = ob.eval(Y)
                if len(self.obstacles) == 1:
                    d, gr = dd, gg
                else:
                    closer = dd < d
                    d = np.where(closer, dd, d)
                    gr = np.where(closer[:, None], gg, gr)
            if self.exempt is not None:
                d = np.array(d, float)
                d.reshape(n, k)[self.exempt] = np.inf
            pen = np.maximum(self.radius + self.clearance - d, 0.0).reshape(n, k)
            nrm = np.ascontiguousarray(gr).reshape(n, k, 3)
            self._pen = (pen, nrm, d.reshape(n, k))
            kc = self._contact_k()
        else:
            pen, nrm, kc = np.zeros((n, k)), np.zeros((n, k, 3)), 0.0
        g = np.zeros(self.N) if mode >= 1 else np.zeros(1)
        ab = np.zeros((self.u + 1, self.N)) if mode >= 2 else np.zeros((1, 1))
        E, carried = _nb_assemble(x, q, self.l, self.B, self._S(), self.K0, self.offsets, pen, nrm, kc, mode, g, ab,
                                  self.u)
        self._carried = carried
        E += self._extras(x, g if mode >= 1 else None, ab if mode >= 2 else None)
        return E, (g if mode >= 1 else None), (ab if mode >= 2 else None)

    def energy(self, x=None, q=None):
        x = self.x if x is None else x
        q = self.q if q is None else q
        if self.fast:
            return float(self._fast(x, q, 0)[0])
        R, v, p, lv, K = self.strains(x, q)
        G = v - _E3
        E = 0.5 * np.sum(self.l[:, None] * G * G * self._S()) + 0.5 * np.sum(lv[:, None] * (K - self.K0) ** 2 * self.B)
        if self.weight is not None:
            E -= float(np.sum(self._node_len()[:, None] * x * self.weight))
        for node, pt, k in self.springs:
            E += 0.5 * k * float(np.sum((x[node] - pt) ** 2))
        for node, F in self.loads:
            E -= float(np.asarray(F, float) @ x[node])
        if self.obstacles:
            pen, nrm, _, dist = self._contact(x, R)
            self._pen = (pen, nrm, dist)
            E += 0.5 * self._contact_k() * float(np.sum(self._cw()[:, None] * pen ** 2))
        return float(E)

    def _node_len(self):
        m = np.zeros(self.n + 1)
        m[:-1] += 0.5 * self.l
        m[1:] += 0.5 * self.l
        return m

    def _cw(self):
        return self.l / len(self.offsets)

    def _contact(self, x, R):
        """Penetration at each contact point (edges x offsets), the wall's
        normal there, the point's offset in world axes."""
        mid = 0.5 * (x[:-1] + x[1:])
        cw = np.einsum("nij,kj->nki", R, self.offsets)                   # (n, k, 3)
        Y = (mid[:, None, :] + cw).reshape(-1, 3)
        d = np.full(len(Y), np.inf)
        g = np.zeros_like(Y)
        for ob in self.obstacles:
            dd, gg = ob.eval(Y)
            closer = dd < d
            d = np.where(closer, dd, d)
            g = np.where(closer[:, None], gg, g)
        k = len(self.offsets)
        if self.exempt is not None:
            d.reshape(self.n, k)[self.exempt] = np.inf
        pen = np.maximum(self.radius + self.clearance - d, 0.0)
        return pen.reshape(self.n, k), g.reshape(self.n, k, 3), cw, d.reshape(self.n, k)

    # --- gradient and Hessian ------------------------------------------------------
    def assemble(self, hessian: bool = True):
        """Energy, gradient (N,), and the Hessian in upper banded storage."""
        if self.fast:
            return self._fast(self.x, self.q, 2 if hessian else 1)
        n, N = self.n, self.N
        x, q, l = self.x, self.q, self.l
        R, v, p, lv, K = self.strains()
        S = self._S()
        G = v - _E3
        f = l[:, None] * G * S                                           # body-frame force resultants x l
        self._carried = float(np.abs(G * S).max())
        E = 0.5 * float(np.sum(G * f)) + 0.5 * float(np.sum(lv[:, None] * (K - self.K0) ** 2 * self.B))
        g = np.zeros(N)
        Rf = np.einsum("nij,nj->ni", R, f) / l[:, None]
        ge = np.concatenate([-Rf, np.cross(f, v), Rf], axis=1)           # (n, 9)
        # bending
        w, pv = p[:, 0], p[:, 1:]
        m = lv[:, None] * (K - self.K0) * self.B
        Sk = _skew(pv)
        eye = np.eye(3)[None]
        Ja = -(w[:, None, None] * eye - Sk) / lv[:, None, None]
        Jb = (w[:, None, None] * eye + Sk) / lv[:, None, None]
        gb = np.concatenate([np.einsum("nji,nj->ni", Ja, m), np.einsum("nji,nj->ni", Jb, m)], axis=1)
        He = Hb = None
        if hessian:
            J = np.zeros((n, 3, 9))
            Rt = np.swapaxes(R, 1, 2) / l[:, None, None]
            J[:, :, 0:3] = -Rt
            J[:, :, 3:6] = _skew(v)
            J[:, :, 6:9] = Rt
            He = np.einsum("nki,nk,nkj->nij", J, l[:, None] * S[None, :], J)
            # geometric terms: the rotation's second order, and force x rotation
            vf = np.einsum("ni,nj->nij", v, f)
            He[:, 3:6, 3:6] += 0.5 * (vf + np.swapaxes(vf, 1, 2)) - np.einsum("ni,ni->n", v, f)[:, None, None] * eye
            RF = np.einsum("nij,njk->nik", R, _skew(f)) / l[:, None, None]
            He[:, 0:3, 3:6] += RF
            He[:, 3:6, 0:3] += np.swapaxes(RF, 1, 2)
            He[:, 6:9, 3:6] -= RF
            He[:, 3:6, 6:9] -= np.swapaxes(RF, 1, 2)
            Jab = np.concatenate([Ja, Jb], axis=2)                       # (n-1, 3, 6)
            Hb = np.einsum("nki,nk,nkj->nij", Jab, lv[:, None] * self.B[None, :], Jab)
            mv = np.einsum("ni,ni->n", m, pv)
            dg = -(mv / (2 * lv))[:, None, None] * eye
            Hb[:, 0:3, 0:3] += dg
            Hb[:, 3:6, 3:6] += dg
            vm = np.einsum("ni,nj->nij", pv, m)
            Hab = (0.5 * (vm + np.swapaxes(vm, 1, 2)) - 0.5 * mv[:, None, None] * eye
                   + 0.5 * w[:, None, None] * _skew(m)) / lv[:, None, None]
            Hb[:, 0:3, 3:6] += Hab
            Hb[:, 3:6, 0:3] += np.swapaxes(Hab, 1, 2)
        # contact: points across the section at each edge's middle
        self._pen = None
        if self.obstacles:
            kc = self._contact_k()
            pen, nrm, cw, dist = self._contact(x, R)
            self._pen = (pen, nrm, dist)
            cwt = self._cw()
            E += 0.5 * kc * float(np.sum(cwt[:, None] * pen ** 2))
            F = -(kc * cwt[:, None] * pen)[:, :, None] * nrm             # dE/dy per point (n, k, 3)
            Fs = F.sum(axis=1)
            cb = np.broadcast_to(self.offsets[None], (n, len(self.offsets), 3))
            Fb = np.einsum("nji,nkj->nki", R, F)                         # body frame
            ge[:, 0:3] += 0.5 * Fs
            ge[:, 6:9] += 0.5 * Fs
            ge[:, 3:6] += np.cross(cb, Fb).sum(axis=1)
            if hessian and (pen > 0).any():
                act = (pen > 0)
                # y = mid + R c: dy = dx/2 + dx'/2 - R [c]x dth
                Jc = np.zeros((n, len(self.offsets), 3, 9))
                Jc[:, :, :, 0:3] = 0.5 * np.eye(3)
                Jc[:, :, :, 6:9] = 0.5 * np.eye(3)
                Jc[:, :, :, 3:6] = -np.einsum("nij,kjl->nkil", R, _skew(self.offsets))
                nj = np.einsum("nki,nkij->nkj", nrm, Jc)
                wgt = kc * cwt[:, None] * act
                He += np.einsum("nk,nki,nkj->nij", wgt, nj, nj)
                cF = np.einsum("nki,nkj->nij", cb, Fb)
                He[:, 3:6, 3:6] += 0.5 * (cF + np.swapaxes(cF, 1, 2)) - np.einsum("nki,nki->n", cb, Fb)[:, None, None] * eye
        np.add.at(g, self._ds, ge)
        np.add.at(g, self._db, gb)
        if self.weight is not None:
            nl = self._node_len()
            E -= float(np.sum(nl[:, None] * x * self.weight))
            g[(6 * np.arange(n + 1))[:, None] + np.arange(3)] -= nl[:, None] * self.weight
        ab = None
        if hessian:
            ab = np.bincount(self._flats, He[self._ups], minlength=(self.u + 1) * N)
            ab += np.bincount(self._flatb, Hb[self._upb], minlength=(self.u + 1) * N)
            ab = ab.reshape(self.u + 1, N)
        for node, pt, k in self.springs:
            dd = x[node] - pt
            E += 0.5 * k * float(dd @ dd)
            g[6 * node:6 * node + 3] += k * dd
            if hessian:
                ab[self.u, 6 * node:6 * node + 3] += k
        for node, F in self.loads:
            E -= float(np.asarray(F, float) @ x[node])
            g[6 * node:6 * node + 3] -= np.asarray(F, float)
        return E, g, ab

    def _apply(self, step):
        x = self.x + step.reshape(-1)[self._xi]
        dq = qexp(step[self._ti])
        q = qmul(self.q, dq)
        q /= np.linalg.norm(q, axis=1, keepdims=True)
        return x, q

    @property
    def _xi(self):
        return (6 * np.arange(self.n + 1))[:, None] + np.arange(3)

    @property
    def _ti(self):
        return (6 * np.arange(self.n))[:, None] + 3 + np.arange(3)

    def _mask(self):
        u, N = self.u, self.N
        free = ~self.fixed
        M = np.zeros((u + 1, N), bool)
        for k in range(u + 1):             # entry (i = j - k, j) at ab[u - k, j]
            j = np.arange(k, N)
            M[u - k, j] = free[j] & free[j - k]
        return M

    # --- solve ------------------------------------------------------------------
    def solve(self, tol: float = 1e-7, max_iter: int = 300, stable: bool = True, max_move: float | None = None,
              verbose: bool = False) -> dict:
        """Newton to a stable equilibrium from the current state (kept in
        self.x, self.q). Returns {converged, iterations, energy, residual,
        stable}. The residual is the largest unbalanced force or moment,
        relative to the rod's own scale (EI / L^2)."""
        from scipy.linalg import LinAlgError, solveh_banded
        M = self._mask()
        fixed = self.fixed
        L = float(self.l.sum())
        fscale0 = float(self.B[:2].max()) / L ** 2 + 1e-30
        lscale = float(np.mean(self.l))
        max_move = max_move or max(2.0 * lscale, 0.02 * L)
        # among obstacles: no step longer than a wall is thick, and none
        # that puts the centreline into one (a loop pressed on a 2 mm wall
        # would otherwise step through it and settle on the far side)
        reff = self.radius + self.clearance
        deep = max(0.75 * reff, 0.15)
        if self.obstacles:
            max_move = min(max_move, max(1.0, 1.5 * self.radius))
        mu = 0.0
        it = 0
        kicks = 0
        res = np.inf
        E = None
        ok = False
        unstable = False
        E_kick = None
        while it < max_iter:
            it += 1
            E, g, ab = self.assemble()
            g[fixed] = 0.0
            # forces on nodes; moments on edges scaled by the edge length
            gs = g.copy()
            gs[self._ti.ravel()] /= lscale
            # against the forces the rod is carrying, or its own scale when it carries none
            fscale = max(fscale0, self._carried)
            res = float(np.abs(gs).max()) / fscale
            ab = ab * M
            ab[self.u, fixed] = 1.0
            diag = np.maximum(np.abs(ab[self.u]), 1e-12)
            if res < tol:
                if not stable or kicks >= 4:
                    ok = True
                    break
                # a minimum, or a saddle (a strut that should have buckled)?
                # Modes softer than a thousandth of the rod's own stiffness
                # are neutral (an arch between two in-line clamps swings
                # freely about them), not unstable
                lam, mode = self._soft_mode(ab)
                if lam > -1e-3 * fscale0 / L or (E_kick is not None and E > E_kick - 1e-9 * abs(E_kick)):
                    ok = True
                    unstable = lam <= -1e-3 * fscale0 / L
                    break
                step = mode
                step[fixed] = 0.0
                step *= 0.02 * L / max(np.abs(step[self._xi]).max(), lscale * np.abs(step[self._ti]).max(), 1e-12)
                best = None
                for sgn in (1.0, -1.0):
                    xn, qn = self._apply(sgn * step)
                    En = self.energy(xn, qn)
                    if self.obstacles and self._deepest() > deep:
                        continue
                    if best is None or En < best[0]:
                        best = (En, xn, qn)
                if best is None:
                    ok = True
                    break
                E_kick = E
                self.x, self.q = best[1], best[2]
                kicks += 1
                continue
            while True:
                try:
                    a2 = ab.copy()
                    a2[self.u] += mu * diag
                    step = solveh_banded(a2, -g, lower=False, check_finite=False)
                    if g @ step < 0:
                        break
                except LinAlgError:
                    pass
                mu = max(4.0 * mu, 1e-6)
                if mu > 1e12:
                    return self._status(False, it, E, res, False)
            # no node further than max_move, no frame further than 0.4 rad, in one step
            big = max(np.abs(step[self._xi]).max() / max_move, np.abs(step[self._ti]).max() / 0.4)
            if big > 1.0:
                step = step / big
            slope = float(g @ step)
            a = 1.0
            took = False
            pen0 = self._deepest() if self.obstacles else 0.0
            for _ in range(30):
                xn, qn = self._apply(a * step)
                En = self.energy(xn, qn)
                if En <= E + 1e-4 * a * slope + 1e-13 * abs(E) and \
                        (not self.obstacles or self._deepest() <= max(deep, pen0 + 1e-9)):
                    took = True
                    break
                a *= 0.5
            if not took:
                mu = max(10.0 * mu, 1e-6)
                if mu > 1e12:
                    break
                continue
            self.x, self.q = xn, qn
            mu = mu / 4.0 if a == 1.0 else mu
            if mu < 1e-9:
                mu = 0.0
            if verbose:
                print(f"  it {it}: E {En:.6g} res {res:.3g} mu {mu:.2g} a {a:.3g}")
        return self._status(ok, it, E, res, ok and not unstable and kicks < 4)

    def _deepest(self):
        p = getattr(self, "_pen", None)
        return float(p[0].max()) if p is not None else 0.0

    def _soft_mode(self, ab):
        from scipy.linalg import eig_banded
        w, V = eig_banded(ab, lower=False, select="i", select_range=(0, 0), check_finite=False)
        return float(w[0]), V[:, 0].copy()

    def _status(self, ok, it, E, res, stable):
        return {"converged": bool(ok), "iterations": int(it), "energy": float(E), "residual": float(res),
                "stable": bool(stable)}

    def drive(self, set_to, steps: int = 10, on_step=None, min_step: float = 1 / 256, **kw) -> dict:
        """Walk a boundary condition: set_to(s) for s from 0 to 1 moves
        the held ends (and anything else, an obstacle say), and the rod is
        solved at each step from the last. A step that does not converge is
        halved (down to min_step of the whole way). on_step(s, status) is
        called after each solved step. Returns {converged, steps, reached}."""
        s, ds = 0.0, 1.0 / max(int(steps), 1)
        done = 0
        prev = None                      # (s, x, q) of the step before the last
        while s < 1.0 - 1e-12:
            s2 = min(1.0, s + ds)
            keep = (self.x.copy(), self.q.copy())
            st = None
            if prev is not None and s - prev[0] > 1e-12:
                # secant predictor: carry on the way the last step went
                r = (s2 - s) / (s - prev[0])
                dq = qmul(self.q, qconj(prev[2]))
                dq = np.where(dq[:, :1] < 0, -dq, dq)
                vn = np.linalg.norm(dq[:, 1:], axis=1, keepdims=True)
                rv = dq[:, 1:] * (2.0 * np.arctan2(vn, dq[:, :1]) / np.maximum(vn, 1e-300))
                self.x = self.x + r * (self.x - prev[1])
                self.q = qmul(qexp(r * rv), self.q)
                set_to(s2)
                good = True
                if self.obstacles:
                    self.energy()
                    good = self._deepest() <= max(0.75 * (self.radius + self.clearance), 0.15)
                st = self.solve(**kw) if good else {"converged": False}
                if not st["converged"]:
                    self.x, self.q = keep[0].copy(), keep[1].copy()
                    st = None
            if st is None:
                set_to(s2)
                st = self.solve(**kw)
            if st["converged"]:
                prev = (s, keep[0], keep[1])
                s = s2
                done += 1
                if on_step:
                    on_step(s, st)
                continue
            self.x, self.q = keep
            set_to(s)
            ds *= 0.5
            if ds < min_step:
                return {"converged": False, "steps": done, "reached": s}
        return {"converged": True, "steps": done, "reached": 1.0}

    def set_end(self, end: int, point=None, quat=None):
        """Move a held end: its node to `point`, its edge's frame to `quat`."""
        i, e = (0, 0) if end == 0 else (self.n, self.n - 1)
        if point is not None:
            self.x[i] = np.asarray(point, float)
        if quat is not None:
            qn = np.asarray(quat, float)
            qn = qn / np.linalg.norm(qn)
            self.q[e] = qn if qn @ self.q[e] >= 0 else -qn

    # --- what comes out ---------------------------------------------------------
    def curvature(self):
        """Bending curvature (1/mm, about the section's d1 and d2) and twist
        rate (rad/mm) at each inner node, exact for the angle between its
        two edges."""
        _, _, p, lv, K = self.strains()
        s = np.linalg.norm(p[:, 1:], axis=1)
        ang = 2.0 * np.arctan2(s, p[:, 0])                # the true angle; K uses 2 sin(a/2)
        scale = np.where(s > 1e-12, ang / np.maximum(2.0 * s, 1e-300), 1.0)
        return K * scale[:, None]

    def bend_radius(self):
        """Radius of the centreline's bend at each inner node (mm; inf where straight)."""
        k = np.linalg.norm(self.curvature()[:, :2], axis=1)
        return np.where(k > 1e-12, 1.0 / np.maximum(k, 1e-300), np.inf)

    def internal(self):
        """The force each edge carries (n, 3) and the moment at each inner
        node (n-1, 3), in world axes (N, N mm)."""
        R, v, p, lv, K = self.strains()
        nf = np.einsum("nij,nj->ni", R, (v - _E3) * self._S())
        m = (K - self.K0) * self.B
        # the node's frame: its two edges' frames, halfway
        qm = self.q[:-1] + np.where(np.sum(self.q[:-1] * self.q[1:], axis=1, keepdims=True) < 0, -1, 1) * self.q[1:]
        qm /= np.linalg.norm(qm, axis=1, keepdims=True)
        return nf, np.einsum("nij,nj->ni", quat_matrix(qm), m)

    def end_load(self, end: int):
        """What the rod asks of whatever holds an end: the force (N) and the
        moment about the end node (N mm) the holder must supply, world axes."""
        nf, m = self.internal()
        if end == 0:
            F = -nf[0]
            Mo = -m[0] - np.cross(self.x[1] - self.x[0], nf[0])
        else:
            F = nf[-1]
            Mo = m[-1] + np.cross(self.x[-2] - self.x[-1], -nf[-1])
        return F, Mo

    def contact_state(self):
        """Per edge: deepest penetration (mm), nearest wall (mm, less the
        radius), and the contact force on it (N)."""
        if not self.obstacles:
            z = np.zeros(self.n)
            return {"penetration": z, "gap": np.full(self.n, np.inf), "force": z}
        R = quat_matrix(self.q)
        pen, nrm, cw, dist = self._contact(self.x, R)
        f = self._contact_k() * self._cw()[:, None] * pen
        return {"penetration": pen.max(axis=1), "gap": dist.min(axis=1) - self.radius, "force": f.sum(axis=1)}

    def length(self):
        return float(np.linalg.norm(np.diff(self.x, axis=0), axis=1).sum())
