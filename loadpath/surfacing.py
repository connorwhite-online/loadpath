"""
Freeform surfacing: smooth B-spline surfaces drawn from curves.

A product's outside is drawn as curves: a few section curves across it, the
silhouettes it shows from the front, the side and the top, a part line. A
surfacing package (Alias, Fusion's form tools, Rhino) builds the skin
through them. Built instead from boxes, cylinders and fillets, or from
blended distance fields, the skin picks up what nobody drew: a lump where
two blends overlap, a phantom ridge where one blend runs into another, a
flat spot where a fillet runs out. This module builds the skin the
surfacing way, on OpenCascade, so a script can do what a surfacing artist
does:

- `spline` / `fit_curve`: a curve through or near points (open or closed),
  C2 everywhere;
- `fit_surface`: one B-spline surface through a grid of points, least
  squares with a fairing term (the surface's bending energy), so it lies on
  the points and bends no more than they ask. Periodic in either direction
  for closed sections. One surface is C2 inside, so it has no edges of its
  own to show;
- `fair_patch`: the variational patch. A surface (a height over a plane)
  pinned where it must be (a flat plateau, a known edge) and otherwise as
  fair as it can be: the minimum-bending surface between the constraints.
  That is a loft from a body up to a flat face with no ridge or dip nobody
  asked for;
- `network`: a Gordon surface through a curve network (sections one way,
  rails the other), every curve interpolated exactly;
- `loft`: closed sections swept along, optionally held to rails: each
  section is warped smoothly so it passes through every rail, then the
  network is surfaced;
- `fill`: an N-sided patch filling a boundary, held G0 (position) or G1
  (tangent) to the faces round it (OpenCascade's BRepFill_Filling, whose
  bounded constraints stop at G1). Where curvature has to carry across,
  build the region as one surface instead (`fit_surface`, `network`,
  `fair_patch` are C2 inside), and check edges with
  `surfacequality.edge_continuity`;
- `face`, `sew_solid`, `mesh`, `write_step`, `sdf`: the result as a
  face, a closed solid, a trimesh, a STEP file a CAD user can open, and a
  signed distance function for loadpath's field pipelines (an exact skin
  in, a field model's cuts and halves on it).

`surfacequality` checks what comes out (curvature, zebra, continuity across
edges); `sketchmatch` measures it against the designer's sketches.

Parameters run 0..1 in both directions on everything fitted here. Needs the
OpenCascade bindings (`pip install "loadpath[step]"`).
"""
from __future__ import annotations

import numpy as np

_BASIS_CACHE: dict = {}


def _occ():
    try:
        import OCP  # noqa: F401
    except ImportError as e:  # pragma: no cover
        raise ImportError("surfacing needs OpenCascade: pip install 'loadpath[step]'") from e


# ---- B-spline bases, read from OpenCascade itself ---------------------------

def _knots(n, degree, periodic):
    """(knots, mults) for n poles: uniform, clamped at both ends (open) or
    uniform with no ends (periodic). Parameter 0..1."""
    if periodic:
        k = np.linspace(0.0, 1.0, n + 1)
        return k, np.ones(n + 1, int)
    nk = n - degree + 1
    if nk < 2:
        raise ValueError(f"{n} poles are too few for degree {degree}")
    k = np.linspace(0.0, 1.0, nk)
    m = np.ones(nk, int)
    m[0] = m[-1] = degree + 1
    return k, m


def _arr1(vals, kind="real"):
    from OCP.OCP.collections import Array1_double, Array1_int
    a = (Array1_double if kind == "real" else Array1_int)(1, len(vals))
    for i, v in enumerate(vals):
        a.SetValue(i + 1, float(v) if kind == "real" else int(v))
    return a


def _curve(poles, degree=3, periodic=False):
    """A Geom_BSplineCurve with uniform knots on 0..1."""
    from OCP.Geom import Geom_BSplineCurve
    from OCP.gp import gp_Pnt
    from OCP.OCP.collections import Array1_gp_Pnt as TColgp_Array1OfPnt
    P = np.asarray(poles, float)
    n = len(P)
    k, m = _knots(n, degree, periodic)
    a = TColgp_Array1OfPnt(1, n)
    for i, p in enumerate(P):
        a.SetValue(i + 1, gp_Pnt(*map(float, p)))
    return Geom_BSplineCurve(a, _arr1(k), _arr1(m, "int"), degree, periodic)


def basis(n, degree, periodic, t, deriv=0):
    """(len(t), n) matrix: row a holds every pole's weight at parameter t[a]
    (deriv 1, 2: the weights of the derivatives). Read from OpenCascade by
    evaluating a curve with one unit pole at a time, so a fitted net and the
    surface OpenCascade builds from it agree to the last digit."""
    t = np.asarray(t, float)
    key = (n, degree, bool(periodic), deriv, t.tobytes())
    if key in _BASIS_CACHE:
        return _BASIS_CACHE[key]
    _occ()
    from OCP.gp import gp_Pnt, gp_Vec
    B = np.zeros((len(t), n))
    for j in range(n):
        poles = np.zeros((n, 3))
        poles[j, 0] = 1.0
        c = _curve(poles, degree, periodic)
        p, v1, v2 = gp_Pnt(), gp_Vec(), gp_Vec()
        for a, ta in enumerate(t):
            if deriv == 0:
                B[a, j] = c.Value(float(ta)).X()
            elif deriv == 1:
                c.D1(float(ta), p, v1)
                B[a, j] = v1.X()
            else:
                c.D2(float(ta), p, v1, v2)
                B[a, j] = v2.X()
    if len(_BASIS_CACHE) > 256:
        _BASIS_CACHE.clear()
    _BASIS_CACHE[key] = B
    return B


def _second_difference(n, periodic):
    """(n, n) or (n-2, n) second differences of the poles: the discrete
    bending of a control polygon."""
    if periodic:
        D = np.zeros((n, n))
        for i in range(n):
            D[i, (i - 1) % n], D[i, i], D[i, (i + 1) % n] = 1.0, -2.0, 1.0
        return D
    D = np.zeros((max(n - 2, 0), n))
    for i in range(n - 2):
        D[i, i:i + 3] = (1.0, -2.0, 1.0)
    return D


# ---- curves -----------------------------------------------------------------

def chord_params(P, closed=False):
    """Chord-length parameters of points along a polyline, 0..1 (for a
    closed one, 1 is back at the first point, which is not repeated)."""
    P = np.asarray(P, float)
    Q = np.vstack([P, P[:1]]) if closed else P
    s = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(Q, axis=0), axis=1))]
    s = s / s[-1] if s[-1] > 0 else np.linspace(0, 1, len(Q))
    return s[:-1] if closed else s


def resample(P, n, closed=False, start=None):
    """n points evenly spaced by arc length along a polyline. For a closed
    one, `start` (a point) picks where the first lands: the polyline point
    nearest it (keep sections of a loft starting at the same place, or the
    surface twists)."""
    P = np.asarray(P, float)
    if closed and start is not None:
        i = int(np.argmin(np.linalg.norm(P - np.asarray(start, float), axis=1)))
        P = np.roll(P, -i, axis=0)
    t = chord_params(P, closed)
    Q = np.vstack([P, P[:1]]) if closed else P
    tt = np.r_[t, 1.0] if closed else t
    u = np.linspace(0, 1, n, endpoint=not closed)
    return np.column_stack([np.interp(u, tt, Q[:, k]) for k in range(P.shape[1])])


class Curve:
    """A fitted B-spline curve: `poles`, `degree`, `periodic`, and the
    OpenCascade curve (`geom`). `__call__(t)` evaluates at parameters 0..1."""

    def __init__(self, poles, degree=3, periodic=False):
        self.poles = np.asarray(poles, float)
        self.degree, self.periodic = int(degree), bool(periodic)
        self.geom = _curve(self.poles, self.degree, self.periodic)

    def __call__(self, t):
        t = np.atleast_1d(np.asarray(t, float))
        if self.periodic:
            t = np.mod(t, 1.0)
        return np.array([[c.X(), c.Y(), c.Z()] for c in (self.geom.Value(float(a)) for a in t)])

    def sample(self, n):
        return self(np.linspace(0, 1, n, endpoint=not self.periodic))

    def derivative(self, t, k=1):
        from OCP.gp import gp_Pnt, gp_Vec
        out = []
        for a in np.atleast_1d(np.asarray(t, float)):
            p, v1, v2 = gp_Pnt(), gp_Vec(), gp_Vec()
            if k == 1:
                self.geom.D1(float(a), p, v1)
                out.append((v1.X(), v1.Y(), v1.Z()))
            else:
                self.geom.D2(float(a), p, v1, v2)
                out.append((v2.X(), v2.Y(), v2.Z()))
        return np.array(out)

    def curvature(self, n=200):
        """Curvature (1/mm) at n even parameters."""
        t = np.linspace(0, 1, n, endpoint=not self.periodic)
        d1, d2 = self.derivative(t, 1), self.derivative(t, 2)
        return np.linalg.norm(np.cross(d1, d2), axis=1) / np.maximum(np.linalg.norm(d1, axis=1) ** 3, 1e-30)

    def edge(self):
        from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeEdge
        return BRepBuilderAPI_MakeEdge(self.geom).Edge()


def fit_curve(points, n_poles=None, closed=False, degree=3, fair=0.02, params=None):
    """A C2 curve near `points` (an ordered polyline, 2D or 3D): least
    squares with a bending penalty `fair` (relative), so pencil wobble is
    smoothed out rather than followed. Closed curves come out periodic."""
    _occ()
    P = np.asarray(points, float)
    dim = P.shape[1]
    if dim == 2:
        P = np.column_stack([P, np.zeros(len(P))])
    n = int(n_poles or max(degree + 1, min(len(P) // 3, 40)))
    t = chord_params(P, closed) if params is None else np.asarray(params, float)
    B = basis(n, degree, closed, t)
    D = _second_difference(n, closed)
    lam = fair * len(P) / n
    A = B.T @ B + lam * D.T @ D
    C = np.linalg.solve(A + 1e-12 * np.eye(n), B.T @ P)
    c = Curve(C, degree, closed)
    c.dim = dim
    c.fit_error = float(np.linalg.norm(B @ C - P, axis=1).max())
    return c


def spline(points, closed=False, degree=3):
    """A curve through every point (as many poles as points)."""
    P = np.asarray(points, float)
    if len(P) <= degree:
        raise ValueError("spline: need more points than the degree")
    return fit_curve(P, n_poles=len(P), closed=closed, degree=degree, fair=0.0)


# ---- surfaces ---------------------------------------------------------------

class Surface:
    """A fitted B-spline surface. `poles` (nu, nv, 3), degrees, periodic
    flags, the OpenCascade surface (`geom`). `__call__(u, v)` evaluates on
    parameters 0..1 (a grid when both are 1-D arrays and grid=True)."""

    def __init__(self, poles, udeg=3, vdeg=3, uperiodic=False, vperiodic=False):
        self.poles = np.asarray(poles, float)
        self.udeg, self.vdeg = int(udeg), int(vdeg)
        self.uperiodic, self.vperiodic = bool(uperiodic), bool(vperiodic)
        self.geom = self._build()
        self.fit = {}

    def _build(self):
        from OCP.Geom import Geom_BSplineSurface
        from OCP.gp import gp_Pnt
        from OCP.OCP.collections import Array2_gp_Pnt as TColgp_Array2OfPnt
        nu, nv, _ = self.poles.shape
        ku, mu = _knots(nu, self.udeg, self.uperiodic)
        kv, mv = _knots(nv, self.vdeg, self.vperiodic)
        a = TColgp_Array2OfPnt(1, nu, 1, nv)
        for i in range(nu):
            for j in range(nv):
                a.SetValue(i + 1, j + 1, gp_Pnt(*map(float, self.poles[i, j])))
        return Geom_BSplineSurface(a, _arr1(ku), _arr1(kv), _arr1(mu, "int"), _arr1(mv, "int"),
                                   self.udeg, self.vdeg, self.uperiodic, self.vperiodic)

    def __call__(self, u, v, grid=True):
        u, v = np.atleast_1d(np.asarray(u, float)), np.atleast_1d(np.asarray(v, float))
        nu, nv, _ = self.poles.shape
        Bu = basis(nu, self.udeg, self.uperiodic, np.mod(u, 1.0) if self.uperiodic else u)
        Bv = basis(nv, self.vdeg, self.vperiodic, np.mod(v, 1.0) if self.vperiodic else v)
        if grid:
            return np.einsum("ai,ijk,bj->abk", Bu, self.poles, Bv)
        return np.einsum("ai,ijk,aj->ak", Bu, self.poles, Bv)

    def grid(self, nu=60, nv=60):
        u = np.linspace(0, 1, nu, endpoint=not self.uperiodic)
        v = np.linspace(0, 1, nv, endpoint=not self.vperiodic)
        return u, v, self(u, v)

    def face(self, tol=1e-6):
        return face(self, tol)


def _fair_matrices(nu, nv, udeg, vdeg, up, vp, q=None):
    """The thin-plate bending energy of the surface's map (u, v) -> x, as a
    quadratic form in the poles: Suu^2 + 2 Suv^2 + Svv^2 integrated over a
    quadrature grid. Returned as three (m, nu*nv) operators."""
    q = q or (max(2 * nu, 24), max(2 * nv, 24))
    tu = (np.arange(q[0]) + 0.5) / q[0]
    tv = (np.arange(q[1]) + 0.5) / q[1]
    B0u, B1u, B2u = (basis(nu, udeg, up, tu, d) for d in (0, 1, 2))
    B0v, B1v, B2v = (basis(nv, vdeg, vp, tv, d) for d in (0, 1, 2))
    return [np.kron(B2u, B0v), np.sqrt(2.0) * np.kron(B1u, B1v), np.kron(B0u, B2v)], q


def _solve(rows, rhs, fair_ops, lam, n, hard=None):
    """min |rows x - rhs|^2 + lam sum |F x|^2, with hard rows (H x = h) kept
    exactly by a Lagrange multiplier."""
    A = rows.T @ rows
    for F in fair_ops:
        A = A + lam * (F.T @ F)
    b = rows.T @ rhs
    A = A + 1e-10 * np.trace(A) / n * np.eye(n)
    if hard is None or len(hard[0]) == 0:
        return np.linalg.solve(A, b)
    H, h = hard
    K = np.block([[A, H.T], [H, np.zeros((len(H), len(H)))]])
    rhs2 = np.vstack([b, h])
    return np.linalg.lstsq(K, rhs2, rcond=None)[0][:n]


def fit_surface(points, poles=(16, 16), uperiodic=False, vperiodic=False, degree=3, fair=1e-5,
                u=None, v=None):
    """One C2 B-spline surface near a grid of points `points` (mu, mv, 3)
    (rows along u, columns along v; a periodic direction does not repeat
    its first row). `fair` (relative to the data) weighs the bending energy
    against the points: 0 follows them as closely as the poles allow, more
    smooths. The fit's deviation is in `.fit`."""
    _occ()
    X = np.asarray(points, float)
    mu, mv, _ = X.shape
    nu, nv = int(poles[0]), int(poles[1])
    u = np.linspace(0, 1, mu, endpoint=not uperiodic) if u is None else np.asarray(u, float)
    v = np.linspace(0, 1, mv, endpoint=not vperiodic) if v is None else np.asarray(v, float)
    Bu, Bv = basis(nu, degree, uperiodic, u), basis(nv, degree, vperiodic, v)
    R = np.kron(Bu, Bv)
    F, q = _fair_matrices(nu, nv, degree, degree, uperiodic, vperiodic)
    lam = fair * (mu * mv) / (q[0] * q[1])
    P = _solve(R, X.reshape(-1, 3), F, lam, nu * nv).reshape(nu, nv, 3)
    s = Surface(P, degree, degree, uperiodic, vperiodic)
    d = np.linalg.norm(np.einsum("ai,ijk,bj->abk", Bu, P, Bv) - X, axis=-1)
    s.fit = {"maxDeviation": float(d.max()), "meanDeviation": float(d.mean()), "poles": [nu, nv]}
    return s


def fair_patch(x_range, z_range, pins, poles=(24, 24), fair=1e-3, soft=None, degree=3, slopes=None, pin_weight=1e4):
    """A height field y(x, z) over a rectangle, as one B-spline surface: the
    fairest surface (least bending) that passes through `pins`, (k, 3)
    points (x, y, z) held exactly (or, when there are more of them than a
    quarter of the poles, by `pin_weight`), and near `soft`, (k, 4) points (x, y, z,
    weight). `slopes`, (k, 5) (x, z, dy/dx, dy/dz, weight), holds the
    surface's slope at points (a flat plateau is slope 0 at its edge: the
    sweep leaves it tangent). This is a loft that nobody has to draw: from
    a body's face up to a flat plateau, with no ridge or dip between them
    that the pins do not ask for.

    Returns a Surface whose u runs along x, v along z, and whose points are
    (x, y(x, z), z); its `height(x, z)` evaluates y directly (vectorized)."""
    _occ()
    nu, nv = poles
    x0, x1 = map(float, x_range)
    z0, z1 = map(float, z_range)
    U = lambda x: (np.asarray(x, float) - x0) / (x1 - x0)   # noqa: E731
    V = lambda z: (np.asarray(z, float) - z0) / (z1 - z0)   # noqa: E731
    F, q = _fair_matrices(nu, nv, degree, degree, False, False)
    rows, rhs = [], []
    hard_H, hard_h = [], []
    if pins is not None and len(pins):
        pins = np.asarray(pins, float)
        Hb = _rowbasis(nu, nv, degree, U(pins[:, 0]), V(pins[:, 2]))
        if len(pins) <= nu * nv // 4:
            hard_H, hard_h = Hb, pins[:, 1:2]          # few: held exactly
        else:                                          # many (a plateau's area): more rows than poles, so held by weight
            rows.append(Hb * np.sqrt(pin_weight))
            rhs.append(pins[:, 1:2] * np.sqrt(pin_weight))
    if soft is not None and len(soft):
        soft = np.asarray(soft, float)
        Sb = _rowbasis(nu, nv, degree, U(soft[:, 0]), V(soft[:, 2]))
        sw = np.sqrt(soft[:, 3:4])
        rows.append(Sb * sw)
        rhs.append(soft[:, 1:2] * sw)
    if slopes is not None and len(slopes):
        sl = np.asarray(slopes, float)
        uu, vv = U(sl[:, 0]), V(sl[:, 1])
        sw = np.sqrt(sl[:, 4:5])
        # d/dx = d/du / (x1 - x0)
        Du = _rowbasis(nu, nv, degree, uu, vv, du=1) / (x1 - x0)
        Dv = _rowbasis(nu, nv, degree, uu, vv, dv=1) / (z1 - z0)
        rows += [Du * sw, Dv * sw]
        rhs += [sl[:, 2:3] * sw, sl[:, 3:4] * sw]
    n = nu * nv
    R = np.vstack(rows) if rows else np.zeros((0, n))
    b = np.vstack(rhs) if rhs else np.zeros((0, 1))
    # scale the fairing in physical units: the energy over the rectangle
    sx, sz = (x1 - x0), (z1 - z0)
    Fs = [F[0] / sx ** 2, F[1] / (sx * sz), F[2] / sz ** 2]
    area_w = sx * sz / (q[0] * q[1])
    H = (np.asarray(hard_H), np.asarray(hard_h)) if len(hard_H) else None
    Y = _solve(R, b, Fs, fair * area_w, n, H).reshape(nu, nv)
    # poles: the x and z of a uniform clamped net are not exactly the
    # Greville abscissae, so place them there so x(u) and z(v) are linear
    gu = _greville(nu, degree)
    gv = _greville(nv, degree)
    PX = x0 + gu * (x1 - x0)
    PZ = z0 + gv * (z1 - z0)
    poles3 = np.stack(np.broadcast_arrays(PX[:, None], Y, PZ[None, :]), -1)
    s = Surface(poles3, degree, degree)
    s.x_range, s.z_range = (x0, x1), (z0, z1)
    s.height_poles = Y
    s.height = lambda x, z, _s=s: _height(_s, x, z)
    s.height_grad = lambda x, z, _s=s: _height_grad(_s, x, z)
    if pins is not None and len(pins):
        s.fit = {"pinError": float(np.abs(s.height(pins[:, 0], pins[:, 2]) - pins[:, 1]).max())}
    return s


def _greville(n, degree):
    k, m = _knots(n, degree, False)
    flat = np.repeat(k, m)
    return np.array([flat[i + 1:i + 1 + degree].mean() for i in range(n)])


def _rowbasis(nu, nv, degree, u, v, du=0, dv=0):
    Bu = basis(nu, degree, False, np.clip(u, 0, 1), du) if du else _basis_fast(nu, degree, np.clip(u, 0, 1))
    Bv = basis(nv, degree, False, np.clip(v, 0, 1), dv) if dv else _basis_fast(nv, degree, np.clip(v, 0, 1))
    return (Bu[:, :, None] * Bv[:, None, :]).reshape(len(u), -1)


def _basis_fast(n, degree, t, deriv=0):
    """Cox-de Boor for a clamped uniform knot vector, vectorized: the same
    basis `basis` reads from OpenCascade (checked in the tests), fast enough
    for millions of points."""
    k, m = _knots(n, degree, False)
    T = np.repeat(k, m)
    t = np.clip(np.asarray(t, float), 0.0, 1.0)
    if deriv:
        # derivative of degree-p basis from degree p-1 basis
        Nm = _basis_fast_deg(T, n + 1, degree - 1, t)       # n+1 functions of degree p-1 on the same knots
        out = np.zeros((len(t), n))
        for i in range(n):
            a = T[i + degree] - T[i]
            b = T[i + degree + 1] - T[i + 1]
            if a > 0:
                out[:, i] += degree / a * Nm[:, i]
            if b > 0:
                out[:, i] -= degree / b * Nm[:, i + 1]
        if deriv == 1:
            return out
        raise NotImplementedError
    return _basis_fast_deg(T, n, degree, t)


def _basis_fast_deg(T, n, p, t):
    """n basis functions of degree p on the flat knot vector T (len n + p + 1,
    or longer: extra trailing functions are ignored)."""
    nT = len(T)
    N = np.zeros((len(t), nT - 1))
    for i in range(nT - 1):
        if T[i + 1] > T[i]:
            N[:, i] = (t >= T[i]) & (t < T[i + 1])
    # t == 1 belongs to the last non-empty span
    last = max(i for i in range(nT - 1) if T[i + 1] > T[i])
    N[t >= 1.0, :] = 0.0
    N[t >= 1.0, last] = 1.0
    for d in range(1, p + 1):
        M = np.zeros((len(t), nT - 1 - d))
        for i in range(nT - 1 - d):
            a = T[i + d] - T[i]
            b = T[i + d + 1] - T[i + 1]
            if a > 0:
                M[:, i] += (t - T[i]) / a * N[:, i]
            if b > 0:
                M[:, i] += (T[i + d + 1] - t) / b * N[:, i + 1]
        N = M
    return N[:, :n]


def _height(s, x, z):
    x0, x1 = s.x_range
    z0, z1 = s.z_range
    nu, nv = s.height_poles.shape
    u = np.clip((np.asarray(x, float) - x0) / (x1 - x0), 0, 1)
    v = np.clip((np.asarray(z, float) - z0) / (z1 - z0), 0, 1)
    Bu, Bv = _basis_fast(nu, s.udeg, u), _basis_fast(nv, s.vdeg, v)
    return np.einsum("ai,ij,aj->a", Bu, s.height_poles, Bv)


def _height_grad(s, x, z):
    x0, x1 = s.x_range
    z0, z1 = s.z_range
    nu, nv = s.height_poles.shape
    u = np.clip((np.asarray(x, float) - x0) / (x1 - x0), 0, 1)
    v = np.clip((np.asarray(z, float) - z0) / (z1 - z0), 0, 1)
    Bu, Bv = _basis_fast(nu, s.udeg, u), _basis_fast(nv, s.vdeg, v)
    Du, Dv = _basis_fast(nu, s.udeg, u, 1), _basis_fast(nv, s.vdeg, v, 1)
    gx = np.einsum("ai,ij,aj->a", Du, s.height_poles, Bv) / (x1 - x0)
    gz = np.einsum("ai,ij,aj->a", Bu, s.height_poles, Dv) / (z1 - z0)
    return gx, gz


def height_sdf(s, side=+1):
    """A signed distance (near enough: first order, exact on a plane) to a
    height-field surface from `fair_patch`, negative on the -y side (side
    +1: the solid is behind the surface, y below it) or the +y side (-1)."""
    def f(P):
        P = np.asarray(P, float)
        h = s.height(P[:, 0], P[:, 2])
        gx, gz = s.height_grad(P[:, 0], P[:, 2])
        return side * (P[:, 1] - h) / np.sqrt(1.0 + gx * gx + gz * gz)
    return f


# ---- networks and lofts -----------------------------------------------------

def _cardinal(knots, t, periodic):
    """(len(t), len(knots)) weights: cubic spline interpolation through
    values at `knots` (0..1), evaluated at t, as a linear map (each column
    is the spline through a unit value at one knot). Periodic splines treat
    1 as 0; open ones are not-a-knot."""
    from scipy.interpolate import CubicSpline
    knots = np.asarray(knots, float)
    n = len(knots)
    if n == 1:
        return np.ones((len(t), 1))
    if n == 2 and not periodic:
        a = (np.asarray(t) - knots[0]) / (knots[1] - knots[0])
        return np.column_stack([1 - a, a])
    W = np.zeros((len(t), n))
    for j in range(n):
        y = np.zeros(n)
        y[j] = 1.0
        if periodic:
            cs = CubicSpline(np.r_[knots, knots[0] + 1.0], np.r_[y, y[0]], bc_type="periodic")
            W[:, j] = cs(np.mod(np.asarray(t) - knots[0], 1.0) + knots[0])
        else:
            bc = "not-a-knot" if n > 3 else "natural"
            W[:, j] = CubicSpline(knots, y, bc_type=bc)(t)
    return W


def network(u_curves, v_curves, u_at, v_at, uperiodic=False, vperiodic=False, samples=(80, 80),
            poles=(24, 24), fair=1e-6):
    """A Gordon surface through a curve network, then one B-spline surface
    fitted to it.

    `u_curves`: curves running along u (callables t -> (k, 3) points, t in
    0..1, e.g. Curve), one at each v in `v_at`; `v_curves`: running along
    v, one at each u in `u_at`. The curves must meet: u_curves[i](u_at[j])
    == v_curves[j](v_at[i]) (`loft` makes them). The Gordon surface
    S = Lu + Lv - T (Gordon 1969) interpolates every curve exactly; the
    fitted B-spline lies within `.fit["maxDeviation"]` of it."""
    mu, mv = samples
    us = np.linspace(0, 1, mu, endpoint=not uperiodic)
    vs = np.linspace(0, 1, mv, endpoint=not vperiodic)
    Cu = np.stack([np.asarray(c(us)) for c in u_curves])          # (I, mu, 3)
    Cv = np.stack([np.asarray(c(vs)) for c in v_curves])          # (J, mv, 3)
    Q = np.stack([[np.asarray(u_curves[i](np.array([u_at[j]])))[0] for j in range(len(u_at))]
                  for i in range(len(v_at))])                      # (I, J, 3)
    Wv = _cardinal(v_at, vs, vperiodic)                            # (mv, I)
    Wu = _cardinal(u_at, us, uperiodic)                            # (mu, J)
    Lu = np.einsum("bi,iak->abk", Wv, Cu)
    Lv = np.einsum("aj,jbk->abk", Wu, Cv)
    T = np.einsum("aj,bi,ijk->abk", Wu, Wv, Q)
    G = Lu + Lv - T
    s = fit_surface(G, poles, uperiodic, vperiodic, fair=fair)
    mismatch = max(float(np.linalg.norm(np.asarray(v_curves[j](np.array([v_at[i]])))[0] - Q[i, j]))
                   for i in range(len(v_at)) for j in range(len(u_at)))
    s.fit["networkMismatch"] = mismatch
    s.gordon = G
    return s


class _Polyline:
    """A closed or open polyline as a curve: t in 0..1 by arc length."""

    def __init__(self, P, closed):
        self.P = np.asarray(P, float)
        self.closed = closed
        self.t = chord_params(self.P, closed)

    def __call__(self, t):
        t = np.asarray(t, float)
        Q = np.vstack([self.P, self.P[:1]]) if self.closed else self.P
        tt = np.r_[self.t, 1.0] if self.closed else self.t
        if self.closed:
            t = np.mod(t, 1.0)
        return np.column_stack([np.interp(t, tt, Q[:, k]) for k in range(3)])


def loft(sections, v_at=None, rails=None, closed=True, n=96, poles=(24, 16), fair=1e-6, start=None):
    """A surface through closed (or open) section curves, in order along v.

    `sections`: (k, 3) point loops (or callables); each is resampled by arc
    length starting nearest `start` (a point, or a callable of the section
    index), so the sections line up. `v_at`: where each sits along v
    (default: by the distance between section centroids). `rails`: curves
    (callables v -> point, or (m, 3) polylines running from the first
    section to the last) the surface must pass through: a silhouette, a
    part line. Each section is warped smoothly round its loop so it meets
    every rail (the rail's point on that section is the section point
    nearest it), and the network of sections and rails is surfaced as a
    Gordon surface (`network`). u runs round the sections (periodic when
    closed), v along the loft."""
    secs = []
    for i, s in enumerate(sections):
        P = np.asarray(s(np.linspace(0, 1, 4 * n, endpoint=not closed)) if callable(s) else s, float)
        st = start(i) if callable(start) else start
        secs.append(resample(P, n, closed, st))
    secs = np.stack(secs)                                           # (I, n, 3)
    I = len(secs)
    if v_at is None:
        c = secs.mean(axis=1)
        d = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(c, axis=0), axis=1))]
        v_at = d / d[-1]
    v_at = np.asarray(v_at, float)
    u = np.linspace(0, 1, n, endpoint=not closed)
    if not rails:
        sec_curves = [_Polyline(P, closed) for P in secs]
        vs = np.linspace(0, 1, 64)
        Wv = _cardinal(v_at, vs, False)
        G = np.einsum("bi,iak->abk", Wv, secs)
        sf = fit_surface(G, poles, closed, False, fair=fair, u=u, v=vs)
        sf.sections = secs
        return sf
    rail_fns = []
    for r in rails:
        if callable(r):
            rail_fns.append(r)
        else:
            rail_fns.append(_Polyline(np.asarray(r, float), False))
    # the rail's point on each section (the rail point nearest the section),
    # where round the section it falls, and where along the rail
    J = len(rail_fns)
    hit = np.zeros((I, J, 3))
    t_hit = np.zeros((I, J))
    u_at_each = np.zeros((I, J))
    tr = np.linspace(0, 1, 801)
    for j, rf in enumerate(rail_fns):
        R = rf(tr)
        for i in range(I):
            d = np.linalg.norm(R[:, None, :] - secs[i][None, :, :], axis=-1)
            a, b = np.unravel_index(np.argmin(d), d.shape)
            hit[i, j], t_hit[i, j], u_at_each[i, j] = R[a], tr[a], u[b]
    u_at = np.median(u_at_each, axis=0)
    order = np.argsort(u_at)
    u_at, rail_fns, hit, t_hit = u_at[order], [rail_fns[k] for k in order], hit[:, order], t_hit[:, order]
    # warp each section so it passes through every rail at u_at
    W = _cardinal(u_at, u, closed)                                  # (n, J)
    warped = []
    for i in range(I):
        sec = _Polyline(secs[i], closed)
        warped.append(secs[i] + W @ (hit[i] - sec(u_at)))
    warped = np.stack(warped)
    sec_curves = [_Polyline(P, closed) for P in warped]
    # each rail as a curve in v: the rail itself, reparameterized so it
    # reaches each section's hit at that section's v
    from scipy.interpolate import PchipInterpolator
    rail_curves = []
    for j in range(J):
        tv = PchipInterpolator(v_at, t_hit[:, j])
        rail_curves.append(lambda v, _f=rail_fns[j], _t=tv: _f(np.clip(_t(np.asarray(v, float)), 0, 1)))
    sf = network(sec_curves, rail_curves, u_at, v_at, uperiodic=closed, samples=(n, 80), poles=poles, fair=fair)
    sf.sections = warped
    sf.rail_u = u_at
    return sf


# ---- OpenCascade: faces, fills, solids, meshes, STEP --------------------------

def face(s, tol=1e-6):
    """The OpenCascade face of a Surface (or a Geom surface)."""
    from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeFace
    g = s.geom if isinstance(s, Surface) else s
    return BRepBuilderAPI_MakeFace(g, tol).Face()


_CONT = {"G0": "GeomAbs_C0", "G1": "GeomAbs_G1", 0: "GeomAbs_C0", 1: "GeomAbs_G1"}


def fill(boundary, continuity="G1", points=(), degree=3, n_on_curve=15, iterations=2, max_degree=8,
         max_segments=9, tol_3d=1e-4, tol_ang=0.01, tol_curv=0.1):
    """An N-sided patch over a closed boundary.

    `boundary`: a list of (edge, support face or None, continuity) or of
    edges (then `continuity` for all, held to the face each edge already
    bounds when one is given as (edge, face)). An edge with a support face
    is held to it G0 (position) or G1 (tangent plane); one without is a
    free boundary curve. (OpenCascade's filling cannot hold a boundary G2:
    surface a region that needs curvature continuity as one surface.) `points`: (k, 3) points the patch
    passes through. Returns (face, {"G0", "G1", "G2"}: the largest
    deviation from each constraint, mm / radians)."""
    _occ()
    from OCP import GeomAbs
    from OCP.BRepFill import BRepFill_Filling
    from OCP.gp import gp_Pnt
    f = BRepFill_Filling(degree, n_on_curve, iterations, False, tol_3d, 1e-5, tol_ang, tol_curv,
                         max_degree, max_segments)
    for item in boundary:
        if isinstance(item, tuple):
            e = item[0]
            sup = item[1] if len(item) > 1 else None
            c = item[2] if len(item) > 2 else continuity
        else:
            e, sup, c = item, None, continuity
        if c not in _CONT:
            raise ValueError(f"fill: continuity is G0 or G1 on a boundary, got {c!r} (for G2, surface the region as one "
                             "surface: fit_surface, network, fair_patch)")
        order = getattr(GeomAbs, _CONT[c])
        if sup is None:
            f.Add(e, GeomAbs.GeomAbs_C0, True)
        else:
            f.Add(e, sup, order, True)
    for p in points:
        f.Add(gp_Pnt(*map(float, p)))
    f.Build()
    if not f.IsDone():
        raise RuntimeError("fill: OpenCascade could not build the patch")
    err = {"G0": float(f.G0Error()), "G1": float(f.G1Error()), "G2": float(f.G2Error())}
    return f.Face(), err


def wire_of(edges):
    from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeWire
    w = BRepBuilderAPI_MakeWire()
    for e in edges:
        w.Add(e)
    return w.Wire()


def _edge_face_map():
    from OCP.OCP.collections import IndexedDataMap_TopoDS_Shape_List_TopoDS_Shape_TopTools_ShapeMapHasher as Map
    return Map()


def boundary_edges(shape):
    """Free edges of a face or shell (bounding one face only)."""
    from OCP.TopAbs import TopAbs_EDGE, TopAbs_FACE
    from OCP.TopExp import TopExp
    m = _edge_face_map()
    TopExp.MapShapesAndAncestors_s(shape, TopAbs_EDGE, TopAbs_FACE, m)
    from OCP.BRep import BRep_Tool
    from OCP.TopoDS import TopoDS
    out = []
    for i in range(1, m.Extent() + 1):
        faces = list(m.FindFromIndex(i))
        if len(faces) != 1:
            continue
        e = TopoDS.Edge(m.FindKey(i))
        if BRep_Tool.IsClosed_s(e, TopoDS.Face(faces[0])):     # a periodic face's seam
            continue
        out.append(e)
    return out


def sew_solid(faces, tol=1e-3):
    """Faces sewn into one closed solid (raises if they do not close)."""
    from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeSolid, BRepBuilderAPI_Sewing
    from OCP.ShapeFix import ShapeFix_Solid
    from OCP.TopAbs import TopAbs_SHELL
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopoDS import TopoDS
    sw = BRepBuilderAPI_Sewing(tol)
    for f in faces:
        sw.Add(f.face() if isinstance(f, Surface) else f)
    sw.Perform()
    sh = sw.SewedShape()
    ex = TopExp_Explorer(sh, TopAbs_SHELL)
    if not ex.More():
        raise RuntimeError("sew_solid: the faces did not sew into a shell")
    shell = TopoDS.Shell(ex.Current())
    solid = BRepBuilderAPI_MakeSolid(shell).Solid()
    fx = ShapeFix_Solid(solid)
    fx.Perform()
    return fx.Solid()


def mesh(shape, tolerance=0.02, angle=0.1):
    """A trimesh of any OpenCascade shape (or Surface), tessellated to
    `tolerance` mm."""
    import trimesh
    from OCP.BRep import BRep_Tool
    from OCP.BRepMesh import BRepMesh_IncrementalMesh
    from OCP.TopAbs import TopAbs_FACE, TopAbs_REVERSED
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopLoc import TopLoc_Location
    from OCP.TopoDS import TopoDS
    if isinstance(shape, Surface):
        shape = shape.face()
    BRepMesh_IncrementalMesh(shape, tolerance, False, angle, True)
    V, F, off = [], [], 0
    ex = TopExp_Explorer(shape, TopAbs_FACE)
    while ex.More():
        fc = TopoDS.Face(ex.Current())
        loc = TopLoc_Location()
        tri = BRep_Tool.Triangulation_s(fc, loc)
        if tri is not None:
            tr = loc.Transformation()
            pts = np.array([[p.X(), p.Y(), p.Z()] for p in
                            (tri.Node(i).Transformed(tr) for i in range(1, tri.NbNodes() + 1))])
            T = np.array([tri.Triangle(i).Get() for i in range(1, tri.NbTriangles() + 1)]) - 1
            if fc.Orientation() == TopAbs_REVERSED:
                T = T[:, ::-1]
            V.append(pts)
            F.append(T + off)
            off += len(pts)
        ex.Next()
    if not V:
        return trimesh.Trimesh()
    m = trimesh.Trimesh(np.vstack(V), np.vstack(F), process=True)
    m.merge_vertices(digits_vertex=5)
    return m


def write_step(shapes, path, names=None):
    """Write OpenCascade shapes (or Surfaces) to one STEP file."""
    from OCP.BRep import BRep_Builder
    from OCP.IFSelect import IFSelect_RetDone
    from OCP.STEPControl import STEPControl_AsIs, STEPControl_Writer
    from OCP.TopoDS import TopoDS_Compound
    if not isinstance(shapes, (list, tuple)):
        shapes = [shapes]
    comp = TopoDS_Compound()
    b = BRep_Builder()
    b.MakeCompound(comp)
    for s in shapes:
        b.Add(comp, s.face() if isinstance(s, Surface) else s)
    w = STEPControl_Writer()
    w.Transfer(comp, STEPControl_AsIs)
    if w.Write(str(path)) != IFSelect_RetDone:
        raise RuntimeError(f"write_step: could not write {path}")
    return str(path)


def sdf(shape, pitch=0.5, bounds=None, band=3.0):
    """A signed distance function (callable on (N, 3) points, mm, negative
    inside) of a closed solid or watertight mesh: exact within `band` mm of
    the skin (to the tessellation, 0.01 mm), sampled on a `pitch` grid and
    trilinear between nodes. For field pipelines that cut, shell and split
    a skin drawn here."""
    import trimesh
    from scipy.interpolate import RegularGridInterpolator
    m = shape if isinstance(shape, trimesh.Trimesh) else mesh(shape, tolerance=min(0.01, 0.05 * pitch))
    if bounds is not None:
        lo, hi = np.asarray(bounds[0], float), np.asarray(bounds[1], float)
    else:
        lo, hi = m.bounds
    F, org, p = _sdf_grid(m, lo - 4 * pitch, hi + 4 * pitch, pitch, band)
    axes = [org[i] + np.arange(F.shape[i]) * p for i in range(3)]
    gi = RegularGridInterpolator(axes, F, bounds_error=False, fill_value=None)
    far = float(np.abs(F).max())

    def f(P):
        P = np.asarray(P, float)
        Q = np.clip(P, [a[0] for a in axes], [a[-1] for a in axes])
        out = gi(Q)
        # outside the grid: distance to the grid box plus the boundary value
        d = np.linalg.norm(P - Q, axis=1)
        return np.where(d > 0, np.maximum(out, 0) + d, out)
    f.mesh, f.grid, f.far = m, (F, org, p), far
    return f


def _sdf_grid(m, lo, hi, p, band):
    """Signed distance on a grid: sign from the winding number, distance
    exact (point to triangle) within `band`, a Euclidean transform beyond."""
    import igl
    from scipy import ndimage
    n = np.ceil((hi - lo) / p).astype(int) + 1
    axes = [lo[i] + np.arange(n[i]) * p for i in range(3)]
    V = np.asarray(m.vertices, float)
    Fc = np.asarray(m.faces, np.int64)
    inside = np.zeros(tuple(n), bool)
    yz = np.stack(np.meshgrid(axes[1], axes[2], indexing="ij"), -1).reshape(-1, 2)
    step = max(1, int(3_000_000 // len(yz)))
    for i in range(0, n[0], step):
        xs = axes[0][i:i + step]
        P = np.column_stack([np.repeat(xs, len(yz)), np.tile(yz, (len(xs), 1))])
        w = igl.fast_winding_number(V, Fc, P)
        inside[i:i + len(xs)] = (np.abs(w) > 0.5).reshape(len(xs), n[1], n[2])
    F = (ndimage.distance_transform_edt(~inside) - ndimage.distance_transform_edt(inside)) * p
    near = np.abs(F) <= band + p
    idx = np.nonzero(near)
    P = lo + np.column_stack(idx) * p
    out = np.empty(len(P))
    for s in range(0, len(P), 2_000_000):
        d2 = igl.point_mesh_squared_distance(P[s:s + 2_000_000], V, Fc)[0]
        out[s:s + 2_000_000] = np.sqrt(np.maximum(d2, 0))
    F[idx] = np.where(inside[idx], -out, out)
    return F.astype(np.float32), lo, p
