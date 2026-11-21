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


def _flat_knots(n, degree, periodic):
    """The flat knot vector of n poles on 0..1: clamped, or (periodic) the
    uniform knots carried `degree` spans past both ends, whose n + degree
    functions wrap onto the n poles."""
    if periodic:
        return (np.arange(n + 2 * degree + 1) - degree) / n
    k, m = _knots(n, degree, False)
    return np.repeat(k, m)


def _coxdeboor(T, p, t, deriv=0):
    """Every degree-p B-spline on the flat knots T at t (len(t), len(T)-p-1),
    or its derivative of order `deriv` (Cox-de Boor and its derivative
    recurrence). The last non-empty span is closed on the right."""
    if deriv > p:
        return np.zeros((len(t), len(T) - p - 1))
    if deriv:
        N = _coxdeboor(T, p - 1, t, deriv - 1)
        a = T[p:-1] - T[:-p - 1]
        b = T[p + 1:] - T[1:-p]
        with np.errstate(divide="ignore", invalid="ignore"):
            ia = np.where(a > 0, p / a, 0.0)
            ib = np.where(b > 0, p / b, 0.0)
        return N[:, :-1] * ia - N[:, 1:] * ib
    nT = len(T)
    N = np.zeros((len(t), nT - 1))
    span = np.nonzero(T[1:] > T[:-1])[0]
    j = span[np.clip(np.searchsorted(T[span], t, side="right") - 1, 0, len(span) - 1)]
    N[np.arange(len(t)), j] = 1.0
    for d in range(1, p + 1):
        a = T[d:-1] - T[:-d - 1]
        b = T[d + 1:] - T[1:-d]
        with np.errstate(divide="ignore", invalid="ignore"):
            ia = np.where(a > 0, 1.0 / a, 0.0)
            ib = np.where(b > 0, 1.0 / b, 0.0)
        N = (t[:, None] - T[None, :-d - 1]) * ia * N[:, :-1] + (T[None, d + 1:] - t[:, None]) * ib * N[:, 1:]
    return N


def basis(n, degree, periodic, t, deriv=0):
    """(len(t), n) matrix: row a holds every pole's weight at parameter t[a]
    (deriv k: the weights of the k-th derivative). The same numbers
    OpenCascade evaluates for the curve it builds from the poles (checked in
    the tests against `_basis_occ`), computed here in one vectorized pass."""
    t = np.asarray(t, float)
    key = (n, degree, bool(periodic), deriv, t.tobytes())
    if key in _BASIS_CACHE:
        return _BASIS_CACHE[key]
    T = _flat_knots(n, degree, periodic)
    if periodic:
        N = _coxdeboor(T, degree, np.mod(t, 1.0), deriv)
        B = np.zeros((len(t), n))
        for i in range(n + degree):
            B[:, i % n] += N[:, i]
    else:
        B = _coxdeboor(T, degree, np.clip(t, 0.0, 1.0), deriv)[:, :n]
    if len(_BASIS_CACHE) > 256:
        _BASIS_CACHE.clear()
    _BASIS_CACHE[key] = B
    return B


def _basis_occ(n, degree, periodic, t, deriv=0):
    """`basis` read from OpenCascade itself, one unit pole at a time (slow;
    the tests hold `basis` to it)."""
    _occ()
    from OCP.gp import gp_Pnt, gp_Vec
    t = np.asarray(t, float)
    B = np.zeros((len(t), n))
    for j in range(n):
        poles = np.zeros((n, 3))
        poles[j, 0] = 1.0
        c = _curve(poles, degree, periodic)
        p, v1, v2, v3 = gp_Pnt(), gp_Vec(), gp_Vec(), gp_Vec()
        for a, ta in enumerate(t):
            if deriv == 0:
                B[a, j] = c.Value(float(ta)).X()
            elif deriv == 1:
                c.D1(float(ta), p, v1)
                B[a, j] = v1.X()
            elif deriv == 2:
                c.D2(float(ta), p, v1, v2)
                B[a, j] = v2.X()
            else:
                c.D3(float(ta), p, v1, v2, v3)
                B[a, j] = v3.X()
    return B


def _gram(n, degree, periodic, a, b):
    """The (n, n) matrix of integrals over 0..1 of (a-th derivative of pole
    i's weight) x (b-th derivative of pole j's), exact (Gauss points on
    every knot span). Sparse, banded."""
    from scipy import sparse
    key = ("gram", n, degree, bool(periodic), a, b)
    if key in _BASIS_CACHE:
        return _BASIS_CACHE[key]
    T = _flat_knots(n, degree, periodic)
    br = np.unique(np.clip(T, 0.0, 1.0))
    x, w = np.polynomial.legendre.leggauss(degree + 1)
    h = np.diff(br)
    t = (br[:-1, None] + 0.5 * h[:, None] * (x[None, :] + 1.0)).ravel()
    wt = (0.5 * h[:, None] * w[None, :]).ravel()
    Ba = sparse.csr_matrix(basis(n, degree, periodic, t, a))
    Bb = Ba if a == b else sparse.csr_matrix(basis(n, degree, periodic, t, b))
    G = (Ba.T @ sparse.diags(wt) @ Bb).tocsr()
    _BASIS_CACHE[key] = G
    return G


_ENERGY = {"bending": (1.0, 0.0), "jerk": (0.0, 1.0)}


def _energy(nu, nv, udeg, vdeg, up, vp, energy="bending", su=1.0, sv=1.0):
    """The surface map's fairness energy as a sparse quadratic form in the
    poles (nu*nv square), integrated over the patch in real lengths (the u
    side `su` mm long, the v side `sv`):

    - "bending": the thin plate, Suu^2 + 2 Suv^2 + Svv^2. Its minimum is the
      shape a thin elastic sheet takes through the constraints. Curvature
      itself is free to jump in slope where a constraint ends.
    - "jerk": the variation of curvature, Suuu^2 + 3 Suuv^2 + 3 Suvv^2 +
      Svvv^2. Its minimum changes curvature as slowly as the constraints
      allow, so a sweep leaves a flat plateau with curvature growing from
      zero and reflections flow through without a bend.
    - (wb, wj): both, weighted (jerk in mm^2 against bending).
    """
    from scipy import sparse
    wb, wj = _ENERGY[energy] if isinstance(energy, str) else map(float, energy)
    g = lambda d, a, b: _gram(*((nu, udeg, up) if d == 0 else (nv, vdeg, vp)), a, b)   # noqa: E731
    E = sparse.csr_matrix((nu * nv, nu * nv))
    area = su * sv
    if wb:
        for (a, b, c) in ((2, 0, 1.0), (1, 1, 2.0), (0, 2, 1.0)):
            E = E + (wb * c * area / (su ** (2 * a) * sv ** (2 * b))) * sparse.kron(g(0, a, a), g(1, b, b))
    if wj:
        for (a, b, c) in ((3, 0, 1.0), (2, 1, 3.0), (1, 2, 3.0), (0, 3, 1.0)):
            E = E + (wj * c * area / (su ** (2 * a) * sv ** (2 * b))) * sparse.kron(g(0, a, a), g(1, b, b))
    return E.tocsr()


def _solve_sparse(R, rhs, E, hard=None, eps=1e-12):
    """min |R x - rhs|^2 + x'Ex, with hard rows (H x = h) kept by Lagrange
    multipliers (a hair of give, so rows that say the same thing twice do
    not make the system singular; `eps` as an array is each row's own give,
    1 / its weight: a weighted least-squares row without squaring the
    system's condition). Sparse throughout."""
    from scipy import sparse
    from scipy.sparse.linalg import splu
    n = E.shape[0]
    A = (R.T @ R + E).tocsc() if R is not None and R.shape[0] else E.tocsc()
    b = R.T @ rhs if R is not None and R.shape[0] else np.zeros((n, rhs.shape[1] if rhs is not None else 1))
    tr = A.diagonal().mean()
    A = A + (1e-10 * tr) * sparse.identity(n, format="csc")
    if hard is None or hard[0].shape[0] == 0:
        return splu(A).solve(np.asarray(b))
    H, h = sparse.csr_matrix(hard[0]), np.asarray(hard[1], float)
    m = H.shape[0]
    hs = abs(H).max() or 1.0
    give = (eps * hs * hs / tr) * np.ones(m) if np.isscalar(eps) else np.asarray(eps, float)
    K = sparse.bmat([[A, H.T], [H, -sparse.diags(give)]], format="csc")
    return splu(K).solve(np.vstack([np.asarray(b), h]))[:n]


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


def fit_surface(points, poles=(16, 16), uperiodic=False, vperiodic=False, degree=3, fair=1e-5,
                u=None, v=None, energy="bending"):
    """One C2 B-spline surface near a grid of points `points` (mu, mv, 3)
    (rows along u, columns along v; a periodic direction does not repeat
    its first row). `fair` (relative to the data) weighs the fairness
    energy (`energy`: "bending", "jerk" or weights of both, see `_energy`)
    against the points: 0 follows them as closely as the poles allow, more
    smooths. The fit's deviation is in `.fit`."""
    from scipy import sparse
    X = np.asarray(points, float)
    mu, mv, _ = X.shape
    nu, nv = int(poles[0]), int(poles[1])
    u = np.linspace(0, 1, mu, endpoint=not uperiodic) if u is None else np.asarray(u, float)
    v = np.linspace(0, 1, mv, endpoint=not vperiodic) if v is None else np.asarray(v, float)
    Bu, Bv = basis(nu, degree, uperiodic, u), basis(nv, degree, vperiodic, v)
    R = sparse.kron(sparse.csr_matrix(Bu), sparse.csr_matrix(Bv), format="csr")
    E = _energy(nu, nv, degree, degree, uperiodic, vperiodic, energy)
    P = _solve_sparse(R, X.reshape(-1, 3), (fair * mu * mv) * E).reshape(nu, nv, 3)
    s = Surface(P, degree, degree, uperiodic, vperiodic)
    d = np.linalg.norm(np.einsum("ai,ijk,bj->abk", Bu, P, Bv) - X, axis=-1)
    s.fit = {"maxDeviation": float(d.max()), "meanDeviation": float(d.mean()), "poles": [nu, nv]}
    return s


def fair_patch(x_range, z_range, pins, poles=(24, 24), fair=1e-3, soft=None, degree=3, slopes=None, pin_weight=1e4,
               energy="bending"):
    """A height field y(x, z) over a rectangle, as one B-spline surface: the
    fairest surface that passes through `pins`, (k, 3)
    points (x, y, z) held exactly (or, when there are more of them than a
    quarter of the poles, by `pin_weight`), and near `soft`, (k, 4) points (x, y, z,
    weight). `slopes`, (k, 5) (x, z, dy/dx, dy/dz, weight), holds the
    surface's slope at points (a flat plateau is slope 0 at its edge: the
    sweep leaves it tangent). This is a loft that nobody has to draw: from
    a body's face up to a flat plateau, with no ridge or dip between them
    that the pins do not ask for.

    `energy` says what fairest means (`_energy`): "bending" (least bending:
    a thin elastic sheet), "jerk" (least change of curvature: the sweep
    leaves a plateau with its curvature growing from zero, so reflections
    run through the plateau's edge without a bend; use degree 5 for it), or
    (wb, wj) for both.

    Returns a Surface whose u runs along x, v along z, and whose points are
    (x, y(x, z), z); its `height(x, z)` evaluates y directly (vectorized)."""
    from scipy import sparse
    nu, nv = poles
    x0, x1 = map(float, x_range)
    z0, z1 = map(float, z_range)
    U = lambda x: (np.asarray(x, float) - x0) / (x1 - x0)   # noqa: E731
    V = lambda z: (np.asarray(z, float) - z0) / (z1 - z0)   # noqa: E731
    rows, rhs = [], []
    hard = None
    if pins is not None and len(pins):
        pins = np.asarray(pins, float)
        Hb = _rowbasis(nu, nv, degree, U(pins[:, 0]), V(pins[:, 2]))
        if len(pins) <= nu * nv // 4:
            hard = (Hb, pins[:, 1:2])                  # few: held exactly
        else:                                          # many (a plateau's area): more rows than poles, so held by weight
            rows.append(Hb * np.sqrt(pin_weight))
            rhs.append(pins[:, 1:2] * np.sqrt(pin_weight))
    if soft is not None and len(soft):
        soft = np.asarray(soft, float)
        Sb = _rowbasis(nu, nv, degree, U(soft[:, 0]), V(soft[:, 2]))
        sw = np.sqrt(soft[:, 3:4])
        rows.append(sparse.diags(sw[:, 0]) @ Sb)
        rhs.append(soft[:, 1:2] * sw)
    if slopes is not None and len(slopes):
        sl = np.asarray(slopes, float)
        uu, vv = U(sl[:, 0]), V(sl[:, 1])
        sw = sparse.diags(np.sqrt(sl[:, 4]))
        # d/dx = d/du / (x1 - x0)
        Du = _rowbasis(nu, nv, degree, uu, vv, du=1) / (x1 - x0)
        Dv = _rowbasis(nu, nv, degree, uu, vv, dv=1) / (z1 - z0)
        rows += [sw @ Du, sw @ Dv]
        rhs += [sw @ sl[:, 2:3], sw @ sl[:, 3:4]]
    R = sparse.vstack(rows).tocsr() if rows else None
    b = np.vstack(rhs) if rhs else np.zeros((0, 1))
    E = _energy(nu, nv, degree, degree, False, False, energy, x1 - x0, z1 - z0)
    Y = _solve_sparse(R, b, fair * E, hard).reshape(nu, nv)
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
    flat = _flat_knots(n, degree, False)
    return np.array([flat[i + 1:i + 1 + degree].mean() for i in range(n)])


def _rowbasis(nu, nv, degree, u, v, du=0, dv=0, up=False, vp=False, vdegree=None):
    """Sparse (len(u), nu*nv): row a holds every pole's weight at (u[a],
    v[a]) (du, dv: of that partial derivative)."""
    from scipy import sparse
    u, v = np.asarray(u, float), np.asarray(v, float)
    Bu = sparse.csr_matrix(basis(nu, degree, up, u if up else np.clip(u, 0, 1), du))
    Bv = sparse.csr_matrix(basis(nv, vdegree or degree, vp, v if vp else np.clip(v, 0, 1), dv))
    return _rowkron(Bu, Bv)


def _rowkron(A, B):
    """The row-wise Kronecker product of two sparse matrices with the same
    number of rows."""
    from scipy import sparse
    A, B = A.tocsr(), B.tocsr()
    A.sort_indices()
    B.sort_indices()
    m, na, nb = A.shape[0], A.shape[1], B.shape[1]
    ca, cb = np.diff(A.indptr), np.diff(B.indptr)
    cnt = ca * cb
    row = np.repeat(np.arange(m), cnt)
    start = np.r_[0, np.cumsum(cnt)]
    k = np.arange(start[-1]) - np.repeat(start[:-1], cnt)
    cbr = np.maximum(np.repeat(cb, cnt), 1)
    ia = np.repeat(A.indptr[:-1], cnt) + k // cbr
    ib = np.repeat(B.indptr[:-1], cnt) + k % cbr
    return sparse.csr_matrix((A.data[ia] * B.data[ib], (row, A.indices[ia] * nb + B.indices[ib])), shape=(m, na * nb))


def _basis_fast(n, degree, t, deriv=0):
    """The clamped basis (kept by name for callers): `basis`."""
    return basis(n, degree, False, np.asarray(t, float), deriv)


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
    is held to it G0 (position), G1 (tangent plane) or G2 (curvature); one
    without is a free boundary curve. G0 and G1 go to OpenCascade's
    BRepFill_Filling; that cannot hold a bounded edge G2, so a boundary
    with any G2 edge is built by `cap` (the variational patch, which also
    takes G0 and G1 edges and is the fairer of the two). `points`: (k, 3)
    points the patch passes through (G0/G1 only). Returns (face, {"G0",
    "G1", "G2"}: the largest deviation from each constraint, mm / radians /
    1/mm)."""
    _occ()
    conts = [(it[2] if isinstance(it, tuple) and len(it) > 2 else continuity) for it in boundary]
    if any(c in ("G2", 2) for c in conts):
        if len(points):
            raise ValueError("fill: a G2 fill (cap) takes no interior points")
        face_, err = cap([(it if isinstance(it, tuple) else (it, None)) for it in boundary], continuity)
        return face_, {k: err[k] for k in ("G0", "G1", "G2")} | {"G1": float(np.radians(err["G1"]))}
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
            raise ValueError(f"fill: continuity is G0, G1 or G2, got {c!r}")
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


def sew(faces, tol=1e-3):
    """Faces (or Surfaces) sewn along their shared edges into one shape (a
    shell, open or closed), for `surfacequality.edge_continuity`, meshing
    or STEP. `sew_solid` when they close."""
    from OCP.BRepBuilderAPI import BRepBuilderAPI_Sewing
    sw = BRepBuilderAPI_Sewing(tol)
    for f in faces:
        sw.Add(f.face() if isinstance(f, Surface) else f)
    sw.Perform()
    return sw.SewedShape()


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


# ---- curvature-continuous blends, caps, cuts and edits -----------------------
#
# What follows joins surfaces the way a surfacing artist does: a blend or a
# cap that carries the neighbours' curvature across the edge (G2), so a
# reflection runs over the join without a bend, and edits that move a
# surface and leave it as fair as before.

def _edge_frames(face, edge, s):
    """`_edge_frame1` along one edge, or along several that run head to
    tail (a list): walked as one curve by arc length."""
    if not isinstance(edge, (list, tuple)):
        return _edge_frame1(face, edge, s)
    if len(edge) == 1:
        return _edge_frame1(face, edge[0], s)
    probes = [_edge_frame1(face, e, np.array([0.0, 1.0])) for e in edge]
    scale = max(sum(q["length"] for q in probes), 1e-9)
    order = _chain([(q["polyline"][0], q["polyline"][-1]) for q in probes], 1e-3 * scale)
    lens = np.array([probes[i]["length"] for i, _ in order])
    cum = np.r_[0.0, np.cumsum(lens)] / lens.sum()
    s = np.asarray(s, float)
    k = np.clip(np.searchsorted(cum, s, side="right") - 1, 0, len(order) - 1)
    out = None
    for j, (i, rev) in enumerate(order):
        sel = np.nonzero(k == j)[0]
        if not len(sel):
            continue
        loc = np.clip((s[sel] - cum[j]) / max(cum[j + 1] - cum[j], 1e-30), 0, 1)
        fr = _edge_frame1(face, edge[i], 1.0 - loc if rev else loc)
        if out is None:
            out = {key: np.zeros((len(s),) + v.shape[1:]) for key, v in fr.items() if isinstance(v, np.ndarray) and key != "polyline"}
        for key in out:
            out[key][sel] = fr[key]
    line = np.vstack([probes[i]["polyline"][::-1] if rev else probes[i]["polyline"] for i, rev in order])
    out["polyline"] = line
    out["length"] = float(lens.sum())
    out["closed"] = bool(np.linalg.norm(line[0] - line[-1]) < 1e-6 * max(out["length"], 1.0))
    return out


def _edge_frame1(face, edge, s):
    """Along an edge of a face, at arc-length fractions `s` (0..1): the
    points P, the face's unit normal N, the unit direction d in the tangent
    plane square to the edge and pointing out of the face, the second
    derivative c2 of the face's own curve leaving in that direction (so a
    curve that starts with d and c2 continues the face G2), and what the
    face's second fundamental form needs (J: 3D tangent -> uv step, II: its
    2x2 matrix about N)."""
    from OCP.BRep import BRep_Tool
    from OCP.BRepAdaptor import BRepAdaptor_Curve
    from OCP.BRepClass import BRepClass_FaceClassifier
    from OCP.gp import gp_Pnt, gp_Pnt2d, gp_Vec
    from OCP.TopAbs import TopAbs_IN
    ce = BRepAdaptor_Curve(edge)
    t0, t1 = ce.FirstParameter(), ce.LastParameter()
    tt = np.linspace(t0, t1, 4001)
    pts = np.array([[q.X(), q.Y(), q.Z()] for q in (ce.Value(float(a)) for a in tt)])
    arc = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))]
    length = float(arc[-1])
    t = np.interp(np.asarray(s, float), arc / max(length, 1e-30), tt)
    n = len(t)
    closed = bool(np.linalg.norm(pts[0] - pts[-1]) < 1e-7 * max(length, 1.0))
    if face is None:                       # a free curve: position and tangent only
        Pf, Tf = np.zeros((n, 3)), np.zeros((n, 3))
        p, tv = gp_Pnt(), gp_Vec()
        for i, ti in enumerate(t):
            ce.D1(float(ti), p, tv)
            Pf[i] = (p.X(), p.Y(), p.Z())
            Tf[i] = np.array([tv.X(), tv.Y(), tv.Z()]) / max(tv.Magnitude(), 1e-30)
        zz = np.zeros((n, 3))
        return {"P": Pf, "T": Tf, "N": zz, "d": zz, "c2": zz, "J": np.zeros((n, 2, 3)), "II": np.zeros((n, 2, 2)),
                "uv": np.zeros((n, 2)), "length": length, "closed": closed, "polyline": pts}
    surf = BRep_Tool.Surface_s(face)
    c2d = BRep_Tool.CurveOnSurface_s(edge, face, 0.0, 0.0)
    if c2d is None:
        raise ValueError("the edge does not lie on the face (no curve in the face's parameters)")
    P, T, N, D, C2 = (np.zeros((n, 3)) for _ in range(5))
    J, II, UV = np.zeros((n, 2, 3)), np.zeros((n, 2, 2)), np.zeros((n, 2))
    p, tv = gp_Pnt(), gp_Vec()
    q, su, sv, suu, svv, suv = gp_Pnt(), gp_Vec(), gp_Vec(), gp_Vec(), gp_Vec(), gp_Vec()
    v3 = lambda g: np.array([g.X(), g.Y(), g.Z()])   # noqa: E731
    for i, ti in enumerate(t):
        ce.D1(float(ti), p, tv)
        uv = c2d.Value(float(ti))
        surf.D2(uv.X(), uv.Y(), q, su, sv, suu, svv, suv)
        Su, Sv, Suu, Svv, Suv = v3(su), v3(sv), v3(suu), v3(svv), v3(suv)
        nn = np.cross(Su, Sv)
        nn /= max(np.linalg.norm(nn), 1e-30)
        tg = v3(tv)
        tg /= max(np.linalg.norm(tg), 1e-30)
        d = np.cross(nn, tg)
        d /= max(np.linalg.norm(d), 1e-30)
        Ji = np.linalg.pinv(np.column_stack([Su, Sv]))
        P[i], T[i], N[i], D[i], J[i], UV[i] = v3(q), tg, nn, d, Ji, (uv.X(), uv.Y())
        II[i] = [[nn @ Suu, nn @ Suv], [nn @ Suv, nn @ Svv]]
    # which way is out of the face: step in uv from the middle sample
    k = n // 2
    w = J[k] @ D[k]
    ub = [surf.Bounds()[i] for i in range(4)]
    span = max(min(abs(ub[1] - ub[0]), abs(ub[3] - ub[2])), 1e-9)
    span = span if np.isfinite(span) and span < 1e50 else 1.0
    eps = 1e-3 * span / max(np.linalg.norm(w), 1e-30)
    inside = BRepClass_FaceClassifier(face, gp_Pnt2d(*(UV[k] + eps * w)), 1e-9).State() == TopAbs_IN
    outside = BRepClass_FaceClassifier(face, gp_Pnt2d(*(UV[k] - eps * w)), 1e-9).State() == TopAbs_IN
    if inside and not outside:
        D = -D
    W = np.einsum("nij,nj->ni", J, D)
    surf_c2 = np.zeros((n, 3))
    for i, ti in enumerate(t):
        surf.D2(UV[i, 0], UV[i, 1], q, su, sv, suu, svv, suv)
        surf_c2[i] = W[i, 0] ** 2 * v3(suu) + 2 * W[i, 0] * W[i, 1] * v3(suv) + W[i, 1] ** 2 * v3(svv)
    return {"P": P, "T": T, "N": N, "d": D, "c2": surf_c2, "J": J, "II": II, "uv": UV, "length": length,
            "closed": closed, "polyline": pts}


def _polyline_distance(Q, P):
    """Distance from each point of Q to the polyline P."""
    A, B = P[:-1], P[1:]
    AB = B - A
    L2 = np.maximum((AB * AB).sum(1), 1e-30)
    out = np.empty(len(Q))
    for i in range(0, len(Q), 256):
        q = Q[i:i + 256]
        tt = np.clip(np.einsum("qsk,sk->qs", q[:, None, :] - A[None], AB) / L2, 0, 1)
        d = np.linalg.norm(q[:, None, :] - (A[None] + tt[..., None] * AB[None]), axis=-1)
        out[i:i + 256] = d.min(1)
    return out


_BEZ3 = np.array([[1 / 5, 1 / 10, 1 / 30], [1 / 10, 2 / 15, 1 / 10], [1 / 30, 1 / 10, 1 / 5]])   # quadratic Bernstein Gram


def _section_poles(fa, fb, la, lb, ca, cb):
    """The six poles of every quintic section from side a to side b, for
    lead lengths la, lb (mm): position, direction and (for G2) the face's
    own second derivative at each end."""
    ka = 1.0 if ca == "G2" else 0.0
    kb = 1.0 if cb == "G2" else 0.0
    la, lb = la[..., None], lb[..., None]
    Q0 = fa["P"] + 0 * la
    Q1 = Q0 + la * fa["d"] / 5
    Q2 = Q0 + 2 * la * fa["d"] / 5 + ka * la ** 2 * fa["c2"] / 20
    Q5 = fb["P"] + 0 * lb
    Q4 = Q5 + lb * fb["d"] / 5
    Q3 = Q5 + 2 * lb * fb["d"] / 5 + kb * lb ** 2 * fb["c2"] / 20
    return np.stack([Q0, Q1, Q2, Q3, Q4, Q5], axis=-2)


def _smooth1d(x, closed, passes):
    x = np.array(x, float)
    for _ in range(passes):
        if closed:
            x = 0.25 * np.roll(x, 1) + 0.5 * x + 0.25 * np.roll(x, -1)
        else:
            y = x.copy()
            y[1:-1] = 0.25 * x[:-2] + 0.5 * x[1:-1] + 0.25 * x[2:]
            x = y
    return x


def blend(a, b, continuity="G2", lead=1.0, n=24, tol=5e-4, max_n=400):
    """One surface across the gap between two edges, continuing both faces.

    `a`, `b`: (face, edge), the edge on that face's boundary (or cut into
    it, `cut`). `continuity`: "G2" (default: position, tangent plane and
    curvature carried over from each face, so a reflection crosses both
    joins without a bend), "G1" or "G0", or a pair, one for each side.

    Every section across the blend is a quintic curve that leaves face a
    as that face's own curve would (same direction, same curvature) and
    arrives on face b the same way. How far each face's shape reaches into
    the blend (the lead) is not drawn: it is the length that makes the
    section change its curvature least (the least integral of its third
    derivative squared), found for every section and smoothed along the
    edge. `lead` scales it (a number or a pair): above 1 the faces carry
    further and the middle tightens, below 1 the blend turns sooner and
    sits flatter against the gap.

    The edges are walked by arc length, b turned (and, for closed loops,
    rotated) to line up with a. Returns a Surface (u along the edges, cubic;
    v across, quintic) with `.fit` = {"G0": the largest gap to either edge,
    mm, "lead": mean leads, "sections": how many}; sew it between the
    faces. Sections are added until the gap is under `tol`."""
    _occ()
    ca, cb = (continuity, continuity) if isinstance(continuity, str) else continuity
    for c in (ca, cb):
        if c not in ("G0", "G1", "G2"):
            raise ValueError(f"blend: continuity is G0, G1 or G2, got {c!r}")
    la_s, lb_s = (float(lead), float(lead)) if np.isscalar(lead) else map(float, lead)
    probe_a = _edge_frames(a[0], a[1], np.linspace(0, 1, 65))
    probe_b = _edge_frames(b[0], b[1], np.linspace(0, 1, 65))
    closed = probe_a["closed"] and probe_b["closed"]
    # how b's walk maps onto a's: direction, and for loops the start
    Pa, Pb = probe_a["P"][:-1], probe_b["P"][:-1]
    best = (np.inf, 1, 0.0)
    for sgn in (1, -1):
        Q = Pb if sgn == 1 else probe_b["P"][::-1][:-1]
        for sh in (range(64) if closed else [0]):
            c = float(np.linalg.norm(Pa - np.roll(Q, -sh, axis=0), axis=1).sum())
            if not closed:
                c = float(np.linalg.norm(probe_a["P"] - (probe_b["P"] if sgn == 1 else probe_b["P"][::-1]), axis=1).sum())
            if c < best[0]:
                best = (c, sgn, sh / 64.0)
    _, sgn, shift = best
    leads = None
    while True:
        s = np.linspace(0, 1, n, endpoint=False) if closed else _greville(n, 3)
        sb = np.mod(s + shift, 1.0) if sgn == 1 else np.mod(1.0 - s - shift, 1.0)
        if not closed:
            sb = s if sgn == 1 else 1.0 - s
        fa, fb = _edge_frames(a[0], a[1], s), _edge_frames(b[0], b[1], sb)
        if ca == "G0" or cb == "G0":
            gap = fb["P"] - fa["P"]
            gdir = gap / np.maximum(np.linalg.norm(gap, axis=1, keepdims=True), 1e-30)
            if ca == "G0":
                fa = {**fa, "d": gdir}
            if cb == "G0":
                fb = {**fb, "d": -gdir}
        L = np.linalg.norm(fb["P"] - fa["P"], axis=1)
        # the lead that changes curvature least, per section
        g = np.linspace(0.25, 1.75, 13)
        ka, kb = np.meshgrid(g, g, indexing="ij")
        la = ka[None] * L[:, None, None]
        lb = kb[None] * L[:, None, None]
        Q = _section_poles({k: v[:, None, None, :] for k, v in fa.items() if k in ("P", "d", "c2")},
                           {k: v[:, None, None, :] for k, v in fb.items() if k in ("P", "d", "c2")}, la, lb, ca, cb)
        D3 = Q[..., 3:, :] - 3 * Q[..., 2:5, :] + 3 * Q[..., 1:4, :] - Q[..., 0:3, :]
        en = np.einsum("...ik,ij,...jk->...", D3, _BEZ3, D3).reshape(n, -1)
        j = en.argmin(1)
        if leads is None:
            # found once, on the first (coarse) walk, and smoothed along the
            # edge; finer walks read it off, so the lead does not ripple
            leads = (s.copy(), _smooth1d(ka.ravel()[j] * L, closed, 8), _smooth1d(kb.ravel()[j] * L, closed, 8))
        per = 1.0 if closed else None
        la = np.interp(s, leads[0], leads[1], period=per) * la_s
        lb = np.interp(s, leads[0], leads[2], period=per) * lb_s
        Q = _section_poles(fa, fb, la, lb, ca, cb)                    # (n, 6, 3)
        B = basis(n, 3, closed, s)
        poles = np.linalg.solve(B, Q.reshape(n, -1)).reshape(n, 6, 3)
        sf = Surface(poles, 3, 5, uperiodic=closed)
        um = np.linspace(0, 1, 4 * n + 1)[:-1] + 0.5 / (4 * n)
        ga = _polyline_distance(sf(um, np.array([0.0]))[:, 0], probe_a["polyline"]).max()
        gb = _polyline_distance(sf(um, np.array([1.0]))[:, 0], probe_b["polyline"]).max()
        if max(ga, gb) <= tol or n >= max_n:
            break
        n = min(max_n, int(np.ceil(n * 1.6)))
    sf.fit = {"G0": float(max(ga, gb)), "lead": [float(la.mean()), float(lb.mean())], "sections": int(n),
              "continuity": [ca, cb]}
    return sf


def _chain(ends, tol):
    """Order open edges (by their end points) head to tail round a loop:
    [(index, reversed)]."""
    left = list(range(len(ends)))
    order = [(left.pop(0), False)]
    tip = ends[order[0][0]][1]
    while left:
        dist = [(min(np.linalg.norm(ends[i][0] - tip), np.linalg.norm(ends[i][1] - tip)), i) for i in left]
        dmin, i = min(dist)
        if dmin > tol:
            raise ValueError(f"the boundary edges do not close into one loop (a gap of {dmin:.3g} mm)")
        rev = np.linalg.norm(ends[i][1] - tip) < np.linalg.norm(ends[i][0] - tip)
        order.append((i, rev))
        tip = ends[i][0] if rev else ends[i][1]
        left.remove(i)
    return order


def _curve2d(points, closed):
    """A cubic B-spline through 2D points (a curve in a surface's
    parameters), periodic when closed."""
    from OCP.Geom2d import Geom2d_BSplineCurve
    from OCP.gp import gp_Pnt2d
    from OCP.OCP.collections import Array1_gp_Pnt2d
    P = np.asarray(points, float)
    n = len(P)
    t = np.linspace(0, 1, n, endpoint=False) if closed else _greville(n, 3)
    C = np.linalg.solve(basis(n, 3, closed, t), P)
    arr = Array1_gp_Pnt2d(1, n)
    for i, c in enumerate(C):
        arr.SetValue(i + 1, gp_Pnt2d(float(c[0]), float(c[1])))
    k, m = _knots(n, 3, closed)
    return Geom2d_BSplineCurve(arr, _arr1(k), _arr1(m, "int"), 3, closed), t


def _edge_on(surface, points_uv, closed):
    from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeEdge
    from OCP.BRepLib import BRepLib
    c2, _ = _curve2d(points_uv, closed)
    e = BRepBuilderAPI_MakeEdge(c2, surface).Edge()
    BRepLib.BuildCurves3d_s(e)
    return e


def _area(shape):
    from OCP.BRepGProp import BRepGProp
    from OCP.GProp import GProp_GProps
    g = GProp_GProps()
    BRepGProp.SurfaceProperties_s(shape, g)
    return float(g.Mass())


def cap(boundary, continuity="G2", poles=(20, 20), energy=None, tol=2e-4, max_poles=56, margin=0.2, degree=3,
        weights=(1e4, 1e3, 1e2),
        samples=None, lead=1.0, steep=2.0):
    """An N-sided patch over a closed boundary that carries its neighbours'
    curvature across: one B-spline surface, trimmed to the boundary.

    `boundary`: [(edge, face)] or [(edge, face, "G0" | "G1" | "G2")], the
    edges closing one loop, each on the face it bounds. The patch is the
    fairest surface (`energy`, see `_energy`) that meets every edge in
    position, shares the neighbour's tangent plane there (G1) and its
    curvature square to the edge (G2). With position and tangent plane held
    all along an edge, the curvature along the edge is already shared, so
    the one direction across it completes curvature continuity.

    The patch is laid over the boundary's best plane (it must project onto
    that plane as a simple loop) and is free in all three coordinates, so a
    tube's end closes as a dome that leaves the tube's wall tangent. Poles
    are added until every edge is met within `tol` mm.

    Returns (face, {"G0" mm, "G1" degrees, "G2" 1/mm: the largest departure
    from each, measured between the constraint points, "poles"})."""
    _occ()
    from OCP.BRepAdaptor import BRepAdaptor_Curve
    from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeFace, BRepBuilderAPI_MakeWire
    from scipy import sparse
    items = []
    for it in boundary:
        if not isinstance(it, tuple):
            it = (it, None)
        e, f = it[0], it[1] if len(it) > 1 else None
        c = (it[2] if len(it) > 2 else continuity) if f is not None else "G0"
        if c not in ("G0", "G1", "G2"):
            raise ValueError(f"cap: continuity is G0, G1 or G2, got {c!r}")
        items.append((e, f, c))
    if energy is None:
        energy = "jerk" if any(c == "G2" for _, _, c in items) else "bending"
    lengths = []
    for e, f, c in items:
        ce = BRepAdaptor_Curve(e)
        tt = np.linspace(ce.FirstParameter(), ce.LastParameter(), 200)
        pts = np.array([[q.X(), q.Y(), q.Z()] for q in (ce.Value(float(a)) for a in tt)])
        lengths.append(float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum()))
    total = sum(lengths)

    def frames(per_len):
        out = []
        for (e, f, c), ln in zip(items, lengths):
            m = max(9, int(np.ceil(per_len * ln / total)))
            fr = _edge_frames(f, e, np.linspace(0, 1, m))
            if fr["closed"]:
                fr = {k: (v[:-1] if isinstance(v, np.ndarray) and len(v) == m else v) for k, v in fr.items()}
            fr["cont"] = c
            out.append(fr)
        return out

    probe = frames(120)
    allP = np.vstack([fr["P"] for fr in probe])
    c0 = allP.mean(0)
    _, _, Vt = np.linalg.svd(allP - c0, full_matrices=False)
    e1, e2 = Vt[0], Vt[1]
    plane = lambda P: np.column_stack([(P - c0) @ e1, (P - c0) @ e2])     # noqa: E731
    uvp = plane(allP)
    lo, hi = uvp.min(0), uvp.max(0)
    pad = margin * (hi - lo).max()
    lo, hi = lo - pad, hi + pad
    su, sv = hi - lo
    Lc = float(np.sqrt(su * sv))
    nu, nv = map(int, poles)
    while True:
        m_total = int(samples or 5 * (nu + nv))
        frs = frames(m_total)
        P = np.vstack([fr["P"] for fr in frs])
        Nn = np.vstack([fr["N"] for fr in frs])
        m = len(P)
        uv = (plane(P) - lo) / (su, sv)
        # a direction across the boundary in the plane: the edge's tangent, turned
        evs = []
        ctr = (plane(P).mean(0) - lo) / (su, sv)
        off = 0
        for fr in frs:
            tp = np.column_stack([fr["T"] @ e1, fr["T"] @ e2])
            tp /= np.maximum(np.linalg.norm(tp, axis=1, keepdims=True), 1e-9)
            en_ = np.column_stack([-tp[:, 1], tp[:, 0]])
            # pointing into the loop (the way the neighbour's face continues)
            inward = (ctr - uv[off:off + len(en_)]) * (su, sv)
            en_ *= np.sign((en_ * inward).sum(1) + 1e-12)[:, None]
            evs.append(en_)
            off += len(en_)
        ev = np.vstack(evs)
        g1 = np.concatenate([np.full(len(fr["P"]), fr["cont"] != "G0") for fr in frs])
        g2 = np.concatenate([np.full(len(fr["P"]), fr["cont"] == "G2") for fr in frs])
        rb = lambda du=0, dv=0: _rowbasis(nu, nv, degree, uv[:, 0], uv[:, 1], du, dv) / (su ** du * sv ** dv)   # noqa: E731
        B0, Bu, Bv, Buu, Buv, Bvv = rb(), rb(1, 0), rb(0, 1), rb(2, 0), rb(1, 1), rb(0, 2)
        N = nu * nv
        E1 = _energy(nu, nv, degree, degree, False, False, energy, su, sv)
        E = sparse.block_diag([E1, E1, E1], format="csr")
        dot = lambda M, rowsel: sparse.hstack([sparse.diags(Nn[rowsel, k]) @ M[rowsel] for k in range(3)])   # noqa: E731
        # weights: each kind of row scaled to the energy's own size, then
        # position far above tangency above curvature
        es = float(E1.diagonal().mean())
        nrm = lambda M: max(float(M.multiply(M).sum()) / N, 1e-300)     # noqa: E731
        w0 = weights[0] * es / nrm(B0)
        w1 = weights[1] * es / nrm(Bu + Bv)
        w2 = weights[2] * es / nrm(Buu + Bvv)
        # rows, each with a right side c0 + lam c1 + lam^2 c2 in the lead lam
        z = lambda k: np.zeros(k)                                         # noqa: E731
        rows = [sparse.block_diag([B0, B0, B0])]
        rhs = [(P.T.reshape(-1), z(3 * m), z(3 * m))]
        give = [np.full(3 * m, 1.0 / w0)]
        i1 = np.nonzero(g1)[0]
        i2 = np.nonzero(g2)[0]
        Be = sparse.diags(ev[:, 0]) @ Bu + sparse.diags(ev[:, 1]) @ Bv
        Bee = sparse.diags(ev[:, 0] ** 2) @ Buu + sparse.diags(2 * ev[:, 0] * ev[:, 1]) @ Buv + sparse.diags(ev[:, 1] ** 2) @ Bvv
        if len(i1):
            # the patch leaves the edge as the neighbour arrives: across the
            # boundary its first derivative is the neighbour's own direction
            # out of its face, lam long
            # (the direction across the boundary in the plane, lifted into
            # the neighbour's tangent plane: what the patch's derivative is
            # when it continues the neighbour as laid out over the plane.
            # Where the neighbour stands steep to the plane, a tube's wall,
            # the lift runs away: there it is held to `steep` long, along
            # the neighbour's own way out.)
            pn = np.cross(e1, e2)
            e3 = np.outer(ev[:, 0], e1) + np.outer(ev[:, 1], e2)
            npn = Nn @ pn
            safe = np.where(np.abs(npn) > 1e-9, npn, 1e-9)
            lift = e3 - ((Nn * e3).sum(1) / safe)[:, None] * pn[None]
            dall = np.vstack([fr["d"] for fr in frs])
            lift *= np.sign((lift * dall).sum(1) + 1e-300)[:, None]
            ln = np.linalg.norm(lift, axis=1)
            dn = np.where((ln > steep)[:, None], steep * dall, lift)
            rows += [dot(Bu, i1), dot(Bv, i1), sparse.block_diag([Be[i1]] * 3)]
            rhs += [(z(len(i1)), z(len(i1)), z(len(i1))), (z(len(i1)), z(len(i1)), z(len(i1))),
                    (z(3 * len(i1)), dn[i1].T.reshape(-1), z(3 * len(i1)))]
            give += [np.full(2 * len(i1), 1.0 / w1), np.full(3 * len(i1), 1.0 / w1)]
        if len(i2):
            wn = np.einsum("nij,nj->ni", np.vstack([fr["J"] for fr in frs]), dn)
            kn = np.einsum("ni,nij,nj->n", wn, np.vstack([fr["II"] for fr in frs]), wn)
            rows.append(dot(Bee, i2))
            rhs.append((z(len(i2)), z(len(i2)), kn[i2]))
            give.append(np.full(len(i2), 1.0 / w2))
        C = sparse.vstack(rows).tocsr()
        c = np.column_stack([np.concatenate([r[k] for r in rhs]) for k in range(3)])
        X3 = _solve_sparse(None, np.zeros((0, 3)), E, (C, c), np.concatenate(give))      # X = X3 @ (1, lam, lam^2)
        lam = float(lead)
        X = X3 @ np.array([1.0, lam, lam * lam])
        sf = Surface(np.stack([X[k * N:(k + 1) * N].reshape(nu, nv) for k in range(3)], -1), degree, degree)
        err = _cap_errors(sf, items, lengths, plane, lo, (su, sv))
        if err["G0"] <= tol or max(nu, nv) >= max_poles:
            break
        nu, nv = min(max_poles, int(np.ceil(nu * 1.4))), min(max_poles, int(np.ceil(nv * 1.4)))
    # the trimmed face: every boundary edge as a curve in the patch's parameters
    ends, pcs = [], []
    for (e, f, c), ln in zip(items, lengths):
        k = max(12, int(np.ceil(160 * ln / total)))
        fr = _edge_frames(f, e, np.linspace(0, 1, k, endpoint=False) if _is_closed_edge(e) else _greville(k, 3))
        pcs.append(((plane(fr["P"]) - lo) / (su, sv), fr["closed"] and _is_closed_edge(e)))
        ends.append((fr["polyline"][0], fr["polyline"][-1]))
    mw = BRepBuilderAPI_MakeWire()
    if len(items) == 1 and pcs[0][1]:
        loop = pcs[0][0]
        if _signed_area(loop) < 0:
            loop = loop[::-1]
        mw.Add(_edge_on(sf.geom, loop, True))
    else:
        order = _chain(ends, 1e-3 * Lc)
        loop = np.vstack([pcs[i][0][::-1] if rev else pcs[i][0] for i, rev in order])
        flip = _signed_area(loop) < 0
        for i, rev in (order[::-1] if flip else order):
            uvs = pcs[i][0][::-1] if (rev != flip) else pcs[i][0]
            # the ends exactly at the corner the next edge starts from
            a_end, b_end = ends[i][1] if (rev != flip) else ends[i][0], ends[i][0] if (rev != flip) else ends[i][1]
            full = np.vstack([(plane(a_end[None]) - lo) / (su, sv), uvs, (plane(b_end[None]) - lo) / (su, sv)])
            mw.Add(_edge_on(sf.geom, _resample2d(full, len(uvs)), False))
    face_ = BRepBuilderAPI_MakeFace(sf.geom, mw.Wire(), True).Face()
    err["poles"] = [nu, nv]
    err["surface"] = sf
    return face_, err


def _is_closed_edge(e):
    from OCP.BRepAdaptor import BRepAdaptor_Curve
    ce = BRepAdaptor_Curve(e)
    a, b = ce.Value(ce.FirstParameter()), ce.Value(ce.LastParameter())
    return a.Distance(b) < 1e-7


def _signed_area(loop):
    x, y = loop[:, 0], loop[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def _resample2d(P, n):
    """n points along a 2D polyline at the Greville parameters of an n-pole
    cubic (so `_curve2d` interpolates them), ends kept."""
    t = chord_params(P)
    g = _greville(n, 3)
    return np.column_stack([np.interp(g, t, P[:, k]) for k in range(2)])


def _cap_errors(sf, items, lengths, plane, lo, size):
    """How well a cap meets its boundary, measured between its constraint
    points: gap (mm), angle between tangent planes (degrees), jump in the
    curvature square to the edge (1/mm)."""
    g0 = g1 = g2 = 0.0
    nu, nv, _ = sf.poles.shape
    for (e, f, c) in items:
        fr = _edge_frames(f, e, (np.arange(97) + 0.5) / 97)
        uv = (plane(fr["P"]) - lo) / size
        D = {k: (_rowbasis(nu, nv, sf.udeg, uv[:, 0], uv[:, 1], *k) @ sf.poles.reshape(-1, 3)) / (size[0] ** k[0] * size[1] ** k[1])
             for k in ((0, 0), (1, 0), (0, 1), (2, 0), (1, 1), (0, 2))}
        g0 = max(g0, float(np.linalg.norm(D[0, 0] - fr["P"], axis=1).max()))
        if c == "G0":
            continue
        n = np.cross(D[1, 0], D[0, 1])
        n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-30)
        g1 = max(g1, float(np.degrees(np.arccos(np.clip(np.abs((n * fr["N"]).sum(1)), -1, 1))).max()))
        if c != "G2":
            continue
        # the cap's curvature in the neighbour's direction d, against the neighbour's
        Jc = np.stack([np.linalg.pinv(np.column_stack([a, b])) for a, b in zip(D[1, 0], D[0, 1])])
        w = np.einsum("nij,nj->ni", Jc, fr["d"])
        sgn = np.sign((n * fr["N"]).sum(1))
        kc = (w[:, 0] ** 2 * (n * D[2, 0]).sum(1) + 2 * w[:, 0] * w[:, 1] * (n * D[1, 1]).sum(1)
              + w[:, 1] ** 2 * (n * D[0, 2]).sum(1)) * sgn
        wn = np.einsum("nij,nj->ni", fr["J"], fr["d"])
        kn = np.einsum("ni,nij,nj->n", wn, fr["II"], wn)
        g2 = max(g2, float(np.abs(kc - kn).max()))
    return {"G0": g0, "G1": g1, "G2": g2}


def cut(face_, loop, keep="outside", n=None):
    """An opening cut into a face along a closed curve lying on (or near)
    it: `loop`, (k, 3) points round the opening, is projected onto the
    face's surface and interpolated there, so the new edge lies on the
    surface exactly. keep="outside" returns the face with the hole,
    "inside" the piece cut out, "both" the pair. The hole's edge is the
    new face's last edge (`boundary_edges(...)[-1]`, or `hole_edge`), ready
    for `blend` or `cap`."""
    _occ()
    from OCP.BRep import BRep_Tool
    from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeFace, BRepBuilderAPI_MakeWire
    from OCP.GeomAPI import GeomAPI_ProjectPointOnSurf
    from OCP.gp import gp_Pnt
    from OCP.TopoDS import TopoDS
    f = face_.face() if isinstance(face_, Surface) else face_
    surf = BRep_Tool.Surface_s(f)
    L = np.asarray(loop, float)
    if np.linalg.norm(L[0] - L[-1]) < 1e-9:
        L = L[:-1]
    L = resample(L, int(n or max(24, min(len(L), 96))), closed=True)
    uv = np.zeros((len(L), 2))
    pr = GeomAPI_ProjectPointOnSurf()
    for i, p in enumerate(L):
        pr.Init(gp_Pnt(*map(float, p)), surf)
        if pr.NbPoints() == 0:
            raise ValueError("cut: a loop point does not project onto the face")
        uv[i] = pr.LowerDistanceParameters()
    if _signed_area(uv) < 0:
        uv = uv[::-1]
    e = _edge_on(surf, uv, True)
    inner = BRepBuilderAPI_MakeFace(surf, BRepBuilderAPI_MakeWire(e).Wire(), True).Face()
    mf = BRepBuilderAPI_MakeFace(f)
    mf.Add(TopoDS.Wire(BRepBuilderAPI_MakeWire(e).Wire().Reversed()))
    outer = mf.Face()
    if keep == "inside":
        return inner
    if keep == "both":
        return outer, inner
    return outer


def hole_edges(face_):
    """The edges of a face's inner wires (its openings), one list per hole."""
    from OCP.BRepTools import BRepTools, BRepTools_WireExplorer
    from OCP.TopAbs import TopAbs_WIRE
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopoDS import TopoDS
    outer = BRepTools.OuterWire_s(face_)
    out = []
    ex = TopExp_Explorer(face_, TopAbs_WIRE)
    while ex.More():
        w = TopoDS.Wire(ex.Current())
        if not w.IsSame(outer):
            we, es = BRepTools_WireExplorer(w, face_), []
            while we.More():
                es.append(we.Current())
                we.Next()
            out.append(es)
        ex.Next()
    return out


def edit(s, moves, hold=(), freeze=2, energy="bending"):
    """Move a surface by handles and leave it as fair as it was.

    `moves`: [(u, v, (dx, dy, dz))], points of the surface (by parameter)
    and where each should go. `hold`: [(u, v)], points that must stay. The
    change is the fairest one that does it: the displacement field with the
    least `energy` (`_energy`) through the handles, so a pulled point takes
    a smooth swell of surface with it and no ripple, and what was fair
    stays fair. `freeze`: how many pole rows along each open side stay
    exactly as they are (2 keeps the surface's position and tangent there,
    3 its curvature too, so joins made with `blend`/`cap` keep their
    continuity; 0 lets the sides move). Returns a new Surface; `.fit`
    reports the largest handle miss."""
    from scipy import sparse
    nu, nv, _ = s.poles.shape
    N = nu * nv
    E = _energy(nu, nv, s.udeg, s.vdeg, s.uperiodic, s.vperiodic, energy)
    free = np.ones((nu, nv), bool)
    if freeze:
        if not s.uperiodic:
            free[:freeze], free[-freeze:] = False, False
        if not s.vperiodic:
            free[:, :freeze], free[:, -freeze:] = False, False
    idx = np.nonzero(free.ravel())[0]
    pts = [(m[0], m[1], np.asarray(m[2], float)) for m in moves] + [(h[0], h[1], np.zeros(3)) for h in hold]
    u = np.array([p[0] for p in pts], float)
    v = np.array([p[1] for p in pts], float)
    d = np.array([p[2] for p in pts], float)
    Bu = sparse.csr_matrix(basis(nu, s.udeg, s.uperiodic, np.mod(u, 1.0) if s.uperiodic else u))
    Bv = sparse.csr_matrix(basis(nv, s.vdeg, s.vperiodic, np.mod(v, 1.0) if s.vperiodic else v))
    H = _rowkron(Bu, Bv)[:, idx]
    Ef = E[idx][:, idx]
    dP = np.zeros((N, 3))
    dP[idx] = _solve_sparse(None, np.zeros((0, 3)), Ef, (H, d))
    out = Surface(s.poles + dP.reshape(nu, nv, 3), s.udeg, s.vdeg, s.uperiodic, s.vperiodic)
    out.fit = {"handleError": float(np.linalg.norm(H @ dP[idx] - d, axis=1).max()) if len(pts) else 0.0}
    for k in ("x_range", "z_range"):
        if hasattr(s, k):
            setattr(out, k, getattr(s, k))
    return out


def _signed_distance(surf, P):
    """Signed distance from points to a surface: + on the side its normal
    (Su x Sv) points."""
    from OCP.GeomAPI import GeomAPI_ProjectPointOnSurf
    from OCP.gp import gp_Pnt, gp_Vec
    out = np.zeros(len(P))
    pr = GeomAPI_ProjectPointOnSurf()
    q, su, sv = gp_Pnt(), gp_Vec(), gp_Vec()
    for i, p in enumerate(np.asarray(P, float)):
        pr.Init(gp_Pnt(*map(float, p)), surf)
        if pr.NbPoints() == 0:
            out[i] = np.nan
            continue
        u, v = pr.LowerDistanceParameters()
        surf.D1(u, v, q, su, sv)
        n = su.Crossed(sv)
        d = np.array([p[0] - q.X(), p[1] - q.Y(), p[2] - q.Z()])
        out[i] = np.sign(d @ np.array([n.X(), n.Y(), n.Z()]) + 1e-300) * pr.LowerDistance()
    return out


def _split(face_, tool):
    from OCP.BRepAlgoAPI import BRepAlgoAPI_Splitter
    from OCP.OCP.collections import List_TopoDS_Shape
    from OCP.TopAbs import TopAbs_FACE
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopoDS import TopoDS
    a, b = List_TopoDS_Shape(), List_TopoDS_Shape()
    a.Append(face_)
    b.Append(tool)
    sp = BRepAlgoAPI_Splitter()
    sp.SetArguments(a)
    sp.SetTools(b)
    sp.Build()
    out, ex = [], TopExp_Explorer(sp.Shape(), TopAbs_FACE)
    while ex.More():
        out.append(TopoDS.Face(ex.Current()))
        ex.Next()
    return out


def _edges_of(face_):
    from OCP.TopAbs import TopAbs_EDGE
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopoDS import TopoDS
    out, ex = [], TopExp_Explorer(face_, TopAbs_EDGE)
    while ex.More():
        e = TopoDS.Edge(ex.Current())
        if not any(e.IsSame(o) for o in out):
            out.append(e)
        ex.Next()
    return out


def _sample_points(face_, k=40):
    m = mesh(face_, 0.2, 0.3)
    c = np.asarray(m.triangles_center)
    return c[np.linspace(0, len(c) - 1, min(k, len(c))).astype(int)]


def setback(face_a, face_b, radius, keep=None):
    """Face a cut back from where it crosses face b: the part of a further
    than `radius` from b's surface, on one side of b. Returns (the kept
    face, its new edges, which lie `radius` from b). `keep`: a point near
    the part of a to keep (default: its larger side of b). The cut is a's
    intersection with b's surface offset by the radius, so the new edge
    lies on a exactly."""
    _occ()
    from OCP.BRep import BRep_Tool
    from OCP.BRepAdaptor import BRepAdaptor_Curve
    from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeFace
    from OCP.Geom import Geom_OffsetSurface
    fa = face_a.face() if isinstance(face_a, Surface) else face_a
    fb = face_b.face() if isinstance(face_b, Surface) else face_b
    sb = BRep_Tool.Surface_s(fb)
    if keep is not None:
        side = float(np.sign(_signed_distance(sb, np.asarray(keep, float)[None])[0]))
    else:
        parts = _split(fa, fb)
        big = max(parts, key=_area)
        side = float(np.sign(np.nanmedian(_signed_distance(sb, _sample_points(big)))))
    tool = BRepBuilderAPI_MakeFace(Geom_OffsetSurface(sb, side * radius), *sb.Bounds(), 1e-6).Face()
    parts = _split(fa, tool)
    if len(parts) < 2:
        raise ValueError(f"setback: the faces do not cross, or not further than the radius {radius} apart")

    def score(q):
        d = side * _signed_distance(sb, _sample_points(q))
        if keep is not None:
            return -float(np.linalg.norm(_sample_points(q, 200) - np.asarray(keep, float), axis=1).min()) if np.nanmedian(d) > radius else -np.inf
        return _area(q) if np.nanmedian(d) > radius else -np.inf
    kept = max(parts, key=score)
    new = []
    for e in _edges_of(kept):
        ce = BRepAdaptor_Curve(e)
        tt = np.linspace(ce.FirstParameter(), ce.LastParameter(), 7)[1:-1]
        pts = np.array([[q.X(), q.Y(), q.Z()] for q in (ce.Value(float(a)) for a in tt)])
        if np.nanmax(np.abs(side * _signed_distance(sb, pts) - radius)) < 1e-3 * max(radius, 1.0):
            new.append(e)
    if not new:
        raise ValueError("setback: found no edge at the radius on the kept face")
    return kept, new


def fillet(face_a, face_b, radius, continuity="G2", lead=1.0, keep=(None, None), tol=5e-4):
    """Two crossing faces joined by a curvature-continuous blend in place
    of their sharp intersection: each face is cut back to `radius` (a
    number, or one for each face) from the other's surface and a `blend`
    is laid across. Unlike a rolling-ball fillet (an arc: tangent, with a
    jump in curvature where it starts), the blend leaves each face with
    that face's own curvature.

    `keep`: a point near the part of each face to keep (default: each
    face's larger side of the other). Returns (face a cut back, the blend
    Surface, face b cut back); sew them."""
    ra, rb = (float(radius), float(radius)) if np.isscalar(radius) else map(float, radius)
    ka, ea = setback(face_a, face_b, ra, keep[0])
    kb, eb = setback(face_b, face_a, rb, keep[1])
    return ka, blend((ka, ea), (kb, eb), continuity, lead, tol=tol), kb
