"""
Organic surfaces as a few smooth, editable patches.

A topology-optimized body, a scan or an AI-generated mesh (Tripo, Meshy) is
a few hundred thousand triangles and no surface a CAD user can pick,
fillet or offset. step.py's smooth route (brep.py) covers it with a
thousand or two Bezier patches that meet with a corner between them; this
module covers it the way a surfacing artist would: a coarse quad cage
whose Catmull-Clark limit surface is fitted to the mesh, written as
bicubic B-spline patches.

1. cage: Instant Meshes (Jakob et al. 2015, BSD; `pyinstantmeshes`) lays a
   field-aligned all-quad mesh over the surface, its edges following the
   principal curvature directions;
2. fit: the cage's control points move so its limit surface lies on the
   mesh: least squares between limit points (exact, through the
   subdivision matrices) and their closest points on the mesh, both ways
   (surface to mesh, mesh to surface), with a small membrane term on the
   cage so it stays fair where the mesh says nothing;
3. density from the deviation: the cage is laid again finer until the
   two-way deviation is within the tolerance (the noise the input already
   carries, or the process's own scatter: a printed part cannot show a
   smaller difference, and finer patches would only fit the noise);
4. patches: every cage face is one bicubic Bezier patch. Where the face's
   corners are all regular (four edges each) the limit surface *is* a
   bicubic, so the patch is exact and its neighbours meet it with C2
   continuity; around an extraordinary vertex the patch is fitted to the
   exact limit samples with the shared edge curves fixed, so neighbours
   still share their edges exactly (G1 to within a small, reported angle);
5. sew into one closed solid (OpenCascade), check it, write STEP.

The surface is a subdivision surface: the cage (`Cage.V`, `Cage.Q`) is the
editable design, written alongside the STEP as an OBJ an agent or a SubD
modeller can move and re-export.
"""
from __future__ import annotations

import time

import numpy as np

__all__ = ["Cage", "quad_cage", "fit_cage", "surface_mesh", "patches", "to_brep", "export_surface"]

LEVEL = 3            # subdivision levels the fit and the patches sample (8 x 8 cells per face)
SMOOTH = 1e-3        # membrane weight on the cage, relative to the data rows
ITERATIONS = 12      # closest-point / least-squares rounds per fit
SEW = 1e-4           # mm: sewing tolerance; shared edges are identical, so this only absorbs rounding


# ------------------------------------------------------------------ cage

class Cage:
    """A closed, consistently oriented all-quad mesh: V (n, 3), Q (m, 4)."""

    def __init__(self, V, Q):
        self.V = np.asarray(V, float)
        self.Q = np.asarray(Q, np.int64)
        Q = self.Q
        m = len(Q)
        a = Q.ravel()
        b = np.roll(Q, -1, axis=1).ravel()
        nv = max(len(self.V), int(Q.max()) + 1 if Q.size else 0)
        # an edge as one integer (lo n + hi): the same order as unique rows, a fraction of the time
        key = np.minimum(a, b) * nv + np.maximum(a, b)
        u, inv, cnt = np.unique(key, return_inverse=True, return_counts=True)
        E = np.stack([u // nv, u % nv], 1)
        if (cnt != 2).any():
            raise ValueError(f"cage is not closed and manifold: {int((cnt != 2).sum())} edges with "
                             f"{sorted(set(cnt.tolist()) - {2})} faces")
        self.E = E
        self.QE = inv.reshape(m, 4)
        n = len(self.V)
        self.valence = np.bincount(E.ravel(), minlength=n)
        # each directed half-edge once (orientation consistent)
        d = a * n + b
        if len(np.unique(d)) != len(d):
            raise ValueError("cage faces are not consistently oriented")

    @property
    def n(self):
        return len(self.V)

    def irregular_faces(self):
        return (self.valence[self.Q] != 4).any(axis=1)


def _sp():
    import scipy.sparse as sp
    return sp


def subdivision_matrix(cage):
    """Catmull-Clark: the new vertices (old vertices first, then one per
    edge, then one per face) as a sparse (n + e + m, n) matrix of the old,
    and the new quads (face q's child k is 4q + k, at corner k)."""
    sp = _sp()
    V, Q, E, QE = cage.V, cage.Q, cage.E, cage.QE
    n, m, e = len(V), len(Q), len(E)
    Af = sp.csr_matrix((np.full(4 * m, 0.25), (np.repeat(np.arange(m), 4), Q.ravel())), shape=(m, n))
    EV = sp.csr_matrix((np.ones(2 * e), (np.repeat(np.arange(e), 2), E.ravel())), shape=(e, n))
    EF = sp.csr_matrix((np.ones(4 * m), (QE.ravel(), np.repeat(np.arange(m), 4))), shape=(e, m))
    Ae = 0.25 * EV + 0.25 * (EF @ Af)
    val = cage.valence.astype(float)
    VV = (EV.T @ EV).tolil()
    VV.setdiag(0)
    VV = VV.tocsr()
    VF = sp.csr_matrix((np.ones(4 * m), (Q.ravel(), np.repeat(np.arange(m), 4))), shape=(n, m))
    D = sp.diags(1.0 / val ** 2)
    Av = sp.diags((val - 2.0) / val) + D @ VV + D @ (VF @ Af)
    S = sp.vstack([Av, Ae, Af]).tocsr()
    k = np.arange(4)
    vk = Q
    ek = n + QE
    ekm = n + np.roll(QE, 1, axis=1)
    f = (n + e + np.arange(m))[:, None].repeat(4, 1)
    Q2 = np.stack([vk, ek, f, ekm], axis=2).reshape(4 * m, 4)
    del k
    return S, Q2


def limit_matrix(cage):
    """Each vertex's position on the limit surface (Catmull-Clark's
    limit mask on quads: (n^2 v + 4 sum edge neighbours + sum diagonals)
    / (n (n + 5)))."""
    sp = _sp()
    Q, E = cage.Q, cage.E
    n = cage.n
    val = cage.valence.astype(float)
    rows, cols, w = [np.arange(n)], [np.arange(n)], [val ** 2]
    rows += [E[:, 0], E[:, 1]]
    cols += [E[:, 1], E[:, 0]]
    w += [np.full(len(E), 4.0), np.full(len(E), 4.0)]
    rows.append(Q.ravel())
    cols.append(np.roll(Q, -2, axis=1).ravel())
    w.append(np.ones(Q.size))
    r, c, ww = np.concatenate(rows), np.concatenate(cols), np.concatenate(w)
    L = sp.csr_matrix((ww, (r, c)), shape=(n, n))
    return sp.diags(1.0 / (val * (val + 5.0))) @ L


class Hierarchy:
    """The cage subdivided `level` times: M (fine limit points from the
    cage's control points), the fine quads, and for every fine vertex the
    root face and its (u, v) on that face (dyadic, exact parameters of the
    limit surface)."""

    def __init__(self, cage, level=LEVEL):
        sp = _sp()
        self.cage, self.level = cage, level
        m = len(cage.Q)
        P = sp.identity(cage.n, format="csr")
        c = cage
        root = np.arange(m)
        o = np.zeros((m, 2))
        au = np.tile([1.0, 0.0], (m, 1))
        av = np.tile([0.0, 1.0], (m, 1))
        for _ in range(level):
            S, Q2 = subdivision_matrix(c)
            P = (S @ P).tocsr()
            corners = np.stack([o, o + au, o + au + av, o + av], axis=1)       # (m, 4, 2)
            nk = np.roll(corners, -1, axis=1)
            pk = np.roll(corners, 1, axis=1)
            o2 = corners.reshape(-1, 2)
            au2 = ((nk - corners) / 2).reshape(-1, 2)
            av2 = ((pk - corners) / 2).reshape(-1, 2)
            root = np.repeat(root, 4)
            o, au, av = o2, au2, av2
            c = Cage(S @ c.V, Q2)
        self.fine = c
        self.M = (limit_matrix(c) @ P).tocsr()
        self.Pm = P                 # fine control points from the cage's
        self.root, self.o, self.au, self.av = root, o, au, av
        s = 2 ** level
        corners = np.stack([o, o + au, o + au + av, o + av], axis=1)
        ij = np.rint(corners * s).astype(np.int64)
        G = np.full((m, s + 1, s + 1), -1, np.int64)
        G[np.repeat(root, 4), ij[..., 0].ravel(), ij[..., 1].ravel()] = c.Q.ravel()
        if (G < 0).any():
            raise RuntimeError("subdivision grid incomplete")
        self.G = G              # (m, s+1, s+1): fine vertex at (u, v) = (i, j) / s on root face

    def limit(self, C=None):
        return self.M @ (self.cage.V if C is None else C)


# ------------------------------------------------------------------ input

def _warp(P, spots, inverse=False):
    """Space blown up round each spot (centre c, factor k, radius R):
    x -> c + (x - c) s(r), s = 1 + (k - 1) exp(-r^2 / R^2). r s(r) rises
    monotonically for k < 3.2 (its slope is least at r^2 = 1.5 R^2), so the
    map is one to one; `inverse` undoes it (bisection on r, spots in reverse
    order)."""
    P = np.array(P, float)
    for c, k, rad in (spots[::-1] if inverse else spots):
        d = P - np.asarray(c, float)
        r = np.linalg.norm(d, axis=1)
        if not inverse:
            P = P + d * ((k - 1.0) * np.exp(-(r / rad) ** 2))[:, None]
            continue
        lo, hi = r / k, r.copy()
        for _ in range(48):
            mid = 0.5 * (lo + hi)
            big = mid * (1.0 + (k - 1.0) * np.exp(-(mid / rad) ** 2)) > r
            hi, lo = np.where(big, mid, hi), np.where(big, lo, mid)
        P = np.asarray(c, float) + d * (0.5 * (lo + hi) / np.maximum(r, 1e-300))[:, None]
    return P


def quad_cage(mesh, faces, crease=None, seed=0, finer=None):
    """An all-quad cage over a closed triangle mesh with about `faces`
    quads (Instant Meshes, pure-quad mode), its vertices on the mesh.

    `finer`: spots (centre, factor k, radius) where the faces should be k
    times smaller. Instant Meshes lays one edge length everywhere, so it
    lays it over the mesh with space blown up k times round each spot
    (`_warp`) and the quads are mapped back: there its edges come out k
    times shorter, elsewhere as asked."""
    if finer:
        import trimesh

        big = trimesh.Trimesh(_warp(mesh.vertices, finer), np.asarray(mesh.faces), process=False)
        v, f = _quads(big, int(faces * big.area / mesh.area), crease)
        return _cage_from_quads(mesh, _warp(v, finer, inverse=True), f)
    return _cage_from_quads(mesh, *_quads(mesh, faces, crease))


def _quads(mesh, faces, crease):
    # Instant Meshes' pure-quad mode subdivides its quad-dominant mesh once,
    # so ask for a quarter of the faces; the in-memory call hands back
    # triangles, so the quads come through an OBJ
    kw = dict(target_face_count=max(8, int(faces) // 4), pure_quad=True, deterministic=True,
              smooth_iterations=2, crease_angle=-1.0 if crease is None else float(crease))
    v, f = _instant_quads(mesh, kw)
    if not len(f):
        # Instant Meshes lays its field over the input's own vertex graph:
        # slivers (an optimized part's exact bores and mount faces are long
        # triangles, aspect over 10^4 beside 0.07 mm ones) leave it thousands
        # of singularities and nothing to extract (the arm link: 0 faces at
        # every density). The cage only needs the shape, so the layout runs on
        # an even resampling of it; the fit is to `mesh` itself.
        v, f = _instant_quads(_even(mesh, 0.25 * float(np.sqrt(mesh.area / max(int(faces), 1)))), kw)
    if not len(f):
        raise ValueError("Instant Meshes extracted no faces")
    return v, f


def _even(mesh, step):
    """`mesh` resampled with even triangles about `step` mm (marching cubes
    of its signed distance), cached on it: the same surface for Instant
    Meshes' layout, without slivers. The grid is refined until it keeps
    every hole (same Euler number), at most about 12M points."""
    cache = mesh.__dict__.setdefault("_lp_even", {})
    lo, hi = mesh.bounds[0], mesh.bounds[1]
    size = hi - lo
    step = max(min(float(step), float(np.median(mesh.edges_unique_length))), float(np.prod(size) / 12e6) ** (1 / 3))
    key = round(step, 4)
    if key in cache:
        return cache[key]
    import trimesh
    from skimage import measure

    out = None
    for _ in range(3):
        pad = 2.0 * step
        # off the grid by an odd fraction: a flat face lying exactly on a
        # row of samples meshes as a torn sheet
        axes = [np.arange(a - pad, b + pad + step, step) + 0.317 * step for a, b in zip(lo, hi)]
        P = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)
        S = _signed_distance(mesh, P).reshape([len(a) for a in axes])
        vv, ff, _, _ = measure.marching_cubes(S, 0.0, spacing=(step, step, step))
        out = trimesh.Trimesh(vv + [a[0] for a in axes], ff[:, ::-1], process=True)
        if out.euler_number == mesh.euler_number or float(np.prod(size)) / (0.5 * step) ** 3 > 4e7:
            break
        step *= 0.5
    cache[key] = out
    return out


def _signed_distance(mesh, P):
    """Signed distance (negative inside) from points to a closed mesh."""
    try:
        import igl
        return igl.signed_distance(P, np.asarray(mesh.vertices, float), np.asarray(mesh.faces, np.int64))[0]
    except ImportError:
        import trimesh
        return -trimesh.proximity.signed_distance(mesh, P)


def _instant_quads(mesh, kw):
    """Instant Meshes' pure-quad remesh of `mesh`: (vertices, quads), quads
    possibly empty."""
    import os
    import tempfile

    import pyinstantmeshes as im

    with tempfile.TemporaryDirectory() as td:
        src, dst = os.path.join(td, "in.obj"), os.path.join(td, "out.obj")
        with open(src, "w") as fh:
            np.savetxt(fh, np.asarray(mesh.vertices, float), fmt="v %.6f %.6f %.6f")
            np.savetxt(fh, np.asarray(mesh.faces, np.int64) + 1, fmt="f %d %d %d")
        out = os.dup(1)
        null = os.open(os.devnull, os.O_WRONLY)
        os.dup2(null, 1)                      # Instant Meshes logs to stdout
        try:
            im.remesh_file(src, dst, **kw)
        finally:
            os.dup2(out, 1)
            os.close(null)
            os.close(out)
        v, f = [], []
        with open(dst) as fh:
            for line in fh:
                if line.startswith("v "):
                    v.append([float(x) for x in line.split()[1:4]])
                elif line.startswith("f "):
                    f.append([int(x.split("/")[0]) - 1 for x in line.split()[1:]])
    if any(len(x) != 4 for x in f):
        raise RuntimeError("quad remesh returned faces that are not quads")
    return np.asarray(v, float).reshape(-1, 3), np.asarray(f, np.int64).reshape(-1, 4)


def _cage_from_quads(mesh, v, f):
    """The quads cleaned (no repeated corner, unused vertices dropped,
    oriented like `mesh`, holes closed, pinches split, doublets dissolved)."""
    f = f[np.array([len(set(x)) == 4 for x in f.tolist()], bool)]      # no repeated corner
    used = np.unique(f)
    remap = np.full(len(v), -1)
    remap[used] = np.arange(len(used))
    v, f = v[used], remap[f]
    # orient outward like the input
    tri = np.concatenate([f[:, [0, 1, 2]], f[:, [0, 2, 3]]])
    t = v[tri]
    vol = np.einsum("ij,ij->i", t[:, 0], np.cross(t[:, 1], t[:, 2])).sum() / 6
    if vol * float(mesh.volume) < 0:
        f = f[:, ::-1]
    try:
        v, f = _close_holes(v, f)
    except ValueError:
        # a hole with an odd number of sides cannot be tiled by quads on its
        # own corners (every quad tiling has an even boundary); split every
        # quad in four, which doubles every side, and close it then. The
        # new points are on the old quads, off the surface; the fit moves them.
        v, f = _close_holes(*_split_quads(v, f))
    v, f = _split_pinches(v, f)
    v, f = _dissolve_doublets(v, f)
    return Cage(v, f)


def _thickness(mesh, n=4000, q=10):
    """The part's thin walls and members, mm: the `q`th percentile of the
    distance straight through it from points on its surface (inward
    rays)."""
    pts, fi = mesh.sample(n, return_index=True) if hasattr(mesh, "sample") else (None, None)
    nrm = mesh.face_normals[fi]
    o = pts - 1e-4 * float(np.linalg.norm(mesh.extents)) * nrm
    try:
        loc, ray, _ = mesh.ray.intersects_location(o, -nrm, multiple_hits=False)
    except Exception:
        return float(np.linalg.norm(mesh.extents))
    if not len(ray):
        return float(np.linalg.norm(mesh.extents))
    d = np.linalg.norm(loc - o[ray], axis=1)
    return float(np.percentile(d[d > 0], q)) if (d > 0).any() else float(np.linalg.norm(mesh.extents))


def _split_quads(v, f):
    """Each quad in four (edge midpoints, face centres; no smoothing):
    an open quad mesh's every boundary loop doubles in length."""
    n = len(v)
    a = f.ravel()
    b = np.roll(f, -1, axis=1).ravel()
    key = np.minimum(a, b) * n + np.maximum(a, b)
    uk, inv = np.unique(key, return_inverse=True)
    mids = 0.5 * (v[uk // n] + v[uk % n])
    cen = v[f].mean(axis=1)
    m = inv.reshape(-1, 4) + n                      # midpoint of edge (i, i+1)
    c = np.arange(len(f)) + n + len(uk)
    q = []
    for i in range(4):
        q.append(np.stack([f[:, i], m[:, i], c, m[:, (i - 1) % 4]], 1))
    return np.vstack([v, mids, cen]), np.vstack(q)


def _hole_loops(f):
    """The boundary loops of an open quad mesh, each as its vertices in
    order. Walked face by face round each corner, so a vertex two holes (or
    one hole twice) pass through is followed the right way."""
    a = f.ravel()
    b = np.roll(f, -1, axis=1).ravel()
    he = {(int(x), int(y)): i for i, (x, y) in enumerate(zip(a, b))}
    prev = lambda h: 4 * (h // 4) + (h % 4 - 1) % 4  # noqa: E731
    # hole edge (y -> x) for each face half-edge (x -> y) without a twin
    open_ = {h for (x, y), h in he.items() if (y, x) not in he}
    loops, used = [], set()
    for h0 in open_:
        if h0 in used:
            continue
        loop, h = [], h0
        for _ in range(len(open_) + 1):
            used.add(h)
            loop.append(int(b[h]))                    # the hole runs y -> x
            # round x: from face edge x -> y back through faces to the next open one
            x = int(a[h])
            g = prev(h)                               # p -> x in the same face
            for _ in range(64):
                t = he.get((x, int(a[g])))            # twin x -> p
                if t is None:
                    break
                g = prev(t)
            # g is (p -> x) with no twin: the hole goes on x -> p, i.e. face edge g
            h = g
            if h == h0:
                break
        loops.append(loop)
    return loops


def _close_holes(v, f):
    """Instant Meshes can leave a small hole where its extraction gives up;
    a hole with an even number of sides closes with quads round a new
    centre vertex. Returns (v, f); an odd or very long hole raises."""
    n = len(v)
    new_v, new_f = [], []
    for loop in _hole_loops(f):
        L = len(loop)
        if L % 2 or L > 64:
            raise ValueError(f"quad cage has a hole of {L} sides")
        if L == 4:
            new_f.append(loop)
            continue
        ci = n + len(new_v)
        new_v.append(v[loop].mean(axis=0))
        for i in range(0, L, 2):
            new_f.append([ci, loop[i], loop[(i + 1) % L], loop[(i + 2) % L]])
    if not new_f:
        return v, f
    if new_v:
        v = np.vstack([v, new_v])
    return v, np.vstack([f, np.asarray(new_f, np.int64)])


def _split_pinches(v, f):
    """A vertex where two fans of quads touch (Instant Meshes folds a
    quad pair onto itself now and then) becomes one vertex per fan."""
    n = len(v)
    a = f.ravel()
    b = np.roll(f, -1, axis=1).ravel()
    owner = {(int(x), int(y)): i for i, (x, y) in enumerate(zip(a, b))}
    corners = {}
    for i, x in enumerate(a.tolist()):
        corners.setdefault(x, []).append(i)
    f = f.copy()
    extra = []
    for vtx, cs in corners.items():
        left = set(cs)
        fans = []
        while left:
            c0 = left.pop()
            fan = [c0]
            c = c0
            for _ in range(len(cs) + 1):
                q, k = divmod(c, 4)
                prev = int(a[4 * q + (k - 1) % 4])
                h = owner.get((vtx, prev))            # twin of (prev -> vtx)
                if h is None or h == c0:
                    break
                fan.append(h)
                left.discard(h)
                c = h
            fans.append(fan)
        for fan in fans[1:]:
            idx = n + len(extra)
            extra.append(v[vtx])
            for c in fan:
                f[c // 4, c % 4] = idx
    if extra:
        v = np.vstack([v, extra])
    return v, f


def _dissolve_doublets(v, f):
    """A vertex with two edges (a doublet: two quads folded round it) has
    no Catmull-Clark limit worth the name; the two quads become one."""
    for _ in range(100):
        a = f.ravel()
        b = np.roll(f, -1, axis=1).ravel()
        key = np.unique(np.stack([np.minimum(a, b), np.maximum(a, b)], 1), axis=0)
        val = np.bincount(key.ravel(), minlength=len(v))
        bad = np.nonzero((val == 2) & (np.bincount(f.ravel(), minlength=len(v)) == 2))[0]
        if not len(bad):
            break
        w = int(bad[0])
        fs = np.nonzero((f == w).any(1))[0]
        if len(fs) != 2:
            raise ValueError("quad cage has a folded doublet")
        q1, q2 = f[fs[0]], f[fs[1]]
        k1 = int(np.nonzero(q1 == w)[0][0])
        r1 = np.roll(q1, -k1)                     # (w, a, x, b)
        k2 = int(np.nonzero(q2 == w)[0][0])
        r2 = np.roll(q2, -k2)                     # (w, b, y, a)
        if r2[1] != r1[3] or r2[3] != r1[1]:
            break
        merged = np.array([r1[1], r1[2], r1[3], r2[2]])
        keep = np.ones(len(f), bool)
        keep[fs] = False
        f = np.vstack([f[keep], merged[None]])
    used = np.unique(f)
    remap = np.full(len(v), -1)
    remap[used] = np.arange(len(used))
    return v[used], remap[f]


def _closest(mesh, P, edge):
    from .brep import _closest as cl
    return cl(mesh, P, edge=edge)


def _samples(mesh, n, seed=0):
    import trimesh
    P, fi = trimesh.sample.sample_surface_even(mesh, int(n), seed=seed)
    if len(P) < 0.5 * n:
        P, fi = trimesh.sample.sample_surface(mesh, int(n), seed=seed)
    return np.asarray(P, float), np.asarray(mesh.face_normals[fi], float)


def fit_cage(cage, mesh, level=LEVEL, iterations=ITERATIONS, smooth=SMOOTH, edge=None, data=None, fair=0.0, H=None,
             settle=None, weights=None):
    """Move the cage's control points so its limit surface lies on `mesh`
    (two-way closest-point least squares, a membrane term on the cage).
    A round stops early once no control point moves more than `settle`
    mm in an iteration (the closest points no longer change). `weights`
    (one per fine limit point, one per data sample) scale each point's
    pull (`bound_worst`). Returns (cage, hierarchy) with the fitted
    control points in cage.V."""
    from scipy.spatial import cKDTree
    from scipy.sparse.linalg import factorized

    sp = _sp()
    H = H if H is not None else Hierarchy(cage, level)
    M = H.M
    n = cage.n
    edge = edge or _edge_for(mesh)
    if data is None:
        data = _samples(mesh, min(4 * M.shape[0], 400_000))
    X, NX = data
    wP, wX = weights if weights is not None else (np.ones(M.shape[0]), np.ones(len(X)))
    # membrane: each control point near the mean of its neighbours
    E = cage.E
    W = sp.csr_matrix((np.ones(2 * len(E)), (np.r_[E[:, 0], E[:, 1]], np.r_[E[:, 1], E[:, 0]])), shape=(n, n))
    Lap = sp.identity(n) - sp.diags(1.0 / cage.valence) @ W
    C = cage.V.copy()
    lam = smooth * M.shape[0] / n
    stiff = np.ones(n)                      # local membrane factor, raised where the surface folds
    MT = M.T.tocsr()
    MtM = (MT @ M).tocsc()
    LL = Lap @ Lap
    Bend = (fair * M.shape[0] / n) * (LL.T @ LL)
    for rnd in range(4):
        Reg = (Lap.T @ sp.diags(lam * stiff) @ Lap + Bend).tocsc()
        pre = None
        for it in range(iterations):
            P = M @ C
            NP = _limit_normals(H, P)
            q, nq = _closest(mesh, P, edge)
            # surface -> mesh: move only along the mesh normal (points slide); a
            # closest point on the far side of a thin wall (its normal facing
            # away) is not this point's surface: no pull
            ok1 = np.einsum("ij,ij->i", nq, NP) > 0.3
            t1 = P - np.where(ok1, np.einsum("ij,ij->i", P - q, nq), 0.0)[:, None] * nq
            # mesh -> surface: every input sample pulls its nearest limit point
            j = cKDTree(P).query(X, workers=-1)[1]
            Nj = NP[j]
            ok2 = np.einsum("ij,ij->i", NX, Nj) > 0.3
            t2 = P[j] + (np.einsum("ij,ij->i", X - P[j], Nj) * ok2)[:, None] * Nj
            w2 = 0.5 * len(P) / len(X)
            # normal equations of [M; sqrt(w2) M[j]; membrane; bending]: the
            # sample rows gather onto the limit points they pull, so it is
            # M^T diag(1 + w2 count) M, and M[j] (as many rows as samples)
            # is never formed
            cnt = wP + w2 * np.bincount(j, weights=wX, minlength=len(P))
            acc = np.zeros_like(P)
            np.add.at(acc, j, wX[:, None] * t2)
            rhs = MT @ (wP[:, None] * t1 + w2 * acc)
            if pre is None:
                # factor once a round, at the mean pull; conjugate gradients
                # from there (the counts move a little each round)
                pre = factorized((MtM * float(cnt.mean()) + Reg).tocsc())

            def K(x, cnt=cnt):
                return MT @ (cnt[:, None] * (M @ x)) + Reg @ x
            C0 = C
            C = _pcg(K, pre, rhs, C)
            if settle is not None and it >= 2 and float(np.abs(C - C0).max()) < settle:
                break
        folds = _folds(H, C)
        if not len(folds):
            break
        # stiffen the cage round each fold and fit again
        verts = np.unique(cage.Q[folds])
        ring = np.unique(cage.Q[(np.isin(cage.Q, verts)).any(1)])
        stiff[ring] *= 30.0
    H.folds = int(len(_folds(H, C)))
    cage.V = C
    return cage, H


def _pcg(K, pre, B, X0, rtol=1e-7, maxiter=200):
    """Solve K X = B (symmetric positive definite, three right-hand
    sides at once) by conjugate gradients from X0, preconditioned by a
    factorization `pre` of a matrix near K."""
    X = X0.copy()
    R = B - K(X)
    nb = np.maximum(np.linalg.norm(B, axis=0), 1e-300)
    Z = np.stack([pre(R[:, k]) for k in range(R.shape[1])], 1)
    Pd = Z.copy()
    rz = np.einsum("ij,ij->j", R, Z)
    for _ in range(maxiter):
        if (np.linalg.norm(R, axis=0) <= rtol * nb).all():
            break
        KP = K(Pd)
        a = rz / np.maximum(np.einsum("ij,ij->j", Pd, KP), 1e-300)
        X += a * Pd
        R -= a * KP
        Z = np.stack([pre(R[:, k]) for k in range(R.shape[1])], 1)
        rz2 = np.einsum("ij,ij->j", R, Z)
        Pd = Z + (rz2 / np.maximum(rz, 1e-300)) * Pd
        rz = rz2
    return X


def _folds(H, C, cos=0.5):
    """Cage faces whose limit surface folds over: a fine quad turned more
    than 60 degrees from the mean normal of its corners."""
    P = H.M @ C
    Q = H.fine.Q
    fn = np.cross(P[Q[:, 2]] - P[Q[:, 0]], P[Q[:, 3]] - P[Q[:, 1]])
    fn /= np.maximum(np.linalg.norm(fn, axis=1, keepdims=True), 1e-300)
    Nv = _limit_normals(H, P)
    d = np.einsum("ij,ij->i", fn, Nv[Q].mean(axis=1))
    return np.unique(H.root[d < cos])


def _edge_for(mesh):
    ext = float(np.linalg.norm(mesh.bounds[1] - mesh.bounds[0]))
    return max(ext / 300.0, 1e-6)


def _limit_normals(H, P):
    """Vertex normals of the fine limit points (area-weighted from the fine
    quads: at level 3 within a fraction of a degree of the limit's own)."""
    Q = H.fine.Q
    a = P[Q[:, 2]] - P[Q[:, 0]]
    b = P[Q[:, 3]] - P[Q[:, 1]]
    fn = np.cross(a, b)
    N = np.zeros_like(P)
    for k in range(4):
        np.add.at(N, Q[:, k], fn)
    return N / np.maximum(np.linalg.norm(N, axis=1, keepdims=True), 1e-300)


def deviation(H, C, mesh, data=None, edge=None):
    """Two-way distance between the limit surface and the mesh, mm:
    surface -> mesh at the fine limit points, mesh -> surface at samples of
    the mesh against the fine limit mesh (sag at level 3 is far below
    any tolerance in use)."""
    import trimesh

    edge = edge or _edge_for(mesh)
    P = H.M @ C
    q, _ = _closest(mesh, P, edge)
    d1 = np.linalg.norm(P - q, axis=1)
    Q = H.fine.Q
    lm = trimesh.Trimesh(P, np.concatenate([Q[:, [0, 1, 2]], Q[:, [0, 2, 3]]]), process=False)
    if data is None:
        data = _samples(mesh, min(2 * len(P), 300_000), seed=1)
    X = data[0]
    q2, _ = _closest(lm, X, edge)
    d2 = np.linalg.norm(X - q2, axis=1)

    def st(d):
        return {"mean": round(float(d.mean()), 4), "p99": round(float(np.percentile(d, 99)), 4),
                "max": round(float(d.max()), 4)}
    return {"surfaceToMesh": st(d1), "meshToSurface": st(d2),
            "p99": round(float(max(np.percentile(d1, 99), np.percentile(d2, 99))), 4),
            "max": round(float(max(d1.max(), d2.max())), 4)}, (d1, d2, X)


def _closest_limit(H, C, X):
    """Closest points on the fine limit mesh to X."""
    import trimesh

    P = H.M @ C
    Q = H.fine.Q
    lm = trimesh.Trimesh(P, np.concatenate([Q[:, [0, 1, 2]], Q[:, [0, 2, 3]]]), process=False)
    return _closest(lm, X, _edge_for(lm))[0]


def misses(H, C, mesh, d1, d2, X, bound, min_hole, through=True):
    """Where the limit surface misses the mesh by more than `bound` mm,
    grouped into places (limit points and mesh samples within 1 mm of
    each other). Each place is a "hole" when the surface there spans air:
    a ray from its limit points along the surface normal, either way,
    travels farther than four times the miss before it meets the mesh (the
    cage closed a through-hole: Instant Meshes drops handles smaller than
    its faces); else a "surface" miss (a feature narrower than the cage
    follows). A hole's `diameter` is twice its largest miss (the membrane
    sits a radius from the wall: only the surface's own points count, a
    bore's walls miss by its depth) or, with none of those, the median
    width across it along the missed walls' normals; one under the process's `min_hole`
    could not be printed open anyway. A short hole in a thin web turns the
    membrane's normals off its axis and those rays meet its wall, so with
    `through` a place is also a hole when its limit points are the mouth
    of one by `subdedit.holes` (a straight way through air to the
    membrane over the other mouth). Returns (places, kind of each limit
    point: 0 within, 1 surface miss, 2 hole), places worst first."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    from scipy.spatial import cKDTree

    P = H.M @ C
    kind = np.zeros(len(P), np.int8)
    i1 = np.flatnonzero(d1 > bound)
    i2 = np.flatnonzero(d2 > bound)
    if not len(i1) and not len(i2):
        return [], kind
    # a mesh sample that misses stands for its nearest limit point
    near = cKDTree(P).query(X[i2], workers=-1)[1] if len(i2) else np.zeros(0, np.int64)
    pts = np.r_[P[i1], X[i2]]
    d = np.r_[d1[i1], d2[i2]]
    li = np.r_[i1, near]
    NP = _limit_normals(H, P)
    o, n = P[li], NP[li]

    def reach(o, dirs):
        loc, ri, _ = mesh.ray.intersects_location(o + 1e-4 * dirs, dirs, multiple_hits=False)
        w = np.full(len(o), np.inf)
        w[ri] = np.linalg.norm(loc - o[ri], axis=1)
        return w
    air = np.minimum(reach(o, n), reach(o, -n)) > 4.0 * d
    if through and len(i1):
        from .subdedit import holes
        mouth = np.zeros(len(P), bool)
        for h in holes(H, C, mesh, d1 > bound, d1):
            mouth[h["points"]] = True
        air |= mouth[li]
    # across: from a missed mesh sample along its own outward normal to the
    # far wall (inf on an open face)
    across = np.full(len(pts), np.inf)
    if len(i2):
        _, nx = _closest(mesh, X[i2], _edge_for(mesh))
        across[len(i1):] = reach(X[i2], nx)
    pr = cKDTree(pts).query_pairs(1.0, output_type="ndarray")
    k, lab = connected_components(coo_matrix((np.ones(len(pr)), (pr[:, 0], pr[:, 1])), shape=(len(pts),) * 2),
                                  directed=False)
    out = []
    for g in range(k):
        s = lab == g
        hole = bool(air[s].mean() > 0.5)
        kind[li[s]] = 2 if hole else 1
        m = float(d[s].max())
        c = pts[s][np.argmax(d[s])]
        pl = {"at": np.round(c, 2).tolist(), "miss": round(m, 3), "kind": "hole" if hole else "surface"}
        if hole:
            own = s[:len(i1)]
            w = across[s][np.isfinite(across[s])]
            dia = 2.0 * d[:len(i1)][own].max() if own.any() else (float(np.median(w)) if len(w) else 2.0 * m)
            pl["diameter"] = round(float(dia), 2)
            pl["printable"] = bool(pl["diameter"] >= min_hole)
        out.append(pl)
    out.sort(key=lambda p: -p["miss"])
    return out, kind


def noise(mesh, n=20000, rings=2):
    """The scatter the mesh carries at its own sampling scale, mm: the
    median RMS distance of vertices' neighbourhoods (`rings` edges out
    along the mesh, so a double wall or a sheet across a gap is not
    counted) from the quadric through them. A scan or a generated mesh is
    no better than this; a surface fitted closer would be fitting noise."""
    import trimesh

    m = trimesh.Trimesh(np.asarray(mesh.vertices, float), np.asarray(mesh.faces, np.int64), process=False)
    m.merge_vertices()
    V = m.vertices
    nb = m.vertex_neighbors
    rng = np.random.default_rng(0)
    idx = rng.choice(len(V), min(n, len(V)), replace=False)
    r = []
    for i in idx:
        ring = {int(i)}
        front = {int(i)}
        for _ in range(rings):
            front = {w for u in front for w in nb[u]} - ring
            ring |= front
        if len(ring) < 8:
            continue
        P = V[list(ring)] - V[list(ring)].mean(axis=0)
        _, _, vt = np.linalg.svd(P, full_matrices=False)
        x, y, z = P @ vt[0], P @ vt[1], P @ vt[2]
        A = np.stack([x * x, x * y, y * y, x, y, np.ones_like(x)], 1)
        c, *_ = np.linalg.lstsq(A, z, rcond=None)
        r.append(np.sqrt(np.mean((z - A @ c) ** 2) * len(z) / max(len(z) - 6, 1)))
    return float(np.median(r)) if r else 0.0


NEARLY = 1.15       # a cage whose 99th percentile is within this of the tolerance is fitted on, not laid again
ACCURACY = 0.3      # mm: FDM's typical +- (dfm_reference: PLA/PETG/ABS/nylon), loadpath's first process
MIN_HOLE = 2.0      # mm: FDM's smallest hole that prints open (spec.PROCESSES)


def pick_tolerance(accuracy=None, noise=None):
    """How close the surface has to sit to the mesh, mm, and why.

    A printed part is only as true as its process (`accuracy`, +- mm, the
    material's figure in dfm_reference; FDM's 0.3 when not given): a fit
    error a third of it adds 5% to the error the part already has
    (sqrt(1 + 1/9)), which no measurement of the print would show. A scan
    or a generated mesh is only as true as its own `noise`: fitting closer
    than twice its scatter fits the scatter."""
    acc = ACCURACY if accuracy is None else float(accuracy)
    tol, why = acc / 3.0, f"a third of the process accuracy (+-{acc:g} mm)"
    if noise is not None and 2.0 * noise > tol:
        tol, why = 2.0 * noise, f"twice the input's own noise ({noise:.3f} mm RMS)"
    return tol, why


def _fewer_stars(cage, H, mesh, data, check, tol, bound, min_hole, level, dev, progress=None):
    """Star points cancelled in pairs (subdcoarse.cancel: edge rotations
    and diagonal collapses) where the surface can do without them: the
    moved neighbourhoods are fitted again (subdedit.refit) and the moves
    stand only if the 99th percentile holds and no place misses by more
    than twice the bound or closes a printable hole that did not before;
    where a new place misses, the vertices round it are frozen and the
    rest tried once more. Returns (cage, hierarchy, deviation, report)."""
    from scipy.spatial import cKDTree

    from .subdcoarse import cancel
    from .subdedit import refit

    def bad(H, C, d1, d2, X):
        pl, _ = misses(H, C, mesh, d1, d2, X, bound, min_hole)
        return [p for p in pl if p.get("printable") or (p["kind"] == "surface" and p["miss"] > 2.0 * bound)]
    d1, d2, X = deviation(H, cage.V, mesh, data=check)[1]
    before = {tuple(np.round(p["at"], 0)) for p in bad(H, cage.V, d1, d2, X)}
    stars = int((cage.valence != 4).sum())
    frozen = np.zeros(0, np.int64)
    h = float(np.sqrt(mesh.area / len(cage.Q)))
    rep = {"moves": 0, "starPoints": [stars, stars]}
    for attempt in range(2):
        Vn, Qn, touched, taken = cancel(cage.V, cage.Q, frozen=frozen)
        if not taken:
            break
        c2 = Cage(Vn, Qn)
        H2 = Hierarchy(c2, level)
        refit(c2, H2, mesh, touched, data, settle=tol / 50)
        d2v, (e1, e2, X2) = deviation(H2, c2.V, mesh, data=check)
        new = [p for p in bad(H2, c2.V, e1, e2, X2) if tuple(np.round(p["at"], 0)) not in before]
        if progress:
            progress(f"star points {stars} -> {int((c2.valence != 4).sum())} ({taken} moves): "
                     f"p99 {d2v['p99']:.4f} mm, {len(new)} new places over {2 * bound:.2f}")
        if d2v["p99"] <= tol and not new:
            rep = {"moves": int(taken), "starPoints": [stars, int((c2.valence != 4).sum())]}
            return c2, H2, d2v, rep
        if not new:
            break
        near = cKDTree(cage.V).query_ball_point(np.array([p["at"] for p in new]), 3.0 * h)
        frozen = np.unique(np.r_[frozen, np.concatenate([np.asarray(x, np.int64) for x in near])])
    return cage, H, dev, rep


def surface_mesh(mesh, tolerance=None, faces=None, max_faces=12000, rounds=6, level=LEVEL, progress=None,
                 crease=None, accuracy=None, source=None, min_hole=None):
    """Fit a Catmull-Clark surface to a closed mesh. The cage gets denser
    until the two-way deviation's 99th percentile is within `tolerance` mm
    (default `pick_tolerance`: from the process `accuracy` and the noise
    of `source`, the mesh as it came in before any repair, else `mesh`)
    and no hole the process could print (`min_hole`, mm) is closed over.
    Then the worst point is bounded too: every place that misses by more
    than three times the tolerance (the process's own error, so the print
    is off there by at most sqrt(2) of what the process gives anyway)
    pulls harder until it is within (`misses`). Holes under `min_hole`
    the cage closed are reported, not chased: the printer would not have
    left them open. Returns (cage, hierarchy, report)."""
    t0 = time.time()
    ext = float(np.linalg.norm(mesh.bounds[1] - mesh.bounds[0]))
    nz = noise(source if source is not None else mesh)
    if tolerance is not None:
        tol, why = float(tolerance), "given"
    else:
        tol, why = pick_tolerance(accuracy, nz)
    bound = 3.0 * tol
    min_hole = MIN_HOLE if min_hole is None else float(min_hole)
    area = float(mesh.area)
    # a face no wider than the part's thin walls and members: a cage
    # coarser than a member cannot follow it (Instant Meshes then bridges
    # the holes between members, or leaves holes of its own); ext / 12 on
    # a chunky part. (At 1.2 times the thickness the arm link's first cage
    # missed the tolerance on every run, p99 0.125-0.146 against 0.1, and
    # the one laid after it, 1.4 times the faces, met it: a second lay
    # each time for the sake of a fifth of an edge.)
    edge = min(ext / 12.0, _thickness(mesh))
    n = int(faces) if faces else min(max_faces, max(96, int(area / edge ** 2)))
    data = _samples(mesh, 300_000)
    check = _samples(mesh, 200_000, seed=1)
    tries = []
    best = None
    spots = []          # where the cage is laid finer (quad_cage `finer`)
    for r in range(rounds):
        cage = None
        while cage is None:
            for k, f in enumerate((1.0, 1.08, 0.93, 1.17, 1.3, 1.5, 1.8, 2.2)):
                # an odd or pinched hole, a fold at this density: nudge it
                try:
                    cage = quad_cage(mesh, int(n * f), crease=crease, finer=spots)
                    break
                except ValueError as err:
                    if progress:
                        progress(f"quad cage at {int(n * f)} faces: {err}; retrying")
            if cage is None:
                # holes at every nudge: members thinner than the cage's edges
                # (the remesher cannot close a quad strip round them); finer
                if n * 2.2 >= max_faces:
                    raise RuntimeError("no closed quad cage")
                n = int(min(max_faces, n * 2.5))
        cage, H = fit_cage(cage, mesh, level=level, data=data, settle=tol / 50)
        dev, (d1, d2, Xc) = deviation(H, cage.V, mesh, data=check)
        edits = []
        for step in range(2):
            places, _ = misses(H, cage.V, mesh, d1, d2, Xc, bound, min_hole)
            open_holes = [p for p in places if p.get("printable")]
            # a miss twice the bound is a feature narrower than the faces: the
            # pull below cannot bring it in without moving its neighbours
            coarse = [p["miss"] for p in places if p["kind"] == "surface" and p["miss"] > 2.0 * bound]
            if dev["p99"] <= NEARLY * tol and (open_holes or coarse) and not faces:
                # put right in place (subdedit): the bridged hole is cut into
                # this cage, the faces too wide for the shape are refitted or
                # inset, and every other face stays as it is. Laying the
                # cage again finer moves every other place too
                from .subdedit import repair
                cage, H, dev, (d1, d2), left, log = repair(cage, H, mesh, data, check, tol, bound, min_hole, level,
                                                           dev, d1, d2, progress=progress)
                edits += log
                places, _ = misses(H, cage.V, mesh, d1, d2, Xc, bound, min_hole)
                open_holes = [p for p in places if p.get("printable")] if left["holes"] else []
                # a miss the edits left under three times the bound goes to
                # the worst-point stage (pulls, and faces inset where a pull
                # stalls); only a larger one is worth a new cage
                coarse = [left["coarse"]] if left["coarse"] > 3.0 * bound else []
            if step or faces or not tol < dev["p99"] <= NEARLY * tol:
                break
            # nearly there: the closest points were still moving when the
            # fit's rounds ran out. Fitting on from where it stands costs
            # half a new cage and keeps this one's faces
            cage, H = fit_cage(cage, mesh, level=level, data=data, settle=tol / 50, H=H)
            dev, (d1, d2, Xc) = deviation(H, cage.V, mesh, data=check)
            if progress:
                progress(f"cage {len(cage.Q)} faces fitted on: p99 {dev['p99']:.4f} mm")
        tries.append({"faces": len(cage.Q), "irregularVertices": int((cage.valence != 4).sum()), "finerAt": len(spots),
                      "p99": dev["p99"], "max": dev["max"], "printableHolesClosed": len(open_holes),
                      "coarseMiss": max(coarse, default=0.0), "edits": edits})
        if progress:
            progress(f"cage {len(cage.Q)} faces: p99 {dev['p99']:.4f} mm, max {dev['max']:.4f} mm "
                     f"(tolerance {tol:.4f}), {len(open_holes)} printable holes closed")
        best = (cage, H, dev)
        if (dev["p99"] <= tol and not open_holes and not coarse) or len(cage.Q) >= max_faces or faces:
            break
        # bicubic error falls as h^4 on smooth parts, as h^2 where a feature is
        # narrower than a face; the last two tries say which rate this part
        # is on (faces go as h^-2), and the next cage aims a little under
        # the tolerance: a cage finer than needed has more star points, so
        # more patches, and every fit on it is slower
        rate = 2.5
        if len(tries) >= 2 and tries[-2]["p99"] > dev["p99"] > 0:
            rate = 2.0 * np.log(tries[-2]["p99"] / dev["p99"]) / np.log(len(cage.Q) / tries[-2]["faces"])
            rate = float(np.clip(rate, 2.5, 4.0))
        grow = (dev["p99"] / (0.9 * tol)) ** (2.0 / rate)
        # finer only where the shape is tight: a closed hole opens once four
        # or so edges go round it (edge under 0.7 of its diameter: pi d / 4);
        # a surface miss narrower than the faces falls as the edge^2. Only
        # once the part as a whole is within the tolerance: until then every
        # place misses and the density is the question. The rest keeps
        # its density (Instant Meshes' edge times 1/k round each place, at
        # most 3 a try, over three faces round it)
        # One such try: Instant Meshes lays the whole cage afresh round the
        # spots, so a second round of spots moves every other place too
        # (a hole kept open on one lay closes on the next); after it, the
        # cage gets finer everywhere instead.
        h = np.sqrt(area / len(cage.Q))
        for p in (places if dev["p99"] <= tol and not spots else []):
            if p.get("printable"):
                k = h / (0.7 * p["diameter"])
            elif p["kind"] == "surface" and p["miss"] > 2.0 * bound:
                k = np.sqrt(p["miss"] / (0.9 * bound))
            else:
                continue
            spots.append((np.asarray(p["at"], float), float(np.clip(k, 1.3, 3.0)), 3.0 * h))
        if dev["p99"] <= tol and (open_holes or coarse) and tries[-1]["finerAt"]:
            # the spots did not do it: finer everywhere, as for a hole or a
            # coarse miss without them
            if open_holes:
                grow = max(grow, (h / (0.7 * min(p["diameter"] for p in open_holes))) ** 2)
            if coarse:
                grow = max(grow, (max(coarse) / bound) ** (2.0 / rate))
            n = int(min(max_faces, len(cage.Q) * min(4.0, max(1.3, grow))))
        elif dev["p99"] > tol:
            n = int(min(max_faces, len(cage.Q) * min(4.0, max(1.3, grow))))
        else:
            # the next cage is laid afresh and its 99th percentile moves a
            # few percent with it: keep it aimed just under the tolerance
            n = int(min(max_faces, max(n, len(cage.Q) * grow)))
    cage, H, dev = best
    if not faces and dev["p99"] <= tol:
        cage, H, dev, cancelled = _fewer_stars(cage, H, mesh, data, check, tol, bound, min_hole, level, dev,
                                               progress)
    else:
        cancelled = {"moves": 0}
    # the smoothest surface the tolerance allows: raise the bending weight
    # while the deviation stays within it (a noisy input's ripples go first)
    fair, fairs = 0.0, []
    if dev["p99"] <= tol:
        V0 = cage.V.copy()
        f = 4e-3
        while f < 1e3:
            c2 = Cage(V0.copy(), cage.Q)
            c2, _ = fit_cage(c2, mesh, level=level, data=data, fair=f, iterations=6, H=H, settle=tol / 50)
            d2, _ = deviation(H, c2.V, mesh, data=check)
            fairs.append({"bending": f, "p99": d2["p99"], "max": d2["max"]})
            if progress:
                progress(f"bending {f:g}: p99 {d2['p99']:.4f} mm")
            if d2["p99"] > tol:
                break
            fair, V0, dev = f, c2.V.copy(), d2
            f *= 8
        cage.V = V0
    # the worst point: places that miss by more than `bound` pull harder
    # (Lawson's minimax reweighting, sharpened: each point's weight grows
    # with (miss / 0.75 aim)^6 a round, at most x17, once it is within a
    # quarter of the aim, holes aside), so the fit spends its freedom only where the
    # shape is tight, while the 99th percentile stays within the tolerance
    wP, wX = np.ones(H.M.shape[0]), np.ones(len(data[0]))
    dev, (d1, d2, Xc) = deviation(H, cage.V, mesh, data=check)
    # the patches are held within a third of the tolerance of the limit
    # surface (lay_out), so the limit aims that much under the bound
    aim = bound - tol / 3.0
    places, kind = misses(H, cage.V, mesh, d1, d2, Xc, aim, min_hole)
    pulls = insets = 0
    from scipy.spatial import cKDTree

    from .subdedit import inset, refit

    def over(places):
        return max((p["miss"] for p in places if p["kind"] == "surface"), default=0.0)
    while (kind == 1).any():
        stage = 0
        while (kind == 1).any() and stage < 4:
            j = cKDTree(H.M @ cage.V).query(data[0], workers=-1)[1]
            wP *= np.where(kind != 2, np.clip(d1 / (0.75 * aim), 1.0, 1.6) ** 6, 1.0)
            dX = np.linalg.norm(data[0] - _closest_limit(H, cage.V, data[0]), axis=1)
            wX *= np.where(kind[j] != 2, np.clip(dX / (0.75 * aim), 1.0, 1.6) ** 6, 1.0)
            # only the control points round the places that miss are fitted
            # again (subdedit.refit): the rest of the surface is where the
            # fairing left it, and a pull costs seconds, not a whole fit
            c2 = Cage(cage.V.copy(), cage.Q)
            refit(c2, H, mesh, np.unique(cage.Q[H.root[(kind == 1)[H.fine.Q].any(axis=1)]]), data, fair=fair,
                  iterations=8, settle=tol / 50, weights=(wP, wX))
            d2v, (e1, e2, Xc) = deviation(H, c2.V, mesh, data=check)
            if d2v["p99"] > tol:
                break
            was = over(places)
            cage.V, dev, pulls, stage, d1, d2 = c2.V, d2v, pulls + 1, stage + 1, e1, e2
            places, kind = misses(H, cage.V, mesh, d1, d2, Xc, aim, min_hole)
            if progress:
                progress(f"worst-point pull {pulls}: max {dev['max']:.4f} mm, "
                         f"{int(sum(p['kind'] == 'surface' for p in places))} places over {aim:.3f}")
            if over(places) > 0.9 * was:
                break                   # the pull has stopped gaining: the faces are too wide there
        if not (kind == 1).any() or insets >= 3:
            break
        # what pulling cannot bring in gets finer faces, there only: a ring
        # of quads inside the faces that still miss (subdedit.inset), the
        # control points round it fitted again
        res = inset(cage.V, cage.Q, np.unique(H.root[(kind == 1)[H.fine.Q].any(axis=1)]))
        if res is None:
            break
        try:
            c2 = Cage(res[0], res[1])
            H2 = Hierarchy(c2, level)
            refit(c2, H2, mesh, res[2], data, fair=fair, settle=tol / 50)
        except (ValueError, RuntimeError):
            break
        d2v, (e1, e2, Xc) = deviation(H2, c2.V, mesh, data=check)
        if d2v["p99"] > tol or len(_folds(H2, c2.V)):
            break
        cage, H, dev, d1, d2, insets = c2, H2, d2v, e1, e2, insets + 1
        wP, wX = np.ones(H.M.shape[0]), np.ones(len(data[0]))
        places, kind = misses(H, cage.V, mesh, d1, d2, Xc, aim, min_hole)
        if progress:
            progress(f"faces inset where the pull stalled: {len(cage.Q)} faces, max {dev['max']:.4f} mm, "
                     f"{int(sum(p['kind'] == 'surface' for p in places))} places over {aim:.3f}")
    P = H.M @ cage.V
    surface = [p for p in places if p["kind"] == "surface"]
    holes = [p for p in places if p["kind"] == "hole"]
    worst = {"bound": round(bound, 4), "aim": round(aim, 4), "minHole": min_hole, "pulls": pulls, "insets": insets,
             "surfaceMax": round(float(max(d1[kind != 2].max(initial=0.0),
                                           d2[kind[cKDTree(P).query(Xc, workers=-1)[1]] != 2].max(initial=0.0))), 4),
             "surfacePlaces": surface[:10], "holesClosed": holes[:20],
             "printableHolesClosed": int(sum(p["printable"] for p in holes)),
             "eulerNumber": {"mesh": int(mesh.euler_number), "cage": int(cage.n - len(cage.E) + len(cage.Q))}}
    worst["within"] = bool(not surface and not worst["printableHolesClosed"])
    rep = {"bending": fair, "fairing": fairs, "folds": int(len(_folds(H, cage.V))), "tolerance": round(tol, 4),
           "noise": round(nz, 5), "tolerancePicked": why, "cageFaces": len(cage.Q),
           "irregularVertices": int((cage.valence != 4).sum()),
           "deviation": dev, "within": bool(dev["p99"] <= tol), "worst": worst, "starPointsCancelled": cancelled,
           "tries": tries, "seconds": round(time.time() - t0, 1)}
    return cage, H, rep


# ------------------------------------------------------------------ patches

def _bern(d, t):
    from math import comb
    t = np.asarray(t, float)[:, None]
    k = np.arange(d + 1)[None]
    c = np.array([comb(d, i) for i in range(d + 1)], float)[None]
    return c * t ** k * (1 - t) ** (d - k)


def _twins(cage):
    """For every face side (q, k): the other face's side on that edge."""
    m = len(cage.Q)
    flat = cage.QE.ravel()
    order = np.argsort(flat, kind="stable")
    pair = order.reshape(-1, 2)
    tw = np.empty(4 * m, np.int64)
    tw[pair[:, 0]] = pair[:, 1]
    tw[pair[:, 1]] = pair[:, 0]
    return tw // 4, tw % 4


def _side(A, k, s):
    """Side k of grids A (m, s+1, s+1, ...) from corner k to corner k+1."""
    return [A[:, :, 0], A[:, s, :], A[:, ::-1, s], A[:, 0, ::-1]][k]


def _inner(A, k, s):
    """The grid row just inside side k, in the side's direction."""
    return [A[:, :, 1], A[:, s - 1, :], A[:, ::-1, s - 1], A[:, 1, ::-1]][k]


def _ring_normals(cage, C):
    """Exact limit normals at the cage's vertices (Catmull-Clark tangent
    masks, Halstead et al. 1993, over each vertex's ordered ring)."""
    Q = cage.Q
    m = len(Q)
    tq, tk = _twins(cage)
    N = np.zeros((cage.n, 3))
    done = np.zeros(cage.n, bool)
    for q0 in range(m):
        for k0 in range(4):
            v = int(Q[q0, k0])
            if done[v]:
                continue
            done[v] = True
            ring_e, ring_f = [], []
            q, k = q0, k0
            for _ in range(64):
                ring_e.append(C[Q[q, (k + 1) % 4]])
                ring_f.append(C[Q[q, (k + 2) % 4]])
                # previous side of q ends at v; its twin starts at v
                kp = (k - 1) % 4
                q, k = int(tq[4 * q + kp]), int(tk[4 * q + kp])
                if q == q0 and k == k0:
                    break
            n = len(ring_e)
            e, f = np.asarray(ring_e), np.asarray(ring_f)
            j = np.arange(n)
            cn = np.cos(2 * np.pi / n)
            A = 1 + cn + np.cos(np.pi / n) * np.sqrt(2 * (9 + cn))
            c0 = np.cos(2 * np.pi * j / n)
            c1 = np.cos(2 * np.pi * (j + 1) / n)
            t1 = (A * c0) @ e + (c0 + c1) @ f
            s0 = np.sin(2 * np.pi * j / n)
            s1 = np.sin(2 * np.pi * (j + 1) / n)
            t2 = (A * s0) @ e + (s0 + s1) @ f
            nv = np.cross(t1, t2)
            N[v] = nv / max(np.linalg.norm(nv), 1e-300)
    return N


def grid_normals(H, C):
    """Exact limit-surface normals at every sample of every face's grid
    (m, s+1, s+1, 3): away from the cage's vertices every fine vertex has
    four edges, and its limit tangents are the bicubic B-spline's on its
    3 x 3 neighbours (rows across a cage edge come from the face on the
    other side); at the cage's vertices, the general tangent masks."""
    cage = H.cage
    G = H.G
    m, s1, _ = G.shape
    s = s1 - 1
    X = (H.Pm @ C)[G]                                   # fine control points on the grids
    E = np.zeros((m, s1 + 2, s1 + 2, 3))
    E[:, 1:-1, 1:-1] = X
    tq, tk = _twins(cage)
    for k in range(4):
        out = np.empty((m, s1, 3))
        for k2 in range(4):
            sel = tk.reshape(m, 4)[:, k] == k2
            if sel.any():
                qq = tq.reshape(m, 4)[sel, k]
                out[sel] = _inner(X[qq], k2, s)[:, ::-1]
        if k == 0:
            E[:, 1:-1, 0] = out
        elif k == 1:
            E[:, -1, 1:-1] = out
        elif k == 2:
            E[:, 1:-1, -1] = out[:, ::-1]
        else:
            E[:, 0, 1:-1] = out[:, ::-1]
    w = np.array([1.0, 4.0, 1.0]) / 6.0
    Tu = np.zeros((m, s1, s1, 3))
    Tv = np.zeros((m, s1, s1, 3))
    for b in range(3):
        Tu += w[b] * (E[:, 2:, b:b + s1] - E[:, :-2, b:b + s1]) / 2
        Tv += w[b] * (E[:, b:b + s1, 2:] - E[:, b:b + s1, :-2]) / 2
    N = np.cross(Tu, Tv)
    N /= np.maximum(np.linalg.norm(N, axis=-1, keepdims=True), 1e-300)
    Nv = _ring_normals(cage, C)
    for k, (i, j) in enumerate([(0, 0), (s, 0), (s, s), (0, s)]):
        nv = Nv[cage.Q[:, k]]
        # the masks' sign follows the ring's turning; the grid's next sample says which way is out
        ref = N[:, 1 if i == 0 else s - 1, 1 if j == 0 else s - 1]
        nv = nv * np.sign(np.einsum("ij,ij->i", nv, ref))[:, None]
        N[:, i, j] = nv
    return N


def _elevate(c, to):
    """Bezier control points (d+1, ...) raised to degree `to`."""
    c = np.asarray(c, float)
    while len(c) - 1 < to:
        d = len(c) - 1
        n = np.empty((d + 2,) + c.shape[1:])
        n[0], n[-1] = c[0], c[-1]
        for i in range(1, d + 1):
            a = i / (d + 1)
            n[i] = a * c[i - 1] + (1 - a) * c[i]
        c = n
    return c


KINK = 1.0           # degrees: rectangles along the lines meeting at more than this fall back to face patches
TANGENT = 3.0        # weight of "lies in the limit's tangent plane" rows against position rows (mm per unit)
IRREGULAR_DEGREE = 7
REFINE = 0           # times a face at an extraordinary vertex is split before its corner is fitted


def cells(H, refine=REFINE):
    """The patch layout: every cage face whole where its corners are all
    regular, split `refine` times towards each extraordinary corner
    otherwise (its other quarters are exact bicubics). Returns a list of
    (face, i0, j0, size) in the face's sample grid, and for each whether
    it touches an extraordinary vertex."""
    cage = H.cage
    s = H.G.shape[1] - 1
    if s % (2 ** refine):
        raise ValueError("sample grid too coarse for the refinement")
    ev = cage.valence[cage.Q] != 4                         # (m, 4) corner is extraordinary
    cpos = [(0, 0), (s, 0), (s, s), (0, s)]
    out = []

    def touches(q, i0, j0, h):
        for k, (ci, cj) in enumerate(cpos):
            if ev[q, k] and ci in (i0, i0 + h) and cj in (j0, j0 + h):
                return True
        return False

    def walk(q, i0, j0, h, depth):
        t = touches(q, i0, j0, h)
        if t and depth < refine:
            g = h // 2
            for di, dj in ((0, 0), (g, 0), (g, g), (0, g)):
                walk(q, i0 + di, j0 + dj, g, depth + 1)
        else:
            out.append((q, i0, j0, h, t))

    for q in range(len(cage.Q)):
        walk(q, 0, 0, s, 0)
    return out


def patches(H, C, degree=3, irregular=IRREGULAR_DEGREE, tangent=TANGENT, refine=REFINE):
    """Bezier patches over the cage, a list of (d+1, d+1, 3) control point
    arrays (u along a face's corner 0 -> 1, v along 0 -> 3), and their
    layout (`cells`).

    Away from extraordinary vertices the limit surface is exactly bicubic:
    those patches are the limit surface itself and meet their neighbours
    with C2 continuity. A face at an extraordinary vertex is split
    `refine` times towards it; the quarters away from it are exact
    bicubics again, and the small corner piece (degree `irregular`) is
    fitted to the exact limit points on it *and* to the exact limit tangent
    planes, so the patches round an extraordinary vertex share their edge
    curves exactly and lean on the same tangent plane: G1 to within the
    angle `patch_quality` reports."""
    cage = H.cage
    G = H.G
    m, s1, _ = G.shape
    s = s1 - 1
    XG = (H.M @ C)[G]
    lay = cells(H, refine)
    NG = grid_normals(H, C) if any(c[4] for c in lay) else None
    evv = np.zeros(len(H.fine.V), bool)
    evv[G[:, [0, s, s, 0], [0, 0, s, s]][cage.valence[cage.Q] != 4]] = True

    def side_idx(i0, j0, h, k):
        r = np.arange(h + 1)
        if k == 0:
            return i0 + r, np.full(h + 1, j0)
        if k == 1:
            return np.full(h + 1, i0 + h), j0 + r
        if k == 2:
            return i0 + h - r, np.full(h + 1, j0 + h)
        return np.full(h + 1, i0), j0 + h - r

    curves = {}
    tq_, _ = _twins(cage)
    whole_irr = np.zeros(m, bool)              # faces at a star point left whole (refine 0)
    for q, i0, j0, h, irr in lay:
        if irr and h == s:
            whole_irr[q] = True

    def curve(q, i0, j0, h, k):
        ii, jj = side_idx(i0, j0, h, k)
        ids = G[q, ii, jj]
        a, b = int(ids[0]), int(ids[-1])
        key = (min(a, b), max(a, b))
        if key not in curves:
            P = XG[q, ii, jj]
            t = np.arange(h + 1) / h
            fitted = evv[a] or evv[b]
            if h == s and whole_irr[q]:
                fitted = fitted or bool(whole_irr[int(tq_[4 * q + k])])
            if not fitted:
                d = degree
                B = _bern(d, t)
                ends = np.outer(B[:, 0], P[0]) + np.outer(B[:, d], P[-1])
                inner = np.linalg.lstsq(B[:, 1:d], P - ends, rcond=None)[0]
            else:
                d = irregular
                B, dB = _bern(d, t), _dbern(d, t)
                Nn = NG[q, ii, jj]
                ends = np.outer(B[:, 0], P[0]) + np.outer(B[:, d], P[-1])
                dends = np.outer(dB[:, 0], P[0]) + np.outer(dB[:, d], P[-1])
                ni = d - 1
                rows, rhs = [], []
                for kk in range(3):
                    R = np.zeros((h + 1, 3 * ni))
                    R[:, kk::3] = B[:, 1:d]
                    rows.append(R)
                    rhs.append((P - ends)[:, kk])
                R = np.zeros((h + 1, 3 * ni))
                for kk in range(3):
                    R[:, kk::3] = dB[:, 1:d] * Nn[:, kk:kk + 1]
                rows.append(tangent * R)
                rhs.append(-tangent * np.einsum("ij,ij->i", Nn, dends))
                inner = np.linalg.lstsq(np.vstack(rows), np.concatenate(rhs), rcond=None)[0].reshape(ni, 3)
            curves[key] = (a, np.vstack([P[0], inner, P[-1]]))
        a0, cv = curves[key]
        return cv if a0 == int(ids[0]) else cv[::-1]

    out = []
    for q, i0, j0, h, irr in lay:
        d = irregular if irr else degree
        t = np.arange(h + 1) / h
        b = np.zeros((d + 1, d + 1, 3))
        for k in range(4):
            cv = _elevate(curve(q, i0, j0, h, k), d)
            if k == 0:
                b[:, 0] = cv
            elif k == 1:
                b[d, :] = cv
            elif k == 2:
                b[::-1, d] = cv
            else:
                b[0, ::-1] = cv
        B, dB = _bern(d, t), _dbern(d, t)
        n1 = h + 1
        BB = np.einsum("ia,jb->ijab", B, B).reshape(n1 * n1, (d + 1) ** 2)
        border = np.zeros((d + 1, d + 1), bool)
        border[0, :] = border[d, :] = border[:, 0] = border[:, d] = True
        bi = border.ravel()
        known = b.reshape(-1, 3)[bi]
        Xc = XG[q, i0:i0 + n1, j0:j0 + n1].reshape(-1, 3)
        Y = Xc - BB[:, bi] @ known
        if not irr:
            b.reshape(-1, 3)[~bi] = np.linalg.lstsq(BB[:, ~bi], Y, rcond=None)[0]
        else:
            Du = np.einsum("ia,jb->ijab", dB, B).reshape(n1 * n1, -1)
            Dv = np.einsum("ia,jb->ijab", B, dB).reshape(n1 * n1, -1)
            Nq = NG[q, i0:i0 + n1, j0:j0 + n1].reshape(-1, 3)
            ni = int((~bi).sum())
            rows, rhs = [], []
            for kk in range(3):
                R = np.zeros((n1 * n1, 3 * ni))
                R[:, kk::3] = BB[:, ~bi]
                rows.append(R)
                rhs.append(Y[:, kk])
            onb = np.zeros((n1, n1))
            onb[0, :] = onb[-1, :] = onb[:, 0] = onb[:, -1] = 1
            wgt = tangent * np.where(onb.ravel() > 0, 1.0, 0.3)
            for D in (Du, Dv):
                R = np.zeros((n1 * n1, 3 * ni))
                for kk in range(3):
                    R[:, kk::3] = D[:, ~bi] * Nq[:, kk:kk + 1]
                rows.append(wgt[:, None] * R)
                rhs.append(-wgt * np.einsum("ij,ij->i", Nq, D[:, bi] @ known))
            x = np.linalg.lstsq(np.vstack(rows), np.concatenate(rhs), rcond=None)[0]
            b.reshape(-1, 3)[~bi] = x.reshape(ni, 3)
        out.append(b)
    return out, lay


def _patch_eval(b, u, v):
    d = b.shape[0] - 1
    Bu, Bv = _bern(d, u), _bern(d, v)
    return np.einsum("ia,jb,abk->ijk", Bu, Bv, b)


def _patch_normals(b, u, v):
    d = b.shape[0] - 1
    Bu, Bv = _bern(d, u), _bern(d, v)
    Su = np.einsum("ia,ib,abk->ik", _dbern(d, u), Bv, b)
    Sv = np.einsum("ia,ib,abk->ik", Bu, _dbern(d, v), b)
    n = np.cross(Su, Sv)
    return n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-300)


def patch_quality(H, C, bs, lay):
    """How far each patch is from the limit surface it stands for (mm, at
    the limit samples), and the angle between neighbouring patches'
    normals along every shared edge piece (degrees; 0 to rounding between
    exact bicubics)."""
    G = H.G
    XG = (H.M @ C)[G]
    err = []
    side = {}
    worst = []
    for b, (q, i0, j0, h, irr) in zip(bs, lay):
        t = np.arange(h + 1) / h
        gap = _patch_eval(b, t, t) - XG[q, i0:i0 + h + 1, j0:j0 + h + 1]
        err.append((irr, float(np.linalg.norm(gap, axis=2).max())))
        r = np.arange(h + 1)
        sides = [(i0 + r, np.full(h + 1, j0), t, 0 * t), (np.full(h + 1, i0 + h), j0 + r, 1 + 0 * t, t),
                 (i0 + h - r, np.full(h + 1, j0 + h), 1 - t, 1 + 0 * t), (np.full(h + 1, i0), j0 + h - r, 0 * t, 1 - t)]
        for ii, jj, u, v in sides:
            n = _patch_normals(b, u, v)
            for vid, nn in zip(G[q, ii, jj][1:-1], n[1:-1]):
                key = int(vid)
                if key in side:
                    worst.append(np.degrees(np.arccos(np.clip(nn @ side[key], -1, 1))))
                else:
                    side[key] = nn
    e = np.array([x[1] for x in err])
    irr = np.array([x[0] for x in err])
    w = np.array(worst) if worst else np.zeros(1)
    return {"patchToLimit": {"max": round(float(e.max()), 5),
                             "irregularMax": round(float(e[irr].max()), 5) if irr.any() else 0.0,
                             "regularMax": float(e[~irr].max()) if (~irr).any() else 0.0},
            "edgeAngle": {"max": round(float(w.max()), 4), "p99": round(float(np.percentile(w, 99)), 4)},
            "patches": len(bs), "cornerPatches": int(irr.sum())}


def _dbern(d, t):
    t = np.asarray(t, float)
    Bd = _bern(d - 1, t)
    out = np.zeros((len(t), d + 1))
    out[:, :d] -= d * Bd
    out[:, 1:] += d * Bd
    return out


def _bspline_basis(n_spans, t, d=3):
    """Clamped uniform B-spline basis of degree d over [0, n_spans] at t:
    (len(t), n_spans + d)."""
    kn = np.concatenate([np.zeros(d), np.arange(n_spans + 1, dtype=float), np.full(d, float(n_spans))])
    t = np.clip(np.asarray(t, float), 0, n_spans)
    nb = len(kn) - 1
    N = np.zeros((len(t), nb))
    for i in range(nb):
        N[:, i] = (kn[i] <= t) & (t < kn[i + 1])
    N[t >= n_spans, n_spans + d - 1] = 1.0
    N[t >= n_spans, :n_spans + d - 1] = 0.0
    for k in range(1, d + 1):
        M = np.zeros((len(t), nb - k))
        for i in range(nb - k):
            a = kn[i + k] - kn[i]
            b = kn[i + k + 1] - kn[i + 1]
            if a > 0:
                M[:, i] += (t - kn[i]) / a * N[:, i]
            if b > 0:
                M[:, i] += (kn[i + k + 1] - t) / b * N[:, i + 1]
        N = M
    return N


def _oriented(G, q, r):
    """Face q's sample grid turned so its corner r is (0, 0) and its side r
    runs along +i."""
    for k in range(4):
        A = np.rot90(G[q], k)
        s = A.shape[0] - 1
        corner = [G[q, 0, 0], G[q, s, 0], G[q, s, s], G[q, 0, s]][r]
        nxt = [G[q, s, 0], G[q, s, s], G[q, 0, s], G[q, 0, 0]][r]
        if A[0, 0] == corner and A[s, 0] == nxt:
            return k
    raise RuntimeError("face orientation")


def regions(cage, regular, limit=64):
    """Rectangles of regular cage faces, grown greedily: each a list of
    rows of (face, r), r the face's side that runs along the rectangle's
    +u. Inside one the limit surface is one bicubic B-spline."""
    tq, tk = _twins(cage)
    tq, tk = tq.reshape(-1, 4), tk.reshape(-1, 4)
    used = np.zeros(len(cage.Q), bool)

    def step(q, r, d):
        """Across local side d (0 -v, 1 +u, 2 +v, 3 -u): (face, its r)."""
        side = (d + r) % 4
        q2, k2 = int(tq[q, side]), int(tk[q, side])
        return q2, (k2 - (d + 2)) % 4

    out = []
    # seeds far from extraordinary vertices first: big rectangles grow there
    order = np.argsort(-regular.astype(int), kind="stable")
    for q0 in order:
        if used[q0] or not regular[q0]:
            continue
        R = [[(int(q0), 0)]]
        inside = {int(q0)}
        grown = True
        while grown:
            grown = False
            for d in (1, 2, 3, 0):
                if (d in (1, 3) and len(R[0]) >= limit) or (d in (0, 2) and len(R) >= limit):
                    continue
                if d == 1:
                    edge = [row[-1] for row in R]
                elif d == 3:
                    edge = [row[0] for row in R]
                elif d == 2:
                    edge = R[-1]
                else:
                    edge = R[0]
                new = [step(q, r, d) for q, r in edge]
                fs = [q for q, _ in new]
                if any(used[q] or not regular[q] or q in inside for q in fs) or len(set(fs)) != len(fs):
                    continue
                # never close a ring (a rectangle round a tube would meet itself)
                if any(step(q, r, d)[0] in inside for q, r in new):
                    continue
                # the new row/column must itself be a strip: neighbours along it
                along = 2 if d in (1, 3) else 1
                if any(step(*new[i], along) != new[i + 1] for i in range(len(new) - 1)):
                    continue
                if d == 1:
                    for row, x in zip(R, new):
                        row.append(x)
                elif d == 3:
                    for row, x in zip(R, new):
                        row.insert(0, x)
                elif d == 2:
                    R.append(new)
                else:
                    R.insert(0, new)
                inside |= set(fs)
                grown = True
        # R[j][i]: row j along +v, column i along +u
        for row in R:
            for q, _ in row:
                used[q] = True
        out.append(R)
    return out


def pack(H, C, bs, lay):
    """The patch list with every rectangle of whole regular faces merged
    into one clamped bicubic B-spline surface (exact: inside such a
    rectangle the limit surface is one C2 bicubic spline). Returns a list
    of surfaces: ("bezier", poles) or ("bspline", poles (nu, nv, 3), spans
    (a, b))."""
    cage = H.cage
    G = H.G
    s = G.shape[1] - 1
    XG = (H.M @ C)[G]
    whole = {}
    for k, (q, i0, j0, h, irr) in enumerate(lay):
        if h == s and not irr:
            whole[q] = k
    regular = np.zeros(len(cage.Q), bool)
    regular[list(whole)] = True
    out, taken = [], set()
    for R in regions(cage, regular):
        b, a = len(R), len(R[0])
        if a * b == 1:
            continue
        # the samples as one grid: inside the rectangle the limit surface is
        # one uniform bicubic spline, so the clamped one through them is
        # exact, and the fit separates into a solve along u and one along v
        X = np.empty((a * s + 1, b * s + 1, 3))
        for j, row in enumerate(R):
            for i, (q, r) in enumerate(row):
                X[i * s:i * s + s + 1, j * s:j * s + s + 1] = np.rot90(XG[q], _oriented(G, q, r))
        Bu = _bspline_basis(a, np.arange(a * s + 1) / s)
        Bv = _bspline_basis(b, np.arange(b * s + 1) / s)
        poles = np.einsum("ia,ijk,jb->abk", np.linalg.pinv(Bu).T, X, np.linalg.pinv(Bv).T, optimize=True)
        err = float(np.abs(np.einsum("ia,abk,jb->ijk", Bu, poles, Bv, optimize=True) - X).max())
        if err > 1e-6:
            continue                      # not one spline after all (a fold in the layout): keep its patches
        out.append(("bspline", poles, (a, b)))
        taken |= {whole[q] for row in R for q, _ in row}
    for k, bq in enumerate(bs):
        if k not in taken:
            out.append(("bezier", bq))
    return out


# ------------------------------------------------------------------ B-rep

def to_brep(b, sew=SEW):
    """The patches sewn into one closed OpenCascade solid (outward)."""
    from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeFace, BRepBuilderAPI_MakeSolid, BRepBuilderAPI_Sewing
    from OCP.Geom import Geom_BezierSurface
    from OCP.OCP.collections import Array2_gp_Pnt
    from OCP.TopAbs import TopAbs_SHELL
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopoDS import TopoDS

    from .brep import _forward
    from .step import _pnt

    from OCP.Geom import Geom_BSplineSurface
    from OCP.OCP.collections import Array1_double, Array1_int

    def knots(n):
        k = Array1_double(1, n + 1)
        m = Array1_int(1, n + 1)
        for i in range(n + 1):
            k.SetValue(i + 1, float(i))
            m.SetValue(i + 1, 4 if i in (0, n) else 1)
        return k, m

    def breaks(br):
        k = Array1_double(1, len(br))
        m = Array1_int(1, len(br))
        for i, x in enumerate(br):
            k.SetValue(i + 1, float(x))
            m.SetValue(i + 1, 4 if i in (0, len(br) - 1) else 1)
        return k, m

    sw = BRepBuilderAPI_Sewing(sew)
    for item in b:
        if isinstance(item, tuple) and item[0] == "nurbs":
            _, P, bu, bv = item
            arr = Array2_gp_Pnt(1, P.shape[0], 1, P.shape[1])
            for i in range(P.shape[0]):
                for j in range(P.shape[1]):
                    arr.SetValue(i + 1, j + 1, _pnt(P[i, j]))
            ku, mu = breaks(bu)
            kv, mv = breaks(bv)
            sw.Add(BRepBuilderAPI_MakeFace(Geom_BSplineSurface(arr, ku, kv, mu, mv, 3, 3), 1e-7).Face())
            continue
        if isinstance(item, tuple) and item[0] == "bspline":
            _, P, (na, nb) = item
            arr = Array2_gp_Pnt(1, P.shape[0], 1, P.shape[1])
            for i in range(P.shape[0]):
                for j in range(P.shape[1]):
                    arr.SetValue(i + 1, j + 1, _pnt(P[i, j]))
            ku, mu = knots(na)
            kv, mv = knots(nb)
            sw.Add(BRepBuilderAPI_MakeFace(Geom_BSplineSurface(arr, ku, kv, mu, mv, 3, 3), 1e-7).Face())
            continue
        bq = item[1] if isinstance(item, tuple) else item
        n = bq.shape[0]
        arr = Array2_gp_Pnt(1, n, 1, n)
        for i in range(n):
            for j in range(n):
                arr.SetValue(i + 1, j + 1, _pnt(bq[i, j]))
        sw.Add(BRepBuilderAPI_MakeFace(Geom_BezierSurface(arr), 1e-7).Face())
    sw.Perform()
    free = int(sw.NbFreeEdges())
    ex = TopExp_Explorer(sw.SewedShape(), TopAbs_SHELL)
    shells = []
    while ex.More():
        shells.append(TopoDS.Shell(ex.Current()))
        ex.Next()
    if len(shells) != 1:
        raise RuntimeError(f"sewing left {len(shells)} shells, {free} free edges")
    solid = BRepBuilderAPI_MakeSolid(shells[0]).Solid()
    return _forward(solid), free


def write_cage_obj(cage, path):
    with open(path, "w") as fh:
        fh.write("# loadpath subd cage: Catmull-Clark control mesh\n")
        for v in cage.V:
            fh.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for q in cage.Q:
            fh.write("f " + " ".join(str(int(i) + 1) for i in q) + "\n")


def export_surface(mesh, path, tolerance=None, faces=None, degree=3, progress=None, cage_path=None, packed=True,
                   layout="lines", **kw):
    """Mesh in, STEP of a few smooth patches out. Returns a report."""
    import os

    from OCP.BRepCheck import BRepCheck_Analyzer

    from .step import _volume, write_step

    t0 = time.time()
    cage, H, rep = surface_mesh(mesh, tolerance=tolerance, faces=faces, progress=progress, **kw)
    Hp = Hierarchy(Cage(cage.V, cage.Q), LEVEL + 1)
    b = None
    if layout == "lines":
        # rectangles along the cage's own lines, star points at their corners
        from .subdlines import lay_out
        b, q = lay_out(Hp, cage.V, rep["tolerance"], progress=progress)
        if b is not None and q["edgeAngle"]["max"] > KINK:
            b, q["layout"] = None, f"rectangles meet at up to {q['edgeAngle']['max']:.2f} deg"
        if b is not None:
            band = [x for x in b if x[0] == "band"]
            b = [x for x in b if x[0] != "band"]
            if band:
                bs, lay = patches(Hp, cage.V, degree, irregular=7, refine=0)
                keep = set(band[0][1].tolist())
                sel = [k for k, c in enumerate(lay) if c[0] in keep]
                b += pack(Hp, cage.V, [bs[k] for k in sel], [lay[k] for k in sel])
            q["patches"] = len(b)
            rep["patches"] = q
        elif progress:
            progress(f"lines layout: {q.get('layout')}; patches face by face")
    if b is None:
        # a star point's faces whole, as degree-7 patches; split once toward it
        # (quintic corners) where that misses the limit surface by more than a
        # third of the tolerance
        b, lay = patches(Hp, cage.V, degree, irregular=7, refine=0)
        q = patch_quality(Hp, cage.V, b, lay)
        if q["patchToLimit"]["max"] > rep["tolerance"] / 3:
            b, lay = patches(Hp, cage.V, degree, irregular=5, refine=1)
            q = patch_quality(Hp, cage.V, b, lay)
            q["starPointsSplit"] = True
        q["layout"] = "faces"
        rep["patches"] = q
        if packed:
            b = pack(Hp, cage.V, b, lay)
            rep["patches"]["packedSurfaces"] = int(sum(1 for x in b if x[0] == "bspline"))
    # the written patches sit within patchToLimit of the limit surface, so
    # their worst point away from closed holes is within this
    w = rep["worst"]
    w["patchesMax"] = round(w["surfaceMax"] + rep["patches"]["patchToLimit"]["max"], 4)
    w["within"] = bool(w["within"] and w["patchesMax"] <= w["bound"])
    shape, free = to_brep(b)
    valid = bool(BRepCheck_Analyzer(shape).IsValid())
    # points as OpenCascade writes them: rounding the poles of a degree-7
    # patch to 10 nm moves it off its own edges, and a reader cannot mesh it
    wrote = write_step(shape, path, verify=True, decimals=None) if valid and not free else None
    if (wrote is None or not wrote.get("readBackValid", True)) and rep["patches"].get("layout") == "lines":
        # rectangles that do not sew into a closed solid, or that a reader
        # does not get back as one: face by face
        why = f"sews with {free} free edges" if wrote is None else "does not read back as a valid solid"
        wrote = None
        if progress:
            progress(f"lines layout {why}; patches face by face")
        b, lay = patches(Hp, cage.V, degree, irregular=7, refine=0)
        q = patch_quality(Hp, cage.V, b, lay)
        if q["patchToLimit"]["max"] > rep["tolerance"] / 3:
            b, lay = patches(Hp, cage.V, degree, irregular=5, refine=1)
            q = patch_quality(Hp, cage.V, b, lay)
            q["starPointsSplit"] = True
        q["layout"] = f"faces (lines layout {why})"
        rep["patches"] = q
        if packed:
            b = pack(Hp, cage.V, b, lay)
        shape, free = to_brep(b)
        valid = bool(BRepCheck_Analyzer(shape).IsValid())
        w["patchesMax"] = round(w["surfaceMax"] + q["patchToLimit"]["max"], 4)
        w["within"] = bool(not w["surfacePlaces"] and not w["printableHolesClosed"]
                           and w["patchesMax"] <= w["bound"])
    if wrote is None:
        wrote = write_step(shape, path, verify=valid, decimals=None)
    rep.update({"path": str(path), "bytes": os.path.getsize(path), "stepFaces": int(len(b)), "freeEdges": free,
                "valid": bool(valid and wrote.get("readBackValid", valid)),
                "volume": round(_volume(shape), 3), "meshVolume": round(float(mesh.volume), 3),
                "seconds": round(time.time() - t0, 1)})
    if cage_path:
        write_cage_obj(cage, cage_path)
        rep["cage"] = str(cage_path)
    rep["shape"] = shape
    return rep


def tessellate(shape, deflection=0.01, angle=0.1):
    """The B-rep finely triangulated: (trimesh, face index of each
    triangle), for renders and checks that need to see the patches."""
    import trimesh
    from OCP.BRep import BRep_Tool
    from OCP.BRepMesh import BRepMesh_IncrementalMesh
    from OCP.TopAbs import TopAbs_FACE, TopAbs_REVERSED
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopLoc import TopLoc_Location
    from OCP.TopoDS import TopoDS

    BRepMesh_IncrementalMesh(shape, deflection, False, angle, True)
    V, F, I = [], [], []
    off = 0
    ex = TopExp_Explorer(shape, TopAbs_FACE)
    k = 0
    while ex.More():
        f = TopoDS.Face(ex.Current())
        ex.Next()
        loc = TopLoc_Location()
        tri = BRep_Tool.Triangulation_s(f, loc)
        if tri is None:
            k += 1
            continue
        tr = loc.Transformation()
        nn, nt = tri.NbNodes(), tri.NbTriangles()
        v = np.array([[p.X(), p.Y(), p.Z()] for p in (tri.Node(i + 1).Transformed(tr) for i in range(nn))])
        t = np.array([tri.Triangle(i + 1).Get() for i in range(nt)], np.int64) - 1
        if f.Orientation() == TopAbs_REVERSED:
            t = t[:, [0, 2, 1]]
        V.append(v)
        F.append(t + off)
        I.append(np.full(nt, k))
        off += nn
        k += 1
    m = trimesh.Trimesh(np.concatenate(V), np.concatenate(F), process=False)
    return m, np.concatenate(I)


def surface_file(source, out, size=None, scale=None, up=None, tolerance=None, accuracy=None, faces=None,
                 progress=None):
    """`loadpath surface`: any mesh in (repaired by exterior.load_exterior
    when it is open, self-intersecting or carries debris), `out` (.step),
    `out` .cage.obj and .json beside it. Returns the report (no shape)."""
    import json
    import os

    import trimesh

    from .exterior import load_exterior

    ext = load_exterior(source, size=size, scale=scale, up=up)
    src = None
    try:
        raw = trimesh.load(source, force="mesh", process=False)
        raw.apply_transform(ext.transform)
        src = raw
    except Exception:            # an SDF or CAD source: no triangles of its own to read noise from
        pass
    stem = os.path.splitext(out)[0]
    rep = export_surface(ext.mesh, out, tolerance=tolerance, accuracy=accuracy, faces=faces, source=src,
                         cage_path=stem + ".cage.obj", progress=progress)
    rep.pop("shape", None)
    rep["intake"] = {"kind": ext.report.get("kind"), "wrapped": "wrap" in ext.report,
                     "warnings": ext.report.get("warnings", [])}
    rep["ok"] = bool(rep["valid"] and rep["freeEdges"] == 0)
    with open(stem + ".json", "w") as fh:
        json.dump(rep, fh, indent=1, default=float)
    return rep
