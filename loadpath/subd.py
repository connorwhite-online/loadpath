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
        key = np.stack([np.minimum(a, b), np.maximum(a, b)], 1)
        E, inv, cnt = np.unique(key, axis=0, return_inverse=True, return_counts=True)
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

def quad_cage(mesh, faces, crease=None, seed=0):
    """An all-quad cage over a closed triangle mesh with about `faces`
    quads (Instant Meshes, pure-quad mode), its vertices on the mesh."""
    import os
    import tempfile

    import pyinstantmeshes as im

    # Instant Meshes' pure-quad mode subdivides its quad-dominant mesh once,
    # so ask for a quarter of the faces; the in-memory call hands back
    # triangles, so the quads come through an OBJ
    kw = dict(target_face_count=max(8, int(faces) // 4), pure_quad=True, deterministic=True,
              smooth_iterations=2, crease_angle=-1.0 if crease is None else float(crease))
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
    v = np.asarray(v, float)
    f = np.asarray(f, np.int64)
    f = f[(f[:, 0] != f[:, 1]) & (f[:, 1] != f[:, 2]) & (f[:, 2] != f[:, 3]) & (f[:, 3] != f[:, 0])]
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
    v, f = _close_holes(v, f)
    v, f = _split_pinches(v, f)
    v, f = _dissolve_doublets(v, f)
    return Cage(v, f)


def _close_holes(v, f):
    """Instant Meshes can leave a small hole where its extraction gives up;
    a hole with an even number of sides closes with quads round a new
    centre vertex. Returns (v, f); an odd hole raises (the caller retries
    at another density)."""
    n = len(v)
    a = f.ravel()
    b = np.roll(f, -1, axis=1).ravel()
    has = set((a * n + b).tolist())
    nxt = {int(y): int(x) for x, y in zip(a, b) if int(y) * n + int(x) not in has}
    if not nxt:
        return v, f
    new_v, new_f, seen = [], [], set()
    for s0 in list(nxt):
        if s0 in seen:
            continue
        loop, c = [s0], nxt[s0]
        seen.add(s0)
        while c != s0 and c not in seen and c in nxt and len(loop) < 64:
            loop.append(c)
            seen.add(c)
            c = nxt[c]
        if c != s0 or len(loop) % 2:
            raise ValueError(f"quad cage has a hole of {len(loop)} sides")
        L = len(loop)
        if L == 4:
            new_f.append(loop)
            continue
        ci = n + len(new_v)
        new_v.append(v[loop].mean(axis=0))
        for i in range(0, L, 2):
            new_f.append([ci, loop[i], loop[(i + 1) % L], loop[(i + 2) % L]])
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


def fit_cage(cage, mesh, level=LEVEL, iterations=ITERATIONS, smooth=SMOOTH, edge=None, data=None, fair=0.0, H=None):
    """Move the cage's control points so its limit surface lies on `mesh`
    (two-way closest-point least squares, a membrane term on the cage).
    Returns (cage, hierarchy) with the fitted control points in cage.V."""
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
    # membrane: each control point near the mean of its neighbours
    E = cage.E
    W = sp.csr_matrix((np.ones(2 * len(E)), (np.r_[E[:, 0], E[:, 1]], np.r_[E[:, 1], E[:, 0]])), shape=(n, n))
    Lap = sp.identity(n) - sp.diags(1.0 / cage.valence) @ W
    C = cage.V.copy()
    lam = smooth * M.shape[0] / n
    stiff = np.ones(n)                      # local membrane factor, raised where the surface folds
    for rnd in range(4):
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
            j = cKDTree(P).query(X)[1]
            Nj = NP[j]
            ok2 = np.einsum("ij,ij->i", NX, Nj) > 0.3
            t2 = P[j] + (np.einsum("ij,ij->i", X - P[j], Nj) * ok2)[:, None] * Nj
            w2 = 0.5 * len(P) / len(X)
            R = sp.diags(np.sqrt(lam * stiff)) @ Lap
            # bending (thin plate on the cage): what takes the noise out
            Bd = np.sqrt(fair * M.shape[0] / n) * (Lap @ Lap)
            A = sp.vstack([M, np.sqrt(w2) * M[j], R, Bd]).tocsr()
            rhs = np.vstack([t1, np.sqrt(w2) * t2, np.zeros((2 * n, 3))])
            solve = factorized((A.T @ A).tocsc())
            AtB = A.T @ rhs
            C = np.stack([solve(AtB[:, k]) for k in range(3)], axis=1)
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


ACCURACY = 0.3      # mm: FDM's typical +- (dfm_reference: PLA/PETG/ABS/nylon), loadpath's first process


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


def surface_mesh(mesh, tolerance=None, faces=None, max_faces=12000, rounds=5, level=LEVEL, progress=None,
                 crease=None, accuracy=None, source=None):
    """Fit a Catmull-Clark surface to a closed mesh. The cage gets denser
    until the two-way deviation's 99th percentile is within `tolerance` mm
    (default `pick_tolerance`: from the process `accuracy` and the noise
    of `source`, the mesh as it came in before any repair, else `mesh`).
    Returns (cage, hierarchy, report)."""
    t0 = time.time()
    ext = float(np.linalg.norm(mesh.bounds[1] - mesh.bounds[0]))
    nz = noise(source if source is not None else mesh)
    if tolerance is not None:
        tol, why = float(tolerance), "given"
    else:
        tol, why = pick_tolerance(accuracy, nz)
    area = float(mesh.area)
    n = int(faces) if faces else max(96, int(area / (ext / 12.0) ** 2))
    data = _samples(mesh, 300_000)
    check = _samples(mesh, 200_000, seed=1)
    tries = []
    best = None
    for r in range(rounds):
        cage = None
        for k in range(4):                    # an odd hole or a fold at this density: nudge it
            try:
                cage = quad_cage(mesh, int(n * (1 + 0.07 * k)), crease=crease)
                break
            except ValueError as err:
                if progress:
                    progress(f"quad cage at {n} faces: {err}; retrying")
        if cage is None:
            raise RuntimeError("no closed quad cage")
        cage, H = fit_cage(cage, mesh, level=level, data=data)
        dev, _ = deviation(H, cage.V, mesh, data=check)
        tries.append({"faces": len(cage.Q), "irregularVertices": int((cage.valence != 4).sum()),
                      "p99": dev["p99"], "max": dev["max"]})
        if progress:
            progress(f"cage {len(cage.Q)} faces: p99 {dev['p99']:.4f} mm, max {dev['max']:.4f} mm "
                     f"(tolerance {tol:.4f})")
        best = (cage, H, dev)
        if dev["p99"] <= tol or len(cage.Q) >= max_faces or faces:
            break
        # bicubic error falls as h^4 on smooth parts, as h^2 where a feature is narrower than a face:
        # step on the slower rate
        n = int(min(max_faces, len(cage.Q) * min(4.0, max(1.5, dev["p99"] / tol))))
    cage, H, dev = best
    # the smoothest surface the tolerance allows: raise the bending weight
    # while the deviation stays within it (a noisy input's ripples go first)
    fair, fairs = 0.0, []
    if dev["p99"] <= tol:
        V0 = cage.V.copy()
        f = 1e-3
        while f < 1e3:
            c2 = Cage(V0.copy(), cage.Q)
            c2, _ = fit_cage(c2, mesh, level=level, data=data, fair=f, iterations=6, H=H)
            d2, _ = deviation(H, c2.V, mesh, data=check)
            fairs.append({"bending": f, "p99": d2["p99"], "max": d2["max"]})
            if progress:
                progress(f"bending {f:g}: p99 {d2['p99']:.4f} mm")
            if d2["p99"] > tol:
                break
            fair, V0, dev = f, c2.V.copy(), d2
            f *= 4
        cage.V = V0
    rep = {"bending": fair, "fairing": fairs, "folds": int(len(_folds(H, cage.V))), "tolerance": round(tol, 4),
           "noise": round(nz, 5), "tolerancePicked": why, "cageFaces": len(cage.Q),
           "irregularVertices": int((cage.valence != 4).sum()),
           "deviation": dev, "within": bool(dev["p99"] <= tol), "tries": tries, "seconds": round(time.time() - t0, 1)}
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


TANGENT = 3.0        # weight of "lies in the limit's tangent plane" rows against position rows (mm per unit)
IRREGULAR_DEGREE = 5
REFINE = 1           # times a face at an extraordinary vertex is split before its corner is fitted


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

    def curve(q, i0, j0, h, k):
        ii, jj = side_idx(i0, j0, h, k)
        ids = G[q, ii, jj]
        a, b = int(ids[0]), int(ids[-1])
        key = (min(a, b), max(a, b))
        if key not in curves:
            P = XG[q, ii, jj]
            t = np.arange(h + 1) / h
            if not (evv[a] or evv[b]):
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

    sw = BRepBuilderAPI_Sewing(sew)
    for bq in b:
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


def export_surface(mesh, path, tolerance=None, faces=None, degree=3, progress=None, cage_path=None, **kw):
    """Mesh in, STEP of a few smooth patches out. Returns a report."""
    import os

    from OCP.BRepCheck import BRepCheck_Analyzer

    from .step import _volume, write_step

    t0 = time.time()
    cage, H, rep = surface_mesh(mesh, tolerance=tolerance, faces=faces, progress=progress, **kw)
    Hp = Hierarchy(Cage(cage.V, cage.Q), LEVEL + REFINE)
    b, lay = patches(Hp, cage.V, degree)
    rep["patches"] = patch_quality(Hp, cage.V, b, lay)
    shape, free = to_brep(b)
    valid = bool(BRepCheck_Analyzer(shape).IsValid())
    wrote = write_step(shape, path, verify=valid)
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
