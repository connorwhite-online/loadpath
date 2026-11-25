"""
A fitted quad cage edited in place (subd.py's density search).

Instant Meshes lays one edge length over the whole part and lays the whole
cage afresh every time it is asked: a through-hole narrower than its faces
is bridged on one lay and kept on the next, and a finer cage for one tight
place moves every other place. Here the cage that is already fitted is
changed only where it is wrong, and everything else keeps its faces, its
control points and its star points:

- `punch`: a through-hole the cage bridged (`holes`: the limit surface
  spans air over both of its mouths) is cut into it. The face over each
  mouth is removed and the two openings are joined by a tube of quads
  down the hole. A handle costs eight valence-5 vertices whatever the
  mesh (a closed quad mesh has sum(4 - valence) = 4 x Euler number, and a
  handle takes 2 off the Euler number): the eight corners of the two
  removed faces. The same edit lays a bridge of material across a slit
  (`slits`: two holes the remesher ran together, the walls between them
  inside the solid).
- `inset`: faces too wide for the shape under them (a miss the fit cannot
  pull in) get a ring of quads inside their outline. Any local refinement
  of a quad mesh ends its new lines in pairs of 3- and 5-valent vertices;
  a closed ring round the place is the cheapest (eight star points on a
  rectangle of faces, none round a tube).
- `refit`: the control points within a few rings of an edit are fitted
  again, the rest held where they are.
- `repair`: the three in order on a fitted cage, each edit kept only if
  the cage stays closed, unfolded and no worse.
"""
from __future__ import annotations

import numpy as np

RINGS = 4            # rings of faces round an edit whose control points are fitted again


def membranes(H, miss, d1):
    """Connected sets of the limit points marked in `miss` (bool per fine
    vertex), as arrays of fine vertices, deepest (largest d1) first."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    air = np.asarray(miss, bool)
    if not air.any():
        return []
    E = H.fine.E
    keep = air[E[:, 0]] & air[E[:, 1]]
    n = len(air)
    _, lab = connected_components(coo_matrix((np.ones(int(keep.sum())), (E[keep, 0], E[keep, 1])), shape=(n, n)),
                                  directed=False)
    idx = np.flatnonzero(air)
    order = np.argsort(lab[idx], kind="stable")
    idx = idx[order]
    cut = np.flatnonzero(np.diff(lab[idx])) + 1
    groups = np.split(idx, cut)
    groups.sort(key=lambda g: -float(d1[g].max()))
    return groups


def _apart(cage, fa, fb):
    """Whether two faces (or two patches of faces) can be joined by a
    tube: no corner in common and no edge from one to the other (the
    tube's own edges would double it)."""
    A, B = np.unique(cage.Q[fa]), np.unique(cage.Q[fb])
    if np.isin(A, B).any():
        return False
    E = cage.E
    return not ((np.isin(E[:, 0], A) & np.isin(E[:, 1], B)) | (np.isin(E[:, 0], B) & np.isin(E[:, 1], A))).any()


def _outline(Q, faces):
    """The outline of a patch of faces as its vertices in order (the way
    the faces' own sides run), or None when it is not one simple loop."""
    F = Q[np.atleast_1d(faces)]
    a, b = F.ravel(), np.roll(F, -1, axis=1).ravel()
    inner = set(zip(a.tolist(), b.tolist()))
    nxt = {}
    for x, y in inner:
        if (y, x) not in inner:
            if x in nxt:
                return None
            nxt[x] = y
    if len(nxt) < 4:
        return None
    v0 = next(iter(nxt))
    loop, v = [v0], nxt[v0]
    while v != v0:
        if v not in nxt or len(loop) > len(nxt):
            return None
        loop.append(v)
        v = nxt[v]
    return np.asarray(loop, np.int64) if len(loop) == len(nxt) else None


def _wall_faces(H, g, d1):
    """The cage faces a set of limit points lies on, those that miss most first."""
    fq = np.unique(H.root[np.isin(H.fine.Q, g).any(axis=1)])
    return sorted((int(q) for q in fq), key=lambda q: -float(d1[H.G[q]].mean()))


def _in_air(mesh, A, B, n=9):
    """For each pair of points (A[i], B[i]): whether the straight way
    between them stays outside the mesh (its middle 80%: the ends sit on
    the surface that spans the hole)."""
    from .subd import _signed_distance
    A, B = np.atleast_2d(A), np.atleast_2d(B)
    if not len(A):
        return np.zeros(0, bool)
    t = np.linspace(0.1, 0.9, n)[None, :, None]
    S = _signed_distance(mesh, (A[:, None] + t * (B - A)[:, None]).reshape(-1, 3))
    return (S.reshape(len(A), n) > 0).all(axis=1)


def _mouth_face(H, g, d1, taken):
    """The cage face over a hole's mouth: of the faces the membrane's
    deepest point lies on, the one whose limit points miss most."""
    v = int(g[np.argmax(d1[g])])
    fq = np.unique(H.root[(H.fine.Q == v).any(axis=1)])
    fq = [int(q) for q in fq if int(q) not in taken]
    if not fq:
        return None
    return max(fq, key=lambda q: float(d1[H.G[q]].mean()))


def holes(H, C, mesh, miss, d1):
    """Through-holes the cage bridged. `miss` marks the limit points off
    the mesh (by more than the bound); each connected set of them is a
    membrane when it is the mouth of a hole: another membrane faces it
    the other way with the straight way between their deepest points in
    air, or, with no such partner, the way along its own normal is in air
    until it leaves the cage. Returns one entry per hole: the cage faces
    over its mouths, its depth, its diameter (twice the deepest miss: the
    membrane sits a radius from the wall) and the limit points of its
    membranes."""
    import trimesh

    from .subd import _limit_normals
    P = H.M @ C
    NP = _limit_normals(H, P)
    groups = membranes(H, miss, d1)
    if not groups:
        return []
    deep = np.array([int(g[np.argmax(d1[g])]) for g in groups])
    dia = 2.0 * d1[deep]
    Pd, Nd = P[deep], NP[deep]
    k = len(groups)
    # candidate partners: facing the other way, nearest first
    # a hole bridged at both mouths is a few diameters deep at most: two
    # bridged crotches facing each other across the gap between members
    # also have air between them, tens of mm of it
    span = np.linalg.norm(Pd[:, None] - Pd[None], axis=2)
    ii, jj = np.nonzero((Nd @ Nd.T < -0.3) & (np.arange(k)[:, None] < np.arange(k)[None])
                        & (span <= 4.0 * np.maximum(dia[:, None], dia[None])))
    air = _in_air(mesh, Pd[ii], Pd[jj])
    ii, jj = ii[air], jj[air]
    w = np.linalg.norm(Pd[ii] - Pd[jj], axis=1)
    order = np.argsort(w)
    Qf = H.fine.Q
    out, used, taken = [], set(), set()
    Q = H.cage.Q

    def add(fa, fb, depth, d, pts):
        if fa is None or fb is None or fa in taken or fb in taken or not _apart(H.cage, fa, fb):
            fa = fb = None                    # mouths that touch: no room for a tube
        if fa is not None:
            taken.update((fa, fb))
        out.append({"faces": None if fa is None else (fa, fb), "depth": round(float(depth), 3),
                    "diameter": round(float(d), 3), "points": pts,
                    "at": np.round(P[pts[np.argmax(d1[pts])]], 2).tolist()})

    for o in order:
        i, j = int(ii[o]), int(jj[o])
        if i in used or j in used:
            continue
        used.update((i, j))
        fa = _mouth_face(H, groups[i], d1, taken)
        fb = _mouth_face(H, groups[j], d1, taken | {fa})
        add(fa, fb, w[o], max(dia[i], dia[j]), np.concatenate([groups[i], groups[j]]))
    rest = [i for i in range(k) if i not in used]
    if rest:
        # no flagged far side: straight through along the normal
        lm = trimesh.Trimesh(P, np.concatenate([Qf[:, [0, 1, 2]], Qf[:, [0, 2, 3]]]), process=False)
        loc, ray, tri = lm.ray.intersects_location(Pd[rest] - 1e-3 * Nd[rest], -Nd[rest], multiple_hits=False)
        if len(ray):
            air = _in_air(mesh, Pd[rest][ray], loc)
            for r, x, t, a in zip(ray, loc, tri, air):
                i = rest[int(r)]
                # a bridged crotch between two members also has air behind
                # it, tens of mm of it: a hole's far mouth is a few
                # diameters away at most (a deeper one is not bridged at
                # both ends and missed by the pairs above)
                if a and np.linalg.norm(x - Pd[i]) <= 4.0 * dia[i]:
                    add(_mouth_face(H, groups[i], d1, taken), int(H.root[int(t) % len(Qf)]),
                        np.linalg.norm(x - Pd[i]), dia[i], groups[i])
    return out


def punch(V, Q, fa, fb, mesh=None, layers=None):
    """Cut a handle (two faces back to back through the solid: a hole) or
    lay a bridge (two faces looking at each other across a slit: a strut
    of material; the same edit): `fa` and `fb` (a face each, or a patch
    of faces each, their outlines equally long) are removed and their
    openings joined by a tube of quads (`layers` rings of them; by
    default as many as keep the tube's quads about square). The tube's
    new vertices lie on the mesh's hole wall when `mesh` is given.
    Returns (V, Q, new faces, the vertices the cut touched); vertices
    inside a removed patch are left without a face (`_compact`)."""
    A, B = _outline(Q, fa), _outline(Q, fb)
    if A is None or B is None or len(A) != len(B):
        raise ValueError("patches with outlines that do not match")
    n4 = len(A)
    # corner a_i joins b_(s - i): the tube's quads then run the way the
    # removed faces did; the turn s with the shortest joins has no twist
    s = min(range(n4), key=lambda s: float(((V[A] - V[B[(s - np.arange(n4)) % n4]]) ** 2).sum()))
    Bp = B[(s - np.arange(n4)) % n4]
    side = float(np.mean([np.linalg.norm(V[R] - V[np.roll(R, -1)], axis=1).mean() for R in (A, B)]))
    depth = float(np.linalg.norm(V[A] - V[Bp], axis=1).mean())
    L = int(layers) if layers else int(np.clip(round(depth / max(side, 1e-12)), 1, 8))
    rings, new = [A], []
    n = len(V)
    for k in range(1, L):
        new.append(V[A] + (V[Bp] - V[A]) * (k / L))
        rings.append(np.arange(n + n4 * (k - 1), n + n4 * k))
    rings.append(Bp)
    if new:
        Vn = np.concatenate(new)
        if mesh is not None:
            from .subd import _closest, _edge_for
            Vn = _closest(mesh, Vn, _edge_for(mesh))[0]
        V = np.vstack([V, Vn])
    tube = [[u[i], u[(i + 1) % n4], w[(i + 1) % n4], w[i]] for u, w in zip(rings[:-1], rings[1:]) for i in range(n4)]
    keep = np.ones(len(Q), bool)
    keep[np.atleast_1d(fa)] = False
    keep[np.atleast_1d(fb)] = False
    Q2 = np.vstack([Q[keep], np.asarray(tube, np.int64)])
    return V, Q2, np.arange(int(keep.sum()), len(Q2)), np.unique(np.concatenate(rings))


def _compact(V, Q, touched):
    """Vertices no face uses dropped: (V, Q, touched) renumbered."""
    used = np.zeros(len(V), bool)
    used[Q] = True
    if used.all():
        return V, Q, touched
    remap = np.cumsum(used) - 1
    touched = np.asarray(touched, np.int64)
    return V[used], remap[Q], remap[touched[used[touched]]]


def slits(H, C, mesh, d1, bound):
    """Places where the cage is parted and the mesh is whole: Instant
    Meshes now and then runs two holes together through the bridge of
    material between them, and the two walls of that slit end up inside
    the solid, facing each other, where no closest point pulls them out
    (the mesh round them faces the other way). A connected set of limit
    points more than `bound` inside the mesh is such a wall when the way
    along its normal from its deepest point to the cage's next wall (ahead,
    or behind when the walls have passed through each other) stays inside
    the mesh. Returns one entry per slit: the cage faces of the wall and
    of the facing wall (those deepest in the solid first; the face looked
    at leads the facing ones), and the wall's limit points."""
    import trimesh

    from .subd import _limit_normals, _signed_distance
    idx = np.flatnonzero(d1 > bound)
    if not len(idx):
        return []
    P = H.M @ C
    inside = np.zeros(len(P), bool)
    inside[idx] = _signed_distance(mesh, P[idx]) < 0
    groups = membranes(H, inside, d1)
    if not groups:
        return []
    NP = _limit_normals(H, P)
    deep = np.array([int(g[np.argmax(d1[g])]) for g in groups])
    Qf = H.fine.Q
    lm = trimesh.Trimesh(P, np.concatenate([Qf[:, [0, 1, 2]], Qf[:, [0, 2, 3]]]), process=False)
    # the facing wall: ahead along the normal, or behind it when the two
    # walls have passed through each other
    best = {}
    t9 = np.linspace(0.0, 1.0, 9)[None, :, None]
    for sgn in (1.0, -1.0):
        loc, ray, tri = lm.ray.intersects_location(P[deep] + sgn * 1e-3 * NP[deep], sgn * NP[deep],
                                                   multiple_hits=False)
        if not len(ray):
            continue
        gap = np.linalg.norm(loc - P[deep][ray], axis=1)
        ok = (np.einsum("ij,ij->i", lm.face_normals[tri], NP[deep][ray]) < -0.3) & (gap <= 4.0 * d1[deep][ray])
        S = _signed_distance(mesh, (P[deep][ray][:, None] + t9 * (loc - P[deep][ray])[:, None]).reshape(-1, 3))
        ok &= (S.reshape(-1, 9) < 0).all(axis=1)
        for r, x, t, g in zip(ray[ok], loc[ok], tri[ok], gap[ok]):
            if int(r) not in best or g < best[int(r)][0]:
                best[int(r)] = (g, x, int(Qf[int(t) % len(Qf), 0]), int(H.root[int(t) % len(Qf)]))
    out, seen = [], np.zeros(len(P), bool)
    for r in sorted(best):
        _, x, hv, hf = best[r]
        i = int(deep[r])
        if seen[i]:
            continue                                   # the wall facing one already listed
        seen[groups[r]] = True
        far = [g for g in groups if hv in g]
        for g in far:
            seen[g] = True
        out.append({"faces": _wall_faces(H, groups[r], d1),
                    "facing": [hf] + [q for q in (_wall_faces(H, far[0], d1) if far else []) if q != hf],
                    "points": groups[r], "at": np.round(0.5 * (P[i] + x), 2).tolist()})
    return out


def inset(V, Q, faces, shrink=0.5, grow=4.0):
    """The flagged faces inset: a ring of quads is laid inside their
    outline (one per outline edge) and the faces themselves shrink into
    it. Faces outside keep their corners. A notch in the outline costs a
    star point each side of the ring, so notches are filled first (a face
    with two flagged neighbours joins them, as do the faces round a vertex
    the outline passes twice); filling stops at `grow` times the flagged
    faces. Returns (V, Q, the vertices it touched) or None when no clean
    outline comes out."""
    m = len(Q)
    R = np.zeros(m, bool)
    R[np.asarray(faces, np.int64)] = True
    n0 = int(R.sum())
    a = Q.ravel()
    b = np.roll(Q, -1, axis=1).ravel()
    nv = len(V)
    key = a * nv + b
    order = np.argsort(key)
    tw = order[np.searchsorted(key[order], b * nv + a)]            # twin half-edge of each
    tf = (tw // 4).reshape(m, 4)                                   # face across each side
    for _ in range(50):
        out = R[tf] & ~R[:, None]                                   # sides of unflagged faces on flagged ones
        add = (out.sum(axis=1) >= 2)
        bd = (R[:, None] & ~R[tf]).ravel()                          # outline half-edges
        cnt = np.bincount(a[bd], minlength=nv)
        pinch = cnt > 1
        if pinch.any():
            add |= pinch[Q].any(axis=1) & ~R
        if not add.any():
            break
        R |= add
        if R.sum() > grow * n0 + 8 or R.all():
            return None
    else:
        return None
    bd = np.flatnonzero((R[:, None] & ~R[tf]).ravel())
    if not len(bd):
        return None
    ov = np.unique(a[bd])
    new = np.full(nv, -1, np.int64)
    new[ov] = nv + np.arange(len(ov))
    # each outline vertex's copy: toward the middle of its own flagged faces
    cen = V[Q].mean(axis=1)
    acc = np.zeros((nv, 3))
    num = np.zeros(nv)
    fr = np.flatnonzero(R)
    np.add.at(acc, Q[fr].ravel(), np.repeat(cen[fr], 4, axis=0))
    np.add.at(num, Q[fr].ravel(), 1.0)
    Vn = V[ov] + shrink * (acc[ov] / num[ov, None] - V[ov])
    Q2 = Q.copy()
    Q2[fr] = np.where(new[Q[fr]] >= 0, new[Q[fr]], Q[fr])
    ring = np.stack([a[bd], b[bd], new[b[bd]], new[a[bd]]], axis=1)
    touched = np.unique(np.concatenate([Q[fr].ravel(), new[ov]]))
    return np.vstack([V, Vn]), np.vstack([Q2, ring]), touched


def coarse_faces(H, C, d1, d2, X, skip, bound):
    """Cage faces whose limit surface misses the mesh by more than `bound`
    (a limit point, or the limit point nearest a mesh sample that misses),
    limit points in `skip` (a hole's membranes) aside. Returns (faces,
    the largest such miss)."""
    from scipy.spatial import cKDTree
    e = np.where(skip, 0.0, d1)
    i2 = np.flatnonzero(d2 > bound)
    if len(i2):
        near = cKDTree(H.M @ C).query(X[i2], workers=-1)[1]
        np.maximum.at(e, near, np.where(skip[near], 0.0, d2[i2]))
    bad = e > bound
    if not bad.any():
        return np.zeros(0, np.int64), 0.0
    return np.unique(H.root[bad[H.fine.Q].any(axis=1)]), float(e.max())


def refit(cage, H, mesh, touched, data, rings=RINGS, iterations=None, smooth=None, edge=None, settle=None,
          fair=0.0, weights=None, stiff=None, rounds=4):
    """Fit again only the control points within `rings` faces of the
    vertices `touched` (subd.fit_cage's two-way closest-point least
    squares, on the limit points those control points move and the mesh
    samples nearest them, with fit_cage's bending weight `fair` and
    per-point `weights`); every other control point stays where it is.
    Where the fitted surface folds, the membrane round the fold is
    stiffened (x30) and only the control points round it are fitted again
    (`rounds` times at most; the rest has settled already). Changes cage.V
    in place; returns the control points that were free."""
    from scipy.sparse.linalg import factorized
    from scipy.spatial import cKDTree

    from . import subd
    sp = subd._sp()
    iterations = iterations or 2 * subd.ITERATIONS
    smooth = subd.SMOOTH if smooth is None else smooth
    M = H.M
    n = cage.n
    edge = edge or subd._edge_for(mesh)
    free = np.zeros(n, bool)
    free[np.asarray(touched, np.int64)] = True
    for _ in range(rings):
        free[cage.Q[free[cage.Q].any(axis=1)]] = True
    f = np.flatnonzero(free)
    x = np.flatnonzero(~free)
    rows = np.unique(M.tocsc()[:, f].indices)
    Mr = M[rows]
    Mf = Mr[:, f].tocsr()
    MfT = Mf.T.tocsr()
    C = cage.V.copy()
    X, NX = data
    P = M @ C
    near = np.zeros(len(P), bool)
    near[rows] = True
    sel = near[cKDTree(P).query(X, workers=-1)[1]]
    Xs, NXs = X[sel], NX[sel]
    wP, wX = (np.ones(len(rows)), np.ones(len(Xs))) if weights is None else (weights[0][rows], weights[1][sel])
    E = cage.E
    Lap = subd.cage_laplacian(cage)
    w2 = 0.5 * M.shape[0] / len(X)
    Qf = H.fine.Q
    # normals of the rows from the fine quads that hold them: only the limit
    # points those quads touch are evaluated (a refit is local; the whole
    # limit surface each round was most of its time)
    qs = np.flatnonzero(near[Qf].any(axis=1))
    idx, loc = np.unique(Qf[qs], return_inverse=True)
    loc = loc.reshape(-1, 4)
    Msub = M[idx]
    at_rows = np.searchsorted(idx, rows)
    lam = smooth * M.shape[0] / n
    LL = subd.cage_bending(cage, Lap)
    Bend = (fair * M.shape[0] / n) * (LL.T @ LL)
    MtM = (MfT @ Mf).tocsc()
    stiff = np.ones(n) if stiff is None else stiff      # raised where the surface folds, as fit_cage does

    def gather(j, w, T, size):
        return np.stack([np.bincount(j, weights=w * T[:, k], minlength=size) for k in range(3)], 1)
    for rnd in range(1):
        Reg = (Lap.T @ sp.diags(lam * stiff) @ Lap + Bend).tocsr()[f]
        Rff, Rfx = Reg[:, f].tocsc(), Reg[:, x].tocsr()
        pre = None
        for it in range(iterations):
            Ps = Msub @ C
            a = Ps[loc[:, 2]] - Ps[loc[:, 0]]
            b = Ps[loc[:, 3]] - Ps[loc[:, 1]]
            fn = np.cross(a, b)
            N = np.stack([np.bincount(loc.ravel(), weights=np.repeat(fn[:, k], 4), minlength=len(idx))
                          for k in range(3)], 1)
            Pr = Ps[at_rows]
            NP = N[at_rows] / np.maximum(np.linalg.norm(N[at_rows], axis=1, keepdims=True), 1e-300)
            q, nq = subd._closest(mesh, Pr, edge)
            ok1 = np.einsum("ij,ij->i", nq, NP) > 0.3
            t1 = Pr - np.where(ok1, np.einsum("ij,ij->i", Pr - q, nq), 0.0)[:, None] * nq
            j = cKDTree(Pr).query(Xs, workers=-1)[1]
            Nj = NP[j]
            ok2 = np.einsum("ij,ij->i", NXs, Nj) > 0.3
            t2 = Pr[j] + (np.einsum("ij,ij->i", Xs - Pr[j], Nj) * ok2)[:, None] * Nj
            cnt = wP + w2 * np.bincount(j, weights=wX, minlength=len(Pr))
            acc = gather(j, wX, t2, len(Pr))
            # the held control points' share of every row moves to the right-hand side
            held = Pr - Mf @ C[f]
            rhs = MfT @ (wP[:, None] * t1 + w2 * acc - cnt[:, None] * held) - Rfx @ C[x]
            if pre is None:
                # factor once a round at the mean pull, conjugate gradients from
                # there (fit_cage's scheme; the counts move a little each round)
                pre = factorized((MtM * float(cnt.mean()) + Rff).tocsc())

            def K(v, cnt=cnt):
                return MfT @ (cnt[:, None] * (Mf @ v)) + Rff @ v
            Cf = subd._pcg(K, pre, rhs, C[f])
            step = float(np.abs(Cf - C[f]).max())
            C[f] = Cf
            if settle is not None and it >= 2 and step < settle:
                break
    cage.V = C
    folds = subd._folds(H, C)
    if len(folds) and rounds > 1:
        verts = np.unique(cage.Q[folds])
        ring = np.unique(cage.Q[(np.isin(cage.Q, verts)).any(1)])
        stiff = stiff.copy()
        stiff[ring] *= 30.0
        f2 = refit(cage, H, mesh, verts, data, rings=rings, iterations=iterations, smooth=smooth, edge=edge,
                   settle=settle, fair=fair, weights=weights, stiff=stiff, rounds=rounds - 1)
        f = np.union1d(f, f2)
    return f


def _fold_spots(H, C):
    """Where the limit surface folds (subd._folds): one point per folded
    cage face, the mean of its limit points."""
    from . import subd

    f = subd._folds(H, C)
    if not len(f):
        return np.zeros((0, 3))
    P = H.M @ C
    sel = np.isin(H.root, f)
    Qf = H.fine.Q[sel]
    r = H.root[sel]
    c = P[Qf].mean(axis=1)
    return np.stack([np.bincount(np.searchsorted(f, r), weights=c[:, k], minlength=len(f)) for k in range(3)], 1) \
        / np.bincount(np.searchsorted(f, r), minlength=len(f))[:, None]


def _new_folds(H2, C2, H, C):
    """Folds of the edited surface that the cage did not have before
    (farther than a face from every fold it had): a fit can leave a few
    folds on a part (fit_cage stiffens round them and stops), and an edit
    elsewhere is not to blame for them."""
    from scipy.spatial import cKDTree

    new = _fold_spots(H2, C2)
    if not len(new):
        return 0
    old = _fold_spots(H, C)
    if not len(old):
        return len(new)
    h = float(np.sqrt(np.median(np.linalg.norm(C[H.cage.Q[:, 2]] - C[H.cage.Q[:, 0]], axis=1) ** 2) / 2.0))
    return int((cKDTree(old).query(new)[0] > 2.0 * h).sum())


def repair(cage, H, mesh, data, check, tol, bound, min_hole, level, dev, d1, d2, rounds=6, progress=None):
    """The fitted cage put right where it is wrong, in place: bridged
    holes the process could print are cut in (`punch`), faces that miss by
    more than twice the bound are fitted again on their own and, if that
    does not bring them in, inset (`inset`); after each change only the
    control points round it are fitted again (`refit`). A change that
    folds the surface or leaves the cage open is dropped.

    Returns (cage, H, dev, (d1, d2), left, log): `left` is what the edits
    could not put right ({"holes": printable holes still bridged,
    "coarse": the largest miss over twice the bound, 0 with none}), `log`
    one entry per round."""
    from . import subd

    X = check[0]
    log = []
    refitted = bridged = nocut = False

    def state(cage, H, d1, d2):
        sl = slits(H, cage.V, mesh, d1, bound)
        hs = holes(H, cage.V, mesh, d1 > bound, d1)
        skip = np.zeros(len(d1), bool)
        for h in hs:
            skip[h["points"]] = True
        # a crevice narrower than the process prints open (subd.misses
        # "gap") is closed as the printer closes it: no finer faces for it
        skip |= subd.misses(H, cage.V, mesh, d1, d2, X, bound, min_hole, through=False)[1] == 2
        cf, cm = coarse_faces(H, cage.V, d1, d2, X, skip, 2.0 * bound)
        return sl, [h for h in hs if h["diameter"] >= min_hole], cf, cm

    sl, todo, cf, cm = state(cage, H, d1, d2)
    for r in range(rounds):
        cut = [] if nocut else [h for h in todo if h["faces"] is not None]
        if not sl and not cut and not len(cf):
            break
        V, Q = cage.V.copy(), cage.Q.copy()
        entry = {"slitsBridged": 0, "holesCut": 0, "facesInset": 0, "refitOnly": 0}
        touched = []
        gone = np.zeros(len(Q), bool)
        bridges = []
        if sl and not bridged:
            # a bridge of material across each slit, where its walls face
            for x in sl:
                # as wide as the slit: the walls' faces, deepest first, as
                # many from each as give two outlines of one length; else
                # one face from each
                fA, fB = x["faces"], x["facing"]
                tries = [(fA[:k], fB[:k]) for k in range(min(len(fA), len(fB)), 1, -1)]
                tries += [([fa], [fb]) for fa in fA for fb in fB]
                for pa, pb in tries:
                    if gone[pa].any() or gone[pb].any() or not _apart(cage, pa, pb):
                        continue
                    la, lb = _outline(Q, pa), _outline(Q, pb)
                    if la is None or lb is None or len(la) != len(lb):
                        continue
                    Vn, Qn, newf, t = punch(V, Q, pa, pb, layers=1)
                    bridges.append(Qn[newf])
                    gone[pa] = gone[pb] = True
                    touched.append(t)
                    break
        if sl:
            bridged = True                            # one go: a slit that does not bridge is left to the faces
        if bridges:
            Q = np.vstack([Q[~gone]] + bridges)
            entry["slitsBridged"] = len(bridges)
        elif cut:
            # removing faces renumbers the rest: cut all the holes at once
            keep = np.ones(len(Q), bool)
            tubes = []
            for h in cut:
                fa, fb = h["faces"]
                Vn, Qn, newf, t = punch(V, Q, fa, fb, mesh=mesh)
                V = Vn
                tubes.append(Qn[newf])
                keep[[fa, fb]] = False
                touched.append(t)
            # faces flagged coarse keep their corners: refit them by vertex
            if len(cf):
                touched.append(np.unique(Q[cf[keep[cf]]]))
            Q = np.vstack([Q[keep]] + tubes)
            entry["holesCut"] = len(cut)
        elif not len(cf):
            break
        elif not refitted:
            touched.append(np.unique(Q[cf]))
            entry["refitOnly"] = int(len(cf))
            refitted = True
        else:
            res = inset(V, Q, cf)
            if res is None:
                break
            V, Q, t = res
            touched.append(t)
            entry["facesInset"] = int(len(cf))
        try:
            V, Q, tch = _compact(V, Q, np.concatenate(touched))
            c2 = subd.Cage(V, Q, sharp=subd.carry(cage.V, cage.sharp, V))
            if (np.bincount(c2.Q.ravel(), minlength=c2.n) < 3).any():
                raise ValueError("a vertex with two faces")
            H2 = H if (len(Q) == len(cage.Q) and (Q == cage.Q).all()) else subd.Hierarchy(c2, level)
            refit(c2, H2, mesh, tch, data, settle=tol / 50)
            if _new_folds(H2, c2.V, H, cage.V):
                raise ValueError("the surface folds")
        except (ValueError, RuntimeError) as err:
            # this edit is not made; the next round tries the next kind
            entry["dropped"] = str(err)
            log.append(entry)
            if entry["holesCut"]:
                nocut = True
            elif entry["facesInset"]:
                break
            continue
        dv, (e1, e2, _) = subd.deviation(H2, c2.V, mesh, data=check)
        if dv["max"] > 1.2 * max(dev["max"], 2.0 * bound):
            entry["dropped"] = f"the worst point went from {dev['max']:.3f} to {dv['max']:.3f} mm"
            log.append(entry)
            if entry["holesCut"]:
                nocut = True
            elif entry["facesInset"]:
                break
            continue
        cage, H, dev, d1, d2 = c2, H2, dv, e1, e2
        sl, todo, cf, cm = state(cage, H, d1, d2)
        entry.update({"faces": len(cage.Q), "p99": dev["p99"], "max": dev["max"], "holesLeft": len(todo),
                      "coarseMiss": round(cm, 3)})
        log.append(entry)
        if progress:
            progress(f"cage edited in place: {entry['slitsBridged']} slits bridged, {entry['holesCut']} holes cut, {entry['facesInset']} faces inset, "
                     f"{entry['refitOnly']} refitted; {len(cage.Q)} faces, p99 {dev['p99']:.4f} mm, "
                     f"{len(todo)} printable holes closed, coarse miss {cm:.3f}")
    return cage, H, dev, (d1, d2), {"holes": len(todo), "coarse": round(cm, 3) if len(cf) else 0.0}, log
