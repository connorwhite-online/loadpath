"""Sharp edges for organic surfacing: the mesh's feature lines found, and
the cage's own edges laid on them as creases (subd.Cage `sharp`), so the
limit surface turns the corner there exactly instead of rounding it.

A feature line is a chain of mesh edges across which the surface turns by
more than a rounding over one cage face could follow within the
tolerance: a smooth surface rounds a corner of turn theta over about one
face h, and misses it by about h theta / 8 (the sagitta of the arc), so a
corner is a crease once theta > 8 tol / h. A chain shorter than two faces
is noise (a marching-cubes ripple), not a feature.

The cage (Instant Meshes, its field aligned to the creases) has a row of
edges near each feature line but not on it; each line is traced through
the cage's edge graph (shortest path, its cost the edge length weighted by
how far the edge lies off the line) and the path's vertices are moved onto
the line. A path that wanders (longer than the line by a third) or comes
off it by more than half a face is not taken: that line stays rounded."""
import numpy as np


def feature_lines(mesh, tol, h, min_turn=None):
    """Chains of mesh edges where the surface turns by more than a face
    of width `h` can round within `tol`. Returns a list of (points (k, 3),
    closed, (start node, end node)) and the node positions (corners and
    ends shared between lines)."""
    theta = float(min_turn) if min_turn is not None else 8.0 * float(tol) / float(h)
    ang = np.asarray(mesh.face_adjacency_angles)
    sel = ang > theta
    V = np.asarray(mesh.vertices, float)
    if sel.any():
        sel[np.flatnonzero(sel)] = _holds(mesh, np.flatnonzero(sel), theta, h)
    E = np.asarray(mesh.face_adjacency_edges)[sel]
    if not len(E):
        return [], np.zeros((0, 3))
    deg = np.bincount(E.ravel(), minlength=len(V))
    adj = {}
    for k, (a, b) in enumerate(E.tolist()):
        adj.setdefault(a, []).append((b, k))
        adj.setdefault(b, []).append((a, k))
    used = np.zeros(len(E), bool)
    chains = []
    # open chains from every vertex that is not a plain chain point
    starts = [v for v in adj if deg[v] != 2]
    for s in starts:
        for (b, k) in adj[s]:
            if used[k]:
                continue
            path, cur, kk = [s], b, k
            used[kk] = True
            while True:
                path.append(cur)
                if deg[cur] != 2:
                    break
                nxt = [(x, j) for (x, j) in adj[cur] if not used[j]]
                if not nxt:
                    break
                cur, kk = nxt[0]
                used[kk] = True
            chains.append((path, False))
    # closed loops (every vertex of degree 2)
    for k in np.flatnonzero(~used):
        if used[k]:
            continue
        a, b = E[k]
        used[k] = True
        path, cur = [int(a)], int(b)
        while cur != path[0]:
            path.append(cur)
            nxt = [(x, j) for (x, j) in adj[cur] if not used[j]]
            if not nxt:
                break
            cur, j = nxt[0]
            used[j] = True
        chains.append((path, True))
    # a network with a corner every face or so is a staircase (a field rim
    # cut on the grid branches at every step), not lines a cage can carry
    parent = {}

    def root(x):
        while parent.setdefault(x, x) != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    clen = []
    for path, closed in chains:
        P = V[path]
        clen.append(float(np.linalg.norm(np.diff(np.vstack([P, P[:1]]) if closed else P, axis=0), axis=1).sum()))
        parent[root(path[0])] = root(path[-1])
    comp_len, comp_corners = {}, {}
    for (path, closed), L in zip(chains, clen):
        r = root(path[0])
        comp_len[r] = comp_len.get(r, 0.0) + L
        comp_corners.setdefault(r, set()).update(v for v in (path[0], path[-1]) if deg[v] >= 3)
    lines = []
    node_ids = {}
    for (path, closed), L in zip(chains, clen):
        P = V[path]
        r = root(path[0])
        if len(comp_corners[r]) > 0.5 * comp_len[r] / h:
            continue
        # a short chain is noise unless it runs corner to corner (a box's
        # 6 mm side between two 3-way corners is a feature at any face size)
        # and is long enough for a cage edge (half a face): shorter, its two
        # corners are one cage vertex
        if L < 2.0 * h and (closed or deg[path[0]] < 3 or deg[path[-1]] < 3 or L < 0.5 * h):
            continue
        # a sawtooth line (a field rim cut on the grid) is not a line a cage
        # edge can lie on within the tolerance: it stays rounded
        if _jag(P, closed, h) > tol:
            continue
        lines.append((path, P, closed))
    # an open line must end at a corner of the kept lines: one that stops
    # mid-surface is a fragment of a rim whose rest is ragged, and a crease
    # laid on part of a rim tears the patch layout round it
    while True:
        at = {}
        for path, P, closed in lines:
            if not closed:
                for v in (path[0], path[-1]):
                    at[v] = at.get(v, 0) + 1
        keep = [x for x in lines if x[2] or (at[x[0][0]] >= 3 and at[x[0][-1]] >= 3)]
        if len(keep) == len(lines):
            break
        lines = keep
    out = []
    for path, P, closed in lines:
        ends = (None, None)
        if not closed:
            ends = tuple(node_ids.setdefault(int(path[i]), len(node_ids)) for i in (0, -1))
        out.append((P, closed, ends))
    lines = out
    nodes = np.zeros((len(node_ids), 3))
    for v, i in node_ids.items():
        nodes[i] = V[v]
    return lines, nodes


def _jag(P, closed, h):
    """How far the line strays from itself averaged over a face width: the
    zigzag a cubic crease curve through vertices a face apart can't follow
    (a circle's own bend costs only h^2 / 24 r of it)."""
    D = _dense(P, closed, 0.1 * h)
    if closed:
        D = D[:-1]
    if len(D) < 12:
        return 0.0
    k = 5
    if closed:
        W = np.vstack([D[-k:], D, D[:k]])
    else:
        W = D
    c = np.cumsum(np.vstack([np.zeros((1, 3)), W]), axis=0)
    M = (c[2 * k + 1:] - c[:-2 * k - 1]) / (2 * k + 1)
    X = D if closed else D[k:-k]
    return float(np.linalg.norm(X - M, axis=1).max()) if len(X) else 0.0


def _holds(mesh, idx, theta, h):
    """Whether the turn across each adjacency `idx` holds half a face away
    on both sides: a crease's two flanks stay turned, a ripple's agree
    again (the surface a face away is what the cage has to follow)."""
    import igl
    from scipy.spatial import cKDTree

    V = np.asarray(mesh.vertices, float)
    N = np.asarray(mesh.face_normals, float)
    C = np.asarray(mesh.triangles_center, float)
    A = np.asarray(mesh.area_faces, float)
    F = np.asarray(mesh.face_adjacency)[idx]
    Ed = np.asarray(mesh.face_adjacency_edges)[idx]
    m = 0.5 * (V[Ed[:, 0]] + V[Ed[:, 1]])
    d = V[Ed[:, 1]] - V[Ed[:, 0]]
    tree = cKDTree(C)
    k = min(16, len(C))
    Fa = np.asarray(mesh.faces, np.int64)
    side = []
    for s in (0, 1):
        n0 = N[F[:, s]]
        u = np.cross(n0, d)
        u /= np.maximum(np.linalg.norm(u, axis=1), 1e-300)[:, None]
        u *= np.sign(np.einsum("ij,ij->i", u, C[F[:, s]] - m))[:, None]
        p = m + 0.5 * h * u
        dist, j = tree.query(p, k=k)
        j = np.atleast_2d(j)
        dist = np.atleast_2d(dist)
        Nj = N[j]
        # the flank's own sheet: near the probe, facing as the face beside
        # the edge does (not the far side of a thin wall)
        w = A[j] * (dist < 0.5 * h) * (np.einsum("ikj,ij->ik", Nj, n0) > 0)
        # and the face the probe lands on, however large (a CAD mesh's
        # flank is one long triangle whose centre is far off)
        _, f, _ = igl.point_mesh_squared_distance(p, V, Fa)
        nn = (Nj * w[..., None]).sum(1) + N[f] * A[f][:, None]
        side.append(nn / np.maximum(np.linalg.norm(nn, axis=1), 1e-300)[:, None])
    far = np.arccos(np.clip(np.einsum("ij,ij->i", side[0], side[1]), -1, 1))
    return far > 0.5 * theta


def _dense(P, closed, step):
    Q = np.vstack([P, P[:1]]) if closed else P
    out = [Q[0]]
    for a, b in zip(Q[:-1], Q[1:]):
        k = max(1, int(np.ceil(np.linalg.norm(b - a) / step)))
        out += [a + (b - a) * t for t in np.arange(1, k + 1) / k]
    return np.asarray(out)


def _project(P, closed, X):
    """Closest points of X on the polyline P."""
    Q = np.vstack([P, P[:1]]) if closed else P
    A, B = Q[:-1], Q[1:]
    D = B - A
    dd = np.maximum((D * D).sum(1), 1e-300)
    t = np.clip(np.einsum("ijk,jk->ij", X[:, None, :] - A[None], D) / dd, 0.0, 1.0)
    C = A[None] + t[..., None] * D[None]
    d = np.linalg.norm(C - X[:, None, :], axis=2)
    j = d.argmin(1)
    return C[np.arange(len(X)), j], d[np.arange(len(X)), j]


def lay_creases(cage, mesh, tol, h=None, progress=None):
    """The cage with its edges on the mesh's feature lines made creases:
    returns (cage, report). The cage's vertices on each taken line are
    moved onto it."""
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import dijkstra
    from scipy.spatial import cKDTree

    from .subd import Cage

    h = float(h) if h else float(np.sqrt(mesh.area / len(cage.Q)))
    lines, nodes = feature_lines(mesh, tol, h)
    rep = {"lines": len(lines), "taken": 0, "creaseEdges": 0, "refused": {}, "length": 0.0, "takenLength": 0.0}

    def refuse(why):
        rep["refused"][why] = rep["refused"].get(why, 0) + 1
    if not lines:
        return cage, rep
    V = cage.V.copy()
    E = cage.E
    n = len(V)
    tree = cKDTree(V)
    # a corner of three or more lines takes the cage's star point beside it
    # (Instant Meshes puts one near each corner, rarely on it), else the
    # nearest vertex
    star = np.flatnonzero(cage.valence != 4)
    stree = cKDTree(V[star]) if len(star) else None
    node_v = {}
    for i, p in enumerate(nodes):
        j = int(tree.query(p)[1])
        if stree is not None:
            ds, k = stree.query(p)
            if ds < 1.5 * h:
                j = int(star[k])
        node_v[i] = j
    sharp = []
    used_e = set()
    pinned = {}
    lines = sorted(lines, key=lambda x: -len(x[0]))
    dense = [_dense(P, closed, 0.1 * h) for P, closed, _ in lines]
    rep["length"] = float(sum(np.linalg.norm(np.diff(D, axis=0), axis=1).sum() for D in dense))
    for li, (P, closed, ends) in enumerate(lines):
        D = dense[li]
        ct = cKDTree(D)
        mid = 0.5 * (V[E[:, 0]] + V[E[:, 1]])
        dm = ct.query(mid)[0]
        da = ct.query(V)[0]
        ln = np.linalg.norm(V[E[:, 0]] - V[E[:, 1]], axis=1)
        off = np.maximum(dm, np.maximum(da[E[:, 0]], da[E[:, 1]])) / (0.25 * h)
        w = ln * (1.0 + off ** 2)
        far = off > 6.0              # more than a face and a half off the line
        w[far] = np.inf
        if used_e:
            ue = np.array(sorted(used_e))
            key = E[:, 0] * n + E[:, 1]
            w[np.isin(key, ue[:, 0] * n + ue[:, 1])] = np.inf
        ok = np.isfinite(w)
        G = csr_matrix((np.r_[w[ok], w[ok]], (np.r_[E[ok, 0], E[ok, 1]], np.r_[E[ok, 1], E[ok, 0]])), shape=(n, n))
        if closed:
            k = len(D)
            marks = [int(tree.query(D[i])[1]) for i in (0, k // 3, 2 * k // 3)]
            legs = [(marks[0], marks[1]), (marks[1], marks[2]), (marks[2], marks[0])]
        else:
            legs = [(node_v[ends[0]], node_v[ends[1]])]
        path = []
        good = True
        for a, b in legs:
            if a == b:
                good = False
                break
            if path:
                # a loop's later legs keep off the vertices it has taken
                # (else the way back runs over the way out)
                bar = np.zeros(n, bool)
                bar[path] = True
                bar[[a, b]] = False
                wl = w[ok] * 1.0
                wl[bar[E[ok, 0]] | bar[E[ok, 1]]] = np.inf
                fin = np.isfinite(wl)
                Ek = E[ok][fin]
                Gl = csr_matrix((np.r_[wl[fin], wl[fin]], (np.r_[Ek[:, 0], Ek[:, 1]], np.r_[Ek[:, 1], Ek[:, 0]])),
                                shape=(n, n))
            else:
                Gl = G
            dist, pred = dijkstra(Gl, indices=a, return_predecessors=True)
            if not np.isfinite(dist[b]):
                good = False
                break
            seg = [b]
            while seg[-1] != a:
                seg.append(int(pred[seg[-1]]))
            seg = seg[::-1]
            path += seg if not path else seg[1:]
        if not good or len(path) < 2:
            refuse("no path")
            continue
        pv = np.asarray(path)
        if len(set(pv[:-1].tolist())) != len(pv) - 1:
            refuse("path crosses itself")
            continue
        L = float(np.linalg.norm(np.diff(D, axis=0), axis=1).sum())
        Lp = float(np.linalg.norm(np.diff(V[pv], axis=0), axis=1).sum())
        if Lp > (4.0 / 3.0) * L + h:
            refuse("path wanders")
            continue
        # within two faces of a corner the rows turn round it (the cage's
        # star points sit near corners, not on them): a vertex there may
        # come from a face and a half away
        if closed:
            near_end = np.zeros(len(pv), bool)
        else:
            near_end = np.minimum(np.linalg.norm(V[pv] - P[0], axis=1), np.linalg.norm(V[pv] - P[-1], axis=1)) < 2 * h
        if (da[pv] > np.where(near_end, 2.0, 0.75) * h).any():
            refuse("path off the line")
            continue
        pairs = [tuple(sorted((int(a), int(b)))) for a, b in zip(pv[:-1], pv[1:])]
        if any(p in used_e for p in pairs):
            refuse("edge on two lines")
            continue
        X, _ = _project(P, closed, V[pv])
        for v, x in zip(pv.tolist(), X):
            if v not in pinned:         # a corner stays where the first line put it
                V[v] = x
                pinned[v] = x
        used_e.update(pairs)
        sharp += pairs
        rep["taken"] += 1
        rep["takenLength"] += L
    if not sharp:
        return cage, rep
    c2 = Cage(V, cage.Q, sharp=np.asarray(sharp))
    rep["creaseEdges"] = int(c2.crease.sum())
    if progress:
        progress(f"creases: {rep['taken']} of {rep['lines']} feature lines laid on {rep['creaseEdges']} cage edges")
    return c2, rep


def carry(old_V, old_sharp, new_V, tol=1e-9):
    """Crease pairs of a cage carried onto an edited one by position (the
    edit leaves crease vertices where they were)."""
    from scipy.spatial import cKDTree

    if old_sharp is None or not len(old_sharp):
        return np.zeros((0, 2), np.int64)
    d, j = cKDTree(new_V).query(old_V[np.unique(old_sharp)])
    m = dict(zip(np.unique(old_sharp).tolist(), np.where(d <= tol, j, -1).tolist()))
    pairs = np.array([[m[a], m[b]] for a, b in np.asarray(old_sharp).tolist()], np.int64)
    return pairs[(pairs >= 0).all(1)]
