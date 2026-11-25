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

PRESPLIT = 4.0       # long triangles split to this many times the remesh step before it (feature_remesh)


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


def _arc(D, X):
    """Arc length along the closed dense loop D of the closest point of
    each of X."""
    from scipy.spatial import cKDTree

    s = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(D, axis=0), axis=1))]
    return s[cKDTree(D).query(X)[1]], float(s[-1])


class _Straight:
    """Paths of cage edges that go straight through every vertex they pass:
    in and out by edges that share no face. A crease that turns at a
    vertex, into the edge beside the one it came by, leaves that vertex
    one face on one side and three on the other: the limit surface there
    is a dart, not the flank of a straight crease, and no rectangle of
    patches follows it (an L-bracket's traced lines stepped from row to row
    at regular vertices and its patches missed by 0.27 mm). Dijkstra runs
    on half-edges, a step allowed from (u, v) to (v, w) only straight on."""

    def __init__(self, cage):
        E, Q, QE = cage.E, cage.Q, cage.QE
        self.E, self.ne = E, len(E)
        side = set()
        for q in range(len(Q)):
            for k in range(4):
                e1, e2 = int(QE[q, k]), int(QE[q, (k + 1) % 4])
                side.add((min(e1, e2), max(e1, e2)))
        inc = [[] for _ in range(cage.n)]
        turn, at = [], []
        for k, (u, v) in enumerate(E.tolist()):
            inc[u].append(k)
            inc[v].append(k)
        rows, cols = [], []
        for v, es in enumerate(inc):
            for e1 in es:
                h1 = 2 * e1 + (0 if E[e1, 1] == v else 1)        # arriving at v
                for e2 in es:
                    if e2 == e1:
                        continue
                    rows.append(h1)
                    cols.append(2 * e2 + (0 if E[e2, 0] == v else 1))  # leaving v
                    turn.append((min(e1, e2), max(e1, e2)) in side)
                    at.append(v)
        self.all_rows, self.all_cols = np.asarray(rows, np.int64), np.asarray(cols, np.int64)
        self.turn, self.at = np.asarray(turn, bool), np.asarray(at, np.int64)
        self.tail = np.c_[E[:, 0], E[:, 1]].ravel()       # half-edge 2k: E[k,0] -> E[k,1]
        self.head = np.c_[E[:, 1], E[:, 0]].ravel()
        self.free(None)

    def free(self, near, cost=0.0):
        """Vertices where the path may turn (`near`: bool per vertex, or
        None for nowhere), each turn costing `cost` more (mm of path)."""
        ok = ~self.turn if near is None else (~self.turn | near[self.at])
        self.rows, self.cols = self.all_rows[ok], self.all_cols[ok]
        self.pen = np.where(self.turn[ok], float(cost), 0.0)

    def _graph(self, w, extra=0, drop=None):
        from scipy.sparse import csr_matrix

        wh = np.repeat(w, 2)
        if drop is not None:
            wh = wh.copy()
            wh[drop] = np.inf
        ok = np.isfinite(wh[self.cols]) & np.isfinite(wh[self.rows])
        r, c = self.rows[ok], self.cols[ok]
        m = 2 * self.ne + extra
        return csr_matrix((wh[c] + self.pen[ok], (r, c)), shape=(m, m)), wh

    def _walk(self, pred, h, stop):
        hs = [h]
        while hs[-1] not in stop:
            p = int(pred[hs[-1]])
            if p < 0 or p >= 2 * self.ne:
                break
            hs.append(p)
        hs = hs[::-1]
        return [int(self.tail[hs[0]])] + [int(self.head[x]) for x in hs]

    def path(self, a, b, w):
        """Vertices of the least-weight straight path from a to b, or None."""
        from scipy.sparse import csr_matrix
        from scipy.sparse.csgraph import dijkstra

        G, wh = self._graph(w, extra=1)
        S = 2 * self.ne
        out = np.flatnonzero((self.tail == a) & np.isfinite(wh))
        if not len(out):
            return None
        G = (G + csr_matrix((wh[out], (np.full(len(out), S), out)), shape=G.shape)).tocsr()
        dist, pred = dijkstra(G, indices=S, return_predecessors=True)
        into = np.flatnonzero(self.head == b)
        if not len(into) or not np.isfinite(dist[into]).any():
            return None
        h = int(into[np.argmin(dist[into])])
        return self._walk(pred, h, set(out.tolist()))

    def ring(self, D, V, w):
        """The least-weight straight closed path round the loop D (dense,
        closed): the band cut at the loop's start, and the shortest way
        round from each edge that crosses the cut back to it. Its
        vertices, first repeated last; None if there is none."""
        from scipy.sparse.csgraph import dijkstra

        s, L = _arc(D, V)
        st, sh = s[self.tail], s[self.head]
        fwd = (st - sh > 0.5 * L) & np.isfinite(np.repeat(w, 2))    # crosses the cut going on round
        cross = np.abs(st - sh) > 0.5 * L
        src = np.flatnonzero(fwd)
        if not len(src):
            return None
        G, wh = self._graph(w, drop=np.flatnonzero(cross))
        wh[src] = np.repeat(w, 2)[src]            # their own weight, kept
        # the crossing half-edges are sources: their own steps on
        from scipy.sparse import csr_matrix
        ok = np.isin(self.rows, src) & np.isfinite(wh[self.cols]) & ~cross[self.cols]
        G = (G + csr_matrix((wh[self.cols[ok]] + self.pen[ok], (self.rows[ok], self.cols[ok])),
                            shape=G.shape)).tocsr()
        dist, pred = dijkstra(G, indices=src, return_predecessors=True)
        best, pick = np.inf, None
        for i, c in enumerate(src.tolist()):
            # back to c: a half-edge into c's tail that may step straight on to c
            ins = self.rows[self.cols == c]
            if not len(ins):
                continue
            d = dist[i, ins]
            j = int(np.argmin(d))
            if d[j] + wh[c] < best:
                best, pick = d[j] + wh[c], (i, int(ins[j]), c)
        if pick is None or not np.isfinite(best):
            return None
        i, h, c = pick
        hs = [h]
        while hs[-1] != c:
            p = int(pred[i, hs[-1]])
            if p < 0:
                return None
            hs.append(p)
        hs = hs[::-1]                       # c ... h
        path = [int(self.tail[hs[0]])] + [int(self.head[x]) for x in hs]
        if path[0] != path[-1] or len(set(path[:-1])) != len(path) - 1:
            return None
        a = s[path[:-1]]
        step = np.diff(np.r_[a, a[0]])
        step = (step + 0.5 * L) % L - 0.5 * L
        if abs(abs(step.sum()) - L) > 0.25 * L or (step * np.sign(step.sum()) < -0.01 * L).any():
            return None
        return path


def _check(path, closed, P, D, L, V, da, h, used_e, pinned, near_v):
    """Why a traced path cannot be the line's crease (None if it can), and
    if it can: (vertices, edge pairs, [(vertex, its place on the line)])."""
    if path is None or len(path) < 2:
        return "no path", None
    pv = np.asarray(path)
    if len(set(pv[:-1].tolist())) != len(pv) - 1:
        return "path crosses itself", None
    if float(np.linalg.norm(np.diff(V[pv], axis=0), axis=1).sum()) > (4.0 / 3.0) * L + h:
        return "path wanders", None
    # a vertex within two faces of a corner may come from two faces away
    # (the cage's star points sit near corners, not on them)
    if (da[pv] > np.where(near_v[pv], 2.0, 1.5) * h).any():
        return "path off the line", None
    pairs = [tuple(sorted((int(a), int(b)))) for a, b in zip(pv[:-1], pv[1:])]
    if any(p in used_e for p in pairs):
        return "edge on two lines", None
    X = _spread(D, closed, V[pv])
    mv = [(v, x) for v, x in zip(pv.tolist(), X) if v not in pinned]   # a corner stays where the first line put it
    return None, (pv, pairs, mv)


def _spread(D, closed, X):
    """Places on the dense line D for the path's vertices X: spread along
    it in proportion to the path's own edge lengths (an open path's ends
    on the line's ends, a ring from where its first vertex lands, the way
    it runs). The closest point would put two vertices of a jog in the
    path (a step across the rows near a corner) on one place: a crease
    edge of no length, the surface pinched there."""
    s = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(D, axis=0), axis=1))]
    L = float(s[-1])
    t = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(X, axis=0), axis=1))]
    t = t / max(float(t[-1]), 1e-300)
    if closed:
        a, _ = _arc(D, X)
        step = (np.diff(a) + 0.5 * L) % L - 0.5 * L
        sgn = 1.0 if step.sum() >= 0 else -1.0
        q = (a[0] + sgn * t * L) % L
    else:
        q = t * L
    out = np.empty((len(q), 3))
    for c in range(3):
        out[:, c] = np.interp(q, s, D[:, c])
    return out


def _drag(V, mv, pinned, h, mesh, tree):
    """Free vertices near the moved ones `mv` [(vertex, new position)]
    moved along: by the nearest moved vertex's shift times
    exp(-(d / h)^2), d their distance from it, then onto the surface.
    Returns [(vertex, new position)] for them."""
    import igl

    if not mv:
        return []
    ids = np.array([v for v, _ in mv])
    shift = np.array([x for _, x in mv]) - V[ids]
    near = set()
    for js in tree.query_ball_point(V[ids], 2.0 * h):
        near.update(js)
    near -= set(ids.tolist())
    near -= set(pinned)
    if not near:
        return []
    nb = np.array(sorted(near))
    from scipy.spatial import cKDTree

    d, j = cKDTree(V[ids]).query(V[nb])
    P = V[nb] + shift[j] * np.exp(-(d / h) ** 2)[:, None]
    _, _, C = igl.point_mesh_squared_distance(P, np.asarray(mesh.vertices, float), np.asarray(mesh.faces, np.int64))
    return list(zip(nb.tolist(), C))


def _folds(V, Q, vq, vs, X):
    """Whether moving vertices `vs` to `X` turns a face of the cage over:
    a face round them whose triangles turn by more than a right angle
    whichever diagonal splits it."""
    qs = sorted({q for v in vs for q in vq[v]})
    if not qs:
        return False
    W = V.copy()
    W[vs] = X
    F = Q[qs]
    T = np.concatenate([F[:, [0, 1, 2]], F[:, [0, 2, 3]], F[:, [0, 1, 3]], F[:, [1, 2, 3]]])

    def nrm(P):
        return np.cross(P[T[:, 1]] - P[T[:, 0]], P[T[:, 2]] - P[T[:, 0]])
    flip = (np.einsum("ij,ij->i", nrm(V), nrm(W)) < 0.0).reshape(4, -1)
    # a dart (one corner pushed in past its diagonal) still has one split
    # whose triangles both face as before; a face turned over has none
    return bool(((flip[0] | flip[1]) & (flip[2] | flip[3])).any())


def lay_creases(cage, mesh, tol, h=None, progress=None):
    """The cage with its edges on the mesh's feature lines made creases:
    returns (cage, report). The cage's vertices on each taken line are
    moved onto it."""
    from scipy.spatial import cKDTree

    from .subd import Cage

    h = float(h) if h else float(np.sqrt(mesh.area / len(cage.Q)))
    lines, nodes = feature_lines(mesh, tol, h)
    rep = {"lines": len(lines), "taken": 0, "creaseEdges": 0, "refused": {}, "refusedAt": [], "length": 0.0,
           "takenLength": 0.0}

    laid = {}

    def refuse(why):
        refuse_at(li, why)

    def refuse_at(li, why):
        rep["refused"][why] = rep["refused"].get(why, 0) + 1
        P_, closed_, _ = lines[li]
        rep["refusedAt"].append({"why": why, "at": np.round(P_.mean(0), 3).tolist(), "closed": bool(closed_),
                                 "length": round(float(np.linalg.norm(np.diff(dense[li], axis=0), axis=1).sum()), 3)})
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
    straight = _Straight(cage)
    vtree = cKDTree(V)
    vq = [[] for _ in range(n)]          # the faces round each vertex
    for q, f in enumerate(cage.Q.tolist()):
        for v in f:
            vq[v].append(q)
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
        L = float(np.linalg.norm(np.diff(D, axis=0), axis=1).sum())
        if closed:
            near_v = np.zeros(n, bool)
        else:
            near_v = np.minimum(np.linalg.norm(V - P[0], axis=1), np.linalg.norm(V - P[-1], axis=1)) < 2 * h
        # first a path straight through every vertex (turning only within
        # two faces of a corner, where the rows turn round it), then, if
        # that has none or it folds, one that may turn anywhere
        why = None
        for turns in (near_v, np.ones(n, bool)):
            straight.free(turns, 2.0 * h)          # a turn costs two faces of path
            if closed:
                # a loop is one ring round it: the band of edges along it
                # cut open at a seam, the shortest way round from one side
                # of the seam to the other
                path = straight.ring(D, V, w)
            else:
                a, b = node_v[ends[0]], node_v[ends[1]]
                path = straight.path(a, b, w) if a != b else None
            why, got = _check(path, closed, P, D, L, V, da, h, used_e, pinned, near_v)
            if why is None:
                pv, pairs, mv = got
                # how far a vertex may come to the line is not a distance:
                # it may come as far as it can without turning a face of
                # the cage over. Should the plain move fold one, the rows
                # beside it come along, as a row of a net pulled sideways
                # drags its neighbours (`_drag`; not first: near a crease
                # already laid the drag can carry a vertex over the corner)
                if _folds(V, cage.Q, vq, [v for v, _ in mv], np.array([x for _, x in mv])) if mv else False:
                    mv = mv + _drag(V, mv, pinned, h, mesh, vtree)
                    if _folds(V, cage.Q, vq, [v for v, _ in mv], np.array([x for _, x in mv])):
                        why = "path would fold the cage"
            if why is None:
                break
        if why is not None:
            refuse(why)
            continue
        for v, x in mv:
            V[v] = x
        for v in pv.tolist():
            pinned.setdefault(v, V[v].copy())
        used_e.update(pairs)
        laid[li] = (pairs, L)
    # a network of lines is laid whole or not at all: a crease that stops
    # where its neighbour was not laid tears the patch layout round its end
    # and fits worse than the rounded corner (the box cover with 5 of its
    # 31 lines: p99 0.36 mm against 0.18 smooth); a whole network laid and
    # another left rounded has no such end
    parent = list(range(len(lines)))

    def root(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    by_node = {}
    for li, (P, closed, ends) in enumerate(lines):
        for e in ends:
            if e is not None:
                if e in by_node:
                    parent[root(li)] = root(by_node[e])
                by_node[e] = li
    comps = {}
    for li in range(len(lines)):
        comps.setdefault(root(li), []).append(li)
    rep["networks"] = len(comps)
    rep["networksLaid"] = 0
    for members in comps.values():
        if all(li in laid for li in members):
            rep["networksLaid"] += 1
            for li in members:
                sharp += laid[li][0]
                rep["taken"] += 1
                rep["takenLength"] += laid[li][1]
        else:
            for li in members:
                if li in laid:
                    refuse_at(li, "its network not laid whole")
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


def feature_remesh(mesh, step, theta, lines=None):
    """`mesh` remeshed with even triangles about `step` mm whose edges run
    along its feature lines (`feature_lines` at turn `theta`, radians): the
    input Instant Meshes lays its crease-aligned field over.

    Instant Meshes solves its fields on the input's own vertex graph and
    finds creases there by dihedral angle. A CAD mesh is a few long
    triangles per flat face; split fine, its triangles stay slivers fanned
    from the corners, the field smoothed over them does not settle along
    the creases and the extraction tears or its rows miss the lines (the
    box cover: 5 of 31 lines). An even resampling by marching cubes
    (`subd._even`) has no slivers but rounds every crease off. Isotropic
    remeshing (Botsch & Kobbelt 2004: split, collapse, flip to valence 6,
    tangential relaxation, projection) gives both: even triangles, and the
    feature lines' edges kept as constrained edges (split along the line,
    never flipped, collapsed only along it). The long triangles are split
    first (to `step`): from a few huge triangles the remesher's own splits
    do not converge before its iterations end; split to four times `step`
    they do (its splits halve every edge over 4/3 of the target each
    iteration), and the split is the run's memory: a long sliver split to
    a length L goes into (its length / L)^2 triangles (an optimized
    fastened lid, 12000 mm^2 with 70 mm flats: some 12 million at a 0.25 mm
    `step`, out of 15 GB, for 0.43 million remeshed; at four times `step`
    1 million, 1.5 GB at the peak).

    `lines`: the feature lines as `feature_lines` returns them (computed at
    `theta` if not given). Returns a trimesh, cached on `mesh`."""
    import trimesh

    cache = mesh.__dict__.setdefault("_lp_feature_remesh", {})
    if lines is None:
        h = 4.0 * float(step)
        lines, _ = feature_lines(mesh, theta * h / 8.0, h, min_turn=theta)
    skey = round(float(step), 4)
    if skey not in cache:
        v, f = trimesh.remesh.subdivide_to_size(np.asarray(mesh.vertices, float), np.asarray(mesh.faces, np.int64),
                                                max_edge=PRESPLIT * float(step))
        cache[skey] = trimesh.Trimesh(v, f, process=True)
    sub = cache[skey]
    fe = _edges_on(sub, lines, theta)
    # the same lines at a nearby density are the same remesh
    key = (skey, np.unique(np.sort(fe, 1), axis=0).tobytes())
    if key in cache:
        return cache[key]
    try:
        import gpytoolbox
    except ImportError:          # the old route: the split mesh as it is
        cache[key] = sub
        return sub
    U, G = gpytoolbox.remesh_botsch(np.asarray(sub.vertices, float).copy(), np.asarray(sub.faces, np.int32),
                                    i=10, h=float(step), feature_edges=fe.astype(np.int64))
    out = trimesh.Trimesh(U, G, process=False)
    if not (out.is_watertight and out.euler_number == mesh.euler_number):
        out = sub                # a remesh that changed the shape's topology is not the shape
    cache[key] = out
    return out


def _edges_on(mesh, lines, theta):
    """The edges of `mesh` that lie along the feature `lines` (both ends
    and middle on one line, the surface turned there): (k, 2)."""
    if not lines:
        return np.zeros((0, 2), np.int64)
    V = np.asarray(mesh.vertices, float)
    E = np.asarray(mesh.face_adjacency_edges)[np.asarray(mesh.face_adjacency_angles) > 0.5 * theta]
    if not len(E):
        return np.zeros((0, 2), np.int64)
    eps = 1e-6 * float(np.linalg.norm(mesh.bounds[1] - mesh.bounds[0]))
    from scipy.spatial import cKDTree

    mid = 0.5 * (V[E[:, 0]] + V[E[:, 1]])
    on = np.zeros(len(E), bool)
    for P, closed, _ in lines:
        lo, hi = P.min(0) - eps, P.max(0) + eps
        cand = np.flatnonzero(~on & (mid >= lo).all(1) & (mid <= hi).all(1))
        if not len(cand):
            continue
        # each segment of the line checked against the edges near it
        Q = np.vstack([P, P[:1]]) if closed else P
        seg_mid = 0.5 * (Q[:-1] + Q[1:])
        seg_len = np.linalg.norm(Q[1:] - Q[:-1], axis=1)
        tree = cKDTree(seg_mid)
        for c in np.array_split(cand, max(1, len(cand) // 4096)):
            near = tree.query_ball_point(mid[c], 0.5 * float(seg_len.max()) + eps)
            for i, js in zip(c.tolist(), near):
                if not js:
                    continue
                js = np.asarray(js)
                A, B = Q[js], Q[js + 1]
                ok = True
                for x in (V[E[i, 0]], V[E[i, 1]], mid[i]):
                    D = B - A
                    t = np.clip(((x - A) * D).sum(1) / np.maximum((D * D).sum(1), 1e-300), 0, 1)
                    if np.linalg.norm(A + t[:, None] * D - x, axis=1).min() > eps:
                        ok = False
                        break
                on[i] = ok
    return E[on]


def feature_gap(mesh, lines, step, reach):
    """The narrowest face between two feature lines: for each line, the
    median distance across the surface to another line running beside it
    (both turned less than 60 degrees off square to the way across, and
    the way across on the surface: a 2 mm pocket wall between its two
    rims, not a skin's two faces through its thickness); the least of
    these, looked for out to `reach` mm. inf if no lines run beside each
    other that close. `step`: the sampling along the lines."""
    import igl
    from scipy.spatial import cKDTree

    if len(lines) < 2:
        return np.inf
    S, T, L = [], [], []
    for i, (P, closed, _) in enumerate(lines):
        D = _dense(P, closed, step)
        t = np.gradient(D, axis=0)
        t /= np.maximum(np.linalg.norm(t, axis=1), 1e-300)[:, None]
        S.append(D)
        T.append(t)
        L.append(np.full(len(D), i))
    S, T, L = np.vstack(S), np.vstack(T), np.concatenate(L)
    pr = cKDTree(S).query_pairs(float(reach), output_type="ndarray")
    if not len(pr):
        return np.inf
    a, b = pr[:, 0], pr[:, 1]
    d = S[b] - S[a]
    dd = np.linalg.norm(d, axis=1)
    u = d / np.maximum(dd, 1e-300)[:, None]
    ok = (dd > 0.5 * step) & (L[a] != L[b]) & (np.abs((u * T[a]).sum(1)) < 0.5) & (np.abs((u * T[b]).sum(1)) < 0.5)
    a, b, d, dd = a[ok], b[ok], d[ok], dd[ok]
    if not len(a):
        return np.inf
    # across the surface: the straight way over lies on it (a flat or a
    # ruled wall between two rims)
    V, F = np.asarray(mesh.vertices, float), np.asarray(mesh.faces, np.int64)
    sq, _, _ = igl.point_mesh_squared_distance(S[a] + 0.5 * d, V, F)
    on = np.sqrt(sq) < 0.02 * dd + 1e-6 * float(np.linalg.norm(mesh.bounds[1] - mesh.bounds[0]))
    a, b, dd = a[on], b[on], dd[on]
    best = np.full(len(S), np.inf)
    np.minimum.at(best, a, dd)
    np.minimum.at(best, b, dd)
    gaps = []
    for i in range(len(lines)):
        g = best[L == i]
        f = g[np.isfinite(g)]
        if len(f) >= 0.25 * len(g) and len(f):
            gaps.append(float(np.median(f)))
    return min(gaps) if gaps else np.inf
