"""Fewer star points on a fitted quad cage.

Every STEP face of the lines layout (subdlines) is a rectangle between
lines drawn from the cage's star points (vertices with 3 or 5+ edges), so
the file's face count goes with the star points. Instant Meshes leaves
many of them in pairs (a 3 diagonal to a 5, two 3s across a quad) where
its field turns a little and the shape does not ask for it.

Two local moves change which vertices are star points and keep the cage
closed, manifold and of the same genus (Tarini et al. 2010, "Practical quad
mesh simplification"):

- edge rotation: the edge between two quads turns to the other diagonal
  of the hexagon they make; its old ends lose an edge, its new ends gain
  one;
- diagonal collapse: a quad shrinks to a point along one diagonal; the two
  ends become one vertex with va + vc - 2 edges, the other two corners
  lose an edge each.

A move is taken when it leaves fewer star points among the vertices it
touches, or (two moves in a row) when a rotation that changes nothing on
its own lets a second move take two away; never below 3 edges at a vertex,
never a quad turned over. (Polychord collapse, the global move, does not
apply here: on an optimized part nearly every ring of quads crosses
itself.) Whether the surface can do without the old layout is for the fit
to say (subd.surface_mesh fits the moved neighbourhoods again and undoes
the moves whose surface misses)."""

import numpy as np


class _Mesh:
    """A quad cage as mutable lists: quads, each vertex's quads."""

    def __init__(self, V, Q):
        self.V = [np.asarray(p, float) for p in V]
        self.Q = [list(map(int, q)) for q in Q]
        self.alive = [True] * len(self.Q)
        self.vq = [set() for _ in self.V]
        for i, q in enumerate(self.Q):
            for v in q:
                self.vq[v].add(i)
        self.dead_v = set()

    def nbrs(self, v):
        out = set()
        for i in self.vq[v]:
            q = self.Q[i]
            k = q.index(v)
            out.add(q[(k + 1) % 4])
            out.add(q[(k + 3) % 4])
        return out

    def val(self, v):
        return len(self.vq[v])          # closed manifold: edges = quads round a vertex

    def other(self, qi, a, b):
        """The quad other than qi on edge a-b."""
        s = (self.vq[a] & self.vq[b]) - {qi}
        return next(iter(s)) if len(s) == 1 else None

    def normal(self, q):
        P = [self.V[v] for v in q]
        return np.cross(P[2] - P[0], P[3] - P[1])

    def set_quad(self, i, q):
        for v in self.Q[i]:
            self.vq[v].discard(i)
        self.Q[i] = list(q)
        for v in q:
            self.vq[v].add(i)

    def arrays(self):
        keep = [i for i, a in enumerate(self.alive) if a]
        Q = np.array([self.Q[i] for i in keep], np.int64)
        used = np.unique(Q)
        remap = np.full(len(self.V), -1)
        remap[used] = np.arange(len(used))
        return np.array([self.V[i] for i in used]), remap[Q], remap


def _star(v):
    """A vertex's share of the layout: the lines a star point sends out
    (one per edge), none from a regular vertex."""
    return 0 if v == 4 else int(v)


def rotations(m, qa, k):
    """The two rotations of quad qa's edge k: (vertices touched with their
    new edge counts, quads before, quads after), or [] where not allowed."""
    A = m.Q[qa]
    u, w, a1, a2 = A[k], A[(k + 1) % 4], A[(k + 2) % 4], A[(k + 3) % 4]
    qb = m.other(qa, u, w)
    if qb is None:
        return []
    B = m.Q[qb]
    kb = B.index(w)
    if B[(kb + 1) % 4] != u:
        return []
    b1, b2 = B[(kb + 2) % 4], B[(kb + 3) % 4]
    hexa = [w, a1, a2, u, b1, b2]
    if len(set(hexa)) < 6:
        return []
    out = []
    for x, y, qs in ((a1, b1, ([a1, a2, u, b1], [b1, b2, w, a1])),
                     (a2, b2, ([a2, u, b1, b2], [b2, w, a1, a2]))):
        if y in m.nbrs(x):
            continue
        new = {u: m.val(u) - 1, w: m.val(w) - 1, x: m.val(x) + 1, y: m.val(y) + 1}
        if min(new.values()) < 3:
            continue
        out.append((new, (qa, qb), qs))
    return out


def collapses(m, qi):
    """The two diagonal collapses of quad qi: (vertices touched with their
    new edge counts, kept vertex, removed vertex, the other two)."""
    q = m.Q[qi]
    out = []
    for s in (0, 1):
        a, b, c, d = q[s], q[s + 1], q[(s + 2) % 4], q[(s + 3) % 4]
        if c in m.nbrs(a) or (m.nbrs(a) & m.nbrs(c)) != {b, d}:
            continue
        new = {a: m.val(a) + m.val(c) - 2, b: m.val(b) - 1, d: m.val(d) - 1}
        if min(new.values()) < 3:
            continue
        out.append((new, a, c, (b, d)))
    return out


def _gain(m, new, extra=()):
    old = sum(_star(m.val(v)) for v in new) + sum(_star(m.val(v)) for v in extra)
    return sum(_star(n) for n in new.values()) - old


def _rotate(m, mv):
    new, (qa, qb), qs = mv
    n0 = m.normal(m.Q[qa]) + m.normal(m.Q[qb])
    if any(np.dot(m.normal(q), n0) <= 0.0 for q in qs):
        return False
    m.set_quad(qa, qs[0])
    m.set_quad(qb, qs[1])
    return True


def _collapse(m, qi, mv):
    new, a, c, _ = mv
    n0 = [(i, m.normal(m.Q[i])) for i in m.vq[c] | m.vq[a] if i != qi]
    m.V[a] = 0.5 * (m.V[a] + m.V[c])
    for i in list(m.vq[c]):
        if i == qi:
            continue
        m.set_quad(i, [a if v == c else v for v in m.Q[i]])
    for v in list(m.Q[qi]):
        m.vq[v].discard(qi)
    m.alive[qi] = False
    m.dead_v.add(c)
    m.vq[c] = set()
    return all(np.dot(m.normal(m.Q[i]), n) > 0.0 for i, n in n0)


def cancel(V, Q, frozen=(), rounds=6):
    """Edge rotations and diagonal collapses that take star points away,
    greedily, away from `frozen` vertices (holes just punched, say).
    Returns (V, Q, touched vertices in the new numbering, moves taken)."""
    m = _Mesh(V, Q)
    frozen = set(int(v) for v in frozen)
    touched = set()
    taken = 0
    for _ in range(rounds):
        busy = set(frozen)
        done = 0
        # single moves that gain, best first
        moves = []
        for qi in range(len(m.Q)):
            if not m.alive[qi]:
                continue
            for k in range(4):
                for mv in rotations(m, qi, k):
                    g = _gain(m, mv[0])
                    if g < 0:
                        moves.append((g, "r", qi, mv))
            for mv in collapses(m, qi):
                g = _gain(m, mv[0], extra=(mv[2],))
                if g < 0:
                    moves.append((g, "c", qi, mv))
        moves.sort(key=lambda x: x[0])
        for g, kind, qi, mv in moves:
            verts = set(mv[0]) | ({mv[2]} if kind == "c" else set())
            ring = set().union(*(m.nbrs(v) for v in verts)) | verts
            if ring & busy or not m.alive[qi]:
                continue
            # the move is still what it was (nothing near it moved)
            if kind == "r":
                ok = _rotate(m, mv)
            else:
                save = (list(m.V), [list(q) for q in m.Q], list(m.alive), [set(s) for s in m.vq])
                ok = _collapse(m, qi, mv)
                if not ok:
                    m.V, m.Q, m.alive, m.vq = save
                    m.dead_v.discard(mv[2])
            if ok:
                busy |= ring
                touched |= verts
                done += 1
        # two in a row: a rotation that changes nothing on its own, then a
        # move it makes possible next to it
        pairs = 0
        for qi in range(len(m.Q)):
            if not m.alive[qi]:
                continue
            for k in range(4):
                for mv in rotations(m, qi, k):
                    if _gain(m, mv[0]) != 0:
                        continue
                    verts = set(mv[0])
                    ring = set().union(*(m.nbrs(v) for v in verts)) | verts
                    if ring & busy:
                        continue
                    save = ([list(q) for q in m.Q], [set(s) for s in m.vq])
                    if not _rotate(m, mv):
                        continue
                    best = None
                    for qj in set().union(*(m.vq[v] for v in verts)):
                        for k2 in range(4):
                            for mv2 in rotations(m, qj, k2):
                                if set(mv2[0]) & verts and _gain(m, mv2[0]) < 0:
                                    best = best or ("r", qj, mv2)
                        for mv2 in collapses(m, qj):
                            if set(mv2[0]) & verts and _gain(m, mv2[0], extra=(mv2[2],)) < 0:
                                best = best or ("c", qj, mv2)
                    ok = False
                    if best is not None:
                        kind2, qj, mv2 = best
                        if kind2 == "r":
                            ok = _rotate(m, mv2)
                        else:
                            s2 = (list(m.V), [list(q) for q in m.Q], list(m.alive), [set(s) for s in m.vq])
                            ok = _collapse(m, qj, mv2)
                            if not ok:
                                m.V, m.Q, m.alive, m.vq = s2
                                m.dead_v.discard(mv2[2])
                    if ok:
                        v2 = set(mv2[0]) | ({mv2[2]} if kind2 == "c" else set())
                        allv = verts | v2
                        busy |= allv | set().union(*(m.nbrs(v) for v in allv if v not in m.dead_v))
                        touched |= allv
                        pairs += 1
                    else:
                        m.Q, m.vq = save
        taken += done + 2 * pairs
        if not done and not pairs:
            break
    Vn, Qn, remap = m.arrays()
    t = np.array(sorted(v for v in touched if v not in m.dead_v), np.int64)
    return Vn, Qn, remap[t][remap[t] >= 0] if len(t) else t, taken


def _pairs(m, busy):
    """3-5 pairs: a three-edge star point and a five-or-more one on one
    quad (a dislocation of the quad grid: on its own neither can go)."""
    out, seen = [], set()
    for v in range(len(m.V)):
        if v in m.dead_v or v in busy or v in seen or m.val(v) != 3:
            continue
        for qi in m.vq[v]:
            w = next((u for u in m.Q[qi] if u != v and u not in seen and u not in busy and m.val(u) >= 5), None)
            if w is not None:
                out.append((v, w))
                seen |= {v, w}
                break
    return out


def _gains_near(m, verts):
    """A single move round `verts` that takes star points away."""
    ring = set(verts)
    for _ in range(2):
        ring |= set().union(*(m.nbrs(v) for v in ring))
    for qi in set().union(*(m.vq[v] for v in ring)):
        if not m.alive[qi]:
            continue
        for k in range(4):
            if any(_gain(m, mv[0]) < 0 for mv in rotations(m, qi, k)):
                return True
        if any(_gain(m, mv[0], extra=(mv[2],)) < 0 for mv in collapses(m, qi)):
            return True
    return False


def _pair_moves(m, a, b, busy):
    """Rotations that move the 3-5 pair (a, b) whole: a gets an edge, b
    gives one, and a new 3 and 5 appear on one quad beside them."""
    out = []
    for qi in m.vq[a] | m.vq[b]:
        for k in range(4):
            for mv in rotations(m, qi, k):
                new = mv[0]
                if new.get(a) != 4 or new.get(b) != 4 or busy & set(new):
                    continue
                rest = [v for v in new if v not in (a, b)]
                if sorted(new[v] for v in rest) != [3, 5]:
                    continue
                n3 = next(v for v in rest if new[v] == 3)
                n5 = next(v for v in rest if new[v] == 5)
                out.append((mv, n3, n5))
    return out


def glide(V, Q, frozen=(), depth=8):
    """3-5 pairs slid across the cage until they meet star points they
    cancel with. A pair is a dislocation: no single move takes it away,
    but an edge rotation that takes an edge from its five and gives one to
    its three moves it whole one edge on (Peng et al. 2011, "Connectivity
    editing for quadrilateral meshes"). From each pair a breadth-first
    search over such rotations (at most `depth`, each place once) looks
    for the shortest walk that ends where a move takes star points away;
    that walk is made (cancel then takes them), nothing else moves.
    Returns (V, Q, touched vertices in the new numbering, pairs moved)."""
    from collections import deque

    m = _Mesh(V, Q)
    busy = set(int(v) for v in frozen)
    touched = set()
    moved = 0

    def apply(path):
        done = []
        for qa, qb, A, B in path:
            done.append((qa, qb, list(m.Q[qa]), list(m.Q[qb])))
            m.set_quad(qa, A)
            m.set_quad(qb, B)
        return done

    def undo(done):
        for qa, qb, A, B in reversed(done):
            m.set_quad(qa, A)
            m.set_quad(qb, B)

    for s3, s5 in _pairs(m, busy):
        if s3 in busy or s5 in busy or m.val(s3) != 3 or m.val(s5) < 5:
            continue
        seen = {(s3, s5)}
        run = deque([((s3, s5), [])])
        found = None
        while run and found is None:
            (a, b), path = run.popleft()
            if len(path) >= depth:
                continue
            done = apply(path)
            for mv, n3, n5 in _pair_moves(m, a, b, busy):
                if (n3, n5) in seen:
                    continue
                seen.add((n3, n5))
                new, (qa, qb), qs = mv
                n0 = m.normal(m.Q[qa]) + m.normal(m.Q[qb])
                if any(np.dot(m.normal(q), n0) <= 0.0 for q in qs):
                    continue
                step = path + [(qa, qb, list(qs[0]), list(qs[1]))]
                d2 = apply(step[-1:])
                if _gains_near(m, {n3, n5}):
                    found = (step, n3, n5)
                undo(d2)
                if found:
                    break
                run.append(((n3, n5), step))
            undo(done)
        if found is None:
            continue
        step, n3, n5 = found
        apply(step)
        verts = {s3, s5, n3, n5} | {v for qa, qb, A, B in step for v in A + B}
        busy |= verts | set().union(*(m.nbrs(v) for v in verts))
        touched |= verts
        moved += 1
    Vn, Qn, remap = m.arrays()
    t = np.array(sorted(touched), np.int64)
    return Vn, Qn, remap[t][remap[t] >= 0] if len(t) else t, moved


def simplify(V, Q, frozen=(), depth=24, rounds=6):
    """`cancel`, then pairs slid together (`glide`) and cancelled again
    while that takes star points away. Vertices are renumbered by each
    collapse, so the touched and frozen ones are carried by position.
    Returns (V, Q, touched vertices, moves taken)."""
    from scipy.spatial import cKDTree

    V0 = np.asarray(V, float)
    fz = V0[np.asarray(list(frozen), np.int64)] if len(frozen) else np.zeros((0, 3))
    V, Q, t, taken = cancel(V, Q, frozen=frozen, rounds=rounds)
    hit = [V[t]]
    for _ in range(rounds):
        tree = cKDTree(V)
        fr = tree.query(fz)[1] if len(fz) else ()
        Vg, Qg, tg, moved = glide(V, Q, frozen=fr, depth=depth)
        if not moved:
            break
        hit.append(Vg[tg])
        fr = cKDTree(Vg).query(fz)[1] if len(fz) else ()
        V, Q, tc, n = cancel(Vg, Qg, frozen=fr, rounds=rounds)
        hit.append(V[tc])
        taken += moved + n
    P = np.concatenate(hit) if hit else np.zeros((0, 3))
    if not len(P):
        return V, Q, np.zeros(0, np.int64), taken
    return V, Q, np.unique(cKDTree(V).query(P)[1]), taken
