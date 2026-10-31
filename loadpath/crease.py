"""
Sharp creases between field surfaces, without an exact B-rep.

Marching cubes cannot hold a crease: wherever a hard boolean joins two
surfaces at an angle, the triangles that straddle the join cut the corner
off, and the part comes out rounded over about one grid step. sharp.py
fixes that where one side of the boolean is a plain shape or a mesh, by
doing the boolean exactly; between two organic fields (a TPMS clipped by
a shell, two smooth bodies unioned without a blend, a body minus another)
there is no exact surface to hand to a mesh boolean.

But the graph still knows how the field was built. A hard union,
intersect or subtract is a min or a max of its inputs, so at every point
one input (a *leaf*) is the active one, and the crease is where the active
leaf changes: the curve where two leaf surfaces meet (or the corner where
three do). `CreaseTree.from_graph` reads that min/max tree off the graph;
`sharpen(V, F, tree, pitch)` then works on the finished, projected
marching-cubes mesh:

1. every vertex is labelled with its active leaf; an edge whose ends have
   different labels crosses a crease;
2. a vertex very close to the crease (the crease point found from its
   edge is within `snap` of the edge length of it) is moved onto the
   crease and takes both labels, so the split below leaves no slivers;
3. every edge that still crosses a crease is split at the crease point,
   found by Gauss-Newton on the two leaves' zero levels (two constraints:
   the intersection curve; three where three labels meet in a triangle:
   the corner), and every triangle is re-cut by how many of its edges
   were split (two, three or six triangles);
4. near the creases, every triangle thinner than 8 degrees or turned over
   against its own surface's true normal is collapsed away (shortest edge,
   the crease end kept, link condition checked, nothing may turn over);
5. a crease vertex whose triangles still cross another part of the mesh
   or face the wrong way goes back to where the plain mesh had it: the
   crease is kept only where it does no harm.

Edges are left alone where there is no crease to see or the tree does not
describe the mesh: two leaves meeting at under `tangent` degrees (a skin's
outside lying on the body it was shelled from), within a few rings of a hole
the mesher left (a thin wall; the boolean step repairs it), and where
`tree.ignore` says so (sharp.py: near an exact cut, where the mesh is the
extended field). 

Where lattice walls meet a body at under ~45 degrees the material (or the
gap) between them is thinner than one pitch for a pitch or more back from
the true crease: that knife edge is not in the marching-cubes mesh at all,
and no relax or snapping of its vertices brings it back (moonshots round 2,
crease-aware relax tried 2026-10-31: same-side smoothing, own-leaf
projection, a flip guard; none beat the plain relax before this pass).

The split is decided per edge, so both triangles of an edge agree and the
mesh stays watertight; no vertex off the crease moves (other than the few
collapsed into it). A crease point that
Gauss-Newton does not find near the edge (the leaves are tangent there, or
a third leaf owns the spot) falls back to the edge midpoint projected onto
the field, which is what the mesh had. The crease vertices land on the
true intersection to the solver tolerance, and the triangles on each side
lie on their own surface, so the dihedral angle comes out as designed.
"""
from __future__ import annotations

import numpy as np

__all__ = ["CreaseTree", "sharpen"]


class _Leaf:
    __slots__ = ("fn", "sign", "name", "index")

    def __init__(self, fn, sign, name):
        self.fn, self.sign, self.name, self.index = fn, sign, name, -1


class _Node:
    __slots__ = ("kind", "kids")

    def __init__(self, kind, kids):
        self.kind, self.kids = kind, kids      # kind: "min" | "max"


def _negate(t):
    if isinstance(t, _Leaf):
        return _Leaf(t.fn, -t.sign, t.name)
    return _Node("max" if t.kind == "min" else "min", [_negate(k) for k in t.kids])


class CreaseTree:
    """The hard min/max tree above a field: leaves are smooth fields, each
    with a sign (a subtracted input enters negated). `values(P)` gives
    every leaf's signed value, `evaluate(P)` the field and the active leaf."""

    def __init__(self, root, evaluator=None):
        self.root = root
        self.leaves = []
        seen = {}

        def walk(t):
            if isinstance(t, _Leaf):
                # the same field met twice (a body that is both shelled and
                # clipped against) is one surface: one leaf
                key = (t.name, t.sign)
                if key in seen:
                    t.index = seen[key]
                    return
                t.index = seen[key] = len(self.leaves)
                self.leaves.append(t)
            else:
                for k in t.kids:
                    walk(k)
        walk(root)
        self._evaluator = evaluator
        # optional P -> bool: points where the mesh is not this tree's field
        # (sharp.py's extended field near an exact cut): no crease there
        self.ignore = None

    def __len__(self):
        return len(self.leaves)

    # -- construction --

    @classmethod
    def from_graph(cls, g, name=None, shells=True):
        """The tree of graph node `name` (default the output), or None when
        it has no hard boolean between fields at the top (nothing to sharpen).
        Descends through union, intersect and subtract with no blend and,
        with `shells`, through shells (inside: max(f, -(f + t)), outside:
        max(-f, f - t)), whose rims are creases too."""
        from .narrowband import peval  # noqa: F401  (import check: fields are numpy callables)

        def is_hard(nd):
            b = nd.refs.get("blend")
            return b is None or (isinstance(b, float) and abs(b) <= 1e-12)

        def leaf(ref, sign=1.0, plus=None):
            # a leaf: node `ref` (+ an optional thickness node or number)
            def fn(P, memo):
                v = g._val(ref, P, memo)
                if plus is not None:
                    v = v + g._val(plus[0], P, memo) * plus[1]
                return np.broadcast_to(np.asarray(v, float), (len(P),))
            t = None if plus is None else plus[0] if isinstance(plus[0], str) else f"{plus[0]:g}"
            return _Leaf(fn, sign, ref if plus is None else f"{ref}{'+' if plus[1] > 0 else '-'}{t}")

        def build(ref):
            if isinstance(ref, float):
                return None
            nd = g.nodes[ref]
            if nd.op in ("union", "intersect", "subtract") and is_hard(nd):
                kids = [build(x) or leaf(x) for x in nd.refs["inputs"] if not isinstance(x, float)]
                if len(kids) != len(nd.refs["inputs"]):
                    return None
                if nd.op == "union":
                    return _Node("min", kids)
                if nd.op == "intersect":
                    return _Node("max", kids)
                return _Node("max", [kids[0]] + [_negate(k) for k in kids[1:]])
            if shells and nd.op == "shell" and not isinstance(nd.refs.get("input"), float):
                f, t = nd.refs["input"], nd.refs.get("thickness", 0.0)
                side = nd.spec.get("side", "inside")
                inner = build(f) or leaf(f)
                if side == "inside":       # max(f, -(f + t))
                    return _Node("max", [inner, leaf(f, -1.0, (t, 1.0))])
                if side == "outside":      # max(-f, f - t)
                    return _Node("max", [_negate(inner), leaf(f, 1.0, (t, -1.0))])
                return None
            return None

        root = build(name or g.output)
        if root is None:
            return None
        tree = cls(root)

        def ev(P):
            memo = {}
            P = np.asarray(P, float).reshape(-1, 3)
            return np.stack([lf.sign * lf.fn(P, memo) for lf in tree.leaves])
        tree._evaluator = ev
        return tree

    @classmethod
    def from_callables(cls, spec):
        """A tree from nested tuples: ("min"|"max", [kids]) or a callable
        P -> values (a leaf), ("neg", kid) to negate. For experiments."""
        def build(s):
            if callable(s):
                return _Leaf(lambda P, memo, s=s: np.asarray(s(P), float), 1.0, f"leaf{id(s)}")
            if s[0] == "neg":
                return _negate(build(s[1]))
            return _Node(s[0], [build(k) for k in s[1]])
        tree = cls(build(spec))

        def ev(P):
            P = np.asarray(P, float).reshape(-1, 3)
            return np.stack([lf.sign * lf.fn(P, {}) for lf in tree.leaves])
        tree._evaluator = ev
        return tree

    # -- evaluation --

    def values(self, P):
        """(n_leaves, n) signed leaf values, in chunks."""
        P = np.asarray(P, float).reshape(-1, 3)
        out = np.empty((len(self.leaves), len(P)))
        for i in range(0, len(P), 200_000):
            out[:, i:i + 200_000] = self._evaluator(P[i:i + 200_000])
        return out

    def combine(self, vals):
        """(field value, active leaf index) from leaf values."""
        def go(t):
            if isinstance(t, _Leaf):
                return vals[t.index], np.full(vals.shape[1], t.index)
            parts = [go(k) for k in t.kids]
            v = np.stack([p[0] for p in parts])
            lab = np.stack([p[1] for p in parts])
            j = np.argmin(v, axis=0) if t.kind == "min" else np.argmax(v, axis=0)
            idx = np.arange(v.shape[1])
            return v[j, idx], lab[j, idx]
        return go(self.root)

    def evaluate(self, P):
        return self.combine(self.values(P))


# ---- Gauss-Newton onto the meeting of several leaf surfaces ------------------------

def _leaf_grad(tree, P, which, h):
    """Values and central-difference gradients of the leaves `which`
    (n, k) at P (n, 3): (n, k) and (n, k, 3)."""
    n = len(P)
    stencil = np.concatenate([P] + [P + s * h * e for e in np.eye(3) for s in (1.0, -1.0)])
    vals = tree.values(stencil)                       # (L, 7n)
    rows = np.arange(n)[:, None]
    v0 = vals[:, :n].T[rows, which]                   # (n, k)
    g = np.empty(which.shape + (3,))
    for a in range(3):
        vp = vals[:, (1 + 2 * a) * n:(2 + 2 * a) * n].T[rows, which]
        vm = vals[:, (2 + 2 * a) * n:(3 + 2 * a) * n].T[rows, which]
        g[:, :, a] = (vp - vm) / (2 * h)
    return v0, g


def meet(tree, X, which, pitch, iters=12, tol=1e-7):
    """Gauss-Newton from X (n, 3) onto {leaf_i = 0 for i in which[n]}
    (minimum-norm steps, so the nearest point of the curve or corner).
    Returns (points, converged)."""
    X = np.array(X, float)
    which = np.asarray(which)
    h = 1e-4 * pitch
    ok = np.zeros(len(X), bool)
    todo = np.arange(len(X))
    for _ in range(iters):
        if not len(todo):
            break
        v, g = _leaf_grad(tree, X[todo], which[todo], h)
        done = np.abs(v).max(axis=1) < tol * pitch
        ok[todo[done]] = True
        todo, v, g = todo[~done], v[~done], g[~done]
        if not len(todo):
            break
        JJ = np.einsum("nka,nja->nkj", g, g)
        k = which.shape[1]
        JJ += 1e-12 * np.eye(k)[None]
        try:
            lam = np.linalg.solve(JJ, v[:, :, None])[:, :, 0]
        except np.linalg.LinAlgError:
            lam = np.einsum("nkj,nj->nk", np.linalg.pinv(JJ), v)
        step = np.einsum("nk,nka->na", lam, g)
        ln = np.linalg.norm(step, axis=1)
        step *= np.minimum(1.0, pitch / np.maximum(ln, 1e-300))[:, None]
        X[todo] -= step
    if len(todo):
        v, _ = _leaf_grad(tree, X[todo], which[todo], h)
        ok[todo] = np.abs(v).max(axis=1) < tol * pitch
    return X, ok


def _on_part(tree, X, pitch, tol=1e-5):
    f, _ = tree.evaluate(X)
    return np.abs(f) < tol * pitch


# ---- the mesh operation ------------------------------------------------------------

def _edges(F):
    """Unique edges (m, 2) sorted, and face -> edge index (n, 3): edge j of
    a face runs from corner j to corner j + 1."""
    e = np.stack([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]], axis=1).reshape(-1, 2)
    s = np.sort(e, axis=1)
    key = s[:, 0].astype(np.int64) * (int(F.max()) + 1) + s[:, 1]
    uk, inv = np.unique(key, return_inverse=True)
    E = np.stack([uk // (int(F.max()) + 1), uk % (int(F.max()) + 1)], axis=1)
    return E, inv.reshape(-1, 3)


def sharpen(V, F, tree, pitch, snap=0.25, reach=4.0, field=None, stats=None, lab=None, mask=None, clean=True,
            tangent=12.0):
    """Put the creases of `tree` back into the projected marching-cubes
    mesh (V, F). Returns (V, F, crease) with `crease` a bool mask of the
    vertices on a crease (snapped or inserted). `snap` (fraction of an
    edge) moves a vertex onto the crease instead of splitting next to it;
    `reach` (pitches) is how far a crease point may be from its edge.
    `field` (default the tree's own min/max) projects fallback points.
    Two leaves meeting at under `tangent` degrees are no crease."""
    V = np.asarray(V, float)
    F = np.asarray(F, np.int64)
    if field is None:
        def field(P):
            return tree.evaluate(P)[0]
    st = stats if stats is not None else {}
    nl = len(tree)
    if nl > 62:
        return V, F, np.zeros(len(V), bool)
    if lab is None:
        _, lab = tree.evaluate(V)
    if mask is None:
        mask = (np.int64(1) << lab.astype(np.int64))
    mask = np.array(mask, np.int64)
    E, fe = _edges(F)
    cross = (mask[E[:, 0]] & mask[E[:, 1]]) == 0
    # edges that are no crease: an end the tree does not describe (tree.ignore),
    # or two leaves that meet tangentially (a skin's outside lying on the body
    # it was shelled from: no edge to see, and Gauss-Newton is ill-posed there)
    skip = np.zeros(len(E), bool)
    ign = np.zeros(len(V), bool)
    if tree.ignore is not None and cross.any():
        ends = np.unique(E[cross].ravel())
        ign[ends] = np.asarray(tree.ignore(V[ends]), bool)
    # and near a hole (a thin wall the mesher left open, which the boolean
    # step repairs): leave the mesh round it as it was, so the repair holds
    ecount = np.bincount(fe.ravel(), minlength=len(E))
    if (ecount != 2).any():
        near = np.zeros(len(V), bool)
        near[E[ecount != 2].ravel()] = True
        for _r in range(3):
            near[F[near[F].any(axis=1)].ravel()] = True
        ign |= near
        st["nearHoles"] = int(near.sum())
    skip |= ign[E[:, 0]] | ign[E[:, 1]]
    ci = np.flatnonzero(cross & ~skip)
    if len(ci):
        mid = 0.5 * (V[E[ci, 0]] + V[E[ci, 1]])
        _, gr = _leaf_grad(tree, mid, np.stack([lab[E[ci, 0]], lab[E[ci, 1]]], 1), 1e-3 * pitch)
        gr /= np.maximum(np.linalg.norm(gr, axis=2, keepdims=True), 1e-300)
        skip[ci[(gr[:, 0] * gr[:, 1]).sum(1) > np.cos(np.radians(tangent))]] = True
    st["tangentOrIgnored"] = int((cross & skip).sum())
    cross &= ~skip
    st["crossingEdges"] = int(cross.sum())
    crease = (mask & (mask - 1)) != 0          # more than one label: on a crease already
    if not cross.any():
        return V, F, crease
    V = V.copy()
    home = V.copy()             # where each vertex goes back to if its crease tangles

    def crease_points(ei):
        """Crease point of edges ei (from their midpoints), and whether found."""
        a, b = E[ei, 0], E[ei, 1]
        mid = 0.5 * (V[a] + V[b])
        which = np.stack([lab[a], lab[b]], axis=1)
        X, ok = meet(tree, mid, which, pitch)
        # near the edge, and on the part (no third leaf owns it)
        ab = V[b] - V[a]
        t = np.clip(((X - V[a]) * ab).sum(1) / np.maximum((ab * ab).sum(1), 1e-300), 0, 1)
        dseg = np.linalg.norm(X - (V[a] + t[:, None] * ab), axis=1)
        ok &= dseg < reach * pitch
        ok[ok] &= _on_part(tree, X[ok], pitch)
        return X, ok, t

    # 1. snap vertices that sit almost on a crease
    ci = np.flatnonzero(cross)
    X, ok, t = crease_points(ci)
    snapped = 0
    if snap > 0 and ok.any():
        a, b = E[ci, 0], E[ci, 1]
        cand_v = np.r_[a[ok & (t < snap)], b[ok & (t > 1 - snap)]]
        cand_p = np.r_[lab[b[ok & (t < snap)]], lab[a[ok & (t > 1 - snap)]]]
        cand_t = np.r_[t[ok & (t < snap)], 1 - t[ok & (t > 1 - snap)]]
        if len(cand_v):
            # one partner label per vertex (else it is near a corner: leave it)
            order = np.lexsort((cand_t, cand_v))
            cand_v, cand_p, cand_t = cand_v[order], cand_p[order], cand_t[order]
            first = np.r_[True, cand_v[1:] != cand_v[:-1]]
            starts = np.flatnonzero(first)
            pmin = np.minimum.reduceat(cand_p, starts)
            pmax = np.maximum.reduceat(cand_p, starts)
            uv = cand_v[starts]
            # also every crossing edge of the vertex must have that partner
            single = pmin == pmax
            uv, partner, ut = uv[single], pmin[single], cand_t[starts][single]
            ea, eb = E[ci, 0], E[ci, 1]
            # every crossing edge of the vertex must have that partner (else
            # it is near a corner: leave it to the split)
            lo_p = np.full(len(V), nl, np.int64)
            hi_p = np.full(len(V), -1, np.int64)
            for x, y in ((ea, eb), (eb, ea)):
                np.minimum.at(lo_p, x, lab[y])
                np.maximum.at(hi_p, x, lab[y])
            keep = (lo_p[uv] == hi_p[uv]) & (lo_p[uv] == partner)
            uv, partner, ut = uv[keep], partner[keep], ut[keep]
            if len(uv):
                Xs, oks = meet(tree, V[uv], np.stack([lab[uv], partner], axis=1), pitch)
                oks &= np.linalg.norm(Xs - V[uv], axis=1) < 0.75 * pitch
                if oks.any():
                    oks[oks] &= _on_part(tree, Xs[oks], pitch)
                uv, partner, Xs, ut = uv[oks], partner[oks], Xs[oks], ut[oks]
                # no crossing edge may have both ends snapped: keep the nearer
                take = np.zeros(len(V), bool)
                take[uv] = True
                score = np.full(len(V), np.inf)
                score[uv] = ut
                both = take[ea] & take[eb]
                drop = np.where(score[ea[both]] > score[eb[both]], ea[both], eb[both])
                take[drop] = False
                # and after dropping, never two snapped ends again (greedy, repeat)
                for _ in range(4):
                    both = take[ea] & take[eb]
                    if not both.any():
                        break
                    take[np.where(score[ea[both]] > score[eb[both]], ea[both], eb[both])] = False
                sel = take[uv]
                uv, partner, Xs = uv[sel], partner[sel], Xs[sel]
                V[uv] = Xs
                mask[uv] |= np.int64(1) << partner.astype(np.int64)
                crease[uv] = True
                snapped = len(uv)
    st["snapped"] = int(snapped)

    # 2. split the edges that still cross
    cross = ((mask[E[:, 0]] & mask[E[:, 1]]) == 0) & ~skip
    ci = np.flatnonzero(cross)
    X, ok, _ = crease_points(ci)
    # fallback: the midpoint on the field (where the mesh had it)
    from .fields import _project

    mfall = 0.5 * (V[E[ci, 0]] + V[E[ci, 1]])
    for _ in range(3):
        mfall = _project(field, mfall, pitch)
    X[~ok] = mfall[~ok]
    home = np.concatenate([home, mfall])
    st["split"] = int(len(ci))
    st["splitFallback"] = int((~ok).sum())
    nv = len(V)
    epos = np.full(len(E), -1, np.int64)
    epos[ci] = nv + np.arange(len(ci))
    V = np.concatenate([V, X])
    newc = np.zeros(len(V), bool)
    newc[:nv] = crease
    newc[nv:] = ok
    crease = newc

    # 3. re-cut every face by its split edges
    M = epos[fe]                                    # (nf, 3) new vertex on edge j, or -1
    c = (M >= 0).sum(axis=1)
    out = [F[c == 0]]
    # one split edge: (a, m, c), (m, b, c) with a->b the split edge
    idx = np.flatnonzero(c == 1)
    if len(idx):
        j = np.argmax(M[idx] >= 0, axis=1)
        r = np.arange(len(idx))
        a = F[idx, j]
        b = F[idx, (j + 1) % 3]
        cc = F[idx, (j + 2) % 3]
        m = M[idx, j]
        out += [np.stack([a, m, cc], 1), np.stack([m, b, cc], 1)]
        del r
    # two split edges: the lone corner and a quad
    idx = np.flatnonzero(c == 2)
    if len(idx):
        j = np.argmin(M[idx] >= 0, axis=1)          # the unsplit edge b->c
        b = F[idx, j]
        cc = F[idx, (j + 1) % 3]
        a = F[idx, (j + 2) % 3]
        m_ca = M[idx, (j + 1) % 3]
        m_ab = M[idx, (j + 2) % 3]
        out.append(np.stack([a, m_ab, m_ca], 1))
        d1 = np.linalg.norm(V[m_ab] - V[cc], axis=1)
        d2 = np.linalg.norm(V[b] - V[m_ca], axis=1)
        s = d1 <= d2
        out += [np.stack([m_ab, b, cc], 1)[s], np.stack([m_ab, cc, m_ca], 1)[s],
                np.stack([m_ab, b, m_ca], 1)[~s], np.stack([b, cc, m_ca], 1)[~s]]
    # three: a centre at the corner where the three leaves meet
    idx = np.flatnonzero(c == 3)
    st["corners"] = int(len(idx))
    if len(idx):
        tri = F[idx]
        which = lab[tri]
        cen0 = V[M[idx]].mean(axis=1)
        distinct = (which[:, 0] != which[:, 1]) & (which[:, 1] != which[:, 2]) & (which[:, 0] != which[:, 2])
        cen = cen0.copy()
        good = np.zeros(len(idx), bool)
        if distinct.any():
            Xc, okc = meet(tree, cen0[distinct], which[distinct], pitch)
            okc &= np.linalg.norm(Xc - cen0[distinct], axis=1) < reach * pitch
            if okc.any():
                okc[okc] &= _on_part(tree, Xc[okc], pitch)
            sub = np.flatnonzero(distinct)
            cen[sub[okc]] = Xc[okc]
            good[sub[okc]] = True
        if (~good).any():
            from .fields import _project

            q = cen[~good]
            for _ in range(3):
                q = _project(field, q, pitch)
            cen[~good] = q
        q = cen0
        for _ in range(3):
            q = _project(field, q, pitch)
        home = np.concatenate([home, q])
        k = len(V) + np.arange(len(idx))
        V = np.concatenate([V, cen])
        crease = np.r_[crease, good]
        v0, v1, v2 = tri[:, 0], tri[:, 1], tri[:, 2]
        m01, m12, m20 = M[idx, 0], M[idx, 1], M[idx, 2]
        out += [np.stack(t, 1) for t in ((v0, m01, k), (m01, v1, k), (v1, m12, k),
                                          (m12, v2, k), (v2, m20, k), (m20, v0, k))]
    F2 = np.concatenate(out).astype(np.int64)
    if clean:
        V, F2, _, crease, used = cleanup(V, F2, tree, label_masks(tree, V, pitch), crease, pitch, stats=st)
        home = home[used]
        V, crease = untangle(V, F2, tree, crease, home, pitch, stats=st)
    st["creaseVertices"] = int(crease.sum())
    return V, F2, crease


def label_masks(tree, V, pitch, tol=1e-5):
    """Bit mask per vertex of the leaves it lies on: its active leaf, and
    every other leaf within tol * pitch of zero there (a crease vertex is on
    two, a corner on three)."""
    vals = tree.values(V)
    _, lab = tree.combine(vals)
    near = np.abs(vals) < tol * pitch
    bits = (np.int64(1) << np.arange(len(tree), dtype=np.int64))[:, None]
    return (np.int64(1) << lab.astype(np.int64)) | np.bitwise_or.reduce(np.where(near, bits, 0), axis=0)


def _face_normals(V, F):
    n = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    return n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-300)


# ---- clean-up: collapse the slivers and turned-over triangles the split leaves ------

def _min_angle(T):
    out = np.full(len(T), 180.0)
    for i in range(3):
        u = T[:, (i + 1) % 3] - T[:, i]
        w = T[:, (i + 2) % 3] - T[:, i]
        c = (u * w).sum(-1) / np.maximum(np.linalg.norm(u, axis=-1) * np.linalg.norm(w, axis=-1), 1e-300)
        out = np.minimum(out, np.degrees(np.arccos(np.clip(c, -1, 1))))
    return out


def side_normals(tree, V, F, mask):
    """The true outward normal each face should have: the gradient of the
    leaf its three vertices share (None-like zero where they share none)."""
    common = mask[F[:, 0]] & mask[F[:, 1]] & mask[F[:, 2]]
    has = common != 0
    leaf = np.zeros(len(F), np.int64)
    c = common[has]
    # lowest set bit
    leaf[has] = np.log2(c & -c).astype(np.int64)
    cen = V[F].mean(1)
    out = np.zeros((len(F), 3))
    if has.any():
        _, g = _leaf_grad(tree, cen[has], leaf[has][:, None], 1e-5)
        g = g[:, 0]
        out[has] = g / np.maximum(np.linalg.norm(g, axis=1, keepdims=True), 1e-300)
    return out, has


def cleanup(V, F, tree, mask, crease, pitch, min_angle=8.0, passes=6, stats=None):
    """Collapse, near the creases, every triangle that is turned over against
    its surface's true normal or thinner than `min_angle`: its shortest
    collapsible edge goes, the crease end kept (two crease ends only on the
    same curve), and only where the two ends share exactly the two opposite
    vertices (so the mesh stays a closed manifold) and no remaining triangle
    turns over. Returns (V, F, mask, crease)."""
    st = stats if stats is not None else {}
    V = np.asarray(V, float)
    F = np.asarray(F, np.int64).copy()
    mask = np.asarray(mask, np.int64).copy()
    crease = np.asarray(crease, bool).copy()
    done = 0
    for _ in range(passes):
        # the neighbourhood of the creases: faces with a vertex within two rings
        near = crease.copy()
        for _r in range(2):
            fn = near[F].any(axis=1)
            near[F[fn].ravel()] = True
        region = np.flatnonzero(near[F].any(axis=1))
        if not len(region):
            break
        T = V[F[region]]
        ma = _min_angle(T)
        nrm = _face_normals(V, F[region])
        sn, has = side_normals(tree, V, F[region], mask)
        flipped = has & ((nrm * sn).sum(1) < 0)
        bad = region[(ma < min_angle) | flipped]
        if not len(bad):
            break
        # vertex -> faces over the whole mesh, for the vertices involved
        inv = {}
        involved = set(F[bad].ravel().tolist())
        # one ring of faces around the involved vertices
        ring = np.flatnonzero(np.isin(F, list(involved)).any(axis=1))
        for fi in ring:
            for v in F[fi]:
                if v in involved:
                    inv.setdefault(v, set()).add(int(fi))
        alive = np.ones(len(F), bool)
        touched = set()
        changed = 0
        order = bad[np.argsort(ma[np.searchsorted(region, bad)])]
        for fi in order:
            if not alive[fi] or fi in touched:
                continue
            f = F[fi]
            edges = sorted(((np.linalg.norm(V[f[i]] - V[f[(i + 1) % 3]]), f[i], f[(i + 1) % 3]) for i in range(3)))
            for _len, a, b in edges:
                # keep: the crease end; both on a crease only with the same labels
                if crease[a] and crease[b] and mask[a] != mask[b]:
                    continue
                keep, gone = (a, b) if (crease[a] or not crease[b]) else (b, a)
                if keep not in inv or gone not in inv:
                    continue
                fk = {x for x in inv[keep] if alive[x]}
                fg = {x for x in inv[gone] if alive[x]}
                if fk & touched or fg & touched:
                    continue
                shared = fk & fg
                if len(shared) != 2:
                    continue
                nk = set(F[list(fk)].ravel().tolist()) - {keep}
                ng = set(F[list(fg)].ravel().tolist()) - {gone}
                opp = set(F[list(shared)].ravel().tolist()) - {keep, gone}
                if (nk & ng) != opp:
                    continue
                # the triangles that stay: gone -> keep, none may turn over
                moved = np.array(sorted(fg - shared), np.int64)
                if len(moved):
                    old = F[moved]
                    new = np.where(old == gone, keep, old)
                    n0 = _face_normals(V, old)
                    n1 = _face_normals(V, new)
                    if ((n0 * n1).sum(1) < 0.2).any() or (_min_angle(V[new]) < 1e-3).any():
                        continue
                    F[moved] = new
                for x in shared:
                    alive[x] = False
                for x in moved:
                    inv[keep].add(int(x))
                touched |= fk | fg
                changed += 1
                break
        F = F[alive]
        done += changed
        if not changed:
            break
    # drop the vertices nothing uses
    used = np.zeros(len(V), bool)
    used[F] = True
    new = np.cumsum(used) - 1
    st["collapsed"] = int(done)
    return V[used], new[F], mask[used], crease[used], used


def _seg_tri(p, q, T, eps=1e-9):
    """Does segment pq cross triangle T (strictly inside both)?"""
    d = q - p
    e1 = T[:, 1] - T[:, 0]
    e2 = T[:, 2] - T[:, 0]
    h = np.cross(d, e2)
    a = (e1 * h).sum(1)
    ok = np.abs(a) > 1e-14
    f = np.where(ok, 1.0 / np.where(ok, a, 1.0), 0.0)
    s = p - T[:, 0]
    u = f * (s * h).sum(1)
    qq = np.cross(s, e1)
    v = f * (d * qq).sum(1)
    t = f * (e2 * qq).sum(1)
    return ok & (u > eps) & (v > eps) & (u + v < 1 - eps) & (t > eps) & (t < 1 - eps)


def crossing_faces(V, F, faces):
    """Faces among `faces` that cross another face of the mesh (sharing no
    vertex with it): a self-intersection."""
    from scipy.spatial import cKDTree

    faces = np.asarray(faces, np.int64)
    if not len(faces):
        return faces
    C = V[F].mean(1)
    # a face's reach: its own circumradius plus a typical neighbour's
    # (the 99th percentile: the long faces are the crease's own, and those
    # are tested from their side)
    rad = np.linalg.norm(V[F] - C[:, None], axis=2).max(axis=1)
    lst = cKDTree(C).query_ball_point(C[faces], rad[faces] + np.percentile(rad, 99) + 1e-12)
    A = np.repeat(faces, [len(x) for x in lst])
    B = np.concatenate([np.asarray(x, np.int64) for x in lst]) if len(lst) else np.zeros(0, np.int64)
    # only faces whose bounding spheres overlap can cross
    keep = (A != B) & (np.linalg.norm(C[A] - C[B], axis=1) < rad[A] + rad[B] + 1e-12)
    A, B = A[keep], B[keep]
    share = (F[A][:, :, None] == F[B][:, None, :]).any(axis=(1, 2))
    A, B = A[~share], B[~share]
    hit = np.zeros(len(A), bool)
    for X, Y in ((A, B), (B, A)):
        T = V[F[Y]]
        for k in range(3):
            hit |= _seg_tri(V[F[X, k]], V[F[X, (k + 1) % 3]], T)
    return np.unique(np.r_[A[hit], B[hit]])


def untangle(V, F, tree, crease, home, pitch, rounds=4, stats=None):
    """Send crease vertices back to where the mesh had them (`home`) wherever
    the sharpened crease crosses another part of the mesh or turns a
    triangle over: a crease is kept only where it does no harm."""
    V = np.array(V, float)
    crease = np.array(crease, bool)
    back = 0
    near = None
    for _ in range(rounds):
        # only where a crease vertex moved: elsewhere the mesh is as it was;
        # after the first round, only around what the last round sent back
        moved = crease & (np.linalg.norm(V - home, axis=1) > 0.05 * pitch)
        sel = moved[F].any(axis=1)
        if near is not None:
            sel &= near[F].any(axis=1)
        cf = np.flatnonzero(sel)
        if not len(cf):
            break
        bad = crossing_faces(V, F, cf)
        mask = label_masks(tree, V, pitch)
        sn, has = side_normals(tree, V, F[cf], mask)
        flip = cf[has & ((_face_normals(V, F[cf]) * sn).sum(1) < 0)]
        bad = np.union1d(bad, flip)
        verts = np.unique(F[bad].ravel())
        verts = verts[crease[verts]]
        if not len(verts):
            break
        V[verts] = home[verts]
        crease[verts] = False
        back += len(verts)
        # the faces that could cross now: within a few rings of those moved back
        near = np.zeros(len(V), bool)
        near[verts] = True
        for _r in range(3):
            near[F[near[F].any(axis=1)].ravel()] = True
    if stats is not None:
        stats["untangled"] = int(back)
    return V, crease
