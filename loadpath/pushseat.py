"""
A support that only pushes: a face resting on a wall, a table or a mating
part (supports[].push).

Held "fix: all", a bolted face is glued to what it sits on: where the load
pries it off, the glue pulls, and the part passes on a hold it does not have.
A seat carries pressure only. The part's normal displacement on the seat may
not go into it (gap >= 0), the seat's force is a push (lambda >= 0), and it
pushes only where it touches (lambda * gap = 0). Friction is left out: the
fasteners carry what slides and what lifts, which is the safe side.

The contact is solved exactly on the linear model. The seat's nodes are
free; its force is an unknown on a few patches of the face (constant
pressure over each, p_j the unit push on patch j).

A part may have nothing but its seat against some motion (it lies on a
table; one screw line lets it rock), so the stiffness without the seat can
be singular. Each patch therefore gets a spring k_j on its own mean normal
displacement g_j = p_j . u for the solves (K' = K + sum_j k_j p_j p_j', hold),
and the spring is taken out again in closed form: with u0 = K'^-1 f and
w_j = K'^-1 p_j, every answer is u = u0 + sum_j mu_j w_j with
mu_j = lambda_j + k_j g_j, the gaps are g = g0 + C mu (C_ij = p_i . w_j),
and the part's energy in terms of the gaps is
1/2 (g - g0)' A (g - g0) - 1/2 sum_j k_j g_j^2 with A = C^-1. Its minimum
over g >= 0 has lambda = A (g - g0) - k g >= 0 and lambda_j g_j = 0: the
three conditions, whatever the springs were. A - diag(k) is the part's own
stiffness seen from the seat; it is zero along a motion only the seat stops,
which is fine while the load presses that way and a part that flies off
otherwise (resolve says so).

Units: mm, N.
"""
from __future__ import annotations

import numpy as np

__all__ = ["patches", "hold", "resolve", "solve"]

# a patch's spring for the solves, over the modulus times its width: soft beside
# the material under the patch (about E * width), far from nothing. The answer
# does not depend on it (see above); the solver's conditioning does.
HOLD = 0.1
# the part's own stiffness seen from the seat is kept above this share of the
# springs' (the solves' accuracy leaves noise of that order where it is zero)
FLOOR = 1e-7

MAX_PATCHES = 48


def _outline(pts, normal, most: int):
    """Indices of up to `most` of the points that span the face's outline:
    the corners of its convex hull across `normal`, fewest-area-lost first
    when there are more than `most`."""
    from scipy.spatial import ConvexHull, QhullError

    n = np.asarray(normal, float)
    a = np.cross(n, [1.0, 0.0, 0.0] if abs(n[0]) < 0.9 else [0.0, 1.0, 0.0])
    a /= np.linalg.norm(a)
    q = np.stack([pts @ a, pts @ np.cross(n, a)], axis=1)
    try:
        ring = list(ConvexHull(q).vertices)
    except (QhullError, ValueError):
        # a line of nodes (or one): its two ends
        t = q @ (q.max(axis=0) - q.min(axis=0))
        return sorted({int(np.argmin(t)), int(np.argmax(t))})

    def lost(k):
        o, b, c = q[ring[k - 1]], q[ring[k]], q[ring[(k + 1) % len(ring)]]
        return abs((b[0] - o[0]) * (c[1] - o[1]) - (b[1] - o[1]) * (c[0] - o[0]))

    while len(ring) > max(most, 3):
        ring.pop(min(range(len(ring)), key=lost))
    return [int(k) for k in ring]


def patches(seats, xyz, n_dof, most: int = MAX_PATCHES):
    """Unit pushes on the seats' patches: a list of force vectors (n_dof),
    each 1 N along its seat's normal spread over the patch's nodes by their
    share of the face. `seats`: [(node indices, normal (3,), weights)].

    A seat can push anywhere inside the outline of its patches' centres, and
    a part tips about that outline's edge. So the corners of the face's own
    outline are small patches of their own (a rectangle's four corners, a
    disc's rim in a dozen points), and squares fill the rest: the part tips
    where the face ends, not a patch's half width inside it."""
    out = []
    per = max(1, most // max(len(seats), 1))
    for idx, n, w in seats:
        idx, w = np.asarray(idx), np.asarray(w, float)
        if not len(idx):
            continue
        keys = np.full(len(idx), -1)
        if len(idx) > per:
            # a corner is its node and the nodes beside it: one node alone can hang off a sliver of a cut cell
            from scipy.spatial import cKDTree

            tree = cKDTree(xyz[idx])
            near = 1.5 * float(np.median(tree.query(xyz[idx], k=2)[0][:, 1]))
            corners = _outline(xyz[idx], n, min(12, per // 3))
            for k, c in enumerate(corners):
                group = np.array(tree.query_ball_point(xyz[idx[c]], near))
                keys[group[keys[group] < 0]] = k
        else:
            corners = range(len(idx))
            keys[:] = np.arange(len(idx))
        rest = keys < 0
        room = per - len(corners)
        if rest.any():
            # a grid of squares over the face, as coarse as it takes
            pts = xyz[idx[rest]] - xyz[idx].min(axis=0)
            size = float(pts.max()) / max(np.sqrt(room), 1.0)
            while True:
                cell = np.floor(pts / max(size, 1e-12)).astype(int)
                _, sq = np.unique(cell, axis=0, return_inverse=True)
                sq = np.asarray(sq).ravel()
                if sq.max() + 1 <= max(room, 1):
                    break
                size *= 1.25
            keys[rest] = len(corners) + sq
        for k in range(int(keys.max()) + 1):
            m = keys == k
            p = np.zeros(n_dof)
            share = w[m] / w[m].sum()
            for a in range(3):
                p[3 * idx[m] + a] = share * n[a]
            out.append(p)
    return out


def hold(ps, xyz, pitch: float):
    """The springs that hold the seats' patches for the solves, at unit
    modulus: (k per patch in mm, sparse sum_j k_j p_j p_j')."""
    import scipy.sparse as sp

    ks, rows, cols, vals = [], [], [], []
    for p in ps:
        dof = np.nonzero(p)[0]
        nodes = np.unique(dof // 3)
        width = float(np.ptp(xyz[nodes], axis=0).max()) if len(nodes) > 1 else 0.0
        k = HOLD * max(width, pitch)
        ks.append(k)
        v = p[dof]
        rows.append(np.repeat(dof, len(dof)))
        cols.append(np.tile(dof, len(dof)))
        vals.append(k * np.outer(v, v).ravel())
    if not ps:
        return np.zeros(0), None
    n = len(ps[0])
    s = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n, n))
    return np.array(ks), s


def resolve(u0s, ws, ps, ks, strict: bool = True, share=None):
    """Displacements with the seats pushing: for each u0 (the answer on the
    springs `ks`, N/mm per patch, alone), (u, lambda, mu) with lambda (N per
    patch) the seats' forces and mu the pushes that give u on the sprung
    stiffness (u = u0 + mu . w). `strict`: a load that lifts the part off
    its seats with nothing else to hold it raises ValueError. `share`: when
    `ps` are pieces of the patches the springs are on (refine), each piece's
    share of each sprung patch (pieces x patches)."""
    from scipy.optimize import nnls

    p = np.array(ps)
    w = np.array(ws)
    ks = np.asarray(ks, float)
    c = p @ w.T
    c = 0.5 * (c + c.T)
    a = np.linalg.pinv(c, hermitian=True)
    kd = np.diag(ks) if share is None else (share * ks) @ share.T
    h = a - kd
    d, v = np.linalg.eigh(0.5 * (h + h.T))
    dc = np.maximum(d, FLOOR * float(ks.mean()))
    m = np.sqrt(dc)[:, None] * v.T                       # m'm = the stiffness, floored
    out = []
    for u0 in u0s:
        g0 = p @ u0
        b = a @ g0
        g, _ = nnls(m, (v.T @ b) / np.sqrt(dc), maxiter=50 * len(c))
        if strict:
            # what the floor carried: nothing, unless the part is going somewhere
            lost = v @ ((dc - d) * (v.T @ g))
            if np.linalg.norm(lost) > 1e-2 * max(np.linalg.norm(b), 1e-300):
                raise ValueError("a load lifts the part off its seat (supports[].push) with nothing else to hold it: "
                                 "a seat only pushes. Hold the part where its fasteners are.")
        mu = a @ (g - g0)
        lam = np.maximum(mu - kd @ g, 0.0)
        out.append((u0 + mu @ w, lam, mu))
    return out


def _split(p, xyz):
    """A patch's unit push cut in up to four across its two widest
    directions: [(unit push of the piece, its share of the patch)]."""
    dof = np.nonzero(p)[0]
    nodes = np.unique(dof // 3)
    if len(nodes) < 8:
        return []
    pts = xyz[nodes]
    wide = np.argsort(-np.ptp(pts, axis=0))[:2]
    part = np.zeros(len(nodes), int)
    for k, ax in enumerate(wide):
        if np.ptp(pts[:, ax]) > 0:
            part += (pts[:, ax] > np.median(pts[:, ax])).astype(int) << k
    weight = np.linalg.norm(p.reshape(-1, 3)[nodes], axis=1)
    out = []
    for k in np.unique(part):
        m = part == k
        q = np.zeros_like(p)
        own = (3 * nodes[m][:, None] + np.arange(3)).ravel()
        frac = float(weight[m].sum() / weight.sum())
        q[own] = p[own] / frac
        out.append((q, frac))
    return out if len(out) > 1 else []


# rounds of cutting the patches that bear, and the most unit pushes a round may add
REFINE_ROUNDS = 2
REFINE_MOST = 32


def solve(inner, rhs, ps, ks, strict: bool = True, xyz=None):
    """`inner(rhs) -> (stiffness, displacements)` on the sprung stiffness
    (hold) with the seats' forces absent; returns the same with the seats
    pushing (`ps` their patches, `ks` the springs in N/mm), and their forces
    per rhs (per patch of `ps`).

    With `xyz` (node positions) the patches that bear are cut in four and
    solved again, REFINE_ROUNDS times: a part pressed on its seat near a rim
    bears on a band there, and one coarse patch beside it would let it tilt
    about the band. The springs stay on the patches given (the stiffness is
    the same one), each piece taking its share."""
    rhs = list(rhs)
    ps = list(ps)
    stiff, us = inner(rhs + ps)
    us = list(us)
    u0s, ws = us[:len(rhs)], us[len(rhs):]
    share = np.eye(len(ps))
    done = resolve(u0s, ws, ps, ks, strict)
    for _ in range(REFINE_ROUNDS if xyz is not None else 0):
        bear = np.max([d[1] for d in done], axis=0)
        cut = {}
        for j in np.argsort(-bear):
            if bear[j] <= 1e-6 * bear.sum() or 4 * (len(cut) + 1) > REFINE_MOST:
                break
            pieces = _split(ps[j], xyz)
            if pieces:
                cut[int(j)] = pieces
        if not cut:
            break
        new = [q for j in cut for q, _ in cut[j]]
        _, wn = inner(new)
        keep = [j for j in range(len(ps)) if j not in cut]
        rows = [share[j] for j in keep] + [frac * share[j] for j in cut for _, frac in cut[j]]
        ps = [ps[j] for j in keep] + new
        ws = [ws[j] for j in keep] + list(wn)
        share = np.array(rows)
        done = resolve(u0s, ws, ps, ks, strict, share=share)
    # the forces on the patches given: each piece's go to the patch it was cut from
    back = (share > 0).astype(float)
    return stiff, [d[0] for d in done], [d[1] @ back for d in done]
