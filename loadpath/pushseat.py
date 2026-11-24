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

__all__ = ["patches", "hold", "resolve", "solve", "solve_cases", "Seat", "SeatedSpread", "worst_stress"]

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


LIFTS = ("a load lifts the part off its seat (supports[].push) with nothing else to hold it: "
         "a seat only pushes. Hold the part where its fasteners are.")


class Seat:
    """The seats' contact on one sprung stiffness, in the space of the
    patches' gaps: the unit pushes `ps`, their displacements `ws` on the
    sprung stiffness (hold), the springs `ks` (N/mm per sprung patch) and,
    when `ps` are pieces of the sprung patches (refine), each piece's share
    of each (pieces x patches).

    Every question about the contact is a small one here: for the gaps g0
    a load opens on the springs alone (g0 = P u0), `push` gives the gaps,
    the pushes mu that make the answer (u = u0 + mu . w) and the seats'
    forces. The load's compliance with the seat pushing is f.u0 + mu.g0
    (f.w_j = p_j.u0: the sprung stiffness is symmetric), and its gradient in
    the load is 2u: what the directions need (SeatedSpread)."""

    def __init__(self, ps, ws, ks, share=None):
        self.p = np.array(ps)
        self.w = np.array(ws)
        ks = np.asarray(ks, float)
        self.share = np.eye(len(ps)) if share is None else np.asarray(share, float)
        c = self.p @ self.w.T
        c = 0.5 * (c + c.T)
        self.a = np.linalg.pinv(c, hermitian=True)
        self.kd = (self.share * ks) @ self.share.T
        h = self.a - self.kd
        d, v = np.linalg.eigh(0.5 * (h + h.T))
        floor = FLOOR * float(ks.mean()) if ks.size and ks.mean() > 0 else 1e-12 * max(float(np.abs(d).max()), 1e-300)
        self.d, self.v, self.dc = d, v, np.maximum(d, floor)
        self.m = np.sqrt(self.dc)[:, None] * v.T          # m'm = the stiffness, floored

    def __len__(self):
        return len(self.p)

    def gaps(self, u):
        return self.p @ u

    def push(self, g0, strict: bool = False):
        """(gaps, mu, lambda, lifts) for the gaps `g0` the load opens on the
        springs alone. `lifts`: the load takes the part off its seat with
        nothing else to hold it (raised with `strict`)."""
        from scipy.optimize import nnls

        b = self.a @ g0
        g, _ = nnls(self.m, (self.v.T @ b) / np.sqrt(self.dc), maxiter=50 * len(b))
        # what the floor carried: nothing, unless the part is going somewhere
        lost = self.v @ ((self.dc - self.d) * (self.v.T @ g))
        lifts = bool(np.linalg.norm(lost) > 1e-2 * max(np.linalg.norm(b), 1e-300))
        if strict and lifts:
            raise ValueError(LIFTS)
        mu = self.a @ (g - g0)
        lam = np.maximum(mu - self.kd @ g, 0.0)
        return g, mu, lam, lifts

    def apply(self, u0, mu):
        return u0 + mu @ self.w


def resolve(u0s, ws, ps, ks, strict: bool = True, share=None):
    """Displacements with the seats pushing: for each u0 (the answer on the
    springs `ks`, N/mm per patch, alone), (u, lambda, mu) with lambda (N per
    patch) the seats' forces and mu the pushes that give u on the sprung
    stiffness (u = u0 + mu . w). `strict`: a load that lifts the part off
    its seats with nothing else to hold it raises ValueError. `share`: when
    `ps` are pieces of the patches the springs are on (refine), each piece's
    share of each sprung patch (pieces x patches)."""
    seat = Seat(ps, ws, ks, share)
    out = []
    for u0 in u0s:
        _, mu, lam, _ = seat.push(seat.gaps(u0), strict)
        out.append((seat.apply(u0, mu), lam, mu))
    return out


class SeatedSpread:
    """A load case with a force of unknown direction (directions.py) on a
    part with seats. A seat lets go where the force lifts it, so the answer
    is not the sum of the answers along three axes: it is the contact solved
    for each direction. Every direction's is a small problem in the gaps
    (Seat): the sprung answers u0 (the case's fixed part, or None) and ub
    (the uncertain force along x, y, z) give the gaps g0(d) = P u0 + d . P ub,
    and the answer is u(d) = u0 + d . ub + mu(d) . w.

    The compliance with the seat pushing is a convex function of the force
    (twice the most of f.u - u'Ku/2 over the admissible u: a maximum of
    functions linear in f), and the force is linear in d, so over the sphere
    of directions (or a cap of it) its maximum sits where the ascent
    d <- grad c / |grad c| (a minorize-maximize step; grad c = 2 B'u)
    stops: started from the best of a spread of directions.

    `ref`: (mu_ref, u_ref), a push already solved on the current stiffness
    (u_ref = K'^-1 P'mu_ref) while `seat.w` are older: u(d) is then
    u0 + d . ub + u_ref + (mu - mu_ref) . w, exact at mu_ref (the optimizer
    reads the seats' compliance every few solves, simp._solve_seated)."""

    def __init__(self, seat: Seat, f0, u0, fb, ub, spread, ref=None):
        self.seat = seat
        self.spread = spread
        self.f0, self.u0, self.fb, self.ub = f0, u0, list(fb), list(ub)
        self.ref = ref
        self.a = np.array([[self.fb[i] @ self.ub[j] for j in range(3)] for i in range(3)])
        self.a = 0.5 * (self.a + self.a.T)
        if u0 is None:
            self.c00, self.g, self.G0 = 0.0, np.zeros(3), np.zeros(len(seat))
        else:
            self.c00 = float(f0 @ u0)
            self.g = np.array([f0 @ self.ub[j] for j in range(3)])
            self.G0 = seat.gaps(u0)
        self.Gb = np.array([seat.gaps(u) for u in self.ub])           # 3 x patches

    def at(self, d, strict: bool = False):
        """(compliance, gradient in d, mu, lambda, lifts) for direction d."""
        d = np.asarray(d, float)
        g0 = self.G0 + d @ self.Gb
        _, mu, lam, lifts = self.seat.push(g0, strict)
        c = self.c00 + 2.0 * self.g @ d + d @ self.a @ d + mu @ g0
        grad = 2.0 * (self.a @ d + self.g + self.Gb @ mu)
        return float(c), grad, mu, lam, lifts

    def u(self, d, mu):
        u = sum(d[j] * self.ub[j] for j in range(3))
        if self.u0 is not None:
            u = u + self.u0
        if self.ref is not None:
            mu = mu - self.ref[0]
            u = u + self.ref[1]
        return self.seat.apply(u, mu)

    def f(self, d):
        f = sum(d[j] * self.fb[j] for j in range(3))
        return f if self.f0 is None else f + self.f0

    def samples(self, n: int = 200, start=None):
        """[(d, c, mu, lambda, lifts)] over a spread of `n` directions (the
        cap's), the ascent's end from the best few (and from `start`) last;
        the last one is the worst."""
        from .directions import _cap_samples, _to_cap

        sp = self.spread
        axis = np.asarray(sp.nominal, float)
        dirs = list(_cap_samples(sp, n))
        if start is not None:
            dirs.append(np.asarray(start, float))
        rows = [(d, *self.at(d)) for d in dirs]
        order = np.argsort([-r[1] for r in rows])
        best = None
        for k in list(order[:3]) + ([len(rows) - 1] if start is not None else []):
            d, c, grad, mu, lam, lifts = rows[k]
            for _ in range(60):
                n_ = np.linalg.norm(grad)
                if n_ < 1e-300:
                    break
                dn = _to_cap(grad / n_, axis, sp.cos_half)
                cn, gn, mun, lamn, lfn = self.at(dn)
                if cn <= c * (1.0 + 1e-10) or np.linalg.norm(dn - d) < 1e-9:
                    if cn > c:
                        d, c, grad, mu, lam, lifts = dn, cn, gn, mun, lamn, lfn
                    break
                d, c, grad, mu, lam, lifts = dn, cn, gn, mun, lamn, lfn
            if best is None or c > best[1]:
                best = (d, c, grad, mu, lam, lifts)
        out = [(r[0], r[1], r[3], r[4], r[5]) for r in rows]
        out.append((best[0], best[1], best[3], best[4], best[5]))
        return out

    def basis(self):
        """The vectors every direction's answer is a combination of: the
        constant part, ub along x, y, z, and the pushes' w; with the
        coefficients q(d, mu) = (1, d, mu) (coefficients)."""
        const = self.u0 if self.u0 is not None else np.zeros_like(self.ub[0])
        if self.ref is not None:
            const = const + self.ref[1] - self.ref[0] @ self.seat.w
        return const

    @staticmethod
    def coefficients(rows):
        return np.array([np.concatenate([[1.0], r[0], r[2]]) for r in rows])

    def combine(self, q):
        """sum_k q_k basis_k without forming the basis."""
        q = np.asarray(q, float)
        return q[0] * self.basis() + q[1] * self.ub[0] + q[2] * self.ub[1] + q[3] * self.ub[2] \
            + q[4:] @ self.seat.w

    def worst(self, smooth: float, start=None, n: int = 120):
        """The case judged in its worst direction for the optimizer
        (directions.Worst): the exact worst compliance and, for the
        sensitivity, a soft maximum over the directions (directions.SMOOTH),
        as strain energies of a few combinations. The worst one's mu is
        kept (`self.last`) for the next solve's reference push."""
        from .directions import Worst

        rows = self.samples(n, start)
        d, c, mu = rows[-1][0], rows[-1][1], rows[-1][2]
        cs = np.array([r[1] for r in rows])
        w = np.exp(smooth * (cs - c) / max(abs(c), 1e-300))
        w /= w.sum()
        q = self.coefficients(rows)
        gm = (q * w[:, None]).T @ q
        m, vec = np.linalg.eigh(gm)
        keep = np.argsort(-m)[:8]
        keep = [j for j in keep if m[j] > 1e-6 * m.max()]
        energy = [(float(m[j]), self.combine(vec[:, j])) for j in keep]
        u_mean = self.combine(w @ q)
        self.last = (d, mu)
        return Worst(self.u(d, mu), c, d, self.f(d), energy, u_mean)


# von Mises squared as a quadratic form on (sx, sy, sz, txy, tyz, tzx)
def _vm2(s):
    from .directions import _VM

    return np.einsum("...j,jl,...l->...", s, _VM, s)


def worst_stress(case: SeatedSpread, read, mask, scale=None, n: int = 300, strict: bool = False):
    """The direction of `case`'s uncertain force that raises the peak von
    Mises over the nodes in `mask` most, with the seat solved in each
    direction: (d, mu, peak).

    The stress is linear in the answer, and every direction's answer is a
    combination of the few vectors of case.basis (the constant part, three
    axes, the pushes), with coefficients q(d) that the contact makes
    piecewise linear in d. So: q at a spread of directions (and the
    compliance's worst), the vectors that span those answers (a truncated
    SVD of the q's, each column weighed by its vector's size), the stress
    of each (`read(u)` -> (n, 6), times `scale` per node), then the peak
    over directions and nodes, at the nodes whose bound (the triangle
    inequality on von Mises) can reach the best found at a few directions.
    `strict`: a direction that lifts the part off with nothing else to hold
    it raises ValueError (verify)."""
    rows = case.samples(n)
    if strict and any(r[4] for r in rows):
        raise ValueError(LIFTS + " A force of unknown direction lifts it in some direction.")
    q = SeatedSpread.coefficients(rows)
    norms = np.array([np.linalg.norm(case.basis()), *(np.linalg.norm(u) for u in case.ub),
                      *np.linalg.norm(case.seat.w, axis=1)])
    norms = np.maximum(norms, 1e-300)
    qs = q * norms
    a, s, bt = np.linalg.svd(qs, full_matrices=False)
    r = int(min(np.sum(s > 1e-3 * s[0]), 24))
    coef = a[:, :r] * s[:r]                                   # samples x r
    stress = []
    for k in range(r):
        sk = read(case.combine(bt[k] / norms))
        stress.append(sk if scale is None else sk * scale)
    idx = np.nonzero(mask)[0]
    if idx.size == 0:
        idx = np.arange(stress[0].shape[0])
    sk = np.stack([x[idx] for x in stress], axis=1)          # (m, r, 6)
    # a lower bound at a few directions: the compliance's worst and the corners of the q's
    cs = np.array([row[1] for row in rows])
    probe = sorted(set(np.argsort(-cs)[:6].tolist() + np.argmax(np.abs(coef), axis=0).tolist()))
    low = 0.0
    for p_ in probe:
        low = max(low, float(_vm2(np.einsum("k,mkj->mj", coef[p_], sk)).max()))
    vmk = np.sqrt(np.maximum(_vm2(sk), 0.0))                  # (m, r)
    bound = vmk @ np.abs(coef).max(axis=0)
    keep = bound ** 2 >= low * (1.0 - 1e-9)
    sk = sk[keep]
    best, bs = 0.0, int(np.argmax(cs))
    for k0 in range(0, len(coef), 32):
        sig = np.einsum("sk,mkj->smj", coef[k0:k0 + 32], sk)
        v = _vm2(sig).max(axis=1)
        j = int(np.argmax(v))
        if v[j] > best:
            best, bs = float(v[j]), k0 + j
    d, mu = rows[bs][0], rows[bs][2]
    return np.asarray(d, float), mu, float(np.sqrt(max(best, 0.0)))


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
    stiff, us, lam, _ = solve_cases(inner, rhs, ps, ks, strict, xyz)
    return stiff, us, lam


def solve_cases(inner, rhs, ps, ks, strict: bool = True, xyz=None, layout=None, spreads=None):
    """solve, with the cases' layout (directions.expand): a case whose force
    has no known direction (`spreads[ci]`) is not resolved rhs by rhs (the
    seat lets go where the force lifts it, so the axes do not add up): its
    rhs come back as solved on the springs, and the case as a SeatedSpread
    that solves the contact per direction. Returns (stiffness, displacements
    per rhs, seat forces per rhs or None, {case: SeatedSpread}). The patches
    cut finer are the ones that bear in any case and direction."""
    rhs = list(rhs)
    ps = list(ps)
    if layout is None:
        layout = [(i,) for i in range(len(rhs))]
    spreads = spreads or [None] * len(layout)
    stiff, us = inner(rhs + ps)
    us = list(us)
    u0s, ws = us[:len(rhs)], us[len(rhs):]
    share = np.eye(len(ps))

    def judge():
        seat = Seat(ps, ws, ks, share)
        done, cases, bear = {}, {}, np.zeros(len(ps))
        for ci, idx in enumerate(layout):
            if len(idx) == 1:
                i = idx[0]
                _, mu, lam, _ = seat.push(seat.gaps(u0s[i]), strict)
                done[i] = (seat.apply(u0s[i], mu), lam)
                bear = np.maximum(bear, lam / max(lam.sum(), 1e-300))
                continue
            i0 = idx[0]
            case = SeatedSpread(seat, rhs[i0] if i0 is not None else None, u0s[i0] if i0 is not None else None,
                                [rhs[i] for i in idx[1:]], [u0s[i] for i in idx[1:]], spreads[ci])
            rows = case.samples(60)
            if strict and any(r[4] for r in rows):
                raise ValueError(LIFTS + " A force of unknown direction lifts it in some direction.")
            for r in rows:
                bear = np.maximum(bear, r[3] / max(r[3].sum(), 1e-300))
            cases[ci] = case
        return done, cases, bear

    done, cases, bear = judge()
    for _ in range(REFINE_ROUNDS if xyz is not None else 0):
        cut = {}
        for j in np.argsort(-bear):
            if bear[j] <= 1e-6 * max(bear.sum(), 1e-300) or 4 * (len(cut) + 1) > REFINE_MOST:
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
        done, cases, bear = judge()
    # the forces on the patches given: each piece's go to the patch it was cut from
    back = (share > 0).astype(float)
    out_u = [done[i][0] if i in done else u0s[i] for i in range(len(rhs))]
    out_l = [done[i][1] @ back if i in done else None for i in range(len(rhs))]
    return stiff, out_u, out_l, cases
