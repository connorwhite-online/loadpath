"""
A support that only pushes: a face resting on a wall, a table or a mating
part (supports[].push).

Held "fix: all", a bolted face is glued to what it sits on: where the load
pries it off, the glue pulls, and the part passes on a hold it does not have.
A seat carries pressure only. The part's normal displacement on the seat may
not go into it (gap >= 0), the seat's force is a push (lambda >= 0), and it
pushes only where it touches (lambda * gap = 0).

Friction (supports[].friction, mu): Coulomb's, on each patch. The patch's
slip across the seat (its mean displacement along two directions in the
face) and the force across it (tau) obey |tau| <= mu lambda, with no slip
while |tau| < mu lambda and the slip against tau when it is at mu lambda.
Without it the fasteners carry everything that slides, which is the safe
side for them and leaves a part held only by what it rests on (a stand on
a desk, a clamp on a tube) free to slide off; with it, a stand pushed
sideways holds while its weight's friction does, as it does. The load is
taken as applied at once from rest (proportional loading), the usual
reading of a static friction problem; a path that reverses is not tracked.

Preload (supports[].preload, N): the seat is pressed on before the loads
come, by a screw's clamp or a fit, and keeps whatever of that the loads do
not take off: it is set into the part (an interference e, uniform over the
seat) so that, with no load, it pushes with the preload; the loads then
change its force as the part's stiffness has it (a rigid screw: a clamped
joint stays shut until the load takes the clamp off, and its friction
holds with the clamp's force). A screw's head seat (fasteners.py) takes its
screw's preload when the study's seats have friction.

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

__all__ = ["patches", "hold", "resolve", "solve", "solve_cases", "Seat", "SeatedSpread", "worst_stress", "Patches",
           "Rows", "bilateral", "linearized", "direct_many", "obstacles", "seated_solver"]

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


class Rows:
    """What each unit push (row) is: `seat` (its seat's index), `kind` (0
    the push along the seat's normal; 1 and 2 friction along two directions
    in the face), `group` (the row of its patch's normal push), and per
    seat its friction coefficient `mu` and `preload` (N)."""

    def __init__(self, seat, kind, group, mu, preload):
        self.seat = np.asarray(seat, int)
        self.kind = np.asarray(kind, int)
        self.group = np.asarray(group, int)
        self.mu = np.asarray(mu, float)
        self.preload = np.asarray(preload, float)

    @staticmethod
    def plain(n: int):
        return Rows(np.zeros(n), np.zeros(n), np.arange(n), [0.0], [0.0])

    def groups(self):
        """[[normal row, tangent rows...]] per patch, in row order."""
        out = {}
        for r in range(len(self.kind)):
            out.setdefault(int(self.group[r]), []).append(r)
        return [sorted(v, key=lambda r: self.kind[r]) for _, v in sorted(out.items())]

    def take(self, groups):
        """The rows of `groups` ([[old rows]]) renumbered in that order."""
        seat, kind, group = [], [], []
        for g in groups:
            head = len(kind)
            for r in g:
                seat.append(self.seat[r])
                kind.append(self.kind[r])
                group.append(head)
        return Rows(seat, kind, group, self.mu, self.preload)


class Patches(list):
    """patches' unit pushes, a list of force vectors, with `rows` (Rows)."""

    rows: Rows = None


def _across(n):
    n = np.asarray(n, float)
    a = np.cross(n, [1.0, 0.0, 0.0] if abs(n[0]) < 0.9 else [0.0, 1.0, 0.0])
    a /= np.linalg.norm(a)
    return a, np.cross(n, a)


def patches(seats, xyz, n_dof, most: int = MAX_PATCHES):
    """Unit pushes on the seats' patches: a list of force vectors (n_dof),
    each 1 N along its seat's normal spread over the patch's nodes by their
    share of the face; on a seat with friction each patch also has two
    unit forces across it (Patches.rows says which is which). `seats`:
    [(node indices, normal (3,), weights[, friction, preload])].

    A seat can push anywhere inside the outline of its patches' centres, and
    a part tips about that outline's edge. So the corners of the face's own
    outline are small patches of their own (a rectangle's four corners, a
    disc's rim in a dozen points), and squares fill the rest: the part tips
    where the face ends, not a patch's half width inside it."""
    out = Patches()
    seat_of, kind, group = [], [], []
    mus = [float(s[3] or 0.0) if len(s) > 3 else 0.0 for s in seats]
    pre = [float(s[4] or 0.0) if len(s) > 4 else 0.0 for s in seats]
    per = max(1, most // max(len(seats), 1))
    for si, s in enumerate(seats):
        idx, n, w = s[0], s[1], s[2]
        idx, w = np.asarray(idx), np.asarray(w, float)
        n = np.asarray(n, float)
        if not len(idx):
            continue
        dirs = [n] + (list(_across(n)) if mus[si] > 0 else [])
        keys = np.full(len(idx), -1)
        if len(idx) > per:
            # a corner is its node and the nodes beside it: one node alone can hang off a sliver of a cut cell
            from scipy.spatial import cKDTree

            tree = cKDTree(xyz[idx])
            near = 1.5 * float(np.median(tree.query(xyz[idx], k=2)[0][:, 1]))
            corners = _outline(xyz[idx], n, min(12, per // 3))
            for k, c in enumerate(corners):
                group_ = np.array(tree.query_ball_point(xyz[idx[c]], near))
                keys[group_[keys[group_] < 0]] = k
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
            share = w[m] / w[m].sum()
            head = len(out)
            for kd_, dv in enumerate(dirs):
                p = np.zeros(n_dof)
                for a in range(3):
                    p[3 * idx[m] + a] = share * dv[a]
                out.append(p)
                seat_of.append(si)
                kind.append(kd_)
                group.append(head)
    out.rows = Rows(seat_of, kind, group, mus or [0.0], pre or [0.0])
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


SLIDES = ("a load slides the part along its seat (supports[].push, friction) with nothing else to hold it: "
          "the friction the seat's pressure gives is less than the load across it. Hold the part where it is "
          "really held across the seat, or give the seat its preload (a clamp's screw)")


class Seat:
    """The seats' contact on one sprung stiffness, in the space of the
    patches' gaps: the unit pushes `ps`, their displacements `ws` on the
    sprung stiffness (hold), the springs `ks` (N/mm per sprung patch) and,
    when `ps` are pieces of the sprung patches (refine), each piece's share
    of each (pieces x patches). `rows` (Rows; `ps.rows` when it has them)
    says which rows are friction across a patch and what each seat's
    friction and preload are.

    Every question about the contact is a small one here: for the gaps g0
    a load opens on the springs alone (g0 = P u0), `push` gives the gaps,
    the pushes mu that make the answer (u = u0 + mu . w) and the seats'
    forces. The load's compliance with the seat pushing is f.u0 + mu.g0
    (f.w_j = p_j.u0: the sprung stiffness is symmetric), and its gradient in
    the load is 2u: what the directions need (SeatedSpread)."""

    def __init__(self, ps, ws, ks, share=None, rows=None):
        self.p = np.array(ps)
        self.w = np.array(ws)
        ks = np.asarray(ks, float)
        self.share = np.eye(len(ps)) if share is None else np.asarray(share, float)
        rows = rows if rows is not None else getattr(ps, "rows", None)
        self.rows = rows if rows is not None and len(rows.kind) == len(ps) else Rows.plain(len(ps))
        c = self.p @ self.w.T
        c = 0.5 * (c + c.T)
        self.a = np.linalg.pinv(c, hermitian=True)
        self.kd = (self.share * ks) @ self.share.T
        h = self.a - self.kd
        d, v = np.linalg.eigh(0.5 * (h + h.T))
        floor = FLOOR * float(ks.mean()) if ks.size and ks.mean() > 0 else 1e-12 * max(float(np.abs(d).max()), 1e-300)
        self.d, self.v, self.dc = d, v, np.maximum(d, floor)
        self.m = np.sqrt(self.dc)[:, None] * v.T          # m'm = the stiffness, floored
        r = self.rows
        self.normal = r.kind == 0
        # friction: per patch with a coefficient, its normal row, its two rows across, mu
        g = [x for x in r.groups() if len(x) == 3 and r.mu[r.seat[x[0]]] > 0]
        self.fi = np.array([x[0] for x in g], int)
        self.ft = np.array([x[1:] for x in g], int).reshape(-1, 2)
        self.fmu = np.array([r.mu[r.seat[x[0]]] for x in g], float)
        # how far below zero each normal row's gap may go: a clearance;
        # a preload's interference is a negative one (the seat set into the part)
        self.e = np.zeros(len(ps))
        self.slip0 = np.zeros(len(ps))                      # where slip counts from: the preloaded state
        self.unpressed = []
        self.g_pre = np.zeros(len(ps))                      # the gaps the preload leaves, with no load
        if len(self.fi) or (r.preload > 0).any():
            self.hc = (v * self.dc) @ v.T
            self.hc = 0.5 * (self.hc + self.hc.T)
            # the contact's own stiffness per row, the Alart-Curnier scale
            self.rho = np.maximum(np.diag(self.a), 1e-300)
        if (r.preload > 0).any():
            self._preload()

    def __len__(self):
        return len(self.p)

    def gaps(self, u):
        return self.p @ u

    # -- the contact problem in the gaps -------------------------------------------------
    #
    # With b = A g0 and Hc the floored stiffness seen from the seat, the gaps g
    # minimize 1/2 g'Hc g - b'g with g_n >= -e on the normal rows (e < 0: the
    # preload's interference) and, across a patch with friction, Coulomb's law on its
    # slip g_t - slip0 against its force r_t, r = Hc g - b: |r_t| <= mu r_n.

    def _normal_only(self, b, fixed=None):
        """Gaps for the normal rows alone (nnls), the rows across held at
        `fixed` (their values) or absent; in the shifted gaps g + e."""
        from scipy.optimize import nnls

        e = self.e
        if fixed is None and not len(self.fi):
            bb = b + self.hc @ e if e.any() else b
            g, _ = nnls(self.m, (self.v.T @ bb) / np.sqrt(self.dc), maxiter=50 * len(b))
            return g - e
        nr = np.nonzero(self.normal)[0]
        tr = np.nonzero(~self.normal)[0]
        hnn = self.hc[np.ix_(nr, nr)]
        rhs = b[nr] + hnn @ e[nr]
        if len(tr):
            rhs = rhs - self.hc[np.ix_(nr, tr)] @ fixed[tr]
        lo = np.linalg.cholesky(hnn + 1e-14 * np.trace(hnn) / len(nr) * np.eye(len(nr)))
        x, _ = nnls(lo.T, np.linalg.solve(lo, rhs), maxiter=50 * len(nr))
        g = fixed.copy() if fixed is not None else np.zeros(len(b))
        g[nr] = x - e[nr]
        return g

    def _coulomb(self, b, g, slip0, most: int = 200):
        """Gaps with Coulomb friction on the patches that have it, from the
        guess g: the Alart-Curnier equations by a semismooth Newton's method
        with a line search on their residual. Returns (gaps, converged)."""
        hc, rho, e = self.hc, self.rho, self.e
        nr = np.nonzero(self.normal)[0]
        fi, ft, fmu = self.fi, self.ft, self.fmu
        n = len(b)

        def state(x):
            r = hc @ x - b
            an = r[nr] - rho[nr] * (x[nr] + e[nr])
            phi = np.empty(n)
            phi[nr] = r[nr] - np.maximum(an, 0.0)
            jac = np.empty((n, n))
            jac[nr] = hc[nr]
            shut = an > 0
            jac[nr[shut]] = 0.0
            jac[nr[shut], nr[shut]] = rho[nr[shut]]
            if len(fi):
                ai = r[fi] - rho[fi] * (x[fi] + e[fi])
                big = fmu * np.maximum(ai, 0.0)
                rt = r[ft]                                             # (k, 2)
                z = rt - rho[ft] * (x[ft] - slip0[ft])
                nz = np.linalg.norm(z, axis=1)
                stick = (nz <= big) & (ai > 0)
                for k in range(len(fi)):
                    t = ft[k]
                    if stick[k]:
                        phi[t] = rt[k] - z[k]
                        jac[t] = 0.0
                        jac[t, t] = rho[t]
                        continue
                    if ai[k] <= 0 or nz[k] <= 0:
                        phi[t] = rt[k]
                        jac[t] = hc[t]
                        continue
                    nh = z[k] / nz[k]
                    phi[t] = rt[k] - big[k] * nh
                    dr = fmu[k] * hc[fi[k]].copy()
                    dr[fi[k]] -= fmu[k] * rho[fi[k]]
                    dz = hc[t].copy()
                    dz[0, t[0]] -= rho[t[0]]
                    dz[1, t[1]] -= rho[t[1]]
                    proj = (np.eye(2) - np.outer(nh, nh)) * (big[k] / nz[k])
                    jac[t] = hc[t] - np.outer(nh, dr) - proj @ dz
            return phi, jac

        scale = max(float(np.abs(b).max()), float(np.abs(hc @ e).max()) if e.any() else 0.0, 1e-300)
        tol = 1e-10 * scale
        x = g.copy()
        phi, jac = state(x)
        res = float(np.linalg.norm(phi))
        for _ in range(most):
            if res <= tol:
                return x, True
            try:
                dx = np.linalg.solve(jac, -phi)
            except np.linalg.LinAlgError:
                dx = np.linalg.lstsq(jac, -phi, rcond=None)[0]
            t = 1.0
            while t > 1e-6:
                xn = x + t * dx
                pn, jn = state(xn)
                rn = float(np.linalg.norm(pn))
                if rn < (1.0 - 1e-4 * t) * res:
                    break
                t *= 0.5
            else:
                return x, res <= 1e-6 * scale
            x, phi, jac, res = xn, pn, jn, rn
        return x, res <= 1e-6 * scale

    def _gaps(self, b, slip0=None):
        """The contact's gaps for b = A g0."""
        if not len(self.fi):
            return self._normal_only(b)
        slip0 = self.slip0 if slip0 is None else slip0
        g = self._normal_only(b, fixed=slip0)
        g, ok = self._coulomb(b, g, slip0, most=60)
        self.held = 1.0
        if not ok:
            # the load put on in steps from the clamped state (the path a
            # load that grows from nothing takes): a hard case is found step
            # by step, and one past what friction holds stops where it slides
            g, t, dt = self.g_pre.copy(), 0.0, 0.125
            while t < 1.0 and dt > 0.125 / 16:
                tn = min(1.0, t + dt)
                gn, okn = self._coulomb(tn * b, g, slip0, most=40)
                if okn:
                    g, t = gn, tn
                else:
                    dt *= 0.5
            ok = t >= 1.0
            self.held = t
        self.converged = ok
        return g

    def _preload(self):
        """The interference that presses each preloaded seat on with its
        preload when nothing else loads the part (the contact is positively
        homogeneous in it, so a unit one scaled, then corrected a few
        times: the seats that bear can change as it grows), and the state
        it leaves, where slip counts from."""
        r = self.rows
        on = [s for s in range(len(r.preload)) if r.preload[s] > 0]
        rows_of = [np.nonzero(self.normal & (r.seat == s))[0] for s in on]
        want = np.array([r.preload[s] for s in on])
        zero = np.zeros(len(self.p))

        def forces(ev):
            self.e = np.zeros(len(self.p))
            for k, rr in enumerate(rows_of):
                self.e[rr] = -ev[k]
            g = self._gaps(self.a @ zero, slip0=zero)
            lam = self.hc @ g
            return np.array([lam[rr].sum() for rr in rows_of]), g

        unit = 1e-3
        m = np.zeros((len(on), len(on)))
        for k in range(len(on)):
            ev = np.zeros(len(on))
            ev[k] = unit
            m[:, k] = forces(ev)[0] / unit
        # a preload with nothing to press against (the part only moves): none
        live = np.array([m[k, k] > 1e-6 * max(float(np.abs(np.diag(m)).max()), 1e-300) for k in range(len(on))])
        self.unpressed = [on[k] for k in range(len(on)) if not live[k]]
        ev = np.zeros(len(on))
        if live.any():
            ml = m[np.ix_(live, live)]
            ev[live] = np.linalg.lstsq(ml, want[live], rcond=None)[0]
            for _ in range(8):
                f, _ = forces(ev)
                miss = want[live] - f[live]
                if np.abs(miss).max() <= 1e-4 * want[live].max():
                    break
                ev[live] += np.linalg.lstsq(ml, miss, rcond=None)[0]
        _, g = forces(ev)
        self.slip0 = np.where(self.normal, 0.0, g)
        self.g_pre = g

    def push(self, g0, strict: bool = False):
        """(gaps, mu, lambda, lifts) for the gaps `g0` the load opens on the
        springs alone. lambda: the seats' forces per row (pushes on the
        normal rows, friction across). `lifts`: the load takes the part off
        its seat, or slides it along, with nothing else to hold it (raised
        with `strict`)."""
        g0 = np.asarray(g0, float)
        b = self.a @ g0
        self.converged = True
        if not len(self.fi) and not self.e.any():
            from scipy.optimize import nnls

            g, _ = nnls(self.m, (self.v.T @ b) / np.sqrt(self.dc), maxiter=50 * len(b))
        else:
            g = self._gaps(b)
        # what the floor carried: nothing, unless the part is going somewhere
        lost = self.v @ ((self.dc - self.d) * (self.v.T @ g))
        ref = max(np.linalg.norm(b), np.linalg.norm(self.hc @ self.e) if self.e.any() else 0.0, 1e-300)
        lifts = bool(np.linalg.norm(lost) > 1e-2 * ref)
        # Coulomb's law with no answer: the load across is more than the
        # friction holds and nothing else does (the floor alone would, far off)
        slides = not self.converged
        if strict and (lifts or slides):
            across = slides or np.abs(lost[~self.normal]).sum() > np.abs(lost[self.normal]).sum()
            if across and slides and getattr(self, "held", 1.0) > 0:
                raise ValueError(SLIDES + f" (it slides at about {self.held:.0%} of the load)")
            raise ValueError(SLIDES if across else LIFTS)
        lifts = lifts or slides
        mu = self.a @ (g - g0)
        lam = mu - self.kd @ g
        lam[self.normal] = np.maximum(lam[self.normal], 0.0)
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


def worst_stress(case: SeatedSpread, read, mask, scale=None, n: int = 300, strict: bool = False, offset=None,
                 judge=None):
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
    it raises ValueError (verify). `offset`: stress the constant part carries
    beyond its displacement's (a thermal block's: -C eps_th, read at zero
    displacement), counted with the constant part's coefficient.
    `judge(sig) -> peak`: the peak the part
    is read by, from the stress at every node, when it is not von Mises
    over `mask` (a pressure-modified stress for a polymer stronger in
    compression; a peak read between nodes on the counted region's edge):
    the directions von Mises ranks
    highest, and the nominal one, are read with it exactly and the worst
    by it is taken (a seated plate's tet check took von Mises' worst and
    read 4% under its own 'up' case, 2026-11-24)."""
    rows = case.samples(n)
    # the force as written and the six axis directions are always among the
    # candidates: a spread of n directions is ~200/sqrt(n) degrees apart, so
    # a peak near an axis read only off the spread can come out below the
    # same force held along that axis (CI f19510e: "any" read under "up").
    from .directions import _to_cap
    sp = case.spread
    axis = np.asarray(sp.nominal, float)
    for d in (axis, *np.eye(3), *-np.eye(3)):
        d = _to_cap(d / np.linalg.norm(d), axis, sp.cos_half)
        c, _, mu, lam, lifts = case.at(d)
        rows.insert(0, (d, c, mu, lam, lifts))
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
        if offset is not None:
            sk = sk + (bt[k][0] / norms[0]) * offset
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
    vall = np.zeros(len(coef))
    for k0 in range(0, len(coef), 32):
        sig = np.einsum("sk,mkj->smj", coef[k0:k0 + 32], sk)
        v = _vm2(sig).max(axis=1)
        vall[k0:k0 + len(v)] = v
        j = int(np.argmax(v))
        if v[j] > best:
            best, bs = float(v[j]), k0 + j
    if judge is not None:
        nominal = np.asarray(case.spread.nominal, float)
        nominal = nominal / max(np.linalg.norm(nominal), 1e-300)
        c0, _, mu0, lam0, lf0 = case.at(nominal)          # the nominal direction itself, exactly
        rows = list(rows) + [(nominal, c0, mu0, lam0, lf0)]
        cand = sorted(set(np.argsort(-vall)[:12].tolist() + [bs, len(rows) - 1]))
        qx = SeatedSpread.coefficients([rows[c] for c in cand])
        top = -np.inf
        for c, qc in zip(cand, qx):
            sg = read(case.combine(qc))
            if offset is not None:
                sg = sg + qc[0] * offset
            sg = sg if scale is None else sg * scale
            e = float(judge(sg))
            if e > top:
                top, bs = e, c
        d, mu = rows[bs][0], rows[bs][2]
        return np.asarray(d, float), mu, top
    d, mu = rows[bs][0], rows[bs][2]
    return np.asarray(d, float), mu, float(np.sqrt(max(best, 0.0)))


def _split(p, xyz, more=()):
    """A patch's unit push cut in up to four across its two widest
    directions: [(unit push of the piece, its share of the patch)]; with
    `more` (the patch's friction rows), [([pieces of p, then of each],
    share)], cut alike."""
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
        own = (3 * nodes[m][:, None] + np.arange(3)).ravel()
        frac = float(weight[m].sum() / weight.sum())
        qs = []
        for r in (p, *more):
            q = np.zeros_like(r)
            q[own] = r[own] / frac
            qs.append(q)
        out.append((qs if more else qs[0], frac))
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
    rows = getattr(ps, "rows", None)
    ps = list(ps)
    rows = rows if rows is not None and len(rows.kind) == len(ps) else Rows.plain(len(ps))
    if layout is None:
        layout = [(i,) for i in range(len(rhs))]
    spreads = spreads or [None] * len(layout)
    stiff, us = inner(rhs + ps)
    us = list(us)
    u0s, ws = us[:len(rhs)], us[len(rhs):]
    share = np.eye(len(ps))
    normal = rows.kind == 0

    def judge():
        seat = Seat(ps, ws, ks, share, rows)
        done, cases, bear = {}, {}, np.zeros(len(ps))

        def bears(lam):
            lam = np.where(normal, np.maximum(lam, 0.0), 0.0)
            return lam / max(lam.sum(), 1e-300)

        for ci, idx in enumerate(layout):
            if len(idx) == 1:
                i = idx[0]
                _, mu, lam, _ = seat.push(seat.gaps(u0s[i]), strict)
                done[i] = (seat.apply(u0s[i], mu), lam)
                bear = np.maximum(bear, bears(lam))
                continue
            i0 = idx[0]
            case = SeatedSpread(seat, rhs[i0] if i0 is not None else None, u0s[i0] if i0 is not None else None,
                                [rhs[i] for i in idx[1:]], [u0s[i] for i in idx[1:]], spreads[ci])
            rows_ = case.samples(60)
            if strict and any(r[4] for r in rows_):
                raise ValueError(LIFTS + " A force of unknown direction lifts it in some direction.")
            for r in rows_:
                bear = np.maximum(bear, bears(r[3]))
            cases[ci] = case
        return done, cases, bear

    done, cases, bear = judge()
    for _ in range(REFINE_ROUNDS if xyz is not None else 0):
        groups = rows.groups()
        head = {g[0]: g for g in groups}
        cut = {}
        for j in np.argsort(-bear):
            j = int(j)
            if bear[j] <= 1e-6 * max(bear.sum(), 1e-300) or 4 * (len(cut) + 1) > REFINE_MOST:
                break
            if j not in head:
                continue
            g = head[j]
            pieces = _split(ps[g[0]], xyz, [ps[r] for r in g[1:]])
            if pieces:
                cut[j] = pieces
        if not cut:
            break
        # the pieces of a patch with friction carry their own rows across, cut alike
        new, srows, new_groups = [], [], []
        for j, pieces in cut.items():
            g = head[j]
            for qs, frac in pieces:
                qs = qs if isinstance(qs, list) else [qs]
                ng = []
                for r, q in zip(g, qs):
                    ng.append(len(ps) + len(new))
                    new.append(q)
                    srows.append(frac * share[r])
                new_groups.append((g, ng))
        _, wn = inner(new)
        wn = list(wn)
        allp, allw, alls = ps + new, ws + wn, list(share) + srows
        # rows' identities for the pieces: each as the row it was cut from
        ext_seat = np.concatenate([rows.seat, [rows.seat[r] for g, ng in new_groups for r in g]])
        ext_kind = np.concatenate([rows.kind, [rows.kind[r] for g, ng in new_groups for r in g]])
        ext = Rows(ext_seat, ext_kind, np.arange(len(allp)), rows.mu, rows.preload)
        order = [g for g in groups if g[0] not in cut] + [ng for _, ng in new_groups]
        rows = ext.take(order)
        flat = [r for g in order for r in g]
        ps = [allp[r] for r in flat]
        ws = [allw[r] for r in flat]
        share = np.array([alls[r] for r in flat])
        normal = rows.kind == 0
        done, cases, bear = judge()
    # the forces on the patches given: each piece's go to the patch it was cut from
    back = (share > 0).astype(float)
    out_u = [done[i][0] if i in done else u0s[i] for i in range(len(rhs))]
    out_l = [done[i][1] @ back if i in done else None for i in range(len(rhs))]
    return stiff, out_u, out_l, cases


# a seat held for a small motion about a state: its patch's spring, over the
# modulus times its width (HOLD's ten thousand times: a rigid seat to 0.1%)
RIGID = 1000.0


def linearized(seats, mesh, nu: float, modulus, fixed, springs, forces, pitch: float):
    """The seats as springs for a small motion about the state each force
    leaves them in (modal: the rest state, the preload and the part's own
    weight; buckling: each load case's), on `mesh`'s dofs: per force a
    sparse matrix, or None where no seat bears. A patch that bears holds
    the part along its normal both ways (the motion is small beside what
    presses it: the contact stays shut); across it too where it has
    friction and sticks (|tau| below mu lambda). A patch that is open, or
    slides, holds nothing that way. `springs`: the other supports' springs
    (absolute, or None). Seats with nothing to hold (no force, no preload)
    leave the part as the other supports hold it."""
    import scipy.sparse as sp

    from .bucklingopt import _factor
    from .fem import Stiffness

    fixed = np.asarray(fixed, bool)
    mod = np.broadcast_to(np.asarray(modulus, float), (mesh.n_elem,)).copy()
    e_ref = float(np.mean(mod)) if mod.size else 1.0
    ps = patches(seats, mesh.node_xyz, mesh.n_dof)
    if not len(ps):
        return [None] * len(forces)
    ks, held = hold(ps, mesh.node_xyz, pitch)
    free = sp.diags((~fixed).astype(float))
    k = Stiffness(mesh, nu, fixed).assemble(mod) + free @ (e_ref * held) @ free
    if springs is not None:
        k = k + free @ springs @ free
    solve_k, release = _factor(k.tocsr(), mesh.node_ijk)
    try:
        def inner(rhs):
            out = []
            for r in rhs:
                r = np.asarray(r, float).copy()
                r[fixed] = 0.0
                out.append(np.asarray(solve_k(r)).ravel())
            return None, out

        rhs = [np.zeros(mesh.n_dof) if f is None else np.asarray(f, float) for f in forces]
        _, us = inner(rhs + list(ps))
    finally:
        if release is not None:
            release()
    seat = Seat(ps, us[len(rhs):], e_ref * ks)
    out = []
    for u0 in us[:len(rhs)]:
        _, _, lam, _ = seat.push(seat.gaps(u0))
        bear = seat.normal & (lam > 1e-9 * max(float(np.abs(lam).max()), 1e-300))
        on = bear.copy()
        for i, t, m in zip(seat.fi, seat.ft, seat.fmu):
            if bear[i] and np.linalg.norm(lam[t]) < m * lam[i] * (1.0 - 1e-6):
                on[t] = True                                     # stuck: held across too
        if not on.any():
            out.append(None)
            continue
        out.append(_rigid(ps, on, mesh.node_xyz, pitch, e_ref))
    return out


def _rigid(ps, on, xyz, pitch, modulus):
    """Rigid springs (N/mm) on the patch rows `on` of `ps`."""
    import scipy.sparse as sp

    idx = np.nonzero(on)[0]
    width = np.array([float(np.ptp(xyz[np.unique(np.nonzero(ps[i])[0] // 3)], axis=0).max())
                      if np.count_nonzero(ps[i]) > 3 else pitch for i in idx])
    q = sp.csr_matrix(np.array([ps[i] for i in idx]))
    return (q.T @ sp.diags(RIGID * modulus * np.maximum(width, pitch)) @ q).tocsr()


def bilateral(seats, xyz, pitch: float, modulus: float, fixed=None):
    """The seats held both ways as rigid springs (N/mm, on xyz's dofs): along
    each patch's normal, and across where it has friction. For a linear
    design that assumes its seats stay shut (a compliant mechanism's: the
    large-deflection check then reads them as contact, `obstacles`). None
    without seats."""
    if not seats:
        return None
    ps = patches(seats, xyz, 3 * len(xyz))
    if not len(ps):
        return None
    k = _rigid(ps, np.ones(len(ps), bool), xyz, pitch, modulus)      # tangent rows only where mu > 0
    if fixed is not None:
        import scipy.sparse as sp

        free = sp.diags((~np.asarray(fixed, bool)).astype(float))
        k = (free @ k @ free).tocsr()
    return k


def obstacles(seats, xyz):
    """The seats as rigid planes the part may not enter
    (contact.RigidObstacle, for the large-deflection solve), each meeting
    only its own face's nodes, with the seat's friction: a seat the part
    lifts off holds nothing, one it slides on holds mu times its push.
    Seats with a preload are left out (the caller holds them as
    `bilateral`: the clamp keeps the joint shut)."""
    from .contact import RigidObstacle

    out = []
    for s_ in seats or ():
        if len(s_) > 4 and s_[4]:
            continue
        idx = np.asarray(s_[0], int)
        n = np.asarray(s_[1], float)
        n = n / np.linalg.norm(n)
        h = xyz[idx] @ n
        out.append(RigidObstacle("plane", point=xyz[idx[np.argmin(h)]], normal=n, nodes=idx,
                                 friction=float(s_[3] or 0.0) if len(s_) > 3 else 0.0))
    return out


def seated_solver(bc, xyz, pitch: float, modulus: float):
    """For a check that solves the study's cases on its own grid (the
    lattice sizing, the field graph's homogenized check): (bc, solve) where
    bc carries the seats' holding springs beside its own (relative to the
    modulus, as bc.springs) and solve(inner, rhs) -> displacements with the
    seats pushing, `inner(rhs) -> (stiffness or None, displacements)` on
    that bc. (bc, None) without seats."""
    if not getattr(bc, "seats", None):
        return bc, None
    from dataclasses import replace

    ps = patches(bc.seats, xyz, 3 * len(xyz))
    ks, held = hold(ps, xyz, pitch)
    bc_s = replace(bc, springs=held if bc.springs is None else (bc.springs + held).tocsr())

    def run(inner, rhs):
        return solve(inner, rhs, ps, modulus * ks, True, xyz)[1]

    return bc_s, run


def direct_many(k, node_ijk, fixed, rhs):
    """K u = f for a block of right-hand sides (a seated solve's cases and
    unit pushes: a hundred and more) by one factorization; `fixed` dofs
    held at 0. An iterative solve pays for each again."""
    from .bucklingopt import _factor

    fixed = np.asarray(fixed, bool)
    solve, release = _factor(k.tocsr(), node_ijk)
    try:
        b = np.stack([np.where(fixed, 0.0, np.asarray(r, float)) for r in rhs], axis=1)
        x = np.asarray(solve(b)).reshape(b.shape)
    finally:
        if release is not None:
            release()
    return [x[:, j] for j in range(x.shape[1])]
