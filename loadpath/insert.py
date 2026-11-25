"""
A bearing, shaft or pin in a bore: the load reaches the part through a
stiff body that only pushes (loads[].distribution = "bearing").

The load a seated bearing puts on its housing is not a pressure laid on
the bore as drawn: it is whatever pressure keeps the bore in contact with
the outer ring, and the ring is steel. A 608's outer ring is ~1.5 mm of
steel, about 17 times as stiff in ring bending, per mm of width, as a
3 mm wall of PETG over it, so the bore cannot go oval where the ring
touches it: the wall there moves with the ring, as a rigid body. Laid on
as a fixed cosine, the pressure presses as hard on a thin wall over the
bore as on the thick pillars beside it, the thin wall bends as if nothing
were inside it, and a stiff plate beside the bore (a shoulder) is pulled
oval with it. On the 608 pillow block (P1 tets, the same mesh) the
fixed cosine read the shoulder at the shaft's hole 1.3-1.5 and the wall
over the bore 1.7-2.5; a rigid ring that only pushes, 1.6-2.3 and
3.6-4.5: the stiff paths take the load, as they do in service.

The contact is solved exactly on the linear model, on the stiffness of
the solve that asks (the voxel check, the tetrahedral check, the block):
the bore's wall is cut into patches (SECTORS round the axis by BANDS
along it), each a unit pressure square to the wall (p_j); the patches'
displacements w_j = K^-1 p_j give their compliance C_ij = p_i . w_j; the
ring moves as a rigid body q (across the axis, and tilting about the two
axes across it: a ring cannot carry a moment it is not given), and the
patch forces lam >= 0 are those that minimize the complementary energy
1/2 lam' C lam with the ring in equilibrium, N' lam = W (N_j the wrench
patch j puts on the ring, W the load's). Its conditions are the contact's:
no patch pulls, a patch that pushes touches (C lam - N q = 0 there), and
none goes into the ring (>= 0 elsewhere). The pressures replace the fixed
cosine in that case's right-hand side; everything downstream (seats,
directions, stress) is unchanged.

The ring is taken as rigid against the part: right while the part is a
tenth as stiff as steel or less (every polymer and composite); a printed
metal housing is as stiff as the ring, and keeps the fixed pressure
(domain.bearing_forces), which is what a ring as thin as nothing gives.
The axial part of the load bears on the faces that face it, fixed
(domain.bearing_forces). The rest of the case's loads move the bore too,
and the ring pushes against that as well (the cross term g = P' u_rest of
the complementary energy): every ring of a case is solved together, each
in its own equilibrium (Rings). The optimizer designs against the same
contact (simp._Analysis._solve_ringed, the patches' compliance read every
few solves), and by the envelope theorem its compliance sensitivity is
the ordinary one with the ring's pressures in.

A bearing force of unknown direction (direction "any" or a cone) is not
the sum of three axes' answers through a ring that only pushes: per
direction the contact is solved again on the patches' compliance, the
thrust taken on the faces facing it either way (RingedSpread), and the
checks' worst direction found as for a seated case (pushseat.worst_stress).
On a part also held by seats (supports[].push) the seats and the rings
are one contact, solved together for a force as written and per direction
for one of unknown direction (SeatedRings, SeatedRingSpread, below): the
ring's pressures load the seats, and where a seat lets go changes how the
bore deforms under the ring.

Units: mm, N, MPa.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["Insert", "Rings", "RingedSpread", "SeatedRings", "SeatedRingSpread", "plan", "plan_spread", "applies", "redistribute", "ringed_spreads", "SECTORS",
           "BANDS", "RIGID_BELOW"]

# patches round the axis and along it (a band per end: the ring tilts, so
# where along the bore it bears is part of the answer)
SECTORS = 12
BANDS = 2
# the part counts as soft against a steel ring (200 GPa) below this modulus, MPa
RIGID_BELOW = 20e3
# patch pushes solved at once
_BATCH = 8


@dataclass
class Insert:
    """One bearing load's ring on one mesh: in case `case`, the wall nodes
    `nodes` (their unit normals out of the bore into the wall `normals`,
    their shares of the wall `shares`), the ring's centre and axis, the
    load across the axis `force` (N) and the nodal forces (len(at), 3)
    that the fixed pressure put on the nodes `at` for it (`fixed`, replaced)."""
    case: int
    nodes: np.ndarray
    normals: np.ndarray
    shares: np.ndarray
    center: np.ndarray
    axis: np.ndarray
    force: np.ndarray
    fixed: np.ndarray
    at: np.ndarray
    # a force of unknown direction (its size in `force`'s length): the
    # thrust's pressures on the faces that face +axis and -axis, for the
    # whole force along each (len(at), 3); None for a force as written
    thrust: tuple | None = None


def applies(material) -> bool:
    """Whether a steel ring is rigid against `material` (RIGID_BELOW).
    LOADPATH_RING=0 turns the ring off (the fixed pressure everywhere),
    for comparing the two."""
    import os

    if os.environ.get("LOADPATH_RING") == "0":
        return False
    e = getattr(material, "E", None)
    return e is not None and float(e) < RIGID_BELOW


def plan(case: int, shape, pts, nodes, shares, force, fixed) -> Insert | None:
    """The ring of a bearing load on the bore `shape` (a cylinder), on the
    nodes `nodes` lining it at `pts` with `shares` of its wall; None when
    the load has no part across the axis or no wall faces it. Only the
    bore's side wall touches a ring: nodes whose wall faces along the axis
    (a shoulder, a floor) are left out."""
    from .domain import _wall_normals

    ax = getattr(shape, "axis", None)
    if ax is None:
        return None
    force = np.asarray(force, float)
    # (with no part across the axis the ring is still there: it holds the
    # bore round against what the thrust and the rest of the case do to it)
    radial = force - (force @ ax) * ax
    pts = np.asarray(pts, float)
    m = _wall_normals(shape, pts)
    side = np.abs(m @ ax) < 0.5
    if side.sum() < SECTORS:
        return None
    # the ring is round: it pushes square to its own surface, across the axis
    m = m - (m @ ax)[:, None] * ax
    m /= np.maximum(np.linalg.norm(m, axis=1, keepdims=True), 1e-12)
    p0 = np.asarray(shape.p0, float)
    t = (pts[side] - p0) @ ax
    center = p0 + 0.5 * (t.min() + t.max()) * ax
    return Insert(case, np.asarray(nodes)[side], m[side], np.asarray(shares, float)[side],
                  center, np.asarray(ax, float), radial, np.asarray(fixed, float), np.asarray(nodes))


def plan_spread(case: int, shape, pts, nodes, shares, size: float, nominal) -> Insert | None:
    """The ring of a bearing force of unknown direction (`size` N, written
    along `nominal`): its patches as plan's, and the thrust's pressures on
    the faces facing each way along the axis (domain.bearing_forces)."""
    from .domain import bearing_forces

    ax = getattr(shape, "axis", None)
    if ax is None:
        return None
    ax = np.asarray(ax, float)
    probe = np.cross(ax, [1.0, 0.0, 0.0] if abs(ax[0]) < 0.9 else [0.0, 1.0, 0.0])
    ins = plan(case, shape, pts, nodes, shares, probe, np.zeros((len(pts), 3)))
    if ins is None:
        return None
    thrust = []
    for sgn in (1.0, -1.0):
        t = bearing_forces(shape, np.asarray(pts, float), np.asarray(shares, float), sgn * size * ax)
        thrust.append(np.zeros((len(pts), 3)) if t is None else t)
    ins.force = size * np.asarray(nominal, float) / max(np.linalg.norm(nominal), 1e-300)
    ins.thrust = tuple(thrust)
    return ins


def _patches(ins: Insert, xyz):
    """Per patch: (rows into ins.nodes, its unit pressure as nodal forces
    (k, 3)) and the wrench (P, 6) each puts on the ring's centre (force on
    the part; the ring takes the opposite)."""
    ax = ins.axis
    a = np.cross(ax, [1.0, 0.0, 0.0] if abs(ax[0]) < 0.9 else [0.0, 1.0, 0.0])
    a /= np.linalg.norm(a)
    b = np.cross(ax, a)
    r = xyz - ins.center
    ang = np.arctan2(r @ b, r @ a)
    sector = np.floor((ang + np.pi) / (2 * np.pi) * SECTORS).astype(int) % SECTORS
    t = r @ ax
    span = float(t.max() - t.min())
    band = np.zeros(len(t), int) if span <= 0 else np.minimum((BANDS * (t - t.min()) / span).astype(int), BANDS - 1)
    out, wrench = [], []
    for k in np.unique(sector * BANDS + band):
        rows = np.nonzero(sector * BANDS + band == k)[0]
        s = ins.shares[rows]
        if s.sum() <= 0:
            continue
        f = (s / s.sum())[:, None] * ins.normals[rows]
        out.append((rows, f))
        wrench.append(np.concatenate([f.sum(axis=0), np.cross(r[rows], f).sum(axis=0)]))
    return out, np.array(wrench)


def _rigid_basis(ax):
    """The ring's motions that its contact can resist: across the axis,
    and tilting about the two axes across it (6 x 4)."""
    a = np.cross(ax, [1.0, 0.0, 0.0] if abs(ax[0]) < 0.9 else [0.0, 1.0, 0.0])
    a /= np.linalg.norm(a)
    b = np.cross(ax, a)
    z = np.zeros(3)
    return np.stack([np.concatenate([a, z]), np.concatenate([b, z]),
                     np.concatenate([z, a]), np.concatenate([z, b])], axis=1)


def _equilibrium(wrench, load, basis):
    """The ring's equilibrium rows a lam = b (the motions it resists)."""
    a = basis.T @ wrench.T              # (4, P)
    b = basis.T @ load
    keep = np.linalg.norm(a, axis=1) > 1e-12 * max(float(np.abs(a).max()), 1e-300)
    return a[keep], b[keep]


def contact(c, wrench, load, basis, g=None):
    """The patch forces lam >= 0 minimizing 1/2 lam' C lam + g' lam with
    basis' (wrench' lam) = basis' load: the rigid ring's frictionless
    contact on the compliance C (P x P). `g` = P' u_rest, the patches'
    pushes against what the rest of the case's loads do to the bore (the
    cross term of the complementary energy of the whole load)."""
    a, b = _equilibrium(wrench, load, basis)
    return _qp(c, a, b, g)


def _qp(c, a, b, g=None):
    """min 1/2 lam' C lam + g' lam, lam >= 0, a lam = b. nnls on the
    energy's Cholesky factor (1/2 |L' lam + L^-1 g|^2 is the energy less a
    constant) with the equilibrium as stiff rows, then the rows' residue
    fed back (augmented Lagrangian) until it is under 1e-9 of the load."""
    from scipy.linalg import solve_triangular

    c = 0.5 * (c + c.T)
    n = len(c)
    if a.shape[0] == 0 or not np.any(b):
        if g is None or not np.any(g < 0):
            return np.zeros(n)
    sc = float(np.trace(c)) / max(n, 1)
    lo = np.linalg.cholesky(c + 1e-10 * max(sc, 1e-300) * np.eye(n))
    top = np.zeros(n) if g is None else -solve_triangular(lo, np.asarray(g, float), lower=True)
    return _nnls_eq(lo.T, top, a, b, sc)


def _nnls_eq(mt, top, a, b, sc):
    """min |mt lam - top|^2, lam >= 0, a lam = b: the equilibrium rows
    weighted far above the energy's (a force of 1 N off weighs as an energy
    1e6 x a patch's own compliance `sc`), their residue fed back."""
    from scipy.optimize import nnls

    n = mt.shape[1]
    beta = np.sqrt(1e6 * max(sc, 1e-300))
    target = np.asarray(b, float).copy()
    size = max(float(np.linalg.norm(b)), 1e-300)
    lam = np.zeros(n)
    for _ in range(20):
        m = np.vstack([mt, beta * a])
        rhs = np.concatenate([top, beta * target])
        lam, _ = nnls(m, rhs, maxiter=50 * n)
        if a.shape[0] == 0:
            break
        res = b - a @ lam
        if np.linalg.norm(res) <= 1e-9 * size:
            break
        target = target + res
    return lam


class Rings:
    """Every bore's patches on one mesh, as one set: a case with two
    bearings (or a bearing and other loads) solves its rings together,
    each in its own equilibrium, on the cross-compliance of all their
    patches. Pushes are kept sparse (dof, value)."""

    def __init__(self, inserts, xyz, n_dof):
        self.n_dof = n_dof
        self.ps, self.bore, wrench, self.axes = [], [], [], []
        keys, self.cases, self.fixed, self.spread = {}, {}, {}, {}
        for ins in inserts:
            key = (ins.nodes.tobytes(), ins.center.tobytes(), ins.axis.tobytes())
            if key not in keys:
                pats, wr = _patches(ins, xyz[ins.nodes])
                if len(pats) < 3:
                    keys[key] = None
                else:
                    keys[key] = len(self.axes)
                    self.axes.append(np.asarray(ins.axis, float))
                    for (rows, f), w in zip(pats, wr):
                        self.ps.append(((3 * ins.nodes[rows][:, None] + np.arange(3)).ravel(), f.ravel()))
                        self.bore.append(keys[key])
                        wrench.append(w)
            b = keys[key]
            if b is None:
                continue
            if ins.thrust is not None:
                # a force of unknown direction (RingedSpread): one per case
                self.spread[ins.case] = (b, ins)
                continue
            load = np.concatenate([ins.force, np.zeros(3)])
            on = self.cases.setdefault(ins.case, {})
            on[b] = on.get(b, 0.0) + load
            self.fixed.setdefault(ins.case, []).append((ins.at, ins.fixed))
        self.bore = np.array(self.bore, int)
        self.wrench = np.array(wrench).reshape(-1, 6)

    def __len__(self):
        return len(self.ps)

    def dense(self, idx):
        out = []
        for j in idx:
            dof, val = self.ps[j]
            v = np.zeros(self.n_dof)
            v[dof] = val
            out.append(v)
        return out

    def patches_of(self, bores):
        return np.flatnonzero(np.isin(self.bore, list(bores)))

    def gaps(self, u, idx=None):
        """P' u over the patches `idx` (all)."""
        u = np.asarray(u).ravel()
        idx = range(len(self.ps)) if idx is None else idx
        return np.array([float(val @ u[dof]) for dof, val in (self.ps[j] for j in idx)])

    def compliance(self, inner, idx=None, keep=False):
        """C over the patches `idx` (all), solved a few at a time with
        `inner` (list of vectors -> displacements); with `keep`, also the
        patches' displacements W (len(idx) x n_dof)."""
        idx = list(range(len(self.ps)) if idx is None else idx)
        c = np.zeros((len(idx), len(idx)))
        w = np.zeros((len(idx), self.n_dof)) if keep else None
        for b0 in range(0, len(idx), _BATCH):
            for j, u in enumerate(inner(self.dense(idx[b0:b0 + _BATCH]))):
                u = np.asarray(u).ravel()
                c[:, b0 + j] = self.gaps(u, idx)
                if keep:
                    w[b0 + j] = u
        return 0.5 * (c + c.T), w

    def strip(self, f, case):
        """The case's loads without its rings' fixed pressure."""
        out = np.array(f, float, copy=True)
        for at, fx in self.fixed.get(case, ()):
            for k in range(3):
                out[3 * at + k] -= fx[:, k]
        return out

    def solve_case(self, case, c, idx, g=None):
        """The case's patch forces (over `idx`, the patches of its bores)
        on their compliance c, with g = P' u of the rest of its loads."""
        rows, rhs = [], []
        loads = self.cases[case]
        bores = self.bore[idx]
        for b, load in loads.items():
            on = bores == b
            wr = np.zeros((len(idx), 6))
            wr[on] = self.wrench[idx][on]
            a, bb = _equilibrium(wr, load, _rigid_basis(self.axes[b]))
            rows.append(a)
            rhs.append(bb)
        return _qp(c, np.vstack(rows), np.concatenate(rhs), g)

    def thrusts(self, case):
        """A force of unknown direction's thrust along +axis and -axis, for
        a force of 1 N (dense)."""
        _, ins = self.spread[case]
        out = []
        for tk in ins.thrust:
            v = np.zeros(self.n_dof)
            for k in range(3):
                np.add.at(v, 3 * ins.at + k, tk[:, k] / max(np.linalg.norm(ins.force), 1e-300))
            out.append(v)
        return out

    def add(self, f, lam, idx):
        out = np.array(f, float, copy=True)
        for l_, j in zip(lam, idx):
            if l_ != 0.0:
                dof, val = self.ps[j]
                out[dof] += l_ * val
        return out


def redistribute(inner, rhs: list, inserts: list, case_rhs, n_dof: int, xyz) -> list:
    """`rhs` with each insert's fixed pressure replaced by the rigid ring's
    (contact) on the stiffness `inner` solves (`inner(list of vectors)`
    -> displacements, flat). `case_rhs(case)` is the index in `rhs` of that
    case's one right-hand side, or None (a force of unknown direction:
    RingedSpread). The patches are solved once for every case; each case's
    loads without the ring once more, for the ring's pushes against them."""
    rhs = list(rhs)
    rings = Rings(inserts, xyz, n_dof)
    todo = [(ci, case_rhs(ci)) for ci in sorted(rings.cases)]
    todo = [(ci, i) for ci, i in todo if i is not None]
    if not todo:
        return rhs
    bores = set()
    for ci, _ in todo:
        bores |= set(rings.cases[ci])
    idx = rings.patches_of(bores)
    c, _ = rings.compliance(inner, idx)
    pos = {j: k for k, j in enumerate(idx)}
    bases = [rings.strip(rhs[i], ci) for ci, i in todo]
    for b0 in range(0, len(todo), _BATCH):
        part = [b for b in bases[b0:b0 + _BATCH]]
        live = [k for k, b in enumerate(part) if np.any(b)]
        us = dict(zip(live, inner([part[k] for k in live]))) if live else {}
        for k, (ci, i) in enumerate(todo[b0:b0 + _BATCH]):
            own = rings.patches_of(set(rings.cases[ci]))
            sub = np.array([pos[j] for j in own])
            g = rings.gaps(us[k], own) if k in us else None
            lam = rings.solve_case(ci, c[np.ix_(sub, sub)], own, g)
            rhs[i] = rings.add(part[k], lam, own)
    return rhs


def _ascend(at, sp, n: int, start=None):
    """[(d, c, mu, lam, lifts)] of `at(d)` (-> (c, _, mu, lam, lifts)) over a
    spread of `n` directions of the cap `sp` and the axes, then the end of a
    pattern ascent from the best few (and from `start`) last: the worst. (A
    contact's compliance has a gradient only where the patches that bear
    stay the same; the pattern needs none.)"""
    from .directions import _cap_samples, _to_cap

    axis = np.asarray(sp.nominal, float)
    dirs = [_to_cap(d / np.linalg.norm(d), axis, sp.cos_half)
            for d in [*_cap_samples(sp, n), axis, *np.eye(3), *-np.eye(3)]]
    if start is not None:
        dirs.append(_to_cap(np.asarray(start, float), axis, sp.cos_half))
    rows = [(d, *at(d)) for d in dirs]
    order = list(np.argsort([-r[1] for r in rows])[:3]) + ([len(rows) - 1] if start is not None else [])
    best = None
    for k in order:
        d, cv = rows[k][0], rows[k][1]
        step = 0.1
        while step > 1e-4:
            e1 = np.cross(d, [1.0, 0, 0] if abs(d[0]) < 0.9 else [0, 1.0, 0])
            e1 /= np.linalg.norm(e1)
            e2 = np.cross(d, e1)
            moved = False
            for e in (e1, -e1, e2, -e2):
                dn = d + step * e
                dn = _to_cap(dn / np.linalg.norm(dn), axis, sp.cos_half)
                cn = at(dn)[0]
                if cn > cv * (1.0 + 1e-12):
                    d, cv, moved = dn, cn, True
                    break
            if not moved:
                step *= 0.5
        if best is None or cv > best[1]:
            best = (d, cv)
    r = at(best[0])
    out = [(row[0], row[1], row[3], row[4], row[5]) for row in rows]
    out.append((best[0], r[0], r[2], r[3], r[4]))
    return out


class RingedSpread:
    """A load case whose bearing force has no known direction
    (directions.Spread on a ring): for each direction d the ring's contact
    is solved again, so the answer is not the sum of three axes'. With the
    force F = |F| d, its part along the axis bears on the faces that face
    it (t+ or t-, fixed: a shoulder), its part across the axis through the
    ring (lam(d), contact), and the rest of the case is fixed (f0):

        u(d) = u0 + a+ u+ + a- u- + W lam(d),  a+- = |F| max(+-d.axis, 0)

    The compliance is the complementary energy at the contact's minimum,
    the most of a family of functions concave in (d, lam) along lam and
    convex jointly, so it is convex in d: its maximum over the sphere (or
    a cap round the force as written) is found from a spread of directions
    and an ascent from the best few. The interface is pushseat.SeatedSpread's
    (samples, at, coefficients, combine, norms, worst, f, u), so the checks'
    pushseat.worst_stress and the optimizer's directions.collapse take it."""

    def __init__(self, rings: Rings, case, spread, f0, u0, t, ut, c, w, idx, ref=None):
        b, ins = rings.spread[case]
        # (q_ref, u_ref): a push already solved on the current stiffness
        # (u_ref = K^-1 (a+ t+ + a- t- + P lam), q_ref = (a+, a-, lam))
        # while ut and w are older (the optimizer reads the ring's
        # compliance every few solves): exact at the reference
        self.ref = ref
        self.rings, self.case, self.spread = rings, case, spread
        self.axis = rings.axes[b]
        self.size = float(np.linalg.norm(ins.force))
        self.f0, self.u0 = f0, u0
        self.t, self.ut = list(t), list(ut)           # unit-size thrusts (+, -) and their answers
        self.c, self.w, self.idx = c, w, idx
        self.load_basis = _rigid_basis(self.axis)
        self.wrench = rings.wrench[idx]
        z = np.zeros(rings.n_dof)
        fs = [f0 if f0 is not None else z] + self.t
        us = [u0 if u0 is not None else z] + self.ut
        self.gram = np.array([[float(fi @ uj) for uj in us] for fi in fs])
        self.gram = 0.5 * (self.gram + self.gram.T)
        self.G = np.array([rings.gaps(u, idx) for u in us])     # 3 x patches

    def _a(self, d):
        s = float(np.asarray(d, float) @ self.axis)
        return np.array([1.0, self.size * max(s, 0.0), self.size * max(-s, 0.0)])

    def lam(self, d):
        d = np.asarray(d, float)
        radial = self.size * (d - (d @ self.axis) * self.axis)
        a = self._a(d)
        return contact(self.c, self.wrench, np.concatenate([radial, np.zeros(3)]), self.load_basis, a @ self.G)

    def at(self, d, strict: bool = False):
        """(compliance, None, lam, lam, False) for direction d."""
        a = self._a(d)
        lam = self.lam(d)
        g = a @ self.G
        cval = float(a @ self.gram @ a + 2.0 * lam @ g + lam @ self.c @ lam)
        return cval, None, lam, lam, False

    def samples(self, n: int = 200, start=None):
        """[(d, c, lam, lam, False)] over a spread of `n` directions (the
        cap's) and the axes; the ascent's end from the best few (and from
        `start`) last: the worst."""
        return _ascend(self.at, self.spread, n, start)

    def coefficients(self, rows):
        return np.array([np.concatenate([self._a(r[0]), r[2]]) for r in rows])

    def basis(self):
        out = self.u0 if self.u0 is not None else np.zeros(self.rings.n_dof)
        if self.ref is not None:
            q, ur = self.ref
            out = out + ur - (q[0] * self.ut[0] + q[1] * self.ut[1] + q[2:] @ self.w)
        return out

    def norms(self):
        return np.array([np.linalg.norm(self.basis()), *(np.linalg.norm(u) for u in self.ut),
                         *np.linalg.norm(self.w, axis=1)], float)

    def combine(self, q):
        q = np.asarray(q, float)
        return q[0] * self.basis() + q[1] * self.ut[0] + q[2] * self.ut[1] + q[3:] @ self.w

    def u(self, d, mu=None):
        lam = self.lam(d) if mu is None else mu
        return self.combine(np.concatenate([self._a(d), lam]))

    def f(self, d):
        a = self._a(d)
        f = a[1] * self.t[0] + a[2] * self.t[1]
        if self.f0 is not None:
            f = f + self.f0
        return self.rings.add(f, self.lam(d), self.idx)

    def worst(self, smooth: float, start=None, n: int = 120):
        """directions.Worst for the optimizer (pushseat.SeatedSpread.worst)."""
        from .directions import Worst

        rows = self.samples(n, start)
        d, cv, lam = rows[-1][0], rows[-1][1], rows[-1][2]
        cs = np.array([r[1] for r in rows])
        wt = np.exp(smooth * (cs - cv) / max(abs(cv), 1e-300))
        wt /= wt.sum()
        q = self.coefficients(rows)
        gm = (q * wt[:, None]).T @ q
        m, vec = np.linalg.eigh(gm)
        keep = [j for j in np.argsort(-m)[:8] if m[j] > 1e-6 * m.max()]
        energy = [(float(m[j]), self.combine(vec[:, j])) for j in keep]
        u_mean = self.combine(wt @ q)
        self.last = (d, lam)
        u, f = self.u(d, lam), self.f(d)
        return Worst(u, float(f @ u), d, f, energy, u_mean)

    def push(self, d, lam):
        """The ring's part of the load at direction d with patch forces
        lam (the thrust and the pressures), and its q = (a+, a-, lam)."""
        a = self._a(d)
        f = self.rings.add(a[1] * self.t[0] + a[2] * self.t[1], lam, self.idx)
        return f, np.concatenate([a[1:], lam])


def ringed_spreads(inner, rhs, layout, spreads, rings: Rings, c_all=None, w_all=None):
    """{case: RingedSpread} for the cases whose bearing force has no known
    direction, on the stiffness `inner` solves: the patches (with their
    answers W) and the two thrusts are solved here, the rest of the case
    (layout's fixed part) is taken from `rhs` solved by the caller as
    `inner` too: given as (rhs, us)."""
    rhs_f, rhs_u = rhs
    cases = [ci for ci in sorted(rings.spread) if ci < len(layout) and len(layout[ci]) > 1
             and spreads[ci] is not None]
    if not cases:
        return {}
    out = {}
    for ci in cases:
        b, ins = rings.spread[ci]
        idx = rings.patches_of({b})
        c, w = rings.compliance(inner, idx, keep=True)
        t = rings.thrusts(ci)
        ut = [np.asarray(u).ravel() for u in inner(t)]
        i0 = layout[ci][0]
        f0 = rhs_f[i0] if i0 is not None else None
        u0 = np.asarray(rhs_u[i0]).ravel() if i0 is not None else None
        out[ci] = RingedSpread(rings, ci, spreads[ci], f0, u0, t, ut, c, w, idx)
    return out


# -- seats and rings together ---------------------------------------------------------
#
# A part held by seats (pushseat.py) with a bearing in a bore: the ring pushes
# on the bore, the seats on the part's faces, and each lets go where its
# load would pull. Neither answer stands alone: the ring's pressures load
# the seats, and where the seats let go changes how the bore deforms under
# the ring. Both are contact on one linear model, so they are solved as one.
#
# On the sprung stiffness K' (the seats' patch springs, pushseat.hold) every
# patch, seat's and ring's, has its unit push p_j and answer w_j = K'^-1 p_j,
# and their compliance C_ij = p_i . w_j. With G the patches' gaps (P u) and
# G0 those of the case's other loads alone (P u_f), the part's energy in the
# gaps is 1/2 (G - G0)' A (G - G0) - 1/2 g_s' k g_s (A = C^-1; the seats'
# springs taken out exactly, as pushseat.Seat does), and the ring is a rigid
# body q pushed by its load W: minimize that less W'q over g_s >= 0 (a seat
# does not go into its wall) and g_r - N q >= 0 (the bore does not go into
# the ring). Its dual is the complementary energy of the contact forces:
#
#     min 1/2 (b + F)' H^-1 (b + F),  F >= 0,  N_b' F_b = W_b per ring,
#
# with H = A - diag(k, 0) the part's own stiffness seen from its patches and
# b = A G0: F holds every patch's force (a seat's push, a ring's pressure),
# and the gaps are G = H^-1 (b + F). H is singular along a motion only the
# seats stop (the part rests on them); its floor (pushseat.FLOOR) weighs the
# forces along such a motion far above the rest, so they must balance the
# load, and what the floor still carries is a part that lifts off. With no
# seat this is contact()'s problem; with no ring, Seat.push's.
#
# Friction across a seat or a preload is not a minimum of one energy: there
# the seats (Seat.push, with all of its friction) and the rings (contact)
# are solved in turn, each on the other's pushes, until neither moves.

class SeatedRings:
    """One case's seats (pushseat.Seat on the sprung stiffness) and rings
    (the patches `idx` of `rings`, their answers `wr` (len(idx), n_dof) on
    the same stiffness) as one contact."""

    def __init__(self, seat, rings: Rings, idx, wr):
        from .pushseat import _gram

        self.seat, self.rings = seat, rings
        self.idx = np.asarray(idx, int)
        self.wr = wr
        ns, nr = len(seat), len(self.idx)
        self.ns, self.nr = ns, nr
        css = _gram(seat.p, seat.w)
        csr = _gram(seat.p, wr) if ns else np.zeros((0, nr))
        crr = np.array([rings.gaps(u, self.idx) for u in wr]).T
        c = np.zeros((ns + nr, ns + nr))
        c[:ns, :ns], c[:ns, ns:], c[ns:, :ns], c[ns:, ns:] = css, csr, csr.T, crr
        self.c = 0.5 * (c + c.T)
        self.csr, self.crr = self.c[:ns, ns:], self.c[ns:, ns:]
        self.a = np.linalg.pinv(self.c, hermitian=True)
        kd = np.zeros_like(self.c)
        kd[:ns, :ns] = seat.kd
        h = self.a - kd
        d, v = np.linalg.eigh(0.5 * (h + h.T))
        floor = getattr(seat, "floor", None)
        if floor is None:
            floor = 1e-12 * max(float(np.abs(d).max()), 1e-300)
        self.d, self.v, self.dc = d, v, np.maximum(d, floor)
        # the energy's factor: |mt (b + F)|^2 = (b + F)' Hc^-1 (b + F)
        self.mt = (v / np.sqrt(self.dc)).T
        live = d > floor
        self.sc = float(np.median(1.0 / self.dc[live])) if live.any() else float(np.mean(1.0 / self.dc))
        # each ring's equilibrium rows, over all the patches
        self.eq = []
        bores = rings.bore[self.idx]
        for b in np.unique(bores):
            wr6 = np.zeros((ns + nr, 6))
            on = np.flatnonzero(bores == b)
            wr6[ns + on] = rings.wrench[self.idx[on]]
            basis = _rigid_basis(rings.axes[b])
            rows = basis.T @ wr6.T
            keep = np.linalg.norm(rows, axis=1) > 1e-12 * max(float(np.abs(rows).max()), 1e-300)
            self.eq.append((int(b), rows[keep], basis[:, keep]))
        # friction across a seat, or a preload: the two solved in turn
        self.joint = bool(seat.normal.all() and not len(seat.fi) and not seat.e.any())

    def gaps(self, u):
        """G = (P_s u, P_r u)."""
        return np.concatenate([self.seat.gaps(u), self.rings.gaps(u, self.idx)])

    def _rhs(self, loads):
        a = np.vstack([r for _, r, _ in self.eq]) if self.eq else np.zeros((0, self.ns + self.nr))
        b = [bs.T @ np.asarray(loads.get(bo, np.zeros(6)), float) for bo, _, bs in self.eq]
        return a, (np.concatenate(b) if b else np.zeros(0))

    def solve(self, g0, loads, strict: bool = False):
        """(G, mu, F, lifts) for the gaps `g0` the case's other loads open
        on the sprung stiffness and each ring's load {bore: wrench (6,)}:
        the gaps, the pushes mu on the sprung stiffness that give the answer
        (u = u_f + mu_s . w_s + mu_r . w_r), every patch's force F (seats'
        then rings'; a ring's is its mu) and whether the part lifts off its
        seats with nothing else to hold it (raised with `strict`)."""
        from .pushseat import LIFTS

        g0 = np.asarray(g0, float)
        a_eq, b_eq = self._rhs(loads)
        if self.joint:
            b = self.a @ g0
            f = _nnls_eq(self.mt, -self.mt @ b, a_eq, b_eq, self.sc)
            g = self.v @ ((self.v.T @ (b + f)) / self.dc)
            lost = self.v @ ((self.dc - self.d) * (self.v.T @ g))
            ref = max(float(np.linalg.norm(b)), float(np.linalg.norm(f)), 1e-300)
            lifts = bool(np.linalg.norm(lost) > 1e-2 * ref)
            mu = self.a @ (g - g0)
            mu[self.ns:] = f[self.ns:]
        else:
            g, mu, f, lifts = self._in_turn(g0, a_eq[:, self.ns:], b_eq, strict)
        if strict and lifts:
            raise ValueError(LIFTS)
        return g, mu, f, lifts

    def _in_turn(self, g0, a_eq, b_eq, strict: bool = False, most: int = 200):
        """Seats (Seat.push, friction and preload with it) and rings
        (contact) solved in turn on each other's pushes until they agree."""
        ns = self.ns
        fr = np.zeros(self.nr)
        g_s = mu_s = lam_s = None
        lifts = False
        for _ in range(most):
            g_s, mu_s, lam_s, lifts = self.seat.push(g0[:ns] + self.csr @ fr)
            gr = g0[ns:] + self.csr.T @ mu_s
            fn = _qp(self.crr, a_eq, b_eq, gr)
            moved = float(np.abs(fn - fr).sum())
            fr = fn
            if moved <= 1e-10 * max(float(np.abs(fn).sum()), 1e-300):
                break
        g_s, mu_s, lam_s, lifts = self.seat.push(g0[:ns] + self.csr @ fr, strict)
        g = np.concatenate([g_s, g0[ns:] + self.csr.T @ mu_s + self.crr @ fr])
        return g, np.concatenate([mu_s, fr]), np.concatenate([lam_s, fr]), lifts

    def apply(self, u_f, mu):
        """The answer u_f + mu . (w_s, w_r)."""
        u = np.asarray(u_f, float) + mu[:self.ns] @ self.seat.w if self.ns else np.asarray(u_f, float)
        return u + mu[self.ns:] @ self.wr

    def pressures(self, f, mu):
        """The loads `f` with the rings' pressures in."""
        return self.rings.add(f, mu[self.ns:], self.idx)


class SeatedRingSpread:
    """A bearing force of unknown direction on a part that is also held by
    seats: per direction d the seats' and the ring's contact solved together
    (SeatedRings). As RingedSpread, the force's part along the axis bears on
    the faces that face it (t+ or t-, fixed), the part across it through the
    ring, and the rest of the case (f0, with any other ring's load W_b) is
    fixed:

        u(d) = u0 + a+ u+ + a- u- + mu_s(d) . w_s + mu_r(d) . w_r

    with mu the patches' pushes on the sprung stiffness. The compliance is
    the work of the loads, f(d) . u(d) with the ring's pressures in (a seat
    does no work where it bears: its gap is zero there). The interface is
    pushseat.SeatedSpread's, so pushseat.worst_stress, directions.collapse
    and thermal.sensitivities take it.

    `ref`: (q_ref, u_ref) with q_ref = (a+, a-, mu) a push already solved on
    the current stiffness while the patches' answers are older (the
    optimizer reads them every few solves): exact at the reference."""

    def __init__(self, joint: SeatedRings, case, spread, f0, u0, t, ut, ref=None):
        rings = joint.rings
        b, ins = rings.spread[case]
        self.joint, self.rings, self.case, self.spread = joint, rings, case, spread
        self.bore = b
        self.axis = rings.axes[b]
        self.size = float(np.linalg.norm(ins.force))
        self.fixed = {bo: ld for bo, ld in rings.cases.get(case, {}).items()}
        self.f0, self.u0 = f0, u0
        self.t, self.ut = list(t), list(ut)
        self.ref = ref
        z = np.zeros(rings.n_dof)
        fs = [f0 if f0 is not None else z] + self.t
        us = [u0 if u0 is not None else z] + self.ut
        self.gram = np.array([[float(fi @ uj) for uj in us] for fi in fs])
        self.gram = 0.5 * (self.gram + self.gram.T)
        self.G = np.array([joint.gaps(u) for u in us])          # 3 x patches

    def _a(self, d):
        s = float(np.asarray(d, float) @ self.axis)
        return np.array([1.0, self.size * max(s, 0.0), self.size * max(-s, 0.0)])

    def _loads(self, d):
        d = np.asarray(d, float)
        out = dict(self.fixed)
        radial = self.size * (d - (d @ self.axis) * self.axis)
        out[self.bore] = out.get(self.bore, np.zeros(6)) + np.concatenate([radial, np.zeros(3)])
        return out

    def at(self, d, strict: bool = False):
        """(compliance, None, mu, F, lifts) for direction d."""
        a = self._a(d)
        g0 = a @ self.G
        g, mu, f, lifts = self.joint.solve(g0, self._loads(d), strict)
        ns = self.joint.ns
        cval = float(a @ self.gram @ a + mu @ g0 + f[ns:] @ g[ns:])
        return cval, None, mu, f, lifts

    def samples(self, n: int = 200, start=None):
        return _ascend(self.at, self.spread, n, start)

    def coefficients(self, rows):
        return np.array([np.concatenate([self._a(r[0]), r[2]]) for r in rows])

    def _pushes(self):
        return self.joint.seat.w if self.joint.ns else None

    def basis(self):
        out = self.u0 if self.u0 is not None else np.zeros(self.rings.n_dof)
        if self.ref is not None:
            q, ur = self.ref
            out = out + ur - (q[0] * self.ut[0] + q[1] * self.ut[1]) - self.joint.apply(np.zeros_like(ur), q[2:])
        return out

    def norms(self):
        ws = self._pushes()
        return np.array([np.linalg.norm(self.basis()), *(np.linalg.norm(u) for u in self.ut),
                         *(np.linalg.norm(ws, axis=1) if ws is not None else []),
                         *np.linalg.norm(self.joint.wr, axis=1)], float)

    def combine(self, q):
        q = np.asarray(q, float)
        u = q[0] * self.basis() + q[1] * self.ut[0] + q[2] * self.ut[1]
        return self.joint.apply(u, q[3:])

    def u(self, d, mu=None):
        mu = self.at(d)[2] if mu is None else mu
        return self.combine(np.concatenate([self._a(d), mu]))

    def f(self, d):
        a = self._a(d)
        f = a[1] * self.t[0] + a[2] * self.t[1]
        if self.f0 is not None:
            f = f + self.f0
        return self.joint.pressures(f, self.at(d)[2])

    def worst(self, smooth: float, start=None, n: int = 120):
        """directions.Worst for the optimizer (pushseat.SeatedSpread.worst);
        the worst direction and its pushes kept (`self.last`)."""
        from .directions import Worst

        rows = self.samples(n, start)
        d, cv, mu = rows[-1][0], rows[-1][1], rows[-1][2]
        cs = np.array([r[1] for r in rows])
        wt = np.exp(smooth * (cs - cv) / max(abs(cv), 1e-300))
        wt /= wt.sum()
        q = self.coefficients(rows)
        gm = (q * wt[:, None]).T @ q
        m, vec = np.linalg.eigh(gm)
        keep = [j for j in np.argsort(-m)[:8] if m[j] > 1e-6 * m.max()]
        energy = [(float(m[j]), self.combine(vec[:, j])) for j in keep]
        u_mean = self.combine(wt @ q)
        self.last = (d, mu)
        u = self.combine(np.concatenate([self._a(d), mu]))
        a = self._a(d)
        f = self.joint.pressures(a[1] * self.t[0] + a[2] * self.t[1] + (self.f0 if self.f0 is not None else 0.0), mu)
        return Worst(u, cv, d, f, energy, u_mean)

    def push(self, d, mu):
        """The thrust and every patch's push at direction d (to solve as one
        more load: the next reference), and its q = (a+, a-, mu)."""
        a = self._a(d)
        f = self.joint.pressures(a[1] * self.t[0] + a[2] * self.t[1], mu)
        ns = self.joint.ns
        if ns:
            f = f + np.asarray(self.joint.seat.p.T @ mu[:ns], float).ravel()
        return f, np.concatenate([a[1:], mu])
