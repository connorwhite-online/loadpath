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
(domain.bearing_forces). The optimizer designs against the fixed
pressure; the checks judge the part with the ring in it.

Units: mm, N, MPa.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["Insert", "plan", "applies", "redistribute", "SECTORS", "BANDS", "RIGID_BELOW"]

# patches round the axis and along it (a band per end: the ring tilts, so
# where along the bore it bears is part of the answer)
SECTORS = 12
BANDS = 2
# the part counts as soft against a steel ring (200 GPa) below this modulus, MPa
RIGID_BELOW = 20e3


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


def applies(material) -> bool:
    """Whether a steel ring is rigid against `material` (RIGID_BELOW)."""
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
    radial = force - (force @ ax) * ax
    if not np.any(radial):
        return None
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


def contact(c, wrench, load, basis):
    """The patch forces lam >= 0 minimizing 1/2 lam' C lam with
    basis' (wrench' lam) = basis' load: the rigid ring's frictionless
    contact on the compliance C (P x P). nnls on the energy's Cholesky
    factor with the equilibrium as stiff rows, then the rows' residue
    fed back (augmented Lagrangian) until it is under 1e-9 of the load."""
    from scipy.optimize import nnls

    c = 0.5 * (c + c.T)
    n = len(c)
    sc = float(np.trace(c)) / max(n, 1)
    lo = np.linalg.cholesky(c + 1e-10 * max(sc, 1e-300) * np.eye(n))
    a = basis.T @ wrench.T              # (4, P)
    b = basis.T @ load
    keep = np.linalg.norm(a, axis=1) > 1e-12 * max(float(np.abs(a).max()), 1e-300)
    a, b = a[keep], b[keep]
    # the equilibrium rows weighted far above the energy's: a force of 1 N
    # off weighs as an energy 1e6 x a patch's own compliance
    beta = np.sqrt(1e6 * max(sc, 1e-300))
    target = b.copy()
    size = max(float(np.linalg.norm(b)), 1e-300)
    lam = np.zeros(n)
    for _ in range(20):
        m = np.vstack([lo.T, beta * a])
        rhs = np.concatenate([np.zeros(n), beta * target])
        lam, _ = nnls(m, rhs, maxiter=50 * n)
        res = b - a @ lam
        if np.linalg.norm(res) <= 1e-9 * size:
            break
        target = target + res
    return lam


def redistribute(inner, rhs: list, inserts: list, case_rhs, n_dof: int, xyz) -> list:
    """`rhs` with each insert's fixed pressure replaced by the rigid ring's
    (contact) on the stiffness `inner` solves (`inner(list of vectors)`
    -> displacements, flat). `case_rhs(case)` is the index in `rhs` of that
    case's one right-hand side, or None (a force of unknown direction: it
    keeps the fixed pressure). One solve per patch, on the same matrix."""
    rhs = list(rhs)
    for ins in inserts:
        i = case_rhs(ins.case)
        if i is None:
            continue
        pats, wrench = _patches(ins, xyz[ins.nodes])
        if len(pats) < 3:
            continue
        ps = []
        for rows, f in pats:
            p = np.zeros(n_dof)
            for k in range(3):
                p[3 * ins.nodes[rows] + k] = f[:, k]
            ps.append(p)
        ws = inner(ps)
        c = np.array([[float(pi @ wj) for wj in ws] for pi in ps])
        load = np.concatenate([ins.force, np.zeros(3)])
        lam = contact(c, wrench, load, _rigid_basis(ins.axis))
        out = np.array(rhs[i], float, copy=True)
        for k in range(3):
            out[3 * ins.at + k] -= ins.fixed[:, k]
        out += sum(l_ * p for l_, p in zip(lam, ps))
        rhs[i] = out
    return rhs
