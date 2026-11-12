"""
The pour, as the optimizer can see it.

casting.py pours a finished part: it marches the heat equation through
freezing and follows the liquid as it drains. That is the check. It is
not something a density optimizer can differentiate through three hundred
times. This module is the same physics reduced to what decides soundness
and is smooth in the density field:

**Which metal freezes when.** While an alloy freezes it sits inside its
freezing range, so the only temperature difference that can move heat
from one section of the casting to another is that range, a few tens of
degrees, against the several hundred between the metal and its mold. Each
section therefore freezes nearly on its own, in the order of how much
metal it holds per unit of mold surface (its modulus; Chvorinov). The
steady balance of that is

    -div(k_f rho grad m) + h [mold faces] m = rho

on the grid: every cell of metal is a unit source, every face between
metal and mold a loss, and the metal conducts between its cells with its
conductivity scaled by (freezing range) / (metal less mold temperature).
With the loss normalised to one per unit of face, m is a length: for a
lump it is exactly volume / surface, the modulus, and between sections it
blends over the distance the freezing range can carry heat. The freezing
order is the order of m. `order_check` compares it with the full solve.

**Whether each region can be fed.** Metal at x freezes at its m(x) and
needs liquid until then. Liquid comes from the feeder through metal that
is still open, i.e. through cells with m above m(x). So x is fed if some
path from the feeder to x stays above m(x); the best path is the one
whose lowest cell is highest (the widest path), F(x), and the shortfall
D(x) = m(x) - F(x) is zero for fed metal and positive in every pool that
a thinner neck cuts off: a hot spot, and where the shrinkage cavity will
be. This is the rule casting.py's feeding applies step by step, read off
the order field in one pass.

The measure is the share of the metal in such pools, and its gradient
says the two things a foundry does: thicken the neck (raise m at the
bottleneck) or take metal out of the pool (lower m there). The widest
path picks one bottleneck cell per pool cell, so the gradient is exact
almost everywhere: an adjoint solve of the same matrix carries it back
to the densities.

Gravity is not in the measure (a riser feeds downhill; the full pour
checks it), nor is the pressure drop along a long thin path (the Niyama
porosity in the report).

**What it gets wrong, and who decides.** The steady balance gives every
cell the same heat to lose. A real lump is closer to a reservoir: while it
has latent heat left it sits at its freezing temperature and keeps a short
thin section beside it open, as a fin on it, far longer than that
section's own modulus says. Against the full pour the measure agrees on a
long thin neck (two lumps 50 mm apart on a plate: both call it sound from
about 10 mm of plate) and over-predicts where the thin stretch is short
(a lean truss between the same lumps: 0.44% cavity predicted, 0.01%
poured). So in the optimizer the full pour is the judge: `FeedTerm.pour`
runs it on the current design every few iterations, the weight on this
measure climbs only while the pour leaves a cavity, and the measure's
gradient is used for what it is good at, the direction: which neck to
thicken, which pool to lighten.
"""
from __future__ import annotations

import heapq
import math

import numpy as np
import scipy.sparse as sp

__all__ = ["FeedTerm", "order_check", "widest_path"]

try:                                                    # pragma: no cover - exercised where numba is
    import numba

    _njit = numba.njit(cache=True)
except Exception:                                       # noqa: BLE001
    numba = None

    def _njit(f):
        return f


@_njit
def _widest(theta, nx, ny, nz, feed):
    """Widest path from the feeder cells on a grid: for every cell the
    highest level F it can be reached at (the best path's lowest theta,
    itself included) and the cell `b` that sets it (-1: reached from the
    feeder with nothing lower than itself on the way)."""
    n = nx * ny * nz
    F = np.full(n, -1.0)
    b = np.full(n, -1, np.int64)
    done = np.zeros(n, np.bool_)
    heap = [(-1e300, np.int64(-1))]
    heap.pop()
    for f in feed:
        F[f] = theta[f]
        b[f] = -1
        heapq.heappush(heap, (-1e300, np.int64(f)))
    syz = ny * nz
    while len(heap) > 0:
        negf, u = heapq.heappop(heap)
        if done[u]:
            continue
        done[u] = True
        fu = -negf                                      # the level u passes on (the feeder: unbounded)
        i = u // syz
        j = (u // nz) % ny
        k = u % nz
        for d in range(6):
            if d == 0:
                if i == 0:
                    continue
                v = u - syz
            elif d == 1:
                if i == nx - 1:
                    continue
                v = u + syz
            elif d == 2:
                if j == 0:
                    continue
                v = u - nz
            elif d == 3:
                if j == ny - 1:
                    continue
                v = u + nz
            elif d == 4:
                if k == 0:
                    continue
                v = u - 1
            else:
                if k == nz - 1:
                    continue
                v = u + 1
            if done[v]:
                continue
            tv = theta[v]
            if tv <= fu:                                # v itself is the lowest so far: it is fed
                cand = tv
                bv = np.int64(-1)
            else:                                       # something on the way froze first
                cand = fu
                bv = b[u] if b[u] >= 0 else u
                if fu >= 1e299:
                    bv = np.int64(-1)
                    cand = tv
            if cand > F[v]:
                F[v] = cand
                b[v] = bv
                heapq.heappush(heap, (-cand, np.int64(v)))
    return F, b


def widest_path(theta: np.ndarray, feed: np.ndarray):
    """(F, b) grids for an order field and a bool grid of feeder cells:
    F the level each cell can be fed at (<= theta), b the flat index of
    the bottleneck cell that limits it (-1 where it is fed)."""
    nx, ny, nz = theta.shape
    idx = np.flatnonzero(feed.ravel()).astype(np.int64)
    if not len(idx):
        raise ValueError("no feeder cells")
    F, b = _widest(np.ascontiguousarray(theta, float).ravel(), nx, ny, nz, idx)
    return F.reshape(theta.shape), b.reshape(theta.shape)


def conduction_ratio(s) -> float:
    """k_f / (h L) per metre of L: the metal's conductivity scaled by the
    share of the metal-to-mold temperature difference that the freezing
    range is, over the mold's conductance. Returns (k_f, h(M)) helpers as
    a function of the section modulus M (m): the mold's conductance is
    what removes a section's heat in its own freezing time."""
    from .casting import _release, plate_freeze_time

    a = s.alloy
    q, t_mean = _release(a, s.pour_temperature, 1.0)
    dt_mold = max(t_mean - s.mold_temperature, 1.0)
    k_f = a.conductivity * (a.liquidus - a.solidus) / dt_mold

    def h(modulus_m: float) -> float:
        t = plate_freeze_time(2e3 * modulus_m, s)       # a plate of that modulus
        return q * modulus_m / (max(t, 1e-9) * dt_mold)

    return k_f, h


class FeedTerm:
    """The feeding measure of a density grid and its gradient.

    dims: the grid; pitch mm; `setup` a casting.CastSetup (the alloy and
    mold set how far sections blend); `feed`: bool grid of the cells the
    feeder (riser or gate) joins, or None: the feeder goes where the metal
    freezes last, which is where a riser would be stood."""

    def __init__(self, dims, pitch: float, setup, feed: np.ndarray | None = None, outside: float = 0.0):
        self.dims = tuple(int(d) for d in dims)
        self.pitch = float(pitch)
        self.setup = setup
        self.feed = None if feed is None else np.asarray(feed, bool)
        if self.feed is not None and not self.feed.any():
            raise ValueError("casting.feed covers no cell of the design space")
        self.k_f, self._h = conduction_ratio(setup)
        # the cavity a sound casting may hold: casting.design_risers' tolerance, 0.1% of the part
        self.limit = 0.001
        n = int(np.prod(self.dims))
        idx = np.arange(n).reshape(self.dims)
        self._pairs = []                                # (i, j) flat indices of each interior face, per axis
        for ax in range(3):
            lo = [slice(None)] * 3
            hi = [slice(None)] * 3
            lo[ax], hi[ax] = slice(0, -1), slice(1, None)
            self._pairs.append((idx[tuple(lo)].ravel(), idx[tuple(hi)].ravel()))
        # faces on the grid's edge: the mold is beyond them
        edge = np.zeros(self.dims)
        for ax in range(3):
            sl = [slice(None)] * 3
            for end in (0, -1):
                sl[ax] = end
                edge[tuple(sl)] += 1.0
        self._edge = edge.ravel() * (1.0 - outside)
        self._ml = None
        self._top = None
        self.last = None

    # -- the order field ------------------------------------------------------

    def _biot(self, rho):
        """k_f / (h dx) at the design's own modulus."""
        vol = float(rho.sum())
        area = float(self._loss(rho).sum())
        modulus = max(vol / max(area, 1e-9), 0.5) * self.pitch * 1e-3
        return self.k_f / (self._h(modulus) * self.pitch * 1e-3), modulus

    def _loss(self, rho):
        """Mold faces per cell (in faces): rho_i (1 - rho_j) over neighbours."""
        out = rho * self._edge
        for i, j in self._pairs:
            out += np.bincount(i, rho[i] * (1.0 - rho[j]), len(rho))
            out += np.bincount(j, rho[j] * (1.0 - rho[i]), len(rho))
        return out

    def _matrix(self, rho, bi):
        n = len(rho)
        diag = self._loss(rho) + 1e-6
        rows, cols, vals = [], [], []
        for i, j in self._pairs:
            g = bi * rho[i] * rho[j]
            diag += np.bincount(i, g, n) + np.bincount(j, g, n)
            rows += [i, j]
            cols += [j, i]
            vals += [-g, -g]
        rows.append(np.arange(n))
        cols.append(np.arange(n))
        vals.append(diag)
        return sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n, n))

    def _solve(self, K, rhs, x0=None, fresh=True):
        import pyamg
        from scipy.sparse.linalg import cg

        if fresh or self._ml is None:
            self._ml = pyamg.smoothed_aggregation_solver(K, max_coarse=400)
        M = self._ml.aspreconditioner()
        x, info = cg(K, rhs, x0=x0, M=M, rtol=1e-8, maxiter=400)
        return x

    def pour(self, rho_grid, order=None) -> dict:
        """The full pour of this design (cells over half full are metal)
        with a riser where the feeder is: casting.solidify, the liquid
        followed as it drains. Returns {"cavityShare": the shrinkage cavity
        left in the part over its volume, "riser": ...}. The riser stands on
        the part over the feeder and is large enough to freeze last (twice
        the casting's highest modulus), so what is read is the part's own
        feeding, not the riser's size."""
        from .casting import _stamp, solidify

        part = np.asarray(rho_grid, float) > 0.5
        if order is None:
            order = self.order(part.astype(float))
        if self.feed is None:
            at = np.unravel_index(int(np.argmax(np.where(part, order, -1.0))), self.dims)
            cells = np.array([at])
        else:
            cells = np.argwhere(self.feed & part)
            if not len(cells):
                cells = np.argwhere(self.feed)
        ax, sg = self.setup.up
        lat = [d for d in range(3) if d != ax]
        base = np.round(cells.mean(0)).astype(int)
        col = [slice(None)] * 3
        for d in lat:
            col[d] = int(base[d])
        filled = np.flatnonzero(part[tuple(col)])
        if not len(filled):                              # the feeder's column holds no metal yet
            return {"cavityShare": 0.0, "riser": None, "empty": True}
        base[ax] = int(filled.max() if sg > 0 else filled.min())
        # an open cylinder as tall as it is wide has a modulus of a fifth of its diameter
        d_r = 10.0 * float(order[part].max())
        rs = [{"base": [int(v) for v in base], "diameter": d_r, "height": d_r}]
        padded = np.pad(part, 2)
        rs[0]["base"] = [int(v) + 2 for v in base]
        metal, feeders, opn, off = _stamp(padded, rs, self.pitch, self.setup.up)
        r = solidify(metal, self.pitch, self.setup, feeders=feeders, open_top=opn)
        core = tuple(slice(o, o + k) for o, k in zip(off, padded.shape))
        pv = np.nan_to_num(r.void[core]) * padded
        return {"cavityShare": float(pv.sum() / max(padded.sum(), 1)), "riser": rs[0],
                "void": pv[2:-2, 2:-2, 2:-2], "seconds": r.seconds}

    def _through(self, lam, theta, rho, bi):
        """lam . (d rhs / d rho_e - (d K / d rho_e) theta) per cell (the
        blend length is held: it follows the design's overall modulus)."""
        n = len(rho)
        out = lam - self._edge * lam * theta
        for i, j in self._pairs:
            cond = bi * (lam[i] - lam[j]) * (theta[i] - theta[j])
            li, lj = lam[i] * theta[i], lam[j] * theta[j]
            out -= np.bincount(i, cond * rho[j] + (1.0 - rho[j]) * li - rho[j] * lj, n)
            out -= np.bincount(j, cond * rho[i] + (1.0 - rho[i]) * lj - rho[i] * li, n)
        return out

    def order(self, rho_grid):
        """m (mm): the local modulus the freezing order follows."""
        rho = np.clip(np.asarray(rho_grid, float).ravel(), 0.0, 1.0)
        bi, _ = self._biot(rho)
        K = self._matrix(rho, bi)
        return (self._solve(K, rho) * self.pitch).reshape(self.dims)

    # -- the measure ------------------------------------------------------------

    def measure(self, rho_grid, grad: bool = True, hold: bool = False):
        """(value, d value / d rho grid, record). value: the shrinkage
        cavity the part would hold as a share of its volume: 0 when every
        cell freezes toward the feeder."""
        rho = np.clip(np.asarray(rho_grid, float).ravel(), 0.0, 1.0)
        n = len(rho)
        bi, modulus = self._biot(rho)
        K = self._matrix(rho, bi)
        x0 = None if self.last is None else self.last["theta"]
        rhs = rho
        if self.feed is None:
            theta = self._solve(K, rho, x0)
            feed = np.zeros(self.dims, bool)
            feed.ravel()[int(np.argmax(theta * (rho > 0.5)))] = True
        else:
            # a riser is the last metal to freeze: its cells sit at the casting's highest modulus
            # (as last solved), and the metal near them stays open with it (the riser's zone)
            feed = self.feed
            top = self._top if self._top is not None else float(self._solve(K, rho, x0).max())
            pin = 1e3 * feed.ravel()
            K = K + sp.diags(pin)
            rhs = rho + pin * top
            theta = self._solve(K, rhs, x0)
            if not hold:
                self._top = float(theta[~feed.ravel()].max())
        tg = theta.reshape(self.dims)
        F, b = widest_path(tg, feed)
        F, b = F.ravel(), b.ravel()
        # the share of each cell's shrinkage that nothing feeds. Against a mold that takes heat as
        # sqrt(time), a section of modulus m is solid to the fraction sqrt(t / t_f) with t_f ~ m^2;
        # the neck stops passing liquid at the alloy's feeding limit f, at t = f^2 t_f(neck), when
        # the cell behind it is solid to f m_neck / m. From there to f it drains with no supply:
        # f (1 - m_neck / m) of its shrinkage is left as cavity.
        a = self.setup.alloy
        pool = (b >= 0) & (theta > F) & (theta > 1e-9)
        u = np.where(pool, 1.0 - F / np.maximum(theta, 1e-9), 0.0)
        scale = a.beta_prime() * a.feed_limit
        vol = float(rho.sum()) + 1e-30
        value = scale * float(rho @ u) / vol
        rec = {"cavityShare": value, "unfedShare": float((rho * (u > 0.02)).sum() / vol),
               "modulus": modulus * 1e3, "maxModulus": float(theta.max() * self.pitch),
               "feeder": [int(v) for v in np.argwhere(feed)[0]]}
        self.last = {"theta": theta, "unfed": u.reshape(self.dims), "F": F, "b": b, "feed": feed,
                     "rho": rho.reshape(self.dims), "value": value}
        if not grad:
            return value, None, rec
        # d value / d m: at each pool cell + rho F / m^2, at its bottleneck - rho / m
        sel = np.flatnonzero(pool)
        g = np.zeros(n)
        g[sel] += rho[sel] * F[sel] / theta[sel] ** 2
        g -= np.bincount(b[sel], rho[sel] / theta[sel], n)
        g *= scale / vol
        lam = self._solve(K, g, fresh=False)
        # the value moves with rho directly and through m (the blend length and the feeder's
        # level follow the design as last solved; `hold` keeps the level as it is)
        out = scale * u / vol - value / vol + self._through(lam, theta, rho, bi)
        return value, out.reshape(self.dims), rec


def order_check(solid: np.ndarray, pitch: float, setup, result=None) -> dict:
    """The order field against the full solidification solve on a solid
    grid: rank correlation of m with the freezing time over the metal, and
    whether both put the last metal in the same place."""
    from scipy import stats

    from .casting import solidify

    r = result if result is not None else solidify(solid, pitch, setup)
    m = FeedTerm(solid.shape, pitch, setup).order(solid.astype(float))
    t = r.time[solid]
    mm = m[solid]
    rho = float(stats.spearmanr(mm, t).statistic)
    a, b = np.argwhere(solid)[np.argmax(mm)], np.argwhere(solid)[np.argmax(t)]
    return {"rank": rho, "lastByOrder": a.tolist(), "lastBySolve": b.tolist(),
            "apart": float(np.linalg.norm(a - b) * pitch), "result": r, "order": m}


def study_feed(study, mesh, dims, pitch):
    """The FeedTerm of a study whose casting names a feed, or None."""
    if getattr(study, "cast_feed", None) is None or getattr(study, "foundry", None) is None:
        return None
    from .casting import setup
    from .domain import _fraction

    fd = study.foundry
    s = setup(fd["process"], fd["alloy_name"], **{k: fd[k] for k in ("pour_temperature", "mold_temperature",
                                                                      "head", "up") if k in fd})
    feed = None
    if study.cast_feed != "auto":
        centers = mesh.origin + (mesh.elem_ijk + 0.5) * mesh.pitch
        hit = _fraction(list(study.cast_feed), centers, mesh.pitch) > 0.0
        if not hit.any():
            raise ValueError("casting.feed touches no cell of the part: put it where the riser or gate joins")
        feed = np.zeros(tuple(int(d) for d in dims), bool)
        ijk = mesh.elem_ijk[hit]
        feed[ijk[:, 0], ijk[:, 1], ijk[:, 2]] = True
    return FeedTerm(dims, pitch, s, feed)
