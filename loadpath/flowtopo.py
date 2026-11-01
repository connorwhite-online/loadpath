"""
Flow-aware topology optimization of heat sinks: the design decides where the
air goes, and the optimizer is charged for it.

thermal.py optimizes a sink with a fixed convection coefficient on its
surface. That rewards surface wherever it is, so on an air-cooled sink in a
duct it grows a dense pin field that chokes its own flow: rated with real
flow at a fixed fan pressure, it came out 25% worse than a uniform gyroid
and 20% worse than plain plate fins (docs/BENCHMARKS.md). Here the model has
the flow in it, at the cost of a coarser flow physics:

- flow: Darcy through the whole duct (Borrvall & Petersson's porosity
  interpolation: inverse permeability alpha(rho) = alpha_f + (alpha_s -
  alpha_f) rho (1 + q) / (rho + q)), driven by a fixed pressure drop
  between the duct's inlet and outlet faces. Darcy has no wall friction, so
  the open duct's alpha_f is a calibration: set it so a known design (plate
  fins) passes what the lattice Boltzmann rating says it does at that
  pressure;
- heat: conduction in metal and air (k(rho) = k_f + (k_s - k_f) rho^3) and
  advection by the Darcy flux, first-order upwind, the air entering at 0 and
  leaving through the outlet face; heat in on given cells;
- objective: the mean temperature of the heated cells; one volume
  constraint; density filter, Heaviside projection with continuation, MMA.

The gradient is the adjoint through both solves: the heat adjoint lambda
(A^T lambda = dJ/dT), then the flow adjoint mu (K mu = -d(lambda^T A T)/dp),
since the design moves the temperature both by conducting and by steering
the flow. Checked against finite differences in tests/test_flowtopo.py.

Box grids only (the Sink of bench/lattice_heatsink.py): `design`, `solid`,
`fluid` are bool masks on one grid; everything else is an adiabatic,
impermeable wall. Units: m inside, mm at the edges, Pa, W, K.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
from scipy import ndimage
from scipy.sparse.linalg import splu

from .mma import MMA

__all__ = ["FlowSink", "FlowTopoOptions", "optimize_flow_thermal", "calibrate_alpha"]

Q_RAMP = 0.1          # Borrvall-Petersson q: how grey cells resist flow
ALPHA_RATIO = 1e4     # solid's inverse permeability over open duct's
K_PENALTY = 3.0


class FlowSink:
    """The two linear problems and their adjoints on one voxel grid."""

    def __init__(self, design, solid, fluid, heat, h, k_solid, fluid_props, dp, alpha_f, flow_axis=0):
        self.shape = design.shape
        self.live = design | solid | fluid
        self.idx = -np.ones(self.shape, np.int64)
        self.n = int(self.live.sum())
        self.idx[self.live] = np.arange(self.n)
        self.design = design[self.live]
        self.fixed_rho = np.where(solid[self.live], 1.0, 0.0)
        self.heat = heat[self.live]
        self.hm = h * 1e-3
        self.ks, self.kf = k_solid, fluid_props.k
        self.rc = fluid_props.rho * fluid_props.cp
        self.dp = float(dp)
        self.alpha_f = float(alpha_f)
        a, b = [], []
        for ax in range(3):
            s0 = [slice(None)] * 3
            s1 = [slice(None)] * 3
            s0[ax] = slice(0, -1)
            s1[ax] = slice(1, None)
            both = self.live[tuple(s0)] & self.live[tuple(s1)]
            a.append(self.idx[tuple(s0)][both])
            b.append(self.idx[tuple(s1)][both])
        self.fa, self.fb = np.concatenate(a), np.concatenate(b)
        lo = [slice(None)] * 3
        hi = [slice(None)] * 3
        lo[flow_axis], hi[flow_axis] = 0, -1
        self.inlet = self.idx[tuple(lo)][fluid[tuple(lo)]]
        self.outlet = self.idx[tuple(hi)][fluid[tuple(hi)]]

    # -- interpolations
    ratio = ALPHA_RATIO

    def alpha(self, r):
        return self.alpha_f * (1.0 + (self.ratio - 1.0) * r * (1 + Q_RAMP) / (r + Q_RAMP))

    def dalpha(self, r):
        return self.alpha_f * (self.ratio - 1.0) * (1 + Q_RAMP) * Q_RAMP / (r + Q_RAMP) ** 2

    def k(self, r):
        return self.kf + (self.ks - self.kf) * r ** K_PENALTY

    def dk(self, r):
        return (self.ks - self.kf) * K_PENALTY * r ** (K_PENALTY - 1)

    def rho(self, xd):
        r = self.fixed_rho.copy()
        r[self.design] = xd
        return r

    # -- solves
    def solve(self, r):
        n, fa, fb, hm = self.n, self.fa, self.fb, self.hm
        al = self.alpha(r)
        G = 2.0 * hm / (al[fa] + al[fb])                  # m^3/(s Pa) per face (Darcy, mu in alpha)
        Gin = 2.0 * hm / al[self.inlet]
        Gout = 2.0 * hm / al[self.outlet]
        K = sp.coo_matrix((np.concatenate([G, G, -G, -G, Gin, Gout]),
                           (np.concatenate([fa, fb, fa, fb, self.inlet, self.outlet]),
                            np.concatenate([fa, fb, fb, fa, self.inlet, self.outlet]))), shape=(n, n)).tocsc()
        bvec = np.zeros(n)
        np.add.at(bvec, self.inlet, Gin * self.dp)
        Klu = splu(K)
        p = Klu.solve(bvec)
        Q = G * (p[fa] - p[fb])
        F = self.rc * Q
        Fo = self.rc * Gout * p[self.outlet]
        kk = self.k(r)
        g = 2.0 * kk[fa] * kk[fb] / (kk[fa] + kk[fb]) * hm
        fp, fm = np.maximum(F, 0.0), np.maximum(-F, 0.0)
        rows = np.concatenate([fa, fb, fa, fb, fa, fa, fb, fb])
        cols = np.concatenate([fb, fa, fa, fb, fa, fb, fb, fa])
        vals = np.concatenate([-g, -g, g, g, fp, -fm, fm, -fp])
        A = sp.coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsr()
        A = (A + sp.coo_matrix((Fo, (self.outlet, self.outlet)), shape=(n, n))).tocsc()
        q = np.zeros(n)
        q[self.heat] = 1.0 / self.heat.sum()               # 1 W; T scales with power
        Alu = splu(A)
        T = Alu.solve(q)
        return dict(r=r, al=al, G=G, Gin=Gin, Gout=Gout, p=p, F=F, Fo=Fo, kk=kk, g=g, T=T, Klu=Klu, Alu=Alu,
                    flow=float(Gin @ (self.dp - p[self.inlet])))

    def objective(self, s):
        """Mean temperature rise of the heated cells, K per W."""
        return float(s["T"][self.heat].mean())

    def gradient(self, s):
        """dJ/drho on every live cell (the adjoint of both solves)."""
        fa, fb, hm, rc = self.fa, self.fb, self.hm, self.rc
        T, p, F = s["T"], s["p"], s["F"]
        dJdT = np.zeros(self.n)
        dJdT[self.heat] = 1.0 / self.heat.sum()
        lam = s["Alu"].solve(dJdT, trans="T")
        dl = lam[fa] - lam[fb]
        up = np.where(F >= 0, T[fa], T[fb])
        sf = dl * up                                        # d(lam^T A T)/dF per face
        so = lam[self.outlet] * T[self.outlet]
        # flow adjoint: K mu = -w, w = d(lam^T A T)/dp
        w = np.zeros(self.n)
        np.add.at(w, fa, rc * sf * s["G"])
        np.add.at(w, fb, -rc * sf * s["G"])
        np.add.at(w, self.outlet, rc * so * s["Gout"])
        mu = s["Klu"].solve(-w)
        r, kk, al = s["r"], s["kk"], s["al"]
        grad = np.zeros(self.n)
        # conduction
        term = (T[fa] - T[fb]) * dl
        ka, kb = kk[fa], kk[fb]
        dga = 2.0 * kb ** 2 / (ka + kb) ** 2 * hm
        dgb = 2.0 * ka ** 2 / (ka + kb) ** 2 * hm
        dk = self.dk(r)
        np.add.at(grad, fa, -term * dga * dk[fa])
        np.add.at(grad, fb, -term * dgb * dk[fb])
        # flow conductances (through F at fixed p, and the flow residual)
        da = self.dalpha(r)
        dG = -2.0 * hm / (al[fa] + al[fb]) ** 2           # dG/dalpha_a = dG/dalpha_b
        dpf = p[fa] - p[fb]
        cf = -(sf * rc * dpf + (mu[fa] - mu[fb]) * dpf) * dG
        np.add.at(grad, fa, cf * da[fa])
        np.add.at(grad, fb, cf * da[fb])
        i, o = self.inlet, self.outlet
        dGi = -2.0 * hm / al[i] ** 2
        dGo = -2.0 * hm / al[o] ** 2
        np.add.at(grad, i, -(mu[i] * (p[i] - self.dp)) * dGi * da[i])
        np.add.at(grad, o, -(so * rc * p[o] + mu[o] * p[o]) * dGo * da[o])
        return grad


def calibrate_alpha(solid_of_design, fluid_of_design, h, dp, flow_lpm, flow_axis=0):
    """alpha_f for which a given (crisp) design passes `flow_lpm` at `dp` Pa
    in the Darcy model. Darcy is linear and the solid barely conducts flow,
    so one solve at alpha_f = 1 and a rescale does it."""
    from .heatx_thermal import FLUIDS

    fs = FlowSink(np.zeros_like(solid_of_design), solid_of_design, fluid_of_design, solid_of_design, h,
                  1.0, FLUIDS["air"], dp, 1.0, flow_axis)
    s = fs.solve(fs.fixed_rho)
    return s["flow"] / (flow_lpm / 60000.0)


@dataclass
class FlowTopoOptions:
    volume_fraction: float
    filter_radius: float = 2.0         # mm
    max_iterations: int = 120
    beta_max: float = 16.0
    beta_every: int = 15
    move: float = 0.1
    # The solid's flow resistance grows from 10x the open duct's to
    # ALPHA_RATIO over this many iterations. At full contrast from the first
    # iteration a uniform grey start chokes the duct (alpha at rho 0.2 is
    # ~7000x open), the flow and its gradient vanish, and the run settled at
    # 13.7 K/W where plate fins read 4.2 in the same model.
    ratio_iterations: int = 60
    x0: np.ndarray | None = None       # warm start, densities on the design cells
    # hold the volume exactly (two-sided), for comparisons at equal metal;
    # otherwise it is an upper bound and the optimizer may use less (more
    # open duct, more flow)
    equal_volume: bool = False
    beta0: float = 1.0                 # projection sharpness at the start (raise it for a crisp warm start)


def _cone(radius_cells):
    r = int(np.floor(radius_cells))
    g = np.mgrid[-r:r + 1, -r:r + 1, -r:r + 1]
    w = np.maximum(0.0, radius_cells - np.sqrt((g ** 2).sum(0)))
    return w


def _project(x, beta, eta=0.5):
    d = np.tanh(beta * eta) + np.tanh(beta * (1 - eta))
    return (np.tanh(beta * eta) + np.tanh(beta * (x - eta))) / d


def _dproject(x, beta, eta=0.5):
    d = np.tanh(beta * eta) + np.tanh(beta * (1 - eta))
    return beta * (1 - np.tanh(beta * (x - eta)) ** 2) / d


def optimize_flow_thermal(fs: FlowSink, design_grid: np.ndarray, h: float, opts: FlowTopoOptions,
                          on_iter=None) -> dict:
    """Lay out volume_fraction of the design cells to minimize the heated
    cells' mean temperature at the fixed pressure drop. Returns {"x": filtered
    and projected density on the grid (design cells), "objective", "history"}."""
    t0 = time.time()
    w = _cone(opts.filter_radius / h)
    norm = ndimage.convolve(design_grid.astype(float), w, mode="constant")
    norm = np.where(design_grid, norm, 1.0)
    sel = design_grid
    nd = int(sel.sum())

    def filt(xd):
        g = np.zeros(design_grid.shape)
        g[sel] = xd
        return (ndimage.convolve(g, w, mode="constant") / norm)[sel]

    def filt_t(gd):
        g = np.zeros(design_grid.shape)
        g[sel] = gd
        return ndimage.convolve(g / norm, w, mode="constant")[sel]

    # live-cell order of the design cells, matching design_grid[sel] order
    order = fs.idx[sel]
    x = np.full(nd, opts.volume_fraction) if opts.x0 is None else np.clip(np.asarray(opts.x0, float), 0, 1)
    mma = MMA(nd, 2 if opts.equal_volume else 1, move=opts.move)
    beta, since = float(opts.beta0), 0
    hist = []
    for it in range(1, opts.max_iterations + 1):
        frac = 1.0 if opts.ratio_iterations <= 0 else min(1.0, (it - 1) / opts.ratio_iterations)
        fs.ratio = 10.0 ** (1.0 + (np.log10(ALPHA_RATIO) - 1.0) * frac)
        xt = filt(x)
        xp = _project(xt, beta)
        r = fs.fixed_rho.copy()
        r[order] = xp
        s = fs.solve(r)
        J = fs.objective(s)
        gr = fs.gradient(s)[order]
        dJ = filt_t(gr * _dproject(xt, beta))
        vol = xp.mean() / opts.volume_fraction - 1.0
        dv = filt_t(_dproject(xt, beta) / (nd * opts.volume_fraction))
        if it == 1 or frac < 1.0:
            # the model stiffens under the optimizer while the ratio climbs
            scale = 1.0 / max(abs(J), 1e-12)
        if opts.equal_volume:
            xn = mma.update(x, J * scale, dJ * scale, np.array([vol, -vol]), np.stack([dv, -dv]))
        else:
            xn = mma.update(x, J * scale, dJ * scale, np.array([vol]), dv[None, :])
        change = float(np.abs(xn - x).max())
        x = xn
        since += 1
        rec = {"iteration": it, "objective": J, "flowLpm": s["flow"] * 60000.0, "volume": float(xp.mean()),
               "beta": beta, "change": change, "seconds": time.time() - t0}
        hist.append(rec)
        if on_iter:
            on_iter(rec)
        if beta < opts.beta_max and (since >= opts.beta_every or change < 0.01):
            beta, since = min(2 * beta, opts.beta_max), 0
        elif beta >= opts.beta_max and change < 0.005 and frac >= 1.0 and it >= 10:
            break
    xt = filt(x)
    xp = _project(xt, beta)
    g = np.zeros(design_grid.shape)
    g[sel] = xp
    fs.ratio = ALPHA_RATIO
    return {"x": g, "objective": hist[-1]["objective"], "history": hist, "seconds": time.time() - t0}
