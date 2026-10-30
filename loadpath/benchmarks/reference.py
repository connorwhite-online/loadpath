"""
The reference optimizers: line-for-line NumPy ports of the published
educational codes, run on each benchmark's own supports and loads.

- top88: Andreassen, Clausen, Schevenels, Lazarov & Sigmund 2011, "Efficient
  topology optimization in MATLAB using 88 lines of code". 2D, SIMP p = 3,
  Emin = 1e-9, OC with move 0.2, sensitivity filter (ft=1) or density filter
  (ft=2), stops when no density moves more than 0.01. Passive void cells as in
  the paper's section 5.2.
- top3d: Liu & Tovar 2014, "An efficient 3D topology optimization code
  written in Matlab". The same scheme in 3D with the density filter and at
  most 200 iterations.

What is ported: the filter (truncated cone, normalized by its row sums),
the sensitivities, the OC bisection (l2 = 1e9, tolerance 1e-3 relative), the
volume measured on physical densities and the stopping rule. The finite
elements are the evaluator's, which are the same elements (top88's 4-node
plane-stress square, top3d's 8-node brick).

The designs are stored under reference/ as grids in loadpath's layout, so the
suite only re-runs a reference when asked to.
"""
from __future__ import annotations

import os
import time

import numpy as np
import scipy.sparse as sp

from .evaluate import E_VOID, Evaluator
from .problems import Problem

__all__ = ["SOLVERS", "run_reference", "load_reference", "reference_path"]

HERE = os.path.dirname(os.path.abspath(__file__))
SOLVERS = {
    "top88-ft1": "top88, sensitivity filter (Andreassen et al. 2011)",
    "top88-ft2": "top88, density filter (Andreassen et al. 2011)",
    "top3d": "top3d, density filter (Liu & Tovar 2014)",
}
# top88 itself has no iteration cap; this one only keeps a chattering run
# bounded. top3d ships with 200.
TOP88_MAXIT = 1000
TOP3D_MAXIT = 200


def _filter(ev: Evaluator, rmin: float):
    """top88's H and Hs: weights rmin - distance to every cell within
    ceil(rmin) - 1 cells per axis, normalized by the row sum."""
    ijk = np.stack(ev.cell, axis=1)
    dims = np.asarray(ev.p.dims)
    lut = np.full(ev.p.dims, -1, np.int64)
    lut[tuple(ijk.T)] = np.arange(len(ijk))
    reach = int(np.ceil(rmin)) - 1
    span = [range(-reach, reach + 1) if d > 1 else range(1) for d in dims]
    rows, cols, vals = [], [], []
    for dx in span[0]:
        for dy in span[1]:
            for dz in span[2]:
                w = rmin - np.sqrt(dx * dx + dy * dy + dz * dz)
                if w <= 0:
                    continue
                nb = ijk + (dx, dy, dz)
                ok = np.all((nb >= 0) & (nb < dims), axis=1)
                rows.append(np.flatnonzero(ok))
                cols.append(lut[tuple(nb[ok].T)])
                vals.append(np.full(int(ok.sum()), w))
    n = len(ijk)
    h = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n, n))
    return h, np.asarray(h.sum(axis=1)).ravel()


def optimize_reference(p: Problem, solver: str, on_iter=None) -> dict:
    ev = Evaluator(p)
    ft = 1 if solver == "top88-ft1" else 2
    maxit = TOP3D_MAXIT if solver == "top3d" else TOP88_MAXIT
    penal, move = 3.0, 0.2
    h, hs = _filter(ev, p.rmin)
    passive = ~ev.active
    n = len(hs)
    x = np.full(n, p.volfrac)
    x[passive] = 0.0
    xphys = x.copy()
    t0 = time.time()
    change, loop, c = 1.0, 0, 0.0
    while change > 0.01 and loop < maxit:
        loop += 1
        u = ev.solve(E_VOID + xphys ** penal * (1 - E_VOID))
        ue = u[ev.edof]
        ce = np.einsum("ij,jk,ik->i", ue, ev.ke, ue)
        c = float(np.sum((E_VOID + xphys ** penal * (1 - E_VOID)) * ce))
        dc = -penal * (1 - E_VOID) * xphys ** (penal - 1) * ce
        dv = np.ones(n)
        if ft == 1:
            dc = (h @ (x * dc)) / hs / np.maximum(1e-3, x)
        else:
            dc = h @ (dc / hs)
            dv = h @ (dv / hs)
        l1, l2 = 0.0, 1e9
        while (l2 - l1) / (l1 + l2) > 1e-3:
            lmid = 0.5 * (l2 + l1)
            xnew = np.maximum(0, np.maximum(x - move, np.minimum(1, np.minimum(x + move, x * np.sqrt(-dc / dv / lmid)))))
            xnew[passive] = 0.0
            xphys = xnew if ft == 1 else (h @ xnew) / hs
            xphys = np.where(passive, 0.0, xphys)
            if xphys.sum() > p.volfrac * n:
                l1 = lmid
            else:
                l2 = lmid
        change = float(np.abs(xnew - x).max())
        x = xnew
        if on_iter is not None:
            on_iter(loop, c, change)
    grid = np.zeros(p.dims)
    grid[ev.cell] = xphys
    return dict(x=grid, iterations=loop, compliance_reported=c, seconds=time.time() - t0, solver=solver)


def reference_path(p: Problem, solver: str) -> str:
    return os.path.join(HERE, "reference", f"{p.name}.{solver}.npz")


def run_reference(p: Problem, solver: str, on_iter=None) -> dict:
    rec = optimize_reference(p, solver, on_iter)
    os.makedirs(os.path.dirname(reference_path(p, solver)), exist_ok=True)
    np.savez_compressed(reference_path(p, solver), x=rec["x"].astype(np.float32), iterations=rec["iterations"],
                        compliance_reported=rec["compliance_reported"], seconds=rec["seconds"])
    return rec


def load_reference(p: Problem, solver: str) -> dict | None:
    path = reference_path(p, solver)
    if not os.path.exists(path):
        return None
    z = np.load(path)
    return dict(x=z["x"].astype(float), iterations=int(z["iterations"]),
                compliance_reported=float(z["compliance_reported"]), seconds=float(z["seconds"]), solver=solver)
