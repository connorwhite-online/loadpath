"""
Elastic buckling of a lattice's walls and struts, from a periodic unit-cell
eigenvalue solve.

The homogenized check reads a lattice's first yield from its walls' linear
stress. A thin wall or a slender strut can instead buckle first, under a
compressive macro stress well below the one that yields it, and the linear
check does not see that. Here a block of `cells`^3 unit cells on a periodic
voxel grid (the nodes wrap across the block's faces, as lattice_tensors'
solves do) is put under a unit macro stress, and the linear eigenvalue
problem

    K phi = lambda (-K_sigma) phi

gives the smallest load factor lambda at which the walls buckle: the macro
stress, in units of the solid's modulus, at which a perfect infinite lattice
buckles. A one-cell block finds the modes that repeat every cell, a two-cell
block also the ones that alternate between neighbours (often lower); the
long-wave (whole-part, Euler) modes are not in it, a part-level check
(nonlinear.linear_buckling) covers those.

K uses the element in force (the incompatible-mode hex8, which bends without
locking at a few voxels through a wall); K_sigma is the trilinear geometric
stiffness of the linear stress at the Gauss points.

Tables over relative density per cell type and load direction live in
data/lattice_buckling.json (bench/lattice_buckling.py writes them);
`buckling_stress` reads them. Real walls are not perfect: printed walls are
wavy, porous and stair-stepped, and shells lose much of their perfect
buckling load to that. How the check in homogenized.py knocks the perfect
value down (with the as-printed stiffness, lattice_print.py) is documented
there and in docs/VALIDATION.md.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np

__all__ = ["unit_cell_buckling", "buckling_stress", "has_buckling", "TABLE_PATH", "DIRECTIONS"]

TABLE_PATH = Path(__file__).resolve().parent / "data" / "lattice_buckling.json"

# macro uniaxial compression directions the tables hold, in the cell's axes
DIRECTIONS = {"100": (0.0, 0.0, 1.0), "111": (1.0, 1.0, 1.0)}


def _grid(kind, wall_over_cell, n, cells):
    from .lattice import tpms
    from .struts import is_strut, unit_cell_distance

    m = n * cells
    h = 1.0 / n
    axes = (np.arange(m) + 0.5) * h
    g = np.stack(np.meshgrid(axes, axes, axes, indexing="ij"), -1).reshape(-1, 3)
    if is_strut(kind):
        u = g % 1.0
        solid = unit_cell_distance(kind, u, 0.0) < 0.5 * wall_over_cell
    else:
        solid = np.abs(tpms(kind, g, 1.0)) < 0.5 * wall_over_cell
    return np.argwhere(solid.reshape(m, m, m)), m, h


def unit_cell_buckling(kind: str, wall_over_cell: float, direction=(0.0, 0.0, 1.0), n: int | None = None,
                       cells: int = 1, nu: float = 0.35, n_modes: int = 2, direct_max_dof: int = 60_000) -> dict:
    """Perfect-lattice elastic buckling under uniaxial macro compression
    along `direction` (cell axes): the critical macro stress over the
    solid's modulus (`sigmaOverE`), the relative density of the voxel cell,
    and its modulus along the direction (over E), for a block of
    `cells`^3 periodic unit cells, `n` voxels per cell (default: five
    voxels through the wall or strut, 24-72)."""
    import time

    import scipy.sparse as sp
    from scipy.sparse.linalg import LinearOperator, eigsh

    from .fem import CORNERS, elastic_c, hex8_b, hex8_ke

    t0 = time.time()
    if n is None:
        n = int(np.clip(np.ceil(5.0 / float(wall_over_cell)), 24, 72))
    eijk, m, h = _grid(kind, float(wall_over_cell), n, cells)
    corner = (eijk[:, None, :] + CORNERS[None]) % m
    eijk, corner = _connected(eijk, corner, m)
    ne = len(eijk)
    rho = ne / float(m ** 3)
    used, inv = np.unique((corner[..., 0] * m + corner[..., 1]) * m + corner[..., 2], return_inverse=True)
    enode = inv.reshape(ne, 8)
    nn = len(used)
    edof = (3 * enode[:, :, None] + np.arange(3)).reshape(ne, 24)
    rows = np.repeat(edof, 24, axis=1).ravel()
    cols = np.tile(edof, (1, 24)).ravel()
    ke = hex8_ke(nu) * h
    k = sp.csr_matrix((np.tile(ke.ravel(), ne), (rows, cols)), shape=(3 * nn, 3 * nn))
    # keep the block from floating: hold one node; the rest is periodic
    free = np.ones(3 * nn, bool)
    free[:3] = False
    kf = k[free][:, free].tocsc()
    nf = int(free.sum())
    kinv = _factorized(kf, nf <= direct_max_dof)
    xc = (eijk[:, None, :] + CORNERS[None]) * h
    c0 = elastic_c(1.0, nu)
    vol = float(cells ** 3)
    bc = hex8_b(0.0, 0.0, 0.0, nu) / h

    def macro_solve(e):
        eps = np.array([[e[0], e[3] / 2, e[5] / 2], [e[3] / 2, e[1], e[4] / 2], [e[5] / 2, e[4] / 2, e[2]]])
        u0 = (xc @ eps.T).reshape(ne, 24)
        f = np.zeros(3 * nn)
        np.add.at(f, edof, -(u0 @ ke.T))
        u = np.zeros(3 * nn)
        u[free] = kinv(f[free])
        ue = u[edof] + u0
        return ue, ((ue @ bc.T) @ c0.T).sum(axis=0) * h ** 3 / vol

    # effective stiffness, then the macro strain of a unit uniaxial compression
    C = np.zeros((6, 6))
    fields = []
    for comp in range(6):
        e = np.zeros(6)
        e[comp] = 1.0
        ue, s = macro_solve(e)
        C[:, comp] = s
        fields.append(ue)
    d = np.asarray(direction, float)
    d = d / np.linalg.norm(d)
    sig = -np.array([d[0] ** 2, d[1] ** 2, d[2] ** 2, d[0] * d[1], d[1] * d[2], d[2] * d[0]])
    eps = np.linalg.solve(C, sig)
    ue = sum(eps[c] * fields[c] for c in range(6))
    e_dir = 1.0 / float(-sig @ np.linalg.solve(C, -sig))     # modulus along d, over E
    # geometric stiffness of the linear stress at the 2x2x2 Gauss points
    g = 1.0 / np.sqrt(3.0)
    s = 2.0 * CORNERS - 1.0
    w = h ** 3 / 8.0
    kq = np.zeros((ne, 8, 8))
    for xi in (-g, g):
        for eta in (-g, g):
            for zeta in (-g, g):
                dn = np.empty((8, 3))
                dn[:, 0] = 0.125 * s[:, 0] * (1 + eta * s[:, 1]) * (1 + zeta * s[:, 2]) * 2 / h
                dn[:, 1] = 0.125 * s[:, 1] * (1 + xi * s[:, 0]) * (1 + zeta * s[:, 2]) * 2 / h
                dn[:, 2] = 0.125 * s[:, 2] * (1 + xi * s[:, 0]) * (1 + eta * s[:, 1]) * 2 / h
                b = hex8_b(xi, eta, zeta, nu) / h
                sv = (ue @ b.T) @ c0.T                       # (ne, 6) Voigt, tensor shears
                st = np.empty((ne, 3, 3))
                st[:, 0, 0], st[:, 1, 1], st[:, 2, 2] = sv[:, 0], sv[:, 1], sv[:, 2]
                st[:, 0, 1] = st[:, 1, 0] = sv[:, 3]
                st[:, 1, 2] = st[:, 2, 1] = sv[:, 4]
                st[:, 0, 2] = st[:, 2, 0] = sv[:, 5]
                kq += w * np.einsum("ai,eij,bj->eab", dn, st, dn)
    kg_e = np.einsum("eab,ij->eaibj", kq, np.eye(3)).reshape(ne, 576)
    kg = sp.csr_matrix((kg_e.ravel(), (rows, cols)), shape=(3 * nn, 3 * nn))
    kgf = kg[free][:, free].tocsr()
    del kg_e, kq
    op = LinearOperator((nf, nf), matvec=kinv, dtype=float)
    from scipy.sparse.linalg import ArpackNoConvergence

    try:
        mu, _ = eigsh(-kgf, k=n_modes, M=kf.tocsr(), Minv=op, which="LA", tol=1e-4, ncv=max(20, 4 * n_modes),
                      maxiter=600, v0=np.random.default_rng(0).standard_normal(nf))
    except ArpackNoConvergence as e:
        # clustered modes (a cell whose struts buckle in many equal ways):
        # the converged ones, else the iteration's best estimate
        mu = e.eigenvalues if len(e.eigenvalues) else np.array([0.0])
    mu = np.sort(np.atleast_1d(mu))[::-1]
    lam = float(1.0 / mu[0]) if mu[0] > 1e-14 else float("inf")
    return {"kind": kind, "wallOverCell": float(wall_over_cell), "direction": d.tolist(), "cells": cells, "n": n,
            "rho": rho, "elements": ne, "dof": nf, "sigmaOverE": lam, "modulusOverE": e_dir,
            "strainAtBuckling": lam / e_dir, "seconds": time.time() - t0}


def _connected(eijk, corner, m):
    """The voxels of the largest piece joined through shared nodes (a thin
    strut can voxelize into islands that would float)."""
    import scipy.sparse as sp
    from scipy.sparse.csgraph import connected_components

    ne = len(eijk)
    key = ((corner[..., 0] * m + corner[..., 1]) * m + corner[..., 2]).ravel()
    _, node = np.unique(key, return_inverse=True)
    inc = sp.csr_matrix((np.ones(ne * 8), (np.repeat(np.arange(ne), 8), node)), shape=(ne, node.max() + 1))
    _, lab = connected_components(inc @ inc.T, directed=False)
    big = lab == np.argmax(np.bincount(lab))
    return eijk[big], corner[big]


def _factorized(kf, direct):
    """x = K^-1 r: PARDISO (pypardiso, nested-dissection ordering, threaded)
    when installed, else SuperLU for small systems and AMG-preconditioned
    CG above `direct_max_dof` (SuperLU's orderings fill in badly on 3D
    grids: a 170k-dof sheet cell took over 10 minutes and 8 GB)."""
    import scipy.sparse as sp

    try:
        import pypardiso

        from .nonlinear import mkl_threads, on_mkl_thread

        a = sp.csr_matrix(kf)
        solver = pypardiso.PyPardisoSolver()

        def factor():
            mkl_threads(solver)
            solver.factorize(a)

        # (on the one thread MKL is entered from: nonlinear.on_mkl_thread)
        on_mkl_thread(factor)
        return lambda r: on_mkl_thread(solver.solve, a, np.ascontiguousarray(r, float))
    except ImportError:
        pass
    if direct:
        from scipy.sparse.linalg import splu

        return splu(kf.tocsc(), permc_spec="MMD_AT_PLUS_A").solve
    import pyamg
    from scipy.sparse.linalg import cg

    ml = pyamg.smoothed_aggregation_solver(sp.csr_matrix(kf), symmetry="hermitian", max_coarse=400,
                                           coarse_solver="splu")
    pre = ml.aspreconditioner()
    return lambda r: cg(kf, r, rtol=1e-10, maxiter=4000, M=pre)[0]


@lru_cache(maxsize=1)
def _table() -> dict:
    if not TABLE_PATH.exists():
        return {}
    return json.loads(TABLE_PATH.read_text())


def has_buckling(kind: str) -> bool:
    return kind in _table().get("cells", {})


def buckling_stress(kind: str, rho, direction: str | None = None) -> np.ndarray:
    """Perfect-lattice buckling stress over the solid's modulus at relative
    densities `rho`: the table's least over its directions (or the one
    named), log-log in rho between rows, the end rows' power law beyond."""
    row = _table()["cells"][kind]
    xs, keep = np.unique(np.asarray(row["rho"], float), return_index=True)
    dirs = [direction] if direction else list(row["sigmaOverE"])
    ys = np.min([np.asarray(row["sigmaOverE"][dd], float)[keep] for dd in dirs], axis=0)
    lr, ly = np.log(xs), np.log(ys)
    x = np.log(np.clip(np.atleast_1d(np.asarray(rho, float)), 1e-4, 1.0))
    out = np.interp(x, lr, ly)
    if len(xs) > 1:
        lo = (ly[1] - ly[0]) / (lr[1] - lr[0])
        hi = (ly[-1] - ly[-2]) / (lr[-1] - lr[-2])
        out = np.where(x < lr[0], ly[0] + lo * (x - lr[0]), out)
        out = np.where(x > lr[-1], ly[-1] + hi * (x - lr[-1]), out)
    return np.exp(out)
