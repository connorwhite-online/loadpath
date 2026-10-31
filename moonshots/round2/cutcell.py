"""
Cut-cell voxel elasticity with surface stress recovery (moonshot prototype).

The voxel analysis in verify.py keeps a cell when its centre is inside the
part and gives it the full cube stiffness: a staircase boundary, members a
whole number of cells thick, and stress read at staircase nodes. This module
keeps the same structured hex grid and solver but changes three things,
each switchable so the benchmark can attribute the error:

geometry  "stair": verify.py's cells (centre inside), full stiffness.
          "cutS" : every cell with fill >= tau, stiffness scaled by fill.
          "cutQ" : every cell with fill >= tau, stiffness integrated over
                   the n^3 subcells whose centres are inside (finite-cell
                   style; each inside subcell integrated exactly, 2x2x2 Gauss).
element   "std"  : trilinear hex8, full 2x2x2 Gauss (fem.hex8_ke).
          "im"   : hex8 with Wilson-Taylor incompatible modes (9 internal
                   bubble dofs, statically condensed): exact in pure bending
                   on a cube, where the plain hex8 shear-locks. On a cut cell
                   the bubble strains are made mean-free over the material
                   (Taylor's correction on the material domain) so the
                   constant-strain patch test still holds.
recovery  "nodal": verify.py's: corner stress averaged to nodes, max over nodes.
          "surf" : stress evaluated at points ON the true surface, inside the
                   cell holding each point (its own displacement field,
                   bubbles included), no averaging across the staircase.

Everything is in the cell's natural frame; the cube side is h.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from loadpath.fem import CORNERS, HexMesh, Stiffness, _b_matrix, elastic_c

GAUSS = 1.0 / np.sqrt(3.0)


def b_inc(xi, eta, zeta):
    """Strain of the 9 incompatible modes (1-xi^2) etc. per component, unit
    cube (d/dx = 2 d/dxi): (6, 9), columns 3*mode + component."""
    x = (xi, eta, zeta)
    d = np.zeros((3, 3))           # d[mode, axis]
    for m in range(3):
        d[m, m] = -4.0 * x[m]
    b = np.zeros((6, 9))
    for m in range(3):
        dx, dy, dz = d[m]
        col = 3 * m
        b[0, col] = dx
        b[1, col + 1] = dy
        b[2, col + 2] = dz
        b[3, col], b[3, col + 1] = dy, dx
        b[4, col + 1], b[4, col + 2] = dz, dy
        b[5, col], b[5, col + 2] = dz, dx
    return b


def b_batch(xi):
    """_b_matrix at many natural points: (n, 6, 24), unit cube."""
    xi = np.asarray(xi, float)
    sg = 2.0 * CORNERS - 1.0                          # (8, 3)
    f = 1.0 + xi[:, None, :] * sg[None]               # (n, 8, 3)
    dn = np.empty((len(xi), 8, 3))
    dn[..., 0] = 0.25 * sg[None, :, 0] * f[..., 1] * f[..., 2]
    dn[..., 1] = 0.25 * sg[None, :, 1] * f[..., 0] * f[..., 2]
    dn[..., 2] = 0.25 * sg[None, :, 2] * f[..., 0] * f[..., 1]
    return _strain_rows(dn)


def b_inc_batch(xi):
    """b_inc at many natural points: (n, 6, 9)."""
    xi = np.asarray(xi, float)
    dn = np.zeros((len(xi), 3, 3))
    for m in range(3):
        dn[:, m, m] = -4.0 * xi[:, m]
    return _strain_rows(dn)


def _strain_rows(dn):
    n, k, _ = dn.shape
    b = np.zeros((n, 6, 3 * k))
    dx, dy, dz = dn[..., 0], dn[..., 1], dn[..., 2]
    b[:, 0, 0::3], b[:, 1, 1::3], b[:, 2, 2::3] = dx, dy, dz
    b[:, 3, 0::3], b[:, 3, 1::3] = dy, dx
    b[:, 4, 1::3], b[:, 4, 2::3] = dz, dy
    b[:, 5, 0::3], b[:, 5, 2::3] = dz, dx
    return b


class SubcellRule:
    """Per subcell of an n^3 split of the unit cube: integrals (2x2x2 Gauss
    inside the subcell) of the pieces of the element matrices, so a cut
    cell's matrices are an indicator (n^3,) times these."""

    def __init__(self, nu, n=4):
        self.n = n
        c = elastic_c(1.0, nu)
        self.c = c
        ns = n ** 3
        self.S_uu = np.zeros((ns, 24, 24))
        self.S_ua = np.zeros((ns, 24, 9))
        self.S_aa = np.zeros((ns, 9, 9))
        self.P_u = np.zeros((ns, 6, 24))
        self.P_a = np.zeros((ns, 6, 9))
        self.vol = np.full(ns, 1.0 / ns)
        # subcell centres in natural coords, and their (i,j,k)
        ii = np.arange(n)
        sc = -1.0 + (2 * ii + 1) / n
        self.centers = np.stack(np.meshgrid(sc, sc, sc, indexing="ij"), -1).reshape(-1, 3)
        w = 0.125 / ns                        # Gauss weight in the unit cube
        half = 1.0 / n
        for s, (a, b_, cc) in enumerate(self.centers):
            for ga in (-GAUSS, GAUSS):
                for gb in (-GAUSS, GAUSS):
                    for gc in (-GAUSS, GAUSS):
                        p = (a + ga * half, b_ + gb * half, cc + gc * half)
                        bu = _b_matrix(*p)
                        ba = b_inc(*p)
                        self.S_uu[s] += w * bu.T @ c @ bu
                        self.S_ua[s] += w * bu.T @ c @ ba
                        self.S_aa[s] += w * ba.T @ c @ ba
                        self.P_u[s] += w * bu
                        self.P_a[s] += w * ba

    def matrices(self, ind, element="std"):
        """Element matrices (unit E, unit h) for cells with subcell indicator
        ind (ncell, n^3) bool/float. Returns (ke (ncell,24,24), R or None,
        mbar or None): R maps u_e to the bubble amplitudes, mbar is the mean
        bubble strain subtracted on that cell (6,9)."""
        ind = np.asarray(ind, float)
        kuu = np.einsum("es,sij->eij", ind, self.S_uu)
        if element == "std":
            return kuu, None, None
        v = ind @ self.vol                                   # material volume
        pa = np.einsum("es,sij->eij", ind, self.P_a)         # int Ba
        pu = np.einsum("es,sij->eij", ind, self.P_u)         # int Bu
        mbar = pa / v[:, None, None]                         # mean Ba
        c = self.c
        # Ba' = Ba - mbar:  int Bu^T C Ba' = S_ua - Pu^T C mbar
        kua = np.einsum("es,sij->eij", ind, self.S_ua) - np.einsum("eki,kl,elj->eij", pu, c, mbar)
        # int Ba'^T C Ba' = S_aa - Pa^T C mbar - mbar^T C Pa + v mbar^T C mbar
        kaa = (np.einsum("es,sij->eij", ind, self.S_aa)
               - np.einsum("eki,kl,elj->eij", pa, c, mbar)
               - np.einsum("eki,kl,elj->eij", mbar, c, pa)
               + v[:, None, None] * np.einsum("eki,kl,elj->eij", mbar, c, mbar))
        kaa_inv = np.linalg.inv(kaa)
        r = -np.einsum("eij,ekj->eik", kaa_inv, kua)          # (e, 9, 24)
        ke = kuu + np.einsum("eij,ejk->eik", kua, r)
        ke = 0.5 * (ke + ke.transpose(0, 2, 1))
        return ke, r, mbar


def fill_from_inside(inside, origin, dims, h, n=4, chunk=200_000):
    """Fill fraction and subcell indicator per cell of a grid from a vectorized
    point-inside test. Returns (fill (dims), ind (ncells, n^3) for all cells
    in C order). Only for small grids (benchmark)."""
    ncell = int(np.prod(dims))
    ijk = np.stack(np.unravel_index(np.arange(ncell), dims), 1)
    sub = (np.stack(np.meshgrid(*(np.arange(n),) * 3, indexing="ij"), -1).reshape(-1, 3) + 0.5) / n
    ind = np.zeros((ncell, n ** 3), bool)
    for s0 in range(0, ncell, max(1, chunk // n ** 3)):
        e = slice(s0, min(ncell, s0 + max(1, chunk // n ** 3)))
        pts = origin + (ijk[e, None, :] + sub[None]) * h
        ind[e] = inside(pts.reshape(-1, 3)).reshape(-1, n ** 3)
    return ind.mean(1).reshape(dims), ind


def centre_inside(inside, origin, dims, h):
    ncell = int(np.prod(dims))
    ijk = np.stack(np.unravel_index(np.arange(ncell), dims), 1)
    return inside(origin + (ijk + 0.5) * h).reshape(dims)


class CutModel:
    """One analysis on a grid: which cells, their element matrices, and the
    stress recovery."""

    def __init__(self, solid_ijk, h, origin, nu, E, ind=None, geometry="stair", element="std",
                 rule: SubcellRule | None = None):
        self.mesh = HexMesh(solid_ijk, h, origin)
        self.h, self.nu, self.E = h, nu, E
        self.geometry, self.element = geometry, element
        ne = self.mesh.n_elem
        n3 = rule.n ** 3 if rule is not None else 1
        if ind is None or geometry == "stair":
            ind = np.ones((ne, n3), bool)
        self.ind = np.asarray(ind, bool)
        self.fill = self.ind.mean(1)
        self.rule = rule
        self.c = elastic_c(E, nu)
        if geometry == "cutS":
            # whole-cell matrices scaled by the fill (bubble stays whole-cell)
            full = np.ones((1, n3), bool)
            ke1, r1, m1 = rule.matrices(full, element)
            self.ke = ke1  # (1,24,24), scaled at assembly
            self.r = None if r1 is None else np.repeat(r1, ne, 0)
            self.mbar = None if m1 is None else np.repeat(m1, ne, 0)
            self.scale = self.fill.copy()
        else:
            # full cells share one matrix; cut cells get their own
            full = self.fill >= 1.0 - 1e-12
            ke1, r1, m1 = rule.matrices(np.ones((1, n3), bool), element)
            self.ke = np.repeat(ke1, ne, 0)
            self.r = None if r1 is None else np.repeat(r1, ne, 0)
            self.mbar = None if m1 is None else np.repeat(m1, ne, 0)
            cut = np.nonzero(~full)[0]
            for s0 in range(0, len(cut), 4000):
                cc = cut[s0:s0 + 4000]
                k, r, m = rule.matrices(self.ind[cc], element)
                self.ke[cc] = k
                if r is not None:
                    self.r[cc], self.mbar[cc] = r, m
            self.scale = np.ones(ne)

    def stiffness(self, fixed):
        st = Stiffness(self.mesh, self.nu, fixed)
        data = np.zeros(st.nnz)
        ke = self.ke * (self.E * self.h)
        if ke.shape[0] == 1:
            kes = ke * self.scale[:, None, None]
        else:
            kes = ke
        for s0 in range(0, self.mesh.n_elem, 20000):
            e = np.arange(s0, min(self.mesh.n_elem, s0 + 20000))
            st.add_elements(data, kes[e], e)
        return st, st.finish(data)

    def solve_many(self, k, fs):
        """Several right-hand sides on an assembled K (Dirichlet rows as
        Stiffness.finish leaves them): loadpath's CG + rigid-mode AMG."""
        from loadpath.fem import Solver

        solver = Solver(self.mesh, rtol=1e-8, maxiter=6000, rebuild_after_iters=10_000)
        return solver.solve_many(k, fs)

    def solve(self, f, fixed):
        st, k = self.stiffness(fixed)
        free = ~fixed
        u = np.zeros(self.mesh.n_dof)
        kff = k[free][:, free].tocsc()
        if kff.shape[0] < 150_000:
            from scipy.sparse.linalg import spsolve
            u[free] = spsolve(kff, f[free])
        else:
            from scipy.sparse.linalg import cg

            from loadpath.fem import amg_solver
            kff = kff.tocsr()
            ml = amg_solver(kff, symmetry="hermitian", strength="symmetric", max_coarse=500, coarse_solver="splu")
            u[free], info = cg(kff, f[free], rtol=1e-8, maxiter=5000, M=ml.aspreconditioner(cycle="V"))
            if info != 0:
                raise RuntimeError(f"cg info {info}")
        self.u = u
        return u

    # --- stress -----------------------------------------------------------
    def stress_at(self, elems, xi):
        """Stress (n, 6) in element elems[i] at natural point xi[i] (h-scaled)."""
        ue = self.u[self.mesh.edof[elems]]
        out = np.empty((len(elems), 6))
        for i0 in range(0, len(elems), 20000):
            sl = slice(i0, i0 + 20000)
            bu = b_batch(xi[sl]) / self.h
            eps = np.einsum("nij,nj->ni", bu, ue[sl])
            if self.element == "im":
                ba = b_inc_batch(xi[sl]) / self.h
                ba = ba - self.mbar[elems[sl]] / self.h
                alpha = np.einsum("nij,nj->ni", self.r[elems[sl]], ue[sl])
                eps = eps + np.einsum("nij,nj->ni", ba, alpha)
            out[sl] = eps @ self.c.T
        return out

    def nodal_stress(self, weight_by_fill=True):
        """Corner stress averaged over the elements at each node (verify.py's
        recovery), weighted by fill when asked."""
        m = self.mesh
        acc = np.zeros((m.n_nodes, 6))
        wsum = np.zeros(m.n_nodes)
        w = self.fill if weight_by_fill else np.ones(m.n_elem)
        e = np.arange(m.n_elem)
        for k, corner in enumerate(2.0 * CORNERS - 1.0):
            s = self.stress_at(e, np.repeat(corner[None], m.n_elem, 0))
            np.add.at(acc, m.enode[:, k], w[:, None] * s)
            np.add.at(wsum, m.enode[:, k], w)
        return acc / np.maximum(wsum, 1e-30)[:, None]

    def locate(self, pts, min_fill=0.0, reach=1.0):
        """For each point: an active element holding it (or the nearest active
        one among the 27 around it, preferring the fuller) and the natural
        coordinates of the point in it, clipped to `reach` (1 = the cell).
        With min_fill, only cells at least that full are candidates: a
        point in a sliver is read from its fuller neighbour, its field
        extended (reach > 1) to the point."""
        m = self.mesh
        rel = (np.asarray(pts, float) - m.origin) / self.h
        base = np.floor(rel).astype(np.int64)
        dims = m.elem_ijk.max(0) + 3
        flat = (m.elem_ijk[:, 0] * dims[1] + m.elem_ijk[:, 1]) * dims[2] + m.elem_ijk[:, 2]
        table = np.full(int(np.prod(dims)), -1, np.int64)
        table[flat] = np.arange(m.n_elem)
        best = np.full(len(pts), -1, np.int64)
        bestd = np.full(len(pts), np.inf)
        offs = np.array([[a, b, c] for a in (0, -1, 1) for b in (0, -1, 1) for c in (0, -1, 1)])
        for o in offs:
            ijk = base + o
            ok = np.all((ijk >= 0) & (ijk < dims), 1)
            el = np.full(len(pts), -1, np.int64)
            fl = (ijk[ok, 0] * dims[1] + ijk[ok, 1]) * dims[2] + ijk[ok, 2]
            el[ok] = table[fl]
            has = el >= 0
            if min_fill > 0:
                has &= self.fill[np.maximum(el, 0)] >= min_fill
            # distance from point to the cell box, less a hair per fill
            lo = ijk.astype(float)
            gap = np.maximum(lo - rel, 0) + np.maximum(rel - (lo + 1), 0)
            d = np.linalg.norm(gap, axis=1)
            d = d - 1e-3 * np.where(has, self.fill[np.maximum(el, 0)], 0)
            better = has & (d < bestd - 1e-12)
            best[better], bestd[better] = el[better], d[better]
        xi = np.clip(2.0 * (rel - m.elem_ijk[np.maximum(best, 0)]) - 1.0, -reach, reach)
        return best, xi, bestd

    def surface_stress(self, pts, min_fill=0.0, reach=1.0):
        el, xi, d = self.locate(pts, min_fill, reach)
        ok = el >= 0
        s = np.full((len(pts), 6), np.nan)
        s[ok] = self.stress_at(el[ok], xi[ok])
        return s

    def nodal_interp(self, pts, ns):
        """Trilinear interpolation of a nodal field (n_nodes, 6) at points,
        inside the located element."""
        el, xi, _ = self.locate(pts)
        ok = el >= 0
        out = np.full((len(pts), 6), np.nan)
        e, x = el[ok], xi[ok]
        s = 2.0 * CORNERS - 1.0
        n = 0.125 * np.prod(1 + x[:, None, :] * s[None], axis=2)      # (p, 8)
        out[ok] = np.einsum("pk,pkc->pc", n, ns[self.mesh.enode[e]])
        return out
