"""
Matrix-free geometric multigrid for the voxel elasticity problem, so the
optimizer can run on millions of cells instead of ~100k.

The assembled path (fem.Stiffness + pyamg) stores 81 x 3 stiffness entries
per node plus their slot tables: about 2 KB per cell, and pyamg's setup on
top of that. Here nothing per cell is stored but the cell's modulus:

- The fine operator is applied element by element on the dense node grid:
  gather the 8 corner displacements of every cell (8 slice copies), one GEMM
  with the shared 24x24 unit-cube stiffness, scale by the cell's modulus,
  scatter back. ~100 ns per cell, ~16 bytes of state per cell.
- Coarse levels are exact Galerkin operators P^T A P built element by
  element. With trilinear prolongation on a grid coarsened by two, a fine
  cell's nodes interpolate only from the coarse cell that contains it, so
  the coarse cell's matrix is the sum over its eight children of
  P_k^T K_child P_k. On the first coarse level that is sum_k E_child * M_k
  with eight fixed 24x24 matrices M_k: the level is stored as the fine
  moduli themselves (a reshape, no setup). Deeper levels store their 24x24
  matrices outright (1/64 of the cells and fewer).
- Supports: prolongation is masked to free dofs (P' = D P), so a coarse
  correction never moves a held dof and the coarse operator is the exact
  Galerkin product of the held system; cells touching a held node are the
  only ones whose coarse contribution is not a weight times M_k, and they
  are kept as an explicit patch.
- Smoothing is Chebyshev on the Jacobi-scaled operator (symmetric, no
  ordering, vectorizes), the coarsest level is assembled and solved by
  sparse LU or, if still large, one pyamg V-cycle. The V-cycle runs in
  float32; the outer conjugate gradient and its operator run in float64,
  so the converged answer is a float64 one.

Galerkin coarsening is what keeps this robust to SIMP's million-to-one
stiffness contrast: a coarse cell that straddles a strut and void sees the
strut's real stiffness, not an average modulus.

`GridElasticity` takes a fem.HexMesh, a Poisson's ratio (or fem.Layered), the
held dofs and optional springs, and solves for per-cell moduli with
displacements in the HexMesh's own dof numbering, so callers swap it in for
Stiffness + Solver.
"""
from __future__ import annotations

import os

import numpy as np
import scipy.sparse as sp

from .fem import CORNERS, hex8_ke, rigid_body_modes

__all__ = ["GridElasticity", "prolong", "restrict"]

# Corner k of a cell sits at (k & 1, k >> 1 & 1, k >> 2 & 1), as in fem.CORNERS.
_CORNER_OFF = [tuple(int(v) for v in c) for c in CORNERS]


def _child_prolongations() -> np.ndarray:
    """P_k (8, 24, 24): coarse-cell dofs -> dofs of child k (child k is the
    fine cell at offset CORNERS[k] inside the coarse cell)."""
    out = np.zeros((8, 24, 24))
    for k in range(8):
        for q in range(8):
            p = CORNERS[k] + CORNERS[q]          # fine node in fine-cell units, 0..2
            for r in range(8):
                w = 1.0
                for d in range(3):
                    t = p[d] / 2.0
                    w *= t if CORNERS[r][d] else 1.0 - t
                if w:
                    for i in range(3):
                        out[k, 3 * q + i, 3 * r + i] = w
    return out


_PK = _child_prolongations()

# Chebyshev needs an upper bound on the spectrum of D^-1 A. A dozen power
# iterations land ~10% under it, and an underestimate lets the top modes
# grow: on a uniform 500k-cell bar 1.1x took 16 CG iterations, the converged
# bound 6. Overestimating costs far less than underestimating.
LMAX_SAFETY = 1.3


def prolong(c: np.ndarray) -> np.ndarray:
    """Trilinear prolongation of a node-grid field (..., nx+1, ny+1, nz+1) to
    the grid with twice the cells (..., 2nx+1, 2ny+1, 2nz+1)."""
    out = c
    for ax in (-3, -2, -1):
        n = out.shape[ax]
        shape = list(out.shape)
        shape[ax] = 2 * n - 1
        f = np.empty(shape, out.dtype)
        sl_even = [slice(None)] * f.ndim
        sl_even[ax] = slice(0, None, 2)
        sl_odd = [slice(None)] * f.ndim
        sl_odd[ax] = slice(1, None, 2)
        lo = [slice(None)] * f.ndim
        lo[ax] = slice(0, n - 1)
        hi = [slice(None)] * f.ndim
        hi[ax] = slice(1, n)
        f[tuple(sl_even)] = out
        np.add(out[tuple(lo)], out[tuple(hi)], out=f[tuple(sl_odd)])
        f[tuple(sl_odd)] *= 0.5
        out = f
    return out


def restrict(f: np.ndarray) -> np.ndarray:
    """Transpose of `prolong`."""
    out = f
    for ax in (-3, -2, -1):
        n = (out.shape[ax] + 1) // 2
        sl_even = [slice(None)] * out.ndim
        sl_even[ax] = slice(0, None, 2)
        sl_odd = [slice(None)] * out.ndim
        sl_odd[ax] = slice(1, None, 2)
        c = out[tuple(sl_even)].copy()
        half = 0.5 * out[tuple(sl_odd)]
        lo = [slice(None)] * c.ndim
        lo[ax] = slice(0, n - 1)
        hi = [slice(None)] * c.ndim
        hi[ax] = slice(1, n)
        c[tuple(lo)] += half
        c[tuple(hi)] += half
        out = c
    return out


def _slab(cells_yz: int, dtype) -> int:
    """x-planes of cells per chunk: keep the (24, m, ny, nz) gather buffer
    near 32 MB."""
    per_plane = 24 * max(cells_yz, 1) * np.dtype(dtype).itemsize
    return max(1, int(32e6 // per_plane))


def _gather(u, r, out):
    x0, x1, y0, y1, z0, z1 = r
    for c, (dx, dy, dz) in enumerate(_CORNER_OFF):
        out[3 * c:3 * c + 3] = u[:, x0 + dx:x1 + dx, y0 + dy:y1 + dy, z0 + dz:z1 + dz]


def _scatter(f, r, out):
    x0, x1, y0, y1, z0, z1 = r
    for c, (dx, dy, dz) in enumerate(_CORNER_OFF):
        out[:, x0 + dx:x1 + dx, y0 + dy:y1 + dy, z0 + dz:z1 + dz] += f[3 * c:3 * c + 3]


# Tiles the operator is applied over: TILE_X x-planes by TILE_Y y-rows, cut
# to the z-range of cells that carry stiffness, and skipped when none do.
# A converged design fills a third of its box or less, and the box itself
# includes keep-outs, so most of the grid is never touched.
TILE_X, TILE_Y = 8, 16


def _tiles(active: np.ndarray) -> list:
    nx, ny, nz = active.shape
    out = []
    for x0 in range(0, nx, TILE_X):
        x1 = min(x0 + TILE_X, nx)
        for y0 in range(0, ny, TILE_Y):
            y1 = min(y0 + TILE_Y, ny)
            zs = np.flatnonzero(active[x0:x1, y0:y1].any(axis=(0, 1)))
            if len(zs):
                out.append((x0, x1, y0, y1, int(zs[0]), int(zs[-1]) + 1))
    return out


class _Level:
    """One grid level: cells (nx, ny, nz), node grid (3, nx+1, ny+1, nz+1).

    kind "scaled": cell matrix = E[cell] * ke (the fine level).
    kind "weights": cell matrix = sum_k W[k, cell] * M_k, plus an explicit
    patch for cells listed in patch_ijk (their W is zero).
    kind "explicit": cell matrix stored, K (24, 24, nx, ny, nz).
    `mask` (3, nodes...) is 1 on dofs that are free and carry stiffness;
    the operator is mask * K * mask + (1 - mask) (identity on the rest)."""

    def __init__(self, kind, dims, dtype):
        self.kind = kind
        self.dims = tuple(int(d) for d in dims)
        self.dtype = dtype
        self.ke = None
        self.e = None
        self.mcat = None
        self.w = None
        self.patch_ijk = None
        self.patch_k = None
        self.k = None
        self.extra = None            # sparse on the flattened (3, nodes) layout, or None
        self.mask = None
        self.dinv = None
        self.lmax = None
        self.tiles = None

    @property
    def node_shape(self):
        nx, ny, nz = self.dims
        return (3, nx + 1, ny + 1, nz + 1)

    def active(self) -> np.ndarray:
        """Cells with a nonzero matrix outside the patch."""
        if self.kind == "scaled":
            return self.e != 0
        if self.kind == "weights":
            return self.w.any(axis=0)
        return np.einsum("iixyz->xyz", self.k) > 0

    def _cell_kernel(self, buf, r):
        """f (24, cells of region r) = the cells' matrices times buf."""
        x0, x1, y0, y1, z0, z1 = r
        shp = (x1 - x0, y1 - y0, z1 - z0)
        flat = buf.reshape(24, -1)
        if self.kind == "scaled":
            f = (self.ke @ flat).reshape((24,) + shp)
            f *= self.e[x0:x1, y0:y1, z0:z1][None]
            return f
        if self.kind == "weights":
            g = (self.mcat @ flat).reshape((8, 24) + shp)
            w = self.w[:, x0:x1, y0:y1, z0:z1]
            f = g[0] * w[0][None]
            for k in range(1, 8):
                f += g[k] * w[k][None]
            return f
        kk = self.k[:, :, x0:x1, y0:y1, z0:z1].reshape(24, 24, -1)
        return np.einsum("ijn,jn->in", kk, flat).reshape((24,) + shp)

    def apply_k(self, u):
        """Unmasked K u on the node grid."""
        if self.tiles is None:
            self.tiles = _tiles(self.active())
        out = np.zeros(self.node_shape, u.dtype)
        for r in self.tiles:
            buf = np.empty((24, r[1] - r[0], r[3] - r[2], r[5] - r[4]), u.dtype)
            _gather(u, r, buf)
            _scatter(self._cell_kernel(buf, r), r, out)
        if self.patch_ijk is not None and len(self.patch_ijk):
            ue = self._patch_gather(u)
            fe = np.matmul(self.patch_k.astype(u.dtype, copy=False), ue[:, :, None])[:, :, 0]
            self._patch_scatter(fe, out)
        if self.extra is not None:
            out += (self.extra @ u.reshape(-1).astype(np.float64)).astype(u.dtype).reshape(out.shape)
        return out

    def _patch_index(self):
        if getattr(self, "_pidx", None) is None:
            nshape = self.node_shape[1:]
            idx = np.empty((len(self.patch_ijk), 24), np.int64)
            nn = int(np.prod(nshape))
            for c, off in enumerate(_CORNER_OFF):
                node = np.ravel_multi_index(tuple((self.patch_ijk + np.array(off)).T), nshape)
                for i in range(3):
                    idx[:, 3 * c + i] = i * nn + node
            self._pidx = idx
        return self._pidx

    def _patch_gather(self, u):
        return u.reshape(-1)[self._patch_index()]

    def _patch_scatter(self, fe, out):
        flat = out.reshape(-1)
        np.add.at(flat, self._patch_index().ravel(), fe.ravel())

    def apply(self, u):
        """The level's operator: identity on masked-out dofs."""
        m = self.mask.astype(u.dtype, copy=False)
        out = self.apply_k(u * m)
        out *= m
        out += (1 - m) * u
        return out

    def diagonal(self):
        """Diagonal of the unmasked K."""
        nx, ny, nz = self.dims
        d = np.zeros(self.node_shape)
        if self.kind == "scaled":
            dk = np.diag(self.ke)
            for c, (dx, dy, dz) in enumerate(_CORNER_OFF):
                d[:, dx:dx + nx, dy:dy + ny, dz:dz + nz] += dk[3 * c:3 * c + 3, None, None, None] * self.e[None]
        elif self.kind == "weights":
            dm = np.einsum("kii->ki", self.mcat.reshape(8, 24, 24).astype(np.float64))
            dc = (dm.T @ self.w.reshape(8, -1).astype(np.float64)).reshape((24,) + self.dims)
            for c, (dx, dy, dz) in enumerate(_CORNER_OFF):
                d[:, dx:dx + nx, dy:dy + ny, dz:dz + nz] += dc[3 * c:3 * c + 3]
        else:
            dc = np.einsum("iixyz->ixyz", self.k).astype(np.float64)
            for c, (dx, dy, dz) in enumerate(_CORNER_OFF):
                d[:, dx:dx + nx, dy:dy + ny, dz:dz + nz] += dc[3 * c:3 * c + 3]
        if self.patch_ijk is not None and len(self.patch_ijk):
            flat = d.reshape(-1)
            np.add.at(flat, self._patch_index().ravel(),
                      np.einsum("nii->ni", self.patch_k.astype(np.float64)).ravel())
        if self.extra is not None:
            d += self.extra.diagonal().reshape(d.shape)
        return d

    def cell_matrices(self, a, b):
        """Explicit matrices (24, 24, m, ny, nz) of x-slab [a, b), float64."""
        ny, nz = self.dims[1:]
        if self.kind == "explicit":
            return self.k[:, :, a:b].astype(np.float64)
        if self.kind == "scaled":
            return self.ke.astype(np.float64)[:, :, None, None, None] * self.e[a:b][None, None].astype(np.float64)
        mk = self.mcat.reshape(8, 576).astype(np.float64)
        w = self.w[:, a:b]
        out = (mk.T @ w.reshape(8, -1).astype(np.float64)).reshape((24, 24) + w.shape[1:])
        if self.patch_ijk is not None and len(self.patch_ijk):
            sel = (self.patch_ijk[:, 0] >= a) & (self.patch_ijk[:, 0] < b)
            p = self.patch_ijk[sel]
            out[:, :, p[:, 0] - a, p[:, 1], p[:, 2]] += np.moveaxis(self.patch_k[sel].astype(np.float64), 0, -1)
        return out


def _pad_even(a, axes_from=0, fill=0.0):
    pad = [(0, 0)] * a.ndim
    for i in range(3):
        n = a.shape[axes_from + i]
        pad[axes_from + i] = (0, n % 2)
    if any(p[1] for p in pad):
        return np.pad(a, pad, constant_values=fill)
    return a


def _prolong_rows(fine_nodes_ijk: np.ndarray, coarse_node_shape) -> sp.csr_matrix:
    """Scalar trilinear prolongation rows for the given fine nodes."""
    n = len(fine_nodes_ijk)
    rows, cols, vals = [], [], []
    for corner in range(8):
        w = np.ones(n)
        ci = np.empty((n, 3), np.int64)
        for d in range(3):
            i = fine_nodes_ijk[:, d]
            odd = i % 2 == 1
            hi = (corner >> d) & 1
            ci[:, d] = i // 2 + (hi & odd)
            w *= np.where(odd, 0.5, 1.0 if not hi else 0.0)
        ok = w > 0
        rows.append(np.nonzero(ok)[0])
        cols.append(np.ravel_multi_index(tuple(ci[ok].T), coarse_node_shape))
        vals.append(w[ok])
    nc = int(np.prod(coarse_node_shape))
    return sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n, nc))


class GridElasticity:
    """Matrix-free multigrid-preconditioned CG for K(E) u = f on a HexMesh.

    mesh: fem.HexMesh; nu: Poisson's ratio or fem.Layered; fixed: bool per
    mesh dof; springs: sparse (n_dof, n_dof) in mesh dof numbering, added to
    the operator unscaled, or None.

    `set_moduli(e)` takes per-element moduli (0 removes a cell outright; a
    node left with no stiffness is dropped from the solve and comes back as
    zero) and rebuilds the hierarchy; `solve_many(fs, x0s)` then solves.

    coarse_cells: stop coarsening geometrically once a level has fewer
    active cells than this, and assemble that level. degree: Chebyshev
    smoothing degree (pre and post). dtype: precision of the V-cycle.
    device: "auto", "cuda", "mps", "torch" or "cpu" (see gpu.py); None reads
    LOADPATH_DEVICE."""

    def __init__(self, mesh, nu, fixed, springs=None, rtol=1e-6, maxiter=500, coarse_cells=2000,
                 degree=2, dtype=np.float32, lu_max_dofs=60_000, device=None):
        self.mesh = mesh
        self.nu = nu
        self.rtol = rtol
        self.maxiter = maxiter
        self.coarse_cells = int(coarse_cells)
        self.degree = int(degree)
        self.dtype = np.dtype(dtype)
        self.lu_max_dofs = int(lu_max_dofs)
        cells = mesh.elem_ijk.max(axis=0) + 1
        cells = cells + cells % 2
        self.cells = tuple(int(c) for c in cells)
        self.node_shape = tuple(int(c) + 1 for c in cells)
        self.nn = int(np.prod(self.node_shape))
        self.node_flat = np.ravel_multi_index(tuple(mesh.node_ijk.T), self.node_shape)
        self.elem_ijk = mesh.elem_ijk
        self.ke = hex8_ke(nu) * mesh.pitch
        fixed = np.asarray(fixed, bool)
        fg = np.zeros((3, self.nn), bool)
        for i in range(3):
            fg[i, self.node_flat] = fixed[i::3]
        self.fixed_grid = fg.reshape((3,) + self.node_shape)
        self.springs = None
        if springs is not None:
            self.springs = self._to_grid_sparse(sp.csr_matrix(springs))
        self.levels = []
        self.last_iters = 0
        self._coarse = None
        self._lmax_vec = {}
        # torch copy of the hierarchy on a GPU (gpu.py), or None for numpy
        from .gpu import resolve_device

        dev = resolve_device(device)
        self._dev = None
        if dev is not None:
            from .gpu import TorchMG

            self._dev = TorchMG(self, dev)

    # --- layout conversion -------------------------------------------------

    def _to_grid_sparse(self, s):
        g = np.empty(s.shape[0], np.int64)
        for i in range(3):
            g[i::3] = i * self.nn + self.node_flat
        s = s.tocoo()
        return sp.csr_matrix((s.data, (g[s.row], g[s.col])), shape=(3 * self.nn, 3 * self.nn))

    def to_grid(self, v, dtype=np.float64):
        out = np.zeros((3, self.nn), dtype)
        for i in range(3):
            out[i, self.node_flat] = v[i::3]
        return out.reshape((3,) + self.node_shape)

    def from_grid(self, g):
        flat = g.reshape(3, -1)
        out = np.empty(3 * len(self.node_flat))
        for i in range(3):
            out[i::3] = flat[i, self.node_flat]
        return out

    def cell_grid(self, e, dtype=np.float64):
        g = np.zeros(self.cells, dtype)
        g[tuple(self.elem_ijk.T)] = e
        return g

    # --- hierarchy ---------------------------------------------------------

    def set_moduli(self, e):
        """Per-element moduli (mesh order). Rebuilds every level."""
        e = np.asarray(e, float)
        dt = self.dtype
        l0 = _Level("scaled", self.cells, np.float64)
        l0.ke = self.ke.astype(np.float64)
        l0.e = self.cell_grid(e)
        l0.extra = self.springs
        d0 = l0.diagonal()
        alive = d0 > 1e-300
        l0.mask = (alive & ~self.fixed_grid).astype(np.float64)
        l0.dinv = np.where(l0.mask > 0, 1.0 / np.where(alive, d0, 1.0), 1.0)
        self.alive = alive
        self.levels = [l0]
        # The V-cycle's copy of the fine level, in its precision.
        self._l0s = _Level("scaled", self.cells, dt)
        self._l0s.ke = self.ke.astype(dt)
        self._l0s.e = l0.e.astype(dt)
        self._l0s.extra = self.springs
        self._l0s.mask = l0.mask.astype(dt)
        self._l0s.dinv = l0.dinv.astype(dt)
        fine = l0
        n_active = int((e > 0).sum())
        depth = 0
        while n_active > self.coarse_cells and min(fine.dims) >= 2:
            coarse = self._coarsen(fine, first=(depth == 0))
            depth += 1
            d = coarse.diagonal()
            dmax = float(d.max()) if d.size else 0.0
            live = d > 1e-12 * max(dmax, 1e-300)
            coarse.mask = live.astype(dt)
            coarse.dinv = np.where(live, 1.0 / np.where(live, d, 1.0), 1.0).astype(dt)
            n_active = self._active_cells(coarse)
            self.levels.append(coarse)
            fine = coarse
        self._build_coarse(self.levels[-1])
        if self._dev is not None:
            self._dev.load()
            return
        for i in range(len(self.levels)):
            self._op(i).lmax = self._estimate_lmax(i)

    def _active_cells(self, lv):
        if lv.kind == "weights":
            act = lv.w.any(axis=0)
            if lv.patch_ijk is not None and len(lv.patch_ijk):
                act[tuple(lv.patch_ijk.T)] = True
            return int(act.sum())
        return int(np.abs(lv.k).max(axis=(0, 1)).astype(bool).sum())

    def _coarsen(self, fine, first):
        dt = self.dtype
        nx, ny, nz = fine.dims
        half = (nx // 2, ny // 2, nz // 2)
        # pad to even so the next level can halve it again (padding cells
        # carry no stiffness, their nodes are masked out)
        cdims = tuple(h + h % 2 if h > 1 else h for h in half)
        mask_fine = fine.mask if first else None
        if first:
            # children: E of fine cell (2I + CORNERS[k]); cells touching a held
            # dof go to the patch, with their masked matrices
            held = (1 - fine.mask) * self.alive   # free=0 & alive => held
            held_node = held.any(axis=0)
            touch = np.zeros(fine.dims, bool)
            for (dx, dy, dz) in _CORNER_OFF:
                touch |= held_node[dx:dx + nx, dy:dy + ny, dz:dz + nz]
            touch &= fine.e > 0
            w = np.zeros((8,) + cdims, np.float64)
            hx, hy, hz = half
            for k, (a, b, c) in enumerate(_CORNER_OFF):
                w[k, :hx, :hy, :hz] = fine.e[a::2, b::2, c::2]
            ptouch = np.zeros(cdims, bool)
            for k, (a, b, c) in enumerate(_CORNER_OFF):
                ptouch[:hx, :hy, :hz] |= touch[a::2, b::2, c::2]
            lv = _Level("weights", cdims, dt)
            mk = np.einsum("kai,ab,kbj->kij", _PK, fine.ke, _PK)
            lv.mcat = mk.reshape(8 * 24, 24).astype(dt)
            pij = np.argwhere(ptouch)
            pk = np.zeros((len(pij), 24, 24))
            fm = fine.mask
            for k, (a, b, c) in enumerate(_CORNER_OFF):
                fi = 2 * pij + np.array([a, b, c])
                ec = fine.e[tuple(fi.T)]
                dm = np.empty((len(pij), 24))
                for q, (qx, qy, qz) in enumerate(_CORNER_OFF):
                    dm[:, 3 * q:3 * q + 3] = fm[:, fi[:, 0] + qx, fi[:, 1] + qy, fi[:, 2] + qz].T
                kc = ec[:, None, None] * fine.ke[None] * dm[:, :, None] * dm[:, None, :]
                pk += np.matmul(np.matmul(_PK[k].T, kc), _PK[k])
            w[:, ptouch] = 0.0
            lv.w = w.astype(dt)
            lv.patch_ijk = pij
            lv.patch_k = pk.astype(dt)
        else:
            lv = _Level("explicit", cdims, dt)
            k = np.zeros((24, 24) + cdims, dt)
            step = max(1, _slab(ny * nz * 24, np.float64) // 2) * 2
            for a in range(0, nx, step):
                b = min(a + step, nx)
                kf = fine.cell_matrices(a, b)            # (24, 24, m, ny, nz)
                for ch, (ox, oy, oz) in enumerate(_CORNER_OFF):
                    sub = kf[:, :, ox::2, oy::2, oz::2]
                    shp = sub.shape[2:]
                    # P^T K P as two GEMMs
                    t = (_PK[ch].T @ sub.reshape(24, -1)).reshape(24, 24, -1)       # (i, b, n)
                    t = np.tensordot(t, _PK[ch], axes=([1], [0]))                    # (i, n, j)
                    k[:, :, a // 2:a // 2 + shp[0], :shp[1], :shp[2]] += \
                        np.moveaxis(t, 2, 1).reshape((24, 24) + shp).astype(dt)
            lv.k = k
        lv.extra = self._coarse_extra(fine, cdims, mask_fine)
        return lv

    def _coarse_extra(self, fine, cdims, mask_fine):
        s = fine.extra
        if s is None or s.nnz == 0:
            return None
        nshape_f = fine.node_shape[1:]
        nshape_c = tuple(c + 1 for c in cdims)
        nn_f = int(np.prod(nshape_f))
        nn_c = int(np.prod(nshape_c))
        s = s.tocsr()
        dofs = np.unique(np.concatenate([s.tocoo().row, s.tocoo().col]))
        comp, node = np.divmod(dofs, nn_f)
        ijk = np.stack(np.unravel_index(node, nshape_f), axis=1)
        pr = _prolong_rows(ijk, nshape_c).tocoo()
        cols = comp[pr.row] * nn_c + pr.col
        vals = pr.data
        if mask_fine is not None:
            vals = vals * mask_fine.reshape(-1)[dofs[pr.row]]
        p = sp.csr_matrix((vals, (pr.row, cols)), shape=(len(dofs), 3 * nn_c))
        ss = s[dofs][:, dofs]
        return (p.T @ ss @ p).tocsr()

    def _build_coarse(self, lv):
        """Assemble the coarsest level and factor it (or build pyamg on it)."""
        from .fem import HexMesh, Stiffness

        nx, ny, nz = lv.dims
        if lv is self.levels[0]:
            self._coarse = None
            return
        kall = lv.cell_matrices(0, nx)                               # (24, 24, nx, ny, nz)
        act = np.abs(kall).max(axis=(0, 1)) > 0
        ijk = np.argwhere(act)
        if len(ijk) == 0:
            self._coarse = None
            return
        kes = np.moveaxis(kall[:, :, act], -1, 0)                    # (n, 24, 24)
        hm = HexMesh(ijk, 1.0, np.zeros(3))
        st = Stiffness(hm, 0.3, np.zeros(hm.n_dof, bool))
        data = np.zeros(st.nnz)
        st.add_elements(data, kes, np.arange(len(ijk)))
        k = sp.csr_matrix((data, st.indices, st.indptr), shape=(hm.n_dof, hm.n_dof))
        nshape = lv.node_shape[1:]
        nn = int(np.prod(nshape))
        gnode = np.ravel_multi_index(tuple(hm.node_ijk.T), nshape)
        gdof = np.empty(hm.n_dof, np.int64)
        for i in range(3):
            gdof[i::3] = i * nn + gnode
        if lv.extra is not None:
            k = (k + lv.extra[gdof][:, gdof]).tocsr()
        m = lv.mask.reshape(-1)[gdof].astype(np.float64)
        dm = sp.diags(m)
        k = (dm @ k @ dm + sp.diags(1.0 - m)).tocsr()
        k = 0.5 * (k + k.T)
        self._coarse = {"gdof": gdof, "n": hm.n_dof}
        if hm.n_dof <= self.lu_max_dofs:
            from scipy.sparse.linalg import splu

            # SPD: symmetric mode without pivoting. Partial pivoting on these
            # high-contrast matrices quadrupled the fill (6 s against 0.5 s
            # for 7k dofs).
            self._coarse["lu"] = splu(k.tocsc(), permc_spec="MMD_AT_PLUS_A", diag_pivot_thresh=0.0,
                                      options={"SymmetricMode": True})
        else:
            from .fem import amg_solver

            ml = amg_solver(k, B=rigid_body_modes(hm.node_ijk.astype(float)), symmetry="hermitian",
                            strength="symmetric", max_coarse=400, coarse_solver="splu")
            self._coarse["ml"] = ml

    def _coarse_solve(self, b):
        c = self._coarse
        flat = b.reshape(-1)
        rhs = flat[c["gdof"]].astype(np.float64)
        if "lu" in c:
            x = c["lu"].solve(rhs)
        else:
            x = c["ml"].solve(rhs, x0=np.zeros_like(rhs), tol=1e-30, maxiter=1, cycle="V")
        out = np.zeros_like(flat)
        out[c["gdof"]] = x
        return out.reshape(b.shape)

    def _op(self, i):
        return self._l0s if i == 0 else self.levels[i]

    def _estimate_lmax(self, i, its=12):
        """Largest eigenvalue of D^-1 A at level i, by power iteration (a
        Chebyshev smoother needs an upper bound; see LMAX_SAFETY)."""
        lv = self._op(i)
        prev = self._lmax_vec.get(i)
        if prev is not None and prev.shape == lv.node_shape:
            # warm start from the last hierarchy's vector. Not fewer
            # iterations: cutting them to 4 underestimated the bound once
            # cells dropped out of the design, and the CG count tripled.
            v = prev.astype(lv.dinv.dtype) * lv.mask
        else:
            rng = np.random.default_rng(i)
            v = (rng.random(lv.node_shape) - 0.5).astype(lv.dinv.dtype) * lv.mask
        lam = 1.0
        for _ in range(its):
            nv = np.linalg.norm(v)
            if nv == 0:
                return 2.0
            v /= nv
            av = lv.dinv * lv.apply(v) * lv.mask
            lam = float(np.vdot(v, av))
            v = av
        self._lmax_vec[i] = v
        return LMAX_SAFETY * max(lam, 1e-12)

    def _smooth(self, lv, b, x):
        """Chebyshev iteration on D^-1 A over [lmax/30, lmax]."""
        hi = lv.lmax
        lo = hi / 30.0
        theta = 0.5 * (hi + lo)
        delta = 0.5 * (hi - lo)
        sigma = theta / delta
        rho = 1.0 / sigma
        r = b - lv.apply(x) if x is not None else b.copy()
        if x is None:
            x = np.zeros_like(b)
        d = lv.dinv * r / theta
        for it in range(self.degree):
            x += d
            if it + 1 == self.degree:
                break
            r -= lv.apply(d)
            rho_new = 1.0 / (2.0 * sigma - rho)
            d = rho_new * rho * d + (2.0 * rho_new / delta) * (lv.dinv * r)
            rho = rho_new
        return x

    def _vcycle(self, i, b):
        lv = self._op(i)
        if i == len(self.levels) - 1:
            if self._coarse is not None:
                return self._coarse_solve(b)
            x = self._smooth(lv, b, None)
            return self._smooth(lv, b, x)
        x = self._smooth(lv, b, None)
        r = b - lv.apply(x)
        cs = self._op(i + 1).node_shape
        rf = np.zeros((3,) + tuple(2 * n - 1 for n in cs[1:]), r.dtype)
        sx, sy, sz = lv.node_shape[1:]
        rf[:, :sx, :sy, :sz] = r * lv.mask
        xc = self._vcycle(i + 1, restrict(rf).astype(self.dtype, copy=False))
        x += prolong(xc)[:, :sx, :sy, :sz] * lv.mask
        return self._smooth(lv, b, x)

    def precondition(self, r):
        z = self._vcycle(0, r.astype(self.dtype))
        return z.astype(np.float64)

    # --- solve -------------------------------------------------------------

    def apply(self, u):
        """The float64 operator on a node-grid field."""
        return self.levels[0].apply(u)

    def _pcg(self, b, x0):
        a = self.levels[0]
        m = a.mask
        x = np.zeros(a.node_shape) if x0 is None else x0.copy()
        bnorm = float(np.linalg.norm(b))
        if bnorm == 0:
            return np.zeros_like(b), 0, True
        r = b.copy()
        if x0 is not None:
            # Scale the warm start to its energy-optimal multiple, and drop
            # it if even that is worse than zero: after a projection step or
            # a cut of the live cells the last displacement can start 8x
            # further off than zero does (35 CG iterations against 15).
            ax = a.apply(x)
            xax = float(np.vdot(x, ax))
            s = float(np.vdot(b, x)) / xax if xax > 0 else 0.0
            if s > 0:
                rs = b - s * ax
                if np.linalg.norm(rs) < bnorm:
                    x *= s
                    r = rs
                else:
                    x[:] = 0
            else:
                x[:] = 0
        z = self.precondition(r)
        p = z.copy()
        rz = float(np.vdot(r, z))
        tol = self.rtol * bnorm
        for it in range(1, self.maxiter + 1):
            ap = a.apply(p)
            pap = float(np.vdot(p, ap))
            if pap <= 0:
                return x, it, False
            alpha = rz / pap
            x += alpha * p
            r -= alpha * ap
            if float(np.linalg.norm(r)) <= tol:
                return x, it, True
            z = self.precondition(r)
            rz_new = float(np.vdot(r, z))
            p *= rz_new / rz
            p += z
            rz = rz_new
        del m
        return x, self.maxiter, False

    def solve_many(self, fs, x0s=None):
        """Solve for each right-hand side (mesh dof numbering). Rows of dofs
        that carry no stiffness are returned as zero."""
        x0s = x0s or [None] * len(fs)
        out = []
        iters = 0
        for f, x0 in zip(fs, x0s):
            b = self.to_grid(f)
            b *= self.levels[0].mask
            g0 = self.to_grid(x0) * self.levels[0].mask if x0 is not None else None
            x, it, ok = self._dev.solve(b, g0) if self._dev is not None else self._pcg(b, g0)
            iters = max(iters, it)
            if not ok:
                raise RuntimeError(f"linear solve did not converge ({it} iterations); "
                                   "the structure may be a mechanism (check supports)")
            out.append(self.from_grid(x * self.levels[0].mask))
        self.last_iters = iters
        return out

    def live_dofs(self):
        """Bool per mesh dof: carries stiffness in the current moduli."""
        a = self.alive.reshape(3, -1)
        out = np.empty(3 * len(self.node_flat), bool)
        for i in range(3):
            out[i::3] = a[i, self.node_flat]
        return out

    def strain_energy(self, u):
        """Per element u_e^T Ke u_e at unit modulus (mesh order)."""
        g = self.to_grid(u)
        nx, ny, nz = self.cells
        se = np.zeros(self.cells)
        step = _slab(ny * nz, np.float64)
        ke = self.ke
        for a in range(0, nx, step):
            b = min(a + step, nx)
            buf = np.empty((24, b - a, ny, nz))
            _gather(g, (a, b, 0, ny, 0, nz), buf)
            flat = buf.reshape(24, -1)
            se[a:b] = np.einsum("in,in->n", flat, ke @ flat).reshape(b - a, ny, nz)
        return se[tuple(self.elem_ijk.T)]

