"""
Cut cells: the voxel analysis with the part's true boundary inside each cell.

verify.py keeps a cell when its centre is inside the part and gives it the
whole cube's stiffness. Member thickness then rounds to whole cells, and a
member two or three cells thick reads its bending stress tens of percent
off, either way (benchmarks/beams.py: 0.75-1.28x CalculiX at 2 cells,
30-45 degrees to the grid). This module keeps the same grid, nodes and
solvers and changes two things:

- Stiffness: every cell the part touches (fill >= MIN_FILL) is kept, and a
  cell the boundary cuts gets its stiffness integrated over the part of it
  that is solid: an n x n x n split of the cell, each solid sub-cell
  integrated exactly (2x2x2 Gauss), the finite cell method. The element is
  fem.py's incompatible-mode hex8; on a cut cell the bubbles' strain is made
  mean-free over the solid part (Taylor's correction on the material
  domain), so the constant-strain patch test still holds. Full cells keep
  the shared unit-cube matrix, so the assembled path adds the cut cells'
  matrices as explicit elements and the matrix-free multigrid takes them as
  an extra sparse operator (mg.GridElasticity's `springs`, which its
  Galerkin coarsening carries down exactly).
- Stress: a cut cell's corners can sit a cell outside the part, where its
  field means nothing. Nodes near the boundary read the stress ON the true
  surface instead: a linear least-squares fit (superconvergent patch
  recovery) over the element fields at the solid sample points around the
  node's closest surface point. Bending stress is linear through a member,
  so the fit takes it to the surface whole.

The boundary comes from a voxel grid n times finer (`fine`, bool), which
gives the sub-cells, the fill, and (its boundary faces) the surface points.

Accuracy (benchmarks/beams.py, peak / CalculiX): 0.95-1.09 in bending
and 0.95-1.17 in tension from 2 cells thick up, at any angle, except two
cells at 45 degrees (0.86): two cells across a diagonal strip are too few
for a trilinear cell to bend in, which verify's thin-member submodels
resolve (0.97-1.06 at 2.5x finer). The whole-cell staircase reads
0.75-1.95 over the same cases.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from .fem import CORNERS, HexMesh, Stiffness, _hex8_parts, elastic_c

__all__ = ["CutGrid", "SubcellRule", "MIN_FILL", "SUB"]

SUB = 4             # sub-cells per cell edge
MIN_FILL = 0.02     # a cell with less of the part than this is left out (conditioning)
_GAUSS = 1.0 / np.sqrt(3.0)
# how far from the surface (cells) a node reads the surface fit instead of
# its corner average, and the fit's neighbourhood
NEAR_SURFACE = 1.0
FIT_POINTS = 48
FIT_TAPER = 1.5


def _strain_rows(dn):
    """(n, k, 3) gradients of k scalar fields -> (n, 6, 3k) strain rows."""
    n, k, _ = dn.shape
    b = np.zeros((n, 6, 3 * k))
    dx, dy, dz = dn[..., 0], dn[..., 1], dn[..., 2]
    b[:, 0, 0::3], b[:, 1, 1::3], b[:, 2, 2::3] = dx, dy, dz
    b[:, 3, 0::3], b[:, 3, 1::3] = dy, dx
    b[:, 4, 1::3], b[:, 4, 2::3] = dz, dy
    b[:, 5, 0::3], b[:, 5, 2::3] = dz, dx
    return b


def b_nodes(xi):
    """Trilinear strain rows at natural points xi (n, 3): (n, 6, 24), unit cube."""
    xi = np.asarray(xi, float)
    sg = 2.0 * CORNERS - 1.0
    f = 1.0 + xi[:, None, :] * sg[None]
    dn = np.empty((len(xi), 8, 3))
    dn[..., 0] = 0.25 * sg[None, :, 0] * f[..., 1] * f[..., 2]
    dn[..., 1] = 0.25 * sg[None, :, 1] * f[..., 0] * f[..., 2]
    dn[..., 2] = 0.25 * sg[None, :, 2] * f[..., 0] * f[..., 1]
    return _strain_rows(dn)


def b_bubbles(xi):
    """Strain rows of the nine bubbles (1 - xi_m^2) at natural points: (n, 6, 9)."""
    xi = np.asarray(xi, float)
    dn = np.zeros((len(xi), 3, 3))
    for m in range(3):
        dn[:, m, m] = -4.0 * xi[:, m]
    return _strain_rows(dn)


class SubcellRule:
    """Integrals over each sub-cell of an n^3 split of the unit cube of the
    pieces of the element matrices, so a cut cell's matrices are its
    sub-cell indicator times these."""

    def __init__(self, nu, n=SUB):
        self.n = n
        c = elastic_c(1.0, nu)
        self.c = c
        ns = n ** 3
        ii = (2 * np.arange(n) + 1) / n - 1.0
        self.centers = np.stack(np.meshgrid(ii, ii, ii, indexing="ij"), -1).reshape(-1, 3)
        g = np.array([[a, b, cc] for a in (-_GAUSS, _GAUSS) for b in (-_GAUSS, _GAUSS) for cc in (-_GAUSS, _GAUSS)])
        pts = (self.centers[:, None, :] + g[None] / n).reshape(-1, 3)           # (ns*8, 3)
        bu = b_nodes(pts).reshape(ns, 8, 6, 24)
        ba = b_bubbles(pts).reshape(ns, 8, 6, 9)
        w = 0.125 / ns                                   # Gauss weight (det J = 1/8 on the unit cube)
        cb = np.einsum("ij,sgjk->sgik", c, bu)
        ca = np.einsum("ij,sgjk->sgik", c, ba)
        self.S_uu = w * np.einsum("sgia,sgib->sab", bu, cb)
        self.S_ua = w * np.einsum("sgia,sgib->sab", bu, ca)
        self.S_aa = w * np.einsum("sgia,sgib->sab", ba, ca)
        self.P_u = w * bu.sum(1)
        self.P_a = w * ba.sum(1)
        self.vol = np.full(ns, 1.0 / ns)

    def matrices(self, ind):
        """(ke (m, 24, 24), r (m, 9, 24), mbar (m, 6, 9)) at unit modulus and
        unit side for cells with sub-cell indicators ind (m, n^3): r maps the
        nodal displacements to the condensed bubble amplitudes, mbar is the
        mean bubble strain taken off on that cell."""
        ind = np.asarray(ind, float)
        c = self.c
        kuu = ind @ self.S_uu.reshape(len(self.vol), -1)
        kuu = kuu.reshape(-1, 24, 24)
        v = ind @ self.vol
        pa = (ind @ self.P_a.reshape(len(self.vol), -1)).reshape(-1, 6, 9)
        pu = (ind @ self.P_u.reshape(len(self.vol), -1)).reshape(-1, 6, 24)
        mbar = pa / v[:, None, None]
        kua = (ind @ self.S_ua.reshape(len(self.vol), -1)).reshape(-1, 24, 9) \
            - np.einsum("eki,kl,elj->eij", pu, c, mbar)
        kaa = ((ind @ self.S_aa.reshape(len(self.vol), -1)).reshape(-1, 9, 9)
               - np.einsum("eki,kl,elj->eij", pa, c, mbar)
               - np.einsum("eki,kl,elj->eij", mbar, c, pa)
               + v[:, None, None] * np.einsum("eki,kl,elj->eij", mbar, c, mbar))
        kaa = 0.5 * (kaa + kaa.transpose(0, 2, 1))
        r = -np.linalg.solve(kaa, kua.transpose(0, 2, 1))           # (m, 9, 24)
        ke = kuu + kua @ r
        return 0.5 * (ke + ke.transpose(0, 2, 1)), r, mbar


class CutGrid:
    """The cut-cell model of a part given as a bool grid `fine` at pitch / n
    whose (0, 0, 0) corner is `origin` (each cell of the analysis grid is an
    n^3 block of it). `cells` picks the analysis cells (fill >= MIN_FILL by
    default). Attributes: mesh (fem.HexMesh), fill (per cell), cut (bool per
    cell), ke/r/mbar for the cut cells."""

    def __init__(self, fine, n, origin, pitch, nu, min_fill=MIN_FILL):
        fine = np.asarray(fine, bool)
        dims = np.array(fine.shape) // n
        fine = fine[:dims[0] * n, :dims[1] * n, :dims[2] * n]
        self.n, self.pitch, self.nu = n, float(pitch), nu
        self.origin = np.asarray(origin, float)
        blocks = fine.reshape(dims[0], n, dims[1], n, dims[2], n).transpose(0, 2, 4, 1, 3, 5)
        blocks = blocks.reshape(dims[0], dims[1], dims[2], n ** 3)
        fill = blocks.mean(axis=3)
        keep = fill >= min_fill
        ijk = np.argwhere(keep)
        self.mesh = HexMesh(ijk, self.pitch, self.origin)
        self.fill = fill[keep]
        self.ind = blocks[keep]
        self.cut = self.fill < 1.0 - 1e-12
        self.rule = SubcellRule(nu, n)
        cutidx = np.nonzero(self.cut)[0]
        self.cut_index = cutidx
        ke = np.empty((len(cutidx), 24, 24))
        r = np.empty((len(cutidx), 9, 24))
        mb = np.empty((len(cutidx), 6, 9))
        for s in range(0, len(cutidx), 4000):
            sl = slice(s, s + 4000)
            ke[sl], r[sl], mb[sl] = self.rule.matrices(self.ind[cutidx[sl]])
        self.ke_cut, self.r_cut, self.mbar_cut = ke * self.pitch, r, mb
        self.ke_full, self.r_full = _hex8_parts(nu, "im")
        self.ke_full = self.ke_full * self.pitch
        # the true surface, as the fine grid's boundary faces (to within
        # half a fine cell, pitch / 2n), for the surface stress points
        hf = self.pitch / n
        pad = np.pad(fine, 1)
        pts = []
        for ax in range(3):
            d = np.diff(pad.astype(np.int8), axis=ax)
            idx = np.argwhere(d != 0).astype(float)
            # diff index i along ax is the face between fine cells i-1 and i,
            # at i; across it the padded index j is fine cell j-1, centre j-0.5
            idx[:, [a for a in range(3) if a != ax]] -= 0.5
            pts.append(idx)
        self._faces = self.origin + np.concatenate(pts) * hf
        self._fine = fine
        self._fine_h = hf
        self._tree = None

    def restrict(self, keep):
        """The same model on the elements `keep` (bool per element), e.g.
        without pieces no support holds."""
        keep = np.asarray(keep, bool)
        if keep.all():
            return self
        new = object.__new__(CutGrid)
        new.__dict__.update(self.__dict__)
        new.mesh = HexMesh(self.mesh.elem_ijk[keep], self.pitch, self.origin)
        new.fill, new.ind, new.cut = self.fill[keep], self.ind[keep], self.cut[keep]
        pos = np.full(self.mesh.n_elem, -1, np.int64)
        pos[self.cut_index] = np.arange(len(self.cut_index))
        old = np.nonzero(keep)[0]
        k = pos[old[new.cut]]
        new.cut_index = np.nonzero(new.cut)[0]
        new.ke_cut, new.r_cut, new.mbar_cut = self.ke_cut[k], self.r_cut[k], self.mbar_cut[k]
        return new

    def solid_grid(self):
        """(bool grid of the kept cells, origin) as verify.voxelize returns."""
        dims = np.array(self._fine.shape) // self.n
        g = np.zeros(dims, bool)
        g[tuple(self.mesh.elem_ijk.T)] = True
        return g, self.origin

    # --- stiffness ------------------------------------------------------------

    def full_moduli(self, modulus):
        """Per-element moduli for the shared-matrix path: cut cells 0 (their
        matrices go in separately)."""
        e = np.broadcast_to(np.asarray(modulus, float), (self.mesh.n_elem,)).copy()
        e[self.cut] = 0.0
        return e

    def cut_matrix(self, stiff, modulus):
        """The cut cells' stiffness as a CSR matrix in stiff's pattern
        (fixed rows and columns cleared, zero on their diagonal)."""
        mod = np.broadcast_to(np.asarray(modulus, float), (self.mesh.n_elem,))[self.cut_index]
        data = np.zeros(stiff.nnz)
        for s in range(0, len(self.cut_index), 20000):
            sl = slice(s, s + 20000)
            stiff.add_elements(data, self.ke_cut[sl] * mod[sl, None, None], self.cut_index[sl])
        return stiff.finish(data, diag=0.0)

    def assemble(self, stiff: Stiffness, modulus):
        """The assembled K of the cut model (stiff: fem.Stiffness on self.mesh)."""
        return (stiff.assemble(self.full_moduli(modulus)) + self.cut_matrix(stiff, modulus)).tocsr()

    def cut_operator(self, modulus):
        """The cut cells' stiffness as an unconstrained sparse matrix in mesh
        dof numbering (mg.GridElasticity's `springs`)."""
        m = self.mesh
        mod = np.broadcast_to(np.asarray(modulus, float), (m.n_elem,))[self.cut_index]
        edof = m.edof[self.cut_index]
        rows = np.repeat(edof, 24, axis=1).ravel()
        cols = np.tile(edof, (1, 24)).ravel()
        vals = (self.ke_cut * mod[:, None, None]).ravel()
        return sp.csr_matrix((vals, (rows, cols)), shape=(m.n_dof, m.n_dof))

    # --- stress ---------------------------------------------------------------

    def _samples(self, near, u, modulus, thermal=None):
        """Stress at the solid sample points of the elements in `near` (bool
        per element): a full cell's eight Gauss points, a cut cell's solid
        sub-cell centres. Returns (xyz, weight, stress)."""
        m = self.mesh
        c = elastic_c(1.0, self.nu)
        mod = np.broadcast_to(np.asarray(modulus, float), (m.n_elem,))
        th = None if thermal is None else np.asarray(thermal, float)
        pos = np.full(m.n_elem, -1, np.int64)
        pos[self.cut_index] = np.arange(len(self.cut_index))
        g = np.array([[a, b, cc] for a in (-_GAUSS, _GAUSS) for b in (-_GAUSS, _GAUSS) for cc in (-_GAUSS, _GAUSS)])
        xyz, w, sig = [], [], []

        def put(el, xi_pt, weight):
            ue = u[m.edof[el]]
            alpha = ue @ self.r_full.T
            k = pos[el]
            isc = k >= 0
            if isc.any():
                alpha[isc] = np.einsum("pij,pj->pi", self.r_cut[k[isc]], ue[isc])
            bu, ba = b_nodes(xi_pt[None])[0], b_bubbles(xi_pt[None])[0]
            eps = ue @ bu.T + alpha @ ba.T
            if isc.any():
                eps[isc] -= np.einsum("pij,pj->pi", self.mbar_cut[k[isc]], alpha[isc])
            eps /= self.pitch
            if th is not None:
                eps[:, :3] -= th[el][:, None]
            sig.append((eps @ c.T) * mod[el][:, None])
            xyz.append(m.origin + (m.elem_ijk[el] + 0.5 * (xi_pt + 1.0)) * self.pitch)
            w.append(np.full(len(el), weight))

        full = np.nonzero(near & ~self.cut)[0]
        if len(full):
            for q in g:
                put(full, q, 1.0 / 8)
        cut = np.nonzero(near & self.cut)[0]
        if len(cut):
            ind = self.ind[cut]
            for sidx, q in enumerate(self.rule.centers):
                el = cut[ind[:, sidx]]
                if len(el):
                    put(el, q, 1.0 / self.n ** 3)
        if not xyz:
            return np.zeros((0, 3)), np.zeros(0), np.zeros((0, 6))
        return np.concatenate(xyz), np.concatenate(w), np.concatenate(sig)

    def surface_points(self, xyz, k=8):
        """Points on the true surface nearest each of xyz (the mean of the k
        nearest boundary faces of the fine grid, which smooths its
        staircase) and the signed distance to them (mm, negative inside)."""
        from scipy.spatial import cKDTree

        if self._tree is None:
            self._tree = cKDTree(self._faces)
        xyz = np.asarray(xyz, float)
        dist, idx = self._tree.query(xyz, k=k)
        p = self._faces[idx].mean(axis=1)
        ijk = np.floor((xyz - self.origin) / self._fine_h).astype(np.int64)
        ok = np.all((ijk >= 0) & (ijk < np.array(self._fine.shape)), axis=1)
        inside = np.zeros(len(xyz), bool)
        inside[ok] = self._fine[tuple(ijk[ok].T)]
        d = np.linalg.norm(xyz - p, axis=1)
        return p, np.where(inside, -d, d)

    def nodal_stress(self, u, modulus, thermal=None):
        """Stress per node (n_nodes, 6). Nodes within NEAR_SURFACE cells of
        the true surface read the surface fit at their closest surface
        point; the rest the fill-weighted average of their elements' corner
        stresses (verify.nodal_stress's recovery)."""
        from scipy.spatial import cKDTree

        m = self.mesh
        c = elastic_c(1.0, self.nu)
        mod = np.broadcast_to(np.asarray(modulus, float), (m.n_elem,))
        ue = u[m.edof]                                              # (e, 24)
        alpha = ue @ self.r_full.T                                  # (e, 9) bubble amplitudes
        if len(self.cut_index):
            alpha[self.cut_index] = np.einsum("pij,pj->pi", self.r_cut, ue[self.cut_index])
        acc = np.zeros((m.n_nodes, 6))
        # cells whose stress is the printed material's (a bought member's cells are not: stock.py); a node
        # reads the cells of the part round it, never a member's
        mask = getattr(self, "stress_mask", None)
        fill = self.fill if mask is None else self.fill * mask
        wsum = np.bincount(m.enode.ravel(), weights=np.repeat(fill, 8), minlength=m.n_nodes)
        corners = 2.0 * CORNERS - 1.0
        bu, ba = b_nodes(corners), b_bubbles(corners)
        for k in range(8):
            eps = ue @ bu[k].T + alpha @ ba[k].T
            if len(self.cut_index):
                eps[self.cut_index] -= np.einsum("pij,pj->pi", self.mbar_cut, alpha[self.cut_index])
            eps /= self.pitch
            if thermal is not None:
                eps[:, :3] -= np.asarray(thermal, float)[:, None]
            sk = (eps @ c.T) * (mod * fill)[:, None]
            for j in range(6):
                acc[:, j] += np.bincount(m.enode[:, k], weights=sk[:, j], minlength=m.n_nodes)
        out = acc / np.maximum(wsum, 1e-30)[:, None]
        # nodes on cut cells, or touching the grid's empty neighbours
        on_cut = np.zeros(m.n_nodes, bool)
        on_cut[m.enode[self.cut].ravel()] = True
        cnt = np.bincount(m.enode.ravel(), minlength=m.n_nodes)
        on_cut |= cnt < 8
        if mask is not None:
            on_cut &= wsum > 0
        cand = np.nonzero(on_cut)[0]
        if not len(cand):
            return out
        p, d = self.surface_points(m.node_xyz[cand])
        sel = d >= -NEAR_SURFACE * self.pitch     # every node outside, and those just inside
        cand, p = cand[sel], p[sel]
        if not len(cand):
            return out
        # samples from elements within two cells of a surface node
        near_node = np.zeros(m.n_nodes, bool)
        near_node[cand] = True
        near_el = near_node[m.enode].any(axis=1)
        for _ in range(2):
            nn = np.zeros(m.n_nodes, bool)
            nn[m.enode[near_el].ravel()] = True
            near_el = nn[m.enode].any(axis=1)
        if mask is not None:
            near_el &= mask
        xyz, w, sig = self._samples(near_el, u, modulus, thermal)
        if not len(xyz):
            return out
        tree = cKDTree(xyz)
        k = min(FIT_POINTS, len(xyz))
        dist, idx = tree.query(p, k=k)
        out[cand] = _fit_linear(p, xyz, sig, w, dist, idx, self.pitch)
        return out


def _fit_linear(p, xyz, sig, w, dist, idx, h):
    """Weighted least-squares linear field over each point's neighbours,
    evaluated at the point: (p, 6)."""
    out = np.empty((len(p), 6))
    for s in range(0, len(p), 20000):
        sl = slice(s, s + 20000)
        d = (xyz[idx[sl]] - p[sl, None, :]) / h                 # (q, k, 3)
        # Gaussian-ish taper so the fit is local, and the sample's volume
        ww = w[idx[sl]] * np.exp(-(dist[sl] / (FIT_TAPER * h)) ** 2)
        a = np.concatenate([np.ones(d.shape[:2] + (1,)), d], axis=2)  # (q, k, 4)
        aw = a * ww[..., None]
        ata = np.einsum("qki,qkj->qij", aw, a) + 1e-9 * np.eye(4)
        atb = np.einsum("qki,qkc->qic", aw, sig[idx[sl]])
        coef = np.linalg.solve(ata, atb)
        out[sl] = coef[:, 0, :]
    return out


def scan_voxelize(mesh, pitch, origin, dims, chunk=200_000):
    """Solid cells (bool, dims) of a closed mesh on the grid at `origin`,
    cell centres inside by crossing parity along +z, as verify.voxelize
    decides them, but found by rasterizing each triangle onto the cell
    columns it covers instead of casting a ray per column: the sub-cell grid
    has 16x the columns, and per-ray queries took 20 s of a 100 s verify."""
    tri = np.asarray(mesh.triangles, float)
    dims = np.asarray(dims, int)
    jit = pitch * 1e-4
    # column centres, with verify.voxelize's jitter off exact edges
    ox = origin[0] + 0.5 * pitch + 0.37 * jit
    oy = origin[1] + 0.5 * pitch + 0.61 * jit
    i0 = np.clip(np.ceil((tri[:, :, 0].min(1) - ox) / pitch), 0, dims[0]).astype(np.int64)
    i1 = np.clip(np.floor((tri[:, :, 0].max(1) - ox) / pitch) + 1, 0, dims[0]).astype(np.int64)
    j0 = np.clip(np.ceil((tri[:, :, 1].min(1) - oy) / pitch), 0, dims[1]).astype(np.int64)
    j1 = np.clip(np.floor((tri[:, :, 1].max(1) - oy) / pitch) + 1, 0, dims[1]).astype(np.int64)
    ni, nj = np.maximum(i1 - i0, 0), np.maximum(j1 - j0, 0)
    cnt = ni * nj
    cols, zs, sg = [], [], []
    t_all = np.nonzero(cnt)[0]
    start = 0
    while start < len(t_all):
        # triangles whose candidate columns fit in one chunk
        cs = np.cumsum(cnt[t_all[start:]])
        stop = start + max(1, int(np.searchsorted(cs, chunk, side="right")))
        t = t_all[start:stop]
        start = stop
        c = cnt[t]
        tt = np.repeat(t, c)
        k = np.arange(c.sum()) - np.repeat(np.cumsum(c) - c, c)
        ii = i0[tt] + k // nj[tt]
        jj = j0[tt] + k % nj[tt]
        px, py = ox + ii * pitch, oy + jj * pitch
        a, b, cc = tri[tt, 0], tri[tt, 1], tri[tt, 2]
        # 2D barycentrics of the column in the triangle's xy projection
        d = (b[:, 0] - a[:, 0]) * (cc[:, 1] - a[:, 1]) - (cc[:, 0] - a[:, 0]) * (b[:, 1] - a[:, 1])
        ok = np.abs(d) > 1e-18
        dd = np.where(ok, d, 1.0)
        l1 = ((px - a[:, 0]) * (cc[:, 1] - a[:, 1]) - (cc[:, 0] - a[:, 0]) * (py - a[:, 1])) / dd
        l2 = ((b[:, 0] - a[:, 0]) * (py - a[:, 1]) - (px - a[:, 0]) * (b[:, 1] - a[:, 1])) / dd
        l0 = 1.0 - l1 - l2
        hit = ok & (l0 >= 0) & (l1 >= 0) & (l2 >= 0)
        z = l0 * a[:, 2] + l1 * b[:, 2] + l2 * cc[:, 2]
        cols.append((ii * dims[1] + jj)[hit])
        zs.append(z[hit])
        sg.append(np.sign(d)[hit])
    solid = np.zeros(dims, bool)
    if not cols:
        return solid
    col = np.concatenate(cols)
    z = np.concatenate(zs)
    sgn = np.concatenate(sg)
    order = np.lexsort((sgn, z, col))
    col, z, sgn = col[order], z[order], sgn[order]
    # merge near-duplicate hits in a column (a shared edge counted twice),
    # facing the same way only: two faces meeting back to back (a sheet of
    # no thickness, as a boolean leaves where a surface grazes a cut plane)
    # are two crossings, and merging them turned the rest of the column over
    dup = np.zeros(len(z), bool)
    dup[1:] = (col[1:] == col[:-1]) & (np.diff(z) <= 1e-7) & (sgn[1:] == sgn[:-1])
    col, z = col[~dup], z[~dup]
    first = np.r_[True, col[1:] != col[:-1]]
    run = np.cumsum(first) - 1
    starts = np.nonzero(first)[0]
    rank = np.arange(len(col)) - starts[run]
    n_in = np.bincount(run)
    # an odd count drops its last hit (verify.voxelize does the same)
    keep = rank < (n_in[run] // 2) * 2
    col, z, rank = col[keep], z[keep], rank[keep]
    za, zb, cl = z[rank % 2 == 0], z[rank % 2 == 1], col[rank % 2 == 0]
    zc0 = origin[2] + 0.5 * pitch
    ka = np.clip(np.floor((za - zc0) / pitch) + 1, 0, dims[2]).astype(np.int64)   # first centre > za
    kb = np.clip(np.ceil((zb - zc0) / pitch), 0, dims[2]).astype(np.int64)        # first centre >= zb
    diff = np.zeros((dims[0] * dims[1], dims[2] + 1), np.int32)
    np.add.at(diff, (cl, ka), 1)
    np.add.at(diff, (cl, kb), -1)
    return (np.cumsum(diff, axis=1)[:, :-1] > 0).reshape(dims)


def cut_from_mesh(mesh, pitch, nu, box=None, n=SUB, min_fill=MIN_FILL):
    """CutGrid of a closed trimesh on verify.voxelize's grid at `pitch` (the
    same origin and cells), its sub-cells from voxelizing at pitch / n."""
    lo, hi = (np.asarray(b, float) for b in box) if box is not None else np.asarray(mesh.bounds, float)
    dims = np.maximum(np.ceil((hi - lo) / pitch - 1e-6).astype(int), 1)
    origin = lo - 0.5 * (dims * pitch - (hi - lo))
    fine = scan_voxelize(mesh, pitch / n, origin, dims * n)
    return CutGrid(fine, n, origin, pitch, nu, min_fill=min_fill)
