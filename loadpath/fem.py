"""
Linear elastic finite elements on a voxel grid: 8-node hexahedra, all the
same cube, so one element stiffness matrix serves every element and a
density (or a material modulus) just scales it.

Two things make this fast enough to sit inside an optimization loop that
solves it a hundred times:

- Assembly never sorts. The grid is structured, so every stiffness entry
  lands in a CSR slot we can compute up front from (node, neighbor offset).
  Re-assembling for new densities is 64 vectorized scatter-adds.
- The solve is conjugate gradient preconditioned by smoothed-aggregation
  algebraic multigrid (pyamg) with the six rigid-body modes as the near
  null space. The hierarchy is expensive to build and densities change
  slowly, so callers reuse it across iterations (`Solver`).

Units are whatever the caller uses consistently; loadpath uses mm, N, MPa.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import cg

__all__ = ["Layered", "amg_solver", "elastic_c", "hex8_ke", "hex8_b_center", "HexMesh", "Stiffness", "Solver", "von_mises"]

# Local corner order: bit 0 = +x, bit 1 = +y, bit 2 = +z.
CORNERS = np.array([[(c >> 0) & 1, (c >> 1) & 1, (c >> 2) & 1] for c in range(8)])
# The 27 neighbor offsets, lexicographic in (dx, dy, dz). With nodes numbered
# in C order over (i, j, k), that is also ascending node index, so CSR rows
# come out with sorted column indices.
OFFSETS = np.array([[dx, dy, dz] for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)])


def _offset_index(d):
    return (d[..., 0] + 1) * 9 + (d[..., 1] + 1) * 3 + (d[..., 2] + 1)


_VOIGT = ((0, 0), (1, 1), (2, 2), (0, 1), (1, 2), (2, 0))


class Layered:
    """Transversely isotropic elasticity for a part built in layers: the
    same in every direction within a layer, softer across layers. Stands in
    for a Poisson's ratio wherever one is passed (Stiffness, hex8_ke,
    elastic_c), so a layered material threads through the solver unchanged.

    nu: in-plane Poisson's ratio; ez: modulus across layers / in-plane
    modulus; gz: shear modulus between layers / in-plane E/(2(1+nu));
    normal: the build direction (unit vector, the axis layers stack along).
    The Poisson's ratio of in-plane stress onto the layer normal is taken as
    nu, so nu_zp = nu * ez, which keeps C positive definite for ez in (0, 1]
    and nu < 0.5."""

    def __init__(self, nu: float, ez: float = 1.0, gz: float = 1.0, normal=(0.0, 0.0, 1.0)):
        n = np.asarray(normal, float)
        if not (0.0 < ez <= 1.0 and 0.0 < gz <= 1.0):
            raise ValueError("Layered: ez and gz are fractions in (0, 1]")
        self.nu = float(nu)
        self.ez = float(ez)
        self.gz = float(gz)
        self.normal = tuple(n / np.linalg.norm(n))

    def __repr__(self):
        return f"Layered(nu={self.nu}, ez={self.ez}, gz={self.gz}, normal={self.normal})"

    def __float__(self):
        # code that only wants a Poisson's ratio (tet cross-check decks) gets one
        return self.nu

    def c(self) -> np.ndarray:
        """6x6 at unit in-plane modulus, in the solver's frame."""
        nu, ez = self.nu, self.ez
        g = 1.0 / (2.0 * (1.0 + nu))
        s = np.zeros((6, 6))
        s[0, 0] = s[1, 1] = 1.0
        s[0, 1] = s[1, 0] = -nu
        s[0, 2] = s[2, 0] = s[1, 2] = s[2, 1] = -nu
        s[2, 2] = 1.0 / ez
        s[3, 3] = 1.0 / g
        s[4, 4] = s[5, 5] = 1.0 / (self.gz * g)
        local = np.linalg.inv(s)
        # rotate from the frame whose z is the layer normal
        z = np.asarray(self.normal)
        a = np.array([1.0, 0.0, 0.0]) if abs(z[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        x = np.cross(a, z)
        x /= np.linalg.norm(x)
        rot = np.stack([x, np.cross(z, x), z], axis=1)     # columns: local axes in solver frame
        return _rotate_c(local, rot)


def _to_tensor(c):
    t = np.zeros((3, 3, 3, 3))
    for a, (i, j) in enumerate(_VOIGT):
        for b, (k, l) in enumerate(_VOIGT):
            for (p, q) in {(i, j), (j, i)}:
                for (r, s_) in {(k, l), (l, k)}:
                    t[p, q, r, s_] = c[a, b]
    return t


def _rotate_c(c, rot):
    t = np.einsum("pi,qj,rk,sl,ijkl->pqrs", rot, rot, rot, rot, _to_tensor(c))
    out = np.empty((6, 6))
    for a, (i, j) in enumerate(_VOIGT):
        for b, (k, l) in enumerate(_VOIGT):
            out[a, b] = t[i, j, k, l]
    return out


def elastic_c(e: float, nu) -> np.ndarray:
    """Elasticity matrix (6x6, engineering shear strains): isotropic for a
    Poisson's ratio, layered for a `Layered`."""
    if isinstance(nu, Layered):
        return e * nu.c()
    lam = e * nu / ((1 + nu) * (1 - 2 * nu))
    mu = e / (2 * (1 + nu))
    c = np.zeros((6, 6))
    c[:3, :3] = lam
    c[np.arange(3), np.arange(3)] = lam + 2 * mu
    c[np.arange(3, 6), np.arange(3, 6)] = mu
    return c


def _b_matrix(xi, eta, zeta):
    """Strain-displacement matrix (6x24) of the unit cube at a natural point
    (xi, eta, zeta in [-1, 1]). For side h, divide by h."""
    s = 2.0 * CORNERS - 1.0
    dn = np.empty((8, 3))
    dn[:, 0] = 0.25 * s[:, 0] * (1 + eta * s[:, 1]) * (1 + zeta * s[:, 2])
    dn[:, 1] = 0.25 * s[:, 1] * (1 + xi * s[:, 0]) * (1 + zeta * s[:, 2])
    dn[:, 2] = 0.25 * s[:, 2] * (1 + xi * s[:, 0]) * (1 + eta * s[:, 1])
    b = np.zeros((6, 24))
    for i in range(8):
        dx, dy, dz = dn[i]
        col = 3 * i
        b[0, col] = dx
        b[1, col + 1] = dy
        b[2, col + 2] = dz
        b[3, col], b[3, col + 1] = dy, dx
        b[4, col + 1], b[4, col + 2] = dz, dy
        b[5, col], b[5, col + 2] = dz, dx
    return b


def hex8_ke(nu) -> np.ndarray:
    """Stiffness (24x24) of a unit cube with E = 1, 2x2x2 Gauss. For side h
    and modulus E the element matrix is E * h * this."""
    c = elastic_c(1.0, nu)
    g = 1.0 / np.sqrt(3.0)
    ke = np.zeros((24, 24))
    for xi in (-g, g):
        for eta in (-g, g):
            for zeta in (-g, g):
                b = _b_matrix(xi, eta, zeta)
                ke += b.T @ c @ b * 0.125  # det J of the unit cube map = 1/8
    return 0.5 * (ke + ke.T)


def hex8_b_center() -> np.ndarray:
    return _b_matrix(0.0, 0.0, 0.0)


def von_mises(stress: np.ndarray) -> np.ndarray:
    sx, sy, sz, txy, tyz, tzx = stress.T
    return np.sqrt(0.5 * ((sx - sy) ** 2 + (sy - sz) ** 2 + (sz - sx) ** 2)
                   + 3.0 * (txy ** 2 + tyz ** 2 + tzx ** 2))


class HexMesh:
    """The active elements of a voxel grid and the nodes they use.

    elem_ijk: (ne, 3) integer cell indices; pitch: cell size; origin: world
    position of node (0, 0, 0), i.e. the min corner of cell (0, 0, 0)."""

    def __init__(self, elem_ijk: np.ndarray, pitch: float, origin: np.ndarray):
        self.elem_ijk = np.asarray(elem_ijk, np.int64)
        self.pitch = float(pitch)
        self.origin = np.asarray(origin, float)
        ne = len(self.elem_ijk)
        if ne == 0:
            raise ValueError("no active elements")
        dims = self.elem_ijk.max(axis=0) + 2
        self.node_dims = dims
        # Node ids in ascending grid order (what np.unique over the corner
        # indices gave), by marking the node grid instead of sorting 8 ids
        # per cell: linear time and memory, which matters from ~1M cells.
        e_flat = (self.elem_ijk[:, 0] * dims[1] + self.elem_ijk[:, 1]) * dims[2] + self.elem_ijk[:, 2]
        offs = (CORNERS[:, 0] * dims[1] + CORNERS[:, 1]) * dims[2] + CORNERS[:, 2]
        mark = np.zeros(int(np.prod(dims)), bool)
        for o in offs:
            mark[e_flat + o] = True
        used = np.flatnonzero(mark)
        del mark
        self._lookup = np.full(int(np.prod(dims)), -1, np.int64)
        self._lookup[used] = np.arange(len(used))
        self.enode = self._lookup[e_flat[:, None] + offs[None, :]]
        self.node_ijk = np.stack(np.unravel_index(used, tuple(dims)), axis=1).astype(np.int64)
        self.n_nodes = len(used)
        self.node_xyz = self.origin + self.node_ijk * self.pitch
        self.n_dof = 3 * self.n_nodes
        self._edof = None

    @property
    def n_elem(self):
        return len(self.elem_ijk)

    @property
    def edof(self):
        # built on first use: 192 bytes per cell, and the multigrid path
        # never needs it
        if self._edof is None:
            self._edof = (3 * self.enode[:, :, None] + np.arange(3)).reshape(self.n_elem, 24)
        return self._edof

    def elem_centers(self):
        return self.origin + (self.elem_ijk + 0.5) * self.pitch

    def node_id(self, ijk: np.ndarray) -> np.ndarray:
        """Node id at integer node coords, -1 if absent or out of range."""
        ijk = np.asarray(ijk, np.int64)
        d = self.node_dims
        ok = np.all((ijk >= 0) & (ijk < d), axis=-1)
        flat = (ijk[..., 0] * d[1] + ijk[..., 1]) * d[2] + ijk[..., 2]
        out = np.full(ijk.shape[:-1], -1, np.int64)
        out[ok] = self._lookup[flat[ok]]
        return out


class Stiffness:
    """Precomputed CSR layout for a HexMesh; `assemble(scale)` returns the
    global stiffness for per-element moduli `scale` (E_e, already including
    any density interpolation), with Dirichlet conditions applied as
    identity rows so the matrix keeps one fixed shape."""

    def __init__(self, mesh: HexMesh, nu, fixed: np.ndarray):
        self.mesh = mesh
        self.ke = hex8_ke(nu) * mesh.pitch        # E = 1, side h
        self.nu = nu
        nn = mesh.n_nodes
        enode = mesh.enode
        nijk = mesh.node_ijk

        # Which (node, offset) pairs actually share an element.
        adj = np.zeros((nn, 27), bool)
        pair_off = np.empty((8, 8), np.int64)
        for a in range(8):
            for b in range(8):
                pair_off[a, b] = _offset_index(CORNERS[b] - CORNERS[a])
                adj[enode[:, a], pair_off[a, b]] = True
        cnt = adj.sum(axis=1)
        rank = np.cumsum(adj, axis=1) - 1
        row_len = np.repeat(3 * cnt, 3)
        self.indptr = np.concatenate([[0], np.cumsum(row_len)]).astype(np.int64)
        nnz = int(self.indptr[-1])
        indices = np.empty(nnz, np.int64)
        n_idx, o_idx = np.nonzero(adj)                       # sorted by node, then offset
        nbr = mesh.node_id(nijk[n_idx] + OFFSETS[o_idx])
        for i in range(3):
            base = self.indptr[3 * n_idx + i] + 3 * rank[n_idx, o_idx]
            for j in range(3):
                indices[base + j] = 3 * nbr + j
        self.indices = indices
        self.nnz = nnz

        # Per corner pair: the slot of entry (3a+i, 3b+j) is base + i*row + j.
        # int32 slots when they fit: these 64 arrays are the bulk of this
        # object's memory (9 slots per element per corner pair).
        idx_t = np.int32 if nnz < 2 ** 31 - 1 else np.int64
        self._pairs = []
        for a in range(8):
            na = enode[:, a]
            row = 3 * cnt[na]
            for b in range(8):
                base = self.indptr[3 * na] + 3 * rank[na, pair_off[a, b]]
                blk = self.ke[3 * a:3 * a + 3, 3 * b:3 * b + 3].ravel()
                slots = (base[:, None] + np.repeat(np.arange(3), 3)[None, :] * row[:, None]
                         + np.tile(np.arange(3), 3)[None, :])
                self._pairs.append((slots.astype(idx_t), blk))
        self.indptr = self.indptr.astype(idx_t)
        self.indices = self.indices.astype(idx_t)

        fixed = np.asarray(fixed, bool)
        if fixed.shape != (mesh.n_dof,):
            raise ValueError("fixed must be a bool per dof")
        self.fixed = fixed
        row_of_slot = np.repeat(np.arange(mesh.n_dof), row_len)
        self._keep = ~(fixed[row_of_slot] | fixed[indices])
        diag = np.nonzero((row_of_slot == indices) & fixed[row_of_slot])[0]
        self._diag_fixed = diag
        self.free = ~fixed

    def assemble(self, scale: np.ndarray) -> sp.csr_matrix:
        data = np.zeros(self.nnz)
        s = np.asarray(scale, float)[:, None]
        for slots, blk in self._pairs:
            data[slots] += s * blk
        data *= self._keep
        data[self._diag_fixed] = 1.0
        k = sp.csr_matrix((data, self.indices, self.indptr), shape=(self.mesh.n_dof, self.mesh.n_dof))
        k.has_sorted_indices = True
        return k

    def add_elements(self, data: np.ndarray, kes: np.ndarray, elems: np.ndarray) -> None:
        """Scatter per-element 24x24 matrices (a tangent that differs element
        to element, not a scaled copy of one) for elements `elems` into a CSR
        data array of length nnz. Finish with `finish(data)`."""
        for idx, (slots, _) in enumerate(self._pairs):
            a, b = divmod(idx, 8)
            data[slots[elems]] += kes[:, 3 * a:3 * a + 3, 3 * b:3 * b + 3].reshape(len(elems), 9)

    def finish(self, data: np.ndarray, diag: float = 1.0) -> sp.csr_matrix:
        """CSR matrix from scattered data, Dirichlet rows and columns cleared
        and `diag` on their diagonal (1 for a stiffness, 0 for a geometric
        stiffness in an eigenproblem)."""
        data = data * self._keep
        data[self._diag_fixed] = diag
        k = sp.csr_matrix((data, self.indices, self.indptr), shape=(self.mesh.n_dof, self.mesh.n_dof))
        k.has_sorted_indices = True
        return k

    def strain_energy(self, u: np.ndarray) -> np.ndarray:
        """Per element u_e^T Ke u_e at unit modulus (multiply by E_e)."""
        ue = u[self.mesh.edof]
        return np.einsum("ij,jk,ik->i", ue, self.ke, ue)

    def stress(self, u: np.ndarray, modulus: np.ndarray | float) -> np.ndarray:
        """Element-center stress (ne, 6) for per-element modulus."""
        b = hex8_b_center() / self.mesh.pitch
        c = elastic_c(1.0, self.nu)
        strain = u[self.mesh.edof] @ b.T
        return (strain @ c.T) * np.asarray(modulus, float).reshape(-1, 1)


def rigid_body_modes(xyz: np.ndarray) -> np.ndarray:
    n = len(xyz)
    b = np.zeros((3 * n, 6))
    x, y, z = (xyz - xyz.mean(axis=0)).T
    for i in range(3):
        b[i::3, i] = 1.0
    b[0::3, 3], b[1::3, 3] = -y, x
    b[1::3, 4], b[2::3, 4] = -z, y
    b[0::3, 5], b[2::3, 5] = z, -x
    return b


def amg_solver(a, **kw):
    """pyamg.smoothed_aggregation_solver, reproducibly. pyamg estimates
    spectral radii (the prolongation smoother's weight) from np.random's
    global stream, so the hierarchy, and with it every iterate, changed from
    run to run: the printed hook's overhang read 0.040 or 0.042 on the same
    code. Seed the stream for the setup and give the caller's back."""
    import pyamg

    state = np.random.get_state()
    np.random.seed(0)
    try:
        return pyamg.smoothed_aggregation_solver(a, **kw)
    finally:
        np.random.set_state(state)


class Solver:
    """CG + AMG with a reusable hierarchy. `solve(K, f, x0)` rebuilds the
    multigrid hierarchy when asked to, or when the last solve needed too many
    iterations (the old hierarchy no longer matches K well)."""

    # Refresh (rather than rebuild) the hierarchy for a new matrix at most
    # this many times in a row.
    REFRESH_AGE = 8

    def __init__(self, mesh: HexMesh, rtol=1e-6, maxiter=800, rebuild_after_iters=25, refresh=True):
        self.rtol = rtol
        self.refresh = refresh
        self.max_age = self.REFRESH_AGE
        self.maxiter = maxiter
        self.rebuild_after_iters = rebuild_after_iters
        self._b = rigid_body_modes(mesh.node_xyz)
        self._ml = None
        self._age = 0
        self.last_iters = 0
        self.use_amg = True
        try:
            import pyamg  # noqa: F401
        except ImportError:  # pragma: no cover - pyamg is a declared dependency
            self.use_amg = False

    def invalidate(self):
        self._ml = None

    def _precond(self, k):
        if not self.use_amg:
            d = k.diagonal()
            d[d == 0] = 1.0
            return sp.diags(1.0 / d)
        if self._ml is not None and self._ml.levels[0].A is not k:
            if self.refresh and self._age < self.max_age:
                # A new matrix on the same mesh (the optimizer's next design):
                # keep the hierarchy but smooth with the matrix being solved.
                # Smoothing with the previous design's matrix (as this did
                # until 2026-10-30) took 2-5x the CG iterations of a fresh
                # hierarchy, so every other step paid for a rebuild; this
                # cut an arm-link step from 3.1 s to 2.3 s. (Recomputing the
                # coarse operators too, R A P, was slower: the old
                # prolongators span void and solid alike, and on a design
                # that had just moved they converged worse than these.)
                self._ml.levels[0].A = k
                self._age += 1
            else:
                self._ml = None
        if self._ml is None:
            # coarse_solver: pyamg's default is a dense pseudo-inverse (an SVD)
            # of the coarsest level. Aggregation on these stiffness matrices
            # often stops coarsening at a few thousand unknowns, where that SVD
            # alone took ~3s per rebuild; a sparse LU of the same level is
            # milliseconds.
            self._ml = amg_solver(k, B=self._b, symmetry="hermitian", strength="symmetric", max_coarse=400,
                                  coarse_solver="splu")
            self._age = 0
        return self._ml.aspreconditioner(cycle="V")

    def _maxiter(self):
        """A refreshed hierarchy that stops working (the design moved too far
        from the one it was aggregated on) gives up early and is rebuilt,
        rather than running to maxiter."""
        if self._age == 0:
            return self.maxiter
        return min(self.maxiter, 3 * self.rebuild_after_iters)

    def solve(self, k, f, x0=None):
        count = [0]

        def cb(_):
            count[0] += 1

        m = self._precond(k)
        u, info = cg(k, f, x0=x0, rtol=self.rtol, maxiter=self._maxiter(), M=m, callback=cb)
        if info != 0 and self._ml is not None:
            # Stale hierarchy: rebuild once and retry from where we got to.
            self._ml = None
            count[0] = 0
            u, info = cg(k, f, x0=u, rtol=self.rtol, maxiter=self.maxiter, M=self._precond(k), callback=cb)
        self.last_iters = count[0]
        if count[0] > self.rebuild_after_iters:
            self._ml = None
        if info != 0:
            raise RuntimeError(f"linear solve did not converge ({count[0]} iterations); "
                               "the structure may be a mechanism (check supports)")
        return u

    def solve_many(self, k, fs, x0s=None):
        """Solve several right-hand sides (load cases) against one matrix.
        The preconditioner is built once and the solves run on threads: the
        sparse kernels release the GIL for most of their time, which measured
        1.4x for two cases."""
        x0s = x0s or [None] * len(fs)
        if len(fs) == 1:
            return [self.solve(k, fs[0], x0s[0])]
        from concurrent.futures import ThreadPoolExecutor

        m = self._precond(k)
        maxiter = self._maxiter()
        counts = [0] * len(fs)

        def one(i):
            def cb(_):
                counts[i] += 1
            u, info = cg(k, fs[i], x0=x0s[i], rtol=self.rtol, maxiter=maxiter, M=m, callback=cb)
            return u, info

        with ThreadPoolExecutor(min(len(fs), 4)) as ex:
            res = list(ex.map(one, range(len(fs))))
        out = []
        for i, (u, info) in enumerate(res):
            if info != 0:
                # Fall back to the single-solve path, which rebuilds a stale
                # hierarchy and retries.
                self._ml = None
                u = self.solve(k, fs[i], u)
            out.append(u)
        self.last_iters = max(counts)
        if self.last_iters > self.rebuild_after_iters:
            self._ml = None
        return out
