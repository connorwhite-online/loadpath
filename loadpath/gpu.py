"""
GPU backend for the matrix-free multigrid solve (mg.GridElasticity).

Everything the large-grid solver spends its time on is dense array work on
the voxel node grid: eight slice gathers, one GEMM with a shared 24x24
matrix, eight slice scatters, Chebyshev smoothing, trilinear transfers.
That maps one to one onto torch, which runs it on Apple GPUs (MPS) and
NVIDIA GPUs (CUDA) from the same code. Hierarchy setup stays in numpy (about
1% of the time) and the tensors are uploaded once per `set_moduli`.

Precision follows the CPU solver: the V-cycle in float32, the outer
conjugate gradient and its operator in float64, so the answer is the same
float64 one (same CG counts, compliance equal to ~1e-9). A float32 outer CG
was tried for Apple GPUs, which have no float64: on the arm link it
diverged after ~40 iterations where float64 needs 12 (a slender part's
K u cancels away float32's digits), and reliable updates could not rescue
it. So on MPS the CG runs on the CPU (torch, threaded; unified memory keeps
the copies cheap) and the V-cycle, the bulk of the work, on the GPU.

The coarsest level stays where it is (a sparse LU or pyamg on the CPU):
it is a few thousand dofs, and copying its right-hand side over is cheaper
than a device factorization.

Selection: LOADPATH_DEVICE=auto (default: CUDA, then MPS, then torch on the
CPU, which on its own beats the numpy kernels several times over), cuda,
mps, torch (torch on the CPU), or cpu (the numpy solver). torch is an
optional dependency (extra "gpu"); without it everything runs as before.
"""
from __future__ import annotations

import os

import numpy as np

from .mg import _CORNER_OFF, LMAX_SAFETY

__all__ = ["resolve_device", "TorchMG"]

# Bytes of gather buffer per chunk of x-planes (like mg._slab, larger: the
# device has the bandwidth and launch overhead dominates small chunks).
_CHUNK_BYTES = 256e6


def resolve_device(name: str | None = None):
    """The torch device to run the multigrid on, or None for the numpy path."""
    name = (name if name is not None else os.environ.get("LOADPATH_DEVICE", "auto")).strip().lower()
    if name in ("", "cpu", "none", "off", "numpy"):
        return None
    try:
        import torch
    except ImportError:
        if name in ("auto",):
            return None
        raise RuntimeError(f"LOADPATH_DEVICE={name} needs torch (pip install 'loadpath[gpu]')")
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if name == "torch":
        return torch.device("cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("LOADPATH_DEVICE=cuda but torch sees no CUDA device")
    if name == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("LOADPATH_DEVICE=mps but torch sees no Metal device")
    return torch.device(name)


def _prolong(c):
    """mg.prolong on a tensor (..., n0, n1, n2) -> (..., 2n0-1, 2n1-1, 2n2-1)."""
    import torch

    out = c
    for ax in (-3, -2, -1):
        n = out.shape[ax]
        shape = list(out.shape)
        shape[ax] = 2 * n - 1
        f = torch.empty(shape, dtype=out.dtype, device=out.device)
        sl_even = [slice(None)] * f.ndim
        sl_even[ax] = slice(0, None, 2)
        sl_odd = [slice(None)] * f.ndim
        sl_odd[ax] = slice(1, None, 2)
        f[tuple(sl_even)] = out
        f[tuple(sl_odd)] = 0.5 * (out.narrow(ax, 0, n - 1) + out.narrow(ax, 1, n - 1))
        out = f
    return out


def _restrict(f):
    """Transpose of _prolong."""
    out = f
    for ax in (-3, -2, -1):
        n = (out.shape[ax] + 1) // 2
        sl_even = [slice(None)] * out.ndim
        sl_even[ax] = slice(0, None, 2)
        sl_odd = [slice(None)] * out.ndim
        sl_odd[ax] = slice(1, None, 2)
        c = out[tuple(sl_even)].clone()
        half = 0.5 * out[tuple(sl_odd)]
        c.narrow(ax, 0, n - 1).add_(half)
        c.narrow(ax, 1, n - 1).add_(half)
        out = c
    return out


class _DevLevel:
    """A level's operator on a torch device in `dtype`, built from an mg._Level."""

    def __init__(self, lv, dev, dtype):
        import torch

        t = lambda a: torch.as_tensor(np.ascontiguousarray(a), dtype=dtype, device=dev)
        self.kind = lv.kind
        self.dims = lv.dims
        self.node_shape = lv.node_shape
        self.mask = t(lv.mask)
        self.dinv = t(lv.dinv)
        self.lmax = None
        act = lv.active()
        self.box = None
        if act.any():
            lo = [int(np.flatnonzero(act.any(axis=tuple(j for j in range(3) if j != d)))[0]) for d in range(3)]
            hi = [int(np.flatnonzero(act.any(axis=tuple(j for j in range(3) if j != d)))[-1]) + 1 for d in range(3)]
            self.box = (lo[0], hi[0], lo[1], hi[1], lo[2], hi[2])
        if lv.kind == "scaled":
            self.ke = t(lv.ke)
            self.e = t(lv.e)
        elif lv.kind == "weights":
            self.mcat = t(lv.mcat)
            self.w = t(lv.w)
        else:
            # (x, y, z, 24, 24): one batched matvec per cell
            self.k = t(np.moveaxis(lv.k, (0, 1), (-2, -1)))
        self.pidx = self.pk = None
        if lv.patch_ijk is not None and len(lv.patch_ijk):
            self.pidx = torch.as_tensor(lv._patch_index(), device=dev)
            self.pk = t(lv.patch_k)
        self.extra = None
        if lv.extra is not None and lv.extra.nnz:
            s = lv.extra.tocoo()
            self.extra = (torch.as_tensor(s.row.astype(np.int64), device=dev),
                          torch.as_tensor(s.col.astype(np.int64), device=dev), t(s.data))
        cyz = (self.box[3] - self.box[2]) * (self.box[5] - self.box[4]) if self.box else 1
        per_plane = 24 * cyz * 4 * (8 if lv.kind == "weights" else 2)
        self.chunk = max(1, int(_CHUNK_BYTES // per_plane))

    def _kernel(self, flat, r):
        x0, x1, y0, y1, z0, z1 = r
        shp = (x1 - x0, y1 - y0, z1 - z0)
        if self.kind == "scaled":
            f = (self.ke @ flat).reshape((24,) + shp)
            return f.mul_(self.e[x0:x1, y0:y1, z0:z1])
        if self.kind == "weights":
            g = (self.mcat @ flat).reshape((8, 24) + shp)
            w = self.w[:, x0:x1, y0:y1, z0:z1]
            return (g * w[:, None]).sum(0)
        kk = self.k[x0:x1, y0:y1, z0:z1].reshape(-1, 24, 24)
        f = (kk @ flat.T.unsqueeze(-1)).squeeze(-1)           # (n, 24)
        return f.T.reshape((24,) + shp)

    def apply_k(self, u):
        import torch

        out = torch.zeros(self.node_shape, dtype=u.dtype, device=u.device)
        if self.box is not None:
            bx0, bx1, y0, y1, z0, z1 = self.box
            for x0 in range(bx0, bx1, self.chunk):
                x1 = min(x0 + self.chunk, bx1)
                r = (x0, x1, y0, y1, z0, z1)
                buf = torch.cat([u[:, x0 + dx:x1 + dx, y0 + dy:y1 + dy, z0 + dz:z1 + dz]
                                 for (dx, dy, dz) in _CORNER_OFF], 0)
                f = self._kernel(buf.reshape(24, -1), r)
                for c, (dx, dy, dz) in enumerate(_CORNER_OFF):
                    out[:, x0 + dx:x1 + dx, y0 + dy:y1 + dy, z0 + dz:z1 + dz] += f[3 * c:3 * c + 3]
        flat_out = out.view(-1)
        if self.pidx is not None:
            ue = u.reshape(-1)[self.pidx]                                  # (n, 24)
            fe = (self.pk @ ue.unsqueeze(-1)).squeeze(-1)
            flat_out.index_add_(0, self.pidx.reshape(-1), fe.reshape(-1))
        if self.extra is not None:
            rows, cols, vals = self.extra
            flat_out.index_add_(0, rows, vals * u.reshape(-1)[cols])
        return out

    def apply(self, u):
        m = self.mask
        out = self.apply_k(u * m)
        out.mul_(m)
        out.add_((1 - m) * u)
        return out


class TorchMG:
    """Device copy of a GridElasticity hierarchy, and the mixed-precision
    solve on it. `load()` after every set_moduli; `solve(b, x0)` on
    float64 numpy node grids returns (x, device iterations, converged)."""

    def __init__(self, grid, device):
        import torch

        self.torch = torch
        self.g = grid
        self.dev = device
        self.dtype = torch.float32
        self.levels = []
        self._lmax_vec = {}
        # The outer CG and its operator need float64: in float32 a slender
        # part's K u cancels away its digits (displacements are hundreds of
        # times a cell's stretch) and CG diverged after ~40 iterations on
        # the arm link where float64 takes 12. Apple GPUs have no float64,
        # so there the CG runs on the CPU (torch, unified memory: the copies
        # are cheap) and only the V-cycle, ~85% of the work, on the GPU.
        self.cg_dev = torch.device("cpu") if device.type == "mps" else device
        self.a64 = None

    def load(self):
        g = self.g
        self.levels = [_DevLevel(g._l0s, self.dev, self.dtype)] + \
                      [_DevLevel(lv, self.dev, self.dtype) for lv in g.levels[1:]]
        self.a64 = _DevLevel(g.levels[0], self.cg_dev, self.torch.float64)
        for i, lv in enumerate(self.levels):
            lv.lmax = self._estimate_lmax(i)
            # keep the CPU levels' bound in step (diagnostics, CPU fallback)
            g._op(i).lmax = lv.lmax

    def _estimate_lmax(self, i, its=12):
        torch = self.torch
        lv = self.levels[i]
        prev = self._lmax_vec.get(i)
        if prev is not None and tuple(prev.shape) == tuple(lv.node_shape):
            v = prev * lv.mask
        else:
            rng = np.random.default_rng(i)
            v = torch.as_tensor(rng.random(lv.node_shape) - 0.5, dtype=self.dtype, device=self.dev) * lv.mask
        lam = 1.0
        for _ in range(its):
            nv = float(torch.linalg.vector_norm(v))
            if nv == 0:
                return 2.0
            v = v / nv
            av = lv.dinv * lv.apply(v) * lv.mask
            lam = float(torch.dot(v.reshape(-1), av.reshape(-1)))
            v = av
        self._lmax_vec[i] = v
        return LMAX_SAFETY * max(lam, 1e-12)

    def _smooth(self, lv, b, x):
        hi = lv.lmax
        lo = hi / 30.0
        theta = 0.5 * (hi + lo)
        delta = 0.5 * (hi - lo)
        sigma = theta / delta
        rho = 1.0 / sigma
        if x is None:
            r = b.clone()
            x = self.torch.zeros_like(b)
        else:
            r = b - lv.apply(x)
        d = lv.dinv * r / theta
        for it in range(self.g.degree):
            x += d
            if it + 1 == self.g.degree:
                break
            r -= lv.apply(d)
            rho_new = 1.0 / (2.0 * sigma - rho)
            d = rho_new * rho * d + (2.0 * rho_new / delta) * (lv.dinv * r)
            rho = rho_new
        return x

    def _vcycle(self, i, b):
        torch = self.torch
        g = self.g
        lv = self.levels[i]
        if i == len(self.levels) - 1:
            if g._coarse is not None:
                xc = g._coarse_solve(b.cpu().numpy())
                return torch.as_tensor(xc, dtype=self.dtype, device=self.dev)
            x = self._smooth(lv, b, None)
            return self._smooth(lv, b, x)
        x = self._smooth(lv, b, None)
        r = b - lv.apply(x)
        cs = self.levels[i + 1].node_shape
        rf = torch.zeros((3,) + tuple(2 * n - 1 for n in cs[1:]), dtype=r.dtype, device=self.dev)
        sx, sy, sz = lv.node_shape[1:]
        rf[:, :sx, :sy, :sz] = r * lv.mask
        xc = self._vcycle(i + 1, _restrict(rf))
        x += _prolong(xc)[:, :sx, :sy, :sz] * lv.mask
        return self._smooth(lv, b, x)

    def _warm(self, A, b, x, norm, dot):
        """GridElasticity._pcg's warm start: the energy-optimal multiple of
        x0, or zero if even that is worse. Returns (x, r)."""
        ax = A.apply(x)
        xax = dot(x, ax)
        s = dot(b, x) / xax if xax > 0 else 0.0
        if s > 0:
            rs = b - s * ax
            if norm(rs) < norm(b):
                return x * s, rs
        return x.zero_(), b.clone()

    def solve(self, b, x0):
        """GridElasticity._pcg with torch: float64 CG on `cg_dev`, float32
        V-cycle on `dev`. float64 b, x0 (node grids, masked) -> (x,
        iterations, converged)."""
        torch = self.torch
        g = self.g
        f64 = torch.float64
        dot = lambda p, q: float(torch.dot(p.reshape(-1), q.reshape(-1)))
        norm = lambda v: float(torch.linalg.vector_norm(v))
        A = self.a64
        cdev = self.cg_dev
        bd = torch.as_tensor(b, dtype=f64, device=cdev).clone()
        bnorm = norm(bd)
        if bnorm == 0:
            return np.zeros_like(b), 0, True
        if x0 is not None:
            x, r = self._warm(A, bd, torch.as_tensor(x0, dtype=f64, device=cdev).clone(), norm, dot)
        else:
            x, r = torch.zeros_like(bd), bd.clone()
        pre = lambda r: self._vcycle(0, r.to(self.dev, self.dtype)).to(cdev, f64)
        z = pre(r)
        p = z.clone()
        rz = dot(r, z)
        tol = g.rtol * bnorm
        for it in range(1, g.maxiter + 1):
            ap = A.apply(p)
            pap = dot(p, ap)
            if pap <= 0:
                return x.cpu().numpy(), it, False
            alpha = rz / pap
            x.add_(p, alpha=alpha)
            r.sub_(ap, alpha=alpha)
            if norm(r) <= tol:
                return x.cpu().numpy(), it, True
            z = pre(r)
            rz_new = dot(r, z)
            p.mul_(rz_new / rz).add_(z)
            rz = rz_new
        return x.cpu().numpy(), g.maxiter, False
