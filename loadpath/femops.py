"""Compiled hex-element kernels for the SIMP loop, each giving the numpy
expression it replaces bit for bit:

- `element_energy(u, v, edof, ke)`: einsum("ij,jk,ik->i", u[edof], ke,
  v[edof]), the per-element strain energy (sensitivities), with the gather
  fused in: ~16x on 200k elements, the same sum in the same order (j
  outer, k inner, (u_j ke_jk) v_k) as numpy's unoptimized einsum.
- `scatter(data, slots, blks, scale)`: fem.Stiffness.assemble's
  data[slots] += scale * blk over its 64 corner pairs in turn (a pair's
  slots are distinct, so its elements run in parallel and each slot still
  gets its additions in pair order).
- `gather_plan(slots, nnz)` + `scatter_gather(data, plan, blks, scale)`:
  the same sums, slot by slot in one parallel pass (each slot reads its
  contributions in pair order, so the same bits). One parallel region
  instead of 64: with other processes on the cores (the search's pitch
  starts, two eval cases at once) the 64 barriers made an arm-link
  assembly 0.6 s against 0.05 s on one thread.

`available()` is False without numba or with LOADPATH_FEMOPS=numpy.
"""
from __future__ import annotations

import os

import numpy as np

try:
    import numba as nb

    from . import meshops as _meshops  # noqa: F401 - caps numba's threads (LOADPATH_THREADS)
except ImportError:  # pragma: no cover - numba is a dependency
    nb = None

__all__ = ["available", "element_energy", "scatter", "gather_plan", "scatter_gather"]


def available():
    return nb is not None and os.environ.get("LOADPATH_FEMOPS", "").lower() != "numpy"


if nb is not None:

    @nb.njit(cache=True, parallel=True)
    def _energy(u, v, edof, ke, out):
        n, m = edof.shape
        for i in nb.prange(n):
            acc = 0.0
            for j in range(m):
                uj = u[edof[i, j]]
                for k in range(m):
                    acc += uj * ke[j, k] * v[edof[i, k]]
            out[i] = acc

    @nb.njit(cache=True, parallel=True)
    def _scatter(data, slots, blks, scale):
        npair, ne, nb_ = slots.shape
        for p in range(npair):
            for e in nb.prange(ne):
                s = scale[e]
                for t in range(nb_):
                    data[slots[p, e, t]] += s * blks[p, t]

    @nb.njit(cache=True)
    def _scatter_serial(data, slots, blks, scale):
        npair, ne, nb_ = slots.shape
        for p in range(npair):
            for e in range(ne):
                s = scale[e]
                for t in range(nb_):
                    data[slots[p, e, t]] += s * blks[p, t]


    @nb.njit(cache=True)
    def _plan(flat, nnz, ptr, src):
        # counting sort of the flattened (pair, element, entry) slots,
        # stable: each slot's contributions in pair order
        for i in range(flat.size):
            ptr[flat[i] + 1] += 1
        for s in range(nnz):
            ptr[s + 1] += ptr[s]
        fill = ptr[:-1].copy()
        for i in range(flat.size):
            s = flat[i]
            src[fill[s]] = i
            fill[s] += 1

    @nb.njit(cache=True, parallel=True)
    def _gather(data, ptr, src, ne, blks, scale):
        nnz = data.size
        nchunk = 256
        step = (nnz + nchunk - 1) // nchunk
        for c in nb.prange(nchunk):
            for s in range(c * step, min(nnz, (c + 1) * step)):
                acc = data[s]
                for q in range(ptr[s], ptr[s + 1]):
                    f = src[q]
                    t = f % 9
                    pe = f // 9
                    e = pe % ne
                    p = pe // ne
                    acc += scale[e] * blks[p, t]
                data[s] = acc


# below this many elements a parallel region per corner pair costs more than
# it saves (64 thread launches; on a busy machine each can stall)
_SERIAL_BELOW = 20_000


def gather_plan(slots, nnz):
    """For scatter_gather: (ptr, src), each slot's entries of `slots`
    (flattened, in pair order)."""
    flat = np.ascontiguousarray(slots).ravel()
    ptr = np.zeros(nnz + 1, np.int64)
    src = np.empty(flat.size, np.int32 if flat.size < 2 ** 31 - 1 else np.int64)
    _plan(flat, nnz, ptr, src)
    return ptr, src


def scatter_gather(data, plan, blks, scale, ne):
    """scatter(data, slots, blks, scale), bit for bit, from gather_plan(slots)."""
    _gather(data, plan[0], plan[1], ne, np.ascontiguousarray(blks, np.float64),
            np.ascontiguousarray(scale, np.float64))
    return data


def element_energy(u, v, edof, ke):
    """einsum("ij,jk,ik->i", u[edof], ke, v[edof]), bit for bit."""
    u = np.ascontiguousarray(u, np.float64)
    v = u if v is None else np.ascontiguousarray(v, np.float64)
    out = np.empty(len(edof))
    _energy(u, v, np.ascontiguousarray(edof), np.ascontiguousarray(ke, np.float64), out)
    return out


def scatter(data, slots, blks, scale):
    """data[slots[p]] += scale[:, None] * blks[p] for p in order, in place."""
    run = _scatter_serial if slots.shape[1] < _SERIAL_BELOW else _scatter
    run(data, slots, np.ascontiguousarray(blks, np.float64), np.ascontiguousarray(scale, np.float64))
    return data
