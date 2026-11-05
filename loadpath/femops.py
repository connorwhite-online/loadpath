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

__all__ = ["available", "element_energy", "scatter"]


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


# below this many elements a parallel region per corner pair costs more than
# it saves (64 thread launches; on a busy machine each can stall)
_SERIAL_BELOW = 20_000


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
