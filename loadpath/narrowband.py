"""Narrow-band sampling of an implicit field for meshing.

A part defined by a field f(P) < 0 only needs f near its surface: deep
inside or far outside, marching cubes finds nothing. `narrowband_mc` cuts
the meshing grid into blocks of `block` cells, samples f once per block
(at its centre) and skips every block whose centre is further from the
zero level than the field can change across the block (Lipschitz bound
`L` times the half-diagonal). Only the remaining blocks are sampled at
full resolution and marched, one block at a time.

The surviving blocks share their boundary samples with their neighbours,
and marching cubes puts a vertex on an edge from that edge's two samples
alone, so the blocks' meshes join exactly: after merging the shared
vertices the result is the same mesh dense marching cubes gives on the
whole grid (tests/test_narrowband.py checks this), at a fraction of the
field evaluations and with memory set by one batch of blocks, not by the
grid. That is what lets a field go to a finer pitch than a dense grid
fits in memory.

The skip test is only as safe as `L`. For a distance field (shapes,
booleans, shells, offsets) L = 1; smooth unions, remaps of a distance and
plain TPMS level sets in fields.py stay close to it, but a warp-graded
lattice reaches 30 or more. `lipschitz=None` estimates L as the steepest
gradient at 20k random points or between neighbouring block centres,
times `margin`, and never below 1. A bound that is still too low leaves
a crossing in a skipped leaf, which shows as a hole where it meets a
sampled one: fields.field_mesh checks the mesh is watertight and, if it
is not, marches again with every block sampled.
"""
from __future__ import annotations

import math
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np

__all__ = ["narrowband_mc", "march_blocks", "peval", "watertight", "signed_volume", "shell_volumes"]

_BATCH = 2_000_000      # field samples per call
_CHUNK = 250_000        # field samples per thread task


def _threads():
    env = os.environ.get("LOADPATH_THREADS")
    if env:
        return max(1, int(env))
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except AttributeError:
        return max(1, os.cpu_count() or 1)


_POOL = None


def peval(field, P, chunk=_CHUNK):
    """field(P) as float64, in chunks spread over threads. The fields are
    numpy expressions, which release the GIL, so 4 cores evaluate about
    3.7x faster, and every chunk is the same computation, so the result is
    bit for bit the serial one. The first chunk runs alone, so a field that
    builds a cached resource on first use (a warp, a mesh's distance grid)
    builds it once. LOADPATH_THREADS=1 turns this off."""
    global _POOL
    n = len(P)
    out = np.empty(n)
    nt = _threads()
    if nt == 1 or n <= 2 * chunk:
        for i in range(0, n, chunk):
            out[i:i + chunk] = np.asarray(field(P[i:i + chunk]), float)
        return out
    out[:chunk] = np.asarray(field(P[:chunk]), float)
    if _POOL is None or _POOL._max_workers != nt:
        _POOL = ThreadPoolExecutor(nt, thread_name_prefix="loadpath-field")

    def one(i):
        out[i:i + chunk] = np.asarray(field(P[i:i + chunk]), float)

    for _ in _POOL.map(one, range(chunk, n, chunk)):
        pass
    return out


def _grid_points(lo, pitch, i0, n):
    """Points of the sub-grid starting at index i0 (3,) with n (3,) samples."""
    ax = [lo[k] + (i0[k] + np.arange(n[k])) * pitch for k in range(3)]
    X, Y, Z = np.meshgrid(*ax, indexing="ij")
    return np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)


def _eval(field, P):
    return peval(field, P)


def _lipschitz(field, lo, hi, pitch, n=20_000):
    """The steepest gradient seen at n random points (central differences)."""
    P = np.random.default_rng(0).uniform(lo, hi, size=(n, 3))
    h = 0.05 * pitch
    E = np.eye(3) * h
    f = _eval(field, np.concatenate([P + e for e in E] + [P - e for e in E])).reshape(6, n)
    g = (f[:3] - f[3:]) / (2 * h)
    return float(np.sqrt((g * g).sum(axis=0)).max())


def narrowband_mc(field, lo, dims, pitch, block=32, leaf=4, lipschitz=None, margin=1.5, budget=None,
                  progress=None):
    """Marching cubes of field < 0 on the grid lo + i * pitch, i < dims,
    sampling only near the zero level. The grid's outer layer is forced
    outside (+pitch), as fields.field_mesh does, so the mesh is closed.

    Two levels: blocks of `block` cells are tested at their centres; in
    the blocks that pass, leaves of `leaf` cells are tested the same way,
    and only the samples of the leaves that pass are evaluated. A sample
    of a block that no passing leaf touches gets the sign of its leaf at
    a large magnitude: with the Lipschitz bound holding, every edge the
    surface crosses has both its samples evaluated, so the block's mesh is
    the dense one. Each block is marched on its own and the shared
    vertices merged. Raises ValueError, before the fine pass, when the
    passing leaves need more than `budget` samples. Returns (verts (mm),
    faces, stats)."""
    stats = {}
    verts, faces, nv = [], [], 0
    for i0, n, v, fa in march_blocks(field, lo, dims, pitch, block, leaf, lipschitz, margin, budget, progress,
                                     stats):
        verts.append(v)
        faces.append(fa + nv)
        nv += len(v)
    if not verts:
        return np.zeros((0, 3)), np.zeros((0, 3), int), stats
    V, Fc = merge_blocks(np.concatenate(verts), np.concatenate(faces), block)
    return np.asarray(lo, float) + V * pitch, Fc, stats


def merge_blocks(V, F, block):
    """Merge the vertices the blocks share. A vertex on a block's face comes
    out of both blocks bit for bit (same edge, same two samples) and every
    other vertex is unique already, so only the ones on a block plane (a
    coordinate that is a multiple of `block`) go through the (slow) row
    unique. Drops the faces the merge collapses."""
    on = (np.mod(V, block) == 0).any(axis=1)
    idx = np.flatnonzero(on)
    remap = np.arange(len(V))
    if len(idx):
        _, first, inv = np.unique(V[idx], axis=0, return_index=True, return_inverse=True)
        remap[idx] = idx[first][inv.reshape(-1)]
    keep_v = remap == np.arange(len(V))
    new = np.cumsum(keep_v) - 1
    F = new[remap[F]]
    V = V[keep_v]
    ok = (F[:, 0] != F[:, 1]) & (F[:, 1] != F[:, 2]) & (F[:, 0] != F[:, 2])
    return V, F[ok]


def march_blocks(field, lo, dims, pitch, block=32, leaf=4, lipschitz=None, margin=1.5, budget=None,
                 progress=None, stats=None):
    """The narrow-band blocks of narrowband_mc, marched one at a time, in
    x-major order (all the blocks of one x slab before the next): yields
    (i0, n, verts, faces) per block that the surface crosses, verts in grid
    index units (add lo, times pitch, for mm). Fills `stats` once the
    blocks are planned. The field is evaluated in batches of blocks; memory
    is one batch, whatever the grid."""
    from skimage.measure import marching_cubes

    lo = np.asarray(lo, float)
    dims = np.asarray(dims, int)
    B, lf = int(block), int(leaf)
    if B % lf:
        raise ValueError("block must be a multiple of leaf")
    ncells = dims - 1
    nb = np.maximum(1, np.ceil(ncells / B).astype(int))

    def centres(i0, i1):
        """Centres (mm) and half-diagonals of the boxes [i0, i1) in cells."""
        return lo + 0.5 * (i0 + i1) * pitch, 0.5 * np.linalg.norm((i1 - i0) * pitch, axis=-1)

    bi = np.stack(np.meshgrid(*[np.arange(n) for n in nb], indexing="ij"), -1).reshape(-1, 3)
    b0 = bi * B
    b1 = np.minimum(b0 + B, ncells)
    C, half = centres(b0, b1)
    fc = _eval(field, C)
    if lipschitz is None:
        L = _lipschitz(field, lo, lo + ncells * pitch, pitch)
        F3 = fc.reshape(tuple(nb))
        for k in range(3):
            if nb[k] > 1:
                L = max(L, float(np.max(np.abs(np.diff(F3, axis=k)))) / (B * pitch))
        L = max(1.0, L * margin)
    else:
        L = float(lipschitz)
    on_border = (b0 == 0).any(axis=1) | (b1 == ncells).any(axis=1)
    act = (np.abs(fc) <= L * half + pitch) | (on_border & (fc < 0))
    bi, b0, b1 = bi[act], b0[act], b1[act]

    # leaves of the passing blocks
    nl = B // lf
    li = np.stack(np.meshgrid(*[np.arange(nl)] * 3, indexing="ij"), -1).reshape(-1, 3)
    l0 = (b0[:, None, :] + li[None] * lf)
    l1 = np.minimum(l0 + lf, b1[:, None, :])
    real = (l0 < b1[:, None, :]).all(axis=2)            # leaves past a short block's end are empty
    Cl, hl = centres(l0.reshape(-1, 3), l1.reshape(-1, 3))
    fl = np.full(len(Cl), np.inf)
    rf = real.ravel()
    fl[rf] = _eval(field, Cl[rf])
    fl = fl.reshape(len(b0), -1)
    lon = ((l0 == 0) | (l1 == ncells)).any(axis=2)
    lact = real & ((np.abs(fl) <= L * hl.reshape(len(b0), -1) + pitch) | (lon & (fl < 0)))
    keep = lact.any(axis=1)
    b0, b1, fl, lact = b0[keep], b1[keep], fl[keep], lact[keep]

    def mask(k):
        """The samples block k needs: every sample of a passing leaf."""
        m = np.zeros(tuple(b1[k] - b0[k] + 1), bool)
        for j in np.flatnonzero(lact[k]):
            a = li[j] * lf
            m[a[0]:a[0] + lf + 1, a[1]:a[1] + lf + 1, a[2]:a[2] + lf + 1] = True
        return m

    fine = None
    if budget is not None:
        fine = sum(int(mask(k).sum()) for k in range(len(b0)))
        if fine > budget:
            raise ValueError(f"the surface needs {fine / 1e6:.1f}M samples at pitch {pitch:g}, over the "
                             f"{budget / 1e6:.0f}M budget: use a coarser pitch")
    if stats is None:
        stats = {}
    stats.update({"blocks": int(np.prod(nb)), "activeBlocks": int(len(b0)), "leaves": int(lact.sum()),
                  "lipschitz": round(float(L), 3) if np.isfinite(L) else None,
                  "samples": int(len(C) + rf.sum()), "denseSamples": int(np.prod(dims))})

    big = np.float32(1e3 * pitch)
    k = 0
    while k < len(b0):
        # a batch of blocks, one field call
        group, P, cnt = [], [], 0
        while k < len(b0) and (not group or cnt < _BATCH):
            n = b1[k] - b0[k] + 1
            ii = np.argwhere(mask(k))
            P.append(lo + (b0[k] + ii) * pitch)
            group.append((k, n, ii))
            cnt += len(ii)
            k += 1
        F = _eval(field, np.concatenate(P)).astype(np.float32)
        del P
        stats["samples"] += len(F)
        o = 0
        for kk, n, ii in group:
            # unevaluated samples: their leaf's sign, far from zero
            lix = np.minimum(np.indices(tuple(n)).reshape(3, -1).T // lf, nl - 1)
            sgn = np.sign(fl[kk][(lix[:, 0] * nl + lix[:, 1]) * nl + lix[:, 2]])
            f = (sgn * big).astype(np.float32).reshape(tuple(n))
            f[tuple(ii.T)] = F[o:o + len(ii)]
            o += len(ii)
            i0 = b0[kk]
            for ax in range(3):
                if i0[ax] == 0:
                    f[(slice(None),) * ax + (0,)] = pitch
                if i0[ax] + n[ax] == dims[ax]:
                    f[(slice(None),) * ax + (-1,)] = pitch
            f[f == 0.0] = -1e-4 * pitch
            if f.min() >= 0 or f.max() <= 0:
                continue
            v, fa, _, _ = marching_cubes(f, 0.0)
            yield i0, n, v.astype(np.float64) + i0, fa
        if progress:
            progress({"stage": "field", "done": int(k), "of": int(len(b0))})


# ---- fast bookkeeping on big triangle meshes (trimesh's takes tens of seconds) ----

def watertight(F, nv=None):
    """Every edge shared by exactly two faces (trimesh's is_watertight)."""
    F = np.asarray(F, np.int64)
    if not len(F):
        return False
    nv = int(F.max()) + 1 if nv is None else int(nv)
    a = F.ravel()
    b = np.roll(F, -1, axis=1).ravel()
    e = np.sort(np.minimum(a, b) * nv + np.maximum(a, b))
    if len(e) % 2:
        return False
    if not np.array_equal(e[0::2], e[1::2]):
        return False
    return bool(len(e) < 4 or not np.any(e[1:-1:2] == e[2::2]))


def signed_volume(V, F):
    """Enclosed volume (divergence theorem), positive with outward normals."""
    a, b, c = (V[F[:, i]] for i in range(3))
    return float(np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6.0)


def shell_volumes(V, F):
    """(label per face, signed volume per connected shell): shells joined
    through shared vertices. A closed body's outside is positive, the wall
    of a void sealed inside it negative."""
    import scipy.sparse as sp
    from scipy.sparse.csgraph import connected_components

    n = len(V)
    A = sp.coo_matrix((np.ones(2 * len(F), np.int8), (np.r_[F[:, 0], F[:, 1]], np.r_[F[:, 1], F[:, 2]])),
                      shape=(n, n))
    _, lab = connected_components(A, directed=False)
    fl = lab[F[:, 0]]
    a, b, c = (V[F[:, i]] for i in range(3))
    vol = np.bincount(fl, weights=np.einsum("ij,ij->i", a, np.cross(b, c)) / 6.0)
    # renumber to the shells that have faces
    used = np.unique(fl)
    ren = np.full(lab.max() + 1, -1)
    ren[used] = np.arange(len(used))
    return ren[fl], vol[used]
