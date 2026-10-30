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

import numpy as np

__all__ = ["narrowband_mc"]

_BATCH = 2_000_000      # field samples per call


def _grid_points(lo, pitch, i0, n):
    """Points of the sub-grid starting at index i0 (3,) with n (3,) samples."""
    ax = [lo[k] + (i0[k] + np.arange(n[k])) * pitch for k in range(3)]
    X, Y, Z = np.meshgrid(*ax, indexing="ij")
    return np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)


def _eval(field, P):
    out = np.empty(len(P))
    for i in range(0, len(P), _BATCH):
        out[i:i + _BATCH] = np.asarray(field(P[i:i + _BATCH]), float)
    return out


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

    # the samples each block needs: every sample of a passing leaf
    masks = []
    fine = 0
    for k in range(len(b0)):
        n = b1[k] - b0[k] + 1
        m = np.zeros(tuple(n), bool)
        for j in np.flatnonzero(lact[k]):
            a = li[j] * lf
            m[a[0]:a[0] + lf + 1, a[1]:a[1] + lf + 1, a[2]:a[2] + lf + 1] = True
        masks.append(m)
        fine += int(m.sum())
    if budget is not None and fine > budget:
        raise ValueError(f"the surface needs {fine / 1e6:.1f}M samples at pitch {pitch:g}, over the "
                         f"{budget / 1e6:.0f}M budget: use a coarser pitch")

    verts, faces, nv = [], [], 0
    big = np.float32(1e3 * pitch)
    k = 0
    while k < len(b0):
        # a batch of blocks, one field call
        group, P, cnt = [], [], 0
        while k < len(b0) and (not group or cnt < _BATCH):
            n = b1[k] - b0[k] + 1
            ii = np.argwhere(masks[k])
            P.append(lo + (b0[k] + ii) * pitch)
            group.append((k, n, ii))
            cnt += len(ii)
            k += 1
        F = _eval(field, np.concatenate(P)).astype(np.float32)
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
            verts.append(v.astype(np.float64) + i0)
            faces.append(fa + nv)
            nv += len(v)
        if progress:
            progress({"stage": "field", "done": int(k), "of": int(len(b0))})
    stats = {"blocks": int(np.prod(nb)), "activeBlocks": int(len(b0)), "leaves": int(lact.sum()),
             "lipschitz": round(float(L), 3) if np.isfinite(L) else None, "samples": int(len(C) + rf.sum() + fine),
             "denseSamples": int(np.prod(dims))}
    if not verts:
        return np.zeros((0, 3)), np.zeros((0, 3), int), stats
    V = np.concatenate(verts)
    Fc = np.concatenate(faces)
    # vertices on block faces come out twice, bit for bit (same edge, same two samples)
    _, first, inv = np.unique(V, axis=0, return_index=True, return_inverse=True)
    V = V[first]
    Fc = inv.reshape(-1)[Fc]
    keep = (Fc[:, 0] != Fc[:, 1]) & (Fc[:, 1] != Fc[:, 2]) & (Fc[:, 0] != Fc[:, 2])
    return lo + V * pitch, Fc[keep], stats
