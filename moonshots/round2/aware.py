"""Experimental (not landed): crease-aware projection + label-preserving
relax of a RAW marching-cubes mesh, then sharpen. Measured in crease_bench
--aware: slightly better crease accuracy on acute creases, but more slivers
and folds than sharpening the normal pipeline's mesh, so it stays here."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from loadpath.crease import _face_normals, _leaf_grad, _on_part, meet, sharpen  # noqa: E402,F401


# ---- crease-aware projection and relaxation of a raw marching-cubes mesh ----------

def _own_project(tree, V, lab, pitch, iters=6):
    """Newton of each vertex onto its own leaf's zero level (smooth: no
    gradient taken across a crease)."""
    X, ok = meet(tree, V, lab[:, None], pitch, iters=iters)
    return X, ok


def _neighbours(F, n):
    import scipy.sparse as sp

    a = np.r_[F[:, 0], F[:, 1], F[:, 2]]
    b = np.r_[F[:, 1], F[:, 2], F[:, 0]]
    A = sp.coo_matrix((np.ones(2 * len(a), np.float32), (np.r_[a, b], np.r_[b, a])), shape=(n, n)).tocsr()
    A.data[:] = 1.0
    return A, np.maximum(np.diff(A.indptr), 1)


def finish(V, F, tree, pitch, rounds=3, field=None, snap=0.25, stats=None):
    """Project, sharpen and relax a raw marching-cubes mesh of the tree's
    field, crease-aware throughout (the replacement for fields._relax on a
    field with creases):

    1. every vertex goes onto its own active leaf's surface (Newton on that
       leaf alone); one that lands where another leaf takes over has gone
       past the crease, and goes onto the crease instead (both leaves);
    2. sharpen() splits the edges that still cross a crease;
    3. `rounds` of relaxing: the off-crease vertices move halfway to their
       neighbours' centroid within their own surface's tangent plane and back
       onto their leaf; a move that would change a vertex's label (carry it
       over the crease) or turn a triangle over is not made. Crease
       vertices stay put.

    Returns (V, F, crease mask)."""
    st = stats if stats is not None else {}
    V = np.array(V, float)
    F = np.asarray(F, np.int64)
    if field is None:
        def field(P):
            return tree.evaluate(P)[0]
    _, lab = tree.evaluate(V)
    X, ok = _own_project(tree, V, lab, pitch)
    f, act = tree.evaluate(X)
    good = ok & (act == lab) & (np.abs(f) < 1e-5 * pitch)
    # vertices that went past the crease (or did not converge): onto the crease
    mask = np.int64(1) << lab.astype(np.int64)
    past = np.flatnonzero(ok & (act != lab))
    st["pastCrease"] = int(len(past))
    if len(past):
        Xc, okc = meet(tree, V[past], np.stack([lab[past], act[past]], 1), pitch)
        okc &= np.linalg.norm(Xc - V[past], axis=1) < 1.5 * pitch
        if okc.any():
            okc[okc] &= _on_part(tree, Xc[okc], pitch)
        sel = past[okc]
        X[sel] = Xc[okc]
        mask[sel] |= np.int64(1) << act[past[okc]].astype(np.int64)
        good[sel] = True
    rest = np.flatnonzero(~good)
    st["projectFallback"] = int(len(rest))
    if len(rest):
        from .fields import _project

        q = V[rest]
        for _ in range(4):
            q = _project(field, q, pitch)
        X[rest] = q
        _, lab[rest] = tree.evaluate(q)
        mask[rest] = np.int64(1) << lab[rest].astype(np.int64)
    V = X
    V, F, crease = sharpen(V, F, tree, pitch, snap=snap, field=field, stats=st, lab=lab, mask=mask)
    # labels of the new vertices: their own (crease vertices are pinned anyway)
    _, lab = tree.evaluate(V)
    if rounds:
        V = relax(V, F, tree, lab, crease, pitch, rounds, stats=st)
    return V, F, crease


def relax(V, F, tree, lab, pinned, pitch, rounds, stats=None):
    """Label-preserving tangential relaxation (see finish)."""
    import scipy.sparse as sp

    n = len(V)
    A, deg = _neighbours(F, n)
    move = ~np.asarray(pinned, bool)
    # vertex -> faces, for the turn-over check
    vf = sp.coo_matrix((np.ones(3 * len(F)), (F.ravel(), np.repeat(np.arange(len(F)), 3))),
                       shape=(n, len(F))).tocsr()
    rejected = 0
    for _ in range(rounds):
        idx = np.flatnonzero(move)
        if not len(idx):
            break
        _, g = _leaf_grad(tree, V[idx], lab[idx][:, None], 1e-4 * pitch)
        nrm = g[:, 0] / np.maximum(np.linalg.norm(g[:, 0], axis=1, keepdims=True), 1e-12)
        d = (A @ V)[idx] / deg[idx, None] - V[idx]
        d -= (d * nrm).sum(1, keepdims=True) * nrm
        cand = V[idx] + 0.5 * d
        cand, ok = meet(tree, cand, lab[idx][:, None], pitch, iters=4)
        f, act = tree.evaluate(cand)
        ok &= (act == lab[idx]) & (np.abs(f) < 1e-5 * pitch)
        # no triangle may turn over (each move judged alone against the old
        # neighbours: moves of neighbours in the same round are small)
        N0 = _face_normals(V, F)
        Vn = V.copy()
        Vn[idx[ok]] = cand[ok]
        N1 = _face_normals(Vn, F)
        flip = (N0 * N1).sum(1) < 0.5
        badv = np.asarray(vf[:, flip].sum(axis=1)).ravel() > 0
        ok &= ~badv[idx]
        rejected += int((~ok).sum())
        V = V.copy()
        V[idx[ok]] = cand[ok]
    if stats is not None:
        stats["relaxRejected"] = rejected
    return V


