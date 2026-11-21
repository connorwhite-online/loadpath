"""
A support that only pushes: a face resting on a wall, a table or a mating
part (supports[].push).

Held "fix: all", a bolted face is glued to what it sits on: where the load
pries it off, the glue pulls, and the part passes on a hold it does not have.
A seat carries pressure only. The part's normal displacement on the seat may
not go into it (gap >= 0), the seat's force is a push (lambda >= 0), and it
pushes only where it touches (lambda * gap = 0). Friction is left out: the
fasteners carry what slides and what lifts, which is the safe side.

The contact is solved exactly on the linear model, with no change to the
stiffness or its factorization. The seat's nodes are free; its force is an
unknown on a few patches of the face (constant pressure over each). With
w_j the part's displacement under a unit push on patch j, any load's answer
is u = u0 + sum_j lambda_j w_j, and the gaps are g = g0 + C lambda with
C_ij = p_i . w_j (symmetric, positive definite once the fasteners hold the
part). The lambda that satisfies the three conditions minimizes
1/2 lambda' C lambda + g0' lambda over lambda >= 0: a non-negative least
squares problem in as many unknowns as there are patches.

Units: mm, N.
"""
from __future__ import annotations

import numpy as np

__all__ = ["patches", "resolve", "solve"]

MAX_PATCHES = 48


def patches(seats, xyz, n_dof, most: int = MAX_PATCHES):
    """Unit pushes on the seats' patches: a list of force vectors (n_dof),
    each 1 N along its seat's normal spread over the patch's nodes by their
    share of the face. `seats`: [(node indices, normal (3,), weights)]."""
    out = []
    per = max(1, most // max(len(seats), 1))
    for idx, n, w in seats:
        idx, w = np.asarray(idx), np.asarray(w, float)
        pts = xyz[idx] - xyz[idx].min(axis=0)
        size = 0.0
        span = float(pts.max()) if len(pts) else 0.0
        keys = np.zeros(len(idx), int)
        if len(idx) > per:
            # a grid of squares over the face, as coarse as it takes
            size = span / max(np.sqrt(per), 1.0)
            while True:
                cell = np.floor(pts / max(size, 1e-12)).astype(int)
                _, keys = np.unique(cell, axis=0, return_inverse=True)
                keys = np.asarray(keys).ravel()
                if keys.max() + 1 <= per:
                    break
                size *= 1.25
        else:
            keys = np.arange(len(idx))
        for k in range(int(keys.max()) + 1 if len(idx) else 0):
            m = keys == k
            p = np.zeros(n_dof)
            share = w[m] / w[m].sum()
            for a in range(3):
                p[3 * idx[m] + a] = share * n[a]
            out.append(p)
    return out


def resolve(u0s, ws, ps):
    """Displacements with the seats pushing: for each u0 (the answer with the
    seats absent), (u, lambda) with lambda (N per patch) the seats' forces."""
    from scipy.optimize import nnls

    p = np.array(ps)
    w = np.array(ws)
    c = p @ w.T
    c = 0.5 * (c + c.T)
    # C = L L': min 1/2 l'Cl + g0'l  ==  min |L'l + L^-1 g0|^2
    jitter = 1e-12 * float(np.trace(c)) / len(c)
    chol = np.linalg.cholesky(c + jitter * np.eye(len(c)))
    out = []
    for u0 in u0s:
        g0 = p @ u0
        lam, _ = nnls(chol.T, -np.linalg.solve(chol, g0), maxiter=50 * len(c))
        out.append((u0 + lam @ w, lam))
    return out


def solve(inner, rhs, seats, xyz, n_dof):
    """`inner(rhs) -> (stiffness, displacements)` with the seats absent;
    returns the same with the seats pushing, and their forces per rhs."""
    rhs = list(rhs)
    ps = patches(seats, xyz, n_dof)
    stiff, us = inner(rhs + ps)
    us = list(us)
    done = resolve(us[:len(rhs)], us[len(rhs):], ps)
    return stiff, [u for u, _ in done], [lam for _, lam in done]
