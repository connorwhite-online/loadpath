"""The tet confirm's quadratic solve (crosscheck._solve, tetsolve): node
colours that share no element, a parallel assembly with the serial one's
bits, every load case in one CG loop with the answers of a direct solve,
the factored coarse level and the V-cycle one agreeing, and a solve that
repeats to the bit.

Run: `python -m pytest tests/test_tetsolve.py -q` (about half a minute).
"""
import numpy as np
import pytest
import scipy.sparse.linalg as spla

from loadpath import crosscheck as cc
from loadpath import tetsolve


def _block(n=(10, 4, 4), h=1.0):
    """A box of n cubes of edge h, six tetrahedra each: (pts, tets)."""
    ax = [np.arange(k + 1) * h for k in n]
    g = np.stack(np.meshgrid(*ax, indexing="ij"), axis=-1).reshape(-1, 3)
    idx = np.arange(len(g)).reshape([k + 1 for k in n])
    c = [idx[i:i + n[0], j:j + n[1], k:k + n[2]].ravel() for i in (0, 1) for j in (0, 1) for k in (0, 1)]
    paths = [(0, 4, 6, 7), (0, 6, 2, 7), (0, 2, 3, 7), (0, 3, 1, 7), (0, 1, 5, 7), (0, 5, 4, 7)]
    tets = np.concatenate([np.stack([c[a] for a in p], axis=1) for p in paths])
    x = g[tets]
    neg = np.linalg.det(x[:, 1:] - x[:, :1]) < 0
    tets[neg] = tets[neg][:, [0, 2, 1, 3]]
    return g, tets


def _cantilever(n=(10, 4, 4)):
    """P2 cantilever held at x = 0: (nodes, conn, edges, nv, fixed dofs,
    loads: a tip shear and a tip twist)."""
    pts, tets = _block(n)
    nodes, conn, edges = cc._p2_mesh(pts, tets)
    held = np.isclose(nodes[:, 0], 0.0)
    fixed = np.repeat(held, 3)
    tip = np.nonzero(np.isclose(nodes[:, 0], n[0]))[0]
    f1 = np.zeros((len(nodes), 3))
    f1[tip, 2] = -1.0 / len(tip)
    f2 = np.zeros((len(nodes), 3))
    r = nodes[tip, 1:] - nodes[tip, 1:].mean(axis=0)
    f2[tip, 1], f2[tip, 2] = -r[:, 1], r[:, 0]
    loads = [("shear", f1, tip), ("twist", f2, tip)]
    return nodes, conn, edges, len(pts), fixed, loads


def test_colours_share_no_element():
    nodes, conn, *_ = _cantilever()
    nptr, nind = cc._node_pattern(conn, len(nodes))
    q, bounds = tetsolve.color_order(nptr, nind)
    assert sorted(q.tolist()) == list(range(len(nodes)))
    colour = np.empty(len(nodes), np.int64)
    colour[q] = np.searchsorted(bounds, np.arange(len(nodes)), side="right") - 1
    per = colour[conn]
    assert all(len(set(row)) == conn.shape[1] for row in per.tolist())
    assert 10 <= len(bounds) - 1 <= 40


def test_parallel_scatter_has_the_serial_bits():
    nodes, conn, *_ = _cantilever()
    nptr, nind = cc._node_pattern(conn, len(nodes))
    ke = cc._element_matrices(nodes, conn, cc._elastic(2000.0, 0.35))
    a = np.zeros(9 * len(nind))
    b = np.zeros(9 * len(nind))
    cc._scatter(np.ascontiguousarray(conn, np.int64), ke, nptr, nind, a)
    tetsolve.scatter(conn, ke, nptr, nind, b)
    assert np.array_equal(a, b)


def _direct(nodes, conn, fixed, loads):
    k = cc._assemble(nodes, conn, 2000.0, 0.35)
    free = np.nonzero(~fixed)[0]
    kff = k[free][:, free].tocsc()
    out = []
    for _n, f, _ in loads:
        u = np.zeros(3 * len(nodes))
        u[free] = spla.spsolve(kff, f.ravel()[free])
        out.append(u.reshape(-1, 3))
    return out


@pytest.mark.parametrize("coarse", ["factor", "amg"])
def test_every_case_in_one_loop_matches_a_direct_solve(monkeypatch, coarse):
    if coarse == "amg":
        monkeypatch.setenv("LOADPATH_TET_COARSE", "amg")
    nodes, conn, edges, nv, fixed, loads = _cantilever()
    us = cc._solve(nodes, conn, 2000.0, 0.35, fixed, None, loads, nv, edges)
    ref = _direct(nodes, conn, fixed, loads)
    for u, r in zip(us, ref):
        assert np.linalg.norm(u - r) <= 1e-6 * np.linalg.norm(r)
        assert np.all(u[fixed.reshape(-1, 3)] == 0.0)
    # each case's answer is its own CG's, whatever runs beside it
    alone = cc._solve(nodes, conn, 2000.0, 0.35, fixed, None, loads[1:], nv, edges)[0]
    assert np.linalg.norm(alone - us[1]) <= 1e-12 * np.linalg.norm(us[1])


def test_a_solve_repeats_to_the_bit():
    nodes, conn, edges, nv, fixed, loads = _cantilever()
    a = cc._solve(nodes, conn, 2000.0, 0.35, fixed, None, loads, nv, edges)
    b = cc._solve(nodes, conn, 2000.0, 0.35, fixed, None, loads, nv, edges)
    assert all(np.array_equal(x, y) for x, y in zip(a, b))


def test_pins_and_zero_loads():
    # a pin's springs land on the renumbered nodes; a case with no load is zero
    nodes, conn, edges, nv, fixed, loads = _cantilever()
    tip = loads[0][2]
    springs = [(tip, np.repeat(np.eye(3)[None] * 50.0, len(tip), axis=0))]
    zero = ("none", np.zeros_like(loads[0][1]), tip)
    us = cc._solve(nodes, conn, 2000.0, 0.35, fixed, springs, [loads[0], zero], nv, edges)
    k = cc._assemble(nodes, conn, 2000.0, 0.35, extra=springs)
    free = np.nonzero(~fixed)[0]
    ref = np.zeros(3 * len(nodes))
    ref[free] = spla.spsolve(k[free][:, free].tocsc(), loads[0][1].ravel()[free])
    assert np.linalg.norm(us[0].ravel() - ref) <= 1e-6 * np.linalg.norm(ref)
    assert not np.any(us[1])


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-q"]))
