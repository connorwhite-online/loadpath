"""
The rod solver against closed forms (loadpath/rods.py): finite-difference
checks of the gradient and Hessian, the cantilever elastica, pure bending to
an arc, a twisted bar, Euler buckling, the twist that writhes a straight
rod, a ribbon's two stiffnesses, and a rod lying on a plane.

Run: `python tests/test_rods.py`. Under ten seconds.
"""
import sys
import time
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from loadpath.rods import Rod, SdfField, GridField, frame_quat, parallel_frames, qexp, qmul, quat_matrix


def straight(L=100.0, n=60, axis=(1, 0, 0), d1=(0, 0, 1), **kw):
    a = np.asarray(axis, float)
    x = np.linspace(0, L, n + 1)[:, None] * a[None, :]
    return Rod(x, parallel_frames(x, d1), **kw)


def _dense(rod, ab):
    N = rod.N
    H = np.zeros((N, N))
    for k in range(rod.u + 1):
        j = np.arange(k, N)
        H[j - k, j] = ab[rod.u - k, j]
        H[j, j - k] = ab[rod.u - k, j]
    return H


def test_gradient_and_hessian_match_finite_differences():
    rng = np.random.default_rng(0)
    n = 6
    s = np.linspace(0, 30, n + 1)
    x = np.column_stack([s, 3 * np.sin(s / 9), 2 * np.cos(s / 7)]) + 0.2 * rng.normal(size=(n + 1, 3))
    q = qmul(parallel_frames(x, twist=0.7), qexp(0.15 * rng.normal(size=(n, 3))))
    rod = Rod(x, q, bend=(3.0, 11.0), twist=5.0, stretch=400.0, lengths=5.2, weight=[0, 0, -0.01], radius=0.5,
              offsets=[[-1.5, 0, 0], [0, 0, 0], [1.5, 0, 0]])
    rod.K0[:] = 0.01 * rng.normal(size=rod.K0.shape)
    rod.obstacles = [SdfField(lambda p: p[:, 2] - 1.0)]
    rod.contact_k = 7.0
    rod.springs = [(3, np.array([1.0, 2, 3]), 0.5)]
    rod.loads = [(5, np.array([0.3, -0.2, 0.1]))]
    E, g, ab = rod.assemble()
    H = _dense(rod, ab)
    x0, q0 = rod.x.copy(), rod.q.copy()

    def en(step):
        rod.x, rod.q = x0, q0
        return rod.energy(*rod._apply(step))
    N = rod.N
    for i in range(N):
        e = np.zeros(N)
        e[i] = 1e-6
        fd = (en(e) - en(-e)) / 2e-6
        assert abs(fd - g[i]) < 1e-5 * (1 + abs(g).max()), (i, fd, g[i])
    h = 1e-4
    for i in range(N):
        for j in range(i, min(N, i + 12)):
            ei, ej = np.zeros(N), np.zeros(N)
            ei[i], ej[j] = h, h
            fd = (en(ei + ej) - en(ei - ej) - en(-ei + ej) + en(-ei - ej)) / (4 * h * h)
            assert abs(fd - H[i, j]) < 1e-3 * (1 + abs(H).max()) * 1e-2, (i, j, fd, H[i, j])


def test_cantilever_elastica():
    # Bisshopp and Drucker (1945); Mattiasson (1981): tip deflection and
    # shortening of a cantilever under an end load P, by a = P L^2 / EI
    table = {1.0: (0.30172, 0.05643), 2.0: (0.49346, 0.16064), 5.0: (0.71379, 0.38763), 10.0: (0.81061, 0.55500)}
    L, EI = 100.0, 50.0
    n = 80
    h = L / n
    Le = L - 0.5 * h          # a clamp holds its edge: the rod bends from the middle of it on
    for a, (w, u) in table.items():
        rod = straight(L, n, bend=EI, twist=EI)
        rod.clamp(0)
        P = a * EI / Le ** 2
        rod.loads = [(rod.n, np.array([0, 0, -P]))]
        st = rod.solve()
        assert st["converged"] and st["stable"], st
        assert abs(-rod.x[-1, 2] / Le - w) < 1e-3, (a, -rod.x[-1, 2] / Le, w)
        assert abs((L - rod.x[-1, 0]) / Le - u) < 1e-3, (a, (L - rod.x[-1, 0]) / Le, u)
        # what the wall supplies: the load, and its moment about the root
        F, M = rod.end_load(0)
        assert np.allclose(F, [0, 0, P], atol=1e-4 * P), F
        assert abs(abs(M[1]) - P * rod.x[-1, 0]) < 2e-3 * P * L, (M, rod.x[-1])


def test_pure_bending_is_an_arc():
    L, EI, n = 60.0, 20.0, 60
    rod = straight(L, n, bend=EI, twist=EI)
    rod.clamp(0)
    e = rod.edge_dofs(n - 1)
    rod.fixed[e] = True                       # the far frame turned, its node free
    phi = 2.0

    def turn(s):
        rod.q[-1] = qmul(frame_quat([1, 0, 0], [0, 0, 1]), qexp([s * phi, 0, 0]))
    assert rod.drive(turn, steps=8)["converged"]
    r = rod.bend_radius()
    assert np.allclose(r, L / phi * (n - 1) / n, rtol=2e-3), (r.min(), r.max(), L / phi)
    _, M = rod.end_load(0)
    assert abs(np.linalg.norm(M) - EI * phi / (L * (n - 1) / n)) < 1e-3 * EI * phi / L


def test_twist_torque():
    L, GJ, n = 80.0, 7.0, 40
    rod = straight(L, n, bend=30.0, twist=GJ)
    rod.clamp(0).clamp(-1)
    a = 0.8

    def turn(s):
        rod.q[-1] = qmul(frame_quat([1, 0, 0], [0, 0, 1]), qexp([0, 0, s * a]))
    assert rod.drive(turn, steps=4)["converged"]
    k = rod.curvature()
    Lt = L * (n - 1) / n
    assert np.allclose(k[:, 2], a / Lt, rtol=1e-3) and np.abs(k[:, :2]).max() < 1e-8
    _, M = rod.end_load(-1)
    assert abs(abs(M[0]) - GJ * a / Lt) < 1e-3 * GJ * a / Lt, M


def test_euler_buckling_load():
    # both ends built in, pushed together by 1% of the length: the rod must
    # leave the straight state by itself (it is a saddle), and carries
    # P = 4 pi^2 EI / L^2 (1 + shortening / 2L) to first order
    L, EI, n = 100.0, 40.0, 100
    rod = straight(L, n, bend=EI, twist=EI)
    rod.clamp(0).clamp(-1)
    d = 0.01 * L
    assert rod.drive(lambda s: rod.set_end(-1, [L - s * d, 0, 0]), steps=2)["converged"]
    assert np.abs(rod.x[:, 1:]).max() > 0.02 * L, "did not buckle"
    F, _ = rod.end_load(-1)
    Le = L * (n - 1) / n      # the clamps hold an edge each: half an edge stiff at each end
    Pcr = 4 * np.pi ** 2 * EI / Le ** 2
    assert abs(abs(F[0]) / Pcr - (1 + 0.5 * d / L)) < 0.01, (abs(F[0]) / Pcr)


def test_twist_writhes_a_straight_rod():
    # Greenhill: a straight rod with built-in ends, twisted, goes helical at
    # a torque of 8.987 EI / L
    L, EI, GJ, n = 100.0, 30.0, 20.0, 80
    Le = L * (n - 1) / n
    a_c = 8.987 * EI / GJ

    def twisted(a):
        rod = straight(L, n, bend=EI, twist=GJ)
        rod.clamp(0).clamp(-1)

        def turn(s):
            rod.q[-1] = qmul(frame_quat([1, 0, 0], [0, 0, 1]), qexp([0, 0, s * a]))
        st = rod.drive(turn, steps=8)
        assert st["converged"], st
        return np.abs(rod.x[:, 1:]).max()
    assert twisted(0.93 * a_c * Le / L) < 1e-6
    assert twisted(1.07 * a_c) > 3e-4 * L


def test_ribbon_bends_easily_one_way_only():
    L, n = 50.0, 40
    w, t, E = 6.0, 0.2, 3000.0
    EI1, EI2 = E * w * t ** 3 / 12, E * t * w ** 3 / 12
    q = 1e-5
    out = []
    for g in ([0, 0, -q], [0, -q, 0]):
        # d1 (the width) along y, so the thickness is along z
        rod = straight(L, n, d1=(0, 1, 0), bend=(EI1, EI2), twist=E / 2.6 * w * t ** 3 / 3, weight=g)
        rod.clamp(0)
        assert rod.solve()["converged"]
        out.append(np.abs(rod.x[-1, 1:]).max())
    Le = L - 0.5 * L / n
    assert abs(out[0] - q * Le ** 4 / (8 * EI1)) < 0.01 * out[0], (out[0], q * Le ** 4 / (8 * EI1))
    assert abs(out[1] - q * Le ** 4 / (8 * EI2)) < 0.01 * out[1], (out[1], q * Le ** 4 / (8 * EI2))


def test_rod_lies_on_a_plane():
    L, EI, n, r = 120.0, 15.0, 80, 0.6
    wt = 2e-3
    for field in (SdfField(lambda p: p[:, 2]),
                  GridField.sample(lambda p: p[:, 2], [-5, -5, -3], [125, 5, 8], 0.5, pad=0)):
        rod = straight(L, n, bend=EI, twist=EI, weight=[0, 0, -wt], radius=r)
        rod.x[:, 2] += 4.0
        rod.clamp(0).clamp(-1)
        rod.obstacles = [field]
        # slack: the ends 2 mm closer, so its weight lays the middle on the plane
        st = rod.drive(lambda s: rod.set_end(-1, [L - 2.0 * s, 0, 4.0]), steps=4)
        assert st["converged"], st
        c = rod.contact_state()
        assert c["penetration"].max() < 0.05, c["penetration"].max()
        mid = slice(n // 3, 2 * n // 3)
        assert np.allclose(rod.x[mid, 2], r, atol=0.05), rod.x[mid, 2]
        F0, _ = rod.end_load(0)
        F1, _ = rod.end_load(-1)
        # (the held end nodes' own half edges of weight go straight into the clamps)
        assert abs(F0[2] + F1[2] + c["force"].sum() - wt * (L - L / n)) < 1e-3 * wt * L, (F0, F1, c["force"].sum())


def test_compiled_kernels_match_the_reference():
    """The numba assembly (rod.fast) against the numpy one, on a bent,
    twisted ribbon with weight, pressed into a sampled wall."""
    rng = np.random.default_rng(3)
    n = 30
    rod = straight(60.0, n, bend=(4.0, 90.0), twist=6.0, weight=[0, 0, -1e-3], radius=0.4,
                   offsets=np.array([[-1.5, 0, 0], [0, 0, 0], [1.5, 0, 0.0]]))
    rod.x += 0.4 * rng.standard_normal(rod.x.shape)
    rod.q += 0.1 * rng.standard_normal(rod.q.shape)
    rod.q /= np.linalg.norm(rod.q, axis=1)[:, None]
    rod.clamp(0)
    grid = GridField.sample(lambda p: p[:, 2] + 0.2 * np.sin(0.3 * p[:, 0]), [-5, -6, -4], [66, 6, 6], 0.5, pad=0)
    rod.obstacles = [grid]
    rod.fast = grid.fast = True
    E1, g1, H1 = rod.assemble()
    rod.fast = grid.fast = False
    E0, g0, H0 = rod.assemble()
    assert abs(E1 - E0) < 1e-9 * abs(E0), (E1, E0)
    assert np.abs(g1 - g0).max() < 1e-9 * np.abs(g0).max()
    assert np.abs(H1 - H0).max() < 1e-9 * np.abs(H0).max()


def test_frames():
    q = frame_quat([0, 1, 0], [0, 0, 1])
    R = quat_matrix(q[None])[0]
    assert np.allclose(R[:, 2], [0, 1, 0]) and np.allclose(R[:, 0], [0, 0, 1]) and np.isclose(np.linalg.det(R), 1)


if __name__ == "__main__":
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            t0 = time.time()
            try:
                fn()
                print(f"ok   {name} ({time.time() - t0:.1f} s)")
            except Exception:
                fails += 1
                print(f"FAIL {name}")
                traceback.print_exc()
    sys.exit(1 if fails else 0)
