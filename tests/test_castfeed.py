"""
The pour as the optimizer sees it (loadpath/castfeed.py): the freezing-order
field against the full solidification solve, the feeding measure against
the full pour on two lumps joined by a neck, its gradient against finite
differences, and the study field that turns it on.

    python tests/test_castfeed.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from loadpath import casting as C                      # noqa: E402
from loadpath import castfeed as F                     # noqa: E402

SAND = C.setup("sand", "a356")


def test_widest_path():
    # a row of cells fed from its left end: everything past the 2 is held to 2, and the 2 is why
    theta = np.array([5.0, 2.0, 4.0, 3.0, 6.0]).reshape(5, 1, 1)
    feed = np.zeros((5, 1, 1), bool)
    feed[0] = True
    lvl, b = F.widest_path(theta, feed)
    assert lvl.ravel().tolist() == [5.0, 2.0, 2.0, 2.0, 2.0]
    assert b.ravel().tolist() == [-1, -1, 1, 1, 1]
    # two ways round a block: the better one counts
    theta = np.array([[9.0, 1.0, 7.0], [9.0, 5.0, 7.0]]).reshape(2, 3, 1)
    feed = np.zeros((2, 3, 1), bool)
    feed[:, 0] = True
    lvl, b = F.widest_path(theta, feed)
    assert lvl[0, 2, 0] == 5.0 and lvl[1, 2, 0] == 5.0 and b[0, 2, 0] == np.ravel_multi_index((1, 1, 0), (2, 3, 1))


def test_order_field_is_the_modulus():
    # heat out equals heat in: the mold faces carry away exactly what the cells give up
    g = np.zeros((14, 14, 14))
    g[2:12, 2:12, 2:12] = 1.0
    ft = F.FeedTerm(g.shape, 2.0, SAND)
    m = ft.order(g)
    loss = ft._loss(g.ravel()).reshape(g.shape)
    assert abs((loss * m).sum() / (g.sum() * 2.0) - 1) < 1e-3
    # a lump's modulus is volume / surface: 10 cells of 2 mm a side, 20 / 6 mm, and the middle is last
    surface_mean = (loss * m).sum() / loss.sum()
    assert abs(surface_mean / (20.0 / 6.0) - 1) < 1e-3, surface_mean
    assert m[7, 7, 7] > 0.999 * m.max() and m[2, 2, 2] < m[7, 7, 7]
    # a plate: half its thickness
    g = np.zeros((40, 40, 8))
    g[:, :, 2:6] = 1.0
    m = F.FeedTerm(g.shape, 2.0, SAND, outside=1.0).order(g)
    assert abs(m[20, 20, 3] / 4.0 - 1) < 0.02, m[20, 20, 3]


def _bossed():
    g = np.zeros((64, 34, 22), bool)
    g[2:62, 2:32, 2:6] = True                          # 120 x 60 x 8 plate at 2 mm
    ii, jj = np.meshgrid(np.arange(64), np.arange(34), indexing="ij")
    g[:, :, 6:20] |= (((ii - 16.5) ** 2 + (jj - 16.5) ** 2) <= 9.0 ** 2)[:, :, None]
    return g


def test_order_against_the_full_solve():
    g = _bossed()
    oc = F.order_check(g, 2.0, SAND)
    assert oc["rank"] > 0.9, oc["rank"]                # the order of freezing, cell by cell
    assert oc["apart"] < 16.0, oc                      # and both put the last metal in the boss


def _two_lumps(p, neck):
    n = lambda mm: int(round(mm / p))                  # noqa: E731
    g = np.zeros((n(110), n(30), n(30)), bool)
    g[:n(30)] = True
    g[n(80):] = True
    g[n(30):n(80), :, :n(neck)] = True
    g = np.pad(g, 2)
    top = int(np.flatnonzero(g[2 + n(15), 2 + n(15), :]).max())
    feed = np.zeros(g.shape, bool)
    feed[2 + n(6):2 + n(24), 2 + n(6):2 + n(24), top] = True
    return g, feed, [2 + n(15), 2 + n(15), top]


def test_measure_against_the_full_pour():
    # two 30 mm cubes joined by a plate, a riser on one: the far cube is fed through the plate.
    # The measure and the full pour agree on which necks leave a cavity and on its size within 2.5x
    p = 2.0
    got = {}
    for neck in (4, 6, 10, 14, 30):
        g, feed, base = _two_lumps(p, neck)
        ft = F.FeedTerm(g.shape, p, SAND, feed)
        got[neck] = ft.measure(g.astype(float), grad=False)[0]
    assert got[4] > got[6] > got[10] > got[14], got
    assert got[6] > ft.limit and got[14] < ft.limit and got[30] < ft.limit, got
    for neck in (6, 14):
        g, feed, base = _two_lumps(p, neck)
        pour = F.FeedTerm(g.shape, p, SAND, feed).pour(g)["cavityShare"]
        if neck == 6:
            assert pour > ft.limit and 0.4 < got[neck] / pour < 2.5, (got[neck], pour)
        else:
            assert pour < ft.limit, pour


def test_gradient():
    g, feed, _ = _two_lumps(3.0, 9)
    rng = np.random.default_rng(1)
    rho = np.clip(g * 0.75 + 0.2 * rng.random(g.shape), 0, 1)
    ft = F.FeedTerm(g.shape, 3.0, SAND, feed)
    ft.measure(rho)
    bi = ft._biot(rho.ravel())
    ft._biot = lambda r: bi                            # the blend length follows the design from outside
    v, grad, _ = ft.measure(rho, hold=True)
    assert v > 0
    cells = [tuple(c) for c in np.argwhere(g)[:: max(1, int(g.sum()) // 9)]][:9]
    for c in cells:
        e = 1e-4
        hi, lo = rho.copy(), rho.copy()
        hi[c] += e
        lo[c] -= e
        ft.last = None
        v2 = ft.measure(hi, grad=False, hold=True)[0]
        ft.last = None
        v1 = ft.measure(lo, grad=False, hold=True)[0]
        fd = (v2 - v1) / (2 * e)
        assert abs(grad[c] - fd) < 0.03 * abs(fd) + 1e-3 * np.abs(grad).max(), (c, grad[c], fd)
    # on the part itself: there is metal to add on the neck that lowers the cavity (at the
    # bottleneck; further along, new metal is first of all more metal to feed)
    ft = F.FeedTerm(g.shape, 3.0, SAND, feed)
    v, grad, _ = ft.measure(g.astype(float))
    n = lambda mm: int(round(mm / 3.0))                # noqa: E731
    above = grad[2 + n(30):2 + n(80), 2:-2, 2 + n(9)]
    assert v > ft.limit and above.min() < 0, (v, above.min())


def test_study_field():
    from loadpath.domain import build_domain
    from loadpath.schema import lint
    from loadpath.spec import study_from_dict

    d = {"designSpace": [{"type": "box", "min": [0, 0, 0], "max": [60, 20, 12]}],
         "preserve": [{"id": "pad", "shape": {"type": "box", "min": [0, 0, 0], "max": [6, 20, 12]}},
                      {"id": "tip", "shape": {"type": "box", "min": [54, 0, 0], "max": [60, 20, 12]}}],
         "supports": [{"region": "pad"}],
         "loads": [{"region": "tip", "force": [0, -100, 0]}],
         "material": "a356-t6", "goal": {"volumeFraction": 0.4}, "resolution": {"pitch": 3.0},
         "casting": {"draw": "+z", "process": "sand", "alloy": "a356",
                     "feed": {"type": "box", "min": [0, 5, 9], "max": [6, 15, 12]}}}
    assert not [x for x in lint(d) if x["level"] == "error"], lint(d)
    dom = build_domain(study_from_dict(d))
    ft = dom.feeding
    assert ft is not None and ft.feed.any() and ft.feed.sum() < 0.1 * ft.feed.size
    d["casting"]["feed"] = "auto"
    assert build_domain(study_from_dict(d)).feeding.feed is None
    del d["casting"]["feed"]
    assert build_domain(study_from_dict(d)).feeding is None           # the pour is then only reported
    d["casting"] = {"draw": "+z", "feed": "auto"}
    try:
        study_from_dict(d)
    except ValueError as exc:
        assert "process" in str(exc)
    else:
        raise AssertionError("feed without a process")


def test_optimizer_feeds_the_far_lump():
    # two lumps and a lean design space between them, a riser on one, the other loaded in bending.
    # Stiffness alone necks the flanges down where they meet the loaded lump (the moment is small
    # there) and the pour leaves a cavity in it; with casting.feed the flanges reach it at a
    # thickness that feeds it. Judged by the full pour of each delivered design.
    from loadpath.domain import build_domain
    from loadpath.simp import SimpOptions, filter_radius, optimize
    from loadpath.spec import study_from_dict
    from loadpath.study import _making

    base = {"designSpace": [{"type": "box", "min": [0, 0, 0], "max": [110, 30, 30]}],
            "preserve": [{"id": "a", "shape": {"type": "box", "min": [0, 0, 0], "max": [30, 30, 30]}},
                         {"id": "b", "shape": {"type": "box", "min": [80, 0, 0], "max": [110, 30, 30]}}],
            "supports": [{"region": "a"}],
            "loadCases": [{"name": "load", "loads": [{"region": "b", "force": [0, 0, -800]}]}],
            "material": "a356-t6", "goal": {"volumeFraction": 0.25}, "minMember": 4.0,
            "resolution": {"pitch": 3.0}, "symmetry": ["y"],
            "casting": {"draw": "+y", "sides": 2, "process": "sand", "alloy": "a356"}}
    feed = {"type": "box", "min": [6, 6, 27], "max": [24, 24, 30]}
    out = {}
    for name in ("plain", "fed"):
        d = {**base, "casting": dict(base["casting"], **({"feed": feed} if name == "fed" else {}))}
        st = study_from_dict(d)
        dom = build_domain(st)
        r = optimize(dom, st.elastic(design=True),
                     SimpOptions(volume_fraction=0.25, filter_radius=filter_radius(dom, st.min_member),
                                 max_iterations=st.max_iterations, **_making(st)))
        judge = build_domain(study_from_dict({**base, "casting": dict(base["casting"], feed=feed)})).feeding
        out[name] = (judge.pour(dom.full_grid(r.xphys))["cavityShare"], r.compliance, r.history[-1])
    assert "cavityShare" in out["fed"][2] and "cavityShare" not in out["plain"][2]
    assert out["plain"][0] > 2.0 * judge.limit, out["plain"][0]                 # stiffness alone: unsound
    # (on this coarse grid the delivered design is not yet sound; at 2 mm it is: docs/CASTING.md)
    assert out["fed"][0] < 0.7 * out["plain"][0], (out["fed"][0], out["plain"][0])
    assert out["fed"][1] < 1.6 * out["plain"][1]                                # and what it cost in stiffness


TESTS = [test_widest_path, test_order_field_is_the_modulus, test_order_against_the_full_solve,
         test_measure_against_the_full_pour, test_gradient, test_study_field,
         test_optimizer_feeds_the_far_lump]

if __name__ == "__main__":
    import time

    for t in TESTS:
        t0 = time.time()
        t()
        print(f"PASS {t.__name__} ({time.time() - t0:.1f}s)")
    print(f"\nall {len(TESTS)} passed")
