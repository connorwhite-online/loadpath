"""
Tests for the benchmark suite (loadpath/benchmarks): the reference ports, the
shared evaluator and the problem set-up agree with each other and with
loadpath, on problems small enough to run in seconds.

Run: `python3 tests/test_benchmarks.py`.
"""
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from loadpath.benchmarks.evaluate import Evaluator, binarize
from loadpath.benchmarks.problems import PROBLEMS, bridge_3d, cantilever_3d, lbracket_2d, mbb_2d
from loadpath.benchmarks.reference import optimize_reference
from loadpath.benchmarks.run import design_volfrac, domain_for
from loadpath.fem import Stiffness
import scipy.sparse.linalg as spla


def test_reference_ports_meet_volume_and_improve():
    for p, solver in ((mbb_2d(30, 10), "top88-ft1"), (mbb_2d(30, 10), "top88-ft2"),
                      (cantilever_3d(12, 2, 4), "top3d")):
        ev = Evaluator(p)
        c0 = ev.compliance(np.full(p.dims, p.volfrac))
        r = optimize_reference(p, solver)
        assert abs(r["x"].mean() - p.volfrac) < 2e-3, (solver, r["x"].mean())
        assert ev.compliance(r["x"]) < 0.8 * c0, solver
        # the port's own last compliance is the evaluator's, one OC step behind
        assert abs(r["compliance_reported"] / ev.compliance(r["x"]) - 1) < 0.05


def test_evaluator_matches_loadpath_fe_in_3d():
    """Same brick element, same supports and loads: the evaluator and
    loadpath's assembled solve give one compliance for one design."""
    p = cantilever_3d(10, 2, 4)
    ev = Evaluator(p)
    rng = np.random.default_rng(1)
    g = rng.uniform(0.2, 1.0, p.dims)
    dom, _ = domain_for(p)
    x = g[tuple(dom.mesh.elem_ijk.T)]
    e = 1e-9 + x ** 3 * (1 - 1e-9)
    k = Stiffness(dom.mesh, 0.3, dom.fixed).assemble(e)
    u = spla.spsolve(k.tocsc(), dom.loads[0])
    assert abs(dom.loads[0] @ u / ev.compliance(g) - 1) < 1e-8


def test_problem_grids_match_loadpath_domains():
    for p in PROBLEMS.values():
        q = p.full or p
        dom, _ = domain_for(q)
        assert tuple(dom.dims) == tuple(q.dims)
        assert abs(sum(v * len(n) for n, _, v in q.loads) - dom.loads[0].sum()) < 1e-9
        assert 0 < design_volfrac(q, dom) <= 1


def test_large_loaded_face_is_pinned():
    """A load spread over a whole face is pinned solid whatever share of the
    part it is (the bridge deck), or its nodes can end up on void."""
    p = bridge_3d()
    dom, _ = domain_for(p)
    top = dom.mesh.elem_ijk[:, 2] == p.dims[2] - 1
    assert dom.preserve[top].all()


def test_binarize_keeps_densest_cells():
    x = np.array([0.1, 0.9, 0.5, 0.7])
    act = np.ones(4, bool)
    assert binarize(x, act, 0.5).tolist() == [0, 1, 0, 1]
    assert binarize(x, act, None).tolist() == [0, 1, 0, 1]


def test_von_mises_of_a_tension_bar():
    """A solid strip pulled at its end: von Mises is the axial stress."""
    import dataclasses
    p = lbracket_2d(20)
    from loadpath.benchmarks.problems import _nodes
    q = dataclasses.replace(p, void=None, fixed=[(_nodes(0, 0, range(21)), "xz")],
                            loads=[(_nodes(20, 0, range(21)), "x", 1.0 / 21)], stress=True)
    ev = Evaluator(q)
    vm = ev.von_mises(np.ones(q.dims))
    mid = vm.reshape(20, 20)[8:12, 8:12]
    assert np.allclose(mid, 1.0 / 20, rtol=0.05), mid


TESTS = [test_reference_ports_meet_volume_and_improve, test_evaluator_matches_loadpath_fe_in_3d,
         test_problem_grids_match_loadpath_domains, test_large_loaded_face_is_pinned,
         test_binarize_keeps_densest_cells, test_von_mises_of_a_tension_bar]


def main():
    failures = 0
    for t in TESTS:
        t0 = time.time()
        try:
            t()
            print(f"PASS {t.__name__} ({time.time() - t0:.1f}s)")
        except Exception:  # noqa: BLE001
            failures += 1
            print(f"FAIL {t.__name__}")
            traceback.print_exc()
    if failures:
        print(f"\n{failures}/{len(TESTS)} FAILED")
        sys.exit(1)
    print(f"\nall {len(TESTS)} passed")


if __name__ == "__main__":
    main()
