"""pools: thread pools that stay safe beside numba's parallel kernels.

numba's workqueue threading layer (pip numba on macOS without libomp) aborts
the process when two Python threads enter parallel kernels at once; the
load-case solves, verify's submodels and field evaluation then run serially.
"""
import os
import subprocess
import sys
import textwrap
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from loadpath import pools


def test_serial_executor_acts_like_a_pool():
    with pools.SerialExecutor() as ex:
        assert list(ex.map(lambda a, b: a * b, [1, 2, 3], [4, 5, 6])) == [4, 10, 18]
        f = ex.submit(lambda: 1 / 0)
        with pytest.raises(ZeroDivisionError):
            f.result()
        assert ex.submit(sum, [1, 2]).result() == 3
    with pytest.raises(RuntimeError):
        ex.submit(sum, [1])


def test_one_thread_means_serial(monkeypatch):
    monkeypatch.setenv("LOADPATH_THREADS", "1")
    assert isinstance(pools.pool(4), pools.SerialExecutor)
    monkeypatch.delenv("LOADPATH_THREADS")
    assert isinstance(pools.pool(1), pools.SerialExecutor)
    want = ThreadPoolExecutor if pools.threadsafe() else pools.SerialExecutor
    with pools.pool(4) as ex:
        assert isinstance(ex, want)


_WORKQUEUE = textwrap.dedent("""
    import numpy as np, numba as nb
    from loadpath import pools, narrowband
    from loadpath.spec import study_from_dict
    from loadpath.domain import build_domain
    from loadpath.fem import Solver, Stiffness

    s = study_from_dict({
        "designSpace": {"type": "box", "min": [0, 0, 0], "max": [60, 10, 10]},
        "supports": [{"shape": {"type": "box", "min": [-1, -1, -1], "max": [0.01, 11, 11]}}],
        "loadCases": [{"name": "tip", "loads": [
            {"shape": {"type": "box", "min": [59.99, -1, -1], "max": [61, 11, 11]}, "force": [0, 0, -10]}]}],
        "material": {"E": 2000, "nu": 0.3, "strength": 50, "density": 1.0},
        "goal": {"volumeFraction": 0.5}, "resolution": {"pitch": 2.0}})
    d = build_domain(s)
    k = Stiffness(d.mesh, 0.3, d.fixed).assemble(np.full(d.n_elem, 2000.0))
    f = d.loads[0]
    sv = Solver(d.mesh, rtol=1e-8)
    us = sv.solve_many(k, [f, -f, 2 * f, 0.5 * f])
    one = Solver(d.mesh, rtol=1e-8).solve(k, f)
    assert np.allclose(us[0], one, rtol=1e-6, atol=1e-9)
    assert np.allclose(us[1], -us[0], rtol=1e-6, atol=1e-9)

    @nb.njit(parallel=True)
    def _twice(x):
        out = np.empty(len(x))
        for i in nb.prange(len(x)):
            out[i] = 2.0 * x[i, 0]
        return out

    P = np.random.default_rng(0).random((2_000_000, 3))
    got = narrowband.peval(_twice, P, chunk=100_000)
    assert np.array_equal(got, 2.0 * P[:, 0])
    assert pools.threading_layer() == "workqueue", pools.threading_layer()
    assert isinstance(pools.pool(4), pools.SerialExecutor)
    print("ok")
""")


def test_workqueue_runs_load_cases_without_aborting():
    """Under workqueue the threaded paths ran two parallel kernels at once
    and numba aborted the process (SIGABRT, no traceback)."""
    pytest.importorskip("numba")
    env = dict(os.environ, NUMBA_THREADING_LAYER="workqueue", PYTHONPATH=str(ROOT))
    env.pop("LOADPATH_THREADS", None)
    r = subprocess.run([sys.executable, "-c", _WORKQUEUE], env=env, capture_output=True, text=True, timeout=600, check=False)
    assert r.returncode == 0, (r.returncode, r.stdout[-2000:], r.stderr[-4000:])
    assert "ok" in r.stdout
