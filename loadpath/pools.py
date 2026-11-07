"""Thread pools that are safe beside numba's parallel kernels.

loadpath runs some independent work on Python threads (a study's load cases
in fem.Solver.solve_many and the cross-check, verify's submodel boxes and its
buckling check, field evaluation in narrowband.peval), and that work calls
numba parallel=True kernels (amgfast's smoother, femops, mgops, meshops).

numba picks a threading layer the first time a parallel kernel runs: tbb,
then omp, then workqueue. workqueue is the fallback, and it is not thread
safe: two Python threads inside parallel kernels at once abort the process
(SIGABRT, "Concurrent access has been detected"), with no traceback to
catch. pip numba on macOS arm64 lands on workqueue unless libomp is installed
(`brew install libomp`; omppool looks in /opt/homebrew/lib) or tbb is.

`pool(n)` hands out a ThreadPoolExecutor when threads are safe and useful,
and otherwise an executor that runs each task at once on the calling thread:
same interface (submit, map, shutdown, `with`), same results, in order.
Serial when:
- numba's threading layer is workqueue,
- LOADPATH_THREADS=1 (one thread means one thread, not one numba thread per
  Python thread), or
- n <= 1.
"""
from __future__ import annotations

import os
import threading
from concurrent.futures import Executor, Future, ThreadPoolExecutor

__all__ = ["SerialExecutor", "pool", "threading_layer", "threadsafe"]

_LAYER: str | None = None
_LOCK = threading.Lock()


def threading_layer() -> str | None:
    """numba's active threading layer ('tbb', 'omp' or 'workqueue'), or None
    without numba. Starts numba's thread pool if no parallel kernel has run
    yet (numba.threading_layer() only answers after that)."""
    global _LAYER
    if _LAYER is not None:
        return _LAYER or None
    with _LOCK:
        if _LAYER is not None:
            return _LAYER or None
        try:
            import numba
        except ImportError:
            _LAYER = ""
            return None
        try:
            _LAYER = numba.threading_layer()
        except ValueError:   # no parallel kernel has run yet
            try:
                from numba.np.ufunc.parallel import _launch_threads

                _launch_threads()
                _LAYER = numba.threading_layer()
            except Exception:  # noqa: BLE001 - can't tell: assume the unsafe fallback
                _LAYER = "workqueue"
        return _LAYER or None


def threadsafe() -> bool:
    """True when several Python threads may run numba parallel kernels at once."""
    return threading_layer() != "workqueue"


def _one_thread() -> bool:
    env = os.environ.get("LOADPATH_THREADS", "").strip()
    try:
        return bool(env) and int(env) <= 1
    except ValueError:
        return False


class SerialExecutor(Executor):
    """An Executor that runs each task when it is submitted, on the calling
    thread. Exceptions land in the future, as a pool's would."""

    def __init__(self, *_, **__):
        self._shutdown = False

    def submit(self, fn, /, *args, **kwargs):
        if self._shutdown:
            raise RuntimeError("cannot schedule new futures after shutdown")
        f = Future()
        try:
            f.set_result(fn(*args, **kwargs))
        except BaseException as e:  # noqa: BLE001 - delivered through the future
            f.set_exception(e)
        return f

    def map(self, fn, *iterables, timeout=None, chunksize=1, **_):
        # lazily, in order, raising where a pool's map would
        for args in zip(*iterables):
            yield fn(*args)

    def shutdown(self, wait=True, *, cancel_futures=False):
        self._shutdown = True


def pool(n: int, thread_name_prefix: str = "") -> Executor:
    """A ThreadPoolExecutor of n threads for tasks that call numba parallel
    kernels, or a SerialExecutor when threads would abort (workqueue), when
    LOADPATH_THREADS=1, or when n <= 1."""
    if n <= 1 or _one_thread() or not threadsafe():
        return SerialExecutor()
    return ThreadPoolExecutor(n, thread_name_prefix=thread_name_prefix)
