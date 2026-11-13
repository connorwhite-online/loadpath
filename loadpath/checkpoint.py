"""Checkpoints for long runs, so a run killed partway (a container reset, an
out-of-memory kill, a closed laptop) continues where it stopped instead of
starting over: `loadpath run --resume <out>`.

What a run is, for resuming: a sequence of optimizer calls (simp._optimize,
constrained.optimize_constrained) whose inputs follow from the study and
from the results of the calls before them, with meshing and checks in
between that are pure functions of those results. So two records make a run
resumable:

- every finished optimizer call: its result, and the state it left on the
  study's long-lived objects (the thermal model's converged convection
  coefficients, an eigensolver's warm-start vectors), stored under the
  call's key. A resumed run that makes the same call gets the same result
  back at no cost, and the objects as they were after it.
- the call in progress: every few minutes, the loop's whole state between
  two iterations (densities, projection sharpness and the iterations held
  at it, MMA's asymptotes and past iterates, adaptive penalty weights, the
  last displacements the solver warm-starts from, the history) and the
  long-lived objects' state with it. The resumed call starts at the next
  iteration with everything the uninterrupted one had.

A call's key is a hash of what decides its outcome: the design problem
(grid, preserved and pinned cells, supports, loads), the material's
Poisson ratio and every optimizer option (a warm-start density included),
plus how many times this process has made that same call before. Inputs
that differ from the interrupted run's (a changed study, a search that
went another way) give another key, and that call runs fresh: a resumed run
never takes a result for a problem it was not computed for.

The search between calls is replayed, not stored: its decisions follow from
the results, and the replayed results are the stored ones bit for bit. The
exceptions are decisions on the clock (--time-limit, a probe given a share
of the elapsed time): replayed calls take no time, so the resumed run has
more of it left than the interrupted one had.

Checkpoints are on for `loadpath run` (under <out>/checkpoint) and off for
the library unless LOADPATH_CHECKPOINT names a folder. The loop state is
written every EVERY (8) iterations, by count so that two runs of a study
checkpoint at the same iterations, and skipped at an iteration that comes
sooner than LOADPATH_CHECKPOINT_SECONDS (default 0) or than 20 times the last
write took after the last write, so writing costs at most 5% of the run. At
each of those iterations the solver's caches start over, as a resumed run's
do, which is what makes the resumed run match the uninterrupted one to the
bit; the work a reset can lose is up to 8 iterations.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import pickle
import time

import numpy as np

ENV = "LOADPATH_CHECKPOINT"
ENV_SECONDS = "LOADPATH_CHECKPOINT_SECONDS"
PERIOD = 0.0
# checkpoint iterations: every EVERY-th. The assembled solver rebuilds its
# preconditioner every fem.Solver.REFRESH_AGE (8) iterations anyway and the
# reduced system every simp._Analysis.REUSE (6), so starting the caches over
# here costs about nothing, and a reset loses at most 8 iterations.
EVERY = 8
# the loop state is written no more often than this many times what the last
# write took: writing then costs at most 1/COST_RATIO of the run
COST_RATIO = 20.0
FORMAT = 1

_seen: dict[str, int] = {}          # calls made by this process, per input hash
_stats = {"restored": 0, "replayed": 0, "written": 0}


def folder() -> str | None:
    d = os.environ.get(ENV)
    return os.path.abspath(d) if d else None


def stats() -> dict:
    return dict(_stats)


# --- fingerprints -------------------------------------------------------------

_SCALARS = (bool, int, float, complex, str, bytes, type(None), np.generic)


def _feed(h, obj, depth: int, seen: set) -> None:
    """Hash what decides an optimizer call's outcome: arrays by content,
    containers and dataclasses recursively, other objects by their plain
    attributes (depth-limited). Callables and modules by name only."""
    if isinstance(obj, _SCALARS):
        h.update(repr((type(obj).__name__, obj)).encode())
        return
    if isinstance(obj, np.ndarray):
        a = np.ascontiguousarray(obj)
        h.update(f"nd{a.dtype.str}{a.shape}".encode())
        h.update(a.view(np.uint8).reshape(-1) if a.dtype != object else repr(a.tolist()).encode())
        return
    if id(obj) in seen:
        h.update(b"<cycle>")
        return
    seen = seen | {id(obj)}
    if hasattr(obj, "tocsr") and hasattr(obj, "nnz") and depth > 0:      # scipy sparse
        m = obj.tocsr()
        h.update(f"sp{m.shape}".encode())
        for a in (m.data, m.indices, m.indptr):
            _feed(h, a, depth - 1, seen)
        return
    if isinstance(obj, (list, tuple)):
        h.update(f"{type(obj).__name__}{len(obj)}".encode())
        for v in obj:
            _feed(h, v, depth - 1, seen)
        return
    if isinstance(obj, dict):
        h.update(f"dict{len(obj)}".encode())
        for k in sorted(obj, key=repr):
            h.update(repr(k).encode())
            _feed(h, obj[k], depth - 1, seen)
        return
    if callable(obj) or isinstance(obj, type) or depth <= 0:
        h.update(f"<{type(obj).__module__}.{type(obj).__qualname__}>".encode())
        return
    h.update(f"{type(obj).__module__}.{type(obj).__qualname__}".encode())
    if dataclasses.is_dataclass(obj):
        items = [(f.name, getattr(obj, f.name, None)) for f in dataclasses.fields(obj)]
    else:
        items = sorted(getattr(obj, "__dict__", {}).items())
    for k, v in items:
        if k.startswith("_"):
            continue            # caches and scratch, not inputs
        h.update(k.encode())
        _feed(h, v, depth - 1, seen)


def _is_helper(v) -> bool:
    """A long-lived object hung on the domain (the thermal model, the
    vibration model, a warp or blockage term): state the optimizer moves, not
    an input. Its type stands in the key; its state is saved and restored."""
    if isinstance(v, (_SCALARS, np.ndarray, list, tuple, dict)) or callable(v):
        return False
    if hasattr(v, "tocsr") and hasattr(v, "nnz"):
        return False
    if dataclasses.is_dataclass(v) or type(v).__name__.endswith("Mesh"):
        return False            # a study, a grid: inputs
    return hasattr(v, "__dict__")


def domain_fingerprint(domain) -> str:
    h = hashlib.blake2b(digest_size=20)
    for k, v in sorted(vars(domain).items()):
        h.update(k.encode())
        if _is_helper(v):
            h.update(f"<{type(v).__module__}.{type(v).__qualname__}>".encode())
        else:
            _feed(h, v, 6, set())
    return h.hexdigest()


def call_key(kind: str, domain, *inputs) -> str:
    h = hashlib.blake2b(digest_size=20)
    h.update(f"{kind}:{FORMAT}:".encode())
    h.update(domain_fingerprint(domain).encode())
    for v in inputs:
        _feed(h, v, 6, set())
    base = h.hexdigest()
    n = _seen.get(base, 0)
    _seen[base] = n + 1
    return f"{kind}-{base}-{n}"


# --- long-lived objects' state ------------------------------------------------

def domain_helpers(domain) -> dict:
    """The objects on the domain an optimizer call can change (thermal,
    modal, warp, feeding, blockage...)."""
    return {f"domain.{k}": v for k, v in vars(domain).items() if _is_helper(v)}


def _pickled(v) -> bytes | None:
    try:
        return pickle.dumps(v, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception:  # noqa: BLE001 - an unpicklable cache (a factorization) is rebuilt on use
        return None


def helpers_state(helpers: dict, domain=None) -> dict:
    """Each object's attributes that pickle. Shared references to the domain
    itself or its mesh are left out (restored they would become copies), as
    are attributes that do not pickle (solver factorizations, kept as built
    and rebuilt where dropped)."""
    shared = set()
    if domain is not None:
        shared = {id(domain)} | {id(v) for v in vars(domain).values()}
    out = {}
    for name, obj in helpers.items():
        if obj is None or not hasattr(obj, "__dict__"):
            continue
        st = {}
        for k, v in vars(obj).items():
            if id(v) in shared or type(v).__name__ in ("Mesh", "Domain"):
                continue
            b = _pickled(v)
            if b is not None:
                st[k] = b
        out[name] = st
    return out


def restore_helpers(helpers: dict, state: dict | None) -> None:
    for name, st in (state or {}).items():
        obj = helpers.get(name)
        if obj is None:
            continue
        for k, b in st.items():
            try:
                setattr(obj, k, pickle.loads(b))
            except Exception:  # noqa: BLE001 - a read-only attribute keeps its fresh value
                pass


# --- the solver across a checkpoint -------------------------------------------

_FLAG = (bool, int, float, str, type(None))


def flags(obj) -> dict:
    """An object's settled choices: its plain scalar attributes and those of
    the objects it holds (multigrid's strict hierarchy, once taken, is kept
    from then on). Caches (arrays, factorizations) are not flags."""
    out = {}
    for k, v in vars(obj).items():
        if isinstance(v, _FLAG):
            out[k] = v
        elif hasattr(v, "__dict__") and not isinstance(v, type) and not callable(v):
            sub = {kk: vv for kk, vv in vars(v).items() if isinstance(vv, _FLAG)}
            if sub:
                out[k] = {"__flags__": sub}
    return out


def set_flags(obj, fl: dict | None) -> None:
    for k, v in (fl or {}).items():
        if isinstance(v, dict) and "__flags__" in v:
            sub = getattr(obj, k, None)
            if sub is not None and hasattr(sub, "__dict__"):
                for kk, vv in v["__flags__"].items():
                    setattr(sub, kk, vv)
        elif hasattr(obj, k) and isinstance(getattr(obj, k), _FLAG):
            setattr(obj, k, v)


def fresh(old, make):
    """A new solver (make()) carrying the old one's flags."""
    new = make()
    set_flags(new, flags(old))
    return new


# --- files --------------------------------------------------------------------

def _write(path: str, obj) -> float:
    t = time.time()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.{os.getpid()}.tmp"
    with open(tmp, "wb") as fh:
        pickle.dump(obj, fh, protocol=pickle.HIGHEST_PROTOCOL)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)            # a reset mid-write leaves the last good file
    return time.time() - t


def _read(path: str):
    try:
        with open(path, "rb") as fh:
            obj = pickle.load(fh)
    except (OSError, EOFError, pickle.UnpicklingError, AttributeError, ImportError, ValueError):
        return None
    return obj if isinstance(obj, dict) and obj.get("format") == FORMAT else None


class Call:
    """One optimizer call's checkpoints: its finished result (`done`,
    `finish`) and its loop state while it runs (`restore`, `due`, `save`)."""

    def __init__(self, root: str, key: str, domain, helpers: dict | None = None):
        self.root = root
        self.key = key
        self.domain = domain
        self.helpers = dict(domain_helpers(domain), **(helpers or {}))
        self.done_path = os.path.join(root, "calls", f"{key}.done.pkl")
        self.loop_path = os.path.join(root, "calls", f"{key}.loop.pkl")
        p = os.environ.get(ENV_SECONDS)
        self.period = PERIOD if p is None else float(p)
        self.last = time.time()
        self.cost = 0.0

    @classmethod
    def open(cls, kind: str, domain, *inputs, helpers=None) -> "Call | None":
        root = folder()
        if root is None:
            return None
        return cls(root, call_key(kind, domain, *inputs), domain, helpers)

    # finished calls
    def done(self):
        rec = _read(self.done_path)
        if rec is None:
            return None
        restore_helpers(self.helpers, rec.get("helpers"))
        _stats["replayed"] += 1
        return rec["result"]

    def finish(self, result) -> None:
        try:
            _write(self.done_path, {"format": FORMAT, "result": result,
                                    "helpers": helpers_state(self.helpers, self.domain)})
        except OSError:
            return              # a full disk costs the checkpoint, not the run
        try:
            os.remove(self.loop_path)
        except OSError:
            pass

    # the call in progress
    def add_helpers(self, **objs) -> None:
        self.helpers.update({k: v for k, v in objs.items() if v is not None})

    def restore(self) -> dict | None:
        rec = _read(self.loop_path)
        if rec is None:
            return None
        restore_helpers(self.helpers, rec.get("helpers"))
        _stats["restored"] += 1
        import sys
        print(json.dumps({"stage": "resume", "call": self.key, "iteration": rec["state"].get("it")}),
              file=sys.stderr, flush=True)
        return rec["state"]

    def boundary(self, it: int) -> bool:
        """Is iteration `it` (just finished) a checkpoint iteration? Every
        EVERY-th, by count: when a run checkpoints may not hang on the clock,
        or two runs of one study would part at the solver's tolerance."""
        return it > 0 and it % EVERY == 0

    def due(self) -> bool:
        """Write at this boundary: not sooner than LOADPATH_CHECKPOINT_SECONDS
        after the last write, nor than COST_RATIO times what it took."""
        return time.time() - self.last >= max(self.period, COST_RATIO * self.cost)

    def save(self, state: dict) -> None:
        try:
            self.cost = _write(self.loop_path, {"format": FORMAT, "state": state,
                                                "helpers": helpers_state(self.helpers, self.domain)})
            _stats["written"] += 1
        except OSError:
            pass
        self.last = time.time()


# --- the run ------------------------------------------------------------------

RUN_FILE = "run.json"


def code_version() -> str:
    """A hash of loadpath's own source: did the code change under a run?"""
    h = hashlib.blake2b(digest_size=12)
    here = os.path.dirname(os.path.abspath(__file__))
    for dp, dns, fns in os.walk(here):
        dns[:] = sorted(d for d in dns if not d.startswith((".", "__")))
        for f in sorted(fns):
            if f.endswith(".py"):
                with open(os.path.join(dp, f), "rb") as fh:
                    h.update(f.encode() + fh.read())
    return h.hexdigest()


def start(out: str, resume: bool) -> str:
    """Point this process (and the workers it starts) at <out>/checkpoint.
    A fresh run clears what an earlier one left there."""
    import shutil

    root = os.path.abspath(os.path.join(out, "checkpoint"))
    if not resume and os.path.isdir(os.path.join(root, "calls")):
        shutil.rmtree(os.path.join(root, "calls"), ignore_errors=True)
    os.makedirs(root, exist_ok=True)
    os.environ[ENV] = root
    return root


def summary() -> dict:
    root = folder()
    n_done = n_loop = 0
    if root and os.path.isdir(os.path.join(root, "calls")):
        for f in os.listdir(os.path.join(root, "calls")):
            n_done += f.endswith(".done.pkl")
            n_loop += f.endswith(".loop.pkl")
    return {"folder": root, "finishedCalls": n_done, "callsInProgress": n_loop, **stats()}
