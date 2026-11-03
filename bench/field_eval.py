"""
Where `loadpath field` spends its time evaluating the field graph.

    python bench/field_eval.py GRAPH.json [--pitch MM] [--points N] [--save out.npz] [--check out.npz]
                                         [--mesh] [--nodes]

Without --mesh: evaluates the graph's output on N points (default 4M) sampled
in a fixed (seeded) way around its surface, as narrowband/relax do, and
prints throughput. --save keeps the values; --check compares against a saved
run (max abs difference, count of non-identical values). With --mesh: runs
the graph's mesh at --pitch end to end and prints the share of the wall time
spent inside field evaluation. --nodes: exclusive time per graph node
(summed over threads, so it can exceed the wall time).

Result nodes that run a study are analyzed once and their samples pickled
next to --cache (default: GRAPH.results.pkl), so repeated runs time the
field, not the FE.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import resource
import sys
import threading
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np  # noqa: E402

from loadpath import fields as F  # noqa: E402
from loadpath.narrowband import peval  # noqa: E402


class NodeTimer:
    """Exclusive time per node, thread-safe (a per-thread stack)."""

    def __init__(self):
        self.t = defaultdict(float)
        self.n = defaultdict(int)
        self.lock = threading.Lock()
        self.local = threading.local()

    def install(self):
        orig = F.FieldGraph._compute
        timer = self

        def timed(g, nd, P, memo):
            st = getattr(timer.local, "stack", None)
            if st is None:
                st = timer.local.stack = []
            t0 = time.perf_counter()
            st.append(0.0)
            try:
                return orig(g, nd, P, memo)
            finally:
                dt = time.perf_counter() - t0
                child = st.pop()
                if st:
                    st[-1] += dt
                key = nd.op + ":" + nd.name
                with timer.lock:
                    timer.t[key] += dt - child
                    timer.n[key] += len(P)
        F.FieldGraph._compute = timed

    def report(self, top=25):
        tot = sum(self.t.values())
        for k, v in sorted(self.t.items(), key=lambda kv: -kv[1])[:top]:
            print(f"  {v:8.2f} s {100 * v / tot:5.1f}%  {self.n[k] / 1e6:8.1f} Mpts  {k}")
        print(f"  {tot:8.2f} s total (thread-summed)")


class EvalWall:
    """Wall time with at least one top-level evaluate running."""

    def __init__(self):
        self.lock = threading.Lock()
        self.active = 0
        self.t0 = 0.0
        self.wall = 0.0
        self.calls = 0
        self.points = 0

    def install(self):
        orig = F.FieldGraph.evaluate
        w = self

        def ev(g, P, name=None):
            with w.lock:
                if w.active == 0:
                    w.t0 = time.perf_counter()
                w.active += 1
                w.calls += 1
                w.points += len(P)
            try:
                return orig(g, P, name)
            finally:
                with w.lock:
                    w.active -= 1
                    if w.active == 0:
                        w.wall += time.perf_counter() - w.t0
        F.FieldGraph.evaluate = ev


def build(path, cache=None):
    doc = json.load(open(path))
    base = os.path.dirname(os.path.abspath(path))
    cache = cache or os.path.splitext(path)[0] + ".results.pkl"
    g = F.FieldGraph(doc, base_dir=base)
    results = [n for n, nd in g.nodes.items() if nd.op == "result" and "study" in nd.spec]
    if results and os.path.exists(cache):
        g.seed(pickle.load(open(cache, "rb")))
    elif results:
        for n in results:
            g._resource(g.nodes[n], lambda n=n: g._samples(g.nodes[n]))
        pickle.dump({n: (g._built[n], g.analyses.get(n)) for n in results}, open(cache, "wb"))
    return doc, g


def sample_points(g, n, pitch, seed=0):
    """n points near the output's surface (where meshing evaluates): a
    coarse grid's cells that straddle the surface, jittered inside them."""
    lo, hi = (np.asarray(x, float) for x in g.bounds())
    h = max(pitch * 4, float((np.prod(hi - lo) / 3e5) ** (1 / 3)))
    ax = [np.arange(lo[i] - h, hi[i] + 2 * h, h) for i in range(3)]
    G = np.stack(np.meshgrid(*ax, indexing="ij"), -1).reshape(-1, 3)
    f = peval(g.field(), G)
    near = G[np.abs(f) < 1.8 * h]
    rng = np.random.default_rng(seed)
    pick = near[rng.integers(0, len(near), n)]
    return pick + rng.uniform(-0.5 * h, 0.5 * h, pick.shape)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("graph")
    ap.add_argument("--pitch", type=float)
    ap.add_argument("--points", type=int, default=4_000_000)
    ap.add_argument("--save")
    ap.add_argument("--check")
    ap.add_argument("--cache")
    ap.add_argument("--mesh", action="store_true")
    ap.add_argument("--nodes", action="store_true")
    ap.add_argument("--repeat", type=int, default=2)
    a = ap.parse_args()
    t = time.perf_counter()
    doc, g = build(a.graph, a.cache)
    pitch = a.pitch or doc.get("mesh", {}).get("pitch", 0.3)
    g.evaluate(np.zeros((1, 3)))          # build resources (meshes, warps) outside the timing
    print(f"setup {time.perf_counter() - t:.1f} s", flush=True)
    out = {"graph": os.path.basename(a.graph), "pitch": pitch}
    if a.mesh:
        w = EvalWall()
        w.install()
        nt = NodeTimer() if a.nodes else None
        if nt:
            nt.install()
        t = time.perf_counter()
        m = g.mesh(pitch=pitch)
        dt = time.perf_counter() - t
        out.update(seconds=round(dt, 1), evalWall=round(w.wall, 1), evalShare=round(w.wall / dt, 3),
                   evalMpts=round(w.points / 1e6, 1), evalCalls=w.calls, faces=len(m.faces),
                   volume=round(float(m.volume), 2))
        if a.save:
            np.savez(a.save, V=m.vertices, F=m.faces)
        if a.check:
            ref = np.load(a.check)
            same = ref["F"].shape == m.faces.shape and np.array_equal(ref["F"], m.faces)
            out.update(sameFaces=bool(same),
                       maxVertexDiff=float(np.abs(ref["V"] - m.vertices).max()) if same else None)
        if nt:
            nt.report()
    else:
        P = sample_points(g, a.points, pitch)
        f = g.field()
        best = None
        nt = NodeTimer() if a.nodes else None
        for r in range(a.repeat):
            if nt and r == a.repeat - 1:
                nt.install()
            t = time.perf_counter()
            v = peval(f, P)
            dt = time.perf_counter() - t
            best = dt if best is None else min(best, dt)
        out.update(points=len(P), seconds=round(best, 2), mptsPerS=round(len(P) / best / 1e6, 2))
        if a.save:
            np.savez(a.save, P=P, v=v)
        if a.check:
            ref = np.load(a.check)
            assert np.array_equal(ref["P"], P), "different points"
            d = np.abs(ref["v"] - v)
            out.update(maxAbsDiff=float(d.max()), nonIdentical=int((ref["v"] != v).sum()))
        if nt:
            nt.report()
    out["peakRssGB"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6, 2)
    print(json.dumps(out))


if __name__ == "__main__":
    main()
