"""
Benchmark suite: loadpath against the published topology-optimization codes
on the textbook problems (problems.py), scored by one evaluator
(evaluate.py). Separate from the GRIP evals: those ask "is this part good
enough to print", this asks "how close to the optimum does the optimizer get".

For each problem:
- the reference designs (top88 / top3d ports, reference.py), stored, rerun
  with --refresh;
- loadpath's optimizer as a study would run it (robust erosion, Heaviside
  projection, the padded filter for 3D, pruning, pinned load cells), and
  "plain" with erosion, padding and pruning off: the same formulation as the
  references, which separates what the solver does from what loadpath's
  printability choices cost;
- stress problems also run loadpath's minimum-mass stress-constrained
  formulation (constrained.py) with the reference design's peak as the limit.

Scores (evaluate.py): compliance as delivered (penalized, p = 3) and
black-and-white at the target volume; the headline is the lower of the two
(lower is stiffer). loadpath's volume fraction is lowered by its pinned load
and support cells so every design uses the same total material. Stress
problems add the peak von Mises of the black-and-white design. The ratio
column is each design over the best reference.

Run:  python -m loadpath.benchmarks.run [problem ...] --out bench-out/
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

from ..constrained import Limits, optimize_constrained
from ..domain import build_domain
from ..simp import SimpOptions, optimize
from ..spec import study_from_dict
from .evaluate import Evaluator, binarize
from .problems import PROBLEMS, Problem
from .reference import SOLVERS, load_reference, run_reference

NU = 0.3
VARIANTS = {
    "loadpath": {},
    "loadpath-plain": {"erode": 0.0, "prune": False, "pad_filter": False},
}


def _box(nodes, plane):
    lo = nodes.min(axis=0) - 0.25
    hi = nodes.max(axis=0) + 0.25
    if plane:
        lo[1], hi[1] = -0.25, 1.25
    return {"type": "box", "min": lo.tolist(), "max": hi.tolist()}


def _slab(nodes, plane):
    """A plane problem's node set on both faces of the one-cell slab."""
    if not plane:
        return nodes
    other = nodes.copy()
    other[:, 1] = 1
    return np.concatenate([nodes, other])


def domain_for(p: Problem):
    """loadpath's Domain for a benchmark, built the way a study builds it (so
    load and support cells are pinned and symmetry is mirrored as in real
    use), with the supports and loads then set exactly to the problem's."""
    nx, ny, nz = p.dims
    d = {
        "designSpace": [{"type": "box", "min": [0, 0, 0], "max": [nx, ny, nz]}],
        "supports": [{"shape": _box(n, p.plane), "fix": "".join(sorted(c))} for n, c in p.fixed],
        "loadCases": [{"name": "load", "loads": [
            {"shape": _box(n, p.plane), "force": [v * len(n) if a == c else 0.0 for a in "xyz"]}
            for n, c, v in p.loads]}],
        "material": {"name": "unit", "E": 1.0, "nu": NU, "strength": 1.0, "density": 1.0},
        "goal": {"volumeFraction": p.active_volfrac},
        "resolution": {"pitch": 1.0},
    }
    if p.symmetry:
        d["symmetry"] = list(p.symmetry)
    if p.plane:
        d["extrusion"] = "y"
    if p.void is not None:
        # the cut-out as keep-out boxes, one per void column run
        ii, _, kk = np.nonzero(p.void)
        d["keepOut"] = [{"id": "cut", "shape": {"type": "box", "min": [float(ii.min()), -1, float(kk.min())],
                                                  "max": [float(ii.max() + 1) + 1, ny + 1, float(kk.max() + 1) + 1]},
                         "bore": False}]
    study = study_from_dict(d)
    dom = build_domain(study, pitch=1.0)
    if tuple(dom.dims) != tuple(p.dims) or np.abs(dom.origin).max() > 1e-9:
        raise RuntimeError(f"{p.name}: loadpath grid {dom.dims} at {dom.origin} does not match {p.dims}")
    mesh = dom.mesh
    fixed = np.zeros(mesh.n_dof, bool)
    f = np.zeros(mesh.n_dof)
    for nodes, comps in p.fixed:
        ids = mesh.node_id(_slab(nodes, p.plane))
        for c in comps:
            fixed[3 * ids + "xyz".index(c)] = True
    for nodes, c, v in p.loads:
        all_n = _slab(nodes, p.plane)
        share = v * len(nodes) / len(all_n)
        np.add.at(f, 3 * mesh.node_id(all_n) + "xyz".index(c), share)
    if p.plane:
        # the slab is half of a plate symmetric about y = 0: holds y there
        fixed[3 * np.flatnonzero(mesh.node_ijk[:, 1] == 0) + 1] = True
    dom.fixed, dom.loads, dom.springs = fixed, [f], None
    return dom, study


def _grid(dom, values):
    g = np.zeros(dom.dims)
    g[tuple(dom.mesh.elem_ijk.T)] = values
    return g


def _scored_grid(p: Problem, g):
    """The part of a loadpath design the problem scores (the half of a
    mirrored whole)."""
    if p.full is not None:
        return g[p.dims[0]:]
    return g


def design_volfrac(q: Problem, dom) -> float:
    """loadpath's volume fraction counts design cells only; pinned (load,
    support) cells come on top. This is the fraction of the rest that gives
    the same total material as the references."""
    n, pres = dom.mesh.n_elem, int(dom.preserve.sum())
    return max(0.01, (q.active_volfrac * n - pres) / (n - pres))


def run_loadpath(p: Problem, variant: str, log=print):
    q = p.full or p
    dom, study = domain_for(q)
    opts = SimpOptions(volume_fraction=design_volfrac(q, dom), filter_radius=q.rmin,
                       extrusion=1 if q.plane else None, **VARIANTS[variant])
    t0 = time.time()
    res = optimize(dom, NU, opts)
    log(f"  {variant}: {res.iterations} iterations, {time.time() - t0:.0f}s, c {res.compliance:.4g}")
    return dict(x=_scored_grid(p, _grid(dom, res.xphys)), iterations=res.iterations, seconds=time.time() - t0,
                compliance_reported=res.compliance)


def run_constrained(p: Problem, limit: float, log=print):
    q = p.full or p
    dom, _ = domain_for(q)
    # Started from (nearly) solid, a feasible design, as Le et al. 2010 do.
    # Since the fixes in constrained.py (q = 1, a guarded MMA step, the
    # blueprint delivered) it also converges from the reference's volume.
    opts = SimpOptions(volume_fraction=0.95, filter_radius=q.rmin,
                       extrusion=1 if q.plane else None, max_iterations=200)
    t0 = time.time()
    res = optimize_constrained(dom, NU, opts, Limits(stress=limit, deflection=None, modulus=1.0))
    log(f"  stress-constrained: {res.iterations} iterations, {time.time() - t0:.0f}s, vol {res.volume_fraction:.3f}")
    return dict(x=_scored_grid(p, _grid(dom, res.xphys)), iterations=res.iterations, seconds=time.time() - t0,
                compliance_reported=res.compliance)


def bench_problem(p: Problem, refresh=False, log=print) -> dict:
    ev = Evaluator(p)
    target = p.active_volfrac
    designs, rows = {}, []
    for solver in p.references:
        rec = None if refresh else load_reference(p, solver)
        if rec is None:
            log(f"  running {solver} ...")
            rec = run_reference(p, solver)
        designs[solver] = rec
    for v in VARIANTS:
        designs[v] = run_loadpath(p, v, log)
    for name, rec in designs.items():
        s = ev.score(rec["x"], target)
        s.update(design=name, iterations=rec["iterations"], seconds=round(rec["seconds"], 1),
                 compliance_reported=rec["compliance_reported"])
        rows.append(s)
    refs = [r for r in rows if r["design"] in p.references]
    best = min(refs, key=lambda r: r["compliance"])
    for r in rows:
        r["ratio"] = r["compliance"] / best["compliance"]
    out = {"problem": p.name, "title": p.title, "source": p.source, "dims": list(p.dims),
           "volfrac": target, "rmin": p.rmin, "rows": rows, "best_reference": best["design"]}
    if p.stress:
        limit = best["peak_von_mises"]
        try:
            rec = run_constrained(p, limit, log)
        except Exception as exc:  # noqa: BLE001 - a failing formulation is a result
            log(f"  stress-constrained failed: {type(exc).__name__}: {exc}")
            out["stress"] = {"limit": limit, "reference_volume": target, "error": f"{type(exc).__name__}: {exc}"}
            out["_designs"] = designs
            return out
        designs["loadpath-stress"] = rec
        x = ev.cells(rec["x"])
        vol = float(x[ev.active].mean())
        b = np.zeros(p.dims)
        b[ev.cell] = binarize(x, ev.active, vol)
        vm = ev.von_mises(b)
        vm[ev.load_cells()] = 0.0
        out["stress"] = {"limit": limit, "reference_volume": target, "volume": vol,
                         "peak_von_mises": float(vm.max()), "compliance": ev.compliance(b),
                         "iterations": rec["iterations"], "seconds": round(rec["seconds"], 1)}
    out["_designs"] = designs
    return out


def _render(res: dict, path: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    designs = res["_designs"]
    names = list(designs)
    plane = res["dims"][1] == 1
    fig = plt.figure(figsize=(3.6 * len(names), 2.6 if plane else 3.2))
    for k, name in enumerate(names):
        g = designs[name]["x"]
        row = next((r for r in res["rows"] if r["design"] == name), None)
        label = name if row is None else f"{name}\nc = {row['compliance']:.4g}"
        if name == "loadpath-stress":
            label = f"{name}\nvol {res['stress']['volume']:.3f}, peak {res['stress']['peak_von_mises']:.3g}"
        if plane:
            ax = fig.add_subplot(1, len(names), k + 1)
            ax.imshow(1 - g[:, 0, :].T, cmap="gray", origin="lower", vmin=0, vmax=1, interpolation="nearest")
            ax.set_axis_off()
        else:
            from scipy import ndimage
            from skimage.measure import marching_cubes
            ax = fig.add_subplot(1, len(names), k + 1, projection="3d")
            # light smoothing so the voxel steps do not read as the design
            pad = ndimage.gaussian_filter(np.pad(g, 1), 0.6)
            v, f, _, _ = marching_cubes(pad, 0.5)
            ax.plot_trisurf(v[:, 0], v[:, 1], f, v[:, 2], color="#c9c4bc", edgecolor="none", shade=True)
            ax.set_box_aspect(np.array(g.shape) + 2)
            ax.view_init(24, -58)
            ax.set_axis_off()
        ax.set_title(label, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def markdown(results: list) -> str:
    lines = ["| problem | design | compliance | vs best reference | as delivered (p=3) | black/white at target "
             "volume | volume | greyness | iterations | seconds |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for res in results:
        for r in res["rows"]:
            lines.append(f"| {res['problem']} | {r['design']} | {r['compliance']:.4g} | {r['ratio']:.3f} | "
                         f"{r['compliance_grey']:.4g} | {r['compliance_matched']:.4g} | "
                         f"{r['volume']:.3f} | {r['greyness']:.3f} | {r['iterations']} | {r['seconds']:.0f} |")
    stress = [r for r in results if "stress" in r]
    if stress:
        lines += ["", "| problem | reference peak von Mises (limit) | reference volume | "
                      "loadpath stress-constrained volume | its peak von Mises |", "|---|---|---|---|---|"]
        for res in stress:
            s = res["stress"]
            if "error" in s:
                lines.append(f"| {res['problem']} | {s['limit']:.4g} | {s['reference_volume']:.3f} | "
                             f"failed: {s['error']} | |")
                continue
            lines.append(f"| {res['problem']} | {s['limit']:.4g} | {s['reference_volume']:.3f} | {s['volume']:.3f} | "
                         f"{s['peak_von_mises']:.4g} |")
    return "\n".join(lines) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("problems", nargs="*", help=f"subset of {sorted(PROBLEMS)}")
    ap.add_argument("--out", default="bench-out")
    ap.add_argument("--refresh", action="store_true", help="rerun the reference codes")
    ap.add_argument("--no-render", action="store_true")
    a = ap.parse_args(argv)
    names = a.problems or list(PROBLEMS)
    os.makedirs(a.out, exist_ok=True)
    results = []
    for name in names:
        p = PROBLEMS[name]
        print(f"{name}: {p.title}", flush=True)
        res = bench_problem(p, refresh=a.refresh, log=lambda s: print(s, flush=True))
        np.savez_compressed(os.path.join(a.out, f"{name}.designs.npz"),
                            **{k: v["x"].astype(np.float32) for k, v in res["_designs"].items()})
        if not a.no_render:
            _render(res, os.path.join(a.out, f"{name}.png"))
        results.append(res)
    clean = [{k: v for k, v in r.items() if not k.startswith("_")} for r in results]
    with open(os.path.join(a.out, "benchmarks.json"), "w") as fh:
        json.dump(clean, fh, indent=1)
    md = markdown(clean)
    with open(os.path.join(a.out, "benchmarks.md"), "w") as fh:
        fh.write(md)
    print(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
