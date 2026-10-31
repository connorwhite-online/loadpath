"""
Compliant-mechanism benchmarks: the textbook force inverter and a gripper,
run through the published reference code and loadpath's mechanism.py, and
scored by one independent evaluator.

    python -m loadpath.benchmarks.mechanisms [--out benchmark-out/]

The reference is a line-for-line NumPy port of the compliant-mechanism
version of Sigmund's 99-line code, as printed in Bendsøe & Sigmund,
"Topology Optimization" (2003), sec. 2.6 / appendix: 4-node plane-stress
squares, SIMP p = 3 with x >= 0.001, the mesh-independency sensitivity
filter, and optimality criteria damped with exponent 0.3 and move 0.1. The
half inverter is its own example: 40 x 20 cells, volume 0.3, rmin 1.2,
springs of 0.1 at the input and the output, unit input force.

Every design is scored on the plane-stress model the reference uses
(benchmarks/evaluate.py's element) with the same springs: the output
displacement in the wanted direction, as delivered (penalized grey) and
thresholded black and white at the target volume. The black-and-white
reading is what a printed part does. loadpath's design comes from its 3D
optimizer on a one-cell slab (as the compliance benchmarks run), so the
comparison is of designs, not solvers.

Hinges: a mechanism optimum on a voxel grid likes one-node hinges (two
cells meeting at a corner), which carry no load and print as nothing.
`hinges` counts them in each black-and-white design.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp

from .evaluate import _direct, _q4_ke, binarize
from .problems import _nodes

__all__ = ["MechProblem", "MECHANISMS", "MechEvaluator", "optimize_reference_mechanism", "run_loadpath_mechanism",
           "hinges", "bench_mechanism"]

NU = 0.3


@dataclass
class MechProblem:
    name: str
    title: str
    source: str
    dims: tuple                 # (nx, 1, nz) plane problems
    volfrac: float
    rmin: float
    fixed: list                 # [(nodes, comps)] clamps (pinned solid in loadpath, as a study would)
    sym: list                   # [(nodes, comps)] symmetry-line conditions (not pinned)
    inp: tuple                  # (nodes, comp, force per node)
    out: tuple                  # (nodes, comp, sign): the wanted motion is sign * u_comp
    k_in: float = 0.1
    k_out: float = 0.1
    void: np.ndarray | None = None
    notes: str = ""
    extra: dict = field(default_factory=dict)


def inverter(nx=40, nz=20):
    """Half force inverter, Bendsøe & Sigmund (2003) sec. 2.6: the top edge is
    the symmetry line; push right at its left end, and its right end should
    move left. Clamped at the bottom left."""
    return MechProblem(
        name="inverter-2d", title=f"Force inverter, half model ({nx}x{nz})",
        source="Bendsøe & Sigmund 2003, 'Topology Optimization', sec. 2.6 (Sigmund 1997); "
               "the mechanism version of the 99-line code: nelx 40, nely 20, volfrac 0.3, penal 3, rmin 1.2, "
               "springs 0.1",
        dims=(nx, 1, nz), volfrac=0.3, rmin=1.2,
        fixed=[(_nodes(0, 0, [0, 1]), "xz")],
        sym=[(_nodes(range(nx + 1), 0, nz), "z")],
        inp=(_nodes(0, 0, nz), "x", 1.0), out=(_nodes(nx, 0, nz), "x", -1.0))


def gripper(nx=60, nz=30, jaw=(10, 6)):
    """Half gripper after Sigmund 1997: the bottom edge is the symmetry line;
    push right at its left end, and the jaw (the top of a cut-out at the
    bottom right, where the workpiece sits) should close down onto it.
    Clamped at the top left."""
    jx, jz = jaw
    void = np.zeros((nx, 1, nz), bool)
    void[nx - jx:, :, :jz] = True
    return MechProblem(
        name="gripper-2d", title=f"Gripper, half model ({nx}x{nz}, {jx}x{jz} jaw gap)",
        source="Sigmund 1997, 'On the design of compliant mechanisms using topology optimization' (layout); "
               "reference design from the same 99-line mechanism code",
        dims=(nx, 1, nz), volfrac=0.3, rmin=1.2,
        fixed=[(_nodes(0, 0, [nz - 1, nz]), "xz")],
        sym=[(_nodes(range(nx - jx + 1), 0, 0), "z")],
        inp=(_nodes(0, 0, 0), "x", 1.0), out=(_nodes(range(nx - jx, nx + 1), 0, jz), "z", -1.0),
        void=void)


MECHANISMS = {p.name: p for p in (inverter(), gripper())}


class MechEvaluator:
    """top88's plane-stress element on the problem's grid, with the springs."""

    def __init__(self, p: MechProblem):
        self.p = p
        nx, _, nz = p.dims

        def node(i, k):
            return i * (nz + 1) + (nz - k)
        ii, kk = np.meshgrid(np.arange(nx), np.arange(nz), indexing="ij")
        ii, kk = ii.ravel(), kk.ravel()
        nodes = np.stack([node(ii, kk), node(ii + 1, kk), node(ii + 1, kk + 1), node(ii, kk + 1)], axis=1)
        self.edof = np.stack([2 * nodes, 2 * nodes + 1], axis=2).reshape(-1, 8)
        self.ke = _q4_ke(NU)
        self.n_dof = 2 * (nx + 1) * (nz + 1)
        self.cell = (ii, np.zeros_like(ii), kk)
        comp = {"x": 0, "z": 1}
        self.dof = lambda n, c: 2 * node(n[:, 0], n[:, 2]) + comp[c]
        fixed = np.zeros(self.n_dof, bool)
        for nodes_, comps in p.fixed + p.sym:
            for c in comps:
                fixed[self.dof(nodes_, c)] = True
        self.fixed = fixed
        self.free = np.flatnonzero(~fixed)
        n, c, v = p.inp
        self.f = np.zeros(self.n_dof)
        np.add.at(self.f, self.dof(n, c), v)
        n, c, sgn = p.out
        self.l = np.zeros(self.n_dof)
        np.add.at(self.l, self.dof(n, c), sgn / len(n))         # mean wanted motion of the output nodes
        self.springs = np.zeros(self.n_dof)
        din = self.dof(p.inp[0], p.inp[1])
        dout = self.dof(p.out[0], p.out[1])
        np.add.at(self.springs, din, p.k_in / len(din))
        np.add.at(self.springs, dout, p.k_out / len(dout))
        self._rows = np.repeat(self.edof, 8, axis=1).ravel()
        self._cols = np.tile(self.edof, (1, 8)).ravel()
        self.active = np.ones(len(ii), bool) if p.void is None else ~p.void[self.cell]

    def cells(self, grid):
        return np.asarray(grid, float)[self.cell]

    def matrix(self, e):
        data = (e[:, None, None] * self.ke[None]).ravel()
        k = sp.csr_matrix((data, (self._rows, self._cols)), shape=(self.n_dof, self.n_dof))
        return (k + sp.diags(self.springs))[self.free][:, self.free]

    def solve(self, e, rhs):
        u = np.zeros(self.n_dof)
        u[self.free] = _direct(self.matrix(e), rhs[self.free])
        return u

    def output(self, grid, penal=3.0, emin=1e-9):
        x = self.cells(grid)
        u = self.solve(emin + x ** penal * (1 - emin), self.f)
        return float(self.l @ u)

    def peak_stress(self, grid, layers=2) -> float:
        """Peak von Mises (solid cells of a black-and-white design, under the
        input force with the springs), leaving out cells within `layers` of a
        loaded, sprung or clamped node. Thin hinges show up here: they carry
        the whole mechanism's bending in a cell or two."""
        x = self.cells(grid)
        u = self.solve(1e-9 + x ** 3 * (1 - 1e-9), self.f)
        b = 0.5 * np.array([[-1, 0, 1, 0, 1, 0, -1, 0],
                            [0, -1, 0, -1, 0, 1, 0, 1],
                            [-1, -1, -1, 1, 1, 1, 1, -1]], float)
        d = 1 / (1 - NU ** 2) * np.array([[1, NU, 0], [NU, 1, 0], [0, 0, (1 - NU) / 2]])
        st = u[self.edof] @ (d @ b).T
        vm = np.sqrt(st[:, 0] ** 2 + st[:, 1] ** 2 - st[:, 0] * st[:, 1] + 3 * st[:, 2] ** 2)
        vm = np.where(x > 0.5, vm, 0.0)
        p = self.p
        nodes = np.concatenate([p.inp[0], p.out[0]] + [n for n, _ in p.fixed])
        ijk = np.stack(self.cell, axis=1)
        for n in nodes:
            vm[np.all(np.abs(ijk + 0.5 - n) <= layers, axis=1)] = 0.0
        return float(vm.max())

    def score(self, grid) -> dict:
        x = self.cells(grid)
        act = self.active
        out = {"volume": float(x[act].mean()), "greyness": float((4 * x * (1 - x))[act].mean()),
               "output_grey": self.output(grid)}
        b = np.zeros(self.p.dims)
        b[self.cell] = binarize(x, act, self.p.volfrac * x.size / act.sum())
        out["output_bw"] = self.output(b)
        out["hinges"] = hinges(b)
        out["peak_stress_bw"] = self.peak_stress(b)
        out["_bw"] = b
        return out


def hinges(grid) -> int:
    """One-node hinges in a black-and-white plane design: 2x2 windows with
    exactly two solid cells on a diagonal."""
    g = np.asarray(grid)[:, 0, :] > 0.5
    a, b, c, d = g[:-1, :-1], g[1:, :-1], g[:-1, 1:], g[1:, 1:]
    return int(((a & d & ~b & ~c) | (b & c & ~a & ~d)).sum())


def _check(nx, nz, rmin, x, dc):
    """The 99-line code's mesh-independency filter, with x and dc on the
    (nx, nz) grid."""
    out = np.zeros_like(dc)
    r = int(np.floor(rmin))
    for i in range(nx):
        for k in range(nz):
            s = 0.0
            for i2 in range(max(i - r, 0), min(i + r + 1, nx)):
                for k2 in range(max(k - r, 0), min(k + r + 1, nz)):
                    fac = rmin - np.hypot(i - i2, k - k2)
                    if fac > 0:
                        s += fac
                        out[i, k] += fac * x[i2, k2] * dc[i2, k2]
            out[i, k] /= x[i, k] * s
    return out


def optimize_reference_mechanism(p: MechProblem, maxit=300, on_iter=None) -> dict:
    """The 99-line code's mechanism version: minimize the output dof's
    displacement (minus the wanted motion), OC
    with sensitivities clipped positive and damping 0.3, move 0.1. (Its
    dummy load F(dout, 2) = -1 points the way an inverter's output should
    move, so with c = u_out the adjoint is that load's solution.)"""
    ev = MechEvaluator(p)
    nx, _, nz = p.dims
    penal, move = 3.0, 0.1
    act = ev.active.reshape(nx, nz)
    x = np.where(act, p.volfrac * nx * nz / act.sum(), 0.001)
    t0 = time.time()
    loop, change = 0, 1.0
    while change > 0.01 and loop < maxit:
        loop += 1
        xe = x.ravel()
        e = xe ** penal
        u1 = ev.solve(e, ev.f)
        u2 = ev.solve(e, ev.l)             # the book's dummy load: -1 on the output dof, i.e. along the wanted motion
        c = float(-ev.l @ u1)              # minimized: minus the wanted motion
        # d c / d x = p x^(p-1) u2' ke u1 (the book's sign convention)
        dc = penal * xe ** (penal - 1) * np.einsum("ij,jk,ik->i", u2[ev.edof], ev.ke, u1[ev.edof])
        dc = _check(nx, nz, p.rmin, x, dc.reshape(nx, nz))
        l1, l2 = 0.0, 1e5
        # MATLAB's 0/0 is NaN, which ends the bisection when every move
        # shrinks the volume (early on, few cells can help the output)
        while l1 + l2 > 0 and (l2 - l1) / (l1 + l2) > 1e-4:
            lmid = 0.5 * (l2 + l1)
            xnew = np.maximum(0.001, np.maximum(x - move, np.minimum(1.0, np.minimum(
                x + move, x * np.maximum(1e-10, -dc / lmid) ** 0.3))))
            xnew[~act] = 0.001
            if xnew.sum() - p.volfrac * nx * nz > 0:
                l1 = lmid
            else:
                l2 = lmid
        change = float(np.abs(xnew - x).max())
        x = xnew
        if on_iter is not None:
            on_iter(loop, -c, change)
    grid = np.zeros(p.dims)
    grid[:, 0, :] = x
    return dict(x=grid, iterations=loop, output_reported=-c, seconds=time.time() - t0, solver="sigmund-mech")


def _box(nodes):
    lo = nodes.min(axis=0) - 0.25
    hi = nodes.max(axis=0) + 0.25
    lo[1], hi[1] = -0.25, 1.25
    return {"type": "box", "min": lo.tolist(), "max": hi.tolist()}


def _slab(nodes):
    other = nodes.copy()
    other[:, 1] = 1
    return np.concatenate([nodes, other])


def mech_domain(p: MechProblem):
    """loadpath's Domain for a mechanism benchmark, built as a study would
    (clamp, input and output regions pinned solid; the jaw gap a keep-out),
    then with supports and loads set exactly to the problem's. The input
    case carries the force, the output case the unit wanted direction."""
    from ..domain import build_domain
    from ..spec import study_from_dict

    nx, ny, nz = p.dims
    n_in, c_in, v_in = p.inp
    n_out, c_out, sgn = p.out
    d = {
        "designSpace": [{"type": "box", "min": [0, 0, 0], "max": [nx, ny, nz]}],
        "supports": [{"shape": _box(n), "fix": "".join(sorted(c))} for n, c in p.fixed],
        "loadCases": [
            {"name": "in", "loads": [{"shape": _box(n_in), "force": [v_in * len(n_in) if a == c_in else 0.0
                                                                      for a in "xyz"]}]},
            {"name": "out", "loads": [{"shape": _box(n_out), "force": [sgn if a == c_out else 0.0 for a in "xyz"]}]}],
        "material": {"name": "unit", "E": 1.0, "nu": NU, "strength": 1.0, "density": 1.0},
        "goal": {"volumeFraction": p.volfrac},
        "resolution": {"pitch": 1.0},
        "extrusion": "y",
    }
    if p.void is not None:
        ii, _, kk = np.nonzero(p.void)
        d["keepOut"] = [{"id": "gap", "shape": {"type": "box", "min": [float(ii.min()), -1, float(kk.min()) - 1],
                                                  "max": [float(ii.max() + 1) + 1, ny + 1, float(kk.max() + 1)]},
                         "bore": False}]
    dom = build_domain(study_from_dict(d), pitch=1.0)
    if tuple(dom.dims) != tuple(p.dims) or np.abs(dom.origin).max() > 1e-9:
        raise RuntimeError(f"{p.name}: loadpath grid {dom.dims} at {dom.origin} does not match {p.dims}")
    mesh = dom.mesh
    fixed = np.zeros(mesh.n_dof, bool)
    for nodes, comps in p.fixed + p.sym:
        ids = mesh.node_id(_slab(nodes))
        for c in comps:
            fixed[3 * ids + "xyz".index(c)] = True
    fixed[3 * np.flatnonzero(mesh.node_ijk[:, 1] == 0) + 1] = True    # half of a plate symmetric about y = 0
    f = np.zeros(mesh.n_dof)
    np.add.at(f, 3 * mesh.node_id(_slab(n_in)) + "xyz".index(c_in), v_in * len(n_in) / (2 * len(n_in)))
    lo = np.zeros(mesh.n_dof)
    np.add.at(lo, 3 * mesh.node_id(_slab(n_out)) + "xyz".index(c_out), sgn)
    dom.fixed, dom.loads, dom.springs = fixed, [f, lo], None
    return dom


def run_loadpath_mechanism(p: MechProblem, filter_radius=None, log=print, **kw) -> dict:
    from ..mechanism import MechanismOptions, optimize_mechanism

    dom = mech_domain(p)
    n, pres = dom.mesh.n_elem, int(dom.preserve.sum())
    active = n                       # the jaw gap is a keep-out: not in the mesh
    vf = max(0.01, (p.volfrac * np.prod(p.dims) - pres) / (active - pres))
    opts = MechanismOptions(volume_fraction=vf, filter_radius=filter_radius or max(p.rmin, 1.5), modulus=1.0,
                            k_in=p.k_in, k_out=p.k_out, extrusion=1, **kw)
    t0 = time.time()
    r = optimize_mechanism(dom, NU, opts)
    log(f"  loadpath: {r.iterations} iterations, {time.time() - t0:.0f}s, output {r.output_displacement:.4g}")
    g = np.zeros(dom.dims)
    g[tuple(dom.mesh.elem_ijk.T)] = r.xphys
    return dict(x=g, iterations=r.iterations, seconds=time.time() - t0, output_reported=r.output_displacement)


VARIANTS = {
    "loadpath r1.5": {"filter_radius": 1.5},
    "loadpath r2": {"filter_radius": 2.0},
    # eroded / blueprint / dilated: no one-node hinges
    "loadpath robust": {"filter_radius": 2.0, "robust": 0.15, "max_iterations": 200},
}


def bench_mechanism(p: MechProblem, log=print, variants=None) -> dict:
    ev = MechEvaluator(p)
    designs = {"sigmund-mech": optimize_reference_mechanism(p)}
    log(f"  reference: {designs['sigmund-mech']['iterations']} iterations")
    for name, kw in (variants or VARIANTS).items():
        designs[name] = run_loadpath_mechanism(p, log=log, **kw)
    rows = []
    for name, rec in designs.items():
        s = ev.score(rec["x"])
        rec["bw"] = s.pop("_bw")
        rows.append({"design": name, "iterations": rec["iterations"], "seconds": round(rec["seconds"], 1),
                     "output_reported": rec["output_reported"], **s})
    ref = rows[0]
    for r in rows:
        r["bw_over_reference_bw"] = r["output_bw"] / ref["output_bw"] if ref["output_bw"] else None
    return {"problem": p.name, "title": p.title, "source": p.source, "dims": list(p.dims), "volfrac": p.volfrac,
            "rows": rows, "_designs": designs}


def render(res: dict, path: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    designs = res["_designs"]
    names = list(designs)
    fig, axes = plt.subplots(2, len(names), figsize=(3.4 * len(names), 4.6), gridspec_kw={"hspace": 0.45})
    for j, name in enumerate(names):
        row = next(r for r in res["rows"] if r["design"] == name)
        for i, (key, title) in enumerate((("x", f"{name}\ngrey: out {row['output_grey']:.3g}"),
                                          ("bw", f"black/white: out {row['output_bw']:.3g}\n"
                                                 f"peak stress {row['peak_stress_bw']:.2g}, "
                                                 f"{row['hinges']} corner hinges"))):
            ax = axes[i, j]
            ax.imshow(1 - designs[name][key][:, 0, :].T, cmap="gray", origin="lower", vmin=0, vmax=1)
            ax.set_title(title, fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])
    fig.suptitle(res["title"], fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="benchmark-out")
    ap.add_argument("--problems", nargs="*", default=list(MECHANISMS))
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    summary = []
    for name in a.problems:
        p = MECHANISMS[name]
        print(p.title, flush=True)
        res = bench_mechanism(p)
        render(res, os.path.join(a.out, f"{p.name}.png"))
        for r in res["rows"]:
            print(f"  {r['design']:16s} grey {r['output_grey']:.4g}  bw {r['output_bw']:.4g}  "
                  f"hinges {r['hinges']}  peak vm {r['peak_stress_bw']:.3g}  vol {r['volume']:.3f}  "
                  f"grey {r['greyness']:.3f}", flush=True)
        summary.append({k: v for k, v in res.items() if not k.startswith("_")})
    with open(os.path.join(a.out, "mechanisms.json"), "w") as fh:
        json.dump(summary, fh, indent=1, default=float)


if __name__ == "__main__":
    main()
