"""
Thin-member check of the voxel hex8 against CalculiX.

Strips 1, 2, 3 and 5 cells thick, at 0, 30 and 45 degrees to the voxel grid,
are clamped at one end and loaded at the other by a tip shear (bending) or a
tip pull (tension). The voxel model is the production one: cells whose centre
is inside the strip, `fem.Stiffness` with the element in force (incompatible
modes by default, LOADPATH_HEX8=plain for the old trilinear hex8) and
`verify.nodal_stress`. The reference is CalculiX on the exact strip, a
structured quadratic hex mesh (C3D20, four elements through the thickness).

Both are scored the way a peak-stress check reads them: the largest von Mises
over a window away from the clamp and the load (30-70% of the length),
divided pointwise by beam theory (6 F (s_F - s) / (w t^2), or F / (w t)).
`ratio` is voxel / CalculiX on that peak: under 1 is the unsafe direction.
`stiffness` is voxel / CalculiX on the load-point displacement.
"cut" is the cut-cell model (cutcell.py, verify's resolution.cutCells).

    python -m loadpath.benchmarks.beams [--quick] [--out beams.json]
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import tempfile

import numpy as np
from scipy.sparse.linalg import spsolve

from ..fem import HexMesh, Stiffness, von_mises

E, NU, F = 2000.0, 0.35, 1.0
W = 3.0                 # strip width (out of the bending plane), in cells
OFFSET = np.array([0.37, 0.21, 0.0])
ZONE = 1.5              # the end load is a body force over the last 1.5 cells


class Strip:
    """A t x W strip along direction theta in the xy plane, clamped over
    s in [-c, 0], loaded over s in [L - ZONE, L]."""

    def __init__(self, t: float, theta_deg: float):
        self.t = float(t)
        self.L = float(max(12.0 * t, 24.0))
        self.c = 2.0
        th = np.radians(theta_deg)
        self.a = np.array([np.cos(th), np.sin(th), 0.0])
        self.n = np.array([-np.sin(th), np.cos(th), 0.0])

    def local(self, p):
        d = np.asarray(p, float) - OFFSET
        return d @ self.a, d @ self.n, d[..., 2]

    def world(self, s, y, z):
        return OFFSET + np.outer(s, self.a) + np.outer(y, self.n) + np.outer(z, [0.0, 0.0, 1.0])

    def inside(self, p):
        s, y, z = self.local(p)
        return (s >= -self.c) & (s <= self.L) & (np.abs(y) <= self.t / 2) & (z >= 0) & (z <= W)

    def beam(self, load, s, s_f):
        if load == "tension":
            return np.full_like(s, F / (W * self.t))
        return 6.0 * F * (s_f - s) / (W * self.t ** 2)

    def direction(self, load):
        return self.a if load == "tension" else self.n

    @property
    def window(self):
        return 0.3 * self.L, 0.7 * self.L


def voxel(strip: Strip, load: str, element: str | None = None, pitch: float = 1.0) -> dict:
    """The production voxel model of the strip at `pitch` (cells; the strip
    is t cells thick at pitch 1)."""
    lo3 = np.array([-strip.c, -strip.t / 2, 0.0])
    hi3 = np.array([strip.L, strip.t / 2, W])
    corners = strip.world(*np.array(np.meshgrid(*zip(lo3, hi3))).reshape(3, -1))
    h = float(pitch)
    origin = (np.floor(corners.min(0) / h) - 1.0) * h
    origin[2] = 0.0
    dims = np.ceil((corners.max(0) - origin) / h).astype(int) + 1
    dims[2] = int(round(W / h))
    ijk = np.argwhere(np.ones(dims, bool))
    centres = origin + (ijk + 0.5) * h
    ijk = ijk[strip.inside(centres)]
    mesh = HexMesh(ijk, h, origin)
    s_node = strip.local(mesh.node_xyz)[0]
    fixed = np.repeat(s_node < 0, 3)
    # end load: a uniform body force over the cells of the end zone
    s_cell = strip.local(mesh.origin + (mesh.elem_ijk + 0.5) * h)[0]
    zone = s_cell > strip.L - ZONE
    wn = np.bincount(mesh.enode[zone].ravel(), minlength=mesh.n_nodes) / (8.0 * zone.sum())
    f = (F * wn[:, None] * strip.direction(load)[None]).ravel()
    s_f = float(s_cell[zone].mean())
    st = Stiffness(mesh, NU, fixed, element=element)
    k = st.assemble(np.full(mesh.n_elem, E))
    u = np.zeros(mesh.n_dof)
    free = ~fixed
    u[free] = spsolve(k[free][:, free].tocsc(), f[free])
    from ..verify import nodal_stress

    vm = von_mises(nodal_stress(st, u, E))
    lo, hi = strip.window
    win = (s_node >= lo) & (s_node <= hi)
    return {"peak": float(np.max(vm[win] / strip.beam(load, s_node[win], s_f))),
            "displacement": float(f @ u) / F, "cells": int(mesh.n_elem), "element": st.element}


def cut(strip: Strip, load: str, pitch: float = 1.0, n: int = 4) -> dict:
    """The cut-cell model (cutcell.py) of the strip: sub-cells from the
    exact strip, the stress read at the surface near it."""
    from ..cutcell import CutGrid

    h = float(pitch)
    lo3 = np.array([-strip.c, -strip.t / 2, 0.0])
    hi3 = np.array([strip.L, strip.t / 2, W])
    corners = strip.world(*np.array(np.meshgrid(*zip(lo3, hi3))).reshape(3, -1))
    origin = (np.floor(corners.min(0) / h) - 1.0) * h
    origin[2] = 0.0
    dims = np.ceil((corners.max(0) - origin) / h).astype(int) + 1
    dims[2] = int(round(W / h))
    fd = dims * n
    g = [origin[a] + (np.arange(fd[a]) + 0.5) * h / n for a in range(3)]
    pts = np.stack(np.meshgrid(*g, indexing="ij"), -1).reshape(-1, 3)
    fine = strip.inside(pts).reshape(fd)
    cg = CutGrid(fine, n, origin, h, NU)
    mesh = cg.mesh
    s_node = strip.local(mesh.node_xyz)[0]
    fixed = np.repeat(s_node < 0, 3)
    s_cell = strip.local(mesh.origin + (mesh.elem_ijk + 0.5) * h)[0]
    zone = s_cell > strip.L - ZONE
    wz = np.where(zone, cg.fill, 0.0)
    wn = np.zeros(mesh.n_nodes)
    np.add.at(wn, mesh.enode.ravel(), np.repeat(wz / 8.0, 8))
    wn /= wz.sum()
    f = (F * wn[:, None] * strip.direction(load)[None]).ravel()
    s_f = float((s_cell * wz).sum() / wz.sum())
    st = Stiffness(mesh, NU, fixed)
    k = cg.assemble(st, E)
    u = np.zeros(mesh.n_dof)
    free = ~fixed
    u[free] = spsolve(k[free][:, free].tocsc(), f[free])
    vm = von_mises(cg.nodal_stress(u, E))
    lo, hi = strip.window
    win = (s_node >= lo) & (s_node <= hi)
    return {"peak": float(np.max(vm[win] / strip.beam(load, s_node[win], s_f))),
            "displacement": float(f @ u) / F, "cells": int(mesh.n_elem), "cutCells": int(cg.cut.sum())}


# C3D20 node order on the doubled grid: corners, bottom edges, top edges, verticals
_HEX20 = [(0, 0, 0), (2, 0, 0), (2, 2, 0), (0, 2, 0), (0, 0, 2), (2, 0, 2), (2, 2, 2), (0, 2, 2),
          (1, 0, 0), (2, 1, 0), (1, 2, 0), (0, 1, 0), (1, 0, 2), (2, 1, 2), (1, 2, 2), (0, 1, 2),
          (0, 0, 1), (2, 0, 1), (2, 2, 1), (0, 2, 1)]


def calculix(strip: Strip, load: str, through: int = 4, along: float = 0.5) -> dict:
    """CalculiX (ccx on PATH) on the exact strip, quadratic hexes."""
    from ..crosscheck import _read_frd

    exe = shutil.which("ccx")
    if exe is None:
        raise RuntimeError("CalculiX (ccx) is not on PATH (Debian: apt install calculix-ccx)")
    ns = int(round((strip.L + strip.c) / along))
    nt, nw = through, 3
    gs = np.linspace(-strip.c, strip.L, 2 * ns + 1)
    gy = np.linspace(-strip.t / 2, strip.t / 2, 2 * nt + 1)
    gz = np.linspace(0.0, W, 2 * nw + 1)
    shape = (len(gs), len(gy), len(gz))
    elems, ezone = [], []
    for i in range(ns):
        for j in range(nt):
            for k in range(nw):
                elems.append([np.ravel_multi_index((2 * i + a, 2 * j + b, 2 * k + c), shape) for a, b, c in _HEX20])
                ezone.append(0.5 * (gs[2 * i] + gs[2 * i + 2]) > strip.L - ZONE)
    elems = np.array(elems)
    used, conn = np.unique(elems, return_inverse=True)
    conn = conn.reshape(elems.shape)
    ii, jj, kk = np.unravel_index(used, shape)
    s, y, z = gs[ii], gy[jj], gz[kk]
    xyz = strip.world(s, y, z)
    ezone = np.array(ezone)
    bf = F / (ZONE * strip.t * W) * strip.direction(load)
    s_f = strip.L - ZONE / 2
    with tempfile.TemporaryDirectory() as tmp:
        with open(os.path.join(tmp, "job.inp"), "w") as fh:
            fh.write("*NODE,NSET=NALL\n")
            np.savetxt(fh, np.column_stack([np.arange(1, len(xyz) + 1), xyz]), fmt="%d,%.12g,%.12g,%.12g")
            fh.write("*ELEMENT,TYPE=C3D20,ELSET=PART\n")
            rows = np.column_stack([np.arange(1, len(conn) + 1), conn + 1])
            for r in rows:   # 21 integers: CalculiX takes at most 16 per line
                fh.write(",".join(map(str, r[:16])) + ",\n" + ",".join(map(str, r[16:])) + "\n")
            fh.write("*ELSET,ELSET=EZONE\n")
            np.savetxt(fh, np.nonzero(ezone)[0] + 1, fmt="%d")
            fh.write("*NSET,NSET=CLAMP\n")
            np.savetxt(fh, np.nonzero(s <= 1e-9)[0] + 1, fmt="%d")
            fh.write(f"*MATERIAL,NAME=M\n*ELASTIC\n{E},{NU}\n*SOLID SECTION,ELSET=PART,MATERIAL=M\n")
            fh.write("*BOUNDARY\nCLAMP,1,3\n*STEP\n*STATIC\n*DLOAD\n")
            for c, lab in enumerate(("BX", "BY", "BZ")):
                if abs(bf[c]) > 1e-15:
                    fh.write(f"EZONE,{lab},{bf[c]:.12g}\n")
            fh.write("*NODE FILE\nU\n*EL FILE\nS\n*END STEP\n")
        run = subprocess.run([exe, "job"], cwd=tmp, capture_output=True, text=True, timeout=1800,
                             env={**os.environ, "OMP_NUM_THREADS": "1"})
        if run.returncode != 0 or not os.path.exists(os.path.join(tmp, "job.frd")):
            raise RuntimeError(f"CalculiX failed: {(run.stdout + run.stderr)[-2000:]}")
        u, vm, _sig = _read_frd(os.path.join(tmp, "job.frd"), len(xyz))[0]
    lo, hi = strip.window
    win = (s >= lo) & (s <= hi)
    # load-point displacement: the body force's work over the end zone,
    # sampled at the zone's nodes (same weights as the load)
    zn = np.unique(conn[ezone].ravel())
    d = float(np.mean(u[zn] @ strip.direction(load)))
    return {"peak": float(np.max(vm[win] / strip.beam(load, s[win], s_f))),
            "displacement": d, "elements": int(len(conn))}


def run(thick=(1, 2, 3, 5), angles=(0, 30, 45), loads=("bending", "tension"), elements=("cut", "im", "plain")):
    rows = []
    for load in loads:
        for t in thick:
            for th in angles:
                strip = Strip(t, th)
                ref = calculix(strip, load)
                row = {"load": load, "t": t, "theta": th, "calculix": ref}
                for el in elements:
                    v = cut(strip, load) if el == "cut" else voxel(strip, load, el)
                    v["ratio"] = v["peak"] / ref["peak"]
                    v["stiffness"] = ref["displacement"] / v["displacement"]
                    row[el] = v
                rows.append(row)
                print(f"{load:8s} t={t} theta={th:2d}  calculix {ref['peak']:.3f}  " + "  ".join(
                    f"{el} {row[el]['ratio']:.3f} (stiff {row[el]['stiffness']:.3f})" for el in elements), flush=True)
    return rows


def table(rows, elements=("cut", "im", "plain")) -> str:
    head = "| load | cells thick | angle | " + " | ".join(
        f"{el} peak / CalculiX | {el} stiffness / CalculiX" for el in elements) + " |"
    out = [head, "|" + "---|" * (3 + 2 * len(elements))]
    for r in rows:
        out.append(f"| {r['load']} | {r['t']} | {r['theta']} | " + " | ".join(
            f"{r[el]['ratio']:.2f} | {r[el]['stiffness']:.2f}" for el in elements) + " |")
    return "\n".join(out)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--quick", action="store_true", help="1 and 3 cells, 0 and 45 degrees, bending only")
    ap.add_argument("--out", help="write the rows as JSON here")
    a = ap.parse_args(argv)
    kw = dict(thick=(1, 3), angles=(0, 45), loads=("bending",)) if a.quick else {}
    rows = run(**kw)
    print()
    print(table(rows))
    if a.out:
        with open(a.out, "w") as fh:
            json.dump(rows, fh, indent=1)


if __name__ == "__main__":
    main()
