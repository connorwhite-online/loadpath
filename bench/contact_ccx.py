"""
Coulomb friction in loadpath's large-deflection contact (contact.py)
against CalculiX 2.21 on the same voxel meshes (C3D8I, NLGEOM, penalty
contact with friction, linear ramp: both solvers load proportionally).

Cases:
- block: a 4 x 2 x 2 mm elastic block (E 1, nu 0.3) pressed 0.05 mm onto
  a rigid floor through its top face and dragged sideways s mm, mu 0.3.
  Small s sticks over most of the face, large s slides: the drag force
  sweeps the partial-slip range up to mu times the normal force.
- lap: the same block pressed onto a second, deformable block of the same
  part under a one-cell gap (self contact in loadpath; a contact pair
  between the two bodies in CalculiX) and dragged.
- elastica: the 40 mm cantilever of tests/test_contact.py pressed by a dead
  tip load through large deflection onto a rough floor (mu 0.4: the tip
  sticks where it lands; 0.1: it slides back).

CalculiX's rigid floor is a slab 1000 times stiffer, fixed underneath.
Run: `python bench/contact_ccx.py` (needs `ccx` on the path, e.g. apt
install calculix-ccx). Writes bench/results/contact_ccx.json.
"""
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from loadpath.contact import RigidObstacle, SelfContact
from loadpath.fem import CORNERS, HexMesh
from loadpath.nonlinear import solve_nonlinear

E, NU, H = 1.0, 0.3, 0.5
STIFF = 20.0                      # loadpath's default penalty, x E x pitch per node
_WANT = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0), (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)]
_ORDER = [int(np.nonzero((CORNERS == np.array(w)).all(1))[0][0]) for w in _WANT]
# CalculiX C3D8 faces (by the order above): S3 is y-, S5 is y+
_FACE = {"y-": "S3", "y+": "S5"}


def _box(i0, i1, j0, j1, k0, k1):
    return np.stack(np.meshgrid(np.arange(i0, i1), np.arange(j0, j1), np.arange(k0, k1), indexing="ij"),
                    -1).reshape(-1, 3)


class _Inp:
    """A CalculiX deck from loadpath HexMeshes (each a separate body)."""

    def __init__(self, nu=NU):
        self.lines, self.nn, self.ne, self.nu = [], 0, 0, nu

    def body(self, mesh, name, modulus):
        n0, e0 = self.nn, self.ne
        self.lines.append("*NODE")
        self.lines += [f"{n0 + i + 1},{p[0]:.9g},{p[1]:.9g},{p[2]:.9g}" for i, p in enumerate(mesh.node_xyz)]
        self.lines.append(f"*ELEMENT,TYPE=C3D8I,ELSET=E{name}")
        self.lines += [f"{e0 + e + 1}," + ",".join(str(n0 + int(c[k]) + 1) for k in _ORDER)
                       for e, c in enumerate(mesh.enode)]
        self.lines.append(f"*MATERIAL,NAME=M{name}\n*ELASTIC\n{modulus},{self.nu}\n*SOLID SECTION,ELSET=E{name},"
                          f"MATERIAL=M{name}")
        self.nn += mesh.n_nodes
        self.ne += mesh.n_elem
        return n0, e0

    def nset(self, name, ids):
        self.lines.append(f"*NSET,NSET={name}")
        ids = [str(int(i) + 1) for i in ids]
        self.lines += [",".join(ids[i:i + 12]) for i in range(0, len(ids), 12)]

    def face_surface(self, name, mesh, e0, side):
        ax, sg = (1, -1) if side == "y-" else (1, 1)
        ijk = mesh.elem_ijk
        lut = {tuple(c) for c in ijk.tolist()}
        nb = ijk.copy()
        nb[:, ax] += sg
        el = [e for e, c in enumerate(nb.tolist()) if tuple(c) not in lut]
        self.lines.append(f"*SURFACE,NAME={name},TYPE=ELEMENT")
        self.lines += [f"{e0 + e + 1},{_FACE[side]}" for e in el]


def _contact(inp, slave, master, mu, slope):
    inp.lines += [f"*CONTACT PAIR,INTERACTION=SI,TYPE=SURFACE TO SURFACE,ADJUST=0.0\n{slave},{master}",
                  "*SURFACE INTERACTION,NAME=SI",
                  f"*SURFACE BEHAVIOR,PRESSURE-OVERCLOSURE=LINEAR\n{slope:.9g}"]
    if mu > 0:
        inp.lines.append(f"*FRICTION\n{mu},{slope:.9g}")


def _run(inp, workdir, name):
    d = Path(workdir) / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "job.inp").write_text("\n".join(inp.lines) + "\n")
    t0 = time.time()
    p = subprocess.run(["ccx", "-i", "job"], cwd=d, capture_output=True, text=True,
                       env={**os.environ, "OMP_NUM_THREADS": "4"})
    if "Job finished" not in p.stdout:
        raise RuntimeError(f"ccx {name} failed:\n{p.stdout[-3000:]}")
    return d, time.time() - t0


def _totals(dat, label):
    """The last 'total force' block for a node set in the .dat file."""
    txt = dat.read_text()
    blocks = re.findall(r"total force \(fx,fy,fz\) for set " + label.upper() + r"[^\n]*\n\s*\n\s*([-\d.E+ ]+)", txt)
    return [float(v) for v in blocks[-1].split()]


def _read_u(frd, n0, n):
    u = np.zeros((n, 3))
    lines = Path(frd).read_text().splitlines()
    starts = [k for k, l in enumerate(lines) if "DISP" in l]
    for l in lines[starts[-1] + 1:]:
        if l.startswith(" -3"):
            break
        if l.startswith(" -1"):
            node = int(l[3:13]) - 1
            if n0 <= node < n0 + n:
                u[node - n0] = [float(l[13 + 12 * j:25 + 12 * j]) for j in range(3)]
    return u


def block_case(s, mu, lap, workdir, delta=0.05, nu=NU, steps=8):
    """Drag of the block; returns loadpath's and CalculiX's top-face
    reaction (N) and loadpath's time."""
    lift = 2 if lap else 0                # the lap's lower block sits under a one-cell gap
    blk = HexMesh(_box(2, 10, 0, 4, 0, 4) + [0, lift + (1 if lap else 0), 0], H, np.zeros(3))
    low = HexMesh(_box(0, 12, 0, 2, 0, 4), H, np.zeros(3)) if lap else None
    gap = H if lap else 0.0
    # loadpath: one mesh (the lap's two bodies share it, as a part touching itself)
    m = HexMesh(np.concatenate([blk.elem_ijk] + ([low.elem_ijk] if lap else [])), H, np.zeros(3))
    y = m.node_xyz[:, 1]
    top = np.flatnonzero(np.abs(y - y.max()) < 1e-9)
    fixed = np.zeros(m.n_dof, bool)
    if lap:
        fixed[np.repeat(np.abs(y) < 1e-9, 3)] = True
        contact = [SelfContact(friction=mu)]
    else:
        contact = [RigidObstacle("plane", point=(0, 0, 0), normal=(0, 1, 0), friction=mu)]
    move = (s, -(gap + delta), 0.0)
    dofs = np.concatenate([3 * top + j for j in range(3)])
    vals = np.concatenate([np.full(len(top), v) for v in move])
    t0 = time.time()
    r = solve_nonlinear(m, nu, E, fixed, None, (dofs, vals), contact=contact, steps=steps)
    t_lp = time.time() - t0
    lp = r.history[-1]["reaction"]

    inp = _Inp(nu)
    n0, e0 = inp.body(blk, "B", E)
    inp.face_surface("SB", blk, e0, "y-")
    btop = np.flatnonzero(np.abs(blk.node_xyz[:, 1] - blk.node_xyz[:, 1].max()) < 1e-9)
    inp.nset("NTOP", n0 + btop)
    if lap:
        n1, e1 = inp.body(low, "L", E)
        inp.face_surface("SL", low, e1, "y+")
        inp.nset("NFIX", n1 + np.flatnonzero(np.abs(low.node_xyz[:, 1]) < 1e-9))
        master = "SL"
    else:
        floor = HexMesh(_box(0, 16, 0, 2, 0, 8), H, np.array([-1.0, -1.0, -1.0]))
        n1, e1 = inp.body(floor, "F", 1000 * E)
        inp.face_surface("SF", floor, e1, "y+")
        inp.nset("NFIX", n1 + np.flatnonzero(np.abs(floor.node_xyz[:, 1] + 1.0) < 1e-9))
        master = "SF"
    _contact(inp, "SB", master, mu, STIFF * E / H)
    inp.lines += ["*STEP,NLGEOM,INC=1000", "*STATIC", "0.05,1.0,1e-5,0.05", "*BOUNDARY", "NFIX,1,3,0",
                  f"NTOP,1,1,{move[0]}", f"NTOP,2,2,{move[1]}", "NTOP,3,3,0",
                  "*NODE PRINT,NSET=NTOP,TOTALS=ONLY\nRF", "*END STEP"]
    d, t_cx = _run(inp, workdir, f"{'lap' if lap else 'block'}-{s}-{mu}")
    cx = _totals(d / "job.dat", "NTOP")
    return {"s": s, "mu": mu, "nu": nu, "loadpath": lp, "ccx": cx, "loadpathSeconds": t_lp, "ccxSeconds": t_cx,
            "loadpathConverged": r.converged}


def elastica_case(mu, workdir):
    """The tests/test_contact.py rough-floor elastica: floor reaction and
    friction (N) and where the tip ends up, both solvers."""
    L, T, B = 40.0, 2.0, 2.0
    P = 3.0 * E * (B * T ** 3 / 12) / L ** 2
    drop = 0.35 * L
    beam = HexMesh(_box(0, 80, 0, 4, 0, 4), H, np.zeros(3))
    fixed = np.repeat(beam.node_xyz[:, 0] < 1e-9, 3)
    tip = np.flatnonzero(np.abs(beam.node_xyz[:, 0] - L) < 1e-9)
    f = np.zeros(beam.n_dof)
    f[3 * tip + 1] = -P / len(tip)
    yfloor = T / 2 - drop
    t0 = time.time()
    r = solve_nonlinear(beam, NU, E, fixed, f, contact=[RigidObstacle("plane", point=(0, yfloor, 0),
                                                                       normal=(0, 1, 0), friction=mu)], steps=10)
    t_lp = time.time() - t0
    c = r.history[-1]["contact0"]
    pos = (beam.node_xyz + r.u.reshape(-1, 3))[tip].mean(axis=0)
    lp = {"normal": -c["force"][1], "friction": -c["force"][0], "tipX": float(pos[0]), "tipY": float(pos[1]),
          "sliding": c.get("sliding")}

    inp = _Inp()
    n0, e0 = inp.body(beam, "B", E)
    inp.face_surface("SB", beam, e0, "y-")
    inp.nset("NCL", n0 + np.flatnonzero(fixed[0::3]))
    inp.nset("NTIP", n0 + tip)
    # the floor slab: 2 mm cells under the tip's reach
    fl = HexMesh(_box(0, 10, 0, 1, 0, 6), 2.0, np.array([28.0, yfloor - 2.0, -4.0]))
    n1, e1 = inp.body(fl, "F", 1000 * E)
    inp.face_surface("SF", fl, e1, "y+")
    inp.nset("NFIX", n1 + np.flatnonzero(np.abs(fl.node_xyz[:, 1] - (yfloor - 2.0)) < 1e-9))
    inp.nset("NFL", n1 + np.arange(fl.n_nodes))
    _contact(inp, "SB", "SF", mu, STIFF * E / H)
    inp.lines += ["*STEP,NLGEOM,INC=2000", "*STATIC", "0.02,1.0,1e-6,0.05", "*BOUNDARY", "NCL,1,3,0",
                  "NFIX,1,3,0", "*CLOAD"]
    inp.lines += [f"{n0 + int(t) + 1},2,{-P / len(tip):.9e}" for t in tip]
    inp.lines += ["*NODE PRINT,NSET=NFIX,TOTALS=ONLY\nRF", "*NODE FILE\nU", "*END STEP"]
    d, t_cx = _run(inp, workdir, f"elastica-{mu}")
    rf = _totals(d / "job.dat", "NFIX")
    u = _read_u(d / "job.frd", n0, beam.n_nodes)
    cpos = (beam.node_xyz + u)[tip].mean(axis=0)
    # the floor holds the beam up and back: its support reaction is minus that
    cx = {"normal": rf[1], "friction": rf[0], "tipX": float(cpos[0]), "tipY": float(cpos[1])}
    return {"mu": mu, "loadpath": lp, "ccx": cx, "loadpathSeconds": t_lp, "ccxSeconds": t_cx,
            "loadpathConverged": r.converged}


def main():
    out = {"block": [], "lap": [], "elastica": []}
    with tempfile.TemporaryDirectory() as wd:
        for s in (0.002, 0.01, 0.05):
            out["block"].append(block_case(s, 0.3, False, wd))
            print("block", json.dumps(out["block"][-1]))
        # nu 0 for the short drag: with nu 0.3 the faces' own Poisson
        # spreading dominates the friction and the net drag is a small
        # difference of large opposing forces
        for s, nu in ((0.05, 0.0), (0.5, 0.0), (0.5, NU)):
            out["lap"].append(block_case(s, 0.3, True, wd, nu=nu))
            print("lap", json.dumps(out["lap"][-1]))
        for mu in (0.1, 0.4):
            try:
                out["elastica"].append(elastica_case(mu, wd))
            except RuntimeError as e:   # ccx 2.21 cuts back on mu 0.4 now and then
                out["elastica"].append({"mu": mu, "ccxError": str(e)[:300]})
            print("elastica", json.dumps(out["elastica"][-1]))
    path = Path(__file__).parent / "results" / "contact_ccx.json"
    path.write_text(json.dumps(out, indent=1))
    print("wrote", path)


if __name__ == "__main__":
    main()
