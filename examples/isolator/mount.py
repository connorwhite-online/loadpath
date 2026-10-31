"""
A TPU lattice vibration isolator: four gyroid feet under a GRIP-like robot
base whose centre of gravity is off the middle of the footprint.

    python examples/isolator/mount.py [--out isolator-out/] [--pitch 2]

The payload is a 10 mm aluminium plate (140 x 100 mm) carrying a 3.5 kg
robot whose CG sits 12 mm forward, 4 mm to the side and 80 mm up (a
column with the robot's mass at its top). It stands on four 24 x 24 x
30 mm TPU gyroid feet (8 mm cells: 0.5 mm FDM walls allow down to
rho 0.19) at (+-50, +-30) mm, tuned for a 35 Hz bounce so that motor and
fan vibration from about 50 Hz up is isolated.

Two designs at the same total stiffness (the same bounce frequency on
paper):

- uniform: four identical feet;
- graded: per-foot density from isolator.tune_mount (elastic centre under
  the CG), refined twice against the FE model so every foot squashes the
  same under gravity.

Both are checked by voxel FE of the whole mount (homogenized feet, plate,
column, robot mass): natural frequencies, static tilt under gravity,
foot loads, shock margin before the first foot buckles, and base-to-robot
transmissibility for a vertical and a sideways shake (modal superposition,
TPU loss factor). Writes summary.json and the graded feet as STL.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from scipy.sparse.linalg import spsolve

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))

from loadpath.fem import HexMesh, Stiffness  # noqa: E402
from loadpath.isolator import (G, density_for_stiffness, foot_stiffness, mount_modes, rigid_body,  # noqa: E402
                               shock_margin, tune_mount)
from loadpath.lattice import relative_density, sheet_modulus  # noqa: E402
from loadpath.materials import material_from_spec  # noqa: E402
from loadpath.modal import base_transmissibility, loss_factor, modal_analysis  # noqa: E402

KIND = "gyroid"
FEET = np.array([[50.0, 30.0], [50.0, -30.0], [-50.0, 30.0], [-50.0, -30.0]])
WIDTH, HEIGHT, CELL, MIN_WALL = 24.0, 30.0, 8.0, 0.5
PLATE = (140.0, 100.0, 10.0)
ROBOT = {"mass": 3.5, "at": (12.0, 4.0), "height": 80.0}     # kg; column top, mm
AL = {"E": 70_000.0, "density": 2.7}
F_TARGET = 35.0
BAND = (50.0, 400.0)


def build(pitch):
    """Voxel mesh and element groups: 0-3 feet, 4 plate, 5 column."""
    h = pitch
    lo = np.array([-PLATE[0] / 2, -PLATE[1] / 2, 0.0])
    n = np.round([PLATE[0] / h, PLATE[1] / h, ROBOT["height"] / h]).astype(int)
    ijk = np.argwhere(np.ones(n, bool))
    c = lo + (ijk + 0.5) * h
    group = np.full(len(ijk), -1)
    for i, (fx, fy) in enumerate(FEET):
        group[(np.abs(c[:, 0] - fx) < WIDTH / 2) & (np.abs(c[:, 1] - fy) < WIDTH / 2) & (c[:, 2] < HEIGHT)] = i
    group[(c[:, 2] > HEIGHT) & (c[:, 2] < HEIGHT + PLATE[2])] = 4
    cx, cy = ROBOT["at"]
    group[(np.abs(c[:, 0] - cx) < 10) & (np.abs(c[:, 1] - cy) < 10) & (c[:, 2] > HEIGHT + PLATE[2])] = 5
    keep = group >= 0
    return HexMesh(ijk[keep], h, lo), group[keep]


class Mount:
    def __init__(self, pitch, mat):
        self.mat = mat
        self.mesh, self.group = build(pitch)
        m = self.mesh
        xyz = m.node_xyz
        self.fixed = np.zeros(m.n_dof, bool)
        bot = np.flatnonzero(xyz[:, 2] < 1e-6)
        for a in range(3):
            self.fixed[3 * bot + a] = True
        self.bottom = bot
        self.robot = np.flatnonzero(xyz[:, 2] > ROBOT["height"] - 1e-6)
        self.plate_top = np.flatnonzero(np.abs(xyz[:, 2] - HEIGHT - PLATE[2]) < 1e-6)
        self.foot_of_node = np.full(m.n_nodes, -1)
        for i, (fx, fy) in enumerate(FEET):
            sel = (np.abs(xyz[:, 0] - fx) <= WIDTH / 2 + 1e-6) & (np.abs(xyz[:, 1] - fy) <= WIDTH / 2 + 1e-6)
            self.foot_of_node[sel] = i
        self.foot_top = [np.flatnonzero((self.foot_of_node == i) & (np.abs(xyz[:, 2] - HEIGHT) < 1e-6))
                         for i in range(4)]
        # the payload as a rigid body: plate, column and robot (feet left out)
        pay = self.group >= 4
        pts = m.node_xyz[m.enode[pay]].mean(axis=1)
        vol = m.pitch ** 3
        pts = np.vstack([pts, [[*ROBOT["at"], ROBOT["height"]]]])
        kg = np.concatenate([np.full(pay.sum(), AL["density"] * vol * 1e-6), [ROBOT["mass"]]])
        self.mass, self.cg, self.inertia = rigid_body(pts, kg)
        self.st = Stiffness(m, 0.3, self.fixed)
        self.st_free = Stiffness(m, 0.3, np.zeros(m.n_dof, bool))

    def props(self, rho):
        rho = np.asarray(rho, float)
        e = np.full(self.mesh.n_elem, AL["E"])
        d = np.full(self.mesh.n_elem, AL["density"])
        for i in range(4):
            sel = self.group == i
            e[sel] = self.mat.E * sheet_modulus(KIND, rho[i])
            d[sel] = self.mat.density * rho[i]
        return e, d

    def analyse(self, rho, eta, modes=14):
        e, d = self.props(rho)
        res = modal_analysis(self.mesh, e, d, 0.3, self.fixed, modes, [(self.robot, ROBOT["mass"])])
        # static: gravity on everything
        f = np.zeros(self.mesh.n_dof)
        f[2::3] = -res.mass_vector[2::3] * G
        f[self.fixed] = 0.0
        u = spsolve(self.st.assemble(e).tocsc(), f)
        react = self.st_free.assemble(e) @ u
        rz = react[3 * self.bottom + 2]
        loads = np.array([rz[self.foot_of_node[self.bottom] == i].sum() for i in range(4)])
        sag = np.array([-u[3 * nt + 2].mean() for nt in self.foot_top])
        pt = self.mesh.node_xyz[self.plate_top]
        a = np.c_[np.ones(len(pt)), pt[:, :2]]
        coef = np.linalg.lstsq(a, u[3 * self.plate_top + 2], rcond=None)[0]
        tilt = float(np.hypot(coef[1], coef[2]) * 1e3)                        # mrad
        # transmissibility to the robot
        fr = np.linspace(1.0, BAND[1], 800)
        rz_dofs, rx_dofs = 3 * self.robot + 2, 3 * self.robot
        t_zz = base_transmissibility(res, rz_dofs, 2, fr, eta, self.fixed)
        t_zx = base_transmissibility(res, rx_dofs, 2, fr, eta, self.fixed)      # vertical shake -> sideways
        t_xx = base_transmissibility(res, rx_dofs, 0, fr, eta, self.fixed)
        t_xz = base_transmissibility(res, rz_dofs, 0, fr, eta, self.fixed)      # sideways shake -> vertical
        band = fr >= BAND[0]

        def start(t):
            over = np.flatnonzero(t > 1.0)
            return round(float(fr[over[-1] + 1]), 1) if len(over) else 1.0

        zf = res.participation[:, 2]
        out = {
            "density": [round(float(r), 3) for r in rho],
            "frequencies": [round(float(x), 1) for x in res.frequencies[:6]],
            "bounce_Hz": round(float(res.frequencies[int(np.argmax(zf))]), 1),
            "bounceMassFraction": round(float(zf.max()), 3),
            "footLoad_N": [round(float(x), 2) for x in loads],
            "footSag_mm": [round(float(x), 3) for x in sag],
            "sagSpread_mm": round(float(np.ptp(sag)), 3),
            "tilt_mrad": round(tilt, 2),
            "shockMargin_g": round(shock_margin(KIND, rho, self.mat.E, WIDTH, loads), 2),
            "isolationFrom_Hz": {"z": start(t_zz), "x": start(t_xx)},
            "worstInBand": {"zz": round(float(t_zz[band].max()), 3), "xx": round(float(t_xx[band].max()), 3),
                            "zx": round(float(t_zx[band].max()), 3), "xz": round(float(t_xz[band].max()), 3)},
            "peak": {"zz": round(float(t_zz.max()), 2), "zx": round(float(t_zx.max()), 2),
                     "xx": round(float(t_xx.max()), 2), "xz": round(float(t_xz.max()), 2)},
            "at100Hz": {"zz": round(float(np.interp(100, fr, t_zz)), 3), "zx": round(float(np.interp(100, fr, t_zx)), 3)},
            "footMass_g": round(float(sum(self.mat.density * r * WIDTH ** 2 * HEIGHT * 1e-3 for r in rho)), 1),
        }
        curves = {"f": [round(float(x), 2) for x in fr[::4]]}
        for k, t in (("zz", t_zz), ("zx", t_zx), ("xx", t_xx), ("xz", t_xz)):
            curves[k] = [round(float(x), 4) for x in t[::4]]
        return out, curves, sag


def feet_stl(rho, path):
    import trimesh

    from loadpath.fields import FieldGraph

    parts = []
    for (fx, fy), r in zip(FEET, rho):
        doc = {"nodes": {"env": {"op": "shape", "shape": {"type": "box", "min": [-WIDTH / 2, -WIDTH / 2, 0],
                                                            "max": [WIDTH / 2, WIDTH / 2, HEIGHT]}},
                         "core": {"op": "lattice", "type": KIND, "cell": CELL, "density": {"op": "constant",
                                                                                           "value": float(r)}},
                         "plates": {"op": "union", "inputs": [
                             {"op": "shape", "shape": {"type": "box", "min": [-WIDTH / 2, -WIDTH / 2, 0],
                                                       "max": [WIDTH / 2, WIDTH / 2, 1.0]}},
                             {"op": "shape", "shape": {"type": "box", "min": [-WIDTH / 2, -WIDTH / 2, HEIGHT - 1.0],
                                                       "max": [WIDTH / 2, WIDTH / 2, HEIGHT]}}]},
                         "part": {"op": "intersect", "inputs": [{"op": "union", "inputs": ["core", "plates"]}, "env"]}},
               "output": "part"}
        mesh = FieldGraph(doc).mesh(pitch=0.25)
        mesh.apply_translation([fx, fy, 0])
        parts.append(mesh)
    trimesh.util.concatenate(parts).export(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="isolator-out")
    ap.add_argument("--pitch", type=float, default=2.0)
    ap.add_argument("--no-stl", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    mat = material_from_spec("tpu")
    eta = loss_factor(mat)
    mt = Mount(a.pitch, mat)
    rho_min = float(relative_density(KIND, MIN_WALL, CELL))
    rho_max = 0.45
    k_total = mt.mass * (2 * np.pi * F_TARGET) ** 2 / 1e3                  # N/mm
    print(f"payload {mt.mass:.3f} kg, CG {np.round(mt.cg, 1)}, K {k_total:.1f} N/mm, rho >= {rho_min:.3f}", flush=True)
    kmin, kmax = (foot_stiffness(KIND, r, mat.E, WIDTH, HEIGHT)[0] for r in (rho_min, rho_max))
    feet3 = np.c_[FEET, np.full(4, HEIGHT)]

    rows, curves = {}, {}
    # uniform
    ku = np.full(4, k_total / 4)
    rho_u = density_for_stiffness(KIND, ku, mat.E, WIDTH, HEIGHT)
    # graded: elastic centre under the CG
    kg, resid = tune_mount(mt.cg[:2], FEET, k_total, kmin, kmax)
    designs = {"uniform feet": rho_u}
    rho_g = density_for_stiffness(KIND, kg, mat.E, WIDTH, HEIGHT)
    for it in range(3):
        name = "graded feet" if it == 2 else f"graded feet (pass {it})"
        out, cv, sag = mt.analyse(rho_g, eta)
        if it < 2:
            print(name, "sag spread", out["sagSpread_mm"], "mm", flush=True)
            # refine against the FE: a foot that sags more than the mean gets stiffer
            kz = foot_stiffness(KIND, rho_g, mat.E, WIDTH, HEIGHT)[0] * sag / sag.mean()
            kz *= k_total / kz.sum() if it == 0 else 1.0
            rho_g = np.clip(density_for_stiffness(KIND, kz, mat.E, WIDTH, HEIGHT), rho_min, rho_max)
        else:
            rows[name], curves[name] = out, cv
    designs["graded feet"] = rho_g
    out, cv, _ = mt.analyse(rho_u, eta)
    rows["uniform feet"], curves["uniform feet"] = out, cv
    # the rigid-body model for the same feet, against the FE
    for name, rho in designs.items():
        kz, ks = foot_stiffness(KIND, rho, mat.E, WIDTH, HEIGHT)
        fr, _, mf = mount_modes(mt.mass, mt.cg, mt.inertia, feet3, kz, ks)
        rows[name]["rigidModel_Hz"] = [round(float(x), 1) for x in fr]
        rows[name]["rigidBounce_Hz"] = round(float(fr[int(np.argmax(mf[:, 2]))]), 1)
    for k, v in rows.items():
        print(k, json.dumps(v))
    if not a.no_stl:
        feet_stl(rho_g, os.path.join(a.out, "isolator-feet-graded.stl"))
    summary = {"payload_kg": round(mt.mass, 3), "cg_mm": [round(float(x), 1) for x in mt.cg],
               "targetBounce_Hz": F_TARGET, "lossFactor": eta, "rhoMin": round(rho_min, 3),
               "tuneResidual_mm": round(resid, 3), "band_Hz": BAND, "rows": rows, "curves": curves,
               "seconds": round(time.time() - t0)}
    with open(os.path.join(a.out, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=1)


if __name__ == "__main__":
    main()
