"""
Graded lattice against solid topology optimization where lattices should win:
stopping an impact. The case is end-stop.json: a tapered corner bumper for a
GRIP-like robot base that has to stop the 5 kg robot falling 0.9 m (44 J)
without putting more than 6 kN into the base.

    python examples/impact/end_stop.py [--out impact-out/] [--cells 40000]

Four designs of the same envelope:

- graded gyroid: the impact block's own grading (every layer crushes at the
  same force), built as a field graph and re-analysed from the graph's
  density field;
- uniform gyroid of the same mass;
- solid topology optimization of the same envelope at the same mass,
  designed for the 6 kN limit load (what a stiffness-driven generative
  design tool gives you), verified by voxel FEA;
- (for the vibration side) the first natural frequency of the robot
  sitting on each pad.

Also checks the layer model's first collapse load in 3D (elastic
compensation on the homogenized pad) and writes the graded part's STL,
the crush curves and summary.json.
"""
import argparse
import copy
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))

from loadpath.fem import HexMesh  # noqa: E402
from loadpath.fields import FieldGraph  # noqa: E402
from loadpath.impact import (face_load, impact_report, layer_profile, limit_load, plateau_stress,  # noqa: E402
                             sample_core, solid_impact)
from loadpath.lattice import sheet_modulus  # noqa: E402
from loadpath.materials import material_from_spec  # noqa: E402
from loadpath.modal import vibration_report  # noqa: E402


def collapse_3d(graph, spec, mat, pitch=1.0):
    """First collapse load (N) of the homogenized pad in 3D: bottom face held,
    the top face pushed down evenly. Each cell: modulus E C rho^n, strength
    its plateau stress."""
    kind = spec["lattice"]
    inside, rho, lo, h = sample_core(graph, spec["envelope"], spec.get("density"), pitch)
    mesh = HexMesh(np.argwhere(inside), h, lo)
    r = rho[inside]
    z = mesh.node_xyz[:, 2]
    fixed = np.zeros(mesh.n_dof, bool)
    bot = np.flatnonzero(z < lo[2] + 1e-6)
    fixed[3 * bot + 2] = True
    fixed[3 * bot[0]] = fixed[3 * bot[0] + 1] = True            # hold it still sideways
    fixed[3 * bot[-1] + 1] = True
    f = face_load(mesh, 2, -1.0)                               # 1 N in all, as an even pressure
    res = limit_load(mesh, mat.E * sheet_modulus(kind, r), plateau_stress(kind, r, mat.strength), mat.nu, fixed, f,
                     iters=10)
    return res["loadFactor"], res


def solid_to(env_stl, vf, force, cells, out):
    from loadpath.spec import study_from_dict
    from loadpath.study import run_study
    from loadpath.verify import verify_part

    d = {"name": "end-stop-solid",
         "designSpace": [{"type": "mesh", "path": env_stl}],
         "preserve": [{"id": "top", "shape": {"type": "box", "min": [-11, -11, 19], "max": [11, 11, 20]}},
                      {"id": "foot", "shape": {"type": "box", "min": [-20, -20, 0], "max": [20, 20, 1]}}],
         "supports": [{"region": "foot", "fix": "all"}],
         "loadCases": [{"name": "impact", "loads": [{"region": "top", "force": [0, 0, -force]}]}],
         "material": "pa12", "process": "sls",
         "goal": {"volumeFraction": vf},
         "resolution": {"maxElements": cells}}
    study = study_from_dict(d, base_dir=out)
    res = run_study(study)
    o = res.outcomes[res.best if res.best is not None else 0]
    o.mesh.export(os.path.join(out, "end-stop-solid-to.stl"))
    rep = verify_part(study, o.mesh, max_cells=cells)
    case = rep["cases"][0]
    k = force / case["deflection"]
    s = study.material.strength / (rep["safetyFactor"] * force)    # MPa per N
    return o.mesh, rep, k, s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="impact-out")
    ap.add_argument("--cells", type=int, default=40_000)
    ap.add_argument("--no-to", action="store_true", help="skip the solid topology optimization")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    doc = json.loads((HERE / "end-stop.json").read_text())
    mat = material_from_spec(doc["material"])
    spec = doc["impact"]
    rows = {}
    t0 = time.time()

    # 1. grade: the impact block on the starting graph gives the table
    g0 = FieldGraph(doc, base_dir=str(HERE))
    first = impact_report(g0, spec, mat)
    graded = first["graded"]
    print("grading:", {k: graded[k] for k in ("ok", "plateauForce", "coreMass", "rhoMin")}, flush=True)

    # 2. the graded part as a field graph, re-analysed from its own field
    gdoc = copy.deepcopy(doc)
    gdoc["nodes"]["density"] = graded["density"]
    gg = FieldGraph(gdoc, base_dir=str(HERE))
    rg = impact_report(gg, {**spec, "grade": False}, mat)
    rows["graded gyroid"] = {"mass_g": rg["coreMass"], **rg["response"]}

    # 3. a uniform core of the same mass
    udoc = copy.deepcopy(doc)
    udoc["nodes"]["density"] = {"op": "constant", "value": round(rg["meanDensity"], 4)}
    gu = FieldGraph(udoc, base_dir=str(HERE))
    ru = impact_report(gu, {**spec, "grade": False}, mat)
    rows["uniform gyroid"] = {"mass_g": ru["coreMass"], "density": round(rg["meanDensity"], 4), **ru["response"]}

    # 4. 3D check of the layer model's first collapse
    for name, graph in (("graded gyroid", gg), ("uniform gyroid", gu)):
        lam, res = collapse_3d(graph, spec, mat)
        inside, rho, lo, h = sample_core(graph, spec["envelope"], spec.get("density"), 0.25)
        area, mean = layer_profile(inside, rho, h)
        lf = (plateau_stress(spec["lattice"], rho, mat.strength) * inside).sum(axis=(0, 1)) * h ** 2
        f1d = float(lf[area > 0].min())
        rows[name]["firstCollapse3D"] = round(lam, 1)
        rows[name]["collapse3DUpper"] = round(res["upperBound"], 1)
        rows[name]["firstCollapse1D"] = round(f1d, 1)
        print(name, "collapse 3D between", round(lam), "and", round(res["upperBound"]), "N, layer model",
              round(f1d), "N", flush=True)

    # the envelope as a mesh: the TO design space, and the volume for the fraction
    env = g0.mesh(pitch=0.5, name="envelope")
    env_stl = os.path.join(os.path.abspath(a.out), "end-stop-envelope.stl")
    env.export(env_stl)
    vf = rg["coreMass"] / (float(env.volume) * mat.density / 1000.0)

    # 5. solid topology optimization at the same mass, for the 6 kN limit load
    if not a.no_to:
        mesh, rep, k, s = solid_to(env_stl, round(vf, 3), spec["forceLimit"], a.cells, os.path.abspath(a.out))
        si = solid_impact(k, s, mat.strength, rg["energy"], spec["forceLimit"], spec["mass"])
        # it cannot dissipate anything without yielding, so its peak is at least the yield load
        fy = mat.strength / s
        si["peakForceAtLeast"] = round(min(si["peakForce"], fy) if si["yields"] else si["peakForce"], 1)
        rows["solid TO"] = {"mass_g": round(rep["mass"], 2), "volumeFraction": round(vf, 3),
                            "safetyFactorAt6kN": round(rep["safetyFactor"], 2), **si}
        print("solid TO", rows["solid TO"], flush=True)

    # 6. vibration: the robot (5 kg) sitting on each pad
    top = {"op": "shape", "shape": {"type": "box", "min": [-12, -12, 19.5], "max": [12, 12, 21]}}
    base = {"op": "shape", "shape": {"type": "box", "min": [-21, -21, -1], "max": [21, 21, 0.5]}}
    for name, d in (("graded gyroid", gdoc), ("uniform gyroid", udoc)):
        vd = copy.deepcopy(d)
        vd["nodes"].update(top=top, base=base)
        vg = FieldGraph(vd, base_dir=str(HERE))
        vr = vibration_report(vg, {"envelope": "envelope", "density": "density", "lattice": spec["lattice"],
                                   "fixed": "base", "pointMasses": [{"region": "top", "mass": spec["mass"]}],
                                   "modes": 3, "pitch": 1.0}, mat)
        rows[name]["f1_Hz"] = vr["frequencies"][0]

    # the graded part itself
    part = gg.mesh(pitch=doc["mesh"]["pitch"])
    part.export(os.path.join(a.out, "end-stop-graded.stl"))
    rows["graded gyroid"]["meshVolume"] = round(float(part.volume), 1)
    gu.mesh(pitch=doc["mesh"]["pitch"]).export(os.path.join(a.out, "end-stop-uniform.stl"))
    curves = {"graded gyroid": rg["curve"], "uniform gyroid": ru["curve"]}
    out = {"case": doc["description"], "energy_J": round(rg["energy"] / 1000, 2), "forceLimit_N": spec["forceLimit"],
           "rows": rows, "curves": curves, "gradedDensity": graded["density"], "seconds": round(time.time() - t0)}
    with open(os.path.join(a.out, "summary.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    for k, v in rows.items():
        print(k, json.dumps(v))


if __name__ == "__main__":
    main()
