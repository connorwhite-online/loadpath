"""
The touch microphone's push pad with a graded lattice under the skin
instead of ribs, read back on the printable mesh (docs/SWITCH.md, "A graded
lattice instead of ribs").

    python examples/switch/touch_mic_lattice.py --stl <dir> --out tm-lattice --type bcc --cell 4
    python examples/switch/touch_mic_lattice.py --stl <dir> --readback part.stl [--pitch 0.33]

--stl is the directory `touch_mic.py --stl-only` writes (the eval case's
STLs; the product's CAD is not in this repository). The first form runs
switch_lattice.run_switch_lattice: optimize the lattice's density per cell
on the 1 mm homogenized model, mesh the exact lattice joined to the skin,
read the fire forces back on that mesh (cut cells, 0.5 mm), divide each
press's window by how far the reading was off, and go again (--rounds).
The second reads any delivered mesh (ribs, round 17, a lattice) the same
way, so every design is compared on the same footing.

Results on round 17's exterior (0.33 mm cut-cell read-back, 14 presses):
the bare 1.0 mm skin meets 9; ribs as `loadpath switch` delivers them, 0;
ribs and a BCC 4 mm lattice after the read-back correction, 8 each, the
lattice 0.5 cm^3 lighter and clear of every assembly path. The four rim
presses the skin alone puts over their 300 gf ceiling stay over in every
design.
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
CASE = HERE.parents[1] / "loadpath" / "evals" / "cases" / "touch-mic-pad.json"


def study(stl_dir, lattice=None):
    from loadpath.spec import study_from_dict
    raw = json.load(open(CASE))
    doc = {k: v for k, v in raw.items() if k not in ("expect", "assets")}
    if lattice:
        doc["switch"] = {**doc["switch"], "lattice": lattice}
    return study_from_dict(doc, base_dir=os.path.abspath(stl_dir))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stl", required=True, help="the touch-mic-pad case's STLs (touch_mic.py --stl-only)")
    ap.add_argument("--out", default="touch-mic-lattice")
    ap.add_argument("--type", default="bcc")
    ap.add_argument("--cell", type=float, default=4.0)
    ap.add_argument("--mode", default="where-needed", choices=["fill", "where-needed"])
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--iterations", type=int, default=80)
    ap.add_argument("--readback", help="read this mesh back instead of designing")
    ap.add_argument("--pitch", type=float, default=0.33, help="mm, the read-back grid")
    a = ap.parse_args()
    from loadpath.switch_lattice import LatticeOptions, readback, run_switch_lattice
    if a.readback:
        import trimesh
        rb = readback(study(a.stl), trimesh.load(a.readback, force="mesh"), a.pitch)
        print(json.dumps(rb, indent=1))
        return
    st = study(a.stl)

    def note(ev):
        if ev.get("stage") != "optimize" or ev["iteration"] % 20 == 0:
            print(json.dumps({k: v for k, v in ev.items() if k != "forcesGf"}, default=float), flush=True)

    rep = run_switch_lattice(st, a.out, LatticeOptions(type=a.type, cell=a.cell, mode=a.mode), rounds=a.rounds,
                             readback_pitch=0.5, final_pitch=a.pitch, iterations=a.iterations, on_progress=note)
    json.dump(rep, open(os.path.join(a.out, "report.json"), "w"), indent=1, default=float)
    for r in rep["rounds"]:
        print(f"round {r['round']}: design {r['designInWindow']}/14, read back {r['readback']['inWindow']}/14, "
              f"{r['volume'] / 1000:.2f} cm^3")
    print("final read-back", rep["inWindow"], np.round(rep["finalReadback"]["forceGf"]).astype(int).tolist())
    print("powder", rep["powder"], "paths (mm^3 added)", rep["paths"])


if __name__ == "__main__":
    main()
