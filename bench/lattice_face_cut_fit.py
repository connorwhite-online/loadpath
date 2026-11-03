"""
How much stiffness a lattice keeps where a bare face cuts it, case by case.

Every skinless lattice part with a full-resolution result on file (the bars
of bench/results/lattice-aniso*.json, the as-laid/turned pairs of
lattice-orient-confirm*.json) is read homogenized at several free-surface
knockdowns; per case and load the knockdown that matches the resolved
compliance is interpolated, alongside how square the faces cut the cells
(homogenized.face_alignment). A knockdown model that depends on the cut is
fitted on these.

    python bench/lattice_face_cut_fit.py [--cells 30000] [--knockdowns 0,0.25,0.5,0.75,0.9]

Writes bench/results/lattice-face-cut.json.
"""
import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "bench"))

import lattice_aniso_validate as lav
from lattice_orient_confirm import designs

from loadpath.homogenized import homogenized_check
from loadpath.spec import study_from_dict

RES = ROOT / "bench" / "results"


def cases():
    """(name, doc, study dict, resolved cases [{name, compliance, peak}])"""
    out = []
    for f in ("lattice-aniso.json", "lattice-aniso-skin.json", "lattice-aniso-beams.json"):
        for name, row in json.loads((RES / f).read_text()).items():
            if "skin" in name or name not in lav.CASES or "resolved" not in row:
                continue
            doc, sd = lav.CASES[name]()
            out.append((name, doc, sd, [{"name": c["name"], "compliance": c["compliance"], "peak": c["cellPeak"]}
                                        for c in row["resolved"]["cases"]]))
    seen = set()
    for f in ("lattice-orient-confirm.json", "lattice-orient-confirm-lbracket.json"):
        for name, row in json.loads((RES / f).read_text()).items():
            laid, turned, sd, _ = designs(name)
            for tag, doc in (("asLaid", laid), ("turned", turned)):
                key = f"{name.split(':')[0]}/asLaid" if tag == "asLaid" else f"{name}/turned"
                if key in seen:
                    continue
                seen.add(key)
                out.append((key, doc, sd, [{"name": c["name"], "compliance": c["compliance"],
                                            "peak": c.get("cellPeak", c.get("peakStress"))}
                                           for c in row[tag]["resCases"]]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", type=int, default=30000)
    ap.add_argument("--knockdowns", default="0,0.25,0.5,0.75,0.9")
    ap.add_argument("--only", default="")
    ap.add_argument("--grid", default="", help="square:bias pairs (e.g. 0.2:0.5,0.2:1.0): read each case under "
                                               "the cut-angle model with those parameters instead of the sweep")
    ap.add_argument("--out", default=str(RES / "lattice-face-cut.json"))
    a = ap.parse_args()
    ks = [float(x) for x in a.knockdowns.split(",")]
    out = json.loads(Path(a.out).read_text()) if Path(a.out).exists() else {}
    for name, doc, sd, res in cases():
        if a.only and not any(o in name for o in a.only.split(",")):
            continue
        st = study_from_dict(sd)
        cell = float(doc["nodes"]["lat"]["cell"])
        t0 = time.time()
        rows = []
        grid = [None if p == "model" else tuple(float(v) for v in p.split(":")) for p in a.grid.split(",")] \
            if a.grid else None
        if grid:
            row = out.get(name, {}).get("grid", {})
            for par in grid:
                h = homogenized_check(copy.deepcopy(doc), st, cells=a.cells, contact_band=cell, face_cut=par)
                row["model" if par is None else ":".join(f"{v:g}" if len(par) > 2 else str(v) for v in par)] = [{"name": c["name"], "compliance": c["compliance"] / r["compliance"],
                                        "peak": c["peakStress"] / r["peak"]} for c, r in zip(h["cases"], res)]
            out.setdefault(name, {})["grid"] = row
            print(name, f"({time.time() - t0:.0f} s)", flush=True)
            Path(a.out).write_text(json.dumps(out, indent=1))
            continue
        for k in ks:
            h = homogenized_check(copy.deepcopy(doc), st, cells=a.cells, contact_band=cell, free_surface=k)
            rows.append({"knockdown": k, "cases": [{"name": c["name"], "compliance": c["compliance"],
                                                    "peak": c["peakStress"]} for c in h["cases"]],
                         "latticeExposed": h["latticeExposed"], "faceAlignment": h.get("faceAlignment")})
        fit = []
        for ci, rc in enumerate(res):
            c = np.array([r["cases"][ci]["compliance"] for r in rows])
            # compliance rises with the knockdown: the knockdown that matches
            kk = float(np.interp(rc["compliance"], c, ks, left=np.nan, right=np.nan)) if np.all(np.diff(c) > 0) \
                else None
            fit.append({"name": rc["name"], "resCompliance": rc["compliance"], "resPeak": rc["peak"],
                        "knockdownMatch": kk,
                        "compliance": (c / rc["compliance"]).round(3).tolist(),
                        "peak": [round(r["cases"][ci]["peak"] / rc["peak"], 3) for r in rows]})
        out[name] = {**out.get(name, {}), "sweep": rows, "fit": fit, "seconds": time.time() - t0}
        print(name, " | ".join(f"{f['name']} k*={f['knockdownMatch']}" for f in fit),
              f"({time.time() - t0:.0f} s)", flush=True)
        Path(a.out).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
