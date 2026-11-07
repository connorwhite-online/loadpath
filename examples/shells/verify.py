"""
Run both halves of a split (`loadpath split`, or handheld.py) and check the
parts as built: they sit together with no overlap, the cover lifts off its
base (its snap hooks taken out: they bend past the ledges), and the first
snap's strain read on cut cells against the beam formula.

    python examples/shells/verify.py /tmp/handheld/split [--quick]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))


def delivered(run_dir: str) -> str:
    """The best outcome's mesh from a `loadpath run` folder."""
    rep = os.path.join(run_dir, "report.json")
    if os.path.isfile(rep):
        r = json.load(open(rep))
        for k in ("best", "bestMesh", "mesh"):
            v = r.get(k)
            if isinstance(v, str) and v.endswith(".stl"):
                return v if os.path.isabs(v) else os.path.join(run_dir, v)
    stls = sorted(glob.glob(os.path.join(run_dir, "*.stl")), key=os.path.getmtime)
    if not stls:
        raise FileNotFoundError(f"no STL in {run_dir}")
    return stls[-1]


def _against(normal):
    """The opposite direction: '+z' -> '-z', a vector negated."""
    if isinstance(normal, str):
        return ("-" if normal.startswith("+") else "+") + normal.lstrip("+-")
    return [-float(x) for x in normal]


def main(argv=None):
    import trimesh

    from loadpath.shells import check_closed, verify_snap
    ap = argparse.ArgumentParser()
    ap.add_argument("split_dir")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args(argv)
    d = args.split_dir
    rep = json.load(open(os.path.join(d, "split.json")))
    studies = sorted(glob.glob(os.path.join(d, "*.study.json")))
    meshes = {}
    for sp in studies:
        side = "base" if sp.endswith("-base.study.json") else "cover"
        out = os.path.join(d, f"run-{side}")
        cmd = [sys.executable, "-m", "loadpath", "run", sp, "--out", out, "--no-render", "--viewer", "none"]
        if args.quick:
            cmd.append("--quick")
        p = subprocess.run(cmd, capture_output=True, text=True)
        if p.returncode not in (0, 1):
            print(p.stdout[-2000:], p.stderr[-2000:])
            raise SystemExit(f"run of the {side} failed")
        meshes[side] = delivered(out)
    B = trimesh.load(meshes["base"], force="mesh")
    C = trimesh.load(meshes["cover"], force="mesh")
    hooks = [g["hook"] for g in rep.get("snapsGeometry") or [] if (rep["snaps"] or {}).get("hookOn") == "cover"]
    # hooks on the base: take them out of the base instead
    if (rep.get("snaps") or {}).get("hookOn") == "base":
        out = check_closed(C, B, _against(rep["cut"]["normal"]), hooks=[g["hook"] for g in rep["snapsGeometry"]])
    else:
        out = check_closed(B, C, rep["cut"]["normal"], hooks=hooks)
    res = {"meshes": meshes, "closed": out, "volume": {"base": round(float(B.volume), 1),
                                                       "cover": round(float(C.volume), 1)}}
    if rep.get("snapsGeometry"):
        sn = rep["snaps"]
        v = verify_snap({**rep["snapsGeometry"][0], "deflection": sn["deflection"]},
                        json.load(open(studies[0])).get("material", "pla"))
        res["snap"] = {"formula": sn["strain"], "cutCells": round(v["strain"], 5),
                       "ratio": round(v["strain"] / sn["strain"], 3), "allowable": sn["allowable"],
                       "forceCutCells": round(v["force"], 2), "forceFormula": sn["deflectionForce"]}
    print(json.dumps(res, indent=1, default=float))
    json.dump(res, open(os.path.join(d, "verify.json"), "w"), indent=1, default=float)
    return res


if __name__ == "__main__":
    main()
