"""
GRIP v0.3's five structural links, regrown and checked against what GRIP
recorded when they were made.

The studies in studies/ are copied verbatim from GRIP
(github.com/connorwhite-online/grip, branch generative-redesign,
generative/v3/studies/, written by generative/v3.py). They are a real arm's
links with real interfaces: servo cradles, horn hubs, cycloidal-drive
sockets, a counterbalance spring, hex-driver access to every bolt.

    python examples/grip/run.py                      # all five, regrow + check
    python examples/grip/run.py --links wrist_yoke   # one
    python examples/grip/run.py --verify-only DIR    # check STLs already in DIR

For each link: run the study, then re-verify the delivered mesh on a 150k-cell
grid exactly as the study's confirm stage does and draw its interfaces close up
(render.render_interfaces). Writes <out>/<link>.stl, <link>-interfaces.png and
<out>/scorecard.json, and prints a table against GRIP's recorded v0.3 numbers.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
LINKS = ["turret", "upper_arm", "forearm", "wrist_yoke", "wrist_link"]

# What GRIP recorded for v0.3 (generative/README.md and v3/out/summary.json at
# 3689f8d): mass in g and the study's own safety factor at delivery.
RECORDED = {
    "turret": (236.3, 2.41), "upper_arm": (178.6, 2.81), "forearm": (77.6, 4.19),
    "wrist_yoke": (24.3, 4.41), "wrist_link": (67.2, 4.13),
}


def _study(link):
    from loadpath.spec import study_from_dict
    d = os.path.join(HERE, "studies")
    return study_from_dict(json.load(open(os.path.join(d, f"{link}.json"))), base_dir=d)


def _solve(link):
    os.environ.setdefault("LOADPATH_WORKERS", "1")
    from loadpath.study import run_study
    t = time.time()
    r = run_study(_study(link))
    if r.best is None:
        return link, None, time.time() - t
    return link, r.outcomes[r.best].mesh, time.time() - t


def check(link, mesh, out):
    from loadpath.render import render_interfaces
    from loadpath.verify import surface_thickness, verify_part
    from loadpath.domain import choose_pitch
    st = _study(link)
    render_interfaces(mesh, st, os.path.join(out, f"{link}-interfaces.png"))
    # the study's own confirm stage: 150k cells, stress within one
    # optimization cell of a load or support left out of the peak
    pitch = choose_pitch(st)
    rep = verify_part(st, mesh, max_cells=150_000, min_pitch=0.3 * pitch, contact_band=pitch)
    p1, p5 = surface_thickness(mesh)
    return {
        "g": round(rep["mass"], 1), "sf": round(rep["safetyFactor"], 2),
        "p1RayMm": round(p1, 2), "p5RayMm": round(p5, 2), "minMember": st.min_member,
        "creaseMm": round(rep.get("creaseLength") or 0.0, 1),
        "boundaryCutMm2": round(rep.get("boundaryCutArea") or 0.0, 1),
        "keepOutIntrusionMm3": round(rep["keepOutIntrusion"], 3),
        "preserveMissingMm3": round(rep["preserveMissing"], 3),
        "closedBores": [b["id"] for b in rep.get("closedBores") or []],
        "bodies": rep["bodies"],
        "cases": {c["name"]: {"peakMPa": round(c["peakStress"], 1), "peakAt": [round(v, 1) for v in c["peakAt"]],
                              "deflMm": round(c["deflection"], 3)} for c in rep["cases"]},
    }


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--links", default=",".join(LINKS))
    ap.add_argument("--out", default="grip-out")
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--verify-only", metavar="DIR", help="check <DIR>/<link>.stl instead of regrowing")
    args = ap.parse_args(argv)
    links = args.links.split(",")
    os.makedirs(args.out, exist_ok=True)
    import trimesh
    meshes, secs = {}, {}
    if args.verify_only:
        for l in links:
            meshes[l] = trimesh.load(os.path.join(args.verify_only, f"{l}.stl"))
    else:
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(args.jobs, mp_context=mp.get_context("spawn")) as ex:
            for l, mesh, s in ex.map(_solve, links):
                secs[l] = round(s)
                if mesh is not None:
                    # check the file that ships: the export drops degenerate
                    # faces the in-memory mesh can still carry
                    mesh.export(os.path.join(args.out, f"{l}.stl"))
                    meshes[l] = trimesh.load(os.path.join(args.out, f"{l}.stl"))
    card = {}
    for l in links:
        if l not in meshes:
            card[l] = {"error": "no outcome met the goal"}
        else:
            card[l] = check(l, meshes[l], args.out)
        if l in secs:
            card[l]["seconds"] = secs[l]
        json.dump(card, open(os.path.join(args.out, "scorecard.json"), "w"), indent=1)
    print(f"{'link':<11} {'recorded':>15} {'now':>15} {'p5 ray':>7} {'crease':>7} {'cut':>7}")
    for l in links:
        c, (g0, sf0) = card[l], RECORDED[l]
        if "error" in c:
            print(f"{l:<11} {g0:>7.1f} g {sf0:>5.2f}   {c['error']}")
            continue
        print(f"{l:<11} {g0:>7.1f} g {sf0:>5.2f} {c['g']:>7.1f} g {c['sf']:>5.2f} {c['p5RayMm']:>7} "
              f"{c['creaseMm']:>7} {c['boundaryCutMm2']:>7}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
