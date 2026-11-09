"""The robot arm, SLS edition: each v0.3 link rebuilt as a field graph that leans on
what powder-bed nylon allows and FDM does not.

The v0.3 links (density topology optimization, sized for SLS PA12-GF but
solid) give the load path. Here each one is inflated into a smooth organic
envelope inside its own design space, then hollowed: a stress-graded skin
over a sparse gyroid core that holds the skin up (no thin sheet left to
wrinkle), solid round every interface and bolt, and the powder let out
through drain holes placed where the stress is lowest. Nothing in it could
be printed on FDM without support trapped inside; on SLS it needs none.

    python examples/robot-arm/sls.py graph --links forearm --out sls-work
    loadpath tune sls-work/forearm/graph.json --out sls-work/forearm/tune

First result (2026-11-01, forearm, PA12, derived loads, homogenized on 2M
cells): the best sandwich found reads SF 1.99 at 90.2 g, against 2.25 at
77.5 g for the solid v0.3 forearm on the same check. The forearm's peak sits
at the J3 hub and the member roots beside it, which hollowing does not
relieve; a sandwich pays on long stiffness-bound members (the arm link),
not on a compact strength-bound link like this one.

Kept where it wins (meshed at 0.35-0.4 mm, drained, PA12, 150k voxels / tet):
upper arm 185.8 g, SF 1.76 / 2.01 (v0.3 184.5 g, 1.81 / 1.66); wrist link
49.5 g, 3.66 / 3.73 (v0.3 43.3 g, 3.39 / 1.63). The homogenized search read
the wrist link 18% light (40.7 g): drain bosses and filled specks add mass.
The tune's 1-2M-cell confirms took 47-52 min each on a 4-core box and
rejected passing candidates over stray voxel bodies; mesh and check instead.

Interfaces stay as the study wrote them: within `margin` of every preserved
region the part is the v0.3 part, unchanged, and every keep-out (bolts,
drivers, servo bodies, the neighbours' sweeps) is cut exactly at the end.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
LINKS = ("wrist_yoke", "wrist_link", "forearm", "upper_arm", "turret")
CYCLES = {"dynamic": 1e5, "pose": 1e5, "push": 1e4, "stall": 1e3}
# the v0.3 parts as built (the arm repository's generative/v3/out); set ARM_V3_OUT
V3 = os.environ.get("ARM_V3_OUT", os.path.join("..", "arm", "generative", "v3", "out"))
MATERIAL = "pa12"


def study_doc(link):
    """The v0.3 study with the loads derived from the arm (as v0.3 was built)."""
    st = json.load(open(os.path.join(HERE, "studies", f"{link}.json")))
    der = json.load(open(os.path.join(HERE, "derived", f"{link}.derived.json")))
    st["loadCases"] = der["loadCases"]
    for c in st["loadCases"]:
        c.setdefault("cycles", CYCLES.get(c["name"].split("-")[2], 1e5))
    st.pop("chain", None)
    st["material"] = MATERIAL
    st["process"] = "sls"
    return st


def _union(shapes):
    return shapes[0] if len(shapes) == 1 else {"type": "union", "shapes": shapes}


def graph(link, study_path, v3_stl, offset=2.5, skin=(1.0, 2.5), wall=(0.8, 1.2), cell=10.0, margin=3.0):
    st = json.load(open(study_path))
    sdir = os.path.dirname(study_path)
    ds = [s if "type" in s else s["shape"] for s in st["designSpace"]]
    ko_exact = [k["shape"] for k in st["keepOut"] if k["shape"].get("type") != "mesh"]
    ko_mesh = [k["shape"] for k in st["keepOut"] if k["shape"].get("type") == "mesh"]
    pres = [p["shape"] for p in st["preserve"]]
    bores = [k["shape"] for k in st["keepOut"] if k.get("bore", True) and k["shape"].get("type") == "cylinder"
             and k["shape"].get("r", 99) < 4.0]
    n = {
        # smooth: the v0.3 STL is a marching-cubes mesh with ~0.8 mm facets; swollen and meshed
        # at 0.35 mm, its exact distance printed every facet as a hammered grain on the skin
        "v3": {"op": "mesh", "path": os.path.basename(v3_stl), "band": 10.0, "smooth": True,
               "description": "the v0.3 part: its load path, facets taken off"},
        "space": {"op": "shape", "shape": _union(ds)},
        "keepout": {"op": "shape", "shape": _union(ko_exact)},
        "preserve": {"op": "shape", "shape": _union(pres)},
        "stress": {"op": "result", "study": os.path.basename(study_path), "part": os.path.basename(v3_stl),
                   "pitch": 1.0, "scale": "peak", "description": "von Mises of the v0.3 part over its peak"},
        "grown": {"op": "offset", "input": "v3", "distance": offset,
                  "description": "tuned: how far the members swell"},
        "hubZone": {"op": "offset", "input": "preserve", "distance": margin},
        "inSpace": {"op": "intersect", "inputs": ["grown", "space"], "blend": 1.0},
        "members": {"op": "subtract", "inputs": ["inSpace", "hubZone", "keepout"]},
        "hubs": {"op": "intersect", "inputs": ["v3", "hubZone"],
                 "description": "the v0.3 part, unchanged, round every interface"},
        "envelope": {"op": "union", "inputs": ["members", "hubs", "preserve"], "blend": 2.0},
        "skinWall": {"op": "remap", "input": "stress", "from": [0.05, 0.7], "to": list(skin), "curve": "smoothstep"},
        "skin": {"op": "shell", "input": "envelope", "thickness": "skinWall"},
        "coreWall": {"op": "remap", "input": "stress", "from": [0.05, 0.6], "to": list(wall), "curve": "sqrt"},
        "core": {"op": "lattice", "type": "gyroid", "cell": cell, "thickness": "coreWall"},
        "fill": {"op": "intersect", "inputs": ["core", "envelope"]},
        "solid": {"op": "union", "inputs": ["hubs", "preserve"]},
        "joined": {"op": "union", "inputs": ["skin", "fill", "solid"], "blend": 1.2},
        "clipped": {"op": "intersect", "inputs": ["joined", "envelope"]},
    }
    cuts = ["clipped", "keepout"]
    for i, k in enumerate(ko_mesh):
        p = k["path"] if os.path.isabs(k["path"]) else os.path.join(sdir, k["path"])
        n[f"keepoutMesh{i}"] = {"op": "mesh", "path": os.path.basename(p)}
        cuts.append(f"keepoutMesh{i}")
    if bores:
        n["collars"] = {"op": "intersect", "inputs": [
            {"op": "shell", "input": {"op": "shape", "shape": _union(bores)}, "side": "outside", "thickness": 2.0},
            "envelope"], "description": "a solid 2 mm wall round every bolt bore"}
        n["joined"]["inputs"].append("collars")
    n["part"] = {"op": "subtract", "inputs": cuts}
    keep = ["keepout", "preserve"] + [c for c in cuts[2:]]
    return {
        "name": f"arm-sls-{link}",
        "description": f"The robot arm {link}, SLS edition: the v0.3 load path swollen into a smooth envelope, "
                       "stress-graded skin over a gyroid core, solid at every interface, drained",
        "nodes": n,
        "output": "part",
        "mesh": {"pitch": 0.35},
        "material": MATERIAL,
        "process": "sls",
        "minWall": 0.8,
        "drains": {"keepClear": keep},
        "tune": {
            "study": os.path.basename(study_path),
            "params": [
                {"path": "nodes/grown/distance", "min": 0.5, "max": 6.0},
                {"path": "nodes/skinWall/to/0", "min": 0.8, "max": 2.5},
                {"path": "nodes/skinWall/to/1", "min": 1.0, "max": 5.0},
                {"path": "nodes/coreWall/to/1", "min": 0.8, "max": 2.5},
            ],
            "analysis": "homogenized",
            "evaluations": 24,
            "cells": 120000,
        },
    }


def _paths(link, out):
    d = os.path.join(out, link)
    os.makedirs(d, exist_ok=True)
    return d, os.path.join(d, "study.json"), os.path.join(d, "graph.json")


def cmd_graph(link, out):
    d, sp, gp = _paths(link, out)
    st = study_doc(link)
    for k in st["keepOut"]:          # mesh keep-outs (the J5 sweep) next to the study
        if k["shape"].get("type") == "mesh":
            shutil.copy(os.path.join(HERE, "studies", k["shape"]["path"]), d)
    json.dump(st, open(sp, "w"), indent=1)
    shutil.copy(os.path.join(V3, f"{link}.stl"), os.path.join(d, "v3.stl"))
    json.dump(graph(link, sp, os.path.join(d, "v3.stl")), open(gp, "w"), indent=1)
    print("wrote", gp)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["graph"])
    ap.add_argument("--links", default=",".join(LINKS))
    ap.add_argument("--out", default="sls-work")
    a = ap.parse_args(argv)
    for link in a.links.split(","):
        {"graph": cmd_graph}[a.cmd](link, a.out)


if __name__ == "__main__":
    sys.exit(main())
