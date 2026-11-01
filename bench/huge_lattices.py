"""
Meshing time of large lattice parts, end to end (FieldGraph.mesh: field
sampling, marching cubes, relax/projection, creases, exact booleans).

    python bench/huge_lattices.py CASE PITCH [--profile]

CASE: gyroid-box (80x80x40 mm box, graded gyroid, 5 mm cells, shelled skin),
octet-box (same box, octet struts), arm-link (examples/fields/arm-link-graded.json,
FE stress replaced by a cheap stand-in so meshing runs alone).
Prints seconds, faces, volume, watertight. --profile prints cProfile's
top functions (--tottime sorts by own time); --save PATH.npz keeps the mesh.

4-core cloud box, 2026-11-01 (numpy paths at f38bd76 -> compiled kernels):
gyroid-box 0.4 mm 327 -> 58 s (5.9M faces), octet-box 0.4 mm 384 -> 67 s,
arm-link 0.4 mm 14.2 -> 10.6 s; gyroid-box 0.25 mm 133 s (14.8M faces).
"""
import copy, json, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np  # noqa: E402

from loadpath.fields import FieldGraph  # noqa: E402


def box(core):
    return {"name": f"{core}-box", "nodes": {
        "env": {"op": "shape", "shape": {"type": "box", "min": [0, 0, 0], "max": [80, 80, 40]}},
        "skin": {"op": "shell", "input": "env", "thickness": 1.2},
        "px": {"op": "coordinate", "axis": "x"},
        "wall": {"op": "remap", "input": "px", "from": [0, 80], "to": [0.5, 1.2] if core == "gyroid" else [0.8, 1.6]},
        "core": ({"op": "lattice", "type": "gyroid", "cell": 5, "thickness": "wall"} if core == "gyroid"
                 else {"op": "lattice", "type": "octet", "cell": 5, "thickness": "wall", "blend": 0.3}),
        "fill": {"op": "intersect", "inputs": ["core", "env"]},
        "part": {"op": "union", "inputs": ["skin", "fill"]},
    }, "output": "part"}


def arm():
    d = json.load(open(ROOT / "examples/fields/arm-link-graded.json"))
    d["nodes"]["stress"] = {"op": "remap", "input": "nearBore", "from": [3, 60], "to": [0.5, 0.05]}
    return d


CASES = {"gyroid-box": lambda: box("gyroid"), "octet-box": lambda: box("octet"), "arm-link": arm}

if __name__ == "__main__":
    case, pitch = sys.argv[1], float(sys.argv[2])
    g = FieldGraph(CASES[case]())
    t = time.time()
    if "--profile" in sys.argv:
        import cProfile, pstats
        pr = cProfile.Profile(); pr.enable()
    m = g.mesh(pitch=pitch)
    dt = time.time() - t
    if "--profile" in sys.argv:
        pr.disable(); pstats.Stats(pr).sort_stats(*(["tottime"] if "--tottime" in sys.argv else ["cumulative"])).print_stats(45)
    from loadpath.narrowband import watertight, signed_volume
    V, F = np.asarray(m.vertices), np.asarray(m.faces)
    out = {"case": case, "pitch": pitch, "seconds": round(dt, 1), "faces": len(F),
           "volume": round(float(signed_volume(V, F)), 1), "watertight": bool(watertight(F, len(V))),
           "creases": {k: v for k, v in (m.metadata.get("creases") or {}).items() if not isinstance(v, (list, dict))}}
    print(json.dumps(out))
    if "--save" in sys.argv:
        np.savez_compressed(sys.argv[sys.argv.index("--save") + 1], V=V, F=F)
