"""Tables from results_beams.json: peak vm / beam theory per case and method,
and error statistics per thickness."""
import json
import os
import sys

import numpy as np

path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "results_beams.json")
rows = json.load(open(path))
METHODS = [
    ("voxel today", "stair/std", "nodal"),
    ("stair+IM", "stair/im", "nodal"),
    ("cutS (fill-scaled)", "cutS/std", "surfD"),
    ("cutQ+std, surface", "cutQ/std", "surfD"),
    ("cutQ+IM, surface", "cutQ/im", "surfD"),
    ("cutQ+IM, surface ext", "cutQ/im", "surfX0.5_2.0"),
    ("cutQ+std, surface ext", "cutQ/std", "surfX0.5_2.0"),
]
extra = [m for m in sys.argv[2:]]
for e in extra:
    k, rec = e.split(":")
    METHODS.append((e, k, rec))


def val(r, k, rec):
    return r["voxel"][k][rec]


print("| load | t (cells) | angle | tet max | tet p99 | " + " | ".join(m[0] for m in METHODS) + " |")
print("|" + "---|" * (5 + len(METHODS)))
for r in rows:
    tet = r.get("tet", {})
    print(f"| {r['load']} | {r['t']} | {r['theta']} | {tet.get('peak', float('nan')):.2f} | {tet.get('p99', float('nan')):.2f} | "
          + " | ".join(f"{val(r, k, rec):.2f}" for _, k, rec in METHODS) + " |")

print()
print("Per thickness: mean |ratio - 1| and [min, max] of the ratio over loads and angles")
print("| method | " + " | ".join(f"t={t}" for t in sorted({r['t'] for r in rows})) + " | all |")
print("|---|" + "---|" * (len({r['t'] for r in rows}) + 1))
for name, k, rec in [("tet p99", None, None)] + METHODS:
    cells = []
    allv = []
    for t in sorted({r['t'] for r in rows}):
        v = np.array([r["tet"]["p99"] if k is None else val(r, k, rec) for r in rows if r["t"] == t])
        allv.extend(v)
        cells.append(f"{np.mean(np.abs(v - 1)):.2f} [{v.min():.2f}, {v.max():.2f}]")
    allv = np.array(allv)
    cells.append(f"{np.mean(np.abs(allv - 1)):.2f} [{allv.min():.2f}, {allv.max():.2f}]")
    print(f"| {name} | " + " | ".join(cells) + " |")

print()
print("Deflection (work-conjugate) / tet, bending")
print("| t | angle | " + " | ".join(k for k in ["stair/std", "stair/im", "cutS/std", "cutQ/std", "cutQ/im"]) + " |")
print("|---|---|" + "---|" * 5)
for r in rows:
    if r["load"] != "bending" or "tet" not in r:
        continue
    print(f"| {r['t']} | {r['theta']} | " + " | ".join(f"{r['voxel'][k]['defl'] / r['tet']['defl']:.2f}"
                                                  for k in ["stair/std", "stair/im", "cutS/std", "cutQ/std", "cutQ/im"]) + " |")
