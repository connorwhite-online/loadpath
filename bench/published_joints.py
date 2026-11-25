"""
loadpath's checks against published tests of printed plates pulled through
a pin in a hole (loadpath/data/published_joints.json).

Each specimen is rebuilt as an STL (a plate with the hole) and a study:
the plate held over its grip, a 1 kN load on the hole as a steel pin that
only pushes (`"distribution": "bearing"`, insert.py), the paper's
material. The checks are linear, so the load at which the reading reaches
the strength is the predicted first failure:

    predicted = 1 kN x strength / peak    (no allowance)

and the safety factor the check reports carries verify.STRESS_ALLOWANCE
(1.15) on top. Readings:

- voxel: verify_part, the check every run and `loadpath verify` makes;
- tet: crosscheck (P2 tetrahedra), the confirm the search makes;
- banded: the voxel check as it read before the bore was counted (the
  band round a load's region left out of the peak), for the record;
- length: the voxel check reading over --length mm, uncapped by the
  plate's thickness (Ahmed & Susmel 2018 measured 4.6 mm for FDM PLA), a
  sensitivity only: the shipped check caps the radius at 4/9 of the
  thinnest member, which on these plates is the whole reading.

    python bench/published_joints.py                  # all, voxel + tet
    python bench/published_joints.py --only hertel-0 --no-tet
    python bench/published_joints.py --size 0.8       # tet element size (mm)

Writes bench/results/published-joints.json.
"""
import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DATA = ROOT / "loadpath" / "data" / "published_joints.json"
OUT = ROOT / "bench" / "results" / "published-joints.json"
FORCE = 1000.0


def load():
    return json.loads(DATA.read_text())


def specimen(sp, sources):
    """(study dict, trimesh part) of one specimen: x along the plate, the
    grip at x = 0, the hole `hole` mm from the far end, the pin pulling +x."""
    import trimesh

    L, W, t, D = (float(sp[k]) for k in ("L", "W", "t", "D"))
    xh = L - float(sp["hole"])
    box = trimesh.creation.box(extents=[L, W, t])
    box.apply_translation([L / 2, 0, 0])
    cyl = trimesh.creation.cylinder(radius=D / 2, height=3 * t, sections=128)
    cyl.apply_translation([xh, 0, 0])
    part = box.difference(cyl)
    src = sources[sp["source"]]
    mat = sp.get("material") or src["material"]
    if isinstance(mat, dict):
        mat = {k: v for k, v in mat.items() if k != "measured"}
    g = float(sp["grip"])
    study = {
        "name": sp["id"],
        "designSpace": {"type": "box", "min": [0, -W / 2, -t / 2], "max": [L, W / 2, t / 2]},
        "keepOut": [{"id": "hole", "shape": {"type": "cylinder", "center": [xh, 0, 0], "axis": "z", "length": t,
                                              "r": D / 2}}],
        "supports": [{"shape": {"type": "box", "min": [-0.5, -W / 2 - 0.5, -t / 2 - 0.5],
                                "max": [g, W / 2 + 0.5, t / 2 + 0.5]}, "fix": "all"}],
        "loadCases": [{"name": "pin", "loads": [{"region": "hole", "force": [FORCE, 0, 0],
                                                  "distribution": "bearing"}]}],
        "material": mat,
        "process": {"type": "fdm", "buildDirection": "+z"},
        "goal": {"safetyFactor": 1.0},
    }
    return study, part


def _row(peak, at, strength, measured, xh, D):
    pred = FORCE * strength / peak
    ang = float(np.degrees(np.arctan2(abs(at[1]), at[0] - xh)))
    return {"peak": round(peak, 2), "at": [round(float(v), 2) for v in at],
            "fromHole": round(float(np.hypot(at[0] - xh, at[1])) - D / 2, 2), "angle": round(ang, 1),
            "predicted": round(pred, 1), "overMeasured": round(pred / measured, 3)}


def voxel(study, part, cells=None):
    from loadpath import quick
    from loadpath.domain import choose_pitch
    from loadpath.spec import study_from_dict
    from loadpath.verify import verify_part

    st = study_from_dict(study)
    pitch = choose_pitch(st)
    if cells:
        rep = verify_part(st, part, max_cells=cells, min_pitch=None, contact_band=pitch, max_pitch=None)
    else:
        rep = verify_part(st, part, max_cells=quick.verify_cells(60_000), min_pitch=0.5 * pitch, contact_band=pitch,
                          max_pitch=pitch)
    c = rep["cases"]["pin"] if isinstance(rep["cases"], dict) else rep["cases"][0]
    return c["peakStress"], c["peakAt"], st.material.strength, rep.get("analysisPitch")


def tet(study, part, size):
    from loadpath.crosscheck import crosscheck
    from loadpath.spec import study_from_dict

    st = study_from_dict(study)
    r = crosscheck(st, part, size=size)
    c = r["cases"][0]
    return c["peakStress"], c["peakAt"], st.material.strength, r["nodes"]


def banded(study, part):
    """The voxel check with the ring's bore left out of the peak, as before."""
    import loadpath.domain as dom

    orig = dom.boundary_conditions

    def bc(*a, **k):
        out = orig(*a, **k)
        out.case_ring_nodes = []
        return out
    import loadpath.verify as ver
    dom.boundary_conditions = bc
    vb = getattr(ver, "boundary_conditions", None)
    if vb is not None:
        ver.boundary_conditions = bc
    try:
        return voxel(study, part)
    finally:
        dom.boundary_conditions = orig
        if vb is not None:
            ver.boundary_conditions = vb


def with_length(study, part, length):
    """The voxel check reading over `length` whatever the part's thinnest
    member (crosscheck.reading_radius caps the radius at 4/9 of it, which
    on a thin plate is the whole reading)."""
    import loadpath.crosscheck as cc

    orig = cc.stress_radius
    cc.stress_radius = lambda st, size: (max(length, 1.5 * size), {"radius": length, "of": "sensitivity"})
    try:
        return voxel(study, part)
    finally:
        cc.stress_radius = orig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", help="ids starting with this")
    ap.add_argument("--no-tet", action="store_true")
    ap.add_argument("--size", type=float, default=None, help="tet element size, mm (default 0.8 on 2 mm "
                    "plates, 1.0 on thicker: 0.5 on the 2 mm strip ran out of 15 GB)")
    ap.add_argument("--length", type=float, default=4.6, help="material length for the sensitivity row, mm")
    ap.add_argument("--reuse", action="store_true", help="keep readings already saved")
    args = ap.parse_args()
    data = load()
    old = json.loads(OUT.read_text())["specimens"] if (args.reuse and OUT.exists()) else {}
    out = dict(old)
    for sp in data["specimens"]:
        if args.only and not sp["id"].startswith(args.only):
            continue
        study, part = specimen(sp, data["sources"])
        xh, D, meas = float(sp["L"]) - float(sp["hole"]), float(sp["D"]), float(sp["failureLoad"])
        row = dict(old.get(sp["id"], {}))
        row.update({"measured": meas, "mode": sp["mode"], "source": sp["source"]})
        t0 = time.time()
        peak, at, strength, pitch = voxel(study, part)
        row["voxel"] = {**_row(peak, at, strength, meas, xh, D), "pitch": pitch}
        peak, at, strength, _ = banded(study, part)
        row["banded"] = _row(peak, at, strength, meas, xh, D)
        peak, at, strength, _ = with_length(study, part, args.length)
        row["length"] = {**_row(peak, at, strength, meas, xh, D), "radius": args.length}
        if not args.no_tet and not (args.reuse and "tet" in row):
            size = args.size or (0.8 if float(sp["t"]) <= 2 else 1.0)
            peak, at, strength, n = tet(study, part, size)
            row["tet"] = {**_row(peak, at, strength, meas, xh, D), "size": size, "nodes": n}
        out[sp["id"]] = row
        print(sp["id"], f"{time.time() - t0:.0f}s", "measured", meas,
              *(f"{k} {row[k]['predicted']:.0f} ({row[k]['overMeasured']:.2f}x, {row[k]['angle']:.0f} deg)"
                for k in ("voxel", "tet", "banded", "length") if k in row), flush=True)
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps({"force": FORCE, "specimens": out}, indent=1))


if __name__ == "__main__":
    main()
