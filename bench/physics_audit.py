"""
The physics audit's before / after numbers (docs/PHYSICS.md).

Each block reads the same part or the same duty twice: once the way
loadpath read it before the audit (von Mises against tensile strength with
only the tension across layers on the bond; a fixed convection coefficient;
fluids at room temperature) and once as it reads it now.

    python bench/physics_audit.py                 # the fast blocks (~5 min on 4 cores)
    python bench/physics_audit.py --parts DIR     # also re-verify delivered eval parts (DIR/<case>.stl)

Writes bench/results/physics_audit.json.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import trimesh  # noqa: E402

from loadpath import failure  # noqa: E402
from loadpath.spec import study_from_dict  # noqa: E402
from loadpath.verify import verify_part  # noqa: E402


@contextlib.contextmanager
def legacy():
    """The criteria before the audit: plain von Mises, and a layer bond
    loaded only by the tension normal to it."""
    eq, bond = failure.equivalent_stress, failure.bond_stress

    def tension_only(sig, n, shear_ratio=0.0):
        e, tens, tau = bond(sig, n, 0.0)
        return tens, tens, tau

    failure.equivalent_stress = lambda sig, m=1.0: eq(sig, 1.0)
    failure.bond_stress = tension_only
    try:
        yield
    finally:
        failure.equivalent_stress, failure.bond_stress = eq, bond


def _box(lo, hi):
    b = trimesh.creation.box(extents=np.subtract(hi, lo))
    b.apply_translation(np.add(lo, hi) / 2)
    return b


def _row(rep):
    c = min(rep["cases"], key=lambda c: c["safetyFactor"])
    env = rep.get("serviceEnvelope") or {}
    return {"safetyFactor": round(float(rep["safetyFactor"]), 3), "limitedBy": c.get("limitedBy"),
            "peakStress": round(float(c["peakStress"]), 2),
            "interlayerStress": None if c.get("interlayerStress") is None else round(float(c["interlayerStress"]), 2),
            "heldHours": env.get("heldHours"), "cycles": env.get("cycles"), "maxTemperature": env.get("maxTemperature")}


def boss():
    """A 16 mm PETG boss printed standing (layers across its axis), 40 mm
    tall, held at its foot: twisted, pulled, pushed and bent at its top."""
    mesh = trimesh.creation.cylinder(radius=8, height=40, sections=96)
    mesh.apply_translation([0, 0, 20])
    out = {}
    for name, load in (("twist 3 N·m", {"force": [0, 0, 0], "moment": [0, 0, 3000]}),
                       ("pull 800 N", {"force": [0, 0, 800]}), ("push 800 N", {"force": [0, 0, -800]}),
                       ("side 60 N", {"force": [60, 0, 0]})):
        st = study_from_dict({
            "designSpace": {"type": "cylinder", "p0": [0, 0, 0], "p1": [0, 0, 40], "r": 8},
            "supports": [{"shape": {"type": "box", "min": [-9, -9, -1], "max": [9, 9, 0.1]}}],
            "loadCases": [{"name": name, "loads": [{"shape": {"type": "box", "min": [-9, -9, 39.9],
                                                              "max": [9, 9, 41]}, **load}]}],
            "material": "petg", "process": {"type": "fdm", "buildDirection": "+z"},
            "goal": {"volumeFraction": 0.5}})
        with legacy():
            before = _row(verify_part(st, mesh, max_cells=30_000, contact_band=4.0))
        after = _row(verify_part(st, mesh, max_cells=30_000, contact_band=4.0))
        out[name] = {"before": before, "after": after}
        print(f"  boss, {name:12s} SF {before['safetyFactor']:5.2f} -> {after['safetyFactor']:5.2f} "
              f"({after['limitedBy']}); holds ~{_span(after['heldHours'])}, ~{_count(after['cycles'])} cycles, "
              f"to ~{after['maxTemperature']:.0f} C")
    return out


def _span(h):
    if h is None:
        return "indefinitely"
    return f"{h:.0f} h" if h < 8760 else f"{h / 8760:.1f} years"


def _count(n):
    return "unlimited" if n is None else f"{n:.2g}"


def delivered(parts_dir):
    """Delivered eval parts, verified with the criteria before and now."""
    from loadpath.domain import choose_pitch

    out = {}
    for case, stl in (("printed-hook", "printed-hook"), ("wall-bracket", "wall-bracket"),
                      ("motor-mount", "motor-mount"), ("bell-crank", "bell-crank"), ("mbb-beam", "mbb-beam"),
                      ("robot-wrist-yoke", "grip-wrist-yoke")):
        path = Path(parts_dir) / f"{stl}.stl"
        spec = ROOT / "loadpath" / "evals" / "cases" / f"{case}.json"
        if not path.exists() or not spec.exists():
            continue
        st = study_from_dict(json.loads(spec.read_text()), base_dir=str(spec.parent))
        mesh = trimesh.load(str(path), force="mesh")
        pitch = choose_pitch(st)
        kw = {"max_cells": 60_000, "contact_band": pitch, "max_pitch": pitch}
        t0 = time.time()
        with legacy():
            before = _row(verify_part(st, mesh, **kw))
        rep = verify_part(st, mesh, **kw)
        after = _row(rep)
        lay = (rep.get("failure") or {}).get("layers") or {}
        built = {a: (None if v["safetyFactor"] is None else round(v["safetyFactor"], 2))
                 for a, v in (lay.get("ifBuiltAlong") or {}).items()}
        out[case] = {"material": st.material.name, "before": before, "after": after, "ifBuiltAlong": built,
                     "compressionRatio": rep["failure"]["compressionRatio"]}
        print(f"  {case:18s} {st.material.name:9s} SF {before['safetyFactor']:5.2f} -> {after['safetyFactor']:5.2f}; "
              f"layers along x/y/z {built or '-'}; holds ~{_span(after['heldHours'])}, "
              f"~{_count(after['cycles'])} cycles, to ~{after['maxTemperature'] or float('nan'):.0f} C "
              f"({time.time() - t0:.0f} s)")
    return out


def heat_sink():
    """A 40 x 40 mm aluminium plate with 1.5 W into its underside and 0-14
    plate fins 20 mm tall, in still air: the hottest point with a fixed
    8 W/(m^2 K) and with the coefficient solved from the surface."""
    out = {}
    for fins in (0, 4, 8, 14):
        parts = [_box([0, 0, 0], [40, 40, 4])]
        if fins:
            step = (40 - 1.5) / (fins - 1)
            parts += [_box([i * step, 0, 3.99], [i * step + 1.5, 40, 24]) for i in range(fins)]
        mesh = trimesh.boolean.union(parts, engine="manifold") if fins else parts[0]
        row = {"area": round(float(mesh.area))}
        for label, conv in (("fixed", 8), ("natural", "natural")):
            st = study_from_dict({
                "designSpace": {"type": "box", "min": [0, 0, 0], "max": [40, 40, 24]},
                "supports": [{"shape": {"type": "box", "min": [-1, -1, -1], "max": [2, 41, 0.1]}}],
                "loadCases": [{"name": "c", "loads": [{"shape": {"type": "box", "min": [38, -1, -1],
                                                                 "max": [41, 41, 0.1]}, "force": [0, 0, -1]}]}],
                "material": "alsi10mg", "goal": {"volumeFraction": 0.5},
                "thermal": {"heat": [{"shape": {"type": "box", "min": [10, 10, -1], "max": [30, 30, 0.6]},
                                      "watts": 1.5}], "expansion": False, "convection": conv}})
            th = verify_part(st, mesh, max_cells=80_000, refine=0)["thermal"]
            row[label] = round(float(th["maxTemperature"]), 1)
            if th.get("convection"):
                row["h"] = {k: round(v, 2) for k, v in th["convection"]["h"].items()}
                row["view"] = round(th["convection"]["meanView"], 2)
        out[str(fins)] = row
        print(f"  {fins:2d} fins ({row['area']} mm²): {row['fixed']:.1f} C at a fixed 8 W/(m² K), "
              f"{row['natural']:.1f} C in still air (mean h {row['h']['mean']:.1f}, view {row['view']:.2f})")
    return out


def exchanger():
    """A charge-air cooler's duty rated with room-temperature fluids and
    with each fluid at its own state."""
    from loadpath.heatx_thermal import FLUIDS, rate

    geo = ("gyroid", 8.0, 0.6, 250 * 120 * 100.0, [100.0, 120.0])
    hot = {"kgPerS": 0.08, "tIn": 150}
    cold = {"lPerMin": 20, "tIn": 45}
    ports = {"port_diameter": (60.0, 19.0)}
    before = rate(*geo, {"air": {**hot, "fluid": FLUIDS["air"]}, "coolant": {**cold, "fluid": FLUIDS["glycol50"]},
                         "wall": "alsi10mg"}, **ports)
    after = rate(*geo, {"air": {**hot, "fluid": "air", "pressure": 2.5},
                        "coolant": {**cold, "fluid": "glycol50"}, "wall": "alsi10mg"}, **ports)

    def row(r):
        return {"effectiveness": round(r["effectiveness"], 3), "duty": round(r["duty"]),
                **{f"{n}{k}": round(r["fluids"][n][key]) for n in ("air", "coolant")
                   for k, key in (("Re", "re"), ("PressureDrop", "pressureDrop"), ("H", "h"))}}
    out = {"before": row(before), "after": row(after)}
    for k in ("airPressureDrop", "coolantPressureDrop", "airH", "coolantH", "duty"):
        print(f"  {k:20s} {out['before'][k]:8d} -> {out['after'][k]:8d} ({out['after'][k] / out['before'][k] - 1:+.0%})")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", help="folder of delivered eval STLs (<case>.stl)")
    ap.add_argument("--only", nargs="*", help="blocks to run: boss delivered heatsink exchanger")
    args = ap.parse_args()
    want = set(args.only or ["boss", "delivered", "heatsink", "exchanger"])
    dest = ROOT / "bench" / "results" / "physics_audit.json"
    out = json.loads(dest.read_text()) if dest.exists() else {}
    if "exchanger" in want:
        print("exchanger: fluids at room temperature -> at their state")
        out["exchanger"] = exchanger()
    if "boss" in want:
        print("standing boss, FDM PETG: von Mises + bond tension -> pressure-modified + bond tension and shear")
        out["boss"] = boss()
    if "heatsink" in want:
        print("finned plate in still air")
        out["heatSink"] = heat_sink()
    if "delivered" in want and args.parts:
        print("delivered eval parts")
        out["delivered"] = delivered(args.parts)
    dest.parent.mkdir(exist_ok=True)
    dest.write_text(json.dumps(out, indent=1, default=float) + "\n")
    print(f"wrote {dest}")


if __name__ == "__main__":
    main()
