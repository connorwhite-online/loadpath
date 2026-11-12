"""
A part designed inside a scanned space: the bay stand-in.

The scene (loadpath.scansim.bay_with_pipe) is a small bay: a floor, a back
wall and two side walls, two bosses on the floor with M8 tapped holes in
their tops, and a 25 mm pipe across the bay above and behind them. It is
the stand-in for a real space (an engine bay) that a part has to fit into,
bolt to and hold something in. It is scanned in software at Artec-class
accuracy from the open front and top only, as a real bay would be, so the
undersides and the backs are never seen.

From that scan alone:

1. the fits find the pipe, the bosses, their tops and the two tapped holes,
   each with its +- band; the holes snap to M8 (ISO 2306 tap drill 6.8 mm),
   the pipe is left unsnapped (25.0 and 25.4 both lie in its band: a
   caliper settles it);
2. `draft_around` writes the study: bolt to the bosses (an M8 screw through
   the part into each tapped hole: ISO 273 clearance hole, head and driver
   path kept clear, seat and face rings, preload), carry the pipe in a ring
   round it, keep the scanned bay out by its measured clearance plus SLS's
   scatter, and stay inside the bay's free space;
3. `loadpath run` designs the part;
4. the part is checked against the **true** scene, not the scan: no
   overlap with the bay, inside the free space, the screw paths clear on
   the true hole centres, material on each boss top and round the pipe.

    python examples/scan/fit_in_space.py --out out/fit-in-space [--quick] [--time-limit 10]

What it shows today (seed 1, --quick): every check passes on the true
scene. The bay is kept out as everything the sensor's rays did not see
empty (loadpath.freespace, from each point's sensor position in the PLY),
not as a reconstructed surface: a reconstruction of an open scene seen from
inside isn't a closed object, and that version reached 0.37 mm into a boss
edge. The draft says 42% of the ring round the pipe is space the rays never
reached (behind and under it), the honest rescan prompt.

The load is the brief, not the scan: 75 N down on the pipe (a 1.5 kg
section at 5 g of shock, a choice for the example).
"""
import argparse
import json
import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from loadpath.scan import DEVICES, fit_primitives, load_scan  # noqa: E402
from loadpath.scansim import DEVICE_PROFILES, bay_with_pipe, ring_views, simulate, write_ply  # noqa: E402
from loadpath.scanstd import snap_features  # noqa: E402
from loadpath.scanstudy import draft_around  # noqa: E402


def scan_bay(out, seed=1):
    mesh, truth = bay_with_pipe()
    views = ring_views([0, -10, 20], 350, 8, elevations=(20.0, 45.0, 70.0))
    views = views[views[:, 1] < 30]          # the bay is open to the front and the top only
    prof, dev = DEVICE_PROFILES["artec-eva"], DEVICES["artec-eva"]
    P, N, meta = simulate(mesh, views, spacing=prof["spacing"], sigma_noise=prof["noise"], sigma_sys=dev["sigma"],
                          scale=dev["scale"], seed=seed)
    # each point with the sensor position it was measured from (the Imprint
    # format's rays): what the scan saw empty is then kept apart from what it did not
    path = write_ply(os.path.join(out, "bay.ply"), P, N, meta["viewpoints"])
    print(f"scanned {len(P)} points; the scan's true range bias {meta['biases'][0]:+.3f} mm, "
          f"scale error {meta['scaleError']:+.5f}")
    return mesh, truth, path


def pick(feats, truth):
    """The features the brief names, found by where they are (an agent reads
    the summary; the example finds them by the truth's positions)."""
    cyl = feats["cylinders"]
    pipe = min((c for c in cyl if c["kind"] == "bar"),
               key=lambda c: abs(c["diameter"] - 25) + abs(np.asarray(c["p0"])[2] - 45))
    tops = []
    for b in truth["bosses"]:
        c = np.asarray(b["center"], float)
        tops.append(min((p for p in feats["planes"] if p["normal"][2] > 0.99),
                        key=lambda p: np.linalg.norm(np.asarray(p["center"]) - c))["id"])
    taps = [c["id"] for c in cyl if c["kind"] == "bore"]
    return pipe["id"], tops, taps


def check_against_truth(part_path, scene, truth, study):
    """The designed part against the true bay. Returns rows of (check, ok, detail)."""
    import trimesh
    from loadpath.shapes import manifold_from_trimesh, trimesh_from_manifold
    part = trimesh.load(part_path, force="mesh")
    mp, ms = manifold_from_trimesh(part), manifold_from_trimesh(scene)
    rows = []
    inter = mp ^ ms
    overlap = inter.volume()
    depth = 0.0
    if overlap > 0:
        x = trimesh_from_manifold(inter)
        # how deep the part goes into the bay: the overlap's points' depth
        # below the bay's surface (a face-to-face contact is 0)
        depth = float(np.max(np.abs(trimesh.proximity.signed_distance(scene, x.vertices))))
    rows.append(("no overlap with the bay", depth <= 0.05, f"{overlap:.2f} mm^3, at most {depth:.2f} mm deep"))
    lo, hi = (np.asarray(x, float) for x in truth["region"])
    b = part.bounds
    inside = bool(np.all(b[0] >= lo - 1.0) and np.all(b[1] <= hi + 1.0))
    rows.append(("inside the free space", inside, f"part {np.round(b, 1).tolist()}"))
    by_id = {f["id"]: f for f in study.get("fasteners", [])}
    for i, boss in enumerate(truth["bosses"]):
        c = np.asarray(boss["center"], float)
        # the screw (M8: 8 mm) on the TRUE hole centre, from the boss top up
        # through the part's grip: no material in its way
        f = next((f for f in by_id.values() if np.linalg.norm(np.asarray(f["seat"])[:2] - c[:2]) < 2.0), None)
        if f is None:
            rows.append((f"boss {i}: bolted", False, "no fastener on this boss"))
            continue
        top = float(np.asarray(f["seat"])[2])
        screw = trimesh.creation.cylinder(radius=4.0, height=top - c[2], sections=64)
        screw.apply_translation([c[0], c[1], 0.5 * (c[2] + top)])
        hit = (mp ^ manifold_from_trimesh(screw)).volume()
        rows.append((f"boss {i}: M8 screw path clear on the true centre", hit < 0.5, f"{hit:.2f} mm^3 in the way"))
        # the part bears on the boss top: material in the annulus just above it
        ring = trimesh.creation.annulus(r_min=5.0, r_max=7.5, height=1.5, sections=64)
        ring.apply_translation([c[0], c[1], c[2] + 0.75 + 0.3])
        sat = (mp ^ manifold_from_trimesh(ring)).volume() / ring.volume
        rows.append((f"boss {i}: part seated on the boss top", sat > 0.5, f"{100 * sat:.0f}% of the ring filled"))
    p = truth["pipe"]
    r = 0.5 * p["diameter"]
    shell = trimesh.creation.annulus(r_min=r, r_max=r + 2.0, height=20.0, sections=128)
    shell.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [0, 1, 0]))
    shell.apply_translation(p["point"])
    held = (mp ^ manifold_from_trimesh(shell)).volume() / shell.volume
    rows.append(("pipe held (material round it)", held > 0.3, f"{100 * held:.0f}% of a 2 mm shell"))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="out/fit-in-space")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--time-limit", type=float, default=10.0, help="minutes for loadpath run")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    scene, truth, path = scan_bay(a.out, a.seed)
    s = load_scan(path, device="artec-eva", units="mm")
    feats = fit_primitives(s)
    pipe, tops, taps = pick(feats, truth)
    # the brief says the bosses are tapped (an insert of the same bore reads alike);
    # the pipe's family is not known
    snap_features(feats, {t: ["tapped"] for t in taps})
    for c in feats["cylinders"]:
        print(f"   {c['id']} {c['kind']} {c['diameter']:.3f} +-{c['diameterBand']:.3f} "
              f"{c.get('standard', {}).get('name') or c.get('standardChoices') or ''}")
    r = draft_around(s, feats, os.path.join(a.out, "draft"), support=tops, loads={pipe: [0, 0, -75]},
                     space=truth["region"], process="sls", material="pa12", name="bay-bracket")
    for t in r["todo"]:
        print("todo:", t)
    run = [sys.executable, "-m", "loadpath", "run", r["studyPath"], "--out", os.path.join(a.out, "run"),
           "--viewer", "none", "--no-bundle", "--no-render", "--time-limit", str(a.time_limit)]
    if a.quick:
        run.append("--quick")
    subprocess.run(run, check=True, stdout=subprocess.DEVNULL)
    stl = os.path.join(a.out, "run", "best.stl")
    rows = check_against_truth(stl, scene, truth, r["study"])
    for name, ok, detail in rows:
        print(f"{'ok  ' if ok else 'MISS'} {name}: {detail}")
    with open(os.path.join(a.out, "truth-check.json"), "w") as fh:
        json.dump([{"check": n, "ok": bool(o), "detail": d} for n, o, d in rows], fh, indent=1)
    return 0 if all(o for _, o, _ in rows) else 1


if __name__ == "__main__":
    sys.exit(main())
