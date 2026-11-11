"""
Does the scan's stated uncertainty cover the truth? Monte Carlo over
synthetic scans of known geometry (loadpath.scansim: a 31.8 mm handlebar and
a bike light hanging off it), each with fresh random noise, per-view range
bias and scale error drawn from the device's numbers.

For each scan: the bar's fitted diameter against 31.8 (inside diameterBand?),
the light's top face against its true height (inside offsetBand?), and the
reconstructed surface against the true one (the share of true surface
points no farther outside the measured surface than its one-sided U). At 95%
confidence about 95% should be covered; much less means the bands are too
tight (dangerous: parts that do not fit), much more that they are loose
(parts with more play than needed).

    python examples/scan/validate.py --device artec-eva --runs 12
    python examples/scan/validate.py --device splat --runs 12     # splats, scaled by a caliper reference
"""
import argparse
import json
import os
import sys
import tempfile
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from loadpath.scan import DEVICES, fit_primitives, load_scan, reconstruct  # noqa: E402
from loadpath.scansim import DEVICE_PROFILES, bar_and_light, ring_views, simulate, write_ply, write_splat_ply  # noqa: E402


def one(device, seed, tmp, recon=False, bias="common"):
    mesh, truth = bar_and_light()
    views = ring_views([0, 30, -30], 350 if device != "leica-rtc360" else 1500, 6)
    prof = DEVICE_PROFILES.get(device, DEVICE_PROFILES["artec-eva"])
    if device == "splat":
        # photogrammetric points: noise like a good capture, no metric scale
        P, N, meta = simulate(mesh, views, spacing=0.8, sigma_noise=0.15, seed=seed)
        path = os.path.join(tmp, f"s{seed}.ply")
        write_splat_ply(path, P, N, spacing=0.8, thickness=0.1, scale_factor=0.0123, seed=seed)
        s = load_scan(path)
        f0 = fit_primitives(s)
        bar = max((c for c in f0["cylinders"] if c["kind"] == "bar"), key=lambda c: c["points"])
        s = load_scan(path, reference=[{"feature": bar["id"], "diameter": 31.8, "tolerance": 0.05}])
        d_known = True
    else:
        dev = DEVICES[device]
        P, N, meta = simulate(mesh, views, spacing=prof["spacing"], sigma_noise=prof["noise"],
                              sigma_sys=dev["sigma"], scale=dev["scale"], corr=prof["corr"], seed=seed, bias=bias)
        path = os.path.join(tmp, f"s{seed}.ply")
        write_ply(path, P)
        s = load_scan(path, device=device, units="mm", max_points=80_000)
        d_known = False
    f = fit_primitives(s)
    out = {"seed": seed, "points": s.report["points"], "noise": s.report["noise"]["median"]}
    bars = [c for c in f["cylinders"] if c["kind"] == "bar" and abs(c["axis"][0]) > 0.99]
    if bars and not d_known:
        b = max(bars, key=lambda c: c["points"])
        out["barErr"] = b["diameter"] - truth["bar"]["diameter"]
        out["barBand"] = b["diameterBand"]
    tops = [p for p in f["planes"] if p["normal"][2] > 0.99 and abs(p["center"][2] - truth["lightTop"]["z"]) < 5]
    if tops:
        t = max(tops, key=lambda p: p["points"])
        # the top's height above the bar's axis: what a mount spanning them must get right
        out["topErr"] = t["center"][2] - truth["lightTop"]["z"]
        out["topBand"] = t["offsetBand"]
    if recon:
        import trimesh
        m, info = reconstruct(s)
        rng = np.random.default_rng(seed)
        T, _ = trimesh.sample.sample_surface(mesh, 20000, seed=int(rng.integers(1 << 30)))
        # how far each true point is outside the measured surface (+ out)
        sd = -trimesh.proximity.ProximityQuery(m).signed_distance(T)
        Uq = s.uncertainty(P=T)
        out["surfaceCovered"] = float(np.mean(sd <= Uq))
        out["surfaceU"] = float(np.median(Uq))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="artec-eva")
    ap.add_argument("--runs", type=int, default=12)
    ap.add_argument("--recon", type=int, default=3, help="runs that also check the reconstructed surface")
    ap.add_argument("--bias", choices=["common", "view"], default="common",
                    help="the device's systematic error: one for the scan (common) or one per view")
    ap.add_argument("--out")
    a = ap.parse_args()
    rows = []
    t0 = time.time()
    with tempfile.TemporaryDirectory() as tmp:
        for i in range(a.runs):
            r = one(a.device, 100 + i, tmp, recon=i < a.recon, bias=a.bias)
            rows.append(r)
            print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}), flush=True)
    summ = {"device": a.device, "bias": a.bias, "runs": a.runs, "seconds": round(time.time() - t0, 1)}
    for key, band in (("barErr", "barBand"), ("topErr", "topBand")):
        v = [(r[key], r[band]) for r in rows if key in r]
        if v:
            e = np.array([x[0] for x in v])
            b = np.array([x[1] for x in v])
            summ[key] = {"found": len(v), "covered": round(float(np.mean(np.abs(e) <= b)), 3),
                         "rmsError": round(float(np.sqrt((e ** 2).mean())), 4),
                         "meanBand": round(float(b.mean()), 4)}
    sc = [r["surfaceCovered"] for r in rows if "surfaceCovered" in r]
    if sc:
        summ["surface"] = {"covered": round(float(np.mean(sc)), 4),
                           "medianU": round(float(np.median([r["surfaceU"] for r in rows if "surfaceU" in r])), 4)}
    print(json.dumps(summ, indent=1))
    if a.out:
        with open(a.out, "w") as fh:
            json.dump({"summary": summ, "runs": rows}, fh, indent=1)


if __name__ == "__main__":
    main()
