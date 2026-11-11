"""
A light mount designed round a scan: scan in, fitted part out.

The scene is a 31.8 mm handlebar with a bike light held where it should
hang. It is scanned in software (loadpath.scansim) at Artec-class accuracy:
rays from twelve sensor positions, 0.05 mm range noise, one 0.1 mm range bias
for the whole scan, 0.03% scale. From that scan alone:

1. `loadpath scan` fits the bar (a convex cylinder) and the light's faces,
   each with its +- band;
2. it drafts a study: clamp the bar (a slip fit on the bar's measured
   diameter, held for every bar in its band), carry the light on its top
   face, keep the light itself out by its measured clearance plus SLS's
   scatter;
3. `loadpath run` designs the part.

    python examples/scan/handlebar_mount.py --out out/scan-mount [--quick]

The loads are the brief, not the scan: a 150 g light at 40 g of bump is
about 60 N down; a 25 N sideways knock with 40 N down is the second case.
"""
import argparse
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from loadpath.scan import DEVICES  # noqa: E402
from loadpath.scansim import DEVICE_PROFILES, bar_and_light, ring_views, simulate, write_ply  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="out/scan-mount")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    mesh, truth = bar_and_light()
    prof, dev = DEVICE_PROFILES["artec-eva"], DEVICES["artec-eva"]
    P, N, meta = simulate(mesh, ring_views([0, 30, -30], 350, 6), spacing=prof["spacing"], sigma_noise=prof["noise"],
                          sigma_sys=dev["sigma"], scale=dev["scale"], seed=a.seed)
    scan = write_ply(os.path.join(a.out, "bar-and-light.ply"), P)
    print(f"scanned {len(P)} points; the scan's true range bias {meta['biases'][0]:+.3f} mm, "
          f"scale error {meta['scaleError']:+.5f}")
    feats = subprocess.run([sys.executable, "-m", "loadpath", "scan", scan, "--device", "artec-eva", "--units", "mm",
                            "--features-only"], capture_output=True, text=True, check=True)
    f = json.loads(feats.stdout)
    for row in f["features"]:
        print("  ", row["id"], row["what"])
    # the bar (a convex cylinder along x) and the light's top: the face looking up, just under the bar
    bar = next(r["id"] for r in f["features"] if r["id"].startswith("cyl") and r["what"].startswith("bar"))
    from loadpath.scan import fit_primitives, load_scan
    fp = fit_primitives(load_scan(scan, device="artec-eva", units="mm"))
    top = min((p for p in fp["planes"] if p["normal"][2] > 0.99),
              key=lambda p: abs(p["center"][2] - truth["lightTop"]["z"]))
    out = subprocess.run([sys.executable, "-m", "loadpath", "scan", scan, "--device", "artec-eva", "--units", "mm",
                          "--support", bar, "--load", f"{top['id']}=0,0,-60", "--load", f"{top['id']}=0,-25,-40",
                          "--process", "sls", "--material", "pa12", "--out", os.path.join(a.out, "draft")],
                         capture_output=True, text=True, check=True)
    r = json.loads(out.stdout)
    print("clearance round the light:", json.dumps(r["clearance"]))
    run = [sys.executable, "-m", "loadpath", "run", r["study"], "--out", os.path.join(a.out, "run"), "--viewer", "none"]
    if a.quick:
        run.append("--quick")
    subprocess.run(run, check=True)


if __name__ == "__main__":
    main()
