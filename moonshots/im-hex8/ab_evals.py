"""A/B of two eval runs on one commit (LOADPATH_HEX8=plain vs the default),
from the {"case": ...} lines loadpath.evals.run prints, with the README
gallery numbers alongside. Usage: ab_evals.py --a LOG... --b LOG..."""
import argparse
import json
import re

ap = argparse.ArgumentParser()
ap.add_argument("--a", nargs="+", required=True, help="plain hex8 logs")
ap.add_argument("--b", nargs="+", required=True, help="incompatible-mode logs")
args = ap.parse_args()
readme = open(__file__.rsplit("/moonshots/", 1)[0] + "/README.md").read()
gal = {m.group(1): (float(m.group(2)), float(m.group(3)))
       for m in re.finditer(r"<b>([\w-]+)</b><br><sub>([\d.]+) g · \d+% lighter · SF ([\d.]+)", readme)}


def load(paths):
    out = {}
    for p in paths:
        for line in open(p):
            if line.startswith('{"case"'):
                r = json.loads(line)
                out[r["case"]] = r
    return out


a, b = load(args.a), load(args.b)


def ms(r):
    best = r.get("best") or {}
    if best:
        return best["mass"], best["safetyFactor"], best["volumeFraction"]
    return (r.get("exchanger") or {}).get("mass"), None, None


print("| case | gallery g / SF | plain g / SF (fraction) | IM g / SF (fraction) | mass IM vs plain | pass plain / IM |")
print("|---|---|---|---|---|---|")
tot = [0.0, 0.0]
for name in a:
    if name not in b:
        continue
    (ma, sa, fa), (mb, sb, fb) = ms(a[name]), ms(b[name])
    g = gal.get(name)
    if sa is not None:
        tot[0] += ma
        tot[1] += mb
    f = lambda m, s, fr: f"{m:.1f} / {s:.2f} ({fr:.3f})" if s is not None else f"{m:.1f}"  # noqa: E731
    print(f"| {name} | {f'{g[0]:.1f} / {g[1]:.2f}' if g else '-'} | {f(ma, sa, fa)} | {f(mb, sb, fb)} | "
          f"{mb / ma - 1:+.1%} | {a[name]['pass']} / {b[name]['pass']} |")
print(f"| total (structural) | | {tot[0]:.1f} | {tot[1]:.1f} | {tot[1] / tot[0] - 1:+.1%} | |")
