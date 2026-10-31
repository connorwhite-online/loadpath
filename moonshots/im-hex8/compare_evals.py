"""Per-case mass, SF and time from eval logs (the {"case": ...} lines
loadpath.evals.run prints), against the gallery numbers in README.md.
Usage: python compare_evals.py LOG [LOG ...]"""
import json
import re
import sys

readme = open(__file__.rsplit("/moonshots/", 1)[0] + "/README.md").read()
base = {m.group(1): (float(m.group(2)), float(m.group(3)))
        for m in re.finditer(r"<b>([\w-]+)</b><br><sub>([\d.]+) g · \d+% lighter · SF ([\d.]+)", readme)}
rows = {}
for path in sys.argv[1:]:
    for line in open(path):
        if line.startswith('{"case"'):
            r = json.loads(line)
            rows[r["case"]] = r
print("| case | baseline mass g / SF | IM mass g / SF | mass change | fraction | pass | failed checks | seconds |")
print("|---|---|---|---|---|---|---|---|")
for name, r in rows.items():
    b = r.get("best") or {}
    ex = r.get("exchanger")
    mass = b.get("mass") if b else (ex or {}).get("mass")
    sf = b.get("safetyFactor")
    old = base.get(name)
    failed = [c["check"] for c in r.get("checks", []) if not c.get("pass", True)]
    ch = f"{mass / old[0] - 1:+.1%}" if old and mass else ""
    print(f"| {name} | {f'{old[0]:.1f} / {old[1]:.2f}' if old else '-'} | "
          f"{mass:.1f} / {sf:.2f} | {ch} | {b.get('volumeFraction', '')} | {r.get('pass')} | {','.join(failed) or '-'} | "
          f"{r.get('seconds', 0):.0f} |" if sf is not None else
          f"| {name} | - | {mass:.1f} / - | | | {r.get('pass')} | {','.join(failed) or '-'} | {r.get('seconds', 0):.0f} |")
