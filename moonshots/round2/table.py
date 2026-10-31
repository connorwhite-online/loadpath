"""Print basins.py results as a markdown table: python table.py result.json"""
import json
import sys

for path in sys.argv[1:]:
    d = json.load(open(path))
    print(f"\n### {d['study']} ({d['cells']} cells)\n")
    print("| vf | strategy | mass g | SF | peak MPa | defl mm | margin | dist to nearest | iters | s |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for r in d["rows"]:
        dist = "" if r.get("minDistance") is None else f"{r['minDistance']:.2f}"
        print(f"| {r['vf']:.2f} | {r['strategy']} | {r['mass']:.1f} | {r['sf']:.2f} | {r['peak']:.2f} | "
              f"{r['deflection']:.3f} | {r['margin']:.3f} | {dist} | {r['iterations']} | {r['seconds']:.0f} |")
    by = {}
    for r in d["rows"]:
        by.setdefault(r["vf"], []).append(r)
    print("\n| vf | main SF | best SF (strategy) | SF range | main margin | best margin (strategy) | runs | s |")
    print("|---|---|---|---|---|---|---|---|")
    for vf, rs in sorted(by.items(), reverse=True):
        m = next(r for r in rs if r["strategy"] == "main")
        bs = max(rs, key=lambda r: r["sf"])
        bm = max(rs, key=lambda r: r["margin"])
        print(f"| {vf:.2f} | {m['sf']:.2f} | {bs['sf']:.2f} ({bs['strategy']}) | "
              f"{min(r['sf'] for r in rs):.2f}-{max(r['sf'] for r in rs):.2f} | {m['margin']:.3f} | "
              f"{bm['margin']:.3f} ({bm['strategy']}) | {len(rs)} | {sum(r['seconds'] for r in rs):.0f} |")
