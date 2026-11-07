"""
The fire-force map of a push pad, per press point against its window
(docs/images/switch/): one column per press, the window as a band, one
marker per design. Reads a JSON of {design: {"forceGf": [...]}, "presses":
[{"at", "min", "max"}]} (touch_mic.py writes the rows).

    python examples/switch/force_chart.py forces.json out.png "round 17 as drawn" "skin, base and rails only" "delivered mesh=loadpath"
"""
import json
import sys

import numpy as np

# categorical slots 1-4 of the dataviz reference palette, in fixed order;
# markers differ too, so identity is never color alone
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
MARKERS = ["o", "s", "D", "^"]
INK, MUTED, SURFACE = "#0b0b0b", "#52514e", "#fcfcfb"


def chart(doc, rows, out, title, ylim=(70, 2000)):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pr = doc["presses"]
    order = sorted(range(len(pr)), key=lambda i: (pr[i]["max"], pr[i]["at"][0]))
    x = np.arange(len(order))
    fig, ax = plt.subplots(figsize=(10, 4.6), dpi=200, facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    for j, i in enumerate(order):
        ax.add_patch(plt.Rectangle((j - 0.32, pr[i]["min"]), 0.64, pr[i]["max"] - pr[i]["min"],
                                   facecolor="#e6e5e0", edgecolor="none", zorder=1))
    ax.plot([], [], color="#e6e5e0", lw=8, label="target window")
    n = len(rows)
    for k, (key, label) in enumerate(rows):
        f = np.asarray(doc[key]["forceGf"])[order]
        off = (k - (n - 1) / 2) * 0.16
        ax.scatter(x + off, f, s=34, marker=MARKERS[k], color=COLORS[k], edgecolor=SURFACE, linewidth=1.2,
                   zorder=3, label=label)
    ax.set_yscale("log")
    ax.set_ylim(*ylim)
    ticks = [100, 150, 200, 300, 500, 1000, 1500]
    ax.set_yticks(ticks)
    ax.set_yticklabels([str(t) for t in ticks])
    ax.minorticks_off()
    ax.set_xticks(x)
    ax.set_xticklabels([f"({pr[i]['at'][0]:g}, {pr[i]['at'][1]:g})" for i in order], rotation=40, ha="right",
                       fontsize=8, color=MUTED)
    ax.set_xlabel("press point, mm from the switch (window ceiling rises left to right)", color=MUTED, fontsize=9)
    ax.set_ylabel("force to click, gf", color=MUTED, fontsize=9)
    ax.tick_params(colors=MUTED, labelsize=8, length=0)
    ax.grid(axis="y", color="#e6e5e0", lw=0.8, zorder=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(title, loc="left", color=INK, fontsize=11)
    ax.legend(frameon=False, fontsize=8, ncol=min(n + 1, 3), loc="upper left", labelcolor=INK)
    fig.tight_layout()
    fig.savefig(out, facecolor=SURFACE)
    return out


def main():
    doc = json.load(open(sys.argv[1]))
    rows = []
    for a in sys.argv[3:]:
        key, _, label = a.partition("=")
        rows.append((key, label or key))
    chart(doc, rows, sys.argv[2], "Touch microphone push pad: force to click at each press point")
    print(sys.argv[2])


if __name__ == "__main__":
    main()
