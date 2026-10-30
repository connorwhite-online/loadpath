"""
Procedurally generated eval cases, so tuning stops fitting GRIP.

Every hand-written case is a GRIP-like part, and a fix tuned on them can win
there and lose everywhere else without anyone noticing. A generated case is
drawn from a seed: a family (link, bracket, tri-mount), random proportions,
bore sizes, axis directions, load directions, material and target. The same
seed always gives the same case, so a corpus is a fixed, reproducible
held-out set, and a fresh seed range is a part nobody has tuned on.

Loads are sized from a beam estimate so the solid design space would carry
them at several times the target safety factor: every case is feasible and
needs real material removal. Expectations are the generic ones every part
must meet (one body, nothing in keep-outs, preserved geometry kept, bores
open, SF met); the score that matters is the Pareto massAtTarget, compared
run to run.

    python -m loadpath.evals.run --corpus 8            # seeds 0-7
    python -m loadpath.evals.run --corpus 100:108      # a fresh range
    python -m loadpath.evals.corpus 3                  # print seed 3's study
"""
from __future__ import annotations

import json
import sys

import numpy as np

FAMILIES = ("link", "bracket", "tri-mount")
MATERIALS = ("pla", "petg", "pa12", "asa")
_AX = {"x": 0, "y": 1, "z": 2}
# solid design space carries the load at this many times the target SF
_HEADROOM = (4.0, 8.0)


def _r(x, nd=1):
    return float(round(float(x), nd))


def _bore(rid, center, axis, bore_r, boss_r, length, through=40.0):
    """A preserved boss ring round a bore keep-out, both along `axis`."""
    c = [_r(v) for v in center]
    return ({"id": f"{rid}-boss", "shape": {"type": "cylinder", "center": c, "axis": axis,
                                            "length": _r(length), "r": _r(boss_r)}},
            {"id": f"{rid}-bore", "shape": {"type": "cylinder", "center": c, "axis": axis,
                                            "length": _r(length + through), "r": _r(bore_r)}})


def _force(rng, size, mat, sf, span, section):
    """N for a load whose cantilever bending stress on the solid section is
    strength / (sf * headroom)."""
    b, h = sorted(section)
    k = rng.uniform(*_HEADROOM)
    return mat.strength * b * h * h / (6.0 * span * sf * k)


def generate(seed: int, family: str | None = None, max_elements: int = 8000) -> dict:
    from ..materials import material_from_spec

    rng = np.random.default_rng(seed)
    family = family or FAMILIES[seed % len(FAMILIES)]
    mat_name = MATERIALS[int(rng.integers(len(MATERIALS)))]
    mat = material_from_spec(mat_name)
    sf = float(rng.choice([2.0, 2.5, 3.0]))
    min_member = _r(rng.uniform(2.5, 3.5))
    preserve, keep, supports, loads = [], [], [], []

    if family == "link":
        # two bosses on a bar; bore axes both z (a planar link) or crossed
        L = rng.uniform(70, 150)
        W = rng.uniform(26, 44)
        T = rng.uniform(16, 28)
        r1 = rng.uniform(3.5, 7.0)
        r2 = rng.uniform(3.0, 6.5)
        crossed = rng.random() < 0.35
        lo, hi = [-W / 2 - 2, -W / 2, -T / 2], [L + W / 2 + 2, W / 2, T / 2]
        a2 = "y" if crossed else "z"
        len2 = W if crossed else T
        for rid, c, ax, br, ln in (("a", [0, 0, 0], "z", r1, T), ("b", [L, 0, 0], a2, r2, len2)):
            p, k_ = _bore(rid, c, ax, br, min(br + rng.uniform(3.5, 6.0), 0.5 * (W if ax == "z" else T)), ln)
            preserve.append(p)
            keep.append(k_)
        supports.append({"region": "a-bore"})
        d = rng.normal(size=3)
        d[0] *= 0.3     # mostly across the link: bending, not tension
        d /= np.linalg.norm(d)
        F = _force(rng, None, mat, sf, L, (W, T))
        loads.append({"region": "b-bore", "force": [_r(F * v) for v in d]})
        symmetry = ["z"] if not crossed and abs(d[2]) < 0.15 else []
    elif family == "bracket":
        # a plate bolted to a wall at x=0 (2-4 bolts), loaded at a boss out at +x
        L = rng.uniform(60, 120)
        W = rng.uniform(50, 80)
        H = rng.uniform(50, 80)
        n = int(rng.integers(2, 5))
        lo, hi = [0, -W / 2, -H / 2], [L, W / 2, H / 2]
        br = rng.uniform(1.6, 2.75)
        ax = "y" if rng.random() < 0.5 else "z"   # the load boss's bore
        ey, ez = W / 2 - 7, H / 2 - 7
        if n == 2:
            # two bolts either side of the load boss, not in line with it
            holes = [(0.0, -ez), (0.0, ez)] if ax == "y" else [(-ey, 0.0), (ey, 0.0)]
        else:
            holes = [(-ey, -ez), (ey, ez), (-ey, ez), (ey, -ez)][:n]
        for i, (y, z) in enumerate(holes):
            p, k_ = _bore(f"bolt{i}", [3, y, z], "x", br, br + rng.uniform(2.5, 4.0), 6)
            preserve.append(p)
            keep.append(k_)
            # head access out to the far side: nothing may cover the bolt head
            keep.append({"id": f"bolt{i}-head", "shape": {"type": "cylinder", "p0": [6, _r(y), _r(z)],
                                                           "p1": [_r(L + 1), _r(y), _r(z)], "r": _r(2 * br + 1)}})
            supports.append({"region": f"bolt{i}-bore"})
        lr = rng.uniform(3.0, 5.0)
        c = [L - lr - 5, 0, 0]
        p, k_ = _bore("load", c, ax, lr, lr + rng.uniform(3.5, 4.5), W if ax == "y" else H)
        preserve.append(p)
        keep.append(k_)
        d = rng.normal(size=3)
        d[0] = -abs(d[0]) * 0.5
        d /= np.linalg.norm(d)
        # the bolts and the load boss carry it, not the whole section: a
        # sixth of the beam estimate (seeds 1, 4 missed SF at 67% fill)
        F = _force(rng, None, mat, sf, L, (W, H)) / 6.0
        loads.append({"region": "load-bore", "force": [_r(F * v) for v in d]})
        symmetry = []
    elif family == "tri-mount":
        # three bosses on a triangle in a plate: two hold, one is loaded
        R = rng.uniform(35, 70)
        T = rng.uniform(14, 26)
        ang = np.sort(rng.uniform(0, 2 * np.pi, 3))
        ang = ang[0] + np.array([0, 2.1, 4.2]) + rng.uniform(-0.3, 0.3, 3)
        pts = [[R * np.cos(a), R * np.sin(a), 0.0] for a in ang]
        lo = [min(p[0] for p in pts) - 14, min(p[1] for p in pts) - 14, -T / 2]
        hi = [max(p[0] for p in pts) + 14, max(p[1] for p in pts) + 14, T / 2]
        for i, c in enumerate(pts):
            br = rng.uniform(2.5, 5.5)
            p, k_ = _bore(f"m{i}", c, "z", br, br + rng.uniform(3.5, 6.0), T)
            preserve.append(p)
            keep.append(k_)
        supports += [{"region": "m0-bore"}, {"region": "m1-bore"}]
        d = rng.normal(size=3)
        d /= np.linalg.norm(d)
        span = float(np.linalg.norm(np.array(pts[2]) - 0.5 * (np.array(pts[0]) + np.array(pts[1]))))
        F = _force(rng, None, mat, sf, max(span, 20.0), (hi[1] - lo[1], T)) / 3.0
        loads.append({"region": "m2-bore", "force": [_r(F * v) for v in d]})
        symmetry = []
    else:
        raise ValueError(f"unknown family {family!r}: {FAMILIES}")

    name = f"corpus-{family}-{seed}"
    return {
        "name": name,
        "description": f"Generated ({family}, seed {seed}): not tuned on. Load sized so the solid design "
                       f"space has several times the target SF.",
        "designSpace": [{"type": "box", "min": [_r(v) for v in lo], "max": [_r(v) for v in hi]}],
        "preserve": preserve, "keepOut": keep, "supports": supports,
        "loadCases": [{"name": "load", "loads": loads}],
        "material": mat_name,
        "goal": {"safetyFactor": sf, "volumeFraction": 0.3},
        "minMember": min_member,
        **({"symmetry": symmetry} if symmetry else {}),
        "resolution": {"maxElements": int(max_elements)},
        "expect": {"minSafetyFactor": sf, "bodies": 1, "maxKeepOutIntrusion": 2.0,
                   "maxPreserveMissing": 2.0, "maxClosedBores": 0, "minMassReduction": 0.3},
    }


def seeds(spec: str) -> list[int]:
    """'8' -> 0..7, '100:108' -> 100..107, '3,5,9' -> those."""
    if ":" in spec:
        a, b = spec.split(":")
        return list(range(int(a), int(b)))
    if "," in spec:
        return [int(s) for s in spec.split(",")]
    return list(range(int(spec)))


if __name__ == "__main__":
    print(json.dumps(generate(int(sys.argv[1]), *(sys.argv[2:3] or [None])), indent=1))
