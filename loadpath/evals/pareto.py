"""
Pareto scoring: how good the mass / safety-factor trade-off is, not just
whether the pick cleared a line.

Pass/fail hides progress. A case that gets 15% lighter at the same safety
factor scores the same as before, and a case that misses its target by 2%
scores the same as one that misses by 60%. Every outcome a study tries is a
point (mass, SF); the lightest-for-its-SF points form a front, and the front
says how much mass the target costs whatever the search happened to pick.

    massAtTarget   g, the front interpolated to the goal's safety factor
                   (lower is better; the number to watch). Between the
                   heaviest failing and the lightest passing front point,
                   linear in SF. With no failing point below it, it is the
                   lightest passing mass (an upper bound: "bracketed" false).
    hypervolume    area the front dominates in (1 - mass / solid mass,
                   min(SF / target, 2)) from (0, 0): 0..2, higher is better.
                   Rewards the whole curve, so a search that explores well
                   and finds a better trade-off away from the target still
                   moves it.
    front          the non-dominated points, lightest first.
"""
from __future__ import annotations

SF_CAP = 2.0   # SF / target beyond this buys nothing


def front(points):
    """Non-dominated (mass, sf) points: no other point is at most as heavy and
    at least as safe (strictly better in one). Lightest first."""
    pts = sorted({(float(m), float(s)) for m, s in points if m is not None and s is not None},
                 key=lambda p: (p[0], -p[1]))
    out, best_sf = [], float("-inf")
    for m, s in pts:
        if s > best_sf:
            out.append((m, s))
            best_sf = s
    return out


def mass_at(fr, target):
    """(mass, bracketed) where the front reaches `target` SF, or (None, False)
    when no front point reaches it."""
    passing = [p for p in fr if p[1] >= target]
    if not passing:
        return None, False
    hi = min(passing, key=lambda p: p[0])
    below = [p for p in fr if p[1] < target and p[0] <= hi[0]]
    if not below:
        return hi[0], False
    lo = max(below, key=lambda p: p[0])
    if hi[1] - lo[1] <= 1e-12:
        return hi[0], True
    t = (target - lo[1]) / (hi[1] - lo[1])
    return lo[0] + t * (hi[0] - lo[0]), True


def hypervolume(fr, solid_mass, target):
    """Area dominated by the front in (saving, SF/target) from (0, 0), with
    saving = 1 - mass / solid_mass and SF/target capped at SF_CAP."""
    if not fr or not solid_mass or not target:
        return None
    # in (saving up, safety up) both are maximized; sort by saving descending
    pts = sorted(((1.0 - m / solid_mass, min(s / target, SF_CAP)) for m, s in fr), key=lambda p: -p[0])
    area, prev_y = 0.0, 0.0
    for x, y in pts:
        if y > prev_y and x > 0:
            area += x * (y - prev_y)
            prev_y = y
    return area


def score(outcomes, target):
    """Pareto summary of a study's outcomes (Outcome.summary() dicts or
    anything with mass, safetyFactor, massReduction)."""
    pts = [(o.get("mass"), o.get("safetyFactor")) for o in outcomes]
    fr = front(pts)
    solid = None
    for o in outcomes:
        m, r = o.get("mass"), o.get("massReduction")
        if m and r is not None and r < 1.0:
            solid = m / (1.0 - r)
            break
    out = {"points": len([p for p in pts if None not in p]), "front": [[m, s] for m, s in fr],
           "solidMass": solid}
    if target:
        m, br = mass_at(fr, target)
        out.update(target=target, massAtTarget=m, bracketed=br,
                   hypervolume=hypervolume(fr, solid, target))
    return out
