"""
The part to hand over, in one place: report.json["deliver"].

A run's report lists every outcome it tried; an agent wants the one it
should give the user, whether it passes, which number to quote as its safety
factor (and from which analysis), what set its size, and every warning that
matters for it, gathered from the outcome, its print and process checks, its
durability and the run. Agents testing the brief evals kept missing print and
fatigue warnings that sat three levels down, and quoting the voxel safety
factor where the report had replaced it with a finer one.
"""
from __future__ import annotations

_VF_CEILING = 0.7   # study._VF_MAX: the search never goes past it


def attention(o: dict, study: dict) -> list[str]:
    """What SKILL.md tells an agent to look for, already looked for."""
    out = []
    if o.get("passes") is False:
        out.append("does not meet the goal: raise the volume fraction or the material, or check the loads")
    if (o.get("bodies") or 1) > 1:
        out.append(f"{o['bodies']} separate bodies: something is not connected to a support")
    if (o.get("keepOutIntrusion") or 0) > 1.0:
        out.append(f"keepOutIntrusion {o['keepOutIntrusion']:.1f} mm³: material in a keep-out (report a bug)")
    if (o.get("preserveMissing") or 0) > 1.0:
        out.append(f"preserveMissing {o['preserveMissing']:.1f} mm³: preserved geometry lost (report a bug)")
    mm = study.get("minMember")
    p5 = _p5(o)
    if mm and p5 is not None and p5 < 0.8 * mm:
        out.append(f"p5Member {p5:.2f} mm is under minMember {mm} mm: raise the resolution")
    if o.get("closedBores"):
        out.append(f"closed bores: {o['closedBores']}")
    dur = o.get("durability") or {}
    if dur.get("safetyFactor") is not None and dur["safetyFactor"] < dur.get("target", 1.5):
        out.append(f"durability safety factor {dur['safetyFactor']:.2f} (fatigue/creep) is under "
                   f"{dur.get('target', 1.5):g}: set durability.constrain to size for it")
    out.extend(dur.get("warnings") or [])
    return out


def _p5(o: dict):
    if o.get("p5Member") is not None:
        return o["p5Member"]
    return (o.get("confirm") or {}).get("p5Member")


def _sf_method(o: dict) -> str:
    c = o.get("confirm")
    if not c:
        return "voxel"
    return c.get("method") or "voxel-fine"


def _governing(o: dict, goal: dict) -> str | None:
    """Which of the goal's limits the delivered part is closest to."""
    ms = {}
    if goal.get("safetyFactor") and o.get("safetyFactor"):
        ms["safetyFactor"] = o["safetyFactor"] / goal["safetyFactor"]
    if goal.get("maxDeflection"):
        d = max((c.get("deflection") or 0.0) for c in o.get("cases") or [{}])
        if d > 0:
            ms["maxDeflection"] = goal["maxDeflection"] / d
    dur = o.get("durability") or {}
    if dur.get("safetyFactor") is not None and dur.get("constrain"):
        ms["durability"] = dur["safetyFactor"] / dur.get("target", 1.5)
    return min(ms, key=ms.get) if ms else None


def delivery(summary: dict, study: dict) -> dict | None:
    """report.json["deliver"]: the best outcome, flattened, with every
    warning that applies to it."""
    outs = summary.get("outcomes") or []
    best = summary.get("best")
    if best is None:
        if not outs:
            return None
        best = max(range(len(outs)), key=lambda i: outs[i].get("safetyFactor") or 0.0)
    o = outs[best]
    goal = study.get("goal") or {}
    warn = []
    for w in (list(summary.get("warnings") or []) + list(o.get("warnings") or [])
              + list((o.get("print") or {}).get("warnings") or [])
              + list((o.get("process") or {}).get("warnings") or []) + attention(o, study)):
        if w not in warn:
            warn.append(w)
    cases = [{"name": c.get("name"), "safetyFactor": c.get("safetyFactor"), "deflection": c.get("deflection"),
              **({"worstDirection": c["worstDirection"]} if c.get("worstDirection") else {})}
             for c in o.get("cases") or []]
    d = {
        "outcome": best,
        "passes": o.get("passes") if summary.get("best") is not None else False,
        "stl": o.get("stl"),
        **({"3mf": o["3mf"]} if o.get("3mf") else {}),
        **({"render": o["render"]} if o.get("render") else {}),
        "material": o.get("material"),
        "mass": o.get("mass"),
        "volumeFraction": o.get("volumeFraction"),
        "safetyFactor": o.get("safetyFactor"),
        # which analysis that number comes from: "tet" (body-fitted cross-check),
        # "voxel-fine" (a finer voxel re-analysis) or "voxel" (the design grid)
        "safetyFactorMethod": _sf_method(o),
        **({"target": {k: goal[k] for k in ("safetyFactor", "maxDeflection") if k in goal}}
           if any(k in goal for k in ("safetyFactor", "maxDeflection")) else {}),
        "governedBy": _governing(o, goal),
        "maxDeflection": max((c.get("deflection") or 0.0) for c in o.get("cases") or [{}]) or None,
        "cases": cases,
        "p5Member": _p5(o),
        **({"durabilitySafetyFactor": o["durability"].get("safetyFactor")} if o.get("durability") else {}),
        **{k: o[k] for k in ("bodies", "keepOutIntrusion", "preserveMissing") if o.get(k) is not None},
        **({"process": o["process"]} if o.get("process") else {}),
        **({"print": {k: v for k, v in o["print"].items() if k != "warnings"}} if o.get("print") else {}),
        "warnings": warn,
    }
    if summary.get("best") is None:
        d["warnings"].insert(0, "no outcome meets the goal: this is the strongest one tried, not a passing part")
        sat = _saturated(outs)
        if sat:
            d["warnings"].insert(1, sat)
        top = max((x.get("volumeFraction") or 0.0) for x in outs)
        if top >= _VF_CEILING - 1e-6:
            d["warnings"].insert(1, f"the search stopped at the volume-fraction ceiling ({_VF_CEILING:.0%} of the design "
                                    "space): more material is not allowed, so change the material, the design space or the loads")
    return d


def _saturated(outs: list) -> str | None:
    """When more material hardly helps, the shape is not the problem: a notch
    (a keep-out's corner, a thin interface) caps the safety factor."""
    pts = sorted((o["volumeFraction"], o["safetyFactor"], o) for o in outs
                 if o.get("volumeFraction") and o.get("safetyFactor"))
    import math

    # a sound layout's safety factor grows at least as fast as material
    # (roughly vf^1 to vf^2); under vf^0.6 the material is not where it fails
    if len(pts) < 2 or pts[-1][0] < 1.3 * pts[0][0] or pts[-1][1] <= 0 or pts[0][1] <= 0 or \
            math.log(pts[-1][1] / pts[0][1]) > 0.6 * math.log(pts[-1][0] / pts[0][0]):
        return None
    worst = min(pts[-1][2].get("cases") or [{}], key=lambda c: c.get("safetyFactor") or float("inf"))
    at = worst.get("peakAt")
    where = f" at {[round(v, 1) for v in at]} (case {worst.get('name')!r})" if at else ""
    return (f"the safety factor barely rises with material ({pts[0][1]:.2f} at {pts[0][0]:.0%} of the design "
            f"space, {pts[-1][1]:.2f} at {pts[-1][0]:.0%}): a local hot spot{where} sets it, so look at the "
            "geometry there (a keep-out notching an interface, a thin preserved ring, a fastener's head path) "
            "rather than adding material")
