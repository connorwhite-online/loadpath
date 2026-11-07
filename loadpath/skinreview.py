"""
Where a fixed exterior keeps a push pad off its force map, and what the
designer could change: feedback, never an edit.

`loadpath switch` designs everything inside an industrial designer's
exterior. The exterior itself is kept as a cosmetic skin of a fixed
thickness, and sometimes that skin alone decides a press: on the touch
microphone the bare 1.0 mm skin at one corner already needs 391 gf against
a 300 gf ceiling, and the best interior only brings it to 323. No interior
can fix that. Only the designer can, by thinning the skin there, moving
the rim, or accepting a higher ceiling there.

`skin_review` reads that out of a design, for a person and for an agent:

- **skin only**: the fire force of the bare skin (with the preserved base,
  rails, plunger) at every press point, against its window. A skin that is
  over a ceiling on its own is a risk the interior may not clear. A skin
  under a floor is normal, because the interior's job is to stiffen it.
- **misses**: every press the design misses. Each says which side it
  missed, by how much (gf and percent), the bare skin's reading, how much
  the interior already moved it, and what limits it: `skin` when the bare
  skin alone misses on the same side, `interior` when it does not (a finer
  grid or more iterations can fix that, and the exterior needs nothing).
- **where**: the skin cells whose stiffness sets that press's force most,
  from the force's sensitivity to each cell (one adjoint pair, already
  solved): their centroid, extent and skin thickness.
- **suggestions** for a skin-limited miss, each with its numbers read back
  through the same linear model, the interior held as designed:
  * `target`: the ceiling (or floor) at that press that the design meets
    as it is.
  * `localSkin`: the skin thickness within a radius of the press that
    brings it into its window, for a few radii. Each one lists the presses
    the change pushes out of their own windows and whether the thickness is
    at or over the process's minimum wall.
  * `skinThickness`: one thickness for the whole skin that clears every
    skin-limited miss, with the same side effects listed.

The interior is held fixed in every suggestion. A re-run with the
designer's change can only do as well or better (the held interior is
still a feasible start), so a suggestion is a sufficient change, not the
least one. Thinning in the linear model scales the skin cells' density
(E ~ x^3: bending like the thinner skin, membrane stiffness reading low,
as in switch._thin_skin). Nothing here changes the study, the skin or the
delivered mesh.
"""
from __future__ import annotations

import numpy as np

__all__ = ["skin_review", "skin_preflight", "review_options", "skin_cells", "min_skin_wall"]

# A press is in window within 2% (as switch.py counts it).
TOL = 0.02
# Suggestions aim this far inside the window, so rounding the thickness to
# 0.05 mm does not land it back on the edge.
AIM = 0.03
RADII = (4.0, 8.0)
SCALE_RANGE = (0.3, 2.0)


def skin_cells(study, domain, floor=None):
    """(skin share phi per element, base density of the skin per element).

    phi is the skin's volume share in each cell (4^3 samples). base is the
    density the design gives the skin there: the thin-skin floor where
    there is one (switch._thin_skin), 1 for a skin of preserved whole cells
    (a grid fine enough for it), 0 off the skin. Cells of other preserves
    (the plunger, a base ring) are never skin."""
    from .domain import _fraction

    n = domain.n_elem
    if not any(rid == "skin" for rid, _ in study.shells):
        return np.zeros(n), np.zeros(n)
    shape, _ = study.region_shape("skin")
    c = domain.mesh.elem_centers()
    sub = (np.arange(4) + 0.5) / 4 - 0.5
    offs = np.array([[a, b, d] for a in sub for b in sub for d in sub]) * domain.pitch
    phi = np.zeros(len(c))
    for o in offs:
        phi += shape.contains(c + o)
    phi /= len(offs)
    others = [r.shape for r in study.preserve if r.id != "skin"]
    hard = _fraction(others, c, domain.pitch) >= 0.5 if others else np.zeros(n, bool)
    phi[hard] = 0.0
    base = np.zeros(n)
    if floor is not None and np.any(floor > 0):
        base = np.where(floor > 0, floor, 0.0)
    thick = domain.preserve & (phi > 0) & (base == 0) & ~hard
    base[thick] = 1.0
    return phi, base


def min_skin_wall(study) -> tuple[float | None, str]:
    """The thinnest skin the process makes, and where that number came from:
    switch.skinReview.minWall, else the process's rule card (processcheck),
    else the material's own rule (TPU powder tears thin walls out), else
    the study's minMember."""
    from .processcheck import _MATERIAL_RULES, RULES, rules_for

    b = (study.raw.get("switch") or {}).get("skinReview")
    if isinstance(b, dict) and b.get("minWall") is not None:
        return float(b["minWall"]), "switch.skinReview.minWall"
    proc = study.process
    mat = getattr(study.material, "name", None)
    kind = str(getattr(proc, "kind", "") or "").lower()
    if kind in RULES:
        return float(rules_for(proc, mat)["minWall"]), f"the {kind} rule card (processcheck)"
    rule = _MATERIAL_RULES.get(str(mat or "").lower(), {}).get("minWall")
    if rule is not None:
        return float(rule), f"the {mat} rule (processcheck)"
    if study.min_member:
        return float(study.min_member), "minMember"
    return None, "none given"


def _verdict(f, p):
    if p.max is not None and f > p.max * (1 + TOL):
        return "over"
    if p.min is not None and f < p.min * (1 - TOL):
        return "under"
    return "in"


def _ok(f, p):
    return _verdict(f, p) == "in"


def _r(v, k=1):
    return None if v is None else round(float(v), k)


class _Reader:
    """Fire forces of a design with the skin scaled in some cells."""

    def __init__(self, domain, nu, spec, modulus):
        from .switch import _problem, _Solves
        self.domain, self.nu, self.spec, self.modulus = domain, nu, spec, modulus
        self.prob = _problem(domain, spec)
        self.solves = _Solves(domain, nu)
        self.calls = 0

    def forces(self, xp, grads=False):
        from .switch import _evaluate
        self.calls += 1
        out, G = _evaluate(self.domain, self.nu, self.prob, np.maximum(xp, 1e-9), self.modulus, self.solves,
                           grads=grads)
        return out, G


def _rescale(xp, base, mask, s):
    """The design with the skin in `mask` at s times its thickness. A skin
    cell's density is the skin's share plus the interior's share of the
    rest (the as-printed split, switch._as_printed): only the skin's part
    is scaled."""
    x = np.array(xp, float)
    b = base[mask]
    extra = np.where(b < 1.0, np.clip((x[mask] - b) / np.maximum(1.0 - b, 1e-9), 0.0, 1.0), 0.0)
    sb = s * b
    x[mask] = sb + extra * np.clip(1.0 - sb, 0.0, None)
    return x


def _solve_scale(reader, xp, base, mask, i, goal, side, F, budget=8):
    """The skin scale s on `mask` that brings press i's force to `goal`
    (secant on log F against log s, bracketed in SCALE_RANGE). Returns
    (s, forces at s, reached)."""
    lo, hi = SCALE_RANGE
    s0, F0 = 1.0, F
    s1 = 0.85 if side == "over" else 1.2
    best = None
    for _ in range(budget):
        F1 = reader.forces(_rescale(xp, base, mask, s1))[0]["F"]
        reached = F1[i] <= goal if side == "over" else F1[i] >= goal
        if reached and (best is None or abs(np.log(s1)) < abs(np.log(best[0]))):
            best = (s1, F1)
        # aim 1% past the goal: the force flattens out as the skin thins
        # (the rest of the part takes over), and a secant on a flattening
        # curve creeps up on the goal from the wrong side
        aim = goal * (0.99 if side == "over" else 1.01)
        g0, g1 = np.log(F0[i] / aim), np.log(F1[i] / aim)
        if abs(g1) < 0.005 and reached:
            break
        slope = (g1 - g0) / (np.log(s1) - np.log(s0)) if s1 != s0 else 0.0
        if not np.isfinite(slope) or abs(slope) < 1e-6:
            break
        if slope <= 0:
            # a thinner skin here makes this press stiffer: the region is
            # not where its force comes from
            break
        s2 = float(np.exp(np.log(s1) - g1 / slope))
        s2 = float(np.clip(s2, lo, hi))
        if abs(s2 - s1) < 1e-4:
            break
        s0, F0, s1 = s1, F1, s2
    if best is None:
        return s1, F1, False
    return best[0], best[1], True


def _side_effects(forces, presses, skip, before):
    """Other presses a change moves out of their windows (or back in)."""
    out = []
    for j, (f, p) in enumerate(zip(forces, presses)):
        if j in skip:
            continue
        was, now = _verdict(before[j], p), _verdict(f, p)
        if was != now:
            out.append({"press": j, "at": [_r(v) for v in p.at], "from": was, "to": now,
                        "forceGf": _r(f / _gf()), "beforeGf": _r(before[j] / _gf())})
    return out


def _gf():
    from .switch import GF
    return GF


def _where(domain, G_i, base, phi, xp, axis):
    """The skin cells that set press i's force: the ones whose stiffness
    holds most of its positive sensitivity (thinning them lowers the
    force), up to half of it. Centroid, extent and their mean thickness."""
    skin = base > 0
    sens = np.where(skin, G_i * base, 0.0)
    pos = np.clip(sens, 0.0, None)
    if pos.sum() <= 0:
        return None
    order = np.argsort(-pos)
    cum = np.cumsum(pos[order])
    k = int(np.searchsorted(cum, 0.5 * cum[-1])) + 1
    cells = order[:k]
    c = domain.mesh.elem_centers()[cells]
    w = pos[cells]
    cen = (c * w[:, None]).sum(0) / w.sum()
    others = [a for a in range(3) if a != axis]
    rad = float(np.sqrt((w * ((c[:, others] - cen[others]) ** 2).sum(1)).sum() / w.sum()))
    return {"center": [_r(v) for v in cen], "radius": _r(rad), "cells": int(k),
            "shareOfSensitivity": 0.5}


def skin_review(study, domain, spec, xphys, floor, nu=None, modulus=None, label: str = "design",
                radii=RADII, suggest: bool = True) -> dict:
    """The skin review of a design (see the module docstring). `xphys` is the
    design's density on `domain` (preserve included, thin-skin cells at or
    above `floor`); pass the bare skin as `xphys` for a preflight. Returns a
    JSON-ready dict."""
    import time

    t0 = time.time()
    gf = _gf()
    nu = study.elastic(design=True) if nu is None else nu
    modulus = float(study.material.E) if modulus is None else float(modulus)
    floor = np.zeros(domain.n_elem) if floor is None else np.asarray(floor, float)
    xp = np.maximum(np.asarray(xphys, float), floor)
    phi, base = skin_cells(study, domain, floor)
    skin_t = next((t for rid, t in study.shells if rid == "skin"), None)
    wall, wall_src = min_skin_wall(study)
    out = {"policy": "Suggestions only: loadpath never changes the exterior. Each number is read through the "
                     "linear fire-force model with the interior held as designed; re-run `loadpath switch` with "
                     "the designer's change to re-optimize the interior.",
           "reading": label, "skinThickness": skin_t, "minWall": wall, "minWallFrom": wall_src,
           "skinAtMinWall": None if (wall is None or skin_t is None) else bool(skin_t <= wall + 1e-9)}
    if skin_t is None or not np.any(base > 0):
        out.update({"skinLimited": 0, "misses": [], "summary": "The study has no skin, so there is nothing to "
                                                               "review."})
        return out
    reader = _Reader(domain, nu, spec, modulus)
    bare = np.maximum(domain.preserve.astype(float), floor)
    F_bare = reader.forces(bare)[0]["F"]
    design_is_bare = bool(np.allclose(xp, bare))
    o, G = reader.forces(xp, grads=suggest)
    F = o["F"]
    presses = spec.presses
    out["skinOnly"] = [{"press": i, "at": [_r(v) for v in p.at], "maxGf": _r(p.max / gf if p.max else None),
                        "minGf": _r(p.min / gf if p.min else None), "forceGf": _r(fb / gf),
                        "verdict": _verdict(fb, p)} for i, (p, fb) in enumerate(zip(presses, F_bare))]
    misses = []
    limited = []
    for i, (p, f, fb) in enumerate(zip(presses, F, F_bare)):
        v = _verdict(f, p)
        if v == "in":
            continue
        edge = p.max if v == "over" else p.min
        vb = _verdict(fb, p)
        # Over a ceiling with the bare skin already over it: the skin is the
        # stiffness, and the interior can only carry part of the press to
        # the plunger. Under a floor is the interior's to fix (it stiffens).
        by = "skin" if (v == "over" and vb == "over") else "interior"
        m = {"press": i, "at": [_r(q) for q in p.at], "side": v, "windowGf": [_r(p.min / gf if p.min else None),
                                                                              _r(p.max / gf if p.max else None)],
             "forceGf": _r(f / gf), "byGf": _r((f - edge) / gf), "byPercent": _r(100 * (f / edge - 1)),
             "skinOnlyGf": _r(fb / gf), "limitedBy": by}
        if not design_is_bare:
            m["interiorMovedGf"] = _r((f - fb) / gf)
        if by == "interior":
            m["note"] = ("Under its floor with no interior: an interior stiffens it. Nothing for the designer."
                         if design_is_bare else
                         "The bare skin does not miss on this side, so the interior can fix it: a finer pitch or "
                         "more iterations. Nothing for the designer.")
        if G is not None:
            m["where"] = _where(domain, G["logF"][i], base, phi, xp, spec.press_axis)
        misses.append(m)
        if by == "skin":
            limited.append(m)
    # a skin that is over a ceiling on its own, where the design is in window:
    # the interior cleared it, worth knowing when the exterior changes
    out["skinOnlyOver"] = [r["press"] for r in out["skinOnly"] if r["verdict"] == "over"]
    if suggest:
        mask_all = base > 0
        others_axes = [a for a in range(3) if a != spec.press_axis]
        cen = domain.mesh.elem_centers()

        def option(mask, i, side, group):
            """A skin thickness on `mask` for press i, rounded to 0.05 mm on the
            safe side and read back; with the forces at the minimum wall when
            the thickness it takes is under it."""
            p = presses[i]
            goal = p.max * (1 - AIM) if side == "over" else p.min * (1 + AIM)
            s, _, reached = _solve_scale(reader, xp, base, mask, i, goal, side, F)
            row = {"fromThickness": skin_t, "toThickness": None, "reached": False}
            if reached:
                step = np.floor if side == "over" else np.ceil
                t_new = float(step(skin_t * s / 0.05 - 1e-6) * 0.05)
                Fs = reader.forces(_rescale(xp, base, mask, t_new / skin_t))[0]["F"]
                still = [k for k in group if _verdict(Fs[k], presses[k]) != "in"]
                row.update({"toThickness": round(t_new, 2), "reached": not still, "forceGf": _r(Fs[i] / gf),
                            "stillMissing": still,
                            "printable": None if wall is None else bool(t_new >= wall - 1e-9),
                            "sideEffects": _side_effects(Fs, presses, group, F),
                            "forcesGf": [_r(f / gf) for f in Fs]})
                if wall is not None and t_new < wall - 1e-9 and wall < skin_t - 1e-9:
                    Fw = reader.forces(_rescale(xp, base, mask, wall / skin_t))[0]["F"]
                    row["atMinWall"] = {"thickness": wall, "forceGf": _r(Fw[i] / gf),
                                        "sideEffects": _side_effects(Fw, presses, group, F)}
            return row

        for m in limited:
            i = m["press"]
            p = presses[i]
            side = m["side"]
            sug = []
            # the window the design meets as it is
            if side == "over":
                new = float(np.ceil(F[i] / gf / 5.0) * 5.0)
                sug.append({"kind": "target", "change": "ceiling", "fromGf": _r(p.max / gf), "toGf": new,
                            "text": f"raise the ceiling at {m['at']} to {new:g} gf"})
            else:
                new = float(np.floor(F[i] / gf / 5.0) * 5.0)
                sug.append({"kind": "target", "change": "floor", "fromGf": _r(p.min / gf), "toGf": new,
                            "text": f"lower the floor at {m['at']} to {new:g} gf"})
            at = np.asarray(p.at, float)
            at2 = at[others_axes] if len(at) == 3 else at[:2]
            d2 = np.linalg.norm(cen[:, others_axes] - at2, axis=1)
            discs = [("press", at2, R, d2) for R in radii]
            wh = m.get("where")
            if wh:
                # round the skin that holds most of this press's stiffness
                # (often the fold where the top turns into a wall, not
                # under the finger)
                hc = np.asarray(wh["center"], float)[others_axes]
                discs.append(("stiffestSkin", hc, float(max(radii[0], np.ceil(wh["radius"]))),
                              np.linalg.norm(cen[:, others_axes] - hc, axis=1)))
            for around, c2, R, dd in discs:
                mask = mask_all & (dd <= R)
                if not mask.any():
                    continue
                row = {"kind": "localSkin", "around": around, "center": [_r(v) for v in c2], "radius": R,
                       **option(mask, i, side, [i])}
                if row["toThickness"] is not None:
                    verb = "thin" if row["toThickness"] < skin_t else "thicken"
                    row["text"] = (f"{verb} the skin within {R:g} mm of {row['center']} from {skin_t:g} to "
                                   f"{row['toThickness']:.2f} mm: {row['forceGf']:g} gf there")
                    if row.get("printable") is False:
                        row["text"] += f", under the {wall:g} mm minimum wall ({wall_src})"
                else:
                    row["text"] = (f"no skin thickness within {R:g} mm of {row['center']} from "
                                   f"{SCALE_RANGE[0] * skin_t:.2f} to {SCALE_RANGE[1] * skin_t:.2f} mm brings it "
                                   "into its window")
                sug.append(row)
            m["suggestions"] = sug
        # one thickness for the whole skin that clears every skin-limited miss
        if limited:
            group = [k["press"] for k in limited]
            j = max(limited, key=lambda k: abs(k["byPercent"]))["press"]
            row = {"kind": "skinThickness", "forPresses": group, **option(mask_all, j, limited[0]["side"], group)}
            out["skinThickness"] = row
    n_sk = len(limited)
    if n_sk:
        parts = []
        for m in limited[:3]:
            local = [s for s in m.get("suggestions", []) if s["kind"] == "localSkin" and s.get("reached")]
            ok = [s for s in local if s.get("printable") is not False]
            # the gentlest change: the thickest skin that does it
            best = max(ok, key=lambda s: s["toThickness"]) if ok else None
            tgt = next((s for s in m.get("suggestions", []) if s["kind"] == "target"), None)
            txt = (f"at {m['at']} it needs {m['forceGf']:g} gf, {abs(m['byPercent']):g}% over its "
                   f"{m['windowGf'][1]:g} gf ceiling")
            if not design_is_bare:
                txt += f" (the bare skin alone: {m['skinOnlyGf']:g} gf)"
            if best:
                txt += f". Fix: {best['text']}"
            elif local:
                lo = max(local, key=lambda s: s["toThickness"])
                txt += (f". It takes a {lo['toThickness']:.2f} mm skin within {lo['radius']:g} mm of "
                        f"{lo['center']}, under the "
                        f"{wall:g} mm minimum wall, so {tgt['text'] if tgt else 'raise the ceiling'}")
            elif tgt:
                txt += f". No local skin change found, so {tgt['text']}"
            parts.append(txt)
        more = f" ({n_sk - 3} more listed under misses)" if n_sk > 3 else ""
        whole = out.get("skinThickness") or {}
        tail = ""
        if whole.get("reached"):
            se = whole.get("sideEffects") or []
            tail = (f" One skin thickness for all of it: {whole['toThickness']:.2f} mm everywhere " +
                    (f"(under the {wall:g} mm minimum wall), " if whole.get("printable") is False else "") +
                    ("with no other press leaving its window." if not se else
                     f"but {len(se)} other press{'es' if len(se) > 1 else ''} would leave "
                     f"{'their' if len(se) > 1 else 'its'} window (skinThickness.sideEffects)."))
        out["summary"] = (("The bare skin alone" if design_is_bare else "The fixed skin") +
                          f" is over the ceiling at {n_sk} press{'es' if n_sk > 1 else ''}{more}: " +
                          "; ".join(parts) + "." + tail + " This is the designer's call: loadpath does not change "
                          "the exterior.")
    elif misses:
        out["summary"] = (f"{len(misses)} press{'es' if len(misses) > 1 else ''} miss, but the skin is not the "
                          "cause: " + ("an interior stiffens them." if design_is_bare else
                                       "the interior can fix them (a finer pitch or more iterations).") +
                          " Nothing for the designer.")
    else:
        out["summary"] = "Every press is in its window; the exterior needs nothing."
    out["skinLimited"] = n_sk
    out["misses"] = misses
    out["solves"] = reader.calls
    out["seconds"] = round(time.time() - t0, 1)
    return out


def review_options(study) -> dict | None:
    """switch.skinReview: true (the default), false, or {minWall, radii}.
    None when it is off."""
    b = (study.raw.get("switch") or {}).get("skinReview", True)
    if b is False:
        return None
    b = b if isinstance(b, dict) else {}
    return {"radii": tuple(float(r) for r in b.get("radii", RADII)), "gridCheck": b.get("gridCheck", "auto")}


# A grid this coarse against the skin or the thinnest member reads forces it
# cannot be trusted with: the touch microphone on 1 mm cells (a 1.0 mm skin,
# 1.0 mm members) read round 17's centre at 129 gf, where 0.7 and 0.5 mm
# cells agree on 177-181, and loadpath's 1 mm design, 13 of 14 in window
# there, read 262-992 gf on 0.7 mm cells.
COARSE = 0.75
# A press whose force moves more than this between the grids, or changes
# verdict, is not converged.
GRID_TOL = 0.10


def grid_resolution(study, pitch: float) -> dict:
    """Cells across the skin and across the thinnest member at this pitch,
    and whether that is coarse (pitch > COARSE x the thinner of the two)."""
    skin_t = next((t for rid, t in study.shells if rid == "skin"), None)
    member = float(study.min_member) if study.min_member else None
    sizes = [v for v in (skin_t, member) if v]
    thin = min(sizes) if sizes else None
    return {"pitch": float(pitch), "cellsAcrossSkin": _r(skin_t / pitch, 2) if skin_t else None,
            "cellsAcrossMember": _r(member / pitch, 2) if member else None,
            "coarse": bool(thin is not None and pitch > COARSE * thin + 1e-9),
            "checkPitch": round(COARSE * thin, 3) if thin else None}


def grid_check(study, pitch: float, fine: float, coarse_forces, part=None, coarse_part=None) -> dict:
    """Re-read the bare skin (and, given `part`, the delivered mesh) on a
    finer grid, the way a drawn part is read (its volume share per cell over
    the skin's floor), and compare with the coarse grid's forces press by
    press. coarse_forces: the bare skin's forces on the coarse grid (N);
    coarse_part: the part's (N, e.g. the delivered mesh's read-back)."""
    import time

    from .domain import _fraction, build_domain
    from .shapes import MeshShape
    from .switch import GF, _thin_skin, fire_forces, switch_from_study

    t0 = time.time()
    nu = study.elastic(design=True)
    E = float(study.material.E)
    dom, floor = _thin_skin(study, build_domain(study, fine))
    spec = switch_from_study(study, dom)
    bare = np.maximum(dom.preserve.astype(float), floor)
    rows = {"bareSkin": (coarse_forces, fire_forces(dom, nu, spec, np.maximum(bare, 1e-9), E)["forces"])}
    if part is not None and coarse_part is not None:
        drawn = _fraction([MeshShape(part)], dom.mesh.elem_centers(), dom.pitch)
        rows["part"] = (coarse_part, fire_forces(dom, nu, spec, np.maximum(np.maximum(drawn, floor), 1e-9),
                                                 E)["forces"])
    out = {"pitch": float(pitch), "finePitch": float(dom.pitch)}
    converged = True
    for name, (fc, ff) in rows.items():
        fc, ff = np.asarray(fc, float), np.asarray(ff, float)
        change = ff / np.maximum(fc, 1e-12) - 1.0
        flips = [i for i, p in enumerate(spec.presses) if _verdict(fc[i], p) != _verdict(ff[i], p)]
        ok = bool(np.abs(change).max() <= GRID_TOL and not flips)
        converged &= ok
        out[name] = {"coarseGf": [_r(v / GF) for v in fc], "fineGf": [_r(v / GF) for v in ff],
                     "maxChangePercent": _r(100 * float(np.abs(change).max())),
                     "verdictChanged": flips, "inWindowFine": int(sum(_verdict(v, p) == "in"
                                                                       for v, p in zip(ff, spec.presses))),
                     "converged": ok}
    out["converged"] = bool(converged)
    out["seconds"] = round(time.time() - t0, 1)
    return out


def note_grid(review: dict, res: dict, check: dict | None) -> dict:
    """Put the grid's verdict on a review: `grid` (resolution and check) and,
    when the grid is coarse, a warning at the head of the summary."""
    review["grid"] = {**res, **({"check": check} if check else {})}
    warn = None
    if check and not check["converged"]:
        worst = max((v for k, v in check.items() if isinstance(v, dict)), key=lambda v: v["maxChangePercent"])
        what = "the part" if "part" in check and not check["part"]["converged"] else "the bare skin"
        warn = (f"Grid too coarse: on {check['finePitch']:g} mm cells {what} reads up to "
                f"{worst['maxChangePercent']:g}% different from {check['pitch']:g} mm cells"
                + (f", and {len(worst['verdictChanged'])} press(es) change verdict" if worst["verdictChanged"] else "")
                + ". Treat every number below as unconverged and re-run finer (switch.pitches ending at "
                  f"{check['finePitch']:g} or less) before showing the designer.")
    elif res["coarse"] and not check:
        warn = (f"Coarse grid: {res['cellsAcrossSkin']} cells across the skin and {res['cellsAcrossMember']} across "
                f"the thinnest member at {res['pitch']:g} mm, unchecked. Confirm on {res['checkPitch']:g} mm cells "
                "before showing the designer.")
    if warn:
        review["trusted"] = False
        review["summary"] = warn + " " + review["summary"]
    else:
        review["trusted"] = True
    return review


def skin_preflight(study, pitch: float | None = None) -> dict:
    """The skin review of the bare skin, before any interior is designed:
    which presses the exterior alone puts over their ceilings, where, and
    what would bring them in, and on a coarse grid whether the bare skin
    holds on finer cells (grid_check)."""
    from .domain import build_domain
    from .switch import _thin_skin, switch_from_study

    b = study.raw.get("switch") or {}
    pitch = pitch or (b.get("pitches") or [study.pitch or None])[0]
    domain, floor = _thin_skin(study, build_domain(study, pitch))
    spec = switch_from_study(study, domain)
    bare = np.maximum(domain.preserve.astype(float), floor)
    opts = review_options(study) or {"radii": RADII}
    rep = skin_review(study, domain, spec, bare, floor, label="bare skin", radii=opts["radii"])
    rep["pitch"] = domain.pitch
    res = grid_resolution(study, domain.pitch)
    check = None
    if opts["gridCheck"] is not False and (res["coarse"] or isinstance(opts["gridCheck"], (int, float))):
        fine = float(opts["gridCheck"]) if isinstance(opts["gridCheck"], (int, float)) and \
            not isinstance(opts["gridCheck"], bool) else res["checkPitch"]
        F = [r["forceGf"] * _gf() for r in rep["skinOnly"]]
        check = grid_check(study, domain.pitch, fine, F)
    return note_grid(rep, res, check)
