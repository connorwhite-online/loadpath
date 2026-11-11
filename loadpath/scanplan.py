"""
From a scan and a sentence to a study: which feature the part mounts on,
which face it holds, what it carries, and (for a bar) a split clamp whose
bolts are sized by friction.

The agent turns the user's sentence ("mount this 120 g light under my
handlebar, mountain biking") into a brief:

    {"mountOn": "bar",            # "bar", "wall", or a feature id
     "family": "handlebar",       # optional: the standard sizes the bar can be (scanstd.FAMILIES)
     "hold": None,                # optional: the face it holds (feature id); else the object's
                                  # face that looks at the mount
     "mass": 0.12,                # kg the part carries
     "shock": 5,                  # g of shock, on top of gravity: load cases along each
                                  # axis both ways (MIL-STD-810's way); the clamp is sized
                                  # for the worst direction of all
     "clamp": "split"}            # bar mounts: "split" (two halves, bolted) or "ring" (slides on)

plan_from_brief matches it to the fitted features and returns support and
loads by id, or the questions that still need an answer (asks): no mass,
no shock, two bars of a size, a bar whose band covers several standard
sizes. Nothing is guessed: what the scan cannot say, it asks, once.

clamp_plan sizes the split clamp from the loads:

- the clamp must not slip: the friction it grips with (mu x the normal force
  round the bar) holds the torque about the bar and the axial force, both
  from the object's mass centre, times the safety factor. With bolts' total
  preload F pulling the halves together, the normal force round the bar is
  (8/pi) F for a cosine pressure (contact concentrated where the halves
  push; uniform pressure gives pi F, so this is the low reading), so the
  grip is mu (8/pi) F in force and mu (8/pi) F r in torque;
- the joint must not open: F above the load that pulls the halves apart;
- each bolt's preload is what the part can bear under the washer without
  creeping: a quarter of the material's strength over the washer's ring
  (fasteners.py's reading: plastics relax under sustained bearing stress;
  retorque after a day), and no more than 60% of an 8.8 bolt's proof load;
- the tightening torque is K F d, K = 0.2 (dry steel threads, Shigley);
- the halves are cut apart by a gap that closes on the smallest bar the
  scan's band allows in the largest hole the process may print, so the
  bore grips before the faces meet.

split_clamp cuts the optimized part into body and cap at that gap.
"""
from __future__ import annotations

import json
import math
import os

import numpy as np

__all__ = ["plan_from_brief", "clamp_plan", "split_clamp", "WASHERS", "MU_DEFAULT"]

G = 9.81
# ISO 7089 washer OD, ISO 4032 nut across flats and height, ISO 898-1 8.8
# proof stress x the thread's stress area (N): mm, mm, mm, N
WASHERS = {"M3": 7.0, "M4": 9.0, "M5": 10.0, "M6": 12.0}
NUTS = {"M3": (5.5, 2.4), "M4": (7.0, 3.2), "M5": (8.0, 4.7), "M6": (10.0, 5.2)}
PROOF = {"M3": 580 * 5.03, "M4": 580 * 8.78, "M5": 580 * 14.2, "M6": 580 * 20.1}
NOMINAL = {"M3": 3.0, "M4": 4.0, "M5": 5.0, "M6": 6.0}
# nylon on dry metal: handbook static friction 0.2 to 0.4; the low end, so
# the clamp holds on a greasy bar (give mu for a rubber shim or a knurl)
MU_DEFAULT = 0.2
K_TORQUE = 0.2


def _unit(v):
    v = np.asarray(v, float)
    return v / max(np.linalg.norm(v), 1e-12)


def _gravity(brief, scan):
    g = brief.get("gravity", "-z")
    if isinstance(g, str):
        s = -1.0 if g.strip().startswith("-") else 1.0
        return s * np.eye(3)["xyz".index(g.strip()[-1].lower())]
    return _unit(g)


def _bar_point(f, x):
    a = np.asarray(f["axis"], float)
    p0 = np.asarray(f["p0"], float)
    t = (np.asarray(x, float) - p0) @ a
    return p0 + t * a


def _object_points(scan, seed_point, exclude_masks=()):
    """The scan points of the body that holds `seed_point`: points linked
    within 2.5 spacings, minus the excluded (the bar)."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    from scipy.spatial import cKDTree
    keep = np.ones(len(scan.points), bool)
    for m in exclude_masks:
        keep &= ~m
    idx = np.flatnonzero(keep)
    P = scan.points[idx]
    tree = cKDTree(P)
    rad = 2.5 * float(np.median(scan.spacing))
    pairs = tree.query_pairs(rad, output_type="ndarray")
    n = len(P)
    G_ = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(n, n))
    _, lab = connected_components(G_, directed=False)
    i0 = int(tree.query(seed_point)[1])
    return idx[lab == lab[i0]]


def plan_from_brief(scan, feats: dict, brief: dict) -> dict:
    """Support, loads and asks for draft_around from a brief (module
    docstring). feats are snapped to the standard family first (in place).
    Returns {"support", "loads", "asks", "picked", "families", "centre"}."""
    from .scanstd import snap_features
    brief = dict(brief or {})
    asks, picked = [], {}
    fam = brief.get("family")
    fams = [fam] if isinstance(fam, str) else fam
    snap_features(feats, fams)
    byid = {f["id"]: f for f in feats.get("planes", []) + feats.get("cylinders", [])}
    g = _gravity(brief, scan)
    mount = brief.get("mountOn", "bar")
    mf = None
    if mount in byid:
        mf = byid[mount]
    elif mount == "bar":
        bars = sorted((c for c in feats.get("cylinders", []) if c["kind"] == "bar"), key=lambda c: -c["length"])
        if fams:
            fit = [c for c in bars if c.get("standard") or c.get("standardChoices")]
            bars = fit or bars
        if not bars:
            asks.append({"ask": "mount", "question": "the scan has no bar (a convex cylinder) to mount on: "
                                                     "which feature does the part hold on to?",
                         "options": sorted(byid)})
        else:
            mf = bars[0]
            if len(bars) > 1 and bars[1]["length"] > 0.8 * bars[0]["length"]:
                asks.append({"ask": "mount", "question": "two bars of about the same length: which one?",
                             "options": [b["id"] for b in bars[:4]], "default": bars[0]["id"]})
            picked["mount"] = f"{mf['id']}: the longest bar ({mf['length']:.0f} mm scanned)"
    elif mount == "wall":
        walls = sorted((p for p in feats.get("planes", []) if abs(np.asarray(p["normal"]) @ g) < 0.3),
                       key=lambda p: -p["area"])
        if walls:
            mf = walls[0]
            picked["mount"] = f"{mf['id']}: the largest upright face ({mf['area']:.0f} mm^2)"
        else:
            asks.append({"ask": "mount", "question": "no upright face to mount on: which feature?",
                         "options": sorted(byid)})
    else:
        raise ValueError(f"mountOn must be 'bar', 'wall' or a feature id (have {sorted(byid)})")

    # ---- the face held: the object's face that faces the mount most
    hf = byid.get(brief["hold"]) if brief.get("hold") else None
    if mf is not None and hf is None:
        cands = []
        for p in feats.get("planes", []):
            if p is mf:
                continue
            c = np.asarray(p["center"], float)
            q = _bar_point(mf, c) if mf["type"] == "cylinder" else (
                c - ((c - np.asarray(mf["center"])) @ np.asarray(mf["normal"])) * np.asarray(mf["normal"]))
            d = q - c
            dist = float(np.linalg.norm(d))
            if dist < 1e-9:
                continue
            look = float(np.asarray(p["normal"], float) @ (d / dist))
            if look > 0.3:
                # the solid angle the face fills as seen from the mount
                cands.append((p["area"] * look / dist ** 2, p))
        if cands:
            om, hf = max(cands, key=lambda c: c[0])
            picked["hold"] = (f"{hf['id']}: the face that fills the most of the view from the mount "
                              f"({hf['area']:.0f} mm^2, {om:.2f} sr)")
        else:
            asks.append({"ask": "hold", "question": "no face of the object looks at the mount: which face does "
                                                    "the part hold?", "options": [p["id"] for p in feats.get("planes", [])]})

    # ---- what it carries
    mass, shock = brief.get("mass"), brief.get("shock")
    if mass is None:
        asks.append({"ask": "mass", "question": "what does the object weigh (g)? a kitchen scale is enough"})
    if shock is None:
        asks.append({"ask": "shock", "question": "how hard is it shaken: the worst shock in g on top of gravity "
                                                 "(a bike on rough trail sees several g)?"})
    # ---- the bar's size: a pick when the band covers several standard sizes
    if mf is not None and mf["type"] == "cylinder" and mf.get("standardChoices"):
        asks.append({"ask": "size", "feature": mf["id"],
                     "question": f"the bar measures {mf['diameter']:.2f} +- {mf['diameterBand']:.2f} mm: "
                                 f"which size is it?", "options": mf["standardChoices"]})
    out = {"support": [mf["id"]] if mf is not None else [], "loads": [], "asks": asks, "picked": picked,
           "families": fams, "gravity": g.tolist()}
    if hf is None or mf is None:
        return out
    # the mass centre: the held body's points (a uniform shell stands in for the mass)
    excl = []
    if mf["type"] == "cylinder":
        d = np.linalg.norm(np.cross(scan.points - np.asarray(mf["p0"]), np.asarray(mf["axis"])), axis=1)
        excl.append(np.abs(d - mf["r"]) < max(3 * mf["rBand"], 3 * float(np.median(scan.spacing))))
    ids = _object_points(scan, np.asarray(hf["center"]), excl)
    cg = scan.points[ids].mean(0)
    out["centre"] = cg.round(3).tolist()
    picked["centre"] = (f"the held body's surface centroid ({len(ids)} points) stands in for its mass centre")
    if mass is not None and shock is not None:
        m = float(mass)
        at = cg.round(3).tolist()
        weight = {"feature": hf["id"], "mass": m, "at": at, "gravity": g.round(6).tolist()}
        if float(shock) > 0:
            # the shock along each axis both ways, as MIL-STD-810 (method 516)
            # applies it; the clamp itself is sized for every direction
            sN = m * G * float(shock)
            out["loads"] = [{"name": f"shock{sg}{ax}", "loads": [
                weight, {"feature": hf["id"], "force": (sN * (1 if sg == "+" else -1) * np.eye(3)[i]).round(3).tolist(),
                         "at": at}]} for i, ax in enumerate("xyz") for sg in "+-"]
        else:
            out["loads"] = [{"name": "weight", "loads": [weight]}]
        out["carry"] = {"mass": m, "shock": float(shock), "weightN": round(m * G, 3),
                        "shockN": round(m * G * float(shock), 3)}
    return out


def _demand(bar, cg, weight_vec, shock_n, sf):
    """Worst slip demand (N at the bar's surface: sqrt((T/r)^2 + F_axial^2))
    and worst pull normal to the split over shock directions, x sf."""
    a = np.asarray(bar["axis"], float)
    q = _bar_point(bar, cg)
    arm = np.asarray(cg, float) - q
    k = 4000
    i = np.arange(k) + 0.5
    phi = np.arccos(1 - 2 * i / k)
    th = math.pi * (1 + 5 ** 0.5) * i
    U = np.stack([np.cos(th) * np.sin(phi), np.sin(th) * np.sin(phi), np.cos(phi)], 1)
    F = weight_vec[None, :] + shock_n * U
    T = np.cross(arm[None, :], F) @ a
    Fa = F @ a
    slip = np.sqrt((T / bar["r"]) ** 2 + Fa ** 2)
    j = int(np.argmax(slip))
    return {"slip": float(sf * slip[j]), "torque": float(sf * abs(T[j])), "axial": float(sf * abs(Fa[j])),
            "F": F, "arm": arm, "q": q, "worstForce": F[j].round(3).tolist()}


def clamp_plan(bar: dict, ring: tuple, carry: dict, cg, toward, *, material="pa12", process="sls",
               bar_fit="slip", sf=2.0, mu=MU_DEFAULT, wall=1.5) -> dict:
    """Size a two-half clamp round `bar` (module docstring). ring: (t_lo,
    t_hi) the clamp's span along the bar from p0; carry: {"mass" kg,
    "shock" g}; cg: the carried mass centre; toward: the direction from the
    bar to what the part holds (the body's side; the cap goes opposite)."""
    from .fits import BOLT_HOLES, HOLES, fit_window
    from .materials import material_from_spec
    from .cadimport import ring_wall
    mat = material_from_spec(material)
    a = _unit(bar["axis"])
    n = np.asarray(toward, float)
    n = _unit(n - (n @ a) * a)            # the split plane's normal, across the bar
    lat = np.cross(a, n)
    r = float(bar["r"])
    R = r + ring_wall(r)
    W = float(carry["mass"]) * G
    gvec = np.asarray(carry.get("gravity", [0, 0, -1]), float)
    dem = _demand(bar, cg, W * _unit(gvec), W * float(carry["shock"]), sf)
    # pull that opens the joint: the body pulled away from the bar (along +n)
    pull = float(sf * max(0.0, float((dem["F"] @ n).max())))
    need = max(dem["slip"] / (mu * 8.0 / math.pi), pull)
    p_allow = 0.25 * float(mat.strength)
    t_lo, t_hi = ring
    span = t_hi - t_lo
    choice = None
    for nb in (2, 4):
        for size in ("M3", "M4", "M5", "M6"):
            dw = WASHERS[size]
            hole = BOLT_HOLES[size][1]
            if nb == 4 and span < 2 * (dw + wall):
                continue
            A = math.pi / 4 * (dw ** 2 - hole ** 2)
            Fb = min(p_allow * A, 0.6 * PROOF[size])
            if nb * Fb >= need:
                choice = (nb, size, Fb, A, dw, hole)
                break
        if choice:
            break
    rows = {"mu": mu, "muNote": "nylon on dry metal, the low end of 0.2-0.4 (give mu for a shim or knurl)",
            "safetyFactor": sf, "slipDemandN": round(dem["slip"], 2), "torqueDemandNmm": round(dem["torque"], 1),
            "axialDemandN": round(dem["axial"], 2), "openingDemandN": round(pull, 2),
            "preloadNeededN": round(need, 1), "bearingAllowMPa": round(p_allow, 2),
            "bearingNote": "a quarter of the strength under the washer: plastics relax under sustained bearing; "
                           "retorque after a day"}
    if choice is None:
        rows["verdict"] = "no clamp of up to four M6 bolts grips this load: a longer clamp, a knurled or " \
                          "rubber-lined bore, or a pinned bar"
        return {"ok": False, **rows}
    nb, size, Fb, A, dw, hole = choice
    d = NOMINAL[size]
    # bolts beside the bore, across the split
    e = r + 0.5 * hole + max(wall, 0.5 * (dw - hole))
    ts = [0.5 * (t_lo + t_hi)] if nb == 2 else [t_lo + 0.5 * dw + wall, t_hi - 0.5 * dw - wall]
    p0 = np.asarray(bar["p0"], float)
    bolts = []
    for t in ts:
        for side in (-1, 1):
            base = p0 + t * a + side * e * lat
            bolts.append({"seat": (base - R * n).round(4).tolist(), "nut": (base + R * n).round(4).tolist()})
    # the gap: closes on the smallest bar in the largest hole the process prints
    mu_b = 0.5 * float(bar["diameterBand"])
    mate = float(bar["diameter"])
    win = fit_window(bar_fit, mate, "plastic" if float(mat.density) < 2.0 else "metal")
    lo, hi = win["lo"] + mu_b, win["hi"] - mu_b
    target = 0.5 * (lo + hi) if lo <= hi else mate + mu_b + win["clearance"][0]
    key = next((k for k in HOLES if str(process or "").lower().startswith(k)), None)
    tolp = (HOLES[key]["tol"] + HOLES[key]["tolPerMm"] * mate) if key else 0.0
    close = (target + tolp) - (mate - mu_b)
    gap = round(max(close, 0.0) + tolp, 3)
    grip_n = nb * Fb * mu * 8.0 / math.pi
    out = {"ok": True, **rows, "bolts": nb, "size": size, "washer": dw, "nut": NUTS[size],
           "preloadN": round(Fb, 1), "tighteningTorqueNm": round(K_TORQUE * Fb * d / 1000.0, 3),
           "torqueNote": "T = 0.2 F d (dry steel threads); with a threadlocker or oil, K 0.15-0.18",
           "gripN": round(grip_n, 1), "slipMargin": round(grip_n / max(dem["slip"], 1e-9) * sf, 2),
           "openMargin": round(nb * Fb / max(pull, 1e-9) * sf, 2) if pull > 0 else None,
           "gap": gap, "gapNote": f"closes {close:.2f} mm on the smallest bar ({mate - mu_b:.2f} mm) in the largest "
                                  f"hole the process prints ({target + tolp:.2f} mm), plus the print's scatter",
           "split": {"point": _bar_point(bar, p0 + 0.5 * (t_lo + t_hi) * a).round(4).tolist(),
                     "normal": n.round(6).tolist()},
           "boltPositions": bolts, "lateral": lat.round(6).tolist(), "offset": round(e, 3),
           "capDepth": round(R, 3), "span": [round(t_lo, 3), round(t_hi, 3)]}
    if close > 0.1 * mate:
        out["warning"] = (f"the scan's band leaves {close:.1f} mm to close: the halves meet the bar at two lines, "
                          "not round it; measure the bar (known) for a clamp that grips all round")
    return out


def clamp_study_parts(plan: dict, bar: dict, design_box) -> dict:
    """The fasteners, keep-outs and design space a split clamp adds to a
    study: bolts through the cap into nuts in the body, a driver path over
    each nut, and the cap side of the split limited to the clamp."""
    a = _unit(bar["axis"])
    n = np.asarray(plan["split"]["normal"], float)
    lat = np.asarray(plan["lateral"], float)
    q = np.asarray(plan["split"]["point"], float)
    lo, hi = (np.asarray(v, float) for v in design_box)
    corners = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])]) - q
    fr = np.stack([a, lat, n])
    cc = corners @ fr.T
    clo, chi = cc.min(0), cc.max(0)
    # the body's side: n >= 0 (a sliver past it so the ring stays whole across the cut)
    body_lo = np.array([clo[0], clo[1], -0.5 * plan["gap"] - 0.5])
    body_hi = chi.copy()
    t_lo, t_hi = plan["span"]
    p0 = np.asarray(bar["p0"], float)
    ta = (p0 - q) @ a
    w = plan["washer"]
    cap_lo = np.array([ta + t_lo, -(plan["offset"] + 0.5 * w + 1.5), -(plan["capDepth"] + 1.0)])
    cap_hi = np.array([ta + t_hi, plan["offset"] + 0.5 * w + 1.5, 0.0])

    def obox(l_, h_):
        c = q + fr.T @ (0.5 * (l_ + h_))
        return {"type": "box", "center": c.round(4).tolist(), "size": (h_ - l_).round(4).tolist(),
                "axes": [a.round(6).tolist(), lat.round(6).tolist()]}
    design = [obox(body_lo, body_hi), obox(cap_lo, cap_hi)]
    fasteners, keep = [], []
    af, hnut = plan["nut"]
    for i, b in enumerate(plan["boltPositions"]):
        seat, nut = np.asarray(b["seat"]), np.asarray(b["nut"])
        fasteners.append({"id": f"clamp-{i}", "size": plan["size"], "seat": seat.tolist(),
                          "direction": n.round(6).tolist(), "grip": round(float(np.linalg.norm(nut - seat)), 3),
                          "end": "face", "head": plan["washer"], "preload": plan["preloadN"]})
        # the nut and a nut driver's path out of the part (driver OD about AF + 4 mm)
        reach = float(np.linalg.norm(hi - lo))
        keep.append({"id": f"clamp-{i}-nut", "bore": False, "shape": {
            "type": "cylinder", "p0": nut.round(4).tolist(), "p1": (nut + reach * n).round(4).tolist(),
            "r": round(0.5 * (af + 4.0), 3)}})
    return {"designSpace": design, "fasteners": fasteners, "keepOut": keep}


def split_clamp(mesh, plan: dict, out_dir: str, name: str = "clamp") -> dict:
    """Cut the optimized part (a mesh or STL path) into body and cap at the
    plan's split: a slab `gap` thick through the bar's axis. Writes
    <name>-body.stl and <name>-cap.stl and returns their volumes and
    bodies."""
    import trimesh
    from .shapes import manifold_from_trimesh, trimesh_from_manifold
    m = trimesh.load(mesh, force="mesh") if isinstance(mesh, str) else mesh
    q = np.asarray(plan["split"]["point"], float)
    n = _unit(plan["split"]["normal"])
    g = float(plan["gap"])
    size = 4.0 * float(np.max(m.extents)) + 4.0 * float(np.linalg.norm(m.centroid - q))
    T = np.eye(4)
    z = np.array([0, 0, 1.0])
    v = np.cross(z, n)
    if np.linalg.norm(v) > 1e-9:
        T[:3, :3] = trimesh.transformations.rotation_matrix(math.atan2(np.linalg.norm(v), z @ n), v)[:3, :3]
    elif n[2] < 0:
        T[:3, :3] = np.diag([1, -1, -1])
    M = manifold_from_trimesh(m)
    out = {}
    for label, sgn in (("body", 1.0), ("cap", -1.0)):
        half = trimesh.creation.box(extents=[size, size, size])
        T2 = T.copy()
        T2[:3, 3] = q + n * sgn * (0.5 * g + 0.5 * size)
        half.apply_transform(T2)
        piece = trimesh_from_manifold(M ^ manifold_from_trimesh(half))
        path = os.path.join(out_dir, f"{name}-{label}.stl")
        piece.export(path)
        out[label] = {"path": path, "volume": round(float(piece.volume), 1),
                      "bodies": len(piece.split(only_watertight=False)) if len(piece.faces) else 0}
    with open(os.path.join(out_dir, f"{name}-split.json"), "w") as fh:
        json.dump({"plan": plan, "pieces": out}, fh, indent=1)
    return out
