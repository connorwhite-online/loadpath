"""
Desk stability: will a product stay put where it is set down?

A desk accessory (a button, a dock, a hub, a stand) is pressed, tugged by
its cable and nudged all day while it sits loose on a table. Nothing holds
it but its weight and the friction of its feet. This module answers, for a
rigid product on a flat desk:

- **presses**: a finger pushing at a point on it, the push leaning up to
  `cone` degrees off the surface normal in any direction. Does the product
  slide, or tip up onto an edge of its feet?
- **cables**: a cable at a port pulling with `force` N along `direction`
  (a cable hanging off the desk's edge, or dragged when the laptop moves),
  and a sideways `lever` force at the plug (a stiff cable turned to one
  side) that twists the product about its feet.
- **plug**: pushing a plug in with the product loose on the desk (the
  connector's insertion force): almost never held by weight alone, and the
  report says so, so the instructions can say to hold it.

For each it gives the margin (1 = on the edge of moving), and the least
mass that would hold it with the centre of mass where it is (`minMass`),
so a designer can see what weight a product needs and where ballast helps.

The physics (statics of a rigid body on Coulomb friction):

- Weight W = m g at the centre of mass c. Feet: the contact patches (the
  faces within `tol` of the lowest point, facing down), their convex hull
  the support polygon P.
- A load F at point a, with horizontal part H and vertical part V (V > 0
  pushes down). Normal reaction N = W + V must be > 0 (else it lifts).
- Slide: |H| <= mu N. A twist (moment T about the vertical through the
  centre of pressure) on top: the limit surface of a uniformly loaded patch
  is close to the ellipse (|H| / mu N)^2 + (T / (mu N r_eff))^2 <= 1, with
  r_eff the patch's mean distance from the centre of pressure.
- Tip: the centre of pressure p = (W c_xy + V a_xy - H a_z) / N (moments
  about the desk plane) must lie in P. The tip margin is the distance from
  p to P's nearest edge over the distance from c to it (1 at rest, 0 at
  the tipping point, < 0 tipped).
- minMass: the least m with slide and tip margins >= 0, by bisection
  (monotone in m for a downward or horizontal load).

What it does not model: the feet's compliance (a soft foot rocks before it
tips: read the tip margin as an upper bound), desk tilt, vibration, sticky
feet (adhesion), suction. Friction coefficients are typical values for dry,
clean pairs (`FRICTION`); a dusty desk is lower, so the default design
value is the low end.

Inputs: a brief (`loadpath schema stability`):

    {"parts": [{"id": "body", "mesh": "body.stl", "material": "tpu88a"},
               {"id": "ballast", "mesh": "ballast.stl", "density": 7.9},
               {"id": "board", "mass": 0.002, "at": [0, 16, 4]}],
     "feet": {"mesh": "body.stl", "tol": 0.05},           # or "points": [[x, y], ...]
     "friction": 0.5,                                     # or {"pair": "tpu-wood"}
     "presses": {"points": [[0, 0, 9.6], ...], "force": 3.0, "cone": 20},
     "cables": [{"id": "usb", "at": [0, 29, 6.6], "direction": [0, 1, -0.3],
                 "force": 0.5, "lever": 0.3, "plug": "usb-c"}]}

Mass in kg, lengths in mm, forces in N. z is up, the desk is z = zmin.
"""
from __future__ import annotations

import json
import math
import os
import time

import numpy as np

__all__ = ["FRICTION", "StabilityResult", "support_polygon", "mass_properties", "load_margin",
           "min_mass", "run_stability", "stability_report"]

G = 9.81

# Static friction coefficients, dry and clean. Sources: rubber/elastomer on
# common surfaces (engineering handbooks give 0.5-0.9 for soft elastomers on
# wood, laminate and glass; TPU 85-95A on wood 0.6-0.8); hard plastics
# 0.2-0.4. `design` is the value a check uses by default: the low end, a
# dusty or polished desk.
FRICTION = {
    "tpu-wood": 0.7, "tpu-laminate": 0.6, "tpu-glass": 0.8, "tpu-mat": 0.9,
    "silicone-wood": 0.9, "pla-wood": 0.3, "petg-wood": 0.3, "pa12-wood": 0.3, "metal-wood": 0.3,
    "design": 0.5,
}


def _friction(spec):
    if spec is None:
        return FRICTION["design"], "design (low end of TPU on a desk)"
    if isinstance(spec, (int, float)):
        return float(spec), "given"
    pair = spec.get("pair")
    if pair not in FRICTION:
        raise ValueError(f"friction pair {pair!r}: one of {sorted(FRICTION)} or a number")
    return FRICTION[pair], pair


def _hull(pts):
    """Convex hull of 2D points, counter-clockwise (monotone chain)."""
    P = sorted(set(map(tuple, np.round(np.asarray(pts, float), 6))))
    if len(P) < 3:
        raise ValueError("feet: fewer than three distinct contact points; the product would not stand")

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    lo, hi = [], []
    for p in P:
        while len(lo) >= 2 and cross(lo[-2], lo[-1], p) <= 0:
            lo.pop()
        lo.append(p)
    for p in reversed(P):
        while len(hi) >= 2 and cross(hi[-2], hi[-1], p) <= 0:
            hi.pop()
        hi.append(p)
    return np.array(lo[:-1] + hi[:-1])


def _edge_distance(poly, p):
    """Signed distance from p to the polygon's boundary (> 0 inside), and
    the index of the nearest edge."""
    n = len(poly)
    best, bi = math.inf, -1
    inside = True
    for i in range(n):
        a, b = poly[i], poly[(i + 1) % n]
        e = b - a
        L = float(np.hypot(*e))
        nrm = np.array([e[1], -e[0]]) / L            # outward for a CCW polygon
        s = float((p - a) @ nrm)
        if s > 0:
            inside = False
        t = float(np.clip((p - a) @ e / (L * L), 0, 1))
        dist = float(np.hypot(*(a + t * e - p)))
        if dist < best:
            best, bi = dist, i
    return (best if inside else -best), bi


def _load_mesh(path, base):
    import trimesh
    p = path if os.path.isabs(path) or base is None else os.path.join(base, path)
    m = trimesh.load(p, force="mesh")
    if not m.is_watertight:
        m.fill_holes()
    return m


def support_polygon(feet, base_dir=None, parts_meshes=None):
    """The feet: (polygon (k, 2) CCW, contact triangles' centroids and
    areas (for r_eff), desk z). From `points` [[x, y], ...] (each a small
    pad of equal area) or a `mesh` (the faces within `tol` mm of its lowest
    point, facing down)."""
    if feet.get("points"):
        pts = np.asarray(feet["points"], float)[:, :2]
        z0 = float(feet.get("z", 0.0))
        return _hull(pts), pts, np.ones(len(pts)), z0
    meshes = []
    src = feet.get("mesh")
    for s in ([src] if isinstance(src, str) else (src or [])):
        meshes.append(_load_mesh(s, base_dir))
    if not meshes and parts_meshes:
        meshes = parts_meshes
    if not meshes:
        raise ValueError("feet: give `points` or a `mesh`")
    tol = float(feet.get("tol", 0.05))
    z0 = min(float(m.bounds[0, 2]) for m in meshes)
    C, A, V = [], [], []
    for m in meshes:
        c = m.triangles_center
        sel = (c[:, 2] < z0 + tol) & (m.face_normals[:, 2] < -0.7)
        C.append(c[sel, :2]); A.append(m.area_faces[sel])
        V.append(m.vertices[np.unique(m.faces[sel])][:, :2])
    C, A, V = np.concatenate(C), np.concatenate(A), np.concatenate(V)
    if len(C) == 0:
        raise ValueError("feet: no downward faces at the lowest point")
    return _hull(V), C, A, z0


def mass_properties(parts, base_dir=None):
    """Total mass (kg), centre of mass (mm) and the per-part table. Each
    part: `mesh` + `material` (loadpath material) or `density` (g/cm3), or
    `mass` (kg) + `at` (mm)."""
    from .materials import material_from_spec
    rows, M, S = [], 0.0, np.zeros(3)
    meshes = []
    for p in parts:
        if "mesh" in p:
            m = _load_mesh(p["mesh"], base_dir)
            meshes.append(m)
            if "density" in p:
                rho = float(p["density"])
            elif "material" in p:
                rho = float(material_from_spec(p["material"]).density)
            elif "mass" not in p:
                raise ValueError(f"part {p.get('id')!r}: give a material, a density or a mass")
            vol = abs(float(m.volume)) / 1000.0                 # cm3
            mass = float(p["mass"]) if "mass" in p else vol * rho / 1000.0
            at = np.asarray(p["at"], float) if "at" in p else np.asarray(m.center_mass, float)
            rows.append({"id": p.get("id"), "volume_cm3": round(vol, 4), "mass_g": round(mass * 1000, 3),
                         "centerOfMass": [round(float(x), 3) for x in at]})
        else:
            if "mass" not in p or "at" not in p:
                raise ValueError(f"part {p.get('id')!r}: a part without a mesh needs `mass` (kg) and `at` (mm)")
            mass, at = float(p["mass"]), np.asarray(p["at"], float)
            rows.append({"id": p.get("id"), "mass_g": round(mass * 1000, 3),
                         "centerOfMass": [round(float(x), 3) for x in at]})
        M += mass
        S += mass * at
    if M <= 0:
        raise ValueError("parts: no mass")
    return M, S / M, rows, meshes


def _r_eff(C, A, p):
    """Mean distance of the contact patches from p, area weighted."""
    return float(np.sum(A * np.hypot(C[:, 0] - p[0], C[:, 1] - p[1])) / np.sum(A))


def load_margin(mass, com, poly, C, A, z0, mu, at, force, twist=0.0):
    """Margins for one load: `force` (3,) N at `at` (3,) mm, plus a twist
    (N mm about +z). Returns dict(slide, tip, lift, N, cop)."""
    W = mass * G
    F = np.asarray(force, float)
    H = F[:2]
    V = -F[2]                                   # > 0 pushes down
    N = W + V
    a = np.asarray(at, float)
    h = a[2] - z0
    if N <= 0:
        return {"slide": -1.0, "tip": -1.0, "lift": True, "N": N, "cop": None}
    # moments about the desk plane: horizontal force at height h tips the
    # centre of pressure along H by h |H| / N
    cop = (W * com[:2] + V * a[:2] - H * h) / N
    # the twist about the centre of pressure: the force's own moment
    # (r x H) plus the given twist
    r = a[:2] - cop
    T = r[0] * H[1] - r[1] * H[0] + twist
    reff = max(_r_eff(C, A, cop), 1e-6)
    use = math.hypot(np.hypot(*H) / (mu * N), T / (mu * N * reff))
    slide = 1.0 - use
    d_cop, _ = _edge_distance(poly, cop)
    d_rest, _ = _edge_distance(poly, com[:2])
    tip = d_cop / d_rest if d_rest > 0 else -1.0
    return {"slide": float(slide), "tip": float(tip), "lift": False, "N": float(N),
            "cop": [float(cop[0]), float(cop[1])], "frictionUsed": float(use)}


def _press_forces(force, cone, normal=(0, 0, 1), n_az=16):
    """The push at its worst lean: straight in, then leaning `cone` degrees
    in n_az directions."""
    nrm = np.asarray(normal, float); nrm /= np.linalg.norm(nrm)
    t1 = np.cross(nrm, [1, 0, 0] if abs(nrm[0]) < 0.9 else [0, 1, 0]); t1 /= np.linalg.norm(t1)
    t2 = np.cross(nrm, t1)
    out = [-force * nrm]
    c, s = math.cos(math.radians(cone)), math.sin(math.radians(cone))
    for k in range(n_az):
        ph = 2 * math.pi * k / n_az
        out.append(-force * (c * nrm + s * (math.cos(ph) * t1 + math.sin(ph) * t2)))
    return out


def min_mass(com, poly, C, A, z0, mu, loads, hi=50.0):
    """The least mass (kg) that holds every (at, force, twist) in `loads`
    with the centre of mass at `com`; inf if none up to `hi` kg would."""
    def ok(m):
        return all((lambda r: not r["lift"] and r["slide"] >= 0 and r["tip"] >= 0)(
            load_margin(m, com, poly, C, A, z0, mu, a, f, t)) for a, f, t in loads)
    if ok(1e-6):
        return 0.0
    if not ok(hi):
        return math.inf
    lo_, hi_ = 1e-6, hi
    for _ in range(60):
        mid = math.sqrt(lo_ * hi_)
        lo_, hi_ = (lo_, mid) if ok(mid) else (mid, hi_)
    return hi_


class StabilityResult(dict):
    pass


def _worst(rows, key):
    return min(rows, key=lambda r: (r[key] if r[key] is not None else -9))


def run_stability(brief, base_dir=None):
    """Check a brief (dict, or a path). Returns the report dict."""
    t0 = time.time()
    if isinstance(brief, str):
        base_dir = base_dir or os.path.dirname(os.path.abspath(brief))
        with open(brief) as fh:
            brief = json.load(fh)
    M, com, rows, meshes = mass_properties(brief["parts"], base_dir)
    poly, C, A, z0 = support_polygon(brief.get("feet", {}), base_dir, meshes)
    mu, mu_src = _friction(brief.get("friction"))
    rep = {"mass_g": round(M * 1000, 2), "centerOfMass": [round(float(x), 2) for x in com],
           "comHeight": round(float(com[2] - z0), 2), "parts": rows, "friction": {"mu": mu, "source": mu_src},
           "feet": {"polygon": np.round(poly, 2).tolist(), "area_mm2": round(float(np.sum(A)), 1),
                    "comInset": round(_edge_distance(poly, com[:2])[0], 2)}}
    env = brief.get("envelope_cm3")
    if env:
        rep["apparentDensity"] = round(M * 1000 / float(env), 3)
        rep["apparentDensityNote"] = ("g/cm3 over the closed exterior: under ~1 reads light in the hand "
                                      "(it would float); a solid elastomer block is ~1.1-1.2, a phone ~2")
    cases = []
    pr = brief.get("presses")
    if pr:
        force, cone = float(pr.get("force", 3.0)), float(pr.get("cone", 20.0))
        out = []
        for a in pr["points"]:
            loads = [(a, f, 0.0) for f in _press_forces(force, cone, pr.get("normal", (0, 0, 1)))]
            res = [load_margin(M, com, poly, C, A, z0, mu, *ld) for ld in loads]
            w = min(res, key=lambda r: min(r["slide"], r["tip"]))
            inset = _edge_distance(poly, np.asarray(a, float)[:2])[0]
            out.append({"at": [round(float(x), 2) for x in a], "slide": round(w["slide"], 3), "tip": round(w["tip"], 3),
                        "insideFeet": inset > 0, "inset": round(inset, 2),
                        "minMass_g": _g(min_mass(com, poly, C, A, z0, mu, loads))})
        cases.append({"case": "press", "force": force, "cone": cone, "points": out,
                      "worstSlide": min(o["slide"] for o in out), "worstTip": min(o["tip"] for o in out),
                      "minMass_g": max(o["minMass_g"] for o in out),
                      "note": "a lean of `cone` degrees in the worst direction; a push straight down a "
                              "point inside the feet can never tip or slide it"})
    for cb in brief.get("cables", []) or []:
        a = np.asarray(cb["at"], float)
        d = np.asarray(cb.get("direction", [0, 1, 0]), float); d /= np.linalg.norm(d)
        row = {"case": "cable", "id": cb.get("id"), "at": a.round(2).tolist()}
        if cb.get("force"):
            f = float(cb["force"]) * d
            r = load_margin(M, com, poly, C, A, z0, mu, a, f)
            row["tug"] = {"force": float(cb["force"]), "direction": d.round(3).tolist(),
                          "slide": round(r["slide"], 3), "tip": round(r["tip"], 3), "lift": r["lift"],
                          "minMass_g": _g(min_mass(com, poly, C, A, z0, mu, [(a, f, 0.0)]))}
        if cb.get("lever"):
            # sideways at the plug, square to the cable, both ways
            side = np.cross([0, 0, 1.0], d); side /= max(np.linalg.norm(side), 1e-9)
            loads = [(a, s * float(cb["lever"]) * side, 0.0) for s in (1, -1)]
            res = [load_margin(M, com, poly, C, A, z0, mu, *ld) for ld in loads]
            w = min(res, key=lambda r: r["slide"])
            row["lever"] = {"force": float(cb["lever"]), "slide": round(w["slide"], 3),
                            "minMass_g": _g(min_mass(com, poly, C, A, z0, mu, loads))}
        if cb.get("plug"):
            from .connectors import CONNECTORS
            ins = float(CONNECTORS[cb["plug"]]["insert"]) if cb["plug"] in CONNECTORS else float(cb["plug"])
            f = -ins * d
            r = load_margin(M, com, poly, C, A, z0, mu, a, f)
            m_need = min_mass(com, poly, C, A, z0, mu, [(a, f, 0.0)])
            row["plugIn"] = {"force": ins, "slide": round(r["slide"], 3), "minMass_g": _g(m_need),
                             "note": "pushing the plug home with the product loose on the desk: hold it" if
                                     r["slide"] < 0 else "weight alone holds it"}
        cases.append(row)
    rep["cases"] = cases
    need = [c["minMass_g"] for c in cases if c["case"] == "press"]
    for c in cases:
        for k in ("tug", "lever"):
            if k in c:
                need.append(c[k]["minMass_g"])
    rep["minMass_g"] = max(need) if need else 0.0
    rep["holds"] = bool(rep["mass_g"] >= rep["minMass_g"])
    rep["seconds"] = round(time.time() - t0, 2)
    rep["summary"] = stability_report(rep)
    return StabilityResult(rep)


def _g(m):
    return round(m * 1000, 2) if math.isfinite(m) else None


def stability_report(rep):
    """One paragraph for a designer."""
    s = [f"{rep['mass_g']:.1f} g, centre of mass {rep['comHeight']:.1f} mm up and "
         f"{rep['feet']['comInset']:.1f} mm inside its feet; friction {rep['friction']['mu']:.2f}."]
    for c in rep["cases"]:
        if c["case"] == "press":
            s.append(f"Presses of {c['force']:g} N leaning {c['cone']:g}°: worst slide margin {c['worstSlide']:.2f}, "
                     f"tip {c['worstTip']:.2f}; mass needed {c['minMass_g']} g.")
        else:
            if "tug" in c:
                t = c["tug"]
                s.append(f"Cable {c.get('id') or ''} tug {t['force']:g} N: slide {t['slide']:.2f}, tip {t['tip']:.2f}; "
                         f"needs {t['minMass_g']} g.")
            if "lever" in c:
                s.append(f"Sideways {c['lever']['force']:g} N at the plug: slide {c['lever']['slide']:.2f}; "
                         f"needs {c['lever']['minMass_g']} g.")
            if "plugIn" in c:
                s.append(f"Plugging in ({c['plugIn']['force']:g} N): {c['plugIn']['note']}.")
    s.append("Holds." if rep["holds"] else f"Does not hold: needs {rep['minMass_g']} g.")
    return " ".join(s)
