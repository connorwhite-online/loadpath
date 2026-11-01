"""
Lattice briefs: a part that is a lattice for a reason the structural scorer
alone cannot see. The brief's pass block says which rules apply, and every
rule is judged on the delivered part.stl, not on what the agent says about it:

  (gold study)  part.stl re-verified under the gold study (briefs.score: one
                body, envelope, keep-outs, interfaces, safety factor, mass)
  cellular      it really is a lattice: in a probe box inside the part, the
                filled share is at most maxFill, lines through it cross at
                least minWallsPer10mm walls per 10 mm (a lattice crosses a
                wall every few mm, a solid web or a ribbed part hardly
                ever), optionally a share of straight lines along X and Y
                that see through (minSeeThrough: an open strut lattice), and
                optionally the outside kept (minHull: convex hull volume over
                the design space box)
  drains        no sealed cavity over maxEnclosed mm^3 (powder or resin
                trapped inside)
  print         sliced along the build direction (printcheck.check_print with
                the brief's bridge span): at most maxSupportArea mm^2 that
                needs support (islands included)
  impact        the delivered pad, voxelized and homogenized (each voxel's
                relative density is the part's local filled share, smoothed
                over a few mm so the cells average out, solid face sheets
                kept solid), stopped with the GOLD mass, speed and force
                limit by loadpath's own impact model (impact.impact_report's
                layer model, with the material's crush law and TPU's rate
                stiffening, on a finely resampled curve; see _dense):
                peak force under the limit, crush under maxCrush, inside the
                pad's box, face sheets top and bottom. A field.json with an
                impact block is re-run with the gold numbers too and
                reported (not judged: its envelope and density are the
                agent's claims about the part)
  vibration     natural frequencies of the delivered part (modal.py on the
                part voxelized at `pitch`, each voxel at its filled share of
                the modulus and density) clamped in the gold `fixed` box with
                the gold point masses on the nodes in their boxes: first mode
                at least FREQ_SLACK x minFrequency

Boxes in the pass block are {"min": [x, y, z], "max": [x, y, z]} in the
brief's coordinates.
"""
from __future__ import annotations

import json
import os

import numpy as np

DELIVER = """  part.stl     the part to print (your final, verified outcome)
  field.json   the field graph that produced it, if a field graph did (with its impact or vibration block)
  study.json   the study, if a study produced or checked it
  notes.md     two or three lines: the outcome's mass and how it meets the brief's numbers, and your assumptions
"""

FREQ_SLACK = 0.95        # the agent's analysis grid and ours differ
FORCE_SLACK = 1.05       # homogenized crush from a voxelized part: a few percent either way
CRUSH_SLACK = 1.1
BOX_TOL = 0.3            # mm a delivered pad may stick out of its box (mesh noise)
MAX_ENCLOSED = 50.0      # mm^3 sealed inside before it counts as trapped powder
IMPACT_PITCH = 0.25      # mm voxels for the crush model (TPU walls are ~0.5 mm)
IMPACT_SMOOTH = 2.5      # mm Gaussian over which the local density is averaged (cells up to ~14 mm)

KNOWN = {"safetyFactor", "maxDeflection", "maxMass", "maxSolidFraction", "cellular", "drains", "print",
         "impact", "vibration"}


def _load(folder):
    import trimesh

    stl = os.path.join(folder, "part.stl")
    if not os.path.exists(stl):
        return stl, None
    return stl, trimesh.load(stl, force="mesh")


def _material(brief, spec):
    from ...materials import material_from_spec

    name = spec.get("material") or (brief.get("gold") or {}).get("material")
    return material_from_spec(name)


def _box(b):
    return np.asarray(b["min"], float), np.asarray(b["max"], float)


# --- cellular, drains, print --------------------------------------------------

def cellular(mesh, spec: dict, design_box=None) -> dict:
    """Filled share, walls crossed per 10 mm along X, Y and Z, and the share
    of X and Y lines that see through, in the probe box (0.4 mm voxels)."""
    from ...verify import voxelize

    lo, hi = _box(spec["probe"])
    pitch = float(spec.get("pitch", 0.4))
    solid, _ = voxelize(mesh, pitch, box=(lo, hi))
    walls, see = [], []
    for ax in range(3):
        a = np.moveaxis(solid, ax, -1)
        starts = (a[..., 1:] & ~a[..., :-1]).sum(axis=-1) + a[..., 0]
        walls.append(float(starts.mean() / (a.shape[-1] * pitch) * 10.0))
        if ax < 2:
            see.append(float((~a.any(axis=-1)).mean()))
    out = {"fill": round(float(solid.mean()), 3), "wallsPer10mm": [round(w, 2) for w in walls],
           "seeThrough": round(float(np.mean(see)), 3)}
    if spec.get("minHull") is not None and design_box is not None:
        dlo, dhi = design_box
        out["hull"] = round(float(mesh.convex_hull.volume / np.prod(dhi - dlo)), 3)
    return out


def _cellular_failures(c, spec):
    f = []
    if c["fill"] > spec.get("maxFill", 0.6):
        f.append(f"{c['fill']:.0%} solid inside: not a lattice (at most {spec.get('maxFill', 0.6):.0%})")
    need = spec.get("minWallsPer10mm", 0.6)
    if min(c["wallsPer10mm"]) < need:
        f.append(f"{min(c['wallsPer10mm']):.2f} walls per 10 mm through the inside: not a lattice (at least {need})")
    if spec.get("minSeeThrough") is not None and c["seeThrough"] < spec["minSeeThrough"]:
        f.append(f"{c['seeThrough']:.0%} of lines see through the inside: not an open lattice "
                 f"(at least {spec['minSeeThrough']:.0%})")
    if spec.get("minHull") is not None and c.get("hull") is not None and c["hull"] < spec["minHull"]:
        f.append(f"the outside fills {c['hull']:.0%} of the block: its shape was not kept")
    return f


def enclosed_volume(mesh) -> float:
    """mm^3 sealed inside the part (manifold's negative-volume pieces)."""
    from ...shapes import manifold_from_trimesh

    return float(-sum(v for v in (b.volume() for b in manifold_from_trimesh(mesh).decompose()) if v < 0))


def _print(mesh, spec, material):
    from ...printcheck import check_print

    d = spec.get("direction", "+z")
    direction = {"+x": (0, 1), "-x": (0, -1), "+y": (1, 1), "-y": (1, -1), "+z": (2, 1), "-z": (2, -1)}[d]
    pr = check_print(mesh, direction, "fdm", pitch=spec.get("pitch"), bridge_span=float(spec["bridgeSpan"]),
                     material=material)
    row = {k: pr[k] for k in ("pitch", "supportArea", "supportVolume", "islands", "islandArea", "bridges",
                              "maxBridgeSpan")}
    f = []
    # islands are in the support area too; a few mm^2 of strut tips cut by a
    # hole or the envelope is voxel noise, not a print that fails
    if pr["supportArea"] > spec.get("maxSupportArea", 25.0):
        f.append(f"{pr['supportArea']:.0f} mm^2 needs support printed {d} ({pr['islands']} island(s); "
                 f"bridges up to {spec['bridgeSpan']:g} mm)")
    return {k: (round(v, 2) if isinstance(v, float) else v) for k, v in row.items()}, f


# --- impact ------------------------------------------------------------------

def measure_core(mesh, pitch: float = IMPACT_PITCH, axis: int = 2, smooth: float = IMPACT_SMOOTH):
    """The delivered part as a homogenized core: (inside, rho, lo, fill per
    layer, solid layer flags), crush axis last. `inside` is each layer's
    footprint (convex hull of the material within 4 mm of it), rho the
    filled share smoothed over `smooth` mm (normalized over the footprint,
    face sheets left out of the average and kept solid)."""
    from scipy import ndimage
    from skimage.morphology import convex_hull_image

    from ...verify import voxelize

    solid, origin = voxelize(mesh, pitch)
    occ = np.moveaxis(solid, axis, 2).astype(float)
    nz = occ.shape[2]
    win = max(1, int(round(4.0 / pitch)))
    inside = np.zeros(occ.shape, bool)
    for k in range(nz):
        pr = occ[:, :, max(0, k - win):k + win + 1].any(axis=2)
        if pr.sum() >= 3:
            inside[:, :, k] = convex_hull_image(pr)
    n = inside.sum(axis=(0, 1))
    fill = np.where(n > 0, (occ * inside).sum(axis=(0, 1)) / np.maximum(n, 1), 0.0)
    solid_layer = fill > 0.9
    w = inside & ~solid_layer[None, None, :]
    s = smooth / pitch
    num = ndimage.gaussian_filter(occ * w, s, mode="constant")
    den = ndimage.gaussian_filter(w.astype(float), s, mode="constant")
    rho = np.where(w, num / np.maximum(den, 1e-9), 0.0)
    rho[:, :, solid_layer] = 1.0
    rho = np.where(inside, np.clip(rho, 0.0, 1.0), 0.0)
    origin = np.asarray(origin, float)
    lo = np.r_[np.delete(origin, axis), origin[axis]]
    return inside, rho, lo, fill, solid_layer


def _impact(brief, mesh, spec, folder):
    mat = _material(brief, spec)
    axis = "xyz".index(spec.get("axis", "z"))
    f = []
    row = {}
    lo, hi = _box(spec["box"])
    blo, bhi = np.asarray(mesh.bounds, float)
    out = float(max(np.max(lo - blo), np.max(bhi - hi)))
    if out > BOX_TOL:
        f.append(f"the pad sticks {out:.1f} mm out of its {np.round(hi - lo, 1).tolist()} mm box")
    inside, rho, _, fill, solid_layer = measure_core(mesh, IMPACT_PITCH, axis)
    core = ~solid_layer & (fill > 0)
    row["coreDensity"] = [round(float(fill[core].min()), 3), round(float(fill[core].max()), 3)] if core.any() else None
    if not core.any() or fill[core].max() > 0.7:
        f.append("no lattice core: it is solid through its height")
    if spec.get("faces"):
        ends = [fill[np.flatnonzero(fill > 0)[0]], fill[np.flatnonzero(fill > 0)[-1]]]
        row["faceFill"] = [round(float(e), 2) for e in ends]
        if min(ends) < 0.85:
            f.append("no solid face sheet on the " + ("bottom" if ends[0] < 0.85 else "top"))
    gold = {"envelope": "envelope", "density": "density", "lattice": spec.get("lattice", "gyroid"), "axis": "z",
            "mass": float(spec["mass"]), "velocity": float(spec["velocity"]),
            "forceLimit": float(spec["forceLimit"]), "pitch": IMPACT_PITCH}
    r, rate = stop_impact(inside, rho, IMPACT_PITCH, gold["lattice"], mat, gold["mass"], gold["velocity"])
    row.update(energy=r["energy"], peakForce=r["peakForce"], crush=r["crush"], peakG=r.get("peakG"),
               bottomedOut=r["bottomedOut"], rate=rate)
    if r["bottomedOut"] or r["peakForce"] is None:
        f.append(f"bottoms out: absorbs {r['capacity'] / 1000:.2f} J of {r['energy'] / 1000:.2f} J")
    else:
        if r["peakForce"] > FORCE_SLACK * spec["forceLimit"]:
            f.append(f"peak force {r['peakForce']:.0f} N > {spec['forceLimit']:g} N")
        if spec.get("maxCrush") and r["crush"] > CRUSH_SLACK * spec["maxCrush"]:
            f.append(f"crushes {r['crush']:.1f} mm > {spec['maxCrush']:g} mm")
    row["graph"] = _graph_impact(folder, gold, mat)
    return row, f


def _dense(cv, step=0.01):
    """The crush curve resampled every `step` mm. impact_response inverts
    the energy by linear interpolation between the curve's corners, which on
    an elastic ramp (energy quadratic in crush) reads far too little force
    when the stop ends on it: a solid TPU block came out at 80 N instead of
    about 2 kN. Fine sampling makes the inversion exact enough."""
    from ...impact import CrushCurve

    d = np.unique(np.r_[cv.d, np.arange(0.0, float(cv.d[-1]), step)])
    return CrushCurve(d, np.interp(d, cv.d, cv.f), cv.stroke, cv.layer_force, cv.layer_area, cv.layer_rho,
                      cv.layer_stroke)


def stop_impact(inside, rho, h, kind, material, mass, velocity):
    """impact.impact_report's layer model on a voxel core (crush along the
    grid's last axis): the material's crush law, each layer's force the sum
    of its cells' plateau, an elastomer stiffened for the crush time, and
    the stop found on a finely resampled curve. Returns (response, rate
    factor or None)."""
    from ...impact import (
        crush_curve,
        crush_law,
        impact_response,
        layer_profile,
        plateau_stress,
    )

    law, sy = crush_law(kind, material)
    E = float(material.E)
    area, mean = layer_profile(inside, rho, h, 2)
    lf = (plateau_stress(law, rho, sy) * inside).sum(axis=(0, 1)) * h ** 2
    energy = 1000.0 * 0.5 * mass * velocity ** 2
    resp = impact_response(_dense(crush_curve(law, area, mean, h, sy, E, layer_force=lf)), energy, mass)
    fac = None
    if law.endswith(":elastic"):
        from ...viscoelastic import creep_params, event_factor

        if creep_params(material) is not None:
            for _ in range(4):
                t = 2.0 * float(resp["crush"]) * 1e-3 / velocity
                fac = float(event_factor(material, t))
                cv = crush_curve(law, area, mean, h, sy * fac, E * fac, layer_force=lf * fac)
                resp = impact_response(_dense(cv), energy, mass)
    return resp, (round(fac, 3) if fac else None)


def _graph_impact(folder, gold, mat):
    """The agent's own field graph's impact block, re-run with the gold
    mass, speed, force limit and axis (reported, not judged)."""
    path = os.path.join(folder, "field.json")
    if not os.path.exists(path):
        return None
    try:
        from ...fields import FieldGraph
        from ...impact import impact_report

        with open(path) as fh:
            doc = json.load(fh)
        spec = doc.get("impact")
        if not spec:
            return {"note": "field.json has no impact block"}
        spec = {k: v for k, v in spec.items() if k not in ("energy", "grade")}
        spec.update({k: gold[k] for k in ("mass", "velocity", "forceLimit", "axis")})
        r = impact_report(FieldGraph(doc, base_dir=folder), spec, mat)
        return {"peakForce": r["response"]["peakForce"], "crush": r["response"]["crush"], "ok": r.get("ok")}
    except Exception as exc:  # noqa: BLE001 - the agent's graph is information here
        return {"error": str(exc).strip().splitlines()[-1][:200] if str(exc).strip() else type(exc).__name__}


# --- vibration ---------------------------------------------------------------

def _vibration(brief, mesh, spec):
    from ...fem import HexMesh
    from ...modal import modal_analysis
    from ...verify import voxelize

    mat = _material(brief, spec)
    h = float(spec.get("pitch", 1.0))
    vol = float(abs(mesh.volume))
    h = max(h, (vol / 30_000) ** (1.0 / 3.0))          # at most ~30k cells: a direct eigensolve
    sub = int(spec.get("sub", 3))
    lo = np.floor(np.asarray(mesh.bounds[0]) / h) * h - h
    hi = np.ceil(np.asarray(mesh.bounds[1]) / h) * h + h
    solid, origin = voxelize(mesh, h / sub, box=(lo, hi))
    n = np.array(solid.shape) // sub
    share = solid[:n[0] * sub, :n[1] * sub, :n[2] * sub].reshape(n[0], sub, n[1], sub, n[2], sub).mean(axis=(1, 3, 5))
    keep = share >= 0.1
    hm = HexMesh(np.argwhere(keep), h, origin)
    s = share[keep]
    xyz = hm.node_xyz

    def nodes_in(b):
        blo, bhi = _box(b)
        return np.flatnonzero(np.all((xyz >= blo - 1e-6) & (xyz <= bhi + 1e-6), axis=1))

    f = []
    fn = nodes_in(spec["fixed"])
    if len(fn) == 0:
        return {}, ["nothing where the part is held"]
    fixed = np.zeros(hm.n_dof, bool)
    for a in range(3):
        fixed[3 * fn + a] = True
    pm = []
    for p in spec.get("pointMasses", []):
        nn = nodes_in(p)
        if len(nn) == 0:
            return {}, [f"nothing where the {p['mass']:g} kg mass sits"]
        pm.append((nn, float(p["mass"])))
    # parts not connected to the clamp float: drop them (their modes are 0 Hz)
    from scipy import ndimage

    lbl, _ = ndimage.label(keep)
    held = np.unique(lbl[tuple(np.argwhere(keep)[np.isin(hm.enode, fn).any(axis=1)].T)])
    if not np.isin(lbl[keep], held).all():
        f.append("material not connected to the held face")
    res = modal_analysis(hm, mat.E * s, mat.density * s, mat.nu, fixed, int(spec.get("modes", 3)), pm)
    freqs = [round(float(x), 1) for x in res.frequencies]
    row = {"frequencies": freqs, "cells": int(hm.n_elem), "pitch": round(h, 3)}
    if freqs[0] < FREQ_SLACK * spec["minFrequency"]:
        f.append(f"first mode {freqs[0]:.0f} Hz < {spec['minFrequency']:g} Hz")
    return row, f


# --- the kind's interface ----------------------------------------------------

def score(brief: dict, folder: str, max_cells: int | None = None) -> dict:
    from .. import briefs

    want = brief["pass"]
    stl, mesh = _load(folder)
    if mesh is None:
        return {"pass": False, "failures": ["no part.stl"]}
    if "gold" in brief:
        row = briefs.score(brief, stl, max_cells or briefs.VERIFY_CELLS)
        row.pop("pass", None)
    else:
        row = {"failures": []}
        mat = _material(brief, want.get("impact") or want.get("vibration") or {})
        row["mass"] = round(float(abs(mesh.volume)) * mat.density / 1000.0, 2)
        row["bodies"] = len(mesh.split(only_watertight=False))
        if row["bodies"] != 1:
            row["failures"].append(f"{row['bodies']} bodies")
        if want.get("maxMass") and row["mass"] > briefs.MASS_SLACK * want["maxMass"]:
            row["failures"].append(f"{row['mass']:.1f} g > {want['maxMass']} g")
    f = row["failures"]
    mat = _material(brief, {}) if "gold" in brief else None
    if want.get("cellular"):
        row["cellular"] = cellular(mesh, want["cellular"], _design_box(brief) if "gold" in brief else None)
        f += _cellular_failures(row["cellular"], want["cellular"])
    if want.get("drains"):
        row["enclosedVolume"] = round(enclosed_volume(mesh), 1)
        lim = want["drains"].get("maxEnclosed", MAX_ENCLOSED)
        if row["enclosedVolume"] > lim:
            f.append(f"{row['enclosedVolume']:.0f} mm^3 sealed inside: the powder cannot get out")
    if want.get("print"):
        row["print"], ff = _print(mesh, want["print"], mat)
        f += ff
    if want.get("impact"):
        row["impact"], ff = _impact(brief, mesh, want["impact"], folder)
        f += ff
    if want.get("vibration"):
        row["vibration"], ff = _vibration(brief, mesh, want["vibration"])
        f += ff
    row["pass"] = not f
    return row


def gold(brief: dict) -> dict:
    """The gold study builds (if there is one), the pass block is well formed,
    and an impact brief is feasible: loadpath's own graded core for the
    pad's box (less 1 mm face sheets) meets the force and crush limits."""
    from .. import briefs

    want = brief.get("pass") or {}
    unknown = set(want) - KNOWN
    if unknown:
        return {"ok": False, "error": f"unknown pass keys {sorted(unknown)}"}
    res = {"ok": True}
    if "gold" in brief:
        from ...tasks import check_study

        res = check_study(brief["gold"], base_dir=briefs.ASSETS)
        if not res["ok"]:
            return res
    elif not (want.get("impact") or want.get("vibration")):
        return {"ok": False, "error": "no gold study and no impact or vibration block to judge by"}
    try:
        if want.get("cellular"):
            _box(want["cellular"]["probe"])
        if want.get("print") and "bridgeSpan" not in want["print"]:
            return {"ok": False, "error": "print needs bridgeSpan"}
        if want.get("vibration"):
            v = want["vibration"]
            _box(v["fixed"])
            for p in v.get("pointMasses", []):
                _box(p)
                float(p["mass"])
            float(v["minFrequency"])
        if want.get("impact"):
            res = {**res, **_impact_feasible(brief, want["impact"])}
    except (KeyError, TypeError, ValueError) as exc:
        return {"ok": False, "error": f"pass block: {exc}"}
    return res


def _impact_feasible(brief, spec):
    from ...fields import FieldGraph
    from ...impact import impact_report

    lo, hi = _box(spec["box"])
    core_lo, core_hi = lo.copy(), hi.copy()
    core_lo[2] += 1.0
    core_hi[2] -= 1.0
    doc = {"nodes": {"envelope": {"op": "shape", "shape": {"type": "box", "min": core_lo.tolist(),
                                                              "max": core_hi.tolist()}}}}
    g = FieldGraph(doc)
    s = {"envelope": "envelope", "density": 0.3, "lattice": spec.get("lattice", "gyroid"), "axis": "z",
         "mass": spec["mass"], "velocity": spec["velocity"], "forceLimit": spec["forceLimit"], "grade": True,
         "cell": spec.get("cell", 8.0), "minWall": spec.get("minWall", 0.5), "pitch": 0.5}
    r = impact_report(g, s, _material(brief, spec))["graded"]
    resp = r["response"]
    ok = bool(r["ok"]) and resp["peakForce"] is not None and resp["crush"] <= spec.get("maxCrush", np.inf)
    out = {"ok": ok, "gradedPeakForce": resp["peakForce"], "gradedCrush": resp["crush"],
           "gradedCoreMass": r["coreMass"]}
    if not ok:
        out["error"] = f"no graded core meets it: {resp}"
    return out


# --- reference deliveries (tests, and a sanity check of a brief's numbers) ---

def _mbox(lo, hi):
    import manifold3d as m3d

    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    return m3d.Manifold.cube(tuple(hi - lo)).translate(tuple(lo))


def _mcyl(p0, p1, r, sections=24):
    import trimesh

    from ...shapes import manifold_from_trimesh

    return manifold_from_trimesh(trimesh.creation.cylinder(radius=r, segment=[np.asarray(p0, float),
                                                                              np.asarray(p1, float)],
                                                           sections=sections))


def _union(parts):
    import manifold3d as m3d

    return m3d.Manifold.batch_boolean(list(parts), m3d.OpType.Add)


def _prism_x(pts, x0, x1):
    """A prism along X over the (y, z) polygon `pts` (counter-clockwise)."""
    import trimesh

    from ...shapes import manifold_from_trimesh

    pts = np.asarray(pts, float)
    n = len(pts)
    V = np.vstack([np.c_[np.full(n, x0), pts], np.c_[np.full(n, x1), pts]])
    F = [[0, i + 1, i] for i in range(1, n - 1)] + [[n, n + i, n + i + 1] for i in range(1, n - 1)]
    for i in range(n):
        j = (i + 1) % n
        F += [[i, j, n + j], [i, n + j, n + i]]
    t = trimesh.Trimesh(V, np.array(F), process=True)
    if t.volume < 0:
        t.invert()
    return manifold_from_trimesh(t)


def _design_box(brief):
    ds = brief["gold"]["designSpace"]
    return np.min([s["min"] for s in ds], axis=0), np.max([s["max"] for s in ds], axis=0)


def bcc_struts(lo, hi, cell, cz, d, phase, sections=8):
    """A BCC strut lattice (cell x cell x cz, strut diameter d) clipped to the
    box, its body-centre nodes on `phase` + (cell, cell, cz) / 2."""
    import itertools

    lo, hi, o = (np.asarray(v, float) for v in (lo, hi, phase))
    c = np.array([cell, cell, cz])
    first = np.floor((lo - o) / c).astype(int) - 1
    last = np.ceil((hi - o) / c).astype(int) + 1
    struts = []
    for idx in itertools.product(*(range(first[i], last[i]) for i in range(3))):
        mid = o + (np.array(idx) + 0.5) * c
        for s in itertools.product((-1, 1), repeat=3):
            struts.append(_mcyl(mid, mid + 0.5 * np.array(s) * c, d / 2, sections))
    return _union(struts) ^ _mbox(lo, hi)


def reference(brief: dict, folder: str, quick: bool = True) -> str:
    """Write a reference delivery (part.stl) for one of this kind's briefs
    into `folder`: a defensible answer, not the lightest. `quick` uses
    coarser lattice and mesh settings (tests)."""
    from ...shapes import trimesh_from_manifold

    os.makedirs(folder, exist_ok=True)
    bid = brief["id"]
    if bid == "shoulder-riser-mjf":
        from ...lattice import LatticeOptions, envelope_mesh, lattice_infill
        from .. import briefs

        study = briefs.gold_study(brief)
        o = LatticeOptions.from_spec({"type": "gyroid", "cell": 12, "skin": 0, "minWall": 0.8, "maxWall": 3,
                                      "collar": 1.0, "safetyFactor": 3.5}, study.process, 3.5)
        o.analysis_cells, o.iterations = (12_000, 4) if quick else (30_000, 6)
        if quick:
            o.step = 0.45
        mesh, _ = lattice_infill(study, envelope_mesh(study), None, o)
    elif bid == "elbow-stop-pad-tpu":
        from ...fields import FieldGraph

        sh = lambda lo, hi: {"op": "shape", "shape": {"type": "box", "min": lo, "max": hi}}
        doc = {"nodes": {"core": {"op": "lattice", "type": "gyroid", "cell": 8, "density": 0.19},
                         "env": sh([-15, -15, 1], [15, 15, 14]),
                         "clip": {"op": "intersect", "inputs": ["core", "env"]},
                         "part": {"op": "union", "inputs": ["clip", sh([-15, -15, 0], [15, 15, 1.2]),
                                                            sh([-15, -15, 13.8], [15, 15, 15])]}},
               "output": "part"}
        mesh = FieldGraph(doc).mesh(pitch=0.25)
    elif bid == "tray-arm-lattice-fdm":
        lat = bcc_struts([4, -15, 0], [74, 15, 40], 8.0, 12.0, 2.0, [4, -4, 0], 8 if quick else 12)
        keel = _prism_x([(0, 30), (10, 40), (-10, 40)], 4, 74)      # a 45-degree V top chord
        body = _union([lat, keel, _mbox([0, -15, 0], [4, 15, 40]), _mbox([74, -15, 0], [80, 15, 40]),
                       _mbox([0, -15, 0], [80, 15, 2.5])])
        holes = ([_mcyl([-1, 0, z], [5, 0, z], 2.2) for z in (10, 30)]
                 + [_mcyl([4, 0, z], [12, 0, z], 4.0) for z in (10, 30)]
                 + [_mcyl([73.5, y, 30], [81, y, 30], 2.0) for y in (-8, 8)])
        mesh = trimesh_from_manifold(body - _union(holes))
    elif bid == "camera-boom-mjf":
        w = 1.2                                                     # an open-ended square tube
        tube = _mbox([3, -11, -11], [95, 11, 11]) - _mbox([3 + w, -11 + w, -11 + w], [100, 11 - w, 11 - w])
        body = _union([_mbox([0, -18, -18], [3, 18, 18]), tube, _mbox([70, -11, 7], [95, 11, 11])])
        holes = ([_mcyl([-1, y, z], [5, y, z], 1.7) for y in (-14, 14) for z in (-14, 14)]
                 + [_mcyl([x, 0, 5], [x, 0, 12], 1.1) for x in (72, 93)])
        mesh = trimesh_from_manifold(body - _union(holes))
    else:
        raise ValueError(f"no reference delivery for {bid!r}")
    path = os.path.join(folder, "part.stl")
    mesh.export(path)
    return path
