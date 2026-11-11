"""
Build around a scan: a study whose part fits the scanned object, its
clearances and fits from what the scan measured.

`draft_around(scan, feats, out_dir, support=..., loads=...)` writes:

- the **object kept out**: the reconstructed surface (scan.reconstruct)
  moved out, vertex by vertex, by how far the real surface may sit past it
  (the scan's one-sided uncertainty U there, with the patches the scan
  never saw moved out by their distance to the nearest point) combined with
  the process's scatter (fits.HOLES: half the band as a sigma, root sum
  square with the scan's) and the process's bias (FDM and powder walls grow
  inward by half the hole bias). No fixed margin: an Artec scan of a light
  for SLS gives about 0.3 mm, an iPhone scan about 10 mm, and says why;
- each **bar** the part mounts on (a tube, a handlebar: a convex fitted
  cylinder) as a cylindrical keep-out at its fitted diameter with
  `fit` (default slip) and `mateUncertainty`, the diameter's measured band:
  fits.py then draws the hole so the fit holds for every bar in that band,
  or says the scan cannot hold it (measure the bar and give `known`).
  A bar that runs to the edge of the scan runs on through the design space;
- each **face** the part bears on (a fitted plane: the light's top, a wall)
  as a pad, `pad` mm thick, over the face's footprint, set off the face by
  its own one-sided band; the face's solid side kept out;
- each **bore** in the object (an insert, a socket) kept open along its
  axis out through the design space, so a screw or a shaft still goes in;
- supports and loads on those, by feature id.

The design space is a box round the features the part touches, grown by
`grow` mm (a choice: say what the part may fill).
"""
from __future__ import annotations

import json
import math
import os

import numpy as np

from .scan import Scan, reconstruct, z_value

__all__ = ["draft_around", "object_clearance", "known_dimensions", "scan_to_study", "parse_reference"]


def _process_holes(process):
    from .fits import HOLES
    if process is None:
        return None
    key = str(process).lower()
    for k in HOLES:
        if key.startswith(k):
            return HOLES[k]
    return None


def object_clearance(scan: Scan, info: dict, process=None, size: float | None = None) -> tuple[np.ndarray, dict]:
    """Per reconstructed vertex: how far to keep the part off the measured
    surface (mm), and how it splits. The scan's random, device and scale
    sigmas and the process's (half its +- band) add in quadrature at the
    scan's one-sided confidence; the sampling sagitta, the unseen gap and
    the process's inward wall bias add on top."""
    iv = info["iv"]
    r, s, c = scan.sigma_parts(idx=iv)
    holes = _process_holes(process)
    sp = 0.0
    bias = 0.0
    if holes is not None:
        L = size if size is not None else float(np.ptp(scan.points, 0).max())
        sp = 0.5 * (holes["tol"] + holes["tolPerMm"] * L)
        # a hole closes by -bias on its diameter: a wall comes in by half that
        bias = max(0.0, -0.5 * 0.5 * (holes["bias"] + holes["biasAcross"]))
    z = z_value(scan.confidence)
    stat = z * np.sqrt(r * r + s * s + c * c + sp * sp)
    sag = scan.spacing[iv] ** 2 / (8.0 * scan.rho[iv])
    gap = np.where(info["unseen"], info["dv"], 0.0)
    U = stat + sag + gap + bias
    parts = {"confidence": scan.confidence, "random": round(float(np.median(r)), 4),
             "device": round(float(np.median(s)), 4), "scale": round(float(np.median(c)), 4),
             "process": round(sp, 4), "processBias": round(bias, 4),
             "sagitta": round(float(np.median(sag)), 4),
             "unseenGapMax": round(float(gap.max()), 3),
             "clearance": {"median": round(float(np.median(U)), 4), "p95": round(float(np.percentile(U, 95)), 4),
                           "max": round(float(U.max()), 4)}}
    return U, parts


def known_dimensions(feats: dict, known) -> dict:
    """Replace fitted sizes with measured ones: known = {"cyl-0": {"diameter":
    31.8, "tolerance": 0.05}}. The scan still says where the feature is; the
    caliper says how big. tolerance is +- (2 sigma)."""
    if not known:
        return feats
    z2 = z_value(feats.get("confidence", 0.95), two_sided=True)
    for f in feats.get("cylinders", []):
        k = known.get(f["id"])
        if not k:
            continue
        d = float(k["diameter"])
        tol = float(k.get("tolerance", 0.05))
        f["fitted"] = {"diameter": f["diameter"], "diameterBand": f["diameterBand"]}
        f["r"], f["diameter"] = d / 2, d
        f["diameterBand"] = round(z2 * tol / 2, 4)          # tolerance +- is 2 sigma of the diameter
        f["rBand"] = round(f["diameterBand"] / 2, 4)
        f["source"] = "known (measured)"
        disagree = abs(f["fitted"]["diameter"] - d)
        if disagree > f["fitted"]["diameterBand"] + f["diameterBand"]:
            f["warning"] = (f"the scan read {f['fitted']['diameter']:.3f} mm, {disagree:.2f} mm off the measured "
                            f"{d:g}, more than its band ({f['fitted']['diameterBand']:.3f}): the device's sigma is "
                            "too small, or this is not the feature measured")
    return feats


def _offset_mesh(mesh, U):
    """The surface moved out along its vertex normals by U (per vertex), as
    a manifold solid (union with the original, so nothing moves in)."""
    import trimesh
    from .shapes import manifold_from_trimesh, trimesh_from_manifold
    V = np.asarray(mesh.vertices) + np.asarray(mesh.vertex_normals) * np.asarray(U)[:, None]
    off = trimesh.Trimesh(V, np.asarray(mesh.faces), process=False)
    try:
        m = manifold_from_trimesh(off)
        m = m + manifold_from_trimesh(mesh)
        return trimesh_from_manifold(m)
    except Exception:
        # a fold where U changes fast: the max over a voxelized union instead
        from .exterior import wrap_mesh
        out, *_ = wrap_mesh(trimesh.util.concatenate([off, mesh]))
        return out


def _closed(mesh):
    """A mesh that is still closed once written to STL and read back: a
    boolean can leave two bodies touching along an edge (four faces on it),
    which a file with merged vertices reads as open; those are re-wrapped."""
    import trimesh
    m2 = trimesh.Trimesh(np.asarray(mesh.vertices), np.asarray(mesh.faces), process=True)
    if m2.is_watertight or len(m2.faces) == 0:
        return m2
    from .exterior import wrap_mesh
    p = max(0.05, float(np.percentile(m2.edges_unique_length, 50)))
    out, *_ = wrap_mesh(m2, pitch=p)
    return out


def _cyl_keepout(f, extend_lo, extend_hi):
    a = np.asarray(f["axis"], float)
    p0, p1 = np.asarray(f["p0"], float), np.asarray(f["p1"], float)
    return (p0 - extend_lo * a), (p1 + extend_hi * a)


def _box_of(points, grow):
    lo, hi = np.min(points, 0) - grow, np.max(points, 0) + grow
    return lo, hi


def draft_around(scan: Scan, feats: dict, out_dir: str, *, support=None, loads=None, name: str | None = None,
                 material: str = "pa12", process: str | None = "sls", grow: float = 15.0, pad: float = 2.0,
                 bar_fit: str = "slip", known=None, safety_factor: float = 2.0, mesh_info=None,
                 space=None, keep_object: bool = True) -> dict:
    """Write a study that builds a part around the scanned object (see the
    module docstring) into out_dir and return {"study", "clearance", "todo"}.

    support: feature ids the part is held by (a bar it clamps, a wall it
      bolts to); loads: [(feature id, [fx, fy, fz] N)] or {id: force}, each
      its own load case (the object's weight times its shock, on the face it
      sits on). known: measured sizes that replace fitted ones
      (known_dimensions). space: [[lo], [hi]] box instead of the grown one."""
    import trimesh
    from .cadimport import ring_wall
    from .shapes import manifold_from_trimesh, trimesh_from_manifold, Cylinder

    if not scan.scaled:
        raise ValueError("the scan has no scale: give a reference (a known diameter or distance) or units and "
                         "a device before drafting anything round it")
    os.makedirs(out_dir, exist_ok=True)
    stem = name or os.path.splitext(os.path.basename(scan.path or "scan"))[0]
    feats = known_dimensions(json.loads(json.dumps(feats)), known)
    byid = {f["id"]: f for f in feats.get("planes", []) + feats.get("cylinders", [])}
    sup = [support] if isinstance(support, str) else list(support or [])
    if isinstance(loads, dict):
        loads = list(loads.items())
    loads = [(k, [float(x) for x in v]) for k, v in (loads or [])]
    used = sup + [k for k, _ in loads]
    for k in used:
        if k not in byid:
            raise ValueError(f"no feature {k!r}; the scan has {sorted(byid)}")
    if not sup or not loads:
        raise ValueError("give support (feature ids the part is held by) and loads ([(feature id, force N)]): "
                         "the scan says where things are, not what they carry")
    todo = []
    z1 = z_value(scan.confidence)

    # ---- the design space: a box round what the part touches
    pts = []
    for k in used:
        f = byid[k]
        if f["type"] == "plane":
            c = np.asarray(f["center"])
            e1, e2 = (np.asarray(a) for a in f["axes"])
            (u0, v0), (u1, v1) = f["extent"]
            pts += [c + u * e1 + v * e2 for u in (u0, u1) for v in (v0, v1)]
        else:
            pts += [np.asarray(f["p0"]), np.asarray(f["p1"])]
    pts = np.array(pts)
    # a long bar only counts where the other features are along it
    others = [k for k in used if byid[k]["type"] != "cylinder" or byid[k]["kind"] != "bar"]
    op = None
    if others:
        op = []
        for k in others:
            f = byid[k]
            if f["type"] == "plane":
                c = np.asarray(f["center"])
                e1, e2 = (np.asarray(a) for a in f["axes"])
                (u0, v0), (u1, v1) = f["extent"]
                op += [c + u * e1 + v * e2 for u in (u0, u1) for v in (v0, v1)]
            else:
                op += [np.asarray(f["p0"]), np.asarray(f["p1"])]
        op = np.array(op)
        pts = list(op)
        for k in used:
            f = byid[k]
            if f["type"] == "cylinder" and f["kind"] == "bar":
                a = np.asarray(f["axis"])
                p0 = np.asarray(f["p0"])
                t = (op - p0) @ a
                tl, th = max(t.min(), 0.0), min(t.max(), f["length"])
                for tt in (tl, th):
                    q = p0 + tt * a
                    e1 = np.cross(a, [1, 0, 0] if abs(a[0]) < 0.9 else [0, 1, 0])
                    e1 /= np.linalg.norm(e1)
                    e2 = np.cross(a, e1)
                    R = f["r"] + ring_wall(f["r"])
                    pts += [q + R * (u * e1 + v * e2) for u in (-1, 1) for v in (-1, 1)]
        pts = np.array(pts)
    if space is not None:
        lo, hi = (np.asarray(x, float) for x in space)
    else:
        lo, hi = _box_of(pts, grow)
        todo.append(f"the design space is the box round the features the part touches, grown {grow:g} mm: "
                    "give space= (or grow=) for what the part may really fill")
    design = [{"type": "box", "min": lo.round(3).tolist(), "max": hi.round(3).tolist()}]
    reach = float(np.linalg.norm(hi - lo))

    keep_out, preserve = [], []
    clearance = None
    # ---- bars: fitted cylinders with the measured band
    bar_ids = []
    for f in feats.get("cylinders", []):
        if f["kind"] != "bar" or f["id"] not in used:
            continue
        bar_ids.append(f["id"])
        # runs on past the scan's crop and through the design space
        p0, p1 = _cyl_keepout(f, reach, reach)
        ko = {"id": f["id"], "shape": {"type": "cylinder", "p0": p0.round(4).tolist(), "p1": p1.round(4).tolist(),
                                       "r": round(f["r"], 5)},
              "fit": bar_fit, "mate": round(f["diameter"], 5),
              "mateUncertainty": round(0.5 * f["diameterBand"], 5)}
        keep_out.append(ko)
        # a clamp holds a bar by going round it: a ring (the import's wall for
        # a bore this size) over the stretch of bar the other features span
        # (or the scanned length), its bore following the fit
        a = np.asarray(f["axis"], float)
        q0 = np.asarray(f["p0"], float)
        t_lo, t_hi = 0.0, float(f["length"])
        if others:
            t = (op - q0) @ a
            t_lo, t_hi = max(float(t.min()), 0.0), min(float(t.max()), float(f["length"]))
            if t_hi - t_lo < 2 * ring_wall(f["r"]):
                mid = 0.5 * (t_lo + t_hi)
                t_lo, t_hi = mid - ring_wall(f["r"]), mid + ring_wall(f["r"])
        preserve.append({"id": f"{f['id']}-ring", "shape": {
            "type": "cylinder", "p0": (q0 + t_lo * a).round(4).tolist(), "p1": (q0 + t_hi * a).round(4).tolist(),
            "r": round(f["r"] + ring_wall(f["r"]), 4), "rInner": round(f["r"], 5)}})
        if f.get("warning"):
            todo.append(f"{f['id']}: {f['warning']}")
    # ---- bores in the object: kept open out through the design space
    for f in feats.get("cylinders", []):
        if f["kind"] != "bore":
            continue
        a = np.asarray(f["axis"], float)
        p0, p1 = np.asarray(f["p0"]), np.asarray(f["p1"])
        r = f["r"] + f["rBand"]
        keep_out.append({"id": f"{f['id']}-open", "bore": False,
                         "shape": {"type": "cylinder", "p0": (p0 - reach * a).round(4).tolist(),
                                   "p1": (p1 + reach * a).round(4).tolist(), "r": round(r, 4)}})
    # ---- faces: a pad set off by the face's band; the solid side kept out
    for f in feats.get("planes", []):
        if f["id"] not in used:
            continue
        n = np.asarray(f["normal"], float)
        c = np.asarray(f["center"], float)
        e1, e2 = (np.asarray(a) for a in f["axes"])
        (u0, v0), (u1, v1) = f["extent"]
        # one-sided: how far the real face may stand proud of the fitted one
        off = z1 * f["sigma"]["total"]
        holes = _process_holes(process)
        if holes is not None:
            off = z1 * math.hypot(f["sigma"]["total"], 0.5 * holes["tol"]) + \
                max(0.0, -0.25 * (holes["bias"] + holes["biasAcross"]))
        pad_c = c + n * (off + 0.5 * pad)
        preserve.append({"id": f"{f['id']}-pad", "shape": {
            "type": "box", "center": (pad_c + e1 * 0.5 * (u0 + u1) + e2 * 0.5 * (v0 + v1)).round(4).tolist(),
            "size": [round(u1 - u0, 3), round(v1 - v0, 3), round(pad, 3)],
            "axes": [e1.round(6).tolist(), e2.round(6).tolist()]}})
        # the face's own side: the object (or the wall) is there
        depth = reach
        back = c - n * (0.5 * depth)
        keep_out.append({"id": f"{f['id']}-behind", "bore": False, "shape": {
            "type": "box", "center": (back + e1 * 0.5 * (u0 + u1) + e2 * 0.5 * (v0 + v1) + n * off).round(4).tolist(),
            "size": [round(u1 - u0, 3), round(v1 - v0, 3), round(depth, 3)],
            "axes": [e1.round(6).tolist(), e2.round(6).tolist()]}})
        f["_off"] = off
    # ---- the object itself, moved out by its clearance
    if keep_object:
        if mesh_info is None:
            mesh, info = reconstruct(scan)
        else:
            mesh, info = mesh_info
        dv, iv = scan.nearest(mesh.vertices)
        info = {**info, "dv": dv, "iv": iv}
        U, clearance = object_clearance(scan, info, process, size=float(np.max(hi - lo)))
        solid = _offset_mesh(mesh, U)
        m = manifold_from_trimesh(solid)
        # crop to the design space (and a margin), and leave the bars to their fits
        box = trimesh.creation.box(bounds=[lo - 2.0, hi + 2.0])
        m = m ^ manifold_from_trimesh(box)
        for fid in bar_ids:
            f = byid[fid]
            p0, p1 = _cyl_keepout(f, reach, reach)
            # the bar's own offset surface goes: its fitted bore (fits.py, with
            # the measured band) decides the clearance round it
            rr = f["r"] + f["rBand"] + float(np.max(U)) + 1.0
            m = m - Cylinder(p0, p1, rr, 0.0).to_manifold()
        obj = _closed(trimesh_from_manifold(m))
        if len(obj.faces):
            path = os.path.join(out_dir, f"{stem}-object-clear.stl")
            obj.export(path)
            keep_out.append({"id": "object", "bore": False, "shape": {"type": "mesh", "path": path}})
        if info["report"].get("unseenShare", 0) > 0.01:
            todo.append(f"{100 * info['report']['unseenShare']:.1f}% of the object was not seen (worst gap "
                        f"{info['report']['unseenWorstGap']:.1f} mm): kept clear by the gap, rescan those sides "
                        "to tighten it")
    # ---- supports and loads
    region = {}
    for k in used:
        f = byid[k]
        region[k] = f["id"] if f["type"] == "cylinder" else f"{f['id']}-pad"
    study = {"name": stem,
             "description": f"Drafted by `loadpath scan` round {os.path.basename(scan.path or 'a scan')}: "
                            f"clearances from the scan's measured uncertainty at {scan.confidence:.0%} "
                            f"({scan.device.get('name') or 'no device'}) and the process's scatter.",
             "designSpace": design, "preserve": preserve, "keepOut": keep_out, "material": material,
             "goal": {"safetyFactor": safety_factor},
             "supports": [{"region": region[k]} for k in sup],
             "loadCases": [{"name": f"load-{i + 1}", "loads": [{"region": region[k], "force": fv,
                                                               **({"distribution": "bearing"} if byid[k]["type"] == "cylinder" else {})}]}
                           for i, (k, fv) in enumerate(loads)]}
    if process:
        study["process"] = process
    path = os.path.join(out_dir, f"{stem}.study.json")
    with open(path, "w") as fh:
        json.dump(study, fh, indent=1)
    feats_out = {k: v for k, v in feats.items()}
    for f in feats_out.get("planes", []):
        f.pop("_off", None)
    with open(os.path.join(out_dir, f"{stem}.features.json"), "w") as fh:
        json.dump(feats_out, fh, indent=1)
    return {"study": study, "studyPath": path, "clearance": clearance, "todo": todo}


def _parse_band(txt):
    """'31.8+-0.05' / '31.8±0.05' / '31.8' -> (value, tolerance or None)."""
    t = str(txt).replace("±", "+-")
    if "+-" in t:
        v, tol = t.split("+-")
        return float(v), float(tol)
    return float(t), None


def parse_reference(items) -> list:
    """CLI references: 'cyl-0=31.8+-0.05' (a diameter), 'plane-0/plane-1=40+-0.05'
    (two parallel faces' distance), or a JSON object."""
    out = []
    for it in items or []:
        if isinstance(it, dict):
            out.append(it)
            continue
        it = str(it).strip()
        if it.startswith("{"):
            out.append(json.loads(it))
            continue
        k, _, v = it.partition("=")
        val, tol = _parse_band(v)
        if "/" in k:
            a, b = k.split("/")
            r = {"feature": a, "to": b, "distance": val}
        else:
            r = {"feature": k, "diameter": val}
        if tol is not None:
            r["tolerance"] = tol
        out.append(r)
    return out


def scan_to_study(path, out_dir=None, *, device=None, units=None, scale=None, up=None, crop=None, keep="all",
                  reference=None, known=None, support=None, loads=None, material="pa12", process="sls",
                  grow=15.0, confidence=0.95, bar_fit="slip", max_points=300_000, pad=2.0) -> dict:
    """The whole path: read and measure the scan, fit its features, fix its
    scale from references, and (with out_dir, support and loads) draft the
    study round it. Returns {"report", "features", "summary", ...}."""
    from .scan import fit_primitives, load_scan, apply_reference, scale_features, summarize
    s = load_scan(path, device=device, units=units, scale=scale, up=up, crop=crop, keep=keep,
                  max_points=max_points, confidence=confidence)
    feats = fit_primitives(s)
    refs = parse_reference(reference) if reference else None
    if refs:
        s = apply_reference(s, refs, feats)
        feats = scale_features(feats, s)
    if isinstance(known, (list, tuple)):
        kk = {}
        for it in known:
            k, _, v = str(it).partition("=")
            val, tol = _parse_band(v)
            kk[k] = {"diameter": val, **({"tolerance": tol} if tol is not None else {})}
        known = kk
    out = {"report": s.report, "features": feats, "summary": summarize(s, feats)}
    if out_dir is not None and (support or loads):
        r = draft_around(s, feats, out_dir, support=support, loads=loads, material=material, process=process,
                         grow=grow, known=known, bar_fit=bar_fit, pad=pad)
        out.update({"study": r["study"], "studyPath": r["studyPath"], "clearance": r["clearance"],
                    "todo": s.report.get("todo", []) + r["todo"]})
    else:
        out["todo"] = s.report.get("todo", []) + (
            ["pick support and loads by feature id (summary.features) to draft the study round the object"]
            if out_dir is not None else [])
    return out
