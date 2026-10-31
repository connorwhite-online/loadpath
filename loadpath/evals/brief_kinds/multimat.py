"""
Multi-material briefs: a part printed in two (or more) materials, such as a
rigid core with a TPU grip pad, a bracket with a rubber isolating layer, or
a PolyJet Vero body with an Agilus30 skin.

The agent hands back one body per material (a multi-material 3MF, or one
STL per material). The scorer reads the bodies and re-analyses them under
the GOLD study, ignoring whatever the agent claims:

    pass = the bodies are in materials the brief allows (at least
           pass.minMaterials of them), do not overlap, and form one
           connected part; the part is inside the envelope, clear of every
           gold keep-out and has every gold interface; every gold load case
           reads each material body's safety factor (multimat.
           verify_multimaterial: per-cell material, across-layer strength
           for FDM) >= SF_SLACK x pass.safetyFactor and every interface
           between materials (the bond card from interface_library, face
           by face) >= SF_SLACK x pass.interfaceSafetyFactor; each
           pass.zones entry (where the brief needs a soft or a rigid
           material) is filled and made of the materials it names; each
           pass.isolate entry has no rigid path between its two regions;
           and the mass is under pass.maxMass.

The gold answer: brief["reference"], a list of {material, shape} solids
(earlier ones win where they overlap, keep-outs are cut away), is the
minimum defensible design; gold(brief) builds it, writes it as a 3MF the
way loadpath does and scores it, so the gold passes its own scorer.

Fields of a multimat brief beyond the default kind:
    gold                  a study with a multiMaterial block (the materials
                          the reading uses, any interface override)
    reference             the gold design as [{material, shape}]
    pass.safetyFactor, pass.interfaceSafetyFactor, pass.maxMass
    pass.materials        materials the delivery may use (any others fail)
    pass.minMaterials     default 2
    pass.zones            [{id, region | shape, materials, minShare, minFill?}]
                          minShare: of the part inside the zone, the volume
                          share in `materials`; minFill: of the zone (less
                          keep-outs), the share that is part at all
    pass.isolate          [{id, from, to, materials}]: no connected body of
                          `materials` touches both regions (a rubber layer
                          that must break every rigid path)
"""
from __future__ import annotations

import glob
import os
import re
import tempfile
import time
import xml.etree.ElementTree as ET
import zipfile

import numpy as np

from ..briefs import (
    ASSETS,
    MASS_SLACK,
    MAX_INTRUSION,
    MAX_MISSING,
    MAX_OUTSIDE,
    SF_SLACK,
    gold_study,
)

DELIVER = """  part.3mf     the part as one multi-material 3MF, one body per material, in the coordinates
               the brief uses (loadpath writes it so); or, instead, one STL per material named
               part-<material>.stl (e.g. part-petg.stl, part-tpu.stl), also in the brief's coordinates
  study.json   the study that produced it
  notes.md     two or three lines: the mass, the safety factor of each material and of the bond
               between them, which material is where, and your assumptions
"""

VERIFY_CELLS = 60_000
MAX_OVERLAP = 30.0          # mm^3 two bodies may share (a mesh's facets, not a double-printed volume)
MIN_CONNECTED = 0.97        # share of the delivered volume the analysis must keep (attached to the supports)
VOXEL_CELLS = 400_000       # connectivity grid (pieces, rigid paths)
INTERLOCK_RATIO = 1.8       # raw / smoothed interface area that reads as an interlocking lattice
INTERLOCK_SMOOTH = 2.5      # mm: the majority-vote box (a lattice period is two 0.8 mm beams)

_UNITS = {"micron": 1e-3, "millimeter": 1.0, "centimeter": 10.0, "inch": 25.4, "foot": 304.8, "meter": 1000.0}


# ---------------------------------------------------------------- reading

def _material_key(name):
    """A material name, alias or file-name fragment as a library key, or None."""
    from ...materials import material_from_spec

    try:
        return material_from_spec(str(name).strip()).name
    except (ValueError, TypeError):
        return None


def _stem_material(stem):
    """The material a file name ends with: part-petg, finger_tpu95a, x-pa-cf."""
    from ...materials import _ALIASES, MATERIALS

    s = stem.lower()
    best = None
    for key in list(MATERIALS) + list(_ALIASES):
        for k in {key, key.replace(" ", "-"), key.replace(" ", "_"), key.replace(" ", "")}:
            hit = s == k or re.search(rf"(^|[-_. ]){re.escape(k)}($|[-_. ]t?\d*$)", s)
            if hit and (best is None or len(k) > len(best[0])):
                best = (k, _material_key(key))
    return best[1] if best else None


def _matrix(text):
    """A 3MF transform attribute (12 numbers, row vectors) as a 4x4 column-vector matrix."""
    if not text:
        return np.eye(4)
    v = np.array([float(x) for x in text.split()], float).reshape(4, 3)
    m = np.eye(4)
    m[:3, :3] = v[:3].T
    m[:3, 3] = v[3]
    return m


def _local(tag):
    return tag.rsplit("}", 1)[-1]


def read_3mf(path):
    """The bodies of a 3MF: [(material key or None, vertices in object space,
    faces, transform from object space through components, build transform)].
    Materials come from core basematerials (object pid/pindex, triangle
    pid/p1), else from the object's name or loadpath:material metadata.
    Production-extension components (p:path into another model file, the
    Bambu layout) are followed."""
    with zipfile.ZipFile(path) as z:
        models = {"/" + n.lstrip("/"): z.read(n) for n in z.namelist() if n.lower().endswith(".model")}
        cfg = z.read("Metadata/Slic3r_PE_model.config") if "Metadata/Slic3r_PE_model.config" in z.namelist() \
            else None
    volumes = {}     # PrusaSlicer: object id -> [(first, last) triangle of each volume]
    if cfg is not None:
        for o in ET.fromstring(cfg):
            if _local(o.tag) == "object":
                volumes[o.get("id")] = [(int(v.get("firstid")), int(v.get("lastid"))) for v in o
                                        if _local(v.tag) == "volume"]
    roots, objects, bases, units = {}, {}, {}, {}
    for name, data in models.items():
        root = ET.fromstring(data)
        roots[name] = root
        units[name] = _UNITS.get(root.get("unit", "millimeter"), 1.0)
        for res in root:
            if _local(res.tag) != "resources":
                continue
            for el in res:
                tag = _local(el.tag)
                if tag == "basematerials":
                    bases[(name, el.get("id"))] = [b.get("name") for b in el if _local(b.tag) == "base"]
                elif tag == "object":
                    objects[(name, el.get("id"))] = el

    def mat_of(model, pid, idx, fallback):
        names = bases.get((model, pid))
        if names is not None and idx is not None and 0 <= int(idx) < len(names):
            k = _material_key(names[int(idx)]) or _stem_material(names[int(idx)])
            if k:
                return k
        return fallback

    out = []

    def expand(model, oid, tf, build):
        el = objects.get((model, oid))
        if el is None:
            raise ValueError(f"3MF object {oid} not found in {model}")
        meta = {(_local(m.tag), m.get("name")): (m.text or "") for m in el.iter() if _local(m.tag) == "metadata"}
        fallback = None
        for (_, k), v in meta.items():
            if k and k.endswith("material"):
                fallback = _material_key(v) or fallback
        fallback = fallback or _stem_material(el.get("name", "") or "")
        for child in el:
            tag = _local(child.tag)
            if tag == "mesh":
                vs, ts = None, []
                for part in child:
                    if _local(part.tag) == "vertices":
                        vs = np.array([[float(v.get("x")), float(v.get("y")), float(v.get("z"))] for v in part],
                                      float) * units[model]
                    elif _local(part.tag) == "triangles":
                        ts = [(int(t.get("v1")), int(t.get("v2")), int(t.get("v3")),
                               t.get("pid") or el.get("pid"), t.get("p1") if t.get("pid") or t.get("p1")
                               else el.get("pindex")) for t in part]
                if vs is None or not ts:
                    continue
                faces = np.array([t[:3] for t in ts], np.int64)
                keys = np.array([str(mat_of(model, t[3], t[4], fallback)) for t in ts])
                segs = volumes.get(el.get("id")) if model == "/3D/3dmodel.model" else None
                for a, z_ in segs or [(0, len(ts) - 1)]:
                    for k in sorted(set(keys[a:z_ + 1])):
                        sel = np.zeros(len(ts), bool)
                        sel[a:z_ + 1] = keys[a:z_ + 1] == k
                        out.append((None if k == "None" else k, vs, faces[sel], tf, build))
            elif tag == "components":
                for c in child:
                    ref = None
                    for a, v in c.attrib.items():
                        if _local(a) == "path":
                            ref = "/" + v.lstrip("/")
                    expand(ref or model, c.get("objectid"), tf @ _matrix(c.get("transform")), build)

    for name, root in roots.items():
        for b in root:
            if _local(b.tag) != "build":
                continue
            for item in b:
                if _local(item.tag) == "item":
                    expand(name, item.get("objectid"), np.eye(4), _matrix(item.get("transform")))
    return out


def _mesh(vs, faces, tf):
    import trimesh

    used, inv = np.unique(faces.ravel(), return_inverse=True)
    v = vs[used] @ tf[:3, :3].T + tf[:3, 3]
    m = trimesh.Trimesh(v, inv.reshape(-1, 3), process=True)
    if m.volume < 0:
        m.invert()
    return m


def read_delivery(folder, envelope=None):
    """[(material key, trimesh)] from the agent's folder: part.3mf (any .3mf
    if there is no part.3mf), else STLs named by material. A 3MF keeps its
    meshes in design coordinates and the print orientation in the build
    item (as loadpath writes it); one that baked the bed placement into the
    vertices is read with the build transform when that lands it inside the
    envelope instead. Returns (bodies, source, warnings)."""
    warn = []
    tmf = [os.path.join(folder, "part.3mf")] if os.path.exists(os.path.join(folder, "part.3mf")) else \
        sorted(glob.glob(os.path.join(folder, "*.3mf")))
    if tmf:
        raw = read_3mf(tmf[0])
        if not raw:
            raise ValueError(f"{os.path.basename(tmf[0])} has no mesh")

        def place(with_build):
            return [(k, _mesh(v, f, (b @ t) if with_build else t)) for k, v, f, t, b in raw]

        bodies = place(False)
        if envelope is not None and any(not np.allclose(b, np.eye(4)) for *_, b in raw):
            lo, hi = envelope

            def inside(bs):
                v = np.vstack([m.vertices for _, m in bs])
                return float(np.all((v >= lo - 1.0) & (v <= hi + 1.0), axis=1).mean())

            alt = place(True)
            if inside(alt) > inside(bodies) + 0.2:
                bodies = alt
                warn.append("3MF vertices read through the build transform (they are not in design coordinates)")
        return bodies, os.path.basename(tmf[0]), warn
    import trimesh

    bodies = []
    for p in sorted(glob.glob(os.path.join(folder, "*.stl"))):
        k = _stem_material(os.path.splitext(os.path.basename(p))[0])
        if k is None:
            warn.append(f"{os.path.basename(p)}: no material in the file name, ignored")
            continue
        bodies.append((k, trimesh.load(p, force="mesh")))
    if not bodies:
        raise ValueError("no part.3mf and no STL named by material (part-<material>.stl)")
    return bodies, "stl", warn


# ---------------------------------------------------------------- analysis

def _man(mesh):
    from ...shapes import manifold_from_trimesh

    return manifold_from_trimesh(mesh)


def _solids(bodies):
    """[(material, manifold, trimesh)] for each delivered body with volume.
    Bodies of one material stay separate: two that share a face do not fuse
    into one closed shell, and a shell pair would break ray-parity voxels."""
    from ...shapes import trimesh_from_manifold

    out = []
    for k, m in bodies:
        man = _man(m)
        if man.volume() > 1e-6:
            out.append((k, man, trimesh_from_manifold(man)))
    return out


def _voxels(solids, cells=VOXEL_CELLS):
    """The bodies on one grid: per cell the index into `solids` (-1 void),
    earlier bodies first. Returns (label, origin, pitch)."""
    from ...verify import voxelize

    lo = np.min([m.bounds[0] for _, _, m in solids], axis=0)
    hi = np.max([m.bounds[1] for _, _, m in solids], axis=0)
    pitch = max(float(np.prod(hi - lo) / cells) ** (1.0 / 3.0), 0.1)
    label, origin = None, None
    for i, (_, _, m) in enumerate(solids):
        s, origin = voxelize(m, pitch, box=(lo, hi))
        if label is None:
            label = np.full(s.shape, -1, np.int64)
        label[s & (label < 0)] = i
    return label, origin, pitch


def _interlock_ratio(solids, label, pitch, width=INTERLOCK_SMOOTH):
    """How much more interface the bodies have than their smoothed layout:
    material faces between different materials on the voxel grid, over the
    same count once each material's indicator is box-averaged over `width`
    mm (a majority vote). About 1 for a flat or curved joint, several times
    that for an interlocking beam lattice. None with one material."""
    from scipy import ndimage

    if len({k for k, _, _ in solids}) < 2:
        return None
    keys = sorted({k for k, _, _ in solids})
    mat = np.full(label.shape, -1, np.int64)
    for i, (k, _, _) in enumerate(solids):
        mat[label == i] = keys.index(k)
    solid = mat >= 0
    n = max(3, round(width / pitch) | 1)
    den = ndimage.uniform_filter(solid.astype(np.float32), n)
    votes = np.stack([ndimage.uniform_filter((mat == j).astype(np.float32), n) / np.maximum(den, 1e-6)
                      for j in range(len(keys))])
    smooth = np.where(solid, votes.argmax(axis=0), -1)

    def faces(m):
        c = 0
        for ax in range(3):
            a = np.moveaxis(m, ax, 0)
            x, y = a[1:], a[:-1]
            c += int(((x >= 0) & (y >= 0) & (x != y)).sum())
        return c

    raw, sm = faces(mat), faces(smooth)
    return raw / sm if sm else None


def _shape_of(study, z):
    from ...shapes import shape_from_spec

    return study.region_shape(z["region"])[0] if "region" in z else shape_from_spec(z["shape"], ASSETS)


def _zone_shape(study, z):
    from ...sharp import exact_manifold

    return exact_manifold(_shape_of(study, z))


def analyse(brief, bodies, max_cells=VERIFY_CELLS):
    """Re-analyse delivered [(material key, trimesh)] under the gold study.
    Returns the score row (without "pass")."""
    import manifold3d as m3d

    from ...multimat import parse_multimaterial, verify_multimaterial
    from ...verify import _preserve_missing, _union

    want = brief["pass"]
    study = gold_study(brief)
    gmm = brief["gold"].get("multiMaterial") or {}
    row = {"failures": []}
    f = row["failures"]
    merged = _solids(bodies)
    if not merged:
        f.append("no solid body")
        return row
    unknown = [k for k, _, _ in merged if k is None]
    if unknown:
        f.append(f"{len(unknown)} body/bodies with no material")
        merged = [b for b in merged if b[0] is not None]
    names = list(dict.fromkeys(k for k, _, _ in merged))
    row["materials"] = names
    allowed = [_material_key(m) for m in want.get("materials", [])]
    if allowed:
        bad = [k for k in names if k not in allowed]
        if bad:
            f.append(f"material(s) {bad} not what the brief asks for ({sorted(set(allowed))})")
    if len(names) < want.get("minMaterials", 2):
        f.append(f"{len(names)} material(s); the brief needs {want.get('minMaterials', 2)}")

    # geometry on the whole part
    part = m3d.Manifold.batch_boolean([m for _, m, _ in merged], m3d.OpType.Add) if len(merged) > 1 \
        else merged[0][1]
    vol = float(part.volume())
    body_vol = sum(float(m.volume()) for _, m, _ in merged)
    row["overlap"] = round(body_vol - vol, 1)
    if row["overlap"] > MAX_OVERLAP:
        f.append(f"bodies overlap by {row['overlap']} mm^3")
    # connectivity on voxels: bodies that only share a face stay apart in manifold3d
    from scipy import ndimage

    label, origin, vpitch = _voxels(merged)
    comp, _ = ndimage.label(label >= 0)
    sizes = np.bincount(comp.ravel())[1:]
    row["bodies"] = int((sizes * vpitch ** 3 > 1.0).sum())
    if row["bodies"] != 1:
        f.append(f"{row['bodies']} separate pieces")
    design = _union(study.design_space + [r.shape for r in study.preserve])
    keep = _union([r.shape for r in study.keep_out])
    pres = _union([r.shape for r in study.preserve])
    row["keepOutIntrusion"] = round(float((part ^ keep).volume()) if keep is not None else 0.0, 1)
    row["outsideDesignSpace"] = round(float((part - design).volume()), 1)
    if pres is not None:
        from ...shapes import trimesh_from_manifold
        row["preserveMissing"] = round(_preserve_missing(pres - keep if keep is not None else pres, part,
                                                         trimesh_from_manifold(part)), 1)
    else:
        row["preserveMissing"] = 0.0
    if row["outsideDesignSpace"] > MAX_OUTSIDE:
        f.append(f"{row['outsideDesignSpace']} mm^3 outside the envelope")
    if row["keepOutIntrusion"] > MAX_INTRUSION:
        f.append(f"{row['keepOutIntrusion']} mm^3 in keep-outs")
    if row["preserveMissing"] > MAX_MISSING:
        f.append(f"{row['preserveMissing']} mm^3 of interfaces missing")

    # where each material has to be
    zones = {}
    for z in want.get("zones", []):
        zm = _zone_shape(study, z)
        if keep is not None:
            zm = zm - keep
        ok = {_material_key(m) for m in z["materials"]}
        inside = {}
        for k, m, _ in merged:
            inside[k] = inside.get(k, 0.0) + float((m ^ zm).volume())
        tot = sum(inside.values())
        share = sum(v for k, v in inside.items() if k in ok) / tot if tot > 0 else 0.0
        fill = tot / max(float(zm.volume()), 1e-9)
        zones[z["id"]] = {"share": round(share, 3), "fill": round(fill, 3)}
        if share < z.get("minShare", 0.9):
            f.append(f"zone '{z['id']}': {share:.0%} {'/'.join(sorted(ok))} (needs {z.get('minShare', 0.9):.0%})")
        if z.get("minFill") and fill < z["minFill"]:
            f.append(f"zone '{z['id']}': only {fill:.0%} filled (needs {z['minFill']:.0%})")
    row["zones"] = zones
    centres = None
    for iso in want.get("isolate", []):
        ok = {_material_key(m) for m in iso["materials"]}
        hard = np.isin(label, [i for i, (k, _, _) in enumerate(merged) if k in ok])
        comp, _ = ndimage.label(hard)
        if centres is None:
            idx = np.argwhere(label >= 0)
            centres = (idx, origin + (idx + 0.5) * vpitch)
        idx, pts = centres
        c = comp[tuple(idx.T)]
        a = c[_shape_of(study, iso["from"]).contains(pts) & (c > 0)]
        b = c[_shape_of(study, iso["to"]).contains(pts) & (c > 0)]
        bridged = bool(np.intersect1d(np.unique(a), np.unique(b)).size)
        row.setdefault("isolated", {})[iso["id"]] = not bridged
        if bridged:
            f.append(f"'{iso['id']}': a rigid path joins the two sides ({'/'.join(sorted(ok))})")

    # the FEA: every material and every bond, per gold case
    mats = list(dict.fromkeys(names + [_material_key(m) for m in gmm.get("materials", [])]))[:4]
    if len(mats) < 2:
        mats.append("tpu" if mats[0] != "tpu" else "petg")
    block = {"materials": mats}
    for k in ("interfaces", "interlock", "interfaceSafetyFactor", "requirements"):
        if gmm.get(k) is not None:
            block[k] = gmm[k]
    # An interlocked joint (the materials interpenetrate in a beam lattice)
    # is credited the way loadpath credits it, but only when the geometry
    # shows one: the agent saying so is not enough.
    ratio = _interlock_ratio(merged, label, vpitch)
    row["interfaceRoughness"] = round(ratio, 2) if ratio else None
    if ratio and ratio >= INTERLOCK_RATIO and "interlock" not in block:
        block["interlock"] = True
        row["interlockDetected"] = True
    mm = parse_multimaterial(block, study, ASSETS)
    order = sorted(range(len(merged)), key=lambda i: -mm.materials[mm.index(merged[i][0])].E)
    vb = [(mm.index(merged[i][0]), merged[i][2]) for i in order]   # stiff first: shared cells go rigid
    t0 = time.time()
    try:
        rep = verify_multimaterial(study, mm, vb, max_cells=max_cells)
    except (ValueError, RuntimeError) as exc:
        # verify keeps only what is attached to the supports: a load left on a loose piece lands here
        msg = str(exc).strip().splitlines()[-1] if str(exc).strip() else type(exc).__name__
        f.append(f"analysis failed (is the loaded part attached to the supports?): {msg[:200]}")
        return row
    row["verifySeconds"] = round(time.time() - t0, 1)
    row["analysisPitch"] = round(rep["analysisPitch"], 3)
    kept = rep["cells"] * rep["analysisPitch"] ** 3 / max(vol, 1e-9)
    row["analysedShare"] = round(kept, 3)
    if kept < MIN_CONNECTED:
        f.append(f"only {kept:.0%} of the part is attached to the supports")
    row["mass"] = round(rep["mass"], 2)
    row["materialMass"] = {k: round(v, 2) for k, v in rep["materialMass"].items()}
    row["interfaceArea"] = {k: round(v, 1) for k, v in rep["interfaceArea"].items() if v > 0}
    sf, sfi = want.get("safetyFactor"), want.get("interfaceSafetyFactor", want.get("safetyFactor"))
    row["cases"] = []
    for c in rep["cases"]:
        rc = {"name": c["name"],
              "materials": {k: round(v["safetyFactor"], 2) for k, v in c["materials"].items()},
              "interfaces": {k: round(min(v["safetyFactor"], 1e3), 2) for k, v in c["interfaces"].items()}}
        row["cases"].append(rc)
        for k, v in rc["materials"].items():
            if sf and v < SF_SLACK * sf:
                f.append(f"{k} SF {v:.2f} < {sf} in '{c['name']}'")
        for k, v in rc["interfaces"].items():
            if sfi and v < SF_SLACK * sfi:
                f.append(f"{k} bond SF {v:.2f} < {sfi} in '{c['name']}'")
        for key in ("knock", "indentation"):
            if key in c and not c[key]["meets"]:
                f.append(f"'{c['name']}' {key} requirement not met")
    row["safetyFactor"] = round(rep["safetyFactor"], 3)
    row["interfaceSafetyFactor"] = round(min(rep["interfaceSafetyFactor"], 1e3), 3)
    if len(names) >= 2 and not row["interfaceArea"]:
        f.append("the materials never meet (no interface)")
    mmax = want.get("maxMass")
    if mmax and rep["mass"] > MASS_SLACK * mmax:
        f.append(f"{rep['mass']:.1f} g > {mmax} g")
    return row


def score(brief, folder, max_cells=VERIFY_CELLS) -> dict:
    """Read the agent's bodies from `folder` and re-analyse them under the
    gold study (see the module docstring)."""
    study = gold_study(brief)
    lo, hi = study.bounds()
    try:
        bodies, source, warn = read_delivery(folder, (np.asarray(lo), np.asarray(hi)))
    except (ValueError, OSError, zipfile.BadZipFile, ET.ParseError) as exc:
        return {"pass": False, "failures": [f"no multi-material part: {exc}"]}
    row = analyse(brief, bodies, max_cells)
    row["source"] = source
    if warn:
        row["warnings"] = warn
    row["pass"] = not row["failures"]
    return row


# ---------------------------------------------------------------- the gold

def reference_bodies(brief):
    """brief["reference"] as [(material key, trimesh)]: each solid less the
    earlier ones and the gold keep-outs."""
    import manifold3d as m3d

    from ...shapes import shape_from_spec, trimesh_from_manifold
    from ...sharp import exact_manifold
    from ...verify import _union

    study = gold_study(brief)
    keep = _union([r.shape for r in study.keep_out])
    taken, out = None, []
    for r in brief["reference"]:
        m = exact_manifold(shape_from_spec(r["shape"], ASSETS))
        if keep is not None:
            m = m - keep
        if taken is not None:
            m = m - taken
        taken = m if taken is None else m3d.Manifold.batch_boolean([taken, m], m3d.OpType.Add)
        if m.volume() > 1e-6:
            out.append((_material_key(r["material"]), trimesh_from_manifold(m)))
    return out


def write_delivery(bodies, folder, name="part", process=None, direction=None, slicer="all"):
    """Bodies as the agent would leave them: part.3mf (loadpath's writer)."""
    from ...threemf import write_multimaterial_3mf

    os.makedirs(folder, exist_ok=True)
    spec = [{"name": f"{name} {k}", "mesh": m, "material": k} for k, m in bodies]
    return write_multimaterial_3mf(os.path.join(folder, "part.3mf"), spec, direction=direction, name=name,
                                   process=process, slicer=slicer)


def gold(brief, max_cells=VERIFY_CELLS) -> dict:
    """The gold study lints and builds, and the reference design, written as
    a 3MF and read back, passes the scorer."""
    from ...tasks import check_study

    res = check_study(brief["gold"], base_dir=ASSETS)
    if not res["ok"]:
        return res
    if not brief.get("reference"):
        return {"ok": False, "error": "multimat brief has no reference design"}
    study = gold_study(brief)
    with tempfile.TemporaryDirectory() as d:
        write_delivery(reference_bodies(brief), d, process=study.process, direction=study.print_direction)
        row = score(brief, d, max_cells)
    keys = ("mass", "safetyFactor", "interfaceSafetyFactor", "zones", "verifySeconds")
    out = {"ok": row["pass"], "cells": res.get("cells", ""), "warnings": res.get("warnings", []),
           "score": {k: row.get(k) for k in keys}}
    if not row["pass"]:
        out["error"] = "reference fails its own scorer: " + "; ".join(row["failures"])
    return out
