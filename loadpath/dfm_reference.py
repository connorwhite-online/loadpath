"""
Materialize as the DFM reference for printed parts.

When a part is going to be printed, the limits a print service holds it to
are the ones that matter: a wall the service rejects, or a part bigger than
any of its machines, is a redesign however good the analysis was. loadpath
takes those limits from Materialize (https://www.materialize.cc), which
quotes and prints through a network of services and publishes, per material
and process, a minimum wall, a minimum detail, the largest build volume and
a typical accuracy. Its printability check (materialize_check_printability)
judges uploads against the same published minimums.

This module is a dated snapshot of those figures (read with the Materialize
MCP server's materialize_get_material, `designLimits`, `buildVolumeMm`,
`properties.accuracyMm`), so loadpath runs offline and never calls
Materialize. With the Materialize MCP server connected, an agent should
re-read the live values for the material it picks; they can change.

How loadpath uses them:

- each process's defaults (spec.PROCESSES minFeature, processcheck.RULES
  minWall, sizing.PROCESS_LIMITS minWall) are at least Materialize's
  minimum wall for that process. Where loadpath keeps a stricter margin on
  purpose (SLS TPU walls 1.0 mm over Materialize's 0.8), the card says so
  and Materialize's figure is the hard limit;
- `review()` puts the reference into a run's report (report["materialize"]):
  the limits, the Materialize material id to quote or to publish with
  (metadata.recommendedMaterialId), whether the part fits the build volume,
  and a warning for a wall floor under the published minimum.

Materialize does not publish escape-hole sizes, hole sizes, bridge spans,
warp limits or clearances per process; those stay loadpath's own
(processcheck.RULES, spec.PROCESSES minHole, printcheck.BRIDGE_SPAN, fits).
Volumetric printing is not offered on Materialize, so it has no entry.

    from loadpath.dfm_reference import limits, review
    limits("sls", "tpu88a")["minWall"]        # 0.8, Materialize's hard limit
"""
from __future__ import annotations

__all__ = ["SOURCE", "PROCESSES", "MATERIALS", "limits", "material_entry", "review"]

SOURCE = {
    "name": "Materialize",
    "url": "https://www.materialize.cc",
    "how": "materialize_get_material: designLimits (minWallMm, minDetailMm), buildVolumeMm, "
           "properties.accuracyMm",
    "accessed": "2026-11-08",
}

# loadpath process -> Materialize's process label and its published minimum
# wall and minimum detail (mm). Every material Materialize lists for a
# process carried the same pair on the date above.
PROCESSES = {
    "fdm": {"materialize": "FDM", "minWall": 1.0, "minDetail": 0.8},
    "sls": {"materialize": "SLS", "minWall": 0.8, "minDetail": 0.3},
    "mjf": {"materialize": "MJF", "minWall": 0.8, "minDetail": 0.25},
    "sla": {"materialize": "SLA", "minWall": 0.6, "minDetail": 0.25},
    "dlp": {"materialize": "DLP", "minWall": 0.6, "minDetail": 0.25},
    "jetting": {"materialize": "PolyJet", "minWall": 1.0, "minDetail": 0.25},
    "lpbf": {"materialize": "SLM / DMLS", "minWall": 1.0, "minDetail": 0.25},
}


def _m(mid, name, build, accuracy, shore=None, note=None):
    out = {"id": mid, "name": name, "buildVolume": tuple(build), "accuracy": accuracy}
    if shore:
        out["shore"] = shore
    if note:
        out["note"] = note
    return out


# loadpath material card -> process -> the Materialize material it is
# ordered as. buildVolume (mm) is the largest any service offers for it;
# accuracy (+- mm) is Materialize's typical figure. A card with no entry for
# a process (vero, agilus30: no PolyJet; in625, 15-5ph: not listed) falls
# back to the process limits alone.
MATERIALS = {
    "pla": {"fdm": _m("b6b572b5-4960-564a-8685-39cc53935e18", "PLA", (1000, 1000, 1600), 0.3)},
    "petg": {"fdm": _m("0e379ece-53b6-5aa3-8cdf-7afce73c9873", "PETG", (1000, 1000, 1600), 0.3)},
    "abs": {"fdm": _m("8ce955b4-af00-5c90-a9cf-563a1b454ba2", "ABS", (1000, 1000, 1000), 0.3)},
    "asa": {"fdm": _m("14c9a468-7664-5953-bcdb-a55a7a1d8af5", "ASA", (1000, 1000, 1000), 0.3)},
    "tpu": {"fdm": _m("f797eb82-7d51-4d32-b82a-0e309332006b", "FDM TPU", (1000, 1000, 1600), 0.4,
                      shore="50A-70D in finishes; 95A is the Medium Hard (Shore A91-A100) finish")},
    "pa-cf": {"fdm": _m("426c1bec-69cb-4104-a9dc-5d6121e4a1b8", "Carbon Fiber FDM Nylon", (800, 800, 850), 0.4)},
    "pp": {"fdm": _m("16c182b4-fd3e-4afd-b680-883e0e136aca", "FDM Polypropylene", (601, 601, 601), 0.2)},
    "pa12": {"sls": _m("03a417a3-d9c9-5717-ab7f-3966a8ad4555", "SLS Nylon PA12", (700, 380, 580), 0.3),
             "mjf": _m("67f126d6-a917-5f87-8d26-b87161656539", "HP MJF Nylon PA12", (400, 400, 400), 0.3),
             "fdm": _m("956ff110-4204-40ea-9502-b1b665b821e9", "FDM Nylon", (1120, 720, 620), 0.3)},
    "pa11": {"sls": _m("099d270a-dc04-4efd-8911-28217b55923e", "SLS Nylon PA11", (440, 440, 450), 0.3),
             "mjf": _m("92ef57e1-133f-4678-a392-e460ee5f96ed", "HP MJF Nylon PA11", (400, 300, 400), 0.3)},
    "pa12-gb": {"mjf": _m("6ae232d0-d6ff-42f1-8cfe-eb3a741b6f8f", "HP 3D HR PA 12 GB", (380, 380, 380), 0.3),
                "sls": _m("5debf083-322b-42bb-a29e-38428ac89003", "SLS Glass-Filled Nylon PA12",
                          (520, 520, 830), 0.3)},
    "pa12-gf": {"sls": _m("5debf083-322b-42bb-a29e-38428ac89003", "SLS Glass-Filled Nylon PA12",
                          (520, 520, 830), 0.3)},
    "tpu88a": {"sls": _m("a8c7d8ca-0f22-5b9a-b1db-e5fd4f74881c", "SLS Flexible TPU", (500, 500, 400), 1.8,
                         shore="86-90A", note="no powder TPU softer than ~86A; 50-80A is FDM TPU only"),
               "mjf": _m("953fbba8-854b-4265-8ff3-6c8c20851da7", "MJF Flexible TPU", (450, 450, 450), 1.8,
                         shore="88-90A")},
    "resin": {k: _m("99a7feae-9577-5c71-8956-aa4946a21b6f", "Standard Resin", (3500, 3500, 5000), 0.2)
              for k in ("sla", "dlp")},
    "resin-tough": {k: _m("97f6c587-73ef-51c0-bb6a-aedf666ce59a", "Tough Resin", (1700, 1700, 1700), 0.2)
                    for k in ("sla", "dlp")},
    "resin-rigid": {k: _m("adaa639c-1d68-5814-8c9a-34366fddcb59", "Rigid Resin", (650, 650, 650), 0.2)
                    for k in ("sla", "dlp")},
    "alsi10mg": {"lpbf": _m("d0342e7b-db4e-5426-b972-ca02b2b71bed", "Aluminum (AlSi10Mg)", (600, 600, 600), 0.2)},
    "ti64": {"lpbf": _m("4ea9b3af-3e51-5c14-83f6-ae4512e7c762", "Titanium", (600, 600, 600), 0.2)},
    "316l": {"lpbf": _m("6e54ed5d-fb48-53d7-a7c4-a3381faebf6e", "316L Stainless Steel", (600, 600, 600), 0.3)},
}


def _name(x):
    if x is None:
        return None
    return str(getattr(x, "name", getattr(x, "kind", x))).strip().lower()


def material_entry(material, process) -> dict | None:
    """The Materialize material a loadpath card is ordered as for this
    process (id, name, buildVolume, accuracy, ...), or None."""
    return (MATERIALS.get(_name(material)) or {}).get(_name(process))


def limits(process, material=None) -> dict | None:
    """Materialize's published limits for a process (and material, when it
    is listed): minWall, minDetail (mm), and with a material its id, name,
    buildVolume (mm) and accuracy (+- mm). None for a process Materialize
    does not print (volumetric, milling, casting)."""
    kind = _name(process)
    if kind not in PROCESSES:
        return None
    out = {"reference": SOURCE["name"], "process": kind, **PROCESSES[kind]}
    entry = material_entry(material, kind)
    if entry is not None:
        out["material"] = dict(entry, buildVolume=list(entry["buildVolume"]))
    out["accessed"] = SOURCE["accessed"]
    return out


def _fits(extents, volume) -> bool:
    """Whether a box of these extents fits the build volume in some
    axis-aligned orientation (largest to largest)."""
    return all(e <= v + 1e-9 for e, v in zip(sorted(extents), sorted(volume)))


def review(process, material=None, extents=None, wall_floor=None, measured_wall=None) -> dict | None:
    """The Materialize reference for a run's report: its limits, the part
    against them, and warnings. `extents` (mm) checks the build volume;
    `wall_floor` is the thinnest wall the design allows (process minFeature,
    minMember, skin); `measured_wall` a measured wall (mm, e.g. the 1st
    percentile). None for a process Materialize does not print."""
    lim = limits(process, material)
    if lim is None:
        return None
    rep = dict(lim)
    warnings = []
    mat = lim.get("material")
    label = f"{lim['materialize']}" + (f" {mat['name']}" if mat else "")
    if mat is None and material is not None:
        rep["note"] = (f"{_name(material)} is not listed on Materialize for {lim['materialize']}; "
                       "process limits only (materialize_list_materials for the nearest)")
    if extents is not None and mat is not None:
        ext = [round(float(e), 1) for e in extents]
        rep["extents"] = ext
        rep["fitsBuildVolume"] = _fits(ext, mat["buildVolume"])
        if not rep["fitsBuildVolume"]:
            v = mat["buildVolume"]
            warnings.append(f"{ext[0]:g} x {ext[1]:g} x {ext[2]:g} mm is larger than any {label} build volume on "
                            f"Materialize ({v[0]:g} x {v[1]:g} x {v[2]:g} mm): split the part or pick another "
                            "material")
    for key, w in (("wallFloor", wall_floor), ("measuredWall", measured_wall)):
        if w is None:
            continue
        rep[key] = round(float(w), 3)
        if float(w) < lim["minWall"] - 1e-9:
            what = "the design allows walls down to" if key == "wallFloor" else "walls measure down to"
            warnings.append(f"{what} {float(w):.2f} mm, under Materialize's {lim['minWall']:g} mm minimum wall for "
                            f"{label}: services will flag or reject it")
    rep["warnings"] = warnings
    return rep
