"""
A results bundle (bundle.py) as a Materialize listing: what the part page
can carry today, read from manifest.json. This builds the payload only; it
uploads and publishes nothing. An agent with the Materialize MCP server
passes `upload` to materialize_request_upload_url / materialize_register_upload
and `metadata` as register_upload's (or materialize_update_file's)
metadata, after the owner says to publish.

    from loadpath.adapters.materialize import listing
    plan = listing("out/bundle")

Materialize file metadata takes a name, a description (5000 characters),
design tags, a minimum wall, and a recommended material; models are STL,
OBJ, 3MF, STEP or AMF. The fields themselves (part.glb's safety-factor map,
analysis.vtu, the manifest) have nowhere to go yet: `notCarried` lists them,
and `simulation` is the compact record a part page would show if it took
one.
"""
from __future__ import annotations

import json
import os

__all__ = ["listing"]

_MAX_DESCRIPTION = 5000


def _load(src):
    if isinstance(src, dict):
        return src, None
    path = os.path.join(src, "manifest.json") if os.path.isdir(src) else src
    with open(path) as fh:
        return json.load(fh), os.path.dirname(os.path.abspath(path))


def _f(v, nd=2):
    return "n/a" if v is None else f"{v:.{nd}f}"


def listing(bundle, name: str | None = None) -> dict:
    """{upload, metadata, simulation, notCarried} for the bundle at
    `bundle` (its folder, its manifest.json, or the manifest itself)."""
    from ..dfm_reference import material_entry

    man, folder = _load(bundle)
    part, verdict = man["part"], man["verdict"]
    proc = (part.get("process") or {}).get("kind")
    entry = material_entry(part["material"]["name"], proc) if proc else None
    fields = {f["id"]: f for f in man["fields"]}
    gov = fields.get("safetyFactor/governing") or {}
    lines = []
    sf, tgt = verdict.get("safetyFactor"), verdict.get("target")
    cases = part.get("loadCases") or []
    lines.append(f"Designed and verified with loadpath for {len(cases)} load case"
                 f"{'s' if len(cases) != 1 else ''}, in {part['material']['name']}.")
    if sf is not None:
        lines.append(f"Safety factor {_f(sf)}" + (f" against a target of {tgt:g}" if tgt else "")
                     + (f" ({gov['criterion'].split(';')[0]})" if gov.get("criterion") else "") + ".")
    if verdict.get("mass") is not None:
        lines.append(f"Mass {verdict['mass']:.1f} g"
                     + (f", {100 * verdict['massReduction']:.0f}% lighter than the solid design space"
                        if verdict.get("massReduction") else "") + ".")
    if verdict.get("bucklingFactor") is not None:
        lines.append(f"Buckles at {_f(verdict['bucklingFactor'])}x its loads (linear buckling).")
    loads = [o for o in man["objects"] if o["role"] in ("force", "moment")]
    if loads:
        lines.append("Loads: " + "; ".join(f"{o['case']}: {o['magnitude']:.4g} {o['units']} {o['role']}"
                                           for o in loads[:12]) + ".")
    weak = next((o for o in man["objects"] if o["role"] == "weakest point"), None)
    if weak:
        lines.append(f"Weakest at ({', '.join(f'{v:.1f}' for v in weak['at'])}) mm in case {weak['case']!r}.")
    if verdict.get("passes") is False:
        lines.append("Does not meet its goal as delivered.")
    desc = " ".join(lines)[:_MAX_DESCRIPTION]
    tags = []
    if verdict.get("passes") and sf is not None and tgt and sf >= tgt:
        tags.append("strong")
    if (verdict.get("massReduction") or 0) >= 0.5:
        tags.append("lightweight")
    if fields.get("temperature") is not None and verdict.get("passes"):
        tags.append("heat-resistant")
    meta = {"name": name or part["name"], "description": desc, "designTags": tags}
    if verdict.get("p5Member"):
        meta["minWallThicknessMm"] = round(float(verdict["p5Member"]), 2)
    if entry:
        meta["recommendedMaterialId"] = entry["id"]
    stl = next((f["path"] for f in man["files"] if f["path"].endswith(".stl")), None)
    upload = {"file": os.path.join(folder, stl) if folder and stl else stl, "format": "stl", "fileUnit": "mm"}
    simulation = {"bundle": man["loadpath"]["bundle"], "verdict": verdict,
                  "fields": [{k: f.get(k) for k in ("id", "quantity", "units", "target", "range", "criterion")}
                             for f in man["fields"]],
                  "colorScale": man.get("colorScale")}
    not_carried = [f["path"] for f in man["files"] if not f["path"].endswith(".stl")]
    return {"upload": upload, "metadata": meta, "simulation": simulation, "notCarried": not_carried,
            "note": "nothing is uploaded or published here; publish only with the owner's go-ahead"}
