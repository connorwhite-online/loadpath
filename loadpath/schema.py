"""
JSON Schemas for everything loadpath reads, and a linter that reports every
problem in a document at once, by path, with a suggestion where it can make
one.

    python -m loadpath schema study          # the study, as JSON Schema 2020-12
    python -m loadpath schema heatExchanger  # a heat exchanger job
    python -m loadpath schema lattice        # lattice infill options

The schemas are what an agent (or an editor) validates against before it
asks for a solve; `lint` is what `check` runs first, so a study with three
mistakes gets three messages instead of one per round trip. Unknown keys are
errors when they look like a misspelling of a known one (`keepout`,
`radius`) and warnings otherwise: a misspelt `keepOut` silently dropping
every keep-out is the worst way a study can go wrong.

The parsers (spec.py, heatx.py, lattice.py) stay the authority on meaning;
these schemas describe the same fields, and tests/test_schema.py checks every
example and eval case against them. Add a field here when you add it there.
"""
from __future__ import annotations

import copy
import difflib
import json
import re

__all__ = ["STUDY_SCHEMA", "HEAT_EXCHANGER_SCHEMA", "LATTICE_SCHEMA", "SCHEMAS", "schema", "lint", "format_issues"]

_DIALECT = "https://json-schema.org/draft/2020-12/schema"

_VEC3 = {"type": "array", "items": {"type": "number"}, "minItems": 3, "maxItems": 3,
         "description": "[x, y, z]"}
_POS = {"type": "number", "exclusiveMinimum": 0}
_AXIS = {"anyOf": [{"type": "string", "enum": ["x", "y", "z"]}, _VEC3],
         "description": "'x', 'y', 'z' or a direction vector"}
_SIGNED_AXIS = {"anyOf": [{"type": "string", "pattern": "^[+-]?[xyz]$", "description": "like '-x' or '+z'"}, _VEC3],
                "description": "'+x' ... '-z' or a direction vector"}

_SHAPES = {
    "box": {
        "type": "object",
        "description": "Axis-aligned box: min/max corners, or center/size.",
        "properties": {"type": {"const": "box"}, "min": _VEC3, "max": _VEC3, "center": _VEC3, "size": _VEC3},
        "required": ["type"],
        "x-requireOne": [["min", "max"], ["center", "size"]],
        "additionalProperties": False,
    },
    "cylinder": {
        "type": "object",
        "description": "Cylinder or tube: end points p0/p1, or center/axis/length; radius r, optional rInner.",
        "properties": {"type": {"enum": ["cylinder", "tube"]}, "p0": _VEC3, "p1": _VEC3, "center": _VEC3,
                       "axis": _AXIS, "length": _POS, "r": _POS,
                       "rInner": {"type": "number", "minimum": 0}},
        "required": ["type", "r"],
        "x-requireOne": [["p0", "p1"], ["center", "length"]],
        "additionalProperties": False,
    },
    "sphere": {
        "type": "object",
        "properties": {"type": {"const": "sphere"}, "center": _VEC3, "r": _POS},
        "required": ["type", "center", "r"],
        "additionalProperties": False,
    },
    "mesh": {
        "type": "object",
        "description": "A closed mesh: an STL/3MF/OBJ path relative to the study's folder, or inline stlBase64.",
        "properties": {"type": {"const": "mesh"}, "path": {"type": "string"}, "stlBase64": {"type": "string"}},
        "required": ["type"],
        "x-requireOne": [["path"], ["stlBase64"]],
        "additionalProperties": False,
    },
    "union": {
        "type": "object",
        "description": "Several shapes acting as one region (a servo's four screw bosses).",
        "properties": {"type": {"const": "union"},
                       "shapes": {"type": "array", "items": {"$ref": "#/$defs/shape"}, "minItems": 1}},
        "required": ["type", "shapes"],
        "additionalProperties": False,
    },
}
_SHAPE_TYPES = {"box": "box", "cylinder": "cylinder", "tube": "cylinder", "sphere": "sphere",
                "mesh": "mesh", "union": "union"}

_SHAPE = {
    "title": "a shape object (box, cylinder, tube, sphere, mesh, union)",
    "description": "A shape: box, cylinder (tube), sphere, mesh or union. Units mm.",
    "oneOf": [{"$ref": f"#/$defs/{k}"} for k in _SHAPES],
    "x-discriminator": {"property": "type", "mapping": {k: f"#/$defs/{v}" for k, v in _SHAPE_TYPES.items()}},
}

_REGION = {
    "type": "object",
    "properties": {"id": {"type": "string", "description": "Name loads and supports refer to."},
                   "shape": {"$ref": "#/$defs/shape"}},
    "required": ["shape"],
    "additionalProperties": False,
}

_TARGET = ("A 'region' (a preserve or keepOut id, or one a fastener adds such as 'w1-bore') "
           "or an inline 'shape'.")

_SUPPORT = {
    "type": "object",
    "description": "Where the part is held. " + _TARGET,
    "properties": {
        "region": {"type": "string"},
        "shape": {"$ref": "#/$defs/shape"},
        "fix": {"anyOf": [{"type": "string", "pattern": "^(all|pin|[xyz]{1,3})$",
                           "description": "'all', 'pin', or the axes held, like 'xz'"},
                          {"type": "array", "items": {"type": "boolean"}, "minItems": 3, "maxItems": 3}],
                "description": "'all' (default), 'pin' (a cylinder: turns freely), axes like 'xz', or 3 booleans"},
    },
    "x-requireOne": [["region"], ["shape"]],
    "additionalProperties": False,
}

_LOAD = {
    "type": "object",
    "description": "A force (N) and/or a moment (N·mm, about the region's centroid). " + _TARGET,
    "properties": {
        "region": {"type": "string"},
        "shape": {"$ref": "#/$defs/shape"},
        "force": _VEC3,
        "moment": _VEC3,
        "torque": {**_VEC3, "description": "Alias of moment."},
        "distribution": {"enum": ["uniform", "bearing"],
                         "description": "'bearing' spreads a force over the half of a bore it pushes on"},
    },
    "x-requireOne": [["region"], ["shape"]],
    "additionalProperties": False,
}

_LOAD_CASE = {
    "type": "object",
    "properties": {"name": {"type": "string"},
                   "loads": {"type": "array", "items": {"$ref": "#/$defs/load"}, "minItems": 1},
                   "weight": {"type": "number", "minimum": 0}},
    "required": ["loads"],
    "additionalProperties": False,
}

_MATERIAL_NAMES = ["pla", "petg", "abs", "asa", "pa-cf", "pa12", "pa11", "resin", "resin-tough",
                   "alsi10mg", "ti64", "316l"]
_CUSTOM_MATERIAL = {
    "type": "object",
    "description": "A custom material: E (MPa), nu, strength (MPa), density (g/cm³).",
    "properties": {"name": {"type": "string"}, "E": _POS, "nu": {"type": "number", "minimum": 0, "maximum": 0.4999},
                   "strength": _POS, "density": _POS, "note": {"type": "string"}},
    "required": ["E", "strength", "density"],
    "additionalProperties": False,
}
_ONE_MATERIAL = {"anyOf": [{"type": "string", "description": "A name: " + ", ".join(_MATERIAL_NAMES)
                            + " (aliases such as 'nylon', 'aluminium', 'steel' work)."},
                           _CUSTOM_MATERIAL]}

_FASTENER = {
    "type": "object",
    "description": "A bolted joint as an interface: bearing rings or a thread sleeve are preserved, the bore "
                   "and head path kept clear, and a preload case added. Adds regions <id>-bore, <id>-head, "
                   "<id>-seat, <id>-face or <id>-thread (and <id>-tap).",
    "properties": {
        "id": {"type": "string"},
        "size": {"enum": ["M2", "M2.5", "M3", "M4", "M5", "M6", "M8", "M10"]},
        "seat": {**_VEC3, "description": "Centre of the face the head (or the mating part) bears on."},
        "direction": {**_SIGNED_AXIS, "description": "From the seat into the part, the way the screw goes."},
        "grip": {**_POS, "description": "mm from the seat to the far end."},
        "end": {"enum": ["face", "thread", "none"]},
        "headOn": {"enum": ["this", "mating"]},
        "threadLength": _POS, "insert": _POS, "preload": {"type": "number", "minimum": 0},
        "hole": _POS, "head": _POS, "diameter": _POS, "wall": _POS, "ring": _POS, "sleeveWall": _POS,
        "driver": {"type": "number", "minimum": 0},
    },
    "required": ["seat", "direction", "grip"],
    "x-requireOne": [["size"], ["diameter", "head"], ["hole", "head"]],
    "additionalProperties": False,
}

_GOAL = {
    "type": "object",
    "description": "volumeFraction keeps that share of the design space; safetyFactor and/or maxDeflection "
                   "search for the lightest part that meets them in every load case.",
    "properties": {
        "volumeFraction": {"type": "number", "minimum": 0.02, "maximum": 0.95},
        "safetyFactor": _POS,
        "maxDeflection": {**_POS, "description": "mm, at the loads"},
        "method": {"enum": ["search", "constrained"]},
        "outcomes": {"type": "array", "items": {"type": "number", "minimum": 0.02, "maximum": 0.95},
                     "description": "Extra fixed volume fractions for a trade-off gallery."},
    },
    "x-requireOne": [["volumeFraction"], ["safetyFactor"], ["maxDeflection"]],
    "additionalProperties": False,
}

_PROCESS = {"anyOf": [
    {"enum": ["fdm", "sls", "mjf", "sla", "volumetric"]},
    {"type": "object",
     "properties": {"type": {"enum": ["fdm", "sls", "mjf", "sla", "volumetric"]},
                    "buildDirection": {"anyOf": [
                        {"type": "string", "pattern": "^([+-]?[xyz]|auto)$", "description": "like '+z', or 'auto'"},
                        _VEC3], "description": "the direction layers stack: '+z', a vector, or 'auto'"},
                    "stiffnessAcrossLayers": {"type": "number", "exclusiveMinimum": 0, "maximum": 1},
                    "shearAcrossLayers": {"type": "number", "exclusiveMinimum": 0, "maximum": 1},
                    "anisotropic": {"anyOf": [{"type": "boolean"}, {"const": "bond"}]},
                    "interlayer": {"type": "number", "exclusiveMinimum": 0, "maximum": 1},
                    "minFeature": _POS, "overhangs": {"type": "boolean"}},
     "required": ["type"], "additionalProperties": False},
], "description": "How it is made: sets a minimum member, overhang control (FDM) and the layer-bond check."}

_LATTICE_PROPS = {
    "type": {"enum": ["gyroid", "schwarz-p", "diamond"],
             "description": "TPMS sheet (default gyroid). Per gram, diamond is the stiffest and strongest, "
                            "Schwarz P the weakest and has flat ceilings (FDM)."},
    "cell": {**_POS, "description": "mm per TPMS period (at least 4 x maxWall)"},
    "skin": {"type": "number", "minimum": 0, "description": "mm of solid shell; 0 leaves the lattice open. "
                                                            "Under a skin, SLS, MJF and resin parts get drain holes"},
    "sizing": {"enum": ["homogenized", "solid-stress"],
               "description": "homogenized (default): the core is analyzed as the lattice it will be and each "
                              "cell sized to safetyFactor; solid-stress: graded from the solid part's stress "
                              "(quick, but not a sizing)"},
    "minWall": {**_POS, "description": "mm, where the part idles (at least the process's minFeature)"},
    "maxWall": {**_POS, "description": "mm; homogenized cells that would need more turn solid"},
    "solidAbove": {"type": "number", "exclusiveMinimum": 0, "maximum": 1,
                   "description": "(solid-stress) utilization (stress / allowable) above which material stays solid"},
    "safetyFactor": {**_POS, "description": "target safety factor of the lattice walls and skin "
                                            "(default: the study goal's, else 2)"},
    "collar": {"type": "number", "minimum": 0,
               "description": "mm of solid round preserved shapes, keep-outs, supports and loads (default 2)"},
    "blend": {**_POS, "description": "mm, fillet radius where the lattice meets solid (default maxWall)"},
    "drains": {"enum": ["auto", "yes", "no"],
               "description": "drain holes into enclosed voids: auto = for powder and resin processes"},
    "drainDiameter": {**_POS, "description": "mm (default 4 for SLS/MJF, 3 for resin)"},
    "fillPockets": {"type": "number", "minimum": 0,
                    "description": "mm^3: enclosed voids smaller than this are filled solid (default cell^3 / 8)"},
    "step": {**_POS, "description": "field sampling, mm (default minWall / 2.5)"},
}

STUDY_SCHEMA = {
    "$schema": _DIALECT,
    "$id": "https://github.com/connorwhite-online/loadpath/schema/study.schema.json",
    "title": "loadpath study",
    "description": "A generative design (topology optimization) problem. Units: mm, N, MPa, g/cm³.",
    "type": "object",
    "properties": {
        "name": {"type": "string", "description": "Used to name the output files."},
        "description": {"type": "string"},
        "designSpace": {"anyOf": [{"$ref": "#/$defs/shape"},
                                  {"type": "array", "items": {"$ref": "#/$defs/shape"}, "minItems": 1}],
                        "description": "Shapes material may occupy (their union)."},
        "preserve": {"type": "array", "items": {"$ref": "#/$defs/region"},
                     "description": "Geometry kept exactly: bosses, mounting faces."},
        "keepOut": {"type": "array", "items": {"$ref": "#/$defs/region"},
                    "description": "Volume that must stay empty: bores, bolt access, cables, mating parts."},
        "fasteners": {"type": "array", "items": {"$ref": "#/$defs/fastener"}},
        "supports": {"type": "array", "items": {"$ref": "#/$defs/support"}, "minItems": 1},
        "loadCases": {"type": "array", "items": {"$ref": "#/$defs/loadCase"}, "minItems": 1},
        "loads": {"type": "array", "items": {"$ref": "#/$defs/load"},
                  "description": "Shorthand for a single load case."},
        "material": {"anyOf": [{"$ref": "#/$defs/material"},
                               {"type": "array", "items": {"$ref": "#/$defs/material"}, "minItems": 1}],
                     "description": "A list solves the goal in each and pools the outcomes."},
        "goal": {"$ref": "#/$defs/goal"},
        "minMember": {**_POS, "description": "Thinnest member, mm."},
        "symmetry": {"type": "array", "items": {"enum": ["x", "y", "z"]},
                     "description": "Planes through the design-space centre, by normal axis."},
        "resolution": {"type": "object",
                       "properties": {"maxElements": {"type": "integer", "minimum": 1, "maximum": 400000},
                                      "pitch": _POS},
                       "additionalProperties": False},
        "maxIterations": {"type": "integer", "minimum": 1},
        "gravity": {**_VEC3, "description": "m/s², the part's own weight in every load case"},
        "printDirection": {"type": "string", "pattern": "^[+-]?[xyz]$",
                           "description": "The direction layers stack; turns on overhang control."},
        "extrusion": {"type": "string", "pattern": "^[+-]?[xyz]$",
                      "description": "The part is a profile extruded along this axis (cut from plate)."},
        "process": _PROCESS,
        "lattice": {"type": "object", "properties": _LATTICE_PROPS, "additionalProperties": False,
                    "description": "Optional: fill the delivered part with a stress-graded TPMS lattice."},
        "expect": {"type": "object", "description": "Eval-only pass criteria (loadpath.evals)."},
    },
    "required": ["designSpace", "supports"],
    "x-requireOne": [["loadCases"], ["loads"]],
    "additionalProperties": False,
    "$defs": {**_SHAPES, "shape": _SHAPE, "region": _REGION, "support": _SUPPORT, "load": _LOAD,
              "loadCase": _LOAD_CASE, "material": _ONE_MATERIAL, "fastener": _FASTENER, "goal": _GOAL},
}

_FLUID_NAMES = ["water", "water60", "air", "glycol50", "oil"]
_DUTY_SIDE = {
    "type": "object",
    "properties": {"fluid": {"enum": _FLUID_NAMES}, "lPerMin": _POS, "kgPerS": _POS,
                   "tIn": {"type": "number", "description": "°C"}},
    "x-requireOne": [["lPerMin"], ["kgPerS"]],
    "additionalProperties": False,
}

HEAT_EXCHANGER_SCHEMA = {
    "$schema": _DIALECT,
    "$id": "https://github.com/connorwhite-online/loadpath/schema/heat-exchanger.schema.json",
    "title": "loadpath heat exchanger",
    "description": "A two-fluid sheet-TPMS heat exchanger in any core shape, with headers and ports. "
                   "Units: mm; duty in L/min or kg/s and °C.",
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "description": {"type": "string"},
        "core": {"$ref": "#/$defs/shape"},
        "type": {"enum": ["gyroid", "diamond", "schwarz-p"]},
        "cell": {**_POS, "description": "mm per TPMS period (at least 3 x wall)"},
        "wall": {**_POS, "description": "mm, the sheet between the fluids"},
        "shell": _POS, "header": _POS, "cap": _POS, "portDiameter": _POS, "portLength": _POS,
        "layout": {"enum": ["cross", "counter"]},
        "axes": {"type": "string", "pattern": "^[xyz]{2}$",
                 "description": "cross: A along axes[0], B along axes[1]; counter: A through the ends along "
                                "axes[0], B's side ports along axes[1]"},
        "fluids": {"type": "array", "items": {"type": "string"}, "minItems": 2, "maxItems": 2},
        "build": {"anyOf": [{"enum": ["x", "y", "z", "none"]}, {"type": "null"}, {"const": False}],
                  "description": "The axis pointing up while printing (null: no print shaping)."},
        "skirt": {"type": "boolean"},
        "zones": {"type": "array", "items": {
            "type": "object",
            "properties": {"fluid": {"anyOf": [{"type": "string"}, {"type": "integer"}]},
                           "end": {"enum": ["in", "out"]}, "direction": _SIGNED_AXIS,
                           "minCos": {"type": "number", "minimum": -1, "maximum": 1},
                           "window": {"type": "object",
                                      "properties": {"axis": _SIGNED_AXIS, "from": {"type": "number"},
                                                     "to": {"type": "number"}},
                                      "required": ["axis"], "additionalProperties": False}},
            "required": ["fluid", "end", "direction"], "additionalProperties": False}},
        "step": {**_POS, "description": "grid, mm (default wall / 3); coarser is faster for a first look"},
        "duty": {"type": "object", "description": "Optional: rate it. One entry per fluid name, plus 'wall' "
                                                  "(a material name or W/m·K).",
                 "properties": {"wall": {"anyOf": [{"type": "string"}, _POS]}},
                 "additionalProperties": {"$ref": "#/$defs/dutySide"}},
    },
    "required": ["core"],
    "additionalProperties": False,
    "$defs": {**_SHAPES, "shape": _SHAPE, "dutySide": _DUTY_SIDE},
}

LATTICE_SCHEMA = {
    "$schema": _DIALECT,
    "$id": "https://github.com/connorwhite-online/loadpath/schema/lattice.schema.json",
    "title": "loadpath lattice infill",
    "description": "Fill a finished part with a TPMS sheet lattice whose wall follows the stress.",
    "type": "object",
    "properties": _LATTICE_PROPS,
    "additionalProperties": False,
}

SCHEMAS = {"study": STUDY_SCHEMA, "heatExchanger": HEAT_EXCHANGER_SCHEMA, "lattice": LATTICE_SCHEMA}


def schema(kind: str = "study") -> dict:
    """A copy of one schema: 'study', 'heatExchanger' or 'lattice'."""
    if kind not in SCHEMAS:
        raise ValueError(f"schema kind must be one of {sorted(SCHEMAS)}")
    return copy.deepcopy(SCHEMAS[kind])


# --- linting: a small validator for the subset of JSON Schema used above ---

_TYPES = {"object": dict, "array": list, "string": str, "boolean": bool, "null": type(None)}


def _is(v, t):
    if t == "number":
        return isinstance(v, (int, float)) and not isinstance(v, bool)
    if t == "integer":
        return isinstance(v, int) and not isinstance(v, bool) or (isinstance(v, float) and v.is_integer())
    return isinstance(v, _TYPES[t])


def _resolve(node, root):
    while "$ref" in node:
        node = root["$defs"][node["$ref"].rsplit("/", 1)[1]]
    return node


def _brief(node, root):
    node = _resolve(node, root)
    if "title" in node:
        return node["title"]
    if "enum" in node:
        return "one of " + ", ".join(json.dumps(v) for v in node["enum"])
    if "const" in node:
        return json.dumps(node["const"])
    if node.get("type") == "array" and node.get("minItems") == node.get("maxItems") == 3:
        return "[x, y, z]"
    if "description" in node:
        return node["description"]
    return node.get("type", "a valid value")


def _check(v, node, root, path, out):
    node = _resolve(node, root)
    if "x-discriminator" in node:
        disc = node["x-discriminator"]
        key = disc["property"]
        if not isinstance(v, dict) or key not in v:
            out.append(("error", path, f"must be an object with a '{key}' ({', '.join(disc['mapping'])})"))
            return
        target = disc["mapping"].get(v[key])
        if target is None:
            near = [_ALIASES[v[key]][0]] if v[key] in _ALIASES else \
                difflib.get_close_matches(str(v[key]), list(disc["mapping"]), 1)
            hint = f"; did you mean {near[0]!r}?" if near else ""
            out.append(("error", f"{path}.{key}", f"unknown {key} {v[key]!r}: one of "
                        f"{', '.join(disc['mapping'])}{hint}"))
            return
        _check(v, {"$ref": target}, root, path, out)
        return
    for kw in ("anyOf", "oneOf"):
        if kw in node:
            trials = []
            for alt in node[kw]:
                sub = []
                _check(v, alt, root, path, sub)
                if not any(s[0] == "error" for s in sub):
                    out.extend(sub)
                    return
                trials.append((alt, sub))
            # report against the alternative of the right type, else list them
            typed = [(a, s) for a, s in trials if _type_ok(v, _resolve(a, root))]
            if len(typed) == 1:
                out.extend(typed[0][1])
            else:
                out.append(("error", path, "must be " + " or ".join(_brief(a, root) for a, _ in trials)
                            + f", got {json.dumps(v)[:60]}"))
            return
    if not _type_ok(v, node):
        out.append(("error", path, f"must be {_brief(node, root)}, got {json.dumps(v)[:60]}"))
        return
    if "const" in node and v != node["const"]:
        out.append(("error", path, f"must be {json.dumps(node['const'])}"))
    if "enum" in node and v not in node["enum"]:
        near = difflib.get_close_matches(str(v), [str(e) for e in node["enum"]], 1)
        hint = f"; did you mean {near[0]!r}?" if near else ""
        out.append(("error", path, f"must be {_brief(node, root)}, got {json.dumps(v)}{hint}"))
    if "pattern" in node and isinstance(v, str) and not re.fullmatch(node["pattern"], v):
        out.append(("error", path, f"{v!r} is not valid: {node.get('description', node['pattern'])}"))
    if _is(v, "number"):
        if "minimum" in node and v < node["minimum"]:
            out.append(("error", path, f"must be at least {node['minimum']}, got {v}"))
        if "maximum" in node and v > node["maximum"]:
            out.append(("error", path, f"must be at most {node['maximum']}, got {v}"))
        if "exclusiveMinimum" in node and v <= node["exclusiveMinimum"]:
            out.append(("error", path, f"must be greater than {node['exclusiveMinimum']}, got {v}"))
    if isinstance(v, list):
        if "minItems" in node and len(v) < node["minItems"]:
            out.append(("error", path, f"needs at least {node['minItems']} item(s)"))
        if "maxItems" in node and len(v) > node["maxItems"]:
            out.append(("error", path, f"takes at most {node['maxItems']} items"))
        if "items" in node:
            for i, x in enumerate(v):
                _check(x, node["items"], root, f"{path}[{i}]", out)
    if isinstance(v, dict):
        props = node.get("properties", {})
        for k in node.get("required", []):
            if k not in v:
                out.append(("error", path, f"missing '{k}' ({_brief(props.get(k, {}), root)})"))
        groups = node.get("x-requireOne")
        if groups and not any(all(k in v for k in g) for g in groups):
            out.append(("error", path, "needs " + " or ".join("+".join(g) for g in groups)))
        for k, x in v.items():
            if k in props:
                _check(x, props[k], root, f"{path}.{k}", out)
            elif isinstance(node.get("additionalProperties"), dict):
                _check(x, node["additionalProperties"], root, f"{path}.{k}", out)
            elif node.get("additionalProperties") is False:
                near = [p for p in _ALIASES.get(k, []) if p in props] or \
                    [p for p in props if p.lower() == k.lower()] or difflib.get_close_matches(k, list(props), 1, 0.6)
                if near:
                    out.append(("error", f"{path}.{k}", f"unknown key; did you mean '{near[0]}'?"))
                else:
                    out.append(("warning", f"{path}.{k}", "unknown key, ignored (known: "
                                + ", ".join(props) + ")"))


# common words for a field that difflib would not match
_ALIASES = {"radius": ["r"], "diameter": ["r"], "innerRadius": ["rInner"], "bolts": ["fasteners"],
            "screws": ["fasteners"], "holes": ["keepOut"], "fixed": ["supports"], "constraints": ["supports"],
            "forces": ["loads"], "torque": ["moment"], "domain": ["designSpace"], "bounds": ["designSpace"],
            # shape types
            "cube": ["box"], "cuboid": ["box"], "rect": ["box"], "cyl": ["cylinder"], "rod": ["cylinder"],
            "pipe": ["tube"], "ball": ["sphere"], "stl": ["mesh"], "group": ["union"]}


def _type_ok(v, node):
    if "x-discriminator" in node:
        return isinstance(v, dict)
    t = node.get("type")
    if t is None:
        if "const" in node:
            return v == node["const"] or type(v) is type(node["const"])
        if "enum" in node:
            return any(type(v) is type(e) for e in node["enum"])
        return True
    return any(_is(v, x) for x in (t if isinstance(t, list) else [t]))


def lint(doc, kind: str = "study") -> list[dict]:
    """Every problem in a document against its schema, as
    [{"level": "error" | "warning", "path": "$.keepOut[0].shape", "message"}]."""
    root = SCHEMAS[kind]
    if isinstance(doc, str):
        try:
            doc = json.loads(doc)
        except json.JSONDecodeError as exc:
            return [{"level": "error", "path": "$", "message": f"not valid JSON: {exc}"}]
    out = []
    _check(doc, root, root, "$", out)
    return [{"level": lv, "path": p, "message": m} for lv, p, m in out]


def format_issues(issues: list[dict]) -> str:
    return "\n".join(f"{i['level']}: {i['path']}: {i['message']}" for i in issues)
