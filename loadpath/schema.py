"""
JSON Schemas for everything loadpath reads, and a linter that reports every
problem in a document at once, by path, with a suggestion where it can make
one.

    python -m loadpath schema study          # the study, as JSON Schema 2020-12
    python -m loadpath schema heatExchanger  # a heat exchanger job
    python -m loadpath schema lattice        # lattice infill options
    python -m loadpath schema field          # a field graph (fields.py)

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

from .material_library import LIBRARY as _LIBRARY

__all__ = ["STUDY_SCHEMA", "HEAT_EXCHANGER_SCHEMA", "LATTICE_SCHEMA", "FIELD_SCHEMA", "SCHEMAS", "schema", "lint", "format_issues"]

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
        "description": "A closed mesh: an STL/3MF/OBJ path relative to the study's folder (or a STEP file, "
                       ".step/.stp, tessellated and healed; needs loadpath[step]), or inline stlBase64.",
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

_KEEP_OUT = {
    "type": "object",
    "properties": {**_REGION["properties"],
                   "bore": {"type": "boolean",
                            "description": "false: a clearance volume (a mating part, a sweep) that may end "
                                           "against material; it is not checked for closed ends. Default true "
                                           "(false for a pressurized cavity)."},
                   "wall": {**_POS, "description": "mm of material kept round it, pinned solid: a duct's or a "
                                                   "plenum's skin, with only what is behind it designed (ribs). "
                                                   "A pressurized keep-out gets minMember if not set."}},
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
    "description": "A force (N) and/or a moment (N·mm, about the region's centroid), or a pressure (MPa) "
                   "in a keepOut. " + _TARGET,
    "properties": {
        "region": {"type": "string"},
        "shape": {"$ref": "#/$defs/shape"},
        "force": _VEC3,
        "moment": _VEC3,
        "torque": {**_VEC3, "description": "Alias of moment."},
        "distribution": {"enum": ["uniform", "bearing"],
                         "description": "'bearing' spreads a force over the half of a bore it pushes on"},
        "pressure": {"type": "number",
                     "description": "MPa gauge (1 bar = 0.1 MPa) inside the keepOut named by 'region' (the fluid's "
                                    "volume), on every wall lining it; negative is a vacuum. A study whose only "
                                    "loads are pressures on sealed cavities needs no supports."},

        "direction": {"enum": ["any"],
                      "description": "'any': the force's size may point in any direction (contact, bumps, "
                                     "handling); the part is designed and checked for the worst one, found "
                                     "exactly from three unit-force solves. At most one per load case; a bearing "
                                     "load's uncertain force is spread evenly over the region."},
        "cone": {"type": "number", "exclusiveMinimum": 0, "maximum": 180,
                 "description": "Degrees: the force may tilt up to this far from its written direction, and the "
                                "worst tilt is designed for (180 is the same as direction 'any')."},
    },
    "x-requireOne": [["region"], ["shape"]],
    "additionalProperties": False,
}

_LOAD_CASE = {
    "type": "object",
    "properties": {"name": {"type": "string"},
                   "loads": {"type": "array", "items": {"$ref": "#/$defs/load"}, "minItems": 1},
                   "weight": {"type": "number", "minimum": 0},
                   "cycles": {**_POS, "description": "Times this load is applied over the part's life: adds a "
                                                     "fatigue check (e.g. 1e6 for a servo-driven arm)."},
                   "hours": {**_POS, "description": "Hours this load is held over the part's life: adds creep "
                                                    "rupture and crept deflection."},
                   "temperature": {"type": "number",
                                   "description": "C while this load acts (default durability.temperature)."}},
    "required": ["loads"],
    "additionalProperties": False,
}

_MATERIAL_NAMES = sorted(_LIBRARY)
_CUSTOM_MATERIAL = {
    "type": "object",
    "description": "A custom material: E (MPa), nu, strength (MPa), density (g/cm³); or \"base\": a library "
                   "name and only the values that differ. EZ and strengthZ are across-layer values, used when "
                   "the study's process is the one named in \"measured\".",
    "properties": {"name": {"type": "string"}, "base": {"type": "string"},
                   "E": _POS, "nu": {"type": "number", "minimum": 0, "maximum": 0.4999},
                   "strength": _POS, "density": _POS, "EZ": _POS, "strengthZ": _POS,
                   "measured": {"type": "string"}, "note": {"type": "string"}},
    "x-requireOne": [["E", "strength", "density"], ["base"]],
    "additionalProperties": False,
}
_ONE_MATERIAL = {"anyOf": [{"type": "string", "description": "A name: " + ", ".join(_MATERIAL_NAMES)
                            + " (aliases such as 'nylon', 'aluminium', 'steel' work; every value is cited in "
                            "material_library.py, `loadpath materials`), or a calibrated card measured on a "
                            "printer (loadpath/calibrations/, calibration.py), which replaces a library card "
                            "of the same name."},
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
        "confirm": {"enum": ["auto", "voxel", "tet"],
                    "description": "How the lightest passing part is confirmed: voxel (a finer voxel grid), tet "
                                   "(a body-fitted tetrahedral mesh, which also calibrates the search's voxel "
                                   "stress so it steps down to the real target), or auto (tet when installed and "
                                   "applicable)."},
        "outcomes": {"type": "array", "items": {"type": "number", "minimum": 0.02, "maximum": 0.95},
                     "description": "Extra fixed volume fractions for a trade-off gallery."},
        "minimize": {"enum": ["mass", "printTime", "filament"],
                     "description": "Which passing outcome is delivered: the lightest (default), or (FDM, with a "
                                    "slicer installed) the one that prints fastest or with the least filament, "
                                    "support included, as sliced (graded, with slicer.grade)."},
    },
    "x-requireOne": [["volumeFraction"], ["safetyFactor"], ["maxDeflection"]],
    "additionalProperties": False,
}

_DURABILITY = {
    "type": "object",
    "description": "Fatigue (load cases with cycles), creep (with hours) and bolt-preload relaxation, reported "
                   "under 'durability' in verification. Estimates from published printed-polymer data; see "
                   "loadpath/durability.py for sources and uncertainty.",
    "properties": {
        "temperature": {"type": "number", "description": "Service temperature, C (default 23)."},
        "safetyFactor": {**_POS, "description": "Target fatigue/creep safety factor, on the loads (default 1.5; "
                                                "the fatigue curve is already a >90% survival design curve)."},
        "constrain": {"type": "boolean", "description": "true: the safety-factor search also sizes the part for "
                                                        "fatigue and creep (default false: report only)."},
        "preloadHours": {**_POS, "description": "Hours fasteners hold their preload, for relaxation (default 8760)."},
        "fatigueStrength": {**_POS, "description": "Override: max stress survived for 2e6 cycles, as a share of "
                                                   "the material's strength."},
        "fatigueSlope": {**_POS, "description": "Override: S-N inverse slope k (life ~ stress^-k)."},
        "creepRatio": {"type": "number", "minimum": 0,
                       "description": "Override: creep strain / elastic strain after 1000 h at 23 C."},
        "creepExponent": {**_POS, "description": "Override: n in creep ~ time^n."},
    },
    "additionalProperties": False,
}

_THERMAL_REGION = {
    "region": {"type": "string"},
    "shape": {"$ref": "#/$defs/shape"},
}

_THERMAL = {
    "type": "object",
    "description": "Heat in the part, applied in every load case: the temperature is solved on the part itself "
                   "(conduction through the material, convection off its surface), each cell softens by the "
                   "material's modulus retention at its temperature, and expansion that the supports or the "
                   "part's own cooler material stop becomes stress. The optimizer designs for it and "
                   "verification checks it (stress with the free expansion taken off, strength derated as the "
                   "modulus is). Needs a way for heat out: a fixed temperature or convection.",
    "properties": {
        "ambient": {"type": "number", "description": "C of the surroundings (default 25)."},
        "stressFree": {"type": "number",
                       "description": "C at which the part fits without stress, e.g. when bolted up (default "
                                      "ambient)."},
        "temperatures": {"type": "array", "description": "Surfaces held at a temperature: a motor face, a "
                                                        "heated bed, a cold plate. " + _TARGET,
                         "items": {"type": "object", "properties": {**_THERMAL_REGION, "t": {
                             "type": "number", "description": "C"}},
                                   "required": ["t"], "x-requireOne": [["region"], ["shape"]],
                                   "additionalProperties": False}},
        "heat": {"type": "array", "description": "Heat flowing in. " + _TARGET,
                 "items": {"type": "object", "properties": {
                     **_THERMAL_REGION,
                     "watts": {**_POS, "description": "W into the region"},
                     "spread": {"enum": ["surface", "volume"],
                                "description": "'surface' (default): through the region's surface nodes; "
                                               "'volume': generated in the region's cells (a coil, a "
                                               "resistor potted in)."}},
                     "required": ["watts"], "x-requireOne": [["region"], ["shape"]],
                     "additionalProperties": False}},
        "convection": {"anyOf": [
            {"type": "number", "minimum": 0, "description": "W/(m² K) off every exposed surface"},
            {"type": "array", "items": {"type": "object", "properties": {
                **_THERMAL_REGION,
                "h": {"type": "number", "minimum": 0,
                      "description": "W/(m² K): ~5-10 still air, 20-50 a fan, 500+ water"},
                "fluid": {"type": "number", "description": "C of the fluid (default ambient)"}},
                "required": ["h"], "additionalProperties": False}}],
            "description": "Heat carried off the surface the design exposes (default 8 W/(m² K), still air). "
                           "An entry without region or shape covers every surface."},
        "conductivity": {**_POS, "description": "W/(m K) override (default the material card's)."},
        "expansion": {"anyOf": [{"type": "boolean"}, {"type": "number", "minimum": 0}],
                      "description": "true (default): the material card's expansion coefficient; a number: 1/K "
                                     "(e.g. 7e-5); false: temperature softens the part but does not grow it."},
        "soften": {"type": "boolean",
                   "description": "true (default): moduli and strength fall with temperature (the material "
                                  "card's retention curve)."},
    },
    "additionalProperties": False,
}

_GRADE_LEVEL = {"anyOf": [
    {"type": "array", "prefixItems": [{"type": "number", "exclusiveMinimum": 0, "exclusiveMaximum": 100},
                                      {"type": "integer", "minimum": 1}], "minItems": 2, "maxItems": 2,
     "description": "[infill %, perimeters]"},
    {"type": "object", "properties": {"infill": {"type": "number", "exclusiveMinimum": 0, "exclusiveMaximum": 100},
                                      "perimeters": {"type": "integer", "minimum": 1}},
     "required": ["infill"], "additionalProperties": False}]}

_SLICER = {"anyOf": [
    {"type": "boolean"},
    {"type": "object",
     "properties": {
         "enabled": {"anyOf": [{"type": "boolean"}, {"const": "auto"}],
                     "description": "auto (default): slice when a slicer is installed; true: warn when none is"},
         "executable": {"type": "string", "description": "PrusaSlicer (or SuperSlicer) executable (default: "
                                                          "$LOADPATH_SLICER, else found on PATH)"},
         "profile": {"type": "string", "description": "PrusaSlicer config .ini (File > Export > Export Config) for "
                                                       "your printer's real speeds; relative to the study file"},
         "settings": {"type": "object", "description": "PrusaSlicer config keys over the profile, like "
                                                        "{\"layer_height\": 0.25}"},
         "supports": {"anyOf": [{"type": "boolean"}, {"const": "auto"}],
                      "description": "auto/true (default): the slicer's automatic supports; false: none"},
         "infill": {"type": "number", "minimum": 0, "maximum": 100,
                    "description": "% infill of the plain slice (default 100: the part is verified solid)"},
         "timeout": _POS,
         "grade": {"anyOf": [{"type": "boolean"}, {"type": "object", "properties": {
             "levels": {"type": "array", "items": _GRADE_LEVEL, "minItems": 1,
                        "description": "infill ladder, sparsest first; solid is always the level above "
                                       "(default [[10,3],[15,3],[25,3],[40,3],[60,3]]: walls that change per region make the slicer run perimeters round each region, inside the part)"},
             "safetyFactor": {**_POS, "description": "default: the goal's, else 2"},
             "lineWidth": {**_POS, "description": "mm per perimeter (default 0.45)"},
             "collar": {"type": "number", "minimum": 0,
                        "description": "mm kept solid round preserves, keep-outs, loads, supports (default 2)"},
             "knockdown": {"type": "number", "exclusiveMinimum": 0, "maximum": 1,
                           "description": "infill wall strength / solid (default 0.8)"},
             "analysisCells": {"type": "integer", "minimum": 1000}},
             "additionalProperties": False}],
             "description": "Size per-region infill density and perimeters from the stress, verified on a "
                            "homogenized model, and write them as slicer modifiers in a graded 3MF"}},
     "additionalProperties": False}],
    "description": "FDM: slice each outcome headless (PrusaSlicer) for print time, filament and support "
                   "(report 'slicer'); optionally grade infill by stress (report 'graded', <name>-graded.3mf)."}

_BUILD_SIM = {"anyOf": [
    {"type": "boolean"},
    {"type": "object", "properties": {
        "layer": {**_POS, "description": "mm per analysis layer (lumped real layers; default one voxel)"},
        "plate": {"type": "number", "minimum": 0, "description": "lpbf: mm of build plate modelled under the part "
                                                                   "(default 10; 0 = rigid)"},
        "scale": {**_POS, "description": "multiplies the inherent strain (a calibration for your machine)"},
        "strain": {**_VEC3, "description": "inherent strain (exx, eyy, ezz) in the build frame, overriding "
                                           "the material model (negative = shrink)"},
        "release": {"enum": ["full", "none"],
                    "description": "full: cut off the plate / peeled off the bed and free (default); none: as built"},
        "bedTemperature": {"type": "number", "description": "fdm: C (default per material)"},
        "chamberTemperature": {"type": "number", "description": "fdm: C, air round the part while printing"},
        "bedStiffness": {"type": "number", "minimum": 0,
                         "description": "fdm: N/mm^3 the bed holds the first layer with (default: the value fitted "
                                        "to Armillotta 2018; omit for that)"},
        "relax": {"type": "number", "exclusiveMinimum": 0, "maximum": 1,
                  "description": "fdm: share of the shrink below Tg kept as stress"},
        "recoaterGap": {**_POS, "description": "lpbf: mm of lift at the build plane that counts as recoater "
                                               "contact risk (default 0.05)"},
        "compensate": {"type": "boolean", "description": "also write <name>-compensated.stl, pre-distorted so "
                                                         "the print lands on the design"},
        "iterations": {"type": "integer", "minimum": 1, "description": "compensation rounds (default 2)"},
        "cells": {"type": "integer", "minimum": 1000, "description": "voxels in the simulated part (default 60000)"}},
     "additionalProperties": False}],
    "description": "Simulate the build of the delivered part (report 'buildSim'): layer-by-layer inherent-strain "
                   "residual stress and distortion for lpbf (validated on NIST AM-Bench 2018), fdm warp and bed "
                   "peel (ABS/ASA/PETG...), sls/mjf differential shrink; optionally a pre-distorted STL."}

_PROCESS_KINDS = ["fdm", "sls", "mjf", "sla", "dlp", "volumetric", "jetting", "lpbf"]
_PROCESS = {"anyOf": [
    {"enum": _PROCESS_KINDS},
    {"type": "object",
     "properties": {"type": {"enum": _PROCESS_KINDS},
                    "buildDirection": {"anyOf": [
                        {"type": "string", "pattern": "^([+-]?[xyz]|auto)$", "description": "like '+z', or 'auto'"},
                        _VEC3], "description": "the direction layers stack: '+z', a vector, or 'auto'"},
                    "stiffnessAcrossLayers": {"type": "number", "exclusiveMinimum": 0, "maximum": 1},
                    "shearAcrossLayers": {"type": "number", "exclusiveMinimum": 0, "maximum": 1},
                    "anisotropic": {"anyOf": [{"type": "boolean"}, {"const": "bond"}]},
                    "interlayer": {"type": "number", "exclusiveMinimum": 0, "maximum": 1},
                    "minFeature": _POS, "overhangs": {"type": "boolean"},
                    "minHole": {**_POS, "description": "mm: narrowest hole or gap the optimizer may leave "
                                                       "(default FDM/SLS/MJF 2, SLA/DLP/LPBF 1, volumetric 0.6)"},
                    "escapeHole": {**_POS, "description": "mm: smallest powder escape / resin drain hole "
                                                          "(default SLS 4, MJF 5, SLA/DLP 3.5)"}},
     "required": ["type"], "additionalProperties": False},
], "description": "How it is made: sets a minimum member, overhang control (FDM, LPBF metal) and the layer-bond check. "
                  "lpbf = laser powder-bed fusion metal (AlSi10Mg, Ti64, 316L). sls/mjf (PA12, PA11, PA12-GB, TPU) "
                  "and sla/dlp (resins) get process checks on the delivered part (report 'process': walls, trapped "
                  "powder or resin, warp-prone panels; resin supports, peel area and suction cups)."}

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
    "drainDiameter": {**_POS, "description": "mm (default the process's escape hole: SLS 4, MJF 5, SLA/DLP 3.5, "
                                            "volumetric 3)"},
    "fillPockets": {"type": "number", "minimum": 0,
                    "description": "mm^3: enclosed voids smaller than this are filled solid (default cell^3 / 8)"},
    "step": {**_POS, "description": "field sampling, mm (default minWall / 2.5)"},
}

_FRACTIONS = {"type": "object", "additionalProperties": {"type": "number", "exclusiveMinimum": 0, "maximum": 1},
              "description": "material name -> fraction of the part's volume"}
_MULTI_MATERIAL = {
    "type": "object",
    "description": "Multi-material topology optimization (multimat.py): the optimizer places each material as "
                   "well as the topology, for multi-material FDM (Bambu AMS, Prusa MMU/XL, IDEX) or jetting. "
                   "Delivers one body per material with shared interfaces, an STL each and a multi-material 3MF.",
    "properties": {
        "materials": {"type": "array", "items": {"$ref": "#/$defs/material"}, "minItems": 2, "maxItems": 4,
                      "description": "The materials to place (library names or custom objects)."},
        "assign": {"type": "array", "items": {"type": "object", "properties": {
                       "material": {"type": "string"},
                       "region": {"type": "string", "description": "a preserve or keepOut id"},
                       "shape": {"$ref": "#/$defs/shape"},
                       "near": {"type": "object", "properties": {
                           "region": {"type": "string"}, "shape": {"$ref": "#/$defs/shape"},
                           "distance": {**_POS, "description": "mm from the region's surface"}},
                           "required": ["distance"], "additionalProperties": False},
                       "within": {"type": "string", "description": "near only: limit to inside this region"}},
                   "required": ["material"], "additionalProperties": False},
                   "description": "Cells the study fixes to one material (a pad, a bearing seat); the "
                                  "optimizer places the rest. Earlier entries win."},
        "maxFraction": _FRACTIONS,
        "minFraction": _FRACTIONS,
        "requirements": {"type": "array", "items": {"type": "object", "properties": {
            "case": {"type": "string", "description": "the load case it reads"},
            "knock": {"type": "object", "properties": {
                "energy": {**_POS, "description": "J the knock brings"},
                "maxForce": {**_POS, "description": "N: the peak force it may stop with"},
                "safetyFactor": {**_POS, "description": "on the stored energy's stress (default "
                                                        "interfaceSafetyFactor or the goal's)"}},
                "required": ["energy", "maxForce"], "additionalProperties": False},
            "indentation": {"type": "object", "properties": {
                "relativeTo": {"type": "string", "description": "the same force spread out (a case)"},
                "atLeast": {**_POS, "description": "mm the case's load sinks in beyond relativeTo's"}},
                "required": ["relativeTo", "atLeast"], "additionalProperties": False}},
            "required": ["case"], "additionalProperties": False},
            "description": "Where the part must be soft: a knock that must stop under a force (a bumper), or "
                           "a patch that must give (a grip pad). These cases are not in the stiffness "
                           "objective."},
        "interfaceSafetyFactor": {**_POS, "description": "for the bond between materials (default: the "
                                                        "goal's safetyFactor, else 2)"},
        "interfaces": {"type": "object", "additionalProperties": {"type": "object", "properties": {
                           "tensile": _POS, "shear": _POS, "tensileZ": _POS, "shearZ": _POS},
                           "required": ["tensile"], "additionalProperties": False},
                       "description": "'a+b' -> bond strength (MPa), overriding interface_library.py's "
                                      "sourced card (measure your printer's: it varies a lot)"},
        "interfacePenalty": {"type": "boolean", "description": "constrain the bond in the optimizer "
                                                              "(default true); verification always checks it"},
        "export": {"type": "object", "properties": {
            "slicer": {"enum": ["all", "generic", "prusa", "bambu"]},
            "extruders": {"type": "object", "additionalProperties": {"type": "integer", "minimum": 1},
                          "description": "material name -> extruder / AMS slot (1-based)"}},
            "additionalProperties": False}},
    "required": ["materials"],
    "additionalProperties": False,
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
        "keepOut": {"type": "array", "items": {"$ref": "#/$defs/keepOut"},
                    "description": "Volume that must stay empty: bores, bolt access, cables, mating parts."},
        "fasteners": {"type": "array", "items": {"$ref": "#/$defs/fastener"}},
        "supports": {"type": "array", "items": {"$ref": "#/$defs/support"},
                     "description": "Where the part is held. Required unless every load is a pressure."},
        "skin": {"anyOf": [_POS, {"type": "object", "properties": {
                     "thickness": _POS,
                     "open": {"type": "array", "items": {"type": "string", "pattern": "^[+-]?[xyz]$"},
                              "description": "design-space faces left without skin, like '-z' (where ribs "
                                             "stand on a cover's inside)"}},
                     "required": ["thickness"], "additionalProperties": False}],
                 "description": "Skin-and-rib: mm of the design space's outer surface kept as a solid wall; "
                                "the optimizer designs the ribs behind it."},
        "loadCases": {"type": "array", "items": {"$ref": "#/$defs/loadCase"}, "minItems": 1},
        "loads": {"type": "array", "items": {"$ref": "#/$defs/load"},
                  "description": "Shorthand for a single load case."},
        "material": {"anyOf": [{"$ref": "#/$defs/material"},
                               {"type": "array", "items": {"$ref": "#/$defs/material"}, "minItems": 1}],
                     "description": "A list solves the goal in each and pools the outcomes."},
        "goal": {"$ref": "#/$defs/goal"},
        "minMember": {**_POS, "description": "Thinnest member, mm (default: 2 x the process's minFeature, "
                                             "else 1.4 for the filter)."},
        "minHole": {**_POS, "description": "Narrowest hole or gap, mm (default: the process's minHole, else "
                                           "none). The optimizer pays for the material a hole this narrow "
                                           "would close up with, and the delivered part has it closed."},
        "drains": {"anyOf": [
            {"enum": ["auto"]}, {"type": "boolean"},
            {"type": "object", "properties": {
                "enabled": {"type": "boolean"},
                "diameter": {**_POS, "description": "mm (default SLS 6, MJF 7.5, SLA/DLP 5.5, volumetric 5, "
                                                    "LPBF 6.8, else 6)"},
                "count": {"type": "integer", "minimum": 1, "description": "holes per sealed void (default 2)"},
                "maxPath": {"anyOf": [_POS, {"type": "null"}],
                            "description": "mm: more holes while any powder is farther than this from one, "
                                           "through the void (default 60; null: off)"},
                "iterations": {"type": "integer", "minimum": 0,
                               "description": "optimizer iterations after drilling, so the design carries "
                                              "load round the holes (default 20)"}},
             "additionalProperties": False}],
            "description": "Escape holes into voids the optimizer sealed: 'auto' (default) for powder and resin "
                           "processes and LPBF, true, false, or options. The holes are drilled where the "
                           "wall is thin and the strain energy low, then the design re-converges round them."},
        "symmetry": {"type": "array", "items": {"enum": ["x", "y", "z"]},
                     "description": "Planes through the design-space centre, by normal axis."},
        "resolution": {"type": "object",
                       "properties": {"maxElements": {"type": "integer", "minimum": 1, "maximum": 4000000},
                                      "pitch": _POS},
                       "additionalProperties": False},
        "maxIterations": {"type": "integer", "minimum": 1},
        "gravity": {**_VEC3, "description": "m/s², the part's own weight in every load case"},
        "thermal": _THERMAL,
        "printDirection": {"type": "string", "pattern": "^[+-]?[xyz]$",
                           "description": "The direction layers stack; turns on overhang control."},
        "extrusion": {"type": "string", "pattern": "^[+-]?[xyz]$",
                      "description": "The part is a profile extruded along this axis (cut from plate)."},
        "maxMember": {**_POS, "description": "Thickest member, mm (at least 2 x minMember): thick members "
                                             "hollow out or split."},
        "milling": {"type": "object", "properties": {
                        "axes": {"enum": [2.5, 3, 5], "description": "2.5: flat pocket floors every stepDown; "
                                                                    "5: 3+2 positional, all six sides"},
                        "directions": {"type": "array", "items": {"type": "string", "pattern": "^[+-]?[xyz]$"},
                                       "minItems": 1,
                                       "description": "sides the cutter comes from, like '+z' (default +z; "
                                                      "all six for 5-axis)"},
                        "toolDiameter": {**_POS, "description": "mm, smallest cutter: narrower slots and "
                                                                "tighter inside corners are filled"},
                        "stepDown": {**_POS, "description": "mm between 2.5-axis pocket floors (default 2)"}},
                    "additionalProperties": False,
                    "description": "The part is machined: every void reachable by a cutter coming straight in."},
        "casting": {"anyOf": [{"type": "string", "pattern": "^[+-]?[xyz]$"},
                              {"type": "object", "properties": {
                                  "draw": {"type": "string", "pattern": "^[+-]?[xyz]$",
                                           "description": "the direction the mold opens"},
                                  "sides": {"enum": [1, 2], "description": "2 (default): halves pull both ways "
                                                                           "along the draw; 1: one half pulls off "
                                                                           "a flat parting plane"}},
                               "required": ["draw"], "additionalProperties": False}],
                    "description": "Cast or molded: no undercuts along the draw direction."},
        "pattern": {"type": "object", "properties": {
                        "type": {"const": "circular"},
                        "axis": {"enum": ["x", "y", "z"]},
                        "count": {"type": "integer", "minimum": 2},
                        "center": {**_VEC3, "description": "a point on the axis (default: design-space centre)"}},
                    "required": ["count"], "additionalProperties": False,
                    "description": "Circular pattern: count identical sectors round the axis."},
        "process": _PROCESS,
        "slicer": _SLICER,
        "buildSim": _BUILD_SIM,
        "durability": _DURABILITY,
        "lattice": {"type": "object", "properties": _LATTICE_PROPS, "additionalProperties": False,
                    "description": "Optional: fill the delivered part with a stress-graded TPMS lattice."},
        "multiMaterial": _MULTI_MATERIAL,
        "expect": {"type": "object", "description": "Eval-only pass criteria (loadpath.evals)."},
    },
    "required": ["designSpace"],
    "x-requireOne": [["loadCases"], ["loads"]],
    "additionalProperties": False,
    "$defs": {**_SHAPES, "shape": _SHAPE, "region": _REGION, "keepOut": _KEEP_OUT, "support": _SUPPORT,
              "load": _LOAD, "loadCase": _LOAD_CASE, "material": _ONE_MATERIAL, "fastener": _FASTENER, "goal": _GOAL},
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
        "shell": _POS, "header": _POS,
        "cap": {**_POS, "description": "mm: how deep each plenum seals off the other fluid's channels at the "
                                       "core's ends (default wall + 1); thinner leaves more core open to flow "
                                       "and fewer dead pockets"},
        "portLength": _POS,
        "phase": {"type": "array", "items": {"type": "number"}, "minItems": 3, "maxItems": 3,
                  "description": "mm: shift the TPMS against the core's low corner, which moves where the core's "
                                 "faces cut the channels (a partial channel at an edge starves)"},
        "portDiameter": {"anyOf": [_POS, {"type": "object", "additionalProperties": _POS}],
                         "description": "mm, every port; or one per fluid name, e.g. {\"water\": 8, \"air\": 14} "
                                        "(a gas side's ports set most of its pressure drop)"},
        "layout": {"enum": ["cross", "counter"]},
        "axes": {"type": "string", "pattern": "^[xyz]{2}$",
                 "description": "cross: A along axes[0], B along axes[1]; counter: A through the ends along "
                                "axes[0], B's side ports along axes[1]"},
        "fluids": {"type": "array", "items": {"type": "string"}, "minItems": 2, "maxItems": 2},
        "build": {"anyOf": [{"enum": ["x", "y", "z", "none"]}, {"type": "null"}, {"const": False}],
                  "description": "The axis pointing up while printing (null: no print shaping)."},
        "skirt": {"type": "boolean"},
        "split": {"type": "number", "exclusiveMinimum": 0.05, "exclusiveMaximum": 0.95,
                  "description": "the first fluid's share of the void (default 0.5): the sheet moves toward the "
                                 "second. An air side against a liquid wants more, e.g. 0.66"},
        "process": {"enum": ["fdm", "lpbf"],
                    "description": "fdm (default) or lpbf: laser powder-bed metal (AlSi10Mg, Ti64, 316L). "
                                   "lpbf drops the skirt (outside supports are cut off) and the report's "
                                   "internalCeilings must be near zero: nothing inside can be supported."},
        "cleanouts": {"type": "integer", "minimum": 0,
                      "description": "powder clean-out ports per fluid (default 0): each a bore through the outer "
                                     "wall at the point of that fluid's space farthest from every opening, on a "
                                     "boss to tap for a plug; the report lists each and its flat plug for "
                                     "check_networks. With lpbf the report's powder entry says how far powder "
                                     "must travel to get out."},
        "cleanoutDiameter": {**_POS, "description": "mm (default 4.2: an M5 tap drill)"},
        "closure": {"type": "boolean",
                    "description": "with build (default true): channels close gently under core surface that "
                                   "looks up and flare open over plenums below, so no channel ends in a "
                                   "flat ceiling"},
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
        "expect": {"type": "object", "description": "Eval-only pass criteria (loadpath.evals)."},
        "envelope": {"type": "array", "items": {"type": "number", "exclusiveMinimum": 0}, "minItems": 3,
                     "maxItems": 3,
                     "description": "Instead of a core: the box (mm, x y z) the whole part must fit, ports and "
                                    "headers included. `loadpath heatx --size` then picks the core's size, cell, "
                                    "wall, split, layout and each fluid's port for the duty (sizing.py)."},
        "sizing": {"type": "object", "properties": {
            "process": {"enum": ["fdm", "lpbf", "sls", "mjf", "sla", "dlp"]},
            "maxPressureDrop": {"type": "object", "additionalProperties": _POS,
                                "description": "Pa per fluid name"},
            "minEffectiveness": {"type": "number", "minimum": 0, "maximum": 1},
            "targetDuty": {**_POS, "description": "W"},
            "objective": {"enum": ["mass", "effectiveness", "pressure"]}},
            "additionalProperties": False,
            "description": "With envelope: what the sized exchanger must meet, and what to rank by."},
        "simulate": {"anyOf": [{"type": "boolean"}, {
            "type": "object",
            "properties": {"voxel": {**_POS, "description": "mm (default min(cell/2, port/4, header/3))"},
                           "closures": {"enum": ["design", "lbm"],
                                        "description": "design: the rating's laws (Savoldi pressure drop, "
                                                       "Reynolds 2023 Nu); lbm: both refitted to LBM gyroid "
                                                       "cells (better for thick walls, phi under 0.3)"}},
            "additionalProperties": False}],
            "description": "With duty: simulate the whole exchanger (porous core, open headers, both fluids "
                           "and the wall): flow uniformity, pressure field, temperatures and hot spots, "
                           "beside the epsilon-NTU rating. Seconds to a few minutes."},
        "duty": {"type": "object", "description": "Optional: rate it. One entry per fluid name, plus 'wall' "
                                                  "(a material name or W/m·K).",
                 "properties": {"wall": {"anyOf": [{"type": "string"}, _POS]}},
                 "additionalProperties": {"$ref": "#/$defs/dutySide"}},
    },
    "x-requireOne": [["core"], ["envelope"]],
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

# --- field graphs (fields.py): nTop-style implicit modelling ---

_FIELD = {"title": "a number, a node name or a node object",
          "description": "A number, the name of another node, or an inline node object: anywhere a "
                         "number goes, a field can drive it.",
          "anyOf": [{"type": "number"}, {"type": "string", "description": "a node name"},
                    {"$ref": "#/$defs/node"}]}
_BOX = {"type": "object", "description": "Axis-aligned box (mm).",
        "properties": {"min": _VEC3, "max": _VEC3}, "required": ["min", "max"], "additionalProperties": False}
_COMMON = {"description": {"type": "string", "description": "What the node is for (ignored)."},
           "bounds": {**_BOX, "description": "Where this solid lives, when it cannot be worked out "
                                             "(meshing needs the output's)."}}


def _op(name, desc, props, required=(), require_one=None):
    d = {"type": "object", "description": desc,
         "properties": {"op": {"const": name}, **props, **_COMMON},
         "required": ["op", *required], "additionalProperties": False}
    if require_one:
        d["x-requireOne"] = require_one
    return d


_FIELD_OPS = {
    "constant": _op("constant", "A number everywhere.", {"value": {"type": "number"}}, ["value"]),
    "coordinate": _op("coordinate", "Position along an axis (mm): 'x', 'y', 'z', or a direction from "
                                    "`origin`.", {"axis": _AXIS, "origin": _VEC3}),
    "distance": _op("distance", "Unsigned distance (mm) to a solid's surface (`input`), to points, a "
                                "segment, a plane or a shape. The usual driver of a ramp: thicker near the "
                                "bolts, finer cells near the load.",
                    {"input": _FIELD, "points": {"type": "array", "items": _VEC3, "minItems": 1},
                     "segment": {"type": "array", "items": _VEC3, "minItems": 2, "maxItems": 2},
                     "plane": {"type": "object", "properties": {"point": _VEC3, "normal": _VEC3},
                               "required": ["point", "normal"], "additionalProperties": False},
                     "shape": {"$ref": "#/$defs/shape"},
                     "signed": {"type": "boolean", "description": "with input: keep the sign (negative inside)"}},
                    require_one=[["input"], ["points"], ["segment"], ["plane"], ["shape"]]),
    "shape": _op("shape", "An exact primitive solid (box, cylinder, tube, sphere, union), as in a study.",
                 {"shape": {"$ref": "#/$defs/shape"}}, ["shape"]),
    "mesh": _op("mesh", "A closed mesh (STL/3MF/OBJ) as a solid: exact distance near its surface, so it "
                        "re-meshes smooth.",
                {"path": {"type": "string", "description": "relative to the graph's folder"},
                 "source": {"type": "string", "description": "an in-memory mesh passed to FieldGraph(meshes=)"},
                 "pitch": {**_POS, "description": "mm, far-field voxel grid (default extent/200, >= 0.25)"},
                 "band": {**_POS, "description": "mm from the surface within which distance is exact "
                                                 "(default 3 pitches); make it cover shell thicknesses"}},
                require_one=[["path"], ["source"]]),
    "result": _op("result", "A simulation result as a scalar field: von Mises stress (MPa) or displacement "
                            "(mm) of `part` under `study` (analyzed on demand), a saved .npz (points + values, "
                            "or verify fields), or an in-memory source.",
                  {"path": {"type": "string", "description": ".npz with points and values (save_result_field) "
                                                            "or a grid (origin, pitch, values)"},
                   "source": {"type": "string"},
                   "study": {"type": "string", "description": "study JSON, relative to the graph's folder, or "
                                                              "'example:<name>' for a packaged example"},
                   "part": {"type": "string", "description": "the mesh the study analyzes, or the name of a "
                                                             "node of this graph (meshed for the analysis)"},
                   "pitch": {**_POS, "description": "mm, meshing a node part for the analysis (default 1)"},
                   "quantity": {"type": "string", "description": "vonMises (default) or displacement for a "
                                                                 "study; the array name in an .npz"},
                   "case": {"type": "string", "description": "one load case (default: the worst at each point)"},
                   "scale": {"anyOf": [{"const": "peak"}, _POS],
                             "description": "divide by this, or 'peak' (the 99.5th percentile): remap in "
                                            "fractions of the part's own worst stress"},
                   "cells": {"type": "integer", "minimum": 1000, "description": "analysis size (default 120000)"}},
                  require_one=[["path"], ["source"], ["study", "part"]]),
    "grid": _op("grid", "An imported scalar field on a regular grid (.npz with origin, pitch, values[nx,ny,nz]): "
                        "a temperature map, a CFD pressure, a hand-painted density.",
                {"path": {"type": "string"}, "source": {"type": "string"},
                 "quantity": {"type": "string", "description": "array name (default values)"}},
                require_one=[["path"], ["source"]]),
    "image": _op("image", "A picture as a scalar field from 0 to 1, laid on a plane and carried straight through "
                          "along the third axis: a logo to emboss, a texture to drive a wall, a hand-painted density.",
                 {"path": {"type": "string", "description": "PNG, JPEG, ... in the graph's folder"},
                  "plane": {"enum": ["xy", "xz", "yz"], "description": "the plane it lies in (default xy)"},
                  "origin": {"type": "array", "items": {"type": "number"}, "minItems": 2, "maxItems": 2,
                             "description": "lower-left corner in the plane, mm (default [0, 0])"},
                  "size": {"type": "array", "items": {"type": "number", "exclusiveMinimum": 0}, "minItems": 2,
                           "maxItems": 2, "description": "[width, height] mm"},
                  "channel": {"enum": ["luma", "r", "g", "b", "a"], "description": "default luma"},
                  "invert": {"type": "boolean"}}, ["path", "size"]),
    "remap": _op("remap", "Map `input` from [a, b] to [c, d] through a curve (clamped unless clamp is false). "
                          "Longer from/to lists are a piecewise-linear table (from ascending), e.g. a density "
                          "per height for an impact core.",
                 {"input": _FIELD,
                  "from": {"type": "array", "items": {"type": "number"}, "minItems": 2},
                  "to": {"type": "array", "items": {"type": "number"}, "minItems": 2},
                  "curve": {"anyOf": [
                      {"enum": ["linear", "smoothstep", "smootherstep", "sqrt", "square"]},
                      {"type": "array", "minItems": 2, "description": "points [u, v] over the unit square, u "
                                                                       "ascending from 0 to 1: a drawn curve",
                       "items": {"type": "array", "items": {"type": "number"}, "minItems": 2, "maxItems": 2}}]},
                  "clamp": {"type": "boolean"}}, ["input", "from", "to"]),
    "expr": _op("expr", "Arithmetic over node names, x, y, z, pi, e and abs sqrt exp log sin cos tan floor ceil "
                        "sign min max clamp lerp smoothstep step where, e.g. '0.8 + 1.2 * smoothstep(5, 30, "
                        "stress)'.",
                {"expr": {"type": "string"},
                 "vars": {"type": "object", "additionalProperties": _FIELD,
                          "description": "local names for fields used in expr"}}, ["expr"]),
    "union": _op("union", "Join solids; `blend` (mm, or a field) is the fillet radius where they meet.",
                 {"inputs": {"type": "array", "items": _FIELD, "minItems": 1}, "blend": _FIELD}, ["inputs"]),
    "intersect": _op("intersect", "Keep what is inside every input; `blend` rounds the edges.",
                     {"inputs": {"type": "array", "items": _FIELD, "minItems": 1}, "blend": _FIELD}, ["inputs"]),
    "subtract": _op("subtract", "Cut the second and later inputs out of the first; `blend` fillets the cut.",
                    {"inputs": {"type": "array", "items": _FIELD, "minItems": 2}, "blend": _FIELD}, ["inputs"]),
    "shell": _op("shell", "A wall of `thickness` (mm, or a field: a variable wall) on a solid's surface.",
                 {"input": _FIELD, "thickness": _FIELD,
                  "side": {"enum": ["inside", "outside", "center"], "description": "default inside"}},
                 ["input", "thickness"]),
    "offset": _op("offset", "Grow (positive) or shrink a solid by `distance` (mm, or a field).",
                  {"input": _FIELD, "distance": _FIELD}, ["input", "distance"]),
    "lattice": _op("lattice", "A lattice filling all of space (intersect it with a solid): a TPMS sheet "
                              "(gyroid, diamond, schwarz-p) or a strut cell (cubic, bcc, bccz, fcc, fccz, octet, "
                              "kelvin, diamond-strut) or a stochastic voronoi foam. Cell size, wall or strut "
                              "thickness, relative density and node blend can each be a field; a conformal frame "
                              "lays the cells along a surface.",
                   {"type": {"enum": ["gyroid", "diamond", "schwarz-p", "cubic", "bcc", "bccz", "fcc", "fccz",
                                      "octet", "kelvin", "diamond-strut", "voronoi"],
                             "description": "TPMS sheets are stiffer and stronger per gram (lattice.SHEETS vs "
                                            "STRUTS); struts are open, easy to clear of powder, and print on FDM "
                                            "when the cell is oriented (lattice report's overhangs)"},
                    "cell": {**_FIELD, "description": "mm per period (default 8); a field grades it (voronoi: "
                                                      "mean seed spacing)"},
                    "thickness": {**_FIELD, "description": "sheet wall or strut diameter, mm (default 1)"},
                    "density": {**_FIELD, "description": "relative density instead of thickness (sheets "
                                                         "0.05-0.95, struts 0.02-0.95; the unit-cell tables "
                                                         "set the wall or diameter)"},
                    "blend": {**_FIELD, "description": "strut cells: mm smooth-union radius at the nodes "
                                                       "(node fillets; default 0)"},
                    "form": {"enum": ["sheet", "network", "beam"],
                             "description": "sheet (TPMS default): a wall on the surface; network: one side "
                                            "solid; beam: strut cells (their only form)"},
                    "grading": {"enum": ["warp", "blend"],
                                "description": "how a field cell grades: warp (default) bends the lattice's "
                                               "coordinates so one period is one local cell, continuous "
                                               "everywhere; blend cross-fades fixed periods (cylindrical frame)"},
                    "aspect": {"type": "array", "items": _POS, "minItems": 3, "maxItems": 3,
                               "description": "stretch the cell per axis, e.g. [1, 1, 1.5]: a tall BCC cell's "
                                              "struts rise above 45 degrees and print on FDM unsupported"},
                    "seed": {"type": "integer", "description": "voronoi: random seed (default 0)"},
                    "offset": {**_FIELD, "description": "mm, moves the level surface (network form)"},
                    "origin": _VEC3,
                    "frame": {"type": "object",
                              "description": "cylindrical wraps the cells round an axis, spherical round a "
                                             "centre; conformal lays them along a solid's surface (layers "
                                             "parallel to the skin, the first on it; in-plane cells follow the "
                                             "surface as a stripe pattern)",
                              "properties": {"type": {"enum": ["cartesian", "cylindrical", "spherical",
                                                               "conformal"]},
                                             "axis": {**_AXIS, "description": "cylindrical: the axis; conformal: "
                                                                              "the direction cells line up with "
                                                                              "along the surface (default the "
                                                                              "longest extent)"},
                                             "center": _VEC3,
                                             "radius": {**_POS, "description": "mm where cells are square"},
                                             "surface": {**_FIELD, "description": "conformal: the solid whose "
                                                                                  "surface the cells follow"},
                                             "normalCell": {**_POS, "description": "conformal: mm cell through "
                                                                                   "the thickness (default the "
                                                                                   "cell)"}},
                              "required": ["type"], "additionalProperties": False}},
                   require_one=None),
    "transform": _op("transform", "Move, rotate or scale a field's input (translate, then rotate, then scale).",
                     {"input": _FIELD, "translate": _VEC3,
                      "rotate": {"type": "object", "properties": {"axis": _AXIS, "degrees": {"type": "number"},
                                                                  "center": _VEC3},
                                 "required": ["degrees"], "additionalProperties": False},
                      "scale": _POS}, ["input"]),
    "group": _op("group", "A copy of a sub-graph from the graph's `groups`, with its own param values and "
                          "inputs; it stands for the group's output.",
                 {"group": {"type": "string", "description": "the name of a group in `groups`"},
                  "params": {"type": "object", "additionalProperties": {"type": "number"},
                             "description": "param -> value (the group's defaults otherwise)"},
                  "inputs": {"type": "object", "additionalProperties": {"type": "string"},
                             "description": "the group's input name -> a node of this graph"}}, ["group"]),
}

_NODE = {
    "title": "a node object with an 'op'",
    "description": "A field node: " + ", ".join(_FIELD_OPS) + ".",
    "oneOf": [{"$ref": f"#/$defs/op-{k}"} for k in _FIELD_OPS],
    "x-discriminator": {"property": "op", "mapping": {k: f"#/$defs/op-{k}" for k in _FIELD_OPS}},
}

_MATERIAL_BODY = {
    "type": "object", "required": ["name", "material"], "additionalProperties": False,
    "description": "One material body of a field graph (fields.material_bodies): the part's points inside "
                   "`solid` (default the whole part) where the `where` field is above `above` and below "
                   "`below`, less what earlier bodies took.",
    "properties": {
        "name": {"type": "string", "description": "Names the body's STL (<name>-<body>.stl) and its 3MF part."},
        "material": {"anyOf": [_ONE_MATERIAL, {"type": "array", "items": _ONE_MATERIAL, "minItems": 2,
                                               "maxItems": 2}],
                     "description": "The body's material, or with `mix` the pair [A, B] it blends."},
        "solid": {**_FIELD, "description": "solid node the body is cut from (default the output)"},
        "where": {**_FIELD, "description": "scalar field that picks the body's points inside `solid` "
                                           "(a remapped distance, a `result` stress)"},
        "above": {**_FIELD, "description": "the body is where `where` > this (a number or a field)"},
        "below": {**_FIELD, "description": "the body is where `where` < this (a number or a field)"},
        "mix": {**_FIELD, "description": "0..1, the fraction of material B (material[1]) against A at each "
                                         "point: a graded blend for multi-material jetting, no hard "
                                         "partition. Writes a voxel print job (voxelprint.py) and weighs the "
                                         "body by the blend."},
        "voxel": {"type": "object", "additionalProperties": False,
                  "description": "a mix body's voxel job",
                  "properties": {
                      "pitch": {**_POS, "description": "mm, the fraction grid (default the mesh pitch)"},
                      "printer": {"type": "string", "description": "voxelprint printer (default j850)"},
                      "materials": {"type": "array", "minItems": 2, "maxItems": 2,
                                    "items": {"anyOf": [{"type": "string"}, {"type": "array"}, {"type": "object"}]},
                                    "description": "printer resins for A and B (default from the cards: "
                                                   "vero -> VeroCyan, agilus30 -> Agilus30Blk)"},
                      "maxLayers": {"type": "integer", "minimum": 1, "description": "write only this many "
                                                                                    "layers (previews)"}}},
        "extruder": {"type": "integer", "minimum": 1,
                     "description": "FDM tool for the 3MF (default: one per distinct material, in order)"},
        "color": {"type": "string", "description": "#RRGGBB in the 3MF (default by material)"}},
}

FIELD_SCHEMA = {
    "$schema": _DIALECT,
    "$id": "https://github.com/connorwhite-online/loadpath/schema/field.schema.json",
    "title": "loadpath field graph",
    "description": "An implicit (field) part, nTop style: named nodes, each a solid (signed distance, negative "
                   "inside, mm) or a scalar field, where any numeric parameter can be another field.",
    "type": "object",
    "properties": {
        "name": {"type": "string", "description": "Names the output files."},
        "description": {"type": "string"},
        "nodes": {"type": "object", "additionalProperties": {"$ref": "#/$defs/node"},
                  "description": "name -> node. Names are what other nodes and expr refer to."},
        "output": {"type": "string", "description": "The node to mesh (default the last)."},
        "groups": {"type": "object", "description": "name -> a reusable sub-graph that `group` nodes copy",
                   "additionalProperties": {
                       "type": "object", "additionalProperties": False, "required": ["nodes"],
                       "properties": {
                           "description": {"type": "string"},
                           "params": {"type": "object", "additionalProperties": {"type": "number"},
                                      "description": "param -> default; the group's nodes refer to it by name"},
                           "inputs": {"type": "array", "items": {"type": "string"},
                                      "description": "names the group's nodes use for nodes the instance gives"},
                           "nodes": {"type": "object", "additionalProperties": {"$ref": "#/$defs/node"}},
                           "output": {"type": "string", "description": "the node the group stands for "
                                                                       "(default the last)"}}}},
        "bounds": {**_BOX, "description": "Meshing box (default: worked out from the output)."},
        "mesh": {"type": "object", "properties": {
            "pitch": {**_POS, "description": "mm grid (default: the box over 4M cells); a third of the "
                                             "thinnest wall or finer"},
            "project": {"type": "integer", "minimum": 0, "description": "surface relax+projection rounds "
                                                                       "(default 3; 0 = raw marching cubes)"},
            "sharp": {"type": "boolean", "description": "hard booleans (no blend) with plain shapes and meshes are "
                                                        "done exactly, so bores, bolt faces and keep-outs keep "
                                                        "sharp edges (default true; false meshes the whole field, "
                                                        "every crease rounded over about one pitch)"},
            "creases": {"type": "boolean", "description": "hard booleans between field surfaces (a lattice clipped "
                                                          "by a body, two bodies unioned with no blend, a shell's "
                                                          "rim) keep sharp creases: edges that cross from one "
                                                          "surface to the other are split on the true intersection "
                                                          "curve (default false; not with stream)"},
            "decimate": {**_POS, "description": "mm: thin the triangles wherever the mesh stays this close to "
                                                "the surface (flat and gently curved regions lose most of "
                                                "theirs; creases and small features keep them), e.g. 0.01-0.05"},
            "stream": {"type": "boolean", "description": "write the STL slab by slab without holding the mesh, "
                                                         "in bounded memory and with no cell budget: for parts "
                                                         "of tens of millions of triangles (fine lattices). "
                                                         "Skips the printability check, bodies and the render."}},
                 "additionalProperties": False},
        "step": {"anyOf": [{"type": "boolean"}, {"type": "object", "properties": {
            "faces": {"type": "integer", "minimum": 100, "description": "organic facets in the B-rep "
                                                                      "(default 20000, ~45 MB)"}},
            "additionalProperties": False}],
                 "description": "also write <name>.step (needs loadpath[step]): exact planes, cylinders and spheres "
                                "where the graph's hard booleans use plain shapes, the rest faceted"},
        "material": {**_ONE_MATERIAL, "description": "Reports the mass."},
        "materials": {"type": "array", "minItems": 1, "items": _MATERIAL_BODY,
                      "description": "Multi-material: split the output into material bodies, in precedence "
                                     "order (an earlier body wins where regions overlap, so bodies never "
                                     "overlap and together they are the part). Each body is a solid node's "
                                     "share of the part, a field threshold's (where/above/below: TPU where a "
                                     "distance or a stress says), or both; a body with neither takes what is "
                                     "left. Writes <name>-<body>.stl per body (sharing their interfaces "
                                     "exactly) and a multi-material <name>.3mf; the report gives each body's "
                                     "volume and mass and each interface's area."},
        "minWall": {**_POS, "description": "mm, the printability check's minimum wall (default 0.8)"},
        "build": {**_VEC3, "description": "build direction for the overhang check (default +z)"},
        "check": {"type": "boolean", "description": "run the printability check (default true)"},
        "editor": {"type": "object", "description": "What `loadpath edit` keeps with the graph (node positions in "
                                                    "'layout'); nothing else reads it."},
        "tune": {"type": "object", "additionalProperties": False, "required": ["study", "params"],
                 "description": "`loadpath tune`: turn these numbers for the lightest part that still meets the "
                                "study's limits, each candidate meshed and analyzed (tune.py)",
                 "properties": {
                     "study": {"type": "string", "description": "study JSON path or example:<name>: loads, "
                                                                "supports, material, goal"},
                     "node": {"type": "string", "description": "the node analyzed (default the output)"},
                     "params": {"type": "array", "minItems": 1, "items": {
                         "type": "object", "additionalProperties": False, "required": ["path", "min", "max"],
                         "properties": {
                             "path": {"type": "string", "description": "where the number is, '/'-separated: "
                                                                       "nodes/web/shape/max/2"},
                             "also": {"type": "array", "items": {"type": "string"},
                                      "description": "numbers that move with it ('-' first: the other way)"},
                             "min": {"type": "number"}, "max": {"type": "number"}}}},
                     "safetyFactor": {**_POS, "description": "least safety factor (default the study's goal)"},
                     "maxDeflection": {**_POS, "description": "mm at the loads (default the study's goal)"},
                     "evaluations": {"type": "integer", "minimum": 2, "description": "analyses to spend "
                                                                                       "(default 20)"},
                     "sweep": {"type": "object", "minProperties": 1, "maxProperties": 1,
                               "description": "the trade-off: tune again held to each of these values of one "
                                              "limit, e.g. {\"maxDeflection\": [0.4, 0.6, 0.8]}",
                               "properties": {k: {"type": "array", "minItems": 1, "items": _POS}
                                              for k in ("safetyFactor", "maxDeflection")},
                               "additionalProperties": False},
                     "analysis": {"enum": ["auto", "solid", "homogenized"],
                                  "description": "auto (default): a part with a lattice is analyzed with the "
                                                 "lattice homogenized (unit-cell stiffness and wall stress), "
                                                 "else meshed and verified as a solid"},
                     "pitch": {**_POS, "description": "mm mesh for each candidate (default 1)"},
                     "cells": {"type": "integer", "minimum": 1000, "description": "analysis cells (default 30000)"}}},
        "impact": {"type": "object", "properties": {
            "envelope": {"type": "string", "description": "solid node the lattice core fills (the crush pad)"},
            "density": {"anyOf": [{"type": "string"}, {"type": "number", "exclusiveMinimum": 0, "maximum": 1}],
                        "description": "relative density: a scalar node (the one the lattice node uses) or a "
                                       "number (default 1 = solid)"},
            "lattice": {"enum": ["gyroid", "diamond", "schwarz-p"], "description": "sheet type (default gyroid)"},
            "axis": {"enum": ["x", "y", "z"], "description": "crush direction (default z)"},
            "mass": {**_POS, "description": "kg, the moving mass"},
            "velocity": {**_POS, "description": "m/s at contact"},
            "energy": {**_POS, "description": "J, instead of mass and velocity"},
            "forceLimit": {**_POS, "description": "N, the most the part behind may feel"},
            "pitch": {**_POS, "description": "mm sampling grid (default 0.5)"},
            "grade": {"type": "boolean", "description": "also work out the flattest graded core for this "
                                                         "envelope and give it as a density remap table"},
            "rhoMin": {**_POS, "description": "lowest printable relative density for grading (default 0.12)"},
            "rhoMax": {**_POS, "description": "highest relative density for grading (default 0.5)"},
            "cell": {**_POS, "description": "mm lattice cell: with minWall, raises rhoMin to the thinnest "
                                            "printable wall"},
            "minWall": {**_POS, "description": "mm thinnest printable sheet (default 0.8)"},
            "rate": {"type": "boolean", "description": "elastomers (TPU): stiffen the crush law for the "
                                                       "impact's crush time (viscoelastic.py; default true)"}},
                   "required": ["envelope"], "additionalProperties": False,
                   "description": "Crush and impact of a lattice core (impact.py): force-crush curve, where a "
                                  "mass at a speed stops, peak force and g, and with grade a density table that "
                                  "crushes every layer at the same force. Needs material."},
        "vibration": {"type": "object", "properties": {
            "envelope": {"type": "string", "description": "solid node analysed (homogenized lattice inside)"},
            "density": {"anyOf": [{"type": "string"}, {"type": "number", "exclusiveMinimum": 0, "maximum": 1}],
                        "description": "relative density node or number (default 1 = solid)"},
            "lattice": {"enum": ["gyroid", "diamond", "schwarz-p"]},
            "fixed": {"type": "string", "description": "solid node: nodes inside it are clamped"},
            "pointMasses": {"type": "array", "items": {"type": "object", "properties": {
                "region": {"type": "string", "description": "solid node the mass attaches over"},
                "mass": {**_POS, "description": "kg"}}, "required": ["region", "mass"],
                "additionalProperties": False}},
            "modes": {"type": "integer", "minimum": 1, "maximum": 50, "description": "how many (default 6)"},
            "pitch": {**_POS, "description": "mm analysis grid (default 1)"},
            "minFrequency": {**_POS, "description": "Hz the first mode must clear"},
            "excitation": {**_POS, "description": "Hz of a known shake: reports the worst transmissibility"},
            "lossFactor": {**_POS, "description": "material loss factor (default from modal.LOSS_FACTOR)"}},
                      "required": ["envelope", "fixed"], "additionalProperties": False,
                      "description": "Natural frequencies, effective masses and damping (modal.py). Needs "
                                     "material."},
        "process": {**_PROCESS, "description": "How it is made: sets the minimum wall, the build direction and "
                                               "the drain holes (sls, mjf, sla, dlp, volumetric, lpbf drain "
                                               "sealed voids by default)."},
        "drains": {"anyOf": [
            {"enum": ["auto"]}, {"type": "boolean"},
            {"type": "object", "properties": {
                "enabled": {"type": "boolean"},
                "diameter": {**_POS, "description": "mm (default 1.5 x the process's minimum escape hole or 2 mm "
                                                    "over it, whichever is larger: SLS 6, MJF 7.5, SLA/DLP 5.5, "
                                                    "volumetric 5; LPBF 6.8, an M8 tap drill)"},
                "count": {"type": "integer", "minimum": 1, "description": "holes per void (default 2: one "
                                                                         "low, one high along the build)"},
                "stress": {"type": ["string", "null"], "description": "node that ranks the sites, calmest "
                                                                     "first (default the first result node; "
                                                                     "null: none)"},
                "boss": {**_POS, "description": "mm of solid round each hole (default max(1.5 x min wall, "
                                                "0.4 x radius))"},
                "blend": {"type": "number", "minimum": 0, "description": "mm fillet boss to part "
                                                                         "(default the boss wall)"},
                "maxDepth": {**_POS, "description": "mm: sites at most this far under the outside (default 6)"},
                "fillBelow": {"type": "number", "minimum": 0, "description": "mm^3: smaller voids are filled "
                                                                           "solid (default the escape ball)"},
                "keepClear": {"type": "array", "items": {"type": "string"},
                              "description": "solid nodes the bosses stay out of (bearing seats, mating faces)"},
                "clearance": {"type": "number", "minimum": 0, "description": "mm from keepClear (default 1)"},
                "pitch": {**_POS, "description": "mm grid for finding voids (default the mesh pitch)"},
                "maxPath": {"anyOf": [_POS, {"type": "null"}],
                            "description": "mm: add holes (up to 8 a void) while any powder is farther than this "
                                           "from a hole, through the void (default 60; null: just count)"}},
             "additionalProperties": False}],
            "description": "Drain holes into sealed voids (trapped powder or resin): 'auto' (default) drills "
                           "for powder-bed and resin processes, true always, false never. Holes go at the "
                           "calmest flat sites, each on a solid boss; pockets too small for a hole are filled; "
                           "the report says what (if anything) is still sealed."},
    },
    "required": ["nodes"],
    "additionalProperties": False,
    "$defs": {**_SHAPES, "shape": _SHAPE, "node": _NODE, **{f"op-{k}": v for k, v in _FIELD_OPS.items()}},
}

SCHEMAS = {"study": STUDY_SCHEMA, "heatExchanger": HEAT_EXCHANGER_SCHEMA, "lattice": LATTICE_SCHEMA,
           "field": FIELD_SCHEMA}


def schema(kind: str = "study") -> dict:
    """A copy of one schema: 'study', 'heatExchanger', 'lattice' or 'field'."""
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
            "pipe": ["tube"], "ball": ["sphere"], "stl": ["mesh"], "group": ["union"],
            # material bodies
            "within": ["solid"], "inside": ["solid"], "threshold": ["below", "above"]}


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
    if kind == "field" and isinstance(doc, dict) and not any(lv == "error" for lv, _, _ in out):
        # beyond JSON Schema: references, expressions and cycles
        from .fields import FieldGraph
        try:
            FieldGraph(doc)
        except ValueError as exc:
            out.append(("error", "$.nodes", str(exc)))
    if kind == "study" and isinstance(doc, dict) and not doc.get("supports"):
        # beyond JSON Schema: supports are optional only for pressure vessels
        cases = doc.get("loadCases") or ([{"loads": doc["loads"]}] if isinstance(doc.get("loads"), list) else [])
        loads = [ld for c in cases if isinstance(c, dict) for ld in (c.get("loads") or [])]
        if not loads or not all(isinstance(ld, dict) and ld.get("pressure") for ld in loads):
            out.append(("error", "$.supports", "needs at least one support (where the part is held); only a "
                                               "study whose every load is a pressure may leave them out"))
    return [{"level": lv, "path": p, "message": m} for lv, p, m in out]


def format_issues(issues: list[dict]) -> str:
    return "\n".join(f"{i['level']}: {i['path']}: {i['message']}" for i in issues)
