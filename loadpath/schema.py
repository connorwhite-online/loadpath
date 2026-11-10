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
        "description": "Box: min/max corners, or center/size (axis-aligned), or center/size/axes (turned: "
                       "size along the two axes given and their cross product).",
        "properties": {"type": {"const": "box"}, "min": _VEC3, "max": _VEC3, "center": _VEC3, "size": _VEC3,
                       "radius": {"type": "number", "minimum": 0,
                                  "description": "mm: round every edge and corner (axis-aligned boxes; default 0)"},
                       "axes": {"type": "array", "items": _VEC3, "minItems": 2, "maxItems": 2,
                                "description": "the box's x and y axes as vectors (its z is their cross "
                                               "product); with center and size"}},
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
    "heightfield": {
        "type": "object",
        "description": "One side of a surface given as heights over a plane: the points whose n.x is below "
                       "(or above) h(u, v), h on a square grid in the plane's axes, bilinear between "
                       "samples. A stepped or curved parting surface (loadpath split writes these).",
        "properties": {"type": {"const": "heightfield"}, "normal": _VEC3,
                       "axes": {"type": "array", "items": _VEC3, "minItems": 2, "maxItems": 2,
                                "description": "the plane's u and v axes"},
                       "origin": {"type": "array", "items": {"type": "number"}, "minItems": 2, "maxItems": 2,
                                  "description": "(u, v) of heights[0][0]"},
                       "step": _POS,
                       "heights": {"type": "array", "items": {"type": "array", "items": {"type": "number"}},
                                   "description": "heights[i][j] = h(u0 + i step, v0 + j step) along normal"},
                       "side": {"enum": ["below", "above"]},
                       "offset": {"type": "number", "description": "moves the surface along normal (mm)"},
                       "depth": _POS},
        "required": ["type", "normal", "axes", "origin", "step", "heights"],
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
    "sdf": {
        "type": "object",
        "description": "A sampled signed distance field: an .npz (distance, origin, pitch; mm, negative "
                       "inside) relative to the study's folder, as loadpath.exterior.save_sdf writes it, or "
                       "an OpenVDB level set (.vdb, from Houdini, Blender, nTop or `loadpath sdf`). How "
                       "an SDF tool's shape (libfive, the sdf package, a field graph) comes in.",
        "properties": {"type": {"const": "sdf"}, "path": {"type": "string"},
                       "shroud": {"type": "string", "pattern": "^[0-9a-f]{20}$",
                                  "description": "a generated exterior by its key (what a shroud block "
                                                 "expands to; `loadpath shroud`)"}},
        "x-requireOne": [["path"], ["shroud"]],
        "required": ["type"],
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
    "intersect": {
        "type": "object",
        "description": "Where all the shapes overlap: a half ring is a tube intersected with a box.",
        "properties": {"type": {"const": "intersect"},
                       "shapes": {"type": "array", "items": {"$ref": "#/$defs/shape"}, "minItems": 2}},
        "required": ["type", "shapes"],
        "additionalProperties": False,
    },
    "subtract": {
        "type": "object",
        "description": "The first shape with the others cut away: a ring with a gap is a tube minus a box.",
        "properties": {"type": {"const": "subtract"},
                       "shapes": {"type": "array", "items": {"$ref": "#/$defs/shape"}, "minItems": 2}},
        "required": ["type", "shapes"],
        "additionalProperties": False,
    },
}
_SHAPE_TYPES = {"box": "box", "cylinder": "cylinder", "tube": "cylinder", "sphere": "sphere",
                "mesh": "mesh", "sdf": "sdf", "heightfield": "heightfield", "union": "union", "intersect": "intersect", "subtract": "subtract"}

_SHAPE = {
    "title": "a shape object (box, cylinder, tube, sphere, heightfield, mesh, sdf, union, intersect, subtract)",
    "description": "A shape: box, cylinder (tube), sphere, heightfield, mesh, sdf, union, intersect or subtract. "
                   "Units mm.",
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
                                           "(false for a pressurized cavity). A clearance volume keeps the "
                                           "designed material out but never cuts what is drawn (preserve)."},
                   "cutsPreserve": {"type": "boolean",
                                    "description": "Whether it cuts preserved shapes out of the part. Default: "
                                                   "true for a bore and for a part's way in (insertFrom), false "
                                                   "for a clearance volume. Only a bore ever cuts the skin: a way "
                                                   "in through the exterior is one of skin.openings."},
                   "wall": {**_POS, "description": "mm of material kept round it, pinned solid: a duct's or a "
                                                   "plenum's skin, with only what is behind it designed (ribs). "
                                                   "A pressurized keep-out gets minMember if not set."},
                   "fit": {"enum": ["press", "transition", "slip", "clearance", "close", "normal", "loose",
                                    "nominal", "none"],
                           "description": "A cylindrical bore's fit against the part that goes in it (fits.py): "
                                          "press (a bearing's outer ring, a dowel pin), transition, slip (a "
                                          "locating pin, a removable shaft), clearance (a shaft turning in it), "
                                          "close/normal/loose (bolt holes, ISO 273), nominal (hold the drawn "
                                          "diameter: the default for every bore once the process is known), none "
                                          "(no allowance). The hole is redrawn for the process's hole bias and "
                                          "report fits gives the as-built band and whether the fit holds."},
                   "mate": {**_POS, "description": "Diameter of the mating part, mm (default the cylinder's: draw "
                                                   "the keep-out at the bearing's or pin's size)."},
                   "insertFrom": {"anyOf": [{"type": "string", "pattern": "^[+-]?[xyz]$"}, _VEC3, {
                                      "type": "array", "minItems": 1, "description": "a path of legs as the part "
                                      "leaves its place: every leg but the last with a length; the last goes on "
                                      "out of the design space unless it has one too",
                                      "items": {"anyOf": [{"type": "string", "pattern": "^[+-]?[xyz]$"}, {
                                          "type": "object", "properties": {
                                              "toward": {"anyOf": [{"type": "string", "pattern": "^[+-]?[xyz]$"},
                                                                   _VEC3]},
                                              "length": _POS}, "required": ["toward"],
                                          "additionalProperties": False}, {
                                          "type": "object", "properties": {
                                              "turn": {"type": "number", "minimum": -180, "maximum": 180,
                                                       "description": "degrees the part turns on this leg, "
                                                                      "right-handed about `about` (a board "
                                                                      "tilted out from under a rim)"},
                                              "about": {"anyOf": [{"type": "string", "pattern": "^[+-]?[xyz]$"},
                                                                  _VEC3],
                                                        "description": "the direction of the line it turns "
                                                                       "about"},
                                              "at": {**_VEC3, "description": "a point on that line (mm): the "
                                                                             "pivot"},
                                              "along": {"type": "number", "description": "mm it slides down "
                                                                                         "the line as it turns "
                                                                                         "(default 0)"}},
                                          "required": ["turn", "about", "at"],
                                          "additionalProperties": False}]}}],
                                  "description": "Assembly path: the side the component (or a tool) comes in "
                                                 "from, like '-z', or a vector. The keep-out is swept out of the "
                                                 "design space that way, so nothing generated blocks putting "
                                                 "the part together (a board slid in from below, a driver "
                                                 "reaching a screw). A path's legs are slides {toward, length} "
                                                 "and turns {turn, about, at}: a part that only goes in at a "
                                                 "tilt (motion.py)."},
                   "finish": {"enum": ["auto", "as-built", "ream"],
                              "description": "auto (default): ream when the process cannot hold the fit as "
                                             "printed (the hole is drawn undersize for the reamer); as-built: "
                                             "keep the printed hole; ream: always."}},
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
        "maxForce": {"type": "number", "exclusiveMinimum": 0,
                     "description": "N: the most this support may carry (its reaction's resultant, any case), "
                                    "e.g. a bolt or bearing rating; reported always as report.reactions"},
        "maxMoment": {"type": "number", "exclusiveMinimum": 0,
                      "description": "N·mm: the most moment this support may carry about its centroid"},
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

        "direction": {"enum": ["any", "xy", "xz", "yz"],
                      "description": "'any': the force's size may point in any direction (contact, bumps, "
                                     "handling); the part is designed and checked for the worst one, found "
                                     "exactly from three unit-force solves. At most one per load case; a bearing "
                                     "load's uncertain force is spread evenly over the region. 'xy', 'xz' or "
                                     "'yz': any direction in that plane (a shaft's radial load, a swinging "
                                     "strap, a sideways crash): the case becomes cases 22.5 degrees "
                                     "apart (within 2% of the worst), named case@angle: eight over half "
                                     "the plane, or sixteen over all of it when -F is not the mirror of F "
                                     "(bearing distribution, other loads, a moment, a layered process); a moment on the same "
                                     "load turns with the force (right when its lever arm is along the "
                                     "plane's normal)."},
        "cone": {"type": "number", "exclusiveMinimum": 0, "maximum": 180,
                 "description": "Degrees: the force may tilt up to this far from its written direction, and the "
                                "worst tilt is designed for (180 is the same as direction 'any')."},
        "mass": {**_POS, "description": "kg carried through this region (a payload, the next link, a camera): "
                                        "adds mass x 9.81 x dynamic N along 'gravity', and with 'at' the moment "
                                        "of that force about the region (N·mm), so you never convert by hand."},
        "at": {**_VEC3, "description": "mm: where the force acts or the mass's centre sits (default: on the "
                                       "region): a pulley in front of a motor face, a tool's centre of mass. "
                                       "Its offset from the region's centre adds the moment; `check` echoes "
                                       "the resulting force and moment per case."},
        "dynamic": {"type": "number", "minimum": 1,
                    "description": "Multiplier on the mass's weight: 1 static, 2 a moving robot joint, "
                                   "3 anything that can be dropped or crash (default 1)."},
        "gravity": {**_SIGNED_AXIS, "description": "Which way the mass pulls: '-z' (default), or a vector."},
    },
    "x-requireOne": [["region"], ["shape"]],
    "additionalProperties": False,
}

_LOAD_CASE = {
    "type": "object",
    "properties": {"name": {"type": "string"},
                   "loads": {"type": "array", "items": {"$ref": "#/$defs/load"}, "minItems": 1},
                   "weight": {"type": "number", "minimum": 0},
                   "cycles": {**_POS, "description": "Times this whole case is applied over the part's life (set on the case, not a load): adds a "
                                                     "fatigue check (e.g. 1e6 for a servo-driven arm)."},
                   "hours": {**_POS, "description": "Hours this load is held over the part's life: adds creep "
                                                    "rupture and crept deflection."},
                   "temperature": {"type": "number",
                                   "description": "C while this load acts (default durability.temperature)."},
                   "safetyFactor": {**_POS, "description": "This case's own safety factor in place of the goal's: "
                                                           "a rare crash or drop held to 1.25 beside service loads "
                                                           "at the goal's 2. The search counts the case's reading "
                                                           "times goal / this."},
                   "impact": {
                       "type": "object",
                       "description": "This case is a hit, not a force: its loads say where and which way (their "
                                      "size is only a first guess), and the force comes from stopping the energy "
                                      "through the crusher and the part together, springs in series "
                                      "(impact.series_impact). A bumper's crush force is measured on a rigid "
                                      "backing; a part that gives takes some of the energy itself and the peak "
                                      "force falls (a soft ring can stop a drop far below the bumper's plateau), "
                                      "while a stiff one sees the bumper's full force. The part's compliance at "
                                      "the patch is read from the design itself (with inertiaRelief: the free "
                                      "body, its own mass and what it carries taking part), so the optimizer "
                                      "and the check both see the coupled force. report.cases[].impact gives the "
                                      "peak force, the energy split and whether the linear reading holds.",
                       "properties": {
                           "energy": {**_POS, "description": "J to stop"},
                           "velocity": {**_POS, "description": "m/s at the hit, in place of energy"},
                           "mass": {**_POS, "description": "kg moving at that velocity. Leave out on a free part "
                                                           "(inertiaRelief): the mass the body presents at the "
                                                           "patch along the hit is worked out from its inertia "
                                                           "(a corner hit spins the body, and stops less than "
                                                           "its whole mass)"},
                           "crusher": {"type": "object", "properties": {
                               "distance": {"type": "array", "items": {"type": "number"}, "minItems": 2,
                                            "description": "mm of crush, rising"},
                               "force": {"type": "array", "items": {"type": "number", "minimum": 0}, "minItems": 2,
                                         "description": "N at each distance"}},
                               "required": ["distance", "force"],
                               "description": "what is between the part and what it hits: a force-crush curve, "
                                              "e.g. a field graph's impact report curve (graded.curve). Left out: "
                                              "nothing in between, the part alone stops the hit"}},
                       "additionalProperties": False}},
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
                   "compressionRatio": {"type": "number", "minimum": 1, "maximum": 3,
                                        "description": "compressive over tensile strength (default: the "
                                                       "material family's, failure.py)"},
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
        "grip": {"type": "number", "minimum": 0, "description": "mm from the seat to the far end (0 for a thread or insert starting right at the seat)."},
        "end": {"enum": ["face", "thread", "none"]},
        "headOn": {"enum": ["this", "mating"]},
        "threadLength": {**_POS, "description": "end 'thread': engagement length past the grip (default 2x the "
                                                "screw's diameter)."},
        "insert": {**_POS, "description": "Heat-set insert or nut outer diameter, mm; else a tapped hole."},
        "preload": {"type": "number", "minimum": 0,
                    "description": "N. Default: a quarter of what the seat can bear in this material, at most "
                                   "what a class 8.8 screw is tightened to."},
        "hole": {**_POS, "description": "Clearance hole diameter, mm (default from size)."},
        "head": {**_POS, "description": "Head (or washer) diameter, mm (default from size: the ISO 4762 socket "
                                        "head, e.g. M3 5.5, M4 7, M5 8.5)."},
        "counterbore": {**_POS, "description": "Diameter of the head and driver keep-out, mm: the counterbore "
                                               "the head sits in (default the seat ring's width, at least head + 1)."},
        "diameter": {**_POS, "description": "The screw's nominal diameter, mm, when not given as size."},
        "wall": {**_POS, "description": "mm of seat ring beyond the head's radius (default 15% of the "
                                        "diameter, at least 1). The head and driver keep-out is as wide as "
                                        "that ring, at least the head's radius + 0.5 mm."},
        "ring": {**_POS, "description": "Seat and face ring thickness along the screw, mm (default 40% of the "
                                        "diameter, at least 2). Bosses a brief asks for go in preserve."},
        "sleeveWall": {**_POS, "description": "end 'thread': wall of the sleeve round the tap, mm."},
        "fit": {"enum": ["close", "normal", "loose", "nominal"],
                "description": "The clearance hole: ISO 273 fine (close), medium (normal, the default) or "
                               "coarse (loose), drawn for the process's hole bias; nominal holds a given hole."},
        "driver": {"type": "number", "minimum": 0,
                   "description": "Length of the head and driver keep-out back from the seat, mm (default 60). "
                                  "Shorten it where it would cut through another interface."},
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
        "maxDeflection": {**_POS, "description": "mm, at the loads: the mean displacement of the loaded "
                                                 "nodes. A load spread over a stiff rim reads low while the span "
                                                 "between sags: cap that with clearances."},
        "clearances": {"type": "array", "items": {
            "type": "object",
            "description": "A travel limit away from the loads: no node of the target moves more than maxTravel "
                           "(keep this face within d of where it is). Held by the search with the safety factor "
                           "and deflection; each case reports its travel (cases[].clearances). " + _TARGET,
            "properties": {
                "region": {"type": "string"}, "shape": {"$ref": "#/$defs/shape"},
                "maxTravel": {**_POS, "description": "mm the region may move"},
                "direction": {"anyOf": [{"type": "string", "pattern": "^[+-]?[xyz]$"},
                                        {"type": "array", "items": {"type": "number"}, "minItems": 3,
                                         "maxItems": 3}],
                              "description": "count only travel along this direction (toward what is there, "
                                             "e.g. '-z' for a fan under the deck); default: any direction"},
                "cases": {"type": "array", "items": {"type": "string"},
                          "description": "load case names it applies in (default: every case)"},
                "name": {"type": "string"}},
            "x-requireOne": [["region"], ["shape"]], "required": ["maxTravel"], "additionalProperties": False}},
        "maxMass": {**_POS, "description": "g: a cap the delivered part is checked against (deliver.passes is "
                                           "false over it). The search already returns the lightest part that "
                                           "meets the other limits, so over the cap means a lighter material, "
                                           "smaller preserves or a lower safety factor."},
        "method": {"enum": ["search", "constrained", "auto"]},
        "confirm": {"enum": ["auto", "voxel", "tet"],
                    "description": "How the lightest passing part is confirmed: voxel (a finer voxel grid), tet "
                                   "(a body-fitted tetrahedral mesh, which also calibrates the search's voxel "
                                   "stress so it steps down to the real target), or auto (tet when installed and "
                                   "applicable)."},
        "outcomes": {"type": "array", "items": {"type": "number", "minimum": 0.02, "maximum": 0.95},
                     "description": "Extra fixed volume fractions for a trade-off gallery."},
        "restarts": {"type": "integer", "minimum": 1, "maximum": 8,
                     "description": "Run the safety-factor/deflection search this many times from different "
                                    "starting fractions and deliver the lightest confirmed part (default 1). One "
                                    "search lands in the local optimum its start leads to (7-17% apart in "
                                    "delivered mass on the eval parts); worth it for a final production part, at "
                                    "N times the time."},
        "minimize": {"enum": ["mass", "printTime", "filament"],
                     "description": "Which passing outcome is delivered: the lightest (default), or (FDM, with a "
                                    "slicer installed) the one that prints fastest or with the least filament, "
                                    "support included, as sliced (graded, with slicer.grade)."},
        "buckling": {"type": "boolean",
                     "description": "Whether the safety factor must hold against buckling too (default true): "
                                    "the part's linear buckling load factor, on a grid of at most 12k cells, "
                                    "must reach safetyFactor. false still reports it (report.buckling)."},
    },
    "x-requireOne": [["volumeFraction"], ["safetyFactor"], ["maxDeflection"]],
    "additionalProperties": False,
}

_VIBRATION = {
    "type": "object",
    "description": "The part's natural frequencies (nTop's natural frequency constraint, and past it): the "
                   "delivered part's modes, clamped at the supports (free-free with inertiaRelief), with the "
                   "carried masses, at the part's own temperature when the study has a thermal block (a warm "
                   "polymer is softer and rings lower). minFrequency is a floor on the first mode; excitations "
                   "are bands of forcing (a rotor's once-per-rev and blade-pass lines over the throttle range) "
                   "that no mode may sit in. The search counts both with the safety factor and deflection; "
                   "with masses, the optimizer also stiffens the part against them shaken along each axis "
                   "(inertial load cases). Needs safetyFactor or maxDeflection in the goal.",
    "properties": {
        "minFrequency": {**_POS, "description": "Hz the first natural frequency must reach"},
        "excitations": {"type": "array", "items": {
            "type": "object",
            "description": "A forcing that sweeps a band of frequency: every mode inside it is driven at "
                           "resonance at some speed (magnified 1 / loss factor of the material), and a mode "
                           "just outside is still magnified 1 / |1 - (f / f_n)^2| at the band's edge. The "
                           "report gives, per excitation, the modes in the band and the worst magnification "
                           "of any mode (report.vibration.excitations).",
            "properties": {
                "name": {"type": "string"},
                "band": {"type": "array", "items": _POS, "minItems": 2, "maxItems": 2,
                         "description": "[lo, hi] Hz the forcing sweeps"},
                "maxAmplification": {"type": "number", "exclusiveMinimum": 1,
                                     "description": "the most any mode may magnify this forcing over its static "
                                                    "response; widens the band on both sides by the distance "
                                                    "that takes (3: modes kept out of 0.87 lo to 1.22 hi). "
                                                    "Default: modes only kept out of the band itself"},
            },
            "required": ["band"], "additionalProperties": False}},
        "masses": {"type": "array", "items": {
            "type": "object", "description": "A carried mass (a motor, a gripper, a battery) on " + _TARGET,
            "properties": {"region": {"type": "string"}, "shape": {"$ref": "#/$defs/shape"},
                           "mass": {**_POS, "description": "kg"}},
            "x-requireOne": [["region"], ["shape"]], "required": ["mass"], "additionalProperties": False}},
        "modes": {"type": "integer", "minimum": 1, "maximum": 20, "description": "modes reported (default 3; "
                  "with excitations, as many more as it takes to reach the top of the highest band)"},
        "inOptimizer": {"type": "boolean", "description": "hold the floor and the bands in the optimizer, on the "
                                                          "design's own modes (free-free with what it carries "
                                                          "for an inertiaRelief part; modalopt.py), and add the "
                                                          "masses' inertial load cases for a held part "
                                                          "(default true; false: only the check)"},
    },
    "x-requireOne": [["minFrequency"], ["excitations"]],
    "additionalProperties": False,
}

_AERO = {
    "type": "object",
    "description": "Rotors or fans whose streams the part sits in (a drone's arms and motor mounts under its "
                   "props, a bracket in front of a fan). Each rotor's stream is followed along its own tubes "
                   "from momentum theory (it narrows behind an open rotor and widens ahead of it), and the first "
                   "material each tube meets drags cd x the dynamic pressure there on the tube's section. The "
                   "optimizer charges that drag in its material budget as the mass a hovering craft would carry "
                   "for the same thrust (drag / 9.81), so a gram of thrust lost trades one for one against a "
                   "gram of structure; the report gives the delivered part's drag per rotor (report.aero), and "
                   "the search weighs outcomes by mass plus that (report.effectiveMass).",
    "properties": {
        "rotors": {"type": "array", "minItems": 1, "items": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "center": {**_VEC3, "description": "the disk's centre, mm"},
                "axis": {**_SIGNED_AXIS, "description": "the way the air goes: 'x', '-z' or a vector"},
                "radius": {**_POS, "description": "disk radius, mm"},
                "hub": {"type": "number", "minimum": 0, "description": "hub (or motor bell) radius left out, mm"},
                "thrust": {**_POS, "description": "N at the operating point the drag is weighed at (hover, for "
                                                  "the mass trade)"},
                "ducted": {"type": "boolean", "description": "a duct with no diffuser: the stream leaves at the "
                                                             "disk's area (default false: an open rotor, whose "
                                                             "wake narrows to 0.71 R)"},
                "cd": {**_POS, "description": "drag coefficient of what the stream meets (default 1.2: a "
                                              "slender bluff member; a streamlined one is about 0.1)"},
                "airDensity": {**_POS, "description": "kg/m^3 (default 1.2)"},
                "reach": {**_POS, "description": "mm along the axis the stream is followed, each way (default "
                                                 "3 radii)"},
            },
            "required": ["center", "axis", "radius", "thrust"], "additionalProperties": False}},
        "inOptimizer": {"type": "boolean", "description": "charge the drag in the optimizer (default true; "
                                                          "false: only the report)"},
        "gramsPerNewton": {**_POS, "description": "what a newton of drag is worth in carried mass (default "
                                                  "102: 1000 / 9.81, a craft hovering on these rotors)"},
    },
    "required": ["rotors"],
    "additionalProperties": False,
}

_MASS_PROPERTIES = {
    "type": "object",
    "description": "Mass properties (nTop's mass, centre of mass and moment of inertia responses): every report "
                   "gives mass, centre of mass and the inertia tensor (report.massProperties). With an axis the "
                   "part turns about (a robot link's joint), the optimizer weighs each cell's inertia about it "
                   "into its cost (inertiaWeight), and maxInertia caps the delivered part's inertia about it.",
    "properties": {
        "axis": {"type": "object", "properties": {
            "through": _VEC3, "direction": _SIGNED_AXIS},
            "required": ["through", "direction"], "additionalProperties": False,
            "description": "the axis the part turns about (mm)"},
        "maxInertia": {**_POS, "description": "kg mm^2 about the axis, held by the search"},
        "inertiaWeight": {"type": "number", "minimum": 0, "maximum": 1,
                          "description": "share of a cell's cost in the optimizer that is its inertia about the "
                                         "axis rather than its volume (default 0.5 with an axis): material near "
                                         "the axis comes cheaper"},
        "centerOfMass": {"type": "object", "properties": {
            "target": {"type": "array", "minItems": 3, "maxItems": 3,
                       "items": {"type": ["number", "null"]},
                       "description": "mm [x, y, z]; null leaves that axis free (balance over a z axis: "
                                      "[0, 0, null])"},
            "tolerance": {**_POS, "description": "mm each held axis may be off (default 1)"}},
            "required": ["target"], "additionalProperties": False,
            "description": "where the part's centre of mass must sit (a handheld that balances in the hand, a "
                           "rotor, a part on a gimbal): the optimizer pulls it there and the report checks it"},
    },
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
        "constrain": {"type": "boolean", "description": "The search sizes the part for the life its load "
                                                        "cases declare (fatigue and creep; default true). "
                                                        "false: report only."},
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
            {"const": "natural", "description": "still air: natural convection and radiation, solved per face "
                                                "from its own temperature, direction and what it faces"},
            {"type": "object", "description": "still air with its settings",
             "properties": {"type": {"const": "natural"},
                            "emissivity": {"type": "number", "exclusiveMinimum": 0, "maximum": 1,
                                           "description": "the surface's (default 0.9 polymer, 0.25 bare metal)"},
                            "up": {"description": "the direction opposing gravity (default '+z')"},
                            "pressure": {**_POS, "description": "bar absolute (default 1.013)"}},
             "required": ["type"], "additionalProperties": False},
            {"type": "number", "minimum": 0, "description": "W/(m² K) off every exposed surface"},
            {"type": "array", "items": {"type": "object", "properties": {
                **_THERMAL_REGION,
                "h": {"type": "number", "minimum": 0,
                      "description": "W/(m² K): ~5-10 still air, 20-50 a fan, 500+ water"},
                "fluid": {"type": "number", "description": "C of the fluid (default ambient)"}},
                "required": ["h"], "additionalProperties": False}}],
            "description": "Heat carried off the surface the design exposes. Default \"natural\": still air, "
                           "the coefficient a result of each face's temperature and surroundings "
                           "(convection.py; typically 6-14 W/(m² K) on a polymer, radiation included). A "
                           "number or a list fixes it: a fan, a liquid, a measured value. An entry without "
                           "region or shape covers every surface."},
        "conductivity": {**_POS, "description": "W/(m K) override (default the material card's)."},
        "expansion": {"anyOf": [{"type": "boolean"}, {"type": "number", "minimum": 0}],
                      "description": "true (default): the material card's expansion coefficient; a number: 1/K "
                                     "(e.g. 7e-5); false: temperature softens the part but does not grow it."},
        "soften": {"type": "boolean",
                   "description": "true (default): moduli and strength fall with temperature (the material "
                                  "card's retention curve)."},
        "specificHeat": {**_POS, "description": "J/(kg K) override (default the material card's); with the "
                                                "density it sets how fast the part heats and cools in a cycle."},
        "cycle": {"type": "object",
                  "description": "A duty cycle instead of steady heat: the heat inputs are on for the first 'on' "
                                 "seconds of every 'period'. The repeating temperature cycle is solved; the part "
                                 "is designed for its hot moment, checked at its hot and cold moments, and each "
                                 "cycle counts toward thermal-cycling fatigue (durability).",
                  "properties": {
                      "period": {**_POS, "description": "s per cycle"},
                      "on": {**_POS, "description": "s the heat is on in each cycle (less than period)"},
                      "cycles": {**_POS, "description": "cycles over the part's life, for fatigue (e.g. 3e4: "
                                                        "ten a day for ten years)"},
                      "steps": {"type": "integer", "minimum": 8,
                                "description": "time steps per period (default 48)"},
                      "fatigueWeight": {"type": "number", "minimum": 0,
                                        "description": "How much the optimizer also designs against the "
                                                       "hot-to-cold swing (thermal-cycling fatigue): a p-norm of "
                                                       "the swing's stress over strength retention, weighted "
                                                       "against the hot moment's objective at the start. 0 "
                                                       "(default): hot moment only. Experimental: it helped the "
                                                       "motor mount at 12k cells but not at 45k, so verify the "
                                                       "result. Adding material does not help "
                                                       "thermal cycling (stiffer parts hold more expansion)."},
                      "fatigueSafetyFactor": {"type": "number", "exclusiveMinimum": 0,
                                              "description": "Design against thermal-cycling fatigue as a "
                                                             "constraint: the optimizer (MMA) holds the "
                                                             "hot-to-cold swing's stress under the material's "
                                                             "fatigue strength at `cycles`, over this safety "
                                                             "factor, while it minimizes the hot objective. "
                                                             "Needs cycles; not with fatigueWeight. "
                                                             "Experimental: motor mount at 12k cells, SF 3.5 "
                                                             "took the verified fatigue SF 2.67 -> 3.00 at equal "
                                                             "hot SF, but at 45k it came out lower (2.45 -> "
                                                             "2.05), so verify the result."}},
                  "required": ["period", "on"], "additionalProperties": False},
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
        "bedAdhesion": {"type": "number", "exclusiveMinimum": 0,
                        "description": "fdm: MPa of bond tension the first layer holds before it lets go of the "
                                       "bed (corners lift and the part re-balances; default: the value fitted to "
                                       "Armillotta 2018)"},
        "printLayer": {"type": "number", "minimum": 0,
                       "description": "fdm: the printed layer height, mm (default 0.2). Warp grows as layers "
                                      "thin, so builds simulated at thicker voxel layers are extrapolated to "
                                      "it; 0 turns that off"},
        "plateCells": {"type": "integer", "minimum": 1,
                       "description": "lpbf: voxel layers modelling the build plate, its stiffness scaled to the "
                                      "full plate (default 2; the default k is fitted with it)"},
        "recoaterGap": {**_POS, "description": "lpbf: mm of lift at the build plane that counts as recoater "
                                               "contact risk (default 0.05)"},
        "compensate": {"type": "boolean", "description": "also write <name>-compensated.stl, pre-distorted so "
                                                         "the print lands on the design"},
        "iterations": {"type": "integer", "minimum": 1, "description": "compensation rounds (default 2)"},
        "cells": {"type": "integer", "minimum": 1000, "description": "voxels in the simulated part (default 60000)"},
        "warpWeight": {"type": "number", "minimum": 0,
                       "description": "lpbf/fdm: design against build distortion too. The optimizer minimizes "
                                      "compliance * (1 + warpWeight * D / D0), D the mean squared as-built "
                                      "displacement of the layer-by-layer build on the plate (D0 at the start); "
                                      "0 (default) only verifies. Needs a grid-axis buildDirection. About 1 trades "
                                      "stiffness for a large cut in distortion and recoater lift"},
        "warpLayer": {**_POS, "description": "mm per analysis layer of the warp term (default 4 cells)"},
        "maxDistortion": {**_POS, "description": "lpbf/fdm: a limit, mm, on the largest as-built displacement "
                                                 "(on the plate, end of the build). The optimizer holds it as a "
                                                 "constraint and the search only passes designs that meet it and "
                                                 "the safety factor; report 'warp'. Needs a grid-axis "
                                                 "buildDirection"},
        "maxRecoaterLift": {**_POS, "description": "lpbf: a limit, mm, on how far the part rises at the build "
                                                   "plane as each layer goes down (the recoater's clearance, "
                                                   "~0.05 mm); held like maxDistortion"},
        "brim": {"type": "number", "minimum": 0, "description": "fdm: brim width, mm (default 0): more bed hold under the footprint's "
                                         "edge (an estimate, not validated)"},
        "printSetup": {"oneOf": [{"type": "boolean"}, {"type": "object", "properties": {
            "materials": {"type": "array", "items": {"type": "string"},
                          "description": "materials to try (default the study's)"},
            "orientations": {"type": "array", "items": {"type": "string"},
                             "description": "build directions to try (default the six grid axes)"},
            "bedTemperatures": {"type": "array", "items": {"type": "number"},
                                "description": "bed temperatures, C (default a few around each material's usual)"},
            "brims": {"type": "array", "items": {"type": "number", "minimum": 0},
                      "description": "brim widths to try, mm (default [0, 5])"},
            "maxBed": {"type": "number", "description": "the printer's hottest bed, C (default 110)"},
            "reoptimize": {"type": "boolean", "description": "when no outcome passes a warp limit and no setup "
                                                             "rescues the shape as it is, re-run the study for "
                                                             "the best setup (default true)"}},
            "additionalProperties": False}],
            "description": "fdm: rank print setups (orientation, material, bed temperature, brim) for the "
                           "delivered part on strength and warp (report 'printSetup'; printsetup.py)"},
        "warpInOptimizer": {"type": "boolean",
                            "description": "carry maxDistortion / maxRecoaterLift in the optimizer (default true "
                                           "for lpbf; false for fdm, whose limits are checked on each outcome's "
                                           "re-simulated part: in the optimizer they cost strength and did not "
                                           "lower the warp)"},
        "warpCoarsen": {"type": "integer", "minimum": 1,
                        "description": "the build model in the optimizer runs on a grid this many times "
                                       "coarser (default 2 on grids of 4000 cells or more, else 1)"}},
     "additionalProperties": False}],
    "description": "Simulate the build of the delivered part (report 'buildSim'): layer-by-layer inherent-strain "
                   "residual stress and distortion for lpbf (validated on NIST AM-Bench 2018), fdm warp and bed "
                   "peel (ABS/ASA/PETG...), sls/mjf differential shrink; optionally a pre-distorted STL."}

_EXPLORE_METHODS = ["unrestricted", "fdm", "sls", "mjf", "sla", "dlp", "jetting", "volumetric", "lpbf", "cast", "milled-2.5axis",
                    "milled-3axis", "milled-5axis"]
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
                    "latticeStiffness": {"anyOf": [{"type": "number", "exclusiveMinimum": 0, "maximum": 1.5},
                                                   {"enum": ["as-printed", "designed"]}],
                                         "description": "printed lattice stiffness over the designed geometry's in "
                                                        "the homogenized check: 'as-printed' (default: fitted to 33 "
                                                        "published compression tests per process, lattice_print.py), "
                                                        "'designed' (1), or a measured factor"},
                    "anisotropic": {"anyOf": [{"type": "boolean"}, {"const": "bond"}]},
                    "interlayer": {"type": "number", "exclusiveMinimum": 0, "maximum": 1},
                    "interlayerShear": {"type": "number", "exclusiveMinimum": 0, "maximum": 1,
                                        "description": "the layer bond's shear strength over the in-plane tensile "
                                                       "strength (default interlayer / sqrt(3))"},
                    "minFeature": _POS, "overhangs": {"type": "boolean"},
                    "minHole": {**_POS, "description": "mm: narrowest hole or gap the optimizer may leave "
                                                       "(default FDM/SLS/MJF 2, SLA/DLP/LPBF 1, volumetric 0.6)"},
                    "bridgeSpan": {**_POS, "description": "mm: longest unsupported bridge the printer prints cleanly "
                                   "(default: 15 FDM, 5 resin, 2 LPBF); longer spans count as needing support"},
                    "escapeHole": {**_POS, "description": "mm: smallest powder escape / resin drain hole "
                                                          "(default SLS 4, MJF 5, SLA/DLP 3.5)"},
                    "holeBias": {"type": "number", "description": "mm: as-built minus drawn hole diameter on "
                                 "this printer, measured (default per process in fits.py: FDM -0.2 along the "
                                 "build, -0.35 across; SLS -0.15; MJF -0.1; resin -0.05; LPBF -0.1/-0.2)"},
                    "holeBiasAcross": {"type": "number", "description": "mm: as holeBias, for holes whose "
                                       "axis runs across the layers (default: holeBias when given, else per "
                                       "process). The calibration kit's H-Z coupon measures it"},
                    "holeTolerance": {**_POS, "description": "mm: +- scatter of a hole's as-built diameter on "
                                      "this printer, measured (default per process, growing with diameter)"}},
     "required": ["type"], "additionalProperties": False},
], "description": "How it is made: sets a minimum member, overhang control (FDM, LPBF metal) and the layer-bond check. "
                  "lpbf = laser powder-bed fusion metal (AlSi10Mg, Ti64, 316L). sls/mjf (PA12, PA11, PA12-GB, TPU) "
                  "and sla/dlp (resins) get process checks on the delivered part (report 'process': walls, trapped "
                  "powder or resin, warp-prone panels; resin supports, peel area and suction cups)."}

_LATTICE_PROPS = {
    "type": {"enum": ["gyroid", "schwarz-p", "diamond", "lidinoid", "split-p", "neovius"],
             "description": "TPMS sheet (default gyroid). Per gram, diamond is the stiffest and strongest, "
                            "Schwarz P the weakest and has flat ceilings (FDM); lidinoid and split P are "
                            "finer sheets (twice the surface a cell), Neovius a thick-noded Schwarz P."},
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
                "safetyFactor": {**_POS, "description": "the knock's own safety factor: on its stored "
                                                        "energy, and on materials and bonds at its peak force "
                                                        "(an accidental load; default the goal's / "
                                                        "interfaceSafetyFactor)"}},
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
        "interlock": {"oneOf": [{"type": "boolean"}, {"type": "object", "properties": {
            "pattern": {"enum": ["beams"]},
            "depth": {**_POS, "description": "mm of interlocking band straddling the interface (default 2)"},
            "beamWidth": {**_POS, "description": "mm (default two extrusion widths, 0.8)"},
            "layers": {"type": "integer", "minimum": 1, "description": "print layers per beam (default 2)"},
            "layerHeight": {**_POS, "description": "mm per beam layer, instead of layers"},
            "gain": {"type": "number", "minimum": 1,
                     "description": "interlocked / flat bond strength credited (default 2.5 at 2 mm, sourced)"},
            "pairs": {"type": "array", "items": {"type": "string"}, "description": "'a+b' pairs (default all)"}},
            "additionalProperties": False}],
            "description": "Interlock the materials where they meet (an interlaced beam lattice, as Cura's / "
                           "Bambu's interlocking beams): the delivered bodies interpenetrate and the bond is "
                           "credited with the measured gain, capped by what the lattice carries (interlock.py)"},
        "bondInTopology": {"type": "boolean", "description": "experimental: constrain the bond while the "
                                                            "layout forms (from projection 4), not only in "
                                                            "the final bond stage"},
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

_FORCE = {"type": "number", "exclusiveMinimum": 0}
_SWITCH_CLEARANCE = {
    "type": "object",
    "description": "a fixed gap for a region ({region | shape, gap}), or the room to a component measured per node "
                   "along the press ({component, margin}; region/shape then limit which nodes count)",
    "properties": {
        "id": {"type": "string", "description": "name in the report (default the component or region)"},
        "region": {"type": "string"}, "shape": {"$ref": "#/$defs/shape"},
        "gap": {**_POS, "description": "mm the region may move along the press at the fire force"},
        "component": {"anyOf": [{"type": "string", "description": "a keepOut id: its shape before any insertFrom "
                                                                  "sweep"}, {"$ref": "#/$defs/shape"}]},
        "margin": {"type": "number", "minimum": 0, "description": "mm kept clear of the component (default 0.1)"},
        "reach": {**_POS, "description": "mm, how far below a node to look for the component (default 10)"}},
    "additionalProperties": False}
_SWITCH = {
    "type": "object",
    "description": "`loadpath switch`: a target-force push pad (switch.py). Minimize volume so a press anywhere on "
                   "the touch area fires a switch under the output at a chosen force: a ceiling (any gradient) "
                   "and a floor per press point, with optional press travel, skin clearance and hold cases. Pair "
                   "it with `skin` so the exterior is kept as a thin cosmetic layer and the rest is design "
                   "space. `run` ignores it.",
    "properties": {
        "unit": {"enum": ["N", "gf"], "description": "force unit of max/min/trigger (default N)"},
        "output": {"type": "object", "properties": {
            "region": {"type": "string", "description": "the plunger face: a preserve id"},
            "shape": {"$ref": "#/$defs/shape"},
            "direction": {"type": "string", "pattern": "^[+-]?[xyz]$", "description": "stroke (default -z)"},
            "gap": {"type": "number", "minimum": 0, "description": "mm, plunger to switch stem at rest"},
            "trigger": {"type": "number", "minimum": 0, "description": "the switch's operating force"},
            "travel": {"type": "number", "minimum": 0, "description": "mm, the switch's own travel to the click"},
            "bearsOn": {"type": "object", "description": "what the switch body sits on, so its give counts in the "
                                                         "fire force (default: rigid ground). region/shape: the "
                                                         "seat in this mesh (a preserved flap or boss): the "
                                                         "closure is the plunger's travel less the seat's. part: a "
                                                         "separate fixed part (a flap standing on the desk), its "
                                                         "seat's compliance solved on its own on cut cells. "
                                                         "compliance: a measured one, mm/N. Any may be combined",
                        "properties": {
                            "region": {"type": "string"}, "shape": {"$ref": "#/$defs/shape"},
                            "part": {"type": "object", "properties": {
                                "shape": {"$ref": "#/$defs/shape"},
                                "seat": {"$ref": "#/$defs/shape", "description": "where the switch body sits"},
                                "supports": {"type": "array", "items": {"$ref": "#/$defs/shape"}, "minItems": 1,
                                             "description": "where the part is held (its foot on the desk)"},
                                "material": {"type": "string", "description": "default the study's"},
                                "pitch": {**_POS, "description": "mm cut-cell grid (default 0.4)"}},
                                "required": ["shape", "seat", "supports"], "additionalProperties": False},
                            "compliance": {"type": "number", "minimum": 0, "description": "mm/N, measured"}},
                        "additionalProperties": False}},
            "additionalProperties": False},
        "presses": {"type": "array", "items": {"type": "object", "properties": {
            "at": {"type": "array", "items": {"type": "number"}, "minItems": 2, "maxItems": 3},
            "max": _FORCE, "min": _FORCE}, "required": ["at"], "additionalProperties": False}},
        "field": {"type": "object", "properties": {
            "center": {"type": "array", "items": {"type": "number"}, "minItems": 2, "maxItems": 2},
            "radius": _POS,
            "rings": {"type": "integer", "minimum": 1}, "perRing": {"type": "integer", "minimum": 1},
            "max": {"anyOf": [_FORCE, {"type": "array", "items": _FORCE, "minItems": 2, "maxItems": 2}],
                    "description": "one value, or [at the centre, at the radius]: a radial gradient"},
            "min": {"anyOf": [_FORCE, {"type": "array", "items": _FORCE, "minItems": 2, "maxItems": 2}]}},
            "required": ["center", "radius"], "additionalProperties": False,
            "description": "press points on rings round a centre, with a radial force gradient"},
        "pressDirection": {"type": "string", "pattern": "^[+-]?[xyz]$", "description": "default: the output's"},
        "pressRadius": {**_POS, "description": "mm, the fingertip disc (default 1)"},
        "maxTravel": {**_POS, "description": "mm of press travel at the click (crispness)"},
        "clearance": {"anyOf": [{"$ref": "#/$defs/switchClearance"},
                                {"type": "array", "items": {"$ref": "#/$defs/switchClearance"}, "minItems": 1}],
                      "description": "how far the skin and interior may move toward what is under them at the fire "
                                     "force: one entry, or one per component (each node gets the room to the "
                                     "component under it)"},
        "hold": {"type": "array", "items": {"type": "object", "properties": {
            "case": {"type": "string", "description": "a load case name"},
            "maxDeflection": {**_POS, "description": "mm, force-weighted mean travel under the case"}},
            "required": ["case", "maxDeflection"], "additionalProperties": False},
            "description": "load cases (squeeze, drop) whose deflection is capped: the shape holds"},
        "pitches": {"type": "array", "items": _POS, "minItems": 1,
                    "description": "coarse to fine grids, mm; each starts from the last one's design"},
        "iterations": {"anyOf": [{"type": "integer", "minimum": 1},
                                 {"type": "array", "items": {"type": "integer", "minimum": 1}}]},
        "nonlinear": {"type": "boolean", "description": "re-read one press per distinct target with large "
                                                        "deflections (default true; about half a minute a press on "
                                                        "a 48k-dof push pad with PARDISO, loadpath[fast])"},
        "nonlinearPitch": {**_POS, "description": "mm grid of the large-deflection check (default the first "
                                                  "pitch); the design is read onto it"},
        "skinReview": {"anyOf": [{"type": "boolean"}, {"type": "object", "properties": {
            "minWall": {**_POS, "description": "mm, the thinnest skin the designer will accept (default: the "
                                               "process's or material's minimum wall, else minMember)"},
            "radii": {"type": "array", "items": _POS, "minItems": 1,
                      "description": "mm, the local skin changes tried round each press (default [4, 8])"},
            "gridCheck": {"anyOf": [{"type": "boolean"}, _POS],
                          "description": "read the bare skin and the delivered mesh on cut cells "
                                         "(switch_lattice.readback; skinReview.grid.check) and lead the summary "
                                         "with that. Default: whenever the skin is a density floor or the pitch is "
                                         "over 0.75x the skin or minMember, at half the thinner of the two; a "
                                         "number sets the cut-cell pitch, false skips it"}},
            "additionalProperties": False}],
            "description": "report skinReview (default true): every press the fixed exterior keeps out of its "
                           "window, where, by how much, and the skin thickness, local skin change or target "
                           "change that would fix it. Suggestions only: the exterior is never edited. "
                           "`loadpath switch --skin-review` reads the bare skin alone, before designing"},
        "lattice": {"type": "object", "description": "a graded lattice inside the skin instead of ribs "
                                                     "(switch_lattice.py, `loadpath switch --lattice`): one cell "
                                                     "size, its wall or strut graded per cell to hit each window, "
                                                     "then read back on the printable mesh (cut cells) and the "
                                                     "windows corrected by that reading for the next round",
                    "properties": {
                        "type": {"type": "string", "description": "a TPMS sheet (gyroid, diamond, schwarz-p) or a "
                                                                  "strut cell (octet, bcc, kelvin, ...)"},
                        "cell": {**_POS, "description": "mm, the lattice period (default 6)"},
                        "wallMin": {**_POS, "description": "mm, thinnest wall or strut (default minMember)"},
                        "wallMax": {**_POS, "description": "mm, thickest (default 0.4 cell)"},
                        "mode": {"enum": ["fill", "where-needed"],
                                 "description": "fill: lattice in the whole interior, only its density graded; "
                                                "where-needed: also where there is lattice at all (default fill)"},
                        "gradeRadius": {**_POS, "description": "mm, how fast the density may vary (default half "
                                                               "a cell)"},
                        "regionRadius": {**_POS, "description": "mm, where-needed's presence filter (default 0.75 "
                                                                "cell, so a lattice region is about a cell across)"},
                        "rounds": {"type": "integer", "minimum": 1,
                                   "description": "optimize / read back / correct rounds (default 2)"},
                        "readbackPitch": {**_POS, "description": "mm grid of each round's read-back (default 0.5)"},
                        "finalPitch": {**_POS, "description": "mm grid of the final read-back (default "
                                                              "readbackPitch)"}},
                    "additionalProperties": False}},
    "required": ["output"],
    "x-requireOne": [["presses"], ["field"]],
    "additionalProperties": False,
}

_DIRECTION = {"type": "string", "pattern": "^[+-]?[xyz]$"}
_WHERE = {"type": "object", "properties": {
    "region": {"type": "string", "description": "a preserve or keep-out id (`skin` is the exterior skin)"},
    "shape": {"$ref": "#/$defs/shape"},
    "face": {**_DIRECTION, "description": "the exterior facing this way: the outermost node of every column"},
    "within": {"type": "number", "minimum": 0,
               "description": "with a shape alone: the part within this many mm of the shape (where a component "
                              "touches it: a board's rails, stops and boss)"}},
    "additionalProperties": False,
    "description": "where: a region, a shape, or an exterior face; a region or a face with a shape too is only "
                   "the part of it inside the shape (the skin where the hand is)"}
_WHERES = {"type": "array", "items": _WHERE, "minItems": 1, "description": "several places at once"}
_RANGE = {"anyOf": [{"type": "array", "items": _POS, "minItems": 2, "maxItems": 2},
                    {"type": "object", "properties": {"min": _POS, "max": _POS}, "additionalProperties": False}],
          "description": "[min, max], or {min} / {max} for one side only"}
_FEEL_TARGET = {"type": "object", "properties": {
    "kind": {"enum": ["squeeze", "drop", "isolate", "touch"]},
    "name": {"type": "string", "description": "touch: the zone's name, as `contrast` refers to it"},
    "hold": {**_WHERE, "description": "squeeze, isolate, touch: where the product is held (all three "
                                      "directions; touch default: the face opposite the press)"},
    "press": {**_WHERE, "description": "squeeze: where the hand presses"},
    "direction": {**_DIRECTION, "description": "squeeze, touch: the press direction (touch default -z); drop: the "
                                               "fall direction (default: toward the landing face)"},
    "force": {**_POS, "description": "squeeze: N the hand presses with; touch: N of a fingertip press"},
    "travel": {**_RANGE, "description": "squeeze, touch: mm it gives at `force`, [min, max]; {max} alone is a "
                                        "firmness floor (at least `force` N to push it that far)"},
    "stiffness": {**_RANGE, "description": "squeeze, touch: N/mm, in place of force and travel"},
    "zone": {"$ref": "#/$defs/shape", "description": "touch: the place on the exterior (fingertip presses are "
                                                     "sampled where a press along `direction` meets the part "
                                                     "inside it)"},
    "feel": {"enum": ["pliable", "firm", "rigid"],
             "description": "touch: a preset window under a 4 mm-radius fingertip: pliable 1-4 N/mm (a 2 N touch "
                            "sinks 0.5-2 mm), firm >= 15 N/mm (a 5 N press moves it under 0.33 mm), rigid "
                            ">= 50 N/mm; stiffness or force/travel override the side they give"},
    "fingertip": {**_POS, "description": "touch: mm radius of the pressed disc (default 4)"},
    "spacing": {**_POS, "description": "touch: mm between sampled presses (default 6)"},
    "inset": {"type": "number", "minimum": 0,
              "description": "touch: sampled presses are at least this far inside the zone's edge, mm (default "
                             "half the fingertip, at most a quarter of the spacing)"},
    "points": {"type": "array", "items": {"type": "array", "items": {"type": "number"}, "minItems": 2,
                                          "maxItems": 3}, "minItems": 1,
               "description": "touch: the presses, in place of sampling the zone (2 coordinates across the press "
                              "direction, or 3)"},
    "skinThickness": {"anyOf": [_POS, {"const": "auto"}],
                      "description": "touch: mm of skin under the zone, the exterior unchanged (skin.zones); "
                                     "\"auto\": the thickest skin whose bare presses sit under 0.75 of the "
                                     "ceiling, never under the process's minimum wall (flagged when the zone "
                                     "would need less)"},
    "skinBlend": {"type": "number", "minimum": 0, "description": "touch: mm the skin's thickness ramps over at "
                                                                 "the zone's edge (default 2)"},
    "lands": {**_WHERE, "description": "drop: where it lands (usually {face})"},
    "mass": {"anyOf": [_WHERE, _WHERES], "description": "drop: the part to cushion (its mount; a list: every "
                                                          "place it is held)"},
    "mount": {"anyOf": [_WHERE, _WHERES], "description": "isolate: the mount of the part to keep vibration from "
                                                           "(a list: every place it is held, e.g. rails and a "
                                                           "boss on a lid)"},
    "rigid": {"type": "boolean", "description": "drop, isolate: the part is stiff next to its mounts (a circuit "
                                                "board, a battery, a capsule), so every place it is held moves "
                                                "with it: its stiffness is the sum of its holds', not their "
                                                "average (default false: the load spread evenly)"},
    "kg": {**_POS, "description": "drop, isolate: the part's mass"},
    "height": {**_POS, "description": "drop: m"},
    "maxG": {**_POS, "description": "drop: peak deceleration of the part, in g"},
    "maxTravel": {**_POS, "description": "drop: mm the part may move before it hits something; isolate: mm "
                                         "under a `shockG` knock"},
    "band": {**_POS, "description": "isolate, soft mount: Hz, the lowest frequency to isolate"},
    "upTo": {**_POS, "description": "isolate, stiff mount: Hz the handling band reaches; the part's resonance "
                                    "is kept over it"},
    "maxAmplification": {"type": "number", "exclusiveMinimum": 1,
                         "description": "isolate with upTo: largest amplification in the band (default 1.25)"},
    "maxTransmissibility": {"type": "number", "exclusiveMinimum": 0, "exclusiveMaximum": 1,
                            "description": "isolate: at `band` and above (default 0.5)"},
    "directions": {"type": "array", "items": {"enum": ["x", "y", "z"]}, "minItems": 1,
                   "description": "isolate: default all three"},
    "shockG": {**_POS, "description": "isolate: the knock `maxTravel` is read at, in g (default 1: its weight)"}},
    "required": ["kind"],
    "additionalProperties": False}
_COOLING_COMPONENT = {
    "type": "object",
    "description": "A part inside that makes heat or carries it: a keep-out `region` (the part as drawn, not its "
                   "assembly path) or its own `shape`.",
    "properties": {
        "region": {"type": "string", "description": "the keepOut (or preserve) id of the part"},
        "shape": {"$ref": "#/$defs/shape"},
        "watts": {"type": "number", "minimum": 0, "description": "heat it gives off, W (default 0: it only "
                                                                 "conducts, a battery or a board)"},
        "maxTemp": {"type": "number", "description": "C its hottest point must stay under (a die's junction "
                                                     "limit less the package's own rise, a battery's 45-60 C)"},
        "conductivity": {**_POS, "description": "W/(m K) (default 20: a package on a copper-filled board)"},
        "heatCapacity": {**_POS, "description": "J/K it holds per K, for a `cycle` (default 1.8 J/(cm^3 K) "
                                                "times its volume; a battery is its mass times about 1000 "
                                                "J/(kg K))"}},
    "additionalProperties": False}
_COOLING_CYCLE = {
    "type": "object",
    "description": "Power over time instead of a steady state: the limits hold at each point's hottest moment. "
                   "A burst after idling, charging, a duty cycle (`repeat`).",
    "properties": {
        "phases": {"type": "array", "minItems": 1, "items": {"type": "object", "properties": {
                       "seconds": _POS,
                       "watts": {"type": "object", "additionalProperties": {"type": "number", "minimum": 0},
                                 "description": "W per component id in this phase (others keep their `watts`)"},
                       "airflow": {"type": "object", "properties": {
                                       "speed": {"type": "number", "minimum": 0, "description": "m/s (0: still air)"},
                                       "length": {**_POS, "description": "mm the surface runs along the flow"}},
                                   "required": ["speed", "length"], "additionalProperties": False,
                                   "description": "the air over the outside in this phase (as skin.airflow): a "
                                                  "product that waits in still air and is cooled once it moves "
                                                  "(a drone idling on the ground, then flying). Others keep "
                                                  "the skin's h or airflow"},
                       "h": {**_POS, "description": "instead of airflow: W/(m^2 K) of the outside in this phase"},
                       "steps": {"type": "integer", "minimum": 1, "description": "time steps (default `steps`)"}},
                   "required": ["seconds"], "additionalProperties": False}},
        "start": {"enum": ["first", "ambient"],
                  "description": "first (default): the steady state of the first phase's power in its air; "
                                 "ambient: cold"},
        "repeat": {"type": "boolean", "description": "the repeating state of the phases, from the mean power"},
        "cycles": {"type": "integer", "minimum": 1, "description": "with repeat: cycles run (default 3), the "
                                                                   "last one read"},
        "steps": {"type": "integer", "minimum": 1, "description": "time steps per phase (default 12)"}},
    "required": ["phases"], "additionalProperties": False}
_COOLING = {
    "type": "object",
    "description": "`loadpath cooling`: design the inside of a fixed exterior around the heat its parts make "
                   "(cooling.py). Minimum volume, with each component's hottest point under its `maxTemp` and "
                   "the touched skin under `skin.maxTemp` (or a `skin.contact` class): the interior grows pads "
                   "and ribs that carry heat to the skin and spread it. Enclosed air conducts (still air in "
                   "gaps of a few mm), the outside loses heat by natural convection and radiation, and vents "
                   "the exterior already has carry heat by stack flow. Pair it with `skin`; `switch` and `feel` "
                   "blocks in the same study are designed together with it. A study with a cooling block "
                   "needs no supports or load cases. `run` ignores it.",
    "properties": {
        "ambient": {"type": "number", "description": "room temperature, C (default 25; 35 for a hot day)"},
        "components": {"type": "array", "items": _COOLING_COMPONENT, "minItems": 1},
        "skin": {"type": "object", "properties": {
                     "maxTemp": {"type": "number", "description": "C the touched surface must stay under"},
                     "contact": {"enum": ["continuous", "held", "brief"],
                                 "description": "instead of maxTemp: plastic touch limits after IEC 62368-1 "
                                                "(43, 48, 60 C)"},
                     "shape": {"$ref": "#/$defs/shape", "description": "where it is touched (default: all of "
                                                                         "the outside)"},
                     "h": {**_POS, "description": "W/(m^2 K) to the room, convection and radiation (default 10)"},
                     "airflow": {"type": "object", "properties": {
                                     "speed": {"type": "number", "minimum": 0, "description": "m/s of air along "
                                               "the outside: flight speed, a rotor's or fan's wash"},
                                     "length": {**_POS, "description": "mm the surface runs along the flow"}},
                                 "required": ["speed", "length"], "additionalProperties": False,
                                 "description": "instead of h: forced convection off the outside (flat plate "
                                                "mean, mixed with natural convection) plus radiation"}},
                 "additionalProperties": False},
        "air": {"type": "object", "properties": {
                    "conductivity": {**_POS, "description": "W/(m K) of the air inside (default 0.03)"}},
                "additionalProperties": False},
        "vents": {"type": "array", "items": {"type": "object", "properties": {
                      "id": {"type": "string"},
                      "shape": {"$ref": "#/$defs/shape", "description": "an opening the exterior already has"},
                      "area": {**_POS, "description": "open area, mm^2 (default: the outside faces it covers)"}},
                  "required": ["shape"], "additionalProperties": False},
                  "description": "openings for stack flow: the lowest let air in, the highest out"},
        "gravity": {"type": "string", "pattern": "^[+-]?[xyz]$",
                    "description": "the way down in use, for the vents' stack (default -z)"},
        "conductivity": {**_POS, "description": "W/(m K) of the printed material (default the material card's)"},
        "cycle": _COOLING_CYCLE,
        "robust": {"oneOf": [{"type": "boolean"}, {"type": "number", "minimum": 0, "maximum": 0.4}],
                   "description": "hold the limits on the design eroded and dilated by this much of the "
                                  "projection (true: 0.15; default off)"},
        "margin": {"type": "number", "minimum": 0,
                   "description": "K under each limit the design aims for (default 0): the design grid's error, "
                                  "as the delivered readings on finer grids show it"},
        "pitches": {"type": "array", "items": _POS, "minItems": 1,
                    "description": "mm, coarse to fine (without a switch block)"},
        "iterations": {"oneOf": [{"type": "integer", "minimum": 1},
                                 {"type": "array", "items": {"type": "integer", "minimum": 1}}]},
        "verifyPitches": {"type": "array", "items": _POS, "minItems": 1,
                          "description": "mm grids the delivered mesh's temperatures are read on (default "
                                         "0.7 and 0.5 of the design pitch)"},
        "padGap": {"type": "number", "minimum": 0,
                   "description": "mm: a gap up to this wide between a component and the delivered plastic is "
                                  "read as the thermal pad (or paste) the model assumes (default 1; 0 reads it "
                                  "as air)"}},
    "required": ["components"],
    "additionalProperties": False}
_FEEL = {
    "type": "object",
    "description": "`loadpath feel`: design the inside of a fixed exterior for how it feels (feel.py). Minimum "
                   "volume, with each target a stiffness window between where it is held and where the load "
                   "comes in: `touch` (a zone of the exterior under a fingertip: pliable, firm, or a window; "
                   "`contrast` sets how distinctly zones differ; `skinThickness` thins the skin under a zone), "
                   "`squeeze` (a grip gives `travel` mm at `force` N), `drop` (a part of `kg` "
                   "dropped `height` m sees under `maxG` and moves under `maxTravel`), `isolate` (handling "
                   "vibration from `band` Hz up reaches a part at under `maxTransmissibility`, a soft mount; or "
                   "its resonance stays over a handling band `upTo` Hz, a stiff one). Pair it with "
                   "`skin`; a `switch` block in the same study is designed together with it. A study with a "
                   "feel block needs no supports or load cases. `run` ignores it.",
    "properties": {
        "targets": {"type": "array", "items": _FEEL_TARGET, "minItems": 1},
        "contrast": {"type": "array", "items": {"type": "object", "properties": {
                         "softer": {"type": "string", "description": "a touch zone's name"},
                         "firmer": {"type": "string", "description": "a touch zone's name"},
                         "ratio": {"type": "number", "exclusiveMinimum": 1,
                                   "description": "the firmer zone's softest press over the softer zone's "
                                                  "stiffest (default 2)"}},
                         "required": ["softer", "firmer"], "additionalProperties": False},
                     "description": "touch zones that must feel distinctly different: every press in `firmer` "
                                    "at least `ratio` times as stiff as every press in `softer`"},
        "skinPitch": {**_POS, "description": "mm grid an auto skinThickness is chosen on (default the first "
                                             "pitch, at most 0.5)"},
        "lossFactor": {**_POS, "description": "material loss factor for isolate (default modal.LOSS_FACTOR)"},
        "verifyPitch": {**_POS, "description": "mm grid the delivered mesh is read back on (default half the "
                                               "design pitch: members one design cell thick)"},
        "keepThinMembers": {"type": "boolean", "description": "mesh without thinning one-cell members (default "
                                                             "true)"},
        "cutCells": {"type": "boolean", "description": "read the delivered mesh with cut cells, each cell the "
                                                      "surface cuts integrated over its solid part (default true; "
                                                      "false: as a density, cell fill at E x^3)"},
        "moveSharp": {"type": "number", "exclusiveMinimum": 0, "maximum": 0.5,
                      "description": "MMA move limit once the projection is fully sharp (default 0.03)"},
        "robust": {"type": "boolean", "description": "read each floor on the eroded design and each ceiling on "
                                                    "the dilated one, so the printed part meets both (default "
                                                    "true; false: the blueprint alone, faster, can meet soft "
                                                    "windows with grey cells that print as nothing)"},
        "pitches": {"type": "array", "items": _POS, "minItems": 1,
                    "description": "coarse to fine grids, mm (used when there is no switch block)"},
        "iterations": {"anyOf": [{"type": "integer", "minimum": 1},
                                 {"type": "array", "items": {"type": "integer", "minimum": 1}}]}},
    "required": ["targets"],
    "additionalProperties": False,
}

_SPLIT_PROPS = {
    "normal": {"anyOf": [{"type": "string", "pattern": "^[+-]?[xyz]$"}, {"const": "auto"}, _VEC3],
               "description": "the way the cover comes off: an axis like '+z', a vector (an angled cut), or "
                              "'auto' (default: tries x, y and z and makes the half holding the components the "
                              "base; with a seam, the seam's own plane)"},
    "at": {"type": "number", "description": "the cut's level along that normal (mm; n.x = at for a vector); "
                                            "omitted: searched"},
    "steps": {"type": "array", "items": {"type": "object", "properties": {
        "at": {"type": "number", "description": "the cut's level over this step (mm, as `at`)"},
        "where": {"$ref": "#/$defs/shape",
                  "description": "where the step is: the cut takes this level wherever a line along the "
                                 "normal passes through this shape"}},
        "required": ["at", "where"], "additionalProperties": False},
        "description": "a stepped parting line: the cut sits at `at`, and steps to each step's level over its "
                       "`where` (needs normal and at)"},
    "stepWidth": {**_POS, "description": "mm the cut takes to climb a step (default 0.5)"},
    "seam": {"type": "object", "properties": {
        "points": {"type": "array", "items": _VEC3, "minItems": 3,
                   "description": "the parting line on the exterior, in order round the part (closed)"}},
        "required": ["points"], "additionalProperties": False,
        "description": "a parting line that follows the designer's line on the exterior: the cut passes "
                       "through these points and is smooth inside (it is never seen). Any line the cover can "
                       "lift off from along the normal"},
    "avoid": {"type": "array", "items": {"$ref": "#/$defs/shape"},
              "description": "surfaces the seam must not cross (the designer's A surfaces)"},
    "hidden": {"type": "array", "items": {"$ref": "#/$defs/shape"},
               "description": "where a seam is not seen (an underside, inside an opening): seam there counts a "
                              "quarter of its length in the search, as seam in a crease does"},
    "search": {"enum": ["all", "flat"],
               "description": "with no `at`: 'all' (default) tries flat cuts across x, y and z, then tilted "
                              "planes, stepped lines (over or under each part a cut runs through, round each "
                              "`avoid`), the silhouette and creases in the exterior from the best of them; 'flat' "
                              "the axis planes only"},
    "componentsIn": {"type": "object", "additionalProperties": {"enum": ["base", "cover"]},
                     "description": "which half holds a component (default: the one most of it is in)"},
    "assembly": {"type": "array", "items": {"type": "string"},
                 "description": "component ids in the order they go in (default: as listed)"},
    "clearance": {"type": "number", "minimum": 0,
                  "description": "mm between mating faces of the lip and the snaps as built (default: 0.2, FDM 0.25, or "
                                 "wider where the process band needs it); drawn wider by the process's gap bias"},
    "margin": {"type": "number", "minimum": 0, "description": "mm from any component to a joint (default 0.5)"},
    "stackup": {"enum": ["rss", "worst"],
                "description": "how the joints are sized against the process's tolerance band (fits.HOLES, or a "
                               "calibrated card): 'rss' (default, root sum square) or 'worst' (every face at its "
                               "limit at once); the report gives both"},
    "pathTolerance": {"type": "number", "minimum": 0,
                      "description": "mm of interference allowed on a component's way in (default 0.25)"},
    "lip": {"anyOf": [{"const": False}, {"type": "object", "properties": {
        "height": _POS, "width": _POS, "root": _POS}, "additionalProperties": False}],
        "description": "the alignment lip: a tongue on the base standing up inside the cover's wall (height "
                       "default 2 mm, width default max(minMember, 1.2)); false: none"},
    "screws": {"anyOf": [{"const": False}, {"type": "object", "properties": {
        "from": {"type": "string", "pattern": "^[+-]?[xyz]$",
                 "description": "the base's far face the screws go in through: the designer allows the holes"},
        "size": {"enum": ["M2", "M2.5", "M3", "M4", "M5", "M6", "M8", "M10"]},
        "count": {"anyOf": [{"type": "integer", "minimum": 1}, {"const": "auto"}]},
        "insert": {**_POS, "description": "heat-set insert OD (mm); else a tapped hole"},
        "threadLength": _POS, "headHeight": _POS,
        "fit": {"enum": ["close", "normal", "loose"], "description": "the clearance hole (fits.py)"}},
        "required": ["from"], "additionalProperties": False}],
        "description": "screw posts: a bore up through the base, the head in a counterbore inside the exterior, "
                       "a thread in the cover; each hole is reported in exteriorChanges. Default: none"},
    "snaps": {"anyOf": [{"const": False}, {"type": "object", "properties": {
        "count": {"anyOf": [{"type": "integer", "minimum": 1}, {"const": "auto"}]},
        "on": {"enum": ["auto", "cover", "base"], "description": "the half the hooks hang from"},
        "width": _POS, "thickness": _POS, "length": _POS, "undercut": _POS,
        "leadAngle": {"type": "number", "exclusiveMinimum": 0, "exclusiveMaximum": 90},
        "returnAngle": {"type": "number", "exclusiveMinimum": 0, "maximum": 90,
                        "description": "90 (default): holds until it breaks; lower: comes apart at `pull` N"},
        "ledge": _POS, "root": _POS, "room": _POS, "maxShift": _POS,
        "repeated": {"type": "boolean", "description": "opened and closed often: 60% of the strain allowance"},
        "pull": _POS},
        "additionalProperties": False}],
        "description": "cantilever snaps, sized so the strain as the barb passes the ledge is within the "
                       "material's allowance. Default: on (auto count) when there are no screws"},
}

SPLIT_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "loadpath split block",
    "description": "`loadpath split` (shells.py): the study's exterior cut into a base and a cover, with an "
                   "alignment lip, screw posts and snap fits inside the skin. The cut is checked (or searched): "
                   "every component goes into its half with the other off, the cover comes down over them, and "
                   "the seam misses `avoid`. Out: a study per half and split.json. The exterior is never changed "
                   "except for screw holes through the face named in screws.from.",
    "type": "object",
    "properties": _SPLIT_PROPS,
    "additionalProperties": False,
    "$defs": {**_SHAPES, "shape": _SHAPE},
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
                                             "stand on a cover's inside)"},
                     "openings": {"type": "array", "items": {"type": "object", "properties": {
                         "shape": {"$ref": "#/$defs/shape"},
                         "inward": {"type": "string", "pattern": "^[+-]?[xyz]$",
                                    "description": "the way into the part (default +z)"},
                         "depth": {**_POS, "description": "mm the opening is swept inward (a skin and a bit)"}},
                         "required": ["shape"], "additionalProperties": False},
                         "description": "where the exterior opens into the interior (an underside a flap "
                                        "closes, a port): no skin inside them. The exterior's solid ends at an "
                                        "opening's inner end, and a skin lining that would be a membrane across "
                                        "the cavity behind it"},
                     "of": {"anyOf": [{"$ref": "#/$defs/shape"}, {"type": "array", "items": {"$ref": "#/$defs/shape"}}],
                            "description": "the exterior the skin lines when the design space is only part of it "
                                           "(a split half): the skin follows it and stops at the cut"},
                     "zones": {"type": "array", "items": {"type": "object", "properties": {
                         "shape": {"$ref": "#/$defs/shape"},
                         "thickness": {**_POS, "description": "mm of skin inside the zone"},
                         "blend": {"type": "number", "minimum": 0,
                                   "description": "mm over which the thickness ramps across the zone's edge "
                                                  "(default 2)"}},
                         "required": ["shape", "thickness"], "additionalProperties": False},
                         "description": "places the skin is another thickness, the exterior unchanged: only "
                                        "the inner surface moves (a thinner skin under a touch area so it "
                                        "feels pliable, a thicker one where a rim should be firm)"}},
                     "required": ["thickness"], "additionalProperties": False}],
                 "description": "Skin-and-rib: mm of the design space's outer surface kept as a solid wall; "
                                "the optimizer designs the ribs behind it."},
        "_shroud": {"type": "object", "description": "written by a shroud's expansion (loadpath shroud): "
                                                    "the seed, the envelope and its key; not set by hand"},
        "shroud": {"anyOf": [{"type": "boolean"}, _POS, {"type": "object", "properties": {
                       "thickness": {**_POS, "description": "mm of skin (default 2)"},
                       "reach": {**_POS, "description": "mm, closing radius: gaps and notches between members "
                                                        "narrower than about twice this are bridged by the skin "
                                                        "(default 12)"},
                       "smooth": {**_POS, "description": "mm the envelope's surface is smoothed over, for "
                                                         "sweeping rather than member-following surfaces (default 8)"},
                       "gap": {"type": "number", "minimum": 0,
                               "description": "mm the skin keeps off a concealed region (default 1)"},
                       "conceal": {"type": "array", "items": {"type": "string"},
                                   "description": "preserve or keepOut ids wrapped inside the skin: a cable or "
                                                  "tendon channel (a bore keep-out: it still opens where it "
                                                  "leaves the part), a board, a motor, an exchanger's passages. "
                                                  "Keep-outs with bore false that are not listed stay outside"},
                       "openings": {"type": "array", "items": {"type": "object"},
                                    "description": "as skin.openings: where the generated skin is left open"},
                       "seedFraction": {"type": "number", "exclusiveMinimum": 0, "exclusiveMaximum": 1,
                                        "description": "volume fraction of the coarse free design the envelope "
                                                       "is grown round (default 0.15)"},
                       "seedCells": {"type": "integer", "minimum": 1000,
                                     "description": "cells of the seed grid (default 60000)"},
                       "seedIterations": {"type": "integer", "minimum": 5},
                       "soft": {"type": "number", "minimum": 0,
                                "description": "mm the wrap is inflated and rounded by, for a soft, full form "
                                               "round the parts, like a volume drawn over them and smoothed "
                                               "hard (default 0: as tight as the loads allow; with soft the "
                                               "shape steps are off unless rounds is given). The skin and "
                                               "the interior are designed as usual"},
                       "rounds": {"type": "integer", "minimum": 0,
                                  "description": "shape steps from the skin's load share (default 10; 0: the "
                                                 "seed's smoothed hull)"},
                       "step": {**_POS, "description": "mm, the largest move of a shape step (default 1.5)"},
                       "pitch": {**_POS, "description": "mm, the envelope's grid (default half the seed's, "
                                                        "at most half the skin + 0.25)"}},
                       "additionalProperties": False}],
                   "description": "Generated exterior (shroud.py): no drawn exterior; a smooth skin is grown "
                                  "round the load paths and round what `conceal` lists, becomes the part's "
                                  "outside, and the interior is designed with it carrying load. Bores cut "
                                  "through it; drains open sealed interiors on powder and resin processes. "
                                  "Not with `skin`. `loadpath shroud` writes the expanded study (for "
                                  "`loadpath split`)."},
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
        "mesh": {"type": "object",
                 "description": "How the delivered surface is drawn from the optimized field. Larger radii give "
                                "joints that read as grown rather than assembled; both only ever add material.",
                 "properties": {
                     "close": {"type": "number", "minimum": 0,
                               "description": "Round every concave edge of the organic body by this radius (mm): "
                                              "a closing in distance space. Default half the minMember, at least 1."},
                     "fillet": {"type": "number", "minimum": 0,
                                "description": "Fillet radius (mm) where the organic body meets preserved "
                                               "geometry. Default 1.5 x minMember, at least 3."},
                     "smooth": {"type": "number", "minimum": 0,
                                "description": "Gaussian smoothing of the refined field, in fine cells (default 0.9)."},
                     "taubin": {"type": "integer", "minimum": 0,
                                "description": "Volume-keeping Taubin smoothing passes on the organic surface "
                                               "before the exact booleans (default 25)."}}},
        "resolution": {"type": "object",
                       "properties": {"maxElements": {"type": "integer", "minimum": 1, "maximum": 4000000},
                                      "pitch": _POS,
                                      "keepThinMembers": {
                                          "type": "boolean",
                                          "description": "Mesh members one or two cells thick without "
                                                         "thinning them (volume-keeping smoothing there, "
                                                         "and a debiased stem-smoothing average). "
                                                         "Off by default: the default rounding thins them, "
                                                         "which the searches are tuned to, but a design "
                                                         "made of one-cell walls can come out several "
                                                         "times softer than the optimizer's model (the "
                                                         "camera boom: 16x)."},
                                      "cutCells": {
                                          "type": "boolean",
                                          "description": "Verify on cut cells: each cell the part's surface "
                                                         "cuts gets the stiffness of its solid share, stress is "
                                                         "read on the true surface, and members under 3 cells "
                                                         "thick get a finer submodel. Member thickness then no "
                                                         "longer rounds to whole cells (a 2-cell member at 45 "
                                                         "degrees read bending 25% low), and a sub-cell strut "
                                                         "is seen at its true thickness (the robot arm's wrist link: "
                                                         "whole cells read SF 2.28 where 0.9 mm tets read "
                                                         "0.66, cut cells 0.67). On by default, except for "
                                                         "studies with pressure loads; false for whole cells."},
                                      "pitchStarts": {
                                          "oneOf": [{"type": "integer", "minimum": 1, "maximum": 5},
                                                    {"const": "auto"}],
                                          "description": "Searches run on this many grids 1% apart in "
                                                         "pitch, keeping the lightest passing part: one grid's "
                                                         "result can swing 25-140% with a 1% pitch change. "
                                                         "'auto' (default): 3 for safety-factor and deflection "
                                                         "searches up to 30k cells with no pitch given, else 1."}},
                       "additionalProperties": False},
        "maxIterations": {"type": "integer", "minimum": 1},
        "gravity": {**_VEC3, "description": "m/s², the part's own weight in every load case"},
        "vibration": _VIBRATION,
        "inertiaRelief": {"anyOf": [{"type": "boolean"}, {"type": "object", "properties": {
                              "masses": {"type": "array", "items": {
                                  "type": "object",
                                  "description": "What the free part carries (a motor, a battery, a camera) on "
                                                 + _TARGET + ": its inertia takes its share of every load",
                                  "properties": {"region": {"type": "string"}, "shape": {"$ref": "#/$defs/shape"},
                                                 "mass": {**_POS, "description": "kg"},
                                                 "at": {"type": "array", "items": {"type": "number"}, "minItems": 3,
                                                        "maxItems": 3,
                                                        "description": "mm: its centre of mass, when it sits off its "
                                                                       "mounts (a battery strapped above its posts): "
                                                                       "it is then a rigid body on the region and "
                                                                       "pulls on it with the moment of its inertia"},
                                                 "size": {"type": "array", "items": {"type": "number"},
                                                          "minItems": 3, "maxItems": 3,
                                                          "description": "with at: mm, a box's sides along x, y, "
                                                                         "z, for its moments of inertia"},
                                                 "inertia": {"type": "array",
                                                             "description": "with at: [Ixx, Iyy, Izz] or a 3x3 about "
                                                                            "its centre, kg mm^2 (default from size, "
                                                                            "else a point mass)"}},
                                  "x-requireOne": [["region"], ["shape"]], "required": ["mass"],
                                  "additionalProperties": False}}},
                              "additionalProperties": False}],
                          "description": "A part nothing holds (a drone frame, a dropped or thrown part): no "
                                         "supports; inertia balances each case's loads (inertia relief) and a 3-2-1 "
                                         "hold that takes nothing keeps it still. {masses: [...]} adds what it "
                                         "carries (a drone's motors and battery outweigh its frame), and a "
                                         "vibration block then reads its free-free modes. Not with gravity, "
                                         "thermal or loads of unknown direction."},
        "massProperties": _MASS_PROPERTIES,
        "aero": _AERO,
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
                        "stepDown": {**_POS, "description": "mm between 2.5-axis pocket floors (default 2)"},
                        "rates": {"type": "object", "properties": {
                                      k: {"type": "number", "minimum": 0, "description": d} for k, d in (
                                          ("roughingRate", "cm^3/min removed with a 10 mm end mill"),
                                          ("finishFeed", "mm/min finishing feed"),
                                          ("stepover", "mm between finishing passes (default cutter / 10)"),
                                          ("stockPerKg", "$ per kg of stock"),
                                          ("machinePerHour", "$ per machine hour"),
                                          ("perSetup", "$ per setup (fixture, probing)"),
                                          ("stockMargin", "mm of stock past the part on every side"))},
                                  "additionalProperties": False,
                                  "description": "Overrides for the machining time and cost estimate "
                                                 "(defaults per material family)."}},
                    "additionalProperties": False,
                    "description": "The part is machined: every void reachable by a cutter coming straight in."},
        "casting": {"anyOf": [{"type": "string", "pattern": "^[+-]?[xyz]$"},
                              {"type": "object", "properties": {
                                  "draw": {"type": "string", "pattern": "^[+-]?[xyz]$",
                                           "description": "the direction the mold opens"},
                                  "sides": {"enum": [1, 2], "description": "2 (default): halves pull both ways "
                                                                           "along the draw; 1: one half pulls off "
                                                                           "a flat parting plane"},
                                  "draft": {"type": "number", "minimum": 0, "maximum": 15,
                                            "description": "degrees of draft on every wall (1-3 die casting and "
                                                           "molding, 3-5 sand): applied to the delivered part "
                                                           "and checked facet by facet"},
                                  "parting": {"type": "number",
                                              "description": "two-sided: the parting plane's coordinate along "
                                                             "the draw axis, mm (default: the widest section)"}},
                               "required": ["draw"], "additionalProperties": False}],
                    "description": "Cast or molded: no undercuts along the draw direction."},
        "explore": {"type": "object", "properties": {
                        "methods": {"type": "array", "minItems": 1, "items": {"enum": _EXPLORE_METHODS},
                                    "description": "manufacturing methods to solve for (default fdm, mjf, sla, "
                                                   "lpbf, cast, milled-3axis)"},
                        "materials": {"type": "array", "minItems": 1, "items": {"$ref": "#/$defs/material"},
                                      "description": "library materials (or custom ones); each runs on every "
                                                     "method its card says makes it (default: one per method)"},
                        "routes": {"type": "array", "minItems": 1, "items": {
                            "type": "object", "properties": {"method": {"enum": _EXPLORE_METHODS},
                                                             "material": {"$ref": "#/$defs/material"}},
                            "required": ["method", "material"], "additionalProperties": False},
                                   "description": "method/material pairs outright (instead of methods x materials)"},
                        "quantity": {"type": "integer", "minimum": 1,
                                     "description": "parts ordered: tooling and set-ups are spread over them"},
                        "rank": {"enum": ["mass", "cost", "leadTime"], "description": "ranking order (default mass)"},
                        "draw": {"type": "string", "pattern": "^[+-]?[xyz]$",
                                 "description": "cast: the mold's draw (default the design space's thinnest axis)"},
                        "castSides": {"enum": [1, 2]},
                        "millingDirections": {"type": "array", "minItems": 1,
                                              "items": {"type": "string", "pattern": "^[+-]?[xyz]$"},
                                              "description": "milled: sides the cutter comes from (default both "
                                                             "ends of the thinnest axis; 5-axis all six)"},
                        "toolDiameter": {**_POS, "description": "milled: smallest cutter, mm (default 6)"},
                        "screen": {"type": "boolean", "description": "every route on the coarse grid first, "
                                                                     "full runs only for the Pareto fronts and "
                                                                     "top of the ranking"},
                        "refineTop": {"type": "integer", "minimum": 0,
                                      "description": "screen: also refine this many of the top-ranked (default 2)"},
                        "shareDesigns": {"type": "boolean", "description": "routes of one method reuse each "
                                                                           "other's designs (default true)"},
                        "costs": {"type": "object", "description": "override costing.RATES: {method family: "
                                                                   "{rate: value}}, materialPrice: {material: $/kg}"}},
                    "additionalProperties": False,
                    "description": "`loadpath explore`: solve the study once per manufacturing method and "
                                   "material, with cost and lead time, ranked (explore.py). `run` ignores it."},
        "switch": _SWITCH,
        "feel": _FEEL,
        "split": {"type": "object", "properties": _SPLIT_PROPS, "additionalProperties": False,
                  "description": "`loadpath split`: cut the exterior into a base and a cover with a lip, screw "
                                 "posts and snaps inside the skin (shells.py; `loadpath schema split`). `run` "
                                 "refuses a study that still has one: split it first and run each half"},
        "cooling": _COOLING,
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
        "assets": {"type": "object", "properties": {
            "env": {"type": "string"}, "files": {"type": "array", "items": {"type": "string"}},
            "make": {"type": "string"}}, "required": ["env"], "additionalProperties": False,
            "description": "Eval-only: geometry kept out of this repository (a product's own CAD), read from the "
                           "directory the environment variable `env` names; the case is skipped without it."},
    },
    "required": ["designSpace"],
    "x-requireOne": [["loadCases"], ["loads"], ["feel"], ["cooling"]],
    "additionalProperties": False,
    "$defs": {**_SHAPES, "shape": _SHAPE, "region": _REGION, "keepOut": _KEEP_OUT, "support": _SUPPORT,
              "switchClearance": _SWITCH_CLEARANCE,
              "load": _LOAD, "loadCase": _LOAD_CASE, "material": _ONE_MATERIAL, "fastener": _FASTENER, "goal": _GOAL},
}

_FLUID_NAMES = ["water", "water60", "air", "glycol50", "oil"]
_DUTY_SIDE = {
    "type": "object",
    "properties": {"fluid": {"anyOf": [
                       {"enum": _FLUID_NAMES},
                       {"type": "object", "description": "the fluid's own properties at its working state, SI "
                                                         "(boost air at 2 bar, a coolant mix not listed)",
                        "properties": {"name": {"type": "string"}, "rho": _POS, "cp": _POS, "k": _POS, "mu": _POS},
                        "required": ["rho", "cp", "k", "mu"], "additionalProperties": False}]},
                   "lPerMin": _POS, "kgPerS": _POS,
                   "tIn": {"type": "number", "description": "°C. A named fluid (air, water, glycol50) is taken "
                                                            "at its own state: this temperature, then the mean of "
                                                            "inlet and outlet"},
                   "pressure": {**_POS, "description": "bar absolute, for a gas (default 1.013): boost air at "
                                                       "2.5 bar is 2.5x as dense"}},
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
        "headers": {"enum": ["flat", "loft"],
                    "description": "flat (default): each plenum an even header depth over its zone; loft: a "
                                   "smooth funnel from the zone to its port (the convex hull of the two), so the "
                                   "outside sweeps from core to port with no flat lid or side walls"},
        "headerRibs": {"type": "integer", "minimum": 0,
                       "description": "retired and ignored (with a warning): it stood hand-placed fins on each lofted "
                                      "header that no load sized; the plenum walls are the shell everywhere"},
        "exteriorBlend": {"type": "number", "minimum": 0,
                          "description": "mm, fillet radius where the outside of the core, headers and ports meet "
                                         "(default 0: sharp); only adds material outside, the fluid spaces stay "
                                         "as designed"},
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
        "maxPoints": {"type": "integer", "minimum": 100000,
                      "description": "cap on the design grid's points (default 16M, about 270 bytes each at the "
                                     "peak; a design over the free memory is refused): a car-sized core at a "
                                     "1 mm wall needs about 40M"},
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
                                                 "(default 3 pitches); make it cover shell thicknesses"},
                 "smooth": {"anyOf": [{"type": "boolean"}, _POS],
                            "description": "take a faceted organic mesh's facets off first (true: ripples "
                                           "under 4 mean edges; a number: under that many mm), so an "
                                           "offset or shell of it meshes smooth instead of tracing every "
                                           "facet; edges sharper than smoothCrease stay put"},
                 "smoothCrease": {**_POS, "description": "degrees; sharper edges are kept (default 60)"}},
                require_one=[["path"], ["source"]]),
    "result": _op("result", "A simulation result as a scalar field: von Mises stress (MPa) or displacement "
                            "(mm) of `part` under `study` (analyzed on demand), a saved .npz (points + values, "
                            "or verify fields), or an in-memory source.",
                  {"path": {"type": "string", "description": ".npz with points and values (save_result_field), "
                                                            "a grid (origin, pitch, values), or a point map: a "
                                                            ".csv/.txt of x, y, z and values (another tool's FE "
                                                            "or CFD export)"},
                   "interpolate": {"enum": ["idw", "rbf", "nearest"],
                                   "description": "between scattered samples: inverse distance over the 8 "
                                                  "nearest (default), a thin-plate spline over the 32 nearest "
                                                  "(smooth, held to the samples' range), or the nearest"},
                   "source": {"type": "string"},
                   "study": {"type": "string", "description": "study JSON, relative to the graph's folder, or "
                                                              "'example:<name>' for a packaged example"},
                   "part": {"type": "string", "description": "the mesh the study analyzes, or the name of a "
                                                             "node of this graph (meshed for the analysis)"},
                   "pitch": {**_POS, "description": "mm, meshing a node part for the analysis (default 1)"},
                   "quantity": {"type": "string", "description": "vonMises (default) or displacement for a "
                                                                 "study; the array name in an .npz; the column "
                                                                 "header in a point map (default its 4th column)"},
                   "case": {"type": "string", "description": "one load case (default: the worst at each point)"},
                   "scale": {"anyOf": [{"const": "peak"}, _POS],
                             "description": "divide by this, or 'peak' (the 99.5th percentile): remap in "
                                            "fractions of the part's own worst stress"},
                   "cells": {"type": "integer", "minimum": 1000, "description": "analysis size (default 120000)"}},
                  require_one=[["path"], ["source"], ["study", "part"]]),
    "grid": _op("grid", "An imported scalar field on a regular grid (.npz with origin, pitch, values[nx,ny,nz], "
                        "or an OpenVDB .vdb float grid): a temperature map, a CFD pressure, a "
                        "hand-painted density, a Houdini or Blender volume.",
                {"path": {"type": "string"}, "source": {"type": "string"},
                 "quantity": {"type": "string", "description": "array name (default values); a .vdb grid's name (default the first)"}},
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
                              "kelvin, diamond-strut, or a custom cell from cellGraph), or a stochastic voronoi "
                              "foam or delaunay (tet) lattice; a delaunay lattice with a conformal frame puts its "
                              "nodes on that solid's surface and keeps its struts inside it (a tet lattice from a "
                              "body). Cell size, wall or strut thickness, relative density and node blend can each "
                              "be a field; a conformal frame lays the cells along a surface.",
                   {"type": {"enum": ["gyroid", "diamond", "schwarz-p", "lidinoid", "split-p", "neovius", "cubic", "bcc", "bccz", "fcc", "fccz",
                                      "octet", "kelvin", "diamond-strut", "voronoi", "delaunay", "custom"],
                             "description": "TPMS sheets are stiffer and stronger per gram (lattice.SHEETS vs "
                                            "STRUTS); struts are open, easy to clear of powder, and print on FDM "
                                            "when the cell is oriented (lattice report's overhangs)"},
                    "cell": {**_FIELD, "description": "mm per period (default 8); a field grades it (voronoi, "
                                                      "delaunay: mean seed spacing)"},
                    "cellGraph": {"type": "object", "description": "type custom: the cell's struts, nTop's custom "
                                                                   "unit cell",
                                  "properties": {
                                      "nodes": {"type": "array", "items": _VEC3, "minItems": 2,
                                                "description": "[u, v, w] in cell units (0 to 1 across the cell; "
                                                               "it repeats with period 1)"},
                                      "beams": {"type": "array", "items": {
                                          "type": "array", "items": {"type": "integer", "minimum": 0},
                                          "minItems": 2, "maxItems": 2}, "minItems": 1,
                                          "description": "pairs of node indices"}},
                                  "required": ["nodes", "beams"], "additionalProperties": False},
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
                    "orient": {"description": "turn the cells (not the fields grading them) about the "
                                              "origin: a rotation vector in degrees [rx, ry, rz], or "
                                              "{axis, degrees}, or {zones: [{at, rotate}], blend} to turn "
                                              "them part by part; `loadpath lattice-orient` picks them so "
                                              "the stiff cell directions follow the load",
                               "anyOf": [{"type": "array", "items": {"type": "number"}, "minItems": 3, "maxItems": 3},
                                         {"type": "object", "additionalProperties": False,
                                          "description": "{axis, degrees}; or zones: cells turned zone by zone, "
                                                         "cross-faded over `blend` mm (default one cell) where "
                                                         "the zones meet",
                                          "properties": {
                                              "axis": _AXIS, "degrees": {"type": "number"},
                                              "zones": {"type": "array", "minItems": 1, "items": {
                                                  "type": "object", "additionalProperties": False,
                                                  "required": ["at"],
                                                  "properties": {"at": _VEC3, "rotate": _VEC3}}},
                                              "blend": _POS,
                                              "seam": {"type": "number", "minimum": 0,
                                                       "description": "zones: mm thick solid plate on each face "
                                                                      "between zones, the cells of both ending on "
                                                                      "it as on a skin (a hard split, no blend); "
                                                                      "without one, thin cells of two directions "
                                                                      "meet at small joints that read too stiff"}}}]},
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
                                                                                   "cell)"},
                                             "frameField": {"enum": [True, False, "auto"],
                                                            "description": "conformal: true, cells follow a 3D "
                                                                           "frame field, turning a box's edges "
                                                                           "and corners; false, two stripe "
                                                                           "patterns along the surface and the "
                                                                           "surface's distance as the layer "
                                                                           "coordinate (exact layers); auto "
                                                                           "(default), whichever shears fewer "
                                                                           "cells in the first cell under the "
                                                                           "skin"},
                                             "layerSmoothing": {"type": "number", "minimum": 0, "maximum": 1,
                                                                "description": "conformal: round the layers where "
                                                                               "the surface's distance folds "
                                                                               "(its medial axis): the distance "
                                                                               "is averaged over a sphere this "
                                                                               "times the depth (capped at a "
                                                                               "cell); exact at the skin; 0 "
                                                                               "keeps the sharp fold (default "
                                                                               "0)"}},
                              "required": ["type"], "additionalProperties": False}},
                   require_one=None),
    "transform": _op("transform", "Move, rotate or scale a field's input (translate, then rotate, then scale).",
                     {"input": _FIELD, "translate": _VEC3,
                      "rotate": {"type": "object", "properties": {"axis": _AXIS, "degrees": {"type": "number"},
                                                                  "center": _VEC3},
                                 "required": ["degrees"], "additionalProperties": False},
                      "scale": _POS}, ["input"]),
    "pattern": _op("pattern", "Copies of a body as one solid: in a line or grid (linear), round an axis "
                              "(circular), mirrored across a plane, or along a curve. Each point reads the "
                              "nearest few copies (`neighbors`), so a pattern of hundreds costs a few.",
                   {"input": {**_FIELD, "description": "the body copied"},
                    "type": {"enum": ["linear", "circular", "mirror", "curve"], "description": "default linear"},
                    "step": {**_VEC3, "description": "linear: mm from one copy to the next"},
                    "count": {"type": "integer", "minimum": 1, "description": "copies (linear, circular, curve)"},
                    "directions": {"type": "array", "minItems": 1, "maxItems": 3,
                                   "items": {"type": "object", "properties": {"step": _VEC3,
                                                                              "count": {"type": "integer",
                                                                                        "minimum": 1}},
                                             "required": ["step"], "additionalProperties": False},
                                   "description": "linear: a grid, one {step, count} per direction"},
                    "axis": {**_AXIS, "description": "circular: the axis (default z)"},
                    "center": {**_VEC3, "description": "circular: a point on the axis"},
                    "angle": {"type": "number", "description": "circular: degrees swept (default 360, evenly "
                                                               "round; less spreads the copies end to end)"},
                    "rotate": {"type": "boolean", "description": "circular: turn each copy with the pattern "
                                                                 "(default true)"},
                    "pivot": _VEC3,
                    "plane": {"type": "object", "properties": {"point": _VEC3, "normal": _AXIS},
                              "required": ["normal"], "additionalProperties": False,
                              "description": "mirror: the plane"},
                    "keep": {"type": "boolean", "description": "mirror: keep the original too (default true)"},
                    "path": {"type": "array", "items": _VEC3, "minItems": 2,
                             "description": "curve: the polyline the copies go along"},
                    "closed": {"type": "boolean", "description": "curve: a closed loop (no copy doubled at "
                                                                 "the ends)"},
                    "orient": {"type": "boolean", "description": "curve: turn each copy's x along the curve, "
                                                                 "z toward `up` (default true)"},
                    "up": _AXIS, "anchor": {**_VEC3, "description": "curve: the input's point that goes on "
                                                                     "the curve (default the path's start)"},
                    "blend": _FIELD,
                    "neighbors": {"type": "integer", "minimum": 1, "description": "copies read per point "
                                                                                  "(default 4)"},
                    "exact": {"type": "boolean", "description": "read every copy at every point"}},
                   ["input"]),
    "ribs": _op("ribs", "Ribs on a solid's face: walls of `thickness` standing `height` off its surface "
                        "(inside it, under the face, or outside), laid along curves projected onto it "
                        "(`paths`), a `pattern` (square grid, isogrid, radial spokes and rings), or the "
                        "principal stress lines a study puts in the face (`stressLines`). With `direction` "
                        "the walls are parallel to it (a straight pull); without, square to the surface.",
                {"surface": {**_FIELD, "description": "the solid whose face the ribs stand on"},
                 "thickness": {**_FIELD, "description": "mm wall thickness, or a field (graded ribs)"},
                 "height": {**_FIELD, "description": "mm off the surface: depth under it (side inside) or "
                                                     "height over it (outside); default 5"},
                 "side": {"enum": ["inside", "outside"], "description": "default inside"},
                 "paths": {"type": "array", "items": {"type": "array", "items": _VEC3, "minItems": 2},
                           "description": "polylines [[x, y, z], ...] dropped onto the surface (along "
                                          "`direction`, else to the nearest point)"},
                 "direction": {**_AXIS, "description": "the walls' direction (paths dropped along it); "
                                                       "omit for walls square to the surface"},
                 "pattern": {"type": "object", "description": "a rib pattern, walls along its direction",
                             "properties": {"type": {"enum": ["square", "triangle", "isogrid", "radial"]},
                                            "spacing": {**_POS, "description": "mm between ribs (rings, for "
                                                                               "radial)"},
                                            "angle": {"type": "number", "description": "degrees about the "
                                                                                       "direction"},
                                            "direction": _AXIS, "origin": _VEC3, "center": _VEC3,
                                            "spokes": {"type": "integer", "minimum": 0},
                                            "rings": {"type": "boolean"},
                                            "innerRadius": {"type": "number", "minimum": 0}},
                             "additionalProperties": False},
                 "stressLines": {"type": "object", "description": "ribs along the face's principal stress "
                                                                  "lines under a study's loads",
                                 "properties": {"study": {"type": "string", "description": "study JSON path or "
                                                                                           "example:<name>"},
                                                "part": {"type": "string", "description": "the node or mesh "
                                                                                          "analyzed (default "
                                                                                          "the surface)"},
                                                "spacing": {**_POS, "description": "mm between lines "
                                                                                   "(default 10)"},
                                                "family": {"enum": ["max", "min", "both"],
                                                           "description": "the larger principal stress, the "
                                                                          "other, or both (default)"},
                                                "cutoff": {"type": "number", "minimum": 0,
                                                           "description": "seed only where the stress is over "
                                                                          "this times the face's 95th "
                                                                          "percentile (default 0.15)"},
                                                "merge": {"type": "number", "minimum": 0,
                                                          "description": "mm at which a line joins another "
                                                                         "(default 0.15 spacing)"},
                                                "bridge": {"type": "number", "minimum": 0,
                                                           "description": "mm a line carries straight over low "
                                                                          "stress (default 1.5 spacing)"},
                                                "case": {"type": "string"},
                                                "within": {"type": "object", "properties": {"min": _VEC3,
                                                                                            "max": _VEC3},
                                                           "required": ["min", "max"],
                                                           "additionalProperties": False,
                                                           "description": "only the face inside this box"},
                                                "direction": _AXIS, "pitch": _POS,
                                                "cells": {"type": "integer", "minimum": 1000}},
                                 "required": ["study"], "additionalProperties": False},
                 "pathStep": _POS, "gradientStep": _POS},
                ["surface", "thickness"], require_one=[["paths"], ["pattern"], ["stressLines"]]),
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

_ND_WHERE = {"region": {"type": "string", "description": "solid node: the mesh nodes inside it"},
             "at": {**_VEC3, "description": "mm: the mesh nodes within radius of this point"},
             "radius": {**_POS, "description": "mm round `at` (default 1.5 cells)"}}
_OBSTACLE = {
    "type": "object",
    "description": "A rigid object the part may not enter (frictionless unless friction is set): plane (point, normal: the side the "
                   "part is on), sphere (center, radius), cylinder (center, axis, radius; infinitely long) or "
                   "box (min, max). inside: the part stays inside it instead (a socket, a tube). move: mm the "
                   "object travels at full load (an indenter, a jaw, a snap fit's catch).",
    "properties": {"type": {"enum": ["plane", "sphere", "cylinder", "box"]},
                   "name": {"type": "string"},
                   "point": _VEC3, "normal": _VEC3, "center": _VEC3, "radius": _POS, "axis": _AXIS,
                   "min": _VEC3, "max": _VEC3, "inside": {"type": "boolean"},
                   "move": {**_VEC3, "description": "mm at full load"},
                   "stiffness": {**_POS, "description": "penalty, x E x pitch N/mm per node (default 20)"},
                   "friction": {"type": "number", "minimum": 0,
                                "description": "Coulomb coefficient mu: sticks until the tangential force "
                                               "reaches mu x the normal force, then slides (default: the "
                                               "contact block's, else 0)"}},
    "required": ["type"],
    "additionalProperties": False,
}
_LARGE_DEFLECTION = {
    "type": "object",
    "description": "Large-deflection check (nonlinear.py: total Lagrangian hex8, neo-Hookean, Newton with "
                   "load steps and a buckling watch) of a part meant to bend: a TPU finger, a flexure, a snap "
                   "fit, a compliant gripper. Drive it by dead loads, prescribed displacements and tendons "
                   "(cables through guides on the part); contact with rigid obstacles and the part's own "
                   "surface. Needs material.",
    "properties": {
        "part": {"type": "string", "description": "solid node analysed (default the output)"},
        "fixed": {"type": "string", "description": "solid node: nodes inside it are clamped"},
        "pitch": {**_POS, "description": "mm analysis grid (default 1; coarsened to fit maxCells)"},
        "maxCells": {"type": "integer", "minimum": 100, "description": "most cells (default 20000)"},
        "nu": {"type": "number", "exclusiveMinimum": 0, "maximum": 0.499,
               "description": "Poisson's ratio (default the material's)"},
        "model": {"enum": ["neo-hookean", "svk"], "description": "hyperelastic law (default neo-hookean)"},
        "steps": {"type": "integer", "minimum": 1, "description": "first guess at load steps (default 10)"},
        "loads": {"type": "array", "items": {"type": "object", "properties": {
            **_ND_WHERE, "force": {**_VEC3, "description": "N, dead (keeps its direction), spread evenly"}},
            "required": ["force"], "x-requireOne": [["region"], ["at"]], "additionalProperties": False}},
        "displacements": {"type": "array", "items": {"type": "object", "properties": {
            **_ND_WHERE, "move": {**_VEC3, "description": "mm at full load: an actuator's stroke"},
            "free": {"type": "boolean", "description": "leave the directions with zero move free (default "
                                                       "false: held)"}},
            "required": ["move"], "x-requireOne": [["region"], ["at"]], "additionalProperties": False},
            "description": "Prescribed motion; the report gives the force it takes (actuatorForce, N)."},
        "tendons": {"type": "array", "items": {"type": "object", "properties": {
            "path": {"type": "array", "minItems": 2, "items": {"anyOf": [
                {**_VEC3, "description": "a fixed point in space (the spool, a palm pulley)"},
                {"type": "object", "properties": _ND_WHERE, "x-requireOne": [["region"], ["at"]],
                 "additionalProperties": False, "description": "a guide on the part; the cable slides "
                                                               "through it"}]},
                     "description": "Points the cable runs straight between; the last is the knot."},
            "tension": {**_POS, "description": "N, constant (scaled with the load steps)"},
            "stroke": {**_POS, "description": "mm of cable a servo winds in (displacement driven)"},
            "stiffness": {**_POS, "description": "N/mm of the cable and servo, with stroke"}},
            "required": ["path"], "x-requireOne": [["tension"], ["stroke", "stiffness"]],
            "additionalProperties": False},
            "description": "Cables through guides on the part: the load follows the part as it curls."},
        "contact": {"type": "object", "properties": {
            "obstacles": {"type": "array", "items": _OBSTACLE},
            "self": {"type": "boolean", "description": "the part's surface on itself (fingers meeting, a "
                                                       "finger folding, a gap closing)"},
            "stiffness": {**_POS, "description": "penalty for all, x E x pitch N/mm per node (default 20)"},
            "friction": {"type": "number", "minimum": 0,
                         "description": "Coulomb coefficient for all contacts, the part on itself included "
                                        "(default 0: frictionless). PLA on PLA ~0.3-0.5, TPU ~0.6-1, PTFE "
                                        "~0.05-0.1."}},
                    "additionalProperties": False},
        "probes": {"type": "array", "items": {"type": "object", "properties": {
            **_ND_WHERE, "name": {"type": "string"},
            "direction": {**_VEC3, "description": "report the mean travel along this (mm)"}},
            "required": ["direction"], "x-requireOne": [["region"], ["at"]], "additionalProperties": False}},
        "maxStrain": {**_POS, "description": "largest allowed strain: sets ok"}},
    "required": ["fixed"],
    "additionalProperties": False,
}


FIELD_SCHEMA = {
    "$schema": _DIALECT,
    "$id": "https://github.com/connorwhite-online/loadpath/schema/field.schema.json",
    "title": "loadpath field graph",
    "description": "An implicit (field) part: named nodes, each a solid (signed distance, negative "
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
                                                          "curve (default true; not with stream)"},
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
                                                                      "(default 20000, ~14 MB)"},
            "surface": {"enum": ["facets", "smooth"],
                        "description": "organic regions as flat facets (default) or smooth B-spline patches "
                                       "(experimental: CAD-friendlier, not smaller; falls back to facets when "
                                       "OpenCascade's booleans fail)"},
            "patches": {"type": "integer", "minimum": 50, "description": "surface \"smooth\": organic triangles of the patch layout, two to a patch, refined where it misses (default 2000)"},
            "tolerance": {"type": "number", "exclusiveMinimum": 0,
                          "description": "mm the facets may move where an organic region keeps a TPMS core's "
                                         "tunnels over the face budget rather than close them (default 0.05)"},
            "lattice": {"enum": ["solid", "exact", "cavities", "mesh"],
                        "description": "lattice parts: the lattice solid in the STEP (default; the lattice is in "
                                       "the STL/3MF) or \"exact\": every lattice as smooth patches on its true "
                                       "surface, node blends and fillets included: strut cells (octet, bcc, "
                                       "cubic, ...), Voronoi foams, TPMS sheets or networks; graded, conformal; "
                                       "under a skin or open at the outside (latticestep.export_void_step: the "
                                       "envelope less the void, sealed voids as inner shells, open pores cut in "
                                       "one boolean; ~1 KB a patch). \"cavities\": the older route for one TPMS "
                                       "sheet lattice under a closed skin (its two sides' cavities). \"mesh\": "
                                       "the printed part's own mesh as facets, lattice and all, simplified "
                                       "within `tolerance` with every tunnel kept (fast: a minute or two for a "
                                       "skinned, drained gyroid link)"},
            "latticeGrid": {"type": "number", "exclusiveMinimum": 0,
                            "description": "lattice \"exact\": patch size, mm (default the smaller of cell / 6 "
                                           "and half the thinnest strut or wall); coarser is smaller and misses "
                                           "tight fillets; much over half the thinnest strut can lose struts"}},
            "additionalProperties": False}],
                 "description": "also write <name>.step (needs loadpath[step]): exact planes, cylinders and spheres "
                                "where the graph's hard booleans use plain shapes, the rest faceted or smooth; "
                                "with lattice nodes, the part with its lattices solid (drain bores included) or, "
                                "lattice \"exact\", the lattice itself as smooth patches"},
        "lattice3mf": {"anyOf": [{"type": "boolean"}, {"type": "object", "properties": {
            "decimate": {"type": "number", "exclusiveMinimum": 0,
                         "description": "body and region meshes within this many mm of the surface (default 0.02)"},
            "grid": {"type": "number", "exclusiveMinimum": 0,
                     "description": "voxel size, mm, of the 3D images a graded TPMS reads (wall, cell, lattice "
                                    "coordinates, blend radius; default min(cell / 6, 1))"},
            "meshReaders": {"type": "boolean",
                            "description": "mark the lattice region meshes type \"other\" (not the Beam Lattice "
                                           "spec's \"model\") so PrusaSlicer 2.x opens a beam lattice file, "
                                           "printing the struts' prism fallback; level sets still need an "
                                           "implicit-capable reader"}},
            "additionalProperties": False}],
            "description": "also write <name>-lattice.3mf (needs loadpath[3mf]): every lattice node as a lattice, "
                           "TPMS as an implicit level set (3MF Volumetric & Implicit; graded cell, wall and "
                           "conformal frames through 3D images; a blended union with the body kept as a smooth "
                           "union in the level set), strut cells as a beam lattice with prism fallback triangles, "
                           "each clipped by its own region, beside the rest of the part as a mesh"},
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
                                                 "else meshed and verified as a solid; homogenized on a part with no "
                                                 "lattice reads it on the same fill-weighted grid (thin skins "
                                                 "without a fine mesh)"},
                     "pitch": {**_POS, "description": "mm mesh for each candidate (default 1)"},
                     "cells": {"type": "integer", "minimum": 1000, "description": "analysis cells (default 30000)"},
                     "checkCells": {"type": "integer", "minimum": 1000,
                                    "description": "most cells for the confirming check, which runs at two cells "
                                                   "through the thinnest shell wall (default 2000000)"}}},
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
            "striker": {"type": "object", "properties": {
                            "side": {"enum": ["+", "-"], "description": "the end of the axis the hit comes from "
                                                                        "(default +)"},
                            "shape": {"enum": ["flat", "sphere", "cylinder"],
                                      "description": "a wall or floor (default), a ball, or a pole or rounded edge"},
                            "radius": {**_POS, "description": "mm, sphere or cylinder"},
                            "along": {"enum": ["x", "y", "z"], "description": "the cylinder's axis (across the hit)"},
                            "at": {"type": "array", "items": {"type": "number"}, "minItems": 2, "maxItems": 2,
                                   "description": "where a ball or pole is centred, in the two axes across the "
                                                  "hit (default: where the core is touched first)"}},
                        "additionalProperties": False,
                        "description": "What the core meets, and from which side: the contact model. A curved "
                                       "core against a wall (a round tyre, a domed pad) or any core against a "
                                       "ball or pole is touched at a point or a line first, and the force grows "
                                       "with the contact patch. Each column of the core along the hit is "
                                       "squeezed only by what the striker has travelled past its own face "
                                       "(a crushing foam does not spread load sideways); its cells crush "
                                       "weakest first. Without a striker the core is a stack of layers that "
                                       "each carry the whole load (right for a flat-ended block between "
                                       "plates)."},
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
        "largeDeflection": _LARGE_DEFLECTION,
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


_EXTERIOR_SOURCE = {
    "type": "object",
    "description": "any exterior through the intake (loadpath.exterior.load_exterior; `loadpath exterior`): a "
                   "generated or scanned mesh is wrapped watertight, scaled and turned +z up",
    "properties": {
        "path": {"type": "string", "description": "STEP, STL/OBJ/PLY/3MF/GLB/glTF, an .npz or .vdb SDF grid, or "
                                                  "file.py[:name] (an SDF function or build123d/CadQuery object)"},
        "size": {**_POS, "description": "the real longest extent, mm (a generated mesh comes at unit scale)"},
        "scale": {**_POS, "description": "a factor instead of size (metres to mm: 1000)"},
        "up": {"type": "string", "pattern": "^[+-]?[xyz]$",
               "description": "the source's up axis (default +y for GLB/glTF, else +z)"},
        "place": {"enum": ["keep", "center", "origin"],
                  "description": "keep the source frame (default unless scaled or turned) or centre it on z = 0"},
        "pitch": {**_POS, "description": "mm, the wrap's grid (default longest extent / 320)"},
        "wrap": {"enum": ["auto", "always", "never"]},
        "keep": {"enum": ["largest", "all"], "description": "bodies kept (default the largest: debris dropped)"},
        "seal": {**_POS, "description": "mm, holes sealed by the flood wrap (no libigl)"}},
    "required": ["path"], "additionalProperties": False}
_HANDOFF_SHAPE = {"anyOf": [{"type": "string", "description": "a mesh path (STL/3MF/OBJ/GLB; open or broken "
                                                              "meshes are wrapped), a STEP file or an .npz/.vdb SDF"},
                            {"$ref": "#/$defs/shape"}, _EXTERIOR_SOURCE]}
_AXIS = {"type": "string", "pattern": "^[+-]?[xyz]$"}
HANDOFF_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "loadpath handoff brief",
    "description": "`loadpath handoff`: the designer's exterior and component CAD in, a push-pad study out "
                   "(handoff.py). Keep-outs per component with a clearance per component, insertFrom found from "
                   "the assembly order, supports from what the part stands on, the plunger and its seat from the "
                   "switch's geometry. The exterior is never changed.",
    "type": "object",
    "properties": {
        "name": {"type": "string"}, "description": {"type": "string"},
        "exterior": {**_HANDOFF_SHAPE, "description": "the exterior: the designer's surfacing, or any shape "
                                                      "(generated, scanned, an SDF) through the exterior intake; "
                                                      "never changed"},
        "skin": {"anyOf": [_POS, {"type": "object"}], "description": "the skin kept of the exterior (as a study's "
                                                                     "skin; default 1.0 mm)"},
        "openings": {"type": "array", "items": {"anyOf": [_HANDOFF_SHAPE, {"type": "object", "properties": {
            "shape": _HANDOFF_SHAPE, "inward": _AXIS, "depth": _POS}, "required": ["shape"],
            "additionalProperties": False}]},
            "description": "where the exterior opens into the interior: no skin across their inner ends "
                           "(skin.openings, swept a skin and 0.2 mm inward by default)"},
        "expect": {"type": "object", "description": "eval and tutorial criteria, passed to the study (expect.coarse."
                                                 "maxElements: the grid quick mode may not go coarser than)"},
        "fixed": {"type": "array", "items": {"type": "object", "properties": {
            "id": {"type": "string"}, "shape": _HANDOFF_SHAPE}, "required": ["shape"],
            "additionalProperties": False},
            "description": "the designer's own internal geometry, kept as drawn (rails, a base ring): preserved, "
                           "and in the way of every assembly path"},
        "components": {"type": "array", "minItems": 1, "items": {"type": "object", "properties": {
            "id": {"type": "string"}, "shape": _HANDOFF_SHAPE,
            "role": {"enum": ["switch", "seat", "part"], "description": "switch: the plunger is read off it; "
                                                                        "seat: what the switch bears on"},
            "installedOn": {"type": "string", "description": "a component it is fitted to first and travels with "
                                                             "(a switch on a flap, a connector on a board)"},
            "insertFrom": {"description": "a hint, tried first; with fixed: true taken as given (an axis, a "
                                          "vector or a path of legs, as keepOut.insertFrom)"},
            "fixed": {"type": "boolean", "description": "take insertFrom as given, no search"},
            "external": {"type": "boolean", "description": "comes from outside after the part is closed (a "
                                                           "plug): no clearance entry"},
            "clearance": {"type": "boolean", "description": "a clearance entry for it (default true, except "
                                                            "the switch, its seat and external parts)"},
            "pathTolerance": {"type": "number", "minimum": 0, "description": "mm of interference allowed on its "
                                                                             "way in (a snap bead's reach)"}},
            "required": ["shape"], "additionalProperties": False}},
        "assembly": {"type": "array", "items": {"type": "string"},
                     "description": "component ids in the order they go in (default: as listed); components "
                                    "installedOn another travel with it"},
        "switch": {"type": "object", "description": "the switch block of the study, with: component (the "
                                                    "switch's id), gap (plunger to actuator, default 0.4 mm), "
                                                    "trigger, travel, plungerRadius (default the actuator's), "
                                                    "columnClearance (default 1 mm round the plunger), bearsOn "
                                                    "(a seat component; default the one with role seat), "
                                                    "clearanceMargin (default 0.2 mm), clearanceRegion (the nodes "
                                                    "held to their room, default 'skin'; null for every node), and "
                                                    "the presses, field, "
                                                    "hold, unit... as in a study"},
        "rest": {**_AXIS, "description": "the side it stands on (default the press direction): supports there"},
        "restDepth": {**_POS, "description": "mm of the exterior's far side held (default 0.3)"},
        "supports": {"type": "array", "description": "supports as given instead"},
        "keepOut": {"type": "array", "description": "the designer's own clear zones, as given"},
        "pathTolerance": {"type": "number", "minimum": 0, "description": "mm of interference allowed on a way "
                                                                         "in (crush ribs, a lip; default 0.25)"},
        "maxSlide": {**_POS, "description": "mm, longest first leg of a two-leg path (default 12)"},
        "material": {"type": "string"}, "minMember": _POS, "process": {}, "goal": {"type": "object"},
        "resolution": {"type": "object"}, "loadCases": {"type": "array"}},
    "required": ["exterior", "components", "switch"],
    "additionalProperties": False,
    "$defs": {**_SHAPES, "shape": _SHAPE},
}

_FACE_LIST = {"type": "array", "items": _AXIS}
_SQUEEZE = {"type": "object", "properties": {
    "name": {"type": "string"}, "axis": {"type": "string", "pattern": "^[+-]?[xyz]$"},
    "force": {"type": "number", "exclusiveMinimum": 0, "description": "N, each side"},
    "radius": _POS, "at": {"type": "array", "items": {"type": "number"}, "minItems": 3, "maxItems": 3}},
    "required": ["force"], "additionalProperties": False}
_PRESS = {"type": "object", "properties": {
    "name": {"type": "string"}, "face": _AXIS,
    "at": {"type": "array", "items": {"type": "number"}, "minItems": 2, "maxItems": 3,
           "description": "a point on the exterior, or its two coordinates across the face"},
    "force": {"type": "number", "exclusiveMinimum": 0}, "radius": _POS},
    "required": ["at", "force"], "additionalProperties": False}
_CONNECTOR = {"type": "object", "properties": {
    "component": {"type": "string", "description": "the socket's component id (a board's connector: "
                                                   "installedOn the board, whose mount takes the loads)"},
    "type": {"enum": ["usb-c", "usb-a", "usb-micro-b"],
             "description": "the plug's spec loads (default usb-c: 20 N in, 20 N out, 8 N / 20 N sideways at "
                            "15 mm); any of the numbers below overrides it"},
    "name": {"type": "string", "description": "case name prefix (default the component id)"},
    "out": {**_AXIS, "description": "the way the plug comes out (default: out through the opening the "
                                    "component is aligned to)"},
    "insert": {**_POS, "description": "N pushing the plug home"},
    "withdraw": {**_POS, "description": "N pulling it out"},
    "side": {"type": "object", "properties": {"thin": _POS, "wide": _POS}, "additionalProperties": False,
             "description": "N across the plug's thin and wide sides at `lever`, while it still works"},
    "lever": {**_POS, "description": "mm out from the socket's front edge the side loads act (default 15)"},
    "abuse": {"type": "boolean", "description": "also the wrenching moments at which the plug may break "
                                                "(usb-c 2.0 / 3.5 N·m); default false"},
    "wrench": {"type": "object", "properties": {"thin": _POS, "wide": _POS}, "additionalProperties": False,
               "description": "N·mm wrenching moments, with abuse"},
    "maxShift": {"type": "number", "minimum": 0, "description": "mm the board's mount may move in these cases "
                                                                "(default 0.1; 0 for no cap)"}},
    "required": ["component"], "additionalProperties": False}
_MOUNT_HOLE = {"type": "object", "properties": {
    "at": {"type": "array", "items": {"type": "number"}, "minItems": 2, "maxItems": 3,
           "description": "mm: the hole's centre, [x, y] on the board (z = 0 its underside)"},
    "screw": {"type": "string", "pattern": "^M\\d+(\\.\\d+)?$", "description": "ISO metric size, e.g. 'M2.5'"},
    "diameter": {**_POS, "description": "mm, a screw with no ISO size"},
    "pad": {"type": "number", "minimum": 0, "description": "mm across the copper pad the sleeve bears on"},
    "drill": _POS, "ref": {"type": "string"}}, "required": ["at"], "additionalProperties": False}
_REF_MAP = {"type": "object", "additionalProperties": {"type": "number"}}
_KICAD = {"anyOf": [{"type": "string", "description": "the .kicad_pcb file"}, {"type": "object", "properties": {
    "file": {"type": "string", "description": "the .kicad_pcb file"},
    "models": {"anyOf": [{"type": "string"}, {"type": "array", "items": {"type": "string"}}],
               "description": "folders to find the footprints' 3D models in (also KiCad's own model folder "
                              "variables and the board's folder)"},
    "useModels": {"type": "boolean", "description": "read 3D models for part bodies and heights (default true)"},
    "heights": {**_REF_MAP, "description": "mm a part stands off its face, by reference (over model and "
                                           "package)"},
    "power": {**_REF_MAP, "description": "W of heat a part gives off, by reference (over a footprint Power "
                                         "field)"},
    "maxTemp": {**_REF_MAP, "description": "C a part must stay under, by reference"},
    "masses": {**_REF_MAP, "description": "g, by reference (over the volume estimate)"},
    "connectors": {"type": "object", "additionalProperties": {"type": "object", "properties": {
        "opening": {"type": "string", "description": "the exterior opening it faces out through (alignTo)"},
        "recess": {"type": "number", "minimum": 0, "description": "mm short of the exterior (default 0.3)"},
        "out": {**_SIGNED_AXIS, "description": "which way it faces on the board, when the board cannot say"},
        "type": {"enum": ["usb-c", "usb-a", "usb-micro-b"], "description": "the plug, when the footprint name "
                                                                         "does not say"},
        "plugLoads": {"type": "boolean", "description": "write its plug loads (loads.connector) when aligned "
                                                        "to an opening and typed (default true)"},
        "insert": _POS, "withdraw": _POS, "side": {"type": "object"}, "lever": _POS, "abuse": {"type": "boolean"},
        "wrench": {"type": "object"}, "maxShift": {"type": "number", "minimum": 0}},
        "additionalProperties": False}, "description": "by reference: what each connector faces and carries"},
    "separate": {"type": "array", "items": {"type": "string"},
                 "description": "more parts (references) to make components of their own, installed on the "
                                "board (connectors and parts with power always are)"},
    "defaultHeight": {**_POS, "description": "mm for a part with no model, field or known package (default 3; "
                                             "each one is listed)"},
    "leadProtrusion": {"type": "number", "minimum": 0, "description": "mm through-hole leads stand out of the far "
                                                                      "face (default 1.5)"},
    "edgeReach": {"type": "number", "minimum": 0, "description": "mm from the board edge a connector faces out "
                                                                 "over it (default 2)"},
    "copperFill": {"type": "number", "minimum": 0, "maximum": 1,
                   "description": "copper area fraction per layer, for mass and conductivity (default 0.5)"}},
    "required": ["file"], "additionalProperties": False}],
    "description": "a KiCad board (loadpath/kicad.py): the outline, stackup thickness and every part's body "
                   "(3D model, else footprint outline at its package height), mass, the mounting holes it is "
                   "screwed down by, its connectors (each its own component on the board, facing out over its "
                   "edge, with plug loads when aligned to an opening) and the heat its parts give off"}

ENCLOSURE_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "loadpath enclosure brief",
    "description": "`loadpath enclosure`: a product from its outside in (enclosure.py). The designer's exterior is "
                   "fixed (kept as the study's skin, never edited); the components are placed inside it "
                   "(placement.py) or taken as given, each gets its assembly path, a mount, and its share of the "
                   "loads (drops, squeezes, presses, plugs, heat, a push pad, feel targets); the interior is designed.",
    "type": "object",
    "properties": {
        "name": {"type": "string"}, "description": {"type": "string"},
        "exterior": {**_HANDOFF_SHAPE, "description": "the designer's exterior surfacing, a closed solid"},
        "skin": {"anyOf": [_POS, {"type": "object"}], "description": "the skin kept of the exterior (default 1.0 "
                                                                     "mm; as a study's skin)"},
        "openings": {"type": "array", "items": {"anyOf": [_HANDOFF_SHAPE, {"type": "object", "properties": {
            "id": {"type": "string"}, "shape": _HANDOFF_SHAPE, "inward": _AXIS, "depth": _POS},
            "required": ["shape"], "additionalProperties": False}]},
            "description": "where the exterior opens into the interior (a port, a door): no skin across them, a "
                           "component may reach into one (alignTo), and assembly paths come in through them. "
                           "inward defaults to the axis toward the exterior's centre"},
        "fixed": {"type": "array", "items": {"type": "object", "properties": {
            "id": {"type": "string"}, "shape": _HANDOFF_SHAPE}, "required": ["shape"],
            "additionalProperties": False}, "description": "the designer's own internal geometry, kept as drawn"},
        "components": {"type": "array", "minItems": 1, "items": {"type": "object", "properties": {
            "id": {"type": "string"},
            "shape": {**_HANDOFF_SHAPE, "description": "its CAD: in product coordinates when place is given, "
                                                       "in its own frame when placed"},
            "size": {"type": "array", "items": _POS, "minItems": 3, "maxItems": 3,
                     "description": "or a box this size (mm), centred on its own origin"},
            "center": {"type": "array", "items": {"type": "number"}, "minItems": 3, "maxItems": 3,
                       "description": "with size: the box's centre in its own frame (a part on a board)"},
            "mass": {"type": "number", "minimum": 0, "description": "kg: carried through its mount in drops"},
            "watts": {"type": "number", "minimum": 0, "description": "W of heat it gives off"},
            "maxTemp": {"type": "number", "description": "C it must stay under (for a cooling block)"},
            "conductivity": {**_POS, "description": "W/(m K), for a cooling block (default 20: a package on a "
                                                    "copper-filled board; a cell is nearer 1.5)"},
            "place": {"enum": ["auto", "given"], "description": "auto: placed by the agent; given: as drawn. "
                                                                "Default auto when it has size or any placement "
                                                                "rule, given otherwise"},
            "rotate": {"anyOf": [{"enum": ["any", "none", "x", "y", "z"]}, {"type": "array"}],
                       "description": "quarter turns it may take: any (24), about one axis (4), none"},
            "against": {**_AXIS, "description": "pushed to that side of the cavity (a battery on the floor)"},
            "alignTo": {"type": "object", "properties": {
                "opening": {"type": "string"}, "face": _AXIS,
                "recess": {"type": "number", "minimum": 0, "description": "mm short of the exterior (default 0.3)"}},
                "required": ["opening"], "additionalProperties": False,
                "description": "its face (in its own frame) out through an opening, on the opening's axis"},
            "under": {"type": "object", "properties": {
                "at": {"type": "array", "items": {"type": "number"}, "minItems": 3, "maxItems": 3},
                "direction": _AXIS, "face": _AXIS, "gap": {"type": "number", "minimum": 0}},
                "required": ["at"], "additionalProperties": False,
                "description": "its face on the line through `at` along `direction` (the press), `gap` under "
                               "the skin (a switch under a press point)"},
            "near": {"type": "array", "items": {"type": "string"}, "description": "components to keep it close to"},
            "margin": {"type": "number", "minimum": 0, "description": "mm of room kept round it (default "
                                                                      "placement.margin)"},
            "installedOn": {"type": "string", "description": "fitted to another component first and placed and "
                                                             "assembled with it; its CAD in that one's frame"},
            "mount": {"anyOf": [{"type": "boolean"}, {"enum": ["cradle", "holes"]}, {"type": "object", "properties": {
                "type": {"enum": ["cradle", "holes"]}, "side": _AXIS, "thickness": _POS,
                "holes": {"type": "array", "items": _MOUNT_HOLE,
                          "description": "holes: the screw holes, in the component's own frame"},
                "threadLength": {**_POS, "description": "holes: mm of thread each screw engages (default 2 d)"},
                "driver": {"type": "boolean", "description": "holes: keep the driver's line from each head to "
                                                             "the skin clear (default true)"}},
                "additionalProperties": False}],
                "description": "how it is held: a preserved cradle under its face on `side` (default the "
                               "`against` side, else its nearest wall), or (`holes`, a KiCad board's default "
                               "when it has mounting holes) screws through its holes into threaded sleeves "
                               "on its underside (or the side nearest `side`), shank, head and driver kept "
                               "clear; false for none. Its group's mass is carried through it"},
            "kicad": _KICAD,
            "role": {"enum": ["switch", "seat", "part"], "description": "with a switch block (see handoff)"},
            "external": {"type": "boolean", "description": "put on from outside (a door, a plug): no path search"},
            "insertFrom": {"description": "a hint for its assembly path, tried first"},
            "fixed": {"type": "boolean", "description": "take insertFrom as given"},
            "clearance": {"type": "boolean"},
            "tilt": {"anyOf": [{"type": "boolean"}, {"type": "number", "minimum": 0, "maximum": 90}],
                     "description": "how far this part may be turned on its way in, degrees (default the "
                                    "brief's tilt); false or 0: straight slides only (a cell that must stay "
                                    "level, a part on a rail)"},
            "pathTolerance": {"type": "number", "minimum": 0}},
            "x-requireOne": [["shape"], ["size"], ["kicad"]], "additionalProperties": False}},
        "assembly": {"type": "array", "items": {"type": "string"}, "description": "order the components go in"},
        "tilt": {"anyOf": [{"type": "boolean"}, {"type": "number", "minimum": 0, "maximum": 90}],
                 "description": "how far a part may be turned from how it sits on its way in, degrees (default "
                                "45). A part with no way in by straight slides is tried turning: tilted in "
                                "through an opening shorter than it is, pivoted about an edge, laid flat "
                                "(motion.py). The way in found has the least tilt; enclosure.json's "
                                "assembly.steps says it in numbered steps. false or 0: straight slides only"},
        "placement": {"type": "object", "properties": {
            "pitch": _POS, "beam": {"type": "integer", "minimum": 1, "maximum": 8},
            "margin": {"type": "number", "minimum": 0, "description": "mm round every part (default 0.5)"},
            "centerOfMass": {"type": "object", "properties": {
                "target": {"type": "array", "items": {"type": "number"}, "minItems": 3, "maxItems": 3},
                "low": _AXIS, "weight": _POS}, "additionalProperties": False},
            "centerWeight": {"type": "number", "minimum": 0},
            "assemblyAware": {"type": "boolean", "description": "keep only positions each part can still be "
                                                                "assembled into (default true)"}},
            "additionalProperties": False},
        "loads": {"type": "object", "properties": {
            "drop": {"type": "object", "properties": {
                "g": {"type": "number", "exclusiveMinimum": 0, "description": "design deceleration in g (default 50)"},
                "faces": _FACE_LIST, "contactDepth": _POS}, "additionalProperties": False,
                "description": "a drop onto each face: every component's mass x g toward it, through its mount, "
                               "the floor's reaction on that face of the exterior"},
            "squeeze": {"anyOf": [_SQUEEZE, {"type": "array", "items": _SQUEEZE}],
                        "description": "a hand squeezing across an axis"},
            "press": {"anyOf": [_PRESS, {"type": "array", "items": _PRESS}],
                      "description": "a finger or a weight on the exterior, held by the far face"},
            "connector": {"anyOf": [_CONNECTOR, {"type": "array", "items": _CONNECTOR}],
                          "description": "a plug pushed into a component's socket, pulled out and levered "
                                         "sideways (connectors.py), through the mount its board sits on, the "
                                         "hand holding the face across from the port; the board's travel is "
                                         "capped (maxShift) so the socket's solder joints are not flexed"},
            "rattle": {"type": "object", "properties": {
                "band": {**_POS, "description": "handling band, Hz: each mounted component's resonance is kept "
                                                "above it (default 150)"},
                "maxAmplification": {**_POS, "description": "worst amplification of the hand's motion at the "
                                                            "component over the band (default 1.25)"},
                "g": {**_POS, "description": "handling acceleration, g: a knock or a shake (default 5)"},
                "maxTravel": {**_POS, "description": "mm a component may move on its mount under that "
                                                    "acceleration before it rattles against its neighbours "
                                                    "(default 0.1)"},
                "hold": {"description": "where the hand holds the product, a feel location (default the whole "
                                        "skin)"},
                "components": {"type": "array", "items": {"type": "string"},
                               "description": "component ids to check (default every component with a mass "
                                              "and a mount)"}},
                "additionalProperties": False,
                "description": "no component rattles or rings when the product is handled: an isolate "
                               "(upTo) feel target per mounted component, between the hand and its mount"}},
            "additionalProperties": False},
        "rest": {**_AXIS, "description": "the side it is held by for the analysis (default -z): a small patch of "
                                         "the exterior there"},
        "supportRadius": _POS, "supports": {"type": "array"}, "keepOut": {"type": "array"},
        "loadCases": {"type": "array", "description": "more load cases, as in a study"},
        "switch": {"type": "object", "description": "a push pad, as in a handoff brief"},
        "feel": {"type": "object", "description": "feel targets, as in a study"},
        "thermal": {"type": "object"}, "cooling": {"type": "object"}, "split": {"type": "object"},
        "pathTolerance": {"type": "number", "minimum": 0}, "maxSlide": _POS,
        "material": {"type": "string"}, "minMember": _POS, "process": {}, "goal": {"type": "object"},
        "resolution": {"type": "object"}, "maxIterations": {"type": "integer"}, "printDirection": {},
        "drains": {}},
    "required": ["exterior", "components"],
    "additionalProperties": False,
    "$defs": {**_SHAPES, "shape": _SHAPE},
}


# --- wires: cables laid with rod physics (wires.py) ---

_CABLE = {"anyOf": [
    {"type": "string", "description": "a name: awg<N>-<insulation> ('awg26-pvc', 'awg22-silicone'), "
                                      "ffc-<conductors>x<pitch> ('ffc-10x0.5'), fpc-<width>x<thickness>, usb2, "
                                      "coax-<od>"},
    {"type": "object", "properties": {
        "name": {"type": "string"},
        "type": {"enum": ["wire", "bundle", "cable", "coax", "ffc", "fpc", "round", "flat"],
                 "description": "wire: one insulated wire; bundle: `wires` of them side by side in a sleeve; "
                                "cable: `wires` in a jacket; coax; ffc: a flat flexible cable (conductors, "
                                "pitch); fpc: a flex circuit (width, thickness); round / flat: by its numbers"},
        "awg": {"type": "number"}, "conductorArea": {**_POS, "description": "mm^2, instead of awg"},
        "insulation": {"enum": ["pvc", "pvc105", "silicone", "ptfe", "fep", "tpe", "xlpe", "pur"]},
        "strands": {"type": "integer", "minimum": 1}, "solid": {"type": "boolean"},
        "strandCoupling": {**_POS, "description": "stiffness over the slipped-strand bound (default 1)"},
        "wires": {"type": "integer", "minimum": 1}, "wall": _POS,
        "jacket": {"enum": ["pvc", "pvc105", "silicone", "ptfe", "fep", "tpe", "xlpe", "pur"]},
        "jacketWall": _POS, "shield": {"type": "boolean"},
        "od": {**_POS, "description": "mm across (round cables)"}, "core": _POS,
        "dielectric": {"type": "string"},
        "conductors": {"type": "integer", "minimum": 1}, "pitch": _POS,
        "width": _POS, "thickness": _POS, "copper": {"type": "number", "minimum": 0},
        "copperFill": {"type": "number", "minimum": 0, "maximum": 1}, "film": {"enum": ["polyimide", "pet"]},
        "bendStiffness": {"anyOf": [_POS, {"type": "array", "items": _POS, "minItems": 2, "maxItems": 2}],
                          "description": "N mm^2, measured or from a datasheet (wires.stiffness_from_droop); "
                                         "[through the thickness, edgewise] for a flat"},
        "twistStiffness": {**_POS, "description": "N mm^2"},
        "minBendRadius": {"anyOf": [_POS, {"type": "object", "properties": {"static": _POS, "dynamic": _POS},
                                           "additionalProperties": False}],
                          "description": "mm, fixed and flexing; default house rules (3x and 10x a wire's "
                                         "diameter, 10x and 100x a flat's thickness)"},
        "maxTwist": {"anyOf": [_POS, {"type": "object", "properties": {"static": _POS, "dynamic": _POS},
                                      "additionalProperties": False}], "description": "deg per m"},
        "gramsPerMeter": {"type": "number", "minimum": 0}, "maxTemp": {"type": "number"}},
     "additionalProperties": False}]}
_PORT_LIMITS = {
    "relief": {"type": "number", "minimum": 0, "description": "mm the cable leaves straight (a boot)"},
    "maxPull": {**_POS, "description": "N the connector takes along its axis"},
    "maxSide": {**_POS, "description": "N across it"},
    "maxMoment": {**_POS, "description": "N mm about it"}}
_SIGNED = {"anyOf": [{"type": "string", "pattern": "^[+-]?[xyz]$"}, _VEC3]}
_WIRE_PORT = {"type": "object", "properties": {
    "at": {**_VEC3, "description": "mm: where the cable leaves the connector"},
    "direction": {**_SIGNED, "description": "the way the cable leaves it"},
    "up": {**_SIGNED, "description": "a flat cable's face normal there (which way its contacts face)"},
    "body": {"type": "string", "description": "the body the port is on: it moves with it"},
    **_PORT_LIMITS}, "required": ["at", "direction"], "additionalProperties": False}
_VIA = {"type": "array", "items": {"type": "object", "properties": {
    "at": _VEC3, "along": {"type": "number", "minimum": 0, "description": "mm along the cable (default: its "
                                                                         "nearest point)"},
    "body": {"type": "string"}}, "required": ["at"], "additionalProperties": False},
    "description": "clips: points the cable is held at"}


def _wire_item(port):
    return {"type": "object", "properties": {
        "id": {"type": "string"}, "cable": _CABLE, "from": port, "to": port,
        "length": {**_POS, "description": "mm, port to port: a cable cut to length. Default: the shortest way "
                                          "round plus slack"},
        "slack": {"type": "number", "minimum": 0, "description": "over the route's length (default 0.08)"},
        "minLength": _POS,
        "current": {"type": "number", "minimum": 0, "description": "A: resistance, voltage drop, self-heating"},
        "via": _VIA,
        "routes": {"type": "integer", "minimum": 1, "maximum": 4, "description": "ways round to try (default 1)"},
        "roll": {"type": "boolean", "description": "hold the far end's roll about its own axis (default: a "
                                                   "flat yes, a round wire no)"},
        "segments": {"type": "integer", "minimum": 8, "maximum": 400},
        "reach": {**_POS, "description": "enclosure only: mm the later part must be able to stand out of its "
                                         "place with the cable already plugged in (default 12): room for "
                                         "fingers to mate the connector"},
        "plugged": {"enum": ["before", "after"], "description": "enclosure only: 'before' (default) the cable "
                                                                "is plugged in while the later part is still "
                                                                "outside and follows it home; 'after' its "
                                                                "connector is reached once the part is in"}},
        "required": ["cable", "from", "to"], "additionalProperties": False}


_MOTION = {"type": "object", "properties": {
    "type": {"enum": ["revolute", "prismatic", "path"]},
    "axis": _SIGNED, "origin": _VEC3,
    "range": {"type": "array", "items": {"type": "number"}, "minItems": 2, "maxItems": 2,
              "description": "degrees (revolute) or mm (prismatic); the cables are laid at 0"},
    "legs": {"type": "array", "items": {"type": "object", "properties": {"toward": _SIGNED, "length": _POS},
                                        "required": ["toward", "length"], "additionalProperties": False},
             "description": "type path: straight legs the moving bodies travel, in order"},
    "steps": {"type": "integer", "minimum": 2, "maximum": 200},
    "flexing": {"type": "boolean", "description": "false: a move made once (a crash, a part shifting on its "
                                                  "strap, a lid opened for service), held to the cable's fixed "
                                                  "bend radius; default true: a joint that cycles, held to its "
                                                  "flexing radius"}}, "additionalProperties": False,
    "description": "what the bodies marked `moves` do; every cable with a port on one is followed through it"}
WIRES_SCHEMA = {
    "$schema": _DIALECT,
    "title": "loadpath wires brief",
    "description": "`loadpath wires`: a product's cables laid and checked with rod physics (wires.py, rods.py). "
                   "Each cable is a Cosserat rod with its own bending and twist stiffness, started along the "
                   "clearance-aware shortest way between its ports and settled against the bodies; checked for "
                   "bend radius, twist, tension, pinching and connector loads, and followed through a motion "
                   "(a joint's range, a part's way in).",
    "type": "object",
    "properties": {
        "name": {"type": "string"}, "description": {"type": "string"},
        "bodies": {"type": "array", "items": {"type": "object", "properties": {
            "id": {"type": "string"}, "shape": _HANDOFF_SHAPE,
            "inside": {"type": "boolean", "description": "a cavity the cables must stay in (an exterior)"},
            "skin": {"type": "number", "minimum": 0, "description": "with inside: wall thickness kept off"},
            "moves": {"type": "boolean", "description": "goes with the motion"},
            "color": {"type": "string"}}, "required": ["shape"], "additionalProperties": False},
            "description": "what the cables must go round (or stay inside)"},
        "wires": {"type": "array", "minItems": 1, "items": _wire_item(_WIRE_PORT)},
        "motion": _MOTION,
        "gravity": {**_SIGNED, "description": "which way is down; cables then carry their weight"},
        "pitch": {**_POS, "description": "mm: the grid each body's distance field is sampled on (default 0.4)"},
        "clearance": {"type": "number", "minimum": 0, "description": "mm kept between a cable and a body"},
        "routes": {"type": "integer", "minimum": 1, "maximum": 4},
        "wiresCollide": {"type": "boolean", "description": "later cables go round earlier ones (default true; "
                                                           "not during a motion)"},
        "channels": {"anyOf": [{"type": "boolean"}, {"type": "object", "properties": {
            "clearance": {"type": "number", "minimum": 0}}, "additionalProperties": False}],
            "description": "write each cable's channel (the tube it occupies over the motion) as an STL"}},
    "required": ["wires"],
    "additionalProperties": False,
    "$defs": {**_SHAPES, "shape": _SHAPE},
}

# an enclosure's cables: ports on its components, wires between them, and
# heat in the placement search (enclosure.py, placement.py)
_ENC = ENCLOSURE_SCHEMA["properties"]
_ENC["components"]["items"]["properties"]["ports"] = {
    "type": "array", "items": {"type": "object", "properties": {
        "id": {"type": "string"},
        "at": {**_VEC3, "description": "mm, in the component's own frame: where a cable leaves it"},
        "direction": {**_SIGNED, "description": "the way the cable leaves, in the component's own frame"},
        "up": {**_SIGNED, "description": "a flat cable's face normal there"},
        **_PORT_LIMITS}, "required": ["id", "at", "direction"], "additionalProperties": False},
    "description": "its connectors: where cables leave it and which way (wires)"}
_ENC["wires"] = {
    "type": "array", "items": _wire_item({"anyOf": [
        {"type": "string", "pattern": r"^[^.]+\.[^.]+$", "description": "'component.port'"}, _WIRE_PORT]}),
    "description": "the cables between the components' ports (or fixed points in product coordinates). A "
                   "layout is kept only if every cable can be laid in it within its bend radius and length; "
                   "each is then followed along its part's way in, and its channel is kept clear of material"}
_ENC["placement"]["properties"]["heat"] = {
    "anyOf": [{"type": "boolean"}, {"type": "object", "properties": {
        "weight": {"type": "number", "minimum": 0, "description": "mm of layout cost per unit of a part's "
                                                                  "temperature budget used (default 30)"},
        "ambient": {"type": "number", "description": "C (default 25)"},
        "conductivity": {**_POS, "description": "W/(m K) of the stuff between the parts (default 0.1)"},
        "up": {**_AXIS, "description": "which way is up in use, when there is one: heat rises, so a part "
                                       "above a hot one is charged half as much again"}},
        "additionalProperties": False}],
    "description": "keep parts that make heat (watts) away from each other and from parts with a low maxTemp, "
                   "more so when the hot one sits below (`up`): a point-source estimate that ranks "
                   "positions; the cooling block is the real check. Default on when any component has watts"}
_ENC["wiresOptions"] = {"type": "object", "properties": {
    "pitch": {**_POS, "description": "mm: the grid each body is sampled on for cable contact (default 0.4)"},
    "channelClearance": {"type": "number", "minimum": 0, "description": "mm kept clear round each cable "
                                                                        "(default 0.6)"}},
    "additionalProperties": False}
_ENC["placement"]["properties"]["wires"] = {
    "type": "object", "properties": {
        "weight": {"type": "number", "minimum": 0, "description": "layout cost per mm of cable (default 0.15)"},
        "physics": {"type": "boolean", "description": "lay each candidate's cables as rods (default true); "
                                                      "false: the routed length only"}},
    "additionalProperties": False}

_STACK_PORT = {"type": "object", "properties": {
    "id": {"type": "string"},
    "at": {**_VEC3, "description": "mm in the board's frame: the origin at its centre, z = 0 its underside"},
    "direction": {**_SIGNED, "description": "the way the cable leaves it"},
    "up": _SIGNED, **_PORT_LIMITS}, "required": ["id", "at", "direction"], "additionalProperties": False}
STACK_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "loadpath stack",
    "description": "Boards on standoffs: the order and the gaps searched for board temperature, centre of mass "
                   "and height, with the cables between boards laid as rods (stack.py).",
    "type": "object",
    "properties": {
        "name": {"type": "string"}, "description": {"type": "string"},
        "boards": {"type": "array", "minItems": 1, "maxItems": 6, "items": {"type": "object", "properties": {
            "id": {"type": "string"},
            "size": {**_VEC3, "description": "mm: [x, y, thickness]"},
            "mass": {"type": "number", "minimum": 0, "description": "kg"},
            "watts": {"type": "number", "minimum": 0, "description": "W of heat it gives off"},
            "maxTemp": {"type": "number", "description": "C the board may run at"},
            "above": {"type": "number", "minimum": 0, "description": "mm its tallest part stands over its top face"},
            "below": {"type": "number", "minimum": 0, "description": "mm its tallest part hangs under it"},
            "position": {"enum": ["top", "bottom"], "description": "where it has to be (a gyro on top, say)"},
            "ports": {"type": "array", "items": _STACK_PORT},
            "kicad": {**_KICAD, "description": "a KiCad board: size, mass, watts, maxTemp, above, below and ports "
                                               "read off it (kicad.stack_board); what the entry says wins"}},
            "required": ["id"], "x-requireOne": [["size"], ["kicad"]], "additionalProperties": False}},
        "space": {"type": "object", "properties": {
            "height": {**_POS, "description": "mm: z the stack must stay under (the keep-out box's top)"},
            "base": {"type": "number", "description": "mm: z of the plate it stands on (default 0)"},
            "baseGap": {"type": "number", "minimum": 0, "description": "mm under the bottom board (default 2)"},
            "center": {"type": "array", "items": {"type": "number"}, "minItems": 2, "maxItems": 2},
            "openTop": {"type": "boolean", "description": "the top board's top face sees the air (default true)"},
            "openBottom": {"type": "boolean"}}, "additionalProperties": False},
        "standoffs": {"type": "object", "properties": {
            "pattern": _POS, "count": {"type": "integer", "minimum": 0}, "diameter": _POS,
            "material": {"enum": ["nylon", "aluminium", "aluminum", "brass", "steel", "tpu"]},
            "fill": {"type": "number", "exclusiveMinimum": 0, "maximum": 1,
                     "description": "solid fraction of the standoff's section (default 0.6: a threaded spacer)"},
            "baseGap": {"type": "number", "minimum": 0}}, "additionalProperties": False},
        "airflow": {"type": "object", "properties": {
            "speed": {"type": "number", "minimum": 0, "description": "m/s of the stream the stack sits in"},
            "direction": {"type": "string", "pattern": "^[+-]?[xy]$", "description": "along the boards"}},
            "additionalProperties": False},
        "ambient": {"type": "number", "description": "C (default 25)"},
        "clearance": {"type": "number", "minimum": 0, "description": "mm between one board's parts and the next's"},
        "gapStep": _POS, "maxExtraGap": {"type": "number", "minimum": 0},
        "alternatives": {"type": "integer", "minimum": 0}, "maxLays": {"type": "integer", "minimum": 1},
        "goal": {"type": "object", "properties": {
            "centerOfMass": {"anyOf": [{"enum": ["low", "high"]}, {"type": "number", "description": "mm: target z"}]},
            "weights": {"type": "object", "properties": {
                "temperature": {"type": "number", "minimum": 0}, "centerOfMass": {"type": "number", "minimum": 0},
                "height": {"type": "number", "minimum": 0}}, "additionalProperties": False}},
            "additionalProperties": False},
        "mount": {"anyOf": [{"type": "string"}, {"type": "object"}],
                  "description": "the frame's region id (or a shape) the stack stands on: copied into the result's "
                                 "inertiaRelief.masses entry"},
        "wires": {"type": "array", "items": _wire_item({"anyOf": [
            {"type": "string", "pattern": "^[^.]+\\.[^.]+$", "description": "'board.port'"}, _WIRE_PORT]})}},
    "required": ["boards"],
    "additionalProperties": False,
}

SCHEMAS = {"stack": STACK_SCHEMA, "study": STUDY_SCHEMA, "heatExchanger": HEAT_EXCHANGER_SCHEMA, "lattice": LATTICE_SCHEMA,
           "field": FIELD_SCHEMA, "handoff": HANDOFF_SCHEMA, "split": SPLIT_SCHEMA,
           "enclosure": ENCLOSURE_SCHEMA, "wires": WIRES_SCHEMA}


def schema(kind: str = "study") -> dict:
    """A copy of one schema: 'study', 'heatExchanger', 'lattice', 'field', 'handoff', 'split', 'enclosure',
    'wires' or 'stack'."""
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
                if k in ("cycles", "hours", "temperature") and ".loads[" in path:
                    out.append(("error", f"{path}.{k}", f"{k} goes on the load case, not on a load "
                                "(it applies to the whole case)"))
                elif near:
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
    if kind == "study" and isinstance(doc, dict) and not doc.get("supports") and not doc.get("feel") \
            and not doc.get("cooling") and not doc.get("inertiaRelief"):
        # beyond JSON Schema: supports are optional only for pressure vessels
        cases = doc.get("loadCases") or ([{"loads": doc["loads"]}] if isinstance(doc.get("loads"), list) else [])
        loads = [ld for c in cases if isinstance(c, dict) for ld in (c.get("loads") or [])]
        if not loads or not all(isinstance(ld, dict) and ld.get("pressure") for ld in loads):
            out.append(("error", "$.supports", "needs at least one support (where the part is held); only a "
                                               "study whose every load is a pressure, or one with inertiaRelief "
                                               "(a free part), may leave them out"))
    return [{"level": lv, "path": p, "message": m} for lv, p, m in out]


def format_issues(issues: list[dict]) -> str:
    return "\n".join(f"{i['level']}: {i['path']}: {i['message']}" for i in issues)
