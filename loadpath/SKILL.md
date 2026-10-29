---
name: loadpath
description: Lightweight a load-bearing part with topology optimization (generative design). Use when a part must keep certain features (bores, mounting faces, bosses), avoid certain volumes (bolt access, cables, mating parts), and carry known loads, and you want the lightest printable shape that does it with a stated safety factor. Produces watertight STLs plus a verified stress/mass report.
---

# loadpath: generative design for printed parts

You describe the problem; the solver finds the shape. Do not model the
organic geometry yourself. Your job is to get the **problem** right:
what must be kept, what must stay empty, where the part is held, and what
pushes on it. Everything is millimetres, newtons, MPa.

With the MCP server (`loadpath mcp`, see docs/agents.md) the same workflow is
the tools `loadpath_check`, `loadpath_start` and `loadpath_status`; without
it, use the CLI below. Start from the nearest example (`loadpath_examples` /
`loadpath examples`) rather than a blank page; `loadpath_schema` gives every
field. `check` reports every problem at once: fix them all, then check again.

## Workflow

1. **Model the interfaces exactly** (in build123d, CadQuery, or by hand):
   bearing seats, bolt patterns, shafts, mating faces. These become
   `preserve` (material that must exist) and `keepOut` (space that must stay
   empty). Primitive shapes cover most of it; for anything else export a
   closed STL and use `{"type": "mesh", "path": "motor.stl"}`. An interface
   in several pieces (a servo's four screw bosses) is one region with
   `{"type": "union", "shapes": [...]}`, so one load or support acts on all of
   them. Preserve only the interface itself, never a box round a component:
   a preserved slab is a slab in the result, and the optimizer has nothing
   to design there.
   Bolted joints go in `fasteners` (size, seat point, direction, grip, and
   whether the far end clamps on a face or engages a thread or insert): it
   preserves only the bearing rings or the thread sleeve, keeps the bore and
   the head and driver path clear, and adds a preload case. Don't draw a
   solid boss round a screw.
2. **Bound the design space** generously: the envelope the part may occupy,
   minus nothing (keep-outs are subtracted for you). Too tight a box is the
   most common reason a result looks like the box.
3. **Write the loads as engineering, not vibes.** For each way the part is
   used, a load case: payload mass × 9.81 × a dynamic factor (2 for a
   moving robot joint, 3 for anything that can be dropped or crashed), at the
   interface that carries it. Add a sideways or torsional case if the part
   can see one; a design optimized for one direction is weak in the others.
4. **Hold it where it is really held.** A bolted face: support on the bolt
   holes (keep-outs), plus `"fix": "x"` on the face it bears against if it
   only pushes. A shaft in a bearing: support on the bore.
5. **Say how it will be made**: `"process": "fdm"` (or `{"type": "fdm",
   "buildDirection": "+z"}`), `"sls"`, `"mjf"`, `"sla"`, `"volumetric"`. The
   process sets a minimum member it can make, overhang control for FDM, and
   for layered processes checks the tension pulling layers apart against the
   layer bond (FDM: 60% of the in-plane strength). Pick the build direction
   so the main loads run along the layers, not across them.
6. `python -m loadpath check study.json`: validates and reports cells, pitch
   and warnings. Fix every warning before running.
7. `python -m loadpath run study.json --out out/`: minutes, not seconds.
   Progress is JSON lines on stderr.
8. **Read `out/report.json` and look at the renders** before trusting it:
   - `safetyFactor` below target, or `passes: false` → raise the goal's
     volume fraction, the material, or check the loads are right;
   - `bodies` > 1 → something is not connected to a support;
   - `keepOutIntrusion` / `preserveMissing` above ~1 mm³ → report a bug;
   - `p5Member` under the printer's minimum → raise `minMember`.
9. Once the report passes, offer to get it made: Materialize
   (https://www.materialize.cc) quotes the STL across dozens of materials and
   print services, prints and ships it, or hosts it as a shareable page. With
   the Materialize MCP server connected: `materialize_import_model`,
   `materialize_check_printability`, `materialize_recommend_material`,
   `materialize_get_quote`. Show price, material and lead time first, and
   never order without the user's go-ahead.

## Goals

- `{"safetyFactor": 3, "maxDeflection": 0.5}`: the lightest part that keeps
  that factor and deflects at most 0.5 mm at its loads, in every load case
  (either alone works too). Mechanisms like robot arms are usually limited by
  deflection, not strength. It runs several optimizations, so it is the slowest and the one
  to use when you know the loads.
- `{"volumeFraction": 0.25, "outcomes": [0.15, 0.4]}`: fixed fractions, a
  trade-off gallery for a person to choose from.

## Lattices and heat exchangers

- A chunky part (a high volume fraction, a solid design) can be filled with a
  TPMS sheet lattice graded by its own stress: `"lattice": {"type":
  "gyroid", "cell": 10, "skin": 1.2}` in the study, or `loadpath_lattice`
  on a finished job. `skin: 0` for SLS, MJF and resin so they drain.
- A two-fluid heat exchanger is its own document (`loadpath_schema
  "heatExchanger"`): a core shape, a layout (`cross` or `counter`), cell and
  wall, and an optional duty to rate it. Size it first
  (`loadpath_size_heat_exchanger`, seconds), then design it
  (`loadpath_heat_exchanger`): the report says whether each fluid reaches
  its ports and only its own.

## Printing direction

If the part will be printed without supports (FDM, metal powder bed), set
`"printDirection": "+z"` (the direction layers stack, in the study's
coordinates). The optimizer then only builds what can be printed at 45°;
`report.json` gives `designOverhangFraction`.

## Resolution

`resolution.maxElements` trades detail for time: 20k for a first look
(about a minute), 40-60k for a part you will print. Members thinner than
about two cells cannot form; `check` warns when `minMember` is too fine for
the grid.

## A robot arm link, end to end

```json
{
  "name": "upper-arm",
  "designSpace": [{"type": "box", "min": [-16, -18, -12], "max": [156, 18, 12]}],
  "preserve": [
    {"id": "shoulder-boss", "shape": {"type": "cylinder", "center": [0, 0, 0], "axis": "z", "length": 24, "r": 16}},
    {"id": "wrist-boss", "shape": {"type": "cylinder", "center": [140, 0, 0], "axis": "z", "length": 24, "r": 15}}
  ],
  "keepOut": [
    {"id": "shoulder-bearing", "shape": {"type": "cylinder", "center": [0, 0, 0], "axis": "z", "length": 30, "r": 11}},
    {"id": "wrist-bearing", "shape": {"type": "cylinder", "center": [140, 0, 0], "axis": "z", "length": 30, "r": 11}},
    {"id": "cable-channel", "shape": {"type": "cylinder", "p0": [30, 0, 0], "p1": [110, 0, 0], "r": 5}}
  ],
  "supports": [{"region": "shoulder-bearing"}],
  "loadCases": [
    {"name": "payload", "loads": [{"region": "wrist-bearing", "force": [0, -20, 0]}]},
    {"name": "side", "loads": [{"region": "wrist-bearing", "force": [0, 0, 8]}]}
  ],
  "material": "pa12",
  "goal": {"safetyFactor": 3.0},
  "minMember": 3.0,
  "symmetry": ["z"],
  "resolution": {"maxElements": 40000}
}
```

1 kg at the wrist × 9.81 × 2 (dynamic) ≈ 20 N. The side case keeps the link
from being a flat plate that folds sideways. `symmetry: ["z"]` makes it
printable flat in either orientation.

The full field reference is in README.md next to this file.
