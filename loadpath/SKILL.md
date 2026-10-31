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
   When the direction is genuinely unknown (contact, bumps, handling, a
   push at the tool "from anywhere"), say so instead of guessing:
   `{"region": "tool", "force": [0, 0, -20], "direction": "any"}` designs
   and checks for the force's size in its worst direction, and `"cone": 30`
   for anything within 30 degrees of the written one. The worst case is
   exact (three solves, a 3x3 eigenproblem), and `report.json` gives each
   case's `worstDirection`. It costs stiffness in the nominal direction (the
   arm link: about half again more compliant under its payload), so use it
   for the uncertain load and keep known loads as plain vectors.
4. **Hold it where it is really held.** A bolted face: support on the bolt
   holes (keep-outs), plus `"fix": "x"` on the face it bears against if it
   only pushes. A shaft in a bearing: support on the bore.
   A housing that holds a fluid (a boost plenum, a coolant jacket): make
   the fluid's volume a keep-out with a `wall` (mm), and load it with
   `{"region": "<id>", "pressure": MPa}`. The wall stays as given and the
   optimizer ribs it. A sealed vessel needs no supports; anything with an
   open port or a flange is held where it is really held. For a cover or
   panel that must stay closed, `"skin": {"thickness": t, "open": ["-z"]}`
   keeps its outer surface and ribs the inside.
5. **Say how it will be made**: `"process": "fdm"` (or `{"type": "fdm",
   "buildDirection": "+z"}`), `"sls"`, `"mjf"`, `"sla"`, `"dlp"`, `"volumetric"`, or
   `"lpbf"` for powder-bed metal (AlSi10Mg, Ti64, 316L). The
   process sets a minimum member it can make, overhang control for FDM, and
   for layered processes checks the tension pulling layers apart against the
   layer bond (FDM: 60% of the in-plane strength); a safety-factor goal
   sizes the part against the bond too. Pick
   the build direction so the main loads run along the layers, or set
   `"buildDirection": "auto"` to let it compare all six (slower: every
   direction half-designed on a coarse grid, the best three finished; with
   no overhang control, one design checked per axis). `report.json` → `print` says whether it needs
   support, its bridges, and whether the first layer holds; `run` writes a
   `.3mf` per part already standing the right way up for the slicer.
   FDM outcomes are also sliced headless when PrusaSlicer is installed
   (`report.json` → `slicer`: `printTime` s, `filamentVolume` cm^3,
   `filamentMass` g, `supportVolume`); a warning says when no slicer was
   found. For an FDM user hours matter more than grams: `"goal":
   {"safetyFactor": 2, "minimize": "printTime"}` delivers the passing outcome
   that prints fastest. `"slicer": {"grade": true}` sizes infill per region
   from the stress (sparse where it idles, solid round bolts and hot spots,
   checked to the goal's safety factor) and `run` writes `<stem>-graded.3mf`
   with the regions as slicer modifiers; `graded` in the report gives its
   safety factor, its slice and the time and filament saved against the
   solid slice. The part is verified solid, so print the plain 3MF at 100%
   infill or the graded one, never the plain one at a default 15%.
   `slicer.profile` takes your printer's PrusaSlicer .ini for real times.
   Powder nylon (`"sls"`, `"mjf"`: `pa12`, `pa11`, `pa12-gb`, `tpu88a`) and
   resin (`"sla"`, `"dlp"`: `resin`, `resin-tough`, `resin-rigid`) parts get
   `report.json` → `process` on the delivered outcome: walls against the
   process minimum, powder or resin trapped in a cavity or behind an exit
   under the escape-hole size (SLS 4, MJF 5, resin 3.5 mm; override with
   `"escapeHole"`), thin flat panels that warp (powder), and for resin the
   supports, the largest layer (peel) and suction cups. Its warnings are the
   outcome's. Fix each before ordering the print.
6. **Say how long it has to last.** A load that repeats (a servo lifting,
   an arm swinging) gets `"cycles"` in its load case: how many times over
   the part's life (a hobby arm: 1e5-1e6). A load that is held (a part
   hanging under its own weight, a pose held all day) gets `"hours"`.
   `"durability": {"temperature": 40}` sets the service temperature
   (motors run warm; PETG creeps fast above ~55 C, PLA above ~45 C).
   The report's `durability` block gives fatigue life and Miner damage,
   creep rupture margin and crept deflection, and with `fasteners` how much
   clamp load is left after `preloadHours`. Printed plastics are poor in
   fatigue (PLA survives ~10% of its strength for 2e6 cycles), so a part
   with a static SF of 3 can fail at a million cycles. Add `"constrain":
   true` to size the part for durability (`durability.safetyFactor`,
   default 1.5). The data is published printed-polymer data where it
   exists and flagged `assumed` or `rough` where it does not: treat it as
   an estimate, and test the part that matters.
7. `python -m loadpath check study.json`: validates and reports cells, pitch
   and warnings. Fix every warning before running.
8. `python -m loadpath run study.json --out out/`: minutes, not seconds.
   Progress is JSON lines on stderr.
9. **Read `out/report.json` and look at the renders** before trusting it:
   - `safetyFactor` below target, or `passes: false` → raise the goal's
     volume fraction, the material, or check the loads are right;
   - `bodies` > 1 → something is not connected to a support;
   - `keepOutIntrusion` / `preserveMissing` above ~1 mm³ → report a bug;
   - `p5Member` under the printer's minimum → raise `minMember`;
   - `durability.safetyFactor` under its target → `durability.constrain`,
     a stronger material, or a build direction that keeps cyclic tension
     off the layer bond (`fatigueLimitedBy: "layer bond"`).
10. Once the report passes, offer to get it made: Materialize
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
  With `pip install loadpath[crosscheck]` the pick is confirmed on a
  body-fitted tetrahedral mesh (`goal.confirm`, default `auto`): the report's
  `safetyFactor` is then the tet reading (`confirm.method: "tet"`, voxel
  reading beside it), and the search steps down to where that meets the
  target, typically 0-45% lighter. A warning that the cross-check reads the
  pick below target names the spot the voxel grid cannot resolve: look there.
- `{"volumeFraction": 0.25, "outcomes": [0.15, 0.4]}`: fixed fractions, a
  trade-off gallery for a person to choose from.

## Lattices and heat exchangers

- A chunky part (a high volume fraction, a solid design) can be filled with a
  TPMS sheet lattice graded by its own stress: `"lattice": {"type":
  "gyroid", "cell": 10, "skin": 1.2}` in the study, or `loadpath_lattice`
  on a finished job. `skin: 0` for SLS, MJF and resin so they drain.
- Don't use a lattice to make a part stiffer per gram: solid topology
  optimization wins that. Use one to stop an impact (a bumper, a hard stop, a
  drop pad): give a field graph an `impact` block (envelope, density node,
  mass and velocity or energy, forceLimit, `"grade": true`, cell and minWall).
  It returns the peak force and crush, and a density table that crushes every
  layer at the same force. A `vibration` block gives natural frequencies with
  point masses. See docs/IMPACT.md in the repo.
- A two-fluid heat exchanger is its own document (`loadpath_schema
  "heatExchanger"`): a core shape, a layout (`cross` or `counter`), cell and
  wall, and an optional duty to rate it. Size it first
  (`loadpath_size_heat_exchanger`, seconds), then design it
  (`loadpath_heat_exchanger`): the report says whether each fluid reaches
  its ports and only its own. With a duty, `"simulate": true` also
  simulates the whole exchanger (porous core, open headers, both fluids and
  the wall; porous.py, about a minute): how evenly each fluid shares out
  over the core (`entry.uniformity`, 1 = even), core vs header pressure
  drop, outlet temperatures and the hottest wall point. Use it to compare
  header depths and port sizes (`porous.sweep`); the rating alone assumes
  even flow. What it found on the 30 mm crossflow cube: size ports per
  fluid (`"portDiameter": {"water": 10, "air": 18}`; a gas side's ports
  set most of its pressure drop), and a thinner seal band (`cap` 1.5 for a
  1 mm wall) opens the starved edge channels; plenum depth hardly matters.
  Together: water 745 -> 267 Pa, air 8.5 -> 1.9 kPa, effectiveness +8%
  (example heat-exchanger-crossflow-tuned).
- Holding a mesh from somewhere else (your own field model, a CAD export)?
  `loadpath_check_printability`, `loadpath_check_isolation` and
  `loadpath_check_fit` check it directly. From Python, `check_dfm` and
  `check_networks` also take the implicit field itself (`f(P) < 0` inside,
  with bounds), which skips meshing and is the fast path for a part built
  from a field.

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

## Length scales and sealed voids

`minMember` (mm) is the thinnest member, `minHole` the narrowest hole or
gap. A `process` sets both if the study does not (2 x its minFeature for
members: SLS/MJF/FDM 1.6; holes FDM/SLS/MJF 2, SLA/DLP/LPBF 1). With
neither, the filter still holds members to 1.4 mm, so a fine grid cannot
grow sub-millimetre skins. A hole narrower than `minHole` comes out filled.

Stiffness likes closed sections, so a fine grid can seal a void (a skin
over a cable channel). For powder and resin processes and LPBF
(`"drains": "auto"`, the default) the optimizer then drills escape holes
(SLS 6 mm, MJF 7.5, SLA/DLP 5.5, LPBF 6.8; at least 2 per void, more until
no powder is over 60 mm from one) through thin, low-stress wall, and runs on
so the part carries its load round them. `"drains": true` forces it for
any process, `false` turns it off, or give `{"diameter", "count",
"maxPath", "iterations"}`. `report.json`'s `optimizer.escape` lists the
holes; `enclosedCavityVolume` should read 0.

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

For a link in a mechanism with a URDF, don't write those loads by hand:
derive them, and the neighbours' swept space, from the arm itself.

```bash
loadpath assembly robot.urdf --link forearm --study forearm.json --region j4-mount \
    --payload 1.0 --payload-at 0,0,0.07 --payload-radius 0.03 --out derived/
loadpath run derived/forearm.derived.json
```

It sweeps poses over the joint limits, speeds and accelerations (scaled to
what each joint's rated effort can deliver), a scattered payload, a push at
the tool and each joint upstream stalled against an obstacle, does
Newton-Euler for the wrench the part carries at its far joint, and keeps the
few samples that set the peak stress everywhere in the solid design space
(`assembly.json` says how close they come: `envelopeRatio`). The parent and
child links, swept through their joints' ranges, become `bore: false` mesh
keep-outs with the study's preserved interfaces cut out. Inputs worth
setting: `--ramp` (seconds to full speed, what the controller allows),
`--push` (N, the hardest contact at the tool; a stalled joint pushes up to it) and `--balanced`
for joints a spring counterbalances. Python: `loadpath.assembly`
(`Robot.from_urdf`, `sweep_loads`, `governing_cases`, `swept_keepouts`,
`derive_study`).

The full field reference is in README.md next to this file.
