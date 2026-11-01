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
   empty). If the part already exists as CAD, start from it:
   `loadpath import part.step --out draft/` (MCP `loadpath_import_cad`) finds the
   bores, bolt patterns and mounting faces and writes a draft study with them
   preserved and kept out; pass `--material`, `--support` and `--load ID=FX,FY,FZ`
   (interface ids from `--features-only`) so it carries the real setup. Primitive shapes cover most of it; for anything else export a
   closed STL and use `{"type": "mesh", "path": "motor.stl"}`. An interface
   in several pieces (a servo's four screw bosses) is one region with
   `{"type": "union", "shapes": [...]}`, so one load or support acts on all of
   them. `intersect` and `subtract` (same `shapes` list; subtract cuts the
   rest from the first) make partial shapes: a half ring is a tube
   intersected with a box, a ring with a gap a tube minus a box. Preserve only the interface itself, never a box round a component:
   a preserved slab is a slab in the result, and the optimizer has nothing
   to design there.
   A bearing seat narrower than the part: the seat (its OD, its width) is a
   keep-out and the region the load or support acts on, a narrower keep-out
   runs through for the shaft or inner race, and the preserved ring covers
   the seat's width. Keep-outs win over preserves: `check` warns when one
   eats into a preserved piece (a fastener's driver path clipping a boss).
   Anything inserted (a shaft, pin, bearing, cable) needs its keep-out to
   run past the design space on the side it goes in from, or the part may
   close over it; `check` warns about a bore with no open end.
   Bolted joints go in `fasteners` (size, seat point, direction, grip, and
   whether the far end clamps on a face or engages a thread or insert): it
   preserves only the bearing rings or the thread sleeve, keeps the bore and
   the head and driver path clear, and adds a preload case. Heads default
   to ISO 4762 socket heads (M3 5.5, M4 7, M5 8.5 mm); give `head` for a
   washer or pan head. The head and driver keep-out is as wide as the seat
   ring (head + 2 x wall: M3 7.5, M4 9 mm); `counterbore` sets its diameter
   when the user gives a counterbore or key clearance. Don't
   draw a solid boss round a screw.
2. **Bound the design space** generously: the envelope the part may occupy,
   minus nothing (keep-outs are subtracted for you). Too tight a box is the
   most common reason a result looks like the box.
3. **Write the loads as engineering, not vibes.** For each way the part is
   used, a load case: payload mass × 9.81 × a dynamic factor (2 for a
   moving robot joint, 3 for anything that can be dropped or crashed), at the
   interface that carries it. Let loadpath do that arithmetic:
   `{"region": "wrist", "mass": 0.6, "dynamic": 2, "gravity": "-y", "at": [160, 0, 0]}`
   is the force of 0.6 kg at 2 g along -y plus, because its centre sits at
   `at` (mm), the moment of that force about the region. `at` works on a
   plain `force` too (a belt pulling on a pulley in front of a motor face);
   `check` → `resolvedLoads` shows the force and moment each case comes to.
   Moments written by hand are N·mm (1 N·m = 1000), about the region's centre. A shaft in a
   bore: add `"distribution": "bearing"` so a radial force presses on the
   half of the bore it pushes, as a real shaft does. Add a sideways or torsional case if the part
   can see one; a design optimized for one direction is weak in the others.
   A link that carries or is driven by a servo or motor: add a case with
   its full stall torque about the joint axis at that interface (a 1.5 N·m
   servo is `"moment": [0, 1500, 0]` about y), since a jam or a hard stop
   puts all of it through the part; it often sets the size.
   When the direction is genuinely unknown (contact, bumps, handling, a
   push at the tool "from anywhere"), say so instead of guessing:
   `{"region": "tool", "force": [0, 0, -20], "direction": "any"}` designs
   and checks for the force's size in its worst direction, and `"cone": 30`
   for anything within 30 degrees of the written one. The worst case is
   exact (three solves, a 3x3 eigenproblem), and `report.json` gives each
   case's `worstDirection`. It costs stiffness in the nominal direction (the
   arm link: about half again more compliant under its payload), so use it
   for the uncertain load and keep known loads as plain vectors. A cone is
   3D: it tilts the force out of plane too, which a thin plate-like part (or
   one that would swing on a pin rather than resist sideways) cannot carry.
   For an in-plane uncertainty (a shaft's radial load, a swinging strap, a
   sideways crash) use `"direction": "xz"` (or `"xy"`, `"yz"`): the case
   becomes eight cases round half that plane, named `case@0` ... `case@157.5`,
   or sixteen round all of it (to `case@337.5`) when -F is not the mirror of
   F: a bearing distribution, other loads in the case, a moment, or a
   layered process. Each is a solve per step, so the run takes that many
   times longer; the mass form cannot take a plane, so write the force
   (kg x 9.81 x the dynamic factor).
4. **Hold it where it is really held.** A bolted face: support on the bolt
   holes (keep-outs), plus `"fix": "x"` on the face it bears against if it
   only pushes. A shaft in a bearing: support on the bore.
   A part that hangs on a rod, hook or peg bears only where it touches:
   support the strip of the hole that carries the weight (`fix` on a
   narrow box over its top), not the whole hole, or the model hides the
   bending at the hook's mouth.
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
   support (a warning says so when it does), its bridges, and whether the
   first layer holds; spans past `process.bridgeSpan` (mm; default 15 FDM)
   count as needing support, so set it for your printer. `run` writes a
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
   `report.json` → `deliver` → `process`: walls against the
   process minimum, powder or resin trapped in a cavity or behind an exit
   under the escape-hole size (SLS 4, MJF 5, resin 3.5 mm; override with
   `"escapeHole"`), thin flat panels that warp (powder), and for resin the
   supports, the largest layer (peel) and suction cups. Its warnings are the
   outcome's. Fix each before ordering the print.
   `"buildSim": true` (lpbf, fdm, sls, mjf) simulates the build of the
   delivered part: `report.json` → `buildSim` gives `distortion` (mm, off the
   plate or bed), `residualStress` (MPa vs yield), `recoater.maxLift` (LPBF:
   above ~0.05 mm the recoater can hit it) and `peel` (FDM: mean pull on the
   bed, MPa). Add `"compensate": true` for `<stem>-compensated.stl`,
   pre-distorted to print to shape. Read `strain.kSource`: only IN625 and
   15-5 PH are calibrated (NIST AM-Bench); for other alloys the size of the
   distortion is a rough guide (could be off 2x), though comparing designs
   is still sound. `"warpWeight": 1` (lpbf/fdm, grid-axis buildDirection)
   designs against build distortion too, but at a fixed mass it halved the
   safety factor in the demo. Prefer a limit with the safety-factor goal:
   `"maxDistortion": 0.2` (mm, as built on the plate) makes the search
   deliver the lightest part meeting both (report `warp`; on the demo
   bracket a 40% cut cost no measurable mass, 65% was infeasible at SF 2).
   `maxRecoaterLift` exists but the optimizer's lumped layers misjudge
   lift: check `buildSim.recoater` on the result. For FDM (PETG, nylon, PP), warp is
   extrapolated to the printed layer (`printLayer`, default 0.2 mm), and
   the limit is checked on each outcome's delivered part
   (`report.warp.delivered`), because the optimizer's grid still reads it
   2-3.5x low. On a bed-printed beam the lift barely moves with the shape,
   so a tight FDM limit finds no part. Lower the warp with the process
   instead: bed and chamber temperature, `bedAdhesion`, the orientation, or
   PETG over nylon. Nylon and PP shrink from their crystallization
   temperature and warp far more than PETG. FDM: `bedAdhesion` (MPa) lets
   the first layer let go; `bedFailed` in the report means the part
   popped off. docs/BUILDSIM.md has the validation.
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
   and warnings. Fix every warning before running; `notes` are information
   (for example that `minMember` set the pitch, so there are fewer cells
   than `maxElements`).
8. `python -m loadpath run study.json --out out/`: run it in the background
   (`nohup ... &`, or the MCP `loadpath_start` job) and poll, never in a
   foreground call with a time limit. `out/status.json` says `running`
   (with its `pid`), `done` or `failed`; each outcome is written to
   `out/partial/` (STL plus a line in `outcomes.jsonl`) as the search finds
   it, so a run killed late still leaves its verified parts.
   Progress is JSON lines on stderr. A `safetyFactor` goal runs several
   optimizations plus a fine re-check of the pick: at 20k cells on one core
   expect 10-30 minutes (a fixed `volumeFraction` is a few minutes).
9. **Read `out/report.json` → `deliver`, and look at the render** before
   trusting it. `deliver` is the part to hand over: `stl`, `passes`, `mass`,
   `safetyFactor` and `safetyFactorMethod` (which analysis it comes from:
   `tet`, `voxel-fine`, or `voxel`), `governedBy` (the limit that set its
   size), per-case numbers, and `warnings` gathered from every check (print,
   process, durability). Quote those numbers. To check any STL against a
   study later (an edited part, an older outcome):
   `loadpath verify study.json part.stl` (its own grid, as the run's
   outcome check; `--cells N` for a grid of about N cells, finer or not, so it
   reads a few percent off the run's fine re-check; quote `deliver` for a
   delivered part). `deliver.tetCrossCheck` says why the body-fitted check
   did not run when it did not. Then:
   - `safetyFactor` below target, or `passes: false` → raise the goal's
     volume fraction, the material, or check the loads are right;
   - `bodies` > 1 → something is not connected to a support;
   - a "mesh lost N% of the optimized solid cells" warning (`meshFidelity`)
     → the mesh came out thinner than the optimized design, so the part
     reads softer than it should and the search picks a heavier outcome.
     The verified numbers hold; if the part is too heavy, a hand-built
     field graph of the same layout (`loadpath field`) can do better;
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
  It is isotropic, so it never runs for a `process` (FDM layers, SLS, resin)
  or gravity or pressure loads: those are confirmed on a finer voxel grid,
  `deliver.tetCrossCheck` says why. That fine re-check is the memory peak
  `check` reports (`peakMemoryGB`); `LOADPATH_CONFIRM_CELLS=60000` lowers it.
- `{"volumeFraction": 0.25, "outcomes": [0.15, 0.4]}`: fixed fractions, a
  trade-off gallery for a person to choose from.

## Heat, fatigue and creep

- A part next to something hot (a stepper face at 70 C, a heated bed):
  a `thermal` block. `temperatures` hold surfaces at a temperature
  (`region` or inline `shape`, `t` in C), `heat` puts watts in,
  `convection` takes it off (default 8 W/(m² K), still air), `ambient`
  25 by default. The temperature is solved on the part; every load case
  then softens and derates each cell for its temperature and carries the
  restrained expansion. `thermal.cycle` (`period`, `on` s, `cycles`) adds
  heat-up/cool-down fatigue. See examples/thermal/motor-mount-hot.json.
- Repeated loads: `cycles` on the load case (not on a load). A held load:
  `hours` on the load case (creep rupture and crept deflection). A load of
  unknown in-plane direction with `cycles` counts its cycles once, at the
  worst direction.
- `durability`: `temperature` (C the cyclic and held cases run at; it
  derates only the fatigue and creep checks, so a part in a hot enclosure
  also needs a `thermal` block, e.g. `temperatures` at that value, for its
  static safety factor), `safetyFactor` (default 1.5), and
  `constrain: true` to size the part for it rather than only check it.
  With `fasteners`, the preload is also checked held for `preloadHours`
  (default a year).
- The result is `durability` in each outcome (`fatigueSafetyFactor`,
  `creepSafetyFactor`, per-case `life`), and `deliver.durabilitySafetyFactor`.

## Lattices and heat exchangers

- A chunky part (a high volume fraction, a solid design) can be filled with a
  TPMS sheet lattice graded by its own stress: `"lattice": {"type":
  "gyroid", "cell": 10, "skin": 1.2}` in the study, or `loadpath_lattice`
  on a finished job. `skin: 0` for SLS, MJF and resin so they drain.
- Field graphs take strut lattices too (`"type": "octet"`, `kelvin`, `bcc`,
  `fcc`, `voronoi`, ...; `thickness` is the strut diameter, `blend` fillets
  the nodes) and conformal cells (`"frame": {"type": "conformal",
  "surface": "envelope"}`: layers parallel to the skin, no cell cut by it).
  For strength per gram a gyroid sheet beats every strut cell (arm link at
  equal mass: SF 16 vs 9-13); pick struts for open, drainable cells, very
  low density, or FDM (`"aspect": [1, 1, 1.5]` BCC prints unsupported; the
  field report's `lattices` block gives the bridging strut length).
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
  Only an envelope, not a core? Give `"envelope": [x, y, z]` (the whole
  part, ports included) and `"sizing": {"process": "lpbf", "maxPressureDrop":
  {...}, "minEffectiveness": 0.7, "objective": "mass"}` instead of `core`,
  and `loadpath_size_heat_exchanger` picks the core's size, cell, wall,
  split, layout, plenum depth and each fluid's port within the process's
  wall and channel limits (sizing.py), returning ranked specs to design.
  For gas sides at high flow the plenum depth matters (a 6 mm plenum behind
  a 47 mm charge-air port cost more than the core). `sizing.verify` then
  simulates a candidate whole; big cross cores skip the design and use
  `porous.simulate_box` (minutes; its header dp is an upper bound).
  On the command line: `loadpath check exchanger.json` lints the spec and
  prints the design grid (`gridStep`, `gridPoints`) and `peakMemoryGB`
  (about 400 bytes a grid point) against what is free, in milliseconds.
  `loadpath heatx spec.json --size --out DIR` writes `sizing.json` (the
  ranked result) and `best.json` (the top spec) to DIR; `--objective`
  is `mass`, `effectiveness` or `pressure` (as `sizing.objective`; the
  flag wins), `--min-wall` and `--splits` narrow the sweep, `--axes xy|yx`
  (or the spec's `axes`) pins which way the first fluid runs. Sized cores
  sit centred in the envelope's frame ([0, 0, 0] to the envelope), so the
  ports land inside it. The sizer returns only designs the design step can
  mesh within its grid cap (16M points, a grid of at most wall / 2; thin
  LPBF walls on a large core are dropped and `note` says so): raise it with
  `--max-points N` for both steps if the machine has the memory. Its
  `solidVolume` and `coreMass` are the core's sheet only; the design
  report's `mass` (g, from `duty.wall`) is the whole part. `loadpath heatx`
  refuses a design whose estimated peak is over the free memory (`--force`
  runs it anyway) and keeps `status.json` in the out folder (`running` with
  its stage, `done`, or `failed` with the error; a run killed for memory
  stays `running` with its pid). `report.warnings` names an outer wall
  under the process's minimum, and for lpbf over 300 mm^2 of flat internal
  ceiling or over 20 mm^3 of trapped powder. The rating's `duty` is the heat
  moved (W, positive whichever fluid is listed first; `dutySign` -1 when
  the first fluid takes it, `hotFluid` names the hot one).
- Holding a mesh from somewhere else (your own field model, a CAD export)?
  `loadpath_check_printability`, `loadpath_check_isolation` and
  `loadpath_check_fit` check it directly. From Python, `check_dfm` and
  `check_networks` also take the implicit field itself (`f(P) < 0` inside,
  with bounds), which skips meshing and is the fast path for a part built
  from a field.

## Multi-material parts

For a part printed in two to four materials at once (Bambu AMS, Prusa
MMU/XL, IDEX, PolyJet): a rigid core with a TPU grip pad, a bumper, an
isolator. Add a `multiMaterial` block; the optimizer places each material
as well as the topology (`loadpath_schema` has every field):

```json
"multiMaterial": {
  "materials": ["pla", "tpu"],
  "assign": [{"region": "pad", "material": "tpu"},
             {"region": "pivot-boss", "material": "pla"}],
  "interfaceSafetyFactor": 2.0,
  "export": {"slicer": "bambu", "extruders": {"pla": 1, "tpu": 2}}
}
```

- `materials`: library names or custom objects; each keeps its own
  stiffness, Poisson's ratio and strength. `assign` fixes cells to one
  material: `region` (a preserve or keepOut id), a `shape`, or `near`
  (`{"region" or "shape", "distance"}` mm from it, optionally `within` a
  region); earlier entries win. Assign the bosses and bearing seats to the
  rigid material and the pad to the soft one, or the optimizer may put
  rubber round a pin. `maxFraction` / `minFraction` cap or floor a
  material's share of the part.
- `requirements` make it soft where it has to be: `knock` (`energy` J,
  `maxForce` N: a bumper that stops a knock under that peak force) or
  `indentation` (`relativeTo` a case with the same force spread out,
  `atLeast` mm: a pad that gives). Those cases leave the stiffness
  objective, so at least one other case must remain.
- The bond: every interface face of the delivered bodies is checked against
  the pair's strength (`interfaces: {"pla+tpu": {"tensile", "shear"}}`
  overrides the sourced card; measure your printer's) at
  `interfaceSafetyFactor` (default the goal's safetyFactor, else 2).
  `interlock: true` interlaces the bodies in a beam lattice at the
  interface and credits the bond with its gain. Stress beside supports and
  load nodes is left out of the check; a body thinner than that band (a
  2 mm pad loaded on its face) is checked closer in, and the report's
  `warnings` say so. A material or bond no cell was read for is listed in
  `unchecked`, and the design does not pass.
- A safety-factor goal runs a short search over volume fractions; it starts
  above the share of the design space that `preserve` fills (a warning says
  when it moved the start).
- Delivers, in `--out`: an STL per material (`<name>-<material>.stl`, the
  study's coordinates), `<name>.3mf` (one object, one part per material
  with its extruder; `export.slicer` picks the PrusaSlicer or Bambu/Orca
  layout), `optimum.npz` (the design fields) and `report.json`: `mass`,
  `materialMass`, `safetyFactor` (worst material), `interfaceSafetyFactor`
  (worst bond), per case `materials` and `interfaces`, `requirementsMet`,
  `passes`, `outcomes` (each fraction tried) and `print` (supports,
  bridges, islands, when the study has a build direction). `report["3mf"]`
  says where the 3MF puts the part: `"bed"` (default: build direction to
  +z, on the bed at 128,128; `transform` maps study coordinates to the
  file's) or `"study"` (`loadpath run --3mf-coordinates study`: identity,
  as the STLs). `status.json`, `partial/` and the progress lines (an
  `outcome` per fraction verified, with mass and safety factors per
  material and bond) work as for one material.
- Verify an edited multi-material part from Python (`loadpath verify` is
  single-material). The bodies are in the study's coordinates and must not
  overlap; list the stiff material first (a cell both claim goes to the
  first):

  ```python
  import json, trimesh
  from loadpath.spec import study_from_dict
  from loadpath.multimat import verify_multimaterial
  st = study_from_dict(json.load(open("study.json")), base_dir=".")
  mm = st.multi_material
  bodies = [(mm.index("pla"), trimesh.load("out/finger-pla.stl")),
            (mm.index("tpu"), trimesh.load("out/finger-tpu.stl"))]
  rep = verify_multimaterial(st, mm, bodies)
  rep["safetyFactor"], rep["interfaceSafetyFactor"], rep.get("unchecked"), rep.get("warnings")
  ```

  Only what is connected to the supports is analysed.

## Printing direction

A `process` of `fdm` or `lpbf` already designs self-supporting along its
`buildDirection` (the optimizer only builds what prints at 45°); setting
`printDirection` as well changes nothing. Without a process, set
`"printDirection": "+z"` (the direction layers stack, in the study's
coordinates). Preserved geometry is kept as drawn, so a preserved overhang
(a platform, a ring's underside) still needs support: chamfer it in the
preserve or accept the supports `deliver` → `print` reports.

## Resolution

`resolution.maxElements` trades detail for time: 20k for a first look,
40-60k for a part you will print (each optimization scales a little worse
than linearly with cells; a safety-factor search runs several). Members thinner than
about two cells cannot form; `check` warns when `minMember` is too fine for
the grid. `maxElements` is a target the pitch is estimated from, so the grid
can come out a few percent over; set `resolution.pitch` for an exact grid.
A safety-factor search re-checks its pick on a finer grid (up to 150k
cells, about 55 kB a cell): that stage is capped to the memory free at the
time, and `LOADPATH_CONFIRM_CELLS` sets it outright on a shared machine.

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
the tool, each joint upstream stalled against an obstacle, and the part's
far joint jammed at its full rated torque (a hard stop or a blocked
hand: a servo-driven link must survive its own servo's stall torque), does
Newton-Euler for the wrench the part carries at its far joint, and keeps the
few samples that set the peak stress everywhere in the solid design space
(`assembly.json` says how close they come: `envelopeRatio`). The parent and
child links, swept through their joints' ranges, become `bore: false` mesh
keep-outs with the study's preserved interfaces cut out. Inputs worth
setting: `--ramp` (seconds to full speed, what the controller allows),
`--push` (N, the hardest contact at the tool in any direction, default 20: often the case that sizes the part, so set it from the job, 0 if nothing pushes) and `--balanced`
for joints a spring counterbalances. Write the study in the URDF's world
frame at the home pose (mm), or pass `--frame link` if it is in the link's own
frame (origin at its joint): a mismatch silently gives wrong moment arms. The
study needs no load cases (they are replaced; `fasteners` and their preload are
kept), though `loadpath check` on it alone wants one: check the derived study.
`sweptKeepOuts` is empty when the URDF has no collision or visual geometry.
The derived loads are the sweep's own accelerations: no extra dynamic factor. Python: `loadpath.assembly`
(`Robot.from_urdf`, `sweep_loads`, `governing_cases`, `swept_keepouts`,
`derive_study`).

The full field reference is in README.md next to this file.
