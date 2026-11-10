---
name: loadpath
description: Design a load-bearing part with topology optimization (generative design) from a plain request ("a bracket for a 10 kg shelf", "a lighter version of this STEP", "the forearm for this URDF"). You work out the interfaces to keep (bores, mounting faces, bosses), the volumes to leave empty (bolt access, cables, mating parts) and the loads, then get the lightest printable shape that holds a stated safety factor. Produces watertight STLs plus a verified stress/mass report.
---

# loadpath: generative design for printed parts

The user says what they want to build; you work out the problem and the
solver finds the shape. Do not model the organic geometry yourself. Your job
is to get the **problem** right: what must be kept, what must stay empty,
where the part is held, and what pushes on it. Everything is millimetres,
newtons, MPa.

Don't hand the user a questionnaire. Most of the problem is in what they
already gave you or in the world:

- **Interfaces** come from their CAD (`loadpath_import_cad` finds bores, bolt
  patterns and mounting faces), from the parts it meets (a servo's or
  bearing's datasheet dimensions, a standard bolt), or from the description.
- **Loads** come from what the part does: the mass it holds times gravity
  and a shock factor, a motor's stall torque, a URDF's masses and joint
  limits (`loadpath assembly`), a hand push, a drop.
- **Clearances** come from the fasteners (`fasteners` sizes the clearance
  hole, head and driver path) and the mating parts and sweeps (keep-outs).
- **Fits and tolerances** come from what goes in each hole: a bearing
  presses in, a locating pin slips, a shaft turns, a bolt passes. Draw the
  keep-out at the mating part's diameter and give it a `fit`; loadpath
  redraws the hole for the process (FDM holes print a few tenths small,
  powder closes SLS and MJF holes) and the report says what each hole will
  measure as built and whether the fit holds, needs a tuned printer, or
  needs a reamer.
- **Material, process and safety factor** come from how they'll make it and
  what failure would cost.

Pick a defensible reading, say each assumption in one line in your reply
(screw size, where the load sits, the safety factor and why), and ask only
for what you can't infer and would change the part. Then let `check` lint
the setup and the report confirm every interface survived and every keep-out
stayed clear.

With the MCP server (`loadpath mcp`, see docs/agents.md) the same workflow is
the tools `loadpath_check`, `loadpath_start` and `loadpath_status`; without
it, use the CLI below. Start from the nearest example (`loadpath_examples` /
`loadpath examples`) rather than a blank page; `loadpath_schema` gives every
field. `check` reports every problem at once: fix them all, then check again.

## When to use it

Use loadpath when the shape should come from the loads: brackets, links,
mounts and housings around fixed interfaces, lattice-filled or graded parts,
compliant mechanisms, heat exchangers and cold plates. Draw parts whose shape
is set by function rather than load (enclosures, cosmetic covers, gears,
threads) in CAD instead. loadpath doesn't replace CAD: model the interfaces
there, let loadpath grow the material between them, and bring the result
back as STEP or STL (`loadpath step part.stl` turns any closed STL, a
gyroid-cored one too, into one STEP solid; `loadpath sdf part.stl --out part.vdb` writes an OpenVDB level set, and .vdb grids come in as sdf shapes or field `grid` inputs; `loadpath slices part.stl` (`--format cli-binary` or `slc`) into
powder-bed layers, a CLI file of contours and hatches; `loadpath nest a.stl:6 b.stl:2
--process sls --material pa12` packs a job into one build volume; `loadpath fe-export study.json part.stl --out part.bdf`
hands the part to Nastran or Abaqus with its loads and supports; `loadpath external-flow part.stl --speed 15
--direction -x --study study.json` puts it in a wind or water tunnel and adds the surface forces as a load case; a field
graph's `ribs` node stiffens a face with ribs along curves, a grid or the face's stress lines). Every report
also says what each support carries (`report.reactions`), the bolt or bearing
load to size hardware from; cap one with `maxForce` on the support. Its edge over drawing the part yourself is that every
result comes with an independent check (safety factor, deflection,
buckling, printability) you can report to the user.

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
   Give every hole something goes into a `fit` on its keep-out, drawn at
   the mating part's diameter: `press` (a bearing's outer ring, a dowel
   pin), `transition`, `slip` (a locating pin, a shaft you take out),
   `clearance` (a shaft that turns, a hinge pin). Fastener holes get ISO 273
   `normal` by default (`fit: "close"` or `"loose"` on the fastener). With a
   `process` (or `milling`/`casting`) every other bore is held at its drawn
   size. Plastic fits are wider than ISO; metal ones are ISO 286 (H7/p6,
   H7/k6, H7/g6, H8/f7). Read `deliver.fits` and pass its `instruction`
   lines on: "ream to 21.82-21.94 mm" or "print the kit's hole ladder and set
   process.holeBias" is something the user does after printing. A measured
   `process.holeBias` (as-built minus drawn diameter; `holeBiasAcross` for
   holes across the layers) replaces the default. To measure it, the
   calibration kit's H-XY / H-Z coupons are a numbered hole ladder read
   go/no-go with a drill bit (no calipers); a calibrated card carries it.
   Bolted joints go in `fasteners` (size, seat point, direction, grip, and
   whether the far end clamps on a face or engages a thread or insert): it
   preserves only the bearing rings or the thread sleeve, keeps the bore and
   the head and driver path clear, and adds a preload case. Heads default
   to ISO 4762 socket heads (M3 5.5, M4 7, M5 8.5 mm); give `head` for a
   washer or pan head. The head and driver keep-out is as wide as the seat
   ring (head + 2 x wall: M3 7.5, M4 9 mm); `counterbore` sets its diameter
   when the user gives a counterbore or key clearance. Which part holds the
   thread decides `end`: a screw through this part into a servo horn, a
   motor face, a nut or any mating part's thread is `end: "face"` (this part
   gets a clearance hole); `end: "thread"` or `"insert"` only when this part
   is tapped or takes a heat-set insert. Servo horns and motor faces are
   tapped, so the part bolted to them gets clearance holes. Don't
   add a solid boss round a screw on your own (the fastener keeps what the
   joint needs), but a boss the user describes ("a 12 mm boss, 5 mm thick")
   is an interface: preserve it as given, alongside the fastener. A bolted foot: hold the bores sideways
   and the face's normal only on a washer-sized disc under each bolt; a
   slab over the whole face glues the part down, and a load that lifts or
   tips it then reads far too strong (a stepper bracket: SF 3.5 glued, 0.98
   bolted).
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
   A rare event the part may use most of its strength for (a crash, a
   drop, a hard landing) beside service loads: give that case its own
   `"safetyFactor": 1.25`; the search counts its reading times goal / 1.25,
   so service cases keep the goal's factor and the crash its own.
   A link that carries or is driven by a servo or motor: add a case with
   its full stall torque about the joint axis at that interface (a 1.5 N·m
   servo is `"moment": [0, 1500, 0]` about y), since a jam or a hard stop
   puts all of it through the part; it often sets the size. Where the
   servo itself holds the part (the link is supported on the horn), the
   jam reacts at the far end instead: a force there of torque / arm (2.5 N·m
   at 100 mm is 25 N, perpendicular to the link in the joint's plane). A
   screw through this part into a horn's or mating part's thread is
   `headOn: "this"` (default) with `end: "face"`.
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
   A part nothing holds (a drone arm between motors, a part that is
   dropped or thrown): `"inertiaRelief": true` and no supports; its own
   inertia balances the loads. When it carries more than it weighs (a
   drone frame's motors and battery), say so:
   `"inertiaRelief": {"masses": [{"region": "battery", "mass": 0.1}]}`, or a
   thrust case pulls against the frame's own corners instead of the battery
   (a dumbbell bar read half its bending moment without its end masses). A
   `vibration` block on such a part reads its free-free modes, carried
   masses included. A part that carries a motor or a camera
   and must not ring: `"vibration": {"minFrequency": 60, "masses":
   [{"region": "motor-seat", "mass": 0.12}]}` (Hz, kg) holds its first
   natural frequency with the safety factor. When something forces it
   over a range of speeds (a rotor's once-per-rev and blade-pass lines
   from idle to full throttle), give the range instead of a floor:
   `"vibration": {"excitations": [{"name": "blade pass", "band": [536,
   1490]}]}` keeps every mode out of the band (add `"maxAmplification": 3`
   to keep them far enough away that none magnifies the forcing more than
   that), and the report lists the modes in each band. With a `thermal`
   block the modes are those of the warm part (PA12 under an 80 C motor
   keeps 45% of its stiffness); a free part (`inertiaRelief`) takes a
   `thermal` block too. Structure under a propeller or in front of a fan blocks thrust:
   `"aero": {"rotors": [{"center": [52, 52, 12], "axis": "-z", "radius":
   44.5, "hub": 14, "thrust": 0.78, "ducted": true}]}` (thrust at hover, N)
   charges the optimizer for the drag of what each stream meets, as the
   mass a hovering craft would carry instead, and the report gives
   `aero` and `effectiveMass`. A link that turns:
   `"massProperties": {"axis": {"through": [0, 0, 0], "direction": "z"}}`
   keeps its inertia about the joint down, and every report's
   `massProperties` gives mass, centre of mass and inertia.
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
   **When the part will be printed, take its design limits from
   Materialize, not from memory or loadpath's defaults.** With the
   Materialize MCP server connected, `materialize_list_materials` (filter by
   name, like "TPU" or "PA12") then `materialize_get_material` give each
   material's `designLimits` per process (`minWallMm`, `minDetailMm`) and
   its build volume. Set `process.minFeature`, `minMember` and any `skin`
   thickness to at least those, and use the larger of Materialize's figure
   and the user's own rule or loadpath's material card (loadpath keeps SLS
   TPU at 1.0 mm as a margin over Materialize's 0.8). Say which you used.
   Without the server, `loadpath.dfm_reference` holds a dated snapshot of
   the same figures, and every process default already sits at or above it
   (minimum wall: FDM 1.0, SLS/MJF 0.8, SLA/DLP 0.6, PolyJet 1.0, LPBF metal
   1.0 mm). A run's `report.json` → `materialize` carries the limits, the
   Materialize material `id` to quote it as (and to publish it with, as
   `recommendedMaterialId`), `fitsBuildVolume`, and a warning when the
   design allows walls under the minimum. `loadpath materials <name>` shows
   the same per card. Materialize publishes no escape-hole, hole, bridge or
   clearance sizes; those stay loadpath's (`processcheck.RULES`).
   After meshing, `materialize_check_printability` on the STL is the
   printer's own check; fix its blockers before quoting.
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
   PETG over nylon. `"printSetup": true` does that search for you:
   orientation, material (`"materials"`), bed temperature and brim, ranked on
   strength, warp and whether the print stands up (report `printSetup`).
   When nothing passes as set up, it says which setup does, or
   re-optimizes for the best one. Nylon and PP shrink from their crystallization
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
8. `python -m loadpath run study.json --out out/`: run it in the background,
   detached from your shell (`setsid nohup loadpath run study.json --out out
   > run.log 2>&1 < /dev/null &`; a plain `nohup ... &` dies with the tool
   call that started it), or the MCP `loadpath_start` job, and wait with
   `loadpath status out --wait 540` (it blocks until the run ends or 540 s
   pass, so it fits under a 10-minute tool timeout; exit code 0 done, 1
   failed, 3 still running: call it again) or `loadpath_status(job_id,
   wait=540)`. Never run it in a foreground call with a time limit, and do
   not end your turn while it runs unless something will wake you: a
   one-shot agent that stops to "check back later" never comes back. A
   safety-factor search takes 10 minutes to a few hours depending on how it
   goes; if you have a deadline, say so: `loadpath run ... --time-limit 45`
   (or `loadpath_start(..., time_limit=45)`) stops the search then and
   delivers the best outcome verified so far, and while it runs `status`
   shows `bestSoFar` (the lightest passing part yet, its STL in
   `out/partial/`). It says `running` (with its `pid` and elapsed time), `done`, `failed`,
   `killed` (a signal), or `died` (the process is gone without a word: out of
   memory, or killed outright); each outcome is written to
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
   - a "meshing cut N mm^3 loose from material the optimized field had
     joined to the part" warning (`reconstruction.droppedFragments`, each
     with its volume, bounding box and `connectedInField`) → a meshing
     artifact, not the design: report a bug with the study and the
     outcome's `<name>-NN.field.npz` (the optimized field `run` saves
     beside each STL); `loadpath remesh study.json out/<name>-NN.field.npz`
     meshes and verifies that field again without optimizing;
   - `keepOutIntrusion` / `preserveMissing` above ~1 mm³ → report a bug;
   - `p5Member` under the printer's minimum → raise `minMember`;
   - `durability.safetyFactor` under its target → `durability.constrain`,
     a stronger material, or a build direction that keeps cyclic tension
     off the layer bond (`fatigueLimitedBy: "layer bond"`).
   When you report a passing part, say what the numbers are: a linear
   simulation with datasheet material values, not a tested strength
   (docs/VALIDATION.md). For anything safety-critical, the printed part
   needs a load test.
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
  `maxDeflection` is the mean travel of the loaded nodes: a load spread over
  a stiff rim reads low while the span between sags (a fan plate read 0.8 mm
  at its loads while its deck's middle dropped 3.9). To cap travel anywhere
  else, add `"clearances": [{"region": "deck", "maxTravel": 2, "direction":
  "-z"}]`: no node of that region moves more than 2 mm (down, with a
  direction; any way without); `cases` limits it to named load cases. Each
  case reports `clearances` with the travel, the limit and where. A
  `thinPreserves` warning means a preserved wall or deck is under 3 analysis
  cells thick: its bending reads stiff (up to ~2x), so check it on tets.
  If no fraction passes the finer-grid confirm, nothing is delivered as
  passing and the run's warnings say so.
  With `pip install loadpath[crosscheck]` the pick is confirmed on a
  body-fitted tetrahedral mesh (`goal.confirm`, default `auto`): the report's
  `safetyFactor` is then the tet reading (`confirm.method: "tet"`, voxel
  reading beside it), and the search steps down to where that meets the
  target, typically 0-45% lighter. A warning that the cross-check reads the
  pick below target names the spot the voxel grid cannot resolve: look there.
  It is isotropic, so it never runs for a `process` (FDM layers, SLS, resin)
  or gravity or pressure loads: those are confirmed on a finer voxel grid,
  `deliver.tetCrossCheck` says why. That fine re-check is the memory peak
  `check` reports (`peakMemoryGB`); `LOADPATH_CONFIRM_CELLS=60000` lowers it,
  never below 1.5x the cells the pick passed on.
  `check` also reports numba's `threadingLayer`: on `workqueue` (pip numba
  on macOS without `brew install libomp`) load cases solve one after
  another, which is safe but slower; it warns when the study has several.
  `LOADPATH_THREADS=1` keeps a run on one thread, `LOADPATH_WORKERS` caps
  its processes.
  `goal.method` (default `auto`) picks the optimizer: the volume-fraction
  search first, then the stress-constrained method (MMA, finer grid) when
  the search's part idles (`stressUtilization`, mean over near-peak von
  Mises, under 0.15) and stiffness is not what sets its mass; the lighter
  part is kept and `methodChoice` says which and why. Brackets with one
  hot corner come out 20-55% lighter that way, at up to 4.5x the time.
  `"search"` or `"constrained"` force one.
- `{"volumeFraction": 0.25, "outcomes": [0.15, 0.4]}`: fixed fractions, a
  trade-off gallery for a person to choose from.

## Cast and milled parts

Not every part is printed. Leave `process` out and say how it is made:

- **Cast or molded**: `"casting": {"draw": "+z", "sides": 2, "draft": 2}`. Nothing
  may undercut the die along `draw`; `draft` (degrees, 1-3 die casting and
  molding, 3-5 sand) is built into the delivered walls and read facet by facet
  in `report.json` → `manufacturing.draft` (`underDraftedArea`, mm^2). Run
  holes along the draw; a hole across it needs a side core and is an undercut.
- **Milled**: `"milling": {"axes": 3, "directions": ["+z"], "toolDiameter": 6}`
  (`axes: 5` cuts from all six sides, `2.5` puts pocket floors every `stepDown`).
  Every void must be reachable by a cutter coming straight in from a listed
  side; slots and inside corners narrower than the cutter are filled.
  `manufacturing.machining` gives tool reach (`toolUnreachableVolume`),
  setups and a rough time and cost (`minutes`, `cost`); quote them when asked
  how long it takes. `milling.rates` overrides the shop rates.
- **Material**: use a wrought or cast card, not an AM one: `6061-t6`,
  `7075-t6`, `a356-t6` (cast), `304` (`loadpath materials` lists them with
  sources); a custom `{E, nu, strength, density}` works for anything else.
  The examples `cast-arm-link` and `milled-motor-plate` show both.

## Comparing ways to make it (explore)

When the user has not settled how the part is made, or asks what it would
cost, `loadpath explore study.json --out out` (MCP `loadpath_explore`)
solves the study once per manufacturing method and material, like Fusion's
generative outcomes: by default fdm (PETG), mjf (PA12), sla (tough resin),
lpbf (AlSi10Mg), cast (A356-T6, drawn along the thinnest axis) and
milled-3axis (6061-T6, cut from both ends of the thinnest axis). Each route
is a full verified run in its own folder; the report ranks them (`rank`:
mass, cost or leadTime), names the lightest, cheapest and fastest passing
route (`picks`), the Pareto fronts (`pareto.massCost`, `costLeadTime`) and
which route is cheapest at 1, 10, 100 and 1000 parts (`crossover`). Narrow
or widen it with `explore.methods`, `explore.materials` (each runs on every
method its card makes) or explicit `explore.routes`; `quantity` spreads
tooling and set-ups. `unitCost` and `leadDays` are service-bureau estimates
(costing.py, about +-50%): say so, and offer a real quote for the route the
user picks. `--plan` lists the routes without running them. It costs about
one study per route (routes of one method share designs), so start with the
methods that are plausible for the user.

## Inside a designer's exterior (push pads, switches)

When the user hands over industrial-design surfacing and the inside is
free: the exterior mesh is the `designSpace` (`{"type": "mesh", "path":
...}`), `skin: {"thickness": 1.0, "open": ["-z"]}` keeps a thin cosmetic
layer of it (open the side the components go in from), and every
component is a `keepOut` with `"insertFrom": "-z"` (the side it comes in
from, or a vector): the keep-out is swept out of the part so no rib blocks
assembly.

Set the skin and `minMember` from the process, not from memory. With the
Materialize MCP server connected, `materialize_list_materials` then
`materialize_get_material` give the material's design limits (minimum
wall, minimum detail, maximum size): SLS flexible TPU lists 0.8 mm walls
and 0.3 mm detail. Use the larger of that and the user's own rule (many
designers hold SLS TPU to 1.0 mm). Once the part is meshed,
`materialize_check_printability` on the STL reports thin walls and other
blockers; fix those before quoting. Give tool access the same way as
components (a cylinder for a driver, `insertFrom` its side).

For a push pad over a switch, add a `switch` block and run
`loadpath switch study.json --out out`: `output` is the plunger (a
preserve id, stroke `direction`, `gap` to the stem, the switch's
`trigger` force and own `travel`), `field` (or `presses`) gives each press
point's force window (`max` may be `[centre, rim]`, a radial gradient;
`min` is the floor a resting thumb must not reach; `"unit": "gf"`),
`clearance` caps the skin's travel toward the board (one entry per
component with `component` and `margin` gives each node its own room),
`output.bearsOn` counts the give of what the switch sits on, `hold` caps the
mean deflection under a load case (a side squeeze, a drop). Start coarse
(`pitches: [1.0]`) to see whether the windows can be met at all, then
refine (`[1.0, 0.7, 0.5]`). The report gives each press's linear and
as-printed (thresholded) fire force; `"nonlinear": true` adds a
large-deflection reading, which is experimental (slow, and it does not
converge on thin-skin designs yet). Read the skin alone first:
`loadpath switch study.json --skin-review` (minutes) lists the
presses the bare skin already puts over their ceilings. Structure inside
can carry such a press to the plunger and pull it down, but only so far
(the touch microphone's corner: 476 gf on the bare skin read on cut
cells, 426 with the best interior so far, ceiling 300; a skin half as
thick still reads 422). Quote `skinReview.summary`, which leads with the cut-cell reading
(`grid.check`: the bare skin and the delivered part read on the mesh
itself on 0.5 mm cells; the design grid's thin-skin model can be 20-35% off,
and a 1 mm design met 13 of 14 windows that read 1 of 14 on finer cells).
`cutCellSkinLimited` and `cutCellTargets` are the presses the bare skin
alone puts over their ceilings and the ceilings it meets; `wholeSkin.cutCell`
is a whole-skin thickness read the same way. The design grid's account and
its local suggestions are in `estimateSummary` and `misses`, which are
estimates. Every switch report's `skinReview` says which misses the
skin decides (`misses[].limitedBy == "skin"`), where (`where`: the
stiffest skin, often the fold into a wall), and what would fix each one:
`target` (the ceiling the design meets), `localSkin` (a thickness within
a radius), `wholeSkin` (one thickness for the whole skin, read again on cut cells), each with the presses
it would push out of window and whether it clears the minimum wall. Never
change the exterior yourself: relay `skinReview.summary` and those
options to the designer and let them choose. `limitedBy == "interior"`
needs no designer: re-run finer or longer. See docs/SWITCH.md.

**Any exterior first goes through `loadpath exterior SRC --out dir`
(MCP `loadpath_exterior`; docs/EXTERIOR.md).** A designer's STEP or mesh,
an AI-generated or scanned mesh (a Tripo/Meshy GLB: open, double-walled,
debris, unit scale, Y-up), an SDF function or build123d object
(`file.py:name`), or an `.npz` SDF grid: out comes one watertight skin and
a report. Give `size` (the real longest extent, mm) for a generated mesh:
it has no units. Read `wrap.patched` (new surface made up over holes) and
`capacity.thinRegions` (too thin for anything but skin) and tell the
designer; never reshape the exterior. Then let `loadpath enclosure` (below)
place the parts from what the device does, or lay them out yourself
against the room inside (the generated-exterior tutorial's `layout.py`
reads the button off the shape and checks every part's room) and hand them
to `handoff` or `enclosure` as given. Handoff takes the same sources directly, as a
path or `{path, size, up}`.

**Any product inside a designer's exterior: `loadpath enclosure brief.json
--out dir [--run]` (MCP `loadpath_enclosure`; `loadpath schema
enclosure`).** Use it whenever the outside is fixed (a rendering, a
surfacing model) and the inside is yours, not only for push pads. The
brief is the exterior, its `openings` (ports, doors; draw each just
through the skin), the designer's `fixed` geometry, and the `components`:
CAD or a `size` box, `mass` (kg), `watts`, and where it goes. Leave
placement to loadpath when the designer did not draw it (`place: "auto"`,
the default for a `size` or any rule): `rotate` (any, an axis, none),
`against` a side (a battery on the floor), `alignTo {opening, face,
recess}` (a USB-C out through its port), `under {at, direction, gap}` (a
switch under a press point), `near` other parts, and
`placement.centerOfMass`. Give the cables too whenever parts are joined
by them: each component's `ports` (`{id, at, direction}` in its own
frame: where the connector is and which way the cable leaves it) and the
brief's `wires` (`{id, from: "battery.lead", to: "board.batt", cable,
length?}`; cable: `"awg26-silicone"`, `"ffc-10x0.5"`, a bundle, a
datasheet's `minBendRadius`). Each candidate position's cables are then
laid as elastic rods: a pigtail that cannot reach, or would bend past its
radius, rules the position out; the chosen layout's cables are followed
along each part's way in (`wires.<id>.assembly.reach`: how far out the
part can be with the plug mated) and their channels are kept clear.
`watts` on one part and `maxTemp` on another keeps them apart. When the
board exists in KiCad, give the file instead of a box: `{"id": "board",
"kicad": {"file": "board.kicad_pcb", "connectors": {"J1": {"opening":
"usb-port"}}, "power": {"U3": 0.6}, "maxTemp": {"U3": 85}}}` (docs/KICAD.md;
`loadpath kicad board.kicad_pcb --out dir` or MCP `loadpath_kicad` to look
first): outline, part heights (3D models, else package heights; read
`heightFrom` and give `heights` for any "assumed"), masses, its connectors
(each its own component facing out over its edge, plug loads written when
aligned to an opening), heat, and screw sleeves under its mounting holes
instead of a cradle. A stack board takes `{"id", "kicad"}` the same way. Read
`enclosure.json`'s `wires`; a cable alone, or across a joint over its
range of motion (flexing radius, coming taut), is `loadpath wires
brief.json` (MCP `loadpath_wires`, docs/WIRES.md): its `ends.<end>.force`
is the strain-relief load on a connector or solder pads, `clips` the load
on each `via` tie, and `motion.flexing: false` marks a one-off move. Boards
on standoffs (a flight controller over a speed controller) are `loadpath
stack brief.json` (MCP `loadpath_stack`): order and gaps by heat in the
real airflow, centre of mass and whether the harness between boards can
bend; put its `inertiaRelief.masses` entry and `envelope` in the carrying
part's study. Parts fitted to another (`installedOn`) are
drawn in that part's frame and move with it. The `assembly` order
matters: each part's position is kept only if it can still get there past
the skin and the parts that went in before it, and its way in (up to four
straight legs, or slides and pivots up to `tilt` degrees for a part that
only goes in tilted, such as a board longer than its door) is checked
exactly and swept clear as its keep-out; `enclosure.json`'s
`assembly.steps` is the assembly in numbered steps to hand the builder. Each
massed part gets a preserved cradle (`mount`) on a side that is in no
other part's way in. `loads`: `drop {g, faces}` (each part's mass at g
toward the face it lands on, the floor's reaction on that face), `squeeze
{axis, force}`, `press {at, face, force}`, `connector {component, type}` (a plug pushed in,
pulled out and levered sideways at the USB-C, USB-A or Micro-B spec
forces, into the socket's board mount, its travel capped), plus `loadCases`, `switch`
(as in handoff) and `feel` blocks. `split` hands the study to shells.py
(one study per half). Read `enclosure.json` before running: a part with
no place or no way in is in `warnings`; tell the designer what would fix
it (an opening, a smaller part, a different order), never edit the
exterior. `--run` designs it and writes `checks.json`: the exterior read
against the delivered surface (`exterior.untouched`, max proud, coverage)
and every load case verified on the delivered part. See
docs/ENCLOSURE.md.

**Start from the designer's CAD with `loadpath handoff brief.json --out
dir` (MCP `loadpath_handoff`; `loadpath schema handoff`).** The brief is
the exterior, the openings in it (an underside a flap closes: no skin is
kept across an opening's inner end, which would otherwise be a membrane a
skin thick across the cavity), the designer's own internal geometry
(`fixed`: rails, a base ring), each component's CAD (STL or STEP; `role:
"switch"` for the switch, `"seat"` for what it sits on, `installedOn` for
parts fitted to another first, `external` for a plug), the `assembly`
order, and the switch's trigger, travel, gap and press windows. It writes
a self-contained study (meshes copied into `dir/cad/`) and `handoff.json`
saying how each value was found: `insertFrom` per component from the
assembly order (one axis, or slide then drop; a part with no clear way in
is reported `blocked`, with what it hits: tell the designer, do not
invent a path), the plunger tip `gap` over the switch's actuator read off
its geometry, the supports where it stands, `switch.output.bearsOn` (the
seat part's give, solved on its own) and one `switch.clearance` entry per
component (each skin node's own room to the part under it, less
`margin`). Paths are translations only: if the real one tilts, give it as
`insertFrom` with `fixed: true`. `pathTolerance` on a component allows a
snap bead's interference. Check `handoff.json` before running; `--run`
designs it straight away.

**A product built in two parts: `loadpath split study.json --out dir`
(MCP `loadpath_split`; `loadpath schema split`).** The study is the whole
product (exterior as designSpace, a `skin`, components as keep-outs with
insertFrom) plus a `split` block. Give `normal` and `at` when the designer
drew the parting line; otherwise it searches: flat cuts across x, y and z,
then tilted planes, stepped lines (over or under each part a flat cut runs
through, round each `avoid`), the silhouette and any crease in the
exterior (a reveal, a shoulder, a character line). It ranks them by seam
that shows (seam in a crease or a `hidden` region counts a quarter), how
far components travel to go in, parts cut through, and for FDM a rim that
is not flat, then lays out the joints on the best four and takes the one
whose snaps fit and hold (`candidates[].final`; `search: "flat"` for axis
planes only). It checks that every component goes into its half
with the other off and that the cover lifts off without catching on
anything. `avoid` lists the designer's A surfaces the seam must not
cross. The joints go inside the skin and the exterior is never changed:
an alignment `lip`, cantilever `snaps` sized to the material's strain
allowance (hooks hang from either half and step in where the wall curves),
and `screws` only if the designer names the face holes may go through
(`screws.from`). Every hole is listed in `split.json` `exteriorChanges`,
so show those to the designer. It writes one study per half; run each
one. Check `split.json` before running: `cut.why` on any rejected
candidate, `snaps.strain` against `snaps.allowable` (an FDM beam bent
across layers gets a warning: print the halves on their sides, or use
SLS), and `warnings` for joints that did not fit. `shells.verify_snap`
reads a hook's strain on cut cells, and `shells.check_closed` checks the
built halves for overlap. A cut need not be flat across an axis:
`normal` a vector tilts it, `steps` [{at, where}] climbs over part of the
product, and `seam: {points}` follows a parting line the designer drew on
the exterior (`cut.surface` in split.json is the surface it used).
`snaps.allowableBasis` says whether the
allowance was measured (a calibration card fitted from the kit's S-XY /
S-Z snap ladders) or is the house rule. See docs/SHELLS.md.

A graded lattice instead of ribs: add `switch.lattice` (`type`, `cell`,
`mode: "where-needed"`) or `--lattice bcc:4:where-needed`. Under a thin
TPU skin only strut cells are soft enough (a 1.0 mm-walled gyroid is a
third to half solid). Each round is read back on the printable mesh with
cut cells (`switch_lattice.readback`, also good for ribs or a drawn part:
`examples/switch/touch_mic_lattice.py --readback part.stl`) and the
windows are divided by that reading's error for the next round. Quote the
read-back (0.33 mm) to a designer, not the design model: on the touch
microphone the 1 mm model read the skin ~30% soft, and ribs that met 13
of 14 there met 0 of 14 on their mesh. The report also gives powder escape
and the material added inside each keep-out and its path.

## Designing for feel (grip, drop, handling noise)

Inside the same kind of exterior, a `feel` block designs for how the
product feels, and `loadpath feel study.json --out out` runs it (MCP:
`loadpath_feel`). Each target is a stiffness window between where the
product is held and where the load comes in; a location is a `region`
(`skin` is the exterior skin), a `shape`, or an exterior `face` like
`"-x"`, a region or face with a `shape` limited to it.

- `squeeze`: `hold`, `press`, `direction`, `force` N and `travel: [min,
  max]` mm. Ask the user what a firm grip should feel like.
- `drop`: `lands` (a face), `mass` (the part's mount), `kg`, `height` m,
  `maxG` (what the part survives) and `maxTravel` (the room it has).
  These need maxTravel >= 2 height / maxG; a study that asks for less is
  refused with that sentence, so tell the user the room the fall needs.
- `isolate`: `hold` (where the hand is), `mount`, `kg`, and either `band`
  (soft mount: isolate from band Hz up, `maxTransmissibility`) or `upTo`
  (stiff mount: keep the resonance over a handling band up to upTo Hz).
  A part on rails moulded into the base cannot be soft-mounted; use
  `upTo`.

- `touch`: a `zone` (a shape) of the exterior under a fingertip, pressed
  along `direction` (default -z), `feel: pliable|firm|rigid` or a
  `stiffness` / `force`+`travel` window (a travel max alone is a firmness
  floor). `feel.contrast: [{softer, firmer, ratio}]` makes zones
  distinctly different. `skinThickness: "auto"` thins the skin under a
  pliable zone, never under the process minimum wall (TPU 1.0 mm); a zone
  that would need less comes back in `touchSkin` with `belowMinWall` and
  the thickness it `needs`: tell the designer, with the options in its
  note. Report `touchZones.delivered`. Design on 1 mm cells and let the
  read-back check finer (0.5 mm design did not converge on the soft-top
  example). Ask the designer which places should feel pliable and which
  firm; never move the exterior to get there.

No supports or load cases are needed. A `switch` block in the same study
is designed with it. Read the bare skin first (`examples/feel/mic_body.py
--baseline` shows how): a target the exterior and preserved parts already
miss cannot be met from inside. The report gives each target on the
design, `thresholded`, `delivered` (the STL read back on its own cut
cells, half the design pitch and no finer than 0.5 mm) and, for isolate,
`modal`. Report the delivered reading. Design on `pitches: [1.0, 0.7]`
when a 1 mm design misses on the delivered reading (it closed the
microphone body's drop). See docs/FEEL.md.

## Designing around electronics heat

Inside the same kind of exterior, a `cooling` block designs the interior
around the heat the parts inside make, and `loadpath cooling study.json
--out out` runs it (MCP: `loadpath_cooling`). `components`: each a keep-out
`region` (or `shape`) with `watts`, `maxTemp` (C) and `conductivity`
(default 20 W/(m K); a battery 1-3, with no watts). `skin`: `maxTemp`, or
`contact: continuous | held | brief` (plastic touch limits 43 / 48 / 60 C
after IEC 62368-1), and a `shape` where it is touched. `ambient` (default
25; ask the user how hot a room it must work in). `vents`: openings the
exterior already has, with `gravity` the way down in use; never add one
yourself, suggest it to the designer. The interior grows pads and ribs
that carry heat from the hot parts and spread it under the skin; still air
conducts in the gaps, the outside loses heat at 10 W/(m^2 K).

Read the bare skin and a filled interior first (`examples/cooling/handheld.py
--baseline`). The skin's mean rise is about watts / (h x outside area)
whatever is inside: if that alone is over the touch limit, or the filled
interior still misses a limit, no interior meets it, and the designer
decides (more area, vents, a metal spreader, less power). Feel and switch
blocks in the same study are designed with it. Report the `delivered`
readings (the STL on two finer grids) and their `gridChange`. See
docs/COOLING.md.

Ask how the product is used before designing for a steady load: most
handhelds idle, then burst (a game, a call, a recording) or charge. A
`cycle` (`phases`: each `seconds`, `watts` per component id, `steps`; then
`start: first | ambient` or `repeat: true`) holds every limit at its
hottest moment, so the parts' and the plastic's heat capacity carry the
burst; give a battery its `heatCapacity` (J/K, about 1 J/K per gram:
45 J/K for 45 g). A steady design at the burst's power asks for
far more material, or for something no interior can do.

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
  also needs `"thermal": {"ambient": 55}` (the whole part at that
  temperature, softened for it) for its static safety factor), `safetyFactor` (default 1.5), and
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
- Printed lattices are softer than their design (33 published compression
  tests: 1.0-4.1x). Name the study's `process` and the homogenized check
  reads the print: an as-printed stiffness factor per process (FDM, SLS/MJF,
  resin), strut or sheet cell and density, fitted to those tests and held
  out (median error 1.23x, worst 2x; report `latticePrint`). FDM gyroid at
  rho 0.2 reads 0.65 of the design, FDM struts ~0.4, SLS ~0.93. Its
  first-yield reading then sits under every measured strength (sheets
  read strain-based, struts from the stress they carry). It also
  checks wall and strut buckling (`cases[].bucklingFactor`, `governs`).
  Still say a lattice part's deflection is +-20% (2x at worst) unless the
  material is calibrated on the printed gyroid coupon, whose measured
  factor replaces the fit; `process.latticeStiffness` sets one by hand
  (docs/VALIDATION.md, `loadpath.lattice_print`).
- Don't use a lattice to make a part stiffer per gram: solid topology
  optimization wins that. Use one to stop an impact (a bumper, a hard stop, a
  drop pad): give a field graph an `impact` block (envelope, density node,
  mass and velocity or energy, forceLimit, `"grade": true`, cell and minWall).
  It returns the peak force and crush, and a density table that crushes every
  layer at the same force. If the core is round where it is hit (a tyre, a
  dome) or the thing hit is (a ball, a pole, an edge), add
  `"striker": {"side": "+", "shape": "flat" | "sphere" | "cylinder", "radius"}`:
  the force then grows with the contact patch, and the layer model without
  it reads low (17% on a round tyre against a wall). A `vibration` block gives natural frequencies with
  point masses. See docs/IMPACT.md in the repo.
- For a part meant to bend (a TPU finger, a flexure, a snap fit, a
  compliant gripper), give the field graph a `largeDeflection` block:
  `fixed` node, dead `loads`, prescribed `displacements`, `tendons` (path of
  fixed points and guides `{"at": [x, y, z]}`; `tension`, or a servo's
  `stroke` and `stiffness`), `contact` (`obstacles`: plane, sphere,
  cylinder, box, optionally `move`d; `"self": true` for the part on itself)
  and `probes`. It reports how far the load got, peak strain, travel,
  tendon tension and each contact's force and penetration. Frictionless
  unless `contact.friction` (or an obstacle's `friction`) sets a Coulomb
  mu; then each contact also reports its friction force and the nodes
  sliding. A self-contact solve where faces rotate fast can stop short and says so
  (`loadReached`). See docs/MECHANISMS.md.
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
  (about 270 bytes a grid point) against what is free, in milliseconds.
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
