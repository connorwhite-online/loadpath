# loadpath: context for whoever picks this up

loadpath is an open-source, agent-native generative
design library: topology optimization for printed parts, driven by
constraints, keep-outs and loads, with its own verification and an eval loop.
It is published at https://github.com/connorwhite-online/loadpath (MIT).
It started inside another project's CAD service and now lives in this
repository.

## The goal

Parity with Fusion 360's generative design or better, open source, driven by
agents (MCP, CLI, JSON studies). The north star for capability and finish is
**nTop**: implicit (field) modelling, smooth blends, field-driven lattices,
and later thermal / heat-exchanger design. Parts are almost always 3D printed:
optimize for the hobbyist (FDM) without locking out SLS, MJF, resin or
volumetric printing.

## The test subject

Most hand-written eval cases are links of an open-source 6-axis printed robot
arm (examples/robot-arm/): real interfaces (servo cradles, horn hubs,
cycloidal-drive sockets with driver access) and loads derived from its URDF.
A procedurally generated corpus (loadpath/evals/corpus.py) keeps tuning from
fitting those parts alone. What goes wrong on a real part becomes an eval
metric.

## Architecture (modules)

| Module | Role |
| --- | --- |
| `spec.py`, `shapes.py`, `materials.py` | JSON study -> Study; shapes (box, turned box (`axes`), cylinder/tube, sphere, mesh, union); 16 materials, each tagged with the processes that make it; `process` (fdm/sls/mjf/sla/dlp/volumetric/lpbf) |
| `processcheck.py` | SLS/MJF and SLA/DLP checks on a delivered part (mesh or field): rule cards (`RULES`), walls, enclosed and narrow-exit powder/resin, warp-prone thin panels, resin peel area, suction cups and supports (via printcheck). run_study puts it in the best outcome's `report["process"]` |
| `domain.py`, `fem.py` | voxel domain, boundary conditions (forces, moments, bearing loads, pins, gravity), hex8 FEA, CG + smoothed-aggregation AMG |
| `mg.py`, `gpu.py` | matrix-free geometric multigrid for large grids (SIMP uses it from 100k cells; `LOADPATH_SOLVER=mg|assembled`); `gpu.py` runs its solve with torch on CUDA, Apple Metal (all float32 on the GPU: iterative refinement with a rigid-motion-free cell kernel, within ~4e-6 of float64; `LOADPATH_FP32=0` puts the CG back on the CPU in float64, `=1` forces float32-only anywhere) or the CPU threads (`LOADPATH_DEVICE=auto|cuda|mps|torch|cpu`, extra `gpu`) |
| `simp.py` | SIMP with filter, Heaviside continuation, robust erode/open (minMember; 1.4 mm default), closing at delivery (minHole), OC; live-cell reduced solves; dead-material pruning; escape-hole rounds |
| `conjugate.py` | conjugate heat for sinks and cold plates in a duct: flow.py LBM flow at a target pressure drop, then solid + fluid heat in one finite-volume solve (upwind advection, AIR multigrid); thermal resistance and energy balance; `bench/lattice_heatsink.py` rates lattices against solid TO |
| `flowtopo.py` | flow-aware heat sink topology optimization: Darcy flow (Borrvall-Petersson) + advection-conduction at a fixed pressure drop, adjoint gradients, MMA with permeability continuation; not yet competitive with plates (docs/BENCHMARKS.md) |
| `escape.py` | escape holes for density designs: finds voids the optimizer sealed on its own grid, fills tiny ones, drills straight holes through thin calm wall (mirrored, maxPath through the void), pinned void while the optimizer re-converges |
| `directions.py` | loads of unknown direction (`"direction": "any"`, `"cone"`): exact worst-case compliance from three axis solves (3x3 eigenproblem, trust-region on a cap for a cone or a fixed load alongside), soft max for the optimizer's sensitivity, worst von Mises direction for verify |
| `mma.py`, `constrained.py` | MMA (from the published method), stress/deflection-constrained mode |
| `mechanism.py` | compliant mechanisms (max output displacement against springs; several input/output pairs; `robust` eroded/blueprint/dilated so no one-node hinges). Validated against a port of Sigmund's 99-line mechanism code (`python -m loadpath.benchmarks.mechanisms`: inverter +1-2%, gripper +1%). TPU gripper driven by a servo: examples/mechanism/gripper.py, docs/MECHANISMS.md |
| `switch.py` | target-force push pads inside a fixed exterior (study `switch` block, `loadpath switch`): fire force per press F_i = (gap + travel + T (c + c_seat))/a_i from one solve per press + one plunger (output.bearsOn: the switch's seat in the mesh, r = l - m, or a separate part's compliance c_seat, part_compliance on cut cells; clearance per component: each node's room to the part under it by a ray along the press) solve; min volume with per-press log-force ceiling/floor (MMA), press travel, skin clearance (KS), hold cases; coarse-to-fine `pitches`; thin skin on coarse grids as a density floor (`_thin_skin`); void cells dropped and one SuperLU factorization for every press; `check_fire_nonlinear` re-reads fire force with large deflection. examples/switch/touch_pad.py, docs/SWITCH.md. keepOut `insertFrom` (spec.py, shapes.Sweep) keeps assembly paths clear |
| `feel.py` | feel targets inside a fixed exterior (study `feel` block, `loadpath feel`): `squeeze` (grip travel at a force), `drop` (peak g and travel of a part on its mount: 2E/travel^2 <= k <= (G m g)^2/2E), `isolate` (soft mount from `band` Hz, or resonance kept over a handling band `upTo`); each a log-compliance window between its own holds and loads, one solve per direction, targets sharing a hold share a factorization; added to switch.optimize_switch as `providers` (with or without a switch block); floors on the eroded and ceilings on the dilated design; delivered mesh meshed with thin members kept and read back at half pitch; isolate checked with modes (modal.py). examples/feel/mic_body.py, docs/FEEL.md |
| `handoff.py` | the designer's handoff (`loadpath handoff`, MCP `loadpath_handoff`, schema `handoff`): exterior, openings, fixed geometry and component CAD in, a push-pad study out: keep-outs and a clearance entry per component, insertFrom per component from the assembly order (six axes, then slide-then-drop; blocked parts reported with what they hit; parts installedOn another travel with it; fixed material less the components' places), supports from the side it stands on, plunger tip `gap` over the switch's actuator, bearsOn the seat part. Meshes copied into out/cad/ |
| `shells.py` | split shells (`loadpath split`, MCP `loadpath_split`, schema `split`, `apply_split`): a study's fixed exterior cut into a base and a cover. The cut given or searched: axis planes, then non-flat cuts from parting.py (tilted planes, steps over parts and `avoid`, the silhouette, creases in the exterior's profile); score = seam that shows (a quarter in a crease or `hidden`) + 4 x component travel + 20 per component across the cut + depth penalty + FDM rim-not-flat; the best four get joints and the one that holds best wins (15 per missing joint, 50 over strain), checked: each component's way into its half with the other off (handoff.assembly_paths), the cover lifting off without catching on a base component, the seam missing `avoid`. Joints inside the skin: an alignment lip (marching-cubes band along the inner wall, simplified mesh in out/cad/), screw posts (fasteners.py entries in each half, bore fit by fits.py; holes only through the face `screws.from` names, reported in exteriorChanges; posts step in from the wall where the face curves), cantilever snaps (hooks on either half, shifted in where the wall curves; strain 1.5 t y / L^2 against `snap_allowable`; `verify_snap` reads the hook on cut cells). Each half's components sweep out through its own cut face; loads only on the cover reach the base on its rim. `check_closed` checks built halves for overlap closed and on the way off |
| `shroud.py` | generated exteriors (study `shroud`, `loadpath shroud`): expand_shroud (from spec.study_from_dict) runs a coarse free seed, wraps seed + preserve + `conceal` (grown by skin + gap + a pitch) by closing (`reach`), fills cavities, smooths the SDF (`smooth`), blends interfaces in and non-concealed bore:false keep-outs out, clips to the design space; then level-set shape steps on the skin alone (velocity log4 energy share against a quantile price, 5 prices tried, kept if skin volume x max(c/c_seed, 1)^4 drops; stiffness x volume alone drives it back to the box). Result: designSpace {type: sdf, shroud: key} (cache LOADPATH_CACHE/shroud) + `skin`. `coreshape.py`: an exchanger core cut down by its own simulated duty (porous.py) and `compare` against the box and a same-volume box |
| `skinreview.py` | feedback for the designer when a fixed exterior keeps a push pad off its force map (report `skinReview`, `loadpath switch --skin-review`): bare-skin reading per press, misses classed skin- or interior-limited, the stiffest skin from the fire-force sensitivity, and suggested target, local skin and whole-skin thickness changes read back with the interior held, with side effects and the minimum wall. Never edits the exterior |
| `nonlinear.py` | large deflection (total Lagrangian hex8, neo-Hookean, Newton, force, stroke or tendon (`Tendon`: cable through guides, follows the part) loads, stability check that follows buckled branches), linear buckling, `verify_mechanism` (also `MechanismOptions(verify_nonlinear=True)`), `stroke_mechanism` (input driven through a stroke against a workpiece spring: travel, actuator and grip force) |
| `mechanism_nl.py` | compliant mechanisms optimized on the large-deflection response (prescribed input stroke, Newton on the full grid, adjoint, Wang-Lazarov-Sigmund void interpolation, backtracking on failed solves; per-element modulus and Poisson's ratio). Damped Newton steps; sharpening retried up to three times. Inverter at 30% stroke: +20% over the linear design (examples/mechanism/large_stroke.py). Curling TPU finger with `robust`: 8.7 mm tip for a 5 mm pull vs 0.7 mm (examples/mechanism/curling_finger.py); without robust it relies on grey hinges. Tendon input (`nonlinear.Tendon` constant tension, `ServoTendon` wound-in cable with give) and `RigidCylinder` contact with a grip-force objective; `tendon_grip` checks the thresholded design. Grip example: 9x a plain bar (examples/mechanism/tendon_grip.py) |
| `lattice_tensors.py` | anisotropic lattice homogenization: `unit_cell_tensors` (six periodic unit-cell solves: the full 6x6 stiffness, symmetrized over the cell's cubic or tetragonal group, and a wall-stress surface fitted over ~15k macro-stress directions, a quartic floored by a positive-definite quadratic, never under the sampled peaks), tables in `data/lattice_tensors.json` for every sheet and strut cell (`bench/lattice_tensors.py`), `tensors_at` (any density, to the solid at 1), Voigt rotations. `homogenized.homogenized_check` uses it by default: each lattice cell its own rotated tensor (frame from the lattice coordinates' Jacobian and any transform nodes above it), per-element condensed hex8 (`fem.hex8_ke_batch`), matrix-free under the isotropic multigrid on big grids; `anisotropic=False` or LOADPATH_LATTICE_ANISO=0 gives the old scalar reading. Validation: `bench/lattice_aniso_validate.py` (resolved voxel FE and CalculiX). Stretched (`aspect`) cells: `aspect_table` solves them on first use and caches them (`LOADPATH_CACHE`). `lattice.lattice_infill` sizes on the same tensors (`LatticeOptions.anisotropic`) |
| `lattice_print.py` | as-printed lattice stiffness: `print_factor(process, cell, rho)` = 1 / exp(process + strut + density terms), capped at 1, fitted to 33 published printed-lattice compression tests (`bench/lattice_print_fit.py`, held out: median modulus error 1.68x -> 1.23x). `homogenized_check` applies it to the lattice's stiffness by the study's process (`process.latticeStiffness` or a calibrated card's `lattice.stiffnessFactor` override; `as_printed=False` or LOADPATH_LATTICE_AS_PRINTED=0 for the design); wall stress is read strain-based for sheets, from the carried stress for struts (`STRAIN_SHARE`, chosen and held out a paper at a time). Report `latticePrint` |
| `lattice_buckling.py` | periodic unit-cell linear buckling (`unit_cell_buckling`: 2x2x2 strut blocks, one sheet cell; Euler-checked), tables `data/lattice_buckling.json` (`bench/lattice_buckling.py`, needs pypardiso to be quick). `homogenized_check` compares each lattice cell's most compressive macro stress with `BUCKLING_KNOCKDOWN` (0.5) x the table x its as-printed stiffness: `cases[].bucklingFactor`, `yieldSafetyFactor`, `governs` |
| `lattice_orient.py` | lattice orientation as a design variable: `optimize_orientation` turns a field graph's lattice (its `orient`, which turns the cells and not their grading fields) for the best safety factor or least compliance: one analysis for the macro stresses, a tensor-only screen of ~1,500 rotations, full homogenized checks of the best, Nelder-Mead. CLI `loadpath lattice-orient`. Mass win: `bench/lattice_orient_mass.py`. `optimize_zones` (`--zones N`): orient.zones, cells turned zone by zone (k-means on principal stress axis + position), cross-faded where zones meet (fields.orient_weights); `bench/lattice_zones_mass.py` |
| `lattice_grade.py` | field-graph lattice wall graded to the stress: `grade_lattice` sizes a `density` grid (fully stressed, 28 candidate densities per cell on the tensor stress surfaces, skin-junction factor) on the homogenized check, `orient` 'uniform' or N zones turned for the least material; CLI `loadpath lattice-grade` (writes graded.json + density.npz). `bench/lattice_grade_mass.py` |
| `impact.py` | crush and impact for TPMS lattice cores: plastic collapse of a unit cell by the elastic compensation method (`collapse_strength`, fits in `CRUSH`), Gibson-Ashby densification, the layered crush model (`crush_curve`, weakest layer first), where a mass at a speed stops (`impact_response`: crush, peak force, g, efficiency), `grade_for_impact` (density per layer so every layer crushes at one force), `solid_impact` (a solid part as a spring), `limit_load` (3D elastic compensation on a homogenized voxel model, crushable-foam yield), `grade_3d` (per-cell grading that shares each layer's crush force where the load goes and empties idle cells), `crush_law` (elastic buckling law for elastomers such as TPU, with rebound). Field graph `impact` block; guide and the corner-bumper case in docs/IMPACT.md |
| `modal.py` | (studies: `vibration` block, `verify.modal_check` on the delivered part at cells' filled share, the search's margin (f / fmin)^2, the masses' stiffness-only inertial cases in the optimizer, `domain.inertial_cases`) natural frequencies, mass-normalized modes and effective mass fractions on the voxel mesh (lumped mass, per-cell modulus and density so homogenized lattices work), point masses, loss-factor damping and single-mode and base-excitation (`base_transmissibility`, modal superposition) transmissibility. Field graph `vibration` block |
| `isolator.py` | lattice vibration isolators (tuned at TPU's dynamic modulus): homogenized foot stiffness, rigid payload-on-feet modes, `tune_mount` (per-foot stiffness with the elastic centre under the CG, so the payload sits level and bounce does not rock), shock margin before the first TPU foot buckles. Case: `examples/isolator/mount.py` |
| `viscoelastic.py` | elastomer (TPU) time and rate dependence: log-time creep law with one rate m (sourced 0.045, or a calibration card's `creep.m` from the K-25 coupon), `creep_factor` (sag after t), `dynamic_factor` (stiffness at f Hz), `event_factor` (stiffening in an impact of duration T; the `impact` block applies it to elastomers unless `"rate": false`) |
| `am.py` | overhang control (Langelaar filter) |
| `manufacture.py` | non-additive constraints: milling (2.5/3/5-axis access, cutter diameter, pocket floors) and casting (draw, 1/2-sided) through one tool/mold access filter on the projected field, circular patterns (tied sectors), maxMember (local volume penalty); `report` measures undercuts and the thickest member on the delivered mesh, the draft of every cast facet (`draft_report`; `drafted_distance` builds the draft into the meshed field in reconstruct), and a milled part's tool reach and machining time/cost (`machining_report`) |
| `explore.py` | one study, an outcome per manufacturing route (method x material: fdm, sls, mjf, sla, dlp, jetting, volumetric, lpbf, cast, milled-2.5/3/5axis): each route a full `loadpath run` of the varied study in its own folder (routes of one method in turn with shared design memory, methods in parallel, unchanged routes read back), ranked by mass/cost/lead time with Pareto fronts and the cheapest route per quantity; `loadpath explore`, MCP `loadpath_explore`, study field `explore` |
| `costing.py` | service-bureau cost and lead-time estimate per method from the delivered mesh (filament + printer hours, powder + build share, resin + layers, LPBF powder + melt and recoat hours + post-processing, casting tooling + metal + foundry, milling programming + set-ups + stock + roughing/finishing hours); every rate in `RATES`/`MATERIAL_PRICE`, overridable (`explore.costs`) |
| `reconstruct.py` | field -> printable mesh: cubic upsampling, opening at min member, fade at keep-outs, smooth-min rounding at the design boundary (flush at supports and loads), 3 mm fillets into preserved geometry, exact booleans |
| `verify.py` | independent voxelization + FEA of the delivered mesh, hotspot submodels, layer-bond check, boundary-cut metric, stress fields |
| `failure.py` | what breaks a printed part: the pressure-modified von Mises stress (compressive over tensile strength by material family, `material.compressionRatio`) read against tensile strength, and the layer bond's tension and shear together (`process.interlayerShear`); verify.py and crosscheck.py both read it; `report["failure"]` (and `layers.ifBuiltAlong` when a study gives no build direction). docs/PHYSICS.md |
| `convection.py` | still air (`thermal.convection: "natural"`, the default): per-face natural convection by direction and film temperature, choked between facing walls, plus radiation by emissivity and each face's view of the room (rays through the voxel grid); thermal.StudyHeat iterates it with its own temperatures |
| `crosscheck.py` | second opinion on a tet mesh: gmsh + native P2 assembly + p-multigrid; CalculiX deck writer |
| `study.py` | the search on safety factor / deflection, fine-grid confirm of the pick, multi-material, parallel |
| `render.py` | clay renders, stress maps, interface close-ups |
| `viewer.py`, `viewer.html` | interactive HTML viewer (three.js): stress / safety factor / displacement fields, deformation, section plane, compare, optimization playback, heat exchanger fluids |
| `editor.py`, `editor.html` | browser editor for field graphs (`loadpath edit`): node graph, inspector sliders, the graph compiled to GLSL and raymarched live (numbers are uniforms), bakes (mesh / result / grid / cell warp) as 3D textures, local server for save, bake and run |
| `preview.py` | the editor's live stress preview: the field sampled on a coarse grid (no mesh), surface cells at their filled share, matrix-free multigrid, a stress map as a 3D texture in about a second |
| `assembly.py` | loads and design space from a mechanism: URDF reader, Newton-Euler over a pose / speed / acceleration / payload / push / stall sweep limited by each joint's rated effort, governing cases by greedy stress-envelope cover on the solid design space (six unit-wrench solves), neighbours' swept volumes as mesh keep-outs. `loadpath assembly`; the robot-arm comparison in examples/robot-arm/derive.py |
| `cadimport.py` | CAD in: STEP (OpenCascade, `loadpath[step]`) or STL/3MF/OBJ read and healed (open shells fanned closed; five of the robot arm's link STEPs are open shells), bores (exact cylinders from the B-rep, fitted from a mesh; merged across split faces, coaxial steps as one bore, through or blind, bolt or bore), exposed flat faces with the bolts opening onto them, interfaces, a suggested support and placeholder load, `draft_study` (design space part/hull/box, rings and pads preserved cut to the part, mating faces cleared). `loadpath import`, MCP `loadpath_import_cad`; mesh shapes take .step paths. docs/IMPORT.md |
| `fe_export.py` | FE model export (`loadpath fe-export`): the cross-check's quadratic tet mesh, held nodes and nodal forces as a Nastran SOL 101 deck (GRID*/CTETRA/MAT1 or MAT9/SPC1/FORCE*, pins as CORD2C frames) or an Abaqus / CalculiX deck; `read_nastran` reads its own decks back |
| `aero.py` | External flow (`loadpath external-flow`): the part in a lattice Boltzmann tunnel (flow.py kernels, sponge layers at both ends), drag/side/lift and moment by momentum exchange, the wall forces as a load case of k-means patches; Reynolds number capped to what the grid holds (`reScale`); see docs/EXTERNAL-FLOW.md |
| `fan.py` | axial fan rotors: blade-element design and performance (Euler work, simple radial equilibrium or an even exit for a resistance close behind, Carter deviation, Lieblein diffusion factor and equivalent diffusion profile loss with a low-Re factor, Howell endwall loss as a passage average; tip band, shroud gap or open tip with Prandtl loss), `optimize` (most flow, or any objective, through a system curve within a shaft power, speed range and axial depth, stall margin kept), `strength` (band hoop, blade root pull and bending), `rotor_mesh` (airfoil blades, bellmouth band, revolved hub; one watertight body). A correlation model, not CFD: docs/FANS.md |
| `ribs.py` | Rib layouts for the field graph's `ribs` op: pattern distances (square, isogrid, radial), polylines projected onto a surface (along a direction or to the nearest point), the foot of a point on a surface, and evenly spaced principal stress lines from a verify fields dict (stress from the displacement's local least-squares gradient) |
| `patterns.py` | The field graph's `pattern` op: copies (R, t) for linear, circular, mirror and along-curve patterns, and their union evaluated over each point's nearest copies |
| `nesting.py` | build-volume nesting (`loadpath nest`): parts and copies on the plate (turned about z) or stacked in the powder cake (24 orientations), conservative voxel grids grown by the gap, free positions for every translation by FFT correlation, the gap measured on the placed surfaces |
| `vdb.py` | OpenVDB `.vdb` float grids in and out in pure Python (5-4-3 tree; zip, blosc, active-mask, half float read; zip or none written): `save_sdf` to a .vdb, sdf shapes, exteriors and the field `grid` op read them |
| `layerfiles.py` | powder-bed layer files (`loadpath slices`): exact contours per layer (outer counter-clockwise, holes clockwise by nesting), even-odd scan hatches turned 67 degrees a layer, ASCII CLI written and read back |
| inertia relief | study `inertiaRelief`: `domain.inertia_balance` (f - M Phi a), `boundary_conditions` balances on the mesh as solid and holds it 3-2-1 (`pressure.stabilize` on every node); `simp._relieved` rebalances on the design's densities each iteration with the inertia term in the sensitivity (u taken M-orthogonal to the rigid modes so da/dx drops out) |
| `mcp_server.py`, `__main__.py` | local MCP server and CLI |
| `checks/` | checks on any mesh, no study needed: `check_dfm` (wall thickness, overhangs, trapped voids), `check_networks` (two-fluid isolation), `check_fit` (a board fits its enclosure), `fea_probe` (relative hotspot finder), `opening` |
| `flow.py` | lattice-Boltzmann flow + heat (numba, optional `flow` extra): periodic TPMS cells for f and Nu (`cell_flow`), whole networks port to port with the pressure drop split by region and the core's flow sharing (`network_flow`, `exchanger_flow` for a heatx design beside its rating). TRT collisions: BGK's permeability moves with viscosity |
| `sdf_kit.py`, `exchanger.py` | field-modelling toolkit (primitives, TPMS, smooth booleans, exact mesh booleans) and the rectangular dual-fluid `exchanger_core` built from it; `heatx.py` is the newer, study-driven exchanger. Old `loadpath.checks.sdf_kit` / `.exchanger` imports still resolve |
| `fields.py` | nTop-style field graphs (JSON or Python): named solid and scalar nodes, every numeric parameter can be a field (variable shell, variable fillet, graded TPMS cell/wall/density), `result` nodes from verify stress, exact mesh distance, smooth meshing (marching cubes + relax + projection). `loadpath field`, schema `field`, MCP `loadpath_field`; guide in docs/fields.md |
| `sharp.py`, `step.py` | exact sharp edges for field graphs: `plan(graph)` splits a graph at its hard booleans with plain shapes or meshes, meshes the organic side from a field extended into what the shape removes, and does the booleans exactly (manifold3d); `edge_accuracy` measures how far a mesh is from its exact shapes' surfaces. `step.export_step` turns the same plan (reconstruct.py attaches one to every study part) into an OpenCascade B-rep: exact planes/cylinders/spheres, organic leaves faceted and decimated, written as STEP (`run --step`, `field --step`, extra `step` = cadquery-ocp) |
| `thermal.py` | steady conduction on the voxel grid (fixed T, heat input, design-dependent convection), heat-sink optimization (MMA), and thermo-mechanical design: temperature softens the material (MODULUS_RETENTION) and, with `alpha=`, expands it (EXPANSION, `fem.thermal_load`); objective then twice the elastic strain energy, mechanical + thermal adjoints. `thermoelastic` rates a design. In the study JSON as `thermal` (spec.Thermal): `StudyHeat` puts it on the domain, simp re-solves the design's temperature every iteration with exact gradients, verify_part derates moduli and strength and takes the free expansion off the stress. Duty cycles (`transient`, `solve_cycle`, thermal.cycle): hot moment designed for (steady rise scaled node by node to the cycle, refreshed every 5 iterations), hot and cold checked, thermal-cycling fatigue in durability; `fatigueWeight` adds a p-norm of the hot-to-cold swing's stress over retention (support/load band left out) to the objective |
| `latticestep.py`, `ibrep.py` | lattice parts as exact CAD (`"step": {"lattice": "exact"}`): `ibrep` meshes an implicit level set by manifold surface nets (pinches split apart) into bicubic Bezier patches on the true field that sew exactly; `latticestep.export_lattice_step` takes a TPMS sheet sealed under a skin as the envelope less each side's cavities, and anything else (strut cells, Voronoi, networks, graded, open) as the envelope less one void level set carried past the envelope, the open part cut by splitting faces and keeping pieces by the fields; the report has `ok`, `complete`, `readBack`, `stlDistance` |
| `narrowband.py` | field meshing that samples only near the surface: blocks and leaves tested against a Lipschitz bound, the rest skipped; same mesh as dense marching cubes, from 20-50% of the samples on solid parts, so the cell budget counts sampled cells, not the box. `fields.field_mesh` uses it by default (falls back to every block if the mesh is not watertight); `lattice_infill` meshes the exact TPMS through it (`exact_surface`), so walls are the specified thickness, not the grid's. `march_blocks` yields the blocks slab by slab; `fields.field_stl` (`mesh.stream`) streams them to an STL in bounded memory, seams matched across slabs. `peval` spreads field evaluation over threads (bit-identical); `watertight`/`signed_volume`/`shell_volumes` replace trimesh's slow bookkeeping on big meshes |
| `selfx.py` | compiled (numba) self-intersection search: every face's box in a spatial hash, exact triangle-triangle test, parallel over faces; `crease.crossing_faces` uses it (the relax's unfold and the crease pass call it ~20 times a mesh), same faces as the numpy k-d tree path (plus the rare long face that path's radius missed), ~35x faster. `LOADPATH_SELFX=numpy` forces the numpy path |
| `meshops.py` | compiled mesh bookkeeping for the field mesher (face/vertex normals, vertex-face CSR, edge adjacency and edge pairs, face lookups), each with its numpy fallback (`LOADPATH_MESHOPS=numpy`); kernels compile once (~18 s) into numba's cache, then load in ~1.5 s |
| `fieldops.py` | compiled hot field ops (numba, nogil so peval threads them): double-precision TPMS (bit for bit numpy's, ~4x) and the graded-cell warp lookup (~5x); `lattice.tpms` and `fields._CellWarp` use them, `LOADPATH_FIELDOPS=numpy` turns them off |
| `fieldjit.py` | a field graph compiled to one numba kernel per node asked for: every node, point by point, in one pass (no whole-array temporaries), numbers passed as a parameter array so the kernel depends on the graph's structure only (a tune reuses it; cached on disk, `LOADPATH_FIELDJIT_CACHE`); nodes it does not compile (mesh distance, points, strut/conformal/zoned/density lattices, blend grading) run on numpy first and are read as inputs. `FieldGraph.evaluate` uses it; bit for bit numpy's but for BLAS 3-vector dots and numpy's SIMD pow/exp/log (~1e-15), and under `lattice.single_precision` libm float32 trig vs numpy's SIMD (~1e-6 mm). `LOADPATH_FIELDJIT=0` turns it off |
| `knn.py` | compiled k-d tree whose search starts from the previous point's neighbours (mesher queries come in order): `MeshField`'s k=12 nearest centroids, same as cKDTree in the same order, ~2.7x faster on the mesher's queries; `LOADPATH_KNN=kdtree` keeps scipy's |
| `amgfast.py` | pyamg's three Gauss-Seidel smoothers and scipy's BSR matrix product (pyamg setup, on all cores) (gauss_seidel, bsr_gauss_seidel, block_gauss_seidel) as GIL-free numba kernels, bit for bit pyamg's, so fem.Solver.solve_many's and crosscheck's load-case threads overlap; installed by fem.amg_solver, `LOADPATH_AMGFAST=0` keeps pyamg's. The robot-arm upper arm full eval 1528 -> 1140 s, same design |
| `mgops.py` | compiled kernels for mg.py on numba's threads: the matrix-free operator as two parallel passes (block gather + 24x24 product with the cell loop innermost, then a node-by-node scatter), the Chebyshev smoother's and CG's fused updates, dot products off OpenBLAS (its spinning threads halved the operator's speed), prolong/restrict as one gather pass. Same numbers to rounding, not to the bit; constrained trial parts ~2.6x faster in the optimizer. `LOADPATH_MGOPS=numpy` restores the numpy path |
| `femops.py` | compiled SIMP kernels, bit for bit numpy's: per-element strain energy (gather fused, ~16x) and Stiffness.assemble's scatter; `LOADPATH_FEMOPS=numpy` turns them off |
| `decimate.py` | adaptive decimation of a field mesh against the field: shortest-first edge collapses in independent sets, each kept only if every reshaped triangle stays within `tol` of the surface and does not turn over; link condition keeps it watertight. `mesh.decimate` (mm), also per slab in `field_stl` with seams locked |
| `drains.py` | drain / escape holes for a field graph's sealed voids (powder, resin): finds them on a grid of the field, fills specks, drills the process's hole at the calmest flat sites (stress from the graph's `result` node) on smooth-unioned bosses, relabels until nothing is sealed, then checks the powder path and processcheck's escape ball. Holes are graph nodes. run_field calls it when the graph has a `process` |

Two of some things, on purpose for now:

- **Voxelizers.** `verify.voxelize` fills cells by ray parity through
  trimesh's ray tree on an unpadded grid whose origin is the min corner of
  cell (0,0,0); the optimizer domain and `verify_part` depend on that
  alignment. `voxels.py` is the scalable one: a scanline rasterizer that
  bins triangles straight to voxel columns (no ray tree, flat memory: a
  5.5M-triangle gyroid exchanger voxelizes in seconds), padded grids with
  the origin at the centre of voxel (0,0,0), `field_solid` for parts built
  from a field, and a tiled distance-transform wall probe. `check_dfm` and
  `check_networks` use it. `sdf_kit._solid_voxels` (trimesh surface voxels
  plus a parity vote per pocket) is still behind `from_mesh` and
  `check_fit`. Moving `verify.voxelize` onto `voxels.mesh_solid` changes
  eval numbers, so it belongs with a re-baseline.
- **FEA.** `fem.py` is the real solver: material modulus, units,
  smoothed-aggregation AMG, used by the optimizer, `verify.py` and
  `study.py`. `checks/fea.py` (`fea_probe`) is a seconds-scale critic with
  E = 1 that only ranks hotspots on a mesh with no study behind it; its
  numbers are relative by design. Rebuilding it on `fem.HexMesh` would drop
  ~150 lines; do it when its output can be re-pinned.
| `evals/` | cases with engineering requirements; `python -m loadpath.evals.run` scores them |

## How to run things

```bash
pip install -e ".[render,mcp,crosscheck]"         # from the repo root: numpy scipy pyamg scikit-image trimesh manifold3d (+ matplotlib, mcp, gmsh)
python3 tests/test_loadpath.py                  # 22 tests, ~2.5 min (crosscheck test skips without gmsh)
python3 tests/test_nonlinear.py                 # elastica, Euler, post-buckling, ~2 min
python3 tests/test_impact.py                    # crush, impact grading, limit load, modal, isolators, ~40 s
python -m loadpath check study.json             # validate a study
python -m loadpath run study.json --out out/    # optimize
python -m loadpath.evals.run --jobs auto --out /tmp/evals   # full suite; auto = as many cases as fit in memory
python -m loadpath.evals.run --tier coarse --corpus 6 --baseline loadpath/evals/baselines/coarse.json \
    --gate regressions --out /tmp/coarse                 # smoke tier for every push, ~7 min on 4 cores
python tests/test_evals.py                      # pareto, tiers, corpus, critic
python -m loadpath mcp                          # local MCP server
```


Python API worth knowing: `run_study`, `verify_part(..., fields=True)`,
`crosscheck(study, mesh, size=1.5)`, `render_interfaces(mesh, study, path)`,
`render_stress(mesh, fields, path, strength, safety_factor=2)`,
`optimize_mechanism(domain, nu, MechanismOptions(...))`.

## Lessons that cost something to learn

- **Preserve interfaces only** (screw bosses, bearing seats), never a box round a
  component: a preserved slab is a slab in the result. Every preserved piece
  needs a load path through it, or it comes out loose (the search now fails
  such results and says why).
- **Look closely.** Whole-part renders and SF numbers missed sawn-off faces,
  hard creases at bosses and material over bolt seats; a reviewer found them by
  zooming in. Run `render_interfaces` on every delivered part and turn each
  finding into a metric.
- **Fastener seats are keep-outs**: head + washer + driver, from the face out.
- **Grid-dependent stress**: the voxel check reads SF differently at different
  resolutions; the fine-grid confirm and the tet cross-check exist because of
  it. The cross-check says the voxel solver is ~30% conservative on the yoke.
- **Mechanism framing matters**: "curl as far as possible" gives a leaf spring.
- **Run the eval suite after any reconstruction or boundary change.** Finish
  fixes that looked right on the arm thinned plates and starved support faces on
  other cases; only the suite showed it. Results move a little run to run
  (the search path depends on outcomes), so judge a change on the whole table.
- **A support region holds the nodes inside it** (solid regions with nodes in
  them). Taking a node plane half a cell outside too let the optimizer hang
  the printed hook 0.75 mm off its wall.
- **Watertight and one body is not sound.** The fan bracket (9815ad1) passed
  both with its surface folded back through its arms: hairline slits 0.06-0.2
  mm across that a tet check read as cracks. Design-space faces on the fine
  grid put the field on the level there; marching cubes left zero-area
  triangles, and trimesh's smoothing operator, whose rows then sum under 1,
  threw their vertices 30-150 mm toward the world origin. The on-level guard
  now keeps such nodes off the level, and `_smooth_surface` steps by
  `reconstruct._umbrella`, whose rows always sum to 1
  (tests/test_reconstruct_slits.py finds the folds by section crossings).

## Open problems

Current gaps and their evidence are tracked in docs/VALIDATION.md (what each
model has been checked against), docs/assumption-audit.md (assumptions
still standing) and docs/PHYSICS.md (the physics lens: every default,
boundary condition and check held against what happens to a real part,
with the ranked list of what is still a convenience), and docs/NTOP-PARITY.md holds the whole library against
nTop's block reference. Physical validation of printed parts is the largest one.
