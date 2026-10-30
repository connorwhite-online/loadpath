# loadpath: context for whoever picks this up

State as of 2026-10-29. loadpath is an open-source, agent-native generative
design library: topology optimization for printed parts, driven by
constraints, keep-outs and loads, with its own verification and an eval loop.
It is published at https://github.com/connorwhite-online/loadpath (MIT,
Linear MTR-354). It was built inside Materialize's CAD sidecar
(`cad-runner/loadpath/`, branch `claude/project-thread-um8gfo`, PR
connorwhite-online/materialize#467, Linear MTR-352); new loadpath work happens
in this repository, and the Materialize copy is the sidecar's vendored
version.

## The goal

Parity with Fusion 360's generative design or better, open source, driven by
agents (MCP, CLI, JSON studies). The north star for capability and finish is
**nTop**: implicit (field) modelling, smooth blends, field-driven lattices,
and later thermal / heat-exchanger design. Parts are almost always 3D printed:
optimize for the hobbyist (FDM) without locking out SLS, MJF, resin or
volumetric printing.

## The test subject: GRIP

GRIP (connorwhite-online/grip, PR #1, branch `generative-redesign`, folder
`generative/`) is an open-source 6-axis robot arm. It is the standing test
subject: every library change should be exercised on it, and what goes wrong on
it becomes an eval metric.

- `chain.py` (v0.2) and `v3.py` (v0.3): write a loadpath study per link from the
  joint layout, with loads from URDF statics (1 kg payload, reach pose, 1.5x
  dynamic factor, the joint above at rated torque, a 20 N side push), run them,
  feed masses back. `v3.py remesh` redraws parts from saved density fields
  (`*.field.npz`) after a reconstruction change, in seconds.
- v0.3 architecture (`architecture.py` trade study): NEMA 17 cycloidal drives
  at J1-J3, a zero-free-length spring counterbalancing J2, Feetech bus servos
  (STS3215/3250) at the wrist. Printed links 761 g shipped -> ~562 g.
- `hand.py`: three-finger soft hand, one servo, tendon + balance plate, TPU
  flexure fingers, optional pinned PA nails, generated palm. `finger_mechanism.py`
  tries designing the finger as a compliant mechanism (experimental).
- `sims.py`: stress maps of every link. `generative/print/`: print-ready files.
- Connor is printing the FDM PETG wrist yoke (`print/grip-wrist-yoke-PETG-FDM-v3.stl`)
  to validate it. Servo dimensions are from datasheets (marked VERIFY).

## Architecture (modules)

| Module | Role |
| --- | --- |
| `spec.py`, `shapes.py`, `materials.py` | JSON study -> Study; shapes (box, cylinder/tube, sphere, mesh, union); 16 materials, each tagged with the processes that make it; `process` (fdm/sls/mjf/sla/dlp/volumetric/lpbf) |
| `processcheck.py` | SLS/MJF and SLA/DLP checks on a delivered part (mesh or field): rule cards (`RULES`), walls, enclosed and narrow-exit powder/resin, warp-prone thin panels, resin peel area, suction cups and supports (via printcheck). run_study puts it in the best outcome's `report["process"]` |
| `domain.py`, `fem.py` | voxel domain, boundary conditions (forces, moments, bearing loads, pins, gravity), hex8 FEA, CG + smoothed-aggregation AMG |
| `mg.py`, `gpu.py` | matrix-free geometric multigrid for large grids (SIMP uses it from 100k cells; `LOADPATH_SOLVER=mg|assembled`); `gpu.py` runs its solve with torch on CUDA, Apple Metal (V-cycle on the GPU, float64 CG on the CPU) or the CPU threads (`LOADPATH_DEVICE=auto|cuda|mps|torch|cpu`, extra `gpu`) |
| `simp.py` | SIMP with filter, Heaviside continuation, robust erode/open, OC; live-cell reduced solves; dead-material pruning |
| `mma.py`, `constrained.py` | MMA (from the published method), stress/deflection-constrained mode |
| `mechanism.py` | compliant mechanisms (max output displacement against springs; several input/output pairs) |
| `nonlinear.py` | large deflection (total Lagrangian hex8, neo-Hookean, Newton, force, stroke or tendon (`Tendon`: cable through guides, follows the part) loads, stability check that follows buckled branches), linear buckling, `verify_mechanism` (also `MechanismOptions(verify_nonlinear=True)`) |
| `am.py` | overhang control (Langelaar filter) |
| `manufacture.py` | non-additive constraints: milling (2.5/3/5-axis access, cutter diameter, pocket floors) and casting (draw, 1/2-sided) through one tool/mold access filter on the projected field, circular patterns (tied sectors), maxMember (local volume penalty); `report` measures undercuts and the thickest member on the delivered mesh |
| `reconstruct.py` | field -> printable mesh: cubic upsampling, opening at min member, fade at keep-outs, smooth-min rounding at the design boundary (flush at supports and loads), 3 mm fillets into preserved geometry, exact booleans |
| `verify.py` | independent voxelization + FEA of the delivered mesh, hotspot submodels, layer-bond check, boundary-cut metric, stress fields |
| `crosscheck.py` | second opinion on a tet mesh: gmsh + native P2 assembly + p-multigrid; CalculiX deck writer |
| `study.py` | the search on safety factor / deflection, fine-grid confirm of the pick, multi-material, parallel |
| `render.py` | clay renders, stress maps, interface close-ups |
| `viewer.py`, `viewer.html` | interactive HTML viewer (three.js): stress / safety factor / displacement fields, deformation, section plane, compare, optimization playback, heat exchanger fluids |
| `mcp_server.py`, `__main__.py` | local MCP server and CLI |
| `checks/` | checks on any mesh, no study needed: `check_dfm` (wall thickness, overhangs, trapped voids), `check_networks` (two-fluid isolation), `check_fit` (a board fits its enclosure), `fea_probe` (relative hotspot finder), `opening` |
| `sdf_kit.py`, `exchanger.py` | field-modelling toolkit (primitives, TPMS, smooth booleans, exact mesh booleans) and the rectangular dual-fluid `exchanger_core` built from it; `heatx.py` is the newer, study-driven exchanger. Old `loadpath.checks.sdf_kit` / `.exchanger` imports still resolve |
| `fields.py` | nTop-style field graphs (JSON or Python): named solid and scalar nodes, every numeric parameter can be a field (variable shell, variable fillet, graded TPMS cell/wall/density), `result` nodes from verify stress, exact mesh distance, smooth meshing (marching cubes + relax + projection). `loadpath field`, schema `field`, MCP `loadpath_field`; guide in docs/fields.md |
| `narrowband.py` | field meshing that samples only near the surface: blocks and leaves tested against a Lipschitz bound, the rest skipped; same mesh as dense marching cubes, from 20-50% of the samples on solid parts, so the cell budget counts sampled cells, not the box. `fields.field_mesh` uses it by default (falls back to every block if the mesh is not watertight); `lattice_infill` meshes the exact TPMS through it (`exact_surface`), so walls are the specified thickness, not the grid's |
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
python -m loadpath check study.json             # validate a study
python -m loadpath run study.json --out out/    # optimize
python -m loadpath.evals.run --jobs auto --out /tmp/evals   # full suite; auto = as many cases as fit in memory
python -m loadpath.evals.run --tier coarse --corpus 6 --baseline loadpath/evals/baselines/coarse.json \
    --gate regressions --out /tmp/coarse                 # smoke tier for every push, ~7 min on 4 cores
python tests/test_evals.py                      # pareto, tiers, corpus, critic
python -m loadpath mcp                          # local MCP server
```

GRIP (clone connorwhite-online/grip, branch `generative-redesign`), with
loadpath on PYTHONPATH:

```bash
python generative/v3.py run --links forearm,wrist_yoke,wrist_link --jobs 3 --max-elements 120000
python generative/v3.py remesh --links wrist_yoke   # after a reconstruct.py change, seconds
python generative/sims.py                            # stress maps of every link
python generative/hand.py fingers|studies|run        # the hand
```

Python API worth knowing: `run_study`, `verify_part(..., fields=True)`,
`crosscheck(study, mesh, size=1.5)`, `render_interfaces(mesh, study, path)`,
`render_stress(mesh, fields, path, strength, safety_factor=2)`,
`optimize_mechanism(domain, nu, MechanismOptions(...))`.

## Today's record (2026-10-28), in order

1. GRIP v0.3 links regenerated around bus servos and the J2 spring; stress maps.
2. Connor's QA of the wrist yoke: sawn-off faces, hard creases at bosses,
   material over bolt seats. Fixed in reconstruction (boundary rounding,
   fillets) and in the study (screw seats as keep-outs); interface close-ups
   and `boundaryCutArea` added so these are caught without a human.
3. Dead-end "tendrils" pruned; preserved boxes replaced by screw bosses (the
   yoke went 41 -> 19-24 g).
4. Three-finger soft hand (hand.py); compliant-mechanism optimizer; the fin-ray
   finger experiment (finds some wrap, not yet better than the hinged finger:
   parked).
5. Manufacturing `process`; FDM PETG yoke designed per build direction (the
   direction changed mass by half: 25.7 / 30.7 / 39.4 g).
6. Independent cross-check: 644 -> ~100 s on the yoke with the same answer.
   Decimating the surface first was 2x faster but under-read the peak by a
   third (rejected); hotspot-adaptive refinement barely helped (off).
7. Eval round 9 caught regressions from step 2; fixed. See the table below.

## Lessons that cost something to learn

- **Preserve interfaces only** (screw bosses, bearing seats), never a box round a
  component: a preserved slab is a slab in the result. Every preserved piece
  needs a load path through it, or it comes out loose (the search now fails
  such results and says why).
- **Look closely.** Whole-part renders and SF numbers missed sawn-off faces,
  hard creases at bosses and material over bolt seats; Connor found them by
  zooming in. Run `render_interfaces` on every delivered part and turn each
  finding into a metric.
- **Fastener seats are keep-outs**: head + washer + driver, from the face out.
- **Grid-dependent stress**: the voxel check reads SF differently at different
  resolutions; the fine-grid confirm and the tet cross-check exist because of
  it. The cross-check says the voxel solver is ~30% conservative on the yoke.
- **Mechanism framing matters**: "curl as far as possible" gives a leaf spring.
- **Linear only so far**: no large deflection (TPU fingers), contact, buckling.
- **Run the eval suite after any reconstruction or boundary change.** Finish
  fixes that looked right on GRIP thinned plates and starved support faces on
  other cases; only the suite showed it. Results move a little run to run
  (the search path depends on outcomes), so judge a change on the whole table.
- **A support region holds the nodes inside it** (solid regions with nodes in
  them). Taking a node plane half a cell outside too let the optimizer hang
  the printed hook 0.75 mm off its wall.

## Eval status (round 9f, 2026-10-28): 6 of 8 pass

| Case | Pass | Mass g | SF | p5 member mm | Note |
| --- | --- | --- | --- | --- | --- |
| arm-link-driven | yes | 48.1 | 11.9 | 3.41 | |
| arm-link | yes | 57.8 | 13.5 | 3.44 | |
| bell-crank | yes | 24.0 | 3.10 | 4.19 | |
| grip-upper-arm | yes | 209.8 | 4.65 | 3.31 | |
| mbb-beam | yes | 23.8 | 1.64 | 2.56 | 64% lighter (was 49%) |
| motor-mount | yes | 25.8 | 2.73 | 3.41 | |
| printed-hook | NO | 33.6 | 3.49 | 1.46 | thin wall plate |
| wall-bracket | NO | 32.8 | 4.29 | 1.95 | thin seat |

The two failures are one open problem: the grid is centered (for exact
symmetry), so on a support face that is not a symmetry plane the first cell
layer overhangs the design space by up to half a cell. The optimizer counts
that overhang as material; the delivered part is cut at the real face, so the
pad against the wall is thinner than designed. Aligning the grid to the face
fixed the geometry but moved every case's search path (3 of 8 lost). Giving
overhanging cells only the stiffness of the part of them that exists (round
9i) left both failures unchanged and made the arm link 4.5 g heavier, so it
was reverted too. Both are thin pads at a support face where the measured
p5 member is exactly one or less than one cell (1.46, 1.95 mm at a 1.95 mm
pitch): the next thing to try is a finer pitch near supports, or the
reconstruction extruding the pad to the design face it bears on.

## Next

1. Validate the printed yoke against its predicted stiffness; feed the result back.
2. Speed up tet meshing (the remaining cross-check time); try CalculiX in a server image (`apt install calculix-ccx`).
3. Stronger, cleaner blends between preserved and organic geometry; crease and fastener-clearance metrics in the evals.
4. Field-driven lattices (TPMS) graded by stress; then conduction (thermal) as a second physics.
5. Nonlinear: large deflection for compliant parts, buckling.
6. Point GRIP's `generative/` scripts at the published package (`pip install`
   from the loadpath repository) instead of a Materialize checkout.
