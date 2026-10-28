# loadpath: context for whoever picks this up

State as of 2026-10-28. loadpath is an open-source (Apache-2.0), agent-native
generative design library: topology optimization for printed parts, driven by
constraints, keep-outs and loads, with its own verification and an eval loop.
It lives in Materialize's CAD sidecar today (`cad-runner/loadpath/`, branch
`claude/project-thread-um8gfo`, PR connorwhite-online/materialize#467, Linear
MTR-352) and is meant to move to its own repository.

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
| `spec.py`, `shapes.py`, `materials.py` | JSON study -> Study; shapes (box, cylinder/tube, sphere, mesh, union); 12 materials; `process` (fdm/sls/mjf/sla/volumetric) |
| `domain.py`, `fem.py` | voxel domain, boundary conditions (forces, moments, bearing loads, pins, gravity), hex8 FEA, CG + smoothed-aggregation AMG |
| `simp.py` | SIMP with filter, Heaviside continuation, robust erode/open, OC; live-cell reduced solves; dead-material pruning |
| `mma.py`, `constrained.py` | MMA (from the published method), stress/deflection-constrained mode |
| `mechanism.py` | compliant mechanisms (max output displacement against springs; several input/output pairs) |
| `am.py` | overhang control (Langelaar filter) |
| `reconstruct.py` | field -> printable mesh: cubic upsampling, opening at min member, fade at keep-outs and the design boundary, fillets into preserved geometry, exact booleans |
| `verify.py` | independent voxelization + FEA of the delivered mesh, hotspot submodels, layer-bond check, boundary-cut metric, stress fields |
| `crosscheck.py` | second opinion on a tet mesh: gmsh + native P2 assembly + p-multigrid; CalculiX deck writer |
| `study.py` | the search on safety factor / deflection, fine-grid confirm of the pick, multi-material, parallel |
| `render.py` | clay renders, stress maps, interface close-ups |
| `mcp_server.py`, `__main__.py` | local MCP server and CLI |
| `evals/` | cases with engineering requirements; `python -m loadpath.evals.run` scores them |

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

## Next

1. Validate the printed yoke against its predicted stiffness; feed the result back.
2. Speed up tet meshing (the remaining cross-check time); try CalculiX in the sidecar image (`apt install calculix-ccx`).
3. Stronger, cleaner blends between preserved and organic geometry; crease and fastener-clearance metrics in the evals.
4. Field-driven lattices (TPMS) graded by stress; then conduction (thermal) as a second physics.
5. Nonlinear: large deflection for compliant parts, buckling.
6. Split into its own repository with GRIP as the reference project.
