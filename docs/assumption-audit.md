# Assumption audit

State as of 2026-10-30 (main ad38801). Every stage of loadpath was built on
a choice that was reasonable at the time and then stopped being questioned.
This page writes those choices down, asks what a 10x better answer would
look like, and ranks the answers by payoff against effort. It is meant to be
re-run: the checklist at the top is the reusable part, the audit below is one
pass of it.

## The checklist

Apply it to any stage, module or workflow before deciding it is done. For
each question, write the assumption in one line, then the best alternative
you can think of, then a rough payoff (what number moves, by how much) and
effort (S: a day, M: a thread, L: several threads).

1. **Hardware.** What machine does this assume (4 CPU cores, float64, one
   process, RAM)? What if it had a GPU, 100 cores, or 10x the memory? What if
   it had to run on a laptop in 10 seconds?
2. **Algorithm.** Is this the method from the first paper we read? What do
   the best published codes (and commercial tools) do now? Is a solve
   converged tighter than the decision it feeds needs?
3. **Representation.** Are we storing the thing in the form the next stage
   needs (voxels vs fields vs meshes vs B-rep)? How much is lost converting
   between forms, and could the conversion disappear?
4. **Proxy vs goal.** Are we optimizing the thing the user cares about
   (print time, cost, measured strength) or a stand-in (compliance, mass,
   voxel stress)? How big is the gap, and is it measured?
5. **Conservatism.** Where do we add margin "to be safe"? Margin on a
   printed part is mass and print time. Is the margin measured or guessed?
6. **Inputs.** Who supplies the inputs (loads, design space, material), how
   do they know, and what happens when they are wrong by 30%?
7. **Ground truth.** What would prove this stage right or wrong? A
   reference solver, a published benchmark, a printed and broken part? Do we
   have it?
8. **Human in the loop.** Where does Connor (or an agent) still have to look,
   zoom, or judge by hand? Can that judgement become a metric?
9. **Feedback speed.** How long from a change to knowing whether it helped?
   What would make that 10x shorter?
10. **Coverage.** What cases is this tuned on? Would it hold on parts we have
    never seen (not GRIP-like)?

A finding is worth a thread when the payoff is a number we report (mass,
SF, time, pass rate) moving by 20% or more, or a whole class of parts
becoming possible.

## Ranked findings

Payoff: H (a headline number moves 20%+ or a new class of parts), M (10-20%
or a clear workflow win), L. Effort: S/M/L as above. "Covered" names the
existing workstream that already owns it.

| # | Stage | Assumption | 10x alternative | Payoff | Effort | Covered |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | Verification, search | Safety factor is read from voxel stress, which the tet cross-check measured 24-49% conservative | Make the pick on accurate stress: tet/CalculiX confirm of the final pick, or a per-part voxel-to-tet correction, or body-fitted stress recovery at hotspots | H: 15-30% less mass at the same real SF on every part | M | done: `goal.confirm` tet (see stage notes) |
| 2 | Objective | Mass is the cost | Print time, filament and support from a real slicer (Orca/Prusa CLI) as the cost the search minimizes and reports | H for FDM users: time, not grams, is what they pay | M | done: slicer.py, `goal.minimize` ([slicer.md](slicer.md)) |
| 3 | Export, lattices | Infill and grading are geometry we mesh | FDM: emit 3MF with per-region slicer modifiers (infill density, walls) driven by the stress field, so the slicer grades it; meshed TPMS only where it pays | H: graded FDM parts with no million-triangle meshes, slicer-native | M | done: infill.py, `slicer.grade` ([slicer.md](slicer.md)) |
| 4 | Ground truth | Material cards are datasheet numbers; no part has been tested | Calibration loop: coupon generator + a tiny measurement format; printed GRIP yoke stiffness and break load feed material cards and the conservatism factor | H for trust; sets #1's correction | M (needs prints) | partly (Connor printing the yoke) |
| 5 | Human QA | Defects are found by Connor zooming into renders | A vision-model critic on `render_interfaces` close-ups that files findings as eval metrics (creases, sawn faces, material over seats) | M-H: catches the class of bug numbers missed | S-M | done: loadpath/critic.py, `--critic` |
| 6 | Evals | One ~2 h suite on 14 GRIP-like cases, pass/fail | Tiered: 5-min coarse smoke on every push, full suite nightly; Pareto (mass vs SF) score not pass/fail; a procedurally generated case corpus to stop tuning to GRIP | M-H: 20x faster feedback, less overfitting | M | done: `--tier coarse` (~7 min), pareto.py, corpus.py |
| 7 | Inputs (loads) | Loads are exact vectors the agent writes by hand | Loads derived from the mechanism (URDF / joint torques / pose sweep, as GRIP's chain.py does ad hoc) and robust to direction and magnitude scatter (worst case over a cone) | H for robot parts: real loads are a pose envelope, not one vector | M | yes: assembly.py (`loadpath assembly`); GRIP's committed links drop from SF 2.7-4.2 to 0.6-2.1 under swept loads (examples/grip/README.md) |
| 8 | Inputs (design space) | The user draws the design space and keep-outs | Derive them: assembly envelope minus neighbours' swept volumes over the joint's range of motion, fastener access paths | M-H: removes the hardest manual step for agents | M | partly: assembly.swept_keepouts (neighbours' swept volumes); envelope and fastener access still by hand |
| 9 | Optimizer | Gradient descent from a uniform start gives one topology | Several distinct topologies (multi-start, diversity penalty, different filter radii) for the agent to choose from, as Fusion does | M | S-M | partly (warm-start thread) |
| 10 | Solver tolerance | Every inner solve converges to the same rtol | Inexact solves early (tolerance tied to the projection beta and the design change), tight only at the end | M: 1.5-3x optimizer time | S | fold into GPU / scale threads |
| 11 | Hardware | CPU, float64 assembled AMG, 4 cores | GPU matrix-free, mixed precision | H | M-L | yes: GPU offload, scale to millions |
| 12 | Representation | Voxel density -> isosurface -> mesh; delivered part differs from the optimized one | Optimize the implicit field itself | H | L | yes: smooth implicit fields, field-driven design |
| 13 | Physics | Linear static, small strain | Nonlinear, buckling (done); modal / dynamic for robot links (stiffness at speed, resonance with servo control) | M for GRIP | M | new, lower priority |
| 14 | Heat exchangers | Correlations for h and dp | LBM flow | H | L | yes: flow sim |
| 15 | Search | Outer search re-optimizes per volume fraction | MMA mass-minimization with stress constraints straight to the target (exists in constrained.py) as the default, outer search only to close the verified gap | M: fewer full optimizations | S-M | partly (outcome search thread) |
| 16 | Orientation | Build direction chosen by coarse re-runs | Faster pick | M | S | yes: auto-orientation speed |
| 17 | Export | STL / 3MF mesh is the deliverable | Editable output: STEP via B-rep fitting, or the field graph itself as the editable source (nTop style) | M for CAD users | L | partly (fields) |
| 18 | Dev process | Threads compare runs on different machines | One pinned reference machine per comparison (already a rule), plus stored eval artifacts so a regression is a diff, not a re-run | L-M | S | partly |

## Stage notes

### Inputs and study authoring (spec.py, schema.py)

The study assumes the author knows the loads, the design space and the
material. For GRIP that knowledge lives in the GRIP repo (`chain.py`,
`v3.py` derive loads from URDF statics at one reach pose with a 1.5x
factor). A robot link actually sees an envelope of poses, accelerations and
payload offsets, and the worst case for one member is rarely the worst case
for another. Loads from a pose sweep, collapsed into a handful of governing
cases, is both more correct and less work for the agent (#7). The design
space is drawn by hand as boxes and cylinders; in an assembly it is the
space nothing else occupies or sweeps through (#8).

### Domain and voxelization (domain.py, verify.voxelize, voxels.py)

Uniform cubic cells everywhere. The thin-pad failures on the hook and
bracket were one cell thick at a support face; a finer pitch near supports
and loads (octree or a two-level grid) attacks that directly and is cheaper
than refining everywhere. Two voxelizers exist on purpose until a
re-baseline (CONTEXT.md).

### FEA (fem.py, mg.py)

Hex8 on the voxel grid, CG with smoothed-aggregation AMG, or matrix-free
geometric multigrid for large grids. GPU and scale are owned (#11). The
remaining assumption is that every solve inside the optimization loop needs
the same tolerance (#10): early iterations at low projection sharpness
change the design by tens of percent per step, so a 1e-2 residual is enough
there.

### Optimization formulation (simp.py, constrained.py, mma.py)

Compliance with a volume constraint is a stand-in for "lightest part that
meets SF and deflection". constrained.py already solves the real problem
with MMA; the outer search in study.py still re-runs whole optimizations
per volume fraction (#15). Both start from one place and return one
topology (#9). The objective is mass; for an FDM hobbyist, print time and
support removal are the cost (#2).

### Constraints and manufacturing (am.py, manufacture.py, processcheck.py)

Overhang control uses a fixed 45 degree stencil. Real printers bridge short
spans, print steeper overhangs with fans, and need supports that a slicer
decides, not a stencil. A slicer in the loop (#2) gives support volume and
print time as measured numbers rather than modelled ones.

### Reconstruction (reconstruct.py)

The delivered surface is an isosurface smoothed after the fact, so the part
verified is not the part optimized and verify has to catch the difference.
Owned by the implicit-field threads (#12). Crease and boundary-cut metrics
came from Connor finding defects by eye; that loop is #5.

### Verification and confirm (verify.py, crosscheck.py, study.py)

The search picks the lightest outcome whose voxel safety factor meets the
target, confirmed on a finer voxel grid. The tet cross-check (fTetWild +
P2 or CalculiX) reads peaks 24-49% lower on motor-mount and bell-crank, and
the voxel stress moves with the grid (turret SF 2.41 at 150k cells, 1.92 at
120k). So the margin the search holds is part real, part grid artefact, and
the artefact is paid for in mass on every part (#1). The cross-check runs in
~1-2 minutes on 4 cores, cheap enough to confirm the final pick, and the
GPU thread will make it cheaper.

Done (2026-10-30): with the cross-check installed the search's pick is
confirmed on tets, every voxel safety factor is scaled by that pick's
tet/voxel ratio, and the search steps down until the tet reading meets the
target (study.py, `goal.confirm`). On 7 strength-bound evals, against main
on the same cloud box: MBB 26.4 -> 13.8 g (tet SF 1.70 for 1.5), wrist yoke
25.3 -> 19.8 g (2.43 for 2), wall bracket 57.2 -> 45.8 g (4.06 for 2.5),
fastener bracket 47.3 -> 38.8 g (4.92 for 2.5), bell crank 24.8 -> 24.3 g,
hook 31.3 g both, turret 148.8 g both: 361 -> 323 g in total (-11%), -16%
per case on average. What was learned:

- Voxel is not uniformly conservative. The tet/voxel ratio was 1.03 (yoke)
  to 2.1 (fastener bracket), and the turret went the other way: SF 1.21 on
  tets against 2.54, its peak on the wall of a J1 bolt's driver access 1.3
  mm from the held bore. Three steps up to 0.24 left that wall, and the
  mass, unchanged, so a pick that fails on tets is confirmed on voxels and
  the tet reading goes out as a warning instead of a step up.
- The tet peak needs a contact band of two elements (and no less than the
  voxel one). At the 1.3 mm voxel band the MBB beam's peak was the edge of
  its rigid support and read SF 1.62-2.07 on three meshes of one part
  (fTetWild is not deterministic, even single-threaded); at 3 mm, 3.00-3.05.
  A mesh twice as fine moved the peak under 8%, inside the 1.15 allowance.
- The ratio moves with the design (1.39-1.81 across MBB fractions), so
  every step down is confirmed on tets, not trusted to the ratio. Several
  cases stop at a voxel cliff (wall bracket: voxel SF 2.65 at 7.8%, 0.8 at
  7.1%) with the tet reading still well over target: the next gains are
  confirming those neighbours on tets directly, and cut-cell boundary
  stiffness so the voxel reading is closer to begin with.
- The voxel reading also moves with the cell budget, and most on thin
  sections. Tuning the arm link's field graph (tune.py, 2026-10-30), the
  start part read 0.500 mm deflection on 30k cells and 0.487 mm on 60k, while
  a thinner web candidate read 0.485 and 0.623 (+28%); safety factors moved
  5-20% the same way. A search on the coarse grid alone walks into the
  sections the coarse grid flatters. tune.py now scales coarse readings by
  the start's coarse/fine ratio and re-checks its winners on twice the cells;
  the same drift is likely in any study outcome that sits on a limit with
  members only two or three cells thick.

### Thermal and flow (thermal.py, heatx_thermal.py)

Correlations for the exchanger, conduction-only topology optimization.
Flow is owned (#14). Intercooler LBM cell data is the validation set.

### Printability and export (checks/, threemf.py)

STL and 3MF meshes are the output. For FDM, the slicer already knows how to
grade infill and add walls; driving its modifiers from the stress field
(#3) gets a graded part with none of the meshing cost of a lattice, and
prints faster than a TPMS of the same stiffness on most FDM machines. For
CAD users the missing output is something editable (#17).

### Agent and MCP workflow (mcp_server.py, tasks.py, SKILL.md)

The agent writes a JSON study, starts a job, waits, reads a report. It never
sees the part, so it cannot notice what Connor notices when he zooms in
(#5). It also cannot cheaply ask "what if": every question is a full study.
A coarse preview mode (seconds, one fixed volume fraction, no confirm)
would let an agent iterate on the study itself before paying for the real
run; this pairs with the tiered evals (#6).

### Evals and development process (evals/, tests/)

The suite is 14 cases, all GRIP-like by design, scored pass/fail, about two
hours on the cloud box at `--jobs 1`. A change is judged a few times a day
at best, and a fix can be tuned to these cases without anyone noticing (#6).
Pass/fail also hides progress: a case that goes from 2% to 15% lighter at
the same SF scores the same. Nothing in the suite is a physical measurement
(#4).

## Top new workstreams

Not covered by any running thread, in order:

1. **Stop paying mass for voxel conservatism** (#1). Confirm the final pick
   on the tet cross-check (or calibrate a per-part voxel/tet factor at the
   hotspots) and let the search step down to the real SF. Success: eval
   masses down 15%+ with every case's tet SF still at target.
2. **Slicer in the loop for FDM** (#2, #3). Run Orca or Prusa slicer
   headless on each outcome for print time, filament and support; report it,
   offer it as the objective; export stress-graded 3MF modifiers instead of
   meshed infill for FDM.
3. **Loads and design space from the mechanism** (#7, #8). Pose sweep from
   URDF to governing load cases with direction scatter; design space and
   keep-outs from the assembly and swept volumes. Move the GRIP-specific
   logic into loadpath.
4. **A vision critic and faster evals** (#5, #6). Model review of interface
   close-ups as a metric; a 5-minute coarse tier; Pareto scoring; a
   generated case corpus.
5. **Physical calibration loop** (#4). Coupon generator, measurement
   format, ingest into material cards and #1's correction. Needs Connor's
   printer, so it should start with the yoke he is already printing.
   Built (calibration.py, docs/CALIBRATION.md, examples/grip/yoke_test.py);
   waiting on the prints and measurements.
