# What it takes to design any printed part first try

loadpath does well on the parts it has been iterated on. This document asks
the other question: what must be true for a part **nobody has tuned on** to
come out right the first time, and where loadpath is not there yet. The
evidence is the held-out set (`loadpath/evals/heldout`, rules and run history
in its README): common parts written cold from SKILL.md alone and never tuned
against.

## The chain a first-try part has to survive

A part is right first try only if every link below holds. A failure at any
link is a failed part, and the earlier the link the more users it hits,
because every study passes through the early ones.

1. **The problem statement is the real problem.** The agent turns a sentence
   into loads, supports, interfaces and a space. Most first tries die here:
   an envelope that cannot hold a path, a goal no material in that space can
   meet, a load on the wrong face. The library cannot know intent, but it
   can know physics: whether the space is connected, whether the solid block
   would meet the goal, which goal is the binding one, and what number would
   be reachable. It must say that before or instead of a long run.
2. **The boundary is physical.** Real parts are held by bolts, clips, pins
   and friction, not by glued faces. A glued bore gives a stress peak that
   is not there and hides the ones that are (bearing, slip, pull-through).
   What a joint can carry (bolt slip, thread strip, bearing crush) is part
   of whether the part works.
3. **The governing limit is found, not assumed.** Strength, stiffness,
   buckling, layer bond, creep, fatigue, minimum gauge, natural frequency.
   For small consumer parts the governing limit is often minimum printable
   wall or stiffness, not stress. The tool has to find which one binds and
   size to it.
4. **Manufacturing is a cost, not a veto.** Orientation, supports, bridges
   and minimum feature trade against mass and strength. A rule applied as a
   hard veto (never an overhang) can make a part impossible that every
   printer prints daily with supports.
5. **The geometry that comes out is the geometry that was analysed.**
   Small parts on a coarse grid have features one or two cells wide.
   Reconstruction must not thin, split or detach what the solve relied on,
   and the delivered part must be one body unless more were asked for.
6. **Verification is honest and legible.** A pass means a body-fitted
   check at the interfaces, with the reading's own uncertainty. A fail
   names the place, the cause, and the change that fixes it in the agent's
   terms (move this, thicken that, this goal needs 1.3x the depth).
7. **Function beyond load.** Clearance for the tool and the hand, the
   mating part's motion, snap and living-hinge strain, threads, sealing,
   airflow. Common parts are mostly function with some load.
8. **Time.** An agent iterates on the problem statement. If each try takes
   15 minutes, first try is the only try that fits a conversation, and a
   wrong statement should fail in seconds.
9. **Ground truth.** Printed and broken coupons. Until then every margin is
   a model's opinion (see docs/PHYSICS.md, calibration/).

## Where loadpath stood, cold

Run on main a2881fa from a pristine checkout, 2026-11-20, 4 cloud cores,
`maxElements` 40000. "First try" is a clean `check` and a delivered part
that passes.

**First try 3 of 11 as scored then; 2 of 11 in fact. A passing part delivered 4 of 11 (3 in fact). Clean check 7 of 11.**
The drawer pull's pass was false: the tetrahedral check that confirmed it ignored that its load may come from any direction (found later, below).

| Part | Cold result | What went wrong |
|---|---|---|
| drawer pull (PLA) | passed as scored, 11.2 g, 17 min | a false pass: SF 2.56 confirmed pulled straight, 0.76 in the worst direction, for 2 asked |
| tablet stand (PLA) | passes, 47.1 g, 16 min | |
| pipe saddle clamp (PETG) | passes, 6.7 g, 14 min | |
| caster fork (PETG) | passes, 20.6 g, 44 min | `check` gave 8 warnings, all about loadpath's own fastener rings |
| corner bracket for 2020 extrusion (PETG) | no part, SF 2.48 for 2.5 at the 70% ceiling, 16 min | the solid block reads 2.78: a part existed and was not delivered |
| VESA adapter plate (PETG) | two loose straps, four times, 6 min | pieces that do not need each other; the search added material; 8 fastener-ring warnings |
| hinge leaf (PETG) | two pieces, SF 0 | a named build direction banned every overhang; the goal (SF 3 on a 4 mm leaf) is also out of reach, and nothing said so |
| crank lever (PA12) | fails, ran to the ceiling, 5 min | asked to deflect under 1 mm where its solid envelope deflects 2.24: told to "raise the volume fraction" |
| C-clamp (PETG) | fails, SF 1.0 for 2, 4 min | peak at the throat's sharp inside corner; solid block reads 1.38 |
| action-camera mount (PETG) | fails, SF 0.82 for 2.5, 3 min | peak at the slots; solid block reads 0.80; 4 fastener-ring warnings |
| pipe clamp, envelope too short (a wrong problem statement, kept as a test of diagnosis) | two feet, SF 5.2, not diagnosed | the design space was in two pieces and nothing said so |

Six of the seven failures are not the optimizer failing to find a shape.
Four are problem statements that cannot work as written (a goal the
envelope cannot meet, a space in pieces) answered with a long run and
wrong advice; one is a rule applied as a veto; one is a part that existed
and was thrown away. The seventh (the VESA plate) is a real modelling gap.

### After two rounds of fixes (all 11 re-run)

First try 7 of 11 (cold 2), a passing part 6 of 11 (cold 3), clean check
10 of 11 (cold 7).

| Part | Cold | With the fixes |
|---|---|---|
| drawer pull | false pass, 11.2 g, true SF 0.76 | passes, 12.0 g, SF 2.20 in the worst direction, 15 min |
| tablet stand | passes, 47.1 g, 16 min | passes, 29.3 g, 16 min |
| pipe saddle clamp | passes, 6.7 g, 14 min | passes, 5.9 g, 24 min |
| caster fork | passes, 20.6 g, 8 check warnings | passes, 18.9 g, clean check, 51 min |
| corner bracket | no part | passes, 10.3 g, tet-confirmed margin 1.08, 31 min |
| VESA plate | two straps | passes, one piece, 13.5 g, 30 min |
| hinge leaf | two pieces, SF 0 | one piece; told in 3 min the solid block reads SF 0.83 for 3 |
| crank lever | wrong advice | told in 6 min: needs 2.2x the stiffness, or 1.31x the depth, or a 2.5 mm limit |
| C-clamp | wrong advice | told in 5 min: the envelope carries 69% of the load, peak at the throat |
| action-camera mount | wrong advice | told in 3 min: the envelope carries 32% of the load, peak at the slots |
| pipe clamp, short envelope | not diagnosed | `check` says the space is in two pieces before the run |

Read with care. This is one run per part, on fixes made after seeing the
cold results (general causes, but chosen with these failures in view), on a
machine shared with other runs, so the minutes are rough. Eight parts ran
on the first round's code (main ceaa9f7); the drawer pull, tablet stand and
pipe clamp on the second round's (main d4eed7e).

What the re-run itself showed:

- **A false pass, found and fixed.** The drawer pull's load is 150 N from
  any direction. The grid check judges such a load in its worst direction;
  the tetrahedral check that confirms a part, and whose safety factor
  replaces the grid's, solved it as written only. The part delivered on
  the second round's code (10.5 g) was confirmed at SF 2.7 for 2, pulled
  straight (15 MPa, 0.15 mm), where the grid read 0.64 pulled sideways
  (45 MPa, 0.9 mm); each reading held as its own mesh was refined, which
  is what showed they were answering different questions. The cold run's
  11.2 g part was the same (2.56 against 0.76). The tetrahedral check now
  judges the worst direction (0.52 on that part), and the drawer pull
  re-run with it is 12.0 g at SF 2.20. Any study with `"direction": "any"`
  or a cone and a tet-confirmed part was exposed; no other packaged case
  uses one.

- **A mesh fold, found and fixed.** On the first round's code the drawer
  pull came out at 17.8 g. Eight of its twelve designs (five of nine in the
  cold run) were refused as unsound meshes: the surface smoothing put
  vertices next to a keep-out back where they started, eased over two
  rings, and where the smoothing had moved the surface further than its
  edges are long (the pocket at the blind end of a tapped hole) the surface
  folded through itself. The blend now eases over the shift it undoes
  (`reconstruct._ease_over_shift`); all twelve mesh sound. (The 17.8 g
  part was, as it turned out, a real pass and the lighter ones before it
  were not: see the false pass above.)
- **Where the minutes go.** Timed on a shelf bracket, the supports decision
  was a quarter of the run (456 s), most of it a self-supporting search the
  screen already showed could not win; it now runs only where it can
  (78 s, same part). The rest of a 20 to 30 minute run is the ordinary
  search going on to lighter parts once one passes, and on the pipe clamp
  the constrained method at a fine grid (about 460 s a pass, repeated while
  the tet margin is above 1.12). Read off the progress events of two small
  passing parts (15 to 16 min each): the tetrahedral confirm is 35 to 40%
  (60 to 135 s a time, three to five times), the grid verify about 25%
  (27 s, nine times), the optimizer about 15%, the supports decision about
  12% and meshing about 7%. So most of a passing run is checking the
  candidates, not finding them.
- **Impossible goals on small studies** are now read on the solid block
  before the parallel searches start (545 s to 47 s on an impossible shelf
  bracket). The four above were timed before that.
- **Readings still spread.** The corner bracket's coarse screen read SF 1.5
  where the tet check gives 2.69, and the tablet stand's delivered part
  read 19 on the search's coarse grid and 8.0 on tets. That one is settled:
  on equal terms the grid gives 10.2 at 60k cells and tets 8.25 to 8.49, so
  the two model the same problem and the 19 was grid coarseness, on a part
  with a margin of four either way.
- The four parts that are told what their envelope carries are not passes.
  Checked by hand since (gap 7), the limits are real for the studies as
  written.

### A second batch: snap fits, ducts, enclosure parts

Six parts of kinds the first batch did not have, written cold the same
way and run on main c9b09b4: a spring clip for a broom handle and a
cantilever latch hook (snap fits), a pressurized hose elbow and a
part-cooling fan duct (ducts), a junction-box cover and a wall box that
carries a board (enclosure parts, both with a closed skin).

| Part | Cold | After this round's fixes |
|---|---|---|
| tool clip (PETG) | **first try**: 3.1 g, SF 2.71, 18 min | |
| latch hook (PA12, SLS) | passes, 1.2 g, SF 2.14, 16 min; check warns that the foot is held as if glued | |
| hose elbow (PETG, 2 bar) | passes, 16.2 g, SF 5.4, 20 min; two check warnings (wall under two cells; flange held as if glued) | passes, 16.4 g, SF 5.1, 22 min; the wall warning is gone |
| fan duct (PLA) | refused at check, twice: gravity written as a number, then two forces of unknown direction in one case. Third writing passes: 14.1 g, 46 min, with 4 of its 5 designs thrown away (below) | passes, 15.3 g, every design kept; 65 min on the finer grid, stopped at its time limit before the finer check; its 1.2 mm wall is 1.6 cells at four times the budget |
| box cover (PETG, 1.6 mm skin) | passes, 31.3 g, SF 2.36, 46 min; check warns the skin is 1.2 cells | **first try**: 28.5 g, SF 2.75, 48 min, check clean |
| wall box (PLA, 2 mm skin) | passes, 56.0 g, SF 3.65, 40 min; check warns the skin is 1.2 cells | passes, 53.3 g, 57 min, stopped at its time limit before the tet check; the skin is 1.9 cells at four times the budget and check still says so |

Every kind delivered a part that passes, so the pipeline carries over to
parts it was not built on. First try is 1 of 6 cold and 2 of 6 after: what
stands between is the check warnings, each a true statement about the
study or the grid. What the batch showed:

- **A blind tapped hole's floor was read as a blocked hole** (fixed).
  The delivered-part check fails a bore the part closes over, and it
  counted the far end of a thread a screw comes into from the mating part.
  The fan duct lost four of five designs to it and was handed the lightest
  only because its hole's end happened to come out open; the first batch's
  caster fork lost one the same way.
- **A wall the study pins was left under two cells** (fixed). A skin or a
  duct's wall that thin bends too stiffly on the grid; check said so and
  the run went ahead on it. The wall now sets the cell, up to four times
  the cell budget. It costs time on the larger skinned parts.
- **Were the glued passes real?** Yes, for these two. Held under their
  screws only, as check advises (twins `latch-hook-pa12-b`,
  `hose-elbow-petg-b`), both are a clean check and a passing part on that
  second try: the latch hook the same 1.2 g at SF 2.15, the elbow 16.3 g
  at SF 4.1 where the glued flange read 5.1.
- **Held as if glued** (open). Two of six studies held a bolted face over
  its whole area against a load that pulls it off, the mistake check
  already names. The physical statement is a face that pushes and cannot
  pull; a support that does that would make the natural way of writing it
  the right one.
- **A shake in any direction on a mass held by two screws** (open) cannot
  be said as one load on two fastener regions; a case takes one force of
  unknown direction.
- **A snap fit's own load is a deflection** (open, gap 12): the jaw is
  opened 2.5 mm by the handle. A study has forces only, so the clip is
  sized for the yank and not for the snap.
- **Memory**: the fan duct's 1.2 mm wall took 6 to 7 GB in the delivered-part
  check on a 40k-cell study; three runs side by side were killed for it.
- **Time**: 16 to 65 minutes, and on thin-walled parts it is the check of
  each candidate, not the optimizer. The wall box (a 19k-cell study) spends
  58% of its run in five delivered-part checks of about 280 s each: the
  check's own grid goes to 0.52 mm to read a 2 mm skin, some 330k cells,
  solved for three cases of which two are forces of unknown direction
  (four solves each). The hose elbow spends 43% there, the optimizer 27%.
  The supports decision, the results bundle and the viewer then take 7 to
  11 minutes after the part is already chosen (wall box 436 s of 2385,
  box cover 385 s of 2899). A candidate's check costing more than its
  design is the first thing to change for this class of part: every
  candidate has the same skin, and only the one delivered needs the
  finest reading.

### A third batch: mounts, a bearing block, a tube joint

Six more everyday parts written cold on main 2cfb875 (`"batch": 3`): a
608 pillow block, a pegboard hammer holder, a 20 mm tube tee, an
under-desk laptop bracket, a Raspberry Pi wall mount and a filament spool
arm. Five of the six are screwed to a wall or a desk, the commonest way
a printed part is held.

| Part | Cold | After this round's fixes |
|---|---|---|
| pillow block 608 (PETG) | fails: the block meets "0% of the goal, a limit other than strength" | fails honestly: the solid envelope carries 93% of the side load at SF 2.5 (2.33), and the run says so |
| pegboard hammer holder (PLA) | no report: killed at 37 min | **passes**, 12.6 g, SF 2.10, 40 min; check warns, truly, that its horizontal pegs print on supports |
| tube tee 20 mm (PETG) | **first try**, 16.6 g, SF 3.5: a false pass, its rail bore glued all round | check warns the slip bore is held as if glued; twin `-b` holds the rail as a radial seat with friction: **first try**, 59.0 g, SF 2.56, 67 min: a sleeve round the rail, as a tee that cannot bond to its tube has to be |
| laptop bracket (PLA) | fails: "0% of the goal" | **passes**, 6.4 g, SF 2.08, 42 min |
| Pi wall mount (PETG) | fails: "0% of the goal" | fails: 23.4 g, comes out in two pieces (gap 6) |
| spool arm (PLA) | passes, 19.9 g, SF 7.2, 43 min | passes, 13.9 g, SF 8.7, 45 min; its check warning is true (printed along +x, the wall plate hangs off the arm's side) |

What the batch showed:

- **A screwed part held off nothing** (fixed; 3 of 6). The studies held
  the screws' bores and put a seat (`supports[].push`) where the part
  presses on the wall. A bore holds across its axis only, and a seat only
  pushes, so a load that pulls the part off the wall had nothing to react
  it: the solve flew off, the run read SF 0 on the solid block and blamed
  "a limit other than strength". Physically the screw's head holds the
  part along the screw. `fasteners._screw_holds` now adds that hold (a
  seat on the head's seat, pushing along the screw) wherever a screw's
  bore is a support and nothing fixed holds the part along it, and
  `domain._seated_statics` solves the rigid-body statics of the supports
  at check: a load nothing holds is refused in under a second with the
  direction it lifts the part and what to write instead.
- **A hole's roof read as unprintable drawn geometry** (fixed). A
  horizontal hole's top is a bridge across the layers, as every slicer
  prints it; the reach check now takes bridges and short ledges
  (`domain._reach_bridged`) before calling preserved material unreachable.
- **A tube in a slip bore held as if glued** (fixed as a warning plus a
  new hold). The tee passed first try on a rail bore fixed all round, so
  it grew a saddle bonded to the tube. `supports[].push = "radial"` is a
  bore that pushes outward on its wall where it bears (twelve seats round
  the axis, friction carried over), and check names a slip or clearance
  bore held fixed across its axis.
- **The pillow block's limit is real**. Its bearing seat sits 6 mm from
  the envelope's edge; the solid block reads SF 2.33 for 2.5. Before the
  screw fix it read 0 for the wrong reason.
- **A part in two pieces** (open, gap 6). The Pi mount's board loads split
  evenly onto its four standoffs, and each screw carries the two on its
  side: no load needs the left half joined to the right, so the search
  keeps two pieces, and each is judged a part on its own. Physically the
  board itself would tie them, and a part is also handled and assembled
  in one piece; neither is a load today.
- **A seated part's check was slow and heavy** (fixed). The contact solve
  factored the same matrix three times (cases, seat patches, the patches
  cut finer); now once: the pegboard holder's check 59 to 36 s, 4.45 to
  4.03 GB, same numbers. The tube tee's check had 149 right-hand sides
  (cases, and a unit push per seat patch and friction row) on 262k dofs;
  its factor wanted 4.1 GB against the third of free memory one factor
  may take, so all 149 ran as CG: 84 minutes for one check. With 16 or
  more right-hand sides a factor may take 0.6 of the free memory, after
  the tet check the search runs ahead of need (a process of its own, 8 GB
  here) has given way: the same check in 97 s, SF 1.523 against CG's
  1.527. That prefetch process also outlived a run killed for memory; it
  now dies with its search. And a failing run no longer verifies its
  last part a second time for the viewer past its time limit.
- **Open: the optimizer on a seat with friction is slow.** The tee's
  iterations went from 8 s to 130 s as the design sharpened (31k cells),
  and the run ended at 67 min against its 40 min limit, 18 of those after
  the limit in one optimization's last steps.

### Thin walls: cells through the wall, or a shell on the spline surface

The question behind that cost: a pinned wall is a surface with a thickness,
and since the freeform work it can be a spline surface (docs/FREEFORM.md).
A shell model on that surface has unknowns in proportion to the wall's
area, not its volume. Is counting cells through the wall the wrong model?

What was measured, on parts that are not in the held-out set (a ribbed lid
with a 1.5 mm skin, a tray with 2 mm walls, a curved duct with a 1.6 mm
wall, and a cantilever plate against plate theory, which is what a shell
element would return):

| Cells through the wall (cut cells) | 6 | 4 | 3 | 2.5 | 2 | 1.5 |
|---|---|---|---|---|---|---|
| Lid, safety factor | 2.00 | 1.88 | 1.93 | 1.87 | 2.09 | 2.01 |
| Tray, safety factor | 5.31 | 5.44 | 5.13 | 5.75 | 5.38 | 5.99 |
| Duct, safety factor | 1.68 | 1.55 | 1.49 | 1.71 | 1.75 | 2.10 |
| Duct, deflection (mm) | 0.524 | 0.521 | 0.525 | 0.529 | 0.527 | 0.499 |
| Lid, seconds for the solve and read | 82 | 45 | 34 | 17 | 10 | 7 |

- **The four-cell rule answers a problem cut cells do not have.** It was
  written for whole cells, where a wall rounds to a cell (a 2 mm wall read
  2.4 mm). A cut cell integrates the wall's true thickness, and its
  element (incompatible modes) carries linear bending strain in a single
  layer, which is the assumption a shell element makes. Down to two cells
  the readings move by the scatter of the peak reading (6 to 8%), with no
  trend; deflection holds to 1 to 5%, and the plate to 2 to 4% of plate
  theory. Below about 1.5 cells both fall off. On whole cells the same lid
  at three cells read 2.0 where 3.1 was right, so pressure and thermal
  studies, which use whole cells, keep four.
- **What a shell would add is small here, because the limit is not in the
  skin.** On the tray and the duct the governing stress sits where the
  wall meets the floor or the material that mounts it, a three-dimensional
  junction a shell cannot read (the lid's peak was not located); a
  shell skin would still need solid ribs tied to it and a solid reading at
  every junction. Its saving over two cut cells is in unknowns that are
  no longer the cost: at two cells a wall has 8 A / t² cells (area A,
  thickness t), which fits the 400k budget until A / t² passes 50,000. A
  1.2 mm wall reaches that at 720 cm², a housing some 190 mm on a side.
  Past that size a shell on the spline midsurface is the right model and
  cells are not; under it they read the same thing.
- **Where the spline surface does pay today** is under 1.5 cells: there the
  error is in knowing where the wall is inside a cell, and the surface's own
  signed distance gives that exactly where a meshed copy sampled four times
  a cell does not. `switch._thin_skin` already integrates a skin from its
  signed distance in the click, feel and cooling pipelines. Brought into
  `loadpath run` it would let the design grid stay at the user's cell
  budget, in place of refining to half the wall (gap 12b's cost).

Done: a search's own reading of a candidate takes two cells through the
wall where cut cells are on (`verify.SEARCH_SHELL_CELLS`); the part it picks
is still read again at four, or on tetrahedra. On the lid the candidate's
check went from 251k cells and 38 s to 71k and 19 s, reading 2.08 for 2.02,
and the run delivered the same part (15.0 g, tets 2.38 for 2.32). The
saving is half the check, not the fifth the solve alone suggests: reading
stress on the surface and the submodels at hotspots do not shrink with the
grid.

Done: the signed-distance wall in `run` (`loadpath/thinwall.py`). The design
grid no longer refines to two cells through the thinnest pinned wall; it
stays at the study's own budget, and each cell a wall crosses takes the
wall's own stiffness, integrated over 4^3 sub-cells from the wall's signed
distance (the verification's cut cells, the click's thin skin), as a fixed
operator beside the pins' springs. The rest of such a cell is still the
optimizer's, at its share of a cell's stiffness and volume; where a
keep-out touches it, it is air. The grid refines only where a wall is
thinner than one cell. A cantilever plate 1.5 mm thick, off the grid's
planes, against its own reading at three cells:

| Cells through the plate | 1.0 | 1.4 | 1.7 | 2.1 | 3 |
|---|---|---|---|---|---|
| Cut cells, tip deflection / beam theory | 0.81 | 0.83 | 0.85 | 0.89 | 0.91 |
| Whole cells (wall rounds to cells) | 0.91 | 0.29 | 0.53 | 0.33 | 0.91 |

Whole cells are right only where the wall happens to land on a whole number
of cells, and 3.5x too stiff where it does not; cut cells hold within 11%
from one cell through (under one they read stiff: 1.4x at two thirds of a
cell, 1.9x at under half, which check warns of). Two dev parts, not in the
held-out set, the same box (4 cores, 15 GB), before and after:

| Part | Grid | Search | Delivered | Safety factor (tets) |
|---|---|---|---|---|
| Duct, 1.6 mm wall, cantilevered, before | 176k cells (0.8 mm) | 44 min | 26.7 g | 5.4 (5.8) |
| Duct, after | 44k cells (1.3 mm) | 18 min | 22.0 g | 3.8 (4.3) |
| Ribbed lid, 1.5 mm skin, before | 163k cells (0.75 mm) | 33 min | 20.7 g | 2.18 (2.39) |
| Ribbed lid, after | 32k cells (1.3 mm) | 18 min | 20.7 g | 2.05 (2.36) |

The duct's "before" is all wall: its load needed nothing behind the wall,
and the drawn 1.6 mm wall came out 2.5 mm (205 mm^2 of section for 129),
because the run meshed the wall's whole cells as part of the organic body
and the exact wall went in under it. With the wall off the field the
delivered wall is the drawn one. (At first the constrained method kept whole cells on the refined grid; see
below.)

Measured afterwards on the held-out fan duct (-c), never tuned on: first
try, 9.6 g at safety factor 17.6 in 28 minutes, against 15.3 g in 65 before.
Pressure studies and studies with bought stock take the cut walls too.
The constrained method now runs on the cut walls too. It reads a wall
cell's stress as the verification reads a cut cell: the wall's own stress
from the cut cell's kinematics (its bending is carried by the condensed
bubbles) at each solid sub-cell centre, weighted by the wall's share there,
inside the same material-length average; the cell's quarter points carry
only what the design fills of the rest. Read the first way (whole-cell
element at the quarter points, weighted by the wall's share) the ribbed lid's
best passing constrained part was 22.4 g (tet safety factor 2.33 for 2).
Read at the sub-cells, its part loop passed at 19.9 g (tet 2.14), lighter
than the search's 20.7 g on the same walls. On a cantilever plate the
reading at mid-span sits within 0.8-1.25x of beam theory at 1 and 1.4 mm
cells. `LOADPATH_WALL_CONSTRAINED=0` keeps the refined whole-cell grid for
it. Open: thermal studies (their expansion load is per whole cell); a
midsurface shell for housings past the size above.

## Gaps, ranked by how many users they hit

"Hits" is the share of the held-out parts where the gap decided the
outcome on the cold run, plus a judgement of how often a new user meets it.

| # | Gap | Link | Hits | State |
|---|---|---|---|---|
| 1 | A goal the envelope cannot meet is found by running to the 70% ceiling, then answered with "raise the volume fraction" | 1, 6 | 5 of 11, every new user | **Fixed**: a failing search verifies the solid envelope, stops if the goal is out of reach, and says what the block reads, where it peaks and what would reach the goal (`report.bound`) |
| 2 | `check` warns on loadpath's own fastener rings, so a clean check means nothing | 1 | 4 of 11 at check, more during the run; every study with a bolt | **Fixed**: the warning measures the wall the cut leaves, not the share cut |
| 3 | A named build direction forbids every overhang, so parts split or balloon | 4 | 2 of 11, every FDM study that names a direction | **Fixed**: the direction sets the layers; supports are decided by cost unless `overhangs` is declared |
| 4 | The solid block passes but the search stops at 70% and hands back nothing | 6 | 1 of 11 | **Fixed**: the block is delivered when nothing lighter passes, with a note that the envelope sets the mass |
| 5 | A design space in separate pieces runs for minutes and passes or fails without saying so | 1 | 1 of 11, common with keep-outs through thin envelopes | **Fixed**: flagged by `check` before the run |
| 6 | A part in pieces: the search adds material, which never joins pieces that do not need each other | 5 | 2 of 17 (VESA plate; Pi wall mount in the third batch), any plate or frame with paired mounts | **Fixed** on a development part (see "One piece" below): the state that makes a part one piece (handled and installed, held at one support while it locates the others) is held in the optimizer once a run splits on separate supports, or from the start with `onePiece`. Not yet re-run on the held-out set |
| 7 | The safety factor is set at something the study drew, not by the shape: a keep-out's square inside corner, a thin section, a bore held as if glued | 2 | 4 of 11, every part with a clearance cut or a bore | **Partly**: checked by hand, the four limits are real for the studies as written (hinge: 150 N on a 26 mm arm of a 4 mm PETG leaf is 40 MPa nominal against 47; crank: beam theory gives 1.9 mm of the 2.24 read; C-clamp: 10 MPa nominal, 34 at the throat's square corner, and the block reads 1.37, 1.73 and 2.36 with the corner square, at 4 mm and at 8 mm radius). A peak on a keep-out's square edge is now named as the notch to round; followed on the C-clamp (8 mm radius at the throat, nothing else changed), the second try passes: 45.9 g, SF 2.19 for 2, 19 min. A strength shortfall now also says how much more section the weak spot needs (the square root of the shortfall in depth if it bends, the shortfall in area if it is pulled or sheared). Open: how much radius a notch needs; a bore held rigidly all round (docs/PHYSICS.md, open item 1); the 0.86 mm reading length is assumed for PETG and PLA |
| 8 | Small parts take 5 to 17 minutes | 8 | all | **Partly**: an impossible goal is read on the solid block first (under a minute on small studies); the supports decision searches only where it can win. Open: 16 to 50 minutes for a part that passes, most of it the search for lighter parts and the constrained method's passes |
| 8a | A part passed by the confirming check on a load other than the one asked: a force from any direction checked as written | 6 | 1 of 11, every study with a direction-uncertain load | **Fixed**: the tetrahedral check judges the worst direction |
| 8b | A sound design thrown away because its mesh folds next to a keep-out | 5 | 1 of 11 seen (8 of its 12 designs), any part with a blind tapped hole | **Fixed**: the smoothing's blend back from a keep-out eases over the shift it undoes |
| 9 | Joint capacity is not checked: bolt slip, thread strip, insert pull-out, clamp force on a tube | 2 | most bolted parts | **Open** |
| 10 | The governing limit is not named up front (stiffness, strength, minimum wall) | 3 | 2 of 11 | **Partly**: the bound note names it after two failed tries; should be in `check` |
| 11 | Enclosure for a common board from a sentence: placement fails with an error naming the wrong component, and is slow | 1, 7 | every enclosure user | **Open** |
| 12 | Function parts with no route today: snap fits and living hinges by strain, ducts from two ports, threads, gears | 7 | very common requests | **Partly**: six held-out cases written cold (second batch above): snap fits, ducts and skinned enclosure parts pass as force-loaded parts. Open: a prescribed deflection as a load (the snap itself), a duct designed from its ports and flow |
| 12a | A sound design failed because the floor of a blind tapped hole is read as a blocked hole | 5 | 2 of 17, any thread in a thick part | **Fixed** |
| 12b | A skin or duct wall under two cells: warned about, then designed on anyway | 4 | 4 of 6 in the second batch, every skinned part given a cell budget | **Fixed**: the wall sets the cell, up to four times the budget |
| 12c | A bolted face held over its whole area against a load that lifts it | 2 | 2 of 6 in the second batch | **Fixed**: `supports[].push` is a seat that carries pressure only, solved as contact in the optimizer and the check; `check` names it where a face is held as if glued. A shelf bracket on two screws: 5.0 g with the face glued (a hold it does not have), no part on the screw pads alone (the block reads 1.36 for 2), 8.5 g at 2.42 on a seat. With a force of unknown direction the contact is solved in each direction (a seat lets go where the force lifts it, so the axes do not add up), in the space of the patches' gaps from the axis solves and the patches' unit pushes: exact against a seated solve of each direction to 1e-9; the worst for compliance by ascent on its convexity, the worst for stress on a truncated basis of the answers. A bracket on two screws and a seat, 60 N of any direction: 4.4x stiffer in its worst direction than the design for the force as written. The tetrahedral check solves the seat itself too (the patches' springs exact in its CG operator): a plate pried off its seat lifts 15.7 mm on tets, 15.2 on the grid, and CalculiX now takes seats as well (the patches' springs as SPRING1 on *EQUATION means; it reads the native tet check's lift and stress within 0.02%). Friction (`supports[].friction`, Coulomb on each patch: two force rows across it, |tau| <= mu lambda, solved by a semismooth Newton on the Alart-Curnier equations in the gaps' space, the load put on in steps when that fails) and preload (`supports[].preload`, N: the seat set into the part by the interference that gives that force with no load, so a rigid screw's joint stays shut until the load takes the clamp off): a block resting on a table at mu 0.3 holds 29 N across on 100 N and slides at 31 N; a plate clamped between two jaws by 200 N holds about 2 mu times the clamp force; a screw's head seat takes its screw's preload when the study's seats have friction. `check` counts friction too (the octagon round each node's friction disc), so a stand on a desk is no longer refused for being pushed across. A part that only its seat stops (it lies on a table, or rocks on one screw line) is solved too: the seat is held on springs for the solves and they are taken out again exactly; the seat's force is in `report.reactions`; a load that lifts such a part off is refused by name. In the check the patches that bear are cut in four, twice, so a part pressed on its seat near a rim bears on a band there and does not tilt about one coarse patch (far-end lift of a pressed plate 1.15 mm to 0.10 mm). |
| 12e | A screwed part held across its screws and pushed on a seat: nothing holds it off the wall | 1, 2 | 3 of 6 in the third batch, every part screwed to a wall | **Fixed**: a screw whose bore is a support holds the part by its head along the screw (`fasteners._screw_holds`); `check` solves the supports' rigid-body statics and refuses a load nothing holds in under a second |
| 12f | A tube in a slip bore held fixed all round: a false pass, the part bonded to its tube | 2 | 1 of 6 in the third batch, every clamp, tee or socket on a tube | **Fixed**: `supports[].push: "radial"` (the bore pushes outward where it bears, friction carried over); `check` names a slip or clearance bore held fixed across its axis. The tee: 16.6 g glued, 59.0 g held as it is |
| 12g | A seated part's check: 149 right-hand sides on CG, 84 min | 8 | 1 of 6 | **Fixed**: one factor for all of them, given more memory when there are many (97 s) |
| 12d | Parts come out lumpy, beaded along edges and lopsided | 8 | all 12 rendered parts, worst on small ones | **Fixed** in four places, each found on development parts (see "Why parts looked misshapen" below) |
| 13 | Physical ground truth | 9 | all | **Open**: calibration prints (calibration/) |


## One piece: the state the load cases leave out

Every load case is solved with every support held at once. Two halves that
each stand on their own screws carry their own loads to their own
supports, and no case asks one to hold the other, so the compliance design
keeps them apart at any fraction. Physically a part is one piece because of
a state the cases leave out: on its own, handled and installed, it is held
by one interface (the hand, the first screw) and itself locates the others
(the next screw's hole over the wall's, the standoffs under the board's
holes). In that state a part in two pieces drops one.

`onepiece.py` writes that state on the study's own grid: held at the
largest support group and free at the others, each other group pushed by a
unit force along x, y and z. The requirement is that the part be at least
as stiff there as the least the process makes: a round bar of the minimum
member straight from the held group to each other one (L/EA plus two
directions of cantilever bending, 2L^3/3EI). Any part joined by real
material meets it by a wide margin; one in pieces misses it by the void's
stiffness ratio, about a million. The force cancels, since bound and
compliance are read under the same one. The optimizer holds it as a penalty
on log(C/bound) past 1 (its pull is the same relative size whether the part
is in pieces or just over), weighted up while over and kept once under. A
design that meets it is not touched.

When it is on: the search switches it on once an outcome comes out in
pieces that each stand on separate supports (`onepiece.split_supports`),
remakes that fraction from scratch, and from then on neither recalls nor
warm-starts from a design in pieces. `"onePiece": true` turns it on from the
first run. Parts that are one piece on their own never pay for the second
solve.

Measured on a development part (a ceiling plate for two lamps, each hook
nearer its own screw; tests/test_one_piece.py), not on the held-out set:

- Optimizer alone at 12%, pitch 2: two pieces without it, one with it, and
  the joined part is the stiffer (compliance 5745 to 5349). At 8%: 6066 to
  5919.
- Full `loadpath run` (SF 2.5): 7.2 g before, because every fraction under
  0.15 came out in two pieces and the search could go no lighter; 4.7 g
  after, one piece, SF 10.5 (the envelope's bosses set the mass), 17 min
  against 20.

Open: the constrained method (`goal.method: "constrained"`) does not carry
it yet; the Pi wall mount has not been re-run.

## Why parts looked misshapen

The held-out parts rendered on 2026-11-21 were lumpy, beaded and lopsided.
Four causes, none of them in the meshing's smoothing, all found and fixed
on development parts (a rod holder, a shelf bracket, a reducer duct):

1. **Loads on empty cells.** A load on a face that lines a keep-out (a
   rod's saddle, a shaft's seat) was pinned solid only when its region
   touched under 3% of the cells. On a small part it touches more, so the
   load sat on cells the optimizer was free to empty: the rod holder's
   compliance rose 80x as the projection sharpened, and the design it
   ended on was ragged. The cells behind loaded nodes are now pinned one
   minimum member deep at any size. Same grid, same 30% of the envelope:
   SF 2.3 before, 4.5 after.
2. **Symmetric problems, lopsided parts.** Where two layouts tie, round-off
   picks one and every later iteration builds on the pick. A problem that
   is its own mirror image is now held to that plane
   (`domain._found_symmetry`).
3. **A staircase handed to the mesher.** The delivered field is solid or
   void, so its surface followed the cells and the smoothing turned steps
   into ripples. The mesh is now drawn from the filtered field the
   projection was cut from, which says where in each cell the surface is
   (`simp.boundary_field`).
4. **Beads on every edge.** Delivery returned thin members the opening had
   lost by bringing the blueprint back in a ball round each lost cell.
   Edges and sheets at the opening's cut came back in patches: 594 stray
   surface cells on a 40k-cell part, a row along every edge. The blueprint
   now comes back round the whole analysed design, one surface.
5. **Torn rims round screw bosses.** Three causes at one place, the wall of
   a screw head's clearance beside a boss on a small part. Between the
   clearance and the part's wall there was less room than any member, and
   the material there was delivered as a wall tapering to a feather edge:
   material now stays only where a ball of the minimum member's diameter
   fits outside the keep-outs (`reconstruct._member_room`). Beside drawn
   shapes the material is cut exactly on a keep-out's face, but no design
   cell touches a keep-out, so the field's surface wandered across the face:
   the field at the clearance is now carried through the face, so the cut
   is square. And smoothing was undone wherever the surface came near a
   keep-out, which along such a cut left the mesher's own facets: not where
   the surface runs through the keep-out.
6. **Symmetry lost to rounding.** A part an odd number of half cells long
   overhangs the grid by a quarter cell at each face, so the sub-cell
   samples of its edge cells land on the faces themselves. Rounding decided
   each one, a single corner column fell out of the mesh, and the mirror
   planes went with it: a plain 54 x 42 block at 0.8 mm lost both. The
   samples now sit a hair inside the quarter points, so both ends of a
   part are counted the same way.
7. **A drawn thin wall carved with the members.** The room-for-a-member
   step of item 5 also took a drawn wall thinner than a member out of the
   field (a duct's 1.2 mm wall, run to a knife edge between two air
   passages); the wall came back as the blueprint alone and the two
   surfaces crossed, so the part was not delivered. Found by the held-out
   re-run, reproduced on a development elbow. Drawn shapes keep their
   place; only the optimizer's material is held to the member size.
8. **A loaded bore lined thinner than the mesh keeps.** The cells lining
   a small loaded keep-out (a pin's bore) were pinned one cell thick so
   the load rests on material. A one-cell wall is not a member: the mesh
   kept it only where the optimizer had backed it, and a fork's axle
   bores came out open under the axle while the analysis had a full
   ring. The lining is pinned a minimum member thick all round, as the
   larger loaded regions of item 1 already were.
9. **A supported part counted as falling over.** The print check read
   tipping from the part's own feet on the bed; a part that needs
   supports is printed with them, and the columns that reach the bed
   carry it as its feet do. A fork laid on its side, the direction it
   is strongest in, stood on a 5 mm^2 foot beside its supports, counted
   as falling over, and was ranked last: the search then designed it in
   a weaker direction, 2 g heavier and webbed. The centre of mass is
   read against the part's feet and its supports together
   (`comMarginUnsupported` keeps the bare reading).
10. **A false "the mesh lost material" reading.** The comparison of the
   delivered mesh with the optimized field read each cell a quarter cell
   off its centre, toward the low side of every axis, so the whole low
   face of a part that fills its envelope counted as lost. The arm link
   reported 14% of its solid cells lost with 3% of their centres outside
   the mesh. It is read at the cell centres. (The arm link's reported
   1 mm pinholes are not in the run's part: it has 21 openings, none
   under 4 mm across. They come from the surfacing step.)
11. **Members meeting drawn shapes at a step.** (Connor, 2026-11-21:
   "weird transitions galore between keep-out areas and generated
   volumes".) Four causes where a member reaches a drawn boss, pad or
   wall, found on a development lug and the motor mount
   (`tests/test_drawn_transitions.py`):
   - *The field fell off a cliff at the drawn face.* The cells inside a
     drawn shape are pinned or empty, so the member's field ended at the
     face and the fillet blended a member into a wall it could not see
     into. The field is now carried into the drawn shape from the cells
     round it (`reconstruct._drawn_carried`), so a member continues into
     the boss and the smooth union blends where they cross. Only a face no
     generated material touches is cleared to come out exact: where the
     field reaches the face (a skin wrapping a bearing ring thinner than a
     cell, the arm link's shoulder boss) the cells keep what the analysis
     had, or the skin breaks into patches on the exact face.
   - *One fillet size for every member.* The blend was a fixed 3 mm
     whatever met the boss. A shoulder's stress concentration is set by
     the radius over the member's thickness (Peterson: Kt about 1.4 at
     r/d 0.25), so the fillet is now a quarter of the member's local
     thickness, never under the old size (`_stress_sized_k`).
   - *Fillet tails grazing exact faces.* A fillet tangent to a drawn face
     leaves the exact boolean a seam at a grazing angle: ledges, slivers,
     torn strips along a pad's top. The fillet is drawn against the face
     set a little inside, removed only deep inside the drawn shape, and a
     drawn flat face no member lands on is cleared so it comes out as
     drawn (`_clear_free_faces`). Bores and mating faces stay exact.
   - *A pin bearing on air.* A pin's load region runs the length of its
     keep-out, past the end of a short boss, and the clearance past the
     boss was pinned solid as if loaded: a dome grew over the boss top.
     A load on a keep-out that passes through drawn material bears only
     where it meets that material (`domain._region_nodes`), and a loaded
     bore's lining is measured from its own surface (`strap()`).

   Motor mount, same field, old against new meshing: SF 7.43 at 29.45 g
   against 7.26 at 29.9 g; the pads' lips and slivers are gone and their
   circles read exactly. Quick search A/B: 30.2 g SF 3.75 old, 29.0 g
   SF 3.75 new. Bell crank quick search: 20.4 g SF 3.60 new against
   21.9 g SF 3.58 old (and 6 minutes against 20). Fastener bracket, same
   field: the same mass and shape; its reading moves 6.25 to 5.96 from the
   pins bearing on the bosses only, a load placement, not a geometry change. `LOADPATH_DRAWN_CARRY=0` and `LOADPATH_DRAWN_BEARS=0`
   turn the meshing and domain halves off.

12. **Knife edges where a keep-out crosses the body.** (2026-11-24, after
   the bike phone mount's phone side came out with sharp, broken edges at
   its slots and the phone's way in.) A keep-out was cut exactly after
   meshing: wherever a slot, an opening or a way in crossed the generated
   surface the rim was a knife edge, and where it crossed at a slant a
   feather edge thinner than the process prints (then cut off by
   `_drop_unformed`, which left a ragged edge of its own). The field is
   now cut first with a G2 round (`reconstruct._soft_cut`, the pipeline's
   form of `sdf_kit.soft_subtract`), and the exact cut afterwards only
   trims the fit. The round's size is physics, not taste:
   - a free rim (nothing bears on it): a quarter of the member it ends, as
     at any shoulder (Peterson: little gained past r/t 0.25);
   - a rim a load or support acts on (a pin's bore, a screw's seat): an
     edge break of 0.1 t, since a round there is bearing lost: r off each
     end of a bore of length t raises its bearing stress by 1 / (1 - 2r/t);
   - where the body lies along the keep-out's face (a floor under a
     board's room) no round, or the smooth max would shave the wall (the
     wall box lost 6% of its mass so before this was weighted by angle);
   - beside a drawn shape its own edge stands.
   A plate with a free slot and a supported bore
   (`tests/test_drawn_transitions.py`): the slot's rim is a 1.5 mm round
   (was flat to within 0.6 mm of the face, then a drop), the bore's wall
   unchanged over its depth, 1.2% of the volume. The slanted bore of
   `test_feather_edges_at_a_bore_are_cut_to_what_prints` now has no feather
   to drop. `build` gives a component's pocket the same edge break where it
   runs through the body; a drawn `cut` interface stays exact, its faces
   being the spec (the pod's port is checked present to 98%; a round on it
   is drawn with it, as the pod's USB-C mouth is). `LOADPATH_SOFT_CUT=0`
   turns both off.

### Fixed: parts left heavier than they need to be

Read from the search logs of the 2026-11-21 re-run: the search's time, not
the shape. Fixed on development parts (shelf bracket, printed hook, wall
bracket), then measured on the four held-out parts at their 40-minute limit.

- **The time limit never reached the small parts' searches.** A part under
  30k cells is searched on three grids side by side in worker processes,
  and the workers were started without the run's clock. `--time-limit` is
  now a deadline every stage reads (`study._deadline`): the workers get it,
  no part is started that the time left cannot hold (the last part's time
  is the estimate), the constrained stage is not opened with less than two
  search parts' time left, and the build-direction settle is skipped past
  the limit (the part on supports is delivered, which a study naming no
  overhang rule allows). The caster fork's settle had run 15 minutes past
  the limit; now the fork stops at 32 minutes, its last part declined with
  8.8 minutes left and 9.1 needed.
- **The constrained limit crept.** While every constrained run passes, the
  limit was relaxed by the square root of each reading. It is now relaxed
  by the step the measured slope of reading against limit (log-log, over
  the runs so far, never weaker than a quarter power) says lands on 1.06
  (`study._relax_step`). The tool clip read 1.30, 1.21, then 0.99 and 1.09:
  in the band in four runs. Its ten-minute box (a cost bound on a bonus)
  had closed after two; with room on the clock, a stage holding a lighter
  passing part that has not landed in [1, 1.12) now runs on to the clock.
- **A block that barely carries the load was read and thrown away.** The
  solid block read before the pitch starts went unused when in reach, and
  each grid's search read it again after two failed parts. Its margin now
  goes to the searches, and a step up from a failing part reaches at least
  to where the line from that part to the block crosses the target
  (`study._toward_block`): a margin that barely grows with the material
  (a seat sets the peak) is taken at its word.

| Held-out part | Before (main a0608aa) | After (main 5fceed6) |
|---|---|---|
| Tool clip | 4.5 g | 4.0 g, SF 2.19 for 2, 22 min |
| Corner bracket | 12.3 g | 7.8 g, SF 2.53 for 2.5, 30 min |
| Caster fork | 22.2 g, settle 15 min past the limit | 21.0 g, SF 2.68 for 2.5, 32 min |
| Tablet stand | 35.7 g | 32.4 g, 13 min |

## Rule for work under this document

Fix the general cause, never the case. A change is kept only if it follows
from the physics or from the statement of the problem, and it is proved on
the held-out set by re-running it, not by tuning to it. If a held-out case
turns out to be wrong as a problem statement, it stays in the set as a test
of diagnosis.
