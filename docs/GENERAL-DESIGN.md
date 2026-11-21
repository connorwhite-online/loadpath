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
grid. Open: the signed-distance skin in `run`; a midsurface shell for
housings past the size above.

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
| 6 | A part in pieces: the search adds material, which never joins pieces that do not need each other | 5 | 1 of 11, any plate or frame with paired mounts | **Partly**: the search stops and says which load is missing. Open: the load that makes a part one piece (being carried, assembly) is not modelled |
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
| 12c | A bolted face held over its whole area against a load that lifts it | 2 | 2 of 6 in the second batch | **Fixed**: `supports[].push` is a seat that carries pressure only, solved as contact in the optimizer and the check; `check` names it where a face is held as if glued. A shelf bracket on two screws: 5.0 g with the face glued (a hold it does not have), no part on the screw pads alone (the block reads 1.36 for 2), 8.5 g at 2.42 on a seat. Open: with a force of unknown direction; on the tetrahedral check; friction |
| 12d | Parts come out lumpy, beaded along edges and lopsided | 8 | all 12 rendered parts, worst on small ones | **Fixed** in four places, each found on development parts (see "Why parts looked misshapen" below) |
| 13 | Physical ground truth | 9 | all | **Open**: calibration prints (calibration/) |


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

## Rule for work under this document

Fix the general cause, never the case. A change is kept only if it follows
from the physics or from the statement of the problem, and it is proved on
the held-out set by re-running it, not by tuning to it. If a held-out case
turns out to be wrong as a problem statement, it stays in the set as a test
of diagnosis.
