# Held-out parts: the cold benchmark

Common parts nobody has tuned loadpath on: a corner bracket, a hinge leaf, a
drawer pull, a VESA plate, a caster fork, a pipe clamp, a crank lever, a
tablet stand, a C-clamp, an action-camera mount; and a second batch
(`"batch": 2`): a tool clip, a latch hook, a hose elbow, a fan duct, a
junction-box cover, a wall box; and a third (`"batch": 3`): a 608 pillow
block, a pegboard hammer holder, a 20 mm tube tee, an under-desk laptop
bracket, a Raspberry Pi wall mount, a spool arm; and a fourth (`"batch": 4`):
a ceiling plant hook, a handwheel on a D-shaft, a bed-leg riser, a strap
joiner, a rope pulley on a 608 bearing, a 13 mm box-end wrench. Each study was written once,
cold, from SKILL.md alone, the way an agent meeting loadpath for the first
time would write it, and run without looking at loadpath's code.

The question the set answers: **does a part loadpath has never seen come out
right the first time?** The eval cases (`loadpath/evals/cases`) cannot answer
it, because the library has been iterated against them.

## Rules

1. **Never tune against this set.** No case-specific constants, no threshold
   picked so a case here passes. A fix is made for a cause that follows from
   physics or from the problem statement, checked on other studies, and only
   then measured here.
2. **Never edit a case to make it pass.** If a case turns out to be a wrong
   problem statement (a goal no material in that space can meet, an envelope
   that cannot hold a path), it stays: an agent will write the same mistake,
   and the test is whether loadpath says so quickly and usefully. Such cases
   carry a `note` and `pass.diagnosis`. A corrected twin may be added beside
   it under a new id.
3. **Add, do not replace.** New cases are written cold and their first run
   is recorded below before anything is changed for them.
4. **First try means first try.** `firstTry` is: `loadpath check` is clean
   (no warnings) and `loadpath run` delivers a part that passes, from the
   study as written.

## Running

```
python -m loadpath.evals.heldout list
python -m loadpath.evals.heldout run --out runs/NAME --jobs 2 [ids...]
python -m loadpath.evals.heldout table runs/NAME
```

## Run history

| Date | Code | First try | Passing part | Clean check | Notes |
|---|---|---|---|---|---|
| 2026-11-20 | main a2881fa (cold, pristine checkout) | 3 of 11 | 4 of 11 | 7 of 11 | The baseline. Per-part results in docs/GENERAL-DESIGN.md. In fact 2 and 3 of 11: the drawer pull's pass was false (its any-direction load was confirmed pulled straight) |
| 2026-11-20 | first fixes (pull request 14, main ceaa9f7) | 6 of 11 | 5 of 11 | 10 of 11 | Corner bracket, VESA plate pass; caster fork check clean; hinge, crank, C-clamp, action-cam mount told what their envelope carries in 3 to 6 min; short pipe clamp flagged at check. Drawer pull regressed to 17.8 g (folded meshes refused); tablet stand's run was cut off by the machine |
| 2026-11-20 | second fixes (pull request 16, main d4eed7e): the three not settled above re-run | 7 of 11 | 6 of 11 | 10 of 11 | Drawer pull 10.5 g, tablet stand 29.3 g, pipe clamp 5.9 g; the other eight carried from the row above |
| 2026-11-20 | tet check judges the worst direction (pull request 19): drawer pull re-run | 7 of 11 | 6 of 11 | 10 of 11 | Drawer pull 12.0 g at SF 2.20 in the worst direction. Its 10.5 g in the row above was a false pass (0.64); its 17.8 g in the row before was a real one |
| 2026-11-20 | second batch of six (snap fits, ducts, enclosure parts), cold on main c9b09b4 | 1 of 6 | 5 of 6 | 1 of 6 | Tool clip first try; latch hook, hose elbow, box cover, wall box pass with a check warning; fan duct refused at check (twins -b refused, -c passes 14.1 g) |
| 2026-11-20 | wall sets the cell; a tap's floor is not a closed bore: elbow, cover, wall box re-run | 2 of 6 | 5 of 6 | 2 of 6 | Box cover first try, 28.5 g; elbow's wall warning gone; wall box still 1.9 cells and 57 min |
| 2026-11-21 | main c95c289: fan duct -c re-run; twins held under their screws (latch hook -b, hose elbow -b) | | | | Fan duct 15.3 g with every design kept, 65 min; both twins clean check and pass (1.2 g SF 2.15; 16.3 g SF 4.1): the glued passes were real, the elbow's margin 25% high |
| 2026-11-21 | clean-parts fixes (main a0608aa): the twelve parts that are meant to pass, re-run and re-rendered | 11 of 12 | 11 of 12 | 11 of 12 | Fixed on development parts only (docs/GENERAL-DESIGN.md, "Why parts looked misshapen"). Lighter or cleaner: pipe clamp 3.9 g (was 5.9), wall box 47.8 g (53.3), box cover 26.1 g and symmetric (28.5), hose elbow without its stray strut. Same: latch hook, drawer pull, VESA plate. Heavier: tool clip 4.5 g (3.1), corner bracket 12.3 g (10.3), tablet stand 35.7 g (29.3), caster fork 22.2 g (18.9). Fan duct: the part is sound again (14.9 g) but its last confirmation ran a 15 GB machine out of memory, so no report |
| 2026-11-21 | thin walls as cut cells on the study's own grid (thinwall.py, main 4a7dd4d): fan duct -c re-run, alone | | | | Fan duct first try, 9.6 g, SF 17.6, 28 min (was 15.3 g in 65 min, or 14.9 g and out of memory): 27k cells at 1.2 mm instead of the finer grid two cells through its 1.2 mm wall, and the wall delivered at its drawn thickness. Measured after the change, which was made and checked on development parts |
| 2026-11-21 | main cb5d90e (thin walls as cut cells, a2f7c72): fan duct -c re-run | 12 of 12 | 12 of 12 | 12 of 12 | Fan duct 9.6 g at SF 17 in 15 min (was 14.9 g with a dimpled skin and no report): the drawn wall comes out smooth at its drawn thickness. The other eleven carried from the row above |
| 2026-11-21 | main 9144e49: box cover re-run | 12 of 12 | 12 of 12 | 12 of 12 | Box cover 25.6 g at SF 2.11 (was 26.1 g): the skin's underside is flat, without the creases of the whole-cell wall. Search still bracketing at the 40 min limit |
| 2026-11-21 | main 2423249-5fceed6: search on the clock (deadline in every stage, constrained limit stepped by its response, solid block brackets the search): the four parts left heavy re-run, one at a time | | | | Tool clip 4.0 g (was 4.5), corner bracket 7.8 g (12.3), caster fork 21.0 g in 32 min (22.2 g, 55 min), tablet stand 32.4 g (35.7); all four first try. Fixed on development parts (docs/GENERAL-DESIGN.md, "Fixed: parts left heavier") |
| 2026-11-24 | third batch of six (mounts screwed to a wall, a bearing block, a tube joint), cold on main 2cfb875 | 1 of 6 | 2 of 6 | 3 of 6 | Tube tee first try, 16.6 g, but a false pass (its rail bore glued all round); spool arm passes 19.9 g; pillow block, laptop bracket, Pi mount read "0% of the goal" (screws held across only, nothing held them off the wall); pegboard holder killed at 37 min |
| 2026-11-24 | screws hold by the head, hole roofs bridge, radial seats, seated checks factored once (main ebfc654-cf41c04), one at a time | 2 of 6 | 4 of 6 | 4 of 6 | Laptop bracket first try 6.4 g; spool arm 13.9 g (was 19.9); pegboard holder 12.6 g; twin `tube-tee-20mm-petg-b` (rail as a radial seat) first try 59.0 g, SF 2.56, 67 min. Pillow block fails honestly (block reads 2.33 for 2.5); Pi mount comes out in two pieces (gap 6). The spool arm's and pegboard holder's check warnings are true (supports under drawn features in their build direction) |
| 2026-11-24 | main 0def539: a part held in one piece (onepiece.py): Pi mount re-run, alone | 3 of 6 | 5 of 6 | 4 of 6 | Pi wall mount first try, 5.4 g, one piece, SF 3.11 (was 23.4 g in two pieces). Fixed on a development part (docs/GENERAL-DESIGN.md, "One piece") |
| 2026-11-25 | main 27e7c41: bearing loads square to the wall and through a rigid ring that only pushes (insert.py); seat springs low rank in the multigrid: pillow block re-run, alone | 4 of 6 | 6 of 6 | 4 of 6 | Pillow block first try, 11.0 g, SF 2.51 (fine grid 2.67; tets 2.47, within the tet check's step-back rule), 22 min (was "fails honestly", 2.33 on the block). Checked on a development part (docs/GENERAL-DESIGN.md) |

The scorer leaves a warning about this machine's free memory out of the
"clean check" count: it is about the machine, not the study. A diagnosis
case (`pass.diagnosis`) counts as first try when `check` says what is wrong.
