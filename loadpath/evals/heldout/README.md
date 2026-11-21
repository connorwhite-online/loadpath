# Held-out parts: the cold benchmark

Common parts nobody has tuned loadpath on: a corner bracket, a hinge leaf, a
drawer pull, a VESA plate, a caster fork, a pipe clamp, a crank lever, a
tablet stand, a C-clamp, an action-camera mount; and a second batch
(`"batch": 2`): a tool clip, a latch hook, a hose elbow, a fan duct, a
junction-box cover, a wall box. Each study was written once,
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

The scorer leaves a warning about this machine's free memory out of the
"clean check" count: it is about the machine, not the study. A diagnosis
case (`pass.diagnosis`) counts as first try when `check` says what is wrong.
