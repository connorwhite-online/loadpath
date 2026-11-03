# Eval baselines

`coarse.json` and `full.json` hold the scores `python -m loadpath.evals.run --baseline` compares against. Re-save one with `--save-baseline` only when a change is meant to move the numbers, and add a line here saying why.

## coarse.json history

- **2026-11-03 (after 87ad5d4).** Re-saved from main with whole cells (cut cells off), 21/21 pass. Big moves vs the Nov 1 baseline (ab4bf9e), checked before saving:
  - The Nov 1 baseline was saved without a tet mesher, so every SF case was confirmed on voxels, which read safe-side heavy. With tet confirm the same ab4bf9e code gives mbb-beam 15.7 g, so most of mbb-beam (22.2 -> 14.3 g), grip-wrist-yoke (24.0 -> 17.2) and grip-turret (145 -> 138) is the confirm, not the optimizer.
  - wall-bracket (81.9 -> 23.0 g) and corpus-bracket-1 (139.9 -> 36.0) are df2521f, goal.method "auto" picking the constrained optimizer (search alone still gives 72.3 g wall bracket on both ab4bf9e and today's main). Both pass the tet confirm (wall bracket SF 3.40 vs goal 2.5). **Caveat:** the 23-24 g wall bracket buckles at a load factor of 1.43-1.61 (linear buckling, 1.2 mm grid), below the 2.5 goal; the 72 g search design reads 43.5. No buckling check runs in verify or the evals yet, so this baseline locks in a part that is strong enough but not stable enough.
