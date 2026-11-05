# The eval suite: 15 parts with hard requirements

## Problem

An optimizer can look good on the problems it was tuned on and fail on the
next part. loadpath keeps a fixed set of real parts, each with requirements
it must meet, and reruns all of them after any change to the solver. A
change that makes any part fail, or heavier at its target safety factor by
more than 5%, is a regression.

## Setup

Each case is a loadpath study (`loadpath/evals/cases/*.json`) plus a list of
checks on the delivered mesh: mass, safety factor, deflection, keep-out
intrusion, lost interface geometry, number of bodies, thinnest member, and
process checks where the case has a process. The cases:

| Case | Material, process | Goal | What it exercises |
| --- | --- | --- | --- |
| arm-link | PA12 | SF 3, 0.5 mm at the wrist | A 140 mm robot-arm link with 22 mm bearing seats and a cable channel; 40 N payload, 15 N side load. Stiffness-bound. |
| arm-link-driven | PA12 | SF 3, 0.5 mm | The same link held the way a servo holds it: a pin at the shoulder, driven through four M3 horn holes. |
| arm-link-cast | PA12 | SF 3, 0.5 mm | The same link with no undercuts along z (two-part mold). |
| arm-link-sandwich | PA12 | SF 3, 0.5 mm, under 45 g | The link as a field graph: graded skin over a gyroid core. Must also reject a 38 g variant that deflects too much. |
| bell-crank | AlSi10Mg, LPBF | SF 3, 0.1 mm | An aluminium linkage crank, 500 N on one arm. |
| robot-turret | PA12-GF, SLS | SF 2, under 236 g | The robot arm's turret: J1 bolt ring with driver access, J2 drive socket, motor and spring anchor. |
| robot-upper-arm | PA12-GF, SLS | SF 2, 0.54 mm, under 214 g | The robot arm's upper arm with drive sockets, motor, cable run and driver access. |
| robot-wrist-yoke | PA12-GF, SLS | SF 2, under 26 g | The robot arm's wrist yoke: a servo horn hub below and a servo cradle above. |
| robot-wrist-link-mjf | PA12, MJF | SF 2, under 46 g | The robot arm's wrist link in MJF, with powder-bed checks (walls, trapped powder, warp-prone panels). |
| mbb-beam | PA12 | SF 1.5 | The textbook simply supported beam. |
| motor-mount | PETG, FDM | three fixed volume fractions | A NEMA 17 bracket; the motor body is an STL keep-out. |
| printed-hook | PETG, FDM upright | SF 2.5, no support | A wall hook carrying 300 N at 92 mm, printed without support. |
| wall-bracket | PLA, FDM | SF 2.5 | A 120 x 100 mm shelf bracket on two M5 screws, 150 N. |
| wall-bracket-fasteners | PLA, FDM | SF 2.5 | The same bracket with its screws modelled as fasteners with preload. |
| lpbf-intercooler-core | AlSi10Mg, LPBF | effectiveness 0.7, pressure-drop limits | A two-fluid gyroid heat exchanger with no unsupported internal ceilings and a powder clean-out per fluid. |

Each safety-factor case searches over volume fractions for the lightest
part that meets its goal. The search's pick is then re-meshed as an STL and
checked by a fresh analysis on a tetrahedral mesh of that STL (P2 elements,
150,000-cell voxel grid for the comparison), so the reported numbers belong
to the delivered part, not to the optimizer's density field.

## Results

Full tier, baseline re-saved 2026-11-03 (main `6c97b57` plus the cast fix
that ships with it), with tet confirms, on a 4-core, 15 GB cloud machine with
two cases at a time. All 15 cases pass. Total compute time is about 5.8 hours.

"Solid" is the design space filled solid: the mass of the envelope the
part was allowed to occupy. It is an easy baseline. A part that is 90%
lighter than its envelope is not 90% lighter than a part a person would
draw. For comparisons with real, conventionally designed parts, see the
[the robot-arm case study](robot-arm.md) and the GE bracket in [benchmarks](benchmarks.md).

| Case | Mass | Solid envelope | SF (goal) | Deflection (limit) | What sets the size |
| --- | --- | --- | --- | --- | --- |
| arm-link | 55.3 g | 125.3 g | 8.92 (3) | 0.458 mm (0.5) | deflection |
| arm-link-driven | 48.4 g | 124.5 g | 11.2 (3) | 0.437 mm (0.5) | search step* |
| arm-link-cast | 67.8 g | 125.3 g | 20.1 (3) | 0.478 mm (0.5) | deflection, no undercuts |
| arm-link-sandwich | 42.7 g | | 6.65 (3) | 0.432 mm (0.5) | fixed design (field graph) |
| bell-crank | 23.4 g | 150.7 g | 3.22 (3) | 0.073 mm (0.1) | stress |
| robot-turret | 140.4 g | 1651 g | 2.85 (2) | 0.729 mm | interfaces (see below) |
| robot-upper-arm | 207.7 g | 1692 g | 7.33 (2) | 0.519 mm (0.54) | deflection |
| robot-wrist-yoke | 17.3 g | 248.3 g | 3.88 (2) | 0.229 mm | search step* |
| robot-wrist-link-mjf | 40.2 g | 773.5 g | 5.83 (2) | 0.490 mm | search step* (layered, tet-checked) |
| mbb-beam | 13.8 g | 65.4 g | 1.51 (1.5) | 3.182 mm | stress |
| motor-mount | 27.6 g | 89.2 g | 3.65 | 0.430 mm (0.5) | fixed fraction (trade-off set) |
| printed-hook | 24.2 g | 151.4 g | 2.73 (2.5) | 1.240 mm | stress, overhangs |
| wall-bracket | 26.9 g | 419.2 g | 2.85 (2.5) | 0.692 mm (1.5) | stress (constrained method)† |
| wall-bracket-fasteners | 26.5 g | 421.6 g | 2.57 (2.5) | 0.566 mm (1.5) | stress (constrained method)† |
| lpbf-intercooler-core | 131.8 g | | | | effectiveness 0.77 (0.7), 1.0 mm wall between fluids, 126 mm² flat ceiling (limit 200) |

\* The search brackets in volume-fraction steps. The next lighter step
(45.0 g for arm-link-driven, about 16 g for the yoke, 36-37 g for the MJF
wrist link) did not meet the goal on the delivered mesh, so the lightest
passing step is reported, with margin to spare. The wrist link was 31.4 g on
the Nov 1 baseline: the stem smoothing added on Nov 3 (`6c3c642`) costs it
about 9% at its target, because its 1.8 mm members lose more of the mesh;
with the smoothing off the same code gives 31.4 g again.

† `goal.method: "auto"` now picks the constrained optimizer for these, which
is why they dropped from 64 and 41 g. They are strong enough but not stable
enough: linear buckling puts the ~25 g bracket at a load factor of 1.4-1.6
against the 2.5 goal, and no buckling check runs in the evals yet.

A high safety factor on a stiffness-bound part (the arm links, the upper
arm) is expected: the deflection limit sizes the part, and stress is far
from the limit.

| | | |
| --- | --- | --- |
| ![arm link](../images/cases/arm-link.png) | ![arm link, driven](../images/cases/arm-link-driven.png) | ![arm link, cast](../images/cases/arm-link-cast.png) |
| arm-link | arm-link-driven | arm-link-cast |
| ![bell crank](../images/cases/bell-crank.png) | ![wall bracket](../images/cases/wall-bracket.png) | ![printed hook](../images/cases/printed-hook.png) |
| bell-crank | wall-bracket | printed-hook |
| ![motor mount](../images/cases/motor-mount.png) | ![MBB beam](../images/cases/mbb-beam.png) | ![fastened bracket](../images/cases/wall-bracket-fasteners.png) |
| motor-mount | mbb-beam | wall-bracket-fasteners |

Stress on the delivered part, worst load case, blue (none) to red (the
part's 98th percentile):

| | | |
| --- | --- | --- |
| ![](../images/stress/bell-crank.png) | ![](../images/stress/robot-upper-arm.png) | ![](../images/stress/printed-hook.png) |

Renders are from the same run as the table (three.js, smooth normals).
Every view and stress map is in [GALLERY.md](../GALLERY.md).

## Limitations

- **All numbers are simulation.** The analysis is linear elastic with
  datasheet material values. No part in the suite has been printed and
  tested ([VALIDATION.md](../VALIDATION.md)).
- **The turret's tet check disagrees with its voxel check.** The voxel
  check reads SF 2.85. The tet check reads 1.2-1.6 at a wall beside a J1
  bolt's driver access, below the goal of 2, and no heavier outcome fixed it.
  The case falls back to the voxel reading with a warning. About 76% of the
  turret's mass is fixed interface geometry, so the optimizer has little
  room to move there. Treat the turret as unresolved.
- **Run-to-run spread.** Restarting the search from a different volume
  fraction moves mass at the target safety factor by 3.5-7%; a 3% change of
  grid pitch moves it by about 5% (median). Mass differences under about 5%
  between runs are noise.
- **Tet meshes are not deterministic.** fTetWild meshes change from run to
  run, which moves tet readings by about 3%.
- **The suite is small and robot-arm-heavy.** A generated corpus of brackets,
  links and three-point mounts (`--corpus N`) runs in the coarse tier to
  catch fixes that only help the hand-written cases.

## Reproduce

```bash
pip install -e ".[crosscheck]"        # the tet mesher; the full tier refuses to run without it
python -m loadpath.evals.run --out evals-out/                     # full tier, hours
python -m loadpath.evals.run --tier coarse --corpus 6 \
    --baseline loadpath/evals/baselines/coarse.json --gate regressions   # coarse tier, under 30 min
```

The exit code is the number of failing cases. `evals-out/scorecard.md` has
the table above; `loadpath/evals/baselines/full.json` holds the numbers
quoted here.
