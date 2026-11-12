# Convergence and repeatability of the search

An agent that asks for the same part twice should get the same part. Until
2026-11-11 it did not: the printed hook came out at 27.2, 28.7 and 30.5 g on
three full runs of the same code, and the safety factor of its first trial
read 3.32 on one run and 2.13 on another. Nothing in the code is random. The
optimizer was amplifying differences in the last bit of a linear solve into
differences in the part.

This page records what was amplifying them, how each cause was isolated, what
replaced it, and how to measure it again. The short version: three places in
`simp.optimize` took a step that the physics of the problem did not support,
two stopped a calculation before its answer was settled, and the search
stepped on a stress reading that was not the one its parts are confirmed
with. Each is now sized from a quantity the run measures.

## How to measure it

Set `LOADPATH_JITTER=<seed>` and the search starts from a design with 1e-9
relative noise on it (`<seed>:<size>` for another size; 1e-15 is last-bit
noise, what a different machine gives). Run the same study with two seeds and compare
compliance iteration by iteration. In a well-behaved descent the two stay
1e-9 apart or close in. If the difference grows by a steady factor per
iteration, something is amplifying it, and the growth rate tells you how
strongly. Switching one mechanism off at a time and repeating the pair is
how each cause below was found. A pair of single optimizes takes a minute or
two; no full study is needed.

## Cause 1: the step was past the stable step at the sharpest projection

The optimality-criteria update moves each cell by at most a move limit. The
limit was fixed: 0.2, halved once the projection passed beta 4. At beta 32
the projected density changes about beta/2 times as fast as the design
variable, so a move of 0.1 swings a boundary cell from void to solid and
back. The design flickered between two states with period 2, compliance
rose about 1% an iteration instead of falling, the change never dropped
under the tolerance, and the stall rule ended the run at whatever iteration
it happened to fire. Two bell-crank runs separated by a factor of 2 every 3
iterations in that stage.

A fixed number cannot be right here, because the stable step depends on the
part, the stage and the mesh. The run can measure it. Along the last step
dx, the slope of the Lagrangian is a = g_old . dx before the step and
b = g_new . dx after it (g = compliance gradient plus the volume multiplier
times the volume gradient). If the slope fell to zero the step was exact; if
it changed sign the step overshot. The secant estimate of the exact step is
t = a / (a - b) times the one taken, so the move limit is multiplied by t,
kept within 0.5 to 1.2 per iteration, and never above the configured limit.
At a projection doubling the curvature doubles, so the limit is halved
there and re-measured from the next step.

Measured stable steps at beta 32 are 0.01 to 0.03, roughly 1/beta, which is
what the argument above predicts. Compliance in the last stage now falls
monotonically on the bell crank and the hook; on the small lug in the tests
its largest single rise is 1e-4 against 1e-2 before.

The stage and stop rules use the same measurement: a stage only counts as
settled when the largest change is below the tolerance and below the move
limit, so a run cannot pass the test merely because its limit is small.

`SimpOptions.adaptive_move` (default on) switches this; off restores the
fixed schedule.

## Cause 2: the volume multiplier was solved to 1e-4

Each update solves for the multiplier that puts the blueprint volume on its
target. Bisection stopped once the bracket was within 1e-4. That makes the
step a staircase in its inputs: a 1e-10 difference in a sensitivity either
leaves the multiplier alone or moves every free cell by 5e-5. Two runs that
were 1e-10 apart were 5e-5 apart one iteration later.

The multiplier is now solved to 1e-13 of the target (`simp.VOLUME_RTOL`).
It moves little from one iteration to the next, so the solve brackets
outward from the last one and Illinois false position lands on the root:
about 8 evaluations of the filter and projection a step where the bisection
took 24, and no solves. (The bisection was 9% of a hook run.)

## Cause 3: the overhang filter's kink on a grey design

For printed parts the overhang filter (Langelaar) takes a smooth minimum
of each cell and its support, with a smoothing width of 1e-4. The smooth
minimum is nearly a kink where the two are equal, and on the uniform grey
design the search starts from every cell is exactly there. With the filter
off, two hook runs agree to 11 digits after 20 iterations. With it on they
separated by a factor of 2 an iteration from the first.

The width that makes sense is the scale on which the densities themselves
are resolved, and that is set by the projection. `am.overhang_eps` scales
the width with (32 / beta) squared, capped at 1e-2, so the filter is smooth
while the design is grey and is exactly the old value at beta 32, where the
search ends. The delivered part is filtered at the final width, as before.

The width follows the design's own sharpness, not how far a run is through
its schedule. Every study starts from a coarse warm-up run that stops at
beta 4. Tying the width to the run's last stage gave that warm-up the sharp
rule on a design still grey: two warm-ups a rounding error apart separated
by a factor of 2 an iteration and handed the fine run starts 0.1% apart.
With the width tied to beta they hand over starts within 1e-8.

## Cause 4: the linear solve was stopped where its answer still depends on the last bit

Early iterations solved the elasticity system to a relative residual of
1e-2, later ones to 1e-3: a rough gradient is enough to descend on. But a
conjugate-gradient solve on a stiffness matrix with nine decades of
contrast is itself sensitive. Stopped that early, which answer it stops at
depends on the last bit of its input. Two hook runs that started one
rounding error apart (`LOADPATH_JITTER=1:1e-15` and `2:1e-15`) took the same
number of solver iterations in every solve and were still 4e-7 apart in
compliance after the second one. The sharpening projection then grows that
by about a thousand, and the two runs deliver layouts that differ in 0.4%
of their cells (2.1% before causes 1 to 3 were fixed). A last-bit
difference is what another machine, thread count or library build gives,
so this is the part of the spread a user actually sees.

A solve that has converged does not have this freedom: both runs are then
within the tolerance of the one true answer. `SimpOptions.solve_rtol` is now
1e-4 at every stage. Measured on the hook's first trial (warm-up, fine run,
mesh, check), two runs one rounding error apart:

| Solve tolerance | Apart after warm-up | Delivered mass apart | Safety factor apart | Time |
| --- | --- | --- | --- | --- |
| 1e-2 to 1e-3 (before) | 5e-7 | 0.50% | 0.07% | 1.0 |
| 1e-4 | 8e-9 | 0.13% | 0.01% | 1.1 |
| 1e-5 | 6e-9 | 0.06% | 0.05% | about 1.4 |
| 1e-7 | 2e-10 | 0.04% | 0.04% | 1.7 |

Below 1e-4 the tolerance is no longer what separates the runs (cause 5).
Where the system fits in memory the solve is now a factorization, exact to
rounding, which gives the tightest row without the iterations and is the
faster solve as well (BENCHMARKS.md, solver routing).

## Cause 5: the run stopped when its compliance stalled

The sharpest stage ran until six iterations bought less than 0.2%. Two
things are wrong with that. The design is still creeping when the rule
fires, so two runs a last bit apart fire it at different iterations: on the
hook's first trial 83 against 85, 0.02% apart in mass and 0.1% in safety
factor, with every other cause above already fixed and the solves exact.
And the iterations it waits through buy nothing. Cut at 35, 40, 50, 60 and
75 iterations the same hook trial read safety factor 1.97, 2.01, 2.10, 2.14
and 2.16 at 32.6, 32.8, 33.1, 33.4 and 33.6 g: one line of strength against
mass. The long tail only brings the delivered volume up to the asked one,
which the search does more cheaply by asking for a larger fraction.

The sharpest stage now ends on its schedule like every stage before it,
once the volume is on target. The same two runs: 40 iterations each, mass
equal to 8 digits, safety factor to 16. A problem with a weight still
climbing (buckling, vibration, warp, centre of mass, casting feed, a
maximum member, a temperature swing) is not at rest on a schedule and
keeps the stall rule.

## Cause 6: the search stepped on a different stress than the check that confirms it

The search sizes each trial from the voxel check and confirms the part it
picks on tetrahedra. The tetrahedra read stress averaged over the
material's length (STRESS-READ.md); the voxel check read the peak at a
node. A point peak on a stair-stepped boundary moves with the layout, the
average does not, so the ratio between the two checks was not a constant
the search could calibrate once: over seven hook designs of one search the
point reading was 0.76 to 0.92 of the tetrahedra's, and the calibration
factor moved from 1.35 to 1.71 between two designs. A search stepping on
that lands on a different fraction every time one design's ratio moves.

Two changes make the two checks one measurement:

- `verify._MaterialAverage` averages the voxel stress tensor over the same
  length before any criterion is taken (von Mises, the layer bond,
  submodels). `LOADPATH_STRESS_READ=point` restores the old reading.
- The voxel check left out one cell round each support, the tetrahedra
  twice their element size. The voxel peak sat in the strip the tetrahedra
  left out. `study.contact_band` is now the one band for every check.

With both, the voxel reading is 0.92 to 1.02 of the tetrahedra's over the
same seven designs, and the hook's three searches (three grids 1% apart in
pitch) confirm at 40.1, 40.0 and 40.6 g. Main's three, same machine, same
day: 36.8, 39.9 and 42.6 g.

## What is left: which layout a fraction lands on

Parts of one mass differ in strength by which design the fraction started
from. Fresh hook designs on main's code had strength per mass squared
(x1000) of 2.14, 2.44 and 1.84 at 20, 22.5 and 26%, and 1.51 warm-started
from a neighbour. The lightest hook main finds (36.8 g, with 29% to spare
on the tetrahedra) is one such draw; the other two grids did not repeat it.
The repeatable search lands on one layout every time (1.9 to 2.0), which is
the middle of main's draws, not the best.

The hook breaks at its layer bond, and the optimizer minimizes compliance
with an isotropic stiffness: it cannot tell a member loaded across the
layers from one loaded along them. Every hook the search delivered was
limited by the bond with in-plane strength standing idle (a margin of 4 to
6 in plane beside 2 to 2.5 on the bond). Three things came of following
that up (2026-11-12).

**A design stiffness cut across the layers is a weak lever.** With the
across-layer terms at the bond's strength ratio (`process.anisotropic:
"bond"`) the search's hook came out 33.4 g against 39.7 g, on one grid of
three; on the other two it was no lighter. Softer still is worse: at the
ratio to the powers 2, 3 and 4 the 33 g hook read safety factor 2.05, 1.80
and 1.43 against 2.56. Compliance puts material where it is stiff, so a
mode made soft is one the design stops using, and the shear a cantilever
has to carry across its layers ends up in frames that bend. It stays
opt-in.

**The delivery step dropped members the optimizer had analysed.** Half of
the bond-aware hooks came out broken: a print-support column, analysed one
cell across and counted on to hold up everything above it, was not grown
back by the opening, and the overhang rule then removed the top chord's
root (safety factor 0.2 for an analysed 2.5). The opening now returns the
blueprint round every analysed cell it lost (`simp._keep_cores`); a design
that lost none is delivered exactly as before.

**The pick is finished on the limit it breaks by.** After the search's
pick is confirmed, a part limited by its layer bond goes to the optimizer
that holds stress as a limit, now with the bond in that limit as
verification reads it, and has material taken off in short strides, each
one checked (`study._search.strength_polish`). Printed hook, full tier,
tet-confirmed, grams:

| Design stiffness | Search's pick | After the polish |
| --- | --- | --- |
| isotropic (default) | 39.7 | 33.7 |
| bond ratio (opt-in) | 33.4 | 29.7 |

Main's three draws were 36.8, 39.9 and 42.6 g. The polish does not remove
the dependence on layout: on the two unlucky grids it took 37.3 to 36.3 g
and 40.1 to 39.4 g, and a run keeps the best of its grids. Held at one
limit for 120 iterations with no check in between, the same optimizer
reached 23 to 24 g at 80% of the target, so the hook has more to give; what
stops it is that the optimizer's stress reading and the check's drift apart
as the design moves, and that buckling starts to bind near 27 g. Both are
the next steps: one stress reading for the optimizer and the check, and
buckling held in the polish.

## What it costs

Nothing. The whole hook search (three grids, every trial, every
confirmation) took 1298 s on the new code and 2336 s on main, side by side
on one machine. A trial runs 40 iterations where main ran 49 and the first
version of these fixes 87.

## Where the step is still scheduled

Two limits keep the fixed step, because neither is a function whose slope
along a step can be read:

- A maximum member size. It is enforced by a weight that is ramped while
  the solid's 95th-percentile ball is over the limit. With the weight at its
  cap the measured step is a few thousandths and the limit is still unmet
  after 200 iterations, where the fixed step meets it in 80. That limit
  wants a proper constraint (a multiplier solved each step, like volume)
  before its step can be measured.
- A temperature-swing limit, which takes its own constrained step.

## What was tried and dropped

- A deadbeat gain (multiply the limit by t within 0.5 to 2). The controller
  hunts: two hook layouts ended 9% apart.
- A step per cell or per neighbourhood from the same secant. Fewer
  iterations, but noisier and no longer monotone.
- Raising the penalty from 1 to 3 in stages. The growth is already there at
  penalty 1 once the overhang filter is on.
- A sharper overhang maximum (P = 8). The filter becomes badly conditioned.
- An uncapped smoothing width. Worse compliance, no gain in repeatability.
- Damping the update's exponent by the measured fraction instead of
  shortening the move limit. Cells far from balance still move the full
  limit, compliance jumps back up, and the run stops early on a greyer
  design.
- A tighter bisection alone. It removes the staircase and leaves the other
  two.
