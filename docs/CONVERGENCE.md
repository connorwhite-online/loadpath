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
and a fourth stopped a calculation before its answer was settled. Each one
is now sized from a quantity the run measures.

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

Bisection now only brackets the root. Illinois false position then lands on
it to 1e-13 of the target (`simp.VOLUME_RTOL`), which costs 3 to 5 more
evaluations of the projection and no solves.

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

Below 1e-4 the tolerance is no longer what separates the runs (next
section), so that is where it sits. An exact direct solve would give the
tightest row without the iterations, where memory allows one.

## What is left: the sharpest stage

At beta 32 a boundary cell's projected density changes 16 times as fast as
its design variable, and whichever tolerance the solves use, a difference
of 1e-9 entering that stage leaves it near 3e-4 in compliance. A shorter
step (aiming at 0.7 or 0.5 of the measured one) and a wider overhang
corner were both tried there and neither changes it. The stop rule then
adds its own share: the run ends when six iterations bought less than
0.2%, the design is still creeping when it does, and two runs 3e-4 apart
can stop a few iterations apart. Together that is the 0.1% in delivered
mass in the table. It is 30 times smaller than what the search steps on
(the hook's safety factor read 3.32 on one run and 2.13 on another before;
the same trial now reads 2.034 and 2.033), but it is not zero.

## What it costs

The measured step is smaller than the old fixed one in the sharp stages, so
a run takes more optimizer iterations: 94 against 60 on the hook, 96 against
70 on the bell crank. The designs it reaches carry the asked volume to
rounding; the old runs under-delivered about 2%, which read as lighter parts
with correspondingly less stiffness.

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
