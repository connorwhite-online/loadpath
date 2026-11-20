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

RESULTS_TABLE

## Gaps, ranked by how many users they hit

"Hits" is the share of the held-out parts where the gap decided the
outcome on the cold run, plus a judgement of how often a new user meets it.

| # | Gap | Link | Hits | State |
|---|---|---|---|---|
| 1 | A goal the envelope cannot meet is found by running to the 70% ceiling, then answered with "raise the volume fraction" | 1, 6 | 4 of 11, every new user | **Fixed**: a failing search verifies the solid envelope, stops if the goal is out of reach, and says what the block reads, where it peaks and what would reach the goal (`report.bound`) |
| 2 | `check` warns on loadpath's own fastener rings, so a clean check means nothing | 1 | 5 of 11, every study with a bolt | **Fixed**: the warning measures the wall the cut leaves, not the share cut |
| 3 | A named build direction forbids every overhang, so parts split or balloon | 4 | 2 of 11, every FDM study that names a direction | **Fixed**: the direction sets the layers; supports are decided by cost unless `overhangs` is declared |
| 4 | The solid block passes but the search stops at 70% and hands back nothing | 6 | 1 of 11 | **Fixed**: the block is delivered when nothing lighter passes, with a note that the envelope sets the mass |
| 5 | A design space in separate pieces runs for minutes and passes or fails without saying so | 1 | 1 of 11, common with keep-outs through thin envelopes | **Fixed**: flagged by `check` before the run |
| 6 | A part in pieces: the search adds material, which never joins pieces that do not need each other | 5 | 1 of 11, any plate or frame with paired mounts | **Partly**: the search stops and says which load is missing. Open: the load that makes a part one piece (being carried, assembly) is not modelled |
| 7 | Glued bores and sharp inside corners set the safety factor: the peak is the model's, not the part's | 2 | 3 of 11, every part with a bore or a throat | **Open**: compliant one-sided supports at bores, bearing and pull-through read at the joint, a fillet the process would leave at an inside corner |
| 8 | Small parts take 5 to 17 minutes | 8 | all | **Open**: the envelope bound first (seconds) would stop the impossible ones; coarse-first search for the rest |
| 9 | Joint capacity is not checked: bolt slip, thread strip, insert pull-out, clamp force on a tube | 2 | most bolted parts | **Open** |
| 10 | The governing limit is not named up front (stiffness, strength, minimum wall) | 3 | 2 of 11 | **Partly**: the bound note names it after two failed tries; should be in `check` |
| 11 | Enclosure for a common board from a sentence: placement fails with an error naming the wrong component, and is slow | 1, 7 | every enclosure user | **Open** |
| 12 | Function parts with no route today: snap fits and living hinges by strain, ducts from two ports, threads, gears | 7 | very common requests | **Open**: no held-out cases yet; write them cold first |
| 13 | Physical ground truth | 9 | all | **Open**: calibration prints (calibration/) |


## Rule for work under this document

Fix the general cause, never the case. A change is kept only if it follows
from the physics or from the statement of the problem, and it is proved on
the held-out set by re-running it, not by tuning to it. If a held-out case
turns out to be wrong as a problem statement, it stays in the set as a test
of diagnosis.
