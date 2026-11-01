# Compliant mechanisms

Parts that are meant to bend: gripper jaws, flexure hinges, snap fits,
inverters. `loadpath/mechanism.py` designs them by topology optimization
(Sigmund 1997; Bendsøe & Sigmund 2003, sec. 2.6): push at an input, and an
output should move as far as possible in a chosen direction against a
spring (the workpiece), with a spring at the input too (the actuator).
`loadpath/nonlinear.py` then checks the result the way it will be used,
with large deflections.

```python
from loadpath.mechanism import MechanismOptions, optimize_mechanism
from loadpath.nonlinear import stroke_mechanism, verify_mechanism

r = optimize_mechanism(domain, nu, MechanismOptions(
    volume_fraction=0.3, filter_radius=2.0, modulus=26,     # TPU
    k_in=50, k_out=0.25,                                    # N/mm: servo, workpiece
    robust=0.15))                                           # no one-node hinges
rep = stroke_mechanism(domain, r.xphys, 26, 0.45, stroke=3.0, k_out=0.25)
```

`domain.loads[0]` is the input force (its direction is the stroke's),
`domain.loads[1]` the output's wanted direction (any magnitude).

## What it does

- **Objective:** the output's mean travel along its direction, maximized
  with MMA. The adjoint sensitivity takes both signs, which optimality
  criteria can only use by clipping.
- **Springs:** `k_in` and `k_out` are totals in N/mm, shared by the nodes
  of each region. Before 2026-10-31 a push along one axis got 3x the spring
  it asked for.
- **Robust option (`robust`)** (Wang, Lazarov & Sigmund 2011): the filtered
  field is projected three ways, eroded (threshold 0.5 + robust), blueprint
  (0.5) and dilated (0.5 - robust), and the worst output is maximized (a
  smooth minimum). A one-node hinge, two cells meeting only at a corner,
  disconnects in the eroded design, so the optimum keeps its hinges about a
  filter radius wide. That is printable, and it tolerates a line width of
  print error either way. The blueprint is what is delivered.
- **Several input/output pairs** (`pairs`): contact anywhere along a finger.
- **Checks:**
  - `verify_mechanism` re-solves the solid cells under the input force
    with large deflections and checks buckling.
  - `stroke_mechanism` drives the input through a stroke (prescribed, as a
    servo or lead screw does) against a workpiece spring. It reports the
    output's travel, the actuator force, the grip force, the geometric and
    mechanical advantage, and the peak stress and strain, step by step.

## Validation: the textbook benchmarks

`python -m loadpath.benchmarks.mechanisms` runs the force inverter and a
gripper through two optimizers:

- A line-for-line NumPy port of the compliant-mechanism version of
  Sigmund's 99-line code: plane-stress squares, SIMP p = 3, the
  sensitivity filter, and optimality criteria with damping 0.3 and move
  0.1.
- loadpath's 3D optimizer on a one-cell slab.

One independent plane-stress evaluator scores every design. It uses the
reference's own element and springs, at equal volume, and reads each design
black and white (as printed) and grey (as delivered).

| | output, black and white | greyness | one-node hinges |
|---|---|---|---|
| **Inverter** (40 x 20, the book's own example: volume 0.3, rmin 1.2, springs 0.1) | | | |
| 99-line code | 1.157 | 0.108 | 0 |
| loadpath, r = 1.5 | **1.180** (+2.0%) | 0.002 | 0 |
| loadpath, r = 2 | 1.171 (+1.2%) | 0.008 | 0 |
| loadpath robust | 1.108 (-4.2%) | 0.038 | 0 |
| **Gripper** (60 x 30, 10 x 6 jaw gap, after Sigmund 1997) | | | |
| 99-line code | 1.403 | 0.127 | 0 |
| loadpath, r = 1.5 | 1.392 (-0.8%) | 0.005 | 2 |
| loadpath, r = 2 | **1.415** (+0.9%) | 0.014 | 1 |
| loadpath robust | 1.356 (-3.4%) | 0.040 | 0 |

![inverter](images/mechanism/inverter-2d.png)
![gripper](images/mechanism/gripper-2d.png)

loadpath matches the textbook optimum within about 1-2%, with nearly
black-and-white designs. The reference's designs are 11-13% grey, so they
need a threshold before they can be printed. The robust option gives up
3-4% of output and in exchange removes the corner hinges.

## A TPU gripper, driven by a servo

`examples/mechanism/gripper.py` uses the gripper layout at 1 mm per cell,
printed flat in TPU 95A as an 8 mm plate. The servo is stiff
(400 N/mm through the horn), the workpiece soft (2 N/mm), and the stroke
3 mm. All three designs are optimized for the same springs and volume, and
then driven through the stroke by the nonlinear solver (`stroke_mechanism`,
about 4 min in all):

| | textbook code | loadpath | loadpath robust |
|---|---|---|---|
| one-node hinges | 7 | 1 | **0** |
| free jaw travel per 3 mm stroke | 8.4 mm | 6.7 mm | 3.9 mm |
| grip force on the workpiece at 3 mm | solver stopped at 0.6 mm (1.5 N) | **6.7 N** | 5.4 N |
| actuator force for that | 4.8 N at 0.6 mm | 26 N | 25 N |
| peak strain | 0.9% at 0.6 mm | 4.5% | 7.3% |
| grip held for 1 min / 1 h / 1 day (relaxation) | n/a | 6.2 / 5.3 / 4.8 N | 5.0 / 4.2 / 3.8 N |
| servo force for a 0.2 s close | n/a | 31 N | 30 N |

![TPU gripper](images/mechanism/tpu-gripper.png)

- **The textbook design is a mechanism of corner hinges.** With a stiff
  actuator it puts seven one-node hinges in its jaw linkage. They give it the
  biggest free travel, because a corner hinge costs almost no strain energy,
  but they print as nothing. The large-deflection solve of the
  gripping case inverts elements at those corners and stops at a fifth of
  the stroke.
- **loadpath's designs work through the full stroke.** The plain design
  grips harder (it keeps one corner hinge). The robust one has none, and it
  is the one to print.
- **TPU relaxes.** Held at full stroke, the grip force falls to 64% of its
  first-second value in a day (`viscoelastic.py`, m = 0.045, a sourced
  value until a K-25 coupon is measured). Size the grip for the force left
  after the hold time, or keep the servo under closed-loop force control.
  A quick close takes about 20% more servo force than a slow one, because
  TPU is stiffer at speed.

## Large strokes: optimizing on the large-deflection response

`loadpath/mechanism_nl.py` designs on the response the part will actually
have (Pedersen, Buhl & Sigmund 2001). The input is driven through a
prescribed stroke, the full grid is solved with large deflections (Newton,
total Lagrangian hex8, neo-Hookean), and an adjoint gives the gradient of
the output's travel at the end of the stroke. Void cells use the
Wang, Lazarov & Sigmund (2014) interpolation, so near-empty cells fall back
to a linear response and do not invert. A failed Newton solve backs the
design halfway toward the last good one. The modulus and Poisson's ratio
can be per element, for multi-material parts.

```python
from loadpath.mechanism_nl import optimize_mechanism_nonlinear
r = optimize_mechanism_nonlinear(domain, nu, MechanismOptions(
    volume_fraction=0.3, filter_radius=1.5, k_out=0.02), stroke=12.0)
```

`python examples/mechanism/large_stroke.py` (about 2 min) compares it with
the linear design on the textbook inverter driven 12 cells (30% of its
length), both judged by `stroke_mechanism`:

| | linear design | large-deflection design |
|---|---|---|
| output travel at the full stroke | 2.98 | **3.57** (+20%) |
| what linear theory promised the linear design | 5.39 | |
| one-node hinges | 1 | **0** |
| peak strain | 17.6% | 16.4% |

![large-stroke inverter](images/mechanism/large-stroke-inverter.png)

The linear model over-promises the linear design by 81%. Its output peaks
near 10 cells of stroke and then stalls as its members swing past their
leverage. The design optimized on the large-deflection response keeps
pulling through the whole stroke, with no corner hinge.

## A TPU finger that curls

`examples/mechanism/curling_finger.py` (about 20 min) designs a 40 x 10 mm
TPU 95A finger, printed flat. It is clamped along the top half of its
base. A tendon or servo horn pulls the bottom of the base 5 mm back, and
the tip should curl down against a soft object (0.01 N/mm). Both designs
use the robust option, and both are judged by driving them through the
stroke with the nonlinear solver:

| | linear design | large-deflection design |
|---|---|---|
| tip travel for a 5 mm pull | 0.7 mm (solve stops at 0.75 mm of pull) | **8.7 mm** (full stroke) |
| one-node hinges | 8 | **0** |
| pull force at 5 mm | | 7.1 N |
| peak strain | | 45% (TPU 95A breaks past 400%) |

![curling finger](images/mechanism/curling-finger.png)

Three things made it work, all now in `mechanism_nl.py`:

- **Damped Newton steps.** A full step can invert a solid cell next to
  soft void ones. The step is halved until no cell inverts. Before this,
  a sharper design often failed to reach the stroke and the optimizer
  stayed grey.
- **Sharpening is retried, not abandoned.** If a sharper projection cannot
  reach the stroke, the optimizer keeps the last one for a while and tries
  again; it stops sharpening only after three refusals.
- **The robust option.** Without it the optimizer finds a grey answer: its
  tip travel (11-25 mm) runs through grey cells and one-cell grey hinges.
  Sharpened, that answer drops from 11 mm to 0.01 mm, and thresholded, the
  tip falls off. With robust, the eroded design must work too, so a grey
  hinge is worth nothing.

The check (`stroke_mechanism`) now also reports a tip held only by the
plate's symmetry plane as detached; it used to read 0 mm.

## A tendon-driven finger that grips

`mechanism_nl` also takes a tendon as the input and an object to close on,
and then maximizes the grip force:

```python
from loadpath.mechanism_nl import RigidCylinder, ServoTendon, optimize_mechanism_nonlinear, tendon_grip
ten = ServoTendon(path=[spool_xyz, guide_nodes, ..., knot_nodes], stroke=8.0, stiffness=0.5)
obj = RigidCylinder(center=(30, 0, -16.5), radius=15)
r = optimize_mechanism_nonlinear(domain, nu, opts, tendons=[ten], contact=obj)
check = tendon_grip(domain, r.xphys, E, nu, [ten], obj)     # thresholded, full pull
```

- **Tendons.** A `nonlinear.Tendon` is a cable at constant tension through
  guides on the part; it follows the part as it curls. A `ServoTendon` is
  wound in by a servo: a set length of cable (`stroke`), with the cable
  and servo giving like a spring (`stiffness`), so the pull can never pass
  stiffness x stroke. Use the servo version for optimization. At constant
  tension the soft, grey early designs snap over the object, and no solver
  can follow a snap under a fixed force; a wound-in cable is displacement
  driven and can.
- **Contact.** `RigidCylinder` is a fixed object (a cylinder through the
  plate). Each node is pushed out of it by a smooth penalty, weighted by
  the density around the node (cubed), so empty cells cannot grip. The
  grip force is the sum of the normal forces, and its gradient (contact,
  tendon load stiffness and all) matches finite differences.

`python examples/mechanism/tendon_grip.py` (about 25 min) designs a 50 x 10
mm TPU 95A finger, its base clamped to the palm. The tendon runs from the
spool behind the palm through guides on the palm side to a knot at the
tip. A 30 mm cylinder (a GRIP-sized handle) sits 1.5 mm below. The servo
winds in 8 mm of cable at 0.5 N/mm. Forces are per mm of finger width
(a 15 mm wide finger grips 15 times as much):

| | plain palm-side bar, same volume | optimized (robust) |
|---|---|---|
| grip force | 0.046 N | **0.42 N** (9x) |
| tendon tension at full wind | 2.27 N | 0.88 N |
| grip per newton of tendon pull | 0.02 | **0.48** |
| one-node hinges | 0 | 0 |

![tendon grip](images/mechanism/tendon-grip.png)

The bar curls its tip past the cylinder and touches it with one edge. The
optimized finger is a lever: a diagonal strut from the top of the palm
mount carries the tendon's pull down into a knee that presses on the top of
the cylinder, while the tip beyond it buckles up out of the way. It does not
wrap the object; it pushes on it, which is what "most normal force" asks
for. Two things to know before printing it:

- **It cut the middle guide free.** Printed, that guide would be a loose
  bead on the cable. Leave it out: the cable runs straight from the palm
  to the next guide, clear of the cylinder (checked on the deformed shape).
- **The object is fixed.** The grip is what the finger presses on a held
  object; a second finger or the palm on the other side is not modelled.

## Not yet

- **Contact with one rigid cylinder only**, fixed in space; no friction,
  no second finger, and the cable does not touch the object (check its
  path on the deformed shape).
- **Wrapping grasps:** maximizing the normal force gives a pressing
  finger. A finger that wraps would need the contact spread over the
  object (more nodes in contact, or a pull-out force objective).
- **2.5D plates only** for the gripper above. The optimizer itself is 3D;
  GRIP's tendon-driven TPU finger (examples/grip/finger.py) has the
  nonlinear analysis but not yet a mechanism design.
