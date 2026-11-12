# External flow: a part in a stream, held to published drag

A drone arm, a canopy, a duct, a sensor mast on a vehicle or a bracket in a
current carries a load nobody writes down: the air or water pushing on it.
Topology optimizers take a force vector as given, and CFD packages stop at
a pressure plot. `loadpath external-flow` puts the part in a virtual wind
tunnel and hands back that load in the form the optimizer reads, a study
load case, so the flow and the structure are one step apart instead of two
tools apart. And it says how far to trust the number: every figure below
is against a published measurement or a resolved simulation.

```
loadpath external-flow arm.stl --speed 15 --direction -x --study arm.json
loadpath external-flow strut.stl --speed 15 --span z          # a slice of a long strut
loadpath external-flow duct.stl --rotors rotors.json          # hover: a rotor disk blowing on the part
```

prints the drag, side force and lift (N), their coefficients on the frontal
area, the moment about the centroid, the shedding frequency (Strouhal
number) where the wake sheds, and writes `arm-flow.json`: the study with a
new load case, `flow-15ms`, of surface patches each carrying its share of
the pressure and friction. The Python call is
`loadpath.aero.external_flow(mesh, speed, direction, medium="air")`;
`medium` takes `air`, `water`, `glycol50`, `oil` or a `Fluid`.

## Against published drag

`python examples/external-flow/benchmarks.py` runs these; each row is one
call with the defaults unless it says otherwise.

| Case | Re | Run at Re | Cells across | Cd | Published | Error | Minutes |
|---|---|---|---|---|---|---|---|
| Sphere | 20 | 20 | 24 | 2.715 | 2.73 (Dennis & Walker) | -0.6% | 5 |
| Sphere | 100 | 100 | 24 | 1.100 | 1.087 (Johnson & Patel) | +1.2% | 5 |
| Sphere | 300 | 300 | 24 | 0.688 | 0.656 to 0.671 | +2.5% over the top of the range | 16 |
| Sphere | 1000 | 1000 | 48 (one layer) | 0.515 | 0.47 to 0.48 (Clift et al.) | +8.5%, not settled (standard error 1.5%) | 34 |
| Cylinder, spanning | 20 | 20 | 32 | 2.037 | 2.05 (Dennis & Chang) | -0.6% | 21 |
| Cylinder, spanning | 40 | 40 | 32 | 1.532 | 1.52 | +0.8% | 22 |
| Cylinder, spanning | 100 | 100 | 32 | 1.355, St 0.160 | 1.33 to 1.35, St 0.164 to 0.166 (Henderson, Williamson) | +1.1%, St -3% | 43 |
| Square strut, spanning | 22,000 | 576 | 32 (one layer) | 1.79, St 0.136 | 2.1, St 0.132 (Lyn et al.); 1.84 at Re 500 (Sohankar et al.) | -16% on the physical stream, -3% on the Reynolds number it ran at; St +3% | 11 (four cores) |
| Cube, face on | 20,000 | 576 | 32 (one layer) | 0.96 | 1.05 (Hoerner) | -8%, not settled (standard error 4%) | 13 |
| Disk, face on | 20,000 | 576 | 32 (one layer) | 1.16 | 1.12 to 1.17 (Hoerner) | inside the range, not settled (standard error 4%) | 7 |

Minutes are on two of this machine's four cores (two rows ran side by side), at about 15 million cell
updates a second on all four. "Not settled" rows hit the time limit before the standard error of the mean
drag fell inside 1%.

What the table says. Where the grid holds the boundary layer and the wake
is steady or sheds cleanly (the rows to Re 300) the drag is within 1.2%,
2.5% at Re 300, and the cylinder's shedding frequency within 3%. The
sphere at Re 100 read 1.216 (12% high) in 5.6 minutes on the solver this
replaces, a uniform grid with staircase walls in a tunnel six diameters
wide. Past that the table is honest about two limits. A chaotic wake
(sphere at Re 1000) reads 8.5% high and does not settle inside the time
limit. And a part in a stream faster than the grid resolves is run at a
lower Reynolds number: a disk and a cube lose little by it (inside the
range, and 8% low), a long square strut loses 16%, because a strut's drag
keeps rising with Reynolds number to 2e4 (the run is right for the
Reynolds number it was made at, to 3%).

## How it works

Three places decide the drag of a body in a stream, and each has its own
answer (`windtunnel.py`):

- **At the wall**, where the boundary layer forms and leaves. The wall is
  where the surface is, not on the voxel staircase: every lattice link
  that crosses the mesh is cut at the exact crossing, and the populations
  bounce back from there (interpolated bounce-back, Bouzidi et al. 2001).
  On the sphere at Re 100 and 16 cells that alone takes the error from
  7.5% to 3.4%.
- **In the layer next to the wall**. A laminar boundary layer is about
  3 d / sqrt(Re) thick where it leaves a bluff body, and the grid needs
  about four cells in it. Levels of cells, each half the size of the one
  outside it, hug the surface (`--layers`): cells go where the boundary
  layer is, not through the whole box round the body. On the sphere at
  Re 100 one layer reads the same drag as a full box of the fine cells to
  0.4%, at a fifth of the run time.
- **Far away**, where the stream must not know there is a tunnel. The
  grid coarsens by halves away from the body, so the far boundary sits 10
  body lengths to the sides and 24 behind, and the body blocks under 0.2%
  of the section (a strut that spans the tunnel: 1.2%), at a few percent
  of the run's cost.

The levels trade the fluid's state where they overlap (density, velocity,
viscous stress), interpolated cubically. That last word matters more than
it looks: at a tunnel's Mach number a 1% error in the velocity handed
across a level's edge is a 10% error in the pressure that balances it, and
linear interpolation near a curved body is 1% off. Collisions are
regularized, with a subgrid viscosity for what the grid cannot hold: WALE
on the level at the wall, because it vanishes in a laminar boundary layer,
and Smagorinsky on the levels away from it, because WALE also vanishes in
a plane shear layer (a strut's separated shear layer, crossing to a
coarser level with no subgrid viscosity at all, blew the run up). The
stress's isotropic part, which is sound and nothing air at these speeds
carries, is damped. The stream is
brought up over two passes of the body by accelerating all the fluid at
once (no pressure wave has to carry the news in from the far boundary),
with a passing side gust: a real stream is never exactly symmetric, and a
cylinder in a perfectly symmetric one never sheds; a strut that spans the
tunnel also gets a 2% ripple along its span, or its wake would stay
exactly two-dimensional, which past Re 200 no wake is. The force is the
momentum the fluid hands the wall, link by link, every step; the run
averages in windows until two agree to 1%, or, for a chaotic wake, until
the standard error of the mean is inside 1%.

## How far to trust it

- **The Reynolds number the grid resolves.** N cells across the part's
  thickness hold a boundary layer for Re up to about (0.75 N)^2: 324 on 24
  cells, 1296 with one layer, 5184 with two. Past that the run is made at
  the Reynolds number the grid resolves, with the physical dynamic
  pressure, and the report says so (`reynoldsRun`, `reScale`, `note`,
  `boundaryLayerCells`). What that costs depends on the shape, and the
  table above has one of each: a body whose flow leaves at a sharp rim in
  three dimensions (a disk, a cube, most brackets and canopies' edges)
  reads within 8%; a round body reads the drag of the lower Reynolds
  number, which for a sphere or cylinder between 1e3 and 2e5 is within
  about 20% of the physical one (0.40 to 0.47, 0.95 to 1.2); a long sharp
  strut reads 16% low. Below a run Reynolds number of 500 friction still
  adds to the drag (a cube reads 1.6 at Re 81 against 1.05), and the note
  says the reading is high: add layers. A drone arm in prop wash is at
  Re 5e3 and a canopy in cruise at 1e5, so on this machine those are
  capped runs: good to 10 to 20%, against 12% for the old solver on the
  easiest case there is.
- **Not in it**: the drag crisis (a smooth body past Re 2e5, where the
  boundary layer goes turbulent before it leaves and the drag falls by
  two thirds); a streamlined section's attached turbulent boundary layer;
  gusts; fluid-structure coupling. A wing section at a wing's Reynolds
  number wants `propeller.py`'s section polars, not this.
- **Run time** on four cores at about 15 million cell updates a second:
  a sphere or a bracket at the defaults (24 cells, Re to 324) 5 minutes;
  with the one layer a faster stream brings in, 7 to 35 minutes; a slice of
  a strut 10 to 40. The time goes as the cells across to the fourth power
  (the cells, and the steps to pass the body), which is why layers, not a
  finer box, are how a higher Reynolds number is bought.

That is the gap it closes in [NTOP-PARITY.md](NTOP-PARITY.md): nTop's
external CFD feeds loads to its structural blocks; this does the same for
an agent, in one command, with its error against published drag on the
page.

## A rotor blowing on the part

`rotors=[{center, axis, radius, thrust, hub}]` (`--rotors`) puts actuator
disks in the tunnel: each a pressure jump of thrust / area over its
annulus, on a grid level of its own (a sixteenth of its radius, so a disk
fifteen times wider than the arm under it stays affordable). With no
stream the tunnel opens on every side and runs along the rotor's axis:
hover. The report gives the air through each disk, the wake a diameter
behind it, and the load on the part along the rotor's axis as a share of
the thrust.

Checked so far: a free disk in still air. Momentum theory says a disk of
area A carrying thrust T passes v_i = sqrt(T / (2 rho A)); an 89 mm disk at
0.78 N (a 3.5 in drone's hover) passes 7.53 m/s against 7.38 (+2%), with
the jet a diameter down carrying the thrust as momentum flux, and a 40 mm
disk 7.83 against 7.84 (the test). Rotor runs keep the fluid's own
viscosity (a jet at a capped Reynolds number is a different jet).

Not validated: a duct round the disk. A long bell-mouthed duct (lip radius
12 mm, 60 mm long, no tip gap) should carry as much thrust as its rotor
(ideal theory); here it carries 46% of it, and the disk passes 10.7 m/s
against the ideal 14.8. The direction is right (the lip pulls, the flow
through the disk rises) and the size is not established: the lip's
suction peak is a few cells wide on this grid, and friction on the duct's
wall at this Reynolds number is real but unmeasured here. Treat a duct's
thrust from this as a lower bound, and `aero.duct_performance` (momentum
theory with a measured lip loss) as the design number, until a duct case
with published thrust is in the table above. An arm in a rotor's wash has
not been run coupled either (the disk needs a grid fifteen times the
arm's width); the spanning-strut rows above give the section's drag
coefficient that `rotor_blockage` below takes as `cd`.

## Structure in a rotor's stream

`aero.rotor_blockage(mesh, center, axis, radius, thrust, hub=, ducted=)` prices what a part costs a propeller or fan by
sitting in its air. The stream is followed along its own tubes, as momentum theory draws them: the speed on the axis
(v_i at the disk, 2 v_i far behind, falling off ahead) sets every tube's radius by continuity, so the wake narrows to
0.71 R behind an open rotor and the inflow widens ahead of it (`ducted` holds the stream at the disk's area and
sqrt(2) v_i, a duct with no diffuser). The first material each tube meets drags `cd q` on the tube's section there,
against the thrust; what lies behind it sits in its wake. It returns the blocked area, the drag, the share of thrust
lost and the carried mass that would cost a hovering craft as much (drag / 9.81). On a 3.5 in drone a 4 x 35 mm arm
10 mm under a prop at hover loses 0.9% of that motor's thrust (0.7 g of equivalent payload); at full throttle the wake
is 2.8 times faster and the drag 7.7 times larger.

The same model is a cost inside the optimizer. A study's `aero` block names the rotors:

```json
"aero": {"rotors": [{"name": "front left", "center": [52, 52, 12], "axis": "-z", "radius": 44.5, "hub": 14,
                     "thrust": 0.78, "ducted": true}]}
```

The design's drag in each stream is charged in the material budget as the mass that would weigh as much on a craft
hovering on these rotors, so a gram of thrust lost trades one for one against a gram of structure: the optimizer
takes an arm out of the stream, or thins it where the air is fastest, exactly when that is lighter overall. On a
density field a cell blocks its density cubed of a tube (as SIMP's stiffness: a grey cell is mostly air), which is
the first-hit rule exactly once the design is solid or void, and its gradient is checked against finite differences.
A plate from a wall to a tip load with a prop over its middle runs a beam straight under the prop when left alone
(5 g of thrust blocked on a 9 g part); charged for it, the optimizer splits the beam round the stream. The report
gives the delivered part's drag per rotor (`report.aero`) and `report.effectiveMass` (mass plus the drag's
equivalent), which is what the search for the lightest passing part compares. `thrust` is the operating point the
trade is weighed at: hover, for a craft that mostly hovers. `gramsPerNewton` changes the exchange rate for a fan
that is not holding a craft up.

What it leaves out: the stream is the actuator disk's (uniform over the disk, speed from the axis formula, no swirl,
no tip vortices), every member is a bluff body at one `cd` (1.2; a streamlined section is about 0.1 and the model
does not know the section), and a blocked rotor's own loss of efficiency is not counted.

## What a duct gives and costs

A duct round a rotor keeps the wake from narrowing, so the same power moves more air more slowly: momentum theory
gives a duct with no diffuser 2^(1/3) = 1.26 times an open rotor's thrust at equal induced power, with the rotor
carrying half of it and suction on the inlet lip the rest. Whether a small printed duct earns its grams is what it
loses of that. `aero.duct_performance(radius, thrust, tip_gap=, lip_radius=, length=, diffuser_angle=,
outer_radius=, height=, hub=)` takes the ideal and charges each loss as pressure the rotor must make up or flow it
must pump twice:

- the lip is a pipe entrance of that radius (`inlet_loss`: half a dynamic head at a sharp edge, 0.13 at a radius of
  7% of the bore, 0.03 at a bellmouth);
- the wall is pipe friction over the length behind the disk;
- a diffuser widens the exit only as far as its wall holds the flow (7 degrees of half-angle) and recovers 85% of
  the ideal;
- the tip gap leaks: the rotor's pressure rise drives a jet back through the gap's ring, air that is pumped and
  makes no thrust.

It returns the thrust against the same rotor in the open at equal induced power (`thrustRatio`, `thrustGained`),
against an open rotor as big as the duct's outside (`thrustRatioSameSize`: the same width spent on blade), the
thrust the duct's lip carries (`ductThrust`, which is a load on whatever holds the duct, along the thrust) and what
is left on the blades (`rotorThrust`). With `cruise=` (m/s, edgewise) it adds the drag: the momentum drag of
turning the swallowed air to the duct's axis, a bound a short duct does not reach, and the duct's outside as a
cylinder in cross flow.

A 3.5 in drone's duct (89 mm prop, 1 mm tip gap, 6.5 mm lip, 16 mm of wall behind the disk, 112 mm outside) at
0.78 N of hover: 1.125 times the open prop's thrust at equal power, of an ideal 1.26; the lip costs most (0.13 of a
dynamic head), the gap leaks 4% of the flow; 43% of the thrust is on the lip. Doubling the gap to 2 mm leaves 1.09,
a sharp lip 0.93: worse than no duct. And an open prop filling the duct's 112 mm outside makes 5% more thrust than
the ducted 89 mm one (`thrustRatioSameSize` 0.95): where a bigger prop fits, it beats the duct, and the duct's
case is what else it does (guarding the blades, carrying load). At 15 m/s of forward flight the same duct drags up
to 1.4 N, more than it lifts, with the cross flow at 1.4 times the disk speed, past where the windward lip stalls.

In a study, `aero.rotors[].duct` `{tipGap, lipRadius, lengthAbove, lengthBelow, diffuserAngle, outerRadius}` puts
this in the report (`report.aero.rotors[].duct`), with the rotor's `thrust` then the whole unit's. The gain is
credited in `report.effectiveMass` in the same currency as blockage (`thrustGained` x `gramsPerNewton`,
`report.aero.ductEquivalentMass`), so a ducted design and an open one compare on one number: structure, plus thrust
blocked, less thrust gained. `aero.cruiseSpeed` adds `report.aero.cruise`: the ducts' drag there and the thrust a
craft in level flight adds to hold it (thrust holds weight and drag at right angles, so sqrt(T^2 + D^2) - T). That
is reported and not charged: how much of a flight is cruise is the mission's to say. A drawn duct is the same in
every design the optimizer tries, so its credit is a constant there; it decides between studies, not within one.

What it leaves out: both rotors are ideal disks, so the open rotor's tip loss, which a duct's wall stops, is not
credited (`open_factor` takes it from a blade model: the ratio rises by its 2/3 power); profile power is in neither,
so at equal shaft power the gain is smaller by the share of power that is not induced; a gap wide enough for the tip
vortex to re-form (a few percent of the radius) loses more than the leak alone; a duct too short for its exit to
reach ambient pressure, and a lip stalled by cross wind, are not modelled. The numbers sit inside what small ducts
measure (Pereira 2008: gains from none to tens of percent, falling with the gap and the lip radius), but no duct of
ours has been on a thrust stand yet.
