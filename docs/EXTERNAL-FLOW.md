# External flow: wind and water loads on a part

A drone arm, a fairing, a sensor mast on a vehicle or a bracket in a
current carries a load nobody writes down: the air or water pushing on it.
Topology optimizers take a force vector as given, and CFD packages stop at
a pressure plot. `loadpath external-flow` puts the part in a virtual wind
tunnel and hands back that load in the form the optimizer reads, a study
load case, so the flow and the structure are one step apart instead of two
tools apart.

```
loadpath external-flow arm.stl --speed 15 --direction -x --study arm.json
```

prints the drag, side force and lift (N), their coefficients on the
frontal area, the moment about the centroid, and writes `arm-flow.json`:
the study with a new load case, `flow-15ms`, of surface patches each
carrying its share of the pressure and friction. The Python call is
`loadpath.aero.external_flow(mesh, speed, direction, medium="air")`;
`medium` takes `air`, `water`, `glycol50`, `oil` or a `Fluid`.

## How it works

The same lattice Boltzmann kernels as the exchanger and lattice-cell flow
(`flow.py`: D3Q19, two-relaxation-time, Smagorinsky subgrid viscosity,
halfway bounce-back walls). The part is voxelized in a tunnel turned to
the stream (3 body lengths upstream, 6 downstream, 2.5 to the side, sides
periodic), the inlet is ramped up from rest, sponge layers at both ends
soak up the sound waves, and the force is summed by momentum exchange on
every wall link every step once the wake has developed (`--cells` sets
the voxels across the part's frontal size, 16 by default).

## How far to trust it

| Case | loadpath | Reference |
| --- | --- | --- |
| Sphere, Re 100, 12 cells | Cd 1.21 | 1.09 (Schiller-Naumann) |
| Sphere, Re 100, 18 cells | Cd 1.24 (smaller tunnel) | 1.09 |
| Cube face-on, run at Re 250, 10 cells | Cd 1.00 | about 1.0 to 1.1 |
| 10 x 30 mm plate face-on, run at Re 200, 8 cells | Cd 1.03 | about 1.1 to 1.2 |

It reads high on smooth bodies, the safe side for a structural load, and
most of that is the finite tunnel, not the grid. It is a coarse
large-eddy simulation, not a design-grade aerodynamics code:

- Above about 25 times the body length in voxels (and above `--re-max`,
  2e4 by default) the grid cannot hold the physical Reynolds number, so the
  run is made at the highest one it can hold, with the physical dynamic
  pressure. The report says so (`reScale`, `note`). Sharp-edged parts
  separate at their edges whatever the Reynolds number, so their drag
  barely moves. Smooth bodies get their subcritical drag, which is higher
  than the drag crisis would give: conservative for structure, wrong for
  a drag-reduction study.
- Gusts, vortex shedding loads (the force history is in the report for a
  look) and fluid-structure coupling are out.

That is the gap it closes in [NTOP-PARITY.md](NTOP-PARITY.md): nTop's
external CFD feeds loads to its structural blocks; this does the same for
an agent, in one command, and says how far to trust it.

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
