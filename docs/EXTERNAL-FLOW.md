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
sitting in its air: each column of the disk is followed up and downstream to the first material it meets, and that
patch drags `cd q dA` against the thrust, with q from the actuator-disk speed at that distance (v_i at the disk, 2 v_i
far behind, falling off ahead; `ducted` holds it at sqrt(2) v_i, a duct with no diffuser). It returns the blocked
area, the drag, the share of thrust lost and the carried mass that would cost as much. On a 3.5 in drone a 4 x 35 mm
arm 10 mm under a prop at hover loses 1% of that motor's thrust (0.8 g of equivalent payload); at full throttle the
wake is 2.8 times faster and the drag 7.7 times larger.
