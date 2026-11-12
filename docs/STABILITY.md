# Desk stability

`loadpath stability brief.json [--out report.json]` (Python:
`loadpath.stability.run_stability`) answers whether a product set loose on
a desk stays put when it is pressed, tugged by its cable, nudged sideways
at the plug, or has a plug pushed in. It is statics of a rigid body on
Coulomb friction; the method is in the module's docstring.

It reports, for each case, a slide margin and a tip margin (1 at rest, 0
on the point of moving, below 0 moved), and `minMass_g`: the least mass
that would hold, with the centre of mass where it is. That is the number
to weigh ballast against. Exit code 0 when every press, tug and lever
holds (plugging in is reported, not required: weight almost never holds a
plug being pushed home, and the report says to hold the product).

## The brief

```json
{
  "parts": [
    {"id": "body", "mesh": "body.stl", "material": "tpu88a"},
    {"id": "ballast", "mesh": "ballast.stl", "material": "316l"},
    {"id": "electronics", "mass": 0.002, "at": [0, 16, 5]}
  ],
  "feet": {"mesh": "body.stl", "tol": 0.05},
  "friction": {"pair": "tpu-wood"},
  "envelope_cm3": 29.0,
  "presses": {"points": [[0, 0, 9.6]], "force": 5.0, "cone": 25.0},
  "cables": [{"id": "usb-c", "at": [0, 28.6, 6.6], "direction": [0, 0.96, -0.29],
              "force": 0.3, "lever": 0.3, "plug": "usb-c"}]
}
```

- `parts`: a mesh with a `material` (any loadpath material) or a
  `density` (g/cm3), or a `mass` (kg) at a point `at` (mm).
- `feet`: `points` [[x, y], ...] for discrete feet, or a `mesh` whose
  downward faces within `tol` mm of its lowest point are the contact.
- `friction`: a number, or `{"pair": ...}` from `FRICTION` (tpu-wood 0.7,
  tpu-laminate 0.6, tpu-glass 0.8, tpu-mat 0.9, silicone-wood 0.9, hard
  plastics and metal on wood 0.3). Left out: 0.5, the low end of TPU on a
  desk.
- `envelope_cm3`: the closed exterior's volume. The report then gives the
  apparent density: under about 1 g/cm3 reads light in the hand (it would
  float), solid rubber is 1.1 to 1.2, a phone about 2.
- `presses`: points on the surface, the force in N and the cone the push
  may lean within.
- `cables`: `force` N along `direction` (the tug), `lever` N square to it
  at the plug (the twist), and `plug` (a connector name from
  `connectors.py`, or the insertion force in N).

## Making it heavier: massfill

`loadpath massfill brief.json --out fill.stl` (Python:
`loadpath.massfill.run_massfill`) fills the room a product doesn't need
with its own material (or any other), bottom up, and reports the most it
can reach (`maxFill_g`) and, with a `target_g`, whether it gets there.
Room is everything inside the exterior that is:

- off each keep-out by its `clearance` (and, with `noOverhang`, never over
  its plan above its base, so the part still drops in from above);
- `margin` below the skin's travel envelope (`travel.envelope`, an npz of
  `nodes` and `travel` from `readback(..., travel_at=N)`): the room the skin
  never reaches when pressed at N;
- over `floor`, if given.

With `moving` (a flap on a hinge: `axis` {point, direction}, `against` the
fixed part, `parts` that move, `fuseTo` the moving part, `clearance`,
`angles`), the room splits in two: fill that rides with the moving part
(its whole swing clears the fixed part) and fill fused to the fixed part
(clear of everything the moving side sweeps, the moving fill included,
checked again exactly on the meshes). Each fill reaches about a voxel into
what it is fused to: union it with that part. The fills are opened to
`minSection` mm (1.0 by default), meshed and smoothed, then cut exactly off
the keep-outs.

```json
{"exterior": "closed.stl", "floor": 2.7, "pitch": 0.25, "material": "tpu88a",
 "keepouts": [{"id": "board", "mesh": "board.stl", "clearance": 0.15, "noOverhang": true},
              {"id": "switch", "mesh": "switch.stl", "clearance": 0.3, "noOverhang": true},
              {"id": "wires", "mesh": "wires.stl", "clearance": 0.3}],
 "travel": {"envelope": "rb.env.npz", "margin": 0.5},
 "moving": {"against": "body.stl", "parts": ["flap.stl", "board.stl", "switch.stl"], "fuseTo": "flap.stl",
            "axis": {"point": [0, 28.54, 3.45], "direction": [0.996, -0.085, 0]}, "clearance": 0.3},
 "target_g": 30, "otherMass_g": 12.4, "out": "fill.stl"}
```

A fill fused to a wall stiffens the wall: read the feel back on the filled
part before trusting it (the travel margin keeps the fill off the moving
skin, not off the walls that hold it).

## Finding room for ballast

The switch read-back (`switch_lattice.readback(..., travel_at=N)`) also
returns `travelEnvelope`: how far each node of the part moves along the
press axis, over all presses, each at `travel_at` N or its fire force if
that is higher, with the switch taken as a stop once it has clicked. Room
under the skin that is clear of `node z - travel` plus a margin is room the
skin never reaches, and ballast can fill it without touching the feel.
The touch-mic's round 20 does this (touch-mic `hardware/r20/ballast_room.py`).

## What it found on the touch-mic

A 30 g pad on TPU feet: presses anywhere hold (a push inside the feet can
never tip or slide it). A 0.3 N cable tug needs 50 g at mu 0.5 and 33 g on
wood. A 0.3 N sideways nudge at the plug needs about 90 to 130 g, more than
the pad can hold at its size, so the cable's stiffness matters more than
the ballast there.

## Finishing the inside: seats

`loadpath.seats.finish_seats(part, keepouts, region)` reworks the inside
of a part round the components it holds, so it reads as designed rather
than as clearance boxes. On a fine signed-distance grid (`pitch`, 0.1 mm):

- a closing of radius `fillet` rounds every inside corner, where a rib,
  fin or block meets the floor it stands on (the root of a stiffener is
  where it wants material);
- each keep-out is cut back by true distance at its `clearance`, so a
  pocket is the component grown by a ball and its own corners are round;
  `noOverhang` keeps its plan clear above its base; `leadIn` (deg) flares
  the walls over `leadHeight` mm (1.0) from `leadFrom` mm above its base so
  it drops in by itself, then the wall rises straight;
- an opening of radius `round_` softens the outside edges (rims, the tops
  of walls);
- the result is meshed (marching cubes, a `smooth`-voxel blur and Taubin
  smoothing), taken inside `region` only and joined to the exact part
  outside it, and the keep-outs are cut again exactly, so the blur cannot
  loosen a seat.

Keep the region off mating faces (a hinge strap, a bead and rebate): they
stay exact. A lead-in eats the wall beside the part by
`leadHeight * tan(leadIn)`: where that wall is a stop under a soft skin,
leave the lead-in off and let the rounded rim do.

**Read the feel back after finishing**, at the pitch the unfinished part
was read at. A finished part's marching-cubes surface meets the exact part
along the region's faces; keep those faces off the part's own surfaces
(start the region inside a floor, not on its top face), since a cut plane
that grazes a face leaves back-to-back sheets of no thickness. The
read-back's voxelizer counts those as two crossings since this round (it
had merged them as one shared edge and turned whole columns over, which
read as a hole in the skin).
