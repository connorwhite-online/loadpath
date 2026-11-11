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
