# Products from the outside in

An industrial designer can stop at the outside. They draw the shape and
the surfaces people see and touch, and list what has to go inside: a
battery, a board and its connector, a motor, a switch. Where those parts
sit, how they get in, what holds them, how the product survives a fall and
how it feels in the hand is all invisible, and it is the work between a
rendering and a product that can be built. `loadpath enclosure` does that
work from one brief, and never touches the exterior.

```
python -m loadpath enclosure brief.json --out out/ [--run]
```

The brief (`loadpath schema enclosure`) is the exterior, its openings,
the components and what the product has to survive:

```json
{
  "exterior": "exterior.stl", "skin": 1.2,
  "openings": [{"id": "usb-port", "shape": "port.stl", "inward": "+y"},
               {"id": "battery-door", "shape": "door-opening.stl", "inward": "+z"}],
  "components": [
    {"id": "board", "shape": "board.stl", "mass": 0.008, "rotate": "z"},
    {"id": "usbc", "shape": "usbc.stl", "installedOn": "board",
     "alignTo": {"opening": "usb-port", "face": "-y", "recess": 0.3}},
    {"id": "battery", "size": [45, 30, 6], "mass": 0.032, "against": "-z", "rotate": "z"},
    {"id": "motor", "shape": {"type": "cylinder", "center": [0, 0, 0], "axis": "z", "length": 3.2, "r": 5},
     "mass": 0.002, "near": ["board"], "against": "+z"},
    {"id": "door", "shape": "door.stl", "place": "given", "external": true, "mount": false}],
  "assembly": ["motor", "board", "battery", "door"],
  "loads": {"drop": {"g": 300}, "squeeze": {"axis": "y", "force": 150}},
  "material": "pa12", "process": "sls", "minMember": 1.2, "goal": {"safetyFactor": 1.5}
}
```

## What it does

**1. It places the parts** (`placement.py`), unless the designer drew
them. Each part, or a group (a board with the connector soldered to it,
drawn in the board's frame), is tried in every orientation it may take
(`rotate`: all 24 quarter turns, the four about one axis, or none) at
every position on a grid through the cavity: the exterior less its skin,
plus the openings a part may reach into. For one orientation every
position is tested at once: the cells a part may not enter (the skin, the
designer's fixed geometry, parts already placed, each grown by the
`margin`) are cross-correlated with the part's own cells by FFT, and a
zero is a fit. Among the fits the rules choose:

| rule | what it does |
|---|---|
| `against: "-z"` | pushed to that side of the cavity: a battery on the floor |
| `alignTo: {opening, face, recess}` | that face of the part out through the opening, on its axis, `recess` mm short of the exterior: a connector in its port |
| `under: {at, direction, gap}` | that face on the line through `at`, `gap` under the skin: a switch under the spot a finger presses |
| `near: [ids]` | close to other parts: short wires |
| `placement.centerOfMass` | the layout's centre of mass toward a point, or low along an axis |

The parts go in one at a time, the most constrained first, as a beam
search that keeps the best few layouts, so an early choice that boxes a
later part in can be undone. Each position the grid finds is pushed the
rest of the way along its rule and checked on exact distances at dense
samples of the part's surface: the room reported per part
(`roomToSkin`, `roomToParts`) is measured, not the grid's.

**2. It keeps only what can be assembled.** A layout nobody can put
together is no layout. In the `assembly` order, a position is kept only
if the part can still get there past the skin, the fixed geometry and the
parts that went in before it. The way in is found in configuration space:
every translation of the part tested at once (FFT again), then flooded
outward from where it sits one straight leg at a time, so the first path
found has the fewest legs (up to four: a board that lifts off its rest,
slides its connector out of the port, then drops out through the door).
Each path is then walked exactly at the part's surface points, and if the
grid was half a cell too generous it is searched again tighter. The path
becomes the part's keep-out (`insertFrom`), swept the whole way, so no
generated material can block it.

**3. It mounts what has mass.** Each part with a mass gets a preserved
cradle under one face: the side given (`mount.side`), else the side it is
pushed against, else the side it comes to rest on as it goes in, else its
nearest wall; and never a side that sits in another part's way in. The
cradles are printed, so every path is checked again with them in place.

**4. It writes the loads.** `drop: {g, faces}` adds a case per face the
product can land on: every part's mass at `g` toward that face, carried
through its cradle (with the moment of its offset centre of mass), and the
floor's equal and opposite reaction on that face of the exterior. `g` is
the design deceleration you choose (a few hundred for a hard-floor drop of
a small rigid product; `loadpath feel` designs drop cushioning when the
interior should soften the fall). `squeeze` presses two patches of the
exterior together across an axis; `press` pushes a patch with the far face
held. Load cases, a `switch` block (a push pad; see SWITCH.md, the
plunger and its seat read off the placed parts) and a `feel` block pass
through. Component `watts` go into the study's `thermal` block when the
brief has one.

**5. It writes an ordinary study**: the exterior as its `skin` (preserved,
never edited, with no skin across the openings), the whole interior as
design space, components as keep-outs with their paths, cradles as
preserves, supports on a small patch of the rest face (the drop and
squeeze cases balance themselves; the patch only holds the part still).
`out/cad/` has every placed part as an STL in product coordinates and
`out/enclosure.json` says how each value was found.

**6. With `--run` it designs and checks it.** `run` designs for strength
and stiffness (`switch` or `feel` when those blocks are there), then
`out/checks.json` reads the delivered part the way the designer and the
engineer would:

- **the exterior untouched**: points sampled on the delivered surface,
  each one's exact distance to the exterior mesh (nothing proud of it),
  and points sampled on the exterior, each one's distance to the delivered
  surface (all of it still there, openings aside);
- **every load case on the delivered part**: `verify` on cut cells (or
  tets where installed), not the optimizer's grid, with the safety factor
  per case and no part's keep-out or path intruded on.

## Split shells, cooling, any exterior

A closed exterior with no openings is assembled with its cover off: give
the brief a `split` block and the study is handed to `shells.py`, which
picks or takes the parting line, finds the paths into the base, and adds
the lips, bosses and snaps (one study per half). A `cooling` block
(components' `watts` and `maxTemp`, skin touch-temperature limits, vent
sizing) passes through to `cooling.py`. The exterior can be any mesh,
STEP, a generated mesh that needs repair, or an SDF: `handoff`'s intake
(`exterior.py`) cleans it first.

## The example

`examples/enclosure/sensor_puck.py` writes a generic handheld sensor puck
(an 84 x 56 x 24 mm pebble with a USB-C port at one end and a battery door
underneath), its parts and the brief above, then runs it.

