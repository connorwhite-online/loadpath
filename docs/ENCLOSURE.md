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
`examples/enclosure/render.py` draws the result.

![sensor puck: the exterior, the parts as placed, the delivered interior](images/enclosure/sensor-puck.png)

Nothing about where the parts go is in the brief beyond the rules. On
0.6 mm cells (the skin two cells thick) loadpath placed all four in three
minutes:

- **The board and its USB-C connector** turned so the connector's face
  points out of the port and sits on the port's axis, 0.31 mm short of the
  outer surface (0.3 asked). The recess is the face's own stand-off; the
  rest of the part keeps its margin from the walls, which the brief sets
  to 0.3 mm for the board (0.5 elsewhere).
- **The battery** went on the floor, straight over the door.
- **The motor** went up against the ceiling, 0.5 mm under the skin and
  directly over the board (4.7 mm above it), so its wires stay short.

Each part has a checked way in, in the assembly order (motor, board,
battery, door):

- The motor and the battery drop straight in through the door.
- The board, with the connector soldered on, comes out the way it went
  in. It slides 11.85 mm along +y to free the connector from the port,
  then drops out through the door. Its turn sits mid-way between the
  lengths that clear both rims.

Every path is swept as a keep-out, and no keep-out overlaps the skin by any
amount (`pathsIntoSkin` 0.0 mm³ for each part). The cradles went under the
board (-y), the battery (+y) and the motor (+z).

`run` then designed the interior for six 300 g drops, one onto each face,
and a 150 N squeeze across the width, out of an 84.9 g design space:
43.8 g at 30% of it, 34.1 g at 15%, and 29.8 g at 8.2%, the lightest that
passed before its 110-minute limit. `checks.json` reads the 29.8 g part
on 130,703 cut cells at 0.42 mm, not on the optimizer's grid, and
`examples/enclosure/tets.py` reads it again on 567k 10-node tets:

| check | result |
|---|---|
| exterior proud of the designer's surface | 0.0 mm (34,034 samples) |
| exterior surface still there | 100% |
| watertight STL | yes |
| material in any keep-out or way in | 0.000005 mm³ |
| safety factor, worst case: cut cells / tets | 1.9 (squeeze-y) / 4.2 (drop -x); goal 1.5 |
| drops -z / +z / -x / +x / -y / +y, cut cells | 2.7 / 4.8 / 3.6 / 3.6 / 3.8 / 3.4 |
| drops -z / +z / -x / +x / -y / +y, tets | 6.3 / 5.3 / 4.2 / 4.3 / 5.6 / 5.8 |
| squeeze-y: cut cells / tets | 1.9 / 5.2 |
| buckling factor | 6.7 |

The two readings agree that every case passes, but not on the margins.
They are furthest apart on the squeeze, whose load goes in through small
patches on the skin. The cut cells read the stress peak under a patch
higher, so they are the reading to design to.

The first runs of this example found five bugs, and all of them are fixed:

- **A rim notch.** The board's way out cleared the door's rim by 0.04 mm
  too little. Because a keep-out is cut from everything, that path cut
  7.9 mm³ out of the skin's rim.
- **Material back in the keep-outs.** The mesh cleanup's pinch filler put
  up to 18 mm³ back inside the ways in.
- **A leaky STL.** 14 sliver edges that only an STL reader's weld exposes
  made the file read as not watertight. The pass that removes them first
  crashed on an unwelded mesh, and now welds before it starts.
- **A recess doubled by the margin.** An aligned connector kept its
  0.5 mm margin from the exterior on top of the 0.3 mm recess asked, so it
  stood 0.82 mm back.
