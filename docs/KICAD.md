# A real board in: KiCad

An agent can now design a circuit board end to end. What it cannot do is
decide what holds the board: where it sits in the product, how it gets in,
what it is screwed to, what a USB plug does to it every day, and where its
heat goes. Topology optimization codes stop at a density field and know
nothing of boards; script and AI CAD draw a box round the board and know
nothing of loads. loadpath is the layer between, and this page is the
door in from the electronics side: **Claude can design the board;
loadpath designs the part it lives in.**

A board is not a box. Its outline is whatever was cut, it is as thick as
its stackup, its parts stand up from both faces to their own heights, it
is held by screws through its mounting holes, its connectors face out over
particular edges and get pushed on, and some of its parts make heat. All
of that is already in the `.kicad_pcb` (and the 3D models it points to),
so none of it is typed in again or guessed:

```
python -m loadpath kicad board.kicad_pcb --out board/ [--models DIR] [--power U3=0.6]
```

writes `board.json` (everything read, and where each number came from),
the board with its parts as one STL, `components.json` (what an enclosure
brief takes) and `stack-board.json` (what a stack takes). Usually there is
no separate step: name the file where the board goes.

In an enclosure brief ([ENCLOSURE.md](ENCLOSURE.md)) the board is a
component:

```json
{"id": "board", "rotate": "z", "margin": 0.3,
 "kicad": {"file": "board.kicad_pcb",
           "connectors": {"J1": {"opening": "usb-port"}},
           "power": {"U3": 0.6}, "maxTemp": {"U3": 85}}}
```

and in a stack brief ([WIRES.md](WIRES.md), `loadpath stack`) a board is
`{"id": "fc", "kicad": "fc.kicad_pcb"}`. MCP: `loadpath_kicad` reads a
board and returns the same summary.

## What is read, and from where

Every part in `board.json` says where its height, mass and heat came from
(`heightFrom`, `massFrom`, `wattsFrom`), so an agent can see which numbers
are the board's and which are estimates.

| | from |
|---|---|
| outline | Edge.Cuts lines, arcs, circles, rectangles and polygons (on the board or inside footprints), chained into loops; the largest is the board, loops inside it are cut-outs |
| thickness | the stackup's copper and dielectric layers, else `general.thickness` |
| board mass | FR-4 (1850 kg/m³) for the dielectric, copper (8960 kg/m³) at `copperFill` of each layer (default 0.5) |
| board conductivity | copper (390 W/(m K)) and FR-4 (0.8) in parallel by thickness: the in-plane value a cooling block spreads heat with |
| part body | its 3D model when the file is found (STEP with the `step` extra, or VRML), placed exactly as KiCad's 3D viewer places it, its solids as modelled (each open sheet its own hull), less what is through the board; else its fab outline (else courtyard, else pads) stood up to its height |
| part height | the brief's `heights`, else a `Height` field on the footprint, else the 3D model, else the package's standard maximum seated height read off its library name (JEDEC SOIC, SSOP, TSSOP, MSOP, QFN, QFP, PLCC, DIP, SOT, TO outlines; MLCC and chip-resistor thicknesses by size; axial bodies by diameter; 2.54 mm headers; USB-C, micro-B, USB-A, JST SH/PH/XH; ESP32 modules; `kicad._HEIGHTS` lists each with its source), else `defaultHeight` (3 mm), listed as assumed |
| leads | through-hole pins stand `leadProtrusion` (1.5 mm; IPC-A-610 allows up to 2.5) out of the far face |
| part mass | a `Mass` field (g), else body volume at its family's effective density (moulded ICs 2.0 g/cm³, MLCCs 5.5, resistors 3.5, electrolytics 1.25, connectors 1.2): estimates, labelled so |
| mounting holes | MountingHole footprints (and a lone large drill on an H reference): centre, drill, copper pad, screw size (from the name, else the largest ISO size through the drill) |
| connectors | Connector and TerminalBlock footprints and J/P references; which way each faces (below) |
| heat | a `Power` field (W or mW; also `Pd`, `Dissipation`, `Watts`) and `MaxTemp`, or the brief's `power` and `maxTemp` maps. A board file carries no power numbers, so none are assumed: a board with none says so |

Model files are looked for where KiCad looks (`${KICAD*_3DMODEL_DIR}`,
`${KIPRJMOD}`), in `models` folders given, and in a KiCad install's own
library (macOS, Linux, Windows); `.wrl` and `.step` stand in for each
other, since KiCad's library ships both. Checked against KiCad's own
library: the GCT USB4085 receptacle's STEP model lands on its footprint's
fab outline to 0.01 mm and stands 3.45 mm off the board.

**Which way a connector faces.** A connector that overhangs a board edge,
or sits within `edgeReach` (2 mm) of one, faces out along that edge's
outward normal; a vertical one (by its name) faces up off its face; a
horizontal one in the middle of the board is taken to face from its
contacts toward the far side of its body, and that is flagged as a guess
(`connectors.J4.out` settles it). The mouth is the body's outermost point
that way, at mid height.

## What it becomes in an enclosure

- **The board**, as one component: the outline less its cut-outs and
  holes, with every part that stays on it, at the mass of the laminate
  plus those parts and the in-plane conductivity above.
- **Each connector and each part that makes heat** (and any in
  `separate`) as its own component `installedOn` the board, in the
  board's frame: so a connector can be aligned to an opening (`alignTo`,
  from `connectors.<ref>.opening`) and a cable can leave from it (`ports`:
  `mate` at the mouth, the way it faces), and heat sits where the part is.
- **Screws, not a cradle.** A board with mounting holes is held on them
  (`mount.type: holes`): under each hole a threaded sleeve the board's
  pad bears on (tap drill 0.84 d, wall max(1.5 mm, 0.35 d), 2 d of
  thread, the same rules as `fasteners`), and kept clear of material the
  screw's shank through the board, its head (ISO 4762, d high) and the
  driver's line from the head to the skin. The sleeves go on the board's
  underside unless the brief's `side` is nearer its top or something
  below is in the way (another part, or its way in); a board with no
  holes falls back to a cradle. The optimizer grows what carries the
  sleeves to the skin: the drop cases put the board's mass (laminate,
  parts, connectors) through them, as a real board's inertia goes.
- **Plug loads.** A USB-C, USB-A or micro-B connector aligned to an opening
  gets its plug cases (connectors.py: 20 N in, 20 N out, 8 N and 20 N
  sideways at 15 mm for USB-C), through the screws' sleeves, with the
  board's travel capped so the solder joints are not flexed.
  `plugLoads: false` turns them off; any number overrides.

## What it becomes in a stack

`size` (the outline's bounding box and the stackup thickness), `mass`,
`watts`, `maxTemp` (the lowest any part allows: one temperature per board
is what the stack model has), `above` and `below` (the tallest part or
lead on each face) and `ports` (each connector's mouth and facing, in the
board's frame), with `standoffs.count` from the mounting holes when the
brief does not say. Anything the entry says itself wins.

## A worked board

`tests/data/kicad/sensor-board.kicad_pcb` is a 40 x 28 mm board made of
KiCad library footprints: four M2.5 holes, a USB-C receptacle over the -y
edge, a QFN with `Power 350mW` and `MaxTemp 85` fields, an SOT-223
regulator, 0805 capacitors (one on the back), an electrolytic with a 3D
model, a JST SH lead on the +y edge and a 2.54 mm header. Read: 1.58 mm
stackup, 3.4 g of laminate and 4.7 g with its parts, every height from a
package rule or the model (none assumed), the USB-C facing -y with its
mouth 2.51 mm past the edge, the JST facing +y, the header up.

Put in a 56 x 42 x 22 mm box with a USB-C slot in the -y end and a lid
on top (`tests/test_kicad.py`), the board is placed with the receptacle
0.31 mm short of the outside (brief: 0.3), goes in from the top and slides
into the port, and is held on four M2.5 sleeves under its holes,
with the drop and the four USB-C plug cases going through them.

## Limits

- The body of a part with a model is its model's solids, not the hull of
  the whole model (a USB-C socket's hull took in its tabs and legs, and
  its keep-out and way in came out a squared block). A solid the model
  leaves open (a sheet, not a closed shell) stands as its own convex hull,
  so the notches inside one such sheet are still kept clear. A mid-mount
  socket's shell below the board, in its slot or past the edge, is kept.
- No power is guessed. Heat is only what a field or the brief says.
- Part masses without a `Mass` field are envelope estimates; on most
  boards the battery and the laminate dominate a drop case anyway.
- Hole sleeves assume thread-forming screws into the print (or an
  insert: give `threadLength`); standoffs and nuts are not drawn.
- KiCad 6 to 10 files; KiCad 5's `module` and `fp_text` forms are read,
  but its 3D model offsets in inches are the only unit difference handled.
