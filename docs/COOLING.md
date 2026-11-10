# Cooling: interiors designed around the heat electronics make

An industrial designer hands over the exterior. The parts that go inside
are known, and some of them make heat: a processor, a radio, a charger
chip. Each has a temperature it must stay under, a battery has a lower one
still, and the outside has its own limit: the surface a hand holds must not
get hotter than is comfortable or safe to touch. Everything inside the
exterior is invisible, so it should be designed for exactly that, with the
least material, and together with whatever else the inside has to do (a
grip that does not creak, a push pad's click, a drop).

`loadpath cooling` does this. As with push pads ([SWITCH.md](SWITCH.md))
and feel targets ([FEEL.md](FEEL.md)) the exterior is kept as a thin
`skin`, the components are keep-outs (with their assembly paths), and
everything else is design space. A `cooling` block says what makes heat
and what must stay cool:

```json
"cooling": {
  "ambient": 30,
  "components": [
    {"region": "soc", "watts": 1.5, "maxTemp": 70},
    {"region": "pmic", "watts": 0.3, "maxTemp": 70},
    {"region": "board", "conductivity": 20},
    {"region": "battery", "conductivity": 1.5, "maxTemp": 45}
  ],
  "skin": {"contact": "held"}
}
```

```
python -m loadpath cooling handheld.study.json --out hh-out/
```

A component is a keep-out `region` (the part as drawn: a keep-out with an
`insertFrom` path is swept along it in the study, the part itself is not)
or its own `shape`. `watts` is the heat it gives off, spread over its
volume; `maxTemp` is what its hottest point must stay under; a component
with no watts (a board, a battery) only conducts. `conductivity` defaults
to 20 W/(m K), a package on a copper-filled board in plane; a lithium
pouch is 1-3 across its layers.

The skin limit is `skin.maxTemp` in C, or a contact class for plastic
surfaces after IEC 62368-1's TS1 limits: `continuous` 43 C (held over 8 h),
`held` 48 C (10 min to 8 h, a handheld product in use), `brief` 60 C (1 to
10 min). Metal surfaces have lower limits; check the edition the product is
certified to. `skin.shape` limits it to where the product is touched.

A study with a `cooling` block needs no supports or load cases. `feel` and
`switch` blocks in the same study are designed together with it.

## The model

Per grid cell, inside the exterior:

- **The printed material** (the skin and the designed interior) conducts
  at its card's conductivity (PA 12 0.25 W/(m K), PETG 0.20). A designed
  cell is k_air + x^3 (k - k_air): a void cell is air, not nothing. A
  polymer conducts only six to eight times better than air, so the air in
  the gaps carries a real share of the heat.
- **The air inside** conducts at `air.conductivity` (0.03 W/(m K)). In a
  handheld product the gaps are a few mm. Across a 5 mm gap at a 10 K
  difference the Rayleigh number is about 100, far under the ~1700 where
  natural convection starts in an enclosed layer: the air inside is close
  to still, and conduction is what moves heat through it.
- **The outside** loses heat to the room by natural convection and
  radiation together, `skin.h` (default 10 W/(m^2 K): about 5 of each for
  a matte surface a few tens of K over room temperature). This is the
  number that sets the skin's mean temperature: all the heat leaves through
  it, so the mean rise of the outside is about W / (h A) whatever is
  inside. The interior only decides how evenly.
- **Vents** (`vents`): openings the exterior already has, each a `shape`
  (and an `area`, or the outside faces it covers). Loadpath never cuts the
  exterior. With vents at different heights (`gravity`, default `-z`: the
  way down in use), room air is drawn through by buoyancy: Q = Cd A
  sqrt(2 g H dT / T) with Cd 0.6, the inlets and outlets in series, carrying
  rho c_p Q dT out, with dT the inside air's mean rise. This is a
  well-mixed estimate: good for whether vents help and by how much, not for
  where the draft goes.

The limits are read as a p-norm (p = 12) of the node temperatures, scaled
to the true maximum of the last solve, so the gradient is smooth and the
limit is the hot spot, not an average. They go to the same optimizer as
the push pads and feel targets (minimum volume, MMA, thin skins as cut
cells, coarse to fine). Each costs one solve of a scalar system per
iteration, factored once (PARDISO when installed) and shared by the
forward and adjoint solves. The adjoint gradients match central finite
differences to 1e-7 (tests/test_cooling.py).

**What the two limits trade.** Material run from a hot chip straight to
the nearest wall cools the chip, and makes a hot spot on the wall over it.
Spread under more of the skin, it cools the chip a little less and keeps
every patch of the outside cooler. A battery under the board wants the
board's heat to go anywhere but through it. The optimizer finds the
geometry that meets all of them with the least material.

**Not modelled.** Contact resistance between a component and a printed pad
on it (a thermal pad or paste is assumed: without one, a dry printed
contact adds 1-10 K per W on a small chip). On the delivered part, a gap
of up to `padGap` mm (default 1) between a component and the plastic is
read as that pad, so every grid reads it the same way: a gap under a cell
wide was otherwise air or contact by where the grid fell (the enclosure
speaker's amp read 57 C on two grids and 66 C on two others). Air between
two components is read as air on any grid, and each component is read on
its own nodes, not on those it shares with a neighbour across a gap. Also
not modelled: forced air, the hand as a heat
sink and sunlight. The plastic's softening with temperature is in the
`thermal` block, not here.

## Bursts, charging and duty cycles

A handheld product is rarely at a steady load. It idles, then runs a game
or a video call for a few minutes, or charges for an hour. Its parts and
its plastic hold heat, so a five-minute burst at twice the steady power can
peak lower than the steady state would. Designing for the steady state at
the burst's power asks for far more than the product needs, or for
something no interior can do. A `cycle` says how power changes over time:

```json
"cycle": {"phases": [
  {"seconds": 60,  "watts": {"soc": 0.3, "pmic": 0.1}, "steps": 1},
  {"seconds": 300, "watts": {"soc": 3.0, "pmic": 0.5}, "steps": 15},
  {"seconds": 300, "watts": {"soc": 0.3, "pmic": 0.1}, "steps": 10}]}
```

Each phase lists the watts that change (other components keep their
`watts`). The run starts from the steady state of the first phase (`start:
"ambient"` starts cold), or with `repeat: true` it is the repeating state
of the phases, run `cycles` times (default 3) from the steady state at the
mean power and read on the last. Every limit holds at each point's hottest
moment: the processor's peak at the end of the burst, the outside's a while
later, as the heat reaches it. The report gives each peak and when it
happens (`peakAt`, seconds).

The air outside can change with the phase too. A drone waits on the ground
with its video link on in still air, and is cooled by the stream only once
it flies; a handheld is put down, then carried. A phase's `airflow:
{speed, length}` (as `skin.airflow`; speed 0 is still air: natural
convection and radiation off the same run) or `h` is the outside's
coefficient for that phase, and the others keep the skin's:

```json
"cycle": {"phases": [
  {"seconds": 180, "watts": {"vtx": 4.0}, "airflow": {"speed": 0, "length": 60}},
  {"seconds": 240, "watts": {"vtx": 4.0}, "airflow": {"speed": 8, "length": 60}}]}
```

The start state sits in the first phase's air (with `repeat`, in the time
mean). A box at 0.5 W that waits ten minutes and then flies ten at 8 m/s
peaks at the end of the wait and ends the flight at under half of it; each
phase follows its own exponential to 3% of the exact answer.

The model is backward Euler on the same conduction system with a lumped
heat capacity per node: the material's density times its card's specific
heat in each designed cell (in proportion to its density, the rest is
air), each component's `heatCapacity` (J/K; default 1.8 J/(cm^3 K) times
its volume, about right for silicon, board and a lithium cell), air at
1.2 J/(L K). Each step length (and each phase's outside air) is factored once, and the adjoint runs back
through every step on the same factors, so the gradient of the hottest
moment is exact (tests/test_cooling.py checks it against finite
differences to 1e-3, and a near-isothermal box heating up against
W/(hA)(1 - e^(-t/tau)) to 1.5% at 40 steps).

## How it was checked

- **A heated sphere in a shell.** A 1 W sphere (4 mm) at the centre of a
  PA 12 ball (20 mm) losing heat at h has an exact answer:
  W/(4 pi k)(1/a - 1/b) + W/(4 pi b^2 h). This found a real error. A grid's
  stepped outside has |nx| + |ny| + |nz| times the area of the surface it
  follows, 1.5 times on a sphere, and the outside read a third too cool,
  at every grid size. Each outside face now convects over the true surface
  it stands for (1 / (|nx| + |ny| + |nz|) of a face, the normal from the
  exterior's distance field). The ball's outside now reads within 1% and
  the centre within 3% at 0.5 mm (`tests/test_cooling.py`). Flat faces are
  unchanged; the handheld's rounded edges read about 1 K warmer than
  before the fix, and the example below was designed again with it.
- **Heat balance.** All the heat leaves through the outside and the vents
  (to 1e-4 W without vents, 0.2% with them).
- **Gradients.** The adjoint matches central finite differences, steady
  and over a cycle.
- **Grids.** The delivered part is read on two finer grids than it was
  designed on (below).

## Checks

After the run the report gives every component's hottest point, the
touched skin's hot spot (and where it is) and mean, and where the heat
leaves (`heatOut`: the skin and the vents), each read three ways:

- `design`: the optimizer's own density field;
- `thresholded`: black and white, as printed;
- `delivered`: the delivered STL, read back on two finer grids (default
  0.7 and 0.5 of the design pitch; `verifyPitches`), each cell solid
  where at least half of it is inside the mesh. `gridChange` is how much
  the two differ, as a share of each temperature rise; quote a delivered
  reading where it is small.

Read the bare skin and a filled interior first (`examples/cooling/handheld.py
--baseline` shows how). If the filled interior still misses a limit, no
interior meets it: the exterior's area, vents, a metal spreader or less
power have to change, and those are the designer's to decide.

## Example: a handheld device

`examples/cooling/handheld.py`: a generic 120 x 64 x 16 mm handheld with
rounded edges, MJF PA 12, a 1.5 mm skin. A board at one end carries a
1.5 W processor and a 0.3 W power chip (both under 70 C); the battery
beside it must stay under 45 C; the outside under 48 C (held); a 25 C
room.

Read before anything is designed (`--baseline`, 1 mm cells):

| interior | processor | power chip | battery | skin hot spot | skin mean |
|---|---|---|---|---|---|
| bare skin (parts in still air) | 93.3 C (limit 70) | 89.1 C | 30.1 C | 51.4 C (limit 48) | 34.4 C |
| filled solid | 54.9 C | 50.0 C | 33.9 C | 47.1 C | 34.4 C |

Bare, the processor is 23 K over its limit and the skin over it is 3 K
over the touch limit. Filled solid, both are met, at 89.6 cm^3 of PA 12
inside. The skin's mean is the same either way: W / (h A).

Designed on 0.7 mm cells (90 iterations, 32 min on the 4-core cloud box,
`--pitches 0.7 --iterations 90`, aiming 1 K under each limit: `margin`),
the interior is 23% of the design space: pads off the processor and power
chip that spread into a thin liner under the skin over the board, and
almost nothing at the battery end. 43.5 g of PA 12 in all, skin included.

![handheld sections](images/cooling/handheld-sections.png)

| reading | processor | power chip | battery | skin hot spot |
|---|---|---|---|---|
| design (0.7 mm) | 69.0 C | 63.0 C | 35.2 C | 47.0 C |
| black and white (0.7 mm) | 68.3 C | 62.4 C | 35.2 C | 47.3 C |
| delivered STL, 0.49 mm | 63.6 C | 59.0 C | 33.5 C | 47.8 C |
| delivered STL, 0.35 mm | 65.1 C | 59.7 C | 34.8 C | 48.0 C |

Every limit is met on the delivered part. The two delivered grids agree
within 3.8% of each temperature rise on the processor and 0.8% on the skin
hot spot (the battery's rise is small, so its 1.3 K reads as 14%). The
skin reads 1 K warmer delivered than designed, at the foot of the pad
under the board on the back: the margin is that 1 K.

**Design on 0.7 mm, not 1 mm.** The same study designed on 1 mm cells met
its limits there and read 3-4 K hot at 0.5 mm: the processor is 1.2 mm
thick, the board 1.6 mm and the gaps around them under a cell, and a 1 mm
grid conducts across them more than they do. 0.5 and 0.35 mm agreed
within 1% on the processor.

**Combined with a structural target.** `--press` adds a feel target: the
back panel under a 20 N thumb press in its middle closes by no more than
0.1 mm. The bare skin closes 0.56 mm. The press never binds: the liner the
heat needs under the skin over the board is also what stiffens the panel,
to 0.023 mm on this design. A target that is met for free costs
nothing; it is there so a change to the layout that thins the liner cannot
quietly make the panel creak. `tests/test_cooling.py` has a case where the
grip does bind (a squeeze across a small box).

**Designed for a burst.** `--burst` designs the same handheld for how it
is used instead: idling at 0.4 W, five minutes at 3.5 W (a game, a call,
a recording), then idle again, the battery holding 45 J/K. Held steady at
3.5 W no interior meets the limits: the outside averages 16.5 K over the
room, and filled solid the hot spot reads 68 C. Over the burst a filled
interior peaks at 56.5 C on the processor and 43.5 C outside (1 mm), and
the bare skin at 108 C and 48.5 C, so the interior has work to do.
Designed on 0.7 mm cells (90 iterations, 3.1 h: each iteration is 26 time
steps forward and back):

| reading | processor (at 360 s) | power chip | battery | skin hot spot |
|---|---|---|---|---|
| design (0.7 mm) | 69.0 C | 57.7 C | 28.8 C | 46.9 C |
| black and white (0.7 mm) | 68.4 C | 57.1 C | 28.8 C | 48.3 C |
| delivered STL, 0.49 mm | 61.7 C | 52.4 C | 28.3 C | 47.1 C |
| delivered STL, 0.35 mm | 65.2 C | 54.3 C | 28.8 C | 49.3 C |

52.4 g, 38% of the design space: twice the power for five minutes takes
more material than the steady 1.8 W (43.5 g), not less, and the part
meets a load no steady design could. The finest read misses the touch
limit by 1.3 K, and the two delivered grids differ by 9% of the rise (3.8%
for the steady design), so this result is not settled: the delivered
reading on a third grid, and why 0.49 mm reads cool on both designs, are
the next things to check.


## Lessons from building it

- A part whose interior lines the whole inside of a closed skin encloses
  that interior as one cavity, and reconstruct used to drop the cavity's
  surface with the small bodies (decompose() gives a cavity a negative
  volume) and deliver the part solid. Fixed on main (32b8293); a closed
  exterior's delivered volume is worth a glance against the design's.
- The mesher's closing filled slots under 2 mm as too narrow to print. In
  a cooling part an air gap is the design: the handheld's mesh bridged a
  1.4 mm gap between a pad under the board and the back skin, 2 g more
  and the back 0.5 K over its limit. Cooling parts mesh without it.
- The touch limit is usually set by the exterior, not the interior: the
  skin's mean rise is W / (h A) whatever is inside, so at 1.8 W this
  21,200 mm^2 exterior averages about 8.5 K over the room, and the interior
  can only decide how far over that mean the hot spot sits.
