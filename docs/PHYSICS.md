# Physics: designing around the real world

loadpath sits between a topology optimizer, which stops at a density
field, and scripted or AI CAD, which draws shapes without knowing the
loads. What it adds is the part of engineering that is physics: what the
material does when it is pulled, squeezed, held, cycled and warmed; how a
printer lays it down; how heat actually leaves a surface. A part designed
by an agent is only as real as those models. So the standing rule for the
library is:

> Question everything as physics. Nothing is a box, a constant or a rule
> of thumb because that was convenient; every such choice is a claim about
> the real world, and it is either the best representation we can make or
> it is on the list below.

This page is one full pass of that rule over the library (2026-11-10). It
builds on [the assumption audit](assumption-audit.md), which asked "what
would be 10x better?" of every stage; this one asks a narrower question of
every default, boundary condition, convenient shape and check: **is this
what happens to a real part?** Findings are ranked by how much they distort
real parts, the top of the list is fixed in code in the same pass, and
`bench/physics_audit.py` shows each fix as a before and after.

## How to apply the lens

For any number, shape or check, in a module or in a study:

1. **Say what it claims.** "h = 8 W/(m² K)" claims that heat leaves every
   face of every part at the same rate whatever its temperature, direction
   and surroundings.
2. **Find what decides it in the world.** Buoyancy, radiation, the gap to
   the next wall.
3. **Ask whether its inputs are already known.** Usually they are: the
   surface temperature is being solved for, the geometry is on the grid.
   Then the number is a result, not an input.
4. **Ask what a wrong value does.** Which way does it push a design, by how
   much, and would anyone notice before a part breaks or a product runs
   hot?
5. **If it stays a guess, say so in the report**, with its source and a
   quality (`measured`, `assumed`, `rough`), a way to override it, and the
   coupon or measurement that would replace it.

A check that reads a proxy (von Mises for "breaks", area for "cools",
"passes once" for "lasts") gets the same five questions.

## What this pass found and fixed

Seven things, in the order they distort real parts. Numbers are from
`bench/physics_audit.py` (`bench/results/physics_audit.json`) on the
4-core cloud box.

### 1. Layers fail in shear too, and squeezing is not pulling (`failure.py`)

The static check read von Mises stress against the tensile strength, and
on a layered part one more thing: the tension normal to the layers against
the bond's strength. That is a metal's yield criterion plus half of a
delamination criterion.

- **The bond.** A layer bond is slid apart by shear in its plane just as
  it is pulled apart by tension across it. A boss twisted about its build
  axis, a tab sheared across its layers or a short cantilever printed
  standing loads the bonds almost purely in shear, and the tension-only
  check passed them at the in-plane strength. The bond is now read with
  the quadratic interaction used for any weak plane,
  `(σn⁺/Z)² + (τ/S)² = 1`, with `Z = interlayer × strength` and the bond's
  shear strength `S` defaulting to `Z/√3` (`process.interlayerShear`
  overrides it). Each case reports `interlayerTension`, `interlayerShear`,
  where the bond is worst and which of the bond or the in-plane material
  limits it.
- **Pressure.** A polymer yields later squeezed than pulled, and earlier
  when pulled three ways at once, which von Mises reads as no stress at
  all. The in-plane measure is now the pressure-modified von Mises stress
  (Raghava, Caddell and Yeh 1973) with the material's compressive over
  tensile strength `m`: 1.2 for glassy printed polymers, 1.15 for nylons
  and polypropylene, 1 for metals and elastomers
  (`material.compressionRatio` overrides it). Bending-dominated parts read
  the same as before (their tension side governs); struts and seats in
  compression read up to 17% lower, notch roots and bosses in triaxial
  tension higher.

A 16 mm PETG boss printed standing, 40 mm tall, held at its foot:

| Load at the top | Safety factor before | Now | Limited by |
| --- | --- | --- | --- |
| Twist, 3 N·m | 3.27 | 2.29 | layer bond (shear) |
| Pull, 800 N | 3.36 | 3.36 | layer bond (tension) |
| Push, 800 N | 7.98 | 9.69 | in-plane |
| Side, 60 N | 2.98 | 2.98 | layer bond (bending tension) |

The twisted boss was passed 43% too strong. The tet cross-check and
CalculiX read the same two criteria (`crosscheck.py`), so the confirm step
and the voxel reading still agree on what failure means.

Both ratios are `assumed`: datasheets rarely give compressive strength or
interlayer shear. `report["failure"]` names the criterion, the ratios and
their source. See "What would replace the guesses" below.

### 2. A force over a region is a uniform traction (`domain.py`, `crosscheck.py`)

A load is given as "N, total over the region". It was shared equally
between the region's nodes. On a voxel face a corner node stands for a
quarter of an interior node's area and an edge node for half, so equal
shares put 4x the traction on the corners and 2x on the edges of every
loaded face; on a tet mesh they put more force wherever the mesher made
smaller triangles. The contact band (stress near loads and supports is left
out of the peak) hid most of it, and the rest read as stress the real part
does not have: the pulled boss above read 7.4 MPa for a 4.0 MPa pull
before, 5.2 MPa now.

Each node now takes its own share of the loaded surface (the cells it
touches on the voxel grid, a third of each loaded triangle on tets), for
forces, the couple that carries a moment, bearing loads and loads of
unknown direction. A bar pulled on its end face reads one stress to the
last node.

### 3. Loads nobody timed (`durability.envelope`, `report["serviceEnvelope"]`)

A static check reads every load as applied once, briefly, at 23 °C. Real
loads are held (a shelf bracket, for years), repeated (a latch, daily) and
warm (a dashboard in the sun), whether or not the study says so, and
plastics are poor at all three. loadpath had the models (fatigue curves,
creep rupture, strength against temperature) but applied them only to
cases that declared `cycles`, `hours` or `temperature`.

Every verification now turns each case's stress into the three limits the
same material data implies, at no extra solve: the hours the load can be
held before creep rupture, the number of times it can be applied before
fatigue, and the temperature at which the short-term margin is gone. They
go in `serviceEnvelope` (per case and the worst), and when a case declares
nothing the delivery's `attention` list says so in one sentence, for
example:

> twist: no duration or repeat count given, so checked as applied once,
> briefly, at room temperature. As delivered the material data puts the
> limits at: held about 5 years; applied about 610 times; up to about
> 70 C (each alone, at a safety factor of 1; assumed data).

That sentence is the finding. A PETG part that passes its static check at
a safety factor of 2.3 survives about 600 applications of that load on the
design curve (Ezeh and Susmel's for PLA, scaled by strength: 90% survival,
so most parts last longer, but it is what the data supports). Nothing was
wrong with the static check; the question it answered was not the one a
latch asks. Declaring the case's `cycles` makes the durability check hold
it, and `durability.constrain` sizes the part for it.

### 4. A printed part with no build direction is still printed in layers (`report["failure"]["layers"]`)

A study without a `process` (or with the direction left to `auto` until
orientation runs) was checked as an isotropic solid. PETG is only ever
printed in layers whose bonds hold about 38% of the in-plane strength
(Prusament's interlayer test); most hand-written eval cases name no
process, so their safety factors are the solid material's.

Verification now reads, from the same stresses, what the bonds would carry
for layers stacked along x, y and z (`ifBuiltAlong`), with the material's
own bond ratio, and `attention` gets a sentence when any direction is
below the reported safety factor. The same boss, pulled, with no process:
safety factor 7.8 as a solid; 3.3 if the layers stack along z. Declaring
`process.buildDirection` (or `"auto"`) designs for it, as before.

### 5. Still air is a result, not an input (`convection.py`)

A `thermal` block that named no convection got 8 W/(m² K) on every face.
In still air heat leaves by buoyancy and by radiation, and both depend on
what that number ignores: the surface's temperature, which way it faces,
how tall it is, what it is made of (a printed polymer radiates as much as
it convects; bare aluminium barely radiates), and what it looks at. Two
fins closer than their boundary layers share the same air, and a surface
facing another surface of the same part radiates to it, not to the room.

`"convection": "natural"` is now the default: per face, a plate
correlation by direction (Churchill and Chu for vertical faces, McAdams
for faces looking up or down) with air's properties at the film
temperature, capped by the parallel-plate channel correlation (Bar-Cohen
and Rohsenow) at the gap the face looks into, plus linearized radiation at
the material's emissivity times the face's view factor to the room (nine
rays through the voxel grid). The solve repeats with its own temperatures
until the coefficients settle (two to four passes). A number or a list
still fixes h for a fan, a liquid or a measured value.

A 40 × 40 mm aluminium plate, 1.5 W into its underside, plate fins 20 mm
tall, hottest point:

| Fins | Surface mm² | Fixed 8 W/(m² K) | Still air | Mean h | View of the room |
| --- | --- | --- | --- | --- | --- |
| 0 | 3,840 | 73.3 °C | 60.1 °C | 11.2 | 1.00 |
| 4 | 10,480 | 42.9 °C | 43.0 °C | 8.0 | 0.63 |
| 8 | 17,120 | 36.0 °C | 40.5 °C | 5.7 | 0.41 |
| 14 | 27,080 | 32.0 °C | 46.5 °C | 2.5 | 0.27 |

With a fixed coefficient more fins are always better, without limit, which
is what anything sized on it will build. In still air eight fins beat
fourteen, and fourteen are worse than four: the optimum fin spacing every
heat-sink handbook gives, here because the physics is in the model and not
because a rule was added. The bare plate was the other way round: 13 K
cooler than the fixed number said.

### 6. Fluids at their own state (`heatx_thermal.fluid_at`, `duty_fluid`)

Named fluids were property sets "near room temperature". Water's viscosity
halves between 25 and 60 °C; charge air at 2.5 bar and 150 °C is 1.75x as
dense as room air and 30% more viscous. Reynolds number, pressure drop and
h all follow. A duty that gives a temperature (`tIn`) now gets air (ideal
gas, Sutherland viscosity and conductivity), water and 50% glycol at their
state; a gas takes `pressure` (bar absolute); the rating reads each fluid
at the mean of its inlet and outlet, which it solves for in a few passes,
and converts a volume flow at the inlet's density. Properties given
explicitly are used as given. A charge-air cooler's duty (0.08 kg/s of air
at 150 °C and 2.5 bar against 20 l/min of glycol at 45 °C):

| | Room-temperature fluids | At their state |
| --- | --- | --- |
| Air pressure drop | 1,798 Pa | 913 Pa (−49%) |
| Air h | 260 W/(m² K) | 283 (+9%) |
| Coolant h | 1,862 W/(m² K) | 2,209 (+19%) |
| Heat moved | 8,043 W | 8,171 W |

A core sized against the old pressure drop was about twice as open as it
needed to be.

### 7. What the evals' delivered parts read now

The eval set's delivered parts, re-read with the criteria before and now
(`python bench/physics_audit.py --parts DIR`). Both columns use the same
quick reading (60,000 cells, a contact band of one cell), which is harsher
than the eval's own verification, so compare the columns with each other
and not with the scorecard's safety factors. None of these cases declares a
build direction, hours or cycles.

| Part | Material | SF before | SF now | Bond SF if layers are along x / y / z | Holds | Cycles | To |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Printed hook | PETG | 1.32 | 1.31 | 0.49 / 1.31 / 0.49 | 24 h | 28 | 57 °C |
| Wall bracket | PLA | 2.76 | 2.95 | 1.08 / 1.08 / 1.08 | 46 years | 2,400 | 58 °C |
| Motor mount | PETG | 4.49 | 5.18 | 3.50 / 2.32 / 2.19 | indefinitely | 54,000 | 78 °C |
| Bell crank | AlSi10Mg | 3.14 | 3.14 | 3.50 / 2.80 / 10.5 | indefinitely | 1.4 million | 300 °C |
| MBB beam | PA12 | 0.82 | 0.79 | 1.30 / 4.38 / 0.92 | - | - | - |
| Robot-arm wrist yoke | PA12-GF | 1.83 | 1.78 | 3.38 / 3.11 / 2.40 | 2,500 h | 3,900 | no curve |

The criterion itself moves these parts by −4% to +15%: compression-led
parts (the motor mount, the bracket) gain, parts led by tension with a
hydrostatic pull lose a little, the metal part does not move. The two
columns that were not reported at all say more. The hook is sound only if
its layers run along y; built along x or z its bonds carry less than half
the load, and nothing in the study or the report said so. And a part that
passes a static check is not a part that lasts: the bracket's static
margin of nearly 3 is about 2,400 load cycles, and the hook's hold time
is a day. How long a part must last is the user's intent, so the agent
declares it; the report's job is to make the trade impossible to miss.

## What is still a convenience, ranked

Not fixed in this pass. Ranked by how much each distorts a real part;
"owner" is where it is being worked or should go.

| # | Where | What the code assumes | What happens to a real part | Distortion | Owner |
| --- | --- | --- | --- | --- | --- |
| 1 | Supports (`domain.py`) | Every held node is fixed: a rigid, bonded clamp | A mount is a bolted joint: it presses only where it touches, slides unless friction holds, and gives. Rigid clamps over-stiffen the part (the calibration yoke read 3x too stiff bonded to its dummy) and put singular stress at their edges | High: stiffness, load sharing between supports and every near-support peak | new thread: one-sided, compliant supports in the linear check |
| 2 | Contact band (`verify.py`) | Stress within a band of loads and supports is left out of the peak | Parts often break exactly there: under a bolt head, at a clamp's edge. The band hides the solver's artefact and the real concentration together | High where interfaces govern (turret: 76% of mass is interfaces) | follows #1: with real contact and uniform tractions (done) the band can shrink |
| 3 | Thin members (`verify.py`, `fits.py`) | The delivered geometry is the printed geometry | A 1.4 mm member printed 0.2 mm under is 37% less stiff in bending and 27% weaker; FDM walls come in whole bead widths; thin SLS walls are weaker than thick ones. Only bores carry a tolerance today | High for parts sized by the minimum member | new: as-printed section on thin members (calibration coupons S, T) |
| 4 | The search (`study.py`) | The service envelope is reported, not designed for, unless a case declares its life | A part an agent ships without reading `attention` still lasts ~600 cycles | High, but now visible | decided: no library default. Service life is intent, so the agent declares it; the check should flag an undeclared life and the report should give the life-versus-margin trade (next) |
| 5 | Process (`spec.py`) | No `process` means isotropic; the bond reading is a report | The part is designed as a solid and printed in layers | High on FDM, now visible | decided: infer the process from the material and choose the layer orientation by default (next) |
| 6 | Load direction (`spec.py`) | A force points exactly where the study says | Nobody loads a hook within 0° of vertical. `direction: cone` exists; nothing applies it by default | Medium to high on slender, single-direction parts | new: a default misalignment, or an off-axis capacity reading (three extra solves) |
| 7 | Incidental loads | Only declared cases exist | Parts are dropped, plugged into, leaned on and shipped. `feel.drop` covers a part on a mount inside an enclosure; structural studies have nothing | Medium to high for handheld parts | partly in flight (touch-mic USB insertion load); general handling case is new |
| 8 | Moisture (`material_library.py`) | Nylon cards are dry as printed | Nylons lose stiffness and strength as they take up water: a little for PA12, about half for PA6-based filaments | Medium for nylons in service | new: `conditioned` values on the cards |
| 9 | Buckling (`verify._buckling`) | Linear eigenvalue against the goal's safety factor, short-term modulus | Real members buckle below it (imperfections; lattices already take 0.5), and a held load buckles later still as the modulus creeps | Medium for slender compression members | new, small |
| 10 | Residual stress (`buildsim.py`) | The build simulation predicts warp; the service check starts stress-free | A printed part carries the stress it cooled with into service | Medium for LPBF, low for polymers | new |
| 11 | Cooling and heat-sink optimizers (`cooling.py`, `thermal.optimize_heat_sink`) | Fixed `H_OUT = 10` on an enclosure's skin and a fixed h in the sink optimizer; air inside an enclosure conducts with "a little for convection" | The same as fix 5: the optimizer is rewarded for close fins | Medium | new: use `convection.natural_h` there (the optimizer needs its gradient) |
| 12 | Voxel submodels by loads (`verify._submodel_peak`) | With the default band (1.5 analysis cells) a 2.5x finer submodel can read the edge of a load as a peak (180 MPa on a 4 MPa pull) | The study runner passes the design grid's cell as the band, so runs are unaffected; a bare `verify_part` call is not | Low in runs, high for direct API use | small fix |
| 13 | In-plane FDM anisotropy (`am.py`) | Transversely isotropic about the build axis | Beads follow the perimeter; a thin member is all perimeter and stronger along itself than the card says, weaker across | Low to medium | slicer-in-the-loop thread |
| 14 | Linear elasticity | Stress and deflection scale with load up to failure | PETG and nylons soften well before their tensile strength; deflection near a low safety factor is under-read | Low at SF ≥ 2 | nonlinear verify exists (opt-in) |
| 15 | Exchanger flow (`flow.py`, `porous.py`) | Constant properties along the pass (now at the mean), incompressible | A gas accelerates as it cools or heats; 2-5% on pressure drop | Low | flow thread |
| 16 | Ageing | None | UV, hydrolysis (PLA, PETG outdoors), plasticizer loss | Unknown, slow | not planned |

In flight elsewhere and left to their threads: wire and flex-cable physics,
tilted insertion paths, the generated shell (physics-shaped exchanger
exteriors), and the touch-mic's USB insertion load.

## What would replace the guesses

Every `assumed` value above is one coupon or one measurement away from
`measured`. The calibration kit ([CALIBRATION.md](CALIBRATION.md)) already
prints tensile and flexural coupons flat and standing; these would join
it:

| Value | Default | Coupon | Reading |
| --- | --- | --- | --- |
| `compressionRatio` | 1.2 / 1.15 by family | a 10 mm cube or short cylinder, crushed flat in a vise with a kitchen scale or a bathroom scale under it | load at yield over the T-XY dog-bone's |
| `process.interlayerShear` | `interlayer/√3` | a lap or notched shear coupon printed standing (two notches, loaded in tension, shears one layer plane) | failure load over the sheared area |
| Fatigue curve for PETG, ABS, ASA | PLA's, scaled | the F-XY cantilever on a servo or a reciprocating saw | cycles to break at two stress levels |
| Emissivity | 0.9 polymer, 0.25 bare metal | none needed for polymers; an IR thermometer against a contact probe for metal | ratio of readings |
| Still-air h | the correlations | the heat-sink row above with a resistor and a thermocouple | hottest point, fins up and fins sideways |

## Using it

Nothing to turn on. A study run or `loadpath verify` now returns:

- `failure`: the criteria used, their ratios and sources; `layers` when no
  build direction was given.
- `cases[].interlayerTension`, `interlayerShear`, `interlayerAt`,
  `limitedBy` on layered parts.
- `serviceEnvelope`: `heldHours`, `cycles`, `maxTemperature`, per case and
  worst, and the `note` that goes to `deliver.attention`.
- `thermal.convection` when the block's convection is `"natural"` (the
  default): mean, least and most h, the share of faces looking into a gap
  and the mean view of the room.
- In a heat-exchanger rating, `fluids.<name>.properties`: the state each
  fluid was read at.

New study fields: `material.compressionRatio`, `process.interlayerShear`,
`thermal.convection: "natural"` or `{"type": "natural", "emissivity",
"up", "pressure"}`, and `pressure` on a duty's fluid.

Results move for existing studies in three ways: loaded faces are no
longer over-stressed at their edges (peaks near loads fall, deflections
shift by a few percent), layered parts with shear across their layers read
lower, and a `thermal` block with no `convection` now solves still air
instead of 8 W/(m² K). Tests: `tests/test_physics.py`.
