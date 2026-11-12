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

Eight things, the first seven in the order they distort real parts. Numbers are from
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
  The optimizer holds it too. The constrained method's stress limit reads
  each cell on von Mises and on this bond criterion, whichever is nearer
  its limit, and a search pick limited by its bond is then trimmed on that
  limit (the strength polish, CONVERGENCE.md): the printed hook went from
  39.7 to 33.7 g.

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
latch asks.

How long a part must last is not physics, it is what the part is for, so
the library does not pick a default. It makes the question impossible to
miss and cheap to answer:

- `loadpath check` returns `questions` when a load case gives neither
  `hours` nor `cycles`: which cases, what to decide, and the static safety
  factor each life needs in this material.
- `serviceEnvelope.trade` prices each life on the delivered part: 10^3 to
  10^6 cycles, a month, a year and ten years under load, each with the
  share of the strength the stress may reach, whether the part meets it,
  and how far its peak stress has to fall. A member sized by stress grows
  by about the inverse of that in section, which is the mass the life
  costs.
- A case that declares its life is designed for it. The search holds the
  fatigue and creep safety factor (`durability.safetyFactor`, 1.5) without
  a second switch; `durability.constrain: false` goes back to reporting
  only.

For PETG the trade reads: 1,000 cycles need a static safety factor of 2.5,
100,000 need 5.8, a million need 8.8; a year under load needs 2.0.

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

A powder bed's or a resin vat's minimum wall is read the same way, by
what the print loses (`processcheck._walls`): the process forms what a
ball of the minimum wall's diameter sweeps inside the part, and material
it cannot reach that stands more than that wall (plus a grid cell) off the
printed part is a wall, strut or fin that will not form. Closer than that,
the print only rounds an edge, as it rounds every edge. A percentile of
wall thickness judged this before, and judged a part by how much other
material it had: the robot wrist link's bore lips, the same 107 thin
cells, read 1.31 mm on a 30.7 g part and 0.57 mm on a 20.7 g one. Where
the designed surface meets a keep-out at a slant, the knife edge between
them is finer than any design grid sees; `reconstruct` cuts it along the
ball's sweep, so it ends in a round, as it will print. A drawn shape cut
thin by keep-outs is the designer's and stays: the warning names it (the
wrist link's idler shoulder leaves 0.58 mm between its parent's mount
hole and the shaft hole).

That reading is for a part handed to `verify`. A study that is designed no
longer gets there as a solid at all: a material says how it is made. PETG
is a filament, PA12 a sintered powder, AlSi10Mg a laser-melted one, so a
study that names no `process` takes its material's (`spec._infer_process`),
with that process's minimum wall and hole, its layer bond, its powder
escapes. Where layers matter (fdm, lpbf) the direction is not assumed to
be +z, and supports are not assumed to be worse than the strength it
takes to do without them. `orient.choose_build_direction` prices every
way of building the part by what it costs to make at equal strength
(`costing.estimate` with the process's rates): the free design printed on
the supports it needs (their material, machine time and removal from every
supported face) in each of the six axis directions, and, where that could
be beaten, the design grown under that direction's overhang rule. Height
counts (layers to recoat), and so does strength: a weaker direction needs
more material to carry the same loads. Whether a self-supporting design
pays depends on how thin its members are, so that choice is made at the
fraction the search delivers (`orient.settle_supports`): the LPBF bell
crank at 25% reads the same either way, at 12.5% SF 10.4 on 1.8 cm^3 of
support against 5.2 self-supporting. Ranked by strength per gram alone,
the crank stood on its 3 mm edge (6% more per gram on a coarse grid, 66 mm
of layers, 1.6 cm^3 of support) and read SF 0.60 designed in full. Which
process is intent, and a study that names one keeps it; which way the
layers run and whether it stands on supports are physics and cost, so the
library decides.
`loadpath check` says what it took. `"process": "none"` asks for an ideal
isotropic solid on purpose (a benchmark, a part made some other way), and a
study with a `milling` or `casting` block is milled or cast.

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

### 8. A part answers to stress over a volume, not at a point (`failure.MATERIAL_LENGTH`, `crosscheck.py`)

The tetrahedral re-check read its peak at one mesh node. At a re-entrant
corner or the rim of a held face linear elasticity has no peak to read: the
stress grows without limit as the mesh is refined, and the value at the
nearest node is a property of where the mesher happened to put that node.
One wrist-yoke surface, meshed six times, read SF 3.23 to 3.66; its hub's
rim read 31, 44 and 71 MPa on three of those meshes; lighter parts in the
constrained method's loop read 1.77, 2.07 and 3.09 for one design. The
search steers on that number.

A material does not answer to a point either. It fails when a volume large
enough to open a crack or a shear band reaches its strength (Neuber's
micro-support, Peterson's notch sensitivity, Taylor's theory of critical
distances), and that volume has been measured for printed polymers: MJF
PA 12 breaks at notches and cracks when the strain energy averaged over a
control volume of 0.86 mm radius reaches the plain material's (Avanzini et
al., Eng. Fract. Mech. 303, 2024), and printed PLA's critical distance is
4.6 mm (Ahmed & Susmel, FFEMS 41, 2018). So the re-check now reads the
stress tensor averaged over the material within that radius of each point
(the ball is cut by the part's surface, so at a surface it is the half
inside), and then the failure criteria of section 1 on the average.

| Wrist yoke, one surface | before | now |
| --- | --- | --- |
| SF on three 1.5 mm meshes | 3.39, 3.64, 3.62 (3.23 to 3.66 over six) | 3.73, 3.65, 3.73 |
| SF on a 1.0 mm mesh | 3.09 | 3.68 |
| Deflection on the three meshes, mm | 0.294, 0.303, 0.303 (0.284 to 0.308 over six) | 0.325, 0.327, 0.323 |

What changed with it:

- **The radius is material data** (`failure.material_length`): 0.86 mm for
  the nylons (measured on PA 12) and for the glassy thermoplastics (assumed:
  PLA's own is longer, so this averages over less than it would allow). For
  metals, brittle resins and elastomers no length is taken, and the stress
  is read at what the mesh resolves, half an element, which is also the
  floor for every material. The report says which
  (`confirm.stressReading`).
- **Bending members gain what bending bars gain.** On a smooth gradient the
  average at a surface sits 3/8 of the radius deep. For EOS PA 2200's 4 mm
  flexural bar that predicts failure at 1.19x the tensile strength; the
  data sheet says 58 against 48 MPa, 1.21x. A 3 mm member in pure bending
  reads 21% lower than its outer fibre; a member in tension reads the
  same.
- **A thin member limits the radius.** That bending gain is
  1 / (1 - 3r/4t) for a member of thickness t, and it cannot pass 1.5, the
  most a rectangular section carries between first yield and collapse. So
  the radius is held to 4/9 of the thinnest member the design may have
  (`minMember`; a propeller blade's thinnest section), and not below what
  the mesh resolves. Without it a 1 mm blade averaged over 0.86 mm read
  almost none of its bending stress. Studies with members of 2 mm and up
  are unchanged.
- **The band left out near loads and supports is measured to the regions
  as drawn**, not to the nodes a mesh put in them, and the peak is read on
  the band's edge itself, between nodes. A hot spot the edge passes through
  no longer counts or not as a node falls inside or outside.
- **Supports and loads take the surface nodes within the surface's own
  fidelity of their region** (twice the mesher's envelope, at least a fifth
  of an element), not within 0.6 elements. The wider catch held a ring of
  the part beyond every support wherever the mesh had a row of nodes there.
  Held closer to how it is drawn, the yoke reads 8% lower, and that is the
  more honest number.
- **Grids are sized from the machine's memory, not from what is free that
  minute** (`study._plan_bytes`): the constrained method's grid and the tet
  element size were set from free memory at the moment they started, so
  the same study got a different pitch every run (the wrist link 130k cells
  one time). Only when the memory really there cannot hold the planned grid
  is it made coarser.

One limit, measured: the average is only as good as the field under it. On
the yoke's blended surfaces the default 1.5 mm elements and a 1.0 mm mesh
agree (3.65 to 3.73 against 3.68). At a perfectly sharp inside corner (an L
bracket, `tests/test_stress_read.py`) elements twice the radius under-read
by about 10%: 14.7 to 15.0 MPa at 1.7 mm, 16.4 to 17.2 at 1.2 mm, 16.4 at
0.9 mm. Refining the mesh to the radius where the stress is highest is the
next step; a delivered part's corners are blended, so it has not been
needed yet.

The idealized clamp is still there (row 1 below): the band still hides what
a rigid support does to the stress beside it. What is new is that
everything outside the band is a number the part has, on any mesh.

## What is still a convenience, ranked

Not fixed in this pass. Ranked by how much each distorts a real part;
"owner" is where it is being worked or should go.

| # | Where | What the code assumes | What happens to a real part | Distortion | Owner |
| --- | --- | --- | --- | --- | --- |
| 1 | Supports (`domain.py`) | Every held node is fixed: a rigid, bonded clamp | A mount is a bolted joint: it presses only where it touches, slides unless friction holds, and gives. Rigid clamps over-stiffen the part (the calibration yoke read 3x too stiff bonded to its dummy) and put singular stress at their edges | High: stiffness, load sharing between supports and every near-support peak | new thread: one-sided, compliant supports in the linear check |
| 2 | Contact band (`verify.py`) | Stress within a band of loads and supports is left out of the peak | Parts often break exactly there: under a bolt head, at a clamp's edge. The band hides the solver's artefact and the real concentration together | High where interfaces govern (turret: 76% of mass is interfaces) | follows #1: with real contact and uniform tractions (done) the band can shrink. Outside it the tet re-check reads stress over the material's length (section 8), so the band no longer has to hide mesh noise |
| 3 | Thin members (`verify.py`, `fits.py`) | The delivered geometry is the printed geometry | A 1.4 mm member printed 0.2 mm under is 37% less stiff in bending and 27% weaker; FDM walls come in whole bead widths; thin SLS walls are weaker than thick ones. Only bores carry a tolerance today | High for parts sized by the minimum member | new: as-printed section on thin members (calibration coupons S, T) |
| 4 | The search (`study.py`) | A study that times none of its loads is still designed as applied once | A part an agent ships without answering the check's question lasts ~600 cycles | High, but asked and priced | no library default (decided): the agent declares the life from the user's intent, and a declared life is designed for |
| 5 | Process (`spec.py`) | A process named without `buildDirection` still builds along +z (fdm, lpbf) | The study's z is taken for the printer's; the best direction can be 2x stronger across the bonds | Medium (no process at all now takes the material's and chooses) | default `buildDirection` to `"auto"`; costs one coarse free design and its per-axis checks, plus the self-supporting designs that could beat supports |
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
| 17 | Thermal duty cycle tutorial and `examples/thermal` | Their mounts are declared ideal solids (`"process": "none"`) so the tutorial stays about heat | Designed as printed, the cold-designed PETG mount comes out 27 g and survives the cycle it is meant to fail | Low (a tutorial, declared) | redo the tutorial as a printed part, with its full-grid numbers |

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
  worst, `trade` (what each service life costs), and the `note` that goes
  to `deliver.attention`. `loadpath check` returns `questions` for loads
  nobody timed.
- `thermal.convection` when the block's convection is `"natural"` (the
  default): mean, least and most h, the share of faces looking into a gap
  and the mean view of the room.
- In a heat-exchanger rating, `fluids.<name>.properties`: the state each
  fluid was read at.

New study fields: `material.compressionRatio`, `process.interlayerShear`,
`thermal.convection: "natural"` or `{"type": "natural", "emissivity",
"up", "pressure"}`, and `pressure` on a duty's fluid; `"process": "none"`.

Results move for existing studies in three ways: loaded faces are no
longer over-stressed at their edges (peaks near loads fall, deflections
shift by a few percent), layered parts with shear across their layers read
lower, and a `thermal` block with no `convection` now solves still air
instead of 8 W/(m² K). A study with no `process` is now designed for its
material's process and a chosen layer direction, so FDM and LPBF parts
that were designed as solids come out heavier and printable. Tests:
`tests/test_physics.py`.
