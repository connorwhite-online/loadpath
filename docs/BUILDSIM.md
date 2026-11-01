# Build-process simulation: warp, residual stress, compensation

`loadpath.buildsim` predicts what printing does to a part: the residual stress
it is left with, how far it moves when it comes off the plate or bed, whether
the build plane lifts into the recoater, and how hard an FDM part pulls on the
bed. It then feeds that back into design two ways: a pre-distorted mesh that
prints to the nominal shape, and a topology optimizer that weighs build
distortion against stiffness.

```json
"process": {"type": "lpbf"},
"buildSim": {"compensate": true}
```

adds `report.buildSim` to the delivered outcome (distortion, residual stress,
recoater lift, warnings) and writes `<name>-compensated.stl` next to the part.
Any mesh, from the command line (the walk-through is
[docs/tutorials/lpbf-buildsim](tutorials/lpbf-buildsim/)):

```sh
loadpath buildsim part.stl --material ti64 --pitch 0.5 --out sim/          # report.json, part-printed.stl
loadpath buildsim part.stl --material ti64 --pitch 0.5 --compensate 2 --out comp/   # + part-compensated.stl
loadpath buildsim comp/part-compensated.stl --nominal part.stl --material ti64 --pitch 0.5 --out check/
```

`deviation` in its report is the printed surface against the design
(`--nominal`, default the part itself) after a best fit, both ways, as a
scan comparison reads it. From Python:

```python
from loadpath.buildsim import BuildSetup, simulate_part, compensate_mesh
r = simulate_part(mesh, BuildSetup(process="lpbf", material="in625"), build="+z")
r.report["distortion"], r.report["residualStress"], r.report["recoater"]
fixed, rep = compensate_mesh(mesh, BuildSetup(process="lpbf", material="ti64"), iterations=2)
```

## How it works

Every process here leaves stress the same way: a layer goes down hot and
stress-free, bonds to the part below, then shrinks as it cools, and what it is
bonded to stops part of the shrink. The solver adds the part one analysis
layer at a time on the same voxel hex mesh as the rest of loadpath. Each new
layer is stress-free in the shape it is laid into (bottom on the distorted
part below, top on the nominal plane the recoater or nozzle levels) and gets
an *inherent strain*: the shrink it would take if free. Each layer is one
linear solve over the cells printed so far, and every cell's stress is
carried forward. Then the part is released, cut off the plate or peeled off
the bed, and springs back. A full release frees each separate piece with a
statically determinate (3-2-1) hold and removes its best-fit rigid motion, as
a CMM would.

| process | inherent strain | held by |
|---|---|---|
| lpbf | `-k * yield / E`, isotropic in the layer plane for rotating scans; `k` fitted on AM-Bench | the build plate, modelled as plate cells bolted at the underside (`plate` mm, default 10) |
| fdm | `-relax * alpha * (Tg - T(z))`, where `T(z)` runs from the bed temperature down to the chamber's; after printing, the whole part cools on the bed to room temperature | the bed, as an elastic foundation (`bed_stiffness`, N/mm³) or rigid |
| sls / mjf | `-ds * min(depth / depth_sat, 1)`: thick sections crystallize more and shrink more than thin walls; the bulk 3-4% shrink is assumed taken out by the printer's scale factor | nothing: the part cools free in the cake |

Layers are lumped: one analysis layer is `layer` mm, one voxel by default.
Lumping changes the answer, because a thick slab laid in one go bends a
compliant part less than thin layers do. The test beam below shows how much.
For that reason the LPBF factor is fitted at the voxel size the benchmark
runs at, and the pitch study further down gives the sensitivity.

## Verification: the progressive-deposition beam

Take a strip laid layer by layer on a free substrate, each layer shrinking
along its length once laid. Beam theory gives its curvature exactly for any
layer schedule (Tsui & Clyne 1997, progressive deposition;
`buildsim.progressive_beam` integrates it). The 3D solver matches it:

| lumping | FE curvature | beam theory | ratio |
|---|---|---|---|
| 0.5 mm layers | 2.840e-3 /mm | 2.884e-3 /mm | 0.985 |
| 1.0 mm layers | 2.089e-3 /mm | 2.113e-3 /mm | 0.988 |

For the same strip, beam theory gives 4.0e-3 /mm with 0.01 mm layers, so
0.5 mm lumps capture 72% of the thin-layer curvature on this compliant strip.
On a part held by a thick plate the gap is smaller (see the pitch study).

`tests/test_buildsim.py` covers this case, plus four more:
- a uniform shrink of a free part leaves it stress-free;
- a part cut off a plate springs up;
- a hot chamber warps less than a cold one;
- the warp optimizer's adjoint gradient matches finite differences to 1e-8.

## Validation: NIST AM-Bench 2018 bridge (LPBF)

AMB2018-01 is a 75 × 5 × 12 mm bridge on twelve legs. It was printed in IN625
and in 15-5 PH on an EOS M270 at 20 µm layers, on a 12.7 mm plate of the same
alloy. NIST then cut the legs off the plate by wire EDM, leaving the base
block attached, and measured the rise of 11 ridges on top by CMM (Phan et al.
2019, IMMI 8:318; NIST AMB2018-01 pages). In the 2018 blind challenge, six
model submissions predicted the free-end rise between -45% and +47% of the
measured value, and none met NIST's criterion (Levine et al. 2020, IMMI 9:1).

`bench/buildsim_validate.py` builds the bridge from NIST's drawing (legs, 45°
window overhangs, ridges) at 0.5 mm voxels on the 12.7 mm plate, modelled
as two voxel layers whose modulus is scaled so they are as compliant through
the thickness as the whole plate (`plateCells`, the default). It lays the
bridge in 0.5 mm analysis layers, cuts the legs off along x = 0-56 mm, and
reads the ridges.

![AM-Bench bridge: measured vs loadpath](images/buildsim/ambench-bridge.png)

| | free-end rise | all 11 ridges |
|---|---|---|
| IN625 measured | 1.276 mm | |
| IN625 loadpath, k = 0.839 fitted on this data | 1.275 mm | rms error 0.007 mm |
| 15-5 PH measured | 1.168 mm | |
| 15-5 PH loadpath with IN625's k (a prediction) | 1.882 mm (+61%) | rms error 0.33 mm |
| 2018 blind submissions | 0.70-1.88 mm | |

What this shows, and what it does not:

- **The IN625 match is a fit, not a prediction.** One number, `k`, sets the
  magnitude. The shape is not fitted: one scalar cannot bend the curve. The
  model reproduces the whole profile (the near-parabola from the free end,
  the root near x = 56-60 mm, and the flat attached end) to 7 µm rms, 0.6% of
  the tip rise.
- **The transfer to another alloy failed.** The yield-strain law
  `eps = -k * yield / E` carried from IN625 to 15-5 PH over-predicts by 61%.
  15-5 PH turns martensitic as it cools, and the transformation expands,
  undoing part of the shrink, which the law cannot know about. So 15-5 PH has
  its own fitted `k = 0.522` (7 µm rms), which leaves it with no independent
  check.
- **Other alloys are uncalibrated.** Ti64, 316L and AlSi10Mg use IN625's `k`,
  and `report.buildSim.strain.kSource` says so. For Ti64 that gives
  -0.0083 in-plane. Published elastic calibrations on twin cantilevers put
  Ti64 at -0.0047 / -0.0020 (Simufact, Ninpetch et al. 2022, quoted from a
  search summary, not read). For anything that matters, print a cantilever,
  cut it, and set `buildSim.scale`, the same calibration the commercial
  inherent-strain codes ask for.
- **Residual stress on the plate**, along leg L7, compared with the contour
  method: about -200 MPa near the plate in both, and +580 to +720 MPa in the
  top 1 mm against a measured ~600. Through the solid section (z 7-10 mm) the
  model reads +170 to +320 MPa, while the contour cut crosses zero near z = 8
  and reaches only about +100 MPa. That gap is a linear model with no
  yielding: the reported peak (2800 MPa at the sharp leg roots) is a
  singularity, not a stress. The report flags anything above yield.

![Longitudinal stress, leg L7](images/buildsim/ambench-stress.png)

**Pitch study and speed** (IN625 at the fitted k, the 4-core cloud box):

| voxels | plate | cells | free-end rise | time |
|---|---|---|---|---|
| 0.5 mm | 2 layers, scaled (default) | 35k | 1.275 mm | 65 s |
| 0.5 mm | whole 12.7 mm plate | 85k | 1.178 mm | 181 s |
| 1 mm | 2 layers, scaled | 5k | 1.025 mm (-20%) | 7 s |
| 1 mm | whole plate | 11k | 0.984 mm | 36 s |

The first version took 12 min at 0.5 mm on the whole plate. Two changes
cut that to 65 s: the reduced plate (most of the cells were plate), and
mg.py's matrix-free multigrid for big layer solves (cells not yet printed
get modulus 0), since the AMG setup per layer had been 60% of the time.
The reduced plate is not free: the bridge rises 8% more on it than on the
whole plate (the scaled layers match the plate's give through its
thickness, not how it spreads load sideways), so
`k` was refitted with it, 0.839 against 0.909 on the whole plate; set
`plateCells` high to model the whole plate, then use `scale` 1.08. Only a
plate bolted along its underside is reduced; a free one bends, which the
scaling cannot match. At 1 mm voxels the rise reads 20% low, and
`simulate_part` sizes the grid to about 60k cells by default, so on large
parts the voxels coarsen and the prediction reads low by about that much.

## FDM: warp against Armillotta et al. 2018 (ABS)

Armillotta, Bellotti and Cavallaro measured how flat the top face of ABS
bars came out of a Stratasys Dimension Elite: closed chamber at 75 °C,
0.254 mm layers, a soluble-support raft, flatness taken by 3D scan after
removal (Robotics and CIM 50:140-152, 2018). `bench/buildsim_fdm.py` runs
their nine 20 mm-wide bars as strips at 0.254 mm voxels, one voxel per
printed layer, so nothing is lumped (their tests showed width has no
effect). Two numbers were fitted: the share of the sub-Tg shrink the part
keeps (`relax`, least squares) and the raft's hold (`bed_stiffness`, swept
over 1.5-100 N/mm³ at 0.508 mm and checked at 3 and 4 at 0.254 mm). The bed
is an elastic foundation in all three directions, pinned only against
in-plane drift at its centre: an earlier version also held three corner
nodes down in z, which stopped exactly the corners that lift. The best pair
is 0.895 and 4 N/mm³, which are now the defaults:

| bar (length x height, mm) | measured warp | loadpath |
|---|---|---|
| 60 x 1.5 | 0.38 | 0.62 |
| 60 x 3.5 | 0.45 | 0.40 |
| 60 x 5.5 | 0.42 | 0.28 |
| 100 x 1.5 | 0.58 | 0.85 |
| 100 x 3.5 | 0.67 | 0.71 |
| 100 x 5.5 | 0.68 | 0.58 |
| 140 x 1.5 | 0.93 | 0.91 |
| 140 x 3.5 | 1.05 | 0.89 |
| 140 x 5.5 | 0.97 | 0.78 |

The rms error is 0.16 mm and the mean error 24%. Two things hold up: the
magnitude, and the growth with length (the measured warp grows 2.5x from
60 to 140 mm; the model grows 1.5-2.8x). One thing does not: the measured
warp peaks at 2-4 mm of height, while the model falls steadily with height.
Armillotta's own analytic model misses that peak too; they put it down to
reheating of the layers below above Tg, and to plasticity near Tg, and this
model has neither. With a rigid (glued) bed, the same bars barely warp
(0.02 mm at full shrink), because a part held flat while every layer goes
down holds a nearly uniform stress, and a uniform stress releases without
bending. The bed's give is what lets the warp build up during the print.

**The peak, tried and not fixed.** Two physical mechanisms were tested
against it, and neither reproduces it, so neither is on by default:

- *Bed-bond failure* (`buildSim.bedAdhesion`, MPa). Each bed node whose bond
  tension `kb * u_z` passes the adhesion lets go, and the part re-balances,
  layer by layer and again on cool-down; the report gives `detachedArea` and
  `bedFailed` (the whole footprint let go). Swept over bed 3-100 N/mm³ and
  adhesion 0.03-3 MPa (with relax refitted, which scales the adhesion too),
  the best fit with any letting go was rms 0.18 mm against 0.13 without, at
  0.508 mm voxels. Letting go flattens the tall bars (they curl free while
  printing, and later layers go down level on the curled ones) and leaves
  the thin ones, the opposite of the measured trend. It gives a peak at
  3.5 mm only on the 100 and 140 mm bars at bed 10-30 N/mm³ and 0.3 MPa,
  with the 60 mm bars falling. It stays as an option, because a bond that
  lets go is how prints fail, and `bedFailed` is a useful warning.
- *Reheating of the layers below* (an anneal of the strain already locked in
  when a hot layer goes down on it) moved the trend the wrong way and was
  dropped.
- The corner-pin fix gave a peak on the 140 mm bars at 0.508 mm voxels
  (0.87, 0.96, 0.84 against 0.93, 1.05, 0.97), but it went away at the
  printed 0.254 mm layers, so it was voxel lumping, not physics.

The thin bars are the miss: the model over-predicts the 1.5 mm bars by up
to 65% on the short lengths. The likely cause is something this model does
not have, thin bars cooling through Tg layer by layer in a few seconds
while their layers below are still soft.

The report also gives the bed's pull on the part (`peel`: the mean over the
area in tension, its maximum, and the net lift force). The maximum sits at an
edge, where a bonded joint's peel stress is singular and grows as the voxels
shrink, so the warning goes on the mean. ASA, PETG and PLA use the same two
fitted numbers with their own Tg and CTE, and none of them is validated.
PP and nylon use them too, with their crystallization temperature in
place of Tg (below).

### Semi-crystalline polymers: PP and nylon

An amorphous road (ABS, ASA, PETG, PLA) turns solid at its glass
transition. A semi-crystalline one (polypropylene, nylon) is still a melt
well below its melting point and turns solid when it crystallizes on
cooling, near 115 °C for PP and 150 °C for PA12, far above its Tg (PP's is
below 0 °C). With a Tg the model would see almost no shrink. So PP and PA12
take their crystallization temperature (`BUILD["fdm"]` rows with `tc`,
estimates from typical DSC cooling scans), and the shrink below it is
`relax * cte * (Tc - T)` as before, which is why they warp so much more
than PETG on the same bed. The report's `strain.solidifies` says which
model ran.

**A blind check on PP.** No published warp measurement with its geometry
and settings could be found for PETG or for nylon. The nearest is Samy et
al. 2022 (Polymers 14:2746): isotactic PP plates, 50 x 50 x 2 mm in four
0.5 mm layers, bed 100 °C, enclosed at 25 °C, scanned for the z warpage at a
point near a corner, (7.8, 2.1) mm. `bench/buildsim_fdm_pp.py` runs it with
nothing fitted to PP (relax and the bed from the ABS fit above; PP's card
values are estimates: 1.1 GPa, 120 µm/(m K), Tc 115 °C):

| | warpage at the point (mm) |
|---|---|
| measured, five raster patterns | 0.80-1.25 (0.80 lines 90/90, 0.83 concentric, 1.10-1.25 others) |
| Samy et al.'s own model (crystallization kinetics, viscoelastic) | 0.75-1.09 |
| loadpath, after removal, over the plate's lowest point | 0.53 |
| loadpath, the corner itself | 0.83 |

loadpath reads a third low at the point (0.53 against 0.80 for the
flattest raster), and the paper does not say exactly what the point was
measured against, so this is a check of magnitude, not a validation.
The model has no raster: Samy et al. measured up to 50% more warp for some
patterns than others, which an isotropic shrink cannot tell apart. With
Tg instead of Tc the same plate would hardly warp at all.

### FDM in the optimizer

The warp term and the warp limits work for FDM on the same build model:
the elastic bed (corners free to lift), each road solid below its Tg or
Tc, and the cool-down on the bed at the end. On a PETG bar it moves within
0.5% of `simulate_build`. For FDM, `maxDistortion` is the largest movement
along the build direction at the end, cooled on the bed: the lift off the
bed. The in-plane shrink is left out because it is uniform and the slicer
scales it away.

**FDM warp depends on the layer thickness, so it is extrapolated to the
printed layer.** The 3-4x gap between the optimizer's FDM warp and the
re-simulation of the delivered part came from here. On a solid 60 x 4 x 3 mm
PA12 bar on the bed, the lift is set by the analysis layer and not by the
voxel pitch, and it keeps growing as the layers thin:

| analysis layer, mm | 1 | 0.5 | 0.25 | 0.125 |
|---|---|---|---|---|
| lift, mm (any pitch from 0.125 to 0.5) | 0.298 | 0.455 | 0.621 | 0.795 |

Each halving adds about the same amount (0.157, 0.165, 0.174 mm), so the warp
is close to linear in log(layer). The optimizer's coarse grid built in
3.7 mm lumps and the re-simulation in about 1.1 mm ones, against 0.2 mm in
the printer. Every FDM build now runs a second time at twice the lumps and
takes u = (1 + k) u(t) - k u(2t), with k = log2(t / printed layer).
`buildSim.printLayer` sets the printed layer (default 0.2 mm; 0 turns the
correction off). The optimizer's term carries it exactly, because both
builds go through the adjoint and the gradients match finite differences.
The report's `layerExtrapolation` gives the raw number. From 0.5 mm lumps
the correction lands within 3% of a direct 0.25 mm build, where the plain
0.5 mm build reads 25% low (a test checks this). It needs four or more lumps
through the part. With fewer it still reads low (0.52 against about 0.68 mm
for the 3 mm bar in 1 mm lumps). The Armillotta fit used one voxel per
printed layer, so it does not change.

On the PA12 beam below, at 28% volume, the re-simulation of the same
delivered part now settles at 2.3-2.9 mm from 8,000 to 60,000 cells
(2.33, 2.45, 2.93 and 2.53 mm). Before, it read 1.0-1.5 mm and was still
climbing with the cell count. The optimizer's own number rose from 0.34 to
1.55 mm.

**What is left: the optimizer's grid.** A 12 mm tall part on a 1.85 mm
density grid has six layers, and thin members are one or two cells.
There, the optimizer's FDM warp still reads 2-3.5x below the delivered part,
and the ratio changes from design to design. So FDM warp limits are now
checked on the delivered part. In the safety-factor search, each outcome's
mesh is re-simulated (`simulate_part` at `buildSim.cells`, at most 30,000,
on the bed, about 1-3 minutes) into `report.warp.delivered`. The margin
uses that number, and the optimizer's limit for the next try is scaled by
`deliveredOverOptimizer` from the last one. The metal build model keeps its
full-grid check, which passed its re-simulation on the Ti64 bracket.

**A warp limit on this FDM beam: no part passes, because the warp is not
the optimizer's to move.** `bench/buildsim_fdm_warp_limit.py` runs a
120 x 24 x 12 mm beam printed flat on the bed, bolted at two preserved end
pads, 200 N at mid-span, SF 2. The PA12 runs, with every lift taken from
the delivered part:

| PA12 | no limit | maxDistortion 0.85x (2.1 mm) | 0.5x (1.24 mm) |
|---|---|---|---|
| best part | 11.5 g, SF 2.05, lift 2.48 mm | none passed | none passed |
| outcomes tried | 30%: 3.52 mm; 26%: 2.94 mm; 28%: 2.48 mm | 30%: 3.52; 53%: 3.39 (SF 0.78); 70%: 3.71 (SF 0.09) | 30%: 3.52; 60%: 2.46 (SF 0.54); 70%: 2.59 (SF 0.64) |

Across every design and volume fraction tried, the delivered lift stays at
2.5-3.7 mm. Pushing the optimizer toward the limit costs the load path
(SF down to 0.09) and does not lower it. The worst lift is not in the
preserved pads, so exempting them would not help. It sits in the bottom
chord on the bed, just inside the pads (x 22-26 and 92-96 mm, 1-3 mm up).
On this part the warp is set by the material, the height and the length on
the bed, not by the topology. The levers are the process: bed and chamber
temperature, a brim or `bedAdhesion`, a lower or tilted part, or PETG. In
PETG the same beam lifts 0.36-0.40 mm, a seventh of nylon's, in line with
nylon's reputation on open printers. A part whose warp lives in what the
optimizer may change (long thin floors, wide flanges) is where an FDM limit
should bite, and that has not been shown yet.

**Calibration print.** The FDM fit is ABS in a 75 °C chamber, so for PETG on
an open printer it is a guess. `bench/warp_coupon.py` writes five flat
bars in Armillotta's shapes (20 mm wide, 60-140 mm long, 1.5 and 3.5 mm
tall), with the model's on-bed lift and off-bed flatness for each, and the
measuring steps (feeler gauges on glass). The PETG set for an A1 mini at a
70 °C bed predicts 0.17-0.30 mm flatness and 0.05-0.12 mm lift on the bed.
Measured flatness refits `relax` and the bed hold as `bench/buildsim_fdm.py`
does.

## SLS / MJF

The differential-shrink model (`ds` = 0.3% between a thin wall and a section
4 mm deep or more) is a heuristic sized to the 0.2-0.5% differential shrink
the nylon sintering literature reports. It is not fitted to a measured part,
and the report labels it as a heuristic. Use it to compare designs (a thick
boss on a thin panel bows the panel), not to predict millimetres.

## Pre-distortion compensation

`compensate_mesh` simulates the part, moves every vertex against its
predicted displacement, simulates the moved shape, corrects by whatever error
is left, and repeats. On a Ti64 overhang bracket (a 30 mm arm on a 6 mm post,
0.5 mm voxels, full release):

| round | max deviation from design | rms |
|---|---|---|
| print the design | 1.52 mm | 0.76 mm |
| compensated once | 0.20 mm | 0.12 mm |
| compensated twice | 0.11 mm | 0.05 mm |

What remains is about one voxel's worth of resolution. The study writes
`<name>-compensated.stl` when `buildSim.compensate` is true.

## Warp-aware topology optimization

`optimize_warp(grid, supports, force, WarpOptions(...))` minimizes
`c / c0 + w * D / D0`, where `c` is service compliance and `D` is the mean
squared as-built displacement from the layer-by-layer build on the plate.
The gradient of the layer-by-layer solve comes from the adjoint: two solves
per analysis layer, sharing one factorization. It matches finite differences
to 1e-8.

`bench/buildsim_optimize.py` runs a Ti64 shelf bracket, 40 x 4 x 24 mm
(2.5D, 1 mm voxels), built standing on the plate. It is bolted along x = 0,
and its top corner is loaded. Two designs at the same volume target, both
re-simulated afterwards with the full build model (5 mm plate, then cut off):

| | compliance only (w = 0) | compliance + distortion (w = 1) |
|---|---|---|
| volume fraction | 0.315 | 0.298 |
| service compliance | 1.76 N mm | 3.61 N mm (2.0x) |
| max distortion on the plate | 0.51 mm | 0.088 mm (-83%) |
| max distortion after cut-off | 0.39 mm | 0.18 mm (-54%) |
| lift at the build plane (recoater) | 0.31 mm | 0.056 mm |
| residual von Mises p95 on the plate | 523 MPa | 927 MPa |

| compliance only | warp-aware |
|---|---|
| ![](images/buildsim/bracket-w0.png) | ![](images/buildsim/bracket-w1.png) |

(Colour is the as-built displacement, from 0 in dark purple to 0.8 mm.)

The compliance-only design is a classic truss whose top chord runs
straight out from the flange at the very top, so it is printed last as a
30 mm horizontal span, which curls up as it goes down: 0.31 mm of lift at
the build plane, six times a typical 0.05 mm recoater clearance. The
warp-aware design builds nothing flat and high. A post stands on the plate
under the load pad, a strut runs along the plate back to the flange, and a
tie climbs from that post to the top of the flange at about 30°. The price is
real: twice the service compliance at w = 1, and more stress locked in on
the plate (most of which the cut releases). The weight `w` sets that trade.
Both the trade and the recoater gain come from the optimizer's view of the
build, which no compliance-only optimizer has.

### In a study

The same term is in the study optimizer. Set a weight under `buildSim`:

```json
"process": {"type": "lpbf", "buildDirection": "+z"},
"buildSim": {"warpWeight": 1}
```

The optimizer then minimizes `compliance * (1 + w * D / D0)`, with `D` the
mean squared as-built displacement of the part on its plate and `c0`, `D0`
from the first iteration, and the stress and volume goals still apply as
they do without it. `warpWeight` 0 (the default) leaves the optimizer alone,
and `buildSim` then only verifies and compensates the delivered part. Around
1 is a big trade, as in the table above. The term needs a build direction
along a grid axis (`+z`, `-y`, ...), so a tilted or `auto` build is refused,
and it needs `lpbf` or `fdm`. Layers are lumped to `warpLayer` mm, by default
4 cells. Each iteration costs two sparse solves per lumped layer on top of
the service solve, so a 30-layer part runs several times slower than
compliance alone (7x in the demo below).

**Demo through a study** (`bench/buildsim_study_warp.py`): the Ti64 shelf
bracket of the table above, 40 x 6 x 24 mm, at a fixed 35% volume, run
through `run_study` with `warpWeight` 0 and 1. The study's own `buildSim`
pass checks each delivered part (about 8000 cells, 5 mm plate, as built on
the plate):

| | warpWeight 0 | warpWeight 1 |
|---|---|---|
| mass | 10.0 g | 9.9 g |
| safety factor | 2.07 | 0.96 |
| max as-built distortion | 0.51 mm | 0.17 mm (-66%) |
| rms as-built distortion | 0.13 mm | 0.038 mm (-72%) |
| recoater lift | 0.20 mm | 0.054 mm (-73%) |
| residual von Mises p95 | 933 MPa | 992 MPa |
| study time (cloud box) | 2.5 min | 17 min |

The trade is the same as with the Python API: the build gets much flatter
and the part loses half its strength at the same mass, so at w = 1 a
safety-factor goal that the plain design meets fails. Use a smaller weight
(untested here), or a volume that leaves room, when strength must hold; the
study reports both numbers. The run is about 7x slower.

### Warp limits in the safety-factor search

A weight trades strength for flatness blindly. A limit does not: set
`buildSim.maxDistortion` (mm, the largest as-built displacement on the
plate) and/or `buildSim.maxRecoaterLift` (mm, how far the part rises at the
build plane as each layer goes down), keep the safety-factor goal, and the
search delivers the lightest part that meets both:

```json
"goal": {"safetyFactor": 2},
"process": {"type": "lpbf", "buildDirection": "+z"},
"buildSim": {"maxDistortion": 0.2}
```

Inside each optimizer run the limit is a constraint held by an adaptive
weight (raised while the design is over it, eased off under it), steered
by the exact gradient of a 16-norm of the displacements rescaled to the
true maximum at the current design. Each outcome's report gets `warp`
(the as-built measures of the delivered field) and its margin includes
limit / measured, so a part over a limit fails like a part under its
safety factor and the search moves on.

**Speed.** The build model runs on a grid 2x coarser than the optimizer's
(`warpCoarsen`, default 2 at 4000 cells and up): 8x fewer cells and half
the layers. Averaging moduli misreads thin members, so every 10 iterations
one forward build on the full grid re-measures the bias and the coarse
measures are scaled by it. On the bracket's 7000-cell grid a 60-iteration
run takes about 2.5 min with the limit (53 s with the coarse grid alone, no
bias correction), where the full-grid term ran at about 10-15 s per
iteration (1003 s for the single-fraction study in the table above, against 147 s without it). The pass/fail check reads the full grid.

**What a limit costs** (`bench/buildsim_warp_limit.py`; the Ti64 shelf
bracket, SF 2, 1500 N; `warp` is the optimizer's build model, `buildSim`
the study's independent re-simulation of the delivered mesh with 1-voxel
layers on a 5 mm plate):

| limit | mass | SF | warp: max / lift | buildSim: max / lift | study time |
|---|---|---|---|---|---|
| none | 11.4 g | 2.67 | 0.34 / 0.15 mm | 0.49 / 0.086 mm | 28 min |
| maxDistortion 0.20 mm (0.6x) | 10.0 g | 2.11 | 0.17 / 0.033 mm | 0.28 / 0.067 mm | 15 min |
| maxDistortion 0.12 mm (0.35x) | none passed | | | | 14 min |
| maxRecoaterLift 0.054 mm (0.35x) | 14.0 g (+23%) | 2.13 | 0.27 / 0.045 mm | 0.27 / 0.104 mm | 38 min |

- A distortion limit at 0.6x of the unlimited design cost no mass here: the
  limited part came out lighter (10.0 against 11.4 g), with the
  independent check's distortion down 44%. That is the search's noise as
  much as anything (the unlimited search failed at 10.4 g with SF 1.65 and
  stopped at 11.4), so read it as "no measurable cost", not a saving.
- At 0.35x nothing met both: the warp-limited designs reached SF 0.84 at
  most, so the study reports no passing part. That is the answer to give:
  this bracket, built this way up, cannot be that flat at SF 2.
- The lift limit cost 23% mass, and it is the weak one: the optimizer's
  model met it (0.045 mm), but the independent re-simulation reads 0.104
  mm, more than the unlimited part's 0.086. Recoater lift is a per-layer
  quantity, and lumping layers (the optimizer's build lays 4 cells at a
  time) misjudges it. Prefer `maxDistortion`; treat `maxRecoaterLift` as
  steering, and read `buildSim.recoater` on the result.
- The unlimited row still carried the build model (its limit was set out of
  reach to report `warp`), which is why it took 28 min; a study without
  warp terms runs the same search in a few minutes.

## Limits

- Speed. Each analysis layer is one solve over the part plus two layers
  of plate. AM-Bench at 0.5 mm (35k cells, 25 layers) takes 65 s on the
  4-core cloud box, and `compensate` runs the simulation iterations + 2
  times. The warp term in the optimizer is the slow part (see above).

- Linear elastic throughout. There is no plasticity, so peaks at sharp roots
  exceed yield (flagged in the report), and there is no creep or stress
  relief. Stress-relief heat treatment before cut-off would remove most of
  the LPBF distortion this predicts.
- One inherent strain per alloy and process, isotropic in plane, which suits
  rotating scans. Stripe or island patterns that do not rotate are
  anisotropic: give `strain` explicitly.
- No thermal history and no preheat. For AlSi10Mg on a plate preheated to
  150-250 °C, the measured distortion falls toward zero (Buchbinder 2014);
  scale it down.
- Voxel and lumping dependence, as quantified above. The LPBF calibration
  holds at 0.5 mm voxels and 0.5 mm analysis layers.
- FDM is fitted to one machine family (an enclosed Stratasys printer, ABS
  on a support raft) and misses the measured warp peak at 2-4 mm of
  height (bed-bond failure and reheating were tried and do not explain it). ASA, PETG and PLA are unvalidated, and an open-frame printer on
  a PEI sheet holds the part differently (`bedStiffness`).
