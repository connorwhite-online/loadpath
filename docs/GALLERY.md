# Gallery

Rendered 2026-11-01 from the tet-confirmed full baseline run (`063c233`) by `python docs/gallery.py`.

Regenerate with the commands at the top of [`gallery.py`](gallery.py). Stress maps colour worst-case von Mises stress against the part's own 98th percentile, to show the load paths, on the shape bent by the worst load case, exaggerated; the margin against the material is the SF. "Under envelope" compares each part with its design space filled solid, not with a conventionally designed part. All numbers are simulated ([VALIDATION.md](VALIDATION.md)).

## arm-link-cast

67.8 g · 46% under envelope · SF 20.85 · pass

<sub>deflection 0.478 mm · p5Member 5.35 mm · creaseLength 0.0 mm · volumeFraction 45% · 494 s</sub>

![arm-link-cast](images/cases/arm-link-cast-views.png)

![arm-link-cast stress](images/stress/arm-link-cast-views.png)

## arm-link-driven

51.3 g · 59% under envelope · SF 12.50 · pass

<sub>deflection 0.362 mm · p5Member 3.41 mm · creaseLength 0.0 mm · volumeFraction 32% · 374 s</sub>

![arm-link-driven](images/cases/arm-link-driven-views.png)

![arm-link-driven stress](images/stress/arm-link-driven-views.png)

## arm-link

54.2 g · 57% under envelope · SF 10.97 · pass

<sub>deflection 0.489 mm · p5Member 1.46 mm · creaseLength 0.0 mm · volumeFraction 36% · 901 s</sub>

![arm-link](images/cases/arm-link-views.png)

![arm-link stress](images/stress/arm-link-views.png)

## bell-crank

23.8 g · 84% under envelope · SF 4.90 · pass

<sub>deflection 0.066 mm · p5Member 4.19 mm · creaseLength 0.0 mm · volumeFraction 7% · 374 s</sub>

![bell-crank](images/cases/bell-crank-views.png)

![bell-crank stress](images/stress/bell-crank-views.png)

## grip-turret

142.1 g · 91% under envelope · SF 2.45 · pass

<sub>deflection 0.767 mm · p5Member 5.49 mm · creaseLength 32.7 mm · volumeFraction 2% · 1025 s</sub>

![grip-turret](images/cases/grip-turret-views.png)

![grip-turret stress](images/stress/grip-turret-views.png)

## grip-upper-arm

208.1 g · 88% under envelope · SF 5.83 · pass

<sub>deflection 0.515 mm · p5Member 4.96 mm · creaseLength 35.3 mm · volumeFraction 7% · 1383 s</sub>

![grip-upper-arm](images/cases/grip-upper-arm-views.png)

![grip-upper-arm stress](images/stress/grip-upper-arm-views.png)

## grip-wrist-link-mjf

41.0 g · 95% under envelope · SF 3.71 · pass

<sub>deflection 0.530 mm · p5Member 1.83 mm · creaseLength 0.5 mm · volumeFraction 6% · 2234 s</sub>

![grip-wrist-link-mjf](images/cases/grip-wrist-link-mjf-views.png)

![grip-wrist-link-mjf stress](images/stress/grip-wrist-link-mjf-views.png)

## grip-wrist-yoke

22.0 g · 91% under envelope · SF 4.24 · pass

<sub>deflection 0.112 mm · p5Member 3.41 mm · creaseLength 0.0 mm · volumeFraction 8% · 810 s</sub>

![grip-wrist-yoke](images/cases/grip-wrist-yoke-views.png)

![grip-wrist-yoke stress](images/stress/grip-wrist-yoke-views.png)

## mbb-beam

14.2 g · 78% under envelope · SF 1.51 · pass

<sub>deflection 3.008 mm · p5Member 1.01 mm · creaseLength 0.0 mm · volumeFraction 23% · 1607 s</sub>

![mbb-beam](images/cases/mbb-beam-views.png)

![mbb-beam stress](images/stress/mbb-beam-views.png)

## motor-mount

27.6 g · 69% under envelope · SF 4.07 · pass

<sub>deflection 0.431 mm · p5Member 1.46 mm · creaseLength 0.0 mm · volumeFraction 30% · 129 s</sub>

![motor-mount](images/cases/motor-mount-views.png)

![motor-mount stress](images/stress/motor-mount-views.png)

## printed-hook

27.4 g · 82% under envelope · SF 2.73 · pass

<sub>deflection 1.041 mm · p5Member 3.41 mm · creaseLength 0.0 mm · volumeFraction 18% · 2438 s</sub>

![printed-hook](images/cases/printed-hook-views.png)

![printed-hook stress](images/stress/printed-hook-views.png)

## wall-bracket-fasteners

41.0 g · 90% under envelope · SF 2.89 · pass

<sub>deflection 0.552 mm · p5Member 4.55 mm · creaseLength 0.0 mm · volumeFraction 6% · 790 s</sub>

![wall-bracket-fasteners](images/cases/wall-bracket-fasteners-views.png)

![wall-bracket-fasteners stress](images/stress/wall-bracket-fasteners-views.png)

## wall-bracket

64.4 g · 85% under envelope · SF 2.75 · pass

<sub>deflection 0.199 mm · p5Member 4.55 mm · creaseLength 0.0 mm · volumeFraction 11% · 381 s</sub>

![wall-bracket](images/cases/wall-bracket-views.png)

![wall-bracket stress](images/stress/wall-bracket-views.png)

## Heat exchangers

### box-crossflow

Gyroid crossflow in a box: A left to right, B front to back

![box-crossflow](images/heatx/box-crossflow.png)

### cylinder-counterflow

Gyroid counterflow in a cylinder: A through the ends, B in and out the side

![cylinder-counterflow](images/heatx/cylinder-counterflow.png)

### Printed in metal

The `lpbf-intercooler-core` eval case built along +z in AlSi10Mg. Nothing inside a metal part can be supported, so every flat ceiling over a channel is a defect: with `closure` the channels end in peaked roofs under the top skin and flare open over plenums below.

![lpbf ceilings](images/heatx/lpbf-ceilings.png)

## Manufacturing constraints

The `arm-link` eval case (SF 3, 0.5 mm at the wrist, PA12) with each of loadpath's manufacturing constraints. Cast two-sided along z and milled from +z and -z, every hole runs straight through; the cable channel along x is the one undercut left (it needs a side core), and the report says so (`manufacturing.access.undercutInKeepOuts`). maxMember 6 mm came out 6.8 mm thick at its thickest, under a 2 mm analysis cell over (the free part: 9.8 mm), and 2 g lighter.

![arm link](images/manufacturing/arm-link.png)

GRIP's wrist yoke in 6061-T6 aluminium at SF 2: free (38.4 g, SF 4.8) and 5-axis milled with a 4 mm cutter (38.8 g, SF 3.7): every void reachable from one of the six sides, no slot under 4 mm.

![wrist yoke](images/manufacturing/wrist-yoke.png)
