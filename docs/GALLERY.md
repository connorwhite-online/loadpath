# Gallery

Rendered 2026-10-29 from `f5b3a9d` by `python docs/gallery.py`.

Regenerate with the commands at the top of [`gallery.py`](gallery.py). Stress maps colour worst-case von Mises stress against the part's own 98th percentile, to show the load paths, on the shape bent by the worst load case, exaggerated; the margin against the material is the SF.

## arm-link

58.9 g · 53% lighter · SF 13.19 · pass

<sub>deflection 0.450 mm · p5Member 3.47 mm · creaseLength 0.0 mm · volumeFraction 39% · 0 s</sub>

![arm-link](images/cases/arm-link-views.png)

![arm-link stress](images/stress/arm-link-views.png)

## arm-link-driven

46.2 g · 63% lighter · SF 7.98 · pass

<sub>deflection 0.494 mm · p5Member 3.41 mm · creaseLength 0.0 mm · volumeFraction 28% · 0 s</sub>

![arm-link-driven](images/cases/arm-link-driven-views.png)

![arm-link-driven stress](images/stress/arm-link-driven-views.png)

## bell-crank

25.2 g · 83% lighter · SF 3.44 · pass

<sub>deflection 0.056 mm · p5Member 4.19 mm · creaseLength 0.0 mm · volumeFraction 8% · 0 s</sub>

![bell-crank](images/cases/bell-crank-views.png)

![bell-crank stress](images/stress/bell-crank-views.png)

## grip-turret

172.8 g · 90% lighter · SF 2.33 · pass

<sub>deflection 0.382 mm · p5Member 5.49 mm · creaseLength 44.7 mm · volumeFraction 4% · 0 s</sub>

![grip-turret](images/cases/grip-turret-views.png)

![grip-turret stress](images/stress/grip-turret-views.png)

## grip-upper-arm

207.9 g · 88% lighter · SF 7.18 · pass

<sub>deflection 0.513 mm · p5Member 4.96 mm · creaseLength 38.2 mm · volumeFraction 7% · 1728 s</sub>

![grip-upper-arm](images/cases/grip-upper-arm-views.png)

![grip-upper-arm stress](images/stress/grip-upper-arm-views.png)

## grip-wrist-yoke

17.7 g · 92% lighter · SF 3.10 · pass

<sub>deflection 0.152 mm · p5Member 1.46 mm · creaseLength 2.9 mm · volumeFraction 8% · 957 s</sub>

![grip-wrist-yoke](images/cases/grip-wrist-yoke-views.png)

![grip-wrist-yoke stress](images/stress/grip-wrist-yoke-views.png)

## mbb-beam

26.4 g · 60% lighter · SF 1.56 · pass

<sub>deflection 1.669 mm · p5Member 2.65 mm · creaseLength 0.0 mm · volumeFraction 40% · 994 s</sub>

![mbb-beam](images/cases/mbb-beam-views.png)

![mbb-beam stress](images/stress/mbb-beam-views.png)

## motor-mount

27.5 g · 69% lighter · SF 2.73 · pass

<sub>deflection 0.322 mm · p5Member 1.46 mm · creaseLength 0.8 mm · volumeFraction 30% · 297 s</sub>

![motor-mount](images/cases/motor-mount-views.png)

![motor-mount stress](images/stress/motor-mount-views.png)

## printed-hook

31.3 g · 79% lighter · SF 2.95 · pass

<sub>deflection 0.624 mm · p5Member 3.41 mm · creaseLength 0.0 mm · volumeFraction 20% · 398 s</sub>

![printed-hook](images/cases/printed-hook-views.png)

![printed-hook stress](images/stress/printed-hook-views.png)

## wall-bracket

61.3 g · 85% lighter · SF 4.37 · pass

<sub>deflection 0.096 mm · p5Member 4.55 mm · creaseLength 0.0 mm · volumeFraction 11% · 519 s</sub>

![wall-bracket](images/cases/wall-bracket-views.png)

![wall-bracket stress](images/stress/wall-bracket-views.png)

## wall-bracket-fasteners

47.3 g · 89% lighter · SF 3.59 · pass

<sub>deflection 0.166 mm · p5Member 4.55 mm · creaseLength 0.0 mm · volumeFraction 8% · 448 s</sub>

![wall-bracket-fasteners](images/cases/wall-bracket-fasteners-views.png)

![wall-bracket-fasteners stress](images/stress/wall-bracket-fasteners-views.png)

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
