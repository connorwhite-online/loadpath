# Gallery

Rendered 2026-10-31 from `25376bc` by `python docs/gallery.py`.

Regenerate with the commands at the top of [`gallery.py`](gallery.py). Stress maps colour worst-case von Mises stress against the part's own 98th percentile, to show the load paths, on the shape bent by the worst load case, exaggerated; the margin against the material is the SF.

## arm-link

59.6 g · 52% lighter · SF 14.06 · pass

<sub>deflection 0.452 mm · p5Member 3.48 mm · creaseLength 0.0 mm · volumeFraction 40% · 561 s</sub>

![arm-link](images/cases/arm-link-views.png)

![arm-link stress](images/stress/arm-link-views.png)

## arm-link-cast

71.3 g · 43% lighter · SF 19.39 · pass

<sub>deflection 0.435 mm · p5Member 3.69 mm · creaseLength 0.0 mm · volumeFraction 49% · 867 s</sub>

![arm-link-cast](images/cases/arm-link-cast-views.png)

![arm-link-cast stress](images/stress/arm-link-cast-views.png)

## arm-link-driven

47.3 g · 62% lighter · SF 11.25 · pass

<sub>deflection 0.471 mm · p5Member 3.41 mm · creaseLength 0.0 mm · volumeFraction 28% · 822 s</sub>

![arm-link-driven](images/cases/arm-link-driven-views.png)

![arm-link-driven stress](images/stress/arm-link-driven-views.png)

## bell-crank

24.3 g · 84% lighter · SF 3.34 · pass

<sub>deflection 0.062 mm · p5Member 4.19 mm · creaseLength 0.0 mm · volumeFraction 7% · 384 s</sub>

![bell-crank](images/cases/bell-crank-views.png)

![bell-crank stress](images/stress/bell-crank-views.png)

## grip-turret

149.1 g · 91% lighter · SF 2.32 · pass

<sub>deflection 0.511 mm · p5Member 5.49 mm · creaseLength 35.9 mm · volumeFraction 3% · 2031 s</sub>

![grip-turret](images/cases/grip-turret-views.png)

![grip-turret stress](images/stress/grip-turret-views.png)

## grip-upper-arm

209.5 g · 88% lighter · SF 5.14 · pass

<sub>deflection 0.498 mm · p5Member 4.97 mm · creaseLength 33.5 mm · volumeFraction 7% · 1744 s</sub>

![grip-upper-arm](images/cases/grip-upper-arm-views.png)

![grip-upper-arm stress](images/stress/grip-upper-arm-views.png)

## grip-wrist-link-mjf

37.1 g · 95% lighter · SF 3.08 · pass

<sub>deflection 0.556 mm · p5Member 1.83 mm · creaseLength 0.9 mm · volumeFraction 5% · 1893 s</sub>

![grip-wrist-link-mjf](images/cases/grip-wrist-link-mjf-views.png)

![grip-wrist-link-mjf stress](images/stress/grip-wrist-link-mjf-views.png)

## grip-wrist-yoke

23.7 g · 90% lighter · SF 2.39 · pass

<sub>deflection 0.221 mm · p5Member 3.41 mm · creaseLength 0.0 mm · volumeFraction 8% · 1030 s</sub>

![grip-wrist-yoke](images/cases/grip-wrist-yoke-views.png)

![grip-wrist-yoke stress](images/stress/grip-wrist-yoke-views.png)

## mbb-beam

22.5 g · 66% lighter · SF 1.52 · pass

<sub>deflection 1.955 mm · p5Member 2.52 mm · creaseLength 0.0 mm · volumeFraction 35% · 299 s</sub>

![mbb-beam](images/cases/mbb-beam-views.png)

![mbb-beam stress](images/stress/mbb-beam-views.png)

## motor-mount

27.5 g · 69% lighter · SF 3.87 · pass

<sub>deflection 0.321 mm · p5Member 1.46 mm · creaseLength 0.0 mm · volumeFraction 30% · 182 s</sub>

![motor-mount](images/cases/motor-mount-views.png)

![motor-mount stress](images/stress/motor-mount-views.png)

## printed-hook

31.0 g · 80% lighter · SF 2.95 · pass

<sub>deflection 0.640 mm · p5Member 3.41 mm · creaseLength 0.0 mm · volumeFraction 20% · 534 s</sub>

![printed-hook](images/cases/printed-hook-views.png)

![printed-hook stress](images/stress/printed-hook-views.png)

## wall-bracket

64.6 g · 85% lighter · SF 2.57 · pass

<sub>deflection 0.133 mm · p5Member 4.55 mm · creaseLength 0.0 mm · volumeFraction 12% · 299 s</sub>

![wall-bracket](images/cases/wall-bracket-views.png)

![wall-bracket stress](images/stress/wall-bracket-views.png)

## wall-bracket-fasteners

45.7 g · 89% lighter · SF 2.51 · pass

<sub>deflection 0.192 mm · p5Member 4.55 mm · creaseLength 0.0 mm · volumeFraction 7% · 582 s</sub>

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
