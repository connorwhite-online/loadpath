# Calibrating against real prints

Every number loadpath reports rests on a material card (Young's modulus,
strength, density) and the process's across-layer factors. Those come from
datasheets. A given printer, filament and slicer profile move all of them,
and nothing in the evals is a physical measurement (finding #4 in
[assumption-audit.md](assumption-audit.md)). `loadpath/calibration.py`
closes that loop with coupons you can test at home and a fit that writes a
calibrated card.

You need a kitchen scale, calipers, a few weights you can hang (water
bottles weighed on the scale are fine), a ruler, and a luggage scale with a
max-hold for the break tests.

## 1. Print the kit

```bash
python -m loadpath calibrate kit --material petg --out calibration/
```

| Coupon | Printed | Test | Gives |
| --- | --- | --- | --- |
| F-XY | flat | cantilever, hang 0.2 / 0.5 / 1 kg, read the drop | E along the layers |
| F-Z | standing | the same | E across the layers (`stiffnessAcrossLayers`) |
| T-XY | flat, two copies | pull to break with a luggage scale | tensile strength |
| T-Z | standing, two copies | the same | strength across layers (`interlayer`) |
| L-GY | flat | cantilever with a gyroid core | how a printed lattice compares with `lattice.SHEETS` |

Every coupon's mass gives the as-printed density. The STLs sit on the bed in
their print orientation and fit a 180 mm bed. Print them at 100% infill with
the walls, layer height and temperatures you use for real parts; the
cantilevers are 12 x 5 mm, so the walls fill most of them anyway.

The kit also writes `predictions.json` (what the starting card predicts for
each coupon) and `measurements.json` (the form to fill in).

## 2. Test and write it down

Masses in grams, lengths in mm, loads in kg. A cantilever's `points` are
`[kg hung, mm the hole dropped]`; with three or more the fit takes a slope
with an offset, so a slightly wrong zero or a settling clamp does not bias
the modulus. Measure each beam's width and thickness with calipers: the
modulus goes as 1/thickness³, so 0.1 mm on a 5 mm beam is 6%. A break that
starts at a hole or a taper instead of the gauge is flagged (it reads low).

## 3. Fit

```bash
python -m loadpath calibrate fit calibration/measurements.json \
    --predictions calibration/predictions.json --name petg-a1mini --save
```

The fit uses closed-form beam theory, not the voxel model, so the card is
measured independently of the solver it feeds. It writes a card with:

- `material`: E, strength, density, pasteable into any study as a custom material;
- `process`: `interlayer` and `stiffnessAcrossLayers` for that printer;
- `lattice.stiffnessFactor`: printed gyroid stiffness over the homogenized model, at the density that actually printed (wall-width error taken out);
- `parts` and `modelFactor.stiffness`: for each part test, measured stiffness over verify's prediction at the calibrated modulus: what the voxel model, bonded joints and slicer fill still miss;
- `kept`: what had no measurement and stayed at the starting card.

With `--save` the card goes to `loadpath/calibrations/<name>.json`, and a
study can then say `"material": "petg-a1mini"`. Its `process` factors apply
to a study whose process is the same type, unless the study sets its own.

## Part tests: the GRIP wrist yoke

A part test checks the whole chain, not just the material: a finished part on
a fixture with a known weight, its measured deflection against verify's.
`examples/grip/yoke_test.py` builds one for the wrist yoke printed from
GRIP's `generative/print/grip-wrist-yoke-PETG-FDM-v3.stl`:

- `fixture-base.stl`: stands in for the J4 horn. A 25 mm plate gripped in a
  vise by its long edge, a 22 mm boss meeting the hub, four M2.5 pilot holes
  on the horn's bolt circle.
- `fixture-dummy.stl`: stands in for the J5 servo. Like the servo, it touches
  the yoke only at its eight screw bosses (pads with M2 pilot holes), and it
  has an eye on top for the weight.

With the plate upright in the vise, the yoke sticks out sideways and a
weight on the eye bends it the way the arm's reach does: a moment about the
J5 axis direction plus a force across it.

Prediction (PETG card, 400k-cell assembly; the same to 1% at 150k):

| | per kg on the eye |
| --- | --- |
| Dummy rotation | 1.13 mrad (a laser on the dummy moves its spot 3.4 mm on a wall 3 m away) |
| Eye drop | 0.069 mm (a dial indicator reads it; a ruler cannot) |
| Share of the rotation | yoke 74%, horn boss 11%, plate 15% |
| Yoke peak stress | 3.2 MPa, at the hub's edge |

Up to 4 kg stays under a third of the card's strength. Two things the fixture
design taught: a solid dummy bonded wherever it touched the yoke stiffened the
cradle about 3x (a real servo bears only on the bosses), and a 15 mm plate
gripped 55 mm out turned about as much as the yoke did.

The printed yoke predates GRIP de445fd: its hub's underside sits about 2.5 mm
below the study's hub, solid over where the J4 horn disc goes. The base's boss
meets that underside, so the test runs on the part as printed.
