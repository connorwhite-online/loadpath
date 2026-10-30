# A slicer in the FDM loop

Finding #2 and #3 of the [assumption audit](assumption-audit.md): mass was the
cost loadpath minimized, and infill was something you either meshed as a
lattice or left to the slicer's default. For an FDM user the cost is hours on
the printer, and the slicer already knows how to print a part sparse inside.
So loadpath now slices its own outcomes and tells the slicer where the part can
be sparse.

## What runs

- **Every FDM outcome is sliced** headless with PrusaSlicer (or SuperSlicer)
  standing the way it was designed to print. `report.json` → `slicer` has
  `printTime` (s), `filamentVolume` (cm³), `filamentMass` (g), `supportVolume`
  and `supportShare`, and the extruded volume of each feature (perimeters,
  infill, support). It is on by default when a slicer is installed
  (`"slicer": false` turns it off, `"slicer": true` warns when none is found).
  The plain slice is at 100% infill, because the part was verified solid.
- **Print time as the objective.** `"goal": {"safetyFactor": 2, "minimize":
  "printTime"}` (or `"filament"`) delivers the passing outcome that slices
  fastest, graded where grading is on, instead of the lightest. The two differ
  when a lighter topology needs more support or more perimeter length.
- **Graded infill.** `"slicer": {"grade": true}` sizes infill density per
  region from the stress and writes `<stem>-graded.3mf`, the part plus nested
  modifier volumes (PrusaSlicer's `Slic3r_PE_model.config`), sparse where the
  part idles and solid round bolts, supports, loads and hot spots.
  `report.json` → `graded` has the graded part's safety factor, the share of
  the part at each level, its own slice, and `printTimeSaved` /
  `filamentSaved` against the solid slice.

PrusaSlicer installs from Ubuntu's archive (`apt-get install prusa-slicer`) or
from prusa3d.com; `LOADPATH_SLICER` points at any other location. Orca Slicer
and Bambu Studio open the graded 3MF (they import PrusaSlicer's modifier
metadata), but their command lines are not driven yet.

## How grading is sized

The same homogenized fully stressed design `lattice.py` uses for TPMS cores
(`infill.py`):

1. Each analysis cell is part skin and part infill. The skin is the perimeters
   (count × line width) and is solid. The rest is gyroid infill at the
   level's density ρ: a sheet of modulus E·0.648ρ^1.43 whose walls carry
   3.01ρ^-1.10 times the macro stress (lattice.SHEETS, from unit-cell FEA;
   PrusaSlicer's gyroid is the same surface one extrusion wide), with a 0.8
   strength knockdown for single-line walls.
2. A voxel FEA of the graded part gives each cell's stress. Holding that
   stress, each cell takes the lowest level of the ladder (default 10, 15, 25,
   40, 60% gyroid, then solid) whose skin and infill walls both stay under
   strength / (SF × verify's 1.15 allowance). The analysis repeats as the load
   moves onto what got stiffer, raising levels only after a pass that fell
   short, until the graded part reads the target.
3. The graded part's safety factor is the verified solid part's, scaled by how
   much the graded one reads below the solid one on the same grid and loads,
   so it carries the delivered part's refined verification (submodels,
   interlayer check). Deflection is scaled the same way, and the ladder's
   floor steps up while it breaks `goal.maxDeflection`.

Holding stress rather than strain matters: at a fixed strain a denser gyroid's
walls carry more stress, not less, so a strain-held pick only ever chose the
sparsest level or solid.

**Walls stay the same everywhere by default.** A modifier that changes the
perimeter count makes PrusaSlicer split the layer there and run perimeters
round each region, inside the part. On the shelf bracket that took external
perimeters from 6 to 16 cm³ and added 45 minutes. The ladder's levels can
still set perimeters per level (`"levels": [[10, 2], [40, 4]]`) when that is
worth it.

## Results

Shelf bracket (`loadpath/examples/shelf-bracket.json`, PETG, +z), the outcome
at 25% of the design space, which verifies at SF 10.4 solid. Graded to the
goal's SF 3 in 30 s (25k analysis cells, two passes). Sliced with the generic
profile (0.4 mm nozzle, 0.2 mm layers, MK3S-class speeds, auto supports):

| Print | Time | Filament | Safety factor |
|---|---|---|---|
| Solid, as verified | 5 h 39 min | 84.9 g | 10.4 |
| Graded (`slicer.grade`) | 4 h 42 min | 55.3 g | 3.0 (target 3) |
| Uniform 15% gyroid | 3 h 40 min | 42.1 g | not checked |

The graded part prints 17% faster on 35% less filament and is still sized to
the goal. Uniform 15% is faster again, but nothing says it holds the load; the
graded 3MF is the cheapest print that has been checked. About 12 g of each is
support, which grading does not touch: the orientation decides that.

The whole search with `"goal": {"safetyFactor": 3, "minimize": "printTime"}`
and `"slicer": {"grade": true}` (12k cells, 12 min on the cloud box, slicing
and grading included):

| Outcome | Mass | SF | Solid print | Graded print (SF) |
|---|---|---|---|---|
| vf 0.150 | 50.7 g | 7.59 | 4 h 11 min, 55.7 g | 3 h 31 min, 40.0 g (3.82) |
| vf 0.112 | 37.6 g | 4.27 | 3 h 20 min, 41.8 g | 3 h 15 min, 35.4 g (3.04) |
| vf 0.103 | 34.5 g | 3.45 | **3 h 11 min**, 38.7 g | 3 h 21 min, 35.3 g (3.03) |

Two lessons. Grading pays on a part with margin to spend (16% faster at vf
0.15) and little on one the search already sized to the goal: at vf 0.103
there is only 15% of margin, the thin members are mostly skin anyway, and the
graded print is 10 minutes slower for its region boundaries while it saves
3 g. So `minimize: printTime` compares each outcome's solid and graded prints
and takes the cheaper; here that is the lightest outcome, solid. The first
version picked by graded time only and delivered vf 0.112 at 3 h 15 over a
3 h 11 solid print.

## Limits

- The infill constants are estimates (sheet-gyroid FEA and a knockdown).
  Printed coupons are what will tighten them (audit finding #4).
- The generic profile's absolute times are for a bed slinger; a CoreXY machine
  is two to three times faster. Pass your printer's config
  (`"slicer": {"profile": "my-printer.ini"}`, from PrusaSlicer's File > Export
  > Export Config) for absolute numbers. Comparisons between outcomes hold
  either way.
- Grading sizes infill on the topology the search delivered. It does not yet
  feed back into the search (a heavier topology that grades sparser could
  print faster); `minimize: printTime` compares graded times of up to three
  passing outcomes, which covers part of that.
