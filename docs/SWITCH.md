# Push pads: force-tuned interiors under a fixed exterior

An industrial designer hands over the surfacing. The board, the switch
and the other parts that sit inside are known, and so is how the product
should feel: a press anywhere on the touch area should click at about the
same force, or along a gradient someone picked, and a thumb resting on it
should not set it off. Everything inside the exterior is invisible, so it
should be designed for that feel, with the least material.

`loadpath switch` does this. The exterior is kept as a thin cosmetic skin
(the study's `skin`), so the skin's own thickness under the touch area,
the walls and the rim are design space too. The board and switch are
keep-outs, each with its **assembly path** (`insertFrom`), so nothing
generated gets in the way when the part is put together. A study's
`switch` block says where the plunger is, how far it is from the stem,
what force fires the switch, and the force window at each press point.

```json
"skin": {"thickness": 0.7, "open": ["-z"]},
"keepOut": [
  {"id": "board",  "insertFrom": "-z", "bore": false, "shape": {...}},
  {"id": "switch", "insertFrom": "-z", "bore": false, "shape": {...}}
],
"switch": {
  "unit": "gf",
  "output": {"region": "plunger", "direction": "-z", "gap": 0.4, "trigger": 65, "travel": 0.25},
  "field": {"center": [0, 0], "radius": 12, "max": [160, 200], "min": 110},
  "clearance": {"region": "skin", "gap": 1.6},
  "hold": [{"case": "squeeze", "maxDeflection": 0.25}],
  "pitches": [1.0, 0.7]
}
```

```
python -m loadpath switch pad.study.json --out pad-out/
```

The whole example is `examples/switch/touch_pad.py`. It builds a
synthetic touch pad, writes its study, and runs it.

## The model

The plunger has to cross the gap g, then push the stem through the
switch's own travel s until the stem's force reaches the trigger force T.
For a press at point i, with K the stiffness of the current design:

```
u_i = K^-1 f_i     (f_i: 1 N spread over a fingertip disc at i)
lam = K^-1 l       (l: the plunger face's mean travel along the stroke)
a_i = l . u_i      plunger travel per newton pressed at i
c   = l . lam      the plunger's own compliance
F_i = (g + s + T c) / a_i
```

This follows from superposition of the press and the stem's reaction.
Every quantity is a product of two solves, so every sensitivity comes
from solves that are already done: one per press point and one for the
plunger, per iteration. The press travel at the click,
`d_i = (f_i . u_i) F_i - a_i T`, is the "mush" before the click.

The optimizer minimizes volume with one constraint per press point (MMA),
written on log F so that a factor of two weighs the same at 100 gf and at
800 gf:

| constraint | what it is for |
|---|---|
| `F_i <= max_i` | the ceiling: any gradient over the area (`field.max: [centre, rim]`) |
| `F_i >= min_i` | the floor: a resting thumb does not fire it |
| `d_i <= maxTravel` | crispness |
| skin travel over `clearance` <= gap | the skin does not reach the board (KS over nodes and presses) |
| mean deflection under each `hold` case <= limit | squeeze, drop: the rim keeps its shape |

## What it does on the touch pad

The synthetic pad is 50 x 59 x 9 mm in SLS TPU 88A. It has a 1 mm deep
dish in its top, a 0.7 mm skin, a 6 x 6 mm switch (65 gf, 0.25 mm of
travel) whose stem is 0.4 mm under the plunger, and 13 press points on two
rings out to 12 mm. The target is 160 gf at the centre rising to 200 gf at
the outer ring, never under 110 gf.

| design (1 mm cells) | fire force | interior used |
|---|---|---|
| skin only | 76-122 gf: under the floor everywhere | 0 |
| filled | 730-2370 gf | 100% |
| **optimized** | **150 gf centre, 162-164 inner ring, 200 outer ring: all in their windows** | 28% |

That run took 80 iterations and 281 s on 4 CPU cores. Starting from a
uniform 50% fill, the forces went from 166-445 gf to inside their windows
by about iteration 20. The rest of the run removed material while keeping
the forces in place.

## Two things that had to change

**A thin skin on a coarse grid.** Preserved cells are whole cells, so a
0.7 mm skin on 1 mm cells is a 1 mm skin. Bending goes as thickness cubed,
so the click came out about three times too stiff, and the optimizer could
only add material on top of that. It removed everything and stopped at
the skin, which is the same thing the touch-mic run saw with its whole
shell preserved. When the skin is under two cells thick, `switch` now
turns each cell the skin passes through into a design cell. The cell's
density is floored at its share of skin volume φ (4^3 samples per cell).
Under SIMP's E ~ x^3, a cell at φ bends like the real skin (φh = t gives
E h^3 φ^3 = E t^3). Membrane stiffness reads low by φ^2. This stands in
for the skin on coarse grids only. From two cells per skin thickness up,
the skin is ordinary preserved cells.

**Void cells.** A pad's optimum is mostly empty. With empty cells kept in
the system at E_MIN, preconditioned CG crawled: 60 s an iteration on 17k
cells. Each solve now keeps only the cells above a density of 0.02 that
connect to a support, plus the preserved ones. Up to 120k unknowns it
factors that system once (SuperLU) and back-substitutes every press point
from it, which takes 3-8 s an iteration. Above that it uses the
matrix-free multigrid (`mg.py`), which runs on a GPU with
`LOADPATH_DEVICE`.

## Coarse to fine

`pitches: [1.0, 0.7, 0.5]` runs the problem on each grid in turn. Each
grid starts from the previous one's design (trilinear), with sharpening
backed off one step.

## Checks on the result

- **Thresholded reading.** The same linear fire forces on the black and
  white design, as it will print (`thresholdedForceGf`).
- **Large deflection** (`check_fire_nonlinear`). TPU skins stiffen as
  they bow. For one press point per distinct target, the solid cells are
  re-solved with a neo-Hookean large-deflection model. The press and the
  stem's reaction at the trigger force are applied together, and secant
  steps find the press force that brings the plunger to gap + travel. The
  load is elastic, so the order the two loads arrive in does not change
  the final state. A snap-through before the click shows up as a solve
  that does not converge. On a coarse grid the thin-skin cells are kept at
  modulus E x^3.
- **The delivered mesh** is the isosurface joined to the exact preserved
  geometry, so the cosmetic skin's outside is the designer's mesh
  unchanged.

## Assembly paths

`"insertFrom": "-z"` on a keep-out sweeps it out of the design space on
that side (or along any vector). A board slides in from below, a switch
drops onto it, a driver reaches a screw. As a plain keep-out, the
component's final place stays empty, but a rib could still grow across
the way it has to come in. With `insertFrom`, the whole path stays empty.
Boxes and cylinders moving along one of their own axes are swept exactly.
Anything else is swept as a union of copies a fraction of a millimetre
apart. A swept keep-out is a clearance, not a bore, so it may end against
material.

## Limits

- The model is linear apart from the large-deflection check. Contact
  between the plunger and the stem is modelled by its reaction at the
  click, not as an obstacle on the way there.
- Press directions are along an axis (`pressDirection`). The fingertip
  lands on the outermost node of each column within `pressRadius`.
- `MeshShape` signed distances come from a 160-cell grid over the mesh's
  largest extent (0.37 mm on a 59 mm part). That sets how well a 0.7 mm
  skin is placed. The exterior in the delivered mesh is exact either way.
