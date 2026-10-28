# loadpath

Generative design for parts you are going to print. Describe what a part must
keep, what it must avoid, where it is held and how hard it is pushed; loadpath
removes material until only the load path is left, meshes the result as a
watertight printable part with exact bores and faces, and then checks that part
with a fresh finite element analysis in real units.

It is the open counterpart of the generative design in Fusion or nTop, built
from standard, well-documented methods so the results can be trusted and the
code can be read:

- **SIMP topology optimization** on a voxel grid: density per cell, penalized
  stiffness, a cone density filter for a minimum member size, Heaviside
  projection with continuation for crisp 0/1 designs, optimality-criteria
  updates, multiple load cases, symmetry planes.
- **Overhang control** for a chosen build direction (Langelaar's AM filter):
  material that would need support earns nothing, so parts grow their own
  45° supports.
- **Minimum member size** that actually holds: stiffness is credited to the
  eroded design and the delivered part is the opened one (robust
  formulation), so no sub-minimum skins or webs.
- **MMA** (Svanberg's Method of Moving Asymptotes, implemented here from the
  published method) for minimum mass subject to stress and deflection
  limits, with relaxed, adaptively normalized P-mean stress aggregation and
  adjoint sensitivities.
- **Linear elastic FEA** with 8-node hexahedra, solved by conjugate gradient
  with a smoothed-aggregation multigrid preconditioner (pyamg). Checked against
  Euler-Bernoulli beam theory in the tests.
- **Exact reconstruction**: marching cubes on the smoothed density, then
  manifold3d booleans that add back the preserved geometry and cut the
  keep-outs exactly, so a 22mm bearing seat is 22mm.
- **Verification of the delivered mesh**, not the density field: mass, peak
  von Mises stress and safety factor per load case, deflection, keep-out
  intrusion and preserve loss (exact booleans), body count, thinnest member.

Units are fixed: millimetres, newtons, megapascals, g/cm³.

## Quick start

```bash
pip install ./loadpath[render]           # or: pip install loadpath, once published
python -m loadpath check study.json          # validate, show the voxel problem
python -m loadpath run study.json --out out  # optimize, mesh, verify, render
```

`out/` gets one STL and one render per outcome, and `report.json` with the
numbers. Progress streams to stderr as one JSON object per line.

### For agents (MCP)

`pip install "loadpath[mcp]"`, then add a local server to any MCP client
(Claude Code, Cursor, Claude Desktop):

```json
{"mcpServers": {"loadpath": {"command": "loadpath", "args": ["mcp"]}}}
```

Tools: `loadpath_guide` (the study format), `loadpath_check`,
`loadpath_start`, `loadpath_status` (outcomes, STL paths and a render),
`loadpath_cancel`. Everything runs on the agent's own machine; jobs live in
`~/.loadpath/jobs` (`LOADPATH_HOME`).

From Python:

```python
from loadpath.spec import study_from_dict
from loadpath.study import run_study

result = run_study(study_from_dict(study))
best = result.outcomes[result.best]
best.mesh.export("part.stl")
print(best.report["mass"], best.report["safetyFactor"])
```

## The study

```json
{
  "name": "arm-link",
  "designSpace": [{"type": "box", "min": [-16, -18, -12], "max": [156, 18, 12]}],
  "preserve": [
    {"id": "shoulder-boss", "shape": {"type": "cylinder", "center": [0, 0, 0], "axis": "z", "length": 24, "r": 16}},
    {"id": "wrist-boss", "shape": {"type": "cylinder", "center": [140, 0, 0], "axis": "z", "length": 24, "r": 15}}
  ],
  "keepOut": [
    {"id": "shoulder-bearing", "shape": {"type": "cylinder", "center": [0, 0, 0], "axis": "z", "length": 30, "r": 11}},
    {"id": "wrist-bearing", "shape": {"type": "cylinder", "center": [140, 0, 0], "axis": "z", "length": 30, "r": 11}},
    {"id": "cable-channel", "shape": {"type": "cylinder", "p0": [30, 0, 0], "p1": [110, 0, 0], "r": 5}}
  ],
  "supports": [{"region": "shoulder-bearing"}],
  "loadCases": [
    {"name": "payload", "loads": [{"region": "wrist-bearing", "force": [0, -20, 0]}]},
    {"name": "side", "loads": [{"region": "wrist-bearing", "force": [0, 0, 8]}]}
  ],
  "material": "pa12",
  "goal": {"safetyFactor": 3.0},
  "minMember": 3.0,
  "symmetry": ["z"],
  "resolution": {"maxElements": 40000}
}
```

| Field | Meaning |
| --- | --- |
| `designSpace` | Shapes material may occupy (their union). |
| `preserve` | Geometry kept exactly: bosses, mounting faces, tabs. Pinned solid in the optimization and unioned back exactly. |
| `keepOut` | Volume that must stay empty: bores, bolt access, cable runs, mating parts. Cut exactly at the end. |
| `supports` | Where the part is held: a region id or an inline shape, with `fix` = `"all"` (default), axes like `"xz"`, or three booleans. |
| `loadCases` | Each a list of forces (N) on regions. The optimizer stiffens for all cases together; verification reports each. |
| `material` | A name (`pla`, `petg`, `abs`, `asa`, `pa-cf`, `pa12`, `pa11`, `resin`, `resin-tough`, `alsi10mg`, `ti64`, `316l`) or `{E, nu, strength, density}`. A list solves the goal for each and pools the outcomes. |
| `goal` | `{"volumeFraction": 0.25}` keeps that share of the design space. `{"safetyFactor": 3}` and/or `{"maxDeflection": 0.5}` (mm, at the loads) search for the lightest part that meets them in every case. `outcomes: [0.2, 0.4]` adds a trade-off gallery. `"method": "constrained"` (experimental) minimizes mass with the limits as MMA constraints instead of searching over volume fractions. |
| `extrusion` | Optional axis (`x`, `y`, `z`): the part is a profile extruded along it, cuttable from plate (waterjet, laser, 2.5-axis milling). |
| `gravity` | Optional `[gx, gy, gz]` in m/s²: the part's own weight in every load case. |
| `printDirection` | Optional, `+x` … `-z`: the direction layers stack. Turns on overhang control (Langelaar's AM filter), so the optimizer's own surfaces are self-supporting at 45°. |
| `minMember` | Thinnest member in mm (sets the filter radius). |
| `symmetry` | Planes through the design-space centre, by normal axis. |
| `resolution` | `maxElements` (default 60,000) or an explicit `pitch` in mm. |

Shapes: `box` (`min`/`max` or `center`/`size`), `cylinder` (`p0`/`p1` or
`center`/`axis`/`length`, with `r` and optional `rInner` for a tube), `sphere`,
and `mesh` (a closed STL by `path`, relative to the study file, or inline as
`stlBase64`).

A load or support on a keep-out means "through the material lining it", which
is how a shaft pushes on a bore or a bolt holds a hole. Loads take a `force`
(N) and/or a `moment` (N·mm, about the region's centroid: a motor's torque);
`"distribution": "bearing"` spreads a force over the half of a bore it pushes
on, as a shaft does. Supports take `"fix": "pin"` on a cylinder for a joint
that turns: held radially and axially, free to rotate.

## Reading the report

- `safetyFactor` is the material strength over the peak von Mises stress,
  worst case. Stress within one optimization cell of where a load or support
  is applied is left out of the peak (`contactBand`): an idealized load
  produces stress there that belongs to the idealization, not the part. The
  analysis is linear and the material numbers are typical datasheet values;
  printed parts are weaker across layers. Treat it as an engineering estimate
  for comparing designs and catching an undersized part, not a certification.
- `keepOutIntrusion` and `preserveMissing` are mm³ from exact booleans and
  should be zero to rounding.
- `p5Member` is the 5th-percentile member thickness on the medial ridge.

## Evals

`python -m loadpath.evals.run --out evals-out/` runs real parts with hard
requirements (a robot arm link, a shelf bracket, a NEMA 17 motor mount with an
STL keep-out, an aluminium bell crank, a wall hook printed without supports,
the textbook MBB beam) and scores the delivered meshes. Exit code is the
number of failing cases. Run it after any change to the solver.

## Print or host the result

A part that passes its report can be printed in dozens of materials or hosted
and shared on [Materialize](https://www.materialize.cc). Nothing in loadpath
calls out to it; it is just where the STL goes next.

## License

MIT.

## Limits

- Linear elastic, small deflection, isotropic. No buckling, fatigue or
  contact; no anisotropy for printed layers.
- Overhang control is the 45° stencil only, along a grid axis.
- Peak stress from a voxel model is refined at hotspots and carries a 1.15
  allowance in the safety factor; it lands within about ±13% of the
  textbook plate-with-hole value in the tests.
- Pure Python and SciPy. A 40,000-cell study with two load cases takes a few
  minutes on a laptop core; a safety-factor search runs several of those.
