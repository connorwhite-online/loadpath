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
  45° supports. Any direction, including tilted ones (`am.TiltedAMFilter`).
- **Print-aware design** (`process`): layered (transversely isotropic)
  stiffness about the build direction in verification, with the layer bond
  checked; the safety-factor search sizes the part against it, and the
  optimizer can design with the layered stiffness too (`anisotropic`, off by
  default: on the motor mount and hook it did not beat isotropic, see am.py). `"buildDirection": "auto"` compares the six directions on a coarse
  grid and keeps the strongest per gram (`orient.py`). Every layered result
  is sliced like a slicer would (`printcheck.py`: bridges up to 15 mm are
  fine, mid-air islands, support volume, bed contact and tipping) and written
  as a 3MF standing the way it was designed to print (`threemf.py`).
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

Tools: `loadpath_guide` (the study format), `loadpath_schema`,
`loadpath_examples`, `loadpath_check`, `loadpath_start`, `loadpath_status`
(outcomes, STL paths and renders), `loadpath_lattice`,
`loadpath_heat_exchanger`, `loadpath_size_heat_exchanger`, `loadpath_jobs`,
`loadpath_cancel`, and checks on any mesh file: `loadpath_check_printability`,
`loadpath_check_isolation`, `loadpath_check_fit`. Everything runs on the agent's own machine; jobs live in
`~/.loadpath/jobs` (`LOADPATH_HOME`). See [docs/agents.md](../docs/agents.md).

From Python:

```python
from loadpath.spec import study_from_dict
from loadpath.study import run_study

result = run_study(study_from_dict(study))
best = result.outcomes[result.best]
best.mesh.export("part.stl")
print(best.report["mass"], best.report["safetyFactor"])
```

### See it in 3D

`loadpath.viewer` writes one HTML file a person can open in any browser: orbit
the part, colour it by von Mises stress, safety factor or displacement for
each load case (or the worst of them), scrub or animate an exaggerated
deformation, cut it open with a section plane, hover to read the value under
the pointer, put two outcomes (or the part it replaces) side by side, and play
the optimization back iteration by iteration, every run of a safety-factor
search in order.

```python
from loadpath.viewer import HistoryRecorder, study_viewer

rec = HistoryRecorder()                        # keeps the density field every 2 iterations
result = run_study(study, on_progress=rec)
study_viewer(result, "part.html", history=rec, outcomes="all",
             baseline="old-part.stl")          # optional: verified under the same loads
```

The page carries its own data (a few MB) and loads three.js from a CDN.
`heat_exchanger_viewer(mesh, spaces, report, "hx.html")` does the same for
`design_heat_exchanger(..., fluid_meshes=True)`: each fluid is a layer, and
the section plane shows the two networks interleave. `Viewer` is the scene
underneath, for anything else (`add_part` with fields, `add_overlay`,
`add_arrow`, `add_study_context`).

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
| `keepOut` | Volume that must stay empty: bores, bolt access, cable runs, mating parts, a fluid's volume. Cut exactly at the end. `wall: 1.5` keeps 1.5 mm of material round it, pinned solid (a duct's or plenum's skin; see pressure below). |
| `fasteners` | Optional bolted joints as interfaces: `{id, size: "M5", seat, direction, grip, end: "face"\|"thread"\|"none", headOn: "this"\|"mating"}`. A screw driven from the mating part into this one is `headOn: "mating", end: "thread"`. Adds a seat ring and a face ring or thread sleeve (preserve), the bore and the head and driver path (keep-outs, `<id>-bore`, `<id>-head`), and a self-balanced `preload` case, so the optimizer grows the clamp path instead of keeping a solid boss. See `fasteners.py`. |
| `supports` | Where the part is held: a region id or an inline shape, with `fix` = `"all"` (default), axes like `"xz"`, or three booleans. Optional when every load is a pressure on a sealed cavity. |
| `skin` | Optional skin-and-rib: `1.2` or `{thickness: 1.2, open: ["-z"]}` keeps that many mm of the design space's outer surface as a solid wall (except on the `open` faces) and designs only what is behind it, so the optimizer grows ribs under a skin instead of a free-form part. |
| `loadCases` | Each a list of forces (N) on regions. The optimizer stiffens for all cases together; verification reports each. |
| `material` | A name (`pla`, `petg`, `abs`, `asa`, `pa-cf`, `pa12`, `pa11`, `resin`, `resin-tough`, `alsi10mg`, `ti64`, `316l`) or `{E, nu, strength, density}`. A list solves the goal for each and pools the outcomes. |
| `goal` | `{"volumeFraction": 0.25}` keeps that share of the design space. `{"safetyFactor": 3}` and/or `{"maxDeflection": 0.5}` (mm, at the loads) search for the lightest part that meets them in every case. `outcomes: [0.2, 0.4]` adds a trade-off gallery. `"method": "constrained"` (experimental) minimizes mass with the limits as MMA constraints instead of searching over volume fractions. |
| `extrusion` | Optional axis (`x`, `y`, `z`): the part is a profile extruded along it, cuttable from plate (waterjet, laser, 2.5-axis milling). |
| `gravity` | Optional `[gx, gy, gz]` in m/s²: the part's own weight in every load case. |
| `process` | Optional: `"fdm"`, `"sls"`, `"mjf"`, `"sla"`, `"volumetric"`, or `{type, buildDirection, interlayer, stiffnessAcrossLayers, shearAcrossLayers, anisotropic (false \| true \| "bond"), minFeature, overhangs}`. `buildDirection` is `+x` … `-z`, a vector `[x, y, z]`, or `"auto"`. Sets a minimum member, overhang control (FDM), layered stiffness and the layer-bond check. |
| `printDirection` | Optional, `+x` … `-z` or a vector `[x, y, z]`: the direction layers stack. Turns on overhang control (Langelaar's AM filter), so the optimizer's own surfaces are self-supporting at 45°. |
| `minMember` | Thinnest member in mm (sets the filter radius). |
| `symmetry` | Planes through the design-space centre, by normal axis. |
| `resolution` | `maxElements` (default 60,000) or an explicit `pitch` in mm. |
| `process` | `"fdm"`, `"sls"`, `"mjf"`, `"sla"`, `"volumetric"`, or `{type, buildDirection, interlayer, minFeature, overhangs}`: sets a minimum member, overhang control (FDM) and the layer-bond check. |
| `lattice` | Optional: also fill the best outcome with a stress-graded TPMS lattice, `{type: "gyroid"\|"diamond"\|"schwarz-p", cell, skin, minWall, maxWall, solidAbove, safetyFactor}`. `skin: 0` leaves it open so powder or resin drains. |

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

A **pressure** acts in a keep-out, the fluid's volume:
`{"region": "plenum", "pressure": 0.25}` is 2.5 bar gauge (MPa; negative is a
vacuum) on every wall lining it, normal to the wall. The keep-out keeps a
`wall` (default `minMember`) that is pinned solid, so the pressure boundary
cannot be optimized open and the load does not depend on the design; the
optimizer then puts material behind the wall where it stiffens it: ribs.
A sealed cavity's pressure balances on itself, so a study whose only loads
are pressures needs no `supports` (it is held by a statically determinate
3-2-1 support that takes no load); an open duct's or a flanged housing's
needs them where the part is held, because the thrust on its open ends goes
there. The report's `deflection` for a pressure-only case is the most any
wetted wall bulges. Checked against Lamé (tests/test_pressure.py): a thick
sphere reads 5-7% over the exact bore stress and 6-9% over its bulge; a
closed thin cylinder (3.7 cells through the wall) 8-11% over the hoop
stress. Verification resolves the thinnest wall or skin with four cells (up
to 400,000 cells): a wall voxelizes to a whole number of cells, and at two
or three a 2 mm plenum wall read 2.4 mm and 30% under its bending stress; at
four, the flat wall of a 90 x 60 mm plenum reads the clamped-plate centre
stress (50 MPa at 2.5 bar). Walls under two optimization cells are warned
about. The tetrahedral cross-check does not apply pressure yet.

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
- `print` (layered processes): `supportVolume`/`supportArea` a slicer would
  add, `bridges` and `maxBridgeSpan`, `islands` (starts in mid-air),
  `bed` (`contactArea`, `feet`, `smallestFoot`, `comMargin` = how far the
  centre of mass sits inside the contact, `aspect`), and `warnings`.
- `python -m loadpath orient part.stl [--study s.json] [--3mf part.3mf]`
  ranks build orientations of any mesh (the axes and its own flat faces) by
  layer-bond safety under the study's loads, support and bed risk.

## Evals

`python -m loadpath.evals.run --out evals-out/` runs real parts with hard
requirements (a robot arm link, a shelf bracket, a NEMA 17 motor mount with an
STL keep-out, an aluminium bell crank, a wall hook printed without supports,
the textbook MBB beam) and scores the delivered meshes. Exit code is the
number of failing cases. Run it after any change to the solver.

## License

MIT.

## Limits

- Linear elastic, small deflection. Printed layers are modelled as
  transversely isotropic stiffness plus a tension-across-layers strength
  check; interlayer shear is not checked separately. No buckling, fatigue or
  contact.
- Overhang control is the 45° stencil on the voxel grid; off-axis directions
  use the 26-neighbour cells within 45° of straight down, which is coarser.
  Bridges are recognized by the print check, not designed for.
- Peak stress from a voxel model is refined at hotspots and carries a 1.15
  allowance in the safety factor; it lands within about ±13% of the
  textbook plate-with-hole value in the tests.
- Pure Python and SciPy. A 40,000-cell study with two load cases takes a few
  minutes on a laptop core; a safety-factor search runs several of those.
