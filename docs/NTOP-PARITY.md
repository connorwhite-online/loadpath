# loadpath against nTop, block by block

nTop is the bar for engineering design software that works in fields:
implicit geometry, lattices, simulation, topology optimization and print
preparation in one graph. loadpath aims to match it as a library an agent
drives (JSON studies and field graphs, a CLI, Python, a local MCP server)
rather than a GUI, and to go past it where an agent needs more: checks on
the delivered part, process physics, and designing the inside of a fixed
exterior.

This page holds loadpath against nTop's own block reference
([docs.ntop.com](https://docs.ntop.com), about 740 blocks), area by area,
and names the loadpath module that is the evidence for each row. Speed and
the interactive GUI are left out: speed is tracked in
[BENCHMARKS.md](BENCHMARKS.md), and the browser editor (`loadpath edit`)
exists but is not where the work goes. Last audited 2026-11-08.

**Have** means a study or graph can do what the nTop blocks do, with a
test. **Partial** means part of it. **Missing** means not yet.

## Implicit modelling

| nTop | loadpath | Status |
| --- | --- | --- |
| Primitives, booleans, smooth booleans, shell, offset, transforms | `fields.py` ops `shape`, `union`/`intersect`/`subtract` with `blend`, `shell`, `offset`, `transform` | Have |
| Field-driven parameters (variable thickness, fillet, cell) | every numeric parameter takes a field | Have |
| Ramps, remap, math blocks | `remap` (curves, drawn curves), `expr` | Have |
| Image to field | `image` | Have |
| Implicit body from mesh / CAD | `mesh` (exact distance near the surface), `loadpath exterior`, `loadpath sdf`, CAD import | Have |
| Custom blocks | `group` | Have |
| Noise fields (simplex, cellular) | `noise`, `cellular` in `expr` | Have (new) |
| Surface textures (emboss, deboss, bubbles, roughness) | an `offset` by a noise, image or lattice field ([fields.md](fields.md#noise-and-surface-textures)) | Have (new) |
| Point maps (scattered FE / CFD data as a field) | `result` reads `.csv` / `.txt` / `.npz`, idw, thin-plate spline or nearest ([fields.md](fields.md#point-maps)) | Have (new) |
| Linear / circular / mirror / along-curve patterns of bodies | field `pattern` op: linear grids, circular (turned or not, part sweeps), mirror and along a curve, each point reading its nearest copies ([fields.md](fields.md#patterns)); studies' circular `pattern` for symmetric design | Have (new) |
| Rib design (graphs projected on faces, ribs from graphs) | field `ribs` op: curves projected onto a face (walls along a pull direction or square to the surface), square / isogrid / radial patterns, and ribs along the face's principal stress lines under a study, 7% stiffer than a square grid at equal mass on a plate ([fields.md](fields.md#ribs)); `cores` ribbed box sections | Have (new) |

## Lattices

| nTop | loadpath | Status |
| --- | --- | --- |
| Graph unit cells (cubic, BCC, FCC, octet, Kelvin, diamond ...) | `struts.py`, nine cells | Have |
| Custom unit cell | `type: custom` with `cellGraph` nodes and beams | Have (new) |
| TPMS (gyroid, diamond, Schwarz P, lidinoid, split P, neovius), walled TPMS | all six as sheet or network, field-graded, homogenized fits for each (`lattice.py`), 3MF implicit export | Have (lidinoid, split P, Neovius new, with full 6x6 tensors and stress surfaces) |
| Cell maps: rectangular, cylindrical, spherical, conformal | `frame` cartesian, cylindrical, spherical, conformal (frame field) | Have |
| Stochastic: Voronoi, Delaunay | `voronoi`, `delaunay` | Have (Delaunay new) |
| Tet lattice from a body, conformal lattice from a volume mesh | `delaunay` with a conformal frame: nodes on the skin, struts inside | Have (new) |
| Field-graded thickness and cell size | `thickness`, `density`, `cell` fields; `lattice-grade` sizes them to the stress | Have, and more |
| Homogenization | full 6x6 tensors and stress surfaces per cell, as-printed knockdown, unit-cell buckling, orientation optimization (`lattice_tensors.py`, `lattice_print.py`, `lattice_buckling.py`, `lattice_orient.py`) | Have, and more |
| Beam lattice export | 3MF beam lattice (`lattice3mf.py`), exact lattice STEP (`latticestep.py`) | Have |

## Topology optimization

| nTop | loadpath | Status |
| --- | --- | --- |
| Compliance with a volume or mass target, several load cases | `simp.py`, the search for the lightest part that meets a safety factor and deflection | Have |
| Stress and displacement constraints | `goal.safetyFactor`, `maxDeflection`, the constrained method (`constrained.py`) | Have |
| Natural frequency response and constraint | `vibration.minFrequency` with carried masses ([README](../loadpath/README.md#the-study)) | Have (new) |
| Mass, centre of mass, moment of inertia responses | `report.massProperties` on every part; `massProperties.axis` weighs inertia into the design, `maxInertia` caps it, `centerOfMass` {target, tolerance} pulls the centre of mass to a point on any of its axes in the optimizer and checks it in the report | Have (new) |
| Overhang, min / max feature, symmetry, pattern repetition, extrusion | `printDirection` / `process`, `minMember`, `maxMember`, `symmetry`, `pattern`, `extrusion` | Have |
| Demold, 3- and 5-axis milling | `casting` (draw, sides, draft), `milling` (2.5 / 3 / 5 axis, tool diameter) | Have |
| Passive regions | `preserve`, `keepOut` | Have |
| Thermal topology optimization | `thermal.py` (heat sinks, thermo-elastic) | Have |
| Buckling as a constraint | `bucklingopt.py` (nTop has no buckling constraint) | Have, beyond nTop |
| Field / parameter optimization | `tune`, `lattice-grade`, `explore` | Have |
| Reaction force response | `report.reactions` per support and case (force, moment, share); `maxForce` / `maxMoment` caps held by the search | Have (new); a cap is a check the search holds, not an optimizer gradient |

## Simulation

| nTop | loadpath | Status |
| --- | --- | --- |
| Linear static | voxel hex8 with cut cells, body-fitted tet cross-check, CalculiX deck (`verify.py`, `crosscheck.py`) | Have |
| Modal | `modal.py`: modes, effective masses, damping, transmissibility | Have |
| Linear buckling | `nonlinear.linear_buckling`, on every delivered part | Have |
| Steady and transient thermal | `thermal.py` (cycles, softening, expansion) | Have |
| Lattices as solids, beams or homogenized | resolved check, homogenized check | Have |
| Static analysis with inertia relief (unsupported bodies) | `inertiaRelief: true`: loads balanced by the part's own rigid-body inertia, in the check and the optimizer ([README](../loadpath/README.md#the-study)) | Have (new) |
| CFD | lattice-Boltzmann flow for exchangers and lattice cells (`flow.py`, `conjugate.py`); `loadpath external-flow` puts any part in a wind or water tunnel and writes the surface forces as a load case ([EXTERNAL-FLOW.md](EXTERNAL-FLOW.md); coarse LES, reads high, capped Reynolds number) | Have (new, coarse) |
| Nonlinear, contact, fatigue, creep, impact | `nonlinear.py`, `contact.py`, `durability.py`, `impact.py` | Beyond nTop |
| (none: nTop has no cable or harness model) | `wires.py`, `rods.py`: wires, bundles and flat cables as Cosserat rods with contact, bend radius, twist and flex-life checks through a joint's motion; cables inside the enclosure placement search and their channels kept clear; connector and clip loads; board stacks ordered and spaced by heat, centre of mass and whether their harness can bend ([WIRES.md](WIRES.md)) | Beyond nTop (new) |

## Design for additive manufacturing

| nTop | loadpath | Status |
| --- | --- | --- |
| Build orientation, minimum support orientation | `orient.py` | Have |
| Overhang, thickness, trapped powder | `check_dfm`, `processcheck.py`, `drains.py` | Have |
| Slicing, hatching, slice export (CLI, SLC) | `loadpath slices`: contours and hatches in ASCII or binary CLI, or 3D Systems SLC contours (`--format`, `layerfiles.py`); FDM G-code through a slicer (`slicer.py`) | Have (new); EOS SLI and Renishaw MTT are closed formats with no public spec, and EOSPRINT and QuantAM take CLI in |
| Build volumes, nesting | build-volume fit per material (`dfm_reference.py`); `loadpath nest` packs parts and copies on the plate or stacked in an SLS / MJF cake, gap measured (`nesting.py`) | Have (new) |

## Export and exchange

| nTop | loadpath | Status |
| --- | --- | --- |
| STL, 3MF, OBJ | yes | Have |
| CAD body from implicit (STEP / Parasolid) | STEP from fields and lattices (`step.py`, `latticestep.py`) | Have |
| FE mesh and model export | `loadpath fe-export`: the part in 10-node tets with its material, supports, pins and load cases as a Nastran SOL 101 deck (`.bdf`) or Abaqus / CalculiX (`.inp`) (`fe_export.py`) | Have (new) |
| Voxel grids (OpenVDB) | OpenVDB `.vdb` float grids in and out in pure Python (`vdb.py`): `loadpath sdf part.stl --out part.vdb` writes a narrow-band level set, sdf shapes and exteriors read level sets, the field `grid` op samples any float volume (a Houdini or Blender density, a temperature); zip, blosc, active-mask and half-float files read, checked against the OpenVDB library both ways; `.npz` too | Have (new) |

## Workflow and automation

| nTop | loadpath | Status |
| --- | --- | --- |
| nTop Automate, notebooks, JSON inputs | CLI, JSON studies and graphs with schemas, Python API, local MCP server | Have |
| Design exploration | `explore` (method x material, cost and lead time, Pareto), `goal.outcomes` | Have |

## What is left

By count: about 60 rows have it, none partial, none missing. External flow
is now in, coarse: it reads drag on the high side and caps the Reynolds
number, which is right for structural loads and not for aerodynamic design. The open problem nTop does not
solve either is physical validation of printed parts
([VALIDATION.md](VALIDATION.md), [CALIBRATION.md](CALIBRATION.md)).
