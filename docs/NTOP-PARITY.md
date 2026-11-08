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
| Linear / circular patterns of bodies | `expr` with `mod`; studies' circular `pattern` | Partial |
| Rib design (graphs projected on faces, ribs from graphs) | `cores` ribbed box sections; skin-and-rib studies | Partial |

## Lattices

| nTop | loadpath | Status |
| --- | --- | --- |
| Graph unit cells (cubic, BCC, FCC, octet, Kelvin, diamond ...) | `struts.py`, nine cells | Have |
| Custom unit cell | `type: custom` with `cellGraph` nodes and beams | Have (new) |
| TPMS (gyroid, diamond, Schwarz P, lidinoid, split P, neovius), walled TPMS | gyroid, diamond, Schwarz P as sheet or network | Partial |
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
| Mass, centre of mass, moment of inertia responses | `report.massProperties` on every part; `massProperties.axis` weighs inertia into the design, `maxInertia` caps it | Have (new); centre of mass reported, not constrained |
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
| CFD | lattice-Boltzmann flow for exchangers and lattice cells (`flow.py`, `conjugate.py`), not general external flow | Partial |
| Nonlinear, contact, fatigue, creep, impact | `nonlinear.py`, `contact.py`, `durability.py`, `impact.py` | Beyond nTop |

## Design for additive manufacturing

| nTop | loadpath | Status |
| --- | --- | --- |
| Build orientation, minimum support orientation | `orient.py` | Have |
| Overhang, thickness, trapped powder | `check_dfm`, `processcheck.py`, `drains.py` | Have |
| Slicing, hatching, slice export (CLI) | `loadpath slices`: contours and hatches in a CLI file (`layerfiles.py`); FDM G-code through a slicer (`slicer.py`) | Have (CLI new); SLC, EOS and Renishaw native formats not yet |
| Build volumes, nesting | | Missing |

## Export and exchange

| nTop | loadpath | Status |
| --- | --- | --- |
| STL, 3MF, OBJ | yes | Have |
| CAD body from implicit (STEP / Parasolid) | STEP from fields and lattices (`step.py`, `latticestep.py`) | Have |
| FE mesh and model export | CalculiX / Abaqus `.inp` | Partial (no Nastran) |
| Voxel grids | `.npz` signed distance in and out | Partial (no OpenVDB) |

## Workflow and automation

| nTop | loadpath | Status |
| --- | --- | --- |
| nTop Automate, notebooks, JSON inputs | CLI, JSON studies and graphs with schemas, Python API, local MCP server | Have |
| Design exploration | `explore` (method x material, cost and lead time, Pareto), `goal.outcomes` | Have |

## What is left

By count: about 52 rows have it, 7 partial, 2 missing. The missing ones
are build volumes and nesting;
the partial ones that matter most are the three TPMS nTop has and loadpath
does not, Nastran export and general CFD. The open problem nTop does not
solve either is physical validation of printed parts
([VALIDATION.md](VALIDATION.md), [CALIBRATION.md](CALIBRATION.md)).
