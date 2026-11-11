# Results bundle: the simulation travels with the part

Topology-optimization codes stop at a density field and CAD tools only ever
see the final mesh. Everything that explains a loadpath part is lost at that
handoff: where it is weakest and by which criterion, which load each arm
carries, what each keep-out keeps clear, how hot it runs, how the fluid
moves through it. The results bundle keeps it. Every `loadpath run` writes
one to `out/bundle/`, and `loadpath bundle study.json part.stl` writes one for
any part, so a viewer, a print service, a reviewer or another agent can see
the same evidence the verification produced, in software they already have.

```
out/bundle/
  part.glb        glTF 2.0: the part coloured by its governing safety factor,
                  the real fields as vertex attributes, loads, supports,
                  keep-outs, preserved regions and the weakest point as named nodes
  analysis.vtu    the analysis model itself: the hexahedral cells the part was
                  verified on, with stress tensor, equivalent stress,
                  displacement, safety factor, what limits it, contact band
  surface.vtp     the delivered surface with the same fields read onto it
  part.stl        the part to make
  tradeoff.csv    every candidate the search verified: mass, safety factor,
                  deflection, buckling, which one was delivered
  steps.json      the numbered assembly moves, when the run planned them
  manifest.json   what every field and object means
```

A heat exchanger simulated whole (`porous.simulate_exchanger(...,
fields=True)`) adds `exchanger.vti` (each fluid's velocity, pressure,
temperature and cells, the wall's temperature) and `exchanger.vdb` (the same
as fog volumes) with `bundle.exchanger_grids`; a lattice-Boltzmann network
flow (`flow.network_flow(..., fields=True)`) does the same with
`bundle.flow_grids`. Both pass to `export_bundle(flows=[...])`.

## The manifest is the point

A file can store an array called "SF". It cannot say that it is the least
over the load cases of the pressure-modified von Mises and layer-bond margins,
against its target, read with a 1.15 stress allowance, with the contact
band under the point loads left out. `manifest.json` says exactly that, for
every field (`loadpath schema bundle` is its JSON Schema). From the two-case
bracket in `tests/data/` (`--quick`, a volume-fraction goal, so no target),
printed along -y in FDM:

```json
{"id": "safetyFactor/side", "quantity": "safety factor", "case": "side", "units": "1",
 "criterion": "pressure-modified von Mises equivalent stress against the material's strength; the layer bond along the build direction (bond at 38% of the in-plane strength, tension and shear combined)",
 "target": null, "betterIs": "higher",
 "range": {"min": 0.667, "max": 1468.0, "p01": 1.142, "worstAt": [1.2754, 1.2126, 2.4769]},
 "reported": 0.667, "limitedBy": {"0": "material (in-plane)", "1": "layer bond"},
 "reason": "how many times this case's loads the part carries at each point before the material (or the bond between layers) reaches its strength",
 "excludes": "nodes in the contact band (contactBand = 1) are not part of the reading"}
```

Objects carry their physical reason too: a keep-out is a bore a fastener
passes through, a clearance volume a moving part sweeps, or a pressurized
passage with its pinned wall; a support says what it holds and what reaction
it passes on; a load gives its vector, units and case, and says its drawn
arrow length is display only. The same records sit on the glTF nodes as
`extras.loadpath`, so a viewer that reads only the GLB still knows what it is
showing.

## Physics, not paint

Every value is the verification's own: `verify_part(..., fields=True)` now
returns the nodal safety factor by the same criteria as each case's reported
number (equivalent stress over hot strength retention, the layer bond along
the build direction, the cold end of a thermal cycle), what limits it at each
node, the contact band, the stress tensor and the analysis cells. On a part
verified without submodels the map's least value equals the reported safety
factor exactly (`tests/test_bundle.py`); with submodels the reported number
reads finer at the hotspots and can sit lower, and the manifest says so
(`reported` beside `range.min`). Inside a contact band the governing map
takes the nearest value outside it, as the reported number does; the raw
values stay in `analysis.vtu` with the band marked. Only two things are
display choices, and both are named in the manifest: the colour scale (dark
red under 1, red to amber up to the target, green at twice it, grey where
material carries little) and arrow lengths.

## Units and frames

The files are in loadpath's units (mm, N, MPa, C) and frame (z up). glTF's
convention is metres and y up, so `part.glb`'s root node carries the scale
and rotation between the two: any viewer that honours node transforms shows
the part the right size and the right way up.

## Checked in real viewers

- `part.glb`: the Khronos glTF validator reports 0 errors, 0 warnings, 0
  infos; three.js's GLTFLoader loads it with every attribute
  (`_safety_factor`, `_stress_0`, ...) and node extras, and VTK's glTF reader
  opens it.
- `analysis.vtu`, `surface.vtp`, `.vti`: VTK's XML readers (what ParaView
  uses) read every array back; the hexahedra have positive volume in VTK's
  node order; grid indexing matches point for point.
- `.vdb`: loadpath's OpenVDB writer (`vdb.py`, read by Houdini, Blender and
  OpenVDB tools).

## Listing a part: the Materialize adapter

`loadpath.adapters.materialize.listing("out/bundle")` turns a bundle into a
Materialize file listing: the STL to upload, and metadata (a description that
quotes the safety factor against its target and criterion, mass, buckling,
the loads and the weakest point; design tags such as `strong` and
`lightweight` only when the numbers earn them; the minimum wall; the
recommended material id). It uploads and publishes nothing; an agent passes
the payload to the Materialize MCP server once the owner says to. Materialize
takes STL/OBJ/3MF/STEP/AMF and plain-text metadata today, so the fields
themselves (the GLB's safety-factor map, the analysis, the manifest) are
listed under `notCarried`, with a compact `simulation` record a part page
could show once it takes one. The adapter lives outside loadpath's core: the
bundle is plain open formats, for any service.
