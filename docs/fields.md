# Field-driven design

`loadpath field` builds a part from a graph of implicit fields, the way nTop
does: every solid is a signed distance (mm, negative inside), every other
quantity is a scalar field, and any number in the graph can be replaced by a
field. The wall follows the stress, the lattice cell grows away from the
bolts, the fillet is larger where the load comes in.

```
loadpath field part.json --out out/          # mesh it, check it, write part.stl + report.json
loadpath schema field                        # the JSON Schema
```

From Python: `FieldGraph(doc).mesh()` or `run_field(doc, out_dir)` in
`loadpath.fields`; agents can call `loadpath_field` over MCP.

## Editing in the browser

```
loadpath edit part.json                      # opens http://127.0.0.1:8765: edit, preview, Save, Run
loadpath edit part.json --html editor.html   # one self-contained page, no server
```

![the field editor on the arm link](images/fields/editor.png)

The editor shows the graph as nodes and wires, an inspector for the
selected node, and a live preview of the part above them. Drag from a node's
output dot onto another node's port to wire a field into a parameter; pick
"number" in the inspector to unwire it. Every number has a slider. The
preview redraws as a slider moves: the page compiles the graph to a GLSL
signed-distance function and raymarches it on the GPU, and every number in
the graph is a shader uniform, so only a change of structure (an op, a
wire, an expression) recompiles, in milliseconds.

- **Solid** shows the output or the selected node; **Colour** paints any
  scalar node on it (the stress, a wall ramp, a distance); a **section**
  cuts it open and paints the cut face the same way, which is how you see
  a graded lattice inside its skin. Hover reads the position and the
  colour field's value.
- What the GPU cannot evaluate is baked on the server and sent as a 3D
  texture: a `mesh` node's distance, a `result` node's stress (its study
  runs on load: about a minute for the arm link), a `grid`, and the cell
  warp of a graded lattice (the same map the mesher solves, so the
  preview's walls sit where the part's will; it re-bakes in under a
  second as a cell slider moves). A bake that a later edit made stale
  stays on screen until its replacement arrives; an analysis re-runs only
  when asked, and later bakes and Run reuse it while its part is unchanged.
- **Colour → wall thickness** paints every wall by its thickness, traced
  in along the surface normal: red under the process's smallest feature
  (`minFeature`: FDM and SLS 0.8 mm, SLA 0.5, LPBF 0.4, ...), amber under
  twice it. Hover reads the thickness. On the arm link it shows the skin
  over the gyroid's voids at 1 to 1.6 mm, thicker where a lattice wall
  backs it.

  ![wall thickness on the arm link](images/fields/editor-walls.png)

  **Colour → overhangs** paints faces that point down past 45 degrees
  from vertical along the build direction (the graph's `build`, else +z)
  red, and those within 10 degrees of it amber; the face on the plate is
  left alone. `minWall` in the graph overrides the process's wall.

- **Mass** is live: the page weighs the output on the GPU as you edit
  (its volume on a grid of samples at two spacings, extrapolated; within
  about 1% of the meshed part) and multiplies by the material's density.
  **Fit mass** finds the value of any one number (a wall, a blend, a remap
  end) that gives a target mass, in a second or two; **Sweep** charts the
  mass across that number's range, and clicking a point uses that value.
- **Run** meshes the graph with `run_field`, exactly as `loadpath field`
  does, at the graph's pitch, then shows the mesh, the report (mass,
  watertightness, sealed voids, drains, printability) and links to the
  STL, report.json and the full viewer. Runs go to `loadpath-edit/` beside
  the graph (`--out`).
- **Save** writes the graph back to its file. Inline nodes get names
  (`owner-key`) when the graph is opened, which is the same graph;
  node positions go under `editor`, which nothing else reads. The JSON tab
  edits the text directly; Ctrl+Z undoes. Ctrl+D duplicates the selected
  node; Ctrl+C and Ctrl+V copy it between graphs, and pasting node JSON
  (`{"nodes": {...}}` or one node) adds it, renaming what clashes and
  keeping the pasted nodes wired to each other.
- The standalone page (`--html`) carries its bakes, so it opens anywhere,
  offline once three.js is cached. It cannot re-bake or run: Run
  downloads the JSON and gives the `loadpath field` command.

The preview is checked against the mesher: `tests/test_editor.py`
evaluates every op on the GPU at 4000 points and compares them with
`FieldGraph.evaluate` (to 1e-6 mm for shapes, booleans, shells, distances,
remaps, expressions and lattices; to the bake's grid for meshes). The
server listens on 127.0.0.1 unless `--host` says otherwise: it reads and
writes files in the graph's folder.

## The graph

```json
{
  "name": "bracket",
  "nodes": {
    "body":  {"op": "mesh", "path": "bracket.stl"},
    "load":  {"op": "result", "study": "bracket-study.json", "part": "bracket.stl"},
    "skinT": {"op": "remap", "input": "load", "from": [2, 20], "to": [1.0, 3.0], "curve": "smoothstep"},
    "skin":  {"op": "shell", "input": "body", "thickness": "skinT"},
    "core":  {"op": "lattice", "type": "gyroid", "cell": 6, "thickness": 0.8},
    "fill":  {"op": "intersect", "inputs": ["core", "body"]},
    "part":  {"op": "intersect", "inputs": [{"op": "union", "inputs": ["skin", "fill"], "blend": 1.0}, "body"]}
  },
  "output": "part",
  "mesh": {"pitch": 0.3},
  "material": "petg"
}
```

A parameter is a number, the name of another node, or an inline node
object. `output` names the node to mesh (default: the last one).
`loadpath check`-style linting runs first: misspelt keys, unknown ops,
references to nodes that don't exist, cycles and bad expressions all come
back at once with suggestions.

## Operations

| op | what it is | fields it takes |
| --- | --- | --- |
| `shape` | exact box, cylinder, tube, sphere or union (same JSON as a study) | |
| `mesh` | a closed STL/3MF/OBJ as a solid; distance exact near the surface | |
| `constant`, `coordinate` | a number; position along x/y/z or a direction | |
| `distance` | unsigned distance to a solid (`input`), points, a segment, a plane or a shape | `input` |
| `result` | von Mises or displacement of a part under a study (run on demand, also on one of the graph's own nodes), or a saved `.npz` | |
| `grid` | an imported scalar field on a regular grid (`.npz`: origin, pitch, values) | |
| `remap` | `from` [a, b] to `to` [c, d], linear / smoothstep / smootherstep / sqrt / square or a drawn curve (`[[u, v], ...]` over the unit square; the editor has a curve editor), clamped | `input` |
| `expr` | arithmetic over node names and x, y, z: `"0.8 + 1.2 * smoothstep(5, 30, stress)"` | `vars` |
| `union`, `intersect`, `subtract` | booleans; `blend` is the fillet radius | `inputs`, `blend` |
| `shell` | a wall inside, outside or centred on a surface | `input`, `thickness` |
| `offset` | grow or shrink | `input`, `distance` |
| `lattice` | gyroid, diamond or Schwarz P; sheet or network; cartesian or cylindrical frame | `cell`, `thickness` or `density`, `offset` |
| `transform` | translate, rotate, scale an input | `input` |

`expr` knows `abs sqrt exp log sin cos tan floor ceil sign min max clamp
lerp smoothstep step where`, comparisons and `a if c else b`; nothing else
(no attributes, no imports).

## What the field-driven operations mean

- **Variable wall** (`shell` with a field `thickness`): the wall at each point
  of the surface is the field's value there. Drive it from `result` to put
  material where the stress is. `"scale": "peak"` on the result divides it
  by the part's own peak (99.5th percentile), so the remap reads in
  fractions of the worst stress (`"from": [0.1, 0.8]`) and works whatever the
  load level.
- **Variable fillet** (`blend` as a field): a smooth union whose radius
  changes along the joint. Larger where members meet under load, small where
  they don't.
- **Graded lattice**: `cell` as a field bends the lattice's coordinates so
  that one period is one local cell (a Poisson solve for the map whose
  gradient is 1/cell, over the output's box). Walls stay continuous through
  the whole grading, with no seams between cell sizes; where the cell field
  changes direction faster than the map can follow, cells come out a few
  percent off square. `grading: "blend"` instead cross-fades lattices of
  fixed periods (needed with a cylindrical frame; it leaves crumbs where
  they disagree). `density` uses the homogenized tables in `lattice.py` to
  turn a relative density into a wall.
- **Cylindrical lattice** (`"frame": {"type": "cylindrical", "axis": "z",
  "radius": 20}`): cells wrapped round an axis, with a whole number of cells
  round the circumference, for tubes and rings.

## Meshes come out smooth

Marching cubes alone leaves every vertex on a grid edge, up to a third of a
pitch off a curved surface. `field_mesh` then relaxes the vertices along the
surface and projects each one onto the field's zero level along its
gradient, so vertices sit on the true surface to about 1e-5 mm: a bore is
round and a sweep is smooth at any pitch. A crease (two flat faces meeting)
comes out rounded over about one pitch; when an edge must be exact (a
bearing seat, a mating face), mesh the organic part with fields and cut the
feature with `sdf_kit.mesh_subtract`.

Pick the pitch from the thinnest wall: a third of it or finer.

### Very large parts

A fine lattice through a big part is tens of millions of triangles, more
than fits in memory as one mesh. `"mesh": {"stream": true}` (or
`loadpath field --stream`) writes the STL slab by slab instead: memory is
one slab of blocks 32 cells thick whatever the part, there is no cell
budget, and the report still has the volume, area, mass and a watertight
check (each slab's open edges are matched against its neighbours'). The
checks that need the whole mesh (printability, bodies, the render) are
skipped. Field evaluation runs on every core (`LOADPATH_THREADS` caps it).

## Drain holes

A skin over a lattice traps whatever the lattice was printed in: powder
(SLS, MJF, metal powder bed) or uncured resin (SLA, DLP, volumetric). With
`"process": "sls"` (or `mjf`, `sla`, `dlp`, `volumetric`, `lpbf`) the
graph drills its own drain holes before it is meshed:

```json
{"process": "sls", "drains": {"keepClear": ["bores"]}}
```

1. The output is sampled on a grid (the mesh pitch) and every void that
   cannot reach the outside is found, down to the slivers where a lattice
   wall grazes the skin.
2. Voids smaller than the escape ball (`fillBelow`) are filled solid.
3. Every other void gets `count` holes (default 2), then more until no
   powder is over `maxPath` (default 60 mm) from one. Powder removal is
   where a part fails in production, so the holes are well over the
   process's minimum escape hole: 1.5 x it or 2 mm over, whichever is
   larger (SLS 6 mm, MJF 7.5, SLA/DLP 5.5, volumetric 5; LPBF 6.8, an M8
   tap drill, so it can be plugged). `diameter` and `"maxPath": null`
   take it back toward the minimum. The
   sites are ranked by the graph's own stress (the first `result` node, or
   `stress`), calmest first: a hole is a stress raiser. The first goes
   lowest along the build and the second highest, well apart, so air or
   rinse flows through, and resin has a drain at the bottom and a vent at
   the top. A site has to be on the void's wall just under the skin, with
   a straight way out, on a face flat out past the boss (no edges, no
   bores or seats: a concave face is refused), and clear of `keepClear`.
4. Each hole gets a solid boss (1.5 x the minimum wall, smooth-unioned in)
   so the bore has a clean wall and the lattice walls it cuts are tied in.
5. The grid is labelled again; anything a boss walled off gets the same
   treatment, up to four rounds.

The holes, bosses and fill are ordinary nodes added to the graph
(`drain-bore-1`, `drain-boss-1`, ..., output `drained`). The report's
`drains` says what was done and checks it: `sealedAfter` (on the grid),
each void's `farthestFromOpening` (the powder path, mm) and
`unreachedVolume`, processcheck's escape ball on the drilled field and on
the delivered mesh (`meshEscape`), and the mesh's own `enclosedVoids`.
`drains.ok` is true only when all of them are clear. `"drains": false`
turns it off; `"drains": true` drills for any process.

Sites are chosen calm first: the stress a hole sees is the worse of its
inner end's and its mouth's (a bending part is calm at the core, not at
the skin). The holes `maxPath` adds go where the powder is farthest,
calmest first, so they sit hotter than the first two.

On the arm-link example in PA12 (`"process": "sls"`): 47 sealed voids
(both gyroid labyrinths, the cable channel and 378 slivers by the bearing
collars, 16.6 cm^3) become 14 holes of 6 mm and 0.4 cm^3 of fill. No
powder is more than 50 mm from a hole (two minimum-size holes a void left
it 124 mm away). The meshed part has no sealed shell and the 4 mm escape
ball reaches every chamber. The first holes sit at 4 to 15% of the peak
stress, the added ones at 17 to 30%. Under the arm-link study
(verify_part, 400k cells): undrilled SF 13.4 and 0.587 mm, drilled SF
13.5 and 0.647 mm, 0.9 g lighter. The holes cost about 10% of stiffness,
not strength.


## Pitfalls

- A big `blend` in a union bulges past the inputs by up to blend/4: a link
  whose bosses are unioned to a web with a 12 mm blend gets domed boss
  faces. Intersect with a slab (or the design box) afterwards.
- Union a lattice with its skin inside the envelope, then intersect the
  result with the envelope again, or the blend pushes lattice ridges through
  the outer skin.
- A closed skin over a lattice seals the voids in. The report counts them
  (`enclosedVoids`, `enclosedVoidVolume`). Give the graph a `process` and
  it drills them (next section); without one nothing is drilled.
- A bare lattice fills all of space: intersect it with a solid, or give the
  graph `bounds`.

## Example

[`examples/fields/arm-link-graded.json`](../examples/fields/arm-link-graded.json)
is the arm-link example rebuilt as a field part: an organic envelope from
smooth-unioned bosses and web, analyzed solid under the example's loads by a
`result` node; that stress sets the skin thickness and the gyroid wall, the
gyroid cell grows away from the bearings, solid collars ring the bearing
seats, a liner wraps the cable channel and the fillet where everything meets
is larger near the bearings. `python examples/fields/arm_link.py` meshes it,
and a uniform lattice of about the same mass, and verifies both.
