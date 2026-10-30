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
| `remap` | `from` [a, b] to `to` [c, d], linear / smoothstep / smootherstep / sqrt / square, clamped | `input` |
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
