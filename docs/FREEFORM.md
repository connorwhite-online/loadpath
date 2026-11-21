# Freeform surfacing: a skin drawn from curves, checked before anyone looks

A product's outside starts as curves: a few sections across it, the
silhouettes it shows from the front, side and top, a part line. A
surfacing package builds the skin through them. Built instead from boxes,
cylinders and fillets, or from blended distance fields, the skin picks up
what nobody drew: a lump where two blends overlap, a phantom ridge where
one blend runs into another, a flat spot where a fillet runs out. Then the
loop is a person spotting each flaw in a render.

Three modules close that loop in code, on OpenCascade (`pip install
"loadpath[step,surface]"`):

| Module | Does |
|---|---|
| `loadpath.surfacing` | builds surfaces from curves |
| `loadpath.surfacequality` | measures how fair they are |
| `loadpath.sketchmatch` | measures them against the designer's sketches, in mm |

## Surfacing

```python
from loadpath import surfacing as S

c = S.fit_curve(points, closed=True)              # C2, periodic; pencil wobble smoothed
s = S.fit_surface(grid, poles=(20, 10), uperiodic=True)
p = S.fair_patch((x0, x1), (z0, z1), pins, soft=soft)   # the fairest height field through pins
L = S.loft(sections, rails=[silhouette])          # sections warped onto rails, Gordon surface
cap, err = S.cap([(edge, neighbour_face, "G2")])  # N-sided patch carrying its neighbours' curvature
b = S.blend((face_a, edge_a), (face_b, edge_b))   # G2 across a gap; lead=0.7 turns sooner
a2, bl, b2 = S.fillet(face_a, face_b, 3.0)        # two crossing faces, cut back and blended G2
holed = S.cut(face, loop)                         # an opening, its edge on the surface
s2 = S.edit(s, [(0.5, 0.5, (0, 3, 0))], freeze=3) # pull a handle; the surface stays fair
solid = S.sew_solid([L.face(), cap, ...])
S.write_step(solid, "skin.step")                  # opens in Fusion, Rhino, SolidWorks
f = S.sdf(solid, pitch=0.4)                       # exact signed distance, for field pipelines
```

- **One surface, no seams.** `fit_surface`, `network` and `fair_patch`
  each make one B-spline surface, C2 inside. A region that has to carry
  curvature across (a sweep up to a plateau, a grip into a head) is best
  built as one surface: there is then no edge to break.
- **Fair by construction.** Every fit minimizes the surface's bending
  energy (thin plate) against the points it has to meet. `fair_patch` is
  the variational patch: pinned flat where a plateau is, held softly to a
  body away from it, and nothing else. The sweep between is the
  minimum-bending surface, so it has no ridge or dip nobody asked for.
- **Lofts with rails.** `loft` resamples each section by arc length from
  a common start (so the surface does not twist), warps it smoothly round
  its loop until it meets every rail, then surfaces the network of sections
  and rails as a Gordon surface (S = Lu + Lv - T), which interpolates every
  curve.
- **Fills, to G2.** `cap` (and `fill` with "G2") closes an N-sided
  boundary with one trimmed B-spline surface that shares each neighbour's
  position, tangent plane and curvature across the edge. It is the fairest
  such surface, free in all three coordinates, so a tube's end closes as a
  dome. G0 and G1 fills still go to OpenCascade's `BRepFill_Filling`, which
  cannot hold a bounded edge G2.
- **Blends, to G2.** `blend((face_a, edge_a), (face_b, edge_b))` is one
  surface across a gap. Every section leaves face a as that face's own
  curve would and arrives on face b the same way. How far each face's shape
  reaches into the blend (the lead) is the length that changes the
  section's curvature least. `lead` scales it, per side.
- **Fillets that carry curvature.** `fillet(face_a, face_b, radius)` takes
  two faces that cross, cuts each back to the radius from the other's
  surface (`setback`: the face's intersection with the other's offset, so
  the new edge lies on the face exactly) and lays a `blend` across. A
  rolling-ball fillet is an arc, tangent with a curvature jump where it
  starts; this leaves each face with that face's own curvature.
- **Openings.** `cut(face, loop)` cuts an opening along a curve projected
  onto the face, so the new edge lies on the surface exactly, ready for
  `blend` or `cap`. `hole_edges` lists them.
- **Edits.** `edit(surface, moves, hold, freeze)` moves a surface by
  handles with the fairest displacement that does it: a pulled point takes
  a smooth swell with it and no ripple. `freeze=3` leaves the three pole
  rows along each side alone, so joins keep their G2.
- **Two meanings of fair.** `energy="bending"` is the thin plate.
  `energy="jerk"` is the least change of curvature: a sweep leaves a flat
  plateau with its curvature growing from zero (peak curvature a third
  lower, its rate of change a fifth, in the test), and it is what a G2
  constraint needs. A thin plate's curvature is free to jump where a
  constraint ends, so it cannot hold one.
- **Into fields.** `sdf` turns a closed B-rep into a signed distance that
  is exact within a band of the skin. A field model (cuts, shells, halves,
  ducts) then runs unchanged on a surfaced skin, and blends between exact
  fields come out even.

## From a document

An agent that writes JSON, not Python, gives the same steps as a document
(`loadpath schema freeform`; MCP `loadpath_freeform`):

```
loadpath freeform examples/freeform/pod.json --out out/pod
```

`surfaces` are named: a `loft` through sections (each a list of points, or
a ring `{centre, radii, x, y, exponent}`), a patch through a grid of
`points`, a `flat`. `steps` then run in order, each one of `cut` (an
opening along a loop), `fillet` (two crossing surfaces), `blend` (across
the gap between two open edges), `cap` (a loop of open edges closed by one
patch) and `edit` (handles). An open edge is never numbered: it is named by
a point near it (`{"surface": "body", "edge": [0, -10, 0]}` is the body's
open end nearest that point), so a document survives a change of section
count or pole count. Joins are G2 unless a step says otherwise.

Out: `<name>.step`, `.stl`, `<name>-joins.png` (the close-up sheet below)
and `<name>.json`: what each step measured (gap mm, tangent degrees,
curvature jump 1/mm), every join graded, and `closed`. The pod document
builds the eight-face solid of the example below in 10 s, all seven joins
graded curvature.

## Quality

```python
from loadpath import surfacequality as Q

rows, worst = Q.edge_continuity(solid)    # G0 gap, G1 angle, G2 jump on every shared edge
k = Q.surface_curvature(face)             # principal curvatures over a face
js = Q.join_sheet(solid, "joins.png")     # every join graded (crease / tangent / curvature), a close-up of each
r = Q.fairness(mesh, min_radius=2.5, ignore=grooves)
r["wobble"], r["tight"], r["score"]       # flaws, clustered and placed
Q.quality_sheet([before, after], "q.png", cameras)   # zebra + curvature maps
```

`joins` groups the shared edges by the pair of faces they join (a boss's
root, an opening's lip, a cap's rim) and grades each: a crease, tangent
only, or curvature continuous. `join_sheet` frames a zebra close-up on
every one. It reads any B-rep, so a designer's STEP exterior can be
audited for where its reflections will bend before anything is built on it.

`fairness` works on any mesh (a field model's marching-cubes skin, a scan,
a tessellated B-rep). Curvature comes from a weighted quadric fitted at each
vertex in the frame of its smoothed normal, so mesh noise and long thin CAD
triangles do not read as curvature. It flags:

- **tight** spots, where a radius is under the smallest the design means
  to have;
- **wobbles**, where the mean curvature departs from itself smoothed over
  `scale` mm: lumps, dips and phantom ridges. The broad form is the
  smoothed part and is not counted.

`ignore` leaves out what is meant to be sharp (grooves, grilles, rims).
Zebra stripes are the reflection of a row of strip lights. They stay smooth
over a G2 surface, bend sharply at a curvature break, kink at a tangent
break and wobble over a lump.

### Any exterior, read only

`loadpath joins exterior.step` (MCP `loadpath_check_joins`) runs
`join_sheet` on a file and changes nothing. A STEP is read face by face as
above. A mesh has no faces to say where one surface ends, so
`mesh_joins` finds its edges and rounds from the mesh itself: an edge's
dihedral angle over the mean height of its two triangles is the curvature
across it; edges that turn more than 40 degrees at once are creases, and
faces tighter than the part's smallest extent gather into rounds. Each
round is read across its width: curvature already at its full value at the
rim is a constant radius (an arc: graded `tangent`, the curvature jumps
where it starts), curvature that grows in from the rim is `curvature`, and
under four facets across the mesh cannot tell (`unresolved`). Tests hold
it to an arc and an eased round of the same corner at 8, 16 and 32 facets.

What it cannot see on a mesh: a tangent-only join between two broad
surfaces (no round between them), since both sides are below the
threshold; that needs the STEP.

## Sketch matching

```python
from loadpath import sketchmatch as SM

m = SM.match_view(mesh, "back", "page.jpg", crop=(x0, y0, x1, y1), seed=(0.5, 0.8))
m["mean"], m["p95"], m["iou"], m["segments"]   # mm both ways; signed, round the outline
rail = SM.outline_curve(m)                      # the sketch's outline as a curve in model mm
SM.match_sheet([m, ...], "vs-sketch.png")
```

The sketch's outermost closed stroke is traced (`reference.sketch_strokes`
and `sketch_silhouette`: the page's light divided out, grid dots dropped,
gaps bridged). It is laid over the model's orthographic silhouette by a
similarity fit (scale and position, rotation optional) from several
starts, and the two outlines are compared point by point, both ways. `+`
is where the model stands outside the drawing (fat), `-` where it falls
inside (thin). A photographed page has no scale of its own, so the fitted
scale absorbs overall size and the numbers measure shape. Pass `scale` to
hold one view's scale for the others. `exclude` drops a neighbouring
view's strokes that run into this one.

The page is a photo of a pencil drawing, so a few millimetres is the
drawing's own wobble. The numbers say where the model departs from the
drawing by more than that.

## Example: a handheld scanner's shell

A handheld 3D scanner had a shell built from blended fields. Its face (two camera standoffs and
an optics island, lofted up from the body) and its grip were rebuilt with
`fair_patch` and `loft`, and the head-grip blend now runs between exact
fields. Same pitch, same checks, designed grooves left out:

| | fields | surfaced |
|---|---|---|
| Wobble area (lumps, dips, ridges) | 16,100 mm² | 5,600 mm² |
| Curvature off its smoothed self, RMS | 43 /m | 18 /m |
| Spots tighter than R 2.5 | 529 mm² | 45 mm² |

## Example: a pod where every join carries curvature

`examples/freeform/pod.py` lofts a body from six sections, closes both
ends with caps, fillets a mount boss standing through the top, and
blends a recessed flat seat down from a second opening. Eight faces, a
closed solid, 17 s. Built tangent-only and then curvature-continuous:

| Largest over every shared edge | G1 | G2 |
|---|---|---|
| Gap | 0.0004 mm | 0.0004 mm |
| Angle between tangent planes | 0.04° | 0.04° |
| Jump in curvature across the edge | 0.069 /mm | 0.0002 /mm |

Known limits: a cap's boundary has to project onto its best plane as a
simple loop; `cut` does not cross a periodic surface's seam. A `blend`
between unlike loops (a round post over an oval opening) is wider where
the gap is wider, so its curvature varies round the loop; where an even
join is wanted, `fillet` cuts both faces back by the same distance.
