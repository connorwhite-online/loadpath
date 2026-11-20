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
cap, err = S.fill([(edge, neighbour_face, "G1")]) # N-sided patch, tangent to its neighbours
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
- **Fills.** OpenCascade's `BRepFill_Filling` holds a boundary G0 or G1.
  It cannot hold a bounded edge G2, and `fill` says so rather than quietly
  building G1.
- **Into fields.** `sdf` turns a closed B-rep into a signed distance that
  is exact within a band of the skin. A field model (cuts, shells, halves,
  ducts) then runs unchanged on a surfaced skin, and blends between exact
  fields come out even.

## Quality

```python
from loadpath import surfacequality as Q

rows, worst = Q.edge_continuity(solid)    # G0 gap, G1 angle, G2 jump on every shared edge
k = Q.surface_curvature(face)             # principal curvatures over a face
r = Q.fairness(mesh, min_radius=2.5, ignore=grooves)
r["wobble"], r["tight"], r["score"]       # flaws, clustered and placed
Q.quality_sheet([before, after], "q.png", cameras)   # zebra + curvature maps
```

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
| Wobble area (lumps, dips, ridges) | 16,100 mm² | 5,400 mm² |
| Curvature off its smoothed self, RMS | 43 /m | 15 /m |
| Spots tighter than R 2.5 | 529 mm² | 14 mm² |
