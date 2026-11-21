# Organic surfacing: any mesh in, editable smooth STEP out

A topology-optimized body, a scan or an AI-generated mesh (Tripo, Meshy) is
hundreds of thousands of triangles and no surface a CAD user can pick,
fillet, offset or shell. Topology-optimization codes stop at that mesh;
`loadpath surface` turns it into what a surfacing artist would have drawn:

    pip install "loadpath[surface]"
    loadpath surface generated.glb --size 72 --out pebble.step
    # pebble.step       one closed solid of B-spline patches
    # pebble.cage.obj   the Catmull-Clark control cage (the editable design)
    # pebble.json       tolerance and why, deviation both ways, smoothness

    # a loadpath part: the body smooth, bores and mounts still exact cylinders
    export_step(part, "part.step", surface="subd")

## How

1. **Intake.** Open, self-intersecting, double-walled meshes are wrapped by
   `exterior.load_exterior` (generalized winding number) onto the source's
   own triangles.
2. **Quad cage.** Instant Meshes (Jakob et al. 2015, BSD, `pyinstantmeshes`)
   lays an all-quad mesh whose edges follow the principal curvature
   directions. Its occasional leftovers (a small hole, a pinched vertex, a
   doublet) are repaired: holes are traced face by face round each corner,
   so one that passes a vertex twice is followed right, and an odd-sided
   hole (no quad tiling has an odd boundary) is closed after splitting every
   quad in four. Slivers in the input (exact bores and mount faces cut by
   manifold booleans are long triangles beside 0.07 mm ones) leave Instant
   Meshes nothing to extract at any density; the cage is then laid over an
   even resampling of the same shape and fitted to the original. The first
   density comes from the part: faces no wider than 1.2 x its thin walls and
   members (10th percentile of the distance straight through it).
3. **Fit.** The cage's control points move until its Catmull-Clark limit
   surface lies on the mesh: least squares between exact limit points and
   their closest points on the mesh, both ways, with correspondences across
   a thin wall rejected by their normals and the cage stiffened wherever the
   surface would fold.
4. **Tolerance from physics.** The cage is laid again finer until the
   two-way deviation (99th percentile) is within the tolerance, then faired:
   the bending weight rises until that budget is spent, so noise goes and
   design stays. The tolerance is a third of the process accuracy (FDM
   +-0.3 mm gives 0.1 mm: it adds 5% to the printed error, sqrt(1 + 1/9)),
   or twice the input's own measured noise when that is larger (fitting
   closer would fit the noise).

   **The worst point, not only the 99th percentile.** Every place where the
   surface misses the mesh by more than the process accuracy (three times
   the tolerance) is found and sorted. A place where the surface spans air
   (a ray along its normal goes on through air both ways) is a through-hole
   the cage closed: Instant Meshes drops handles smaller than its faces,
   and which ones it drops changes from run to run. Its diameter is twice
   the surface's miss there (the membrane sits a radius from the wall). A
   hole the process could print open (FDM: 2 mm and up) makes the cage
   finer, to an edge under 0.7 of the diameter (four edges round it); a
   smaller one is reported in `worst.holesClosed`, since the printer would
   not leave it open either. Every other miss is a surface miss. One over
   twice the accuracy is a feature narrower than the faces, and the cage
   gets finer for it too; the rest pull harder (minimax reweighting, each
   point's weight times (miss / 0.75 aim)^6 a round, at most x17) until every place is within the accuracy
   less the patches' own third of the tolerance, so the fit spends its
   freedom only where the shape is tight, and the 99th percentile still
   holds. `worst.patchesMax` bounds the written faces (limit miss plus
   patch-to-limit). Euler numbers of mesh and cage are reported side by
   side, so a lost handle is never silent.
5. **Patches along the cage's own lines.** Every edge line leaving a star
   point (a vertex with 3 or 5+ edges) is drawn on until it meets another
   star point or a line already drawn (a motorcycle graph). Between the
   lines lie rectangles of cage faces with star points only at their
   corners, and each rectangle is one bicubic B-spline face. Every line is
   fitted once as a cubic spline (exact where the limit surface is
   bicubic; toward a star point its knots halve four times, to 1/16 of an
   edge, and it is fitted to the exact limit points and tangent planes),
   and both rectangles beside it take that same curve as their edge, so
   neighbours share edges exactly. Inside, the poles are fitted to the
   exact limit points and tangent planes (Halstead masks at the star
   points). Face count goes with the star points, not the cage: a few
   hundred faces where face-by-face patches gave thousands. Where the
   layout does not come out (a line that closes on itself, a fit outside a
   third of the tolerance, neighbours meeting at more than 1 degree) the
   faces are written one by one as before: regular ones exact bicubics
   packed into rectangles, star point ones degree-7 patches.
6. **Solid.** Sewn, checked, written; for a part with exact features the
   smooth body is cut by the exact cylinders, boxes and spheres in
   OpenCascade.

## Numbers (2026-11-21)

| input | cage | STEP faces | time | deviation p99 / max | worst kink | valid, volume |
|---|---|---|---|---|---|---|
| Tripo-style pebble (holes, double wall, jitter) | 5642 quads | 241 (was 525) | 2.5 min (was 12.5) | 0.071 / 0.27 mm, every place within 0.3 (was 0.095 / 0.35) | 0.005 deg (was 0.63) | yes, -0.004% (was -0.14%) |
| optimized arm link body (quick run, 2026-11-21) | 13255 quads (was 8383) | 957 (was 4173) | 10.4 min (was 20.5) | 0.083 / 0.27 mm away from two 1 mm pinholes (was 0.093 / 0.65) | 0.048 deg | yes, -0.01% (the old route left 4 free edges: not a solid) |

Tolerance 0.1 mm (a third of FDM's +-0.3 mm) on both, on the 99th
percentile; the worst point is held to the accuracy itself. On the arm
link the first rows of this table (6389 quads, 598 faces, 3.2 min) held
only the 99th percentile: the worst point was 1.37 mm on one run and 0.58
on the next, where the cage had closed a 2.7 mm through-hole at a member
join on one and not the other. Now the density search keeps going until
every hole the printer could make is open (a 13k cage) and the worst
point is pulled in; what is left closed is two ~1 mm pinholes FDM cannot
print open (`worst.holesClosed`, mesh Euler number -32, cage -28). The
cost is the uniform cage: Instant Meshes lays one density everywhere, so
the whole part pays for its tightest hole. Same machine (4
cores), same input, run one at a time. Earlier (2026-11-17): soft plate +
exact bore 146 B-spline faces + 1 cylinder, 250 s.

Where the time went: the fit's closest points (now a compiled kernel,
`closestjit.py`), the least squares (the sample rows are gathered onto the
limit points they pull, so the normal equations are M^T diag M and the
matrix with one row per sample is never formed; conjugate gradients from
the last answer, preconditioned by one factorization a round; a round stops
once no control point moves more than a fiftieth of the tolerance) and the
density search (the next cage's size from the error rate the last two
tries show, aiming just under the tolerance: the old step overshot to a
cage half again as fine as needed, with more star points).

**Face by face (the fallback).** Rectangles of regular cage faces are grown
greedily and each becomes one clamped bicubic B-spline; faces at a star
point stay whole as degree-7 patches, split once toward it where that
misses the limit by more than a third of the tolerance. The file's points
are written unrounded: rounding a degree-7 patch's poles to 10 nm moved it
off its own edges and readers could not mesh it.

## Not yet

- Adaptive density: Instant Meshes lays the cage at one density, so flats
  get as many cage faces (and star points, and so patches) as the
  tightest hole or blend. The worst-point bound now says exactly where
  finer faces are needed (`worst`); a cage refined only there (local
  3-refinement templates, or a remesher with a sizing field) is the lever
  on both time and face count.
- Zebra stripes still wobble on large flats of field parts: the fit follows
  the marching-cubes mesh's small ripples within the tolerance.
- Sharp creases on an organic body (a generated mesh's hard rim) are
  rounded to within the tolerance, not kept as creases.
- A coarse first cage can come out with holes round thin members; the
  density search then steps finer (2.5x) until it closes.
