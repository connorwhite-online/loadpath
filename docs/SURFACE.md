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
   doublet) are repaired.
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
5. **Patches.** Where a face's corners all have four edges the limit
   surface is exactly bicubic: those patches *are* the surface and meet with
   C2 continuity. A face at a star point (3 or 5+ edges) is split once
   toward it; the small corner piece is a quintic fitted to the exact limit
   points and the exact limit tangent planes (Halstead masks), so the
   patches round a star point share edges exactly and meet G1 to a fraction
   of a degree (reported).
6. **Solid.** Sewn, checked, written; for a part with exact features the
   smooth body is cut by the exact cylinders, boxes and spheres in
   OpenCascade.

## Numbers (2026-11-17)

| input | cage | STEP faces | tolerance | deviation p99 / max | worst kink | valid, re-read |
|---|---|---|---|---|---|---|
| Tripo-style pebble (holes, double wall, jitter) | 4954 quads | 426 (102 packed B-splines) | 0.1 mm (FDM) | 0.066 / 0.34 mm | 0.29 deg | yes |
| soft plate + exact bore (field part) | 1879 quads | 146 B-spline + 1 cylinder | 0.1 mm | 0.087 / 0.25 mm | 0.7 deg | yes |

The pebble's surface lands as close to its true shape (the generator's
SDF, before it was broken) as the noisy input does: p99 0.11 mm vs 0.10.

**Packing.** Rectangles of regular cage faces are grown greedily and each
becomes one clamped bicubic B-spline (exact: inside such a rectangle the
limit surface is one C2 spline; a rectangle never closes round a tube).
Faces at a star point stay whole as degree-7 patches; where that misses the
limit by more than a third of the tolerance they are split once toward the
star point. The file's points are written unrounded: rounding a degree-7
patch's poles to 10 nm moved it off its own edges and readers could not
mesh it.

## Not yet

- Packing is greedy: rectangles end at T-junctions, so some faces are long
  thin strips; tracing separatrices from the star points would give a
  cleaner layout.
- Zebra stripes still wobble on large flats of field parts: the fit follows
  the marching-cubes mesh's small ripples within the tolerance.
- Adaptive density: Instant Meshes lays the cage at one density, so flats
  get as many patches as tight blends.
- Sharp creases on an organic body (a generated mesh's hard rim) are
  rounded to within the tolerance, not kept as creases.
