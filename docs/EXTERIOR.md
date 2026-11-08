# Any exterior in

You or your agent draw the shape; loadpath makes everything inside it work.

The exterior is the one thing loadpath never designs. A designer surfaces it
in Rhino, Plasticity or Fusion; an agent roughs it out in build123d or an
SDF library; a generator such as Tripo or Meshy makes it from a render; a
phone scans it. Each source is wrong for a solver in its own way, so every
exterior goes through one intake first:

```bash
loadpath exterior generated.glb --size 78.6 --skin 1.2 --out exterior/
```

```python
from loadpath.exterior import load_exterior
ext = load_exterior("generated.glb", size=78.6, skin=1.2)   # or a STEP, an STL, a function, a build123d part
ext.save("exterior/")       # exterior/exterior.stl + exterior/exterior.json
```

`loadpath handoff` takes the same sources wherever a shape goes (the
exterior, the designer's fixed geometry, each component): a path, a shape
dict, or `{"path": ..., "size": ..., "up": ...}` for the intake's options.
Its report carries each intake's findings under `intake`.

## What it reads

| source | how |
|---|---|
| STEP (`.step`, `.stp`) | tessellated and sewn by OpenCascade, open shells closed (`loadpath[step]`) |
| a clean mesh (STL, OBJ, PLY, 3MF, GLB/glTF) | used as is: one closed, consistently wound body |
| a broken mesh: AI-generated, scanned, exported badly | wrapped (below) |
| an SDF function `f(P) -> d` over (N, 3) points in mm, negative inside | marching cubes; its box found by sampling when not given. Numpy expressions, the `sdf` package's objects and libfive-style `f(x, y, z)` all work |
| a build123d or CadQuery object | tessellated through OpenCascade |
| `file.py[:name]` | the module's `name` (default `exterior`): a function of points is an SDF, a function of nothing is called and its result read |
| an SDF grid (`.npz`) | the format below |

## The wrap

A generated mesh comes with most of these: holes; a second, inner copy of
the surface (a double wall); patches that cross the surface; patches wound
the wrong way; loose debris; unit scale instead of millimetres; Y up
(glTF's convention). The wrap decides inside and outside on a grid and puts
the new surface back on the source's own triangles:

- **inside**: by the generalized winding number (libigl,
  `pip install "loadpath[exterior]"`), after each connected sheet is wound
  consistently and turned to face out. It reads holes, self-intersections,
  double walls and flipped patches right. Pockets sealed inside the solid
  are filled: an exterior is solid, and its interior is loadpath's to design.
  Without libigl, a flood fill from outside, which seals holes up to `seal`
  mm (4 grid cells by default) and is good to about a pitch;
- **the surface**: within 3 cells of it, every grid node takes its exact
  distance to the source's triangles, so the zero level lands on them:
  detail down to about a grid cell stays where the generator put it. The
  default grid is the longest extent / 320 (0.25 mm on an 80 mm part);
- **bodies**: the largest is kept; the rest (debris) are listed with their
  volumes and dropped (`keep="all"` keeps them);
- **scale and frame**: `size` (the real longest extent) or `scale`; `up`
  (default +y for GLB/glTF); `place` centres it on z = 0 when it was moved,
  and keeps the source's frame for a designer's CAD so the component CAD
  still lines up. The report's `transform` maps the source's frame to the
  result's (`Exterior.apply(mesh)` moves a part drawn in the source frame).

The report says how far the surface moved (`wrap.onSource`), where new
surface had to be made up over holes (`wrap.patched`: area and where), how
much of the source was left behind inside (inner shells) or outside
(debris, fins thinner than the grid), and, given `skin`, the room inside:
the cavity once the skin is in, the largest ball that fits, and the regions
too thin for anything but skin (`capacity.thinRegions`, each with its area
and box).

### Where generated meshes break down

- **Holes are bridged, not designed.** A hole is closed by the shortest
  surface the winding number gives, flat across it. Small holes do not
  matter; a large one (a missing back panel) gets a surface nobody drew.
  `wrap.patched` says where; that part of the exterior is the designer's.
- **Folded inner sheets that pierce the skin.** When a generator's inner
  wall folds over itself and pokes through the outer one, the winding
  number cancels between them and that region reads as outside: a groove
  where the shape is solid. A double wall that stays inside is fine.
- **Detail under a grid cell.** Engraving, fine knurling and thin fins below
  the pitch go. Lower `pitch` (memory grows with its cube).
- **Scale is a guess without `size`.** Generated meshes have no units. A
  shape under 5 across is flagged as unit-scale or metres.
- **Thin is thin.** A lanyard loop or a raised rim narrower than twice the
  skin is all skin: solid, and nothing goes inside it. The report lists
  these regions; they are often where the generator invented detail.
- **What the generator could not know.** Nothing in a generated mesh says
  where the button is, which side opens, or what the wall must carry. That
  comes from the description; see the
  [tutorial](tutorials/generated-exterior/).

## SDF grids out and back

```bash
loadpath sdf part.stl --pitch 0.25          # part.sdf.npz
```

Any part (a delivered interior, an exterior, an SDF function) as a signed
distance grid for SDF tools: `distance` (float32, nx x ny x nz, mm,
negative inside), `origin` (the node [0, 0, 0], mm), `pitch` (mm), `units`
("mm"), trilinear between nodes. A study takes one back as a shape,
`{"type": "sdf", "path": "part.sdf.npz"}`, which is how an SDF tool's shape
comes in without code in the study (`exterior.save_sdf(f, path, bounds=...)`
samples a function).

```bash
loadpath sdf part.stl --pitch 0.25 --out part.vdb   # an OpenVDB level set
```

A `.vdb` path writes an OpenVDB narrow-band level set (3 voxels each side,
OpenVDB's default) that Houdini, Blender and nTop open directly, and a
`.vdb` level set from any of them comes back as an sdf shape or an
exterior the same way. The field graph's `grid` op samples any float grid
in a `.vdb` (`"quantity"` names the grid), so a density or temperature
volume made elsewhere can drive a lattice or a remap. `loadpath.vdb` reads
and writes the files in pure Python: zip, blosc (with the `blosc` package),
active-mask and half-float files read, and files it writes open in the
OpenVDB library to float precision.
