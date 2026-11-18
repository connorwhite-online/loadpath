# A first exterior round the parts (`envelope.py`)

Everything else in loadpath starts from an exterior somebody drew. Before
there is one, a designer still wants the smallest, best-handling form the
parts allow: which surfaces the stack fixes, where the hand goes, how much
room a style has. `envelope.py` builds that starting form as a signed
distance field and checks it the way the designer's own surfacing will be
checked later. It does not replace the surfacing: the form goes to
`exterior.load_exterior` and `loadpath enclosure` like any SDF, and a
designer's model comes back through the same checks.

## Inputs

- **A style field**, drawn with the primitives here or any SDF:
  - `sweep(P, spine, half_w, half_d, n)`: a superellipse section along a
    Catmull-Rom spine, with sizes per spine point. Grips, handles, booms.
  - `funnel(P, floor_center, axis, floor_half, half_angles, depth)`: a
    recess whose walls slope at a field of view's angles. Cut it from a body
    with `smax(body, -funnel)`, and a lens or laser sits at the bottom of a
    scoop that never vignettes it.
  - `sdf_kit.smin` / `smax` blend and trim.
- **Keep-outs**: `Keepout(name, mesh_or_shape_or_callable, gap)`, each part
  and how far the outside must stay from it (the skin plus air). A mesh's
  distance comes from a voxel grid padded by `max(12, 3 gap)` mm, so the
  field is right where blends look and a lower bound beyond.
- **The hand**: `data/hands.json` (hand length and breadth by percentile,
  Pheasant & Haslegrave, *Bodyspace*; grip rules of thumb, marked `verify`).

## Calls

| call | what it gives |
| --- | --- |
| `wrap(style, keepouts, k)` | the style blended (smooth union) with every keep-out: the form only grows where a part needs room, never cuts one |
| `pushes(style, wrapped, bounds)` | where the parts pushed the style out, as boxes with how far (mm) and how much (cm³): the surfaces the stack fixes, for the designer |
| `clearances(f, keepouts)` | for each mesh keep-out: the smallest depth of its surface inside the form against the gap asked |
| `view_check(f, apex, axis, half_angles)` | rays through a lens's or laser's field of view: blocked if a ray never leaves the form or comes back into it after leaving |
| `grip_targets(hand_length)` | working grip diameter, circumference and trigger reach for a hand |
| `grip_report(mesh, origin, axis, heights, trigger_face, backstrap_dir)` | the grip's sections (perimeter, equivalent diameter, extent) and the trigger reach, next to the targets for each percentile |

## Worked use

A handheld stereo scanner head (two cameras on a 150 mm baseline, three
lasers, a compute module and two 21700 cells in a pistol grip): the style
is a lozenge head, a swept grip raked 16°, and one visor of view funnels;
twenty parts are keep-outs. The wrap is meshed with
`exterior.mesh_from_sdf`, split into a printed-metal rear cap (the
heatsink) and a nylon shell, and handed to `loadpath enclosure` with the
parts `place: "given"` for mounts, drops and squeeze. The report says:

- where the parts pushed the style (the compute stack set the rear, the
  bench ends set the corners);
- that every field of view is clear through the visor;
- that the grip's 150 mm circumference is set by the two cells (a
  50th-percentile hand wants about 115-125 mm). A smaller-cell variant is
  the only way down.

## Limits

- Distances away from the surface are first-order estimates. That is
  enough for blends and offsets near the surface, not for a far offset.
- The hand data are anthropometric percentiles and rules of thumb, not a
  grip-comfort model. Confirm with foam mock-ups.
- `view_check` treats the window glass as open (it is not part of `f`).
  Glass reflections and crosstalk between emitters and cameras behind one
  window need baffles, which this does not model.
