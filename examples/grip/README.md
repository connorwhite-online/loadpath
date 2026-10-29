# GRIP v0.3 links

The five structural links of [GRIP](https://github.com/connorwhite-online/grip),
a 6-axis printed robot arm, as loadpath studies. GRIP is loadpath's standing
test subject: real interfaces (servo cradles and their screw bosses, horn hubs,
cycloidal-drive sockets with hex-driver access to every bolt, a J2
counterbalance spring) and real loads (statics over the arm's URDF at full
reach with 1 kg at the gripper and a 1.5x dynamic factor, each joint above at
its rated torque, a 20 N push along its axis). Goal: SF 2 in SLS PA12-GF.

`studies/` is copied verbatim from GRIP's `generative/v3/studies/` (branch
`generative-redesign`, commit 3689f8d, written by `generative/v3.py`). GRIP is
CERN-OHL-W-2.0; these study files describe its parts.

```bash
python examples/grip/run.py                         # regrow all five, re-verify, close-ups
python examples/grip/run.py --links wrist_yoke      # ~25 min on 4 cores
python examples/grip/run.py --verify-only DIR       # check DIR/<link>.stl
```

Each link is re-verified on a 150k-cell grid and drawn close up at every
interface (`render.render_interfaces`); `scorecard.json` has mass, SF with the
peak's location per case, ray member thickness, crease length, sawn-off area,
keep-out intrusion and lost preserved geometry.

## What GRIP recorded vs loadpath main (2026-10-29)

"Recorded" is GRIP's v0.3 summary (the study's own check at delivery).
"Regrown" is the study run again on main at 8de87b2 and checked the way
the study's confirm stage checks it: 150k cells, with stress within one
optimization cell of a load or support left out of the peak. The turret
was regrown on main with the closing fillet as the grip-turret eval case.

| Link | Recorded | Regrown on main | Ray p5 member | Crease | Notes |
| --- | --- | --- | --- | --- | --- |
| Turret | 236 g, SF 2.41 | 257 g, SF 2.25 | | 139 mm | eval case grip-turret |
| Upper arm | 179 g, SF 2.81 | pending | | | GRIP's part has ~1070 mm² sawn flat on the design box |
| Forearm | 78 g, SF 4.19 | **57 g**, SF 2.54 | 3.24 mm | 3 mm | |
| Wrist yoke | 24 g, SF 4.41 | 24 g, SF 4.35 | 2.97 mm | 26 mm | overbuilt; eval case grip-wrist-yoke asks for 20 g |
| Wrist link | 67 g, SF 4.13 | **52 g**, SF 2.96 | 2.24 mm | 0 mm | thinnest members under the 3 mm minimum |

### Known problem in the studies: no room for the servo horn

GRIP's `servo_child` (v3.py) keeps the horn bolts and their drivers clear,
but not the horn itself (0–4 mm above the servo's horn face) or the servo
body the child rotates round. Every wrist yoke so far, GRIP's printed one
included, is solid across the J4 horn disc from about 1.5 mm above the horn
face (~900 mm³); the regrown yoke also cuts ~900 mm³ into the J4 servo's
roll sweep. The studies here are left as GRIP wrote them; the eval case
`grip-wrist-yoke` adds `j4-horn` (r 12, 0.5 mm below to 4 mm above the
face) and `j4-servo-sweep` (r 37, the servo's depth below the face). The
fix belongs in GRIP's `servo_child`.
