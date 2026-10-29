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

## What GRIP recorded vs loadpath main

GRIP's v0.3 numbers are the study's own check at delivery. "Re-verified" is
the same delivered STL checked by main at 8de87b2 (150k cells, with the
hotspot submodels the verifier now runs). "Regrown" is the study run again
on main.

| Link | GRIP recorded | Re-verified on main | Regrown on main |
| --- | --- | --- | --- |
| Turret | 236 g, SF 2.41 | SF 1.57, crease 512 mm | pending |
| Upper arm | 179 g, SF 2.81 | SF 2.09, 1067 mm² sawn off | pending |
| Forearm | 78 g, SF 4.19 | SF 2.14, crease 278 mm | pending |
| Wrist yoke | 24 g, SF 4.41 | SF 2.49, crease 206 mm | 23.8 g, SF 4.22, crease 26 mm, p5 2.97 mm |
| Wrist link | 67 g, SF 4.13 | SF 4.12, p5 member 1.07 mm, 87 mm² sawn off | pending |
