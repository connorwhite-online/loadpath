# loadpath

Engineering checks and generative design for parts you are going to print,
built for people and coding agents that make CAD.

CAD generators (build123d, CadQuery, text-to-CAD tools) make geometry. They
don't tell you whether a part is strong enough, thick enough or will fit. loadpath does:

- **Optimize.** Describe what a part must keep, what it must avoid, where it is held
  and how hard it is pushed. loadpath returns the lightest printable shape that
  carries the loads, verified by a fresh FEA of the delivered mesh.
- **Check.** `loadpath.checks` has printability (wall thickness, overhangs, trapped
  voids), component fit, fluid-network isolation, a quick FEA probe and an SDF toolkit.
- **Model with fields.** A part can be a graph of implicit fields, nTop style: the
  skin thickens where the stress is, the lattice cell grows away from the bolts, the
  fillet is larger where the load comes in, and the mesh comes out smooth
  ([docs/fields.md](docs/fields.md)).
- **For the process you print with.** FDM (bridges, islands, bed, layer bond), powder
  nylon (SLS, MJF: trapped powder, escape holes, warp-prone panels), resin (SLA, DLP:
  supports, peel area, suction cups, trapped resin) and powder-bed metal (internal
  ceilings, powder clean-outs), each with material cards.
- **Sliced, not guessed (FDM).** With PrusaSlicer installed, every outcome is sliced
  headless for print time, filament and support; `"goal": {"minimize": "printTime"}`
  delivers the fastest passing part, and `"slicer": {"grade": true}` sizes infill
  region by region from the stress and writes it as slicer modifiers in a graded 3MF
  ([docs/slicer.md](docs/slicer.md)).

### Why agents like it

- **One JSON study in, one JSON report out.** Keep-outs, preserved faces, loads and
  goals are plain shapes and numbers an agent can write from a CAD script; the report
  says pass or fail and why, in units (`safetyFactor`, `p5Member`, `bodies`, ...).
- **MCP or CLI.** `loadpath mcp` is a stdio MCP server; `loadpath run study.json`
  does the same from a shell. No GUI, no GPU, no license server.
- **Renders it can look at.** Every outcome comes with clay renders, and stress maps
  are one call away, so a vision model can sanity-check the shape before the user does.
- **Every mistake at once.** `loadpath check` validates against a published JSON
  Schema and returns every problem by path, with "did you mean" for a misspelt key,
  so an agent fixes a study in one round trip.
- **A skill that knows the pitfalls.** [`loadpath/SKILL.md`](loadpath/SKILL.md) tells
  the agent how to write a study and what to check in the report before trusting it.

Units are millimetres, newtons, megapascals. See [`loadpath/README.md`](loadpath/README.md)
for the method, the study format, goals and limits.

<!-- gallery:start -->

## Examples

The eval cases (11/11 passing): GRIP robot-arm style parts, each optimized and then checked by a fresh FEA of the delivered mesh. Blue arrows are loads, red is keep-out. Every view, stress map and number is in [docs/GALLERY.md](docs/GALLERY.md).

<table>
<tr>
<td align="center" valign="top"><img src="docs/images/cases/arm-link.png" width="250"><br><b>arm-link</b><br><sub>58.9 g · 53% lighter · SF 13.19 · pass</sub></td>
<td align="center" valign="top"><img src="docs/images/cases/arm-link-driven.png" width="250"><br><b>arm-link-driven</b><br><sub>46.2 g · 63% lighter · SF 7.98 · pass</sub></td>
<td align="center" valign="top"><img src="docs/images/cases/bell-crank.png" width="250"><br><b>bell-crank</b><br><sub>25.2 g · 83% lighter · SF 3.44 · pass</sub></td>
</tr>
<tr>
<td align="center" valign="top"><img src="docs/images/cases/grip-turret.png" width="250"><br><b>grip-turret</b><br><sub>172.8 g · 90% lighter · SF 2.33 · pass</sub></td>
<td align="center" valign="top"><img src="docs/images/cases/grip-upper-arm.png" width="250"><br><b>grip-upper-arm</b><br><sub>207.9 g · 88% lighter · SF 7.18 · pass</sub></td>
<td align="center" valign="top"><img src="docs/images/cases/grip-wrist-yoke.png" width="250"><br><b>grip-wrist-yoke</b><br><sub>17.7 g · 92% lighter · SF 3.10 · pass</sub></td>
</tr>
<tr>
<td align="center" valign="top"><img src="docs/images/cases/mbb-beam.png" width="250"><br><b>mbb-beam</b><br><sub>26.4 g · 60% lighter · SF 1.56 · pass</sub></td>
<td align="center" valign="top"><img src="docs/images/cases/motor-mount.png" width="250"><br><b>motor-mount</b><br><sub>27.5 g · 69% lighter · SF 2.73 · pass</sub></td>
<td align="center" valign="top"><img src="docs/images/cases/printed-hook.png" width="250"><br><b>printed-hook</b><br><sub>31.3 g · 79% lighter · SF 2.95 · pass</sub></td>
</tr>
<tr>
<td align="center" valign="top"><img src="docs/images/cases/wall-bracket.png" width="250"><br><b>wall-bracket</b><br><sub>61.3 g · 85% lighter · SF 4.37 · pass</sub></td>
<td align="center" valign="top"><img src="docs/images/cases/wall-bracket-fasteners.png" width="250"><br><b>wall-bracket-fasteners</b><br><sub>47.3 g · 89% lighter · SF 3.59 · pass</sub></td>
</tr>
</table>

Where the load goes: von Mises stress on the part bent by its worst load case:

<table>
<tr>
<td align="center" valign="top"><img src="docs/images/stress/bell-crank.png" width="250"><br><sub>bell-crank</sub></td>
<td align="center" valign="top"><img src="docs/images/stress/grip-upper-arm.png" width="250"><br><sub>grip-upper-arm</sub></td>
<td align="center" valign="top"><img src="docs/images/stress/printed-hook.png" width="250"><br><sub>printed-hook</sub></td>
</tr>
</table>

Two-fluid heat exchangers: two isolated channel networks through one sheet gyroid, with headers that print without support.

![box-crossflow](docs/images/heatx/box-crossflow.png)

Printed in metal, nothing inside can be supported: the channels end in peaked roofs instead of flat ceilings (red).

![metal ceilings](docs/images/heatx/lpbf-ceilings.png)

Made another way: the same arm link printed, cast (no undercuts along z), 3-axis milled from both sides with a 6 mm cutter, and with members capped at 6 mm. Each meets SF 3 and 0.5 mm.

![manufacturing constraints](docs/images/manufacturing/arm-link.png)

<sub>Rendered 2026-10-29 from `f5b3a9d` by `python docs/gallery.py`.</sub>

<!-- gallery:end -->

### Field-driven parts

A part can also be a graph of fields ([docs/fields.md](docs/fields.md)). Here the
arm link's skin and gyroid wall follow its own stress (colour) and the cells grow
away from the bearings. At the same 52 g as a uniform lattice it has SF 14.8 and
0.65 mm deflection, against SF 11.8 and 0.74 mm (`python examples/fields/arm_link.py`).

![graded vs uniform arm link](docs/images/fields/arm-link-graded-vs-uniform.png)

A graded cell bends the lattice's coordinates, so walls run continuously from 4 mm
to 12 mm cells:

![graded gyroid](docs/images/fields/graded-cell.png)

A skin over a lattice traps powder or resin. Give the graph a `"process"` and it
drills its own drain holes, sized well over the process minimum (6 mm for SLS) and
added until no powder is more than 60 mm from one, on bosses, at the calmest flat
faces clear of the bearing seats. Then it checks that nothing is still sealed and
the escape ball reaches every chamber. The arm link takes 14 holes; SF 13.5
against 13.4 undrilled, 10% softer.

![drain holes in the arm link](docs/images/fields/arm-link-drains-top.png)
![drain holes in the arm link, underside](docs/images/fields/arm-link-drains.png)

## Install

```
pip install "loadpath[mcp] @ git+https://github.com/connorwhite-online/loadpath"
loadpath examples shelf-bracket > bracket.json
loadpath check bracket.json            # every problem at once, with suggestions
loadpath run bracket.json --out out/   # optimize, mesh, verify, render
loadpath run bracket.json --recall     # re-run: start from what earlier runs found
loadpath mcp                           # stdio MCP server for agents
```

Also: `loadpath lattice` (fill a part with a stress-graded TPMS lattice),
`loadpath heatx` (two-fluid TPMS heat exchangers), `loadpath field` (a part as a
graph of fields, [docs/fields.md](docs/fields.md)), `loadpath schema` (JSON
Schemas, also in [`schema/`](schema/)).

Large studies (100k+ cells) solve with matrix-free multigrid; install the `gpu`
extra (`loadpath[mcp,gpu]`, PyTorch) to run it on an NVIDIA GPU, an Apple
Silicon GPU, or torch's threaded CPU kernels. Without it everything runs on
numpy as before.

- [Quickstart](docs/quickstart.md): install to a verified part in ten minutes.
- [Agents](docs/agents.md): the MCP server, its tools and the loop to follow.
  [`loadpath/SKILL.md`](loadpath/SKILL.md) is the skill.
- [Reference](loadpath/README.md): the method, every study field, the report, limits.
- [Benchmarks](docs/BENCHMARKS.md): loadpath against top88 and top3d on the textbook
  problems (MBB beam, L-bracket, 3D cantilever, 3D bridge) and the GE bracket challenge.
- [Examples](loadpath/examples/): a fastened bracket, a robot arm link, a
  lattice-filled pedal, two heat exchangers.

## Tests

```
python3 tests/test_loadpath.py
python3 tests/run_tests.py
python3 tests/test_interface.py
python3 tests/test_pressure.py
python3 tests/test_process.py
python3 tests/test_fields.py
python3 tests/test_recall.py
python3 tests/test_evals.py
python3 -m loadpath.evals.run --tier coarse --corpus 6 --out evals-coarse/   # minutes: every push
python3 -m loadpath.evals.run --out evals-out/                               # full suite
python3 tests/test_benchmarks.py
python3 -m loadpath.benchmarks.run --out bench-out/
python3 docs/gallery.py --evals evals-out/    # refresh the gallery renders from that run
```

## License

MIT
