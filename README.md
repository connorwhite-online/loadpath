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

The eval cases (12/12 passing): GRIP robot-arm style parts, each optimized and then checked by a fresh FEA of the delivered mesh. Blue arrows are loads, red is keep-out. Every view, stress map and number is in [docs/GALLERY.md](docs/GALLERY.md).

<table>
<tr>
<td align="center" valign="top"><img src="docs/images/cases/arm-link-driven.png" width="250"><br><b>arm-link-driven</b><br><sub>46.2 g · 63% lighter · SF 7.98 · pass</sub></td>
<td align="center" valign="top"><img src="docs/images/cases/arm-link.png" width="250"><br><b>arm-link</b><br><sub>58.9 g · 53% lighter · SF 14.78 · pass</sub></td>
<td align="center" valign="top"><img src="docs/images/cases/bell-crank.png" width="250"><br><b>bell-crank</b><br><sub>24.8 g · 84% lighter · SF 3.50 · pass</sub></td>
</tr>
<tr>
<td align="center" valign="top"><img src="docs/images/cases/grip-turret.png" width="250"><br><b>grip-turret</b><br><sub>148.8 g · 91% lighter · SF 2.54 · pass</sub></td>
<td align="center" valign="top"><img src="docs/images/cases/grip-upper-arm.png" width="250"><br><b>grip-upper-arm</b><br><sub>207.7 g · 88% lighter · SF 7.71 · pass</sub></td>
<td align="center" valign="top"><img src="docs/images/cases/grip-wrist-yoke.png" width="250"><br><b>grip-wrist-yoke</b><br><sub>25.3 g · 90% lighter · SF 3.10 · pass</sub></td>
</tr>
<tr>
<td align="center" valign="top"><img src="docs/images/cases/mbb-beam.png" width="250"><br><b>mbb-beam</b><br><sub>26.4 g · 60% lighter · SF 1.56 · pass</sub></td>
<td align="center" valign="top"><img src="docs/images/cases/motor-mount.png" width="250"><br><b>motor-mount</b><br><sub>27.5 g · 69% lighter · SF 3.35 · pass</sub></td>
<td align="center" valign="top"><img src="docs/images/cases/printed-hook.png" width="250"><br><b>printed-hook</b><br><sub>31.3 g · 79% lighter · SF 2.95 · pass</sub></td>
</tr>
<tr>
<td align="center" valign="top"><img src="docs/images/cases/wall-bracket-fasteners.png" width="250"><br><b>wall-bracket-fasteners</b><br><sub>47.3 g · 89% lighter · SF 3.60 · pass</sub></td>
<td align="center" valign="top"><img src="docs/images/cases/wall-bracket.png" width="250"><br><b>wall-bracket</b><br><sub>57.2 g · 86% lighter · SF 2.87 · pass</sub></td>
</tr>
</table>

Where the load goes: von Mises stress, blue (none) to red (the part's 98th percentile), on the part bent by its worst load case:

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

<sub>Rendered 2026-10-30 from `254be35` by `python docs/gallery.py`.</sub>

<!-- gallery:end -->

## Install

```
pip install "loadpath[mcp] @ git+https://github.com/connorwhite-online/loadpath"
loadpath examples shelf-bracket > bracket.json
loadpath check bracket.json            # every problem at once, with suggestions
loadpath run bracket.json --out out/   # optimize, mesh, verify, render
loadpath mcp                           # stdio MCP server for agents
```

Also: `loadpath lattice` (fill a part with a stress-graded TPMS lattice),
`loadpath heatx` (two-fluid TPMS heat exchangers), `loadpath schema` (JSON
Schemas, also in [`schema/`](schema/)).

- [Quickstart](docs/quickstart.md): install to a verified part in ten minutes.
- [Agents](docs/agents.md): the MCP server, its tools and the loop to follow.
  [`loadpath/SKILL.md`](loadpath/SKILL.md) is the skill.
- [Reference](loadpath/README.md): the method, every study field, the report, limits.
- [Examples](loadpath/examples/): a fastened bracket, a robot arm link, a
  lattice-filled pedal, two heat exchangers.

## Tests

```
python3 tests/test_loadpath.py
python3 tests/run_tests.py
python3 tests/test_interface.py
python3 tests/test_pressure.py
python3 -m loadpath.evals.run --out evals-out/
python3 docs/gallery.py --evals evals-out/    # refresh the gallery renders from that run
```

## License

MIT
