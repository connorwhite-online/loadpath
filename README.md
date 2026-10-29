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
python3 -m loadpath.evals.run --out evals-out/
python3 docs/gallery.py --evals evals-out/    # refresh the gallery renders from that run
```

## License

MIT
