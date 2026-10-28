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

Units are millimetres, newtons, megapascals. See [`loadpath/README.md`](loadpath/README.md)
for the method, the study format, goals and limits.

## Install

```
pip install "loadpath[mcp]"        # CLI + local MCP server for agents
loadpath run study.json --out out/
loadpath mcp                       # stdio MCP server
```

Agents: [`loadpath/SKILL.md`](loadpath/SKILL.md) is the skill. It tells the agent how
to write a study, run it, and read the report before trusting it.

## Tests

```
python3 tests/test_loadpath.py
python3 tests/run_tests.py
python3 -m loadpath.evals.run --out evals-out/
```

## Print or host the result

When a part passes, [Materialize](https://www.materialize.cc) can print it in many
materials or host and share it. loadpath never calls out to it.

## License

MIT
