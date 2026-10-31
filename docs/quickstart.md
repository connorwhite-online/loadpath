# Quickstart

Ten minutes from install to a printable, verified part. Units everywhere are
millimetres, newtons, megapascals.

## 1. Install

```bash
pip install "loadpath[mcp] @ git+https://github.com/connorwhite-online/loadpath"
loadpath --help
```

(`[mcp]` adds the MCP server and the renderer; plain `loadpath` is the solver
alone. It is not on PyPI yet.)

## 2. Start from an example

```bash
loadpath examples                      # what ships with the package
loadpath examples shelf-bracket > bracket.json
```

`shelf-bracket` is a PETG wall bracket: two M5 screws into the wall, a pad the
shelf sits on 100 mm out, 10 kg of shelf with a factor of 3. Open it: every
field is described in [`loadpath/README.md`](../loadpath/README.md#the-study),
and `loadpath schema study` prints the JSON Schema (also at
[`schema/study.schema.json`](../schema/study.schema.json), for editors).

## 3. Check it

```bash
loadpath check bracket.json
```

Check validates the whole study against the schema and reports every problem
at once, by path, with a suggestion where it can make one:

```json
{"ok": false,
 "issues": [
   {"level": "error", "path": "$.keepout", "message": "unknown key; did you mean 'keepOut'?"},
   {"level": "error", "path": "$.supports[0].fix", "message": "'pinned' is not valid: 'all', 'pin', or the axes held, like 'xz'"}
 ]}
```

When it passes it shows the voxel problem: cells, pitch, the regions loads and
supports can name (including the ones `fasteners` add, like `upper-bore`), and
warnings. Fix every warning you can before you run.

## 4. Run it

```bash
loadpath run bracket.json --out out/
```

A few minutes. Progress streams to stderr as JSON lines. `out/` gets one STL
and one render per outcome, `report.json`, and `viewer.html`: open it in a
browser to orbit the part, colour it by stress or safety factor, animate the
deformation, cut it open and replay the optimization (`--viewer all` for every
outcome, `--viewer none` to skip it).

## 5. Read the report before trusting the part

For each outcome: `mass`, `safetyFactor` and `deflection` per load case,
`passes` against the goal, `bodies` (1, or something is not connected),
`keepOutIntrusion` and `preserveMissing` (mm³, should be ~0), `minMember` and
`p5Member` (the thinnest members). Look at the render. If it fails the goal,
the loads, the supports or the design space are the usual suspects, in that
order.

## Beyond plain parts

**Lattice infill.** Add `"lattice": {"type": "gyroid", "cell": 10, "skin": 1.2}`
to a study (see `loadpath examples pedal-lattice`) and the best outcome is also
filled with a TPMS sheet lattice whose walls follow the stress: solid where
the part works hard, lattice where it idles. Or lattice any finished part:

```bash
loadpath lattice bracket.json out/shelf-bracket-25.stl --out out/
```

Use `"skin": 0` for SLS, MJF and resin, so powder or resin can drain.

**Heat exchangers.** Two fluids through one gyroid (or diamond, Schwarz P)
sheet, in any core shape, with self-supporting headers and ports:

```bash
loadpath examples heat-exchanger-counterflow > cooler.json
loadpath heatx cooler.json --size          # pick cell and wall for the duty, seconds
loadpath heatx cooler.json --out hx/       # design, check isolation, rate, render
```

`loadpath schema heatExchanger` lists every field.

## Tutorials

[tutorials/](tutorials/) has six workflows end to end (FDM bracket, graded SLS
lattice with drains, heat exchanger with simulation, impact bumper, field
editor, arm link from a URDF), each with the numbers you should get.

## For agents

See [agents.md](agents.md): the MCP server, its tools, and the loop an agent
should follow.
