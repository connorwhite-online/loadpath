# loadpath for agents

loadpath is built to be driven by a coding agent: everything is JSON in and
JSON out, every input has a JSON Schema, validation reports every problem at
once, and long solves run as background jobs the agent polls.

## Connect

```bash
pip install "loadpath[mcp] @ git+https://github.com/connorwhite-online/loadpath"
```

```json
{"mcpServers": {"loadpath": {"command": "loadpath", "args": ["mcp"]}}}
```

(Claude Code: `claude mcp add loadpath -- loadpath mcp`.) The server runs on
the agent's machine; nothing leaves it. Jobs live in `~/.loadpath/jobs`
(`LOADPATH_HOME`) and survive a restart of the server.

Install the skill too: [`loadpath/SKILL.md`](../loadpath/SKILL.md) tells the
agent how to turn a part description into a study (the engineering, not the
syntax). `loadpath_guide` returns the same text.

## Tools

| Tool | What it does | Time |
| --- | --- | --- |
| `loadpath_guide` | The skill and the field reference. Read once. | instant |
| `loadpath_schema(kind)` | JSON Schema for `study`, `heatExchanger` or `lattice`. | instant |
| `loadpath_examples(name?)` | The example list, or one example document to adapt. | instant |
| `loadpath_check(study, base_dir?)` | Every schema problem at once (with "did you mean"), then the voxel problem: cells, pitch, region ids, warnings. | ~1 s |
| `loadpath_start(study, base_dir?, lattice?)` | Validate and run a study as a job. `lattice` also fills the best outcome with a lattice. | minutes |
| `loadpath_status(job_id)` | Progress while running; when done, every outcome (mass, safety factor and deflection per case, pass/fail, keep-out and preserve checks, an `attention` list), renders as images, and `viewer`: an interactive 3D page for the user. | instant |
| `loadpath_lattice(job_id, outcome?, options?)` | Fill a finished outcome with a stress-graded TPMS lattice, as a new job. | a minute or two |
| `loadpath_heat_exchanger(spec, base_dir?)` | Design, check and rate a two-fluid TPMS heat exchanger, as a job. | a minute or more |
| `loadpath_size_heat_exchanger(spec, max_pressure_drop?, min_wall?)` | Sweep cell and wall for a duty before designing. | seconds |
| `loadpath_check_printability(path, min_wall?, overhang_deg?, build?, pitch?, slicer?)` | Any STL/3MF/OBJ: wall thickness in mm with its ± (the voxel pitch) and the thinnest spots, overhang area for a build direction, trapped voids. `slicer` adds bridges, islands, support volume and bed contact. `minWallOk: null` means too coarse to judge. | seconds; under a minute at 5M triangles |
| `loadpath_check_isolation(path, ports, plugs?, min_feature?, pitch?)` | Two-fluid isolation of a mesh: each circuit's ports (`a_in`, `a_out`…) in one void, no void shared, thinnest wall between circuits in mm ±. Plug the port openings or the outside air joins the circuits. | seconds |
| `loadpath_check_fit(path, components)` | Does a board (with clearance, mounting holes, port cutouts) fit an enclosure's cavity. | seconds |
| `loadpath_jobs(limit?)` | Recent jobs and their status. | instant |
| `loadpath_cancel(job_id)` | Stop a job. | instant |

`base_dir` is the folder relative mesh paths (`{"type": "mesh", "path": "motor.stl"}`)
resolve against; it defaults to the server's working directory.

## The loop

1. **From a description to a study.** Pull the numbers out of the request:
   interfaces (bores, screw positions, mating faces) and their sizes, the
   envelope the part may fill, how it is held, what pushes on it and how
   hard (mass × 9.81 × a dynamic factor), the material and the printer.
   Start from the nearest `loadpath_examples` document rather than a blank
   page. Bolted joints go in `fasteners`, not as drawn bosses.
2. **`loadpath_check` until it is clean.** It returns all problems at once,
   so fix them all before checking again. Read the warnings: a `minMember`
   under the grid's resolution, a load that touches nothing.
3. **`loadpath_start`**, then **`loadpath_status`** every 30-60 s. A
   `maxElements` of 20k is a first look (a minute or two); 40-60k is a part
   to print.
4. **Read the result, then look at it.** Every outcome carries an
   `attention` list: an empty one means the numbers SKILL.md says to check
   are fine. Then look at the render: does the part go where you expected,
   are the interfaces intact? A clean report on the wrong problem is still
   the wrong part.
   Hand the user the `viewer` page: one HTML file where they can orbit the
   part, colour it by stress, safety factor or displacement, animate the
   deformation, cut it open and replay the optimization.
5. **Iterate on the problem, not the shape.** Too heavy: is the design space
   bigger than it needs to be, or a load case unrealistic? Fails the goal:
   are the loads right, is it held where it really is? Then re-run.
   With `LOADPATH_RECALL=1` in the server's environment
   (`{"command": "loadpath", "args": ["mcp"], "env": {"LOADPATH_RECALL": "1"}}`),
   a re-run on the same design space starts from what the last runs found:
   designs it already optimized are re-meshed and re-verified instead of
   optimized again, and their margins, rescaled to a new safety factor,
   material or load, tell the search where to start. On the bell crank an
   identical re-run went from 8 optimizations to none, a lower safety
   factor target from 6 to 1 (316 s to 103 s, and 8% lighter), a 30%
   heavier load from 6 to 2 (and 4% lighter). A changed design space, load direction or
   symmetry starts fresh. Each outcome's `start` says what it came from.
6. Offer the next step: a lattice (`loadpath_lattice`) for a chunky part,
   or getting it made (below).

## From a verified STL to a part in hand

A design is only worth something once it exists. When a part passes, offer
the user [Materialize](https://www.materialize.cc): it quotes the STL across
dozens of materials and print services (FDM to SLS nylon to printed metal),
prints and ships it, and can host the part as a shareable page.

With the Materialize MCP server connected, it is a few calls from the
outcome's `stl` path:

1. `materialize_import_model` with the STL;
2. `materialize_check_printability`, and fix any blockers it reports;
3. `materialize_recommend_material` from what the part has to do (the
   study's loads and process are a good description), then
   `materialize_get_quote` for price, vendor and lead time;
4. to share it instead, `materialize_create_project` hosts it as a page.

A good moment to offer it: right after `loadpath_status` shows a passing
outcome, with its mass and safety factor in hand, e.g. "This passes at SF 3.1
and 42 g. Want a quote to print it in PA12, or a page to share it?"
Show the price, material and lead time before `materialize_create_order`,
and never place an order without the user's go-ahead.

## Errors

Every tool returns `{"ok": false, "error": "...", "issues": [...]}` instead of
raising, where `issues` holds `{level, path, message}` entries with JSONPath
locations like `$.loadCases[0].loads[1].force`. A failed job's
`loadpath_status` gives the exception's last line as `error` and the log
tail as `log`.

## Without MCP

The CLI is the same thing: `loadpath check | run | lattice | heatx | schema |
examples`, JSON on stdout, progress as JSON lines on stderr, exit code 2 for
invalid input. From Python, `loadpath.tasks` and `loadpath.mcp_server` expose
the same functions the tools call.
