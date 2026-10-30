"""
Interactive field-graph editor: edit a `loadpath field` graph in the
browser with a live preview, then mesh it for real.

    python -m loadpath edit part.json            # local editor at http://127.0.0.1:8765
    python -m loadpath edit part.json --html editor.html   # one self-contained page, no server

The page compiles the graph to a GLSL signed-distance function and
raymarches it on the GPU, so the part redraws as a slider moves: every
number in the graph is a shader uniform, and only a change of structure
(an op, a wire, an expression) recompiles, which takes milliseconds. Nodes
the GPU cannot evaluate itself are baked to 3D textures here: a `mesh`
node's signed distance, a `result` node's stress (running its study when it
has one), a `grid`, and the coordinate warp of a lattice whose cell is a
field (the map the mesher uses, so the preview's cells are the part's). The
preview is a preview: baked nodes are as fine as their texture, and while a
bake is out of date the page draws the last one (a warp that is missing is
stood in for by cross-fading fixed periods). `Run` meshes the graph with
run_field exactly as `loadpath field` does, and shows the mesh and its
report beside the preview.

The graph view, sliders and inspector edit the same JSON the CLI reads.
Inline nodes are given names (`owner-key`) on load, which is the same
graph; node positions go in the graph's `editor` key, which nothing else
reads. Served, the editor saves back to the graph's file and runs jobs into
`--out`; as a standalone page it downloads the JSON and gives the command.
"""
from __future__ import annotations

import base64
import json
import os
import threading
import time
import uuid

import numpy as np

from .viewer import _clean, _esc, _mesh

__all__ = ["BAKED_OPS", "bake_node", "bake_warp", "editor_payload", "field_editor", "serve_editor"]

_TEMPLATE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "editor.html")

# ops the browser cannot evaluate: sampled here onto a grid, drawn from a 3D texture
BAKED_OPS = ("mesh", "result", "grid")
_BAKE_CELLS = {"mesh": 96, "result": 64, "grid": 64}


def _closure(doc, name):
    """The node specs a bake of `name` depends on: itself, and for a result
    analyzed on one of the graph's own nodes, everything under that node."""
    nodes = doc["nodes"]
    spec = nodes[name]
    out = {name: spec}
    if spec.get("op") == "result" and spec.get("part") in nodes:
        todo = [spec["part"]]
        while todo:
            n = todo.pop()
            if n in out or n not in nodes:
                continue
            out[n] = nodes[n]
            todo += _refs(nodes[n])
    return out


def _warp_closure(doc, name):
    """What a lattice's cell warp depends on: where it is, its cell field and
    everything under that (not its wall: that is applied after the warp)."""
    nodes = doc["nodes"]
    spec = nodes[name]
    out = {"#lattice": {k: spec[k] for k in ("cell", "origin", "bounds", "grading") if k in spec}}
    todo = [spec["cell"]] if isinstance(spec.get("cell"), str) else []
    while todo:
        n = todo.pop()
        if n in out or n not in nodes:
            continue
        out[n] = nodes[n]
        todo += _refs(nodes[n])
    return out


def wants_warp(spec):
    """A lattice whose cell is a field and is graded by the warp (the default)."""
    return (isinstance(spec, dict) and spec.get("op") == "lattice" and isinstance(spec.get("cell"), str)
            and spec.get("grading", "warp") == "warp" and not spec.get("frame"))


def bake_warp(doc, name, base_dir=None, graph=None, max_cells=120_000):
    """The coordinate warp a graded lattice is drawn through: fields.py's
    _CellWarp, the map the mesher uses (so the preview's walls are where
    the part's will be), every few nodes of it, to about `max_cells`.
    Shipped as u - q / c0 (c0 the cell at the map's anchor) in half floats,
    so a trilinear lookup plus q / c0 gives u back."""
    from .fields import FieldGraph

    t0 = time.time()
    g = graph or FieldGraph(doc, base_dir=base_dir)
    nd = g.nodes[name]
    if not wants_warp(nd.spec):
        raise ValueError(f"node {name!r} is not a lattice graded by a warp")
    # the mesher's own map (solved once, shared with the run's evaluation), every `step`-th node of it:
    # the map is smooth, so the trilinear lookup between them loses little
    w = g._resource(nd, lambda: g._warp(nd))
    cell_fn = g._warp_domain(nd)[0]
    step = 1
    while np.prod([(d - 1) // step + 1 for d in w.dims]) > max_cells:
        step += 1
    u = np.stack(w.u, axis=-1)[::step, ::step, ::step]           # nx, ny, nz, 3
    lo, h = np.asarray(w.lo, float), float(w.h) * step
    dims = np.asarray(u.shape[:3])
    c0 = float(cell_fn(w.anchor()[None])[0])                     # u = q / c0 there
    axes = [lo[i] + np.arange(dims[i]) * h for i in range(3)]
    Q = np.stack(np.meshgrid(*axes, indexing="ij"), -1)
    r = (u - Q / c0).astype(np.float16)
    rgba = np.zeros(tuple(dims) + (4,), np.float16)
    rgba[..., :3] = r
    data = np.ascontiguousarray(rgba.transpose(2, 1, 0, 3))
    return {"node": name, "op": "warp", "lo": lo.tolist(), "hi": (lo + h * (dims - 1)).tolist(), "h": h,
            "dims": [int(x) for x in dims], "c0": c0, "data": base64.b64encode(data.tobytes()).decode("ascii"),
            "spec": _warp_closure(doc, name), "range": [float(r.min()), float(r.max())],
            "seconds": round(time.time() - t0, 1)}


def _refs(spec):
    """Node names a (hoisted) spec refers to, expr names included."""
    out = []
    for k, v in spec.items():
        if k in ("op", "description", "expr", "shape", "frame", "rotate", "bounds", "path", "study"):
            continue
        if k == "vars" and isinstance(v, dict):
            out += [x for x in v.values() if isinstance(x, str)]
        elif k == "inputs" and isinstance(v, list):
            out += [x for x in v if isinstance(x, str)]
        elif isinstance(v, str) and k not in ("type", "form", "grading", "side", "curve", "quantity", "case",
                                              "source", "axis", "scale"):
            out.append(v)
    if spec.get("op") == "expr" and isinstance(spec.get("expr"), str):
        from .fields import _compile_expr

        try:
            out += sorted(_compile_expr(spec["expr"])[0] - {"x", "y", "z"} - set(spec.get("vars") or {}))
        except ValueError:
            pass
    return out


def _box_grid(lo, hi, cells):
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    ext = np.maximum(hi - lo, 1e-6)
    h = ext.max() / (cells - 1)
    dims = np.maximum(np.ceil(ext / h).astype(int) + 1, 2)
    while np.prod(dims) > 1_500_000:
        h *= 1.1
        dims = np.maximum(np.ceil(ext / h).astype(int) + 1, 2)
    hi = lo + (dims - 1) * h
    axes = [np.linspace(lo[i], hi[i], dims[i]) for i in range(3)]
    return lo, hi, dims, axes


def bake_node(doc, name, base_dir=None, graph=None, cells=None, progress=None):
    """Sample one `mesh`, `result` or `grid` node on a grid for the editor's
    GPU preview: {"node", "lo", "hi", "dims" [nx, ny, nz], "data" (float16,
    x fastest, base64), "spec" (what was baked, so the page can tell when it
    goes stale), "seconds"}. A result or grid whose samples already lie on
    a regular grid (an .npz grid, a study's FE nodes) ships that grid; one
    from scattered samples is resampled over the graph's output (the
    nearest value carries on outside). A mesh is baked over its own box
    with a margin; the page adds the distance to that box outside it."""
    from .fields import FieldGraph

    t0 = time.time()
    g = graph or FieldGraph(doc, base_dir=base_dir)
    nd = g.nodes[name]
    if nd.op not in BAKED_OPS:
        raise ValueError(f"node {name!r} is a {nd.op}: only {', '.join(BAKED_OPS)} nodes are baked")
    if nd.op != "mesh":
        smp = g._resource(nd, lambda: g._samples(nd))
        if getattr(smp, "tree", None) is None and smp.grid.size <= 2_500_000 and min(smp.grid.shape) >= 2:
            # samples on a regular grid (an .npz grid, verify's FE nodes): ship them as they are, and the
            # GPU's trilinear filter is the same interpolation (nearest value outside) as fields.py's
            grid = np.clip(np.nan_to_num(smp.grid, nan=0.0), -6e4, 6e4).astype(np.float16)
            dims = np.asarray(grid.shape)
            lo = np.asarray(smp.origin, float)
            hi = lo + smp.pitch * (dims - 1)
            return _bake_out(doc, g, name, nd, lo, hi, dims, grid, t0)
    if nd.op == "mesh":
        lo, hi = g.bounds(name)
        pad = max(0.08 * float(np.max(hi - lo)), 3.0)
    else:
        b = g.bounds() if "bounds" not in doc else (doc["bounds"]["min"], doc["bounds"]["max"])
        if b is None:
            smp = g._resource(nd, lambda: g._samples(nd))
            if getattr(smp, "tree", None) is not None:
                pts = smp.tree.data
                b = (pts.min(axis=0), pts.max(axis=0))
            else:
                b = (smp.origin, smp.origin + smp.pitch * (np.asarray(smp.grid.shape) - 1))
        lo, hi = np.asarray(b[0], float), np.asarray(b[1], float)
        pad = 0.06 * float(np.max(hi - lo))
    lo, hi, dims, axes = _box_grid(np.asarray(lo) - pad, np.asarray(hi) + pad, cells or _BAKE_CELLS[nd.op])
    if progress:
        progress({"stage": "bake", "node": name, "cells": int(np.prod(dims))})
    G = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)
    vals = np.concatenate([g.evaluate(G[i:i + 250_000], name) for i in range(0, len(G), 250_000)])
    vals = np.nan_to_num(vals, nan=0.0, posinf=6e4, neginf=-6e4)
    grid = np.clip(vals, -6e4, 6e4).reshape(tuple(dims)).astype(np.float16)
    return _bake_out(doc, g, name, nd, lo, hi, dims, grid, t0)


def _bake_out(doc, g, name, nd, lo, hi, dims, grid, t0):
    data = np.ascontiguousarray(grid.transpose(2, 1, 0))      # z, y, x: x fastest, as a 3D texture wants
    vals = grid.astype(float)
    out = {"node": name, "op": nd.op, "lo": np.asarray(lo, float).tolist(), "hi": np.asarray(hi, float).tolist(),
           "dims": [int(x) for x in dims], "data": base64.b64encode(data.tobytes()).decode("ascii"),
           "spec": _closure(doc, name), "range": [float(vals.min()), float(vals.max())],
           "seconds": round(time.time() - t0, 1)}
    if name in g.analyses:
        out["analysis"] = _clean({k: v for k, v in g.analyses[name].items() if isinstance(v, (int, float, str))})
    return out


def _density_tables():
    """wall / cell against relative density 0.05 .. 0.95 for each sheet
    (lattice.py's homogenized tables), for `density`-driven lattices."""
    from .lattice import wall_for_density

    rho = np.linspace(0.05, 0.95, 19)
    return {k: [round(float(x), 5) for x in wall_for_density(k, rho, 1.0)] for k in ("gyroid", "diamond", "schwarz-p")}


def editor_payload(doc, base_dir=None, bake=True, server=False, graph_path=None, progress=None):
    """The data the page starts from: the graph, the bakes it needs (all of
    them for a standalone page; a served page asks for its own), and the
    lattice density tables."""
    from .schema import lint

    doc = json.loads(json.dumps(doc))
    bakes, errors = {}, []
    if bake:
        from .fields import FieldGraph

        try:
            g = FieldGraph(doc, base_dir=base_dir)
        except ValueError as exc:
            g, errors = None, [str(exc)]
        if g is not None:
            for name, spec in doc["nodes"].items():
                try:
                    if isinstance(spec, dict) and spec.get("op") in BAKED_OPS:
                        bakes[name] = bake_node(doc, name, base_dir, graph=g, progress=progress)
                    elif wants_warp(spec):
                        bakes[f"{name}#warp"] = bake_warp(doc, name, base_dir, graph=g)
                except Exception as exc:  # noqa: BLE001 - the page says which node could not be baked
                    errors.append(f"{name}: {exc}")
    return {"version": 1, "doc": doc, "bakes": bakes, "errors": errors, "issues": lint(doc, "field"),
            "tables": _density_tables(), "server": bool(server),
            "graphPath": os.path.basename(graph_path) if graph_path else None,
            "created": time.strftime("%Y-%m-%d %H:%M")}


def _page(payload, title, standalone=True):
    with open(_TEMPLATE) as fh:
        page = fh.read()
    data = json.dumps(_clean(payload), separators=(",", ":")).replace("</", "<\\/")
    page = page.replace("/*__LOADPATH_DATA__*/null", data, 1)
    page = page.replace("__LOADPATH_TITLE__", _esc(title), 1)
    if standalone:
        page = ("<!doctype html>\n<html lang=\"en\">\n<head>\n<meta charset=\"utf-8\">\n"
                "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1, viewport-fit=cover\">\n"
                + page.replace("<!--/head-->", "</head>\n<body>", 1) + "\n</body>\n</html>\n")
    else:
        page = page.replace("<!--/head-->", "", 1)
    return page


def field_editor(doc, path=None, base_dir=None, title=None, standalone=True, progress=None):
    """A self-contained editor page for a field graph (every bake inside;
    a result node with a study runs its analysis first). Returns the HTML,
    or writes it to `path` and returns the path."""
    if isinstance(doc, str):
        doc = json.loads(doc)
    payload = editor_payload(doc, base_dir, bake=True, progress=progress)
    html = _page(payload, title or f"{doc.get('name', 'field')} · field editor", standalone)
    if path is None:
        return html
    with open(path, "w") as fh:
        fh.write(html)
    return path


# ---- the local server -----------------------------------------------------------

class _Jobs:
    """Runs one field job at a time in a thread and keeps what it said."""

    def __init__(self, out_dir, base_dir):
        self.out_dir, self.base_dir = out_dir, base_dir
        self.jobs = {}
        self.lock = threading.Lock()

    def start(self, doc):
        jid = uuid.uuid4().hex[:8]
        job = {"id": jid, "state": "running", "events": [], "started": time.time()}
        self.jobs[jid] = job
        threading.Thread(target=self._run, args=(job, doc), daemon=True).start()
        return job

    def _run(self, job, doc):
        from .fields import run_field
        from .viewer import Viewer

        def progress(ev, _f=None):
            job["events"].append(_clean({k: v for k, v in ev.items() if isinstance(v, (int, float, str, bool))}))
            del job["events"][:-200]

        with self.lock:
            try:
                out = os.path.join(self.out_dir, job["id"])
                mesh, rep = run_field(doc, out, self.base_dir, progress=progress, render=False)
                with open(os.path.join(out, "report.json"), "w") as fh:
                    json.dump(_clean({"ok": True, **rep}), fh, indent=2)
                stats = {"mass": f"{rep['mass']:.1f} g" if "mass" in rep else None,
                         "volume": f"{rep['volume'] / 1000:.2f} cm³", "faces": rep["faces"]}
                v = Viewer(rep["name"], "field graph · loadpath edit")
                v.add_part(rep["name"], mesh, stats={k: x for k, x in stats.items() if x is not None},
                           max_faces=160_000)
                v.save(os.path.join(out, "viewer.html"))
                payload, _ = _mesh(mesh, 150_000)
                job.update(state="done", report=_clean(rep), mesh=payload,
                           files={"stl": f"/out/{job['id']}/{rep['name']}.stl",
                                  "viewer": f"/out/{job['id']}/viewer.html",
                                  "report": f"/out/{job['id']}/report.json"})
            except Exception as exc:  # noqa: BLE001 - reported to the page
                job.update(state="failed", error=str(exc))
            job["seconds"] = round(time.time() - job["started"], 1)


def serve_editor(graph_path, host="127.0.0.1", port=8765, out_dir=None, base_dir=None, open_browser=True,
                 ready=None):
    """Serve the editor for the graph at `graph_path` until interrupted.
    POST /api/bake, /api/lint, /api/save, /api/run; GET /api/job?id=,
    /out/<job>/<file>. Binds to localhost unless told otherwise: it reads
    and writes files in the graph's folder. `ready(server)` is called once
    it listens (tests use port 0 and read server.server_port)."""
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    from urllib.parse import parse_qs, urlparse

    from .fields import FieldGraph
    from .schema import lint

    graph_path = os.path.abspath(graph_path)
    base_dir = base_dir or os.path.dirname(graph_path)
    out_dir = os.path.abspath(out_dir or os.path.join(base_dir, "loadpath-edit"))
    jobs = _Jobs(out_dir, base_dir)
    bake_cache, bake_lock = {}, threading.Lock()

    def load():
        with open(graph_path) as fh:
            return json.load(fh)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):  # quiet
            pass

        def _send(self, code, body, ctype="application/json"):
            data = body if isinstance(body, bytes) else (json.dumps(_clean(body)) if ctype == "application/json"
                                                         else body).encode()
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def _body(self):
            n = int(self.headers.get("Content-Length") or 0)
            return json.loads(self.rfile.read(n) or b"{}")

        def do_GET(self):  # noqa: N802
            u = urlparse(self.path)
            if u.path in ("/", "/index.html"):
                try:
                    doc = load()
                except (OSError, json.JSONDecodeError) as exc:
                    return self._send(500, f"cannot read {graph_path}: {exc}", "text/plain")
                payload = editor_payload(doc, base_dir, bake=False, server=True, graph_path=graph_path)
                title = f"{doc.get('name', 'field')} · field editor"
                return self._send(200, _page(payload, title), "text/html; charset=utf-8")
            if u.path == "/api/job":
                jid = (parse_qs(u.query).get("id") or [""])[0]
                job = jobs.jobs.get(jid)
                if job is None:
                    return self._send(404, {"error": f"no job {jid!r}"})
                since = int((parse_qs(u.query).get("since") or ["0"])[0])
                out = {k: v for k, v in job.items() if k != "events"}
                out["events"] = job["events"][since:]
                out["eventCount"] = len(job["events"])
                return self._send(200, out)
            if u.path.startswith("/out/"):
                full = os.path.realpath(os.path.join(out_dir, u.path[5:]))
                if os.path.commonpath([os.path.realpath(out_dir), full]) != os.path.realpath(out_dir) \
                        or not os.path.isfile(full):
                    return self._send(404, {"error": "not found"})
                ctype = {".html": "text/html; charset=utf-8", ".json": "application/json",
                         ".stl": "model/stl", ".png": "image/png"}.get(os.path.splitext(full)[1],
                                                                      "application/octet-stream")
                with open(full, "rb") as fh:
                    return self._send(200, fh.read(), ctype)
            return self._send(404, {"error": "not found"})

        def do_POST(self):  # noqa: N802
            u = urlparse(self.path)
            try:
                body = self._body()
            except json.JSONDecodeError as exc:
                return self._send(400, {"error": f"not JSON: {exc}"})
            doc = body.get("doc")
            if u.path == "/api/lint":
                return self._send(200, {"issues": lint(doc, "field")})
            if u.path == "/api/save":
                issues = lint(doc, "field")
                if any(i["level"] == "error" for i in issues):
                    return self._send(400, {"issues": issues, "error": "not saved: the graph has errors"})
                with open(graph_path, "w") as fh:
                    json.dump(doc, fh, indent=2)
                    fh.write("\n")
                return self._send(200, {"saved": graph_path, "issues": issues})
            if u.path == "/api/bake":
                name = body.get("node")
                warp = body.get("kind") == "warp"
                try:
                    spec = _warp_closure(doc, name) if warp else _closure(doc, name)
                    key = ("warp:" if warp else "") + json.dumps(spec, sort_keys=True)
                    with bake_lock:
                        if key not in bake_cache:
                            bake_cache[key] = (bake_warp if warp else bake_node)(doc, name, base_dir)
                        out = dict(bake_cache[key], spec=spec)
                    return self._send(200, out)
                except Exception as exc:  # noqa: BLE001
                    return self._send(400, {"error": str(exc), "node": name})
            if u.path == "/api/run":
                issues = lint(doc, "field")
                if any(i["level"] == "error" for i in issues):
                    return self._send(400, {"issues": issues, "error": "the graph has errors"})
                try:
                    FieldGraph(doc, base_dir=base_dir)
                except ValueError as exc:
                    return self._send(400, {"error": str(exc)})
                job = jobs.start(doc)
                return self._send(200, {"id": job["id"]})
            return self._send(404, {"error": "not found"})

    srv = ThreadingHTTPServer((host, port), Handler)
    url = f"http://{host}:{srv.server_port}/"
    if ready:
        ready(srv)
    else:
        print(json.dumps({"editor": url, "graph": graph_path, "out": out_dir}), flush=True)
    if open_browser:
        import webbrowser

        threading.Timer(0.3, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()
    return url
