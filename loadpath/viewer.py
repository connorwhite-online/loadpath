"""
Interactive viewers: one self-contained HTML file per study (or heat
exchanger) that opens in any browser. Orbit the part, colour it by von
Mises stress, safety factor or displacement for each load case, scrub an
exaggerated deformation, cut it open with a section plane, play the
optimization back iteration by iteration, and put an outcome beside another
(or beside the part it replaces) to compare.

    from loadpath.viewer import HistoryRecorder, study_viewer
    rec = HistoryRecorder()
    result = run_study(study, on_progress=rec)
    study_viewer(result, "bracket.html", history=rec)

The file carries its own data (meshes and fields as packed arrays) and
loads three.js from a CDN, so it needs a network connection the first time
it is opened and nothing else: no server, no Python. render.py is for
reports and vision models; this is for people.

Lower level: `Viewer` takes parts, fields, overlays and history frames
directly, for results that did not come out of run_study.
"""
from __future__ import annotations

import base64
import json
import os
import time

import numpy as np

__all__ = ["HistoryRecorder", "Viewer", "study_viewer", "heat_exchanger_viewer"]

_TEMPLATE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "viewer.html")

# overlay colours (sRGB hex); the page keeps its own palette for the UI
_KEEP_OUT = "#e0533d"
_PRESERVE = "#8a94a6"
_SUPPORT = "#2f9e6f"
_DESIGN = "#5b7fd6"
_LOAD = "#2f5bd3"
_FLUIDS = ["#2f6fd6", "#e8892b", "#3aa57a", "#b04fc7"]


# -- packing ---------------------------------------------------------------

def _b64(a) -> str:
    return base64.b64encode(np.ascontiguousarray(a).tobytes()).decode("ascii")


def _f32(a) -> dict:
    a = np.asarray(a, np.float32)
    return {"t": "f32", "n": int(a.shape[-1]) if a.ndim > 1 else 1, "d": _b64(a.ravel())}


def _q16(a) -> dict:
    """Quantize to 16 bits per component against the array's own range:
    a tenth of a micron on a 100 mm part, a sixty-thousandth of a stress
    range. Half the bytes of float32, for history frames and fields."""
    a = np.asarray(a, np.float64)
    n = int(a.shape[-1]) if a.ndim > 1 else 1
    a2 = a.reshape(-1, n)
    lo, hi = a2.min(axis=0) if len(a2) else np.zeros(n), a2.max(axis=0) if len(a2) else np.ones(n)
    span = np.where(hi > lo, hi - lo, 1.0)
    q = np.round((a2 - lo) / span * 65535.0).astype(np.uint16)
    return {"t": "q16", "n": n, "lo": lo.tolist(), "hi": (lo + span).tolist(), "d": _b64(q.ravel())}


def _idx(faces) -> dict:
    f = np.asarray(faces)
    if f.size and f.max() < 65536:
        return {"t": "u16", "n": 3, "d": _b64(f.astype(np.uint16).ravel())}
    return {"t": "u32", "n": 3, "d": _b64(f.astype(np.uint32).ravel())}


def _decimate(mesh, faces):
    """Quadric decimation (trimesh, with the fast_simplification package),
    else vertex clustering on a grid sized to land near `faces`: coarser
    looking, but it never ships a 20 MB page because a package is missing."""
    if faces is None or len(mesh.faces) <= faces:
        return mesh
    try:
        return mesh.simplify_quadric_decimation(face_count=int(faces))
    except Exception:  # noqa: BLE001 - a display proxy only
        pass
    import trimesh
    cell = float(np.sqrt(mesh.area / (0.5 * faces)))
    for _ in range(6):
        key = np.floor(mesh.vertices / cell).astype(np.int64)
        _, inv = np.unique(key, axis=0, return_inverse=True)
        inv = inv.ravel()
        f = inv[mesh.faces]
        f = f[(f[:, 0] != f[:, 1]) & (f[:, 1] != f[:, 2]) & (f[:, 0] != f[:, 2])]
        if len(f) <= 1.3 * faces:
            break
        cell *= 1.25
    v = np.zeros((inv.max() + 1, 3))
    np.add.at(v, inv, mesh.vertices)
    v /= np.bincount(inv)[:, None]
    return trimesh.Trimesh(v, f, process=True)


def _mesh(mesh, max_faces=None, quantize=False) -> dict:
    m = _decimate(mesh, max_faces)
    return {"p": (_q16 if quantize else _f32)(np.asarray(m.vertices)), "i": _idx(m.faces)}, m


def _manifold_mesh(shape):
    import trimesh
    mf = shape.to_manifold().to_mesh()
    return trimesh.Trimesh(np.asarray(mf.vert_properties)[:, :3], np.asarray(mf.tri_verts), process=False)


def _clean(o):
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return _clean(o.tolist())
    if isinstance(o, (np.floating, float)):
        f = float(o)
        return None if f != f or f in (float("inf"), float("-inf")) else f
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


# -- history ---------------------------------------------------------------

class HistoryRecorder:
    """Pass as run_study's on_progress (or call it from yours) to keep the
    density field every `every` iterations, as float16 (120k cells is 240 kB
    a frame). A search runs the optimizer several times, at different
    volume fractions; each run is kept, in order. `chain` is called with
    every event too, so this can sit in front of a progress printer."""

    def __init__(self, every: int = 2, chain=None):
        self.every = max(1, int(every))
        self.chain = chain
        self.runs: list[dict] = []

    def __call__(self, event, xp=None):
        if self.chain is not None:
            self.chain(event, xp)
        if event.get("stage") != "optimize":
            return
        if "iteration" not in event:
            self.runs.append({"volumeFraction": event.get("volumeFraction"), "frames": []})
            return
        if xp is None:
            return
        if not self.runs:
            self.runs.append({"volumeFraction": event.get("volumeFraction"), "frames": []})
        run = self.runs[-1]
        frame = {"iteration": int(event["iteration"]), "compliance": event.get("compliance"),
                 "volume": event.get("volume"), "beta": event.get("beta"),
                 "x": np.asarray(xp, np.float16).copy()}
        if frame["iteration"] % self.every == 0 or not run["frames"]:
            run["frames"].append(frame)
            run["last"] = None
        else:
            run["last"] = frame   # so the end of a run is kept even off the stride

    def _frames(self):
        for r in self.runs:
            yield r["frames"] + ([r["last"]] if r.get("last") is not None else [])

    @property
    def frame_count(self) -> int:
        return sum(len(f) for f in self._frames())


def _iso(domain, x, level=0.5, max_faces=None):
    """The solid (density above `level`) as a surface in mm."""
    import trimesh
    from skimage.measure import marching_cubes

    g = np.pad(domain.full_grid(np.asarray(x, np.float32)), 1)
    if g.max() <= level:
        return None
    v, f, _, _ = marching_cubes(g, level, spacing=(domain.pitch,) * 3)
    v = v + np.asarray(domain.origin) - 0.5 * domain.pitch
    m = trimesh.Trimesh(v, f[:, ::-1], process=True)
    return _decimate(m, max_faces)


def _history_payload(study, recorder, max_frames=90, max_faces=6000):
    from .domain import build_domain

    runs = list(recorder._frames())
    flat = [(ri, f) for ri, fr in enumerate(runs) for f in fr]
    if not flat:
        return None
    domain = build_domain(study)
    if any(len(f["x"]) != domain.n_elem for _, f in flat):
        return None   # recorded on another grid (a different resolution): nothing to place it on
    if len(flat) > max_frames:
        # spread evenly, and always keep each run's last frame (its converged design)
        keep = set(np.linspace(0, len(flat) - 1, max(2, max_frames - len(runs))).round().astype(int).tolist())
        ends = np.cumsum([len(fr) for fr in runs]) - 1
        keep |= set(int(e) for e in ends if e >= 0)
        flat = [flat[i] for i in sorted(keep)]
    frames = []
    for ri, f in flat:
        m = _iso(domain, f["x"], max_faces=max_faces)
        frames.append(_clean({
            "run": ri, "volumeFraction": recorder.runs[ri]["volumeFraction"], "iteration": f["iteration"],
            "compliance": f["compliance"], "volume": f["volume"], "beta": f["beta"],
            "mesh": None if m is None else _mesh(m, quantize=True)[0]}))
    return {"runs": len(runs), "frames": frames}


# -- the scene -------------------------------------------------------------

class Viewer:
    """A scene for the HTML viewer. Units are mm, N and MPa, z up."""

    def __init__(self, title: str = "loadpath", subtitle: str | None = None):
        self.title = title
        self.subtitle = subtitle
        self.parts: list[dict] = []
        self.overlays: list[dict] = []
        self.arrows: list[dict] = []
        self.history = None
        self.notes: list[str] = []

    def add_part(self, name, mesh, fields=None, strength=None, allowance=1.0, target_sf=None,
                 stats=None, max_faces=80_000, color=None, kind="part"):
        """A selectable body. `fields` is verify_part(..., fields=True)["fields"]
        (analysis nodes; per case von Mises and displacement), mapped onto
        the drawn surface by nearest node. `stats` is shown beside it: a dict
        of label -> value (already formatted or a number)."""
        payload, m = _mesh(mesh, max_faces)
        part = {"name": name, "kind": kind, "mesh": payload, "stats": _clean(stats or {}),
                "color": color, "strength": strength, "allowance": allowance, "targetSF": target_sf,
                "cases": []}
        if fields is not None and fields.get("cases"):
            from scipy.spatial import cKDTree
            # inverse-distance over the nearest few analysis nodes: one node
            # each speckles the surface where it cuts across voxel corners
            dist, idx = cKDTree(fields["nodes"]).query(np.asarray(m.vertices), k=4)
            w = 1.0 / np.maximum(dist, 1e-6) ** 2
            w /= w.sum(axis=1, keepdims=True)
            for c in fields["cases"]:
                vm = (np.asarray(c["vonMises"])[idx] * w).sum(axis=1)
                d = (np.asarray(c["displacement"])[idx] * w[..., None]).sum(axis=1)
                part["cases"].append({"name": c["name"], "vonMises": _q16(vm), "displacement": _q16(d),
                                      "peak": float(vm.max()),
                                      "maxDisplacement": float(np.linalg.norm(d, axis=1).max())})
        self.parts.append(part)
        return part

    def add_overlay(self, name, mesh, color="#8a94a6", opacity=0.25, group="context", visible=True,
                    max_faces=40_000, section=True):
        """A context body drawn translucent (keep-outs, the design space, a
        fluid space). `section` says whether the section plane cuts it."""
        payload, _ = _mesh(mesh, max_faces)
        self.overlays.append({"name": name, "group": group, "color": color, "opacity": float(opacity),
                              "visible": bool(visible), "section": bool(section), "mesh": payload})

    def add_arrow(self, at, direction, label="", case=None, color=_LOAD):
        d = np.asarray(direction, float)
        n = float(np.linalg.norm(d))
        if n == 0:
            return
        self.arrows.append({"at": np.asarray(at, float).tolist(), "dir": (d / n).tolist(),
                            "label": label, "case": case, "color": color})

    def add_study_context(self, study, design_space=True):
        """Keep-outs, preserved regions, supports, load arrows and the
        design space envelope, as toggleable layers."""
        for r in study.keep_out:
            try:
                self.add_overlay(f"keep-out · {r.id}", _manifold_mesh(r.shape), _KEEP_OUT, 0.18, "keep-outs")
            except Exception:  # noqa: BLE001 - context only
                pass
        for r in study.preserve:
            try:
                self.add_overlay(f"preserve · {r.id}", _manifold_mesh(r.shape), _PRESERVE, 0.22, "preserved",
                                 visible=False)
            except Exception:  # noqa: BLE001
                pass
        for i, s in enumerate(study.supports):
            try:
                shape = s.shape if s.shape is not None else study.region_shape(s.region)[0]
                label = s.region or f"support {i + 1}"
                self.add_overlay(f"support · {label}", _manifold_mesh(shape), _SUPPORT, 0.35, "supports")
            except Exception:  # noqa: BLE001
                pass
        if design_space:
            for i, s in enumerate(study.design_space):
                try:
                    self.add_overlay(f"design space {i + 1}" if len(study.design_space) > 1 else "design space",
                                     _manifold_mesh(s), _DESIGN, 0.07, "design space", visible=False)
                except Exception:  # noqa: BLE001
                    pass
        for case in study.load_cases:
            for ld in case.loads:
                try:
                    shape = ld.shape if ld.shape is not None else study.region_shape(ld.region)[0]
                except ValueError:
                    continue
                lo, hi = shape.bounds()
                f = np.asarray(ld.force, float)
                mag = float(np.linalg.norm(f))
                if mag > 0:
                    self.add_arrow(0.5 * (np.asarray(lo) + np.asarray(hi)), f, f"{mag:.0f} N", case.name)
                mom = float(np.linalg.norm(ld.moment))
                if mom > 0:
                    self.add_arrow(0.5 * (np.asarray(lo) + np.asarray(hi)), ld.moment,
                                   f"{mom / 1000:.2f} N·m", case.name, color="#8b5cf6")

    def to_dict(self) -> dict:
        return _clean({"version": 1, "title": self.title, "subtitle": self.subtitle, "parts": self.parts,
                       "overlays": self.overlays, "arrows": self.arrows, "history": self.history,
                       "notes": self.notes, "created": time.strftime("%Y-%m-%d %H:%M")})

    def html(self, standalone: bool = True) -> str:
        """The page. `standalone=False` leaves out the doctype, html, head and
        body tags, for hosts that wrap a page in their own skeleton."""
        with open(_TEMPLATE) as fh:
            page = fh.read()
        data = json.dumps(self.to_dict(), separators=(",", ":")).replace("</", "<\\/")
        page = page.replace("/*__LOADPATH_DATA__*/null", data, 1)
        page = page.replace("__LOADPATH_TITLE__", _esc(self.title), 1)
        if standalone:
            page = ("<!doctype html>\n<html lang=\"en\">\n<head>\n<meta charset=\"utf-8\">\n"
                    "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1, viewport-fit=cover\">\n"
                    + page.replace("<!--/head-->", "</head>\n<body>", 1) + "\n</body>\n</html>\n")
        else:
            page = page.replace("<!--/head-->", "", 1)
        return page

    def save(self, path, standalone: bool = True) -> str:
        with open(path, "w") as fh:
            fh.write(self.html(standalone))
        return path


def _esc(s: str) -> str:
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _stats(o, study) -> dict:
    r = o.report
    out = {"mass": f"{r['mass']:.1f} g"}
    if r.get("massReduction") is not None:
        out["lighter than the design space"] = f"{100 * r['massReduction']:.0f}%"
    out["volume fraction"] = f"{o.volume_fraction:.3f}"
    sf = r.get("safetyFactor")
    if sf is not None and np.isfinite(sf):
        out["safety factor"] = f"{sf:.2f}" + (f"  (goal {study.goal.safety_factor:g})" if study.goal.safety_factor else "")
    cases = r.get("cases") or []
    if cases:
        d = max(c["deflection"] for c in cases)
        out["deflection at loads"] = f"{d:.3f} mm" + (f"  (limit {study.goal.max_deflection:g})"
                                                      if study.goal.max_deflection else "")
    if r.get("p5Member") is not None:
        out["thin members (p5)"] = f"{r['p5Member']:.2f} mm"
    if o.passes is not None:
        out["meets the goal"] = "yes" if o.passes else "no"
    return out


def study_viewer(result, path=None, history: HistoryRecorder | None = None, outcomes="best",
                 baseline=None, baseline_name="original", fields=True, title=None, max_faces=80_000,
                 standalone=True):
    """A viewer for a StudyResult. `outcomes` is "best", "all" or a list of
    indices; each becomes a selectable part. `fields` verifies each again
    with stress fields (a few seconds to a minute a part: it is the same
    analysis the study ran). `baseline` (a trimesh or a mesh file) is the
    part this replaces, verified under the same loads, to compare against.
    `history` is a HistoryRecorder the study ran with. Returns the HTML, and
    writes it to `path` when given."""
    import trimesh

    from .verify import STRESS_ALLOWANCE, verify_part

    study = result.study
    pitch = (result.domain or {}).get("pitch")
    vkw = {"min_pitch": 0.5 * pitch, "contact_band": pitch} if pitch else {}
    if outcomes == "best":
        idx = [result.best if result.best is not None else (len(result.outcomes) - 1)] if result.outcomes else []
    elif outcomes == "all":
        idx = list(range(len(result.outcomes)))
    else:
        idx = list(outcomes)
    v = Viewer(title or study.name, f"{study.material.name} · {len(study.load_cases)} load case"
               + ("s" if len(study.load_cases) != 1 else ""))
    mat = study.material
    # keep the page light enough to open on a phone: ~160k faces across the parts
    n_parts = len(idx) + (baseline is not None)
    max_faces = max(20_000, min(max_faces, 160_000 // max(1, n_parts)))
    for i in idx:
        o = result.outcomes[i]
        rep = verify_part(study, o.mesh, fields=True, **vkw) if fields else None
        name = f"vf {o.volume_fraction:.2f} · {o.report['mass']:.1f} g"
        if i == result.best:
            name = "★ " + name
        v.add_part(name, o.mesh, rep["fields"] if rep else None, mat.strength, STRESS_ALLOWANCE,
                   study.goal.safety_factor, _stats(o, study), max_faces=max_faces)
    if baseline is not None:
        bm = trimesh.load(baseline, force="mesh") if isinstance(baseline, (str, os.PathLike)) else baseline
        rep = verify_part(study, bm, fields=True, **vkw) if fields else None
        st = {"mass": f"{rep['mass']:.1f} g"} if rep else {}
        if rep:
            st["safety factor"] = f"{rep['safetyFactor']:.2f}"
            st["deflection at loads"] = f"{max(c['deflection'] for c in rep['cases']):.3f} mm"
        v.add_part(baseline_name, bm, rep["fields"] if rep else None, mat.strength, STRESS_ALLOWANCE,
                   study.goal.safety_factor, st, max_faces=max_faces, kind="baseline")
    v.add_study_context(study)
    if history is not None:
        v.history = _history_payload(study, history)
    page = v.html(standalone)
    if path:
        with open(path, "w") as fh:
            fh.write(page)
    return page


def heat_exchanger_viewer(mesh, spaces: dict, report: dict | None = None, path=None, title="heat exchanger",
                          max_faces=120_000, standalone=True):
    """A viewer for design_heat_exchanger(..., fluid_meshes=True): the part,
    and each fluid's space as its own layer, cut open with the section
    plane to see the two networks interleave."""
    v = Viewer(title, "dual-fluid heat exchanger")
    st = {}
    if report:
        units = (("solidVolume", "solid", 1e-3, "cm³"), ("coreVolume", "core", 1e-3, "cm³"),
                 ("cell", "gyroid cell", 1, "mm"), ("wall", "wall", 1, "mm"),
                 ("minWallBetweenFluids", "thinnest wall between fluids", 1, "mm"),
                 ("minOuterWall", "thinnest outer wall", 1, "mm"), ("overhangArea", "overhang", 1, "mm²"))
        for k, lab, f, u in units:
            if isinstance(report.get(k), (int, float)):
                st[lab] = f"{report[k] * f:.3g} {u}"
        if "isolated" in report:
            st["fluids kept apart"] = "yes" if report["isolated"] else "no"
    v.add_part(title, mesh, stats=st, max_faces=max_faces)
    for i, (name, fm) in enumerate(spaces.items()):
        v.add_overlay(f"fluid · {name}", fm, _FLUIDS[i % len(_FLUIDS)], 0.85, f"fluid {name}", visible=(i == 0),
                      max_faces=max_faces)
    if path:
        v.save(path, standalone)
    return v.html(standalone)
