"""
Design memory: what earlier runs of the same design space found, so a rerun
starts at the answer instead of searching for it again.

A safety-factor study spends most of its time finding the volume fraction:
the bell crank optimizes six fractions (0.25, 0.125, 0.0625, 0.094, 0.078,
0.070) and confirms two picks on a finer grid before it settles, each
optimization 20-30 s and each confirm about a minute. Agents rerun studies
with small edits (another target, a stronger material, a heavier load), and
every rerun used to walk that whole path again.

What carries over, and why it is safe:

- The optimizer minimizes compliance at a fixed volume fraction. Its design
  does not depend on the load's overall scale or the material's E or
  strength, only on the domain, the loads' directions and relative sizes,
  Poisson's ratio and the code. Stress is linear in load and in nothing else
  here (verification is linear elastic). So for a past outcome on the same
  domain under the same load pattern, its safety factor and deflection under
  the new study are its old ones rescaled by strength, E and load scale, and
  its margin is known before anything runs.
- With the same model (Poisson's ratio, process, code) a past outcome is
  *replayed*: the stored density field is meshed and verified afresh (the
  part delivered is always verified by today's code) and the optimizer is
  skipped. The fine-grid confirmation of a replayed pick is rescaled from
  the stored one rather than run again.
- With a different model (another Poisson's ratio, a changed solver), the
  stored margins still say roughly where the answer is, and the nearest
  stored design is the starting design instead of a coarse solve.

Off unless asked for: LOADPATH_RECALL=1 (or a folder path) turns it on for
everything, `loadpath run --recall` for one run. Eval runs leave it off so a
scorecard measures the optimizer, not the cache. Stored under
~/.cache/loadpath/designs, one folder per domain; LOADPATH_RECALL_MB caps the
total (default 1000, oldest domains dropped first).
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
from dataclasses import dataclass

import numpy as np

__all__ = ["Recall", "enabled"]

_MAX_ENTRIES = 32           # per domain, newest kept
# A replayed fraction stands in for a requested one this close to it
# (relative): the search's own stopping rule accepts a bracket of 10%.
_SNAP = 0.04


def enabled() -> bool:
    v = os.environ.get("LOADPATH_RECALL", "").strip()
    return bool(v) and v.lower() not in ("0", "false", "off", "no")


def _root() -> str:
    v = os.environ.get("LOADPATH_RECALL", "").strip()
    if v and v not in ("1", "true", "on", "yes") and (os.sep in v or v.startswith(".")):
        return os.path.expanduser(v)
    base = os.environ.get("XDG_CACHE_HOME") or os.path.expanduser("~/.cache")
    return os.path.join(base, "loadpath", "designs")


def _h(*parts) -> str:
    h = hashlib.sha1()
    for p in parts:
        if isinstance(p, np.ndarray):
            h.update(str((p.dtype.str, p.shape)).encode())
            h.update(np.ascontiguousarray(p).tobytes())
        elif hasattr(p, "tocsr"):
            c = p.tocsr()
            h.update(_h(c.indptr, c.indices, np.round(c.data, 9)).encode())
        else:
            h.update(repr(p).encode())
        h.update(b"|")
    return h.hexdigest()[:20]


_CODE = None


# The modules a design, its mesh and its verification come out of (and what
# they import). A change to any of them retires replays; a change elsewhere
# (the search in study.py, rendering, the server) leaves them standing.
_DESIGN_CODE = ("am", "constrained", "domain", "durability", "fasteners", "fem", "manufacture", "materials",
                "mg", "mma", "pressure", "printcheck", "reconstruct", "shapes", "simp", "spec", "thermal",
                "verify", "voxels")


def _code_key() -> str:
    global _CODE
    if _CODE is None:
        here = os.path.dirname(os.path.abspath(__file__))
        h = hashlib.sha1()
        for m in _DESIGN_CODE:
            with open(os.path.join(here, m + ".py"), "rb") as fh:
                h.update(fh.read())
        _CODE = h.hexdigest()[:20]
    return _CODE


def _load_scale(domain) -> float:
    s = max((float(np.abs(f).max()) for f in domain.loads if f is not None and np.size(f)), default=0.0)
    return s if s > 0 else 1.0


@dataclass
class _Pred:
    entry: dict
    margin: float | None       # predicted, under this study
    confirm: float | None      # predicted fine-grid margin, when one is stored
    exact: bool                # same model: replayable


class Recall:
    """One study's view of the memory for its domain. Build with `open`."""

    def __init__(self, folder, keys, study, domain, scale):
        self.folder = folder
        self.keys = keys
        self.study = study
        self.domain = domain
        self.scale = scale
        self.index = []
        self.dirty = False
        self.used = {"replayed": 0, "warm": 0, "confirmsSkipped": 0}
        try:
            with open(os.path.join(folder, "index.json")) as fh:
                self.index = json.load(fh).get("entries", [])
        except (OSError, ValueError):
            self.index = []

    @classmethod
    def open(cls, study, domain, radius: float, making: dict) -> "Recall | None":
        if not enabled():
            return None
        d = domain
        mirror = {k: _h(np.asarray(v)) for k, v in sorted(d.mirror.items())}
        geometry = _h(d.dims, round(float(d.pitch), 6), np.round(np.asarray(d.origin, float), 6),
                      d.mesh.elem_ijk.astype(np.int32), d.preserve, d.fixed, d.support_nodes, d.load_nodes,
                      d.springs, mirror, sorted((k, repr(v)) for k, v in making.items()),
                      round(float(radius), 6))
        scale = _load_scale(d)
        loads = _h([np.round(np.asarray(f, float) / scale, 9) for f in d.loads],
                   None if d.gravity is None else np.round(np.asarray(d.gravity, float) / scale, 12),
                   list(d.case_weights), study.max_iterations)
        proc = study.raw.get("process")
        model = _h(repr(study.elastic(design=True)), repr(study.elastic()),
                   json.dumps(proc, sort_keys=True, default=str), _code_key())
        folder = os.path.join(_root(), geometry)
        return cls(folder, {"geometry": geometry, "loads": loads, "model": model}, study, domain, scale)

    # -- predictions -----------------------------------------------------

    def _margin(self, sf, defl, e) -> float | None:
        """`e`'s reading (sf, max deflection) rescaled to this study."""
        st = self.study
        g = st.goal
        mat = st.material
        k = self.scale / e["loadScale"]
        if st.durability is not None and st.durability.constrain:
            return None
        ms = []
        if g.safety_factor is not None:
            if sf is None:
                return None
            ms.append(sf * (mat.strength / e["strength"]) / k / g.safety_factor)
        if g.max_deflection is not None:
            if defl is None:
                return None
            d = defl * k * (e["E"] / mat.E)
            ms.append(g.max_deflection / d if d > 0 else float("inf"))
        return min(ms) if ms else None

    def predictions(self) -> list[_Pred]:
        out = []
        for e in self.index:
            if e["loads"] != self.keys["loads"]:
                continue
            m = self._margin(e.get("sf"), e.get("defl"), e)
            c = e.get("confirm")
            mc = self._margin(c.get("safetyFactor"), c.get("defl"), e) if c else None
            out.append(_Pred(e, m, mc, e["model"] == self.keys["model"]))
        return out

    def _decisive(self, p: _Pred, ratio: float = 1.0) -> float | None:
        """The margin a pick is judged by: its confirmation when stored, else
        its analysis-grid margin discounted by `ratio`, the worst confirm /
        analysis ratio seen (the bell crank's 0.083 read 1.03 rescaled and
        0.97 on the fine grid)."""
        if p.margin is None:
            return None
        return min(p.margin, p.confirm) if p.confirm is not None else p.margin * ratio

    def _confirm_ratio(self, preds) -> float:
        rs = [p.confirm / p.margin for p in preds if p.confirm is not None and p.margin]
        return min(1.0, min(rs)) if rs else 1.0

    def first_fraction(self, default: float) -> tuple[float, str | None]:
        """Where a safety-factor search should start: the lightest stored
        fraction predicted to pass, or one step past the heaviest that is
        predicted to fail."""
        ps = self.predictions()
        ratio = self._confirm_ratio(ps)
        preds = [(p.entry["vf"], self._decisive(p, ratio), p.exact) for p in ps]
        preds = [q for q in preds if q[1] is not None]
        if not preds:
            return default, None
        passing = [q for q in preds if q[1] >= 1.0]
        failing = [q for q in preds if q[1] < 1.0]
        if passing:
            vf, m, _ = min(passing, key=lambda q: q[0])
            lo = max((q for q in failing if q[0] < vf), key=lambda q: q[0], default=None)
            if m >= 1.12 and lo is not None:
                # the answer lies between: interpolate log-log (margin ~ vf^a)
                a = np.log(m / lo[1]) / np.log(vf / lo[0]) if vf > lo[0] and m > lo[1] > 0 else 1.0
                guess = vf * (1.05 / m) ** (1.0 / max(a, 0.3))
                return float(self.snap(np.clip(guess, lo[0], vf))), f"between recalled {lo[0]:.3f} and {vf:.3f}"
            if m >= 1.12:
                return float(self.snap(vf * 1.05 / m)), f"below recalled {vf:.3f} (margin {m:.2f})"
            return float(vf), f"recalled {vf:.3f} (margin {m:.2f})"
        vf, m, _ = max(failing, key=lambda q: q[0])
        return float(self.snap(min(vf * 1.05 / max(m, 1e-9), vf * 2.0))), f"above recalled {vf:.3f} (margin {m:.2f})"

    def _replayable(self):
        return [p.entry for p in self.predictions() if p.exact]

    def snap(self, vf: float, lo: float | None = None, hi: float | None = None, skip=()) -> float:
        """A replayable stored fraction near `vf` (within _SNAP), or strictly
        inside (lo, hi) when the search is bisecting a bracket."""
        cands = [e["vf"] for e in self._replayable() if all(abs(e["vf"] - v) > 1e-9 for v in skip)]
        if lo is not None and hi is not None:
            inside = [v for v in cands if lo < v < hi]
            if inside:
                return min(inside, key=lambda v: abs(v - vf))
        near = [v for v in cands if abs(v - vf) <= _SNAP * vf]
        return min(near, key=lambda v: abs(v - vf)) if near else vf

    def replay(self, vf: float):
        """A stored outcome at exactly this fraction under the same model:
        (x, xphys, entry), or None."""
        for e in sorted(self._replayable(), key=lambda e: -e["time"]):
            if abs(e["vf"] - vf) <= 1e-9:
                arrays = self._arrays(e)
                if arrays is not None:
                    self.used["replayed"] += 1
                    return arrays[0], arrays[1], e
        return None

    def warm_start(self, vf: float):
        """The stored design under these loads nearest to `vf` (any model):
        (x, label), or None."""
        cands = [e for e in self.index if e["loads"] == self.keys["loads"]]
        if not cands:
            return None
        e = min(cands, key=lambda e: (abs(np.log(e["vf"] / vf)), -e["time"]))
        arrays = self._arrays(e)
        if arrays is None:
            return None
        self.used["warm"] += 1
        return arrays[0], f"warm from recalled {e['vf']:.3f}"

    def _rescaled(self, e) -> dict | None:
        """`e`'s stored fine-grid reading in this study's units, in
        verify_part's shape (the fields study._confirm reads)."""
        c = e.get("confirm")
        if not c or self.study.durability is not None:
            return None
        mat = self.study.material
        k = self.scale / e["loadScale"]
        s = (mat.strength / e["strength"]) / k
        d = k * (e["E"] / mat.E)
        return {"safetyFactor": c["safetyFactor"] * s, "analysisPitch": c["analysisPitch"],
                "analysisCells": c["analysisCells"], "p5Member": c.get("p5Member"),
                "minMember": c.get("minMember"),
                "cases": [{**cs, "peakStress": cs["peakStress"] * k, "deflection": cs["deflection"] * d,
                           **({"safetyFactor": cs["safetyFactor"] * s} if "safetyFactor" in cs else {})}
                          for cs in c["cases"]]}

    def confirm_for(self, vf: float) -> dict | None:
        """The stored fine-grid reading of a replayed outcome, rescaled."""
        for e in self._replayable():
            if abs(e["vf"] - vf) <= 1e-9:
                rep = self._rescaled(e)
                if rep is not None:
                    self.used["confirmsSkipped"] += 1
                    return rep
        return None

    # -- storing -----------------------------------------------------------

    def _arrays(self, e):
        try:
            with np.load(os.path.join(self.folder, e["file"])) as z:
                x, xp = z["x"], z["xphys"]
        except (OSError, KeyError, ValueError):
            return None
        if len(x) != self.domain.n_elem:
            return None
        return x.astype(float), xp.astype(float)

    def add(self, vf: float, x, xphys, report: dict):
        """Remember one outcome (replacing a stored one at the same fraction
        under the same loads and model)."""
        if x is None or xphys is None:
            return
        cases = report.get("cases") or []
        st = self.study
        e = {"vf": float(vf), "loads": self.keys["loads"], "model": self.keys["model"],
             "loadScale": self.scale, "strength": st.material.strength, "E": st.material.E,
             "sf": report.get("safetyFactor"),
             "defl": max((c["deflection"] for c in cases), default=None),
             "mass": report.get("mass"), "material": st.material.name, "time": time.time()}
        old = [o for o in self.index if o["loads"] == e["loads"] and o["model"] == e["model"]
               and abs(o["vf"] - e["vf"]) <= 1e-9]
        if old and old[0].get("confirm") and self._rescaled(old[0]) is not None:
            e["confirm"] = _stored_confirm(self._rescaled(old[0]))
        name = _h(e["loads"], e["model"], round(e["vf"], 12)) + ".npz"
        e["file"] = name
        try:
            os.makedirs(self.folder, exist_ok=True)
            np.savez_compressed(os.path.join(self.folder, name), x=np.asarray(x, np.float32),
                                xphys=np.asarray(xphys, np.float32))
        except OSError:
            return
        self.index = [o for o in self.index if o not in old] + [e]
        self.dirty = True

    def confirmed(self, vf: float, rep: dict):
        """Attach a fine-grid reading (verify_part's report, in this study's
        units) to the outcome stored at `vf` by this run."""
        for e in self.index:
            if e["loads"] == self.keys["loads"] and e["model"] == self.keys["model"] \
                    and abs(e["vf"] - vf) <= 1e-9 and e["strength"] == self.study.material.strength \
                    and e["E"] == self.study.material.E and e["loadScale"] == self.scale:
                e["confirm"] = _stored_confirm(rep)
                self.dirty = True
                return

    def save(self):
        if not self.dirty:
            return
        try:
            # merge what other processes stored meanwhile (outcomes run in parallel)
            with open(os.path.join(self.folder, "index.json")) as fh:
                theirs = json.load(fh).get("entries", [])
        except (OSError, ValueError):
            theirs = []
        mine = {e["file"] for e in self.index}
        self.index += [e for e in theirs if e["file"] not in mine]
        keep = sorted(self.index, key=lambda e: -e["time"])[:_MAX_ENTRIES]
        live = {e["file"] for e in keep}
        for e in self.index:
            if e["file"] not in live:
                try:
                    os.remove(os.path.join(self.folder, e["file"]))
                except OSError:
                    pass
        self.index = keep
        try:
            os.makedirs(self.folder, exist_ok=True)
            tmp = os.path.join(self.folder, f"index.json.{os.getpid()}")
            with open(tmp, "w") as fh:
                json.dump({"entries": keep, "study": self.study.name}, fh)
            os.replace(tmp, os.path.join(self.folder, "index.json"))
        except OSError:
            return
        self.dirty = False
        _prune(os.path.dirname(self.folder), keep=self.folder)


def _stored_confirm(rep: dict) -> dict:
    return {"safetyFactor": rep["safetyFactor"],
            "defl": max((c["deflection"] for c in rep["cases"]), default=None),
            "analysisPitch": rep.get("analysisPitch"), "analysisCells": rep.get("analysisCells"),
            "p5Member": rep.get("p5Member"), "minMember": rep.get("minMember"),
            "cases": [{k: c[k] for k in ("name", "peakStress", "peakAt", "deflection", "safetyFactor") if k in c}
                      for c in rep["cases"]]}


def _prune(root: str, keep: str):
    cap = float(os.environ.get("LOADPATH_RECALL_MB", "1000")) * 1e6
    try:
        dirs = [os.path.join(root, d) for d in os.listdir(root)]
    except OSError:
        return
    sizes = []
    for d in dirs:
        try:
            files = [os.path.join(d, f) for f in os.listdir(d)]
            sizes.append((max(os.path.getmtime(f) for f in files) if files else 0.0,
                          sum(os.path.getsize(f) for f in files), d))
        except OSError:
            continue
    total = sum(s for _, s, _ in sizes)
    for _, s, d in sorted(sizes):
        if total <= cap:
            break
        if d != keep:
            shutil.rmtree(d, ignore_errors=True)
            total -= s
