"""
Split shells: a fixed exterior cut into a base and a cover that go together
round the components, with the joints that hold them, all inside the skin.

A product's exterior is drawn whole; it is built in two parts. Where the
cut goes is a designer's call as much as an engineer's: the seam shows, so
it is short and away from the surfaces the designer cares about; each half
has to pull off the other without hooking it (no undercut across the seam);
every component has to go into its half with the other half off, and the
cover has to come down over the components already in place. `apply_split`
checks all of that for a given cut, or searches: cuts across each axis,
then the non-flat ones parting.py proposes from the best of them (tilted
planes, steps over a part, the silhouette, a crease in the exterior),
ranked by the seam that shows, how far parts travel in, parts cut
through, printability and the joints that fit. A cut need not be a flat
axis plane:
it can be tilted (`normal` a vector), step up over part of the product
(`steps`), or follow a parting line the designer drew on the exterior
(`seam`). Any surface that is a height over the plane across the lift
direction parts cleanly along it, so that is the general cut: the seam
line is kept where it was drawn, and inside, where nobody sees it, the
surface runs smoothly between.

Then the joints, every one inside the skin (the exterior is never changed;
a screw needs a hole, so screws come in only through a face the designer
names, `screws.from`, and each hole is reported):

- **an alignment lip**: a tongue on the base, standing up inside the
  cover's wall, a clearance off it on both sides. It locates the halves
  across the seam and hides the light gap. Broken where a screw post or a
  snap sits;
- **screw posts** (`screws`): a clearance bore up through the base from the
  named face, the head in a counterbore inside the exterior, and a thread
  (tapped, or a heat-set insert) in the cover. Each is a fasteners.py entry
  in each half, so the optimizer grows the compression cone between the
  rings instead of inheriting a solid boss, and the bore's fit is drawn for
  the process (fits.py);
- **cantilever snaps** (`snaps`): a beam hanging from the cover with a barb
  (lead-in face, square return) that springs under a ledge on the base's
  inner wall. The beam is sized so the strain while the barb passes the
  ledge stays within the material's allowance (`snap_allowable`), and the
  room it bends into is kept clear in the base. `verify_snap` solves the
  hook as drawn on cut cells and reads its root strain.

Out: a study per half (ordinary studies: the half's exterior with its skin
open on the cut face, the joints as preserve and keep-out regions, the
components as keep-outs whose assembly path leaves through the cut), and a
report: the cut and why, the paths, each joint with its numbers, and the
holes the exterior gets (only where the designer allowed them).

    "split": {
      "normal": "+z",                  # the cover comes off toward +z; "auto" (or no normal) tries x, y, z
      "at": 14.0,                      # the cut's level along that axis; omitted: searched
      "steps": [{"at": 18, "where": shape}],  # or: the cut climbs to 18 over `where` (needs normal and at)
      "seam": {"points": [[x, y, z], ...]},   # or: through a line drawn on the exterior (normal: its plane)
      "avoid": [shape, ...],           # surfaces the seam must not cross (the designer's A surfaces)
      "componentsIn": {"battery": "base"},   # which half holds a component (default: where most of it is)
      "clearance": 0.2,                # mm between mating faces (lip, snaps)
      "lip": {"height": 2.0, "width": 1.2} | false,
      "screws": {"from": "-z", "size": "M2", "count": 4, "insert": 3.2, "fit": "normal"} | false,
      "snaps": {"count": "auto", "width": 6, "thickness": 1.2, "undercut": 0.8, "leadAngle": 30,
                "repeated": false, "pull": 10} | false,
      "margin": 0.5                    # mm from components to any joint
    }
"""
from __future__ import annotations

import copy
import json
import math
import os
import time

import numpy as np

__all__ = [
    "apply_split",
    "check_closed",
    "run_split",
    "seam_length",
    "snap_allowable",
    "snap_cantilever",
    "verify_snap",
]

_AX = {"x": 0, "y": 1, "z": 2}
# mm a component may slide before it drops in, while the search checks
# cuts (a cut given is checked to handoff's 12 mm): a part with no way in
# costs a full slide-and-drop search, minutes on a mesh product
SEARCH_SLIDE = 4.0


def _axis(side):
    s = str(side).strip().lower()
    key = s.lstrip("+-")
    if key not in _AX:
        raise ValueError(f"split.normal is an axis like '+z' or 'auto', got {side!r}")
    return _AX[key], (-1 if s.startswith("-") else 1)


def _name(a: int, sg: int) -> str:
    return ("+" if sg > 0 else "-") + "xyz"[a]


def _unit(a: int, sg: int) -> np.ndarray:
    v = np.zeros(3)
    v[a] = float(sg)
    return v


def _r(v, k=4):
    return np.round(np.asarray(v, float), k).tolist()


# ---------------------------------------------------------------------------
# materials: how far a snap beam may bend


def snap_allowable(material, process: str | None = None, across_layers: bool = False,
                   repeated: bool = False) -> dict:
    """The strain a snap beam may reach as its barb passes the ledge.

    House rule, not a measured value: 60% of the published strain at
    yield, or 30% of the strain at break where only that is published (a
    polymer's break strain is far past where it stops springing back),
    capped at 5% (unfilled nylon's permissible short-term strain in the
    resin makers' snap-fit guides; printed parts are not better than
    moulded). Metals: 60% of yield strain. Across FDM layers (a beam whose
    length runs along the build direction) the strain scales by the
    process's interlayer factor, or the card's own Z elongation where it
    gives one. A snap opened and closed again and again: 60% of that.

    Returns {strain, basis}."""
    from .material_library import LIBRARY
    from .materials import material_from_spec
    from .spec import PROCESSES

    mat = material_from_spec(material[0] if isinstance(material, list) else material)
    name = str(getattr(mat, "name", "")).lower()
    try:
        from .calibration import load_card
        cal = load_card(name) if name else None
    except Exception:  # noqa: BLE001
        cal = None
    same = cal and (not process or (cal.get("process") or {}).get("type") in (None, process))
    measured = _measured_snap(cal, across_layers, repeated) if same else None
    if measured is not None:
        return {**measured, "material": name, "E": float(mat.E), "strength": float(mat.strength)}
    if cal and cal.get("basedOn"):
        name = str(cal["basedOn"]).lower()
    card = LIBRARY.get(name) or {}
    el = (card.get("values") or {}).get("elongation")
    metal = float(mat.density) > 2.0
    basis = []
    if metal or el is None:
        eps = 0.6 * float(mat.strength) / float(mat.E)
        basis.append(f"60% of strength/E ({100 * mat.strength / mat.E:.2f}%)")
    else:
        val, _src, note = float(el[0]), el[1], str(el[2])
        z = None
        if across_layers and "Z:" in note:
            try:
                z = float(note.split("Z:")[1].split(")")[0].strip().split()[0])
            except ValueError:
                z = None
        if z is not None:
            val = z
            basis.append(f"across layers: the card's Z elongation {z:g}%")
        frac = 0.6 if "yield" in note else 0.3
        eps = frac * val / 100.0
        basis.append(f"{int(frac * 100)}% of the card's elongation ({val:g}% {'at yield' if frac == 0.6 else 'at break'})")
        cap = 0.05
        if eps > cap:
            eps = cap
            basis.append("capped at 5%")
        if across_layers and z is None and process in PROCESSES:
            f = float(PROCESSES[process]["interlayer"])
            if f < 1.0:
                eps *= f
                basis.append(f"x{f:g} across layers ({process})")
    if repeated:
        eps *= 0.6
        basis.append("x0.6: opened and closed repeatedly")
    return {"strain": float(eps), "basis": "; ".join(basis), "material": name, "E": float(mat.E),
            "strength": float(mat.strength)}


def _measured_snap(card: dict, across_layers: bool, repeated: bool) -> dict | None:
    """The allowance a calibration card measured on its S-XY / S-Z snap
    ladders (calibration.fit_snaps), or None when it has none for this
    case. Across layers without an S-Z reading, S-XY's scales by the
    card's interlayer factor; repeated without a ten-closing reading,
    x0.6 as in the house rule."""
    sn = card.get("snaps") or {}
    key = "Across" if across_layers else ""
    eps = sn.get(f"strain{key}")
    basis = [f"measured on the {'S-Z' if across_layers else 'S-XY'} snap ladder ({card.get('name')})"]
    if eps is None and across_layers and sn.get("strain") is not None:
        f = float((card.get("process") or {}).get("interlayer", 1.0))
        eps = float(sn["strain"]) * f
        basis = [f"measured on the S-XY snap ladder ({card.get('name')}), x{f:g} across layers (card interlayer)"]
    if eps is None:
        return None
    eps = float(eps)
    if repeated:
        r = sn.get(f"strainRepeated{key}")
        if r is not None:
            eps = min(eps, float(r))
            basis.append("over ten closings")
        else:
            eps *= 0.6
            basis.append("x0.6: opened and closed repeatedly (no ten-closing reading)")
    out = {"strain": eps, "basis": "; ".join(basis), "measured": True}
    if sn.get("pushFactor"):
        out["pushFactor"] = float(sn["pushFactor"])
    return out


def joint_tolerance(holes: dict | None, build, n, size: float) -> dict:
    """How far a printed joint face may sit from where it is drawn, for a
    process (fits.HOLES, or a calibrated card's numbers through
    fits._process_of): `face` (+- mm, half the process's size band at the
    part's size: what each face of a lip or snap may be off by), `bias`
    (how much a drawn gap closes: printed walls grow into gaps as they do
    into holes) and `repeat` (the share of the band a tuned printer
    holds). A gap's faces run along n, so the hole bias that applies is
    the one for holes along n: along the build or across it."""
    if not holes:
        return {"face": 0.0, "bias": 0.0, "repeat": 1.0, "basis": "no process: tolerances not checked"}
    n = np.asarray(n, float)
    across = build is not None and abs(float(np.asarray(build, float) @ n)) < 0.7
    bias = float(holes["biasAcross"] if across else holes["bias"])
    e = 0.5 * (float(holes["tol"]) + float(holes.get("tolPerMm", 0.0)) * float(size))
    return {"face": e, "bias": abs(bias), "repeat": float(holes.get("repeat", 1.0)),
            "across": bool(across), "calibrated": bool(holes.get("calibrated")),
            "basis": (f"+-{e:.3f} mm a face (half the process band at {size:.0f} mm), gaps drawn "
                      f"{abs(bias):.2f} mm wider for the {'across-layer' if across else 'along-build'} bias"
                      + (" (calibrated)" if holes.get("calibrated") else " (design-guide numbers, fits.HOLES)"))}


def _stack(nominal: float, parts: dict) -> dict:
    """A gap or an overlap and what moves it: worst case (every
    contributor at its limit) and root sum square."""
    vals = [abs(float(v)) for v in parts.values()]
    worst, rss = float(sum(vals)), float(math.sqrt(sum(v * v for v in vals)))
    return {"nominal": round(nominal, 3), "worst": round(nominal - worst, 3), "rss": round(nominal - rss, 3),
            "contributors": {k: round(abs(float(v)), 3) for k, v in parts.items()}}


def _verdict(st: dict, repeat: float) -> str:
    if st["worst"] >= 0:
        return "holds at worst case"
    if st["rss"] >= 0:
        tuned = st["nominal"] - repeat * (st["nominal"] - st["worst"])
        return ("holds by RSS; at worst case it needs a tuned printer" if tuned >= 0
                else "holds by RSS only: worst case closes it even on a tuned printer")
    return "fails by RSS: widen it"


def snap_cantilever(L: float, t: float, b: float, y: float, E: float, lead_deg: float = 30.0,
                    return_deg: float = 90.0, mu: float = 0.3) -> dict:
    """A constant-section rectangular cantilever snap (the resin makers'
    snap-fit guides): the strain at its root when its tip moves y, the
    force that takes, and the push to assemble it and pull to part it.

        strain     e = 1.5 t y / L^2
        deflection P = b t^2 E e / (6 L)
        push       W = P (mu + tan a) / (1 - mu tan a)

    A return face at 90 degrees does not let go: W runs to infinity and
    `pull` is None (it holds until the ledge or the hook breaks)."""
    eps = 1.5 * t * y / L ** 2
    P = b * t ** 2 * E * eps / (6.0 * L)

    def mating(a_deg):
        ta = math.tan(math.radians(a_deg))
        den = 1.0 - mu * ta
        return None if den <= 1e-6 or a_deg >= 89.0 else P * (mu + ta) / den

    return {"strain": eps, "deflectionForce": P, "push": mating(lead_deg), "pull": mating(return_deg)}


# ---------------------------------------------------------------------------
# geometry helpers


def _exterior(study: dict, base_dir):
    from .shapes import MeshShape, Union, shape_from_spec

    ds = study.get("designSpace")
    specs = ds if isinstance(ds, list) else [ds]
    shapes = []
    for s in specs:
        sh = shape_from_spec(s, base_dir)
        if isinstance(sh, MeshShape):
            # joints fit to a clearance of 0.2 mm: a finer distance grid than
            # the study default (1/160 of the part, ~0.5 mm on a remote)
            sh._GRID_CELLS = 480
        shapes.append(sh)
    return (shapes[0] if len(shapes) == 1 else Union(shapes)), specs


class _Cut:
    """The parting surface. The cover is where n.x > h(u, v), h a height
    over the plane (u, v) = (e1.x, e2.x) across the lift direction n: the
    cover lifts off along n, and any surface that is a height over that
    plane parts along it. A plane has one height (an axis cut, or an
    angled one); a stepped parting line and one that follows a seam the
    designer drew carry h on a grid."""

    def __init__(self, n, h0=0.0, grid=None, e1=None, kind="plane", axis=None, at=None):
        n = np.asarray(n, float)
        self.n = n / np.linalg.norm(n)
        self.axis = axis
        if axis is not None:
            b, c = [i for i in range(3) if i != axis[0]]
            self.e1, self.e2 = _unit(b, 1), _unit(c, 1)
        else:
            if e1 is None:
                k = int(np.argmin(np.abs(self.n)))
                e1 = np.cross(self.n, _unit(k, 1))
            e1 = np.asarray(e1, float) - (np.asarray(e1, float) @ self.n) * self.n
            self.e1 = e1 / np.linalg.norm(e1)
            self.e2 = np.cross(self.n, self.e1)
        self.h0 = float(h0)
        self.grid = grid                   # (origin (u0, v0), step, H) or None
        self.kind = kind
        self.at = at

    @classmethod
    def plane_axis(cls, a, sg, at):
        return cls(_unit(a, sg), sg * float(at), axis=(a, sg), at=round(float(at), 3))

    @classmethod
    def plane(cls, n, at):
        n = np.asarray(n, float)
        n = n / np.linalg.norm(n)
        k = int(np.argmax(np.abs(n)))
        if abs(n[k]) > 1 - 1e-9:
            sg = 1 if n[k] > 0 else -1
            return cls.plane_axis(k, sg, sg * float(at))
        return cls(n, float(at), at=round(float(at), 3))

    @property
    def planar(self):
        return self.grid is None

    def flipped(self):
        g = None
        if self.grid is not None:
            o, st, H = self.grid
            g = (o, st, -H)
        c = _Cut(-self.n, -self.h0, g, e1=self.e1, kind=self.kind,
                 axis=None if self.axis is None else (self.axis[0], -self.axis[1]), at=self.at)
        c.e1, c.e2 = self.e1, self.e2
        for k in ("steps", "tilt", "seam_points"):
            if hasattr(self, k):
                setattr(c, k, getattr(self, k))
        return c

    def uv(self, P):
        P = np.asarray(P, float)
        return P @ self.e1, P @ self.e2

    def h(self, P):
        P = np.asarray(P, float)
        if self.grid is None:
            return np.full(P.shape[:-1], self.h0)
        o, st, H = self.grid
        u, v = self.uv(P)
        fi = np.clip((u - o[0]) / st, 0.0, H.shape[0] - 1.000001)
        fj = np.clip((v - o[1]) / st, 0.0, H.shape[1] - 1.000001)
        i, j = np.floor(fi).astype(int), np.floor(fj).astype(int)
        a, b = fi - i, fj - j
        return ((1 - a) * (1 - b) * H[i, j] + a * (1 - b) * H[i + 1, j] + (1 - a) * b * H[i, j + 1]
                + a * b * H[i + 1, j + 1])

    def phi(self, P):
        """Height above the surface along n: > 0 on the cover's side."""
        P = np.asarray(P, float)
        return P @ self.n - self.h(P)

    def point(self, u, v, S=0.0):
        """The world point at (u, v), S below the surface along -n."""
        u, v, S = np.broadcast_arrays(np.asarray(u, float), np.asarray(v, float), np.asarray(S, float))
        base = u[..., None] * self.e1 + v[..., None] * self.e2
        return base + (self.h(base) - S)[..., None] * self.n

    def name(self, sign=1):
        """The direction sign * n: '+z' for an axis, else a vector."""
        if self.axis is not None:
            return _name(self.axis[0], sign * self.axis[1])
        return _r(sign * self.n, 6)

    def label(self):
        if self.axis is not None and self.planar:
            return f"{'xyz'[self.axis[0]]} = {self.at:g}"
        if self.planar:
            return f"the plane n.x = {self.at:g}, n = {_r(self.n, 3)}"
        return {"steps": "a stepped parting line", "silhouette": "the silhouette (the widest line round the part)",
                "crease": "a crease in the exterior"}.get(self.kind, "the seam line drawn on the exterior")

    def footprint(self, ext, pad=1.0):
        lo, hi = ext.bounds()
        C = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
        u, v = C @ self.e1, C @ self.e2
        w = C @ self.n
        return (u.min() - pad, u.max() + pad), (v.min() - pad, v.max() + pad), (w.min(), w.max())

    def half_spec(self, ext, side, offset=0.0, pad=5.0):
        """The half's region as a shape spec: below the surface (base) or
        above it (cover), `offset` mm along n."""
        lo, hi = ext.bounds()
        lo, hi = lo - pad, hi + pad
        if self.axis is not None and self.planar:
            a, sg = self.axis
            at = self.at + sg * offset
            if (side == "cover") == (sg > 0):
                lo[a] = at
            else:
                hi[a] = at
            return {"type": "box", "min": _r(lo), "max": _r(hi)}
        (u0, u1), (v0, v1), (w0, w1) = self.footprint(ext, pad)
        R = float(np.linalg.norm(hi - lo)) + pad
        if self.planar:
            c = 0.5 * (u0 + u1) * self.e1 + 0.5 * (v0 + v1) * self.e2 + (self.h0 + offset) * self.n
            c = c + (0.5 * R if side == "cover" else -0.5 * R) * self.n
            return {"type": "box", "center": _r(c), "size": _r([u1 - u0, v1 - v0, R]),
                    "axes": [_r(self.e1, 9), _r(self.e2, 9)]}
        o, st, H = self.grid
        return {"type": "heightfield", "normal": _r(self.n, 9), "axes": [_r(self.e1, 9), _r(self.e2, 9)],
                "origin": _r(o), "step": st, "heights": np.round(H, 4).tolist(),
                "side": "above" if side == "cover" else "below", "offset": float(offset), "depth": round(R, 2)}

    def band_spec(self, ext, side, width):
        """A slice `width` mm thick against the surface on one side: where
        a cover sits on the base's rim."""
        if side == "cover":
            return {"type": "intersect", "shapes": [self.half_spec(ext, "cover"),
                                                    self.half_spec(ext, "base", offset=width)]}
        return {"type": "intersect", "shapes": [self.half_spec(ext, "base"),
                                                self.half_spec(ext, "cover", offset=-width)]}

    def report(self):
        out = {"normal": self.name(), "kind": self.kind}
        if self.planar:
            out["at"] = self.at
        else:
            o, st, H = self.grid
            out["levels"] = [round(float(H.min()), 3), round(float(H.max()), 3)]
        if getattr(self, "tilt", None):
            out["tilt"] = self.tilt
        if getattr(self, "steps", None):
            out["steps"] = [{k: v for k, v in s.items() if k != "where"} for s in self.steps]
        return out


def _smoothstep(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3 - 2 * x)


def _grid_for(cut, ext, step=None):
    (u0, u1), (v0, v1), _ = cut.footprint(ext, pad=2.0)
    st = float(step or max(0.5, max(u1 - u0, v1 - v0) / 400.0))
    nu, nv = int(np.ceil((u1 - u0) / st)) + 1, int(np.ceil((v1 - v0) / st)) + 1
    return np.array([u0, v0]), st, nu, nv


def _stepped_cut(ext, n_spec, at, steps, width, base_dir):
    """A plane at `at` that climbs to each step's level over its `where`."""
    from scipy import ndimage

    from .shapes import shape_from_spec

    base = _cut_plane(n_spec, at)
    o, st, nu, nv = _grid_for(base, ext)
    U = o[0] + np.arange(nu) * st
    V = o[1] + np.arange(nv) * st
    H = np.full((nu, nv), base.h0)
    _, _, (w0, w1) = base.footprint(ext, 0.0)
    S = np.arange(w0, w1 + 0.5, 0.5)
    uu, vv = np.meshgrid(U, V, indexing="ij")
    for k, stp in enumerate(steps):
        sh = shape_from_spec(stp["where"], base_dir)
        lvl = _cut_plane(n_spec, float(stp["at"])).h0
        mask = np.zeros((nu, nv), bool)
        for s in S:
            P = uu[..., None] * base.e1 + vv[..., None] * base.e2 + s * base.n
            mask |= (sh.sdf(P.reshape(-1, 3)) <= 0).reshape(nu, nv)
        if not mask.any():
            raise ValueError(f"split.steps[{k}].where does not cross the part along the normal")
        d = (ndimage.distance_transform_edt(mask) - ndimage.distance_transform_edt(~mask)) * st
        wgt = _smoothstep(d / max(width, st) + 0.5)
        H = H + wgt * (lvl - H)
    return _Cut(base.n, base.h0, (o, st, H), kind="steps", axis=base.axis, at=base.at)


def _seam_cut(ext, n_spec, points):
    """The surface through a parting line drawn on the exterior: h at the
    line's points, smooth (harmonic) in between and past it."""
    from scipy.sparse import coo_matrix, identity
    from scipy.sparse.linalg import spsolve

    P = np.asarray(points, float)
    if P.ndim != 2 or P.shape[1] != 3 or len(P) < 3:
        raise ValueError("split.seam.points is a list of [x, y, z] round the part, at least 3")
    if n_spec in (None, "auto"):
        c = P - P.mean(axis=0)
        n = np.linalg.svd(c, full_matrices=False)[2][-1]
        n = n if n[int(np.argmax(np.abs(n)))] > 0 else -n
        k0 = int(np.argmax(np.abs(n)))
        if abs(n[k0]) > math.cos(math.radians(15.0)):
            # within 15 degrees of an axis: lift straight along it (a line
            # that rises toward one end is not a reason to tilt the tool)
            n = _unit(k0, 1)
    else:
        n = _vec_of(n_spec)
    k = int(np.argmax(np.abs(n)))
    axis = (k, 1 if n[k] > 0 else -1) if abs(n[k]) > 1 - 1e-9 else None
    cut = _Cut(n, 0.0, axis=axis)
    o, st, nu, nv = _grid_for(cut, ext)
    # the line, closed, every quarter cell
    Q = np.vstack([P, P[:1]])
    q, _, _ = _resample(Q, 0.25 * st)
    u, v = cut.uv(q)
    i = np.clip(np.rint((u - o[0]) / st).astype(int), 0, nu - 1)
    j = np.clip(np.rint((v - o[1]) / st).astype(int), 0, nv - 1)
    fix = {}
    for ii, jj, hh in zip(i, j, q @ cut.n):
        fix.setdefault((ii, jj), []).append(hh)
    N = nu * nv
    idx = np.arange(N).reshape(nu, nv)
    rows, cols, vals = [], [], []
    for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        a = idx[max(0, -di):nu - max(0, di), max(0, -dj):nv - max(0, dj)].ravel()
        b = idx[max(0, di):nu + min(0, di), max(0, dj):nv + min(0, dj)].ravel()
        rows += [a, a]
        cols += [b, a]
        vals += [-np.ones(len(a)), np.ones(len(a))]
    L = coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(N, N)).tocsr()
    rhs = np.zeros(N)
    fixed = np.zeros(N, bool)
    for (ii, jj), hs in fix.items():
        fixed[idx[ii, jj]] = True
        rhs[idx[ii, jj]] = float(np.mean(hs))
    D = identity(N, format="csr").multiply(fixed[:, None]).tocsr()
    A = L.multiply((~fixed)[:, None]).tocsr() + D
    H = spsolve(A.tocsc(), rhs).reshape(nu, nv)
    c = _Cut(n, 0.0, (o, st, H), kind="seam", axis=axis)
    c.e1, c.e2 = cut.e1, cut.e2
    c.seam_points = P
    return c


def _vec_of(spec):
    if isinstance(spec, str):
        a, sg = _axis(spec)
        return _unit(a, sg)
    v = np.asarray(spec, float)
    if v.shape != (3,) or not np.linalg.norm(v) > 0:
        raise ValueError(f"split.normal is an axis like '+z', a vector or 'auto', got {spec!r}")
    return v / np.linalg.norm(v)


def _cut_plane(n_spec, at):
    if isinstance(n_spec, str):
        a, sg = _axis(n_spec)
        return _Cut.plane_axis(a, sg, at)
    return _Cut.plane(_vec_of(n_spec), at)


def _inside(ext, step=1.5):
    """Points inside the exterior on a grid (cached on it): how deep each
    half runs along a cut's normal."""
    got = getattr(ext, "_lp_inside", None)
    if got is not None:
        return got
    lo, hi = ext.bounds()
    st = max(step, float(np.max(hi - lo)) / 80.0)
    ax = [np.arange(lo[i] + 0.5 * st, hi[i], st) for i in range(3)]
    G = np.stack(np.meshgrid(*ax, indexing="ij"), -1).reshape(-1, 3)
    P = G[ext.sdf(G) < 0]
    try:
        ext._lp_inside = P
    except AttributeError:
        pass
    return P


def _depths(ext, cut):
    if cut.axis is not None and cut.planar:
        lo, hi = ext.bounds()
        a, sg = cut.axis
        lo_n, hi_n = (lo[a], hi[a]) if sg > 0 else (-hi[a], -lo[a])
        atn = sg * cut.at
        return atn - lo_n, hi_n - atn
    f = cut.phi(_inside(ext))
    return float(max(0.0, -f.min())), float(max(0.0, f.max()))


def _contours(ext, cut, level: float, step: float = 0.2) -> list[np.ndarray]:
    """Closed curves where the exterior's signed distance is `level` on the
    cut surface, as (n, 3) arrays."""
    from skimage import measure

    (u0, u1), (v0, v1), _ = cut.footprint(ext, pad=1.0)
    ub = np.arange(u0, u1 + step, step)
    uc = np.arange(v0, v1 + step, step)
    UU, VV = np.meshgrid(ub, uc, indexing="ij")
    G = cut.point(UU, VV)
    F = ext.sdf(G.reshape(-1, 3)).reshape(len(ub), len(uc))
    out = []
    for cnt in measure.find_contours(F, level):
        if len(cnt) < 8:
            continue
        out.append(cut.point(ub[0] + cnt[:, 0] * step, uc[0] + cnt[:, 1] * step))
    return out


def _length(p: np.ndarray) -> float:
    return float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum())


def seam_length(ext, a, at=None) -> float:
    """mm of seam: the exterior's outline on the cut (an axis and a level,
    or a cut surface)."""
    cut = a if isinstance(a, _Cut) else _Cut.plane_axis(int(a), 1, float(at))
    return float(sum(_length(c) for c in _contours(ext, cut, 0.0)))


def _resample(p: np.ndarray, ds: float):
    seg = np.linalg.norm(np.diff(p, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    n = max(int(s[-1] / ds), 8)
    t = np.linspace(0.0, s[-1], n, endpoint=False)
    q = np.stack([np.interp(t, s, p[:, i]) for i in range(3)], axis=1)
    return q, t, float(s[-1])


def _grad(ext, p, h=0.1):
    g = np.zeros_like(p)
    for i in range(3):
        e = np.zeros(3)
        e[i] = h
        g[:, i] = (ext.sdf(p + e) - ext.sdf(p - e)) / (2 * h)
    return g


def _half_skin(ext, skin: dict, cut, side, base_dir=None):
    """(half design space, the half's skin shape, the skin dict, the half
    region): the skin lines the whole exterior and stops at the cut."""
    from .shapes import Boolean, shape_from_spec
    from .spec import _skin

    region = shape_from_spec(cut.half_spec(ext, side))
    ds = Boolean("intersect", [ext, region])
    sk = dict(skin)
    sk.pop("of", None)
    _t, shape = _skin(sk, [ext], base_dir)
    shape = Boolean("intersect", [shape, region])
    return ds, shape, sk, region


def _samples(shape, step, cap=6000):
    from .handoff import _samples as hs
    return hs(shape, step=step, cap=cap)


# ---------------------------------------------------------------------------
# the cut


def _assign(comps, cut, given):
    """Which half holds each component: given, or where most of it is."""
    out = {}
    for c in comps:
        if c["id"] in given:
            out[c["id"]] = given[c["id"]]
            continue
        P = _samples(c["shape"], 0.6, 3000)
        up = float(np.mean(cut.phi(P) > 0))
        out[c["id"]] = "cover" if up > 0.5 else "base"
    for c in comps:      # a part installed on another goes with it
        p = c.get("installedOn")
        if p and p in out and c["id"] not in given:
            out[c["id"]] = out[p]
    return out


def _check_cut(ext, skin, t, comps, cut, given, tol, avoid, order, base_dir=None, max_slide=12.0):
    """Everything about one cut: seam, separation, assembly in each half."""
    from .handoff import _Obstacles, _path_check, assembly_paths

    n = cut.n
    lo, hi = ext.bounds()
    rec = {**cut.report(), "_cut": cut}
    curves = _contours(ext, cut, 0.0)
    rec["seam"] = round(float(sum(_length(c) for c in curves)), 1)
    if not curves:
        rec.update(ok=False, why="the cut misses the exterior")
        return rec
    if avoid:
        P = np.vstack(curves)
        hit = [i for i, s in enumerate(avoid) if (s.sdf(P) <= 0.3).any()]
        if hit:
            rec.update(ok=False, why=f"the seam crosses avoid[{hit}]")
            return rec
    where = _assign(comps, cut, given)
    rec["componentsIn"] = where
    halves = {}
    for side in ("base", "cover"):
        halves[side] = _half_skin(ext, skin, cut, side, base_dir)
    # Two halves of one solid cut by a surface that is a height over the
    # plane across n always part along n; what can hook is a component. The
    # cover lifts off along +n with its components: they must not catch on
    # the base's components, and the cover's skin must not catch on a base
    # component standing up past the cut (a connector tucked under the
    # cover's wall).
    base_c = [c for c in comps if where[c["id"]] == "base"]
    cov_c = [c for c in comps if where[c["id"]] == "cover"]
    mov = [_samples(c["shape"], 0.5, 3000) for c in cov_c]
    tall = []
    for c in base_c:
        if (cut.phi(_samples(c["shape"], 0.8, 1500)) > 0).any():
            tall.append(c)
    if tall:
        sk_pts = _samples(halves["cover"][1], min(0.5, 0.6 * t), 20000)
        su, sv = cut.uv(sk_pts)
        keep = np.zeros(len(sk_pts), bool)
        for c in tall:
            cu, cv = cut.uv(_samples(c["shape"], 0.8, 1500))
            keep |= ((su > cu.min() - 1.0) & (su < cu.max() + 1.0) & (sv > cv.min() - 1.0) & (sv < cv.max() + 1.0))
        if keep.any():
            mov.append(sk_pts[keep])
    ok = True
    if mov and base_c:
        obs = _Obstacles([c["shape"] for c in base_c], [c["id"] for c in base_c])
        ok, pen, who, at_pt, dist = _path_check(np.vstack(mov), obs, [(n, None)], lo - 1, hi + 1, tol, step=0.5)
    rec["separates"] = bool(ok)
    if not ok:
        rec.update(ok=False, why=f"the cover catches on {who} lifting off along {cut.name()} "
                                 f"({pen:.2f} mm deep at {_r(at_pt, 2)}, after {dist:.1f} mm)")
        return rec
    rec["paths"] = {}
    for side in ("base", "cover"):
        mine = [c for c in comps if where[c["id"]] == side]
        if not mine:
            continue
        hint = cut.name(1 if side == "base" else -1)       # in through the cut
        cl = [{**c, "insertFrom": c.get("insertFrom") or hint} for c in mine]
        ids = [c["id"] for c in mine]
        od = [i for i in order if i in ids] + [i for i in ids if i not in order]
        paths = assembly_paths(cl, od, [(f"{side} skin", halves[side][1])], (lo, hi), tol=tol,
                               max_slide=max_slide, hint_exit=max_slide < 12.0)
        for k, v in paths.items():
            rec["paths"][k] = {"half": side, "insertFrom": v.get("insertFrom"), "how": v.get("how"),
                               **({"blocked": v["blocked"]} if v.get("insertFrom") is None else {})}
    blocked = [k for k, v in rec["paths"].items() if v["insertFrom"] is None]
    if blocked:
        rec.update(ok=False, why=f"no way in for {blocked} with the other half off"
                   + (f" (in the search, a slide of up to {max_slide:g} mm, then out through the cut, is "
                      "tried; give the cut to check every path)" if max_slide < 12.0 else ""))
        return rec
    db, dc = _depths(ext, cut)
    rec["depth"] = {"base": round(db, 2), "cover": round(dc, 2)}
    rec["ok"] = True
    return rec


def _search(ext, skin, t, comps, given, tol, avoid, order, normal=None, levels=None, min_depth=6.0,
            curved=True, pkind=None, hidden=None, avoid_specs=(), base_dir=None, max_checks=12):
    """Cuts across x, y and z at a spread of levels, and (curved) the
    non-flat ones parting.py proposes from the best of them: tilted planes,
    stepped lines, the silhouette and creases. Each is scored (`score`),
    the best are checked (_check_cut) until four pass, and the passing ones
    come back ranked; apply_split then places the joints on the best four
    and takes the one that holds best."""
    lo, hi = ext.bounds()
    if normal in (None, "auto"):
        dirs = [_unit(a, 1) for a in range(3)]
    else:
        dirs = [_vec_of(normal)]
    cands = []
    for d in dirs:
        base = _Cut.plane(d, 0.0)
        _, _, (w0, w1) = base.footprint(ext, 0.0)
        span = w1 - w0
        if levels is not None:
            lv = list(levels)
        else:
            m = min(min_depth, 0.3 * span)
            k = int(np.clip(round(span / 3.0), 5, 12))
            lv = [base.h0 + x for x in np.linspace(w0 + m, w1 - m, k)]
            if base.axis is not None:
                lv = [base.axis[1] * x for x in lv]          # levels are coordinates on an axis
        for at in lv:
            c = _cut_plane(base.name() if base.axis is not None else d, round(float(at), 3))
            cands.append(c)
    samples = {c["id"]: _samples(c["shape"], 0.8, 600) for c in comps}

    def score(cut):
        """mm, lower is better: the seam's length where it shows (a quarter
        of it where it runs in a crease or a `hidden` region), plus four
        times how far each component travels to go in (a battery dropped
        8 mm into a tray beats one slid 60 mm down a tube), plus 20 for each
        component standing across the cut, 10 per mm a half is too shallow
        for a joint, and for FDM half the seam that does not lie on one
        plane (the half cannot stand on its rim to print). A rule of thumb;
        give `at` to overrule it."""
        from .parting import HIDDEN_WEIGHT, rim_flat, seam_visibility
        curves = _contours(ext, cut, 0.0)
        seam = float(sum(_length(c) for c in curves))
        if not curves:
            return None
        where = _assign(comps, cut, given)
        travel, cross = 0.0, 0
        for c in comps:
            sd = cut.phi(samples[c["id"]])
            cross += int((sd > 0).any() and (sd < 0).any())
            travel += max(0.0, float(-sd.min())) if where[c["id"]] == "base" else max(0.0, float(sd.max()))
        depth = _depths(ext, cut)
        shallow = sum(max(0.0, min_depth - d) for d in depth)
        if cut.planar and cut.kind == "plane":
            seen, hid = seam, 0.0                # an axis plane: kept as before
            if hidden:
                seen, hid = seam_visibility(ext, curves, hidden, deg=1e9)
        else:
            seen, hid = seam_visibility(ext, curves, hidden)
        flat = 1.0 if cut.planar else rim_flat(curves, cut)
        fdm = 0.5 * seam * (1.0 - flat) if pkind == "fdm" else 0.0
        sc = seen + HIDDEN_WEIGHT * hid + 4.0 * travel + 20.0 * cross + 10.0 * shallow + fdm
        return {"score": round(sc, 1), "travel": round(travel, 1), "seam": round(seam, 1),
                "seamSeen": round(seen, 1), "seamHidden": round(hid, 1), "crossing": cross,
                "rimFlat": round(flat, 3)}

    scored = []
    for c in cands:
        s_ = score(c)
        if s_ is not None:
            scored.append((s_, c))
    if curved and levels is None:
        from .parting import propose
        # the two lift directions whose best flat cut scores best, and their
        # two best levels: the non-flat cuts start from there
        best_dir = {}
        for s_, c in scored:
            key = tuple(np.round(c.n, 6))
            best_dir.setdefault(key, []).append((s_["score"], c))
        ranked_dirs = sorted(best_dir.values(), key=lambda L: min(x[0] for x in L))[:2]
        for L in ranked_dirs:
            L.sort(key=lambda x: x[0])
            c0 = L[0][1]
            lv_n = []
            for _sc, c in L:
                if all(abs(c.h0 - x) > 1.0 for x in lv_n):
                    lv_n.append(c.h0)
                if len(lv_n) == 2:
                    break
            base = _Cut.plane(c0.n, 0.0)
            for c in propose(ext, base, lv_n, comps, samples, avoid_specs, min_depth=min_depth):
                s_ = score(c)
                if s_ is not None:
                    scored.append((s_, c))
    pre = sorted(((s_["score"], i, s_, c) for i, (s_, c) in enumerate(scored)), key=lambda x: x[:2])
    rows, best, n_ok, n_checked = [], np.inf, 0, 0
    for sc_, _i, s_, c in pre:
        # the best four that work are all checked: apply_split places joints
        # on each and takes the one that holds best. A cut that fails costs
        # a full path search for the part with no way in, so at most
        # max_checks are checked once one works.
        if (sc_ >= best and n_ok >= 4) or (n_ok >= 1 and n_checked >= max_checks):
            rows.append({**c.report(), **s_, "ok": None,
                         "why": "not checked: scores worse than a cut that works" if sc_ >= best and n_ok >= 4
                         else f"not checked: {max_checks} cuts checked already", "_cut": c})
            continue
        t_c = time.time()
        n_checked += 1
        r = _check_cut(ext, skin, t, comps, c, given, tol, avoid, order, base_dir, SEARCH_SLIDE)
        if normal in (None, "auto") and r.get("ok"):
            # the half with the components is the base (the cover is the
            # lid that comes off): flip the normal if they sit in the other
            w = r.get("componentsIn") or {}
            if sum(v == "cover" for v in w.values()) > sum(v == "base" for v in w.values()):
                r2 = _check_cut(ext, skin, t, comps, c.flipped(), given, tol, avoid, order, base_dir, SEARCH_SLIDE)
                if r2.get("ok"):
                    r = r2
        r.update({k: v for k, v in s_.items() if k != "seam"})
        r["seconds"] = round(time.time() - t_c, 1)
        if r.get("ok"):
            best = min(best, sc_)
            n_ok += 1
        rows.append(r)
    ok = [r for r in rows if r.get("ok")]
    if not ok:
        return None, rows
    ok.sort(key=lambda r: r["score"])
    return ok[0], rows


# ---------------------------------------------------------------------------
# joints


def _site_frames(ext, cut, t):
    """Points on the inner wall at the cut (o), inward (u) and along the
    wall (w), every 0.5 mm along the longest inner outline; arc length s and
    the outline's length."""
    curves = _contours(ext, cut, -t, step=0.15)
    if not curves:
        raise ValueError("the skin fills the cut: nothing inside the wall at this level")
    c = max(curves, key=_length)
    o, s, total = _resample(c, 0.5)
    n = cut.n
    g = _grad(ext, o)
    g -= np.outer(g @ n, n)
    u = -g / np.maximum(np.linalg.norm(g, axis=1, keepdims=True), 1e-9)
    w = np.cross(n, u)
    return o, u, w, s, total


class _Frame:
    """World points from local (U inward, W along the wall, S down into the
    base) at a site."""

    def __init__(self, o, u, w, n):
        self.o, self.u, self.w, self.n = o, u, w, n

    def p(self, U, W, S):
        return self.o + np.multiply.outer(U, self.u) + np.multiply.outer(W, self.w) - np.multiply.outer(S, self.n)

    def box(self, U, W, S) -> dict:
        (u0, u1), (w0, w1), (s0, s1) = U, W, S
        c = self.p(0.5 * (u0 + u1), 0.5 * (w0 + w1), 0.5 * (s0 + s1))
        return {"type": "box", "center": _r(c), "size": _r([u1 - u0, w1 - w0, s1 - s0]),
                "axes": [_r(self.u, 6), _r(self.w, 6)]}

    def box_points(self, U, W, S, k=4):
        g = np.stack(np.meshgrid(np.linspace(*U, k), np.linspace(*W, k), np.linspace(*S, k), indexing="ij"),
                     -1).reshape(-1, 3)
        return self.p(g[:, 0], g[:, 1], g[:, 2])


def _comp_clear(comps, pts, margin) -> bool:
    for c in comps:
        if (c["shape"].sdf(pts) < margin).any():
            return False
    return True


def _snap_geometry(S: dict, wall: dict | None = None):
    """The hook, ledge and rooms of one snap in local coordinates. `wall`:
    where the inner wall is (U) over each part's depth, when it curves in
    below the cut: the hook moves in by `shift` and the ledge and root
    reach out to the wall."""
    t, b, y, c, L = S["thickness"], S["width"], S["undercut"], S["clearance"], S["length"]
    w = wall or {}
    sh = float(w.get("shift", 0.0))
    a = math.radians(S["leadAngle"])
    hb = y / math.tan(a) + 0.4             # barb: lead-in, then a 0.4 mm land
    hl = S["ledge"]
    root = S["root"]
    m = S["room"]
    an = S["anchor"]
    reach = y + c + t + y + m + sh         # as deep inward as the beam bent in, plus room
    g = {"hb": hb, "shift": sh,
         "root": ((w.get("root", 0.0) - an, y + c + t + sh), (-b / 2, b / 2), (-root, 0.0)),
         # into the root and the barb a little: solids that only touch stay
         # two pieces in a mesh boolean
         "beam": ((y + c + sh, y + c + t + sh), (-b / 2, b / 2), (-0.5 * root, L + 0.3)),
         "barb": ((c + sh, y + c + t + sh), (-b / 2, b / 2), (L, L + hb)),
         "ledge": ((w.get("ledge", 0.0) - an, y + sh), (-b / 2 - 1.0, b / 2 + 1.0),
                   (L - hl - S["play"], L - S["play"])),
         # the other half keeps clear: above the ledge, beside it (where the
         # barb passes, bent in), and the barb's landing under it
         "roomAbove": ((w.get("above", 0.0) + 0.05, reach), (-b / 2 - c, b / 2 + c), (0.0, L - hl - S["play"])),
         "roomBeside": ((y + sh, reach), (-b / 2 - c, b / 2 + c), (L - hl - S["play"], L)),
         "roomBelow": ((w.get("below", 0.0) + 0.05, reach), (-b / 2 - c, b / 2 + c), (L, L + hb + m))}
    # the lead-in: a plane through the barb's tip corner and the land
    A = np.array([y + c + sh, L + hb])
    B = np.array([c + sh, L + hb - y / math.tan(a)])
    T = (B - A) / np.linalg.norm(B - A)
    N = np.array([T[1], -T[0]])            # (U, S), pointing at the wall and down: off the barb
    g["lead"] = (A, N)
    return g


def _wall_profile(ext, f, t, S_levels, reach=4.0, step=0.05):
    """U of the inner wall (sdf = -t) at each depth S, from the frame's
    origin along u: > 0 where the wall has curved in."""
    U = np.arange(-reach, reach + step, step)
    S_levels = np.asarray(S_levels, float)
    UU, SS = np.meshgrid(U, S_levels)
    d = ext.sdf(f.p(UU.ravel(), np.zeros(UU.size), SS.ravel())).reshape(UU.shape)
    inside = d < -t
    first = np.argmax(inside, axis=1)
    return np.where(inside.any(axis=1), U[first], np.inf)


def _snap_shapes(f: _Frame, S: dict, wall: dict | None = None) -> dict:
    g = _snap_geometry(S, wall)
    A, N = g["lead"]
    nw = N[0] * f.u - N[1] * f.n           # (U, S) -> world (S runs along -n)
    big = 40.0
    ca = f.p(A[0], 0.0, A[1]) - 0.5 * big * nw
    tw = np.cross(nw, f.w)
    cut = {"type": "box", "center": _r(ca), "size": [big, big, big], "axes": [_r(nw, 6), _r(f.w, 6)]}
    _ = tw
    barb = {"type": "intersect", "shapes": [f.box(*g["barb"]), cut]}
    hook = {"type": "union", "shapes": [f.box(*g["root"]), f.box(*g["beam"]), barb]}
    rooms = {"type": "union", "shapes": [f.box(*g["roomAbove"]), f.box(*g["roomBeside"]),
                                         f.box(*g["roomBelow"])]}
    return {"hook": hook, "ledge": f.box(*g["ledge"]), "room": rooms, "geom": g}


def _spread(feasible: np.ndarray, s: np.ndarray, total: float, k: int, taken: list, keep: float):
    """k sites spread evenly round the outline among the feasible ones, at
    least `keep` mm (arc) from the sites in `taken`. Returns indices."""
    if k <= 0:
        return []
    ok = feasible.copy()
    for j in taken:
        d = np.abs(s - s[j])
        d = np.minimum(d, total - d)
        ok &= d >= keep
    idx = np.flatnonzero(ok)
    if len(idx) < k:
        return None
    best, best_cost = None, np.inf
    for off in np.arange(0.0, total / k, 1.0):
        pick, cost = [], 0.0
        for i in range(k):
            target = (off + i * total / k) % total
            d = np.abs(s[idx] - target)
            d = np.minimum(d, total - d)
            j = int(np.argmin(d))
            if d[j] > 0.3 * total / k or idx[j] in pick:
                pick = None
                break
            pick.append(int(idx[j]))
            cost += float(d[j])
        if pick is not None and cost < best_cost:
            best, best_cost = pick, cost
    return best


def _ray_exit(ext, p, d, reach, step=0.1):
    """mm from p (inside) along d to where the exterior ends."""
    ts = np.arange(0.0, reach, step)
    v = ext.sdf(p[None, :] + ts[:, None] * d[None, :])
    out = np.flatnonzero(v > 0)
    return float(ts[out[0]]) if len(out) else None


def _lip_mesh(ext, cut, t, c, w, h, root, gaps: list, room: bool, step=0.15):
    """The lip (room=False: base preserve) or the room the cover keeps
    clear round it (room=True), as a closed mesh: a band following the
    inner wall, marching cubes on a fine grid in the cut's own (u, v, S)
    coordinates (S down from the cut surface along -n), the gaps (local
    boxes) cut out."""
    import trimesh
    from skimage import measure

    from .shapes import shape_from_spec

    if room:
        s0, s1 = -(h + c), 0.05            # S: down into the base (S < 0 is the cover side)
        din, dout = t - 0.05, t + 2 * c + w
    else:
        s0, s1 = -h, root
        din, dout = t + c, t + c + w
    pad = 2 * step
    (u0, u1), (v0, v1), _ = cut.footprint(ext, pad=pad)
    U = np.arange(u0, u1 + step, step)
    V = np.arange(v0, v1 + step, step)
    Sx = np.arange(s0 - pad, s1 + pad + step, step)
    dims = (len(U), len(V), len(Sx))
    F = np.empty(dims)
    gap_shapes = [shape_from_spec(g) for g in gaps]
    rows = max(1, 2_000_000 // (len(V) * len(Sx)))
    for i0 in range(0, len(U), rows):
        uu, vv, ss = np.meshgrid(U[i0:i0 + rows], V, Sx, indexing="ij")
        q = cut.point(uu, vv, ss).reshape(-1, 3)
        S = ss.ravel()
        d = ext.sdf(q)
        f = np.maximum(d + din, -(d + dout))
        if not room:
            # below the cut the lip runs into the base's skin, so it is one
            # piece with it
            f = np.where(S > 0, np.maximum(d + (t - 0.3), -(d + dout)), f)
        f = np.maximum(f, np.maximum(s0 - S, S - s1))
        for gs in gap_shapes:
            glo_, ghi_ = gs.bounds()
            m_ = (f < 2 * step) & np.all((q >= glo_ - 2 * step) & (q <= ghi_ + 2 * step), axis=1)
            if m_.any():
                f[m_] = np.maximum(f[m_], -gs.sdf(q[m_]))
        F[i0:i0 + rows] = f.reshape(uu.shape)
    F[[0, -1], :, :] = F[:, [0, -1], :] = F[:, :, [0, -1]] = step
    if F.min() >= 0:
        return None
    verts, faces, _, _ = measure.marching_cubes(F, 0.0, spacing=(step, step, step))
    world = cut.point(U[0] + verts[:, 0], V[0] + verts[:, 1], Sx[0] + verts[:, 2])
    mesh = trimesh.Trimesh(world, faces[:, ::-1], process=True)
    if mesh.volume < 0:
        mesh.invert()
    # through manifold: marching cubes leaves pinched vertices where the band
    # thins out; a manifold mesh is what the study's mesh shapes need
    from .shapes import manifold_from_trimesh, trimesh_from_manifold
    man = manifold_from_trimesh(mesh)
    if hasattr(man, "simplify"):
        man = man.simplify(0.02)           # flat faces need no 0.15 mm triangles: 500k -> ~6k faces
    out = trimesh_from_manifold(man)
    return out if out.is_watertight else mesh


# ---------------------------------------------------------------------------
# the halves


def _components_from(study, components, base_dir):
    from .handoff import _fine_mesh_shape

    if components is None:
        components = [k for k in study.get("keepOut") or [] if k.get("insertFrom") is not None
                      or k.get("component")]
    out = []
    for i, c in enumerate(components):
        cid = c.get("id") or f"component-{i}"
        spec = c.get("shape") or c.get("spec")
        if isinstance(spec, str):
            spec = {"type": "mesh", "path": spec}
        spec = copy.deepcopy(spec)
        if spec.get("type") == "mesh" and base_dir and not os.path.isabs(spec["path"]):
            spec["path"] = os.path.join(base_dir, spec["path"])
        out.append({"id": cid, "spec": spec, "shape": _fine_mesh_shape(spec, None),
                    "installedOn": c.get("installedOn"), "insertFrom": None})
    return out


def _desk(ext, cut):
    """A thin slab under the base's far face, across the lift direction."""
    lo, hi = ext.bounds()
    if cut.axis is not None:
        a, sg = cut.axis
        slo, shi = lo - 1.0, hi + 1.0
        if sg > 0:
            slo[a], shi[a] = lo[a] - 0.5, lo[a] + 0.3
        else:
            slo[a], shi[a] = hi[a] - 0.3, hi[a] + 0.5
        return {"type": "box", "min": _r(slo), "max": _r(shi)}
    (u0, u1), (v0, v1), _ = cut.footprint(ext, 1.0)
    low = float((_inside(ext) @ cut.n).min()) - 0.75
    c = 0.5 * (u0 + u1) * cut.e1 + 0.5 * (v0 + v1) * cut.e2 + (low + 0.1) * cut.n
    return {"type": "box", "center": _r(c), "size": _r([u1 - u0, v1 - v0, 0.8]),
            "axes": [_r(cut.e1, 9), _r(cut.e2, 9)]}


def _rim(ext, cut, side, width=0.4):
    """The half's slice against the cut, `width` thick: where the cover
    sits on the base."""
    if cut.axis is not None and cut.planar:
        lo, hi = ext.bounds()
        a, sg = cut.axis
        at = cut.at
        rlo, rhi = lo - 1.0, hi + 1.0
        if side == "cover":
            rlo[a], rhi[a] = (at, at + width) if sg > 0 else (at - width, at)
        else:
            rlo[a], rhi[a] = (at - width, at) if sg > 0 else (at, at + width)
        return {"type": "box", "min": _r(rlo), "max": _r(rhi)}
    return cut.band_spec(ext, side, width)


def _region_side(spec, cut, base_dir):
    from .shapes import shape_from_spec
    try:
        P = _samples(shape_from_spec(spec, base_dir), 0.6, 2000)
    except Exception:  # noqa: BLE001
        return "both"
    up = float(np.mean(cut.phi(P) > 0))
    return "cover" if up > 0.95 else "base" if up < 0.05 else "both"


def _rim_patch(rim: dict, ld: dict, study: dict, cut, ext, base_dir) -> dict:
    """The stretch of a half's rim a load on the other half comes in
    through: the rim within a sphere round the load's point (projected onto
    a flat cut), grown until it holds material; a load spread wider than
    half the part (a drop's floor) comes in over the whole rim."""
    from .shapes import shape_from_spec
    spec = None
    if ld.get("shape") is not None:
        spec = ld["shape"]
    elif ld.get("region") is not None:
        for p in list(study.get("preserve") or []) + list(study.get("keepOut") or []):
            if p.get("id") == ld["region"]:
                spec = p.get("shape")
                break
    try:
        lo, hi = shape_from_spec(spec, base_dir).bounds()
    except Exception:  # noqa: BLE001
        return rim
    elo, ehi = ext.bounds()
    if np.max(hi - lo) > 0.5 * np.max(ehi - elo):
        return rim
    at = np.asarray(ld.get("at"), float) if ld.get("at") is not None else 0.5 * (lo + hi)
    if cut.axis is not None and cut.planar:
        at = at.copy()
        at[cut.axis[0]] = cut.at
    r = max(12.0, 0.5 * float(np.max(hi - lo)))
    band = shape_from_spec(rim, base_dir)
    for _ in range(8):
        patch = {"type": "intersect", "shapes": [copy.deepcopy(rim),
                                                   {"type": "sphere", "center": _r(at), "r": round(r, 3)}]}
        try:
            P = _samples(shape_from_spec(patch, base_dir), 0.5, 2000)
        except Exception:  # noqa: BLE001
            P = np.zeros((0, 3))
        if len(P) and (np.maximum(ext.sdf(P), band.sdf(P)) <= 0).sum() >= 20:
            return patch
        r *= 1.6
    return rim


def _load_force(ld) -> np.ndarray:
    """The net force of one load entry, N: a force, or a carried mass's
    weight x dynamic factor along its gravity."""
    if ld.get("force") is not None:
        f = np.asarray(ld["force"], float)
        return f if f.shape == (3,) else np.zeros(3)
    if ld.get("mass") is not None:
        g = ld.get("gravity", "-z")
        if isinstance(g, str):
            k = g.strip().lower()
            k = k if k[:1] in "+-" else "+" + k
            v = np.zeros(3)
            v["xyz".index(k[1])] = 1.0 if k[0] == "+" else -1.0
        else:
            v = np.asarray(g, float)
            v = v / max(np.linalg.norm(v), 1e-12)
        return 9.81 * float(ld["mass"]) * float(ld.get("dynamic", 1.0)) * v
    return np.zeros(3)


def _halves_under(P, halves: dict, ext, reach: float) -> set:
    """The halves a load or support at sample points P bears on: those with
    material under it at least `reach` deep (half a cell). Being on a
    half's side of the cut is not enough: a press on the top skin near a
    tilted seam can reach the base only in a sliver thinner than a cell, or
    in the air above it, and a load there touches no cell of the base. With
    no half that deep, the one it reaches furthest into."""
    e = ext.sdf(P)
    depth = {k: float(np.maximum(h.sdf(P), e).min()) for k, h in halves.items()}
    deep = {k for k, d in depth.items() if d <= -reach}
    return deep or {min(depth, key=depth.get)}


def apply_split(study: dict, split: dict | None = None, components: list | None = None,
                out_dir: str = "loadpath-split", base_dir: str | None = None) -> dict:
    """Cut a study's exterior into a base and a cover, with the lip, screw
    posts and snaps between them. See the module docstring and `loadpath
    schema split`.

    study: an ordinary study whose designSpace is the exterior, with a
    `skin` (the exterior that is kept). split: the split block (default
    study["split"]). components: [{id, shape, installedOn?}] (default: the
    study's keep-outs that carry an insertFrom). Writes the lip meshes to
    <out_dir>/cad/ and returns {"studies": {"base", "cover"}, "report"}."""
    t0 = time.time()
    study = copy.deepcopy(study)
    split = copy.deepcopy(split if split is not None else study.get("split") or {})
    study.pop("split", None)
    if study.get("skin") is None:
        raise ValueError("a split shell keeps the exterior: the study needs a skin (its wall, mm)")
    skin = study["skin"] if isinstance(study["skin"], dict) else {"thickness": float(study["skin"])}
    t = float(skin["thickness"])
    ext, ext_specs = _exterior(study, base_dir)
    comps = _components_from(study, components, base_dir)
    comp_ids = {c["id"] for c in comps}
    given = dict(split.get("componentsIn") or {})
    bad = [k for k in given if k not in comp_ids]
    if bad:
        raise ValueError(f"split.componentsIn names {bad}, which are not components ({sorted(comp_ids)})")
    tol = float(split.get("pathTolerance", 0.25))
    from .shapes import shape_from_spec
    avoid = [shape_from_spec(s, base_dir) for s in split.get("avoid") or []]
    order = list(split.get("assembly") or [c["id"] for c in comps])
    from .fits import _process_of
    pkind, build, _holes = _process_of(study)
    from .spec import PROCESSES
    min_member = float(study.get("minMember") or (PROCESSES.get(pkind, {}).get("minFeature") or 0.8))
    warnings = []

    # the cut
    normal = split.get("normal", "auto")
    sv_ = split.get("screws")
    if isinstance(sv_, dict) and sv_.get("from") and normal in (None, "auto"):
        # screws go in through the base's far face: the cut is across that axis
        fa_, fs_ = _axis(sv_["from"])
        normal = _name(fa_, -fs_)
    given_cut = None
    if split.get("seam"):
        seam = split["seam"]
        given_cut = _seam_cut(ext, normal, seam.get("points") if isinstance(seam, dict) else seam)
    elif split.get("steps"):
        if normal in (None, "auto") or split.get("at") is None:
            raise ValueError("split.steps needs normal and at: the cut's level away from the steps")
        given_cut = _stepped_cut(ext, normal, float(split["at"]), split["steps"],
                                 float(split.get("stepWidth", 0.5)), base_dir)
    elif split.get("at") is not None and normal not in (None, "auto"):
        given_cut = _cut_plane(normal, float(split["at"]))
    if given_cut is not None:
        cut = _check_cut(ext, skin, t, comps, given_cut, given, tol, avoid, order, base_dir)
        if normal in (None, "auto") and cut.get("ok"):
            # a seam's own plane: the half holding the components is the base
            w_ = cut.get("componentsIn") or {}
            if sum(v == "cover" for v in w_.values()) > sum(v == "base" for v in w_.values()):
                c2 = _check_cut(ext, skin, t, comps, given_cut.flipped(), given, tol, avoid, order, base_dir)
                if c2.get("ok"):
                    cut = c2
        rows = [cut]
        how = "as given"
        if not cut.get("ok"):
            warnings.append(f"the given cut does not work: {cut.get('why')}")
    else:
        levels = [float(split["at"])] if split.get("at") is not None else None
        mode = split.get("search", "all")
        if mode not in ("all", "flat"):
            raise ValueError("split.search is 'all' (default: flat, tilted, stepped, silhouette and crease cuts) "
                             "or 'flat' (axis planes only)")
        hidden = [shape_from_spec(s_, base_dir) for s_ in split.get("hidden") or []]
        cut, rows = _search(ext, skin, t, comps, given, tol, avoid, order, normal, levels,
                            curved=mode == "all", pkind=pkind, hidden=hidden,
                            avoid_specs=list(zip(split.get("avoid") or [], avoid)), base_dir=base_dir)
        how = "searched"
        if cut is None:
            raise ValueError("no cut (flat, tilted, stepped or curved) separates the halves with every component able to go in: "
                             + "; ".join(sorted({str(r.get('why')) for r in rows})[:6]))
    def _joints(cut):
        """Place the screws and snaps for a cut; returns what the rest needs."""
        warnings = []
        wanted = 0
        r_ring = r_sleeve = 0.0
        C = cut["_cut"]
        n = C.n
        where = cut.get("componentsIn") or _assign(comps, C, given)

        # the sites round the inner wall
        o, u, w, s, total = _site_frames(ext, C, t)
        frames = [_Frame(o[i], u[i], w[i], n) for i in range(len(o))]
        # clearances: wide enough that two faces, each off by the process's
        # band, still clear by RSS; drawn wider by the gap bias
        size = float(np.max(np.subtract(*ext.bounds()[::-1])))
        tolj = joint_tolerance(_holes, build, n, size)
        c_base = 0.25 if pkind == "fdm" else 0.2
        worst_case = split.get("stackup", "rss") == "worst"
        if split.get("stackup", "rss") not in ("rss", "worst"):
            raise ValueError("split.stackup is 'rss' (default) or 'worst'")
        k2 = 2.0 if worst_case else math.sqrt(2)
        c_target = float(split["clearance"]) if split.get("clearance") is not None else \
            max(c_base, round(k2 * tolj["face"] + 0.05, 2))
        c_fit = c_target + tolj["bias"]
        gap_stack = _stack(c_target, {"thisFace": tolj["face"], "otherFace": tolj["face"]})
        gap_stack["drawn"] = round(c_fit, 3)
        gap_stack["holds"] = _verdict(gap_stack, tolj["repeat"])
        if gap_stack["rss"] < 0:
            warnings.append(f"split.clearance {c_target:g} mm is under the process's RSS band "
                            f"({math.sqrt(2) * tolj['face']:.2f} mm): the lip and snaps may bind")
        margin = float(split.get("margin", 0.5))

        # screws
        screws_v = split.get("screws", False)
        screws = []
        exterior_changes = []
        taken = []
        if screws_v:
            from .fasteners import METRIC
            sc = dict(screws_v) if isinstance(screws_v, dict) else {}
            if not sc.get("from"):
                raise ValueError("split.screws.from names the face the screws go in through (e.g. '-z'): a screw "
                                 "needs a hole in the exterior, and only the designer can say where one may go")
            if float(_vec_of(sc["from"]) @ -n) < 0.7:
                raise ValueError(f"split.screws.from must be the base's far face, toward {C.name(-1)}, for a cut "
                                 f"lifting off toward {C.name()}")
            size = sc.get("size", "M2")
            if size not in METRIC:
                raise ValueError(f"split.screws.size must be one of {sorted(METRIC)}")
            hole, head = METRIC[size]
            nominal = hole / 1.1
            r_ring = 0.5 * head + max(1.0, 0.15 * nominal)
            insert = sc.get("insert")
            r_hole_c = 0.5 * float(insert) if insert else 0.42 * nominal
            r_sleeve = (0.5 * float(insert) if insert else 0.5 * nominal) + max(1.5, 0.35 * nominal)
            tl = float(sc.get("threadLength", 2.5 * nominal if not insert else max(4.0, 1.5 * float(insert))))
            head_h = float(sc.get("headHeight", nominal))
            inset = max(r_ring, r_sleeve) + 0.2
            r_pocket = max(0.5 * head + 0.5, r_ring)
            feas = np.zeros(len(o), bool)
            info = [None] * len(o)
            ang = np.linspace(0, 2 * np.pi, 16, endpoint=False)
            max_in = float(sc.get("maxInset", 12.0))

            def screw_site(f, ins):
                cpt = f.p(ins, 0.0, 0.0)
                down = _ray_exit(ext, cpt, -n, 200.0)
                if down is None or down < head_h + 3.0:
                    return None
                up = _ray_exit(ext, cpt, n, 200.0)
                if up is None or up < tl + t + 0.5:
                    return None
                ring = np.outer(np.cos(ang), f.u) + np.outer(np.sin(ang), f.w)
                levels = np.linspace(-tl, down - head_h - 0.3, 12)      # along -n from the cut (S), to the seat
                bore = np.vstack([cpt + (0.5 * hole + 0.8) * ring - lv * n for lv in levels])
                if (ext.sdf(bore) > -0.6).any():
                    return None
                # the head's counterbore sits in a flat enough patch of the face
                pocket = np.vstack([cpt + (r_pocket + 0.8) * ring - lv * n
                                    for lv in np.linspace(down - head_h - 0.3, down - 0.4, 3)])
                if (ext.sdf(pocket) > -0.2).any():
                    return None
                col = np.vstack([cpt + r * ring - lv * n for r in (0.0, 0.5 * r_ring, r_ring) for lv in levels])
                if not _comp_clear(comps, col, margin):
                    return None
                return cpt, down, up, ins

            for i, f in enumerate(frames):
                if i % 2:
                    continue
                # against the wall if it can be; further in where the face
                # curves up to the wall (a rounded bottom edge)
                for ins in np.arange(inset, inset + max_in + 1e-9, 1.0):
                    got = screw_site(f, float(ins))
                    if got is not None:
                        feas[i] = True
                        info[i] = got
                        break
            count = sc.get("count", "auto")
            k = max(2, int(round(total / 90.0))) if count == "auto" else int(count)
            wanted += k
            pick = _spread(feas, s, total, k, [], 0.0)
            kk = k
            while pick is None and kk > 1:
                kk -= 1
                pick = _spread(feas, s, total, kk, [], 0.0)
            pick = pick or []
            if len(pick) < k:
                warnings.append(f"room for {len(pick)} of {k} screws along the wall (feasible sites: {int(feas.sum())}): "
                                "a smaller size, or a cut where the wall runs deeper, fits more")
            fit = sc.get("fit", "normal")
            for j, i in enumerate(pick):
                cpt, down, _up, ins_used = info[i]
                sid = f"screw-{j + 1}"
                bottom = cpt - down * n
                seat = bottom + (head_h + 0.3) * n
                grip = float((cpt - seat) @ n)
                screws.append({"id": sid, "site": int(i), "at": _r(cpt, 3), "size": size, "inset": round(ins_used, 2),
                               "base": {"id": sid, "size": size, "seat": _r(seat), "direction": C.name(),
                                        "grip": round(grip, 3), "end": "face", "headOn": "this",
                                        "counterbore": round(2 * r_pocket, 3), "fit": fit},
                               "cover": {"id": sid, "size": size, "seat": _r(cpt), "direction": C.name(),
                                         "grip": 0.0, "end": "thread", "headOn": "mating",
                                         "threadLength": round(tl, 3),
                                         **({"insert": float(insert)} if insert else {})},
                               "hole": {"face": sc["from"], "at": _r(bottom, 3), "diameter": round(2 * r_pocket, 2),
                                        "depth": round(head_h + 0.3, 2)}})
                exterior_changes.append({"what": f"{sid} counterbore", "face": sc["from"], "at": _r(bottom, 2),
                                         "diameter": round(2 * r_pocket, 2), "depth": round(head_h + 0.3, 2),
                                         "allowedBy": "split.screws.from"})
                taken.append(i)
            _ = r_hole_c

        # snaps
        snaps_v = split.get("snaps", {} if not screws else False)
        snaps = []
        if snaps_v is not False and snaps_v is not None:
            sv = dict(snaps_v) if isinstance(snaps_v, dict) else {}
            across = bool(pkind == "fdm" and build is not None and abs(float(np.asarray(build) @ n)) > 0.7)
            allow = snap_allowable(study.get("material", "pla"), pkind, across_layers=across,
                                   repeated=bool(sv.get("repeated", False)))
            E = allow["E"]
            # the barb's catch on the ledge (undercut less the gap) moves with
            # the hook's and the ledge's faces, the barb's own size, and how
            # far the lip lets the halves shift: size the beam for the most
            # it bends (RSS) and the undercut for the least it catches
            # the lip locates the halves to its clearance (without one the
            # snaps' own side gaps do, which are the same)
            band = {"hookFace": tolj["face"], "ledgeFace": tolj["face"], "lipPlay": c_target,
                    "barbSize": 0.5 * (float((_holes or {}).get("tol", 0.0))
                                       + float((_holes or {}).get("tolPerMm", 0.0)) * 1.0)}
            rss_u = sum(band.values()) if worst_case else math.sqrt(sum(v * v for v in band.values()))
            y_min = c_target + rss_u + 0.35
            y = float(sv.get("undercut", max(0.8, round(y_min, 2))))
            band["barbSize"] = 0.5 * (float((_holes or {}).get("tol", 0.0)) + float((_holes or {}).get("tolPerMm", 0.0)) * y)
            catch = _stack(y - c_target, band)
            catch["holds"] = ("catches at worst case" if catch["worst"] > 0.1 else
                              "catches by RSS; at worst case it may slip: a larger undercut, or a tuned printer"
                              if catch["rss"] > 0.1 else "may not catch: increase the undercut")
            if catch["rss"] <= 0.1:
                warnings.append(f"snap undercut {y:g} mm leaves {catch['rss']:.2f} mm of catch by RSS after the "
                                "process band and the lip's play: increase split.snaps.undercut")
            defl = (y - c_target) + (catch["nominal"] - catch["worst" if worst_case else "rss"])
            defl_worst = (y - c_target) + (catch["nominal"] - catch["worst"])
            tb = float(sv.get("thickness", max(min_member, 1.2)))
            b = float(sv.get("width", 6.0))
            lead = float(sv.get("leadAngle", 30.0))
            L_req = math.sqrt(1.5 * tb * defl / allow["strain"]) * 1.1
            L = float(sv.get("length", max(L_req, 4.0)))
            S = {"thickness": tb, "width": b, "undercut": y, "clearance": c_fit, "length": L, "leadAngle": lead,
                 "ledge": float(sv.get("ledge", max(1.5, min_member))), "root": float(sv.get("root", 2.0)),
                 "anchor": max(0.3, min(1.0, t - 0.3)), "play": 0.05, "room": float(sv.get("room", 0.5))}
            g0 = _snap_geometry(S)

            def feasible(fr):
                feas = np.zeros(len(fr), bool)
                walls = [None] * len(fr)
                depth_need = L + g0["hb"] + S["room"]
                hl = S["ledge"] + S["play"]
                Sl = np.linspace(-S["root"], depth_need, 28)
                for i, f in enumerate(fr):
                    dn = _ray_exit(ext, f.p(y + c_fit + tb, 0.0, 0.0), -f.n, 200.0)
                    upn = _ray_exit(ext, f.p(y + c_fit + tb, 0.0, 0.0), f.n, 200.0)
                    if dn is None or dn < depth_need + t + 0.3 or upn is None or upn < S["root"] + t:
                        continue
                    if not C.planar:
                        # the seam runs level across the hook (not on a step's climb)
                        hw_ = 0.5 * b + 1.0 + c_fit
                        span = C.h(np.stack([f.o + k_ * hw_ * f.w for k_ in (-1.0, 0.0, 1.0)]))
                        if float(np.ptp(span)) > 0.25:
                            continue
                    prof = _wall_profile(ext, f, t, Sl)
                    if not np.isfinite(prof).all():
                        continue

                    def rng(s0, s1, fn):
                        m_ = (Sl >= s0 - 0.3) & (Sl <= s1 + 0.3)
                        return float(fn(prof[m_]))
                    shift = max(0.0, rng(0.0, L + g0["hb"], np.max))
                    if shift > float(sv.get("maxShift", 3.0)):
                        continue
                    wall = {"shift": shift, "root": rng(-S["root"], 0.0, np.min), "ledge": rng(L - hl, L, np.min),
                            "above": rng(0.0, L - hl, np.max), "below": rng(L, depth_need, np.max)}
                    g = _snap_geometry(S, wall)
                    barb = f.box_points(*g["barb"], k=3)
                    if (ext.sdf(barb) > -(t + 0.5 * c_fit)).any():
                        continue
                    # the ledge and the root reach into the skin, not through it
                    ok_anchor = True
                    for nm in ("ledge", "root"):
                        bx = g[nm]
                        u0 = bx[0][0] + 0.1
                        pts = f.box_points((u0, u0), bx[1], bx[2], k=3)
                        d = ext.sdf(pts)
                        if (d > -0.15).any() or (d < -t - 0.05).any():
                            ok_anchor = False
                            break
                    if not ok_anchor:
                        continue
                    pts = np.vstack([f.box_points(*g[k], k=4) for k in ("beam", "barb", "roomAbove", "roomBeside",
                                                                      "roomBelow")])
                    if _comp_clear(comps, pts, margin) and (ext.sdf(pts) < -t + 0.05).all():
                        feas[i] = True
                        walls[i] = wall
                return feas, walls

            # the hook hangs from the cover into the base, or stands up from the
            # base into the cover: whichever half has straight wall enough
            on = sv.get("on", "auto")
            if on not in ("auto", "cover", "base"):
                raise ValueError("split.snaps.on is 'cover', 'base' or 'auto'")
            sides = ["cover", "base"] if on == "auto" else [on]
            best_side, feas, snap_frames, snap_walls = None, None, None, None
            for sd in sides:
                fr = frames if sd == "cover" else [_Frame(f.o, f.u, np.cross(-n, f.u), -n) for f in frames]
                fs, wl = feasible(fr)
                if best_side is None or fs.sum() > feas.sum():
                    best_side, feas, snap_frames, snap_walls = sd, fs, fr, wl
            count = sv.get("count", "auto")
            if count == "auto":
                k = 2 * max(1, int(round(total / (180.0 if screws else 100.0))))
            else:
                k = int(count)
            keep = 0.5 * b + (max(r_ring, r_sleeve) if screws else 0.0) + 3.0
            wanted += k
            pick = _spread(feas, s, total, k, taken, keep)
            while pick is None and k > 1:
                k -= 1
                pick = _spread(feas, s, total, k, taken, keep)
            if not pick:
                warnings.append("no room for a snap along the inner wall at this cut (depth, a curving wall, or "
                                "components in the way)")
                pick = []
            calc = snap_cantilever(L, tb, b, defl, E, lead, float(sv.get("returnAngle", 90.0)))
            worst_strain = 1.5 * tb * defl_worst / L ** 2
            if allow.get("pushFactor"):
                # the card's snap ladder pushed this much of the formula's force
                calc = {**calc, **{k: None if calc[k] is None else calc[k] * allow["pushFactor"]
                                   for k in ("push", "pull")}}
            for j, i in enumerate(pick):
                sh = _snap_shapes(snap_frames[i], S, snap_walls[i])
                snaps.append({"id": f"snap-{j + 1}", "site": int(i), "at": _r(frames[i].o, 3), "hookOn": best_side,
                              "inward": _r(frames[i].u, 4), "shapes": sh, "n": snap_frames[i].n,
                              "shift": round(snap_walls[i]["shift"], 3)})
                taken.append(i)
            snap_summary = {"count": len(pick), "hookOn": best_side, "length": round(L, 2), "thickness": tb, "width": b, "undercut": y,
                            "deflection": round(defl, 3), "deflectionWorst": round(defl_worst, 3),
                            "strainWorst": round(worst_strain, 5), "catch": catch, "leadAngle": lead, "returnAngle": float(sv.get("returnAngle", 90.0)),
                            "barbHeight": round(g0["hb"], 2), "strain": round(calc["strain"], 5),
                            "allowable": round(allow["strain"], 5), "allowableBasis": allow["basis"],
                            "acrossLayers": across, "ok": calc["strain"] <= allow["strain"] * 1.0001,
                            "deflectionForce": round(calc["deflectionForce"], 2),
                            "push": None if calc["push"] is None else round(calc["push"], 2),
                            "pull": None if calc["pull"] is None else round(calc["pull"], 2),
                            "holds": "until the ledge or hook breaks (square return)" if calc["pull"] is None
                            else f"{calc['pull']:.1f} N each"}
            if across:
                warnings.append("the snap beams run along the FDM build direction: they bend across layers. Print the "
                                "halves on their sides, or use SLS/MJF, for full strength")
            if not snap_summary["ok"]:
                warnings.append(f"snap strain {100 * calc['strain']:.2f}% is over the allowance "
                                f"{100 * allow['strain']:.2f}%: lengthen, thin or reduce the undercut")
        else:
            snap_summary = None

        return {"tolerance": {**tolj, "clearance": gap_stack,
                              "snapCatch": snap_summary.get("catch") if snap_summary else None},
                "cut": C, "n": n, "where": where, "frames": frames, "c_fit": c_fit,
                "screws": screws, "snaps": snaps, "snap_summary": snap_summary,
                "exterior_changes": exterior_changes, "r_ring": r_ring, "r_sleeve": r_sleeve,
                "warnings": warnings, "placed": len(screws) + len(snaps), "wanted": wanted}

    J = _joints(cut)
    if how == "searched":
        # the best four cuts by score, each with its joints placed: a cut
        # where the wall curves away leaves no room for a snap, and one
        # whose snaps bend past the allowance does not hold. Each joint
        # short of the count asked for adds 15, a snap over its strain
        # allowance 50, each warning 10.
        ranked = sorted([r for r in rows if r.get("ok")], key=lambda r: r["score"])[:4]

        def held(r, J_):
            want = J_.get("wanted") or 0
            sn = J_["snap_summary"]
            pen = 15.0 * max(0, want - J_["placed"]) + (50.0 if sn and not sn["ok"] else 0.0) \
                + 10.0 * len(J_["warnings"])
            r["joints"] = {"placed": J_["placed"], "wanted": want,
                           "snaps": len(J_["snaps"]), "screws": len(J_["screws"]),
                           **({"strain": sn["strain"], "strainWorst": sn["strainWorst"],
                               "allowable": sn["allowable"]} if sn else {}),
                           "warnings": len(J_["warnings"])}
            r["final"] = round(r["score"] + pen, 1)
            return r["final"]

        best_f = held(cut, J)
        for r in ranked[1:]:
            J2 = _joints(r)
            f2 = held(r, J2)
            if f2 < best_f:
                J, cut, best_f = J2, r, f2
        cut["chosenFor"] = "joints placed" if cut is not ranked[0] else "score"
    C, where, frames = J["cut"], J["where"], J["frames"]
    c_fit, screws, snaps, snap_summary = J["c_fit"], J["screws"], J["snaps"], J["snap_summary"]
    exterior_changes, r_ring = J["exterior_changes"], J["r_ring"]
    warnings += J["warnings"]

    # the lip
    lip_v = split.get("lip", {})
    cad = os.path.join(out_dir, "cad")
    os.makedirs(cad, exist_ok=True)
    lip = None
    if lip_v is not False and lip_v is not None:
        lv = dict(lip_v) if isinstance(lip_v, dict) else {}
        lw = float(lv.get("width", max(min_member, 1.2)))
        lh = float(lv.get("height", 2.0))
        lroot = float(lv.get("root", 1.5))
        gaps = []
        for sc in screws:
            f = frames[sc["site"]]
            if sc["inset"] - r_ring > c_fit + lw + c_fit + 0.3:
                continue                   # stands clear of the lip
            gaps.append(f.box((-2.0, 12.0), (-(r_ring + c_fit + 0.5), r_ring + c_fit + 0.5), (-lh - 2, lroot + 2)))
        for sn in snaps:
            f = frames[sn["site"]]
            half = 0.5 * snap_summary["width"] + c_fit + 0.5
            gaps.append(f.box((-2.0, 12.0), (-half, half), (-lh - 2, lroot + 2)))
        m_lip = _lip_mesh(ext, C, t, c_fit, lw, lh, lroot, gaps, room=False)
        m_room = _lip_mesh(ext, C, t, c_fit, lw, lh, lroot, gaps, room=True)
        if m_lip is None or m_room is None:
            warnings.append("the lip does not fit inside the wall at this cut; left off")
        else:
            m_lip.export(os.path.join(cad, "lip.stl"))
            m_room.export(os.path.join(cad, "lip-room.stl"))
            lip = {"height": lh, "width": lw, "clearance": round(c_fit, 3), "gaps": len(gaps),
                   "volume": round(float(m_lip.volume), 1)}

    # the two studies
    studies = {}
    transferred = []
    halves = {sd: shape_from_spec(C.half_spec(ext, sd)) for sd in ("base", "cover")}
    res = study.get("resolution") or {}
    reach = 0.5 * float(res.get("pitch") or study.get("minMember") or 1.0)
    for side in ("base", "cover"):
        region_spec = C.half_spec(ext, side)
        region = shape_from_spec(region_spec)
        ext_spec = copy.deepcopy(ext_specs[0]) if len(ext_specs) == 1 else {"type": "union", "shapes": ext_specs}
        sk = dict(skin)
        # the skin lines the whole exterior and stops at the cut
        sk["of"] = copy.deepcopy(ext_spec)
        h = {k: copy.deepcopy(v) for k, v in study.items()
             if k not in ("designSpace", "skin", "preserve", "keepOut", "supports", "loadCases", "fasteners",
                          "name", "description")}
        h["name"] = f"{study.get('name', 'part')}-{side}"
        h["description"] = (f"{side} of {study.get('name', 'the part')}, split along {C.label()} "
                            f"(loadpath split); the cover comes off toward {C.name()}")
        h["designSpace"] = {"type": "intersect", "shapes": [ext_spec, region_spec]}
        h["skin"] = sk
        pres, keep, sup, fast = [], [], [], []
        ids = set()
        for p in study.get("preserve") or []:
            side_p = _region_side(p["shape"], C, base_dir)
            if side_p in (side, "both"):
                pres.append(copy.deepcopy(p))
                ids.add(p.get("id"))
        for k in study.get("keepOut") or []:
            k2 = copy.deepcopy(k)
            cid = k2.get("id")
            through = C.name(1 if side == "base" else -1)      # out through this half's cut face
            if cid in comp_ids:
                mine = where.get(cid) == side
                path = (cut.get("paths") or {}).get(cid, {}).get("insertFrom") if mine else None
                # the other half's component: this half comes down over it,
                # so it sweeps out through the cut too
                k2["insertFrom"] = path if path is not None else through
            keep.append(k2)
            ids.add(cid)
        for c in comps:
            if c["id"] not in ids:
                mine = where.get(c["id"]) == side
                path = (cut.get("paths") or {}).get(c["id"], {}).get("insertFrom") if mine else None
                keep.append({"id": c["id"], "shape": c["spec"], "bore": False,
                             "insertFrom": path or C.name(1 if side == "base" else -1)})
        if lip is not None:
            if side == "base":
                pres.append({"id": "lip", "shape": {"type": "mesh", "path": "cad/lip.stl"}})
            else:
                keep.append({"id": "lip-room", "shape": {"type": "mesh", "path": "cad/lip-room.stl"},
                             "bore": False})
        joint_loads = []
        pull = float((split.get("snaps") or {}).get("pull", 10.0)) if isinstance(split.get("snaps"), dict) else 10.0
        for sn in snaps:
            fn = sn["n"]
            if side == sn["hookOn"]:
                pres.append({"id": f"{sn['id']}-hook", "shape": sn["shapes"]["hook"]})
                # pulled apart, the ledge pushes the barb away from its root
                joint_loads.append({"shape": copy.deepcopy(sn["shapes"]["hook"]["shapes"][2]),
                                    "force": _r(-pull * fn, 6)})
            else:
                pres.append({"id": f"{sn['id']}-ledge", "shape": sn["shapes"]["ledge"]})
                keep.append({"id": f"{sn['id']}-room", "shape": sn["shapes"]["room"], "bore": False})
                joint_loads.append({"region": f"{sn['id']}-ledge", "force": _r(pull * fn, 6)})
        for sc in screws:
            fast.append(copy.deepcopy(sc[side]))
        h["preserve"] = pres
        h["keepOut"] = keep
        if fast:
            h["fasteners"] = list(study.get("fasteners") or []) + fast
        elif study.get("fasteners"):
            h["fasteners"] = copy.deepcopy(study["fasteners"])
        region_ids = {p.get("id") for p in pres} | {k.get("id") for k in keep} | {"skin"}
        for f in fast:
            region_ids |= {f"{f['id']}-seat", f"{f['id']}-face", f"{f['id']}-thread", f"{f['id']}-bore",
                           f"{f['id']}-head", f"{f['id']}-tap"}

        def mine_target(it):
            if "region" in it:
                return it["region"] in region_ids
            try:
                P_ = _samples(shape_from_spec(it["shape"], base_dir), 0.8, 800)
            except Exception:  # noqa: BLE001
                return True
            return side in _halves_under(P_, halves, ext, reach)

        for su in study.get("supports") or []:
            if mine_target(su):
                sup.append(copy.deepcopy(su))
        lcs = []
        rim_here = None
        for lc in study.get("loadCases") or []:
            all_loads = lc.get("loads") or []
            mine = [mine_target(ld) for ld in all_loads]
            loads = [copy.deepcopy(ld) for ld, m in zip(all_loads, mine) if m]
            if loads:
                # what the case puts on the other half comes into this one
                # through the seam: a drop onto the cover's face pushes the
                # base's components' weight back through the rim, where it
                # had only the support to react on (the tool grip's base read
                # 0.21 on a 3 mm support sphere carrying 300 N)
                # Each comes in on the stretch of rim nearest where it acts,
                # so the moment comes across too: a squeeze on the cover's
                # face put at the whole rim's middle twisted the tool grip's
                # base about its one support (compliance 3e7 vs 500)
                rim_here = rim_here if rim_here is not None else _rim(ext, C, side)
                for ld, m in zip(all_loads, mine):
                    F = _load_force(ld)
                    if m or np.linalg.norm(F) <= 1e-9:
                        continue
                    loads.append({"shape": _rim_patch(rim_here, ld, study, C, ext, base_dir),
                                  "force": _r(F, 6)})
                lcs.append({**copy.deepcopy(lc), "loads": loads})
        if not sup:
            if side == "base":
                # stands on its far face (a desk)
                sup = [{"shape": _desk(ext, C)}]
            else:
                # sits on the base along its rim
                sup = [{"shape": _rim(ext, C, "cover")}]
        if side == "base":
            # what the cover carries comes into the base along the seam: each
            # case acting only on the cover, its resultant on the base's rim
            rim = _rim(ext, C, "base")
            have = {lc.get("name") for lc in lcs}
            for lc in study.get("loadCases") or []:
                if lc.get("name") in have:
                    continue
                F = sum((_load_force(ld) for ld in lc.get("loads") or []), np.zeros(3))
                if np.linalg.norm(F) > 0:
                    lcs.append({"name": f"{lc.get('name', 'case')}-through-seam",
                                "loads": [{"shape": rim, "force": _r(F, 6)}]})
                    transferred.append(lc.get("name"))
        if joint_loads:
            lcs.append({"name": "snap-pull", "loads": joint_loads})
        if not lcs and not fast:
            raise ValueError(f"the {side} has no load case: give the study loads on it, snaps or screws")
        h["supports"] = sup
        if lcs:
            h["loadCases"] = lcs
        else:
            h.pop("loadCases", None)
        studies[side] = h

    from .schema import lint
    lint_rows = {side: [x for x in lint(st, "study") if x["level"] == "error"] for side, st in studies.items()}
    report = {"cut": {**{k: v for k, v in cut.items() if k != "paths" and not k.startswith("_")}, "how": how,
                      "label": C.label(),
                      **({"surface": C.half_spec(ext, "base")} if not C.planar else {})},
              "candidates": [{k: v for k, v in r.items() if k in ("normal", "at", "kind", "levels", "tilt", "steps",
                                                                  "seam", "seamSeen", "seamHidden", "rimFlat",
                                                                  "crossing", "travel", "score", "joints", "final",
                                                                  "ok", "why", "depth", "seconds")}
                             for r in rows],
              "assembly": {"componentsIn": where, "paths": cut.get("paths") or {},
                           "order": "the base's components in order with the cover off; then the cover comes "
                                    f"down along {C.name(-1)}"},
              "lip": lip, "screws": [{k: v for k, v in sc.items() if k not in ("base", "cover", "site")}
                                     for sc in screws],
              "snaps": None if snap_summary is None else {**snap_summary,
                                                          "sites": [{"id": sn["id"], "at": sn["at"],
                                                                     "inward": sn["inward"], "hookOn": sn["hookOn"]}
                                                                    for sn in snaps]},
              "exteriorChanges": exterior_changes, "clearance": round(c_fit, 3), "tolerance": J["tolerance"],
              "loadsThroughSeam": transferred, "warnings": warnings, "lint": lint_rows,
              "seconds": round(time.time() - t0, 1)}
    report["snapsGeometry"] = [{"id": sn["id"], **{k: v for k, v in sn["shapes"].items() if k != "geom"}}
                               for sn in snaps]
    return {"studies": studies, "report": report}


# ---------------------------------------------------------------------------
# checks on the parts as built


def verify_snap(snap: dict, material, pitch: float | None = None, deflection: float | None = None) -> dict:
    """The hook as drawn (report snapsGeometry[i]["hook"]), solved on cut
    cells: its root held, the barb pushed in by the undercut. Reads the
    largest surface strain (max principal stress / E, in the beam) and the
    force it takes. Compare with snap_cantilever's numbers."""
    import trimesh
    from scipy.sparse.linalg import spsolve

    from .cutcell import cut_from_mesh
    from .fem import Stiffness
    from .materials import material_from_spec
    from .shapes import shape_from_spec

    t0 = time.time()
    mat = material_from_spec(material[0] if isinstance(material, list) else material)
    hook = snap["hook"]
    root_box, beam_box = hook["shapes"][0], hook["shapes"][1]
    sh = shape_from_spec(hook)
    man = sh.to_manifold().to_mesh()
    tm = trimesh.Trimesh(np.asarray(man.vert_properties)[:, :3], np.asarray(man.tri_verts), process=True)
    R = np.asarray(beam_box["axes"], float)
    u, w = R[0], R[1]
    tb = float(beam_box["size"][0])
    pitch = float(pitch or tb / 4.0)
    nu = mat.nu
    cg = cut_from_mesh(tm, pitch, nu)
    root = shape_from_spec(root_box)
    rc = np.asarray(root_box["center"], float)
    bc = np.asarray(beam_box["center"], float)
    down = (bc - rc) / np.linalg.norm(bc - rc)      # root -> beam: along the beam (the -n of the frame)

    def held_nodes(mesh):
        xyz_ = mesh.node_xyz
        return (root.sdf(xyz_) <= 0.5 * pitch) & ((xyz_ - rc) @ down <= 0.0)

    # only the cells joined to the root (a corner cell touching nothing
    # else would leave the system singular)
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    m0 = cg.mesh
    ne, nn = m0.n_elem, m0.n_nodes
    rows = np.repeat(np.arange(ne), 8)
    A = coo_matrix((np.ones(ne * 8), (rows, ne + m0.enode.ravel())), shape=(ne + nn, ne + nn))
    _k, lab = connected_components(A, directed=False)
    good = np.unique(lab[ne + np.flatnonzero(held_nodes(m0))])
    cg = cg.restrict(np.isin(lab[:ne], good))
    m = cg.mesh
    xyz = m.node_xyz
    # held where the root meets its half (the root block's far half)
    ext_r = np.asarray(root_box["size"], float)
    held = held_nodes(m)
    barb = shape_from_spec(hook["shapes"][2])
    tip = barb.sdf(xyz) <= 0.5 * pitch
    if not held.any() or not tip.any():
        raise ValueError("verify_snap: no nodes at the root or the barb")
    ndof = m.n_dof
    fixed = np.zeros(ndof, bool)
    fixed[(3 * np.flatnonzero(held)[:, None] + np.arange(3)).ravel()] = True
    K = cg.assemble(Stiffness(m, nu, fixed), float(mat.E)).tocsr()
    f = np.zeros(ndof)
    tipn = np.flatnonzero(tip)
    for k in range(3):
        f[3 * tipn + k] = u[k] / len(tipn)           # 1 N on the barb, inward
    free = np.flatnonzero(~fixed)
    x = np.zeros(ndof)
    x[free] = spsolve(K[free][:, free], f[free])
    disp = float(np.mean(x.reshape(-1, 3)[tipn] @ u))
    y = float(deflection) if deflection is not None else float(snap.get("deflection") or 0.8)
    scale = y / disp
    sig = cg.nodal_stress(x * scale, float(mat.E))
    # max principal stress along the beam
    along_b = (xyz - rc) @ down
    in_beam = (along_b > 0.5 * ext_r[2] + pitch) & (along_b < along_b.max() - 1.0)
    sxx = np.zeros(len(xyz))
    S = np.zeros((len(xyz), 3, 3))
    S[:, 0, 0], S[:, 1, 1], S[:, 2, 2] = sig[:, 0], sig[:, 1], sig[:, 2]
    S[:, 0, 1] = S[:, 1, 0] = sig[:, 3]
    S[:, 1, 2] = S[:, 2, 1] = sig[:, 4]
    S[:, 0, 2] = S[:, 2, 0] = sig[:, 5]
    sxx = np.einsum("i,nij,j->n", down, S, down)
    peak = float(np.abs(sxx[in_beam]).max()) if in_beam.any() else float("nan")
    _ = w
    return {"strain": peak / float(mat.E), "force": scale, "deflection": y, "pitch": pitch,
            "cells": int(m.n_elem), "seconds": round(time.time() - t0, 1),
            "held": "the root block's cover side; the cover itself taken as rigid (reads the beam a little stiff)"}


def check_closed(base_mesh, cover_mesh, normal, hooks: list | None = None, step: float = 0.5,
                 travel: float | None = None) -> dict:
    """The built halves together: overlap where they sit closed (should be
    nothing), and the cover lifting off along `normal` with its snap hooks
    taken out (they bend past the ledges; verify_snap checks that), the
    most they overlap on the way."""
    import manifold3d as m3d

    from .shapes import manifold_from_trimesh, shape_from_spec

    n = _vec_of(normal)
    B = manifold_from_trimesh(base_mesh)
    C = manifold_from_trimesh(cover_mesh)
    closed = float((B ^ C).volume())
    if hooks:
        H = m3d.Manifold.batch_boolean([shape_from_spec(h).to_manifold() for h in hooks], m3d.OpType.Add)
        C = C - H
    lo, hi = np.asarray(cover_mesh.bounds)
    blo, bhi = np.asarray(base_mesh.bounds)
    ext_n = lambda a_, b_: float(np.abs(n) @ (np.asarray(b_) - np.asarray(a_)))  # noqa: E731
    travel = float(travel or ext_n(blo, bhi) + ext_n(lo, hi))
    worst, at = 0.0, 0.0
    for d in np.arange(step, travel + step, step):
        v = float((B ^ C.translate(tuple(d * n))).volume())
        if v > worst:
            worst, at = v, float(d)
    return {"closedOverlap": round(closed, 4), "liftOverlap": round(worst, 4), "liftOverlapAt": at,
            "ok": closed < 1.0 and worst < 0.5}


def run_split(study_path: str, out_dir: str, split: dict | None = None, base_dir: str | None = None) -> dict:
    """Read a study (with a `split` block, or give one), write
    <out>/<name>-base.study.json, <name>-cover.study.json and split.json;
    every mesh either names is copied into <out>/cad/."""
    study = json.load(open(study_path))
    base = base_dir or os.path.dirname(os.path.abspath(study_path))
    os.makedirs(out_dir, exist_ok=True)
    r = apply_split(study, split, out_dir=out_dir, base_dir=base)
    paths = {}
    for side, st in r["studies"].items():
        st = json.loads(json.dumps(st))
        _localize(st, out_dir, base)
        p = os.path.join(out_dir, f"{st['name']}.study.json")
        json.dump(st, open(p, "w"), indent=1)
        paths[side] = p
        r["studies"][side] = st
    json.dump(r["report"], open(os.path.join(out_dir, "split.json"), "w"), indent=1, default=float)
    r["studyPaths"] = paths
    return r


def _localize(study, out_dir, base_dir):
    """Point every mesh at a copy in <out>/cad/ (the lip meshes are there
    already), so each half's study is self-contained."""
    import shutil
    cad = os.path.join(out_dir, "cad")
    os.makedirs(cad, exist_ok=True)
    done = {}

    def walk(v):
        if isinstance(v, dict):
            if v.get("type") == "mesh" and isinstance(v.get("path"), str) and not v["path"].startswith("cad/"):
                src = v["path"] if os.path.isabs(v["path"]) else os.path.join(base_dir or ".", v["path"])
                src = os.path.abspath(src)
                if src not in done:
                    stem, ext = os.path.splitext(os.path.basename(src))
                    name, k = stem + ext, 1
                    while os.path.exists(os.path.join(cad, name)) and \
                            os.path.abspath(os.path.join(cad, name)) != src:
                        name = f"{stem}-{k}{ext}"
                        k += 1
                    if os.path.abspath(os.path.join(cad, name)) != src:
                        shutil.copyfile(src, os.path.join(cad, name))
                    done[src] = name
                v["path"] = f"cad/{done[src]}"
            for x in v.values():
                walk(x)
        elif isinstance(v, list):
            for x in v:
                walk(x)
    walk(study)
