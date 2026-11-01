"""
Multi-material topology optimization: the optimizer decides where each
material goes, as well as where material goes at all.

A multi-material FDM printer (Bambu AMS, Prusa MMU / XL, an IDEX) or a
material jetter prints a stiff plastic and a rubber in one part. The
interesting designs use each where it pays: a PETG or PA-CF skeleton that
carries the load and a TPU pad that conforms to what it grips, or a bumper
that turns a knock into a soft, long stop instead of a peak force the
bracket cannot take. This module optimizes that layout.

Design variables, per cell: a density x (material or void, as in simp.py)
and N - 1 material fractions y (stick-breaking: w_0 = y_0, w_1 = (1 - y_0)
y_1, ..., the last material takes what is left), so the N weights w_i are
non-negative and sum to one. Both are cone-filtered and projected with the
same Heaviside continuation, which makes the material map crisp as well as
the topology. The cell's Lame parameters are

    lambda_e = lambda_min + x^p sum_i w_i^p lambda_i     (and mu the same)

so every material keeps its own Poisson's ratio (TPU's 0.45 is not PETG's
0.38, and a pad's indentation depends on it): fem.hex8_ke is linear in
(lambda, mu), so the element matrix is lambda_e K_lambda + mu_e K_mu and
one assembly carries a different material in every cell. (With the
condensed bubble element fem.py uses, K is homogeneous in (lambda, mu)
rather than linear: K_lambda and K_mu are then its derivatives at the cell's
Lame ratio, tabulated and interpolated, LameStiffness.) The power on w
makes a 50/50 blend cost the material of both and give the stiffness of
neither, so the optimizer picks one. Mass is linear: rho_e = x sum w_i rho_i.

What a design can be asked for (MMA, mma.py, carries the constraints):

* the usual: compliance of the load cases, at a volume fraction (or the
  lightest that meets a safety factor / deflection, by a search like
  study.py's);
* how much of each material: `maxFraction` / `minFraction` of the design;
* soft where it has to be. A knock case (`"knock": {"energy": J,
  "maxForce": N}`) wants the part soft at the knock point: a mass that hits
  with energy E stops on the part as on a spring, with peak force
  F = sqrt(2 E / c), c the part's compliance there (mm/N), so the peak
  force limit is a lower limit on c (impact.py's solid_impact, as a
  constraint). An indentation case (`"indentation": {"relativeTo": case,
  "atLeast": mm}`) asks a small patch to sink at least that far more under
  its own load than the same patch moves when the same force is spread
  over the pad (relativeTo): the grip pad's conformity, read at the patch
  so the finger's bending there cancels;
* interfaces that hold. Two materials meet at a bond that is weaker than
  either (interface_library.py sources it per pair: PETG/TPU bonds,
  PLA/PETG barely does). The traction across each interface (normal pull
  and shear, on the interface's own normal, from the material field's
  gradient) is checked against the pair's strength. Once the layout has
  converged, a bond stage freezes the topology and moves only the materials
  to take stress off the bond. Verification then reads the bond on the
  delivered voxel model face by face. With `interlock`, the materials
  interpenetrate in a beam lattice where they meet (interlock.py) and the
  bond is credited with the measured gain of an interlocked joint.

Delivery: the solid is meshed by reconstruct.py as any study is, then cut
into one body per material with exact booleans (manifold3d) against a
mesh of the material field, so neighbouring bodies share their interface
surface exactly: no gaps, no overlap, which is what a slicer needs to give
each body its own extruder. Outputs are one STL per material and a
multi-material 3MF (threemf.write_multimaterial_3mf).

The analysis is linear elastic. TPU at small strain is a fair spring for a
pad or a bumper's first millimetres; large strokes belong to nonlinear.py.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage

from .fem import CORNERS, HexMesh, Solver, Stiffness, _b_matrix
from .materials import Material, material_from_spec

__all__ = ["MultiMaterial", "Requirement", "parse_multimaterial", "optimize_multimaterial",
           "run_multimaterial", "verify_multimaterial", "split_bodies", "lame", "LameStiffness",
           "MultiMaterialResult", "write_bodies_3mf"]

PENALTY = 3.0
E_FLOOR = 1e-6          # void stiffness, as a fraction of the stiffest material
_CONN26 = np.ones((3, 3, 3), bool)


# ---------------------------------------------------------------- parsing

@dataclass
class Requirement:
    """A load case that asks the part to be soft somewhere."""
    case: str
    kind: str                       # "knock" | "indentation"
    energy: float | None = None     # J (knock)
    max_force: float | None = None  # N (knock)
    relative_to: str | None = None  # case name (indentation)
    at_least: float | None = None   # mm (indentation)
    safety_factor: float | None = None  # on the stored energy's stress (knock)

    def to_dict(self):
        if self.kind == "knock":
            return {"case": self.case, "knock": {"energy": self.energy, "maxForce": self.max_force}}
        return {"case": self.case, "indentation": {"relativeTo": self.relative_to, "atLeast": self.at_least}}


@dataclass
class MultiMaterial:
    materials: list                                  # Material, at least two
    assign: list = field(default_factory=list)       # (kind, payload, material index)
    max_fraction: dict = field(default_factory=dict)  # index -> fraction of the solid
    min_fraction: dict = field(default_factory=dict)
    requirements: list = field(default_factory=list)
    interface_safety_factor: float | None = None
    interfaces: dict = field(default_factory=dict)   # "a+b" -> {"tensile", "shear"} override
    interface_penalty: bool = True
    extruders: dict = field(default_factory=dict)    # material name -> extruder (1-based)
    slicer: str = "all"
    interlock: object = None                         # interlock.Interlock, or None: flat bonds
    bond_in_topology: bool = False                   # experimental: bond constrained from beta 4

    @property
    def names(self):
        return [m.name for m in self.materials]

    def index(self, name):
        key = str(name).strip().lower()
        for i, m in enumerate(self.materials):
            if m.name.lower() == key:
                return i
        try:
            want = material_from_spec(name).name
        except ValueError:
            want = None
        for i, m in enumerate(self.materials):
            if want is not None and m.name == want:
                return i
        raise ValueError(f"multiMaterial: {name!r} is not one of the materials {self.names}")


def parse_multimaterial(d, study=None, base_dir=None) -> MultiMaterial | None:
    """The study's `multiMaterial` block (see schema.py), or None."""
    if d is None or d is False:
        return None
    if not isinstance(d, dict):
        raise ValueError("multiMaterial is an object: {materials: [..], assign?, maxFraction?, ...}")
    mats_in = d.get("materials")
    if not isinstance(mats_in, list) or len(mats_in) < 2:
        raise ValueError("multiMaterial.materials lists two to four materials (names or custom objects)")
    if len(mats_in) > 4:
        raise ValueError("multiMaterial takes at most four materials")
    mats = [material_from_spec(m) for m in mats_in]
    if len({m.name for m in mats}) != len(mats):
        raise ValueError("multiMaterial.materials has the same material twice")
    mm = MultiMaterial(mats)
    for i, a in enumerate(d.get("assign") or []):
        if not isinstance(a, dict) or "material" not in a:
            raise ValueError(f"multiMaterial.assign[{i}] needs a material and a region, shape or near")
        k = mm.index(a["material"])
        if "region" in a:
            mm.assign.append(("region", str(a["region"]), k))
        elif "shape" in a:
            from .shapes import shape_from_spec
            mm.assign.append(("shape", shape_from_spec(a["shape"], base_dir), k))
        elif "near" in a:
            n = a["near"]
            if not isinstance(n, dict) or "distance" not in n or not ("region" in n or "shape" in n):
                raise ValueError(f"multiMaterial.assign[{i}].near is {{region or shape, distance (mm)}}")
            if float(n["distance"]) <= 0:
                raise ValueError(f"multiMaterial.assign[{i}].near.distance must be > 0 (mm)")
            src = ("region", str(n["region"])) if "region" in n else ("shape", n["shape"])
            within = a.get("within")
            mm.assign.append(("near", (src, float(n["distance"]), within), k))
        else:
            raise ValueError(f"multiMaterial.assign[{i}] needs region, shape or near")
    for key, dest in (("maxFraction", mm.max_fraction), ("minFraction", mm.min_fraction)):
        for name, v in (d.get(key) or {}).items():
            if not (0.0 < float(v) <= 1.0):
                raise ValueError(f"multiMaterial.{key}.{name} is a fraction of the part's volume (0..1]")
            dest[mm.index(name)] = float(v)
    for i, r in enumerate(d.get("requirements") or []):
        if not isinstance(r, dict) or "case" not in r:
            raise ValueError(f"multiMaterial.requirements[{i}] needs a case and knock or indentation")
        if "knock" in r:
            k = r["knock"]
            if float(k.get("energy", 0)) <= 0 or float(k.get("maxForce", 0)) <= 0:
                raise ValueError(f"multiMaterial.requirements[{i}].knock needs energy (J) and maxForce (N) > 0")
            mm.requirements.append(Requirement(str(r["case"]), "knock", energy=float(k["energy"]),
                                               max_force=float(k["maxForce"]),
                                               safety_factor=float(k["safetyFactor"]) if k.get("safetyFactor")
                                               else None))
        elif "indentation" in r:
            k = r["indentation"]
            if "relativeTo" not in k or float(k.get("atLeast", 0)) <= 0:
                raise ValueError(f"multiMaterial.requirements[{i}].indentation needs relativeTo (a case) and "
                                 "atLeast (mm) > 0")
            mm.requirements.append(Requirement(str(r["case"]), "indentation", relative_to=str(k["relativeTo"]),
                                               at_least=float(k["atLeast"])))
        else:
            raise ValueError(f"multiMaterial.requirements[{i}] needs knock or indentation")
    if d.get("interfaceSafetyFactor") is not None:
        mm.interface_safety_factor = float(d["interfaceSafetyFactor"])
        if mm.interface_safety_factor <= 0:
            raise ValueError("multiMaterial.interfaceSafetyFactor must be positive")
    for pair, v in (d.get("interfaces") or {}).items():
        if not isinstance(v, dict) or "tensile" not in v:
            raise ValueError(f"multiMaterial.interfaces.{pair} is {{tensile (MPa), shear? (MPa)}}")
        names = [s.strip() for s in str(pair).split("+")]
        if len(names) != 2:
            raise ValueError(f"multiMaterial.interfaces key {pair!r} is 'a+b'")
        a, b = sorted(mm.materials[mm.index(n)].name for n in names)
        c = {"tensile": float(v["tensile"]), "shear": float(v.get("shear", v["tensile"])), "source": "study"}
        for k_ in ("tensileZ", "shearZ"):
            if k_ in v:
                c[k_] = float(v[k_])
        mm.interfaces[f"{a}+{b}"] = c
    mm.interface_penalty = bool(d.get("interfacePenalty", True))
    mm.bond_in_topology = bool(d.get("bondInTopology", False))
    if d.get("interlock") not in (None, False):
        from .interlock import parse_interlock
        mm.interlock = parse_interlock(d["interlock"], getattr(study, "process", None))
        for p in mm.interlock.pairs or []:
            for n_ in p.split("+"):
                mm.index(n_)
    ex = d.get("export") or {}
    for name, e in (ex.get("extruders") or {}).items():
        mm.extruders[mats[mm.index(name)].name] = int(e)
    mm.slicer = str(ex.get("slicer", "all"))
    if study is not None:
        names = [c.name for c in study.load_cases]
        for r in mm.requirements:
            if r.case not in names:
                raise ValueError(f"multiMaterial requirement names case {r.case!r}; cases: {names}")
            if r.relative_to is not None and r.relative_to not in names:
                raise ValueError(f"multiMaterial indentation relativeTo {r.relative_to!r}; cases: {names}")
        for kind, payload, _ in mm.assign:
            if kind == "region":
                study.region_shape(payload)
    return mm


# ------------------------------------------------------------ interfaces

def interface_card(mm: MultiMaterial, i: int, j: int, process=None, interlock: bool = True) -> dict:
    """Bond strength (MPa) between materials i and j: the study's override,
    else interface_library's sourced or conservative card; with the
    study's interlock credited (interlock.interlocked_card) unless
    `interlock` is False."""
    card = _flat_card(mm, i, j, process)
    il = mm.interlock
    if interlock and il is not None and il.applies(mm.materials[i].name, mm.materials[j].name):
        from .interlock import interlocked_card
        return interlocked_card(card, mm.materials[i], mm.materials[j], il)
    return card


def _flat_card(mm, i, j, process=None) -> dict:
    a, b = sorted((mm.materials[i].name, mm.materials[j].name))
    if f"{a}+{b}" in mm.interfaces:
        c = dict(mm.interfaces[f"{a}+{b}"])
        c.setdefault("tensileZ", c["tensile"])
        c.setdefault("shearZ", c["shear"])
        return c
    try:
        from .interface_library import interface_strength
        card = interface_strength(mm.materials[i], mm.materials[j], getattr(process, "kind", process))
        return {"tensile": card.tensile, "shear": card.shear,
                "tensileZ": card.tensile_z if card.tensile_z is not None else card.tensile,
                "shearZ": card.shear_z if card.shear_z is not None else card.shear,
                "bond": card.bond, "source": card.source, "conservative": card.conservative}
    except (ImportError, ValueError):
        # no card: a quarter of the weaker material's across-layer strength
        weak = min((m.strength_z or m.strength) for m in (mm.materials[i], mm.materials[j]))
        return {"tensile": 0.25 * weak, "shear": 0.25 * weak, "tensileZ": 0.25 * weak, "shearZ": 0.25 * weak,
                "bond": "unknown", "source": "default", "conservative": True}


# ------------------------------------------------------- per-cell elastics

def lame(E, nu):
    E, nu = np.asarray(E, float), np.asarray(nu, float)
    return E * nu / ((1 + nu) * (1 - 2 * nu)), E / (2 * (1 + nu))


def _ke_parts(q: float = 0.5, element: str | None = None):
    """Unit-cube hex8 stiffness split as K = lambda K_l + mu K_m, at the
    Lame ratio q = lambda / (lambda + mu).

    For the plain trilinear hex8 K is linear in (lambda, mu) and the split
    does not depend on q. The element fem.py uses (fem.hex8_element) has
    its bubble modes condensed out, which divides by C, so K is only
    homogeneous of degree one: K_l and K_m are its partial derivatives at
    the cell's ratio (Euler: lambda dK/dlambda + mu dK/dmu = K exactly)."""
    from .fem import _b_bubble, hex8_element

    element = element or hex8_element()
    cl = np.zeros((6, 6))
    cl[:3, :3] = 1.0
    cm = np.diag([2.0, 2.0, 2.0, 1.0, 1.0, 1.0])
    g = 1.0 / np.sqrt(3.0)
    parts = {}
    for name, c in (("l", cl), ("m", cm)):
        kuu, kua, kaa = np.zeros((24, 24)), np.zeros((24, 9)), np.zeros((9, 9))
        for xi in (-g, g):
            for eta in (-g, g):
                for zeta in (-g, g):
                    b = _b_matrix(xi, eta, zeta)
                    ba = _b_bubble(xi, eta, zeta)
                    kuu += b.T @ c @ b * 0.125
                    kua += b.T @ c @ ba * 0.125
                    kaa += ba.T @ c @ ba * 0.125
        parts[name] = (kuu, kua, kaa)
    if element == "plain":
        kl, km = parts["l"][0], parts["m"][0]
    else:
        kua = q * parts["l"][1] + (1 - q) * parts["m"][1]
        kaa = q * parts["l"][2] + (1 - q) * parts["m"][2]
        r = -np.linalg.solve(kaa, kua.T)
        # dK = dKuu + dKua R + R^T dKau + R^T dKaa R, R = -Kaa^-1 Kau
        kl, km = ((p[0] + p[1] @ r + r.T @ p[1].T + r.T @ p[2] @ r) for p in (parts["l"], parts["m"]))
    return 0.5 * (kl + kl.T), 0.5 * (km + km.T)


# Lame ratios at which the split is tabulated (q = lambda / (lambda + mu),
# 0.41 at nu = 0.3, 0.9 at nu = 0.45); between them it is interpolated
# linearly, and every material's own ratio is a node, so a crisp cell's
# matrix is exact.
_Q_GRID = np.linspace(0.0, 0.98, 50)


class LameStiffness:
    """fem.Stiffness's CSR layout with a (lambda, mu) pair per element.
    `ratios` are the Lame ratios lambda / (lambda + mu) of the materials (and
    the void floor) the cells will mix, tabulated exactly."""

    def __init__(self, mesh: HexMesh, fixed: np.ndarray, ratios=()):
        from .fem import hex8_element

        self.mesh = mesh
        self.base = Stiffness(mesh, 0.3, fixed)
        self.element = hex8_element()
        if self.element == "plain":
            q = np.zeros(1)
        else:
            q = np.unique(np.concatenate([_Q_GRID, np.asarray(ratios, float).ravel()]))
        self.q = q
        self.kl, self.km = [], []
        self._blocks = []
        for qq in q:
            kl, km = _ke_parts(qq, self.element)
            self.kl.append(kl * mesh.pitch)
            self.km.append(km * mesh.pitch)
            self._blocks.append([(self.kl[-1][3 * a:3 * a + 3, 3 * b:3 * b + 3].ravel(),
                                  self.km[-1][3 * a:3 * a + 3, 3 * b:3 * b + 3].ravel())
                                 for a in range(8) for b in range(8)])
        self._lm = None

    def _bins(self, lam, mu):
        lam = np.broadcast_to(np.asarray(lam, float), (self.mesh.n_elem,))
        mu = np.broadcast_to(np.asarray(mu, float), (self.mesh.n_elem,))
        s = lam + mu
        if len(self.q) == 1:
            z = np.zeros(len(lam), np.int64)
            return lam, mu, s, z, np.zeros(len(lam)), np.ones(len(lam))
        q = np.clip(lam / s, self.q[0], self.q[-1])
        j = np.clip(np.searchsorted(self.q, q, side="right") - 1, 0, len(self.q) - 2)
        dq = self.q[j + 1] - self.q[j]
        return lam, mu, s, j, (q - self.q[j]) / dq, dq

    def _nodes(self, j, w):
        """(node, element indices, weight) for every tabulated ratio in use."""
        out = []
        for g in np.unique(np.concatenate([j, j + 1]) if len(self.q) > 1 else j):
            lo, hi = np.nonzero(j == g)[0], np.nonzero(j + 1 == g)[0] if len(self.q) > 1 else np.zeros(0, int)
            el = np.concatenate([lo, hi])
            wt = np.concatenate([1.0 - w[lo], w[hi]])
            keep = wt != 0.0
            if keep.any():
                out.append((int(g), el[keep], wt[keep]))
        return out

    def assemble(self, lam, mu, springs=None):
        st = self.base
        data = np.zeros(st.nnz)
        lam, mu, _, j, w, _ = self._bins(lam, mu)
        self._lm = (np.array(lam), np.array(mu))
        for g, el, wt in self._nodes(j, w):
            a, b = (wt * lam[el])[:, None], (wt * mu[el])[:, None]
            for (slots, _), (bl, bm) in zip(st._pairs, self._blocks[g]):
                data[slots[el]] += a * bl + b * bm
        k = st.finish(data)
        if springs is not None:
            k = (k + springs).tocsr()
        return k

    def apply(self, ue, lam=None, mu=None):
        """Per element K_e u_e (ne, 24)."""
        lam, mu = (lam, mu) if lam is not None else self._lm
        lam, mu, _, j, w, _ = self._bins(lam, mu)
        out = np.zeros_like(ue)
        for g, el, wt in self._nodes(j, w):
            out[el] += (wt * lam[el])[:, None] * (ue[el] @ self.kl[g]) + (wt * mu[el])[:, None] * (ue[el] @ self.km[g])
        return out

    def pair(self, x, y, lam=None, mu=None):
        """Per element (x^T dK/dlambda y, x^T dK/dmu y), the derivatives of
        the tabulated K (interpolation slope included), at the moduli last
        assembled unless given."""
        lam, mu = (lam, mu) if lam is not None else self._lm
        lam, mu, s, j, w, dq = self._bins(lam, mu)
        n = len(lam)
        fl = np.zeros((2, n))
        fm = np.zeros((2, n))
        for g in np.unique(j):
            el = np.nonzero(j == g)[0]
            for side in (0, 1) if len(self.q) > 1 else (0,):
                kl, km = self.kl[g + side], self.km[g + side]
                fl[side, el] = np.einsum("ij,jk,ik->i", x[el], kl, y[el])
                fm[side, el] = np.einsum("ij,jk,ik->i", x[el], km, y[el])
        if len(self.q) == 1:
            return fl[0], fm[0]
        el_ = (1 - w) * fl[0] + w * fl[1]
        em_ = (1 - w) * fm[0] + w * fm[1]
        d = (lam * (fl[1] - fl[0]) + mu * (fm[1] - fm[0])) / dq       # x^T dK/dq y
        return el_ + d * mu / s ** 2, em_ - d * lam / s ** 2

    def energies(self, u, lam=None, mu=None):
        """Per element (u^T dK/dlambda u, u^T dK/dmu u); lambda times the
        first plus mu times the second is the element's u^T K u."""
        ue = u[self.mesh.edof]
        return self.pair(ue, ue, lam, mu)

    def stress(self, u, lam, mu):
        """Element-centre stress (ne, 6), engineering shear strain (the
        bubbles have no strain at the centre)."""
        b = _b_matrix(0.0, 0.0, 0.0) / self.mesh.pitch
        eps = u[self.mesh.edof] @ b.T
        tr = eps[:, :3].sum(axis=1)
        s = np.empty_like(eps)
        s[:, :3] = 2.0 * mu[:, None] * eps[:, :3] + (lam * tr)[:, None]
        s[:, 3:] = mu[:, None] * eps[:, 3:]
        return s


def _von_mises(s):
    sx, sy, sz, txy, tyz, tzx = s.T
    return np.sqrt(0.5 * ((sx - sy) ** 2 + (sy - sz) ** 2 + (sz - sx) ** 2) + 3.0 * (txy ** 2 + tyz ** 2 + tzx ** 2))


_VOIGT_T = np.array([[0, 3, 5], [3, 1, 4], [5, 4, 2]])   # (xx yy zz xy yz zx) -> tensor


def _traction_util(s, n, card, build_axis=None):
    """Interface utilization (>= 0; 1 = at the bond's strength) of stress s
    (k, 6) on unit normals n (k, 3): (pull / S_t)^2 + (shear / S_s)^2,
    square-rooted. Compression across a bond does not open it. A normal
    along the build axis is a stacked interface (one material printed on
    the other) and takes the card's across-layer values; in between, a
    blend by n_z^2."""
    t = np.einsum("kij,kj->ki", s[:, _VOIGT_T], n)
    sn = np.einsum("ki,ki->k", t, n)
    tau = np.linalg.norm(t - sn[:, None] * n, axis=1)
    st, ss = card["tensile"], card["shear"]
    if build_axis is not None:
        c2 = n[:, build_axis] ** 2
        st = st + (card["tensileZ"] - st) * c2
        ss = ss + (card["shearZ"] - ss) * c2
    return np.sqrt((np.maximum(sn, 0.0) / st) ** 2 + (tau / ss) ** 2)


def _build_axis(study):
    pd = getattr(study, "print_direction", None)
    if pd is None and study.process is not None and study.process.build not in (None, "auto"):
        pd = study.process.build
    if pd is None:
        return 2 if (study.process is not None and study.process.kind == "fdm") else None
    if isinstance(pd, tuple) and len(pd) == 2 and isinstance(pd[0], (int, np.integer)):
        return int(pd[0])
    v = np.abs(np.asarray(pd, float).ravel())
    return int(np.argmax(v)) if v.size == 3 else 2


# ----------------------------------------------------------- weights

def _weights(y):
    """Stick-breaking: y (N-1, n) in [0,1] -> w (N, n) and dw/dy (N, N-1, n)."""
    m, n = y.shape
    N = m + 1
    w = np.empty((N, n))
    dw = np.zeros((N, m, n))
    one = 1.0 - y
    for i in range(N):
        prev = np.prod(one[:i], axis=0) if i else np.ones(n)
        w[i] = prev * (y[i] if i < m else 1.0)
        for j in range(min(i + 1, m)):
            if j < i:
                others = np.prod(np.delete(one[:i], j, axis=0), axis=0) if i > 1 else np.ones(n)
                dw[i, j] = -others * (y[i] if i < m else 1.0)
            else:   # j == i < m
                dw[i, j] = prev
    return w, dw


def _y_for(k, N):
    """The y that puts all of a cell into material k."""
    y = np.zeros(N - 1)
    if k < N - 1:
        y[k] = 1.0
    return y


def _project(xt, beta, eta=0.5):
    den = np.tanh(beta * eta) + np.tanh(beta * (1 - eta))
    return (np.tanh(beta * eta) + np.tanh(beta * (xt - eta))) / den


def _dproject(xt, beta, eta=0.5):
    den = np.tanh(beta * eta) + np.tanh(beta * (1 - eta))
    return beta * (1 - np.tanh(beta * (xt - eta)) ** 2) / den


# ------------------------------------------------------------ optimizer

def assignment(study, domain, mm: MultiMaterial) -> np.ndarray:
    """Per cell the material index the study fixes there, or -1."""
    from .domain import _fraction

    centers = domain.mesh.elem_centers()
    out = np.full(domain.n_elem, -1, np.int64)
    for kind, payload, k in mm.assign:
        if kind == "region":
            hit = _fraction([study.region_shape(payload)[0]], centers, domain.pitch) >= 0.5
        elif kind == "shape":
            hit = _fraction([payload], centers, domain.pitch) >= 0.5
        else:
            (src_kind, src), dist, within = payload
            if src_kind == "region":
                shape = study.region_shape(src)[0]
            else:
                from .shapes import shape_from_spec
                shape = shape_from_spec(src)
            hit = shape.sdf(centers) <= dist
            if within is not None:
                hit &= study.region_shape(str(within))[0].contains(centers)
        out[hit & (out < 0)] = k
    return out


def _case_force(f):
    return float(np.linalg.norm([f[0::3].sum(), f[1::3].sum(), f[2::3].sum()]))


@dataclass
class MMOptimum:
    x: np.ndarray              # design density variables, per cell
    y: np.ndarray              # (N-1, n) material variables
    xphys: np.ndarray          # projected density
    w: np.ndarray              # (N, n) projected material weights
    compliance: list
    iterations: int
    seconds: float
    history: list
    interface_weight: float
    requirement_values: dict


def optimize_multimaterial(study, domain, mm: MultiMaterial, volume_fraction: float, radius: float,
                           max_iterations: int | None = None, x0=None, y0=None, beta_max: float = 16.0,
                           on_iter=None, should_stop=None) -> MMOptimum:
    from .mma import MMA
    from .simp import _symmetrize, density_filter

    t0 = time.time()
    N = len(mm.materials)
    n = domain.n_elem
    mesh = domain.mesh
    max_iterations = int(max_iterations or study.max_iterations)
    Es = np.array([m.E for m in mm.materials])
    lam_i, mu_i = lame(Es, np.array([m.nu for m in mm.materials]))
    rho_i = np.array([m.density for m in mm.materials])
    e_max = Es.max()
    lam_min, mu_min = lame(E_FLOOR * e_max, 0.3)
    springs = e_max * domain.springs if domain.springs is not None else None
    stiff = LameStiffness(mesh, domain.fixed, ratios=np.concatenate([lam_i / (lam_i + mu_i),
                                                                     [lam_min / (lam_min + mu_min)]]))
    solver = Solver(mesh, rtol=1e-6, maxiter=2000)
    h, hs = density_filter(domain, radius)
    hs_y = np.asarray(h @ np.ones(n)).ravel()
    pres = domain.preserve
    design = ~pres
    fixed_mat = assignment(study, domain, mm)
    free_y = fixed_mat < 0
    nx, ny = int(design.sum()), int(free_y.sum())
    m_y = N - 1
    names = list(domain.case_names)
    loads = list(domain.loads)
    forces = [_case_force(f) for f in loads]
    req_cases = {r.case for r in mm.requirements}
    weights = [0.0 if names[c] in req_cases else float(domain.case_weights[c]) for c in range(len(loads))]
    if not any(weights):
        raise ValueError("multiMaterial: every load case is a softness requirement; one has to be the stiffness "
                         "objective")
    build_axis = _build_axis(study)
    pairs = [(i, j) for i in range(N) for j in range(i + 1, N)]
    cards = {p: interface_card(mm, *p, study.process) for p in pairs}
    flat_cards = {p: interface_card(mm, *p, study.process, interlock=False) for p in pairs}
    sf_int = mm.interface_safety_factor or (study.goal.safety_factor if study.goal.safety_factor else 2.0)
    # a knock with its own safetyFactor is judged at it (an accidental
    # load, at its peak), every other case at the interface's
    case_sf = [sf_int] * len(domain.loads)
    for r in mm.requirements:
        if r.kind == "knock" and r.safety_factor:
            case_sf[list(domain.case_names).index(r.case)] = r.safety_factor
    mirror = domain.mirror or {}

    # the y a fixed cell holds
    y_fixed = np.zeros((m_y, n))
    for k in range(N):
        cells = fixed_mat == k
        if cells.any():
            y_fixed[:, cells] = _y_for(k, N)[:, None]
    x = np.full(n, volume_fraction) if x0 is None else np.asarray(x0, float).copy()
    x[pres] = 1.0
    if y0 is None:
        y = np.vstack([np.full(n, 1.0 / (N - i)) for i in range(m_y)])
    else:
        y = np.asarray(y0, float).reshape(m_y, n).copy()
    y[:, ~free_y] = y_fixed[:, ~free_y]

    nvar = nx + m_y * ny
    # constraints: volume, material caps/floors, requirements
    cons_names = ["volume"] + [f"max:{mm.materials[k].name}" for k in mm.max_fraction] + \
                 [f"min:{mm.materials[k].name}" for k in mm.min_fraction] + [f"req:{r.case}" for r in mm.requirements] + \
        [f"energy:{r.case}" for r in mm.requirements if r.kind == "knock"]
    use_int = bool(mm.interface_penalty)
    if use_int:
        cons_names.insert(len(cons_names) - sum(r.kind == "knock" for r in mm.requirements), "bondStage:stiffness")
    # stress scale per case: a knock is judged at its peak force (updated each solve)
    scales = [1.0] * len(loads)
    # what a cell can store elastically before it fails, mJ/mm^3: S^2 / 2E.
    # TPU stores ~6x what PETG does per volume, which is why bumpers are TPU.
    resil = np.array([m.strength ** 2 / (2.0 * m.E) for m in mm.materials])
    def new_mma(nvar=nvar, move=0.15):
        return MMA(nvar, len(cons_names), move=move)

    opt = new_mma()
    beta = 1.0 if x0 is None else 4.0
    beta_every = max(8, max_iterations // 6)
    us = [None] * len(loads)
    c0 = None
    history = []
    gamma, p0 = 0.0, None
    # the cases a softness requirement reads, solved on the Voigt model
    req_idx = sorted({names.index(r.case) for r in mm.requirements} |
                     {names.index(r.relative_to) for r in mm.requirements if r.relative_to})
    solver_v = Solver(mesh, rtol=1e-6, maxiter=2000)
    uv = {}
    # The stored-energy limit skips the cells that touch the knock's loaded
    # nodes or a support: a load spread over a face's nodes concentrates
    # there as no real contact does (verification leaves the same band out).
    # Without this the limit could never be met under a face load, and MMA,
    # chasing it at full projection, emptied the bracket.
    away = {}
    for r in mm.requirements:
        if r.kind == "knock":
            ci = names.index(r.case)
            nodes = np.abs(loads[ci].reshape(-1, 3)).sum(axis=1) > 0
            nodes |= domain.support_nodes
            away[ci] = (~nodes[mesh.enode].any(axis=1)).astype(float)
    frozen = False
    f0_frozen = None
    beta_max_at = 0 if beta >= beta_max else None
    stage_b_start = None
    # iterations kept for the bond stage at the end
    stage_b = max_iterations - max(10, max_iterations // 5) if use_int else max_iterations
    s_lag = None
    it = 0
    req_vals = {}
    for it in range(1, max_iterations + 1):
        xt = (h @ x) / hs
        xb = _project(xt, beta)
        dxb = _dproject(xt, beta)
        xb[pres], dxb[pres] = 1.0, 0.0
        yt = np.vstack([(h @ y[j]) / hs_y for j in range(m_y)])
        yb = _project(yt, beta)
        dyb = _dproject(yt, beta)
        yb[:, ~free_y], dyb[:, ~free_y] = y_fixed[:, ~free_y], 0.0
        w, dw = _weights(yb)
        wp = w ** PENALTY
        xpp = xb ** PENALTY
        lam_e = lam_min + xpp * (wp.T @ lam_i)
        mu_e = mu_min + xpp * (wp.T @ mu_i)
        k = stiff.assemble(lam_e, mu_e, springs)
        us = solver.solve_many(k, loads, us)
        comp = [float(f @ u) for f, u in zip(loads, us)]
        if c0 is None:
            c0 = [max(c, 1e-30) for c in comp]
        for r in mm.requirements:
            if r.kind == "knock":
                ci = names.index(r.case)
                scales[ci] = float(np.sqrt(2.0 * r.energy * 1000.0 * comp[ci] / max(comp[ci], 1e-30) ** 2))
        # d(lam_e)/d xb, d(lam_e)/d w_i
        dlam_dx = PENALTY * xb ** (PENALTY - 1) * (wp.T @ lam_i)
        dmu_dx = PENALTY * xb ** (PENALTY - 1) * (wp.T @ mu_i)
        dwp = PENALTY * w ** (PENALTY - 1)                     # (N, n)
        energies = [stiff.energies(u, lam_e, mu_e) for u in us]

        def grad_c(ci):
            """dC_ci / d(xb) and / d(yb) (m_y, n)."""
            el, em = energies[ci]
            gx = -(el * dlam_dx + em * dmu_dx)
            gw = -(xpp * dwp) * (el[None, :] * lam_i[:, None] + em[None, :] * mu_i[:, None])   # (N, n)
            gy = np.einsum("in,ijn->jn", gw, dw)
            return gx, gy

        grads = [grad_c(ci) for ci in range(len(loads))]
        f0 = sum(wc * c / c0i for wc, c, c0i in zip(weights, comp, c0))
        gx0 = sum(wc / c0i * g[0] for wc, c0i, g in zip(weights, c0, grads))
        gy0 = sum(wc / c0i * g[1] for wc, c0i, g in zip(weights, c0, grads))

        # interfaces: a lagged constraint. The bond utilization of every cell
        # (traction on the interface normal, from the last solve) is held
        # fixed; the constraint is its p-mean over the interface band
        # (x * 4 w_i w_j), so the optimizer meets it by moving the interface
        # to where the bond is not loaded, or by having less of it there.
        int_g = None
        if use_int and (frozen or (mm.bond_in_topology and beta >= 4.0)):
            # every case: a softness case is a real load too (the poke's
            # edge presses on the pad's bond as hard as anything)
            stress = [stiff.stress(u, lam_e, mu_e) * sc for u, sc in zip(us, scales)]
            normals = _material_normals(domain, w)
            num, den = 0.0, 0.0
            gnum_x, gden_x = np.zeros(n), np.zeros(n)
            gnum_w, gden_w = np.zeros((N, n)), np.zeros((N, n))
            worst = 0.0
            # the interlock's lattice stays off the outer surface: there the
            # bond is flat, as verification judges it
            shallow = None
            if mm.interlock is not None:
                sg = domain.full_grid((xb > 0.5).astype(float)) > 0.5
                dep = (ndimage.distance_transform_edt(np.pad(sg, 1))[1:-1, 1:-1, 1:-1] - 0.5) * domain.pitch
                shallow = dep[tuple(mesh.elem_ijk.T)] < (mm.interlock.avoid or mm.interlock.beam_width)
            for p in pairs:
                util = np.zeros(n)
                for s_, sfc in zip(stress, case_sf):
                    u_ = _traction_util(s_, normals[p], cards[p], build_axis)
                    if shallow is not None and "interlock" in cards[p] and shallow.any():
                        u_[shallow] = _traction_util(s_[shallow], normals[p][shallow], flat_cards[p], build_axis)
                    util = np.maximum(util, sfc * u_)
                uq = util ** PNORM
                i, j = p
                band = 4.0 * w[i] * w[j]
                num += float((xb * band * uq).sum())
                den += float((xb * band).sum())
                gnum_x += band * uq
                gden_x += band
                gnum_w[i] += xb * 4.0 * w[j] * uq
                gnum_w[j] += xb * 4.0 * w[i] * uq
                gden_w[i] += xb * 4.0 * w[j]
                gden_w[j] += xb * 4.0 * w[i]
                sel = band > 0.5
                if sel.any():
                    worst = max(worst, float(util[sel].max()) * sf_int)
            if den > 1e-6:
                Pm = (num / den) ** (1.0 / PNORM)
                dP = Pm / (PNORM * num) if num > 0 else 0.0
                # ln P: a bond many times over its strength must not swamp
                # the other constraints' scale in MMA
                dP = dP / max(Pm, 1e-12)
                int_g = (float(np.log(max(Pm, 1e-12))),
                         dP * (gnum_x - num / den * gden_x),
                         np.einsum("in,ijn->jn", dP * (gnum_w - num / den * gden_w), dw))
            gamma = worst
        else:
            worst = None

        # constraints in (xb, yb) space
        g_vals, g_x, g_y = [], [], []
        ntot = float(n)
        g_vals.append(xb.mean() / volume_fraction - 1.0)
        g_x.append(np.full(n, 1.0 / (ntot * volume_fraction)))
        g_y.append(np.zeros((m_y, n)))
        solid = max(float(xb.sum()), 1e-9)
        for kk, cap in mm.max_fraction.items():
            v = float((xb * w[kk]).sum())
            g_vals.append(v / (cap * solid) - 1.0)
            g_x.append((w[kk] * solid - v) / (cap * solid ** 2))
            g_y.append(np.einsum("n,jn->jn", xb / (cap * solid), dw[kk]))
        for kk, lo in mm.min_fraction.items():
            v = float((xb * w[kk]).sum())
            g_vals.append(1.0 - v / (lo * solid))
            g_x.append(-(w[kk] * solid - v) / (lo * solid ** 2))
            g_y.append(-np.einsum("n,jn->jn", xb / (lo * solid), dw[kk]))
        # Softness on the penalized model (as the objective) by default. The
        # Voigt bound (density and material mixed linearly, so grey is as
        # stiff as it could be) was tried so grey core could not fake a soft
        # pad: on the bracket it met the knock by geometry instead and left a
        # bolt boss floating, so it is off (VOIGT_SOFTNESS).
        if req_idx and VOIGT_SOFTNESS:
            lam_v = lam_min + xb * (w.T @ lam_i)
            mu_v = mu_min + xb * (w.T @ mu_i)
            kv = stiff.assemble(lam_v, mu_v, springs)
            got = solver_v.solve_many(kv, [loads[c] for c in req_idx], [uv.get(c) for c in req_idx])
            uv = dict(zip(req_idx, got))
            dlv_dx, dmv_dx = w.T @ lam_i, w.T @ mu_i
            wfac, sol_v = np.broadcast_to(xb, w.shape), solver_v
        elif req_idx:
            lam_v, mu_v, kv, sol_v = lam_e, mu_e, k, solver
            uv = {c: us[c] for c in req_idx}
            dlv_dx, dmv_dx, wfac = dlam_dx, dmu_dx, xpp * dwp
        if req_idx:

            def grad_v(ca, cb):
                """d(f_ca . u_cb) on the Voigt model = -u_ca^T dK u_cb."""
                ua, ub = uv[ca][mesh.edof], uv[cb][mesh.edof]
                el, em = stiff.pair(ua, ub, lam_v, mu_v)
                gx = -(el * dlv_dx + em * dmv_dx)
                gw = -wfac * (el[None, :] * lam_i[:, None] + em[None, :] * mu_i[:, None])
                return gx, np.einsum("in,ijn->jn", gw, dw)

        for r in mm.requirements:
            ci = names.index(r.case)
            F = max(forces[ci], 1e-12)
            cv = float(loads[ci] @ uv[ci])
            if r.kind == "knock":
                # c = C / F^2 (mm/N) >= 2 E / Fmax^2 ; E in J = 1000 N mm
                cmin = 2.0 * r.energy * 1000.0 / r.max_force ** 2
                c = cv / F ** 2
                req_vals[r.case] = {"compliance": c, "peakForce": float(np.sqrt(2.0 * r.energy * 1000.0 / c))}
                gcx, gcy = grad_v(ci, ci)
                g_vals.append(1.0 - c / cmin)
                g_x.append(-gcx / (F ** 2 * cmin))
                g_y.append(-gcy / (F ** 2 * cmin))
            else:
                ri = names.index(r.relative_to)
                Fr = max(forces[ri], 1e-12)
                # how far the patch sinks under its own load, less how far the
                # same patch moves when the force is spread (relativeTo) and
                # scaled to the same size: both read at the patch, so the
                # finger's bending there cancels. (Comparing with the spread
                # case's own mean displacement mixed in the tip's larger
                # bending.)
                mut = float(loads[ci] @ uv[ri])
                dind = cv / F - mut / Fr
                req_vals[r.case] = {"indentation": dind}
                gcx, gcy = grad_v(ci, ci)
                gmx, gmy = grad_v(ci, ri)
                tgt = INDENT_MARGIN * r.at_least
                g_vals.append(1.0 - dind / tgt)
                g_x.append(-(gcx / F - gmx / Fr) / tgt)
                g_y.append(-(gcy / F - gmy / Fr) / tgt)
        if use_int:
            if int_g is not None and not frozen:
                # experimental (bondInTopology): the bond as a constraint while
                # the layout forms, now that the stored-energy limit keeps MMA
                # from cutting the load path to unload it
                g_vals.append(int_g[0])
                g_x.append(int_g[1])
                g_y.append(int_g[2])
            elif int_g is None or f0_frozen is None:
                g_vals.append(-1.0)
                g_x.append(np.zeros(n))
                g_y.append(np.zeros((m_y, n)))
            else:
                # the bond stage: the bond's p-mean is the objective, and the
                # stiffness objective may get at most STIFFNESS_GIVE worse
                lim = f0_frozen * (1.0 + STIFFNESS_GIVE)
                g_vals.append(f0 / lim - 1.0)
                g_x.append(gx0 / lim)
                g_y.append(gy0 / lim)
                f0, gx0, gy0 = int_g
        for r in mm.requirements:
            if r.kind != "knock":
                continue
            ci = names.index(r.case)
            gv, gxx, gyy, peak_r = _energy_constraint(
                stiff, sol_v, kv, loads[ci], uv[ci], float(loads[ci] @ uv[ci]), r.energy * 1000.0, lam_v, mu_v,
                xb, w, dlv_dx, dmv_dx, *((xb, np.ones_like(w)) if VOIGT_SOFTNESS else (xpp, dwp)), dw, lam_i, mu_i, resil,
                (r.safety_factor or sf_int) ** 2, max(volume_fraction * n, 1.0), away[ci])
            req_vals[r.case]["energyUtilization"] = peak_r
            # as ln P, for the same reason as the interface
            pv = max(gv + 1.0, 1e-12)
            g_vals.append(float(np.log(pv)))
            g_x.append(gxx / pv)
            g_y.append(gyy / pv)

        # chain to the variables: projection, then the filter's transpose
        def to_vars(gx, gy):
            vx = (h.T @ (gx * dxb / hs))[design]
            vy = np.concatenate([(h.T @ (gy[j] * dyb[j] / hs_y))[free_y] for j in range(m_y)]) if m_y else []
            return np.concatenate([vx, vy])

        df0 = to_vars(gx0, gy0)
        df = np.vstack([to_vars(a, b) for a, b in zip(g_x, g_y)])
        if frozen:
            df0, df = df0[nx:], df[:, nx:]
        rec = {"iteration": it, "beta": beta, "objective": f0, "compliance": comp,
               "volume": float(xb.mean()),
               "fractions": {mm.materials[i].name: float((xb * w[i]).sum() / solid) for i in range(N)},
               "constraints": dict(zip(cons_names, map(float, g_vals))),
               "interfaceWeight": gamma, **({"interfaceUtil": worst} if worst is not None else {}),
               **({"requirements": {k: dict(v) for k, v in req_vals.items()}} if req_vals else {})}
        history.append(rec)
        if on_iter is not None:
            on_iter(rec, xb)
        if should_stop is not None and should_stop():
            break
        vars_now = np.concatenate(([] if frozen else [x[design]]) + [y[j][free_y] for j in range(m_y)])
        new = opt.update(vars_now, f0, df0, np.asarray(g_vals), df)
        change = float(np.max(np.abs(new - vars_now))) if len(new) else 0.0
        off = 0 if frozen else nx
        if not frozen:
            x[design] = new[:nx]
        for j in range(m_y):
            y[j][free_y] = new[off + j * ny: off + (j + 1) * ny]
        if mirror:
            x = _symmetrize(x, mirror)
            y = np.vstack([_symmetrize(y[j], mirror) for j in range(m_y)])
        x[pres] = 1.0
        if frozen:
            if it >= max_iterations or (change < 0.005 and all(g <= 0.01 for g in g_vals)):
                break
            continue
        if (it % beta_every == 0 or change < 0.01) and beta < beta_max:
            beta = min(2.0 * beta, beta_max)
            if beta >= beta_max:
                beta_max_at = it
            opt = new_mma()   # fresh asymptotes for the sharper problem
        elif beta >= beta_max and it - beta_max_at >= 8 and \
                (it >= stage_b or (change < 0.005 and all(g <= 0.01 for g in g_vals))):
            if use_int and ny and pairs:
                # The bond stage: topology frozen, only the material map
                # moves. Constraining the bond while the topology was still
                # forming (tried first) let MMA cut a weak bond's load path
                # instead: the bracket's shelf pad came off the part, and a
                # beam's TPU tip off the beam. With the layout fixed it can
                # only slide the interface out of the stress, or trade a
                # little stiffness for less interface where the bond is loaded.
                frozen = True
                f0_frozen = f0
                opt = new_mma(nvar=m_y * ny, move=0.1)
                stage_b_start = it
            else:
                break
    # final, at the delivered projection
    xt = (h @ x) / hs
    xb = _project(xt, beta)
    xb[pres] = 1.0
    yt = np.vstack([(h @ y[j]) / hs_y for j in range(m_y)])
    yb = _project(yt, beta)
    yb[:, ~free_y] = y_fixed[:, ~free_y]
    w, _ = _weights(yb)
    out = MMOptimum(x, y, xb, w, history[-1]["compliance"] if history else [], it, time.time() - t0, history,
                    gamma, req_vals)
    out.bond_stage = stage_b_start
    return out


PNORM = 10.0
# how much stiffness the bond stage may give up to unload the interfaces
STIFFNESS_GIVE = 0.10
INDENT_MARGIN = 1.1
VOIGT_SOFTNESS = False


def _energy_constraint(stiff, solver, k, f, u, comp, energy, lam_e, mu_e, xb, w, dlam_dx, dmu_dx, xpp, dwp,
                       dw, lam_i, mu_i, resil, sf2, n_solid, mask=None):
    """A knock stops when the part has stored its energy E: at peak force
    the strain energy density in a cell is sed_e * 2E / C (the case's
    field scaled to the peak). Each cell can store R_e = x_e sum_i w_i
    S_i^2 / (2 E_i) before it fails (void stores nothing); the constraint
    is the p-norm of r_e = SF^2 * sed_peak / R_e over every cell, at most
    1. Thin necks that buy compliance by geometry fail it, and so does a
    knock carried through void (its strain is huge where nothing can hold
    it: the cut-off bumper that weighting by density let through); a TPU
    bumper, storing the knock in rubber, passes. sed in void falls as x^3
    and R as x, so empty cells away from the load read near zero.
    Returns (g, dg/dxb, dg/dyb, max r over the solid)."""
    h3 = stiff.mesh.pitch ** 3
    el, em = stiff.energies(u, lam_e, mu_e)
    sed = 0.5 * (lam_e * el + mu_e * em) / h3
    Rw = w.T @ resil
    R = np.maximum(xb * Rw, 1e-12)
    a = sf2 * 2.0 * energy / comp
    m = np.ones_like(R) if mask is None else mask
    r = a * sed / R * m
    q = PNORM
    S = float((r ** q).sum()) / n_solid
    P = S ** (1.0 / q)
    g = P - 1.0
    dP_dr = P ** (1.0 - q) * r ** (q - 1.0) / n_solid
    coef = dP_dr * a / R / h3 * m
    ue = u[stiff.mesh.edof]
    ke_u = stiff.apply(ue, lam_e, mu_e)
    rhs = np.bincount(stiff.mesh.edof.ravel(), weights=(coef[:, None] * ke_u).ravel(),
                      minlength=stiff.mesh.n_dof)
    rhs -= float((dP_dr * r).sum()) / comp * f
    rhs[stiff.base.fixed] = 0.0
    psi = solver.solve_many(k, [rhs])[0]
    pe = psi[stiff.mesh.edof]
    cl, cm = stiff.pair(pe, ue, lam_e, mu_e)
    g_lam = dP_dr * a / R * m * 0.5 * el / h3 - cl
    g_mu = dP_dr * a / R * m * 0.5 * em / h3 - cm
    gx = g_lam * dlam_dx + g_mu * dmu_dx - dP_dr * r / R * Rw
    gw = (xpp * dwp) * (g_lam[None, :] * lam_i[:, None] + g_mu[None, :] * mu_i[:, None])
    gw -= (dP_dr * r / R * xb)[None, :] * resil[:, None]
    gy = np.einsum("in,ijn->jn", gw, dw)
    return g, gx, gy, float(r[xb > 0.5].max()) if np.any(xb > 0.5) else float(r.max())


def _material_normals(domain, w):
    """Per pair (i, j): unit normal per cell of the interface between i and
    j, from the gradient of w_i - w_j on the grid (smoothed one cell)."""
    out = {}
    N = len(w)
    ijk = tuple(domain.mesh.elem_ijk.T)
    for i in range(N):
        for j in range(i + 1, N):
            g = domain.full_grid(w[i] - w[j])
            g = ndimage.gaussian_filter(g, 0.8)
            gr = np.stack([np.gradient(g, axis=a)[ijk] for a in range(3)], axis=1)
            nrm = np.linalg.norm(gr, axis=1)
            gr[nrm < 1e-12] = (0.0, 0.0, 1.0)
            nrm[nrm < 1e-12] = 1.0
            out[(i, j)] = gr / nrm[:, None]
    return out


# ------------------------------------------------------------ delivery

def material_labels(domain, xphys, w, min_cells: int = 6) -> np.ndarray:
    """Per cell the material it delivers (argmax w), -1 where void; material
    specks under `min_cells` take the label of what surrounds them."""
    lab = np.argmax(w, axis=0).astype(np.int64)
    lab[xphys <= 0.5] = -1
    g = domain.full_grid(lab.astype(float), fill=-1).astype(np.int64)
    for k in range(len(w)):
        cc, ncc = ndimage.label(g == k, structure=_CONN26)
        if ncc <= 1:
            continue
        sizes = np.bincount(cc.ravel())
        small = np.isin(cc, np.nonzero(sizes < min_cells)[0]) & (cc > 0)
        if small.any():
            other = np.where((g >= 0) & (g != k) & ~small, g, -1)
            # nearest other-material cell
            dist, idx = ndimage.distance_transform_edt(other < 0, return_indices=True)
            near = other[tuple(idx)]
            g[small & (near >= 0)] = near[small & (near >= 0)]
    return g[tuple(domain.mesh.elem_ijk.T)]


def _region_mesh(domain, labels, k, upsample=3):
    """Closed mesh of the region where material k is the delivered label,
    extended through void (nearest material) and a pad round the grid, so
    an exact boolean with the solid gives material k's body."""
    import trimesh
    from skimage.measure import marching_cubes

    g = domain.full_grid(labels.astype(float), fill=-1)
    have = g >= 0
    _, idx = ndimage.distance_transform_edt(~have, return_indices=True)
    g = g[tuple(idx)]
    ind = (g == k).astype(float)
    pad = 2
    ind = np.pad(ind, pad, mode="edge")
    fine = ndimage.zoom(ind, upsample, order=1)
    fine = ndimage.gaussian_filter(fine, 0.6 * upsample)
    fine = np.pad(fine, 1, constant_values=0.0)
    if fine.max() <= 0.5:
        return None
    step = domain.pitch / upsample
    verts, faces, _, _ = marching_cubes(fine, 0.5, spacing=(step,) * 3)
    # zoom aligns corners: zoomed sample a (fine index a + 1, after the pad
    # of one) sits at coarse index a (s - 1) / (S - 1), s and S the coarse
    # and zoomed sizes, and coarse cell c's centre is at
    # origin + (c - pad + 0.5) * pitch
    scale = ((np.array(ind.shape) - 1) * domain.pitch) / np.maximum(np.array(fine.shape) - 2 - 1, 1)
    verts = (verts / step - 1.0) * scale + domain.origin - pad * domain.pitch + 0.5 * domain.pitch
    m = trimesh.Trimesh(verts, faces[:, ::-1], process=True)
    if m.volume < 0:
        m.invert()
    return m


def _region_mesh_interlocked(domain, labels, k, il, build_axis=2, phase=None, names=None):
    """As _region_mesh, with the interlock's beam lattice (interlock.py) in
    a band round material k's boundary, on a grid fine enough for its
    beams (half a beam wide in plane, half a beam layer along the build
    axis)."""
    import trimesh
    from skimage.measure import marching_cubes

    from .interlock import interlocked_indicator

    g = domain.full_grid(labels.astype(float), fill=-1)
    have = g >= 0
    _, idx = ndimage.distance_transform_edt(~have, return_indices=True)
    g = g[tuple(idx)].astype(np.int64)
    pad = 2
    gp = np.pad(g, pad, mode="edge")
    ind = ndimage.gaussian_filter((gp == k).astype(float), 0.6)
    solid = ndimage.gaussian_filter(np.pad(have, pad).astype(float), 0.6)
    # the material on the other side of k's boundary, for il.pairs
    other = np.where(gp != k, gp, -1)
    _, oidx = ndimage.distance_transform_edt(other < 0, return_indices=True)
    near_other = other[tuple(oidx)]
    spacing = np.full(3, 0.5 * il.beam_width)
    spacing[build_axis] = 0.5 * il.layer_height
    lo = domain.origin - pad * domain.pitch
    hi = lo + np.array(gp.shape) * domain.pitch
    n = np.maximum(np.ceil((hi - lo) / spacing).astype(int), 2)
    axes = [lo[a] + (np.arange(n[a]) + 0.5) * spacing[a] for a in range(3)]
    # coarse index (fractional) of each fine point: coarse cell c's centre is lo + (c + 0.5) pitch
    ci = [(ax - lo[a]) / domain.pitch - 0.5 for a, ax in enumerate(axes)]
    grid = np.meshgrid(*ci, indexing="ij")
    fine = ndimage.map_coordinates(ind, grid, order=1, mode="nearest") > 0.5
    # boundary avoidance (as Cura's): no beams within a beam (and half a
    # cell, the coarse solid's error) of the part's surface, where the band
    # would leave slivers; the bond there stays flat
    fsolid = ndimage.map_coordinates(solid, grid, order=1, mode="constant") > 0.5
    avoid = il.avoid if il.avoid is not None else il.beam_width + 0.5 * domain.pitch
    mask = ndimage.distance_transform_edt(fsolid, sampling=spacing) >= avoid
    del fsolid
    if names is not None and il.pairs:
        ok = np.array([j >= 0 and j != k and il.applies(names[k], names[j]) for j in range(len(names))] + [False])
        nn = ndimage.map_coordinates(near_other.astype(float), grid, order=0, mode="nearest").astype(np.int64)
        mask &= ok[np.where(nn >= 0, nn, len(names))]
    del grid
    origin = np.array([ax[0] for ax in axes])
    if phase is None:
        phase = domain.origin
    fine = interlocked_indicator(fine, spacing, origin, il, build_axis, phase, mask)
    if not fine.any():
        return None
    f = ndimage.gaussian_filter(fine.astype(np.float32), 0.35)
    f = np.pad(f, 1, constant_values=0.0)
    verts, faces, _, _ = marching_cubes(f, 0.5, spacing=tuple(spacing))
    verts = verts - spacing + origin
    m = trimesh.Trimesh(verts, faces[:, ::-1], process=True)
    if m.volume < 0:
        m.invert()
    return m


def split_bodies(domain, labels, solid_mesh, names, interlock=None, build_axis=2) -> list:
    """Cut the delivered solid into one body per material with exact
    booleans: [(material index, trimesh)], bodies sharing their interfaces.
    With an interlock (interlock.Interlock), the bodies interpenetrate in
    its beam lattice where they meet."""
    from .shapes import manifold_from_trimesh, trimesh_from_manifold

    whole = manifold_from_trimesh(solid_mesh)
    present = [k for k in range(len(names)) if np.any(labels == k)]
    if len(present) <= 1:
        return [(present[0] if present else 0, solid_mesh)]
    rest = whole
    out = []
    phase = np.asarray(solid_mesh.bounds[0], float)
    for k in present[:-1]:
        if interlock is not None:
            reg = _region_mesh_interlocked(domain, labels, k, interlock, build_axis, phase, names)
        else:
            reg = _region_mesh(domain, labels, k)
        if reg is None:
            continue
        body = rest ^ manifold_from_trimesh(reg)
        rest = rest - manifold_from_trimesh(reg)
        if body.volume() > 0:
            out.append((k, trimesh_from_manifold(body)))
    if rest.volume() > 0:
        out.append((present[-1], trimesh_from_manifold(rest)))
    return out


# ----------------------------------------------------------- verification

def verify_multimaterial(study, mm: MultiMaterial, bodies, max_cells: int = 90_000, min_pitch=None,
                         contact_band: float | None = None) -> dict:
    """Voxel FEA of the delivered bodies, each cell its own material:
    per-material peak stress against that material's strength (and across
    layers for FDM), every interface face's bond utilization, deflection,
    mass, and what each softness requirement reads."""
    from scipy.spatial import cKDTree

    from .domain import boundary_conditions
    from .verify import voxelize

    t0 = time.time()
    vol = sum(float(m.volume) for _, m in bodies)
    pitch = (vol / max_cells) ** (1.0 / 3.0)
    if min_pitch:
        pitch = max(pitch, float(min_pitch))
    lo = np.min([m.bounds[0] for _, m in bodies], axis=0)
    hi = np.max([m.bounds[1] for _, m in bodies], axis=0)
    label = None
    origin = None
    for k, m in bodies:
        s, origin = voxelize(m, pitch, box=(lo, hi))
        if label is None:
            label = np.full(s.shape, -1, np.int64)
        label[s & (label < 0)] = k
    solid = label >= 0
    lbl, _ = ndimage.label(solid)
    ijk = np.argwhere(solid)
    hm = HexMesh(ijk, pitch, origin)
    tol = 0.5 * (contact_band or 0.0) + 0.5 * pitch
    bc = boundary_conditions(study, hm, check_loads=False, tol_floor=tol)
    sup = np.unique(lbl[tuple(ijk[bc.support_nodes[hm.enode].any(axis=1)].T)])
    keep = np.isin(lbl[tuple(ijk.T)], sup)
    if not keep.all():
        ijk = ijk[keep]
        hm = HexMesh(ijk, pitch, origin)
        bc = boundary_conditions(study, hm, check_loads=False, tol_floor=tol)
    mat = label[tuple(ijk.T)]
    Es = np.array([m.E for m in mm.materials])
    nus = np.array([m.nu for m in mm.materials])
    lam_i, mu_i = lame(Es, nus)
    lam_e, mu_e = lam_i[mat], mu_i[mat]
    stiff = LameStiffness(hm, bc.fixed, ratios=lam_i / (lam_i + mu_i))
    springs = Es.max() * bc.springs if bc.springs is not None else None
    k = stiff.assemble(lam_e, mu_e, springs)
    solver = Solver(hm, rtol=1e-6, maxiter=4000, rebuild_after_iters=10_000)
    us = solver.solve_many(k, list(bc.loads))
    band = max(1.5 * pitch, contact_band or 0.0)
    build_axis = _build_axis(study)
    fdm = study.process is not None and study.process.kind == "fdm"
    names = [c.name for c in study.load_cases]
    N = len(mm.materials)
    pairs = [(i, j) for i in range(N) for j in range(i + 1, N)]
    cards = {p: interface_card(mm, *p, study.process) for p in pairs}
    flat = {p: interface_card(mm, *p, study.process, interlock=False) for p in pairs}
    req = {r.case: r for r in mm.requirements}
    # interlocked bond only where the lattice is: avoid (mm) inside the surface
    il = mm.interlock
    depth = (ndimage.distance_transform_edt(np.pad(label >= 0, 1))[1:-1, 1:-1, 1:-1] - 0.5) * pitch
    depth_c = depth[tuple(ijk.T)]
    il_avoid = (il.avoid if il.avoid is not None else il.beam_width) if il is not None else np.inf

    # interface faces: neighbours along each axis with different materials
    lut = np.full(label.shape, -1, np.int64)
    lut[tuple(ijk.T)] = np.arange(len(ijk))
    faces = []   # (cell a, cell b, axis)
    for ax in range(3):
        nb = ijk.copy()
        nb[:, ax] += 1
        ok = nb[:, ax] < label.shape[ax]
        j = np.full(len(ijk), -1, np.int64)
        j[ok] = lut[tuple(nb[ok].T)]
        hit = (j >= 0)
        hit[hit] &= mat[hit] != mat[j[hit]]
        a = np.nonzero(hit)[0]
        faces.append((a, j[a], ax))
    face_area = sum(len(a) for a, _, _ in faces) * pitch ** 2
    deep_area = sum(int((np.minimum(depth_c[a], depth_c[b]) >= il_avoid).sum()) for a, b, _ in faces) * pitch ** 2
    pair_area = {}
    for a, b, _ in faces:
        for p in pairs:
            sel = ((mat[a] == p[0]) & (mat[b] == p[1])) | ((mat[a] == p[1]) & (mat[b] == p[0]))
            pair_area[p] = pair_area.get(p, 0.0) + float(sel.sum()) * pitch ** 2

    cases = []
    worst = {"material": np.inf, "interface": np.inf}
    centres = hm.elem_centers()
    warnings = []
    narrowed = {}    # what the contact band would have hidden -> the band it was checked at
    # The stress next to a support or a spread load node is a voxel
    # artifact, so cells within `band` of them are left out. A body thinner
    # than the band (a 2 mm pad loaded on its face) would then go unchecked,
    # its bond with it; there the band narrows (to half, then to cells
    # touching no contact node, then none) until something is read.
    tiers = [band, 0.5 * band, 0.0, None]

    def _beyond(d, t):
        return d >= t if t > 0 else d > 0

    def _away(el_d, ok):
        for t in tiers:
            sel = ok if t is None else ok & _beyond(el_d, t)
            if sel.any():
                return sel, t
        return ok, None

    for ci, case in enumerate(study.load_cases):
        f, u = bc.loads[ci], us[ci]
        F = _case_force(f)
        comp = float(f @ u)
        contact = bc.support_nodes | bc.case_load_nodes[ci]
        node_d = cKDTree(hm.node_xyz[contact]).query(hm.node_xyz, k=1)[0] if contact.any() \
            else np.full(len(hm.node_xyz), np.inf)
        el_d = node_d[hm.enode].min(axis=1)
        scale = 1.0
        rec = {"name": case.name, "force": F, "compliance": comp}
        r = req.get(case.name)
        if r is not None and r.kind == "knock":
            c = comp / max(F, 1e-12) ** 2
            peak_f = float(np.sqrt(2.0 * r.energy * 1000.0 / c))
            scale = peak_f / max(F, 1e-12)
            rec["knock"] = {"energy": r.energy, "compliance": c, "peakForce": peak_f, "maxForce": r.max_force,
                            "meets": bool(peak_f <= r.max_force * 1.0001),
                            "stopDistance": float(peak_f * c)}
        s = stiff.stress(u, lam_e, mu_e) * scale
        vm = _von_mises(s)
        disp = np.linalg.norm(u.reshape(-1, 3), axis=1) * scale
        rec["maxDisplacement"] = float(disp.max())
        per = {}
        for kk in range(N):
            if not (mat == kk).any():
                continue
            sel, t = _away(el_d, mat == kk)
            m = mm.materials[kk]
            if t != band:
                narrowed[(case.name, m.name)] = t
            peak = float(vm[sel].max())
            sf = m.strength / peak if peak > 0 else np.inf
            entry = {"peakStress": peak, "safetyFactor": sf,
                     "at": centres[sel][np.argmax(vm[sel])].tolist(),
                     **({"contactBand": t} if t != band else {})}
            if fdm and build_axis is not None and m.strength_z:
                sz = float(np.maximum(s[sel][:, build_axis], 0).max())
                entry["acrossLayers"] = {"stress": sz, "safetyFactor": m.strength_z / sz if sz > 0 else np.inf}
                sf = min(sf, entry["acrossLayers"]["safetyFactor"])
                entry["safetyFactor"] = sf
            per[m.name] = entry
            worst["material"] = min(worst["material"], sf)
        rec["materials"] = per
        # interfaces: traction from the two cells' mean stress, on the axis normal
        ints = {}
        for p in pairs:
            utils, where = [], []
            on_p = [(((mat[a] == p[0]) & (mat[b] == p[1])) | ((mat[a] == p[1]) & (mat[b] == p[0])))
                    for a, b, _ in faces]
            if not any(o.any() for o in on_p):
                continue
            fd = np.concatenate([np.minimum(el_d[a], el_d[b]) for a, b, _ in faces])
            _, t = _away(fd, np.concatenate(on_p))
            for (a, b, ax), op in zip(faces, on_p):
                sel = op if t is None else op & _beyond(np.minimum(el_d[a], el_d[b]), t)
                if not sel.any():
                    continue
                sm = 0.5 * (s[a[sel]] + s[b[sel]])
                nrm = np.zeros((int(sel.sum()), 3))
                nrm[:, ax] = 1.0
                ut_ = _traction_util(sm, nrm, cards[p], build_axis)
                shallow = np.minimum(depth_c[a[sel]], depth_c[b[sel]]) < il_avoid
                if shallow.any() and cards[p] is not flat[p] and "interlock" in cards[p]:
                    ut_[shallow] = _traction_util(sm[shallow], nrm[shallow], flat[p], build_axis)
                utils.append(ut_)
                where.append(0.5 * (centres[a[sel]] + centres[b[sel]]))
            if not utils:
                continue
            ut = np.concatenate(utils)
            wh = np.concatenate(where)
            # The face-by-face peak sits on the voxel staircase's re-entrant
            # corners, a grid artifact the smooth delivered interface does not
            # have; the 99th percentile is what the bond as a whole sees.
            p99 = float(np.percentile(ut, 99.0)) if len(ut) >= 50 else float(ut.max())
            name = f"{mm.materials[p[0]].name}+{mm.materials[p[1]].name}"
            ints[name] = {"faces": int(len(ut)), "peakUtilization": float(ut.max()),
                          "utilization": p99, "safetyFactor": 1.0 / p99 if p99 > 0 else np.inf,
                          "at": wh[np.argmax(ut)].tolist(), **({"contactBand": t} if t != band else {})}
            if t != band:
                narrowed[(case.name, name)] = t
            worst["interface"] = min(worst["interface"], ints[name]["safetyFactor"])
        rec["interfaces"] = ints
        cases.append(rec)
    for r in mm.requirements:
        if r.kind != "indentation":
            continue
        ci, ri = names.index(r.case), names.index(r.relative_to)
        F, Fr = _case_force(bc.loads[ci]), _case_force(bc.loads[ri])
        d = cases[ci]["compliance"] / F - float(bc.loads[ci] @ us[ri]) / Fr
        cases[ci]["indentation"] = {"value": d, "atLeast": r.at_least, "meets": bool(d >= r.at_least * 0.999),
                                    "relativeTo": r.relative_to}

    volumes = {}
    for k, m in bodies:
        volumes[mm.materials[k].name] = volumes.get(mm.materials[k].name, 0.0) + float(m.volume)
    mass = {n_: v * mm.materials[mm.index(n_)].density / 1000.0 for n_, v in volumes.items()}
    rep = {
        "analysisPitch": pitch, "cells": int(len(ijk)), "seconds": time.time() - t0,
        "volume": vol, "mass": float(sum(mass.values())),
        "bodies": [{"material": mm.materials[k].name, "volume": float(m.volume),
                    "mass": float(m.volume) * mm.materials[k].density / 1000.0,
                    "watertight": bool(m.is_watertight)} for k, m in bodies],
        "materialMass": mass,
        "materialFraction": {k_: v / vol for k_, v in volumes.items()} if vol > 0 else {},
        "interfaceArea": {f"{mm.materials[p[0]].name}+{mm.materials[p[1]].name}": a for p, a in pair_area.items()},
        "interfaceCards": {f"{mm.materials[p[0]].name}+{mm.materials[p[1]].name}": cards[p] for p in pairs},
        "cases": cases,
        "safetyFactor": float(worst["material"]),
        "interfaceSafetyFactor": float(worst["interface"]),
        "maxDisplacement": max((c["maxDisplacement"] for c in cases if c["name"] not in req
                                or req[c["name"]].kind == "knock"), default=0.0),
        "requirementsMet": all(c.get("knock", c.get("indentation", {"meets": True}))["meets"] for c in cases),
    }
    rep["voxelInterfaceArea"] = face_area
    for (cname, what), t in narrowed.items():
        warnings.append(f"case {cname!r}: {what} lies within the {band:.2g} mm contact band of the supports or "
                        "loads; checked " + ("on every cell, contact concentrations included" if t is None else
                                             "on the cells touching no support or load node" if t == 0 else
                                             f"at {t:.2g} mm from them") + " instead")
    # a material or bond the delivered bodies have that no case read: the
    # report must not pass on a check it never made
    unchecked = [f"{mm.materials[p[0]].name}+{mm.materials[p[1]].name}" for p in pairs
                 if pair_area.get(p, 0.0) > 0 and not any(
                     f"{mm.materials[p[0]].name}+{mm.materials[p[1]].name}" in c["interfaces"] for c in cases)]
    unchecked += [mm.materials[k_].name for k_ in sorted(set(mat.tolist()))
                  if not any(mm.materials[k_].name in c["materials"] for c in cases)]
    if unchecked:
        rep["unchecked"] = unchecked
        warnings.append(f"not checked: {', '.join(unchecked)} (no cell or interface face was read); "
                        "the design does not pass on it")
    if warnings:
        rep["warnings"] = warnings
    if il is not None:
        rep["interlockedFraction"] = deep_area / face_area if face_area > 0 else 0.0
    return rep


# ------------------------------------------------------------- the study

@dataclass
class MultiMaterialResult:
    study: object
    mm: MultiMaterial
    bodies: list                  # (material index, trimesh)
    report: dict
    outcomes: list                # every fraction tried: report summaries
    optimum: MMOptimum | None = None
    domain: object = None

    def write(self, out_dir, name=None, coordinates: str = "bed") -> dict:
        """One STL per material (study coordinates), a multi-material 3MF and
        report.json. `coordinates`: "bed" (default) places the 3MF for
        printing (build direction to +z, lowest point at z = 0, centred on
        the bed at 128, 128); "study" keeps the study's coordinates, as the
        STLs have them. report["3mf"] says which, with the transform."""
        import json
        from pathlib import Path

        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        name = name or self.study.name
        files = {}
        for k, m in self.bodies:
            p = out / f"{name}-{self.mm.materials[k].name}.stl"
            m.export(p)
            files[self.mm.materials[k].name] = str(p)
        bodies = [{"name": f"{name} {self.mm.materials[k].name}", "mesh": m, "material": self.mm.materials[k],
                   **({"extruder": self.mm.extruders[self.mm.materials[k].name]}
                      if self.mm.materials[k].name in self.mm.extruders else {})}
                  for k, m in self.bodies]
        tmf = None
        try:
            tmf = write_bodies_3mf(out / f"{name}.3mf", bodies, direction=self.study.print_direction,
                                   name=name, process=self.study.process, slicer=self.mm.slicer,
                                   coordinates=coordinates)
            files["3mf"] = tmf["path"]
        except ImportError:
            pass
        if self.optimum is not None:
            # the design fields, so the bodies can be re-split without re-optimizing
            np.savez_compressed(out / "optimum.npz", x=self.optimum.x, y=self.optimum.y,
                                xphys=self.optimum.xphys, w=self.optimum.w)
            files["optimum"] = str(out / "optimum.npz")
        rep = dict(self.report)
        rep["files"] = files
        self.delivered_3mf = tmf
        if tmf is not None:
            rep["3mf"] = tmf
        rep["outcomes"] = self.outcomes
        (out / "report.json").write_text(json.dumps(_jsonable(rep), indent=2))
        files["report"] = str(out / "report.json")
        return files


def write_bodies_3mf(path, bodies, direction=None, name="part", process=None, slicer="all",
                            coordinates: str = "bed") -> dict:
    """threemf.write_multimaterial_3mf, with a choice of where the parts sit:
    "bed" (its default: the print transform, build direction to +z, lowest
    point at z = 0, footprint centred on (128, 128)) or "study" (the build
    item's transform set to identity, so the 3MF's coordinates are the
    study's, as the STLs and verification have them; a slicer drops it to
    the bed itself). Returns {path, coordinates, transform (4x4, study ->
    file coordinates), note}."""
    import re
    import zipfile
    from types import SimpleNamespace

    from . import threemf

    if coordinates not in ("bed", "study"):
        raise ValueError(f"coordinates is 'bed' or 'study', got {coordinates!r}")
    p = threemf.write_multimaterial_3mf(path, bodies, direction=direction, name=name, process=process, slicer=slicer)
    d = direction if direction is not None else (2, 1)
    if coordinates == "bed":
        allv = np.vstack([np.asarray(b["mesh"].vertices, float) for b in bodies])
        tf = threemf.print_transform(SimpleNamespace(vertices=allv), d)
        note = ("placed for printing: build direction to +z, lowest point at z = 0, centred on the bed at "
                "(128, 128); `transform` maps study coordinates to the file's (the STLs keep the study's)")
    else:
        # the writer always applies the print transform on the build item;
        # put identity there instead (the meshes themselves are in study
        # coordinates in both slicer layouts)
        with zipfile.ZipFile(p) as z:
            parts = {i.filename: z.read(i.filename) for i in z.infolist()}
        model = parts["3D/3dmodel.model"].decode()
        ident = "1 0 0 0 1 0 0 0 1 0 0 0"
        model = re.sub(r"(<build\b.*?</build>)",
                       lambda m: re.sub(r'transform="[^"]*"', f'transform="{ident}"', m.group(1)), model, flags=re.S)
        parts["3D/3dmodel.model"] = model.encode()
        with zipfile.ZipFile(p, "w", zipfile.ZIP_DEFLATED) as z:
            for k, v in parts.items():
                z.writestr(k, v)
        tf = np.eye(4)
        note = "in the study's coordinates (identity transform), as the STLs; the slicer places it on the bed"
    return {"path": str(p), "coordinates": coordinates, "transform": tf.tolist(), "note": note}


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.floating, float)):
        v = float(o)
        return None if not np.isfinite(v) else v
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return _jsonable(o.tolist())
    return o


def _margin(study, mm, rep):
    """The design's margin on its goals (>= 1 passes): per case, the
    materials against the goal's safety factor and the bonds against the
    interface's, except a knock with its own safetyFactor, judged at it."""
    g = study.goal
    ms = []
    sfi = mm.interface_safety_factor or g.safety_factor
    own = {r.case: r.safety_factor for r in mm.requirements if r.kind == "knock" and r.safety_factor}
    if rep.get("cases"):
        for c in rep["cases"]:
            sm, si = own.get(c["name"], g.safety_factor), own.get(c["name"], sfi)
            if sm:
                for v in c.get("materials", {}).values():
                    ms.append(v["safetyFactor"] / sm)
            if si:
                for v in c.get("interfaces", {}).values():
                    if v["safetyFactor"] < np.inf:
                        ms.append(v["safetyFactor"] / si)
    else:
        if g.safety_factor:
            ms.append(rep["safetyFactor"] / g.safety_factor)
        if sfi and rep["interfaceSafetyFactor"] < np.inf:
            ms.append(rep["interfaceSafetyFactor"] / sfi)
    if g.max_deflection:
        ms.append(g.max_deflection / max(rep["maxDisplacement"], 1e-12))
    return min(ms) if ms else None


def _attempt_summary(vf, rep) -> dict:
    """One fraction tried, as the progress event and report.outcomes give it."""
    out = {"volumeFraction": vf, "mass": rep["mass"], "safetyFactor": rep["safetyFactor"],
           "interfaceSafetyFactor": rep["interfaceSafetyFactor"], "margin": rep["margin"],
           "passes": rep["passes"], "materialFraction": rep["materialFraction"],
           "requirementsMet": rep.get("requirementsMet")}
    # the worst case per material and per bond, so a failed step says which
    per, bond = {}, {}
    for c in rep.get("cases") or []:
        for k, v in (c.get("materials") or {}).items():
            if k not in per or v["safetyFactor"] < per[k]["safetyFactor"]:
                per[k] = {"safetyFactor": v["safetyFactor"], "case": c["name"]}
        for k, v in (c.get("interfaces") or {}).items():
            if k not in bond or v["safetyFactor"] < bond[k]["safetyFactor"]:
                bond[k] = {"safetyFactor": v["safetyFactor"], "case": c["name"]}
    if per:
        out["materials"] = per
    if bond:
        out["interfaces"] = bond
    for k in ("unchecked", "warnings"):
        if rep.get(k):
            out[k] = rep[k]
    return out


def _write_partial(mm, vf, bodies, summary):
    """With LOADPATH_PARTIAL_DIR set (`loadpath run` sets it to out/partial),
    each fraction's bodies (an STL per material) and summary as the search
    verifies it, so a run killed later keeps what it found (study.py does
    the same for one material)."""
    import json
    import os

    d = os.environ.get("LOADPATH_PARTIAL_DIR")
    if not d:
        return
    try:
        os.makedirs(d, exist_ok=True)
        stem = f"vf{vf:.4f}".replace(".", "p")
        stls = {}
        for k, m in bodies:
            name = mm.materials[k].name
            m.export(os.path.join(d, f"{stem}-{name}.stl"))
            stls[name] = f"{stem}-{name}.stl"
        with open(os.path.join(d, "outcomes.jsonl"), "a") as fh:
            fh.write(json.dumps(_jsonable({"stl": stls, **summary})) + "\n")
    except Exception:  # noqa: BLE001  (a convenience: never fail the run for it)
        pass


def _print_check(study, mm, bodies, rep):
    """printcheck.check_print on the union of the bodies (support, bridges,
    islands, bed contact), as verify.py gives a single-material part, when
    the study has a build direction; None without one."""
    proc = study.process
    build = study.print_direction if study.print_direction is not None else \
        (proc.build if proc is not None and proc.build not in (None, "auto") else None)
    if build is None or not bodies:
        return None
    try:
        from .printcheck import check_print
        from .shapes import manifold_from_trimesh, trimesh_from_manifold

        whole = manifold_from_trimesh(bodies[0][1])
        for _, m in bodies[1:]:
            whole = whole + manifold_from_trimesh(m)
        vols = {}
        for k, m in bodies:
            vols[k] = vols.get(k, 0.0) + float(m.volume)
        main = mm.materials[max(vols, key=vols.get)]
        pc = check_print(trimesh_from_manifold(whole), build, process=proc.kind if proc is not None else "fdm",
                         material=main, bridge_span=getattr(proc, "bridge_span", None))
        return {k: v for k, v in pc.items() if k != "bridgeList"}
    except Exception as exc:  # noqa: BLE001 - the parts and the report stand without it
        rep.setdefault("warnings", []).append(f"print check not run: {type(exc).__name__}: {exc}")
        return None


def run_multimaterial(study, on_progress=None, should_stop=None, max_runs: int = 5) -> MultiMaterialResult:
    """Optimize, mesh, split and verify a multi-material study. With a
    volumeFraction goal, one design; with a safety factor or deflection,
    the lightest of a short search over fractions that meets it (and its
    interfaces' safety factor, and every softness requirement)."""
    from .domain import build_domain
    from .reconstruct import ReconstructOptions, reconstruct
    from .simp import filter_radius

    mm = study.multi_material
    emit = on_progress or (lambda *a, **k: None)
    if study.gravity is not None:
        emit({"stage": "warning", "message": "multiMaterial leaves self-weight out of the optimization"})
    domain = build_domain(study)
    if mm.interlock is not None and mm.interlock.avoid is None:
        mm.interlock.avoid = mm.interlock.beam_width + 0.5 * domain.pitch
    radius = filter_radius(domain, study.length_scale, study.min_hole)
    emit({"stage": "domain", "cells": domain.n_elem, "pitch": domain.pitch, "warnings": domain.warnings})
    tried = {}

    def attempt(vf, start=None):
        def on_iter(rec, _):
            emit({"stage": "optimize", "volumeFraction": vf, **rec})

        res = optimize_multimaterial(study, domain, mm, vf, radius, x0=start[0] if start else None,
                                     y0=start[1] if start else None, on_iter=on_iter, should_stop=should_stop)
        emit({"stage": "mesh", "volumeFraction": vf})
        solid, _ = reconstruct(study, domain, res.xphys, ReconstructOptions())
        labels = material_labels(domain, res.xphys, res.w)
        bodies = split_bodies(domain, labels, solid, mm.names)
        emit({"stage": "verify", "volumeFraction": vf})
        try:
            rep = verify_multimaterial(study, mm, bodies, min_pitch=0.5 * domain.pitch, contact_band=domain.pitch)
        except (RuntimeError, ValueError) as exc:
            # a design the bodies' voxel model cannot hold (a load left on a
            # piece the supports do not reach) fails; the search goes on
            vol = sum(float(m.volume) for _, m in bodies)
            rep = {"mass": sum(float(m.volume) * mm.materials[k_].density / 1000.0 for k_, m in bodies),
                   "volume": vol, "safetyFactor": 0.0, "interfaceSafetyFactor": 0.0, "maxDisplacement": np.inf,
                   "requirementsMet": False, "materialFraction": {}, "cases": [],
                   "bodies": [{"material": mm.materials[k_].name, "volume": float(m.volume)} for k_, m in bodies],
                   "warnings": [f"verification failed: {exc}"]}
        rep["optimization"] = {"iterations": res.iterations, "seconds": res.seconds,
                               "compliance": res.compliance, "interfaceWeight": res.interface_weight,
                               "requirements": res.requirement_values,
                               "fractions": res.history[-1]["fractions"] if res.history else {}}
        rep["volumeFraction"] = vf
        m = _margin(study, mm, rep)
        rep["margin"] = m
        # a bond or material verification could not read passes nothing
        rep["passes"] = bool((m is None or m >= 1.0) and rep["requirementsMet"] and not rep.get("unchecked"))
        tried[vf] = (res, bodies, rep)
        summary = _attempt_summary(vf, rep)
        emit({"stage": "outcome", **summary})
        _write_partial(mm, vf, bodies, summary)
        return rep

    g = study.goal
    # Preserved cells are solid at any fraction, so a fraction at or under
    # their share of the design space cannot be met: the volume constraint
    # stays violated and MMA runs a step that means nothing.
    pres_share = float(np.mean(domain.preserve))
    vf_floor = min(0.9, pres_share + 0.05)
    search_warnings = []

    def _warn(msg):
        search_warnings.append(msg)
        emit({"stage": "warning", "message": msg})

    if g.safety_factor is None and g.max_deflection is None:
        vf = float(g.volume_fraction)
        if vf <= pres_share:
            _warn(f"goal.volumeFraction {vf:.2f} is at or under the preserved share of the design space "
                  f"({pres_share:.2f}): the volume constraint cannot be met; raise it above {pres_share:.2f}")
        attempt(vf)
    else:
        vf = float(g.volume_fraction or 0.3)
        if vf < vf_floor:
            _warn(f"preserved regions are {pres_share:.0%} of the design space; the search starts at a volume "
                  f"fraction of {vf_floor:.2f} instead of {vf:.2f}")
            vf = vf_floor
        lo, hi = None, None    # highest failing, lowest passing
        for _ in range(max_runs):
            above = [v for v in tried if v > vf]
            src = tried[min(above)][0] if above else None
            rep = attempt(vf, (src.x, src.y) if src is not None else None)
            if rep["passes"]:
                hi = vf if hi is None else min(hi, vf)
            else:
                lo = vf if lo is None else max(lo, vf)
            if hi is None:
                if vf >= 0.9:
                    break
                vf = min(0.9, vf * 1.5)
            elif lo is None:
                vf = max(0.03, vf_floor, vf * 0.7)
            else:
                if hi - lo < 0.01:
                    break
                vf = 0.5 * (lo + hi)
            if vf in tried:
                break
    passing = [v for v, t in tried.items() if t[2]["passes"]]
    if passing:
        pick = min(passing, key=lambda v: tried[v][2]["mass"])
    else:
        pick = max(tried, key=lambda v: (tried[v][2]["margin"] or 0.0))
    res, bodies, rep = tried[pick]
    rep = dict(rep)
    if search_warnings:
        rep["warnings"] = search_warnings + list(rep.get("warnings") or [])
    if mm.interlock is not None and len(bodies) > 1:
        # verification judged the smooth interface against the interlocked
        # card; the delivered bodies carry the lattice
        emit({"stage": "interlock", "volumeFraction": pick})
        solid, _ = reconstruct(study, domain, res.xphys, ReconstructOptions())
        labels = material_labels(domain, res.xphys, res.w)
        try:
            locked = split_bodies(domain, labels, solid, mm.names, interlock=mm.interlock,
                                  build_axis=_build_axis(study) or 2)
            rep["interlock"] = {**mm.interlock.to_dict(), "verifiedOn": "smooth interface, interlocked card",
                                "bodies": [{"material": mm.materials[k_].name, "volume": float(m_.volume),
                                            "watertight": bool(m_.is_watertight)} for k_, m_ in locked]}
            bodies = locked
        except Exception as exc:   # noqa: BLE001 - a failed lattice leaves the smooth bodies
            rep.setdefault("warnings", []).append(f"interlock not built ({exc}); bodies have flat interfaces")
    rep["print"] = _print_check(study, mm, bodies, rep)
    rep["multiMaterial"] = {"materials": [m.to_dict() for m in mm.materials],
                            "requirements": [r.to_dict() for r in mm.requirements],
                            "interfaceSafetyFactor": mm.interface_safety_factor,
                            "interlock": mm.interlock.to_dict() if mm.interlock is not None else None}
    summary = [_attempt_summary(v, t[2]) for v, t in sorted(tried.items())]
    return MultiMaterialResult(study, mm, bodies, rep, summary, res, domain)
