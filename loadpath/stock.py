"""
Bought stock in a printed design: tubes, rods and plates the part is built
round.

A printed polymer is a hundredth as stiff as carbon fibre and a twentieth
as strong, so a frame that has to span a distance (a drone arm, a robot
link, a bike's stays) is lighter as bought stock between printed nodes than
printed whole. The stock is not designed: it is a fixed body with its own
material, placed in the study. What is designed is the printed part round
it, and what has to be checked is the member itself and the joint that
holds it.

    "stock": [{
      "id": "arm", "type": "tube", "material": "cfrp-pultruded",
      "from": [0, 0, 0], "to": [180, 0, 0], "od": 10, "wall": 1,
      "joint": {"type": "bond", "adhesive": "epoxy"}
    }]

What a member does to a study (expand_stock):

- a keep-out `<id>`: the member's body plus the joint's gap, so the printed
  part has the bore the member goes in (compensated for the process by
  fits.py: the bore prints at the size the joint needs);
- per joint a preserved sleeve `<id>-sleeve-<k>` round it, as thin as the
  process prints: the surface the joint acts on. Everything else round the
  member is the optimizer's;
- a load or support that names the member acts on the member itself.

What it does to the analysis. The member's cells are in the mesh the
optimizer and the verification solve, with the member's own elasticity: an
orthotropic matrix per cell, turned to the member's axis (a tube's fibres
run along it, its wall is ten times softer across them). The solver's grid
keeps one element matrix for the printed material, so the member enters in
two parts: its own stiffness, share x K_member, a fixed sparse operator
added the way a pin's springs are (`stock_operator`, in
domain.boundary_conditions, so the optimizer and the verification solve
the same assembly), and a factor on each of its cells' printed modulus,
the share of the cell that really is printed (`mesh.stock_scale`, applied
wherever a solver sets element moduli). The operator only ever adds, so
the stiffness stays positive whatever a solver does to the printed
modulus (heat, penalization, a reduced system). Its mass enters
(`mesh.stock_density`), for self-weight and for a free part's inertia.

What is read from the solved assembly (`assess`), per load case:

- the member: the force and moment it carries at every station along it,
  from the free body of its own cells (exact for the discrete model:
  the member's nodal forces beyond a cut sum to the section's resultant),
  then its stresses in closed form (axial, bending, torsion, shear on the
  real section, not on voxels a wall-thickness wide) against the
  laminate's strengths, wall buckling (NASA SP-8007's orthotropic cylinder)
  and Euler buckling of the free length;
- the joint: the force the printed part puts on the member at each node
  of the interface, resolved into shear along the surface and tension or
  pressure across it. A bond reads shear and peel averaged over the
  overlap its lap-shear strength was measured on (ASTM D1002's 12.7 mm);
  a clamp reads the pressure its screws' preload puts on the member
  against the slip the loads ask for (Coulomb); a pin reads bearing, shear-
  out and net section at the bore; every joint reads the tube wall under
  the pressure on it: hoop stress from the uniform part, ring bending from
  the part that ovalizes it.

The interface is solved as stuck (shared nodes). That is a bond; for a
clamp it holds while the slip check does. A member's end face buried in
the print is not joined: a butt on a socket's floor cannot pull, so the
sides carry everything (the member's last cell layer there carries
nothing). Not modelled: slip itself, the
adhesive's own compliance (the shear a real bondline spreads over its
transfer length reads sharper here, which is why it is read over the test
overlap rather than at a node), the member's own thermal expansion (with
a thermal block only the print grows; carbon barely does along its
fibres), and fatigue of the joint.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field

import numpy as np

__all__ = ["LAMINAE", "STOCK", "ADHESIVES", "StockMaterial", "Member", "Joint", "expand_stock", "parse_stock",
           "stock_material", "laminate", "stock_operator", "stock_cells", "assembly_mesh", "assess", "stock_mass",
           "describe_cards"]

# ---------------------------------------------------------------------------
# Cards. Every number with where it is from, as material_library does.
# ---------------------------------------------------------------------------

SOURCES = {
    "toray-2510": ("Toray 2510 prepreg system data sheet, P707AG-15 unidirectional T700G tape and F6273C-07M "
                   "T700S plain weave, oven cure, room temperature dry (AGATE allowables programme)",
                   "https://www.toraycma.com/wp-content/uploads/2510-Prepreg-System.pdf"),
    "tap-pultruded": ("Tap Plastics, Pultruded Carbon Rods & Tubes: minimum properties, standard modulus fibre "
                      "in bisphenol epoxy vinyl ester, 60% fibre volume (tube)",
                      "https://www.tapplastics.com/image/pdf/Pultruded_Carbon_Rods__Tubes.pdf"),
    "3m-dp460": ("3M Scotch-Weld DP460 product data sheet: overlap shear after 7 days at 24 C, substrates solvent "
                 "wiped, abraded and wiped again", "https://conro.com/downloads/Manuals/3M-Scotch-Weld-DP460-TDS.pdf"),
    "mjf-pa12-adhesives": ("Adhesive bonding of Multi Jet Fusion PA12 and glass-bead PA12, Polymers 17(22) 3020 "
                           "(2025): single-lap shear, four industrial adhesives",
                           "https://www.mdpi.com/2073-4360/17/22/3020"),
    "estimate": ("not on a datasheet; the note says why this number", ""),
}

# A lamina: one ply, 1 along the fibres, 2 across them in the ply, 3 through
# it. MPa. Xt/Xc along the fibres, Yt/Yc across, S in-plane shear, ILSS
# interlaminar (short beam) shear.
LAMINAE = {
    "carbon-epoxy-ud": {
        "title": "Unidirectional standard-modulus carbon / epoxy prepreg tape (Toray T700G / 2510)",
        "E1": (125000, "toray-2510", "0 deg tensile modulus"),
        "E2": (8410, "toray-2510", "90 deg tensile modulus"),
        "E3": (8410, "estimate", "through the ply as across it: both are the matrix between fibres"),
        "G12": (4230, "toray-2510", "in-plane shear modulus"),
        "G13": (4230, "estimate", "as G12: shear along the fibres"),
        "G23": (2900, "estimate", "E2 / (2 (1 + nu23)) with nu23 0.45, typical of carbon / epoxy tape"),
        "nu12": (0.31, "toray-2510", "Poisson's ratio"),
        "nu13": (0.31, "estimate", "as nu12"),
        "nu23": (0.45, "estimate", "typical of unidirectional carbon / epoxy"),
        "Xt": (2172, "toray-2510", "0 deg tensile strength"),
        "Xc": (1448, "toray-2510", "0 deg compressive strength"),
        "Yt": (44.3, "toray-2510", "90 deg tensile strength"),
        "Yc": (199, "toray-2510", "90 deg compressive strength"),
        "S": (154, "toray-2510", "in-plane shear strength at ultimate"),
        "ILSS": (86.2, "toray-2510", "short beam shear strength"),
        "density": (1.517, "toray-2510", "laminate density, 54.4% fibre volume"),
    },
    "carbon-epoxy-fabric": {
        "title": "Plain-weave standard-modulus carbon / epoxy prepreg (Toray T700S-12K / 2510)",
        "E1": (55800, "toray-2510", "0 deg (warp) tensile modulus"),
        "E2": (56000, "toray-2510", "90 deg (fill) tensile modulus"),
        "E3": (8410, "estimate", "through the ply: the unidirectional tape's transverse modulus, same resin"),
        "G12": (4210, "toray-2510", "in-plane shear modulus"),
        "G13": (3500, "estimate", "between the tape's G12 and G23: half the fibres run each way"),
        "G23": (3500, "estimate", "as G13"),
        "nu12": (0.04, "toray-2510", "Poisson's ratio"),
        "nu13": (0.3, "estimate", "matrix-governed"),
        "nu23": (0.3, "estimate", "matrix-governed"),
        "Xt": (912, "toray-2510", "0 deg tensile strength"),
        "Xc": (709, "toray-2510", "0 deg compressive strength"),
        "Yt": (772, "toray-2510", "90 deg tensile strength"),
        "Yc": (698, "toray-2510", "90 deg compressive strength"),
        "S": (133, "toray-2510", "in-plane shear strength at ultimate"),
        "ILSS": (60, "toray-2510", "short beam shear strength"),
        "density": (1.502, "toray-2510", "laminate density, 49.6% fibre volume"),
    },
    "carbon-pultruded": {
        "title": "Pultruded standard-modulus carbon / vinyl ester, all fibres along the length (Tap Plastics tube)",
        "E1": (134000, "tap-pultruded", "tensile modulus, tube minimum"),
        "E2": (8410, "estimate", "the prepreg tape's transverse modulus (toray-2510): matrix-governed, similar "
                                 "fibre volume"),
        "E3": (8410, "estimate", "as E2: a pultrusion is the same across the fibres in every direction"),
        "G12": (4230, "estimate", "the prepreg tape's in-plane shear modulus (toray-2510)"),
        "G13": (4230, "estimate", "as G12"),
        "G23": (2900, "estimate", "E2 / (2 (1 + nu23))"),
        "nu12": (0.31, "estimate", "the prepreg tape's (toray-2510)"),
        "nu13": (0.31, "estimate", "as nu12"),
        "nu23": (0.45, "estimate", "typical of unidirectional carbon / epoxy"),
        "Xt": (1650, "tap-pultruded", "tensile strength, tube minimum"),
        "Xc": (1370, "tap-pultruded", "flexural strength, tube minimum: a pultrusion in bending fails on its "
                                      "compression side, and the sheet gives no compressive strength for tubes "
                                      "(rods: 1900 at 67% fibre)"),
        "Yt": (44.3, "estimate", "the prepreg tape's 90 deg tensile strength (toray-2510): nothing but resin "
                                 "holds a pultrusion across its fibres, which is why they split"),
        "Yc": (199, "estimate", "the prepreg tape's 90 deg compressive strength (toray-2510)"),
        "S": (41.3, "tap-pultruded", "ultimate shear strength, tube minimum"),
        "ILSS": (41.3, "tap-pultruded", "ultimate shear strength, tube minimum"),
        "density": (1.5, "tap-pultruded", "density"),
    },
}

# What is sold: a lamina and how its plies are turned. `layup` is (angle from
# the member's axis in degrees, share of the wall).
STOCK = {
    "cfrp-pultruded": {
        "title": "Pultruded carbon tube, rod or strip: every fibre along the length",
        "lamina": "carbon-pultruded", "layup": [(0.0, 1.0)],
        "note": "stiffest and cheapest along its length, splits across it: clamp and bolt it with care"},
    "cfrp-roll-wrapped": {
        "title": "Roll-wrapped carbon tube: unidirectional prepreg, plies along the tube and round it",
        "lamina": "carbon-epoxy-ud", "layup": [(0.0, 0.67), (90.0, 0.33)],
        "note": "two thirds of the plies along the tube and a third round it is an estimate of a common "
                "bending tube; sellers rarely publish the layup, so give `layup` when you know it"},
    "cfrp-fabric": {
        "title": "Carbon plate or tube laid up from 0/90 woven fabric",
        "lamina": "carbon-epoxy-fabric", "layup": [(0.0, 1.0)],
        "note": "the usual carbon sheet; equally stiff along and across, soft in shear"},
    "cfrp-quasi-isotropic": {
        "title": "Carbon plate, quasi-isotropic 0/+45/-45/90 unidirectional plies",
        "lamina": "carbon-epoxy-ud", "layup": [(0.0, 0.25), (45.0, 0.25), (-45.0, 0.25), (90.0, 0.25)],
        "note": "the plate to bolt through: fibres in every direction round a hole"},
}
_STOCK_ALIASES = {"carbon": "cfrp-roll-wrapped", "cfrp": "cfrp-roll-wrapped", "carbon tube": "cfrp-roll-wrapped",
                  "pultruded": "cfrp-pultruded", "pultruded carbon": "cfrp-pultruded",
                  "carbon rod": "cfrp-pultruded", "carbon plate": "cfrp-fabric", "carbon sheet": "cfrp-fabric",
                  "woven carbon": "cfrp-fabric", "roll wrapped": "cfrp-roll-wrapped",
                  "roll-wrapped": "cfrp-roll-wrapped", "quasi": "cfrp-quasi-isotropic"}

# Adhesives. `lapShear` is the single-lap strength (MPa) by what it is
# bonded to; a joint between two materials holds what the weaker side does.
# `overlap` is the test's overlap (mm): the length the strength is an
# average over (ASTM D1002: 12.7 mm).
ADHESIVES = {
    "epoxy": {
        "title": "Two-part structural epoxy (values from 3M Scotch-Weld DP460)",
        "lapShear": {
            "composite": (6.9, "3m-dp460", "FRP, substrate failure: the laminate's surface ply let go first"),
            "aluminium": (19.3, "3m-dp460", "cold rolled steel's value, abraded: aluminium bought as tube is "
                                            "abraded, not etched (etched aluminium 31.0)"),
            "steel": (19.3, "3m-dp460", "cold rolled steel, abraded"),
            "stainless": (27.6, "3m-dp460", "stainless steel, abraded"),
            "abs": (4.0, "3m-dp460", "ABS"),
            "pc": (3.4, "3m-dp460", "polycarbonate"),
            "nylon": (3.8, "mjf-pa12-adhesives", "epoxies on Multi Jet Fusion PA12: 30% under the polyurethane's "
                                                 "5.0"),
            "petg": (3.4, "estimate", "polycarbonate's value (3m-dp460): no sheet lists printed PETG; a "
                                      "copolyester bonds no better"),
            "pla": (4.0, "estimate", "ABS's value (3m-dp460): no sheet lists printed PLA"),
        },
        "gap": (0.15, "estimate", "mm bondline: structural epoxies are strongest at 0.1 to 0.2 mm"),
        "overlap": 12.7,
    },
    "polyurethane": {
        "title": "Two-part structural polyurethane (values from 3M Scotch-Weld DP620NS on printed PA12)",
        "lapShear": {
            "nylon": (5.0, "mjf-pa12-adhesives", "Multi Jet Fusion PA12, 5.0 +- 0.35 MPa: the best of four "
                                                 "adhesives on it"),
            "composite": (5.0, "estimate", "no more than it holds on the nylon it joins the member to"),
        },
        "gap": (0.2, "estimate", "mm bondline"),
        "overlap": 12.7,
    },
}
_ADHESIVE_ALIASES = {"dp460": "epoxy", "structural epoxy": "epoxy", "pu": "polyurethane", "dp620": "polyurethane",
                     "dp620ns": "polyurethane"}
# which lapShear row a printed or stock material reads
_FAMILY = (("pa", "nylon"), ("nylon", "nylon"), ("petg", "petg"), ("pet", "petg"), ("pla", "pla"),
           ("abs", "abs"), ("asa", "abs"), ("pc", "pc"), ("polycarbonate", "pc"), ("cfrp", "composite"),
           ("carbon", "composite"), ("gfrp", "composite"), ("6061", "aluminium"), ("7075", "aluminium"),
           ("alsi", "aluminium"), ("a356", "aluminium"), ("304", "stainless"), ("316", "stainless"),
           ("steel", "steel"), ("4130", "steel"), ("ti", "steel"))

# Static friction of a printed polymer clamp on a member: not on any
# datasheet. 0.2 is the low end of dry polymer on cured epoxy or anodized
# aluminium (0.2 to 0.4); carbon assembly paste roughly doubles it. Give
# joint.friction when you have measured it.
FRICTION = 0.2
# mm kept clear round a member where it is not joined (joint spans given)
CLEARANCE = 2.0


def _val(card, key):
    v = card[key]
    return float(v[0]) if isinstance(v, tuple) else float(v)


@dataclass
class StockMaterial:
    """A member's material in its own axes: 1 along the member, 2 across it
    in the wall (round a tube, across a plate's width), 3 through the wall.
    MPa, g/cm^3. Strengths are the laminate's: Xt/Xc along the member, Yt/Yc
    in direction 2, S in the wall's plane, ILSS between plies, Zc through
    the wall in compression."""
    name: str
    E1: float
    E2: float
    E3: float
    G12: float
    G13: float
    G23: float
    nu12: float
    nu13: float
    nu23: float
    density: float
    Xt: float
    Xc: float
    Yt: float
    Yc: float
    S: float
    ILSS: float
    Zc: float
    isotropic: bool = False
    family: str = "composite"
    note: str = ""
    sources: tuple = ()

    def compliance(self) -> np.ndarray:
        """6x6 compliance in the member's axes, the solver's Voigt order
        (11, 22, 33, 12, 23, 31)."""
        s = np.zeros((6, 6))
        s[0, 0], s[1, 1], s[2, 2] = 1.0 / self.E1, 1.0 / self.E2, 1.0 / self.E3
        s[0, 1] = s[1, 0] = -self.nu12 / self.E1
        s[0, 2] = s[2, 0] = -self.nu13 / self.E1
        s[1, 2] = s[2, 1] = -self.nu23 / self.E2
        s[3, 3], s[4, 4], s[5, 5] = 1.0 / self.G12, 1.0 / self.G23, 1.0 / self.G13
        return s

    def stiffness(self) -> np.ndarray:
        return np.linalg.inv(self.compliance())

    def to_dict(self) -> dict:
        keys = ("E1", "E2", "E3", "G12", "G13", "G23", "nu12", "nu13", "nu23", "density", "Xt", "Xc", "Yt", "Yc",
                "S", "ILSS", "Zc")
        out = {"name": self.name, **{k: round(float(getattr(self, k)), 4) for k in keys}}
        if self.note:
            out["note"] = self.note
        if self.sources:
            out["sources"] = list(self.sources)
        return out


def _q_ply(lam, theta):
    """Plane-stress stiffness (3x3, engineering shear) of a ply turned
    `theta` degrees from the laminate's x axis."""
    e1, e2, g12, nu12 = lam["E1"], lam["E2"], lam["G12"], lam["nu12"]
    nu21 = nu12 * e2 / e1
    d = 1.0 - nu12 * nu21
    q = np.array([[e1 / d, nu12 * e2 / d, 0.0], [nu12 * e2 / d, e2 / d, 0.0], [0.0, 0.0, g12]])
    return _turn(q, theta)


def _strain_to_ply(theta):
    """Laminate strain (ex, ey, gxy) -> the ply's own (e1, e2, g12)."""
    c, s = np.cos(np.radians(theta)), np.sin(np.radians(theta))
    return np.array([[c * c, s * s, c * s], [s * s, c * c, -c * s], [-2 * c * s, 2 * c * s, c * c - s * s]])


def _turn(q, theta):
    t = _strain_to_ply(theta)
    return t.T @ q @ t


def laminate(lamina: dict, layup, name: str = "laminate", note: str = "", sources=()) -> StockMaterial:
    """A laminate's own constants and strengths from its lamina and layup
    [(angle deg, share), ...], by classical lamination theory (a balanced,
    symmetric wall: shares, not a stacking order, so no bending-extension
    coupling).

    Stiffness: the plies' plane-stress stiffnesses averaged by share, then
    inverted (Ex = 1 / a11 and so on). Through the wall the plies are in
    series: E3 the lamina's, the interlaminar shear moduli compliance-
    averaged.

    Strength, per direction of load: the load at which a ply's fibres break
    when the fibres alone can carry that load (the matrix cracks first and
    the fibres take over: netting, the ply stresses found with the matrix
    stiffness discounted), else the load at which the matrix first fails
    (nothing takes over: a unidirectional tube pulled across its fibres, a
    0/90 wall in shear)."""
    lam = {k: (_val(lamina, k) if not isinstance(lamina[k], str) else lamina[k]) for k in lamina if k != "title"}
    plies = [(float(a), float(f)) for a, f in layup]
    tot = sum(f for _, f in plies)
    if tot <= 0 or any(f < 0 for _, f in plies):
        raise ValueError("layup shares must be positive")
    plies = [(a, f / tot) for a, f in plies]
    a_full = sum(f * _q_ply(lam, th) for th, f in plies)
    comp = np.linalg.inv(a_full)
    ex, ey, gxy = 1.0 / comp[0, 0], 1.0 / comp[1, 1], 1.0 / comp[2, 2]
    nuxy = -comp[0, 1] / comp[0, 0]
    # fibres only: E1 along each ply's fibres
    net = {**lam, "E2": 1e-6 * lam["E1"], "G12": 1e-6 * lam["E1"], "nu12": 0.0}
    a_net = sum(f * _q_ply(net, th) for th, f in plies)

    def strength(load):
        load = np.asarray(load, float)
        # can the fibres alone carry it? (the netting stiffness resists this load about as the laminate does)
        e_full = comp @ load
        try:
            e_net = np.linalg.solve(a_net, load)
        except np.linalg.LinAlgError:
            e_net = None
        carried = e_net is not None and float(load @ e_net) < 50.0 * float(load @ e_full)
        best = np.inf
        if carried:
            for th, f in plies:
                e1 = float((_strain_to_ply(th) @ e_net)[0])
                s1 = lam["E1"] * e1
                if abs(s1) > 1e-12:
                    best = min(best, (lam["Xt"] if s1 > 0 else lam["Xc"]) / abs(s1))
            return best
        for th, f in plies:
            ep = _strain_to_ply(th) @ e_full
            q = _q_ply(lam, 0.0)
            s1, s2, t12 = q @ ep
            for v, pos, neg in ((s1, lam["Xt"], lam["Xc"]), (s2, lam["Yt"], lam["Yc"])):
                if abs(v) > 1e-12:
                    best = min(best, (pos if v > 0 else neg) / abs(v))
            if abs(t12) > 1e-12:
                best = min(best, lam["S"] / abs(t12))
        return best

    g13 = 1.0 / sum(f * (np.cos(np.radians(th)) ** 2 / lam["G13"] + np.sin(np.radians(th)) ** 2 / lam["G23"])
                    for th, f in plies)
    g23 = 1.0 / sum(f * (np.sin(np.radians(th)) ** 2 / lam["G13"] + np.cos(np.radians(th)) ** 2 / lam["G23"])
                    for th, f in plies)
    m = StockMaterial(name, ex, ey, lam["E3"], gxy, g13, g23, nuxy, lam["nu13"], lam["nu23"], lam["density"],
                      Xt=strength([1, 0, 0]), Xc=strength([-1, 0, 0]), Yt=strength([0, 1, 0]),
                      Yc=strength([0, -1, 0]), S=strength([0, 0, 1]), ILSS=lam["ILSS"], Zc=lam["Yc"],
                      note=note, sources=tuple(sources))
    # the through-wall Poisson's ratios are estimates: keep the tensor positive definite whatever the layup
    for _ in range(40):
        if np.all(np.linalg.eigvalsh(m.compliance()) > 0):
            break
        m.nu13 *= 0.8
        m.nu23 *= 0.8
    return m


def _family_of(name: str) -> str | None:
    n = name.lower()
    for key, fam in _FAMILY:
        if n.startswith(key) or key in n.split("-"):
            return fam
    return None


def stock_material(spec) -> StockMaterial:
    """A stock card's name ("cfrp-pultruded", ...), a library material's
    (a metal: "6061-t6", "304"), {"base": card, "layup": [[angle, share],
    ...]} for the tube you bought, or a custom object: isotropic {E, nu,
    strength, density} or orthotropic {E1, E2, G12, nu12, Xt, Xc, Yt, Yc, S,
    density, ...}."""
    if isinstance(spec, StockMaterial):
        return spec
    if isinstance(spec, str):
        key = spec.strip().lower()
        key = _STOCK_ALIASES.get(key, key)
        if key in STOCK:
            c = STOCK[key]
            lam = LAMINAE[c["lamina"]]
            srcs = sorted({v[1] for v in lam.values() if isinstance(v, tuple)})
            return laminate(lam, c["layup"], key, c["note"], srcs)
        from .materials import material_from_spec
        try:
            m = material_from_spec(spec)
        except ValueError:
            from .materials import MATERIALS
            raise ValueError(f"unknown stock material {spec!r}; stock cards: {sorted(STOCK)}; or a library "
                             f"material ({', '.join(k for k in sorted(MATERIALS) if _family_of(k) in ('aluminium', 'steel', 'stainless'))}, ...), "
                             "or {E, nu, strength, density}") from None
        return _isotropic(m.name, m.E, m.nu, m.strength, m.density, m.note)
    if isinstance(spec, dict):
        if spec.get("base") is not None and (spec.get("layup") is not None or spec.get("lamina") is not None):
            key = _STOCK_ALIASES.get(str(spec["base"]).strip().lower(), str(spec["base"]).strip().lower())
            if key not in STOCK:
                raise ValueError(f"stock material base {spec['base']!r} is not a stock card ({sorted(STOCK)})")
            c = STOCK[key]
            lam_name = spec.get("lamina", c["lamina"])
            if lam_name not in LAMINAE:
                raise ValueError(f"unknown lamina {lam_name!r}; known: {sorted(LAMINAE)}")
            layup = spec.get("layup", c["layup"])
            try:
                layup = [(float(a), float(f)) for a, f in layup]
            except (TypeError, ValueError):
                raise ValueError("stock material layup is [[angle in degrees from the member's axis, share of "
                                 "the wall], ...]") from None
            return laminate(LAMINAE[lam_name], layup, str(spec.get("name", f"{key}-custom")),
                            "layup as given", sorted({v[1] for v in LAMINAE[lam_name].values()
                                                      if isinstance(v, tuple)}))
        if "E1" in spec:
            try:
                e1, e2 = float(spec["E1"]), float(spec["E2"])
                g12, nu12 = float(spec["G12"]), float(spec.get("nu12", 0.3))
                e3 = float(spec.get("E3", e2))
                m = StockMaterial(
                    str(spec.get("name", "custom")), e1, e2, e3, g12, float(spec.get("G13", g12)),
                    float(spec.get("G23", e3 / 2.9)), nu12, float(spec.get("nu13", nu12)),
                    float(spec.get("nu23", 0.45 if e3 == e2 else 0.3)), float(spec["density"]),
                    Xt=float(spec["Xt"]), Xc=float(spec.get("Xc", spec["Xt"])), Yt=float(spec["Yt"]),
                    Yc=float(spec.get("Yc", spec["Yt"])), S=float(spec["S"]),
                    ILSS=float(spec.get("ILSS", spec["S"])), Zc=float(spec.get("Zc", spec.get("Yc", spec["Yt"]))),
                    family=str(spec.get("family", "composite")))
            except KeyError as exc:
                raise ValueError(f"orthotropic stock material is missing {exc.args[0]!r} (needs E1, E2, G12, "
                                 "Xt, Yt, S, density; MPa and g/cm^3)") from None
            if not np.all(np.linalg.eigvalsh(m.compliance()) > 0):
                raise ValueError("orthotropic stock material is not positive definite: check the Poisson's "
                                 "ratios against the moduli (nu12^2 < E1 / E2)")
            return m
        from .materials import material_from_spec
        m = material_from_spec(spec)
        return _isotropic(m.name, m.E, m.nu, m.strength, m.density, m.note,
                          family=str(spec.get("family") or _family_of(m.name) or "steel"))
    raise ValueError("stock material must be a name or an object")


def _isotropic(name, e, nu, strength, density, note="", family=None) -> StockMaterial:
    g = e / (2.0 * (1.0 + nu))
    # a ductile metal yields in shear at strength / sqrt(3) (von Mises)
    return StockMaterial(name, e, e, e, g, g, g, nu, nu, nu, density, Xt=strength, Xc=strength, Yt=strength,
                         Yc=strength, S=strength / np.sqrt(3.0), ILSS=strength / np.sqrt(3.0), Zc=strength,
                         isotropic=True, family=family or _family_of(name) or "steel", note=note)


def adhesive(spec, printed: str, member: StockMaterial) -> dict:
    """The bond between a printed material and a member: {"name",
    "strength" (MPa, lap shear: the weaker side's), "tension" (MPa across
    the bondline), "gap" (mm), "overlap" (mm the strength is an average
    over), "limitedBy", "source"}. `spec`: an ADHESIVES name, or {"strength",
    "tension"?, "gap"?, "overlap"?, "base"?}."""
    custom = spec if isinstance(spec, dict) else {}
    key = str(custom.get("base", spec if isinstance(spec, str) else "epoxy")).strip().lower()
    key = _ADHESIVE_ALIASES.get(key, key)
    if key not in ADHESIVES:
        raise ValueError(f"unknown adhesive {key!r}; known: {sorted(ADHESIVES)}, or {{\"strength\": MPa}}")
    card = ADHESIVES[key]
    out = {"name": key if not custom else str(custom.get("name", f"{key}-custom")),
           "gap": float(custom.get("gap", _val(card, "gap"))), "overlap": float(custom.get("overlap", card["overlap"]))}
    if custom.get("strength") is not None:
        out.update(strength=float(custom["strength"]), limitedBy="as given", source="given")
    else:
        rows = []
        for who, fam in (("the printed part", _family_of(printed)), ("the member", member.family)):
            row = card["lapShear"].get(fam)
            if row is None:
                known = sorted(card["lapShear"])
                raise ValueError(f"adhesive {key!r} has no lap-shear strength on {who}'s material "
                                 f"({printed if who == 'the printed part' else member.name}; it lists {known}). "
                                 "Give joint.adhesive {\"strength\": MPa} from a lap-shear test of the pair")
            rows.append((row[0], who, row[1], row[2]))
        s, who, src, note = min(rows)
        out.update(strength=float(s), limitedBy=f"{who} ({note})", source=src)
    # across the bondline: adhesives on polymers peel at or under their shear strength, and no sheet above
    # gives a tensile butt strength; the shear strength stands in
    out["tension"] = float(custom.get("tension", out["strength"]))
    if out["strength"] <= 0 or out["gap"] < 0 or out["overlap"] <= 0:
        raise ValueError("adhesive strength and overlap must be > 0, gap >= 0")
    return out


def describe_cards() -> dict:
    """The stock cards as their laminate constants, and the adhesives, for
    `loadpath materials` and the docs."""
    def src(card):
        return {k: {"value": v[0], "source": v[1], "note": v[2]} for k, v in card.items() if isinstance(v, tuple)}

    return {"stock": {k: {"title": c["title"], "lamina": c["lamina"], "layup": [list(p) for p in c["layup"]],
                          "note": c["note"], **stock_material(k).to_dict()} for k, c in STOCK.items()},
            "laminae": {k: {"title": c["title"], **src(c)} for k, c in LAMINAE.items()},
            "adhesives": {k: {"title": c["title"], "overlap": c["overlap"],
                              "gap": {"value": c["gap"][0], "source": c["gap"][1], "note": c["gap"][2]},
                              "lapShear": src(c["lapShear"])} for k, c in ADHESIVES.items()},
            "sources": {k: {"title": t, "url": u} for k, (t, u) in SOURCES.items()}}


# ---------------------------------------------------------------------------
# Members and joints
# ---------------------------------------------------------------------------

JOINT_TYPES = ("bond", "clamp", "pin")


@dataclass
class Joint:
    kind: str                       # bond | clamp | pin
    span: tuple                     # (s0, s1) mm along the member from `from`
    gap: float = 0.0                # mm, radial: bondline or slip clearance
    sleeve: float = 0.0             # mm wall of the preserved sleeve (0: none)
    bond: dict | None = None        # adhesive(): strength, tension, overlap, ...
    friction: float = FRICTION      # clamp
    fasteners: tuple = ()           # clamp: ids of the screws that close it
    clamp_force: float | None = None   # clamp: N of normal force, when no screws are modelled
    pin_d: float | None = None      # pin: diameter, mm
    pin_at: float | None = None     # pin: mm along the member
    pin_axis: tuple | None = None   # pin: direction through the member
    index: int = 0
    spec: dict = field(default_factory=dict)    # as written (the adhesive's and so on), for the study's raw

    def to_spec(self) -> dict:
        out = {k: v for k, v in self.spec.items() if k not in ("span", "sleeve", "gap")}
        out.update(type=self.kind, span=[float(self.span[0]), float(self.span[1])], gap=float(self.gap),
                   sleeve=float(self.sleeve) if self.sleeve > 0 else False)
        return out

    @property
    def length(self):
        return float(self.span[1] - self.span[0])


@dataclass
class Member:
    id: str
    kind: str                       # tube | rod | plate
    material: StockMaterial
    p0: np.ndarray
    p1: np.ndarray
    od: float = 0.0                 # tube, rod
    wall: float = 0.0               # tube (rod: od / 2)
    width: float = 0.0              # plate
    thickness: float = 0.0          # plate
    e2: np.ndarray | None = None    # plate: the width's direction; round: any normal to the axis
    joints: list = field(default_factory=list)
    clearance: float = CLEARANCE
    index: int = 0
    held: bool = False              # a support names it

    @property
    def axis(self):
        d = self.p1 - self.p0
        return d / np.linalg.norm(d)

    @property
    def length(self):
        return float(np.linalg.norm(self.p1 - self.p0))

    @property
    def e3(self):
        return np.cross(self.axis, self.e2)

    @property
    def round(self):
        return self.kind in ("tube", "rod")

    @property
    def ro(self):
        return 0.5 * self.od

    @property
    def ri(self):
        return max(self.ro - self.wall, 0.0) if self.kind == "tube" else 0.0

    @property
    def gap(self):
        return max((j.gap for j in self.joints), default=0.0)

    def local(self, pts):
        """(s along the member, y along e2, z along e3) of world points."""
        d = np.asarray(pts, float) - self.p0
        return d @ self.axis, d @ self.e2, d @ self.e3

    def point(self, s):
        return self.p0 + self.axis * float(s)

    def section(self) -> dict:
        """Area (mm^2), second moments about e2 and e3 (mm^4), polar moment
        (round) or torsion constant, the distances to the extreme fibres."""
        if self.round:
            ro, ri = self.ro, self.ri
            a = np.pi * (ro * ro - ri * ri)
            i = 0.25 * np.pi * (ro ** 4 - ri ** 4)
            return {"A": a, "I2": i, "I3": i, "J": 2.0 * i, "c2": ro, "c3": ro}
        w, t = self.width, self.thickness
        lo, hi = min(w, t), max(w, t)
        # Saint-Venant torsion constant of a solid rectangle
        j = hi * lo ** 3 * (1.0 / 3.0 - 0.21 * lo / hi * (1.0 - lo ** 4 / (12.0 * hi ** 4)))
        return {"A": w * t, "I2": w * t ** 3 / 12.0, "I3": t * w ** 3 / 12.0, "J": j, "c2": 0.5 * w, "c3": 0.5 * t}

    def volume(self) -> float:
        return self.section()["A"] * self.length

    def mass(self) -> float:
        """g."""
        return self.volume() * self.material.density / 1000.0

    def _solid(self, grow: float, s0=None, s1=None, hollow=True):
        from .shapes import Cylinder, OrientedBox
        s0 = 0.0 if s0 is None else float(s0)
        s1 = self.length if s1 is None else float(s1)
        if self.round:
            return Cylinder(self.point(s0), self.point(s1), self.ro + grow, self.ri if hollow else 0.0)
        c = self.point(0.5 * (s0 + s1))
        return OrientedBox(c, [s1 - s0, self.width + 2.0 * grow, self.thickness + 2.0 * grow],
                           [self.axis, self.e2])

    def to_spec(self, item: dict) -> dict:
        """The member as the study's raw keeps it: as written, with every
        joint's span and sizes resolved (a re-parse changes nothing)."""
        out = {k: v for k, v in item.items() if k not in ("joint", "joints")}
        out["joints"] = [j.to_spec() for j in self.joints]
        if self.held:
            out["_held"] = True
        return out

    def body(self):
        """The member as bought (shapes.Shape)."""
        return self._solid(0.0)

    def fe_shape(self):
        """What the analysis takes as the member: its body plus the joint's
        gap (the adhesive, or a clamp's closed clearance, is part of the
        solid)."""
        return self._solid(self.gap)

    def free_spans(self) -> list:
        """The stretches not in a joint, [(s0, s1)]."""
        out, s = [], 0.0
        for j in sorted(self.joints, key=lambda j: j.span[0]):
            if j.span[0] - s > 1e-6:
                out.append((s, j.span[0]))
            s = max(s, j.span[1])
        if self.length - s > 1e-6:
            out.append((s, self.length))
        return out

    def surface_normal(self, y, z):
        """Outward unit normal of the section at local (y, z), as (ny, nz),
        and the distance from the axis for a torque's lever."""
        y, z = np.asarray(y, float), np.asarray(z, float)
        if self.round:
            r = np.maximum(np.hypot(y, z), 1e-12)
            return y / r, z / r, r
        # the nearer pair of faces
        dy, dz = 0.5 * self.width - np.abs(y), 0.5 * self.thickness - np.abs(z)
        on_z = dz <= dy
        ny = np.where(on_z, 0.0, np.sign(y) + (y == 0))
        nz = np.where(on_z, np.sign(z) + (z == 0), 0.0)
        return ny, nz, np.hypot(y, z)

    def bond_area(self, s0, s1) -> float:
        per = 2.0 * np.pi * self.ro if self.round else 2.0 * (self.width + self.thickness)
        return per * (s1 - s0)


def _vec3(v, label):
    try:
        a = np.asarray(v, float).reshape(-1)
    except (TypeError, ValueError):
        a = np.zeros(0)
    if a.shape != (3,) or not np.all(np.isfinite(a)):
        raise ValueError(f"{label} must be [x, y, z] in mm")
    return a


def _pos(v, label, zero=False):
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not np.isfinite(v) or v < 0 or (v == 0 and not zero):
        raise ValueError(f"{label} must be a {'non-negative' if zero else 'positive'} number (mm)")
    return float(v)


def _design_spans(m: Member, design, wall: float, step: float = 0.5) -> list:
    """The stretches of a member that run inside the design space: where
    most of a ring just outside its surface is in a design shape."""
    n = max(int(np.ceil(m.length / step)), 2)
    s = (np.arange(n) + 0.5) * m.length / n
    ang = np.arange(8) * np.pi / 4.0
    if m.round:
        r = m.ro + m.gap + 0.5 * wall
        ring = np.stack([r * np.cos(ang), r * np.sin(ang)], axis=1)
    else:
        hw, ht = 0.5 * m.width + m.gap + 0.5 * wall, 0.5 * m.thickness + m.gap + 0.5 * wall
        ring = np.array([[hw, 0], [hw, ht], [0, ht], [-hw, ht], [-hw, 0], [-hw, -ht], [0, -ht], [hw, -ht]], float)
    pts = (m.p0[None, None] + s[:, None, None] * m.axis[None, None]
           + ring[None, :, 0, None] * m.e2[None, None] + ring[None, :, 1, None] * m.e3[None, None])
    inside = np.zeros(pts.shape[:2], bool)
    for sh in design:
        inside |= sh.contains(pts.reshape(-1, 3)).reshape(inside.shape)
    on = inside.sum(axis=1) >= 5
    spans, start = [], None
    half = 0.5 * m.length / n
    for i, v in enumerate(on):
        if v and start is None:
            start = s[i] - half
        if start is not None and (not v or i == len(on) - 1):
            end = s[i] + half if v else s[i] - half
            if end - start >= 2.0:
                # (rounded for the study's raw, and never past the member's own ends)
                a_, b_ = max(round(start, 3), 0.0), min(round(end, 3), m.length)
                if m.length - b_ < 2e-3:
                    b_ = m.length                  # to the member's end, not a micron short of it
                spans.append((float(a_), float(b_)))
            start = None
    return spans


def parse_stock(items, printed: str = "pla", design=None, member_size: float = 1.4, fastener_ids=None) -> list:
    """`stock` as Members. `design`: the design space's shapes (for the
    joints' default spans); `member_size`: the thinnest member the study
    prints (the sleeves' wall); `fastener_ids`: the study's fasteners, to
    check a clamp's against (None: not checked)."""
    if items is None:
        return []
    if not isinstance(items, list):
        raise ValueError("stock is a list of bought members (tubes, rods, plates)")
    out, seen = [], set()
    for i, it in enumerate(items):
        label = f"stock[{i}]"
        if not isinstance(it, dict):
            raise ValueError(f"{label} must be an object")
        sid = it.get("id")
        if not isinstance(sid, str) or not sid:
            raise ValueError(f"{label}.id is required (loads, supports and the report name the member by it)")
        if sid in seen:
            raise ValueError(f"{label}.id {sid!r} is used twice")
        seen.add(sid)
        kind = it.get("type", "tube")
        if kind not in ("tube", "rod", "plate"):
            raise ValueError(f"{label}.type must be 'tube', 'rod' or 'plate'")
        if "from" not in it or "to" not in it:
            raise ValueError(f"{label} needs from and to: the two ends of its axis, [x, y, z] mm")
        p0, p1 = _vec3(it["from"], f"{label}.from"), _vec3(it["to"], f"{label}.to")
        if np.linalg.norm(p1 - p0) < 1e-6:
            raise ValueError(f"{label}.from and .to must differ")
        try:
            mat = stock_material(it.get("material", "cfrp-roll-wrapped" if kind == "tube" else
                                        "cfrp-pultruded" if kind == "rod" else "cfrp-fabric"))
        except ValueError as exc:
            raise ValueError(f"{label}.material: {exc}") from None
        a = (p1 - p0) / np.linalg.norm(p1 - p0)
        m = Member(sid, kind, mat, p0, p1, index=i)
        if kind in ("tube", "rod"):
            m.od = _pos(it.get("od"), f"{label}.od (outer diameter)")
            if kind == "tube":
                if it.get("wall") is not None:
                    m.wall = _pos(it["wall"], f"{label}.wall")
                elif it.get("innerDiameter") is not None:
                    m.wall = 0.5 * (m.od - _pos(it["innerDiameter"], f"{label}.innerDiameter"))
                else:
                    raise ValueError(f"{label}: a tube needs wall (mm) or innerDiameter")
                if not 0 < m.wall < 0.5 * m.od:
                    raise ValueError(f"{label}.wall must be under half the outer diameter (a rod is type 'rod')")
            else:
                m.wall = 0.5 * m.od
            ref = np.array([1.0, 0, 0]) if abs(a[0]) < 0.9 else np.array([0, 1.0, 0])
            e2 = np.cross(a, ref)
            m.e2 = e2 / np.linalg.norm(e2)
        else:
            m.width = _pos(it.get("width"), f"{label}.width")
            m.thickness = _pos(it.get("thickness"), f"{label}.thickness")
            if it.get("widthAxis") is None:
                raise ValueError(f"{label}.widthAxis is required for a plate: the direction its width runs, "
                                 "[x, y, z] (its thickness is normal to that and to its length)")
            w = _vec3(it["widthAxis"], f"{label}.widthAxis")
            w = w - (w @ a) * a
            if np.linalg.norm(w) < 1e-6:
                raise ValueError(f"{label}.widthAxis must not run along the member")
            m.e2 = w / np.linalg.norm(w)
        m.clearance = _pos(it.get("clearance", CLEARANCE), f"{label}.clearance")
        m.held = bool(it.get("_held"))
        joints_in = it.get("joints")
        if joints_in is None:
            joints_in = [it.get("joint") or {"type": "bond"}]
            explicit = False
        else:
            explicit = True
            if not isinstance(joints_in, list) or not joints_in:
                raise ValueError(f"{label}.joints is a non-empty list (or give one `joint`)")
        specs = []
        for k, js in enumerate(joints_in):
            jl = f"{label}.joint" + (f"s[{k}]" if explicit else "")
            if isinstance(js, str):
                js = {"type": js}
            if not isinstance(js, dict):
                raise ValueError(f"{jl} must be an object")
            jk = js.get("type", "bond")
            if jk not in JOINT_TYPES:
                raise ValueError(f"{jl}.type must be one of {list(JOINT_TYPES)}")
            j = Joint(jk, (0.0, 0.0), spec=dict(js))
            if jk == "bond":
                try:
                    j.bond = adhesive(js.get("adhesive", "epoxy"), printed, mat)
                except ValueError as exc:
                    raise ValueError(f"{jl}.adhesive: {exc}") from None
                j.gap = j.bond["gap"]
            if "gap" in js:
                j.gap = _pos(js["gap"], f"{jl}.gap", zero=True)
                if j.bond is not None:
                    j.bond["gap"] = j.gap
            if jk == "clamp":
                j.friction = float(js.get("friction", FRICTION))
                if not 0 < j.friction <= 1.5:
                    raise ValueError(f"{jl}.friction is the static friction coefficient of the printed clamp on "
                                     "the member, in (0, 1.5]")
                fs = js.get("fasteners") or []
                if isinstance(fs, str):
                    fs = [fs]
                bad = [f for f in fs if f not in fastener_ids] if fastener_ids is not None else []
                if bad:
                    raise ValueError(f"{jl}.fasteners {bad} are not ids in the study's fasteners")
                j.fasteners = tuple(fs)
                if js.get("clampForce") is not None:
                    j.clamp_force = _pos(js["clampForce"], f"{jl}.clampForce (N)")
                if not j.fasteners and j.clamp_force is None:
                    raise ValueError(f"{jl}: a clamp needs what closes it: fasteners (ids of the study's "
                                     "fasteners across its split) or clampForce (N of normal force on the "
                                     "member)")
            if jk == "pin":
                j.pin_d = _pos(js.get("diameter"), f"{jl}.diameter (the pin's or bolt's)")
                if js.get("at") is None:
                    raise ValueError(f"{jl}.at is required: mm along the member from `from` to the pin")
                j.pin_at = _pos(js["at"], f"{jl}.at", zero=True)
                ax = _vec3(js.get("direction", m.e3 if kind == "plate" else m.e2), f"{jl}.direction")
                ax = ax - (ax @ a) * a
                if np.linalg.norm(ax) < 1e-6:
                    raise ValueError(f"{jl}.direction must cross the member, not run along it")
                j.pin_axis = tuple(ax / np.linalg.norm(ax))
                if kind == "tube" and j.pin_d >= 0.6 * m.od:
                    raise ValueError(f"{jl}.diameter {j.pin_d:g} mm is over 0.6 of the tube's diameter: nothing "
                                     "is left beside the hole")
            sl = js.get("sleeve")
            if sl is False or sl == 0:
                j.sleeve = 0.0
            elif sl is None or sl is True:
                j.sleeve = float(member_size)
            else:
                j.sleeve = _pos(sl, f"{jl}.sleeve (the sleeve's wall, mm; false for none)")
            span = js.get("span")
            if span is not None:
                try:
                    s0, s1 = float(span[0]), float(span[1])
                except (TypeError, ValueError, IndexError):
                    raise ValueError(f"{jl}.span is [from, to] in mm along the member from its `from` end") from None
                if not (0 <= s0 < s1 <= m.length + 1e-3):
                    raise ValueError(f"{jl}.span {list(span)} must lie on the member (0 to {m.length:g} mm)")
                j.span = (s0, min(s1, m.length))
            elif explicit and len(joints_in) > 1:
                raise ValueError(f"{jl}.span is required when a member has several joints")
            specs.append(j)
        m.joints = specs
        if len(specs) == 1 and specs[0].span == (0.0, 0.0):
            # where it runs in the design space; one joint object stands for each stretch
            j0 = specs[0]
            spans = _design_spans(m, design, j0.sleeve or member_size) if design else [(0.0, m.length)]
            if not spans:
                raise ValueError(f"{label} never runs through the design space, so the printed part cannot hold "
                                 "it: move it, or extend the design space round where it is joined")
            if j0.kind == "pin":
                spans = [sp for sp in spans if sp[0] - 1e-6 <= j0.pin_at <= sp[1] + 1e-6] or spans[:1]
            m.joints = [copy.copy(j0) for _ in spans]
            for k, (j, sp) in enumerate(zip(m.joints, spans)):
                j.span = sp
                if j.bond is not None:
                    j.bond = dict(j.bond)
        m.joints.sort(key=lambda j: j.span[0])
        for k, j in enumerate(m.joints):
            j.index = k
            if k and j.span[0] < m.joints[k - 1].span[1] - 1e-6:
                raise ValueError(f"{label}: joints {k - 1} and {k} overlap along the member")
            if j.kind == "pin" and not (j.span[0] <= j.pin_at <= j.span[1]):
                raise ValueError(f"{label}: the pin at {j.pin_at:g} mm is outside its joint's span {list(j.span)}")
        out.append(m)
    return out


def expand_stock(d: dict, base_dir=None) -> dict:
    """The study with each `stock` member turned into its regions: the
    keep-out `<id>` (body plus gap), a sleeve per joint, a clear zone round
    the stretches that are not joined, and loads and supports that name the
    member moved onto the member itself. The members are recorded in
    d["_stock"] (spec keeps them on the Study). Returns d itself when there
    is no stock."""
    if not d.get("stock") or d.get("_stock") is not None:
        return d
    from .shapes import shape_from_spec

    d = copy.deepcopy(d)
    ds = d.get("designSpace")
    ds = [ds] if isinstance(ds, dict) else (ds or [])
    try:
        design = [shape_from_spec(s, base_dir) for s in ds]
    except (ValueError, KeyError, TypeError):
        design = []            # study_from_dict reports the design space's own errors
    mat_in = d.get("material", "pla")
    printed = mat_in[0] if isinstance(mat_in, list) else mat_in
    printed = printed if isinstance(printed, str) else str((printed or {}).get("base") or (printed or {}).get("name") or "pla")
    size = d.get("minMember")
    if not isinstance(size, (int, float)) or isinstance(size, bool) or size <= 0:
        proc = d.get("process")
        kind = proc if isinstance(proc, str) else (proc or {}).get("type") if isinstance(proc, dict) else None
        from .spec import DEFAULT_MIN_MEMBER
        size = DEFAULT_MIN_MEMBER
        if kind:
            try:
                from .spec import PROCESSES
                size = max(size, 2.0 * float(PROCESSES[str(kind).lower()]["minFeature"]))
            except (KeyError, ImportError, TypeError, ValueError):
                pass
    fids = [f.get("id") for f in d.get("fasteners") or [] if isinstance(f, dict)]
    fids += [str(r.get("id"))[:-5] for r in d.get("keepOut") or []
             if isinstance(r, dict) and str(r.get("id", "")).endswith("-bore")]
    members = parse_stock(d["stock"], printed, design, float(size), fids)
    preserve = list(d.get("preserve") or [])
    keep_out = list(d.get("keepOut") or [])
    taken = {r.get("id") for r in preserve + keep_out if isinstance(r, dict)}
    for m in members:
        if m.id in taken:
            raise ValueError(f"stock id {m.id!r} is already a preserve or keepOut id")
        ko = {"id": m.id, "shape": m._solid(m.gap, hollow=False).to_spec()}
        if m.round:
            # the bore the member goes in, held at member + gap as built (fits.py compensates for the process);
            # a clamp's is a slip fit on the member
            if all(j.kind == "clamp" for j in m.joints):
                ko["shape"] = m._solid(0.0, hollow=False).to_spec()
                ko.update(fit="slip", mate=m.od)
            else:
                ko["fit"] = "nominal"
        keep_out.append(ko)
        for j in m.joints:
            if j.sleeve > 0:
                if m.round:
                    r_in = float(ko["shape"]["r"])
                    sh = {"type": "cylinder", "p0": np.round(m.point(j.span[0]), 5).tolist(),
                          "p1": np.round(m.point(j.span[1]), 5).tolist(), "r": r_in + j.sleeve, "rInner": r_in}
                else:
                    sh = {"type": "subtract", "shapes": [m._solid(m.gap + j.sleeve, *j.span).to_spec(),
                                                         m._solid(m.gap, -1.0, m.length + 1.0).to_spec()]}
                preserve.append({"id": f"{m.id}-sleeve-{j.index}", "shape": sh})
            if j.kind == "pin":
                half = (m.ro if m.round else 0.5 * max(m.width, m.thickness)) + m.gap + j.sleeve + 1.0
                c = m.point(j.pin_at)
                ax = np.asarray(j.pin_axis)
                keep_out.append({"id": f"{m.id}-pin-{j.index}",
                                 "shape": {"type": "cylinder", "p0": np.round(c - ax * half, 5).tolist(),
                                           "p1": np.round(c + ax * half, 5).tolist(), "r": 0.5 * j.pin_d},
                                 "fit": "slip"})
        if any(j.span != (0.0, 0.0) for j in m.joints):
            for k, (s0, s1) in enumerate(m.free_spans()):
                keep_out.append({"id": f"{m.id}-clear-{k}", "bore": False,
                                 "shape": m._solid(m.clearance, s0, s1, hollow=False).to_spec()})
    d["preserve"], d["keepOut"] = preserve, keep_out
    # a load or support on a member acts on the member: its stretches outside the joints, or all of it
    by_id = {m.id: m for m in members}

    def onto(item, hold=False):
        if isinstance(item, dict) and item.get("region") in by_id and item.get("shape") is None:
            m = by_id[item["region"]]
            m.held = m.held or hold
            free = m.free_spans() or [(0.0, m.length)]
            shapes = [m._solid(0.0, s0, s1).to_spec() for s0, s1 in free]
            item = dict(item)
            item.pop("region")
            item["shape"] = shapes[0] if len(shapes) == 1 else {"type": "union", "shapes": shapes}
        return item

    if d.get("supports"):
        d["supports"] = [onto(s, True) for s in d["supports"]]
    for key in ("loadCases",):
        for c in d.get(key) or []:
            if isinstance(c, dict) and c.get("loads"):
                c["loads"] = [onto(ld) for ld in c["loads"]]
    if d.get("loads"):
        d["loads"] = [onto(ld) for ld in d["loads"]]
    # kept in raw with every span resolved: a study re-parsed from its raw (orientation, explore) reads
    # the same members and adds no regions twice
    d["_stock"] = [m.to_spec(it) for m, it in zip(members, d["stock"])]
    d["_stockPrinted"] = printed
    d["_stockMember"] = float(size)
    return d


# ---------------------------------------------------------------------------
# The member in the solver
# ---------------------------------------------------------------------------

_SAMPLES = 4          # per axis: 64 points a cell for its share of a member


def _shares(shape, centers, pitch, floor=True):
    """The share of each cell inside `shape`, from _SAMPLES^3 points.
    `floor`: a cell the shape only touches counts as one sample's worth."""
    n = _SAMPLES
    off = (np.arange(n) + 0.5) / n - 0.5
    out = np.zeros(len(centers))
    lo, hi = shape.bounds()
    near = np.all((centers > lo - pitch) & (centers < hi + pitch), axis=1)
    if not near.any():
        return out
    c = centers[near]
    hit = np.zeros(len(c))
    for dx in off:
        for dy in off:
            for dz in off:
                hit += shape.contains(c + np.array([dx, dy, dz]) * pitch)
    share = hit / n ** 3
    if not floor:
        out[near] = share
        return out
    # the 8 points domain._fraction classes cells by: a cell one of them puts in the shape is in it here too
    # (it is no design cell, so left out it would be a hole in the interface)
    from .domain import _SUB
    touch = np.zeros(len(c), bool)
    for o in _SUB:
        touch |= shape.contains(c + o * pitch)
    out[near] = np.where(touch, np.maximum(share, 1.0 / n ** 3), share)
    return out


def _rotations(m: Member, centers):
    """(bin index per cell, rotation matrices (bins, 3, 3)): columns are the
    member's axes 1, 2, 3 in world coordinates at each bin. A plate has one;
    a round member's turn with the angle round its axis, binned at 3
    degrees."""
    a = m.axis
    if not m.round:
        return np.zeros(len(centers), np.int64), np.stack([a, m.e2, m.e3], axis=1)[None]
    _, y, z = m.local(centers)
    th = np.arctan2(z, y)
    nb = 120
    b = np.floor((th + np.pi) / (2.0 * np.pi) * nb).astype(np.int64) % nb
    tb = (np.arange(nb) + 0.5) / nb * 2.0 * np.pi - np.pi
    er = np.cos(tb)[:, None] * m.e2[None] + np.sin(tb)[:, None] * m.e3[None]
    eh = np.cross(a[None], er)
    rot = np.stack([np.broadcast_to(a, er.shape), eh, er], axis=2)
    return b, rot


def _rotate_many(c, rot):
    """fem._rotate_c for a stack of rotations (n, 3, 3) -> (n, 6, 6)."""
    from .fem import _VOIGT, _to_tensor

    t = np.einsum("npi,nqj,nrk,nsl,ijkl->npqrs", rot, rot, rot, rot, _to_tensor(c), optimize=True)
    out = np.empty((len(rot), 6, 6))
    for i, (p, q) in enumerate(_VOIGT):
        for j, (r, s) in enumerate(_VOIGT):
            out[:, i, j] = t[:, p, q, r, s]
    return out


class StockCells:
    """The members on one mesh: which cells, how much of each, and each
    member's own stiffness as a sparse matrix (N/mm) for reading what it
    carries. Built once per mesh (`stock_cells`)."""

    def __init__(self, study, mesh):
        from .fem import hex8_ke, hex8_ke_batch

        self.mesh = mesh
        self.members = list(study.stock)
        h = mesh.pitch
        centers = mesh.origin + (mesh.elem_ijk + 0.5) * h
        n = mesh.n_elem
        self.share = np.zeros(n)                 # the cell's share that is member (all members)
        self.owner = np.full(n, -1, np.int64)
        self.relief = np.zeros(n, bool)
        self.per = []
        for mi, m in enumerate(self.members):
            # the cells: every one the member or its joint's gap reaches (what the domain pins); the share:
            # the member's own body in each (the gap is adhesive or air, not member)
            cells = np.flatnonzero((_shares(m.fe_shape(), centers, h) >= 1.0 / _SAMPLES ** 3) & (self.owner < 0))
            f = np.zeros(n)
            f[cells] = np.minimum(_shares(m.body(), centers[cells], h, floor=False), 1.0 - self.share[cells])
            self.owner[cells] = mi
            self.share[cells] += f[cells]
            # An end face buried in the print is not a joint: the adhesive is on the member's sides, a
            # butt on a socket's floor cannot pull, and stuck nodes there would let a stiff tube lever
            # on the floor and take the load off the bond that has to hold it. The member's last layer
            # of cells at such an end carries nothing (its mass stays).
            sa, ya, za = m.local(centers)
            inside = (np.hypot(ya, za) < m.ro) if m.round else \
                ((np.abs(ya) < 0.5 * m.width) & (np.abs(za) < 0.5 * m.thickness))
            other = np.ones(n, bool)
            other[cells] = False
            live = np.ones(len(cells), bool)
            ends = []
            for end, lo, hi, layer in ((0, -1.5 * h, 0.0, sa[cells] < h),
                                       (1, m.length, m.length + 1.5 * h, sa[cells] > m.length - h)):
                if (other & inside & (sa > lo) & (sa < hi)).any():
                    live &= ~layer
                    ends.append(end)
            b, rot = _rotations(m, centers[cells])
            ke = hex8_ke_batch(_rotate_many(m.material.stiffness(), rot))[0] * h
            kes = ke[b] * (f[cells] * live)[:, None, None]
            self.relief[cells[~live]] = True
            import scipy.sparse as sp
            ed = mesh.edof[cells]
            k = sp.csr_matrix((kes.reshape(-1), (np.repeat(ed, 24, axis=1).reshape(-1),
                                                 np.tile(ed, (1, 24)).reshape(-1))),
                              shape=(mesh.n_dof, mesh.n_dof))
            nodes = np.unique(mesh.enode[cells])
            s, y, z = m.local(mesh.node_xyz[nodes])
            self.per.append({"cells": cells, "share": f[cells], "k": k, "nodes": nodes, "s": s, "y": y, "z": z,
                             "buriedEnds": ends})
        self.cells = np.flatnonzero(self.owner >= 0)
        # what the grid already puts in these cells at the printed modulus: all of the cell in the optimizer
        # (whole cells), the cell's solid share on a cut-cell verification grid; `stock_print` is the share of
        # the cell that really is printed (a sleeve's part of a cell the member's wall crosses)
        fill = getattr(mesh, "stock_fill", None)
        fill = np.ones(n) if fill is None else np.asarray(fill, float)
        printed = getattr(mesh, "stock_print", None)
        printed = np.clip(fill - self.share, 0.0, 1.0) if printed is None else \
            np.minimum(np.asarray(printed, float), np.clip(1.0 - self.share, 0.0, 1.0))
        self.printed = printed
        self.fill = fill
        # the share of the printed element matrix to take off (all but a millionth-of-a-member's worth: a
        # cell that is only gap keeps its nodes in the system)
        self.sub = np.clip(fill - printed, 0.0, 1.0)
        self.sub[self.cells] = np.minimum(self.sub[self.cells], np.maximum(fill[self.cells] - 1e-4, 0.0))
        nu = getattr(mesh, "stock_elastic", None)
        self.nu = study.elastic() if nu is None else nu
        self.e_print = float(study.material.E)
        self.ke_print = hex8_ke(self.nu) * h * self.e_print
        rho_p = float(study.material.density)
        dens = np.zeros(n)
        for mi, m in enumerate(self.members):
            p = self.per[mi]
            dens[p["cells"]] += p["share"] * m.material.density / rho_p
        dens[self.cells] -= self.sub[self.cells]
        # The share of each cell's own (printed) stiffness that stays: every solver multiplies its
        # element moduli by this (mesh.stock_scale), so the member's operator only ever adds. Taking
        # the printed share off as a fixed operator went negative wherever a solver had lowered the
        # cell's modulus first (the thermal block derates every cell by its temperature: a drone
        # frame's first solve broke down on both solvers).
        self.scale = np.ones(n)
        self.scale[self.cells] = 1.0 - self.sub[self.cells] / np.maximum(fill[self.cells], 1e-12)
        # relative to the printed material's density, on top of the cell's own (domain.node_mass, gravity_force)
        self.density = dens

    def operator(self, fixed=None):
        """Sum over member cells of share x K_member, N/mm (held dofs' rows
        and columns cleared). The printed material's share of those cells is
        the grid's own, scaled by `scale`."""
        import scipy.sparse as sp

        mesh = self.mesh
        k = None
        for p in self.per:
            k = p["k"] if k is None else k + p["k"]
        c = self.cells
        if k is None or not len(c):
            return None
        k = k.tocsr()
        if fixed is not None and np.any(fixed):
            free = sp.diags((~np.asarray(fixed, bool)).astype(float))
            k = (free @ k @ free).tocsr()
        k.eliminate_zeros()
        return k


def stock_cells(study, mesh) -> StockCells | None:
    """The study's members on `mesh` (cached on the mesh), or None."""
    if not getattr(study, "stock", None):
        return None
    sc = getattr(mesh, "_stock_cells", None)
    if sc is None or sc.key != id(study.stock):
        sc = StockCells(study, mesh)
        sc.key = id(study.stock)
        mesh._stock_cells = sc
        mesh.stock_density = sc.density
        mesh.stock_scale = sc.scale
    return sc


def stock_mass(study) -> float:
    """The members' mass, g (each at its own density)."""
    return float(sum(m.mass() for m in getattr(study, "stock", None) or []))


def stock_operator(study, mesh, fixed=None):
    """The members' stiffness on a mesh as the operator the solvers add to
    theirs, at unit printed modulus (domain.BoundaryConditions.springs'
    unit), or None when the mesh holds none of them."""
    sc = stock_cells(study, mesh)
    if sc is None:
        return None
    k = sc.operator(fixed)
    return None if k is None else (k / sc.e_print).tocsr()


def assembly_mesh(study, mesh):
    """The printed part and the members as one solid (trimesh): what the
    verification's grid is cut from, so the member's cells are there to
    carry its stiffness."""
    import manifold3d as m3d

    from .shapes import manifold_from_trimesh, trimesh_from_manifold
    from .sharp import exact_manifold

    parts = [manifold_from_trimesh(mesh)] + [exact_manifold(m.fe_shape()) for m in study.stock]
    return trimesh_from_manifold(m3d.Manifold.batch_boolean(parts, m3d.OpType.Add))


# ---------------------------------------------------------------------------
# Reading the solved assembly
# ---------------------------------------------------------------------------

def _wall_buckling(m: Member) -> float | None:
    """Axial stress (MPa) at which a tube's wall buckles: the classical
    orthotropic cylinder (NASA SP-8007, eq. for a moderately long
    orthotropic cylinder under axial compression, minimized over the wave
    numbers) times SP-8007's knock-down for imperfections,
    1 - 0.901 (1 - exp(-sqrt(r / t) / 16)). None for a solid section."""
    if m.kind != "tube":
        return None
    mat = m.material
    t, r = m.wall, m.ro - 0.5 * m.wall
    nu12 = mat.nu12
    nu21 = nu12 * mat.E2 / mat.E1
    d = 1.0 - nu12 * nu21
    ex, ey, exy, gxy = mat.E1 * t / d, mat.E2 * t / d, nu12 * mat.E2 * t / d, mat.G12 * t
    dx, dy = mat.E1 * t ** 3 / (12 * d), mat.E2 * t ** 3 / (12 * d)
    dxy = 2.0 * (nu12 * mat.E2 * t ** 3 / (12 * d) + 2.0 * mat.G12 * t ** 3 / 12.0)
    length = max(m.length, 4.0 * r)
    best = np.inf
    for mm in range(1, 400):
        al = mm * np.pi / length
        if al * np.sqrt(r * t) > 12.0:
            break
        for n in range(0, 40):
            be = n / r
            a11 = ex * al ** 2 + gxy * be ** 2
            a22 = ey * be ** 2 + gxy * al ** 2
            a12 = (exy + gxy) * al * be
            a13 = exy / r * al
            a23 = ey / r * be
            a33 = dx * al ** 4 + dxy * al ** 2 * be ** 2 + dy * be ** 4 + ey / r ** 2
            det3 = (a11 * (a22 * a33 - a23 * a23) - a12 * (a12 * a33 - a23 * a13) + a13 * (a12 * a23 - a22 * a13))
            det2 = a11 * a22 - a12 * a12
            if det2 > 0:
                best = min(best, det3 / det2 / al ** 2)
    if not np.isfinite(best):
        return None
    knock = 1.0 - 0.901 * (1.0 - np.exp(-np.sqrt(r / t) / 16.0))
    return float(knock * best / t)


def _member_stress(m: Member, n_ax, v, mb, tq):
    """(axial stress with its sign at the worse fibre, shear stress) from a
    section's axial force N, shear V, bending moment Mb and torque T."""
    sec = m.section()
    if m.round:
        bend = mb * m.ro / sec["I2"]
        ro, ri = m.ro, m.ri
        shear_v = (4.0 * v / (3.0 * sec["A"])) * (ro * ro + ro * ri + ri * ri) / (ro * ro + ri * ri)
        tau = abs(tq) * ro / sec["J"] + shear_v
    else:
        # mb as (about e2, about e3)
        bend = abs(mb[0]) * 0.5 * m.thickness / sec["I2"] + abs(mb[1]) * 0.5 * m.width / sec["I3"]
        lo = min(m.width, m.thickness)
        tau = abs(tq) * lo / sec["J"] + 1.5 * v / sec["A"]
    ax = n_ax / sec["A"]
    return ax, bend, tau


def _strength_factor(mat: StockMaterial, tens, comp, tau):
    """The factor on the loads at which the section fails, and what by:
    fibre tension with shear (Hashin's quadratic), compression, shear; a
    metal by von Mises yield."""
    if mat.isotropic:
        vm = max(np.sqrt(tens ** 2 + 3.0 * tau ** 2), np.sqrt(comp ** 2 + 3.0 * tau ** 2))
        return (mat.Xt / vm if vm > 0 else np.inf), "yield"
    out = []
    if tens > 0 or tau > 0:
        fi = (max(tens, 0.0) / mat.Xt) ** 2 + (tau / mat.S) ** 2
        out.append((1.0 / np.sqrt(fi) if fi > 0 else np.inf,
                    "shear" if (tau / mat.S) > (max(tens, 0.0) / mat.Xt) else "fibre tension"))
    if comp > 0:
        out.append((mat.Xc / comp, "fibre compression"))
    return min(out) if out else (np.inf, "")


def _free_length(m: Member, held_s) -> list:
    """The member's unsupported lengths as [(s0, s1, K)]: between joints (or
    supports on it) pinned both ends (K 1: the node's rotational restraint
    is not credited); a stretch with a free end, K 2."""
    marks = sorted([(j.span[0], j.span[1]) for j in m.joints] + list(held_s))
    out, s = [], 0.0
    first = True
    for a, b in marks:
        if a - s > 1e-6:
            out.append((s, a, 2.0 if first else 1.0))
        s = max(s, b)
        first = False
    if m.length - s > 1e-6:
        out.append((s, m.length, 2.0))
    return out


def _fmt(x, nd=3):
    if x is None:
        return None
    x = float(x)
    if not np.isfinite(x):
        return None
    return round(x, nd)


def assess(study, mesh, cases, residual=None, goal_sf=None) -> dict:
    """The members and their joints in a solved assembly. `mesh`: the
    fem.HexMesh the cases were solved on (verify's analysis grid); `cases`:
    [(name, u, f)] with u the displacement and f the applied nodal force per
    dof; `residual(u, f)` -> K u - f per dof (the supports' reactions on
    held dofs), or None when nothing on a member is held. Returns the
    report's `stock` block."""
    sc = stock_cells(study, mesh)
    h = mesh.pitch
    xyz = mesh.node_xyz
    names = [c[0] for c in cases]
    pre = names.index("preload") if "preload" in names else None
    goal = float(goal_sf) if goal_sf else None

    def counted(sf, name):
        return sf * (study.case_factor(name) if goal else 1.0)

    # each node's share of a member's interface force, per case: what the printed part puts on the member
    def tractions(p, u, f):
        t = p["k"] @ u - f
        if residual is not None:
            t = t - residual(u, f)
        out = t.reshape(-1, 3)[p["nodes"]]
        return out

    has_print = np.zeros(mesh.n_nodes, bool)
    printed_el = (sc.printed > 0.05) | (sc.owner < 0)
    has_print[np.unique(mesh.enode[printed_el])] = True
    members_out, joints_out, warnings = [], [], []
    worst = (np.inf, "", "")
    for mi, m in enumerate(sc.members):
        p = sc.per[mi]
        mat, sec = m.material, m.section()
        nodes, s_n = p["nodes"], p["s"]
        x_n = xyz[nodes]
        if not len(nodes):
            warnings.append(f"stock {m.id!r} is outside the analysed volume: nothing of it was checked")
            continue
        s_lo, s_hi = float(s_n.min()), float(s_n.max())
        stations = np.arange(s_lo + 0.5 * h, s_hi, h)
        held = ([(max(s_lo, 0.0), min(s_hi, m.length))] if not m.free_spans() else list(m.free_spans())) \
            if m.held else []
        buck_wall = _wall_buckling(m)
        rows = []
        m_sf = (np.inf, "", None, "")
        for ci, (name, u, f) in enumerate(cases):
            fext = (p["k"] @ u).reshape(-1, 3)[nodes]           # everything outside the member acting on it
            best = None
            for s in stations:
                far = s_n > s
                if not far.any() or far.all():
                    continue
                c = m.point(s)
                ff = fext[far].sum(axis=0)
                mo = np.cross(x_n[far] - c, fext[far]).sum(axis=0)
                n_ax = float(ff @ m.axis)
                vv = ff - n_ax * m.axis
                tq = float(mo @ m.axis)
                mbv = mo - tq * m.axis
                if m.round:
                    ax, bend, tau = _member_stress(m, n_ax, float(np.linalg.norm(vv)), float(np.linalg.norm(mbv)), tq)
                else:
                    ax, bend, tau = _member_stress(m, n_ax, float(np.linalg.norm(vv)),
                                                   (float(mbv @ m.e2), float(mbv @ m.e3)), tq)
                tens, comp = max(ax + bend, 0.0), max(bend - ax, 0.0)
                sf, why = _strength_factor(mat, tens, comp, tau)
                if buck_wall is not None and comp > 0 and buck_wall / comp < sf:
                    sf, why = buck_wall / comp, "wall buckling"
                row = {"at": float(s), "axial": n_ax, "shear": float(np.linalg.norm(vv)),
                       "moment": float(np.linalg.norm(mbv)), "torque": tq, "tension": tens, "compression": comp,
                       "shearStress": tau, "sf": sf, "why": why}
                if best is None or sf < best["sf"]:
                    best = row
            if best is None:
                continue
            # Euler buckling of each free length under the axial compression it carries
            euler = None
            ei = mat.E1 * min(sec["I2"], sec["I3"])
            for a, b, kf in _free_length(m, held):
                mid = 0.5 * (a + b)
                far = s_n > mid
                if not far.any() or far.all() or b - a < 2.0 * h:
                    continue
                n_ax = float(fext[far].sum(axis=0) @ m.axis)
                if n_ax < -1e-6 * max(float(np.abs(fext).sum()), 1e-12):
                    pcr = np.pi ** 2 * ei / (kf * (b - a)) ** 2
                    e_sf = pcr / -n_ax
                    if euler is None or e_sf < euler[0]:
                        euler = (e_sf, a, b, kf)
            sf, why = best["sf"], best["why"]
            if euler is not None and euler[0] < sf:
                sf, why = euler[0], "column buckling"
            row = {"name": name, "at": [round(float(v), 3) for v in m.point(best["at"])],
                   "station": _fmt(best["at"], 2), "axialForce": _fmt(best["axial"]),
                   "shearForce": _fmt(best["shear"]), "bendingMoment": _fmt(best["moment"]),
                   "torque": _fmt(best["torque"]), "tensileStress": _fmt(best["tension"]),
                   "compressiveStress": _fmt(best["compression"]), "shearStress": _fmt(best["shearStress"]),
                   "safetyFactor": _fmt(sf), "limitedBy": why}
            if euler is not None and euler[0] < 1e4:      # (a thousandth of a newton of compression is not a column)
                row["columnBuckling"] = {"factor": _fmt(euler[0]), "span": [_fmt(euler[1], 2), _fmt(euler[2], 2)],
                                         "effectiveLengthFactor": euler[3]}
            rows.append(row)
            if counted(sf, name) < m_sf[0]:
                m_sf = (counted(sf, name), name, sf, why)
        mrow = {"id": m.id, "type": m.kind, "material": mat.to_dict(), "length": _fmt(m.length, 2),
                "mass": _fmt(m.mass(), 3),
                "section": {k: _fmt(v, 4) for k, v in sec.items()},
                **({"od": m.od, "wall": m.wall} if m.round else {"width": m.width, "thickness": m.thickness}),
                **({"wallBucklingStress": _fmt(buck_wall, 1)} if buck_wall is not None else {}),
                "cases": rows, "safetyFactor": _fmt(m_sf[2]) if m_sf[2] is not None else None,
                "limitedBy": m_sf[3], "worstCase": m_sf[1]}
        # how well the grid holds the member's own stiffness: its cells' share against the section
        vox_a = float(p["share"].sum()) * h ** 3 / max(s_hi - s_lo, h)
        mrow["gridArea"] = _fmt(vox_a / sec["A"], 3)
        members_out.append(mrow)
        if m_sf[2] is not None and m_sf[0] < worst[0]:
            worst = (m_sf[0], f"member {m.id}", m_sf[3])
        if goal and m_sf[2] is not None and m_sf[0] < goal:
            warnings.append(f"stock {m.id!r} holds {m_sf[2]:.2f}x its loads in case {m_sf[1]!r} ({m_sf[3]}), "
                            f"under the safety factor {goal:g}: a larger section, a stiffer layup or a shorter "
                            "free length")

        # ---- joints
        ny, nz, _ = m.surface_normal(p["y"], p["z"])
        n_w = ny[:, None] * m.e2[None] + nz[:, None] * m.e3[None]        # outward normal per node
        th = np.arctan2(p["z"], p["y"])
        t_all = [tractions(p, u, f) for _, u, f in cases]
        for j in m.joints:
            s0, s1 = j.span
            sel = (s_n >= s0 - 0.5 * h) & (s_n <= s1 + 0.5 * h)
            jid = f"{m.id}/{j.index}"
            if not sel.any():
                warnings.append(f"joint {jid} is outside the analysed volume: not checked")
                continue
            # patches: a cell along the member by two round it
            ns = max(int(round((s1 - s0) / h)), 1)
            ds = (s1 - s0) / ns
            per = 2.0 * np.pi * m.ro if m.round else 2.0 * (m.width + m.thickness)
            nt = max(int(per // (2.0 * h)), 4)
            if m.round:
                ti = np.floor((th + np.pi) / (2.0 * np.pi) * nt).astype(int) % nt
            else:
                # walk the rectangle's perimeter: which face, and how far along it
                hw, ht = 0.5 * m.width, 0.5 * m.thickness
                face_z = np.abs(nz) > 0
                along = np.where(face_z, np.where(nz > 0, p["y"] + hw, 2 * hw + 2 * ht + (hw - p["y"])),
                                 np.where(ny > 0, 2 * hw + (ht - p["z"]), 4 * hw + 2 * ht + (p["z"] + ht)))
                ti = np.clip((along / per * nt).astype(int), 0, nt - 1)
            si = np.clip(((s_n - s0) / ds).astype(int), 0, ns - 1)
            area = per * (s1 - s0) / (ns * nt)
            bonded = np.zeros((ns, nt), bool)
            np.logical_or.at(bonded, (si[sel], ti[sel]), has_print[nodes][sel])
            cover = float(bonded.mean())
            c_j = m.point(0.5 * (s0 + s1))
            r_mid = m.ro if m.round else 0.5 * np.hypot(m.width, m.thickness)
            window = max(int(round(min(j.bond["overlap"] if j.bond else 12.7, s1 - s0) / ds)), 1)

            def patches(t):
                tn = np.einsum("ij,ij->i", t, n_w)                      # + pulls the surface outward
                ts = t - tn[:, None] * n_w
                g_n = np.zeros((ns, nt))
                g_s = np.zeros((ns, nt, 3))
                np.add.at(g_n, (si[sel], ti[sel]), tn[sel])
                np.add.at(g_s, (si[sel], ti[sel]), ts[sel])
                return g_n / area, g_s / area                           # MPa

            def windowed(g):
                if window <= 1 or ns <= 1:
                    return g
                k = np.ones(window) / window
                return np.apply_along_axis(lambda v: np.convolve(v, k, mode="valid"), 0, g)

            pre_n = patches(t_all[pre])[0] if pre is not None else None
            jrows, j_sf = [], (np.inf, "", None, "")
            for ci, (name, u, f) in enumerate(cases):
                t = t_all[ci]
                ft = t[sel].sum(axis=0)
                mo = np.cross(x_n[sel] - c_j, t[sel]).sum(axis=0)
                f_ax, tq = float(ft @ m.axis), float(mo @ m.axis)
                vv = float(np.linalg.norm(ft - f_ax * m.axis))
                mb = float(np.linalg.norm(mo - tq * m.axis))
                g_n, g_s = patches(t)
                tau = np.linalg.norm(g_s, axis=2)
                row = {"name": name, "axialForce": _fmt(f_ax), "torque": _fmt(tq), "shearForce": _fmt(vv),
                       "bendingMoment": _fmt(mb)}
                # what would slide the member in its seat: along it and round it; a plate, along its faces
                slide = float(np.hypot(f_ax, tq / r_mid)) if m.round else \
                    float(np.linalg.norm(ft - (ft @ m.e3) * m.e3))
                checks = []
                a_b = max(cover, 1e-9) * per * (s1 - s0)
                row["meanShear"] = _fmt(slide / a_b)
                if j.kind == "bond":
                    tw = np.linalg.norm(windowed(g_s), axis=2)
                    nw = np.maximum(windowed(g_n), 0.0)
                    fi = (tw / j.bond["strength"]) ** 2 + (nw / j.bond["tension"]) ** 2
                    k = np.unravel_index(int(np.argmax(fi)), fi.shape)
                    sfb = 1.0 / np.sqrt(fi[k]) if fi[k] > 0 else np.inf
                    row.update(bondShear=_fmt(tw[k]), bondPeel=_fmt(nw[k]))
                    checks.append((sfb, "bond shear" if tw[k] / j.bond["strength"] >= nw[k] / j.bond["tension"]
                                   else "bond peel"))
                if j.kind == "clamp":
                    if name == "preload":
                        jrows.append(row)
                        continue
                    if pre_n is not None and j.fasteners:
                        squeeze = np.maximum(-(pre_n + g_n), 0.0)
                        normal = float(squeeze.sum() * area)
                    else:
                        normal = float(j.clamp_force or 0.0)
                    row["clampForce"] = _fmt(normal)
                    row["slipForce"] = _fmt(slide)
                    checks.append((j.friction * normal / slide if slide > 1e-9 else np.inf, "clamp slip"))
                if j.kind == "pin":
                    r_w = m.ro - 0.5 * m.wall if m.kind == "tube" else r_mid
                    walls = 2.0 if m.kind == "tube" else 1.0
                    t_w = m.wall if m.kind == "tube" else (m.od if m.kind == "rod" else m.thickness)
                    if m.round:
                        per_wall = float(np.hypot(f_ax / walls, tq / (walls * r_w)))
                    else:
                        per_wall = float(np.linalg.norm(ft - (ft @ np.asarray(j.pin_axis)) * np.asarray(j.pin_axis)))
                    bearing = per_wall / (j.pin_d * t_w)
                    edge = min(j.pin_at, m.length - j.pin_at)
                    shear_out = per_wall / (2.0 * max(edge, 1e-6) * t_w)
                    net_a = sec["A"] - walls * j.pin_d * t_w
                    net = abs(f_ax) / max(net_a, 1e-9)
                    row.update(bearingStress=_fmt(bearing), shearOutStress=_fmt(shear_out), netStress=_fmt(net),
                               edgeDistance=_fmt(edge, 2))
                    checks.append((mat.Xc / bearing if bearing > 0 else np.inf, "bearing at the bore"))
                    checks.append((mat.S / shear_out if shear_out > 0 else np.inf, "shear-out to the end"))
                    checks.append(((mat.Xt if f_ax >= 0 else mat.Xc) / net if net > 0 else np.inf, "net section"))
                # the member's wall under the pressure on it (with the clamp's preload, when there is one)
                g_tot = g_n + pre_n if (pre_n is not None and name != "preload") else g_n
                crush = _crush(m, g_tot, ds, nt)
                if crush is not None:
                    row.update(wallPressure=_fmt(crush["pressure"]), wallStress=_fmt(crush["stress"]))
                    checks.append((crush["sf"], crush["why"]))
                sf, why = min(checks) if checks else (np.inf, "")
                row.update(safetyFactor=_fmt(sf), limitedBy=why)
                jrows.append(row)
                if counted(sf, name) < j_sf[0]:
                    j_sf = (counted(sf, name), name, sf, why)
            jrow = {"id": jid, "member": m.id, "type": j.kind, "span": [_fmt(s0, 2), _fmt(s1, 2)],
                    "length": _fmt(s1 - s0, 2), "area": _fmt(per * (s1 - s0), 1), "contact": _fmt(cover, 3),
                    **({"adhesive": {k: (round(v, 4) if isinstance(v, float) else v) for k, v in j.bond.items()}}
                       if j.bond else {}),
                    **({"friction": j.friction, "fasteners": list(j.fasteners)} if j.kind == "clamp" else {}),
                    **({"pinDiameter": j.pin_d, "pinAt": j.pin_at} if j.kind == "pin" else {}),
                    "cases": jrows, "safetyFactor": _fmt(j_sf[2]) if j_sf[2] is not None else None,
                    "limitedBy": j_sf[3], "worstCase": j_sf[1]}
            if (0 in p["buriedEnds"] and s0 < h) or (1 in p["buriedEnds"] and s1 > m.length - h):
                jrow["endFace"] = f"not counted as joined (the member's last {h:.2g} mm carries nothing)"
            if j_sf[2] is not None and np.isfinite(j_sf[2]) and j.kind == "bond" and "shear" in j_sf[3] and goal:
                # shear over the overlap falls as 1 / length until the joint is longer than the test overlap
                need = (s1 - s0) * goal / max(j_sf[0], 1e-9)
                jrow["lengthForGoal"] = _fmt(need, 1)
            joints_out.append(jrow)
            if j_sf[2] is not None and j_sf[0] < worst[0]:
                worst = (j_sf[0], f"joint {jid}", j_sf[3])
            if s0 <= 1e-6 and s1 >= m.length - 1e-6:
                warnings.append(f"joint {jid} runs the whole length of {m.id!r}: the design space covers all of it, "
                                "so it is read as one sleeve bonded end to end. For sockets at the ends and a free "
                                "tube between, give `joints` with a `span` each")
            if cover < 0.6:
                warnings.append(f"joint {jid}: the printed part touches {cover:.0%} of the member there; "
                                "a bond or a clamp wants a full sleeve (joint.sleeve)")
            if goal and j_sf[2] is not None and j_sf[0] < goal:
                tip = {"bond shear": "a longer joint (extend the design space along the member, or joint.span), "
                                     "a stronger adhesive on this plastic, or a pin through it",
                       "bond peel": "a longer joint, so the moment is carried over more length, or a clamp",
                       "clamp slip": "more screws or preload, a longer clamp, or friction paste "
                                     "(joint.friction)",
                       "bearing at the bore": "a larger pin or a thicker wall",
                       "shear-out to the end": "the pin further from the member's end",
                       "net section": "a smaller pin or a larger member"}.get(j_sf[3],
                                                                                "less pressure on the wall: a longer joint, or a thicker-walled member")
                warnings.append(f"joint {jid} ({j.kind}) holds {j_sf[2]:.2f}x its loads in case {j_sf[1]!r} "
                                f"({j_sf[3]}), under the safety factor {goal:g}: {tip}")
    out = {"members": members_out, "joints": joints_out,
           "mass": _fmt(sum(m.mass() for m in sc.members), 3),
           "safetyFactor": _fmt(worst[0]) if np.isfinite(worst[0]) else None,
           "limitedBy": worst[1], "mode": worst[2],
           "interface": "stuck (shared nodes): a bond; a clamp while its slip check holds"}
    if goal and np.isfinite(worst[0]):
        out["margin"] = _fmt(worst[0] / goal, 4)
    if warnings:
        out["warnings"] = warnings
    return out


def _crush(m: Member, g_n, ds, nt):
    """A round member's wall under the pressure the joint puts on it.
    `g_n` (stations, sectors): the normal traction, MPa, + outward. The
    pressure is averaged along the member over the width of wall that
    resists a ring of load (2 / beta, the cylinder's bending boundary
    layer), then split round the circumference: the uniform part is hoop
    stress p r / t, each ovalizing harmonic n >= 2 bends the wall with
    M = p_n r^2 / (n^2 - 1) (an inextensional ring; the n = 1 part is the
    member's shear, already read). A rod reads the peak pressure against
    its transverse compressive strength."""
    if not m.round:
        p = float(np.maximum(-g_n, 0.0).max()) if g_n.size else 0.0
        if p <= 0:
            return None
        return {"pressure": p, "stress": p, "sf": m.material.Zc / p, "why": "crushing through the thickness"}
    mat = m.material
    if m.kind == "rod":
        p = float(np.maximum(-g_n, 0.0).max()) if g_n.size else 0.0
        if p <= 0:
            return None
        return {"pressure": p, "stress": p, "sf": mat.Yc / p, "why": "crushing across the rod"}
    t, r = m.wall, m.ro - 0.5 * m.wall
    nu = mat.nu12 * mat.nu12 * mat.E2 / mat.E1
    lam = (mat.E1 * t * t * r * r / (3.0 * (1.0 - nu) * mat.E2)) ** 0.25      # 1 / beta, orthotropic
    w = max(int(round(2.0 * lam / ds)), 1)
    g = g_n * (m.ro / r)                         # the same force on the wall's mid-surface
    if w > 1 and g.shape[0] > 1:
        k = np.ones(min(w, g.shape[0])) / min(w, g.shape[0])
        g = np.apply_along_axis(lambda v: np.convolve(v, k, mode="valid"), 0, g)
    worst = None
    for row in g:
        c = np.fft.rfft(row) / nt
        p0 = float(c[0].real)
        hoop = p0 * r / t                                        # + tension
        mom = sum(2.0 * abs(c[n]) * r * r / (n * n - 1.0) for n in range(2, min(len(c), nt // 2)))
        bend = 6.0 * mom / (t * t)
        tens, comp = bend + max(hoop, 0.0), bend + max(-hoop, 0.0)
        cands = []
        if tens > 0:
            cands.append((mat.Yt / tens, "wall splitting (hoop tension)", tens))
        if comp > 0:
            cands.append((mat.Yc / comp, "wall crushing (hoop compression)", comp))
        if p0 < 0:
            # a ring's collapse under external pressure: p_cr = 3 E I / r^3 per unit length
            pcr = mat.E2 * t ** 3 / (4.0 * r ** 3)
            cands.append((pcr / -p0, "wall collapse (external pressure)", -p0 * r / t))
        if not cands:
            continue
        sf, why, stress = min(cands)
        if worst is None or sf < worst["sf"]:
            worst = {"pressure": float(np.abs(row).max()), "stress": float(stress), "sf": float(sf), "why": why}
    return worst
