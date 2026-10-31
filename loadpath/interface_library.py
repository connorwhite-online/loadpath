"""
The interface library: how strongly two printed materials hold to each
other, every number with its source.

A multi-material study (an AMS FDM print of PETG and TPU, a PolyJet part of
Vero and Agilus30) has a boundary where one material meets the other. That
boundary is usually the weakest place in the part, and the optimizer checks
the stress across it against these numbers. Same-material boundaries are not
here: a material's own across-layer strength (material_library strengthZ,
or the process interlayer factor) covers them.

Each value is (MPa, source id, note), as in material_library.py; `SOURCES`
says what each id is and where to read it. The papers were read on
2026-10-31.

Two orientations. "tensile"/"shear" are for a boundary inside a layer (the
two materials side by side, the nozzle switching within each layer);
"tensileZ"/"shearZ" are for a boundary between layers (one material printed
on top of the other). A stacked boundary is the weaker one in the FDM data:
Xue et al. 2022 found the in-layer boundary gets a micro zig-zag interlock
from each extruder switch, and the stacked one is a smooth plane. Where a
source measured only one orientation, the other takes the same number
(source "estimate").

How a number is chosen. The lowest reported configuration that a slicer
produces by default: a flat butt joint, no designed interlocking, the lower
end of the reported replicates, the less favourable print order. Most
papers report tensile (butt joint) strength; where shear was not measured
it is 0.6 x tensile (source "estimate"): the von Mises ratio 1/sqrt(3),
which matches the two cross-checks we have (PLA/TPU stacked: 0.37 MPa shear,
Brancewicz-Steinmetz 2021, against 0.49-0.58 MPa tensile, Tamburrino 2019;
ABS/ABS lap shear 14.9 MPa, Harris 2019, against 25.4 MPa across-layer
tensile, Polymaker). Where only lap shear was measured the tensile value is
taken equal to it, not scaled up.

A pair not in the table gets `DEFAULT_FRACTION` (5%) of the weaker
material's across-layer strength, flagged conservative. The poorly bonding
pairs measured here sit around it (PLA/TPU stacked 0.29 MPa = 4.5% of TPU's
6.4 MPa across layers); pairs with affinity sit well above (PETG/TPU 23%,
ABS/TPU 73%); only PLA/PETG, which is sold as a support pair because it
does not bond, sits below.

Caveat: interface strength is a process property more than a material one.
It moves by factors of several with the nozzle temperature of the second
material (Xue 2022: 0.39 to 5.1 MPa for PLA/TPU), the layer height, the
print order (Tamburrino 2019: PETG on PLA 0.56-0.82 MPa, PLA on PETG
0.04-0.08), the chamber and bed temperature (Yin et al. 2018: bed
temperature dominates ABS/TPU), and any interlocking geometry (dovetails,
interleaved layers: 2-15x). PLA and PETG barely bond at all; Bambu Lab
uses each as a breakaway support interface for the other. A part that
leans on a boundary gets coupons printed on the target printer.
"""
from __future__ import annotations

from dataclasses import dataclass

from .material_library import Source

__all__ = ["SOURCES", "INTERFACES", "InterfaceCard", "DEFAULT_FRACTION", "SHEAR_FROM_TENSILE",
           "interface_strength", "describe_interface", "interface_names"]

ACCESSED = "2026-10-31"

# unknown pair: this share of the weaker material's across-layer strength
DEFAULT_FRACTION = 0.05
# shear strength = this x tensile where only tensile was measured (1/sqrt(3))
SHEAR_FROM_TENSILE = 0.6

SOURCES = {
    "tamburrino-2019": Source(
        "Tamburrino, Graziosi, Bordegoni, 'The influence of slicing parameters on the multi-material adhesion "
        "mechanisms of FDM printed parts: an exploratory study', Virtual and Physical Prototyping 14(4), 2019, "
        "doi:10.1080/17452759.2019.1607758 (accepted manuscript)",
        "https://re.public.polimi.it/retrieve/e0c31c10-852a-4599-e053-1705fe0aef77/Accepted_manuscript_VPP.pdf",
        "butt-joined bars printed upright (the boundary is between layers), Ultimaker 3, 0.3 mm layers, open "
        "chamber, 3 replicas; Ultimaker CPE (a PETG-family copolyester) 250 C, PLA 205 C, TPU 95A 225 C; peak "
        "stress = breaking load / bonded area, read off Figs. 10 and 22"),
    "xue-2022": Source(
        "Xue, Boudaoud, Robin, Cruz Sanchez, Daya, 'Influence of layer thickness and nozzle temperature on the "
        "interlocking adhesion strength of additive manufactured multi-material interface', Solid Freeform "
        "Fabrication Symposium 2022",
        "https://utw10945.utweb.utexas.edu/sites/default/files/2022/"
        "Influence%20of%20Layer%20Thickness%20and%20Nozzle%20Temperatur.pdf",
        "PLA/TPU butt-joint tensile, specimens printed flat (boundary inside each layer), Raise3D E2, PLA "
        "210 C, TPU 230-250 C, 0.1-0.3 mm layers, n=4; all failed at the boundary (Table 2)"),
    "brancewicz-2021": Source(
        "Brancewicz-Steinmetz, Sawicki, Byczkowska, 'The Influence of 3D Printing Parameters on Adhesion "
        "between Polylactic Acid (PLA) and Thermoplastic Polyurethane (TPU)', Materials 14(21):6464, 2021",
        "https://pmc.ncbi.nlm.nih.gov/articles/PMC8585249",
        "two coaxial cylinders printed one on the other, sheared across the boundary; Ultimaker PLA 200 C, "
        "Fillamentum Flexfill TPU 98A 210 C, 0.2 mm layers; mean over surface patterns 0.37 MPa (PLA first) "
        "and 0.39 MPa (TPU first)"),
    "zatloukal-2025": Source(
        "Zatloukal, Viry, Mizera, Stoklasek, Miskarik, Bednarik, 'Optimizing Interfacial Adhesion and "
        "Mechanical Performance of Multimaterial Joints Fabricated by Material Extrusion', Materials "
        "18(16):3846, 2025",
        "https://pmc.ncbi.nlm.nih.gov/articles/PMC12387702/",
        "ISO 527-1 type 1A bars with a standard butt joint (Test 2), n=10, Original Prusa XL, 0.2 mm layers, "
        "100% infill; PLA 225 C, PETG 250 C, ASA 260 C; tooth joint and interlayer-pressed joint also tested"),
    "harris-2019": Source(
        "Harris, Jursik, Rochefort, Walker, 'Additive Manufacturing With Soft TPU - Adhesion Strength in "
        "Multimaterial Flexible Joints', Frontiers in Mechanical Engineering 5:37, 2019",
        "https://www.frontiersin.org/journals/mechanical-engineering/articles/10.3389/fmech.2019.00037/full",
        "single lap shear (ASTM D1002/D3164), one 0.2 mm TPU layer between ABS or ASA substrates, 4-5 mm "
        "overlap, TPU 218 C, ABS/ASA 240 C, bed 110 C, chamber 70 C, n=4; failed by disbonding at the top "
        "(TPU-under-substrate) boundary"),
    "aberdeen-2024": Source(
        "Aberdeen, Forghani, Sochol, 'Geometric determinants of material jetting-enabled bi-material interface "
        "integrity using PolyJet 3D printing', ASME MSEC2024-122414, 2024",
        "https://par.nsf.gov/servlets/purl/10620996",
        "VeroWhite / Agilus30 Black bars, 30 x 10 mm flat (control) boundary, Objet500 Connex3, printed "
        "flat, matte, n=4, ASTM D638-guided at 12.5 mm/min; the flat control reached 1.074 MPa and failed "
        "in the Agilus30, not at the boundary"),
    "bambu-pla-petg": Source(
        "Bambu Lab Wiki, 'Printing guide for using PLA Basic and PETG HF/PETG Basic for support'",
        "https://wiki.bambulab.com/en/filament-acc/filament/h2d-pla-and-petg-mutual-support",
        "'PLA and PETG don't bond well to each other - and that's exactly what makes them a great support "
        "pair'"),
    "stratasys-overmold": Source(
        "Stratasys Direct, 'PolyJet Flex & PolyJet Over-Mold' material specifications (02/2018)",
        "https://www.rcoeng.com/hubfs/pj_flex_and_overmold_material_datasheet_201802.pdf",
        "rigid VeroWhitePlus and a Shore A Agilus30-based material printed in one build from two STL files; "
        "Agilus30 + Vero digital materials 40A-90A reach 3-14 MPa; no interface strength given"),
    "estimate": Source(
        "loadpath estimate: no measured value for this pair or orientation", "",
        "derived as the note says (shear = 0.6 x tensile; the other orientation equal to the measured one); "
        "replace with coupons printed on your printer where it matters"),
    "default": Source(
        "loadpath default for a pair with no data", "",
        f"{DEFAULT_FRACTION:.0%} of the weaker material's across-layer strength (module docstring)"),
}

_S = SHEAR_FROM_TENSILE


def _est_shear(t, what="tensile"):
    return (round(_S * t, 3), "estimate", f"{_S} x the {what} value (shear not measured)")


# key: the two library material keys, sorted, joined with "+". "bondSource":
# the source of the bond rating where no value cites it
INTERFACES = {
    # ------------------------------------------------------------- FDM
    "petg+tpu": {
        "process": "fdm", "bond": "good",
        "tensile": (1.48, "estimate", "taken equal to the stacked value; no in-layer test found (in-layer "
                    "boundaries tested at least as strong for PLA/TPU, Xue 2022)"),
        "shear": _est_shear(1.48),
        "tensileZ": (1.48, "tamburrino-2019", "CPE/TPU butt joint, CPE printed first, lowest replicate "
                     "(zig-zag top layers 1.48-2.27, default lines 1.84-2.24)"),
        "shearZ": _est_shear(1.48),
        "note": "the best-bonding FDM pair tested by Tamburrino 2019: interlocking did not raise it, failed "
                "joints stayed partly joined. Print the PETG first; TPU first was weaker.",
    },
    "pla+tpu": {
        "process": "fdm", "bond": "poor",
        "tensile": (1.55, "xue-2022", "in-layer butt joint, TPU 230 C, 0.2 mm layers (0.39-5.06 MPa over "
                    "230-250 C and 0.1-0.3 mm; 2.68 at 240 C)"),
        "shear": _est_shear(1.55),
        "tensileZ": (0.29, "tamburrino-2019", "stacked butt joint, PLA first, lowest replicate (zig-zag "
                     "0.29-0.44, default lines 0.49-0.58)"),
        "shearZ": (0.37, "brancewicz-2021", "stacked coaxial cylinders in shear, PLA first, mean of the "
                   "surface-pattern series (TPU first 0.39)"),
        "note": "low affinity; every Xue 2022 specimen failed at the boundary. Hotter TPU and thinner layers "
                "at the boundary help most; plasma treatment lifts ~0.5 to >=0.7 MPa (LIST, 2025).",
    },
    "petg+pla": {
        "process": "fdm", "bond": "none", "bondSource": "bambu-pla-petg",
        "tensile": (1.6, "zatloukal-2025", "standard butt joint, ISO 527 1A bar (vs PLA 50, PETG 43 MPa "
                    "alone); a tooth joint reached 4.0"),
        "shear": _est_shear(1.6),
        "tensileZ": (0.04, "tamburrino-2019", "stacked butt joint, PLA printed first, lowest replicate "
                     "(0.04-0.08; with the PETG first 0.56-0.82). Some bars broke in the grips"),
        "shearZ": _est_shear(0.04),
        "note": "do not design on this boundary: Bambu Lab uses PLA and PETG as breakaway supports for each "
                "other because they don't bond. Interleaved, pressed layers (Zatloukal 2025) reached 25 MPa, "
                "so a mechanical interlock is the only way to carry load across it.",
    },
    "abs+tpu": {
        "process": "fdm", "bond": "fair",
        "tensile": (4.7, "estimate", "taken equal to the stacked lap-shear value, not scaled up"),
        "shear": (4.7, "estimate", "taken equal to the stacked lap-shear value"),
        "tensileZ": (4.7, "estimate", "taken equal to the lap-shear value, not scaled up; no butt-joint test"),
        "shearZ": (4.7, "harris-2019", "single lap shear, TPU printed on ABS, 4.7 +- 0.2 MPa (ABS/ABS 14.9)"),
        "note": "measured with a 70 C chamber and 110 C bed; an open printer bonds less (Yin et al. 2018: "
                "bed temperature is the main factor for ABS/TPU). Print the ABS first.",
    },
    "asa+tpu": {
        "process": "fdm", "bond": "fair",
        "tensile": (6.4, "estimate", "taken equal to the stacked lap-shear value, not scaled up"),
        "shear": (6.4, "estimate", "taken equal to the stacked lap-shear value"),
        "tensileZ": (6.4, "estimate", "taken equal to the lap-shear value, not scaled up; no butt-joint test"),
        "shearZ": (6.4, "harris-2019", "single lap shear, TPU printed on ASA, 6.4 +- 0.3 MPa (ASA/ASA 16.1)"),
        "note": "as ABS/TPU: 70 C chamber, 110 C bed; less on an open printer. Print the ASA first.",
    },
    "asa+petg": {
        "process": "fdm", "bond": "fair",
        "tensile": (8.8, "zatloukal-2025", "standard butt joint, ISO 527 1A bar (8.9 in the discussion); a "
                    "tooth joint reached 27.1"),
        "shear": _est_shear(8.8),
        "tensileZ": (8.8, "estimate", "taken equal to the butt-joint value (orientation not stated)"),
        "shearZ": _est_shear(8.8),
        "note": "bonded mostly at the specimen edges in the standard joint (Zatloukal 2025).",
    },
    "pa-cf+pla": {
        "process": "fdm", "bond": "poor",
        "tensile": (0.85, "estimate", "no published test found: the default, 5% of PLA's 17 MPa across layers"),
        "shear": _est_shear(0.85),
        "tensileZ": (0.85, "estimate", "as tensile"),
        "shearZ": _est_shear(0.85),
        "note": "unsourced. Nylon prints at 260-300 C, far above PLA's 60 C glass transition, and the two are "
                "chemically unlike; Bambu Lab sells a separate breakaway 'Support for PA/PET'. Test coupons.",
    },
    "pa-cf+petg": {
        "process": "fdm", "bond": "poor",
        "tensile": (0.9, "estimate", "no published test found: the default, 5% of PETG's 18 MPa across layers"),
        "shear": _est_shear(0.9),
        "tensileZ": (0.9, "estimate", "as tensile"),
        "shearZ": _est_shear(0.9),
        "note": "unsourced; nylon and copolyester are chemically unlike and carbon fibre reduces the polymer "
                "at the surface. Test coupons.",
    },
    # --------------------------------------------------------- jetting
    "agilus30+vero": {
        "process": "jetting", "bond": "good", "bondSource": "stratasys-overmold",
        "tensile": (1.07, "aberdeen-2024", "flat VeroWhite/Agilus30 boundary in tension (1.074 MPa); it "
                    "failed in the Agilus30, so the bond is at least this"),
        "shear": _est_shear(1.07),
        "tensileZ": (1.07, "estimate", "taken equal to the in-plane value: the materials are jetted and "
                     "UV-cured together, layer by layer, in both orientations"),
        "shearZ": _est_shear(1.07),
        "note": "co-cured in one build (Stratasys over-mold); the boundary is limited by the soft side. "
                "1.07 MPa is below Agilus30's datasheet 2.4-3.1 MPa: size and print effects, keep the lower.",
    },
}

_PROCESS_ALIASES = {"polyjet": "jetting", "material jetting": "jetting", "fff": "fdm", "fdm": "fdm"}


@dataclass(frozen=True)
class InterfaceCard:
    pair: tuple             # the two material keys, sorted
    tensile: float          # MPa, boundary inside a layer (side by side)
    shear: float            # MPa
    bond: str               # good | fair | poor | none | unknown
    source: str             # the source ids, comma separated ("default" for an unknown pair)
    note: str
    conservative: bool      # True: no measurement for this pair (a default or an estimate)
    tensile_z: float | None = None   # MPa, boundary between layers (stacked)
    shear_z: float | None = None
    process: str = ""

    def to_dict(self):
        return {"pair": list(self.pair), "tensile": self.tensile, "shear": self.shear, "bond": self.bond,
                "tensileZ": self.tensile_z, "shearZ": self.shear_z, "source": self.source,
                "note": self.note, "conservative": self.conservative, "process": self.process}


def _key(m):
    """A material (name, alias, '<base>-custom' name or Material) as its
    library key."""
    from .materials import _ALIASES   # lazy: materials imports material_library
    name = str(getattr(m, "name", m)).strip().lower()
    while name.endswith("-custom"):
        name = name[: -len("-custom")]
    return _ALIASES.get(name, name)


def _material(m, key):
    """The Material for `m`: its own object if it is one, else the library's."""
    from .materials import MATERIALS
    if hasattr(m, "strength"):
        return m
    if key in MATERIALS:
        return MATERIALS[key]
    raise ValueError(f"unknown material {m!r}; known: {sorted(MATERIALS)}")


def interface_names():
    return sorted(INTERFACES)


def _pair_key(a, b):
    ka, kb = _key(a), _key(b)
    if ka == kb:
        raise ValueError(f"{ka!r} with itself is not an interface: use the material's own across-layer "
                         "strength (strengthZ / process interlayer)")
    return tuple(sorted((ka, kb)))


def interface_strength(a, b, process=None):
    """The InterfaceCard for materials `a` and `b` (names, aliases or
    Material objects; order does not matter). `process` ("fdm",
    "jetting"...), when given, must match the measured pair's process;
    otherwise, and for a pair with no data, the card is the conservative
    default: DEFAULT_FRACTION of the weaker across-layer strength."""
    pair = _pair_key(a, b)
    proc = _PROCESS_ALIASES.get(str(process).strip().lower(), str(process).strip().lower()) \
        if process is not None else None
    e = INTERFACES.get("+".join(pair))
    if e is not None and (proc is None or proc == e["process"]):
        srcs = []
        for k in ("tensile", "shear", "tensileZ", "shearZ"):
            if k in e and e[k][1] not in srcs:
                srcs.append(e[k][1])
        if e.get("bondSource") and e["bondSource"] not in srcs:
            srcs.append(e["bondSource"])
        return InterfaceCard(pair, float(e["tensile"][0]), float(e["shear"][0]), e["bond"],
                             ",".join(srcs), e["note"], all(s == "estimate" for s in srcs),
                             float(e["tensileZ"][0]) if "tensileZ" in e else None,
                             float(e["shearZ"][0]) if "shearZ" in e else None, e["process"])
    weak = min((m.strength_z or m.strength) for m in (_material(a, _key(a)), _material(b, _key(b))))
    t = round(DEFAULT_FRACTION * weak, 3)
    why = (f"no data for {pair[0]}+{pair[1]}" + (f" by {proc}" if e is not None else "")
           + f": {DEFAULT_FRACTION:.0%} of the weaker across-layer strength ({weak:g} MPa); print coupons")
    return InterfaceCard(pair, t, round(SHEAR_FROM_TENSILE * t, 3), "unknown", "default", why, True,
                         t, round(SHEAR_FROM_TENSILE * t, 3), proc or "")


def describe_interface(a, b, process=None):
    """An interface as JSON for reports and the CLI: each value with its
    source's title and url."""
    card = interface_strength(a, b, process)
    e = INTERFACES.get("+".join(card.pair)) if card.source != "default" else None
    vals = {}
    if e is not None:
        for k in ("tensile", "shear", "tensileZ", "shearZ"):
            if k in e:
                v, src, note = e[k]
                s = SOURCES[src]
                vals[k] = {"value": v, "note": note, "source": s.title, **({"url": s.url} if s.url else {})}
    else:
        s = SOURCES["default"]
        for k, v in (("tensile", card.tensile), ("shear", card.shear), ("tensileZ", card.tensile_z),
                     ("shearZ", card.shear_z)):
            vals[k] = {"value": v, "note": card.note, "source": s.title}
    bs = SOURCES.get(e.get("bondSource", "")) if e is not None else None
    return {"pair": "+".join(card.pair), "process": card.process, "bond": card.bond, "note": card.note,
            **({"bondSource": {"source": bs.title, "url": bs.url, "note": bs.note}} if bs else {}),
            "conservative": card.conservative, "values": vals, "accessed": ACCESSED}
