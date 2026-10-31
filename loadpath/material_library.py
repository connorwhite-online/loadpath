"""
The material library: one card per material, every number with its source.

A study names a material ("petg", "pa12", "ti64"); materials.MATERIALS is
built from these cards, and so are thermal.CONDUCTIVITY and the across-
layer factors a card supplies to a layered process (spec.study_from_dict).
Each value is (number, source id, note); `SOURCES` says what each id is and
where to read it. The datasheets were read on 2026-10-30.

How a card is chosen. Each generic name takes its values from one named
datasheet, so the numbers on a card are consistent with each other (the
same printer, slicer profile and test). Brands differ more than the
anisotropy does: PETG's in-plane strength reads 47 MPa on Prusament's
sheet and 32 MPa on Polymaker's, and PLA's across-layer strength is 33% of
in-plane on Prusament's (an interlayer adhesion test) and 93% on
Polymaker's (a Z-printed tensile bar). docs/materials.md lists the spread.
A part that matters gets coupons printed and a calibrated card
(calibration.py), which overrides the library card of the same name.

What the code reads:
  E, nu, strength, density      every analysis (materials.Material)
  strengthZ                     across-layer strength -> process.interlayer
  EZ                            across-layer modulus -> process stiffnessAcrossLayers
  conductivity                  thermal.CONDUCTIVITY (heat paths, thermal TO)
  hdt                           heat deflection temperature at 0.45 MPa: the
                                temperature checks warn above it (durability)
  cte                           thermal expansion, um/(m K) -> thermal.EXPANSION
  retention                     modulus kept vs temperature -> thermal.MODULUS_RETENTION
                                (and, as a proxy, strength: durability)
The rest (elongation, Tg, hdt18, fatigue) is carried for the report and
for people; fatigue and creep design values live in durability.DURABILITY,
which cites its own literature.

Values a datasheet does not give are marked with the source "estimate"
(Poisson's ratio for polymers, mostly) and say why that number.
"""
from __future__ import annotations

from dataclasses import dataclass

__all__ = ["Source", "SOURCES", "LIBRARY", "card", "card_names", "value", "describe"]

ACCESSED = "2026-10-30"


@dataclass(frozen=True)
class Source:
    title: str
    url: str = ""
    note: str = ""

    def to_dict(self):
        return {"title": self.title, "url": self.url, "note": self.note}


SOURCES = {
    "prusament-pla": Source(
        "Prusament PLA technical datasheet, v1.1 (27-07-2022), Prusa Polymers",
        "https://www.matterhackers.com/r/xFvGl8",
        "ISO 527-1 on specimens printed on an Original Prusa i3 MK3, 100% rectilinear infill; "
        "'Vertical xz' is printed on edge, not across layers; interlayer adhesion is Prusa's own test"),
    "prusament-petg": Source(
        "Prusament PETG technical datasheet, v1.1 (16-02-2022), Prusa Polymers",
        "https://www.3dprintergear.com.au/assets/files/technical-data-sheet%20(1).pdf",
        "as Prusament PLA"),
    "polylite-abs": Source(
        "Polymaker PolyLite ABS TDS v5.2",
        "https://cdn.shopify.com/s/files/1/0789/2979/files/PolyLite_ABS_TDS_V5.2.pdf",
        "ISO 527 printed X-Y and Z specimens, 100% infill"),
    "polylite-asa": Source(
        "Polymaker PolyLite ASA TDS v5.1",
        "https://cdn.shopify.com/s/files/1/0603/6609/files/PolyLite_ASA_TDS_V5.1.pdf",
        "ISO 527 printed X-Y and Z specimens, 100% infill"),
    "ultimaker-tpu95a": Source(
        "UltiMaker TPU 95A technical data sheet v2.01 (15-09-2025)",
        "https://um-support-files.ultimaker.com/materials/2.85mm/tds/TPU-95A/TDS-TPU95A-V2.01-250915.pdf",
        "ASTM D3039 printed XY / YZ / Z specimens"),
    "bambu-paht-cf": Source(
        "Bambu Lab PAHT-CF technical data sheet v2",
        "https://uwaterloo.ca/rapid-prototyping-centre/sites/default/files/uploads/documents/"
        "bambu_paht-cf_technical_data_sheet_v2.pdf",
        "ISO 527, dry state, printed X-Y and Z specimens, 100% infill"),
    "hp-5200-whitepaper": Source(
        "HP 3D Printing materials for the HP Jet Fusion 5200 Series: Mechanical Properties (white paper)",
        "https://info.sculpteo.com/hubfs/Material%20documentation/PA11%20HP/"
        "TDS%20HP%20pa12%20pa12gb%20pa11%20pp%20(1).pdf",
        "ASTM D638 Type I, Balanced print profile, 349-part test job, XY and Z averages"),
    "basf-tpu88a": Source(
        "BASF Forward AM Ultrasint TPU 88A technical data sheet v2.2 (08/2021)",
        "https://3dprinting.rpamerica.us/hubfs/BASF/UltrasintTM%20TPU%20Line.pdf",
        "laser sintered; DIN 53504 S2 tensile, ISO 527-2 1A modulus, X and Z"),
    "formlabs-standard": Source(
        "Formlabs Standard Resins (Clear, White, Grey, Black) material properties, Rev 02 (14-08-2023)",
        "https://8baht.com/storage/product_dowloads/65d4b84c383c5.pdf",
        "ASTM D638-14, Form 2 at 100 um, post-cured; Grey V4"),
    "formlabs-tough2000": Source(
        "Formlabs Tough 2000 Resin technical data sheet V1, Rev 01 (10-07-2020)",
        "https://cdn.goengineer.com/formlabs-tough-2000-resin-technical-data-sheet.pdf",
        "ASTM D638-14, Form 2 at 100 um, post-cured 120 min at 80 C"),
    "formlabs-rigid10k": Source(
        "Formlabs Rigid 10K Resin technical data sheet, Rev 01 (06-09-2020)",
        "https://core-electronics.com.au/attachments/rigid10k-TDS-ENUS-0.pdf",
        "ASTM D638-14, UV post-cure column (60 min at 70 C)"),
    "addup-alsi10mg": Source(
        "AddUp AlSi10Mg material data sheet, version US-10252023-0129 (FormUp 350, 50 um layers)",
        "https://addupsolutions.com/wp-content/uploads/2023/10/AlSi10Mg-Material-Data-Sheet.pdf",
        "ASTM E8, machined specimens, as built, >= 27 samples across the plate"),
    "addup-316l": Source(
        "AddUp 316L SS material data sheet, version US-07172023-0134 (FormUp 350, 40 um layers)",
        "https://addupsolutions.com/wp-content/uploads/2023/09/AddUp-316L-SS-Materials-Data-Sheet.pdf",
        "ASTM E8, machined specimens, as built, >= 39 samples across the plate"),
    "eos-ti64": Source(
        "EOS Titanium Ti64 material data sheet (EOS M 290 400 W), 07.2022",
        "https://www.eos.info/05-datasheet-images/Assets_MDS_Metal/EOS_Titanium_Ti64/"
        "Material_DataSheet_EOS_Titanium_Ti64_EOSM290_EOSM300-4_EOSM400_EOSM400-4_en.pdf",
        "ISO 6892-1; heat treated 800 C / 4 h in argon (the usual delivered state)"),
    "eos-ti64-grade5": Source(
        "EOS Titanium Ti64 Grade 5 material data sheet, status 01/2024",
        "https://EOS.info/05-datasheet-images/Assets_MDS_Metal/EOS_Titanium_Ti64_Grade5/"
        "material_datasheet_eos_titanium_ti64_grade5_en_web.pdf",
        "fatigue strength at 1e7 cycles, heat treated, ISO 12107 statistics"),
    "tychanicz-2025": Source(
        "Tychanicz-Kwiecien et al. 2025, thermal conductivity of FDM-printed PLA and PETG "
        "(as used by heatx_thermal.WALL_CONDUCTIVITY)", "", "dense printed parts, in plane"),
    "handbook": Source(
        "typical wrought/cast handbook value for the alloy (ASM Handbook class data)", "",
        "used where the AM datasheet gives none; AM parts sit within a few percent"),
    "estimate": Source(
        "loadpath estimate: no datasheet value", "",
        "typical of the polymer family; replace with a calibrated card where it matters"),
    "dma-estimate": Source(
        "loadpath estimate: read off typical DMA storage-modulus curves for the polymer family and its heat "
        "deflection temperature", "",
        "approximate; the cliff at the glass transition is where sources disagree most. Measure it "
        "(DMA, or a heated cantilever) if a part runs warm"),
    "durability": Source("durability.DURABILITY (fatigue and creep design values, cited there)"),
}

# (value, source id, note)
def _c(title, processes, measured=None, **values):
    """A card. `measured`: the process the datasheet's specimens were made
    by (default the first of `processes`); its across-layer values apply to
    studies printed by that process only."""
    return {"title": title, "processes": tuple(processes), "measured": measured or processes[0],
            "values": values}


_RET = "(C, share of the 23 C modulus), piecewise linear; past the last point the last value holds"
_NU_POLY = (0.36, "estimate", "datasheets don't list it; 0.35-0.40 is typical of glassy thermoplastics")

LIBRARY = {
    # -------------------------------------------------------------- FDM
    "pla": _c("PLA (values from Prusament PLA)", ("fdm",),
              E=(2300, "prusament-pla", "tensile modulus, horizontal"),
              nu=_NU_POLY,
              strength=(51, "prusament-pla", "tensile yield strength, horizontal"),
              strengthZ=(17, "prusament-pla", "interlayer adhesion"),
              density=(1.24, "prusament-pla", "ISO 1183"),
              elongation=(2.9, "prusament-pla", "% at yield, horizontal"),
              hdt=(55, "prusament-pla", "ISO 75, 0.45 MPa"),
              hdt18=(55, "prusament-pla", "ISO 75, 1.80 MPa"),
              conductivity=(0.19, "tychanicz-2025", "W/(m K)"),
              fatigue=("pla", "durability", "Ezeh & Susmel design curve"),
              cte=(68, "estimate", "um/(m K) near 23 C; typical printed PLA, in plane"),
              retention=([(23, 1.0), (45, 0.92), (55, 0.6), (60, 0.2), (65, 0.04)], "dma-estimate", _RET)),
    "petg": _c("PETG (values from Prusament PETG)", ("fdm",),
               E=(1500, "prusament-petg", "tensile modulus, horizontal"),
               nu=(0.38, "estimate", "copolyester; datasheets don't list it"),
               strength=(47, "prusament-petg", "tensile yield strength, horizontal"),
               strengthZ=(18, "prusament-petg", "interlayer adhesion"),
               density=(1.27, "prusament-petg", "ISO 1183"),
               elongation=(5.1, "prusament-petg", "% at yield, horizontal"),
               hdt=(68, "prusament-petg", "ISO 75, 0.45 MPa"),
               hdt18=(68, "prusament-petg", "ISO 75, 1.80 MPa"),
               conductivity=(0.20, "tychanicz-2025", "W/(m K)"),
               fatigue=("petg", "durability", "PLA curve scaled by strength (assumed)"),
               cte=(68, "estimate", "um/(m K) near 23 C; typical copolyester"),
               retention=([(23, 1.0), (40, 0.93), (55, 0.8), (65, 0.6), (75, 0.25), (85, 0.05)], "dma-estimate", _RET)),
    "abs": _c("ABS (values from Polymaker PolyLite ABS)", ("fdm",),
              E=(2174, "polylite-abs", "Young's modulus X-Y"),
              EZ=(1835, "polylite-abs", "Young's modulus Z"),
              nu=(0.35, "estimate", "datasheets don't list it"),
              strength=(33.3, "polylite-abs", "tensile strength X-Y"),
              strengthZ=(25.4, "polylite-abs", "tensile strength Z"),
              density=(1.12, "polylite-abs", "ISO 1183"),
              elongation=(2.7, "polylite-abs", "% at break, X-Y"),
              tg=(101.1, "polylite-abs", "DSC"),
              hdt=(99.6, "polylite-abs", "ISO 75, 0.45 MPa"),
              hdt18=(98.2, "polylite-abs", "ISO 75, 1.8 MPa"),
              conductivity=(0.17, "estimate", "W/(m K), bulk ABS"),
              fatigue=("abs", "durability", "PLA curve scaled by strength (assumed)"),
              cte=(90, "estimate", "um/(m K) near 23 C; typical ABS"),
              retention=([(23, 1.0), (60, 0.85), (80, 0.65), (95, 0.25), (105, 0.04)], "dma-estimate", _RET)),
    "asa": _c("ASA (values from Polymaker PolyLite ASA)", ("fdm",),
              E=(2379, "polylite-asa", "Young's modulus X-Y"),
              EZ=(1965, "polylite-asa", "Young's modulus Z"),
              nu=(0.35, "estimate", "datasheets don't list it"),
              strength=(43.8, "polylite-asa", "tensile strength X-Y"),
              strengthZ=(32.0, "polylite-asa", "tensile strength Z"),
              density=(1.13, "polylite-asa", "ISO 1183"),
              elongation=(6.7, "polylite-asa", "% at break, X-Y"),
              tg=(97.8, "polylite-asa", "DSC"),
              hdt=(102.6, "polylite-asa", "ISO 75, 0.45 MPa"),
              hdt18=(100.2, "polylite-asa", "ISO 75, 1.8 MPa"),
              conductivity=(0.17, "estimate", "W/(m K), as ABS"),
              fatigue=("asa", "durability", "PLA curve scaled by strength (assumed)"),
              cte=(95, "estimate", "um/(m K) near 23 C; typical ASA"),
              retention=([(23, 1.0), (60, 0.85), (80, 0.65), (95, 0.25), (105, 0.04)], "dma-estimate", _RET)),
    # 95A: linear analysis holds only to a few percent strain; flexures
    # want verify_nonlinear (nonlinear.py)
    "tpu": _c("TPU 95A (values from UltiMaker TPU 95A); small strains only", ("fdm",),
              E=(67, "ultimaker-tpu95a", "tensile modulus XY"),
              EZ=(56, "ultimaker-tpu95a", "tensile modulus Z"),
              nu=(0.45, "estimate", "elastomers are nearly incompressible"),
              strength=(23.7, "ultimaker-tpu95a", "tensile stress at break XY"),
              strengthZ=(6.4, "ultimaker-tpu95a", "tensile stress at break Z"),
              density=(1.21, "estimate", "the TDS gives none; typical TPU 95A filament 1.20-1.22"),
              elongation=(560, "ultimaker-tpu95a", "% at break XY (> 560)"),
              hdt=(50.3, "ultimaker-tpu95a", "ISO 75-2/B, 0.455 MPa"),
              conductivity=(0.20, "estimate", "W/(m K)"),
              fatigue=("tpu", "durability", "rough estimate"),
              cte=(150, "estimate", "um/(m K) near 23 C; typical TPU 95A"),
              retention=([(23, 1.0), (40, 0.85), (60, 0.7), (80, 0.55), (100, 0.4)], "dma-estimate", _RET)),
    "pa-cf": _c("carbon-filled nylon (values from Bambu PAHT-CF, dry)", ("fdm",),
                E=(3860, "bambu-paht-cf", "Young's modulus X-Y, dry"),
                EZ=(2180, "bambu-paht-cf", "Young's modulus Z, dry"),
                nu=(0.35, "estimate", "datasheets don't list it"),
                strength=(92, "bambu-paht-cf", "tensile strength X-Y, dry"),
                strengthZ=(47, "bambu-paht-cf", "tensile strength Z, dry"),
                density=(1.06, "bambu-paht-cf", "ISO 1183"),
                tg=(70, "bambu-paht-cf", "DSC"),
                hdt=(194, "bambu-paht-cf", "ISO 75, 0.45 MPa"),
                hdt18=(170, "bambu-paht-cf", "ISO 75, 1.8 MPa"),
                conductivity=(0.40, "estimate", "W/(m K), fibre-filled, in plane"),
                fatigue=("pa-cf", "durability", "rough estimate"),
                cte=(30, "estimate", "um/(m K) along the laid fibres; across them several times more"),
                retention=([(23, 1.0), (50, 0.8), (80, 0.55), (120, 0.35), (150, 0.2)], "dma-estimate", _RET)),
    # ------------------------------------------------------- powder bed
    "pa12": _c("PA 12 (values from HP 3D HR PA 12, MJF)", ("sls", "mjf", "fdm"), "mjf",
               E=(1900, "hp-5200-whitepaper", "tensile modulus XY (95% of parts 1650-2200)"),
               EZ=(1900, "hp-5200-whitepaper", "tensile modulus Z"),
               nu=(0.39, "estimate", "nylon 12; datasheets don't list it"),
               strength=(50, "hp-5200-whitepaper", "tensile strength XY (95% of parts 45-55)"),
               strengthZ=(50, "hp-5200-whitepaper", "tensile strength Z"),
               density=(1.01, "hp-5200-whitepaper", "ASTM D792"),
               elongation=(17, "hp-5200-whitepaper", "% at break XY (Z: 9)"),
               conductivity=(0.25, "estimate", "W/(m K), sintered PA12"),
               fatigue=("pa12", "durability", "Van Hooreweder & Kruth 2014"),
               cte=(109, "estimate", "um/(m K) near 23 C; typical sintered PA12"),
               retention=([(23, 1.0), (45, 0.75), (80, 0.45), (120, 0.25), (160, 0.1)], "dma-estimate", _RET)),
    "pa11": _c("PA 11 (values from HP 3D HR PA 11, MJF): tougher than PA12", ("sls", "mjf"), "mjf",
               E=(1700, "hp-5200-whitepaper", "tensile modulus XY"),
               EZ=(1800, "hp-5200-whitepaper", "tensile modulus Z"),
               nu=(0.39, "estimate", "as PA12"),
               strength=(54, "hp-5200-whitepaper", "tensile strength XY"),
               strengthZ=(54, "hp-5200-whitepaper", "tensile strength Z"),
               density=(1.05, "hp-5200-whitepaper", "ASTM D792"),
               elongation=(40, "hp-5200-whitepaper", "% at break XY (Z: 25)"),
               conductivity=(0.25, "estimate", "W/(m K), as PA12"),
               fatigue=("pa11", "durability", "as PA12 (assumed)"),
               cte=(100, "estimate", "um/(m K) near 23 C; typical PA11"),
               retention=([(23, 1.0), (45, 0.75), (80, 0.45), (120, 0.25), (160, 0.1)], "dma-estimate", _RET)),
    "pa12-gb": _c("glass-bead PA 12 (values from HP 3D HR PA 12 GB): stiff, stable, brittle", ("sls", "mjf"), "mjf",
                  E=(2900, "hp-5200-whitepaper", "tensile modulus XY"),
                  EZ=(3000, "hp-5200-whitepaper", "tensile modulus Z"),
                  nu=(0.37, "estimate", "filled nylon"),
                  strength=(31, "hp-5200-whitepaper", "tensile strength XY"),
                  strengthZ=(30, "hp-5200-whitepaper", "tensile strength Z"),
                  density=(1.30, "hp-5200-whitepaper", "ASTM D792"),
                  elongation=(9, "hp-5200-whitepaper", "% at break XY"),
                  hdt=(170, "hp-5200-whitepaper", "ASTM D648, 0.45 MPa, XY"),
                  hdt18=(113, "hp-5200-whitepaper", "ASTM D648, 1.82 MPa, XY")),
    "tpu88a": _c("TPU 88A (values from BASF Ultrasint TPU 88A, SLS): a linear analysis is a rough estimate "
                 "past a few percent strain; run verify_nonlinear", ("sls", "mjf"),
                 E=(75, "basf-tpu88a", "tensile modulus X (ISO 527-2 1A)"),
                 EZ=(75, "basf-tpu88a", "tensile modulus Z"),
                 nu=(0.45, "estimate", "elastomers are nearly incompressible"),
                 strength=(8, "basf-tpu88a", "tensile strength X (DIN 53504 S2)"),
                 strengthZ=(7, "basf-tpu88a", "tensile strength Z"),
                 density=(1.10, "basf-tpu88a", "printed part, ISO 1183-1"),
                 elongation=(270, "basf-tpu88a", "% at break X (Z: 130)")),
    # ------------------------------------------------------------ resin
    "resin": _c("standard resin (values from Formlabs Grey V4, post-cured)", ("sla", "dlp"),
                E=(2800, "formlabs-standard", "tensile modulus, post-cured"),
                nu=(0.38, "estimate", "acrylate photopolymer"),
                strength=(65, "formlabs-standard", "ultimate tensile strength, post-cured"),
                density=(1.18, "estimate", "the TDS gives none; cured acrylates 1.15-1.20"),
                elongation=(6, "formlabs-standard", "% at break, post-cured"),
                hdt=(73, "formlabs-standard", "ASTM D648, 0.45 MPa"),
                hdt18=(58, "formlabs-standard", "ASTM D648, 1.8 MPa"),
                conductivity=(0.18, "estimate", "W/(m K)"),
                fatigue=("resin", "durability", "rough estimate"),
                cte=(79, "formlabs-standard", "um/(m K), 0-150 C, post-cured (ASTM E831)"),
                retention=([(23, 1.0), (40, 0.8), (50, 0.5), (60, 0.2), (70, 0.05)], "dma-estimate", _RET)),
    "resin-tough": _c("tough resin (values from Formlabs Tough 2000, post-cured)", ("sla", "dlp"),
                      E=(2200, "formlabs-tough2000", "tensile modulus, post-cured"),
                      nu=(0.40, "estimate", "toughened photopolymer"),
                      strength=(46, "formlabs-tough2000", "ultimate tensile strength, post-cured"),
                      density=(1.15, "estimate", "the TDS gives none"),
                      elongation=(48, "formlabs-tough2000", "% at break, post-cured"),
                      hdt=(63, "formlabs-tough2000", "ASTM D648, 0.45 MPa"),
                      hdt18=(53, "formlabs-tough2000", "ASTM D648, 1.8 MPa"),
                      cte=(91, "formlabs-tough2000", "um/(m K), 0-150 C, post-cured (ASTM E831)"),
                      conductivity=(0.18, "estimate", "W/(m K)"),
                      fatigue=("resin-tough", "durability", "rough estimate"),
                      retention=([(23, 1.0), (40, 0.75), (50, 0.45), (60, 0.2), (70, 0.05)], "dma-estimate", _RET)),
    "resin-rigid": _c("glass-filled rigid resin (values from Formlabs Rigid 10K, UV cure): brittle", ("sla", "dlp"),
                      E=(10000, "formlabs-rigid10k", "tensile modulus, UV post-cure"),
                      nu=(0.33, "estimate", "highly glass-filled photopolymer"),
                      strength=(65, "formlabs-rigid10k", "ultimate tensile strength, UV post-cure"),
                      density=(1.65, "estimate", "the TDS gives none; glass-filled acrylates 1.6-1.7"),
                      elongation=(1, "formlabs-rigid10k", "% at break, UV post-cure"),
                      cte=(47, "formlabs-rigid10k", "um/(m K), 0-150 C, UV post-cure (ASTM E831)")),
    # ------------------------------------------------------------- LPBF
    "alsi10mg": _c("LPBF AlSi10Mg, as built (values from AddUp)", ("lpbf",),
                   E=(72000, "addup-alsi10mg", "modulus XY, as built (Z 71 GPa)"),
                   EZ=(71000, "addup-alsi10mg", "modulus Z, as built"),
                   nu=(0.33, "handbook", "aluminium alloys"),
                   strength=(275, "addup-alsi10mg", "yield strength XY, as built"),
                   strengthZ=(239, "addup-alsi10mg", "yield strength Z, as built"),
                   density=(2.67, "addup-alsi10mg", "theoretical; parts 99.5% dense"),
                   elongation=(5, "addup-alsi10mg", "% at failure, XY and Z"),
                   conductivity=(125, "addup-alsi10mg", "W/(m K), low end of the sheet's 120-160 at 25 C"),
                   fatigue=("alsi10mg", "durability", "rough, as-built surface"),
                   cte=(21, "addup-alsi10mg", "um/(m K), 20-100 C"),
                   retention=([(23, 1.0), (150, 0.92), (200, 0.85), (300, 0.6)], "dma-estimate", _RET)),
    "ti64": _c("LPBF Ti-6Al-4V, heat treated (values from EOS)", ("lpbf",),
               E=(110000, "eos-ti64", "modulus XY and Z, +-15 GPa"),
               EZ=(110000, "eos-ti64", "modulus Z"),
               nu=(0.34, "handbook", "Ti-6Al-4V"),
               strength=(1000, "eos-ti64", "yield strength Rp0.2 XY, heat treated (min 860)"),
               strengthZ=(1000, "eos-ti64", "yield strength Rp0.2 Z, heat treated"),
               density=(4.41, "eos-ti64", "part density"),
               elongation=(13.5, "eos-ti64", "% at break, heat treated"),
               conductivity=(6.7, "handbook", "W/(m K), Ti-6Al-4V"),
               fatigueStrength=(595, "eos-ti64-grade5", "MPa at 1e7 cycles, heat treated (Grade 5 sheet)"),
               fatigue=("ti64", "durability", "rough, as-built surface"),
               cte=(8.6, "handbook", "um/(m K), 20-100 C, Ti-6Al-4V"),
               retention=([(23, 1.0), (300, 0.85), (500, 0.75)], "dma-estimate", _RET)),
    "316l": _c("LPBF 316L stainless, as built (values from AddUp)", ("lpbf",),
               E=(193000, "addup-316l", "modulus XY, as built"),
               EZ=(160000, "addup-316l", "modulus Z, as built"),
               nu=(0.30, "handbook", "austenitic stainless"),
               strength=(699, "addup-316l", "yield strength XY, as built"),
               strengthZ=(632, "addup-316l", "yield strength Z, as built"),
               density=(8.0, "addup-316l", "theoretical; parts 99.9% dense"),
               conductivity=(15, "addup-316l", "W/(m K) at 20 C"),
               fatigue=("316l", "durability", "rough, as-built surface"),
               cte=(17.2, "addup-316l", "um/(m K), RT-200 C (the sheet's literature value)"),
               retention=([(23, 1.0), (300, 0.9), (500, 0.82)], "dma-estimate", _RET)),
}


def card_names():
    return sorted(LIBRARY)


def card(name):
    """The library card (dict) for a material name, or None."""
    return LIBRARY.get(str(name).strip().lower())


def value(name, key, default=None):
    """A card's number for `key` (e.g. "strengthZ"), or default."""
    c = card(name)
    if c is None or key not in c["values"]:
        return default
    return c["values"][key][0]


def describe(name):
    """A card as JSON: every value with its source's title and url."""
    c = card(name)
    if c is None:
        return None
    vals = {}
    for k, (v, src, note) in c["values"].items():
        s = SOURCES[src]
        vals[k] = {"value": v, "note": note, "source": s.title, **({"url": s.url} if s.url else {})}
    return {"name": str(name).strip().lower(), "title": c["title"], "processes": list(c["processes"]),
            "measured": c["measured"],
            "values": vals, "accessed": ACCESSED}
