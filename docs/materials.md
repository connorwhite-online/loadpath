# Materials

A study names its material (`"material": "petg"`). Each name is a card in
[`loadpath/material_library.py`](../loadpath/material_library.py). Every number
on a card cites the datasheet it came from, with that sheet's version and a link.
`loadpath materials` lists the cards, and `loadpath materials petg` prints one card
with its sources. Every analysis reads the card: stress against strength,
deflection in mm, mass in grams, conduction, the across-layer bond and the
temperature warnings.

## The cards

In-plane values first, then across layers (Z) where the sheet measured it.
Strength means yield for metals and tensile strength for polymers.

| name | source | E (MPa) | E_z | strength (MPa) | strength_z | density | HDT 0.45 MPa |
|---|---|---|---|---|---|---|---|
| `pla` | Prusament PLA v1.1 (2022) | 2300 | - | 51 | 17 (interlayer adhesion) | 1.24 | 55 C |
| `petg` | Prusament PETG v1.1 (2022) | 1500 | - | 47 | 18 (interlayer adhesion) | 1.27 | 68 C |
| `abs` | Polymaker PolyLite ABS v5.2 | 2174 | 1835 | 33.3 | 25.4 | 1.12 | 99.6 C |
| `asa` | Polymaker PolyLite ASA v5.1 | 2379 | 1965 | 43.8 | 32.0 | 1.13 | 102.6 C |
| `tpu` (95A) | UltiMaker TPU 95A v2.01 (2025) | 67 | 56 | 23.7 | 6.4 | 1.21* | 50.3 C |
| `pa-cf` | Bambu PAHT-CF v2, dry | 3860 | 2180 | 92 | 47 | 1.06 | 194 C |
| `pa12` | HP 3D HR PA 12 (MJF 5200 white paper) | 1900 | 1900 | 50 | 50 | 1.01 | - |
| `pa11` | HP 3D HR PA 11 (same) | 1700 | 1800 | 54 | 54 | 1.05 | - |
| `pa12-gb` | HP 3D HR PA 12 GB (same) | 2900 | 3000 | 31 | 30 | 1.30 | 170 C |
| `tpu88a` | BASF Ultrasint TPU 88A v2.2 (SLS) | 75 | 75 | 8 | 7 | 1.10 | - |
| `resin` | Formlabs Grey V4, post-cured | 2800 | - | 65 | - | 1.18* | 73 C |
| `resin-tough` | Formlabs Tough 2000, post-cured | 2200 | - | 46 | - | 1.15* | 63 C |
| `resin-rigid` | Formlabs Rigid 10K, UV cure | 10000 | - | 65 | - | 1.65* | - |
| `alsi10mg` | AddUp AlSi10Mg (FormUp 350), as built | 72000 | 71000 | 275 | 239 | 2.67 | - |
| `ti64` | EOS Ti64 (M 290), heat treated | 110000 | 110000 | 1000 | 1000 | 4.41 | - |
| `316l` | AddUp 316L (FormUp 350), as built | 193000 | 160000 | 699 | 632 | 8.0 | - |

\* The datasheet gives no density for these, so the value is a marked estimate.
Polymer Poisson's ratios are estimates for the same reason (0.35-0.45). Card
conductivities come from Tychanicz-Kwiecien et al. 2025 for PLA and PETG and
from the AddUp sheets for AlSi10Mg and 316L. Ti64 uses a handbook value, and
the rest are estimates. Fatigue and creep design values stay in
`durability.DURABILITY`, which cites its own literature (Ezeh & Susmel for
FDM PLA, Van Hooreweder & Kruth for sintered PA12). EOS's Ti64 Grade 5 sheet
gives 595 MPa at 1e7 cycles heat treated; the card carries that value for
reference.

## Across layers

When a card measured its across-layer values on the study's process, those
values set the process's factors:

- `interlayer` becomes strength_z / strength;
- `stiffnessAcrossLayers` becomes E_z / E.

So a PETG study with `"process": "fdm"` checks its layer bonds at 18/47 = 0.38
of the in-plane strength, not the generic FDM 0.6. A card applies only to
the process its specimens were made by. HP's PA12 card is MJF data, so an FDM
PA12 study keeps the FDM defaults.

The study's own `process.interlayer` or `stiffnessAcrossLayers` always wins.

## Brands differ more than you'd think

Each generic name takes all its numbers from one datasheet, so they belong
together (one printer, one profile, one test). Other datasheets for the same
polymer can differ a lot:

- **PLA:** Prusament gives 2.3 GPa and 51 MPa. Polymaker PolyLite PLA v5.1
  gives 2.6 GPa and 46.6 MPa, with 43.5 MPa across layers (93%). Prusament's
  interlayer adhesion test gives 33%.
- **PETG:** Prusament gives 1.5 GPa and 47 MPa. PolyLite PETG v5.2 gives
  1.5 GPa, 31.9 MPa in plane and 13.4 MPa across layers.
- **TPU 95A:** UltiMaker gives a 67 MPa modulus, BASF Ultrafuse 48 MPa and
  Bambu 9.2 MPa. These are different tests, and all of them are only valid
  at small strain.
- **Nylon CF:** Bambu PAHT-CF gives 3.9 GPa dry. Polymaker PA6-CF v5 gives
  7.5 GPa dry and 5.7 GPa conditioned.

The cards favour conservative across-layer values. Even so, a part that
matters should be calibrated. Print the coupons, fit a card
([CALIBRATION.md](CALIBRATION.md)), and save it under the library name (for
example `petg`). It then replaces the library card for every study on that
machine, including its process factors.

## Changing one value

To change a single number, give a custom material with a base:

```json
"material": {"base": "petg", "strength": 40, "note": "our supplier's sheet"}
```

It keeps the rest of the PETG card, including the across-layer values. A
fully custom material still works: `{"E", "nu", "strength", "density"}`, with
optional `EZ`, `strengthZ` and `measured` (the process those were measured on).

## Temperatures

`durability.temperature` (or a load case's `temperature`) at or above the
card's HDT at 0.45 MPa adds a warning to the durability report. The part
will sag there, whatever the stress check says. Each card also carries how much
modulus it keeps with temperature (`retention`, feeding
`thermal.MODULUS_RETENTION`). It is read off typical DMA curves for the polymer
family and checked against the card's HDT, so treat it as an estimate.

Thermal expansion (`cte`, feeding `thermal.EXPANSION`) comes from the Formlabs
sheets for the resins (Grey 79, Tough 2000 91, Rigid 10K 47 um/(m K), 0-150 C)
and from AddUp for AlSi10Mg (21) and 316L (17.2). The filament datasheets
don't give it, so the FDM, nylon and Ti64 values are marked estimates and
handbook values.

## What changed from the earlier cards (2026-10-31)

The earlier cards held round numbers with no sources. Moving to the
datasheet values changed the following:

- **PLA:** modulus 3500 to 2300 MPa. 3.5 GPa is a moulded-PLA number, and
  printed parts deflect about 50% more than the old card said.
- **PETG:** modulus 2000 to 1500 MPa.
- **ABS:** strength 35 to 33.3 MPa.
- **ASA:** strength 40 to 43.8 MPa.
- **TPU:** modulus 26 to 67 MPa.
- **PA-CF:** 4500 MPa and 70 MPa became 3860 MPa and 92 MPa.
- **PA12:** 1800 MPa and 48 MPa became 1900 MPa and 50 MPa.
- **Resins:** the standard and tough resins got stiffer and stronger.
- **AlSi10Mg:** yield 230 to 275 MPa.
- **Ti64:** yield 900 to 1000 MPa.
- **316L:** yield 450 to 699 MPa, as built.

The coarse eval tier still passes every case on these values. Its baseline
(`loadpath/evals/baselines/coarse.json`) was re-saved: the biggest changes were
mbb-beam at -7% massAtTarget and generated link 3 at +6%. The full tier has
been re-run too (`--jobs 1`; the GRIP wrist link needs `LOADPATH_WORKERS=1` to fit
in 15 GB), and all 14 cases pass. Its baseline is `loadpath/evals/baselines/full.json`.
On the library PA12 card, arm-link went from 59.6 g at SF 14.1 to 55.4 g at SF 14.6,
and grip-wrist-link-mjf stayed at 43.7 g, with SF 4.39 to 4.59.
