"""
As-printed lattice stiffness: how much softer a printed lattice is than its
designed geometry, by process, cell family and density, fitted to published
compression tests.

Checked against 33 printed lattices (validation.py,
data/published_experiments.json), loadpath's homogenized check reads the
designed geometry 1.0-4.1x stiffer than the prints measured (median 1.68x
either way), and its resolved voxel solve agrees with the homogenized one, so
the gap is the print, not the model: walls and struts thinner and rougher
than drawn, porous, stair-stepped, weakly bonded, and (in the tests) modulus
read from crosshead travel. The factor here is

    k = 1 / exp(a_process + b_strut [strut cell] + c ln(rho / 0.2))

with rho the lattice's relative density (clipped to the tested 0.1-0.4 for
the density term) and k capped at 1: the check never reads a lattice stiffer
than designed. The density term says the prints stiffen more slowly with
density than the design does (the papers' denser lattices are the
over-predicted ones); the strut term that strut cells lose more than sheets.

Fitted coefficients (bench/lattice_print_fit.py, least squares on log
modulus ratio over all 33 specimens, results in
bench/results/lattice-print-fit.json):

    process group  exp(a)   from
    fdm            1.54     Shevchenko 2023 (PLA sheets), Guerra Silva 2021 (ABS, PC struts)
    sls            1.08     Shevchenko 2023 (PA12 sheets), Abou-Ali 2022 (PA11 sheets and struts)
    resin          1.33     Shevchenko 2023 (LCD resin sheets)
    strut cells    x1.58
    density        x1.34 per e-fold of rho / 0.2

Held out (each specimen predicted from a fit to the other 32), the median
error in modulus falls from 1.68x to 1.23x (worst 4.12x to 2.04x); leaving
out a whole paper, its specimens read a median 0.92-1.17x of the measurement. Per-cell-type terms (Schwarz P,
Kelvin, octet) did not improve the held-out error and are not used.

Process groups: mjf is read as sls (no MJF lattice tests found), sla and dlp
as the LCD resin. volumetric, jetting and lpbf have no printed-lattice data
here: their factor is None and the check reads the designed geometry, with
a note.

Strength: a sheet print fails at about the strain its design predicts for
first yield, not the stress (the PLA Schwarz P lattice that broke at 0.84x
its predicted first-yield stress is 2.4x softer than the same geometry in
SLS PA12 and failed at the same strain), so for sheets the homogenized
check reads wall stress from the strain the softer lattice takes. Strut
lattices read from the macro stress they carry: their first yield was safe
that way on every strut specimen, and the strain reading made FDM strut
lattices 2.5x more conservative than they need (STRAIN_SHARE). Held out,
first yield is under the measured strength on all 33 specimens.

A calibrated card (calibration.py, gyroid coupon L-GY) that measured its own
lattice stiffness factor replaces this fit; so does a study's
process.latticeStiffness (a number, or "designed").
"""
from __future__ import annotations

import numpy as np

__all__ = ["STRAIN_SHARE", "strain_share", "PRINT_FIT", "PROCESS_GROUP", "print_factor", "process_group", "features", "fit", "fit_rows"]

# bench/lattice_print_fit.py writes these (tests/test_lattice_print.py checks
# they match a refit of bench/results/published-experiments.json)
PRINT_FIT = {
    "intercept": {"fdm": 0.43111, "sls": 0.07451, "resin": 0.2811},
    "strut": 0.45728,
    "densitySlope": 0.29423,
    "rhoRef": 0.2,
    "rhoRange": [0.1, 0.4],
}

# How the check reads wall stress in a softer printed lattice: 1 = from the
# strain it takes (stress k^-1 higher than the carried macro stress says),
# 0 = from the stress it carries. Chosen per family as the least value that
# keeps every published specimen's first-yield reading under its measured
# strength, then held out a paper at a time (bench/lattice_print_fit.py):
# - sheets 1: the FDM PLA Schwarz P lattice needs it (1.19x at 0, 0.82x at
#   1). With Shevchenko 2023 held out the rest of the data asks for 0, and
#   that paper's PLA P then reads 1.19x, so the sheet value rests on one
#   paper and the physical reading (the PLA print failed at the strain its
#   design yields at, 2.4x softer than the same geometry in SLS);
# - struts 0: safe on all 20 strut specimens (0.18-0.91x), and with either
#   strut paper held out the other still asks for 0 and reads safe on it
#   (Abou-Ali 0.65-0.91x, Guerra Silva 0.18-0.78x). At 1 the FDM strut
#   lattices read 0.07-0.41x of their plateau.
STRAIN_SHARE = {"sheet": 1.0, "strut": 0.0}

PROCESS_GROUP = {"fdm": "fdm", "sls": "sls", "mjf": "sls", "sla": "resin", "dlp": "resin"}

SOURCES = {
    "fdm": ["shevchenko2023", "guerrasilva2021"],
    "sls": ["shevchenko2023", "abouali2022"],
    "resin": ["shevchenko2023"],
}

FEATURES = ("fdm", "sls", "resin", "strut", "lrho")


def strain_share(lattice_kind: str) -> float:
    from .struts import is_strut

    return STRAIN_SHARE["strut" if is_strut(lattice_kind) else "sheet"]


def process_group(kind: str | None) -> str | None:
    return PROCESS_GROUP.get(str(kind).lower()) if kind else None


def features(group: str, strut: bool, rho, rho_ref: float = 0.2, rho_range=(0.1, 0.4)) -> np.ndarray:
    """Design rows (n, 5) for the fit: one-hot process group, strut flag,
    log density over the reference (clipped to the tested range)."""
    rho = np.atleast_1d(np.asarray(rho, float))
    x = np.zeros((len(rho), len(FEATURES)))
    x[:, FEATURES.index(group)] = 1.0
    x[:, 3] = float(bool(strut))
    x[:, 4] = np.log(np.clip(rho, *rho_range) / rho_ref)
    return x


def print_factor(process_kind: str | None, lattice_kind: str, rho, fit_: dict | None = None):
    """As-printed stiffness over designed (<= 1) at relative densities rho,
    or None where the process has no printed-lattice data."""
    from .struts import is_strut

    group = process_group(process_kind)
    if group is None:
        return None
    f = fit_ or PRINT_FIT
    b = np.array([f["intercept"]["fdm"], f["intercept"]["sls"], f["intercept"]["resin"], f["strut"],
                  f["densitySlope"]])
    x = features(group, is_strut(lattice_kind), rho, f["rhoRef"], tuple(f["rhoRange"]))
    return np.minimum(np.exp(-(x @ b)), 1.0)


def fit_rows(results: dict) -> list[dict]:
    """Rows for the fit from bench/results/published-experiments.json: group,
    strut flag, rho, the homogenized and measured modulus and strength."""
    from .struts import is_strut
    from .validation import process_kind

    rows = []
    for sid, r in results.items():
        rows.append({"id": sid, "source": r["source"], "group": process_group(process_kind(r["process"])),
                     "strut": bool(is_strut(r["lattice"]["type"])), "type": r["lattice"]["type"],
                     "rho": float(r["table"]["rho"]), "E": float(r["homogenized"]["E"]),
                     "Emeas": float(r["measured"]["E"]), "firstYield": float(r["homogenized"]["firstYield"]),
                     "strength": float(r["measured"]["strength"])})
    return rows


def fit(rows: list[dict], ridge: float = 1e-3) -> dict:
    """Least squares of log(designed / measured modulus) on the features
    (a light ridge keeps a group missing from `rows` at zero)."""
    x = np.vstack([features(r["group"], r["strut"], r["rho"]) for r in rows])
    y = np.log([r["E"] / r["Emeas"] for r in rows])
    b = np.linalg.solve(x.T @ x + ridge * np.eye(x.shape[1]), x.T @ y)
    return {"intercept": {"fdm": float(b[0]), "sls": float(b[1]), "resin": float(b[2])}, "strut": float(b[3]),
            "densitySlope": float(b[4]), "rhoRef": 0.2, "rhoRange": [0.1, 0.4]}
