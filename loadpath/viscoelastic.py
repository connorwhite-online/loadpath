"""
Time and rate dependence of elastomer stiffness: creep under a steady load
and stiffening under fast loads, for TPU feet, pads and crush cores.

A TPU part is a viscoelastic solid. Its modulus depends on how long the
load has been on: a material card's E comes from a test lasting seconds; a
foot under a payload for a year is softer, a mount vibrating at 35 Hz or a
pad hit in 15 ms is stiffer. Over the times that matter here the creep
compliance of printed TPU grows with the logarithm of time, the form
isolator makers give for their polyurethanes (Getzner: "the increase in
deformation is proportional to the time logarithm"):

    J(t) / J(t_ref) = 1 + m ln(t / t_ref)

and the effective modulus at a time scale t is E_ref / (1 + m ln(t / t_ref)),
below 1 for long holds and above 1 for fast events. A vibration at
frequency f samples the time scale 1 / (2 pi f); an impact that crushes
for duration T samples T.

Sourced value for printed TPU, m = 0.045 per e-fold (0.10 per decade),
t_ref = 10 s (the duration of a tensile test, where the card's E is from):

- Fit to a generalized Maxwell model of 100-minute compression creep on
  extrusion-printed TPU cylinders (CMC 86(3), 2025, "creep" parameter set:
  E0 25.9, E1 7.3 (tau 177 s), E2 18.8 MPa (tau 2.9 s)): E(10 s) / E(6000 s)
  = 1.29 over ln(600) = 6.4, so m = 0.045.
- Check: the 35 Hz to 10 s stiffness ratio this gives is 1.53, inside the
  1.3-1.6 dynamic-to-static ratio isolator makers quote for polyurethane
  (GERB NOVODAMP data sheet).
- Ten years under load: J grows 1.78x over the 10 s value. Getzner quote
  less than 50% extra deflection after 10 years within the recommended load
  range for their PU foams, so 1.78 is on the cautious side for a lightly
  loaded foot.

Not covered: temperature (TPU softens quickly above ~40 C), large strain
(the law is linear viscoelastic), and the frequency dependence of the loss
factor (modal.LOSS_FACTOR stays a single number; the log law itself implies
tan delta ~ pi m / 2 = 0.07, about half the 0.12 we use, which is a 10 Hz DMA
value: printed TPU has damping the creep data alone do not show).

A calibration card can override m (card["creep"]["m"]), measured with the
calibration kit's creep coupon (see calibration.creep_coupons).
"""
from __future__ import annotations

import numpy as np

__all__ = ["CREEP", "creep_params", "modulus_factor", "creep_factor", "dynamic_factor", "event_factor",
           "YEAR", "DAY"]

DAY = 86_400.0
YEAR = 365.25 * DAY

# m: compliance growth per e-fold of time; tRef: the time scale of the
# material card's E (s)
CREEP = {
    "tpu": {"m": 0.045, "tRef": 10.0,
            "source": "printed TPU compression creep, CMC 86(3) 2025 (GM fit); 35 Hz/static 1.53 vs PU 1.3-1.6"},
}


def creep_params(material):
    """(m, t_ref) for a material, or None if it is not modelled as
    viscoelastic. A saved calibration card's creep block wins."""
    name = getattr(material, "name", str(material)).lower()
    try:
        from .calibration import load_card

        card = load_card(name) or {}
    except Exception:  # noqa: BLE001 - a missing or broken card is no card
        card = {}
    base = card.get("basedOn", name)
    p = CREEP.get(name) or CREEP.get(base)
    if card.get("creep"):
        p = {**(p or {"tRef": 10.0}), **card["creep"]}
    if not p:
        return None
    return float(p["m"]), float(p.get("tRef", 10.0))


def modulus_factor(material, seconds) -> np.ndarray:
    """E(t) / E_card at time scale `seconds` (1 for materials not modelled)."""
    p = creep_params(material)
    t = np.asarray(seconds, float)
    if p is None:
        return np.ones_like(t)
    m, tr = p
    # floor the log so very fast events cannot blow the factor up
    return 1.0 / np.maximum(1.0 + m * np.log(np.maximum(t, 1e-6) / tr), 0.25)


def creep_factor(material, seconds) -> np.ndarray:
    """Deflection under a steady load after `seconds`, over the deflection
    the card's E predicts."""
    return 1.0 / modulus_factor(material, seconds)


def dynamic_factor(material, frequency) -> np.ndarray:
    """Dynamic stiffness at `frequency` (Hz) over the card's stiffness."""
    return modulus_factor(material, 1.0 / (2 * np.pi * np.asarray(frequency, float)))


def event_factor(material, duration) -> np.ndarray:
    """Stiffening for a load applied over `duration` seconds (an impact's
    crush time)."""
    return modulus_factor(material, duration)
