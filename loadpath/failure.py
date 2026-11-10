"""
What breaks a printed part: the stress measures the checks read against
strength.

Von Mises against the tensile strength is the yield criterion of a ductile
metal: it has no idea whether a point is being pulled apart or squeezed,
and no idea the part was laid down in layers. A printed polymer differs on
both counts, and both change which point of a part fails first.

Pressure. A glassy or semi-crystalline polymer yields at a higher stress
in compression than in tension, because hydrostatic pressure closes the
free volume its chains shear through and hydrostatic tension opens it (and
opens the crazes and voids a print is full of). The standard description is
the pressure-modified von Mises criterion (Raghava, Caddell & Yeh, "The
macroscopic yield behaviour of polymers", J. Mater. Sci. 8, 1973): with
T and C the tensile and compressive strengths and m = C / T,

    sigma_eq = [(m - 1) I1 + sqrt((m - 1)^2 I1^2 + 4 m sigma_vm^2)] / (2 m)

against T, where I1 is the trace of the stress. It is sigma in uniaxial
tension, sigma / m in uniaxial compression, sigma_vm when m = 1, and it is
not zero under pure hydrostatic tension, which von Mises reads as no stress
at all: the root of a notch or a boss pulled three ways at once fails
earlier than its von Mises stress says, and the seat under a bolt head or a
strut in compression later. m is 1.2-1.35 across the amorphous polymers of
that paper; printed parts are taken at the low end (`COMPRESSION`), metals
and elastomers at 1.

Layers. A layered part has a second, weaker way to fail: along the bond
between two layers. A bond is pulled apart by the tension normal to it and
slid apart by the shear in its plane, and it does not matter to the bond
which of the two loads it; reading only the normal tension passes a boss
twisted about the build axis, or a short tab sheared across its layers, at
the full in-plane strength. The interaction is the usual quadratic one for
a weak plane (Hashin's matrix mode, the delamination criterion of
laminates):

    (sigma_n+ / Z)^2 + (tau / S)^2 = 1

with Z the bond's tensile strength (process.interlayer x strength), tau the
shear traction on the layer plane and S the bond's shear strength
(process.interlayerShear x strength; by default the bond is as much weaker
in shear as in tension, S = Z / sqrt(3), the shear strength of a von Mises
solid of strength Z). Compression across the layers closes the bond and
does not count against it; the friction it adds is ignored (conservative).
`bond_stress` returns the equivalent sqrt(sigma_n+^2 + (Z / S)^2 tau^2),
read against Z.

Both ratios are material data the datasheets rarely give; the defaults are
flagged "assumed" in the report (`failure`), a study or a calibrated card
overrides them, and docs/PHYSICS.md says what coupon measures each.
"""
from __future__ import annotations

import numpy as np

__all__ = ["COMPRESSION", "compression_ratio", "equivalent_stress", "bond_stress", "bond_shear_ratio",
           "criterion"]

_RAGHAVA = "Raghava, Caddell & Yeh, J. Mater. Sci. 8 (1973): compressive / tensile yield 1.2-1.35 for glassy polymers"

# compressive / tensile strength by material family: (ratio, source, quality)
COMPRESSION = {
    "glassy": (1.2, _RAGHAVA + "; printed parts taken at the low end", "assumed"),
    "semicrystalline": (1.15, "nylons and polypropylene yield 1.1-1.3x higher in compression (pressure-dependent "
                              "yield); low end for printed parts", "assumed"),
    "elastomer": (1.0, "elastomers are strain-governed: no asymmetry taken", "assumed"),
    "metal": (1.0, "metals yield alike in tension and compression", "measured"),
}

_FAMILY = {
    "pla": "glassy", "petg": "glassy", "abs": "glassy", "asa": "glassy", "resin": "glassy", "vero": "glassy",
    "pa12": "semicrystalline", "pa11": "semicrystalline", "pa": "semicrystalline", "pp": "semicrystalline",
    "tpu": "elastomer", "tpu88a": "elastomer", "agilus30": "elastomer",
}


def _family(material) -> str:
    try:
        from .thermal import _material_key

        key = _material_key(material)
    except Exception:
        key = str(getattr(material, "name", material)).lower()
    if key in _FAMILY:
        return _FAMILY[key]
    root = key.split("-")[0]
    if root in _FAMILY:
        return _FAMILY[root]
    # a custom material: polymers are under ~20 GPa and ~2.5 g/cm^3
    e, rho = getattr(material, "E", None), getattr(material, "density", None)
    if e is not None and rho is not None and (e > 20_000 or rho > 2.5):
        return "metal"
    if e is not None and e < 200:
        return "elastomer"
    return "glassy" if e is not None else "metal"


def compression_ratio(material) -> tuple:
    """(m = compressive / tensile strength, source, quality) for a
    materials.Material: its own `compression` when the study or a card gives
    one, else its family's."""
    own = getattr(material, "compression", None)
    if own is not None:
        return float(own), "given with the material", "study"
    return COMPRESSION[_family(material)]


def equivalent_stress(sig: np.ndarray, m: float = 1.0) -> np.ndarray:
    """The stress to read against the tensile strength, from stress tensors
    (n, 6) as (sx, sy, sz, txy, tyz, tzx): the pressure-modified von Mises
    stress for a compressive / tensile strength ratio m (von Mises at 1)."""
    sig = np.asarray(sig, float)
    sx, sy, sz, txy, tyz, tzx = sig.T
    vm2 = 0.5 * ((sx - sy) ** 2 + (sy - sz) ** 2 + (sz - sx) ** 2) + 3.0 * (txy ** 2 + tyz ** 2 + tzx ** 2)
    if m == 1.0:
        return np.sqrt(vm2)
    a = (m - 1.0) * (sx + sy + sz)
    return (a + np.sqrt(a * a + 4.0 * m * vm2)) / (2.0 * m)


def bond_shear_ratio(process) -> float:
    """Z / S: the layer bond's tensile over its shear strength."""
    inter = float(getattr(process, "interlayer", 1.0))
    shear = getattr(process, "interlayer_shear", None)
    if shear is None:
        return float(np.sqrt(3.0))
    return inter / float(shear)


def bond_stress(sig: np.ndarray, n, shear_ratio: float = float(np.sqrt(3.0))):
    """The layer bond's equivalent stress for layers stacked along unit `n`:
    (equivalent, tension normal to the layers (>= 0), shear on the layer
    plane), each per row of `sig`. The equivalent reads against the bond's
    tensile strength; `shear_ratio` is that strength over the bond's shear
    strength."""
    sig = np.asarray(sig, float)
    n = np.asarray(n, float)
    n = n / np.linalg.norm(n)
    sx, sy, sz, txy, tyz, tzx = sig.T
    # traction on the layer plane, t = sigma . n
    tx = sx * n[0] + txy * n[1] + tzx * n[2]
    ty = txy * n[0] + sy * n[1] + tyz * n[2]
    tz = tzx * n[0] + tyz * n[1] + sz * n[2]
    sn = tx * n[0] + ty * n[1] + tz * n[2]
    tau = np.sqrt(np.maximum(tx * tx + ty * ty + tz * tz - sn * sn, 0.0))
    tens = np.maximum(sn, 0.0)
    return np.sqrt(tens * tens + (shear_ratio * tau) ** 2), tens, tau


def criterion(material, process=None) -> dict:
    """What a report says about the criteria it used (report["failure"])."""
    m, src, quality = compression_ratio(material)
    out = {"criterion": "pressure-modified von Mises" if m != 1.0 else "von Mises",
           "compressionRatio": m, "compressionSource": src, "compressionQuality": quality}
    if process is not None and getattr(process, "build", None) not in (None, "auto") \
            and getattr(process, "interlayer", 1.0) < 1.0:
        given = getattr(process, "interlayer_shear", None) is not None
        out["bond"] = {"criterion": "tension and shear on the layer plane, quadratic",
                       "tensile": process.interlayer,
                       "shear": process.interlayer / bond_shear_ratio(process),
                       "shearQuality": "study" if given else "assumed",
                       "shearSource": "process.interlayerShear" if given else
                       "the bond taken as much weaker in shear as in tension: interlayer / sqrt(3) of the "
                       "in-plane tensile strength"}
    return out
