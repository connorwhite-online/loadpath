"""
What a process builds unsupported inside a closed part, and the least shape
change that makes a round channel or vault build as drawn.

Inside a heat exchanger's plenum or a port's bore nothing can be supported:
a support grown in a closed channel never comes out. Whether a round
channel's top needs any change depends on the process, not on "printing" in
general:

- powder-bed polymer (SLS, MJF) and binder jetting: every layer lies on the
  unfused powder under it. Nothing overhangs; round stays round at any size.
- powder-bed metal (laser powder-bed fusion, "SLM / DMLS" on Materialize):
  the powder holds a layer up but does not carry its heat away, so a surface
  flatter than ~45 degrees from level drosses and curls. A level round bore
  still builds as drawn up to about 8 mm across (the design guidance the
  service bureaus and machine makers publish for AlSi10Mg, Ti64 and 316L):
  its top is an arch whose flat crown is short enough to close.
- FDM: a level round bore's crown is a short bridge between the two sides
  of the arch; a slicer bridges printcheck.BRIDGE_SPAN (15 mm) cleanly.
- resin (SLA, DLP): the slicer cannot put supports inside a closed channel;
  short spans hold between the arch's sides (printcheck.BRIDGE_SPAN 5 mm).

The arch model behind all of them: a round top of radius r is flatter than
the critical angle a over the chord 2 r sin a around its crown. A process
that builds a round bore of diameter D_max closes a crown chord of
b = D_max sin a, so a circle of radius rho = b / (2 sin a) = D_max / 2 builds
as drawn. A bigger bore (or vault) keeps its round bottom and sides up to
where its surface turns a from level, runs on along those tangents, and
closes over the top with that small circle, tangent to both flanks: the
convex hull of the bore's circle and a crown circle of radius rho on its
axis. Its height over the axis is the least of any top that builds:

    h = (r - rho) / cos(a) + rho              (h = r at r = rho)

Every joint is tangent, so the bore stays smooth all round, and the crown is
a round tip rho across, never a point. A taller round top (a half ellipse
with the same short crown) is no better built and grows with r squared: a
46 mm metal bore would rise 206 mm over its axis against this one's 31 mm.
It adds flow area, so its pressure drop only falls.

Materialize (dfm_reference) publishes minimum walls and details per process,
not overhang angles or self-supporting bore sizes; these figures are
loadpath's own, from the sources above.
"""
from __future__ import annotations

import math

__all__ = ["SELF_SUPPORT", "DESIGN_MARGIN", "needs_shaping", "round_limit", "crown_radius", "crown_height",
           "crown_area", "arch_sdf"]

DESIGN_MARGIN = 0.9

# per process: does anything inside need to self-support ("shaping"), the
# critical angle from level (degrees), and the largest level round bore that
# builds as drawn (mm). None: any size.
SELF_SUPPORT = {
    "fdm": {"shaping": True, "criticalAngle": 45.0, "roundBore": 15.0 / math.sin(math.radians(45.0)),
            "why": "the crown flatter than 45 degrees is bridged; a slicer bridges 15 mm (printcheck.BRIDGE_SPAN)"},
    "lpbf": {"shaping": True, "criticalAngle": 45.0, "roundBore": 8.0,
             "why": "powder-bed metal: level round holes up to ~8 mm build without support (published DMLS/SLM "
                    "design guidance); past that the crown drosses and sags, and no support comes out of a channel"},
    "sla": {"shaping": True, "criticalAngle": 45.0, "roundBore": 5.0 / math.sin(math.radians(45.0)),
            "why": "resin: no supports inside a closed channel; ~5 mm spans hold (printcheck.BRIDGE_SPAN)"},
    "dlp": {"shaping": True, "criticalAngle": 45.0, "roundBore": 5.0 / math.sin(math.radians(45.0)),
            "why": "as SLA"},
    "sls": {"shaping": False, "criticalAngle": 0.0, "roundBore": None,
            "why": "powder-bed polymer: the unfused powder carries every layer"},
    "mjf": {"shaping": False, "criticalAngle": 0.0, "roundBore": None,
            "why": "powder-bed polymer: the unfused powder carries every layer"},
}


def _entry(process):
    try:
        return SELF_SUPPORT[process]
    except KeyError:
        raise ValueError(f"no self-support figures for process {process!r}; one of {sorted(SELF_SUPPORT)}") from None


def needs_shaping(process) -> bool:
    """Does anything inside a part built by `process` have to hold itself up?"""
    return bool(_entry(process)["shaping"])


def round_limit(process) -> float:
    """The largest level round bore (mm) `process` builds as drawn (inf: any)."""
    d = _entry(process)["roundBore"]
    return math.inf if d is None else float(d)


def crown_radius(process) -> float:
    """The radius of the round tip that closes a crown `process` cannot build
    round (module docstring): DESIGN_MARGIN of the largest round bore's, so a
    meshed part stays inside the limit it is checked against. inf: any."""
    return DESIGN_MARGIN * 0.5 * round_limit(process)


def crown_height(r, process) -> float:
    """How far over its axis a level round channel of radius r must rise for
    `process` to build it: r while the circle builds as drawn, else the
    tangent flanks and round tip's (module docstring)."""
    r = float(r)
    e = _entry(process)
    if not e["shaping"] or 2.0 * r <= round_limit(process):
        return r
    rho = crown_radius(process)
    a = math.radians(e["criticalAngle"])
    return max((r - rho) / math.cos(a) + rho, r)


def crown_area(r, h, rho) -> float:
    """Cross-section area of a bore of radius r whose top rises h over its axis
    on tangent flanks to a round tip of radius rho (the hull of the two circles,
    crown centre h - rho over the axis). pi r^2 while h <= r."""
    r, h, rho = float(r), float(h), min(float(rho), float(r))
    d = h - rho
    if h <= r + 1e-12 or d <= r - rho:
        return math.pi * r * r
    beta = math.asin((r - rho) / d)
    tangent = math.sqrt(d * d - (r - rho) ** 2)
    return 0.5 * r * r * (math.pi + 2 * beta) + 0.5 * rho * rho * (math.pi - 2 * beta) + (r + rho) * tangent


def arch_sdf(l, u, r, h, rho):
    """Signed distance (< 0 inside) across a bore, l to the side and u up from
    its axis: the hull of its circle (radius r) and a crown circle (radius
    rho, centre h - rho up), tangent flanks between. h may be an array (a vault
    that rises along its length); where h - rho <= r - rho the crown circle is
    inside and this is the circle."""
    import numpy as np

    rho = np.float32(min(float(rho), float(r)))
    r = np.float32(r)
    d = np.maximum(np.asarray(h, dtype=np.float32) - rho, np.float32(1e-6))
    b = np.minimum((r - rho) / d, np.float32(1.0))
    c = np.sqrt(np.maximum(1.0 - b * b, 0.0))
    x = np.abs(l)
    k = c * u - b * x
    below = np.sqrt(x * x + u * u) - r
    above = np.sqrt(x * x + (u - d) ** 2) - rho
    flank = c * x + b * u - r
    return np.where(k < 0.0, below, np.where(k > c * d, above, flank)).astype(np.float32, copy=False)
