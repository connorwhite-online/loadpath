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
b = D_max sin a. A bigger bore (or vault) keeps its round bottom and sides
and rises over its axis as a half ellipse instead of a half circle, just
tall enough that its own flatter-than-a chord is b again:

    h = r tan(a) sqrt((2 r / b)^2 - 1)        (h = r at r = D_max / 2)

The ellipse meets the circle with the same (vertical) tangent at the sides,
so the bore stays smooth all round: an egg, not the 45 degree point of a
teardrop. It also adds flow area, so its pressure drop only falls.

Materialize (dfm_reference) publishes minimum walls and details per process,
not overhang angles or self-supporting bore sizes; these figures are
loadpath's own, from the sources above.
"""
from __future__ import annotations

import math

__all__ = ["SELF_SUPPORT", "DESIGN_MARGIN", "needs_shaping", "round_limit", "crown_height"]

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


def crown_height(r, process) -> float:
    """How far over its axis a level round channel of radius r must rise for
    `process` to build it: r while the circle builds as drawn, else the half
    ellipse's height (module docstring)."""
    r = float(r)
    e = _entry(process)
    if not e["shaping"] or 2.0 * r <= round_limit(process):
        return r
    a = math.radians(e["criticalAngle"])
    # the egg's crown is sized to DESIGN_MARGIN of the chord the round limit closes, so a
    # meshed part stays inside the limit it is checked against
    b = DESIGN_MARGIN * round_limit(process) * math.sin(a)
    return max(r * math.tan(a) * math.sqrt((2.0 * r / b) ** 2 - 1.0), r)
