"""A handheld pod surfaced from six sections: every join curvature-continuous.

    python examples/freeform/pod.py [out_dir]

One lofted body, then what a surfacing artist adds by hand and this does in
code (docs/FREEFORM.md):

- both ends closed by G2 caps (domes that continue the body's curvature);
- a mount boss standing through the top, filleted in (both faces cut back
  3 mm and blended G2), its top capped G2;
- a recessed flat (a sensor window's seat) blended down from a second
  opening, G2 where it leaves the body.

`continuity="G1"` builds the same pod tangent-only, for the comparison the
notes draw. Writes pod.step, pod.stl and a report of every shared edge.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from loadpath import surfacequality as Q    # noqa: E402
from loadpath import surfacing as S         # noqa: E402


def body():
    """Six sections along z: a waist, a head."""
    z = np.array([0.0, 22.0, 48.0, 76.0, 100.0, 118.0])
    a = np.array([13.0, 17.0, 15.0, 19.0, 22.0, 17.0])      # half width (x)
    b = np.array([10.0, 13.0, 12.0, 14.0, 15.0, 12.0])      # half depth (y)
    lift = np.array([0.0, 1.0, 2.5, 3.0, 1.5, -1.0])        # the spine rises and falls in y
    th = np.linspace(0, 2 * np.pi, 120, endpoint=False) - np.pi / 2      # the seam underneath
    secs = [np.c_[ai * np.cos(th), bi * np.sin(th) + li, np.full(len(th), zi)] for zi, ai, bi, li in zip(z, a, b, lift)]
    return S.loft(secs, poles=(28, 14), start=lambda i: secs[i][0])


def on_body(s, u, v, r_u, r_v, n=72):
    """A loop on the body round the surface point (u, v): an ellipse in the
    body's parameters, as (n, 3) points."""
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return s(u + r_u * np.cos(t), v + r_v * np.sin(t), grid=False)


def frame_at(s, u, v):
    p = s(np.array([u]), np.array([v]))[0, 0]
    e = 1e-4
    du = s(np.array([u + e]), np.array([v]))[0, 0] - s(np.array([u - e]), np.array([v]))[0, 0]
    dv = s(np.array([u]), np.array([v + e]))[0, 0] - s(np.array([u]), np.array([v - e]))[0, 0]
    n = np.cross(du, dv)
    n /= np.linalg.norm(n)
    c = s(np.array([0.0, 0.5]), np.array([v])).mean(axis=(0, 1))
    if n @ (p - c) < 0:
        n = -n
    x = du / np.linalg.norm(du)
    return p, n, x, np.cross(n, x)


def ring(c, x, y, r, n=64):
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return c + r * (np.cos(t)[:, None] * x + np.sin(t)[:, None] * y)


def build(continuity="G2"):
    from OCP.BRepAdaptor import BRepAdaptor_Curve
    s = body()
    f = s.face()
    zmid = lambda e: BRepAdaptor_Curve(e).Value(0.3).Z()      # noqa: E731
    ends = sorted(S.boundary_edges(f), key=zmid)
    # the mount boss: a round post standing through the top, filleted in
    p, n, x, y = frame_at(s, 0.5, 0.36)
    post = S.fit_surface(np.stack([ring(p + n * h, x, y, 4.5, 48) for h in np.linspace(-4.0, 10.0, 8)], 1), (16, 5),
                         uperiodic=True, fair=1e-9).face()
    top = max(S.boundary_edges(post), key=lambda e: float(np.array(
        [BRepAdaptor_Curve(e).Value(0.3).X(), BRepAdaptor_Curve(e).Value(0.3).Y(), BRepAdaptor_Curve(e).Value(0.3).Z()]) @ n))
    f, boss_blend, post = S.fillet(f, post, 3.0, continuity, keep=(None, p + 9.0 * n))
    # the window seat: a second opening, a flat 2.2 mm below the skin
    p2, n2, x2, y2 = frame_at(s, 0.5, 0.74)
    f = S.cut(f, on_body(s, 0.5, 0.74, 0.11, 0.085))
    seat_c = p2 - 2.2 * n2
    seat = S.fit_surface(np.stack([seat_c + a * x2 * 7.5 + b_ * y2 * 6.0 for a in np.linspace(-1.3, 1.3, 8)
                                   for b_ in np.linspace(-1.3, 1.3, 8)]).reshape(8, 8, 3), (4, 4), fair=0.0)
    t = np.linspace(0, 2 * np.pi, 64, endpoint=False)
    seat_in = S.cut(seat, seat_c + 7.5 * np.cos(t)[:, None] * x2 + 6.0 * np.sin(t)[:, None] * y2, keep="inside")
    seat_hole = max(S.hole_edges(f), key=lambda h: np.mean([zmid(e) for e in h]))[0]
    parts = {"body": f, "post": post, "seat": seat_in}
    parts["boss blend"] = boss_blend
    parts["seat blend"] = S.blend((f, seat_hole), (seat_in, S.boundary_edges(seat_in)[0]), (continuity, "G1"), lead=(1.0, 0.7))
    report = {"continuity": continuity, "blends": {k: parts[k].fit for k in ("boss blend", "seat blend")}, "caps": {}}
    for name, e, on in (("tail cap", ends[0], f), ("head cap", ends[1], f), ("post cap", top, post)):
        parts[name], err = S.cap([(e, on, continuity)])
        report["caps"][name] = {k: err[k] for k in ("G0", "G1", "G2", "poles")}
    solid = S.sew_solid(list(parts.values()))
    rows, summ = Q.edge_continuity(solid)
    report["edges"] = summ
    return solid, parts, report


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "out/freeform-pod"
    os.makedirs(out, exist_ok=True)
    for c in ("G2", "G1"):
        solid, parts, report = build(c)
        m = S.mesh(solid, 0.01, 0.05)
        tag = "" if c == "G2" else "-g1"
        S.write_step(solid, os.path.join(out, f"pod{tag}.step"))
        m.export(os.path.join(out, f"pod{tag}.stl"))
        report["joins"] = [{k: j[k] for k in ("faces", "grade", "G0", "G1", "G2")}
                           for j in Q.join_sheet(solid, os.path.join(out, f"pod{tag}-joins.png"), mesh=m)]
        report["watertight"] = bool(m.is_watertight)
        report["faces"] = len(parts)
        with open(os.path.join(out, f"pod{tag}.json"), "w") as fh:
            json.dump(report, fh, indent=1)
        print(c, json.dumps(report["edges"]), "watertight", m.is_watertight, "faces", len(parts))
