"""What the designer drew, checked on the delivered part.

loadpath only adds material inside a fixed exterior: it never edits the
exterior and never removes what is preserved. Two readings say whether a
delivered mesh kept that promise:

- `exterior`: the part against the design space's own surface (a mesh
  exterior), nothing proud and all of it still there bar the skin's
  declared openings (enclosure.exterior_check). A clearance keep-out that
  crossed the skin cut a hole the designer never drew: the touch
  microphone's USB-C plug path left a square 12 x 6 mm hole where round
  17 has a 9.4 x 3.8 mm stadium with a rounded mouth.
- `drawn`: per preserved region, the share of it (less the keep-outs that
  cut it, spec.cuts_of) missing from the part. The same part's board clear zone
  had taken 91% of its rails, with their stops, lips and crush ribs.

Both are cheap (a few seconds) and run on every switch and feel part;
evals fail a case whose part does not keep them.
"""
from __future__ import annotations


MISSING_SHARE = 0.02     # a preserved region may lack this share of itself (mesh tolerance)


def drawn_check(study, mesh) -> dict:
    """{"regions": [{id, volume, missing, missingShare, ok}], "ok"}."""
    from .shapes import manifold_from_trimesh
    from .verify import _preserve_missing, drawn_kept

    part = manifold_from_trimesh(mesh)
    rows = []
    for r in study.preserve:
        try:
            # less the keep-outs that cut it (a bore; for all but the skin,
            # a part's way in)
            p = drawn_kept(study, [r])
        except Exception as exc:  # noqa: BLE001 - a shape with no exact solid is not checked
            rows.append({"id": r.id, "error": f"{type(exc).__name__}: {exc}"})
            continue
        v = float(p.volume())
        if v <= 1e-6:
            continue
        miss = float(_preserve_missing(p, part, mesh))
        share = miss / v
        rows.append({"id": r.id, "volume": round(v, 3), "missing": round(miss, 3),
                     "missingShare": round(share, 4), "ok": bool(share <= MISSING_SHARE)})
    return {"regions": rows, "ok": all(r.get("ok", True) for r in rows)}


def exterior_report(study, mesh) -> dict | None:
    """The part against a mesh design space's surface, bar the skin's
    openings; None when the design space is not one mesh or there is no
    skin (a part designed to be cut away anywhere has no exterior to keep)."""
    from .enclosure import exterior_check
    from .shapes import MeshShape
    from .spec import skin_openings
    if len(study.design_space) != 1 or not isinstance(study.design_space[0], MeshShape):
        return None
    sk = study.raw.get("skin")
    if not sk or (isinstance(sk, dict) and sk.get("open")):
        return None
    base = getattr(study, "base_dir", None)
    # the cuts the openings made (a box round a drawn port opens the port only)
    openings = skin_openings(sk.get("openings") or [], study.design_space[0], base) if isinstance(sk, dict) \
        else []
    drawn = [r.shape for r in study.preserve if r.id != "skin"]
    out = exterior_check(mesh, study.design_space[0], openings=openings, drawn=drawn)
    if not out["untouched"]:
        out["why"] = ("the part is missing some of the exterior (a keep-out crossing the skin where it has no "
                      "opening?)" if out["coverage"] < 0.999 else "the part stands proud of the exterior")
    return out
