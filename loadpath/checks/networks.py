"""
Dual-network isolation verifier for exchanger-class parts (heat exchangers,
manifolds, anything carrying two fluids). Watertightness is not the interesting
invariant for these — NETWORK TOPOLOGY is: each fluid domain must connect its
own inlet/outlet pair, with zero cross-leak paths.

`check_networks(mesh, ports)` voxelizes the part, labels the void space with
26-connectivity, and maps each declared port to the void component it lands
in. Port naming convention: ports of the same circuit share the prefix before
the first underscore ("a_in" / "a_out" are one pair).

Resolution honesty: isolation verified at voxel pitch p means "no leak path
wider than ~p" — a pinhole narrower than a voxel can slip through the
occupancy grid. The report always carries the pitch used; surface UI copy
should state exactly that bound, not "leak-free".
"""
import os

import numpy as np
from scipy import ndimage

from sdf_kit import _solid_voxels

__all__ = ["check_networks"]

# Voxel-count ceiling for the occupancy grid (bool grid + int32 labels —
# 128M cells ≈ 128MB + 512MB transient, well under the child's RLIMIT_AS).
# The pitch chooser coarsens to fit under this rather than allocating
# unboundedly on a huge part.
_VOXEL_BUDGET = int(os.environ.get("CAD_NETWORKS_VOXEL_BUDGET", str(128_000_000)))

# 3x3x3 ones = 26-connectivity. Deliberately the LEAKIEST connectivity for the
# void: a diagonal voxel chain counts as a connected fluid path, so we err
# toward reporting leaks rather than hiding them.
_CONN = np.ones((3, 3, 3), bool)

_DILATION_CAP = 20  # min-wall search cutoff, in voxels


def _gap_voxels(mask_a, mask_b):
    """Solid separation between two void components, in voxels: dilate one
    mask a step at a time (26-neighborhood) until it overlaps the other. First
    overlap at step i means i-1 voxels of material in between. None when the
    gap exceeds the dilation cap — 'comfortably thick' is all we need then."""
    grow = mask_a if mask_a.sum() <= mask_b.sum() else mask_b
    other = mask_b if grow is mask_a else mask_a
    for i in range(1, _DILATION_CAP + 1):
        grow = ndimage.binary_dilation(grow, structure=_CONN)
        if np.any(grow & other):
            return i - 1
    return None


def _refill_plugs(solid, origin, pitch, plugs):
    """Virtually refill port openings before flood-fill. A real exchanger's
    ports open to the outside world, which connects BOTH plenums to the same
    air component and would flag every part as a leak. Two plug shapes:
      - capsule {"a", "b", "r"} — a port bore, refilled;
      - box {"c": center, "h": half-extents} — a full-face manifold window
        (hood lofts open the whole rectangular face; no capsule can cap
        that without bulging into the plenum and swallowing the probes).
    Both are rasterized with a 1.5-voxel dilation so grid aliasing can't
    leave a void thread along the wall. Plugs only ADD solid: they can mask
    a genuine leak only if the leak runs through the plugged volume itself,
    which the opening's subtraction already made void by construction."""
    shape = np.asarray(solid.shape)
    for plug in plugs:
        if "c" in plug and "h" in plug:
            c = np.asarray(plug["c"], float)
            h = np.asarray(plug["h"], float) + 1.5 * pitch
            lo = np.maximum(
                np.floor((c - h - origin) / pitch).astype(int), 0)
            hi = np.minimum(
                np.ceil((c + h - origin) / pitch).astype(int) + 1, shape)
            if np.any(lo >= hi):
                continue  # box entirely off-grid
            solid[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]] = True
            continue
        a = np.asarray(plug["a"], float)
        b = np.asarray(plug["b"], float)
        r = float(plug["r"]) + 1.5 * pitch
        lo = np.maximum(
            np.floor((np.minimum(a, b) - r - origin) / pitch).astype(int), 0)
        hi = np.minimum(
            np.ceil((np.maximum(a, b) + r - origin) / pitch).astype(int) + 1,
            shape)
        if np.any(lo >= hi):
            continue  # plug entirely off-grid
        idx = np.moveaxis(np.indices(tuple(hi - lo)), 0, -1) + lo
        centers = origin + idx * pitch
        ab = b - a
        denom = float(ab @ ab)
        if denom <= 0.0:
            t = np.zeros(centers.shape[:-1])
        else:
            t = np.clip(((centers - a) @ ab) / denom, 0.0, 1.0)
        dist = np.linalg.norm(centers - (a + t[..., None] * ab), axis=-1)
        sub = solid[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
        sub[dist <= r] = True


def check_networks(mesh, ports, pitch=None, plugs=None, min_feature=None):
    """Verify fluid-network isolation of `mesh` against declared `ports`.

    ports: list of {"name": str, "point": [x, y, z] mm, "r": mm} spheres —
    inlet/outlet locations in the void space, named `<circuit>_<role>`.
    plugs: optional list of {"a", "b", "r"} capsules to virtually refill
    before labeling — pass each port bore here whenever the ports open to the
    outside, or the shared outside air reads as a cross-circuit leak.
    min_feature: the smallest structure (mm) the verdict must resolve —
    typically min(wall thickness, channel width). Declaring it makes the
    pitch chooser fine enough to resolve it (pitch <= min_feature/2), and
    makes the check refuse to render a verdict it can't stand behind: a
    probe coarser than the walls reads aliasing as leaks (measured: a
    sound 1.2mm-wall exchanger "leaked" at a 2.66mm probe).

    Returns {
      "pitch": voxel pitch used (mm) — the honesty bound on leak width,
      "components": void components touching any port,
      "ports": {name: component id or None (port found no void)},
      "isolated": every port hit exactly one component, same-prefix ports
                  share a component, and no component carries two prefixes,
      "minWallVoxels": thinnest solid separation between port-bearing
                       components (voxels), None if single/too-far,
      "plugs": number of plug capsules refilled,
    }
    OR, when the part is so large that a min_feature-resolving pitch would
    blow the voxel budget, an INCONCLUSIVE report {"pitch", "inconclusive":
    True, "reason", "plugs"} — no isolated verdict at all, because a verdict
    at that resolution would be noise. Callers must treat inconclusive as
    "not verified", never as "leaked".

    Default pitch (no min_feature) = bounding-box max extent / 64, floored
    at 0.4mm — coarse on purpose: cheap enough to always run. Declare
    min_feature (or pass an explicit pitch) when the geometry has walls or
    channels finer than that.
    """
    extents = np.asarray(mesh.extents, float)
    if pitch is None:
        pitch = max(float(np.max(extents)) / 64.0, 0.4)
        if min_feature is not None:
            # Resolve the declared finest feature with >=2 voxels across it.
            pitch = min(pitch, max(float(min_feature) / 2.0, 0.2))
    pitch = float(pitch)

    # Coarsen (never refine) to fit the voxel budget on huge parts.
    def _cells(p):
        return float(np.prod(np.ceil(extents / p) + 4))

    while _cells(pitch) > _VOXEL_BUDGET:
        pitch = round(pitch * 1.25, 4)

    if min_feature is not None and pitch > float(min_feature) / 2.0 + 1e-9:
        return {
            "pitch": pitch,
            "inconclusive": True,
            "reason": (
                f"probe pitch {pitch:g}mm cannot resolve the declared "
                f"{float(min_feature):g}mm minimum feature within the voxel "
                "budget — isolation is UNVERIFIED at this scale (a verdict "
                "here would alias walls into phantom leaks)"
            ),
            "plugs": len(plugs) if plugs else 0,
        }

    solid, origin = _solid_voxels(mesh, pitch, pad=2)
    if plugs:
        _refill_plugs(solid, origin, pitch, plugs)
    labels, _ = ndimage.label(~solid, structure=_CONN)
    shape = np.asarray(labels.shape)

    assignment = {}
    ambiguous = False
    for port in ports:
        pt = np.asarray(port["point"], float)
        r = float(port["r"])
        lo = np.maximum(np.floor((pt - r - origin) / pitch).astype(int), 0)
        hi = np.minimum(np.ceil((pt + r - origin) / pitch).astype(int) + 1, shape)
        if np.any(lo >= hi):
            assignment[port["name"]] = None  # sphere entirely off-grid
            continue
        sub = labels[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
        idx = np.moveaxis(np.indices(sub.shape), 0, -1) + lo
        centers = origin + idx * pitch
        hits = sub[np.linalg.norm(centers - pt, axis=-1) <= r]
        hits = hits[hits > 0]
        if hits.size == 0:
            assignment[port["name"]] = None  # sphere touched only solid
            continue
        ids, counts = np.unique(hits, return_counts=True)
        if ids.size > 1:
            ambiguous = True  # straddles two voids — cannot claim isolation
        assignment[port["name"]] = int(ids[np.argmax(counts)])

    port_comps = sorted({c for c in assignment.values() if c is not None})

    # isolation: every port resolved cleanly, each circuit prefix lives in
    # exactly one component, and no component hosts two circuits.
    isolated = not ambiguous and all(c is not None for c in assignment.values())
    if isolated and assignment:
        prefix_of = {}  # component -> circuit prefix
        comp_of = {}    # circuit prefix -> component
        for name, comp in assignment.items():
            prefix = name.split("_")[0]
            if prefix_of.setdefault(comp, prefix) != prefix:
                isolated = False  # two circuits share a void — a leak
                break
            if comp_of.setdefault(prefix, comp) != comp:
                isolated = False  # one circuit split across voids — a blockage
                break

    min_wall = None
    if len(port_comps) >= 2:
        gaps = [
            _gap_voxels(labels == a, labels == b)
            for i, a in enumerate(port_comps)
            for b in port_comps[i + 1:]
        ]
        gaps = [g for g in gaps if g is not None]
        min_wall = min(gaps) if gaps else None

    return {
        "pitch": pitch,
        "components": len(port_comps),
        "ports": assignment,
        "isolated": bool(isolated),
        "minWallVoxels": min_wall,
        "plugs": len(plugs) if plugs else 0,
    }
