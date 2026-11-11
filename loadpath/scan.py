"""
Scans in: a lidar scan, a structured-light scan, a photogrammetry cloud or
a Gaussian splat as the measured world a part is designed around.

A scan is a measurement, and a part built around one is only as good as
what the scan knows about where each surface is. So everything here comes
with how well it is known, and the clearances come from that, not from a
margin someone picked:

- **the device**: its published accuracy (the systematic part, which no
  amount of points averages away) and its scale error (DEVICES; a unit's
  calibration certificate wins: pass device={...});
- **the noise, measured**: at every point, the scatter of its neighbours
  about a local quadric (a smooth surface's own curvature is not noise), and
  how many points the surface estimate there averages (correlated scatter,
  as on a phone scan's smoothed mesh, counts for less);
- **the sampling**: points a spacing h apart on a surface of radius rho miss
  its bulge between them by h^2 / (8 rho); where the scan saw nothing at all,
  the surface is not known, and the report says where and how big;
- **the scale**: a splat or a photogrammetry cloud has none until a known
  dimension is given (`reference`); its tolerance and the fit's own
  uncertainty set the scale's, which grows with distance from the datum.

What comes out:

- `Scan`: the points in mm, oriented normals, and per point the measured
  noise, spacing, curvature radius and the one-sided surface uncertainty U
  at the confidence asked (default 95%);
- `fit_primitives`: planes (mounting faces, a wall, a table) and cylinders
  (bores, and the tubes and bars a clamp goes round), each with its
  parameters' uncertainty from the least-squares fit, the device and the
  scale: a bar's diameter as 31.79 +- 0.21 mm (95%), not 31.79;
- `reconstruct`: a watertight surface of the scanned object, every vertex
  with its uncertainty, and the unseen patches flagged;
- `draft_around` (scanstudy.py): a study that builds a part around the
  object: the object kept out by its own uncertainty plus the process's
  scatter, mounting faces and bars as supports and loads, bores and bars
  fitted (fits.py) with the measured diameter's band.

    from loadpath.scan import load_scan, fit_primitives
    s = load_scan("bar.ply", device="iphone-lidar")
    feats = fit_primitives(s)

    loadpath scan bar.ply --device iphone-lidar --features-only
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass, field

import numpy as np

__all__ = ["DEVICES", "Scan", "load_scan", "fit_primitives", "apply_reference", "reconstruct", "device_spec",
           "z_value", "summarize", "scale_features", "calibrate_device"]

# Published accuracy of common scanners, as sigma (mm) of a point's position
# along the surface normal that does not average out (calibration, range
# bias, registration), scale (relative), and the source. Where a source gives
# a +- bound with no confidence, it is read as 2 sigma; a datasheet "up to"
# is the best case and is used as 1 sigma. Your unit's calibration
# certificate, or a scan of a known artefact (calibrate_device), wins.
DEVICES = {
    "iphone-lidar": {
        "sigma": 5.0, "scale": 0.01, "kind": "phone lidar (ARKit depth, fused mesh)",
        "source": "Luetzenburg, Kroon & Bjork 2021, Sci. Rep. 11:22221 (iPhone 12 Pro): absolute accuracy "
                  "+-1 cm, objects under ~10 cm not resolved. Apple publishes no figure."},
    "ipad-lidar": {
        "sigma": 5.0, "scale": 0.01, "kind": "tablet lidar (ARKit depth, fused mesh)",
        "source": "same sensor as iphone-lidar (Luetzenburg et al. 2021); Vogt et al. 2021, Technologies 9:25, "
                  "found it unsuited to small parts against an industrial scanner"},
    "artec-eva": {
        "sigma": 0.1, "scale": 0.0003, "kind": "handheld structured light",
        "source": "Artec Eva datasheet: 3D point accuracy up to 0.1 mm, accuracy over distance up to 0.03% "
                  "over 100 cm"},
    "artec-leo": {
        "sigma": 0.1, "scale": 0.0003, "kind": "handheld structured light",
        "source": "Artec Leo datasheet: 3D point accuracy up to 0.1 mm, up to 0.03% over 100 cm"},
    "leica-rtc360": {
        "sigma": 1.9, "scale": 0.0001, "kind": "terrestrial laser scanner",
        "source": "Leica RTC360 datasheet: 3D point accuracy 1.9 mm at 10 m (2.9 mm at 20 m)"},
    "leica-blk360": {
        "sigma": 6.0, "scale": 0.0003, "kind": "terrestrial laser scanner (first generation)",
        "source": "Leica BLK360 (G1) datasheet: 3D point accuracy 6 mm at 10 m"},
    "faro-focus": {
        "sigma": 1.0, "scale": 0.0001, "kind": "terrestrial laser scanner",
        "source": "FARO Focus Premium/S datasheet: ranging error +-1 mm"},
    "splat": {
        "sigma": None, "scale": None, "kind": "Gaussian splat (photogrammetric: no metric scale)",
        "source": "no published accuracy: a splat's geometry is fitted to images. Its scale comes only from a "
                  "reference; give device={'sigma': ...} measured on a known artefact to bound its bias"},
    "photogrammetry": {
        "sigma": None, "scale": None, "kind": "photogrammetry (no metric scale)",
        "source": "no published accuracy: give a reference for scale and a measured sigma"},
    "measured": {
        "sigma": 0.0, "scale": 0.0, "kind": "synthetic or exact (testing)",
        "source": "no device error: the measured scatter only"},
}

_UNITS = {"m": 1000.0, "cm": 10.0, "mm": 1.0, "in": 25.4, "ft": 304.8, "um": 1e-3}


def z_value(confidence: float, two_sided: bool = False) -> float:
    """The standard normal quantile for a confidence (0.95 -> 1.645
    one-sided, 1.96 two-sided)."""
    from scipy.stats import norm
    c = float(confidence)
    if not 0.5 < c < 1.0:
        raise ValueError("confidence is between 0.5 and 1 (e.g. 0.95)")
    return float(norm.ppf(1 - (1 - c) / 2) if two_sided else norm.ppf(c))


def device_spec(device) -> dict:
    """A device's sigma (mm), scale (relative) and source, from a DEVICES
    name or a dict {"sigma", "scale", "source"} (a dict may name a base
    device under "base" and override its numbers)."""
    if device is None:
        return {"name": None, "sigma": None, "scale": None, "kind": "unknown", "source": "not given"}
    if isinstance(device, str):
        if device not in DEVICES:
            raise ValueError(f"device must be one of {sorted(DEVICES)} or a dict {{sigma, scale}}; got {device!r}")
        return {"name": device, **DEVICES[device]}
    d = dict(device)
    base = d.pop("base", None)
    out = {"name": base or "custom", **(DEVICES[base] if base else {"kind": "custom", "source": "given"})}
    out.update(d)
    if base and "source" not in d:
        out["source"] = DEVICES[base]["source"] + " (overridden: " + ", ".join(sorted(d)) + ")"
    return out


@dataclass
class Scan:
    points: np.ndarray                    # (N, 3) mm
    normals: np.ndarray                   # (N, 3) unit, oriented outward
    spacing: np.ndarray                   # (N,) mm, sqrt(area per point)
    noise: np.ndarray                     # (N,) mm, measured scatter about a local quadric
    rho: np.ndarray                       # (N,) mm, smallest local curvature radius (inf on flats)
    k_eff: np.ndarray                     # (N,) effective independent points per local estimate
    extra: np.ndarray | None = None       # (N,) mm, a splat's own thickness along its normal
    colors: np.ndarray | None = None
    device: dict = field(default_factory=dict)
    scale: float = 1.0                    # applied to the file's coordinates (incl. units)
    scale_sigma: float | None = None      # relative; None: the scale is not known
    datum: np.ndarray | None = None       # where the scale is exact (a reference's centre)
    confidence: float = 0.95
    viewpoint: np.ndarray | None = None
    report: dict = field(default_factory=dict)
    path: str | None = None

    def sigma_parts(self, P=None, idx=None):
        """Per point (or at points P, from their nearest scan point): the
        1-sigma random part (noise over the points averaged), the device's
        systematic part, and the scale's (relative scale x distance from the
        datum), in mm."""
        if idx is None:
            idx = np.arange(len(self.points)) if P is None else self.nearest(P)[1]
        X = self.points[idx] if P is None else np.asarray(P, float)
        rnd2 = self.noise[idx] ** 2 / np.maximum(self.k_eff[idx], 1.0)
        if self.extra is not None:
            rnd2 = rnd2 + self.extra[idx] ** 2 / np.maximum(self.k_eff[idx], 1.0)
        sys = float(self.device.get("sigma") or 0.0)
        ss = self.scale_sigma
        dat = self.datum if self.datum is not None else self.points.mean(0)
        sc = (ss if ss is not None else 0.0) * np.linalg.norm(X - dat, axis=1)
        return np.sqrt(rnd2), np.full(len(idx), sys), sc

    def uncertainty(self, P=None, idx=None, confidence=None, sagitta=True):
        """One-sided surface uncertainty U (mm): how far the real surface may
        sit outside the measured one at `confidence`: z * sqrt(random^2 +
        device^2 + scale^2), plus the sampling sagitta h^2 / (8 rho)."""
        if idx is None and P is not None:
            idx = self.nearest(P)[1]
        if idx is None:
            idx = np.arange(len(self.points))
        r, s, c = self.sigma_parts(P, idx)
        z = z_value(confidence or self.confidence)
        U = z * np.sqrt(r * r + s * s + c * c)
        if sagitta:
            U = U + self.spacing[idx] ** 2 / (8.0 * self.rho[idx])
        return U

    @property
    def scaled(self) -> bool:
        return self.scale_sigma is not None

    _tree = None

    def tree(self):
        if self._tree is None:
            from scipy.spatial import cKDTree
            self._tree = cKDTree(self.points)
        return self._tree

    def nearest(self, P):
        d, i = self.tree().query(np.asarray(P, float), workers=-1)
        return d, i

    def transformed(self, s: float, s_sigma: float | None, datum):
        """A copy with every length times s (the scale from a reference)."""
        import copy
        out = copy.copy(self)
        out.points = self.points * s
        out.spacing = self.spacing * s
        out.noise = self.noise * s
        out.rho = self.rho * s
        out.extra = None if self.extra is None else self.extra * s
        out.scale = self.scale * s
        out.scale_sigma = s_sigma
        out.datum = None if datum is None else np.asarray(datum, float) * s
        out.viewpoint = None if self.viewpoint is None else self.viewpoint * s
        out._tree = None
        out.report = dict(self.report)
        return out


# ---------------------------------------------------------------- local statistics

def _knn(P, k):
    from scipy.spatial import cKDTree
    t = cKDTree(P)
    d, i = t.query(P, k=k + 1, workers=-1)
    return t, d[:, 1:], i[:, 1:]


def _pca_normals(P, idx):
    Q = P[idx] - P[idx].mean(1, keepdims=True)
    C = np.einsum("nki,nkj->nij", Q, Q)
    w, v = np.linalg.eigh(C)
    return v[:, :, 0], w


def _local_quadric(P, N, idx, chunk=50_000):
    """Per point: rms scatter of its neighbours about the local quadric
    z = a u^2 + b uv + c v^2 + d u + e v + f (the surface's own curvature is
    not noise), corrected for the 6 fitted parameters, and the smallest
    principal curvature radius of the fit."""
    n = len(P)
    k = idx.shape[1] + 1
    noise = np.empty(n)
    rho = np.empty(n)
    resid = np.empty((n, k))
    for s in range(0, n, chunk):
        e = min(n, s + chunk)
        nn = N[s:e]
        ref = np.where(np.abs(nn[:, 0:1]) < 0.9, np.array([[1.0, 0, 0]]), np.array([[0, 1.0, 0]]))
        e1 = np.cross(nn, ref)
        e1 /= np.linalg.norm(e1, axis=1, keepdims=True)
        e2 = np.cross(nn, e1)
        nb = np.concatenate([np.arange(s, e)[:, None], idx[s:e]], 1)
        Q = P[nb] - P[s:e, None, :]
        u = np.einsum("nki,ni->nk", Q, e1)
        v = np.einsum("nki,ni->nk", Q, e2)
        w = np.einsum("nki,ni->nk", Q, nn)
        sc = np.maximum(np.sqrt((u * u + v * v).mean(1)), 1e-12)[:, None]
        un, vn = u / sc, v / sc                       # conditioning
        A = np.stack([un * un, un * vn, vn * vn, un, vn, np.ones_like(un)], 2)
        AtA = np.einsum("nki,nkj->nij", A, A) + 1e-9 * np.eye(6)
        Atb = np.einsum("nki,nk->ni", A, w)
        x = np.linalg.solve(AtA, Atb[..., None])[..., 0]
        r = w - np.einsum("nki,ni->nk", A, x)
        resid[s:e] = r
        noise[s:e] = np.sqrt((r * r).sum(1) / max(k - 6, 1))
        a, b, c = x[:, 0] / sc[:, 0] ** 2, x[:, 1] / sc[:, 0] ** 2, x[:, 2] / sc[:, 0] ** 2
        # shape operator of the height field at the origin (slope ~0): [[2a, b], [b, 2c]]
        tr, det = 2 * (a + c), 4 * a * c - b * b
        disc = np.sqrt(np.maximum(tr * tr / 4 - det, 0))
        kmax = np.maximum(np.abs(tr / 2 + disc), np.abs(tr / 2 - disc))
        rho[s:e] = 1.0 / np.maximum(kmax, 1e-12)
    return noise, rho, resid


def _orient(P, N, idx, viewpoint=None, viewpoints=None):
    """Consistent outward normals: toward the sensor when its position is
    known, else propagated over a minimum spanning tree of the neighbour
    graph (Hoppe et al. 1992), each component then flipped so its normals
    point away from its centroid on balance."""
    if viewpoints is not None:
        s = np.sign(np.einsum("ij,ij->i", N, viewpoints - P))
        return N * np.where(s == 0, 1, s)[:, None], "sensor"
    if viewpoint is not None:
        s = np.sign(np.einsum("ij,ij->i", N, np.asarray(viewpoint)[None] - P))
        return N * np.where(s == 0, 1, s)[:, None], "sensor"
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import breadth_first_order, connected_components, minimum_spanning_tree
    n, k = idx.shape
    rows = np.repeat(np.arange(n), k)
    cols = idx.ravel()
    w = 1.0 - np.abs(np.einsum("ij,ij->i", N[rows], N[cols])) + 1e-6
    G = coo_matrix((w, (rows, cols)), shape=(n, n)).tocsr()
    G = G.maximum(G.T)
    T = minimum_spanning_tree(G)
    T = T + T.T
    ncomp, lab = connected_components(T, directed=False)
    N = N.copy()
    for c in range(ncomp):
        members = np.flatnonzero(lab == c)
        root = members[np.argmax(P[members, 2])]
        if N[root, 2] < 0:
            N[root] *= -1
        order, pred = breadth_first_order(T, root, directed=False, return_predecessors=True)
        flip = np.zeros(n, bool)
        # walk parents before children: a child flips when it disagrees with its (final) parent
        for v in order[1:]:
            p = pred[v]
            pn = -N[p] if flip[p] else N[p]
            flip[v] = float(N[v] @ pn) < 0
        N[order] *= np.where(flip[order], -1.0, 1.0)[:, None]
        cen = P[members].mean(0)
        if np.einsum("ij,ij->", N[members], P[members] - cen) < 0:
            N[members] *= -1
    return N, "propagated"


def _k_eff(resid, idx, noise):
    """Independent points per local estimate: k, shrunk when neighbours'
    scatter is correlated (a fused phone mesh is smoothed: its neighbours
    move together, and averaging them buys nothing)."""
    k = idx.shape[1] + 1
    r0 = resid[:, 0]
    rn = resid[:, 1:].mean(1)
    den = np.sqrt((r0 * r0).mean() * (rn * rn).mean())
    rho = float(np.clip((r0 * rn).mean() / den, 0.0, 0.99)) if den > 0 else 0.0
    keff = k * (1 - rho) / (1 + rho)
    return np.full(len(noise), max(keff, 1.0)), rho


def _splat_points(sp, P):
    """A splat's surface point is its mean, its normal the axis of its
    smallest scale, its thickness that scale (the surface is somewhere in
    it). Near-transparent splats (opacity under 0.1) say nothing about where
    a surface is, and are dropped."""
    q = np.asarray(sp["quats"], float)
    q = q / np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-12)
    w, x, y, z = q.T
    R = np.stack([
        np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], 1),
        np.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], 1),
        np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], 1)], 1)
    s = np.asarray(sp["scales"], float)
    j = np.argmin(s, 1)
    nrm = R[np.arange(len(R)), :, j]
    thick = s[np.arange(len(s)), j]
    keep = np.asarray(sp["opacity"]) >= 0.1
    return keep, nrm, thick


def _voxel_down(P, pitch, extras):
    key = np.floor((P - P.min(0)) / pitch).astype(np.int64)
    _, inv, cnt = np.unique(key, axis=0, return_inverse=True, return_counts=True)
    inv = inv.ravel()
    m = len(cnt)

    def avg(a):
        out = np.zeros((m,) + a.shape[1:])
        np.add.at(out, inv, a)
        return out / cnt.reshape((-1,) + (1,) * (a.ndim - 1))
    return avg(P), {k: (None if v is None else avg(v)) for k, v in extras.items()}, cnt


def load_scan(src, *, device=None, units: str | None = None, scale: float | None = None, up: str | None = None,
              crop=None, keep: str = "all", max_points: int = 300_000, k: int = 16,
              viewpoint=None, confidence: float = 0.95, reference=None) -> Scan:
    """Read a scan and measure it (see the module docstring).

    device: a DEVICES name or {sigma, scale, source}; without one, only the
      measured scatter counts and the report says so.
    units: the file's length unit ("m", "mm", ...). LAS, E57 and PCL clouds
      are metres by convention; splats have none (give a reference). For any
      other file it is required: a guess here moves every clearance.
    scale: a factor instead (applied after units).
    up: the file's up axis ("+y" for glTF, most splat trainers' "-y").
    crop: [[xmin, ymin, zmin], [xmax, ymax, zmax]] in the file's units, to
      cut the object out of a room.
    keep: "all" clusters (the default: a scene's separate objects, the bar
      and the light hanging off it, are all real) less specks under 0.5% of
      the points, or "largest" (one object out of a room).
    max_points: points beyond this are averaged into voxels first (which
      averages their noise too; the measured scatter is the averaged one).
    reference: known dimensions that fix the scale (apply_reference).
    """
    from .scanio import read_scan
    raw = read_scan(src)
    P = np.asarray(raw["points"], float)
    rep = {"source": os.path.basename(os.fspath(src)), "format": raw["format"], "pointsRead": int(len(P)),
           "todo": []}
    spec = device_spec(device if device is not None else ("splat" if "splat" in raw else None))
    if spec.get("sigma") is None and device is not None or "splat" in raw and device is None:
        rep["todo"].append(f"{spec.get('name')}: no published accuracy, so nothing bounds this scan's systematic "
                           "error (only its scatter and the scale reference count); scan a known artefact and "
                           "pass device={'base': ..., 'sigma': measured} to bound it")
    if device is None and "splat" not in raw:
        rep["todo"].append("no device given: only the measured scatter counts, not the scanner's systematic "
                           f"error; pass device= one of {sorted(k for k in DEVICES if DEVICES[k]['sigma'])}"
                           " or {sigma, scale}")
    extras = {"normals": raw.get("normals"), "colors": raw.get("colors"), "thick": None,
              "viewpoints": raw.get("viewpoints")}
    if crop is not None:
        lo, hi = np.asarray(crop, float)
        m = np.all((P >= lo) & (P <= hi), 1)
        P = P[m]
        extras = {k2: (None if v is None else np.asarray(v)[m]) for k2, v in extras.items()}
        if "splat" in raw:
            raw["splat"] = {k2: np.asarray(v)[m] for k2, v in raw["splat"].items()}
        rep["cropped"] = int(len(P))
    if "splat" in raw:
        keepm, nrm, thick = _splat_points(raw["splat"], P)
        P = P[keepm]
        extras = {k2: (None if v is None else np.asarray(v)[keepm]) for k2, v in extras.items()}
        extras["normals"] = nrm[keepm]
        extras["thick"] = thick[keepm]
        rep["splat"] = {"gaussians": int(len(keepm)), "kept": int(keepm.sum()),
                        "droppedTransparent": int((~keepm).sum()),
                        "thicknessMedian": float(np.median(thick[keepm])) if keepm.any() else None}
    if len(P) < 50:
        raise ValueError(f"{src}: {len(P)} points left (crop?)")

    # units -> mm
    u = units or raw.get("unitsHint")
    if u is None and "splat" not in raw:
        raise ValueError(f"{src}: give units= ('m', 'mm', 'cm', 'in'): this format does not say, and a guess "
                         "would move every clearance (or scale= a factor to mm)")
    if u is not None and u not in _UNITS:
        raise ValueError(f"units must be one of {sorted(_UNITS)}")
    f_units = _UNITS[u] if u else 1.0
    rep["units"] = u or "arbitrary (splat)"
    if u and not units:
        rep["unitsAssumed"] = f"{u} ({raw['format']} convention)"
    fac = f_units * (float(scale) if scale else 1.0)
    P = P * fac
    if extras["thick"] is not None:
        extras["thick"] = extras["thick"] * fac
    vp = raw.get("viewpoint") if viewpoint is None else np.asarray(viewpoint, float) / fac * fac
    if vp is not None and viewpoint is None:
        vp = np.asarray(vp, float) * fac
    if extras["viewpoints"] is not None:
        extras["viewpoints"] = extras["viewpoints"] * fac

    # up axis -> +z
    if up:
        from .exterior import _up_matrix
        T = _up_matrix(up)
        P = P @ T.T
        if extras["normals"] is not None:
            extras["normals"] = extras["normals"] @ T.T
        if vp is not None:
            vp = T @ vp
        if extras["viewpoints"] is not None:
            extras["viewpoints"] = extras["viewpoints"] @ T.T

    # bound the point count: average into voxels
    if len(P) > max_points:
        ext = np.ptp(P, 0)
        # a voxel pitch from the area the points cover
        _, d0, _ = _knn(P[np.random.default_rng(0).choice(len(P), min(len(P), 20000), replace=False)], 1)
        h0 = float(np.median(d0))
        pitch = h0 * math.sqrt(len(P) / max_points) * 1.05
        for _ in range(6):
            P2, ex2, cnt = _voxel_down(P, pitch, {k2: v for k2, v in extras.items() if k2 != "viewpoints"})
            if len(P2) <= max_points * 1.05:
                break
            pitch *= 1.2
        rep["voxelAveraged"] = {"pitch": round(pitch, 4), "points": int(len(P2)),
                                "meanPerVoxel": round(float(cnt.mean()), 2)}
        P = P2
        vps = None
        if extras["viewpoints"] is not None:
            vps = None                      # per-point sensor positions do not survive averaging
        extras = {**ex2, "viewpoints": vps}
        if extras["normals"] is not None:
            nn = np.linalg.norm(extras["normals"], axis=1, keepdims=True)
            extras["normals"] = extras["normals"] / np.maximum(nn, 1e-12)
        del ext

    # outliers: points whose neighbours are much farther than everyone's (floaters, mixed pixels)
    _, d, idx = _knn(P, k)
    md = d.mean(1)
    thr = md.mean() + 3 * md.std()
    ok = md <= thr
    if (~ok).any():
        P = P[ok]
        extras = {k2: (None if v is None else np.asarray(v)[ok]) for k2, v in extras.items()}
        rep["outliersRemoved"] = int((~ok).sum())
        _, d, idx = _knn(P, k)

    # connected clusters: specks (under 0.5% of the points, at least 30)
    # go; with keep="largest", everything but the largest
    if keep not in ("all", "largest"):
        raise ValueError("keep is 'all' or 'largest'")
    if True:
        from scipy.sparse import coo_matrix
        from scipy.sparse.csgraph import connected_components
        h_med = float(np.median(d[:, :6].mean(1)))
        link = d < 3 * h_med
        rows = np.repeat(np.arange(len(P)), k)[link.ravel()]
        cols = idx.ravel()[link.ravel()]
        G = coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(P), len(P)))
        nc, lab = connected_components(G, directed=False)
        if nc > 1:
            cnt = np.bincount(lab)
            if keep == "largest":
                m = lab == np.argmax(cnt)
            else:
                m = cnt[lab] >= max(30, 0.005 * len(P))
            rep["clusters"] = {"found": int(nc), "kept": int(m.sum()), "dropped": int((~m).sum())}
            P = P[m]
            extras = {k2: (None if v is None else np.asarray(v)[m]) for k2, v in extras.items()}
            _, d, idx = _knn(P, k)

    # normals: PCA where the file has none; oriented
    pca_n, ev = _pca_normals(P, idx)
    N = extras["normals"]
    if N is None:
        N = pca_n
        N, how = _orient(P, N, idx, vp, extras.get("viewpoints"))
        rep["normals"] = f"estimated (PCA of {k} neighbours), oriented: {how}"
    else:
        N = np.asarray(N, float)
        if "splat" in raw:
            N, how = _orient(P, N, idx, vp, None)
            rep["normals"] = f"splat smallest axes, oriented: {how}"
        else:
            rep["normals"] = "from the file"

    # spacing: area per point from the k-neighbour disc
    rk = d[:, -1]
    spacing = rk * math.sqrt(math.pi / (k + 1))
    noise, rho, resid = _local_quadric(P, N, idx)
    # smooth the per-point estimates over the neighbourhood (one point's
    # scatter is itself a noisy number)
    noise = np.median(np.concatenate([noise[:, None], noise[idx]], 1), 1)
    rho = np.median(np.concatenate([rho[:, None], rho[idx]], 1), 1)
    keff, corr = _k_eff(resid, idx, noise)
    s = Scan(points=P, normals=N, spacing=spacing, noise=noise, rho=rho, k_eff=keff,
             extra=extras["thick"], colors=extras["colors"], device=spec, viewpoint=vp,
             confidence=confidence, path=os.fspath(src), scale=fac)
    if u is not None:
        # metric units (a lidar, a structured-light scanner): the scale is
        # known to the device's scale error (zero, with a todo, when no
        # device says what it is)
        s.scale_sigma = float(spec.get("scale") or 0.0)
    rep.update({
        "points": int(len(P)),
        "bounds": np.round([P.min(0), P.max(0)], 3).tolist(),
        "spacing": {"median": round(float(np.median(spacing)), 4), "p95": round(float(np.percentile(spacing, 95)), 4)},
        "noise": {"median": round(float(np.median(noise)), 4), "p95": round(float(np.percentile(noise, 95)), 4),
                  "neighbourCorrelation": round(corr, 3), "independentPerEstimate": round(float(keff[0]), 2)},
        "device": {k2: spec.get(k2) for k2 in ("name", "kind", "sigma", "scale", "source")},
        "confidence": confidence,
    })
    if s.extra is not None:
        rep["noise"]["splatThicknessMedian"] = round(float(np.median(s.extra)), 4)
    s.report = rep
    if reference:
        s = apply_reference(s, reference)
    elif not s.scaled:
        rep["todo"].append("the scale is not known (a splat, or no units/device): give reference= a known "
                           "dimension (a fitted feature's diameter or two features' distance) before "
                           "trusting any size")
    _finish_report(s)
    return s


def _finish_report(s: Scan):
    if s.scaled:
        U = s.uncertainty()
        s.report["uncertainty"] = {"median": round(float(np.median(U)), 4),
                                   "p95": round(float(np.percentile(U, 95)), 4),
                                   "confidence": s.confidence, "oneSided": True}
    else:
        s.report.pop("uncertainty", None)


# ---------------------------------------------------------------- primitives

def _unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


def _perp_frame(a):
    e1 = _unit(np.cross(a, [1.0, 0, 0] if abs(a[0]) < 0.9 else [0, 1.0, 0]))
    return e1, np.cross(a, e1)


def _cyl_residual(P, c, a, r):
    q = P - c
    t = q @ a
    return np.linalg.norm(q - np.outer(t, a), axis=1) - r


def _refine_cylinder(P, w, c, a, r):
    """Weighted Gauss-Newton on (axis: 2 angles, point: 2 offsets, r)."""
    from scipy.optimize import least_squares
    a = _unit(a)
    e1, e2 = _perp_frame(a)
    c0 = c - ((c - P.mean(0)) @ a) * a
    sw = np.sqrt(w)

    def unpack(x):
        ax = _unit(a + x[0] * e1 + x[1] * e2)
        cc = c0 + x[2] * e1 + x[3] * e2
        return cc, ax, x[4]

    def fun(x):
        cc, ax, rr = unpack(x)
        return sw * _cyl_residual(P, cc, ax, rr)
    sol = least_squares(fun, np.array([0, 0, 0, 0, r], float), method="lm", x_scale="jac")
    cc, ax, rr = unpack(sol.x)
    J = sol.jac
    return cc, ax, float(abs(rr)), J, sol.fun / np.maximum(sw, 1e-12)


def _cov(J, resid_w, n_eff, npar):
    """Parameter covariance of a weighted least-squares fit, the residuals'
    variance estimated from the fit, inflated by n / n_eff when neighbouring
    residuals are correlated (fewer independent points than residuals)."""
    n = len(resid_w)
    s2 = float((resid_w ** 2).sum()) / max(n - npar, 1)
    JtJ = J.T @ J
    try:
        C = np.linalg.inv(JtJ) * s2
    except np.linalg.LinAlgError:
        C = np.linalg.pinv(JtJ) * s2
    return C * (n / max(n_eff, npar + 1.0))


def _component(scan, sel):
    """The largest connected patch of the selected points."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    ids = np.flatnonzero(sel)
    if len(ids) < 3:
        return ids
    from scipy.spatial import cKDTree
    t = cKDTree(scan.points[ids])
    kk = min(8, len(ids) - 1)
    d, j = t.query(scan.points[ids], k=kk + 1, workers=-1)
    h = scan.spacing[ids]
    link = d[:, 1:] < 2.5 * np.maximum(h[:, None], h[j[:, 1:]])
    rows = np.repeat(np.arange(len(ids)), kk)[link.ravel()]
    cols = j[:, 1:].ravel()[link.ravel()]
    G = coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(ids), len(ids)))
    nc, lab = connected_components(G, directed=False)
    return ids[lab == np.argmax(np.bincount(lab))]


def _area(scan, ids):
    return float((scan.spacing[ids] ** 2).sum())


def fit_primitives(scan: Scan, *, min_points: int | None = None, max_features: int = 24, tries: int = 1000,
                   seed: int = 0, kinds=("plane", "cylinder")) -> dict:
    """Planes and cylinders in a scan, found by RANSAC on points with
    normals (Schnabel, Wahl & Klein 2007, simplified) and refined by
    weighted least squares, each with its uncertainty.

    Inlier bands come from the measured scatter, not a set tolerance: a
    point belongs to a surface when it is within 3 sigma of its own measured
    noise and its normal agrees to the angle that noise allows. A cylinder
    is only a cylinder when its patch is curved past the noise: the sagitta
    of its width, w^2 / (8 r), at least 3 sigma (else it is a plane as far as
    the scan can tell).

    Each feature: parameters, the points it rests on, area (mm^2), and the
    1-sigma and banded uncertainty of its size and position: least squares
    (with the neighbours' correlation), the device's systematic sigma, and
    the scale's (relative, times the feature's size or distance from the
    datum)."""
    rng = np.random.default_rng(seed)
    P, N = scan.points, scan.normals
    n = len(P)
    min_points = min_points or max(40, int(0.001 * n))
    noise = np.maximum(scan.noise, 1e-6 * float(np.ptp(P, 0).max()))
    band = 3.0 * np.sqrt(noise ** 2 + (scan.extra ** 2 if scan.extra is not None else 0.0))
    # normal agreement allowed by the normals' own measured scatter: the
    # median angle between a point's normal and its neighbours' (on a curved
    # patch that includes the curvature, which only widens it), three of
    # those; at least 3 degrees, at most 45
    _, nb8 = scan.tree().query(P, k=9, workers=-1)
    cosn = np.clip(np.abs(np.einsum("ni,nki->nk", N, N[nb8[:, 1:]])), 0, 1)
    th = np.median(np.arccos(cosn), 1)
    th = np.median(np.concatenate([th[:, None], th[nb8[:, 1:]]], 1), 1)
    cosang = np.cos(np.clip(3 * th, math.radians(3), math.radians(45)))
    size = float(np.ptp(P, 0).max())
    kc = (64, 256, 1024)
    free = np.ones(n, bool)
    seedable = np.ones(n, bool)           # a failed candidate's patch seeds no more candidates
    tree = scan.tree()
    feats = []
    z2 = z_value(scan.confidence, two_sided=True)

    def plane_inliers(c, nv, I=None, B=0.0):
        I = slice(None) if I is None else I
        dist = (P[I] - c) @ nv
        return free[I] & (np.abs(dist) <= np.maximum(band[I], B)) & (np.abs(N[I] @ nv) >= cosang[I])

    def cyl_inliers(c, a, r, I=None, B=0.0):
        I = slice(None) if I is None else I
        PI = P[I]
        res = _cyl_residual(PI, c, a, r)
        q = PI - c
        radial = q - np.outer(q @ a, a)
        rad = radial / np.maximum(np.linalg.norm(radial, axis=1, keepdims=True), 1e-12)
        return free[I] & (np.abs(res) <= np.maximum(band[I], B)) & \
            (np.abs(np.einsum("ij,ij->i", N[I], rad)) >= cosang[I])

    fails = 0
    for _ in range(max_features * 3):
        if len(feats) >= max_features or fails >= 6:
            break
        idx_free = np.flatnonzero(free)
        idx_seed = np.flatnonzero(free & seedable)
        if len(idx_free) < min_points or len(idx_seed) < min_points // 4:
            break
        best = None
        # candidates are scored on a random subsample of the free points
        # (Schnabel et al. score on subsets too); the winner is refit on all
        sub = idx_free if len(idx_free) <= 6000 else rng.choice(idx_free, 6000, replace=False)
        need = min_points * len(sub) / len(idx_free)
        for _t in range(tries):
            i = idx_seed[rng.integers(len(idx_seed))]
            cand = []
            if "plane" in kinds:
                cand.append(("plane", P[i], N[i]))
            if "cylinder" in kinds:
                # a cylinder from one point's neighbourhood: its normals lie
                # square to the axis (the smallest eigenvector of their
                # scatter), its points on a circle across it; averaging a
                # neighbourhood beats two noisy normals
                # at several scales: a big radius needs a wide patch to show its
                # curvature above the normals' scatter, a small one a narrow patch
                kk = int(kc[rng.integers(len(kc))])
                _, nb = tree.query(P[i], k=min(kk, n))
                nb = nb[::max(1, len(nb) // 256)]
                Nn = N[nb]
                w_, v_ = np.linalg.eigh(Nn.T @ Nn)
                if w_[1] > 4 * w_[0] + 1e-9 * w_[2]:
                    a = v_[:, 0]
                    e1, e2 = _perp_frame(a)
                    Q = P[nb]
                    x, y = Q @ e1, Q @ e2
                    A = np.stack([x, y, np.ones_like(x)], 1)
                    sol = np.linalg.lstsq(A, x * x + y * y, rcond=None)[0]
                    cx, cy = sol[0] / 2, sol[1] / 2
                    r = math.sqrt(max(sol[2] + cx * cx + cy * cy, 0.0))
                    if 0 < r < 2 * size:
                        cand.append(("cylinder", cx * e1 + cy * e2, a, r))
            for cd in cand:
                if cd[0] == "plane":
                    m = plane_inliers(cd[1], cd[2], sub)
                else:
                    m = cyl_inliers(cd[1], cd[2], cd[3], sub)
                cnt = int(m.sum())
                if cnt >= need and (best is None or cnt > best[0]):
                    best = (cnt, cd)
        if best is None:
            break
        cd = best[1]
        feat = _refine(scan, cd, plane_inliers, cyl_inliers, min_points, z2)
        if os.environ.get("LOADPATH_SCAN_DEBUG"):
            print("candidate", cd[0], best[0], "free", len(idx_free), "seed", len(idx_seed), "->",
                  None if feat is None else (feat["type"], feat["points"]))
        if feat is None:
            # the candidate's patch seeds no more candidates (its points stay
            # free to belong to another surface)
            m = plane_inliers(cd[1], cd[2]) if cd[0] == "plane" else cyl_inliers(cd[1], cd[2], cd[3])
            seedable[_component(scan, m)] = False
            fails += 1
            continue
        fails = 0
        free[feat.pop("_ids")] = False
        feats.append(feat)
    planes = [f for f in feats if f["type"] == "plane"]
    cyls = [f for f in feats if f["type"] == "cylinder"]
    planes.sort(key=lambda f: -f["area"])
    cyls.sort(key=lambda f: -f["area"])
    for i, f in enumerate(planes):
        f["id"] = f"plane-{i}"
    for i, f in enumerate(cyls):
        f["id"] = f"cyl-{i}"
    out = {"units": "mm", "confidence": scan.confidence, "scaled": scan.scaled,
           "covered": round(float(1 - free.mean()), 3), "planes": planes, "cylinders": cyls}
    chk = _self_consistency(cyls)
    if chk:
        out["selfConsistency"] = chk
    return out


def _self_consistency(cyls):
    """Pieces of one surface fitted apart (a bar seen past a clamp, a mug
    either side of its handle) must agree on their radius to their bands.
    When they do not, the scan disagrees with itself by more than its
    stated uncertainty: the device's systematic error is at least the
    difference, and every band from this scan is too tight."""
    rows = []
    for i, a in enumerate(cyls):
        for b in cyls[i + 1:]:
            if a["kind"] != b["kind"]:
                continue
            da, db = np.asarray(a["axis"]), np.asarray(b["axis"])
            if abs(float(da @ db)) < math.cos(math.radians(5)):
                continue
            # the same line: b's axis point within a radius-tenth of a's line
            pa, pb = np.asarray(a["p0"]), np.asarray(b["p0"])
            off = (pb - pa) - ((pb - pa) @ da) * da
            if np.linalg.norm(off) > 0.1 * max(a["r"], b["r"]) + a["rBand"] + b["rBand"]:
                continue
            diff = abs(a["r"] - b["r"])
            allowed = math.hypot(a["rBand"], b["rBand"])
            rows.append({"features": [a["id"], b["id"]], "radiusDifference": round(diff, 4),
                         "allowed": round(allowed, 4), "consistent": bool(diff <= allowed)})
    bad = [r for r in rows if not r["consistent"]]
    if bad:
        worst = max(r["radiusDifference"] for r in bad)
        return {"pairs": rows, "verdict": f"pieces of one surface disagree by up to {worst:.2f} mm, more than "
                f"their bands allow: the scan's systematic error is at least that; give device= its real "
                f"sigma (or a larger one) before trusting a fit"}
    return {"pairs": rows, "verdict": "consistent"} if rows else None


def _refine(scan, cd, plane_inliers, cyl_inliers, min_points, z2):
    P, N = scan.points, scan.normals
    w_all = 1.0 / np.maximum(scan.noise, 1e-9) ** 2
    if cd[0] == "plane":
        c, nv = cd[1], cd[2]
        B = 0.0
        for _ in range(5):
            ids = _component(scan, plane_inliers(c, nv, B=B))
            if len(ids) < min_points:
                return None
            w = w_all[ids] / w_all[ids].mean()
            c = (P[ids] * w[:, None]).sum(0) / w.sum()
            Q = (P[ids] - c) * np.sqrt(w)[:, None]
            _, _, vt = np.linalg.svd(Q, full_matrices=False)
            nv2 = vt[2] * (1 if vt[2] @ nv >= 0 else -1)
            nv = nv2
            # the band is the scatter about the surface itself: a stereo or
            # phone scan's low-frequency waviness is not in the local
            # quadric's scatter, and leaving it out shatters a table into tiles
            B = 3.0 * 1.4826 * float(np.median(np.abs((P[ids] - c) @ nv)))
        ids = _component(scan, plane_inliers(c, nv, B=B))
        if len(ids) < min_points:
            return None
        res = (P[ids] - c) @ nv
        keff = float(scan.k_eff[0])
        n_eff = max(len(ids) * keff / (16 + 1), 4.0)
        rms = float(np.sqrt((res ** 2).mean()))
        s_off = rms / math.sqrt(n_eff)
        e1, e2 = _perp_frame(nv)
        uv = np.stack([(P[ids] - c) @ e1, (P[ids] - c) @ e2], 1)
        spread = np.sqrt(np.maximum(np.var(uv, 0), 1e-12))
        s_tilt = rms / math.sqrt(n_eff) / float(np.mean(spread))       # rad
        f = {"type": "plane", "normal": nv.round(6).tolist(), "center": c.round(4).tolist(),
             "area": round(_area(scan, ids), 2), "points": int(len(ids)),
             "extent": [np.round(uv.min(0), 3).tolist(), np.round(uv.max(0), 3).tolist()],
             "axes": [e1.round(6).tolist(), e2.round(6).tolist()],
             "rms": round(rms, 4)}
        f.update(_bands(scan, c, s_off, 0.0, z2, ids))
        f["tiltSigmaDeg"] = round(math.degrees(s_tilt), 4)
        if not scan.scaled:
            f["note"] = "unscaled: positions in the file's units x the units factor only"
        # outward normal agrees with the points' normals on balance
        if (N[ids] @ nv).mean() < 0:
            f["normal"] = (-nv).round(6).tolist()
        f["_ids"] = ids
        return f
    c, a, r = cd[1], cd[2], cd[3]
    B = 0.0
    for _ in range(5):
        ids = _component(scan, cyl_inliers(c, a, r, B=B))
        if len(ids) < min_points:
            return None
        w = w_all[ids] / w_all[ids].mean()
        c, a, r, J, res = _refine_cylinder(P[ids], w, c, a, r)
        B = 3.0 * 1.4826 * float(np.median(np.abs(res)))
    ids = _component(scan, cyl_inliers(c, a, r, B=B))
    if len(ids) < min_points:
        return None
    w = w_all[ids] / w_all[ids].mean()
    c, a, r, J, res = _refine_cylinder(P[ids], w, c, a, r)
    q = P[ids] - c
    t = q @ a
    e1, e2 = _perp_frame(a)
    ang = np.sort(np.arctan2(q @ e2, q @ e1))
    gaps = np.diff(np.concatenate([ang, ang[:1] + 2 * math.pi]))
    span = float(2 * math.pi - gaps.max())
    # width of the patch across the axis (the chord), and the sagitta test
    chord = 2 * r * math.sin(min(span, math.pi) / 2)
    sag = chord ** 2 / (8 * r)
    noise_med = max(float(np.median(scan.noise[ids])), 1.4826 * float(np.median(np.abs(res))))
    if sag < 3 * noise_med:
        return None                            # flat as far as this scan can tell
    rms = float(np.sqrt((res ** 2).mean()))
    keff = float(scan.k_eff[0])
    n_eff = max(len(ids) * keff / (16 + 1), 6.0)
    C = _cov(J, res * np.sqrt(w), n_eff, 5)
    s_r = float(math.sqrt(max(C[4, 4], 0.0)))
    s_axis = float(math.sqrt(max(C[0, 0] + C[1, 1], 0.0)))
    s_pos = float(math.sqrt(max(C[2, 2] + C[3, 3], 0.0)))
    # concave (a bore: the normals point at the axis) or convex (a bar, a boss)
    radial = q - np.outer(t, a)
    radial /= np.maximum(np.linalg.norm(radial, axis=1, keepdims=True), 1e-12)
    concave = float(np.einsum("ij,ij->i", N[ids], radial).mean()) < 0
    p0 = c + t.min() * a
    p1 = c + t.max() * a
    f = {"type": "cylinder", "kind": "bore" if concave else "bar", "axis": a.round(6).tolist(),
         "p0": p0.round(4).tolist(), "p1": p1.round(4).tolist(), "length": round(float(np.ptp(t)), 3),
         "r": round(r, 4), "diameter": round(2 * r, 4), "spanDeg": round(math.degrees(span), 1),
         "area": round(_area(scan, ids), 2), "points": int(len(ids)), "rms": round(rms, 4),
         "sagitta": round(sag, 4), "axisSigmaDeg": round(math.degrees(s_axis), 4),
         "axisPositionSigma": round(s_pos, 4)}
    f.update(_bands(scan, 0.5 * (p0 + p1), s_r, r, z2, ids, diameter=True))
    if span < math.radians(120):
        f["note"] = (f"only {math.degrees(span):.0f} degrees of it seen: the radius rests on a short arc "
                     "(its band says how much that costs)")
    f["_ids"] = ids
    return f


def _bands(scan, centre, s_fit, size, z2, ids, diameter=False):
    """A feature's 1-sigma parts and its two-sided band: the fit, the
    device's systematic sigma (a bias along the normals moves a radius by
    the full bias), the scale (relative, times the size; and times the
    distance from the datum for a position)."""
    dev = float(scan.device.get("sigma") or 0.0)
    ss = scan.scale_sigma
    s_scale = (ss or 0.0) * size
    dat = scan.datum if scan.datum is not None else scan.points.mean(0)
    s_pos_scale = (ss or 0.0) * float(np.linalg.norm(np.asarray(centre) - dat))
    if diameter:
        tot = math.sqrt(s_fit ** 2 + dev ** 2 + s_scale ** 2)
        out = {"sigma": {"fit": round(s_fit, 5), "device": round(dev, 5), "scale": round(s_scale, 5),
                         "total": round(tot, 5)},
               "rBand": round(z2 * tot, 4), "diameterBand": round(2 * z2 * tot, 4)}
    else:
        tot = math.sqrt(s_fit ** 2 + dev ** 2 + s_pos_scale ** 2)
        out = {"sigma": {"fit": round(s_fit, 5), "device": round(dev, 5), "scale": round(s_pos_scale, 5),
                         "total": round(tot, 5)},
               "offsetBand": round(z2 * tot, 4)}
    if ss is None:
        out["unscaled"] = True
    return out


# ---------------------------------------------------------------- scale

def apply_reference(scan: Scan, reference, feats: dict | None = None) -> Scan:
    """Fix the scale from known dimensions. Each reference:

      {"feature": "cyl-0", "diameter": 31.8, "tolerance": 0.05}
      {"feature": "plane-0", "to": "plane-1", "distance": 40.0, "tolerance": 0.05}
      {"points": [[x, y, z], [x, y, z]], "distance": 85.6, "tolerance": 0.05}

    in mm (known) and the scan's current coordinates (features are fitted
    first when not given). tolerance is +- (read as 2 sigma; default the
    last digit given). The scale is the weighted mean of known / measured;
    its relative sigma combines each measurement's sigma and each known
    dimension's; with two or more, their scatter is checked against that
    (chi^2) and the larger of the two is kept. The datum (where the scale's
    error is zero) is the references' centre."""
    refs = reference if isinstance(reference, list) else [reference]
    if feats is None and any("feature" in r for r in refs):
        # fit with the scatter alone: the device's systematic error is not part of a measured ratio
        feats = fit_primitives(scan)
    byid = {f["id"]: f for f in (feats or {}).get("planes", []) + (feats or {}).get("cylinders", [])}
    vals, sig, cens, rows = [], [], [], []
    for r in refs:
        tol = r.get("tolerance")
        if "feature" in r:
            f = byid.get(r["feature"])
            if f is None:
                raise ValueError(f"reference: no feature {r['feature']!r} (have {sorted(byid)})")
            if "diameter" in r:
                known = float(r["diameter"])
                meas = 2 * f["r"]
                s_meas = 2 * f["sigma"]["fit"]
                cen = 0.5 * (np.asarray(f["p0"]) + np.asarray(f["p1"]))
            elif "to" in r:
                g = byid[r["to"]]
                n1 = np.asarray(f["normal"])
                known = float(r["distance"])
                meas = abs(float((np.asarray(g["center"]) - np.asarray(f["center"])) @ n1))
                s_meas = math.hypot(f["sigma"]["fit"], g["sigma"]["fit"])
                cen = 0.5 * (np.asarray(f["center"]) + np.asarray(g["center"]))
            else:
                raise ValueError("a feature reference gives diameter, or to + distance")
        elif "points" in r:
            a, b = (np.asarray(p, float) for p in r["points"])
            known = float(r["distance"])
            meas = float(np.linalg.norm(b - a))
            # a picked point sits on the measured surface to its own scatter
            ia = scan.nearest([a, b])[1]
            s_meas = float(np.sqrt((scan.noise[ia] ** 2).sum()))
            cen = 0.5 * (a + b)
        else:
            raise ValueError("a reference names a feature or two points")
        if tol is None:
            txt = repr(r.get("diameter", r.get("distance")))
            dec = len(txt.split(".")[1]) if "." in txt else 0
            tol = 0.5 * 10 ** (-dec)
        s_known = float(tol) / 2.0
        s = known / meas
        rel = math.hypot(s_meas / meas, s_known / known)
        vals.append(s)
        sig.append(rel * s)
        cens.append(cen)
        rows.append({"known": known, "measured": round(meas, 5), "scale": round(s, 6),
                     "relSigma": round(rel, 6), **{k: v for k, v in r.items() if k in ("feature", "to")}})
    v, sg = np.array(vals), np.array(sig)
    w = 1 / sg ** 2
    s = float((w * v).sum() / w.sum())
    s_sig = float(1 / math.sqrt(w.sum()))
    chi = None
    if len(v) > 1:
        chi = float((((v - s) / sg) ** 2).sum() / (len(v) - 1))
        s_sig *= math.sqrt(max(chi, 1.0))
    dev_scale = float(scan.device.get("scale") or 0.0) if scan.device.get("scale") is not None else 0.0
    datum = np.mean(cens, 0)
    out = scan.transformed(s, None, datum)
    # what remains of the scale's error: the reference's (the device's own
    # scale error is what the reference replaces)
    out.scale_sigma = s_sig / s
    # the report's lengths were in the file's units: now in mm
    for key in ("spacing", "noise"):
        for k2 in ("median", "p95", "splatThicknessMedian"):
            if isinstance(out.report.get(key, {}).get(k2), (int, float)):
                out.report[key] = {**out.report[key], k2: round(out.report[key][k2] * s, 4)}
    if "bounds" in out.report:
        out.report["bounds"] = (np.asarray(out.report["bounds"]) * s).round(3).tolist()
    out.report["scale"] = {"factor": round(s, 6), "relSigma": round(s_sig / s, 6), "references": rows,
                           "datum": (datum * s).round(3).tolist(),
                           "replacesDeviceScale": dev_scale or None}
    if chi is not None:
        out.report["scale"]["chi2PerDof"] = round(chi, 3)
        if chi > 4:
            out.report["todo"] = out.report.get("todo", []) + [
                f"the references disagree (chi^2/dof {chi:.1f}): one is misread, or the scan is distorted"]
    out.report["todo"] = [t for t in out.report.get("todo", []) if not t.startswith("the scale is not known")]
    _finish_report(out)
    return out


# ---------------------------------------------------------------- surface

def reconstruct(scan: Scan, pitch: float | None = None, budget: int = 6_000_000, k: int = 8):
    """A watertight surface of the scanned object.

    Near the points it is the zero set of an implicit moving least squares
    field (Kolluri 2008): at x, the Gaussian-weighted mean of (x - p) . n over
    the nearest points, kernel width their spacing, so the noise is averaged
    the way the uncertainty assumed. Inside and outside away from the points
    come from the points' generalized winding number (Barill et al. 2018,
    libigl; a direct sum on a subsample without it), which closes what the
    scan did not see without inventing a surface there: those patches are
    flagged.

    Returns (mesh, info): info["U"] per vertex (mm, one-sided at the scan's
    confidence: the nearest point's U plus, on unseen patches, the distance
    to the nearest point), info["unseen"] per vertex, and a report."""
    import trimesh
    from skimage.measure import marching_cubes
    P, N = scan.points, scan.normals
    lo, hi = P.min(0), P.max(0)
    h_med = float(np.median(scan.spacing))
    pad = 4 * h_med
    lo, hi = lo - pad, hi + pad
    vol = float(np.prod(hi - lo))
    p = float(pitch) if pitch else max(h_med, (vol / budget) ** (1 / 3))
    dims = np.ceil((hi - lo) / p).astype(int) + 1
    from scipy import ndimage
    # nodes near the points: a raster of the points, dilated by the band
    occ = np.zeros(dims, bool)
    ijk = np.clip(np.round((P - lo) / p).astype(int), 0, dims - 1)
    occ[ijk[:, 0], ijk[:, 1], ijk[:, 2]] = True
    band_cells = max(2, int(math.ceil(3 * float(np.percentile(scan.spacing, 90)) / p)))
    dist_cells = ndimage.distance_transform_edt(~occ)
    near = dist_cells <= band_cells
    G = lo + p * np.argwhere(near)
    tree = scan.tree()
    d, j = tree.query(G, k=k, workers=-1)
    hj = scan.spacing[j]
    sw = 1.5 * np.maximum(hj[:, :1], p)
    wts = np.exp(-(d / sw) ** 2)
    f = (np.einsum("nki,nki->nk", G[:, None, :] - P[j], N[j]) * wts).sum(1) / np.maximum(wts.sum(1), 1e-300)
    F = np.empty(dims)
    F[near] = f
    # far from the points: inside/outside by the winding number, on a coarse
    # lattice (it is smooth there) and spread to the cells
    st = 3
    coarse = np.argwhere(np.ones(tuple((dims + st - 1) // st), bool)) * st
    wn = _winding(scan, lo + p * coarse).reshape(tuple((dims + st - 1) // st))
    sign = np.where(wn > 0.5, -1.0, 1.0)
    sign = np.repeat(np.repeat(np.repeat(sign, st, 0), st, 1), st, 2)[:dims[0], :dims[1], :dims[2]]
    F[~near] = sign[~near] * band_cells * p
    verts, faces, _, _ = marching_cubes(F, 0.0, spacing=(p, p, p))
    verts = verts + lo
    mesh = trimesh.Trimesh(verts, faces, process=True)
    # every closed body bigger than a speck (a scene's separate objects are all real)
    parts = [m for m in mesh.split(only_watertight=False)]
    if len(parts) > 1:
        big = max(m.area for m in parts)
        parts = [m for m in parts if m.area >= 0.002 * big]
        mesh = trimesh.util.concatenate(parts)
    if mesh.volume < 0:
        mesh.invert()
    dv, iv = scan.nearest(mesh.vertices)
    unseen = dv > 2 * np.maximum(scan.spacing[iv], p)
    U = scan.uncertainty(idx=iv) if scan.scaled else np.full(len(iv), np.nan)
    U = U + np.where(unseen, dv, 0.0)
    a_v = np.zeros(len(mesh.vertices))
    np.add.at(a_v, mesh.faces.ravel(), np.repeat(mesh.area_faces / 3, 3))
    info = {"U": U, "unseen": unseen,
            "report": {"pitch": round(p, 4), "vertices": int(len(mesh.vertices)), "watertight": bool(mesh.is_watertight),
                       "volume": round(float(mesh.volume), 2), "area": round(float(mesh.area), 2),
                       "unseenArea": round(float(a_v[unseen].sum()), 2),
                       "unseenShare": round(float(a_v[unseen].sum() / max(mesh.area, 1e-9)), 4),
                       "unseenWorstGap": round(float(dv[unseen].max()) if unseen.any() else 0.0, 3),
                       "surfaceToPoints": {"p50": round(float(np.median(dv[~unseen])), 4) if (~unseen).any() else None,
                                           "p95": round(float(np.percentile(dv[~unseen], 95)), 4) if (~unseen).any() else None}}}
    if scan.scaled:
        info["report"]["U"] = {"median": round(float(np.nanmedian(U)), 4), "p95": round(float(np.nanpercentile(U, 95)), 4),
                               "max": round(float(np.nanmax(U)), 4)}
    return mesh, info


def _winding(scan: Scan, Q):
    if len(Q) == 0:
        return np.zeros(0)
    P, N = scan.points, scan.normals
    A = scan.spacing ** 2
    try:
        import igl
        fn = getattr(igl, "fast_winding_number_for_points", None) or getattr(igl, "fast_winding_number", None)
        if fn is not None:
            return np.asarray(fn(P.astype(np.float64), N.astype(np.float64), A.astype(np.float64),
                                 np.asarray(Q, np.float64)), float).ravel()
    except Exception:
        pass
    # direct sum over a subsample (area-weighted dipoles)
    rng = np.random.default_rng(0)
    m = min(len(P), 4000)
    sel = rng.choice(len(P), m, replace=False)
    Ps, Ns, As = P[sel], N[sel], A[sel] * (len(P) / m)
    out = np.empty(len(Q))
    for s in range(0, len(Q), 20000):
        q = Q[s:s + 20000]
        D = Ps[None] - q[:, None]
        r3 = np.maximum(np.linalg.norm(D, axis=2), 1e-9) ** 3
        out[s:s + 20000] = (np.einsum("qpi,pi->qp", D, Ns) * As / r3).sum(1) / (4 * math.pi)
    return out


def summarize(scan: Scan, feats: dict | None = None) -> dict:
    """The short read: the scan's measured numbers and each feature in a line."""
    rows = []
    for f in (feats or {}).get("cylinders", []):
        what = (f"{f['kind']} d {f['diameter']:.3f} +- {f['diameterBand']:.3f} mm, {f['length']:.1f} mm long, "
                f"{f['spanDeg']:.0f} deg seen")
        rows.append({"id": f["id"], "what": what})
    for f in (feats or {}).get("planes", []):
        rows.append({"id": f["id"], "what": f"plane {f['area']:.0f} mm^2, normal "
                     f"{np.round(f['normal'], 3).tolist()}, offset +- {f['offsetBand']:.3f} mm"})
    r = scan.report
    return {"points": r.get("points"), "spacing": r.get("spacing", {}).get("median"),
            "noise": r.get("noise", {}).get("median"), "device": r.get("device", {}).get("name"),
            "scaled": scan.scaled, "uncertainty": r.get("uncertainty"), "features": rows, "todo": r.get("todo", [])}


def scale_features(feats: dict, scan: Scan) -> dict:
    """Features fitted before a reference fixed the scale, carried to the
    scaled scan (same ids, every length times the factor, bands recomputed
    with the scale's sigma and the device's): refitting could renumber them."""
    import copy
    s = float(scan.report.get("scale", {}).get("factor", 1.0))
    out = copy.deepcopy(feats)
    z2 = z_value(scan.confidence, two_sided=True)
    for f in out.get("planes", []):
        f["center"] = (np.asarray(f["center"]) * s).round(4).tolist()
        f["extent"] = (np.asarray(f["extent"]) * s).round(3).tolist()
        f["area"] = round(f["area"] * s * s, 2)
        f["rms"] = round(f["rms"] * s, 4)
        fit = f["sigma"]["fit"] * s
        f.pop("unscaled", None)
        f.pop("note", None)
        f.update(_bands(scan, np.asarray(f["center"]), fit, 0.0, z2, None))
    for f in out.get("cylinders", []):
        for k in ("p0", "p1"):
            f[k] = (np.asarray(f[k]) * s).round(4).tolist()
        for k in ("length", "r", "diameter", "rms", "sagitta", "axisPositionSigma"):
            f[k] = round(f[k] * s, 4)
        f["area"] = round(f["area"] * s * s, 2)
        fit = f["sigma"]["fit"] * s
        f.pop("unscaled", None)
        f.update(_bands(scan, 0.5 * (np.asarray(f["p0"]) + np.asarray(f["p1"])), fit, f["r"], z2, None,
                        diameter=True))
    out["scaled"] = scan.scaled
    chk = _self_consistency(out.get("cylinders", []))
    if chk:
        out["selfConsistency"] = chk
    return out


def calibrate_device(scan: Scan, feats: dict, known, confidence: float = 0.95) -> dict:
    """A device's systematic sigma measured on a known artefact: scan
    something with caliper-measured diameters or face-to-face distances (a
    printed step block, a can, a gauge block), fit it, and compare.

      known = [{"feature": "cyl-0", "diameter": 66.2, "tolerance": 0.05},
               {"feature": "plane-0", "to": "plane-1", "distance": 40.0}]

    Each comparison's error, less what its fit and the caliper explain, is
    a draw of the device's bias (a diameter carries it twice, a distance
    between two faces about sqrt 2 times). The sigma returned is the upper
    `confidence` bound from those draws (chi-squared on n of them), so three
    references give a cautious number and ten a tight one. Pass it back as
    device={"base": ..., "sigma": result["sigma"]}."""
    from scipy.stats import chi2
    byid = {f["id"]: f for f in feats.get("planes", []) + feats.get("cylinders", [])}
    rows, draws = [], []
    for r in known:
        f = byid[r["feature"]]
        tol = float(r.get("tolerance", 0.05)) / 2
        if "diameter" in r:
            err = f["diameter"] - float(r["diameter"])
            expl = math.hypot(2 * f["sigma"]["fit"], tol)
            per = 2.0
        else:
            g = byid[r["to"]]
            n1 = np.asarray(f["normal"])
            meas = abs(float((np.asarray(g["center"]) - np.asarray(f["center"])) @ n1))
            err = meas - float(r["distance"])
            expl = math.sqrt(f["sigma"]["fit"] ** 2 + g["sigma"]["fit"] ** 2 + tol ** 2)
            per = math.sqrt(2.0)
        b2 = max(err * err - expl * expl, 0.0) / (per * per)
        draws.append(b2)
        rows.append({**{k: v for k, v in r.items()}, "error": round(err, 4), "explained": round(expl, 4)})
    n = len(draws)
    if n == 0:
        raise ValueError("calibrate_device needs at least one known dimension")
    s2 = float(np.mean(draws))
    upper = math.sqrt(n * s2 / chi2.ppf(1 - confidence, n)) if s2 > 0 else 0.0
    return {"sigma": round(upper, 4), "sigmaEstimate": round(math.sqrt(s2), 4), "n": n,
            "confidence": confidence, "rows": rows,
            "note": "upper bound on the systematic sigma from n comparisons; more references tighten it"}
