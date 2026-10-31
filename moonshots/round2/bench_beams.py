"""
Benchmark: slender strips 1, 2, 3, 5 cells thick, at 0, 30, 45 degrees to
the voxel grid, cantilevered, under a tip shear (bending) and a tip pull
(tension). Peak von Mises in a window away from the clamp and the load is
compared with beam theory (sigma = 6 F (sF - s) / (w t^2), or F / (w t)),
pointwise: the reported number is max over the window of vm / sigma_beam,
which is what a peak-stress reading over- or under-states by. The P2 tet
solve (crosscheck.py's) on the same strip is the second reference.

Run:  python moonshots/round2/bench_beams.py [--quick] [--out results.json]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from cutcell import CutModel, SubcellRule, centre_inside, fill_from_inside  # noqa: E402

from loadpath.fem import von_mises  # noqa: E402

E, NU, F = 2000.0, 0.35, 1.0
H = 1.0
OFFSET = np.array([0.37, 0.21, 0.0]) * H
W = 3.0 * H
TAU = 0.02          # drop cut cells under this fill


class Strip:
    def __init__(self, t, theta_deg, h=H):
        self.t, self.h = t, h
        self.L = max(12.0 * t, 24.0 * h)
        self.c = 2.0 * h
        th = np.radians(theta_deg)
        self.a = np.array([np.cos(th), np.sin(th), 0.0])
        self.n = np.array([-np.sin(th), np.cos(th), 0.0])
        self.p0 = OFFSET

    def local(self, p):
        d = np.asarray(p, float) - self.p0
        return d @ self.a, d @ self.n, d[..., 2]

    def inside(self, p):
        s, y, z = self.local(p)
        return (s >= -self.c) & (s <= self.L) & (np.abs(y) <= self.t / 2) & (z >= 0) & (z <= W)

    def corners(self):
        out = []
        for s in (-self.c, self.L):
            for y in (-self.t / 2, self.t / 2):
                for z in (0.0, W):
                    out.append(self.p0 + s * self.a + y * self.n + np.array([0, 0, z]))
        return np.array(out)

    def trimesh(self):
        import trimesh
        box = trimesh.creation.box(extents=[self.L + self.c, self.t, W])
        m = np.eye(4)
        m[:3, 0], m[:3, 1], m[:3, 2] = self.a, self.n, [0, 0, 1]
        m[:3, 3] = self.p0 + (0.5 * (self.L - self.c)) * self.a + np.array([0, 0, W / 2])
        box.apply_transform(m)
        return box

    def surface_points(self, lo, hi, ds):
        s = np.arange(lo, hi + 1e-9, ds)
        pts = []
        for y in np.linspace(-self.t / 2, self.t / 2, max(3, int(round(self.t / ds)) + 1)):
            for z in (0.0, W):
                pts.append(np.stack([s, np.full_like(s, y), np.full_like(s, z)], 1))
        for z in np.linspace(0, W, max(3, int(round(W / ds)) + 1)):
            for y in (-self.t / 2, self.t / 2):
                pts.append(np.stack([s, np.full_like(s, y), np.full_like(s, z)], 1))
        loc = np.concatenate(pts)
        return self.p0 + loc[:, :1] * self.a + loc[:, 1:2] * self.n + loc[:, 2:3] * np.array([0, 0, 1.0])


def beam_sigma(strip, load, s, s_f):
    if load == "tension":
        return np.full_like(s, F / (W * strip.t))
    return 6.0 * F * (s_f - s) / (W * strip.t ** 2)


def load_dir(strip, load):
    return strip.a if load == "tension" else strip.n


def end_load(strip, model, rule, d):
    """The end load as a uniform body force over the material of the end
    zone (s > L - 1.5 h), consistent nodal weights from each inside subcell.
    Returns (f, s of the resultant, loaded node mask)."""
    m = model.mesh
    sc = (rule.centers + 1.0) / 2.0                                  # (n3, 3) in [0,1]
    xyz = m.origin + (m.elem_ijk[:, None, :] + sc[None]) * m.pitch   # (e, n3, 3)
    s = strip.local(xyz)[0]
    w = model.ind * (s > strip.L - 1.5 * strip.h)                     # (e, n3)
    from loadpath.fem import CORNERS
    sgn = 2.0 * CORNERS - 1.0
    nsh = 0.125 * np.prod(1 + rule.centers[:, None, :] * sgn[None], axis=2)  # (n3, 8)
    we = w @ nsh                                                      # (e, 8)
    wn = np.zeros(m.n_nodes)
    np.add.at(wn, m.enode.ravel(), we.ravel())
    tot = w.sum()
    f = (F * wn[:, None] / tot * d[None]).ravel()
    s_f = float((s * w).sum() / tot)
    return f, s_f, wn > 0


def voxel_runs(strip, load, rule, combos, h=H, tag=""):
    cs = strip.corners()
    origin = np.floor(cs.min(0) / h) * h - h
    origin[2] = 0.0
    dims = np.ceil((cs.max(0) - origin) / h).astype(int) + 1
    dims[2] = int(round(W / h))
    t0 = time.time()
    fill, ind = fill_from_inside(strip.inside, origin, dims, h, n=rule.n)
    ctr = centre_inside(strip.inside, origin, dims, h)
    t_fill = time.time() - t0
    flat_fill = fill.ravel()
    out = {}
    lo, hi = 0.3 * strip.L, 0.7 * strip.L
    surf = strip.surface_points(lo, hi, H / 4)
    s_surf = strip.local(surf)[0]
    for geom, elem in combos:
        t0 = time.time()
        sel = ctr.ravel() if geom == "stair" else flat_fill >= TAU
        ijk = np.argwhere(sel.reshape(dims))
        model = CutModel(ijk, h, origin, NU, E, ind=ind[sel], geometry=geom, element=elem, rule=rule)
        m = model.mesh
        s_node = strip.local(m.node_xyz)[0]
        fixed = np.repeat(s_node < 0, 3)
        d = load_dir(strip, load)
        f, s_f, loaded = end_load(strip, model, rule, d)
        u = model.solve(f, fixed)
        defl = float(f @ u) / F      # work-conjugate end displacement
        t_solve = time.time() - t0
        ns = model.nodal_stress(weight_by_fill=(geom != "stair"))
        vm_n = von_mises(ns)
        win = (s_node >= lo) & (s_node <= hi)
        r_nodal = float(np.max(vm_n[win] / beam_sigma(strip, load, s_node[win], s_f)))
        vm_si = von_mises(model.nodal_interp(surf, ns))
        vm_sd = von_mises(model.surface_stress(surf))
        extra = {}
        for mf in (0.25, 0.5):
            for reach in (1.5, 2.0):
                v = von_mises(model.surface_stress(surf, min_fill=mf, reach=reach))
                extra[f"surfX{mf}_{reach}"] = float(np.nanmax(v / beam_sigma(strip, load, s_surf, s_f)))
        sig = beam_sigma(strip, load, s_surf, s_f)
        out[f"{geom}/{elem}{tag}"] = {
            "nodal": r_nodal,
            "surfN": float(np.nanmax(vm_si / sig)),
            "surfD": float(np.nanmax(vm_sd / sig)),
            **extra,
            "defl": defl, "cells": int(m.n_elem), "seconds": t_solve,
        }
    out["fill_seconds"] = t_fill
    return out


def tet_run(strip, load):
    import pytetwild

    from loadpath.crosscheck import _boundary_nodes, _nodal_von_mises, _p2_mesh, _solve

    mesh = strip.trimesh()
    size = min(strip.t, W) / 3.0
    t0 = time.time()
    cwd = os.getcwd()
    with tempfile.TemporaryDirectory() as tmp:
        os.chdir(tmp)
        try:
            pts, tets = pytetwild.tetrahedralize(np.asarray(mesh.vertices, float), np.asarray(mesh.faces, np.int32),
                                                 num_threads=1, edge_length_abs=float(size), epsilon=1e-3,
                                                 stop_energy=50.0, num_opt_iter=20)
        finally:
            os.chdir(cwd)
    pts, tets = np.asarray(pts, float), np.asarray(tets, np.int64)
    x = pts[tets]
    neg = np.linalg.det(x[:, 1:] - x[:, :1]) < 0
    tets[neg] = tets[neg][:, [0, 2, 1, 3]]
    nodes, conn, edges = _p2_mesh(pts, tets)
    s_all = strip.local(nodes)[0]
    fixed = np.repeat(s_all < 0, 3)
    d = load_dir(strip, load)
    # the end load as a uniform body force over the end zone, consistent
    # P2 nodal weights (vertices -V/20, edge nodes V/5): equal forces per
    # node put the resultant off the material centroid (a bending moment)
    xc = pts[tets].mean(1)
    zone = strip.local(xc)[0] > strip.L - 1.5 * strip.h
    vol = np.abs(np.linalg.det(pts[tets[:, 1:]] - pts[tets[:, :1]])) / 6.0
    wv = np.zeros(len(nodes))
    np.add.at(wv, conn[zone, :4].ravel(), np.repeat(-vol[zone] / 20.0, 4))
    np.add.at(wv, conn[zone, 4:].ravel(), np.repeat(vol[zone] / 5.0, 6))
    fv = F * wv[:, None] / vol[zone].sum() * d[None]
    s_f = float((strip.local(xc[zone])[0] * vol[zone]).sum() / vol[zone].sum())
    loaded = np.unique(conn[zone].ravel())
    u = _solve(nodes, conn, E, NU, fixed, None, [("c", fv, loaded)], len(pts), edges)[0]
    vm = _nodal_von_mises(nodes, conn, u.ravel(), E, NU, len(pts))
    surf = _boundary_nodes(tets)
    s_v = strip.local(pts[surf])[0]
    lo, hi = 0.3 * strip.L, 0.7 * strip.L
    win = (s_v >= lo) & (s_v <= hi)
    ratio = vm[surf][win] / beam_sigma(strip, load, s_v[win], s_f)
    r = float(np.max(ratio))
    p99 = float(np.percentile(ratio, 99))
    defl = float((fv * u).sum()) / F
    return {"peak": r, "p99": p99, "defl": defl, "tets": int(len(tets)), "seconds": time.time() - t0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--no-tet", action="store_true")
    ap.add_argument("--tet-from", help="reuse the tet results of an earlier results file")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "results_beams.json"))
    a = ap.parse_args()
    rule = SubcellRule(NU, 4)
    combos = [("stair", "std"), ("stair", "im"), ("cutS", "std"), ("cutQ", "std"), ("cutS", "im"), ("cutQ", "im")]
    thick = [1, 2, 3, 5] if not a.quick else [1, 3]
    angles = [0, 30, 45] if not a.quick else [0, 30]
    rows = []
    prev = {}
    if a.tet_from:
        for r in json.load(open(a.tet_from)):
            if "tet" in r:
                prev[(r["load"], r["t"], r["theta"])] = r["tet"]
    for load in ("bending", "tension"):
        for t in thick:
            for th in angles:
                strip = Strip(t * H, th)
                r = {"load": load, "t": t, "theta": th, "voxel": voxel_runs(strip, load, rule, combos)}
                # verify.py re-solves hotspots on a 2.5x finer submodel: the
                # same grid globally stands in for it here
                r["voxel"].update(voxel_runs(strip, load, rule, [("stair", "std"), ("stair", "im"), ("cutQ", "im")],
                                             h=H / 2.5, tag="@h/2.5"))
                if (load, t, th) in prev:
                    r["tet"] = prev[(load, t, th)]
                elif not a.no_tet:
                    r["tet"] = tet_run(strip, load)
                rows.append(r)
                v = r["voxel"]
                tet = r.get("tet", {}).get("peak", float("nan"))
                print(f"{load:8s} t={t} th={th:2d} tet={tet:.3f}/{r.get('tet', {}).get('p99', float('nan')):.3f} " + " ".join(
                    f"{k}:{v[k]['nodal']:.2f}/{v[k]['surfD']:.2f}" for k, _ in [(f'{g}/{e}', 0) for g, e in combos]),
                    flush=True)
    with open(a.out, "w") as fh:
        json.dump(rows, fh, indent=1)


if __name__ == "__main__":
    main()
