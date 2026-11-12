"""
External flow: a part in a stream of air or water, as a wind tunnel would
put it (windtunnel.py: lattice Boltzmann on nested grids, walls at the
true surface), and what a part costs a rotor by sitting in its stream.

`external_flow` puts the part in a tunnel whose far boundaries are ten and
more body lengths away, brings the stream up, and measures the force on
the part by momentum exchange on every wall link, time-averaged once the
wake has developed. It returns the drag, side force and lift (N) and their
coefficients on the frontal area, the moment about a point, the shedding
frequency, and `load_case`: the surface forces as a study load case of
patches, so a wind or water load goes straight into a structural study (a
drone arm, a fairing, a bracket on a vehicle, a mount in a current).

How far to trust it (docs/EXTERNAL-FLOW.md has the table against published
drag): where the grid holds the boundary layer, about four cells in a
layer 3 d / sqrt(Re) thick, spheres and cylinders to Re 300 read within
1 to 3%. N cells across hold Re to about (0.75 N)^2; a faster stream is
run at the Reynolds number the grid resolves, with the physical dynamic
pressure, and the report says so (`reynoldsRun`, `reScale`, `note`): 8%
on a cube, 16% low on a long square strut, the drag of the lower Reynolds
number on a round body. The drag crisis and a streamlined section's
attached turbulent boundary layer are out of reach.
"""
from __future__ import annotations

import time

import numpy as np


__all__ = ["external_flow", "boundary_layer_cells", "rotor_blockage", "RotorStream", "BlockageCost", "duct_performance",
           "inlet_loss"]


def _rotation_to_x(direction):
    """Rotation taking unit `direction` to +x (rows: the new axes)."""
    d = np.asarray(direction, float)
    d = d / np.linalg.norm(d)
    a = np.array([0.0, 0.0, 1.0]) if abs(d[2]) < 0.9 else np.array([0.0, 1.0, 0.0])
    y = np.cross(a, d)
    y /= np.linalg.norm(y)
    z = np.cross(d, y)
    return np.stack([d, y, z])


def _frame(direction, span=None):
    """Rotation to the tunnel's frame (rows: its axes in the part's): x
    along the stream, and z along `span` for a body that runs the width of
    the tunnel."""
    d = np.asarray(direction, float)
    d = d / np.linalg.norm(d)
    if span is None:
        return _rotation_to_x(d)
    z = np.asarray(span, float)
    z = z - (z @ d) * d
    if np.linalg.norm(z) < 1e-6 * max(np.linalg.norm(span), 1e-30):
        raise ValueError("span is along the stream: a spanning body lies across it")
    z /= np.linalg.norm(z)
    return np.stack([d, np.cross(z, d), z])


BL_THICKNESS = 3.0     # laminar boundary layer where it leaves a bluff body: about 3 d / sqrt(Re) (Schlichting)


def boundary_layer_cells(reynolds: float, cells: float) -> float:
    """Cells across the laminar boundary layer where it separates from a
    bluff body `cells` across at Reynolds number `reynolds`: its thickness
    there is about 3 d / sqrt(Re) (1.2 d / sqrt(Re) at the stagnation
    point of a cylinder, grown two- to three-fold by the shoulder)."""
    return BL_THICKNESS * cells / np.sqrt(max(reynolds, 1e-12))


def external_flow(mesh, speed: float, direction=(1, 0, 0), *, medium="air", cells: int = 24, layers: int | None = None,
                  max_layers: int = 1, span=None, rotors=None, about=None, re_max: float | None = None,
                  bl_cells: float = 4.0, resolve: bool | None = None, domain=None, fine=None, shell: int = 6,
                  u_lattice: float = 0.1, sgs: str = "wale", develop: float = 10.0, window: float = 8.0,
                  tolerance: float = 0.01, max_time: float = 100.0, patches: int = 24, interpolated: bool = True,
                  verbose: bool = False) -> dict:
    """Time-averaged flow round `mesh` (mm) in a stream of `speed` m/s along
    `direction` (the way the fluid moves), in a virtual wind tunnel
    (windtunnel.py: nested lattice Boltzmann grids, walls at the surface).
    medium: a heatx_thermal fluid name ("air", "water", ...) or Fluid.

    Resolution. `cells` is the voxels across the body's thickness to the
    stream (its smaller frontal extent) on the finest box of the grid;
    `layers` more levels, each half the cell, hug the surface. A laminar
    boundary layer is about 3 d / sqrt(Re) thick where it leaves the body,
    and where it leaves sets the drag, so the grid resolves a Reynolds
    number of about (3 N / bl_cells)^2 on N cells across (bl_cells cells
    in the layer). With `layers` None, layers are added until that reaches
    the physical Reynolds number, up to `max_layers`; past it the run is
    made at the Reynolds number the grid resolves, with the physical
    dynamic pressure, and the report says so (`reynoldsRun`, `reScale`,
    `note`). Between Re 1e3 and 2e5 a bluff body's coefficient barely
    moves, so that reading carries; a drag crisis (smooth bodies past Re
    2e5) is not in it.

    span: a direction (part frame) along which the body runs the whole
    width of the tunnel, periodic: a slice of a long strut or cylinder.
    The tunnel is as wide as the mesh is long that way.

    rotors: [{center, axis, radius, thrust, hub}] (mm, N; axis the way the
    air is pushed): actuator disks, each a pressure jump thrust / area
    over its annulus, so a part can sit in a propeller's wash, alone
    (speed 0: hover; the tunnel then runs along the first rotor's axis) or
    with a stream. The reference speed is then the faster of the stream
    and 2.2 times the fastest wake. With rotors the run keeps the
    physical viscosity (`resolve` False) instead of the Reynolds number
    the grid resolves on the part: a jet is not a boundary layer's to scale.

    The run ramps the stream up over two convective times (the time the
    stream takes to pass the body), develops for `develop`, then averages
    in windows of `window` until two in a row agree to `tolerance` (or
    `max_time`). about: the point moments are taken about (mm, default the
    part's centroid).

    Returns drag (N, along the stream), force (N, part frame),
    coefficients on the frontal area, moment (N mm), strouhal (shedding,
    on the reference length, when the side force has a clear period), the
    Reynolds numbers (on referenceLength), levels, converged, and
    `load_case` (the surface forces as a study load case).

    u_lattice is the stream's speed in lattice units (Mach 0.17 at 0.1). A
    run that goes unstable (a strongly shedding sharp section passes air at
    twice the stream's speed) is made again at 0.7 of it, and `latticeSpeed`
    says so."""
    call = dict(locals())
    import trimesh

    from .cutcell import scan_voxelize
    from .heatx_thermal import FLUIDS, Fluid
    from .windtunnel import Tunnel, frontal_area, layout, wall_links

    fl = medium if isinstance(medium, Fluid) else FLUIDS[medium]
    t0 = time.time()
    rotors = list(rotors or [])
    if speed <= 0 and not rotors:
        raise ValueError("no stream and no rotor: nothing moves the fluid")
    if speed > 0:
        R = _frame(direction, span)
    else:
        R = _frame(rotors[0]["axis"], span)
    m = mesh.copy() if isinstance(mesh, trimesh.Trimesh) else trimesh.Trimesh(*mesh)
    T = np.eye(4)
    T[:3, :3] = R
    m.apply_transform(T)
    tri = np.asarray(m.triangles, float)
    lo, hi = np.asarray(m.bounds, float)
    ext = hi - lo
    span_axis = 2 if span is not None else None
    thick = float(ext[1]) if span is not None else float(min(ext[1], ext[2]))
    L = float(max(ext))
    nu_phys = fl.mu / fl.rho
    # rotors, in the tunnel's frame
    rot = []
    w_max = 0.0
    b_lo, b_hi = lo.copy(), hi.copy()
    include = []
    for r in rotors:
        c = R @ np.asarray(r["center"], float)
        a = R @ (np.asarray(r["axis"], float) / np.linalg.norm(r["axis"]))
        rad, hub, thrust = float(r["radius"]), float(r.get("hub", 0.0)), float(r["thrust"])
        area = np.pi * (rad * rad - hub * hub) * 1e-6
        v_i = float(np.sqrt(thrust / (2.0 * fl.rho * area)))
        w_max = max(w_max, 2.0 * v_i)
        rot.append({"c": c, "a": a, "radius": rad, "hub": hub, "thrust": thrust, "area": area, "v_i": v_i})
        reach = rad * np.sqrt(np.maximum(1.0 - a * a, 0.0)) + 0.1 * rad
        rot[-1]["lo"], rot[-1]["hi"] = c - reach, c + reach
        b_lo = np.minimum(b_lo, c - reach)
        b_hi = np.maximum(b_hi, c + reach)
        # the disk on a grid of a sixteenth of its radius, or the body's if that is coarser
        include.append((c - reach, c + reach, rad / 16.0))
    u_ref = max(float(speed), 2.2 * w_max)
    u_char = float(speed) if speed > 0 else w_max
    re_phys = u_ref * thick * 1e-3 / nu_phys
    # how many surface layers the boundary layer asks for
    def resolved(n_layers):
        return (BL_THICKNESS * cells * 2 ** n_layers / bl_cells) ** 2
    if layers is None:
        layers = 0
        while layers < max_layers and resolved(layers) < min(re_phys, re_max or np.inf):
            layers += 1
    # a rotor's jet, and the flow it drives through a duct, are not a boundary layer's to
    # scale: at the Reynolds number a coarse grid resolves on the part the jet is treacle (a
    # long duct lost four fifths of its lip's thrust to wall friction). With rotors the run
    # keeps the physical viscosity, the subgrid model takes up what the grid cannot hold,
    # and the part's own boundary layers are thicker than life (boundaryLayerCells says)
    if resolve is None:
        resolve = not rotors
    re_run = min(re_phys, resolved(layers) if resolve else np.inf, re_max or np.inf)
    re_scale = re_run / re_phys
    h_box = thick / cells

    def vox(hh, corner, dims):
        return scan_voxelize(m, hh, np.asarray(corner, float), np.asarray(dims))

    if domain is None:
        # a spanning body blocks a strip of the tunnel, not a spot: the sides go further out
        domain = (20.0, 40.0, 40.0) if span is not None else (12.0, 10.0, 24.0)
    if fine is None:
        # the finest box reaches past where the wake rolls up (further behind a strut)
        fine = (0.5, 1.5, 4.0) if span is not None else (0.5, 0.5, 1.5)
    lev = layout(lo, hi, h_box, thick, domain=domain, fine=fine, span_axis=span_axis, shells=layers,
                 shell=shell, voxelize=vox, domain_ref=max(L, float(np.max(b_hi - b_lo))), include=include)
    h = lev[-1]["h"]
    fshape = np.array(lev[-1]["shape"])
    solid = vox(h, lev[-1]["origin"] - 0.5 * h, fshape)
    per = (False, False, span is not None)
    bid, q, xyz_t, missed = wall_links(solid, tri, lev[-1]["origin"], h, per)
    if not interpolated:
        q[q >= 0] = 0.5
    n_across = thick / h
    nu_lu = u_lattice * n_across / re_run
    tun = Tunnel(lev, solid, nu_lu, u_lattice, bid=bid, q=q, periodic=per, sgs=sgs, open_far=speed <= 0)
    f_unit = fl.rho * (h * 1e-3) ** 2 * (u_ref / u_lattice) ** 2        # N per lattice force (finest level)
    # rotor disks: a pressure jump thrust / area, spread over a few cells along the axis
    frc_base = {}
    for r in rot:
        # the finest level that holds the whole disk carries it, and so does every finer
        # one where they overlap, each at its own cell
        kd = 0
        for k in range(len(lev) - 1, -1, -1):
            c0_ = lev[k]["origin"] - 0.5 * lev[k]["h"]
            c1_ = c0_ + np.array(lev[k]["shape"]) * lev[k]["h"]
            if np.all(r["lo"] >= c0_ + 2 * lev[k]["h"]) and np.all(r["hi"] <= c1_ - 2 * lev[k]["h"]):
                kd = k
                break
        r["level"] = kd
        sig = lev[kd]["h"]
        norm = None
        for k in range(kd, len(lev)):
            hk = lev[k]["h"]
            xs = tun.centres(k)
            X, Y, Z = np.meshgrid(xs[0] - r["c"][0], xs[1] - r["c"][1], xs[2] - r["c"][2], indexing="ij", sparse=True)
            s_ax = X * r["a"][0] + Y * r["a"][1] + Z * r["a"][2]
            rr = np.sqrt(np.maximum(X * X + Y * Y + Z * Z - s_ax * s_ax, 0.0))
            # the pressure jump thrust / area, spread along the axis (a Gaussian one cell of
            # the disk's level wide), its edges a cell soft
            wgt = (np.exp(-0.5 * (s_ax / sig) ** 2) / (sig * np.sqrt(2.0 * np.pi)) * hk
                   * np.clip((r["radius"] - rr) / sig + 0.5, 0.0, 1.0) * np.clip((rr - r["hub"]) / sig + 0.5, 0.0, 1.0))
            wgt[np.abs(s_ax) > 4.0 * sig] = 0.0
            # per cell, in this level's lattice force: (T / A) x cell face area x share of the jump
            dp_lu = r["thrust"] / (r["area"] * 1e6) * hk * hk / (fl.rho * (hk * 1e-3) ** 2 * (u_ref / u_lattice) ** 2)
            if norm is None:
                tot_w = float(wgt.sum()) * dp_lu
                t_lu = r["thrust"] / (fl.rho * (hk * 1e-3) ** 2 * (u_ref / u_lattice) ** 2)
                if tot_w <= 0:
                    raise ValueError("a rotor disk falls between the grid's cells: more cells")
                norm = t_lu / tot_w                      # the discrete disk carries the thrust exactly
            if k not in frc_base:
                frc_base[k] = np.zeros((3,) + tuple(lev[k]["shape"]), np.float32)
            for ax in range(3):
                frc_base[k][ax] += (norm * dp_lu * r["a"][ax]) * wgt
    # run: ramp, develop, then average in windows until two agree
    area = frontal_area(tri, h / 4.0) * 1e-6
    scale_len = max(float(ext[0]), float(np.sqrt(area)) * 1e3, max([2.0 * r["radius"] for r in rot], default=0.0))
    u_lat_char = u_lattice * u_char / u_ref
    tc = scale_len / h / u_lat_char                                  # finest steps per convective time
    nfs = tun.fine_per_step
    ramp = 2.0 * tc
    u_lat_stream = u_lattice * float(speed) / u_ref
    hist = []
    t_hist = []
    prev = np.zeros(3)
    fine = tun.levels[-1]
    means = []
    side_rms = []
    t_avg0 = ramp + develop * tc
    next_win = t_avg0 + window * tc
    win_start_i = None
    converged = False
    by_error = False
    ramping = True
    while True:
        s = tun.steps
        if s < ramp:
            x = s / ramp
            fac = 0.5 * (1.0 - np.cos(np.pi * x))
            # with a passing side gust (10% of the stream, there and back): a real stream is
            # never exactly symmetric, and a wake that sheds needs something to start it
            gust = 0.1 * u_lat_char * 2.0 * np.pi / ramp * np.cos(2.0 * np.pi * x)
            tun.set_stream(u_lat_stream * fac, u_lat_stream * 0.5 * np.pi * np.sin(np.pi * x) / ramp,
                           (gust, 0.0) if span is not None else (0.8 * gust, 0.6 * gust))
            for k, base in frc_base.items():
                tun.set_force(k, base * np.float32(fac))
        elif ramping:
            ramping = False
            tun.set_stream(u_lat_stream, 0.0)
            if span is not None:
                # a strut's wake is three-dimensional past Re 200, and nothing in a uniform
                # stream on a uniform section says so: 2% of the stream, one wave along the span
                tun.ripple(0.02 * u_lat_char)
            for k, base in frc_base.items():
                tun.set_force(k, base)
        if not tun.sampling and s >= ramp:
            tun.sampling = True
        tun.step()
        if tun.sampling:
            cur = fine.facc.sum(axis=0)
            hist.append((cur - prev) / nfs)
            t_hist.append(tun.steps)
            prev = cur
        if tun.steps % (50 * nfs) < nfs:
            rmin, rmax = float(fine.rho.min()), float(fine.rho.max())
            if not np.isfinite(rmin) or rmax > 0.5 or rmin < -0.5:
                if verbose:
                    for k, L in enumerate(tun.levels):
                        r_ = np.nan_to_num(np.abs(L.rho), nan=np.inf)
                        print(f"  level {k} cell {L.h:.3g}: worst density at {np.unravel_index(int(r_.argmax()), r_.shape)}"
                              f" of {L.shape}, {float(r_.max()):.3g}", flush=True)
                if u_lattice > 0.04:
                    if verbose:
                        print(f"  unstable at lattice speed {u_lattice:.3g}: again at {0.7 * u_lattice:.3g}", flush=True)
                    del tun
                    return external_flow(**{**call, "u_lattice": 0.7 * u_lattice})
                raise RuntimeError("the flow run went unstable: more cells across the body, or a lower re_max")
        if tun.steps >= t_avg0 and win_start_i is None:
            win_start_i = len(hist)
            snaps = [(fine.fb.copy(), fine.facc.sum(axis=0), tun.n_samples, len(hist))]
        if tun.steps >= next_win:
            wm = np.mean(hist[win_start_i:], axis=0)
            means.append(wm)
            side_rms.append(float(np.std(np.asarray(hist[win_start_i:])[:, 1:], axis=0).max()))
            win_start_i = len(hist)
            snaps.append((fine.fb.copy(), fine.facc.sum(axis=0), tun.n_samples, len(hist)))
            next_win += window * tc
            if verbose:
                print(f"  t {tun.steps / tc:6.1f}  window mean F {wm * f_unit}  {time.time() - t0:.0f} s", flush=True)
            if len(means) >= 2:
                ref_f = max(np.linalg.norm(means[-1]), 1e-30)
                # the mean has settled, and so has what sheds: a wake still growing its
                # oscillation reads a steady mean for a while
                grown = (abs(side_rms[-1] - side_rms[-2]) <= 0.15 * side_rms[-1]
                         or side_rms[-1] <= 1e-3 * ref_f)
                if np.linalg.norm(means[-1] - means[-2]) <= tolerance * ref_f and grown:
                    converged = True
                    break
                # a chaotic wake's windows never agree that closely: there the mean over all of
                # them (less the first) is the answer once its standard error is inside the
                # tolerance
                if len(means) >= 4 and grown:
                    mx = np.array([v[0] for v in means[1:]])
                    if mx.std(ddof=1) / np.sqrt(len(mx)) <= tolerance * abs(mx.mean()):
                        converged = True
                        by_error = True
                        break
            if tun.steps >= ramp + max_time * tc:
                break
    hist = np.array(hist)
    # the mean over the last two windows where they agreed; over every window but the
    # first where the run stopped on the standard error, or at its time limit
    if converged and not by_error:
        fb0, fa0, n0, i_0 = snaps[-3] if len(snaps) >= 3 else snaps[0]
    else:
        fb0, fa0, n0, i_0 = snaps[1] if len(snaps) >= 4 else snaps[0]
    std_err = None
    if len(means) >= 3:
        mx = np.array([v[0] for v in means[1:]])
        std_err = float(mx.std(ddof=1) / np.sqrt(len(mx)) * f_unit)
    n_s = max(tun.n_samples - n0, 1)
    F_lu = (fine.facc.sum(axis=0) - fa0) / n_s
    sample = hist[i_0:]
    # a steady flow settles exponentially (the wake fills in on the viscous time): three
    # windows falling toward a limit, with nothing shedding, give the limit (Aitken)
    settled = None
    if len(means) >= 3:
        d1, d2 = means[-2][0] - means[-3][0], means[-1][0] - means[-2][0]
        steady = sample[:, 0].std() <= 2.0 * abs(d2) + 1e-30 and side_rms[-1] <= 1e-3 * abs(means[-1][0])
        if steady and d1 * d2 > 0 and abs(d2) < 0.9 * abs(d1):
            settled = float(means[-1][0] + d2 * d2 / (d1 - d2))
    F_t = F_lu * f_unit                                   # tunnel frame, N
    F_part = R.T @ F_t
    q_dyn = 0.5 * fl.rho * speed ** 2
    fw_t = (fine.fb - fb0) / n_s * f_unit
    xyz = xyz_t @ R
    fw = fw_t @ R
    c0 = np.asarray(about, float) if about is not None else np.asarray(mesh.center_mass if mesh.is_volume
                                                                         else mesh.centroid, float)
    M = np.cross(xyz - c0, fw).sum(axis=0)
    drag = float(F_t[0])
    if settled is not None:
        # the load case and the other components keep the measured average
        drag = settled * f_unit
    # shedding: the side force's dominant period over the averaging time
    strouhal = None
    whole = hist[snaps[0][3]:]
    side = whole[:, 1] if whole[:, 1].std() >= whole[:, 2].std() else whole[:, 2]
    if len(side) > 16 and speed > 0 and side.std() > 1e-3 * abs(F_lu[0]):
        # the period between upward crossings of the mean, when they come evenly
        v = side - side.mean()
        k = np.nonzero((v[:-1] < 0) & (v[1:] >= 0))[0]
        if len(k) >= 4:
            tx = k + v[k] / (v[k] - v[k + 1])               # in samples
            per_ = np.diff(tx)
            if per_.std() < 0.2 * per_.mean():
                strouhal = float(n_across / (per_.mean() * nfs * u_lat_stream))
    rep = {
        "speed": speed, "direction": R[0].tolist(), "medium": fl.name, "drag": drag, "force": F_part.tolist(),
        "moment": M.tolist(), "about": c0.tolist(), "frontalArea": float(area * 1e6), "dynamicPressure": q_dyn,
        "dragCoefficient": float(drag / (q_dyn * area)) if speed > 0 else None,
        "sideCoefficient": float(F_t[1] / (q_dyn * area)) if speed > 0 else None,
        "liftCoefficient": float(F_t[2] / (q_dyn * area)) if speed > 0 else None,
        "dragFluctuation": float(sample[:, 0].std() * f_unit),
        "sideFluctuation": float(side.std() * f_unit), "strouhal": strouhal,
        "reynolds": re_phys, "reynoldsRun": re_run, "reScale": re_scale, "referenceLength": thick,
        "referenceSpeed": u_ref, "bodyLength": L, "voxel": float(h), "cellsAcross": float(n_across), "layers": int(layers),
        "boundaryLayerCells": float(boundary_layer_cells(re_run, n_across)),
        "grid": [int(v) for v in fshape],
        "levels": [{"cell": float(s_["h"]), "grid": [int(v) for v in s_["shape"]]} for s_ in lev],
        "blockage": float(area * 1e6 / ((lev[0]["shape"][1] * lev[0]["h"]) * (lev[0]["shape"][2] * lev[0]["h"]))),
        "steps": int(tun.steps), "time": float((tun.steps - ramp) / tc), "converged": bool(converged),
        "windowMeans": [float(v[0] * f_unit) for v in means], "settledByExtrapolation": settled is not None,
        "dragMeasured": float(F_t[0]), "dragStandardError": std_err,
        "seconds": round(time.time() - t0, 1), "latticeSpeed": float(u_lattice), "missedLinks": int(missed),
        "forceHistory": [float(f"{v:.4g}") for v in (sample[:, 0] * f_unit)[-200:]],
        "wallForceTotal": fw.sum(axis=0).tolist(),
    }
    if rot:
        # the air through each disk, on the finest box
        tun.fill_parents()
        vel = u_ref / u_lattice
        rep["rotors"] = []
        for r in rot:
            Lb = tun.levels[r["level"]]
            xs = tun.centres(r["level"])
            X, Y, Z = np.meshgrid(xs[0] - r["c"][0], xs[1] - r["c"][1], xs[2] - r["c"][2], indexing="ij", sparse=True)
            s_ax = X * r["a"][0] + Y * r["a"][1] + Z * r["a"][2]
            rr = np.sqrt(np.maximum(X * X + Y * Y + Z * Z - s_ax * s_ax, 0.0))
            on = (np.abs(s_ax) <= 0.75 * Lb.h) & (rr <= r["radius"]) & (rr >= r["hub"]) & (Lb.ntype != 1)
            ua = Lb.u[0] * r["a"][0] + Lb.u[1] * r["a"][1] + Lb.u[2] * r["a"][2]
            out = {"thrust": r["thrust"], "inducedVelocity": r["v_i"], "cell": float(Lb.h),
                   "diskVelocity": float(ua[on].mean() * vel) if on.any() else None}
            # the wake a diameter behind the disk
            far = (np.abs(s_ax - 2.0 * r["radius"]) <= 0.75 * Lb.h) & (rr <= 1.5 * r["radius"]) & (Lb.ntype != 1)
            if far.any():
                uf = ua[far] * vel
                out["wakeVelocity"] = float(uf.max())
            # the load on the part along this rotor's axis, as a share of its thrust
            out["download"] = float(F_t @ r["a"])
            out["downloadShare"] = float(F_t @ r["a"] / r["thrust"])
            rep["rotors"].append(out)
    if re_scale < 1:
        rep["note"] = (f"run at Re {re_run:.3g} on the {thick:.3g} mm facing the stream (the physical {re_phys:.3g} "
                       f"needs a boundary layer {bl_cells:g} cells thick: more cells or layers); forces are at the "
                       "physical dynamic pressure with that Reynolds number's coefficient")
        if re_run < 500:
            rep["note"] += ("; below Re 500 friction still adds to a bluff body's drag (a cube reads 1.5 at Re 100 "
                            "against 1.05 past 1e3), so this reading is high for the physical stream: add layers")
    rep["load_case"] = _load_case(xyz, fw, h, patches, speed)
    rep["_surface"] = (xyz, fw)
    rep["_tunnel"] = tun
    return rep


def _load_case(xyz, fw, h, patches, speed):
    """The wall forces as a study load case: the wetted surface cut into
    `patches` patches (k-means on position and normal-ish force direction),
    each a box round its voxels carrying its share."""
    from scipy.cluster.vq import kmeans2

    if len(xyz) == 0:
        return None
    k = int(min(patches, len(xyz)))
    scale = np.ptp(xyz, axis=0).max() or 1.0
    dirs = fw / np.maximum(np.linalg.norm(fw, axis=1, keepdims=True), 1e-30)
    feats = np.hstack([xyz / scale, 0.3 * dirs])
    _, lab = kmeans2(feats, k, seed=0, minit="++")
    loads = []
    for j in range(k):
        s = lab == j
        if not s.any():
            continue
        F = fw[s].sum(axis=0)
        if np.linalg.norm(F) < 1e-9:
            continue
        lo = xyz[s].min(axis=0) - h
        hi = xyz[s].max(axis=0) + h
        loads.append({"shape": {"type": "box", "min": lo.round(4).tolist(), "max": hi.round(4).tolist()},
                      "force": [float(v) for v in F]})
    return {"name": f"flow-{speed:g}ms", "loads": loads}


# --- structure in a rotor's or fan's stream --------------------------------

G = 9.81                       # m/s^2
OPACITY_POWER = 3.0            # a grey cell's share of a column it blocks: density ** this (as SIMP's stiffness)


def _actuator_speed(v_disk: float, radius: float, s):
    """Axial speed on a rotor's axis at signed distance s (mm, + downstream)
    from an actuator disk passing v_disk: v_disk (1 + s / sqrt(R^2 + s^2)),
    twice v_disk far behind and falling off to nothing far ahead."""
    s = np.asarray(s, float)
    return v_disk * (1.0 + s / np.sqrt(radius * radius + s * s))


def _disk_frame(axis):
    a = np.asarray(axis, float)
    a = a / np.linalg.norm(a)
    t = np.array([1.0, 0.0, 0.0]) if abs(a[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = np.cross(a, t)
    e1 /= np.linalg.norm(e1)
    return a, e1, np.cross(a, e1)


class RotorStream:
    """A rotor's stream through a grid of cells, as momentum theory draws
    it, and the drag of the material it meets.

    The disk (centre, unit `axis` pointing the way the air goes, `radius`
    mm, `hub` mm left out) passes v_i = sqrt(T / (2 rho A)) at `thrust` N.
    The stream is followed along its own tubes: the speed on the axis,
    v(s) = v_i (1 + s / sqrt(R^2 + s^2)), sets every tube's radius by
    continuity, r(s) = r_disk sqrt(v_i / v(s)), so the wake narrows to
    R / sqrt(2) far behind the disk and the inflow widens ahead of it:
    material beside the narrowed wake is out of the stream, and material
    ahead of the disk beyond its radius is in it. `ducted`: a duct with no
    diffuser holds the stream at the disk's area, so it passes the disk and
    leaves at sqrt(T / (rho A)) (all the thrust on the rotor), straight.

    Each tube (a bin of the disk's area dA) is walked in the direction the
    air goes, ahead of the disk and behind it apart (the rotor fills in what
    is ahead of it). The first material it meets, at s, drags
    cd (rho v(s)^2 / 2) on the tube's section there, dA v_disk / v(s); what
    lies behind that sits in its wake and adds nothing. On a density field
    a cell blocks its share density^3 of the tube (OPACITY_POWER, as SIMP's
    stiffness: a grey cell is mostly air), so the drag is
    sum_k q_k o_k prod_{j<k} (1 - o_j) down each tube, exact for a part
    that is solid or void, and smooth for the optimizer.

    The grid: `origin` (min corner), `pitch`, `dims`, and `cell_index`
    (dims-shaped ints: each cell's place in the density vector, -1 for
    none), or None for a dense field in C order. `smooth`: densities read
    by trilinear interpolation at half-cell steps (False: the cell each
    point falls in, at whole-cell steps; a crisp voxel model, which
    rotor_blockage reads with tubes half a cell across).
    A slender member's cd is 1.0-1.2 (round to square), a flat plate face
    on 1.28; a streamlined one is about 0.1."""

    def __init__(self, origin, pitch: float, dims, center, axis, radius: float, thrust: float, hub: float = 0.0,
                 cd: float = 1.2, rho: float = 1.2, reach: float | None = None, ducted: bool = False,
                 cell_index=None, smooth: bool = True, bin: float | None = None):
        import scipy.sparse as sp

        origin = np.asarray(origin, float)
        dims = np.asarray(dims, int)
        c = np.asarray(center, float)
        a, e1, e2 = _disk_frame(axis)
        self.radius, self.hub, self.thrust, self.cd, self.ducted = float(radius), float(hub), float(thrust), cd, ducted
        reach = 3.0 * radius if reach is None else float(reach)
        area_m2 = np.pi * (radius * 1e-3) ** 2
        self.v_i = float(np.sqrt(max(thrust, 0.0) / (2.0 * rho * area_m2)))
        v_disk = self.v_i * (np.sqrt(2.0) if ducted else 1.0)
        self.disk_area = float(np.pi * (radius ** 2 - hub ** 2))
        n_cells = int(np.prod(dims)) if cell_index is None else int(cell_index.max()) + 1
        self.n = n_cells
        # the disk's area in bins of one cell (`bin`: of another size)
        b = float(pitch if bin is None else bin)
        nb = int(np.ceil(radius / b))
        uc = (np.arange(-nb, nb) + 0.5) * b
        u0, v0 = (g.ravel() for g in np.meshgrid(uc, uc, indexing="ij"))
        r0 = np.hypot(u0, v0)
        keep = (r0 <= radius) & (r0 >= hub)
        u0, v0 = u0[keep], v0[keep]
        self.d_a = b * b * self.disk_area / max(len(u0) * b * b, 1e-30)     # bins sum to the disk's area exactly
        self.n_bins = len(u0)
        # the stretch of the axis the grid spans
        corners = origin + np.array([[i, j, k] for i in (0, 1) for j in (0, 1) for k in (0, 1)]) * dims * pitch
        sc = (corners - c) @ a
        ds = 0.5 * pitch if smooth else pitch
        s_lo, s_hi = max(-reach, float(sc.min())), min(reach, float(sc.max()))
        n_s = max(int(np.floor((s_hi - s_lo) / ds)), 0)
        rows, cols, vals, samp_bin, samp_s = [], [], [], [], []
        n_samp = 0
        base = np.outer(u0, e1) + np.outer(v0, e2)
        for k in range(n_s):
            s_k = s_lo + (k + 0.5) * ds
            v = float(_actuator_speed(v_disk, radius, s_k))
            if ducted and s_k >= 0:
                v = v_disk
            if v <= 0.05 * v_disk:
                continue                        # far ahead: next to no air moves, and the tubes fan out without end
            pts = c + s_k * a + np.sqrt(v_disk / v) * base
            g = (pts - origin) / pitch - 0.5
            if smooth:
                i0 = np.floor(g).astype(int)
                fr = g - i0
                ids, wts = [], []
                for di in (0, 1):
                    for dj in (0, 1):
                        for dk in (0, 1):
                            ijk = i0 + (di, dj, dk)
                            w = (np.where(di, fr[:, 0], 1 - fr[:, 0]) * np.where(dj, fr[:, 1], 1 - fr[:, 1])
                                 * np.where(dk, fr[:, 2], 1 - fr[:, 2]))
                            ids.append(ijk)
                            wts.append(w)
            else:
                ids, wts = [np.round(g).astype(int)], [np.ones(len(g))]
            got = np.zeros(len(g), bool)
            layer = []
            for ijk, w in zip(ids, wts):
                ok = np.all((ijk >= 0) & (ijk < dims), axis=1) & (w > 1e-9)
                flat = np.zeros(len(g), np.int64)
                if cell_index is None:
                    flat[ok] = np.ravel_multi_index(ijk[ok].T, dims)
                else:
                    flat[ok] = cell_index[tuple(ijk[ok].T)]
                    ok &= flat >= 0
                got |= ok
                layer.append((ok, flat, w))
            if not got.any():
                continue
            local = np.cumsum(got) - 1
            for ok, flat, w in layer:
                rows.append(n_samp + local[ok])
                cols.append(flat[ok])
                vals.append(w[ok])
            m = int(got.sum())
            samp_bin.append(np.flatnonzero(got))
            samp_s.append(np.full(m, s_k))
            # N per unit opacity: cd q on the tube's section here, dA v_disk / v
            n_samp += m
        if n_samp == 0:
            self.w = None
            return
        samp_bin = np.concatenate(samp_bin)
        samp_s = np.concatenate(samp_s)
        w = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n_samp, n_cells))
        # each tube's samples in the order the air meets them, ahead of the disk and behind it apart
        side = (samp_s >= 0).astype(int)
        order = np.lexsort((samp_s, side, samp_bin))
        self.w = w[order].tocsr()
        self.wt = self.w.T.tocsr()
        samp_bin, samp_s, side = samp_bin[order], samp_s[order], side[order]
        seg = samp_bin * 2 + side
        self.start = np.flatnonzero(np.r_[True, seg[1:] != seg[:-1]])
        self.seg_of = np.cumsum(np.r_[True, seg[1:] != seg[:-1]]) - 1
        self.seg_side = side[self.start]
        v = _actuator_speed(v_disk, radius, samp_s)
        if ducted:
            v = np.where(samp_s >= 0, v_disk, v)
        self.q = cd * 0.5 * rho * v * v_disk * self.d_a * 1e-6           # N: cd (rho v^2 / 2) (dA v_disk / v)

    def drag(self, x, grad: bool = False):
        """Drag (N) of the field `x` (one density per cell), with what is
        blocked: {"drag", "downstream": {"drag", "blockedArea"}, "upstream":
        {...}} (areas in mm^2 of the disk whose tubes meet material), and
        with `grad` its derivative per cell."""
        out = {"drag": 0.0, "downstream": {"blockedArea": 0.0, "drag": 0.0}, "upstream": {"blockedArea": 0.0, "drag": 0.0}}
        if self.w is None:
            return (out, np.zeros(self.n)) if grad else out
        rho = np.clip(self.w @ np.asarray(x, float), 0.0, 1.0)
        o = np.minimum(rho ** OPACITY_POWER, 1.0 - 1e-9)
        lg = np.log1p(-o)
        cs = np.cumsum(lg)
        excl = cs - lg                                   # sum of logs before each sample, over all tubes
        t = np.exp(excl - excl[self.start][self.seg_of])  # transmission down to each sample in its own tube
        term = self.q * o * t
        seg_drag = np.add.reduceat(term, self.start)
        ends = np.r_[self.start[1:], len(o)] - 1
        seg_block = 1.0 - t[ends] * (1.0 - o[ends])
        for name, sd in (("upstream", 0), ("downstream", 1)):
            m = self.seg_side == sd
            out[name] = {"blockedArea": float(seg_block[m].sum() * self.d_a), "drag": float(seg_drag[m].sum())}
        out["drag"] = float(seg_drag.sum())
        if not grad:
            return out
        ct = np.cumsum(term)
        behind = (seg_drag + (ct - term)[self.start])[self.seg_of] - ct      # drag of what lies behind each sample
        d_o = self.q * t - behind / (1.0 - o)
        d_rho = d_o * OPACITY_POWER * rho ** (OPACITY_POWER - 1.0)
        return out, self.wt @ d_rho


def rotor_blockage(mesh, center, axis, radius: float, thrust: float, hub: float = 0.0, cd: float = 1.2,
                   rho: float = 1.2, reach: float | None = None, pitch: float = 0.5, ducted: bool = False) -> dict:
    """What a part costs a rotor by sitting in its stream (drone arms and
    motor mounts under a prop, a bracket in front of a fan): the drag of the
    material the disk's air meets, from momentum theory (RotorStream, on the
    part voxelized at `pitch`). The same model is a cost inside the
    optimizer (a study's `aero.rotors`, BlockageCost).

    Returns the blocked area (mm^2 of the disk whose stream meets material,
    and as a share of the disk), the drag (N) and the same as thrust lost
    (share) and as a carried mass that would cost a hovering craft as much
    (g, drag / 9.81), upstream and downstream apart."""
    import trimesh

    from .verify import voxelize

    m = mesh if isinstance(mesh, trimesh.Trimesh) else trimesh.Trimesh(*mesh)
    a, e1, e2 = _disk_frame(axis)
    c = np.asarray(center, float)
    reach = 3.0 * radius if reach is None else float(reach)
    # a box round everything the stream can reach: the inflow widens ahead of the disk
    wide = radius / np.sqrt(0.05)
    pts = np.array([c + s * a + u * e1 + v * e2 for s in (-reach, reach) for u in (-wide, wide) for v in (-wide, wide)])
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    mlo, mhi = np.asarray(m.bounds, float)
    out = {"inducedVelocity": None, "diskArea": round(float(np.pi * (radius ** 2 - hub ** 2)), 1)}
    if np.any(mhi < lo) or np.any(mlo > hi):
        stream = RotorStream(lo, pitch, (1, 1, 1), c, a, radius, thrust, hub, cd, rho, reach, ducted, smooth=False)
        got = stream.drag(np.zeros(1))
    else:
        solid, origin = voxelize(m, pitch, box=(np.maximum(lo, mlo - pitch), np.minimum(hi, mhi + pitch)))
        stream = RotorStream(origin, pitch, solid.shape, c, a, radius, thrust, hub, cd, rho, reach, ducted,
                             smooth=False, bin=0.5 * pitch)
        got = stream.drag(solid.ravel().astype(float))
    out["inducedVelocity"] = round(stream.v_i, 3)
    for side in ("downstream", "upstream"):
        out[side] = {"blockedArea": round(got[side]["blockedArea"], 1), "drag": round(got[side]["drag"], 5)}
    total = out["downstream"]["drag"] + out["upstream"]["drag"]
    out.update({"blockedShare": round((out["downstream"]["blockedArea"] + out["upstream"]["blockedArea"])
                                      / max(out["diskArea"], 1e-9), 4),
                "drag": round(total, 5), "thrustLost": round(total / thrust, 4) if thrust > 0 else 0.0,
                "equivalentMass": round(total / G * 1000.0, 2), "cd": cd})
    return out


class BlockageCost:
    """A study's `aero.rotors` as a cost in the optimizer's material budget:
    the drag of the design in each rotor's stream (RotorStream on the
    design grid), charged as the material that would weigh on a hovering
    craft as much, drag / g, in cells of the study's material. A gram of
    thrust lost under a prop then trades one for one against a gram of
    structure, so the optimizer moves an arm out of the stream, or thins it
    where the air is fastest, exactly when that is lighter overall.
    `rotors`: spec.Rotor list; `density` g/cm^3."""

    def __init__(self, domain, rotors, density: float, grams_per_newton: float = 1000.0 / G):
        mesh = domain.mesh
        idx = np.full(domain.dims, -1, np.int64)
        idx[tuple(mesh.elem_ijk.T)] = np.arange(mesh.n_elem)
        self.streams = [RotorStream(mesh.origin, mesh.pitch, domain.dims, r.center, r.axis, r.radius, r.thrust,
                                    r.hub, r.cd, r.rho, r.reach, r.ducted, cell_index=idx) for r in rotors]
        self.names = [r.name for r in rotors]
        cell_g = density * 1e-3 * mesh.pitch ** 3
        self.cells_per_newton = grams_per_newton / cell_g
        self.grams_per_newton = grams_per_newton
        self.base = 0.0
        # what the drawn (preserved) geometry alone costs: no design choice, so off the budget
        self.base = self.cost(domain.preserve.astype(float))[0]

    def drag(self, x) -> float:
        return float(sum(st.drag(x)["drag"] for st in self.streams))

    def cost(self, x, grad: bool = False):
        """(cells of material the field's drag is worth, less the drawn
        geometry's own; its derivative per cell or None; drag N)."""
        total, g = 0.0, (np.zeros(len(x)) if grad else None)
        for st in self.streams:
            if grad:
                got, gi = st.drag(x, grad=True)
                g += gi * self.cells_per_newton
            else:
                got = st.drag(x)
            total += got["drag"]
        return total * self.cells_per_newton - self.base, g, total


# Entrance loss of a pipe inlet by the radius of its lip over the bore
# (fraction of the dynamic head inside; Idelchik, Handbook of Hydraulic
# Resistance, and Crane TP-410: a sharp flush edge 0.5, a well-rounded
# bellmouth 0.03-0.04). What a duct's lip fails to recover.
_INLET_K = ((0.0, 0.50), (0.02, 0.28), (0.04, 0.24), (0.06, 0.15), (0.10, 0.09), (0.15, 0.04), (0.20, 0.03))
DIFFUSER_MAX_HALF_ANGLE = 7.0     # deg: a conical diffuser's wall holds the flow to about here
DIFFUSER_EFFICIENCY = 0.85        # pressure recovered of the ideal, attached
TIP_DISCHARGE = 0.8               # tip-gap leakage jet (turbomachinery tip-leakage models use 0.6-0.84)


def inlet_loss(lip_radius: float, bore: float) -> float:
    """Entrance loss coefficient of a duct lip of radius `lip_radius` on a
    bore of diameter `bore` (same unit)."""
    r = max(float(lip_radius), 0.0) / float(bore)
    xs, ks = zip(*_INLET_K)
    return float(np.interp(r, xs, ks))


def _duct_state(thrust, speed, area, sigma, k_loss, gap_area, rho):
    """A ducted disk making `thrust` in an axial stream `speed`: (disk speed,
    exit speed, rotor pressure rise, leak share, shaft power)."""
    # T = rho A v (w - V), w = v / sigma
    b, c = -speed * sigma, -thrust * sigma / (rho * area)
    v = 0.5 * (-b + np.sqrt(b * b - 4.0 * c))
    w = v / sigma
    dp = 0.5 * rho * (w * w - speed * speed) + k_loss * 0.5 * rho * v * v
    leak = TIP_DISCHARGE * gap_area / area * np.sqrt(max(2.0 * dp / rho, 0.0)) / v
    return v, w, dp, leak, dp * area * v * (1.0 + leak)


def _open_thrust(power, speed, area, rho, kappa):
    """Thrust an open rotor makes from `power` (induced, with factor kappa)
    in an axial stream: P = kappa T (V + v_i), T = 2 rho A v_i (V + v_i)."""
    lo, hi = 0.0, 1e6
    for _ in range(200):
        t = 0.5 * (lo + hi)
        vi = 0.5 * (-speed + np.sqrt(speed * speed + 2.0 * t / (rho * area)))
        if kappa * t * (speed + vi) > power:
            hi = t
        else:
            lo = t
    return 0.5 * (lo + hi)


def duct_performance(radius: float, thrust: float, tip_gap: float = 0.0, lip_radius: float = 0.0,
                     length: float = 0.0, diffuser_angle: float = 0.0, outer_radius: float | None = None,
                     height: float | None = None, hub: float = 0.0, rho: float = 1.2, speed: float = 0.0,
                     cruise: float | None = None, open_factor: float = 1.0) -> dict:
    """What a duct round a rotor gives and costs, against the same rotor in
    the open at equal induced power. mm, N, m/s; `radius` is the blade tip's.

    The ideal first (momentum theory): a duct whose flow leaves at sigma x
    the disk's area keeps the wake from contracting, so the same thrust
    needs less induced power, T^1.5 / sqrt(4 sigma rho A) against T^1.5 /
    sqrt(2 rho A) in the open; at equal power that is (2 sigma)^(1/3) the
    thrust (1.26 with no diffuser), and the rotor itself carries only 1 /
    (2 sigma) of it. The rest is suction on the inlet lip: the duct's
    thrust, which its wall carries into the frame.

    Then what a small duct loses, each as total pressure the rotor must
    make up or as flow it must pump twice:

    - the lip: a pipe entrance of that radius (inlet_loss), from a sharp
      edge's half a dynamic head to a bellmouth's few percent;
    - the wall: pipe friction over `length` below the disk (Blasius);
    - a diffuser (`diffuser_angle`, half-angle in degrees over `length`):
      it widens the exit only as far as its wall holds the flow
      (DIFFUSER_MAX_HALF_ANGLE) and recovers DIFFUSER_EFFICIENCY of the
      ideal;
    - the tip gap: the rotor's pressure rise drives a jet back through the
      gap's ring, C_d A_gap sqrt(2 dp / rho), air that is pumped and makes
      no thrust. (The same order as turbomachinery's rule of 1-2% of
      efficiency per 1% of blade height. A gap wide enough for the tip
      vortex to re-form, a few percent of the radius, loses more than
      this: read the number there as the most the duct gives.)

    Both rotors are ideal disks otherwise, so the ratio is the duct's own
    doing. A real open rotor also loses at its blade tips, which a duct's
    wall stops: `open_factor` is the open rotor's induced power over the
    ducted one's for that (a blade-element model's tip loss; about 1.05 to
    1.1 for a small two- or three-blade prop), and the ratio rises by its
    2/3 power. Profile power is left out of both: at equal shaft power the
    gain is smaller by the share of power that is not induced.

    `speed`: an axial stream (climb), where the duct's gain fades as the
    wake needs less holding. `cruise`: an edgewise speed (a multirotor
    flying forward). A duct that turns all the air it swallows to
    its axis pays the momentum drag m' V: `momentumDrag` is that bound (a
    duct short against its bore turns less of it), and `profileDrag` its own
    outside as a short cylinder in cross flow (`outer_radius`, `height`).
    Past a cross flow of about half the disk speed the windward lip stalls
    and the hover gain is not kept, which this does not model
    (`crossflowRatio` says where it is).

    Returns thrustRatio (ducted over open, same blade radius, equal induced
    power), thrustGained (N at that power), thrustRatioSameSize (the open
    rotor as big as the duct's outside: what the duct's width would do as
    blade), idealRatio, ductThrust and rotorThrust (N at `thrust`),
    ductShare, the losses, diskVelocity and exitVelocity (m/s), power (W,
    induced), and with `cruise`: momentumDrag, profileDrag, drag (N)."""
    r_d = (radius + tip_gap) * 1e-3
    # the blades' own annulus makes the thrust: the gap's ring carries the leak back
    area = np.pi * ((radius * 1e-3) ** 2 - (hub * 1e-3) ** 2)
    bore = 2.0 * r_d
    lm = max(length, 0.0) * 1e-3
    half = min(max(diffuser_angle, 0.0), DIFFUSER_MAX_HALF_ANGLE)
    r_e = r_d + lm * np.tan(np.radians(half))
    sigma = float(r_e ** 2 / r_d ** 2)
    k_in = inlet_loss(lip_radius, 2.0 * (radius + tip_gap))
    k_diff = (1.0 - DIFFUSER_EFFICIENCY) * (1.0 - 1.0 / sigma ** 2)
    gap_area = np.pi * (r_d ** 2 - (radius * 1e-3) ** 2)
    # wall friction at the disk speed's Reynolds number (two passes: K depends on v weakly)
    k_f = 0.0
    for _ in range(3):
        v, w, dp, leak, power = _duct_state(thrust, speed, area, sigma, k_in + k_diff + k_f, gap_area, rho)
        re = max(v * bore / 1.5e-5, 1.0)
        k_f = 0.316 * re ** -0.25 * lm / bore
    v, w, dp, leak, power = _duct_state(thrust, speed, area, sigma, k_in + k_diff + k_f, gap_area, rho)
    ideal_power = _duct_state(thrust, speed, area, sigma, 0.0, 0.0, rho)[4]
    a_open = np.pi * ((radius * 1e-3) ** 2 - (hub * 1e-3) ** 2)
    t_open = _open_thrust(power, speed, a_open, rho, open_factor)
    t_open_ideal = _open_thrust(ideal_power, speed, a_open, rho, 1.0)
    rotor = float(min(dp * area, thrust * 10.0))
    out = {"thrustRatio": float(thrust / t_open), "idealRatio": float(thrust / t_open_ideal),
           "rotorThrust": rotor, "ductThrust": float(thrust - rotor), "ductShare": float(1.0 - rotor / thrust),
           "exitAreaRatio": sigma, "diskVelocity": float(v), "exitVelocity": float(w), "power": float(power),
           "losses": {"lip": k_in, "wall": float(k_f), "diffuser": float(k_diff), "tipLeak": float(leak)},
           "thrustGained": float(thrust - t_open)}
    if outer_radius is not None:
        a_big = np.pi * ((outer_radius * 1e-3) ** 2 - (hub * 1e-3) ** 2)
        out["thrustRatioSameSize"] = float(thrust / _open_thrust(power, speed, a_big, rho, open_factor))
    if cruise is not None and cruise > 0:
        hv = _duct_state(thrust, 0.0, area, sigma, k_in + k_diff + k_f, gap_area, rho)[0]
        mom = rho * area * hv * cruise
        ro = (outer_radius if outer_radius is not None else radius + tip_gap) * 1e-3
        hh = (height if height is not None else length) * 1e-3
        prof = 1.0 * 0.5 * rho * cruise ** 2 * 2.0 * ro * hh
        out.update(cruiseSpeed=float(cruise), momentumDrag=float(mom), profileDrag=float(prof),
                   drag=float(mom + prof), crossflowRatio=float(cruise / hv))
    return out

