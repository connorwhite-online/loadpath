"""
The field solver behind `loadpath antenna` (loadpath/rf.py) against results
that are exact or measured: the modes of a closed metal box, a box part
filled with plastic (the transcendental solution, with the plastic's face
between grid planes), a half-wave dipole's 73 ohm and its energy balance,
and the line-fed patch Sheen et al. built and measured in 1990.

    python tests/test_rf.py
"""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from loadpath import rf                                # noqa: E402

MM = 1e-3


def _peaks(v, dt, f, floor=0.05):
    t = (np.arange(len(v)) + 0.5) * dt
    V = np.abs(np.exp(-2j * np.pi * f[:, None] * t[None, :]) @ (v * np.hanning(len(v))))
    return np.array([f[i] for i in range(1, len(f) - 1) if V[i] > V[i - 1] and V[i] > V[i + 1]
                     and V[i] > floor * V.max()])


def _pulse(tau):
    return lambda t: math.exp(-((t - 4 * tau) / tau) ** 2)


def test_graded_lines():
    x = rf.graded_lines([(0.0, 0.5), (10.0, 0.5), (12.0, 0.5)], 3.0, -20.0, 30.0)
    d = np.diff(x)
    assert x[0] == -20.0 and x[-1] == 30.0 and np.all(d > 0)
    for fixed in (0.0, 10.0, 12.0):
        assert np.min(np.abs(x - fixed)) < 1e-9, fixed            # the copper's edges are grid lines
    assert d.max() <= 3.0 + 1e-9 and d.min() > 0.1                 # no sliver cells (they set the time step)
    assert np.max(d[1:] / d[:-1]) < 1.6 and np.min(d[1:] / d[:-1]) > 1 / 1.6
    print("graded lines ok: %d lines, cells %.2f to %.2f" % (len(x), d.min(), d.max()))


def test_cavity_modes():
    """An empty metal box 30 x 20 x 10 mm: f = c/2 sqrt((m/a)^2 + (n/b)^2)."""
    a, b, d = 0.03, 0.02, 0.01
    sim = rf.Sim(np.linspace(0, a, 31), np.linspace(0, b, 21), np.linspace(0, d, 11), pml=0, f0=9e9)
    p = sim.add_port(2, [(7, 5, k) for k in range(10)], resistance=5e4)
    sim.build()
    sim.step(30000, _pulse(40e-12))
    pk = _peaks(np.array(p.v), sim.dt, np.linspace(5e9, 20e9, 6001))
    exact = sorted(rf.C0 / 2 * math.sqrt((m / a) ** 2 + (n / b) ** 2) for m in (1, 2, 3) for n in (1, 2))[:4]
    for fe in exact:
        err = np.min(np.abs(pk - fe)) / fe
        assert err < 0.004, (fe, pk)
    print("cavity modes ok:", np.round(np.array(exact) / 1e9, 3), "GHz, all within 0.4%")


def _part_filled_exact(m, n, t, er, kind, a, b, d):
    from scipy.optimize import brentq
    kc2 = (m * math.pi / a) ** 2 + (n * math.pi / b) ** 2

    def g(f):
        k0 = 2 * math.pi * f / rf.C0
        b1, b2 = np.emath.sqrt(er * k0 * k0 - kc2), np.emath.sqrt(k0 * k0 - kc2)
        if kind == "te":
            v = np.tan(b1 * t) / b1 * np.cos(b2 * (d - t)) + np.sin(b2 * (d - t)) / b2
        else:
            v = (b1 / er) * np.tan(b1 * t) * np.cos(b2 * (d - t)) + b2 * np.sin(b2 * (d - t))
        return float(np.real(v * np.cos(b1 * t)))
    fs = np.linspace(1e9, 14e9, 4000)
    vals = [g(f) for f in fs]
    return [brentq(g, fs[i], fs[i + 1]) for i in range(len(fs) - 1) if vals[i] * vals[i + 1] < 0]


def test_part_filled_cavity():
    """A 40 x 30 x 20 mm box with 6.3 mm of permittivity 2.7 on its floor:
    the plastic's face falls 0.3 of the way through a cell, so this is the
    check on how a cell cut by a surface is treated (field along the face:
    the mean of permittivity; across it: the mean of its inverse)."""
    a, b, d, t, er, h = 0.040, 0.030, 0.020, 0.0063, 2.7, 0.001
    x, y, z = (np.arange(0, L + h / 2, h) for L in (a, b, d))
    frac = np.clip((t - z[:-1]) / h, 0, 1)
    eps = np.ones((len(x) - 1, len(y) - 1, len(z) - 1), np.float32) * (1 + frac * (er - 1))[None, None, :]
    inv = np.ones_like(eps) * ((1 - frac) + frac / er)[None, None, :]
    f = np.linspace(3e9, 12e9, 9001)
    for kind, mode, edges in (("te", (1, 0), ("y", [(13, j, 7) for j in range(len(y) - 1)])),
                              ("tm", (1, 1), ("z", [(13, 9, k) for k in range(len(z) - 1)]))):
        sim = rf.Sim(x, y, z, eps_r=eps, pml=0, f0=8e9, inv_eps_r=inv)
        p = sim.add_port({"y": 1, "z": 2}[edges[0]], edges[1], 1e5)
        sim.build()
        sim.step(40000, _pulse(30e-12))
        pk = _peaks(np.array(p.v), sim.dt, f, floor=0.03)
        fe = _part_filled_exact(*mode, t, er, kind, a, b, d)[0]
        err = np.min(np.abs(pk - fe)) / fe
        assert err < 0.004, (kind, fe, pk[:4])
        print("part-filled cavity %s%d%d: exact %.4f GHz, solved within %.2f%%" % (kind.upper(), *mode, fe / 1e9,
                                                                               100 * err))


def _ext(a, n, lo=True, hi=True):
    return np.r_[(a[0] - (a[1] - a[0]) * np.arange(n, 0, -1)) if lo else [], a,
                 (a[-1] + (a[-1] - a[-2]) * np.arange(1, n + 1)) if hi else []]


def test_dipole():
    """A thin 60 mm dipole in free space: at its first resonance the
    radiation resistance is 73 ohm (the textbook half-wave value, and what
    NEC-2 gives: 2.347 GHz, 72 ohm for this wire), and with no loss in the
    model every watt the port accepts must cross the box round it."""
    L, d, pad, npml = 0.060, 0.001, 0.030, 8
    x = _ext(rf.graded_lines([(-d, d), (0, d), (d, d)], 0.006, -pad, pad), npml)
    z = _ext(rf.graded_lines([(-L / 2, d), (0, d), (L / 2, d)], 0.006, -L / 2 - pad, L / 2 + pad), npml)
    sim = rf.Sim(x, x.copy(), z, pml=npml, f0=2.4e9)
    i = int(np.argmin(abs(x)))
    k0, k1, kc = (int(np.argmin(abs(z - v))) for v in (-L / 2, L / 2, 0.0))
    sim.pec_z[i, i, k0:k1] = True
    sim.add_port(2, [(i, i, kc)], 50.0)
    f = np.linspace(1.5e9, 3.3e9, 181)
    out = sim.run(f)
    zin = out["z"]
    s = np.where(np.diff(np.sign(zin.imag)) > 0)[0][0]
    fr = f[s] - zin.imag[s] * (f[s + 1] - f[s]) / (zin.imag[s + 1] - zin.imag[s])
    R = float(np.interp(fr, f, zin.real))
    eff = out["radiationEfficiency"]
    assert out["rungDown"], out["residual"]
    assert abs(fr - 2.347e9) / 2.347e9 < 0.025, fr
    assert abs(R - 73.0) < 5.0, R
    assert np.all(np.abs(eff - 1.0) < 0.01), (eff.min(), eff.max())
    # the far field from the box: a half-wave dipole's pattern (Balanis eq.
    # 4-64) and 2.15 dBi, and the power in it equals the power through the box
    from loadpath import antenna as A
    pt = A.pattern(sim, out)
    th = pt["theta"][:, 0]
    for i, r in enumerate(pt["byFrequency"]):
        assert abs(r["farOverFlux"] - 1) < 0.02, r["farOverFlux"]
        kl = math.pi * r["f"] * L / rf.C0
        ana = ((np.cos(kl * np.cos(th)) - np.cos(kl)) / np.maximum(np.sin(th), 1e-9)) ** 2
        g = pt["byFrequency"][i]["gain"][:, 0]
        assert np.abs(g / g.max() - ana / ana.max())[2:-2].max() < 0.05     # thin-wire sine current, 1.5 to 3.3 GHz
    mid = pt["byFrequency"][len(pt["byFrequency"]) // 2]
    assert abs(mid["skyRhcp"] - (mid["skyGain"] - 3.01)) < 0.05      # a linear wave is half right-hand
    assert abs(mid["directivity"] - 2.2) < 0.2 and abs(mid["upperHemisphere"] - 0.5) < 0.01, {k: v for k, v in mid.items() if k != "gain"}
    print("dipole ok: resonance %.3f GHz (NEC-2 2.347), %.1f ohm, radiated / accepted %.4f to %.4f"
          % (fr / 1e9, R, eff.min(), eff.max()))


def test_measured_patch():
    """Sheen, Ali, Abouzahra & Kong 1990 (IEEE Trans. MTT 38(7), 849-857):
    a 12.45 x 16 mm patch on 0.794 mm Duroid (permittivity 2.2), fed by a
    2.46 mm line. They measured its first resonance at 7.5 GHz."""
    h, Lp, Wp, wl, off, feed, pad, npml, r = 0.794 * MM, 12.45 * MM, 16.0 * MM, 2.46 * MM, 2.09 * MM, 20 * MM, \
        6 * MM, 8, 0.4 * MM
    x = _ext(rf.graded_lines([(0, r), (off, r), (off + wl, r), (Lp, r)], 1.5 * MM, -pad, Lp + pad), npml)
    y = _ext(rf.graded_lines([(-feed, r), (0, r), (Wp, r)], 1.5 * MM, -feed - pad, Wp + pad), npml)
    z = _ext(rf.graded_lines([(0, h / 3), (h, h / 3)], 1.5 * MM, 0, h + pad), npml, lo=False)
    nx, ny, nz = len(x) - 1, len(y) - 1, len(z) - 1
    kz = int(np.argmin(abs(z - h)))
    eps = np.ones((nx, ny, nz), np.float32)
    eps[:, :, :kz] = 2.2
    sim = rf.Sim(x, y, z, eps_r=eps, pml=(npml, npml, npml, npml, 0, npml), f0=10e9)
    X, Y = np.meshgrid(0.5 * (x[1:] + x[:-1]), 0.5 * (y[1:] + y[:-1]), indexing="ij")
    sim.pec_sheet(2, kz, ((X > 0) & (X < Lp) & (Y > 0) & (Y < Wp)) | ((X > off) & (X < off + wl) & (Y > -feed)
                                                                   & (Y <= 0)))
    j0 = int(np.argmin(abs(y + feed)))
    sim.add_port(2, [[(i, j0, k) for k in range(kz)] for i in range(nx + 1) if off - 1e-9 <= x[i] <= off + wl + 1e-9],
                 50.0)
    f = np.linspace(2e9, 20e9, 721)
    out = sim.run(f, bandwidth=18e9, monitor=[7.5e9], max_time=6e-9)
    s = 20 * np.log10(np.abs(out["s11"]))
    dips = [f[i] for i in range(1, len(f) - 1) if s[i] < s[i - 1] and s[i] < s[i + 1] and s[i] < -6]
    assert dips and abs(dips[0] - 7.5e9) / 7.5e9 < 0.02, dips      # 7.40 at these 0.4 mm cells, 7.48 at 0.25
    print("measured patch ok: first resonance %.3f GHz (measured 7.5)" % (dips[0] / 1e9))


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print("all rf tests passed")
