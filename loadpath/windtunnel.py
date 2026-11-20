"""
A virtual wind tunnel: lattice Boltzmann on nested grids, fine at the body
and coarser by halves out to a far boundary many body lengths away.

What sets the drag of a body in a stream is decided in three places, and a
uniform grid can afford none of them at once:

- at the wall, where the boundary layer forms and leaves. Walls sit where
  the surface is, between the grid's nodes (interpolated bounce-back,
  Bouzidi et al. 2001, from the exact distance along every lattice link to
  the mesh), not on the voxel staircase, and the grid there is the finest;
- in the near wake, where the separated shear layers roll up: the fine
  grids reach downstream of the body;
- far away, where the stream must not know there is a tunnel: the coarsest
  grid carries the far field out to where the body's blockage of the
  section is a fraction of a percent, at a cost of a few percent of the run.

Each level is a box of cubic cells half the size of its parent's, stepped
twice for each of the parent's steps (so every level runs at the same
lattice speed and Mach number, and the lattice viscosity doubles with each
refinement). Levels overlap by two parent cells. The child's rim is filled
from the parent, interpolated in space and time; the parent under the
child is filled from the child. What crosses is the fluid's state, not the
populations: density, velocity and the viscous stress, the last rescaled so
the momentum flux is the same on both sides (Dupuis & Chopard 2003;
Lagrava et al. 2012), with the populations rebuilt from them.

Collisions are regularized (Latt & Chopard 2006): each step the
populations are rebuilt from density, velocity and stress alone, which
drops the lattice's non-hydrodynamic modes, the ones that blow up a
two-relaxation-time scheme when the viscosity is small. Past what the grid
resolves, a subgrid viscosity takes up the small eddies: WALE (Nicoud &
Ducros 1999), which vanishes in pure shear, so a laminar boundary layer
stays laminar up to where it separates, as it does on a body at a drone's
Reynolds numbers, and switches on in the rolled-up wake. Populations are
stored as deviations from the fluid at rest (float32 holds the 1e-4
differences the force is made of).

The force on the body is the momentum the populations hand the wall, link
by link, each step (Ladd 1994; Mei et al. 2002 on curved walls).

The step is most of a run, and its kernel is generated source (see
_step_source): the collision and the streaming of a row of cells written
out direction by direction with literal velocities and offsets, rather
than a loop over the lattice's tables with every neighbour's index wrapped
by a remainder. Same arithmetic, the same numbers out, at about twice the
cells a second; one kernel for each closure and kind of level, compiled the
first time a level needs it (about 15 s each, then cached on disk with the
field kernels). LOADPATH_WINDTUNNEL_STEP=cells runs the cell-by-cell kernel
it replaced, which tests hold it to.

`Tunnel` is the solver; aero.external_flow drives it. Lattice units inside
(finest level's cell and step); the caller scales to newtons.
"""
from __future__ import annotations

import os

import numpy as np

from .flow import C19, OPP19, W19, _need_numba

__all__ = ["Tunnel", "layout", "wall_links"]

FLUID, SOLID, FAR, OUT, SKIP = 0, 1, 2, 3, 4
SGS_NONE, SGS_SMAGORINSKY, SGS_WALE = 0, 1, 2
_K = {}


def _kernels():
    if _K:
        return _K
    _need_numba()
    import numba as nb

    C, W, OPP = C19, W19, OPP19

    @nb.njit(parallel=True, fastmath=True, cache=True)
    def step(f, fn, ntype, per, tau0, sgs, csq, u, un, rho, P, tau, bid, q, facc, fb, do_force,
             sig, has_sig, ufar, gx, gy, gz, frc, has_frc):
        """One stream-and-collide step (pull) of one level. f: post-collision
        populations less the weights. Writes fn and the moments (rho less
        one, un, the pre-collision non-equilibrium stress P, tau)."""
        nx, ny, nz = ntype.shape
        for x in nb.prange(nx):
            fin = np.empty(19)
            g = np.empty((3, 3))
            fxl = 0.0
            fyl = 0.0
            fzl = 0.0
            for y in range(ny):
                for z in range(nz):
                    t = ntype[x, y, z]
                    if t == 1 or t == 2 or t == 4:
                        continue
                    if t == 3:
                        # open boundary: ambient pressure, the velocity of the node inward
                        ix = np.int64(x)
                        iy = np.int64(y)
                        iz = np.int64(z)
                        if x == 0:
                            ix = 1
                        elif x == nx - 1:
                            ix = nx - 2
                        if not per[1]:
                            if y == 0:
                                iy = 1
                            elif y == ny - 1:
                                iy = ny - 2
                        if not per[2]:
                            if z == 0:
                                iz = 1
                            elif z == nz - 1:
                                iz = nz - 2
                        vx = u[0, ix, iy, iz]
                        vy = u[1, ix, iy, iz]
                        vz = u[2, ix, iy, iz]
                        usq = vx * vx + vy * vy + vz * vz
                        for i in range(19):
                            cu = C[i, 0] * vx + C[i, 1] * vy + C[i, 2] * vz
                            fn[x, y, z, i] = W[i] * (3.0 * cu + 4.5 * cu * cu - 1.5 * usq)
                        rho[x, y, z] = 0.0
                        un[0, x, y, z] = vx
                        un[1, x, y, z] = vy
                        un[2, x, y, z] = vz
                        for k in range(6):
                            P[k, x, y, z] = 0.0
                        tau[x, y, z] = tau0
                        continue
                    b = bid[x, y, z]
                    for i in range(19):
                        sx = x - C[i, 0]
                        sy = y - C[i, 1]
                        sz = z - C[i, 2]
                        if per[0]:
                            if sx < 0:
                                sx += nx
                            elif sx >= nx:
                                sx -= nx
                        if per[1]:
                            if sy < 0:
                                sy += ny
                            elif sy >= ny:
                                sy -= ny
                        if per[2]:
                            if sz < 0:
                                sz += nz
                            elif sz >= nz:
                                sz -= nz
                        if ntype[sx, sy, sz] == 1:
                            # a wall on the link toward -c_i, a fraction qq of the link away
                            j = OPP[i]
                            qq = 0.5
                            if b >= 0:
                                qq = q[b, j]
                            fj = f[x, y, z, j]
                            val = fj
                            if qq < 0.5:
                                tx = x + C[i, 0]
                                ty = y + C[i, 1]
                                tz = z + C[i, 2]
                                if per[0]:
                                    if tx < 0:
                                        tx += nx
                                    elif tx >= nx:
                                        tx -= nx
                                if per[1]:
                                    if ty < 0:
                                        ty += ny
                                    elif ty >= ny:
                                        ty -= ny
                                if per[2]:
                                    if tz < 0:
                                        tz += nz
                                    elif tz >= nz:
                                        tz -= nz
                                if 0 <= tx < nx and 0 <= ty < ny and 0 <= tz < nz and ntype[tx, ty, tz] != 1:
                                    val = 2.0 * qq * fj + (1.0 - 2.0 * qq) * f[tx, ty, tz, j]
                            elif qq > 0.5:
                                val = fj / (2.0 * qq) + (1.0 - 1.0 / (2.0 * qq)) * f[x, y, z, i]
                            fin[i] = val
                            if do_force:
                                mom = fj + val
                                mx = C[j, 0] * mom
                                my = C[j, 1] * mom
                                mz = C[j, 2] * mom
                                fxl += mx
                                fyl += my
                                fzl += mz
                                if b >= 0:
                                    fb[b, 0] += mx
                                    fb[b, 1] += my
                                    fb[b, 2] += mz
                        else:
                            fin[i] = f[sx, sy, sz, i]
                    dr = 0.0
                    jx = 0.0
                    jy = 0.0
                    jz = 0.0
                    pxx = 0.0
                    pyy = 0.0
                    pzz = 0.0
                    pxy = 0.0
                    pxz = 0.0
                    pyz = 0.0
                    for i in range(19):
                        v = fin[i]
                        dr += v
                        jx += v * C[i, 0]
                        jy += v * C[i, 1]
                        jz += v * C[i, 2]
                        pxx += v * C[i, 0] * C[i, 0]
                        pyy += v * C[i, 1] * C[i, 1]
                        pzz += v * C[i, 2] * C[i, 2]
                        pxy += v * C[i, 0] * C[i, 1]
                        pxz += v * C[i, 0] * C[i, 2]
                        pyz += v * C[i, 1] * C[i, 2]
                    r = 1.0 + dr
                    Fx = gx * r
                    Fy = gy * r
                    Fz = gz * r
                    if has_frc:
                        Fx += frc[0, x, y, z]
                        Fy += frc[1, x, y, z]
                        Fz += frc[2, x, y, z]
                    vx = (jx + 0.5 * Fx) / r
                    vy = (jy + 0.5 * Fy) / r
                    vz = (jz + 0.5 * Fz) / r
                    usq = vx * vx + vy * vy + vz * vz
                    # the non-equilibrium stress: the second moment less the
                    # equilibrium's, with the forcing's share (Guo)
                    third = dr / 3.0
                    pxx += vx * Fx - third - r * vx * vx
                    pyy += vy * Fy - third - r * vy * vy
                    pzz += vz * Fz - third - r * vz * vz
                    pxy += 0.5 * (vx * Fy + vy * Fx) - r * vx * vy
                    pxz += 0.5 * (vx * Fz + vz * Fx) - r * vx * vz
                    pyz += 0.5 * (vy * Fz + vz * Fy) - r * vy * vz
                    tl = tau0
                    if sgs == 1:
                        Q = np.sqrt(pxx * pxx + pyy * pyy + pzz * pzz + 2.0 * (pxy * pxy + pxz * pxz + pyz * pyz))
                        tl = 0.5 * (tau0 + np.sqrt(tau0 * tau0 + 18.0 * 1.41421356 * csq * Q / r))
                    elif sgs == 2:
                        # velocity gradient by central differences of the last step's velocity;
                        # a solid neighbour is at rest
                        for a in range(3):
                            xp = x + (1 if a == 0 else 0)
                            yp = y + (1 if a == 1 else 0)
                            zp = z + (1 if a == 2 else 0)
                            xm = x - (1 if a == 0 else 0)
                            ym = y - (1 if a == 1 else 0)
                            zm = z - (1 if a == 2 else 0)
                            if per[0]:
                                if xp < 0:
                                    xp += nx
                                elif xp >= nx:
                                    xp -= nx
                                if xm < 0:
                                    xm += nx
                                elif xm >= nx:
                                    xm -= nx
                            if per[1]:
                                if yp < 0:
                                    yp += ny
                                elif yp >= ny:
                                    yp -= ny
                                if ym < 0:
                                    ym += ny
                                elif ym >= ny:
                                    ym -= ny
                            if per[2]:
                                if zp < 0:
                                    zp += nz
                                elif zp >= nz:
                                    zp -= nz
                                if zm < 0:
                                    zm += nz
                                elif zm >= nz:
                                    zm -= nz
                            sp = ntype[xp, yp, zp] == 1
                            sm = ntype[xm, ym, zm] == 1
                            for c in range(3):
                                up = 0.0 if sp else u[c, xp, yp, zp]
                                um = 0.0 if sm else u[c, xm, ym, zm]
                                g[c, a] = 0.5 * (up - um)
                        ss = 0.0
                        sd = 0.0
                        g2t = 0.0
                        for a in range(3):
                            for c in range(3):
                                g2t += g[a, c] * g[c, a]
                        for a in range(3):
                            for c in range(3):
                                s_ac = 0.5 * (g[a, c] + g[c, a])
                                ss += s_ac * s_ac
                                g2 = 0.0
                                g2r = 0.0
                                for k in range(3):
                                    g2 += g[a, k] * g[k, c]
                                    g2r += g[c, k] * g[k, a]
                                d_ac = 0.5 * (g2 + g2r)
                                if a == c:
                                    d_ac -= g2t / 3.0
                                sd += d_ac * d_ac
                        den = ss ** 2.5 + sd ** 1.25
                        if den > 1e-30:
                            tl = tau0 + 3.0 * csq * sd ** 1.5 / den
                    om = 1.0 - 1.0 / tl
                    # the forcing: the regularized populations carry no first moment of their
                    # own, so the source gives the half of F the velocity shift has not (their
                    # sum is F a step, whatever tau), and half the (uF + Fu) of Guo's second moment
                    of = 0.5
                    tr = (pxx + pyy + pzz) / 3.0
                    omb = 1.0 - 1.0 / max(tl, 1.0)
                    uF = vx * Fx + vy * Fy + vz * Fz
                    s = 0.0
                    if has_sig:
                        s = sig[x, y, z]
                    for i in range(19):
                        cx = C[i, 0]
                        cy = C[i, 1]
                        cz = C[i, 2]
                        cu = cx * vx + cy * vy + cz * vz
                        feq = W[i] * (dr + r * (3.0 * cu + 4.5 * cu * cu - 1.5 * usq))
                        # the shear part of the stress relaxes at the fluid's viscosity; its
                        # isotropic part is sound, which air at these speeds does not carry, and
                        # relaxes at tau 1 or slower (a bulk viscosity: nothing incompressible
                        # feels it, and the lattice's pressure waves die instead of ringing
                        # between the levels of a tunnel a strut spans)
                        fne = W[i] * 4.5 * (cx * cx * (pxx - tr) + cy * cy * (pyy - tr) + cz * cz * (pzz - tr)
                                            + 2.0 * (cx * cy * pxy + cx * cz * pxz + cy * cz * pyz))
                        fnb = W[i] * 4.5 * tr * (cx * cx + cy * cy + cz * cz - 1.0)
                        cF = cx * Fx + cy * Fy + cz * Fz
                        out = feq + om * fne + omb * fnb + of * W[i] * (3.0 * (cF - uF) + 9.0 * cu * cF)
                        if s > 0.0:
                            # sponge toward the free stream (inlet and sides)
                            cuf = cx * ufar
                            out += s * (W[i] * (3.0 * cuf + 4.5 * cuf * cuf - 1.5 * ufar * ufar) - out)
                        elif s < 0.0:
                            # sponge toward ambient pressure at the local velocity (outlet)
                            out += -s * (W[i] * (3.0 * cu + 4.5 * cu * cu - 1.5 * usq) - out)
                        fn[x, y, z, i] = out
                    rho[x, y, z] = dr
                    un[0, x, y, z] = vx
                    un[1, x, y, z] = vy
                    un[2, x, y, z] = vz
                    P[0, x, y, z] = pxx
                    P[1, x, y, z] = pyy
                    P[2, x, y, z] = pzz
                    P[3, x, y, z] = pxy
                    P[4, x, y, z] = pxz
                    P[5, x, y, z] = pyz
                    tau[x, y, z] = tl
            if do_force:
                facc[x, 0] += fxl
                facc[x, 1] += fyl
                facc[x, 2] += fzl

    @nb.njit(parallel=True, fastmath=True, cache=True)
    def gather(rho, u, P, tau, gi, off, per, M):
        """The parent's state at the child's rim cells gi (child indices),
        cubic between parent cell centres (four a side). At a tunnel's Mach
        number a 1% error in the velocity handed across is a 10% error in
        the pressure that balances it (dp = rho c du), and linear
        interpolation in the curved flow near a body is 1% off.
        M: (n, 11) rho, u, P, tau.

        The 64 parent cells are summed a line of four along z at a time, the
        line's four weights applied before the weight of its (x, y), and each
        of the 11 sums kept in a register until the end. Summed one cell at
        a time into M (gather_cells, kept to test this against), every one
        of the 704 terms waited on the last store to M, and that wait, not
        the arithmetic, was most of the time (2.3x)."""
        nx, ny, nz = rho.shape
        p0 = per[0]
        p1 = per[1]
        p2 = per[2]
        for n in nb.prange(gi.shape[0]):
            xc = off[0] + 0.5 * (gi[n, 0] + 0.5) - 0.5
            b0 = int(np.floor(xc))
            t = xc - b0
            xb = b0 - 1
            wx0 = -t * (t - 1.0) * (t - 2.0) / 6.0
            wx1 = (t + 1.0) * (t - 1.0) * (t - 2.0) / 2.0
            wx2 = -(t + 1.0) * t * (t - 2.0) / 2.0
            wx3 = (t + 1.0) * t * (t - 1.0) / 6.0
            xc = off[1] + 0.5 * (gi[n, 1] + 0.5) - 0.5
            b0 = int(np.floor(xc))
            t = xc - b0
            yb = b0 - 1
            wy0 = -t * (t - 1.0) * (t - 2.0) / 6.0
            wy1 = (t + 1.0) * (t - 1.0) * (t - 2.0) / 2.0
            wy2 = -(t + 1.0) * t * (t - 2.0) / 2.0
            wy3 = (t + 1.0) * t * (t - 1.0) / 6.0
            xc = off[2] + 0.5 * (gi[n, 2] + 0.5) - 0.5
            b0 = int(np.floor(xc))
            t = xc - b0
            zb = b0 - 1
            w0 = -t * (t - 1.0) * (t - 2.0) / 6.0
            w1 = (t + 1.0) * (t - 1.0) * (t - 2.0) / 2.0
            w2 = -(t + 1.0) * t * (t - 2.0) / 2.0
            w3 = (t + 1.0) * t * (t - 1.0) / 6.0
            z0 = zb
            z1 = zb + 1
            z2 = zb + 2
            z3 = zb + 3
            if p2:
                if z0 < 0:
                    z0 += nz
                elif z0 >= nz:
                    z0 -= nz
                if z1 < 0:
                    z1 += nz
                elif z1 >= nz:
                    z1 -= nz
                if z2 < 0:
                    z2 += nz
                elif z2 >= nz:
                    z2 -= nz
                if z3 < 0:
                    z3 += nz
                elif z3 >= nz:
                    z3 -= nz
            s0 = 0.0
            s1 = 0.0
            s2 = 0.0
            s3 = 0.0
            s4 = 0.0
            s5 = 0.0
            s6 = 0.0
            s7 = 0.0
            s8 = 0.0
            s9 = 0.0
            s10 = 0.0
            for dx in range(4):
                xi = xb + dx
                if p0:
                    if xi < 0:
                        xi += nx
                    elif xi >= nx:
                        xi -= nx
                wx = wx0 if dx == 0 else (wx1 if dx == 1 else (wx2 if dx == 2 else wx3))
                for dy in range(4):
                    yi = yb + dy
                    if p1:
                        if yi < 0:
                            yi += ny
                        elif yi >= ny:
                            yi -= ny
                    wxy = wx * (wy0 if dy == 0 else (wy1 if dy == 1 else (wy2 if dy == 2 else wy3)))
                    r = rho[xi, yi]
                    s0 += wxy * (w0 * r[z0] + w1 * r[z1] + w2 * r[z2] + w3 * r[z3])
                    r = u[0, xi, yi]
                    s1 += wxy * (w0 * r[z0] + w1 * r[z1] + w2 * r[z2] + w3 * r[z3])
                    r = u[1, xi, yi]
                    s2 += wxy * (w0 * r[z0] + w1 * r[z1] + w2 * r[z2] + w3 * r[z3])
                    r = u[2, xi, yi]
                    s3 += wxy * (w0 * r[z0] + w1 * r[z1] + w2 * r[z2] + w3 * r[z3])
                    r = P[0, xi, yi]
                    s4 += wxy * (w0 * r[z0] + w1 * r[z1] + w2 * r[z2] + w3 * r[z3])
                    r = P[1, xi, yi]
                    s5 += wxy * (w0 * r[z0] + w1 * r[z1] + w2 * r[z2] + w3 * r[z3])
                    r = P[2, xi, yi]
                    s6 += wxy * (w0 * r[z0] + w1 * r[z1] + w2 * r[z2] + w3 * r[z3])
                    r = P[3, xi, yi]
                    s7 += wxy * (w0 * r[z0] + w1 * r[z1] + w2 * r[z2] + w3 * r[z3])
                    r = P[4, xi, yi]
                    s8 += wxy * (w0 * r[z0] + w1 * r[z1] + w2 * r[z2] + w3 * r[z3])
                    r = P[5, xi, yi]
                    s9 += wxy * (w0 * r[z0] + w1 * r[z1] + w2 * r[z2] + w3 * r[z3])
                    r = tau[xi, yi]
                    s10 += wxy * (w0 * r[z0] + w1 * r[z1] + w2 * r[z2] + w3 * r[z3])
            M[n, 0] = s0
            M[n, 1] = s1
            M[n, 2] = s2
            M[n, 3] = s3
            M[n, 4] = s4
            M[n, 5] = s5
            M[n, 6] = s6
            M[n, 7] = s7
            M[n, 8] = s8
            M[n, 9] = s9
            M[n, 10] = s10

    @nb.njit(parallel=True, fastmath=True, cache=True)
    def gather_cells(rho, u, P, tau, gi, off, per, M):
        """The parent's state at the child's rim cells gi (child indices),
        cubic between parent cell centres (four a side). At a tunnel's Mach
        number a 1% error in the velocity handed across is a 10% error in
        the pressure that balances it (dp = rho c du), and linear
        interpolation in the curved flow near a body is 1% off.
        M: (n, 11) rho, u, P, tau. (The sum one cell at a time; `gather`
        is the same sum taken a line at a time.)"""
        nx, ny, nz = rho.shape
        for n in nb.prange(gi.shape[0]):
            base = np.empty(3, np.int64)
            wt = np.empty((3, 4))
            for a in range(3):
                xc = off[a] + 0.5 * (gi[n, a] + 0.5) - 0.5
                b0 = int(np.floor(xc))
                t = xc - b0
                base[a] = b0 - 1
                wt[a, 0] = -t * (t - 1.0) * (t - 2.0) / 6.0
                wt[a, 1] = (t + 1.0) * (t - 1.0) * (t - 2.0) / 2.0
                wt[a, 2] = -(t + 1.0) * t * (t - 2.0) / 2.0
                wt[a, 3] = (t + 1.0) * t * (t - 1.0) / 6.0
            for k in range(11):
                M[n, k] = 0.0
            for dx in range(4):
                xi = base[0] + dx
                if per[0]:
                    if xi < 0:
                        xi += nx
                    elif xi >= nx:
                        xi -= nx
                for dy in range(4):
                    yi = base[1] + dy
                    if per[1]:
                        if yi < 0:
                            yi += ny
                        elif yi >= ny:
                            yi -= ny
                    wxy = wt[0, dx] * wt[1, dy]
                    for dz in range(4):
                        zi = base[2] + dz
                        if per[2]:
                            if zi < 0:
                                zi += nz
                            elif zi >= nz:
                                zi -= nz
                        w = wxy * wt[2, dz]
                        M[n, 0] += w * rho[xi, yi, zi]
                        M[n, 1] += w * u[0, xi, yi, zi]
                        M[n, 2] += w * u[1, xi, yi, zi]
                        M[n, 3] += w * u[2, xi, yi, zi]
                        for k in range(6):
                            M[n, 4 + k] += w * P[k, xi, yi, zi]
                        M[n, 10] += w * tau[xi, yi, zi]

    @nb.njit(parallel=True, fastmath=True, cache=True)
    def scatter(f, rho, u, un, P, tau, gi, M):
        """Rim cells of a child from the parent's state M: the same density,
        velocity and total viscosity (so the relaxation time's excess over
        1/2 doubles), the same strain rate (half per lattice step), and the
        populations rebuilt from them, post-collision."""
        for n in nb.prange(gi.shape[0]):
            x = gi[n, 0]
            y = gi[n, 1]
            z = gi[n, 2]
            dr = M[n, 0]
            vx = M[n, 1]
            vy = M[n, 2]
            vz = M[n, 3]
            tc = M[n, 10]
            tf = 0.5 + 2.0 * (tc - 0.5)
            sc = tf / (2.0 * tc)
            pxx = M[n, 4] * sc
            pyy = M[n, 5] * sc
            pzz = M[n, 6] * sc
            pxy = M[n, 7] * sc
            pxz = M[n, 8] * sc
            pyz = M[n, 9] * sc
            r = 1.0 + dr
            usq = vx * vx + vy * vy + vz * vz
            om = 1.0 - 1.0 / tf
            tr = (pxx + pyy + pzz) / 3.0
            for i in range(19):
                cx = C[i, 0]
                cy = C[i, 1]
                cz = C[i, 2]
                cu = cx * vx + cy * vy + cz * vz
                feq = W[i] * (dr + r * (3.0 * cu + 4.5 * cu * cu - 1.5 * usq))
                fne = W[i] * 4.5 * (cx * cx * pxx + cy * cy * pyy + cz * cz * pzz
                                    + 2.0 * (cx * cy * pxy + cx * cz * pxz + cy * cz * pyz) - tr)
                f[x, y, z, i] = feq + om * fne
            rho[x, y, z] = dr
            u[0, x, y, z] = vx
            u[1, x, y, z] = vy
            u[2, x, y, z] = vz
            un[0, x, y, z] = vx
            un[1, x, y, z] = vy
            un[2, x, y, z] = vz
            P[0, x, y, z] = pxx
            P[1, x, y, z] = pyy
            P[2, x, y, z] = pzz
            P[3, x, y, z] = pxy
            P[4, x, y, z] = pxz
            P[5, x, y, z] = pyz
            tau[x, y, z] = tf

    @nb.njit(parallel=True, fastmath=True, cache=True)
    def restrict(frho, fu, fP, ftau, fnt, cf, crho, cu, cun, cP, ctau, ci, off):
        """Parent cells ci under a child, from the child's eight cells (the
        fluid ones): mean density, momentum and stress, the stress rescaled
        to the parent's relaxation time."""
        for n in nb.prange(ci.shape[0]):
            x = ci[n, 0]
            y = ci[n, 1]
            z = ci[n, 2]
            fx = 2 * (x - off[0])
            fy = 2 * (y - off[1])
            fz = 2 * (z - off[2])
            cnt = 0.0
            dr = 0.0
            jx = 0.0
            jy = 0.0
            jz = 0.0
            p0 = 0.0
            p1 = 0.0
            p2 = 0.0
            p3 = 0.0
            p4 = 0.0
            p5 = 0.0
            tt = 0.0
            for dx in range(2):
                for dy in range(2):
                    for dz in range(2):
                        a = fx + dx
                        b = fy + dy
                        c = fz + dz
                        if fnt[a, b, c] == 1:
                            continue
                        cnt += 1.0
                        rr = 1.0 + frho[a, b, c]
                        dr += frho[a, b, c]
                        jx += rr * fu[0, a, b, c]
                        jy += rr * fu[1, a, b, c]
                        jz += rr * fu[2, a, b, c]
                        tf = ftau[a, b, c]
                        tcl = 0.5 + 0.5 * (tf - 0.5)
                        sc = 2.0 * tcl / tf
                        p0 += sc * fP[0, a, b, c]
                        p1 += sc * fP[1, a, b, c]
                        p2 += sc * fP[2, a, b, c]
                        p3 += sc * fP[3, a, b, c]
                        p4 += sc * fP[4, a, b, c]
                        p5 += sc * fP[5, a, b, c]
                        tt += tcl
            if cnt == 0.0:
                cnt = 1.0
                tt = 1.0
            dr /= cnt
            r = 1.0 + dr
            vx = jx / cnt / r
            vy = jy / cnt / r
            vz = jz / cnt / r
            pxx = p0 / cnt
            pyy = p1 / cnt
            pzz = p2 / cnt
            pxy = p3 / cnt
            pxz = p4 / cnt
            pyz = p5 / cnt
            tc = tt / cnt
            usq = vx * vx + vy * vy + vz * vz
            om = 1.0 - 1.0 / tc
            tr = (pxx + pyy + pzz) / 3.0
            for i in range(19):
                cx = C[i, 0]
                cy = C[i, 1]
                cz = C[i, 2]
                cuv = cx * vx + cy * vy + cz * vz
                feq = W[i] * (dr + r * (3.0 * cuv + 4.5 * cuv * cuv - 1.5 * usq))
                fne = W[i] * 4.5 * (cx * cx * pxx + cy * cy * pyy + cz * cz * pzz
                                    + 2.0 * (cx * cy * pxy + cx * cz * pxz + cy * cz * pyz) - tr)
                cf[x, y, z, i] = feq + om * fne
            crho[x, y, z] = dr
            cu[0, x, y, z] = vx
            cu[1, x, y, z] = vy
            cu[2, x, y, z] = vz
            cun[0, x, y, z] = vx
            cun[1, x, y, z] = vy
            cun[2, x, y, z] = vz
            cP[0, x, y, z] = pxx
            cP[1, x, y, z] = pyy
            cP[2, x, y, z] = pzz
            cP[3, x, y, z] = pxy
            cP[4, x, y, z] = pxz
            cP[5, x, y, z] = pyz
            ctau[x, y, z] = tc

    @nb.njit(cache=True)
    def link_cuts(tri, origin, h, ntype, bid, q, per):
        """Where each wall link meets the surface: for boundary node b and
        direction i with a solid node at x + c_i, q[b, i] is the fraction of
        the link from the node to the first triangle it crosses."""
        nx, ny, nz = ntype.shape
        eps = 1e-9
        for t in range(tri.shape[0]):
            a = tri[t, 0]
            e1 = tri[t, 1] - a
            e2 = tri[t, 2] - a
            lo = np.empty(3)
            hi = np.empty(3)
            for k in range(3):
                lo[k] = min(tri[t, 0, k], tri[t, 1, k], tri[t, 2, k])
                hi[k] = max(tri[t, 0, k], tri[t, 1, k], tri[t, 2, k])
            i0 = int(np.floor((lo[0] - origin[0]) / h)) - 1
            i1 = int(np.ceil((hi[0] - origin[0]) / h)) + 1
            j0 = int(np.floor((lo[1] - origin[1]) / h)) - 1
            j1 = int(np.ceil((hi[1] - origin[1]) / h)) + 1
            k0 = int(np.floor((lo[2] - origin[2]) / h)) - 1
            k1 = int(np.ceil((hi[2] - origin[2]) / h)) + 1
            for ii in range(i0, i1 + 1):
                x = ii % nx if per[0] else ii
                if x < 0 or x >= nx:
                    continue
                for jj in range(j0, j1 + 1):
                    y = jj % ny if per[1] else jj
                    if y < 0 or y >= ny:
                        continue
                    for kk in range(k0, k1 + 1):
                        z = kk % nz if per[2] else kk
                        if z < 0 or z >= nz:
                            continue
                        b = bid[x, y, z]
                        if b < 0:
                            continue
                        px = origin[0] + ii * h - a[0]
                        py = origin[1] + jj * h - a[1]
                        pz = origin[2] + kk * h - a[2]
                        for i in range(1, 19):
                            if q[b, i] < 0.0:
                                continue
                            dx = C[i, 0] * h
                            dy = C[i, 1] * h
                            dz = C[i, 2] * h
                            # Moller-Trumbore
                            hx = dy * e2[2] - dz * e2[1]
                            hy = dz * e2[0] - dx * e2[2]
                            hz = dx * e2[1] - dy * e2[0]
                            det = e1[0] * hx + e1[1] * hy + e1[2] * hz
                            if abs(det) < 1e-30:
                                continue
                            inv = 1.0 / det
                            uu = (px * hx + py * hy + pz * hz) * inv
                            if uu < -eps or uu > 1.0 + eps:
                                continue
                            qx = py * e1[2] - pz * e1[1]
                            qy = pz * e1[0] - px * e1[2]
                            qz = px * e1[1] - py * e1[0]
                            vv = (dx * qx + dy * qy + dz * qz) * inv
                            if vv < -eps or uu + vv > 1.0 + eps:
                                continue
                            tt = (e2[0] * qx + e2[1] * qy + e2[2] * qz) * inv
                            if tt < -1e-6 or tt > 1.0 + 1e-6:
                                continue
                            if tt < q[b, i]:
                                q[b, i] = tt

    @nb.njit(cache=True)
    def shadow(tri, lo, pitch, cover):
        """Pixels (centres) of the y-z plane the triangles cover, seen along x."""
        ny, nz = cover.shape
        for t in range(tri.shape[0]):
            y0 = tri[t, 0, 1]
            z0 = tri[t, 0, 2]
            y1 = tri[t, 1, 1]
            z1 = tri[t, 1, 2]
            y2 = tri[t, 2, 1]
            z2 = tri[t, 2, 2]
            d = (y1 - y0) * (z2 - z0) - (y2 - y0) * (z1 - z0)
            if abs(d) < 1e-30:
                continue
            j0 = max(int(np.floor((min(y0, y1, y2) - lo[0]) / pitch - 0.5)), 0)
            j1 = min(int(np.ceil((max(y0, y1, y2) - lo[0]) / pitch - 0.5)), ny - 1)
            k0 = max(int(np.floor((min(z0, z1, z2) - lo[1]) / pitch - 0.5)), 0)
            k1 = min(int(np.ceil((max(z0, z1, z2) - lo[1]) / pitch - 0.5)), nz - 1)
            for j in range(j0, j1 + 1):
                py = lo[0] + (j + 0.5) * pitch
                for k in range(k0, k1 + 1):
                    pz = lo[1] + (k + 0.5) * pitch
                    l1 = ((py - y0) * (z2 - z0) - (y2 - y0) * (pz - z0)) / d
                    l2 = ((y1 - y0) * (pz - z0) - (py - y0) * (z1 - z0)) / d
                    if l1 >= 0.0 and l2 >= 0.0 and l1 + l2 <= 1.0:
                        cover[j, k] = True

    _K.update(step=step, gather=gather, gather_cells=gather_cells, scatter=scatter, restrict=restrict, link_cuts=link_cuts, shadow=shadow)
    if os.environ.get("LOADPATH_WINDTUNNEL_STEP", "rows") != "cells":
        from .fieldjit import _kernel

        c19, opp19 = np.ascontiguousarray(C19, np.int64), np.ascontiguousarray(OPP19, np.int64)

        built = {}

        def rows(*a):
            # a[5] is the closure (2: WALE), a[18] whether the level has a sponge
            key = (a[5] == 2, bool(a[18]))
            if key not in built:
                built[key] = _kernel(_step_source(*key))
            return built[key](*a, c19, opp19)

        _K["step"] = rows
        _K["step_cells"] = step
    return _K


def _collide_rows(wale, sponge):
    """The collision of one row of cells as straight-line source: every
    direction written out with its own literal velocities, no branches (the
    limits and cases are min, max and selects), reading and writing only the
    row buffers. The arithmetic is what the cell-by-cell kernel does; what
    goes is its bookkeeping (a loop over directions reading the velocity and
    weight tables, an index wrapped by a remainder for every neighbour),
    which cost more than the physics."""
    C, W = C19, W19
    L = ["z = U(zz)"]
    L += [f"v{i} = bi[{i}, z]" for i in range(19)]

    def signed(k, pre="v"):
        t = " ".join(("+ " if C[i][k] > 0 else "- ") + f"{pre}{i}" for i in range(19) if C[i][k] != 0)
        return t[2:] if t.startswith("+ ") else t

    def second(a, b):
        t = " ".join(("+ " if C[i][a] * C[i][b] > 0 else "- ") + f"v{i}" for i in range(19) if C[i][a] * C[i][b] != 0)
        return t[2:] if t.startswith("+ ") else t

    L.append("dr = " + " + ".join(f"v{i}" for i in range(19)))
    L += [f"jx = {signed(0)}", f"jy = {signed(1)}", f"jz = {signed(2)}"]
    L += [f"pxx = {second(0, 0)}", f"pyy = {second(1, 1)}", f"pzz = {second(2, 2)}",
          f"pxy = {second(0, 1)}", f"pxz = {second(0, 2)}", f"pyz = {second(1, 2)}"]
    L += ["r = 1.0 + dr",
          "Fx = gx * r + bF[0, z]", "Fy = gy * r + bF[1, z]", "Fz = gz * r + bF[2, z]",
          "vx = (jx + 0.5 * Fx) / r", "vy = (jy + 0.5 * Fy) / r", "vz = (jz + 0.5 * Fz) / r",
          "usq = vx * vx + vy * vy + vz * vz",
          "third = dr / 3.0",
          "pxx += vx * Fx - third - r * vx * vx",
          "pyy += vy * Fy - third - r * vy * vy",
          "pzz += vz * Fz - third - r * vz * vz",
          "pxy += 0.5 * (vx * Fy + vy * Fx) - r * vx * vy",
          "pxz += 0.5 * (vx * Fz + vz * Fx) - r * vx * vz",
          "pyz += 0.5 * (vy * Fz + vz * Fy) - r * vy * vz"]
    if wale:
        g = [[f"g{c}{a}" for a in range(3)] for c in range(3)]
        L += [f"g{c}{a} = bg[{3 * c + a}, z]" for c in range(3) for a in range(3)]
        L.append("g2t = " + " + ".join(f"{g[a][c]} * {g[c][a]}" for a in range(3) for c in range(3)))
        L += ["ss = 0.0", "sd = 0.0"]
        for a in range(3):
            for c in range(3):
                L.append(f"s_ac = 0.5 * ({g[a][c]} + {g[c][a]})")
                L.append("ss += s_ac * s_ac")
                g2 = " + ".join(f"{g[a][k]} * {g[k][c]}" for k in range(3))
                g2r = " + ".join(f"{g[c][k]} * {g[k][a]}" for k in range(3))
                L.append(f"d_ac = 0.5 * (({g2}) + ({g2r}))" + (" - g2t / 3.0" if a == c else ""))
                L.append("sd += d_ac * d_ac")
        # ss^2.5 + sd^1.25 and sd^1.5 by square roots (the same numbers to rounding, and no calls to pow)
        L += ["rsd = np.sqrt(sd)",
              "den = ss * ss * np.sqrt(ss) + sd * np.sqrt(rsd)",
              "tl = tau0 + (3.0 * csq * sd * rsd / max(den, 1e-30) if den > 1e-30 else 0.0)"]
    else:
        L += ["Q = np.sqrt(pxx * pxx + pyy * pyy + pzz * pzz + 2.0 * (pxy * pxy + pxz * pxz + pyz * pyz))",
              "tl = 0.5 * (tau0 + np.sqrt(tau0 * tau0 + 18.0 * 1.41421356 * csq * Q / r)) if smag else tau0"]
    L += ["om = 1.0 - 1.0 / tl",
          "tr = (pxx + pyy + pzz) / 3.0",
          "omb = 1.0 - 1.0 / max(tl, 1.0)",
          "uF = vx * Fx + vy * Fy + vz * Fz",
          "dxx = pxx - tr", "dyy = pyy - tr", "dzz = pzz - tr"]
    if sponge:
        L += ["s = bs[z]", "sp = max(s, 0.0)", "sn = max(-s, 0.0)"]
    for i in range(19):
        cx, cy, cz = (int(v) for v in C[i])
        w = repr(float(W[i]))

        def lin(x, y, z_):
            t = " ".join(("+ " if c > 0 else "- ") + n for c, n in ((cx, x), (cy, y), (cz, z_)) if c)
            return (t[2:] if t.startswith("+ ") else t) if t else "0.0"
        L.append(f"cu = {lin('vx', 'vy', 'vz')}")
        L.append(f"feq = {w} * (dr + r * (3.0 * cu + 4.5 * cu * cu - 1.5 * usq))")
        q = [n for c, n in ((cx * cx, "dxx"), (cy * cy, "dyy"), (cz * cz, "dzz")) if c]
        q2 = [("+ " if c > 0 else "- ") + n for c, n in ((cx * cy, "pxy"), (cx * cz, "pxz"), (cy * cz, "pyz")) if c]
        shear = (" + ".join(q) if q else "0.0") + (f" + 2.0 * (0.0 {' '.join(q2)})" if q2 else "")
        L.append(f"fne = {w} * 4.5 * ({shear})")
        L.append(f"fnb = {w} * 4.5 * tr * {float(cx * cx + cy * cy + cz * cz - 1)!r}")
        L.append(f"cF = {lin('Fx', 'Fy', 'Fz')}")
        L.append(f"out = feq + om * fne + omb * fnb + 0.5 * {w} * (3.0 * (cF - uF) + 9.0 * cu * cF)")
        if sponge:
            # toward the free stream where s > 0 (inlet and sides), toward ambient
            # pressure at the local velocity where s < 0 (outlet); one of the two is zero
            L.append(f"cuf = {float(cx)!r} * ufar")
            L.append(f"out = out + sp * ({w} * (3.0 * cuf + 4.5 * cuf * cuf - 1.5 * ufar * ufar) - out) "
                     f"+ sn * ({w} * (3.0 * cu + 4.5 * cu * cu - 1.5 * usq) - out)")
        L.append(f"bo[{i}, z] = out")
    L += ["bo[19, z] = dr", "bo[20, z] = vx", "bo[21, z] = vy", "bo[22, z] = vz", "bo[23, z] = pxx", "bo[24, z] = pyy",
          "bo[25, z] = pzz", "bo[26, z] = pxy", "bo[27, z] = pxz", "bo[28, z] = pyz", "bo[29, z] = tl"]
    return L


def _pull_rows():
    """The streaming of one row as source: (text set once a row, text per
    cell). Each direction is written out with its own literal offsets,
    reading its source row through a view taken once a row, so a link costs
    a byte compared and a number copied; the wall's arithmetic (interpolated
    bounce-back, the momentum handed over) is the same as it was, spelled
    for that one link."""
    C, OPP = C19, OPP19
    rows = ["            xm = x - 1", "            xp = x + 1", "            ym = y - 1", "            yp = y + 1",
            "            if p0:", "                if xm < 0:", "                    xm += nx",
            "                if xp >= nx:", "                    xp -= nx",
            "            else:", "                xm = max(xm, 0)", "                xp = min(xp, nx - 1)",
            "            if p1:", "                if ym < 0:", "                    ym += ny",
            "                if yp >= ny:", "                    yp -= ny",
            "            else:", "                ym = max(ym, 0)", "                yp = min(yp, ny - 1)"]
    name = {1: "m", 0: "o", -1: "p"}        # the source is at x - c
    idx = {("x", 1): "xm", ("x", 0): "x", ("x", -1): "xp", ("y", 1): "ym", ("y", 0): "y", ("y", -1): "yp"}
    combos = sorted({(int(c[0]), int(c[1])) for c in C})
    for cx, cy in combos:
        v = name[cx] + name[cy]
        rows.append(f"            N{v} = ntype[{idx['x', cx]}, {idx['y', cy]}]")
        rows.append(f"            F{v} = f[{idx['x', cx]}, {idx['y', cy]}]")
    cell = ["                zm = z - 1", "                zp = z + 1",
            "                if p2:", "                    if zm < 0:", "                        zm += nz",
            "                    if zp >= nz:", "                        zp -= nz",
            "                else:", "                    zm = max(zm, 0)", "                    zp = min(zp, nz - 1)"]
    P = " " * 16
    for i in range(19):
        cx, cy, cz = (int(v) for v in C[i])
        j = int(OPP[i])
        v = name[cx] + name[cy]
        sz = {1: "zm", 0: "z", -1: "zp"}[cz]
        if i == 0:
            cell.append(f"{P}bi[0, z] = Foo[z, 0]")
            continue
        jx, jy, jz = (int(c) for c in C[j])
        cell += [f"{P}if N{v}[{sz}] == 1:",
                 f"{P}    fj = Foo[z, {j}]",
                 f"{P}    val = wall(f, ntype, q, b, fj, Foo[z, {i}], {j}, x + {cx}, y + {cy}, z + {cz}, p0, p1, p2)",
                 f"{P}    bi[{i}, z] = val",
                 f"{P}    if do_force:",
                 f"{P}        mom = fj + val"]
        for comp, cj, acc in (("mx", jx, "fxl"), ("my", jy, "fyl"), ("mz", jz, "fzl")):
            cell.append(f"{P}        {comp} = {float(cj)!r} * mom")
            cell.append(f"{P}        {acc} += {comp}")
        cell += [f"{P}        if b >= 0:",
                 f"{P}            fb[b, 0] += mx", f"{P}            fb[b, 1] += my", f"{P}            fb[b, 2] += mz",
                 f"{P}else:",
                 f"{P}    bi[{i}, z] = F{v}[{sz}, {i}]"]
    return "\n".join(rows) + "\n", "\n".join(cell) + "\n"


_STEP_HEAD = '''import numpy as np
import numba as nb

U = np.uint64


@nb.njit(fastmath=True, cache=True)
def wall(f, ntype, q, b, fj, fi, j, tx, ty, tz, p0, p1, p2):
    """What comes back along a link that meets a wall a fraction qq of the
    link away (interpolated bounce-back; halfway where the cut is not
    known). fj, fi: this cell's populations toward the wall and away from
    it; (tx, ty, tz): the cell behind."""
    qq = 0.5
    if b >= 0:
        qq = q[b, j]
    val = fj
    if qq < 0.5:
        nx, ny, nz = ntype.shape
        if p0:
            if tx < 0:
                tx += nx
            elif tx >= nx:
                tx -= nx
        if p1:
            if ty < 0:
                ty += ny
            elif ty >= ny:
                ty -= ny
        if p2:
            if tz < 0:
                tz += nz
            elif tz >= nz:
                tz -= nz
        if 0 <= tx < nx and 0 <= ty < ny and 0 <= tz < nz and ntype[tx, ty, tz] != 1:
            val = 2.0 * qq * fj + (1.0 - 2.0 * qq) * f[tx, ty, tz, j]
    elif qq > 0.5:
        val = fj / (2.0 * qq) + (1.0 - 1.0 / (2.0 * qq)) * fi
    return val


@nb.njit(parallel=True, fastmath=True, cache=True)
def kernel(f, fn, ntype, per, tau0, sgs, csq, u, un, rho, P, tau, bid, q, facc, fb, do_force,
           sig, has_sig, ufar, gx, gy, gz, frc, has_frc, C, OPP):
    nx, ny, nz = ntype.shape
    smag = sgs == 1
    p0 = per[0]
    p1 = per[1]
    p2 = per[2]
    for x in nb.prange(nx):
        # one row of cells at a time, in three passes: pull the populations
        # into bi (walls, link by link: branches), collide bi into bo with no
        # branches, write the row back
        bi = np.zeros((19, nz))
        bo = np.zeros((30, nz))        # the 19 populations out, then the 11 moments
        bg = np.zeros((9, nz))
        bF = np.zeros((3, nz))
        bs = np.zeros(nz)
        fxl = 0.0
        fyl = 0.0
        fzl = 0.0
        for y in range(ny):
#ROWS#
            for z in range(nz):
                t = ntype[x, y, z]
                if t != 0:
                    for i in range(19):
                        bi[i, z] = 0.0
                    for i in range(9):
                        bg[i, z] = 0.0
                    for i in range(3):
                        bF[i, z] = 0.0
                    bs[z] = 0.0
                    continue
                b = bid[x, y, z]
#PULL#
                if has_frc:
                    bF[0, z] = frc[0, x, y, z]
                    bF[1, z] = frc[1, x, y, z]
                    bF[2, z] = frc[2, x, y, z]
                if has_sig:
                    bs[z] = sig[x, y, z]
                if sgs == 2:
                    # velocity gradient by central differences of the last step's velocity;
                    # a solid neighbour is at rest
                    for a in range(3):
                        xp = x + (1 if a == 0 else 0)
                        yp = y + (1 if a == 1 else 0)
                        zp = z + (1 if a == 2 else 0)
                        xm = x - (1 if a == 0 else 0)
                        ym = y - (1 if a == 1 else 0)
                        zm = z - (1 if a == 2 else 0)
                        if per[0]:
                            if xp < 0:
                                xp += nx
                            elif xp >= nx:
                                xp -= nx
                            if xm < 0:
                                xm += nx
                            elif xm >= nx:
                                xm -= nx
                        if per[1]:
                            if yp < 0:
                                yp += ny
                            elif yp >= ny:
                                yp -= ny
                            if ym < 0:
                                ym += ny
                            elif ym >= ny:
                                ym -= ny
                        if per[2]:
                            if zp < 0:
                                zp += nz
                            elif zp >= nz:
                                zp -= nz
                            if zm < 0:
                                zm += nz
                            elif zm >= nz:
                                zm -= nz
                        sp_ = ntype[xp, yp, zp] == 1
                        sm_ = ntype[xm, ym, zm] == 1
                        for c in range(3):
                            up = 0.0 if sp_ else u[c, xp, yp, zp]
                            um = 0.0 if sm_ else u[c, xm, ym, zm]
                            bg[3 * c + a, z] = 0.5 * (up - um)
'''

_STEP_TAIL = '''            for z in range(nz):
                t = ntype[x, y, z]
                if t == 0:
                    for i in range(19):
                        fn[x, y, z, i] = bo[i, z]
                    rho[x, y, z] = bo[19, z]
                    un[0, x, y, z] = bo[20, z]
                    un[1, x, y, z] = bo[21, z]
                    un[2, x, y, z] = bo[22, z]
                    for k in range(6):
                        P[k, x, y, z] = bo[23 + k, z]
                    tau[x, y, z] = bo[29, z]
                elif t == 3:
                    # open boundary: ambient pressure, the velocity of the node inward
                    ix = np.int64(x)
                    iy = np.int64(y)
                    iz = np.int64(z)
                    if x == 0:
                        ix = 1
                    elif x == nx - 1:
                        ix = nx - 2
                    if not per[1]:
                        if y == 0:
                            iy = 1
                        elif y == ny - 1:
                            iy = ny - 2
                    if not per[2]:
                        if z == 0:
                            iz = 1
                        elif z == nz - 1:
                            iz = nz - 2
                    vx = u[0, ix, iy, iz]
                    vy = u[1, ix, iy, iz]
                    vz = u[2, ix, iy, iz]
                    usq = vx * vx + vy * vy + vz * vz
                    for i in range(19):
                        cu = C[i, 0] * vx + C[i, 1] * vy + C[i, 2] * vz
                        fn[x, y, z, i] = W[i] * (3.0 * cu + 4.5 * cu * cu - 1.5 * usq)
                    rho[x, y, z] = 0.0
                    un[0, x, y, z] = vx
                    un[1, x, y, z] = vy
                    un[2, x, y, z] = vz
                    for k in range(6):
                        P[k, x, y, z] = 0.0
                    tau[x, y, z] = tau0
        if do_force:
            facc[x, 0] += fxl
            facc[x, 1] += fyl
            facc[x, 2] += fzl
'''


# the rows the velocity gradient reads, taken once a row. The neighbour below
# the first cell is the last one (round a periodic axis; the cell-by-cell
# kernel's index of -1 reads the same cell on a closed one, where no fluid
# cell sits on the face)
_GRADIENT_VIEWS = """            gxm = x - 1
            if gxm < 0:
                gxm += nx
            gxp = x + 1
            if gxp >= nx:
                gxp -= nx if p0 else 1
            gym = y - 1
            if gym < 0:
                gym += ny
            gyp = y + 1
            if gyp >= ny:
                gyp -= ny if p1 else 1
            GNxm = ntype[gxm, y]
            GNxp = ntype[gxp, y]
            GNym = ntype[x, gym]
            GNyp = ntype[x, gyp]
            GNoo = ntype[x, y]
""" + "".join(f"            GU{c}{n} = u[{c}, {i}]\n" for c in range(3)
              for n, i in (("xm", "gxm, y"), ("xp", "gxp, y"), ("ym", "x, gym"), ("yp", "x, gyp"), ("oo", "x, y")))


def _gradient_rows():
    """The velocity gradient of one cell for the WALE closure, as source:
    central differences of the last step's velocity, a solid neighbour at
    rest, each of the six neighbours read through its row's view (the
    cell-by-cell kernel builds and wraps three indices for every one)."""
    P = " " * 16
    L = ["gzm = z - 1", "if gzm < 0:", "    gzm += nz", "gzp = z + 1", "if gzp >= nz:", "    gzp -= nz if p2 else 1"]
    for a, (m, p_, zm, zp) in enumerate((("xm", "xp", "z", "z"), ("ym", "yp", "z", "z"), ("oo", "oo", "gzm", "gzp"))):
        L += [f"sp_ = GN{p_}[{zp}] == 1", f"sm_ = GN{m}[{zm}] == 1"]
        for c in range(3):
            L += [f"up = 0.0 if sp_ else GU{c}{p_}[{zp}]", f"um = 0.0 if sm_ else GU{c}{m}[{zm}]",
                  f"bg[{3 * c + a}, z] = 0.5 * (up - um)"]
    return "".join(P + ln + "\n" for ln in L)


def _step_source(wale, sponge):
    """Source of the step kernel for one closure (WALE or not) and one kind
    of level (with a sponge or without): the row passes of _STEP_HEAD and
    _STEP_TAIL round the generated collision. One kernel for each of the
    four, compiled when a level first asks for it, so a run compiles only
    what it uses."""
    mid = "            for zz in range(nz):\n" + "\n".join(" " * 16 + ln for ln in _collide_rows(wale, sponge)) + "\n"
    head = _STEP_HEAD.replace("U = np.uint64\n", "U = np.uint64\nW = np.array(" + repr([float(w) for w in W19]) + ")\n")
    rows, cell = _pull_rows()
    if wale:
        rows += _GRADIENT_VIEWS
    head = head.replace("#ROWS#\n", rows).replace("#PULL#\n", cell)
    head = head[:head.index("                if sgs == 2:\n")]
    if wale:
        head += _gradient_rows()
    return _blocked(head, mid, _STEP_TAIL)


_BLOCK = 64


def _blocked(head, mid, tail):
    """The three passes over a row, taken a block of _BLOCK cells at a time
    with buffers of that fixed size. With buffers as long as the row (a
    length known only at run time) the compiler cannot show that the
    collision's rows of numbers do not overlap, and leaves it scalar; with
    a fixed size it turns the collision into vector instructions. The
    arithmetic and its order are unchanged."""
    import re

    buf = re.compile(r"\b(bi|bo|bg|bF)\[([^\]]*), z\]")

    def local(text):                  # the buffers are indexed from the block's start
        return buf.sub(r"\1[\2, zb]", text).replace("bs[z]", "bs[zb]")

    first = "            for z in range(nz):\n"
    k = head.index(first)
    top, pull = head[:k], head[k + len(first):]
    for name in ("bi", "bo", "bg", "bF"):
        top = re.sub(rf"{name} = np.zeros\(\((\d+), nz\)\)", rf"{name} = np.zeros((\1, {_BLOCK}))", top)
    top = top.replace("bs = np.zeros(nz)", f"bs = np.zeros({_BLOCK})")
    assert ", nz))" not in top and "np.zeros(nz)" not in top
    k = tail.index("        if do_force:\n")
    write, end = tail[:k], tail[k:]
    assert write.startswith(first) and mid.startswith("            for zz in range(nz):\n")
    body = (f"            for z0 in range(0, nz, {_BLOCK}):\n"
            f"                z1 = min(z0 + {_BLOCK}, nz)\n"
            "                for z in range(z0, z1):\n"
            "                    zb = z - z0\n"
            + "".join("    " + ln + "\n" for ln in local(pull).splitlines())
            + "                for zz in range(z1 - z0):\n"
            + "".join("    " + ln + "\n" for ln in mid.splitlines()[1:])
            + "                for z in range(z0, z1):\n"
            "                    zb = z - z0\n"
            + "".join("    " + ln + "\n" for ln in local(write[len(first):]).splitlines()))
    return top + body + end


def frontal_area(triangles, pitch):
    """Area (mm^2) of a mesh's shadow along x: what the stream sees, exact to
    the pixel `pitch` (a voxel count at the flow grid's cell is 1-2% off on
    a round body, and the drag coefficient is quoted on it)."""
    K = _kernels()
    tri = np.ascontiguousarray(triangles, np.float64)
    lo = tri[:, :, 1:].reshape(-1, 2).min(axis=0)
    hi = tri[:, :, 1:].reshape(-1, 2).max(axis=0)
    n = np.maximum(np.ceil((hi - lo) / pitch).astype(int), 1)
    cover = np.zeros((int(n[0]), int(n[1])), np.bool_)
    K["shadow"](tri, lo, float(pitch), cover)
    return float(cover.sum()) * pitch * pitch


def wall_links(solid, mesh_triangles, origin, h, periodic=(False, False, False)):
    """Boundary nodes of a voxel solid and the cut of each of their wall
    links by the surface. Returns (bid: index of each boundary node or -1,
    q: (n, 19) link fractions, -1 where the link meets no wall, xyz of the
    nodes). A link the triangles miss (a seam) is cut halfway."""
    K = _kernels()
    per = np.asarray(periodic, bool)
    fluid = ~solid
    hit = np.zeros(solid.shape, bool)
    nbs = []
    for i in range(1, 19):
        c = C19[i]
        nb_ = np.roll(solid, shift=(-c[0], -c[1], -c[2]), axis=(0, 1, 2))
        for a in range(3):
            if not per[a] and c[a]:
                sl = [slice(None)] * 3
                sl[a] = -1 if c[a] == 1 else 0
                nb_[tuple(sl)] = False
        m = fluid & nb_
        nbs.append(m)
        hit |= m
    idx = np.argwhere(hit)
    bid = np.full(solid.shape, -1, np.int32)
    bid[hit] = np.arange(len(idx), dtype=np.int32)
    q = np.full((len(idx), 19), -1.0, np.float32)
    for i, m in zip(range(1, 19), nbs):
        q[bid[m], i] = 2.0
    if len(idx):
        K["link_cuts"](np.ascontiguousarray(mesh_triangles, np.float64), np.asarray(origin, np.float64), float(h),
                       solid.astype(np.int8), bid, q, per)
    missed = q > 1.5
    q[missed] = 0.5
    np.clip(q, -1.0, 1.0, out=q)
    q[(q >= 0) & (q < 0.02)] = 0.02
    return bid, q, origin + idx * h, int(missed.sum())


def _erode(mask, per):
    """Cells of `mask` whose 26 neighbours are all in it (outside the array
    counts as out, except across a periodic axis)."""
    m = mask
    for ax in range(3):
        w = [(0, 0)] * 3
        w[ax] = (1, 1)
        m = np.pad(m, w, mode="wrap" if per[ax] else "constant")
    out = np.ones(mask.shape, bool)
    nx, ny, nz = mask.shape
    for dx in range(3):
        for dy in range(3):
            for dz in range(3):
                out &= m[dx:dx + nx, dy:dy + ny, dz:dz + nz]
    return out


def _depth(mask, per):
    """Distance (cells) of each cell of `mask` from the nearest cell outside
    it; the array's edge is outside, except across a periodic axis."""
    from scipy.ndimage import distance_transform_edt

    m = mask
    pads = []
    for ax in range(3):
        w = [(0, 0)] * 3
        n = 1 if not per[ax] else min(8, mask.shape[ax])
        w[ax] = (n, n)
        pads.append(n)
        m = np.pad(m, w, mode="wrap" if per[ax] else "constant")
    d = distance_transform_edt(m)
    return d[pads[0]:pads[0] + mask.shape[0], pads[1]:pads[1] + mask.shape[1], pads[2]:pads[2] + mask.shape[2]]


def layout(lo, hi, h, ref, *, domain=(12.0, 10.0, 24.0), fine=(0.5, 0.5, 1.5), buffer=(8, 8, 16), span_axis=None,
           max_levels=9, shells=0, shell=6, voxelize=None,
           domain_ref=None, include=(), max_cells=4e7):
    """Nested boxes round a body with bounds lo..hi (mm, the stream along
    +x): the finest (cell h) reaches `fine` x ref past the body (upstream,
    to the sides, downstream), each coarser level `buffer` of its own cells
    past its child, and the coarsest fills the tunnel, `domain` x ref past
    the body. With `span_axis` (1 or 2) the body runs the whole width of the
    tunnel on that axis, periodic (the cell is then adjusted, by up to a
    quarter, so the span is a whole number of the coarsest cells).
    `domain_ref`: the length the domain is measured in, if not ref.
    `include`: [(lo, hi, cell)] boxes that must lie inside a level of that
    cell or finer (the coarsest such): a rotor disk far wider than the body
    is thick wants a tenth of its radius, not the body's cell.

    `shells` more levels, each half the cell again, hug the surface: a level
    of cells within `shell` of its parent's cells of the body, which is
    where a boundary layer needs them and a box would spend them on the
    free stream (`voxelize(h, corner, dims)` gives the body's cells on a
    grid). h is the boxes' finest cell; the finest shell's is h / 2^shells.

    Returns a list, coarsest first, of {"h", "origin" (centre of cell
    (0,0,0)), "shape", "off" (this level's corner in its parent's cells),
    "mask" (which of the parent's cells under it the level covers, or None
    for all)}."""
    lo = np.asarray(lo, float)
    hi = np.asarray(hi, float)
    h0 = float(h)
    per = np.zeros(3, bool)
    if span_axis is not None:
        per[span_axis] = True

    def pads(u, s, d):
        return np.array([u, s, s], float), np.array([d, s, s], float)

    best = None
    for n in range(1, max_levels + 1):
        H = h0 * 2 ** (n - 1)
        h = h0
        dref = ref if domain_ref is None else domain_ref
        pl, ph = pads(domain[0] * dref, domain[1] * dref, domain[2] * dref)
        d_lo = lo - pl
        d_hi = hi + ph
        if span_axis is not None:
            # the span is a whole number of the coarsest cells: the cell gives a little
            span = hi[span_axis] - lo[span_axis]
            m = int(round(span / H))
            if m < 2:
                break
            H = span / m
            h = H / 2 ** (n - 1)
            if abs(h / h0 - 1.0) > 0.25:
                continue
            d_lo[span_axis], d_hi[span_axis] = lo[span_axis], hi[span_axis]
        dims0 = np.ceil((d_hi - d_lo) / H - 1e-9).astype(int)
        # desired boxes, finest first
        boxes = []
        # a shell level needs its own reach and three cells of its parent round it
        fmin = (shell + 4) * h if shells else 0.0
        pl, ph = pads(*[max(v * ref, fmin) for v in fine])
        b_lo, b_hi = lo - pl, hi + ph
        hl = h
        todo = list(include)
        for lev in range(n - 1):
            for inc in list(todo):
                if 2.0 * hl > inc[2] or lev == n - 2:
                    b_lo = np.minimum(b_lo, np.asarray(inc[0], float) - 4.0 * hl)
                    b_hi = np.maximum(b_hi, np.asarray(inc[1], float) + 4.0 * hl)
                    todo.remove(inc)
            boxes.append((b_lo.copy(), b_hi.copy()))
            hl *= 2.0
            pl, ph = pads(*[v * hl for v in buffer])
            b_lo, b_hi = b_lo - pl, b_hi + ph
        boxes = boxes[::-1]                              # coarsest refined level first
        levels = [{"h": H, "origin": d_lo + 0.5 * H, "shape": tuple(int(v) for v in dims0), "off": None,
                   "corner": d_lo.copy(), "mask": None}]
        ok = True
        for k, (b_lo, b_hi) in enumerate(boxes):
            par = levels[-1]
            hp = par["h"]
            a = np.floor((b_lo - par["corner"]) / hp + 1e-9).astype(int)
            b = np.ceil((b_hi - par["corner"]) / hp - 1e-9).astype(int)
            pshape = np.array(par["shape"])
            for ax in range(3):
                if per[ax]:
                    a[ax], b[ax] = 0, pshape[ax]
            free = ~per
            if np.any(a[free] < 3) or np.any(b[free] > pshape[free] - 3) or np.any((b - a)[free] < 6):
                ok = False
                break
            levels.append({"h": hp / 2.0, "origin": par["corner"] + a * hp + 0.25 * hp,
                           "shape": tuple(int(v) for v in 2 * (b - a)), "off": a, "corner": par["corner"] + a * hp,
                           "mask": None})
        if not ok:
            break
        best = levels
    if best is None:
        raise ValueError("the tunnel is too small for the body at this resolution: fewer cells or a larger domain")
    if sum(float(np.prod(L["shape"])) for L in best) > max_cells:
        raise ValueError("the tunnel would hold over %.0f million cells: fewer cells across the body" % (max_cells / 1e6))
    for _ in range(int(shells)):
        from scipy.ndimage import distance_transform_edt

        par = best[-1]
        hp = par["h"]
        pshape = np.array(par["shape"])
        body = voxelize(hp, par["corner"], pshape)
        if per.any():
            # distance across the periodic span: the body runs the whole of it
            dist = distance_transform_edt(~np.concatenate([body] * 3, axis=span_axis))
            n_s = pshape[span_axis]
            sl = [slice(None)] * 3
            sl[span_axis] = slice(n_s, 2 * n_s)
            dist = dist[tuple(sl)]
        else:
            dist = distance_transform_edt(~body)
        mask = dist <= shell
        idx = np.argwhere(mask)
        a, b = idx.min(axis=0), idx.max(axis=0) + 1
        for ax in range(3):
            if per[ax]:
                a[ax], b[ax] = 0, pshape[ax]
        free = ~per
        if np.any(a[free] < 3) or np.any(b[free] > pshape[free] - 3):
            raise ValueError("a shell level does not fit in its parent: a larger `fine` margin or a thinner shell")
        best.append({"h": hp / 2.0, "origin": par["corner"] + a * hp + 0.25 * hp,
                     "shape": tuple(int(v) for v in 2 * (b - a)), "off": a, "corner": par["corner"] + a * hp,
                     "mask": np.ascontiguousarray(mask[a[0]:b[0], a[1]:b[1], a[2]:b[2]])})
    return best


class _Level:
    pass


class Tunnel:
    """Nested lattice Boltzmann grids round a voxel body.

    levels: from `layout`. solid: bool array on the finest level. nu: the
    finest level's lattice viscosity. u0: the stream's lattice speed (+x).
    q, bid: wall link cuts from `wall_links` (or None: halfway walls).
    periodic: axes the tunnel wraps on (a body spanning the tunnel).
    open_far: every outer face at ambient pressure, open both ways.
    """

    def __init__(self, levels, solid, nu, u0, *, bid=None, q=None, periodic=(False, False, False), sgs="wale",
                 c_sgs=None, sponge=6, sponge_strength=0.15, force=None, open_far=False):
        self.K = _kernels()
        self.per = np.asarray(periodic, bool)
        self.u0 = float(u0)
        self.ufar = 0.0
        self.gx = 0.0
        self.gy = 0.0
        self.gz = 0.0
        self.sgs = {"none": SGS_NONE, "smagorinsky": SGS_SMAGORINSKY, "wale": SGS_WALE}[sgs]
        if c_sgs is None:
            c_sgs = 0.325 if sgs == "wale" else 0.1
        self.csq = float(c_sgs) ** 2
        n = len(levels)
        self.levels = []
        for k, spec in enumerate(levels):
            L = _Level()
            L.h = spec["h"]
            L.origin = np.asarray(spec["origin"], float)
            L.shape = tuple(spec["shape"])
            L.off = None if spec["off"] is None else np.asarray(spec["off"], np.int64)
            # viscosity is the same in physical units: the lattice value doubles each refinement
            L.nu = nu / 2 ** (n - 1 - k)
            L.tau0 = 0.5 + 3.0 * L.nu
            # WALE is for the level that holds the wall: it vanishes in a laminar boundary
            # layer, where Smagorinsky's would thicken it. It also vanishes in any plane shear,
            # so a strut's separated shear layer, crossing to a coarser level that cannot
            # resolve it, would meet no subgrid viscosity at all and blow up there. The levels
            # away from the wall take Smagorinsky's.
            L.sgs, L.csq = self.sgs, self.csq
            if self.sgs == SGS_WALE and k < n - 1:
                L.sgs, L.csq = SGS_SMAGORINSKY, 0.1 ** 2
            L.f = np.zeros(L.shape + (19,), np.float32)
            L.fn = np.zeros(L.shape + (19,), np.float32)
            L.u = np.zeros((3,) + L.shape, np.float32)
            L.un = np.zeros((3,) + L.shape, np.float32)
            L.rho = np.zeros(L.shape, np.float32)
            L.P = np.zeros((6,) + L.shape, np.float32)
            L.tau = np.full(L.shape, L.tau0, np.float32)
            L.ntype = np.zeros(L.shape, np.int8)
            L.bid = np.full(L.shape if k == n - 1 and bid is not None else (1, 1, 1), -1, np.int32)
            L.q = np.zeros((1, 19), np.float32)
            L.sig = np.zeros((1, 1, 1), np.float32)
            L.has_sig = False
            L.frc = np.zeros((3, 1, 1, 1), np.float32)
            L.has_frc = False
            L.facc = np.zeros((L.shape[0], 3))
            L.fb = np.zeros((1, 3))
            self.levels.append(L)
        fine = self.levels[-1]
        if bid is not None:
            fine.bid = bid
            fine.q = q
            fine.fb = np.zeros((len(q), 3))
        else:
            fine.bid = np.full(fine.shape, -1, np.int32)
        # the outer boundary: free stream on the inlet and the sides, ambient pressure out
        L0 = self.levels[0]
        # (open_far: ambient pressure all round, the air free to come and go, for a rotor
        # in still air, which draws its inflow from every side)
        edge = OUT if open_far else FAR
        L0.ntype[0] = edge
        L0.ntype[-1] = OUT
        for ax in (1, 2):
            if not self.per[ax]:
                sl = [slice(None)] * 3
                sl[ax] = 0
                L0.ntype[tuple(sl)] = edge
                sl[ax] = -1
                L0.ntype[tuple(sl)] = edge
        self._far = L0.ntype == FAR
        if sponge:
            ns = int(sponge)
            sig = np.zeros(L0.shape, np.float32)
            d = np.full(L0.shape, 1e9, np.float32)
            ix = np.arange(L0.shape[0], dtype=np.float32)[:, None, None]
            d = np.minimum(d, ix)
            for ax in (1, 2):
                if not self.per[ax]:
                    i = np.arange(L0.shape[ax], dtype=np.float32)
                    i = np.minimum(i, L0.shape[ax] - 1 - i)
                    sh = [1, 1, 1]
                    sh[ax] = -1
                    d = np.minimum(d, i.reshape(sh))
            s_in = sponge_strength * np.clip(1.0 - d / ns, 0.0, 1.0) ** 2
            dout = (L0.shape[0] - 1 - ix) * np.ones(L0.shape, np.float32)
            s_out = sponge_strength * np.clip(1.0 - dout / ns, 0.0, 1.0) ** 2
            sig = np.where(s_in > 0, -s_in if open_far else s_in, -s_out).astype(np.float32)
            L0.sig = sig
            L0.has_sig = True
        # coupling between each level and its child
        for k in range(n - 1):
            par, ch = self.levels[k], self.levels[k + 1]
            a = ch.off
            cs = np.array(ch.shape) // 2
            mask = levels[k + 1].get("mask")
            mask = np.ones(tuple(cs), bool) if mask is None else np.asarray(mask, bool)
            act = mask.repeat(2, axis=0).repeat(2, axis=1).repeat(2, axis=2)
            # the child's rim: its outermost layer of cells (any with a lattice neighbour
            # outside the level), refilled from the parent before each of its steps
            m = act
            for ax in range(3):
                w = [(0, 0)] * 3
                w[ax] = (1, 1)
                m = np.pad(m, w, mode="wrap" if self.per[ax] else "constant")
            rim = np.zeros(ch.shape, bool)
            nx, ny, nz = ch.shape
            for i in range(1, 19):
                c = C19[i]
                rim |= ~m[1 + c[0]:1 + c[0] + nx, 1 + c[1]:1 + c[1] + ny, 1 + c[2]:1 + c[2] + nz]
            rim &= act
            ch.ntype[~act] = SKIP
            ch.ntype[rim] = SKIP
            ch.gi = np.ascontiguousarray(np.argwhere(rim), np.int64)
            ch.M0 = np.zeros((len(ch.gi), 11), np.float32)
            ch.M1 = np.zeros((len(ch.gi), 11), np.float32)
            ch.Mh = np.zeros((len(ch.gi), 11), np.float32)
            ch.have = False
            # the parent under the child: computed by the parent for two cells in from
            # the child's edge, filled from the child for the next two, unused deeper
            # (by distance from the child's edge, so the parent's own cells keep clear of
            # the body on a shell's diagonals too; filled: every cell the parent's
            # streaming or the rim's interpolation can reach)
            e2 = _depth(mask, self.per) > 2.0
            reach = ~_erode(_erode(e2, self.per), self.per)
            sub = (slice(a[0], a[0] + cs[0]), slice(a[1], a[1] + cs[1]), slice(a[2], a[2] + cs[2]))
            par.ntype[sub][e2] = SKIP
            par.ci = np.ascontiguousarray(np.argwhere(e2 & reach) + a, np.int64)
            par.ci_all = np.ascontiguousarray(np.argwhere(e2) + a, np.int64)
        fine.ntype[solid] = SOLID
        if force is not None:
            self.set_force(*force)
        self.steps = 0                                   # finest-level steps taken
        self.sampling = False
        self.n_samples = 0

    def set_force(self, level, frc):
        """A body force field (3, nx, ny, nz) on one level, in that level's
        lattice units (force per cell per step)."""
        L = self.levels[level]
        L.frc = np.ascontiguousarray(frc, np.float32)
        L.has_frc = True

    def centres(self, level):
        """Cell-centre coordinates of a level along each axis (mm)."""
        L = self.levels[level]
        return [L.origin[a] + np.arange(L.shape[a]) * L.h for a in range(3)]

    def ripple(self, amplitude, axis=2, across=1):
        """Add a side velocity (along `across`) that varies as one sine wave
        along the periodic `axis`, to every level: divergence-free, and the
        only thing that tells a spanwise-uniform body's wake that the third
        dimension is there. Without it the wake of a strut stays exactly
        two-dimensional, which past Re 200 no real wake is."""
        length = self.levels[0].shape[axis] * self.levels[0].h
        for k, L in enumerate(self.levels):
            z = (np.arange(L.shape[axis]) + 0.5) * L.h
            dv = (amplitude * np.sin(2.0 * np.pi * z / length)).astype(np.float32)
            sh = [1, 1, 1]
            sh[axis] = -1
            dv = dv.reshape(sh) * (L.ntype == FLUID)
            for i in range(1, 19):
                if C19[i][across]:
                    L.f[..., i] += np.float32(3.0 * W19[i] * C19[i][across]) * dv

    def set_stream(self, u, accel=0.0, side=(0.0, 0.0)):
        """The far field's speed (lattice) and the acceleration that takes
        the whole stream there with it (a uniform body force: the fluid
        speeds up everywhere at once, as a body pulled through still air
        sees it, with no pressure wave to carry the news in from the far
        boundary)."""
        self.ufar = float(u)
        self.gx = float(accel)
        self.gy, self.gz = float(side[0]), float(side[1])
        L0 = self.levels[0]
        cu = C19[:, 0] * u
        feq = (W19 * (3.0 * cu + 4.5 * cu * cu - 1.5 * u * u)).astype(np.float32)
        for arr in (L0.f, L0.fn):
            arr[self._far] = feq
        for arr in (L0.u, L0.un):
            arr[0][self._far] = u

    def _step(self, k):
        L = self.levels[k]
        K = self.K
        n = len(self.levels)
        ch = self.levels[k + 1] if k + 1 < n else None
        if ch is not None and not ch.have:
            K["gather"](L.rho, L.u, L.P, L.tau, ch.gi, ch.off, self.per, ch.M0)
        # a lattice acceleration scales with the step, so it doubles each level up
        sc = 2 ** (n - 1 - k)
        gx = self.gx * sc
        K["step"](L.f, L.fn, L.ntype, self.per, L.tau0, L.sgs, L.csq, L.u, L.un, L.rho, L.P, L.tau,
                  L.bid, L.q, L.facc, L.fb, self.sampling and k == n - 1, L.sig, L.has_sig, self.ufar, gx,
                  self.gy * sc, self.gz * sc, L.frc, L.has_frc)
        L.f, L.fn = L.fn, L.f
        L.u, L.un = L.un, L.u
        if ch is None:
            self.steps += 1
            if self.sampling:
                self.n_samples += 1
            return
        K["gather"](L.rho, L.u, L.P, L.tau, ch.gi, ch.off, self.per, ch.M1)
        K["scatter"](ch.f, ch.rho, ch.u, ch.un, ch.P, ch.tau, ch.gi, ch.M0)
        self._step(k + 1)
        np.add(ch.M0, ch.M1, out=ch.Mh)
        ch.Mh *= 0.5
        K["scatter"](ch.f, ch.rho, ch.u, ch.un, ch.P, ch.tau, ch.gi, ch.Mh)
        self._step(k + 1)
        # the parent's state round the rim after this step is its state before the next
        ch.M0, ch.M1 = ch.M1, ch.M0
        ch.have = True
        K["restrict"](ch.rho, ch.u, ch.P, ch.tau, ch.ntype, L.f, L.rho, L.u, L.un, L.P, L.tau, L.ci, ch.off)

    def step(self):
        """One step of the coarsest level (2^(levels-1) of the finest)."""
        self._step(0)

    @property
    def fine_per_step(self):
        return 2 ** (len(self.levels) - 1)

    def force(self):
        """Mean force on the body per finest step since sampling began
        (lattice units of the finest level), and resets nothing."""
        return self.levels[-1].facc.sum(axis=0) / max(self.n_samples, 1)

    def fill_parents(self):
        """Fill every parent cell under a child from the child (for a look
        at the whole field: the run itself only fills the overlap)."""
        for k in range(len(self.levels) - 2, -1, -1):
            par, ch = self.levels[k], self.levels[k + 1]
            self.K["restrict"](ch.rho, ch.u, ch.P, ch.tau, ch.ntype, par.f, par.rho, par.u, par.un, par.P, par.tau,
                               par.ci_all, ch.off)

    def cells(self):
        return [int(np.prod(L.shape)) for L in self.levels]

    def work_per_step(self):
        """Cell updates per coarsest step."""
        return sum(int((L.ntype == FLUID).sum()) * 2 ** k for k, L in enumerate(self.levels))
