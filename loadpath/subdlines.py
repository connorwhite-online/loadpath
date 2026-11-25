"""
A subdivision surface's patches laid along its own lines: fewer, larger
faces (subd.py's patch writer).

subd.py writes every cage face that touches an extraordinary vertex (a
star point: three, five or more edges) as its own degree-7 Bezier patch,
and merges only the regular faces into rectangles. On a fitted cage most
faces are regular, but a hundred star points with four or five faces
each are four hundred patches, most of the STEP.

Here every edge line leaving a star point is drawn on until it meets
another star point or a line already drawn (a motorcycle graph, Eppstein
et al. 2008). The pieces between the lines are rectangles of cage faces
with star points only at their corners, and each is one B-spline surface:

- along the lines the limit surface is a cubic spline, exact, except on
  the edge that ends at a star point, where the spline's knots are halved
  toward the star point (1/2, 1/4, ... 1/16 of the edge) and the curve is
  fitted to the exact limit points and tangent planes;
- every line is fitted once, and the two rectangles beside it take the
  same curve as their edge (by refitting it on their own knots, exact,
  since their knots hold the line's): they share their edges exactly;
- inside, the poles are fitted to the exact limit points and tangent
  planes, so rectangles round a star point meet G1 to a fraction of a
  degree (reported); away from star points the fit is exact, and
  neighbours meet C2.

Returns the surfaces or None, and subd.py keeps its own patches when the
layout does not come out (a line that closes on itself, a fit outside the
tolerance).
"""
from __future__ import annotations

import numpy as np

DEPTH = 4            # knots halved this many times toward a star point (1/2 ... 1/16 of the edge)
TANGENT = 3.0        # weight of the tangent-plane rows against position rows (subd.TANGENT)
PIN = 1e4            # weight that holds a line on the limit point at every cage vertex


def _sp():
    import scipy.sparse as sp
    return sp


def _full(br, d=3):
    """Clamped knot vector from sorted breakpoints."""
    br = np.asarray(br, float)
    return np.concatenate([np.full(d, br[0]), br, np.full(d, br[-1])])


def basis(br, t, d=3, der=False):
    """Clamped B-spline basis over breakpoints `br` at t: (len(t), len(br)
    + d - 1), and its derivative when `der`."""
    kn = _full(br, d)
    t = np.clip(np.asarray(t, float), br[0], br[-1])
    nb = len(kn) - 1
    N = np.zeros((len(t), nb))
    for i in range(nb):
        N[:, i] = (kn[i] <= t) & (t < kn[i + 1])
    last = t >= br[-1]
    if last.any():
        N[last] = 0.0
        N[last, len(kn) - d - 2] = 1.0
    prev = None
    for k in range(1, d + 1):
        M = np.zeros((len(t), nb - k))
        for i in range(nb - k):
            a = kn[i + k] - kn[i]
            b = kn[i + k + 1] - kn[i + 1]
            if a > 0:
                M[:, i] += (t - kn[i]) / a * N[:, i]
            if b > 0:
                M[:, i] += (kn[i + k + 1] - t) / b * N[:, i + 1]
        prev, N = N, M
    if not der:
        return N
    D = np.zeros_like(N)
    for i in range(N.shape[1]):
        a = kn[i + d] - kn[i]
        b = kn[i + d + 1] - kn[i + 1]
        if a > 0:
            D[:, i] += d / a * prev[:, i]
        if b > 0:
            D[:, i] -= d / b * prev[:, i + 1]
    return N, D


def _halves(n, start, end):
    """Breakpoints 0..n, halved `start` times into the first span (toward
    a star point there) and `end` times into the last."""
    br = list(range(n + 1))
    for j in range(1, start + 1):
        br.append(2.0 ** -j)
    for j in range(1, end + 1):
        br.append(n - 2.0 ** -j)
    return np.unique(np.asarray(br, float))


def _one_sided(H, XG, NG):
    """Normals on the grids of faces beside a crease taken from the face's
    own side: across a crease the limit surface turns a corner, and the
    two-sided masks give the mean of the two sides' normals there."""
    cage = H.cage
    Q, QE = cage.Q, cage.QE
    sh = cage.sharpness
    on = cage.crease[QE].any(axis=1) | (sh[Q] > 0).any(axis=1)
    s = XG.shape[1] - 1
    NG = NG.copy()
    for q in np.flatnonzero(on):
        X = XG[q]
        # second-order differences inside the face, one-sided at its sides
        Xu = np.gradient(X, axis=0, edge_order=2)
        Xv = np.gradient(X, axis=1, edge_order=2)
        N = np.cross(Xu, Xv)
        N /= np.maximum(np.linalg.norm(N, axis=2, keepdims=True), 1e-300)
        # oriented as the face's own normals
        if np.einsum("ijk,ijk->", N, NG[q]) < 0:
            N = -N
        mask = np.zeros((s + 1, s + 1), bool)
        for k in range(4):
            if cage.crease[QE[q, k]]:
                side = [(slice(None), 0), (s, slice(None)), (slice(None), s), (0, slice(None))][k]
                mask[side] = True
        for k, (i, j) in enumerate(((0, 0), (s, 0), (s, s), (0, s))):
            if sh[Q[q, k]] > 0:
                mask[i, j] = True
        NG[q][mask] = N[mask]
    return NG


def motorcycles(cage):
    """Every edge line leaving a star point, drawn on one edge at a time
    (all at once, so none takes more than its share) until it meets a
    star point or a line already drawn. Returns the drawn edges (bool per
    cage edge)."""
    from collections import deque

    from .subd import _twins
    Q, QE = cage.Q, cage.QE
    tq, tk = _twins(cage)
    tq, tk = tq.reshape(-1, 4), tk.reshape(-1, 4)
    cut = np.zeros(len(cage.E), bool)
    hit = cage.valence != 4
    hit = hit.copy()
    ev = np.argwhere(cage.valence[Q] != 4)
    run = deque((int(q), int(k)) for q, k in ev)
    cr = getattr(cage, "crease", None)
    if cr is not None and cr.any():
        # creases are lines already drawn (a patch boundary: the surface
        # turns a corner there); where one ends or turns at a regular vertex
        # a line goes on straight from it, as from a star point
        cut |= cr
        hit |= cage.sharpness > 0
        for q, k in np.argwhere(cr[QE]):
            w = int(Q[q, (k + 1) % 4])
            if cage.valence[w] == 4:
                q2, k2 = int(tq[q, (k + 1) % 4]), int(tk[q, (k + 1) % 4])
                run.append((q2, (k2 + 1) % 4))
    while run:
        q, k = run.popleft()
        e = QE[q, k]
        if cut[e]:
            continue
        cut[e] = True
        w = int(Q[q, (k + 1) % 4])
        if hit[w]:
            continue
        hit[w] = True
        q2, k2 = int(tq[q, (k + 1) % 4]), int(tk[q, (k + 1) % 4])
        run.append((q2, (k2 + 1) % 4))
    return cut


def lay_out(H, C, tolerance, depth=DEPTH, tangent=TANGENT, progress=None, merge=True):
    """The cage's limit surface as rectangles along its lines: (surfaces,
    report) or (None, report) when the layout does not come out. A
    surface is ("nurbs", poles (nu, nv, 3), u breakpoints, v breakpoints)
    or, for a band with no corner (round a tube, no star point on it),
    subd.pack's exact pieces."""
    from .subd import _twins, grid_normals
    sp = _sp()
    cage = H.cage
    Q, QE, G = cage.Q, cage.QE, H.G
    m = len(Q)
    s = G.shape[1] - 1
    XG = (H.M @ C)[G]
    NG = grid_normals(H, C)
    crease_fine = set()
    if cage.crease.any():
        NG = _one_sided(H, XG, NG)
        crease_fine = set(np.unique(H.fine.sharp).tolist())
    tq, tk = _twins(cage)
    tq, tk = tq.reshape(-1, 4), tk.reshape(-1, 4)
    evv = cage.valence != 4
    evf = evv[Q].any(axis=1)
    cut = motorcycles(cage)

    def cont(q, k):
        q2, k2 = int(tq[q, (k + 1) % 4]), int(tk[q, (k + 1) % 4])
        return q2, (k2 + 1) % 4

    def twin(q, k):
        return int(tq[q, k]), int(tk[q, k])

    def start(q, k):
        return int(Q[q, k])

    # ---- rectangles between the lines
    def find_pieces():
        lab = np.full(m, -1)
        pieces = []
        for q0 in range(m):
            if lab[q0] >= 0:
                continue
            lab[q0] = len(pieces)
            stack, comp = [q0], []
            while stack:
                q = stack.pop()
                comp.append(q)
                for k in range(4):
                    if not cut[QE[q, k]]:
                        q2 = int(tq[q, k])
                        if lab[q2] < 0:
                            lab[q2] = lab[q0]
                            stack.append(q2)
            pieces.append(comp)
        return pieces
    def step(q, r, d):
        side = (d + r) % 4
        q2, k2 = int(tq[q, side]), int(tk[q, side])
        return q2, (k2 - (d + 2)) % 4

    def rect(comp):
        cs = set(comp)
        for q in comp:
            for r in range(4):
                if cut[QE[q, r]] and cut[QE[q, (r + 3) % 4]]:
                    break
            else:
                continue
            R, x = [], (q, r)
            while True:
                row, y = [x], x
                while not cut[QE[y[0], (1 + y[1]) % 4]]:
                    y = step(*y, 1)
                    row.append(y)
                    if len(row) > len(comp):
                        return None
                R.append(row)
                if cut[QE[x[0], (2 + x[1]) % 4]]:
                    break
                x = step(*x, 2)
                if len(R) > len(comp):
                    return None
            fs = [f for row in R for f, _ in row]
            if len(set(len(row) for row in R)) != 1 or len(fs) != len(cs) or set(fs) != cs:
                return None
            return R
        return None


    # A line closes on itself where a crease runs right round a hole's
    # rim with no star point on it (Instant Meshes keeps the rim's rows
    # regular): one cubic curve cannot start and end there, and a whole
    # many-hole part used to go to face-by-face patches for it (thousands
    # of faces, sewn with free edges). The ring is cut open at a seam: the
    # vertex where most lines already meet it, the seam a stop, and a line
    # drawn straight out from it on each side until it meets another (as a
    # motorcycle from a star point), so every rectangle beside the ring has
    # a corner there and none runs across the seam.
    seams = set()
    for _seam_round in range(4 * m + 1):
        ring = None
        # a rectangle that wraps round a member (two opposite sides share a
        # cage vertex: the motorcycles closed a band round it with one line) is
        # one face whose opposite edges meet; a reader rebuilding its 2-D curves
        # by projection puts an edge on the wrong side of the seam and the face
        # comes back invalid. Cut it across the middle (the new line ends on
        # the two lines it runs between) and look again.
        for _ in range(64):
            pieces = find_pieces()
            wrapped = 0
            for comp in pieces:
                R = rect(comp)
                if R is None:
                    continue
                b, a = len(R), len(R[0])
                vs = lambda hs: {int(Q[q, k]) for q, k in hs} | {int(Q[q, (k + 1) % 4]) for q, k in hs}
                v0 = vs([(q, r % 4) for q, r in R[0]])
                v1 = vs([(q, (r + 2) % 4) for q, r in R[-1]])
                u0 = vs([(q, (r + 3) % 4) for q, r in (row[0] for row in R)])
                u1 = vs([(q, (r + 1) % 4) for q, r in (row[-1] for row in R)])
                if v0 & v1 and b >= 2:
                    cut[[QE[q, (r + 2) % 4] for q, r in R[b // 2 - 1]]] = True
                    wrapped += 1
                elif u0 & u1 and a >= 2:
                    cut[[QE[q, (r + 1) % 4] for q, r in (row[a // 2 - 1] for row in R)]] = True
                    wrapped += 1
            if not wrapped:
                break
        merged = _merge(cage, cut, find_pieces, rect, evv) if merge else 0
        if merged:
            pieces = find_pieces()

        # ---- lines: maximal straight runs of drawn edges
        # A line that meets a crease stops there: the surface turns a corner
        # at the crease, and one C2 curve laid on through it misses the turn
        # (0.21-0.27 mm a face off an L-bracket's edges); a line along a crease
        # goes on along it
        shp = cage.sharpness
        crs = cage.crease

        def stops(v, e1, e2):
            # a crease's free end (one crease edge, a dart) is smooth across:
            # the line, and the rectangle side along it, run on through
            return bool(evv[v] or v in seams or shp[v] >= 3 or (shp[v] == 2 and not (crs[e1] and crs[e2])))
        def halvings(v):
            # knots halved toward each star point; one fewer at a star point on
            # a crease, whose sectors are one-sided fans of a face or two: four
            # halvings there swing (a 1.3 deg kink between rectangles at a
            # plate's pocket corner). Per point: the lines and every rectangle
            # round it agree
            return 0 if not evv[v] else depth - 1 if shp[v] >= 2 else depth
        line_of = {}                              # edge -> (line, index, same direction as the line)
        lines = []
        for q0 in range(m):
            for k0 in range(4):
                e0 = QE[q0, k0]
                if not cut[e0] or e0 in line_of:
                    continue
                # back up to the line's start
                h = (q0, k0)
                for _ in range(len(cut) + 1):
                    v = start(*h)
                    if evv[v]:
                        break
                    b = twin(*cont(*twin(*h)))
                    if not cut[QE[b]] or stops(v, QE[h], QE[b]):
                        break
                    h = b
                    if h == (q0, k0):
                        ring = (q0, k0)
                        break
                if ring is not None:
                    break
                hs = [h]
                while True:
                    v = int(Q[hs[-1][0], (hs[-1][1] + 1) % 4])
                    if evv[v]:
                        break
                    n = cont(*hs[-1])
                    if not cut[QE[n]] or stops(v, QE[hs[-1]], QE[n]):
                        break
                    if n == hs[0]:
                        ring = hs[0]
                        break
                    hs.append(n)
                L = len(lines)
                for i, hh in enumerate(hs):
                    line_of[int(QE[hh])] = (L, i, start(*hh))
                lines.append(hs)
                if ring is not None:
                    break
            if ring is not None:
                break

        if ring is None:
            break
        if not _cut_seam(ring, cut, seams, cage, evv, cont):
            return None, {"layout": "a line closes on itself"}
        line_of, lines = {}, []
    # ---- fit every line once
    curves = []
    for hs in lines:
        n = len(hs)
        v0 = start(*hs[0])
        v1 = int(Q[hs[-1][0], (hs[-1][1] + 1) % 4])
        br = _halves(n, halvings(v0), halvings(v1))
        # a span beside a face with a star point corner is two cubics: a knot at its middle
        mids = [i + 0.5 for i, (q, k) in enumerate(hs) if evf[q] or evf[int(tq[q, k])]]
        br = np.unique(np.r_[br, mids])
        t, P, N, near = [], [], [], []
        for i, (q, k) in enumerate(hs):
            from .subd import _side
            t.append(i + np.arange(s + 1) / s)
            P.append(_side(XG[q:q + 1], k, s)[0])
            N.append(_side(NG[q:q + 1], k, s)[0])
            near.append(np.full(s + 1, (i == 0 and evv[v0]) or (i == n - 1 and evv[v1])))
        t, P, N, near = np.concatenate(t), np.concatenate(P), np.concatenate(N), np.concatenate(near)
        B, D = basis(br, t, der=True)
        nb = B.shape[1]
        pin = np.isclose(t, np.round(t))                       # the cage's vertices
        w = np.where(pin, PIN, 1.0)
        rows = [sp.kron(sp.identity(3), sp.csr_matrix(w[:, None] * B))]
        rhs = [(w[:, None] * P).T.ravel()]
        if near.any():
            Dn = D[near]
            Nn = N[near]
            rows.append(tangent * sp.hstack([sp.csr_matrix(Nn[:, c:c + 1] * Dn) for c in range(3)]))
            rhs.append(np.zeros(int(near.sum())))
        A = sp.vstack(rows).tocsr()
        x = _solve(A, np.concatenate(rhs))
        curves.append((br, x.reshape(3, nb).T))

    def evalc(L, t):
        br, Pc = curves[L]
        return basis(br, t) @ Pc

    from .subd import _oriented
    out = []
    band = np.zeros(m, bool)
    worst = exact = 0.0
    worst_at = None
    side_normals = {}
    angles = []
    for comp in pieces:
        R = rect(comp)
        if R is None:
            if evf[comp].any():
                return None, {"layout": "a piece beside a star point is not a rectangle"}
            band[comp] = True
            continue
        b, a = len(R), len(R[0])
        X = np.empty((a * s + 1, b * s + 1, 3))
        NX = np.empty_like(X)
        Gr = np.empty((a * s + 1, b * s + 1), np.int64)
        Rg = np.zeros((a * s + 1, b * s + 1), bool)              # samples on faces without a star point
        for j, row in enumerate(R):
            for i, (q, r) in enumerate(row):
                k = _oriented(G, q, r)
                X[i * s:i * s + s + 1, j * s:j * s + s + 1] = np.rot90(XG[q], k)
                NX[i * s:i * s + s + 1, j * s:j * s + s + 1] = np.rot90(NG[q], k)
                Gr[i * s:i * s + s + 1, j * s:j * s + s + 1] = np.rot90(G[q], k)
                if not evf[q]:
                    Rg[i * s:i * s + s + 1, j * s:j * s + s + 1] = True
        # the four sides: (half-edges in order along +u / +v, the rectangle's parameter)
        sides = {
            "v0": [(q, r % 4) for q, r in R[0]],
            "v1": [(q, (r + 2) % 4) for q, r in R[-1]],                # these two run against the side
            "u0": [(q, (r + 3) % 4) for q, r in (row[0] for row in R)],
            "u1": [(q, (r + 1) % 4) for q, r in (row[-1] for row in R)],
        }

        def on_line(hs, forward):
            """(line, parameter at the side's start, +1/-1 per unit of the side)."""
            L, i, v = line_of[int(QE[hs[0]])]
            if (start(*hs[0]) == v) == forward:
                return L, float(i), 1.0
            return L, float(i + 1), -1.0

        # v0 runs +u as the faces' own sides; v1, u0 were listed against their sides
        info = {"v0": on_line(sides["v0"], True), "u1": on_line(sides["u1"], True),
                "v1": on_line(sides["v1"], False), "u0": on_line(sides["u0"], False)}
        for key, hs in sides.items():
            if any(line_of.get(int(QE[hh]), (-1,))[0] != info[key][0] for hh in hs):
                return None, {"layout": "a rectangle's side leaves its line"}
        cv = {"00": _vert(Q, R[0][0], 0), "a0": _vert(Q, R[0][-1], 1),
              "ab": _vert(Q, R[-1][-1], 2), "0b": _vert(Q, R[-1][0], 3)}
        corner_ev = {k: evv[v] for k, v in cv.items()}
        dep = {k: halvings(v) for k, v in cv.items()}

        def mapped(key, n):
            L, t0, dt = info[key]
            br = curves[L][0]
            lo, hi = sorted((t0, t0 + dt * n))
            inside = br[(br > lo + 1e-12) & (br < hi - 1e-12)]
            return (inside - t0) / dt

        bu = _halves(a, max(dep["00"], dep["0b"]), max(dep["a0"], dep["ab"]))
        bv = _halves(b, max(dep["00"], dep["a0"]), max(dep["0b"], dep["ab"]))
        bu = np.unique(np.r_[bu, mapped("v0", a), mapped("v1", a)])
        bv = np.unique(np.r_[bv, mapped("u0", b), mapped("u1", b)])
        nu, nv = len(bu) + 2, len(bv) + 2

        def edge_poles(key, n, br):
            L, t0, dt = info[key]
            tt = np.unique(np.r_[np.linspace(0, n, 8 * n * s + 1), br])
            Pv = evalc(L, t0 + dt * tt)
            Bt = basis(br, tt)
            Pp = np.linalg.lstsq(Bt, Pv, rcond=None)[0]
            Pp[0], Pp[-1] = Pv[0], Pv[-1]
            return Pp

        poles = np.full((nu, nv, 3), np.nan)
        poles[:, 0] = edge_poles("v0", a, bu)
        poles[:, -1] = edge_poles("v1", a, bu)
        poles[0, :] = edge_poles("u0", b, bv)
        poles[-1, :] = edge_poles("u1", b, bv)
        # rows: every sample in a face at a star point corner, four a face
        # elsewhere (a cubic span is fixed by four points; the fit there is exact)
        gap = max(1, s // 4)
        iu = np.arange(a * s + 1)
        iv = np.arange(b * s + 1)
        ku = (iu % gap == 0) | (((iu <= s) & (corner_ev["00"] or corner_ev["0b"])) |
                                ((iu >= (a - 1) * s) & (corner_ev["a0"] or corner_ev["ab"])))
        kv = (iv % gap == 0) | (((iv <= s) & (corner_ev["00"] or corner_ev["a0"])) |
                                ((iv >= (b - 1) * s) & (corner_ev["0b"] or corner_ev["ab"])))
        iu, iv = iu[ku], iv[kv]
        Xs, NXs = X[np.ix_(iu, iv)], NX[np.ix_(iu, iv)]
        tu, tv = iu / s, iv / s
        Bu, Du = basis(bu, tu, der=True)
        Bv, Dv = basis(bv, tv, der=True)
        known = ~np.isnan(poles[..., 0])
        kidx = known.ravel()
        A = sp.kron(sp.csr_matrix(Bu), sp.csr_matrix(Bv)).tocsc()
        Ak, Au = A[:, kidx], A[:, ~kidx]
        y = Xs.reshape(-1, 3) - Ak @ poles.reshape(-1, 3)[kidx]
        # the tangent planes at every sample too: met exactly where the limit
        # is a bicubic, and where it is not (round a star point) they hold the
        # slope across the rectangle's edges, which positions alone leave
        # loose: kinks of several degrees between neighbours without them
        near = np.ones((len(iu), len(iv)), bool)
        ni = int((~kidx).sum())
        if ni:
            blocks = [sp.kron(sp.identity(3), Au)]
            rhs = [y.T.ravel()]
            if near.any():
                nr = near.ravel()
                Nn = NXs.reshape(-1, 3)[nr]
                edge_row = np.zeros_like(near)
                edge_row[0, :] = edge_row[-1, :] = edge_row[:, 0] = edge_row[:, -1] = True
                wgt = tangent * np.where(edge_row.ravel()[nr], 1.0, 0.3)
                pk = poles.reshape(-1, 3)[kidx]
                for Dm in (sp.kron(sp.csr_matrix(Du), sp.csr_matrix(Bv)).tocsc(),
                           sp.kron(sp.csr_matrix(Bu), sp.csr_matrix(Dv)).tocsc()):
                    Dr = Dm[nr]
                    Dk, Di = Dr[:, kidx], Dr[:, ~kidx]
                    blocks.append(sp.diags(wgt) @ sp.hstack([sp.diags(Nn[:, c]) @ Di for c in range(3)]))
                    rhs.append(-wgt * np.einsum("ij,ij->i", Nn, Dk @ pk))
            xs = _solve(sp.vstack(blocks).tocsr(), np.concatenate(rhs))
            poles.reshape(-1, 3)[~kidx] = xs.reshape(3, ni).T
        Bu, Du = basis(bu, np.arange(a * s + 1) / s, der=True)
        Bv, Dv = basis(bv, np.arange(b * s + 1) / s, der=True)
        S = np.einsum("ia,abk,jb->ijk", Bu, poles, Bv, optimize=True)
        dev = np.linalg.norm(S - X, axis=2)
        err = float(dev.max())
        if err > worst:
            worst_at = X.reshape(-1, 3)[int(dev.argmax())]
        worst = max(worst, err)
        if Rg.any():
            exact = max(exact, float(dev[Rg].max()))
        # normals along the four sides, keyed by the fine vertex they sit on
        Su = np.einsum("ia,abk,jb->ijk", Du, poles, Bv, optimize=True)
        Sv = np.einsum("ia,abk,jb->ijk", Bu, poles, Dv, optimize=True)
        Nr = np.cross(Su, Sv)
        Nr /= np.maximum(np.linalg.norm(Nr, axis=2, keepdims=True), 1e-300)
        for sl in ((slice(1, -1), 0), (slice(1, -1), -1), (0, slice(1, -1)), (-1, slice(1, -1))):
            for vid, nn in zip(Gr[sl].ravel(), Nr[sl].reshape(-1, 3)):
                if int(vid) in crease_fine:
                    continue            # a crease: the surfaces meet at its corner, not smoothly
                o = side_normals.get(int(vid))
                if o is None:
                    side_normals[int(vid)] = nn
                else:
                    angles.append(np.degrees(np.arccos(np.clip(abs(nn @ o), -1, 1))))
        out.append(("nurbs", poles, bu, bv))
        if progress and len(out) % 200 == 0:
            progress(f"lines: {len(out)} rectangles")
    rep = {"layout": "lines", "lines": len(lines), "rectangles": len(out), "merged": merged,
           "bandFaces": int(band.sum()),
           "patchToLimit": {"max": round(worst, 5), "regularMax": exact,
                            "at": np.round(worst_at, 3).tolist() if worst > 0 else None},
           "packedSurfaces": len(out),
           "edgeAngle": {"max": round(float(max(angles)) if angles else 0.0, 4),
                         "p99": round(float(np.percentile(angles, 99)) if angles else 0.0, 4)}}
    if worst > tolerance / 3:
        rep["layout"] = (f"a rectangle misses the limit surface by {worst:.4f} mm"
                         f" at {np.round(worst_at, 2).tolist()}")
        return None, rep
    if band.any():
        out += [("band", np.flatnonzero(band))]
    return out, rep


def _merge(cage, cut, find_pieces, rect, evv):
    """Rectangles side by side that together make one rectangle become one:
    the stretch of line between them is taken out (`cut` changed in place).
    The motorcycle graph runs every line on until it meets another, so a
    line often parts two rectangles of the same width; one B-spline covers
    both, its knots holding both rectangles' (a B-rep face may meet several
    neighbours along one side). A stretch touching a star point stays: a
    star point is a corner of every surface round it. A union whose
    opposite sides meet (wrapped round a member) is not taken. Returns the
    number of merges."""
    Q, QE = cage.Q, cage.QE
    star_edge = evv[cage.E].any(axis=1) | getattr(cage, "crease", np.zeros(len(cage.E), bool))
    merges = 0
    for _ in range(64):
        pieces = find_pieces()
        lab = np.empty(len(Q), np.int64)
        for i, comp in enumerate(pieces):
            lab[comp] = i
        shape = {}
        for i, comp in enumerate(pieces):
            R = rect(comp)
            if R is not None:
                shape[i] = (len(R), len(R[0]))
        # the two pieces each drawn edge parts
        fe = QE.ravel()
        ff = np.repeat(np.arange(len(Q)), 4)
        drawn = cut[fe]
        o = np.argsort(fe[drawn], kind="stable")
        e2, f2 = fe[drawn][o], ff[drawn][o]
        pairs = {}
        for k in range(0, len(e2) - 1, 2):
            a, b = int(lab[f2[k]]), int(lab[f2[k + 1]])
            if a == b or a not in shape or b not in shape:
                continue
            pairs.setdefault((min(a, b), max(a, b)), []).append(int(e2[k]))
        used = set()
        n0 = merges
        # largest unions first: they take the most lines out
        for (a, b), es in sorted(pairs.items(), key=lambda kv: -(np.prod(shape[kv[0][0]]) + np.prod(shape[kv[0][1]]))):
            if a in used or b in used or star_edge[es].any():
                continue
            cut[es] = False
            R = rect(pieces[a] + pieces[b])
            if R is None or _wraps(Q, R):
                cut[es] = True
                continue
            used |= {a, b}
            merges += 1
        if merges == n0:
            break
    return merges


def _cut_seam(ring, cut, seams, cage, evv, cont):
    """Open a ring of drawn edges with no stop on it (`lay_out`): pick the
    vertex on it where most other lines meet it, make it a stop, and draw
    a line straight out from it on each side until it meets another line
    or a star point. False if the ring has no vertex to open it at."""
    Q, QE, E = cage.Q, cage.QE, cage.E
    hs, h = [ring], cont(*ring)
    while h != ring and len(hs) <= len(Q) * 4:
        hs.append(h)
        h = cont(*h)
    ring_e = {int(QE[q, k]) for q, k in hs}
    deg = np.bincount(E[cut].ravel(), minlength=cage.n)
    verts = [int(Q[q, k]) for q, k in hs]
    cand = [v for v in verts if v not in seams and not evv[v] and cage.valence[v] == 4]
    if not cand:
        return False
    v = max(cand, key=lambda x: deg[x])
    seams.add(v)
    qs, ks = np.nonzero(Q == v)
    for q, k in zip(qs.tolist(), ks.tolist()):
        h = (q, k)                                      # a half-edge leaving v
        if int(QE[q, k]) in ring_e or cut[QE[q, k]]:
            continue
        for _ in range(len(Q) * 4):
            e = int(QE[h])
            w = int(Q[h[0], (h[1] + 1) % 4])
            on_line = deg[w] > 0
            cut[e] = True
            deg[E[e]] += 1
            if on_line or evv[w] or cut[QE[cont(*h)]]:
                break
            h = cont(*h)
    return True


def _wraps(Q, R):
    """A rectangle (rows of (face, side)) whose opposite sides share a
    cage vertex."""
    def vs(hs):
        return {int(Q[q, k]) for q, k in hs} | {int(Q[q, (k + 1) % 4]) for q, k in hs}
    v0 = vs([(q, r % 4) for q, r in R[0]])
    v1 = vs([(q, (r + 2) % 4) for q, r in R[-1]])
    u0 = vs([(q, (r + 3) % 4) for q, r in (row[0] for row in R)])
    u1 = vs([(q, (r + 1) % 4) for q, r in (row[-1] for row in R)])
    return bool(v0 & v1) or bool(u0 & u1)


def _vert(Q, qr, corner):
    """The cage vertex at a rectangle's corner: face (q, r)'s local corner
    `corner` (0 at (0, 0), 1 at (+u, 0), 2 at (+u, +v), 3 at (0, +v))."""
    q, r = qr
    return int(Q[q, (r + corner) % 4])


def _solve(A, b):
    """Least squares through the normal equations (sparse, small bandwidth)."""
    from scipy.sparse.linalg import spsolve
    AtA = (A.T @ A).tocsc()
    return spsolve(AtA, A.T @ b)
