"""
Lattice parts as 3MF with the lattice kept as a lattice, so a part goes to
another tool without a 40-65 MB STL of lattice triangles:

- TPMS lattices (gyroid, diamond, Schwarz P, sheet or network) as a level
  set of the 3MF Volumetric & Implicit extension: the TPMS formula itself
  as an implicit function (the same expression loadpath meshes, so the
  surface is exact), its wall thickness a constant or, when a field grades
  it, sampled into a 3D image (16-bit PNG stack) the function reads;
- strut lattices (cubic, BCC, octet, Kelvin, ...) as a beam lattice of the
  Beam Lattice extension: every strut a beam, its radius at each end from
  the thickness field.

Either is clipped by the part's envelope (the part with its lattice solid,
as in the STEP), and the rest of the part, skin, bosses and bores, is an
ordinary mesh object beside it; a printer or slicer unions the two. What
does not carry over: the blend fillets where the lattice meets the body
(junctions come out sharp), graded cells and non-cartesian frames (those
lattices are refused here; ship the STL). Written with lib3mf
(pip install "loadpath[3mf]"), which also reads the file back as a check.
"""
from __future__ import annotations

import io
import os

import numpy as np

UNIT_MM = "millimeter"


def available() -> bool:
    try:
        import lib3mf  # noqa: F401
    except ImportError:
        return False
    return True


def _need():
    try:
        import lib3mf
    except ImportError as e:
        raise RuntimeError('lattice 3MF needs lib3mf: pip install "loadpath[3mf]"') from e
    return lib3mf


# ---- a small expression graph: evaluated with numpy, written as implicit nodes

class X:
    """A scalar expression of the position (x, y, z), mm."""

    def __init__(self, op, args=(), value=None):
        self.op, self.args, self.value = op, tuple(args), value

    def __add__(self, o):
        return X("add", (self, _x(o)))

    __radd__ = __add__

    def __sub__(self, o):
        return X("sub", (self, _x(o)))

    def __rsub__(self, o):
        return X("sub", (_x(o), self))

    def __mul__(self, o):
        return X("mul", (self, _x(o)))

    __rmul__ = __mul__

    def __truediv__(self, o):
        return X("div", (self, _x(o)))

    def __neg__(self):
        return X("sub", (_x(0.0), self))


def _x(v):
    return v if isinstance(v, X) else X("const", value=float(v))


def sin(a):
    return X("sin", (a,))


def cos(a):
    return X("cos", (a,))


def sqrt(a):
    return X("sqrt", (a,))


def fabs(a):
    return X("abs", (a,))


def fmax(a, b):
    return X("max", (_x(a), _x(b)))


def image(name, lo, size):
    """The red channel of image3d `name` (registered with the writer) over
    the box lo .. lo + size, as a scalar of the position."""
    return X("image", value=(name, np.asarray(lo, float), np.asarray(size, float)))


POS = {k: X("pos", value=i) for i, k in enumerate("xyz")}


def evaluate(e, P, images=None, memo=None):
    """e at points P (n, 3); `images` {name: (grid, offset, scale)} for image
    nodes, sampled the way the extension says (trilinear between voxel
    centres, clamped)."""
    memo = {} if memo is None else memo
    if id(e) in memo:
        return memo[id(e)]
    a = [evaluate(x, P, images, memo) for x in e.args]
    op = e.op
    if op == "pos":
        r = P[:, e.value]
    elif op == "const":
        r = np.full(len(P), e.value)
    elif op == "image":
        name, lo, size = e.value
        grid, off, scale = images[name]
        r = _sample(grid, (P - lo) / size) * scale + off
    else:
        r = {"add": lambda: a[0] + a[1], "sub": lambda: a[0] - a[1], "mul": lambda: a[0] * a[1],
             "div": lambda: a[0] / a[1], "sin": lambda: np.sin(a[0]), "cos": lambda: np.cos(a[0]),
             "sqrt": lambda: np.sqrt(a[0]), "abs": lambda: np.abs(a[0]),
             "max": lambda: np.maximum(a[0], a[1])}[op]()
    memo[id(e)] = r
    return r


def _sample(grid, uvw):
    """Trilinear sample of grid[z, y, x] at normalized uvw (voxel centres at
    (i + 0.5) / n), clamped."""
    from scipy.ndimage import map_coordinates

    nz, ny, nx = grid.shape
    c = np.stack([np.clip(uvw[:, 2] * nz - 0.5, 0, nz - 1), np.clip(uvw[:, 1] * ny - 0.5, 0, ny - 1),
                  np.clip(uvw[:, 0] * nx - 0.5, 0, nx - 1)])
    return map_coordinates(grid, c, order=1, mode="nearest")


class _Emitter:
    """X expressions as nodes of one lib3mf implicit function."""

    def __init__(self, lib, model, fn, images):
        self.lib, self.model, self.fn, self.images = lib, model, fn, images
        self.n = 0
        self.memo = {}
        pos = fn.AddInput("pos", "position", lib.ImplicitPortType.Vector)
        self.dec = fn.AddDecomposeVectorNode(self._id("pos"), "position", "")
        fn.AddLink(pos, self.dec.GetInputA())
        self.posport = pos

    def _id(self, s):
        self.n += 1
        return f"{s}_{self.n}"

    def port(self, e):
        if id(e) in self.memo:
            return self.memo[id(e)]
        lib, fn, S = self.lib, self.fn, self.lib.ImplicitNodeConfiguration.ScalarToScalar
        if e.op == "pos":
            out = (self.dec.GetOutputX, self.dec.GetOutputY, self.dec.GetOutputZ)[e.value]()
        elif e.op == "const":
            nd = fn.AddConstantNode(self._id("c"), "", "")
            nd.SetConstant(e.value)
            out = nd.GetOutputValue()
        elif e.op == "image":
            out = self._image(e)
        else:
            add = {"add": fn.AddAdditionNode, "sub": fn.AddSubtractionNode, "mul": fn.AddMultiplicationNode,
                   "div": fn.AddDivisionNode, "sin": fn.AddSinNode, "cos": fn.AddCosNode, "sqrt": fn.AddSqrtNode,
                   "abs": fn.AddAbsNode, "max": fn.AddMaxNode}[e.op]
            nd = add(self._id(e.op), S, "", "")
            fn.AddLink(self.port(e.args[0]), nd.GetInputA())
            if len(e.args) > 1:
                fn.AddLink(self.port(e.args[1]), nd.GetInputB())
            out = nd.GetOutputResult()
        self.memo[id(e)] = out
        return out

    def _image(self, e):
        name, lo, size = e.value
        lib, fn = self.lib, self.fn
        V = lib.ImplicitNodeConfiguration.VectorToVector
        # uvw = (pos - lo) / size, then the image function's red channel
        lo_n = fn.AddConstVecNode(self._id("lo"), "", "")
        lo_n.SetVector(lib.Vector((float(lo[0]), float(lo[1]), float(lo[2]))))
        sz = fn.AddConstVecNode(self._id("size"), "", "")
        sz.SetVector(lib.Vector((float(size[0]), float(size[1]), float(size[2]))))
        sub = fn.AddSubtractionNode(self._id("rel"), V, "", "")
        fn.AddLink(self.posport, sub.GetInputA())
        fn.AddLink(lo_n.GetOutputVector(), sub.GetInputB())
        div = fn.AddDivisionNode(self._id("uvw"), V, "", "")
        fn.AddLink(sub.GetOutputResult(), div.GetInputA())
        fn.AddLink(sz.GetOutputVector(), div.GetInputB())
        rid = fn.AddResourceIdNode(self._id("img"), "", "")
        rid.SetResource(self.images[name])
        call = fn.AddFunctionCallNode(self._id("call"), "", "")
        fn.AddLink(rid.GetOutputValue(), call.GetInputFunctionID())
        inp = call.AddInput("pos", "uvw")
        inp.SetType(lib.ImplicitPortType.Vector)
        fn.AddLink(div.GetOutputResult(), inp)
        out = call.AddOutput("red", "value")
        out.SetType(lib.ImplicitPortType.Scalar)
        return out


# ---- lattices

_TPMS = ("gyroid", "diamond", "schwarz-p")


def tpms_expr(kind, cell, thickness, form="sheet", origin=(0, 0, 0)):
    """Signed distance (mm, negative inside) of a TPMS sheet or network as an
    expression: loadpath.lattice.tpms (value over gradient magnitude, the
    magnitude floored at 0.3 k) less half the wall."""
    k = 2.0 * np.pi / float(cell)
    x, y, z = ((POS[c] - float(o)) * k for c, o in zip("xyz", origin))
    sx, cx, sy, cy, sz, cz = sin(x), cos(x), sin(y), cos(y), sin(z), cos(z)
    if kind == "gyroid":
        f = sx * cy + sy * cz + sz * cx
        g = (cx * cy - sz * sx, cy * cz - sx * sy, cz * cx - sy * sz)
    elif kind == "diamond":
        f = sx * sy * sz + sx * cy * cz + cx * sy * cz + cx * cy * sz
        g = (cx * sy * sz + cx * cy * cz - sx * sy * cz - sx * cy * sz,
             sx * cy * sz - sx * sy * cz + cx * cy * cz - cx * sy * sz,
             sx * sy * cz - sx * cy * sz - cx * sy * sz + cx * cy * cz)
    elif kind == "schwarz-p":
        f = cx + cy + cz
        g = (-1.0 * sx, -1.0 * sy, -1.0 * sz)
    else:
        raise ValueError(f"TPMS kind must be one of {', '.join(_TPMS)}")
    gn = fmax(sqrt(g[0] * g[0] + g[1] * g[1] + g[2] * g[2]) * k, 0.3 * k)
    d = f / gn
    half = 0.5 * thickness
    if form == "sheet":
        return fabs(d) - half
    if form == "network":
        return d - half
    raise ValueError("TPMS form must be 'sheet' or 'network'")


def lattice_beams(kind, cell, lo, hi, origin=(0, 0, 0), aspect=(1, 1, 1)):
    """The struts of a periodic strut cell tiled over the box lo..hi (mm):
    (vertices (n, 3), beams (m, 2) vertex index pairs), each strut once."""
    from .struts import canonical, cell_segments

    seg = np.asarray(cell_segments(canonical(kind)), float)          # (s, 2, 3), period 1
    size = float(cell) * np.asarray(aspect, float)
    o = np.asarray(origin, float)
    i0 = np.floor((np.asarray(lo, float) - o) / size).astype(int) - 1
    i1 = np.ceil((np.asarray(hi, float) - o) / size).astype(int) + 1
    ijk = np.stack(np.meshgrid(*[np.arange(a, b) for a, b in zip(i0, i1)], indexing="ij"), -1).reshape(-1, 3)
    ends = (seg[None] + ijk[:, None, None, :]).reshape(-1, 2, 3)    # in cell units
    key = np.round(ends * 1e4).astype(np.int64)
    # each strut once whichever way round: ends in lexicographic order
    d = key[:, 1] - key[:, 0]
    r = np.arange(len(d))
    back = d[r, np.argmax(d != 0, axis=1)] < 0
    key[back] = key[back][:, ::-1]
    ends[back] = ends[back][:, ::-1]
    _, uniq = np.unique(key.reshape(len(key), -1), axis=0, return_index=True)
    E = ends[uniq] * size + o
    lo_, hi_ = np.asarray(lo, float), np.asarray(hi, float)
    keep = ((E.max(axis=1) >= lo_) & (E.min(axis=1) <= hi_)).all(1)
    E = E[keep]
    V, inv = np.unique(np.round(E.reshape(-1, 3), 6), axis=0, return_inverse=True)
    B = inv.reshape(-1, 2)
    B = B[B[:, 0] != B[:, 1]]
    return V, B


# ---- the file

def write_lattice_3mf(path, body, envelope, lattices, name="part", check=True):
    """Write the 3MF: `body` (trimesh, the part without its lattices) and,
    per lattice, a level set or a beam lattice clipped by `envelope` (trimesh,
    the part with its lattices solid). `lattices`: dicts from
    lattice_from_graph. Returns a report (objects, bytes, beams, nodes)."""
    lib = _need()
    w = lib.get_wrapper()
    model = w.CreateModel()
    model.SetUnit(lib.ModelUnit.MilliMeter)
    rep = {"path": path, "objects": [], "lattices": []}

    def mesh_object(m, label):
        mo = model.AddMeshObject()
        mo.SetName(label)
        V = np.asarray(m.vertices, float)
        F = np.asarray(m.faces, np.int64)
        mo.SetGeometry([lib.Position((float(a), float(b), float(c))) for a, b, c in V],
                       [lib.Triangle((int(a), int(b), int(c))) for a, b, c in F])
        return mo

    ident = lib.Transform()
    for i in range(4):
        for j in range(3):
            ident.Fields[i][j] = 1.0 if i == j else 0.0
    env = mesh_object(envelope, f"{name} envelope (lattice clip)")
    if body is not None and len(body.faces):
        bo = mesh_object(body, f"{name} body")
        model.AddBuildItem(bo, ident)
        rep["objects"].append({"name": f"{name} body", "triangles": len(body.faces)})
    for lat in lattices:
        if lat["kind"] in _TPMS:
            images = {}
            for nm, (grid, off, scale) in lat.get("images", {}).items():
                images[nm] = _image3d(lib, model, nm, grid, off, scale)
            fn = model.AddImplicitFunction()
            fn.SetDisplayName(f"{lat['name']} ({lat['kind']})")
            em = _Emitter(lib, model, fn, images)
            out = fn.AddOutput("distance", "signed distance, mm", lib.ImplicitPortType.Scalar)
            fn.AddLink(em.port(lat["expr"]), out)
            ls = model.AddLevelSet()
            ls.SetName(f"{name} {lat['name']}")
            ls.SetFunction(fn)
            ls.SetChannelName("distance")
            ls.SetMesh(env)
            ls.SetMeshBBoxOnly(False)
            ls.SetMinFeatureSize(float(lat.get("minFeature", 0.0)))
            ls.SetFallBackValue(1.0)
            model.AddBuildItem(ls, ident)
            rep["lattices"].append({"name": lat["name"], "as": "levelset", "kind": lat["kind"],
                                    "nodes": em.n, "graded": bool(images)})
        else:
            V, B, R = lat["vertices"], lat["beams"], lat["radii"]
            mo = model.AddMeshObject()
            mo.SetName(f"{name} {lat['name']}")
            mo.SetGeometry([lib.Position((float(a), float(b), float(c))) for a, b, c in V], [])
            bl = mo.BeamLattice()
            bl.SetMinLength(1e-4)
            bl.SetClipping(lib.BeamLatticeClipMode.Inside, env.GetUniqueResourceID())
            beams = []
            for (a, b) in B:
                bm = lib.Beam()
                bm.Indices[0], bm.Indices[1] = int(a), int(b)
                bm.Radii[0], bm.Radii[1] = float(R[a]), float(R[b])
                bm.CapModes[0] = bm.CapModes[1] = lib.BeamLatticeCapMode.Sphere
                beams.append(bm)
            bl.SetBeams(beams)
            model.AddBuildItem(mo, ident)
            rep["lattices"].append({"name": lat["name"], "as": "beamlattice", "kind": lat["kind"],
                                    "beams": len(B), "nodes": len(V)})
    wr = model.QueryWriter("3mf")
    wr.WriteToFile(path)
    if any(x["as"] == "levelset" for x in rep["lattices"]):
        _require_volumetric(path)
    rep["bytes"] = os.path.getsize(path)
    if check:
        rep["readBack"] = read_back(path)
    return rep


def _require_volumetric(path):
    """Mark the volumetric extension required as well as the implicit one:
    the spec says a producer of a level set MUST, and lib3mf 2.5 marks
    only the implicit namespace."""
    import re
    import zipfile

    tmp = path + ".tmp"
    with zipfile.ZipFile(path) as zi, zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zo:
        for it in zi.infolist():
            data = zi.read(it.filename)
            if it.filename == "3D/3dmodel.model":
                head, rest = data[:4096].decode("utf-8"), data[4096:]

                def fix(m):
                    req = m.group(1).split()
                    return 'requiredextensions="' + " ".join(dict.fromkeys(["v", *req])) + '"'

                head = re.sub(r'requiredextensions="([^"]*)"', fix, head, count=1)
                data = head.encode("utf-8") + rest
            zo.writestr(it, data)
    os.replace(tmp, path)


def _image3d(lib, model, nm, grid, off, scale):
    """grid[z, y, x] in [0, 1] as a 16-bit greyscale image stack; returns
    its function (the red channel, scaled and offset back to the values)."""
    from PIL import Image

    nz, ny, nx = grid.shape
    st = model.AddImageStack(nx, ny, nz)       # columns (u = x), rows (v = y), sheets (w = z)
    st.SetColumnCount(nx)
    st.SetRowCount(ny)
    for k in range(nz):
        im = Image.fromarray(np.round(np.clip(grid[k], 0, 1) * 65535).astype(np.uint16), mode="I;16")
        buf = io.BytesIO()
        im.save(buf, format="PNG")
        st.CreateSheetFromBuffer(k, f"/3D/volumetric/{nm}/sheet{k:04d}.png", list(buf.getvalue()))
    f = model.AddFunctionFromImage3D(st)
    f.SetDisplayName(nm)
    f.SetFilter(lib.TextureFilter.Linear) if hasattr(lib, "TextureFilter") else None
    f.SetTileStyles(lib.TextureTileStyle.Clamp, lib.TextureTileStyle.Clamp, lib.TextureTileStyle.Clamp)
    f.SetOffset(float(off))
    f.SetScale(float(scale))
    return f


def read_back(path):
    """Read the file with lib3mf and count what is in it."""
    lib = _need()
    w = lib.get_wrapper()
    m = w.CreateModel()
    m.QueryReader("3mf").ReadFromFile(path)
    out = {"meshes": 0, "triangles": 0, "beams": 0, "levelsets": 0, "functions": 0, "build": 0}
    it = m.GetMeshObjects()
    while it.MoveNext():
        mo = it.GetCurrentMeshObject()
        out["meshes"] += 1
        out["triangles"] += mo.GetTriangleCount()
        out["beams"] += mo.BeamLattice().GetBeamCount()
    it = m.GetLevelSets()
    while it.MoveNext():
        out["levelsets"] += 1
    it = m.GetFunctions()
    while it.MoveNext():
        out["functions"] += 1
    it = m.GetBuildItems()
    while it.MoveNext():
        out["build"] += 1
    return out


def lattice_from_graph(g, name, lo, hi, grid=None):
    """A lattice node of FieldGraph `g` as the writer's dict: the TPMS
    expression (thickness constant, or sampled over lo..hi into a 3D image
    at `grid` mm, default the smaller of cell / 6 and 1 mm), or the beams
    tiled over lo..hi with a radius per node. Raises ValueError for what the
    formats cannot say (a graded cell, a frame, an offset, Voronoi foams)."""
    from .struts import canonical, is_strut

    nd = g.nodes[name]
    s, r = nd.spec, nd.refs
    kind = s.get("type", "gyroid")
    tp = {"gyroid": "gyroid", "diamond": "diamond", "schwarz-p": "schwarz-p", "schwarz_p": "schwarz-p",
          "schwarzP": "schwarz-p"}.get(kind)
    strut = is_strut(kind)
    if (s.get("frame") or {}).get("type", "cartesian") != "cartesian":
        raise ValueError(f"lattice {name!r}: only the cartesian frame goes into 3MF as a lattice")
    if isinstance(r.get("cell"), str) or isinstance(s.get("cell"), dict):
        raise ValueError(f"lattice {name!r}: a graded cell does not go into 3MF as a lattice")
    if "offset" in s:
        raise ValueError(f"lattice {name!r}: an offset lattice does not go into 3MF as a lattice")
    if strut and canonical(kind) == "voronoi":
        raise ValueError(f"lattice {name!r}: a Voronoi foam does not go into 3MF as a beam lattice")
    if tp is None and not strut:
        raise ValueError(f"lattice {name!r}: unknown lattice type {kind!r}")
    if tp is not None and "aspect" in s:
        raise ValueError(f"lattice {name!r}: a stretched TPMS cell does not go into 3MF as a lattice")
    cell = float(s.get("cell", 8.0))
    origin = tuple(float(v) for v in s.get("origin", (0, 0, 0)))
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)

    def wall(P):
        memo = {}
        if "thickness" in r:
            return np.broadcast_to(np.asarray(g._val(r["thickness"], P, memo), float), (len(P),))
        if "density" in r:
            rho = np.clip(np.broadcast_to(np.asarray(g._val(r["density"], P, memo), float), (len(P),)), 0.05, 0.95)
            if strut:
                from .struts import strut_diameter_for_density

                return strut_diameter_for_density(kind, rho, cell)
            from .lattice import wall_for_density

            return wall_for_density(tp, rho, cell)
        return np.ones(len(P))

    out = {"name": name, "kind": tp or canonical(kind), "cell": cell}
    if strut:
        V, B = lattice_beams(kind, cell, lo, hi, origin, s.get("aspect", (1, 1, 1)))
        out.update(vertices=V, beams=B, radii=0.5 * wall(V))
        return out
    h = float(grid or min(cell / 6.0, 1.0))
    n = np.maximum(np.ceil((hi - lo) / h).astype(int), 1)
    size = n * h
    ax = [lo[i] + (np.arange(n[i]) + 0.5) * h for i in range(3)]
    Z, Y, Xg = np.meshgrid(ax[2], ax[1], ax[0], indexing="ij")
    P = np.stack([Xg.ravel(), Y.ravel(), Z.ravel()], axis=1)
    t = wall(P).reshape(Z.shape)
    t0, t1 = float(t.min()), float(t.max())
    if t1 - t0 <= 1e-6 * max(t1, 1e-9):
        thick = t0
        out["thickness"] = round(t0, 4)
    else:
        img = f"{name}-wall"
        out["images"] = {img: ((t - t0) / (t1 - t0), t0, t1 - t0)}
        thick = image(img, lo, size)
        out["thickness"] = {"min": round(t0, 4), "max": round(t1, 4), "grid": h, "voxels": n.tolist()}
    out["expr"] = tpms_expr(tp, cell, thick, s.get("form", "sheet"), origin)
    out["minFeature"] = round(0.5 * t0, 4)
    return out
