"""
Lattice parts as 3MF with the lattice kept as a lattice, so a part goes to
another tool without a 40-65 MB STL of lattice triangles:

- TPMS lattices (gyroid, diamond, Schwarz P, sheet or network) as a level
  set of the 3MF Volumetric & Implicit extension: the TPMS formula itself
  as an implicit function (the expression loadpath meshes). What a field
  grades is sampled into 3D images (16-bit PNG stacks) the function reads:
  the wall, an offset, and for a graded cell (warp grading) or a conformal
  frame the lattice coordinates themselves, which are smooth, so trilinear
  samples of them keep the surface within a few microns;
- strut lattices (cubic, BCC, octet, Kelvin, ...) as a beam lattice of the
  Beam Lattice extension: every strut a beam, its radius at each end from
  the thickness field. The beam object also carries the struts as
  octagonal prisms inside the beams, so a reader without beam support (a
  mesh-only slicer) still gets a printable lattice, about 10% thinner.

Each lattice is clipped by its own region: the part with that lattice solid
and the others gone. The rest of the part, skin, bosses and bores, is an
ordinary mesh object beside them. Where the graph blends a lattice into the
body, its level set is the whole blend, max(smin(body, lattice clipped,
r), region) with the body and region as signed distances to their meshes
(the extension's mesh node) and r the graph's blend radius, so the fillets
carry over. Strut lattices are written without their fillets, and with a
constant cell in the cartesian frame only. Written with lib3mf
(pip install "loadpath[3mf]"), which also reads the file back as a check.
"""
from __future__ import annotations

import io
import os

import numpy as np


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


def fmin(a, b):
    return X("min", (_x(a), _x(b)))


def image(name, lo, size):
    """3D image `name` (registered with the writer) over the box lo .. lo +
    size, as a scalar of the position."""
    return X("image", value=(name, np.asarray(lo, float), np.asarray(size, float)))


def mesh(name):
    """Signed distance (negative inside) to mesh `name` (registered with the
    writer)."""
    return X("mesh", value=name)


def smin(a, b, k):
    """fields._smin (polynomial smooth minimum, radius k), as an expression."""
    a, b, k = _x(a), _x(b), _x(k)
    h = fmin(fmax(0.5 + 0.5 * (b - a) / k, 0.0), 1.0)
    return b * (1.0 - h) + a * h - k * h * (1.0 - h)


POS = {k: X("pos", value=i) for i, k in enumerate("xyz")}


def evaluate(e, P, images=None, meshes=None, memo=None):
    """e at points P (n, 3); `images` {name: (grid[z, y, x], offset, scale)}
    sampled the way the extension says (trilinear between voxel centres,
    clamped), `meshes` {name: trimesh} for mesh distances."""
    memo = {} if memo is None else memo
    if id(e) in memo:
        return memo[id(e)]
    a = [evaluate(x, P, images, meshes, memo) for x in e.args]
    op = e.op
    if op == "pos":
        r = P[:, e.value]
    elif op == "const":
        r = np.full(len(P), e.value)
    elif op == "image":
        name, lo, size = e.value
        grid, off, scale = images[name]
        r = _sample(grid, (P - lo) / size) * scale + off
    elif op == "mesh":
        from .sharp import mesh_distance

        # exact near the surface; beyond `cap` only the side matters here
        m, cap = meshes[e.value], 4.0
        r = mesh_distance(m, P, cap=cap, signed=True)
        far = ~(np.abs(r) < cap)
        if far.any():
            r[far] = np.where(m.contains(P[far]), -cap, cap)
    else:
        r = {"add": lambda: a[0] + a[1], "sub": lambda: a[0] - a[1], "mul": lambda: a[0] * a[1],
             "div": lambda: a[0] / a[1], "sin": lambda: np.sin(a[0]), "cos": lambda: np.cos(a[0]),
             "sqrt": lambda: np.sqrt(a[0]), "abs": lambda: np.abs(a[0]),
             "max": lambda: np.maximum(a[0], a[1]), "min": lambda: np.minimum(a[0], a[1])}[op]()
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


def _quantize(grid):
    """The grid as 16-bit levels in [0, 1], and the offset and scale back."""
    lo_, hi_ = float(grid.min()), float(grid.max())
    span = hi_ - lo_ if hi_ > lo_ else 1.0
    return np.round((grid - lo_) / span * 65535) / 65535, lo_, span


class _Emitter:
    """X expressions as nodes of one lib3mf implicit function."""

    def __init__(self, lib, fn, images, meshes):
        self.lib, self.fn, self.images, self.meshes = lib, fn, images, meshes
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
        fn, S = self.fn, self.lib.ImplicitNodeConfiguration.ScalarToScalar
        if e.op == "pos":
            out = (self.dec.GetOutputX, self.dec.GetOutputY, self.dec.GetOutputZ)[e.value]()
        elif e.op == "const":
            nd = fn.AddConstantNode(self._id("c"), "", "")
            nd.SetConstant(e.value)
            out = nd.GetOutputValue()
        elif e.op == "image":
            out = self._image(e)
        elif e.op == "mesh":
            rid = fn.AddResourceIdNode(self._id("meshid"), "", "")
            rid.SetResource(self.meshes[e.value])
            mn = fn.AddMeshNode(self._id("meshdist"), e.value, "")
            fn.AddLink(rid.GetOutputValue(), mn.GetInputMesh())
            fn.AddLink(self.posport, mn.GetInputPos())
            out = mn.GetOutputDistance()
        else:
            add = {"add": fn.AddAdditionNode, "sub": fn.AddSubtractionNode, "mul": fn.AddMultiplicationNode,
                   "div": fn.AddDivisionNode, "sin": fn.AddSinNode, "cos": fn.AddCosNode, "sqrt": fn.AddSqrtNode,
                   "abs": fn.AddAbsNode, "max": fn.AddMaxNode, "min": fn.AddMinNode}[e.op]
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


def tpms_unit(kind, x, y, z):
    """loadpath.lattice.tpms at period 1 over coordinate expressions x, y, z
    (cell units): value over gradient magnitude (floored at 0.3 k), in
    cell units."""
    k = 2.0 * np.pi
    return tpms_unit_trig(kind, [(sin(u * k), cos(u * k)) for u in (x, y, z)])


def tpms_unit_trig(kind, trig):
    """tpms_unit from each coordinate's (sin, cos) of 2 pi u instead of u:
    a phase that wraps (a conformal frame's) is sampled continuously as its
    sine and cosine, where an image of u itself jumps by a cycle."""
    k = 2.0 * np.pi
    (sx, cx), (sy, cy), (sz, cz) = trig
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
    return f / fmax(sqrt(g[0] * g[0] + g[1] * g[1] + g[2] * g[2]) * k, 0.3 * k)


def _wall(d, thickness, form):
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


def strut_prisms(V, B, R, clip=None, sides=8):
    """The beams as `sides`-gonal frustums inside them (radius R at each
    end, vertices on the beam's circle), unioned and clipped by `clip` (a
    trimesh): a closed mesh a reader without beam support can print."""
    import manifold3d as m3d
    import trimesh

    from .shapes import manifold_from_trimesh

    parts = []
    for a, b in B:
        p, q = V[a], V[b]
        L = float(np.linalg.norm(q - p))
        if L < 1e-6:
            continue
        c = m3d.Manifold.cylinder(L, float(R[a]), float(R[b]), sides)
        z = (q - p) / L
        x = np.cross(z, (1.0, 0, 0) if abs(z[0]) < 0.9 else (0, 1.0, 0))
        x /= np.linalg.norm(x)
        y = np.cross(z, x)
        M = np.column_stack([x, y, z, p])                        # 3 x 4
        parts.append(c.transform(M.tolist()))
    if not parts:
        return trimesh.Trimesh()
    u = m3d.Manifold.batch_boolean(parts, m3d.OpType.Add)
    if clip is not None:
        u = u ^ manifold_from_trimesh(clip)
    return _solid(u)


def _solid(u):
    """A manifold3d result as a trimesh, without the specks (under 0.01
    mm3: pinholes where prisms meet at a node, slivers where a strut
    grazes the region) that a slicer would count as parts."""
    import trimesh

    me = u.to_mesh()
    t = trimesh.Trimesh(np.asarray(me.vert_properties)[:, :3], np.asarray(me.tri_verts), process=False)
    parts = t.split(only_watertight=False)
    if len(parts) > 1:
        keep = [x for x in parts if abs(x.volume) > 1e-2]
        t = trimesh.util.concatenate(keep) if keep else trimesh.Trimesh()
    return t


# ---- the file

def _union(a, b):
    from .shapes import manifold_from_trimesh

    if b is None or not len(b.faces):
        return a
    return _solid(manifold_from_trimesh(a) + manifold_from_trimesh(b))


def write_lattice_3mf(path, body, lattices, name="part", check=True, prisms=True, mesh_readers=False):
    """Write the 3MF: `body` (trimesh or None, the part without its
    lattices) and per lattice (dicts from lattice_from_graph, each with its
    `clip` trimesh, the region it fills) a level set or a beam lattice
    clipped by that region; a lattice with a `blend` radius becomes the
    blend of body and lattice. Returns a report.

    The region meshes are objects of type "model" outside the build, as
    the Beam Lattice spec asks of a clipping mesh. PrusaSlicer 2.x refuses
    any file with an object outside the build; `mesh_readers` marks the
    regions type "other" instead, which it skips, so it opens a beam
    lattice file (printing the prisms), and puts the body into the (first)
    beam lattice's mesh, so it gets one object rather than two it would
    lay apart on the bed. lib3mf reads either."""
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
    bo, items = None, []
    # for mesh-only readers, the body goes into the first beam lattice's
    # mesh (a beam lattice object is its triangles and its beams), as one
    # shell with the prisms: a slicer lays separate objects (or a components
    # object's parts) apart on the bed
    merge = (mesh_readers and body is not None and len(body.faces) and prisms
             and lattices and all(x["kind"] not in _TPMS for x in lattices))
    if body is not None and len(body.faces) and not merge:
        bo = mesh_object(body, f"{name} body")
        items.append(bo)
        rep["objects"].append({"name": f"{name} body", "triangles": int(len(body.faces))})
    for lat in lattices:
        clip = mesh_object(lat["clip"], f"{name} {lat['name']} region")
        if mesh_readers:
            clip.SetType(lib.ObjectType.Other)
        if lat["kind"] in _TPMS:
            images = {nm: _image3d(lib, model, nm, *im) for nm, im in lat.get("images", {}).items()}
            meshes = {"region": clip}
            e = lat["expr"]
            blend = lat.get("blend")
            if blend is not None and bo is not None:
                meshes["body"] = bo
                rg = mesh("region")
                e = fmax(smin(mesh("body"), fmax(e, rg), blend), rg)
            fn = model.AddImplicitFunction()
            fn.SetDisplayName(f"{lat['name']} ({lat['kind']})")
            em = _Emitter(lib, fn, images, meshes)
            out = fn.AddOutput("distance", "signed distance, mm", lib.ImplicitPortType.Scalar)
            fn.AddLink(em.port(e), out)
            ls = model.AddLevelSet()
            ls.SetName(f"{name} {lat['name']}")
            ls.SetFunction(fn)
            ls.SetChannelName("distance")
            ls.SetMesh(clip)
            ls.SetMeshBBoxOnly(False)
            ls.SetMinFeatureSize(float(lat.get("minFeature", 0.0)))
            ls.SetFallBackValue(1.0)
            items.append(ls)
            rep["lattices"].append({"name": lat["name"], "as": "levelset", "kind": lat["kind"], "nodes": em.n,
                                    "images": sorted(images), "blended": blend is not None and bo is not None})
        else:
            V, B, R = lat["vertices"], lat["beams"], lat["radii"]
            pm = strut_prisms(V, B, R, lat["clip"]) if prisms else None
            if merge and not rep["objects"]:
                pm = _union(body, pm)
                rep["objects"].append({"name": f"{name} {lat['name']}", "triangles": int(len(pm.faces)),
                                       "with": "body"})
            mo = mesh_object(pm, f"{name} {lat['name']}") if pm is not None and len(pm.faces) else None
            if mo is None:
                mo = model.AddMeshObject()
                mo.SetName(f"{name} {lat['name']}")
            nv = mo.GetVertexCount()
            for p in V:
                mo.AddVertex(lib.Position((float(p[0]), float(p[1]), float(p[2]))))
            bl = mo.BeamLattice()
            bl.SetMinLength(1e-4)
            bl.SetClipping(lib.BeamLatticeClipMode.Inside, clip.GetUniqueResourceID())
            beams = []
            for (a, b) in B:
                bm = lib.Beam()
                bm.Indices[0], bm.Indices[1] = int(a) + nv, int(b) + nv
                bm.Radii[0], bm.Radii[1] = float(R[a]), float(R[b])
                bm.CapModes[0] = bm.CapModes[1] = lib.BeamLatticeCapMode.Sphere
                beams.append(bm)
            bl.SetBeams(beams)
            items.append(mo)
            rep["lattices"].append({"name": lat["name"], "as": "beamlattice", "kind": lat["kind"],
                                    "beams": int(len(B)), "nodes": int(len(V)),
                                    "prismTriangles": int(len(pm.faces)) if pm is not None else 0})
    for it in items:
        model.AddBuildItem(it, ident)
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
    f.SetFilter(lib.TextureFilter.Linear)
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
    for get, key in ((m.GetLevelSets, "levelsets"), (m.GetFunctions, "functions"), (m.GetBuildItems, "build")):
        it = get()
        while it.MoveNext():
            out[key] += 1
    return out


# ---- from a field graph

def _grid(lo, hi, h):
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    n = np.maximum(np.ceil((hi - lo) / h).astype(int), 1)
    ax = [lo[i] + (np.arange(n[i]) + 0.5) * h for i in range(3)]
    Z, Y, Xg = np.meshgrid(ax[2], ax[1], ax[0], indexing="ij")
    return np.stack([Xg.ravel(), Y.ravel(), Z.ravel()], axis=1), Z.shape, n * h


def lattice_from_graph(g, name, lo, hi, grid=None):
    """A lattice node of FieldGraph `g` as the writer's dict: the TPMS
    expression, what a field grades in it sampled over lo..hi into 3D images
    at `grid` mm (default the smaller of cell / 6 and 1 mm); or the beams
    tiled over lo..hi with a radius per node. Raises ValueError for what the
    formats cannot say."""
    from .struts import canonical, is_strut

    nd = g.nodes[name]
    s, r = nd.spec, nd.refs
    kind = s.get("type", "gyroid")
    tp = {"gyroid": "gyroid", "diamond": "diamond", "schwarz-p": "schwarz-p", "schwarz_p": "schwarz-p",
          "schwarzP": "schwarz-p"}.get(kind)
    strut = is_strut(kind)
    ftype = (s.get("frame") or {}).get("type", "cartesian")
    graded = isinstance(r.get("cell"), str)
    if tp is None and not strut:
        raise ValueError(f"lattice {name!r}: unknown lattice type {kind!r}")
    if strut and canonical(kind) in ("voronoi", "delaunay"):
        raise ValueError(f"lattice {name!r}: a stochastic ({canonical(kind)}) lattice does not go into 3MF as a "
                         "beam lattice")
    if strut and (graded or ftype != "cartesian" or "offset" in s):
        raise ValueError(f"lattice {name!r}: a strut lattice goes into 3MF with a constant cell in the "
                         "cartesian frame and no offset")
    if ftype in ("cylindrical", "spherical"):
        raise ValueError(f"lattice {name!r}: a {ftype} frame's coordinates wrap round, which a 3D image "
                         "cannot sample; ship the STL")
    if graded and s.get("grading", "warp") != "warp":
        raise ValueError(f"lattice {name!r}: blend grading does not go into 3MF (use the default warp)")
    origin = tuple(float(v) for v in s.get("origin", (0, 0, 0)))
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    cell0 = float(s["cell"]) if isinstance(s.get("cell"), (int, float)) else None

    def local_cell(P):
        if cell0 is not None:
            return np.full(len(P), cell0)
        return np.broadcast_to(np.asarray(g._val(r["cell"], P, {}), float), (len(P),))

    def wall(P):
        memo = {}
        if "thickness" in r:
            return np.broadcast_to(np.asarray(g._val(r["thickness"], P, memo), float), (len(P),))
        if "density" in r:
            rho = np.clip(np.broadcast_to(np.asarray(g._val(r["density"], P, memo), float), (len(P),)),
                          0.02 if strut else 0.05, 0.95)
            if strut:
                from .struts import strut_diameter_for_density

                return strut_diameter_for_density(kind, rho, local_cell(P))
            from .lattice import wall_for_density

            return wall_for_density(tp, rho, local_cell(P))
        return np.ones(len(P))

    out = {"name": name, "kind": tp or canonical(kind)}
    if strut:
        V, B = lattice_beams(kind, cell0 or 8.0, lo, hi, origin, s.get("aspect", (1, 1, 1)))
        out.update(vertices=V, beams=B, radii=0.5 * wall(V), cell=cell0 or 8.0)
        return out

    images = {}
    cmin = cell0 or float(np.min(local_cell(_grid(lo, hi, 2.0)[0])))
    # a conformal frame's phases turn faster than one cycle a cell where the
    # pattern bends, and its layer coordinate kinks on the medial axis:
    # interpolated at cell / 8 the part's inside/outside differs from the
    # field over 2.7% of the volume (tube, box), at cell / 12 1.4%, cell / 16
    # 0.9%; cell / 12 (8x the voxels of a graded cell's images)
    h = float(grid or min(cmin / (12.0 if ftype == "conformal" else 6.0), 1.0))
    lo, hi = lo - h, hi + h       # a voxel of margin: clamped samples hold still past the outer voxel centres
    P, shape, size = _grid(lo, hi, h)

    def field(tag, v, const_ok=True):
        """v at the grid: a constant, or an image expression."""
        v = np.asarray(v, float).reshape(shape)
        if const_ok and float(np.ptp(v)) <= 1e-6 * max(float(np.abs(v).max()), 1e-9):
            return float(v.flat[0]), None
        q, off, span = _quantize(v)
        images[f"{name}-{tag}"] = (q, off, span)
        return image(f"{name}-{tag}", lo, size), {"min": round(float(v.min()), 4), "max": round(float(v.max()), 4)}

    if ftype == "conformal":
        # the two in-plane coordinates are phases (mod 1): an image of the
        # phase jumps by a cycle at every wrap and trilinear sampling
        # smears it across a voxel, so ship each one's sine and cosine
        # (renormalized after sampling) and the layer coordinate as is;
        # the defect fill (solid nodes at phase defects) as its own image
        # (a frame field's layer coordinate is a phase too: all three)
        memo = {}
        U, cl = g.lattice_coords(nd, P, memo)
        fill = memo.get(("_defect_fill", nd.name))
        nphase = 3 if g._resource(nd, lambda: g._conformal(nd)).frame_field else 2
        trig = []
        for i in range(nphase):
            se = field(f"s{i}", np.sin(2 * np.pi * U[:, i]), const_ok=False)[0]
            ce = field(f"c{i}", np.cos(2 * np.pi * U[:, i]), const_ok=False)[0]
            nrm = fmax(sqrt(se * se + ce * ce), 1e-3)
            trig.append((se / nrm, ce / nrm))
        if nphase == 2:
            u2 = field("u2", U[:, 2], const_ok=False)[0] * (2.0 * np.pi)
            trig.append((sin(u2), cos(u2)))
        cexp, cinfo = field("cell", cl)
        d = tpms_unit_trig(tp, trig)
        if fill is not None:
            fexp, _ = field("fill", fill)
            d = d * fexp
        if "aspect" in s:
            d = d * float(np.min(s["aspect"]))
        d = d * cexp
        out["cell"] = cinfo or round(cexp, 4)
        out["coordinates"] = {"grid": h, "voxels": list(shape[::-1]), "phases": "sin/cos"}
    elif graded:
        # the lattice's own coordinates (cell units) and its cell, as images
        U, cl = g.lattice_coords(nd, P)
        Ux = [field(f"u{i}", U[:, i], const_ok=False)[0] for i in range(3)]
        cexp, cinfo = field("cell", cl)
        d = tpms_unit(tp, *Ux)
        if "aspect" in s:
            d = d * float(np.min(s["aspect"]))
        d = d * cexp
        out["cell"] = cinfo or round(cexp, 4)
        out["coordinates"] = {"grid": h, "voxels": list(shape[::-1])}
    else:
        if "aspect" in s:
            a = np.asarray(s["aspect"], float)
            Uc = [(POS[ax] - o) / (cell0 * a_) for ax, o, a_ in zip("xyz", origin, a)]
            d = tpms_unit(tp, *Uc) * float(np.min(a)) * cell0
        else:
            Uc = [(POS[ax] - o) / cell0 for ax, o in zip("xyz", origin)]
            d = tpms_unit(tp, *Uc) * cell0
        out["cell"] = cell0
    if "offset" in r:
        oexp, _ = field("offset", np.broadcast_to(np.asarray(g._val(r["offset"], P, {}), float), (len(P),)))
        d = d - oexp
    t = wall(P)
    texp, tinfo = field("wall", t)
    out["thickness"] = tinfo or round(texp, 4)
    out["expr"] = _wall(d, texp, s.get("form", "sheet"))
    out["images"] = images
    out["minFeature"] = round(0.5 * float(np.min(t)), 4)
    return out


def blend_of(g, name):
    """The blend radius where the graph unions lattice `name` into the
    rest: the first blended union with an input that depends on the lattice
    and one that does not. (number, or a node name, or None)."""
    memo = {}

    def deps(n):
        if n in memo:
            return memo[n]
        memo[n] = False
        nd = g.nodes.get(n)
        if nd is None:
            return False
        if n == name:
            memo[n] = True
            return True
        hit = False
        for v in (nd.refs or {}).values():
            for x in (v if isinstance(v, list) else v.values() if isinstance(v, dict) else [v]):
                if isinstance(x, str) and deps(x):
                    hit = True
        memo[n] = hit
        return hit

    for nd in g.nodes.values():
        if nd.op != "union" or "blend" not in nd.spec:
            continue
        ins = [x for x in nd.refs.get("inputs", []) if isinstance(x, str)]
        with_ = [deps(x) for x in ins]
        if any(with_) and not all(with_):
            return nd.refs["blend"]
    return None
