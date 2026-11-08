"""
FE model export: a delivered part, meshed in quadratic tetrahedra, with the
study's material, supports and load cases, as a deck another solver runs.

- Nastran bulk data (`.bdf`, `.nas`, `.dat`): SOL 101, one SUBCASE per load
  case. GRID* / FORCE* in large-field format (16-character reals), CTETRA
  10-node, MAT1 (or MAT9 for a layered process: the same transversely
  isotropic stiffness loadpath analyses with, in the global frame), SPC1
  per support, and a pin as a cylindrical frame (CORD2C) on its grids with
  the radial and axial components held. Units mm, N, MPa, tonne/mm^3.
- Abaqus / CalculiX input (`.inp`): the deck the tetrahedral cross-check
  runs CalculiX on (crosscheck.write_abaqus), one static step per case.

The mesh, held nodes and nodal forces are exactly the cross-check's, so a
deck run elsewhere answers the same problem loadpath's second opinion does
(`loadpath crosscheck --backend ccx` runs the Abaqus form here).
"""
from __future__ import annotations

import time

import numpy as np

from .spec import Study

__all__ = ["fe_model", "export_fe", "write_nastran", "read_nastran"]


def fe_model(study: Study, mesh, size: float | None = None, mesher: str | None = None) -> dict:
    """The study on a quadratic tet mesh of `mesh`: nodes (vertices first,
    then edge mid-nodes), conn (n, 10), supports [(held, fix, pin)],
    loads [(name, force per node (n, 3), load vertices)], el (elasticity),
    size."""
    from .crosscheck import (_boundary_nodes, _load_vectors, _p2_mesh, _supports_on, _tet_mesh,
                             element_size, fit_size)

    asked = size or element_size(study)
    size = asked if size else fit_size(float(mesh.volume), asked)
    pts, tets = _tet_mesh(mesh, size, mesher=mesher)
    surf = _boundary_nodes(tets)
    tol = 0.6 * size
    loads = _load_vectors(study, pts, surf, tol)
    nodes, conn, edges = _p2_mesh(pts, tets)
    # a load on a corner node only (the cross-check's lumping): the
    # mid-nodes carry none, as the native solve has it
    loads = [(n, np.vstack([f, np.zeros((len(nodes) - len(pts), 3))]), ln) for n, f, ln in loads]
    _fixed, supports, _blocks, _held = _supports_on(study, pts, nodes, edges, surf, tol)
    return {"nodes": nodes, "conn": conn, "supports": supports, "loads": loads, "el": study.elastic(study.material),
            "size": size, "vertices": len(pts)}


def _large(name: str, fields) -> str:
    """A large-field card: the name with '*', then 16-character fields,
    four to a line, continued with '*' lines."""
    def fmt(v):
        if v is None:
            return " " * 16
        if isinstance(v, (int, np.integer)):
            return f"{int(v):>16d}"
        if isinstance(v, str):
            return f"{v:<16s}"
        return f"{float(v):16.9E}"[:16]
    out, row = [], f"{name + '*':<8s}"
    for i, v in enumerate(fields):
        if i and i % 4 == 0:
            out.append(row)
            row = f"{'*':<8s}"
        row += fmt(v)
    out.append(row)
    return "\n".join(out) + "\n"


def _mat9(el, E):
    """The layered elasticity as MAT9's 21 upper-triangle terms, row by row
    (strain order xx, yy, zz, xy, yz, zx, engineering shears; Nastran's)."""
    c = E * np.asarray(el.c() if callable(el.c) else el.c, float)
    # loadpath's Voigt order is xx, yy, zz, xy, yz, zx, as Nastran's
    return [c[i, j] for i in range(6) for j in range(i, 6)]


def write_nastran(fh, study: Study, model: dict, title: str = "loadpath") -> None:
    """The model as Nastran bulk data (SOL 101)."""
    mat = study.material
    nodes, conn, supports, loads, el = model["nodes"], model["conn"], model["supports"], model["loads"], model["el"]
    fh.write(f"$ {title}: written by loadpath (mm, N, MPa, tonne/mm^3)\n")
    fh.write("SOL 101\nCEND\n")
    fh.write(f"TITLE = {title[:64]}\nECHO = NONE\nSPC = 1\nDISPLACEMENT(PLOT) = ALL\nSTRESS(PLOT) = ALL\n"
             "SPCFORCES(PLOT) = ALL\n")
    for k, (name, _f, _l) in enumerate(loads):
        fh.write(f"SUBCASE {k + 1}\n  LABEL = {str(name)[:64]}\n  LOAD = {100 + k + 1}\n")
    fh.write("BEGIN BULK\nPARAM,POST,-1\n")
    # pins: a cylindrical frame per pin, on its grids (r, theta, z)
    cd = np.zeros(len(nodes), int)
    cid = 0
    for held, _fix, pin in supports:
        if pin is None:
            continue
        cid += 1
        a = np.asarray(pin[0], float)
        ax = np.asarray(pin[1], float)
        b = a + ax
        ref = np.array([1.0, 0, 0]) if abs(ax[0]) < 0.9 else np.array([0, 1.0, 0])
        c = a + np.cross(ax, np.cross(ref, ax))
        fh.write(_large("CORD2C", [cid, 0, *a, *b, *c]))
        cd[held[cd[held] == 0]] = cid
    for i, (x, y, z) in enumerate(nodes):
        fh.write(_large("GRID", [i + 1, 0, float(x), float(y), float(z), int(cd[i]) or None]))
    for i, row in enumerate(conn):
        g = (row + 1).tolist()
        fh.write(_large("CTETRA", [i + 1, 1, *g]))
    fh.write(_large("PSOLID", [1, 1]))
    from .fem import Layered

    rho = getattr(mat, "density", None)
    rho = None if rho is None else float(rho) * 1e-9          # g/cm^3 -> tonne/mm^3
    if isinstance(el, Layered):
        fh.write(_large("MAT9", [1, *_mat9(el, mat.E), rho]))
    else:
        nu = float(el)
        fh.write(_large("MAT1", [1, float(mat.E), None, nu, rho]))
    for held, fix, pin in supports:
        comps = "".join(str(a + 1) for a in range(3) if fix[a])
        if pin is not None:
            # radial and axial in the pin's frame; an axis the study also
            # holds in global terms stays with the frame (as CalculiX's)
            comps = "".join(sorted(set(comps) | {"1", "3"}))
        if not comps:
            continue
        ids = (held + 1).tolist()
        for s in range(0, len(ids), 60):
            fh.write(_large("SPC1", [1, int(comps), *ids[s:s + 60]]))
    for k, (_name, f, _l) in enumerate(loads):
        sid = 100 + k + 1
        i = np.nonzero(np.abs(f).sum(axis=1))[0]
        for n in i:
            # in the basic frame, whatever frame the grid's output is in
            fh.write(_large("FORCE", [sid, int(n + 1), 0, 1.0, *map(float, f[n])]))
    fh.write("ENDDATA\n")


def read_nastran(path) -> dict:
    """Read back what write_nastran writes (GRID*, CTETRA, MAT1/MAT9, SPC1,
    FORCE*, CORD2C, SUBCASE labels): {nodes, conn, spc: [(comps, ids)],
    forces: {sid: (ids, vectors)}, mat, cords, subcases}. For round trips
    and tests; not a general bulk-data reader."""
    lines = open(path).read().splitlines()
    cards, i = [], 0
    while i < len(lines):
        ln = lines[i]
        if ln.startswith("$") or not ln.strip():
            i += 1
            continue
        if ln[:8].rstrip().endswith("*") and not ln.startswith("*"):
            name = ln[:8].rstrip()[:-1]
            body = ln[8:]
            fields = [body[j:j + 16].strip() for j in range(0, 64, 16)]
            i += 1
            while i < len(lines) and lines[i].startswith("*"):
                body = lines[i][8:]
                fields += [body[j:j + 16].strip() for j in range(0, 64, 16)]
                i += 1
            cards.append((name, fields))
            continue
        cards.append(("CASE", [ln]))
        i += 1

    def num(v):
        return float(v) if v else None
    nodes, conn, spc, forces, cords, subcases, mat = {}, [], [], {}, {}, [], {}
    for name, f in cards:
        if name == "GRID":
            nodes[int(f[0])] = (float(f[2]), float(f[3]), float(f[4]), int(f[5]) if len(f) > 5 and f[5] else 0)
        elif name == "CTETRA":
            conn.append([int(v) for v in f[2:12]])
        elif name == "SPC1":
            spc.append((f[1], [int(v) - 1 for v in f[2:] if v]))
        elif name == "FORCE":
            sid = int(f[0])
            ids, vec = forces.setdefault(sid, ([], []))
            ids.append(int(f[1]))
            vec.append([float(f[3]) * float(v) for v in f[4:7]])
        elif name == "CORD2C":
            cords[int(f[0])] = [float(v) for v in f[2:11]]
        elif name == "MAT1":
            mat = {"type": "MAT1", "E": float(f[1]), "nu": float(f[3]), "rho": num(f[4])}
        elif name == "MAT9":
            mat = {"type": "MAT9", "G": [float(v) for v in f[1:22]]}
        elif name == "CASE" and f[0].strip().startswith("LABEL"):
            subcases.append(f[0].split("=", 1)[1].strip())
    ids = sorted(nodes)
    return {"nodes": np.array([nodes[k][:3] for k in ids]), "cd": np.array([nodes[k][3] for k in ids]),
            "conn": np.array(conn, int) - 1, "spc": spc,
            "forces": {k: (np.array(v[0]) - 1, np.array(v[1])) for k, v in forces.items()},
            "mat": mat, "cords": cords, "subcases": subcases}


def export_fe(study: Study, mesh, path: str, fmt: str | None = None, size: float | None = None,
              mesher: str | None = None) -> dict:
    """Write the part's FE model to `path`: Nastran (.bdf, .nas, .dat) or
    Abaqus / CalculiX (.inp), by `fmt` or the extension. Returns a summary
    (format, nodes, elements, element size, cases, supports, seconds)."""
    t0 = time.time()
    ext = str(path).rsplit(".", 1)[-1].lower()
    fmt = (fmt or {"bdf": "nastran", "nas": "nastran", "dat": "nastran", "inp": "abaqus"}.get(ext, "")).lower()
    if fmt not in ("nastran", "abaqus"):
        raise ValueError("format must be nastran (.bdf, .nas, .dat) or abaqus (.inp)")
    if study.self_equilibrated:
        raise ValueError("the deck needs supports: a part held by pressure alone or inertia relief has none to "
                         "write (Nastran's would take a SUPORT card)")
    model = fe_model(study, mesh, size, mesher)
    title = getattr(study, "name", None) or "loadpath"
    with open(path, "w") as fh:
        if fmt == "nastran":
            write_nastran(fh, study, model, title=title)
        else:
            from .crosscheck import write_abaqus

            write_abaqus(fh, study.material, model["nodes"], model["conn"], model["supports"], model["loads"],
                         model["el"], heading=f"{title}: written by loadpath (mm, N, MPa)")
    return {"format": fmt, "path": str(path), "nodes": int(len(model["nodes"])), "elements": int(len(model["conn"])),
            "element": "CTETRA (10-node)" if fmt == "nastran" else "C3D10", "elementSize": model["size"],
            "cases": [n for n, _f, _l in model["loads"]], "supports": len(model["supports"]),
            "units": "mm, N, MPa, tonne/mm^3", "seconds": round(time.time() - t0, 2)}
