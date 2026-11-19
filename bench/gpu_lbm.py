"""
How fast would loadpath's lattice Boltzmann run on this machine's GPU?

One self-contained measurement: the same D3Q19 step (pull streaming, halfway
bounce-back at a sphere, regularized collision, Smagorinsky subgrid viscosity,
float32, the ingredients of loadpath/windtunnel.py's kernel without its
nesting and curved walls) written twice:

- for the CPU, as a numba kernel on all cores (what loadpath ships today);
- for the GPU, as a WGSL compute shader run through wgpu (pip install wgpu:
  Metal on a Mac, Vulkan on Linux, D3D12 on Windows, no vendor SDK).

It checks the two give the same field, then times both.

    pip install numpy numba wgpu
    python gpu_lbm.py            # 64, 96, 128 cells a side
    python gpu_lbm.py 160 200    # other sizes
    python gpu_lbm.py --cpu      # the CPU kernel alone
    python gpu_lbm.py --group=4,4,4   # another workgroup size

MLUPS = million lattice-cell updates per second.
"""
import json
import platform
import sys
import time

import numpy as np

C = np.array([[0, 0, 0], [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1],
              [1, 1, 0], [-1, -1, 0], [1, -1, 0], [-1, 1, 0], [1, 0, 1], [-1, 0, -1], [1, 0, -1], [-1, 0, 1],
              [0, 1, 1], [0, -1, -1], [0, 1, -1], [0, -1, 1]], np.int32)
W = np.array([1 / 3] + [1 / 18] * 6 + [1 / 36] * 12, np.float32)
OPP = np.array([0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15, 18, 17], np.int32)
TAU0 = 0.52
CS2 = 0.17 ** 2
U0 = 0.05


def setup(n):
    x = np.arange(n) - n / 2 + 0.5
    X, Y, Z = np.meshgrid(x - n / 6, x, x, indexing="ij")
    solid = (X * X + Y * Y + Z * Z) < (n / 8) ** 2
    u = np.array([U0, 0.0, 0.0], np.float32)
    cu = C @ u
    feq = (W * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * float(u @ u))).astype(np.float32)
    f = np.empty((19, n, n, n), np.float32)
    f[:] = feq[:, None, None, None]
    return f, solid


# ------------------------------------------------------------------- CPU --

def cpu_kernel():
    import numba as nb
    f32 = np.float32
    Cc, Wc, Oc = C.copy(), W.copy(), OPP.copy()

    @nb.njit(parallel=True, fastmath=True, cache=False)
    def step(f, fn, solid, tau0, cs2):
        _, nx, ny, nz = f.shape
        for x in nb.prange(nx):
            fin = np.empty(19, f32)
            for y in range(ny):
                for z in range(nz):
                    if solid[x, y, z]:
                        continue
                    for i in range(19):
                        sx = (x - Cc[i, 0]) % nx
                        sy = (y - Cc[i, 1]) % ny
                        sz = (z - Cc[i, 2]) % nz
                        if solid[sx, sy, sz]:
                            fin[i] = f[Oc[i], x, y, z]
                        else:
                            fin[i] = f[i, sx, sy, sz]
                    rho = f32(0.0)
                    jx = f32(0.0)
                    jy = f32(0.0)
                    jz = f32(0.0)
                    for i in range(19):
                        rho += fin[i]
                        jx += fin[i] * Cc[i, 0]
                        jy += fin[i] * Cc[i, 1]
                        jz += fin[i] * Cc[i, 2]
                    ux = jx / rho
                    uy = jy / rho
                    uz = jz / rho
                    usq = ux * ux + uy * uy + uz * uz
                    pxx = f32(0.0)
                    pyy = f32(0.0)
                    pzz = f32(0.0)
                    pxy = f32(0.0)
                    pxz = f32(0.0)
                    pyz = f32(0.0)
                    for i in range(19):
                        cu = Cc[i, 0] * ux + Cc[i, 1] * uy + Cc[i, 2] * uz
                        feq = Wc[i] * rho * (f32(1.0) + f32(3.0) * cu + f32(4.5) * cu * cu - f32(1.5) * usq)
                        d = fin[i] - feq
                        pxx += d * Cc[i, 0] * Cc[i, 0]
                        pyy += d * Cc[i, 1] * Cc[i, 1]
                        pzz += d * Cc[i, 2] * Cc[i, 2]
                        pxy += d * Cc[i, 0] * Cc[i, 1]
                        pxz += d * Cc[i, 0] * Cc[i, 2]
                        pyz += d * Cc[i, 1] * Cc[i, 2]
                    pn = np.sqrt(pxx * pxx + pyy * pyy + pzz * pzz + f32(2.0) * (pxy * pxy + pxz * pxz + pyz * pyz))
                    tau = f32(0.5) * (tau0 + np.sqrt(tau0 * tau0 + f32(25.455844) * cs2 * pn / rho))
                    keep = f32(1.0) - f32(1.0) / tau
                    tr = (pxx + pyy + pzz) / f32(3.0)
                    for i in range(19):
                        cx = f32(Cc[i, 0])
                        cy = f32(Cc[i, 1])
                        cz = f32(Cc[i, 2])
                        cu = cx * ux + cy * uy + cz * uz
                        feq = Wc[i] * rho * (f32(1.0) + f32(3.0) * cu + f32(4.5) * cu * cu - f32(1.5) * usq)
                        q = (cx * cx * pxx + cy * cy * pyy + cz * cz * pzz
                             + f32(2.0) * (cx * cy * pxy + cx * cz * pxz + cy * cz * pyz) - tr)
                        fn[i, x, y, z] = feq + keep * Wc[i] * f32(4.5) * q
    return step


# ------------------------------------------------------------------- GPU --

def wgsl(group=(8, 8, 4)):
    """The step as straight-line WGSL: every direction written out with its
    own literal offsets (no loops over constant arrays, no integer modulo),
    and the walls as one bit per link in a mask per cell."""
    L = ["struct Params { n: vec4<u32>, tau0: f32, cs2: f32, pad0: f32, pad1: f32 };",
         "@group(0) @binding(0) var<storage, read> fin: array<f32>;",
         "@group(0) @binding(1) var<storage, read_write> fout: array<f32>;",
         "@group(0) @binding(2) var<storage, read> walls: array<u32>;",
         "@group(0) @binding(3) var<uniform> p: Params;",
         f"@compute @workgroup_size({group[0]}, {group[1]}, {group[2]})",
         "fn main(@builtin(global_invocation_id) id: vec3<u32>) {",
         "  let nx = p.n.x; let ny = p.n.y; let nz = p.n.z;",
         "  if (id.x >= nx || id.y >= ny || id.z >= nz) { return; }",
         "  let N = nx * ny * nz;",
         "  let cell = (id.x * ny + id.y) * nz + id.z;",
         "  let w = walls[cell];",
         "  if ((w & 0x80000000u) != 0u) { return; }",
         "  let xm = select(id.x - 1u, nx - 1u, id.x == 0u); let xp = select(id.x + 1u, 0u, id.x == nx - 1u);",
         "  let ym = select(id.y - 1u, ny - 1u, id.y == 0u); let yp = select(id.y + 1u, 0u, id.y == ny - 1u);",
         "  let zm = select(id.z - 1u, nz - 1u, id.z == 0u); let zp = select(id.z + 1u, 0u, id.z == nz - 1u);"]
    pick = lambda c, m, o, p_: m if c > 0 else p_ if c < 0 else o       # the source is at x - c
    for i in range(19):
        cx, cy, cz = (int(v) for v in C[i])
        src = f"({pick(cx, 'xm', 'id.x', 'xp')} * ny + {pick(cy, 'ym', 'id.y', 'yp')}) * nz + {pick(cz, 'zm', 'id.z', 'zp')}"
        if i == 0:
            L.append("  let f0 = fin[cell];")
        else:
            L.append(f"  var f{i} = fin[{i}u * N + {src}];")
            L.append(f"  if ((w & {1 << i}u) != 0u) {{ f{i} = fin[{int(OPP[i])}u * N + cell]; }}")
    def lin(k):
        pos = [f"f{i}" for i in range(19) if C[i][k] > 0]
        neg = [f"f{i}" for i in range(19) if C[i][k] < 0]
        return "(" + " + ".join(pos) + ") - (" + " + ".join(neg) + ")"
    L.append("  let rho = " + " + ".join(f"f{i}" for i in range(19)) + ";")
    L.append(f"  let ux = ({lin(0)}) / rho; let uy = ({lin(1)}) / rho; let uz = ({lin(2)}) / rho;")
    L.append("  let usq = ux * ux + uy * uy + uz * uz;")
    def term(c, name):
        return "" if c == 0 else (f" + {name}" if c > 0 else f" - {name}")
    for i in range(19):
        cx, cy, cz = (int(v) for v in C[i])
        cu = (term(cx, "ux") + term(cy, "uy") + term(cz, "uz")).strip()
        cu = cu[2:] if cu.startswith("+ ") else ("-" + cu[2:] if cu.startswith("- ") else "0.0")
        L.append(f"  let cu{i} = {cu};")
        L.append(f"  let e{i} = {float(W[i])!r} * rho * (1.0 + 3.0 * cu{i} + 4.5 * cu{i} * cu{i} - 1.5 * usq);")
        L.append(f"  let d{i} = f{i} - e{i};")
    def mom(a, b):
        t = [(int(C[i][a] * C[i][b]), i) for i in range(19) if C[i][a] * C[i][b] != 0]
        return " ".join(("+ " if s > 0 else "- ") + f"d{i}" for s, i in t)[2:] if t[0][0] > 0 else "0.0 " + " ".join(
            ("+ " if s > 0 else "- ") + f"d{i}" for s, i in t)
    for nm, (a, b) in {"pxx": (0, 0), "pyy": (1, 1), "pzz": (2, 2), "pxy": (0, 1), "pxz": (0, 2), "pyz": (1, 2)}.items():
        L.append(f"  let {nm} = {mom(a, b)};")
    L.append("  let pn = sqrt(pxx * pxx + pyy * pyy + pzz * pzz + 2.0 * (pxy * pxy + pxz * pxz + pyz * pyz));")
    L.append("  let tau = 0.5 * (p.tau0 + sqrt(p.tau0 * p.tau0 + 25.455844 * p.cs2 * pn / rho));")
    L.append("  let keep = (1.0 - 1.0 / tau) * 4.5;")
    L.append("  let tr = (pxx + pyy + pzz) / 3.0;")
    for i in range(19):
        cx, cy, cz = (int(v) for v in C[i])
        q = []
        for c, nm in ((cx * cx, "pxx"), (cy * cy, "pyy"), (cz * cz, "pzz")):
            if c:
                q.append(f"+ {nm}")
        for c, nm in ((cx * cy, "pxy"), (cx * cz, "pxz"), (cy * cz, "pyz")):
            if c:
                q.append(("+ " if c > 0 else "- ") + f"2.0 * {nm}")
        L.append(f"  fout[{i}u * N + cell] = e{i} + keep * {float(W[i])!r} * ({' '.join(q)[2:] if q else '0.0'} - tr);")
    L.append("}")
    return "\n".join(L)


def wall_mask(solid):
    """Bit i set where the link a cell pulls direction i along starts in a
    wall; the top bit where the cell itself is solid."""
    w = np.zeros(solid.shape, np.uint32)
    for i in range(1, 19):
        src = np.roll(solid, shift=tuple(int(v) for v in C[i]), axis=(0, 1, 2))     # src[x] = solid[x - c]
        w |= src.astype(np.uint32) << np.uint32(i)
    w |= solid.astype(np.uint32) << np.uint32(31)
    return w


class Gpu:
    def __init__(self, group=(8, 8, 4)):
        import wgpu
        self.wgpu = wgpu
        self.adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
        info = self.adapter.info
        self.name = f"{info.get('device', '')} ({info.get('backend_type', '')}, {info.get('adapter_type', '')})"
        lim = self.adapter.limits
        want = {}
        for key in lim:
            k = key.replace("-", "_")
            if k in ("max_storage_buffer_binding_size", "max_buffer_size"):
                want[key] = lim[key]
        self.max_binding = int(next((lim[k] for k in lim if k.replace("-", "_") == "max_storage_buffer_binding_size"),
                                    128 << 20))
        self.device = self.adapter.request_device_sync(required_limits=want)
        self.module = self.device.create_shader_module(code=wgsl(group))
        self.group = group
        self.pipeline = self.device.create_compute_pipeline(layout="auto",
                                                            compute={"module": self.module, "entry_point": "main"})

    def load(self, f, solid):
        wgpu, dev = self.wgpu, self.device
        n = f.shape[1]
        self.n = n
        use = wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC | wgpu.BufferUsage.COPY_DST
        self.bufs = [dev.create_buffer_with_data(data=np.ascontiguousarray(f), usage=use) for _ in range(2)]
        sb = dev.create_buffer_with_data(data=np.ascontiguousarray(wall_mask(solid)), usage=wgpu.BufferUsage.STORAGE)
        par = np.zeros(8, np.uint32)
        par[:3] = n
        par[4:6] = np.array([TAU0, CS2], np.float32).view(np.uint32)
        pb = dev.create_buffer_with_data(data=par, usage=wgpu.BufferUsage.UNIFORM)
        lay = self.pipeline.get_bind_group_layout(0)

        def group(a, b):
            return dev.create_bind_group(layout=lay, entries=[
                {"binding": 0, "resource": {"buffer": a, "offset": 0, "size": a.size}},
                {"binding": 1, "resource": {"buffer": b, "offset": 0, "size": b.size}},
                {"binding": 2, "resource": {"buffer": sb, "offset": 0, "size": sb.size}},
                {"binding": 3, "resource": {"buffer": pb, "offset": 0, "size": pb.size}}])
        self.groups = [group(self.bufs[0], self.bufs[1]), group(self.bufs[1], self.bufs[0])]
        self.at = 0

    def step(self, steps):
        gx, gy, gz = (-(-self.n // g) for g in self.group)
        done = 0
        while done < steps:
            enc = self.device.create_command_encoder()
            for _ in range(min(16, steps - done)):
                cp = enc.begin_compute_pass()
                cp.set_pipeline(self.pipeline)
                cp.set_bind_group(0, self.groups[self.at])
                cp.dispatch_workgroups(gx, gy, gz)
                cp.end()
                self.at ^= 1
                done += 1
            self.device.queue.submit([enc.finish()])

    def sync(self):
        """Wait for the queue: read four bytes back."""
        return bytes(self.device.queue.read_buffer(self.bufs[self.at], 0, 4))

    def read(self):
        n = self.n
        return np.frombuffer(self.device.queue.read_buffer(self.bufs[self.at]), np.float32).reshape(19, n, n, n).copy()


def main():
    sizes = [int(a) for a in sys.argv[1:] if a.isdigit()] or [64, 96, 128]
    quick = "--quick" in sys.argv
    out = {"machine": platform.platform(), "processor": platform.processor() or platform.machine(), "rows": []}
    try:
        import numba
        step = cpu_kernel()
        out["cpuThreads"] = numba.get_num_threads()
    except ImportError:
        step = None
        print("numba missing: GPU only")
    group = (8, 8, 4)
    for a in sys.argv[1:]:
        if a.startswith("--group="):
            group = tuple(int(v) for v in a.split("=")[1].split(","))
    out["workgroup"] = list(group)
    try:
        if "--cpu" in sys.argv:
            raise RuntimeError("skipped (--cpu)")
        gpu = Gpu(group)
        out["gpu"] = gpu.name
        print("GPU:", gpu.name, " largest buffer binding", gpu.max_binding >> 20, "MB")
    except Exception as e:
        gpu = None
        print("no GPU through wgpu:", repr(e)[:300])
    for n in sizes:
        f0, solid = setup(n)
        cells = int((~solid).sum())
        row = {"n": n, "cells": cells}
        check = 20
        ref = None
        if step is not None:
            f, fn = f0.copy(), f0.copy()
            step(f, fn, solid, np.float32(TAU0), np.float32(CS2))        # compile
            f = f0.copy()
            for _ in range(check):
                step(f, fn, solid, np.float32(TAU0), np.float32(CS2))
                f, fn = fn, f
            ref = f.copy()
            steps = 4 if quick else max(10, int(3e8 / max(cells, 1) / 4))
            t = time.perf_counter()
            for _ in range(steps):
                step(f, fn, solid, np.float32(TAU0), np.float32(CS2))
                f, fn = fn, f
            dt = time.perf_counter() - t
            row["cpuMlups"] = round(cells * steps / dt / 1e6, 1)
        if gpu is not None and f0.nbytes <= gpu.max_binding:
            try:
                gpu.load(f0, solid)
                gpu.step(check)
                got = gpu.read()
                if ref is not None:
                    fl = ~solid
                    row["maxDifference"] = float(np.abs(got[:, fl] - ref[:, fl]).max())
                steps = 4 if quick else max(32, int(4e9 / max(cells, 1) / 16) * 16)
                gpu.step(16)
                gpu.sync()
                t = time.perf_counter()
                gpu.step(steps)
                gpu.sync()
                dt = time.perf_counter() - t
                row["gpuMlups"] = round(cells * steps / dt / 1e6, 1)
                row["gpuSteps"] = steps
            except Exception as e:
                row["gpuError"] = repr(e)[:300]
        elif gpu is not None:
            row["gpuError"] = f"{f0.nbytes >> 20} MB of populations is over this device's buffer limit"
        print(row, flush=True)
        out["rows"].append(row)
    print(json.dumps(out))
    if "--json" in sys.argv:
        with open(sys.argv[sys.argv.index("--json") + 1], "w") as fh:
            json.dump(out, fh, indent=1)


if __name__ == "__main__":
    main()
