"""
A fixed camera and a z-buffer: the same view of a part on every build.

Comparing a part against a reference image needs the part drawn from the
camera the reference was drawn or rendered from, every time, to the pixel.
`Camera` is that camera (a position, what it looks at, a lens or an
orthographic width, an image size); `render` draws triangles through it
with a z-buffer (numba), returning per pixel the depth, the triangle and a
shaded grey: enough for silhouettes, outlines, depth comparisons and clay
images an agent or a person can judge. No GL context, so it runs headless
and in CI.

A camera in a spec:

    {"position": [-140, 150, 50], "lookAt": [0, 8, 6], "lens": 85,
     "width": 1344, "height": 768}                      perspective
    {"direction": "top", "width": 800, "height": 800}   orthographic, framed on the part

`lens` is a focal length in mm on a 36 mm sensor across the image's longer
side (Blender's default, and a full-frame camera's); `fov` (degrees across
the longer side) does the same job. `ortho` is the width in mm an
orthographic image spans. `direction` is one of top, bottom, front, back,
left, right, or a vector the camera looks along; with no `ortho` the view
is framed on what is drawn.
"""
from __future__ import annotations

import math

import numpy as np

try:
    from numba import njit
except Exception:  # noqa: BLE001
    def njit(*a, **k):
        return (lambda f: f) if not (a and callable(a[0])) else a[0]

__all__ = ["Camera", "render", "shade_image", "outline", "DIRECTIONS"]

# name: (the way the camera looks, image up)
DIRECTIONS = {
    "top": ((0, 0, -1), (0, 1, 0)), "bottom": ((0, 0, 1), (0, 1, 0)),
    "front": ((0, 1, 0), (0, 0, 1)), "back": ((0, -1, 0), (0, 0, 1)),
    "right": ((-1, 0, 0), (0, 0, 1)), "left": ((1, 0, 0), (0, 0, 1)),
}


def _unit(v):
    v = np.asarray(v, float)
    n = np.linalg.norm(v)
    if n < 1e-12:
        raise ValueError("a camera direction cannot be zero")
    return v / n


class Camera:
    """A pinhole or orthographic camera. `project(P)` gives pixel x, y
    (x right, y down) and depth along the view (mm in front of the eye)."""

    def __init__(self, position=None, look_at=None, up=None, lens=None, fov=None, ortho=None, direction=None,
                 width=800, height=800, sensor=36.0):
        self.width, self.height = int(width), int(height)
        if self.width < 8 or self.height < 8:
            raise ValueError("a camera image is at least 8 x 8 pixels")
        self.ortho = None if ortho is None else float(ortho)
        self.frame = False
        if direction is not None:
            if isinstance(direction, str):
                if direction not in DIRECTIONS:
                    raise ValueError(f"camera.direction is one of {sorted(DIRECTIONS)} or a vector, got {direction!r}")
                fwd, up0 = DIRECTIONS[direction]
                up = up if up is not None else up0
            else:
                fwd = direction
            self.forward = _unit(fwd)
            self.position = None if position is None else np.asarray(position, float)
            if self.ortho is None and lens is None and fov is None:
                self.frame = True          # orthographic, framed on what is drawn
            if self.position is None and (lens is not None or fov is not None):
                raise ValueError("a perspective camera needs a position (and lookAt)")
        else:
            if position is None or look_at is None:
                raise ValueError("a camera is {position, lookAt, lens | fov | ortho} or {direction}")
            self.position = np.asarray(position, float)
            self.forward = _unit(np.asarray(look_at, float) - self.position)
        upv = np.asarray(up if up is not None else (0, 0, 1), float)
        if abs(float(np.dot(_unit(upv), self.forward))) > 0.999:
            upv = np.array([0.0, 1.0, 0.0])     # looking straight down the up axis
        self.right = _unit(np.cross(self.forward, upv))
        self.up = np.cross(self.right, self.forward)
        longer = max(self.width, self.height)
        if self.ortho is None and not self.frame:
            if fov is not None:
                self.focal = 0.5 * longer / math.tan(math.radians(float(fov)) / 2)
            else:
                self.focal = float(lens if lens is not None else 50.0) / float(sensor) * longer
        else:
            self.focal = None
        self._centre = np.zeros(3) if self.position is None else self.position

    @classmethod
    def from_spec(cls, d: dict):
        if not isinstance(d, dict):
            raise ValueError("camera is an object: {position, lookAt, lens} or {direction}")
        return cls(position=d.get("position"), look_at=d.get("lookAt"), up=d.get("up"), lens=d.get("lens"),
                   fov=d.get("fov"), ortho=d.get("ortho"), direction=d.get("direction"),
                   width=d.get("width", 800), height=d.get("height", 800), sensor=d.get("sensor", 36.0))

    @property
    def perspective(self) -> bool:
        return self.focal is not None

    def fit(self, points, margin: float = 1.12):
        """An orthographic camera with no width frames these points."""
        if not self.frame and self.position is not None:
            return self
        P = np.asarray(points, float).reshape(-1, 3)
        u, v = P @ self.right, P @ self.up
        cu, cv = 0.5 * (u.min() + u.max()), 0.5 * (v.min() + v.max())
        if self.frame:
            self.ortho = margin * max(np.ptp(u), np.ptp(v) * self.width / self.height, 1e-6)
            self.frame = False
        back = float((P @ self.forward).min()) - 10.0
        self._centre = cu * self.right + cv * self.up + back * self.forward
        self.position = self._centre
        return self

    def project(self, P):
        P = np.asarray(P, float).reshape(-1, 3)
        if self.frame or self.position is None:
            self.fit(P)
        d = P - self.position
        x, y, z = d @ self.right, d @ self.up, d @ self.forward
        if self.perspective:
            zc = np.maximum(z, 1e-6)
            px = self.focal * x / zc
            py = self.focal * y / zc
        else:
            s = self.width / self.ortho
            px, py = s * x, s * y
        return np.stack([0.5 * self.width + px, 0.5 * self.height - py, z], axis=1)

    def mm_per_pixel(self, depth: float) -> float:
        """What one pixel spans at `depth` mm in front of the camera."""
        if self.perspective:
            return float(depth) / self.focal
        return self.ortho / self.width

    def to_spec(self) -> dict:
        out = {"position": [round(float(x), 4) for x in self.position],
               "lookAt": [round(float(x), 4) for x in self.position + 100.0 * self.forward],
               "up": [round(float(x), 4) for x in self.up], "width": self.width, "height": self.height}
        if self.perspective:
            out["fov"] = round(math.degrees(2 * math.atan(0.5 * max(self.width, self.height) / self.focal)), 4)
        else:
            out["ortho"] = round(float(self.ortho), 4)
        return out


@njit(cache=True)
def _raster(S, W, H, depth, owner):
    """S: (n, 3, 3) screen x, y, depth per corner. Nearest depth wins."""
    for t in range(S.shape[0]):
        x0, y0, z0 = S[t, 0, 0], S[t, 0, 1], S[t, 0, 2]
        x1, y1, z1 = S[t, 1, 0], S[t, 1, 1], S[t, 1, 2]
        x2, y2, z2 = S[t, 2, 0], S[t, 2, 1], S[t, 2, 2]
        if z0 <= 1e-6 or z1 <= 1e-6 or z2 <= 1e-6:
            continue
        den = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
        if abs(den) < 1e-12:
            continue
        ia = max(int(math.floor(min(x0, x1, x2))), 0)
        ib = min(int(math.ceil(max(x0, x1, x2))), W - 1)
        ja = max(int(math.floor(min(y0, y1, y2))), 0)
        jb = min(int(math.ceil(max(y0, y1, y2))), H - 1)
        for j in range(ja, jb + 1):
            py = j + 0.5
            for i in range(ia, ib + 1):
                px = i + 0.5
                a = ((y1 - y2) * (px - x2) + (x2 - x1) * (py - y2)) / den
                b = ((y2 - y0) * (px - x2) + (x0 - x2) * (py - y2)) / den
                c = 1.0 - a - b
                if a < -1e-9 or b < -1e-9 or c < -1e-9:
                    continue
                z = a * z0 + b * z1 + c * z2
                if z < depth[j, i]:
                    depth[j, i] = z
                    owner[j, i] = t


def render(meshes, camera: Camera):
    """Draw one mesh or a list through `camera`. Returns {"depth" (H, W; inf
    where nothing is), "mask", "owner" (triangle index into the stacked
    meshes, -1 for none), "which" (mesh index per pixel), "normal" (H, W, 3
    in world axes), "camera"}."""
    if not isinstance(meshes, (list, tuple)):
        meshes = [meshes]
    tris, norms, which = [], [], []
    for k, m in enumerate(meshes):
        V = np.asarray(m.vertices, float)[np.asarray(m.faces)]
        tris.append(V)
        norms.append(np.asarray(m.face_normals, float))
        which.append(np.full(len(V), k))
    T = np.concatenate(tris) if tris else np.zeros((0, 3, 3))
    N = np.concatenate(norms) if norms else np.zeros((0, 3))
    which = np.concatenate(which) if which else np.zeros(0, int)
    if camera.frame or camera.position is None:
        camera.fit(T.reshape(-1, 3))
    S = camera.project(T.reshape(-1, 3)).reshape(-1, 3, 3)
    depth = np.full((camera.height, camera.width), np.inf)
    owner = np.full((camera.height, camera.width), -1, np.int64)
    if len(S):
        _raster(np.ascontiguousarray(S), camera.width, camera.height, depth, owner)
    mask = owner >= 0
    normal = np.zeros((camera.height, camera.width, 3))
    normal[mask] = N[owner[mask]]
    wh = np.full(owner.shape, -1)
    wh[mask] = which[owner[mask]]
    return {"depth": depth, "mask": mask, "owner": owner, "which": wh, "normal": normal, "camera": camera}


def shade_image(r, colors=None, background=(246, 245, 242), light=(-0.45, -0.6, 0.66)):
    """A clay image (H, W, 3 uint8) of a render: lambert from a key light
    fixed to the camera, plus a soft fill, with the silhouette and depth
    steps drawn darker so a hole or a step reads. `colors`: an RGB per mesh."""
    cam = r["camera"]
    mask = r["mask"]
    n = r["normal"]
    # the light rides with the camera: left, above and behind it
    L = _unit(light[0] * cam.right + light[2] * cam.up + light[1] * cam.forward)
    facing = -(n @ cam.forward)
    nn = np.where((facing < 0)[..., None], -n, n)        # a back face seen through a cut still shades
    lam = np.clip(nn @ L, 0, 1)
    rim = np.clip(-(nn @ cam.forward), 0, 1)
    val = 0.30 + 0.55 * lam + 0.15 * rim
    img = np.empty(mask.shape + (3,), float)
    img[:] = np.asarray(background, float)
    base = np.full(mask.shape + (3,), 214.0)
    if colors is not None:
        for k, c in enumerate(colors):
            base[r["which"] == k] = np.asarray(c, float)
    img[mask] = (base * val[..., None])[mask]
    # depth steps: an edge where the depth jumps by more than a few pixels' worth
    d = np.where(mask, r["depth"], np.nan)
    with np.errstate(invalid="ignore"):
        gx = np.abs(np.diff(d, axis=1, prepend=d[:, :1]))
        gy = np.abs(np.diff(d, axis=0, prepend=d[:1]))
    scale = np.nanmedian(d[mask]) if mask.any() else 1.0
    px = cam.mm_per_pixel(scale if np.isfinite(scale) else 1.0)
    # a surface seen at a slant changes depth from pixel to pixel by px * tan
    # of its slant: only a jump well past that is a step
    cosv = np.clip(np.abs(n @ cam.forward), 0.05, 1.0)
    slope = np.sqrt(1.0 - cosv ** 2) / cosv
    step = np.maximum(np.nan_to_num(gx), np.nan_to_num(gy)) > px * (4.0 + 3.0 * slope)
    img[step & mask] *= 0.55
    img[outline(mask)] = (40, 40, 40)
    return np.clip(img, 0, 255).astype(np.uint8)


def outline(mask):
    """The pixels of a mask that touch what is outside it."""
    m = np.asarray(mask, bool)
    inner = m.copy()
    inner[1:] &= m[:-1]
    inner[:-1] &= m[1:]
    inner[:, 1:] &= m[:, :-1]
    inner[:, :-1] &= m[:, 1:]
    return m & ~inner
