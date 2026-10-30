"""
A real slicer in the FDM loop: print time, filament and support material
measured by slicing the delivered part, not modelled.

For an FDM user the cost of a part is mostly hours on the printer, and grams
are only a proxy for that: two designs of the same mass can print in very
different times once one of them needs a forest of support. So every FDM
outcome is sliced headless (PrusaSlicer's command line, or a fork that keeps
it: SuperSlicer) standing the way it was designed to print, and the G-code
is read back for

- printTime (s), the slicer's own estimate under its machine limits;
- filament (mm, cm^3, g), and how much of it is support;
- the extruded volume of each feature type (perimeters, infill, support...).

`goal.minimize: "printTime"` then picks the fastest passing outcome instead
of the lightest, and `slicer.grade` (infill.py) turns the verified stress
into per-region infill and wall counts the slicer prints as modifiers.

The part is verified solid, so the default slice is solid too (100% infill):
what was analysed is what gets printed. Sparse infill is only safe where the
stress allows it, which is what grading works out.

The built-in profile is a generic 0.4 mm nozzle, 0.2 mm layer bed slinger at
moderate speeds (Prusa MK3S class); a CoreXY printer runs two to three times
faster. Pass your own PrusaSlicer config (`slicer.profile`, an .ini exported
from File > Export > Export Config) for absolute times; for comparing
outcomes against each other the generic one is enough.

Orca Slicer and Bambu Studio open the 3MF this writes (PrusaSlicer's
modifier metadata), but their command lines differ and are not driven here.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass, field

import numpy as np

__all__ = ["SlicerOptions", "find_slicer", "slice_mesh", "read_gcode", "DEFAULT_PROFILE"]

# PrusaSlicer config keys. Bed 256 x 256 to match threemf's plate centre.
DEFAULT_PROFILE = {
    "printer_technology": "FFF",
    "gcode_flavor": "marlin2",
    "bed_shape": "0x0,256x0,256x256,0x256",
    "max_print_height": "256",
    "nozzle_diameter": "0.4",
    "filament_diameter": "1.75",
    "layer_height": "0.2",
    "first_layer_height": "0.2",
    "extrusion_width": "0.45",
    "perimeter_extrusion_width": "0.45",
    "external_perimeter_extrusion_width": "0.45",
    "infill_extrusion_width": "0.45",
    "solid_infill_extrusion_width": "0.45",
    "top_infill_extrusion_width": "0.4",
    "support_material_extrusion_width": "0.35",
    "perimeters": "3",
    "top_solid_layers": "5",
    "bottom_solid_layers": "4",
    "top_solid_min_thickness": "0.7",
    "bottom_solid_min_thickness": "0.5",
    "fill_density": "100%",
    "fill_pattern": "gyroid",
    "infill_anchor": "2.5",
    "perimeter_speed": "45",
    "small_perimeter_speed": "25",
    "external_perimeter_speed": "25",
    "infill_speed": "80",
    "solid_infill_speed": "80",
    "top_solid_infill_speed": "40",
    "support_material_speed": "50",
    "support_material_interface_speed": "80%",
    "bridge_speed": "30",
    "gap_fill_speed": "40",
    "travel_speed": "180",
    "first_layer_speed": "20",
    "max_volumetric_speed": "11.5",
    "perimeter_acceleration": "800",
    "infill_acceleration": "1000",
    "bridge_acceleration": "1000",
    "first_layer_acceleration": "800",
    "default_acceleration": "1000",
    "machine_limits_usage": "time_estimate_only",
    "machine_max_acceleration_extruding": "1250",
    "machine_max_acceleration_travel": "1250",
    "machine_max_acceleration_x": "1000",
    "machine_max_acceleration_y": "1000",
    "machine_max_feedrate_x": "200",
    "machine_max_feedrate_y": "200",
    "support_material": "1",
    "support_material_auto": "1",
    # overhangs flatter than 45 degrees from horizontal get support: the
    # angle loadpath's overhang control designs to, less printcheck's
    # 5 degree tolerance
    "support_material_threshold": "40",
    "support_material_contact_distance": "0.2",
    "support_material_spacing": "2",
    "support_material_style": "grid",
    "skirts": "1",
    "brim_width": "0",
    "use_relative_e_distances": "1",
    "layer_gcode": "G92 E0",
    "gcode_comments": "0",
}

# PrusaSlicer's ";TYPE:" feature names, by what they are
_SUPPORT_TYPES = ("Support material", "Support material interface")
_PART_TYPES = ("Perimeter", "External perimeter", "Overhang perimeter", "Internal infill", "Solid infill",
               "Top solid infill", "Bridge infill", "Gap fill")

_NAMES = ("prusa-slicer", "PrusaSlicer", "prusaslicer", "superslicer", "SuperSlicer")


def find_slicer(path: str | None = None) -> str | None:
    """The slicer executable: `path`, $LOADPATH_SLICER, or PrusaSlicer (or
    SuperSlicer) on PATH or in its usual macOS place. None if there is none."""
    for cand in (path, os.environ.get("LOADPATH_SLICER")):
        if cand:
            return cand if (os.path.isfile(cand) or shutil.which(cand)) else None
    for n in _NAMES:
        p = shutil.which(n)
        if p:
            return p
    for p in ("/Applications/PrusaSlicer.app/Contents/MacOS/PrusaSlicer",
              "/Applications/Original Prusa Drivers/PrusaSlicer.app/Contents/MacOS/PrusaSlicer"):
        if os.path.isfile(p):
            return p
    return None


@dataclass
class SlicerOptions:
    enabled: object = "auto"          # "auto": when FDM and a slicer is installed; True: warn if none
    executable: str | None = None
    profile: str | None = None        # PrusaSlicer .ini, applied before `settings`
    settings: dict = field(default_factory=dict)
    supports: object = "auto"         # "auto"/True: slicer's own auto supports; False: none
    infill: float = 100.0             # percent infill of the ungraded slice (100 = as verified)
    timeout: float = 600.0
    grade: dict | None = None         # infill.GradeOptions spec, or None

    @classmethod
    def from_spec(cls, spec, base_dir: str | None = None) -> "SlicerOptions":
        if spec is None:
            return cls()
        if isinstance(spec, bool):
            return cls(enabled=spec)
        if not isinstance(spec, dict):
            raise ValueError("slicer is true/false or an object")
        enabled = spec.get("enabled", "auto")
        if enabled not in (True, False, "auto"):
            raise ValueError("slicer.enabled is true, false or \"auto\"")
        prof = spec.get("profile")
        if prof is not None and base_dir and not os.path.isabs(prof):
            prof = os.path.join(base_dir, prof)
        infill = float(spec.get("infill", 100.0))
        if not 0.0 <= infill <= 100.0:
            raise ValueError("slicer.infill is a percentage, 0..100")
        sup = spec.get("supports", "auto")
        if sup not in (True, False, "auto"):
            raise ValueError("slicer.supports is true, false or \"auto\"")
        settings = spec.get("settings") or {}
        if not isinstance(settings, dict):
            raise ValueError("slicer.settings is an object of PrusaSlicer config keys")
        grade = spec.get("grade")
        if grade is True:
            grade = {}
        elif grade is False:
            grade = None
        elif grade is not None and not isinstance(grade, dict):
            raise ValueError("slicer.grade is true or an object")
        return cls(enabled=enabled, executable=spec.get("executable"), profile=prof,
                   settings={str(k): _fmt_setting(v) for k, v in settings.items()}, supports=sup,
                   infill=infill, timeout=float(spec.get("timeout", 600.0)), grade=grade)

    def config(self) -> dict:
        """The config keys the slice is run with (the profile file aside)."""
        cfg = {} if self.profile else dict(DEFAULT_PROFILE)
        cfg["fill_density"] = f"{self.infill:g}%"
        if self.infill >= 100.0:
            cfg["fill_pattern"] = "rectilinear"   # PrusaSlicer refuses gyroid at 100%
        cfg["support_material"] = "0" if self.supports is False else "1"
        cfg["use_relative_e_distances"] = "1"     # read_gcode sums E per feature
        cfg.setdefault("layer_gcode", "G92 E0")    # which PrusaSlicer requires with it
        cfg.update(self.settings)
        return cfg


def _fmt_setting(v) -> str:
    if isinstance(v, bool):
        return "1" if v else "0"
    if isinstance(v, (list, tuple)):
        return ",".join(str(x) for x in v)
    return str(v)


def _version(exe: str) -> str:
    try:
        out = subprocess.run([exe, "--help"], capture_output=True, text=True, timeout=30).stdout
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    first = out.strip().splitlines()[0] if out.strip() else ""
    return first.split(" based on")[0].strip() or "unknown"


_TIME_RE = re.compile(r"(\d+)\s*([dhms])")


def _seconds(text: str) -> float:
    unit = {"d": 86400, "h": 3600, "m": 60, "s": 1}
    return float(sum(int(n) * unit[u] for n, u in _TIME_RE.findall(text)))


def read_gcode(path: str, density: float | None = None) -> dict:
    """Print time, filament and per-feature extruded volume from a
    PrusaSlicer G-code (relative E). `density` g/cm^3 gives grams."""
    feat: dict[str, float] = {}
    cur = "Other"
    meta: dict[str, str] = {}
    dia = 1.75
    relative = None
    with open(path, "r", errors="replace") as fh:
        for line in fh:
            if line.startswith(";TYPE:"):
                cur = line[6:].strip()
                continue
            if line.startswith("; ") and " = " in line:
                k, v = line[2:].split(" = ", 1)
                meta[k.strip()] = v.strip()
                continue
            if line.startswith("M83"):
                relative = True
            elif line.startswith("M82"):
                relative = False
            if line.startswith("G1 ") or line.startswith("G2 ") or line.startswith("G3 "):
                i = line.find(" E")
                if i >= 0:
                    j = i + 2
                    k = j
                    while k < len(line) and (line[k].isdigit() or line[k] in ".-"):
                        k += 1
                    try:
                        e = float(line[j:k])
                    except ValueError:
                        continue
                    if e > 0:
                        feat[cur] = feat.get(cur, 0.0) + e
    if "filament_diameter" in meta:
        dia = float(meta["filament_diameter"].split(",")[0])
    if relative is False:
        raise ValueError("read_gcode needs relative extrusion (use_relative_e_distances = 1)")
    area = np.pi * (0.5 * dia) ** 2 / 1000.0          # cm^3 per mm of filament
    vol = {k: v * area for k, v in feat.items()}
    support = sum(vol.get(k, 0.0) for k in _SUPPORT_TYPES)
    total_mm = float(meta.get("filament used [mm]", sum(feat.values())))
    total_cm3 = float(meta.get("filament used [cm3]", total_mm * area))
    t = meta.get("estimated printing time (normal mode)")
    out = {
        "printTime": _seconds(t) if t else None,
        "printTimeText": t,
        "filamentLength": total_mm,
        "filamentVolume": total_cm3,
        "supportVolume": float(support),
        "supportShare": float(support / total_cm3) if total_cm3 > 0 else 0.0,
        "features": {k: round(v, 4) for k, v in sorted(vol.items(), key=lambda kv: -kv[1])},
        "layerHeight": float(meta["layer_height"]) if "layer_height" in meta else None,
    }
    if density is not None:
        out["filamentMass"] = total_cm3 * density
        out["supportMass"] = float(support) * density
    return out


def slice_mesh(mesh, build=None, material=None, opts: SlicerOptions | None = None, *, name: str = "part",
               object_settings: dict | None = None, modifiers: list | None = None,
               keep: str | None = None) -> dict:
    """Slice `mesh` (trimesh, mm) standing along `build` (as threemf takes
    it) with `opts`; `object_settings` and `modifiers` (dicts with "mesh",
    "settings", "name") are the per-region config graded infill prints
    with. Returns read_gcode's report plus the slicer's name and version and
    the slicing time. `keep` copies the G-code there."""
    from .threemf import write_3mf

    opts = opts or SlicerOptions()
    exe = find_slicer(opts.executable)
    if exe is None:
        raise FileNotFoundError("no slicer found: install PrusaSlicer (or set LOADPATH_SLICER to its executable)")
    t0 = time.time()
    with tempfile.TemporaryDirectory(prefix="loadpath-slice-") as tmp:
        tmf = os.path.join(tmp, "part.3mf")
        write_3mf(tmf, mesh, build, name=name, material=material, object_settings=object_settings,
                  modifiers=modifiers)
        ini = os.path.join(tmp, "loadpath.ini")
        with open(ini, "w") as fh:
            fh.write("".join(f"{k} = {v}\n" for k, v in opts.config().items()))
        gcode = os.path.join(tmp, "part.gcode")
        cmd = [exe]
        if opts.profile:
            cmd += ["--load", opts.profile]
        cmd += ["--load", ini, "--export-gcode", "--output", gcode, tmf]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=opts.timeout)
        if proc.returncode != 0 or not os.path.isfile(gcode):
            msg = (proc.stderr or proc.stdout or "").strip().splitlines()[-3:]
            raise RuntimeError(f"slicer failed ({proc.returncode}): {' / '.join(msg)}")
        density = getattr(material, "density", None)
        rep = read_gcode(gcode, density)
        if keep:
            shutil.copyfile(gcode, keep)
            rep["gcode"] = keep
    rep["slicer"] = _version(exe)
    rep["profile"] = opts.profile or "loadpath generic (0.4 mm nozzle, 0.2 mm layers, MK3S-class speeds)"
    rep["infill"] = opts.infill
    rep["seconds"] = time.time() - t0
    return rep
