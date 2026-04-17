"""Auto-extract MIDAS parameters from a dataset file.

Given a single dataset file (a raw frame, HDF5, or Zarr), infer as many
parameters as possible so the wizard can pre-fill prompts. Examples:

  /data/exp/sample_000042.ge3 →
    RawFolder = /data/exp
    FileStem  = sample
    Padding   = 6
    Ext       = .ge3
    StartNr   = <min number found in dir>
    EndNr     = <max number found in dir>

  /data/run/mydata.h5 →
    (above — treating h5 as the raw "file") plus
    OmegaStart / OmegaStep / NrPixelsY / NrPixelsZ / Wavelength
    if attrs or named datasets are present (best-effort, beamline-dependent).

Design:
  - No hard h5py / zarr import at module load; guarded imports inside functions.
    Keeps the core package pure stdlib per pyproject.toml.
  - Every extracted value is tagged with a confidence level and a source tag,
    so the wizard can color-code prompts (green = high confidence, yellow =
    sniffed, gray = fallback).
  - Returns a DiscoveryResult; callers decide what to do with it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path as FsPath
from typing import Any


# Regex for `<stem>_<zero-padded-number><ext>`. Greedy on the stem so that
# `sample_run2_000042.tif` → stem="sample_run2", num=000042, ext=".tif".
_FRAME_RE = re.compile(r"^(?P<stem>.+?)_(?P<num>\d+)(?P<ext>\.[^.]+)$")

# HDF5/Zarr metadata paths. Beamlines vary — these are the common ones.
# Each entry is probed in order; first hit wins.

_H5_DATA_PATHS = [
    "/exchange/data",                                    # APS convention (many beamlines)
    "/measurement/instrument/detector/data",             # NeXus / APS 1-ID
    "/measurement/instrument/detector_1/data",           # NeXus multi-detector
    "/entry/data/data",                                  # NeXus generic
    "/entry/instrument/detector/data",                   # NeXus NXdetector
    "/image",                                            # ESRF/DESY custom
]

_H5_OMEGA_PATHS = [
    "/measurement/instrument/omega",                     # APS 1-ID
    "/measurement/process/scan_parameters/OmegaStart",   # MIDAS pre-analysis
    "/analysis/process/analysis_parameters/OmegaStart",  # MIDAS post-analysis
    "/entry/sample/rotation_angle",                      # NeXus standard
    "/entry/instrument/goniometer/omega",                # NeXus goniometer
    "/exchange/theta",                                   # tomography
    "/metadata/omega",                                   # DESY PETRA III
    "/measurement/sample/omega",                         # APS 6-BM
]

_H5_WAVELENGTH_PATHS = [
    "/measurement/instrument/source/wavelength",         # NeXus NXsource
    "/measurement/instrument/monochromator/wavelength",  # NeXus NXmonochromator
    "/entry/instrument/beam/incident_wavelength",        # NeXus NXbeam
    "/entry/instrument/source/wavelength",
    "/metadata/wavelength",                              # DESY
    "/measurement/energy",                               # sometimes energy in keV
]

_H5_DETECTOR_GEOMETRY_PATHS = {
    "Lsd": [
        "/measurement/instrument/detector/distance",
        "/measurement/instrument/detector_1/distance",
        "/entry/instrument/detector/distance",
        "/measurement/detector/distance",
    ],
    "px": [
        "/measurement/instrument/detector/x_pixel_size",
        "/entry/instrument/detector/x_pixel_size",
        "/measurement/detector/pixel_size",
    ],
}

_H5_BEAM_CENTER_PATHS = [
    ("/measurement/instrument/detector/beam_center_y",
     "/measurement/instrument/detector/beam_center_z"),
    ("/entry/instrument/detector/beam_center_x",
     "/entry/instrument/detector/beam_center_y"),
    ("/measurement/detector/beam_center_y",
     "/measurement/detector/beam_center_z"),
]


@dataclass
class DiscoveryResult:
    """What we extracted from a dataset file."""

    extracted: dict[str, Any] = field(default_factory=dict)
    """param name → value"""

    confidence: dict[str, str] = field(default_factory=dict)
    """param name → 'high' | 'medium' | 'low'"""

    source: dict[str, str] = field(default_factory=dict)
    """param name → where we got it from ('filename', 'dir-scan', 'h5-attr', ...)"""

    warnings: list[str] = field(default_factory=list)
    """Things that went wrong or ambiguous."""


# ─── Filename parsing ────────────────────────────────────────────────────────


def parse_frame_filename(path: str | FsPath) -> dict[str, Any] | None:
    """Try to parse `stem_NNNNNN.ext` → {stem, num, padding, ext}.

    Returns None if the filename doesn't match the expected pattern. In that
    case we fall back to treating the file as-is (e.g. HDF5).
    """
    p = FsPath(path)
    m = _FRAME_RE.match(p.name)
    if not m:
        return None
    num_str = m.group("num")
    return {
        "stem": m.group("stem"),
        "num": int(num_str),
        "padding": len(num_str),
        "ext": m.group("ext"),
    }


def scan_directory_for_range(
    folder: FsPath, stem: str, padding: int, ext: str
) -> tuple[int, int, int] | None:
    """Scan `folder` for files matching `<stem>_<N:padding>.ext`.

    Returns (min_number, max_number, count) or None if nothing matches.
    """
    pattern = f"{stem}_*{ext}"
    numbers = []
    for fp in folder.glob(pattern):
        m = _FRAME_RE.match(fp.name)
        if m and m.group("stem") == stem and m.group("ext") == ext:
            numbers.append(int(m.group("num")))
    if not numbers:
        return None
    return (min(numbers), max(numbers), len(numbers))


# ─── HDF5 / Zarr probing (optional, guarded imports) ─────────────────────────


def _scalar(ds) -> float | None:
    """Coerce an HDF5/Zarr dataset to a scalar float, or None on failure."""
    try:
        val = ds[()]
    except Exception:
        return None
    try:
        if hasattr(val, "size"):
            if val.size == 1:
                return float(val.item())
            if val.size == 0:
                return None
        return float(val)
    except (TypeError, ValueError):
        return None


def _probe_hdf5(path: FsPath, result: DiscoveryResult) -> None:
    """Best-effort HDF5 metadata extraction. No-op if h5py unavailable or
    file unreadable.

    Probes common NeXus / APS / DESY / ESRF paths for: detector shape, pixel
    size, sample-detector distance, beam center, wavelength, ω rotation.
    """
    try:
        import h5py  # type: ignore
    except ImportError:
        result.warnings.append("h5py not installed — skipping HDF5 introspection.")
        return

    def _record(key: str, value, source_tag: str, confidence: str = "high"):
        result.extracted.setdefault(key, value)
        result.confidence.setdefault(key, confidence)
        result.source.setdefault(key, source_tag)

    try:
        with h5py.File(path, "r") as f:
            # Detector shape from the data array
            for data_path in _H5_DATA_PATHS:
                if data_path in f:
                    ds = f[data_path]
                    if ds.ndim == 3:
                        _, nz, ny = ds.shape
                        _record("NrPixelsY", int(ny), f"h5:{data_path}.shape")
                        _record("NrPixelsZ", int(nz), f"h5:{data_path}.shape")
                    elif ds.ndim == 2:
                        nz, ny = ds.shape
                        _record("NrPixelsY", int(ny), f"h5:{data_path}.shape")
                        _record("NrPixelsZ", int(nz), f"h5:{data_path}.shape")
                    break

            # Wavelength (Å). Some beamlines store energy (keV); we guess by
            # magnitude: < 10 → Å; > 100 → keV; warn on ambiguous middle.
            for attr_path in _H5_WAVELENGTH_PATHS:
                if attr_path not in f:
                    continue
                wl = _scalar(f[attr_path])
                if wl is None:
                    continue
                if 0.05 <= wl <= 5.0:
                    _record("Wavelength", wl, f"h5:{attr_path}")
                elif wl > 1000:  # eV
                    wl_A = 12398.4 / wl
                    _record("Wavelength", wl_A, f"h5:{attr_path} (converted from eV)")
                elif 5 < wl < 500:  # likely keV
                    wl_A = 12.3984 / wl
                    _record("Wavelength", wl_A, f"h5:{attr_path} (converted from keV)")
                else:
                    result.warnings.append(
                        f"Ambiguous wavelength {wl} at {attr_path}; expected Å (0.1–3) "
                        f"or keV (10–200)."
                    )
                break

            # Sample-detector distance. NeXus often stores in metres.
            for key, paths in _H5_DETECTOR_GEOMETRY_PATHS.items():
                for attr_path in paths:
                    if attr_path not in f:
                        continue
                    val = _scalar(f[attr_path])
                    if val is None:
                        continue
                    # Heuristic: MIDAS expects µm. If value < 10 it's probably metres.
                    if key == "Lsd" and val < 10:
                        val = val * 1e6
                        source_note = f"h5:{attr_path} (× 1e6, m→µm)"
                    elif key == "px" and val < 0.001:
                        val = val * 1e6
                        source_note = f"h5:{attr_path} (× 1e6, m→µm)"
                    else:
                        source_note = f"h5:{attr_path}"
                    _record(key, val, source_note)
                    break

            # Beam center as (y, z) pair
            for y_path, z_path in _H5_BEAM_CENTER_PATHS:
                if y_path in f and z_path in f:
                    y = _scalar(f[y_path])
                    z = _scalar(f[z_path])
                    if y is not None and z is not None:
                        _record("BC", [y, z], f"h5:{y_path}+{z_path}")
                        break

            # ω rotation array — infer OmegaStart / OmegaStep / OmegaEnd
            for attr_path in _H5_OMEGA_PATHS:
                if attr_path not in f:
                    continue
                try:
                    omega = f[attr_path][()]
                except Exception:
                    continue
                if hasattr(omega, "shape") and omega.ndim == 1 and len(omega) >= 2:
                    start = float(omega[0])
                    step = float(omega[1] - omega[0])
                    end = float(omega[-1])
                    _record("OmegaStart", start, f"h5:{attr_path}[0]")
                    _record("OmegaStep", step, f"h5:{attr_path}[diff]")
                    _record("OmegaEnd", end, f"h5:{attr_path}[-1]")
                    break
                # Scalar OmegaStart (from MIDAS-style metadata group)
                val = _scalar(f[attr_path])
                if val is not None:
                    _record("OmegaStart", val, f"h5:{attr_path}")
                    break

    except OSError as e:
        result.warnings.append(f"Could not open {path} as HDF5: {e}")


def _probe_zarr(path: FsPath, result: DiscoveryResult) -> None:
    """Best-effort Zarr metadata extraction (including MIDAS `.zip` analysis files)."""
    try:
        import zarr  # type: ignore
    except ImportError:
        result.warnings.append("zarr not installed — skipping Zarr introspection.")
        return

    try:
        z = zarr.open(str(path), mode="r")
    except Exception as e:
        result.warnings.append(f"Could not open {path} as Zarr: {e}")
        return

    # MIDAS analysis.zip puts params at analysis/process/analysis_parameters/
    midas_param_group = None
    try:
        midas_param_group = z["analysis/process/analysis_parameters"]
    except Exception:
        pass

    if midas_param_group is not None:
        # Keys commonly found in MIDAS analysis Zarrs
        key_map = {
            "Lsd": "Lsd",
            "YCen": None,  # handled specially below into BC
            "ZCen": None,
            "Wavelength": "Wavelength",
            "PixelSize": "px",
            "tx": "tx", "ty": "ty", "tz": "tz",
            "RhoD": "RhoD",
            "LatticeParameter": "LatticeConstant",
            "SpaceGroup": "SpaceGroup",
        }
        for zarr_key, midas_key in key_map.items():
            if midas_key is None:
                continue
            if zarr_key in midas_param_group:
                try:
                    val = midas_param_group[zarr_key][...]
                    # Reduce to scalar or list
                    if hasattr(val, "size") and val.size == 1:
                        val = val.item()
                    elif hasattr(val, "tolist"):
                        val = val.tolist()
                    result.extracted.setdefault(midas_key, val)
                    result.confidence.setdefault(midas_key, "high")
                    result.source.setdefault(midas_key, f"zarr:analysis/.../{zarr_key}")
                except Exception:
                    continue
        # Synthesize BC from YCen + ZCen
        try:
            if "YCen" in midas_param_group and "ZCen" in midas_param_group:
                y = float(midas_param_group["YCen"][0])
                zc = float(midas_param_group["ZCen"][0])
                result.extracted.setdefault("BC", [y, zc])
                result.confidence.setdefault("BC", "high")
                result.source.setdefault("BC", "zarr:analysis/.../YCen+ZCen")
        except Exception:
            pass

    # Scan parameters
    try:
        scan = z["measurement/process/scan_parameters"]
        for zarr_key, midas_key in [("OmegaStart", "OmegaStart"),
                                     ("step", "OmegaStep"),
                                     ("OmegaEnd", "OmegaEnd")]:
            if zarr_key in scan:
                try:
                    val = scan[zarr_key][...]
                    if hasattr(val, "size") and val.size == 1:
                        val = val.item()
                    result.extracted.setdefault(midas_key, val)
                    result.confidence.setdefault(midas_key, "high")
                    result.source.setdefault(midas_key, f"zarr:measurement/.../{zarr_key}")
                except Exception:
                    continue
    except Exception:
        pass


# ─── Top-level entry point ───────────────────────────────────────────────────


def discover_from_file(dataset_file: str | FsPath) -> DiscoveryResult:
    """Infer MIDAS parameters from a single dataset file.

    Extraction order:
      1. Filename pattern → RawFolder, FileStem, Padding, Ext
      2. Directory scan   → StartNr, EndNr (and count)
      3. HDF5 / Zarr probe → geometry, omega, wavelength (if applicable)
    """
    result = DiscoveryResult()
    path = FsPath(dataset_file).expanduser().resolve()

    if not path.exists():
        result.warnings.append(f"Path does not exist: {path}")
        return result

    # 1. Filename parse
    if path.is_file():
        parsed = parse_frame_filename(path)
        result.extracted["RawFolder"] = str(path.parent)
        result.confidence["RawFolder"] = "high"
        result.source["RawFolder"] = "filename"

        if parsed:
            result.extracted["FileStem"] = parsed["stem"]
            result.extracted["Padding"] = parsed["padding"]
            result.extracted["Ext"] = parsed["ext"]
            result.extracted["StartNr"] = parsed["num"]
            for k in ("FileStem", "Padding", "Ext"):
                result.confidence[k] = "high"
                result.source[k] = "filename"
            result.confidence["StartNr"] = "medium"  # confirmed after dir scan
            result.source["StartNr"] = "filename"

            # 2. Directory scan
            scan = scan_directory_for_range(
                path.parent, parsed["stem"], parsed["padding"], parsed["ext"]
            )
            if scan is not None:
                lo, hi, cnt = scan
                result.extracted["StartNr"] = lo
                result.extracted["EndNr"] = hi
                result.confidence["StartNr"] = "high"
                result.confidence["EndNr"] = "high"
                result.source["StartNr"] = f"dir-scan ({cnt} files)"
                result.source["EndNr"] = f"dir-scan ({cnt} files)"

    # 3. HDF5 / Zarr probe (by extension)
    suffix = path.suffix.lower() if path.is_file() else ""
    if suffix in (".h5", ".hdf5"):
        _probe_hdf5(path, result)
    elif suffix in (".zarr", ".zip"):  # MIDAS writes .zip but it's a Zarr
        _probe_zarr(path, result)
    elif path.is_dir() and any(
        (path / suffix_name).exists()
        for suffix_name in (".zarray", ".zgroup", "zarr.json")
    ):
        _probe_zarr(path, result)

    return result


def discover_from_calibration_file(param_file: str | FsPath) -> DiscoveryResult:
    """Seed a DiscoveryResult from an existing MIDAS text param file.

    Useful when the user has a `refined_MIDAS_params.txt` from
    AutoCalibrateZarr and wants to pre-populate the wizard.
    """
    from .parser import parse_typed

    parsed = parse_typed(param_file)
    result = DiscoveryResult()
    for key, value in parsed.values.items():
        result.extracted[key] = value
        result.confidence[key] = "high"
        result.source[key] = f"param-file:{FsPath(param_file).name}"
    for key, line_no in parsed.unknown_keys:
        result.warnings.append(f"Unknown key {key!r} at line {line_no}")
    return result


def merge(*results: DiscoveryResult) -> DiscoveryResult:
    """Merge multiple DiscoveryResults. Earlier wins on conflict (use order
    to express priority: wizard_result = merge(user_overrides, from_calibration,
    from_dataset_probe))."""
    out = DiscoveryResult()
    for r in results:
        for key, val in r.extracted.items():
            if key not in out.extracted:
                out.extracted[key] = val
                out.confidence[key] = r.confidence.get(key, "medium")
                out.source[key] = r.source.get(key, "?")
        out.warnings.extend(r.warnings)
    return out
