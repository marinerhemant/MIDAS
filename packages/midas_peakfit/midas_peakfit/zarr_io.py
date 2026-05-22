"""Zarr 2.18 reader: parses metadata + delivers per-frame pixel data.

Replaces the C tool's libzip + manual Blosc1 decompression path with
``zarr.ZipStore`` + ``numcodecs`` (which call libblosc1 underneath, so output
bytes are identical).
"""
from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Optional, Tuple

import numpy as np
import zarr

from midas_peakfit.params import (
    DEFAULT_NR_PIXELS,
    DEFAULT_PIXEL_SIZE,
    ZarrParams,
)

# ─── Zarr dtype string → canonical pixel-type label ─────────────────────────
_DTYPE_NAME_MAP = {
    "uint8": "uint8",
    "u1": "uint8",
    "<u1": "uint8",
    "uint16": "uint16",
    "u2": "uint16",
    "<u2": "uint16",
    ">u2": "uint16",
    "int16": "int16",
    "i2": "int16",
    "<i2": "int16",
    "uint32": "uint32",
    "u4": "uint32",
    "<u4": "uint32",
    "int32": "int32",
    "i4": "int32",
    "<i4": "int32",
    "float32": "float32",
    "f4": "float32",
    "<f4": "float32",
    "float64": "float64",
    "f8": "float64",
    "<f8": "float64",
}


def canonical_pixel_type(zarr_dtype: str) -> str:
    s = str(zarr_dtype)
    return _DTYPE_NAME_MAP.get(s, _DTYPE_NAME_MAP.get(s.lstrip("<>"), s))


def _bytes_per_px(pixel_type: str) -> int:
    return {
        "uint8": 1,
        "int8": 1,
        "uint16": 2,
        "int16": 2,
        "uint32": 4,
        "int32": 4,
        "float32": 4,
        "float64": 8,
    }[pixel_type]


# ─── ZipStore context ────────────────────────────────────────────────────────
@contextmanager
def open_zarr(path: str | Path) -> Iterator[zarr.hierarchy.Group]:
    """Open the Zarr ZipStore for read; yields the root group."""
    store = zarr.ZipStore(str(path), mode="r")
    try:
        yield zarr.open_group(store=store, mode="r")
    finally:
        store.close()


# ─── Helpers for reading single-element arrays ───────────────────────────────
def _scalar(group: zarr.hierarchy.Group, key: str, default=None, *, cast=None):
    """Read a 1-element array; return its scalar value (or default)."""
    if key not in group:
        return default
    arr = group[key][...]
    if arr.size == 0:
        return default
    val = arr.flat[0]
    if cast is not None:
        val = cast(val)
    return val


def _array(group: zarr.hierarchy.Group, key: str):
    """Read full array; return None if absent."""
    if key not in group:
        return None
    return group[key][...]


def _string(group: zarr.hierarchy.Group, key: str) -> Optional[str]:
    """Read a string-valued zarr key. Strings are stored as bytes-typed arrays."""
    if key not in group:
        return None
    raw = group[key][...]
    if raw.size == 0:
        return None
    val = raw.flat[0]
    if isinstance(val, bytes):
        return val.decode("utf-8", errors="replace").rstrip("\x00").strip()
    if isinstance(val, np.bytes_):
        return bytes(val).decode("utf-8", errors="replace").rstrip("\x00").strip()
    return str(val).strip()


# ─── Parameter parser ────────────────────────────────────────────────────────
def parse_zarr_params(path: str | Path) -> ZarrParams:
    """Parse all ImageMetadata + AnalysisParams from a Zarr archive.

    Returns a fully-populated ``ZarrParams`` with ``finalize()`` already called.
    """
    p = ZarrParams()

    with open_zarr(path) as root:
        # ── Frame data shape + dtype ────────────────────────────────
        # IMPORTANT: zarr shape is [nFrames, NrPixelsZ, NrPixelsY] —
        # Z is the slow axis (rows), Y is the fast axis (cols). This matches
        # PeaksFittingOMPZarrRefactor.c lines 1967-1970:
        #   sscanf(..., "%d%d%d", &nFrames, &NrPixelsZ, &NrPixelsY)
        if "exchange/data" in root:
            data = root["exchange/data"]
            shape = data.shape
            if len(shape) == 3:
                p.nFrames = int(shape[0])
                p.NrPixelsZ = int(shape[1])
                p.NrPixelsY = int(shape[2])
            elif len(shape) == 2:
                p.nFrames = int(shape[0])
                p.NrPixelsZ = p.NrPixelsY = int(shape[1])
            p.pixelType = canonical_pixel_type(data.dtype)
            p.bytesPerPx = _bytes_per_px(p.pixelType)

        if "exchange/dark" in root:
            p.nDarks = int(root["exchange/dark"].shape[0])
        if "exchange/flood" in root:
            p.nFloods = int(root["exchange/flood"].shape[0])
        if "exchange/mask" in root:
            p.nMasks = int(root["exchange/mask"].shape[0])

        # ── Pixel-type override from scan_parameters/datatype ───────
        sp = root.get("measurement/process/scan_parameters", default=None)
        if sp is not None:
            override = _string(sp, "datatype")
            if override:
                p.pixelType = canonical_pixel_type(override)
                p.bytesPerPx = _bytes_per_px(p.pixelType)

            v = _scalar(sp, "start", cast=float)
            if v is not None:
                p.omegaStart = float(v)
            v = _scalar(sp, "step", cast=float)
            if v is not None:
                p.omegaStep = float(v)
            v = _scalar(sp, "doPeakFit", cast=int)
            if v is not None:
                p.doPeakFit = int(v)

            oc = _array(sp, "omegaCenter")
            if oc is not None and oc.size > 0:
                p.omegaCenter = np.asarray(oc, dtype=np.float64).reshape(-1)

        # ── Analysis parameters ─────────────────────────────────────
        ap = root.get("analysis/process/analysis_parameters", default=None)
        if ap is not None:
            # Scalars
            mapping = {
                "SkipFrame": ("skipFrame", int),
                "LocalMaximaOnly": ("localMaximaOnly", int),
                "MaxNPeaks": ("maxNPeaks", int),
                "MinNrPx": ("minNrPx", int),
                "MaxNrPx": ("maxNrPx", int),
                "DoFullImage": ("DoFullImage", int),
                "LayerNr": ("LayerNr", int),
                "NPanelsY": ("NPanelsY", int),
                "NPanelsZ": ("NPanelsZ", int),
                "PanelSizeY": ("PanelSizeY", int),
                "PanelSizeZ": ("PanelSizeZ", int),
                "YCen": ("Ycen", float),
                "ZCen": ("Zcen", float),
                "PixelSize": ("px", float),
                "Lsd": ("Lsd", float),
                "Width": ("Width", float),
                "Wavelength": ("Wavelength", float),
                "ReferenceRingCurrent": ("bc", float),
                "UpperBoundThreshold": ("IntSat", float),
                "zDiffThresh": ("zDiffThresh", float),
                "tx": ("tx", float),
                "ty": ("ty", float),
                "tz": ("tz", float),
                "p0": ("p0", float),
                "p1": ("p1", float),
                "p2": ("p2", float),
                "p3": ("p3", float),
                "p4": ("p4", float),
                "p5": ("p5", float),
                "p6": ("p6", float),
                "p7": ("p7", float),
                "p8": ("p8", float),
                "p9": ("p9", float),
                "p10": ("p10", float),
                "p11": ("p11", float),
                "p12": ("p12", float),
                "p13": ("p13", float),
                "p14": ("p14", float),
            }
            for zkey, (attr, cast) in mapping.items():
                v = _scalar(ap, zkey, cast=cast)
                if v is not None:
                    setattr(p, attr, cast(v))

            # Canonical distortion in the v2 harmonic basis. Prefer the v2
            # names written by calibrate-v2 (iso_R2/4/6, a1..a6, phi1..phi6);
            # fall back to the legacy p0..p14 for old archives.
            from midas_distortion import P_COEF_NAMES, v2_coeffs_from_named
            named = {nm: _scalar(ap, nm, cast=float) for nm in P_COEF_NAMES}
            named.update({f"p{i}": getattr(p, f"p{i}") for i in range(15)
                          if _scalar(ap, f"p{i}", cast=float) is not None})
            p.dist_coeffs_v2 = v2_coeffs_from_named(named)

            # RhoD or MaxRingRad (either form accepted)
            v = _scalar(ap, "RhoD", cast=float)
            if v is None:
                v = _scalar(ap, "MaxRingRad", cast=float)
            if v is not None:
                p.RhoD = float(v)

            # BadPxIntensity → presence flips makeMap=1
            if "BadPxIntensity" in ap:
                p.BadPxIntensity = float(_scalar(ap, "BadPxIntensity", default=0.0))
                p.makeMap = 1

            # Strings
            rf = _string(ap, "ResultFolder")
            if rf:
                p.ResultFolder = rf
            psf = _string(ap, "PanelShiftsFile")
            if psf:
                p.PanelShiftsFile = psf
            rcm = _string(ap, "ResidualCorrectionMap")
            if rcm:
                p.ResidualCorrectionMap = rcm

            # Arrays: ImTransOpt
            a = _array(ap, "ImTransOpt")
            if a is not None and a.size > 0:
                # Sentinel from ffGenerateZipRefactor: [-1] means "no transforms"
                arr = np.asarray(a, dtype=np.int32).reshape(-1)
                if arr.size == 1 and arr[0] == -1:
                    p.TransOpt = []
                else:
                    p.TransOpt = arr.tolist()
                p.nImTransOpt = len(p.TransOpt)

            # RingThresh: shape (nRings, 2) → (ringNr, threshold) pairs
            rt = _array(ap, "RingThresh")
            if rt is not None and rt.ndim == 2 and rt.shape[1] >= 2:
                # Skip sentinel default of [[0, 0]]
                rt = np.asarray(rt, dtype=np.float64)
                non_zero = ~((rt[:, 0] == 0) & (rt[:, 1] == 0))
                if non_zero.any():
                    rt_filt = rt[non_zero]
                    p.RingNrs = rt_filt[:, 0].astype(int).tolist()
                    p.Thresholds = rt_filt[:, 1].astype(float).tolist()
                    p.nRingsThresh = len(p.RingNrs)

            # Panel gaps
            pgy = _array(ap, "PanelGapsY")
            if pgy is not None and pgy.size > 0:
                p.PanelGapsY = np.asarray(pgy, dtype=np.int32).reshape(-1).tolist()
            pgz = _array(ap, "PanelGapsZ")
            if pgz is not None and pgz.size > 0:
                p.PanelGapsZ = np.asarray(pgz, dtype=np.int32).reshape(-1).tolist()

    p.finalize()
    return p


# ─── Calibration loader ──────────────────────────────────────────────────────
def load_corrections(path: str | Path, p: ZarrParams) -> None:
    """Read raw dark, flood, mask arrays into ``p`` (averaged where applicable).

    Stored shape is (NrPixelsZ, NrPixelsY), matching the on-disk zarr layout.
    These are not yet square-padded or transformed — that happens in
    ``preprocess.prepare_*`` once at startup.

    Mutates ``p`` in place.
    """
    Z, Y = p.NrPixelsZ, p.NrPixelsY

    with open_zarr(path) as root:
        if p.nDarks > 0 and "exchange/dark" in root:
            dark_arr = np.asarray(root["exchange/dark"][:], dtype=np.float64)
            if p.skipFrame > 0 and dark_arr.shape[0] > p.skipFrame:
                dark_arr = dark_arr[p.skipFrame:]
            p.dark = dark_arr.mean(axis=0).astype(np.float64)
        else:
            p.dark = np.zeros((Z, Y), dtype=np.float64)

        if p.nFloods > 0 and "exchange/flood" in root:
            flood_arr = np.asarray(root["exchange/flood"][0], dtype=np.float64)
            flood_arr = np.where(flood_arr == 0, 1.0, flood_arr)
            p.flood = flood_arr
        else:
            p.flood = np.ones((Z, Y), dtype=np.float64)

        if p.nMasks > 0 and "exchange/mask" in root:
            p.mask = np.asarray(root["exchange/mask"][0], dtype=np.float64)
            print(f"Mask is found. Number of mask pixels: {int((p.mask > 0).sum())}")
        else:
            p.mask = np.zeros((Z, Y), dtype=np.float64)

    if p.ResidualCorrectionMap:
        try:
            map_data = np.fromfile(p.ResidualCorrectionMap, dtype=np.float64)
            if map_data.size == Y * Z:
                # On disk the map is (NrPixelsZ, NrPixelsY) z-major — element
                # (z, y) holds ΔR(Y=y, Z=z), the convention written by
                # midas_calibrate_v2.compat.to_integrate and read by transforms.
                # geometry.compute_rt_eta adds it as Rt[A=Y, B=Z] += map[A, B],
                # so we need map indexed [Y, Z]: reshape to disk layout then
                # transpose. (A plain reshape(Y, Z) silently transposes the
                # correction — invisible on square detectors, wrong otherwise.)
                p.residualMap = map_data.reshape(Z, Y).T.copy()
                print(
                    f"Loaded residual correction map: "
                    f"{p.ResidualCorrectionMap} ({Y}x{Z})"
                )
            else:
                print(
                    f"Warning: residual map {p.ResidualCorrectionMap} "
                    f"has {map_data.size} elements, expected {Y * Z}; ignoring."
                )
        except (OSError, ValueError) as e:
            print(f"Warning: could not load residual map "
                  f"{p.ResidualCorrectionMap}: {e}")


# ─── Frame reader ────────────────────────────────────────────────────────────
def read_frame(path: str | Path, frameNr: int) -> np.ndarray:
    """Read a single decompressed frame as float64. Adjusts for skipFrame
    is the caller's responsibility — pass the absolute frame index.
    """
    with open_zarr(path) as root:
        return np.asarray(root["exchange/data"][frameNr], dtype=np.float64)


def read_frames(path: str | Path, start: int, end: int) -> np.ndarray:
    """Read frames [start, end) as float64. Returns shape (end-start, Y, Z)."""
    with open_zarr(path) as root:
        return np.asarray(root["exchange/data"][start:end], dtype=np.float64)


def frame_omega(p: ZarrParams, frameNr: int) -> float:
    """Compute omega for an absolute frame index, honoring omegaCenter override."""
    if p.omegaCenter is not None and frameNr < p.omegaCenter.size:
        return float(p.omegaCenter[frameNr])
    return p.omegaStart + p.omegaStep * frameNr


__all__ = [
    "open_zarr",
    "parse_zarr_params",
    "load_corrections",
    "read_frame",
    "read_frames",
    "frame_omega",
    "canonical_pixel_type",
]
