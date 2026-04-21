"""I/O helpers — primarily the zarr.zip packaging that MIDASIntegrator
consumes.

The C binary expects a very specific zarr-zip schema:

    exchange/data                          # (n_frames, ny, nz) float/uint
    analysis/process/analysis_parameters/<key>/0   # 1-element zarr array per scalar

Minimum keys required for a successful integration (from
``src/c/IntegratorZarrOMP.c``):
    Wavelength, Lsd, RMin, RMax, EtaMin, EtaMax, RBinSize, EtaBinSize,
    PixelSize (or PixelSizeY + PixelSizeZ), X, Y, Z, U, V, W

We support single-frame TIFF → zarr.zip here; multi-frame HDF5 input
(the production workflow) ports from ``utils/ffGenerateZipRefactor.py``
in v0.2.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import Iterable, Mapping, Optional, Union

import numpy as np

from ._config import IntegrationConfig

__all__ = ["make_zarr_zip"]


# Keys that IntegratorZarrOMP reads via ``ReadZarrChunk(..., sizeof(int))``
# (4-byte int32). Everything else is read as ``sizeof(double)`` (8-byte float64).
_INT32_KEYS = frozenset({
    "Normalize", "SkipFrame", "DoPeakFit", "FitROIPadding",
    "SumImages", "OmegaSumFrames", "SaveIndividualFrames",
})


def make_zarr_zip(
    image: Union[str, Path, np.ndarray],
    config: IntegrationConfig,
    output_path: Union[str, Path],
    *,
    dark: Union[str, Path, np.ndarray, None] = None,
    extra_params: Optional[Mapping[str, float]] = None,
    chunk_y: int = 512,
    chunk_z: int = 512,
) -> Path:
    """Pack a single 2D detector image into MIDAS-schema zarr.zip.

    Parameters
    ----------
    image : path or ndarray
        2D input frame. If a path, read via tifffile. If 3D, the first frame
        is used and a warning is emitted — multi-frame support lands in v0.2.
    config : IntegrationConfig
        Geometry + binning. All scalars needed by IntegratorZarrOMP are
        written into ``analysis/process/analysis_parameters/…``.
    output_path : path
        ``.zarr.zip`` or ``.zip`` file to write.
    dark : path or ndarray, optional
        Dark frame to subtract before packaging. If omitted, the bundle is
        stored raw and MIDAS applies dark via its own parameter file.
    extra_params : mapping, optional
        Extra analysis parameters to store verbatim (``U, V, W, SHpL`` etc.
        for peak-fit runs).
    """
    import zarr
    from numcodecs import Blosc

    img = _load_image(image)
    if img.ndim == 3:
        img = img[0]  # multi-frame → first frame (TODO v0.2)
    if dark is not None:
        darr = _load_image(dark)
        if darr.ndim == 3:
            darr = darr[0]
        img = img.astype(np.float32) - darr.astype(np.float32)

    # Stack into (n_frames, ny, nz). IntegratorZarrOMP expects this layout.
    data_3d = img[np.newaxis, ...].astype(np.float32)

    out = Path(output_path)
    # zarr writes into a directory; we zip it up for the .zarr.zip layout the
    # binary expects. Use a temp workspace so partial writes never leak.
    with tempfile.TemporaryDirectory(prefix="mac_zarr_") as tmp:
        store_dir = Path(tmp) / "store"
        store_dir.mkdir()

        root = zarr.open(str(store_dir), mode="w")
        compressor = Blosc(cname="zstd", clevel=1, shuffle=Blosc.BITSHUFFLE)

        exchange = root.create_group("exchange")
        exchange.create_dataset(
            "data",
            data=data_3d,
            chunks=(1, chunk_y, chunk_z),
            compressor=compressor,
        )

        # IntegratorZarrOMP's ZarrReader unconditionally invokes
        # blosc1_decompress, so every chunk — including the 1-element
        # scalar parameter arrays — MUST be compressed with numcodecs.Blosc
        # (which writes blosc1 format).
        #
        # The C reader also expects specific numpy dtypes per key — int32
        # vs float64 mismatches produce "destSize != uncompressedSize"
        # decompression errors. Keys listed in _INT32_KEYS are read via
        # `sizeof(int)` (4 bytes); everything else is `sizeof(double)` (8).
        params_compressor = Blosc(cname="lz4", clevel=1, shuffle=Blosc.BITSHUFFLE)
        params = root.create_group("analysis/process/analysis_parameters")
        merged = _build_param_dict(config, extra_params or {})
        str_params = _build_string_params(config)
        for key, value in merged.items():
            dtype = np.int32 if key in _INT32_KEYS else np.float64
            params.create_dataset(
                key, data=np.asarray([value], dtype=dtype),
                compressor=params_compressor,
            )
        for key, value in str_params.items():
            # Byte-string scalar: MIDAS reads via ReadZarrString → raw bytes
            # + blosc1 decompression, then null-terminates.
            arr = np.array([np.bytes_(str(value).encode("utf-8"))])
            params.create_dataset(key, data=arr, compressor=params_compressor)

        # Zip up the store into the final .zarr.zip.
        # shutil.make_archive adds ".zip" — we let it and then rename.
        tmp_zip = Path(tmp) / "bundle"
        shutil.make_archive(str(tmp_zip), "zip", root_dir=store_dir)
        shutil.move(str(tmp_zip.with_suffix(".zip")), str(out))

    return out


def _build_param_dict(
    config: IntegrationConfig,
    extra: Mapping[str, float],
) -> dict[str, float]:
    """Assemble the minimal ``analysis_parameters`` payload.

    IntegratorZarrOMP expects scalar arrays under specific key names; we
    map from IntegrationConfig's Pythonic names.
    """
    p: dict[str, float] = {
        "Wavelength": float(config.wavelength),
        "Lsd": float(config.lsd),
        "RMin": float(config.r_min),
        "RMax": float(config.r_max if config.r_max is not None else
                      min(config.nr_pixels_y, config.nr_pixels_z) / 2.0),
        "EtaMin": float(config.eta_min),
        "EtaMax": float(config.eta_max),
        "RBinSize": float(config.r_bin_size),
        "EtaBinSize": float(config.eta_bin_size),
        "PixelSize": float(config.pixel_size),
        "PixelSizeY": float(config.pixel_size),
        "PixelSizeZ": float(config.pixel_size),
        "Normalize": 1,
    }
    # GSAS-II peak-fit defaults (harmless placeholders when DoPeakFit=0).
    p.setdefault("X", 0.0)
    p.setdefault("Y", 0.0)
    p.setdefault("Z", 0.0)
    p.setdefault("U", 0.0)
    p.setdefault("V", 0.0)
    p.setdefault("W", 0.0)
    p.setdefault("SHpL", 0.0)
    p.setdefault("Polariz", float(config.polarization_fraction))

    if config.q_min is not None:
        p["QMin"] = float(config.q_min)
    if config.q_max is not None:
        p["QMax"] = float(config.q_max)
    if config.q_bin_size is not None:
        p["QBinSize"] = float(config.q_bin_size)

    for k, v in extra.items():
        # Skip string-valued keys — they go through _build_string_params.
        if isinstance(v, str):
            continue
        p[k] = float(v)
    return p


_STRING_KEYS = frozenset({
    "ResultFolder", "GapFile", "BadPxFile", "MaskFile", "PanelShiftsFile",
})


def _build_string_params(config: IntegrationConfig) -> dict[str, str]:
    """Byte-string scalar params (read via ReadZarrString in the C binary).

    Every zarr.zip bundle MUST have at least ``ResultFolder`` set, otherwise
    IntegratorZarrOMP tries to open ``(null)/Map.bin``. Default to "."
    because the Integrator wrapper stages Map.bin into the same dir as
    the zarr.zip.
    """
    s: dict[str, str] = {"ResultFolder": "."}
    if config.mask_file:
        s["MaskFile"] = config.mask_file
    if config.panel_shifts_file:
        s["PanelShiftsFile"] = config.panel_shifts_file
    # Any extra string-valued analysis_parameters the user set via
    # config.extra_params get forwarded too.
    for k, v in config.extra_params.items():
        if k in _STRING_KEYS and isinstance(v, str):
            s[k] = v
    return s


def _load_image(image: Union[str, Path, np.ndarray]) -> np.ndarray:
    if isinstance(image, np.ndarray):
        return image
    path = Path(image)
    if path.suffix.lower() in (".tif", ".tiff"):
        import tifffile
        return tifffile.imread(path)
    raise ValueError(
        f"Unsupported image format: {path.suffix}. Load into numpy yourself "
        f"for non-TIFF inputs.")
