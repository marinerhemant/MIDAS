"""Reading APS ``/exchange`` HDF5 tomography files, and writing results back.

The input layout this targets::

    /exchange/data                       (n_frames, n_z, n_x)  projections
    /exchange/dark                       (n_z, n_x)            dark reference
    /exchange/bright                     (2, n_z, n_x)         white references
    /analysis/process/analysis_parameters/{CropXL,CropXR,CropZL,CropZR,shift}
    /analysis/process/analysis_parameters/RotationAngle        (optional)
    /measurement/process/scan_parameters/{start,step}

``h5py`` is an optional dependency — install ``midas-tomo[hdf5]`` — so this
module is imported lazily by the CLI rather than from ``__init__``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

__all__ = [
    "ExchangeScan",
    "read_exchange",
    "crop_slice",
    "stage_exchange_to_binary",
    "write_recon_hdf5",
]

log = logging.getLogger(__name__)

_PARAMS = "analysis/process/analysis_parameters"
_SCAN = "measurement/process/scan_parameters"


def crop_slice(left: int, right: int, length: int) -> slice:
    """Slice that removes *left* from the start and *right* from the end.

    Exists because the obvious spelling is wrong. The legacy driver wrote
    ``arr[left:-right]``, which for the perfectly ordinary ``right == 0``
    becomes ``arr[left:0]`` — an **empty array**, silently. A scan with no
    right-hand crop would reconstruct nothing, with no error to explain it.
    """
    if left < 0 or right < 0:
        raise ValueError(f"crop amounts must be non-negative, got {left}, {right}")
    stop = length - right
    if stop <= left:
        raise ValueError(
            f"crop leaves nothing: length {length}, removing {left} from the "
            f"start and {right} from the end"
        )
    return slice(left, stop)


@dataclass
class ExchangeScan:
    """Everything :func:`read_exchange` recovered from the file."""

    data: np.ndarray          # (n_frames, n_z, n_x) uint16 projections
    dark: np.ndarray          # (n_z, n_x) float32
    whites: np.ndarray        # (2, n_z, n_x) float32
    angles: np.ndarray        # (n_frames,) degrees
    shift: float              # rotation-axis shift from the file
    rotation_angle: float     # detector roll, degrees (0 = none)
    source: Path

    @property
    def n_frames(self) -> int:
        return int(self.data.shape[0])

    @property
    def det_xdim(self) -> int:
        return int(self.dark.shape[1])

    @property
    def det_ydim(self) -> int:
        return int(self.dark.shape[0])


def _rotate_stack(stack: np.ndarray, angle_deg: float) -> np.ndarray:
    """Rotate each 2-D plane of *stack* about its centre.

    The legacy driver used OpenCV for this, which is a heavy dependency for
    one affine warp. SciPy does the same job and is far more likely to be
    present already; it is imported here so that a scan with no roll
    (``RotationAngle`` absent or 0, the overwhelming majority) needs neither.

    Note the legacy code built its rotation matrix about ``(nX/2, nY/2)``
    where ``nX, nY = dark.shape`` — i.e. ``(rows/2, cols/2)`` — while OpenCV's
    centre argument is ``(x, y) = (cols/2, rows/2)``. On a non-square detector
    that rotated about the wrong point. ``scipy.ndimage.rotate`` takes the
    plane axes directly and has no such ambiguity.
    """
    try:
        from scipy.ndimage import rotate
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ImportError(
            "this scan has RotationAngle != 0, which needs SciPy for the "
            "detector-roll correction. Install scipy, or set RotationAngle to "
            "0 if the roll is not wanted."
        ) from exc

    planes = (-2, -1)
    return rotate(stack, angle_deg, axes=planes, reshape=False, order=1, mode="nearest")


def read_exchange(
    path: str | Path,
    *,
    slab: tuple[int, int] | None = None,
    apply_rotation: bool = True,
) -> ExchangeScan:
    """Read an APS ``/exchange`` file.

    Parameters
    ----------
    path
        HDF5 file.
    slab
        ``(z0, z1)`` in *cropped* coordinates, to read only part of the stack
        — used by cleanup tuning, which needs a few slices rather than the
        whole volume. None reads everything.
    apply_rotation
        Apply the ``RotationAngle`` detector-roll correction if present.

    Returns
    -------
    ExchangeScan
    """
    try:
        import h5py
    except ImportError as exc:
        raise ImportError(
            "reading /exchange HDF5 needs h5py. Install it with "
            "`pip install midas-tomo[hdf5]` or `conda install h5py`."
        ) from exc

    path = Path(path)
    with h5py.File(path, "r") as hf:
        def param(name: str, default=None):
            key = f"{_PARAMS}/{name}"
            if key not in hf:
                if default is None:
                    raise KeyError(f"{path} has no {key}")
                return default
            return hf[key][0]

        dxl, dxr = int(param("CropXL")), int(param("CropXR"))
        dzl, dzr = int(param("CropZL")), int(param("CropZR"))
        shift = float(param("shift"))
        rot = float(param("RotationAngle", 0.0))

        dark_ds = hf["exchange/dark"]
        nz_full, nx_full = dark_ds.shape[-2], dark_ds.shape[-1]
        zsl = crop_slice(dzl, dzr, nz_full)
        xsl = crop_slice(dxl, dxr, nx_full)

        if slab is not None:
            z0, z1 = slab
            if (z1 - z0) % 2:
                raise ValueError(
                    f"slab must span an even number of slices (the engine "
                    f"reconstructs in pairs); got {z0}..{z1}"
                )
            zsl = slice(zsl.start + z0, zsl.start + z1)

        dark = np.asarray(dark_ds[zsl, xsl], dtype=np.float32)
        whites = np.asarray(hf["exchange/bright"][:, zsl, xsl], dtype=np.float32)
        data = np.asarray(hf["exchange/data"][:, zsl, xsl], dtype=np.uint16)

        n_frames = data.shape[0]
        start = float(hf[f"{_SCAN}/start"][0])
        step = float(hf[f"{_SCAN}/step"][0])
        angles = start + step * np.arange(n_frames, dtype=np.float64)

    if rot and apply_rotation:
        log.info("applying detector roll of %.4f deg", rot)
        data = _rotate_stack(data.astype(np.float32), rot).astype(np.uint16)
        dark = _rotate_stack(dark, rot)
        whites = _rotate_stack(whites, rot)

    log.info(
        "read %s: %d frames, %dx%d after crop (X %d..%d, Z %d..%d), shift %.3f",
        path.name, n_frames, dark.shape[1], dark.shape[0],
        xsl.start, xsl.stop, zsl.start, zsl.stop, shift,
    )
    return ExchangeScan(
        data=data, dark=dark, whites=whites, angles=angles,
        shift=shift, rotation_angle=rot, source=path,
    )


def stage_exchange_to_binary(
    path: str | Path,
    out_bin: str | Path,
    *,
    apply_rotation: bool = True,
    frames_per_chunk: int = 64,
) -> ExchangeScan:
    """Convert an ``/exchange`` file to the engine's binary layout, streaming.

    Same output as reading with :func:`read_exchange` and letting
    :func:`~midas_tomo.api.run_tomo` stage it, but the projections are copied a
    chunk of frames at a time instead of being materialised whole. That matters
    because reading in Python is what lets the C be built without HDF5 — it
    would be a poor trade if it also imposed a "whole scan must fit in RAM"
    limit the C reader did not have.

    Returns an :class:`ExchangeScan` whose ``data`` is an **empty** array: the
    projections went to disk, not into memory. Everything else (dark, whites,
    angles, shift, dimensions) is populated, and ``n_frames`` is corrected to
    the real count.

    Parameters
    ----------
    frames_per_chunk
        Frames per read/write. Larger is faster and uses more memory; the
        default keeps the buffer at a few hundred MB for a typical detector.
    """
    try:
        import h5py
    except ImportError as exc:
        raise ImportError(
            "reading /exchange HDF5 needs h5py. Install it with "
            "`pip install midas-tomo[hdf5]` or `conda install h5py`."
        ) from exc

    path, out_bin = Path(path), Path(out_bin)
    out_bin.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(path, "r") as hf:
        def param(name: str, default=None):
            key = f"{_PARAMS}/{name}"
            if key not in hf:
                if default is None:
                    raise KeyError(f"{path} has no {key}")
                return default
            return hf[key][0]

        dxl, dxr = int(param("CropXL")), int(param("CropXR"))
        dzl, dzr = int(param("CropZL")), int(param("CropZR"))
        shift = float(param("shift"))
        rot = float(param("RotationAngle", 0.0))

        dark_ds = hf["exchange/dark"]
        zsl = crop_slice(dzl, dzr, dark_ds.shape[-2])
        xsl = crop_slice(dxl, dxr, dark_ds.shape[-1])

        dark = np.asarray(dark_ds[zsl, xsl], dtype=np.float32)
        whites = np.asarray(hf["exchange/bright"][:, zsl, xsl], dtype=np.float32)
        if rot and apply_rotation:
            dark = _rotate_stack(dark, rot)
            whites = _rotate_stack(whites, rot)

        data_ds = hf["exchange/data"]
        n_frames = int(data_ds.shape[0])
        start = float(hf[f"{_SCAN}/start"][0])
        step = float(hf[f"{_SCAN}/step"][0])
        angles = start + step * np.arange(n_frames, dtype=np.float64)

        with out_bin.open("wb") as f:
            dark.tofile(f)
            whites.tofile(f)
            for i0 in range(0, n_frames, frames_per_chunk):
                i1 = min(i0 + frames_per_chunk, n_frames)
                chunk = np.asarray(data_ds[i0:i1, zsl, xsl])
                if rot and apply_rotation:
                    chunk = _rotate_stack(chunk.astype(np.float32), rot)
                chunk.astype(np.uint16).tofile(f)

    log.info("staged %s -> %s (%d frames, %dx%d)", path.name, out_bin.name,
             n_frames, dark.shape[1], dark.shape[0])
    return ExchangeScan(
        data=np.empty((n_frames, 0, 0), dtype=np.uint16),  # on disk, not in RAM
        dark=dark, whites=whites, angles=angles,
        shift=shift, rotation_angle=rot, source=path,
    )


def write_recon_hdf5(
    path: str | Path,
    recon: np.ndarray,
    *,
    angles: np.ndarray | None = None,
    shifts: np.ndarray | None = None,
    metadata: dict[str, Any] | None = None,
    compression: str | None = "gzip",
) -> Path:
    """Write a reconstruction cube to HDF5 in an NXtomo-flavoured layout.

    The engine's own output is a bare ``.bin`` whose shape lives only in the
    filename, which does not survive being copied or renamed. This writes the
    array with its shape, angles and provenance attached.

    Parameters
    ----------
    recon : ndarray
        ``(n_shifts, n_slices, X, X)`` as returned by ``run_tomo``.
    metadata
        Extra key/value pairs stored as attributes on ``/entry``. Anything
        not a str/int/float is stringified.
    """
    try:
        import h5py
    except ImportError as exc:
        raise ImportError(
            "writing HDF5 needs h5py. Install with `pip install midas-tomo[hdf5]`."
        ) from exc

    recon = np.asarray(recon)
    if recon.ndim != 4:
        raise ValueError(f"recon must be 4-D (shift, slice, y, x); got {recon.shape}")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as hf:
        entry = hf.create_group("entry")
        entry.attrs["NX_class"] = "NXentry"
        entry.attrs["definition"] = "NXtomoproc"
        entry.attrs["creator"] = "midas-tomo"

        from . import __version__
        entry.attrs["midas_tomo_version"] = __version__

        rec = entry.create_group("reconstruction")
        rec.attrs["NX_class"] = "NXprocess"
        # Chunk a slice at a time: readers almost always want one image, not
        # a stripe through the whole cube.
        chunks = (1, 1, recon.shape[2], recon.shape[3]) if compression else None
        rec.create_dataset(
            "data", data=recon, compression=compression, chunks=chunks,
        )
        rec["data"].attrs["axes"] = "shift:slice:y:x"

        if angles is not None:
            rec.create_dataset("rotation_angle", data=np.asarray(angles, dtype=np.float64))
            rec["rotation_angle"].attrs["units"] = "degrees"
        if shifts is not None:
            rec.create_dataset("axis_shift", data=np.asarray(shifts, dtype=np.float64))
            rec["axis_shift"].attrs["units"] = "pixels"

        for k, v in (metadata or {}).items():
            entry.attrs[k] = v if isinstance(v, (str, int, float)) else str(v)

    log.info("wrote %s %s", path, recon.shape)
    return path
