"""Image readers — TIFF, HDF5, GE binary, CBF.  File format auto-detected
from the extension; ``data_loc`` argument is for HDF5 dataset path.

Bad-pixel sentinels
-------------------
Detectors mark module gaps and dead pixels with an out-of-band value rather
than a flag.  The convention is not consistent between vendors:

- Pilatus writes **negative** values (``-1`` gap, ``-2`` overflow), so the
  familiar ``img[img < 0] = 0`` catches them.
- Dectris EIGER writes the **largest representable unsigned value**
  (``2**32-1`` for uint32).  That is the opposite end of the range, so every
  ``< 0`` guard fails open and a fitter is handed 4.29e9 as a photon count.

:func:`read_image` therefore detects the unsigned dtype-max sentinel by
default, zeroes those pixels, and warns.  Pass ``return_mask=True`` to get the
boolean mask back so it can be written out for ``midas-integrate-v2 --mask``.
"""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import Optional

import numpy as np


class BadPixelSentinelWarning(UserWarning):
    """Raised when a frame carries out-of-band bad-pixel values.

    Not an error: the pixels are zeroed and reported.  It fires so that a
    sentinel is never silently averaged into a profile or a ring centroid.
    """


def _split_sentinel(arr: np.ndarray, bad_value):
    """Return ``(arr_with_sentinels_zeroed, mask)``.

    ``mask`` is a 2-D boolean array (``True`` = bad) or ``None`` when nothing
    was flagged.  For a 3-D stack the mask is the union over frames, so a pixel
    bad in any frame is bad in the result — the conservative choice, and it
    keeps the mask meaningful after the frames are averaged.

    Runs on the raw integer data **before** any averaging.  Averaging first
    would blend the sentinel with real counts into a value that no longer
    equals the sentinel and could not be detected at all.
    """
    if bad_value is None:
        return arr, None
    if bad_value == "auto":
        if arr.dtype.kind != "u":
            return arr, None            # signed / float: the < 0 convention
        sentinel = np.iinfo(arr.dtype).max
    else:
        sentinel = bad_value

    hit = arr == sentinel
    if not hit.any():
        return arr, None
    mask = hit.any(axis=0) if hit.ndim == 3 else hit
    arr = np.where(hit, 0, arr)
    return arr, mask


def read_image(
    path: str | Path,
    *,
    data_loc: str = "exchange/data",
    skip_frame: int = 0,
    im_trans: tuple = (),
    data_type: int = 1,
    bad_value="auto",
    return_mask: bool = False,
):
    """Read a 2-D image from any supported format.

    Parameters
    ----------
    path : file path.  Extension determines the reader.
    data_loc : HDF5 dataset path (only for .h5 / .hdf5).
    skip_frame : number of leading frames to skip in multi-frame files.
    im_trans : tuple of MIDAS image-transformation codes
        (1 = flip Y, 2 = flip Z, 3 = transpose).
    data_type : raw-binary numeric type (only for GE-style files):
        1 = uint16, 2 = float64, 3 = float32, 4 = uint32, 5 = int32.
    bad_value : ``"auto"`` (default) flags the largest representable value of
        an unsigned integer frame as a bad-pixel sentinel — the EIGER / Dectris
        convention.  Pass a number to name the sentinel explicitly (e.g. ``-1``
        for a Pilatus gap), or ``None`` to disable the check entirely and get
        the raw values.
    return_mask : also return the boolean bad-pixel mask.

    Returns
    -------
    np.ndarray of shape (nz, ny), float64 — or ``(image, mask)`` when
    ``return_mask`` is set.  ``mask`` is ``True`` at bad pixels, and is all
    ``False`` when nothing was flagged.
    """
    p = Path(path)
    ext = p.suffix.lower()
    if ext in (".tif", ".tiff"):
        img, mask = _read_tiff(p, bad_value=bad_value)
    elif ext in (".h5", ".hdf5", ".hdf", ".nxs"):
        img, mask = _read_hdf5(p, data_loc=data_loc, skip_frame=skip_frame,
                               bad_value=bad_value)
    elif ext == ".cbf":
        img, mask = _read_cbf(p, bad_value=bad_value)
    elif ".ge" in p.name.lower():
        img, mask = _read_ge(p, data_type=data_type, skip_frame=skip_frame,
                             bad_value=bad_value)
    else:
        raise ValueError(f"Unrecognised image format: {p}")

    if mask is not None:
        warnings.warn(
            f"{p.name}: {int(mask.sum())} of {mask.size} pixels "
            f"({100.0 * mask.mean():.3f} %) carry the bad-pixel sentinel; they "
            f"have been set to 0. Pass them to the integrator as a mask "
            f"(read_image(..., return_mask=True)) — they are not counts.",
            BadPixelSentinelWarning,
            stacklevel=2,
        )

    # the mask has to ride along through the same flips, or it stops lining up
    for opt in im_trans:
        if opt == 1:
            img = img[:, ::-1]
            mask = None if mask is None else mask[:, ::-1]
        elif opt == 2:
            img = img[::-1, :]
            mask = None if mask is None else mask[::-1, :]
        elif opt == 3:
            img = img.T
            mask = None if mask is None else mask.T

    img = np.ascontiguousarray(img.astype(np.float64))
    if not return_mask:
        return img
    if mask is None:
        mask = np.zeros(img.shape, dtype=bool)
    return img, np.ascontiguousarray(mask)


def read_dark(path: str | Path | None, **kwargs) -> Optional[np.ndarray]:
    """Same as :func:`read_image` but returns ``None`` if path is None or empty."""
    if path is None or str(path) == "":
        return None
    return read_image(path, **kwargs)


# ----------------------------------------------------------- backends

def _read_tiff(path: Path, *, bad_value="auto"):
    try:
        import tifffile
        raw = np.asarray(tifffile.imread(str(path)))
    except ImportError:
        from PIL import Image
        raw = np.asarray(Image.open(str(path)))
    raw, mask = _split_sentinel(raw, bad_value)
    return raw.astype(np.float64), mask


def _read_hdf5(path: Path, *, data_loc: str, skip_frame: int, bad_value="auto"):
    import h5py
    with h5py.File(str(path), "r") as f:
        dset = f[data_loc]
        data = dset[skip_frame:] if dset.ndim >= 3 else dset[...]
    data, mask = _split_sentinel(data, bad_value)
    if data.ndim == 3:
        return np.mean(data, axis=0).astype(np.float64), mask
    return data.astype(np.float64), mask


def _read_ge(path: Path, *, data_type: int = 1, skip_frame: int = 0,
             bad_value="auto"):
    """GE binary frame reader.  Tries the standard 8192-byte header first,
    then no header.  Reshapes to 2048², 4096², or 1024² square."""
    dtype_map = {1: np.uint16, 2: np.float64, 3: np.float32,
                 4: np.uint32, 5: np.int32}
    np_dtype = dtype_map.get(data_type, np.uint16)

    def _try(offset):
        arr = np.fromfile(str(path), dtype=np_dtype, offset=offset)
        total = len(arr)
        for side in (2048, 4096, 1024, 512):
            frame = side * side
            if total >= frame and total % frame == 0:
                n = total // frame
                arr = arr.reshape(n, side, side)
                if skip_frame >= n:
                    raise ValueError(
                        f"skip_frame={skip_frame} ≥ {n} frames")
                stack, mask = _split_sentinel(arr[skip_frame:], bad_value)
                return np.mean(stack, axis=0).astype(np.float64), mask
        raise ValueError(f"can't reshape {total} pixels into a square frame")

    try:
        return _try(8192)
    except (ValueError, Exception):
        return _try(0)


def _read_cbf(path: Path, *, bad_value="auto"):
    """CBF reader via MIDAS's own ``midas_zipper`` reader.

    CBF is the Crystallographic Binary File format used by Pilatus / Eiger /
    Varex.  We deliberately do **not** use ``fabio`` here: fabio returns the
    raw frame, whereas the entire MIDAS pipeline (``midas_zipper`` →
    ``midas_peakfit`` → indexer) works in the transposed/double-flipped MIDAS
    convention (``pixels.reshape(nrows, ncols).T[::-1, ::-1]``).  Reading with
    fabio yields a Y↔Z-transposed, both-axes-flipped beam centre and MIRRORED
    tilts/distortion that are invalid for the pipeline.  Using the same reader
    the pipeline uses guarantees the calibration geometry is consistent with
    downstream indexing.
    """
    try:
        from midas_zipper._read_cbf import read_cbf
    except ImportError as exc:  # pragma: no cover - dependency declared in pyproject
        raise RuntimeError(
            "CBF reading requires the `midas-zipper` package (a declared "
            "midas-calibrate-v2 dependency). Install with "
            "`pip install midas-zipper`."
        ) from exc
    _header, data = read_cbf(str(path), check_md5=False)
    data, mask = _split_sentinel(data, bad_value)
    return data.astype(np.float64), mask


__all__ = ["read_image", "read_dark", "BadPixelSentinelWarning"]
