"""Image readers — TIFF, HDF5, GE binary, CBF.  File format auto-detected
from the extension; ``data_loc`` argument is for HDF5 dataset path.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np


def read_image(
    path: str | Path,
    *,
    data_loc: str = "exchange/data",
    skip_frame: int = 0,
    im_trans: tuple = (),
    data_type: int = 1,
) -> np.ndarray:
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

    Returns
    -------
    np.ndarray of shape (nz, ny), float64.
    """
    p = Path(path)
    ext = p.suffix.lower()
    if ext in (".tif", ".tiff"):
        img = _read_tiff(p)
    elif ext in (".h5", ".hdf5", ".hdf", ".nxs"):
        img = _read_hdf5(p, data_loc=data_loc, skip_frame=skip_frame)
    elif ext == ".cbf":
        img = _read_cbf(p)
    elif ".ge" in p.name.lower():
        img = _read_ge(p, data_type=data_type, skip_frame=skip_frame)
    else:
        raise ValueError(f"Unrecognised image format: {p}")
    for opt in im_trans:
        if opt == 1:
            img = img[:, ::-1]
        elif opt == 2:
            img = img[::-1, :]
        elif opt == 3:
            img = img.T
    return np.ascontiguousarray(img.astype(np.float64))


def read_dark(path: str | Path | None, **kwargs) -> Optional[np.ndarray]:
    """Same as :func:`read_image` but returns ``None`` if path is None or empty."""
    if path is None or str(path) == "":
        return None
    return read_image(path, **kwargs)


# ----------------------------------------------------------- backends

def _read_tiff(path: Path) -> np.ndarray:
    try:
        import tifffile
        return np.asarray(tifffile.imread(str(path)), dtype=np.float64)
    except ImportError:
        from PIL import Image
        return np.asarray(Image.open(str(path)), dtype=np.float64)


def _read_hdf5(path: Path, *, data_loc: str, skip_frame: int) -> np.ndarray:
    import h5py
    with h5py.File(str(path), "r") as f:
        dset = f[data_loc]
        data = dset[skip_frame:] if dset.ndim >= 3 else dset[...]
        if data.ndim == 3:
            return np.mean(data, axis=0).astype(np.float64)
        return data.astype(np.float64)


def _read_ge(path: Path, *, data_type: int = 1, skip_frame: int = 0) -> np.ndarray:
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
                return np.mean(arr[skip_frame:], axis=0).astype(np.float64)
        raise ValueError(f"can't reshape {total} pixels into a square frame")

    try:
        return _try(8192)
    except (ValueError, Exception):
        return _try(0)


def _read_cbf(path: Path) -> np.ndarray:
    """CBF reader via the ``fabio`` package.

    CBF is the Crystallographic Binary File format used by Pilatus / Eiger.
    ``fabio`` is a declared dependency of this package, so the import should
    always succeed; the guard only produces a clear message if it was
    removed from the environment.
    """
    try:
        import fabio
    except ImportError as exc:  # pragma: no cover - dependency guaranteed by pyproject
        raise RuntimeError(
            "CBF reading requires the `fabio` package (a declared "
            "midas-calibrate-v2 dependency). Install with `pip install fabio`."
        ) from exc
    return fabio.open(str(path)).data.astype(np.float64)


__all__ = ["read_image", "read_dark"]
