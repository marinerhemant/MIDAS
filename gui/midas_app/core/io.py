"""Unified MIDAS image loaders.

A single ``load_frame(path, frame=0, **opts)`` dispatches by extension:
  .tif/.tiff   -> tifffile
  .h5/.hdf5/.nxs/.hdf -> h5py (dataset path required for HDF5)
  .zarr/.zip   -> zarr (assumes /exchange/data layout)
  .bz2         -> decompress to temp + recurse
  .ge[1-5]/.bin/raw -> headered raw binary (default 8192-byte header, uint16)

All loaders return float32 2D arrays. Optional transpose/flip/mask are applied.
"""

from __future__ import annotations

import os
import bz2
import tempfile
from typing import Optional

import numpy as np

try:
    import tifffile
except ImportError:
    tifffile = None

try:
    import h5py
except ImportError:
    h5py = None

try:
    import zarr
except ImportError:
    zarr = None


# ── Generic helpers ───────────────────────────────────────────────────

def _apply_orient(data: np.ndarray, transpose: bool, hflip: bool, vflip: bool) -> np.ndarray:
    if transpose:
        data = np.transpose(data)
    if hflip and vflip:
        data = data[::-1, ::-1].copy()
    elif hflip:
        data = data[::-1, :].copy()
    elif vflip:
        data = data[:, ::-1].copy()
    return data


def _apply_mask(data: np.ndarray, mask: Optional[np.ndarray]) -> np.ndarray:
    if mask is not None and mask.shape == data.shape:
        data = data.copy()
        data[mask == 1] = 0
    return data


def _decompress_bz2(path: str) -> str:
    """Decompress .bz2 to a temp file. Caller must remove the result."""
    fd, temp = tempfile.mkstemp(suffix='.bin')
    try:
        with bz2.open(path, 'rb') as src, os.fdopen(fd, 'wb') as dst:
            while chunk := src.read(1 << 20):
                dst.write(chunk)
    except Exception:
        os.close(fd)
        if os.path.exists(temp):
            os.remove(temp)
        raise
    return temp


# ── Format readers ────────────────────────────────────────────────────

def _read_tiff(path: str, frame: int) -> np.ndarray:
    if tifffile is None:
        raise ImportError("tifffile not installed")
    return tifffile.imread(path, key=frame)


def _read_h5(path: str, frame: int, dataset: str) -> np.ndarray:
    if h5py is None:
        raise ImportError("h5py not installed")
    with h5py.File(path, 'r') as f:
        if dataset not in f:
            raise KeyError(f"dataset '{dataset}' not in {path}")
        dset = f[dataset]
        if dset.ndim == 3:
            return np.asarray(dset[frame, :, :])
        return np.asarray(dset[:])


def _read_zarr(path: str, frame: int, dataset: str = 'exchange/data') -> np.ndarray:
    if zarr is None:
        raise ImportError("zarr not installed")
    store = zarr.open(path, mode='r')
    dset = store[dataset]
    if dset.ndim == 3:
        return np.asarray(dset[frame, :, :])
    return np.asarray(dset[:])


def _read_raw(path: str, frame: int, ny: int, nz: int,
              header: int = 8192, bytes_per_pixel: int = 2) -> np.ndarray:
    dtype = np.uint16 if bytes_per_pixel == 2 else np.int32
    with open(path, 'rb') as f:
        f.seek(header + frame * bytes_per_pixel * ny * nz, os.SEEK_SET)
        raw = np.fromfile(f, dtype=dtype, count=ny * nz)
    if raw.size != ny * nz:
        raise IOError(f"Short read: got {raw.size} of {ny*nz} pixels at frame {frame}")
    return raw.reshape((ny, nz))


# ── Public API ────────────────────────────────────────────────────────

def load_frame(path: str, frame: int = 0, *,
               ny: Optional[int] = None, nz: Optional[int] = None,
               header: int = 8192, bytes_per_pixel: int = 2,
               hdf5_dataset: str = '/exchange/data',
               zarr_dataset: str = 'exchange/data',
               transpose: bool = False, hflip: bool = False, vflip: bool = False,
               mask: Optional[np.ndarray] = None) -> np.ndarray:
    """Load a single 2D frame from any supported MIDAS image format.

    For raw GE/binary inputs ``ny`` and ``nz`` must be supplied.
    """
    if path.endswith('.bz2'):
        temp = _decompress_bz2(path)
        try:
            return load_frame(temp, frame=frame, ny=ny, nz=nz,
                              header=header, bytes_per_pixel=bytes_per_pixel,
                              hdf5_dataset=hdf5_dataset, zarr_dataset=zarr_dataset,
                              transpose=transpose, hflip=hflip, vflip=vflip, mask=mask)
        finally:
            if os.path.exists(temp):
                os.remove(temp)

    ext = os.path.splitext(path)[1].lower()

    if ext in ('.tif', '.tiff'):
        data = _read_tiff(path, frame)
    elif ext in ('.h5', '.hdf5', '.hdf', '.nxs'):
        data = _read_h5(path, frame, hdf5_dataset)
    elif ext == '.zarr' or path.endswith('.zarr.zip') or ext == '.zip':
        data = _read_zarr(path, frame, zarr_dataset)
    else:
        if ny is None or nz is None:
            raise ValueError(f"raw loader needs ny/nz for {path}")
        data = _read_raw(path, frame, ny, nz, header=header, bytes_per_pixel=bytes_per_pixel)

    data = data.astype(np.float32, copy=False)
    data = _apply_orient(data, transpose, hflip, vflip)
    return _apply_mask(data, mask)


def load_max(path: str, n_frames: int, start: int = 0, **kwargs) -> np.ndarray:
    """Pixel-wise max over [start, start+n_frames)."""
    out = None
    for i in range(start, start + n_frames):
        frame = load_frame(path, frame=i, **kwargs)
        out = frame if out is None else np.maximum(out, frame)
    return out


def load_sum(path: str, n_frames: int, start: int = 0, **kwargs) -> np.ndarray:
    """Pixel-wise sum over [start, start+n_frames). Returns float32."""
    out = None
    for i in range(start, start + n_frames):
        frame = load_frame(path, frame=i, **kwargs).astype(np.float32)
        out = frame if out is None else (out + frame)
    return out


def detect_format(path: str) -> str:
    """Return a short string describing the file format."""
    if path.endswith('.bz2'):
        return 'bz2-' + detect_format(path[:-4])
    ext = os.path.splitext(path)[1].lower()
    if ext in ('.tif', '.tiff'):
        return 'tiff'
    if ext in ('.h5', '.hdf5', '.hdf', '.nxs'):
        return 'hdf5'
    if ext == '.zarr' or path.endswith('.zarr.zip') or ext == '.zip':
        return 'zarr'
    if ext.startswith('.ge'):
        return 'ge-raw'
    return 'raw'


def n_frames(path: str, hdf5_dataset: str = '/exchange/data',
             zarr_dataset: str = 'exchange/data',
             ny: Optional[int] = None, nz: Optional[int] = None,
             header: int = 8192, bytes_per_pixel: int = 2) -> int:
    """Best-effort frame count. Returns 1 when not determinable."""
    fmt = detect_format(path)
    if fmt == 'bz2-tiff' or fmt == 'tiff':
        if tifffile is None:
            return 1
        try:
            with tifffile.TiffFile(path) as tf:
                return len(tf.pages)
        except Exception:
            return 1
    if fmt == 'hdf5':
        if h5py is None:
            return 1
        try:
            with h5py.File(path, 'r') as f:
                if hdf5_dataset in f:
                    d = f[hdf5_dataset]
                    return d.shape[0] if d.ndim == 3 else 1
        except Exception:
            return 1
    if fmt == 'zarr':
        if zarr is None:
            return 1
        try:
            store = zarr.open(path, mode='r')
            d = store[zarr_dataset]
            return d.shape[0] if d.ndim == 3 else 1
        except Exception:
            return 1
    if fmt in ('ge-raw', 'raw') and ny and nz:
        try:
            sz = os.path.getsize(path) - header
            return max(1, sz // (bytes_per_pixel * ny * nz))
        except OSError:
            return 1
    return 1
