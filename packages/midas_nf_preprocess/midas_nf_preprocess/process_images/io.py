"""Frame sources for the NF processing pipeline -- TIFF stacks and HDF5.

Two acquisition layouts, one interface
--------------------------------------
1-ID writes **one file per frame** and the frame index is arithmetic
(``NF_HEDM/src/ProcessImagesCombined.c`` L820-L856)::

  filename = "<data_directory>/<orig_filename>_<NNNNNN>.<ext_orig>"
  index    = (layer-1) * NrFilesPerLayer + RawStartNr + (layer-1) * WFImages + j

20-ID HT-HEDM writes **one HDF5 per detector distance**, with every frame of
that distance inside ``exchange/data`` as ``(N, Z, Y)`` uint16.  The file index
therefore advances per DISTANCE, not per frame.

Those two do not share a "path of frame j" abstraction, which is why this module
is organised around :class:`FrameSource` -- "give me frames [lo, hi) of layer L"
-- rather than around a path list.  ``frame_paths`` and ``load_tiff_stack``
survive unchanged for the 1-ID callers that already use them.

Why a source and not just a bigger loader
-----------------------------------------
``process_all`` used to load a whole layer into RAM before doing anything.  On
the 20-ID Oryx that is 1442 x 4600 x 5320, i.e. **141 GB at fp32** -- which is
what made HDF5 support pointless on its own and kept 20-ID out of the pipeline
entirely.  A source reads a block at a time, so peak memory is set by the block
size and not by the scan length.

Reading is single-threaded ON PURPOSE
-------------------------------------
``h5py`` is not safe under concurrent reads unless HDF5 itself was built
thread-safe, which is not guaranteed here.  ``process_layer`` therefore reads
each block in the parent thread and only the per-frame PROCESSING is threaded.
Do not "optimise" this by moving the read inside the pool.

Pixel scaling is never inferred
-------------------------------
``PixelScale`` divides raw counts on the way in and defaults to **1.0**.  It is
not auto-detected, because the encoding is per SCAN, not per detector: on the
same 20-ID camera serial, ``nfdev_jul26`` is 10-bit stored x64 (max 65472) and
``bt_20id_jul26b`` is 12-bit unscaled (max 4092).  Dividing the second by 64
turns "threshold 2" into "threshold 128", which thresholds the PEDESTAL and
makes background look like signal -- a mistake that produced three consecutive
wrong distance calibrations before it was caught.  :func:`check_pixel_scale`
warns when the data disagrees with the setting, but the value stays the user's.
"""

from __future__ import annotations

import warnings

from pathlib import Path
from typing import Optional, Union

import numpy as np
import tifffile
import torch

from .params import ProcessParams


# ----------------------------------------------------------------------
# 1-ID: one file per frame
# ----------------------------------------------------------------------

def frame_paths(params: ProcessParams, layer_nr: int) -> list[str]:
    """Return the RAW TIFF paths for a given 1-indexed layer.

    ``NrFilesPerDistance`` is the RAW image count per distance -- what the
    detector wrote -- so this returns exactly that many paths and the layer
    stride is the same number, independent of ``SumFrames``. Summing is applied
    afterwards by :func:`load_tiff_stack`, which collapses consecutive groups of
    ``SumFrames`` into one output frame.

    That independence is the point: changing SumFrames must not change WHICH
    files are read, only how they are grouped. When the count was instead read
    as post-sum, raising SumFrames alone silently multiplied the file demand and
    walked off the end of the scan.
    """
    if layer_nr < 1:
        raise ValueError(f"layer_nr must be >= 1, got {layer_nr}")
    base = f"{params.data_directory}/{params.orig_filename}"
    # C: StartNr = RawStartNr + (nLayers - 1) * WFImages
    # C: FileNr  = ((nLayers - 1) * NrFilesPerLayer) + StartNr + j
    start = params.raw_start_nr + (layer_nr - 1) * params.wf_images
    base_idx = (layer_nr - 1) * params.nr_files_per_distance
    n = params.nr_files_per_distance
    return [
        f"{base}_{base_idx + start + j:06d}.{params.ext_orig}" for j in range(n)
    ]


# ----------------------------------------------------------------------
# 20-ID: one file per distance
# ----------------------------------------------------------------------

def layer_file(params: ProcessParams, layer_nr: int) -> str:
    """Path of the single container file holding all frames of one layer.

    Used by the HDF5 layout, where ``layer_nr`` indexes the DETECTOR DISTANCE
    and one file holds that distance's whole omega sweep.  The index advances
    by one per distance::

        <data_directory>/<orig_filename>_<RawStartNr + layer_nr - 1>.<ext_orig>

    e.g. ``NF_Au_cube_0802_000708.h5`` / ``_000709.h5`` / ``_000710.h5`` for the
    three distances of one layer, with ``RawStartNr 708``.

    A sample layer (a second y position) is a separate parameter file with its
    own ``RawStartNr``.  Guessing a stride between sample layers from the file
    numbering would be inventing a convention: at 20-ID the six files of a
    two-layer scan are grouped distance-MINOR, and that grouping was established
    from the beam-shadow observables, not from the names.
    """
    if layer_nr < 1:
        raise ValueError(f"layer_nr must be >= 1, got {layer_nr}")
    idx = params.raw_start_nr + (layer_nr - 1)
    return (f"{params.data_directory}/{params.orig_filename}"
            f"_{idx:06d}.{params.ext_orig}")


def is_hdf5(params: ProcessParams) -> bool:
    """Whether ``extOrig`` names an HDF5 container."""
    return str(params.ext_orig).lower().lstrip(".") in ("h5", "hdf5", "hdf", "nxs")


def check_pixel_scale(block: np.ndarray, scale: float) -> None:
    """Warn when the data's encoding disagrees with ``PixelScale``.

    Two failure directions, both silent without this:

    * ``scale == 1`` but every value is a multiple of 64 and the maximum runs
      past 12-bit full scale -- the frames are 10-bit stored x64 and every
      threshold is 64x too small.
    * ``scale == 64`` but the values are NOT multiples of 64 -- the frames are
      unscaled and every threshold is 64x too large, which thresholds the
      pedestal rather than the signal.

    Warns; never corrects.  The encoding is a property of the scan and the
    parameter file is where it gets declared.
    """
    finite = block[np.isfinite(block)]
    if finite.size == 0:
        return
    vmax = float(finite.max())
    nz = finite[finite > 0]
    all_mult_64 = bool(nz.size) and bool(np.all(np.mod(nz, 64) == 0))
    if scale == 1.0 and all_mult_64 and vmax > 4095:
        warnings.warn(
            f"PixelScale is 1 but every non-zero value is a multiple of 64 and "
            f"the maximum is {vmax:.0f} (past 12-bit full scale). This looks "
            f"like 10-bit data stored x64, in which case every threshold is 64x "
            f"too small. Set PixelScale 64 if that is what this scan is -- the "
            f"encoding is per SCAN, so check np.unique on a frame rather than "
            f"inheriting it from another campaign.",
            RuntimeWarning, stacklevel=3,
        )
    elif scale == 64.0 and not all_mult_64:
        warnings.warn(
            f"PixelScale is 64 but the values are not multiples of 64 (max "
            f"{vmax:.0f}). If these frames are unscaled, dividing by 64 makes "
            f"every threshold 64x too large and thresholds the PEDESTAL, so the "
            f"background reads as signal. Check np.unique on one frame.",
            RuntimeWarning, stacklevel=3,
        )


# ----------------------------------------------------------------------
# Frame sources
# ----------------------------------------------------------------------

class FrameSource:
    """Block-wise reader for one layer, in POST-SUM frame units.

    Subclasses implement :meth:`_read_raw`.  Summing, pixel scaling and shape
    validation are handled here so both layouts behave identically.

    Attributes
    ----------
    n_frames : int
        Post-sum frame count -- what the outputs are sized by.
    shape : tuple
        ``(n_frames, Z, Y)``.  Present so a source can stand in for a stack
        where only the frame count is needed.
    """

    def __init__(self, params: ProcessParams, layer_nr: int):
        self.params = params
        self.layer_nr = int(layer_nr)
        self.n_sum = params.n_sum
        self.n_raw = params.n_raw_per_distance
        self.n_frames = params.n_frames_per_distance
        self.nz = int(params.nr_pixels_z)
        self.ny = int(params.nr_pixels_y)
        self.scale = float(getattr(params, "pixel_scale", 1.0) or 1.0)
        self._checked_scale = False

    @property
    def shape(self) -> tuple:
        return (self.n_frames, self.nz, self.ny)

    def _read_raw(self, lo: int, hi: int) -> np.ndarray:
        """Raw frames ``[lo, hi)`` as float32 ``[hi-lo, Z, Y]``, unscaled."""
        raise NotImplementedError

    def _read_raw_rows(self, lo: int, hi: int, r0: int, r1: int) -> np.ndarray:
        """Raw frames ``[lo, hi)``, rows ``[r0, r1)`` -- ``[hi-lo, r1-r0, Y]``.

        Default reads whole frames and slices.  Override where the format can
        read a row band directly.
        """
        return self._read_raw(lo, hi)[:, r0:r1, :]

    def read_rows(self, idx, r0: int, r1: int) -> np.ndarray:
        """Post-sum frames ``idx``, rows ``[r0, r1)`` -- ``[len(idx), r1-r0, Y]``.

        The row band is what keeps the temporal median inside a memory budget:
        the whole-layer median needs every frame resident, this needs
        ``len(idx) x (r1-r0) x Y``.
        """
        idx = [int(i) for i in idx]
        out = np.zeros((len(idx), r1 - r0, self.ny), dtype=np.float32)
        for k, i in enumerate(idx):
            raw = self._read_raw_rows(i * self.n_sum, (i + 1) * self.n_sum, r0, r1)
            out[k] = raw.sum(axis=0) if self.n_sum > 1 else raw[0]
        if self.scale != 1.0:
            out /= self.scale
        return out

    def read_block(self, lo: int, hi: int) -> np.ndarray:
        """Post-sum frames ``[lo, hi)`` as float32 ``[hi-lo, Z, Y]``.

        With ``SumFrames n``, post-sum frame k is the sum of raw frames
        ``[k*n, (k+1)*n)``; the raw frames are accumulated as they are read so
        the peak buffer is the POST-SUM size, exactly as in
        :func:`load_tiff_stack`.
        """
        if lo < 0 or hi > self.n_frames or lo >= hi:
            raise ValueError(
                f"block [{lo}, {hi}) out of range for {self.n_frames} frames")
        n_out = hi - lo
        out = np.zeros((n_out, self.nz, self.ny), dtype=np.float32)
        raw = self._read_raw(lo * self.n_sum, hi * self.n_sum)
        for j in range(raw.shape[0]):
            out[j // self.n_sum] += raw[j]
        if not self._checked_scale:
            check_pixel_scale(raw, self.scale)
            self._checked_scale = True
        if self.scale != 1.0:
            out /= self.scale
        return out

    def read_frames(self, idx) -> np.ndarray:
        """Post-sum frames at the given indices, gathered one at a time.

        For scattered indices (the temporal-median subsample, the threshold
        probe).  Contiguous work should use :meth:`read_block`.
        """
        idx = [int(i) for i in idx]
        out = np.empty((len(idx), self.nz, self.ny), dtype=np.float32)
        for k, i in enumerate(idx):
            out[k] = self.read_block(i, i + 1)[0]
        return out

    def _validate_shape(self, shape: tuple, what: str) -> None:
        if tuple(shape[-2:]) != (self.nz, self.ny):
            raise ValueError(
                f"{what}: frame shape {tuple(shape[-2:])} != expected "
                f"({self.nz}, {self.ny})")

    def _validate(self, arr: np.ndarray, what: str) -> None:
        self._validate_shape(arr.shape, what)

    def close(self) -> None:
        pass

    def __enter__(self) -> "FrameSource":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


class TiffFrameSource(FrameSource):
    """One file per frame -- the 1-ID layout."""

    def __init__(self, params: ProcessParams, layer_nr: int):
        super().__init__(params, layer_nr)
        self.paths = frame_paths(params, layer_nr)
        if not self.paths:
            raise ValueError(
                f"No frames to load for layer {layer_nr} (NrFilesPerDistance=0).")

    def _read_raw(self, lo: int, hi: int) -> np.ndarray:
        out = np.empty((hi - lo, self.nz, self.ny), dtype=np.float32)
        for k, j in enumerate(range(lo, hi)):
            arr = tifffile.imread(self.paths[j])
            self._validate(arr, self.paths[j])
            out[k] = arr.astype(np.float32, copy=False)
        return out


class Hdf5FrameSource(FrameSource):
    """One file per layer, all frames in one dataset -- the 20-ID layout.

    The file handle is held open for the life of the source: reopening per
    block would re-pay the chunk-index read every time, and the 20-ID files are
    chunked (1, 1500, 1960) against a 4600x5320 frame, so a single frame already
    costs ~35 MB of chunk reads.
    """

    def __init__(self, params: ProcessParams, layer_nr: int):
        super().__init__(params, layer_nr)
        try:
            import h5py
        except ImportError as exc:                    # pragma: no cover
            raise ImportError(
                "Reading HDF5 frames needs h5py. Install it, or convert to "
                "TIFF -- though conversion is discouraged: the pixel encoding "
                "and the omega-sign question both survive it and become "
                "harder to see."
            ) from exc
        self.path = layer_file(params, layer_nr)
        if not Path(self.path).exists():
            raise FileNotFoundError(
                f"{self.path} not found. With extOrig {params.ext_orig} one "
                f"file holds one DETECTOR DISTANCE, and the index is "
                f"RawStartNr + layer - 1 (RawStartNr={params.raw_start_nr}, "
                f"layer={layer_nr}). A second sample layer is a separate "
                f"parameter file with its own RawStartNr.")
        self.data_loc = str(getattr(params, "data_loc", "exchange/data"))
        self._f = h5py.File(self.path, "r")
        if self.data_loc not in self._f:
            keys = list(self._f.keys())
            self._f.close()
            raise KeyError(
                f"{self.path} has no dataset {self.data_loc!r} "
                f"(DataLoc). Top-level keys: {keys}")
        self._d = self._f[self.data_loc]
        if self._d.ndim != 3:
            self._f.close()
            raise ValueError(
                f"{self.path}:{self.data_loc} has shape {self._d.shape}; "
                f"expected 3-D (N, Z, Y).")
        n_avail = int(self._d.shape[0])
        if self.n_raw > n_avail:
            self._f.close()
            raise ValueError(
                f"NrFilesPerDistance {self.n_raw} exceeds the {n_avail} frames "
                f"in {self.path}:{self.data_loc}. Note the frame count can "
                f"legitimately EXCEED 360 degrees -- 20-ID scans have been seen "
                f"with 1442 frames spanning -180 to +180.25, where the last two "
                f"duplicate the first two -- so set NrFilesPerDistance from the "
                f"omega range, not from the frame count.")
        self._validate_shape(self._d.shape, self.path)

    def _read_raw(self, lo: int, hi: int) -> np.ndarray:
        return self._d[lo:hi].astype(np.float32, copy=False)

    def _read_raw_rows(self, lo: int, hi: int, r0: int, r1: int) -> np.ndarray:
        # Row-band slab read: HDF5 serves this without materialising the frame.
        return self._d[lo:hi, r0:r1, :].astype(np.float32, copy=False)

    def close(self) -> None:
        f = getattr(self, "_f", None)
        if f is not None:
            f.close()
            self._f = None


def open_source(params: ProcessParams, layer_nr: int) -> FrameSource:
    """Frame source for one layer, chosen from ``extOrig``."""
    if is_hdf5(params):
        return Hdf5FrameSource(params, layer_nr)
    return TiffFrameSource(params, layer_nr)


# ----------------------------------------------------------------------
# Whole-layer loads
# ----------------------------------------------------------------------

def load_stack(
    params: ProcessParams,
    layer_nr: int,
    device: Union[str, torch.device] = "cpu",
    dtype: torch.dtype = torch.float64,
    block: int = 64,
) -> torch.Tensor:
    """Load a whole layer into a tensor ``[N, Z, Y]``, either layout.

    Materialises the layer.  On a large detector that is exactly the cost
    :class:`FrameSource` exists to avoid -- 1442 x 4600 x 5320 is 141 GB at
    fp32 -- so prefer the streaming path in ``process_layer`` for anything at
    20-ID scale, and use this for tests, notebooks and 1-ID-sized scans.
    """
    with open_source(params, layer_nr) as src:
        staging = np.zeros((src.n_frames, src.nz, src.ny), dtype=np.float32)
        for lo in range(0, src.n_frames, block):
            hi = min(lo + block, src.n_frames)
            staging[lo:hi] = src.read_block(lo, hi)
    return torch.from_numpy(staging).to(device=device, dtype=dtype)


def load_tiff_stack(
    params: ProcessParams,
    layer_nr: int,
    device: Union[str, torch.device] = "cpu",
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """Load all frames for one layer into a tensor of shape ``[N, Z, Y]``.

    The C code reads pixels in scanline order with shape (NrPixelsZ, NrPixelsY).
    We follow that convention: the first spatial axis is Z (rows), the second is Y.
    Pixel intensities are uint16 in the source TIFFs; we promote to ``dtype`` to keep
    the autograd path live for downstream ops.

    Retained under its original name for callers that predate HDF5 support;
    :func:`load_stack` is the layout-agnostic entry point.
    """
    return load_stack(params, layer_nr, device=device, dtype=dtype)


def from_tensor(
    stack: torch.Tensor,
    *,
    nr_pixels_y: Optional[int] = None,
    nr_pixels_z: Optional[int] = None,
) -> torch.Tensor:
    """Validate and return a stack tensor for use in tests / notebooks.

    Accepts an ``[N, Z, Y]`` tensor and checks shape against the expected pixel grid
    if provided. Returns the same tensor (no copy) for shape conformance.
    """
    if stack.ndim != 3:
        raise ValueError(f"Expected 3D tensor [N, Z, Y], got shape {tuple(stack.shape)}")
    n, z, y = stack.shape
    if nr_pixels_z is not None and z != nr_pixels_z:
        raise ValueError(f"Z mismatch: tensor has {z}, expected {nr_pixels_z}")
    if nr_pixels_y is not None and y != nr_pixels_y:
        raise ValueError(f"Y mismatch: tensor has {y}, expected {nr_pixels_y}")
    return stack
