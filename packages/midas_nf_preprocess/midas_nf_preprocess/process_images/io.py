"""TIFF stack loader for the NF processing pipeline.

Mirrors the file naming convention from
``NF_HEDM/src/ProcessImagesCombined.c`` L820-L856:

  filename = "<data_directory>/<orig_filename>_<NNNNNN>.<ext_orig>"
  index    = (layer-1) * NrFilesPerLayer + RawStartNr + (layer-1) * WFImages + j

where ``j`` runs over [0, NrFilesPerLayer).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import numpy as np
import tifffile
import torch

from .params import ProcessParams


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
    """
    paths = frame_paths(params, layer_nr)
    if not paths:
        raise ValueError(f"No frames to load for layer {layer_nr} (NrFilesPerDistance=0).")
    device = torch.device(device)
    n_sum = max(1, int(getattr(params, "sum_frames", 1) or 1))
    n_out = len(paths) // n_sum

    # Use CPU staging buffer; move to device at the end. tifffile reads return numpy
    # arrays, so the staging buffer is always CPU regardless of target device.
    #
    # The sum accumulates IN PLACE into the output slot rather than reading the
    # whole raw stack and reducing afterwards: at 2048^2 a 1800-frame distance is
    # 30 GB as float32, and summing after the fact would need all of it resident
    # at once. Accumulating keeps the buffer at the POST-SUM size (10 GB for
    # SumFrames 3), so summing lowers peak memory instead of raising it.
    staging = np.zeros(
        (n_out, params.nr_pixels_z, params.nr_pixels_y), dtype=np.float32
    )
    for j, path in enumerate(paths):
        arr = tifffile.imread(path)
        if arr.shape != (params.nr_pixels_z, params.nr_pixels_y):
            raise ValueError(
                f"{path}: shape {arr.shape} != expected "
                f"({params.nr_pixels_z}, {params.nr_pixels_y})"
            )
        staging[j // n_sum] += arr.astype(np.float32, copy=False)

    return torch.from_numpy(staging).to(device=device, dtype=dtype)


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
