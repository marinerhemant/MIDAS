"""midas-nf-preprocess: Differentiable PyTorch port of NF-HEDM preprocessing.

Bundles the preprocessing stages that sit between raw experimental inputs
and the orientation-fitting step (``FitOrientationOMP``):

  - ``hex_grid``         : voxel grid generation (port of MakeHexGrid)
  - ``tomo_filter``      : grid masking from a tomography image
                           (port of filterGridfromTomo)
  - ``seed_orientations``: NF seed-orientation generation (cache, from-scratch,
                           or from an FF Grains.csv)
  - ``diffr_spots``      : per-orientation diffraction-spot prediction
                           (differentiable port of MakeDiffrSpots; wraps
                           ``midas_diffract.HEDMForwardModel.calc_bragg_geometry``)
  - ``process_images``   : raw TIFF -> SpotsInfo.bin
                           (differentiable port of ProcessImagesCombined)

The shared ``device`` module provides device/dtype resolution
(env vars ``MIDAS_NF_PREPROCESS_DEVICE`` / ``MIDAS_NF_PREPROCESS_DTYPE``)
matching the convention used by ``midas-transforms``.
"""

__version__ = "0.3.0"

from . import (
    device,
    diffr_spots,
    hex_grid,
    process_images,
    seed_orientations,
    tomo_filter,
)
from .device import resolve_device, resolve_dtype, apply_cpu_threads

__all__ = [
    "__version__",
    "device",
    "diffr_spots",
    "hex_grid",
    "process_images",
    "seed_orientations",
    "tomo_filter",
    "resolve_device",
    "resolve_dtype",
    "apply_cpu_threads",
]
