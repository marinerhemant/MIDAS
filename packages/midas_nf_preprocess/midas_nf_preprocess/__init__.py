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

# Register the HDF5 filter plugins (bitshuffle, LZ4, zstd) that Eiger / Dectris
# and ESRF files are written with.  hdf5plugin ships the plugin binaries but
# only registers them with the HDF5 library when it is imported — declaring it
# as a dependency is not enough.  Without this, reading such a dataset fails
# with "can't open directory (/usr/local/lib/plugin)".
try:  # pragma: no cover - environment-dependent
    import hdf5plugin as _hdf5plugin  # noqa: F401
except ImportError:  # HDF5 files with no plugin filter still read fine
    pass

__version__ = "0.7.2"

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
