"""midas-grain-odf: differentiable per-grain ODF inversion from FF-HEDM."""

# Register the HDF5 filter plugins (bitshuffle, LZ4, zstd) that Eiger / Dectris
# and ESRF files are written with.  hdf5plugin ships the plugin binaries but
# only registers them with the HDF5 library when it is imported — declaring it
# as a dependency is not enough.  Without this, reading such a dataset fails
# with "can't open directory (/usr/local/lib/plugin)".
try:  # pragma: no cover - environment-dependent
    import hdf5plugin as _hdf5plugin  # noqa: F401
except ImportError:  # HDF5 files with no plugin filter still read fine
    pass

__version__ = "0.1.2"

from midas_grain_odf.odf import (
    GrainODF,
    ParticleODF,
    BinghamMixtureODF,
    AnisotropicBinghamODF,
    VoxelGridODF,
)
from midas_grain_odf.forward_helpers import (
    forward_orientations,
    grain_avg_predicted_spots,
)
from midas_grain_odf.spot_extract import (
    splat_spots_to_patches,
    splat_spots_to_patches_dense,
    splat_spots_to_patches_sparse,
    SpotPatchSpec,
)
from midas_grain_odf.losses import (
    compute_delta_table,
    odf_weighted_predicted_centroid,
    shape_mse_loss,
)
from midas_grain_odf.inversion import (
    fit_grain_odf,
    fit_grain_odf_multistart,
    GrainFitResult,
)

__all__ = [
    "GrainODF",
    "ParticleODF",
    "BinghamMixtureODF",
    "AnisotropicBinghamODF",
    "VoxelGridODF",
    "forward_orientations",
    "grain_avg_predicted_spots",
    "splat_spots_to_patches",
    "splat_spots_to_patches_dense",
    "splat_spots_to_patches_sparse",
    "SpotPatchSpec",
    "compute_delta_table",
    "odf_weighted_predicted_centroid",
    "shape_mse_loss",
    "fit_grain_odf",
    "fit_grain_odf_multistart",
    "GrainFitResult",
]
