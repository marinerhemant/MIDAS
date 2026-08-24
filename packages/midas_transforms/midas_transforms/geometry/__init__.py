"""Geometry primitives shared by V-map / soft-attribution / forward modeling.

* :class:`SampleGrid` — voxel positions, grain map, sample mask packaged as
  torch tensors on a single device.
* :class:`BeamProfile` (+ :class:`TopHat`, :class:`Gaussian`) —
  differentiable beam-fraction-over-voxel kernels, ``torch.nn.Module``-based
  so refinable beam parameters integrate with :mod:`torch.optim`.

These live in :mod:`midas_transforms` (rather than :mod:`midas_pipeline`) so
that the forward model in :mod:`midas_transforms.radius.forward_model` can
consume them without a circular dependency on the orchestrator package.
"""
from .absorption import absorption_factor, path_length_in_sample
from .beam import BeamProfile, Gaussian, TopHat
from .detector_mask import (
    bigdet_cell_index,
    build_active_area_bitset,
    build_active_area_bitset_from_zarr,
    pack_bitset,
    write_big_detector_mask,
)
from .registration import (
    CheckResult,
    centroid_containment_check,
    meta_null,
    sinogram_check,
)
from .sample import SampleGrid
from .sample_shape import SampleShape
from .tomo import (
    from_array,
    from_midas_tomo_bin,
    from_nxtomoproc,
    from_square_uint8,
    parse_recon_filename,
    threshold_sensitivity,
)

__all__ = [
    "BeamProfile",
    "CheckResult",
    "Gaussian",
    "SampleGrid",
    "SampleShape",
    "TopHat",
    "absorption_factor",
    "bigdet_cell_index",
    "build_active_area_bitset",
    "build_active_area_bitset_from_zarr",
    "centroid_containment_check",
    "from_array",
    "from_midas_tomo_bin",
    "from_nxtomoproc",
    "from_square_uint8",
    "meta_null",
    "pack_bitset",
    "parse_recon_filename",
    "path_length_in_sample",
    "sinogram_check",
    "threshold_sensitivity",
    "write_big_detector_mask",
]
