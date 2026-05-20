"""calc_radius — replaces ``CalcRadiusAllZarr`` (442 LoC of C).

Filters merged peaks by ring membership (``|R - RingRad| < Width``), computes
Bragg angle, grain volume, grain radius, and per-ring powder intensity. Writes
the 24-column ``Radius_StartNr_*_EndNr_*.csv``.

Submodule :mod:`theoretical` provides a *theoretical* (structure-factor based)
per-ring intensity reference for V_grain normalization — required for pf-HEDM
samples with only a handful of grains, where the empirical powder reference
is biased.  See :func:`midas_transforms.radius.theoretical.theoretical_intensity_per_ring`.
"""

from .core import calc_radius, RadiusResult
from .forward_model import predicted_spot_intensities
from .refine import RefineResult, refine_vmap_joint
from .theoretical import (
    FFGrainTensors,
    FFSpotTensors,
    RingTable,
    SpotTensors,
    aggregate_per_grain,
    aggregate_per_voxel,
    load_ff_grains_to_tensors,
    load_ff_spots_to_tensors,
    load_rings_from_hkls_csv,
    load_spots_from_input_extra_info_csvs,
    per_spot_relative_volume,
    refine_K_per_ring_closed_form,
    theoretical_intensity_per_ring,
)

__all__ = [
    "calc_radius",
    "RadiusResult",
    # theoretical reference for V_grain
    "RingTable",
    "SpotTensors",
    "aggregate_per_grain",
    "aggregate_per_voxel",
    "load_rings_from_hkls_csv",
    "load_spots_from_input_extra_info_csvs",
    "per_spot_relative_volume",
    "theoretical_intensity_per_ring",
    # forward model + K refinement
    "predicted_spot_intensities",
    "refine_K_per_ring_closed_form",
    # joint V + K refinement
    "RefineResult",
    "refine_vmap_joint",
    # FF plumbing
    "FFGrainTensors",
    "FFSpotTensors",
    "load_ff_grains_to_tensors",
    "load_ff_spots_to_tensors",
]
