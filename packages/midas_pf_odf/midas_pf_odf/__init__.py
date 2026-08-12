"""midas-pf-odf — joint per-grain peak-shape inversion of pf-HEDM data.

Phase 1: per-voxel (R, ε) from peak shapes. Phase 2: per-voxel ODF.
"""

__version__ = "0.1.1"

from midas_pf_odf.simulate import (
    SinglePhaseGrainPlant,
    plant_single_grain,
    simulate_grain_patches,
)
from midas_pf_odf.forward import (
    joint_grain_forward,
    soft_beam_gate,
)
from midas_pf_odf.inversion import (
    neighbor_edges_from_grid_ij,
    fit_grain_peakshape,
    GrainPeakFitResult,
    IdentifiabilityMode,
)
from midas_pf_odf.validation import (
    recovery_metrics,
    holdout_score,
)
from midas_pf_odf.multi_grain import (
    MultiGrainPlant,
    plant_multi_grain,
    split_into_grains,
    simulate_multi_grain,
    fit_multi_grain,
)
from midas_pf_odf.centroid_baseline import (
    fit_grain_centroid_baseline,
    measured_centroids_from_patches,
    predicted_centroids,
)
from midas_pf_odf.calibrate import (
    RawFrameCalibration,
    calibrate_raw_frame_geometry,
    layer_model_factory,
    measure_patch_offsets,
)
from midas_pf_odf.io import (
    PFGrainDataset,
    distortion_from_paramstest,
    saturation_threshold_from_paramstest,
    load_pf_grain,
    build_model_from_paramstest,
    geometry_from_paramstest,
    parse_paramstest,
    assemble_grain_patch_data,
    crop_patches_from_frames,
    build_model_from_zarr,
    geometry_from_zarr,
    read_zarr_params,
    ZarrFrameSource,
)

__all__ = [
    "SinglePhaseGrainPlant",
    "plant_single_grain",
    "simulate_grain_patches",
    "joint_grain_forward",
    "soft_beam_gate",
    "fit_grain_peakshape",
    "GrainPeakFitResult",
    "IdentifiabilityMode",
    "neighbor_edges_from_grid_ij",
    "recovery_metrics",
    "holdout_score",
    "MultiGrainPlant",
    "plant_multi_grain",
    "split_into_grains",
    "simulate_multi_grain",
    "fit_multi_grain",
    "fit_grain_centroid_baseline",
    "measured_centroids_from_patches",
    "predicted_centroids",
    "PFGrainDataset",
    "load_pf_grain",
    "build_model_from_paramstest",
    "geometry_from_paramstest",
    "parse_paramstest",
    "assemble_grain_patch_data",
    "crop_patches_from_frames",
    "build_model_from_zarr",
    "geometry_from_zarr",
    "read_zarr_params",
    "ZarrFrameSource",
    "distortion_from_paramstest",
    "saturation_threshold_from_paramstest",
    "RawFrameCalibration",
    "calibrate_raw_frame_geometry",
    "layer_model_factory",
    "measure_patch_offsets",
]
