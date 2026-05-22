"""midas-integrate-v2 — differentiable radial integration.

Companion to ``midas-integrate`` (the production CPU/GPU integration
package). v2 sits alongside v1, not in place of it: where v1's
``Map.bin`` is treated as a constant (the right call for fast batch
integration), v2 keeps the full path from calibration parameters →
``(R, η)`` per pixel → bin assignment → integrated profile in PyTorch
with autograd, so a loss on the integrated profile can flow gradient
back to the geometry, distortion, and residual-correction parameters.

The v1 hot path (sparse-CSR matmul over a precomputed bin assignment)
is preserved — v2's trick is wrapping it in an :class:`autograd.Function`
that supplies the implicit gradient on the backward pass.
"""
from __future__ import annotations

__version__ = "0.3.1"

from .spec import IntegrationSpec, DISTORTION_NAMES
from .compat import (
    spec_from_v1_params,
    spec_from_v1_paramstest,
    v1_params_from_spec,
    bc_to_poni,
    poni_to_bc,
    make_pyfai_integrator,
)
from .binning import (
    build_map, MapCache,
    SoftBinGeometry, integrate_soft, integrate_soft_batch,
    HardBinGeometry, integrate_hard, integrate_hard_batch,
    SubpixelBinGeometry, integrate_subpixel, integrate_subpixel_batch,
    PolygonBinGeometry, integrate_polygon,
    apply_trans_opt_forward, needs_trans_opt,
    write_map_bin_from_geometry,
    normalise_mask, mask_fraction,
    LearnableMask, sparsity_prior, smoothness_prior,
    LearnableGain, gain_unity_prior, gain_smoothness_prior, gain_module_block_prior,
    integrate_hard_with_variance,
    integrate_subpixel_with_variance,
    integrate_polygon_with_variance,
)
from .losses import (
    EtaSliceLoss, WedgeLoss, RingMaskedLoss,
)
from .bootstrap import (
    estimate_BC_from_image,
    estimate_initial_spec,
)
from .io import (
    write_csv, write_xye, write_fxye, write_esg, write_dat,
    write_2d_csv, write_h5,
    ProfileMetadata, build_provenance,
)
from .streaming import (
    FrameSource, NumpyArraySource, TIFFGlobSource,
    HDF5FrameSource, ZarrFrameSource,
    GEBinaryFrameSource, EDFFrameSource,
    FrameNormalizer, reject_cosmic_rays, reject_spatial_spikes,
    azimuthal_sigma_clip, azimuthal_sigma_clip_multi_panel,
    integrate_stream,
)
from .ring_detect import (
    CALIBRANTS, DetectedRing, MaterialMatch,
    detect_rings, suggest_material,
)
from .kernels import (
    integrate,
    profile_1d,
    IntegrationGeometry,
    build_geometry,
)
from .forward import pixel_to_REta_from_spec, eval_pixel_REta
from .diff import integrate_diff, profile_1d_diff, soft_bin_indices_weights
from .losses import (
    ProfileMSELoss,
    ProfileWeightedMSELoss,
    EtaUniformityLoss,
    PeakPositionLoss,
    GaussianPriorLoss,
    MultiImageLoss,
    BatchedSpecLoss,
)
from . import pdf
from . import dac
from . import texture
from . import diagnostics
from . import pipelines
from . import grazing
from . import inelastic
from .corrections import (
    PerRingOffsets,
    RBFResidualCorrection,
    IdentityResidualCorrection,
    integrate_with_corrections,
    delta_r_k_from_R,
    assign_ring,
    PolarizationCorrection,
    SolidAngleCorrection,
    polarization_factor,
    solid_angle_factor_flat,
    solid_angle_factor_tilted,
    build_q_bin_edges_in_R,
    EmptySubtraction,
    CylindricalAbsorption,
    ComptonSubtraction,
)

__all__ = [
    "__version__",
    "IntegrationSpec",
    "DISTORTION_NAMES",
    "spec_from_v1_params",
    "spec_from_v1_paramstest",
    "v1_params_from_spec",
    "bc_to_poni",
    "poni_to_bc",
    "make_pyfai_integrator",
    "build_map",
    "MapCache",
    "SoftBinGeometry",
    "integrate_soft",
    "integrate_soft_batch",
    "HardBinGeometry",
    "integrate_hard",
    "integrate_hard_batch",
    "SubpixelBinGeometry",
    "integrate_subpixel",
    "integrate_subpixel_batch",
    "PolygonBinGeometry",
    "integrate_polygon",
    "apply_trans_opt_forward",
    "needs_trans_opt",
    "write_map_bin_from_geometry",
    "integrate",
    "profile_1d",
    "IntegrationGeometry",
    "build_geometry",
    "pixel_to_REta_from_spec",
    "eval_pixel_REta",
    "integrate_diff",
    "profile_1d_diff",
    "soft_bin_indices_weights",
    "PerRingOffsets",
    "RBFResidualCorrection",
    "IdentityResidualCorrection",
    "integrate_with_corrections",
    "delta_r_k_from_R",
    "assign_ring",
    "PolarizationCorrection",
    "SolidAngleCorrection",
    "polarization_factor",
    "solid_angle_factor_flat",
    "build_q_bin_edges_in_R",
    "EmptySubtraction",
    "CylindricalAbsorption",
    "ComptonSubtraction",
    "pdf",
    "ProfileMSELoss",
    "ProfileWeightedMSELoss",
    "EtaUniformityLoss",
    "PeakPositionLoss",
    "GaussianPriorLoss",
    "MultiImageLoss",
    "BatchedSpecLoss",
    # quasi-2D losses
    "EtaSliceLoss",
    "WedgeLoss",
    "RingMaskedLoss",
    # mask + variance
    "normalise_mask",
    "mask_fraction",
    "LearnableMask",
    "sparsity_prior",
    "smoothness_prior",
    "LearnableGain",
    "gain_unity_prior",
    "gain_smoothness_prior",
    "gain_module_block_prior",
    "integrate_hard_with_variance",
    "integrate_subpixel_with_variance",
    "integrate_polygon_with_variance",
    # bootstrap
    "estimate_BC_from_image",
    "estimate_initial_spec",
    # output writers
    "write_csv",
    "write_xye",
    "write_fxye",
    "write_esg",
    "write_dat",
    "write_2d_csv",
    "write_h5",
    "ProfileMetadata",
    "build_provenance",
    # streaming
    "FrameSource",
    "NumpyArraySource",
    "TIFFGlobSource",
    "HDF5FrameSource",
    "ZarrFrameSource",
    "GEBinaryFrameSource",
    "EDFFrameSource",
    "FrameNormalizer",
    "reject_cosmic_rays",
    "reject_spatial_spikes",
    "azimuthal_sigma_clip",
    "azimuthal_sigma_clip_multi_panel",
    "integrate_stream",
    # ring detection
    "CALIBRANTS",
    "DetectedRing",
    "MaterialMatch",
    "detect_rings",
    "suggest_material",
]
