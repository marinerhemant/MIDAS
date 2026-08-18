"""midas-dct-tt -- differentiable forward + inverse for DCT and topotomography.

DCT (diffraction contrast tomography) and TT (topotomography) share one forward
operator and differ only in scan convention:

===========  ===============================  ==========  =================  ==================
             acceptance                       projection  rotation axis      projections/grain
===========  ===============================  ==========  =================  ==================
**DFXM**     divergence + bandwidth + obj NA  magnified   goniometer         n/a (real-space)
**TT**       divergence + bandwidth           parallel    **parallel to G**  10^2-10^3, chosen
**DCT**      divergence + bandwidth           parallel    lab vertical       10-60, fixed
===========  ===============================  ==========  =================  ==================

so this package is a scan-convention layer plus a projection/reconstruction
layer over ``midas_dfxm`` and ``midas_diffract``, not a new physics stack.

Phase 2 (shape reconstruction)::

    from midas_dct_tt import forward_operator, sirt, reconstruct_differentiable, dice

Phase 1 (forward core)::

    from midas_dct_tt import (
        tt_alignment, tt_resolution, psi_scan, PlaneDetector,
        topograph_image, topograph_stack, bragg_flashes, dct_frames,
    )

Phase 0 (conventions, geometry, containers)::

    from midas_dct_tt import (
        tt_alignment, dct_sample_rotation, TTAlignment,
        bragg_angle_deg, missing_cone_half_angle_deg, diffracted_beam_direction,
        GrainShape, sphere_grain, plate_grain, faceted_grain, attach_uniform_field,
    )

The one geometric result to know before using any of it: with the rotation axis
parallel to ``G``, the diffracted beam sits at ``90 - theta`` from that axis, so
**TT is laminography with a missing cone of half-angle theta**, not tomography.
See :func:`midas_dct_tt.geometry.missing_cone_half_angle_deg`.

Units: micrometers, degrees, Angstrom (wavelength/lattice); reciprocal vectors
carry the ``2*pi`` convention (``|G| = 2*pi/d``), matching ``midas_dfxm``.

See ``implementation_plan.md``. Not released -- this package is private until
``RELEASE_CHECKLIST.md`` is worked through.
"""
__version__ = "0.1.0"

from .conventions import (
    DCT_OMEGA_SIGN_AERO,
    DCT_OMEGA_SIGN_CCW,
    TTAlignment,
    align_vector_to,
    dct_sample_rotation,
    tt_alignment,
)
from .geometry import (
    LAB_X,
    LAB_Y,
    LAB_Z,
    angle_between_deg,
    angle_kh_to_axis_deg,
    bragg_angle_deg,
    bragg_condition_residual,
    diffracted_beam_direction,
    diffracted_wavevector,
    incident_wavevector,
    missing_cone_half_angle_deg,
    rotation_axis_for_reflection,
    wavevector_magnitude,
)
from .acceptance import tt_resolution, tt_resolution_widths
from .extinction import (
    kinematical_validity,
    coherent_path_um,
    extinction_weights,
    direct_beam_transmission,
    extinction_length_um,
    kinematical_path_limit_um,
    primary_extinction_factor,
)
from .forward import (
    PlaneDetector,
    dct_frames,
    expand_subvoxels,
    topograph_image,
    topograph_stack,
    voxel_scattering,
)
from .grain import GrainShape, logits_from_signed_distance
from .inverse import (
    fit_uniform_deformation,
    identifiability_rank,
    intensity_scales,
    local_Q,
    profiled_intensity_residual,
)
from .pairing import Spot, assign_spots, friedel_pairs, friedel_partner_omega, unassigned
from .recon import (
    axial_elongation,
    backproject,
    dice,
    forward_operator,
    iou,
    reconstruct_differentiable,
    sirt,
    total_variation,
)
from .acceptance import (
    AnisotropicTTResolution,
    ObjectiveFreeAcceptance,
    orientation_resolution_deg,
    tt_resolution_aniso,
)
from .field_inverse import (
    affine_basis,
    cross_validate_lambda,
    decompose_affine,
    fit_deformation_field,
    select_lambda_discrepancy,
    smoothness_penalty,
    weighted_corr,
)
from .detector import (
    add_photon_noise,
    effective_dof,
    noise_power_spectrum,
    add_two_stage_noise,
    add_read_noise,
    apply_psf,
    simulate_detector,
)
from .chain import (
    FF_RADIUS_FLOOR_UM,
    GrainRecord,
    TTScanPlan,
    radius_is_suspect,
    read_grains_csv,
    select_tt_candidates,
    tt_scan_plan,
)
from .esrf import (
    ESRF_LENGTH_UNIT_UM,
    ESRFGeometry,
    esrf_center_px,
    esrf_detector,
    esrf_detector_basis,
    esrf_omega_sign,
    load_esrf_parameters,
    mm_to_um,
    rodrigues_to_crystal_to_sample,
)
from .goniometer import (
    INSTRUMENT_OFFSETS_ID11,
    best_reachable_pair,
    instrument_transformation,
    reachable_reflections,
    reciprocal_basis,
    tilt_branches,
    topotomo_tilts,
)
from .rotation_coverage import (
    best_reflection_pair,
    rotation_conditioning,
    sensitivity_moment,
    separation_for_conditioning,
)
from .planning import (
    ReflectionSetReport,
    accessible_reflections,
    rank_reflection_sets,
    second_moment_tensor,
    strain_leakage_for_strain,
    strain_leakage_index,
)
from .project import (
    offdetector_fraction,
    parallel_projection,
    pixel_coordinates,
    project_rays_to_plane,
    splat_bilinear,
)
from .scan import BraggFlash, bragg_flashes, dct_omega_scan, psi_scan
from .io import (
    attach_uniform_field,
    box_grain,
    faceted_grain,
    grain_from_sdf,
    plate_grain,
    regular_grid,
    sphere_grain,
)

__all__ = [
    "__version__",
    # geometry
    "LAB_X",
    "LAB_Y",
    "LAB_Z",
    "angle_between_deg",
    "angle_kh_to_axis_deg",
    "bragg_angle_deg",
    "bragg_condition_residual",
    "diffracted_beam_direction",
    "diffracted_wavevector",
    "incident_wavevector",
    "missing_cone_half_angle_deg",
    "rotation_axis_for_reflection",
    "wavevector_magnitude",
    # conventions
    "DCT_OMEGA_SIGN_AERO",
    "DCT_OMEGA_SIGN_CCW",
    "TTAlignment",
    "align_vector_to",
    "dct_sample_rotation",
    "tt_alignment",
    # acceptance (na = 0)
    "AnisotropicTTResolution",
    "ObjectiveFreeAcceptance",
    "orientation_resolution_deg",
    "tt_resolution",
    "tt_resolution_aniso",
    "tt_resolution_widths",
    # scan builders
    "BraggFlash",
    "bragg_flashes",
    "dct_omega_scan",
    "psi_scan",
    # projection + forward
    "PlaneDetector",
    "dct_frames",
    "expand_subvoxels",
    "offdetector_fraction",
    "parallel_projection",
    "pixel_coordinates",
    "project_rays_to_plane",
    "splat_bilinear",
    "topograph_image",
    "topograph_stack",
    "voxel_scattering",
    # extinction / kinematical validity
    "direct_beam_transmission",
    "extinction_length_um",
    "kinematical_path_limit_um",
    "kinematical_validity",
    "primary_extinction_factor",
    # reconstruction (Phase 2)
    "axial_elongation",
    "backproject",
    "dice",
    "forward_operator",
    "iou",
    "reconstruct_differentiable",
    "sirt",
    "total_variation",
    # deformation inverse (Phase 3)
    "fit_uniform_deformation",
    "identifiability_rank",
    "intensity_scales",
    "profiled_intensity_residual",
    "local_Q",
    # scan planning (Phase 4)
    "ReflectionSetReport",
    "accessible_reflections",
    "rank_reflection_sets",
    "second_moment_tensor",
    "strain_leakage_for_strain",
    "strain_leakage_index",
    # rotation-field conditioning: which pair determines the rotation
    "best_reflection_pair",
    "rotation_conditioning",
    "sensitivity_moment",
    "separation_for_conditioning",
    # goniometer: which of those pairs the stage can actually reach
    "INSTRUMENT_OFFSETS_ID11",
    "best_reachable_pair",
    "instrument_transformation",
    "reachable_reflections",
    "reciprocal_basis",
    "tilt_branches",
    "topotomo_tilts",
    # FF/NF grain map -> TT scan plan (Phase 6)
    "FF_RADIUS_FLOOR_UM",
    "GrainRecord",
    "TTScanPlan",
    "radius_is_suspect",
    "read_grains_csv",
    "select_tt_candidates",
    "tt_scan_plan",
    # 12-D field inverse (Phase 8)
    "affine_basis",
    "cross_validate_lambda",
    "decompose_affine",
    "fit_deformation_field",
    "select_lambda_discrepancy",
    "smoothness_penalty",
    "weighted_corr",
    # detector realism (Phase 7)
    "add_photon_noise",
    "effective_dof",
    "noise_power_spectrum",
    "add_two_stage_noise",
    "add_read_noise",
    "apply_psf",
    "coherent_path_um",
    "extinction_weights",
    "simulate_detector",
    # ESRF interoperability (Phase 5)
    "ESRF_LENGTH_UNIT_UM",
    "ESRFGeometry",
    "esrf_center_px",
    "esrf_detector",
    "esrf_detector_basis",
    "esrf_omega_sign",
    "load_esrf_parameters",
    "rodrigues_to_crystal_to_sample",
    "mm_to_um",
    # pairing / indexing (Phase 2)
    "Spot",
    "assign_spots",
    "friedel_pairs",
    "friedel_partner_omega",
    "unassigned",
    # containers + generators
    "GrainShape",
    "logits_from_signed_distance",
    "attach_uniform_field",
    "box_grain",
    "faceted_grain",
    "grain_from_sdf",
    "plate_grain",
    "regular_grid",
    "sphere_grain",
]
