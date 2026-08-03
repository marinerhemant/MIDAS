"""midas-dfxm -- differentiable forward + inverse for Dark-Field X-ray Microscopy.

Phase 0 (conventions + deformation field):
    from midas_dfxm import (
        GoniometerSetting, rotation_matrix, rot_x, rot_y, rot_z,
        DeformationField, deform_reflection, polar_decomposition,
        make_uniform_field, with_orientation_gradient, with_screw_dislocation,
    )

Builds on midas-stress (orientation/strain), midas-hkls (structure factors),
midas-2d (continuous-q F, resolution), midas-defect (Stroh contrast solver, GND),
midas-invert (fit/UQ). See implementation_plan.md.
"""
from .conventions import (
    GoniometerSetting,
    rot_x,
    rot_y,
    rot_z,
    rotation_matrix,
)
from .field import (
    DeformationField,
    deform_reflection,
    polar_decomposition,
    reciprocal_basis,
    small_strain_from_F,
)
from .io import (
    fcc_reference_crystal,
    load_borbely_field,
    make_uniform_field,
    with_orientation_gradient,
    with_screw_dislocation,
    with_uniform_strain,
)
from .resolution import ResolutionFunction, aligned_resolution, poulsen_resolution_widths
from .detector import apply_psf, detector_model, quantize_16bit
from .optics import ObjectiveOptics, diffracted_beam_direction, two_theta_from_k_out
from .scan import (
    bragg_two_theta_deg,
    mosaicity_scan,
    reference_q_nom,
    rocking_scan,
    scan_angles,
)
from .forward import (
    dfxm_image,
    dfxm_stack,
    mosaicity_curve,
    structure_factor_intensity,
    voxel_intensity,
)
from .dislocation import (
    StrohDislocation,
    cubic_stiffness,
    dislocation_deformation_field,
    dislocation_dipole,
    edge_dislocation_wall,
    stroh_dislocation,
)
from .typing import (
    classify_character,
    dfxm_contrast,
    dilatation_ratio,
    g_dot_b,
    recover_burgers,
    reflection_signal,
    visibility_series,
)
from .inverse import (
    normal_strain,
    recover_strain_direct,
    recover_strain_regularised,
    strain_covariance,
    strain_design_matrix,
    strain_identifiability,
)
from .detect import (
    DislocationLabel,
    identify_dislocation,
    match_residual,
    weak_beam_stack,
)
from .validate import cross_check_voxel_intensity, voxel_intensity_numpy
from .physics import fit_gnd_density, fit_wall_spacing, gnd_curvature_field
from .generators import (
    dislocation_pileup,
    ensemble_density_per_um2,
    field_from_deformation_gradient,
    field_from_strain,
    random_dislocation_ensemble,
)
from .plasticity import (
    gnd_identifiability,
    multislip_gnd_field,
    nye_from_densities,
    recover_gnd_densities,
    slip_dislocation_types,
)
from .mosaicity_fit import fit_orientation_mosaicity, moment_orientation
from .joint_inverse import fit_dislocation_field, fit_dislocation_ensemble, signal_peaks
from .finite_beam import beam_integrated_observable
from .beamline import (Illumination, beamline_resolution, crl_abcd, crl_focal_length, crl_image, crl_na, delta_beryllium)
from .coherence import (
    coherent_image,
    dislocation_exit_wave,
    coherent_psf,
    dislocation_phase,
    exit_amplitude,
    incoherent_image,
    partially_coherent_image,
)
from .aberration import (
    ABERRATIONS,
    aberrated_psf,
    coeffs_to_tensor,
    convolve_psf,
    fit_aberration,
    wiener_deconvolve,
    zernike_terms,
)
from .field_inverse import (
    angular_condition_number,
    angular_sensitivity_matrix,
    decompose_deformation,
    deformation_covariance,
    deformation_design_matrix,
    deformation_identifiability,
    deformation_observable,
    fisher_information,
    crlb_trace,
    select_reflections_greedy,
    recover_deformation_direct,
    recover_deformation_regularised,
    reference_Q,
)
from .coded_aperture import (
    coded_forward,
    coded_identifiability,
    coding_matrix,
    de_bruijn,
    decode_nnls,
    decode_regularized,
    depth_axis,
)
from .pink_beam import (
    EPS_MONO,
    EPS_PINK,
    axial_reciprocal_width,
    intensity_gain,
    pink_beam_res_cov,
    pink_beam_resolution,
    strain_resolution_ratio,
)
from .cpfem import (
    cpfem_true_strain,
    load_cpfem_field,
    save_cpfem_field,
    validate_dfxm_on_cpfem,
)

__all__ = [
    "GoniometerSetting",
    "rotation_matrix",
    "rot_x",
    "rot_y",
    "rot_z",
    "DeformationField",
    "deform_reflection",
    "polar_decomposition",
    "reciprocal_basis",
    "small_strain_from_F",
    "fcc_reference_crystal",
    "make_uniform_field",
    "with_orientation_gradient",
    "with_uniform_strain",
    "with_screw_dislocation",
    "load_borbely_field",
    "ResolutionFunction",
    "aligned_resolution",
    "poulsen_resolution_widths",
    "detector_model",
    "apply_psf",
    "quantize_16bit",
    "ObjectiveOptics",
    "diffracted_beam_direction",
    "two_theta_from_k_out",
    "bragg_two_theta_deg",
    "reference_q_nom",
    "mosaicity_scan",
    "rocking_scan",
    "scan_angles",
    "dfxm_image",
    "dfxm_stack",
    "mosaicity_curve",
    "voxel_intensity",
    "structure_factor_intensity",
    "StrohDislocation",
    "stroh_dislocation",
    "cubic_stiffness",
    "dislocation_deformation_field",
    "edge_dislocation_wall",
    "dislocation_dipole",
    "g_dot_b",
    "reflection_signal",
    "dfxm_contrast",
    "dilatation_ratio",
    "classify_character",
    "visibility_series",
    "recover_burgers",
    "normal_strain",
    "strain_design_matrix",
    "strain_identifiability",
    "recover_strain_direct",
    "recover_strain_regularised",
    "strain_covariance",
    "identify_dislocation",
    "DislocationLabel",
    "weak_beam_stack",
    "match_residual",
    "cross_check_voxel_intensity",
    "voxel_intensity_numpy",
    "gnd_curvature_field",
    "fit_gnd_density",
    "fit_wall_spacing",
    "field_from_deformation_gradient",
    "field_from_strain",
    "dislocation_pileup",
    "random_dislocation_ensemble",
    "ensemble_density_per_um2",
    "slip_dislocation_types",
    "nye_from_densities",
    "gnd_identifiability",
    "recover_gnd_densities",
    "multislip_gnd_field",
    "load_cpfem_field",
    "save_cpfem_field",
    "cpfem_true_strain",
    "validate_dfxm_on_cpfem",
    "fit_orientation_mosaicity",
    "moment_orientation",
    "reference_Q",
    "deformation_design_matrix",
    "deformation_identifiability",
    "deformation_observable",
    "recover_deformation_direct",
    "recover_deformation_regularised",
    "deformation_covariance",
    "fisher_information",
    "crlb_trace",
    "select_reflections_greedy",
    "angular_sensitivity_matrix",
    "angular_condition_number",
    "decompose_deformation",
    "fit_dislocation_field",
    "fit_dislocation_ensemble",
    "signal_peaks",
    "beam_integrated_observable",
    "Illumination",
    "beamline_resolution",
    "crl_abcd",
    "crl_focal_length",
    "crl_image",
    "crl_na",
    "delta_beryllium",
    "de_bruijn",
    "coding_matrix",
    "coded_forward",
    "coded_identifiability",
    "decode_nnls",
    "decode_regularized",
    "depth_axis",
    "EPS_MONO",
    "EPS_PINK",
    "axial_reciprocal_width",
    "intensity_gain",
    "pink_beam_resolution",
    "pink_beam_res_cov",
    "strain_resolution_ratio",
]

__version__ = "0.3.0"
