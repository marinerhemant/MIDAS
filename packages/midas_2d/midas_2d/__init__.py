"""midas-2d -- differentiable diffraction for 2D / few-layer & ultrafast work.

Finite-size forward core:
    from midas_2d import (
        interference_factor, truncation_rod, nanoplatelet_rod,
        rod_intensity, structure_factor_q,
        cdse_zincblende, build_crystal_tensor,
        make_spectrum, monochromatic,
    )

Bragg coherent diffraction imaging (:mod:`midas_2d.bcdi`,
:mod:`midas_2d.bcdi_io`, :mod:`midas_2d.coherent`):
    from midas_2d import (
        q_basis, conjugate_real_basis, oversampling, sheared_to_lab,
        object_to_amplitude, detector_signal, sample_counts,
        speckle_from_atoms, atoms_to_object,
        load_bcdi, phase_retrieval,
    )

Builds on midas-hkls (structure factors, form factors, lattice, hkl gen) and
midas-pink (energy spectrum).  All forward paths are torch-differentiable and
run on CPU / CUDA / MPS.  See the README and ``midas_2d.examples`` for worked
end-to-end scripts.
"""
from .bcdi import (
    bragg_geometry,
    conjugate_real_basis,
    detector_distance_for_oversampling,
    detector_signal,
    object_to_amplitude,
    oversampling,
    q_basis,
    q_grid,
    rotation_to_bragg,
    rocking_step_for_oversampling,
    sample_counts,
    shear_angles_deg,
    sheared_to_lab,
    speckle_from_atoms,
    atoms_to_object,
    atom_sum_cost,
)
from .bcdi_io import BCDIData, list_datasets, load_bcdi
from .coherent import bcdi_forward, coherent_speckle, phase_retrieval
from .debye import (
    atomic_form_factors,
    coherent_amplitude,
    coherent_intensity,
    debye_intensity,
    ensemble_intensity,
)
from .disorder import (
    AnisotropicMSD,
    TransientMSD,
    dwf_amplitude,
    msd_tensor_from_frames,
)
from .dynamics import (
    msd_from_stiffness,
    recover_stiffness,
    stiffness_from_msd,
    thermal_ensemble,
)
from .energy import make_spectrum, monochromatic
from .instrument import (
    add_poisson_noise,
    poisson_nll,
    project_to_detector,
    resolution_convolve,
    solid_angle_polarization,
)
from .inverse import cosine_loss, fit, laplace_uncertainty, relative_l2_loss
from .ml import ParameterMLP, make_dataset, train_surrogate
from .phonon import (
    apply_out_of_plane_strain,
    bragg_timeseries,
    fit_coherent_phonon,
    strain_wave,
)
from .realdata import debye_reference_numpy, load_profile
from .active_learning import (
    fisher_information,
    next_best_measurement,
    rank_measurements,
    sensitivity,
)
from .ensemble import polydisperse_rod, recover_thickness_distribution
from .latent_dynamics import discover_eom, integrate_latent_ode, library_terms
from .md_integrator import (
    bragg_from_trajectory,
    coherent_mode_kick,
    harmonic_force,
    recover_potential_from_movie,
    velocity_verlet,
)
from .multimodal import (
    carrier_density,
    fit_multimodal,
    lattice_temperature_from_carriers,
    optical_signal,
    strain_two_channel,
    xray_only_degeneracy,
)
from .strain_profile import (
    acoustic_pulse,
    apply_depth_displacement,
    depth_resolved_amplitude,
    depth_resolved_intensity,
    exponential_strain,
    linear_strain,
    recover_depth_strain,
    strain_to_displacement,
    temperature_to_displacement,
    temperature_to_msd,
    thermal_rod,
)
from .thermal_transport import (
    fit_electron_phonon_coupling,
    fit_thermal_diffusivity,
    heat_diffusion_1d,
    lattice_T_to_intensity_ratio,
    two_temperature_model,
)
from .rocking import (
    analytic_rod_model,
    fwhm,
    md_rod_model,
    reciprocal_space_map,
    rocking_curve,
    thickness_from_fwhm,
    thickness_loss_scan,
)
from .forward import rod_intensity, structure_factor_q
from .io import (
    build_crystal_tensor,
    cdse_supercell,
    cdse_zincblende,
    load_xyz_frames,
    supercell_from_crystal,
)
from .shape_factor import (
    interference_factor,
    laue_interference_1d,
    nanoplatelet_rod,
    truncation_rod,
)

__version__ = "0.3.3"

__all__ = [
    "interference_factor",
    "laue_interference_1d",
    "nanoplatelet_rod",
    "truncation_rod",
    "rod_intensity",
    "structure_factor_q",
    "cdse_zincblende",
    "build_crystal_tensor",
    "cdse_supercell",
    "supercell_from_crystal",
    "load_xyz_frames",
    "atomic_form_factors",
    "coherent_amplitude",
    "coherent_intensity",
    "debye_intensity",
    "ensemble_intensity",
    "AnisotropicMSD",
    "TransientMSD",
    "dwf_amplitude",
    "msd_tensor_from_frames",
    "analytic_rod_model",
    "md_rod_model",
    "rocking_curve",
    "fwhm",
    "thickness_from_fwhm",
    "reciprocal_space_map",
    "thickness_loss_scan",
    "fit",
    "laplace_uncertainty",
    "relative_l2_loss",
    "cosine_loss",
    "coherent_speckle",
    "bcdi_forward",
    "phase_retrieval",
    # BCDI geometry (q-space basis and its real-space conjugate)
    "bragg_geometry",
    "q_basis",
    "conjugate_real_basis",
    "oversampling",
    "shear_angles_deg",
    "detector_distance_for_oversampling",
    "rocking_step_for_oversampling",
    "sheared_to_lab",
    # BCDI forward chain (differentiable) + external-data ingest
    "q_grid",
    "rotation_to_bragg",
    "object_to_amplitude",
    "detector_signal",
    "sample_counts",
    "speckle_from_atoms",
    "atoms_to_object",
    "atom_sum_cost",
    "BCDIData",
    "load_bcdi",
    "list_datasets",
    "stiffness_from_msd",
    "msd_from_stiffness",
    "thermal_ensemble",
    "recover_stiffness",
    "strain_wave",
    "apply_out_of_plane_strain",
    "bragg_timeseries",
    "fit_coherent_phonon",
    "make_dataset",
    "ParameterMLP",
    "train_surrogate",
    "project_to_detector",
    "solid_angle_polarization",
    "poisson_nll",
    "add_poisson_noise",
    "resolution_convolve",
    "load_profile",
    "debye_reference_numpy",
    "depth_resolved_amplitude",
    "depth_resolved_intensity",
    "apply_depth_displacement",
    "linear_strain",
    "exponential_strain",
    "acoustic_pulse",
    "strain_to_displacement",
    "temperature_to_displacement",
    "temperature_to_msd",
    "thermal_rod",
    "recover_depth_strain",
    "two_temperature_model",
    "lattice_T_to_intensity_ratio",
    "fit_electron_phonon_coupling",
    "heat_diffusion_1d",
    "fit_thermal_diffusivity",
    "harmonic_force",
    "velocity_verlet",
    "bragg_from_trajectory",
    "coherent_mode_kick",
    "recover_potential_from_movie",
    "carrier_density",
    "lattice_temperature_from_carriers",
    "strain_two_channel",
    "optical_signal",
    "fit_multimodal",
    "xray_only_degeneracy",
    "library_terms",
    "integrate_latent_ode",
    "discover_eom",
    "polydisperse_rod",
    "recover_thickness_distribution",
    "sensitivity",
    "fisher_information",
    "rank_measurements",
    "next_best_measurement",
    "make_spectrum",
    "monochromatic",
]
