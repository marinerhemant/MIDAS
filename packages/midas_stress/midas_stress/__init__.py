"""midas-stress: Crystallographic stress-strain analysis.

Provides Voigt-Mandel tensor conversions, Hooke's law with single-crystal
stiffness, orientation/misorientation math, and mechanical equilibrium
constraints for polycrystalline stress analysis.
"""

__version__ = "0.2.2"

# --- Tensor / Voigt ---
from .tensor import (
    tensor_to_voigt,
    voigt_to_tensor,
    tensor_to_voigt_engineering,
    lattice_params_to_A_matrix,
    lattice_params_to_strain,
    strain_grain_to_lab,
    strain_lab_to_grain,
    rotation_voigt_mandel,
    hydrostatic,
    deviatoric,
    von_mises,
)

# --- Hooke's law ---
from .hooke import hooke_stress

# --- Materials ---
from .materials import (
    cubic_stiffness,
    hexagonal_stiffness,
    get_stiffness,
    list_materials,
    STIFFNESS_LIBRARY,
    d0_sensitivity,
    d0_sensitivity_table,
)

# --- Equilibrium ---
from .equilibrium import (
    volume_average_stress_constraint,
    hydrostatic_deviatoric_decomposition,
    hydrostatic_deviatoric_decomposition_weighted,
    equilibrium_correction_uncertainty,
    d0_correction_strain_level,
    correct_d0,
    recover_d0,
    recover_d0_cubic_free_standing,
)

# --- Orientation / Misorientation ---
from .orientation import (
    misorientation,
    misorientation_om,
    misorientation_om_batch,
    misorientation_quat_batch,
    euler_to_orient_mat,
    euler_to_orient_mat_batch,
    orient_mat_to_quat,
    orient_mat_to_euler,
    quaternion_product,
    quat_to_orient_mat,
    fundamental_zone,
    make_symmetries,
    axis_angle_to_orient_mat,
    rodrigues_to_orient_mat,
)

# --- Frame conversions ---
from .frames import (
    R_MIDAS_TO_APS,
    R_APS_TO_MIDAS,
    lab_to_sample_rotation,
    vector_midas_to_aps,
    vector_aps_to_midas,
    orient_midas_to_aps,
    orient_aps_to_midas,
    tensor_midas_to_aps,
    tensor_aps_to_midas,
    tensor_lab_to_sample,
    grains_midas_to_sample,
)

# --- High-level pipeline ---
from .pipeline import compute_stress

# --- I/O ---
from .io import (
    read_grains,
    read_grains_csv,
    read_grains_h5,
    example_data_path,
)

# --- Plasticity / slip-system analysis ---
from .plasticity import (
    get_slip_systems,
    get_slip_systems_for_material,
    list_slip_families,
    slip_systems_to_lab,
    schmid_factor,
    resolved_shear_stress,
    dominant_slip_system,
    active_systems_from_crss,
    yield_proximity,
    taylor_factor,
    HCP_RATIOS,
)
