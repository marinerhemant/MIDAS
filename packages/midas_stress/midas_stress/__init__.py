"""midas-stress: Crystallographic stress-strain analysis.

Provides Voigt-Mandel tensor conversions, Hooke's law with single-crystal
stiffness, orientation/misorientation math, and mechanical equilibrium
constraints for polycrystalline stress analysis.

As of 0.6.0, the entire public API (orientation, frames, tensor,
equilibrium constraints, materials, plasticity, elastic_inverse) accepts
torch.Tensor inputs transparently and returns torch tensors on the
input's device/dtype. Existing NumPy callers see no API change. See
the `*_torch.py` test files for per-module parity contracts.

PyTorch is an OPTIONAL dependency. Every NumPy code path works without it;
torch is imported lazily and only when a tensor input or a torch-only entry
point (``fit_joint_d0_stiffness``, ``loo_influence_stages``) is actually used.
So that ``import midas_stress`` stays torch-free, submodules are loaded on first
attribute access (PEP 562) rather than eagerly here.
"""
from __future__ import annotations

# Register the HDF5 filter plugins (bitshuffle, LZ4, zstd) that Eiger / Dectris
# and ESRF files are written with.  hdf5plugin ships the plugin binaries but
# only registers them with the HDF5 library when it is imported — declaring it
# as a dependency is not enough.  Without this, reading such a dataset fails
# with "can't open directory (/usr/local/lib/plugin)".
try:  # pragma: no cover - environment-dependent
    import hdf5plugin as _hdf5plugin  # noqa: F401
except ImportError:  # HDF5 files with no plugin filter still read fine
    pass

import importlib

__version__ = "0.10.0"

# public name -> submodule it lives in. Loaded lazily on first access, so
# importing the package pulls in nothing heavy (and no torch).
_EXPORTS = {
    # tensor / Voigt
    "tensor_to_voigt": "tensor", "voigt_to_tensor": "tensor",
    "tensor_to_voigt_engineering": "tensor", "lattice_params_to_A_matrix": "tensor",
    "lattice_params_to_strain": "tensor", "strain_grain_to_lab": "tensor",
    "strain_lab_to_grain": "tensor", "rotation_voigt_mandel": "tensor",
    "hydrostatic": "tensor", "deviatoric": "tensor", "von_mises": "tensor",
    # hooke
    "hooke_stress": "hooke",
    # materials
    "cubic_stiffness": "materials", "hexagonal_stiffness": "materials",
    "get_stiffness": "materials", "list_materials": "materials",
    "STIFFNESS_LIBRARY": "materials", "d0_sensitivity": "materials",
    "d0_sensitivity_table": "materials",
    # equilibrium
    "volume_average_stress_constraint": "equilibrium",
    "hydrostatic_deviatoric_decomposition": "equilibrium",
    "hydrostatic_deviatoric_decomposition_weighted": "equilibrium",
    "equilibrium_correction_uncertainty": "equilibrium",
    "d0_correction_strain_level": "equilibrium", "correct_d0": "equilibrium",
    "recover_d0": "equilibrium", "recover_d0_cubic_free_standing": "equilibrium",
    "recover_d0_anisotropic": "equilibrium",
    # orientation
    "misorientation": "orientation", "misorientation_om": "orientation",
    "misorientation_om_batch": "orientation", "misorientation_quat_batch": "orientation",
    "euler_to_orient_mat": "orientation", "euler_to_orient_mat_batch": "orientation",
    "orient_mat_to_quat": "orientation", "orient_mat_to_euler": "orientation",
    "quaternion_product": "orientation", "quat_to_orient_mat": "orientation",
    "fundamental_zone": "orientation", "make_symmetries": "orientation",
    "axis_angle_to_orient_mat": "orientation", "rodrigues_to_orient_mat": "orientation",
    "matrix_mult_f33": "orientation",
    # diffraction
    "calc_eta_angle_all": "diffraction",
    # frames
    "R_MIDAS_TO_APS": "frames", "R_APS_TO_MIDAS": "frames",
    "lab_to_sample_rotation": "frames", "vector_midas_to_aps": "frames",
    "vector_aps_to_midas": "frames", "orient_midas_to_aps": "frames",
    "orient_aps_to_midas": "frames", "tensor_midas_to_aps": "frames",
    "tensor_aps_to_midas": "frames", "tensor_lab_to_sample": "frames",
    "grains_midas_to_sample": "frames",
    "TOMO_IN_PLANE": "frames", "tomo_grid_to_midas": "frames",
    "midas_to_tomo_grid": "frames", "tomo_slice_for_z": "frames",
    # pipeline
    "compute_stress": "pipeline",
    # io
    "read_grains": "io", "read_grains_csv": "io", "read_grains_h5": "io",
    "example_data_path": "io",
    # elastic-constant inverse (NumPy)
    "fit_single_crystal_stiffness": "elastic_inverse",
    "symmetry_parameterisation": "elastic_inverse",
    "stiffness_from_cij": "elastic_inverse", "build_stage_matrix": "elastic_inverse",
    "build_stage_matrix_voigt": "elastic_inverse",
    "build_stage_matrix_reuss": "elastic_inverse",
    # elastic-constant inverse (torch-only: pulls torch on access)
    "fit_joint_d0_stiffness": "elastic_inverse_torch",
    "loo_influence_stages": "elastic_inverse_torch",
    # plasticity
    "get_slip_systems": "plasticity", "get_slip_systems_for_material": "plasticity",
    "list_slip_families": "plasticity", "slip_systems_to_lab": "plasticity",
    "schmid_factor": "plasticity", "resolved_shear_stress": "plasticity",
    "dominant_slip_system": "plasticity", "active_systems_from_crss": "plasticity",
    "yield_proximity": "plasticity", "taylor_factor": "plasticity",
    "HCP_RATIOS": "plasticity",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str):
    """PEP 562 lazy attribute access — import the owning submodule on demand."""
    mod = _EXPORTS.get(name)
    if mod is None:
        raise AttributeError(f"module 'midas_stress' has no attribute {name!r}")
    value = getattr(importlib.import_module(f".{mod}", __name__), name)
    globals()[name] = value          # cache so subsequent access is direct
    return value


def __dir__():
    return sorted(set(globals()) | set(_EXPORTS))
