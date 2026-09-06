"""midas-hkls — pure-Python crystallography & HKL list generator.

sginfo-equivalent (Ralf W. Grosse-Kunstleve, 1994-96) via Hall-symbol parsing.

Base API (always available)::

    from midas_hkls import SpaceGroup, Lattice, generate_hkls, Reflection
    from midas_hkls import Atom, Crystal
    from midas_hkls import form_factor

Differentiable structure factors (requires ``midas-hkls[torch]``)::

    from midas_hkls import structure_factors, powder_intensity

CIF I/O (requires ``midas-hkls[cif]``)::

    from midas_hkls import read_cif, write_cif
"""
from .crystal import Atom, B_to_U, Crystal, U_to_B, parse_phase_atoms
from .form_factors import (
    available_elements,
    coefficients,
    form_factor,
    form_factor_batch,
    register_ion,
    registered_ions,
)
from .hkl_gen import Reflection, generate_hkls, reflections_to_dataframe
from .param_basis import read_phase_basis
from .lattice import (
    Lattice, PowderLatticeFit, refine_lattice_from_d_spacings,
)
from .nf_hkls import emit_nf_hkls_csv, write_nf_hkls_csv
from .point_group import (
    LAUE_TO_PROPER_GROUP,
    laue_class,
    plane_normal,
    plane_normals,
    point_group_rotations,
    proper_group_symbol,
    proper_rotations_from_space_group,
)
from .distortion_mode import (
    SubcellSetting, supercell_to_subcell, subcell_to_supercell,
    supercell_hkl_to_subcell, splits_under, diagonal_B_can_express,
    can_distinguish_modes,
)
from .phase_id import (
    PhaseCandidate, PhaseMatch, candidate_d_lines, worst_relative_residual,
    global_minimax_scale, chance_worst_residual, identify_phase,
)
from .ub_refine import (
    UBFit, refine_ub_from_gvectors, ub_to_cell, ub_to_u_b,
    cell_from_metric, drlv2,
)
from .cell_series import (
    CellDeformation, TransitionVerdict, cell_deformation, explained_fraction,
    track_cell, detect_transition, poisson_upper_p, min_detectable_excess,
)
from .ab_initio import (
    LatticeCandidate, AbInitioResult, index_ab_initio, patterson,
    chance_score, score_vector, refine_vector, find_candidate_vectors,
    select_basis, reduce_basis,
)
from .conventional import (
    ConventionalCell, metric_symmetry, to_conventional,
    to_conventional_from_fit,
)
from .ab_splitting import (
    hkl_box_from_geometry, distortion_rank, distortion_condition,
    ab_separable, shear_separable, partner_multiplicity, index_asymmetry,
)
from .cell_constrained import (
    FREE_PARAMS, ConstrainedFit, DomainData,
    refine_cell_constrained, refine_cell_joint, split_with_error,
)
from .lattice_symmetry import (
    Holohedry, holohedry, lattice_symmetry_operations,
    tolerance_from_fit, holohedry_from_fit,
)
from .niggli import NiggliCell, niggli_reduce, same_lattice
from .space_group import SpaceGroup, list_space_groups
from .symops import SymOp

__version__ = "0.10.0"

__all__ = [
    "hkl_box_from_geometry", "distortion_rank", "distortion_condition",
    "ab_separable", "shear_separable", "partner_multiplicity",
    "index_asymmetry",
    "FREE_PARAMS", "ConstrainedFit", "DomainData",
    "refine_cell_constrained", "refine_cell_joint", "split_with_error",
    "Holohedry", "holohedry", "lattice_symmetry_operations",
    "tolerance_from_fit", "holohedry_from_fit",
    "NiggliCell", "niggli_reduce", "same_lattice",
    "ConventionalCell", "metric_symmetry", "to_conventional",
    "to_conventional_from_fit",
    "LatticeCandidate", "AbInitioResult", "index_ab_initio", "patterson",
    "chance_score", "score_vector", "refine_vector", "find_candidate_vectors",
    "select_basis", "reduce_basis",
    "CellDeformation", "TransitionVerdict", "cell_deformation",
    "explained_fraction", "track_cell", "detect_transition",
    "poisson_upper_p", "min_detectable_excess",
    "UBFit", "refine_ub_from_gvectors", "ub_to_cell", "ub_to_u_b",
    "cell_from_metric", "drlv2",
    "PhaseCandidate", "PhaseMatch", "candidate_d_lines",
    "worst_relative_residual", "global_minimax_scale",
    "chance_worst_residual", "identify_phase",
    "SubcellSetting", "supercell_to_subcell", "subcell_to_supercell",
    "supercell_hkl_to_subcell", "splits_under", "diagonal_B_can_express",
    "can_distinguish_modes",
    "CLASSICAL_ELECTRON_RADIUS_A",
    "extinction_length_um",
    "kinematical_path_limit_um",
    "primary_extinction_factor",
    "refraction_shift_deg",
    "susceptibility_chi0",
    "Atom",
    "parse_phase_atoms",
    "read_phase_basis",
    "B_to_U",
    "Crystal",
    "LAUE_TO_PROPER_GROUP",
    "Lattice",
    "PowderLatticeFit",
    "refine_lattice_from_d_spacings",
    "Reflection",
    "SpaceGroup",
    "SymOp",
    "U_to_B",
    "available_elements",
    "coefficients",
    "register_ion",
    "registered_ions",
    "emit_nf_hkls_csv",
    "form_factor",
    "form_factor_batch",
    "generate_hkls",
    "laue_class",
    "list_space_groups",
    "plane_normal",
    "plane_normals",
    "point_group_rotations",
    "proper_group_symbol",
    "proper_rotations_from_space_group",
    "reflections_to_dataframe",
    "write_nf_hkls_csv",
]


from .extinction import (
    CLASSICAL_ELECTRON_RADIUS_A,
    extinction_length_um,
    kinematical_path_limit_um,
    primary_extinction_factor,
    refraction_shift_deg,
)


def __getattr__(name: str):  # pragma: no cover - lazy attribute access
    """Lazily import torch / CIF helpers so the base install stays light."""
    if name == "susceptibility_chi0":
        from .extinction import susceptibility_chi0
        return susceptibility_chi0
    if name in {"structure_factors", "structure_factor"}:
        from .structure_factor import structure_factors
        return structure_factors
    if name == "structure_factor_intensity":
        from .structure_factor import structure_factor_intensity
        return structure_factor_intensity
    if name == "f2_normalised":
        from .structure_factor import f2_normalised
        return f2_normalised
    if name == "powder_intensity":
        from .intensity import powder_intensity
        return powder_intensity
    if name == "intensity_from_crystal":
        from .intensity import intensity_from_crystal
        return intensity_from_crystal
    if name == "attach_intensities":
        from .intensity import attach_intensities
        return attach_intensities
    if name == "lorentz_polarization":
        from .intensity import lorentz_polarization
        return lorentz_polarization
    if name == "anomalous_correction":
        from .anomalous import anomalous_correction
        return anomalous_correction
    if name in {"wavelength_to_energy_eV", "energy_eV_to_wavelength"}:
        from . import anomalous
        return getattr(anomalous, name)
    if name in {
        "linear_absorption_coefficient",
        "mass_attenuation_coefficient",
        "element_density",
        "atomic_mass",
        "available_elements_absorption",
    }:
        from . import absorption
        return getattr(absorption, name)
    if name == "read_cif":
        from .io.cif import read_cif
        return read_cif
    if name == "write_cif":
        from .io.cif import write_cif
        return write_cif
    raise AttributeError(f"module 'midas_hkls' has no attribute {name!r}")
