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
from .crystal import Atom, B_to_U, Crystal, U_to_B
from .form_factors import (
    available_elements,
    coefficients,
    form_factor,
    form_factor_batch,
)
from .hkl_gen import Reflection, generate_hkls, reflections_to_dataframe
from .lattice import Lattice
from .nf_hkls import emit_nf_hkls_csv, write_nf_hkls_csv
from .space_group import SpaceGroup, list_space_groups
from .symops import SymOp

__version__ = "0.4.0"

__all__ = [
    "Atom",
    "B_to_U",
    "Crystal",
    "Lattice",
    "Reflection",
    "SpaceGroup",
    "SymOp",
    "U_to_B",
    "available_elements",
    "coefficients",
    "emit_nf_hkls_csv",
    "form_factor",
    "form_factor_batch",
    "generate_hkls",
    "list_space_groups",
    "reflections_to_dataframe",
    "write_nf_hkls_csv",
]


def __getattr__(name: str):  # pragma: no cover - lazy attribute access
    """Lazily import torch / CIF helpers so the base install stays light."""
    if name in {"structure_factors", "structure_factor"}:
        from .structure_factor import structure_factors
        return structure_factors
    if name == "structure_factor_intensity":
        from .structure_factor import structure_factor_intensity
        return structure_factor_intensity
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
    if name == "read_cif":
        from .io.cif import read_cif
        return read_cif
    if name == "write_cif":
        from .io.cif import write_cif
        return write_cif
    raise AttributeError(f"module 'midas_hkls' has no attribute {name!r}")
