"""Compatibility adapters between v2 IntegrationSpec and other formats."""
from .from_v1 import spec_from_v1_params, spec_from_v1_paramstest
from .to_v1 import v1_params_from_spec
from .pyfai import (
    bc_to_poni, poni_to_bc, poni_file_to_row_col, make_pyfai_integrator,
    read_poni, poni_file_to_bc, orientation_flips, describe_orientation,
)

__all__ = [
    "spec_from_v1_params",
    "spec_from_v1_paramstest",
    "v1_params_from_spec",
    "bc_to_poni",
    "poni_to_bc",
    "poni_file_to_row_col",
    "make_pyfai_integrator",
    "read_poni",
    "poni_file_to_bc",
    "orientation_flips",
    "describe_orientation",
]
