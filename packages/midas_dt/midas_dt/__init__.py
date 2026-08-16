"""midas-dt: diffraction / X-ray computed tomography (XRD-CT).

Takes a scan of detector frames over (translation, rotation) to per-voxel
diffraction patterns and the maps derived from them.

**Scope.** Powder-like XRD-CT: continuous rings, integrated azimuthally. If
the rings break into discrete spots the sample is coarse-grained and this is
the wrong tool -- that is scanning-3DXRD, handled by ``midas_index``'s PF mode
and ``midas_pf_odf``. The dividing line is operational: continuous at the
working (R, eta) bin size, or not.
"""

from __future__ import annotations

__version__ = "0.4.3"

from .channels import Channel, channels_from_legacy_params
from .conventions import (
    ADDITIVE_FIT_OUTPUTS,
    FIT_OUTPUT_NAMES,
    RECON_SIGN,
    aps_1id_omega,
    fit_output_index,
    is_additive,
    recon_size,
    unsnake,
)
from .reduce import FrameReducer, ReducedFrame, poisson_variance
from .branches import (
    BranchResult,
    compare,
    format_comparison,
    run_fit_then_recon,
    run_recon_then_fit,
)
from .absorption import (
    attenuated_projection_matrix,
    attenuation_factors,
    correct_reconstruction,
    mu_from_transmission,
    uniform_mu,
)
from .direct import DirectResult, laplace_sigma, run_direct
from .tensor_strain import (
    COMPONENT_NAMES,
    DeviatoricStrain,
    TensorResult,
    deviatoric_design,
    fit_tensor_strain,
    q_hat_sample_frame,
    strain_to_radius,
)
from .iterative import sirt, tv_reconstruct
from .index_rings import (
    ALPHA_U3O8,
    CEO2,
    IndexResult,
    PhaseCandidate,
    RingMatch,
    index_rings,
)
from .io import read_legacy_reconstruction, write_maps_hdf5, write_result
from .maps import (
    StrainMap,
    d_spacing_map,
    phase_fraction_map,
    radius_to_d_spacing,
    radius_to_two_theta,
    strain_map,
)
from .peakfit import LineoutFit, fit_lineout
from .center import CentreResult, centre_of_mass_shift, find_centre
from .recon import Reconstruction, reconstruct
from .sinogram import SinogramStack, assemble
from .geometry import (
    DTGeometry,
    from_calibration,
    geometry_from_legacy_params,
    parse_legacy_params,
    spec_from_calibration,
)
from .rings import Ring, find_rings, rolling_baseline
from .scan import PILATUS_1475x1679, DTScan, RawFormat, detect_snake, frames_in_file

__all__ = [
    "__version__",
    "ADDITIVE_FIT_OUTPUTS",
    "ALPHA_U3O8",
    "BranchResult",
    "CEO2",
    "DirectResult",
    "IndexResult",
    "PhaseCandidate",
    "RingMatch",
    "CentreResult",
    "LineoutFit",
    "StrainMap",
    "Channel",
    "DTGeometry",
    "DTScan",
    "FIT_OUTPUT_NAMES",
    "PILATUS_1475x1679",
    "RECON_SIGN",
    "RawFormat",
    "Ring",
    "FrameReducer",
    "ReducedFrame",
    "Reconstruction",
    "SinogramStack",
    "aps_1id_omega",
    "assemble",
    "centre_of_mass_shift",
    "channels_from_legacy_params",
    "compare",
    "find_centre",
    "find_rings",
    "d_spacing_map",
    "read_legacy_reconstruction",
    "write_maps_hdf5",
    "write_result",
    "fit_lineout",
    "attenuated_projection_matrix",
    "attenuation_factors",
    "correct_reconstruction",
    "index_rings",
    "laplace_sigma",
    "mu_from_transmission",
    "sirt",
    "tv_reconstruct",
    "uniform_mu",
    "phase_fraction_map",
    "radius_to_d_spacing",
    "radius_to_two_theta",
    "strain_map",
    "format_comparison",
    "run_direct",
    # tensor strain (deviatoric, direct inversion)
    "DeviatoricStrain",
    "TensorResult",
    "COMPONENT_NAMES",
    "fit_tensor_strain",
    "q_hat_sample_frame",
    "deviatoric_design",
    "strain_to_radius",
    "run_fit_then_recon",
    "run_recon_then_fit",
    "detect_snake",
    "fit_output_index",
    "frames_in_file",
    "from_calibration",
    "geometry_from_legacy_params",
    "is_additive",
    "parse_legacy_params",
    "spec_from_calibration",
    "poisson_variance",
    "recon_size",
    "rolling_baseline",
    "reconstruct",
    "unsnake",
]
