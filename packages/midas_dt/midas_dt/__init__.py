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

__version__ = "0.5.0"

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
from .azimuthal import (
    RingExtraction,
    area_and_centroid,
    azimuthal_rebin,
    background_from_ring_free,
    radial_half_correlation,
    count_maxima,
    extract_ring,
    mad_filter,
    ring_free_mask,
    ring_windows,
    snr_per_eta,
    strain_from_centroid,
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
    # per-azimuth ring extraction (area -> texture, centroid -> strain)
    "RingExtraction",
    "area_and_centroid",
    "azimuthal_rebin",
    "background_from_ring_free",
    "radial_half_correlation",
    "count_maxima",
    "extract_ring",
    "mad_filter",
    "ring_free_mask",
    "ring_windows",
    "snr_per_eta",
    "strain_from_centroid",
    # texture / ODF -- requires midas-dt[texture], imported lazily (see below)
    "CubicGSH",
    "LadderResult",
    "SymGSH",
    "UniaxialODFModel",
    "cubic_rotations",
    "explained_by_polynomial",
    "fibre_cos_theta",
    "fit_uniaxial_ladder",
    "hermans_parameter",
    "hkl_family",
    "invariant_basis",
    "kappa_for_halfwidth",
    "kernel_to_gsh",
    "legendre_even",
    "radial_coeffs",
    "sample_kernel",
    "uniaxial_design",
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


# ------------------------------------------------------------------ texture
# The ODF / texture modules are imported LAZILY, on first attribute access.
#
# They need floors above this package's core ones -- `scipy.special.sph_harm_y`
# arrived in scipy 1.15 and `numpy.trapezoid` in numpy 2.0 -- and symmetry for
# anything but cubic comes from the optional `midas-hkls`. Importing them eagerly
# would make `import midas_dt` fail on an environment that is perfectly capable of
# running every reconstruction path, which is the overwhelmingly common case.
#
# Install with: pip install midas-dt[texture]
_TEXTURE_MODULES = {
    "gsh": ("CubicGSH", "SymGSH", "cubic_rotations", "hkl_family",
            "invariant_basis", "sph_harm_vec", "wigner_D"),
    "texture_kernel": ("halfwidth_deg", "kappa_for_halfwidth", "kernel_profile",
                       "kernel_to_gsh", "radial_coeffs", "sample_kernel",
                       "sample_kernel_angles"),
    "odf_uniaxial": ("LadderResult", "UniaxialODFModel",
                     "explained_by_polynomial", "fibre_cos_theta",
                     "fit_uniaxial_ladder", "hermans_parameter",
                     "legendre_even", "normalisation_c_l", "uniaxial_design"),
}
_TEXTURE_OWNER = {name: mod for mod, names in _TEXTURE_MODULES.items()
                  for name in names}


def __getattr__(name: str):
    """PEP 562 lazy access to the texture symbols."""
    mod_name = _TEXTURE_OWNER.get(name)
    if mod_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib
    try:
        mod = importlib.import_module(f".{mod_name}", __name__)
    except ImportError as exc:
        raise ImportError(
            f"midas_dt.{name} needs the texture extra: "
            f"pip install 'midas-dt[texture]'  (scipy>=1.15 for sph_harm_y, "
            f"numpy>=2.0 for trapezoid, midas-hkls for non-cubic symmetry). "
            f"Underlying error: {exc}"
        ) from exc
    value = getattr(mod, name)
    globals()[name] = value          # cache, so this runs once per symbol
    return value


def __dir__():                       # pragma: no cover - interactive convenience
    return sorted(__all__)
