"""midas-pdf — differentiable, error-propagating total-scattering / G(r) pipeline.

Thin Faber-Ziman normalization layer over ``midas-hkls`` (form factors) and
``midas-integrate-v2`` (integration, Compton, sine FT with σ propagation).

Typical use::

    from midas_pdf import Composition, i_of_q_to_Gr, delta_pdf
    comp = Composition({"Si": 1, "O": 2})
    G, sigma_G, S = i_of_q_to_Gr(q, I_q, comp, r, wavelength_A=0.1665,
                                 sigma_intensity=sigma_I, q_max=22.0)
"""
# macOS (and some Linux setups) can hit an OpenMP-duplicate abort at import
# time because torch and numpy each ship their own libomp copy.  We set the
# env var *before* torch loads so users get a clean import instead of a
# cryptic runtime crash on their first line.  Users who prefer to control
# it themselves can already have set it, in which case setdefault is a no-op.
import os as _os
_os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from .composition import Composition
from .compton import breit_dirac_factor, incoherent_scattering
from .conventions import (
    pair_distribution_g,
    radial_distribution_R,
    structure_function_F,
    total_correlation_T,
)
from .corrections import (
    apply_detector_efficiency,
    detector_efficiency,
    flat_plate_transmission,
    linear_attenuation_um,
)
from .deltapdf import delta_pdf, significant_mask
from .fluorescence import expected_fluorescence
from .frontend import image_to_Gr, image_to_iq
from .gr import G_of_r, R_px_to_Q, estimate_background, fourier_sine_transform
from .cross_section import differential_cross_section, total_cross_section
from .ms import (
    CYLINDER_SLAB_FACTOR,
    cylinder_effective_tau,
    ms_background_on_grid,
    multiple_scattering_mc,
    multiple_scattering_mc_cylinder,
    slab_double_scattering,
    slab_optical_params,
    slab_single_scattering_factor,
)
from .ms_transport import slab_transport_ms
from .multiple_scattering import lumped_background, polynomial_basis
from .structure import build_pair_list, pdffit_gr, refine_structure
from .normalize import faber_ziman_S
from .pipeline import i_of_q_to_Gr
from .refine import RefineResult, refine_normalization
from .validate import (
    debye_scattering_intensity,
    interatomic_distances,
    synthetic_powder_image,
)

__version__ = "0.1.2"

__all__ = [
    # core
    "Composition",
    "faber_ziman_S",
    "i_of_q_to_Gr",
    "fourier_sine_transform",
    "G_of_r",
    "R_px_to_Q",
    "estimate_background",
    # front-end (pixels → G(r))
    "image_to_iq",
    "image_to_Gr",
    # convention / output-function family
    "structure_function_F",
    "pair_distribution_g",
    "total_correlation_T",
    "radial_distribution_R",
    # corrections (Hubbell Compton, detector efficiency, absorption, fluorescence)
    "incoherent_scattering",
    "breit_dirac_factor",
    "detector_efficiency",
    "apply_detector_efficiency",
    "flat_plate_transmission",
    "linear_attenuation_um",
    "expected_fluorescence",
    # multiple scattering (Tier-1 lumped smooth background)
    "lumped_background",
    "polynomial_basis",
    # multiple scattering (Tier-2/3: cross-section + first-principles estimators)
    "differential_cross_section",
    "total_cross_section",
    "slab_single_scattering_factor",
    "slab_optical_params",
    "multiple_scattering_mc",
    "slab_double_scattering",
    "slab_transport_ms",
    "multiple_scattering_mc_cylinder",
    "cylinder_effective_tau",
    "CYLINDER_SLAB_FACTOR",
    "ms_background_on_grid",
    # differentiable refinement
    "refine_normalization",
    "RefineResult",
    # small-box structure modelling (PDFfit-style, error-aware)
    "pdffit_gr",
    "build_pair_list",
    "refine_structure",
    # Δ-PDF
    "delta_pdf",
    "significant_mask",
    # model-free references / forward model
    "debye_scattering_intensity",
    "interatomic_distances",
    "synthetic_powder_image",
]
