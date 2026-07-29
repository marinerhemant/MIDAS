"""midas-pink: pink-beam HEDM with spectrum-aware differentiable inversion.

Extends ``midas-diffract`` from monochromatic to arbitrary illumination
spectra by integrating ``S(E)`` as a discrete weighted sum over per-energy
mono forward evaluations. Unifies mono, pink, and white HEDM under a
single differentiable computational graph with no change to the loss,
the optimiser, or the parameterisation.

See the companion paper for methodology:

    Sharma, Andrejevic, Cherukara, "Pink-Beam High-Energy Diffraction
    Microscopy via a Differentiable Forward Model with Spectrum-Aware
    Inversion," IUCrJ (submitted, 2026).

Quick start
-----------
    import torch
    import midas_diffract as md
    import midas_pink as mp

    # 1. Build a parameterised spectrum (fixed Gaussian for known-S(E))
    spec = mp.ParameterisedSpectrum(
        E0_keV=71.6764, half_bw=0.03, n_samples=51,
        init_kind="gaussian", init_rel_bw=1e-2, fixed=True,
    )

    # 2. Build the per-energy mono model bank from a geometry factory
    bank = mp.build_pink_bank(
        spec, space_group=sg, lattice=lat,
        geom_factory=lambda lam_A: md.HEDMGeometry(...),
        two_theta_max_deg=8.0,
    )

    # 3. Pick ROIs centred on each spot, splat observation + prediction
    plan = mp.plan_rois_from_state(bank, euler.unsqueeze(0), pos.unsqueeze(0),
                                    lattice_params=latc, roi_h=31, roi_w=31)
    rois = mp.splat_rois(bank, plan, euler.unsqueeze(0), pos.unsqueeze(0),
                         latc, sigma_psf_px=1.5)

    # 4. Recover grain state (orientation, lattice, optionally mosaicity)
    result = mp.recover_grain_state(bank, plan, observed_rois,
                                    init_euler, init_pos, init_lattice,
                                    cfg=mp.RecoveryConfig(...))
"""

__version__ = "0.1.0"

from .spectrum import (
    ParameterisedSpectrum,
    H_C_KEV_A,
)
from .inverse import (
    PinkBank,
    RoiPlan,
    RecoveryConfig,
    build_pink_bank,
    pink_forward_stacked,
    plan_rois_from_state,
    splat_rois,
    splat_rois_3d,
    roi_l2_loss,
    compute_centroids,
    centroid_loss,
    recover_grain_state,
    recover_two_stage,
    recover_joint,
    fit_spectrum_to_rois,
)

__all__ = [
    # spectrum
    "ParameterisedSpectrum",
    "H_C_KEV_A",
    # inverse
    "PinkBank",
    "RoiPlan",
    "RecoveryConfig",
    "build_pink_bank",
    "pink_forward_stacked",
    "plan_rois_from_state",
    "splat_rois",
    "splat_rois_3d",
    "roi_l2_loss",
    "compute_centroids",
    "centroid_loss",
    "recover_grain_state",
    "recover_two_stage",
    "recover_joint",
    "fit_spectrum_to_rois",
    # version
    "__version__",
]
