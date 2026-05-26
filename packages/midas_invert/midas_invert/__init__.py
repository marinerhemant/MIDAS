"""midas-invert -- domain-agnostic differentiable-inversion primitives.

Shared across MIDAS (HEDM, Laue, pf-/grain-ODF, 2D/ultrafast):

    from midas_invert import fit, cosine_loss, relative_l2_loss   # gradient fitting
    from midas_invert import laplace_uncertainty                  # UQ
    from midas_invert import fisher_information, rank_measurements, next_best_measurement
    from midas_invert import mixture_deconvolution                # recover a distribution
    from midas_invert import ParameterMLP, train_surrogate        # amortised inference
    from midas_invert import discover_dynamics                    # SINDy
    from midas_invert.experiments import plan_strain_measurements # HEDM design forwards
    from midas_invert.kinetics import fit_jmak, discover_rate_law # in-situ kinetics
"""
from .design import (
    fisher_information,
    greedy_optimal_design,
    information_matrix,
    jacobian,
    next_best_measurement,
    rank_measurements,
    sensitivity,
)
from .mixture import mixture_deconvolution
from .optimize import cosine_loss, fit, relative_l2_loss
from .sindy import discover_dynamics, integrate_latent_ode, library_terms
from .surrogate import ParameterMLP, train_surrogate
from .uq import laplace_uncertainty

__version__ = "0.1.0a0"

__all__ = [
    "fit",
    "cosine_loss",
    "relative_l2_loss",
    "laplace_uncertainty",
    "sensitivity",
    "fisher_information",
    "rank_measurements",
    "next_best_measurement",
    "jacobian",
    "information_matrix",
    "greedy_optimal_design",
    "discover_dynamics",
    "integrate_latent_ode",
    "library_terms",
    "mixture_deconvolution",
    "ParameterMLP",
    "train_surrogate",
]
