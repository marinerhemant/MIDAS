"""Pluggable loss / log-likelihood / log-prior contributions.

Each module exposes a callable that maps ``(packed_params, observations) ->
scalar loss``.  The top-level objective is a sum of contributions; the
inference backend (LM/LBFGS/Adam/VI/HMC) consumes the result.
"""
from .pseudo_strain import pseudo_strain_residual, pseudo_strain_loss
from .prior import sum_log_prior
from .multi_image import multi_image_loss

__all__ = [
    "pseudo_strain_residual", "pseudo_strain_loss",
    "sum_log_prior", "multi_image_loss",
]
