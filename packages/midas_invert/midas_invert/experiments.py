"""HEDM / Laue experiment-design forwards on top of the generic Fisher core.

Generic D-optimal selection lives in :mod:`.design`; this module supplies the
HEDM-specific forward closures (strain-tensor determination from per-reflection
normal strain) and convenience planners.

Strain physics: a reflection with reciprocal direction ``g = (g1, g2, g3)``
measures the normal strain along ``g``::

    eps_n = g^T E g
          = g1^2 e11 + g2^2 e22 + g3^2 e33 + 2 g2 g3 e23 + 2 g1 g3 e13 + 2 g1 g2 e12

(Voigt order ``[e11, e22, e33, e23, e13, e12]``).  Recovering the full 6-component
strain therefore needs a *diverse* set of g-directions; this module picks the
D-optimal subset via :func:`.design.greedy_optimal_design`.

Folded in from the standalone ``midas_expdesign`` package (consolidation pass).
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    import torch

__all__ = ["strain_normal_coeffs", "strain_forward",
           "plan_strain_measurements", "design_logdet"]


def strain_normal_coeffs(g_hats):
    """Coefficient matrix ``C`` (M, 6) mapping the Voigt strain to per-reflection
    normal strain: ``eps_n = C @ eps6``.  ``g_hats`` (M, 3) is normalised here."""
    import torch
    g = torch.as_tensor(g_hats, dtype=torch.float64)
    g = g / torch.linalg.vector_norm(g, dim=1, keepdim=True).clamp(min=1e-12)
    g1, g2, g3 = g[:, 0], g[:, 1], g[:, 2]
    return torch.stack([g1 * g1, g2 * g2, g3 * g3,
                        2 * g2 * g3, 2 * g1 * g3, 2 * g1 * g2], dim=1)


def strain_forward(g_hats):
    """Return ``forward_fn(eps6) -> per-reflection normal strain`` for use with
    the Fisher / optimal-design helpers in :mod:`.design`."""
    C = strain_normal_coeffs(g_hats)

    def forward_fn(eps6):
        return C.to(eps6.dtype) @ eps6
    return forward_fn


def plan_strain_measurements(g_hats, k, *, sigma=1.0, prior_precision=1e-6,
                             labels=None):
    """Pick the ``k`` D-optimal reflections to determine the strain tensor."""
    import torch
    from .design import greedy_optimal_design
    eps0 = torch.zeros(6, dtype=torch.float64)
    return greedy_optimal_design(strain_forward(g_hats), eps0, k, sigma=sigma,
                                 prior_precision=prior_precision, labels=labels)


def design_logdet(g_hats, *, sigma=1.0, prior_precision=0.0):
    """Log-det of the Fisher information for a given reflection set (D-optimality
    score; higher = better-conditioned strain determination)."""
    import torch
    from .design import information_matrix
    eps0 = torch.zeros(6, dtype=torch.float64)
    F = information_matrix(strain_forward(g_hats), eps0, sigma=sigma)
    if prior_precision:
        F = F + prior_precision * torch.eye(6, dtype=F.dtype)
    return float(torch.linalg.slogdet(F)[1])
