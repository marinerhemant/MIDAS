"""SAXS model: form factor + polydispersity + optional interparticle S(Q).

Wraps the primitives in :mod:`midas_pdf.saxs.form_factors` into a
uniform :class:`SAXSModel` that predicts

    I(Q) = scale · N · P(Q) · S(Q) + background

where:
  * ``P(Q)`` is the polydispersity-averaged |F(Q)|²
    (single-particle form factor).
  * ``S(Q)`` is the interparticle structure factor (Percus-Yevick).
  * ``scale``, ``N`` and ``background`` are refinable scalars.

Polydispersity uses a lognormal distribution of the shape's primary size
parameter (radius, or the equatorial radius for ellipsoids), with fixed
width ``sigma_D`` (relative std of ln(D)).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Optional

import numpy as np
import torch

from .form_factors import (
    sphere_form_factor_squared,
    ellipsoid_form_factor_squared,
    cylinder_form_factor_squared,
    percus_yevick_S,
)

_SHAPE_FACTORY: Dict[str, Callable] = {
    "sphere":    sphere_form_factor_squared,
    "ellipsoid": ellipsoid_form_factor_squared,
    "cylinder":  cylinder_form_factor_squared,
}


# ---------------------------------------------------------------------------
# Polydispersity — lognormal on the primary size parameter
# ---------------------------------------------------------------------------

def lognormal_quadrature_nodes(
    D_median: float | torch.Tensor,
    sigma_lnD: float | torch.Tensor,
    *,
    n_nodes: int = 21,
    n_sigma: float = 4.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Discrete lognormal distribution nodes + weights over ln(D).

    Returns ``(D_nodes, weights)`` such that
    ``Σ weights == 1`` (normalised probability).
    """
    D_median = torch.as_tensor(D_median, dtype=torch.float64)
    sigma_lnD = torch.as_tensor(sigma_lnD, dtype=torch.float64)
    # nodes over ln(D) in [mu - n_sigma σ, mu + n_sigma σ], mu = ln(D_median)
    mu = torch.log(D_median)
    lo = mu - n_sigma * sigma_lnD
    hi = mu + n_sigma * sigma_lnD
    lnD = torch.linspace(float(lo.detach()), float(hi.detach()), n_nodes,
                          dtype=torch.float64)
    # unnormalised lognormal density: (1/D) · exp(-(lnD − μ)² / 2σ²)
    z = (lnD - mu) / sigma_lnD
    w = torch.exp(-0.5 * z ** 2)
    w = w / w.sum()
    D = torch.exp(lnD)
    return D, w


# ---------------------------------------------------------------------------
# SAXSModel
# ---------------------------------------------------------------------------

@dataclass
class SAXSModel:
    """Polydisperse SAXS forward model with optional hard-sphere S(Q).

    Attributes
    ----------
    shape : "sphere" | "ellipsoid" | "cylinder"
    polydispersity : if not None, treated as σ(ln D) for a lognormal
        distribution of the primary size parameter.
    S_Q_model : optional structure factor (currently only "hard_sphere_PY")
    """
    shape: str = "sphere"
    polydispersity: Optional[float] = None
    n_poly_nodes: int = 21
    S_Q_model: Optional[str] = None                  # None or "hard_sphere_PY"

    def I(
        self,
        q: torch.Tensor | np.ndarray,
        *,
        # primary size parameter (radius for sphere/cylinder, equatorial b for ellipsoid)
        D_median: float | torch.Tensor,
        aspect_ratio: Optional[float | torch.Tensor] = None,
        length_A: Optional[float | torch.Tensor] = None,
        volume_fraction: Optional[float | torch.Tensor] = None,
        scale: float | torch.Tensor = 1.0,
        background: float | torch.Tensor = 0.0,
    ) -> torch.Tensor:
        """Predict I(Q) = scale · <|F|²>(Q) · S(Q) + background.

        Parameters
        ----------
        D_median : primary size (Å); radius for sphere/cylinder,
            equatorial for ellipsoid.
        aspect_ratio : rotation-axis / equatorial for ellipsoid. Ignored
            for other shapes.
        length_A : length for cylinder. Ignored for other shapes.
        volume_fraction : if ``S_Q_model`` is set, drives the PY S(Q).
        scale, background : instrument scale + additive background.
        """
        q_t = torch.as_tensor(q, dtype=torch.float64)
        D_median_t = torch.as_tensor(D_median, dtype=torch.float64)

        # Polydispersity quadrature
        if self.polydispersity is not None and self.polydispersity > 0:
            D_nodes, w_nodes = lognormal_quadrature_nodes(
                D_median_t, self.polydispersity, n_nodes=self.n_poly_nodes)
        else:
            D_nodes = D_median_t.unsqueeze(0)
            w_nodes = torch.ones(1, dtype=torch.float64)

        # Evaluate |F|² at each node
        ffs = []
        for D_node in D_nodes:
            if self.shape == "sphere":
                F2 = sphere_form_factor_squared(q_t, D_node)
            elif self.shape == "ellipsoid":
                if aspect_ratio is None:
                    raise ValueError("ellipsoid requires aspect_ratio")
                a = D_node * torch.as_tensor(aspect_ratio, dtype=torch.float64)
                F2 = ellipsoid_form_factor_squared(q_t, a, D_node)
            elif self.shape == "cylinder":
                if length_A is None:
                    raise ValueError("cylinder requires length_A")
                F2 = cylinder_form_factor_squared(q_t, D_node,
                                                    torch.as_tensor(length_A,
                                                                     dtype=torch.float64))
            else:
                raise ValueError(f"unknown shape {self.shape!r}")
            ffs.append(F2)
        # Weighted average
        F2_avg = torch.stack(ffs, dim=0)                     # (n_nodes, nQ)
        P_of_Q = (F2_avg * w_nodes[:, None]).sum(dim=0)      # (nQ,)

        # Structure factor
        if self.S_Q_model == "hard_sphere_PY":
            if volume_fraction is None:
                raise ValueError("hard_sphere_PY needs volume_fraction")
            S = percus_yevick_S(q_t, D_median_t, volume_fraction)
        elif self.S_Q_model is None:
            S = torch.ones_like(q_t)
        else:
            raise ValueError(f"unknown S_Q_model {self.S_Q_model!r}")

        return (torch.as_tensor(scale, dtype=torch.float64)
                * P_of_Q * S
                + torch.as_tensor(background, dtype=torch.float64))


__all__ = ["SAXSModel", "lognormal_quadrature_nodes"]
