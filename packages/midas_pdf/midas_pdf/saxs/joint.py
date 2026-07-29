"""Simultaneous SAXS + PDF refinement.

The key insight: SAXS constrains the *particle size* (via the Guinier and
finite-size Porod regime) and *inter-particle correlation* (via the
structure factor), while PDF constrains the *local atomic pair
distribution* (via peak positions and near-neighbour amplitudes). The
two are coupled through the **finite-size damping** the particle
imposes on G(r):

    G_calc(r) = G_bulk(r) · γ(r, D)

where γ(r, D) is the sphere characteristic function
γ(r, D) = 1 - 3r/(2D) + (r/D)³/2 for r ≤ D, 0 otherwise.

The joint χ² sums the SAXS and PDF residuals with configurable weights.

Rev-8 API is intentionally minimal — it wraps :class:`SAXSModel` +
:func:`midas_pdf.structure.pdffit_gr` and delivers a single MAP fit; the
Bayesian posterior over the joint parameter set uses the same Pyro
scaffolding as :mod:`midas_pdf.bayesian_refine`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from .model import SAXSModel

_FOUR_PI = 4.0 * float(np.pi)


# ---------------------------------------------------------------------------
# Finite-size damping (spherical characteristic function)
# ---------------------------------------------------------------------------

def sphere_characteristic_function(
    r: torch.Tensor | np.ndarray,
    diameter_A: float | torch.Tensor,
) -> torch.Tensor:
    """γ(r, D) = 1 − 3r/(2D) + (r/D)³/2 for 0 ≤ r ≤ D, else 0.

    Scaled with G(r) to model finite-size damping of the PDF.
    """
    r_t = torch.as_tensor(r, dtype=torch.float64)
    D_t = torch.as_tensor(diameter_A, dtype=torch.float64)
    x = r_t / D_t
    gamma = 1.0 - 1.5 * x + 0.5 * x ** 3
    return torch.where(x < 1.0, gamma, torch.zeros_like(gamma))


# ---------------------------------------------------------------------------
# Joint refinement result
# ---------------------------------------------------------------------------

@dataclass
class JointRefineResult:
    fitted: Dict[str, float]                 = field(default_factory=dict)
    uncertainty: Dict[str, float]            = field(default_factory=dict)
    G_calc: Optional[torch.Tensor]           = None
    I_saxs_calc: Optional[torch.Tensor]      = None
    chi2_pdf: float                          = float("nan")
    chi2_saxs: float                         = float("nan")
    chi2_total: float                        = float("nan")
    loss_history: List[float]                = field(default_factory=list)


# ---------------------------------------------------------------------------
# Joint refinement driver — MAP fit
# ---------------------------------------------------------------------------

def joint_refine(
    *,
    # PDF side
    crystal_tensor,
    r_pdf: torch.Tensor,
    G_obs: torch.Tensor,
    pairs,
    sigma_G: Optional[torch.Tensor] = None,
    # SAXS side
    q_saxs: torch.Tensor,
    I_saxs: torch.Tensor,
    sigma_I: Optional[torch.Tensor] = None,
    saxs_model: Optional[SAXSModel] = None,
    # Refinable inits
    init_a: Optional[float] = None,
    init_u_iso: float = 0.006,
    init_scale_pdf: float = 1.0,
    init_diameter_A: float = 100.0,
    init_scale_saxs: float = 1.0,
    init_background_saxs: float = 0.0,
    # Fit
    weights_saxs_pdf: Tuple[float, float] = (10.0, 1.0),
    n_steps: int = 200,
    lr: float = 0.05,
) -> JointRefineResult:
    """Joint SAXS + PDF refinement (MAP).

    Fits ``(a, U_iso, scale_pdf, diameter_A, scale_saxs, background_saxs)``
    against a weighted joint χ². The particle diameter is shared between
    the SAXS forward model and the PDF finite-size damping — that's the
    physical link that ties the two datasets together.
    """
    from midas_pdf.structure import pdffit_gr
    if saxs_model is None:
        saxs_model = SAXSModel(shape="sphere", polydispersity=0.15)

    r_t = torch.as_tensor(r_pdf, dtype=torch.float64)
    G_t = torch.as_tensor(G_obs, dtype=torch.float64)
    w_G = (torch.ones_like(G_t) if sigma_G is None
           else 1.0 / torch.as_tensor(sigma_G, dtype=torch.float64).clamp(min=1e-12) ** 2)
    q_t = torch.as_tensor(q_saxs, dtype=torch.float64)
    I_t = torch.as_tensor(I_saxs, dtype=torch.float64)
    w_I = (torch.ones_like(I_t) if sigma_I is None
           else 1.0 / torch.as_tensor(sigma_I, dtype=torch.float64).clamp(min=1e-12) ** 2)

    lat0 = crystal_tensor.lattice_params.detach()
    angles = lat0[3:]
    a0 = float(lat0[0]) if init_a is None else float(init_a)

    theta = torch.tensor(
        [a0, init_u_iso, init_scale_pdf, init_diameter_A,
         init_scale_saxs, init_background_saxs],
        dtype=torch.float64, requires_grad=True)
    names = ["a", "u_iso", "scale_pdf", "diameter_A",
             "scale_saxs", "background_saxs"]

    w_saxs, w_pdf = weights_saxs_pdf

    def model_G(th):
        lp = torch.cat([th[0].reshape(1).expand(3), angles])
        G_bulk = pdffit_gr(crystal_tensor, r_t, pairs,
                            scale=th[2], u_iso=th[1], lattice_params=lp)
        gamma = sphere_characteristic_function(r_t, th[3])
        return G_bulk * gamma

    def model_I(th):
        return saxs_model.I(
            q_t, D_median=th[3] / 2.0,        # convert diameter → radius
            scale=th[4], background=th[5],
        )

    def joint_chi2(th):
        G_calc = model_G(th)
        I_calc = model_I(th)
        c_pdf = (w_G * (G_t - G_calc) ** 2).sum()
        c_saxs = (w_I * (I_t - I_calc) ** 2).sum()
        return w_saxs * c_saxs + w_pdf * c_pdf, c_pdf, c_saxs

    # LBFGS with a small max_iter is the right tool for a quasi-quadratic
    # joint likelihood: Adam struggles with the mixed parameter scales
    # (a ~ 3.5 Å, u_iso ~ 0.006 Å², D ~ 100 Å); LBFGS's implicit Hessian
    # rescaling handles that natively.
    opt = torch.optim.LBFGS([theta], lr=lr, max_iter=n_steps,
                              tolerance_grad=1e-8, tolerance_change=1e-12,
                              line_search_fn="strong_wolfe")
    history: List[float] = []

    def closure():
        opt.zero_grad()
        loss, *_ = joint_chi2(theta)
        loss.backward()
        history.append(float(loss.detach()))
        return loss

    opt.step(closure)
    loss_final, c_pdf, c_saxs = joint_chi2(theta)

    # Hessian for uncertainties (may be expensive on the polydisperse
    # SAXS model; caller can request via compute_uncertainty=False on
    # future revisions).
    uncert: Dict[str, float] = {k: float("nan") for k in names}
    try:
        th_leaf = theta.detach().clone().requires_grad_(True)
        H = torch.autograd.functional.hessian(
            lambda t: joint_chi2(t)[0], th_leaf)
        cov = 2.0 * torch.linalg.inv(H)
        sig = torch.sqrt(torch.diagonal(cov).clamp(min=0.0))
        uncert = {k: float(sig[i]) for i, k in enumerate(names)}
    except Exception:
        pass                # keep NaN-filled uncert on failure

    with torch.no_grad():
        G_calc = model_G(theta)
        I_calc = model_I(theta)

    return JointRefineResult(
        fitted={k: float(theta[i].detach()) for i, k in enumerate(names)},
        uncertainty=uncert,
        G_calc=G_calc.detach(),
        I_saxs_calc=I_calc.detach(),
        chi2_pdf=float(c_pdf.detach()),
        chi2_saxs=float(c_saxs.detach()),
        chi2_total=float(loss_final.detach()),
        loss_history=history,
    )


__all__ = [
    "sphere_characteristic_function",
    "joint_refine",
    "JointRefineResult",
]
