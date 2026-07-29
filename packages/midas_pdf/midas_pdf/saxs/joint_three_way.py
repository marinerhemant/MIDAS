"""Three-way simultaneous SAXS + SANS + PDF refinement.

Small-angle NEUTRON scattering (SANS) is physically identical to SAXS
except for the contrast: X-rays scatter off the electron cloud with
effective density $\\rho_e Z$, while neutrons scatter off nuclei with
element-specific scattering-length density (SLD) that can be
manipulated by isotope substitution (D₂O vs H₂O being the canonical
example).

Consequences for a joint fit:

  * Both SANS and SAXS share the **particle shape** (diameter D,
    polydispersity, shell geometry, ...).
  * Each has its **own scale + background** because the absolute
    contrasts differ.
  * Deuteration ("contrast matching") lets SANS suppress one component
    of a multi-phase particle while SAXS still sees everything.
    Combining SAXS + SANS + PDF over-constrains the system and can
    disentangle otherwise-degenerate structural parameters.

Rev 11 wires SANS as a third dataset alongside the Rev 7--8 joint
SAXS+PDF pipeline. The particle diameter is shared across all three
observables; separate scale factors and backgrounds absorb the
independent instrument / contrast variables.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from .joint import sphere_characteristic_function
from .model import SAXSModel


@dataclass
class ThreeWayJointResult:
    fitted: Dict[str, float]                 = field(default_factory=dict)
    uncertainty: Dict[str, float]            = field(default_factory=dict)
    G_calc: Optional[torch.Tensor]           = None
    I_saxs_calc: Optional[torch.Tensor]      = None
    I_sans_calc: Optional[torch.Tensor]      = None
    chi2_pdf: float                          = float("nan")
    chi2_saxs: float                         = float("nan")
    chi2_sans: float                         = float("nan")
    chi2_total: float                        = float("nan")
    loss_history: List[float]                = field(default_factory=list)


def joint_refine_three_way(
    *,
    # PDF
    crystal_tensor,
    r_pdf: torch.Tensor,
    G_obs: torch.Tensor,
    pairs,
    sigma_G: Optional[torch.Tensor] = None,
    # SAXS
    q_saxs: torch.Tensor,
    I_saxs: torch.Tensor,
    sigma_I_saxs: Optional[torch.Tensor] = None,
    saxs_model: Optional[SAXSModel] = None,
    # SANS
    q_sans: torch.Tensor,
    I_sans: torch.Tensor,
    sigma_I_sans: Optional[torch.Tensor] = None,
    sans_model: Optional[SAXSModel] = None,
    # Refinable inits
    init_a: Optional[float] = None,
    init_u_iso: float = 0.006,
    init_scale_pdf: float = 1.0,
    init_diameter_A: float = 100.0,
    init_scale_saxs: float = 1.0,
    init_background_saxs: float = 0.0,
    init_scale_sans: float = 1.0,
    init_background_sans: float = 0.0,
    # Fit hyper-parameters
    weights: Tuple[float, float, float] = (10.0, 10.0, 1.0),   # (SAXS, SANS, PDF)
    n_steps: int = 100,
    lr: float = 0.5,
) -> ThreeWayJointResult:
    """Joint MAP + Laplace fit of PDF + SAXS + SANS with a **shared
    particle diameter**.

    Uses the same finite-size damping γ(r, D) as the two-way fit; the
    SANS forward model reuses SAXSModel (form factor is
    contrast-agnostic) with its own scale + background.
    """
    from midas_pdf.structure import pdffit_gr
    if saxs_model is None:
        saxs_model = SAXSModel(shape="sphere", polydispersity=0.05)
    if sans_model is None:
        sans_model = SAXSModel(shape=saxs_model.shape,
                                polydispersity=saxs_model.polydispersity,
                                n_poly_nodes=saxs_model.n_poly_nodes,
                                S_Q_model=saxs_model.S_Q_model)

    r_t = torch.as_tensor(r_pdf, dtype=torch.float64)
    G_t = torch.as_tensor(G_obs, dtype=torch.float64)
    w_G = (torch.ones_like(G_t) if sigma_G is None
           else 1.0 / torch.as_tensor(sigma_G, dtype=torch.float64).clamp(min=1e-12) ** 2)
    q_saxs_t = torch.as_tensor(q_saxs, dtype=torch.float64)
    I_saxs_t = torch.as_tensor(I_saxs, dtype=torch.float64)
    w_saxs = (torch.ones_like(I_saxs_t) if sigma_I_saxs is None
              else 1.0 / torch.as_tensor(sigma_I_saxs, dtype=torch.float64).clamp(min=1e-12) ** 2)
    q_sans_t = torch.as_tensor(q_sans, dtype=torch.float64)
    I_sans_t = torch.as_tensor(I_sans, dtype=torch.float64)
    w_sans = (torch.ones_like(I_sans_t) if sigma_I_sans is None
              else 1.0 / torch.as_tensor(sigma_I_sans, dtype=torch.float64).clamp(min=1e-12) ** 2)

    lat0 = crystal_tensor.lattice_params.detach()
    angles = lat0[3:]
    a0 = float(lat0[0]) if init_a is None else float(init_a)

    theta = torch.tensor(
        [a0, init_u_iso, init_scale_pdf, init_diameter_A,
         init_scale_saxs, init_background_saxs,
         init_scale_sans, init_background_sans],
        dtype=torch.float64, requires_grad=True)
    names = ["a", "u_iso", "scale_pdf", "diameter_A",
             "scale_saxs", "background_saxs",
             "scale_sans", "background_sans"]

    w_saxs_channel, w_sans_channel, w_pdf_channel = weights

    def model_G(th):
        lp = torch.cat([th[0].reshape(1).expand(3), angles])
        G_bulk = pdffit_gr(crystal_tensor, r_t, pairs,
                            scale=th[2], u_iso=th[1], lattice_params=lp)
        return G_bulk * sphere_characteristic_function(r_t, th[3])

    def model_I_saxs(th):
        return saxs_model.I(q_saxs_t, D_median=th[3] / 2.0,
                             scale=th[4], background=th[5])

    def model_I_sans(th):
        return sans_model.I(q_sans_t, D_median=th[3] / 2.0,
                             scale=th[6], background=th[7])

    def chi2(th):
        G_calc = model_G(th)
        I_s = model_I_saxs(th)
        I_n = model_I_sans(th)
        c_pdf  = (w_G * (G_t - G_calc) ** 2).sum()
        c_saxs = (w_saxs * (I_saxs_t - I_s) ** 2).sum()
        c_sans = (w_sans * (I_sans_t - I_n) ** 2).sum()
        return (w_pdf_channel * c_pdf
                + w_saxs_channel * c_saxs
                + w_sans_channel * c_sans), c_pdf, c_saxs, c_sans

    opt = torch.optim.LBFGS([theta], lr=lr, max_iter=n_steps,
                              tolerance_grad=1e-8, tolerance_change=1e-12,
                              line_search_fn="strong_wolfe")
    history: List[float] = []

    def closure():
        opt.zero_grad()
        loss, *_ = chi2(theta)
        loss.backward()
        history.append(float(loss.detach()))
        return loss

    opt.step(closure)
    loss_final, c_pdf, c_saxs, c_sans = chi2(theta)

    uncert: Dict[str, float] = {k: float("nan") for k in names}
    try:
        th_leaf = theta.detach().clone().requires_grad_(True)
        H = torch.autograd.functional.hessian(
            lambda t: chi2(t)[0], th_leaf)
        cov = 2.0 * torch.linalg.inv(H)
        sig = torch.sqrt(torch.diagonal(cov).clamp(min=0.0))
        uncert = {k: float(sig[i]) for i, k in enumerate(names)}
    except Exception:
        pass

    with torch.no_grad():
        G_calc = model_G(theta)
        I_saxs_calc = model_I_saxs(theta)
        I_sans_calc = model_I_sans(theta)

    return ThreeWayJointResult(
        fitted={k: float(theta[i].detach()) for i, k in enumerate(names)},
        uncertainty=uncert,
        G_calc=G_calc.detach(),
        I_saxs_calc=I_saxs_calc.detach(),
        I_sans_calc=I_sans_calc.detach(),
        chi2_pdf=float(c_pdf.detach()),
        chi2_saxs=float(c_saxs.detach()),
        chi2_sans=float(c_sans.detach()),
        chi2_total=float(loss_final.detach()),
        loss_history=history,
    )


__all__ = ["ThreeWayJointResult", "joint_refine_three_way"]
