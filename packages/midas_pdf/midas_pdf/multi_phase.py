"""Multi-phase and core-shell PDF fitting.

For composite materials, a single-phase small-box refiner
(:func:`midas_pdf.structure.refine_structure`) is inadequate. Real
samples often contain:

  * Two or more crystalline phases (α-brass + β-brass, austenite +
    martensite, Ni + Cu alloy end-members).
  * Core-shell nanoparticles: metallic core + oxide / ligand shell,
    each with its own lattice and finite-size damping.
  * Nanocrystal + amorphous matrix.

This module extends the single-phase forward model to a weighted sum:

    G_total(r) = Σ w_i · G_i(r) · γ_i(r)

with per-phase lattice, U_iso, and finite-size diameter, plus mixing
weights that sum to 1. Every parameter is torch-differentiable so the
whole ensemble is refinable via the same LBFGS + Hessian machinery.

Core-shell is a specialisation with two phases (``core``, ``shell``)
whose weights are constrained by the volume ratio of the geometric
shells rather than being free.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from .saxs.joint import sphere_characteristic_function


# ---------------------------------------------------------------------------
# Forward: weighted sum of pdffit_gr per phase
# ---------------------------------------------------------------------------

def multi_phase_gr(
    crystal_tensors: Sequence,
    pairs_list: Sequence,
    r: torch.Tensor,
    *,
    weights: torch.Tensor | Sequence[float],
    u_isos: torch.Tensor | Sequence[float],
    scales: Optional[torch.Tensor | Sequence[float]] = None,
    lattice_params_list: Optional[Sequence[torch.Tensor]] = None,
    diameters_A: Optional[torch.Tensor | Sequence[float]] = None,
    q_broad: float | torch.Tensor = 0.0,
) -> torch.Tensor:
    """Weighted sum of per-phase G(r) contributions.

    Parameters
    ----------
    crystal_tensors : list of ``crystal.to_torch()`` objects — the
        phases to combine.
    pairs_list : list of PairList objects (one per phase); result of
        :func:`midas_pdf.structure.build_pair_list` for each crystal.
    r : shared r-grid (Å).
    weights : mixing weights ``w_i`` (auto-normalised to sum 1).
    u_isos : isotropic displacement per phase (Å²).
    scales : optional per-phase multiplicative scale (default 1.0).
    lattice_params_list : optional list of 6-tensors overriding each
        phase's stored lattice parameters (for refinement).
    diameters_A : optional per-phase finite-size damping diameters (Å).
        If ``None``, no damping is applied.
    q_broad : shared Q-space instrumental broadening.

    Returns
    -------
    G_total(r) : (R,) tensor summed with the auto-normalised weights.
    """
    from .structure import pdffit_gr

    n = len(crystal_tensors)
    if len(pairs_list) != n:
        raise ValueError("pairs_list must have same length as crystal_tensors")
    r_t = torch.as_tensor(r, dtype=torch.float64)
    w = torch.as_tensor(weights, dtype=torch.float64)
    if w.numel() != n:
        raise ValueError(f"weights must have length {n}, got {w.numel()}")
    w = w / w.sum().clamp(min=1e-30)                    # auto-normalise
    if scales is None:
        scales = torch.ones(n, dtype=torch.float64)
    else:
        scales = torch.as_tensor(scales, dtype=torch.float64)
    u = torch.as_tensor(u_isos, dtype=torch.float64)

    G_total = torch.zeros_like(r_t)
    for i in range(n):
        lp = (lattice_params_list[i] if lattice_params_list is not None
              else crystal_tensors[i].lattice_params.detach())
        G_i = pdffit_gr(crystal_tensors[i], r_t, pairs_list[i],
                         scale=scales[i], u_iso=u[i],
                         lattice_params=lp, q_broad=q_broad)
        if diameters_A is not None:
            D_i = torch.as_tensor(diameters_A, dtype=torch.float64)[i]
            G_i = G_i * sphere_characteristic_function(r_t, D_i)
        G_total = G_total + w[i] * G_i
    return G_total


# ---------------------------------------------------------------------------
# Multi-phase small-box refiner
# ---------------------------------------------------------------------------

@dataclass
class MultiPhaseResult:
    fitted: Dict[str, float]              = field(default_factory=dict)
    uncertainty: Dict[str, float]         = field(default_factory=dict)
    G_calc: Optional[torch.Tensor]        = None
    chi2_reduced: float                   = float("nan")
    loss_history: List[float]             = field(default_factory=list)
    n_phases: int                         = 0
    weights_normalised: List[float]       = field(default_factory=list)


def refine_multi_phase(
    crystal_tensors: Sequence,
    r: torch.Tensor,
    G_obs: torch.Tensor,
    pairs_list: Sequence,
    *,
    sigma_obs: Optional[torch.Tensor] = None,
    init_a: Optional[Sequence[float]] = None,
    init_u_iso: float | Sequence[float] = 0.006,
    init_scale: float | Sequence[float] = 1.0,
    init_weights: Optional[Sequence[float]] = None,
    diameters_A: Optional[Sequence[float]] = None,
    steps: int = 200,
    lr: float = 0.05,
) -> MultiPhaseResult:
    """Fit a weighted multi-phase model to ``G_obs``.

    Refines per-phase ``a``, ``u_iso``, ``scale``, plus mixing weights
    (auto-renormalised each call). Diameters are optional and fixed
    (not refined) here; refinement of D is layered on top via
    :mod:`midas_pdf.saxs.joint`.
    """
    from .structure import pdffit_gr

    n = len(crystal_tensors)
    r_t = torch.as_tensor(r, dtype=torch.float64)
    G_t = torch.as_tensor(G_obs, dtype=torch.float64)
    w_obs = (torch.ones_like(G_t) if sigma_obs is None
             else 1.0 / torch.as_tensor(sigma_obs, dtype=torch.float64).clamp(min=1e-12) ** 2)

    # Initial lattice constants
    if init_a is None:
        a0s = [float(ct.lattice_params.detach()[0]) for ct in crystal_tensors]
    else:
        a0s = [float(v) for v in init_a]
        if len(a0s) != n:
            raise ValueError(f"init_a length {len(a0s)} != n_phases {n}")
    if isinstance(init_u_iso, (int, float)):
        u0s = [float(init_u_iso)] * n
    else:
        u0s = [float(v) for v in init_u_iso]
    if isinstance(init_scale, (int, float)):
        s0s = [float(init_scale)] * n
    else:
        s0s = [float(v) for v in init_scale]
    if init_weights is None:
        w0s = [1.0 / n] * n
    else:
        w0s = [float(v) for v in init_weights]
        if len(w0s) != n:
            raise ValueError(f"init_weights length {len(w0s)} != n_phases {n}")

    angles_list = [ct.lattice_params.detach()[3:] for ct in crystal_tensors]

    # Flat parameter vector: [a_1, u_1, s_1, ..., a_n, u_n, s_n, w_1, ..., w_n]
    theta_init = []
    for i in range(n):
        theta_init.extend([a0s[i], u0s[i], s0s[i]])
    theta_init.extend(w0s)
    theta = torch.tensor(theta_init, dtype=torch.float64, requires_grad=True)

    def model(th):
        lp_list = []
        u_list = []
        s_list = []
        for i in range(n):
            a_i = th[3 * i]
            lp_list.append(torch.cat([a_i.reshape(1).expand(3), angles_list[i]]))
            u_list.append(th[3 * i + 1])
            s_list.append(th[3 * i + 2])
        w_raw = th[3 * n:]
        return multi_phase_gr(
            crystal_tensors, pairs_list, r_t,
            weights=w_raw,
            u_isos=torch.stack(u_list),
            scales=torch.stack(s_list),
            lattice_params_list=lp_list,
            diameters_A=diameters_A,
        )

    def chi2(th):
        res = G_t - model(th)
        return (w_obs * res * res).sum()

    opt = torch.optim.LBFGS([theta], lr=lr, max_iter=steps,
                              line_search_fn="strong_wolfe")
    history: List[float] = []

    def closure():
        opt.zero_grad()
        loss = chi2(theta)
        loss.backward()
        history.append(float(loss.detach()))
        return loss

    opt.step(closure)

    # Unpack results
    fitted: Dict[str, float] = {}
    for i in range(n):
        fitted[f"a_{i}"] = float(theta[3 * i].detach())
        fitted[f"u_iso_{i}"] = float(theta[3 * i + 1].detach())
        fitted[f"scale_{i}"] = float(theta[3 * i + 2].detach())
    weights_raw = theta[3 * n:].detach()
    weights_norm = weights_raw / weights_raw.sum().clamp(min=1e-30)
    for i in range(n):
        fitted[f"weight_{i}"] = float(weights_norm[i])

    # Uncertainties
    uncert: Dict[str, float] = {k: float("nan") for k in fitted}
    try:
        th_leaf = theta.detach().clone().requires_grad_(True)
        H = torch.autograd.functional.hessian(chi2, th_leaf)
        cov = 2.0 * torch.linalg.inv(H)
        sig = torch.sqrt(torch.diagonal(cov).clamp(min=0.0))
        for i in range(n):
            uncert[f"a_{i}"] = float(sig[3 * i])
            uncert[f"u_iso_{i}"] = float(sig[3 * i + 1])
            uncert[f"scale_{i}"] = float(sig[3 * i + 2])
    except Exception:
        pass

    with torch.no_grad():
        G_calc = model(theta)
        ndof = max(int(G_t.numel()) - theta.numel(), 1)
        chi2_red = float((w_obs * (G_t - G_calc) ** 2).sum()) / ndof

    return MultiPhaseResult(
        fitted=fitted, uncertainty=uncert, G_calc=G_calc.detach(),
        chi2_reduced=chi2_red,
        loss_history=history,
        n_phases=n,
        weights_normalised=[float(w) for w in weights_norm],
    )


# ---------------------------------------------------------------------------
# Core-shell PDF forward + refine
# ---------------------------------------------------------------------------

def core_shell_pdf_gr(
    core_crystal,
    shell_crystal,
    r: torch.Tensor,
    core_pairs,
    shell_pairs,
    *,
    R_core_A: float | torch.Tensor,
    shell_thickness_A: float | torch.Tensor,
    u_iso_core: float | torch.Tensor,
    u_iso_shell: float | torch.Tensor,
    scale_core: float | torch.Tensor = 1.0,
    scale_shell: float | torch.Tensor = 1.0,
    lattice_core: Optional[torch.Tensor] = None,
    lattice_shell: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Core-shell nanoparticle G(r).

    Two phases weighted by their geometric volume fractions:

        V_core  = (4/3) π R_core³
        V_shell = (4/3) π ((R_core + t_shell)³ − R_core³)

    Each phase gets its own finite-size damping γ(r, D) with
    ``D_core = 2 R_core`` and ``D_total = 2 (R_core + t_shell)``. The
    shell characteristic function is approximated as the outer sphere's
    γ; the more accurate hollow-shell form is a Rev-15+ upgrade.
    """
    from .structure import pdffit_gr

    R_c = torch.as_tensor(R_core_A, dtype=torch.float64)
    t_s = torch.as_tensor(shell_thickness_A, dtype=torch.float64)
    R_o = R_c + t_s
    V_c = (4.0 / 3.0) * float(np.pi) * R_c ** 3
    V_s = (4.0 / 3.0) * float(np.pi) * (R_o ** 3 - R_c ** 3)
    w_c = V_c / (V_c + V_s)
    w_s = 1.0 - w_c

    r_t = torch.as_tensor(r, dtype=torch.float64)
    lp_c = (lattice_core if lattice_core is not None
            else core_crystal.lattice_params.detach())
    lp_s = (lattice_shell if lattice_shell is not None
            else shell_crystal.lattice_params.detach())

    G_core = pdffit_gr(core_crystal, r_t, core_pairs,
                        scale=scale_core, u_iso=u_iso_core,
                        lattice_params=lp_c)
    G_core = G_core * sphere_characteristic_function(r_t, 2.0 * R_c)
    G_shell = pdffit_gr(shell_crystal, r_t, shell_pairs,
                         scale=scale_shell, u_iso=u_iso_shell,
                         lattice_params=lp_s)
    G_shell = G_shell * sphere_characteristic_function(r_t, 2.0 * R_o)

    return w_c * G_core + w_s * G_shell


@dataclass
class CoreShellResult:
    fitted: Dict[str, float]              = field(default_factory=dict)
    uncertainty: Dict[str, float]         = field(default_factory=dict)
    G_calc: Optional[torch.Tensor]        = None
    chi2_reduced: float                   = float("nan")
    loss_history: List[float]             = field(default_factory=list)
    volume_fractions: Tuple[float, float] = (float("nan"), float("nan"))


def refine_core_shell(
    core_crystal,
    shell_crystal,
    r: torch.Tensor,
    G_obs: torch.Tensor,
    core_pairs,
    shell_pairs,
    *,
    sigma_obs: Optional[torch.Tensor] = None,
    init_a_core: Optional[float] = None,
    init_a_shell: Optional[float] = None,
    init_u_iso_core: float = 0.006,
    init_u_iso_shell: float = 0.010,
    init_R_core_A: float = 30.0,
    init_shell_thickness_A: float = 10.0,
    init_scale_core: float = 1.0,
    init_scale_shell: float = 1.0,
    steps: int = 200,
    lr: float = 0.1,
) -> CoreShellResult:
    """LBFGS refinement of a core-shell PDF model.

    Refines: ``a_core, a_shell, u_iso_core, u_iso_shell, R_core,
    shell_thickness, scale_core, scale_shell``.  Volume fractions are
    NOT independent — they're determined by ``R_core`` and
    ``shell_thickness``.
    """
    r_t = torch.as_tensor(r, dtype=torch.float64)
    G_t = torch.as_tensor(G_obs, dtype=torch.float64)
    w_obs = (torch.ones_like(G_t) if sigma_obs is None
             else 1.0 / torch.as_tensor(sigma_obs, dtype=torch.float64).clamp(min=1e-12) ** 2)

    a_c0 = (float(core_crystal.lattice_params.detach()[0])
            if init_a_core is None else float(init_a_core))
    a_s0 = (float(shell_crystal.lattice_params.detach()[0])
            if init_a_shell is None else float(init_a_shell))
    angles_c = core_crystal.lattice_params.detach()[3:]
    angles_s = shell_crystal.lattice_params.detach()[3:]

    theta = torch.tensor(
        [a_c0, a_s0, init_u_iso_core, init_u_iso_shell,
         init_R_core_A, init_shell_thickness_A,
         init_scale_core, init_scale_shell],
        dtype=torch.float64, requires_grad=True)
    names = ["a_core", "a_shell", "u_iso_core", "u_iso_shell",
              "R_core_A", "shell_thickness_A",
              "scale_core", "scale_shell"]

    def model(th):
        lp_c = torch.cat([th[0].reshape(1).expand(3), angles_c])
        lp_s = torch.cat([th[1].reshape(1).expand(3), angles_s])
        return core_shell_pdf_gr(
            core_crystal, shell_crystal, r_t, core_pairs, shell_pairs,
            R_core_A=th[4], shell_thickness_A=th[5],
            u_iso_core=th[2], u_iso_shell=th[3],
            scale_core=th[6], scale_shell=th[7],
            lattice_core=lp_c, lattice_shell=lp_s,
        )

    def chi2(th):
        res = G_t - model(th)
        return (w_obs * res * res).sum()

    opt = torch.optim.LBFGS([theta], lr=lr, max_iter=steps,
                              line_search_fn="strong_wolfe")
    history: List[float] = []

    def closure():
        opt.zero_grad()
        loss = chi2(theta)
        loss.backward()
        history.append(float(loss.detach()))
        return loss

    opt.step(closure)

    fitted = {names[i]: float(theta[i].detach()) for i in range(len(names))}
    uncert = {k: float("nan") for k in names}
    try:
        th_leaf = theta.detach().clone().requires_grad_(True)
        H = torch.autograd.functional.hessian(chi2, th_leaf)
        cov = 2.0 * torch.linalg.inv(H)
        sig = torch.sqrt(torch.diagonal(cov).clamp(min=0.0))
        uncert = {k: float(sig[i]) for i, k in enumerate(names)}
    except Exception:
        pass

    with torch.no_grad():
        R_c = fitted["R_core_A"]
        R_o = R_c + fitted["shell_thickness_A"]
        V_c = (4.0 / 3.0) * float(np.pi) * R_c ** 3
        V_s = (4.0 / 3.0) * float(np.pi) * (R_o ** 3 - R_c ** 3)
        f_c = V_c / (V_c + V_s)
        G_calc = model(theta)
        ndof = max(int(G_t.numel()) - theta.numel(), 1)
        chi2_red = float((w_obs * (G_t - G_calc) ** 2).sum()) / ndof

    return CoreShellResult(
        fitted=fitted, uncertainty=uncert,
        G_calc=G_calc.detach(), chi2_reduced=chi2_red,
        loss_history=history,
        volume_fractions=(float(f_c), float(1.0 - f_c)),
    )


__all__ = [
    "multi_phase_gr", "MultiPhaseResult", "refine_multi_phase",
    "core_shell_pdf_gr", "CoreShellResult", "refine_core_shell",
]
