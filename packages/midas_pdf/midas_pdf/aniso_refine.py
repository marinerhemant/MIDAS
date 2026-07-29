"""Rev-15 anisotropic-ADP + partial-occupancy PDF refinement.

The Rev-8 :func:`midas_pdf.structure.refine_structure` refines a single
isotropic ``u_iso`` and full occupancy on every site.  Real fits of low-
symmetry crystals, oxide superlattices, and solid solutions need:

* per-site **anisotropic** displacement tensors ``U_ij`` (3×3 symmetric,
  6 unique entries), and
* per-site **partial occupancy** (defect-vacancy / solid-solution work).

This module adds a differentiable L-BFGS refiner that co-refines those two
sets of DOFs together with the cubic-lattice constant and PDF scale, and
reports Hessian-derived 1σ on every DOF.  It preserves the diagonal
constraint on ``U`` (only the six unique entries — ``U11 U22 U33 U12 U13 U23``
— are stored and refined; the tensor is reconstructed symmetrically each
forward pass).  A positive-definiteness soft penalty keeps the fit
physical without needing a hard constraint.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch

from .structure import PairList, pdffit_gr


__all__ = [
    "u_vector_to_matrix",
    "u_matrix_to_vector",
    "refine_aniso_occupancy",
    "AnisoRefineResult",
]


def u_vector_to_matrix(u_vec: torch.Tensor) -> torch.Tensor:
    """Expand (n_uc, 6) unique entries [U11 U22 U33 U12 U13 U23] to
    (n_uc, 3, 3) symmetric tensors, differentiable."""
    u = torch.as_tensor(u_vec, dtype=torch.float64)
    if u.ndim != 2 or u.shape[1] != 6:
        raise ValueError(f"u_vec must be (n_uc, 6), got {tuple(u.shape)}")
    n = u.shape[0]
    U = torch.zeros((n, 3, 3), dtype=torch.float64)
    U[:, 0, 0] = u[:, 0]
    U[:, 1, 1] = u[:, 1]
    U[:, 2, 2] = u[:, 2]
    U[:, 0, 1] = u[:, 3]; U[:, 1, 0] = u[:, 3]
    U[:, 0, 2] = u[:, 4]; U[:, 2, 0] = u[:, 4]
    U[:, 1, 2] = u[:, 5]; U[:, 2, 1] = u[:, 5]
    return U


def u_matrix_to_vector(U: torch.Tensor) -> torch.Tensor:
    """Pack a (n_uc, 3, 3) symmetric tensor into (n_uc, 6) unique entries."""
    U = torch.as_tensor(U, dtype=torch.float64)
    if U.ndim != 3 or U.shape[1:] != (3, 3):
        raise ValueError(f"U must be (n_uc, 3, 3), got {tuple(U.shape)}")
    return torch.stack([U[:, 0, 0], U[:, 1, 1], U[:, 2, 2],
                        U[:, 0, 1], U[:, 0, 2], U[:, 1, 2]], dim=1)


class AnisoRefineResult(dict):
    def __getattr__(self, name):
        try: return self[name]
        except KeyError: raise AttributeError(name) from None


def refine_aniso_occupancy(
    crystal_tensor,
    r: torch.Tensor | np.ndarray,
    G_obs: torch.Tensor | np.ndarray,
    pairs: PairList,
    *,
    sigma_obs: Optional[torch.Tensor | np.ndarray] = None,
    init_a: Optional[float] = None,
    init_u_iso: float = 0.006,
    init_scale: float = 1.0,
    refine_aniso: bool = True,
    refine_occupancy: bool = False,
    steps: int = 200,
    lr: float = 0.05,
    positive_definite_penalty: float = 1e3,
) -> AnisoRefineResult:
    """Co-refine cubic-``a``, ``scale``, per-site aniso ADPs and (optional)
    per-site occupancy against ``G_obs``.

    Parameter layout is:

        θ = [a, scale] + (6 * n_uc aniso entries if refine_aniso else 0)
                       + (n_uc occupancies if refine_occupancy else 0)

    On ``refine_aniso=False`` the fit still uses ``u_iso`` on every site so
    the interface stays uniform.  The positive-definite penalty adds
    ``λ · sum(relu(-eigvals(U)))²`` to χ² — cheap, differentiable, and
    it turns off automatically once every U is p.d.
    """
    r_t = torch.as_tensor(r, dtype=torch.float64)
    G_t = torch.as_tensor(G_obs, dtype=torch.float64)
    w = (torch.ones_like(G_t) if sigma_obs is None
         else 1.0 / torch.as_tensor(sigma_obs, dtype=torch.float64).clamp(min=1e-12) ** 2)

    lat0 = crystal_tensor.lattice_params.detach()
    angles = lat0[3:]
    a0 = float(lat0[0]) if init_a is None else float(init_a)
    n_uc = pairs.n_uc

    theta_parts = [torch.tensor([a0, init_scale], dtype=torch.float64)]
    n_aniso = 6 * n_uc if refine_aniso else 0
    if refine_aniso:
        u_init = torch.tensor([init_u_iso, init_u_iso, init_u_iso, 0.0, 0.0, 0.0],
                              dtype=torch.float64)
        theta_parts.append(u_init.repeat(n_uc))
    n_occ = n_uc if refine_occupancy else 0
    if refine_occupancy:
        theta_parts.append(torch.ones(n_uc, dtype=torch.float64))
    theta = torch.cat(theta_parts).clone().requires_grad_(True)

    def unpack(th):
        a = th[0]
        scale = th[1]
        idx = 2
        u_aniso = None
        occupancy = None
        if refine_aniso:
            u_flat = th[idx:idx + n_aniso].reshape(n_uc, 6)
            u_aniso = u_vector_to_matrix(u_flat)
            idx += n_aniso
        if refine_occupancy:
            occupancy = th[idx:idx + n_occ].clamp(min=0.0, max=1.0)
            idx += n_occ
        return a, scale, u_aniso, occupancy

    def model(th):
        a, scale, u_aniso, occupancy = unpack(th)
        lp = torch.cat([a.reshape(1).expand(3), angles])
        kwargs = {"lattice_params": lp, "scale": scale}
        if u_aniso is not None:
            kwargs["u_aniso"] = u_aniso
        else:
            kwargs["u_iso"] = init_u_iso
        if occupancy is not None:
            kwargs["occupancy"] = occupancy
        return pdffit_gr(crystal_tensor, r_t, pairs, **kwargs)

    def chi2(th):
        res = G_t - model(th)
        loss = (w * res * res).sum()
        if refine_aniso and positive_definite_penalty > 0.0:
            _, _, u_aniso, _ = unpack(th)
            eigs = torch.linalg.eigvalsh(u_aniso)                        # (n_uc, 3)
            neg = torch.relu(-eigs)
            loss = loss + positive_definite_penalty * (neg * neg).sum()
        return loss

    opt = torch.optim.LBFGS([theta], lr=lr, max_iter=steps,
                             line_search_fn="strong_wolfe")
    history: list[float] = []

    def closure():
        opt.zero_grad()
        loss = chi2(theta)
        loss.backward()
        history.append(float(loss.detach()))
        return loss

    opt.step(closure)

    with torch.no_grad():
        th_det = theta.detach()
        a_val, scale_val, u_aniso_val, occ_val = unpack(th_det)
        G_fit = model(th_det)
        ndof = max(int(G_t.numel()) - theta.numel(), 1)
        chi2_red = float((w * (G_t - G_fit) ** 2).sum()) / ndof

    # Hessian uncertainties (may be huge for weakly-identified aniso entries).
    uncert = {}
    try:
        th_leaf = theta.detach().clone().requires_grad_(True)
        H = torch.autograd.functional.hessian(chi2, th_leaf)
        cov = 2.0 * torch.linalg.pinv(H)
        sig = torch.sqrt(torch.diagonal(cov).clamp(min=0.0))
        uncert["a"] = float(sig[0])
        uncert["scale"] = float(sig[1])
        idx = 2
        if refine_aniso:
            uncert["u_aniso"] = sig[idx:idx + n_aniso].reshape(n_uc, 6).tolist()
            idx += n_aniso
        if refine_occupancy:
            uncert["occupancy"] = sig[idx:idx + n_occ].tolist()
    except Exception:
        uncert = {"error": "Hessian failed"}

    return AnisoRefineResult(
        fitted={
            "a": float(a_val),
            "scale": float(scale_val),
            "u_aniso": u_aniso_val.tolist() if u_aniso_val is not None else None,
            "occupancy": occ_val.tolist() if occ_val is not None else None,
        },
        uncertainty=uncert,
        G_calc=G_fit.detach(),
        chi2_reduced=chi2_red,
        loss=history[-1] if history else float("nan"),
        history=history, loss_history=history,
    )
