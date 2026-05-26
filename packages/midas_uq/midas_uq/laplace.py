"""Hessian-based Laplace approximation of the grain-state posterior.

Adapted from proto12. Gaussian noise model:
    NLL(theta) = (1/2) * sum_i ||(pred_i(theta) - obs_i) / sigma_i||^2
H = Hessian of NLL at the converged state; Laplace covariance = H^{-1}.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

from midas_diffract import HEDMForwardModel, SpotMatchingLoss

from ._common import (
    DEG2RAD, RAD2DEG, GrainState, default_loss,
)
from .spots import _fit_grain_spots


@dataclass
class LaplaceResult:
    """Laplace-approximation posterior covariance at the converged state.

    Attributes
    ----------
    covariance : numpy.ndarray (D, D)
        Laplace covariance for the parameter vector (euler [3] | latc [6]).
    eigenvalues : numpy.ndarray (D,)
        Hessian eigenvalues, smallest first (post pseudo-inverse).
    condition_number : float
    misori_p95_deg : float
        From a Monte-Carlo sample of the Gaussian posterior.
    lattice_p95_A : float
    """
    covariance: np.ndarray
    eigenvalues: np.ndarray
    condition_number: float
    misori_p95_deg: float
    lattice_p95_A: float


def _freeze_associations(
    model: HEDMForwardModel,
    state: GrainState,
    observations: torch.Tensor,
    max_dist: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Resolve observed->predicted spot correspondences at the converged
    state and return their flat indices so the NLL Hessian can be computed
    on a fixed, differentiable assignment."""
    pos = state.pos if state.pos is not None else torch.zeros(3, dtype=state.euler_rad.dtype)
    with torch.no_grad():
        spots = model(state.euler_rad.unsqueeze(0), pos.unsqueeze(0),
                      lattice_params=state.latc)
        coords, valid = HEDMForwardModel.predict_spot_coords(
            spots, space="angular",
        )
        pred = coords.squeeze().reshape(-1, 3)
        vmask = valid.squeeze().reshape(-1) > 0.5
        valid_idx = torch.nonzero(vmask, as_tuple=False).squeeze(-1)
        pred_v = pred[vmask]
        dists = torch.cdist(observations, pred_v)
        min_d, nn = dists.min(dim=1)
        keep = min_d < max_dist
        return valid_idx[nn[keep]], torch.nonzero(keep, as_tuple=False).squeeze(-1)


def laplace_covariance(
    model: HEDMForwardModel,
    state: GrainState,
    observations: torch.Tensor,
    sigma_vec: torch.Tensor,
    *,
    refine_first: bool = False,
    n_mc_samples: int = 2000,
    seed: int = 0,
) -> LaplaceResult:
    """Compute Laplace covariance of (euler | latc) at the converged state.

    Parameters
    ----------
    model : HEDMForwardModel
    state : GrainState
        Converged state (output of `half_half_spots` -> per_half_states or a
        full-set fit). Pass `refine_first=True` to refine the state inside
        the call if `state` is only a seed.
    observations : Tensor (N, 3)
        Observed spot angular coordinates in radians.
    sigma_vec : Tensor (3,)
        Per-coordinate measurement noise standard deviations in radians.
    refine_first : bool
        If True, run `_fit_grain_spots` before computing the Hessian.
    n_mc_samples : int
        Number of Monte-Carlo draws for posterior summaries.
    seed : int

    Returns
    -------
    LaplaceResult
    """
    if refine_first:
        state = _fit_grain_spots(model, observations, state, default_loss())

    pos = state.pos if state.pos is not None else torch.zeros(
        3, dtype=state.euler_rad.dtype, device=state.euler_rad.device,
    )

    pred_idx, obs_idx = _freeze_associations(model, state, observations)

    def nll(z: torch.Tensor) -> torch.Tensor:
        eul = z[:3]
        lat = z[3:]
        spots = model(eul.unsqueeze(0), pos.unsqueeze(0), lattice_params=lat)
        coords, _ = HEDMForwardModel.predict_spot_coords(spots, space="angular")
        pred_flat = coords.squeeze().reshape(-1, 3)
        pred_match = pred_flat[pred_idx]
        obs_match = observations[obs_idx]
        resid = (pred_match - obs_match) / sigma_vec.unsqueeze(0)
        return 0.5 * torch.sum(resid ** 2)

    z_star = torch.cat([state.euler_rad, state.latc]).detach().clone()
    # Hessian-pinv-eigvals core delegated to the shared midas_invert leaf
    # (one Laplace implementation across MIDAS / laue_torch / midas_uq).
    from midas_invert import laplace_uncertainty as _laplace
    _res = _laplace(nll, z_star)
    Sigma = _res["cov"].detach().cpu().numpy()
    w = _res["eigvals"].detach().cpu().numpy()
    cond = _res["cond_number"]

    # Monte-Carlo sample misori + lattice distributions
    rng = np.random.default_rng(seed)
    try:
        L = np.linalg.cholesky(Sigma + 1e-18 * np.eye(Sigma.shape[0]))
        deltas = (L @ rng.standard_normal((Sigma.shape[0], n_mc_samples))).T
    except np.linalg.LinAlgError:
        w_c, V_c = np.linalg.eigh(Sigma)
        w_c = np.clip(w_c, 0, None)
        deltas = (rng.standard_normal((n_mc_samples, Sigma.shape[0]))
                  * np.sqrt(w_c)) @ V_c.T

    R_star = HEDMForwardModel.euler2mat(state.euler_rad).cpu().numpy()
    mis = np.empty(n_mc_samples); lat = np.empty(n_mc_samples)
    for i in range(n_mc_samples):
        eul_i = state.euler_rad.cpu().numpy() + deltas[i, :3]
        R_i = HEDMForwardModel.euler2mat(
            torch.tensor(eul_i, dtype=state.euler_rad.dtype),
        ).cpu().numpy()
        trace = float((R_star.T @ R_i).trace())
        mis[i] = math.degrees(math.acos(max(-1.0, min(1.0, (trace - 1) / 2))))
        lat[i] = float(np.max(np.abs(deltas[i, 3:6])))

    return LaplaceResult(
        covariance=Sigma,
        eigenvalues=w,
        condition_number=cond,
        misori_p95_deg=float(np.percentile(mis, 95)),
        lattice_p95_A=float(np.percentile(lat, 95)),
    )
