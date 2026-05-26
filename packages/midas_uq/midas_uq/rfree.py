"""R_free analog: train vs holdout loss gap during refinement.

Run a refinement on a random subset of observed spots while tracking the
loss on the held-out spots at each L-BFGS outer step. Useful in the
low-n_obs/DOF regime (sparse grains, mosaic grains, pf-HEDM) where the
gap is statistically distinguishable from zero.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

from midas_diffract import HEDMForwardModel, SpotMatchingLoss

from ._common import GrainState, default_loss, _associate


@dataclass
class RFreeResult:
    train_losses: np.ndarray
    holdout_losses: np.ndarray
    misori_history_deg: np.ndarray
    lattice_history_A: np.ndarray
    n_train: int
    n_hold: int

    @property
    def gap_final(self) -> float:
        tr = float(self.train_losses[-1])
        ho = float(self.holdout_losses[-1])
        return (ho - tr) / max(tr, 1e-30)


def rfree_gap(
    model: HEDMForwardModel,
    init: GrainState,
    observations: torch.Tensor,
    *,
    train_fraction: float = 0.5,
    seed: int = 0,
    loss: Optional[SpotMatchingLoss] = None,
    phase1_steps: int = 10,
    phase2_steps: int = 10,
    phase3_steps: int = 8,
    lbfgs_max_iter: int = 20,
) -> RFreeResult:
    """Track train + holdout loss across the three-phase L-BFGS refinement.

    Returns
    -------
    RFreeResult with per-step train/holdout losses and grain-state history.
    """
    if loss is None: loss = default_loss()
    n = observations.shape[0]
    g = torch.Generator(device="cpu").manual_seed(seed)
    perm = torch.randperm(n, generator=g)
    n_train = int(train_fraction * n)
    train_obs = observations[perm[:n_train]]
    hold_obs = observations[perm[n_train:]]

    opt_euler = init.euler_rad.clone()
    opt_latc = init.latc.clone()
    pos = init.pos if init.pos is not None else torch.zeros(
        3, dtype=opt_euler.dtype, device=opt_euler.device,
    )

    def _loss_on(obs):
        with torch.no_grad():
            spots = model(opt_euler.unsqueeze(0), pos.unsqueeze(0),
                          lattice_params=opt_latc)
            coords, valid = HEDMForwardModel.predict_spot_coords(spots, space="angular")
            pred = coords.squeeze().reshape(-1, 3)
            vmask = valid.squeeze().reshape(-1) > 0.5
            pred_v = pred[vmask]
            pred_match, obs_match = _associate(pred_v, obs, 0.5)
            if pred_match.shape[0] < 5:
                return float("inf")
            return float(loss(pred_match, obs_match))

    train_history = []
    hold_history = []
    mis_history = []
    lat_history = []

    def record():
        train_history.append(_loss_on(train_obs))
        hold_history.append(_loss_on(hold_obs))
        with torch.no_grad():
            R_init = HEDMForwardModel.euler2mat(init.euler_rad)
            R_cur = HEDMForwardModel.euler2mat(opt_euler)
            tr = float((R_init.T @ R_cur).diagonal().sum())
            import math
            mis_history.append(math.degrees(math.acos(max(-1.0, min(1.0, (tr - 1) / 2)))))
            lat_history.append(float((opt_latc - init.latc)[:3].abs().max()))

    def closure_factory(params):
        def closure():
            for p in params:
                if p.grad is not None: p.grad.zero_()
            spots = model(opt_euler.unsqueeze(0), pos.unsqueeze(0),
                          lattice_params=opt_latc)
            coords, valid = HEDMForwardModel.predict_spot_coords(spots, space="angular")
            pred = coords.squeeze().reshape(-1, 3)
            vmask = valid.squeeze().reshape(-1) > 0.5
            if vmask.sum() == 0:
                return torch.tensor(1e12, dtype=opt_euler.dtype, requires_grad=True)
            pred_v = pred[vmask]
            pred_match, obs_match = _associate(pred_v, train_obs, 0.5)
            if pred_match.shape[0] < 5:
                return torch.tensor(1e12, dtype=opt_euler.dtype, requires_grad=True)
            l = loss(pred_match, obs_match)
            l.backward()
            return l
        return closure

    record()  # initial state
    opt_euler.requires_grad_(True); opt_latc.requires_grad_(False)
    o = torch.optim.LBFGS([opt_euler], lr=1.0, max_iter=lbfgs_max_iter,
                          line_search_fn="strong_wolfe")
    for _ in range(phase1_steps):
        o.step(closure_factory([opt_euler])); record()

    opt_euler.requires_grad_(False); opt_latc.requires_grad_(True)
    o = torch.optim.LBFGS([opt_latc], lr=1.0, max_iter=lbfgs_max_iter,
                          line_search_fn="strong_wolfe")
    for _ in range(phase2_steps):
        o.step(closure_factory([opt_latc])); record()

    opt_euler.requires_grad_(True); opt_latc.requires_grad_(True)
    o = torch.optim.LBFGS([opt_euler, opt_latc], lr=0.5, max_iter=lbfgs_max_iter,
                          line_search_fn="strong_wolfe")
    for _ in range(phase3_steps):
        o.step(closure_factory([opt_euler, opt_latc])); record()

    return RFreeResult(
        train_losses=np.array(train_history, dtype=np.float64),
        holdout_losses=np.array(hold_history, dtype=np.float64),
        misori_history_deg=np.array(mis_history, dtype=np.float64),
        lattice_history_A=np.array(lat_history, dtype=np.float64),
        n_train=int(train_obs.shape[0]),
        n_hold=int(hold_obs.shape[0]),
    )
