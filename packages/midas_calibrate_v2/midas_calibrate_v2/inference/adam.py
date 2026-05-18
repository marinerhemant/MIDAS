"""Adam optimiser over packed parameters — primary driver for NN-residual training.

Unlike LBFGS, Adam is well-suited to mini-batch / stochastic objectives, which
makes it the right backend for joint training of geometry parameters with a
neural-network residual model.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import torch

from ..parameters.pack import (
    PackInfo, refined_indices, refined_bounds,
    write_refined_back, unpack_spec, pack_spec,
)
from ..parameters.spec import CalibrationSpec


@dataclass
class AdamConfig:
    lr: float = 1e-2
    nn_lr: float = 1e-3   # separate, lower LR for NN-residual
    n_steps: int = 1000
    log_every: int = 50
    fallback_span: float = 1.0


def adam_minimise(
    spec: CalibrationSpec,
    loss_fn: Callable[[Dict[str, torch.Tensor]], torch.Tensor],
    *,
    config: AdamConfig = AdamConfig(),
    extra_params: Optional[Iterable[torch.nn.Parameter]] = None,
    dtype=torch.float64, device="cpu",
    verbose: bool = False,
) -> Tuple[Dict[str, torch.Tensor], List[float]]:
    """Adam over the refined parameters (and optional NN params).

    Parameters
    ----------
    extra_params : iterable of nn.Parameter
        Extra parameters (e.g., from a NN-residual model).  Optimised at
        ``config.nn_lr``.
    """
    from midas_peakfit.reparam import x_to_u, u_to_x

    x_full, info = pack_spec(spec, dtype=dtype, device=device)
    lo, hi = refined_bounds(spec, info, fallback_span=config.fallback_span,
                             dtype=dtype, device=device)
    refined_idx = refined_indices(info).to(device)
    x_ref = x_full.index_select(0, refined_idx).clone()
    u = x_to_u(x_ref, lo, hi).clone().detach().requires_grad_(True)

    param_groups = [{"params": [u], "lr": config.lr}]
    if extra_params is not None:
        param_groups.append({"params": list(extra_params), "lr": config.nn_lr})

    optim = torch.optim.Adam(param_groups)

    losses: List[float] = []
    for step in range(config.n_steps):
        optim.zero_grad()
        x_ref_now = u_to_x(u, lo, hi)
        x_full_now = write_refined_back(x_full, x_ref_now, info)
        unpacked = unpack_spec(x_full_now, info, spec)
        loss = loss_fn(unpacked)
        loss.backward()
        optim.step()
        losses.append(float(loss.detach()))
        if verbose and (step % config.log_every == 0):
            print(f"[adam {step:5d}] loss = {losses[-1]:.6e}")

    with torch.no_grad():
        x_ref_final = u_to_x(u, lo, hi)
        x_full_final = write_refined_back(x_full, x_ref_final, info)
        unpacked_final = unpack_spec(x_full_final, info, spec)
    return unpacked_final, losses


__all__ = ["AdamConfig", "adam_minimise"]
