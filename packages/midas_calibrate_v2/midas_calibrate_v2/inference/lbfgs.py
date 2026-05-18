"""LBFGS over a packed parameter vector with sigmoid-box reparameterisation.

The closure receives the *bounded* parameter dict (post-sigmoid); the
optimiser sees the unbounded ``u``-vector.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple

import torch

from ..parameters.pack import (
    PackInfo, refined_indices, refined_bounds, refined_subset,
    write_refined_back, unpack_spec,
)
from ..parameters.spec import CalibrationSpec


@dataclass
class LBFGSConfig:
    max_iter: int = 200
    tolerance_grad: float = 1e-6
    tolerance_change: float = 1e-9
    history_size: int = 20
    line_search_fn: Optional[str] = "strong_wolfe"
    fallback_span: float = 1.0   # for parameters without explicit bounds


def lbfgs_minimise(
    spec: CalibrationSpec,
    loss_fn: Callable[[Dict[str, torch.Tensor]], torch.Tensor],
    *,
    config: LBFGSConfig = LBFGSConfig(),
    dtype=torch.float64, device="cpu",
) -> Tuple[Dict[str, torch.Tensor], float, int]:
    """Minimise ``loss_fn(unpacked)`` over the refined parameters.

    The bounded ↔ unbounded reparameterisation uses sigmoid-box transforms
    (see :mod:`midas_peakfit.reparam`) for refined parameters with explicit
    bounds; refined parameters without bounds get a fallback ±span.

    Returns
    -------
    (final_unpacked_dict, final_loss, iterations)
    """
    from midas_peakfit.reparam import x_to_u, u_to_x
    from ..parameters.pack import pack_spec

    x_full, info = pack_spec(spec, dtype=dtype, device=device)
    lo, hi = refined_bounds(spec, info, fallback_span=config.fallback_span,
                             dtype=dtype, device=device)
    refined_idx = refined_indices(info).to(device)
    x_ref = x_full.index_select(0, refined_idx).clone()
    u = x_to_u(x_ref, lo, hi).clone().detach().requires_grad_(True)

    optim = torch.optim.LBFGS(
        [u],
        max_iter=config.max_iter,
        tolerance_grad=config.tolerance_grad,
        tolerance_change=config.tolerance_change,
        history_size=config.history_size,
        line_search_fn=config.line_search_fn,
    )

    n_iter = [0]
    last_loss = [float("inf")]

    def closure():
        optim.zero_grad()
        x_ref_now = u_to_x(u, lo, hi)
        x_full_now = write_refined_back(x_full, x_ref_now, info)
        unpacked = unpack_spec(x_full_now, info, spec)
        loss = loss_fn(unpacked)
        loss.backward()
        n_iter[0] += 1
        last_loss[0] = float(loss.detach())
        return loss

    optim.step(closure)

    # Final unpack at the converged u.
    with torch.no_grad():
        x_ref_final = u_to_x(u, lo, hi)
        x_full_final = write_refined_back(x_full, x_ref_final, info)
        unpacked_final = unpack_spec(x_full_final, info, spec)
    return unpacked_final, last_loss[0], n_iter[0]


__all__ = ["LBFGSConfig", "lbfgs_minimise"]
