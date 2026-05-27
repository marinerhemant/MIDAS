"""Joint residual closure that concatenates powder + HEDM + gauge + prior.

The residual closure is the *only* per-iteration entry point the LM driver
sees. It's a pure function of the unpacked parameter dict — perfectly
compatible with :func:`midas_peakfit.lm_minimise` (which forward-mode-AD
Jacobians the closure) and :func:`midas_peakfit.laplace_at_map` /
:func:`midas_peakfit.fisher_at_map`.

Loss-weighting strategy (paper-4 §4.3):

    * powder pseudo-strain residual is dimensionless (~1e-4 RMS at well-fit MAP)
    * HEDM spot residual is in pixels (~0.3 px RMS at well-fit MAP)

We expose ``w_powder`` and ``w_hedm`` so the user can match the two scales.
A reasonable default for ``kind="pixel"`` HEDM is ``w_powder=1.0`` and
``w_hedm = 1.0 / r_typical_px`` (roughly 3.0 px so w_hedm ~ 0.3) — but the
runner scripts sweep this for the paper figure.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional

import torch

from midas_peakfit import (
    gaussian_prior_residual,
    zero_sum_residual,
)


@dataclass
class JointWeights:
    """Per-modality scalar weights applied to the residual block before LM
    sees it.  LM minimises ``Σ_i (w_i r_i)²``; weights enter the cost as
    ``w_i²`` (equivalent to setting σ_residual = 1/w_i)."""
    w_powder: float = 1.0
    w_hedm: float = 1.0
    # Gauge λ in physical residual units (sqrt(λ) is what's actually
    # multiplied into the gauge rows; default matches paper-3 §9).
    lambda_gauge: float = 1e6


def joint_residual(
    unpacked: Dict[str, torch.Tensor],
    *,
    powder_residual_fn: Callable[[Dict[str, torch.Tensor]], torch.Tensor],
    hedm_residual_fn: Callable[[Dict[str, torch.Tensor]], torch.Tensor],
    spec=None,
    weights: JointWeights = JointWeights(),
    gauge_blocks: Optional[list] = None,
) -> torch.Tensor:
    """Concatenated residual vector for joint LM.

    Parameters
    ----------
    unpacked
        Dict produced by :func:`midas_peakfit.unpack_spec` on the joint spec.
    powder_residual_fn
        ``unpacked -> [M_powder]`` tensor of powder pseudo-strain residuals.
        Typically a closure built around
        :func:`midas_calibrate_v2.loss.pseudo_strain.pseudo_strain_residual`.
    hedm_residual_fn
        ``unpacked -> [M_hedm]`` tensor of HEDM spot residuals.  Typically
        a closure built around
        :func:`midas_fit_grain.spec_residual.hedm_spot_residual` with a
        pre-bound :class:`HEDMResidualBundle`.
    spec
        The full joint :class:`ParameterSpec`.  Required to emit
        Gaussian-prior rows for any parameter that carries a prior; pass
        ``None`` to disable the prior block.
    weights
        Per-modality weights and gauge λ.
    gauge_blocks
        List of parameter-block names to constrain to Σ=0 (per-panel gauge
        fixing).  Default is paper-3's standard set.
    """
    if gauge_blocks is None:
        gauge_blocks = [
            "panel_delta_yz", "panel_delta_theta",
            "panel_delta_lsd", "panel_delta_p2",
            "delta_r_k",
        ]

    rp = powder_residual_fn(unpacked)
    rh = hedm_residual_fn(unpacked)

    pieces = [
        weights.w_powder * rp,
        weights.w_hedm * rh,
    ]

    rg = zero_sum_residual(unpacked, block_names=gauge_blocks,
                            lambda_zs=weights.lambda_gauge)
    if rg.numel() > 0:
        pieces.append(rg)

    if spec is not None:
        rprior = gaussian_prior_residual(unpacked, spec)
        if rprior.numel() > 0:
            pieces.append(rprior)

    return torch.cat([p.flatten() for p in pieces if p.numel() > 0])


__all__ = ["JointWeights", "joint_residual"]
