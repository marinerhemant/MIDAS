"""NN-augmented residual training pipeline.

Two-stage training (default, recommended):
  Stage A — freeze geometry, train the conv NN on the Stage-2 residual.
  Stage B — thaw geometry, joint fine-tune.

Single-stage joint training is also available via ``mode="joint"``.

Regularisation: weight decay + smoothness penalty ∫|∇f|² over the field.
Without smoothness the NN absorbs harmonics that should live in p₀..p₁₄.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import torch

from midas_calibrate.params import CalibrationParams as V1Params

from ..compat.from_v1 import spec_from_v1_params
from ..forward.nn_residual import NNResidualConfig, ResidualConvNet
from ..forward.panels import PanelLayout
from ..inference.adam import AdamConfig, adam_minimise
from ..loss.nn_regularizer import smoothness_penalty, weight_decay
from ..loss.pseudo_strain import pseudo_strain_residual
from ..parameters.spec import CalibrationSpec
from ._common import FittedDataset, run_estep_v1
from .single import autocalibrate as autocalibrate_single


@dataclass
class NNCalibrationResult:
    spec: CalibrationSpec
    nn_model: ResidualConvNet
    map_unpacked: Dict[str, torch.Tensor]
    losses: List[float]
    harmonic_drift: Dict[str, float]   # per-harmonic drift between pre/post NN


def _residual_with_nn(
    fits: FittedDataset,
    unpacked: Dict[str, torch.Tensor],
    nn_model: ResidualConvNet,
    *,
    panel_layout: Optional[PanelLayout],
) -> torch.Tensor:
    """Pseudo-strain residual augmented by a NN ΔR sampled at fit points."""
    r_base = pseudo_strain_residual(
        fits.Y_pix, fits.Z_pix, fits.ring_two_theta_deg, unpacked,
        rho_d=fits.rho_d, weights=fits.weights,
        panel_layout=panel_layout, panel_idx=fits.panel_idx,
    )
    delta_R = nn_model.sample(fits.Y_pix, fits.Z_pix)
    # Convert ΔR (pixels) into the same strain units as r_base.
    pxY = unpacked["pxY"]
    pxZ = unpacked.get("pxZ", pxY)
    px_mean = 0.5 * (pxY + pxZ)
    from ..forward.bragg import R_ideal_px
    R_pred = R_ideal_px(fits.ring_two_theta_deg, unpacked["Lsd"], px_mean)
    return r_base - delta_R / R_pred


def autocalibrate_nn(
    v1_params: V1Params,
    image: np.ndarray,
    *,
    dark: Optional[np.ndarray] = None,
    spec: Optional[CalibrationSpec] = None,
    panel_layout: Optional[PanelLayout] = None,
    nn_config: Optional[NNResidualConfig] = None,
    mode: str = "two_stage",      # "two_stage" | "joint"
    n_iter_seed: int = 5,
    n_steps_nn: int = 1500,
    nn_lr: float = 1e-3,
    weight_decay_coef: float = 1e-4,
    smoothness_coef: float = 1e-3,
    dtype=torch.float64, device: str = "cpu",
    verbose: bool = True,
) -> NNCalibrationResult:
    """Train a conv NN ΔR residual on top of the analytical model.

    Returns the trained network, MAP geometry, training loss curve, and
    a per-harmonic drift report (proves the NN didn't absorb physical
    signal that should live in pₖ).
    """
    if spec is None:
        spec = spec_from_v1_params(v1_params)
    if nn_config is None:
        nn_config = NNResidualConfig(
            detector_H_px=v1_params.NrPixelsY,
            detector_W_px=v1_params.NrPixelsZ,
        )

    # Stage 0: seed geometry via the alternating engine.
    if verbose:
        print("[nn] seeding geometry via alternating engine...")
    seed_result = autocalibrate_single(
        v1_params, image, dark=dark, spec=spec, panel_layout=panel_layout,
        n_iter=n_iter_seed, dtype=dtype, device=device, verbose=verbose,
    )
    fits = seed_result.fits_final
    if fits is None:
        raise RuntimeError("seed alternating run produced no fits")

    # Snapshot harmonic coefficients for drift report.
    from ..forward.distortion import P_COEF_NAMES
    harmonic_pre = {n: float(seed_result.unpacked[n]) for n in P_COEF_NAMES
                     if n in seed_result.unpacked}

    # Build the NN.
    nn_model = ResidualConvNet(nn_config).to(device=device, dtype=dtype)

    if mode == "two_stage":
        # Stage A: freeze geometry; train NN only.
        if verbose:
            print(f"[nn] Stage A: training NN with geometry frozen ({n_steps_nn} steps)...")
        unpacked = {k: v.detach() for k, v in seed_result.unpacked.items()}

        optim = torch.optim.Adam(nn_model.parameters(), lr=nn_lr)
        losses_A: List[float] = []
        for step in range(n_steps_nn):
            optim.zero_grad()
            r = _residual_with_nn(fits, unpacked, nn_model, panel_layout=panel_layout)
            data = 0.5 * (r * r).sum()
            reg = weight_decay_coef * weight_decay(nn_model) \
                + smoothness_coef * smoothness_penalty(nn_model.field())
            loss = data + reg
            loss.backward()
            optim.step()
            losses_A.append(float(loss.detach()))
            if verbose and step % 200 == 0:
                print(f"[nn A {step:5d}] loss={losses_A[-1]:.6e}")

        # Stage B: thaw geometry; joint fine-tune.
        if verbose:
            print("[nn] Stage B: joint fine-tune...")

        def loss_fn(unpacked_now: Dict[str, torch.Tensor]) -> torch.Tensor:
            r = _residual_with_nn(fits, unpacked_now, nn_model, panel_layout=panel_layout)
            data = 0.5 * (r * r).sum()
            reg = weight_decay_coef * weight_decay(nn_model) \
                + smoothness_coef * smoothness_penalty(nn_model.field())
            return data + reg

        unpacked_B, losses_B = adam_minimise(
            spec, loss_fn,
            config=AdamConfig(lr=1e-3, nn_lr=nn_lr, n_steps=n_steps_nn // 2),
            extra_params=nn_model.parameters(),
            dtype=dtype, device=device, verbose=verbose,
        )
        losses = losses_A + losses_B
        map_unpacked = unpacked_B
    elif mode == "joint":
        if verbose:
            print(f"[nn] joint training ({n_steps_nn} steps)...")

        def loss_fn(unpacked_now: Dict[str, torch.Tensor]) -> torch.Tensor:
            r = _residual_with_nn(fits, unpacked_now, nn_model, panel_layout=panel_layout)
            data = 0.5 * (r * r).sum()
            reg = weight_decay_coef * weight_decay(nn_model) \
                + smoothness_coef * smoothness_penalty(nn_model.field())
            return data + reg

        map_unpacked, losses = adam_minimise(
            spec, loss_fn,
            config=AdamConfig(lr=1e-2, nn_lr=nn_lr, n_steps=n_steps_nn),
            extra_params=nn_model.parameters(),
            dtype=dtype, device=device, verbose=verbose,
        )
    else:
        raise ValueError(f"unknown mode {mode!r}; use 'two_stage' or 'joint'")

    # Harmonic drift report — should be small.
    harmonic_drift = {}
    for k, v_pre in harmonic_pre.items():
        v_post = float(map_unpacked[k])
        harmonic_drift[k] = v_post - v_pre

    if verbose:
        print("[nn] harmonic drift (post − pre):")
        for k, d in harmonic_drift.items():
            print(f"  {k:5s}  Δ={d:+.4e}")

    return NNCalibrationResult(
        spec=spec, nn_model=nn_model, map_unpacked=map_unpacked,
        losses=losses, harmonic_drift=harmonic_drift,
    )


__all__ = ["NNCalibrationResult", "autocalibrate_nn"]
