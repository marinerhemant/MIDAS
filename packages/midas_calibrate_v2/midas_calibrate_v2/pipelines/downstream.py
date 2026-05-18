"""Downstream HEDM strain coupling (M6).

Two phases:

- :func:`sensitivity_diagnostic`  Compute ∂ε_grain / ∂θ_calib.  No
  optimisation, just calibration validation.  Tells you which calibration
  parameters move which strain components on a real HEDM dataset.

- :func:`joint_with_downstream`   Auxiliary HEDM loss in the calibration
  objective: ``L = L_calibrant + λ L_HEDM``.  Calibration adjusts to
  minimise calibrant pseudo-strain *and* downstream grain-strain noise
  jointly.

Both phases require a user-supplied differentiable HEDM evaluator (work in
flight in midas_diffract / midas_grain_odf).  The pipeline ships in v2.0
with the public API; the evaluator is the user's plug-in point.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch

from midas_calibrate.params import CalibrationParams as V1Params

from ..compat.from_v1 import spec_from_v1_params
from ..forward.panels import PanelLayout
from ..inference.lbfgs import lbfgs_minimise, LBFGSConfig
from ..loss.pseudo_strain import pseudo_strain_residual
from ..parameters.spec import CalibrationSpec
from ._common import FittedDataset, run_estep_v1
from .single import autocalibrate as autocalibrate_single


HEDMEvaluator = Callable[[Dict[str, torch.Tensor]], torch.Tensor]
"""``unpacked -> [n_grains, 6]`` strain tensor or ``[n_grains]`` scalar
score.  Must be differentiable in the calibration parameters.
"""


@dataclass
class SensitivityReport:
    parameter_names: List[str]
    parameter_values: torch.Tensor      # [N_ref] MAP values
    sensitivity: torch.Tensor           # [N_ref] |∂L_HEDM / ∂θ| at MAP
    sensitivity_signed: torch.Tensor    # [N_ref] ∂L_HEDM / ∂θ at MAP


def sensitivity_diagnostic(
    v1_params: V1Params,
    image: np.ndarray,
    hedm_evaluator: HEDMEvaluator,
    *,
    dark: Optional[np.ndarray] = None,
    spec: Optional[CalibrationSpec] = None,
    panel_layout: Optional[PanelLayout] = None,
    n_iter_seed: int = 5,
    dtype=torch.float64, device: str = "cpu",
    verbose: bool = True,
) -> SensitivityReport:
    """Compute ∂L_HEDM / ∂θ_calib at the calibrant MAP.

    Tells you which calibration parameters carry the most leverage on
    downstream science.  No optimisation is performed.
    """
    if spec is None:
        spec = spec_from_v1_params(v1_params)

    map_result = autocalibrate_single(
        v1_params, image, dark=dark, spec=spec, panel_layout=panel_layout,
        n_iter=n_iter_seed, dtype=dtype, device=device, verbose=verbose,
    )

    # Pack the unpacked dict into a flat refinable subset, request grad on it,
    # then call the HEDM evaluator and backprop.
    refined_names: List[str] = []
    refined_inits: List[torch.Tensor] = []
    for name, p in spec.parameters.items():
        if not p.refined:
            continue
        v = map_result.unpacked[name].detach().clone()
        v.requires_grad_(True)
        refined_names.append(name)
        refined_inits.append(v)

    unpacked_with_grad = {**map_result.unpacked, **dict(zip(refined_names, refined_inits))}
    hedm_out = hedm_evaluator(unpacked_with_grad)
    if hedm_out.ndim > 0:
        loss = (hedm_out * hedm_out).sum()
    else:
        loss = hedm_out
    grads = torch.autograd.grad(loss, refined_inits, allow_unused=True)

    sens_signed = torch.cat([
        (g.detach().reshape(-1) if g is not None else torch.zeros(v.numel(), dtype=v.dtype))
        for g, v in zip(grads, refined_inits)
    ])
    sens_abs = sens_signed.abs()

    if verbose:
        print("[downstream] sensitivity ∂L_HEDM/∂θ at calibration MAP:")
        for i, n in enumerate(refined_names):
            print(f"  {n:<24s}  ∂L/∂θ = {float(sens_signed[i]):+.4e}")

    vals = torch.cat([v.detach().reshape(-1) for v in refined_inits])
    return SensitivityReport(
        parameter_names=refined_names,
        parameter_values=vals,
        sensitivity=sens_abs,
        sensitivity_signed=sens_signed,
    )


def joint_with_downstream(
    v1_params: V1Params,
    image: np.ndarray,
    hedm_evaluator: HEDMEvaluator,
    *,
    lambda_hedm: float = 1.0,
    dark: Optional[np.ndarray] = None,
    spec: Optional[CalibrationSpec] = None,
    panel_layout: Optional[PanelLayout] = None,
    n_iter_seed: int = 3,
    lbfgs_max_iter: int = 200,
    dtype=torch.float64, device: str = "cpu",
    verbose: bool = True,
) -> Dict[str, torch.Tensor]:
    """Joint calibration + downstream HEDM loss minimisation.

    Loss:  L = ½‖r_calibrant‖² + λ_hedm · ‖HEDM(unpacked)‖².

    Returns the converged unpacked dict.
    """
    if spec is None:
        spec = spec_from_v1_params(v1_params)

    seed_result = autocalibrate_single(
        v1_params, image, dark=dark, spec=spec, panel_layout=panel_layout,
        n_iter=n_iter_seed, dtype=dtype, device=device, verbose=verbose,
    )
    fits = seed_result.fits_final
    if fits is None:
        raise RuntimeError("seed run produced no fits")

    def loss_fn(unpacked: Dict[str, torch.Tensor]) -> torch.Tensor:
        r = pseudo_strain_residual(
            fits.Y_pix, fits.Z_pix, fits.ring_two_theta_deg, unpacked,
            rho_d=fits.rho_d, weights=fits.weights,
            panel_layout=panel_layout, panel_idx=fits.panel_idx,
        )
        L_cal = 0.5 * (r * r).sum()
        hedm_out = hedm_evaluator(unpacked)
        if hedm_out.ndim > 0:
            L_hedm = (hedm_out * hedm_out).sum()
        else:
            L_hedm = hedm_out
        return L_cal + lambda_hedm * L_hedm

    map_unpacked, final_loss, n_used = lbfgs_minimise(
        spec, loss_fn,
        config=LBFGSConfig(max_iter=lbfgs_max_iter),
        dtype=dtype, device=device,
    )
    if verbose:
        print(f"[downstream-joint] loss={final_loss:.6e}  iters={n_used}  "
              f"λ_hedm={lambda_hedm}")
    return map_unpacked


__all__ = ["HEDMEvaluator", "SensitivityReport",
           "sensitivity_diagnostic", "joint_with_downstream"]
