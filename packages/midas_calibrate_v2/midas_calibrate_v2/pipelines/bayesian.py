"""Bayesian calibration pipeline.

Three modes (priority order):

- ``mode="laplace"``  MAP via LM, then Hessian-at-MAP for marginal 1σ.
                      Cheap, ships first.
- ``mode="vi"``       Mean-field Gaussian VI via pyro.  Validates Laplace.
- ``mode="hmc"``      NUTS via pyro.  Slow on ~25-dim with image-scale data;
                      gated explicitly by the user.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from midas_calibrate.params import CalibrationParams as V1Params

from ..compat.from_v1 import spec_from_v1_params
from ..forward.panels import PanelLayout
from ..inference.laplace import LaplaceResult, laplace_at_map, report_laplace
from ..loss.pseudo_strain import pseudo_strain_residual
from ..loss.prior import sum_log_prior
from ..parameters.spec import CalibrationSpec
from ._common import FittedDataset, run_estep_v1
from .single import autocalibrate as autocalibrate_single


@dataclass
class BayesianResult:
    map_unpacked: Dict[str, torch.Tensor]
    laplace: Optional[LaplaceResult] = None
    vi_margins: Optional[Dict[str, Tuple[float, float]]] = None
    vi_elbo_trace: Optional[List[float]] = None
    hmc_samples: Optional[Dict[str, torch.Tensor]] = None


def _build_log_likelihood(fits: FittedDataset, spec: CalibrationSpec,
                            panel_layout: Optional[PanelLayout]):
    """Return ``log_likelihood_fn(unpacked) -> scalar``.

    Noise model: pseudo-strain residuals are Gaussian with σ from the v1 SNR
    weighting (treated as a Cauchy-like spread to be robust to outliers).
    """
    def log_lik(unpacked: Dict[str, torch.Tensor]) -> torch.Tensor:
        r = pseudo_strain_residual(
            fits.Y_pix, fits.Z_pix, fits.ring_two_theta_deg, unpacked,
            rho_d=fits.rho_d, weights=fits.weights,
            panel_layout=panel_layout, panel_idx=fits.panel_idx,
        )
        # Standard Gaussian log-likelihood (σ=1 since residuals are
        # already weighted; user can pass priors to encode physical scale).
        return -0.5 * (r * r).sum() + sum_log_prior(unpacked, spec)
    return log_lik


def autocalibrate_bayesian(
    v1_params: V1Params,
    image: np.ndarray,
    *,
    mode: str = "laplace",
    dark: Optional[np.ndarray] = None,
    spec: Optional[CalibrationSpec] = None,
    panel_layout: Optional[PanelLayout] = None,
    laplace_ridge: float = 1e-9,
    vi_steps: int = 2000,
    vi_lr: float = 1e-2,
    hmc_warmup: int = 200,
    hmc_samples: int = 500,
    n_iter_map: int = 5,
    dtype=torch.float64, device: str = "cpu",
    verbose: bool = True,
) -> BayesianResult:
    """Run Bayesian calibration.

    Mode ``"laplace"`` always runs first (MAP + Hessian).  Modes ``"vi"`` and
    ``"hmc"`` additionally run their respective samplers.
    """
    if spec is None:
        spec = spec_from_v1_params(v1_params)

    # Step 1: MAP via the alternating engine — gets us close to the posterior
    # mode quickly.
    map_result = autocalibrate_single(
        v1_params, image, dark=dark, spec=spec, panel_layout=panel_layout,
        n_iter=n_iter_map, dtype=dtype, device=device, verbose=verbose,
    )
    map_unpacked = map_result.unpacked
    fits = map_result.fits_final
    if fits is None:
        raise RuntimeError("MAP run produced no fitted points")

    log_lik = _build_log_likelihood(fits, spec, panel_layout)

    def nll(unpacked: Dict[str, torch.Tensor]) -> torch.Tensor:
        return -log_lik(unpacked)

    out = BayesianResult(map_unpacked=map_unpacked)

    if mode in ("laplace", "vi", "hmc"):
        if verbose:
            print("[bayesian] computing Laplace approximation at MAP...")
        out.laplace = laplace_at_map(
            spec, nll, map_unpacked,
            ridge=laplace_ridge, dtype=dtype, device=device,
        )
        if verbose:
            print(report_laplace(out.laplace))

    if mode == "vi":
        from ..inference.vi import vi_run, VIConfig
        if verbose:
            print(f"[bayesian] running pyro VI ({vi_steps} steps)...")
        _, margins, elbo = vi_run(
            spec, log_lik,
            config=VIConfig(n_steps=vi_steps, lr=vi_lr),
            dtype=dtype, device=device, verbose=verbose,
        )
        out.vi_margins = margins
        out.vi_elbo_trace = elbo

    if mode == "hmc":
        from ..inference.hmc import hmc_run, HMCConfig
        if verbose:
            print(f"[bayesian] running pyro NUTS ({hmc_samples} samples after "
                  f"{hmc_warmup} warmup)...")
        out.hmc_samples = hmc_run(
            spec, log_lik,
            config=HMCConfig(n_warmup=hmc_warmup, n_samples=hmc_samples),
            dtype=dtype, device=device,
        )

    return out


__all__ = ["BayesianResult", "autocalibrate_bayesian"]
