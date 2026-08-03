"""Nelder-Mead polish on the hard ``FracOverlap`` objective.

After L-BFGS converges on the soft Gaussian-splat surrogate (which has
nice gradients but a slightly different basin floor than the discrete
pixel-set ``FracOverlap`` the C code optimises), this module runs a
short ``scipy.optimize.minimize(method="Nelder-Mead")`` step that
refines the orientation against the **hard** objective directly. The
polish:

- consumes the L-BFGS-warmed Eulers as the initial simplex centre;
- enforces the same ``±OrientTol`` box the L-BFGS used (matching the C
  NLopt LN_NELDERMEAD bounds);
- minimises ``1 − hard_fraction`` evaluated through the same forward
  model + ``ObsVolume.hard_fraction`` lookup the screen uses;
- returns the polished Eulers and the final hard fraction, both
  consistent with what the C ``FitOrientationOMP`` writes to
  ``MicFileBinary``.

This is what makes the Python pipeline objective-bit-exact with the C
reference (modulo NM convergence noise). It also keeps the L-BFGS
warmup path so we still get the gradient-driven basin search; the NM
step is cheap from a good seed (~50–100 hard-frac evals).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
from scipy.optimize import minimize

from midas_diffract.forward import HEDMForwardModel

from .obs_volume import ObsVolume


@dataclass
class PolishResult:
    """Outcome of one :func:`polish_hard_frac` call.

    Attributes
    ----------
    eul : torch.Tensor (3,)
        Polished Euler triplet (Bunge ZXZ, radians).
    hard_frac : float
        Hard FracOverlap at ``eul``. Same metric the C
        ``CalcFracOverlap`` reports.
    n_evals : int
        Number of hard-frac evaluations the NM ran. Useful for
        diagnostics; expect ~50–100 for a well-warmed seed.
    converged : bool
        True if scipy reported convergence under the configured
        tolerances; False for max-iter exit.
    """
    eul: torch.Tensor
    hard_frac: float
    n_evals: int
    converged: bool


def polish_hard_frac(
    model: HEDMForwardModel,
    obs: ObsVolume,
    warmed_eul: torch.Tensor,
    pos_um: torch.Tensor,
    tol_rad: float,
    *,
    max_iter: int = 200,
    xatol: float = 1e-5,
    fatol: float = 1e-5,
    adaptive: bool = True,
    refl_weight: "torch.Tensor | None" = None,
) -> PolishResult:
    """Refine ``warmed_eul`` against hard FracOverlap via Nelder-Mead.

    Parameters
    ----------
    model : HEDMForwardModel
        Forward model already configured with the right hkls / wavelength /
        geometry. Must run on the same device as ``obs``.
    obs : ObsVolume
        Decoded observation bitmap. ``hard_fraction`` is called inside
        the NM closure.
    warmed_eul : torch.Tensor shape (3,)
        L-BFGS-converged Eulers, in radians, on the same device/dtype as
        the forward model.
    pos_um : torch.Tensor shape (3,)
        Voxel centroid in lab-frame µm.
    tol_rad : float
        Half-width of the ±-box in radians (matches the L-BFGS tanh box,
        which in turn matches the C NLopt ``[x0 ± OrientTol]`` bounds).
    max_iter : int
        scipy NM ``maxiter``. The C NLopt call uses 5000 evals and a 30 s
        time-limit; from a warm seed we typically need < 200.
    xatol, fatol : float
        scipy NM tolerances. Match the C NLopt's ``xtol_rel = ftol_rel =
        1e-5``.
    adaptive : bool
        scipy's adaptive simplex (Gao & Han 2010) — better behaved on
        non-smooth objectives like discrete FracOverlap.
    """
    if warmed_eul.shape != (3,):
        raise ValueError(f"warmed_eul must be (3,), got {warmed_eul.shape}")
    if pos_um.shape != (3,):
        raise ValueError(f"pos_um must be (3,), got {pos_um.shape}")

    device = warmed_eul.device
    dtype = warmed_eul.dtype
    seed = warmed_eul.detach().cpu().numpy().astype(np.float64)
    bounds = [(seed[i] - tol_rad, seed[i] + tol_rad) for i in range(3)]

    pos_b = pos_um.unsqueeze(0)

    def neg_hard_frac(eul_np: np.ndarray) -> float:
        eul = torch.tensor(eul_np, dtype=dtype, device=device)
        with torch.no_grad():
            spots = model(eul.unsqueeze(0), pos_b)
            hf = float(obs.hard_fraction(
                spots.frame_nr, spots.y_pixel, spots.z_pixel, spots.valid,
                refl_weight=refl_weight,
            ))
        return 1.0 - hf

    res = minimize(
        neg_hard_frac, seed, method="Nelder-Mead",
        bounds=bounds,
        options={
            "xatol": xatol, "fatol": fatol,
            "maxiter": max_iter, "adaptive": adaptive,
        },
    )

    final_eul = torch.tensor(res.x, dtype=dtype, device=device)
    final_frac = max(0.0, 1.0 - float(res.fun))
    return PolishResult(
        eul=final_eul,
        hard_frac=final_frac,
        n_evals=int(res.nfev),
        converged=bool(res.success),
    )
