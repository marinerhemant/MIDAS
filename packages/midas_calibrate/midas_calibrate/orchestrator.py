"""End-to-end calibration orchestrator (alternating engine).

Runs the E↔M loop with optional outlier rejection between iterations.  The
joint differentiable engine is a separate entry point in `joint.py`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np

from .estep import CakeProfile, run_estep
from .params import CalibrationParams
from .refine import FittedPoint, RefineResult, refine_geometry
from .rings import RingTable, build_ring_table


@dataclass
class IterRecord:
    iteration: int
    n_fitted: int
    cost: float
    rc: int
    mean_strain_uE: float
    Lsd: float
    BC_y: float
    BC_z: float
    ty: float
    tz: float


@dataclass
class CalibrationResult:
    params: CalibrationParams
    history: List[IterRecord] = field(default_factory=list)
    fits_final: Optional[List[FittedPoint]] = None
    cake_final: Optional[CakeProfile] = None


def _sigma_clip(fits: List[FittedPoint], result: RefineResult,
                 rt: RingTable, params: CalibrationParams,
                 sigma_factor: float) -> List[FittedPoint]:
    """Reject fitted points whose strain residual exceeds sigma_factor·σ."""
    if not fits:
        return fits
    # Compute per-point strain at the refined params.
    import torch
    from .geometry_torch import predict_R_at_pixel
    from .param_vector import pack

    px = 0.5 * (params.pxY + params.pxZ) if params.pxZ > 0 else params.pxY
    Y = torch.tensor([p.Y_pix for p in fits], dtype=torch.float64)
    Z = torch.tensor([p.Z_pix for p in fits], dtype=torch.float64)
    x = pack(result.params, dtype=torch.float64)
    R_obs = predict_R_at_pixel(Y, Z, x, px=px, rho_d=result.params.RhoD).numpy()
    R_pred = np.array([rt.r_ideal_px[p.ring_idx] for p in fits])
    strain = np.abs(1.0 - R_obs / R_pred)
    threshold = strain.mean() + sigma_factor * strain.std()
    return [p for p, s in zip(fits, strain) if s <= threshold]


def autocalibrate(
    params: CalibrationParams,
    image: np.ndarray,
    *, dark: Optional[np.ndarray] = None,
    mask: Optional[np.ndarray] = None,
    verbose: bool = True,
) -> CalibrationResult:
    """Full alternating E↔M calibration.

    ``mask`` is an optional bad-pixel mask, same shape and orientation as
    ``image`` (apply the same ImTransOpt to both). **Nonzero means BAD.**
    Masked pixels are removed from both the numerator and the denominator of
    every cake cell; see :func:`midas_calibrate.estep.integrate_cake`.

    Returns a CalibrationResult with refined parameters and per-iteration history.
    """
    params.validate()
    history: List[IterRecord] = []
    rt: Optional[RingTable] = None
    fits_final: Optional[List[FittedPoint]] = None
    cake_final: Optional[CakeProfile] = None

    for it in range(params.nIterations):
        # Rebuild the ring table at the current geometry (Lsd/wavelength may have changed).
        rt = build_ring_table(params)

        # E-step: integrate image, extract fitted (Y_pix, Z_pix) per ring × η.
        cake, fits = run_estep(params, image, rt, dark=dark, mask=mask)

        # Optional sigma-clip rejection: must have a previous iteration's fit
        # for the strain estimate.
        if params.RemoveOutliersBetweenIters and history:
            from .refine import RefineResult as _RR
            prev = _RR(params=params, cost=history[-1].cost, rc=history[-1].rc,
                       mean_strain_uE=history[-1].mean_strain_uE,
                       n_points_used=history[-1].n_fitted)
            fits = _sigma_clip(fits, prev, rt, params, params.OutlierFactor)

        if not fits:
            if verbose:
                print(f"[iter {it}] no usable fits — aborting")
            break

        # M-step: refine geometry.
        result = refine_geometry(
            params, rt, fits,
            max_iter=200,
            huber_delta=(params.HuberDelta if params.Loss.lower() == "huber" else None),
            verbose=False,
        )
        params = result.params

        rec = IterRecord(
            iteration=it, n_fitted=len(fits),
            cost=result.cost, rc=result.rc,
            mean_strain_uE=result.mean_strain_uE,
            Lsd=params.Lsd, BC_y=params.BC_y, BC_z=params.BC_z,
            ty=params.ty, tz=params.tz,
        )
        history.append(rec)
        cake_final = cake
        fits_final = fits

        if verbose:
            print(f"[iter {it}] n_fits={len(fits):4d}  rc={result.rc}  "
                  f"strain={result.mean_strain_uE:8.1f}μϵ  "
                  f"Lsd={params.Lsd:.2f}  BC=({params.BC_y:.3f},{params.BC_z:.3f})  "
                  f"ty={params.ty:.4f}  tz={params.tz:.4f}")

        # Convergence check
        if len(history) >= 2:
            prev = history[-2].mean_strain_uE
            cur = history[-1].mean_strain_uE
            if cur < 1.0 or abs(prev - cur) < 0.01 * max(prev, 1.0):
                if verbose:
                    print(f"[iter {it}] converged (mean strain stationary)")
                break

    return CalibrationResult(params=params, history=history,
                              fits_final=fits_final, cake_final=cake_final)
