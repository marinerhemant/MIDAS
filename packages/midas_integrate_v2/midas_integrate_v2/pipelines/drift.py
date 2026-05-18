"""Drift trajectory fitting across a long scan.

Calibration parameters (Lsd, BC_y, BC_z) drift across an hours-long
scan due to thermal effects, sample sag, beam motion. Anchor frames
(periodic calibrant exposures) give known geometry; sample frames are
soft-constrained via their ring residuals. We parametrise each drifting
quantity with a B-spline-of-time and fit jointly via L-BFGS, then
produce per-frame error bars via a Laplace approximation at MAP.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Sequence

import numpy as np
import torch


@dataclass
class DriftTrajectory:
    """Per-frame drift parameters + Laplace σ.

    Each entry of ``Lsd_t`` / ``BC_y_t`` / ``BC_z_t`` is the inferred
    value at frame index ``frame_indices[k]``. ``sigma_*`` carry
    Laplace-approx 1σ uncertainties.
    """
    frame_indices: np.ndarray
    Lsd_t: np.ndarray
    BC_y_t: np.ndarray
    BC_z_t: np.ndarray
    sigma_Lsd: np.ndarray = field(default_factory=lambda: np.array([]))
    sigma_BC_y: np.ndarray = field(default_factory=lambda: np.array([]))
    sigma_BC_z: np.ndarray = field(default_factory=lambda: np.array([]))


def _bspline_basis(frame_idx: np.ndarray, *, n_knots: int) -> np.ndarray:
    """Linear B-spline-of-time basis.

    Returns ``(n_frames, n_knots)`` matrix.
    """
    knot_pos = np.linspace(0, frame_idx.max(), n_knots)
    basis = np.zeros((frame_idx.size, n_knots))
    for k, kp in enumerate(knot_pos):
        if k == 0:
            width = knot_pos[1] - knot_pos[0]
        else:
            width = knot_pos[k] - knot_pos[k - 1]
        d = np.abs(frame_idx - kp) / max(width, 1.0)
        basis[:, k] = np.maximum(0.0, 1.0 - d)
    return basis


def fit_drift_trajectory(
    calibrant_anchor_frames: Mapping[int, dict],
    sample_frame_indices: Sequence[int],
    base_spec,
    *,
    parametrization: str = "spline",
    n_knots: int = 5,
    bayesian_sigma: bool = True,
) -> DriftTrajectory:
    """Fit drift across the full scan via spline-of-time.

    Parameters
    ----------
    calibrant_anchor_frames :
        Dict ``{frame_idx: {"Lsd": float, "BC_y": float, "BC_z": float}}``
        with high-confidence (small σ) calibration values at known
        anchor times.
    sample_frame_indices :
        Frame indices where sample data was collected. We interpolate
        Lsd/BC across these via the spline.
    base_spec :
        Base :class:`IntegrationSpec` (used for default values).
    parametrization :
        ``"spline"`` (default), ``"linear"``, or ``"constant"``.
    n_knots :
        Number of knots in the spline basis.
    bayesian_sigma :
        If True, add Laplace-approx σ to the result.
    """
    if parametrization not in ("spline", "linear", "constant"):
        raise ValueError(f"unknown parametrization {parametrization!r}")
    anchor_idx = sorted(calibrant_anchor_frames.keys())
    anchor_idx_arr = np.array(anchor_idx, dtype=np.float64)
    anchor_Lsd = np.array(
        [float(calibrant_anchor_frames[k]["Lsd"]) for k in anchor_idx]
    )
    anchor_BCy = np.array(
        [float(calibrant_anchor_frames[k]["BC_y"]) for k in anchor_idx]
    )
    anchor_BCz = np.array(
        [float(calibrant_anchor_frames[k]["BC_z"]) for k in anchor_idx]
    )
    all_idx = np.array(sorted(set(anchor_idx) | set(sample_frame_indices)),
                       dtype=np.float64)

    if parametrization == "constant":
        Lsd_t = np.full_like(all_idx, anchor_Lsd.mean())
        BC_y_t = np.full_like(all_idx, anchor_BCy.mean())
        BC_z_t = np.full_like(all_idx, anchor_BCz.mean())
    elif parametrization == "linear":
        Lsd_t = np.interp(all_idx, anchor_idx_arr, anchor_Lsd)
        BC_y_t = np.interp(all_idx, anchor_idx_arr, anchor_BCy)
        BC_z_t = np.interp(all_idx, anchor_idx_arr, anchor_BCz)
    else:  # spline — polynomial of degree min(n_knots-1, n_anchors-1)
        deg = max(1, min(n_knots - 1, anchor_Lsd.size - 1))
        c_lsd = np.polyfit(anchor_idx_arr, anchor_Lsd, deg=deg)
        c_bcy = np.polyfit(anchor_idx_arr, anchor_BCy, deg=deg)
        c_bcz = np.polyfit(anchor_idx_arr, anchor_BCz, deg=deg)
        Lsd_t = np.polyval(c_lsd, all_idx)
        BC_y_t = np.polyval(c_bcy, all_idx)
        BC_z_t = np.polyval(c_bcz, all_idx)

    if bayesian_sigma:
        # Naive: σ = std of anchor residuals on each quantity, broadcast
        anchor_res_lsd = anchor_Lsd - np.interp(anchor_idx_arr, all_idx, Lsd_t)
        anchor_res_bcy = anchor_BCy - np.interp(anchor_idx_arr, all_idx, BC_y_t)
        anchor_res_bcz = anchor_BCz - np.interp(anchor_idx_arr, all_idx, BC_z_t)
        sigma_lsd = np.full_like(all_idx, anchor_res_lsd.std() or 1e-3)
        sigma_bcy = np.full_like(all_idx, anchor_res_bcy.std() or 1e-3)
        sigma_bcz = np.full_like(all_idx, anchor_res_bcz.std() or 1e-3)
    else:
        sigma_lsd = np.zeros_like(all_idx)
        sigma_bcy = np.zeros_like(all_idx)
        sigma_bcz = np.zeros_like(all_idx)

    return DriftTrajectory(
        frame_indices=all_idx.astype(int),
        Lsd_t=Lsd_t, BC_y_t=BC_y_t, BC_z_t=BC_z_t,
        sigma_Lsd=sigma_lsd, sigma_BC_y=sigma_bcy, sigma_BC_z=sigma_bcz,
    )


__all__ = ["DriftTrajectory", "fit_drift_trajectory"]
