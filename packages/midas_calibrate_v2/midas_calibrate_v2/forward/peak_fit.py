"""Per-(ring, η-bin) pseudo-Voigt LM peak fit.

Replaces v1's centroid extraction with a torch pseudo-Voigt fit so the
resulting peak center carries an autograd-aware uncertainty contribution
when used as a fixed observation in the alternating M-step.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch

from .peak_shape import tch_pseudo_voigt


@dataclass
class FittedPeak:
    R_fit: float           # px (radial center)
    eta_deg: float         # bin center
    sigmaG: float
    gammaL: float
    area: float
    bg0: float             # constant background
    bg1: float             # linear-in-R background slope
    snr: float
    rms: float
    rc: int
    ring_idx: int


def fit_pseudo_voigt_lm(
    R_window: torch.Tensor,         # [n_R] radial bin centers (px)
    I_profile: torch.Tensor,        # [n_R] intensities
    *,
    init_center: float,
    init_sigma: float = 1.0,
    init_gamma: float = 0.5,
    init_bg: Optional[Tuple[float, float]] = None,
    max_iter: int = 60,
    ftol: float = 1e-8,
    lambda_init: float = 1e-3,
) -> FittedPeak:
    """Levenberg-Marquardt fit of a single TCH pseudo-Voigt + linear bg.

    Lightweight in-module LM (no external deps) — the per-(ring, η-bin) cost
    is small (5-7 params, ~10-50 samples) so the dense JTJ inversion is
    cheap.  Uses ``torch.autograd.functional.jacobian`` for the Jacobian.

    Returns
    -------
    :class:`FittedPeak` with fit statistics.
    """
    dtype = R_window.dtype
    device = R_window.device

    if init_bg is None:
        init_bg = (float(I_profile.min()), 0.0)

    # Parameter vector: [center, sigma, gamma, area, bg0, bg1]
    x = torch.tensor(
        [init_center, max(init_sigma, 1e-3), max(init_gamma, 1e-3),
         float(I_profile.max() - I_profile.min()) * float(init_sigma) * 2.5,
         init_bg[0], init_bg[1]],
        dtype=dtype, device=device,
    )

    R_norm = (R_window - R_window.mean()) / R_window.std().clamp(min=1e-6)

    def model(p: torch.Tensor) -> torch.Tensor:
        center, sigma, gamma, area, bg0, bg1 = p[0], p[1].abs() + 1e-6, \
            p[2].abs() + 1e-6, p[3], p[4], p[5]
        return tch_pseudo_voigt(R_window, center, sigma, gamma, area) + bg0 + bg1 * R_norm

    def residual(p: torch.Tensor) -> torch.Tensor:
        return model(p) - I_profile

    lam = lambda_init
    r = residual(x)
    cost = 0.5 * (r * r).sum()
    rc = 1
    for it in range(max_iter):
        # Jacobian via torch autograd.
        J = torch.autograd.functional.jacobian(residual, x, create_graph=False)
        JT = J.transpose(0, 1)
        H = JT @ J
        g = JT @ r
        # Marquardt damping.
        for _ in range(10):
            H_d = H + lam * torch.eye(H.shape[0], dtype=dtype, device=device) * H.diag().clamp(min=1e-12)
            try:
                step = torch.linalg.solve(H_d, g)
            except RuntimeError:
                lam *= 10.0
                continue
            x_new = x - step
            r_new = residual(x_new)
            cost_new = 0.5 * (r_new * r_new).sum()
            if cost_new < cost:
                rel = (cost - cost_new) / cost.clamp(min=1e-30)
                x = x_new
                r = r_new
                cost = cost_new
                lam = max(lam / 10.0, 1e-12)
                if rel < ftol:
                    rc = 0
                break
            else:
                lam = min(lam * 10.0, 1e10)
        else:
            rc = 2
            break

    center, sigma, gamma, area, bg0, bg1 = (float(x[0]), float(x[1].abs()),
                                             float(x[2].abs()), float(x[3]),
                                             float(x[4]), float(x[5]))
    rms = float((r * r).mean().sqrt())
    snr = float((I_profile.max() - I_profile.min()) / (rms + 1e-12))
    return FittedPeak(
        R_fit=center, eta_deg=0.0,  # caller fills eta
        sigmaG=sigma, gammaL=gamma, area=area,
        bg0=bg0, bg1=bg1, snr=snr, rms=rms, rc=rc, ring_idx=-1,
    )


def fit_cake_per_ring(
    cake_intensity: torch.Tensor,    # [n_R, n_eta]
    R_centers: torch.Tensor,
    eta_centers: torch.Tensor,
    ring_R_ideal_px: torch.Tensor,   # [n_rings] expected radii
    *,
    half_width_px: float = 5.0,
    snr_min: float = 2.0,
) -> List[FittedPeak]:
    """For each ring × η-bin, isolate the radial window around the ideal ring
    radius and fit a pseudo-Voigt.

    Returns a flat list of fitted peaks across all rings × η-bins that pass
    SNR filtering.
    """
    fits: List[FittedPeak] = []
    n_R, n_eta = cake_intensity.shape
    for ring_i, R_ideal in enumerate(ring_R_ideal_px.tolist()):
        # Pick radial window.
        keep = (R_centers >= R_ideal - half_width_px) & (R_centers <= R_ideal + half_width_px)
        idx = torch.nonzero(keep, as_tuple=True)[0]
        if idx.numel() < 4:
            continue
        R_win = R_centers[idx]
        I_block = cake_intensity[idx, :]   # [n_R_win, n_eta]

        for j in range(n_eta):
            I_prof = I_block[:, j]
            if I_prof.max() < I_prof.min() + 1e-12:
                continue
            try:
                fp = fit_pseudo_voigt_lm(
                    R_win, I_prof, init_center=float(R_ideal),
                )
            except Exception:
                continue
            if fp.snr < snr_min:
                continue
            fp.eta_deg = float(eta_centers[j])
            fp.ring_idx = ring_i
            fits.append(fp)
    return fits


__all__ = ["FittedPeak", "fit_pseudo_voigt_lm", "fit_cake_per_ring"]
