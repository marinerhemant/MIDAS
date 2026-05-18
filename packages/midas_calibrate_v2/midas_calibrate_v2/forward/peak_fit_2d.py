"""Batched 2-D pseudo-Voigt LM peak fitter.

Per (ring, η-bin) region we fit a 2-D pseudo-Voigt over a *2-D* window of
the cake spanning ±half_R px in R and ±half_eta_deg in η.  Compared to
the 1-D fitter in :mod:`peak_fit_batched`, this:

- Uses much more data per region (n_R_win × n_eta_win samples vs n_R_win)
  → tighter LM convergence on each peak's center.
- Captures tangential (η-direction) broadening of the peak.
- Lets the radial center fit decouple from η-bin discretisation when the
  ring tilts diagonally across the bin.

Model (original Wertheim pV in 2-D, axis-aligned):

    I(R, η) = bg₀ + bg₁·(R - R_mid)
           + A · [η_v · L₂(ΔR, Δη; γ_R, γ_η)
                + (1 - η_v) · G₂(ΔR, Δη; σ_R, σ_η)]

where (ΔR, Δη) = (R - center_R, η - center_η) and the 2-D Gaussian /
Lorentzian factor:

    G₂ = exp(-½(ΔR/σ_R)² - ½(Δη/σ_η)²) / (2π σ_R σ_η)
    L₂ = 1 / [π² γ_R γ_η · (1 + (ΔR/γ_R)²) · (1 + (Δη/γ_η)²)]

Parameters per region (10):

    [center_R, center_η,        # offsets relative to (R_ideal, η_bin_center)
     σ_R, σ_η, γ_R, γ_η,        # widths
     η_v,                        # mixing fraction ∈ [0, 1]
     area, bg₀, bg₁]
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import math
import numpy as np
import torch

from midas_peakfit import GenericLMConfig, lm_solve_generic, u_to_x


_2PI = 2.0 * math.pi
_PI2 = math.pi * math.pi


def _pV_2d(
    R: torch.Tensor, eta: torch.Tensor,           # [B, M]
    cR: torch.Tensor, cE: torch.Tensor,            # [B, 1]
    sR: torch.Tensor, sE: torch.Tensor,            # σ_R, σ_η
    gR: torch.Tensor, gE: torch.Tensor,            # γ_R, γ_η
    eta_v: torch.Tensor,                           # [B, 1]
    area: torch.Tensor,                            # [B, 1]
) -> torch.Tensor:
    dR = R - cR
    dE = eta - cE
    G = torch.exp(-0.5 * (dR * dR) / (sR * sR + 1e-30)
                   -0.5 * (dE * dE) / (sE * sE + 1e-30)) \
        / (_2PI * sR * sE + 1e-30)
    L = 1.0 / (_PI2 * gR * gE
               * (1.0 + (dR * dR) / (gR * gR + 1e-30))
               * (1.0 + (dE * dE) / (gE * gE + 1e-30)))
    return area * (eta_v * L + (1.0 - eta_v) * G)


def _residual_and_jac_factory(
    R_grid: torch.Tensor,        # [B, M] (centered)
    E_grid: torch.Tensor,        # [B, M] (centered)
    R_mid: torch.Tensor,         # [B]
    I_block: torch.Tensor,       # [B, M]
):
    """Closed-form Jacobian for the 2-D pV residual (10 params).

    Same recipe as the 1-D version: stack analytical ∂r/∂x_i columns
    into [B, M, 10], then chain through the sigmoid reparam.  Avoids the
    50× row-by-row backprop cost over M=343 samples per region.
    """
    R_mid_b = R_mid.unsqueeze(-1)

    def residual_fn(u, lo, hi):
        x = u_to_x(u, lo, hi)
        cR = x[:, 0:1]
        cE = x[:, 1:2]
        sR = x[:, 2:3].abs() + 1e-6
        sE = x[:, 3:4].abs() + 1e-6
        gR = x[:, 4:5].abs() + 1e-6
        gE = x[:, 5:6].abs() + 1e-6
        eta_v = x[:, 6:7].clamp(0.0, 1.0)
        area = x[:, 7:8]
        bg0 = x[:, 8:9]
        bg1 = x[:, 9:10]
        dR = R_grid - cR
        dE = E_grid - cE
        G = torch.exp(-0.5 * (dR * dR) / (sR * sR + 1e-30)
                       -0.5 * (dE * dE) / (sE * sE + 1e-30)) \
            / (_2PI * sR * sE + 1e-30)
        L = 1.0 / (_PI2 * gR * gE
                    * (1.0 + (dR * dR) / (gR * gR + 1e-30))
                    * (1.0 + (dE * dE) / (gE * gE + 1e-30)))
        peak = area * (eta_v * L + (1.0 - eta_v) * G)
        return peak + bg0 + bg1 * (R_grid - R_mid_b) - I_block

    def jacobian_fn(u, lo, hi):
        x = u_to_x(u, lo, hi)
        cR_raw = x[:, 0:1]; cE_raw = x[:, 1:2]
        sR_raw = x[:, 2:3]; sE_raw = x[:, 3:4]
        gR_raw = x[:, 4:5]; gE_raw = x[:, 5:6]
        eta_v_raw = x[:, 6:7]
        area = x[:, 7:8]; bg0 = x[:, 8:9]; bg1 = x[:, 9:10]
        sR = sR_raw.abs() + 1e-6
        sE = sE_raw.abs() + 1e-6
        gR = gR_raw.abs() + 1e-6
        gE = gE_raw.abs() + 1e-6
        eta_v = eta_v_raw.clamp(0.0, 1.0)

        dR = R_grid - cR_raw
        dE = E_grid - cE_raw
        s2R = sR * sR; s2E = sE * sE
        g2R = gR * gR; g2E = gE * gE
        dR2 = dR * dR; dE2 = dE * dE

        # Gaussian factor
        G = torch.exp(-0.5 * dR2 / s2R - 0.5 * dE2 / s2E) \
            / (_2PI * sR * sE)
        # Lorentzian per-axis
        denomR = 1.0 + dR2 / g2R
        denomE = 1.0 + dE2 / g2E
        L = 1.0 / (_PI2 * gR * gE * denomR * denomE)
        peak = area * (eta_v * L + (1.0 - eta_v) * G)
        r = peak + bg0 + bg1 * (R_grid - R_mid_b) - I_block

        # ∂G/∂cR = G · dR / sR², ∂G/∂cE = G · dE / sE²
        dGdcR = G * (dR / s2R)
        dGdcE = G * (dE / s2E)
        # ∂L/∂cR: L · 2 dR / (gR² · denomR)
        dLdcR = L * (2.0 * dR / (g2R * denomR))
        dLdcE = L * (2.0 * dE / (g2E * denomE))
        dr_dcR = area * (eta_v * dLdcR + (1.0 - eta_v) * dGdcR)
        dr_dcE = area * (eta_v * dLdcE + (1.0 - eta_v) * dGdcE)

        # ∂G/∂σ_R = G · (dR² - σ_R²) / (σ_R · σ_R²)
        dGdsR = G * ((dR2 - s2R) / (s2R * sR))
        dGdsE = G * ((dE2 - s2E) / (s2E * sE))
        dr_dsR = area * (1.0 - eta_v) * dGdsR
        dr_dsE = area * (1.0 - eta_v) * dGdsE

        # ∂L/∂γ_R: -L/γ_R + L · 2dR²/(γ_R³ · denomR)
        #        = L · ((2dR²/(γ_R² · denomR) - 1) / γ_R)
        # equivalent form: L · (dR² - γ_R²·denomR) ... easier numerically:
        dLdgR = L * ((2.0 * dR2 / (g2R * denomR) - 1.0) / gR)
        dLdgE = L * ((2.0 * dE2 / (g2E * denomE) - 1.0) / gE)
        dr_dgR = area * eta_v * dLdgR
        dr_dgE = area * eta_v * dLdgE

        # ∂peak/∂η_v = area · (L - G)
        dr_dev = area * (L - G)
        # ∂peak/∂A = η_v L + (1-η_v) G
        dr_dA = eta_v * L + (1.0 - eta_v) * G
        # bg derivatives
        dr_dbg0 = torch.ones_like(R_grid)
        dr_dbg1 = R_grid - R_mid_b

        J_x = torch.stack([dr_dcR, dr_dcE, dr_dsR, dr_dsE, dr_dgR, dr_dgE,
                            dr_dev, dr_dA, dr_dbg0, dr_dbg1], dim=-1)

        span = hi - lo
        sig_u = torch.sigmoid(u)
        dxdu = span * sig_u * (1.0 - sig_u)
        sign_sR = torch.where(sR_raw >= 0, 1.0, -1.0)
        sign_sE = torch.where(sE_raw >= 0, 1.0, -1.0)
        sign_gR = torch.where(gR_raw >= 0, 1.0, -1.0)
        sign_gE = torch.where(gE_raw >= 0, 1.0, -1.0)
        dxdu_eff = dxdu.clone()
        dxdu_eff[:, 2:3] = dxdu[:, 2:3] * sign_sR
        dxdu_eff[:, 3:4] = dxdu[:, 3:4] * sign_sE
        dxdu_eff[:, 4:5] = dxdu[:, 4:5] * sign_gR
        dxdu_eff[:, 5:6] = dxdu[:, 5:6] * sign_gE
        J = J_x * dxdu_eff.unsqueeze(1)
        return r, J

    return residual_fn, jacobian_fn


def _residual_fn_factory(
    R_grid: torch.Tensor,        # [B, M]  R coordinates flattened over (R, η)
    E_grid: torch.Tensor,        # [B, M]  η coordinates
    R_mid: torch.Tensor,         # [B]     window mid-R for bg slope reference
    I_block: torch.Tensor,       # [B, M]
):
    R_mid_b = R_mid.unsqueeze(-1)
    def residual_fn(u: torch.Tensor, lo: torch.Tensor, hi: torch.Tensor) -> torch.Tensor:
        x = u_to_x(u, lo, hi)               # [B, 10]
        cR     = x[:, 0:1]
        cE     = x[:, 1:2]
        sR     = x[:, 2:3].abs() + 1e-6
        sE     = x[:, 3:4].abs() + 1e-6
        gR     = x[:, 4:5].abs() + 1e-6
        gE     = x[:, 5:6].abs() + 1e-6
        eta_v  = x[:, 6:7].clamp(0.0, 1.0)
        area   = x[:, 7:8]
        bg0    = x[:, 8:9]
        bg1    = x[:, 9:10]
        model = _pV_2d(R_grid, E_grid, cR, cE, sR, sE, gR, gE, eta_v, area) \
            + bg0 + bg1 * (R_grid - R_mid_b)
        return model - I_block
    return residual_fn


@dataclass
class BatchedFits2D:
    R_fit: torch.Tensor          # [B] absolute R (px)
    eta_deg: torch.Tensor        # [B] eta-bin center (deg)
    eta_offset: torch.Tensor     # [B] fitted η offset within the window (deg)
    ring_idx: torch.Tensor
    sigma_R: torch.Tensor
    sigma_eta: torch.Tensor
    gamma_R: torch.Tensor
    gamma_eta: torch.Tensor
    eta_v: torch.Tensor
    area: torch.Tensor
    snr: torch.Tensor
    rms: torch.Tensor
    rc: torch.Tensor


def fit_cake_per_ring_2d(
    cake_intensity: torch.Tensor,        # [n_R, n_eta]
    R_centers: torch.Tensor,             # [n_R]
    eta_centers: torch.Tensor,           # [n_eta]
    ring_R_ideal_px: torch.Tensor,       # [n_rings]
    *,
    half_window_R_px: float = 8.0,
    half_window_eta_deg: float = 3.0,    # ±3° of η around each bin center
    max_iter: int = 50,
    snr_min: float = 2.0,
    snip_window: int = 0,                # 0 = no SNIP
    dtype=torch.float64,
    device: str = "cpu",
    verbose: bool = False,
) -> BatchedFits2D:
    """2-D batched pV fit across (ring × η-bin) regions.

    Each region's window is ``n_R_win × n_eta_win`` samples.  Returns a
    BatchedFits2D analogous to the 1-D version's BatchedFits, with
    ``R_fit`` in cake R coordinates (px) and a fitted η-offset within the
    window (used by callers that want to invert (R, η) back to (Y, Z)).
    """
    cake_intensity = cake_intensity.to(dtype=dtype, device=device)
    R_centers = R_centers.to(dtype=dtype, device=device)
    eta_centers = eta_centers.to(dtype=dtype, device=device)
    ring_R_ideal_px = ring_R_ideal_px.to(dtype=dtype, device=device)
    n_R_full, n_eta_full = cake_intensity.shape
    n_rings = ring_R_ideal_px.shape[0]

    dR = float(R_centers[1] - R_centers[0])
    dE = float(eta_centers[1] - eta_centers[0])
    n_R_win = max(7, int(round(2.0 * half_window_R_px / dR)))
    if n_R_win % 2 == 0: n_R_win += 1
    n_E_win = max(3, int(round(2.0 * half_window_eta_deg / dE)))
    if n_E_win % 2 == 0: n_E_win += 1

    # For each (ring, eta_bin), build a [n_R_win, n_E_win] window into
    # cake_intensity.  The eta indices wrap around for bins near ±180°.
    R_win_per_ring: List[torch.Tensor] = []   # each [n_R_win]
    R_lo_idx: List[int] = []
    rings_kept: List[int] = []
    for k in range(n_rings):
        R_id = float(ring_R_ideal_px[k])
        if R_id < float(R_centers[0]) or R_id > float(R_centers[-1]):
            continue
        c_idx = int(torch.argmin((R_centers - R_id).abs()))
        lo_i = max(0, c_idx - n_R_win // 2)
        hi_i = lo_i + n_R_win
        if hi_i > n_R_full:
            hi_i = n_R_full
            lo_i = hi_i - n_R_win
        R_win_per_ring.append(R_centers[lo_i:hi_i])
        R_lo_idx.append(lo_i)
        rings_kept.append(k)

    if not R_win_per_ring:
        empty = torch.empty(0, dtype=dtype, device=device)
        empty_l = torch.empty(0, dtype=torch.long, device=device)
        return BatchedFits2D(empty, empty, empty, empty_l, empty, empty,
                              empty, empty, empty, empty, empty, empty, empty_l)

    n_rings_kept = len(rings_kept)

    # Eta windows: build per-region eta indices [B, n_E_win].  Wrap around.
    half_E = n_E_win // 2
    eta_idx_offsets = torch.arange(-half_E, half_E + 1, device=device)
    # Wrap eta_centers so each bin gets a window centered on it.
    # Build per (ring × eta_bin) index map.
    R_pieces, E_pieces, I_pieces = [], [], []
    eta_arr_pieces = []
    eta_offset_pieces = []
    ring_arr_pieces = []
    R_ideal_pieces = []
    R_mid_pieces = []
    for ki, k_ring in enumerate(rings_kept):
        lo_i = R_lo_idx[ki]
        hi_i = lo_i + n_R_win
        # cake[lo_i:hi_i, :] : [n_R_win, n_eta_full]
        R_win = R_centers[lo_i:hi_i]                          # [n_R_win]
        # Per η-bin slab: roll eta axis so each bin has its own window.
        for j in range(n_eta_full):
            j_idxs = (j + eta_idx_offsets) % n_eta_full
            E_win = eta_centers[j_idxs]                        # [n_E_win]
            I_win = cake_intensity[lo_i:hi_i][:, j_idxs]       # [n_R_win, n_E_win]
            # Flatten into [n_R_win * n_E_win] — order: outer R, inner η.
            R_grid = R_win.unsqueeze(-1).expand(-1, n_E_win).reshape(-1)
            E_grid = E_win.unsqueeze(0).expand(n_R_win, -1).reshape(-1)
            I_flat = I_win.reshape(-1)
            R_pieces.append(R_grid - float(ring_R_ideal_px[k_ring]))   # centered R
            E_pieces.append(E_grid - float(eta_centers[j]))             # centered η
            I_pieces.append(I_flat)
            eta_arr_pieces.append(float(eta_centers[j]))
            ring_arr_pieces.append(k_ring)
            R_ideal_pieces.append(float(ring_R_ideal_px[k_ring]))
            R_mid_pieces.append(0.5 * (float(R_win[0]) + float(R_win[-1]))
                                 - float(ring_R_ideal_px[k_ring]))

    B = len(R_pieces)
    M = n_R_win * n_E_win
    if verbose:
        print(f"  [pV-2D-batched] B={B} regions, window={n_R_win}×{n_E_win} bins, "
              f"M={M} samples", flush=True)

    R_centered = torch.stack(R_pieces, dim=0)   # [B, M]
    E_centered = torch.stack(E_pieces, dim=0)   # [B, M]
    I_block = torch.stack(I_pieces, dim=0)      # [B, M]
    if snip_window > 0:
        from .snip import subtract_snip_background
        I_block = subtract_snip_background(
            I_block, window_max=snip_window, use_lls=True, floor_at_zero=True,
        )
    eta_arr = torch.tensor(eta_arr_pieces, dtype=dtype, device=device)
    ring_arr = torch.tensor(ring_arr_pieces, dtype=torch.long, device=device)
    R_ideal_arr = torch.tensor(R_ideal_pieces, dtype=dtype, device=device)
    R_mid_arr = torch.tensor(R_mid_pieces, dtype=dtype, device=device)

    # Init.
    I_min = I_block.min(dim=1).values
    I_max = I_block.max(dim=1).values
    bg0_init = I_min
    area_init = (I_max - I_min) * 1.5
    half_R = max(half_window_R_px, 1.0)
    half_E_d = max(half_window_eta_deg, 0.5)
    x0 = torch.stack([
        torch.zeros(B, dtype=dtype, device=device),     # cR
        torch.zeros(B, dtype=dtype, device=device),     # cE
        torch.full((B,), 0.8, dtype=dtype, device=device),  # σ_R
        torch.full((B,), 0.5 * half_E_d, dtype=dtype, device=device),  # σ_η
        torch.full((B,), 0.5, dtype=dtype, device=device),  # γ_R
        torch.full((B,), 0.5 * half_E_d, dtype=dtype, device=device),  # γ_η
        torch.full((B,), 0.5, dtype=dtype, device=device),  # η_v
        area_init,
        bg0_init,
        torch.zeros(B, dtype=dtype, device=device),     # bg1
    ], dim=-1)
    lo = torch.stack([
        torch.full((B,), -half_R, dtype=dtype, device=device),
        torch.full((B,), -half_E_d, dtype=dtype, device=device),
        torch.full((B,), 0.05, dtype=dtype, device=device),
        torch.full((B,), 0.05, dtype=dtype, device=device),
        torch.full((B,), 0.05, dtype=dtype, device=device),
        torch.full((B,), 0.05, dtype=dtype, device=device),
        torch.full((B,), 0.0, dtype=dtype, device=device),
        torch.full((B,), 0.0, dtype=dtype, device=device),
        I_min - 10.0 * (I_max - I_min + 1.0).abs(),
        torch.full((B,), -1e6, dtype=dtype, device=device),
    ], dim=-1)
    hi = torch.stack([
        torch.full((B,), half_R, dtype=dtype, device=device),
        torch.full((B,), half_E_d, dtype=dtype, device=device),
        torch.full((B,), half_R, dtype=dtype, device=device),
        torch.full((B,), 4.0 * half_E_d, dtype=dtype, device=device),
        torch.full((B,), half_R, dtype=dtype, device=device),
        torch.full((B,), 4.0 * half_E_d, dtype=dtype, device=device),
        torch.full((B,), 1.0, dtype=dtype, device=device),
        torch.full((B,), 100.0, dtype=dtype, device=device) * (I_max - I_min + 1.0).abs(),
        I_max + 10.0 * (I_max - I_min + 1.0).abs(),
        torch.full((B,), 1e6, dtype=dtype, device=device),
    ], dim=-1)

    # Analytic Jacobian (10-param 2-D pV).  Closed-form derivatives for
    # the original Wertheim pV in 2-D + linear-in-R bg.  ~30× faster than
    # the reverse-mode autograd default at our M=343 sample size.
    residual_fn, jacobian_fn = _residual_and_jac_factory(
        R_centered, E_centered, R_mid_arr, I_block,
    )
    cfg = GenericLMConfig(max_iter=max_iter, ftol_rel=1e-8, xtol_rel=1e-8)
    x_final, cost, rc = lm_solve_generic(
        x0, lo, hi, residual_fn=residual_fn, jacobian_fn=jacobian_fn, config=cfg,
    )

    cR     = x_final[:, 0]
    cE     = x_final[:, 1]
    sR     = x_final[:, 2].abs()
    sE     = x_final[:, 3].abs()
    gR     = x_final[:, 4].abs()
    gE     = x_final[:, 5].abs()
    eta_v  = x_final[:, 6].clamp(0.0, 1.0)
    area   = x_final[:, 7]

    R_fit = R_ideal_arr + cR
    eta_offset = cE

    from midas_peakfit.reparam import x_to_u
    u_final = x_to_u(x_final, lo, hi)
    r_final = residual_fn(u_final, lo, hi)
    rms = (r_final * r_final).mean(dim=-1).sqrt()
    snr = (I_max - I_min) / (rms + 1e-12)

    return BatchedFits2D(
        R_fit=R_fit, eta_deg=eta_arr, eta_offset=eta_offset,
        ring_idx=ring_arr,
        sigma_R=sR, sigma_eta=sE,
        gamma_R=gR, gamma_eta=gE,
        eta_v=eta_v, area=area,
        snr=snr, rms=rms, rc=rc,
    )


__all__ = ["BatchedFits2D", "fit_cake_per_ring_2d"]
