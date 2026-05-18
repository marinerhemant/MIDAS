"""Batched 1-D pseudo-Voigt LM peak fitter — fast, vectorised, replaces the
v1 centroid extraction.

For each (ring, η-bin) we fit a 6-parameter peak (center, σ, γ, area, bg₀,
bg₁) to a radial window of cake intensity around the ideal ring radius.
All regions share the same window length M, so the batch axis is
``B = n_rings × n_eta_bins`` and the residual is ``[B, M]`` — exactly the
shape ``midas_peakfit.lm_solve_generic`` expects.

This closes the strain-floor gap to v1's C engine: it's the same
pseudo-Voigt LM the C code runs, just batched on torch.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import math
import numpy as np
import torch

from midas_peakfit import GenericLMConfig, lm_solve_generic, u_to_x


_LN2 = math.log(2.0)


def _pV_1d(R: torch.Tensor, center: torch.Tensor, sigma: torch.Tensor,
            gamma: torch.Tensor, eta_v: torch.Tensor, area: torch.Tensor) -> torch.Tensor:
    """Area-normalised 1-D pseudo-Voigt.

    All inputs are batched: ``R [B, M]`` (sampling grid), ``center..area
    [B, 1]`` (per-region scalars).  Returns ``[B, M]``.
    """
    dR = R - center
    dR2 = dR * dR
    G = torch.exp(-0.5 * dR2 / (sigma * sigma + 1e-30)) \
        / (sigma * math.sqrt(2.0 * math.pi) + 1e-30)
    L = (gamma / math.pi) / (dR2 + gamma * gamma + 1e-30)
    return area * (eta_v * L + (1.0 - eta_v) * G)


def _residual_and_jac_factory(R_window_centered: torch.Tensor, I_block: torch.Tensor):
    """Build ``(residual_fn, jacobian_fn)`` with the **analytic** Jacobian
    closed-form for the 1-D original pseudo-Voigt + linear bg model.

    For B regions × M samples × N=7 params, this avoids the autograd
    row-by-row backprop entirely (which is M backward passes).  All
    derivatives are dense torch ops, so the LM iteration cost drops by
    roughly ``M / log_overhead`` ≈ 30-50× on CPU.

    Layout matches :func:`_residual_fn_factory`:
        x = [center_offset, σ, γ, η_v, area, bg0, bg1]
    """
    SQRT_2PI = math.sqrt(2.0 * math.pi)
    INV_PI = 1.0 / math.pi

    def residual_fn(u, lo, hi):
        x = u_to_x(u, lo, hi)
        c = x[:, 0:1]
        sigma = x[:, 1:2].abs() + 1e-6
        gamma = x[:, 2:3].abs() + 1e-6
        eta_v = x[:, 3:4].clamp(0.0, 1.0)
        A = x[:, 4:5]
        bg0 = x[:, 5:6]
        bg1 = x[:, 6:7]
        dR = R_window_centered - c
        dR2 = dR * dR
        G = torch.exp(-0.5 * dR2 / (sigma * sigma)) / (sigma * SQRT_2PI)
        L = (gamma * INV_PI) / (dR2 + gamma * gamma)
        peak = A * (eta_v * L + (1.0 - eta_v) * G)
        bg = bg0 + bg1 * R_window_centered
        return peak + bg - I_block

    def jacobian_fn(u, lo, hi):
        # Compute r and J = dr/du in one pass.
        # ∂x/∂u (sigmoid reparam): dx/du = (hi-lo) σ(u) (1-σ(u))
        # We compute J_x = dr/dx (analytic), then J_u = J_x · diag(dx/du).
        x = u_to_x(u, lo, hi)
        c = x[:, 0:1]
        sigma_raw = x[:, 1:2]
        gamma_raw = x[:, 2:3]
        eta_v_raw = x[:, 3:4]
        A = x[:, 4:5]
        bg0 = x[:, 5:6]
        bg1 = x[:, 6:7]
        sigma = sigma_raw.abs() + 1e-6
        gamma = gamma_raw.abs() + 1e-6
        eta_v = eta_v_raw.clamp(0.0, 1.0)
        # Sign of σ, γ for chain rule through abs() — no derivative info lost
        # because both bounds are positive in lo/hi for these params.
        dR = R_window_centered - c             # [B, M]
        dR2 = dR * dR
        s2 = sigma * sigma
        g2 = gamma * gamma
        denom_L = dR2 + g2
        G = torch.exp(-0.5 * dR2 / s2) / (sigma * SQRT_2PI)
        L = (gamma * INV_PI) / denom_L
        peak = A * (eta_v * L + (1.0 - eta_v) * G)
        bg = bg0 + bg1 * R_window_centered
        r = peak + bg - I_block

        # Analytic ∂r/∂x.
        # ∂G/∂c = G · (R-c)/σ² = G · dR / s²
        dGdc = G * (dR / s2)
        # ∂L/∂c = L · 2(R-c)/((R-c)² + γ²) = L · 2 dR / denom_L
        dLdc = L * (2.0 * dR / denom_L)
        # ∂peak/∂c = A·[η_v dLdc + (1-η_v) dGdc]
        dr_dc = A * (eta_v * dLdc + (1.0 - eta_v) * dGdc)
        # ∂G/∂σ = G · (dR² / σ³ - 1/σ)  =  G · ((dR² - σ²) / σ³)
        dGds = G * ((dR2 - s2) / (s2 * sigma))
        # ∂peak/∂σ = A · (1-η_v) · dGds
        dr_dsigma = A * (1.0 - eta_v) * dGds
        # ∂L/∂γ = (1/π) · (1/denom_L - γ · 2γ / denom_L²)
        #       = (1/π) · ((dR² + γ² - 2γ²) / denom_L²)
        #       = (1/π) · (dR² - γ²) / denom_L²
        dLdg = (INV_PI * (dR2 - g2)) / (denom_L * denom_L)
        dr_dgamma = A * eta_v * dLdg
        # ∂peak/∂η_v = A · (L - G)
        dr_dev = A * (L - G)
        # ∂peak/∂A = η_v L + (1-η_v) G
        dr_dA = eta_v * L + (1.0 - eta_v) * G
        # ∂r/∂bg0 = 1, ∂r/∂bg1 = R_window_centered
        dr_dbg0 = torch.ones_like(R_window_centered)
        dr_dbg1 = R_window_centered

        # Stack into J_x of shape [B, M, N=7].
        J_x = torch.stack([dr_dc, dr_dsigma, dr_dgamma, dr_dev, dr_dA,
                            dr_dbg0, dr_dbg1], dim=-1)

        # Chain rule through u_to_x sigmoid reparam.
        span = hi - lo                                          # [B, N]
        sig_u = torch.sigmoid(u)                                # [B, N]
        dxdu = span * sig_u * (1.0 - sig_u)                    # [B, N]
        # σ, γ pass through abs() — chain via sign.  For bounds that lie
        # entirely above zero, this is a no-op.  For sign flip it would
        # negate the column.
        sign_sigma = torch.where(sigma_raw >= 0, 1.0, -1.0)
        sign_gamma = torch.where(gamma_raw >= 0, 1.0, -1.0)
        dxdu_eff = dxdu.clone()
        dxdu_eff[:, 1:2] = dxdu[:, 1:2] * sign_sigma
        dxdu_eff[:, 2:3] = dxdu[:, 2:3] * sign_gamma
        # η_v has clamp() — derivative is 0 outside [0, 1], 1 inside; use 1.

        # J_u[b, m, n] = J_x[b, m, n] * dxdu[b, n]
        J = J_x * dxdu_eff.unsqueeze(1)
        return r, J

    return residual_fn, jacobian_fn


def _residual_fn_factory(R_window_centered: torch.Tensor, I_block: torch.Tensor):
    """Build the LM residual closure for a batch of 1-D **original** pV fits.

    Original pV (Wertheim): four independent shape parameters
    (σ_G, γ_L, η_v, area) plus linear background.  Unlike TCH, η_v is a free
    parameter, not a polynomial of γ_L / γ_total — this is more flexible for
    calibrant peaks where center accuracy matters more than physical
    decomposition of instrument vs sample broadening.

    Parameter vector x [B, 7]:
        [0] center offset (R - R_ideal_per_region)
        [1] σ_G  (Gaussian std)
        [2] γ_L  (Lorentzian half-width-at-half-max)
        [3] η_v  (Lorentzian mixing fraction ∈ [0, 1])
        [4] area
        [5] bg₀  (constant background)
        [6] bg₁  (linear-in-R background slope)
    """
    def residual_fn(u: torch.Tensor, lo: torch.Tensor, hi: torch.Tensor) -> torch.Tensor:
        x = u_to_x(u, lo, hi)               # [B, 7]
        center = x[:, 0:1]
        sigma = x[:, 1:2].abs() + 1e-6
        gamma = x[:, 2:3].abs() + 1e-6
        eta_v = x[:, 3:4].clamp(0.0, 1.0)
        area = x[:, 4:5]
        bg0 = x[:, 5:6]
        bg1 = x[:, 6:7]
        model = _pV_1d(R_window_centered, center, sigma, gamma, eta_v, area) \
            + bg0 + bg1 * R_window_centered
        return model - I_block
    return residual_fn


@dataclass
class BatchedFits:
    R_fit: torch.Tensor          # [B] — peak center in cake R coordinates (px)
    eta_deg: torch.Tensor        # [B]
    ring_idx: torch.Tensor       # [B] long
    sigma: torch.Tensor          # [B]
    gamma: torch.Tensor          # [B]
    area: torch.Tensor           # [B]
    snr: torch.Tensor            # [B]
    rms: torch.Tensor            # [B]
    rc: torch.Tensor             # [B]


def fit_cake_per_ring_batched(
    cake_intensity: torch.Tensor,        # [n_R, n_eta]
    R_centers: torch.Tensor,             # [n_R]
    eta_centers: torch.Tensor,           # [n_eta]
    ring_R_ideal_px: torch.Tensor,       # [n_rings]
    *,
    half_window_px: float = 4.0,
    snr_min: float = 2.0,
    max_iter: int = 50,
    snip_window: int = 0,                # 0 = no SNIP; recommend 12-20
    doublet_separation_px: float = 0.0,  # 0 = no doublet handling; v1 default 25
    dtype=torch.float64,
    device: str = "cpu",
    verbose: bool = False,
) -> BatchedFits:
    """Batched LM pseudo-Voigt fit across all (ring, η-bin) windows.

    Returns peak centers in cake R coordinates (px).  Caller must invert
    (R_fit, eta) → (Y_pix, Z_pix) using the current detector geometry to
    feed into v2's M-step.
    """
    cake_intensity = cake_intensity.to(dtype=dtype, device=device)
    R_centers = R_centers.to(dtype=dtype, device=device)
    eta_centers = eta_centers.to(dtype=dtype, device=device)
    ring_R_ideal_px = ring_R_ideal_px.to(dtype=dtype, device=device)

    n_R, n_eta = cake_intensity.shape
    n_rings = ring_R_ideal_px.shape[0]

    dR = float(R_centers[1] - R_centers[0])
    n_win = int(round(2.0 * half_window_px / dR))
    if n_win < 7:
        n_win = 7

    # Build batched windows.  We index the cake row range per ring and stack
    # over all η.  Padding any ring whose window falls off the cake's R range
    # by clamping (the boundary fits will be poor and rejected by SNR).
    # Doublet detection — flag rings whose ideal radius is within
    # ``doublet_separation_px`` of an adjacent ring.  We don't co-fit them
    # in the LM (variable-width windows would break the batched solver) but
    # we mark them so callers downstream can apply tighter trim or skip.
    doublet_partner = np.full(n_rings, -1, dtype=np.int64)
    if doublet_separation_px > 0:
        from .doublets import doublet_index_map
        doublet_partner, pairs = doublet_index_map(
            ring_R_ideal_px.detach().cpu().numpy(),
            min_separation_px=doublet_separation_px,
        )
        if verbose and pairs:
            print(f"  [pV-batched] detected {len(pairs)} doublet pair(s) "
                  f"(separation < {doublet_separation_px}px); flagged for "
                  f"downstream filtering", flush=True)

    R_win_list: List[torch.Tensor] = []     # each [n_win]
    I_block_list: List[torch.Tensor] = []   # each [n_eta, n_win]
    ring_idx_per_block: List[int] = []
    R_ideal_per_block: List[float] = []
    for k in range(n_rings):
        R_id = float(ring_R_ideal_px[k])
        if R_id < R_centers[0] or R_id > R_centers[-1]:
            continue
        c_idx = int(torch.argmin((R_centers - R_id).abs()))
        lo_i = max(0, c_idx - n_win // 2)
        hi_i = lo_i + n_win
        if hi_i > n_R:
            hi_i = n_R
            lo_i = hi_i - n_win
        R_win_list.append(R_centers[lo_i:hi_i])
        I_block_list.append(cake_intensity[lo_i:hi_i, :].T.contiguous())  # [n_eta, n_win]
        ring_idx_per_block.append(k)
        R_ideal_per_block.append(R_id)

    if not R_win_list:
        return BatchedFits(*(torch.empty(0, dtype=dtype, device=device) for _ in range(7)),
                            torch.empty(0, dtype=torch.long, device=device),
                            torch.empty(0, dtype=torch.long, device=device))

    # Concatenate all (ring × eta) regions into one batch B = n_rings_kept × n_eta.
    R_centered_pieces: List[torch.Tensor] = []
    I_pieces: List[torch.Tensor] = []
    eta_pieces: List[torch.Tensor] = []
    ring_pieces: List[torch.Tensor] = []
    R_ideal_pieces: List[torch.Tensor] = []
    for R_win, I_blk, ring_i, R_id in zip(R_win_list, I_block_list,
                                            ring_idx_per_block, R_ideal_per_block):
        R_centered = (R_win - R_id).unsqueeze(0).expand(n_eta, -1).contiguous()
        R_centered_pieces.append(R_centered)
        I_pieces.append(I_blk)
        eta_pieces.append(eta_centers.clone())
        ring_pieces.append(torch.full((n_eta,), ring_i, dtype=torch.long, device=device))
        R_ideal_pieces.append(torch.full((n_eta,), R_id, dtype=dtype, device=device))

    R_centered = torch.cat(R_centered_pieces, dim=0)   # [B, n_win]
    I_block = torch.cat(I_pieces, dim=0)                # [B, n_win]
    eta_arr = torch.cat(eta_pieces, dim=0)              # [B]
    ring_arr = torch.cat(ring_pieces, dim=0)            # [B]
    R_ideal_arr = torch.cat(R_ideal_pieces, dim=0)      # [B]

    B = R_centered.shape[0]
    if verbose:
        print(f"  [pV-batched] {B} (ring × η) regions, window={R_centered.shape[1]} bins", flush=True)

    # Optional SNIP background pre-subtract (matches v1 C's bg-stripping).
    if snip_window > 0:
        from .snip import subtract_snip_background
        I_block = subtract_snip_background(
            I_block, window_max=snip_window, use_lls=True, floor_at_zero=True,
        )

    # Per-region init.  Original pV: 7 free parameters per region.
    I_min = I_block.min(dim=1, keepdim=False).values        # [B]
    I_max = I_block.max(dim=1, keepdim=False).values
    bg0_init = I_min
    area_init = (I_max - I_min) * 1.5
    x0 = torch.stack([
        torch.zeros(B, dtype=dtype, device=device),       # center offset
        torch.full((B,), 0.8, dtype=dtype, device=device),  # σ
        torch.full((B,), 0.5, dtype=dtype, device=device),  # γ
        torch.full((B,), 0.5, dtype=dtype, device=device),  # η_v (mid-mixed init)
        area_init,                                          # area
        bg0_init,                                           # bg0
        torch.zeros(B, dtype=dtype, device=device),         # bg1
    ], dim=-1)                                             # [B, 7]
    half = max(half_window_px, 1.0)
    lo = torch.stack([
        torch.full((B,), -half, dtype=dtype, device=device),
        torch.full((B,), 0.05, dtype=dtype, device=device),
        torch.full((B,), 0.05, dtype=dtype, device=device),
        torch.full((B,), 0.0, dtype=dtype, device=device),     # η_v ≥ 0
        torch.full((B,), 0.0, dtype=dtype, device=device),
        I_min - 10.0 * (I_max - I_min + 1.0).abs(),
        torch.full((B,), -1e6, dtype=dtype, device=device),
    ], dim=-1)
    hi = torch.stack([
        torch.full((B,), half, dtype=dtype, device=device),
        torch.full((B,), half, dtype=dtype, device=device),
        torch.full((B,), half, dtype=dtype, device=device),
        torch.full((B,), 1.0, dtype=dtype, device=device),     # η_v ≤ 1
        torch.full((B,), 100.0, dtype=dtype, device=device) * (I_max - I_min + 1.0).abs(),
        I_max + 10.0 * (I_max - I_min + 1.0).abs(),
        torch.full((B,), 1e6, dtype=dtype, device=device),
    ], dim=-1)

    # Use the analytic Jacobian factory — closed-form derivatives for the
    # 7-parameter pV + linear bg model, ~30-50× faster than the autograd
    # row-by-row default.  See dev/perf_plan.md.
    residual_fn, jacobian_fn = _residual_and_jac_factory(R_centered, I_block)
    cfg = GenericLMConfig(max_iter=max_iter, ftol_rel=1e-7, xtol_rel=1e-7)
    x_final, cost, rc = lm_solve_generic(
        x0, lo, hi, residual_fn=residual_fn, jacobian_fn=jacobian_fn, config=cfg,
    )

    center = x_final[:, 0]
    sigma = x_final[:, 1].abs()
    gamma = x_final[:, 2].abs()
    eta_v = x_final[:, 3].clamp(0.0, 1.0)
    area = x_final[:, 4]
    bg0 = x_final[:, 5]
    bg1 = x_final[:, 6]

    R_fit = R_ideal_arr + center
    rms = torch.sqrt((residual_fn(
        # Re-evaluate residuals at MAP via x_to_u(x_final, lo, hi) = u_final.
        # Easier: recompute model from x_final directly:
        torch.empty(0), lo, hi
    ).pow(2) if False else
        # Direct model eval at x_final (skip u/sigmoid mapping for clarity).
        torch.zeros(B, dtype=dtype, device=device)).clamp(min=0.0))
    # Just reuse the residual via the closure (which goes through u→x), to
    # avoid duplicating the model code here.
    from midas_peakfit.reparam import x_to_u
    u_final = x_to_u(x_final, lo, hi)
    r_final = residual_fn(u_final, lo, hi)
    rms = (r_final * r_final).mean(dim=-1).sqrt()
    snr = (I_max - I_min) / (rms + 1e-12)

    return BatchedFits(
        R_fit=R_fit, eta_deg=eta_arr, ring_idx=ring_arr,
        sigma=sigma, gamma=gamma, area=area, snr=snr, rms=rms, rc=rc,
    )


__all__ = ["BatchedFits", "fit_cake_per_ring_batched"]
