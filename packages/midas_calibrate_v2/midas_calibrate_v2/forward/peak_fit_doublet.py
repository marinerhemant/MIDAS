"""Batched 2-peak pseudo-Voigt LM for doublet ring co-fitting.

When two rings sit within ``DoubletSeparation`` of each other (default
25 px in v1 C), fitting them as independent 1-peak Wertheim pV models
runs into label-swap degeneracy: the LM may converge with the inner
ring's center inside the outer ring's window or vice versa.

This module fits each doublet pair once with a shared 2-peak model:

    I(R) = bg₀ + bg₁(R - R_mid)
        + A₁ · pV(R; c₁, σ₁, γ₁, η_v1)
        + A₂ · pV(R; c₂, σ₂, γ₂, η_v2)

13 parameters per region (vs 7 for the singleton pV).  Standard
batched-LM via ``midas_peakfit`` with an analytic Jacobian.

Used by callers (e.g. ``pipelines.single_pv``) in a two-pass pattern:
fit singletons in one batched call, doublets in another, merge results.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import torch

from midas_peakfit import GenericLMConfig, lm_solve_generic, u_to_x


_SQRT_2PI = math.sqrt(2.0 * math.pi)
_INV_PI = 1.0 / math.pi


@dataclass
class DoubletFits:
    """Per (ring-pair × η-bin) doublet fit result.  Both peaks emitted
    independently so the caller can append them to its singleton FitDataset.
    """
    R_fit_lo: torch.Tensor       # [B] center of the lower-radius peak (px)
    R_fit_hi: torch.Tensor       # [B] center of the upper-radius peak (px)
    eta_deg: torch.Tensor        # [B]
    ring_idx_lo: torch.Tensor    # [B] long
    ring_idx_hi: torch.Tensor    # [B] long
    sigma_lo: torch.Tensor
    sigma_hi: torch.Tensor
    gamma_lo: torch.Tensor
    gamma_hi: torch.Tensor
    eta_v_lo: torch.Tensor
    eta_v_hi: torch.Tensor
    area_lo: torch.Tensor
    area_hi: torch.Tensor
    snr: torch.Tensor
    rms: torch.Tensor
    rc: torch.Tensor


def _pV_1d(R: torch.Tensor, c: torch.Tensor, sigma: torch.Tensor,
            gamma: torch.Tensor, eta_v: torch.Tensor, A: torch.Tensor
            ) -> torch.Tensor:
    """Original Wertheim pV.  All ``[B, *]`` broadcast-compatible."""
    dR = R - c
    G = torch.exp(-0.5 * (dR * dR) / (sigma * sigma)) / (sigma * _SQRT_2PI)
    L = (gamma * _INV_PI) / ((dR * dR) + gamma * gamma)
    return A * (eta_v * L + (1.0 - eta_v) * G)


def _residual_and_jac_factory_doublet(
    R_window_centered: torch.Tensor, I_block: torch.Tensor,
    R_lo_offset: torch.Tensor, R_hi_offset: torch.Tensor,
):
    """Build ``(residual_fn, jacobian_fn)`` for the 13-param 2-peak model.

    Parameter vector x [B, 13]:
        [0] center_offset_1     (relative to R_lo_seed)
        [1] σ_1
        [2] γ_1
        [3] η_v1
        [4] A_1
        [5] center_offset_2     (relative to R_hi_seed)
        [6] σ_2
        [7] γ_2
        [8] η_v2
        [9] A_2
        [10] bg0
        [11] bg1
        [12] R_mid (passed in via R_window_centered's offset; this slot
                    is unused — kept to match a 13-element layout caller
                    may want to extend)
    R_window_centered: ``[B, M]`` — R coords centred such that 0 is at
    the midpoint of the doublet window.
    R_lo_offset, R_hi_offset: ``[B, 1]`` — seed-to-midpoint offsets so
    each peak's centre fits relative to its own seed.
    """
    R_grid = R_window_centered

    def residual_fn(u, lo, hi):
        x = u_to_x(u, lo, hi)
        c1 = R_lo_offset + x[:, 0:1]
        s1 = x[:, 1:2].abs() + 1e-6
        g1 = x[:, 2:3].abs() + 1e-6
        ev1 = x[:, 3:4].clamp(0.0, 1.0)
        A1 = x[:, 4:5]
        c2 = R_hi_offset + x[:, 5:6]
        s2 = x[:, 6:7].abs() + 1e-6
        g2 = x[:, 7:8].abs() + 1e-6
        ev2 = x[:, 8:9].clamp(0.0, 1.0)
        A2 = x[:, 9:10]
        bg0 = x[:, 10:11]
        bg1 = x[:, 11:12]
        peak1 = _pV_1d(R_grid, c1, s1, g1, ev1, A1)
        peak2 = _pV_1d(R_grid, c2, s2, g2, ev2, A2)
        return peak1 + peak2 + bg0 + bg1 * R_grid - I_block

    def jacobian_fn(u, lo, hi):
        x = u_to_x(u, lo, hi)
        c1 = R_lo_offset + x[:, 0:1]
        s1_raw = x[:, 1:2]
        g1_raw = x[:, 2:3]
        ev1 = x[:, 3:4].clamp(0.0, 1.0)
        A1 = x[:, 4:5]
        c2 = R_hi_offset + x[:, 5:6]
        s2_raw = x[:, 6:7]
        g2_raw = x[:, 7:8]
        ev2 = x[:, 8:9].clamp(0.0, 1.0)
        A2 = x[:, 9:10]
        bg0 = x[:, 10:11]
        bg1 = x[:, 11:12]
        s1 = s1_raw.abs() + 1e-6
        g1 = g1_raw.abs() + 1e-6
        s2 = s2_raw.abs() + 1e-6
        g2 = g2_raw.abs() + 1e-6

        dR1 = R_grid - c1
        dR2 = R_grid - c2
        dR1sq = dR1 * dR1
        dR2sq = dR2 * dR2
        s1sq = s1 * s1; g1sq = g1 * g1
        s2sq = s2 * s2; g2sq = g2 * g2
        denomL1 = dR1sq + g1sq
        denomL2 = dR2sq + g2sq
        G1 = torch.exp(-0.5 * dR1sq / s1sq) / (s1 * _SQRT_2PI)
        L1 = (g1 * _INV_PI) / denomL1
        G2 = torch.exp(-0.5 * dR2sq / s2sq) / (s2 * _SQRT_2PI)
        L2 = (g2 * _INV_PI) / denomL2
        peak1 = A1 * (ev1 * L1 + (1.0 - ev1) * G1)
        peak2 = A2 * (ev2 * L2 + (1.0 - ev2) * G2)
        r = peak1 + peak2 + bg0 + bg1 * R_grid - I_block

        # Derivatives — same formulas as singleton, applied per-peak.
        dG1dc = G1 * (dR1 / s1sq)
        dL1dc = L1 * (2.0 * dR1 / denomL1)
        dr_dc1 = A1 * (ev1 * dL1dc + (1.0 - ev1) * dG1dc)
        dG1ds = G1 * ((dR1sq - s1sq) / (s1sq * s1))
        dr_ds1 = A1 * (1.0 - ev1) * dG1ds
        dL1dg = (_INV_PI * (dR1sq - g1sq)) / (denomL1 * denomL1)
        dr_dg1 = A1 * ev1 * dL1dg
        dr_dev1 = A1 * (L1 - G1)
        dr_dA1 = ev1 * L1 + (1.0 - ev1) * G1

        dG2dc = G2 * (dR2 / s2sq)
        dL2dc = L2 * (2.0 * dR2 / denomL2)
        dr_dc2 = A2 * (ev2 * dL2dc + (1.0 - ev2) * dG2dc)
        dG2ds = G2 * ((dR2sq - s2sq) / (s2sq * s2))
        dr_ds2 = A2 * (1.0 - ev2) * dG2ds
        dL2dg = (_INV_PI * (dR2sq - g2sq)) / (denomL2 * denomL2)
        dr_dg2 = A2 * ev2 * dL2dg
        dr_dev2 = A2 * (L2 - G2)
        dr_dA2 = ev2 * L2 + (1.0 - ev2) * G2

        dr_dbg0 = torch.ones_like(R_grid)
        dr_dbg1 = R_grid

        J_x = torch.stack([
            dr_dc1, dr_ds1, dr_dg1, dr_dev1, dr_dA1,
            dr_dc2, dr_ds2, dr_dg2, dr_dev2, dr_dA2,
            dr_dbg0, dr_dbg1,
        ], dim=-1)

        span = hi - lo
        sig_u = torch.sigmoid(u)
        dxdu = span * sig_u * (1.0 - sig_u)
        sign = torch.where(
            torch.stack([torch.zeros_like(s1_raw),
                         s1_raw, g1_raw, torch.zeros_like(s1_raw),
                         torch.zeros_like(s1_raw),
                         torch.zeros_like(s1_raw),
                         s2_raw, g2_raw,
                         torch.zeros_like(s1_raw),
                         torch.zeros_like(s1_raw),
                         torch.zeros_like(s1_raw),
                         torch.zeros_like(s1_raw)], dim=-1).squeeze(-2) >= 0,
            1.0, -1.0,
        )
        # Simpler: only σ and γ pass through abs — flip sign on those columns.
        dxdu_eff = dxdu.clone()
        dxdu_eff[:, 1:2] = dxdu[:, 1:2] * torch.where(s1_raw >= 0, 1.0, -1.0)
        dxdu_eff[:, 2:3] = dxdu[:, 2:3] * torch.where(g1_raw >= 0, 1.0, -1.0)
        dxdu_eff[:, 6:7] = dxdu[:, 6:7] * torch.where(s2_raw >= 0, 1.0, -1.0)
        dxdu_eff[:, 7:8] = dxdu[:, 7:8] * torch.where(g2_raw >= 0, 1.0, -1.0)

        J = J_x * dxdu_eff.unsqueeze(1)
        return r, J

    return residual_fn, jacobian_fn


def fit_doublet_pairs(
    cake_intensity: torch.Tensor,        # [n_R, n_eta]
    R_centers: torch.Tensor,
    eta_centers: torch.Tensor,
    ring_R_ideal_px: torch.Tensor,       # [n_rings]
    *,
    pair_indices: List[Tuple[int, int]],  # [(i_lo, i_hi), ...] one per doublet
    half_window_px: float = 4.0,
    max_iter: int = 50,
    snr_min: float = 2.0,
    snip_window: int = 0,
    dtype=torch.float64,
    device: str = "cpu",
    verbose: bool = False,
) -> DoubletFits:
    """Batched 2-peak pV fit across all (doublet pair × η-bin) regions.

    All windows are forced to the same M = ``2 * half_window_px / dR + sep``.
    The seed centres are placed at the two ring R_ideal_px values; the LM
    refines small offsets around them.
    """
    cake_intensity = cake_intensity.to(dtype=dtype, device=device)
    R_centers = R_centers.to(dtype=dtype, device=device)
    eta_centers = eta_centers.to(dtype=dtype, device=device)
    ring_R_ideal_px = ring_R_ideal_px.to(dtype=dtype, device=device)
    n_R, n_eta = cake_intensity.shape

    if not pair_indices:
        empty = torch.empty(0, dtype=dtype, device=device)
        empty_l = torch.empty(0, dtype=torch.long, device=device)
        return DoubletFits(*[empty] * 13, empty_l)

    # Pad windows to the longest doublet width.
    dR_bin = float(R_centers[1] - R_centers[0])
    max_sep = max(
        float(ring_R_ideal_px[hi] - ring_R_ideal_px[lo])
        for lo, hi in pair_indices
    )
    n_win = int(round((max_sep + 2.0 * half_window_px) / dR_bin))
    if n_win % 2 == 0:
        n_win += 1
    half_win_R = 0.5 * (n_win - 1) * dR_bin

    R_pieces = []
    I_pieces = []
    eta_pieces = []
    ring_lo_pieces = []
    ring_hi_pieces = []
    R_lo_offset_pieces = []
    R_hi_offset_pieces = []
    for lo_i, hi_i in pair_indices:
        R_lo = float(ring_R_ideal_px[lo_i])
        R_hi = float(ring_R_ideal_px[hi_i])
        R_mid = 0.5 * (R_lo + R_hi)
        c_idx = int(torch.argmin((R_centers - R_mid).abs()))
        lo_idx = max(0, c_idx - n_win // 2)
        hi_idx = lo_idx + n_win
        if hi_idx > n_R:
            hi_idx = n_R
            lo_idx = hi_idx - n_win
        if lo_idx < 0:
            continue
        R_win = R_centers[lo_idx:hi_idx]              # [n_win]
        # Center R_win on R_mid → R_grid in [-half_win_R, +half_win_R]
        R_centered = (R_win - R_mid).unsqueeze(0).expand(n_eta, -1).contiguous()
        I_win = cake_intensity[lo_idx:hi_idx, :].T.contiguous()  # [n_eta, n_win]
        R_pieces.append(R_centered)
        I_pieces.append(I_win)
        eta_pieces.append(eta_centers.clone())
        ring_lo_pieces.append(torch.full((n_eta,), lo_i, dtype=torch.long, device=device))
        ring_hi_pieces.append(torch.full((n_eta,), hi_i, dtype=torch.long, device=device))
        R_lo_offset_pieces.append(torch.full((n_eta, 1), R_lo - R_mid, dtype=dtype, device=device))
        R_hi_offset_pieces.append(torch.full((n_eta, 1), R_hi - R_mid, dtype=dtype, device=device))

    if not R_pieces:
        empty = torch.empty(0, dtype=dtype, device=device)
        empty_l = torch.empty(0, dtype=torch.long, device=device)
        return DoubletFits(*[empty] * 13, empty_l)

    R_centered_all = torch.cat(R_pieces, dim=0)              # [B, n_win]
    I_block = torch.cat(I_pieces, dim=0)
    eta_arr = torch.cat(eta_pieces, dim=0)
    ring_lo = torch.cat(ring_lo_pieces, dim=0)
    ring_hi = torch.cat(ring_hi_pieces, dim=0)
    R_lo_off = torch.cat(R_lo_offset_pieces, dim=0)
    R_hi_off = torch.cat(R_hi_offset_pieces, dim=0)
    B = R_centered_all.shape[0]

    if verbose:
        print(f"  [pV-doublet] {B} (pair × η) regions, window={n_win} bins, "
              f"max sep={max_sep:.2f}px", flush=True)

    if snip_window > 0:
        from .snip import subtract_snip_background
        I_block = subtract_snip_background(
            I_block, window_max=snip_window, use_lls=True, floor_at_zero=True,
        )

    # Init: σ=0.8, γ=0.5, η_v=0.5, area scaled to peak amplitude.
    I_min = I_block.min(dim=1).values
    I_max = I_block.max(dim=1).values
    half = max(half_window_px, 1.0)
    A_init = (I_max - I_min) * 1.5
    x0 = torch.stack([
        torch.zeros(B, dtype=dtype, device=device),       # c1 offset
        torch.full((B,), 0.8, dtype=dtype, device=device),
        torch.full((B,), 0.5, dtype=dtype, device=device),
        torch.full((B,), 0.5, dtype=dtype, device=device),
        A_init,
        torch.zeros(B, dtype=dtype, device=device),       # c2 offset
        torch.full((B,), 0.8, dtype=dtype, device=device),
        torch.full((B,), 0.5, dtype=dtype, device=device),
        torch.full((B,), 0.5, dtype=dtype, device=device),
        A_init,
        I_min,
        torch.zeros(B, dtype=dtype, device=device),
    ], dim=-1)
    lo_b = torch.stack([
        torch.full((B,), -half, dtype=dtype, device=device),
        torch.full((B,), 0.05, dtype=dtype, device=device),
        torch.full((B,), 0.05, dtype=dtype, device=device),
        torch.full((B,), 0.0, dtype=dtype, device=device),
        torch.full((B,), 0.0, dtype=dtype, device=device),
        torch.full((B,), -half, dtype=dtype, device=device),
        torch.full((B,), 0.05, dtype=dtype, device=device),
        torch.full((B,), 0.05, dtype=dtype, device=device),
        torch.full((B,), 0.0, dtype=dtype, device=device),
        torch.full((B,), 0.0, dtype=dtype, device=device),
        I_min - 10.0 * (I_max - I_min + 1.0).abs(),
        torch.full((B,), -1e6, dtype=dtype, device=device),
    ], dim=-1)
    hi_b = torch.stack([
        torch.full((B,), half, dtype=dtype, device=device),
        torch.full((B,), half, dtype=dtype, device=device),
        torch.full((B,), half, dtype=dtype, device=device),
        torch.full((B,), 1.0, dtype=dtype, device=device),
        torch.full((B,), 100.0, dtype=dtype, device=device) * (I_max - I_min + 1.0).abs(),
        torch.full((B,), half, dtype=dtype, device=device),
        torch.full((B,), half, dtype=dtype, device=device),
        torch.full((B,), half, dtype=dtype, device=device),
        torch.full((B,), 1.0, dtype=dtype, device=device),
        torch.full((B,), 100.0, dtype=dtype, device=device) * (I_max - I_min + 1.0).abs(),
        I_max + 10.0 * (I_max - I_min + 1.0).abs(),
        torch.full((B,), 1e6, dtype=dtype, device=device),
    ], dim=-1)

    residual_fn, jacobian_fn = _residual_and_jac_factory_doublet(
        R_centered_all, I_block, R_lo_off, R_hi_off,
    )
    cfg = GenericLMConfig(max_iter=max_iter, ftol_rel=1e-7, xtol_rel=1e-7)
    x_final, cost, rc = lm_solve_generic(
        x0, lo_b, hi_b, residual_fn=residual_fn,
        jacobian_fn=jacobian_fn, config=cfg,
    )

    c1 = x_final[:, 0]
    s1 = x_final[:, 1].abs()
    g1 = x_final[:, 2].abs()
    ev1 = x_final[:, 3].clamp(0.0, 1.0)
    A1 = x_final[:, 4]
    c2 = x_final[:, 5]
    s2 = x_final[:, 6].abs()
    g2 = x_final[:, 7].abs()
    ev2 = x_final[:, 8].clamp(0.0, 1.0)
    A2 = x_final[:, 9]
    R_lo_off_flat = R_lo_off.squeeze(-1)
    R_hi_off_flat = R_hi_off.squeeze(-1)
    # Convert centred offsets back to absolute R_fit.
    R_mid_per = R_centered_all[:, R_centered_all.shape[1] // 2] * 0  # placeholder
    # The centring used R_mid = midpoint, so absolute c1 = R_mid + R_lo_off + c1_offset
    # But R_centered_all = R_win - R_mid, so to recover absolute we need R_mid_per_b.
    # Reconstruct R_mid_per_b from R_lo_off (= R_lo - R_mid) and ring_R_ideal_px[lo_i]:
    R_lo_abs = ring_R_ideal_px[ring_lo]
    R_hi_abs = ring_R_ideal_px[ring_hi]
    R_mid_per_b = 0.5 * (R_lo_abs + R_hi_abs)
    R_fit_lo_abs = R_mid_per_b + R_lo_off_flat + c1
    R_fit_hi_abs = R_mid_per_b + R_hi_off_flat + c2

    # rms via residual at final x.
    from midas_peakfit.reparam import x_to_u
    u_final = x_to_u(x_final, lo_b, hi_b)
    r_final = residual_fn(u_final, lo_b, hi_b)
    rms = (r_final * r_final).mean(dim=-1).sqrt()
    snr = (I_max - I_min) / (rms + 1e-12)

    return DoubletFits(
        R_fit_lo=R_fit_lo_abs, R_fit_hi=R_fit_hi_abs,
        eta_deg=eta_arr,
        ring_idx_lo=ring_lo, ring_idx_hi=ring_hi,
        sigma_lo=s1, sigma_hi=s2,
        gamma_lo=g1, gamma_hi=g2,
        eta_v_lo=ev1, eta_v_hi=ev2,
        area_lo=A1, area_hi=A2,
        snr=snr, rms=rms, rc=rc,
    )


__all__ = ["DoubletFits", "fit_doublet_pairs"]
