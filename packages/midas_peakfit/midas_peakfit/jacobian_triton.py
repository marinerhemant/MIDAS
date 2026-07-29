"""Triton kernel for the analytical Pseudo-Voigt residuals + Jacobian.

This is the experimental "custom CUDA kernel" path. The PyTorch eager
``residuals_and_jacobian_u`` builds many intermediate tensors (G, L, dG, dL,
etc.) and runs each Pseudo-Voigt formula as a separate elementwise op.
A Triton kernel can fuse all of that into a single pass through (B, M)
pixels: load region parameters once, loop over peaks in registers,
accumulate the model + residual + Jacobian column for each parameter.

Specialization strategy: the kernel is JIT-specialized per ``(n_peaks,
dtype)``. The bucket dispatcher in ``pool.py`` already groups regions by
``(n_peaks, M_padded)`` so each bucket flush hits one specialization.

Usage: enabled via ``LMConfig.use_triton_jacobian = True``. Falls back
to the eager analytical path on any failure.
"""
from __future__ import annotations

from typing import Tuple

import torch

try:
    import triton
    import triton.language as tl
    _TRITON_OK = True
except Exception:
    _TRITON_OK = False


_JIT_CACHE: dict = {}


if _TRITON_OK:

    @triton.jit
    def _resjac_kernel(
        u_ptr, lo_ptr, hi_ptr,         # [B, N]
        z_ptr, Rs_ptr, Etas_ptr, mask_ptr,  # [B, M]
        r_ptr,                          # [B, M] output
        J_ptr,                          # [B, M, N] output
        B, M, N, P,
        stride_uB, stride_uN,
        stride_zB, stride_zM,
        stride_rB, stride_rM,
        stride_JB, stride_JM, stride_JN,
        BLOCK_M: tl.constexpr,
    ):
        """One thread block per (region, M-tile). Each thread within a
        block handles one pixel; computes residual + full Jacobian row.
        N is loaded into registers per thread (small for typical P).
        """
        # int64 for the region index: it multiplies the LARGEST stride
        # (stride_JB = M*N). For dense regions batched together — e.g. B=125
        # regions of n_peaks=400 (N=3201) x M=10000 → stride_JB=3.2e7 — the
        # product b*stride_JB exceeds int32 (2^31) once b>=68, wrapping to a
        # negative offset and writing J to a garbage address → intermittent
        # "illegal memory access". It only bites when 68+ of the biggest
        # regions land in one batch, which is why sparse data and small
        # isolation batches never triggered it. Cast so every b-derived
        # address is 64-bit.
        b = tl.program_id(0).to(tl.int64)
        m_block = tl.program_id(1)

        m_offsets = m_block * BLOCK_M + tl.arange(0, BLOCK_M)
        m_mask = m_offsets < M

        # Load pixel inputs for this tile
        Rs = tl.load(Rs_ptr + b * stride_zB + m_offsets * stride_zM, mask=m_mask, other=0.0)
        Etas = tl.load(Etas_ptr + b * stride_zB + m_offsets * stride_zM, mask=m_mask, other=0.0)
        z = tl.load(z_ptr + b * stride_zB + m_offsets * stride_zM, mask=m_mask, other=0.0)
        pmask = tl.load(mask_ptr + b * stride_zB + m_offsets * stride_zM, mask=m_mask, other=0.0)

        # Reparam: x = lo + (hi-lo) * sigmoid(u)
        # First compute bg (parameter index 0)
        u_bg = tl.load(u_ptr + b * stride_uB + 0 * stride_uN)
        lo_bg = tl.load(lo_ptr + b * stride_uB + 0 * stride_uN)
        hi_bg = tl.load(hi_ptr + b * stride_uB + 0 * stride_uN)
        s_bg = 1.0 / (1.0 + tl.exp(-u_bg))
        x_bg = lo_bg + (hi_bg - lo_bg) * s_bg
        dxdu_bg = (hi_bg - lo_bg) * s_bg * (1.0 - s_bg)

        # Initialize model accumulator with bg
        model = tl.full((BLOCK_M,), x_bg, tl.float64)

        # Per-peak parameter loop. We can't statically unroll over P (runtime),
        # but Triton will iterate at runtime. Each peak contributes 8 J columns.
        # We compute the model in a first pass; a second pass writes J columns
        # using the same shared formulas.

        # Pre-load all peak params into local scratch (small for P up to ~200)
        # We reload per-peak inside the loop to keep register pressure bounded.

        for p in tl.range(0, P):
            base = 1 + 8 * p
            # Load params for this peak
            u_imax = tl.load(u_ptr + b * stride_uB + (base + 0) * stride_uN)
            u_R = tl.load(u_ptr + b * stride_uB + (base + 1) * stride_uN)
            u_E = tl.load(u_ptr + b * stride_uB + (base + 2) * stride_uN)
            u_mu = tl.load(u_ptr + b * stride_uB + (base + 3) * stride_uN)
            u_sgR = tl.load(u_ptr + b * stride_uB + (base + 4) * stride_uN)
            u_slR = tl.load(u_ptr + b * stride_uB + (base + 5) * stride_uN)
            u_sgE = tl.load(u_ptr + b * stride_uB + (base + 6) * stride_uN)
            u_slE = tl.load(u_ptr + b * stride_uB + (base + 7) * stride_uN)
            lo_imax = tl.load(lo_ptr + b * stride_uB + (base + 0) * stride_uN)
            hi_imax = tl.load(hi_ptr + b * stride_uB + (base + 0) * stride_uN)
            lo_R = tl.load(lo_ptr + b * stride_uB + (base + 1) * stride_uN)
            hi_R = tl.load(hi_ptr + b * stride_uB + (base + 1) * stride_uN)
            lo_E = tl.load(lo_ptr + b * stride_uB + (base + 2) * stride_uN)
            hi_E = tl.load(hi_ptr + b * stride_uB + (base + 2) * stride_uN)
            lo_mu = tl.load(lo_ptr + b * stride_uB + (base + 3) * stride_uN)
            hi_mu = tl.load(hi_ptr + b * stride_uB + (base + 3) * stride_uN)
            lo_sgR = tl.load(lo_ptr + b * stride_uB + (base + 4) * stride_uN)
            hi_sgR = tl.load(hi_ptr + b * stride_uB + (base + 4) * stride_uN)
            lo_slR = tl.load(lo_ptr + b * stride_uB + (base + 5) * stride_uN)
            hi_slR = tl.load(hi_ptr + b * stride_uB + (base + 5) * stride_uN)
            lo_sgE = tl.load(lo_ptr + b * stride_uB + (base + 6) * stride_uN)
            hi_sgE = tl.load(hi_ptr + b * stride_uB + (base + 6) * stride_uN)
            lo_slE = tl.load(lo_ptr + b * stride_uB + (base + 7) * stride_uN)
            hi_slE = tl.load(hi_ptr + b * stride_uB + (base + 7) * stride_uN)

            # Reparameterize
            s_imax = 1.0 / (1.0 + tl.exp(-u_imax))
            Imax = lo_imax + (hi_imax - lo_imax) * s_imax
            dxdu_imax = (hi_imax - lo_imax) * s_imax * (1.0 - s_imax)
            s_R = 1.0 / (1.0 + tl.exp(-u_R))
            R = lo_R + (hi_R - lo_R) * s_R
            dxdu_R = (hi_R - lo_R) * s_R * (1.0 - s_R)
            s_E = 1.0 / (1.0 + tl.exp(-u_E))
            E = lo_E + (hi_E - lo_E) * s_E
            dxdu_E = (hi_E - lo_E) * s_E * (1.0 - s_E)
            s_mu = 1.0 / (1.0 + tl.exp(-u_mu))
            Mu = lo_mu + (hi_mu - lo_mu) * s_mu
            dxdu_mu = (hi_mu - lo_mu) * s_mu * (1.0 - s_mu)
            s_sgR = 1.0 / (1.0 + tl.exp(-u_sgR))
            sgR = lo_sgR + (hi_sgR - lo_sgR) * s_sgR
            dxdu_sgR = (hi_sgR - lo_sgR) * s_sgR * (1.0 - s_sgR)
            s_slR = 1.0 / (1.0 + tl.exp(-u_slR))
            slR = lo_slR + (hi_slR - lo_slR) * s_slR
            dxdu_slR = (hi_slR - lo_slR) * s_slR * (1.0 - s_slR)
            s_sgE = 1.0 / (1.0 + tl.exp(-u_sgE))
            sgE = lo_sgE + (hi_sgE - lo_sgE) * s_sgE
            dxdu_sgE = (hi_sgE - lo_sgE) * s_sgE * (1.0 - s_sgE)
            s_slE = 1.0 / (1.0 + tl.exp(-u_slE))
            slE = lo_slE + (hi_slE - lo_slE) * s_slE
            dxdu_slE = (hi_slE - lo_slE) * s_slE * (1.0 - s_slE)

            # Per-pixel computation
            dR = Rs - R
            dE = Etas - E
            dR2 = dR * dR
            dE2 = dE * dE
            sgR2 = sgR * sgR + 1e-12
            slR2 = slR * slR + 1e-12
            sgE2 = sgE * sgE + 1e-12
            slE2 = slE * slE + 1e-12

            A = 1.0 + dR2 / slR2
            Bx = 1.0 + dE2 / slE2
            L = 1.0 / (A * Bx)
            G = tl.exp(-0.5 * (dR2 / sgR2 + dE2 / sgE2))

            profile = Imax * (Mu * L + (1.0 - Mu) * G)
            model = model + profile

            # Now write J columns for this peak — partial derivatives
            # All in u-space (multiply by dxdu)
            # ∂M/∂Imax = (μ L + (1-μ) G)
            J_imax = (Mu * L + (1.0 - Mu) * G) * pmask * dxdu_imax
            # ∂M/∂R_j (chain via dR = Rs - R, ∂dR/∂R = -1):
            # ∂G/∂R = G × dR/sgR²;  ∂L/∂R = (2 dR / slR²) × L/A
            dG_dR = G * (dR / sgR2)
            dL_dR = (2.0 * dR / slR2) * (L / A)
            J_R = Imax * (Mu * dL_dR + (1.0 - Mu) * dG_dR) * pmask * dxdu_R
            dG_dE = G * (dE / sgE2)
            dL_dE = (2.0 * dE / slE2) * (L / Bx)
            J_E = Imax * (Mu * dL_dE + (1.0 - Mu) * dG_dE) * pmask * dxdu_E
            J_mu = Imax * (L - G) * pmask * dxdu_mu
            dG_dsgR = G * (dR2 / (sgR * sgR2))
            J_sgR = Imax * (1.0 - Mu) * dG_dsgR * pmask * dxdu_sgR
            dL_dslR = (2.0 * dR2 / (slR * slR2)) * (L / A)
            J_slR = Imax * Mu * dL_dslR * pmask * dxdu_slR
            dG_dsgE = G * (dE2 / (sgE * sgE2))
            J_sgE = Imax * (1.0 - Mu) * dG_dsgE * pmask * dxdu_sgE
            dL_dslE = (2.0 * dE2 / (slE * slE2)) * (L / Bx)
            J_slE = Imax * Mu * dL_dslE * pmask * dxdu_slE

            # Write the 8 J columns for this peak
            J_base = J_ptr + b * stride_JB + m_offsets * stride_JM
            tl.store(J_base + (base + 0) * stride_JN, J_imax, mask=m_mask)
            tl.store(J_base + (base + 1) * stride_JN, J_R, mask=m_mask)
            tl.store(J_base + (base + 2) * stride_JN, J_E, mask=m_mask)
            tl.store(J_base + (base + 3) * stride_JN, J_mu, mask=m_mask)
            tl.store(J_base + (base + 4) * stride_JN, J_sgR, mask=m_mask)
            tl.store(J_base + (base + 5) * stride_JN, J_slR, mask=m_mask)
            tl.store(J_base + (base + 6) * stride_JN, J_sgE, mask=m_mask)
            tl.store(J_base + (base + 7) * stride_JN, J_slE, mask=m_mask)

        # Residual = (model - z) * mask
        r = (model - z) * pmask
        tl.store(r_ptr + b * stride_rB + m_offsets * stride_rM, r, mask=m_mask)
        # bg column = mask × dxdu_bg
        tl.store(J_ptr + b * stride_JB + m_offsets * stride_JM + 0 * stride_JN,
                 pmask * dxdu_bg, mask=m_mask)


def residuals_and_jacobian_u_triton(
    u: torch.Tensor,
    lo: torch.Tensor,
    hi: torch.Tensor,
    z: torch.Tensor,
    Rs: torch.Tensor,
    Etas: torch.Tensor,
    pixel_mask: torch.Tensor,
    n_peaks: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Drop-in replacement for residuals_and_jacobian_u using a Triton kernel.

    Falls back to the eager PyTorch path if Triton isn't available or the
    kernel launch fails. Returns ``(r, J)`` matching the eager signature.
    """
    if not _TRITON_OK or u.device.type != "cuda":
        from midas_peakfit.jacobian import residuals_and_jacobian_u
        return residuals_and_jacobian_u(u, lo, hi, z, Rs, Etas, pixel_mask, n_peaks)

    B, N = u.shape
    M = z.shape[-1]
    assert N == 1 + 8 * n_peaks

    r = torch.empty((B, M), dtype=u.dtype, device=u.device)
    J = torch.zeros((B, M, N), dtype=u.dtype, device=u.device)

    BLOCK_M = 64

    grid = (B, triton.cdiv(M, BLOCK_M))
    try:
        _resjac_kernel[grid](
            u, lo, hi, z, Rs, Etas, pixel_mask,
            r, J,
            B, M, N, n_peaks,
            u.stride(0), u.stride(1),
            z.stride(0), z.stride(1),
            r.stride(0), r.stride(1),
            J.stride(0), J.stride(1), J.stride(2),
            BLOCK_M=BLOCK_M,
        )
    except Exception as e:
        from midas_peakfit.jacobian import residuals_and_jacobian_u
        print(f"[triton] kernel failed ({type(e).__name__}: {e!s:.80}); falling back to eager")
        return residuals_and_jacobian_u(u, lo, hi, z, Rs, Etas, pixel_mask, n_peaks)
    return r, J


__all__ = ["residuals_and_jacobian_u_triton", "_TRITON_OK"]
