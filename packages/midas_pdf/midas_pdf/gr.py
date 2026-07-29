"""S(Q) → G(r): thin re-export of the reused sine-FT machinery.

The sine Fourier transform with rigorous σ propagation already lives in
``midas_integrate_v2.pdf`` and is the validated, novel piece. We do not
reimplement it here — we re-export it so ``midas-pdf`` users have a single
import surface and so the dependency on the reused implementation is explicit.

    G(r) = (2/π) ∫ Q [S(Q) - 1] sin(Qr) W(Q) dQ
    σ²(G(r)) = (2/π · ΔQ)² Σ_q [Q sin(Qr) W(Q)]² σ²(S(Q))
"""
from __future__ import annotations

from midas_integrate_v2.pdf import (
    R_px_to_Q,
    estimate_background,
    fourier_sine_transform,
)

__all__ = [
    "R_px_to_Q",
    "estimate_background",
    "fourier_sine_transform",
    "G_of_r",
]

# Friendly alias matching midas-pdf naming.
G_of_r = fourier_sine_transform
