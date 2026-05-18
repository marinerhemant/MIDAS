"""SNIP / Morháč LLS background subtraction.

The Statistics-sensitive Non-linear Iterative Peak-clipping algorithm
(Morháč et al. 1997) extracts a smooth baseline from a 1-D signal by
iteratively clipping each sample to the mean of its symmetric neighbours
within a window of decreasing width.  Peak features are clipped down;
smoothly-varying background remains.

Reference: Morháč, M. et al., NIM A 401 (1997) 113-132.

Used as a pre-fit background for the per-(ring, η-bin) pV peak fit, so
the model itself doesn't need a baseline term — closer to v1 C's
``CalibrantIntegratorOMP`` recipe.

Implementation is pure-torch and fully vectorised across the batch axis.
"""
from __future__ import annotations

from typing import Optional

import torch


def lls_forward(y: torch.Tensor) -> torch.Tensor:
    """Log-Log-Sqrt (LLS) transform: y_LLS = log(log(sqrt(y+1)+1)+1).

    Standard pre-conditioner for SNIP — compresses large values so the
    clipping window operates on a roughly Gaussian-noise-distributed
    signal.
    """
    return torch.log(torch.log(torch.sqrt(y.clamp(min=0.0) + 1.0) + 1.0) + 1.0)


def lls_inverse(y_lls: torch.Tensor) -> torch.Tensor:
    """Inverse LLS: y = (exp(exp(y_LLS) - 1) - 1)² - 1."""
    inner = torch.exp(y_lls) - 1.0
    return (torch.exp(inner) - 1.0) ** 2 - 1.0


def snip_background(
    y: torch.Tensor,
    *,
    window_max: int = 16,
    n_iter_per_window: int = 1,
    use_lls: bool = True,
) -> torch.Tensor:
    """Compute the SNIP baseline of a batched 1-D signal.

    Parameters
    ----------
    y : tensor of shape ``[..., M]``.  Last dim is the radial axis.
    window_max : maximum half-window in bins; should be ~peak FWHM.
    n_iter_per_window : repetitions per window size (Morháč suggests 1).
    use_lls : apply the LLS transform before / after (recommended).

    Returns
    -------
    bg : tensor of same shape as ``y``, the smoothly-varying background.
    """
    y_in = lls_forward(y) if use_lls else y.clone()
    out = y_in.clone()
    for p in range(window_max, 0, -1):
        # Symmetric neighbour window: out[i] = min(out[i], 0.5*(out[i-p]+out[i+p]))
        # Implemented via roll + min.  Boundary samples use one-sided neighbours.
        for _ in range(n_iter_per_window):
            left = torch.roll(out, shifts=p, dims=-1)
            right = torch.roll(out, shifts=-p, dims=-1)
            avg = 0.5 * (left + right)
            out = torch.minimum(out, avg)
    if use_lls:
        return lls_inverse(out)
    return out


def subtract_snip_background(
    y: torch.Tensor,
    *,
    window_max: int = 16,
    n_iter_per_window: int = 1,
    use_lls: bool = True,
    floor_at_zero: bool = True,
) -> torch.Tensor:
    """Convenience wrapper: ``y - snip_background(y, ...)``."""
    bg = snip_background(y, window_max=window_max,
                          n_iter_per_window=n_iter_per_window,
                          use_lls=use_lls)
    diff = y - bg
    if floor_at_zero:
        diff = diff.clamp(min=0.0)
    return diff


__all__ = ["lls_forward", "lls_inverse", "snip_background",
           "subtract_snip_background"]
