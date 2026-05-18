"""Per-pixel mask handling for v2 binning geometries.

A mask is a 2D ``(NrPixelsZ, NrPixelsY)`` array; non-zero entries mark
pixels to **exclude** from integration (beam stop, module gaps, dead
pixels, hot pixels, sample-shadow region, …). Convention matches v1:
1.0 = masked.

The mask is applied at **build time** (not integration time):

- :class:`HardBinGeometry`, :class:`SubpixelBinGeometry`,
  :class:`PolygonBinGeometry`: masked pixels never appear in the
  geometry's `pix_idx` or `flat_bin`. Integration is unchanged.
- :class:`SoftBinGeometry`: the soft path keeps every pixel (it's
  differentiable in geometry, no sense pruning), but applies the
  mask via a multiplicative weight at integrate time.

This is faster than masking at integrate time because the masked
pixels never enter the index_add / scatter, AND it makes the geometry
correctly normalised (per-bin pixel count excludes masked pixels).
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch


def normalise_mask(
    mask,
    *,
    NrPixelsY: int,
    NrPixelsZ: int,
) -> Optional[np.ndarray]:
    """Coerce a mask into a uniform numpy bool array.

    Accepts:
      - ``None`` (no mask) → returns None.
      - 2D numpy/torch array of shape ``(NrPixelsZ, NrPixelsY)``.
      - 2D bool, float, int — non-zero treated as "masked".

    Returns a contiguous bool ndarray of shape ``(NrPixelsZ,
    NrPixelsY)`` where True = masked.
    """
    if mask is None:
        return None
    if isinstance(mask, torch.Tensor):
        m = mask.detach().cpu().numpy()
    else:
        m = np.asarray(mask)
    if m.shape != (NrPixelsZ, NrPixelsY):
        raise ValueError(
            f"mask shape {m.shape} does not match detector "
            f"({NrPixelsZ}, {NrPixelsY})"
        )
    return np.ascontiguousarray(m != 0)


def mask_fraction(mask: Optional[np.ndarray]) -> float:
    """Fraction of pixels that are masked (0 if mask is None)."""
    if mask is None:
        return 0.0
    return float(mask.sum()) / float(mask.size)


__all__ = ["normalise_mask", "mask_fraction"]
