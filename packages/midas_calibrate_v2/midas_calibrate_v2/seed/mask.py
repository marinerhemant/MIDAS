"""Detector mask helpers — auto-detect panel gaps from sentinel intensity
values, erode to suppress panel-edge artefacts, return a clean valid-pixel
mask for arc detection / circle fitting.

Multi-panel detectors (Pilatus, Eiger) write sentinel values in the
inter-module gaps:
- ``-1``: gap (no detector here)
- ``-2``: bad / dead pixel
- ``0`` or NaN: depending on raw format

The chord-bisector and circle-fit BC seeds get biased by panel-edge
"arcs" — strong intensity gradients between gap and active pixels create
linear features that connected-components labels as arcs.  Masking these
out before arc detection is what v1's ``AutoCalibrateZarr`` does
implicitly via the ``MaskFile`` parameter; we do it inline here.
"""
from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np


def detect_panel_mask(
    image: np.ndarray,
    *,
    sentinel_values: Sequence[float] = (-1.0, -2.0, 0.0),
    sentinel_tolerance: float = 0.5,
    intensity_floor: float = 1.0,
) -> np.ndarray:
    """Auto-detect a binary mask where ``1=valid``, ``0=gap/bad``.

    Recognises the standard MIDAS sentinel values plus an "intensity floor"
    cutoff — any pixel below ``intensity_floor`` is treated as invalid
    (covers the case of zeroed-out gaps in some HDF5 / TIFF outputs).
    """
    mask = np.ones_like(image, dtype=bool)
    for s in sentinel_values:
        mask &= np.abs(image - s) > sentinel_tolerance
    mask &= image >= intensity_floor
    mask &= np.isfinite(image)
    return mask


def erode_mask(mask: np.ndarray, *, iterations: int = 2) -> np.ndarray:
    """Binary erosion to drop pixels at the panel-active-area boundary.

    Important: panel-edge pixels often have anomalously bright readings
    (charge spreading, edge effects).  Eroding the valid mask by 2-3
    pixels suppresses these without losing meaningful diffraction
    signal.
    """
    from scipy import ndimage
    return ndimage.binary_erosion(mask, iterations=int(max(iterations, 0)))


def apply_mask_for_arcs(
    image: np.ndarray,
    mask: Optional[np.ndarray] = None,
    *,
    erode_iter: int = 2,
    fill_value: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Pre-process an image for arc detection.

    Returns ``(image_masked, mask_used)``.  If ``mask`` is None, auto-detects
    sentinels via :func:`detect_panel_mask`.  The mask is then eroded by
    ``erode_iter`` pixels and applied to the image (gap pixels filled with
    ``fill_value``).
    """
    if mask is None:
        mask = detect_panel_mask(image)
    if erode_iter > 0:
        mask = erode_mask(mask, iterations=erode_iter)
    out = image.astype(np.float64, copy=True)
    out[~mask] = fill_value
    return out, mask


__all__ = ["detect_panel_mask", "erode_mask", "apply_mask_for_arcs"]
