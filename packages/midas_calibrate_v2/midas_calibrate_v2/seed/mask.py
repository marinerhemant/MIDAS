"""Detector mask helpers — auto-detect panel gaps from sentinel intensity
values, erode to suppress panel-edge artefacts, return a clean valid-pixel
mask for arc detection / circle fitting.

Multi-panel detectors (Pilatus, Eiger) write sentinel values in the
inter-module gaps:
- ``-1``: gap (no detector here)
- ``-2``: bad / dead pixel
- ``0`` or NaN: depending on raw format
- unsigned-integer dtype max (``2**16-1``, ``2**32-1``): Eiger's dead-pixel
  marker on ``uint16``/``uint32`` frames — see ``pipelines.auto.calibrate``'s
  entry-point sentinel handling, which this mirrors.

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
    (covers the case of zeroed-out gaps in some HDF5 / TIFF outputs) — and,
    for ``uint16``/``uint32`` images, the dtype-max dead-pixel sentinel
    (``pipelines.auto.calibrate`` masks this at its own entry point for the
    same reason: on a raw Eiger/Pilatus frame these are the brightest local
    maxima on the detector, and any that fall inside a ring mask would
    otherwise be picked up as top-SNR peaks).
    """
    mask = np.ones_like(image, dtype=bool)
    for s in sentinel_values:
        mask &= np.abs(image - s) > sentinel_tolerance
    mask &= image >= intensity_floor
    mask &= np.isfinite(image)
    if image.dtype == np.uint32:
        mask &= image != np.iinfo(np.uint32).max
    elif image.dtype == np.uint16:
        mask &= image != np.iinfo(np.uint16).max
    return mask


def sentinel_mask_from_raw(image: np.ndarray) -> Optional[np.ndarray]:
    """Bad-pixel mask (True = bad) from the RAW detector sentinels, or ``None``.

    Recognised markers: ``-1``/``-2`` (gap / dead) on signed-integer and float
    frames, and the dtype maximum on ``uint32`` (Eiger) and ``uint16`` frames.
    Must be given the image BEFORE any clipping or clamping (``img[img < 1] = 1``
    erases the markers and turns every module gap into a valid pixel of
    intensity 1, which the cake then counts as covered).

    The mask is in the RAW orientation of ``image``, nonzero = bad, which is
    what ``integrate_cake(mask=)`` takes. Only the sentinels are masked, not
    every pixel below some floor: a legitimate zero-count pixel is data.

    This is the one definition behind ``calibrate()``'s seeding and
    ``autocalibrate_pv``'s E-step, so the two cannot drift apart.
    """
    image = np.asarray(image)
    bad = np.zeros(image.shape, dtype=bool)
    if np.issubdtype(image.dtype, np.signedinteger) or np.issubdtype(image.dtype, np.floating):
        bad |= (image == -1) | (image == -2)
    if image.dtype == np.uint32:
        bad |= (image == np.iinfo(np.uint32).max)
    elif image.dtype == np.uint16:
        bad |= (image == np.iinfo(np.uint16).max)
    return bad if bool(bad.any()) else None


def apply_sentinel_mask(image, dark=None, mask=None):
    """Every pipeline's single entry rule: sentinel pixels never reach the cake.

    Takes the RAW ``image`` (and ``dark``, if separate), finds the gap / dead
    markers in either, and returns ``(image, dark, mask, n_masked)`` with

    * ``mask``  = the caller's mask (nonzero = bad) UNIONED with the sentinels,
      never replaced by them: a user mask is kept, and a user who masked some
      pixels must still not get the gaps;
    * ``image`` / ``dark`` copies with the sentinel positions set to 0, so a
      -1 or a dtype-max never enters an SNR profile or a subtraction.

    A frame without sentinels, and a mask of the wrong shape, are returned
    untouched (the latter so the cake builder raises its own informative error).

    Must be called on the raw frame, NOT on ``image - dark`` and NOT after a
    ``img[img < 1] = 1`` clamp: a subtraction of integer frames produces exact
    -1/-2 values legitimately, and a clamp erases the markers, turning every
    module gap into valid pixels (1 after the clamp, 2-3 once the raw dark's own
    -1/-2 is subtracted) that the cake counts as covered.
    """
    sent = sentinel_mask_from_raw(image)
    if dark is not None and np.shape(dark) == np.shape(image):
        d = sentinel_mask_from_raw(dark)
        if d is not None:
            sent = d if sent is None else (sent | d)
    if sent is None:
        return image, dark, mask, 0
    if mask is not None:
        mask = np.asarray(mask)
        if mask.shape != sent.shape:
            return image, dark, mask, 0
        sent = sent | (mask != 0)
        n_new = int(sent.sum())
    else:
        n_new = int(sent.sum())
    image = np.where(sent, 0.0, np.asarray(image, dtype=np.float64))
    if dark is not None and np.shape(dark) == sent.shape:
        dark = np.where(sent, 0.0, np.asarray(dark, dtype=np.float64))
    return image, dark, sent, n_new


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
