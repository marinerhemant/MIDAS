"""Per-frame pre-processing: decompress → convert → square-pad → transform →
transpose → dark/flood/threshold/mask.

Order matters and is preserved exactly from ``processImageFrame`` in
``PeaksFittingOMPZarrRefactor.c`` (lines 1380-1440).
"""
from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - typing only
    from midas_peakfit.background import BackgroundBins


# ─── Image transformations (matching applyImageTransformations_d) ───────────
def apply_image_transformations(
    image: np.ndarray, transform_options: List[int]
) -> np.ndarray:
    """Apply ``transform_options`` in order. Operates on a square (N, N) array.

    Codes (from C source):
        0: no-op
        1: flip-horizontal (along Y / row axis) — ``image[l, m] := image[l, N-m-1]``
        2: flip-vertical   (along Z / col axis) — ``image[l, m] := image[N-l-1, m]``
        3: transpose                            — ``image[l, m] := image[m, l]``

    Operates on a copy; returns the transformed array.
    """
    if not transform_options:
        return image.copy()

    out = image.copy()
    for code in transform_options:
        if code == 1:
            out = out[:, ::-1].copy()
        elif code == 2:
            out = out[::-1, :].copy()
        elif code == 3:
            out = out.T.copy()
        # 0 / unknown → no-op
    return out


def make_square_image(
    img_asym: np.ndarray, NrPixels: int, NrPixelsY: int, NrPixelsZ: int
) -> np.ndarray:
    """Pad an asymmetric (Z, Y) image to a square (NrPixels, NrPixels) image.

    Mirrors ``makeSquareImage_d`` in PeaksFittingOMPZarrRefactor.c. The on-disk
    zarr layout is row-major ``(NrPixelsZ, NrPixelsY)`` (Z slow, Y fast); the
    output square has ``NrPixels = max(NrPixelsZ, NrPixelsY)``. Bytes outside
    ``[:NrPixelsZ, :NrPixelsY]`` are zero.

    Verified against C: in both ``Y > Z`` (single memcpy) and ``Z > Y``
    (line-by-line memcpy) cases, the resulting square's ``[:Z, :Y]`` block
    equals the input.
    """
    if img_asym.shape != (NrPixelsZ, NrPixelsY):
        raise ValueError(
            f"make_square_image expected shape ({NrPixelsZ}, {NrPixelsY}), "
            f"got {img_asym.shape}"
        )
    if NrPixelsY == NrPixelsZ == NrPixels:
        return img_asym.astype(np.float64, copy=True)
    out = np.zeros((NrPixels, NrPixels), dtype=np.float64)
    out[:NrPixelsZ, :NrPixelsY] = img_asym.astype(np.float64, copy=False)
    return out


def transpose_square(image: np.ndarray) -> np.ndarray:
    """Equivalent to ``transposeMatrix`` in C. NumPy's ``.T`` returns a view;
    we materialize a contiguous copy to match C's eager layout.
    """
    return np.ascontiguousarray(image.T)


def prepare_dark(
    raw_dark: np.ndarray,
    NrPixels: int,
    NrPixelsY: int,
    NrPixelsZ: int,
    transform_options: List[int],
) -> np.ndarray:
    """Average + square-pad + transform + transpose.

    ``raw_dark`` is shape (nDarks, Y, Z) or (Y, Z). The C tool sums
    each transformed dark and divides by nDarks; since the operations
    (square-pad, flip, transpose, transpose) are linear, averaging
    *first* and then transforming produces identical results.

    Output shape: (NrPixels, NrPixels) float64.
    """
    if raw_dark.ndim == 3:
        avg = raw_dark.mean(axis=0)
    else:
        avg = raw_dark
    sq = make_square_image(avg.astype(np.float64), NrPixels, NrPixelsY, NrPixelsZ)
    transformed = apply_image_transformations(sq, transform_options)
    return transpose_square(transformed)


def prepare_flood(
    raw_flood: Optional[np.ndarray],
    NrPixels: int,
    NrPixelsY: int,
    NrPixelsZ: int,
    transform_options: List[int],  # accepted but unused; matches C semantics
) -> np.ndarray:
    """Flood field. The C tool reads the on-disk flood as a *raw*
    ``double[NrPixels × NrPixels]`` block (no square-pad, no transform, no
    transpose) — see PeaksFittingOMPZarrRefactor.c:1311-1322.

    We accept either a pre-cooked (NrPixels, NrPixels) array (used as-is) or
    fall back to ones. Zero entries are replaced with 1.0 to avoid div-by-zero.
    """
    if raw_flood is None:
        return np.ones((NrPixels, NrPixels), dtype=np.float64)
    arr = raw_flood.astype(np.float64, copy=False)
    if arr.shape != (NrPixels, NrPixels):
        # Best-effort pad/crop: assume input is (Z, Y) asymmetric;
        # only [:Z, :Y] of output is populated, rest stays as 1.0.
        out = np.ones((NrPixels, NrPixels), dtype=np.float64)
        zlim = min(NrPixelsZ, arr.shape[0])
        ylim = min(NrPixelsY, arr.shape[1])
        out[:zlim, :ylim] = arr[:zlim, :ylim]
        arr = out
    return np.where(arr == 0, 1.0, arr)


def prepare_mask(
    raw_mask: Optional[np.ndarray],
    NrPixels: int,
    NrPixelsY: int,
    NrPixelsZ: int,
    transform_options: List[int],  # accepted but unused; matches C semantics
) -> np.ndarray:
    """Mask. The C tool only square-pads the mask — no transforms, no transpose
    (PeaksFittingOMPZarrRefactor.c:1356-1364). Pixels with value > 0 are masked.

    Stored shape: (NrPixels, NrPixels) with the [:Z, :Y] block populated from
    the asymmetric input; the rest is zero.
    """
    if raw_mask is None:
        return np.zeros((NrPixels, NrPixels), dtype=np.float64)
    return make_square_image(
        raw_mask.astype(np.float64, copy=False), NrPixels, NrPixelsY, NrPixelsZ
    )


# ─── Per-frame pipeline ──────────────────────────────────────────────────────
def correct_frame(
    raw_frame: np.ndarray,
    *,
    NrPixels: int,
    NrPixelsY: int,
    NrPixelsZ: int,
    transform_options: List[int],
    dark: np.ndarray,
    flood: np.ndarray,
    good_coords: np.ndarray,
    bc: float,
    bad_px_intensity: float,
    make_map: int,
    bg_bins: "Optional[BackgroundBins]" = None,
) -> np.ndarray:
    """Replicate the C ``processImageFrame`` corrections (lines 1414-1440).

    Steps:
      1. ``image_d`` = square-padded float64 of ``raw_frame``
      2. (if ``make_map==1``) replace pixels equal to ``bad_px_intensity`` with 0
      3. apply ImTransOpt sequence
      4. transpose to analysis frame
      5. mask via goodCoords; subtract dark, divide by flood, multiply by bc
      6. (optional) subtract the local per-(ring, sector) background

    Does NOT apply the ``good_coords`` threshold -- that is
    :func:`apply_threshold`. :func:`preprocess_frame` composes the two and is
    what the peak search calls; this ungated form exists for noise and SNR
    measurement, which must see sub-threshold pixels.

    ``bg_bins`` is opt-in and defaults to ``None``, which reproduces the C
    behaviour exactly. When supplied (``BgSubtract 1``), step 6 removes the
    azimuthally-varying background *before* the threshold is applied, so
    ``RingThresh`` becomes a height above local background rather than an
    absolute detector count. See :mod:`midas_peakfit.background` for why that
    matters: the background varies by ~20 sigma around a single ring band, so
    one absolute number cannot serve the whole band.

    NOTE: with background subtraction on, the surviving pixel intensities are
    background-subtracted, so ``IntegratedIntensity`` downstream is likewise
    background-subtracted. That is the intended meaning, but it does change
    the numbers relative to a ``BgSubtract 0`` run -- they are not comparable.

    Returns: corrected, UNGATED (NrPixels, NrPixels) float64.
    """
    image_d = make_square_image(
        raw_frame.astype(np.float64, copy=False), NrPixels, NrPixelsY, NrPixelsZ
    )

    if make_map == 1 and bad_px_intensity != 0.0:
        image_d = np.where(image_d == bad_px_intensity, 0.0, image_d)

    image_d = apply_image_transformations(image_d, transform_options)
    img = transpose_square(image_d)

    out = np.zeros_like(img)
    keep = good_coords > 0
    if keep.any():
        corr = (img[keep] - dark[keep]) / flood[keep] * bc

        if bg_bins is not None:
            # Subtract the local background BEFORE thresholding, so the
            # threshold is a height above background everywhere on the ring.
            # Estimated on the corrected frame (dark/flood/bc applied), which
            # is the same quantity the threshold is compared against.
            from midas_peakfit.background import local_background

            corr_full = np.zeros_like(img)
            corr_full[keep] = corr
            bg, _ = local_background(corr_full, bg_bins)
            corr = corr - bg[keep]

        out[keep] = corr
    return out


def apply_threshold(
    corrected: np.ndarray, good_coords: np.ndarray
) -> np.ndarray:
    """The final gate of :func:`preprocess_frame`, split out.

    ``corrected`` values below their pixel's ``good_coords`` entry drop to 0;
    out-of-band pixels are already 0. Kept separate because the *ungated*
    frame is what any noise/SNR measurement has to be made on -- once this gate
    has run, every sub-threshold pixel is 0, so a local background is
    identically zero and its MAD collapses, which silently makes every SNR come
    out as 0 and every noise sigma far too small.
    """
    out = corrected.copy()
    keep = good_coords > 0
    if keep.any():
        vals = out[keep]
        out[keep] = np.where(vals < good_coords[keep], 0.0, vals)
    return out


__all__ = [
    "apply_image_transformations",
    "apply_threshold",
    "correct_frame",
    "make_square_image",
    "transpose_square",
    "prepare_dark",
    "prepare_flood",
    "prepare_mask",
    "preprocess_frame",
]


def preprocess_frame(
    raw_frame: np.ndarray,
    *,
    NrPixels: int,
    NrPixelsY: int,
    NrPixelsZ: int,
    transform_options: List[int],
    dark: np.ndarray,
    flood: np.ndarray,
    good_coords: np.ndarray,
    bc: float,
    bad_px_intensity: float,
    make_map: int,
    bg_bins: "Optional[BackgroundBins]" = None,
) -> np.ndarray:
    """Replicate the C ``processImageFrame`` corrections (lines 1414-1440).

    ``correct_frame`` followed by ``apply_threshold``. This is what the peak
    search calls; the split exists so the ungated frame is available to the
    threshold calculator.

    Returns: ``imgCorrBC`` shape (NrPixels, NrPixels) float64.
    """
    corrected = correct_frame(
        raw_frame, NrPixels=NrPixels, NrPixelsY=NrPixelsY, NrPixelsZ=NrPixelsZ,
        transform_options=transform_options, dark=dark, flood=flood,
        good_coords=good_coords, bc=bc, bad_px_intensity=bad_px_intensity,
        make_map=make_map, bg_bins=bg_bins,
    )
    return apply_threshold(corrected, good_coords)
