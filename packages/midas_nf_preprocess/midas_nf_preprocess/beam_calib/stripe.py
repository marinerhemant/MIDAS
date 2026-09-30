"""Beam centre ``zbc`` and the tilt seed ``tx`` from the direct beam.

The direct beam, where it reaches the detector, is a thin horizontal stripe
(vertically focused).  That makes the *vertical* centroid sharp and gives
``zbc`` per detector distance directly.

What this CANNOT give you
-------------------------
``ybc``.  Horizontally the stripe is a broad, slit-defined band whose centre
is the centre of the *illuminated region* -- set by the slits, unrelated to
where the rotation axis is.  Use :mod:`.shadow` for ``ybc``.  Handbook §6e
records the estimator error here costing 66 px of scatter.

``tx`` from the stripe slope is usually WEAK.  :func:`stripe_tilt` therefore
returns the fit residual alongside the angle so the caller can see whether it
is a seed or merely a bound; on `nfdev_jul26` the residual was 2.55 px against
only 3.65 px of linear signal, i.e. a bound of |tx| < 0.15 deg rather than a
measurement of 0.075 deg.

Handbook: §6a, §6d, §6f, §7a.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "StripeFit", "StripeScanFit", "TiltFit",
    "find_stripe", "find_stripe_scan", "stripe_tilt",
]


@dataclass
class StripeFit:
    """Vertical position and extent of the direct-beam stripe."""

    row_centroid: float
    row_peak: int
    fwhm_rows: int
    height_um: Optional[float]
    band_lo_col: int
    band_hi_col: int
    band_width_um: Optional[float]
    peak_value: float
    #: Row centroids of EVERY horizontal band found in the image (this one
    #: included).  More than one entry means the choice was ambiguous from
    #: this image alone -- see :func:`find_stripe_scan`.
    candidate_rows: Tuple[float, ...] = ()

    def zbc(self, n_pixels_z: int) -> float:
        """Beam centre in the MIDAS convention (``NrPixelsZ-1 - row``).

        The flip is a property of the detector/writer chain -- verify it for a
        new beamline rather than inheriting it (handbook §3h).
        """
        return (n_pixels_z - 1) - self.row_centroid


def _row_profile(image: np.ndarray) -> np.ndarray:
    prof = image.mean(axis=1).astype(np.float64)
    return prof - np.median(prof)


def _find_bands(
    prof: np.ndarray, candidate_frac: float, frac_of_peak: float,
) -> list:
    """Separate horizontal bands in a row profile.

    A band is seeded by a contiguous run of rows above ``candidate_frac`` of
    the global peak, then grown outward while the profile stays above
    ``frac_of_peak`` of THAT band's own peak.  Bands whose grown extents touch
    are merged.  Returns ``[(lo, hi), ...]`` inclusive row ranges, top first.
    """
    peak = float(prof.max())
    seed = prof > candidate_frac * peak
    n = prof.size
    runs = []
    r = 0
    while r < n:
        if seed[r]:
            a = r
            while r + 1 < n and seed[r + 1]:
                r += 1
            runs.append((a, r))
        r += 1
    bands: list = []
    for a, b in runs:
        thr = frac_of_peak * float(prof[a:b + 1].max())
        lo, hi = a, b
        while lo > 0 and prof[lo - 1] > thr:
            lo -= 1
        while hi < n - 1 and prof[hi + 1] > thr:
            hi += 1
        if bands and lo <= bands[-1][1] + 1:
            bands[-1] = (bands[-1][0], max(hi, bands[-1][1]))
        else:
            bands.append((lo, hi))
    return bands


def _fit_band(
    image: np.ndarray, prof: np.ndarray, lo: int, hi: int,
    px_um: Optional[float], frac_of_peak: float,
) -> StripeFit:
    """Centroid, FWHM and column extent of ONE band (rows ``lo..hi``)."""
    seg_prof = prof[lo:hi + 1]
    peak = float(seg_prof.max())
    row_peak = lo + int(np.argmax(seg_prof))
    above_half = lo + np.where(seg_prof >= 0.5 * peak)[0]
    fwhm = int(above_half[-1] - above_half[0] + 1) if above_half.size else 0

    rows = np.arange(lo, hi + 1)
    w = np.where(seg_prof > frac_of_peak * peak, np.clip(seg_prof, 0, None), 0.0)
    row_c = float((rows * w).sum() / w.sum())

    band = image[above_half[0]:above_half[-1] + 1, :].sum(axis=0) \
        if above_half.size else image[lo:hi + 1, :].sum(axis=0)
    band = band - np.median(band)
    lit = np.where(band > frac_of_peak * band.max())[0]
    c_lo, c_hi = (int(lit[0]), int(lit[-1])) if lit.size else (-1, -1)

    return StripeFit(
        row_centroid=row_c, row_peak=row_peak, fwhm_rows=fwhm,
        height_um=(fwhm * px_um) if px_um else None,
        band_lo_col=c_lo, band_hi_col=c_hi,
        band_width_um=((c_hi - c_lo + 1) * px_um) if (px_um and c_lo >= 0) else None,
        peak_value=peak,
    )


def _all_bands(
    image: np.ndarray, px_um: Optional[float], frac_of_peak: float,
    candidate_frac: float,
) -> list:
    prof = _row_profile(image)
    if float(prof.max()) <= 0:
        raise ValueError("no positive signal in the row profile -- "
                         "is this really a direct-beam image?")
    fits = [_fit_band(image, prof, lo, hi, px_um, frac_of_peak)
            for lo, hi in _find_bands(prof, candidate_frac, frac_of_peak)]
    rows = tuple(f.row_centroid for f in fits)
    for f in fits:
        f.candidate_rows = rows
    return fits


def find_stripe(
    image: np.ndarray,
    *,
    px_um: Optional[float] = None,
    frac_of_peak: float = 0.05,
    candidate_frac: float = 0.2,
    row_hint: Optional[float] = None,
) -> StripeFit:
    """Locate the direct-beam stripe in ONE background-free image.

    ``image`` should be the TEMPORAL MEDIAN over omega, not a single frame:
    the direct beam is the one feature that does not move, so the median both
    suppresses Bragg spots and keeps the beam at full strength.

    One image cannot tell the beam stripe from a STATIONARY band (scatter or
    a detector/filter feature that does not move with the detector).  On the
    2021 1-ID DetZBeamPos scan such a band sits ~65 px above the true stripe;
    the old single-peak code returned zbc 63 against a true 29.6.  If you have
    images at two or more detector Z positions, use :func:`find_stripe_scan`,
    which keeps only the band that moves with the detector.

    Selection criterion when several bands are found (every band whose row
    profile exceeds ``candidate_frac`` of the global peak is a candidate):

    * ``row_hint`` given -> the band whose centroid is nearest that row;
    * otherwise -> the band with the highest row-profile peak, AND a
      ``UserWarning`` listing every candidate row.

    The centroid is computed over the chosen band only; separate bands are
    never blended into one centroid.  All candidate centroids are returned in
    :attr:`StripeFit.candidate_rows`.
    """
    fits = _all_bands(image, px_um, frac_of_peak, candidate_frac)
    if len(fits) == 1:
        return fits[0]
    if row_hint is not None:
        return min(fits, key=lambda f: abs(f.row_centroid - row_hint))
    best = max(fits, key=lambda f: f.peak_value)
    warnings.warn(
        f"find_stripe: {len(fits)} horizontal bands at rows "
        f"{[round(f.row_centroid, 1) for f in fits]}; returning the brightest "
        f"(row {best.row_centroid:.1f}). A single image cannot distinguish the "
        "beam stripe from a stationary band -- use find_stripe_scan over a "
        "DetZ scan, or pass row_hint.",
        UserWarning, stacklevel=2,
    )
    return best


@dataclass
class StripeScanFit:
    """The direct-beam stripe tracked across a detector-Z scan."""

    #: One :class:`StripeFit` per input image, for the band that MOVES with DetZ.
    fits: list
    det_z_um: Tuple[float, ...]
    #: Fitted d(row)/d(DetZ) in rows per µm; expected magnitude ``1 / px_um``.
    rows_per_um: float
    resid_rms_px: float
    #: Row centroids (first image) of bands rejected as stationary.
    stationary_rows: Tuple[float, ...]


def find_stripe_scan(
    images: Sequence[np.ndarray],
    det_z_um: Sequence[float],
    *,
    px_um: float,
    frac_of_peak: float = 0.05,
    candidate_frac: float = 0.2,
    tol_px: float = 3.0,
    slope_rtol: float = 0.25,
) -> StripeScanFit:
    """Find the beam stripe as the band that TRACKS the detector Z motion.

    ``images[i]`` is a background-free (median) image taken at detector
    position ``det_z_um[i]`` (µm).  Moving the detector by ``dz`` moves the
    real beam stripe by ``|dz| / px_um`` rows; a stationary band (scatter, a
    detector feature) stays put.  Every band in every image is a candidate.
    A track is the line through a band in the first image and a band in the
    image farthest in Z; it is kept only if

    * every other image has a band within ``tol_px`` of the line, and
    * ``|slope|`` is within ``slope_rtol`` of ``1 / px_um``.  The sign is NOT
      assumed -- it depends on the stage and readout convention.

    Bands whose track is flat (moves <= ``tol_px`` over the scan) are
    reported in :attr:`StripeScanFit.stationary_rows` and rejected.

    Raises ``ValueError`` if the Z travel is too small to separate a moving
    band from a stationary one, or if the number of tracking bands is not
    exactly one.
    """
    if len(images) != len(det_z_um) or len(images) < 2:
        raise ValueError("need >= 2 images and one det_z_um per image")
    z = np.asarray(det_z_um, dtype=float)
    expected = 1.0 / float(px_um)
    far = int(np.argmax(np.abs(z - z[0])))
    travel_px = abs(z[far] - z[0]) * expected
    if travel_px <= 2.0 * tol_px:
        raise ValueError(
            f"DetZ travel is only {travel_px:.1f} px (<= 2*tol_px = "
            f"{2 * tol_px:.1f}); a moving stripe cannot be told from a "
            "stationary band. Use a wider DetZ scan.")

    per_image = [_all_bands(im, px_um, frac_of_peak, candidate_frac)
                 for im in images]
    tracks = []
    stationary = []
    for f0 in per_image[0]:
        r0 = f0.row_centroid
        for ff in per_image[far]:
            slope = (ff.row_centroid - r0) / (z[far] - z[0])
            chosen = []
            for zi, fits_i in zip(z, per_image):
                pred = r0 + slope * (zi - z[0])
                near = min(fits_i, key=lambda f: abs(f.row_centroid - pred))
                if abs(near.row_centroid - pred) > tol_px:
                    break
                chosen.append(near)
            else:
                if abs(slope) * abs(z[far] - z[0]) <= tol_px:
                    stationary.append(r0)
                elif abs(abs(slope) - expected) <= slope_rtol * expected:
                    tracks.append(chosen)

    if len(tracks) != 1:
        raise ValueError(
            f"find_stripe_scan: {len(tracks)} bands track the detector motion "
            f"(expected exactly 1 with |d row / d z| = {expected:.4g} per um "
            f"+- {100 * slope_rtol:.0f}%). Band rows per image: "
            f"{[[round(f.row_centroid, 1) for f in fi] for fi in per_image]}")

    chosen = tracks[0]
    rows = np.array([f.row_centroid for f in chosen])
    A = np.column_stack([np.ones_like(z), z])
    coef, *_ = np.linalg.lstsq(A, rows, rcond=None)
    resid = rows - A @ coef
    return StripeScanFit(
        fits=chosen,
        det_z_um=tuple(float(v) for v in z),
        rows_per_um=float(coef[1]),
        resid_rms_px=float(np.sqrt((resid ** 2).mean())),
        stationary_rows=tuple(sorted(set(stationary))),
    )


@dataclass
class TiltFit:
    """Stripe slope, with the evidence needed to judge whether it is usable."""

    slope_px_per_px: float
    tilt_deg: float
    resid_rms_px: float
    signal_px: float
    n_blocks: int

    @property
    def is_measurement(self) -> bool:
        """True only when the linear signal clearly exceeds the scatter."""
        return self.signal_px > 3.0 * self.resid_rms_px

    @property
    def bound_deg(self) -> float:
        """Magnitude below which the tilt cannot be distinguished from zero."""
        return float(np.degrees(np.arctan(
            3.0 * self.resid_rms_px / max(self.span_px, 1.0))))

    span_px: float = 1.0


def stripe_tilt(
    image: np.ndarray,
    stripe: StripeFit,
    *,
    n_blocks: int = 20,
    half_rows: int = 40,
) -> TiltFit:
    """Slope of the stripe across the detector -> a seed (or bound) for ``tx``.

    Always inspect :attr:`TiltFit.is_measurement`.  A slope whose residual is
    comparable to the signal is a BOUND, not a seed, and feeding it in as if
    it were measured puts a fake tilt into the geometry.
    """
    if stripe.band_lo_col < 0:
        raise ValueError("stripe has no illuminated band")
    r0 = max(stripe.row_peak - half_rows, 0)
    r1 = min(stripe.row_peak + half_rows, image.shape[0])
    sub = image[r0:r1, :]
    rows = np.arange(r0, r1)

    edges = np.linspace(stripe.band_lo_col, stripe.band_hi_col, n_blocks + 1).astype(int)
    xs, cs = [], []
    for a, b in zip(edges[:-1], edges[1:]):
        if b <= a:
            continue
        seg = sub[:, a:b].sum(axis=1).astype(np.float64)
        seg = np.clip(seg - np.median(seg), 0, None)
        if seg.sum() <= 0:
            continue
        cs.append(float((rows * seg).sum() / seg.sum()))
        xs.append(0.5 * (a + b))
    if len(xs) < 3:
        raise ValueError("too few usable column blocks for a slope fit")

    x = np.asarray(xs, dtype=float)
    c = np.asarray(cs, dtype=float)
    A = np.column_stack([np.ones_like(x), x])
    coef, *_ = np.linalg.lstsq(A, c, rcond=None)
    resid = c - A @ coef
    span = float(x.max() - x.min())
    return TiltFit(
        slope_px_per_px=float(coef[1]),
        tilt_deg=float(np.degrees(np.arctan(coef[1]))),
        resid_rms_px=float(np.sqrt((resid ** 2).mean())),
        signal_px=float(abs(coef[1]) * span),
        n_blocks=len(xs),
        span_px=span,
    )
