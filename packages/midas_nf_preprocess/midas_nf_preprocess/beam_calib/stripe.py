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

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np

__all__ = ["StripeFit", "TiltFit", "find_stripe", "stripe_tilt"]


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

    def zbc(self, n_pixels_z: int) -> float:
        """Beam centre in the MIDAS convention (``NrPixelsZ-1 - row``).

        The flip is a property of the detector/writer chain -- verify it for a
        new beamline rather than inheriting it (handbook §3h).
        """
        return (n_pixels_z - 1) - self.row_centroid


def find_stripe(
    image: np.ndarray,
    *,
    px_um: Optional[float] = None,
    frac_of_peak: float = 0.05,
) -> StripeFit:
    """Locate the direct-beam stripe in a background-free image.

    ``image`` should be the TEMPORAL MEDIAN over omega, not a single frame:
    the direct beam is the one feature that does not move, so the median both
    suppresses Bragg spots and keeps the beam at full strength.
    """
    prof = image.mean(axis=1).astype(np.float64)
    prof = prof - np.median(prof)
    peak = float(prof.max())
    if peak <= 0:
        raise ValueError("no positive signal in the row profile -- "
                         "is this really a direct-beam image?")
    row_peak = int(np.argmax(prof))
    above_half = np.where(prof >= 0.5 * peak)[0]
    fwhm = int(above_half[-1] - above_half[0] + 1) if above_half.size else 0

    sel = np.where(prof > frac_of_peak * peak)[0]
    seg = np.clip(prof[sel], 0, None)
    row_c = float((sel * seg).sum() / seg.sum())

    band = image[above_half[0]:above_half[-1] + 1, :].sum(axis=0) \
        if above_half.size else image.sum(axis=0)
    band = band - np.median(band)
    lit = np.where(band > frac_of_peak * band.max())[0]
    lo, hi = (int(lit[0]), int(lit[-1])) if lit.size else (-1, -1)

    return StripeFit(
        row_centroid=row_c, row_peak=row_peak, fwhm_rows=fwhm,
        height_um=(fwhm * px_um) if px_um else None,
        band_lo_col=lo, band_hi_col=hi,
        band_width_um=((hi - lo + 1) * px_um) if (px_um and lo >= 0) else None,
        peak_value=peak,
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
