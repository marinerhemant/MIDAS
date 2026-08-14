"""Finding the rings in an azimuthally-integrated profile.

Which radii carry signal decides which channels are worth reconstructing, so
this is a planning step, not a diagnostic. It is also the input to phase
indexing, and an incomplete ring list is the fastest way to fail to index a
correct cell.

The trap it exists to avoid: a *global* median is the wrong baseline. Powder
background falls steeply with radius, so a global threshold is far above the
background at low R and below it at high R -- it finds the strong inner rings,
misses most of the outer ones, and returns a list that looks plausible. A
rolling median tracks the background instead.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

__all__ = ["Ring", "rolling_baseline", "find_rings"]

log = logging.getLogger(__name__)


@dataclass
class Ring:
    """One detected ring."""

    radius_px: float
    d_spacing_a: float | None
    height: float          # above the local baseline
    prominence: float
    width_px: float
    snr: float

    def describe(self) -> str:
        d = f"{self.d_spacing_a:7.4f}" if self.d_spacing_a is not None else "      -"
        return (f"R {self.radius_px:7.2f} px  d {d} A  "
                f"height {self.height:9.3g}  SNR {self.snr:6.1f}  "
                f"width {self.width_px:5.2f} px")


def rolling_baseline(profile: np.ndarray, window: int = 51) -> np.ndarray:
    """Rolling-median background.

    A global median is not usable here: powder background falls steeply with
    radius, so one threshold is simultaneously too high at low R and too low
    at high R.

    ``window`` must be comfortably wider than a peak and narrower than the
    background's own curvature; ~50 bins at 1 px binning is a reasonable start.
    """
    p = np.asarray(profile, dtype=np.float64)
    if window < 3:
        raise ValueError(f"window must be >= 3, got {window}")
    if window % 2 == 0:
        window += 1
    half = window // 2
    padded = np.pad(p, half, mode="edge")
    # Strided view -> one median per position, no Python loop.
    view = np.lib.stride_tricks.sliding_window_view(padded, window)
    return np.median(view, axis=-1)


def find_rings(
    radii_px: np.ndarray,
    profile: np.ndarray,
    *,
    geometry=None,
    baseline_window: int = 51,
    min_snr: float = 3.0,
    min_separation_px: float = 3.0,
    max_rings: int | None = None,
) -> list[Ring]:
    """Detect rings in a radial profile.

    Parameters
    ----------
    min_snr : float
        Height above the local baseline, in units of the local noise. 3.0 is
        deliberately permissive: for indexing, a missed ring is more damaging
        than a spurious one, because a correct cell then looks wrong.
    min_separation_px : float
        Minimum spacing between detected peaks. Set near the instrumental peak
        width; too large silently merges close pairs, which is how a real
        doublet becomes one mis-measured ring.

    Returns
    -------
    list[Ring]
        Sorted by radius. ``d_spacing_a`` is filled when *geometry* is given.
    """
    try:
        from scipy.signal import find_peaks, peak_widths
    except ImportError as exc:
        raise ImportError(
            "ring finding needs scipy. Install with `pip install scipy`."
        ) from exc

    r = np.asarray(radii_px, dtype=np.float64)
    p = np.asarray(profile, dtype=np.float64)
    if r.shape != p.shape:
        raise ValueError(f"radii {r.shape} and profile {p.shape} must match")

    base = rolling_baseline(p, baseline_window)
    resid = p - base
    # Local noise from the residual's MAD -- robust to the peaks themselves,
    # which a plain std is not.
    mad = np.median(np.abs(resid - np.median(resid)))
    noise = float(1.4826 * mad) if mad > 0 else float(np.std(resid))
    if noise <= 0:
        log.warning("profile has zero noise estimate; no rings reported")
        return []

    step = float(np.median(np.diff(r))) if r.size > 1 else 1.0
    distance = max(1, int(round(min_separation_px / max(step, 1e-9))))
    idx, props = find_peaks(resid, height=min_snr * noise, distance=distance)
    if idx.size == 0:
        log.info("no rings above %.1f sigma", min_snr)
        return []

    widths = peak_widths(resid, idx, rel_height=0.5)[0] * step
    prom = props.get("prominences")
    if prom is None:
        from scipy.signal import peak_prominences
        prom = peak_prominences(resid, idx)[0]

    d = None
    if geometry is not None:
        from .maps import radius_to_d_spacing
        d = radius_to_d_spacing(r[idx], geometry)

    rings = [
        Ring(radius_px=float(r[j]),
             d_spacing_a=(float(d[k]) if d is not None else None),
             height=float(resid[j]), prominence=float(prom[k]),
             width_px=float(widths[k]), snr=float(resid[j] / noise))
        for k, j in enumerate(idx)
    ]
    rings.sort(key=lambda x: x.radius_px)
    if max_rings is not None and len(rings) > max_rings:
        keep = sorted(rings, key=lambda x: x.snr, reverse=True)[:max_rings]
        rings = sorted(keep, key=lambda x: x.radius_px)
    log.info("found %d rings above %.1f sigma (noise %.4g)",
             len(rings), min_snr, noise)
    return rings
