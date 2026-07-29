"""Kullback-Leibler and Jensen-Shannon divergence for binned histograms."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def _normalize_hist(h: NDArray[np.floating], bin_widths: NDArray[np.floating]) -> NDArray[np.floating]:
    h = np.asarray(h, dtype=float).clip(min=0)
    mass = float((h * bin_widths).sum())
    if mass <= 0:
        raise ValueError("histogram has zero total mass")
    return h / mass


def _bin_widths_from_centers(bin_centers: NDArray[np.floating]) -> NDArray[np.floating]:
    bc = np.asarray(bin_centers, dtype=float)
    if bc.size < 2:
        raise ValueError("need at least 2 bin centers")
    # Treat as uniform; for non-uniform bins caller should pass widths via
    # *_widths variants below.
    dx = bc[1] - bc[0]
    return np.full_like(bc, dx)


def kl_divergence(
    observed_hist: NDArray[np.floating],
    reference_hist: NDArray[np.floating],
    bin_centers: NDArray[np.floating],
    epsilon: float = 1e-10,
) -> float:
    """KL(observed || reference) in nats.

    Both histograms are renormalized to integrate to 1 over ``bin_centers``
    before evaluation. ``epsilon`` regularises zero-probability bins in the
    reference (avoids -inf) and clips zero-probability bins in the observed
    distribution (which contribute 0 in the limit).
    """
    widths = _bin_widths_from_centers(bin_centers)
    p = _normalize_hist(observed_hist, widths)
    q = _normalize_hist(reference_hist, widths)
    q = q + epsilon
    p_safe = p + epsilon
    integrand = np.where(p > 0, p * np.log(p_safe / q), 0.0)
    return float((integrand * widths).sum())


def jensen_shannon_divergence(
    observed_hist: NDArray[np.floating],
    reference_hist: NDArray[np.floating],
    bin_centers: NDArray[np.floating],
) -> float:
    """JS divergence in nats. Symmetric, bounded in ``[0, ln 2]``."""
    widths = _bin_widths_from_centers(bin_centers)
    p = _normalize_hist(observed_hist, widths)
    q = _normalize_hist(reference_hist, widths)
    m = 0.5 * (p + q)
    # Compute log args on a safe denominator first, then mask zero contributions
    # so log() never sees 0 (avoids RuntimeWarning).
    safe_m = np.where(m > 0, m, 1.0)
    log_pm = np.log(np.where(p > 0, p / safe_m, 1.0))
    log_qm = np.log(np.where(q > 0, q / safe_m, 1.0))
    return 0.5 * float((p * log_pm * widths).sum()) + 0.5 * float((q * log_qm * widths).sum())


__all__ = ["kl_divergence", "jensen_shannon_divergence"]
