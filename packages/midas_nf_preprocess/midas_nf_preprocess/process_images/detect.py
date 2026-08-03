"""Matched-filter spot detection, self-calibrated from the data.

The problem this solves
-----------------------
The goal is to threshold LOWER without perturbing intensities.  Denoising --
NLM in particular -- rewrites every pixel to get there, and is slow.  It is
also, measured, not the best way to get there.

The fix is to separate DETECTION from MEASUREMENT.  A filter decides *which*
pixels belong to spots; the intensities are then read from the **untouched**
median-corrected residual::

    score  = gaussian_filter(residual, sigma)      # detection only
    mask   = connected(score > threshold, >= min_px)
    values = residual[mask]                        # ORIGINAL intensities

Why a plain Gaussian
--------------------
Measured against the alternatives at a *matched false-positive budget* (see
below for how the budget is measured), on three independent datasets:

======================  ===============  ==============  ==============
detector                 nf_Ce_ht525_s2   Au_cube_0802    s6061_NF
======================  ===============  ==============  ==============
raw threshold                 49 blobs        20 blobs      (baseline)
NLM (h = sigma_MAD)           42 blobs        20 blobs
**matched Gaussian**       **420 blobs**   **115 blobs**
gain over NLM                  10.0x            5.8x
======================  ===============  ==============  ==============

NLM is not useless -- on ``nf_Ce_ht525_s2`` it reproduces its documented 3.0x
gain in >= 30 px area over a raw threshold.  But it buys that with a large
false-positive load (180 false blobs at threshold 2), and once false positives
are held fixed a matched Gaussian dominates it on every axis: more blobs, more
area, ~47x fewer single-pixel specks, and milliseconds instead of ~27 s/frame.

The optimum sigma is a broad plateau at **0.6-1.0 px** on both 1-ID (2048^2,
px 1.48 um) and 20-ID (4600x5320, px 0.548 um) data, so the setting transfers,
but it is cheap to confirm per dataset and :func:`calibrate_detector` does.

How the false-positive budget is measured -- no simulation
----------------------------------------------------------
Real spots are POSITIVE excursions of the median-corrected residual.  Running
the identical detector on the **negated** residual therefore counts false
positives at matched settings, from the data itself.

Caveat, stated because it matters: for Poisson counting data the positive tail
is heavier than the negative one, so this UNDER-counts false positives in an
absolute sense.  It is applied identically to every detector and every setting,
so the comparison and the calibration are fair; the returned FP number is a
lower bound, not an absolute rate.

Why sigma is scanned rather than derived
----------------------------------------
Estimating the PSF width analytically was tried and rejected.  A moment-based
estimate is inflated by any background pedestal, and an autocorrelation
estimate -- white noise contributes only at lag 0, so the residual ACF should
give the spot width -- swung 3.11 to 4.39 px between adjacent frames of the
same scan because large-scale background structure dominates the ACF at small
lags.  Scanning a handful of sigmas costs milliseconds and optimises the
quantity actually wanted.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np

__all__ = ["DetectorCalibration", "matched_filter_score", "detect_mask",
           "calibrate_detector", "gaussian_blur_torch",
           "detect_labels_torch", "DEFAULT_SIGMAS"]

#: Scanned by default.  0.0 means "no filter", i.e. a plain threshold.
DEFAULT_SIGMAS: Tuple[float, ...] = (0.0, 0.5, 0.7, 0.85, 1.0, 1.4, 2.0)

_STRUCT = np.ones((3, 3), int)


@dataclass
class DetectorCalibration:
    """Chosen operating point, with the evidence for it."""

    sigma: float
    threshold: float
    n_blobs: int
    n_false: int
    area: int
    fp_budget: int
    n_evaluated: int

    @property
    def ok(self) -> bool:
        return self.n_blobs > 0 and self.n_false <= self.fp_budget

    def report(self) -> str:
        return (f"matched filter: sigma={self.sigma:.2f} threshold={self.threshold:.3f}"
                f" -> {self.n_blobs} blobs ({self.area} px), {self.n_false} false"
                f" positives (budget {self.fp_budget}, measured on the negated"
                f" residual, a LOWER BOUND for Poisson data);"
                f" {self.n_evaluated} settings evaluated")


def matched_filter_score(resid, sigma: float):
    """Detection score.  ``sigma <= 0`` returns the residual unfiltered."""
    from scipy import ndimage as ndi

    a = np.asarray(resid, dtype=np.float32)
    return a if sigma <= 0 else ndi.gaussian_filter(a, float(sigma))


def detect_mask(resid, *, sigma: float, threshold: float, min_px: int = 4):
    """Boolean mask of detected spots.

    The mask comes from the FILTERED score; nothing about ``resid`` is altered,
    so ``resid[mask]`` are the original intensities.
    """
    from scipy import ndimage as ndi

    score = matched_filter_score(resid, sigma)
    lab, n = ndi.label(score > threshold, structure=_STRUCT)
    if n == 0:
        return np.zeros(np.shape(resid), dtype=bool)
    sizes = np.bincount(lab.ravel())
    keep = np.where(sizes >= min_px)[0]
    keep = keep[keep > 0]
    return np.isin(lab, keep)


def _count(score, threshold: float, min_px: int) -> Tuple[int, int]:
    from scipy import ndimage as ndi

    lab, n = ndi.label(score > threshold, structure=_STRUCT)
    if n == 0:
        return 0, 0
    s = np.bincount(lab.ravel())[1:]
    big = s >= min_px
    return int(big.sum()), int(s[big].sum())


def calibrate_detector(
    resid,
    *,
    sigmas: Sequence[float] = DEFAULT_SIGMAS,
    fp_budget: int = 5,
    min_px: int = 4,
    n_thresholds: int = 18,
) -> DetectorCalibration:
    """Pick ``(sigma, threshold)`` maximising detections within an FP budget.

    The budget is enforced against false positives counted on the NEGATED
    residual -- see the module docstring for what that number does and does not
    mean.

    Returns the best operating point found.  Check :attr:`DetectorCalibration.ok`;
    when nothing meets the budget the least-bad point is returned with
    ``ok == False`` rather than an exception, so a caller can fall back.
    """
    r = np.asarray(resid, dtype=np.float32)
    best: Optional[DetectorCalibration] = None
    fallback: Optional[DetectorCalibration] = None
    n_eval = 0

    for sigma in sigmas:
        pos = matched_filter_score(r, sigma)
        neg = matched_filter_score(-r, sigma)
        hi = float(np.percentile(pos, 99.995))
        if not np.isfinite(hi) or hi <= 0:
            continue
        for t in np.linspace(0.05 * hi, 0.8 * hi, n_thresholds):
            nb, area = _count(pos, float(t), min_px)
            nf, _ = _count(neg, float(t), min_px)
            n_eval += 1
            cand = DetectorCalibration(float(sigma), float(t), nb, nf, area,
                                       fp_budget, n_eval)
            if nf <= fp_budget:
                if best is None or nb > best.n_blobs:
                    best = cand
            elif fallback is None or nf < fallback.n_false:
                fallback = cand

    out = best or fallback or DetectorCalibration(0.0, float("inf"), 0, 0, 0,
                                                  fp_budget, n_eval)
    out.n_evaluated = n_eval
    return out


# ---------------------------------------------------------------------------
# torch path -- used per frame, so it must stay on the device
# ---------------------------------------------------------------------------

def gaussian_blur_torch(x, sigma: float, truncate: float = 4.0):
    """Separable Gaussian blur matching ``scipy.ndimage.gaussian_filter``.

    ``reflect`` padding to match scipy's default mode, so a calibration done
    with the numpy path transfers to the torch path unchanged.
    """
    import torch
    import torch.nn.functional as F

    if sigma <= 0:
        return x
    rad = max(1, int(truncate * float(sigma) + 0.5))
    t = torch.arange(-rad, rad + 1, dtype=torch.float32, device=x.device)
    k = torch.exp(-(t ** 2) / (2.0 * float(sigma) ** 2))
    k = (k / k.sum()).to(x.dtype)
    a = x[None, None]
    a = F.pad(a, (rad, rad, 0, 0), mode="reflect")
    a = F.conv2d(a, k.view(1, 1, 1, -1))
    a = F.pad(a, (0, 0, rad, rad), mode="reflect")
    a = F.conv2d(a, k.view(1, 1, -1, 1))
    return a[0, 0]


def detect_labels_torch(resid, *, sigma: float, threshold: float,
                        min_px: int = 4):
    """Detection mask -> connected-component labels, entirely in torch.

    Returns ``(labels, n_components, score)``.  ``resid`` is NOT modified, so
    the caller reads intensities from it directly.
    """
    import torch

    from .peaks import label_components

    score = gaussian_blur_torch(resid, sigma)
    with torch.no_grad():
        mask = score.detach() > float(threshold)
        labels, n = label_components(mask, return_n=True)
        if min_px > 1 and n > 0:
            flat = labels.reshape(-1)
            counts = torch.bincount(flat)
            too_small = counts < int(min_px)
            if bool(too_small.any()):
                drop = torch.zeros_like(counts, dtype=torch.bool)
                drop[too_small] = True
                drop[0] = True                      # background stays 0
                labels = torch.where(drop[labels], torch.zeros_like(labels), labels)
                n = int((torch.bincount(labels.reshape(-1)) > 0).sum().item()) - 1
    return labels, max(n, 0), score
