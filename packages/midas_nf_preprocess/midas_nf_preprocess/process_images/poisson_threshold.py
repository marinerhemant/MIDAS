"""Spot detection against a LOCAL Poisson background (``SpotDetect poisson``).

The problem this solves
-----------------------
Photon-starved NF frames (20-ID Oryx, dark-subtracted at the DAQ) are mostly
zeros with sparse single counts, and the count rate is not uniform across the
detector: on ``bt_20id_sep26`` PUR #255 the mean residual is ~1.18 counts/px in
one band and ~1.47 in another (rows 2900-3700). A single global threshold is
then either too high for the quiet rows (weak spots lost) or too low for the
busy ones (background clumps detected). Measured there: dropping the NLM+global
threshold from 8 to 4 counts recovered 4.3x more FF-validated voxels, but ~23 %
of the new blobs were peakless background clumps pinned to the busy band at
every omega.

Neither existing global route fixes that. ``BlanketSigma`` scales one threshold
by a noise estimate; the matched detector calibrates one threshold by counting
false positives on the NEGATED residual, which has no negative tail when the
background is sparse counts over a zero median (its own docstring's caveat).

The method
----------
1. **Local rate.** ``rate(z, y)`` = mean over evenly spaced frames of the
   median-corrected residual, clipped at ``clip`` so spots (few frames, many
   counts) do not raise it, then box-smoothed over ``smooth_px``.
2. **Statistic.** The window sum ``S`` of the clamped residual over a ``k x k``
   window. Under background alone ``S ~ Poisson(k^2 * rate)``.
3. **Threshold map.** ``T(z, y)`` = the smallest integer with
   ``P(Poisson(k^2 rate) > T) <= p``, ``p = fp_per_frame / n_pixels``: the
   same expected number of background false alarms per frame wherever the
   detector sits, busy or quiet. Quantiles are tabulated on a rate grid.
4. **Detect.** ``S > T``, connected components, ``>= min_px``. The residual
   itself is not modified, so intensities downstream are the original ones.

Dispersion is MEASURED, not assumed (added 2026-09-29 after the first real-data
run). A scintillator-coupled camera does not record photon counts: on the
20-ID Oryx one X-ray lights a cluster of pixels, ADU come in steps of 2-4, and
the 3x3 window sum of the background has var/mean of about 2.7 (MAD-based,
measured in both a quiet and a busy band of PUR #255), not 1, so a Poisson
threshold was far too low (3878 blobs/frame). The window sum's local MEAN and
VARIANCE are therefore both estimated from the sampled frames, and the threshold
is the upper quantile of a negative binomial with those two moments -- Poisson
where var = mean. The variance is robust (MAD over frames): the plain variance
read 22.8 in the busy band, an artefact of the few windows that ever contain a
real spot, and setting the threshold from it was ~10x too high there.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F


def auto_clip(rate_guess: float) -> float:
    """A per-pixel clip for the rate estimate: generous for background, below spots."""
    return float(max(3.0, np.ceil(rate_guess + 6.0 * np.sqrt(max(rate_guess, 1e-6)))))


def _box_mean(img: torch.Tensor, k: int) -> torch.Tensor:
    if k <= 1:
        return img
    pad = k // 2
    x = F.pad(img[None, None], (pad, k - 1 - pad, pad, k - 1 - pad), mode="replicate")
    return F.avg_pool2d(x, kernel_size=k, stride=1)[0, 0]


def window_sum(resid: torch.Tensor, k: int) -> torch.Tensor:
    """Sum of the clamped (>= 0) residual over a ``k x k`` window centred on each pixel."""
    return _box_mean(torch.clamp(resid, min=0), k) * (k * k)


def local_rate_from_stack(stack: torch.Tensor, median: torch.Tensor, *, n_frames: int = 60,
                          clip: float = 0.0, smooth_px: int = 32) -> torch.Tensor:
    """Background rate map ``[Z, Y]`` from a materialised ``[N, Z, Y]`` stack."""
    n = stack.shape[0]
    idx = np.unique(np.linspace(0, n - 1, min(int(n_frames) or n, n)).round().astype(int))
    r = torch.clamp(stack[idx].to(torch.float32) - median.to(torch.float32), min=0)
    c = float(clip) if clip > 0 else auto_clip(float(r.mean()))
    return _box_mean(torch.clamp(r, max=c).mean(0), int(smooth_px))


def streaming_local_rate(source, median: torch.Tensor, *, n_frames: int = 60, clip: float = 0.0,
                         smooth_px: int = 32, row_block: int = 460) -> torch.Tensor:
    """Background rate map without holding the layer: evenly spaced frames, row bands.

    ``source`` is a :class:`~midas_nf_preprocess.process_images.io.FrameSource`.
    """
    total = int(source.n_frames)
    idx = sorted(set(int(i) for i in np.linspace(0, total - 1, min(int(n_frames) or total, total)).round()))
    nz, ny = int(source.nz), int(source.ny)
    med = median.detach().to("cpu", torch.float32)
    # first pass on a band to set the clip if not given
    c = float(clip)
    if c <= 0:
        mid = nz // 2
        band = torch.from_numpy(source.read_rows(idx[:: max(1, len(idx) // 8)], mid, min(mid + 64, nz))).float()
        c = auto_clip(float(torch.clamp(band - med[mid:min(mid + 64, nz)], min=0).mean()))
    out = torch.empty((nz, ny), dtype=torch.float32)
    blk = int(row_block) if row_block and row_block > 0 else nz
    for r0 in range(0, nz, blk):
        r1 = min(r0 + blk, nz)
        band = torch.from_numpy(source.read_rows(idx, r0, r1)).float()
        out[r0:r1] = torch.clamp(torch.clamp(band - med[r0:r1], min=0), max=c).mean(0)
    return _box_mean(out, int(smooth_px))


def threshold_map(rate: torch.Tensor, *, window: int = 3, fp_per_frame: float = 5.0,
                  n_levels: int = 256) -> torch.Tensor:
    """Per-pixel integer threshold on the window sum for a fixed false-alarm budget.

    ``T`` is the smallest integer with ``P(Poisson(window^2 * rate) > T) <= p``,
    ``p = fp_per_frame / rate.numel()``. Tabulated on ``n_levels`` rates between
    the map's min and max (log-spaced), each pixel taking the table entry at or
    above its own rate, so the realised false-alarm rate never exceeds target
    from the tabulation.
    """
    from scipy.stats import poisson

    lam = rate.detach().to("cpu", torch.float64).numpy() * (window * window)
    p = float(fp_per_frame) / lam.size
    lo, hi = max(lam.min(), 1e-6), max(lam.max(), 1e-6) * 1.0001
    grid = np.geomspace(lo, hi, int(n_levels)) if hi > lo else np.array([hi])
    t_grid = poisson.isf(p, grid)                        # smallest t with P(X > t) <= p
    j = np.clip(np.searchsorted(grid, lam, side="left"), 0, len(grid) - 1)
    return torch.from_numpy(t_grid[j].astype(np.float32)).to(rate.device)


def _moments_from_window_sums(S: torch.Tensor, *, smooth_px: int,
                              robust: bool) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-pixel mean and variance over frames of ``S`` ``[n, Z, Y]``, then box-smoothed.

    ``robust`` uses ``(1.4826 * MAD)^2`` over frames for the variance. The plain
    variance is dominated by the few windows that ever sit on a real spot: on
    PUR #255 it read var/mean = 22.8 in the busy band, against 2.7 from the MAD
    (2.6-2.7 in BOTH bands) -- an 8x inflation that would set the threshold far too
    high exactly where the static background is strongest. MAD is ~4 % low for a
    right-skewed count distribution at these means (quiet band: 36.4 vs 38.1).
    """
    mean = S.mean(0)
    if robust:
        med = S.median(0).values
        mad = (S - med).abs().median(0).values
        var = (1.4826 * mad) ** 2
    else:
        var = S.var(0, unbiased=False)
    return _box_mean(mean, int(smooth_px)), _box_mean(var, int(smooth_px))


def _sampled_frames(n: int, n_frames: int) -> np.ndarray:
    return np.unique(np.linspace(0, n - 1, min(int(n_frames) or n, n)).round().astype(int))


def local_moments_from_stack(stack: torch.Tensor, median: torch.Tensor, *, window: int = 3,
                             n_frames: int = 60, clip: float = 0.0, smooth_px: int = 9,
                             robust: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
    """Local mean and variance of the window sum ``[Z, Y]`` from a ``[N, Z, Y]`` stack.

    Each sampled frame's clamped, clipped residual is window-summed; the mean and
    variance over frames are then box-smoothed over ``smooth_px``. ``clip``
    (per pixel, 0 = auto from the mean) limits how much any one spot pixel can
    add. ``smooth_px`` defaults to 9. The background of dark-subtracted sparse-count
    data has static structure at that scale (PUR #255: ~0.10 counts/px in a quiet
    band, ~0.45 in a busy one, against a 1.6 mean) and ~90 % of pixels have a
    temporal median of 0, so median subtraction does not remove it. Whether the
    finer map matters for DETECTION is not established: on PUR #255 frames 9 px
    and 32 px gave the same blob counts, lit area and static overlap to within
    0.1 %. It is kept because it is cheap and a static clump cannot then raise
    only its own neighbourhood's expected sum unnoticed.
    """
    n = stack.shape[0]
    idx = _sampled_frames(n, n_frames)
    med = median.to(torch.float32)
    c = float(clip)
    if c <= 0:
        c = auto_clip(float(torch.clamp(stack[idx[: max(1, len(idx) // 6)]].float() - med, min=0).mean()))
    S = torch.stack([window_sum(torch.clamp(stack[int(j)].float() - med, min=0, max=c), int(window))
                     for j in idx])
    return _moments_from_window_sums(S, smooth_px=smooth_px, robust=robust)


def streaming_local_moments(source, median: torch.Tensor, *, window: int = 3, n_frames: int = 60,
                            clip: float = 0.0, smooth_px: int = 9, robust: bool = True,
                            row_block: int = 460) -> Tuple[torch.Tensor, torch.Tensor]:
    """:func:`local_moments_from_stack` without holding the layer: row bands with a halo of
    ``max(window // 2, smooth_px // 2) + 1`` rows so window sums and the smoothing are exact at band edges."""
    total = int(source.n_frames)
    idx = _sampled_frames(total, n_frames).tolist()
    nz, ny = int(source.nz), int(source.ny)
    med = median.detach().to("cpu", torch.float32)
    c = float(clip)
    if c <= 0:
        mid = nz // 2
        band = torch.from_numpy(source.read_rows(idx[:: max(1, len(idx) // 8)], mid, min(mid + 64, nz))).float()
        c = auto_clip(float(torch.clamp(band - med[mid:min(mid + 64, nz)], min=0).mean()))
    h = max(int(window) // 2, int(smooth_px) // 2) + 1
    mean = torch.empty((nz, ny)); var = torch.empty((nz, ny))
    blk = int(row_block) if row_block and row_block > 0 else nz
    for r0 in range(0, nz, blk):
        r1 = min(r0 + blk, nz); a0, a1 = max(r0 - h, 0), min(r1 + h, nz)
        band = torch.from_numpy(source.read_rows(idx, a0, a1)).float()
        res = torch.clamp(band - med[a0:a1], min=0, max=c)
        S = torch.stack([window_sum(f, int(window)) for f in res])
        m, v = _moments_from_window_sums(S, smooth_px=smooth_px, robust=robust)
        mean[r0:r1] = m[r0 - a0:r0 - a0 + (r1 - r0)]; var[r0:r1] = v[r0 - a0:r0 - a0 + (r1 - r0)]
    return mean, var


def threshold_map_nb(mean: torch.Tensor, var: torch.Tensor, *, fp_per_frame: float = 5.0,
                     n_levels: int = 64) -> torch.Tensor:
    """Per-pixel threshold on the window sum from its local mean and variance.

    Negative binomial with matched moments (``r = mean^2/(var-mean)``,
    ``p = mean/var``); where ``var <= mean`` it is Poisson(mean). Upper quantile
    at ``P(S > T) <= fp_per_frame / n_pixels``. Tabulated on a (mean, dispersion)
    grid, each pixel taking the entry at or above its own values, so tabulation
    can only make the threshold conservative.
    """
    from scipy.stats import nbinom, poisson

    mu = np.maximum(mean.detach().cpu().double().numpy(), 1e-6)
    D = np.maximum(var.detach().cpu().double().numpy() / mu, 1.0)          # dispersion var/mean
    p_fa = float(fp_per_frame) / mu.size
    def grid(x):
        lo, hi = float(x.min()), float(x.max()) * 1.0001
        return np.geomspace(lo, hi, n_levels) if hi > lo * 1.0001 else np.array([hi])
    gm, gd = grid(mu), grid(D)
    M, DD = np.meshgrid(gm, gd, indexing="ij")
    T = np.where(DD <= 1.0 + 1e-9, poisson.isf(p_fa, M),
                 nbinom.isf(p_fa, M / np.maximum(DD - 1.0, 1e-9), 1.0 / DD))
    i = np.clip(np.searchsorted(gm, mu, side="left"), 0, len(gm) - 1)
    k = np.clip(np.searchsorted(gd, D, side="left"), 0, len(gd) - 1)
    return torch.from_numpy(T[i, k].astype(np.float32)).to(mean.device)


def detect_labels_poisson(resid: torch.Tensor, thr_map: torch.Tensor, *, window: int = 3,
                          min_px: int = 4) -> Tuple[torch.Tensor, int, torch.Tensor]:
    """Mask = window sum > threshold map -> connected-component labels.

    Returns ``(labels, n_components, score)`` with ``score = S - T`` (positive
    where detected). ``resid`` is not modified.
    """
    from .peaks import label_components

    S = window_sum(resid.to(torch.float32), int(window))
    score = S - thr_map.to(S.device)
    with torch.no_grad():
        labels, n = label_components(score > 0, return_n=True)
        if min_px > 1 and n > 0:
            counts = torch.bincount(labels.reshape(-1))
            drop = counts < int(min_px)
            drop[0] = True
            labels = torch.where(drop[labels], torch.zeros_like(labels), labels)
            n = int((torch.bincount(labels.reshape(-1)) > 0).sum().item()) - 1
    return labels, max(n, 0), score


def expected_false_alarms(rate: torch.Tensor, thr_map: torch.Tensor, window: int = 3) -> float:
    """Expected background exceedances per frame for a rate map and threshold map."""
    from scipy.stats import poisson

    lam = rate.detach().cpu().double().numpy() * (window * window)
    return float(poisson.sf(thr_map.detach().cpu().double().numpy(), lam).sum())
