"""Image-level sensitivity control.

Synthetic spots on a candidate's line pattern (free scale drawn from a window,
random azimuth) are added as Poisson counts to REAL frames, and the identical
per-window pass is run. Recovery and "called off-matrix" rates vs intensity give
the detection limit of THIS dataset. A list-level injection (adding d-values to
a detection list) says nothing about sensitivity and is not offered.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

from midas_peakfit.snapshot_detect import valid_region

from .config import SnapshotConfig
from .geometry import Maps
from .io import list_frames, load_mask, sum_frames
from .pipeline import process_image


def _place(img, maps: Maps, d_lines, s_range, taken, rng, allowed, sep=12):
    """Place one spot on the candidate pattern INSIDE the region where the detector
    may report a spot (``allowed``: valid, margin, maximum angle). Placing spots where
    detection is excluded by design would mix coverage with sensitivity."""
    for _ in range(200):
        d = rng.choice(d_lines) * rng.uniform(*s_range)
        t = np.degrees(2 * np.arcsin(maps.wavelength / (2 * d)))
        band = (np.abs(maps.tth - t) < 0.006) & allowed
        if not band.any():
            continue
        rr, cc = np.nonzero(band)
        k = rng.integers(len(rr))
        r0, c0 = rr[k] + rng.uniform(-0.5, 0.5), cc[k] + rng.uniform(-0.5, 0.5)
        ri, ci = int(round(r0)), int(round(c0))
        if ri < 8 or ci < 8 or ri > img.shape[0] - 9 or ci > img.shape[1] - 9:
            continue
        if (img[ri - 6:ri + 7, ci - 6:ci + 7] < 0).any():
            continue
        if any((r0 - a) ** 2 + (c0 - b) ** 2 < sep * sep for a, b in taken):
            continue
        return r0, c0, d
    return None


def _inject(img, r0, c0, flux, sigma, rng):
    ri, ci = int(round(r0)), int(round(c0))
    rr, cc = np.mgrid[ri - 6:ri + 7, ci - 6:ci + 7]
    lam = flux * np.exp(-((rr - r0) ** 2 + (cc - c0) ** 2) / (2 * sigma ** 2)) / (2 * np.pi * sigma ** 2)
    img[ri - 6:ri + 7, ci - 6:ci + 7] += rng.poisson(lam)


def injection_curve(cfg: SnapshotConfig, maps: Maps, matrix_lines, fit_lines, T: float,
                    d_lines: Sequence[float], window: Sequence[int], W: int, sigma_matrix: float,
                    levels=(5, 10, 20, 40, 80, 160, 320), n_images=20, per_image=6,
                    s_range=(1.0, 1.025), r_match=2.5, extra=()) -> dict:
    """Recovery and off-matrix rate vs injected counts PER FRAME (x W in a sum)."""
    rng = np.random.default_rng(cfg.seed)
    files = list_frames(cfg.frames)
    mask = load_mask(cfg.mask, cfg.flip)
    starts = np.linspace(window[0], window[1] - W + 1, n_images).astype(int)
    d_lines = np.asarray(d_lines, float)
    on = maps.tth[np.isfinite(maps.tth)]
    dmin = maps.wavelength / (2 * np.sin(np.radians(on.max() / 2)))
    d_lines = d_lines[d_lines > dmin]
    ok0 = sum_frames(files[starts[0]:starts[0] + 1], cfg.flip, mask, cfg.invalid_below) >= 0
    allowed = valid_region(ok0, 3, cfg.margin_px + 2)
    if cfg.tth_max:
        allowed &= maps.tth < cfg.tth_max - 0.05
    # coverage: fraction of the pattern's ring pixels (detector-wide) where a spot can be reported
    ring = np.zeros_like(ok0)
    for d in d_lines:
        t = np.degrees(2 * np.arcsin(maps.wavelength / (2 * d)))
        ring |= np.abs(maps.tth - t) < 0.006
    coverage = float((ring & allowed).sum() / max((ring & ok0).sum(), 1))
    out = {lv: dict(n=0, recovered=0, off=0) for lv in levels}
    for st in starts:
        base_img = sum_frames(files[st:st + W], cfg.flip, mask, cfg.invalid_below)
        base = process_image(base_img.copy(), W, int(st), cfg, maps, matrix_lines, fit_lines, T, extra)[1]
        for lv in levels:
            img = base_img.copy()
            taken = [(r, c) for r, c in base[:, 1:3]]
            inj = []
            for _ in range(per_image):
                pl = _place(img, maps, d_lines, s_range, taken, rng, allowed)
                if pl is None:
                    continue
                _inject(img, pl[0], pl[1], lv * W, cfg.sigma_px, rng)
                taken.append(pl[:2])
                inj.append(pl)
            det = process_image(img, W, int(st), cfg, maps, matrix_lines, fit_lines, T, extra)[1]
            for r0, c0, _ in inj:
                out[lv]["n"] += 1
                if len(det):
                    dist = np.hypot(det[:, 1] - r0, det[:, 2] - c0)
                    j = int(np.argmin(dist))
                    if dist[j] < r_match:
                        out[lv]["recovered"] += 1
                        rel = det[j, 9]
                        if np.isfinite(rel) and abs(rel) > cfg.nsig_off * sigma_matrix:
                            out[lv]["off"] += 1
    curve = {str(lv): dict(n=v["n"], recovered=v["recovered"] / max(v["n"], 1),
                           called_off_matrix=v["off"] / max(v["n"], 1)) for lv, v in out.items()}
    return dict(curve=curve, pattern_coverage=coverage,
                note="spots placed only where detection is allowed; pattern_coverage is the fraction "
                     "of the pattern's ring pixels in that region")
