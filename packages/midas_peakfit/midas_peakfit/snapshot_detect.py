"""Spot detection on single (or summed) still frames with sparse spots.

For data where the sample does not rotate: each frame holds a few compact Bragg
spots on a low, slowly varying background (possibly a broad diffuse halo). The
detector here is deliberately simple and deliberately *calibrated*:

1. **Background is local**, not a per-ring mean. A per-ring mean underestimates
   the background wherever it varies with azimuth (detector corners, shadows),
   and single photons there then look like very significant spots. The local
   estimate is a masked box mean after capping bright pixels at a coarse
   background (typically the ring mean) plus ``cap_nsig`` Poisson sigmas.
2. **The threshold is measured**, not chosen: pure-Poisson null images are drawn
   from the frame's own background and mask, and the threshold is the score that
   yields ``fa_per_image`` false peaks per image. No absolute count floor is
   applied -- a floor that scales with the number of summed frames cancels the
   gain from summing.
3. Two scores are available. ``"gauss"``: Gaussian matched filter divided by its
   Poisson standard deviation (fast, inefficient at very low counts).
   ``"poisson"``: aperture counts against the aperture's expected background,
   scored as ``-log10 P(X >= N | mu)`` (exact at low counts).

Nothing here knows about materials, rings or phases; the coarse background is
an input. Geometry (pixel -> scattering angle) is the caller's business.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy import ndimage
from scipy.stats import poisson

__all__ = [
    "SnapshotDetectorConfig",
    "ring_mean_background",
    "local_background",
    "valid_region",
    "score_map",
    "calibrate_threshold",
    "measure_spot_sigma",
    "detect_spots",
    "SPOT_FIELDS",
]

SPOT_FIELDS = ("row", "col", "flux", "peak", "score")


@dataclass
class SnapshotDetectorConfig:
    sigma_px: float = 1.7          # matched-filter width (measure it: measure_spot_sigma)
    local_box: int = 31            # local background box (px)
    cap_nsig: float = 5.0          # cap bright pixels before the local mean
    edge_px: int = 3               # erosion of the valid mask (score support)
    margin_px: int = 0             # extra exclusion distance from invalid pixels
    fa_per_image: float = 0.05     # target false peaks per image
    n_null_images: int = 40
    statistic: str = "gauss"       # "gauss" | "poisson"
    aperture_radius: float = 2.5   # poisson aperture radius (px)
    flux_box: int = 3              # half-size of the flux / centroid box


def ring_mean_background(img: np.ndarray, ok: np.ndarray, ring_index: np.ndarray,
                         min_pixels: int = 30) -> tuple[np.ndarray, np.ndarray]:
    """Mean intensity per ring bin and its per-pixel map.

    ``ring_index`` is a non-negative integer bin per pixel (e.g. from a
    scattering-angle map). The *mean* is used on purpose: at a few ms per frame
    most pixels hold 0 or 1 count and a median rails at those values.
    """
    b = ring_index[ok]
    nb = int(ring_index.max()) + 1
    num = np.bincount(b, weights=img[ok].astype(np.float64), minlength=nb)
    cnt = np.bincount(b, minlength=nb)
    prof = np.where(cnt >= min_pixels, num / np.maximum(cnt, 1), np.nan)
    return prof, np.nan_to_num(prof)[ring_index]


def local_background(img: np.ndarray, ok: np.ndarray, coarse: np.ndarray,
                     box: int = 31, cap_nsig: float = 5.0) -> np.ndarray:
    """Masked local mean after capping bright pixels at ``coarse`` (see module doc)."""
    cap = coarse + cap_nsig * np.sqrt(coarse + 1.0)
    x = np.where(ok & (img > cap), coarse, np.where(ok, img, 0.0))
    num = ndimage.uniform_filter(np.where(ok, x, 0.0), box)
    den = ndimage.uniform_filter(ok.astype(np.float64), box)
    return np.where(ok, num / np.maximum(den, 1e-6), 0.0)


def valid_region(ok: np.ndarray, edge_px: int = 3, margin_px: int = 0) -> np.ndarray:
    """Pixels where a peak may be reported: eroded valid mask, optionally at least
    ``margin_px`` from any invalid pixel (module edges and gaps carry artefacts)."""
    safe = ndimage.binary_erosion(ok, iterations=max(edge_px, 1))
    if margin_px > 0:
        safe &= ndimage.distance_transform_edt(ok) >= margin_px
    return safe


def _disk(radius: float) -> np.ndarray:
    r = int(np.ceil(radius))
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    return (yy * yy + xx * xx <= radius * radius).astype(np.float64)


def score_map(img: np.ndarray, ok: np.ndarray, bkg: np.ndarray,
              cfg: SnapshotDetectorConfig) -> tuple[np.ndarray, np.ndarray]:
    """Return (residual, score). Higher score = less compatible with background."""
    res = np.where(ok, img - bkg, 0.0)
    if cfg.statistic == "gauss":
        okf = ok.astype(np.float64)
        norm = ndimage.gaussian_filter(okf, cfg.sigma_px)
        F = ndimage.gaussian_filter(res, cfg.sigma_px) / np.maximum(norm, 1e-6)
        k2 = 1.0 / (4 * np.pi * cfg.sigma_px ** 2)
        V = np.maximum(bkg, 1e-3) * k2 / np.maximum(norm, 1e-6) ** 2
        return res, F / np.sqrt(V)
    if cfg.statistic == "poisson":
        k = _disk(cfg.aperture_radius)
        N = ndimage.convolve(np.where(ok, img, 0.0), k, mode="constant")
        mu = ndimage.convolve(np.where(ok, bkg, 0.0), k, mode="constant")
        Ni = np.rint(np.maximum(N, 0)).astype(np.int64)
        with np.errstate(divide="ignore"):
            s = -np.log10(np.clip(poisson.sf(Ni - 1, np.maximum(mu, 1e-6)), 1e-300, 1.0))
        return res, s
    raise ValueError(f"unknown statistic {cfg.statistic!r}")


def _peaks(score, safe, T):
    mx = ndimage.maximum_filter(score, size=3)
    return np.argwhere((score == mx) & (score > T) & safe)


def calibrate_threshold(img: np.ndarray, ok: np.ndarray, coarse: np.ndarray,
                        cfg: SnapshotDetectorConfig,
                        rng: Optional[np.random.Generator] = None,
                        ring_index: Optional[np.ndarray] = None) -> float:
    """Threshold giving ``cfg.fa_per_image`` false peaks per image on Poisson nulls.

    The null images are Poisson draws of this image's local background (same
    mask), processed exactly as real images: coarse background re-estimated
    (from ``ring_index`` if given, else the same ``coarse``), local background,
    score, local maxima.
    """
    rng = rng if rng is not None else np.random.default_rng(0)
    bkg = local_background(img, ok, coarse, cfg.local_box, cfg.cap_nsig)
    safe = valid_region(ok, cfg.edge_px, cfg.margin_px)
    scores = []
    for _ in range(cfg.n_null_images):
        nul = np.where(ok, rng.poisson(np.maximum(bkg, 0.0)), 0).astype(np.float64)
        c = ring_mean_background(nul, ok, ring_index)[1] if ring_index is not None else coarse
        bk = local_background(nul, ok, c, cfg.local_box, cfg.cap_nsig)
        _, s = score_map(nul, ok, bk, cfg)
        mx = ndimage.maximum_filter(s, size=3)
        scores.append(s[(s == mx) & safe])
    allp = np.sort(np.concatenate(scores))[::-1]
    k = int(np.floor(cfg.fa_per_image * cfg.n_null_images))
    return float(allp[k]) if k < len(allp) else float(allp[-1])


def _pix_gauss(rc, amp, r0, c0, sig, bg):
    """Isotropic 2-D Gaussian integrated over unit pixels (erf form), plus a constant."""
    from scipy.special import erf
    r, c = rc
    k = np.sqrt(2.0) * sig
    fr = 0.5 * (erf((r + 0.5 - r0) / k) - erf((r - 0.5 - r0) / k))
    fc = 0.5 * (erf((c + 0.5 - c0) / k) - erf((c - 0.5 - c0) / k))
    return amp * fr * fc + bg


def measure_spot_sigma(img: np.ndarray, ok: np.ndarray, coarse: np.ndarray,
                       cfg: SnapshotDetectorConfig, n_spots: int = 200, isolation_px: float = 10.0,
                       half: int = 4) -> tuple[float, int]:
    """Width of real compact spots: median sigma (px) of a pixel-integrated isotropic Gaussian
    fitted to each of the ``n_spots`` highest-scoring isolated strong peaks (no other strong peak
    within ``isolation_px``). The matched filter and any injection test should use this, not an
    assumed width: on one detector real spots measured ~1.0 px against an assumed 1.7 px, and
    at equal flux a real-width spot scored 1.6-1.8x higher.
    Returns (sigma, number of spots used); (nan, 0) if none qualify."""
    from scipy.optimize import curve_fit
    bkg = local_background(img, ok, coarse, cfg.local_box, cfg.cap_nsig)
    res, s = score_map(img, ok, bkg, cfg)
    safe = valid_region(ok, max(cfg.edge_px, half + 1), cfg.margin_px)
    mx = ndimage.maximum_filter(s, size=3)
    rr, cc = np.nonzero((s == mx) & safe & np.isfinite(s))
    if rr.size == 0:
        return float("nan"), 0
    sc = s[rr, cc]
    top = np.sort(sc)[::-1][:n_spots]
    strong = sc >= 0.2 * float(np.median(top))       # strong peaks only: noise maxima in spot tails
    rr, cc = rr[strong], cc[strong]                    # would break the isolation test
    order = np.argsort(s[rr, cc])[::-1][:4 * n_spots]
    rr, cc = rr[order], cc[order]
    pts = np.c_[rr, cc].astype(float)
    y, x = np.mgrid[-half:half + 1, -half:half + 1].astype(float)
    sig = []
    for i in range(len(rr)):
        d = np.hypot(*(pts - pts[i]).T)
        d[i] = np.inf
        if d.min() < isolation_px:
            continue
        r, c = rr[i], cc[i]
        if r < half or c < half or r + half >= img.shape[0] or c + half >= img.shape[1]:
            continue
        if not ok[r - half:r + half + 1, c - half:c + half + 1].all():
            continue
        box = img[r - half:r + half + 1, c - half:c + half + 1].astype(float)
        b0 = float(np.median(np.r_[box[0], box[-1], box[:, 0], box[:, -1]]))
        try:
            p, _ = curve_fit(_pix_gauss, (y.ravel(), x.ravel()), box.ravel(),
                             p0=(max(box.sum() - b0 * box.size, 1.0), 0.0, 0.0, 1.2, b0),
                             sigma=np.sqrt(np.maximum(box.ravel(), 1.0)),
                             bounds=([0, -2, -2, 0.3, -np.inf], [np.inf, 2, 2, half, np.inf]), maxfev=2000)
        except (RuntimeError, ValueError):
            continue
        sig.append(float(p[3]))
        if len(sig) >= n_spots:
            break
    return (float(np.median(sig)), len(sig)) if sig else (float("nan"), 0)


def detect_spots(img: np.ndarray, ok: np.ndarray, coarse: np.ndarray, threshold: float,
                 cfg: SnapshotDetectorConfig) -> np.ndarray:
    """Detect spots; returns a float array with columns :data:`SPOT_FIELDS`.

    ``row``/``col`` are background-subtracted intensity centroids in a
    ``(2*flux_box+1)`` box, ``flux`` is the net counts in that box, ``peak`` the
    raw pixel value at the maximum, ``score`` the detection score.
    """
    bkg = local_background(img, ok, coarse, cfg.local_box, cfg.cap_nsig)
    res, s = score_map(img, ok, bkg, cfg)
    safe = valid_region(ok, cfg.edge_px, cfg.margin_px)
    B = cfg.flux_box
    rr = np.arange(-B, B + 1)
    out = []
    H, Wd = img.shape
    for r, c in _peaks(s, safe, threshold):
        if r < B or c < B or r >= H - B or c >= Wd - B:
            continue
        box = res[r - B:r + B + 1, c - B:c + B + 1]
        pos = np.clip(box, 0, None)
        w = pos.sum()
        if w <= 0:
            continue
        out.append((r + (pos.sum(axis=1) * rr).sum() / w,
                    c + (pos.sum(axis=0) * rr).sum() / w,
                    float(box.sum()), float(img[r, c]), float(s[r, c])))
    return np.array(out, dtype=np.float64).reshape(-1, len(SPOT_FIELDS))
