"""Independent ring-position gate, in PIXELS.

The gates in :mod:`midas_calibrate_v2.pipelines.diagnostics` grade the fit with the fit's own
residual, so a geometry that converged onto a wrong but self-consistent solution can still
pass them (``converged=True`` is not a quality check). This gate does not use the fit. It
takes the image, a 2theta map for the geometry under test, and the calibrant's ideal ring
2theta list, then measures where each ring really is, per ring and per azimuth sector, with
its own peak fit, and reports the radial offset from the ideal ring in pixels.

Procedure, per ring ``k`` (ideal 2theta ``t_k``):

1. take the pixels with ``|2theta_map - t_k| < window`` (window = at most ``max_window_px``
   pixels, and at most 0.45 of the 2theta gap to the neighbouring ring), unmasked only;
2. bin them by 2theta into bins of ``bin_px`` pixels and by azimuth about the beam pixel
   (the minimum of the map) into ``sector_deg`` sectors, and average the intensity per cell;
3. fit Gaussian + linear background to each sector's profile; a sector is valid only if its
   window is covered on both flanks with no hole longer than 2 bins (isolated empty bins are
   skipped) and the fitted centre is within ``max_centre_px`` of the ideal ring;
4. convert ``(centre - t_k)`` to pixels with the map's OWN local radial gradient (degrees of 2theta
   per pixel, averaged over the cell's pixels), so the unit is the true radial pixel on a tilted
   plane and no distance has to be supplied (the earlier ``Lsd / px / cos^2`` form is exact only
   at the beam centre and was off by up to +-30 % with azimuth at 45 deg tilt; found by /verify).

One global scale ``s`` (``tan 2theta_obs = s tan 2theta_ideal``, a least-squares fit over all
valid cells) is removed first, because wavelength and distance are degenerate and a ring
position test cannot see them; ``s`` is reported. The gate is therefore BLIND to ``Lsd`` and to
the wavelength: calibrating the 45 deg frame with a wavelength 1.9 % too high gave PASS, 0.060 px,
scale 1.00002, with the distance 1.8 % short (hard rule 9 of the calibrate-integrate doc set).

Because a wrong geometry loses cells (the rings fall outside the window), the number of
valid cells is part of the verdict: a cell is POPULATED when every bin of its window holds
pixels (so coverage gaps and off-detector sectors do not count against the geometry), and
fewer than ``min_fraction`` of the populated cells finding a ring FAILs whatever the
residual of the survivors says.

What a PASS means: the rings sit where the map says, up to ONE global scale, in every azimuth sector
that could be checked. It does not validate the wavelength or the distance (a free Lsd absorbs
0.4 % of either, 1-2 px at the rim), and it sees a centre/tilt error only above about 0.4 px / 0.07-0.15
deg (resolution measured on noisy synthetic frames). Cells whose fitted centre is more than
``max_centre_px`` from the ideal ring are dropped, so a localized error is detected through the lost
cells: a verdict is WARN when more than ``warn_fraction`` of the populated cells are lost or when
>= 10 % of the azimuth sectors lose more than half of theirs.

Defaults: PASS if median |residual| < 0.3 px and p95 < 1 px (the sub-pixel standard used
for the real tilted-detector frames; MIDAS values on those: 0.04-0.2 px median),
WARN if median < 0.5 px, otherwise FAIL. These are a screen for "sub-pixel", not a precision
claim: the peak-fit centroiding floor is about 0.2 px per cell on real frames.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
from scipy.optimize import least_squares

__all__ = ["RingPositionGate", "ring_position_gate"]


@dataclass
class RingPositionGate:
    """Result of :func:`ring_position_gate`. All residuals are radial, in pixels."""

    verdict: str                      # "PASS" | "WARN" | "FAIL"
    median_px: float
    p90_px: float
    p95_px: float
    max_px: float
    n_fits: int                       # valid (ring, sector) cells
    n_cells_expected: int             # populated (ring, sector) windows
    n_rings_used: int
    scale: float                      # removed global scale (1 = none needed)
    per_ring_median_px: Dict[float, float] = field(default_factory=dict)
    reasons: List[str] = field(default_factory=list)
    bad_sectors: List[int] = field(default_factory=list)   # azimuth sectors that lost more than half their cells

    def __str__(self) -> str:
        why = ("; " + "; ".join(self.reasons)) if self.reasons else ""
        return (f"ring-position gate {self.verdict}: median {self.median_px:.3f} px, "
                f"p95 {self.p95_px:.3f} px, max {self.max_px:.2f} px, "
                f"{self.n_fits}/{self.n_cells_expected} cells on {self.n_rings_used} rings, "
                f"scale {self.scale:.5f}{why}")


def _window_populated(pop: np.ndarray, edge: int, min_fraction: float = 0.85,
                      max_gap: int = 2) -> bool:
    """A window is usable when it is covered on BOTH flanks and has no long hole.

    Sparse single empty bins (inner rings, where a 0.25 px bin of a 10 deg sector holds only a few
    pixels) are tolerated and skipped in the fit. A window cut by the detector edge or a panel
    gap is not: it is missing a flank or a run of bins, and a one-sided fit is biased (tens to
    hundreds of microstrain in every integrator tested), so such cells are dropped.
    """
    if pop.mean() < min_fraction or not pop[:edge].any() or not pop[-edge:].any():
        return False
    run = best = 0
    for v in pop:
        run = 0 if v else run + 1
        best = max(best, run)
    return best <= max_gap


def _fit_peak(x: np.ndarray, y: np.ndarray, sigma0: float):
    """Gaussian + linear background; returns (centre, amplitude) or None."""
    if len(x) < 15 or not np.all(np.isfinite(y)) or np.any(y <= 0):
        return None
    bg0 = float(np.percentile(y, 10))
    amp0 = float(y.max() - bg0)
    if amp0 <= 0:
        return None
    c0 = float(x[np.argmax(y)])

    def res(p):
        return p[0] + p[1] * (x - c0) + p[2] * np.exp(-0.5 * ((x - p[3]) / p[4]) ** 2) - y

    try:
        r = least_squares(res, [bg0, 0.0, amp0, c0, sigma0],
                          x_scale=[bg0 + 1e-9, 1.0, amp0, sigma0 * 0.25, sigma0 * 0.25])
    except Exception:
        return None
    if not r.success or r.x[2] <= 0 or abs(r.x[4]) < 0.15 * sigma0 or abs(r.x[4]) > 8 * sigma0:
        return None
    return float(r.x[3]), float(r.x[2])


def ring_position_gate(
    image: np.ndarray,
    two_theta_map_deg: np.ndarray,
    ring_two_theta_deg: Sequence[float],
    *,
    lsd_um: Optional[float] = None,
    px_um: Optional[float] = None,
    mask: Optional[np.ndarray] = None,
    sector_deg: float = 10.0,
    bin_px: float = 0.25,
    max_window_px: float = 8.0,
    min_window_px: float = 3.0,
    max_centre_px: float = 3.0,
    min_fraction: float = 0.5,
    warn_fraction: float = 0.8,
    scale_warn: float = 1e-3,
    min_rings: int = 3,
    pass_median_px: float = 0.3,
    pass_p95_px: float = 1.0,
    warn_median_px: float = 0.5,
) -> RingPositionGate:
    """Measure ring positions in ``image`` against the 2theta map and report pixel offsets.

    Parameters
    ----------
    image : (NZ, NY) array, MIDAS convention ``image[row=Z, col=Y]``.
    two_theta_map_deg : same shape, 2theta (degrees) of every pixel for the geometry UNDER
        TEST (include distortion; e.g. from ``pixel_to_REta_from_spec``).
    ring_two_theta_deg : ideal 2theta of the calibrant rings (all that could be on the
        detector; rings outside the map's range are skipped).
    lsd_um, px_um : accepted for compatibility and ignored: the 2theta -> pixel scale now comes from
        the map's own gradient.
    mask : bool array, True = bad pixel (excluded).
    """
    img = np.asarray(image, dtype=np.float64)
    tth = np.asarray(two_theta_map_deg, dtype=np.float64)
    if img.shape != tth.shape:
        raise ValueError(f"image {img.shape} and two_theta_map {tth.shape} differ")
    ok = np.isfinite(img) & np.isfinite(tth)
    if mask is not None:
        ok &= ~np.asarray(mask, dtype=bool)
    r0, c0 = np.unravel_index(int(np.nanargmin(np.where(np.isfinite(tth), tth, np.inf))), tth.shape)
    rows, cols = np.indices(tth.shape)
    az = (np.degrees(np.arctan2(rows - r0, cols - c0)) + 360.0) % 360.0
    n_sec = int(round(360.0 / sector_deg))
    sec = np.minimum((az // sector_deg).astype(np.int64), n_sec - 1)
    rings = np.sort(np.asarray([t for t in ring_two_theta_deg if tth[ok].min() < t < tth[ok].max()],
                               dtype=np.float64))
    if len(rings) > 1:                                  # coincident hkl give the same ring twice: a 0 gap
        rings = rings[np.concatenate(([True], np.diff(rings) > 1e-4))]   # would drop both as 'crowded'
    gy, gx = np.gradient(tth)
    gmag = np.hypot(gx, gy)                             # degrees of 2theta per pixel, radial (rings are level sets)
    cen_obs: List[float] = []
    cen_ref: List[float] = []
    ring_of: List[int] = []
    grad_of: List[float] = []
    sec_of: List[int] = []
    pop_sec = np.zeros(n_sec)                           # populated windows per sector
    fnd_sec = np.zeros(n_sec)                           # of which found a ring
    n_expected = 0
    n_rings_used = 0
    for ik, t0 in enumerate(rings):
        sel0 = ok & (np.abs(tth - t0) < 0.02 * max(t0, 1.0))
        if sel0.sum() < 200:
            continue
        pix_deg = float(np.median(gmag[sel0]))             # 2theta per radial pixel at this ring
        gaps = []
        if ik > 0:
            gaps.append(t0 - rings[ik - 1])
        if ik < len(rings) - 1:
            gaps.append(rings[ik + 1] - t0)
        win = min(max_window_px * pix_deg, 0.45 * min(gaps)) if gaps else max_window_px * pix_deg
        if win < min_window_px * pix_deg:
            continue                                         # crowded ring: not resolvable
        w = bin_px * pix_deg
        sel = ok & (np.abs(tth - t0) < win)
        if sel.sum() < 200:
            continue
        nb = int(math.ceil(2 * win / w))
        b = np.minimum(((tth[sel] - (t0 - win)) / w).astype(np.int64), nb - 1)
        flat = sec[sel] * nb + b
        S = np.bincount(flat, weights=img[sel], minlength=n_sec * nb).reshape(n_sec, nb)
        C = np.bincount(flat, minlength=n_sec * nb).reshape(n_sec, nb)
        Gs = np.bincount(sec[sel], weights=gmag[sel], minlength=n_sec) / np.maximum(np.bincount(sec[sel], minlength=n_sec), 1)
        centres = t0 - win + (np.arange(nb) + 0.5) * w
        n_here = 0
        edge = max(3, nb // 8)
        for s_ in range(n_sec):
            pop = C[s_] > 0
            if not _window_populated(pop, edge):
                continue                                  # sector not fully on the detector: not a cell
            n_expected += 1
            pop_sec[s_] += 1
            fit = _fit_peak(centres[pop], S[s_][pop] / C[s_][pop], sigma0=1.0 * pix_deg)
            if fit is None or abs(fit[0] - t0) > max_centre_px * pix_deg:
                continue
            fnd_sec[s_] += 1
            cen_obs.append(fit[0])
            cen_ref.append(t0)
            ring_of.append(ik)
            grad_of.append(float(Gs[s_]))
            sec_of.append(s_)
            n_here += 1
        if n_here:
            n_rings_used += 1
    reasons: List[str] = []
    if not cen_obs:
        return RingPositionGate("FAIL", float("nan"), float("nan"), float("nan"), float("nan"), 0,
                                max(n_expected, 1), 0, float("nan"), {}, ["no ring found in any sector"])
    co = np.radians(np.asarray(cen_obs))
    cr = np.radians(np.asarray(cen_ref))
    tn_o, tn_r = np.tan(co), np.tan(cr)
    scale = float((tn_o * tn_r).sum() / (tn_r * tn_r).sum())
    d_tth_deg = np.degrees(np.arctan(tn_o / scale) - cr)      # degrees, after the global scale
    dpx = d_tth_deg / np.asarray(grad_of)                     # / (degrees per radial pixel of THIS cell)
    a = np.abs(dpx)
    per_ring = {round(float(rings[k]), 4): float(np.median(a[np.asarray(ring_of) == k]))
                for k in sorted(set(ring_of))}
    med, p90, p95, mx = (float(np.median(a)), float(np.percentile(a, 90)),
                         float(np.percentile(a, 95)), float(a.max()))
    frac = len(a) / max(n_expected, 1)
    pop_ok = pop_sec >= 3
    bad = [int(i) for i in np.where(pop_ok & (fnd_sec < 0.5 * np.maximum(pop_sec, 1)))[0]]
    if frac < min_fraction:
        reasons.append(f"only {frac:.0%} of the populated cells found a ring "
                       f"(< {min_fraction:.0%}): the geometry puts the rings elsewhere; "
                       "the survivors' residual understates the error")
    if n_rings_used < min_rings:
        reasons.append(f"only {n_rings_used} rings usable (< {min_rings})")
    if reasons or not (med < pass_median_px and p95 < pass_p95_px):
        verdict = "FAIL" if (reasons or med >= warn_median_px) else "WARN"
        if not reasons:
            reasons.append(f"median {med:.2f} px / p95 {p95:.2f} px outside the pass limits "
                           f"({pass_median_px} / {pass_p95_px})")
    else:
        verdict = "PASS"
    if verdict != "FAIL" and (frac < warn_fraction or len(bad) >= max(2, int(0.1 * pop_ok.sum()))):
        verdict = "WARN"
        reasons.append(f"{1 - frac:.0%} of the populated cells lost their ring"
                       + (f"; {len(bad)} azimuth sectors lost more than half of theirs ({bad[0] * sector_deg:g}-{(bad[-1] + 1) * sector_deg:g} deg): "
                          "a LOCALIZED error (one panel, one arc) is not caught by the median" if bad else ""))
    if abs(scale - 1.0) > scale_warn:
        reasons.append(f"global scale {scale:.4f} differs from 1 by more than {scale_warn:.1%}: the measured "
                       "rings and the ideal ring list disagree by a common factor (wrong lattice "
                       "constant, wavelength used to build the list, or a map in the wrong units). "
                       "NOTE the converse does not hold: a fit with a free Lsd absorbs a wrong "
                       "wavelength and still reads 1.000")
        if verdict == "PASS":
            verdict = "WARN"
    return RingPositionGate(verdict, med, p90, p95, mx, int(len(a)), int(n_expected), n_rings_used,
                            scale, per_ring, reasons, bad)
