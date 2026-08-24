"""Regional-maxima detection + moment-based initial parameter seeding.

Mirrors ``findRegionalMaxima`` and the seeding block of ``fit2DPeaks`` in
``PeaksFittingOMPZarrRefactor.c``. Returns ready-to-fit parameter arrays
``x, xl, xu`` (8 params per peak + 1 background) along with per-pixel
(R, Eta) and ancillary fields.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from midas_peakfit.connected import Region
from midas_peakfit.geometry import calc_eta_angle, calc_eta_angle_np, RAD2DEG
from midas_peakfit.panels import Panel, get_panel_index, apply_panel_correction
from midas_peakfit.uncertainty import classify_peak_quality


# 8-neighbor offsets matching C dx/dy
_NEIGH_OFFSETS = np.array(
    [
        (1, 0), (-1, 0), (0, 1), (0, -1),
        (1, 1), (1, -1), (-1, 1), (-1, -1),
    ],
    dtype=np.int32,
)


@dataclass
class SeededRegion:
    """A region with regional-maxima detected and initial fit parameters set up.

    Field naming mirrors the C variables for traceability.
    """

    region_id: int
    n_peaks: int
    n_pixels: int
    raw_sum: float
    threshold: float
    mask_touched: int  # 0 or 1

    # Maxima seeds (peak pixel positions and intensities)
    maxY: np.ndarray  # int32, shape (n_peaks,)
    maxZ: np.ndarray  # int32, shape (n_peaks,)
    maxima_values: np.ndarray  # float64, shape (n_peaks,)

    # Per-pixel (in this region) z, R, Eta (in *fitted* coords; Eta in degrees)
    pixels_y: np.ndarray  # int32, shape (n_pixels,)  — row in imgCorrBC = Y
    pixels_z: np.ndarray  # int32, shape (n_pixels,)  — col in imgCorrBC = Z
    z_values: np.ndarray  # float64
    Rs: np.ndarray  # float64
    Etas: np.ndarray  # float64

    # Initial guess + lower/upper bounds: shape (1 + 8*n_peaks,)
    x0: np.ndarray
    xl: np.ndarray
    xu: np.ndarray

    # Per-peak initial center (peakR, peakEta) and (maxY, maxZ) for outputs
    peak_R: np.ndarray  # float64, shape (n_peaks,) — initial seed R
    peak_Eta: np.ndarray  # float64, shape (n_peaks,) — initial seed Eta

    # Per-peak background-subtracted integrated intensity (M_0 from
    # Modregger 2025) and the photon-count regime quality flag. Always
    # populated; cost is one sum over Voronoi pixels which is computed
    # anyway during sigma seeding.
    peak_M0: np.ndarray  # float64, shape (n_peaks,) — integrated counts
    peak_quality: np.ndarray  # int8,  shape (n_peaks,) — 0/1/2 (see uncertainty.py)

    # Optional higher moments along R and η. Populated only when
    # ``seed_region(..., compute_moments=True)``; ``None`` in the default
    # fast path. Used to derive Modregger shot-noise σ via
    # ``uncertainty.compute_moment_sigma``.
    peak_M2_R: Optional[np.ndarray] = None  # float64, shape (n_peaks,)
    peak_M2_Eta: Optional[np.ndarray] = None
    peak_M4_R: Optional[np.ndarray] = None
    peak_M4_Eta: Optional[np.ndarray] = None


def _per_pixel_r_eta(
    region: Region,
    Ycen: float,
    Zcen: float,
    panels: List[Panel],
) -> tuple[np.ndarray, np.ndarray]:
    """Compute (R, Eta) per pixel, applying optional panel correction.

    Mirrors C lines 870-885 of ``fit2DPeaks``:
        Rs[i]   = sqrt((r + pdY - yCen)^2 + (c + pdZ - zCen)^2)
        Etas[i] = calcEtaAngle(-(r + pdY) + yCen, c + pdZ - zCen)
    """
    r_idx = region.pixel_rows.astype(np.float64)
    c_idx = region.pixel_cols.astype(np.float64)

    if panels:
        # Per-pixel panel correction (loop kept simple — region is small)
        pdY = np.zeros_like(r_idx)
        pdZ = np.zeros_like(c_idx)
        for i, (r, c) in enumerate(zip(r_idx, c_idx)):
            pidx = get_panel_index(r, c, panels)
            if pidx >= 0:
                yc, zc = apply_panel_correction(r, c, panels[pidx])
                pdY[i] = yc - r
                pdZ[i] = zc - c
        rr = r_idx + pdY
        cc = c_idx + pdZ
    else:
        rr = r_idx
        cc = c_idx

    dY = rr - Ycen
    dZ = cc - Zcen
    Rs = np.sqrt(dY * dY + dZ * dZ)
    Etas = calc_eta_angle_np(-dY, dZ)
    return Rs, Etas


def find_regional_maxima(
    region: Region,
    img_corr: np.ndarray,
    mask: np.ndarray,
    int_sat: float,
    max_n_peaks: int,
) -> Optional[tuple[np.ndarray, np.ndarray, np.ndarray, int]]:
    """Detect 8-neighbor regional maxima within a region.

    Returns ``(maxY, maxZ, maxima_values, mask_touched)`` or ``None`` if the
    region is saturated (``z[i] > int_sat`` for any pixel).

    Replicates ``findRegionalMaxima`` in PeaksFittingOMPZarrRefactor.c, including
    fallback to the middle pixel when no local maxima are found, and the
    cap at ``max_n_peaks`` (greedy by intensity).

    **Saturation discards the WHOLE region, and nothing records it.** One pixel
    over ``IntSat`` (from ``UpperBoundThreshold``) drops every peak in the
    region, with no flag column and, until 2026-08-22, no count either. That
    cuts two ways and both are invisible downstream:

    * a saturated reflection is a *strong* one, so its absence is scored as
      incompleteness -- the grain is penalised for having produced too much
      signal;
    * it is also the brightest contributor to that ring's ``powder_int``
      normalisation (``midas_transforms/radius/core.py:153``), so removing it
      biases every grain volume on the ring upward.

    Callers now count these (``sr is None`` is the saturation case and the only
    one) and the per-frame total is reported. Emitting the peaks *with* a
    saturated flag instead of dropping them would be the fuller fix, but it
    changes spot counts on every existing dataset and needs its own decision.
    """
    z = region.intensities
    if (z > int_sat).any():
        return None  # saturated → 0 peaks

    Y = region.pixel_rows.astype(np.int64)
    Z = region.pixel_cols.astype(np.int64)
    n = z.size
    nrPixels = img_corr.shape[0]

    # Mask-touch flag: mask is (NrPixels, NrPixels) in (Z, Y) layout.
    # See PeaksFittingOMPZarrRefactor.c:659 — mask access is mask[col*N + row]
    # i.e. mask[Z, Y]. We mirror that here: index (Z[i], Y[i]).
    mask_touched = 0
    if mask is not None and mask.size > 0:
        in_bounds = (Z >= 0) & (Z < nrPixels) & (Y >= 0) & (Y < nrPixels)
        if in_bounds.any():
            mvals = mask[Z[in_bounds], Y[in_bounds]]
            if (mvals == 1).any():
                mask_touched = 1

    # 8-neighbor regional-max test using img_corr direct lookup.
    # A pixel is a local max iff for every in-bounds neighbor with
    # img_corr[neigh] > 0, img_corr[neigh] <= z_self (strict ">" in C).
    is_max = np.ones(n, dtype=bool)
    for dy, dz in _NEIGH_OFFSETS:
        nY = Y + dy
        nZ = Z + dz
        valid = (nY >= 0) & (nY < nrPixels) & (nZ >= 0) & (nZ < nrPixels)
        if not valid.any():
            continue
        # neighbor values; for invalid coords use 0 so they don't count
        n_vals = np.zeros(n, dtype=np.float64)
        n_vals[valid] = img_corr[nY[valid], nZ[valid]]
        # neighbor "in region" iff img_corr > 0
        higher = (n_vals > 0) & (n_vals > z)
        is_max &= ~higher

    if is_max.any():
        max_idx = np.where(is_max)[0]
    else:
        # Fallback: middle pixel as single seed
        max_idx = np.array([n // 2])

    n_peaks = max_idx.size
    if n_peaks > max_n_peaks:
        # Greedy: pick brightest, zero it (effectively, sort and take top)
        order = max_idx[np.argsort(-z[max_idx])][:max_n_peaks]
        max_idx = order
        n_peaks = max_n_peaks

    maxY = Y[max_idx].astype(np.int32)
    maxZ = Z[max_idx].astype(np.int32)
    maxima_values = z[max_idx].astype(np.float64)
    return maxY, maxZ, maxima_values, mask_touched


def seed_region(
    region: Region,
    img_corr: np.ndarray,
    mask: np.ndarray,
    *,
    Ycen: float,
    Zcen: float,
    int_sat: float,
    max_n_peaks: int,
    panels: List[Panel],
    compute_moments: bool = False,
) -> Optional[SeededRegion]:
    """Build a fully-seeded ``SeededRegion`` ready for batched LM fitting.

    Returns ``None`` if the region should be skipped. Saturation is the **only**
    reason this returns ``None``, which is what lets callers count saturated
    regions without re-testing — see :func:`find_regional_maxima` for why that
    loss is worth counting.

    ``compute_moments``: if True, additionally populate ``peak_M2_R``,
    ``peak_M2_Eta``, ``peak_M4_R``, ``peak_M4_Eta`` on the returned region
    so that downstream code can derive Modregger 2025 shot-noise σ. The
    M_0 and quality fields are populated unconditionally.
    """
    maxima = find_regional_maxima(region, img_corr, mask, int_sat, max_n_peaks)
    if maxima is None:
        return None
    maxY, maxZ, maxima_values, mask_touched = maxima
    n_peaks = int(maxY.size)
    n_pixels = region.n_pixels

    # Per-pixel (R, Eta) — used for fitting and moment estimation
    Rs, Etas = _per_pixel_r_eta(region, Ycen, Zcen, panels)

    R_min, R_max = float(Rs.min()), float(Rs.max())
    Eta_min, Eta_max = float(Etas.min()), float(Etas.max())

    maxRWidth = (R_max - R_min) / 2.0 + 1.0
    maxEtaWidth = (Eta_max - Eta_min) / 2.0 + math.degrees(
        math.atan(2.0 / (R_max + R_min)) if (R_max + R_min) != 0 else 0.0
    )
    if (Eta_max - Eta_min) > 180.0:
        maxEtaWidth -= 180.0
    if maxRWidth < 0.1:
        maxRWidth = 0.1
    if maxEtaWidth < 0.1:
        maxEtaWidth = 0.1

    # Fallback uniform width estimate
    width = math.sqrt(n_pixels / n_peaks)
    if width > maxRWidth:
        width = maxRWidth

    # Initial peak (R, Eta) from maxima — note: NO panel correction here.
    dY_max = maxY.astype(np.float64) - Ycen
    dZ_max = maxZ.astype(np.float64) - Zcen
    peak_R = np.sqrt(dY_max * dY_max + dZ_max * dZ_max)
    peak_Eta = calc_eta_angle_np(-dY_max, dZ_max)

    # Moment-based sigma per peak (Voronoi partition by maxima distance)
    bg_est = region.threshold / 2.0
    val = region.intensities - bg_est
    pos_mask = val > 0

    estimSigmaR = np.full(n_peaks, width, dtype=np.float64)
    estimSigmaEta = np.full(n_peaks, width, dtype=np.float64)

    # Per-peak Voronoi-partitioned moments. ``sumW`` is M_0 (Modregger
    # 2025); M_2 along R/η is (sumWR2/sumW), (sumWEta2/sumW) and is reused
    # both as the initial Pseudo-Voigt sigma and as the input to the
    # closed-form shot-noise σ. M_4 is computed only when requested.
    sumW = np.zeros(n_peaks, dtype=np.float64)
    sumWR2 = np.zeros(n_peaks, dtype=np.float64)
    sumWEta2 = np.zeros(n_peaks, dtype=np.float64)
    sumWR4 = np.zeros(n_peaks, dtype=np.float64) if compute_moments else None
    sumWEta4 = np.zeros(n_peaks, dtype=np.float64) if compute_moments else None

    if pos_mask.any():
        Rs_p = Rs[pos_mask]
        Etas_p = Etas[pos_mask]
        val_p = val[pos_mask]

        # Distance to each peak: shape (P, n_peaks)
        dR = Rs_p[:, None] - peak_R[None, :]
        dE = Etas_p[:, None] - peak_Eta[None, :]
        d2 = dR * dR + dE * dE
        closest = np.argmin(d2, axis=1)  # (P,)

        dR_c = Rs_p - peak_R[closest]
        dE_c = Etas_p - peak_Eta[closest]
        np.add.at(sumW, closest, val_p)
        np.add.at(sumWR2, closest, val_p * dR_c * dR_c)
        np.add.at(sumWEta2, closest, val_p * dE_c * dE_c)
        if compute_moments:
            np.add.at(sumWR4, closest, val_p * dR_c ** 4)
            np.add.at(sumWEta4, closest, val_p * dE_c ** 4)

        ok = sumW > 0
        if ok.any():
            sR = np.sqrt(sumWR2[ok] / sumW[ok])
            sE = np.sqrt(sumWEta2[ok] / sumW[ok])
            sR = np.clip(sR, 0.1, maxRWidth)
            sE = np.clip(sE, 0.1, None)
            estimSigmaR[ok] = sR
            estimSigmaEta[ok] = sE

    # M_0 and quality flag (Modregger 2025 + paper Appendix A caveats).
    peak_M0 = sumW.copy()
    peak_quality = classify_peak_quality(peak_M0)

    # Higher moments — normalize to M_n = Σw·δ^n / Σw, with NaN where M_0 ≤ 0.
    if compute_moments:
        safe_M0 = np.where(sumW > 0, sumW, 1.0)
        peak_M2_R = np.where(sumW > 0, sumWR2 / safe_M0, np.nan)
        peak_M2_Eta = np.where(sumW > 0, sumWEta2 / safe_M0, np.nan)
        peak_M4_R = np.where(sumW > 0, sumWR4 / safe_M0, np.nan)
        peak_M4_Eta = np.where(sumW > 0, sumWEta4 / safe_M0, np.nan)
    else:
        peak_M2_R = peak_M2_Eta = peak_M4_R = peak_M4_Eta = None

    # Build x0, xl, xu with shape (1 + 8*n_peaks,)
    n = 1 + 8 * n_peaks
    x0 = np.zeros(n, dtype=np.float64)
    xl = np.zeros(n, dtype=np.float64)
    xu = np.zeros(n, dtype=np.float64)

    # Background
    x0[0] = region.threshold / 2.0
    xl[0] = 0.0
    xu[0] = region.threshold

    for i in range(n_peaks):
        dEta = RAD2DEG * math.atan(1.0 / max(peak_R[i], 1e-9))
        # Initial values
        x0[8 * i + 1] = maxima_values[i]      # Imax
        x0[8 * i + 2] = peak_R[i]             # R
        x0[8 * i + 3] = peak_Eta[i]           # Eta
        x0[8 * i + 4] = 0.5                   # Mu
        x0[8 * i + 5] = estimSigmaR[i]        # SigmaGR
        x0[8 * i + 6] = estimSigmaR[i]        # SigmaLR
        x0[8 * i + 7] = estimSigmaEta[i]      # SigmaGEta
        x0[8 * i + 8] = estimSigmaEta[i]      # SigmaLEta

        # Lower bounds
        xl[8 * i + 1] = maxima_values[i] / 2.0
        xl[8 * i + 2] = peak_R[i] - 1.0
        xl[8 * i + 3] = peak_Eta[i] - dEta
        xl[8 * i + 4] = 0.0
        xl[8 * i + 5] = 0.01
        xl[8 * i + 6] = 0.01
        xl[8 * i + 7] = 0.005
        xl[8 * i + 8] = 0.005

        # Upper bounds
        xu[8 * i + 1] = maxima_values[i] * 5.0
        xu[8 * i + 2] = peak_R[i] + 1.0
        xu[8 * i + 3] = peak_Eta[i] + dEta
        xu[8 * i + 4] = 1.0
        xu[8 * i + 5] = 2.0 * maxRWidth
        xu[8 * i + 6] = 2.0 * maxRWidth
        xu[8 * i + 7] = 2.0 * maxEtaWidth
        xu[8 * i + 8] = 2.0 * maxEtaWidth

    # Clamp x0 inside [xl, xu] (matches C lines 1037-1042)
    np.clip(x0, xl, xu, out=x0)

    return SeededRegion(
        region_id=region.id,
        n_peaks=n_peaks,
        n_pixels=n_pixels,
        raw_sum=region.raw_sum,
        threshold=region.threshold,
        mask_touched=mask_touched,
        maxY=maxY,
        maxZ=maxZ,
        maxima_values=maxima_values,
        pixels_y=region.pixel_rows.astype(np.int32),
        pixels_z=region.pixel_cols.astype(np.int32),
        z_values=region.intensities.astype(np.float64),
        Rs=Rs.astype(np.float64),
        Etas=Etas.astype(np.float64),
        x0=x0,
        xl=xl,
        xu=xu,
        peak_R=peak_R.astype(np.float64),
        peak_Eta=peak_Eta.astype(np.float64),
        peak_M0=peak_M0,
        peak_quality=peak_quality,
        peak_M2_R=peak_M2_R,
        peak_M2_Eta=peak_M2_Eta,
        peak_M4_R=peak_M4_R,
        peak_M4_Eta=peak_M4_Eta,
    )


__all__ = ["SeededRegion", "find_regional_maxima", "seed_region"]
