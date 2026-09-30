"""Post-fit row construction: build the 29-column ``peakRow`` for each fitted peak.

Mirrors PeaksFittingOMPZarrRefactor.c lines 1628-1662. Field order is locked
by ``PeaksFittingConsolidatedIO.h`` (``PEAK_COL_NAMES``).
"""
from __future__ import annotations

from typing import Tuple

import numpy as np

from midas_peakfit.geometry import yz_from_r_eta

# 29-column count (asserted as N_PEAK_COLS in the C header).
N_PEAK_COLS = 29


def build_peak_rows(
    *,
    spot_id_start: int,
    omega: float,
    Ycen: float,
    Zcen: float,
    n_peaks_in_region: int,
    n_pixels_in_region: int,
    raw_sum_intensity: float,
    mask_touched: int,
    return_code: int,
    fit_rmse: float,
    bg: float,
    # Per-peak (length n_peaks_in_region):
    Imax: np.ndarray,
    R: np.ndarray,
    Eta: np.ndarray,
    sigmaGR: np.ndarray,
    sigmaLR: np.ndarray,
    sigmaGEta: np.ndarray,
    sigmaLEta: np.ndarray,
    Mu: np.ndarray,
    integrated_intensity: np.ndarray,
    n_pixels_per_peak: np.ndarray,  # int
    maxY: np.ndarray,  # int (regional maxima Y row)
    maxZ: np.ndarray,  # int (regional maxima Z col)
    raw_imax: np.ndarray,  # intensity at maxima before fit
) -> np.ndarray:
    """Return ``rows`` shape ``(n_peaks_in_region, 29)`` float64.

    Field equations (matching C source):
        SigmaR    = max(SigmaGR, SigmaLR)
        SigmaEta  = max(SigmaGEta, SigmaLEta)
        YCen_out  = -yCenArray + Ycen   where yCenArray = -R*sin(Eta)
        ZCen_out  =  zCenArray + Zcen   where zCenArray =  R*cos(Eta)
        diffY     = maxY + yCenArray - Ycen
        diffZ     = maxZ - zCenArray - Zcen
    """
    n = n_peaks_in_region
    rows = np.zeros((n, N_PEAK_COLS), dtype=np.float64)

    # Convert (R, Eta) → fitted (yCenArray, zCenArray)
    yCenArray, zCenArray = yz_from_r_eta(R, Eta)

    SigmaR = np.maximum(sigmaGR, sigmaLR)
    SigmaEta = np.maximum(sigmaGEta, sigmaLEta)

    spot_ids = np.arange(spot_id_start, spot_id_start + n, dtype=np.float64)

    rows[:, 0] = spot_ids                           # SpotID
    rows[:, 1] = integrated_intensity               # IntegratedIntensity
    rows[:, 2] = omega                              # Omega
    rows[:, 3] = -yCenArray + Ycen                  # YCen
    rows[:, 4] = zCenArray + Zcen                   # ZCen
    rows[:, 5] = Imax                               # IMax
    rows[:, 6] = R                                  # Radius
    rows[:, 7] = Eta                                # Eta
    rows[:, 8] = SigmaR                             # SigmaR  = max(σGR, σLR)
    rows[:, 9] = SigmaEta                           # SigmaEta = max(σGE, σLE)
    rows[:, 10] = n_pixels_per_peak.astype(np.float64)  # NrPixels (per peak)
    rows[:, 11] = float(n_pixels_in_region)         # NrPxTot
    rows[:, 12] = float(n_peaks_in_region)          # nPeaks
    rows[:, 13] = maxY.astype(np.float64)           # maxY
    rows[:, 14] = maxZ.astype(np.float64)           # maxZ
    rows[:, 15] = maxY.astype(np.float64) + yCenArray - Ycen  # diffY
    rows[:, 16] = maxZ.astype(np.float64) - zCenArray - Zcen  # diffZ
    rows[:, 17] = raw_imax                          # rawIMax
    rows[:, 18] = float(return_code)                # returnCode
    rows[:, 19] = fit_rmse                          # retVal
    rows[:, 20] = bg                                # BG
    rows[:, 21] = sigmaGR                           # SigmaGR
    rows[:, 22] = sigmaLR                           # SigmaLR
    rows[:, 23] = sigmaGEta                         # SigmaGEta
    rows[:, 24] = sigmaLEta                         # SigmaLEta
    rows[:, 25] = Mu                                # MU
    rows[:, 26] = raw_sum_intensity                 # RawSumIntensity (region-wide)
    rows[:, 27] = float(mask_touched)               # maskTouched
    rows[:, 28] = fit_rmse                          # FitRMSE (= retVal)

    return rows


def build_saturated_rows(
    sat_regions,
    *,
    omega: float,
    Ycen: float,
    Zcen: float,
    spot_id_start: int,
) -> np.ndarray:
    """29-col rows for saturated (unfitted) regions, one row per region.

    Same column layout as :func:`build_peak_rows` so every reader works, with:
    ``IntegratedIntensity`` = ``RawSumIntensity`` = summed clipped pixels (a
    LOWER BOUND), ``IMax`` = brightest pixel, position = intensity-weighted
    centroid, ``SigmaR``/``SigmaEta`` = RMS extents, ``returnCode`` =
    ``SATURATED_RETURN_CODE`` (-2), fit-only fields (BG, Mu, split sigmas,
    FitRMSE) = 0.
    """
    from midas_peakfit.seeds import SATURATED_RETURN_CODE

    rows = np.zeros((len(sat_regions), N_PEAK_COLS), dtype=np.float64)
    for i, s in enumerate(sat_regions):
        yArr, zArr = yz_from_r_eta(np.array([s.R]), np.array([s.Eta]))
        y_out = float(-yArr[0] + Ycen)
        z_out = float(zArr[0] + Zcen)
        r = rows[i]
        r[0] = spot_id_start + i
        r[1] = s.raw_sum
        r[2] = omega
        r[3] = y_out
        r[4] = z_out
        r[5] = s.max_value
        r[6] = s.R
        r[7] = s.Eta
        r[8] = s.sigma_R
        r[9] = s.sigma_Eta
        r[10] = float(s.n_pixels)
        r[11] = float(s.n_pixels)
        r[12] = 1.0
        r[13] = float(s.maxY)
        r[14] = float(s.maxZ)
        r[15] = float(s.maxY) - y_out
        r[16] = float(s.maxZ) - z_out
        r[17] = s.max_value
        r[18] = float(SATURATED_RETURN_CODE)
        r[26] = s.raw_sum
        r[27] = float(s.mask_touched)
    return rows


__all__ = ["N_PEAK_COLS", "build_peak_rows", "build_saturated_rows"]
