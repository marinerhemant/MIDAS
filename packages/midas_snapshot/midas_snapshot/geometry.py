"""Pixel -> scattering angle maps and the matrix reference lines."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class Maps:
    tth: np.ndarray        # degrees, geometry frame
    eta: np.ndarray
    wavelength: float
    ring_index: np.ndarray  # integer bin per pixel (tth_step), for coarse backgrounds
    tth0: float
    tth_step: float
    pixel_deg: float        # median angular size of one pixel in 2theta


def build_maps(geometry_file: str, tth_step: float) -> Maps:
    from midas_defect.geometry import Geometry, detector_angle_maps
    g = Geometry.from_paramstest(geometry_file)
    tth, eta = detector_angle_maps(g)
    tth0 = float(np.nanmin(tth))
    ring = np.floor((np.nan_to_num(tth, nan=tth0) - tth0) / max(tth_step * 4, 1e-6)).astype(np.int64)
    gy, gx = np.gradient(tth)
    pix = float(np.nanmedian(np.hypot(gy, gx)))
    return Maps(tth.astype(np.float64), eta.astype(np.float64), float(g.wavelength_A), ring, tth0,
                tth_step, pix)


def fit_window(cfg_window, maps: "Maps") -> float:
    """Ring-centroid half-window: explicit, else six pixels' worth of 2theta (never below
    0.06 deg). A window of only a pixel or two puts the edge background on the ring
    flanks and biases the centroid."""
    return float(cfg_window) if cfg_window else max(0.06, 6.0 * maps.pixel_deg)


def matrix_lines(cif: Optional[str], lam: float, tth_max: float) -> np.ndarray:
    """Allowed d-lines of the matrix structure on the detector (basis absences dropped)."""
    if not cif:
        return np.array([])
    from midas_hkls.feature_phase import allowed_d_lines
    from midas_hkls.io.cif import read_cif
    d_min = lam / (2 * np.sin(np.radians(tth_max / 2)))
    return allowed_d_lines(read_cif(cif), d_min, 50.0)


def rel_to_lines(d: np.ndarray, lines: np.ndarray, scale: float) -> np.ndarray:
    """Signed relative distance of each d to the nearest scaled line."""
    d = np.atleast_1d(np.asarray(d, float))
    if len(lines) == 0 or not np.isfinite(scale):
        return np.full(d.shape, np.nan)
    L = np.sort(lines) * scale
    j = np.clip(np.searchsorted(L, d), 1, len(L) - 1)
    k = np.where(np.abs(d - L[j - 1]) < np.abs(d - L[j]), j - 1, j)
    return d / L[k] - 1.0
