"""Detector geometry: tilt rotation + radial distortion → goodCoords map.

Replicates the per-pixel computation in ``PeaksFittingOMPZarrRefactor.c``
lines 2680-2750. Vectorized via NumPy; the goodCoords map is computed once
at startup and reused across all frames.
"""
from __future__ import annotations

import math
from typing import List, Optional, Tuple

import numpy as np

from midas_peakfit.panels import Panel
from midas_peakfit.params import ZarrParams

DEG2RAD = math.pi / 180.0
RAD2DEG = 180.0 / math.pi


# ─── Eta angle (matches calcEtaAngle in C) ──────────────────────────────────
def calc_eta_angle_np(y: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Vectorized port of ``calcEtaAngle``: ``alpha = RAD2DEG * acos(z/r)``,
    then negate when y > 0.

    Matches C bit-for-bit modulo float64 arithmetic.
    """
    r = np.sqrt(y * y + z * z)
    # Avoid division by zero at the origin
    r_safe = np.where(r == 0, 1.0, r)
    alpha = RAD2DEG * np.arccos(np.clip(z / r_safe, -1.0, 1.0))
    return np.where(y > 0, -alpha, alpha)


def calc_eta_angle(y: float, z: float) -> float:
    """Scalar form."""
    r = math.sqrt(y * y + z * z)
    if r == 0.0:
        return 0.0
    alpha = RAD2DEG * math.acos(max(-1.0, min(1.0, z / r)))
    return -alpha if y > 0 else alpha


def yz_from_r_eta(R: np.ndarray, Eta: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Inverse: ``Y = -R sin(Eta), Z = R cos(Eta)`` (Eta in degrees)."""
    eta_rad = Eta * DEG2RAD
    return -R * np.sin(eta_rad), R * np.cos(eta_rad)


# ─── Tilt rotation matrix ────────────────────────────────────────────────────
def tilt_matrix(tx: float, ty: float, tz: float) -> np.ndarray:
    """Returns ``Rx(tx) · (Ry(ty) · Rz(tz))`` matching C ``matrixMultiply33``
    composition order: ``TRs = Rx · Ry · Rz``.
    """
    txr, tyr, tzr = math.radians(tx), math.radians(ty), math.radians(tz)
    Rx = np.array(
        [[1, 0, 0], [0, math.cos(txr), -math.sin(txr)], [0, math.sin(txr), math.cos(txr)]],
        dtype=np.float64,
    )
    Ry = np.array(
        [[math.cos(tyr), 0, math.sin(tyr)], [0, 1, 0], [-math.sin(tyr), 0, math.cos(tyr)]],
        dtype=np.float64,
    )
    Rz = np.array(
        [[math.cos(tzr), -math.sin(tzr), 0], [math.sin(tzr), math.cos(tzr), 0], [0, 0, 1]],
        dtype=np.float64,
    )
    TRint = Ry @ Rz
    return Rx @ TRint


# ─── Per-pixel Rt (radial pixel coord with distortion) ───────────────────────
def compute_rt_eta(p: ZarrParams, panels: List[Panel]) -> Tuple[np.ndarray, np.ndarray]:
    """Compute distortion-corrected (Rt, Eta) for every pixel in (NrPixels, NrPixels).

    Matches C lines 2700-2742 for every (a, b) pixel pair. Returns:
        Rt: float64[NrPixels, NrPixels] in *pixels* (after `/= px`, residual
            correction, Lsd re-projection)
        Eta: float64[NrPixels, NrPixels] in degrees (-180, +180]

    Uses the asymmetric C convention: ``Yc = (-a + Ycen) * px`` (Y row
    flipped), ``Zc = (b - Zcen) * px``.
    """
    N = p.NrPixels
    a = np.arange(N, dtype=np.float64)
    b = np.arange(N, dtype=np.float64)
    A, B = np.meshgrid(a, b, indexing="ij")  # A:(N,N) Y row, B:(N,N) Z col

    # Per-panel offsets (vectorized panel-index lookup)
    if panels:
        from midas_peakfit.panels import panel_index_map

        # Use NrPixels x NrPixels even if NrPixelsY/Z are smaller — C does same.
        pmap = panel_index_map(N, N, panels)
        dLsd_arr = np.zeros((N, N), dtype=np.float64)
        dP2_arr = np.zeros((N, N), dtype=np.float64)
        for pn in panels:
            mask = pmap == pn.id
            if mask.any():
                dLsd_arr[mask] = pn.dLsd
                dP2_arr[mask] = pn.dP2
    else:
        dLsd_arr = np.zeros((N, N), dtype=np.float64)
        dP2_arr = np.zeros((N, N), dtype=np.float64)

    panelLsd = p.Lsd + dLsd_arr
    panelP2 = p.p2 + dP2_arr

    # Yc / Zc in physical units
    Yc = (-A + p.Ycen) * p.px
    Zc = (B - p.Zcen) * p.px

    # Apply tilt rotation: ABC = (0, Yc, Zc); ABCPr = TRs @ ABC.
    TRs = tilt_matrix(p.tx, p.ty, p.tz)
    # Vectorized: ABCPr_i = TRs[i,0]*0 + TRs[i,1]*Yc + TRs[i,2]*Zc
    ABCPr0 = TRs[0, 1] * Yc + TRs[0, 2] * Zc
    ABCPr1 = TRs[1, 1] * Yc + TRs[1, 2] * Zc
    ABCPr2 = TRs[2, 1] * Yc + TRs[2, 2] * Zc

    X = panelLsd + ABCPr0
    Y = ABCPr1
    Z = ABCPr2

    # Avoid X=0
    X_safe = np.where(X == 0, 1.0, X)
    Rad = (panelLsd / X_safe) * np.sqrt(Y * Y + Z * Z)
    Eta = calc_eta_angle_np(Y, Z)

    # Radial distortion via the shared midas_distortion kernel (single source
    # of truth, identical to transforms + calibrate-v2). Coefficients are the
    # canonical v2 harmonics (p.dist_coeffs_v2; built by the loader from the v2
    # names, or the legacy p0..p14). The per-panel p2 offset (dP2_arr) is added
    # on top as an isotropic ρ² term — iso_R2 is linear, so a per-panel
    # `iso_R2 + dP2` is equivalent to base-D + dP2·ρ².
    from midas_distortion import distortion_factor, v1_to_v2_coeffs, P_COEF_NAMES

    RNorm = Rad / p.RhoD
    coeffs = p.dist_coeffs_v2
    if coeffs is None:  # defensive: params built without the loader
        coeffs = v1_to_v2_coeffs(np.array([getattr(p, f"p{i}") for i in range(15)],
                                          dtype=np.float64))
    DistortFunc = distortion_factor(RNorm, Eta, np.asarray(coeffs, dtype=np.float64))
    DistortFunc = DistortFunc + dP2_arr * (RNorm * RNorm)  # per-panel p2 offset
    Rt = Rad * DistortFunc / p.px

    if p.residualMap is not None:
        # Residual map shape is (NrPixelsY, NrPixelsZ); pad to (N, N) with zeros
        Y_, Z_ = p.NrPixelsY, p.NrPixelsZ
        if Y_ == N and Z_ == N:
            Rt = Rt + p.residualMap
        else:
            padded = np.zeros((N, N), dtype=np.float64)
            padded[:Y_, :Z_] = p.residualMap
            Rt = Rt + padded

    Rt = Rt * (p.Lsd / panelLsd)
    return Rt, Eta


def compute_good_coords(
    p: ZarrParams, panels: List[Panel], ringRads: Optional[np.ndarray]
) -> np.ndarray:
    """Build the per-pixel threshold map ``goodCoords[NrPixels, NrPixels]``.

    For ``DoFullImage=1``: every pixel = ``Thresholds[0]``.
    Else: per-ring band ``[ringRads[r] - Width, ringRads[r] + Width]`` sets
    ``goodCoords = Thresholds[r]``. Pixels outside all bands are 0.
    """
    N = p.NrPixels
    if p.DoFullImage == 1:
        thresh0 = p.Thresholds[0] if p.Thresholds else 0.0
        return np.full((N, N), thresh0, dtype=np.float64)

    if ringRads is None or len(p.RingNrs) == 0:
        return np.zeros((N, N), dtype=np.float64)

    Rt, _ = compute_rt_eta(p, panels)

    out = np.zeros((N, N), dtype=np.float64)
    W = p.Width  # already in pixels (Width /= px done in finalize())
    for r in range(p.nRingsThresh):
        lo = ringRads[r] - W
        hi = ringRads[r] + W
        in_band = (Rt > lo) & (Rt < hi)
        out[in_band] = p.Thresholds[r]
    return out


def load_ring_radii(p: ZarrParams, result_folder: str) -> Optional[np.ndarray]:
    """Read ``{result_folder}/hkls.csv``: col 4 = ringNr, col 10 = radius.
    Returns a (nRingsThresh,) array aligned to ``p.RingNrs``, or None if absent.

    For ``DoFullImage=1``: returns zeros (unused).
    """
    if p.DoFullImage == 1:
        return np.zeros(max(1, p.nRingsThresh), dtype=np.float64)
    if p.nRingsThresh == 0:
        return None

    import os

    fn = os.path.join(result_folder, "hkls.csv")
    if not os.path.exists(fn):
        return None

    # hkls.csv: header row, then 11 whitespace-separated columns
    # (h k l D-spacing RingNr g1 g2 g3 Theta 2Theta Radius)
    # The C tool's sscanf maps "%*s %*s %*s %*s %d %*s %*s %*s %*s %*s %lf"
    # to (RingNr=col 4, Radius=col 10). Match those indices exactly.
    radii_by_ring: dict[int, float] = {}
    with open(fn, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            tokens = line.replace(",", " ").split()
            try:
                ring_nr = int(tokens[4])
                radius = float(tokens[10])
            except (IndexError, ValueError):
                continue
            radii_by_ring[ring_nr] = radius

    # hkls.csv stores radius in microns; the C tool divides by px to get pixels
    # (PeaksFittingOMPZarrRefactor.c:2550). Match that here.
    out = np.zeros(p.nRingsThresh, dtype=np.float64)
    for i, rn in enumerate(p.RingNrs):
        out[i] = radii_by_ring.get(rn, 0.0) / p.px
    return out


__all__ = [
    "DEG2RAD",
    "RAD2DEG",
    "calc_eta_angle",
    "calc_eta_angle_np",
    "yz_from_r_eta",
    "tilt_matrix",
    "compute_rt_eta",
    "compute_good_coords",
    "load_ring_radii",
]
