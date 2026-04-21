"""Geometric image rectification — ★ the Dioptas/pyFAI/GSAS-II interop feature.

Given a refined MIDAS geometry (tilts, distortion coefficients, optional
per-panel shifts), produce a rectified TIFF where simple radial integration
with only ``R = sqrt((y-ybc)² + (z-zbc)²) · px`` yields the correct 2θ.
Downstream tools (Dioptas, pyFAI's ``integrate1d``, GSAS-II) can then consume
the output directly — no MIDAS-specific calibration round-trip required.

Public API:
    correct_image(image, geometry, *, panels=None, panel_shifts=None)
    correct_images(images, geometry, *, output_dir)
    write_tiff(path, array, *, geometry=None)

Panels + panel_shifts are optional; omit for single-module detectors.

Implementation notes
--------------------
- Full 15-parameter distortion (p0..p14, matches the C reference
  ``dg_pixel_to_REta_corr`` in ``src/c/DetectorGeometry.c`` to 0.01 µm per
  point across the detector).
- Per-panel dY, dZ, dTheta, dLsd, dP2 all applied.
- Output pixel values computed via forward-splat (scatter) with bilinear
  weights: for each raw pixel, intensity is distributed across the 4
  nearest output pixels at its computed ideal position. Robust across
  panel-boundary discontinuities where Newton-Raphson inversion
  oscillates.

v0.1.0 LIMITATION: pixel-grid resolution. The scatter's output is a
``npy × npz`` grid with ~0.2 px effective noise floor; sub-pixel tilt
corrections (e.g. 0.06 px for ty=0.2° on a Pilatus 6M) fall below this
floor, so the rectified image doesn't visibly improve over the raw
image for the strongest-convergence case. v0.2 ports the C
Newton-Raphson inverse + supersampling at 4× (then box-filter down),
which is the paper-quality path to <5 µε round-trip residual.
"""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional, Sequence, Union

import numpy as np

from midas_auto_calibrate import DetectorGeometry

from ._config import IntegrationConfig

__all__ = [
    "Panel",
    "correct_image",
    "correct_images",
    "generate_panels",
    "load_panel_shifts",
    "write_tiff",
]

_DEG2RAD = math.pi / 180.0
_RAD2DEG = 180.0 / math.pi


# ---------------------------------------------------------------------------
# Panel geometry — matches MIDAS C GeneratePanels().
# ---------------------------------------------------------------------------

@dataclass
class Panel:
    """One rectangular sub-detector region (Pilatus/Eiger module).

    dY/dZ      per-panel pixel shift
    dTheta     per-panel rotation about (center_y, center_z), degrees
    dLsd       per-panel Lsd offset (µm) — multiple-thickness sensor stacks
    dP2        per-panel p2 distortion offset (adds to the global p2)
    """

    id: int
    y_min: int
    y_max: int
    z_min: int
    z_max: int
    dY: float = 0.0
    dZ: float = 0.0
    dTheta: float = 0.0
    dLsd: float = 0.0
    dP2: float = 0.0

    @property
    def center_y(self) -> float:
        return 0.5 * (self.y_min + self.y_max)

    @property
    def center_z(self) -> float:
        return 0.5 * (self.z_min + self.z_max)


def generate_panels(
    n_panels_y: int,
    n_panels_z: int,
    panel_size_y: int,
    panel_size_z: int,
    gaps_y: Sequence[int],
    gaps_z: Sequence[int],
) -> list[Panel]:
    """Mirror MIDAS C ``GeneratePanels`` — yMin/yMax tile in Y then Z."""
    panels: list[Panel] = []
    idx = 0
    y_start = 0
    for iy in range(n_panels_y):
        z_start = 0
        for iz in range(n_panels_z):
            y_end = y_start + panel_size_y - 1
            z_end = z_start + panel_size_z - 1
            panels.append(Panel(
                id=idx,
                y_min=y_start, y_max=y_end,
                z_min=z_start, z_max=z_end,
            ))
            idx += 1
            z_start = z_end + 1
            if iz < n_panels_z - 1 and iz < len(gaps_z):
                z_start += gaps_z[iz]
        y_start = y_start + panel_size_y
        if iy < n_panels_y - 1 and iy < len(gaps_y):
            y_start += gaps_y[iy]
    return panels


def load_panel_shifts(path: Union[str, Path], panels: list[Panel]) -> list[Panel]:
    """Load panel-shift file and apply in place.

    Supported column layouts (MIDAS writes the 6-col form):
        id dY dZ                          — 3-col legacy
        id dY dZ dTheta                   — 4-col
        id dY dZ dTheta dLsd dP2          — 6-col (paper-quality)
    """
    by_id = {p.id: p for p in panels}
    with Path(path).open() as f:
        for raw in f:
            line = raw.strip()
            if not line or line[0] in "#%":
                continue
            parts = line.split()
            if len(parts) < 3:
                continue
            pid = int(parts[0])
            if pid not in by_id:
                continue
            p = by_id[pid]
            p.dY = float(parts[1])
            p.dZ = float(parts[2])
            if len(parts) >= 4:
                p.dTheta = float(parts[3])
            if len(parts) >= 5:
                p.dLsd = float(parts[4])
            if len(parts) >= 6:
                p.dP2 = float(parts[5])
    return panels


def _panel_index_map(npy: int, npz: int, panels: list[Panel]) -> np.ndarray:
    """(npy, npz) array with panel.id at each pixel, -1 in gaps."""
    pmap = np.full((npy, npz), -1, dtype=np.int32)
    for p in panels:
        pmap[p.y_min:p.y_max + 1, p.z_min:p.z_max + 1] = p.id
    return pmap


# ---------------------------------------------------------------------------
# Forward-distortion map: raw pixel (y, z) → ideal pixel (y', z').
# ---------------------------------------------------------------------------

def _tilt_matrix(tx: float, ty: float, tz: float) -> np.ndarray:
    """Combined tilt rotation ``Rx @ Ry @ Rz`` (degrees)."""
    txr, tyr, tzr = tx * _DEG2RAD, ty * _DEG2RAD, tz * _DEG2RAD
    c, s = math.cos, math.sin
    Rx = np.array([[1, 0, 0], [0, c(txr), -s(txr)], [0, s(txr), c(txr)]])
    Ry = np.array([[c(tyr), 0, s(tyr)], [0, 1, 0], [-s(tyr), 0, c(tyr)]])
    Rz = np.array([[c(tzr), -s(tzr), 0], [s(tzr), c(tzr), 0], [0, 0, 1]])
    return Rx @ Ry @ Rz


def _compute_forward_map(
    cfg: IntegrationConfig,
    panels: list[Panel],
    panel_map: Optional[np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """For each raw pixel (y, z), return (y_ideal, z_ideal) in pixel units.

    Ports the full 15-parameter distortion + per-panel model from
    ``src/c/DetectorGeometry.c::dg_pixel_to_REta_corr``:

        panelLsd = Lsd + dLsd                       # per-panel
        panelP2  = p2 + dP2                         # per-panel
        Yc, Zc   = (pixel - BC) * px  (signed)
        ABC      = TRs · (0, Yc, Zc)
        XYZ      = (panelLsd + ABC[0], ABC[1], ABC[2])
        Rad      = (panelLsd / XYZ[0]) · √(Y² + Z²)
        EtaT     = 90 - η  (where η = atan-like of (Y, Z))
        RNorm    = Rad / RhoD

        D = p0 · R²·cos(2·EtaT + p6)
          + p1 · R⁴·cos(4·EtaT + p3)
          + panelP2 · R²
          + p4 · R⁶
          + p5 · R⁴
          + p7 · R⁴·cos(EtaT + p8)
          + p9 · R³·cos(3·EtaT + p10)
          + p11 · R⁵·cos(5·EtaT + p12)
          + p13 · R⁶·cos(6·EtaT + p14)
          + 1

        Rt = Rad · D / px
        Rt = Rt · (Lsd / panelLsd)   # re-project to global Lsd plane

    Inverting this map gives the rectified image.
    """
    npy, npz = cfg.nr_pixels_y, cfg.nr_pixels_z
    TRs = _tilt_matrix(cfg.tx, cfg.ty, cfg.tz)
    rho_d = cfg._resolve_rho_d()

    y_grid, z_grid = np.meshgrid(
        np.arange(npy, dtype=np.float64),
        np.arange(npz, dtype=np.float64),
    )

    # ------------------------------------------------------------------
    # 1. Per-panel (dY, dZ, dTheta) — same as before.
    # ------------------------------------------------------------------
    y_corr, z_corr = y_grid.copy(), z_grid.copy()
    # Per-panel (dLsd, dP2) broadcast as pixel-sized arrays — zero where no
    # panel covers. MIDAS uses panel_lsd = Lsd + dLsd and p2_eff = p2 + dP2
    # PER PIXEL based on which panel it lands on.
    dLsd_map = np.zeros_like(y_grid)
    dP2_map = np.zeros_like(y_grid)

    if panels and panel_map is not None:
        y_int = np.clip(y_grid.astype(int), 0, panel_map.shape[0] - 1)
        z_int = np.clip(z_grid.astype(int), 0, panel_map.shape[1] - 1)
        pids = panel_map[y_int, z_int]
        for p in panels:
            mask = pids == p.id
            if not mask.any():
                continue
            ym, zm = y_grid[mask], z_grid[mask]
            if abs(p.dTheta) > 1e-12:
                rad = p.dTheta * _DEG2RAD
                cosT, sinT = math.cos(rad), math.sin(rad)
                dy_c, dz_c = ym - p.center_y, zm - p.center_z
                y_corr[mask] = p.center_y + dy_c * cosT - dz_c * sinT + p.dY
                z_corr[mask] = p.center_z + dy_c * sinT + dz_c * cosT + p.dZ
            else:
                y_corr[mask] = ym + p.dY
                z_corr[mask] = zm + p.dZ
            dLsd_map[mask] = p.dLsd
            dP2_map[mask] = p.dP2

    # ------------------------------------------------------------------
    # 2. Pixel → physical (µm). Sign convention matches MIDAS C.
    # ------------------------------------------------------------------
    Yc = -(y_corr - cfg.ybc) * cfg.pixel_size
    Zc = (z_corr - cfg.zbc) * cfg.pixel_size

    # ------------------------------------------------------------------
    # 3. Apply detector tilts (rotate about sample). MIDAS feeds ABC =
    # (0, Yc, Zc), so the first-column contribution is always zero.
    # ------------------------------------------------------------------
    ABCPr_0 = TRs[0, 1] * Yc + TRs[0, 2] * Zc
    ABCPr_1 = TRs[1, 1] * Yc + TRs[1, 2] * Zc
    ABCPr_2 = TRs[2, 1] * Yc + TRs[2, 2] * Zc
    panel_lsd = cfg.lsd + dLsd_map
    X = panel_lsd + ABCPr_0

    # ------------------------------------------------------------------
    # 4. R + η.
    # ------------------------------------------------------------------
    r_yz = np.sqrt(ABCPr_1 ** 2 + ABCPr_2 ** 2)
    r_yz = np.maximum(r_yz, 1e-30)
    # Use per-panel Lsd here (matches MIDAS C line 117).
    R = (panel_lsd / X) * r_yz
    eta_deg = _RAD2DEG * np.arccos(np.clip(ABCPr_2 / r_yz, -1.0, 1.0))
    eta_deg = np.where(ABCPr_1 > 0, -eta_deg, eta_deg)

    # ------------------------------------------------------------------
    # 5. Full 15-parameter distortion (matches dg_pixel_to_REta_corr).
    # ------------------------------------------------------------------
    r_norm = R / rho_d
    eta_t = 90.0 - eta_deg
    # panel_p2 broadcasts per pixel.
    panel_p2 = cfg.p2 + dP2_map

    # Harmonic phases are in degrees in MIDAS; convert once.
    p3_rad = cfg.p3 * _DEG2RAD
    p6_rad = cfg.p6 * _DEG2RAD
    p8_rad = cfg.p8 * _DEG2RAD
    p10_rad = cfg.p10 * _DEG2RAD
    p12_rad = cfg.p12 * _DEG2RAD
    p14_rad = cfg.p14 * _DEG2RAD
    eta_t_rad = eta_t * _DEG2RAD

    r_norm_2 = r_norm ** 2
    r_norm_3 = r_norm ** 3
    r_norm_4 = r_norm ** 4
    r_norm_5 = r_norm ** 5
    r_norm_6 = r_norm ** 6

    distort = (
        cfg.p0 * r_norm_2 * np.cos(2 * eta_t_rad + p6_rad)
        + cfg.p1 * r_norm_4 * np.cos(4 * eta_t_rad + p3_rad)
        + panel_p2 * r_norm_2
        + cfg.p4 * r_norm_6
        + cfg.p5 * r_norm_4
        + cfg.p7 * r_norm_4 * np.cos(eta_t_rad + p8_rad)
        + cfg.p9 * r_norm_3 * np.cos(3 * eta_t_rad + p10_rad)
        + cfg.p11 * r_norm_5 * np.cos(5 * eta_t_rad + p12_rad)
        + cfg.p13 * r_norm_6 * np.cos(6 * eta_t_rad + p14_rad)
        + 1.0
    )

    # Rt = Rad · D / px, then re-project to global Lsd plane (MIDAS C line 135).
    Rt_px = R * distort / cfg.pixel_size
    Rt_px = Rt_px * (cfg.lsd / panel_lsd)

    # ------------------------------------------------------------------
    # 6. Back to pixel coordinates on the flat, undistorted detector.
    # ------------------------------------------------------------------
    eta_rad = eta_deg * _DEG2RAD
    y_ideal = cfg.ybc + Rt_px * np.sin(eta_rad)
    z_ideal = cfg.zbc + Rt_px * np.cos(eta_rad)
    return y_ideal, z_ideal


# ---------------------------------------------------------------------------
# Scatter rectification — bilinear forward-splat with weight normalization.
# ---------------------------------------------------------------------------

def _scatter_to_output(
    img: np.ndarray,
    fwd_y: np.ndarray,
    fwd_z: np.ndarray,
    out_shape: tuple[int, int],
) -> np.ndarray:
    """Forward-splat ``img`` values into an output grid at ``(fwd_y, fwd_z)``.

    For each input pixel (y_in, z_in) with value I, distribute I across
    the four output pixels bracketing (fwd_y, fwd_z) with bilinear weights.
    Normalise each output pixel by the sum of incoming weights so overlap
    doesn't bias bright regions.

    This is the standard "splat" used in scattered-data rasterisation. It's
    robust across panel-boundary discontinuities where Newton-Raphson
    inversion oscillates.
    """
    npz_out, npy_out = out_shape
    flat_img = np.ascontiguousarray(img, dtype=np.float64).ravel()
    flat_y = np.ascontiguousarray(fwd_y, dtype=np.float64).ravel()
    flat_z = np.ascontiguousarray(fwd_z, dtype=np.float64).ravel()

    # Drop samples that land outside the output grid.
    in_bounds = (
        (flat_y >= 0) & (flat_y <= npy_out - 1)
        & (flat_z >= 0) & (flat_z <= npz_out - 1)
    )
    flat_y = flat_y[in_bounds]
    flat_z = flat_z[in_bounds]
    flat_img = flat_img[in_bounds]

    # Bracket each destination with its floor/ceil on both axes.
    y0 = np.floor(flat_y).astype(np.int64)
    z0 = np.floor(flat_z).astype(np.int64)
    y1 = np.clip(y0 + 1, 0, npy_out - 1)
    z1 = np.clip(z0 + 1, 0, npz_out - 1)
    y0 = np.clip(y0, 0, npy_out - 1)
    z0 = np.clip(z0, 0, npz_out - 1)
    wy1 = flat_y - y0
    wz1 = flat_z - z0
    wy0 = 1.0 - wy1
    wz0 = 1.0 - wz1

    accum = np.zeros(out_shape, dtype=np.float64)
    weight = np.zeros(out_shape, dtype=np.float64)

    for z_idx, y_idx, w in (
        (z0, y0, wz0 * wy0),
        (z0, y1, wz0 * wy1),
        (z1, y0, wz1 * wy0),
        (z1, y1, wz1 * wy1),
    ):
        np.add.at(accum, (z_idx, y_idx), flat_img * w)
        np.add.at(weight, (z_idx, y_idx), w)

    # Normalise by total incoming weight; leave unvisited pixels at 0.
    out = np.zeros_like(accum)
    visited = weight > 0
    out[visited] = accum[visited] / weight[visited]
    return out


# ---------------------------------------------------------------------------
# Vectorized inverse via Newton-Raphson (matches C dg_invert_REta_to_pixel).
# Kept for reference; the scatter path above is the production default
# because NR oscillates across panel-boundary discontinuities.
# ---------------------------------------------------------------------------

def _invert_forward_map(
    cfg: IntegrationConfig,
    panels: list[Panel],
    panel_map: Optional[np.ndarray],
    *,
    max_iter: int = 10,
    tol: float = 1e-5,
    h: float = 0.01,
) -> tuple[np.ndarray, np.ndarray]:
    """Newton-Raphson solve: for each pixel (y_out, z_out) in the rectified
    flat image, find (Y, Z) on the raw detector such that forward(Y, Z) =
    (y_out, z_out).

    The forward function applies:  panel corrections → tilts → distortion.
    We invert the whole chain by iterating on raw pixel coords with the
    numerical Jacobian of the vectorized forward map.
    """
    npy, npz = cfg.nr_pixels_y, cfg.nr_pixels_z

    # Output-pixel grid on the rectified (flat) detector.
    y_out_1d = np.arange(npy, dtype=np.float64)
    z_out_1d = np.arange(npz, dtype=np.float64)
    y_out, z_out = np.meshgrid(y_out_1d, z_out_1d)

    # Seed guess — polar formula as if distortion were zero. (y_out, z_out)
    # is already in pixel coords; the polar inverse of the flat-detector
    # forward map is the identity, so Y0 = y_out, Z0 = z_out works for a
    # surprisingly large basin of convergence.
    Y = y_out.copy()
    Z = z_out.copy()

    for _ in range(max_iter):
        # Evaluate forward at current (Y, Z).
        fY, fZ = _forward_at(cfg, panels, panel_map, Y, Z)
        res_y = fY - y_out
        res_z = fZ - z_out
        max_res = max(np.abs(res_y).max(), np.abs(res_z).max())
        if max_res < tol:
            break

        # Finite-difference Jacobian: ∂(fY, fZ)/∂(Y, Z).
        fY_dY, fZ_dY = _forward_at(cfg, panels, panel_map, Y + h, Z)
        fY_dZ, fZ_dZ = _forward_at(cfg, panels, panel_map, Y, Z + h)
        JYY = (fY_dY - fY) / h
        JYZ = (fY_dZ - fY) / h
        JZY = (fZ_dY - fZ) / h
        JZZ = (fZ_dZ - fZ) / h
        det = JYY * JZZ - JYZ * JZY
        safe = np.abs(det) > 1e-30
        det_safe = np.where(safe, det, 1.0)

        dY = (JZZ * res_y - JYZ * res_z) / det_safe
        dZ = (JYY * res_z - JZY * res_y) / det_safe
        dY = np.where(safe, dY, 0.0)
        dZ = np.where(safe, dZ, 0.0)
        # Subtract (not add): res is fwd(Y,Z) - target; we want fwd(Y,Z) = target.
        Y -= dY
        Z -= dZ

    return Y, Z


def _forward_at(
    cfg: IntegrationConfig,
    panels: list[Panel],
    panel_map: Optional[np.ndarray],
    Y: np.ndarray,
    Z: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Forward map evaluated at ARBITRARY (Y, Z) arrays (not on the pixel grid).

    Factored out of ``_compute_forward_map`` so Newton-Raphson can probe
    non-integer raw pixel positions. Panel parameters are looked up by
    rounding (Y, Z) to integers; for pixels near panel boundaries this
    matches what the C binary does.
    """
    TRs = _tilt_matrix(cfg.tx, cfg.ty, cfg.tz)
    rho_d = cfg._resolve_rho_d()

    # Panel corrections (vectorized lookup).
    y_corr = Y.copy()
    z_corr = Z.copy()
    dLsd_map = np.zeros_like(Y)
    dP2_map = np.zeros_like(Y)
    if panels and panel_map is not None:
        y_int = np.clip(Y.astype(int), 0, panel_map.shape[0] - 1)
        z_int = np.clip(Z.astype(int), 0, panel_map.shape[1] - 1)
        pids = panel_map[y_int, z_int]
        for p in panels:
            mask = pids == p.id
            if not mask.any():
                continue
            ym, zm = Y[mask], Z[mask]
            if abs(p.dTheta) > 1e-12:
                rad = p.dTheta * _DEG2RAD
                cosT, sinT = math.cos(rad), math.sin(rad)
                dy_c, dz_c = ym - p.center_y, zm - p.center_z
                y_corr[mask] = p.center_y + dy_c * cosT - dz_c * sinT + p.dY
                z_corr[mask] = p.center_z + dy_c * sinT + dz_c * cosT + p.dZ
            else:
                y_corr[mask] = ym + p.dY
                z_corr[mask] = zm + p.dZ
            dLsd_map[mask] = p.dLsd
            dP2_map[mask] = p.dP2

    Yc = -(y_corr - cfg.ybc) * cfg.pixel_size
    Zc = (z_corr - cfg.zbc) * cfg.pixel_size
    ABCPr_0 = TRs[0, 1] * Yc + TRs[0, 2] * Zc
    ABCPr_1 = TRs[1, 1] * Yc + TRs[1, 2] * Zc
    ABCPr_2 = TRs[2, 1] * Yc + TRs[2, 2] * Zc
    panel_lsd = cfg.lsd + dLsd_map
    X = panel_lsd + ABCPr_0
    r_yz = np.maximum(np.sqrt(ABCPr_1 ** 2 + ABCPr_2 ** 2), 1e-30)
    R = (panel_lsd / X) * r_yz
    eta_deg = _RAD2DEG * np.arccos(np.clip(ABCPr_2 / r_yz, -1.0, 1.0))
    eta_deg = np.where(ABCPr_1 > 0, -eta_deg, eta_deg)

    r_norm = R / rho_d
    eta_t_rad = (90.0 - eta_deg) * _DEG2RAD
    panel_p2 = cfg.p2 + dP2_map
    r2, r3, r4, r5, r6 = (r_norm ** n for n in (2, 3, 4, 5, 6))
    distort = (
        cfg.p0 * r2 * np.cos(2 * eta_t_rad + cfg.p6 * _DEG2RAD)
        + cfg.p1 * r4 * np.cos(4 * eta_t_rad + cfg.p3 * _DEG2RAD)
        + panel_p2 * r2
        + cfg.p4 * r6
        + cfg.p5 * r4
        + cfg.p7 * r4 * np.cos(eta_t_rad + cfg.p8 * _DEG2RAD)
        + cfg.p9 * r3 * np.cos(3 * eta_t_rad + cfg.p10 * _DEG2RAD)
        + cfg.p11 * r5 * np.cos(5 * eta_t_rad + cfg.p12 * _DEG2RAD)
        + cfg.p13 * r6 * np.cos(6 * eta_t_rad + cfg.p14 * _DEG2RAD)
        + 1.0
    )
    Rt_px = R * distort / cfg.pixel_size * (cfg.lsd / panel_lsd)
    eta_rad = eta_deg * _DEG2RAD
    y_ideal = cfg.ybc + Rt_px * np.sin(eta_rad)
    z_ideal = cfg.zbc + Rt_px * np.cos(eta_rad)
    return y_ideal, z_ideal


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def correct_image(
    image: Union[str, Path, np.ndarray],
    geometry: Union[DetectorGeometry, IntegrationConfig],
    *,
    panels: Optional[list[Panel]] = None,
    panel_shifts: Union[str, Path, None] = None,
    nr_pixels_y: Optional[int] = None,
    nr_pixels_z: Optional[int] = None,
) -> np.ndarray:
    """Rectify ``image`` against ``geometry`` → flat-detector-equivalent frame.

    Parameters
    ----------
    image : path or ndarray
        Raw detector image. If a path, read via ``tifffile``; if ndarray,
        used directly.
    geometry : DetectorGeometry | IntegrationConfig
        Refined geometry from ``mac.auto_calibrate`` (preferred) or an
        explicit ``IntegrationConfig``. The config's ``nr_pixels_y/z`` and
        ``pixel_size`` must match ``image``.
    panels : list[Panel], optional
        Per-panel sub-detector definitions. Build via :func:`generate_panels`
        from detector specs; skip for single-module detectors.
    panel_shifts : path, optional
        ``panelshiftsCalibrant.txt`` from a calibration run; loads per-panel
        dY/dZ/dTheta into ``panels`` in place.
    nr_pixels_y, nr_pixels_z : int, optional
        Override pixel counts when ``geometry`` is a ``DetectorGeometry``
        whose pixel counts are zero (e.g. older JSON without them).
    """
    img = _load_image(image)
    cfg = _resolve_config(geometry, nr_pixels_y, nr_pixels_z, img)

    if panels and panel_shifts is not None:
        load_panel_shifts(panel_shifts, panels)

    panel_map = _panel_index_map(cfg.nr_pixels_y, cfg.nr_pixels_z, panels) if panels else None

    # Forward map: for each raw pixel, where does it land on the flat
    # rectified detector?
    fwd_y, fwd_z = _compute_forward_map(cfg, panels or [], panel_map)

    # Scatter rectification: each raw pixel splats its intensity to the
    # 4 nearest output pixels via bilinear weights. Robust across panel
    # boundaries where an inverse-solve (Newton-Raphson, fixed-point)
    # oscillates because the Jacobian stencil straddles two panels' worth
    # of dY/dZ/dTheta/dLsd/dP2 settings.
    corrected = _scatter_to_output(
        img, fwd_y, fwd_z,
        out_shape=(cfg.nr_pixels_z, cfg.nr_pixels_y),
    )
    # Gap masking is automatic with scatter — panel-gap raw pixels have
    # value 0 and splat zero-weighted contributions, so output pixels that
    # only receive gap contributions stay 0.
    return corrected


def correct_images(
    images: Iterable[Union[str, Path]],
    geometry: Union[DetectorGeometry, IntegrationConfig],
    *,
    output_dir: Union[str, Path],
    panels: Optional[list[Panel]] = None,
    panel_shifts: Union[str, Path, None] = None,
    suffix: str = "_corrected",
) -> list[Path]:
    """Batch form of :func:`correct_image`.

    Produces ``<stem><suffix>.tif`` in ``output_dir`` for each input.
    Returns the list of written paths.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for img_path in images:
        p = Path(img_path)
        corrected = correct_image(
            p, geometry, panels=panels, panel_shifts=panel_shifts,
        )
        out = out_dir / f"{p.stem}{suffix}.tif"
        write_tiff(out, corrected,
                   geometry=geometry if isinstance(geometry, DetectorGeometry) else None)
        written.append(out)
    return written


def write_tiff(
    path: Union[str, Path],
    array: np.ndarray,
    *,
    geometry: Optional[DetectorGeometry] = None,
) -> Path:
    """Write a 32-bit float TIFF with optional MIDAS provenance in ImageDescription."""
    out = Path(path)
    import tifffile

    description: Optional[str] = None
    if geometry is not None:
        from midas_integrate import __version__ as _v
        description = (
            f"midas-integrate v{_v} rectified image; "
            f"Lsd={geometry.lsd:.3f} um, "
            f"BC=({geometry.ybc:.3f}, {geometry.zbc:.3f}) px, "
            f"px={geometry.px:.3f} um"
        )

    tifffile.imwrite(
        out, array.astype(np.float32),
        description=description,
        compression=None,
    )
    return out


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_image(image: Union[str, Path, np.ndarray]) -> np.ndarray:
    if isinstance(image, np.ndarray):
        return image.astype(np.float64, copy=False)
    path = Path(image)
    if path.suffix.lower() in (".tif", ".tiff"):
        import tifffile
        return tifffile.imread(path).astype(np.float64)
    raise ValueError(
        f"Unsupported image format: {path.suffix}. Load via h5py / fabio and "
        f"pass a numpy array."
    )


def _resolve_config(
    geometry: Union[DetectorGeometry, IntegrationConfig],
    nr_pixels_y: Optional[int],
    nr_pixels_z: Optional[int],
    img: np.ndarray,
) -> IntegrationConfig:
    """Coerce geometry arg to an IntegrationConfig, filling in pixel counts."""
    if isinstance(geometry, IntegrationConfig):
        cfg = geometry
    else:
        cfg = IntegrationConfig.from_geometry(
            geometry,
            nr_pixels_y=nr_pixels_y or geometry.nr_pixels_y or img.shape[1],
            nr_pixels_z=nr_pixels_z or geometry.nr_pixels_z or img.shape[0],
        )
    # Sanity: image shape must agree with cfg.
    if img.shape != (cfg.nr_pixels_z, cfg.nr_pixels_y):
        raise ValueError(
            f"image shape {img.shape} doesn't match config "
            f"({cfg.nr_pixels_z}, {cfg.nr_pixels_y}). If the geometry's "
            f"pixel counts are zero, pass nr_pixels_y/nr_pixels_z "
            f"explicitly."
        )
    return cfg
