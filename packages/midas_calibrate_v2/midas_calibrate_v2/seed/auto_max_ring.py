"""Auto-detect the maximum usable ring number — port of v1's
``AutoCalibrateZarr.auto_detect_max_ring``.

Three cascading criteria:

1. Detector extent — the ring must fit within 95% of the BC-to-corner
   distance.
2. Adjacent ring separation — drop after ``max_overlap_run`` consecutive
   pairs separated by < ``min_separation_px``.
3. SNR — if a background-subtracted image is supplied, drop rings with
   SNR < ``snr_threshold`` once we hit ``max_low_snr_run`` consecutive
   low-SNR rings.

Returns ``0`` (= no limit) if all rings are usable, else the maximum
ring index (1-based, matching v1's convention).
"""
from __future__ import annotations

from typing import Optional

import numpy as np


def auto_detect_max_ring(
    sim_rads_px: np.ndarray,
    npy: int, npz: int,
    bc_y: float, bc_z: float,
    *,
    data: Optional[np.ndarray] = None,
    min_separation_px: float = 5.0,
    max_overlap_run: int = 3,
    snr_threshold: float = 3.0,
    max_low_snr_run: int = 3,
) -> int:
    """Return the max usable ring index (1-based), or 0 for "no limit".

    Parameters
    ----------
    sim_rads_px : array of simulated ring radii in pixels (sorted ascending).
    npy, npz : detector dimensions (pixels).
    bc_y, bc_z : beam-center pixel coordinates.
    data : optional background-subtracted image (shape compatible with the
        detector).  Used for the SNR filter.

    Notes
    -----
    Identical to v1's logic; only the logger calls were dropped (caller
    can wrap with their own logging).
    """
    sim_rads_px = np.asarray(sim_rads_px, dtype=np.float64)
    if sim_rads_px.size <= 1:
        return 0

    # Criterion 1: detector extent.
    corners = np.array([
        [0, 0], [0, npz - 1], [npy - 1, 0], [npy - 1, npz - 1],
    ], dtype=float)
    max_R = max(
        np.sqrt((c[0] - bc_y) ** 2 + (c[1] - bc_z) ** 2) for c in corners
    )
    extent_limit = 0.95 * max_R

    # Criterion 2: ring separation.
    close_run = 0
    separation_limit = len(sim_rads_px)
    for i in range(len(sim_rads_px)):
        if sim_rads_px[i] > extent_limit:
            separation_limit = i
            break
        if i < len(sim_rads_px) - 1:
            sep = sim_rads_px[i + 1] - sim_rads_px[i]
            if sep < min_separation_px:
                close_run += 1
                if close_run >= max_overlap_run:
                    separation_limit = i + 2 - max_overlap_run
                    break
            else:
                close_run = 0
    max_ring = separation_limit

    # Criterion 3: SNR via radial profile.
    if data is not None and max_ring > 0:
        try:
            nz_img, ny_img = data.shape
            yy, zz = np.meshgrid(np.arange(ny_img), np.arange(nz_img))
            R_img = np.sqrt((yy - bc_y) ** 2 + (zz - bc_z) ** 2)
            max_r_int = int(np.ceil(
                sim_rads_px[min(max_ring, len(sim_rads_px)) - 1]
            )) + 30
            max_r_int = min(max_r_int, int(np.max(R_img)))
            r_idx = np.clip(R_img.astype(int), 0, max_r_int)
            radial_sum = np.bincount(
                r_idx.ravel(), weights=data.ravel(), minlength=max_r_int + 1,
            )
            radial_count = np.bincount(r_idx.ravel(), minlength=max_r_int + 1)
            radial_count[radial_count == 0] = 1
            radial_profile = radial_sum / radial_count

            snr_limit = max_ring
            low_snr_run = 0
            for i in range(min(max_ring, len(sim_rads_px))):
                rc = int(round(sim_rads_px[i]))
                if rc >= len(radial_profile):
                    snr_limit = i
                    break
                pk_hw = max(8, int(0.03 * rc))
                bg_hw = 3 * pk_hw
                lo_pk = max(0, rc - pk_hw)
                hi_pk = min(len(radial_profile), rc + pk_hw + 1)
                peak_val = np.max(radial_profile[lo_pk:hi_pk])
                lo_bg = max(0, rc - bg_hw)
                hi_bg = min(len(radial_profile), rc + bg_hw + 1)
                bg_mask = np.ones(hi_bg - lo_bg, dtype=bool)
                core_lo = max(0, rc - pk_hw - lo_bg)
                core_hi = min(hi_bg - lo_bg, rc + pk_hw + 1 - lo_bg)
                bg_mask[core_lo:core_hi] = False
                bg_vals = radial_profile[lo_bg:hi_bg][bg_mask]
                if len(bg_vals) > 2:
                    bg_mean = np.mean(bg_vals)
                    bg_std = max(np.std(bg_vals), 1e-10)
                    snr = (peak_val - bg_mean) / bg_std
                else:
                    snr = 0.0
                if snr < snr_threshold:
                    low_snr_run += 1
                    if low_snr_run >= max_low_snr_run:
                        snr_limit = i + 1 - max_low_snr_run
                        break
                else:
                    low_snr_run = 0
            max_ring = min(max_ring, snr_limit)
        except Exception:
            pass

    if max_ring >= len(sim_rads_px) or max_ring <= 0:
        if (max_ring <= 0 and separation_limit > 0
                and separation_limit < len(sim_rads_px)):
            return min(separation_limit, 25)
        return 0
    return int(max_ring)


__all__ = ["auto_detect_max_ring"]
