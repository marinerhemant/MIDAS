"""Signed per-spot residual decomposition (obs vs fitted-grain prediction).

Motivation (emerson_oct25 / recon_3580 investigation, 2026-07): the legacy
``Grains.csv`` columns report only the *euclidean* position error (DiffPos)
and the *absolute* omega error (DiffOme). Those magnitudes hide the structure
of the misfit. Decomposing every matched spot's residual into **signed**
components makes systematics legible:

- ``dY, dZ``       : signed lab-frame detector residuals (obs - exp, µm).
- ``dRad``         : radial component (+ = observed further from beam center).
                     A ring-dependent median dRad/R (ppm) is a d-spacing /
                     reference-lattice or distance calibration error.
- ``dTan``         : tangential (eta-direction) component. Dominated by
                     grain-level orientation scatter; an eta-antisymmetric
                     median is a beam-center / tilt signature.
- ``dOme``         : signed omega residual (deg, wrapped). A bias vs eta is
                     the classic wedge / rotation-axis-tilt signature.
- ``internal_angle``: angle between observed and predicted scattering
                     vectors (deg). Its per-grain median is the orientation-
                     space misfit floor (intragranular spread when it is
                     intensity- and width-independent).

Everything here is pure numpy post-processing of the per-spot residual table
collected while building SpotMatrix rows; there is nothing differentiable to
preserve, so no torch pass-through is required.
"""

from __future__ import annotations

import math
from typing import Dict, Optional

import numpy as np


__all__ = [
    "SPOT_RESIDUAL_COLS",
    "build_spot_residual_row",
    "decompose_residuals",
]


# Column layout of the per-spot residual table assembled alongside the
# SpotMatrix rows (one row per resolved grain-spot claim).
SPOT_RESIDUAL_COLS = (
    "grain_idx",        # 0  index into the out_grains list (NOT GrainID)
    "spot_id",          # 1
    "ring_nr",          # 2
    "eta_deg",          # 3  MIDAS convention atan2(-YLab, ZLab)
    "dy_um",            # 4  y_obs - y_exp   (position-corrected obs)
    "dz_um",            # 5  z_obs - z_exp
    "drad_um",          # 6  radial component of (dy, dz)
    "dtan_um",          # 7  tangential component of (dy, dz)
    "dome_deg",         # 8  signed omega residual, wrapped to [-180, 180)
    "internal_angle_deg",  # 9
    "r_exp_um",         # 10 expected radial distance sqrt(y_exp^2 + z_exp^2)
)


def build_spot_residual_row(
    grain_idx: int,
    spot_id: float,
    ring_nr: float,
    fb_row: np.ndarray,
) -> Optional[list]:
    """Build one residual-table row from a FitBest per-spot record.

    ``fb_row`` is one 22-double FitBest row (see ``io/binary.py``):
    col 1/2 = YObsCorrPos/ZObsCorrPos (position-corrected observed, µm),
    col 3 = OmegaObsCorrPos (deg), col 7/8/9 = YExp/ZExp/OmegaExp,
    col 19 = InternalAngle (deg).

    Returns ``None`` when the expected position is degenerate (all-zero
    padding row).
    """
    y_obs = float(fb_row[1]); z_obs = float(fb_row[2])
    y_exp = float(fb_row[7]); z_exp = float(fb_row[8])
    r_exp = math.hypot(y_exp, z_exp)
    if r_exp <= 0.0:
        return None
    dy = y_obs - y_exp
    dz = z_obs - z_exp
    # Radial / tangential unit vectors at the *observed* spot azimuth; use
    # the observed position so the decomposition degrades gracefully when
    # the prediction is far off.
    r_obs = math.hypot(y_obs, z_obs)
    if r_obs <= 0.0:
        return None
    ur_y = y_obs / r_obs
    ur_z = z_obs / r_obs
    drad = dy * ur_y + dz * ur_z
    dtan = -dy * ur_z + dz * ur_y
    dome = (float(fb_row[3]) - float(fb_row[9]) + 180.0) % 360.0 - 180.0
    eta_deg = math.degrees(math.atan2(-y_obs, z_obs))
    return [
        float(grain_idx), float(spot_id), float(ring_nr), eta_deg,
        dy, dz, drad, dtan, dome, float(fb_row[19]), r_exp,
    ]


def _median_or_nan(x: np.ndarray) -> float:
    return float(np.median(x)) if x.size else float("nan")


def _mad_std(x: np.ndarray) -> float:
    """Robust std via scaled MAD (1.4826 * median(|x - median|))."""
    if x.size == 0:
        return float("nan")
    return float(1.4826 * np.median(np.abs(x - np.median(x))))


def decompose_residuals(
    spot_tbl: np.ndarray,
    n_grains: int,
    eta_bin_deg: float = 30.0,
) -> Dict[str, np.ndarray]:
    """Aggregate the per-spot residual table into diagnostic arrays.

    Parameters
    ----------
    spot_tbl : ndarray (n_spots, 11)
        Rows built by :func:`build_spot_residual_row`
        (layout :data:`SPOT_RESIDUAL_COLS`).
    n_grains : int
        Number of grains in the output list; per-grain arrays are this long
        (NaN where a grain contributed no residual rows).
    eta_bin_deg : float
        Width of the azimuthal bins for the global eta profiles.

    Returns
    -------
    dict of numpy arrays, keys prefixed for the ``residuals/`` h5 group:
        per-grain (n_grains,):
            ``grain_med_dy_um, grain_med_dz_um, grain_med_drad_um,
            grain_med_dtan_um, grain_med_dome_deg,
            grain_med_internal_angle_deg, grain_mad_dtan_um,
            grain_n_spots``
        per-ring (n_rings,):
            ``ring_nr, ring_med_drad_um, ring_drad_ppm, ring_mad_drad_um,
            ring_n_spots``
        eta profile (n_bins,):
            ``eta_bin_lo_deg, eta_med_drad_um, eta_med_dtan_um,
            eta_med_dome_deg, eta_n_spots``
        scalars (0-d arrays):
            ``overall_med_{dy,dz,drad,dtan}_um, overall_med_dome_deg,
            overall_mad_{drad,dtan}_um, overall_mad_dome_deg,
            overall_med_internal_angle_deg``
    """
    out: Dict[str, np.ndarray] = {}
    if spot_tbl.size == 0:
        spot_tbl = np.zeros((0, len(SPOT_RESIDUAL_COLS)))

    gidx = spot_tbl[:, 0].astype(np.int64)
    ring = spot_tbl[:, 2].astype(np.int64)
    eta = spot_tbl[:, 3]
    dy, dz = spot_tbl[:, 4], spot_tbl[:, 5]
    drad, dtan = spot_tbl[:, 6], spot_tbl[:, 7]
    dome = spot_tbl[:, 8]
    iang = spot_tbl[:, 9]
    r_exp = spot_tbl[:, 10]

    # ---- per-grain aggregates ------------------------------------------
    per = {
        "grain_med_dy_um": dy,
        "grain_med_dz_um": dz,
        "grain_med_drad_um": drad,
        "grain_med_dtan_um": dtan,
        "grain_med_dome_deg": dome,
        "grain_med_internal_angle_deg": iang,
    }
    order = np.argsort(gidx, kind="stable")
    g_sorted = gidx[order]
    bounds = np.searchsorted(g_sorted, np.arange(n_grains + 1))
    for key, col in per.items():
        col_sorted = col[order]
        vals = np.full(n_grains, np.nan)
        for gi in range(n_grains):
            lo, hi = bounds[gi], bounds[gi + 1]
            if hi > lo:
                vals[gi] = np.median(col_sorted[lo:hi])
        out[key] = vals
    mad = np.full(n_grains, np.nan)
    dtan_sorted = dtan[order]
    for gi in range(n_grains):
        lo, hi = bounds[gi], bounds[gi + 1]
        if hi > lo:
            mad[gi] = _mad_std(dtan_sorted[lo:hi])
    out["grain_mad_dtan_um"] = mad
    out["grain_n_spots"] = np.diff(bounds).astype(np.int32)

    # ---- per-ring radial profile (calibration signature) ----------------
    rings = np.unique(ring[ring > 0])
    ring_med = np.array([_median_or_nan(drad[ring == rn]) for rn in rings])
    ring_mad = np.array([_mad_std(drad[ring == rn]) for rn in rings])
    ring_ppm = np.array([
        _median_or_nan(1e6 * drad[m] / r_exp[m])
        for m in (ring == rn for rn in rings)
    ])
    out["ring_nr"] = rings.astype(np.int32)
    out["ring_med_drad_um"] = ring_med
    out["ring_mad_drad_um"] = ring_mad
    out["ring_drad_ppm"] = ring_ppm
    out["ring_n_spots"] = np.array(
        [int((ring == rn).sum()) for rn in rings], dtype=np.int64
    )

    # ---- eta profiles (beam-center / tilt / wedge signatures) ------------
    n_bins = max(1, int(round(360.0 / eta_bin_deg)))
    edges = -180.0 + eta_bin_deg * np.arange(n_bins)
    bin_idx = np.clip(
        ((eta + 180.0) // eta_bin_deg).astype(np.int64), 0, n_bins - 1
    )
    out["eta_bin_lo_deg"] = edges
    for key, col in (
        ("eta_med_drad_um", drad),
        ("eta_med_dtan_um", dtan),
        ("eta_med_dome_deg", dome),
    ):
        out[key] = np.array(
            [_median_or_nan(col[bin_idx == b]) for b in range(n_bins)]
        )
    out["eta_n_spots"] = np.array(
        [int((bin_idx == b).sum()) for b in range(n_bins)], dtype=np.int64
    )

    # ---- global scalars ---------------------------------------------------
    out["overall_med_dy_um"] = np.float64(_median_or_nan(dy))
    out["overall_med_dz_um"] = np.float64(_median_or_nan(dz))
    out["overall_med_drad_um"] = np.float64(_median_or_nan(drad))
    out["overall_med_dtan_um"] = np.float64(_median_or_nan(dtan))
    out["overall_med_dome_deg"] = np.float64(_median_or_nan(dome))
    out["overall_mad_drad_um"] = np.float64(_mad_std(drad))
    out["overall_mad_dtan_um"] = np.float64(_mad_std(dtan))
    out["overall_mad_dome_deg"] = np.float64(_mad_std(dome))
    out["overall_med_internal_angle_deg"] = np.float64(_median_or_nan(iang))
    return out


def summarize_residuals(diag: Dict[str, np.ndarray]) -> str:
    """One-paragraph human-readable summary for the pipeline log."""
    lines = [
        "[pg-residuals] signed residual decomposition (obs - fitted-grain):",
        "[pg-residuals]   median dY=%+.1f dZ=%+.1f dRad=%+.1f dTan=%+.1f um, "
        "dOme=%+.4f deg" % (
            diag["overall_med_dy_um"], diag["overall_med_dz_um"],
            diag["overall_med_drad_um"], diag["overall_med_dtan_um"],
            diag["overall_med_dome_deg"],
        ),
        "[pg-residuals]   robust scatter (MAD-std): dRad=%.1f dTan=%.1f um, "
        "dOme=%.3f deg; median internal angle=%.3f deg" % (
            diag["overall_mad_drad_um"], diag["overall_mad_dtan_um"],
            diag["overall_mad_dome_deg"],
            diag["overall_med_internal_angle_deg"],
        ),
    ]
    rn = diag["ring_nr"]
    if rn.size:
        ppm = ", ".join(
            "r%d:%+.0f" % (int(r), p)
            for r, p in zip(rn, diag["ring_drad_ppm"])
        )
        lines.append(
            "[pg-residuals]   per-ring dR/R (ppm): %s" % ppm
        )
        ppm_arr = diag["ring_drad_ppm"]
        finite = ppm_arr[np.isfinite(ppm_arr)]
        if finite.size and abs(np.median(finite)) > 200.0:
            lines.append(
                "[pg-residuals]   NOTE: median dR/R = %+.0f ppm -> reference "
                "lattice / wavelength likely mis-calibrated by that fraction "
                "(shows up as a uniform fake hydrostatic strain)."
                % float(np.median(finite))
            )
    return "\n".join(lines)
