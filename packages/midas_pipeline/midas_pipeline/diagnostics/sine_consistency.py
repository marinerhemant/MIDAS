"""One grain, one sine: does each find_grains grain trace a single sinusoid in its own sinogram?

A grain at one position (x, y) appears, in reflection row h at rotation omega_h, at scan position
c + a sin(omega_h) + b cos(omega_h). So the centre of mass of every row of a grain's sinogram must fall on
one sine. It does not when find_grains has merged two regions that share an orientation, when a grain is
cut by the scanned field (interior tomography), or when the geometry, the omega sign or the positions
convention is wrong for part of the data. The test needs no reference map and costs a least-squares fit
per grain.

Measured (2026-09-28): ESRF ma5608 alumina Z855 median residual 1.98 scans, 33 of 204 grains > 5 scans;
20-ID-E Fe9Cr pf 0.74 / 0.71 scans, none > 5 (manuals/pf-hedm/LAB_NOTEBOOK.md section 10). The flag is a
pointer to look at the grain's sinogram, not a verdict: a residual can also come from a grain larger than
the field or from few reflections.

Writes ``Output/SineConsistency.csv``: grain, n_rows, resid_median_scans, resid_p90_scans, centre_scan,
amplitude_scans, phase_deg, flagged.
"""

from __future__ import annotations

import glob
import logging
from pathlib import Path
from typing import Dict, Tuple, Union

import numpy as np

LOG = logging.getLogger(__name__)

__all__ = ["fit_one_sine", "sine_consistency"]


def fit_one_sine(sino: np.ndarray, omega_deg: np.ndarray) -> Tuple[Dict, np.ndarray]:
    """Fit c + a sin(w) + b cos(w) to the row centres of mass of one grain's sinogram.

    ``sino`` (n_rows, n_scans), ``omega_deg`` (n_rows,). Rows without signal are ignored. Returns
    (stats, residuals); stats has n_rows, resid_median, resid_p90, centre, amplitude, phase_deg
    (NaN when fewer than 4 rows carry signal)."""
    s = np.asarray(sino, float); w = np.asarray(omega_deg, float).ravel()
    tot = s.sum(1); keep = tot > 0
    nan = {"n_rows": int(keep.sum()), "resid_median": np.nan, "resid_p90": np.nan,
           "centre": np.nan, "amplitude": np.nan, "phase_deg": np.nan}
    if keep.sum() < 4:
        return nan, np.zeros(0)
    x = (s[keep] * np.arange(s.shape[1])).sum(1) / tot[keep]
    t = np.radians(w[keep])
    A = np.c_[np.ones_like(t), np.sin(t), np.cos(t)]
    p, *_ = np.linalg.lstsq(A, x, rcond=None)
    r = x - A @ p
    return ({"n_rows": int(keep.sum()), "resid_median": float(np.median(np.abs(r))),
             "resid_p90": float(np.percentile(np.abs(r), 90)), "centre": float(p[0]),
             "amplitude": float(np.hypot(p[1], p[2])), "phase_deg": float(np.degrees(np.arctan2(p[2], p[1])))}, r)


def sine_consistency(output_dir: Union[str, Path], *, flag_scans: float = 5.0) -> Dict:
    """Run :func:`fit_one_sine` for every grain of ``output_dir``'s sinos_raw / omegas / nrHKLs
    (find_grains output), write SineConsistency.csv, and return a summary."""
    O = Path(output_dir)
    f = glob.glob(str(O / "sinos_raw_*.bin"))
    if not f:
        raise FileNotFoundError(f"no sinos_raw_*.bin in {O}")
    nG, nH, nS = map(int, f[0][:-4].split("_")[-3:])
    sino = np.fromfile(f[0], np.float64).reshape(nG, nH, nS)
    om = np.fromfile(glob.glob(str(O / "omegas_*.bin"))[0], np.float64).reshape(nG, nH)
    nr = np.fromfile(glob.glob(str(O / "nrHKLs_*.bin"))[0], np.int32)
    rows = []
    for g in range(nG):
        st, _ = fit_one_sine(sino[g, :nr[g]], om[g, :nr[g]])
        flag = bool(np.isfinite(st["resid_median"]) and st["resid_median"] > flag_scans)
        rows.append((g, st["n_rows"], st["resid_median"], st["resid_p90"], st["centre"], st["amplitude"], st["phase_deg"], int(flag)))
    arr = np.array(rows, dtype=float)
    np.savetxt(O / "SineConsistency.csv", arr, delimiter=",",
               header="grain,n_rows,resid_median_scans,resid_p90_scans,centre_scan,amplitude_scans,phase_deg,flagged",
               fmt=["%d", "%d", "%.4f", "%.4f", "%.3f", "%.3f", "%.2f", "%d"], comments="")
    med = arr[:, 2][np.isfinite(arr[:, 2])]
    summary = {"n_grains": nG, "median_resid_scans": float(np.median(med)) if med.size else float("nan"),
               "n_flagged": int(arr[:, 7].sum()), "flag_scans": flag_scans,
               "flagged": [int(g) for g in arr[arr[:, 7] > 0, 0]]}
    msg = (f"sine consistency: {nG} grains, median row-COM residual {summary['median_resid_scans']:.2f} scans; "
           f"{summary['n_flagged']} grain(s) > {flag_scans:g} scans")
    if summary["n_flagged"]:
        LOG.warning("%s: %s - look at their sinograms (two regions of one orientation merged, a grain cut by the field, "
                    "or a geometry / omega / positions error). %s", msg, summary["flagged"][:20], O / "SineConsistency.csv")
    else:
        LOG.info("%s. %s", msg, O / "SineConsistency.csv")
    return summary
