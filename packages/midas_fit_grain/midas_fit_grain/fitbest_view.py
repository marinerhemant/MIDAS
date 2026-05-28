"""Column-name API + per-grain accessor for ``FitBest.bin``.

The FitBest binary holds per-spot matched data for every refined grain. It's a
``(n_grains, MaxNHKLS=5000, 22)`` float64 sparse tensor — most rows are zero
padding because each grain matches only ~100 spots on average. The 22 columns
mirror ``SpotsComp`` in ``FF_HEDM/src/FitPosOrStrainsOMP.c:505-522`` and the
c-omp ``FitUnified.c:505-522`` (both refiners use the same layout). For
downstream consumers we keep the C convention explicit:

  col  0   SpotID                    — observed SpotID matched at this row
  cols 1-2 obs Y_lab / Z_lab (µm)    — corrected (DetCor + tilt) detector pos
  col  3   obs Omega (deg)
  cols 4-6 sample-frame ĝ            — already ω-rotated; unit vector
  cols 7-8 PRED Y_lab / Z_lab (µm)   — refiner's predicted detector position
  col  9   PRED Omega (deg)          — refiner's predicted ω
  cols 10-12 theor g-vector          — symmetry-aligned canonical hkl direction
  cols 13-15 RAW obs Y/Z/ω           — uncorrected (raw image-frame) values
  cols 16-18 raw obs misc            — pre-correction etas, intensities, etc.
  col  19  minIA                     — internal angle of obs to predicted
  col  20  diffLen (µm)              — ‖obs(Y,Z) − pred(Y,Z)‖
  col  21  diffOme (deg)             — |obs_ω − pred_ω|

The original ``read_fit_best`` returns the bare ``(n_grains, 5000, 22)`` array;
this module wraps it with a typed per-grain ``FitBestGrainView`` so callers
don't repeat the column-index arithmetic everywhere (and so the column
contract is checked once, in one place).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from .io_binary import FIT_BEST_NCOLS, MAX_NHKLS_DEFAULT, read_fit_best


# Column-name table. Keep in sync with FitPosOrStrainsOMP.c:505-522
# and FitUnified.c:505-522 (the c-omp refiner; same layout).
FITBEST_COLS = {
    "spot_id":      0,
    "obs_y_um":     1,
    "obs_z_um":     2,
    "obs_ome_deg":  3,
    "g_obs_x":      4,
    "g_obs_y":      5,
    "g_obs_z":      6,
    "pred_y_um":    7,
    "pred_z_um":    8,
    "pred_ome_deg": 9,
    "g_pred_x":     10,
    "g_pred_y":     11,
    "g_pred_z":     12,
    "raw_y":        13,
    "raw_z":        14,
    "raw_ome":      15,
    "raw_misc_a":   16,
    "raw_misc_b":   17,
    "raw_misc_c":   18,
    "min_ia_deg":   19,
    "diff_len_um":  20,
    "diff_ome_deg": 21,
}


@dataclass
class FitBestGrainView:
    """Per-grain matched-spot view onto a slice of FitBest.bin.

    All arrays are ``(n_spots,)`` covering only the nonzero rows of this
    grain's matched-spot block (the zero-padded tail is dropped).

    Use :func:`grain_view` or :func:`from_array` to construct; this class is
    just a typed container.

    Attributes
    ----------
    grain_idx : int
        Row index in FitBest.bin (the seed index — ID-1 in process_grains' Grains.csv).
    spot_id : ndarray (n_spots,) int64
    obs_y_um, obs_z_um, obs_ome_deg : observed corrected position + ω
    pred_y_um, pred_z_um, pred_ome_deg : refiner-emitted prediction at the refined state
    g_obs_xyz : (n_spots, 3) sample-frame observed ĝ (already ω-rotated)
    diff_len_um, diff_ome_deg : per-spot residual norms the refiner already wrote
    min_ia_deg : per-spot internal-angle metric
    """
    grain_idx: int
    spot_id: np.ndarray
    obs_y_um: np.ndarray
    obs_z_um: np.ndarray
    obs_ome_deg: np.ndarray
    pred_y_um: np.ndarray
    pred_z_um: np.ndarray
    pred_ome_deg: np.ndarray
    g_obs_xyz: np.ndarray
    diff_len_um: np.ndarray
    diff_ome_deg: np.ndarray
    min_ia_deg: np.ndarray

    @property
    def n_spots(self) -> int:
        return int(self.spot_id.size)

    # Convenience — residuals (signed) computed from obs and pred.
    @property
    def dy_um(self) -> np.ndarray:
        return self.obs_y_um - self.pred_y_um

    @property
    def dz_um(self) -> np.ndarray:
        return self.obs_z_um - self.pred_z_um

    @property
    def dome_deg(self) -> np.ndarray:
        # Note: diff_ome_deg in FitBest is |obs - pred|; this returns signed.
        return self.obs_ome_deg - self.pred_ome_deg


def from_array(fitbest_arr: np.ndarray, grain_idx: int) -> FitBestGrainView:
    """Build a :class:`FitBestGrainView` for ``grain_idx`` from a FitBest array
    already in memory.  Drops the zero-padded tail of unwritten spot slots
    (the C code uses SpotID=0 in column 0 as the sentinel for "unused slot",
    consistent with the way SpotsComp is initialised)."""
    if fitbest_arr.ndim != 3 or fitbest_arr.shape[-1] != FIT_BEST_NCOLS:
        raise ValueError(
            f"fitbest_arr must be (n_grains, MaxNHKLS, {FIT_BEST_NCOLS}); "
            f"got {fitbest_arr.shape}"
        )
    if not (0 <= grain_idx < fitbest_arr.shape[0]):
        raise IndexError(f"grain_idx {grain_idx} out of range [0, {fitbest_arr.shape[0]})")
    row = np.asarray(fitbest_arr[grain_idx])
    sid_col = row[:, FITBEST_COLS["spot_id"]]
    nonzero = sid_col != 0
    sub = row[nonzero]
    g_obs = np.column_stack([
        sub[:, FITBEST_COLS["g_obs_x"]],
        sub[:, FITBEST_COLS["g_obs_y"]],
        sub[:, FITBEST_COLS["g_obs_z"]],
    ])
    return FitBestGrainView(
        grain_idx=int(grain_idx),
        spot_id=sub[:, FITBEST_COLS["spot_id"]].astype(np.int64),
        obs_y_um=sub[:, FITBEST_COLS["obs_y_um"]],
        obs_z_um=sub[:, FITBEST_COLS["obs_z_um"]],
        obs_ome_deg=sub[:, FITBEST_COLS["obs_ome_deg"]],
        pred_y_um=sub[:, FITBEST_COLS["pred_y_um"]],
        pred_z_um=sub[:, FITBEST_COLS["pred_z_um"]],
        pred_ome_deg=sub[:, FITBEST_COLS["pred_ome_deg"]],
        g_obs_xyz=g_obs,
        diff_len_um=sub[:, FITBEST_COLS["diff_len_um"]],
        diff_ome_deg=sub[:, FITBEST_COLS["diff_ome_deg"]],
        min_ia_deg=sub[:, FITBEST_COLS["min_ia_deg"]],
    )


def grain_view(
    fitbest_path: str | Path,
    grain_idx: int,
    *,
    n_grains: int,
    max_nhkls: int = MAX_NHKLS_DEFAULT,
) -> FitBestGrainView:
    """Read ``FitBest.bin`` from disk and return one grain's :class:`FitBestGrainView`.

    For repeated access (per-grain audits, UQ, plotting) prefer
    :func:`read_fit_best` once and reuse the array with :func:`from_array`.
    """
    arr = read_fit_best(fitbest_path, n_grains=n_grains, max_nhkls=max_nhkls)
    return from_array(arr, grain_idx)
