"""Front end: raw rotation frames -> spot list, using the production ``midas_defect`` steps.

Nothing is re-implemented: mask, polar-median background and 3-D blob finding are
``midas_defect.ingest``. This wrapper only supplies the (2θ, η) maps from the MIDAS geometry, sizes the
blob threshold from the MEASURED noise instead of a hard-coded count, and drops the large intermediate
stack before returning (a background-subtracted copy of the series is as big as the series).
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import pandas as pd

from midas_defect import ingest

from .geometry import Geometry, spot_directions

__all__ = ["angle_maps", "find_spots"]


def angle_maps(geom: Geometry) -> Tuple[np.ndarray, np.ndarray]:
    """(2θ, η) in degrees for every detector pixel, shape (n_pix_z, n_pix_y)."""
    r, c = np.indices((geom.n_pix_z, geom.n_pix_y), dtype=float)
    _, tth, eta = spot_directions(r.ravel(), c.ravel(), geom)
    return tth.reshape(r.shape), eta.reshape(r.shape)


def find_spots(raw: np.ndarray, geom: Geometry, *, threshold: Optional[float] = None,
               noise_sigmas: float = 7.0, n_sectors: int = 8, low_count_threshold: float = 20.0,
               min_vol: int = 10) -> Tuple[pd.DataFrame, np.ndarray, dict]:
    """Mask, background-subtract and find 3-D blobs.

    ``threshold`` (counts above the polar-median background) defaults to ``noise_sigmas`` times the robust
    noise of the subtracted stack (MAD * 1.4826 of the unmasked voxels), so it follows the data's own
    noise. Returns ``(spots, mask, info)``; ``spots`` carries ``frame/row/col`` (fractional array
    indices) and ``integrated`` (the blob label sum, kept only as the comparison arm: known-limits KL-106).
    """
    mres = ingest.build_mask(raw, low_count_threshold=low_count_threshold)
    tth, eta = angle_maps(geom)
    stack = ingest.subtract_background(raw, tth, eta, mres.mask, n_sectors=n_sectors)
    vals = stack[:, ~mres.mask]
    sample = vals[:: max(1, vals.shape[0] // 40)]            # every ~40th frame: enough for a MAD
    noise = float(1.4826 * np.median(np.abs(sample - np.median(sample))))
    thr = float(noise_sigmas * noise) if threshold is None else float(threshold)
    spots, counts = ingest.find_blobs_3d(stack, mres.mask, threshold=thr, min_vol=min_vol, return_counts=True)
    del stack, vals, sample
    return spots, mres.mask, dict(noise=noise, threshold=thr, blob_counts=counts,
                                  n_masked=int(mres.mask.sum()))
