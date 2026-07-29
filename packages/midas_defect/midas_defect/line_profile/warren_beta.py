"""Per-grain twin-fault probability proxy from FWHM differential.

Twin faults in FCC selectively broaden the (200) reflection relative to the
(111). A simple proxy used in the literature (Warren-Velterop, eq. 13.x) is

    beta_proxy = FWHM(111) / |G_111| - FWHM(200) / |G_200|

with sign chosen so the proxy increases with twin fault density. The
absolute calibration to a true probability requires a profile-fit, but the
proxy is a clean per-grain rank statistic.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def _find_first(refl_indices: np.ndarray, hkls: np.ndarray, target: tuple[int, int, int]) -> int | None:
    """Return the first index in ``refl_indices`` whose (h, k, l) family matches target."""
    canonical = tuple(sorted([abs(int(x)) for x in target], reverse=True))
    for j, ri in enumerate(refl_indices):
        ck = tuple(sorted([abs(int(x)) for x in hkls[ri]], reverse=True))
        if ck == canonical:
            return j
    return None


def warren_beta_proxy_per_grain(
    per_grain_reflections: list[dict],
    hkls: NDArray[np.intp],
    primary_hkl: tuple[int, int, int] = (1, 1, 1),
    secondary_hkl: tuple[int, int, int] = (2, 0, 0),
) -> NDArray[np.floating]:
    """Twin-fault FWHM-differential proxy per grain.

    Returns NaN if either reference reflection is missing for that grain.
    """
    n_grains = len(per_grain_reflections)
    out = np.full(n_grains, np.nan)
    for gi, entry in enumerate(per_grain_reflections):
        ri = entry["refl_indices"]
        if ri.size == 0:
            continue
        j_p = _find_first(ri, hkls, primary_hkl)
        j_s = _find_first(ri, hkls, secondary_hkl)
        if j_p is None or j_s is None:
            continue
        bp = entry["fwhm"][j_p]
        bs = entry["fwhm"][j_s]
        Gp = entry["G_magnitude"][j_p]
        Gs = entry["G_magnitude"][j_s]
        if not (np.isfinite(bp) and np.isfinite(bs) and Gp > 0 and Gs > 0):
            continue
        out[gi] = float(bp / Gp - bs / Gs)
    return out


__all__ = ["warren_beta_proxy_per_grain"]
