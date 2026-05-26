"""Stage-9 volume consistency: compare ∑V_grain to V_sample.

For a FULL FF-HEDM reconstruction with proper sample volume, the sum of
all per-grain volumes should approximately equal the illuminated sample
volume (modulo detection floor, twin overlap, NNLS deflation).

This module provides:

* :func:`compute_volume_consistency` — total grain volume, sample volume,
  packing fraction (= ΣV_grain / V_sample), per-grain V mean/median,
  fraction of grains in box.

The result is emitted into ``meta.json`` so users see whether their
sample volume is consistent with detected grain population.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Optional, Tuple

import numpy as np


@dataclass
class VolumeConsistencyResult:
    """Output of :func:`compute_volume_consistency`.

    Attributes
    ----------
    n_grains : int
    v_sample_um3 : float | None
        Sample volume in µm³ if known (from ``Vsample`` in paramstest
        or computed from ``Hbeam·π·Rsample²``). ``None`` if neither set.
    sum_v_grain_um3 : float
        ``sum(4/3 · π · R_NNLS³)`` over all grains.
    packing_fraction : float | None
        ``sum_v_grain / v_sample`` (None if v_sample is unknown).
    median_r_um : float
    median_v_um3 : float
    p95_r_um : float
    max_r_um : float
    fraction_in_box : float | None
        Fraction of grain centres inside the sample bounding box. ``None``
        if no box is supplied. Bounding box is centered at origin with
        half-extents derived from the sample dimensions when supplied.
    """

    n_grains: int
    v_sample_um3: Optional[float]
    sum_v_grain_um3: float
    packing_fraction: Optional[float]
    median_r_um: float
    median_v_um3: float
    p95_r_um: float
    max_r_um: float
    fraction_in_box: Optional[float]


def compute_volume_consistency(
    *,
    radius_um: np.ndarray,
    positions_um: Optional[np.ndarray] = None,
    v_sample_um3: Optional[float] = None,
    sample_extents_um: Optional[Tuple[float, float, float]] = None,
) -> VolumeConsistencyResult:
    """Compute Stage-9 volume-consistency diagnostics.

    Parameters
    ----------
    radius_um : (n_grains,) float
        Per-grain ``GrainRadius_NNLS`` in µm.
    positions_um : (n_grains, 3) float, optional
        Per-grain (X, Y, Z) in µm. Needed for ``fraction_in_box``.
    v_sample_um3 : float, optional
        Sample volume (if known). If None and ``sample_extents_um`` given,
        derived as the product.
    sample_extents_um : (Lx, Ly, Lz), optional
        Sample bounding-box edge lengths. Used both for v_sample
        (= product) and ``fraction_in_box``.
    """
    R = np.asarray(radius_um, dtype=np.float64)
    R = R[np.isfinite(R) & (R > 0)]
    n_grains = int(R.size)
    if n_grains == 0:
        return VolumeConsistencyResult(
            n_grains=0, v_sample_um3=v_sample_um3, sum_v_grain_um3=0.0,
            packing_fraction=None, median_r_um=0.0, median_v_um3=0.0,
            p95_r_um=0.0, max_r_um=0.0, fraction_in_box=None,
        )

    V = (4.0 / 3.0) * np.pi * R ** 3
    sum_v = float(V.sum())

    if v_sample_um3 is None and sample_extents_um is not None:
        v_sample_um3 = float(np.prod(sample_extents_um))

    packing = (sum_v / v_sample_um3) if v_sample_um3 else None

    in_box = None
    if sample_extents_um is not None and positions_um is not None:
        hx, hy, hz = (s / 2 for s in sample_extents_um)
        pos = np.asarray(positions_um, dtype=np.float64)
        ok = (
            (np.abs(pos[:, 0]) <= hx)
            & (np.abs(pos[:, 1]) <= hy)
            & (np.abs(pos[:, 2]) <= hz)
        )
        in_box = float(ok.mean())

    return VolumeConsistencyResult(
        n_grains=n_grains,
        v_sample_um3=v_sample_um3,
        sum_v_grain_um3=sum_v,
        packing_fraction=packing,
        median_r_um=float(np.median(R)),
        median_v_um3=float(np.median(V)),
        p95_r_um=float(np.percentile(R, 95)),
        max_r_um=float(R.max()),
        fraction_in_box=in_box,
    )


def consistency_as_meta_dict(res: VolumeConsistencyResult) -> dict:
    """Convert a :class:`VolumeConsistencyResult` into a meta.json-friendly dict."""
    return {
        "n_grains":            res.n_grains,
        "v_sample_um3":        res.v_sample_um3,
        "sum_v_grain_um3":     res.sum_v_grain_um3,
        "packing_fraction":    res.packing_fraction,
        "median_r_um":         res.median_r_um,
        "median_v_um3":        res.median_v_um3,
        "p95_r_um":            res.p95_r_um,
        "max_r_um":            res.max_r_um,
        "fraction_in_box":     res.fraction_in_box,
    }
