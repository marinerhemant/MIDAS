"""Stage 4: physics-grounded trust tier assignment.

Given the per-grain quantities emitted by Stage 3
(:class:`PhysicsClusterResult`), assign a tier label per grain.

Multiple schemes are provided side-by-side so we can compare them
empirically on the three real test datasets. The right scheme is the
one whose ``gold`` tier matches user intuition on each dataset.

All schemes use ONLY columns that are physics-grounded and
dataset-comparable:
    - ``hkl_coverage``      observed / expected unique signed (h,k,l)
    - ``hkl_dup_count``     duplicates BEFORE split (= flag for ambiguity)
    - ``splits_emerged_from`` 0 if grain emerged from a clean cluster,
                                k > 0 if from a split

NO calibration-dependent quantities (``pos_bstd``, ``pos_med``, residual
ratios). These were the failure mode of v3 trust.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

TIER_GOLD = 2
TIER_SILVER = 1
TIER_BRONZE = 0
TIER_NAMES = {2: "gold", 1: "silver", 0: "bronze"}


@dataclass
class TrustTierConfig:
    """Threshold knobs for a tier scheme. All thresholds physics-grounded."""

    name: str
    cov_gold: float           # hkl_coverage ≥ this → gold (clean only)
    cov_silver: float         # hkl_coverage ≥ this → silver
    # If True, gold requires hkl_dup_count == 0 AND splits_emerged_from == 0
    require_clean_for_gold: bool = True
    # Optional σ-based gates (None = ignored). Gold requires σ ≤ sigma_gold_um
    # on ALL three (X, Y, Z); silver requires σ ≤ sigma_silver_um.
    sigma_gold_um:   float | None = None
    sigma_silver_um: float | None = None
    # Optional matched-spot floor: gold requires n_spots ≥ n_spots_gold
    n_spots_gold:   int | None = None
    n_spots_silver: int | None = None


SCHEMES: dict[str, TrustTierConfig] = {
    # Strict: only clean clusters with ≥80% coverage are gold
    "strict": TrustTierConfig(
        name="strict",
        cov_gold=0.80, cov_silver=0.60,
        require_clean_for_gold=True,
    ),
    # Loose: ≥60% coverage clean → gold, ≥40% → silver
    "loose": TrustTierConfig(
        name="loose",
        cov_gold=0.60, cov_silver=0.40,
        require_clean_for_gold=True,
    ),
    # Coverage-only: ignore clean-ness; just use coverage terciles
    "coverage_only": TrustTierConfig(
        name="coverage_only",
        cov_gold=0.66, cov_silver=0.33,
        require_clean_for_gold=False,
    ),
    # Sigma-aware: physics gates + per-grain position σ from
    # midas_propagate. Gold requires high coverage AND σ ≤ 100 µm in all
    # three axes AND ≥ 20 matched spots. Silver relaxes σ to 250 µm and
    # n_spots to 10.
    "sigma_aware": TrustTierConfig(
        name="sigma_aware",
        cov_gold=0.80, cov_silver=0.60,
        require_clean_for_gold=True,
        sigma_gold_um=100.0, sigma_silver_um=250.0,
        n_spots_gold=20, n_spots_silver=10,
    ),
}


def assign_tiers(
    hkl_coverage:        np.ndarray,
    hkl_dup_count:       np.ndarray,
    splits_emerged_from: np.ndarray,
    scheme: str = "strict",
    *,
    sigma_X_um:      np.ndarray | None = None,
    sigma_Y_um:      np.ndarray | None = None,
    sigma_Z_um:      np.ndarray | None = None,
    n_spots_matched: np.ndarray | None = None,
) -> np.ndarray:
    """Return (N,) int8 array of tier labels.

    Values: 2 = gold, 1 = silver, 0 = bronze.

    ``hkl_coverage`` of NaN (grain with no expected count available)
    is treated as bronze. When the scheme defines σ / n_spots gates and
    those arrays are supplied, they are AND-combined with the coverage
    gates: a grain becomes gold only when **all** active gates pass.
    """
    cfg = SCHEMES[scheme]
    N = hkl_coverage.shape[0]
    out = np.full(N, TIER_BRONZE, dtype=np.int8)

    cov = np.where(np.isnan(hkl_coverage), 0.0, hkl_coverage)
    clean_mask = (hkl_dup_count == 0) & (splits_emerged_from == 0)

    if cfg.require_clean_for_gold:
        gold = (cov >= cfg.cov_gold) & clean_mask
    else:
        gold = cov >= cfg.cov_gold
    silver = cov >= cfg.cov_silver

    # σ-based gate
    if cfg.sigma_gold_um is not None and sigma_X_um is not None:
        sx = np.where(np.isnan(sigma_X_um), np.inf, sigma_X_um)
        sy = np.where(np.isnan(sigma_Y_um), np.inf, sigma_Y_um)
        sz = np.where(np.isnan(sigma_Z_um), np.inf, sigma_Z_um)
        sig_gold   = (sx <= cfg.sigma_gold_um)   & (sy <= cfg.sigma_gold_um)   & (sz <= cfg.sigma_gold_um)
        sig_silver = (sx <= cfg.sigma_silver_um) & (sy <= cfg.sigma_silver_um) & (sz <= cfg.sigma_silver_um)
        gold = gold & sig_gold
        silver = silver & sig_silver

    # n_spots gate
    if cfg.n_spots_gold is not None and n_spots_matched is not None:
        gold = gold & (n_spots_matched >= cfg.n_spots_gold)
        silver = silver & (n_spots_matched >= cfg.n_spots_silver)

    silver = silver & ~gold
    out[silver] = TIER_SILVER
    out[gold] = TIER_GOLD
    return out


def tier_summary(tiers: np.ndarray) -> dict:
    """Return a fraction-per-tier dict for quick reporting."""
    n = tiers.shape[0]
    if n == 0:
        return {"gold": 0.0, "silver": 0.0, "bronze": 0.0,
                "n_total": 0, "n_gold": 0, "n_silver": 0, "n_bronze": 0}
    n_g = int((tiers == TIER_GOLD).sum())
    n_s = int((tiers == TIER_SILVER).sum())
    n_b = int((tiers == TIER_BRONZE).sum())
    return {
        "n_total":   n,
        "n_gold":    n_g, "n_silver": n_s, "n_bronze": n_b,
        "gold":      n_g / n, "silver":   n_s / n, "bronze":   n_b / n,
    }
