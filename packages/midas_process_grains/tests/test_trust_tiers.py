"""Stage-4 tests: tier assignment from physics-grounded primitives."""
from __future__ import annotations

import numpy as np

from midas_process_grains.compute.trust_tiers import (
    assign_tiers, tier_summary, SCHEMES, TIER_GOLD, TIER_SILVER, TIER_BRONZE,
)


def test_strict_scheme_requires_clean_cluster_for_gold():
    cov = np.array([1.0, 0.9, 0.7, 0.5, 0.85])
    dup = np.array([0, 0, 0, 0, 1])           # last grain had pre-split dup
    splits = np.array([0, 0, 0, 0, 1])        # last grain emerged from split
    out = assign_tiers(cov, dup, splits, scheme="strict")
    assert out[0] == TIER_GOLD       # cov=1.0, clean → gold
    assert out[1] == TIER_GOLD       # cov=0.9, clean → gold
    assert out[2] == TIER_SILVER     # cov=0.7 (≥0.6), clean → silver
    assert out[3] == TIER_BRONZE     # cov=0.5 → bronze
    assert out[4] == TIER_SILVER     # cov=0.85 ≥ 0.8 BUT not clean → demoted to silver


def test_loose_scheme():
    cov = np.array([0.65, 0.45, 0.35])
    dup = np.array([0, 0, 0])
    splits = np.array([0, 0, 0])
    out = assign_tiers(cov, dup, splits, scheme="loose")
    assert out[0] == TIER_GOLD       # 0.65 ≥ 0.60 → gold
    assert out[1] == TIER_SILVER     # 0.45 ≥ 0.40 → silver
    assert out[2] == TIER_BRONZE


def test_coverage_only_ignores_clean_flag():
    cov = np.array([0.7, 0.5])
    dup = np.array([5, 0])
    splits = np.array([3, 0])
    out = assign_tiers(cov, dup, splits, scheme="coverage_only")
    assert out[0] == TIER_GOLD       # 0.7 ≥ 0.66, clean flag ignored → gold
    assert out[1] == TIER_SILVER     # 0.5 ≥ 0.33 → silver


def test_nan_coverage_treated_as_bronze():
    cov = np.array([np.nan, np.nan, 0.9])
    dup = np.zeros(3, dtype=np.int32)
    splits = np.zeros(3, dtype=np.int32)
    out = assign_tiers(cov, dup, splits, scheme="strict")
    assert out[0] == TIER_BRONZE
    assert out[1] == TIER_BRONZE
    assert out[2] == TIER_GOLD


def test_tier_summary_counts_match_input():
    tiers = np.array([2, 2, 1, 0, 0, 0, 2])
    s = tier_summary(tiers)
    assert s["n_total"] == 7
    assert s["n_gold"] == 3
    assert s["n_silver"] == 1
    assert s["n_bronze"] == 3


def test_tier_summary_handles_empty():
    s = tier_summary(np.array([], dtype=np.int8))
    assert s["n_total"] == 0
    assert s["gold"] == 0.0


def test_all_schemes_known():
    assert set(SCHEMES.keys()) >= {"strict", "loose", "coverage_only"}
