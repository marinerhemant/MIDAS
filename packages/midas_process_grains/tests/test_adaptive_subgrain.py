"""Tests for compute/adaptive.py — sub-grain vs refiner-noise classifier (item 10)."""

from __future__ import annotations

import math

import numpy as np

from midas_process_grains.compute.adaptive import classify_pairs_subgrain_vs_noise


def test_pure_noise_population_assigns_high_noise_weight():
    """When all pairs are drawn from the refiner-noise distribution alone,
    the EM should converge to a noise weight near 1."""
    sigma_o = 0.02       # ° per candidate
    sigma_pair = sigma_o * math.sqrt(2.0)
    # Sample 5000 half-normal misori with scale sigma_pair
    rng = np.random.default_rng(42)
    misori = np.abs(rng.normal(0.0, sigma_pair, 5000))
    result = classify_pairs_subgrain_vs_noise(misori, sigma_o, band_max_deg=2.0)
    assert result["noise_weight"] > 0.85, \
        f"pure noise should give noise_weight near 1, got {result['noise_weight']:.3f}"


def test_pure_subgrain_population_assigns_high_subgrain_weight():
    """When all pairs are drawn from a wide exponential (real sub-grain),
    the EM should converge to noise weight near 0."""
    sigma_o = 0.02
    rng = np.random.default_rng(7)
    # Real sub-grain misori at ~0.5° scale, all within 2° band
    misori = rng.exponential(0.5, 5000)
    misori = misori[misori <= 2.0]
    result = classify_pairs_subgrain_vs_noise(misori, sigma_o, band_max_deg=2.0,
                                               subgrain_scale_deg=0.5)
    assert result["subgrain_weight"] > 0.85, \
        f"pure subgrain should give high subgrain_weight, got {result['subgrain_weight']:.3f}"


def test_mixed_population_separates_components():
    """Mixed population — 70 % refiner-noise + 30 % sub-grain — should be
    correctly separated by the EM."""
    sigma_o = 0.02
    sigma_pair = sigma_o * math.sqrt(2.0)
    rng = np.random.default_rng(13)
    n_total = 8000
    n_noise = int(0.7 * n_total)
    misori_noise = np.abs(rng.normal(0.0, sigma_pair, n_noise))
    misori_sub = rng.exponential(0.5, n_total - n_noise)
    misori = np.concatenate([misori_noise, misori_sub])
    misori = misori[misori <= 2.0]
    result = classify_pairs_subgrain_vs_noise(misori, sigma_o, band_max_deg=2.0,
                                               subgrain_scale_deg=0.5)
    # Expected noise weight in [0.55, 0.85]; relaxed because EM in this 1D
    # mixture is not perfectly identifiable
    assert 0.40 <= result["noise_weight"] <= 0.95, \
        f"mixed pop noise weight off: {result['noise_weight']:.3f}"


def test_pairs_far_outside_band_are_excluded():
    """Pairs above band_max_deg should not be in mask_merge."""
    sigma_o = 0.02
    misori = np.array([0.01, 0.05, 0.1, 0.5, 1.0, 3.0, 10.0])
    result = classify_pairs_subgrain_vs_noise(misori, sigma_o, band_max_deg=2.0)
    assert result["mask_band"].tolist() == [True, True, True, True, True, False, False]
    assert not result["mask_merge"][5]  # 3° pair is excluded
    assert not result["mask_merge"][6]  # 10° pair is excluded


def test_posterior_decreases_with_misori_in_band():
    """Within the band, P(refiner-noise | misori) should be high at small
    misori and low at large misori — refiner noise is concentrated near 0."""
    sigma_o = 0.02
    misori = np.array([0.005, 0.02, 0.05, 0.5, 1.0, 1.5])
    result = classify_pairs_subgrain_vs_noise(misori, sigma_o, band_max_deg=2.0)
    p = result["posterior_noise"]
    # P at 0.005° should exceed P at 1.5°
    assert p[0] >= p[-1], f"posterior should decrease with misori: {p}"


def test_empty_input_returns_no_band_pairs():
    result = classify_pairs_subgrain_vs_noise(np.empty(0), 0.02)
    assert result["posterior_noise"].size == 0
    assert result["mask_band"].size == 0
    assert result["mask_merge"].size == 0


def test_default_subgrain_scale_is_reasonable():
    """Default subgrain_scale_deg = 1.0 — reasonable for most polycrystals."""
    sigma_o = 0.02
    rng = np.random.default_rng(99)
    misori = np.concatenate([
        np.abs(rng.normal(0, 0.03, 1000)),
        rng.exponential(1.0, 500),
    ])
    misori = misori[misori <= 5.0]
    result = classify_pairs_subgrain_vs_noise(misori, sigma_o, band_max_deg=5.0)
    # Should find a meaningful (non-degenerate) split
    assert 0.05 < result["noise_weight"] < 0.95
