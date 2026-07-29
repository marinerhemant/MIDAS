"""Rev-15 sequence Δ-PDF tests for time-resolved / operando series."""
from __future__ import annotations

import pytest
import torch

from midas_pdf.deltapdf import (
    cluster_significant_regions,
    sequence_delta_pdf,
    significant_features,
)


# ---------------------------------------------------------------------------
# sequence_delta_pdf
# ---------------------------------------------------------------------------

def test_sequence_shape_first_row_zero_by_construction():
    """With baseline = first frame, ΔG[0] = 0 exactly."""
    T, R = 5, 40
    torch.manual_seed(0)
    G = torch.randn(T, R, dtype=torch.float64)
    dG, sd = sequence_delta_pdf(G)
    assert dG.shape == (T, R)
    assert torch.allclose(dG[0], torch.zeros(R, dtype=torch.float64))
    assert torch.all(sd == 0)                            # no sigma passed


def test_sequence_baseline_mean_removes_average():
    G = torch.tensor([[1.0, 2.0, 3.0],
                      [4.0, 5.0, 6.0],
                      [7.0, 8.0, 9.0]], dtype=torch.float64)
    dG, _ = sequence_delta_pdf(G, baseline="mean")
    assert torch.allclose(dG.mean(dim=0),
                          torch.zeros(3, dtype=torch.float64), atol=1e-10)


def test_sequence_baseline_int_selects_row():
    G = torch.arange(9, dtype=torch.float64).reshape(3, 3)
    dG, _ = sequence_delta_pdf(G, baseline=2)
    assert torch.allclose(dG[2], torch.zeros(3, dtype=torch.float64))


def test_sequence_baseline_out_of_range_raises():
    G = torch.zeros(3, 4, dtype=torch.float64)
    with pytest.raises(IndexError, match="baseline"):
        sequence_delta_pdf(G, baseline=99)


def test_sequence_sigma_addition_matches_two_frame_delta():
    """With baseline = frame 0 and independent σ, σ²(ΔG) = σ_t² + σ_0²."""
    T, R = 3, 5
    G = torch.zeros(T, R, dtype=torch.float64)
    sig = torch.tensor([[0.1] * R, [0.2] * R, [0.3] * R], dtype=torch.float64)
    _, sd = sequence_delta_pdf(G, sigma_stack=sig, baseline=0)
    # Row 1: sqrt(0.2² + 0.1²) = sqrt(0.05) ≈ 0.2236
    expected_row1 = (0.2 ** 2 + 0.1 ** 2) ** 0.5
    assert abs(float(sd[1, 0]) - expected_row1) < 1e-10


def test_sequence_sigma_mean_baseline_correct_covariance():
    """For baseline=mean with T frames of equal σ, the correct
    Var(ΔG_t) = σ² (T-1)/T — accounting for frame t being in the mean.

    Sloppy formula σ_t² + Σσ²/T² would give σ²(1 + 1/T) which is too big
    by a factor of (T+1)/(T-1); at T=4 that's 5/3 ≈ 1.67× too large.
    """
    T, R = 4, 3
    G = torch.zeros(T, R, dtype=torch.float64)
    sig = torch.full((T, R), 0.5, dtype=torch.float64)
    _, sd = sequence_delta_pdf(G, sigma_stack=sig, baseline="mean")
    expected = (0.25 * (T - 1) / T) ** 0.5                        # σ²(T-1)/T
    assert torch.allclose(sd, torch.full_like(sd, expected), atol=1e-10)


def test_sequence_sigma_int_baseline_is_zero_at_baseline_frame():
    """ΔG_t = 0 exactly at t = baseline, so σ(ΔG) must be 0 there —
    not sqrt(2)·σ_t (which would spuriously flag features at the baseline)."""
    T, R = 3, 4
    G = torch.zeros(T, R, dtype=torch.float64)
    sig = torch.full((T, R), 0.1, dtype=torch.float64)
    _, sd = sequence_delta_pdf(G, sigma_stack=sig, baseline=1)
    assert torch.allclose(sd[1], torch.zeros(R, dtype=torch.float64))


def test_sequence_rejects_1d_input():
    with pytest.raises(ValueError, match="G_stack"):
        sequence_delta_pdf(torch.arange(5, dtype=torch.float64))


# ---------------------------------------------------------------------------
# significant_features + cluster_significant_regions
# ---------------------------------------------------------------------------

def test_significant_features_returns_correct_shape():
    dG = torch.tensor([[0.1, 0.5, 0.2],
                       [0.3, 0.4, 0.05]], dtype=torch.float64)
    sd = torch.full_like(dG, 0.1)
    mask = significant_features(dG, sd, n_sigma=3.0)
    assert mask.shape == (2, 3)
    assert mask.dtype == torch.bool


def test_significant_features_flags_only_above_threshold():
    dG = torch.tensor([[0.1, 0.5]], dtype=torch.float64)
    sd = torch.tensor([[0.1, 0.1]], dtype=torch.float64)
    # 3σ threshold = 0.3
    mask = significant_features(dG, sd, n_sigma=3.0)
    assert bool(mask[0, 0]) is False and bool(mask[0, 1]) is True


def test_cluster_extracts_intervals_and_drops_singletons():
    r = torch.linspace(1.0, 5.0, 5, dtype=torch.float64)     # 1..5 step 1
    mask = torch.tensor([[True, True, False, True, False],
                         [False, True, True, True, False]], dtype=torch.bool)
    intervals = cluster_significant_regions(mask, r, min_width_points=2)
    assert len(intervals) == 2
    # Row 0: only [1..2] survives (isolated True at 4 dropped)
    assert intervals[0] == [(1.0, 2.0)]
    # Row 1: [2..4]
    assert intervals[1] == [(2.0, 4.0)]


def test_cluster_1d_input_promoted_to_single_frame():
    r = torch.linspace(1.0, 5.0, 5, dtype=torch.float64)
    mask = torch.tensor([True, True, False, False, False], dtype=torch.bool)
    intervals = cluster_significant_regions(mask, r)
    assert len(intervals) == 1
    assert intervals[0] == [(1.0, 2.0)]


def test_operando_end_to_end():
    """Realistic use: two-atom-shell PDF broadens over time; features on the
    two peaks should light up as significant."""
    r = torch.linspace(1.5, 6.0, 100, dtype=torch.float64)
    T = 5
    G_stack = []
    for t in range(T):
        # Gaussian peaks at 2.5 Å and 4.3 Å, second broadens with t
        w = 0.10 + 0.05 * t
        peak1 = torch.exp(-0.5 * ((r - 2.5) / 0.10) ** 2)
        peak2 = torch.exp(-0.5 * ((r - 4.3) / w) ** 2)
        G_stack.append(peak1 + peak2)
    G_stack = torch.stack(G_stack)
    sigma = torch.full_like(G_stack, 0.02)
    dG, sd = sequence_delta_pdf(G_stack, sigma_stack=sigma, baseline=0)
    mask = significant_features(dG, sd, n_sigma=3.0)
    # No significance in first frame (self-baseline)
    assert not mask[0].any()
    # Later frames should have significance mostly in the peak-2 region (r > 3.5 Å)
    late_r_mask = (r > 3.5) & (r < 5.0)
    assert bool(mask[-1][late_r_mask].any())
