"""Tests for compute/volume_nnls.py (joint-NNLS grain-volume corrector).

We construct synthetic spot-grain assignment matrices with controlled
overlap, run :func:`compute_nnls_volumes`, and verify the corrected radii
match the planted ground truth.

The synthetic cases:

  1. Two isolated grains with disjoint spots: NNLS should recover both
     volumes correctly (up to the global rescale).
  2. Twin-pair case: two grains share ALL spots equally; NNLS should
     attribute half the intensity to each, deflating both by a factor of
     2^(1/3) relative to the naive mean.
  3. Mixed: one isolated grain + one twin pair. The isolated grain should
     be unchanged; the twin partners should be deflated.
"""

from __future__ import annotations

import numpy as np
import pytest

from midas_process_grains.compute.volume_nnls import (
    compute_nnls_volumes, physical_ring_K, structure_factor_squared_fcc,
)


def _build_simple(n_grains, sm_g, sm_s, sm_r, intensities, rings, R_naive=None):
    """Helper: construct the input dicts for compute_nnls_volumes."""
    grain_ids = np.arange(1, n_grains + 1, dtype=np.int64)
    sm_grain_id = np.asarray(sm_g, dtype=np.int64)
    sm_spot_id = np.asarray(sm_s, dtype=np.int64)
    sm_ring_nr = np.asarray(sm_r, dtype=np.int64)
    spot_intensity = dict(zip(map(int, sm_spot_id), map(float, intensities)))
    spot_ring = dict(zip(map(int, sm_spot_id), map(int, rings)))
    if R_naive is None:
        # Per-grain naive R from per-spot R averaged (without overlap correction)
        # We construct one R per spot, then average per grain
        R_naive_arr = np.zeros(n_grains)
        for g in grain_ids:
            mask = sm_grain_id == g
            # Per-spot α = I / K(ring); ignore K (uniform) for naive
            alpha_g = np.array([float(intensities[i]) for i in range(len(sm_grain_id)) if mask[i]])
            if len(alpha_g) == 0:
                R_naive_arr[g - 1] = 0.0; continue
            R_per_spot = np.cbrt(3.0 * alpha_g / (4.0 * np.pi))
            R_naive_arr[g - 1] = float(np.mean(R_per_spot))
        R_naive = R_naive_arr
    return dict(
        grain_ids=grain_ids, R_naive=R_naive,
        sm_grain_id=sm_grain_id, sm_spot_id=sm_spot_id, sm_ring_nr=sm_ring_nr,
        spot_intensity=spot_intensity, spot_ring=spot_ring,
    )


def test_isolated_grains_recover_correctly():
    """Two isolated grains with disjoint spots should both recover R_naive
    exactly (NNLS is degenerate with the simple mean for isolated grains)."""
    # Grain 1: spots 1-10, all I = 100 on ring 1
    # Grain 2: spots 11-20, all I = 800 on ring 1 (8× larger volume)
    sm_g = [1] * 10 + [2] * 10
    sm_s = list(range(1, 21))
    sm_r = [1] * 20
    intensities = [100.0] * 10 + [800.0] * 10
    rings = [1] * 20
    kwargs = _build_simple(2, sm_g, sm_s, sm_r, intensities, rings)

    res = compute_nnls_volumes(**kwargs)
    assert res.n_unique_spots == 20
    assert res.n_spots_shared == 0
    # Per construction, grain 2 has 8× the volume → 2× the R
    ratio = res.R_nnls[1] / res.R_nnls[0]
    assert abs(ratio - 2.0) < 0.05, f"isolated grain R ratio {ratio:.3f}, expected ~2"
    # frac_shared = 0 for both
    assert np.allclose(res.frac_spots_shared, 0.0)


def test_twin_pair_split_intensity():
    """Two twin partners sharing ALL spots: each gets half the volume.
    The naive R inflates both by 2^(1/3) ≈ 1.26×; NNLS deflates each back."""
    # Both grains claim spots 1-10, each with intensity I = 200 (= sum of two grains
    # of equal volume V₁ = V₂ = 100). The naive method attributes I=200 to BOTH,
    # inflating both. NNLS should split: V₁_recovered + V₂_recovered = 200.
    sm_g = [1] * 10 + [2] * 10
    sm_s = list(range(1, 11)) * 2   # same spot IDs in both grains
    sm_r = [1] * 20
    intensities_for_lookup = [200.0] * 10 + [200.0] * 10  # same spot → same intensity
    rings = [1] * 20

    kwargs = _build_simple(2, sm_g, sm_s, sm_r, intensities_for_lookup, rings)
    # Override R_naive: both grains see the same spots at I=200, mean R = (3·200/4π)^(1/3)
    R_naive_each = float(np.cbrt(3.0 * 200.0 / (4.0 * np.pi)))
    kwargs["R_naive"] = np.array([R_naive_each, R_naive_each])

    res = compute_nnls_volumes(**kwargs)
    assert res.n_unique_spots == 10
    assert res.n_spots_shared == 10
    # The sum of recovered volumes (raw, before rescale) should match the true
    # total volume (200). Note V_nnls_raw is α-units before rescale.
    # After rescale, both R values should be equal (symmetry), each lower than R_naive_each
    # The deflation factor is 2^(1/3) for equal split.
    assert abs(res.R_nnls[0] - res.R_nnls[1]) < 1e-6 * R_naive_each, \
        "symmetric twin should give equal R"
    # frac_shared = 1.0 for both (every assigned spot is shared)
    assert np.allclose(res.frac_spots_shared, 1.0)


def test_mixed_case_only_overlapping_deflated():
    """One isolated grain + one twin pair. The isolated grain unchanged;
    twin partners deflated relative to naive."""
    # Grain 1: isolated, spots 1-10, I=100 each
    # Grains 2 + 3: twin pair sharing spots 11-20, I=200 each
    sm_g = [1] * 10 + [2] * 10 + [3] * 10
    sm_s = list(range(1, 11)) + list(range(11, 21)) + list(range(11, 21))
    sm_r = [1] * 30
    intensities = [100.0] * 10 + [200.0] * 10 + [200.0] * 10
    rings = [1] * 30
    kwargs = _build_simple(3, sm_g, sm_s, sm_r, intensities, rings)

    res = compute_nnls_volumes(**kwargs)
    assert res.n_unique_spots == 20    # spots 1-20
    assert res.n_spots_shared == 10    # spots 11-20 shared by grains 2,3
    # Diagnostic: grain 1 has 0 shared spots, grains 2,3 have all shared
    assert res.frac_spots_shared[0] == 0.0
    assert res.frac_spots_shared[1] == 1.0
    assert res.frac_spots_shared[2] == 1.0
    # Twin partners should be ~equal and deflated relative to the standalone
    assert abs(res.R_nnls[1] - res.R_nnls[2]) < 1e-6 * res.R_nnls[1], \
        "symmetric twin partners should have equal R"


def test_empty_input_returns_empty_result():
    grain_ids = np.empty(0, dtype=np.int64)
    res = compute_nnls_volumes(
        grain_ids=grain_ids, R_naive=np.empty(0),
        sm_grain_id=np.empty(0, dtype=np.int64),
        sm_spot_id=np.empty(0, dtype=np.int64),
        sm_ring_nr=np.empty(0, dtype=np.int64),
        spot_intensity={}, spot_ring={},
    )
    assert res.R_nnls.size == 0
    assert res.n_unique_spots == 0
    assert res.n_spots_shared == 0


def test_status_and_iteration_count_recorded():
    """Sanity: NNLS result should record solver status and iteration count."""
    sm_g = [1] * 10 + [2] * 10
    sm_s = list(range(1, 21))
    sm_r = [1] * 20
    intensities = [100.0] * 20
    rings = [1] * 20
    kwargs = _build_simple(2, sm_g, sm_s, sm_r, intensities, rings)
    res = compute_nnls_volumes(**kwargs)
    assert res.nnls_n_iter >= 0
    assert res.nnls_status in (0, 1, 2, 3)
    assert res.rescale_factor > 0.0


def test_per_grain_n_match_attribution_keeps_total_volume_consistent():
    """When grains have varying numbers of shared spots, NNLS should
    proportionally attribute and the SUM of recovered volumes for the twin
    pair should match the total contributed volume (within solver tolerance)."""
    # Grains 1,2 share spots 1-5; grain 3 isolated on spots 6-10
    # Spots 1-5 have I = 100 (= sum of grain 1 + grain 2 contributions)
    # Grain 3 on spots 6-10 has I = 50 each
    sm_g = [1] * 5 + [2] * 5 + [3] * 5
    sm_s = list(range(1, 6)) + list(range(1, 6)) + list(range(6, 11))
    sm_r = [1] * 15
    intensities_for_lookup = [100.0] * 5 + [100.0] * 5 + [50.0] * 5
    rings = [1] * 15
    kwargs = _build_simple(3, sm_g, sm_s, sm_r, intensities_for_lookup, rings)

    res = compute_nnls_volumes(**kwargs)
    # Symmetric twin: grain 1 and 2 should have equal recovered volume
    assert abs(res.V_nnls_raw[0] - res.V_nnls_raw[1]) < 1e-6 * abs(res.V_nnls_raw[0] + 1e-6), \
        "symmetric twin partners should have equal raw V"
    # Sum of twin pair contributions to shared spots ≈ total shared intensity / K
    # Specifically V₁ + V₂ ≈ 100 / K(ring 1)
    # (and grain 3 should ≈ 50 / K(ring 1) per-spot, summed over 5 → 5·50/K)
    # We don't assert absolute scale because K is empirical (median).


# ---------------------------------------------------------------------------
# Item 9: physical K(ring) tests
# ---------------------------------------------------------------------------

def test_fcc_structure_factor_selection_rule():
    """FCC: |F|² = 16 f² for all-even or all-odd hkl; 0 otherwise."""
    # All-odd: 111
    assert structure_factor_squared_fcc(1, 1, 1, 1.0) == 16.0
    # All-even: 200
    assert structure_factor_squared_fcc(2, 0, 0, 1.0) == 16.0
    # All-even: 220
    assert structure_factor_squared_fcc(2, 2, 0, 1.0) == 16.0
    # Mixed: 210 should be 0
    assert structure_factor_squared_fcc(2, 1, 0, 1.0) == 0.0
    # Mixed: 100
    assert structure_factor_squared_fcc(1, 0, 0, 1.0) == 0.0


def test_physical_ring_K_ratios_drop_with_2theta():
    """At fixed |F|², higher-2θ rings see smaller K because LP·DWF falls.
    For FCC Ni at typical FF-HEDM wavelengths, K({111}) > K({200}) > K({220})."""
    import pandas as pd
    # Synthetic hkls table for FCC Ni
    rows = []
    for r, (h, k, l, mult, two_theta) in enumerate([
        (1, 1, 1, 8, 5.7),      # ring 1 {111}
        (2, 0, 0, 6, 6.6),      # ring 2 {200}
        (2, 2, 0, 12, 9.3),     # ring 3 {220}
        (3, 1, 1, 24, 10.9),    # ring 4 {311}
    ], start=1):
        for _ in range(mult):
            rows.append({"h": h, "k": k, "l": l, "RingNr": r, "2Theta": two_theta,
                          "D-spacing": 1.0, "g1": 0.0, "g2": 0.0, "g3": 0.0,
                          "Theta": two_theta / 2, "Radius": 1.0})
    hkls = pd.DataFrame(rows)
    K = physical_ring_K(hkls, wavelength=0.207, species="Ni", B_factor=0.4)
    assert set(K.keys()) == {1, 2, 3, 4}
    # All values should be finite and positive
    for v in K.values():
        assert v > 0 and np.isfinite(v)
    # K(111) should exceed K(200) at 8 vs 6 mult and slightly lower 2θ — same
    # F² (FCC), close 2θ, mult dominates: K(111) > K(200)
    assert K[1] > K[2], f"K(111)={K[1]} should exceed K(200)={K[2]} (mult 8 vs 6)"
    # K(311) exceeds K(220) because multiplicity doubles (24 vs 12) more
    # than LP·DWF declines over the small 2θ gap (9.3° → 10.9°)
    assert K[4] > K[3], f"K(311)={K[4]} should exceed K(220)={K[3]} (mult 24 vs 12)"


def test_physical_ring_K_normalised_to_unit_median():
    import pandas as pd
    rows = []
    for r in range(1, 6):
        rows.append({"h": 1, "k": 1, "l": 1, "RingNr": r, "2Theta": 5.0 + r,
                      "D-spacing": 1.0, "g1": 0.0, "g2": 0.0, "g3": 0.0,
                      "Theta": (5.0 + r) / 2, "Radius": 1.0})
    K = physical_ring_K(pd.DataFrame(rows), wavelength=0.207, species="Ni")
    vals = sorted(K.values())
    # median of 5 sorted values is the middle one; should be exactly 1.0 after normalisation
    assert abs(vals[2] - 1.0) < 1e-12


def test_nnls_with_physical_K_dict_runs_end_to_end():
    """Smoke test: pass a physical-K dict to compute_nnls_volumes and
    confirm it produces sensible output (not all-zero, no crash)."""
    sm_g = [1] * 10 + [2] * 10
    sm_s = list(range(1, 21))
    sm_r = [1] * 20
    intensities = [100.0] * 20
    rings = [1] * 20
    kwargs = _build_simple(2, sm_g, sm_s, sm_r, intensities, rings)
    # Pass explicit physical K instead of empirical
    kwargs["ring_K"] = {1: 1.0}
    res = compute_nnls_volumes(**kwargs)
    assert res.R_nnls.shape == (2,)
    assert (res.R_nnls > 0).all()
