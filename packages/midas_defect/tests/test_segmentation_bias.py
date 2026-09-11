"""Tests for threshold-segmentation bias diagnostics (manuals/defect/ENVELOPE.md section 16)."""

from __future__ import annotations

import numpy as np
import pytest

from midas_defect.segmentation_bias import (
    label_volume_slope,
    mirrored_dead_masks,
    predicted_volume_ratio,
)


def _gaussian_blob(amplitude, sigma=10.0, half=60):
    ax = np.arange(-half, half + 1, dtype=float)
    z, y, x = np.meshgrid(ax, ax, ax, indexing="ij")
    return amplitude * np.exp(-(x**2 + y**2 + z**2) / (2.0 * sigma**2))


@pytest.mark.parametrize("a_over_t", [2.0, 3.0, 20.0])
def test_slope_matches_the_analytic_gaussian_value(a_over_t):
    """A 3-D Gaussian cut at T is a ball of radius sigma*sqrt(2 ln(A/T)), so s = -1.5 / ln(A/T)."""
    T = 10.0
    out = label_volume_slope(_gaussian_blob(a_over_t * T), T, rel_step=0.1)
    assert out["s"] == pytest.approx(-1.5 / np.log(a_over_t), rel=0.08)
    assert out["abs_s"] == pytest.approx(abs(out["s"]))
    assert out["volume_hi"] < out["volume"] < out["volume_lo"]


def test_weaker_feature_has_the_steeper_slope():
    """The mechanism in one line: the closer the cut sits to a feature's peak, the larger |s|."""
    T = 10.0
    strong = label_volume_slope(_gaussian_blob(20.0 * T), T)
    weak = label_volume_slope(_gaussian_blob(2.0 * T), T)
    assert weak["abs_s"] > 3.0 * strong["abs_s"]


def test_label_volume_ratio_follows_amplitude_to_the_slope():
    """Same shape, amplitudes 1.2 apart, one threshold: volumes differ by ~1.2**|s|, not by 1."""
    T, k = 10.0, 1.2
    b = _gaussian_blob(3.0 * T)
    a = _gaussian_blob(3.0 * k * T)
    observed = np.count_nonzero(a >= T) / np.count_nonzero(b >= T)
    s_b = label_volume_slope(b, T)["s"]
    assert observed > 1.15
    assert observed == pytest.approx(predicted_volume_ratio(k, s_b), rel=0.05)


def test_predicted_volume_ratio_reproduces_the_demk_pair():
    assert predicted_volume_ratio(1.172, -1.24) == pytest.approx(1.2175, abs=1e-3)
    assert predicted_volume_ratio(1.172, 1.24) == predicted_volume_ratio(1.172, -1.24)
    assert predicted_volume_ratio(1.0, 5.0) == 1.0


@pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), float("inf")])
def test_predicted_volume_ratio_rejects_bad_amplitude_ratio(bad):
    with pytest.raises(ValueError):
        predicted_volume_ratio(bad, 1.0)


@pytest.mark.parametrize("threshold,rel_step", [(0.0, 0.1), (-5.0, 0.1), (10.0, 0.0), (10.0, 1.0)])
def test_label_volume_slope_rejects_bad_arguments(threshold, rel_step):
    with pytest.raises(ValueError):
        label_volume_slope(np.ones(10), threshold, rel_step=rel_step)


def test_label_volume_slope_undefined_when_nothing_clears_the_upper_step():
    with pytest.raises(ValueError):
        label_volume_slope(np.full(100, 10.5), 10.0, rel_step=0.1)


def _offsets_along_mirror(masks, ca, cb, key_a, key_b):
    return set((masks[key_a] - ca).tolist()), set((-(masks[key_b] - cb)).tolist())


def test_crossing_pair_censors_the_mirror_of_the_partners_dead_column():
    """A clipped by a dead column 3 px right; B clipped 6 px left. Both must lose both, mirrored."""
    ca, cb = 400, 996
    m = mirrored_dead_masks((500, ca), (500, cb), dead_rows=[], dead_cols=[403, 990], mirror_axis="col")
    off_a, off_b_mirrored = _offsets_along_mirror(m, ca, cb, "cols_a", "cols_b")
    assert off_a == off_b_mirrored
    assert 403 in m["cols_a"] and 990 in m["cols_b"]
    assert ca + 6 in m["cols_a"], "A must also lose the mirror of B's dead column"
    assert cb - 3 in m["cols_b"], "B must also lose the mirror of A's dead column"


def test_crossing_pair_rows_are_not_mirrored():
    m = mirrored_dead_masks((500, 400), (502, 996), dead_rows=[831], dead_cols=[], mirror_axis="col")
    assert set((m["rows_a"] - 500).tolist()) == set((m["rows_b"] - 502).tolist())
    assert 831 in m["rows_a"] and 831 in m["rows_b"]


def test_friedel_pair_mirrors_rows_and_not_columns():
    ra, rb = 900, 727
    m = mirrored_dead_masks((ra, 506), (rb, 507), dead_rows=[831, 847], dead_cols=[493], mirror_axis="row")
    off_a, off_b_mirrored = _offsets_along_mirror(m, ra, rb, "rows_a", "rows_b")
    assert off_a == off_b_mirrored
    assert set((m["cols_a"] - 506).tolist()) == set((m["cols_b"] - 507).tolist())


def test_mirrored_dead_masks_is_symmetric_under_swapping_the_pair():
    kw = dict(dead_rows=[195, 831], dead_cols=[243, 487, 981], mirror_axis="col")
    ab = mirrored_dead_masks((600, 300), (604, 1090), **kw)
    ba = mirrored_dead_masks((604, 1090), (600, 300), **kw)
    for x, y in (("rows_a", "rows_b"), ("cols_a", "cols_b")):
        np.testing.assert_array_equal(ab[x], ba[y])
        np.testing.assert_array_equal(ab[y], ba[x])


def test_mirrored_dead_masks_rejects_bad_arguments():
    with pytest.raises(ValueError):
        mirrored_dead_masks((1, 2), (3, 4), [], [], mirror_axis="diagonal")
    with pytest.raises(ValueError):
        mirrored_dead_masks((1, 2, 3), (3, 4), [], [], mirror_axis="col")
