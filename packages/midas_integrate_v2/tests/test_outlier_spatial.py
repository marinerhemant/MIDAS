"""Item 5 — spatial zinger filter (sibling of reject_cosmic_rays)."""
from __future__ import annotations

import numpy as np
import pytest

from midas_integrate_v2.streaming import reject_spatial_spikes


def test_spatial_spikes_detected_and_replaced():
    rng = np.random.default_rng(0)
    img = rng.normal(100.0, 5.0, size=(64, 64))
    # Plant 5 isolated spikes 50× background
    spike_positions = [(8, 8), (16, 32), (40, 40), (50, 12), (30, 55)]
    for (i, j) in spike_positions:
        img[i, j] = 5_000.0
    cleaned, mask = reject_spatial_spikes(img, n_sigma=5.0, method="laplacian")
    for (i, j) in spike_positions:
        assert mask[i, j], f"spike at ({i}, {j}) not flagged"
        assert cleaned[i, j] < 1_000.0
    # No bulk over-detection: each spike of 5000 over ~100 lifts its
    # 4 Laplacian neighbours too, so ~25 hits is expected; reject
    # anything > 50 as clear over-detection.
    assert mask.sum() < 50


def test_spatial_spikes_smooth_image_no_false_positives():
    """A noisy smooth Gaussian peak should produce essentially no
    spike detections — LoG response is dominated by photon noise, not
    by sharp pixel-scale events."""
    rng = np.random.default_rng(2)
    yy, xx = np.mgrid[:64, :64]
    base = 100.0 + 50.0 * np.exp(-((yy - 32) ** 2 + (xx - 32) ** 2) / 200.0)
    img = base + rng.normal(0.0, np.sqrt(base))
    _, mask = reject_spatial_spikes(img, n_sigma=8.0, method="laplacian")
    # < 5% of pixels detected; the dominant signal is noise, not spikes.
    assert mask.sum() < 0.05 * img.size


def test_spatial_spikes_median_method():
    rng = np.random.default_rng(1)
    img = rng.normal(100.0, 5.0, size=(48, 48))
    img[10, 10] = 10_000.0
    cleaned, mask = reject_spatial_spikes(
        img, n_sigma=5.0, method="median", kernel_size=3,
    )
    assert mask[10, 10]
    assert cleaned[10, 10] < 1_000.0


def test_spatial_spikes_flag_only():
    img = np.full((16, 16), 50.0)
    img[5, 5] = 5_000.0
    out, mask = reject_spatial_spikes(img, mode="flag_only")
    assert out[5, 5] == 5_000.0  # untouched
    assert mask[5, 5]


def test_spatial_spikes_replace_with_nan():
    img = np.full((16, 16), 50.0)
    img[5, 5] = 5_000.0
    out, mask = reject_spatial_spikes(img, mode="replace_with_nan")
    assert np.isnan(out[5, 5])
