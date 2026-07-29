"""Tests for detector-fixed artifact screening."""

from __future__ import annotations

import numpy as np

from midas_defect.detector_qc import flag_fixed_pixel_artifacts


def test_fixed_pixel_across_omega_is_flagged():
    """Many features at one pixel spanning wide omega -> artifact (the 1.518 case)."""
    # 8 features all at pixel (355, 831), frames spread across the full rotation
    ids = list(range(1, 9))
    row = [355.0 + 0.5 * (i % 2) for i in range(8)]   # within pixel_tol
    col = [831.0 - 0.5 * (i % 2) for i in range(8)]
    frame = np.linspace(50, 1400, 8).tolist()           # ~340 deg span
    flagged, arts = flag_fixed_pixel_artifacts(ids, row, col, frame)
    assert set(flagged) == set(ids)
    assert len(arts) == 1
    assert arts[0].omega_span_deg > 300
    assert abs(arts[0].row - 355) < 2 and abs(arts[0].col - 831) < 2


def test_real_reflections_not_flagged():
    """Features at varying (row,col), each its own omega -> not flagged."""
    ids = list(range(1, 9))
    rng = np.random.default_rng(0)
    row = rng.uniform(200, 1400, 8).tolist()
    col = rng.uniform(200, 1300, 8).tolist()
    frame = rng.uniform(0, 1440, 8).tolist()
    flagged, arts = flag_fixed_pixel_artifacts(ids, row, col, frame)
    assert flagged == []
    assert arts == []


def test_single_broad_spot_not_flagged():
    """A genuine mosaic spot: same pixel but few entries and narrow omega."""
    ids = [10, 11]
    row = [500.0, 501.0]
    col = [600.0, 600.5]
    frame = [700.0, 705.0]   # ~1.25 deg
    flagged, arts = flag_fixed_pixel_artifacts(ids, row, col, frame)
    assert flagged == []


def test_length_mismatch_raises():
    import pytest

    with pytest.raises(ValueError):
        flag_fixed_pixel_artifacts([1, 2], [1.0], [1.0, 2.0], [1.0, 2.0])
