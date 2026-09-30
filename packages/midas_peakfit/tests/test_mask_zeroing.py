"""Masked detector pixels are zeroed before the peak search, and peaks cut by the
mask are kept and flagged.

The failure this pins (ESRF ma5608, Eiger 4M): the zip's ``exchange/mask`` marks
the module gaps, whose pixels the detector writes as 65535. The peak search only
used the mask for the ``maskTouched`` flag, so the 65535 pixels went through,
tripped the saturation test (``UpperBoundThreshold`` 65000) and deleted the whole
region -- including every real peak 8-connected to a gap. That was 2.41 % of all
spots on that layer, ~66 regions per frame logged as "saturated", verified by an
identical re-run with the gaps zeroed (0 spots lost, 93 % of the recovered spots
within 3 px of a gap).
"""
from __future__ import annotations

import numpy as np
import pytest

from midas_peakfit.connected import find_regions
from midas_peakfit.orchestrator import _require_ring_radii
from midas_peakfit.params import ZarrParams
from midas_peakfit.preprocess import (
    apply_threshold, correct_frame, mask_touch_map, prepare_mask,
)
from midas_peakfit.seeds import find_regional_maxima

N = 64
GAP = 65535.0
INT_SAT = 65000.0


def _frame_with_peak_next_to_gap():
    """Raw (Z, Y) frame: a vertical gap at column 30, a real peak right beside it."""
    raw = np.zeros((N, N))
    raw[:, 30] = GAP
    zz, yy = np.mgrid[0:N, 0:N]
    raw += 500.0 * np.exp(-((zz - 20) ** 2 + (yy - 32) ** 2) / 4.0)   # 2 px from the gap
    raw[:, 30] = GAP
    mask = np.zeros((N, N))
    mask[:, 30] = 1.0
    return raw, mask


def _search(raw, mask):
    gc = np.full((N, N), 5.0)                     # every pixel in band, threshold 5
    kw = dict(NrPixels=N, NrPixelsY=N, NrPixelsZ=N, transform_options=[],
              dark=np.zeros((N, N)), flood=np.ones((N, N)), good_coords=gc, bc=1.0,
              bad_px_intensity=0.0, make_map=0)
    corr = correct_frame(raw, **kw, mask=prepare_mask(mask, N, N, N, []) if mask is not None else None)
    img = apply_threshold(corr, gc)
    return img, find_regions(img, gc)


def _peak_region(regions):
    """The region holding the real peak's maximum (analysis frame is transposed: row = Y)."""
    for r in regions:
        if np.any((r.pixel_rows == 32) & (r.pixel_cols == 20)):
            return r
    return None


def test_without_the_mask_the_peak_dies_with_the_gap():
    img, regions = _search(*_frame_with_peak_next_to_gap()[:1], None)
    reg = _peak_region(regions)
    assert reg is not None and reg.intensities.max() >= GAP, "the peak merged into the gap region"
    assert find_regional_maxima(reg, img, np.zeros((N, N)), INT_SAT, 10) is None


def test_masked_pixels_are_zeroed_and_the_peak_survives():
    raw, mask = _frame_with_peak_next_to_gap()
    img, regions = _search(raw, mask)
    assert img[:, :].max() < GAP, "a masked pixel reached the peak search"
    reg = _peak_region(regions)
    assert reg is not None
    touch = mask_touch_map(prepare_mask(mask, N, N, N, []))
    out = find_regional_maxima(reg, img, touch, INT_SAT, 10)
    assert out is not None, "the real peak is still deleted"
    assert out[3] == 1, "a peak cut by the gap must be flagged maskTouched"


def test_a_peak_far_from_the_mask_is_not_flagged():
    raw = np.zeros((N, N)); zz, yy = np.mgrid[0:N, 0:N]
    raw += 500.0 * np.exp(-((zz - 20) ** 2 + (yy - 50) ** 2) / 4.0)
    mask = np.zeros((N, N)); mask[:, 10] = 1.0
    img, regions = _search(raw, mask)
    touch = mask_touch_map(prepare_mask(mask, N, N, N, []))
    assert [find_regional_maxima(r, img, touch, INT_SAT, 10)[3] for r in regions] == [0]


def test_unmasked_pixels_are_untouched():
    raw, mask = _frame_with_peak_next_to_gap()
    gc = np.full((N, N), 5.0)
    kw = dict(NrPixels=N, NrPixelsY=N, NrPixelsZ=N, transform_options=[],
              dark=np.zeros((N, N)), flood=np.ones((N, N)), good_coords=gc, bc=1.0,
              bad_px_intensity=0.0, make_map=0)
    with_mask = correct_frame(raw, **kw, mask=prepare_mask(mask, N, N, N, []))
    without = correct_frame(raw, **kw)
    off = (mask == 0).T                            # analysis frame is the transpose
    np.testing.assert_array_equal(with_mask[off], without[off])


def test_mask_touch_map_is_a_one_pixel_dilation():
    m = np.zeros((9, 9)); m[4, 4] = 1
    t = mask_touch_map(m)
    assert t.sum() == 9 and t[3:6, 3:6].all()
    assert mask_touch_map(None) is None
    assert not mask_touch_map(np.zeros((5, 5))).any()


def test_missing_ring_radii_is_an_error_not_an_empty_run():
    p = ZarrParams()
    p.nRingsThresh, p.DoFullImage, p.ResultFolder = 3, 0, "/nowhere"
    with pytest.raises(FileNotFoundError, match="hkls.csv"):
        _require_ring_radii(p, None)
    _require_ring_radii(p, np.array([1.0, 2.0, 3.0]))     # radii present: fine
    p.DoFullImage = 1
    _require_ring_radii(p, None)                          # full image needs none
