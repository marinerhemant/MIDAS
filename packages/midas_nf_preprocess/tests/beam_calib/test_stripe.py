"""Direct-beam stripe finder: moving stripe vs a stationary decoy band.

Reproduces the 2021 1-ID DetZBeamPos failure (issue #4): a stationary band
~65 px above the real stripe, brighter than it, was returned as the beam.
The real stripe moves with the detector; the decoy does not.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from midas_nf_preprocess.beam_calib.stripe import find_stripe, find_stripe_scan

NZ, NY = 256, 320
PX_UM = 2.0
ROW0 = 180.0          # true stripe row at the first DetZ position
DECOY_ROW = ROW0 - 65.0
DZ_UM = (0.0, 30.0, 60.0, 90.0)   # stripe moves 0, 15, 30, 45 rows


def _band(row: float, sigma: float, amp: float) -> np.ndarray:
    r = np.arange(NZ, dtype=float)[:, None]
    prof = amp * np.exp(-0.5 * ((r - row) / sigma) ** 2)
    img = np.zeros((NZ, NY))
    img[:, 40:280] = prof
    return img


def _image(stripe_row: float, *, decoy: bool = True, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    img = 10.0 + _band(stripe_row, 1.5, 100.0)
    if decoy:
        img += _band(DECOY_ROW, 4.0, 250.0)   # brighter, broader, stationary
    return img + rng.normal(0.0, 0.5, img.shape)


def _true_rows():
    return [ROW0 + dz / PX_UM for dz in DZ_UM]


def _old_find_stripe_row(image: np.ndarray) -> float:
    """Row centroid as computed by the pre-fix find_stripe (single peak)."""
    prof = image.mean(axis=1).astype(np.float64)
    prof = prof - np.median(prof)
    peak = float(prof.max())
    sel = np.where(prof > 0.05 * peak)[0]
    seg = np.clip(prof[sel], 0, None)
    return float((sel * seg).sum() / seg.sum())


def test_old_algorithm_is_fooled_by_the_decoy():
    # Guard that the synthetic actually reproduces the bug.
    row = _old_find_stripe_row(_image(ROW0))
    assert abs(row - ROW0) > 20.0


def test_scan_tracks_moving_stripe_and_rejects_stationary_band():
    images = [_image(r, seed=i) for i, r in enumerate(_true_rows())]
    fit = find_stripe_scan(images, DZ_UM, px_um=PX_UM)
    got = [f.row_centroid for f in fit.fits]
    np.testing.assert_allclose(got, _true_rows(), atol=0.2)
    assert abs(abs(fit.rows_per_um) - 1.0 / PX_UM) < 0.01
    assert len(fit.stationary_rows) == 1
    assert abs(fit.stationary_rows[0] - DECOY_ROW) < 0.5
    # zbc of the real stripe, not the decoy
    assert abs(fit.fits[0].zbc(NZ) - ((NZ - 1) - ROW0)) < 0.2


def test_scan_sign_is_not_assumed():
    rows = [ROW0 - dz / PX_UM for dz in DZ_UM]   # stripe moves the other way
    images = [_image(r, seed=i) for i, r in enumerate(rows)]
    fit = find_stripe_scan(images, DZ_UM, px_um=PX_UM)
    np.testing.assert_allclose([f.row_centroid for f in fit.fits], rows, atol=0.2)
    assert fit.rows_per_um < 0


def test_scan_refuses_when_nothing_moves():
    images = [_image(ROW0, seed=i) for i in range(len(DZ_UM))]
    with pytest.raises(ValueError, match="0 bands track"):
        find_stripe_scan(images, DZ_UM, px_um=PX_UM)


def test_scan_refuses_too_small_travel():
    images = [_image(ROW0), _image(ROW0 + 1)]
    with pytest.raises(ValueError, match="travel"):
        find_stripe_scan(images, (0.0, 2.0), px_um=PX_UM)


def test_single_image_warns_on_two_bands_and_does_not_blend():
    img = _image(ROW0)
    with pytest.warns(UserWarning, match="2 horizontal bands"):
        s = find_stripe(img)
    # Documented criterion: brightest band, centroid over that band only.
    assert abs(s.row_centroid - DECOY_ROW) < 0.5
    assert len(s.candidate_rows) == 2
    assert any(abs(r - ROW0) < 0.5 for r in s.candidate_rows)


def test_single_image_row_hint_selects_nearest_band():
    s = find_stripe(_image(ROW0), row_hint=ROW0 + 5)
    assert abs(s.row_centroid - ROW0) < 0.2
    assert s.fwhm_rows < 8


def test_single_band_no_warning_matches_old_centroid():
    img = _image(ROW0, decoy=False)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        s = find_stripe(img, px_um=PX_UM)
    assert abs(s.row_centroid - ROW0) < 0.2
    assert abs(s.row_centroid - _old_find_stripe_row(img)) < 0.05
    assert (s.band_lo_col, s.band_hi_col) == (40, 279)
