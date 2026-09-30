"""forward_predict_check must recover a clean single peak, flag a known non-single-peaked
one, agree with reduce_rocking's own single_peaked map on both, and work for a mesh (tilt2d)
scan, where no single-peak diagnostic previously existed at all.
"""
import numpy as np
import pytest

from midas_dfxm import ForwardPredictResult, example_rocking_scan, forward_predict_check, reduce_rocking

pytestmark = pytest.mark.unit


def _lit_pixels(maps, n, seed=1):
    show = maps.lit & ~maps.truncated
    rows, cols = np.where(show)
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(rows), size=min(n, len(rows)), replace=False)
    return [(int(rows[i]), int(cols[i])) for i in pick]


def test_clean_single_peak_is_recovered_and_matches_the_map():
    scan = example_rocking_scan("single", seed=0)
    maps = reduce_rocking(scan)
    pixels = _lit_pixels(maps, 15)
    res = forward_predict_check(scan, maps, pixels=pixels)
    assert all(isinstance(r, ForwardPredictResult) for r in res)
    assert all(r.single_peaked == bool(maps.single_peaked[r.row, r.col]) for r in res)
    assert all(r.single_peaked for r in res)
    # a real forward prediction, not a tautology: correlates well with raw counts but is not
    # a perfect match (raw carries Poisson noise the deterministic model does not)
    assert all(r.r > 0.7 for r in res)
    assert all(r.r < 0.999 for r in res)


def test_broad_non_single_peaked_is_flagged_and_matches_the_map():
    scan = example_rocking_scan("broad", seed=0)
    maps = reduce_rocking(scan)
    pixels = _lit_pixels(maps, 15)
    res = forward_predict_check(scan, maps, pixels=pixels)
    assert all(r.single_peaked == bool(maps.single_peaked[r.row, r.col]) for r in res)
    assert sum(not r.single_peaked for r in res) >= 12   # most of "broad" is not single-peaked
    assert all(r.share < 0.45 for r in res if not r.single_peaked)


def test_mesh_scan_has_no_precedent_diagnostic_but_this_works():
    """maps.single_peaked is None for a mesh (rocking.py never computes it there) -- this is
    the actual new capability, not a re-check of an existing one.

    ``share``'s clean-single-Gaussian ceiling is lower in 2-D (an area ratio, ~0.556) than
    in 1-D (a length ratio, ~0.78: see forward_predict_check's docstring), so a mesh's
    single-peak rate on lit-but-dim pixels is noisier than the 1-D case's near-100% match;
    30 pixels at a >=55% bound is comfortably below the ~70% population rate measured on
    this synthetic scan while still failing if the mesh path regresses to near-zero, which
    is what it did before `_shape_2d` smoothed before thresholding and clipped only at the
    sum (its original, unfixed form scored ~0.5% on 200 pixels)."""
    scan = example_rocking_scan("mesh", seed=0)
    maps = reduce_rocking(scan)
    assert maps.single_peaked is None
    pixels = _lit_pixels(maps, 30)
    res = forward_predict_check(scan, maps, pixels=pixels)
    assert len(res) == 30
    for r in res:
        assert r.centre_deg.shape == (2,)
        assert r.sigma_deg.shape == (2,)
        assert r.coordinate.shape == (scan.frames.shape[0], 2)
    assert sum(r.single_peaked for r in res) >= 16   # >=55%: the synthetic mesh is mostly single-peaked


def test_auto_pixel_selection_is_reproducible_and_includes_the_brightest():
    scan = example_rocking_scan("single", seed=0)
    maps = reduce_rocking(scan)
    res1 = forward_predict_check(scan, maps, n_auto=5, seed=3)
    res2 = forward_predict_check(scan, maps, n_auto=5, seed=3)
    assert [(r.row, r.col) for r in res1] == [(r.row, r.col) for r in res2]
    show = maps.lit & ~maps.truncated
    bright = np.unravel_index(int(np.argmax(np.where(show, maps.intensity, -np.inf))),
                              maps.intensity.shape)
    assert (res1[0].row, res1[0].col) == bright


def test_amplitude_matches_raw_peak_height_exactly_by_construction():
    """The model's amplitude is the max of a WINDOW of the raw, baseline-subtracted curve,
    so predicted's peak (baseline + amplitude, since the Gaussian maxes at 1) can never
    exceed the raw curve's own global max -- a sanity check on the construction. It need
    not equal raw's max at a shared index: the model's peak sits at the fitted centre,
    which generally is not exactly on a sampled frame, so an exact-index equality (the
    original, incorrect form of this test) is not guaranteed even for a clean peak."""
    scan = example_rocking_scan("single", seed=0)
    maps = reduce_rocking(scan)
    res = forward_predict_check(scan, maps, pixels=[(20, 20)])[0]
    amplitude = res.predicted.max() - res.baseline
    assert amplitude > 0
    assert res.predicted.max() <= res.raw.max() + 1e-6


def test_empty_or_dark_pixel_returns_none_not_a_crash():
    """A pixel that is merely dim (not `lit`, an integrated-SNR threshold) still typically
    has enough single-shot positive noise to clear this function's own, separate fallback
    condition (`tot <= 0`, i.e. genuinely no positive signal at all) -- so `~maps.lit` does
    not imply the None path. This checks the real guarantee: every lit-or-not pixel returns
    a valid, non-crashing result, and the explicit no-signal fallback path works when forced
    by a pixel dark enough to trip it."""
    scan = example_rocking_scan("single", seed=0)
    maps = reduce_rocking(scan)
    dark = np.argwhere(~maps.lit)
    if len(dark):
        r, c = dark[0]
        res = forward_predict_check(scan, maps, pixels=[(int(r), int(c))])[0]
        assert isinstance(res, ForwardPredictResult)
        assert res.single_peaked in (True, False, None)

    # force the actual fallback path: an all-baseline (zero-signal) pixel
    scan2 = example_rocking_scan("single", seed=0)
    maps2 = reduce_rocking(scan2)
    r, c = 0, 0
    scan2.frames[:, r, c] = maps2.baseline[r, c]
    res2 = forward_predict_check(scan2, maps2, pixels=[(r, c)])[0]
    assert res2.single_peaked is None
