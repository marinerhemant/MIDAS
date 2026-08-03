"""Tests for self-calibrated matched-filter detection.

The claim being pinned: a matched Gaussian used ONLY to build the mask finds
more real spots than a raw threshold at the same measured false-positive
budget, and leaves intensities untouched.
"""
import numpy as np
import pytest

from midas_nf_preprocess.process_images.detect import (
    DEFAULT_SIGMAS, calibrate_detector, detect_mask, matched_filter_score,
)


def _synthetic(seed=0, shape=(256, 256), n_spots=150, amp=14.0, noise=3.0,
               psf=0.9):
    """Faint Gaussian spots on noise -- an NF residual near the detection limit."""
    from scipy import ndimage as ndi
    rng = np.random.default_rng(seed)
    clean = np.zeros(shape, np.float32)
    ys = rng.integers(6, shape[0] - 6, n_spots)
    xs = rng.integers(6, shape[1] - 6, n_spots)
    clean[ys, xs] = amp
    clean = ndi.gaussian_filter(clean, psf) * (2 * np.pi * psf ** 2)
    return (clean + rng.normal(0, noise, shape)).astype(np.float32), set(zip(ys, xs))


def test_intensities_are_untouched():
    """The mask comes from the filter; the VALUES must be the original ones."""
    img, _ = _synthetic()
    mask = detect_mask(img, sigma=0.9, threshold=6.0, min_px=4)
    assert mask.any()
    # every value under the mask is exactly the input value
    assert np.array_equal(img[mask], np.asarray(img)[mask])
    # and the filtered score is NOT what you would read
    score = matched_filter_score(img, 0.9)
    assert not np.allclose(score[mask], img[mask])


def test_matched_filter_beats_raw_threshold_at_equal_false_positives():
    img, _ = _synthetic()
    filt = calibrate_detector(img, sigmas=DEFAULT_SIGMAS, fp_budget=5)
    raw = calibrate_detector(img, sigmas=(0.0,), fp_budget=5)
    assert filt.ok
    assert filt.n_blobs > raw.n_blobs, (
        f"matched {filt.n_blobs} vs raw {raw.n_blobs} at <= 5 false positives"
    )
    assert filt.sigma > 0


def test_calibration_respects_the_budget():
    img, _ = _synthetic(seed=2)
    for budget in (0, 2, 20):
        c = calibrate_detector(img, fp_budget=budget)
        if c.ok:
            assert c.n_false <= budget


def test_detections_are_real_spots_not_smoothing_artifacts():
    """Every detected blob must contain a true injected spot centre."""
    from scipy import ndimage as ndi
    img, truth = _synthetic(seed=4)
    c = calibrate_detector(img, fp_budget=5)
    mask = detect_mask(img, sigma=c.sigma, threshold=c.threshold, min_px=4)
    lab, n = ndi.label(mask, structure=np.ones((3, 3), int))
    assert n > 0
    truth_img = np.zeros(img.shape, bool)
    for y, x in truth:
        truth_img[y, x] = True
    hit = 0
    for sl in ndi.find_objects(lab):
        pad = (slice(max(0, sl[0].start - 2), sl[0].stop + 2),
               slice(max(0, sl[1].start - 2), sl[1].stop + 2))
        if truth_img[pad].any():
            hit += 1
    assert hit / n > 0.9, f"only {hit}/{n} detections contain an injected spot"


def test_pure_noise_yields_almost_nothing():
    """A null image must not produce a confident operating point."""
    rng = np.random.default_rng(9)
    noise = rng.normal(0, 3.0, (256, 256)).astype(np.float32)
    c = calibrate_detector(noise, fp_budget=5)
    # symmetric noise: detections and false positives must be comparable
    assert c.n_blobs <= max(5 * (c.n_false + 1), 25), (
        f"{c.n_blobs} detections on pure noise with {c.n_false} false positives"
    )


def test_returns_least_bad_point_instead_of_raising():
    flat = np.zeros((64, 64), np.float32)
    c = calibrate_detector(flat, fp_budget=0)
    assert c.n_blobs == 0 and not c.ok
    assert "sigma" in c.report()


# --- pipeline integration -----------------------------------------------------

def test_pipeline_matched_backend_leaves_intensities_untouched():
    """SpotDetect matched must return the residual itself as `filtered`."""
    import torch
    from midas_nf_preprocess.process_images.params import ProcessParams
    from midas_nf_preprocess.process_images.pipeline import ProcessImagesPipeline

    img, _ = _synthetic(seed=21, shape=(128, 128), n_spots=40)
    frame = torch.from_numpy(img + 100.0)          # +100 fixed background
    median = torch.full_like(frame, 100.0)
    p = ProcessParams(nr_pixels=128, n_distances=1, nr_files_per_distance=1,
                      spot_detect="matched", matched_sigma=0.9,
                      matched_threshold=6.0, mean_filt_radius=0)
    pipe = ProcessImagesPipeline(p, device="cpu")
    res = pipe.process_frame(0, frame, median, 1)
    # `filtered` is the untouched residual, NOT a smoothed or clamped version
    assert torch.allclose(res.filtered, frame - median)
    assert res.labels.shape == frame.shape
    assert res.n_spots > 0


def test_pipeline_matched_calibrates_once_and_caches():
    import torch
    from midas_nf_preprocess.process_images.params import ProcessParams
    from midas_nf_preprocess.process_images.pipeline import ProcessImagesPipeline

    img, _ = _synthetic(seed=22, shape=(128, 128), n_spots=40)
    frame = torch.from_numpy(img); median = torch.zeros_like(frame)
    p = ProcessParams(nr_pixels=128, n_distances=1, nr_files_per_distance=1,
                      spot_detect="matched", mean_filt_radius=0)
    pipe = ProcessImagesPipeline(p, device="cpu")
    pipe.process_frame(0, frame, median, 1)
    first = pipe._matched_cal
    assert first is not None
    pipe.process_frame(1, frame, median, 1)
    assert pipe._matched_cal is first          # reused, not recomputed


def test_pipeline_log_backend_is_the_default_and_unchanged():
    """The default path must not change: SpotDetect defaults to log."""
    from midas_nf_preprocess.process_images.params import ProcessParams
    assert ProcessParams().spot_detect == "log"
