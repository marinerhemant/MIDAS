"""Pure-Python parts of centre finding and cleanup scoring.

Binary-free: these exercise the scoring/selection logic on synthetic images.
"""

from __future__ import annotations

import numpy as np
import pytest

from midas_tomo.center import find_center, sharpness, shift_values_for_search
from midas_tomo.cleanup import default_cleanup_grid, load_cleanup_grid, ring_metric


# ------------------------------------------------------------------ center
def test_shift_values_for_search_is_symmetric():
    assert shift_values_for_search(2.0, 0.25) == (-2.0, 2.0, 0.25)


@pytest.mark.parametrize("bad", [0, -1])
def test_shift_values_rejects_nonpositive_width(bad):
    with pytest.raises(ValueError, match="half_width"):
        shift_values_for_search(bad)


def _box(x=64, half=8, amp=1.0):
    img = np.zeros((x, x))
    c = x // 2
    img[c - half:c + half, c - half:c + half] = amp
    return img


def test_variance_prefers_the_concentrated_image():
    # Same total mass, one concentrated and one smeared over 4x the area.
    # Variance is the metric that sees this; TV deliberately does not.
    crisp = _box(half=8, amp=1.0)
    smeared = _box(half=16, amp=0.25)
    assert sharpness(crisp) > sharpness(smeared)


def test_tv_is_near_blur_invariant_for_a_monotonic_edge():
    # For a monotonic ramp, integral |grad f| is the total height change
    # regardless of ramp width, so TV barely moves under blur. This is why
    # `tv` is documented as an artefact metric, not a defocus metric -- and
    # asserting it here stops anyone "fixing" it into a focus score.
    sharp = _box(half=12, amp=1.0)
    x = sharp.shape[0]
    c = x // 2
    ramped = np.zeros_like(sharp)
    for i, v in enumerate(np.linspace(0, 1, 5)):        # 4-px soft edge
        half = 12 + 2 - i // 2
        ramped[c - half:c + half, c - half:c + half] = v
    ratio = abs(sharpness(ramped, method="tv") / sharpness(sharp, method="tv"))
    assert 0.5 < ratio < 2.0, f"TV moved by more than 2x under blur (ratio {ratio})"


def test_tv_prefers_the_artefact_free_image():
    # What `tv` is actually for: streaks add variation that is not in the
    # clean image, so the clean one scores higher (less negative).
    rng = np.random.default_rng(0)
    clean = _box(half=12, amp=1.0)
    streaked = clean.copy()
    streaked[::4, :] += 0.3 * rng.standard_normal(streaked[::4, :].shape)
    assert sharpness(clean, method="tv") > sharpness(streaked, method="tv")


def test_sharpness_rejects_unknown_method():
    with pytest.raises(ValueError, match="unknown method"):
        sharpness(np.zeros((8, 8)), method="entropy")


def test_sharpness_handles_all_nan():
    assert sharpness(np.full((8, 8), np.nan)) == float("-inf")


def _cube_peaked_at(best_idx, n_shifts=9, n_slices=2, x=64):
    """Synthetic multi-shift cube whose sharpness peaks at *best_idx*."""
    cube = np.zeros((n_shifts, n_slices, x, x), dtype=np.float32)
    for i in range(n_shifts):
        # Width grows with distance from best_idx -> variance falls.
        half = 4 + 3 * abs(i - best_idx)
        lo, hi = x // 2 - half, x // 2 + half
        cube[i, :, lo:hi, lo:hi] = 1.0 / (2 * half) ** 2
    return cube


def test_find_center_locates_the_peak():
    cube = _cube_peaked_at(6)
    res = find_center(cube, (-2.0, 2.0, 0.5))
    assert res["best_idx"] == 6
    assert res["best_shift"] == pytest.approx(1.0)
    assert res["well_determined"] is True


def test_find_center_flags_a_flat_landscape(caplog):
    # Every shift identical: the criterion cannot separate them, and saying
    # so is more useful than returning an arbitrary argmax silently.
    cube = np.ones((5, 2, 64, 64), dtype=np.float32)
    res = find_center(cube, (-1.0, 1.0, 0.5))
    assert res["well_determined"] is False


def test_find_center_single_shift_is_not_well_determined():
    cube = _cube_peaked_at(0, n_shifts=1)
    res = find_center(cube, (0.0, 0.0, 1.0))
    assert res["best_shift"] == 0.0
    assert res["well_determined"] is False


def test_find_center_rejects_wrong_rank():
    with pytest.raises(ValueError, match="cube must be 4-D"):
        find_center(np.zeros((4, 8, 8)), (0.0, 1.0, 0.5))


@pytest.mark.parametrize("crop", [0.0, 1.5, -0.2])
def test_find_center_rejects_bad_crop(crop):
    with pytest.raises(ValueError, match="crop must be in"):
        find_center(np.zeros((3, 1, 16, 16)), (0.0, 1.0, 0.5), crop=crop)


# ----------------------------------------------------------------- cleanup
def test_default_grid_sizes_are_odd_and_scale_with_width():
    for w in (128, 512, 2048):
        grid = default_cleanup_grid(w)
        assert grid[0]["snr"] == 0.0, "first entry must be the baseline"
        for cfg in grid[1:]:
            assert cfg["la"] % 2 == 1
            assert cfg["sm"] % 2 == 1
            assert cfg["la"] > cfg["sm"]
    assert default_cleanup_grid(2048)[1]["la"] > default_cleanup_grid(128)[1]["la"]


def test_load_cleanup_grid_skips_comments_and_blanks(tmp_path):
    p = tmp_path / "grid.txt"
    p.write_text("# header\n\n3.0 31 11\n1.5 41 15\n\n")
    assert load_cleanup_grid(p) == [
        {"snr": 3.0, "la": 31, "sm": 11},
        {"snr": 1.5, "la": 41, "sm": 15},
    ]


def test_load_cleanup_grid_rejects_empty(tmp_path):
    p = tmp_path / "grid.txt"
    p.write_text("# nothing but comments\n")
    with pytest.raises(ValueError, match="no valid configs"):
        load_cleanup_grid(p)


def test_ring_metric_penalises_rings():
    x = 128
    y, xx = np.indices((x, x))
    r = np.sqrt((y - x / 2) ** 2 + (xx - x / 2) ** 2)
    smooth = np.exp(-((r / 30.0) ** 2))          # smooth radial feature
    ringy = smooth + 0.2 * (np.sin(r * 2.0) > 0.9)  # sharp concentric spikes
    assert ring_metric(ringy) > ring_metric(smooth)


def test_ring_metric_rejects_tiny_images():
    with pytest.raises(ValueError, match="too small"):
        ring_metric(np.zeros((3, 3)))
