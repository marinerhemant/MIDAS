"""Reducing a measured rocking scan: the answer must be recoverable and the checks must fail.

Every test plants a known answer in synthetic frames that carry what broke real reductions --
a detector pedestal holding most of the counts, and a frame order that can be wrong -- and
requires the reduction to recover the answer, or the check to fire when it should.
"""
import math

import numpy as np
import pytest
import torch

from midas_dfxm import (RockingScan, baseline_sensitivity, centroid_uncertainty,
                        check_frame_order, classify_motors, reduce_rocking)
from midas_dfxm.rocking import _window_sigma

pytestmark = pytest.mark.unit


def _planted(M=41, step_mdeg=5.0, H=40, W=40, fwhm_mdeg=20.0, amp=2000.0, f_ped=0.9,
             repeats=1, seed=0, theta0=15.838):
    """Gaussian rocking curves with a planted centre field inside a lit square."""
    rng = np.random.default_rng(seed)
    x = theta0 + np.arange(M) * step_mdeg / 1000.0
    yy, xx = np.mgrid[0:H, 0:W]
    inside = (yy >= 6) & (yy < H - 6) & (xx >= 6) & (xx < W - 6)
    planted_mdeg = 30.0 * (xx / (W - 1) - 0.5) + np.where(yy > H // 2, 10.0, -10.0)
    centre = x[M // 2] + planted_mdeg / 1000.0
    sigma = fwhm_mdeg / 1000.0 / (2.0 * math.sqrt(2.0 * math.log(2.0)))
    signal = amp * inside * np.exp(-0.5 * ((x[:, None, None] - centre[None]) / sigma) ** 2)
    S = signal.sum(0)[inside].mean()
    pedestal = f_ped / (1.0 - f_ped) * S / M
    reps = rng.poisson(signal + pedestal, size=(repeats,) + signal.shape).astype(np.float32)
    return reps, x, planted_mdeg, inside


def _scan(reps, x, **kw):
    halves = None
    if reps.shape[0] >= 2:
        halves = (reps[0::2].mean(0), reps[1::2].mean(0))
    motors = {"theta": x, "two-theta": np.full_like(x, 31.676), "chi": np.zeros_like(x)}
    return RockingScan.from_arrays(reps.mean(0), motors, halves=halves,
                                   n_repeats=reps.shape[0], **kw)


def _slope(maps, planted, inside):
    good = maps.lit & ~maps.truncated & inside & np.isfinite(maps.value)
    a = planted[good] - np.median(planted[good])
    b = maps.value[good] - np.median(maps.value[good])
    return float(np.dot(a, b) / np.dot(a, a)), b - a


@pytest.mark.parametrize("f_ped", [0.5, 0.9, 0.98])
def test_recovers_the_planted_centre_through_the_pedestal(f_ped):
    reps, x, planted, inside = _planted(f_ped=f_ped)
    maps = reduce_rocking(_scan(reps, x))
    slope, resid = _slope(maps, planted, inside)
    assert 0.97 <= slope <= 1.03, slope
    assert np.sqrt(np.mean(resid ** 2)) < 3.0          # mdeg, against a +/-25 mdeg field
    assert abs(maps.pedestal_share - f_ped) < 0.05


def test_the_raw_first_moment_is_diluted_by_the_pedestal():
    """The control the baseline step exists for. It must fail, and by about 1 - f_ped.

    Pixels are chosen on the planted square, never on the raw reduction's own flags: a first
    version selected on them, got an empty set, and "passed" with a NaN slope.
    """
    reps, x, planted, inside = _planted(f_ped=0.98)
    raw = reduce_rocking(_scan(reps, x), baseline="none", window="full", lit=inside)
    sel = inside & np.isfinite(raw.value)
    a = planted[sel] - np.median(planted[sel])
    b = raw.value[sel] - np.median(raw.value[sel])
    slope = float(np.dot(a, b) / np.dot(a, a))
    assert sel.sum() > 100
    assert np.isfinite(slope) and slope < 0.2, slope


def test_frame_order_check_passes_the_right_order_and_fails_shuffles():
    reps, x, _, _ = _planted(f_ped=0.9, seed=1)
    frames = reps.mean(0)
    motors = {"theta": x, "two-theta": np.full_like(x, 31.676)}
    right = check_frame_order(RockingScan.from_arrays(frames, motors))
    assert right.consistent, right.message
    rng = np.random.default_rng(3)
    passed = 0
    for _ in range(10):
        shuffled = frames[rng.permutation(len(x))]
        passed += check_frame_order(RockingScan.from_arrays(shuffled, motors)).consistent
    assert passed <= 1


def test_strain_scale_matches_the_verified_conversion():
    """-38.527 microstrain per mdeg of 2theta at 2theta = 25.5254 deg (KYay Dec-2025 set)."""
    M, H, W = 41, 20, 20
    tth = 25.5254 - 0.040 + np.arange(M) * 0.002
    sigma = 0.004 / 2.3548
    centre = np.full((H, W), 25.5254)
    centre[:, W // 2:] += 0.010                       # +10 mdeg of 2theta
    frames = 5000.0 * np.exp(-0.5 * ((tth[:, None, None] - centre[None]) / sigma) ** 2) + 100.0
    scan = RockingScan.from_arrays(frames, {"tth": tth, "th": tth / 2.0 - 0.1})
    assert scan.scan_type == "strain"
    maps = reduce_rocking(scan, lit=np.ones((H, W), bool), split_half=False)
    per_mdeg = float(np.median(maps.value[:, W // 2:]) - np.median(maps.value[:, :W // 2])) / 10.0
    assert maps.unit == "microstrain"
    assert abs(per_mdeg - (-38.527)) < 0.05, per_mdeg


def test_windowed_poisson_sigma_is_centroid_uncertainty_on_a_full_window():
    rng = np.random.default_rng(0)
    x = np.linspace(-0.1, 0.1, 25)
    s = 3e4 * np.exp(-0.5 * (x[:, None] / 0.02) ** 2) + rng.normal(0, 5, (25, 7))
    b = np.full(7, 800.0)
    com = (s * x[:, None]).sum(0) / s.sum(0)
    ours = _window_sigma(s, b, x, np.ones_like(s, bool), com, gain=2.0, read_var=0.0)
    ref = centroid_uncertainty(torch.tensor(s.T), x, background_per_bin=torch.tensor(b)[:, None],
                               gain=2.0).numpy()
    np.testing.assert_allclose(ours, ref, rtol=1e-10)


def test_repeat_parity_error_bar_is_calibrated():
    reps, x, planted, inside = _planted(f_ped=0.9, repeats=4, amp=600.0, seed=2)
    maps = reduce_rocking(_scan(reps, x))
    assert maps.split == "repeat parity"
    _, resid = _slope(maps, planted, inside)
    ratio = float(np.std(resid) / maps.sigma_global)
    assert 0.7 <= ratio <= 1.4, ratio


def test_angle_parity_error_bar_is_not_optimistic():
    reps, x, planted, inside = _planted(f_ped=0.9, amp=600.0, seed=4)
    maps = reduce_rocking(_scan(reps, x))
    assert maps.split == "angle parity"
    _, resid = _slope(maps, planted, inside)
    ratio = float(np.std(resid) / maps.sigma_global)
    assert 0.5 <= ratio <= 1.4, ratio


def test_baseline_sensitivity_reports_block_level_systematics():
    reps, x, _, _ = _planted(f_ped=0.9)
    rows = baseline_sensitivity(_scan(reps, x), block=4)
    assert len(rows) == 4
    for r in rows:
        assert r["unit"] == "mdeg" and r["n_blocks"] > 10
        assert np.isfinite(r["block_rms"]) and np.isfinite(r["pixel_spread"])
    outside = [r for r in rows if r["variant"]["baseline"] == "outside_peak"]
    assert all(0.97 <= r["block_slope"] <= 1.03 for r in outside)


def test_baseline_sensitivity_sees_a_real_systematic():
    """The block statistic must be able to fail: no baseline at all dilutes every tilt."""
    reps, x, _, _ = _planted(f_ped=0.98)
    (row,) = baseline_sensitivity(_scan(reps, x), block=4,
                                  variants=[dict(baseline="none", window="full")])
    assert np.isfinite(row["block_slope"]) and row["block_slope"] < 0.2, row


def test_classify_motors_reads_what_moves():
    x = np.linspace(8.5, 9.0, 11)
    assert classify_motors({"theta": x, "two-theta": np.full(11, 22.874)})["scan_type"] == "tilt"
    s = classify_motors({"th": x, "tth": 2 * x + 1.0})
    assert s["scan_type"] == "strain" and s["axes"] == ("tth",)
    assert classify_motors({"chi": x, "phi": x[::-1]})["scan_type"] == "tilt2d"
    note = classify_motors({"th": x, "symx": np.linspace(0, 1, 11)})["notes"]
    assert any("translation" in n for n in note)
    with pytest.raises(ValueError, match="no rocking angle"):
        classify_motors({"th": np.full(11, 8.5), "Num": np.arange(11.0)})


def test_motor_table_must_have_one_row_per_frame():
    with pytest.raises(ValueError, match="one value per frame"):
        RockingScan.from_arrays(np.zeros((5, 4, 4)), {"theta": np.arange(6.0)})


def test_repeat_test_does_not_pass_pure_noise_by_construction():
    """Two independent halves of pure noise must look no more alike than a random pairing.

    A first version chose its pixels from the averaged stack, whose selection depends on how
    the halves are paired, and passed pure noise in 300 of 300 runs (verify 6b35197c71b9,
    statistics lens). The selection must not know the pairing.
    """
    rng = np.random.default_rng(0)
    M, H, W = 31, 32, 32
    x = 15.4 + 0.01 * np.arange(M)
    passed = 0
    for k in range(20):
        A = rng.poisson(300.0, (M, H, W)).astype(np.float32)
        B = rng.poisson(300.0, (M, H, W)).astype(np.float32)
        scan = RockingScan.from_arrays((A + B) / 2, {"th": x}, halves=(A, B), n_repeats=2)
        chk = check_frame_order(scan, n_perm=100, seed=k)
        passed += int(chk.repeat_p_value is not None and chk.repeat_p_value <= 0.01)
    assert passed <= 2, passed
