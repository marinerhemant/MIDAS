"""Pre-reduction quality control: every check must fire on the defect it targets, on nothing
else, and must leave the input scan untouched unless :func:`apply_quality_filter` is called
explicitly.

Each synthetic scan is a broad (sigma ~6 px), Poisson-noisy Gaussian peak per point -- broad
enough that a real PSF-like feature and its own shot noise must never trip the spike or
count-anomaly checks (the built-in negative control every test here relies on), with specific
defects then planted at a known (point, repeat) or point so each test can assert on the exact
victim and nobody else.
"""
import math
import os

import numpy as np
import pytest

from midas_dfxm import RockingScan
from midas_dfxm.scan_quality import (RawRepeatScan, apply_quality_filter, assess_scan_quality,
                                     estimate_gain, load_6idc_repeat_frames)

pytestmark = pytest.mark.unit

H = W = 40
N_POINTS = 14
R = 10
PEDESTAL = 300.0
AMPLITUDE = 4000.0
SIGMA_PX = 6.0


def _clean_repeat_scan(seed=0, n_points=N_POINTS, R=R, amplitude=AMPLITUDE, pedestal=PEDESTAL,
                       envelope=True):
    """A broad, well-exposed Poisson-noisy Gaussian peak at every point: nothing should flag.
    ``envelope=True`` gives the amplitude a rocking-curve shape ACROSS points (a real scan rises
    from its edges to a peak; a flat amplitude at every point is the unbracketed-peak signature)."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:H, 0:W]
    r2 = (yy - H / 2) ** 2 + (xx - W / 2) ** 2
    prof = np.exp(-0.5 * r2 / SIGMA_PX ** 2)
    if envelope:
        c = (n_points - 1) / 2.0
        env = 0.05 + 0.95 * np.exp(-0.5 * ((np.arange(n_points) - c) / (n_points / 6.0)) ** 2)
    else:
        env = np.ones(n_points)
    lam = pedestal + amplitude * env[:, None, None] * prof[None]
    frames = rng.poisson(lam[:, None], size=(n_points, R, H, W)).astype(np.float32)
    x = 15.700 + 0.005 * np.arange(n_points)
    motors = {"th": x, "tth": np.full(n_points, 25.5254)}
    return RawRepeatScan.from_arrays(frames, motors, source="synthetic clean scan")


def _bg_pixel():
    """A pixel far from the peak (row/col 0), where the local level is just the pedestal."""
    return 2, 2


# ----------------------------------------------------------------- rocking-curve scan (bug 3)
H2 = W2 = 60
SIGMA_PX2 = 6.0


def _rocking_curve_scan(seed=0, n_points=41, R=10, peak_point=20, fwhm_points=2.5,
                        amplitude=6000.0, pedestal=PEDESTAL, cutoff_sigma=2.0):
    """A synthetic scan whose LIT-region intensity follows a real rocking curve across points
    (a Gaussian in point-index, ``fwhm_points`` wide -- sharp by default, matching where the two
    retracted point-vs-neighbours attempts failed, see scan_quality._check_pedestal_drift), while
    the off-sample background stays pure pedestal Poisson noise at every point regardless of the
    curve.

    ``cutoff_sigma`` truncates the spatial PSF to EXACTLY zero beyond ``cutoff_sigma * sigma_px``
    (deliberately tight -- 2.0, not e.g. 4.0): a literal Gaussian PSF never reaches zero, and its
    small-but-nonzero tail just outside the "lit" threshold scales with the curve amplitude, which
    very nearly reintroduces the exact curve-shape-aliasing failure mode this check exists to
    avoid (found during development: a looser cutoff of 4 sigma gave real, if small, tail counts
    at the lit/off-sample boundary that correlated with the curve and false-flagged points right
    next to the peak). At 2.0 sigma the PSF's value at the truncation radius is still large
    relative to the lit threshold for any realistic amplitude, so the truncation and the "lit"
    threshold boundary coincide and the off-sample proxy is exactly (not just approximately)
    curve-independent, matching how a real diffracted beam's footprint is not present at all far
    from the diffracting condition.
    """
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:H2, 0:W2]
    r2 = (yy - H2 / 2) ** 2 + (xx - W2 / 2) ** 2
    shape = np.exp(-0.5 * r2 / SIGMA_PX2 ** 2)
    shape = np.where(r2 <= (cutoff_sigma * SIGMA_PX2) ** 2, shape, 0.0)
    sigma_pts = fwhm_points / 2.3548
    pts = np.arange(n_points)
    curve = np.exp(-0.5 * ((pts - peak_point) / sigma_pts) ** 2)
    frames = np.empty((n_points, R, H2, W2), dtype=np.float32)
    for i, c in enumerate(curve):
        lam = pedestal + amplitude * c * shape
        frames[i] = rng.poisson(lam[None], size=(R, H2, W2)).astype(np.float32)
    x = 15.700 + 0.005 * np.arange(n_points)
    motors = {"th": x, "tth": np.full(n_points, 25.5254)}
    return RawRepeatScan.from_arrays(frames, motors, source="synthetic rocking-curve scan")


# --------------------------------------------------------------------------- clean scan
def test_clean_scan_has_nothing_flagged():
    scan = _clean_repeat_scan(seed=1)
    report = assess_scan_quality(scan)
    assert report.verdict == "usable", report.summary()
    assert report.flagged_points == []
    for p in report.points:
        for fr in p.frames:
            assert not fr.flagged, (p.point, fr.repeat, [str(c) for c in fr.flagged_checks])
    assert not any(c.flagged for c in report.scan_checks)


def test_clean_scans_do_not_false_positive_across_several_seeds():
    """No single lucky seed: repeat the clean-scan check a few times."""
    for seed in range(5):
        report = assess_scan_quality(_clean_repeat_scan(seed=100 + seed))
        assert report.verdict == "usable", (seed, report.summary())


# --------------------------------------------------------------------------- frame level
def test_assess_scan_quality_never_mutates_the_input():
    scan = _clean_repeat_scan(seed=2)
    before = scan.frames.copy()
    assess_scan_quality(scan)
    np.testing.assert_array_equal(scan.frames, before)


def test_saturated_frame_is_flagged_and_only_it():
    scan = _clean_repeat_scan(seed=3)
    p, r = 5, 3
    scan.frames[p, r, 10:15, 10:15] = 70000.0          # above the 65535 default ceiling
    report = assess_scan_quality(scan)
    victim = report.points[p].frames[r]
    assert any(c.name == "saturation" and c.flagged for c in victim.checks)
    for rr in range(R):
        if rr == r:
            continue
        sib = report.points[p].frames[rr]
        assert not any(c.name == "saturation" and c.flagged for c in sib.checks)
    for pp in range(N_POINTS):
        if pp == p:
            continue
        for fr in report.points[pp].frames:
            assert not any(c.name == "saturation" and c.flagged for c in fr.checks)


def test_anomalous_repeat_count_is_flagged_against_its_own_point_siblings():
    scan = _clean_repeat_scan(seed=4)
    p, r = 7, 2
    scan.frames[p, r] *= 1.6                            # ~60% more counts than its siblings
    report = assess_scan_quality(scan)
    victim = report.points[p].frames[r]
    assert any(c.name == "count_anomaly" and c.flagged for c in victim.checks), victim.checks
    for rr in range(R):
        if rr == r:
            continue
        sib = report.points[p].frames[rr]
        assert not any(c.name == "count_anomaly" and c.flagged for c in sib.checks)
    # a point elsewhere in the scan is untouched
    other = report.points[0].frames[0]
    assert not any(c.name == "count_anomaly" and c.flagged for c in other.checks)


def test_anomalously_low_repeat_count_is_also_caught():
    scan = _clean_repeat_scan(seed=5)
    p, r = 3, 6
    scan.frames[p, r] *= 0.3
    report = assess_scan_quality(scan)
    victim = report.points[p].frames[r]
    checks = [c for c in victim.checks if c.name == "count_anomaly"]
    assert checks and checks[0].flagged and checks[0].value < 0   # low side: negative z


def test_localized_spike_is_flagged_but_the_real_peak_is_not():
    scan = _clean_repeat_scan(seed=6)
    p, r = 2, 4
    row, col = _bg_pixel()
    scan.frames[p, r, row, col] += 4000.0               # single-pixel cosmic-ray-like spike
    report = assess_scan_quality(scan)
    victim = report.points[p].frames[r]
    spike = [c for c in victim.checks if c.name == "spike"][0]
    assert spike.flagged, victim.checks
    assert spike.value <= 4                             # small cluster, per max_spike_pixels
    # the broad real peak, in this SAME frame, must not also read as a spike elsewhere
    for pp in range(N_POINTS):
        for rr in range(R):
            if (pp, rr) == (p, r):
                continue
            c = [c for c in report.points[pp].frames[rr].checks if c.name == "spike"][0]
            assert not c.flagged, (pp, rr, c.message)


def test_spike_check_does_not_fire_on_a_stronger_but_still_broad_peak():
    """A brighter, still-broad PSF must not be mistaken for a spike: only spatial extent (and
    the sharp discontinuity from a pixel's immediate neighbours) should matter, not amplitude."""
    scan = _clean_repeat_scan(seed=7, amplitude=12000.0)
    report = assess_scan_quality(scan)
    for p in report.points:
        for fr in p.frames:
            spike = [c for c in fr.checks if c.name == "spike"][0]
            assert not spike.flagged, (p.point, fr.repeat, spike.message)


def test_dead_frozen_frame_is_flagged():
    scan = _clean_repeat_scan(seed=8)
    p, r = 9, 1
    scan.frames[p, r] = PEDESTAL                        # perfectly flat: no noise, no structure
    report = assess_scan_quality(scan)
    victim = report.points[p].frames[r]
    assert any(c.name == "dead_frame" and c.flagged for c in victim.checks)
    for rr in range(R):
        if rr == r:
            continue
        sib = report.points[p].frames[rr]
        assert not any(c.name == "dead_frame" and c.flagged for c in sib.checks)


def test_duplicated_frame_is_flagged():
    scan = _clean_repeat_scan(seed=9)
    p, r_dup, r_src = 4, 5, 6
    scan.frames[p, r_dup] = scan.frames[p, r_src].copy()
    report = assess_scan_quality(scan)
    victim = report.points[p].frames[r_dup]
    dead = [c for c in victim.checks if c.name == "dead_frame"][0]
    assert dead.flagged and "identical" in dead.message
    # its source is symmetrically flagged too: with only two identical frames, there is no way
    # to tell which one is the "real" exposure and which is the duplicate, so both are flagged
    src_result = report.points[p].frames[r_src]
    src_dead = [c for c in src_result.checks if c.name == "dead_frame"][0]
    assert src_dead.flagged and "identical" in src_dead.message
    # a repeat at a DIFFERENT point that happens to also be r_src's index is unaffected
    other = report.points[p + 1 if p + 1 < N_POINTS else p - 1].frames[r_src]
    assert not any(c.name == "dead_frame" and c.flagged for c in other.checks)


def test_too_few_surviving_repeats_flags_the_point():
    scan = _clean_repeat_scan(seed=10)
    p = 6
    for r in range(6):                                  # 6 of 10 repeats corrupted: majority bad
        scan.frames[p, r, 10:15, 10:15] = 70000.0
    report = assess_scan_quality(scan)
    survival = [c for c in report.points[p].checks if c.name == "repeat_survival"][0]
    assert survival.flagged
    assert report.points[p].n_surviving <= R // 2


# --------------------------------------------------------------------------- point level
def test_photon_starved_point_is_flagged_by_the_flux_check():
    """Modeled on NOTE_S224_frames.md: a point with (near-)zero real signal above pedestal is an
    absolute, per-point failure -- not a comparison to its neighbours (S224 was starved at EVERY
    point, not relative to them)."""
    scan = _clean_repeat_scan(seed=11)
    p = 8
    rng = np.random.default_rng(999)
    starved = rng.poisson(PEDESTAL, size=(R, H, W)).astype(np.float32)   # background only
    scan.frames[p] = starved
    report = assess_scan_quality(scan)
    flux = [c for c in report.points[p].checks if c.name == "flux_health"][0]
    assert flux.flagged, flux.message
    assert "photon-starved" in flux.message or "no detectable signal" in flux.message
    # neighbouring, well-exposed points must not be dragged down by the starved one
    for pp in (p - 1, p + 1):
        f2 = [c for c in report.points[pp].checks if c.name == "flux_health"][0]
        assert not f2.flagged, (pp, f2.message)
    assert p in [pt.point for pt in report.flagged_points]


def test_point_coordinate_label_carries_the_actual_motor_reading():
    scan = _clean_repeat_scan(seed=12)
    report = assess_scan_quality(scan)
    p3 = report.points[3]
    assert p3.coordinate_label == f"th={scan.motors['th'][3]:.4f}"
    assert abs(p3.coordinate["th"] - scan.motors["th"][3]) < 1e-9


# --------------------------------------------------------------------------- scan level
def test_frame_table_row_count_mismatch_is_refused_at_construction():
    frames = np.zeros((5, 3, 8, 8), dtype=np.float32)
    with pytest.raises(ValueError, match="one value per POINT"):
        RawRepeatScan.from_arrays(frames, {"th": np.arange(6.0)})


def test_duplicated_header_block_heuristic_is_caught():
    """A restarted-scan CSV with a duplicated header block can silently double the row count
    while the frame count/point count still divide evenly -- catch the duplicated VALUES, not
    just a mismatched count (known bug class: the NaMnO2 Dryad 'July log doubles every tilt')."""
    n = 8
    rng = np.random.default_rng(13)
    yy, xx = np.mgrid[0:H, 0:W]
    r2 = (yy - H / 2) ** 2 + (xx - W / 2) ** 2
    lam = PEDESTAL + AMPLITUDE * np.exp(-0.5 * r2 / SIGMA_PX ** 2)
    frames = rng.poisson(lam[None, None], size=(n, R, H, W)).astype(np.float32)
    half = np.array([15.700, 15.705, 15.710, 15.715])
    th = np.concatenate([half, half])                   # duplicated block
    scan = RawRepeatScan.from_arrays(frames, {"th": th, "tth": np.full(n, 25.5254)})
    report = assess_scan_quality(scan)
    check = [c for c in report.scan_checks if c.name == "frame_table_consistency"][0]
    assert check.flagged, check.message
    assert "duplicated header" in check.message


def test_step_irregularity_is_caught():
    scan = _clean_repeat_scan(seed=14)
    scan.motors["th"][7] = scan.motors["th"][6] + 0.080   # a 80 mdeg jump vs the 5 mdeg step
    report = assess_scan_quality(scan)
    check = [c for c in report.scan_checks if c.name == "step_regularity"][0]
    assert check.flagged, check.message


def test_rollup_reports_not_usable_when_most_points_are_flagged():
    scan = _clean_repeat_scan(seed=15)
    rng = np.random.default_rng(1000)
    for p in range(8):                                   # majority of 14 points starved
        scan.frames[p] = rng.poisson(PEDESTAL, size=(R, H, W)).astype(np.float32)
    report = assess_scan_quality(scan)
    assert report.verdict.startswith("not usable"), report.verdict


def test_rollup_reports_usable_with_n_flagged_for_a_minority_defect():
    scan = _clean_repeat_scan(seed=16)
    scan.frames[2, 0, 10:15, 10:15] = 70000.0
    report = assess_scan_quality(scan)
    assert report.verdict.startswith("usable,"), report.verdict
    assert "1 of" in report.verdict


# --------------------------------------------------------------------------- apply_quality_filter
def test_apply_quality_filter_excludes_only_what_was_flagged():
    scan = _clean_repeat_scan(seed=17)
    p_sat, r_sat = 5, 2
    p_hot, r_hot = 9, 4
    scan.frames[p_sat, r_sat, 10:15, 10:15] = 70000.0
    scan.frames[p_hot, r_hot] *= 1.6
    before = scan.frames.copy()
    report = assess_scan_quality(scan)
    filtered = apply_quality_filter(scan, report)

    assert isinstance(filtered, RockingScan)
    # untouched points: plain mean over all R repeats
    for p in range(N_POINTS):
        if p in (p_sat, p_hot):
            continue
        want = scan.frames[p].mean(0)
        np.testing.assert_allclose(filtered.frames[p], want, rtol=1e-5)
    # p_sat: repeat r_sat excluded
    want_sat = np.delete(scan.frames[p_sat], r_sat, axis=0).mean(0)
    np.testing.assert_allclose(filtered.frames[p_sat], want_sat, rtol=1e-5)
    # p_hot: repeat r_hot excluded
    want_hot = np.delete(scan.frames[p_hot], r_hot, axis=0).mean(0)
    np.testing.assert_allclose(filtered.frames[p_hot], want_hot, rtol=1e-5)
    # the original raw scan and its already-computed report are untouched
    np.testing.assert_array_equal(scan.frames, before)


def test_not_calling_apply_quality_filter_leaves_everything_as_is():
    scan = _clean_repeat_scan(seed=18)
    scan.frames[1, 0, 10:15, 10:15] = 70000.0
    before = scan.frames.copy()
    report = assess_scan_quality(scan)
    assert report.flagged_points                        # something WAS flagged
    np.testing.assert_array_equal(scan.frames, before)   # but nothing was excluded/changed


def test_apply_quality_filter_falls_back_when_every_repeat_at_a_point_is_flagged():
    scan = _clean_repeat_scan(seed=19)
    p = 5
    for r in range(R):
        scan.frames[p, r, 10:15, 10:15] = 70000.0        # saturate every repeat at this point
    report = assess_scan_quality(scan)
    filtered = apply_quality_filter(scan, report)
    np.testing.assert_allclose(filtered.frames[p], scan.frames[p].mean(0), rtol=1e-5)
    assert any("every repeat flagged" in n for n in filtered.notes)


def test_apply_quality_filter_can_also_drop_whole_flagged_points():
    scan = _clean_repeat_scan(seed=20)
    p = 3
    rng = np.random.default_rng(2000)
    scan.frames[p] = rng.poisson(PEDESTAL, size=(R, H, W)).astype(np.float32)   # starved point
    report = assess_scan_quality(scan)
    filtered = apply_quality_filter(scan, report, drop_flagged_points=True)
    assert filtered.frames.shape[0] == N_POINTS - 1
    assert p not in filtered.motors["th"].tolist()       # its motor value is gone too


def test_apply_quality_filter_needs_a_repeats_granularity_report():
    scan = _clean_repeat_scan(seed=21)
    rocking = scan.to_rocking_scan()
    degraded_report = assess_scan_quality(rocking)
    with pytest.raises(ValueError, match="granularity"):
        apply_quality_filter(scan, degraded_report)


# --------------------------------------------------------------------------- degraded RockingScan
def test_degraded_mode_on_a_rocking_scan_with_halves():
    scan = _clean_repeat_scan(seed=22)
    rocking = scan.to_rocking_scan()
    assert rocking.halves is not None
    report = assess_scan_quality(rocking)
    assert report.granularity == "halves"
    assert any("DEGRADED" in n for n in report.notes)
    assert report.verdict == "usable"


def test_degraded_mode_on_a_rocking_scan_without_halves():
    scan = _clean_repeat_scan(seed=23, R=1)
    rocking = scan.to_rocking_scan()
    assert rocking.halves is None
    report = assess_scan_quality(rocking)
    assert report.granularity == "average-only"
    assert all(len(p.frames) == 0 for p in report.points)
    # the point-level flux check still runs
    assert all(any(c.name == "flux_health" for c in p.checks) for p in report.points)


def test_degraded_mode_still_catches_a_starved_point_via_flux_health():
    scan = _clean_repeat_scan(seed=24)
    p = 6
    rng = np.random.default_rng(3000)
    scan.frames[p] = rng.poisson(PEDESTAL, size=(R, H, W)).astype(np.float32)
    rocking = scan.to_rocking_scan()
    report = assess_scan_quality(rocking)
    assert report.points[p].flagged


# ------------------------------------------------------------- static hot pixels (bug 1: S168)
def test_static_hot_pixel_is_excluded_from_the_per_frame_spike_check():
    """A pixel elevated by the same spike-sized excess in EVERY repeat of EVERY point (a fixed
    detector defect) must be reported ONCE at the scan level, not as a per-frame spike flag on
    every single frame -- the S168 bug: ~4000 static hot pixels each tripped the spike check in
    every one of 122 frames, reading as "every point flagged" instead of a handful of known-bad
    pixel locations."""
    scan = _clean_repeat_scan(seed=40)
    row, col = _bg_pixel()
    scan.frames[:, :, row, col] += 4000.0             # same location, every point, every repeat
    report = assess_scan_quality(scan)

    hot_check = [c for c in report.scan_checks if c.name == "static_hot_pixels"][0]
    assert not hot_check.flagged, hot_check.message   # one hot pixel is not a sanity-ceiling breach
    assert "1 static hot pixel" in hot_check.message

    n_spike_flags = sum(1 for p in report.points for fr in p.frames
                        for c in fr.checks if c.name == "spike" and c.flagged)
    assert n_spike_flags == 0, n_spike_flags
    assert report.verdict == "usable", report.summary()


def test_random_per_frame_spikes_are_not_masked_as_static_hot_pixels():
    """A real cosmic ray lands at a DIFFERENT pixel each time it happens -- planting several
    one-off spikes at distinct random locations must not get swept into the static-defect mask,
    and must still be caught as ordinary per-frame spikes."""
    scan = _clean_repeat_scan(seed=41)
    rng = np.random.default_rng(123)
    used = set()
    planted = []
    for _ in range(8):
        p = int(rng.integers(0, N_POINTS))
        r = int(rng.integers(0, R))
        while True:
            row, col = int(rng.integers(3, H - 3)), int(rng.integers(3, W - 3))
            if (row, col) not in used:
                used.add((row, col))
                break
        scan.frames[p, r, row, col] += 4000.0
        planted.append((p, r))
    report = assess_scan_quality(scan)

    hot_check = [c for c in report.scan_checks if c.name == "static_hot_pixels"][0]
    assert not hot_check.flagged
    assert "no static hot pixels found" in hot_check.message

    n_caught = sum(1 for (p, r) in planted
                  if any(c.name == "spike" and c.flagged for c in report.points[p].frames[r].checks))
    assert n_caught == len(planted), (n_caught, len(planted))


# --------------------------------------------------------- gain / read-noise estimation (bug 2)
def _known_gain_scan(seed=0, true_gain=2.0, true_read_noise_var=25.0, n_points=20, R=10,
                     pedestal=300.0, amplitude=6000.0):
    """A synthetic scan with a KNOWN, non-unity gain (counts/photon) and read-noise variance:
    ``counts = true_gain * Poisson(mean_photons) + Normal(0, sqrt(true_read_noise_var))``, so
    var(counts) = true_gain*mean(counts) + true_read_noise_var -- exactly the model
    :func:`estimate_gain` fits (see its docstring's "Convention")."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:H, 0:W]
    r2 = (yy - H / 2) ** 2 + (xx - W / 2) ** 2
    prof = np.exp(-0.5 * r2 / SIGMA_PX ** 2)
    c = (n_points - 1) / 2.0
    env = 0.05 + 0.95 * np.exp(-0.5 * ((np.arange(n_points) - c) / (n_points / 6.0)) ** 2)
    frames = np.empty((n_points, R, H, W), dtype=np.float32)
    for i in range(n_points):
        lam_photons = (pedestal + amplitude * env[i] * prof) / true_gain
        photons = rng.poisson(lam_photons[None], size=(R, H, W))
        counts = true_gain * photons + rng.normal(0, math.sqrt(true_read_noise_var), size=(R, H, W))
        frames[i] = counts.astype(np.float32)
    x = 15.700 + 0.005 * np.arange(n_points)
    motors = {"th": x, "tth": np.full(n_points, 25.5254)}
    return RawRepeatScan.from_arrays(frames, motors, source="synthetic known-gain scan")


def test_estimate_gain_recovers_a_known_nontrivial_gain():
    scan = _known_gain_scan(seed=50, true_gain=2.0, true_read_noise_var=25.0)
    gain, read_noise_var, info = estimate_gain(scan)
    assert not info["fallback"], info["notes"]
    assert 1.5 < gain < 2.6, (gain, info)          # recovered within ~25% of the true gain=2.0
    assert 0.0 <= read_noise_var < 120.0, (read_noise_var, info)


def test_estimate_gain_falls_back_to_one_without_enough_repeats():
    scan = _clean_repeat_scan(seed=51, R=1)        # a single repeat/point: no variance to measure
    gain, read_noise_var, info = estimate_gain(scan)
    assert gain == 1.0 and read_noise_var == 0.0
    assert info["fallback"]


def test_estimate_gain_falls_back_on_a_photon_starved_scan_with_no_real_dynamic_range():
    """Real S224 (photon-starved, see NOTE_S224_frames.md) regression: every pixel at every point
    sits at essentially the same pedestal level, so the pooled (mean, variance) pairs are pure
    scatter with no real photon-transfer trend to fit -- a naive fit on this real scan returned
    gain=0.28, a number with no physical meaning, not a real measurement of a different gain."""
    rng = np.random.default_rng(60)
    frames = rng.poisson(PEDESTAL, size=(N_POINTS, R, H, W)).astype(np.float32)
    x = 15.700 + 0.005 * np.arange(N_POINTS)
    motors = {"th": x, "tth": np.full(N_POINTS, 25.5254)}
    scan = RawRepeatScan.from_arrays(frames, motors, source="synthetic photon-starved scan")
    gain, read_noise_var, info = estimate_gain(scan)
    assert gain == 1.0 and read_noise_var == 0.0
    assert info["fallback"]
    assert any("R^2" in n for n in info["notes"])


def test_estimate_gain_skips_degraded_halves_granularity():
    """A RockingScan's ``halves`` are already averages of several repeats each -- they do not
    carry single-exposure photon statistics, so gain estimation must not (mis)use them."""
    scan = _clean_repeat_scan(seed=52)
    rocking = scan.to_rocking_scan()
    gain, read_noise_var, info = estimate_gain(rocking)
    assert gain == 1.0 and info["fallback"]
    assert "granularity" in info["notes"][0]


def test_assess_scan_quality_auto_estimates_gain_by_default():
    scan = _known_gain_scan(seed=53, true_gain=2.0, true_read_noise_var=25.0)
    report = assess_scan_quality(scan)
    assert report.settings["gain"] != 1.0
    assert 1.5 < report.settings["gain"] < 2.6
    assert report.settings["gain_info"] is not None
    assert any("gain auto-estimated" in n for n in report.notes)


def test_assess_scan_quality_explicit_gain_overrides_and_skips_auto_estimate():
    scan = _known_gain_scan(seed=54, true_gain=2.0, true_read_noise_var=25.0)
    report = assess_scan_quality(scan, gain=1.0)
    assert report.settings["gain"] == 1.0
    assert report.settings["read_noise_var"] == 0.0
    assert report.settings["gain_info"] is None
    assert not any("gain auto-estimated" in n for n in report.notes)


def test_wrong_assumed_gain_creates_false_flags_that_auto_estimate_avoids():
    """The concrete payoff of auto-estimating gain: assuming gain=1 on a detector that is really
    gain~3 makes the Poisson-noise floors too tight everywhere, so ordinary shot noise starts
    tripping thresholds meant for real anomalies."""
    scan = _known_gain_scan(seed=45, true_gain=3.0, true_read_noise_var=25.0)
    report_auto = assess_scan_quality(scan)
    report_wrong = assess_scan_quality(scan, gain=1.0)
    assert report_auto.verdict == "usable", report_auto.summary()
    n_wrong = sum(1 for p in report_wrong.points if p.flagged)
    assert n_wrong >= 1, "expected the wrong gain=1 assumption to manufacture at least one flag"


# ------------------------------------------------- point-vs-neighbours pedestal drift (bug 3)
def test_pedestal_drift_no_false_positive_near_a_sharp_peak():
    """The two retracted attempts at this check (see _check_pedestal_drift's docstring) both
    false-flagged points right around a sharp rocking-curve peak. This is the regression test for
    exactly that failure mode: several seeds, a peak only 2.5 points (FWHM) wide."""
    for seed in range(6):
        scan = _rocking_curve_scan(seed=seed)
        report = assess_scan_quality(scan)
        flagged = [(p.point, c.value) for p in report.points for c in p.checks
                  if c.name == "pedestal_drift" and c.flagged]
        assert flagged == [], (seed, flagged)


def test_pedestal_drift_no_false_positive_with_a_broad_peak_either():
    for seed in range(3):
        scan = _rocking_curve_scan(seed=100 + seed, fwhm_points=9.0)
        report = assess_scan_quality(scan)
        flagged = [(p.point, c.value) for p in report.points for c in p.checks
                  if c.name == "pedestal_drift" and c.flagged]
        assert flagged == [], (seed, flagged)


def test_pedestal_drift_catches_a_planted_whole_frame_glitch():
    """A whole-point brightness anomaly (S168 point 32's problem): both "repeats" agree (there is
    no within-point disagreement for _check_count_anomaly to see), and the point is too BRIGHT,
    not starved, so flux_health passes fine -- only the off-sample-vs-neighbours comparison can
    catch it."""
    scan = _rocking_curve_scan(seed=42)
    p_glitch = 17                # near but not AT the peak (20): enough real signal that
                                 # flux_health passes (this point is bright, not starved),
                                 # matching the actual S168 point-32 scenario
    scan.frames[p_glitch] *= 1.03                       # +3% whole-frame brightness anomaly
    report = assess_scan_quality(scan)

    victim = [c for c in report.points[p_glitch].checks if c.name == "pedestal_drift"][0]
    assert victim.flagged, victim.message
    count_anomaly_flags = [c for fr in report.points[p_glitch].frames for c in fr.checks
                           if c.name == "count_anomaly" and c.flagged]
    assert count_anomaly_flags == [], "both repeats were scaled alike -- no WITHIN-point signal"
    flux_check = [c for c in report.points[p_glitch].checks if c.name == "flux_health"][0]
    assert not flux_check.flagged, "too bright, not starved -- flux_health must pass"

    for pp in (p_glitch - 1, p_glitch + 1):
        c = [c for c in report.points[pp].checks if c.name == "pedestal_drift"][0]
        assert not c.flagged, (pp, c.message)


# --------------------------------------------------------------------------- real-layout loader
tifffile = pytest.importorskip("tifffile")
import csv                                                                      # noqa: E402


def _write_csv(path, header, rows):
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)


def _write_indexed_scan(root, npts=9, reps=4, seed=0):
    rng = np.random.default_rng(seed)
    x = 15.4732 + 0.010 * np.arange(npts)
    yy, xx = np.mgrid[0:24, 0:24]
    r2 = (yy - 12) ** 2 + (xx - 12) ** 2
    env = 0.05 + 0.95 * np.exp(-0.5 * ((np.arange(npts) - (npts - 1) / 2.0) / (npts / 6.0)) ** 2)
    lam = 300.0 + 2500.0 * env[:, None, None] * np.exp(-0.5 * r2 / 4.0 ** 2)[None]
    reps_arr = rng.poisson(lam[:, None], size=(npts, reps, 24, 24)).astype(np.float64)
    d = os.path.join(root, "data", "scans_raw", "DFXM_S1")
    os.makedirs(d)
    for pt in range(npts):
        for r in range(reps):
            tifffile.imwrite(os.path.join(d, f"data_{pt * reps + r:05d}.tif"),
                             np.clip(np.rint(reps_arr[pt, r]), 0, 65535).astype(np.uint16))
    mdir = os.path.join(root, "data", "motor_files")
    os.makedirs(mdir)
    table = os.path.join(mdir, "sample_motor_information_S1_0.1s.csv")
    _write_csv(table, ["Num", "tth", "th"], [[i, 25.5254, f"{th:.6f}"] for i, th in enumerate(x)])
    return d, table, reps_arr


def test_load_6idc_repeat_frames_reads_individual_repeats(tmp_path):
    d, table, reps_arr = _write_indexed_scan(str(tmp_path))
    scan = load_6idc_repeat_frames(d, table)
    assert scan.frames.shape == (9, 4, 24, 24)
    np.testing.assert_allclose(scan.frames, reps_arr, atol=1.0)
    assert scan.scan_type == "tilt" and scan.axes == ("th",)
    report = assess_scan_quality(scan)
    assert report.verdict == "usable", report.summary()


def test_load_6idc_repeat_frames_refuses_named_layout(tmp_path):
    os.makedirs(os.path.join(str(tmp_path), "S006"))
    with pytest.raises(FileNotFoundError, match="named layout"):
        load_6idc_repeat_frames(os.path.join(str(tmp_path), "S006"))


def test_stable_noisy_texture_is_not_a_spike_but_a_one_off_is():
    """Real S996 pattern: a pixel elevated in EVERY repeat (150-200 counts on a ~108 background)
    only clears z>8 in one repeat (the hot first one); siblings sit at z ~ 3-7, NOT >8. That is
    stable texture, not a cosmic ray. A pixel elevated in ONE repeat only (sibling z ~ 0) is a
    genuine transient and must still be flagged."""
    rng = np.random.default_rng(11)
    R, H, W = 6, 40, 40
    frames = rng.poisson(108.0, size=(R, H, W)).astype(np.float32)
    # stable texture pixel: elevated everywhere, extra-high in repeat 0
    frames[:, 10, 10] = rng.poisson(165.0, size=R)
    frames[0, 10, 10] = 215.0
    # genuine transient: only repeat 3
    frames[3, 30, 30] = 260.0
    fr = [frames[r] for r in range(R)]
    from midas_dfxm.scan_quality import _check_spike
    stable = _check_spike(fr[0], 1.0, 8.0, 4, 10.0,
                          sibling_frames=[fr[i] for i in range(R) if i != 0])
    assert not stable.flagged, stable.message
    assert "excluded" in stable.message
    transient = _check_spike(fr[3], 1.0, 8.0, 4, 10.0,
                             sibling_frames=[fr[i] for i in range(R) if i != 3])
    assert transient.flagged, transient.message


def test_hot_first_repeat_at_every_point_is_usable_and_called_systematic():
    """Real S996: repeat 0 reads a few % high at most points (documented io_6idc hot first repeat).
    That is excludable, not a reason to call the whole scan unusable; it must be reported as a
    systematic per-acquisition effect, not as random bad frames."""
    scan = _clean_repeat_scan(seed=21, n_points=12, R=10)
    scan.frames[:, 0] *= 1.06
    report = assess_scan_quality(scan)
    assert report.verdict.startswith("usable"), report.verdict
    assert "have flagged repeats only" in report.verdict
    assert any("SYSTEMATIC" in n and "repeat index 0" in n for n in report.notes), report.notes
    assert not any(p.point_level_flagged for p in report.points)


def test_point_with_too_few_surviving_repeats_still_fails_point_level():
    scan = _clean_repeat_scan(seed=22, n_points=12, R=10)
    scan.frames[3, :7] = 65535.0                 # saturate 7 of 10 repeats at point 3
    report = assess_scan_quality(scan)
    assert report.points[3].point_level_flagged
    assert "1 of 12 points flagged" in report.verdict


def test_unbracketed_flat_scan_is_caught_independent_of_gain():
    """S224 pattern: the significant contiguous region is ~21 % of the frame at EVERY point, i.e.
    the scan edges are as bright as its centre and no peak is bracketed. Must be caught, and must
    not depend on the (unstable) auto-gain estimate."""
    flat = _clean_repeat_scan(seed=31, envelope=False)
    for g in (0.278, 1.0, 2.7):
        rep = assess_scan_quality(flat, gain=g)
        chk = [c for c in rep.scan_checks if c.name == "curve_bracketing"][0]
        assert chk.flagged, (g, chk.message)
        assert rep.verdict.startswith("not usable"), rep.verdict
    ok = assess_scan_quality(_clean_repeat_scan(seed=32))
    assert not [c for c in ok.scan_checks if c.name == "curve_bracketing"][0].flagged


def test_scan_with_no_signal_region_anywhere_is_flagged():
    dead = _clean_repeat_scan(seed=33, amplitude=0.0)
    rep = assess_scan_quality(dead, gain=1.0)
    chk = [c for c in rep.scan_checks if c.name == "curve_bracketing"][0]
    assert chk.flagged and "no usable signal region" in chk.message



def test_pedestal_drift_needs_an_effect_size_not_only_a_z_score():
    """S996's off-sample level is so smooth (0.01 counts point to point on 107.7) that a 0.014 %
    dip reads z ~ -9 against the tiny shot-noise floor; that is not a brightness anomaly. A 4 %
    jump (S168 point 32) must still flag, with the default 4-point window that a slow pedestal
    drift cannot dilute."""
    from midas_dfxm.scan_quality import _check_pedestal_drift
    rng = np.random.default_rng(5)
    base = 107.7 + 0.002 * np.arange(40) + rng.normal(0, 0.003, 40)
    floors = np.full(40, 0.0017)
    tiny = base.copy(); tiny[11] -= 0.016
    assert not any(c.flagged for c in _check_pedestal_drift(tiny, floors, 4, 4.0))
    big = base.copy(); big[20] *= 1.04
    ch = _check_pedestal_drift(big, floors, 4, 4.0)
    assert [i for i, c in enumerate(ch) if c.flagged] == [20]
