"""Broad, structured rocking curves: the shape check, window="fixed", and the repeat-excess map.

The fixtures come from :func:`midas_dfxm.example_rocking_scan`: ``single`` (one 5 mdeg peak per
pixel on a smooth 24 mdeg ramp), ``broad`` (a 30 mdeg box with sharp horns whose heights switch
across a line, and no tilt anywhere) and ``step`` (the box plus a real 20 mdeg tilt step). The
thresholds were fixed from the synthetic smoke run in the NX-school dry-run folder before the
tests were written, with margin: single-peaked fraction 1.00 / 0.03-0.05; fake step +18.5 mdeg
from the peak window on ``broad`` against +4.1 from the median; planted step 20 read as 23.5 /
38.6; photon-only repeat-excess spread 0.62-1.38 (single) and 0.79-1.19 (broad), 1.00 +- 0.06
absolute at the true gain; 1 px rms planted motion 1.2-2.1 absolute on ``single`` depending on the seed, so
the test plants 2 px. Repeat shift (signal-weighted rms between halves) on this 64x64 field: photon noise
0.05-0.32 px, planted 2 px gives an envelope shift of 1.0-1.6 px and a fine-detail shift of 0.8-1.1 px
(seeds 1, 3, 5); on a full 6-ID-C frame photon noise gives hundredths of a pixel.
"""
import numpy as np
import pytest

import midas_dfxm as dx


@pytest.fixture(scope="module")
def scans():
    return {k: dx.example_rocking_scan(k, seed=1) for k in ("single", "broad", "step")}


def _good(m):
    return m.lit & ~m.truncated


def _step_across(m, sep=6, w=6):
    """Median value on the right of the mid line minus the left, ``w`` px wide bands ``sep`` px out."""
    v = np.where(_good(m), m.value, np.nan)
    c = v.shape[1] // 2
    return float(np.nanmedian(v[:, c + sep:c + sep + w]) - np.nanmedian(v[:, c - sep - w:c - sep]))


def _jump_fraction(m, sep=5, thr=12.0):
    v = np.where(_good(m), m.value, np.nan)
    a, b = v[:, :-sep], v[:, sep:]
    ok = np.isfinite(a) & np.isfinite(b)
    return float(np.mean(np.abs(a - b)[ok] > thr))


def test_shape_check_separates_one_peak_from_a_box_with_horns(scans):
    single = dx.reduce_rocking(scans["single"])
    broad = dx.reduce_rocking(scans["broad"])
    assert single.single_peaked[single.lit].mean() > 0.97
    assert broad.single_peaked[broad.lit].mean() < 0.15
    assert np.nanmedian(single.peak_share[single.lit]) > 0.7
    assert np.nanmedian(broad.peak_share[broad.lit]) < 0.35
    assert not any("not single-peaked" in n for n in single.notes)
    assert any("not single-peaked" in n for n in broad.notes)


def test_peak_window_steps_where_the_taller_horn_switches_and_the_median_does_not(scans):
    peak = dx.reduce_rocking(scans["broad"])
    fixed = dx.reduce_rocking(scans["broad"], window="fixed")
    # no tilt was planted anywhere
    assert abs(_step_across(peak)) > 12.0
    assert abs(_step_across(fixed)) < 6.0
    assert _jump_fraction(peak) > 0.01
    assert _jump_fraction(fixed) == 0.0
    assert fixed.settings["window"] == "fixed" and fixed.settings["baseline"] == "ends"
    assert fixed.centre_shift is not None and fixed.span_mdeg is not None
    assert 20.0 < np.nanmedian(fixed.span_mdeg[_good(fixed)]) < 32.0   # the 30 mdeg box
    assert np.nanmedian(np.abs(fixed.centre_shift[_good(fixed)])) < 2.0


def test_a_real_tilt_step_is_recovered_by_both_windows(scans):
    peak = dx.reduce_rocking(scans["step"])
    fixed = dx.reduce_rocking(scans["step"], window="fixed")
    assert 15.0 < _step_across(fixed) < 28.0          # planted 20; the horn switch adds a few
    assert _step_across(peak) > 28.0                  # planted 20 plus the horn switch


def test_fixed_window_recovers_a_smooth_ramp_of_narrow_peaks(scans):
    scan = scans["single"]
    m = dx.reduce_rocking(scan, window="fixed")
    truth = scan.meta["truth"]["tilt_mdeg"]
    g = _good(m)
    assert g.sum() > 300
    err = (m.value - (truth - np.median(truth[g])))[g]
    assert np.median(np.abs(err)) < 0.6
    assert m.sigma_global is not None and m.sigma_global < 0.6
    assert m.single_peaked[m.lit].mean() > 0.97


def test_signal_range_can_be_given_and_is_recorded(scans):
    m = dx.reduce_rocking(scans["broad"], window="fixed", signal_range=(7.330, 7.375))
    lo, hi = m.settings["signal_range"]
    assert abs(lo - 7.330) < 1.5e-3 and abs(hi - 7.375) < 1.5e-3
    with pytest.raises(ValueError):
        dx.reduce_rocking(scans["broad"], baseline="ends")      # only with window="fixed"


def test_repeat_excess_is_flat_for_photon_noise_and_rises_with_planted_motion():
    still = dx.example_rocking_scan("single", seed=3)
    moving = dx.example_rocking_scan("single", seed=3, motion_px=2.0)
    ms = dx.reduce_rocking(still, gain=1.0)
    mm = dx.reduce_rocking(moving, gain=1.0)
    lo, hi, abs_still = ms.repeat_excess_spread
    assert 0.5 < lo < 1.0 < hi < 1.6
    assert 0.85 < abs_still < 1.2                     # photon noise only: 1.0 at the true gain
    assert mm.repeat_excess_spread[2] > 1.5           # frames that are not copies of each other
    assert not any("differ by more than photon noise" in n for n in ms.notes)
    # the apparent shift of the intensity pattern between the repeat halves (gain-free)
    assert max(ms.repeat_shift_rms[:4]) < 0.4         # photon noise on this small field: 0.05-0.32 px
    assert max(mm.repeat_shift_rms[:2]) > 0.9         # envelope: planted 2 px rms per frame (seen 1.0-1.6)
    assert max(mm.repeat_shift_rms[2:4]) > 0.5        # fine detail moves with it (rigid plant; seen 0.8-1.1)
    assert any("intensity envelope shifts" in n for n in mm.notes)
    assert not any("intensity envelope shifts" in n for n in ms.notes)


def test_baseline_sensitivity_has_fixed_window_variants(scans):
    rows = dx.baseline_sensitivity(scans["broad"], window="fixed")
    assert len(rows) == 4 and all(np.isfinite(r["block_rms"]) for r in rows)
    assert any("signal_range" in r["variant"] for r in rows)


def test_maps_save_the_new_arrays(tmp_path, scans):
    m = dx.reduce_rocking(scans["broad"], window="fixed")
    m.save(str(tmp_path / "maps.npz"))
    z = np.load(tmp_path / "maps.npz", allow_pickle=True)
    for k in ("single_peaked", "peak_share", "n_features", "centre_shift", "span_mdeg", "repeat_excess"):
        assert k in z.files
