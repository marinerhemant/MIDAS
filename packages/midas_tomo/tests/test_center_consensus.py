"""Automatic rotation-axis shift selection, and its refusals.

``argmax`` over any curve always returns something, so the value of an
automatic centring routine is entirely in knowing when not to believe it. Most
of these tests are about the ``trustworthy`` flag rather than the shift.
"""
from __future__ import annotations

import numpy as np
import pytest

from midas_tomo.center import find_center, find_center_consensus, sharpness


def _cube(n_shifts=11, n_slices=6, x=64, best=5, contrast=1.0, noise=0.0,
          seed=0):
    """A sweep that models what mis-centring actually does to a slice.

    Getting this fixture right took three attempts and the failures are worth
    recording, because each one made the two criteria disagree by construction
    and so tested only the refusal path:

    1. **Blurring a disc.** ``sharpness``'s own docstring says TV is close to
       blur-invariant — for a monotonic edge ``int|grad f|`` is the total
       height change however wide the ramp. Measured: TV picked the
       *blurriest* candidate, every time.
    2. **Splitting into two half-amplitude copies.** Halving the amplitude
       halves the edge gradient, so mean|grad| *fell* as mis-centring grew and
       TV again preferred the worst candidate.

    What mis-centring really does is lose contrast (density smeared over the
    wrong voxels) *and* add high-frequency streaks across the whole field. So
    that is what this builds. The streaks barely move the variance, being
    small in amplitude, but they dominate the gradient — which is precisely
    the asymmetry that makes two criteria worth running.

    3. **Streaks too weak.** With the streak term at 0.012/step against a 6 %
       contrast loss, the well-centred disc's own edge still carried more
       gradient than the artefacts, and TV inverted again. The balance is
       real, not a fixture artefact: ``sharpness(method='tv')`` finds the
       centre only when the artefact gradient exceeds the specimen's own edge
       gradient. On a high-contrast, low-artefact reconstruction it can prefer
       the *most* degraded slice, which is exactly the case the consensus
       refusal exists to catch.
    """
    rng = np.random.default_rng(seed)
    iy, ix = np.mgrid[0:x, 0:x].astype(np.float64)
    r = np.hypot(ix - (x - 1) / 2, iy - (x - 1) / 2)
    disc = contrast / (1.0 + np.exp((r - x * 0.25) / 0.8))
    # A fixed streak pattern, so the only thing varying across the sweep is
    # how much of it there is.
    streaks = np.sin(ix * 2.1) * np.cos(iy * 1.7)

    cube = np.empty((n_shifts, n_slices, x, x), dtype=np.float32)
    for i in range(n_shifts):
        k = abs(i - best)
        img = disc * (1.0 - 0.03 * k) + 0.04 * k * streaks
        for s in range(n_slices):
            cube[i, s] = img + noise * rng.standard_normal((x, x))
    return cube


SWEEP = (-5.0, 5.0, 1.0)          # 11 candidates, step 1.0 -> tol 2.0


# ------------------------------------------------------------- it works

def test_the_consensus_finds_the_planted_shift():
    c = find_center_consensus(_cube(best=5), SWEEP)
    assert c["trustworthy"], c["reason"]
    assert c["best_shift"] == pytest.approx(0.0, abs=1.0)   # index 5 -> 0.0


def test_a_shifted_optimum_is_followed():
    c = find_center_consensus(_cube(best=8), SWEEP)
    assert c["trustworthy"], c["reason"]
    assert c["best_shift"] == pytest.approx(3.0, abs=1.0)


def test_it_scores_several_slices_not_just_the_middle_one():
    """A single slice can be empty, all sample, or sitting on a defect."""
    c = find_center_consensus(_cube(n_slices=8), SWEEP)
    assert len(c["slices"]) == 4
    assert len(set(c["slices"])) == 4
    for m in ("variance", "tv"):
        assert len(c["per_method"][m]["picks"]) == 4


def test_explicit_slices_are_honoured():
    c = find_center_consensus(_cube(n_slices=8), SWEEP, slices=[1, 6])
    assert c["slices"] == [1, 6]


# --------------------------------------------------------- the refusals

def test_a_flat_sweep_is_NOT_trustworthy():
    """Nothing to choose between: every candidate reconstructs the same. The
    argmax still returns an index, and that index means nothing."""
    cube = np.ones((11, 4, 64, 64), dtype=np.float32)
    cube += np.random.default_rng(0).standard_normal(cube.shape) * 1e-9
    c = find_center_consensus(cube, SWEEP)
    assert not c["trustworthy"]
    assert "could not separate" in c["reason"]


def test_disagreeing_criteria_are_NOT_trustworthy():
    """Variance and TV fail differently on purpose. Build a cube where one
    prefers a noisy slice and the other a smooth one, and the routine must
    decline rather than average them into a plausible answer."""
    x, n = 64, 11
    rng = np.random.default_rng(1)
    cube = np.zeros((n, 4, x, x), dtype=np.float32)
    iy, ix = np.mgrid[0:x, 0:x].astype(np.float64)
    r = np.hypot(ix - 31.5, iy - 31.5)
    disc = 1.0 / (1.0 + np.exp((r - 16) / 0.8))
    streaks = np.sin(ix * 2.1) * np.cos(iy * 1.7)
    for i in range(n):
        k = abs(i - 2)                       # the real optimum is i=2
        img = disc * (1.0 - 0.03 * k) + 0.04 * k * streaks
        # A burst of large-amplitude, LOW-frequency noise at i=9. Variance
        # loves it; it adds little gradient, so TV is indifferent. This is the
        # documented failure mode of variance, made to happen on purpose.
        if i == 9:
            coarse = rng.standard_normal((8, 8)).repeat(8, 0).repeat(8, 1)
            img = img + 1.5 * coarse[:x, :x]
        cube[i, :] = img
    c = find_center_consensus(cube, SWEEP)
    assert not c["trustworthy"]
    assert "total variation" in c["reason"] or "variance" in c["reason"]


def test_per_slice_disagreement_is_NOT_trustworthy():
    """Different slices picking different shifts means drift or too coarse a
    sweep, and the median would hide it."""
    x = 64
    cube = np.zeros((11, 4, x, x), dtype=np.float32)
    iy, ix = np.mgrid[0:x, 0:x].astype(np.float64)
    r = np.hypot(ix - 31.5, iy - 31.5)
    disc = 1.0 / (1.0 + np.exp((r - 16) / 0.8))
    streaks = np.sin(ix * 2.1) * np.cos(iy * 1.7)
    best_per_slice = [2, 3, 7, 8]     # interior, so the span rule fires
    for s, b in enumerate(best_per_slice):
        for i in range(11):
            k = abs(i - b)
            cube[i, s] = disc * (1.0 - 0.03 * k) + 0.04 * k * streaks
    c = find_center_consensus(cube, SWEEP)
    assert not c["trustworthy"]
    assert "per-slice picks span" in c["reason"]


def test_a_pick_at_the_sweep_EDGE_is_not_an_interior_optimum():
    """Measured on bt_1id_jun25b NMC811 s5: scoring four evenly spaced slices of a
    small specimen, two rows were empty and their sharpness curves had no
    interior optimum, so argmax returned the bottom of the sweep (-25.00 and
    -23.00). The two rows that did contain sample both found +13.00, the
    human's answer. An edge pick is not evidence and must not be averaged in.
    """
    x = 64
    cube = np.zeros((11, 4, x, x), dtype=np.float32)
    iy, ix = np.mgrid[0:x, 0:x].astype(np.float64)
    r = np.hypot(ix - 31.5, iy - 31.5)
    disc = 1.0 / (1.0 + np.exp((r - 16) / 0.8))
    streaks = np.sin(ix * 2.1) * np.cos(iy * 1.7)
    for s in range(4):
        for i in range(11):
            if s < 2:                      # rows with sample: optimum at i=5
                k = abs(i - 5)
                cube[i, s] = disc * (1.0 - 0.03 * k) + 0.04 * k * streaks
            else:                          # empty rows: monotonic to an edge
                cube[i, s] = 0.02 * i * streaks
    c = find_center_consensus(cube, SWEEP)
    assert not c["trustworthy"]
    assert "edge of the sweep" in c["reason"]
    assert c["per_method"]["variance"]["n_edge_picks"] >= 1


def test_slices_with_signal_finds_the_rows_that_contain_specimen():
    """The fix for the above: choose rows by attenuation, not by spacing."""
    from midas_tomo.center import slices_with_signal

    data = np.full((5, 20, 16), 1000.0)
    # A specimen occupies part of a row, not all of it -- which is the whole
    # reason the statistic is within-row contrast rather than the row mean.
    data[:, 8:12, 4:12] = 300.0
    dark = np.full((20, 16), 100.0)
    whites = np.full((2, 20, 16), 1000.0)
    got = slices_with_signal(data, dark, whites, k=3)
    assert got and all(8 <= s < 12 for s in got)


def test_slices_with_signal_refuses_when_nothing_absorbs():
    from midas_tomo.center import slices_with_signal

    data = np.full((5, 20, 16), 1000.0)
    dark = np.full((20, 16), 100.0)
    whites = np.full((2, 20, 16), 1000.0)
    with pytest.raises(ValueError, match="not in the field of view"):
        slices_with_signal(data, dark, whites, k=3)


def test_slices_with_signal_refuses_swapped_calibration_blocks():
    from midas_tomo.center import slices_with_signal

    data = np.full((5, 20, 16), 1000.0)
    dark = np.full((20, 16), 5000.0)            # darker than the white
    whites = np.full((2, 20, 16), 1000.0)
    with pytest.raises(ValueError, match="mis-assigned"):
        slices_with_signal(data, dark, whites, k=3)


def test_an_untrustworthy_result_still_reports_a_shift_for_inspection():
    cube = np.ones((11, 4, 64, 64), dtype=np.float32)
    c = find_center_consensus(cube, SWEEP)
    assert not c["trustworthy"]
    assert isinstance(c["best_shift"], float)   # populated, but not to be used


def test_a_non_4d_cube_is_refused():
    with pytest.raises(ValueError, match="must be 4-D"):
        find_center_consensus(np.zeros((4, 8, 8)), SWEEP)


def test_the_tolerance_is_adjustable():
    cube = _cube(best=5)
    strict = find_center_consensus(cube, SWEEP, tol=1e-9)
    assert not strict["trustworthy"] or strict["disagreement"] == 0.0


# ------------------------------------------------- the underlying criteria

def test_the_two_criteria_measure_different_things():
    """Guards the premise. If variance and TV ranked identically on every
    input, running both would be theatre and the agreement check worthless."""
    rng = np.random.default_rng(2)
    smooth = np.zeros((32, 32))
    smooth[8:24, 8:24] = 1.0
    noisy = smooth + rng.standard_normal((32, 32)) * 0.3
    assert sharpness(noisy, method="variance") > sharpness(smooth, method="variance")
    assert sharpness(noisy, method="tv") < sharpness(smooth, method="tv")


def test_find_center_still_works_on_its_own():
    r = find_center(_cube(best=7), SWEEP, method="variance")
    assert r["best_shift"] == pytest.approx(2.0, abs=1.0)
    assert r["well_determined"]


def test_an_UNILLUMINATED_row_must_not_score_as_the_strongest_signal():
    """The trap this cost a re-run to find, 2026-08-23.

    Outside the beam ``white - dark`` is ~0, so the transmission ratio is
    noise, the clip floor turns it into ``-log(1e-6) = 13.8``, and a row that
    sees no beam at all scores as the most absorbing row on the detector.
    Measured on the Ce scan: every one of 2048 rows passed a contrast filter
    and the rotation-axis fit still ran over ~700 rows of garbage.

    The illuminated region has to come from the flat field.
    """
    from midas_tomo.center import slices_with_signal

    ny, nx = 40, 32
    dark = np.full((ny, nx), 100.0)
    whites = np.full((2, ny, nx), 100.0)
    whites[:, 10:30, :] = 5000.0                 # beam lights rows 10-29 only
    data = np.full((5, ny, nx), 5000.0)
    data[:, 10:30, :] = 5000.0
    data[:, 15:20, 12:20] = 1500.0               # specimen, inside the beam
    data[:, :10, :] = 100.0                      # unlit rows: dark-level noise
    data[:, 30:, :] = 100.0

    got = slices_with_signal(data, dark, whites, k=4)
    assert got, "the specimen rows should be found"
    assert all(10 <= s < 30 for s in got), f"unlit row selected: {got}"


def test_a_fully_unilluminated_detector_is_refused():
    from midas_tomo.center import slices_with_signal

    data = np.full((5, 20, 16), 100.0)
    dark = np.full((20, 16), 100.0)
    whites = np.full((2, 20, 16), 100.5)
    with pytest.raises(ValueError):
        slices_with_signal(data, dark, whites, k=3)
