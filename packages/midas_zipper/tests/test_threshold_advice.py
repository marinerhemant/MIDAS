"""-preProcThresh is advised against RingThresh (one list of rows, or a single row)."""

from __future__ import annotations

from midas_zipper.ff_zip import threshold_advice

RINGS = [[1, 30], [2, 30], [3, 20]]


def test_no_threshold_no_advice():
    assert threshold_advice({'RingThresh': RINGS}) == []


def test_below_every_ringthresh_is_quiet():
    assert threshold_advice({'preProcThresh': 19, 'RingThresh': RINGS}) == []


def test_equal_to_the_smallest_ringthresh_warns():
    # a float dark mean in the GE/HDF5 readers makes T == RingThresh lossy
    (msg,) = threshold_advice({'preProcThresh': 20, 'RingThresh': RINGS})
    assert "T <= 19" in msg


def test_single_ringthresh_row_is_a_flat_list():
    assert threshold_advice({'preProcThresh': 10, 'RingThresh': [1, 30]}) == []
    assert len(threshold_advice({'preProcThresh': 30, 'RingThresh': [1, 30]})) == 1


def test_no_ringthresh_in_params_is_quiet():
    assert threshold_advice({'preProcThresh': 50}) == []


def test_background_and_snr_cuts_warn():
    assert len(threshold_advice({'preProcThresh': 5, 'RingThresh': RINGS, 'BgSubtract': 1})) == 1
    assert len(threshold_advice({'preProcThresh': 5, 'RingThresh': RINGS, 'MinPeakSNR': 3.0})) == 1
