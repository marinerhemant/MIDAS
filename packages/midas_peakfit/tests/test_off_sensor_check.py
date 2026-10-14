"""Bounds check on reported peak positions (YCen / ZCen on the real sensor)."""
import numpy as np
import pytest

from midas_peakfit.postfit import N_PEAK_COLS, off_sensor_mask, sensor_extent


def _rows(yz):
    r = np.zeros((len(yz), N_PEAK_COLS))
    r[:, 3] = [a for a, _ in yz]
    r[:, 4] = [b for _, b in yz]
    return r


@pytest.mark.parametrize("opts,expect", [
    ([], (1028, 512)), ([1], (1028, 512)), ([2], (1028, 512)),
    ([3], (512, 1028)), ([3, 3], (1028, 512)), ([3, 2, 1], (512, 1028)),
])
def test_sensor_extent_swaps_per_transpose(opts, expect):
    assert sensor_extent(1028, 512, opts) == expect


def test_on_sensor_rows_pass_including_edge_overshoot():
    r = _rows([(0.0, 0.0), (1027.0, 511.0), (-0.4, 511.6), (500.0, 100.0)])
    assert not off_sensor_mask(r, 1028, 512).any()


def test_off_sensor_rows_flagged():
    r = _rows([(300.0, 927.0), (300.0, 411.0), (-3.0, 10.0), (1040.0, 10.0)])
    np.testing.assert_array_equal(
        off_sensor_mask(r, 1028, 512), [True, False, True, True])


def test_empty_rows():
    assert off_sensor_mask(np.zeros((0, N_PEAK_COLS)), 8, 8).shape == (0,)


def test_old_square_flip_would_have_been_flagged():
    """The reported failure: z=100 flipped about the padded square gives 927."""
    NY, NZ, N = 1028, 512, 1028
    z_old = N - 1 - 100
    z_new = NZ - 1 - 100
    assert off_sensor_mask(_rows([(300.0, z_old)]), NY, NZ).all()
    assert not off_sensor_mask(_rows([(300.0, z_new)]), NY, NZ).any()
