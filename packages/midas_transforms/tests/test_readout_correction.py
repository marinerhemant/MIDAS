"""Detector-readout omega timing (OmegaReadoutShift/Axis/Shape/Centre): off by default,
centre-out and split-halves shapes on either axis, additive to the legacy tGap row correction."""
import numpy as np
import pytest

from midas_transforms.fit_setup.core import _omega_correction, _readout_correction
from midas_transforms.params import ZarrParams


def _spots(y, z, ome=10.0):
    s = np.zeros((len(y), 12)); s[:, 1] = ome; s[:, 2] = y; s[:, 3] = z
    return s


def test_off_by_default_is_a_noop():
    s = _spots([0, 1440, 2879], [5, 6, 7])
    out = _readout_correction(s.copy(), shift=0.0, axis=1, shape=0, centre=0, nr_pixels_y=2880, nr_pixels_z=2880)
    np.testing.assert_array_equal(out, s)
    p = ZarrParams()
    assert p.OmegaReadoutShift == 0.0 and p.OmegaReadoutAxis == 0 and p.OmegaReadoutShape == 0


def test_centre_out_along_columns():
    s = _spots([1440.0, 0.0, 2880.0, 720.0], [100.0] * 4)
    out = _readout_correction(s, shift=-0.136, axis=1, shape=0, centre=0, nr_pixels_y=2880, nr_pixels_z=2880)
    # omega -= shift * (1 - |x - 1440| / 1440): centre +0.136, edges 0, quarter +0.068
    np.testing.assert_allclose(out[:, 1], [10.136, 10.0, 10.0, 10.068], atol=1e-12)


def test_split_halves_along_columns_with_centre():
    s = _spots([1000.0, 1600.0], [0.0, 0.0])
    out = _readout_correction(s, shift=0.026, axis=1, shape=1, centre=1451.0, nr_pixels_y=2880, nr_pixels_z=2880)
    np.testing.assert_allclose(out[:, 1], [10.026, 9.974], atol=1e-12)


def test_rows_axis_uses_zcen():
    s = _spots([0.0, 0.0], [1440.0, 0.0])
    out = _readout_correction(s, shift=0.1, axis=0, shape=0, centre=0, nr_pixels_y=2880, nr_pixels_z=2880)
    np.testing.assert_allclose(out[:, 1], [9.9, 10.0], atol=1e-12)


def test_wraps_and_validates():
    s = _spots([1440.0], [0.0], ome=179.95)
    out = _readout_correction(s, shift=-0.136, axis=1, shape=0, centre=0, nr_pixels_y=2880, nr_pixels_z=2880)
    assert -180 <= out[0, 1] <= -179.9
    with pytest.raises(ValueError):
        _readout_correction(_spots([1.0], [1.0]), shift=0.1, axis=2, shape=0, centre=0,
                            nr_pixels_y=10, nr_pixels_z=10)


def test_additive_to_legacy_tgap_row_correction():
    s = _spots([1440.0], [1440.0])
    a = _omega_correction(s.copy(), t_gap=0.1, t_int=0.1, omega_step=0.25, nr_pixels=2880)
    b = _readout_correction(a.copy(), shift=-0.136, axis=1, shape=0, centre=0, nr_pixels_y=2880, nr_pixels_z=2880)
    np.testing.assert_allclose(b[0, 1], 10.0 - 0.125 + 0.136, atol=1e-12)
