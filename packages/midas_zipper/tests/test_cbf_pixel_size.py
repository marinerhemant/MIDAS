"""CBF header pixel size: single value, square pair, and the non-square pair that the FF stack cannot represent."""
import warnings

import pytest

from midas_zipper.ff_zip import _cbf_pixel_um


def test_single_value_and_square_pair():
    assert _cbf_pixel_um(172e-6) == pytest.approx(172.0)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert _cbf_pixel_um((172e-6, 172e-6)) == pytest.approx(172.0)


def test_non_square_pair_warns_and_keeps_first_axis():
    with pytest.warns(UserWarning, match="non-square"):
        assert _cbf_pixel_um((172e-6, 200e-6)) == pytest.approx(172.0)
