"""A parameter file with RBinSize 1 gave 20.1 uE where 0.25 px gave 7.30 uE on the Varex frame (si_rerun/A)."""
from midas_calibrate_v2.pipelines.single_pv import CoarseCakeBinWarning, _coarse_cake_bin_message


def test_coarse_bin_warns():
    m = _coarse_cake_bin_message(1.0)
    assert m and "RBinSize=1" in m and "0.25" in m


def test_fine_or_unset_bin_is_silent():
    assert _coarse_cake_bin_message(0.25) is None
    assert _coarse_cake_bin_message(0.5) is None
    assert _coarse_cake_bin_message(0.0) is None


def test_is_a_userwarning():
    assert issubclass(CoarseCakeBinWarning, UserWarning)
