"""Unit tests for the powder → FF-parameter-file pipeline helpers.

These cover the four silent failures the pipeline exists to prevent; each one
produced a plausible, wrong answer on real 20-ID Varex data before it was
guarded.
"""
import math

import numpy as np
import pytest

from midas_calibrate_v2.pipelines.ff_calibrate import (
    _check_not_mirrored,
    _im_trans_from_template,
    load_calibrant_frame,
    rho_d_for,
)

h5py = pytest.importorskip("h5py")


class _V1:
    """Stand-in for CalibrationParams."""
    def __init__(self, extra=None, BC_y=0.0, BC_z=0.0):
        self.extra = extra or {}
        self.BC_y = BC_y
        self.BC_z = BC_z


def _write(tmp_path, text, name="ps.txt"):
    p = tmp_path / name
    p.write_text(text)
    return p


# ── ImTransOpt resolution ────────────────────────────────────────────────
def test_im_trans_from_extra(tmp_path):
    ps = _write(tmp_path, "ImTransOpt 2\n")
    assert _im_trans_from_template(_V1(extra={"ImTransOpt": 2}), ps) == (2,)


def test_im_trans_falls_back_to_raw_text(tmp_path):
    """CalibrationParams has no ImTransOpt attribute and may not stash it in
    .extra either. Missing it runs the fit with NO transform, which mirrors an
    axis and still converges — so the raw text is the backstop."""
    ps = _write(tmp_path, "Lsd 900000\nImTransOpt 2\nBC 1450 1467\n")
    assert _im_trans_from_template(_V1(), ps) == (2,)


def test_im_trans_multi_valued(tmp_path):
    ps = _write(tmp_path, "ImTransOpt 1 3\n")
    assert _im_trans_from_template(_V1(), ps) == (1, 3)


def test_im_trans_zero_and_absent_mean_no_transform(tmp_path):
    assert _im_trans_from_template(_V1(), _write(tmp_path, "ImTransOpt 0\n")) == ()
    assert _im_trans_from_template(_V1(), _write(tmp_path, "Lsd 900000\n")) == ()


# ── RhoD ─────────────────────────────────────────────────────────────────
def test_rho_d_is_corner_distance_in_micrometres():
    # beam centre at the middle of a 2880 detector, 150 um pixels
    got = rho_d_for(1439.5, 1439.5, 2880, 2880, 150.0)
    assert got == pytest.approx(150.0 * math.hypot(1439.5, 1439.5))


def test_rho_d_uses_the_farthest_corner():
    """An off-centre beam centre must give the distance to the FAR corner, not
    the near one — the polynomial is normalised over the whole detector."""
    off = rho_d_for(400.0, 400.0, 2880, 2880, 150.0)
    assert off == pytest.approx(150.0 * math.hypot(2479.0, 2479.0))
    assert off > rho_d_for(1439.5, 1439.5, 2880, 2880, 150.0)


def test_rho_d_is_nowhere_near_the_two_million_that_broke_indexing():
    """Regression on the value that overran the indexer's 500-ring array."""
    assert rho_d_for(1450.8, 1467.5, 2880, 2880, 150.0) < 4e5


# ── mirror detection ─────────────────────────────────────────────────────
class _Res:
    def __init__(self, BC_y, BC_z):
        self.BC_y = BC_y
        self.BC_z = BC_z


def test_mirror_detected_on_both_axes():
    v1 = _V1(BC_y=1450.805, BC_z=1467.408)
    # what a wrong ImTransOpt actually produced: N-1-BC on both axes
    res = _Res(2879 - 1450.805, 2879 - 1467.408)
    msgs = _check_not_mirrored(res, v1, 2880, 2880)
    assert len(msgs) == 2
    assert "BC_y" in msgs[0] and "BC_z" in msgs[1]


def test_no_mirror_warning_for_a_normal_refinement():
    v1 = _V1(BC_y=1450.805, BC_z=1467.408)
    res = _Res(1450.855, 1467.456)          # the real refined values
    assert _check_not_mirrored(res, v1, 2880, 2880) == []


def test_mirror_check_ignores_a_centred_beam():
    """At the exact centre BC and its mirror coincide, so the test cannot
    discriminate and must not cry wolf."""
    v1 = _V1(BC_y=1439.5, BC_z=1439.5)
    assert _check_not_mirrored(_Res(1439.5, 1439.5), v1, 2880, 2880) == []


# ── frame loading ────────────────────────────────────────────────────────
def _make_h5(path, data, **groups):
    with h5py.File(path, "w") as f:
        f.create_dataset("exchange/data", data=data)
        for k, v in groups.items():
            f.create_dataset(f"exchange/{k}", data=v)
        f.create_group("WM")            # the group the generic loader grabs


def test_median_reduces_frames_and_subtracts_the_dark(tmp_path):
    p = tmp_path / "c.h5"
    data = np.full((5, 4, 4), 100.0)
    data[0] = 9999.0                                   # a zinger the median kills
    _make_h5(p, data, bright=np.full((3, 4, 4), 30.0))
    img = load_calibrant_frame(p, dark_group="exchange/bright")
    assert np.allclose(img, 70.0)


def test_all_zero_dark_raises_rather_than_silently_doing_nothing(tmp_path):
    """/exchange/dark exists on these files and is all zeros; the real dark is
    /exchange/bright. Subtracting the zeros leaves the pedestal in."""
    p = tmp_path / "c.h5"
    _make_h5(p, np.full((3, 4, 4), 100.0), dark=np.zeros((1, 4, 4)))
    with pytest.raises(ValueError, match="all zeros"):
        load_calibrant_frame(p, dark_group="exchange/dark")


def test_missing_group_names_the_available_keys(tmp_path):
    p = tmp_path / "c.h5"
    _make_h5(p, np.full((2, 4, 4), 5.0))
    with pytest.raises(KeyError, match="no dataset"):
        load_calibrant_frame(p, data_group="exchange/nope")


def test_negative_values_are_clipped(tmp_path):
    p = tmp_path / "c.h5"
    _make_h5(p, np.full((2, 4, 4), 10.0), bright=np.full((2, 4, 4), 40.0))
    assert np.all(load_calibrant_frame(p, dark_group="exchange/bright") == 0.0)
