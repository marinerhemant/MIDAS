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


# ── raw GE binary calibrant ──────────────────────────────────────────────
# At 1-ID the calibrant is very often the raw GE file the detector wrote,
# `<stem>_NNNNNN.edf.ge5`, and never converted. load_calibrant_frame used to be
# an unconditional h5py.File(), so `--mode ff` on one died -- and the way round
# it (calling calibrate() directly) skips the RhoD rewrite that --mode ff
# exists for.
def _write_ge(path, stack, header=8192):
    """A GE binary: `header` zero bytes, then N x side x side uint16."""
    with open(path, "wb") as f:
        f.write(b"\0" * header)
        np.ascontiguousarray(stack, dtype=np.uint16).tofile(f)
    return path


def test_raw_ge_calibrant_loads_without_hdf5(tmp_path):
    # 3 frames, not 4: 4 x 512^2 is bit-for-bit the same length as 1 x 1024^2,
    # and _read_ge guesses the side from the length (2048, 4096, 1024, 512),
    # so it would legitimately read that as one 1024^2 frame.
    stack = np.full((3, 512, 512), 100, dtype=np.uint16)
    p = _write_ge(tmp_path / "cal_000099.edf.ge5", stack)
    img = load_calibrant_frame(p)
    assert img.shape == (512, 512)
    assert img.dtype == np.float64
    assert np.allclose(img, 100.0)


def test_raw_ge_default_reduce_is_the_median_that_kills_a_zinger(tmp_path):
    """The documented default is median. read_image's GE reader hard-coded a
    mean, which spreads one zinger over the ring it landed on and drags that
    ring's centroid -- the whole reason the default is median."""
    stack = np.full((5, 512, 512), 100, dtype=np.uint16)
    stack[2, 40, 40] = 60000
    p = _write_ge(tmp_path / "cal_000099.edf.ge5", stack)
    assert load_calibrant_frame(p)[40, 40] == pytest.approx(100.0)
    assert load_calibrant_frame(p, reduce="mean")[40, 40] == pytest.approx(
        (4 * 100 + 60000) / 5.0)


def test_raw_ge_honours_skip_frame(tmp_path):
    stack = np.stack([np.full((512, 512), v, dtype=np.uint16)
                      for v in (1, 7, 7)])
    p = _write_ge(tmp_path / "cal_000099.edf.ge5", stack)
    assert load_calibrant_frame(p, skip_frame=1)[0, 0] == pytest.approx(7.0)


def test_raw_ge_with_a_dark_group_raises_instead_of_ignoring_it(tmp_path):
    """A raw file has no groups. Ignoring --dark-group would leave the
    pedestal in and nothing downstream would say so."""
    p = _write_ge(tmp_path / "cal_000099.edf.ge5",
                  np.full((2, 512, 512), 100, dtype=np.uint16))
    with pytest.raises(ValueError, match="not an HDF5 file"):
        load_calibrant_frame(p, dark_group="exchange/dark")


def test_hdf5_dispatch_is_by_extension_not_by_content(tmp_path):
    """`.ge5.h5` (DM-converted) must still take the HDF5 branch; `.edf.ge5`
    must not. The two differ only in the suffix."""
    p = tmp_path / "cal.ge5.h5"
    _make_h5(p, np.full((2, 4, 4), 42.0))
    assert np.allclose(load_calibrant_frame(p), 42.0)


def test_read_image_frame_reduce_default_is_unchanged(tmp_path):
    """Adding frame_reduce must not move any existing caller: the default is
    still the mean the GE and HDF5 readers always took."""
    from midas_calibrate_v2.io.readers import read_image

    stack = np.full((3, 512, 512), 10, dtype=np.uint16)
    stack[0, 5, 5] = 1000
    p = _write_ge(tmp_path / "x_000001.ge5", stack)
    assert read_image(p)[5, 5] == pytest.approx((1000 + 10 + 10) / 3.0)
    with pytest.raises(ValueError, match="frame_reduce"):
        read_image(p, frame_reduce="mediann")
