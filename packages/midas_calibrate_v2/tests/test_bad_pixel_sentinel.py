"""Bad-pixel sentinels must never reach a fitter as counts.

Pilatus marks gaps with -1/-2, so ``img[img < 0] = 0`` catches them.  Dectris
EIGER marks them with the largest representable unsigned value (2**32-1 for
uint32) — the opposite end of the range, where every ``< 0`` guard fails open.
"""
import numpy as np
import pytest

from midas_calibrate_v2.io.readers import (
    read_image, BadPixelSentinelWarning, _split_sentinel)

SENT32 = 2 ** 32 - 1


def _write_h5(tmp_path, data, name="f.h5"):
    h5py = pytest.importorskip("h5py")
    p = tmp_path / name
    with h5py.File(p, "w") as f:
        f.create_dataset("exchange/data", data=data)
    return p


def test_uint32_sentinel_is_zeroed_and_warned(tmp_path):
    a = np.full((1, 8, 8), 5, dtype=np.uint32)
    a[0, 2, 3] = SENT32
    a[0, 6, 1] = SENT32
    p = _write_h5(tmp_path, a)

    with pytest.warns(BadPixelSentinelWarning, match="bad-pixel sentinel"):
        img, mask = read_image(p, return_mask=True)

    assert img.max() == 5.0, "the sentinel must not survive as a count"
    assert img[2, 3] == 0.0 and img[6, 1] == 0.0
    assert mask.dtype == bool and mask.sum() == 2
    assert mask[2, 3] and mask[6, 1]


def test_sentinel_handled_before_the_frame_average(tmp_path):
    """The averaging in _read_hdf5 is the trap: blend 2**32-1 with real counts
    and the result no longer equals the sentinel, so it can never be found."""
    a = np.full((4, 4, 4), 10, dtype=np.uint32)
    a[0, 1, 1] = SENT32          # bad in ONE frame of four
    p = _write_h5(tmp_path, a)

    with pytest.warns(BadPixelSentinelWarning):
        img, mask = read_image(p, return_mask=True)

    # naive behaviour would give (3*10 + 4294967295)/4 ≈ 1.07e9
    assert img[1, 1] < 10.0, "sentinel leaked into the mean"
    assert mask[1, 1], "a pixel bad in any frame must be masked"
    assert mask.sum() == 1


def test_disabled_by_bad_value_none(tmp_path):
    a = np.full((1, 4, 4), 7, dtype=np.uint32)
    a[0, 0, 0] = SENT32
    p = _write_h5(tmp_path, a)

    import warnings as _w
    with _w.catch_warnings():
        _w.simplefilter("error", BadPixelSentinelWarning)
        img = read_image(p, bad_value=None)          # must not warn
    assert img[0, 0] == float(SENT32)


def test_explicit_sentinel_value(tmp_path):
    """A named sentinel works for signed data too — e.g. a Pilatus -1 gap."""
    a = np.full((1, 4, 4), 3, dtype=np.int32)
    a[0, 2, 2] = -1
    p = _write_h5(tmp_path, a)

    with pytest.warns(BadPixelSentinelWarning):
        img, mask = read_image(p, bad_value=-1, return_mask=True)
    assert img[2, 2] == 0.0 and mask[2, 2]


def test_signed_data_untouched_by_auto(tmp_path):
    """auto only claims the unsigned dtype-max; it must not invent sentinels
    in signed or float data, where the < 0 convention already applies."""
    a = np.full((1, 4, 4), 3, dtype=np.int32)
    a[0, 1, 0] = -1
    a[0, 1, 1] = np.iinfo(np.int32).max
    p = _write_h5(tmp_path, a)

    import warnings as _w
    with _w.catch_warnings():
        _w.simplefilter("error", BadPixelSentinelWarning)
        img = read_image(p)
    assert img[1, 0] == -1.0 and img[1, 1] == float(np.iinfo(np.int32).max)


def test_clean_frame_gives_all_false_mask_and_no_warning(tmp_path):
    a = np.full((1, 5, 5), 42, dtype=np.uint16)
    p = _write_h5(tmp_path, a)

    import warnings as _w
    with _w.catch_warnings():
        _w.simplefilter("error", BadPixelSentinelWarning)
        img, mask = read_image(p, return_mask=True)
    assert img.shape == (5, 5) and mask.shape == (5, 5)
    assert not mask.any()


def test_mask_follows_im_trans(tmp_path):
    """A mask that does not take the same flips as the image is worse than no
    mask: it silently masks the wrong pixels."""
    a = np.full((1, 4, 6), 2, dtype=np.uint32)
    a[0, 0, 0] = SENT32
    p = _write_h5(tmp_path, a)

    with pytest.warns(BadPixelSentinelWarning):
        img, mask = read_image(p, im_trans=(1, 2), return_mask=True)

    assert img.shape == mask.shape
    bad = np.argwhere(mask)
    assert len(bad) == 1
    assert img[tuple(bad[0])] == 0.0
    assert bad[0].tolist() == [3, 5], "flip Y then Z should move (0,0) to (3,5)"


def test_split_sentinel_returns_none_when_clean():
    a = np.arange(16, dtype=np.uint16).reshape(4, 4)
    out, mask = _split_sentinel(a, "auto")
    assert mask is None
    assert out is a
