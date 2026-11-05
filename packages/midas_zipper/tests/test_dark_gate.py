"""The pre-threshold gate is measured from the integer dark the subtraction removes."""

from __future__ import annotations

import numpy as np

from midas_zipper.ff_zip import apply_correction, gate_base


def _run(img, dark, T):
    img = np.asarray(img, dtype=np.uint16).reshape(1, 1, -1)
    dark = np.asarray(dark).reshape(1, -1)
    thr = (gate_base(dark) + T) if T != -1 else gate_base(dark)
    return apply_correction(img, dark, thr).reshape(-1)


def test_float_dark_keeps_a_pixel_exactly_at_the_threshold():
    # dark mean 100.7 -> the subtraction removes 100. A pixel of 105 has corrected value 5 == T and must be kept
    # (before the fix the gate was 105.7 and zeroed it); 104 has corrected value 4 < T and is zeroed.
    out = _run([104, 105, 106], [100.7, 100.7, 100.7], 5)
    assert out.tolist() == [0, 5, 6]


def test_integer_dark_is_unchanged():
    out = _run([104, 105, 106], np.array([100, 100, 100], dtype=np.uint16), 5)
    assert out.tolist() == [0, 5, 6]


def test_float_dark_with_whole_value_is_unchanged():
    assert _run([104, 105, 106], [100.0] * 3, 5).tolist() == [0, 5, 6]


def test_no_threshold_matches_the_old_gate_on_random_data():
    rng = np.random.default_rng(0)
    img = rng.integers(0, 300, size=(2, 6, 50)).astype(np.uint16)
    dark = rng.uniform(80, 120, size=(6, 50))
    for T in (-1, 0):
        old = apply_correction(img, dark, dark + (T if T != -1 else 0))
        new = apply_correction(img, dark, gate_base(dark) + (T if T != -1 else 0))
        assert (old == new).all()


def test_gate_agrees_with_the_subtraction():
    rng = np.random.default_rng(1)
    img = rng.integers(0, 300, size=(2, 6, 50)).astype(np.uint16)
    dark = rng.uniform(80, 120, size=(6, 50))
    for T in (1, 5, 20):
        out = apply_correction(img, dark, gate_base(dark) + T)
        corrected = img.astype(int) - np.trunc(dark).astype(int)
        assert (out[corrected >= T] == corrected[corrected >= T]).all()
        assert (out[corrected < T] == 0).all()
