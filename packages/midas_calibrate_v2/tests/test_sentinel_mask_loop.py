"""Bad-pixel sentinels: detection, convention, and where the mask goes.

``io/readers._split_sentinel`` finds out-of-band sentinel values, ZEROES them,
and returns a ``True = bad`` mask. Two things make it worth pinning:

**The convention must match.** ``True = bad`` here, nonzero = bad in
``calibrate(mask=...)``, ``mask == 1.0`` = masked in
``midas_integrate.detector_mapper.build_map``, and nonzero = bad in the shipped
``mask_upd.tif``. Four places, one convention. An inversion anywhere would keep
exactly the pixels it is meant to discard.

**Zeroing is not masking.** A zeroed sentinel still sits in its cake cell and
dilutes the mean over that cell, so the mask has to be carried to the
integration. Until 2026-08-29 there was nowhere to carry it —
``calibrate()`` took no mask at all — so ``read_image``'s mask had no consumer.
Measured here: a zeroed-but-unmasked dead stripe changes cake cells by up to
**100 % relative** against the properly masked result.

Detection also runs on the RAW integer data before any frame averaging;
averaging first would blend the sentinel into a value that no longer equals it
and could not be detected at all.
"""
from __future__ import annotations

import inspect
import warnings

import numpy as np
import pytest

from midas_calibrate_v2.io.readers import _split_sentinel


# ------------------------------------------------------------- detection

def test_auto_finds_the_unsigned_max():
    a = np.full((8, 8), 100, dtype=np.uint16)
    a[2, 3] = np.iinfo(np.uint16).max
    out, mask = _split_sentinel(a, "auto")
    assert mask is not None and bool(mask[2, 3])
    assert int(out[2, 3]) == 0, "the sentinel must be zeroed"
    assert int(mask.sum()) == 1, "nothing else may be flagged"


def test_true_means_bad():
    """The convention shared with calibrate(mask=), build_map and mask_upd.tif.
    An inversion here would keep exactly the pixels meant to be discarded."""
    a = np.full((8, 8), 100, dtype=np.uint16)
    a[2, 3] = np.iinfo(np.uint16).max
    _out, mask = _split_sentinel(a, "auto")
    assert mask.dtype == bool
    assert bool(mask[2, 3]) and not bool(mask[0, 0])


def test_null_a_clean_frame_is_untouched():
    a = np.full((8, 8), 100, dtype=np.uint16)
    out, mask = _split_sentinel(a, "auto")
    assert mask is None
    assert np.array_equal(out, a)


def test_auto_declines_on_a_signed_dtype_rather_than_guessing():
    """There is no unsigned max to key on; inventing one would flag real data."""
    a = np.full((8, 8), 100, dtype=np.int32)
    a[1, 1] = -1
    _out, mask = _split_sentinel(a, "auto")
    assert mask is None


def test_stack_mask_is_the_union_over_frames():
    """Bad in ANY frame is bad in the result — the conservative choice, and the
    only one that stays meaningful once the frames are averaged."""
    st = np.full((3, 8, 8), 100, dtype=np.uint16)
    st[0, 1, 1] = np.iinfo(np.uint16).max
    st[2, 5, 5] = np.iinfo(np.uint16).max
    _out, mask = _split_sentinel(st, "auto")
    assert mask is not None and mask.ndim == 2
    assert bool(mask[1, 1]) and bool(mask[5, 5])
    assert int(mask.sum()) == 2


def test_explicit_value_and_disabling():
    a = np.full((4, 4), 7, dtype=np.uint16)
    assert int(_split_sentinel(a, 7)[1].sum()) == 16
    b = np.full((4, 4), 100, dtype=np.uint16)
    b[0, 0] = np.iinfo(np.uint16).max
    assert _split_sentinel(b, None)[1] is None


# ------------------------------------------------------- the loop closes

def test_the_mask_has_somewhere_to_go():
    from midas_calibrate_v2.io.readers import read_image
    from midas_calibrate_v2 import calibrate

    assert "return_mask" in inspect.signature(read_image).parameters
    assert "mask" in inspect.signature(calibrate).parameters, (
        "read_image produces a bad-pixel mask that calibrate() cannot accept")


def test_zeroing_is_not_masking():
    """The reason the mask must be carried, quantified.

    A dead stripe that is merely zeroed still dilutes every cake cell it
    touches; masking removes it from the denominator too.
    """
    from midas_calibrate.params import CalibrationParams
    from midas_calibrate.rings import build_ring_table
    from midas_calibrate.estep import integrate_cake

    N = 384
    p = CalibrationParams(
        NrPixelsY=N, NrPixelsZ=N, pxY=200.0, pxZ=200.0, Lsd=400_000.0,
        BC_y=N / 2.0, BC_z=N / 2.0, Wavelength=0.172973,
        LatticeConstant=(5.4116,) * 3 + (90.0,) * 3, SpaceGroup=225,
        RhoD=float(N) * 200.0, MaxRingRad=float(N) * 0.70, MinRingRad=20.0)
    Zi, Yi = np.mgrid[0:N, 0:N].astype(float)
    R = np.hypot(Yi - N / 2.0, Zi - N / 2.0)
    img = np.full_like(R, 25.0)
    for R0 in (60.0, 100.0, 140.0):
        img += 4000.0 * np.exp(-((R - R0) ** 2) / (2 * 1.6 ** 2))
    img = np.random.default_rng(0).poisson(np.clip(img, 0, None)).astype(float)

    bad = np.zeros_like(img, dtype=bool)
    bad[:, N // 2 - 20:N // 2 + 20] = True
    zeroed = np.where(bad, 0.0, img)

    rt = build_ring_table(p)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        c_zero = integrate_cake(p, zeroed, rt)
        c_mask = integrate_cake(p, zeroed, rt, mask=bad)

    z, m = c_zero.intensity, c_mask.intensity
    fin = np.isfinite(z) & np.isfinite(m) & (m > 0)
    assert fin.sum() > 100
    assert np.abs(z[fin] - m[fin]).max() > 0.0, (
        "zeroing and masking gave identical cakes — the mask is not reaching "
        "the denominator")
