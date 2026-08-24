"""The detector mask has to arrive in the C forward model's frame, bit for bit.

Everything here is anchored to two pieces of C that are not going to change to
suit us:

``midas_ckernel/c_src/forward.c:50`` (and its two byte-identical copies)::

    #define TestBit(A, k) (A[(k / 32)] & (1 << (k % 32)))

``midas_ckernel/c_src/forward.c:183-189``::

    YCInt = (int)floor((big_det_size / 2) - (int)(-yl / pixelsize));
    ZCInt = (int)floor((int)(zl / pixelsize) + (big_det_size / 2));
    idx   = (long long int)(YCInt + big_det_size * ZCInt);
    if (!TestBit(bigdet->mask, idx)) KeepSpot = 0;

Note the polarity: a **set** bit keeps the spot. So the bitset is an
*active-area* map, the inverse of ``exchange/mask``, where non-zero means bad.
Getting that backwards would mask the detector and pass the entire sky, which
is why :func:`test_polarity_is_active_not_masked` exists.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from midas_transforms.geometry.detector_mask import (
    BIGDET_WORD_BITS,
    bigdet_cell_index,
    build_active_area_bitset,
    pack_bitset,
    write_big_detector_mask,
)

# A plain geometry: no tilt, no distortion, beam centre in the middle. Chosen
# so the pixel -> lab map is analytic and the tests assert against arithmetic
# rather than against the implementation.
PX = 150.0
NPX = 256
GEOM = dict(
    Lsd=1_000_000.0, BC_y=NPX / 2.0, BC_z=NPX / 2.0,
    px=PX, rho_d=200.0, parallax=0.0,
)


def c_test_bit(bitset: np.ndarray, k: int) -> bool:
    """Reference ``TestBit``, transcribed rather than reused."""
    return bool(int(bitset[k // BIGDET_WORD_BITS]) & (1 << (k % BIGDET_WORD_BITS)))


def c_cell_index(yl: float, zl: float, s: int, px: float) -> tuple[int, int]:
    """Reference of the C index arithmetic, using Python ints and math.trunc.

    C casts double->int by truncating toward zero, and ``s / 2`` is integer
    division. ``math.floor`` on the already-integral outer expression is a
    no-op and is reproduced only for fidelity.
    """
    half = s // 2
    yc = int(math.floor(half - int(math.trunc(-yl / px))))
    zc = int(math.floor(int(math.trunc(zl / px)) + half))
    return yc, zc


# --------------------------------------------------------------------------
# index arithmetic
# --------------------------------------------------------------------------

@pytest.mark.parametrize("s", [64, 4200])
def test_cell_index_matches_the_c_arithmetic(s):
    rng = np.random.default_rng(0)
    yl = rng.uniform(-(s // 2 - 3) * PX, (s // 2 - 3) * PX, 500)
    zl = rng.uniform(-(s // 2 - 3) * PX, (s // 2 - 3) * PX, 500)
    yc, zc = bigdet_cell_index(yl, zl, s, PX)
    for i in range(yl.size):
        eyc, ezc = c_cell_index(float(yl[i]), float(zl[i]), s, PX)
        assert (int(yc[i]), int(zc[i])) == (eyc, ezc), f"row {i}"


def test_truncation_is_toward_zero_not_floor():
    """The distinction only shows on sub-cell negative offsets.

    ``(int)(-yl/px)`` truncates toward zero. Had we used ``np.floor`` the
    result would differ by one cell for every spot whose offset from centre
    falls in the open interval (-1, 0) cells -- a one-pixel systematic shift
    of the whole mask on one side of the beam centre only, which is precisely
    the kind of error that survives a "looks about right" inspection.
    """
    s = 64
    # +/- a third of a pixel either side of centre: both must land on `half`.
    for yl in (0.3 * PX, -0.3 * PX):
        yc, _ = bigdet_cell_index(np.array([yl]), np.array([0.0]), s, PX)
        assert int(yc[0]) == s // 2, f"yl={yl} should stay in the centre cell"
    # and floor() would have disagreed on the negative side
    assert int(np.floor(-0.3)) != int(np.trunc(-0.3))


# --------------------------------------------------------------------------
# bit packing
# --------------------------------------------------------------------------

def test_pack_bitset_roundtrips_through_the_c_testbit():
    rng = np.random.default_rng(7)
    s = 40
    grid = rng.random((s, s)) < 0.3
    bits = pack_bitset(grid)
    assert bits.dtype == np.uint32
    assert bits.size == (s * s) // BIGDET_WORD_BITS + 1, "matches FitUnified.c:1470-1475"
    for z in range(s):
        for y in range(s):
            assert c_test_bit(bits, y + s * z) == bool(grid[z, y]), (
                f"bit {y + s * z} disagrees at (z={z}, y={y}) -- the ravel "
                "order must be [ZCInt][YCInt] to match idx = YCInt + S*ZCInt"
            )


def test_pack_bitset_uses_the_full_word_including_bit_31():
    """``1 << 31`` is the sign bit; a naive int32 build drops or corrupts it."""
    s = 8  # 64 cells -> exactly two full words
    grid = np.zeros((s, s), dtype=bool)
    grid[3, 7] = True                      # bit 31
    bits = pack_bitset(grid)
    assert c_test_bit(bits, 31)
    assert int(bits[0]) == 0x8000_0000


# --------------------------------------------------------------------------
# polarity and the wedge positive control
# --------------------------------------------------------------------------

def _wedge_mask(n: int, half_width_deg: float) -> np.ndarray:
    """Mark BAD every pixel whose azimuth lies within +/-half_width of eta=0.

    Built in the same ``eta = atan2(-Y', Z')`` convention the geometry uses so
    the expected solid angle is exactly ``2*half_width/360``.
    """
    z, y = np.mgrid[0:n, 0:n].astype(np.float64)
    yc = (-y + n / 2.0)
    zc = (z - n / 2.0)
    eta = np.degrees(np.arctan2(-yc, zc))
    return (np.abs(eta) <= half_width_deg).astype(np.uint8)


def test_polarity_is_active_not_masked():
    """A bad pixel must clear its bit; a good pixel must set it."""
    mask = np.zeros((NPX, NPX), dtype=np.uint8)
    mask[NPX // 2 + 10, NPX // 2 + 10] = 1     # one bad pixel, [Z, Y]
    bits, s, stats = build_active_area_bitset(
        mask, dilate_masked=0, off_detector="keep", **GEOM
    )
    assert stats["n_bad_pixels"] == 1
    # One pixel claims 1-4 cells: rasterisation fills the rectangle spanned by
    # its four mapped corners, so a pixel straddling a cell boundary -- which
    # is the *normal* case for an identity-like geometry, where centres land
    # exactly on boundaries -- covers 4. Over-covering the masked set is the
    # conservative direction.
    assert 1 <= stats["n_cells_masked"] <= 4
    assert stats["max_pixel_cell_span"] <= 1

    # the masked cell is clear, a neighbouring good cell is set
    from midas_transforms.fit_setup.transform import apply_tilt_distortion
    import torch
    dt = torch.float64
    g = {k: torch.tensor(float(v), dtype=dt) for k, v in GEOM.items()}
    g["p_coeffs"] = torch.zeros(15, dtype=dt)
    g["tx"] = g["ty"] = g["tz"] = torch.tensor(0.0, dtype=dt)
    for (zpix, ypix), want_set in (
        ((NPX // 2 + 10, NPX // 2 + 10), False),
        ((NPX // 2 + 10, NPX // 2 + 30), True),
    ):
        yl, zl = apply_tilt_distortion(
            torch.tensor([float(ypix)], dtype=dt),
            torch.tensor([float(zpix)], dtype=dt), **g)
        yc, zc = bigdet_cell_index(yl.numpy(), zl.numpy(), s, PX)
        assert c_test_bit(bits, int(yc[0]) + s * int(zc[0])) is want_set, (
            f"pixel (z={zpix}, y={ypix}) polarity wrong"
        )


@pytest.mark.parametrize("half_width_deg", [15.0, 30.0])
def test_wedge_positive_control(half_width_deg):
    """The plan's gate: mask a known azimuthal wedge, count what it removes.

    A mask that drops *zero* predicted spots is a failed push-forward, not a
    clean detector -- so this asserts a specific non-zero fraction, matched to
    the wedge's own solid angle.
    """
    mask = _wedge_mask(NPX, half_width_deg)
    bits, s, stats = build_active_area_bitset(
        mask, dilate_masked=0, off_detector="keep", **GEOM
    )

    # Predicted spots on one ring, uniformly in eta, exactly as forward.c
    # places them: yl = -R sin(eta), zl = R cos(eta).
    r_um = 60.0 * PX
    eta = np.linspace(-180.0, 180.0, 7201, endpoint=False)
    yl = -r_um * np.sin(np.radians(eta))
    zl = r_um * np.cos(np.radians(eta))
    yc, zc = bigdet_cell_index(yl, zl, s, PX)
    kept = np.array([c_test_bit(bits, int(a) + s * int(b))
                     for a, b in zip(yc, zc)])

    dropped_frac = 1.0 - kept.mean()
    expect = 2.0 * half_width_deg / 360.0
    assert dropped_frac > 0.0, "the mask removed nothing -- push-forward failed"
    assert abs(dropped_frac - expect) < 0.02, (
        f"dropped {dropped_frac:.4f} of the ring, wedge predicts {expect:.4f}"
    )


def test_pixel_to_cell_map_loses_no_cells_to_the_truncation_knife_edge():
    """Regression: mapping pixel CENTRES silently dropped whole cells.

    ``apply_tilt_distortion`` reaches the lab frame through a polar round trip
    (sqrt -> atan2 -> sin/cos). For an identity-like geometry a pixel centre
    should land exactly on an integer multiple of ``px``, but it arrives at the
    integer +/- ~1e-13 -- and ``(int)`` truncation toward zero then throws the
    value a whole cell whenever it lands a hair low.

    Measured before the fix, on this 256x256 geometry: 9388 masked pixels
    collapsed onto 7652 distinct cells, and a 30 deg wedge that must remove
    1/6 of a ring removed 0.1325. Corner rasterisation removes the knife edge
    because a pixel on a boundary claims the cells on both sides.

    The bijection reference is built with exact integer arithmetic, which is
    what the polar round trip is *trying* to reproduce.
    """
    mask = _wedge_mask(NPX, 30.0)
    _, s, stats = build_active_area_bitset(
        mask, dilate_masked=0, off_detector="keep", **GEOM)

    half = s // 2
    zz, yy = np.nonzero(mask)
    exact = np.zeros((s, s), dtype=bool)
    exact[half + (zz - NPX // 2), half + (NPX // 2 - yy)] = True

    assert stats["n_cells_masked"] >= int(exact.sum()), (
        f"rasterisation covered {stats['n_cells_masked']} cells but the exact "
        f"integer map needs {int(exact.sum())} -- cells were lost to the "
        "truncation knife edge (this is the 9388 -> 7652 regression)"
    )


def test_off_detector_drop_removes_spots_beyond_the_panel():
    """``drop`` answers "could it have been observed?", ``keep`` isolates the mask."""
    mask = np.zeros((NPX, NPX), dtype=np.uint8)
    bits_drop, s, st = build_active_area_bitset(
        mask, off_detector="drop", dilate_masked=0, **GEOM)
    bits_keep, s2, _ = build_active_area_bitset(
        mask, off_detector="keep", dilate_masked=0, **GEOM)
    assert s == s2
    # A radius well beyond the panel corner.
    r_um = (s // 2 - 1) * PX
    yl = np.array([0.0]); zl = np.array([r_um])
    yc, zc = bigdet_cell_index(yl, zl, s, PX)
    k = int(yc[0]) + s * int(zc[0])
    assert c_test_bit(bits_drop, k) is False, "off-panel must drop under 'drop'"
    assert c_test_bit(bits_keep, k) is True, "off-panel must survive under 'keep'"
    assert st["n_cells_keep_if_keep"] > st["n_cells_keep_if_drop"]


def test_empty_mask_under_keep_sets_everything():
    """The no-op control: no bad pixels and 'keep' must pass every spot.

    This is the analogue of the plan's ``BigDetSize=0`` gate at the level of
    this module -- if it ever fails, the bitset is removing spots for a reason
    that has nothing to do with the mask.
    """
    mask = np.zeros((NPX, NPX), dtype=np.uint8)
    bits, s, stats = build_active_area_bitset(
        mask, off_detector="keep", dilate_masked=1, **GEOM)
    assert stats["n_bad_pixels"] == 0
    assert stats["n_cells_masked_after_dilation"] == 0
    assert stats["n_cells_keep"] == s * s


def test_dilation_only_ever_grows_the_masked_set():
    mask = np.zeros((NPX, NPX), dtype=np.uint8)
    mask[100:104, 100:104] = 1
    _, _, s0 = build_active_area_bitset(
        mask, dilate_masked=0, off_detector="keep", **GEOM)
    _, _, s1 = build_active_area_bitset(
        mask, dilate_masked=1, off_detector="keep", **GEOM)
    assert s1["n_cells_masked_after_dilation"] > s0["n_cells_masked_after_dilation"]
    assert s1["n_cells_keep"] < s0["n_cells_keep"]


def test_large_wedge_warns():
    mask = np.zeros((32, 32), dtype=np.uint8)
    with pytest.warns(RuntimeWarning, match="PRE-wedge"):
        build_active_area_bitset(mask, wedge_deg=1.0, off_detector="keep", **GEOM)


def test_written_file_is_raw_little_endian_uint32(tmp_path):
    grid = np.zeros((8, 8), dtype=bool)
    grid[0, 1] = True
    bits = pack_bitset(grid)
    p = write_big_detector_mask(tmp_path / "BigDetectorMask.bin", bits)
    raw = np.fromfile(p, dtype="<u4")
    assert raw.size == bits.size
    assert np.array_equal(raw, bits)
    assert c_test_bit(raw, 1)
