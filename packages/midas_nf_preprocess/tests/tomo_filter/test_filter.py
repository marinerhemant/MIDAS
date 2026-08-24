"""Tests for the tomo_filter submodule."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from midas_nf_preprocess.hex_grid import make_hex_grid
from midas_nf_preprocess.tomo_filter import (
    bbox_mask,
    filter_grid_by_bbox,
    filter_grid_by_tomo,
    load_square_tomo,
    sample_tomo,
)


# -----------------------------------------------------------------------------
# load_square_tomo
# -----------------------------------------------------------------------------


def test_load_square_tomo_roundtrip(tmp_path):
    arr = np.arange(64, dtype=np.uint8).reshape(8, 8)
    p = tmp_path / "tomo.bin"
    arr.tofile(p)
    out = load_square_tomo(p)
    assert out.shape == (8, 8)
    assert out.dtype == np.uint8
    np.testing.assert_array_equal(out, arr)


def test_load_square_tomo_non_square_raises(tmp_path):
    arr = np.zeros(60, dtype=np.uint8)  # 60 is not a perfect square
    p = tmp_path / "tomo.bin"
    arr.tofile(p)
    with pytest.raises(ValueError, match="not a perfect square"):
        load_square_tomo(p)


# -----------------------------------------------------------------------------
# sample_tomo: coordinate convention matches the C
# -----------------------------------------------------------------------------


def test_sample_tomo_center_pixel():
    """Origin (0, 0) maps to the image center pixel."""
    nr_px = 9
    tomo = np.zeros((nr_px, nr_px), dtype=np.uint8)
    tomo[nr_px - (nr_px // 2), nr_px // 2] = 99  # see C row index nrPxTomo - yPos
    points = torch.tensor([[0.0, 0.0]], dtype=torch.float64)
    values = sample_tomo(points, tomo, px_tomo_um=1.0)
    assert int(values[0]) == 99


def test_sample_tomo_out_of_bounds_returns_zero():
    """Points beyond the image return zero."""
    tomo = np.full((4, 4), 7, dtype=np.uint8)
    points = torch.tensor([[100.0, 100.0]], dtype=torch.float64)
    values = sample_tomo(points, tomo, px_tomo_um=1.0)
    assert int(values[0]) == 0


def test_sample_tomo_y_flip():
    """+y in um maps UP in the image, not down (matches C's nrPxTomo - yPos)."""
    nr_px = 5
    tomo = np.zeros((nr_px, nr_px), dtype=np.uint8)
    # Mark a pixel with positive y in the lab frame -> upper rows of the image.
    # y_um = +1, px = 1 -> y_pos = 1 + 2 = 3; row = 5 - 3 = 2 (upper half).
    tomo[2, 2] = 42  # row 2 is "above" center
    points = torch.tensor([[0.0, 1.0]], dtype=torch.float64)
    values = sample_tomo(points, tomo, px_tomo_um=1.0)
    assert int(values[0]) == 42


def test_sample_tomo_torch_tensor_accepted():
    tomo = torch.zeros((4, 4), dtype=torch.uint8)
    tomo[2, 2] = 5
    points = torch.tensor([[0.0, 0.0]], dtype=torch.float64)
    values = sample_tomo(points, tomo, px_tomo_um=1.0)
    assert int(values[0]) == 5


def test_sample_tomo_non_square_raises():
    tomo = np.zeros((4, 6), dtype=np.uint8)
    with pytest.raises(ValueError, match="square tomo"):
        sample_tomo(torch.zeros(1, 2), tomo, px_tomo_um=1.0)


def test_sample_tomo_wrong_point_shape_raises():
    tomo = np.zeros((4, 4), dtype=np.uint8)
    with pytest.raises(ValueError, match="\\(N, 2\\)"):
        sample_tomo(torch.zeros(4), tomo, px_tomo_um=1.0)


# -----------------------------------------------------------------------------
# filter_grid_by_tomo
# -----------------------------------------------------------------------------


def test_filter_grid_by_tomo_keeps_inside_blob():
    """A grid covering the origin should keep only points landing on the blob.

    The 3x3 blob sits at tomo[4:7, 4:7] of an 11x11 image with px=1um. The
    coordinate convention (filterGridfromTomo.c L39-L43) is:
        col = int(x/px) + 5, row = 11 - (int(y/px) + 5)
    So the blob covers cells where ``int(x)`` in {-1, 0, 1} and ``int(y)`` in
    {0, 1, 2} -- accounting for ``int()`` truncation toward zero, kept points
    have ``-1.99 <= x < 2.0`` and ``0.0 <= y < 3.0``.
    """
    grid = make_hex_grid(grid_size=1.0, r_sample=4.0)
    nr_px = 11
    tomo = np.zeros((nr_px, nr_px), dtype=np.uint8)
    tomo[4:7, 4:7] = 1  # 3x3 blob at the center
    filtered, mask = filter_grid_by_tomo(grid, tomo, px_tomo_um=1.0)
    assert filtered.shape[0] == int(mask.sum())
    assert filtered.shape[0] > 0  # at least one grid point landed on the blob
    assert filtered.shape[0] < grid.shape[0]  # the rest were filtered out
    # int(x_um) must be in {-1, 0, 1}: equivalent to -1 <= x_um and x_um < 2.
    x_int = filtered[:, 2].to(torch.int64)
    y_int = filtered[:, 3].to(torch.int64)
    assert torch.all((x_int >= -1) & (x_int <= 1))
    assert torch.all((y_int >= 0) & (y_int <= 2))


def test_filter_grid_by_tomo_all_zero_keeps_nothing():
    grid = make_hex_grid(grid_size=1.0, r_sample=3.0)
    tomo = np.zeros((9, 9), dtype=np.uint8)
    filtered, mask = filter_grid_by_tomo(grid, tomo, px_tomo_um=1.0)
    assert filtered.shape[0] == 0
    assert int(mask.sum()) == 0


def test_filter_grid_by_tomo_all_one_keeps_within_image():
    grid = make_hex_grid(grid_size=1.0, r_sample=3.0)
    nr_px = 11  # large enough to cover the grid
    tomo = np.ones((nr_px, nr_px), dtype=np.uint8)
    filtered, mask = filter_grid_by_tomo(grid, tomo, px_tomo_um=1.0)
    assert filtered.shape[0] == grid.shape[0]


def test_filter_grid_wrong_shape_raises():
    bad = torch.zeros(10, 3, dtype=torch.float64)
    with pytest.raises(ValueError, match="\\(N, 5\\)"):
        filter_grid_by_tomo(bad, np.zeros((4, 4), dtype=np.uint8), 1.0)


# -----------------------------------------------------------------------------
# bbox_mask + filter_grid_by_bbox
# -----------------------------------------------------------------------------


def test_bbox_mask_geometry():
    grid = make_hex_grid(grid_size=1.0, r_sample=4.0)
    mask = bbox_mask(grid, [-1.0, 1.0, -1.0, 1.0])
    # All True positions must be within the bbox.
    kept = grid[mask]
    assert torch.all(kept[:, 2] >= -1.0)
    assert torch.all(kept[:, 2] <= 1.0)
    assert torch.all(kept[:, 3] >= -1.0)
    assert torch.all(kept[:, 3] <= 1.0)


def test_bbox_mask_invalid_length_raises():
    with pytest.raises(ValueError, match="length"):
        bbox_mask(torch.zeros(3, 5), [0, 1, 2])


def test_bbox_mask_reversed_corners_raises():
    with pytest.raises(ValueError, match="reversed"):
        bbox_mask(torch.zeros(3, 5), [1, 0, 0, 1])


def test_filter_grid_by_bbox():
    grid = make_hex_grid(grid_size=1.0, r_sample=5.0)
    f, m = filter_grid_by_bbox(grid, [-2.0, 2.0, -2.0, 2.0])
    assert f.shape[0] == int(m.sum())


# -----------------------------------------------------------------------------
# The three one-pixel defects in filterGridfromTomo.c:39-43 (found 2026-08-23)
#
# Every test above uses INTEGER coordinates, which is the one case where the
# buggy and the correct conventions agree. That is why these survived.
# -----------------------------------------------------------------------------


def test_defect1_the_bottom_row_used_to_raise_IndexError():
    """``row = n - y_pos`` is ``n`` at ``y_pos == 0`` -- one past the last row.

    In C that reads past the ``calloc``; in Python it raised. Both modes now
    treat it as outside the image, which is the only defensible reading of an
    out-of-bounds read.
    """
    n = 8
    tomo = np.arange(n * n, dtype=np.uint8).reshape(n, n)
    # y_pos == 0  <=>  y/px + n/2 in [0, 1)  <=>  y = -4.0 at px = 1
    pts = torch.tensor([[0.0, -4.0]], dtype=torch.float64)
    for parity in (True, False):
        v = sample_tomo(pts, tomo, 1.0, legacy_c_parity=parity)
        assert v.shape == (1,)          # no exception
    assert int(sample_tomo(pts, tomo, 1.0, legacy_c_parity=True)[0]) == 0


def test_defect1_the_corrected_flip_reaches_the_bottom_row():
    """The other half of the defect: parity mode can never read row n-1
    (the bottom of the image), so a sample touching the bottom edge is
    silently trimmed by one pixel."""
    n = 8
    tomo = np.zeros((n, n), dtype=np.uint8)
    tomo[n - 1, 4] = 77                       # bottom row
    pts = torch.tensor([[0.0, -4.0]], dtype=torch.float64)   # y_pos == 0
    assert int(sample_tomo(pts, tomo, 1.0, legacy_c_parity=False)[0]) == 77
    assert int(sample_tomo(pts, tomo, 1.0, legacy_c_parity=True)[0]) == 0


def test_defect1_the_two_modes_differ_by_exactly_one_row():
    n = 16
    tomo = np.arange(n * n, dtype=np.uint8).reshape(n, n)
    ys = torch.arange(-7.0, 8.0, dtype=torch.float64)
    pts = torch.stack([torch.zeros_like(ys), ys], dim=1)
    legacy = sample_tomo(pts, tomo, 1.0, legacy_c_parity=True)
    fixed = sample_tomo(pts, tomo, 1.0, legacy_c_parity=False)
    # one row apart => the values differ by exactly one row stride, n
    assert torch.all(legacy - fixed == n)


def test_defect2_truncating_the_quotient_moved_75_percent_of_the_grid():
    """The C truncates ``x/px + n/2``; the Python truncated ``x/px`` then
    added. They differ for every negative coordinate with a fractional part.

    This test asserts the CURRENT code matches the C, and that the old
    expression would not -- otherwise it would pass either way.
    """
    n, px = 64, 1.5
    tomo = np.arange(n * n, dtype=np.uint8).reshape(n, n)
    rng = np.random.default_rng(0)
    xy = rng.uniform(-40.0, 40.0, size=(4000, 2))
    pts = torch.tensor(xy, dtype=torch.float64)

    c_x = np.trunc(xy[:, 0] / px + n / 2.0).astype(np.int64)
    c_y = np.trunc(xy[:, 1] / px + n / 2.0).astype(np.int64)
    old_x = np.trunc(xy[:, 0] / px).astype(np.int64) + n // 2
    old_y = np.trunc(xy[:, 1] / px).astype(np.int64) + n // 2
    disagree = (c_x != old_x) | (c_y != old_y)
    assert disagree.mean() > 0.6, "fixture must exercise the divergence"

    ok = (c_x >= 0) & (c_x < n) & (c_y >= 1) & (c_y < n)
    want = np.where(ok, tomo[np.clip(n - c_y, 0, n - 1), np.clip(c_x, 0, n - 1)], 0)
    got = sample_tomo(pts, tomo, px, legacy_c_parity=True).numpy()
    np.testing.assert_array_equal(got, want)


def _column_hit(x_um, n, px=1.0, parity=False):
    """Which column does this x land in? Found by probing, not recomputed."""
    pts = torch.tensor([[x_um, 0.0]], dtype=torch.float64)
    for c in range(n):
        probe = np.zeros((n, n), dtype=np.uint8)
        probe[:, c] = 1
        if int(sample_tomo(pts, probe, px, legacy_c_parity=parity)[0]):
            return c
    return None


def _row_hit(y_um, n, px=1.0, parity=False):
    pts = torch.tensor([[0.0, y_um]], dtype=torch.float64)
    for r in range(n):
        probe = np.zeros((n, n), dtype=np.uint8)
        probe[r, :] = 1
        if int(sample_tomo(pts, probe, px, legacy_c_parity=parity)[0]):
            return r
    return None


def test_defect3_an_odd_image_has_a_pixel_CENTRED_on_the_origin():
    """``(double)n / 2`` vs ``n // 2``.

    With the float centre and an odd n, the centre pixel spans
    ``x/px in [-0.5, 0.5)`` -- it is centred on the rotation axis, which is
    what an odd-width reconstruction grid means. With ``n // 2`` the origin
    sits on that pixel's *edge*, so x = -0.4 px and x = +0.4 px land in
    different columns despite straddling the axis by less than half a pixel.
    """
    n = 9
    assert _column_hit(-0.4, n) == _column_hit(+0.4, n) == n // 2
    assert _column_hit(-0.9, n) == n // 2 - 1
    assert _column_hit(+0.9, n) == n // 2 + 1

    # The old expression is worse than a shift: truncating the quotient TOWARD
    # ZERO makes the centre pixel twice as wide as every other one, because
    # both (-1, 0] and [0, 1) collapse onto it.
    old = lambda x: int(np.trunc(x / 1.0)) + n // 2       # noqa: E731
    assert old(-0.9) == old(+0.9) == n // 2
    assert old(-1.5) == n // 2 - 1 and old(+1.5) == n // 2 + 1


def test_the_corrected_mode_is_symmetric_about_the_row_the_origin_lands_on():
    """The property the off-by-one breaks. Note the reference row is measured,
    not assumed to be ``(n-1)/2``: for even n no pixel is centred on the axis,
    and asserting a half-integer centre is how the previous attempt at this
    test failed."""
    n = 16
    r0 = _row_hit(0.0, n)
    for d in (1.0, 3.0, 5.0):
        up, down = _row_hit(+d, n), _row_hit(-d, n)
        assert (r0 - up) == (down - r0) == d, f"asymmetric at {d}"
    # +y is UP in the image (lower row index) -- the flip is still there
    assert _row_hit(+3.0, n) < r0 < _row_hit(-3.0, n)


def test_filter_grid_by_tomo_forwards_the_parity_flag():
    n = 8
    tomo = np.zeros((n, n), dtype=np.uint8)
    tomo[n - 1, :] = 1                                   # bottom row only
    grid = torch.tensor([[0.0, 0.0, 0.0, -4.0, 0.5]], dtype=torch.float64)
    _, m_legacy = filter_grid_by_tomo(grid, tomo, 1.0, legacy_c_parity=True)
    _, m_fixed = filter_grid_by_tomo(grid, tomo, 1.0, legacy_c_parity=False)
    assert not bool(m_legacy[0]) and bool(m_fixed[0])
