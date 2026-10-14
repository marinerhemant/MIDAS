"""Test pre-processing pipeline (square-pad, transforms, dark/flood)."""
import numpy as np
import pytest

from midas_peakfit.preprocess import (
    apply_image_transformations,
    make_square_image,
    transpose_square,
    preprocess_frame,
)


def test_square_pad_y_gt_z():
    """Z=3, Y=5 input → NrPixels=5; out[:3, :5] = in, rest zero."""
    a = np.arange(15, dtype=np.float64).reshape(3, 5)
    out = make_square_image(a, NrPixels=5, NrPixelsY=5, NrPixelsZ=3)
    np.testing.assert_array_equal(out[:3, :5], a)
    np.testing.assert_array_equal(out[3:, :], 0.0)


def test_square_pad_z_gt_y():
    """Z=5, Y=3 input → NrPixels=5; out[:5, :3] = in, out[:, 3:] = 0."""
    a = np.arange(15, dtype=np.float64).reshape(5, 3)
    out = make_square_image(a, NrPixels=5, NrPixelsY=3, NrPixelsZ=5)
    np.testing.assert_array_equal(out[:5, :3], a)
    np.testing.assert_array_equal(out[:, 3:], 0.0)


def test_apply_transforms_flip_h():
    a = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float64)
    out = apply_image_transformations(a, [1])  # flip horizontal (Y)
    np.testing.assert_array_equal(out, np.array([[3, 2, 1], [6, 5, 4]]))


def test_apply_transforms_flip_v():
    a = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float64)
    out = apply_image_transformations(a, [2])  # flip vertical (Z)
    np.testing.assert_array_equal(out, np.array([[4, 5, 6], [1, 2, 3]]))


def test_apply_transforms_transpose():
    a = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float64)
    out = apply_image_transformations(a, [3])  # transpose
    np.testing.assert_array_equal(out, a.T)


def test_apply_transforms_compose():
    a = np.array([[1, 2], [3, 4]], dtype=np.float64)
    # Sequence: flip-H then transpose
    out = apply_image_transformations(a, [1, 3])
    expected = a[:, ::-1].T
    np.testing.assert_array_equal(out, expected)


def test_preprocess_frame_smoke():
    """Per-frame pipeline runs without error and returns float64 (N, N)."""
    Y, Z = 64, 64
    raw = np.full((Z, Y), 100.0, dtype=np.float64)
    raw[20:25, 20:25] = 200.0
    dark = np.zeros((Y, Y), dtype=np.float64)
    flood = np.ones((Y, Y), dtype=np.float64)
    gc = np.full((Y, Y), 50.0, dtype=np.float64)

    out = preprocess_frame(
        raw,
        NrPixels=Y, NrPixelsY=Y, NrPixelsZ=Z,
        transform_options=[],
        dark=dark, flood=flood, good_coords=gc,
        bc=1.0, bad_px_intensity=0.0, make_map=0,
    )
    assert out.shape == (Y, Y)
    assert out.dtype == np.float64
    # Bright pixels are above threshold; transposed they end up at imgCorrBC[20:25, 20:25]
    assert (out > 0).any()


# ─── Non-square panels: transforms mirror about the REAL sensor edge ─────────
# Same convention as midas_calibrate_v2.io.transforms.apply_im_trans (flip
# within the true (Z, Y) array). Regression: on a 1028x512 Eiger, ImTransOpt 2
# sent ZCen to [516, 1027] because the flip ran over the padded square.
def _analysis_yz(opts, NY, NZ, y0, z0):
    N = max(NY, NZ)
    raw = np.zeros((NZ, NY))
    raw[z0, y0] = 1.0
    sq = make_square_image(raw, N, NY, NZ)
    img = transpose_square(
        apply_image_transformations(sq, opts, NrPixelsY=NY, NrPixelsZ=NZ)
    )
    (y, z), = np.argwhere(img > 0)
    return int(y), int(z)


@pytest.mark.parametrize("NY,NZ", [(1028, 512), (512, 1028), (7, 5), (5, 7)])
@pytest.mark.parametrize("opts", [[], [1], [2], [1, 2], [3], [3, 1], [2, 3], [3, 2, 1]])
def test_nonsquare_matches_rect_transform(NY, NZ, opts):
    """Analysis-frame (y, z) equals the transform of the real (Z, Y) array."""
    y0, z0 = 3, 2
    ref = np.zeros((NZ, NY))
    ref[z0, y0] = 1.0
    for c in opts:
        ref = ref[:, ::-1] if c == 1 else ref[::-1, :] if c == 2 else ref.T
    rz, ry = np.argwhere(ref > 0)[0]          # ref is (Z', Y')
    assert _analysis_yz(opts, NY, NZ, y0, z0) == (int(ry), int(rz))


def test_nonsquare_flip_z_stays_on_sensor():
    y, z = _analysis_yz([2], NY=1028, NZ=512, y0=300, z0=100)
    assert (y, z) == (300, 511 - 100)


def test_nonsquare_flip_y_when_z_larger():
    y, z = _analysis_yz([1], NY=512, NZ=1028, y0=100, z0=300)
    assert (y, z) == (511 - 100, 300)


def test_square_unchanged_by_dims():
    a = np.arange(16.0).reshape(4, 4)
    for opts in ([1], [2], [3], [1, 3]):
        np.testing.assert_array_equal(
            apply_image_transformations(a, opts),
            apply_image_transformations(a, opts, NrPixelsY=4, NrPixelsZ=4),
        )
