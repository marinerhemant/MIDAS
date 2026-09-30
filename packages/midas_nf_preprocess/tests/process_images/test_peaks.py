"""Tests for peak detection: label_components, zero_crossings, find_peaks."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from midas_nf_preprocess.process_images import (
    apply_log,
    auto_temperature,
    build_log_kernel,
    find_peaks,
    label_components,
    zero_crossings,
)
from midas_nf_preprocess.process_images.peaks import _resolve_temperature, _stack_3x3_min

try:
    from scipy.ndimage import label as scipy_label
    HAVE_SCIPY = True
except ImportError:
    HAVE_SCIPY = False


# -----------------------------------------------------------------------------
# label_components: structural tests
# -----------------------------------------------------------------------------


def test_label_components_empty_mask():
    mask = torch.zeros(10, 10, dtype=torch.bool)
    labels, n = label_components(mask, return_n=True)
    assert n == 0
    assert torch.all(labels == 0)


def test_label_components_single_pixel():
    mask = torch.zeros(5, 5, dtype=torch.bool)
    mask[2, 2] = True
    labels, n = label_components(mask, return_n=True)
    assert n == 1
    assert labels[2, 2] == 1
    assert labels.sum() == 1


def test_label_components_two_blobs():
    mask = torch.zeros(10, 10, dtype=torch.bool)
    mask[1:3, 1:3] = True   # blob 1
    mask[6:9, 6:9] = True   # blob 2
    labels, n = label_components(mask, return_n=True)
    assert n == 2
    # All pixels in blob 1 share a label; same for blob 2; and the two labels differ.
    blob1_labels = labels[1:3, 1:3]
    blob2_labels = labels[6:9, 6:9]
    assert (blob1_labels == blob1_labels[0, 0]).all()
    assert (blob2_labels == blob2_labels[0, 0]).all()
    assert blob1_labels[0, 0] != blob2_labels[0, 0]


def test_label_components_8_connectivity():
    """Two diagonally-touching pixels should belong to the same component."""
    mask = torch.zeros(5, 5, dtype=torch.bool)
    mask[1, 1] = True
    mask[2, 2] = True  # diagonal neighbor
    labels, n = label_components(mask, return_n=True)
    assert n == 1
    assert labels[1, 1] == labels[2, 2]


def test_label_components_orthogonal_connectivity():
    mask = torch.zeros(5, 5, dtype=torch.bool)
    mask[2, 1] = True
    mask[2, 2] = True  # right neighbor
    labels, n = label_components(mask, return_n=True)
    assert n == 1


def test_label_components_disconnected_when_not_touching():
    """Two pixels separated by >1 should be different components."""
    mask = torch.zeros(5, 5, dtype=torch.bool)
    mask[1, 1] = True
    mask[1, 3] = True  # not touching (gap of one pixel)
    labels, n = label_components(mask, return_n=True)
    assert n == 2


def test_label_components_contiguous_renumbering():
    """Labels should be 1..K with no gaps."""
    mask = torch.zeros(20, 20, dtype=torch.bool)
    # Several scattered single-pixel blobs at non-trivial positions.
    for (i, j) in [(2, 3), (5, 10), (15, 4), (18, 18), (10, 10)]:
        mask[i, j] = True
    labels, n = label_components(mask, return_n=True)
    assert n == 5
    nonzero = labels[labels != 0]
    assert set(nonzero.tolist()) == {1, 2, 3, 4, 5}


def test_label_components_large_blob():
    """A solid 30x30 block should be one component, with correct label propagation."""
    mask = torch.zeros(40, 40, dtype=torch.bool)
    mask[5:35, 5:35] = True
    labels, n = label_components(mask, max_iters=200, return_n=True)
    assert n == 1
    assert (labels[5:35, 5:35] == 1).all()


def test_label_components_returns_int64():
    mask = torch.ones(3, 3, dtype=torch.bool)
    labels = label_components(mask)
    assert labels.dtype == torch.int64


@pytest.mark.parity
@pytest.mark.skipif(not HAVE_SCIPY, reason="needs scipy")
def test_label_components_matches_scipy_count():
    """Component COUNT (not IDs, since renumbering order may differ) matches scipy."""
    rng = np.random.default_rng(0)
    mask_np = rng.random((30, 30)) > 0.7
    # scipy with 8-connectivity uses structure of all-ones.
    structure = np.ones((3, 3), dtype=bool)
    _, scipy_n = scipy_label(mask_np, structure=structure)
    _, our_n = label_components(
        torch.from_numpy(mask_np), max_iters=200, return_n=True
    )
    assert our_n == scipy_n


@pytest.mark.parity
@pytest.mark.skipif(not HAVE_SCIPY, reason="needs scipy")
def test_label_components_matches_scipy_partition():
    """Same partition into equivalence classes (labels themselves may differ)."""
    rng = np.random.default_rng(1)
    mask_np = rng.random((20, 20)) > 0.6
    structure = np.ones((3, 3), dtype=bool)
    scipy_labels, _ = scipy_label(mask_np, structure=structure)
    our_labels = label_components(torch.from_numpy(mask_np), max_iters=100).numpy()
    # Two pixels are in the same component iff they have the same scipy label
    # iff they have the same our label (excluding background = 0).
    fg = mask_np
    coords = np.argwhere(fg)
    for i in range(len(coords)):
        for j in range(i + 1, min(i + 30, len(coords))):  # spot-check pairs
            ci, cj = tuple(coords[i]), tuple(coords[j])
            same_scipy = scipy_labels[ci] == scipy_labels[cj]
            same_ours = our_labels[ci] == our_labels[cj]
            assert same_scipy == same_ours, (
                f"Disagreement at {ci} vs {cj}: "
                f"scipy={scipy_labels[ci]},{scipy_labels[cj]} "
                f"ours={our_labels[ci]},{our_labels[cj]}"
            )


def test_label_components_wrong_ndim():
    with pytest.raises(ValueError, match="\\[H, W\\]"):
        label_components(torch.zeros(2, 3, 4))


# -----------------------------------------------------------------------------
# zero_crossings
# -----------------------------------------------------------------------------


def test_zero_crossings_constant_signal_no_edges():
    img = torch.ones(10, 10, dtype=torch.float64)
    edges = zero_crossings(img)
    assert not edges.any()


def test_zero_crossings_uniform_negative_no_edges():
    img = -torch.ones(10, 10, dtype=torch.float64)
    edges = zero_crossings(img)
    assert not edges.any()


def test_zero_crossings_half_split():
    """Image with positive top half, negative bottom: a horizontal band of edges."""
    img = torch.ones(10, 10, dtype=torch.float64)
    img[5:, :] = -1.0
    edges = zero_crossings(img)
    # Pixels on the boundary between row 4 (positive) and row 5 (negative)
    # should be flagged on both sides.
    assert edges[4, 5]  # interior of the boundary
    assert edges[5, 5]
    # Far from the boundary: no edges.
    assert not edges[1, 5]
    assert not edges[8, 5]


def test_zero_crossings_borders_excluded():
    """Border pixels are never flagged, even if conditions match."""
    img = torch.ones(10, 10, dtype=torch.float64)
    img[0, :] = -1.0  # top row
    edges = zero_crossings(img)
    # Top row itself is a border -> excluded.
    assert not edges[0, :].any()
    # But row 1 should have some edges (interior pixels with negative neighbors above).
    assert edges[1, 1:-1].any()


# -----------------------------------------------------------------------------
# find_peaks: end-to-end on Gaussian blobs
# -----------------------------------------------------------------------------


def test_find_peaks_recovers_two_blobs(gaussian_blob_image):
    img, expected_centers = gaussian_blob_image()
    kernel = build_log_kernel(radius=4, sigma=2.5)
    out = find_peaks(img, [kernel])
    # Should find at least 2 components (could be more if the LoG splits a blob).
    assert out.n_components >= 2
    # Each expected center should be inside a labeled region.
    for cz, cy in expected_centers:
        assert out.labels[cz, cy] > 0


def test_find_peaks_no_blobs_returns_empty():
    img = torch.zeros(32, 32, dtype=torch.float64)
    kernel = build_log_kernel(radius=4, sigma=1.0)
    out = find_peaks(img, [kernel])
    assert out.n_components == 0
    assert (out.labels == 0).all()


def test_find_peaks_outputs_correct_shapes(gaussian_blob_image):
    img, _ = gaussian_blob_image()
    kernel = build_log_kernel(radius=4, sigma=2.5)
    out = find_peaks(img, [kernel])
    assert out.log_response.shape == img.shape
    assert out.spot_prob.shape == img.shape
    assert out.labels.shape == img.shape
    assert out.labels.dtype == torch.int64


def test_find_peaks_soft_prob_in_unit_range(gaussian_blob_image):
    img, _ = gaussian_blob_image()
    kernel = build_log_kernel(radius=4, sigma=2.5)
    out = find_peaks(img, [kernel])
    assert (out.spot_prob >= 0).all()
    assert (out.spot_prob <= 1).all()


def test_find_peaks_labels_detached(gaussian_blob_image):
    img, _ = gaussian_blob_image()
    img.requires_grad_(True)
    kernel = build_log_kernel(radius=4, sigma=2.5, dtype=torch.float64)
    out = find_peaks(img, [kernel])
    assert out.labels.requires_grad is False
    # log_response and spot_prob should still be in the autograd graph.
    assert out.log_response.requires_grad
    assert out.spot_prob.requires_grad


def test_find_peaks_multiscale_first_kernel_wins(gaussian_blob_image):
    img, _ = gaussian_blob_image()
    k1 = build_log_kernel(radius=4, sigma=2.5)
    k2 = build_log_kernel(radius=4, sigma=1.0)  # narrower, may fragment large blobs
    out_single = find_peaks(img, [k1])
    out_multi = find_peaks(img, [k1, k2])
    # Multi-scale should find at least as many components as single-scale.
    assert out_multi.n_components >= out_single.n_components


def test_find_peaks_no_kernels_raises():
    img = torch.zeros(8, 8, dtype=torch.float64)
    with pytest.raises(ValueError, match="at least one"):
        find_peaks(img, [])


# -----------------------------------------------------------------------------
# Internal: _stack_3x3_min sanity
# -----------------------------------------------------------------------------


def test_stack_3x3_min_propagates_min():
    sentinel = 999
    x = torch.tensor([[5, 5, 5], [5, 1, 5], [5, 5, 5]])
    out = _stack_3x3_min(x, sentinel)
    # Center has min(1, 5*8) = 1; neighbors all see the 1.
    assert (out == 1).all()


# -----------------------------------------------------------------------------
# auto_temperature
# -----------------------------------------------------------------------------


def test_auto_temperature_zero_input_falls_back_to_one():
    x = torch.zeros(8, 8, dtype=torch.float64)
    T = auto_temperature(x)
    assert float(T) == 1.0


def test_auto_temperature_signed_uses_abs():
    """Negative values contribute to the temperature via |x|."""
    pos = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], dtype=torch.float64)
    neg = -pos
    T_pos = auto_temperature(pos)
    T_neg = auto_temperature(neg)
    assert float(T_pos) == float(T_neg)


def test_auto_temperature_scales_linearly_with_input():
    """Doubling the input doubles the temperature (the heuristic is scale-equivariant)."""
    x = torch.linspace(0, 100, 100, dtype=torch.float64)
    T1 = auto_temperature(x)
    T2 = auto_temperature(x * 10)
    assert torch.isclose(T2, T1 * 10, rtol=1e-5)


def test_auto_temperature_floor_applied():
    x = torch.tensor([1e-12, 0.0], dtype=torch.float64)
    T = auto_temperature(x, floor=1e-6)
    assert float(T) == pytest.approx(1e-6)


def test_auto_temperature_detached_no_grad_flow():
    """Temperature is computed under no_grad; backward through sigmoid(x/T)
    should give the same gradient as if T were a Python float."""
    x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], dtype=torch.float64, requires_grad=True)
    T = auto_temperature(x)
    out = torch.sigmoid(x / T).sum()
    out.backward()
    # Compare against the same expression with a literal float for T.
    x2 = x.detach().clone().requires_grad_(True)
    out2 = torch.sigmoid(x2 / float(T)).sum()
    out2.backward()
    assert torch.allclose(x.grad, x2.grad)


def test_auto_temperature_dtype_matches_input():
    x = torch.ones(4, dtype=torch.float32) * 5.0
    T = auto_temperature(x)
    assert T.dtype == torch.float32


def test_auto_temperature_sigmoid_does_not_saturate_at_peak():
    """At the input's high-quantile value, sigmoid(x/T) should be close to but
    < 1 (~0.98 with the default saturation_factor=4)."""
    x = torch.linspace(0, 1000, 1000, dtype=torch.float64)
    T = auto_temperature(x)
    q_val = torch.quantile(x[x > 0], 0.95)
    s = torch.sigmoid(q_val / T)
    assert 0.95 < float(s) < 1.0


def test_resolve_temperature_string_auto():
    """auto picks per-tensor temperatures (img and L scaled separately)."""
    # Linear ramps with well-defined high quantiles.
    img = torch.linspace(0, 1.0, 100, dtype=torch.float64)
    L = torch.linspace(0, 100.0, 100, dtype=torch.float64)
    T_img, T_log = _resolve_temperature("auto", img, L)
    # T_log should be ~100x T_img since L is 100x larger.
    assert torch.isclose(T_log, T_img * 100, rtol=1e-3)


def test_resolve_temperature_float_uses_for_both():
    img = torch.tensor([[0.0, 1.0]])
    L = torch.tensor([[0.0, 10.0]])
    T_img, T_log = _resolve_temperature(2.5, img, L)
    assert float(T_img) == 2.5
    assert float(T_log) == 2.5


def test_resolve_temperature_invalid_string_raises():
    img = torch.zeros(4)
    L = torch.zeros(4)
    with pytest.raises(ValueError, match="Unknown temperature mode"):
        _resolve_temperature("magic", img, L)


def test_resolve_temperature_negative_raises():
    img = torch.zeros(4)
    L = torch.zeros(4)
    with pytest.raises(ValueError, match="must be positive"):
        _resolve_temperature(-1.0, img, L)


# -----------------------------------------------------------------------------
# find_peaks with auto temperature
# -----------------------------------------------------------------------------


def test_find_peaks_auto_temperature_default(gaussian_blob_image):
    img, _ = gaussian_blob_image()
    k = build_log_kernel(radius=4, sigma=2.5)
    out = find_peaks(img, [k])  # default = "auto"
    assert out.temperature_img > 0
    assert out.temperature_log > 0
    # Auto picks intensity-scale-appropriate values, not 1.0.
    assert out.temperature_img != 1.0


def test_find_peaks_explicit_float_overrides_auto(gaussian_blob_image):
    img, _ = gaussian_blob_image()
    k = build_log_kernel(radius=4, sigma=2.5)
    out = find_peaks(img, [k], soft_temperature=10.0)
    assert out.temperature_img == 10.0
    assert out.temperature_log == 10.0


def test_find_peaks_auto_temperature_does_not_saturate(gaussian_blob_image):
    """With raw-amplitude (~1000) inputs and T=1.0, sigmoid(img/T) is ~all-ones;
    auto-temp should keep spot_prob meaningfully below 1 over most of the image."""
    img, _ = gaussian_blob_image()  # peak ~1000
    k = build_log_kernel(radius=4, sigma=2.5)

    out_fixed = find_peaks(img, [k], soft_temperature=1.0)
    out_auto = find_peaks(img, [k], soft_temperature="auto")

    # Fixed T=1 saturates: most positive pixels get spot_prob ≈ 1.
    sat_fixed = (out_fixed.spot_prob > 0.99).float().mean()
    sat_auto = (out_auto.spot_prob > 0.99).float().mean()
    assert sat_auto < sat_fixed


def test_auto_temperature_handles_full_detector_frames():
    """Regression (2026-09-29): torch.quantile raised 'input tensor is too large' on a dense
    4600 x 5320 score map (24.5 M nonzero), crashing SpotDetect matched / poisson on 20-ID frames."""
    import torch
    from midas_nf_preprocess.process_images.peaks import auto_temperature
    g = torch.Generator().manual_seed(0)
    big = torch.rand(4600, 5320, generator=g) + 0.1                 # all nonzero, > 2**24 elements
    small = big[:1000, :1000]
    t_big = float(auto_temperature(big)); t_small = float(auto_temperature(small))
    assert t_big > 0 and abs(t_big - t_small) / t_small < 0.02      # same scale as an exact quantile of a sample
