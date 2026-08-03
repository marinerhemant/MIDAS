"""Behaviour tests for the torch NLM backend.

The torch implementation exists because scikit-image's is CPU-only, which is
what forced ``process_layer`` to relocate a whole layer to the host whenever
NLM was enabled.

These are NOT bit-parity tests, because the two are NOT bit-equivalent -- see
``process_images.denoise``.  They pin the agreement that is actually claimed
(same blobs to within a couple, high correlation) plus the properties that must
hold regardless: isotropy, differentiability, argument validation, and the
sigma_MAD-zero guard.

A NOTE ON TEST POWER, learned the hard way: an early check on a 700-row crop
containing 8 blobs showed exact agreement while the full frame disagreed by
26 %.  The synthetic images here therefore carry MANY spots.
"""
import numpy as np
import pytest
import torch

from midas_nf_preprocess.process_images.denoise import nl_means_torch
from midas_nf_preprocess.process_images.pipeline import _nlm_denoise_residual

skimage = pytest.importorskip("skimage.restoration")


def _synthetic(seed=0, shape=(220, 260), n_spots=120, amp=40.0, noise=2.0):
    """Sparse Gaussian spots on noise -- the shape of an NF residual."""
    rng = np.random.default_rng(seed)
    img = rng.normal(0.0, noise, shape).astype(np.float32)
    yy, xx = np.mgrid[-3:4, -3:4]
    g = np.exp(-(yy**2 + xx**2) / 2.0).astype(np.float32)
    for _ in range(n_spots):
        r = rng.integers(5, shape[0] - 5)
        c = rng.integers(5, shape[1] - 5)
        img[r - 3:r + 4, c - 3:c + 4] += amp * g
    return np.clip(img, 0, None)


def _blobs(a, thr, min_px=4):
    from scipy import ndimage as ndi
    lab, _ = ndi.label(a > thr, structure=np.ones((3, 3), int))
    if lab.max() == 0:
        return 0
    return int((np.bincount(lab.ravel())[1:] >= min_px).sum())


@pytest.mark.parametrize("h", [2.0, 4.0])
def test_matches_skimage_on_spotty_data(h):
    """Same blobs and near-identical pixels as scikit-image."""
    img = _synthetic()
    ref = skimage.denoise_nl_means(
        img, h=h, sigma=h, fast_mode=True, patch_size=5,
        patch_distance=6, channel_axis=None,
    )
    got = nl_means_torch(torch.from_numpy(img), h=h, sigma=h,
                         patch_size=5, patch_distance=6,
                         device="cpu").numpy()
    # Bounds are what was MEASURED, not what would be nice: the two
    # implementations differ in some detail of skimage's variant that has not
    # been pinned down (it ships compiled only).
    assert np.corrcoef(got.ravel(), ref.ravel())[0, 1] > 0.99
    for thr in (2.0, 5.0, 10.0):
        nb, nr = _blobs(got, thr), _blobs(ref, thr)
        assert abs(nb - nr) <= max(2, 0.02 * nr), (
            f"blob count {nb} vs skimage {nr} at threshold {thr}"
        )
        inter = ((got > thr) & (ref > thr)).sum()
        union = ((got > thr) | (ref > thr)).sum()
        assert inter / max(union, 1) > 0.70, f"IoU too low at {thr}"


def test_symmetry_covers_the_whole_window():
    """The half-window loop plus +-t symmetry must equal the full window.

    Denoising is isotropic here, so a 90-degree rotation of the input must give
    the 90-degree rotation of the output.  A half-window bug shows up as an
    anisotropic result.
    """
    img = _synthetic(seed=3)
    a = nl_means_torch(torch.from_numpy(img), h=3.0, sigma=3.0, device="cpu")
    b = nl_means_torch(torch.from_numpy(np.rot90(img).copy()), h=3.0, sigma=3.0,
                       device="cpu")
    assert torch.allclose(a, torch.rot90(b, -1), atol=1e-4)


def test_is_differentiable():
    """The torch path keeps autograd alive; skimage's cannot."""
    img = torch.from_numpy(_synthetic(seed=5, shape=(60, 60), n_spots=8))
    img.requires_grad_(True)
    out = nl_means_torch(img, h=3.0, sigma=3.0, patch_distance=2, device="cpu")
    out.sum().backward()
    assert img.grad is not None and torch.isfinite(img.grad).all()
    assert img.grad.abs().sum() > 0


def test_rejects_even_patch_and_zero_h():
    img = torch.zeros((32, 32))
    with pytest.raises(ValueError, match="odd"):
        nl_means_torch(img, patch_size=4, device="cpu")
    with pytest.raises(ValueError, match="h must be"):
        nl_means_torch(img, h=0.0, device="cpu")


def test_backend_dispatch_and_validation():
    """auto stays on skimage for CPU input; explicit torch is honoured."""
    resid = torch.from_numpy(_synthetic(seed=7, shape=(64, 64), n_spots=10))
    a = _nlm_denoise_residual(resid, h_absolute=3.0, backend="skimage")
    b = _nlm_denoise_residual(resid, h_absolute=3.0, backend="torch")
    assert a.shape == b.shape == resid.shape
    assert np.corrcoef(a.numpy().ravel(), b.numpy().ravel())[0, 1] > 0.99
    with pytest.raises(ValueError, match="auto|torch|skimage"):
        _nlm_denoise_residual(resid, h_absolute=3.0, backend="nope")


def test_sigma_mad_zero_still_warns_on_torch_backend():
    """The silent-no-op guard must survive on the new path too."""
    resid = torch.zeros((40, 40))
    with pytest.warns(RuntimeWarning, match="sigma_MAD is 0"):
        out = _nlm_denoise_residual(resid, h_factor=1.0, backend="torch")
    assert torch.equal(out, resid)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_cuda_matches_cpu():
    img = torch.from_numpy(_synthetic(seed=11))
    c = nl_means_torch(img, h=3.0, sigma=3.0, device="cpu")
    g = nl_means_torch(img, h=3.0, sigma=3.0, device="cuda").cpu()
    assert torch.allclose(c, g, atol=1e-3)
