"""Realistic detector chain (PSF + noise + 16-bit) and Poulsen resolution widths."""
import torch
import pytest

from midas_dfxm.detector import apply_psf, detector_model, quantize_16bit
from midas_dfxm.resolution import poulsen_resolution_widths

DT = torch.float64


def _blob(n=64):
    x = torch.linspace(-3, 3, n, dtype=DT)
    X, Y = torch.meshgrid(x, x, indexing="ij")
    return torch.exp(-0.5 * (X ** 2 + Y ** 2))


@pytest.mark.unit
def test_psf_differentiable_and_conserving():
    img = _blob().requires_grad_(True)
    out = apply_psf(img, 1.5)
    out.sum().backward()
    assert img.grad is not None and torch.isfinite(img.grad).all()
    # a normalized-kernel blur approximately conserves total intensity
    assert abs(float(apply_psf(_blob(), 1.5).sum() - _blob().sum())) / float(_blob().sum()) < 0.05


@pytest.mark.unit
def test_detector_noisefree_differentiable():
    img = _blob().requires_grad_(True)
    out = detector_model(img, psf_sigma_px=1.0, noise=False)
    out.sum().backward()
    assert img.grad is not None and torch.isfinite(img.grad).all()


@pytest.mark.unit
def test_detector_noisy_is_16bit_integers_in_range():
    g = torch.Generator().manual_seed(0)
    out = detector_model(_blob(), psf_sigma_px=1.0, peak_counts=6e4, noise=True, generator=g)
    assert out.min() >= 0 and out.max() <= 65535
    assert torch.allclose(out, out.round())          # integer counts
    # noise present: not identical to the clean scaled image
    clean = detector_model(_blob(), psf_sigma_px=1.0, peak_counts=6e4, noise=False)
    assert float((out - clean).abs().mean()) > 0


@pytest.mark.unit
def test_quantize_range():
    x = torch.tensor([-5.0, 100.4, 70000.0], dtype=DT)
    q = quantize_16bit(x)
    assert q.tolist() == [0.0, 100.0, 65535.0]


@pytest.mark.unit
def test_poulsen_widths_are_a_thin_plate():
    # Poulsen 2017: rock << roll ~ par -> the resolution element is a thin plate
    w = poulsen_resolution_widths(3.0, two_theta_deg=20.0)
    assert w["sigma_rock"] < w["sigma_roll"]
    assert w["sigma_rock"] < w["sigma_par"]
    assert all(v > 0 for v in w.values())
    # scales linearly with |Q0|
    w2 = poulsen_resolution_widths(6.0, two_theta_deg=20.0)
    assert abs(w2["sigma_rock"] / w["sigma_rock"] - 2.0) < 1e-9
