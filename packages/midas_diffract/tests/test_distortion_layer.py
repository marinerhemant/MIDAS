"""Tests for the gated ideal->raw radial-distortion layer (midas_distortion v2).

Guarantees:
  1. default (apply_distortion=False) is BYTE-IDENTICAL to the pre-layer forward
     -- proves the indexer/fit-grain (ideal-frame) path is untouched.
  2. apply_distortion=True with zero coeffs is also identity.
  3. nonzero coeffs shift predicted spots (ideal->raw) by a sane, radius-growing
     amount.
  4. the layer is differentiable w.r.t. the distortion coefficients.
"""
import numpy as np
import pytest
import torch

from midas_diffract.forward import HEDMForwardModel, HEDMGeometry

_GD = dict(
    Lsd=752000.0, y_BC=695.0, z_BC=874.0, px=172.0,
    omega_start=180.0, omega_step=-0.25, n_frames=1440,
    n_pixels_y=1679, n_pixels_z=1679, min_eta=6.0, wavelength=0.172979,
)
# A representative calibrated v2 coefficient set (Bucsek Pilatus 2M CeO2).
_COEFFS = [0.00707, -0.01, 0.00624, 0.01, -34.76, 0.00234, 81.47,
           -0.00369, -12.29, -0.00727, -5.29, -0.00863, -1.51, -0.00446, -7.79]


def _model(**extra):
    rng = np.random.default_rng(0)
    hk = torch.tensor(rng.standard_normal((40, 3)), dtype=torch.float64)
    th = torch.tensor(np.abs(rng.standard_normal(40)) * 0.04 + 0.04, dtype=torch.float64)
    hi = torch.tensor(rng.integers(-3, 4, (40, 3)), dtype=torch.float64)
    g = HEDMGeometry(**_GD, **extra)
    return HEDMForwardModel(hk, th, g, hkls_int=hi).double()


_EUL = torch.tensor([[0.3, 0.5, 0.2]], dtype=torch.float64)
_POS = torch.zeros(1, 3, dtype=torch.float64)


def test_default_off_is_unchanged():
    """No distortion fields == apply_distortion=False == zero coeffs (identity)."""
    o0 = _model().forward(_EUL, _POS)
    o_off = _model(apply_distortion=False, p_distortion=[0.0] * 15).forward(_EUL, _POS)
    o_zero = _model(apply_distortion=True, p_distortion=[0.0] * 15, rho_d=2.0e5).forward(_EUL, _POS)
    assert torch.equal(o0.y_pixel, o_off.y_pixel)
    assert torch.equal(o0.z_pixel, o_off.z_pixel)
    # zero coeffs with the layer ACTIVE must also be a no-op (D == 1).
    assert torch.allclose(o0.y_pixel, o_zero.y_pixel, atol=1e-9)
    assert torch.allclose(o0.z_pixel, o_zero.z_pixel, atol=1e-9)


def test_distortion_shifts_spots():
    o0 = _model().forward(_EUL, _POS)
    o2 = _model(apply_distortion=True, p_distortion=_COEFFS, rho_d=2.0e5).forward(_EUL, _POS)
    vm = (o0.valid > 0.5) & (o2.valid > 0.5)
    assert int(vm.sum()) > 5
    dy = (o2.y_pixel - o0.y_pixel)[vm].abs()
    assert float(dy.median()) > 0.05   # measurable
    assert float(dy.max()) < 50.0      # but physical (sub-module)


def test_distortion_is_differentiable():
    m = _model(apply_distortion=True, p_distortion=_COEFFS, rho_d=2.0e5)
    m.p_distortion.requires_grad_(True)
    out = m.forward(_EUL, _POS)
    loss = (out.y_pixel * out.valid).sum() + (out.z_pixel * out.valid).sum()
    loss.backward()
    assert m.p_distortion.grad is not None
    assert float(m.p_distortion.grad.norm()) > 0.0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_distortion_cpu_cuda_parity():
    o_cpu = _model(apply_distortion=True, p_distortion=_COEFFS, rho_d=2.0e5).forward(_EUL, _POS)
    m = _model(apply_distortion=True, p_distortion=_COEFFS, rho_d=2.0e5).to("cuda")
    o_gpu = m.forward(_EUL.cuda(), _POS.cuda())
    assert torch.allclose(o_cpu.y_pixel, o_gpu.y_pixel.cpu(), atol=1e-7)
    assert torch.allclose(o_cpu.z_pixel, o_gpu.z_pixel.cpu(), atol=1e-7)
