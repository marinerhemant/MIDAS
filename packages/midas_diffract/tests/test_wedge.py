"""Tests for the global wedge angle (rotation-axis / beam non-orthogonality).

The wedge enters the forward as a rigorous geometric transformation
(``midas_diffract.forward`` module doc, "Wedge convention"): the rotation
axis is ``R_y(-W) e_z = (-sin W, 0, cos W)`` and a stage-frame G is seen as
``G_lab = R_y(-W) @ R_z(omega) @ G``. With wedge=0
the forward should be bit-identical to the existing no-wedge code; with
non-zero wedge the omega solver and eta extraction should produce
self-consistent predictions, and the wedge value should be
gradient-recoverable from synthetic observations.

Run with:
    cd packages/midas_diffract
    python -m pytest tests/test_wedge.py -v
"""
import math

import pytest
import torch

from midas_diffract.forward import HEDMForwardModel, HEDMGeometry


DEG2RAD = math.pi / 180.0


def _au_hkls():
    """Cubic Au HKLs through the first few rings (FCC selection rule)."""
    a = 4.08
    wl = 0.22291
    int_hkls = []
    for h in range(-3, 4):
        for k in range(-3, 4):
            for l in range(-3, 4):
                if h == 0 and k == 0 and l == 0:
                    continue
                parities = (h % 2, k % 2, l % 2)
                if not (parities == (0, 0, 0) or parities == (1, 1, 1)):
                    continue
                int_hkls.append((h, k, l))
    int_hkls = torch.tensor(int_hkls, dtype=torch.float64)
    cart = int_hkls / a
    g_norm = torch.linalg.norm(cart, dim=1)
    sin_th = wl * g_norm / 2.0
    keep = sin_th < 0.95
    cart = cart[keep]; sin_th = sin_th[keep]
    thetas = torch.asin(sin_th)
    return cart.float(), thetas.float(), int_hkls[keep].float()


def _ff_geometry(wedge_deg=0.0):
    return HEDMGeometry(
        Lsd=1.0e6, y_BC=1024.0, z_BC=1024.0, px=200.0,
        omega_start=180.0, omega_step=-0.25, n_frames=1440,
        n_pixels_y=2048, n_pixels_z=2048, min_eta=6.0,
        wavelength=0.22291,
        wedge=wedge_deg,
    )


def test_wedge_zero_is_bit_identical_to_no_wedge_default():
    """The default wedge=0 path must reproduce predictions to within
    floating-point noise of an explicitly-zero wedge geometry."""
    hkls, thetas, _ = _au_hkls()
    g_default = HEDMGeometry(
        Lsd=1.0e6, y_BC=1024.0, z_BC=1024.0, px=200.0,
        omega_start=180.0, omega_step=-0.25, n_frames=1440,
        n_pixels_y=2048, n_pixels_z=2048, min_eta=6.0,
        wavelength=0.22291,
    )
    g_zero_wedge = _ff_geometry(0.0)

    m_def = HEDMForwardModel(hkls, thetas, g_default)
    m_zer = HEDMForwardModel(hkls, thetas, g_zero_wedge)

    euler = torch.tensor([[0.3, 0.5, 0.7]])
    pos = torch.zeros(1, 3)

    s_def = m_def(euler, pos)
    s_zer = m_zer(euler, pos)
    torch.testing.assert_close(s_def.omega, s_zer.omega)
    torch.testing.assert_close(s_def.eta, s_zer.eta)
    torch.testing.assert_close(s_def.y_pixel, s_zer.y_pixel)
    torch.testing.assert_close(s_def.z_pixel, s_zer.z_pixel)


def test_wedge_changes_omega():
    """A 0.5° wedge must shift predicted omegas by a non-trivial amount
    (greater than floating-point noise)."""
    hkls, thetas, _ = _au_hkls()
    m_no = HEDMForwardModel(hkls, thetas, _ff_geometry(0.0))
    m_w = HEDMForwardModel(hkls, thetas, _ff_geometry(0.5))

    torch.manual_seed(0)
    euler = torch.rand(3, 3) * (2 * math.pi)
    pos = torch.zeros(3, 3)

    s_no = m_no(euler, pos)
    s_w = m_w(euler, pos)
    valid_both = (s_no.valid > 0.5) & (s_w.valid > 0.5)
    dom = (s_w.omega - s_no.omega) * 180.0 / math.pi
    # Wrap to (-180, 180]
    dom = (dom + 180.0) % 360.0 - 180.0
    dom_valid = dom[valid_both]
    assert dom_valid.numel() > 0
    # Some shifts > 0.01° should be present at 0.5° wedge
    assert dom_valid.abs().max().item() > 1e-2, (
        "wedge appears not to shift omegas at all"
    )


def test_wedge_gradient_finite_difference():
    """Autograd gradient w.r.t. ``model.wedge`` should match a central
    finite-difference estimate on a smooth loss.

    All inputs are upgraded to float64 to keep the FD step out of the
    float32 noise floor.
    """
    hkls, thetas, _ = _au_hkls()
    geom = _ff_geometry(0.0)
    m = HEDMForwardModel(hkls.double(), thetas.double(), geom)
    m.wedge.requires_grad_(True)

    torch.manual_seed(1)
    euler = (torch.rand(3, 3, dtype=torch.float64) * (2 * math.pi))
    pos = torch.zeros(3, 3, dtype=torch.float64)

    with torch.no_grad():
        m.wedge.data.fill_(0.05)
        s0 = m(euler, pos)
        mask = s0.valid > 0.5

    def loss_fn():
        s = m(euler, pos)
        om = s.omega[mask]
        return (om * om).sum()

    loss = loss_fn()
    grad_auto = torch.autograd.grad(loss, m.wedge)[0].detach().clone().item()

    eps = 1e-5
    with torch.no_grad():
        m.wedge.data.fill_(0.05 + eps)
        l_p = loss_fn().item()
        m.wedge.data.fill_(0.05 - eps)
        l_m = loss_fn().item()
        m.wedge.data.fill_(0.05)
    grad_fd = (l_p - l_m) / (2 * eps)
    rel = abs(grad_auto - grad_fd) / (abs(grad_fd) + 1e-12)
    assert rel < 1e-4, (
        f"autograd vs FD mismatch on wedge: auto={grad_auto:.6e}, "
        f"fd={grad_fd:.6e}, rel={rel:.3e}"
    )


def test_lsd_delta_mm_reparameterisation():
    """Optimising ``_Lsd_delta_mm`` should be equivalent to optimising
    ``_Lsd`` directly, with a 1000x scale factor on the gradient.
    Specifically the prediction at delta=1mm should equal the prediction
    at _Lsd + 1000 um."""
    hkls, thetas, _ = _au_hkls()
    geom = _ff_geometry(0.0)
    m = HEDMForwardModel(hkls, thetas, geom)

    torch.manual_seed(2)
    euler = torch.rand(2, 3) * (2 * math.pi)
    pos = torch.zeros(2, 3)

    base_Lsd = float(m._Lsd[0].item())
    # Path A: bump _Lsd by 1000 um directly
    with torch.no_grad():
        m._Lsd.data.fill_(base_Lsd + 1000.0)
        s_a = m(euler, pos)
    # Path B: keep _Lsd at base, bump delta_mm by 1.0
    with torch.no_grad():
        m._Lsd.data.fill_(base_Lsd)
        m._Lsd_delta_mm.data.fill_(1.0)
        s_b = m(euler, pos)

    torch.testing.assert_close(s_a.omega, s_b.omega)
    torch.testing.assert_close(s_a.y_pixel, s_b.y_pixel)
    torch.testing.assert_close(s_a.z_pixel, s_b.z_pixel)


def test_lsd_delta_mm_gradient_propagates():
    """A loss that depends on Lsd should have a non-zero gradient on
    ``_Lsd_delta_mm`` (the optimiser-visible parameter)."""
    hkls, thetas, _ = _au_hkls()
    geom = _ff_geometry(0.0)
    m = HEDMForwardModel(hkls, thetas, geom)
    m._Lsd_delta_mm.requires_grad_(True)

    torch.manual_seed(3)
    euler = torch.rand(2, 3) * (2 * math.pi)
    pos = torch.zeros(2, 3)

    s = m(euler, pos)
    loss = (s.y_pixel * s.valid).pow(2).sum()
    loss.backward()
    assert m._Lsd_delta_mm.grad is not None
    assert m._Lsd_delta_mm.grad.abs().max().item() > 0.0
