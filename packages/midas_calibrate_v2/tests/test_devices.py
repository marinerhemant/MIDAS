"""Device compatibility smoke tests.

Exercise the v2 forward model + autograd on each available device.  CPU
is the production target; CUDA / MPS are best-effort.  Tests skip when
the requested device isn't available rather than failing.
"""
from __future__ import annotations

import pytest
import torch


_DEVICES = ["cpu"]
if torch.cuda.is_available():
    _DEVICES.append("cuda")
if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    _DEVICES.append("mps")


@pytest.mark.parametrize("device", _DEVICES)
def test_geometry_forward_device(device):
    from midas_calibrate_v2.forward.geometry import pixel_to_REta

    if device == "mps":
        # MPS needs float32 — promote.
        dtype = torch.float32
    else:
        dtype = torch.float64

    Y = torch.tensor([100.0, 200.0, 300.0], dtype=dtype, device=device)
    Z = torch.tensor([100.0, 250.0, 400.0], dtype=dtype, device=device)
    Lsd = torch.tensor(1_000_000.0, dtype=dtype, device=device, requires_grad=True)
    BC_y = torch.tensor(1024.0, dtype=dtype, device=device, requires_grad=True)
    BC_z = torch.tensor(1024.0, dtype=dtype, device=device, requires_grad=True)
    tx = torch.tensor(0.0, dtype=dtype, device=device, requires_grad=True)
    ty = torch.tensor(0.1, dtype=dtype, device=device, requires_grad=True)
    tz = torch.tensor(0.2, dtype=dtype, device=device, requires_grad=True)
    p = torch.zeros(15, dtype=dtype, device=device, requires_grad=True)
    parallax = torch.tensor(0.0, dtype=dtype, device=device, requires_grad=True)
    pxY = torch.tensor(200.0, dtype=dtype, device=device, requires_grad=True)
    pxZ = torch.tensor(200.0, dtype=dtype, device=device, requires_grad=True)
    rho_d = torch.tensor(1500.0, dtype=dtype, device=device)
    out = pixel_to_REta(
        Y, Z, Lsd=Lsd, BC_y=BC_y, BC_z=BC_z,
        tx=tx, ty=ty, tz=tz, p_coeffs=p, parallax=parallax,
        pxY=pxY, pxZ=pxZ, rho_d=rho_d,
    )
    loss = out.R_px.sum() + out.eta_deg.sum()
    loss.backward()
    assert Lsd.grad is not None
    assert torch.isfinite(Lsd.grad).item()


@pytest.mark.parametrize("device", _DEVICES)
def test_pseudo_strain_device(device):
    from midas_calibrate_v2.loss.pseudo_strain import pseudo_strain_residual

    dtype = torch.float32 if device == "mps" else torch.float64
    Y = torch.tensor([100.0, 200.0], dtype=dtype, device=device)
    Z = torch.tensor([100.0, 250.0], dtype=dtype, device=device)
    rtt = torch.tensor([5.0, 7.0], dtype=dtype, device=device)
    p = {
        "Lsd": torch.tensor(1e6, dtype=dtype, device=device, requires_grad=True),
        "BC_y": torch.tensor(1024.0, dtype=dtype, device=device, requires_grad=True),
        "BC_z": torch.tensor(1024.0, dtype=dtype, device=device, requires_grad=True),
        "tx": torch.tensor(0.0, dtype=dtype, device=device),
        "ty": torch.tensor(0.0, dtype=dtype, device=device, requires_grad=True),
        "tz": torch.tensor(0.0, dtype=dtype, device=device, requires_grad=True),
        "Parallax": torch.tensor(0.0, dtype=dtype, device=device),
        "pxY": torch.tensor(200.0, dtype=dtype, device=device, requires_grad=True),
        "pxZ": torch.tensor(200.0, dtype=dtype, device=device),
    }
    for i in range(15):
        p[f"p{i}"] = torch.tensor(0.0, dtype=dtype, device=device, requires_grad=True)
    r = pseudo_strain_residual(
        Y, Z, rtt, p,
        rho_d=torch.tensor(1500.0, dtype=dtype, device=device),
    )
    r.sum().backward()
    assert p["Lsd"].grad is not None


@pytest.mark.parametrize("device", _DEVICES)
def test_snip_device(device):
    from midas_calibrate_v2.forward.snip import subtract_snip_background

    dtype = torch.float32 if device == "mps" else torch.float64
    M = 64
    R = torch.linspace(-5, 5, M, dtype=dtype, device=device)
    # Synthetic: gaussian peak on a smooth bg.
    peak = 100.0 * torch.exp(-0.5 * (R - 0.0) ** 2 / 1.0)
    bg_truth = 5.0 + 0.5 * R
    y = peak + bg_truth + 0.1 * torch.randn_like(R)
    bg_corrected = subtract_snip_background(y.unsqueeze(0), window_max=10).squeeze(0)
    # Peak amplitude should be largely preserved post-bg-subtraction.
    assert bg_corrected.max() > 0.5 * 100.0
