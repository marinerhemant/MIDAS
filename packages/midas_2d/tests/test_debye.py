"""MD-coupled engine: cross-check vs analytic shape factor, gradients, ensemble."""
import math

import pytest
import torch

from midas_2d import (
    cdse_supercell,
    coherent_intensity,
    debye_intensity,
    ensemble_intensity,
    rod_intensity,
    build_crystal_tensor,
)

DT = torch.float64
A = 6.077  # CdSe zinc-blende lattice constant


def _rod_qvec(l_grid, a=A, hk=(1.0, 1.0)):
    """Q-vectors along the (h, k, l) rod for a cubic cell: Q = 2pi/a (h,k,l)."""
    h = torch.full_like(l_grid, hk[0])
    k = torch.full_like(l_grid, hk[1])
    hkl = torch.stack([h, k, l_grid], dim=-1)
    return (2.0 * math.pi / a) * hkl


@pytest.mark.unit
def test_explicit_amplitude_matches_analytic_shape_factor():
    """Coherent amplitude over an explicit supercell reproduces the Phase-1
    |F|^2 . |S|^2 fringes along the rod (the lattice sum factorises exactly)."""
    nx, ny, nz = 6, 6, 4
    coords, elements, _cell = cdse_supercell((nx, ny, nz), dtype=DT)

    l = torch.linspace(0.6, 1.4, 400, dtype=DT)
    qv = _rod_qvec(l)
    I_md = coherent_intensity(coords, elements, qv)

    ct = build_crystal_tensor()
    hkl = torch.stack([torch.ones_like(l), torch.ones_like(l), l], dim=-1)
    N = torch.tensor([float(nx), float(ny), float(nz)], dtype=DT)
    I_an = rod_intensity(ct, hkl, N, wavelength_A=1.0, apply_lp=False)

    # Compare normalised shapes (absolute scale matches too, but normalising
    # avoids unit-cell vs supercell bookkeeping).
    a = I_md / I_md.max()
    b = I_an / I_an.max()
    # Peak location and fringe structure must coincide.
    assert torch.allclose(a, b, atol=2e-3), float((a - b).abs().max())


@pytest.mark.unit
def test_debye_powder_is_finite_and_peaks_at_bragg():
    """Orientationally-averaged Debye intensity is finite and shows a peak near
    the (111) magnitude q_111 = 2pi/a * sqrt(3)."""
    coords, elements, _ = cdse_supercell((5, 5, 5), dtype=DT)
    q = torch.linspace(1.0, 3.5, 400, dtype=DT)
    I = debye_intensity(coords, elements, q)
    assert torch.isfinite(I).all()
    q111 = 2.0 * math.pi / A * math.sqrt(3.0)
    # global structure: intensity near q111 exceeds the local background midway.
    i_peak = int(torch.argmin((q - q111).abs()))
    assert I[i_peak] > I.mean()


@pytest.mark.autograd
def test_intensity_differentiable_wrt_coordinates():
    """The novel hook: gradients flow back to atomic positions."""
    coords, elements, _ = cdse_supercell((3, 3, 3), dtype=DT)
    coords = coords.clone().requires_grad_(True)
    qv = _rod_qvec(torch.linspace(0.7, 1.3, 20, dtype=DT))
    I = coherent_intensity(coords, elements, qv)
    I.sum().backward()
    assert coords.grad is not None
    assert torch.isfinite(coords.grad).all()
    assert coords.grad.abs().sum() > 0


@pytest.mark.unit
def test_thermal_spread_suppresses_intensity_anisotropically():
    """Disorder emerges from the coordinates: adding larger out-of-plane (z)
    displacement suppresses an out-of-plane reflection more than the same
    in-plane displacement would -- the anisotropic Debye-Waller signature, with
    no DWF assumed."""
    torch.manual_seed(0)
    coords, elements, _ = cdse_supercell((5, 5, 4), dtype=DT)

    # A reflection with out-of-plane character: (1,1,3) rod point.
    qv = _rod_qvec(torch.tensor([3.0], dtype=DT))

    def ensemble_with_sigma(sx, sy, sz, nframes=24):
        sig = torch.tensor([sx, sy, sz], dtype=DT)
        frames = coords[None] + sig * torch.randn(nframes, *coords.shape, dtype=DT)
        return ensemble_intensity(frames, elements, qv, coherent=True).item()

    I0 = ensemble_with_sigma(0.0, 0.0, 0.0)
    I_inplane = ensemble_with_sigma(0.15, 0.15, 0.0)
    I_outplane = ensemble_with_sigma(0.0, 0.0, 0.15)
    # Out-of-plane motion couples to the l-component of Q here -> stronger
    # suppression of this rod point than pure in-plane motion.
    assert I_outplane < I_inplane < I0


@pytest.mark.device
@pytest.mark.parametrize("device", ["cpu", "cuda", "mps"])
def test_debye_device(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("no CUDA")
    if device == "mps" and not (hasattr(torch.backends, "mps")
                                and torch.backends.mps.is_available()):
        pytest.skip("no MPS")
    dt = torch.float32 if device == "mps" else DT
    coords, elements, _ = cdse_supercell((3, 3, 3), dtype=dt, device=device)
    q = torch.linspace(1.0, 3.0, 50, dtype=dt, device=device)
    I = debye_intensity(coords, elements, q)
    assert I.device.type == device and torch.isfinite(I).all()
