"""Unit tests for midas_pipeline.recon.mlem.

Synthetic disk phantom → forward Radon → MLEM/OSEM → assert recon
recovery RMSE below threshold. Torch path tested for autograd +
multi-device (CUDA / MPS) availability.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from midas_pipeline.recon import (
    back_project,
    forward_project,
    mlem_recon,
    osem_recon,
)


def _disk_phantom(N: int, radius: float = 0.3, value: float = 1.0) -> np.ndarray:
    """Centered disk phantom (NumPy)."""
    coord = np.linspace(-1.0, 1.0, N)
    xx, yy = np.meshgrid(coord, coord)
    return ((xx ** 2 + yy ** 2) < radius ** 2).astype(np.float64) * value


def _angles(n_thetas: int = 36) -> np.ndarray:
    return np.linspace(0.0, 180.0, n_thetas, endpoint=False)


# ---------------------------------------------------------------------------
# Numpy MLEM / OSEM recovery
# ---------------------------------------------------------------------------


def test_mlem_recon_disk_phantom_rmse():
    N = 16
    phantom = _disk_phantom(N)
    angles = _angles(36)
    sino = forward_project(phantom, angles)
    recon = mlem_recon(sino, angles, n_iter=50)
    # Normalize both before comparing; MLEM produces unnormalized intensities.
    p = phantom / max(phantom.max(), 1e-12)
    r = recon / max(recon.max(), 1e-12)
    rmse = float(np.sqrt(np.mean((p - r) ** 2)))
    assert rmse < 0.4, f"MLEM RMSE too high: {rmse:.3f}"


def test_osem_recon_disk_phantom_rmse():
    N = 16
    phantom = _disk_phantom(N)
    angles = _angles(36)
    sino = forward_project(phantom, angles)
    recon = osem_recon(sino, angles, n_iter=20, n_subsets=4)
    p = phantom / max(phantom.max(), 1e-12)
    r = recon / max(recon.max(), 1e-12)
    rmse = float(np.sqrt(np.mean((p - r) ** 2)))
    assert rmse < 0.4, f"OSEM RMSE too high: {rmse:.3f}"


# ---------------------------------------------------------------------------
# Torch path: agreement with NumPy
# ---------------------------------------------------------------------------


def test_forward_project_torch_matches_numpy():
    N = 12
    phantom_np = _disk_phantom(N)
    angles_np = _angles(18)
    sino_np = forward_project(phantom_np, angles_np)

    phantom_t = torch.as_tensor(phantom_np, dtype=torch.float64)
    angles_t = torch.as_tensor(angles_np, dtype=torch.float64)
    sino_t = forward_project(phantom_t, angles_t)

    assert isinstance(sino_t, torch.Tensor)
    np.testing.assert_allclose(sino_t.detach().cpu().numpy(), sino_np, atol=1e-10)


def test_back_project_torch_matches_numpy():
    N = 12
    angles_np = _angles(18)
    rng = np.random.default_rng(0)
    sino_np = rng.uniform(0.0, 1.0, size=(angles_np.shape[0], N)).astype(np.float64)

    img_np = back_project(sino_np, angles_np, N)
    sino_t = torch.as_tensor(sino_np, dtype=torch.float64)
    angles_t = torch.as_tensor(angles_np, dtype=torch.float64)
    img_t = back_project(sino_t, angles_t, N)
    assert isinstance(img_t, torch.Tensor)
    np.testing.assert_allclose(img_t.detach().cpu().numpy(), img_np, atol=1e-10)


def test_mlem_torch_matches_numpy():
    N = 10
    phantom = _disk_phantom(N)
    angles = _angles(12)
    sino_np = forward_project(phantom, angles)
    recon_np = mlem_recon(sino_np, angles, n_iter=10)

    sino_t = torch.as_tensor(sino_np, dtype=torch.float64)
    angles_t = torch.as_tensor(angles, dtype=torch.float64)
    recon_t = mlem_recon(sino_t, angles_t, n_iter=10)
    assert isinstance(recon_t, torch.Tensor)
    np.testing.assert_allclose(
        recon_t.detach().cpu().numpy(), recon_np, atol=1e-8, rtol=1e-6,
    )


def test_osem_torch_matches_numpy():
    N = 10
    phantom = _disk_phantom(N)
    angles = _angles(12)
    sino_np = forward_project(phantom, angles)
    recon_np = osem_recon(sino_np, angles, n_iter=5, n_subsets=2)

    sino_t = torch.as_tensor(sino_np, dtype=torch.float64)
    angles_t = torch.as_tensor(angles, dtype=torch.float64)
    recon_t = osem_recon(sino_t, angles_t, n_iter=5, n_subsets=2)
    np.testing.assert_allclose(
        recon_t.detach().cpu().numpy(), recon_np, atol=1e-8, rtol=1e-6,
    )


# ---------------------------------------------------------------------------
# Torch differentiability
# ---------------------------------------------------------------------------


def test_forward_project_torch_is_differentiable():
    N = 8
    img = torch.rand((N, N), dtype=torch.float64, requires_grad=True)
    angles = torch.linspace(0.0, 180.0, 6, dtype=torch.float64)
    sino = forward_project(img, angles)
    loss = sino.sum()
    loss.backward()
    assert img.grad is not None
    assert torch.isfinite(img.grad).all()


def test_back_project_torch_is_differentiable():
    N = 8
    angles = torch.linspace(0.0, 180.0, 6, dtype=torch.float64)
    sino = torch.rand((6, N), dtype=torch.float64, requires_grad=True)
    img = back_project(sino, angles, N)
    loss = img.sum()
    loss.backward()
    assert sino.grad is not None
    assert torch.isfinite(sino.grad).all()


def test_mlem_torch_is_differentiable():
    N = 8
    angles = torch.linspace(0.0, 180.0, 6, dtype=torch.float64)
    img = torch.rand((N, N), dtype=torch.float64) * 0.5 + 0.5
    sino = forward_project(img, angles).clone().detach().requires_grad_(True)
    recon = mlem_recon(sino, angles, n_iter=3)
    loss = recon.sum()
    loss.backward()
    assert sino.grad is not None
    assert torch.isfinite(sino.grad).all()


# ---------------------------------------------------------------------------
# Device portability (CUDA + MPS), skipped on machines without them
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_mlem_runs_on_cuda():
    N = 8
    phantom = torch.as_tensor(_disk_phantom(N), dtype=torch.float64, device="cuda")
    angles = torch.linspace(0.0, 180.0, 6, dtype=torch.float64, device="cuda")
    sino = forward_project(phantom, angles)
    recon = mlem_recon(sino, angles, n_iter=3)
    assert recon.device.type == "cuda"


@pytest.mark.skipif(
    not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()),
    reason="MPS not available",
)
def test_mlem_runs_on_mps():
    # MPS only supports fp32 well; use fp32 to avoid backend warnings.
    N = 8
    phantom = torch.as_tensor(_disk_phantom(N), dtype=torch.float32, device="mps")
    angles = torch.linspace(0.0, 180.0, 6, dtype=torch.float32, device="mps")
    sino = forward_project(phantom, angles)
    recon = mlem_recon(sino, angles, n_iter=3)
    assert recon.device.type == "mps"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_osem_runs_on_cuda():
    N = 8
    phantom = torch.as_tensor(_disk_phantom(N), dtype=torch.float64, device="cuda")
    angles = torch.linspace(0.0, 180.0, 6, dtype=torch.float64, device="cuda")
    sino = forward_project(phantom, angles)
    recon = osem_recon(sino, angles, n_iter=2, n_subsets=2)
    assert recon.device.type == "cuda"


@pytest.mark.skipif(
    not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()),
    reason="MPS not available",
)
def test_osem_runs_on_mps():
    N = 8
    phantom = torch.as_tensor(_disk_phantom(N), dtype=torch.float32, device="mps")
    angles = torch.linspace(0.0, 180.0, 6, dtype=torch.float32, device="mps")
    sino = forward_project(phantom, angles)
    recon = osem_recon(sino, angles, n_iter=2, n_subsets=2)
    assert recon.device.type == "mps"


# ---------------------------------------------------------------------------
# The 2026-09 rewrite: exact adjoint, measured zeros, padded grid (ESRF ma5608)
# ---------------------------------------------------------------------------


def test_padded_projector_is_the_exact_adjoint():
    import importlib
    M = importlib.import_module("midas_pipeline.recon.mlem")
    g = torch.Generator().manual_seed(0)
    ang = torch.linspace(-80, 85, 17, dtype=torch.float64)
    for N, Mb, chunk in ((41, 29, 2e7), (29, 29, 2e7), (41, 29, 500)):
        P = M._Projector(ang, N, Mb, chunk_elems=chunk)
        x = torch.rand(N, N, dtype=torch.float64, generator=g)
        r = torch.rand(17, Mb, dtype=torch.float64, generator=g)
        lhs, rhs = float((P.forward(x) * r).sum()), float((x * P.adjoint(r)).sum())
        assert abs(lhs - rhs) <= 1e-12 * abs(lhs)


def test_padded_projector_matches_forward_project_at_pad_zero():
    import importlib
    M = importlib.import_module("midas_pipeline.recon.mlem")
    img = _disk_phantom(33, radius=0.4)
    ang = _angles(20)
    P = M._Projector(torch.as_tensor(ang), 33, 33)
    np.testing.assert_allclose(P.forward(torch.as_tensor(img)).numpy(),
                               forward_project(img, ang), atol=1e-9)


def test_measured_zeros_constrain_an_off_centre_grain():
    """The old recipe dropped zero cells from sensitivity and update, so the
    image was unconstrained wherever the grain is absent (0.15 accuracy on a
    Voronoi phantom). An off-centre disc must come back where it is, and
    nowhere else."""
    N = 41
    yy, xx = np.mgrid[0:N, 0:N]
    truth = ((yy - 12) ** 2 + (xx - 27) ** 2 < 25).astype(float)
    ang = np.sort(np.random.default_rng(1).uniform(-90, 90, 60))
    rec = mlem_recon(forward_project(truth, ang), ang, n_iter=60)
    m = rec > 0.5 * rec.max()
    iou = (m & (truth > 0)).sum() / (m | (truth > 0)).sum()
    assert iou > 0.85
    assert rec[truth == 0].sum() < 0.1 * rec[truth > 0].sum()


def test_sample_wider_than_the_field_stays_finite():
    """Interior tomography: material outside the scanned field still crosses
    the beam. The reconstruction must stay finite and put the in-field grain
    where it is."""
    Nf, pad = 41, 20
    Nb = Nf + 2 * pad
    yy, xx = np.mgrid[0:Nb, 0:Nb]
    outside = ((yy - 8) ** 2 + (xx - Nb // 2) ** 2 < 36).astype(float)   # beyond the field
    grain = ((yy - pad - 20) ** 2 + (xx - pad - 14) ** 2 < 36).astype(float)
    assert outside[pad:pad + Nf, pad:pad + Nf].sum() == 0
    ang = np.sort(np.random.default_rng(2).uniform(-90, 90, 60))
    full = forward_project(grain + outside, ang)
    sino = full[:, pad:pad + Nf]                        # detector sees only the field
    rec = mlem_recon(sino, ang, n_iter=50, pad=1.0)
    assert np.isfinite(rec).all()
    t = grain[pad:pad + Nf, pad:pad + Nf] > 0
    assert rec[t].mean() > 3 * rec[~t].mean()
