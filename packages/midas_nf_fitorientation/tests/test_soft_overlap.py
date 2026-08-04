"""Tests for the differentiable soft FracOverlap surrogate."""
from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from midas_nf_fitorientation.obs_volume import ObsVolume
from midas_nf_fitorientation.params import FitParams
from midas_nf_fitorientation.soft_overlap import (
    GeometryOverrides,
    auto_sigma_px,
    build_forward_model,
    cartesian_B_matrix,
    hkls_cart_thetas,
    overrides,
    soft_overlap,
)


def _minimal_params() -> FitParams:
    """Tiny FitParams for unit tests — small detector, short scan,
    so the dense obs volume fits in a few MB.
    """
    p = FitParams()
    p.n_distances = 1
    p.Lsd = [1_000_000.0]
    p.ybc = [64.0]
    p.zbc = [64.0]
    p.px = 200.0
    p.omega_start = -180.0
    p.omega_step_raw = 1.0
    p.start_nr = 1
    p.end_nr = 30          # 30 frames is enough to exercise the indexing
    p.exclude_pole_angle = 6.0
    p.wavelength = 0.172979
    p.lattice_constant = (4.08, 4.08, 4.08, 90.0, 90.0, 90.0)
    p.n_pixels_y = 128
    p.n_pixels_z = 128
    p.tx = p.ty = p.tz = 0.0
    p.wedge = 0.0
    return p


def test_cartesian_B_cubic_is_diagonal():
    B = cartesian_B_matrix((4.08, 4.08, 4.08, 90.0, 90.0, 90.0))
    diag = np.diag(B)
    off = B - np.diag(diag)
    assert np.allclose(off, 0, atol=1e-9)
    # Reciprocal length 1/a
    assert np.allclose(diag, [1 / 4.08] * 3, atol=1e-9)


def test_hkls_cart_thetas_consistency():
    hkls_int = np.array([[1, 0, 0], [1, 1, 0], [1, 1, 1]], dtype=np.float64)
    G_cart, thetas = hkls_cart_thetas(
        hkls_int, (4.08, 4.08, 4.08, 90.0, 90.0, 90.0), 0.172979,
    )
    g_mag = np.linalg.norm(G_cart, axis=-1)
    expected = np.arcsin(g_mag * 0.172979 / 2.0)
    assert np.allclose(thetas, expected)


def test_build_forward_model_smoke():
    p = _minimal_params()
    hkls_int = np.array([[1, 0, 0], [1, 1, 0]], dtype=np.float64)
    model = build_forward_model(p, hkls_int, device="cpu", dtype=torch.float64)
    # Sanity: hkls/thetas registered, n_distances correct
    assert model.hkls.shape == (2, 3)
    assert model.thetas.shape == (2,)
    assert model.n_distances == 1


def test_overrides_round_trip():
    p = _minimal_params()
    hkls_int = np.array([[1, 0, 0]], dtype=np.float64)
    model = build_forward_model(p, hkls_int, device="cpu", dtype=torch.float64)
    orig_Lsd = model._Lsd.detach().clone()
    orig_tilts = model.tilts.detach().clone()

    new_Lsd = torch.tensor([1_500_000.0], dtype=torch.float64)
    new_tilts = torch.tensor([[0.1, 0.2, 0.3]], dtype=torch.float64)
    with overrides(model, GeometryOverrides(Lsd=new_Lsd, tilts=new_tilts)):
        assert torch.allclose(model._Lsd, new_Lsd)
        assert torch.allclose(model.tilts, new_tilts)
        assert model._has_tilts  # forced on under override
    # Restored after context exit
    assert torch.allclose(model._Lsd, orig_Lsd)
    assert torch.allclose(model.tilts, orig_tilts)


def test_soft_overlap_returns_scalar_in_unit_interval():
    p = _minimal_params()
    hkls_int = np.array([[1, 0, 0], [1, 1, 0]], dtype=np.float64)
    model = build_forward_model(p, hkls_int, device="cpu", dtype=torch.float64)
    obs = ObsVolume.from_dense_array(
        np.zeros((1, p.n_frames_per_distance, p.n_pixels_y, p.n_pixels_z),
                 dtype=np.float32),
        device="cpu", dtype=torch.float64,
    )
    eul = torch.tensor([0.5, 0.6, 0.7], dtype=torch.float64)
    pos = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float64)
    overlap = soft_overlap(model, obs, eul, pos, sigma_px=1.0)
    assert overlap.shape == ()
    # Empty obs ⇒ overlap is zero. But at least it should be finite,
    # bounded in [0, 1].
    assert 0.0 <= float(overlap) <= 1.0


def test_soft_overlap_gradient_flows_to_eulers():
    p = _minimal_params()
    hkls_int = np.array([[1, 0, 0]], dtype=np.float64)
    model = build_forward_model(p, hkls_int, device="cpu", dtype=torch.float64)
    # Lit a single pixel near the predicted location for a known orientation
    # so the gradient w.r.t. eulers is nonzero.
    obs_arr = np.zeros((1, p.n_frames_per_distance, p.n_pixels_y, p.n_pixels_z),
                        dtype=np.float32)
    obs_arr[0, 15, 64, 64] = 1.0
    obs = ObsVolume.from_dense_array(obs_arr, device="cpu", dtype=torch.float64)

    eul = torch.tensor([0.5, 0.6, 0.7], dtype=torch.float64, requires_grad=True)
    pos = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float64)
    overlap = soft_overlap(model, obs, eul, pos, sigma_px=2.0)
    # Even if overlap == 0 (spots far from lit pixel), backward should
    # not error.
    overlap.backward()
    assert eul.grad is not None
    assert eul.grad.shape == (3,)


def test_auto_sigma_px_lower_bound():
    # gs much smaller than px → σ clamped to 1
    assert auto_sigma_px(0.1, 200.0) == 1.0


def test_auto_sigma_px_scales_with_voxel():
    # Big voxel: σ scales with gs / (2*sqrt(3)*px)
    s = auto_sigma_px(1000.0, 200.0)
    assert s > 1.0
    # Override wins
    assert auto_sigma_px(1000.0, 200.0, override=2.5) == 2.5
