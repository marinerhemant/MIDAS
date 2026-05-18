"""Item 16 — per-ring χ²/dof diagnostic."""
from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from midas_integrate_v2.diagnostics import per_ring_chi_squared


def test_uniform_ring_gives_low_chi2():
    n_eta, n_r = 72, 64
    eta = torch.linspace(-180.0, 180.0, n_eta, dtype=torch.float64)
    R = torch.linspace(0.0, 64.0, n_r, dtype=torch.float64)
    int2d = torch.zeros(n_eta, n_r, dtype=torch.float64)
    # Plant a perfectly uniform ring at R=20, intensity 1000 per η-bin
    ring_idx = (R - 20.0).abs().argmin().item()
    int2d[:, ring_idx] = 1000.0
    chi2 = per_ring_chi_squared(int2d, eta, R, torch.tensor([20.0]),
                                  capture_radius_px=2.0)
    # Uniform ring: variance = 0 → χ²/dof ~ 0
    assert float(chi2[0]) < 0.01


def test_non_uniform_ring_gives_high_chi2():
    n_eta, n_r = 72, 64
    eta = torch.linspace(-180.0, 180.0, n_eta, dtype=torch.float64)
    R = torch.linspace(0.0, 64.0, n_r, dtype=torch.float64)
    int2d = torch.zeros(n_eta, n_r, dtype=torch.float64)
    ring_idx = (R - 20.0).abs().argmin().item()
    # Plant a strongly modulated η profile: 500 + 300 sin(η)
    int2d[:, ring_idx] = 500.0 + 300.0 * torch.sin(eta * math.pi / 180.0)
    chi2 = per_ring_chi_squared(int2d, eta, R, torch.tensor([20.0]),
                                  capture_radius_px=2.0)
    assert float(chi2[0]) > 5.0


def test_per_ring_returns_correct_shape_and_nan_for_outside_ring():
    n_eta, n_r = 36, 32
    eta = torch.linspace(-180.0, 180.0, n_eta, dtype=torch.float64)
    R = torch.linspace(0.0, 32.0, n_r, dtype=torch.float64)
    int2d = torch.ones(n_eta, n_r, dtype=torch.float64) * 100.0
    rings = torch.tensor([10.0, 20.0, 1000.0])  # last one out of range
    chi2 = per_ring_chi_squared(int2d, eta, R, rings, capture_radius_px=1.0)
    assert chi2.shape == (3,)
    assert float(chi2[0]) < 0.01
    assert float(chi2[1]) < 0.01
    assert torch.isnan(chi2[2])


def test_explicit_sigma_used():
    n_eta, n_r = 36, 16
    eta = torch.linspace(-180.0, 180.0, n_eta, dtype=torch.float64)
    R = torch.linspace(0.0, 16.0, n_r, dtype=torch.float64)
    int2d = torch.ones(n_eta, n_r, dtype=torch.float64) * 100.0
    int2d[0, 8] = 500.0  # one outlier η-bin at ring R=8
    sigma2d = torch.full_like(int2d, 10.0)
    chi2 = per_ring_chi_squared(
        int2d, eta, R, torch.tensor([8.0]),
        capture_radius_px=1.0, sigma2d=sigma2d,
    )
    # The single outlier on n_eta-1 dof gives a noticeable χ²
    assert float(chi2[0]) > 0.5
