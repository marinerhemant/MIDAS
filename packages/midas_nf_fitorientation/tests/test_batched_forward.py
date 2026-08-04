"""Tests for ``forward_batched_grains``: the per-grain reshape that
turns the model's ``(2*B, M)`` output into the ``(B, K=2, M)`` layout
the batched-NM polish needs."""
from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from midas_nf_fitorientation.params import FitParams
from midas_nf_fitorientation.soft_overlap import (
    build_forward_model, forward_batched_grains,
)


def _make_params(n_distances: int = 1) -> FitParams:
    p = FitParams()
    p.n_distances = n_distances
    Lsds = [1_000_000.0, 2_000_000.0][:n_distances]
    p.Lsd = list(Lsds)
    p.ybc = [64.0] * n_distances
    p.zbc = [64.0] * n_distances
    p.px = 200.0
    p.omega_start = -180.0; p.omega_step_raw = 1.0
    p.start_nr = 1; p.end_nr = 30
    p.exclude_pole_angle = 6.0
    p.wavelength = 0.172979
    p.lattice_constant = (4.08, 4.08, 4.08, 90.0, 90.0, 90.0)
    p.n_pixels_y = 128; p.n_pixels_z = 128
    p.tx = p.ty = p.tz = 0.0
    p.wedge = 0.0
    return p


def test_batched_forward_matches_single_grain_calls():
    """Run the model on B grains in one call and B individual calls;
    the per-grain spots tensors must agree element-for-element."""
    p = _make_params(n_distances=1)
    hkls_int = np.array([[1, 0, 0], [1, 1, 0]], dtype=np.float64)
    model = build_forward_model(p, hkls_int, device="cpu", dtype=torch.float64)

    rng = np.random.default_rng(7)
    eulers_np = rng.standard_normal((4, 3)) * 0.5
    positions_np = rng.standard_normal((4, 3)) * 10.0

    eul_b = torch.tensor(eulers_np, dtype=torch.float64)
    pos_b = torch.tensor(positions_np, dtype=torch.float64)

    fn_b, val_b, yp_b, zp_b = forward_batched_grains(model, eul_b, pos_b)

    for i in range(4):
        eul_i = eul_b[i : i + 1]
        pos_i = pos_b[i : i + 1]
        spots_i = model(eul_i, pos_i)

        assert torch.allclose(fn_b[i], spots_i.frame_nr, atol=1e-9), (
            f"grain {i}: frame_nr mismatch"
        )
        assert torch.allclose(val_b[i], spots_i.valid, atol=1e-9), (
            f"grain {i}: valid mismatch"
        )
        # For D=1 layered the model collapses the D dim, so spots.y_pixel
        # has shape (K, M) — same as fn_b's [i] slice has (K, M).
        assert torch.allclose(yp_b[i], spots_i.y_pixel, atol=1e-9), (
            f"grain {i}: y_pixel mismatch"
        )
        assert torch.allclose(zp_b[i], spots_i.z_pixel, atol=1e-9), (
            f"grain {i}: z_pixel mismatch"
        )


def test_batched_forward_multi_distance_keeps_D_axis():
    """For ``n_distances > 1`` the output must carry a leading ``D``
    axis on ``y_pixel`` / ``z_pixel`` (matching the model's layered
    convention) so :meth:`ObsVolume.hard_fraction` can do per-distance
    AND-product."""
    p = _make_params(n_distances=2)
    hkls_int = np.array([[1, 0, 0]], dtype=np.float64)
    model = build_forward_model(p, hkls_int, device="cpu", dtype=torch.float64)

    eul_b = torch.tensor([[0.5, 0.6, 0.7], [0.4, 0.5, 0.6]], dtype=torch.float64)
    pos_b = torch.zeros_like(eul_b)
    fn_b, val_b, yp_b, zp_b = forward_batched_grains(model, eul_b, pos_b)

    assert fn_b.shape[0] == 2          # B
    assert fn_b.ndim == 3              # (B, K, M)
    assert yp_b.ndim == 4              # (D, B, K, M)
    assert yp_b.shape[0] == 2          # D
    assert yp_b.shape[1] == 2          # B
    assert yp_b.shape[2] == 2          # K


def test_batched_forward_rejects_misshaped_inputs():
    p = _make_params()
    hkls_int = np.array([[1, 0, 0]], dtype=np.float64)
    model = build_forward_model(p, hkls_int, device="cpu", dtype=torch.float64)

    eul = torch.zeros(3, dtype=torch.float64)            # missing batch dim
    pos = torch.zeros(1, 3, dtype=torch.float64)
    with pytest.raises(ValueError, match=r"\(B, 3\)"):
        forward_batched_grains(model, eul, pos)

    eul = torch.zeros(2, 3, dtype=torch.float64)
    pos = torch.zeros(3, 3, dtype=torch.float64)         # batch mismatch
    with pytest.raises(ValueError, match="match eul shape"):
        forward_batched_grains(model, eul, pos)
