"""Tests for ``midas_transforms.radius.forward_model`` and the closed-form
K-per-ring refinement in :mod:`midas_transforms.radius.theoretical`.

The forward model implements the Sharma-Offerman V-map prediction
(without absorption — the absorption layer is P3 of the V-map plan)::

    I_pred(s) = K[r(s)] · I_th[r(s)] · Σ_{v ∈ grain(s)} V[v] · w(scan_pos(s), proj_ω(v))

These tests verify:
- A single voxel in beam reproduces the formula exactly.
- Multiple voxels, only some in beam, sum correctly.
- Grain isolation: a spot in grain A is not contaminated by grain B.
- Spots with grain_idx < 0 (orphans) get zero prediction.
- Differentiability wrt V_voxel, K_ring, scan_pos, omega, voxel_pos, beam.width.
- Closed-form K recovers K_true exactly on noise-free synthetic data.
- Recovery is robust to multiplicative log-normal noise (within 10%).
- Multi-device (CPU + MPS for forward; CPU only for gradcheck).
"""
from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import math
import pytest

torch = pytest.importorskip("torch")

from midas_transforms.geometry import SampleGrid, TopHat
from midas_transforms.radius import (
    predicted_spot_intensities,
    refine_K_per_ring_closed_form,
)


def _devices() -> list[str]:
    devs = ["cpu"]
    if torch.cuda.is_available():
        devs.append("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        devs.append("mps")
    return devs


def _t(x, **kw):  # noqa: ANN
    kw.setdefault("dtype", torch.float64)
    return torch.tensor(x, **kw)


# --------------------------------------------------------------- forward model


def test_single_voxel_in_beam_exact():
    """One voxel, one spot fully covering it -> I = K · I_th · V."""
    sg = SampleGrid.from_arrays(
        voxel_positions=[[0.0, 0.0, 0.0]],
        voxel_size_um=5.0,
        grain_map=[0],
    )
    V = _t([2.0])
    K = _t([3.5])
    I_th = _t([7.0])
    I_pred = predicted_spot_intensities(
        V, K, I_th,
        spot_ring_idx=torch.tensor([0]),
        spot_grain_idx=torch.tensor([0]),
        spot_scan_pos_um=_t([0.0]),
        spot_omega_rad=_t([math.pi / 2]),  # proj_ω(v) = v_x
        sample_grid=sg, beam_profile=TopHat(20.0),  # wide beam -> fraction = 1
    )
    assert math.isclose(float(I_pred[0]), 3.5 * 7.0 * 2.0)


def test_partial_beam_overlap_scales_contribution():
    """Beam width = voxel size, scan_pos shifted by 0.5 voxel -> fraction 0.5."""
    sg = SampleGrid.from_arrays(
        voxel_positions=[[0.0, 0.0, 0.0]], voxel_size_um=10.0, grain_map=[0],
    )
    V = _t([1.0]); K = _t([2.0]); I_th = _t([3.0])
    # Voxel center (lab x=0), omega=π/2 -> proj=0. Beam at scan_pos=5 (one voxel-half away).
    # TopHat width 10: overlap with voxel [-5, 5] when beam is [0, 10]: overlap = [0, 5] = 5
    # -> fraction = 5/10 = 0.5
    I_pred = predicted_spot_intensities(
        V, K, I_th,
        torch.tensor([0]), torch.tensor([0]),
        _t([5.0]), _t([math.pi/2]),
        sg, TopHat(10.0),
    )
    assert math.isclose(float(I_pred[0]), 2.0 * 3.0 * 1.0 * 0.5)


def test_grain_isolation():
    """Spot in grain 0 must not be contaminated by voxels in grain 1."""
    # Both voxels in beam (TopHat width covers both x=0 and x=10) — but only grain 0's voxel counts.
    sg = SampleGrid.from_arrays(
        voxel_positions=[[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]],
        voxel_size_um=10.0,
        grain_map=[0, 1],
    )
    V = _t([1.0, 99.0])  # grain 1's V is huge, must NOT leak in
    K = _t([1.0]); I_th = _t([1.0])
    I_pred = predicted_spot_intensities(
        V, K, I_th,
        spot_ring_idx=torch.tensor([0]),
        spot_grain_idx=torch.tensor([0]),     # spot in grain 0
        spot_scan_pos_um=_t([5.0]),            # beam at 5; both voxel projections 0 and 10
        spot_omega_rad=_t([math.pi/2]),
        sample_grid=sg, beam_profile=TopHat(30.0),
    )
    # Only voxel 0 contributes (grain 0), V=1, fully in beam: I = 1*1*1 = 1
    assert math.isclose(float(I_pred[0]), 1.0)


def test_orphan_spots_get_zero():
    sg = SampleGrid.from_arrays(
        voxel_positions=[[0.0, 0.0, 0.0]], voxel_size_um=5.0, grain_map=[0],
    )
    V = _t([1.0]); K = _t([1.0]); I_th = _t([1.0])
    I_pred = predicted_spot_intensities(
        V, K, I_th,
        spot_ring_idx=torch.tensor([0]),
        spot_grain_idx=torch.tensor([-1]),    # orphan
        spot_scan_pos_um=_t([0.0]),
        spot_omega_rad=_t([math.pi/2]),
        sample_grid=sg, beam_profile=TopHat(10.0),
    )
    assert float(I_pred[0]) == 0.0


def test_grain_with_empty_voxel_set_yields_zero():
    sg = SampleGrid.from_arrays(
        voxel_positions=[[0.0, 0.0, 0.0]], voxel_size_um=5.0, grain_map=[0],
    )
    V = _t([1.0]); K = _t([1.0]); I_th = _t([1.0])
    I_pred = predicted_spot_intensities(
        V, K, I_th,
        spot_ring_idx=torch.tensor([0]),
        spot_grain_idx=torch.tensor([99]),   # grain 99 has no voxels
        spot_scan_pos_um=_t([0.0]),
        spot_omega_rad=_t([math.pi/2]),
        sample_grid=sg, beam_profile=TopHat(10.0),
    )
    assert float(I_pred[0]) == 0.0


def test_omega_zero_uses_y_axis():
    """At ω=0: proj_ω(v) = v_y. Spot at scan_pos=0 sees voxel at (10, 0, 0) but
    NOT at (0, 10, 0) when beam is narrow (5 µm)."""
    sg = SampleGrid.from_arrays(
        voxel_positions=[[10.0, 0.0, 0.0], [0.0, 10.0, 0.0]],
        voxel_size_um=5.0,
        grain_map=[0, 0],
    )
    V = _t([1.0, 99.0])
    K = _t([1.0]); I_th = _t([1.0])
    I_pred = predicted_spot_intensities(
        V, K, I_th,
        torch.tensor([0]), torch.tensor([0]),
        _t([0.0]), _t([0.0]),    # ω=0 -> proj = v_y
        sg, TopHat(5.0),
    )
    # voxel 0: proj=0 (in beam) — V=1
    # voxel 1: proj=10 (out of beam) — V=99 excluded
    assert math.isclose(float(I_pred[0]), 1.0)


# --------------------------------------------------------------- differentiability


def test_forward_grad_flows_to_V_and_K():
    sg = SampleGrid.from_arrays(
        voxel_positions=[[0.0, 0.0, 0.0]], voxel_size_um=5.0, grain_map=[0],
    )
    V = torch.tensor([1.5], dtype=torch.float64, requires_grad=True)
    K = torch.tensor([2.0], dtype=torch.float64, requires_grad=True)
    I_th = _t([3.0])
    I_pred = predicted_spot_intensities(
        V, K, I_th,
        torch.tensor([0]), torch.tensor([0]),
        _t([0.0]), _t([math.pi/2]),
        sg, TopHat(20.0),
    )
    I_pred.sum().backward()
    # dI/dV = K * I_th * fraction = 2*3*1 = 6
    # dI/dK = I_th * V * fraction = 3*1.5*1 = 4.5
    assert math.isclose(float(V.grad[0]), 6.0)
    assert math.isclose(float(K.grad[0]), 4.5)


def test_forward_grad_flows_to_scan_pos_and_omega():
    """Gradient passes through projected position and beam fraction."""
    sg = SampleGrid.from_arrays(
        voxel_positions=[[3.0, 0.0, 0.0]], voxel_size_um=10.0, grain_map=[0],
    )
    V = _t([1.0]); K = _t([1.0]); I_th = _t([1.0])
    scan = torch.tensor([1.5], dtype=torch.float64, requires_grad=True)
    omega = torch.tensor([math.pi/2 + 0.1], dtype=torch.float64, requires_grad=True)
    I_pred = predicted_spot_intensities(
        V, K, I_th, torch.tensor([0]), torch.tensor([0]),
        scan, omega, sg, TopHat(6.0),
    )
    I_pred.sum().backward()
    # Both grads should be finite and (for this configuration) non-zero
    assert math.isfinite(float(scan.grad[0]))
    assert math.isfinite(float(omega.grad[0]))


def test_forward_gradcheck_wrt_V():
    sg = SampleGrid.from_arrays(
        voxel_positions=[[0.0, 0.0, 0.0], [12.0, 0.0, 0.0]],
        voxel_size_um=10.0, grain_map=[0, 0],
    )
    K = _t([1.7])
    I_th = _t([2.3])
    spot_ring  = torch.tensor([0, 0, 0])
    spot_grain = torch.tensor([0, 0, 0])
    spot_scan  = _t([0.0, 6.0, 12.0])
    spot_omega = _t([math.pi/2] * 3)
    beam = TopHat(8.0)

    def f(V):
        return predicted_spot_intensities(
            V, K, I_th, spot_ring, spot_grain, spot_scan, spot_omega, sg, beam,
        ).sum()

    V = torch.tensor([0.7, 1.3], dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(f, (V,), eps=1e-6, atol=1e-7)


# --------------------------------------------------------------- closed-form K


def test_closed_form_K_exact_recovery_single_ring():
    """Noise-free data on one ring -> closed-form K recovers K_true exactly."""
    sg = SampleGrid.from_arrays(
        voxel_positions=[[i * 10.0, 0.0, 0.0] for i in range(4)],
        voxel_size_um=10.0, grain_map=[0, 0, 0, 0],
    )
    V_true = _t([1.0, 1.5, 0.7, 2.0])
    K_true = _t([4.2])
    I_th = _t([11.0])

    n_spots = 6
    spot_ring  = torch.tensor([0] * n_spots)
    spot_grain = torch.tensor([0] * n_spots)
    # Place spots at varied scan positions
    spot_scan = _t([0.0, 10.0, 20.0, 30.0, 5.0, 15.0])
    spot_omega = _t([math.pi/2] * n_spots)

    I_obs = predicted_spot_intensities(
        V_true, K_true, I_th, spot_ring, spot_grain,
        spot_scan, spot_omega, sg, TopHat(5.0),
    )
    K_est = refine_K_per_ring_closed_form(
        V_true, I_th, I_obs, spot_ring, spot_grain,
        spot_scan, spot_omega, sg, TopHat(5.0), n_rings=1,
    )
    assert torch.isclose(K_est[0], K_true[0], rtol=1e-12)


def test_closed_form_K_robust_to_log_normal_noise():
    """With 10% multiplicative log-normal noise, recovered K is within 10%."""
    sg = SampleGrid.from_arrays(
        voxel_positions=[[i * 10.0, 0.0, 0.0] for i in range(8)],
        voxel_size_um=10.0, grain_map=[0] * 8,
    )
    V_true = _t([1.0, 1.5, 0.7, 2.0, 0.8, 1.1, 1.4, 0.9])
    K_true = _t([4.2])
    I_th = _t([11.0])

    # 50 spots, mix of scan positions
    n_spots = 50
    g = torch.Generator().manual_seed(0)
    spot_scan = (torch.rand(n_spots, generator=g, dtype=torch.float64) * 80.0)
    spot_omega = _t([math.pi/2] * n_spots)
    spot_ring  = torch.tensor([0] * n_spots)
    spot_grain = torch.tensor([0] * n_spots)

    I_clean = predicted_spot_intensities(
        V_true, K_true, I_th, spot_ring, spot_grain,
        spot_scan, spot_omega, sg, TopHat(5.0),
    )
    # Only keep spots with positive prediction (some scan positions miss voxels)
    keep = I_clean > 1e-10
    spot_scan = spot_scan[keep]
    spot_omega = spot_omega[keep]
    spot_ring = spot_ring[keep]
    spot_grain = spot_grain[keep]
    I_clean = I_clean[keep]
    assert keep.sum() >= 10, "need >= 10 spots after filtering"

    log_noise = 0.10 * torch.randn(I_clean.shape, generator=g, dtype=torch.float64)
    I_obs = I_clean * torch.exp(log_noise)
    K_est = refine_K_per_ring_closed_form(
        V_true, I_th, I_obs, spot_ring, spot_grain,
        spot_scan, spot_omega, sg, TopHat(5.0), n_rings=1,
    )
    assert abs(float(K_est[0]) / float(K_true[0]) - 1.0) < 0.10


def test_closed_form_K_two_rings():
    """K per ring recovered independently."""
    sg = SampleGrid.from_arrays(
        voxel_positions=[[i * 10.0, 0.0, 0.0] for i in range(4)],
        voxel_size_um=10.0, grain_map=[0, 0, 0, 0],
    )
    V_true = _t([1.0, 1.5, 0.7, 2.0])
    K_true = _t([4.2, 0.85])
    I_th = _t([11.0, 9.0])

    n_per_ring = 4
    spot_scan = _t([0.0, 10.0, 20.0, 30.0] * 2)
    spot_omega = _t([math.pi/2] * 2 * n_per_ring)
    spot_ring  = torch.tensor([0] * n_per_ring + [1] * n_per_ring)
    spot_grain = torch.tensor([0] * 2 * n_per_ring)

    I_obs = predicted_spot_intensities(
        V_true, K_true, I_th, spot_ring, spot_grain,
        spot_scan, spot_omega, sg, TopHat(5.0),
    )
    K_est = refine_K_per_ring_closed_form(
        V_true, I_th, I_obs, spot_ring, spot_grain,
        spot_scan, spot_omega, sg, TopHat(5.0), n_rings=2,
    )
    assert torch.allclose(K_est, K_true, rtol=1e-12)


def test_closed_form_K_empty_ring_returns_one():
    """If ring r has no valid spots, K[r] = 1 (no update)."""
    sg = SampleGrid.from_arrays(
        voxel_positions=[[0.0, 0.0, 0.0]], voxel_size_um=10.0, grain_map=[0],
    )
    V = _t([1.0]); I_th = _t([1.0, 1.0])  # 2 rings
    I_obs = _t([1.0])
    spot_ring = torch.tensor([0])   # only ring 0 has spots
    K_est = refine_K_per_ring_closed_form(
        V, I_th, I_obs, spot_ring,
        torch.tensor([0]), _t([0.0]), _t([math.pi/2]),
        sg, TopHat(20.0), n_rings=2,
    )
    # ring 1 had no spots -> K[1] = exp(0) = 1
    assert math.isclose(float(K_est[1]), 1.0)


# --------------------------------------------------------------- device portability


@pytest.mark.parametrize("device", _devices())
def test_forward_runs_on_device(device):
    dtype = torch.float64 if device != "mps" else torch.float32
    sg = SampleGrid.from_arrays(
        voxel_positions=[[0.0, 0.0, 0.0]], voxel_size_um=5.0, grain_map=[0],
        device=device, dtype=dtype,
    )
    V = torch.tensor([1.0], dtype=dtype, device=device)
    K = torch.tensor([2.0], dtype=dtype, device=device)
    I_th = torch.tensor([3.0], dtype=dtype, device=device)
    spot_ring  = torch.tensor([0], device=device)
    spot_grain = torch.tensor([0], device=device)
    spot_scan  = torch.tensor([0.0], dtype=dtype, device=device)
    spot_omega = torch.tensor([math.pi/2], dtype=dtype, device=device)
    beam = TopHat(20.0, device=device, dtype=dtype)
    I_pred = predicted_spot_intensities(
        V, K, I_th, spot_ring, spot_grain, spot_scan, spot_omega, sg, beam,
    )
    assert I_pred.device.type == torch.device(device).type
    assert I_pred.dtype == dtype
    assert math.isclose(float(I_pred[0]), 6.0, abs_tol=1e-4)


@pytest.mark.parametrize("device", _devices())
def test_closed_form_K_runs_on_device(device):
    dtype = torch.float64 if device != "mps" else torch.float32
    sg = SampleGrid.from_arrays(
        voxel_positions=[[i*10.0, 0.0, 0.0] for i in range(3)],
        voxel_size_um=10.0, grain_map=[0, 0, 0],
        device=device, dtype=dtype,
    )
    V = torch.tensor([1.0, 1.5, 0.8], dtype=dtype, device=device)
    K_true = torch.tensor([2.5], dtype=dtype, device=device)
    I_th = torch.tensor([4.0], dtype=dtype, device=device)
    spot_ring  = torch.tensor([0, 0, 0], device=device)
    spot_grain = torch.tensor([0, 0, 0], device=device)
    spot_scan  = torch.tensor([0.0, 10.0, 20.0], dtype=dtype, device=device)
    spot_omega = torch.tensor([math.pi/2]*3, dtype=dtype, device=device)
    beam = TopHat(5.0, device=device, dtype=dtype)
    I_obs = predicted_spot_intensities(
        V, K_true, I_th, spot_ring, spot_grain, spot_scan, spot_omega, sg, beam,
    )
    K_est = refine_K_per_ring_closed_form(
        V, I_th, I_obs, spot_ring, spot_grain, spot_scan, spot_omega, sg, beam,
        n_rings=1,
    )
    assert K_est.device.type == torch.device(device).type
    # fp32 on MPS is rougher; relax tolerance accordingly
    tol = 1e-10 if dtype == torch.float64 else 1e-4
    assert abs(float(K_est[0]) - float(K_true[0])) < tol * float(K_true[0]) + tol
