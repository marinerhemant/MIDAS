"""Tests for `midas_defect.rod_detect` (P1)."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from midas_defect.lattice import cual2_crystal
from midas_defect.rod_detect import (
    QRod,
    find_rods,
    find_rods_iterative_residual,
    refine_rod,
    soft_tube_score,
    tube_score,
    _nearest_low_hkl_to_direction,
)


CPU = torch.device("cpu")


def _plant_rod(rng: np.random.Generator, n_pts: int,
               t_range: tuple, direction: np.ndarray,
               pivot: np.ndarray, noise: float,
               intensity_range: tuple = (50, 200)
               ) -> tuple[np.ndarray, np.ndarray]:
    d = direction / np.linalg.norm(direction)
    ts = np.linspace(t_range[0], t_range[1], n_pts)
    pts = pivot[None, :] + ts[:, None] * d[None, :]
    pts += rng.normal(scale=noise, size=pts.shape)
    ints = rng.uniform(*intensity_range, size=n_pts)
    return pts, ints


# ---------------------------------------------------------------------------
# 1. Synthetic correctness
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_tube_score_inliers_match_perp_threshold():
    """`tube_score` equals Σ I over points within `r_tube` of the line."""
    rng = np.random.default_rng(0)
    pts, ints = _plant_rod(rng, n_pts=50, t_range=(-1, 1),
                           direction=np.array([1.0, 0, 0]),
                           pivot=np.array([0, 0, 0]),
                           noise=0.005, intensity_range=(100, 100))
    # add 200 random points well off-line
    noise_pts = rng.uniform(-2, 2, size=(200, 3))
    noise_pts[:, 1] += 1.0   # push them off the x-axis
    noise_ints = np.full(200, 100.0)
    pts_all = np.vstack([pts, noise_pts])
    ints_all = np.concatenate([ints, noise_ints])

    q = torch.tensor(pts_all, dtype=torch.float64)
    I = torch.tensor(ints_all, dtype=torch.float64)
    piv = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float64)
    dir_ = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)
    S = tube_score(q, I, piv, dir_, r_tube=0.02)
    # Only the planted rod points should be inliers
    assert S.item() == pytest.approx(50 * 100.0, rel=1e-12)


@pytest.mark.unit
def test_find_rods_recovers_3_synthetic_rods(synthetic_rod_cube):
    """Three planted rods along [100], [010], [111] are all recovered."""
    s = synthetic_rod_cube
    rods = find_rods(
        s["qx"], s["qy"], s["qz"], s["intensity"],
        n_cores=200, core_min_separation=0.02,
        pair_min_dist=0.1, pair_max_dist=2.0,
        r_tube=0.03, L_min=0.3, N_min_inliers=20,
        max_voxels_for_scoring=5000,
    )
    assert len(rods) >= 3, f"only found {len(rods)} rods"

    # check direction recovery for each planted rod
    truth_dirs = s["rod_dirs"]
    recovered_dirs = np.stack([r.direction for r in rods[:6]])
    for truth in truth_dirs:
        cos_arr = np.abs(recovered_dirs @ truth)
        assert cos_arr.max() > math.cos(math.radians(3.0)), (
            f"no recovered rod within 3° of truth {truth}; "
            f"max cos = {cos_arr.max():.4f}"
        )


@pytest.mark.unit
def test_find_rods_handles_no_rods_gracefully():
    """With only diffuse noise and no real rods, returns an empty list."""
    rng = np.random.default_rng(1)
    qx = rng.uniform(-1, 1, 500)
    qy = rng.uniform(-1, 1, 500)
    qz = rng.uniform(-1, 1, 500)
    inten = rng.uniform(50, 200, 500)
    rods = find_rods(
        qx, qy, qz, inten,
        n_cores=50, r_tube=0.01, L_min=2.0,    # stringent L_min — no random
        N_min_inliers=100, pair_min_dist=0.2,  # constellation will satisfy it
    )
    assert rods == [] or all(r.length < 2.0 for r in rods)


@pytest.mark.unit
def test_shells_crossed_for_synthetic_rod():
    """A rod that passes through known shell radii reports those shells."""
    rng = np.random.default_rng(2)
    # rod along c-axis (z), pivoted at q_x = 2.07 ≈ |q| of (200) shell
    direction = np.array([0.0, 0.0, 1.0])
    pivot = np.array([2.07, 0.0, 0.0])
    # rod covers t in [-5, 5], so it crosses many shells at varying |q|
    pts, ints = _plant_rod(rng, n_pts=200, t_range=(-5.0, 5.0),
                           direction=direction, pivot=pivot, noise=0.01,
                           intensity_range=(80, 150))
    cr = cual2_crystal()
    rods = find_rods(
        pts[:, 0], pts[:, 1], pts[:, 2], ints,
        n_cores=80, core_min_separation=0.1,
        pair_min_dist=0.5, pair_max_dist=12.0,
        r_tube=0.05, L_min=2.0, N_min_inliers=20,
        crystal=cr,
    )
    assert len(rods) >= 1
    r = rods[0]
    # The rod must cross multiple shells (it spans |q| from 2.07 up to ~5.4)
    assert len(r.shells_crossed) >= 3, (
        f"expected ≥3 shells crossed; got {len(r.shells_crossed)}"
    )


@pytest.mark.unit
def test_nearest_low_hkl_to_direction():
    """Closest crystal-frame (h,k,l) for representative directions."""
    assert _nearest_low_hkl_to_direction(np.array([1.0, 0, 0])) == (1, 0, 0)
    assert _nearest_low_hkl_to_direction(np.array([0, 0, 1.0])) == (0, 0, 1)
    # close to [110] but slightly tilted
    h = _nearest_low_hkl_to_direction(np.array([1.0, 1.01, 0.02]))
    assert h == (1, 1, 0)
    # nearly [111]
    h = _nearest_low_hkl_to_direction(np.array([1.02, 0.98, 1.01]))
    assert h == (1, 1, 1)


# ---------------------------------------------------------------------------
# 2. Autograd correctness
# ---------------------------------------------------------------------------

@pytest.mark.autograd
def test_soft_tube_score_gradient_wrt_direction_and_pivot():
    """Gradient of `-soft_tube_score` w.r.t. (pivot, direction) matches FD."""
    rng = np.random.default_rng(0)
    pts, ints = _plant_rod(rng, n_pts=80, t_range=(-1, 1),
                           direction=np.array([1.0, 0, 0]),
                           pivot=np.zeros(3), noise=0.01)
    q = torch.tensor(pts, dtype=torch.float64)
    I = torch.tensor(ints, dtype=torch.float64)
    # perturb a bit so we have a nonzero gradient
    pivot = torch.tensor([0.05, -0.05, 0.02], dtype=torch.float64, requires_grad=True)
    direction = torch.tensor([1.0, 0.1, 0.0], dtype=torch.float64, requires_grad=True)

    def loss(p_, d_):
        return -soft_tube_score(q, I, p_, d_, r_tube=0.05, sharpness=5.0)

    L = loss(pivot, direction)
    g_p, g_d = torch.autograd.grad(L, (pivot, direction))

    eps = 1e-5
    for k in range(3):
        for grad_auto, var, name in [(g_p, pivot, "pivot"), (g_d, direction, "direction")]:
            plus  = var.detach().clone()
            minus = var.detach().clone()
            plus[k]  += eps
            minus[k] -= eps
            if name == "pivot":
                Lp = loss(plus, direction.detach())
                Lm = loss(minus, direction.detach())
            else:
                Lp = loss(pivot.detach(), plus)
                Lm = loss(pivot.detach(), minus)
            g_fd = (Lp - Lm) / (2 * eps)
            assert grad_auto[k].item() == pytest.approx(
                g_fd.item(), rel=1e-3, abs=1e-5
            ), f"{name}[{k}]: auto={grad_auto[k]:.6e} fd={g_fd:.6e}"


@pytest.mark.autograd
def test_refine_rod_improves_score():
    """Refinement strictly improves (or holds) the soft-tube score."""
    rng = np.random.default_rng(0)
    pts, ints = _plant_rod(rng, n_pts=200, t_range=(-1, 1),
                           direction=np.array([1.0, 0.05, 0.02]),
                           pivot=np.array([0.0, 0.01, 0.0]),
                           noise=0.005)
    q = torch.tensor(pts, dtype=torch.float64)
    I = torch.tensor(ints, dtype=torch.float64)
    # initialize ~3° off
    piv0 = torch.tensor([0.02, 0.02, -0.01], dtype=torch.float64)
    dir0 = torch.tensor([1.0, 0.0, 0.07], dtype=torch.float64)
    dir0 = dir0 / torch.linalg.vector_norm(dir0)
    S0 = soft_tube_score(q, I, piv0, dir0, r_tube=0.03).item()
    piv_r, dir_r, S_r = refine_rod(q, I, piv0, dir0,
                                    r_tube=0.03, n_steps=120, lr=5e-3)
    assert S_r >= S0 - 1e-6, f"refinement did not improve score: {S0} -> {S_r}"


# ---------------------------------------------------------------------------
# 3. Device portability
# ---------------------------------------------------------------------------

@pytest.mark.device
def test_find_rods_device_portable(_device_param):
    """find_rods produces the same dominant rod direction on CPU vs MPS."""
    rng = np.random.default_rng(3)
    pts, ints = _plant_rod(rng, n_pts=300, t_range=(-1, 1),
                           direction=np.array([1.0, 0.5, 0.2]),
                           pivot=np.zeros(3), noise=0.005)
    # add a constellation of Bragg-like clusters
    for _ in range(20):
        c = rng.uniform(-0.5, 0.5, size=3)
        cluster = c[None, :] + rng.normal(scale=0.01, size=(30, 3))
        ci = rng.uniform(100, 300, size=30)
        pts = np.vstack([pts, cluster])
        ints = np.concatenate([ints, ci])

    dtype = torch.float32 if _device_param.type == "mps" else torch.float64

    rods_cpu = find_rods(pts[:, 0], pts[:, 1], pts[:, 2], ints,
                         n_cores=80, core_min_separation=0.05,
                         pair_min_dist=0.3, r_tube=0.03,
                         L_min=0.4, N_min_inliers=20,
                         refine_steps=30, max_voxels_for_scoring=3000,
                         device=CPU, dtype=dtype)
    rods_dev = find_rods(pts[:, 0], pts[:, 1], pts[:, 2], ints,
                         n_cores=80, core_min_separation=0.05,
                         pair_min_dist=0.3, r_tube=0.03,
                         L_min=0.4, N_min_inliers=20,
                         refine_steps=30, max_voxels_for_scoring=3000,
                         device=_device_param, dtype=dtype)

    assert rods_cpu and rods_dev, "no rods detected on one of the devices"
    # The top rod direction should agree across devices to within 5°
    cos_top = abs(float(rods_cpu[0].direction @ rods_dev[0].direction))
    assert cos_top > math.cos(math.radians(5.0)), (
        f"top rod direction disagrees across devices "
        f"(cos = {cos_top:.4f}, expected > {math.cos(math.radians(5.0)):.4f})"
    )


# ---------------------------------------------------------------------------
# 4. Integration with seed_index
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_iterative_residual_finds_two_rods(synthetic_rod_cube):
    """Iterative-residual mode should peel off rod #1, then find rod #2."""
    s = synthetic_rod_cube
    rounds = find_rods_iterative_residual(
        s["qx"], s["qy"], s["qz"], s["intensity"],
        n_iter=3,
        suppress_perp=0.06, suppress_along_pad=0.5, suppress_floor=0.05,
        n_cores=200, core_min_separation=0.02,
        pair_min_dist=0.1, pair_max_dist=2.0,
        r_tube=0.03, L_min=0.3, N_min_inliers=20,
        max_voxels_for_scoring=5000,
    )
    assert len(rounds) >= 2, f"only got {len(rounds)} iterations"
    # Each iteration's top rod should have a different direction
    top_dirs = [r[0].direction for r in rounds]
    for i in range(len(top_dirs)):
        for j in range(i + 1, len(top_dirs)):
            cos_ij = abs(float(top_dirs[i] @ top_dirs[j]))
            angle_ij = math.degrees(math.acos(min(1.0, cos_ij)))
            assert angle_ij > 5.0, (
                f"iter {i+1} and {j+1} got the same rod (Δ={angle_ij:.2f}°)"
            )


@pytest.mark.integration
def test_rod_direction_in_crystal_frame_via_U():
    """Plant a rod along a known crystal axis; verify defect_normal_hkl recovery."""
    rng = np.random.default_rng(4)
    # rotate the c-axis [001]_crystal into some lab direction
    U_true = np.array([
        [ 0.6, -0.8,  0.0],
        [ 0.8,  0.6,  0.0],
        [ 0.0,  0.0,  1.0],
    ])
    # rod direction in lab = U @ [001]_crystal = U[:, 2] = [0, 0, 1]
    rod_lab_dir = U_true @ np.array([0.0, 0.0, 1.0])

    pts, ints = _plant_rod(rng, n_pts=200, t_range=(-2, 2),
                           direction=rod_lab_dir,
                           pivot=np.array([0.5, 0.0, 0.0]), noise=0.005,
                           intensity_range=(100, 200))
    # add some Bragg clusters
    for _ in range(20):
        c = rng.uniform(-1.0, 1.0, size=3)
        cluster = c[None, :] + rng.normal(scale=0.01, size=(30, 3))
        ci = rng.uniform(200, 400, size=30)
        pts = np.vstack([pts, cluster])
        ints = np.concatenate([ints, ci])

    rods = find_rods(pts[:, 0], pts[:, 1], pts[:, 2], ints,
                     n_cores=80, core_min_separation=0.05,
                     pair_min_dist=0.5, r_tube=0.03,
                     L_min=1.0, N_min_inliers=30,
                     U=U_true)

    assert len(rods) >= 1
    # The top rod's defect_normal_hkl should be (0, 0, ±1)
    hkl = rods[0].defect_normal_hkl
    assert hkl in {(0, 0, 1), (0, 0, -1)}, (
        f"expected (0,0,±1) for c-axis rod; got {hkl}"
    )
