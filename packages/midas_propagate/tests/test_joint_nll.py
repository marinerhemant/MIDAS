"""Tests for midas_propagate.joint_nll.

The single end-to-end test (``test_per_grain_hessian_blocks_synthetic``)
forward-simulates one FCC Au grain on a paper-1-shaped FF geometry, treats
those predicted spots as the observations, then verifies:

  * H_gg, H_gc have the right shapes
  * H_gg is positive-(semi)definite — quadratic form at MAP residual=0
  * H_gc is non-trivial in the columns that physically couple to grain
    state (Lsd, ty, tz at minimum)
  * Fisher and full-Hessian paths agree on H_gg in the residual=0 regime
    (where Gauss-Newton == Newton)

Skipped automatically if midas_diffract / midas_hkls are unavailable.
"""
from __future__ import annotations

import math

import pytest
import torch


def _have_diffract():
    try:
        import midas_diffract  # noqa: F401
        return True
    except ImportError:
        return False


def _have_hkls():
    try:
        import midas_hkls  # noqa: F401
        return True
    except ImportError:
        return False


@pytest.mark.skipif(not (_have_diffract() and _have_hkls()),
                    reason="midas_diffract / midas_hkls not installed")
def test_per_grain_hessian_blocks_synthetic():
    """Synthetic FCC Au grain on paper-1-shaped FF geometry. Verify the
    Hessian blocks have the right shapes, H_gg is PSD, and Fisher == full
    Hessian when the residual is exactly zero (at GT)."""
    from midas_diffract import HEDMForwardModel, HEDMGeometry, hkls_for_forward_model
    from midas_hkls import SpaceGroup, Lattice
    from midas_propagate.joint_nll import (
        GrainObs, per_grain_hessian_blocks, make_per_grain_residual,
    )

    DEG2RAD = math.pi / 180.0

    # apply_tilts=True: forward model uses ty/tz directly (instead of
    # assuming the pre-correction at peak-finding has already absorbed
    # them). Required for paper-1's coupling of calibration tilt
    # uncertainty into per-grain residuals.
    geom = HEDMGeometry(
        Lsd=1_000_000.0, y_BC=1024.0, z_BC=1024.0, px=200.0,
        omega_start=0.0, omega_step=0.25, n_frames=1440,
        n_pixels_y=2048, n_pixels_z=2048,
        min_eta=6.0, wavelength=0.172979,
        apply_tilts=True,
    )
    sg = SpaceGroup.from_number(225)
    lat = Lattice.for_system("cubic", a=4.08)
    hkls_cart, thetas, hkls_int = hkls_for_forward_model(
        sg, lat, wavelength_A=geom.wavelength, two_theta_max_deg=15.0,
    )
    model = HEDMForwardModel(
        hkls=hkls_cart, thetas=thetas, geometry=geom, hkls_int=hkls_int,
    )

    gt_euler = torch.tensor([45.0, 30.0, 60.0], dtype=torch.float64) * DEG2RAD
    gt_latc = torch.tensor([4.08, 4.08, 4.08, 90.0, 90.0, 90.0], dtype=torch.float64)
    gt_pos = torch.zeros(3, dtype=torch.float64)

    spots = model(gt_euler.unsqueeze(0), gt_pos.unsqueeze(0),
                   lattice_params=gt_latc)
    det, valid = HEDMForwardModel.predict_spot_coords(spots, space="detector")
    obs = det.squeeze()[valid.squeeze() > 0.5].detach().clone()
    if obs.shape[0] < 8:
        pytest.skip(f"Not enough valid spots ({obs.shape[0]}) for the test")

    # GrainObs at GT (so residual is zero — Fisher == Hessian regime).
    grain_obs = GrainObs(
        spot_id=0,
        euler_rad=gt_euler,
        latc=gt_latc,
        pos_um=gt_pos,
        observed_detector=obs,
    )

    calibration_names = ["Lsd", "BC_y", "BC_z", "ty", "tz"]
    calibration_map = torch.tensor(
        [1_000_000.0, 1024.0, 1024.0, 0.0, 0.0], dtype=torch.float64,
    )
    sigma_obs_detector = torch.full((3,), 0.5, dtype=torch.float64)  # px / frame

    res_f = per_grain_hessian_blocks(
        grain_obs,
        hkls_cart=hkls_cart, hkls_int=hkls_int, thetas=thetas,
        base_geometry=geom, scan_config=None,
        calibration_names=calibration_names,
        calibration_map=calibration_map,
        sigma_obs_detector=sigma_obs_detector,
        method="fisher",
    )

    n_g, n_c = 12, len(calibration_names)
    assert res_f.H_gg.shape == (n_g, n_g)
    assert res_f.H_gc.shape == (n_g, n_c)
    assert res_f.n_spots_matched > 0
    assert torch.isfinite(res_f.H_gg).all()
    assert torch.isfinite(res_f.H_gc).all()

    # H_gg via Fisher is J^T J — PSD by construction.
    eigvals_gg = torch.linalg.eigvalsh(0.5 * (res_f.H_gg + res_f.H_gg.T))
    assert eigvals_gg.min() >= -1e-9, (
        f"H_gg must be PSD; smallest eigval {eigvals_gg.min().item():.3e}"
    )

    # H_gc must be non-trivial — Lsd, ty, tz columns should carry the most
    # signal (these are the calibration params that move spot positions
    # in 2θ and η). Use column L2 norm as a coarse signal check.
    col_norms = torch.linalg.norm(res_f.H_gc, dim=0)
    name_to_col = {n: i for i, n in enumerate(calibration_names)}
    for must_be_nonzero in ("Lsd", "ty", "tz"):
        idx = name_to_col[must_be_nonzero]
        assert col_norms[idx] > 1e-6, (
            f"H_gc column for {must_be_nonzero!r} is suspiciously small: "
            f"||H_gc[:, {idx}]|| = {col_norms[idx].item():.3e}"
        )

    # Compare Fisher vs full Hessian — at GT, residual ~ 0 so they should
    # agree on H_gg up to numerical noise.
    res_h = per_grain_hessian_blocks(
        grain_obs,
        hkls_cart=hkls_cart, hkls_int=hkls_int, thetas=thetas,
        base_geometry=geom, scan_config=None,
        calibration_names=calibration_names,
        calibration_map=calibration_map,
        sigma_obs_detector=sigma_obs_detector,
        method="hessian",
    )
    # Both H_gg estimates should be ~equal at zero-residual MAP (within
    # numerical precision — fp64 with ~50 spots, 12 params).
    rel_err = (res_f.H_gg - res_h.H_gg).abs().max() / res_f.H_gg.abs().max().clamp(min=1e-30)
    assert rel_err < 1e-6, (
        f"Fisher and Hessian H_gg disagree at zero-residual MAP; "
        f"max relative error {rel_err.item():.3e}"
    )


@pytest.mark.skipif(not (_have_diffract() and _have_hkls()),
                    reason="midas_diffract / midas_hkls not installed")
def test_per_grain_residual_is_zero_at_map():
    """Sanity: the residual built by ``make_per_grain_residual`` evaluates
    to zero at the MAP state when observations come from forward-simulating
    that exact state (no spot mis-association above the threshold)."""
    from midas_diffract import HEDMForwardModel, HEDMGeometry, hkls_for_forward_model
    from midas_hkls import SpaceGroup, Lattice
    from midas_propagate.joint_nll import GrainObs, make_per_grain_residual

    DEG2RAD = math.pi / 180.0
    # apply_tilts=True: forward model uses ty/tz directly (instead of
    # assuming the pre-correction at peak-finding has already absorbed
    # them). Required for paper-1's coupling of calibration tilt
    # uncertainty into per-grain residuals.
    geom = HEDMGeometry(
        Lsd=1_000_000.0, y_BC=1024.0, z_BC=1024.0, px=200.0,
        omega_start=0.0, omega_step=0.25, n_frames=1440,
        n_pixels_y=2048, n_pixels_z=2048,
        min_eta=6.0, wavelength=0.172979,
        apply_tilts=True,
    )
    sg = SpaceGroup.from_number(225)
    lat = Lattice.for_system("cubic", a=4.08)
    hkls_cart, thetas, hkls_int = hkls_for_forward_model(
        sg, lat, wavelength_A=geom.wavelength, two_theta_max_deg=15.0,
    )
    model = HEDMForwardModel(
        hkls=hkls_cart, thetas=thetas, geometry=geom, hkls_int=hkls_int,
    )
    gt_euler = torch.tensor([45.0, 30.0, 60.0], dtype=torch.float64) * DEG2RAD
    gt_latc = torch.tensor([4.08, 4.08, 4.08, 90.0, 90.0, 90.0], dtype=torch.float64)
    gt_pos = torch.zeros(3, dtype=torch.float64)

    spots = model(gt_euler.unsqueeze(0), gt_pos.unsqueeze(0),
                   lattice_params=gt_latc)
    det, valid = HEDMForwardModel.predict_spot_coords(spots, space="detector")
    obs = det.squeeze()[valid.squeeze() > 0.5].detach().clone()
    if obs.shape[0] < 8:
        pytest.skip("not enough valid spots")

    grain_obs = GrainObs(
        spot_id=0, euler_rad=gt_euler, latc=gt_latc, pos_um=gt_pos,
        observed_detector=obs,
    )
    calibration_names = ["Lsd", "BC_y", "BC_z", "ty", "tz"]
    calibration_map = torch.tensor(
        [1_000_000.0, 1024.0, 1024.0, 0.0, 0.0], dtype=torch.float64,
    )
    sigma_obs_detector = torch.full((3,), 0.5, dtype=torch.float64)

    r_fn, n_matched = make_per_grain_residual(
        grain_obs,
        hkls_cart=hkls_cart, hkls_int=hkls_int, thetas=thetas,
        base_geometry=geom, scan_config=None,
        calibration_names=calibration_names,
        calibration_map=calibration_map,
        sigma_obs_detector=sigma_obs_detector,
    )
    g_map = torch.cat([gt_euler, gt_latc, gt_pos])
    r_at_map = r_fn(g_map, calibration_map)
    assert r_at_map.numel() == 3 * n_matched
    # Numerically zero (fp64; might be ~1e-10 due to model nonlinearity
    # and projection round-trip).
    assert r_at_map.abs().max() < 1e-6, (
        f"residual at GT should be ~0; max |r| = {r_at_map.abs().max().item():.3e}"
    )
