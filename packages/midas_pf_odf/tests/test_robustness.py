"""Robustness tests: noise sweep, FREE identifiability mode, L-BFGS path.

Same plant as ``test_round_trip_small`` (4x4 single-grain, ε_11
gradient amp 2e-3) — varies one knob at a time and asserts graceful
behavior.
"""

import math
import torch
import pytest

from midas_pf_odf import (
    plant_single_grain, simulate_grain_patches, fit_grain_peakshape,
    IdentifiabilityMode, recovery_metrics,
)

from tests.conftest import (
    make_fcc_hkls, small_scan_config, build_model,
)


@pytest.fixture(scope="module")
def model_fixed():
    G, th, hkls_int = make_fcc_hkls(d_min_A=1.0, h_max=2)
    sc = small_scan_config(sample_size_um=10.0, n_scans=7, beam_size_um=4.0)
    return build_model(sc, hkls_int, G, th)


def _plant_and_data(model, add_noise_sigma=0.0, eps_amp=2e-3, seed=0):
    plant = plant_single_grain(
        grid_shape=(4, 4), voxel_size_um=2.0,
        eps_avg=(0.0,) * 6,
        eps_gradient_voigt=0,
        eps_gradient_amp=eps_amp,
        eps_gradient_dir="x",
    )
    data = simulate_grain_patches(
        plant, model,
        patch_F=5, patch_P=15, sigma_yz=1.0, sigma_f=0.6,
        gate_tau_um=0.5, add_noise_sigma=add_noise_sigma, seed=seed,
    )
    return plant, data


@pytest.mark.parametrize("noise_sigma,eps_tol", [
    (0.0,   1.0e-3),
    (0.01,  1.5e-3),
    (0.05,  3.0e-3),
])
def test_noise_sweep(model_fixed, noise_sigma, eps_tol):
    """Recovery should degrade gracefully with patch-noise σ."""
    plant, data = _plant_and_data(model_fixed, add_noise_sigma=noise_sigma)
    fit = fit_grain_peakshape(
        data, model_fixed,
        voxel_pos=plant.voxel_pos,
        R_init=plant.R_voxel,
        eps_init=torch.zeros_like(plant.eps_voxel),
        lattice_init=plant.lattice,
        identifiability=IdentifiabilityMode.PROJECT_EPS_MEAN_ZERO,
        optimizer="adam",
        inner_steps=200,
        lr_aa=1e-4, lr_eps=1e-3, lr_lat=1e-5,
    )
    rep = recovery_metrics(plant, fit.R_fit, fit.eps_fit)
    assert rep.eps_rms < eps_tol, (
        f"σ={noise_sigma}: ε RMSE {rep.eps_rms:.3e} > tol {eps_tol:.3e}"
    )


def test_free_identifiability_mode(model_fixed):
    """FREE mode (no mean-zero projection) should also recover the
    gradient — when the planted ε mean is zero, both modes are
    equivalent up to optimizer drift."""
    plant, data = _plant_and_data(model_fixed)
    fit = fit_grain_peakshape(
        data, model_fixed,
        voxel_pos=plant.voxel_pos,
        R_init=plant.R_voxel,
        eps_init=torch.zeros_like(plant.eps_voxel),
        lattice_init=plant.lattice,
        identifiability=IdentifiabilityMode.FREE,
        optimizer="adam",
        inner_steps=200,
        lr_aa=1e-4, lr_eps=1e-3, lr_lat=1e-5,
    )
    rep = recovery_metrics(plant, fit.R_fit, fit.eps_fit)
    assert rep.eps_rms < 2e-3
    # 0.06° gate: FREE-mode Adam @ 200 steps lands ~0.03° RMS, but a single
    # outlier voxel (the documented ~2-3% worse-recovery voxels) pushes the RMS
    # to ~0.053° on some BLAS/platforms (Linux CPU vs macOS). The physically
    # meaningful ε recovery is unaffected; keep a loose regression gate.
    assert rep.misorient_rms_deg < 0.06


def test_lbfgs_converges_faster(model_fixed):
    """L-BFGS should hit comparable recovery in far fewer outer steps."""
    plant, data = _plant_and_data(model_fixed)
    fit = fit_grain_peakshape(
        data, model_fixed,
        voxel_pos=plant.voxel_pos,
        R_init=plant.R_voxel,
        eps_init=torch.zeros_like(plant.eps_voxel),
        lattice_init=plant.lattice,
        identifiability=IdentifiabilityMode.PROJECT_EPS_MEAN_ZERO,
        optimizer="lbfgs",
        inner_steps=10,                       # 10 outer × strong-Wolfe inner
        lr_aa=1.0, lr_eps=1.0, lr_lat=1.0,    # L-BFGS line search adapts
    )
    rep = recovery_metrics(plant, fit.R_fit, fit.eps_fit)
    assert rep.eps_rms < 1e-3, (
        f"L-BFGS @ 10 steps ε RMSE = {rep.eps_rms:.3e}"
    )
